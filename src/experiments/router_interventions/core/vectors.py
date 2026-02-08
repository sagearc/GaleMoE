"""Load expert SVD vectors and apply vector interventions (project-out, inject, subtract)."""
from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, Iterator

import torch

logger = logging.getLogger(__name__)


class ExpertVectors:
    """Per-expert SVD vectors loaded from pickle files.

    Cache convention: each file is a raw tensor V of shape (dim, dim) with
    columns = right singular vectors, or a dict with key "Vh" holding a matrix
    with rows = vectors.  The loader normalises both to [top_k, dim].
    """

    def __init__(
        self,
        svd_dir: str,
        layer_idx: int,
        num_experts: int = 8,
        model_tag: str = "mistralai_Mixtral_8x7B_v0.1",
        top_k: int = 1,
    ) -> None:
        self.svd_dir = svd_dir
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.model_tag = model_tag
        self.top_k = top_k
        self._vectors: Dict[int, torch.Tensor] = {}

    # -- I/O ----------------------------------------------------------------

    def load(self) -> None:
        """Load all expert vectors from svd_dir."""
        self._vectors.clear()
        logger.info("Loading SVD vectors from %s (layer %d, top_k=%d)",
                     os.path.abspath(self.svd_dir), self.layer_idx, self.top_k)
        for i in range(self.num_experts):
            path = self._path_for(i)
            self._vectors[i] = self._load_single(path)

    def _path_for(self, expert_idx: int) -> str:
        return os.path.join(
            self.svd_dir,
            f"{self.model_tag}_layer{self.layer_idx}_expert{expert_idx}.pkl",
        )

    def _load_single(self, filepath: str) -> torch.Tensor:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"SVD file not found: {filepath}")
        with open(filepath, "rb") as f:
            obj = pickle.load(f)

        # Extract tensor from various formats
        t = self._extract_tensor(obj, filepath)
        t = torch.as_tensor(t, dtype=torch.float32, device="cpu")

        if t.ndim == 1:
            return t / (t.norm() + 1e-12)

        # 2-D: columns = vectors for square raw tensors; rows = vectors otherwise
        t = self._select_top_k(t)

        # Normalize each vector
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-12)

        return t[0] if t.shape[0] == 1 else t  # [dim] when top_k=1

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _extract_tensor(obj, filepath: str):
        """Pull out the tensor/array from the pickle object."""
        if isinstance(obj, dict):
            for key in ("Vh", "vh", "V", "v"):
                if key in obj:
                    return obj[key]
        if isinstance(obj, (tuple, list)) and len(obj) >= 3:
            return obj[2]
        if isinstance(obj, (torch.Tensor,)):
            return obj
        # numpy arrays are also acceptable (torch.as_tensor handles them)
        try:
            return torch.as_tensor(obj)
        except Exception:
            pass
        raise ValueError(f"Could not extract vector from {filepath}")

    def _select_top_k(self, t: torch.Tensor) -> torch.Tensor:
        """From a 2-D matrix, return the first top_k vectors as [k, dim].

        Square raw cache: columns = vectors → take first k columns, transpose.
        Otherwise: rows = vectors → take first k rows.
        """
        if t.shape[0] == t.shape[1] or t.shape[0] > t.shape[1]:
            # columns = vectors
            k = min(self.top_k, t.shape[1])
            return t[:, :k].T
        # rows = vectors
        k = min(self.top_k, t.shape[0])
        return t[:k]

    # -- access --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._vectors)

    def __getitem__(self, expert_idx: int) -> torch.Tensor:
        return self._vectors[expert_idx]

    def get_single(self, expert_idx: int) -> torch.Tensor:
        """First (top) vector for an expert, always 1-D [dim]."""
        v = self._vectors[expert_idx]
        return v[0] if v.ndim == 2 else v

    def keys(self) -> Iterator[int]:
        return iter(self._vectors)

    def items(self) -> Iterator[tuple[int, torch.Tensor]]:
        return iter(self._vectors.items())


# ---------------------------------------------------------------------------
# Vector interventions
# ---------------------------------------------------------------------------

def _to_float(x: torch.Tensor) -> torch.Tensor:
    return x.float() if not x.is_floating_point() else x


class VectorIntervention:
    """Stateless vector operations for router interventions."""

    # -- direction generators ------------------------------------------------

    @staticmethod
    def make_random(dim: int, seed: int) -> torch.Tensor:
        """Random unit vector in R^dim."""
        g = torch.Generator().manual_seed(seed)
        r = torch.randn(dim, generator=g)
        return r / (r.norm() + 1e-12)

    @staticmethod
    def make_random_in_span(vectors: torch.Tensor, seed: int) -> torch.Tensor:
        """Random unit vector in the span of *vectors* (same SVD rank).

        vectors: [k, dim] (rows = basis) or [dim] (single direction).
        Returns a random unit vector in span(vectors), so the comparison with
        SVD is fair: we remove a random direction in the same subspace.
        """
        g = torch.Generator().manual_seed(seed)
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)
        k, dim = vectors.shape
        coeffs = torch.randn(k, generator=g, dtype=vectors.dtype, device=vectors.device)
        r = (coeffs.unsqueeze(0) @ vectors).squeeze(0)
        return r / (r.norm() + 1e-12)

    @staticmethod
    def make_orthogonal(v: torch.Tensor, seed: int) -> torch.Tensor:
        """Random unit vector orthogonal to *v* (Gram–Schmidt)."""
        g = torch.Generator().manual_seed(seed)
        r = torch.randn(v.shape, generator=g, dtype=v.dtype, device=v.device)
        r = r - torch.dot(r, v) * v
        return r / (r.norm() + 1e-12)

    # -- interventions -------------------------------------------------------

    @staticmethod
    def project_out(base: torch.Tensor, remove: torch.Tensor) -> torch.Tensor:
        """Remove the component(s) of *base* parallel to *remove*.

        Args:
            base: [dim] vector to modify.
            remove: [dim] single direction **or** [k, dim] subspace basis.

        Returns:
            [dim] projected vector (same dtype as *base*).
        """
        orig_dtype = base.dtype
        base = _to_float(base)
        remove = remove.to(device=base.device, dtype=base.dtype)

        if remove.ndim == 1:
            result = base - torch.dot(base, remove) * remove
        elif remove.ndim == 2:
            result = _project_out_subspace(base, remove)
        else:
            raise ValueError(f"remove must be 1-D or 2-D, got {remove.ndim}-D")

        return result.to(orig_dtype)

    @staticmethod
    def inject(base: torch.Tensor, direction: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """base + scale * unit(direction)."""
        orig_dtype = base.dtype
        base, direction = _to_float(base), _to_float(direction.to(base.device))
        return (base + scale * direction / (direction.norm() + 1e-12)).to(orig_dtype)

    @staticmethod
    def subtract(base: torch.Tensor, direction: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """base − scale * unit(direction)."""
        orig_dtype = base.dtype
        base, direction = _to_float(base), _to_float(direction.to(base.device))
        return (base - scale * direction / (direction.norm() + 1e-12)).to(orig_dtype)


def _project_out_subspace(base: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Project *base* out of the subspace spanned by *vectors* [k, dim] (Gram–Schmidt)."""
    result = base.clone()
    for i in range(vectors.shape[0]):
        v = vectors[i].clone()
        for j in range(i):
            v = v - torch.dot(v, vectors[j]) * vectors[j]
        n = v.norm()
        if n > 1e-12:
            v = v / n
            result = result - torch.dot(result, v) * v
    return result
