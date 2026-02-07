"""Vector utilities: load SVD vectors, project out / inject / subtract directions."""
from __future__ import annotations

import os
import pickle
from typing import Dict, Iterator

import torch


class ExpertVectors:
    """Loads and holds per-expert SVD (top singular) vectors from pickle files."""

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

    def load(self) -> None:
        """Load all expert vectors from svd_dir; raise if any file is missing."""
        self._vectors.clear()
        for i in range(self.num_experts):
            path = self._path_for_expert(i)
            self._vectors[i] = self._load_single(path)

    def _path_for_expert(self, expert_idx: int) -> str:
        fname = f"{self.model_tag}_layer{self.layer_idx}_expert{expert_idx}.pkl"
        return os.path.join(self.svd_dir, fname)

    def _load_single(self, filepath: str) -> torch.Tensor:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"SVD file not found: {filepath}")
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        tensor = None
        if isinstance(obj, dict):
            for key in ["Vh", "vh", "V", "v"]:
                if key in obj:
                    tensor = obj[key]
                    break
        elif isinstance(obj, (tuple, list)) and len(obj) >= 3:
            tensor = obj[2]
        elif isinstance(obj, torch.Tensor):
            tensor = obj
        if tensor is None:
            raise ValueError(f"Could not extract vector from {filepath}")
        t = torch.as_tensor(tensor, dtype=torch.float32, device="cpu")
        
        # Handle 2D matrices (Vh/V matrices with multiple singular vectors)
        if t.ndim == 2:
            # If rows > cols, take first top_k columns; otherwise take first top_k rows
            if t.shape[0] > t.shape[1]:
                # Shape: [dim, num_vectors] -> take first top_k columns
                k = min(self.top_k, t.shape[1])
                t = t[:, :k]  # [dim, k]
                t = t.T  # [k, dim] for consistency
            else:
                # Shape: [num_vectors, dim] -> take first top_k rows
                k = min(self.top_k, t.shape[0])
                t = t[:k, :]  # [k, dim]
        elif t.ndim == 1:
            # Single vector: if top_k > 1, we only have one vector
            if self.top_k > 1:
                raise ValueError(
                    f"Requested top_k={self.top_k} but SVD file {filepath} "
                    "contains only a single vector. Need a matrix with multiple vectors."
                )
            # For top_k=1, return as 1D tensor for backward compatibility
            t = t / (t.norm() + 1e-12)
            return t
        
        # Normalize each vector
        norms = t.norm(dim=-1, keepdim=True) + 1e-12
        t = t / norms
        
        # For top_k=1, return as 1D tensor for backward compatibility
        if self.top_k == 1 and t.shape[0] == 1:
            return t[0]  # [dim]
        
        return t  # [k, dim] for k > 1

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, expert_idx: int) -> bool:
        return expert_idx in self._vectors

    def __getitem__(self, expert_idx: int) -> torch.Tensor:
        """Get vectors for expert. Returns [top_k, dim] tensor if top_k > 1, else [dim] tensor."""
        return self._vectors[expert_idx]
    
    def get_single_vector(self, expert_idx: int) -> torch.Tensor:
        """Get the first (top) vector for an expert. Returns [dim] tensor."""
        v = self._vectors[expert_idx]
        if v.ndim == 2:
            return v[0]  # [dim]
        return v  # Already 1D

    def keys(self) -> Iterator[int]:
        return iter(self._vectors)

    def items(self) -> Iterator[tuple[int, torch.Tensor]]:
        return iter(self._vectors.items())


class VectorIntervention:
    """Vector operations: random/orthogonal directions, project-out, inject, subtract.
    Singleton: use VectorIntervention() to get the single shared instance.
    """

    _instance: "VectorIntervention | None" = None

    def __new__(cls: type["VectorIntervention"]) -> "VectorIntervention":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def make_orthogonal(v: torch.Tensor, seed: int) -> torch.Tensor:
        """Return a random unit vector orthogonal to v (Gram–Schmidt)."""
        g = torch.Generator().manual_seed(seed)
        r = torch.randn(v.shape, generator=g, dtype=v.dtype, device=v.device)
        r_orth = r - (torch.dot(r, v) * v)
        return r_orth / (r_orth.norm() + 1e-12)

    @staticmethod
    def make_orthogonal_to_subspace(vectors: torch.Tensor, seed: int) -> torch.Tensor:
        """Return a random unit vector orthogonal to the subspace spanned by vectors.
        vectors: [k, dim]. Returns [dim] unit vector in the orthogonal complement.
        """
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("vectors must be 2D [k, dim] with k >= 1")
        dim = vectors.shape[1]
        g = torch.Generator().manual_seed(seed)
        r = torch.randn(dim, generator=g, dtype=vectors.dtype, device=vectors.device)
        r_orth = VectorIntervention.project_out_subspace(r, vectors)
        n = r_orth.norm()
        if n < 1e-10:
            raise RuntimeError(
                "Random vector lay in the subspace (numerically). Try a different seed."
            )
        return r_orth / n

    @staticmethod
    def make_random(d_dim: int, seed: int) -> torch.Tensor:
        """Return a random unit vector."""
        g = torch.Generator().manual_seed(seed)
        r = torch.randn(d_dim, generator=g)
        return r / (r.norm() + 1e-12)

    @staticmethod
    def project_out(
        base_vector: torch.Tensor,
        remove_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Remove the component of base_vector parallel to remove_vector.
        
        Args:
            base_vector: [dim] tensor to project
            remove_vector: [dim] tensor (single vector) or [k, dim] tensor (multiple vectors)
        
        Returns:
            [dim] tensor with components parallel to remove_vector(s) removed
        """
        if base_vector.device != remove_vector.device or base_vector.dtype != remove_vector.dtype:
            remove_vector = remove_vector.to(
                base_vector.device, dtype=base_vector.dtype
            )
        
        # Handle both single vector and multiple vectors
        if remove_vector.ndim == 1:
            # Single vector: original behavior
            dot = torch.dot(base_vector, remove_vector)
            return base_vector - (dot * remove_vector)
        elif remove_vector.ndim == 2:
            # Multiple vectors: project out the subspace
            return VectorIntervention.project_out_subspace(base_vector, remove_vector)
        else:
            raise ValueError(f"remove_vector must be 1D or 2D, got {remove_vector.ndim}D")
    
    @staticmethod
    def project_out_subspace(
        base_vector: torch.Tensor,
        remove_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Remove the component of base_vector in the subspace spanned by remove_vectors.
        
        Uses Gram-Schmidt orthogonalization to project out the subspace.
        
        Args:
            base_vector: [dim] tensor to project
            remove_vectors: [k, dim] tensor with k vectors spanning the subspace to remove
        
        Returns:
            [dim] tensor with components in the subspace removed
        """
        if base_vector.device != remove_vectors.device or base_vector.dtype != remove_vectors.dtype:
            remove_vectors = remove_vectors.to(
                base_vector.device, dtype=base_vector.dtype
            )
        
        if remove_vectors.ndim != 2:
            raise ValueError(f"remove_vectors must be 2D [k, dim], got shape {remove_vectors.shape}")
        
        # Orthonormalize the remove_vectors using Gram-Schmidt
        # This ensures we properly project out the subspace even if vectors are not orthogonal
        result = base_vector.clone()
        for i in range(remove_vectors.shape[0]):
            v = remove_vectors[i]
            # Orthogonalize v against previous vectors
            for j in range(i):
                v = v - torch.dot(v, remove_vectors[j]) * remove_vectors[j]
            # Normalize
            v_norm = v.norm()
            if v_norm > 1e-12:
                v = v / v_norm
                # Project out this component
                dot = torch.dot(result, v)
                result = result - (dot * v)
        
        return result

    @staticmethod
    def inject(
        base_vector: torch.Tensor,
        direction: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Add a component along direction: base + scale * unit(direction)."""
        if base_vector.device != direction.device:
            direction = direction.to(
                base_vector.device, dtype=base_vector.dtype
            )
        unit = direction / (direction.norm() + 1e-12)
        return base_vector + scale * unit

    @staticmethod
    def subtract(
        base_vector: torch.Tensor,
        direction: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Remove a component along direction: base - scale * unit(direction)."""
        if base_vector.device != direction.device:
            direction = direction.to(
                base_vector.device, dtype=base_vector.dtype
            )
        unit = direction / (direction.norm() + 1e-12)
        return base_vector - scale * unit
