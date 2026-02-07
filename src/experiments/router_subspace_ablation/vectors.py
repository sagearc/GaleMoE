"""Vector utilities: load SVD vectors, project out directions."""
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
    ) -> None:
        self.svd_dir = svd_dir
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.model_tag = model_tag
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

    @staticmethod
    def _load_single(filepath: str) -> torch.Tensor:
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
        if tensor is None:
            raise ValueError(f"Could not extract vector from {filepath}")
        t = torch.as_tensor(tensor, dtype=torch.float32, device="cpu")
        if t.ndim == 2:
            t = t[:, 0] if t.shape[0] > t.shape[1] else t[0]
        return t / (t.norm() + 1e-12)

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, expert_idx: int) -> bool:
        return expert_idx in self._vectors

    def __getitem__(self, expert_idx: int) -> torch.Tensor:
        return self._vectors[expert_idx]

    def keys(self) -> Iterator[int]:
        return iter(self._vectors)

    def items(self) -> Iterator[tuple[int, torch.Tensor]]:
        return iter(self._vectors.items())


class VectorIntervention:
    """Vector operations for subspace ablation: random/orthogonal directions and project-out.
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
        r = torch.randn_like(v, generator=g)
        r_orth = r - (torch.dot(r, v) * v)
        return r_orth / (r_orth.norm() + 1e-12)

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
        """Remove the component of base_vector parallel to remove_vector."""
        if base_vector.device != remove_vector.device:
            remove_vector = remove_vector.to(
                base_vector.device, dtype=base_vector.dtype
            )
        dot = torch.dot(base_vector, remove_vector)
        return base_vector - (dot * remove_vector)
