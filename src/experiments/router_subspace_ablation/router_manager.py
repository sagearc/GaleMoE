"""Context manager for temporarily modifying and restoring Mixtral router weights."""
from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


class RouterManager:
    """Locate, apply, and restore the gate (router) weights for one Mixtral MoE layer."""

    def __init__(self, model: nn.Module, layer_idx: int) -> None:
        self.model = model
        self.layer_idx = layer_idx
        self._gate = self._find_gate()
        self._original = self._gate.weight.data.clone().cpu()
        self._device = self._gate.weight.device
        self._dtype = self._gate.weight.dtype

    def _find_gate(self) -> nn.Linear:
        try:
            return self.model.model.layers[self.layer_idx].block_sparse_moe.gate
        except AttributeError as e:
            raise ValueError(f"Could not find Mixtral gate at layer {self.layer_idx}") from e

    @property
    def original_weights(self) -> torch.Tensor:
        """Copy of the original gate weights [num_experts, dim]."""
        return self._original.clone()

    def apply_weights(self, new_weights: torch.Tensor) -> None:
        """Apply new gate weights (shape must match)."""
        if new_weights.shape != self._original.shape:
            raise ValueError(
                f"Shape mismatch: {new_weights.shape} vs {self._original.shape}"
            )
        self._gate.weight.data = new_weights.to(self._device, dtype=self._dtype)

    def restore(self) -> None:
        """Restore original weights."""
        logger.debug("Restoring original router weights.")
        self._gate.weight.data = self._original.to(self._device, dtype=self._dtype)

    def __enter__(self) -> "RouterManager":
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[object]) -> None:
        self.restore()
