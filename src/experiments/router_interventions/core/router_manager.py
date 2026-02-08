"""Context manager for temporarily modifying and restoring Mixtral router weights."""
from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _dequantize_int8_weight(weight_param: Any) -> torch.Tensor:
    """Dequantize Int8Params to float. weight_param is gate.weight (has .data int8, .SCB scale)."""
    w = weight_param.data
    if w.dtype != torch.int8:
        return w.float()
    scb = getattr(weight_param, "SCB", None)
    if scb is None:
        logger.warning("Gate weight is int8 but has no SCB; using raw float conversion (may be wrong scale).")
        return w.float() / 127.0
    # Row-wise scale: dequant = (int8 * scale) / 127
    scale = scb.unsqueeze(1) if scb.dim() == 1 else scb
    scale = scale.to(w.device)
    return (w.float() * scale) / 127.0


class RouterManager:
    """Locate, apply, and restore the gate (router) weights for one Mixtral MoE layer.
    Gate weights are cloned on first use (original_weights or apply_weights) so that
    a forward pass can be run first to materialize lazy (meta) parameters.
    With 8-bit quantization, we dequantize to float and replace the gate with nn.Linear for
    the whole run (baseline and interventions). So baseline and intervention both use float
    weights and the delta is a clean measure of the intervention.
    """

    def __init__(self, model: nn.Module, layer_idx: int) -> None:
        self.model = model
        self.layer_idx = layer_idx
        self._gate = self._find_gate()
        self._original: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None
        self._quantized: bool = False
        self._replacement_linear: Optional[nn.Linear] = None
        self._block_sparse_moe: Optional[Any] = None

    def _find_gate(self) -> nn.Module:
        try:
            return self.model.model.layers[self.layer_idx].block_sparse_moe.gate
        except AttributeError as e:
            raise ValueError(f"Could not find Mixtral gate at layer {self.layer_idx}") from e

    def _is_quantized_gate(self) -> bool:
        w = self._gate.weight
        return getattr(w, "data", w).dtype == torch.int8 or getattr(w, "SCB", None) is not None

    def _ensure_materialized(self) -> None:
        """Clone gate weights on first use. If still on meta, materialize from state_dict.
        For 8-bit gates, dequantize to float and store; apply_weights will swap in nn.Linear.
        """
        if self._original is not None:
            return
        w_data = self._gate.weight.data
        if w_data.is_meta:
            key = f"model.layers.{self.layer_idx}.block_sparse_moe.gate.weight"
            state = self.model.state_dict()
            if key not in state:
                for k in state:
                    if f"layers.{self.layer_idx}" in k and "block_sparse_moe" in k and "gate" in k:
                        key = k
                        break
                else:
                    raise RuntimeError(
                        f"Gate weights on meta and key {key!r} not found in state_dict. "
                        "Cannot materialize."
                    )
            loaded = state[key]
            if loaded.is_meta:
                raise RuntimeError(
                    "Gate weights still on meta after state_dict(). "
                    "Ensure model is loaded with device_map=\"auto\" (e.g. with --quantization 8bit)."
                )
            target_device = next(self.model.parameters()).device
            self._gate.weight.data = loaded.clone().to(target_device, dtype=loaded.dtype)
            w_data = self._gate.weight.data
        self._quantized = self._is_quantized_gate()
        if self._quantized:
            w_float = _dequantize_int8_weight(self._gate.weight)
            self._original = w_float.cpu().float()
            self._device = w_float.device
            self._dtype = torch.float32
            out_f, in_f = self._original.shape
            compute_dtype = next(self.model.parameters()).dtype
            self._replacement_linear = nn.Linear(
                in_f, out_f, bias=False, device=self._device, dtype=compute_dtype
            )
            self._replacement_linear.weight.data = self._original.to(
                self._device, dtype=compute_dtype
            )
            block = self.model.model.layers[self.layer_idx].block_sparse_moe
            self._block_sparse_moe = block
            block.gate = self._replacement_linear
            logger.info(
                "Gate is quantized (int8); baseline and interventions use dequantized float for comparable deltas."
            )
        else:
            self._original = w_data.clone().cpu()
            self._device = w_data.device
            self._dtype = w_data.dtype

    @property
    def original_weights(self) -> torch.Tensor:
        """Copy of the original gate weights [num_experts, dim] (float when gate is quantized)."""
        self._ensure_materialized()
        return self._original.clone()

    def apply_weights(self, new_weights: torch.Tensor) -> None:
        """Apply new gate weights (shape must match). For quantized gates, temporarily replace with nn.Linear."""
        self._ensure_materialized()
        if new_weights.shape != self._original.shape:
            raise ValueError(
                f"Shape mismatch: {new_weights.shape} vs {self._original.shape}"
            )
        if self._quantized:
            self._apply_weights_quantized(new_weights)
        else:
            self._gate.weight.data = new_weights.to(self._device, dtype=self._dtype)

    def _apply_weights_quantized(self, new_weights: torch.Tensor) -> None:
        """Update the replacement Linear (already swapped in for baseline) with new_weights."""
        device = new_weights.device if new_weights.device.type != "cpu" else self._device
        compute_dtype = self._replacement_linear.weight.dtype
        self._replacement_linear.weight.data = new_weights.to(device=device, dtype=compute_dtype)

    def restore(self) -> None:
        """Restore original weights or original gate module."""
        if self._original is None:
            return
        if self._quantized and self._block_sparse_moe is not None:
            self._block_sparse_moe.gate = self._gate
            self._block_sparse_moe = None
            logger.debug("Restored original quantized gate module.")
        elif not self._quantized:
            logger.debug("Restoring original router weights.")
            self._gate.weight.data = self._original.to(self._device, dtype=self._dtype)

    def __enter__(self) -> "RouterManager":
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[object]) -> None:
        self.restore()
