"""Device map builder for optimal GPU memory usage with priority layers."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class DeviceMapBuilder:
    """Build device_map for HuggingFace models with optimal GPU usage and priority layers.
    
    Estimates layer sizes and fills GPU with as many layers as fit, ensuring priority layers
    (e.g. target layer for experiments) are on GPU. Remaining layers go to CPU.
    """

    def __init__(
        self,
        num_layers: int,
        model_id: str = "mistralai/Mixtral-8x7B-v0.1",
        target_device: str = "cuda:0",
    ) -> None:
        self.num_layers = num_layers
        self.model_id = model_id
        self.target_device = target_device
        self._layer_size_gib: Optional[float] = None

    def _estimate_layer_size_gib(self) -> float:
        """Estimate single layer size in GiB for the model."""
        if self._layer_size_gib is not None:
            return self._layer_size_gib
        
        # Rough estimates for common models (bf16: 2 bytes/param)
        # Mixtral-8x7B: ~46B total params, 32 layers → ~1.4B params/layer → ~2.8 GiB/layer
        # These are approximations; actual can vary by layer
        if "Mixtral-8x7B" in self.model_id or "mixtral-8x7b" in self.model_id.lower():
            self._layer_size_gib = 2.8
        elif "Mixtral-8x22B" in self.model_id or "mixtral-8x22b" in self.model_id.lower():
            self._layer_size_gib = 5.0
        elif "7B" in self.model_id or "7b" in self.model_id.lower():
            # Generic 7B model
            self._layer_size_gib = 0.5
        else:
            # Default: assume ~1 GiB per layer for unknown models
            logger.warning("Unknown model %s; using default layer size 1.0 GiB", self.model_id)
            self._layer_size_gib = 1.0
        return self._layer_size_gib

    def _get_available_gpu_memory_gib(self) -> float:
        """Get available GPU memory in GiB (total - allocated)."""
        if not torch.cuda.is_available():
            return 0.0
        device_idx = int(self.target_device.split(":")[-1]) if ":" in self.target_device else 0
        props = torch.cuda.get_device_properties(device_idx)
        total = props.total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
        return total - allocated

    def build_with_priority_layers(
        self,
        priority_layers: List[int],
        fill_strategy: str = "sequential",
        activation_reserve_gib: float = 2.0,
    ) -> Dict[str, str]:
        """Build device_map ensuring priority_layers are on GPU, fill remaining GPU with other layers.
        
        Args:
            priority_layers: Layer indices that must be on GPU (e.g. [0, 15] for experiments on those layers).
            fill_strategy: How to fill remaining GPU memory:
                - "sequential": fill from layer 0, 1, 2, ... (default)
                - "nearby": prioritize layers near priority_layers
                - "none": only put priority layers and essential modules on GPU
            activation_reserve_gib: Memory to reserve for activations during forward pass (default: 2.0 GiB).
        
        Returns:
            device_map dict suitable for from_pretrained(device_map=...).
        """
        device_map: Dict[str, str] = {}
        layer_size = self._estimate_layer_size_gib()
        available = self._get_available_gpu_memory_gib()
        
        # Reserve space for embeddings, norm, lm_head (~2-4 GiB for Mixtral)
        essential_size = 3.0
        available -= essential_size
        
        # Reserve space for activations during forward pass (critical!)
        # Add extra buffer for safety
        available -= (activation_reserve_gib + 0.5)  # Extra 0.5 GiB safety buffer
        
        device_map["model.embed_tokens"] = self.target_device
        device_map["model.norm"] = self.target_device
        device_map["lm_head"] = self.target_device
        
        logger.info(
            "Building device_map: %d layers, ~%.1f GiB/layer, ~%.1f GiB available GPU (reserved %.1f GiB for activations)",
            self.num_layers, layer_size, available, activation_reserve_gib,
        )
        
        # Priority layers must be on GPU
        for layer_idx in priority_layers:
            if layer_idx < 0 or layer_idx >= self.num_layers:
                raise ValueError(f"priority_layer {layer_idx} out of range [0, {self.num_layers})")
            device_map[f"model.layers.{layer_idx}"] = self.target_device
        
        priority_size = len(priority_layers) * layer_size
        available -= priority_size
        
        if available < 0:
            logger.warning(
                "Priority layers need %.1f GiB but only %.1f GiB available after reservations. "
                "GPU will likely OOM during forward. Consider: reducing --batch-size, --seq-len, or --num-layers.",
                priority_size, available + priority_size,
            )
        
        # Fill remaining GPU memory with other layers based on strategy
        if fill_strategy == "none":
            # Only priority layers on GPU
            pass
        elif fill_strategy == "sequential":
            # Fill from layer 0, 1, 2, ... (skip priority layers already placed)
            for i in range(self.num_layers):
                if i in priority_layers:
                    continue
                if available >= layer_size:
                    device_map[f"model.layers.{i}"] = self.target_device
                    available -= layer_size
                else:
                    break
        elif fill_strategy == "nearby":
            # Fill layers near priority layers first (within distance, then expand)
            # For simplicity, interleave: closest to priority layers first
            priority_set = set(priority_layers)
            distances = {}
            for i in range(self.num_layers):
                if i in priority_set:
                    continue
                distances[i] = min(abs(i - p) for p in priority_layers)
            # Sort by distance (closest first)
            sorted_layers = sorted(distances.keys(), key=lambda x: distances[x])
            for i in sorted_layers:
                if available >= layer_size:
                    device_map[f"model.layers.{i}"] = self.target_device
                    available -= layer_size
                else:
                    break
        else:
            raise ValueError(f"Unknown fill_strategy: {fill_strategy!r}")
        
        # All layers not explicitly assigned go to CPU
        for i in range(self.num_layers):
            key = f"model.layers.{i}"
            if key not in device_map:
                device_map[key] = "cpu"
        
        gpu_layers = sum(1 for k, v in device_map.items() if "layers." in k and v == self.target_device)
        gpu_memory_used = gpu_layers * layer_size + essential_size
        logger.info(
            "Device map: %d/%d layers on %s, %d on CPU (est. %.1f GiB GPU used, %.1f GiB free for activations)",
            gpu_layers, self.num_layers, self.target_device, self.num_layers - gpu_layers,
            gpu_memory_used, available,
        )
        return device_map
