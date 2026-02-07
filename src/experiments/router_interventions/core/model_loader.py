"""Unified model and tokenizer loading with multiple strategies (quantization, device mapping)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .device_mapper import DeviceMapBuilder

if TYPE_CHECKING:
    from .config import ExperimentConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load models and tokenizers with configurable strategies for memory optimization.
    
    Supports:
    - Quantization: 8-bit or 4-bit for memory reduction
    - Device mapping: intelligent GPU filling, single device, or auto
    - Memory reservation: ensures space for activations during forward pass
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer for the model."""
        logger.info("Loading tokenizer from %s", self.config.model_id)
        return AutoTokenizer.from_pretrained(self.config.model_id)

    def load_model(self) -> AutoModelForCausalLM:
        """Load model with configured quantization and device mapping strategy.
        
        Returns:
            Loaded model in eval mode.
        """
        load_kwargs = self._build_load_kwargs()
        
        logger.info("Loading model from %s", self.config.model_id)
        if self.config.quantization:
            logger.info("Using %s quantization", self.config.quantization)
        
        model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **load_kwargs)
        
        # Handle single device loading (no device_map)
        if self.config.use_single_device and torch.cuda.is_available() and not self.config.quantization:
            self._warn_single_device_memory(model)
            model = model.to("cuda")
        
        model.eval()
        return model

    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load both model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer).
        """
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        return model, tokenizer

    def _build_load_kwargs(self) -> dict:
        """Build kwargs for from_pretrained based on config."""
        if self.config.quantization == "8bit":
            return self._build_8bit_kwargs()
        elif self.config.quantization == "4bit":
            return self._build_4bit_kwargs()
        else:
            return self._build_standard_kwargs()

    def _build_8bit_kwargs(self) -> dict:
        """Build kwargs for 8-bit quantization (~4x memory reduction)."""
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,  # Allow CPU offload for modules that don't fit
            ),
            "device_map": "auto",  # Let HuggingFace auto-place the model
        }

    def _build_4bit_kwargs(self) -> dict:
        """Build kwargs for 4-bit quantization (~8x memory reduction)."""
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True,  # Allow CPU offload if needed
            ),
            "device_map": "auto",  # Let HuggingFace auto-place the model
        }

    def _build_standard_kwargs(self) -> dict:
        """Build kwargs for standard (non-quantized) loading with device mapping."""
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": False,  # Avoid meta tensors for router access
        }

        # Determine device mapping strategy
        if self.config.use_single_device:
            load_kwargs["device_map"] = None  # Load to CPU first, then move to cuda
        elif self.config.target_layer_only_gpu:
            # Use intelligent device mapping: fill GPU optimally
            device_map = self._build_intelligent_device_map()
            load_kwargs["device_map"] = device_map
        else:
            # Default: HuggingFace auto device mapping
            load_kwargs["device_map"] = "auto"

        return load_kwargs

    def _build_intelligent_device_map(self) -> dict:
        """Build device map that fills GPU with layers while ensuring target layer is on GPU."""
        builder = DeviceMapBuilder(
            num_layers=self.config.num_layers,
            model_id=self.config.model_id,
        )
        
        # Reserve memory for activations: depends on batch_size and seq_len
        # Formula: base 2.0 GiB + overhead for batch/sequence size
        activation_reserve = 2.0 + (self.config.batch_size * self.config.seq_len * 0.0001)
        
        logger.info(
            "Building intelligent device map (priority: layer %d, reserve %.1f GiB for activations)",
            self.config.layer_idx, activation_reserve,
        )
        
        return builder.build_with_priority_layers(
            priority_layers=[self.config.layer_idx],
            fill_strategy="sequential",
            activation_reserve_gib=activation_reserve,
        )

    def _warn_single_device_memory(self, model: AutoModelForCausalLM) -> None:
        """Warn if single device loading might cause OOM."""
        num_params = sum(p.numel() for p in model.parameters())
        # bf16: 2 bytes per param; add ~10% for overhead
        need_gib = (num_params * 2 * 1.1) / (1024**3)
        
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_gib = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            
            if need_gib > total_gib * 0.9:  # Using >90% of GPU
                logger.warning(
                    "Model needs ~%.1f GiB but GPU has only %.1f GiB. "
                    "Consider using --quantization 8bit or removing --use-single-device.",
                    need_gib, total_gib,
                )
            else:
                logger.info(
                    "Model has ~%.0fM params, needs ~%.1f GiB (GPU has %.1f GiB)",
                    num_params / 1e6, need_gib, total_gib,
                )


def load_model_and_tokenizer(config: ExperimentConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Convenience function to load model and tokenizer from config.
    
    Args:
        config: Experiment configuration.
    
    Returns:
        Tuple of (model, tokenizer).
    """
    loader = ModelLoader(config)
    return loader.load_model_and_tokenizer()
