"""Unified model and tokenizer loading (quantization only)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if TYPE_CHECKING:
    from .config import ExperimentConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load models and tokenizers. Uses quantization (8bit/4bit) or bfloat16, always with device_map=\"auto\"."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer for the model."""
        logger.info("Loading tokenizer from %s", self.config.model_id)
        return AutoTokenizer.from_pretrained(self.config.model_id)

    def load_model(self) -> AutoModelForCausalLM:
        """Load model with quantization or bfloat16; device_map=\"auto\"."""
        load_kwargs = self._build_load_kwargs()
        logger.info("Loading model from %s", self.config.model_id)
        if self.config.quantization:
            logger.info("Using %s quantization", self.config.quantization)
        model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **load_kwargs)
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
        """Build kwargs for non-quantized loading (bfloat16, device_map auto)."""
        return {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": False,
            "device_map": "auto",
        }


def load_model_and_tokenizer(config: ExperimentConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Convenience function to load model and tokenizer from config.
    
    Args:
        config: Experiment configuration.
    
    Returns:
        Tuple of (model, tokenizer).
    """
    loader = ModelLoader(config)
    return loader.load_model_and_tokenizer()
