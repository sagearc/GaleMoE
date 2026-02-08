"""Model and tokenizer loading with optional quantization."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if TYPE_CHECKING:
    from .config import ExperimentConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load HuggingFace model + tokenizer (float / 8bit / 4bit, device_map="auto")."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def load_tokenizer(self) -> AutoTokenizer:
        logger.info("Loading tokenizer from %s", self.config.model_id)
        return AutoTokenizer.from_pretrained(self.config.model_id)

    def load_model(self) -> AutoModelForCausalLM:
        kwargs = self._load_kwargs()
        logger.info("Loading model from %s (%s)", self.config.model_id,
                     self.config.quantization or "float")
        model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **kwargs)
        model.eval()
        return model

    def _load_kwargs(self) -> dict:
        q = self.config.quantization
        if q == "8bit":
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True),
                "device_map": "auto",
            }
        if q == "4bit":
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True),
                "device_map": "auto",
            }
        return {"torch_dtype": torch.bfloat16, "device_map": "auto"}
