"""Data loading for router subspace ablation: abstract batch loader + implementations."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


class BatchLoader(ABC):
    """Abstract base for loaders that produce batches of token IDs for LM evaluation.

    Each batch is a tensor of shape [batch_size, seq_len] (input_ids; labels = input_ids for causal LM).
    """

    @abstractmethod
    def get_batches(self) -> List[torch.Tensor]:
        """Return a list of batches, each of shape [batch_size, seq_len]."""
        ...


def _tokenize_to_batches(
    tokenizer: AutoTokenizer,
    texts: List[str],
    seq_len: int,
    batch_size: int,
) -> List[torch.Tensor]:
    """Shared helper: tokenize texts and chunk into fixed-size batches."""
    if not texts:
        raise ValueError("No texts to tokenize.")
    all_ids: List[torch.Tensor] = []
    for text in texts:
        enc = tokenizer(
            text, return_tensors="pt", padding=False, truncation=False
        )
        all_ids.append(enc["input_ids"].squeeze(0))
    ids = torch.cat(all_ids, dim=0)
    n_tokens = (ids.numel() // seq_len) * seq_len
    if n_tokens == 0:
        raise ValueError("Not enough data to create a single batch.")
    ids = ids[:n_tokens].view(-1, seq_len)
    return [
        ids[i : i + batch_size]
        for i in range(0, len(ids), batch_size)
    ]


class WikitextBatchLoader(BatchLoader):
    """Streams Wikipedia (or wikitext-2), tokenizes, and returns fixed-size batches."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_samples: int = 200,
        seq_len: int = 512,
        batch_size: int = 4,
        min_text_length: int = 100,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.min_text_length = min_text_length
        self._cache_dir = self._resolve_cache_dir()

    @staticmethod
    def _resolve_cache_dir() -> str:
        if not os.environ.get("HF_DATASETS_CACHE") and not os.environ.get("HF_HOME"):
            cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
        return os.environ.get("HF_DATASETS_CACHE") or str(
            Path.home() / ".cache" / "huggingface" / "datasets"
        )

    def _load_streaming_dataset(self):
        try:
            return load_dataset(
                "wikipedia", "20220301.en",
                split="train", streaming=True, cache_dir=self._cache_dir,
            )
        except Exception:
            try:
                return load_dataset(
                    "wikitext", "wikitext-2-raw-v1",
                    split="train", streaming=True, cache_dir=self._cache_dir,
                )
            except Exception as e:
                raise RuntimeError(f"Could not load dataset: {e}") from e

    def _collect_texts(self) -> List[str]:
        ds = self._load_streaming_dataset()
        texts: List[str] = []
        for x in ds:
            if len(texts) >= self.num_samples:
                break
            if len(x["text"]) >= self.min_text_length:
                texts.append(x["text"])
        return texts

    def get_batches(self) -> List[torch.Tensor]:
        texts = self._collect_texts()
        return _tokenize_to_batches(
            self.tokenizer, texts, self.seq_len, self.batch_size
        )


class TextListBatchLoader(BatchLoader):
    """Builds batches from an explicit list of text strings (e.g. eval set or custom corpus)."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        texts: List[str],
        seq_len: int = 512,
        batch_size: int = 4,
    ) -> None:
        self.tokenizer = tokenizer
        self.texts = texts
        self.seq_len = seq_len
        self.batch_size = batch_size

    def get_batches(self) -> List[torch.Tensor]:
        return _tokenize_to_batches(
            self.tokenizer, self.texts, self.seq_len, self.batch_size
        )
