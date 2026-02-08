"""Data loading: abstract batch loader + implementations."""
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
    max_rows: int | None = None,
) -> List[torch.Tensor]:
    """Shared helper: tokenize texts and chunk into fixed-size batches.

    max_rows: If set, use at most this many rows (seq_len-sized segments). Used by WikitextBatchLoader
    so num_samples caps evaluation samples, not source texts.
    """
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
    if max_rows is not None and ids.size(0) > max_rows:
        ids = ids[:max_rows]
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
        # Cap total rows to num_samples so num_samples = evaluation samples (not source text count)
        return _tokenize_to_batches(
            self.tokenizer,
            texts,
            self.seq_len,
            self.batch_size,
            max_rows=self.num_samples,
        )


def _tokenize_titles_to_batches(
    tokenizer: AutoTokenizer,
    titles: List[str],
    seq_len: int,
    batch_size: int,
) -> List[torch.Tensor]:
    """Tokenize short texts (titles), truncate/pad each to seq_len, then batch. Like gate-hook."""
    if not titles:
        raise ValueError("No titles to tokenize.")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    all_ids: List[torch.Tensor] = []
    for text in titles:
        enc = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )
        all_ids.append(enc["input_ids"].squeeze(0))
    ids = torch.stack(all_ids)
    return [
        ids[i : i + batch_size]
        for i in range(0, len(ids), batch_size)
    ]


def _resolve_cache_dir_static() -> str:
    if not os.environ.get("HF_DATASETS_CACHE") and not os.environ.get("HF_HOME"):
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    return os.environ.get("HF_DATASETS_CACHE") or str(
        Path.home() / ".cache" / "huggingface" / "datasets"
    )


class WikiTitlesBatchLoader(BatchLoader):
    """Loads Wikipedia (or Wikitext) titles — short strings, seq_len=32 — like gate-hook. Enables large batch sizes."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_samples: int = 500,
        seq_len: int = 32,
        batch_size: int = 64,
        cache_dir: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.batch_size = batch_size
        self._cache_dir = cache_dir or _resolve_cache_dir_static()

    def _load_titles(self) -> List[str]:
        for config_name in ("20220301.en", "20231101.en"):
            try:
                ds = load_dataset(
                    "wikimedia/wikipedia",
                    config_name,
                    split="train",
                    streaming=True,
                    cache_dir=self._cache_dir,
                    trust_remote_code=True,
                )
                titles: List[str] = []
                for i, x in enumerate(ds):
                    if i >= self.num_samples:
                        break
                    t = x.get("title") or (x.get("text") or "")[:80]
                    if isinstance(t, str) and t.strip():
                        titles.append(t.strip())
                if titles:
                    return titles
            except Exception:
                continue
        try:
            ds = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                split="train",
                cache_dir=self._cache_dir,
            )
            titles = []
            for row in ds:
                if len(titles) >= self.num_samples:
                    break
                line = (row.get("text") or "").strip()
                if line and not line.startswith("=") and len(line) < 200:
                    titles.append(line)
            if titles:
                return titles
        except Exception:
            pass
        raise RuntimeError(
            "Could not load wiki_titles: tried wikimedia/wikipedia and wikitext. "
            "Install datasets and check network."
        )

    def get_batches(self) -> List[torch.Tensor]:
        titles = self._load_titles()
        return _tokenize_titles_to_batches(
            self.tokenizer, titles, self.seq_len, self.batch_size
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
