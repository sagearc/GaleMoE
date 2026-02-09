"""Evaluation: loss and token-distribution comparison."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


class LossEvaluator:
    """Mean causal-LM loss over batches (ignores padding when pad_token_id is set)."""

    def __init__(self, model: torch.nn.Module, pad_token_id: int | None = None) -> None:
        self.model = model
        # Find first non-meta device
        self._device = next(
            (p.device for p in model.parameters() if not p.is_meta), torch.device("cpu")
        )
        self.pad_token_id = pad_token_id

    @torch.no_grad()
    def evaluate(self, batches: List[torch.Tensor]) -> float:
        self.model.eval()
        losses: List[float] = []
        for batch in batches:
            batch = batch.to(self._device)
            labels = batch.clone()
            if self.pad_token_id is not None:
                labels[batch == self.pad_token_id] = -100
            losses.append(self.model(input_ids=batch, labels=labels).loss.item())
        return sum(losses) / len(losses) if losses else 0.0

    @torch.no_grad()
    def get_logits(self, batches: List[torch.Tensor]) -> torch.Tensor:
        """Concatenated logits [total_tokens, vocab] for all batches."""
        self.model.eval()
        parts = []
        for b in batches:
            out = self.model(input_ids=b.to(self._device))
            parts.append(out.logits.flatten(0, 1))
        return torch.cat(parts, dim=0)


class TokenDistributionComparator:
    """Compare next-token distributions before/after intervention (KL or CE)."""

    def __init__(self, metric: str = "kl") -> None:
        if metric not in ("kl", "ce"):
            raise ValueError(f"metric must be 'kl' or 'ce', got {metric!r}")
        self.metric = metric

    def compare(self, logits_before: torch.Tensor, logits_after: torch.Tensor) -> float:
        p_after = F.softmax(logits_after.float(), dim=-1)
        log_p_before = F.log_softmax(logits_before.float(), dim=-1)
        if self.metric == "kl":
            log_p_after = F.log_softmax(logits_after.float(), dim=-1)
            return (p_after * (log_p_after - log_p_before)).sum(-1).mean().item()
        # CE
        return (
            -(
                F.softmax(logits_before.float(), dim=-1)
                * F.log_softmax(logits_after.float(), dim=-1)
            )
            .sum(-1)
            .mean()
            .item()
        )


def confusion_matrix_top_k(
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    top_k: int = 2,
) -> Tuple[torch.Tensor, List[int]]:
    """Confusion matrix of top-1 predictions before vs after, restricted to top_k tokens."""
    pred_b = logits_before.argmax(-1).cpu()
    pred_a = logits_after.argmax(-1).cpu()
    all_ids = torch.cat([pred_b, pred_a])
    uniq, counts = torch.unique(all_ids, return_counts=True)
    k = min(top_k, len(uniq))
    token_ids = uniq[counts.argsort(descending=True)[:k]].tolist()
    idx_map = {tid: i for i, tid in enumerate(token_ids)}
    mat = torch.zeros(k, k, dtype=torch.long)
    for pb, pa in zip(pred_b.view(-1).tolist(), pred_a.view(-1).tolist()):
        if pb in idx_map and pa in idx_map:
            mat[idx_map[pb], idx_map[pa]] += 1
    return mat, token_ids
