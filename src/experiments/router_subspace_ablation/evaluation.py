"""Loss evaluation for router subspace ablation."""
from __future__ import annotations

from typing import List

import torch


class LossEvaluator:
    """Evaluates mean causal LM loss over batches (labels = input_ids)."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self._device = next(model.parameters()).device

    @torch.no_grad()
    def evaluate(self, batches: List[torch.Tensor]) -> float:
        """Return mean loss over the given batches."""
        self.model.eval()
        losses: List[float] = []
        for batch in batches:
            batch = batch.to(self._device)
            out = self.model(input_ids=batch, labels=batch)
            losses.append(out.loss.item())
        return sum(losses) / len(losses) if losses else 0.0
