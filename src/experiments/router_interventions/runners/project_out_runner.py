"""Runner for project-out experiment: remove direction from router rows, compare loss."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..core import (
    ExperimentConfig,
    BatchLoader,
    LossEvaluator,
    RouterManager,
    ExpertVectors,
    VectorIntervention,
)
from ..core.data import TextListBatchLoader, WikitextBatchLoader

logger = logging.getLogger(__name__)

VALID_VARIATIONS = ("svd", "orthogonal", "random")


class ProjectOutRunner:
    """Project out a direction from each expert's router row; compare loss (baseline + svd/orthogonal/random)."""

    def __init__(
        self,
        config: ExperimentConfig,
        batch_loader: Optional[BatchLoader] = None,
    ) -> None:
        self.config = config
        self._batch_loader = batch_loader

    def _make_batch_loader(self, tokenizer: "AutoTokenizer") -> BatchLoader:
        if self.config.dataset == "wikitext":
            return WikitextBatchLoader(
                tokenizer,
                num_samples=self.config.num_samples,
                seq_len=self.config.seq_len,
                batch_size=self.config.batch_size,
            )
        if self.config.dataset == "text":
            if not self.config.text_file:
                raise ValueError("config.dataset is 'text' but config.text_file is not set")
            path = Path(self.config.text_file)
            if not path.exists():
                raise FileNotFoundError(f"Text file not found: {path}")
            texts = path.read_text().strip().splitlines()
            texts = [t.strip() for t in texts if t.strip()]
            return TextListBatchLoader(
                tokenizer,
                texts=texts,
                seq_len=self.config.seq_len,
                batch_size=self.config.batch_size,
            )
        raise ValueError(f"Unknown dataset: {self.config.dataset}")

    def _get_remove_vector(
        self,
        variant: str,
        v_svd: torch.Tensor,
        exp_idx: int,
        intervention: VectorIntervention,
    ) -> torch.Tensor:
        """Return the direction to project out for this variant. Raises on unknown variant."""
        if variant not in VALID_VARIATIONS:
            raise ValueError(
                f"Unknown variant: {variant!r}. Must be one of {VALID_VARIATIONS}"
            )
        seed = self.config.seed + exp_idx
        if variant == "svd":
            return v_svd
        if variant == "orthogonal":
            return intervention.make_orthogonal(v_svd, seed=seed)
        # variant == "random"
        return intervention.make_random(v_svd.shape[0], seed=seed)

    def run(self) -> Dict[str, Any]:
        """Load model and data, run project-out variants, save results to config.output_file."""
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        if self._batch_loader is not None:
            data_loader = self._batch_loader
        else:
            data_loader = self._make_batch_loader(tokenizer)
        batches = data_loader.get_batches()

        expert_vectors = ExpertVectors(
            self.config.svd_dir,
            self.config.layer_idx,
            self.config.num_experts,
            self.config.model_tag,
        )
        expert_vectors.load()
        if len(expert_vectors) == 0:
            raise RuntimeError("No SVD vectors loaded. Check svd_dir and file names.")

        evaluator = LossEvaluator(model)
        results: Dict[str, Any] = {
            "config": vars(self.config),
            "results": {},
        }

        intervention = VectorIntervention()
        with RouterManager(model, self.config.layer_idx) as router:
            logger.info("Evaluating baseline...")
            base_loss = evaluator.evaluate(batches)
            results["baseline_loss"] = base_loss
            logger.info("Baseline loss: %.4f", base_loss)

            for variant in self.config.variations:
                logger.info("--- Variant: %s ---", variant.upper())
                modified = router.original_weights.clone()

                for exp_idx, v_svd in expert_vectors.items():
                    v_remove = self._get_remove_vector(
                        variant, v_svd, exp_idx, intervention
                    )
                    row = modified[exp_idx]
                    modified[exp_idx] = intervention.project_out(row, v_remove)

                router.apply_weights(modified)
                loss = evaluator.evaluate(batches)
                delta = loss - base_loss
                logger.info(
                    "Result [%s]: loss=%.4f, delta=%+.4f", variant, loss, delta
                )
                results["results"][variant] = {"loss": loss, "delta": delta}

        self._save_results(results)
        return results

    def _save_results(self, results: Dict[str, Any]) -> None:
        out_path = Path(self.config.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving results to %s", out_path)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)


def run_project_out_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Run project-out experiment: build runner and run."""
    return ProjectOutRunner(cfg).run()
