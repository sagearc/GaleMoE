"""Runner for project-out experiment: remove direction from router rows, compare loss."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ..core import (
    ExperimentConfig,
    BatchLoader,
    LossEvaluator,
    ModelLoader,
    RouterManager,
    ExpertVectors,
    VectorIntervention,
    Timer,
    timer,
)
from ..core.data import TextListBatchLoader, WikitextBatchLoader
from ..core.memory import log_gpu_memory

logger = logging.getLogger(__name__)

VALID_VARIATIONS = ("svd", "orthogonal", "random", "zero", "shuffle")
PROJECT_OUT_VARIATIONS = ("svd", "orthogonal", "random")  # require expert vectors


class ProjectOutRunner:
    """Project out a direction from each expert's router row; compare loss (baseline + variants including zero/shuffle)."""

    def __init__(
        self,
        config: ExperimentConfig,
        batch_loader: Optional[BatchLoader] = None,
    ) -> None:
        self.config = config
        self._batch_loader = batch_loader

    def _make_batch_loader(self, tokenizer) -> BatchLoader:
        if self.config.dataset == "wikitext":
            return WikitextBatchLoader(
                tokenizer,
                num_samples=self.config.num_samples,
                seq_len=self.config.seq_len,
                batch_size=self.config.batch_size,
            )
        if self.config.dataset == "text":
            if not self.config.text_file:
                raise ValueError(
                    "config.dataset is 'text' but config.text_file is not set"
                )
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
        """Return the direction(s) to project out for this variant. Raises on unknown variant.

        Returns:
            - For variant="svd": returns v_svd as-is (may be [dim] or [k, dim])
            - For variant="orthogonal" or "random": returns [dim] tensor (single vector)
        """
        if variant not in PROJECT_OUT_VARIATIONS:
            raise ValueError(
                f"Unknown project-out variant: {variant!r}. Must be one of {PROJECT_OUT_VARIATIONS}"
            )
        seed = self.config.seed + exp_idx
        if variant == "svd":
            return v_svd
        # For orthogonal and random variants, we need a single vector
        # Get the first vector if v_svd is [k, dim], otherwise use as-is
        if v_svd.ndim == 2:
            v_first = v_svd[0]  # [dim]
        else:
            v_first = v_svd  # [dim]
        if variant == "orthogonal":
            return intervention.make_orthogonal(v_first, seed=seed)
        # variant == "random"
        return intervention.make_random(v_first.shape[0], seed=seed)

    def _k_values_to_run(self) -> list[int]:
        """Return list of top_k values to run."""
        return list(self.config.top_k)

    def run(self) -> Dict[str, Any]:
        """Load model and data, run project-out variants, save results to config.output_file.
        Runs over each k in config.top_k. If len(top_k) > 1, stores results in results['by_k'][k];
        otherwise stores in results['results'] (backward compatible).
        """
        t = Timer()
        t.start("total")

        log_gpu_memory("Before model load")

        # Load model and tokenizer using unified loader
        loader = ModelLoader(self.config)
        with timer("Loading tokenizer and model"):
            tokenizer = loader.load_tokenizer()
            model = loader.load_model()

        log_gpu_memory("After model load")

        with timer("Loading batches"):
            if self._batch_loader is not None:
                data_loader = self._batch_loader
            else:
                data_loader = self._make_batch_loader(tokenizer)
            batches = data_loader.get_batches()
            
            # For quantized models, keep batches on CPU to save memory
            # For non-quantized, pre-move to device if enough memory
            device = next(model.parameters()).device
            if self.config.quantization:
                logger.info("Keeping %d batches on CPU (quantized model)", len(batches))
            elif torch.cuda.is_available():
                # Only pre-move if we have at least 2GB free GPU memory
                free_mem_gib = (torch.cuda.get_device_properties(0).total_memory - 
                               torch.cuda.memory_allocated(0)) / (1024**3)
                if free_mem_gib >= 2.0:
                    with torch.no_grad():
                        batches = [b.to(device) for b in batches]
                    logger.info("Pre-moved %d batches to %s (%.1f GiB free)", 
                               len(batches), device, free_mem_gib)
                else:
                    logger.info("Keeping %d batches on CPU (only %.1f GiB GPU free)", 
                               len(batches), free_mem_gib)

        # Materialize lazy (meta) parameters: run one forward so gate weights are materialized
        with timer("Materializing model (first forward)"):
            with torch.no_grad():
                model.eval()
                first_batch = batches[0]
                device = next(model.parameters()).device
                _ = model(first_batch.to(device))
        log_gpu_memory("After first forward (batches loaded)")

        evaluator = LossEvaluator(model)
        k_values = self._k_values_to_run()
        multi_k = len(k_values) > 1

        results: Dict[str, Any] = {
            "config": vars(self.config),
            "results": {},
        }
        if multi_k:
            results["by_k"] = {}

        # Validate variations
        for v in self.config.variations:
            if v not in VALID_VARIATIONS:
                raise ValueError(
                    f"Unknown variation: {v!r}. Must be one of {VALID_VARIATIONS}"
                )

        intervention = VectorIntervention()
        with RouterManager(model, self.config.layer_idx) as router:
            with timer("Baseline evaluation"):
                logger.info("Evaluating baseline...")
                base_loss = evaluator.evaluate(batches)
                results["baseline_loss"] = base_loss
                logger.info("Baseline loss: %.4f", base_loss)

            orig = router.original_weights
            logger.info("Router weights shape: %s, dtype: %s", orig.shape, orig.dtype)
            num_experts = orig.shape[0]

            # Precompute zero and shuffle once (they don't depend on k)
            baseline_interventions: Dict[str, Dict[str, Any]] = {}
            if "zero" in self.config.variations:
                with timer("Zero intervention"):
                    logger.info("Computing baseline: ZERO (all router weights = 0)")
                    router.apply_weights(torch.zeros_like(orig))
                    loss_zero = evaluator.evaluate(batches)
                    baseline_interventions["zero"] = {
                        "loss": loss_zero,
                        "delta": loss_zero - base_loss,
                    }
                    logger.info(
                        "  → loss=%.4f, delta=%+.4f",
                        loss_zero,
                        baseline_interventions["zero"]["delta"],
                    )
                    router.restore()
            if "shuffle" in self.config.variations:
                with timer("Shuffle intervention"):
                    logger.info("Computing baseline: SHUFFLE (permute router rows)")
                    g = torch.Generator().manual_seed(self.config.seed)
                    perm = torch.randperm(num_experts, generator=g)
                    # No need to clone - permutation creates a new contiguous tensor
                    router.apply_weights(orig[perm])
                    loss_shuffle = evaluator.evaluate(batches)
                    baseline_interventions["shuffle"] = {
                        "loss": loss_shuffle,
                        "delta": loss_shuffle - base_loss,
                    }
                    logger.info(
                        "  → loss=%.4f, delta=%+.4f",
                        loss_shuffle,
                        baseline_interventions["shuffle"]["delta"],
                    )
                    router.restore()

            # Load expert vectors once for max(k) and reuse for each k (avoids repeated disk I/O)
            with timer("Loading expert vectors"):
                max_k = max(k_values)
                logger.info(
                    "Loading expert vectors once (top_k=%d) for all k values", max_k
                )
                expert_vectors = ExpertVectors(
                    self.config.svd_dir,
                    self.config.layer_idx,
                    self.config.num_experts,
                    self.config.model_tag,
                    top_k=max_k,
                )
                expert_vectors.load()
            if len(expert_vectors) == 0:
                raise RuntimeError(
                    "No SVD vectors loaded. Check svd_dir and file names."
                )

            for top_k in k_values:
                if multi_k:
                    logger.info("\n=== Running interventions with top_k=%d ===", top_k)
                    results["by_k"][top_k] = {}

                for variant in self.config.variations:
                    if variant in baseline_interventions:
                        # Reuse precomputed baseline (doesn't depend on k)
                        entry = baseline_interventions[variant]
                    else:
                        # Compute project-out intervention for this k
                        with timer(f"Intervention {variant} (k={top_k})"):
                            logger.info("Computing: %s (k=%d)", variant.upper(), top_k)
                            # Reuse allocation: clone once per k, not per variant
                            if variant == self.config.variations[0] or "modified" not in locals():
                                modified = router.original_weights.clone()
                            else:
                                # Reset to original for next variant (faster than clone)
                                modified.copy_(router.original_weights)

                            # Debug: check shape
                            if modified.dim() != 2:
                                logger.error(
                                    "Modified weights have wrong shape: %s (expected 2D)",
                                    modified.shape,
                                )
                                raise ValueError(
                                    f"Router weights have unexpected shape: {modified.shape}"
                                )

                            for exp_idx, v_svd_full in expert_vectors.items():
                                # Use first top_k vectors (slice of pre-loaded max_k vectors)
                                if v_svd_full.ndim == 2:
                                    v_svd = v_svd_full[:top_k]  # [k, dim]
                                    if v_svd.shape[0] == 1:
                                        v_svd = v_svd[0]  # [dim] for project_out compat
                                else:
                                    v_svd = v_svd_full
                                v_remove = self._get_remove_vector(
                                    variant, v_svd, exp_idx, intervention
                                )
                                row = modified[exp_idx]

                                # Debug: check row shape
                                if row.dim() != 1:
                                    logger.error(
                                        "Row %d has wrong shape: %s (expected 1D)",
                                        exp_idx,
                                        row.shape,
                                    )
                                    raise ValueError(
                                        f"Router row {exp_idx} has unexpected shape: {row.shape}"
                                    )

                                modified[exp_idx] = intervention.project_out(
                                    row, v_remove
                                )

                            router.apply_weights(modified)
                            loss = evaluator.evaluate(batches)
                            entry = {"loss": loss, "delta": loss - base_loss}
                            logger.info(
                                "  → loss=%.4f, delta=%+.4f",
                                entry["loss"],
                                entry["delta"],
                            )

                    if multi_k:
                        results["by_k"][top_k][variant] = entry
                    else:
                        results["results"][variant] = entry

        with timer("Saving results"):
            self._save_results(results)

        t.stop("total")
        logger.info("\n" + "=" * 60)
        logger.info("TIMING SUMMARY")
        t.report()
        logger.info("=" * 60)

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
