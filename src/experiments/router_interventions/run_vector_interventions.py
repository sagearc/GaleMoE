"""Vector interventions: inject, subtract, project_out with loss + token distribution comparison."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from .core import (
    ExperimentConfig,
    BatchLoader,
    LossEvaluator,
    ModelLoader,
    RouterManager,
    ExpertVectors,
    VectorIntervention,
    TokenDistributionComparator,
    confusion_matrix_top_k,
)
from .core.data import TextListBatchLoader, WikitextBatchLoader

logger = logging.getLogger(__name__)

VARIATIONS = ("svd", "orthogonal", "random")
INTERVENTIONS = ("project_out", "inject", "subtract")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_model_device(model: torch.nn.Module) -> torch.device:
    """Get the first non-meta device from model parameters."""
    for param in model.parameters():
        if not param.is_meta:
            return param.device
    logger.warning("All parameters on meta device, defaulting to CPU")
    return torch.device("cpu")

def _make_batches(config: ExperimentConfig, tokenizer):
    if config.dataset == "wikitext":
        return WikitextBatchLoader(tokenizer, config.num_samples, config.seq_len, config.batch_size).get_batches()
    if config.dataset == "text":
        if not config.text_file:
            raise ValueError("dataset='text' requires text_file")
        texts = [t for t in Path(config.text_file).read_text().splitlines() if t.strip()]
        return TextListBatchLoader(tokenizer, texts, config.seq_len, config.batch_size).get_batches()
    raise ValueError(f"Unknown dataset: {config.dataset}")


def _get_direction(variant: str, v_svd, exp_idx: int, seed: int):
    """Get direction vector(s) for intervention."""
    if variant == "svd":
        return v_svd
    dim = int(v_svd.shape[-1])
    s = seed + exp_idx
    if variant == "orthogonal":
        subspace = v_svd.unsqueeze(0) if v_svd.ndim == 1 else v_svd
        return VectorIntervention.make_orthogonal(subspace[0], seed=s)
    if variant == "random":
        return VectorIntervention.make_random(dim, seed=s)
    raise ValueError(f"Unknown variant: {variant}")


def _apply_intervention(kind: str, row, direction, scale: float):
    if kind == "project_out":
        return VectorIntervention.project_out(row, direction)
    if kind == "inject":
        return VectorIntervention.inject(row, direction, scale=scale)
    if kind == "subtract":
        return VectorIntervention.subtract(row, direction, scale=scale)
    raise ValueError(f"Unknown intervention: {kind}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    config: ExperimentConfig,
    *,
    interventions: Sequence[str] = ("project_out", "inject", "subtract"),
    inject_subtract_scale: float = 1.0,
    distribution_metric: str = "kl",
    confusion_top_k: int = 2,
) -> Dict[str, Any]:
    """Run inject/subtract/project_out interventions with loss + token distributions."""
    # Pass layer_idx as priority layer - ensures it's on GPU
    loader = ModelLoader(config, priority_layers=[config.layer_idx], max_gpu_layers=20)
    tokenizer = loader.load_tokenizer()
    model = loader.load_model()

    batches = _make_batches(config, tokenizer)

    with torch.no_grad():
        model.eval()
        device = _get_model_device(model)
        _ = model(batches[0].to(device))

    expert_vecs = ExpertVectors(
        config.svd_dir, config.layer_idx,
        config.num_experts, config.model_tag,
        top_k=config.top_k[0] if config.top_k else 1,
    )
    expert_vecs.load()
    if len(expert_vecs) == 0:
        raise RuntimeError("No SVD vectors loaded. Check svd_dir and file names.")

    evaluator = LossEvaluator(model)
    results: Dict[str, Any] = {
        "config": {
            **vars(config),
            "interventions": list(interventions),
            "inject_subtract_scale": inject_subtract_scale,
            "distribution_metric": distribution_metric,
            "confusion_top_k": confusion_top_k,
        },
        "results": {},
    }

    with RouterManager(model, config.layer_idx) as router:
        logger.info("Evaluating baseline...")
        base_loss = evaluator.evaluate(batches)
        results["baseline_loss"] = base_loss
        logger.info("Baseline loss: %.4f", base_loss)

        baseline_logits = evaluator.get_logits(batches)
        logger.info("Baseline logits shape: %s", tuple(baseline_logits.shape))
        dist_comp = TokenDistributionComparator(metric=distribution_metric)

        for variant in config.variations:
            for kind in interventions:
                key = f"{variant}_{kind}"
                logger.info("--- %s ---", key.replace("_", " ").title())
                modified = router.original_weights.clone()

                for exp_idx, v_svd in expert_vecs.items():
                    direction = _get_direction(variant, v_svd, exp_idx, config.seed)
                    modified[exp_idx] = _apply_intervention(kind, modified[exp_idx], direction, inject_subtract_scale)

                router.apply_weights(modified)
                loss = evaluator.evaluate(batches)
                delta = loss - base_loss
                modified_logits = evaluator.get_logits(batches)
                cm = dist_comp.compare(baseline_logits, modified_logits)
                conf_matrix, conf_token_ids = confusion_matrix_top_k(
                    baseline_logits, modified_logits, top_k=confusion_top_k
                )
                results["results"][key] = {
                    "loss": loss,
                    "delta": delta,
                    "distribution_metric": cm,
                    "distribution_metric_name": distribution_metric,
                    "confusion_matrix": conf_matrix.tolist(),
                    "confusion_token_ids": conf_token_ids,
                }
                logger.info("Result [%s]: loss=%.4f, delta=%+.4f, %s=%.4f",
                            key, loss, delta, distribution_metric, cm)

    q = (config.quantization or "none").lower()
    name = f"vector_interventions_L{config.layer_idx}_{config.dataset}_q{q}.json"
    out_path = Path(config.output_dir) / name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving results to %s", out_path)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    p = argparse.ArgumentParser(description="Vector interventions: inject, subtract, project_out")
    p.add_argument("--svd-dir", required=True, help="Directory with expert SVD pickle files")
    p.add_argument("--layer-idx", type=int, required=True, help="MoE layer index")
    p.add_argument("--output-dir", default="results", help="Folder for result JSON files")
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--model-id", default="mistralai/Mixtral-8x7B-v0.1")
    p.add_argument("--model-tag", default="mistralai_Mixtral_8x7B_v0.1")
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variations", default="svd,orthogonal,random",
                   help="Comma-separated: svd,orthogonal,random")
    p.add_argument("--interventions", default="project_out,inject,subtract",
                   help="Comma-separated: project_out,inject,subtract")
    p.add_argument("--inject-subtract-scale", type=float, default=1.0)
    p.add_argument("--distribution-metric", default="kl", choices=("kl", "ce"))
    p.add_argument("--confusion-top-k", type=int, default=2)
    p.add_argument("--dataset", default="wikitext", choices=("wikitext", "text"))
    p.add_argument("--text-file", default=None)
    p.add_argument("--top-k", default="1")
    p.add_argument("--quantization", default=None, choices=("8bit", "4bit"))

    args = p.parse_args()

    variations = [v.strip() for v in args.variations.split(",") if v.strip()]
    interventions = [i.strip() for i in args.interventions.split(",") if i.strip()]
    top_k = [int(x) for x in args.top_k.split(",") if x.strip()] or [1]
    if args.dataset == "text" and not args.text_file:
        p.error("--dataset=text requires --text-file")

    config = ExperimentConfig(
        svd_dir=args.svd_dir,
        layer_idx=args.layer_idx,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        model_id=args.model_id,
        model_tag=args.model_tag,
        num_experts=args.num_experts,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        variations=variations,
        dataset=args.dataset,
        text_file=args.text_file,
        top_k=top_k,
        quantization=args.quantization,
    )

    run_experiment(
        config,
        interventions=interventions,
        inject_subtract_scale=args.inject_subtract_scale,
        distribution_metric=args.distribution_metric,
        confusion_top_k=args.confusion_top_k,
    )


if __name__ == "__main__":
    main()
