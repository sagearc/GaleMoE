"""
Cross-expert interventions: hijack or transplant SVD vectors between experts.

Two modes:
1. Hijack: Disable source expert, boost all others
   - Source: project_out (disable)
   - All others: inject (steal tokens)
   
2. Transplant: Disable source, boost specific target, block others
   - Source: project_out (disable)
   - Target: inject (receive tokens)
   - Others: project_out (reject tokens)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

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


def _vectors_for_k(v_full: torch.Tensor, k: int) -> torch.Tensor:
    """Select top-k SVD vectors from full loaded set."""
    v = v_full[:k]
    return v


def _get_direction(variant: str, v_svd, exp_idx: int, seed: int):
    """Get direction vector(s) for intervention."""
    if variant == "svd":
        return v_svd
    s = seed + exp_idx
    v1 = v_svd[0] if v_svd.ndim == 2 else v_svd
    if variant == "orthogonal":
        return VectorIntervention.make_orthogonal(v1, s)
    if variant == "random":
        # Random direction in the same SVD rank (span of v_svd), not full R^dim
        return VectorIntervention.make_random_in_span(v_svd, s)
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
    source_expert: int,
    mode: str,
    transplant_target: int = None,
    inject_subtract_scale: float = 1.0,
    distribution_metric: str = "kl",
    confusion_top_k: int = 2,
) -> Dict[str, Any]:
    """
    Cross-expert interventions: hijack or transplant SVD vectors between experts.
    
    Args:
        source_expert: Expert whose SVD vectors to use
        mode: 'hijack' or 'transplant'
        transplant_target: For transplant mode, which expert receives the tokens (required for transplant)
    
    Modes:
        - hijack: Disable source (project_out), boost all others (inject)
        - transplant: Disable source (project_out), boost target (inject), block others (project_out)
    """
    if mode == "transplant" and transplant_target is None:
        raise ValueError("transplant mode requires --transplant-target")
    if mode == "hijack" and transplant_target is not None:
        raise ValueError("hijack mode does not use --transplant-target")
    # Pass layer_idx as priority layer - ensures it's on GPU
    # max_gpu_layers=12 leaves ~45GB headroom for activations (avoids OOM)
    loader = ModelLoader(config, priority_layers=[config.layer_idx], max_gpu_layers=12)
    tokenizer = loader.load_tokenizer()
    model = loader.load_model()

    batches = _make_batches(config, tokenizer)

    with torch.no_grad():
        model.eval()
        device = _get_model_device(model)
        _ = model(batches[0].to(device))

    k_values = list(config.top_k) if config.top_k else [1]
    
    # Load expert vectors with max k
    expert_vecs = ExpertVectors(
        config.svd_dir, config.layer_idx,
        config.num_experts, config.model_tag,
        top_k=max(k_values),
    )
    expert_vecs.load()
    if len(expert_vecs) == 0:
        raise RuntimeError("No SVD vectors loaded. Check svd_dir and file names.")
    
    if source_expert not in expert_vecs:
        raise ValueError(f"Source expert {source_expert} not found in loaded SVD vectors")

    evaluator = LossEvaluator(model)
    
    # Determine expert groups based on mode
    all_experts = list(range(config.num_experts))
    if mode == "hijack":
        inject_experts = [e for e in all_experts if e != source_expert]
        block_experts = []
        mode_str = "hijack"
    else:  # transplant
        inject_experts = [transplant_target]
        block_experts = [e for e in all_experts if e != source_expert and e != transplant_target]
        mode_str = f"transplant_to{transplant_target}"
    
    results: Dict[str, Any] = {
        "config": {
            **vars(config),
            "mode": mode,
            "source_expert": source_expert,
            "transplant_target": transplant_target,
            "inject_experts": inject_experts,
            "block_experts": block_experts,
            "inject_subtract_scale": inject_subtract_scale,
            "distribution_metric": distribution_metric,
            "confusion_top_k": confusion_top_k,
            "k_values": k_values,
        },
        "by_k": {},
    }

    with RouterManager(model, config.layer_idx) as router:
        logger.info("Evaluating baseline...")
        base_loss = evaluator.evaluate(batches)
        results["baseline_loss"] = base_loss
        logger.info("Baseline loss: %.4f", base_loss)

        baseline_logits = evaluator.get_logits(batches)
        logger.info("Baseline logits shape: %s", tuple(baseline_logits.shape))
        dist_comp = TokenDistributionComparator(metric=distribution_metric)

        # Get source expert's full vector (max k)
        source_vec_full = expert_vecs[source_expert]
        logger.info("Source expert %d (loaded %d vectors)", 
                   source_expert, source_vec_full.shape[0] if source_vec_full.ndim > 1 else 1)
        logger.info("Mode: %s", mode)
        logger.info("  - Disable source: %d (project_out)", source_expert)
        logger.info("  - Inject to: %s", inject_experts)
        if block_experts:
            logger.info("  - Block: %s (project_out)", block_experts)

        # Iterate over k values (number of top SVD vectors to use)
        for top_k in k_values:
            logger.info("\n--- top_k=%d ---", top_k)
            k_results: Dict[str, Any] = {}
            
            # Crop source vector to top_k
            source_vec = _vectors_for_k(source_vec_full, top_k)

            for variant in config.variations:
                key = f"{variant}_{mode_str}"
                logger.info("Computing %s (k=%d)", key.upper(), top_k)
                modified = router.original_weights.clone()

                # Get direction from source expert (with k vectors)
                direction = _get_direction(variant, source_vec, source_expert, config.seed)

                # 1. Disable source expert (project_out)
                modified[source_expert] = VectorIntervention.project_out(
                    modified[source_expert], direction, scale=inject_subtract_scale
                )

                # 2. Inject into target expert(s)
                for expert_idx in inject_experts:
                    modified[expert_idx] = VectorIntervention.inject(
                        modified[expert_idx], direction, scale=inject_subtract_scale
                    )

                # 3. Block other experts (transplant mode only)
                for expert_idx in block_experts:
                    modified[expert_idx] = VectorIntervention.project_out(
                        modified[expert_idx], direction, scale=inject_subtract_scale
                    )

                router.apply_weights(modified)
                loss = evaluator.evaluate(batches)
                delta = loss - base_loss
                modified_logits = evaluator.get_logits(batches)
                cm = dist_comp.compare(baseline_logits, modified_logits)
                conf_matrix, conf_token_ids = confusion_matrix_top_k(
                    baseline_logits, modified_logits, top_k=confusion_top_k
                )
                k_results[key] = {
                    "loss": loss,
                    "delta": delta,
                    "distribution_metric": cm,
                    "distribution_metric_name": distribution_metric,
                    "confusion_matrix": conf_matrix.tolist(),
                    "confusion_token_ids": conf_token_ids,
                }
                logger.info("Result [%s, k=%d]: loss=%.4f, delta=%+.4f, %s=%.4f",
                            key, top_k, loss, delta, distribution_metric, cm)
                router.restore()
            
            results["by_k"][top_k] = k_results

    # Log summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: %s mode, source expert %d", mode.upper(), source_expert)
    logger.info("=" * 70)
    logger.info("Baseline loss: %.4f", base_loss)
    
    for k in k_values:
        logger.info("\n--- k=%d ---", k)
        for variant_key, data in results["by_k"][k].items():
            logger.info("  %s: loss=%.4f, Δ=%+.4f", 
                       variant_key, data["loss"], data["delta"])

    q = (config.quantization or "none").lower()
    k_part = "-".join(str(k) for k in k_values)
    scale_str = f"_S{inject_subtract_scale:.1f}" if inject_subtract_scale != 1.0 else ""
    if mode == "hijack":
        name = f"hijack_L{config.layer_idx}_Src{source_expert}_k{k_part}{scale_str}_{config.dataset}_q{q}.json"
    else:
        name = f"transplant_L{config.layer_idx}_Src{source_expert}_To{transplant_target}_k{k_part}{scale_str}_{config.dataset}_q{q}.json"
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

    p = argparse.ArgumentParser(
        description="Cross-expert interventions: hijack or transplant SVD vectors between experts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hijack: Disable expert 5, boost all others (tokens stolen by everyone)
  python -m src.experiments.router_interventions.run_vector_interventions \\
    --svd-dir svd_cache/w3 --layer-idx 5 \\
    --mode hijack --source-expert 5 \\
    --inject-subtract-scale 3.0 --top-k 1,2,4,8

  # Transplant: Disable expert 5, boost expert 7, block others (tokens go to expert 7)
  python -m src.experiments.router_interventions.run_vector_interventions \\
    --svd-dir svd_cache/w3 --layer-idx 5 \\
    --mode transplant --source-expert 5 --transplant-target 7 \\
    --inject-subtract-scale 3.0 --top-k 1,2,4,8
        """
    )
    p.add_argument("--svd-dir", required=True, help="Directory with expert SVD pickle files")
    p.add_argument("--layer-idx", type=int, required=True, help="MoE layer index")
    p.add_argument("--mode", required=True, choices=["hijack", "transplant"],
                   help="hijack: disable source + boost all others; transplant: disable source + boost target + block others")
    p.add_argument("--source-expert", type=int, required=True, help="Source expert to disable")
    p.add_argument("--transplant-target", type=int, default=None,
                   help="Target expert to boost (required for transplant mode)")
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
    p.add_argument("--inject-subtract-scale", type=float, default=3.0,
                   help="Scale for inject/project_out (default 3.0 for strong effects)")
    p.add_argument("--distribution-metric", default="kl", choices=("kl", "ce"))
    p.add_argument("--confusion-top-k", type=int, default=2)
    p.add_argument("--dataset", default="wikitext", choices=("wikitext", "text"))
    p.add_argument("--text-file", default=None)
    p.add_argument("--top-k", default="1,2,4,8", help="Comma-separated k values for SVD vectors")
    p.add_argument("--quantization", default=None, choices=("8bit", "4bit"))

    args = p.parse_args()

    if args.mode == "transplant" and args.transplant_target is None:
        p.error("--mode transplant requires --transplant-target")
    if args.mode == "hijack" and args.transplant_target is not None:
        p.error("--mode hijack does not use --transplant-target")
    if args.transplant_target == args.source_expert:
        p.error("transplant-target cannot be the same as source-expert")
    
    logger.info("Mode: %s", args.mode)
    logger.info("Source expert: %d", args.source_expert)
    if args.mode == "transplant":
        logger.info("Transplant target: %d", args.transplant_target)

    variations = [v.strip() for v in args.variations.split(",") if v.strip()]
    top_k = [int(x.strip()) for x in args.top_k.split(",") if x.strip()] or [1]
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
        source_expert=args.source_expert,
        mode=args.mode,
        transplant_target=args.transplant_target,
        inject_subtract_scale=args.inject_subtract_scale,
        distribution_metric=args.distribution_metric,
        confusion_top_k=args.confusion_top_k,
    )


if __name__ == "__main__":
    main()
