"""Compute SVD of each expert's w1 and save the full Vh basis for project-out on the router.

We compute SVD of the **expert** weight matrix (w1, first linear of the expert MLP).
w1 has shape [d_ff, hidden_dim] (e.g. 14336 x 4096). SVD gives Vh of shape
(min, hidden_dim) = (4096, 4096) — the full basis of 4096 right singular vectors
(4096 singular values; each row of Vh is one vector of length 4096). We save the
entire Vh. run_project_out then uses the first top_k rows to **project out** from
the **router (gate) vector**.

Works for float or quantized models (--quantization 8bit/4bit or omit for float).
Output is always float. Files: {cache_dir}/float/ or {cache_dir}/8bit/,
{model_tag}_layer{L}_expert{i}.pkl as raw tensor V (4096, 4096), columns = vectors. Use the printed path
as --svd_dir in run_project_out (with matching --quantization).
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import torch

from .core import ExperimentConfig
from .core.model_loader import ModelLoader
from .core.router_manager import _dequantize_int8_weight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Subdir name under cache_dir for each quantization (must match what run_project_out expects to pass as svd_dir)
QUANT_SUBDIR = {"8bit": "8bit", "4bit": "4bit", None: "float"}


def _linear_weight_float(linear: Any, quantized: bool) -> torch.Tensor:
    """Return a Linear's weight in float (dequantize if int8)."""
    w = getattr(linear.weight, "data", linear.weight)
    if quantized and w.dtype == torch.int8:
        return _dequantize_int8_weight(linear.weight)
    return w.float()


def get_expert_w1_float(model: Any, layer_idx: int, expert_idx: int, quantized: bool) -> torch.Tensor:
    """Return expert's w1 weight matrix [d_ff, hidden_dim] in float."""
    expert = model.model.layers[layer_idx].block_sparse_moe.experts[expert_idx]
    return _linear_weight_float(expert.w1, quantized)


def run_svd_from_expert(
    model_id: str,
    model_tag: str,
    cache_dir: str,
    layer_indices: list[int],
    num_experts: int = 8,
    quantization: str | None = None,
) -> str:
    """Compute SVD of each expert's w1, save top right singular vector as Vh [1, dim].
    run_project_out uses these to project out from the router (gate) vector.
    """
    quantized = quantization in ("8bit", "4bit")
    if quantization == "4bit":
        logger.warning("4bit gate dequant may be approximate; 8bit is supported.")

    subdir = QUANT_SUBDIR.get(quantization, "float")
    out_path = Path(cache_dir).resolve() / subdir
    out_path.mkdir(parents=True, exist_ok=True)
    svd_dir = str(out_path)

    config = ExperimentConfig(
        svd_dir="",
        layer_idx=layer_indices[0],
        model_id=model_id,
        model_tag=model_tag,
        num_experts=num_experts,
        quantization=quantization,
    )
    loader = ModelLoader(config)
    logger.info(
        "Loading tokenizer and model (%s)...",
        quantization if quantization else "float",
    )
    tokenizer = loader.load_tokenizer()
    model = loader.load_model()

    try:
        device = next(model.parameters()).device
        if device.type == "cuda":
            with torch.no_grad():
                model.eval()
                dummy = torch.zeros(1, 4, dtype=torch.long, device=device)
                _ = model(dummy)
    except Exception as e:
        logger.warning("Skip materialization forward: %s", e)

    for layer_idx in layer_indices:
        logger.info("Layer %d: computing SVD of expert w1...", layer_idx)
        for expert_idx in range(num_experts):
            w1 = get_expert_w1_float(model, layer_idx, expert_idx, quantized=quantized)
            w1 = w1.cpu().float()
            # w1 shape [d_ff, hidden_dim]. SVD: w1 = U @ diag(S) @ Vh
            # Save V with columns = right singular vectors (loader treats square as cols=vectors)
            U, S, Vh = torch.linalg.svd(w1, full_matrices=False)
            V = Vh.T  # (4096, 4096), columns = vectors
            fname = f"{model_tag}_layer{layer_idx}_expert{expert_idx}.pkl"
            path = out_path / fname
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(V.numpy().astype("float32"), f)
        logger.info("Layer %d: saved %d expert matrices V (4096, 4096, cols=vectors) to %s", layer_idx, num_experts, out_path)

    logger.info(
        "Use this path as --svd_dir when running run_project_out (with matching --quantization): %s",
        svd_dir,
    )
    return svd_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute SVD of each expert's w1; save top singular direction. "
        "run_project_out then projects out that direction from the router (gate) vector.",
    )
    parser.add_argument("--model-id", type=str, default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--model-tag", type=str, default="mistralai_Mixtral_8x7B_v0.1")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="svd_cache",
        help="Cache base directory. Files go to cache_dir/float/ or cache_dir/8bit/.",
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        nargs="+",
        required=True,
        help="Layer index (or indices) to export, e.g. 0 5 10 15.",
    )
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["8bit", "4bit"],
        help="Model quantization. Omit for float model.",
    )
    args = parser.parse_args()

    run_svd_from_expert(
        model_id=args.model_id,
        model_tag=args.model_tag,
        cache_dir=args.cache_dir,
        layer_indices=args.layer_idx,
        num_experts=args.num_experts,
        quantization=args.quantization,
    )


if __name__ == "__main__":
    main()
