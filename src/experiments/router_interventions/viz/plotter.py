"""Plotting for router intervention results (bar charts and confusion matrices)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_results(filepath: str | Path) -> Dict[str, Any]:
    """Load results from a JSON file.

    Expected keys: ``config``, ``baseline_loss``, ``results`` (variant -> {loss, delta, ...}).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path) as f:
        return json.load(f)


def plot_ablation_results(
    results: Dict[str, Any],
    *,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    show_baseline: bool = True,
) -> plt.Figure:
    """Plot loss, delta and (if present) distribution metric by variant (bar charts).

    Args:
        results: Dict with ``baseline_loss`` and ``results`` (variant -> {loss, delta, ...}).
                 If variants include ``distribution_metric``, a third subplot is added.
        save_path: If set, save figure to this path.
        figsize: (width, height) for the figure. Default (10, 4) for 2 plots, (14, 4) for 3.
        show_baseline: Include baseline in the loss plot.

    Returns:
        The matplotlib Figure.
    """
    baseline = results.get("baseline_loss")
    res = results.get("results", {})
    if not res:
        raise ValueError("results['results'] is empty or missing")

    variants = list(res.keys())
    losses = [res[v]["loss"] for v in variants]
    deltas = [res[v]["delta"] for v in variants]
    has_dist = "distribution_metric" in res.get(variants[0], {})

    n_plots = 3 if has_dist else 2
    if figsize is None:
        figsize = (14, 4) if has_dist else (10, 4)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 2:
        ax_loss, ax_delta = axes
    else:
        ax_loss, ax_delta, ax_dist = axes

    x = range(len(variants))
    loss_colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22"]
    ax_loss.bar(
        x, losses,
        color=[loss_colors[i % len(loss_colors)] for i in range(len(variants))],
        edgecolor="black", linewidth=0.8,
    )
    ax_loss.set_xticks(x)
    ax_loss.set_xticklabels(
        [v.replace("_", " ").title() for v in variants],
        rotation=45,
        ha="right",
    )
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss by variant")
    if show_baseline and baseline is not None:
        ax_loss.axhline(y=baseline, color="gray", linestyle="--", label="Baseline")
        ax_loss.legend()
    ax_loss.grid(axis="y", alpha=0.3)

    colors_delta = ["#e74c3c" if d >= 0 else "#27ae60" for d in deltas]
    ax_delta.bar(x, deltas, color=colors_delta, edgecolor="black", linewidth=0.8)
    ax_delta.axhline(y=0, color="black", linewidth=0.5)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(
        [v.replace("_", " ").title() for v in variants],
        rotation=45,
        ha="right",
    )
    ax_delta.set_ylabel("Δ Loss (vs baseline)")
    ax_delta.set_title("Loss change by variant")
    ax_delta.grid(axis="y", alpha=0.3)

    if has_dist:
        dist_vals = [res[v]["distribution_metric"] for v in variants]
        dist_name = res[variants[0]].get("distribution_metric_name", "distribution")
        ax_dist.bar(x, dist_vals, color=loss_colors[: len(variants)], edgecolor="black", linewidth=0.8)
        ax_dist.set_xticks(x)
        ax_dist.set_xticklabels(
            [v.replace("_", " ").title() for v in variants],
            rotation=45,
            ha="right",
        )
        ax_dist.set_ylabel(dist_name.upper())
        ax_dist.set_title(f"Token distribution metric ({dist_name})")
        ax_dist.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(Path(save_path), bbox_inches="tight", dpi=150)
    return fig


def plot_confusion_heatmap(
    matrix: np.ndarray | List[List[float]],
    *,
    token_ids: Optional[List[int]] = None,
    token_labels: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (8, 7),
    title: str = "Prediction before vs after",
    cmap: str = "Blues",
) -> plt.Figure:
    """Plot a confusion matrix as a heatmap (rows = pred before, cols = pred after).

    Args:
        matrix: KxK count matrix (list of lists or ndarray).
        token_ids: Optional length-K list of token IDs (used as axis labels if token_labels not set).
        token_labels: Optional length-K list of decoded token strings for axis labels.
        ax: If given, draw on this axes; otherwise create a new figure.
        figsize: Used only when ax is None.
        title: Axes title.
        cmap: Matplotlib colormap name.

    Returns:
        The figure (existing or new).
    """
    mat = np.asarray(matrix, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    k = mat.shape[0]
    if token_labels is not None and len(token_labels) >= k:
        labels = token_labels[:k]
    elif token_ids is not None and len(token_ids) >= k:
        labels = [str(tid) for tid in token_ids[:k]]
    else:
        labels = [str(i) for i in range(k)]
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted after")
    ax.set_ylabel("Predicted before")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Count")
    return fig


def plot_delta_vs_layers(
    results_list: List[Dict[str, Any]],
    k: int,
    *,
    variants: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot loss delta vs layer index for a specific k (one line per variant).

    Use when you have one result file per layer (e.g. from multiple runs with --output-dir).
    Each result must have config.layer_idx and by_k (with key k).

    Args:
        results_list: List of loaded result dicts (one per layer).
        k: Which top_k value to use (must exist in each result's by_k).
        variants: Which variants to plot (default: all in by_k[k]).
        title: Plot title (default: "Loss Δ vs layer (k={k})").
        figsize: Figure size when ax is None.
        ax: Optional axes to plot on.

    Returns:
        The matplotlib Figure.
    """
    # Collect (layer_idx, variant -> delta) from each result
    by_layer: Dict[int, Dict[str, float]] = {}
    for res in results_list:
        cfg = res.get("config", {})
        layer_idx = cfg.get("layer_idx", len(by_layer))
        by_k = res.get("by_k", res.get("results", {}))
        # by_k keys can be int or str in JSON
        k_key = next((key for key in by_k if int(key) == k), None)
        if k_key is None:
            continue
        entry = by_k[k_key]
        if not isinstance(entry, dict):
            continue
        by_layer[layer_idx] = {
            v: entry[v]["delta"] for v in entry
            if isinstance(entry.get(v), dict) and "delta" in entry[v]
        }
    if not by_layer:
        raise ValueError(f"No results found with by_k[{k}]. Check that each file has by_k and key {k}.")
    layers = sorted(by_layer.keys())
    all_variants = set()
    for vmap in by_layer.values():
        all_variants.update(vmap.keys())
    if variants is None:
        variants = sorted(all_variants)
    else:
        variants = [v for v in variants if v in all_variants]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(variants), 1)))
    for i, var in enumerate(variants):
        deltas = [by_layer[l].get(var) for l in layers]
        if any(d is None for d in deltas):
            continue
        ax.plot(layers, deltas, "o-", label=var.replace("_", " ").title(), color=colors[i % len(colors)])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Loss Δ (vs baseline)")
    ax.set_title(title or f"Loss Δ vs layer (k={k})")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_delta_vs_k(
    results: Dict[str, Any],
    *,
    variants: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot loss delta vs k for a specific layer (one line per variant).

    Use when you have one result file with by_k (multiple k values for one layer).

    Args:
        results: Loaded result dict with by_k (keys = k values).
        variants: Which variants to plot (default: all in first by_k entry).
        title: Plot title (default: "Loss Δ vs k (layer {layer_idx})").
        figsize: Figure size when ax is None.
        ax: Optional axes to plot on.

    Returns:
        The matplotlib Figure.
    """
    by_k = results.get("by_k", {})
    if not by_k:
        raise ValueError("results has no 'by_k'. Run with multiple --top-k values.")
    layer_idx = results.get("config", {}).get("layer_idx", "?")
    # Sort k numerically (keys may be int or str)
    k_vals = sorted([int(key) for key in by_k])
    first_entry = by_k.get(str(k_vals[0]), by_k.get(k_vals[0], {}))
    if not isinstance(first_entry, dict):
        all_variants = []
    else:
        all_variants = sorted([
            v for v in first_entry
            if isinstance(first_entry.get(v), dict) and "delta" in first_entry[v]
        ])
    if variants is None:
        variants = all_variants
    else:
        variants = [v for v in variants if v in all_variants]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(variants), 1)))
    for i, var in enumerate(variants):
        deltas = []
        for k in k_vals:
            entry = by_k.get(str(k), by_k.get(k, {}))
            if isinstance(entry, dict) and var in entry and isinstance(entry[var], dict) and "delta" in entry[var]:
                deltas.append(entry[var]["delta"])
            else:
                deltas.append(None)
        if all(d is not None for d in deltas):
            ax.plot(k_vals, deltas, "o-", label=var.replace("_", " ").title(), color=colors[i % len(colors)])
    ax.set_xlabel("k (top singular vectors projected out)")
    ax.set_ylabel("Loss Δ (vs baseline)")
    ax.set_title(title or f"Loss Δ vs k (layer {layer_idx})")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confusion_from_results(
    results: Dict[str, Any],
    variant_key: str,
    *,
    ax: Optional[plt.Axes] = None,
    tokenizer: Any = None,
    top_k_display: int = 15,
    **kwargs: Any,
) -> plt.Figure:
    """Load confusion matrix for one variant from saved results and plot it.

    Reads ``confusion_matrix`` and ``confusion_token_ids`` from
    results["results"][variant_key] (e.g. from run_vector_interventions).

    Args:
        results: Full results dict from load_results(...).
        variant_key: Key in results["results"], e.g. "svd_inject", "orthogonal_subtract".
        ax: Optional axes to plot on.
        tokenizer: Optional HuggingFace tokenizer to decode token_ids to strings.
        top_k_display: Show only the first top_k_display rows/cols (readable labels).
        **kwargs: Passed to plot_confusion_heatmap (e.g. title, figsize).

    Returns:
        The figure.
    """
    res = results.get("results", {}).get(variant_key, {})
    if "confusion_matrix" not in res or "confusion_token_ids" not in res:
        raise ValueError(
            f"Results for {variant_key!r} do not contain confusion_matrix / confusion_token_ids. "
            "Run the vector intervention experiment with the current runner to generate them."
        )
    matrix = res["confusion_matrix"]
    token_ids = res["confusion_token_ids"]
    k = min(len(token_ids), top_k_display)
    matrix = [row[:k] for row in matrix[:k]]
    token_ids = token_ids[:k]
    token_labels = None
    if tokenizer is not None:
        try:
            token_labels = [tokenizer.decode([tid]) or str(tid) for tid in token_ids]
        except Exception:
            token_labels = [str(tid) for tid in token_ids]
    return plot_confusion_heatmap(
        matrix,
        token_ids=token_ids,
        token_labels=token_labels,
        ax=ax,
        title=f"Top-1 prediction: before vs after ({variant_key.replace('_', ' ')})",
        **kwargs,
    )
