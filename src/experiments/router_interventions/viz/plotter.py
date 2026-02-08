"""Plotting for router intervention results.

Helpers
-------
load_results       – load a single JSON result file
load_results_dir   – load all result JSONs from a folder

Plots
-----
plot_delta_vs_k      – loss delta vs k for one layer (one line per variant)
plot_delta_vs_layers – loss delta vs layer for one k (one line per variant)
plot_ablation_results – bar chart of loss / delta for a single run
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_results(filepath: str | Path) -> Dict[str, Any]:
    """Load a single results JSON."""
    with open(filepath) as f:
        data = json.load(f)
    
    # Backward compatibility: if using old format (k-independent stored redundantly),
    # normalize to new format with k_independent section
    if "k_independent" not in data and "by_k" in data:
        # Extract k-independent variants from the first k entry
        first_k = next(iter(data["by_k"].values()), {})
        k_independent = {}
        for variant in ["zero", "shuffle", "random", "orthogonal"]:
            if variant in first_k:
                k_independent[variant] = first_k[variant]
        if k_independent:
            data["k_independent"] = k_independent
            # Remove from all by_k entries to clean up
            for k_entry in data["by_k"].values():
                for variant in k_independent:
                    k_entry.pop(variant, None)
    
    return data


def load_results_dir(results_dir: str | Path, glob: str = "project_out_L*.json") -> List[Dict[str, Any]]:
    """Load all matching JSONs from *results_dir*, sorted by layer index."""
    paths = sorted(Path(results_dir).glob(glob))
    if not paths:
        raise FileNotFoundError(f"No files matching {glob!r} in {results_dir}")
    out = [load_results(p) for p in paths]
    out.sort(key=lambda r: r.get("config", {}).get("layer_idx", 0))
    return out


# ---------------------------------------------------------------------------
# delta vs k  (single layer, multiple k)
# ---------------------------------------------------------------------------

def plot_delta_vs_k(
    results: Dict[str, Any],
    *,
    variants: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8, 5),
    ylim: Optional[tuple[float, float]] = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """One line per variant, x = k, y = loss delta.  Needs ``by_k`` in *results*.
    
    Args:
        ylim: Optional y-axis limits as (ymin, ymax). Use to zoom into smaller deltas.
    """
    by_k = results.get("by_k", {})
    if not by_k:
        raise ValueError("No 'by_k' in results. Run with multiple --top-k values.")
    layer = results.get("config", {}).get("layer_idx", "?")
    k_independent = results.get("k_independent", {})

    k_vals = sorted(int(k) for k in by_k)
    
    # Merge k_independent variants into all k values for plotting
    merged_by_k = {}
    for k in k_vals:
        k_str = str(k)
        merged_by_k[k] = {**k_independent, **by_k.get(k_str, by_k.get(k, {}))}
    
    # Get all variants from merged data
    first = merged_by_k[k_vals[0]]
    all_variants = sorted(v for v in first if isinstance(first.get(v), dict) and "delta" in first[v])
    variants = variants or all_variants

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(variants), 1)))
    for i, var in enumerate(variants):
        deltas = []
        for k in k_vals:
            entry = merged_by_k.get(k, {})
            d = entry.get(var, {}).get("delta")
            deltas.append(d)
        if all(d is not None for d in deltas):
            ax.plot(k_vals, deltas, "o-", label=var, color=colors[i])
    ax.set_xlabel("k (singular vectors projected out)")
    ax.set_ylabel("Loss delta")
    ax.set_title(title or f"Loss delta vs k  (layer {layer})")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# delta vs layer  (single k, multiple layers)
# ---------------------------------------------------------------------------

def plot_delta_vs_layers(
    results_list: List[Dict[str, Any]],
    k: int,
    *,
    variants: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (10, 5),
    ylim: Optional[tuple[float, float]] = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """One line per variant, x = layer, y = loss delta for a fixed *k*.
    
    Args:
        ylim: Optional y-axis limits as (ymin, ymax). Use to zoom into smaller deltas.
    """
    by_layer: Dict[int, Dict[str, float]] = {}
    for res in results_list:
        layer = res.get("config", {}).get("layer_idx")
        if layer is None:
            continue
        
        # Merge k_independent and by_k[k] data
        k_independent = res.get("k_independent", {})
        by_k = res.get("by_k", {})
        k_entry = by_k.get(str(k), by_k.get(k, {}))
        
        # Combine k-independent and k-dependent data
        merged_entry = {**k_independent, **k_entry}
        if not merged_entry:
            continue
            
        by_layer[layer] = {v: merged_entry[v]["delta"] for v in merged_entry
                           if isinstance(merged_entry.get(v), dict) and "delta" in merged_entry[v]}
    if not by_layer:
        raise ValueError(f"No results with by_k[{k}]")

    layers = sorted(by_layer)
    all_vars = sorted({v for d in by_layer.values() for v in d})
    variants = variants or all_vars

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(variants), 1)))
    for i, var in enumerate(variants):
        deltas = [by_layer[l].get(var) for l in layers]
        if any(d is None for d in deltas):
            continue
        ax.plot(layers, deltas, "o-", label=var, color=colors[i])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Loss delta")
    ax.set_title(title or f"Loss delta vs layer  (k={k})")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(layers)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Ablation bar chart (single run)
# ---------------------------------------------------------------------------

def plot_ablation_results(
    results: Dict[str, Any],
    *,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Bar chart of loss + delta for each variant in a single-k run."""
    baseline = results.get("baseline_loss")
    by_k = results.get("by_k", {})
    # Pick first k
    if by_k:
        first_k = sorted(int(k) for k in by_k)[0]
        res = by_k.get(str(first_k), by_k.get(first_k, {}))
    else:
        res = results.get("results", {})
    if not res:
        raise ValueError("Empty results")

    variants = list(res.keys())
    losses = [res[v]["loss"] for v in variants]
    deltas = [res[v]["delta"] for v in variants]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    x = range(len(variants))
    palette = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]

    ax1.bar(x, losses, color=[palette[i % len(palette)] for i in x], edgecolor="k", lw=0.6)
    if baseline is not None:
        ax1.axhline(baseline, color="gray", ls="--", label="baseline")
        ax1.legend()
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, rotation=45, ha="right")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss by variant")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, deltas, color=["#e74c3c" if d >= 0 else "#27ae60" for d in deltas], edgecolor="k", lw=0.6)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(variants, rotation=45, ha="right")
    ax2.set_ylabel("Loss delta")
    ax2.set_title("Delta vs baseline")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Confusion heatmap (for vector interventions)
# ---------------------------------------------------------------------------

def plot_confusion_heatmap(
    matrix,
    *,
    token_ids: Optional[List[int]] = None,
    token_labels: Optional[List[str]] = None,
    figsize: tuple[float, float] = (8, 7),
    title: str = "Prediction before vs after",
    cmap: str = "Blues",
    save_path: str | Path | None = None,
) -> plt.Figure:
    mat = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mat, cmap=cmap, aspect="auto")
    k = mat.shape[0]
    labels = token_labels or ([str(t) for t in token_ids] if token_ids else [str(i) for i in range(k)])
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(labels[:k], rotation=45, ha="right")
    ax.set_yticklabels(labels[:k])
    ax.set_xlabel("Predicted after"); ax.set_ylabel("Predicted before")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_confusion_from_results(
    results: Dict[str, Any],
    variant_key: str,
    **kwargs,
) -> plt.Figure:
    """Plot confusion matrix stored in ``results["results"][variant_key]``."""
    entry = results.get("results", {}).get(variant_key, {})
    if "confusion_matrix" not in entry:
        raise ValueError(f"No confusion_matrix for {variant_key!r}")
    return plot_confusion_heatmap(
        entry["confusion_matrix"],
        token_ids=entry.get("confusion_token_ids"),
        title=f"Top-1 predictions: {variant_key.replace('_', ' ')}",
        **kwargs,
    )
