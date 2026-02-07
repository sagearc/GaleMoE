"""Plotting for router subspace ablation results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt


def load_results(filepath: str | Path) -> Dict[str, Any]:
    """Load ablation results from a JSON file.

    Expected keys: ``config``, ``baseline_loss``, ``results`` (variant -> {loss, delta}).
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
    figsize: tuple[float, float] = (10, 4),
    show_baseline: bool = True,
) -> plt.Figure:
    """Plot loss and delta by variant (bar charts).

    Args:
        results: Dict with ``baseline_loss`` and ``results`` (variant -> {loss, delta}).
        save_path: If set, save figure to this path.
        figsize: (width, height) for the figure.
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

    fig, (ax_loss, ax_delta) = plt.subplots(1, 2, figsize=figsize)

    x = range(len(variants))
    loss_colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22"]
    ax_loss.bar(
        x, losses,
        color=[loss_colors[i % len(loss_colors)] for i in range(len(variants))],
        edgecolor="black", linewidth=0.8,
    )
    ax_loss.set_xticks(x)
    ax_loss.set_xticklabels([v.capitalize() for v in variants])
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss by variant")
    if show_baseline and baseline is not None:
        ax_loss.axhline(y=baseline, color="gray", linestyle="--", label="Baseline")
        ax_loss.legend()
    ax_loss.grid(axis="y", alpha=0.3)

    colors_delta = ["#e74c3c" if d >= 0 else "#27ae60" for d in deltas]
    bars_delta = ax_delta.bar(x, deltas, color=colors_delta, edgecolor="black", linewidth=0.8)
    ax_delta.axhline(y=0, color="black", linewidth=0.5)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels([v.capitalize() for v in variants])
    ax_delta.set_ylabel("Δ Loss (vs baseline)")
    ax_delta.set_title("Loss change by variant")
    ax_delta.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(Path(save_path), bbox_inches="tight", dpi=150)
    return fig
