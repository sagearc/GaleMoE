"""Visualization utilities for router-expert alignment analysis."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.alignment_analysis.base import AlignmentResult, AnalysisMetadata


# =============================================================================
# Style Configuration
# =============================================================================

@dataclass
class PlotStyle:
    """Configuration for plot styling."""
    # Colors (modern palette)
    colors: tuple = (
        "#2ecc71",  # green - alignment
        "#3498db",  # blue - z-score
        "#e74c3c",  # red - delta
        "#9b59b6",  # purple - effect
        "#f39c12",  # orange - shuffle
        "#1abc9c",  # teal
        "#e91e63",  # pink
        "#00bcd4",  # cyan
    )
    
    # Figure settings
    figure_dpi: int = 120
    title_size: int = 14
    label_size: int = 11
    tick_size: int = 10
    legend_size: int = 10
    
    # Grid
    grid_alpha: float = 0.3
    
    def apply(self) -> None:
        """Apply style settings to matplotlib."""
        plt.rcParams.update({
            "figure.dpi": self.figure_dpi,
            "axes.titlesize": self.title_size,
            "axes.labelsize": self.label_size,
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
            "legend.fontsize": self.legend_size,
            "axes.grid": True,
            "grid.alpha": self.grid_alpha,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })


STYLE = PlotStyle()


# =============================================================================
# Utility Functions
# =============================================================================

def get_layer_label(metadata: AnalysisMetadata) -> str:
    """Get a simple layer label for legends."""
    return metadata.layer_label


def extract_layer_number(layer_label: str) -> int:
    """Extract layer number from label like 'L5' -> 5."""
    return AnalysisMetadata.extract_layer_number(layer_label)


def get_summary_columns(df: pd.DataFrame) -> list:
    """Get standard summary columns for analysis."""
    cols = ["align", "delta_vs_shuffle", "z_vs_shuffle", "effect_over_random"]
    if "cos_squared" in df.columns and 1 in df["k"].values:
        cols.append("cos_squared")
    return cols


def display_plot(fig: plt.Figure, close: bool = True) -> None:
    """Display a matplotlib figure, handling Jupyter and regular environments."""
    try:
        from IPython import get_ipython
        from IPython.display import display, Image as IPImage
        
        if get_ipython() is not None:
            # In Jupyter - save to buffer and display
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=STYLE.figure_dpi, bbox_inches="tight")
            buf.seek(0)
            display(IPImage(data=buf.read()))
            buf.close()
        else:
            plt.show()
    except ImportError:
        plt.show()
    
    if close:
        plt.close(fig)


def _prepare_dataframe(
    results: Union[List[AlignmentResult], pd.DataFrame]
) -> pd.DataFrame:
    """Convert results to DataFrame if needed."""
    if isinstance(results, pd.DataFrame):
        return results
    return AlignmentResult.to_dataframe(results)


def _save_or_show(
    fig: plt.Figure,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Optional[Path]:
    """Save figure to file or display it."""
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=STYLE.figure_dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path
    elif show:
        plt.tight_layout()
        plt.show()
        plt.close(fig)
    return None


# =============================================================================
# Core Plotting Functions
# =============================================================================

def plot_alignment_vs_k(
    results: Union[List[AlignmentResult], pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    title: str = "Alignment vs k",
    **kwargs,
) -> plt.Axes:
    """Plot alignment score vs number of singular vectors (k).
    
    Args:
        results: Analysis results or DataFrame
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        **kwargs: Additional arguments passed to plot()
    
    Returns:
        Matplotlib axes
    """
    df = _prepare_dataframe(results)
    summary = df.groupby("k")["align"].mean()
    
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(
        summary.index, summary.values,
        marker="o", linewidth=2, markersize=6,
        color=kwargs.pop("color", STYLE.colors[0]),
        **kwargs
    )
    ax.set_xlabel("k (singular vectors)")
    ax.set_ylabel("Mean Alignment")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=STYLE.grid_alpha)
    
    return ax


def plot_zscore_vs_k(
    results: Union[List[AlignmentResult], pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    title: str = "Z-score vs k",
    **kwargs,
) -> plt.Axes:
    """Plot z-score vs k."""
    df = _prepare_dataframe(results)
    summary = df.groupby("k")["z_vs_shuffle"].mean()
    
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(
        summary.index, summary.values,
        marker="s", linewidth=2, markersize=6,
        color=kwargs.pop("color", STYLE.colors[1]),
        **kwargs
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=2, color="red", linestyle=":", alpha=0.3, label="p<0.05")
    ax.set_xlabel("k (singular vectors)")
    ax.set_ylabel("Mean Z-score")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=STYLE.grid_alpha)
    
    return ax


def plot_delta_vs_k(
    results: Union[List[AlignmentResult], pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    title: str = "Delta vs Shuffle",
    **kwargs,
) -> plt.Axes:
    """Plot delta (alignment - shuffle_mean) vs k."""
    df = _prepare_dataframe(results)
    summary = df.groupby("k")["delta_vs_shuffle"].mean()
    
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(
        summary.index, summary.values,
        marker="^", linewidth=2, markersize=6,
        color=kwargs.pop("color", STYLE.colors[2]),
        **kwargs
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("k (singular vectors)")
    ax.set_ylabel("Delta vs Shuffle")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=STYLE.grid_alpha)
    
    return ax


def plot_effect_vs_k(
    results: Union[List[AlignmentResult], pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    title: str = "Effect over Random",
    **kwargs,
) -> plt.Axes:
    """Plot effect over random baseline vs k."""
    df = _prepare_dataframe(results)
    summary = df.groupby("k")["effect_over_random"].mean()
    
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(
        summary.index, summary.values,
        marker="d", linewidth=2, markersize=6,
        color=kwargs.pop("color", STYLE.colors[3]),
        **kwargs
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("k (singular vectors)")
    ax.set_ylabel("Effect over Random")
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=STYLE.grid_alpha)
    
    return ax


def plot_expert_heatmap(
    results: Union[List[AlignmentResult], pd.DataFrame],
    metric: str = "align",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    center_zero: bool = False,
) -> plt.Axes:
    """Plot heatmap of metric across experts and k values.
    
    Args:
        results: Analysis results
        metric: Column to plot ('align', 'delta_vs_shuffle', 'z_vs_shuffle')
        ax: Matplotlib axes
        title: Plot title
        cmap: Colormap name
        center_zero: Center colormap at zero (for diverging data)
    """
    df = _prepare_dataframe(results)
    pivot = df.pivot_table(values=metric, index="expert", columns="k", aggfunc="mean")
    
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    # Set color limits
    if center_zero:
        vmax = np.abs(pivot.values).max()
        vmin = -vmax
    else:
        vmin, vmax = None, None
    
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Labels
    ax.set_xlabel("k")
    ax.set_ylabel("Expert")
    ax.set_title(title or f"{metric.replace('_', ' ').title()} by Expert")
    
    # Ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(k) for k in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([int(e) for e in pivot.index])
    
    plt.colorbar(im, ax=ax, label=metric.replace("_", " ").title())
    
    return ax


def plot_cos_squared_per_expert(
    results: Union[List[AlignmentResult], pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    title: str = "Cos²(θ) per Expert (k=1)",
) -> Optional[plt.Axes]:
    """Plot cos²(θ) values per expert for k=1."""
    df = _prepare_dataframe(results)
    
    if 1 not in df["k"].values:
        return None
    
    k1_data = df[df["k"] == 1].sort_values("expert")
    
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(
        k1_data["expert"], k1_data["cos_squared"],
        color=STYLE.colors[4], alpha=0.8, edgecolor="white"
    )
    
    # Add value labels on bars
    for bar, val in zip(bars, k1_data["cos_squared"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.3f}", ha="center", va="bottom", fontsize=9
        )
    
    ax.set_xlabel("Expert")
    ax.set_ylabel("Cos²(θ)")
    ax.set_title(title)
    ax.set_xticks(k1_data["expert"])
    ax.grid(True, alpha=STYLE.grid_alpha, axis="y")
    
    return ax


# =============================================================================
# Composite Plots
# =============================================================================

def plot_summary(
    results: Union[List[AlignmentResult], pd.DataFrame],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Optional[Path]:
    """Create a 2x2 summary plot with key metrics.
    
    Args:
        results: Analysis results
        title: Overall figure title
        save_path: Path to save figure (optional)
        show: Whether to display the plot
    
    Returns:
        Path to saved file if save_path provided, else None
    """
    STYLE.apply()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    plot_alignment_vs_k(results, ax=axes[0, 0])
    plot_zscore_vs_k(results, ax=axes[0, 1])
    plot_delta_vs_k(results, ax=axes[1, 0])
    plot_effect_vs_k(results, ax=axes[1, 1])
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    
    return _save_or_show(fig, save_path, show)


def plot_expert_analysis(
    results: Union[List[AlignmentResult], pd.DataFrame],
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Optional[Path]:
    """Create expert-focused analysis with heatmaps and per-expert bars.
    
    Args:
        results: Analysis results
        title: Overall figure title
        save_path: Path to save figure
        show: Whether to display
    
    Returns:
        Path to saved file if save_path provided
    """
    STYLE.apply()
    df = _prepare_dataframe(results)
    has_cos_squared = 1 in df["k"].values
    
    nrows = 2 if has_cos_squared else 1
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 5 * nrows))
    
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    # Row 1: Heatmaps
    plot_expert_heatmap(results, "align", ax=axes[0, 0], title="Alignment")
    plot_expert_heatmap(results, "delta_vs_shuffle", ax=axes[0, 1], 
                       title="Delta vs Shuffle", cmap="RdBu_r", center_zero=True)
    plot_expert_heatmap(results, "z_vs_shuffle", ax=axes[0, 2],
                       title="Z-score", cmap="coolwarm", center_zero=True)
    
    # Row 2: Per-expert bars (if cos_squared available)
    if has_cos_squared:
        plot_cos_squared_per_expert(results, ax=axes[1, 0])
        
        # Alignment per expert at representative k
        k_vals = sorted(df["k"].unique())
        k_repr = 128 if 128 in k_vals else k_vals[len(k_vals) // 2]
        k_data = df[df["k"] == k_repr].sort_values("expert")
        
        axes[1, 1].bar(k_data["expert"], k_data["align"], 
                      color=STYLE.colors[0], alpha=0.8, edgecolor="white")
        axes[1, 1].set_xlabel("Expert")
        axes[1, 1].set_ylabel(f"Alignment (k={k_repr})")
        axes[1, 1].set_title("Alignment per Expert")
        axes[1, 1].set_xticks(k_data["expert"])
        axes[1, 1].grid(True, alpha=STYLE.grid_alpha, axis="y")
        
        # Z-score per expert
        axes[1, 2].bar(k_data["expert"], k_data["z_vs_shuffle"],
                      color=STYLE.colors[1], alpha=0.8, edgecolor="white")
        axes[1, 2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1, 2].set_xlabel("Expert")
        axes[1, 2].set_ylabel(f"Z-score (k={k_repr})")
        axes[1, 2].set_title("Z-score per Expert")
        axes[1, 2].set_xticks(k_data["expert"])
        axes[1, 2].grid(True, alpha=STYLE.grid_alpha, axis="y")
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    
    return _save_or_show(fig, save_path, show)


def plot_full_analysis(
    results: Union[List[AlignmentResult], pd.DataFrame],
    metadata: Optional[AnalysisMetadata] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Optional[Path]:
    """Create comprehensive analysis figure with all plots.
    
    Args:
        results: Analysis results
        metadata: Analysis metadata for title
        save_path: Path to save figure
        show: Whether to display
    
    Returns:
        Path to saved file if save_path provided
    """
    STYLE.apply()
    df = _prepare_dataframe(results)
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    
    # Title
    if metadata:
        title = f"Alignment Analysis: Layer {metadata.layer}"
    else:
        title = "Alignment Analysis"
    fig.suptitle(title, fontsize=18, fontweight="bold")
    
    # Row 1: Main metrics
    plot_alignment_vs_k(results, ax=fig.add_subplot(gs[0, 0]))
    plot_zscore_vs_k(results, ax=fig.add_subplot(gs[0, 1]))
    plot_delta_vs_k(results, ax=fig.add_subplot(gs[0, 2]))
    plot_effect_vs_k(results, ax=fig.add_subplot(gs[0, 3]))
    
    # Row 2: Heatmaps
    plot_expert_heatmap(results, "align", ax=fig.add_subplot(gs[1, 0:2]), title="Alignment Heatmap")
    plot_expert_heatmap(results, "z_vs_shuffle", ax=fig.add_subplot(gs[1, 2:4]),
                       title="Z-score Heatmap", cmap="coolwarm", center_zero=True)
    
    # Row 3: Per-expert analysis
    if 1 in df["k"].values:
        plot_cos_squared_per_expert(results, ax=fig.add_subplot(gs[2, 0]))
    
    # Alignment distribution at representative k
    k_vals = sorted(df["k"].unique())
    k_repr = 128 if 128 in k_vals else k_vals[len(k_vals) // 2]
    k_data = df[df["k"] == k_repr].sort_values("expert")
    
    ax_align = fig.add_subplot(gs[2, 1])
    ax_align.bar(k_data["expert"], k_data["align"], color=STYLE.colors[0], alpha=0.8)
    ax_align.set_xlabel("Expert")
    ax_align.set_ylabel(f"Alignment (k={k_repr})")
    ax_align.set_title("Alignment per Expert")
    ax_align.set_xticks(k_data["expert"])
    ax_align.grid(True, alpha=STYLE.grid_alpha, axis="y")
    
    ax_z = fig.add_subplot(gs[2, 2])
    ax_z.bar(k_data["expert"], k_data["z_vs_shuffle"], color=STYLE.colors[1], alpha=0.8)
    ax_z.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_z.set_xlabel("Expert")
    ax_z.set_ylabel(f"Z-score (k={k_repr})")
    ax_z.set_title("Z-score per Expert")
    ax_z.set_xticks(k_data["expert"])
    ax_z.grid(True, alpha=STYLE.grid_alpha, axis="y")
    
    # Shuffle statistics
    ax_shuf = fig.add_subplot(gs[2, 3])
    shuf_mean = df.groupby("k")["shuffle_mean"].mean()
    shuf_std = df.groupby("k")["shuffle_std"].mean()
    ax_shuf.fill_between(shuf_mean.index, shuf_mean - shuf_std, shuf_mean + shuf_std,
                        alpha=0.3, color=STYLE.colors[4])
    ax_shuf.plot(shuf_mean.index, shuf_mean.values, color=STYLE.colors[4], linewidth=2)
    ax_shuf.set_xlabel("k")
    ax_shuf.set_ylabel("Shuffle Mean ± Std")
    ax_shuf.set_title("Shuffle Statistics")
    ax_shuf.set_xscale("log", base=2)
    ax_shuf.grid(True, alpha=STYLE.grid_alpha)
    
    return _save_or_show(fig, save_path, show)


def plot_comparison(
    results_list: List[tuple],  # List of (metadata, results) tuples
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Optional[Path]:
    """Compare multiple analysis runs.
    
    Args:
        results_list: List of (AnalysisMetadata, results) tuples
        save_path: Path to save figure
        show: Whether to display
    
    Returns:
        Path to saved file if save_path provided
    """
    STYLE.apply()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sort by layer number
    results_list = sorted(results_list, key=lambda x: x[0].layer)
    
    for i, (metadata, results) in enumerate(results_list):
        df = _prepare_dataframe(results)
        summary = df.groupby("k").mean(numeric_only=True)
        label = metadata.layer_label
        color = STYLE.colors[i % len(STYLE.colors)]
        
        # Alignment
        axes[0, 0].plot(summary.index, summary["align"], 
                       marker="o", label=label, color=color, linewidth=2)
        
        # Z-score
        axes[0, 1].plot(summary.index, summary["z_vs_shuffle"],
                       marker="s", label=label, color=color, linewidth=2)
        
        # Delta
        axes[1, 0].plot(summary.index, summary["delta_vs_shuffle"],
                       marker="^", label=label, color=color, linewidth=2)
        
        # Effect
        axes[1, 1].plot(summary.index, summary["effect_over_random"],
                       marker="d", label=label, color=color, linewidth=2)
    
    # Configure axes
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=STYLE.grid_alpha)
        ax.legend()
    
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("Mean Alignment")
    axes[0, 0].set_title("Alignment vs k")
    
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("Mean Z-score")
    axes[0, 1].set_title("Z-score vs k")
    axes[0, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("Delta vs Shuffle")
    axes[1, 0].set_title("Delta vs Shuffle")
    axes[1, 0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    
    axes[1, 1].set_xlabel("k")
    axes[1, 1].set_ylabel("Effect over Random")
    axes[1, 1].set_title("Effect over Random")
    axes[1, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    
    fig.suptitle("Comparison Across Layers", fontsize=16, fontweight="bold")
    
    return _save_or_show(fig, save_path, show)


def plot_shuffle_comparison(
    results_list: List[tuple],  # List of (metadata, results) tuples
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Optional[Path]:
    r"""Compare shuffle statistics across layers.

    Plots three panels:
      1. \Delta  = delta_vs_shuffle  (actual alignment - shuffle mean)
      2. \sigma  = shuffle_std       (shuffle standard deviation)
      3. Shuffle baseline            (shuffle_mean)

    Args:
        results_list: List of (AnalysisMetadata, results) tuples
        save_path: Path to save figure
        show: Whether to display

    Returns:
        Path to saved file if save_path provided
    """
    STYLE.apply()
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Sort by layer number
    results_list = sorted(results_list, key=lambda x: x[0].layer)

    for i, (metadata, results) in enumerate(results_list):
        df = _prepare_dataframe(results)
        summary = df.groupby("k").mean(numeric_only=True)
        label = metadata.layer_label
        color = STYLE.colors[i % len(STYLE.colors)]

        # Delta vs shuffle
        axes[0].plot(summary.index, summary["delta_vs_shuffle"],
                     marker="^", label=label, color=color, linewidth=2)

        # Shuffle std
        axes[1].plot(summary.index, summary["shuffle_std"],
                     marker="s", label=label, color=color, linewidth=2)

        # Shuffle mean (baseline)
        axes[2].plot(summary.index, summary["shuffle_mean"],
                     marker="o", label=label, color=color, linewidth=2)

    # Configure axes
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k")
        ax.grid(True, alpha=STYLE.grid_alpha)
        ax.legend()

    axes[0].set_ylabel(r"$\Delta$ (align $-$ shuffle mean)")
    axes[0].set_title(r"$\Delta$ vs $k$")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    axes[1].set_ylabel(r"$\sigma$ (shuffle std)")
    axes[1].set_title(r"$\sigma$ vs $k$")

    axes[2].set_ylabel("Shuffle Mean (baseline)")
    axes[2].set_title(r"Shuffle Baseline ($\mu_{shuffle}$) vs $k$")

    fig.suptitle("Shuffle Statistics Across Layers", fontsize=16, fontweight="bold")
    fig.tight_layout()

    return _save_or_show(fig, save_path, show)
