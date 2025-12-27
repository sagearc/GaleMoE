"""Generate plots for a single result file analysis."""
from io import BytesIO
from typing import Dict, Optional
import os
import warnings

import matplotlib
# Only use Agg if not in Jupyter notebook (check for IPython)
try:
    get_ipython()  # type: ignore
    # In Jupyter - use default backend (usually inline)
    pass
except NameError:
    # Not in Jupyter - use Agg backend
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.core.data_structures import AlignmentResult
from src.analysis.reporting.results import ResultsManager
from src.analysis.reporting.utils import (
    display_plot_in_jupyter,
    figure_to_bytes,
)


def generate_single_file_plots(
    result_file: str,
    save_to_file: bool = False,
    output_dir: Optional[str] = None,
    display_in_notebook: bool = False
) -> Dict[str, bytes]:
    """
    Generate all plots for a single result file.
    
    Args:
        result_file: Path to result JSON file
        save_to_file: If True, save plots to files. If False, return as bytes
        output_dir: Directory to save plots (if save_to_file=True)
        
    Returns:
        Dictionary mapping plot names to image bytes (or file paths if save_to_file=True)
    """
    manager = ResultsManager()
    metadata, results = manager.load(result_file)
    df = pd.DataFrame([r.__dict__ for r in results])
    
    plots = {}
    
    # Create comprehensive figure with all plots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    fig.suptitle(f'Complete Analysis: Layer {metadata.get("layer", "?")} - {metadata.get("timestamp", "?")}', 
                 fontsize=16, fontweight='bold')
    
    # Summary by k
    summary = df.groupby("k")[["align", "delta_vs_shuffle", "z_vs_shuffle", "effect_over_random"]].mean()
    
    # Plot 1: Alignment vs k
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(summary.index, summary["align"], marker='o', linewidth=2, markersize=6, color='#4CAF50')
    ax1.set_xlabel('k', fontsize=11)
    ax1.set_ylabel('Mean Alignment', fontsize=11)
    ax1.set_title('Alignment vs k', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Plot 2: Z-score vs k
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(summary.index, summary["z_vs_shuffle"], marker='s', linewidth=2, markersize=6, color='#2196F3')
    ax2.set_xlabel('k', fontsize=11)
    ax2.set_ylabel('Mean Z-score', fontsize=11)
    ax2.set_title('Z-score vs k', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 3: Effect over random vs k
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(summary.index, summary["effect_over_random"], marker='^', linewidth=2, markersize=6, color='#FF9800')
    ax3.set_xlabel('k', fontsize=11)
    ax3.set_ylabel('Mean Effect over Random', fontsize=11)
    ax3.set_title('Effect over Random vs k', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 4: Delta vs shuffle vs k
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(summary.index, summary["delta_vs_shuffle"], marker='d', linewidth=2, markersize=6, color='#9C27B0')
    ax4.set_xlabel('k', fontsize=11)
    ax4.set_ylabel('Mean Delta vs Shuffle', fontsize=11)
    ax4.set_title('Delta vs Shuffle vs k', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log', base=2)
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 5: Shuffle mean vs k
    ax5 = fig.add_subplot(gs[1, 1])
    shuffle_mean = df.groupby("k")["shuffle_mean"].mean()
    ax5.plot(shuffle_mean.index, shuffle_mean.values, marker='o', linewidth=2, markersize=6, color='#F44336')
    ax5.set_xlabel('k', fontsize=11)
    ax5.set_ylabel('Shuffle Mean', fontsize=11)
    ax5.set_title('Shuffle Mean vs k', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xscale('log', base=2)
    
    # Plot 6: Shuffle std vs k
    ax6 = fig.add_subplot(gs[1, 2])
    shuffle_std = df.groupby("k")["shuffle_std"].mean()
    ax6.plot(shuffle_std.index, shuffle_std.values, marker='s', linewidth=2, markersize=6, color='#E91E63')
    ax6.set_xlabel('k', fontsize=11)
    ax6.set_ylabel('Shuffle Std', fontsize=11)
    ax6.set_title('Shuffle Std vs k', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_xscale('log', base=2)
    
    # Plot 7: Alignment heatmap (experts × k)
    ax7 = fig.add_subplot(gs[2, 0])
    pivot_align = df.pivot_table(values='align', index='expert', columns='k', aggfunc='mean')
    im7 = ax7.imshow(pivot_align.values, aspect='auto', cmap='viridis')
    ax7.set_xlabel('k', fontsize=11)
    ax7.set_ylabel('Expert', fontsize=11)
    ax7.set_title('Alignment Heatmap (Experts × k)', fontsize=12, fontweight='bold')
    k_values = sorted(df['k'].unique())
    ax7.set_xticks(range(len(k_values)))
    ax7.set_xticklabels([str(k) for k in k_values], rotation=45, ha='right')
    ax7.set_yticks(range(len(pivot_align.index)))
    ax7.set_yticklabels([int(e) for e in pivot_align.index])
    plt.colorbar(im7, ax=ax7, label='Alignment')
    
    # Plot 8: Delta heatmap (experts × k)
    ax8 = fig.add_subplot(gs[2, 1])
    pivot_delta = df.pivot_table(values='delta_vs_shuffle', index='expert', columns='k', aggfunc='mean')
    # Center colormap at zero
    max_abs = max(abs(pivot_delta.values.min()), abs(pivot_delta.values.max()))
    im8 = ax8.imshow(pivot_delta.values, aspect='auto', cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
    ax8.set_xlabel('k', fontsize=11)
    ax8.set_ylabel('Expert', fontsize=11)
    ax8.set_title('Delta Heatmap (Experts × k)', fontsize=12, fontweight='bold')
    ax8.set_xticks(range(len(k_values)))
    ax8.set_xticklabels([str(k) for k in k_values], rotation=45, ha='right')
    ax8.set_yticks(range(len(pivot_delta.index)))
    ax8.set_yticklabels([int(e) for e in pivot_delta.index])
    plt.colorbar(im8, ax=ax8, label='Delta')
    
    # Plot 9: Z-score heatmap (experts × k)
    ax9 = fig.add_subplot(gs[2, 2])
    pivot_z = df.pivot_table(values='z_vs_shuffle', index='expert', columns='k', aggfunc='mean')
    # Center colormap at zero
    max_abs = max(abs(pivot_z.values.min()), abs(pivot_z.values.max()))
    im9 = ax9.imshow(pivot_z.values, aspect='auto', cmap='coolwarm', vmin=-max_abs, vmax=max_abs)
    ax9.set_xlabel('k', fontsize=11)
    ax9.set_ylabel('Expert', fontsize=11)
    ax9.set_title('Z-score Heatmap (Experts × k)', fontsize=12, fontweight='bold')
    ax9.set_xticks(range(len(k_values)))
    ax9.set_xticklabels([str(k) for k in k_values], rotation=45, ha='right')
    ax9.set_yticks(range(len(pivot_z.index)))
    ax9.set_yticklabels([int(e) for e in pivot_z.index])
    plt.colorbar(im9, ax=ax9, label='Z-score')
    
    # Plot 10: Cos²(θ) per expert (k=1) if available
    if 1 in df["k"].values:
        ax10 = fig.add_subplot(gs[3, 0])
        k1_data = df[df["k"] == 1].sort_values("expert")
        ax10.bar(k1_data["expert"], k1_data["cos_squared"], color='#FFC107', alpha=0.7)
        ax10.set_xlabel('Expert', fontsize=11)
        ax10.set_ylabel('Cos²(θ)', fontsize=11)
        ax10.set_title('Cos²(θ) per Expert (k=1)', fontsize=12, fontweight='bold')
        ax10.grid(True, alpha=0.3, axis='y')
        ax10.set_xticks(k1_data["expert"])
    
    # Plot 11: Alignment per expert at k=128 (or median k)
    ax11 = fig.add_subplot(gs[3, 1])
    k_for_expert = 128 if 128 in df["k"].unique() else sorted(df["k"].unique())[len(df["k"].unique()) // 2]
    k_data = df[df["k"] == k_for_expert].sort_values("expert")
    ax11.bar(k_data["expert"], k_data["align"], color='#4CAF50', alpha=0.7)
    ax11.set_xlabel('Expert', fontsize=11)
    ax11.set_ylabel(f'Alignment (k={k_for_expert})', fontsize=11)
    ax11.set_title(f'Alignment per Expert (k={k_for_expert})', fontsize=12, fontweight='bold')
    ax11.grid(True, alpha=0.3, axis='y')
    ax11.set_xticks(k_data["expert"])
    
    # Plot 12: Z-score per expert at k=128 (or median k)
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.bar(k_data["expert"], k_data["z_vs_shuffle"], color='#2196F3', alpha=0.7)
    ax12.set_xlabel('Expert', fontsize=11)
    ax12.set_ylabel(f'Z-score (k={k_for_expert})', fontsize=11)
    ax12.set_title(f'Z-score per Expert (k={k_for_expert})', fontsize=12, fontweight='bold')
    ax12.grid(True, alpha=0.3, axis='y')
    ax12.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax12.set_xticks(k_data["expert"])
    
    # Suppress tight_layout warnings - they're harmless when some axes are incompatible
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, message='.*tight_layout.*')
        try:
            plt.tight_layout()
        except Exception:
            # Some axes may not be compatible with tight_layout, that's okay
            pass
    
    # Display in notebook if requested
    if display_in_notebook:
        display_plot_in_jupyter(fig)
    
    # Save or return
    if save_to_file:
        from pathlib import Path
        if output_dir is None:
            output_dir = manager.output_dir / "plots"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"single_analysis_layer{metadata.get('layer', '?')}_{metadata.get('timestamp', '?')}.png"
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plots["complete_analysis"] = str(filepath)
    else:
        plots["complete_analysis"] = figure_to_bytes(fig)
        plt.close(fig)
    
    return plots

