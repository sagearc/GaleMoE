"""Object-oriented utilities for saving, loading, and comparing analysis results."""
import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy

from src.analysis.core.analyzer import AlignmentAnalyzer
from src.analysis.core.data_structures import AlignmentResult
from src.analysis.reporting.utils import (
    build_run_id,
    get_summary_columns,
    display_plot_in_jupyter,
    extract_layer_number,
    get_layer_label,
)


def _get_project_root() -> Path:
    """Get the project root directory using git.
    
    Uses 'git rev-parse --show-toplevel' to find the git repository root,
    which should be the project root (GaleMoE/).
    
    Falls back to going up from current file if git is not available.
    """
    try:
        # Try to get git root
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        git_root = Path(result.stdout.strip())
        return git_root
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: assume this file is in src/analysis/reporting/, go up 3 levels
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        return project_root


class ResultsManager:
    """Manages saving, loading, and comparing analysis results.
    
    Results directory is always created relative to the project root (GaleMoE/),
    regardless of where the script is run from.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize ResultsManager.
        
        Args:
            output_dir: Optional output directory path. If None, uses 'results' 
                      relative to project root (GaleMoE/). If relative path provided,
                      it's relative to project root. If absolute path provided, uses as-is.
        """
        if output_dir is None:
            project_root = _get_project_root()
            self.output_dir = project_root / "results"
        else:
            output_path = Path(output_dir)
            if output_path.is_absolute():
                self.output_dir = output_path
            else:
                project_root = _get_project_root()
                self.output_dir = project_root / output_dir
        
        self.output_dir.mkdir(exist_ok=True)
    
    def save(
        self,
        results: List[AlignmentResult],
        analyzer: AlignmentAnalyzer,
        layer: int,
        model_id: str,
    ) -> str:
        """
        Save results to a file with metadata for easy comparison across runs.
        
        Args:
            results: List of AlignmentResult objects
            analyzer: AlignmentAnalyzer used for analysis (any method)
            layer: Layer number analyzed
            model_id: Model identifier
            
        Returns:
            Path to saved file
        """
        # Create filename with metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method_name = analyzer.method_name
        
        # Get metadata from analyzer (handles optional fields gracefully)
        metadata_dict = analyzer.get_metadata()
        n_shuffles = metadata_dict.get("n_shuffles", "unknown")
        seed = metadata_dict.get("seed", "unknown")
        
        # Build filename with available metadata
        # Format: layer{layer}_{method}_k{min}-{max}_shuffles{n}_seed{s}_{timestamp}
        filename_parts = [f"layer{layer}", method_name]
        
        # Add k_list info (compact format for long lists)
        k_list = list(analyzer.k_list)
        if len(k_list) <= 5:
            k_str = "_".join(map(str, k_list))
            filename_parts.append(f"k{k_str}")
        else:
            # For long k lists, use range format
            k_str = f"k{k_list[0]}-{k_list[-1]}-{len(k_list)}"
            filename_parts.append(k_str)
        
        # Always include n_shuffles if available (critical parameter)
        if n_shuffles != "unknown":
            filename_parts.append(f"shuffles{n_shuffles}")
        else:
            filename_parts.append("shuffles-unknown")
        
        # Always include seed if available
        if seed != "unknown":
            filename_parts.append(f"seed{seed}")
        
        filename_parts.append(timestamp)
        filename = "_".join(filename_parts) + ".json"
        filepath = self.output_dir / filename
        
        # Prepare data with metadata
        data = {
            "metadata": {
                "model_id": model_id,
                "layer": layer,
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "n_experts": len(set(r.expert for r in results)),
                **metadata_dict,  # Include all method-specific metadata
            },
            "results": [asdict(r) for r in results]
        }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Results saved to: {filepath}")
        return str(filepath)
    
    def load(self, filepath: str) -> Tuple[dict, List[AlignmentResult]]:
        """
        Load results from a saved file.
        
        Args:
            filepath: Path to results JSON file
            
        Returns:
            (metadata_dict, list_of_results)
        """
        filepath = Path(filepath)
        if not filepath.is_absolute():
            # Try relative to output_dir first, then as-is
            potential_path = self.output_dir / filepath
            if potential_path.exists():
                filepath = potential_path
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metadata = data["metadata"]
        results = [AlignmentResult(**r) for r in data["results"]]
        
        return metadata, results
    
    def compare(
        self, 
        result_files: List[str], 
        show_plots: bool = True,
        diagnostic: bool = False
    ) -> pd.DataFrame:
        """
        Compare multiple result files side by side.
        
        Args:
            result_files: List of paths to result JSON files
            show_plots: Whether to display visualization plots (default: True)
            diagnostic: If True, show detailed diagnostic plots in addition to default plots (default: False)
            
        Returns:
            DataFrame with comparison results
        """
        all_data = []
        all_results_data = []  # Store full results for visualization
        
        for filepath in result_files:
            metadata, results = self.load(filepath)
            df = pd.DataFrame([asdict(r) for r in results])
            
            # Store full results with metadata for visualization
            all_results_data.append({
                "filepath": filepath,
                "metadata": metadata,
                "results": results,
                "df": df
            })
            
            # Add metadata columns
            summary_cols = get_summary_columns(df)
            summary = df.groupby("k")[summary_cols].mean()
            summary = summary.reset_index()
            
            # Build run identifier with available metadata
            run_id = build_run_id(metadata)
            summary["run"] = run_id
            summary["timestamp"] = metadata["timestamp"]
            
            all_data.append(summary)
        
        combined = pd.concat(all_data, ignore_index=True)
        print("\nComparison of runs:")
        print("=" * 80)
        print(combined.to_string(index=False))
        
        # Add visualizations if requested
        if show_plots:
            # Always show default comparison plots (both runs on same plots)
            self._plot_comparison(combined, all_results_data)
            
            # Show diagnostic plots if requested
            if diagnostic:
                self._plot_diagnostics(all_results_data)
        
        return combined
    
    
    def _plot_comparison(self, combined_df: pd.DataFrame, all_results_data: List[dict] = None) -> None:
        """Create visualization plots for comparison results.
        
        Args:
            combined_df: DataFrame with comparison data from multiple runs
            all_results_data: List of dicts with full results data for each run
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get unique runs and sort by layer number
            runs = sorted(combined_df["run"].unique(), key=extract_layer_number)
            n_runs = len(runs)
            
            if n_runs == 0:
                return
            
            # Check if any run has cos_squared data (k=1)
            has_cos_squared = "cos_squared" in combined_df.columns and combined_df["cos_squared"].notna().any()
            
            # Check if we have multiple layers to compare
            has_multiple_layers = all_results_data and len(set(m["metadata"].get("layer", None) for m in all_results_data)) > 1
            
            # Always show expert plots if we have multiple runs (even same layer)
            has_multiple_runs = len(runs) > 1
            
            # Create figure with subplots
            # Always include expert plots if we have multiple runs or multiple layers
            if has_cos_squared and (has_multiple_layers or has_multiple_runs):
                fig, axes = plt.subplots(4, 2, figsize=(16, 24))
            elif has_cos_squared:
                fig, axes = plt.subplots(3, 2, figsize=(16, 18))
            elif has_multiple_layers or has_multiple_runs:
                fig, axes = plt.subplots(3, 2, figsize=(16, 18))
            else:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle("Comparison of Analysis Runs", fontsize=16, fontweight='bold')
            
            # Plot 1: Alignment vs k (top left)
            ax = axes[0, 0]
            for run in runs:
                run_data = combined_df[combined_df["run"] == run]
                ax.plot(
                    run_data["k"], 
                    run_data["align"], 
                    marker='o', 
                    label=run,
                    linewidth=2,
                    markersize=6
                )
            ax.set_xlabel('k (number of singular vectors)', fontsize=11)
            ax.set_ylabel('Mean Alignment', fontsize=11)
            ax.set_title('Alignment vs k', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log', base=2)
            ax.legend(loc='best', fontsize=9)
            
            # Plot 2: Z-score vs k (top right)
            ax = axes[0, 1]
            for run in runs:
                run_data = combined_df[combined_df["run"] == run]
                ax.plot(
                    run_data["k"], 
                    run_data["z_vs_shuffle"], 
                    marker='s', 
                    label=run,
                    linewidth=2,
                    markersize=6
                )
            ax.set_xlabel('k (number of singular vectors)', fontsize=11)
            ax.set_ylabel('Mean Z-score', fontsize=11)
            ax.set_title('Z-score vs k', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log', base=2)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.legend(loc='best', fontsize=9)
            
            # Plot 3: Effect over random vs k (bottom left)
            ax = axes[1, 0]
            for run in runs:
                run_data = combined_df[combined_df["run"] == run]
                ax.plot(
                    run_data["k"], 
                    run_data["effect_over_random"], 
                    marker='^', 
                    label=run,
                    linewidth=2,
                    markersize=6
                )
            ax.set_xlabel('k (number of singular vectors)', fontsize=11)
            ax.set_ylabel('Mean Effect over Random', fontsize=11)
            ax.set_title('Effect over Random vs k', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log', base=2)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.legend(loc='best', fontsize=9)
            
            # Plot 4: Delta vs shuffle vs k (bottom right)
            ax = axes[1, 1]
            for run in runs:
                run_data = combined_df[combined_df["run"] == run]
                ax.plot(
                    run_data["k"], 
                    run_data["delta_vs_shuffle"], 
                    marker='d', 
                    label=run,
                    linewidth=2,
                    markersize=6
                )
            ax.set_xlabel('k (number of singular vectors)', fontsize=11)
            ax.set_ylabel('Mean Delta vs Shuffle', fontsize=11)
            ax.set_title('Delta vs Shuffle vs k', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log', base=2)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.legend(loc='best', fontsize=9)
            
            # Plot 5 & 6: Expert comparison (if multiple runs or multiple layers)
            plot_row = 2
            if (has_multiple_layers or has_multiple_runs) and all_results_data:
                # Find a suitable k value that exists in all datasets
                # Prefer k=128, but fallback to median k if not available
                all_k_values = set()
                for results_info in all_results_data:
                    df = results_info["df"]
                    all_k_values.update(df["k"].unique())
                
                if 128 in all_k_values:
                    k_for_expert_plot = 128
                elif len(all_k_values) > 0:
                    # Use median k value
                    k_for_expert_plot = sorted(all_k_values)[len(all_k_values) // 2]
                else:
                    k_for_expert_plot = None
                
                if k_for_expert_plot is not None:
                    # Sort all_results_data by layer number
                    sorted_all_results_data = sorted(all_results_data, key=lambda r: extract_layer_number(get_layer_label(r["metadata"])))
                    # Plot 5: Alignment per expert
                    ax = axes[plot_row, 0]
                    
                    for idx, results_info in enumerate(sorted_all_results_data):
                        metadata = results_info["metadata"]
                        df = results_info["df"]
                        layer = metadata.get("layer", "?")
                        run_id = build_run_id(metadata)
                        
                        # Get expert-level data for the chosen k
                        expert_data = df[df["k"] == k_for_expert_plot].groupby("expert")["align"].mean().sort_index()
                        if len(expert_data) > 0:
                            if has_multiple_layers:
                                # Line plot for multiple layers
                                ax.plot(
                                    expert_data.index,
                                    expert_data.values,
                                    marker='o',
                                    label=f"L{layer}",
                                    linewidth=2,
                                    markersize=6,
                                    alpha=0.8
                                )
                            else:
                                # Scatter plot for same layer, different runs
                                ax.scatter(
                                    expert_data.index,
                                    expert_data.values,
                                    label=run_id,
                                    alpha=0.7,
                                    s=100
                                )
                    
                    ax.set_xlabel('Expert Index', fontsize=11)
                    ax.set_ylabel('Mean Alignment', fontsize=11)
                    title = f'Expert Alignment Comparison (k={k_for_expert_plot})'
                    if has_multiple_layers:
                        title += ' - Across Layers'
                    ax.set_title(title, fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='best', fontsize=9)
                    if len(ax.get_xticks()) > 0:
                        max_expert = int(df["expert"].max()) if len(df) > 0 else 7
                        ax.set_xticks(range(max_expert + 1))
                    
                    # Plot 6: Z-score per expert
                    ax = axes[plot_row, 1]
                    for idx, results_info in enumerate(sorted_all_results_data):
                        metadata = results_info["metadata"]
                        df = results_info["df"]
                        layer = metadata.get("layer", "?")
                        run_id = build_run_id(metadata)
                        
                        expert_data = df[df["k"] == k_for_expert_plot].groupby("expert")["z_vs_shuffle"].mean().sort_index()
                        if len(expert_data) > 0:
                            if has_multiple_layers:
                                # Line plot for multiple layers
                                ax.plot(
                                    expert_data.index,
                                    expert_data.values,
                                    marker='s',
                                    label=f"L{layer}",
                                    linewidth=2,
                                    markersize=6,
                                    alpha=0.8
                                )
                            else:
                                # Scatter plot for same layer, different runs
                                ax.scatter(
                                    expert_data.index,
                                    expert_data.values,
                                    label=run_id,
                                    alpha=0.7,
                                    s=100
                                )
                    
                    ax.set_xlabel('Expert Index', fontsize=11)
                    ax.set_ylabel('Mean Z-score', fontsize=11)
                    title = f'Expert Z-score Comparison (k={k_for_expert_plot})'
                    if has_multiple_layers:
                        title += ' - Across Layers'
                    ax.set_title(title, fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                    ax.legend(loc='best', fontsize=9)
                    if len(ax.get_xticks()) > 0:
                        max_expert = int(df["expert"].max()) if len(df) > 0 else 7
                        ax.set_xticks(range(max_expert + 1))
                    
                    plot_row += 1
            
            # Plot 7 & 8: Cos²(θ) comparison (if available)
            if has_cos_squared and all_results_data:
                # Sort all_results_data by layer number
                sorted_all_results_data = sorted(all_results_data, key=lambda r: extract_layer_number(get_layer_label(r["metadata"])))
                # Plot 7: Cos²(θ) per expert across layers (k=1)
                ax = axes[plot_row, 0]
                for results_info in sorted_all_results_data:
                    metadata = results_info["metadata"]
                    df = results_info["df"]
                    layer = metadata.get("layer", "?")
                    
                    k1_expert = df[(df["k"] == 1) & (df["cos_squared"].notna())].sort_values("expert")
                    if len(k1_expert) > 0:
                        if has_multiple_layers:
                            # Line plot for multiple layers
                            ax.plot(
                                k1_expert["expert"],
                                k1_expert["cos_squared"],
                                marker='o',
                                label=f"Layer {layer}",
                                linewidth=2,
                                markersize=6,
                                alpha=0.8
                            )
                        else:
                            # Scatter plot for single layer
                            ax.scatter(
                                k1_expert["expert"],
                                k1_expert["cos_squared"],
                                label=f"Layer {layer}",
                                alpha=0.7,
                                s=100
                            )
                
                ax.set_xlabel('Expert Index', fontsize=11)
                ax.set_ylabel('Cos²(θ)', fontsize=11)
                title = 'Cos²(θ) per Expert Across Layers (k=1)' if has_multiple_layers else 'Cos²(θ) per Expert (k=1)'
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=9)
                if len(ax.get_xticks()) > 0:
                    max_expert = int(df["expert"].max()) if len(df) > 0 else 7
                    ax.set_xticks(range(max_expert + 1))
                
                # Plot 8: Cos²(θ) statistics comparison
                ax = axes[plot_row, 1]
                cos_stats = []
                for run in runs:
                    run_data = combined_df[(combined_df["run"] == run) & (combined_df["k"] == 1)]
                    if len(run_data) > 0 and run_data["cos_squared"].notna().any():
                        cos_val = run_data["cos_squared"].iloc[0]
                        cos_stats.append({"run": run, "cos_squared": cos_val})
                
                if cos_stats:
                    cos_df = pd.DataFrame(cos_stats)
                    bars = ax.bar(range(len(cos_df)), cos_df["cos_squared"], alpha=0.7)
                    ax.set_xticks(range(len(cos_df)))
                    ax.set_xticklabels(cos_df["run"], rotation=45, ha='right', fontsize=9)
                    ax.set_ylabel('Mean Cos²(θ)', fontsize=11)
                    ax.set_title('Mean Cos²(θ) Comparison (k=1)', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')
                    # Add value labels on bars
                    for i, (bar, val) in enumerate(zip(bars, cos_df["cos_squared"])):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                               f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            try:
                plt.tight_layout()
            except Exception:
                # Some axes may not be compatible with tight_layout, that's okay
                pass
            
            try:
                # Check if we're in Jupyter and what backend is being used
                from IPython import get_ipython
                import matplotlib
                
                is_jupyter = get_ipython() is not None
                backend = matplotlib.get_backend()
                is_agg_backend = backend.lower() == 'agg'
                
                if is_jupyter:
                    # In Jupyter, check backend
                    if is_agg_backend:
                        # Agg backend doesn't display - use Image display
                        from IPython.display import display
                        from io import BytesIO
                        buf = BytesIO()
                        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        from IPython.display import Image as IPImage
                        display(IPImage(data=buf.read()))
                        buf.close()
                        plt.close(fig)
                    else:
                        # Non-Agg backend - try plt.show() (works with %matplotlib inline)
                        try:
                            plt.show()
                        except Exception:
                            # Fallback: convert to image and display
                            from IPython.display import display
                            from io import BytesIO
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            from IPython.display import Image as IPImage
                            display(IPImage(data=buf.read()))
                            buf.close()
                            plt.close(fig)
                else:
                    # Not in Jupyter, try to show, if fails just close
                    try:
                        plt.show()
                    except Exception:
                        plt.close(fig)
            except (ImportError, Exception):
                # Fallback: try to show, if fails just close
                try:
                    plt.show()
                except Exception:
                    plt.close(fig)
            
        except ImportError:
            print("\nNote: Matplotlib not available. Skipping visualization.")
        except Exception as e:
            print(f"\nWarning: Could not create visualization: {e}")
    
    def _plot_diagnostics(self, all_results_data: List[dict]) -> None:
        """Create detailed diagnostic plots comparing multiple runs.
        
        Shows both runs on the same plots for easy comparison.
        
        Args:
            all_results_data: List of dicts with full results data for each run
        """
        try:
            import matplotlib.pyplot as plt
            
            if not all_results_data:
                return
            
            # Sort all_results_data by layer number
            sorted_all_results_data = sorted(all_results_data, key=lambda r: extract_layer_number(get_layer_label(r["metadata"])))
            # Prepare data in format expected by diagnostic plots
            diagnostic_data = []
            for results_info in sorted_all_results_data:
                metadata = results_info["metadata"]
                df = results_info["df"].copy()
                run_id = build_run_id(metadata)
                df['run_id'] = run_id
                df['timestamp'] = metadata.get('timestamp', 'unknown')
                diagnostic_data.append({
                    'metadata': metadata,
                    'results': results_info["results"],
                    'df': df,
                    'filepath': results_info["filepath"]
                })
            
            # Determine layer (use first layer if multiple)
            layer = diagnostic_data[0]['metadata'].get('layer', 'unknown')
            
            print("\n" + "=" * 80)
            print("Diagnostic Plots (showing both runs on same plots)")
            print("=" * 80)
            
            # 1. Shuffle statistics
            self._plot_diagnostic_shuffle_stats(diagnostic_data, layer)
            
            # 2. Z-score decomposition
            self._plot_diagnostic_zscore_decomposition(diagnostic_data, layer)
            
            # 3. Distribution plots (for representative k values)
            self._plot_diagnostic_distributions(diagnostic_data, layer)
            
            # 4. Per-expert breakdown
            self._plot_diagnostic_per_expert_breakdown(diagnostic_data, layer)
            
        except ImportError:
            print("\nNote: Matplotlib/scipy not available. Skipping diagnostic plots.")
        except Exception as e:
            print(f"\nWarning: Could not create diagnostic plots: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_diagnostic_shuffle_stats(self, all_data: List[dict], layer) -> None:
        """Plot shuffle statistics (mean, std, log(std)) with both runs."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Shuffle Statistics vs k (Layer {layer}) - Diagnostic', fontsize=14, fontweight='bold')
        
        # Plot 1: Mean shuffle vs k
        ax = axes[0]
        for data in all_data:
            df = data['df']
            run_id = df['run_id'].iloc[0]
            mean_by_k = df.groupby('k')['shuffle_mean'].mean()
            ax.plot(mean_by_k.index, mean_by_k.values, marker='o', label=run_id, linewidth=2)
        ax.set_xlabel('k', fontsize=11)
        ax.set_ylabel('μ_shuffle(k)', fontsize=11)
        ax.set_title('Shuffle Mean vs k', fontsize=12)
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Plot 2: Std shuffle vs k
        ax = axes[1]
        for data in all_data:
            df = data['df']
            run_id = df['run_id'].iloc[0]
            std_by_k = df.groupby('k')['shuffle_std'].mean()
            ax.plot(std_by_k.index, std_by_k.values, marker='s', label=run_id, linewidth=2)
        ax.set_xlabel('k', fontsize=11)
        ax.set_ylabel('σ_shuffle(k)', fontsize=11)
        ax.set_title('Shuffle Std vs k', fontsize=12)
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Plot 3: Log(std) shuffle vs k
        ax = axes[2]
        for data in all_data:
            df = data['df']
            run_id = df['run_id'].iloc[0]
            std_by_k = df.groupby('k')['shuffle_std'].mean()
            log_std = np.log(std_by_k.values + 1e-10)
            ax.plot(std_by_k.index, log_std, marker='^', label=run_id, linewidth=2)
        ax.set_xlabel('k', fontsize=11)
        ax.set_ylabel('log(σ_shuffle(k))', fontsize=11)
        ax.set_title('Log Shuffle Std vs k', fontsize=12)
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        try:
            plt.tight_layout()
        except Exception:
            # Some axes may not be compatible with tight_layout, that's okay
            pass
        
        display_plot_in_jupyter(fig)
    
    def _plot_diagnostic_zscore_decomposition(self, all_data: List[dict], layer) -> None:
        """Plot z-score decomposition (delta, sigma, z) with both runs."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Z-score Decomposition vs k (Layer {layer}) - Diagnostic', fontsize=14, fontweight='bold')
        
        # Plot 1: Delta vs k
        ax = axes[0]
        for data in all_data:
            df = data['df']
            run_id = df['run_id'].iloc[0]
            delta_by_k = df.groupby('k')['delta_vs_shuffle'].mean()
            ax.plot(delta_by_k.index, delta_by_k.values, marker='o', label=run_id, linewidth=2)
        ax.set_xlabel('k', fontsize=11)
        ax.set_ylabel('Δ(k) = align(k) - μ_shuffle(k)', fontsize=11)
        ax.set_title('Delta vs k', fontsize=12)
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)
        
        # Plot 2: Sigma shuffle vs k
        ax = axes[1]
        for data in all_data:
            df = data['df']
            run_id = df['run_id'].iloc[0]
            std_by_k = df.groupby('k')['shuffle_std'].mean()
            ax.plot(std_by_k.index, std_by_k.values, marker='s', label=run_id, linewidth=2)
        ax.set_xlabel('k', fontsize=11)
        ax.set_ylabel('σ_shuffle(k)', fontsize=11)
        ax.set_title('Shuffle Std vs k', fontsize=12)
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Plot 3: Z-score vs k
        ax = axes[2]
        for data in all_data:
            df = data['df']
            run_id = df['run_id'].iloc[0]
            z_by_k = df.groupby('k')['z_vs_shuffle'].mean()
            ax.plot(z_by_k.index, z_by_k.values, marker='^', label=run_id, linewidth=2)
        ax.set_xlabel('k', fontsize=11)
        ax.set_ylabel('z(k) = Δ(k) / σ_shuffle(k)', fontsize=11)
        ax.set_title('Z-score vs k', fontsize=12)
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)
        
        try:
            plt.tight_layout()
        except Exception:
            # Some axes may not be compatible with tight_layout, that's okay
            pass
        
        display_plot_in_jupyter(fig)
    
    def _plot_diagnostic_distributions(self, all_data: List[dict], layer) -> None:
        """Plot distribution comparisons for representative k values with both runs."""
        import matplotlib.pyplot as plt
        
        # Find available k values
        all_k_values = set()
        for data in all_data:
            all_k_values.update(data['df']['k'].unique())
        
        # Select representative k values that exist
        preferred_k = [32, 128, 512, 2048]
        k_values = [k for k in preferred_k if k in all_k_values]
        if not k_values and all_k_values:
            # Fallback to median k values
            sorted_k = sorted(all_k_values)
            k_values = [sorted_k[len(sorted_k) // 4], sorted_k[len(sorted_k) // 2], sorted_k[3 * len(sorted_k) // 4]]
        
        for k in k_values[:4]:  # Limit to 4 plots
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            # Determine if we're comparing multiple layers
            layers = set(d['metadata'].get('layer', '?') for d in all_data)
            if len(layers) > 1:
                layer_str = f"Layers {sorted(layers)}"
            else:
                layer_str = f"Layer {layer}"
            fig.suptitle(f'Shuffle Distribution vs True Alignment (k={k}, {layer_str}) - Diagnostic', 
                        fontsize=14, fontweight='bold')
            
            for data in all_data:
                df = data['df']
                run_id = df['run_id'].iloc[0]
                k_data = df[df['k'] == k]
                
                if len(k_data) == 0:
                    continue
                
                shuffle_mean = k_data['shuffle_mean'].iloc[0]
                shuffle_std = k_data['shuffle_std'].iloc[0]
                true_align = k_data['align'].mean()
                
                x = np.linspace(max(0, shuffle_mean - 4*shuffle_std), 
                               shuffle_mean + 4*shuffle_std, 200)
                y = stats.norm.pdf(x, shuffle_mean, shuffle_std)
                
                ax.plot(x, y, label=f'{run_id} (shuffle)', linewidth=2, alpha=0.7)
                ax.axvline(true_align, color=ax.lines[-1].get_color(), 
                          linestyle='--', linewidth=2, 
                          label=f'{run_id} (true align)', alpha=0.8)
            
            ax.set_xlabel('Alignment', fontsize=11)
            ax.set_ylabel('Density (approximated)', fontsize=11)
            ax.set_title(f'Distribution Comparison (k={k})', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            try:
                plt.tight_layout()
            except Exception:
                # Some axes may not be compatible with tight_layout, that's okay
                pass
            
            try:
                # Check if we're in Jupyter and what backend is being used
                from IPython import get_ipython
                import matplotlib
                
                is_jupyter = get_ipython() is not None
                backend = matplotlib.get_backend()
                is_agg_backend = backend.lower() == 'agg'
                
                if is_jupyter:
                    # In Jupyter, check backend
                    if is_agg_backend:
                        # Agg backend doesn't display - use Image display
                        from IPython.display import display
                        from io import BytesIO
                        buf = BytesIO()
                        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        from IPython.display import Image as IPImage
                        display(IPImage(data=buf.read()))
                        buf.close()
                        plt.close(fig)
                    else:
                        # Non-Agg backend - try plt.show() (works with %matplotlib inline)
                        try:
                            plt.show()
                        except Exception:
                            # Fallback: convert to image and display
                            from IPython.display import display
                            from io import BytesIO
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                            buf.seek(0)
                            from IPython.display import Image as IPImage
                            display(IPImage(data=buf.read()))
                            buf.close()
                            plt.close(fig)
                else:
                    # Not in Jupyter, try to show, if fails just close
                    try:
                        plt.show()
                    except Exception:
                        plt.close(fig)
            except (ImportError, Exception):
                # Fallback: try to show, if fails just close
                try:
                    plt.show()
                except Exception:
                    plt.close(fig)
    
    def _plot_diagnostic_per_expert_breakdown(self, all_data: List[dict], layer) -> None:
        """Plot per-expert breakdown with both runs."""
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        fig.suptitle(f'Per-Expert Breakdown (Layer {layer}) - Diagnostic', fontsize=14, fontweight='bold')
        
        # Heatmap 1: Alignment per expert vs k
        ax1 = fig.add_subplot(gs[0, 0])
        for idx, data in enumerate(all_data):
            df = data['df']
            run_id = df['run_id'].iloc[0]
            
            pivot = df.pivot_table(values='align', index='expert', columns='k', aggfunc='mean')
            
            im = ax1.imshow(pivot.values, aspect='auto', cmap='viridis', alpha=0.7 if idx > 0 else 1.0)
            if idx == 0:
                plt.colorbar(im, ax=ax1, label='Alignment')
        
        ax1.set_xlabel('k', fontsize=11)
        ax1.set_ylabel('Expert', fontsize=11)
        ax1.set_title('Alignment Heatmap (Experts × k)', fontsize=12)
        k_values = sorted(df['k'].unique())
        ax1.set_xticks(range(len(k_values)))
        ax1.set_xticklabels([str(k) for k in k_values], rotation=45)
        ax1.set_yticks(range(len(pivot.index)))
        ax1.set_yticklabels([int(e) for e in pivot.index])
        
        # Heatmap 2: Delta per expert vs k
        ax2 = fig.add_subplot(gs[0, 1])
        for idx, data in enumerate(all_data):
            df = data['df']
            run_id = df['run_id'].iloc[0]
            
            pivot = df.pivot_table(values='delta_vs_shuffle', index='expert', columns='k', aggfunc='mean')
            
            im = ax2.imshow(pivot.values, aspect='auto', cmap='RdBu_r', alpha=0.7 if idx > 0 else 1.0)
            if idx == 0:
                plt.colorbar(im, ax=ax2, label='Delta')
        
        ax2.set_xlabel('k', fontsize=11)
        ax2.set_ylabel('Expert', fontsize=11)
        ax2.set_title('Delta Heatmap (Experts × k)', fontsize=12)
        ax2.set_xticks(range(len(k_values)))
        ax2.set_xticklabels([str(k) for k in k_values], rotation=45)
        ax2.set_yticks(range(len(pivot.index)))
        ax2.set_yticklabels([int(e) for e in pivot.index])
        
        # Scatter: Alignment vs expert (for k=128 or median k)
        ax3 = fig.add_subplot(gs[1, 0])
        all_k_vals = set()
        for data in all_data:
            all_k_vals.update(data['df']['k'].unique())
        k_fixed = 128 if 128 in all_k_vals else sorted(all_k_vals)[len(all_k_vals) // 2]
        
        for data in all_data:
            df = data['df']
            run_id = df['run_id'].iloc[0]
            k_data = df[df['k'] == k_fixed]
            
            if len(k_data) > 0:
                align_vals = k_data['align'].values
                expert_vals = k_data['expert'].values
                ax3.scatter(expert_vals, align_vals, label=run_id, alpha=0.7, s=100)
        
        ax3.set_xlabel('Expert Index', fontsize=11)
        ax3.set_ylabel(f'Alignment (k={k_fixed})', fontsize=11)
        ax3.set_title(f'Alignment vs Expert (k={k_fixed})', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
        
        # Scatter: Delta vs expert
        ax4 = fig.add_subplot(gs[1, 1])
        for data in all_data:
            df = data['df']
            run_id = df['run_id'].iloc[0]
            k_data = df[df['k'] == k_fixed]
            
            if len(k_data) > 0:
                delta_vals = k_data['delta_vs_shuffle'].values
                expert_vals = k_data['expert'].values
                ax4.scatter(expert_vals, delta_vals, label=run_id, alpha=0.7, s=100)
        
        ax4.set_xlabel('Expert Index', fontsize=11)
        ax4.set_ylabel(f'Delta (k={k_fixed})', fontsize=11)
        ax4.set_title(f'Delta vs Expert (k={k_fixed})', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax4.legend(fontsize=9)
        
        try:
            plt.tight_layout()
        except Exception:
            # Some axes may not be compatible with tight_layout, that's okay
            pass
        
        display_plot_in_jupyter(fig)
    
    def list_results(self, pattern: Optional[str] = None) -> List[Path]:
        """
        List all result files in the output directory.
        
        Args:
            pattern: Optional pattern to filter files (e.g., "layer10_*")
            
        Returns:
            List of result file paths
        """
        if pattern:
            files = list(self.output_dir.glob(pattern))
        else:
            files = list(self.output_dir.glob("*.json"))
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


# Backward compatibility functions
def save_results(
    results: List[AlignmentResult],
    analyzer: AlignmentAnalyzer,
    layer: int,
    model_id: str,
    output_dir: str = "results"
) -> str:
    """Backward compatibility wrapper for ResultsManager.save()."""
    manager = ResultsManager(output_dir)
    return manager.save(results, analyzer, layer, model_id)


def load_results(filepath: str) -> Tuple[dict, List[AlignmentResult]]:
    """Backward compatibility wrapper for ResultsManager.load()."""
    manager = ResultsManager()
    return manager.load(filepath)


def compare_runs(result_files: List[str], show_plots: bool = True, diagnostic: bool = False) -> pd.DataFrame:
    """Backward compatibility wrapper for ResultsManager.compare().
    
    Args:
        result_files: List of paths to result JSON files
        show_plots: Whether to display visualization plots (default: True)
        diagnostic: If True, show detailed diagnostic plots in addition to default plots (default: False)
        
    Returns:
        DataFrame with comparison results
    """
    manager = ResultsManager()
    return manager.compare(result_files, show_plots=show_plots, diagnostic=diagnostic)

