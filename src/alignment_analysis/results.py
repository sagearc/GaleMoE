"""Results management for saving, loading, and comparing analysis results."""
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.alignment_analysis.base import AlignmentAnalyzer, AlignmentResult, AnalysisMetadata
from src.alignment_analysis.plots import (
    display_plot,
    extract_layer_number,
    get_summary_columns,
)
from src.utils import OutputDir


class ResultsManager:
    """Manages saving, loading, and comparing analysis results."""

    def __init__(self, output_dir: Optional[str] = None):
        self._output_dir = OutputDir.resolve(output_dir)

    @property
    def output_dir(self) -> Path:
        return self._output_dir.path
    
    def save(
        self,
        results: List[AlignmentResult],
        analyzer: AlignmentAnalyzer,
        layer: int,
        model_id: str,
    ) -> str:
        """Save results to a file with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method_name = analyzer.method_name
        analyzer_metadata = analyzer.get_metadata()
        n_shuffles = analyzer_metadata.get("n_shuffles", "unknown")
        seed = analyzer_metadata.get("seed", "unknown")
        
        filename_parts = [f"layer{layer}", method_name]
        
        k_list = list(analyzer.k_list)
        if len(k_list) <= 5:
            k_str = "_".join(map(str, k_list))
            filename_parts.append(f"k{k_str}")
        else:
            k_str = f"k{k_list[0]}-{k_list[-1]}-{len(k_list)}"
            filename_parts.append(k_str)
        
        if n_shuffles != "unknown":
            filename_parts.append(f"shuffles{n_shuffles}")
        
        if seed != "unknown":
            filename_parts.append(f"seed{seed}")
        
        filename_parts.append(timestamp)
        filename = "_".join(filename_parts) + ".json"
        filepath = self.output_dir / filename
        
        metadata = AnalysisMetadata(
            model_id=model_id,
            layer=layer,
            timestamp=timestamp,
            datetime=datetime.now().isoformat(),
            n_experts=len(set(r.expert for r in results)),
            method=analyzer_metadata["method"],
            k_list=analyzer_metadata["k_list"],
            n_shuffles=analyzer_metadata.get("n_shuffles"),
            seed=analyzer_metadata.get("seed"),
        )
        
        data = {
            "metadata": metadata.to_dict(),
            "results": [asdict(r) for r in results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return str(filepath)
    
    def load(self, filepath: str) -> Tuple[AnalysisMetadata, List[AlignmentResult]]:
        """Load results from a saved file.
        
        Returns:
            Tuple of (metadata, results) where metadata is an AnalysisMetadata object
        """
        filepath = Path(filepath)
        if not filepath.is_absolute():
            potential_path = self.output_dir / filepath
            if potential_path.exists():
                filepath = potential_path
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metadata = AnalysisMetadata.from_dict(data["metadata"])
        results = [AlignmentResult(**r) for r in data["results"]]
        
        return metadata, results
    
    def compare(
        self, 
        result_files: List[str], 
        show_plots: bool = True,
        diagnostic: bool = False
    ) -> pd.DataFrame:
        """Compare multiple result files side by side."""
        import matplotlib.pyplot as plt
        
        all_data = []
        all_results_data = []
        
        for filepath in result_files:
            metadata, results = self.load(filepath)
            df = AlignmentResult.to_dataframe(results)
            
            all_results_data.append({
                "filepath": filepath,
                "metadata": metadata,
                "results": results,
                "df": df
            })
            
            summary_cols = get_summary_columns(df)
            summary = df.groupby("k")[summary_cols].mean().reset_index()
            
            run_id = metadata.run_id
            summary["run"] = run_id
            summary["timestamp"] = metadata.timestamp
            
            all_data.append(summary)
        
        combined = pd.concat(all_data, ignore_index=True)
        print("\nComparison of runs:")
        print("=" * 80)
        print(combined.to_string(index=False))
        
        if show_plots:
            self._plot_comparison(combined, all_results_data)
        
        return combined
    
    def _plot_comparison(self, combined_df: pd.DataFrame, all_results_data: List[dict]) -> None:
        """Create visualization plots for comparison results."""
        import matplotlib.pyplot as plt
        
        runs = sorted(combined_df["run"].unique(), key=extract_layer_number)
        
        if len(runs) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Comparison of Analysis Runs", fontsize=16, fontweight='bold')
        
        # Plot 1: Alignment vs k
        ax = axes[0, 0]
        for run in runs:
            run_data = combined_df[combined_df["run"] == run]
            ax.plot(run_data["k"], run_data["align"], marker='o', label=run, linewidth=2, markersize=6)
        ax.set_xlabel('k (number of singular vectors)', fontsize=11)
        ax.set_ylabel('Mean Alignment', fontsize=11)
        ax.set_title('Alignment vs k', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.legend(loc='best', fontsize=9)
        
        # Plot 2: Z-score vs k
        ax = axes[0, 1]
        for run in runs:
            run_data = combined_df[combined_df["run"] == run]
            ax.plot(run_data["k"], run_data["z_vs_shuffle"], marker='s', label=run, linewidth=2, markersize=6)
        ax.set_xlabel('k (number of singular vectors)', fontsize=11)
        ax.set_ylabel('Mean Z-score', fontsize=11)
        ax.set_title('Z-score vs k', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize=9)
        
        # Plot 3: Effect over random vs k
        ax = axes[1, 0]
        for run in runs:
            run_data = combined_df[combined_df["run"] == run]
            ax.plot(run_data["k"], run_data["effect_over_random"], marker='^', label=run, linewidth=2, markersize=6)
        ax.set_xlabel('k (number of singular vectors)', fontsize=11)
        ax.set_ylabel('Mean Effect over Random', fontsize=11)
        ax.set_title('Effect over Random vs k', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize=9)
        
        # Plot 4: Delta vs shuffle vs k
        ax = axes[1, 1]
        for run in runs:
            run_data = combined_df[combined_df["run"] == run]
            ax.plot(run_data["k"], run_data["delta_vs_shuffle"], marker='d', label=run, linewidth=2, markersize=6)
        ax.set_xlabel('k (number of singular vectors)', fontsize=11)
        ax.set_ylabel('Mean Delta vs Shuffle', fontsize=11)
        ax.set_title('Delta vs Shuffle vs k', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize=9)
        
        try:
            plt.tight_layout()
        except Exception:
            pass
        
        display_plot(fig)
    
    def list_results(self, pattern: Optional[str] = None) -> List[Path]:
        """List all result files in the output directory."""
        if pattern:
            files = list(self.output_dir.glob(pattern))
        else:
            files = list(self.output_dir.glob("*.json"))
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


class ResultAnalyzer:
    """Analyze existing result files without re-running the analysis."""
    
    def __init__(self, results_dir: Optional[str] = None):
        self.manager = ResultsManager(output_dir=results_dir)
        self.results_dir = self.manager.output_dir
    
    def list_results(self, layer: Optional[int] = None, pattern: Optional[str] = None) -> List[Path]:
        """List available result files."""
        if pattern:
            files = list(self.results_dir.glob(pattern))
        elif layer is not None:
            files = list(self.results_dir.glob(f"layer{layer}_*.json"))
        else:
            files = list(self.results_dir.glob("*.json"))
        
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    
    def load_and_analyze(
        self,
        result_files: List[str],
        compare_runs_flag: bool = True
    ) -> pd.DataFrame:
        """Load result files and generate comparison outputs."""
        resolved_files = []
        for f in result_files:
            f_path = Path(f)
            if not f_path.is_absolute():
                potential_path = self.results_dir / f_path
                if potential_path.exists():
                    resolved_files.append(str(potential_path))
                elif f_path.exists():
                    resolved_files.append(str(f_path))
                else:
                    print(f"Warning: File not found: {f}")
            else:
                resolved_files.append(f)
        
        if not resolved_files:
            raise ValueError("No valid result files found")
        
        print(f"Loading {len(resolved_files)} result file(s)...")
        
        comparison_df = None
        if compare_runs_flag:
            comparison_df = compare_runs(resolved_files, show_plots=True)
        
        return comparison_df
    
    def analyze_latest(self, layer: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Analyze the most recent result file(s) for a layer."""
        files = self.list_results(layer=layer)
        
        if not files:
            raise ValueError(f"No result files found for layer {layer}" if layer else "No result files found")
        
        result_file = str(files[0])
        print(f"Using most recent result file: {result_file}")
        
        return self.load_and_analyze([result_file], **kwargs)


def select_layers_for_comparison(
    result_files: Optional[List[Path]] = None,
    layers: Optional[List[int]] = None,
    output_dir: Optional[str] = None
) -> List[Tuple[AnalysisMetadata, List[AlignmentResult]]]:
    """Select and load result files for specific layers.
    
    This is a convenience function for notebook usage to easily select which layers
    to compare in visualization plots.
    
    Args:
        result_files: Optional list of result file paths. If None, discovers all files.
        layers: List of layer numbers to include. If None, uses all available layers.
        output_dir: Results directory (defaults to 'results/')
        
    Returns:
        List of (metadata, results) tuples, one per layer, sorted by layer number
        
    Example:
        # Compare specific layers
        runs = select_layers_for_comparison(layers=[5, 10, 15, 20])
        plot_comparison(runs)
        
        # Compare all layers
        runs = select_layers_for_comparison()
        plot_comparison(runs)
        
        # Compare every 5th layer
        manager = ResultsManager()
        files = manager.list_results()
        all_layers = sorted({get_layer_from_filename(f) for f in files if get_layer_from_filename(f) is not None})
        runs = select_layers_for_comparison(layers=all_layers[::5])
        plot_comparison(runs)
    """
    import re
    
    def get_layer_from_filename(filepath):
        """Extract layer number from result filename."""
        filename = filepath.name if isinstance(filepath, Path) else Path(filepath).name
        match = re.search(r'layer(\d+)_', filename)
        return int(match.group(1)) if match else None
    
    # Get result files
    manager = ResultsManager(output_dir)
    if result_files is None:
        result_files = manager.list_results()
    
    # Group files by layer
    layer_files = {}
    for f in result_files:
        layer = get_layer_from_filename(f)
        if layer is not None:
            if layer not in layer_files:
                layer_files[layer] = []
            layer_files[layer].append(f)
    
    # Select layers
    if layers is None:
        layers = sorted(layer_files.keys())
    else:
        # Filter to only requested layers that exist
        layers = [l for l in layers if l in layer_files]
    
    # Load most recent file for each layer
    runs = []
    for layer in sorted(layers):
        if layer in layer_files:
            # Take most recent file (last in sorted list)
            latest_file = sorted(layer_files[layer])[-1]
            meta, res = load_results(str(latest_file))
            runs.append((meta, res))
            print(f"Layer {layer}: {latest_file.name}")
    
    print(f"\nLoaded {len(runs)} layer(s) for comparison: {sorted(layers)}")
    return runs


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


def load_results(filepath: str) -> Tuple[AnalysisMetadata, List[AlignmentResult]]:
    """Backward compatibility wrapper for ResultsManager.load().
    
    Returns:
        Tuple of (metadata, results) where metadata is an AnalysisMetadata object
    """
    manager = ResultsManager()
    return manager.load(filepath)


def compare_runs(result_files: List[str], show_plots: bool = True, diagnostic: bool = False) -> pd.DataFrame:
    """Backward compatibility wrapper for ResultsManager.compare()."""
    manager = ResultsManager()
    return manager.compare(result_files, show_plots=show_plots, diagnostic=diagnostic)


def analyze_results(
    result_files: Optional[List[str]] = None,
    layer: Optional[int] = None,
    compare_runs_flag: bool = True
) -> pd.DataFrame:
    """Convenience function to analyze existing result files."""
    analyzer = ResultAnalyzer()
    
    if result_files is None:
        return analyzer.analyze_latest(layer=layer, compare_runs_flag=compare_runs_flag)
    else:
        return analyzer.load_and_analyze(result_files, compare_runs_flag=compare_runs_flag)
