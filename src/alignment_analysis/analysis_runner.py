"""Runner that coordinates weight loading and analysis for router-expert alignment."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd

from src.alignment_analysis.base import AlignmentAnalyzer, AlignmentResult
from src.alignment_analysis.svd import SVDAlignmentAnalyzer
from src.alignment_analysis.loader import MoEWeightsRepository
from src.alignment_analysis.results import save_results
from src.models.model_loader import BaseMoE


class AlignmentRunner:
    """Coordinates weight loading and analysis for router-expert alignment."""
    
    def __init__(
        self,
        moe: BaseMoE,
        repo: Optional[MoEWeightsRepository] = None,
        analyzer: Optional[AlignmentAnalyzer] = None
    ):
        self.moe = moe
        self.repo = repo or MoEWeightsRepository(moe)
        self.analyzer: AlignmentAnalyzer = analyzer or SVDAlignmentAnalyzer()

    def run_layer(self, layer: int) -> List[AlignmentResult]:
        """Run analysis on a single layer."""
        print(f"Analyzing layer {layer}...")
        self.repo.prefetch_layer(layer=layer)
        layer_w = self.repo.load_layer(layer)
        return self.analyzer.analyze_layer(layer_w)

    def run_layers(self, layers: Sequence[int]) -> List[AlignmentResult]:
        """Run analysis on multiple layers."""
        all_res: List[AlignmentResult] = []
        for l in layers:
            all_res.extend(self.run_layer(int(l)))
        return all_res

    def run_full_analysis(
        self,
        layer: int,
        save: bool = True,
        print_summary: bool = True,
    ) -> tuple[List[AlignmentResult], Optional[Path]]:
        """Run complete analysis on a layer: analyze, save, and print summary.
        
        Args:
            layer: Layer number to analyze
            save: Whether to save results to JSON file
            print_summary: Whether to print summary statistics
            
        Returns:
            Tuple of (results list, saved file path or None)
        """
        print(f"Running alignment analysis on Layer {layer}...")
        results = self.run_layer(layer=layer)
        
        # Save results
        saved_path = None
        if save:
            saved_path = save_results(results, self.analyzer, layer, self.moe.model_id)
        
        # Print summary
        if print_summary:
            self._print_summary(results)
        
        return results, saved_path

    def _print_summary(self, results: List[AlignmentResult]) -> None:
        """Print summary statistics for analysis results."""
        pd.set_option('display.max_columns', 20)
        df = AlignmentResult.to_dataframe(results)
        
        print("\nSummary (averaged across experts):")
        summary_cols = ["align", "delta_vs_shuffle", "z_vs_shuffle", "effect_over_random"]
        print(df.groupby("k")[summary_cols].mean())
        
        print("\nTo compare with other runs, use:")
        print("  from src.alignment_analysis import compare_runs")
        print("  compare_runs(['results/file1.json', 'results/file2.json', ...])")
        
        # Print cos_squared summary for k=1
        if 1 in df["k"].values:
            k1_results = df[df["k"] == 1]
            if len(k1_results) > 0:
                print("\n" + "=" * 60)
                print("Cos^2(theta) Alignment Summary (k=1):")
                print("=" * 60)
                print(f"Mean cos^2(theta): {k1_results['cos_squared'].mean():.6f}")
                print(f"Max cos^2(theta):  {k1_results['cos_squared'].max():.6f}")
                print(f"Min cos^2(theta):  {k1_results['cos_squared'].min():.6f}")
                print(f"Std cos^2(theta):  {k1_results['cos_squared'].std():.6f}")
                print("\nPer-expert cos^2(theta) values:")
                for _, row in k1_results.iterrows():
                    print(f"  Expert {int(row['expert'])}: {row['cos_squared']:.6f}")
                print("=" * 60)


def main():
    """CLI entry point for running alignment analysis."""
    import argparse
    from src.models import Mixtral8x7B

    parser = argparse.ArgumentParser(
        description="Run SVD-based router-expert alignment analysis on MoE models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-l", "--layers",
        type=int,
        nargs="+",
        default=list(range(32)),  # All 32 layers (0-31)
        help="Layer number(s) to analyze (can specify multiple: -l 5 10 15). Default: all layers (0-31)",
    )
    parser.add_argument(
        "-k", "--k-list",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        help="List of k values (number of singular vectors)",
    )
    parser.add_argument(
        "-n", "--n-shuffles",
        type=int,
        default=200,
        help="Number of shuffle iterations for baseline",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving results to file",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing summary statistics",
    )

    args = parser.parse_args()

    # Initialize model and analyzer
    moe = Mixtral8x7B()
    analyzer = SVDAlignmentAnalyzer(
        k_list=tuple(args.k_list),
        n_shuffles=args.n_shuffles,
        seed=args.seed,
    )
    runner = AlignmentRunner(moe=moe, analyzer=analyzer)

    # Run analysis on all specified layers
    print(f"Running analysis on {len(args.layers)} layer(s): {args.layers}")
    print("=" * 70)
    
    saved_paths = []
    for layer in args.layers:
        results, saved_path = runner.run_full_analysis(
            layer=layer,
            save=not args.no_save,
            print_summary=not args.no_summary,
        )
        if saved_path:
            saved_paths.append(saved_path)
        print("=" * 70)
    
    # Summary of all saved files
    if saved_paths:
        print(f"\nSaved {len(saved_paths)} result file(s):")
        for path in saved_paths:
            print(f"  - {path}")
        print("\nTo compare results, use:")
        print("  from src.alignment_analysis import compare_runs, load_results")
        print(f"  runs = [load_results('{p}') for p in {[str(p) for p in saved_paths]}]")
        print("  compare_runs(runs)")


if __name__ == "__main__":
    main()

