"""Main entry point for router-expert alignment analysis."""
from src.models import Mixtral8x7B
from src.analysis import (
    SVDAlignmentAnalyzer,
    AlignmentRunner,
    save_results,
)

# -----------------------------
# Example usage with your Mixtral8x7B
# -----------------------------
if __name__ == "__main__":
    moe = Mixtral8x7B()

    # analyzer = SVDAlignmentAnalyzer(k_list=(8, 16, 32), n_shuffles=10, seed=0)
    analyzer = SVDAlignmentAnalyzer(k_list=(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096), n_shuffles=200, seed=0)

    runner = AlignmentRunner(moe=moe, analyzer=analyzer)

    LAYER = 10
    print(f"Running analysis on Layer {LAYER}")
    results = runner.run_layer(layer=LAYER)

    # Save results with metadata
    saved_path = save_results(results, analyzer, LAYER, moe.model_id)
    
    # quick summary
    import pandas as pd
    pd.set_option('display.max_columns', 20)
    df = pd.DataFrame([r.__dict__ for r in results])
    print("\nSummary:")
    print(df.groupby("k")[["align", "delta_vs_shuffle", "z_vs_shuffle", "effect_over_random"]].mean())
    
    print(f"\nTo compare with other runs, use:")
    print(f"  from src.analysis import compare_runs")
    print(f"  compare_runs(['results/file1.json', 'results/file2.json', ...])")