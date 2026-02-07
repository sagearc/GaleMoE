"""Router-expert alignment analysis for Mixture of Experts models.

This module provides tools for analyzing the alignment between router vectors
and expert weight matrices in MoE models using SVD-based methods.

Main classes:
- AlignmentRunner: Orchestrates weight loading and analysis
- SVDAlignmentAnalyzer: SVD-based alignment analysis
- ResultsManager: Save/load/compare analysis results
- MoEWeightsRepository: Load weights from HuggingFace Hub

Data structures:
- LayerWeights: Container for layer weights (router + experts)
- AlignmentResult: Results from alignment analysis

Example usage:
    from src.alignment_analysis import AlignmentRunner, SVDAlignmentAnalyzer, ResultsManager
    from src.models import Mixtral8x7B

    # Create analyzer with custom k values
    analyzer = SVDAlignmentAnalyzer(k_list=[1, 2, 4, 8, 16, 32])
    
    # Create runner with model
    moe = Mixtral8x7B()
    runner = AlignmentRunner(moe, analyzer=analyzer)
    
    # Run analysis
    results = runner.run_layer(4)
    
    # Save results
    manager = ResultsManager()
    manager.save(results, analyzer, layer=4, model_id=moe.model_id)
"""

# Base classes and data structures
from src.alignment_analysis.base import (
    AlignmentAnalyzer,
    AlignmentResult,
    AnalysisMetadata,
    LayerWeights,
)

# SVD implementation
from src.alignment_analysis.svd import (
    SVDAlignmentAnalyzer,
    SVDCache,
)

# Weight loading
from src.alignment_analysis.loader import (
    MoEWeightsRepository,
    ShardCache,
    check_download_config,
    print_download_diagnostics,
)

# Analysis orchestration
from src.alignment_analysis.analysis_runner import AlignmentRunner

# Results management
from src.alignment_analysis.results import (
    ResultAnalyzer,
    ResultsManager,
    analyze_results,
    compare_runs,
    load_results,
    save_results,
)

# Visualization
from src.alignment_analysis.plots import (
    PlotStyle,
    display_plot,
    get_summary_columns,
    plot_alignment_vs_k,
    plot_zscore_vs_k,
    plot_delta_vs_k,
    plot_effect_vs_k,
    plot_expert_heatmap,
    plot_cos_squared_per_expert,
    plot_summary,
    plot_expert_analysis,
    plot_full_analysis,
    plot_comparison,
    plot_shuffle_comparison,
    get_layer_label,
    extract_layer_number,
)

__all__ = [
    # Base
    "AlignmentAnalyzer",
    "AlignmentResult",
    "AnalysisMetadata",
    "LayerWeights",
    # SVD
    "SVDAlignmentAnalyzer",
    "SVDCache",
    # Loader
    "MoEWeightsRepository",
    "ShardCache",
    "check_download_config",
    "print_download_diagnostics",
    # Runner
    "AlignmentRunner",
    # Results
    "ResultAnalyzer",
    "ResultsManager",
    "analyze_results",
    "compare_runs",
    "load_results",
    "save_results",
    # Plots
    "PlotStyle",
    "display_plot",
    "get_summary_columns",
    "plot_alignment_vs_k",
    "plot_zscore_vs_k",
    "plot_delta_vs_k",
    "plot_effect_vs_k",
    "plot_expert_heatmap",
    "plot_cos_squared_per_expert",
    "plot_summary",
    "plot_expert_analysis",
    "plot_full_analysis",
    "plot_comparison",
    "plot_shuffle_comparison",
    "get_layer_label",
    "extract_layer_number",
]
