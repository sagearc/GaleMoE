"""Results management for router-expert alignment analysis."""
from src.analysis.reporting.notebook_html import (
    generate_from_notebook,
    generate_notebook_html,
)
from src.analysis.reporting.result_analysis import (
    ResultAnalyzer,
    analyze_results,
)
from src.analysis.reporting.results import (
    ResultsManager,
    compare_runs,
    load_results,
    save_results,
)
from src.analysis.reporting.single_file_plots import (
    generate_single_file_plots,
)
from src.analysis.reporting.utils import (
    build_run_id,
    display_plot_in_jupyter,
    figure_to_base64,
    figure_to_bytes,
    get_layer_label,
    get_summary_columns,
)

__all__ = [
    "ResultAnalyzer",
    "ResultsManager",
    "analyze_results",
    "build_run_id",
    "compare_runs",
    "display_plot_in_jupyter",
    "figure_to_base64",
    "figure_to_bytes",
    "generate_from_notebook",
    "get_layer_label",
    "generate_notebook_html",
    "generate_single_file_plots",
    "get_summary_columns",
    "load_results",
    "save_results",
]

