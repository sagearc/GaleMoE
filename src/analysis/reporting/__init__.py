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

__all__ = [
    "ResultAnalyzer",
    "ResultsManager",
    "analyze_results",
    "compare_runs",
    "generate_from_notebook",
    "generate_notebook_html",
    "generate_single_file_plots",
    "load_results",
    "save_results",
]

