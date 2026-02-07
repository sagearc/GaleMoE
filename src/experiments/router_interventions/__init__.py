"""Router interventions: project out, inject, or subtract directions from router rows.

Measure loss and token-distribution change (KL/CE) when modifying router weights
using expert SVD directions (or orthogonal/random). Requires precomputed SVD pickle files.

Usage::

    python -m src.experiments.router_interventions.run --svd_dir /path/to/pkl --layer_idx 5
    python -m src.experiments.router_interventions.run_vector_intervention --svd_dir /path/to/pkl --layer_idx 5
"""
from .core import (
    ExperimentConfig,
    BatchLoader,
    TextListBatchLoader,
    WikitextBatchLoader,
    LossEvaluator,
    TokenDistributionComparator,
    RouterManager,
    ExpertVectors,
    VectorIntervention,
)
from .runners import (
    SubspaceAblationRunner,
    run_ablation_experiment,
    run_vector_intervention_experiment,
)
from .viz import (
    load_results,
    plot_ablation_results,
    plot_confusion_heatmap,
    plot_confusion_from_results,
)

__all__ = [
    "ExperimentConfig",
    "BatchLoader",
    "WikitextBatchLoader",
    "TextListBatchLoader",
    "LossEvaluator",
    "TokenDistributionComparator",
    "RouterManager",
    "SubspaceAblationRunner",
    "run_ablation_experiment",
    "ExpertVectors",
    "VectorIntervention",
    "load_results",
    "plot_ablation_results",
    "plot_confusion_heatmap",
    "plot_confusion_from_results",
    "run_vector_intervention_experiment",
]
