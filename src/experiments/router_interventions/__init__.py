"""Router interventions: two experiments.

1. Project out: project out SVDs from all experts, compare loss.
2. Vector interventions: inject, subtract, project_out with many vectors;
   compare loss and token distributions (KL/CE, confusion matrix).

Requires precomputed SVD pickle files.

Usage::

    # Experiment 1: project out SVDs from all experts, compare loss
    python -m src.experiments.router_interventions.run_project_out --svd_dir /path/to/pkl --layer_idx 5

    # Experiment 2: vector interventions (inject, subtract, project_out), loss + token distributions
    python -m src.experiments.router_interventions.run_vector_interventions --svd_dir /path/to/pkl --layer_idx 5
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
    ProjectOutRunner,
    run_project_out_experiment,
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
    "ProjectOutRunner",
    "run_project_out_experiment",
    "ExpertVectors",
    "VectorIntervention",
    "load_results",
    "plot_ablation_results",
    "plot_confusion_heatmap",
    "plot_confusion_from_results",
    "run_vector_intervention_experiment",
]
