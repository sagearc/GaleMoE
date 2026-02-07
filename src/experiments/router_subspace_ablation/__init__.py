"""Router subspace ablation: project out SVD/orthogonal/random from router rows.

Measure loss delta when removing expert-derived directions from the router.
Requires precomputed expert SVD vectors (pickle files).

Usage::

    python -m src.experiments.router_subspace_ablation.run --svd_dir /path/to/pkl --layer_idx 5
"""
from .config import ExperimentConfig
from .data import BatchLoader, TextListBatchLoader, WikitextBatchLoader
from .evaluation import LossEvaluator
from .plotter import load_results, plot_ablation_results
from .router_manager import RouterManager
from .runner import SubspaceAblationRunner, run_ablation_experiment
from .vectors import ExpertVectors, VectorIntervention

__all__ = [
    "ExperimentConfig",
    "BatchLoader",
    "WikitextBatchLoader",
    "TextListBatchLoader",
    "LossEvaluator",
    "RouterManager",
    "SubspaceAblationRunner",
    "run_ablation_experiment",
    "ExpertVectors",
    "VectorIntervention",
    "load_results",
    "plot_ablation_results",
]
