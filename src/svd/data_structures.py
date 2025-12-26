"""Data structures for SVD alignment analysis."""
from dataclasses import dataclass
from typing import List

from torch import Tensor


@dataclass(frozen=True)
class LayerWeights:
    """Container for layer weights (router + experts)."""
    model_id: str
    layer: int
    gate_w: Tensor                 # [n_experts, d_model]
    experts_w_in: List[Tensor]     # list of n_experts tensors, each [d_ff, d_model]


@dataclass(frozen=True)
class AlignmentResult:
    """Results from alignment analysis for a single expert-k combination."""
    model_id: str
    layer: int
    expert: int
    k: int
    align: float
    random_expect_k_over_d: float
    effect_over_random: float
    shuffle_mean: float
    shuffle_std: float
    delta_vs_shuffle: float
    z_vs_shuffle: float

