"""Runner that coordinates repository and analyzer for router-expert alignment analysis."""
from __future__ import annotations

from typing import List, Optional, Sequence

from src.analysis.core.analyzer import AlignmentAnalyzer
from src.analysis.core.data_structures import AlignmentResult
from src.analysis.methods.svd.analyzer import SVDAlignmentAnalyzer
from src.analysis.storage.repository import MoEWeightsRepository
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
        # Default to SVD if no analyzer provided
        self.analyzer: AlignmentAnalyzer = analyzer or SVDAlignmentAnalyzer()

    def run_layer(self, layer: int) -> List[AlignmentResult]:
        """Run analysis on a single layer."""
        # Prefetch files for the specified layer (uses default max_workers=16 for fast downloads)
        self.repo.prefetch_layer(layer=layer)

        layer_w = self.repo.load_layer(layer)
        return self.analyzer.analyze_layer(layer_w)

    def run_layers(self, layers: Sequence[int]) -> List[AlignmentResult]:
        """Run analysis on multiple layers."""
        all_res: List[AlignmentResult] = []
        for l in layers:
            all_res.extend(self.run_layer(int(l)))
        return all_res

