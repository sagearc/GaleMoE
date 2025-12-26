"""Runner that coordinates repository and analyzer."""
from typing import List, Optional, Sequence

from src.svd.repository import MoEWeightsRepository
from src.svd.svd_analyzer import SVDAlignmentAnalyzer
from src.svd.data_structures import AlignmentResult


class SVDMilestoneRunner:
    """Coordinates weight loading and analysis."""
    
    def __init__(
        self,
        moe,
        repo: Optional[MoEWeightsRepository] = None,
        analyzer: Optional[SVDAlignmentAnalyzer] = None
    ):
        self.moe = moe
        self.repo = repo or MoEWeightsRepository(moe)
        self.analyzer = analyzer or SVDAlignmentAnalyzer()

    def run_layer(self, layer: int) -> List[AlignmentResult]:
        """Run analysis on a single layer."""
        # Prefetch files for the specified layer
        self.repo.prefetch_layer(layer=layer, max_workers=1)

        layer_w = self.repo.load_layer(layer)
        return self.analyzer.analyze_layer(layer_w)

    def run_layers(self, layers: Sequence[int]) -> List[AlignmentResult]:
        """Run analysis on multiple layers."""
        all_res: List[AlignmentResult] = []
        for l in layers:
            all_res.extend(self.run_layer(int(l)))
        return all_res

