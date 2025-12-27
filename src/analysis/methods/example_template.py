"""Example template for implementing a new alignment method.

To add a new alignment method:
1. Create a new file in src/analysis/methods/{method_name}/analyzer.py
2. Inherit from AlignmentAnalyzer and implement the required methods
3. Use it with AlignmentRunner - no other changes needed!

Shuffles are OPTIONAL - only implement if your method uses them.
Different methods can use different statistical comparison approaches:
- Shuffles (like SVD method)
- Analytical distributions
- Bootstrap
- Or any other method-specific approach

Example with shuffles:
    from src.analysis import AlignmentRunner
    from src.analysis.methods.pca.analyzer import PCAAnalyzer
    
    analyzer = PCAAnalyzer(k_list=(8, 16, 32), n_shuffles=100, seed=42)
    runner = AlignmentRunner(moe=moe, analyzer=analyzer)
    results = runner.run_layer(layer=10)

Example without shuffles:
    analyzer = SimpleAnalyzer(k_list=(8, 16, 32))  # No shuffles needed
    runner = AlignmentRunner(moe=moe, analyzer=analyzer)
    results = runner.run_layer(layer=10)
"""
from typing import List, Optional, Sequence

from src.analysis.core.analyzer import AlignmentAnalyzer
from src.analysis.core.data_structures import AlignmentResult, LayerWeights


class ExampleAlignmentAnalyzer(AlignmentAnalyzer):
    """Example implementation of a new alignment method.
    
    This is a template showing the minimal interface required.
    Replace this with your actual alignment logic.
    
    NOTE: Shuffles are OPTIONAL. Only include n_shuffles/seed if your
    method uses shuffling for statistical comparison.
    """
    
    def __init__(
        self,
        k_list: Sequence[int] = (8, 16, 32, 64),
        # Shuffles are OPTIONAL - only include if your method uses them
        n_shuffles: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the analyzer.
        
        Args:
            k_list: List of k values to analyze
            n_shuffles: (Optional) Number of shuffles for statistical comparison
            seed: (Optional) Random seed for reproducibility
        """
        self._k_list = tuple(int(k) for k in k_list)
        # Only store if provided (shuffles are optional)
        if n_shuffles is not None:
            self._n_shuffles = int(n_shuffles)
        if seed is not None:
            self._seed = int(seed)
        # Initialize any method-specific components here
    
    @property
    def method_name(self) -> str:
        """Return the name of the alignment method."""
        return "example"  # Change to your method name
    
    @property
    def k_list(self) -> Sequence[int]:
        """Return the list of k values used in the analysis."""
        return self._k_list
    
    # Optional properties - only implement if your method uses shuffles
    # @property
    # def n_shuffles(self) -> int:
    #     """Return the number of shuffles used for statistical comparison."""
    #     return self._n_shuffles
    #
    # @property
    # def seed(self) -> int:
    #     """Return the random seed used for reproducibility."""
    #     return self._seed
    
    def analyze_layer(self, layer_w: LayerWeights) -> List[AlignmentResult]:
        """Analyze alignment for a single layer.
        
        This is where you implement your alignment method.
        You need to:
        1. Process the layer weights (router + experts)
        2. Compute alignment scores for each expert-k combination
        3. Compute shuffle statistics for comparison
        4. Return a list of AlignmentResult objects
        
        Args:
            layer_w: Layer weights containing router and expert weights
            
        Returns:
            List of alignment results for all expert-k combinations
        """
        # TODO: Implement your alignment method here
        # This is just a placeholder that returns empty results
        
        results: List[AlignmentResult] = []
        n_experts = len(layer_w.experts_w_in)
        d_model = int(layer_w.gate_w.shape[1])
        
        for i in range(n_experts):
            for k in self.k_list:
                # TODO: Compute your alignment score here
                align = 0.0  # Placeholder
                random_expect = float(k / d_model)
                effect = float(align - random_expect)
                
                # TODO: Compute statistical comparison
                # This depends on your method:
                # - If using shuffles: compute shuffle_mean, shuffle_std
                # - If using analytical: compute from theoretical distribution
                # - If using bootstrap: compute from bootstrap samples
                # - Or any other approach
                shuffle_mean = 0.0  # Placeholder - replace with your method
                shuffle_std = 1.0  # Placeholder - replace with your method
                delta = float(align - shuffle_mean)
                z = float(delta / shuffle_std) if shuffle_std > 0 else 0.0
                
                results.append(
                    AlignmentResult(
                        model_id=layer_w.model_id,
                        layer=layer_w.layer,
                        expert=i,
                        k=int(k),
                        align=align,
                        random_expect_k_over_d=random_expect,
                        effect_over_random=effect,
                        shuffle_mean=shuffle_mean,
                        shuffle_std=shuffle_std,
                        delta_vs_shuffle=delta,
                        z_vs_shuffle=z,
                    )
                )
        
        return results

