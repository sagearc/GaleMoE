"""Base interface for alignment analyzers."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from src.analysis.core.data_structures import AlignmentResult, LayerWeights


class AlignmentAnalyzer(ABC):
    """Abstract base class for router-expert alignment analyzers.
    
    To implement a new alignment method, inherit from this class and implement:
    - `analyze_layer()`: The main analysis method
    - `method_name`: Property returning the method name (e.g., "svd", "pca", etc.)
    - `k_list`: Property returning the list of k values used
    
    Optional methods (for methods that use shuffles/randomization):
    - `n_shuffles`: Property returning the number of shuffles (if applicable)
    - `seed`: Property returning the random seed (if applicable)
    
    Methods can use any statistical comparison approach - shuffles are just one option.
    """
    
    @abstractmethod
    def analyze_layer(self, layer_w: LayerWeights) -> List[AlignmentResult]:
        """Analyze alignment for a single layer.
        
        Args:
            layer_w: Layer weights containing router and expert weights
            
        Returns:
            List of alignment results for all expert-k combinations.
            Methods can use any statistical comparison approach:
            - Shuffles (set shuffle_mean, shuffle_std, delta_vs_shuffle, z_vs_shuffle)
            - Analytical distributions (set appropriate fields)
            - Bootstrap (set appropriate fields)
            - Or any other method-specific approach
        """
        pass
    
    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the name of the alignment method (e.g., 'svd', 'pca')."""
        pass
    
    @property
    @abstractmethod
    def k_list(self) -> Sequence[int]:
        """Return the list of k values used in the analysis."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return method-specific metadata for results saving.
        
        Override this to provide custom metadata. Default implementation
        includes optional shuffle-related fields if they exist.
        
        Returns:
            Dictionary of metadata key-value pairs
        """
        metadata = {
            "method": self.method_name,
            "k_list": list(self.k_list),
        }
        
        # Add optional shuffle-related metadata if available
        if hasattr(self, 'n_shuffles'):
            metadata["n_shuffles"] = self.n_shuffles
        if hasattr(self, 'seed'):
            metadata["seed"] = self.seed
        
        return metadata

