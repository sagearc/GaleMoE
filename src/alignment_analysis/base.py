"""Base classes and data structures for router-expert alignment analysis."""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
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
    cos_squared: float = 0.0  # cos^2(theta) for k=1, 0.0 otherwise
    
    @classmethod
    def to_dataframe(cls, results: List['AlignmentResult']) -> pd.DataFrame:
        """Convert list of results to DataFrame.
        
        Args:
            results: List of AlignmentResult objects
            
        Returns:
            DataFrame with all results
        """
        return pd.DataFrame([asdict(r) for r in results])


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


@dataclass
class AnalysisMetadata:
    """Metadata for an analysis run."""
    model_id: str
    layer: int
    timestamp: str
    datetime: str
    n_experts: int
    method: str
    k_list: List[int]
    n_shuffles: Optional[int] = None
    seed: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisMetadata':
        """Create AnalysisMetadata from a dictionary.
        
        Args:
            data: Dictionary containing metadata fields
            
        Returns:
            AnalysisMetadata instance
        """
        return cls(
            model_id=data["model_id"],
            layer=data["layer"],
            timestamp=data["timestamp"],
            datetime=data["datetime"],
            n_experts=data["n_experts"],
            method=data["method"],
            k_list=data["k_list"],
            n_shuffles=data.get("n_shuffles"),
            seed=data.get("seed"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}
    
    @property
    def run_id(self) -> str:
        """Build a consistent run identifier."""
        run_parts = [f"L{self.layer}"]
        if self.n_shuffles is not None:
            run_parts.append(f"s{self.n_shuffles}")
        if self.timestamp:
            run_parts.append(self.timestamp)
        return "_".join(run_parts)
    
    @property
    def layer_label(self) -> str:
        """Get a simple layer label for legends (no timestamp)."""
        return f"L{self.layer}"
    
    @staticmethod
    def extract_layer_number(layer_label: str) -> int:
        """Extract layer number from label like 'L5' -> 5, 'L10' -> 10.
        
        Args:
            layer_label: Layer label string (e.g., 'L5')
            
        Returns:
            Layer number, or 999999 if parsing fails
        """
        try:
            return int(layer_label.lstrip('L'))
        except (ValueError, AttributeError):
            return 999999
