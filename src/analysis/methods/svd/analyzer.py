"""SVD-based alignment analyzer for router-expert analysis."""
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Generator, Tensor

from src.analysis.core.analyzer import AlignmentAnalyzer
from src.analysis.core.data_structures import AlignmentResult, LayerWeights
from src.analysis.storage.cache import SVDCache

# Constants
_EPSILON = 1e-12  # Small epsilon for numerical stability


class SVDAlignmentAnalyzer(AlignmentAnalyzer):
    """Analyzes alignment between router vectors and expert weight matrices using SVD."""
    
    def __init__(
        self,
        k_list: Sequence[int] = (8, 16, 32, 64),
        n_shuffles: int = 200,
        seed: int = 0,
        svd_cache: Optional[SVDCache] = None,
    ):
        self._k_list = tuple(int(k) for k in k_list)
        self._n_shuffles = int(n_shuffles)
        self._seed = int(seed)  # Store seed for results metadata
        self.rng = Generator()
        self.rng.manual_seed(seed)
        self.svd_cache = svd_cache or SVDCache()  # Use shared cache by default
    
    @property
    def method_name(self) -> str:
        """Return the name of the alignment method."""
        return "svd"
    
    @property
    def k_list(self) -> Sequence[int]:
        """Return the list of k values used in the analysis."""
        return self._k_list
    
    @property
    def n_shuffles(self) -> int:
        """Return the number of shuffles used for statistical comparison."""
        return self._n_shuffles
    
    @property
    def seed(self) -> int:
        """Return the random seed used for reproducibility."""
        return self._seed

    @staticmethod
    def _align_proj_energy(router_vec: Tensor, V: Tensor, k: int) -> float:
        """
        Projection energy of normalized router vec onto top-k RIGHT singular subspace.
        router_vec: [d_model] - should already be normalized and on CPU
        V: [d_model, d_model] - precomputed right singular vectors from SVD
        k: number of top singular vectors to use
        """
        k = min(k, V.shape[1])
        Vk = V[:, :k]  # [d_model, k]
        proj = Vk.T @ router_vec  # [k] - router_vec already normalized
        return float((proj * proj).sum().clamp(0.0, 1.0).item())
    
    @staticmethod
    def _compute_svd(w_in: Tensor) -> Tensor:
        """
        Precompute SVD once for an expert weight matrix.
        Returns V matrix (right singular vectors).
        """
        W = w_in.float().cpu()
        _, _, Vh = torch.linalg.svd(W, full_matrices=False)
        V = Vh.transpose(0, 1)  # [d_model, d_model]
        return V

    def _shuffle_stats(
        self, 
        layer_w: LayerWeights, 
        expert_Vs: List[Tensor], 
        R: Tensor
    ) -> Dict[int, Tuple[float, float]]:
        """Compute shuffle statistics using precomputed SVDs and pre-normalized router vectors.
        
        Args:
            layer_w: Layer weights (used for number of experts)
            expert_Vs: Precomputed V matrices from SVD for each expert
            R: Pre-normalized router vectors [n_experts, d_model] already on CPU
            
        Returns:
            Dictionary mapping k values to (mean, std) tuples of shuffle statistics
        """
        n = len(layer_w.experts_w_in)
        stats: Dict[int, Tuple[float, float]] = {}

        for k in self.k_list:
            vals = []
            for _ in range(self.n_shuffles):
                perm = torch.randperm(n, generator=self.rng)
                for i in range(n):
                    # Use precomputed V matrix and pre-normalized R[i]
                    vals.append(self._align_proj_energy(R[i], expert_Vs[int(perm[i])], k))
            vals_tensor = torch.tensor(vals)
            mu = float(vals_tensor.mean().item())
            sd = float(vals_tensor.std().item() + _EPSILON)
            stats[int(k)] = (mu, sd)

        return stats

    def _normalize_router_vectors(self, gate_w: Tensor) -> Tensor:
        """Normalize router (gate) vectors for efficient computation.
        
        Args:
            gate_w: Gate weight matrix [n_experts, d_model]
            
        Returns:
            Normalized router vectors [n_experts, d_model]
        """
        R = gate_w.float().cpu()
        R = R / (R.norm(dim=1, keepdim=True) + _EPSILON)
        return R
    
    def _compute_expert_svds(self, layer_w: LayerWeights) -> List[Tensor]:
        """Compute or load SVD decompositions for all experts with caching.
        
        Args:
            layer_w: Layer weights containing expert weights
            
        Returns:
            List of V matrices (right singular vectors) for each expert
        """
        n_experts = len(layer_w.experts_w_in)
        print(f"Precomputing SVDs for layer {layer_w.layer} ({n_experts} experts)...")
        print(f"  Cache directory: {self.svd_cache.cache_dir.absolute()}")
        
        expert_Vs = []
        cached_count = 0
        
        for i, w_in in enumerate(layer_w.experts_w_in):
            # Try to get from cache first
            cached_V = self.svd_cache.get(layer_w.model_id, layer_w.layer, i, w_in)
            if cached_V is not None:
                expert_Vs.append(cached_V)
                cached_count += 1
                print(f"  Expert {i}: Loaded from cache")
            else:
                # Compute and cache
                print(f"  Expert {i}: Computing SVD...")
                V = self._compute_svd(w_in)
                self.svd_cache.put(layer_w.model_id, layer_w.layer, i, V)
                expert_Vs.append(V)
                cache_file = self.svd_cache._get_cache_file(layer_w.model_id, layer_w.layer, i)
                print(f"    → Saved to {cache_file.name}")
        
        if cached_count > 0:
            print(f"✓ Loaded {cached_count}/{n_experts} from cache, computed {n_experts - cached_count} new")
        else:
            print(f"✓ Computed {len(expert_Vs)} SVDs (cached for future use)")
        
        return expert_Vs
    
    def _compute_alignment_scores(
        self,
        expert_Vs: List[Tensor],
        R: Tensor,
    ) -> List[Dict[str, float]]:
        """Compute alignment scores (projection energy) for all expert-k combinations.
        
        Args:
            expert_Vs: Precomputed V matrices for each expert
            R: Pre-normalized router vectors [n_experts, d_model]
            
        Returns:
            List of dictionaries containing alignment scores (expert, k, align)
        """
        n_experts = len(expert_Vs)
        scores = []
        
        for i in range(n_experts):
            for k in self.k_list:
                # Compute alignment score (projection energy)
                align = self._align_proj_energy(R[i], expert_Vs[i], k)
                
                scores.append({
                    "expert": i,
                    "k": int(k),
                    "align": align,
                })
        
        return scores
    
    def _create_alignment_results(
        self,
        layer_w: LayerWeights,
        scores: List[Dict[str, float]],
        shuffle_stats: Dict[int, Tuple[float, float]],
        d_model: int,
    ) -> List[AlignmentResult]:
        """Create AlignmentResult objects from alignment scores.
        
        Computes all derived statistics (random expectation, effect, shuffle comparison, etc.)
        from the raw alignment scores.
        
        Args:
            layer_w: Layer weights (for model_id and layer info)
            scores: List of score dictionaries from _compute_alignment_scores (expert, k, align)
            shuffle_stats: Shuffle statistics dict mapping k -> (mean, std)
            d_model: Model dimension
            
        Returns:
            List of AlignmentResult objects with all computed statistics
        """
        results = []
        
        for score in scores:
            align = score["align"]
            k = score["k"]
            expert = score["expert"]
            
            # Compute baseline expectations
            random_expect = float(k / d_model)
            effect = float(align - random_expect)
            
            # Get shuffle statistics
            mu, sd = shuffle_stats[k]
            delta = float(align - mu)
            z = float(delta / sd) if sd > _EPSILON else 0.0
            
            results.append(
                AlignmentResult(
                    model_id=layer_w.model_id,
                    layer=layer_w.layer,
                    expert=expert,
                    k=k,
                    align=align,
                    random_expect_k_over_d=random_expect,
                    effect_over_random=effect,
                    shuffle_mean=mu,
                    shuffle_std=sd,
                    delta_vs_shuffle=delta,
                    z_vs_shuffle=z,
                )
            )
        
        return results
    
    def analyze_layer(self, layer_w: LayerWeights) -> List[AlignmentResult]:
        """Analyze alignment for a single layer.
        
        This method orchestrates the analysis pipeline:
        1. Normalize router vectors
        2. Compute/load expert SVDs
        3. Compute shuffle statistics
        4. Compute alignment results
        
        Args:
            layer_w: Layer weights containing router and expert weights
            
        Returns:
            List of alignment results for all expert-k combinations
        """
        # Step 1: Normalize router vectors once
        R = self._normalize_router_vectors(layer_w.gate_w)
        
        # Step 2: Compute or load SVDs for all experts (with caching)
        expert_Vs = self._compute_expert_svds(layer_w)
        
        # Step 3: Compute shuffle statistics for null distribution
        shuffle_stats = self._shuffle_stats(layer_w, expert_Vs, R)
        
        # Step 4: Compute alignment scores (projection energy only)
        scores = self._compute_alignment_scores(expert_Vs, R)
        
        # Step 5: Create result objects with all derived statistics
        d_model = int(layer_w.gate_w.shape[1])
        results = self._create_alignment_results(
            layer_w, scores, shuffle_stats, d_model
        )
        
        return results

