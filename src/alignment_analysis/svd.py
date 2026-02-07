"""SVD-based alignment analyzer for router-expert analysis."""
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Generator, Tensor

from src.alignment_analysis.base import AlignmentAnalyzer, AlignmentResult, LayerWeights
from src.utils import OutputDir, sanitize_model_id

# Constants
_EPSILON = 1e-12  # Small epsilon for numerical stability


class SVDCache:
    """Cache for SVD decompositions across the entire network.
    
    Persists to disk so it can be reused across runs.
    Cache directory is always created relative to the project root (GaleMoE/),
    regardless of where the script is run from.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize SVD cache.
        
        Args:
            cache_dir: Optional cache directory path. If None, uses 'svd_cache' 
                      relative to project root. If relative path provided,
                      it's relative to project root. If absolute path provided, uses as-is.
        """
        self._cache_dir = OutputDir.resolve(cache_dir, default_subdir="svd_cache")
        self._memory_cache: Dict[Tuple[str, int, int], Tensor] = {}
    
    @property
    def cache_dir(self) -> Path:
        return self._cache_dir.path
    
    def _get_cache_key(self, model_id: str, layer: int, expert: int) -> Tuple[str, int, int]:
        return (model_id, layer, expert)
    
    def _get_cache_file(self, model_id: str, layer: int, expert: int) -> Path:
        safe_model_id = sanitize_model_id(model_id)
        filename = f"{safe_model_id}_layer{layer}_expert{expert}.pkl"
        return self.cache_dir / filename
    
    def get(self, model_id: str, layer: int, expert: int, w_in: Tensor) -> Optional[Tensor]:
        """Get cached V matrix, or None if not cached.
        
        Args:
            model_id: Model identifier
            layer: Layer number
            expert: Expert index
            w_in: Expert weight matrix (unused, kept for API compatibility)
            
        Returns:
            Cached V matrix if available, None otherwise
        """
        key = self._get_cache_key(model_id, layer, expert)
        
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        cache_file = self._get_cache_file(model_id, layer, expert)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    V = pickle.load(f)
                self._memory_cache[key] = V
                return V
            except Exception as e:
                print(f"Warning: Failed to load cache for {cache_file}: {e}")
        
        return None
    
    def put(self, model_id: str, layer: int, expert: int, V: Tensor) -> None:
        """Cache V matrix to both memory and disk.
        
        Args:
            model_id: Model identifier
            layer: Layer number
            expert: Expert index
            V: V matrix (right singular vectors) to cache
        """
        key = self._get_cache_key(model_id, layer, expert)
        self._memory_cache[key] = V
        
        cache_file = self._get_cache_file(model_id, layer, expert)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(V, f)
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_file}: {e}")
    
    def clear(self) -> None:
        """Clear memory cache (disk cache remains)."""
        self._memory_cache.clear()
    
    def clear_all(self) -> None:
        """Clear both memory and disk cache."""
        self._memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print(f"Cleared all SVD cache files from {self.cache_dir}")


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
        self._seed = int(seed)
        self.rng = Generator()
        self.rng.manual_seed(seed)
        self.svd_cache = svd_cache or SVDCache()
    
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
        """Projection energy of normalized router vec onto top-k RIGHT singular subspace."""
        k = min(k, V.shape[1])
        Vk = V[:, :k]
        proj = Vk.T @ router_vec
        return float((proj * proj).sum().clamp(0.0, 1.0).item())
    
    @staticmethod
    def _align_cos_squared(router_vec: Tensor, V: Tensor) -> float:
        """Squared cosine of angle between router vector and first singular vector."""
        v1 = V[:, 0]
        cos_theta_tensor = router_vec @ v1
        cos_squared_tensor = (cos_theta_tensor * cos_theta_tensor).clamp(0.0, 1.0)
        return float(cos_squared_tensor.item())
    
    @staticmethod
    def _compute_svd(w_in: Tensor) -> Tensor:
        """Precompute SVD once for an expert weight matrix. Returns V matrix."""
        W = w_in.float().cpu()
        _, _, Vh = torch.linalg.svd(W, full_matrices=False)
        return Vh.transpose(0, 1)

    def _shuffle_stats(
        self, 
        layer_w: LayerWeights, 
        expert_Vs: List[Tensor], 
        R: Tensor
    ) -> Dict[int, Tuple[float, float]]:
        """Compute shuffle statistics using precomputed SVDs and pre-normalized router vectors."""
        n = len(layer_w.experts_w_in)
        stats: Dict[int, Tuple[float, float]] = {}

        for k in self.k_list:
            vals = []
            for _ in range(self.n_shuffles):
                perm = torch.randperm(n, generator=self.rng)
                for i in range(n):
                    vals.append(self._align_proj_energy(R[i], expert_Vs[int(perm[i])], k))
            vals_tensor = torch.tensor(vals)
            mu = float(vals_tensor.mean().item())
            sd = float(vals_tensor.std().item() + _EPSILON)
            stats[int(k)] = (mu, sd)

        return stats

    def _normalize_router_vectors(self, gate_w: Tensor) -> Tensor:
        """Normalize router (gate) vectors for efficient computation."""
        R = gate_w.float().cpu()
        return R / (R.norm(dim=1, keepdim=True) + _EPSILON)
    
    def _compute_expert_svds(self, layer_w: LayerWeights) -> List[Tensor]:
        """Compute or load SVD decompositions for all experts with caching."""
        n_experts = len(layer_w.experts_w_in)
        print(f"Precomputing SVDs for layer {layer_w.layer} ({n_experts} experts)...")
        print(f"  Cache directory: {self.svd_cache.cache_dir.absolute()}")
        
        expert_Vs = []
        cached_count = 0
        
        for i, w_in in enumerate(layer_w.experts_w_in):
            cached_V = self.svd_cache.get(layer_w.model_id, layer_w.layer, i, w_in)
            if cached_V is not None:
                expert_Vs.append(cached_V)
                cached_count += 1
                print(f"  Expert {i}: Loaded from cache")
            else:
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
        """Compute alignment scores (projection energy) for all expert-k combinations."""
        n_experts = len(expert_Vs)
        scores = []
        
        for i in range(n_experts):
            for k in self.k_list:
                align = self._align_proj_energy(R[i], expert_Vs[i], k)
                score_dict = {"expert": i, "k": int(k), "align": align}
                
                if k == 1:
                    cos_squared = self._align_cos_squared(R[i], expert_Vs[i])
                    score_dict["cos_squared"] = cos_squared
                
                scores.append(score_dict)
        
        return scores
    
    def _create_alignment_results(
        self,
        layer_w: LayerWeights,
        scores: List[Dict[str, float]],
        shuffle_stats: Dict[int, Tuple[float, float]],
        d_model: int,
    ) -> List[AlignmentResult]:
        """Create AlignmentResult objects from alignment scores."""
        results = []
        
        for score in scores:
            align = score["align"]
            k = score["k"]
            expert = score["expert"]
            cos_squared = score.get("cos_squared", 0.0)
            
            random_expect = float(k / d_model)
            effect = float(align - random_expect)
            
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
                    cos_squared=cos_squared,
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
        """
        R = self._normalize_router_vectors(layer_w.gate_w)
        expert_Vs = self._compute_expert_svds(layer_w)
        shuffle_stats = self._shuffle_stats(layer_w, expert_Vs, R)
        scores = self._compute_alignment_scores(expert_Vs, R)
        d_model = int(layer_w.gate_w.shape[1])
        return self._create_alignment_results(layer_w, scores, shuffle_stats, d_model)
