"""SVD alignment analyzer for MoE models."""
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Generator, Tensor

from src.svd.data_structures import AlignmentResult, LayerWeights
from src.svd.svd_cache import SVDCache


class SVDAlignmentAnalyzer:
    """Analyzes alignment between router vectors and expert weight matrices using SVD."""
    
    def __init__(
        self,
        k_list: Sequence[int] = (8, 16, 32, 64),
        n_shuffles: int = 200,
        seed: int = 0,
        svd_cache: Optional[SVDCache] = None,
    ):
        self.k_list = tuple(int(k) for k in k_list)
        self.n_shuffles = int(n_shuffles)
        self.seed = int(seed)  # Store seed for results metadata
        self.rng = Generator()
        self.rng.manual_seed(seed)
        self.svd_cache = svd_cache or SVDCache()  # Use shared cache by default

    @staticmethod
    def _align_proj_energy(router_vec: Tensor, V: Tensor, k: int) -> float:
        """
        Projection energy of normalized router vec onto top-k RIGHT singular subspace.
        router_vec: [d_model] - should already be normalized and on CPU
        V: [d_model, d_model] - precomputed right singular vectors from SVD
        k: number of top singular vectors to use
        """
        k = min(k, V.shape[1])
        Vk = V[:, :k]           # [d_model, k]
        proj = Vk.T @ router_vec  # [k] - router_vec already normalized
        return float((proj * proj).sum().clamp(0, 1).item())
    
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

    def _shuffle_stats(self, layer_w: LayerWeights, expert_Vs: List[Tensor], R: Tensor) -> Dict[int, Tuple[float, float]]:
        """
        Compute shuffle statistics using precomputed SVDs and pre-normalized router vectors.
        expert_Vs: Precomputed V matrices from SVD for each expert
        R: Pre-normalized router vectors [n_experts, d_model] already on CPU
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
            sd = float(vals_tensor.std().item() + 1e-12)
            stats[int(k)] = (mu, sd)

        return stats

    def analyze_layer(self, layer_w: LayerWeights) -> List[AlignmentResult]:
        """Analyze alignment for a single layer."""
        n_experts = len(layer_w.experts_w_in)
        d_model = int(layer_w.gate_w.shape[1])

        # Pre-normalize gate weights once (major speedup!)
        R = layer_w.gate_w.float().cpu()
        R = R / (R.norm(dim=1, keepdim=True) + 1e-12)  # [n_experts, d_model] normalized

        # Precompute SVDs for all experts ONCE (with caching across runs!)
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

        shuffle = self._shuffle_stats(layer_w, expert_Vs, R)

        results: List[AlignmentResult] = []
        for i in range(n_experts):
            for k in self.k_list:
                # Use precomputed V matrix and pre-normalized R[i]
                align = self._align_proj_energy(R[i], expert_Vs[i], k)

                random_expect = float(k / d_model)
                effect = float(align - random_expect)

                mu, sd = shuffle[int(k)]
                delta = float(align - mu)
                z = float(delta / sd)

                results.append(
                    AlignmentResult(
                        model_id=layer_w.model_id,
                        layer=layer_w.layer,
                        expert=i,
                        k=int(k),
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

