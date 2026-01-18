"""Random vector-based alignment analyzer for router-expert analysis."""
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Generator, Tensor

from src.analysis.core.analyzer import AlignmentAnalyzer
from src.analysis.core.data_structures import AlignmentResult, LayerWeights

# Constants
_EPSILON = 1e-8  # Small epsilon for numerical stability


class RandomVectorAlignmentAnalyzer(AlignmentAnalyzer):
    """
    Alignment analyzer that uses random k vectors sampled from expert weight matrices.
    
    Instead of using SVD to extract principal directions, this method samples k random
    rows from each expert's input weight matrix, normalizes them, and uses them to compute
    alignment with router vectors. This provides an alternative perspective on router-expert
    alignment that doesn't rely on principal component analysis.
    """
    
    def __init__(
        self,
        k_list: Sequence[int] = (8, 16, 32, 64),
        n_shuffles: int = 200,
        seed: int = 0,
    ):
        """
        Initialize random vector alignment analyzer.
        
        Args:
            k_list: List of k values (number of random vectors to sample per expert)
            n_shuffles: Number of shuffles for statistical comparison
            seed: Random seed for reproducibility
        """
        self._k_list = tuple(int(k) for k in k_list)
        self._n_shuffles = int(n_shuffles)
        self._seed = int(seed)  # Store seed for results metadata
        self.rng = Generator()
        self.rng.manual_seed(seed)
    
    @property
    def method_name(self) -> str:
        """Return the name of the alignment method."""
        return "random_vectors"
    
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
    def _align_proj_energy(router_vec: Tensor, random_vectors: Tensor) -> float:
        """
        Projection energy of normalized router vec onto random vectors subspace.
        
        Similar to SVD projection energy, but uses random vectors instead of singular vectors.
        The random_vectors matrix should be [d_model, k] where each column is a normalized
        random vector from the expert weight matrix.
        
        Args:
            router_vec: [d_model] - should already be normalized and on CPU
            random_vectors: [d_model, k] - k normalized random vectors from expert weight matrix
            
        Returns:
            Projection energy (sum of squared projections), clamped to [0, 1]
        """
        k = random_vectors.shape[1]
        if k == 0:
            return 0.0
        
        # Project router vector onto each random vector and sum squared projections
        proj = random_vectors.T @ router_vec  # [k] - router_vec already normalized
        return float((proj * proj).sum().clamp(0.0, 1.0).item())
    
    @staticmethod
    def _align_cos_squared(router_vec: Tensor, random_vectors: Tensor) -> float:
        """
        Squared cosine of angle between router vector and first random vector.
        
        For k=1, this is equivalent to the projection energy, but computed
        more directly as the squared dot product.
        
        Args:
            router_vec: [d_model] - should already be normalized and on CPU
            random_vectors: [d_model, k] - k normalized random vectors (only first is used)
            
        Returns:
            cos^2(theta) where theta is angle between router_vec and first random vector
        """
        if random_vectors.shape[1] == 0:
            return 0.0
        
        v1 = random_vectors[:, 0]  # First random vector [d_model]
        # Since router_vec is normalized, dot product = cos(theta)
        cos_theta_tensor = router_vec @ v1  # [1] tensor
        cos_squared_tensor = (cos_theta_tensor * cos_theta_tensor).clamp(0.0, 1.0)
        return float(cos_squared_tensor.item())
    
    def _sample_random_vectors(
        self,
        w_in: Tensor,
        k: int,
    ) -> Tensor:
        """
        Sample k random rows from expert weight matrix and normalize them.
        
        Args:
            w_in: Expert input weight matrix [d_ff, d_model]
            k: Number of random vectors to sample
            
        Returns:
            Matrix [d_model, k] where each column is a normalized random vector
        """
        d_ff, d_model = w_in.shape
        
        # Sample k random row indices (with replacement if k > d_ff)
        if k > d_ff:
            # Sample with replacement
            row_indices = torch.randint(0, d_ff, (k,), generator=self.rng)
        else:
            # Sample without replacement for better diversity
            row_indices = torch.randperm(d_ff, generator=self.rng)[:k]
        
        # Extract random rows and transpose to get [d_model, k]
        random_rows = w_in[row_indices, :].float().cpu()  # [k, d_model]
        random_vectors = random_rows.T  # [d_model, k]
        
        # Normalize each column (each random vector)
        norms = random_vectors.norm(dim=0, keepdim=True)  # [1, k]
        random_vectors = random_vectors / (norms + _EPSILON)
        
        return random_vectors
    
    def _compute_expert_random_vectors(
        self,
        layer_w: LayerWeights,
        k: int,
    ) -> List[Tensor]:
        """
        Compute random vectors for all experts at a given k.
        
        Args:
            layer_w: Layer weights containing expert weights
            k: Number of random vectors to sample per expert
            
        Returns:
            List of random vector matrices [d_model, k] for each expert
        """
        expert_random_vecs = []
        
        for w_in in layer_w.experts_w_in:
            random_vecs = self._sample_random_vectors(w_in, k)
            expert_random_vecs.append(random_vecs)
        
        return expert_random_vecs

    def _shuffle_stats(
        self, 
        layer_w: LayerWeights, 
        expert_random_vecs_dict: Dict[int, List[Tensor]], 
        R: Tensor
    ) -> Dict[int, Tuple[float, float]]:
        """Compute shuffle statistics using precomputed random vectors and pre-normalized router vectors.
        
        Args:
            layer_w: Layer weights (used for number of experts)
            expert_random_vecs_dict: Dictionary mapping k -> list of random vector matrices for each expert
            R: Pre-normalized router vectors [n_experts, d_model] already on CPU
            
        Returns:
            Dictionary mapping k values to (mean, std) tuples of shuffle statistics
        """
        n = len(layer_w.experts_w_in)
        stats: Dict[int, Tuple[float, float]] = {}

        for k in self.k_list:
            expert_random_vecs = expert_random_vecs_dict[k]
            vals = []
            for _ in range(self.n_shuffles):
                perm = torch.randperm(n, generator=self.rng)
                for i in range(n):
                    # Use precomputed random vectors and pre-normalized R[i]
                    vals.append(self._align_proj_energy(R[i], expert_random_vecs[int(perm[i])]))
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
    
    def _compute_alignment_scores(
        self,
        expert_random_vecs_dict: Dict[int, List[Tensor]],
        R: Tensor,
    ) -> List[Dict[str, float]]:
        """Compute alignment scores (projection energy) for all expert-k combinations.
        
        For k=1, also computes cos^2(theta) as a direct correlation measure.
        
        Args:
            expert_random_vecs_dict: Dictionary mapping k -> list of random vector matrices for each expert
            R: Pre-normalized router vectors [n_experts, d_model]
            
        Returns:
            List of dictionaries containing alignment scores (expert, k, align, cos_squared)
        """
        n_experts = len(R)
        scores = []
        
        for i in range(n_experts):
            for k in self.k_list:
                expert_random_vecs = expert_random_vecs_dict[k]
                # Compute alignment score (projection energy)
                align = self._align_proj_energy(R[i], expert_random_vecs[i])
                
                score_dict = {
                    "expert": i,
                    "k": int(k),
                    "align": align,
                }
                
                # For k=1, also compute cos^2(theta) as correlation-style measure
                if k == 1:
                    cos_squared = self._align_cos_squared(R[i], expert_random_vecs[i])
                    score_dict["cos_squared"] = cos_squared
                
                scores.append(score_dict)
        
        return scores
    
    def _compute_full_alignment_matrix(
        self,
        expert_random_vecs: List[Tensor],
        R: Tensor,
    ) -> Tensor:
        """Compute full router-expert alignment matrix for a given k.
        
        For each router vector r_i, computes alignment with ALL experts' random vector subspaces.
        This creates a [n_experts, n_experts] matrix where entry (i, j) is the alignment
        of router vector i with expert j's random k vectors.
        
        Args:
            expert_random_vecs: List of random vector matrices [d_model, k] for each expert
            R: Pre-normalized router vectors [n_experts, d_model]
            
        Returns:
            Alignment matrix [n_experts, n_experts] where entry (i, j) is align(R[i], Expert[j])
        """
        n_experts = len(expert_random_vecs)
        alignment_matrix = torch.zeros(n_experts, n_experts)
        
        for router_idx in range(n_experts):
            for expert_idx in range(n_experts):
                align = self._align_proj_energy(R[router_idx], expert_random_vecs[expert_idx])
                alignment_matrix[router_idx, expert_idx] = align
        
        return alignment_matrix
    
    def _compute_argmax_metrics(
        self,
        expert_random_vecs: List[Tensor],
        R: Tensor,
    ) -> Tuple[float, float]:
        """Compute argmax accuracy and alignment margin for a given k.
        
        For each router vector r_i (assigned to expert i), computes:
        1. Argmax accuracy: Does expert i achieve the maximum alignment? (1.0 if yes, 0.0 if no)
        2. Alignment margin: align(r_i, Expert_i) - max_{j≠i} align(r_i, Expert_j)
        
        Args:
            expert_random_vecs: List of random vector matrices [d_model, k] for each expert
            R: Pre-normalized router vectors [n_experts, d_model]
            
        Returns:
            Tuple of (mean_argmax_accuracy, mean_alignment_margin) across all router vectors
        """
        n_experts = len(expert_random_vecs)
        alignment_matrix = self._compute_full_alignment_matrix(expert_random_vecs, R)
        
        argmax_accuracies = []
        alignment_margins = []
        
        for router_idx in range(n_experts):
            # Get alignment scores for this router with all experts
            router_alignments = alignment_matrix[router_idx, :]  # [n_experts]
            
            # Correct expert alignment
            correct_align = router_alignments[router_idx].item()
            
            # Find maximum alignment (could be correct expert or another)
            max_align = router_alignments.max().item()
            max_expert_idx = router_alignments.argmax().item()
            
            # Argmax accuracy: 1.0 if correct expert has max, 0.0 otherwise
            argmax_acc = 1.0 if max_expert_idx == router_idx else 0.0
            argmax_accuracies.append(argmax_acc)
            
            # Alignment margin: correct - max(other experts)
            # If correct expert is max, margin = correct - second_max
            # Otherwise, margin = correct - max (negative)
            if max_expert_idx == router_idx:
                # Correct expert is best, compute margin vs second-best
                other_alignments = router_alignments.clone()
                other_alignments[router_idx] = -float('inf')  # Exclude correct expert
                second_max = other_alignments.max().item()
                margin = correct_align - second_max
            else:
                # Another expert is best, margin is negative
                margin = correct_align - max_align
            alignment_margins.append(margin)
        
        mean_argmax_acc = float(torch.tensor(argmax_accuracies).mean().item())
        mean_margin = float(torch.tensor(alignment_margins).mean().item())
        
        return mean_argmax_acc, mean_margin
    
    def _create_alignment_results(
        self,
        layer_w: LayerWeights,
        scores: List[Dict[str, float]],
        shuffle_stats: Dict[int, Tuple[float, float]],
        d_model: int,
        expert_random_vecs_dict: Dict[int, List[Tensor]],
        R: Tensor,
    ) -> List[AlignmentResult]:
        """Create AlignmentResult objects from alignment scores.
        
        Computes all derived statistics (random expectation, effect, shuffle comparison, etc.)
        from the raw alignment scores. Also computes argmax accuracy and alignment margins
        for unique identification analysis.
        
        Args:
            layer_w: Layer weights (for model_id and layer info)
            scores: List of score dictionaries from _compute_alignment_scores (expert, k, align)
            shuffle_stats: Shuffle statistics dict mapping k -> (mean, std)
            d_model: Model dimension
            expert_random_vecs_dict: Dictionary mapping k -> list of random vector matrices for each expert
            R: Pre-normalized router vectors [n_experts, d_model] (for argmax analysis)
            
        Returns:
            List of AlignmentResult objects with all computed statistics
        """
        results = []
        
        # Pre-compute argmax metrics for each k value (shared across all experts for same k)
        argmax_metrics: Dict[int, Tuple[float, float]] = {}
        for k in self.k_list:
            expert_random_vecs = expert_random_vecs_dict[k]
            argmax_acc, margin = self._compute_argmax_metrics(expert_random_vecs, R)
            argmax_metrics[k] = (argmax_acc, margin)
        
        for score in scores:
            align = score["align"]
            k = score["k"]
            expert = score["expert"]
            cos_squared = score.get("cos_squared", 0.0)  # Only set for k=1
            
            # Compute baseline expectations
            # For random vectors, the expected alignment is approximately k/d_model
            # (same as SVD, since we're using k normalized vectors)
            random_expect = float(k / d_model)
            effect = float(align - random_expect)
            
            # Get shuffle statistics
            mu, sd = shuffle_stats[k]
            delta = float(align - mu)
            z = float(delta / sd) if sd > _EPSILON else 0.0
            
            # Get argmax metrics (same for all experts at same k, but stored per expert for consistency)
            argmax_acc, margin = argmax_metrics[k]
            
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
                    argmax_accuracy=argmax_acc,
                    alignment_margin=margin,
                )
            )
        
        return results
    
    def analyze_layer(self, layer_w: LayerWeights) -> List[AlignmentResult]:
        """Analyze alignment for a single layer using random vectors.
        
        This method orchestrates the analysis pipeline:
        1. Normalize router vectors
        2. Sample random vectors for all experts and all k values
        3. Compute shuffle statistics
        4. Compute alignment results
        
        Args:
            layer_w: Layer weights containing router and expert weights
            
        Returns:
            List of alignment results for all expert-k combinations
        """
        # Step 1: Normalize router vectors once
        R = self._normalize_router_vectors(layer_w.gate_w)
        
        # Step 2: Sample random vectors for all experts and all k values
        print(f"Sampling random vectors for layer {layer_w.layer} ({len(layer_w.experts_w_in)} experts)...")
        print(f"  k values: {self.k_list}")
        print(f"  seed: {self.seed} (for reproducibility)")
        
        expert_random_vecs_dict: Dict[int, List[Tensor]] = {}
        for k in self.k_list:
            expert_random_vecs = self._compute_expert_random_vectors(layer_w, k)
            expert_random_vecs_dict[k] = expert_random_vecs
            print(f"  ✓ Sampled {k} random vectors per expert")
        
        # Step 3: Compute shuffle statistics for null distribution
        shuffle_stats = self._shuffle_stats(layer_w, expert_random_vecs_dict, R)
        
        # Step 4: Compute alignment scores (projection energy only)
        scores = self._compute_alignment_scores(expert_random_vecs_dict, R)
        
        # Step 5: Create result objects with all derived statistics
        d_model = int(layer_w.gate_w.shape[1])
        results = self._create_alignment_results(
            layer_w, scores, shuffle_stats, d_model, expert_random_vecs_dict, R
        )
        
        return results

