"""
Router-Centroid Analysis for MoE Models

This script empirically tests the hypothesis that Router/Gate weights act as 
"centroids" for the input data assigned to each expert.

It calculates the Cosine Similarity between the Router's weights and the actual 
mean (centroid) of the hidden states routed to each expert.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing utilities
from src.models.model_loader import Mixtral8x7B


# ============================================================================
# Model Loading Utilities
# ============================================================================

def pick_device() -> torch.device:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_id: str, device: torch.device) -> nn.Module:
    """
    Load the full model using transformers.
    
    NOTE: This downloads/loads the COMPLETE model (all weights, config, tokenizer).
    The transformers library automatically uses cached files if they exist (typically
    in ~/.cache/huggingface/), so if you've already downloaded the model elsewhere,
    it won't re-download - it will just load from cache.
    """
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, 'get_usable_length'):
        def get_usable_length(self, seq_length=None):
            return self.get_seq_length()
        DynamicCache.get_usable_length = get_usable_length
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    
    # Patch torch.Tensor.reshape to fix hidden_size mismatch bug
    import types
    
    original_reshape = torch.Tensor.reshape
    original_view = torch.Tensor.view
    
    def patched_reshape(self, *shape):
        try:
            return original_reshape(self, *shape)
        except RuntimeError as e:
            error_str = str(e)
            if "shape" in error_str and "invalid" in error_str:
                size_match = re.search(r"input of size (\d+)", error_str)
                shape_match = re.search(r"shape '\[(\d+), (\d+), (\d+)\]'", error_str)
                if size_match and shape_match:
                    actual_size = int(size_match.group(1))
                    bsz = int(shape_match.group(1))
                    q_len = int(shape_match.group(2))
                    expected_hidden = int(shape_match.group(3))
                    actual_hidden = actual_size // (bsz * q_len)
                    
                    if actual_hidden != expected_hidden and actual_hidden > 0:
                        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                            fixed_shape = (bsz, q_len, actual_hidden)
                        else:
                            fixed_shape = (bsz, q_len, actual_hidden)
                        print(f"  Fixing reshape: {shape} -> {fixed_shape}")
                        return original_reshape(self, fixed_shape)
            raise
    
    def patched_view(self, *shape):
        try:
            return original_view(self, *shape)
        except RuntimeError as e:
            error_str = str(e)
            if "shape" in error_str and "invalid" in error_str:
                size_match = re.search(r"input of size (\d+)", error_str)
                shape_match = re.search(r"shape '\[(\d+), (\d+), (\d+)\]'", error_str)
                if size_match and shape_match:
                    actual_size = int(size_match.group(1))
                    bsz = int(shape_match.group(1))
                    q_len = int(shape_match.group(2))
                    expected_hidden = int(shape_match.group(3))
                    actual_hidden = actual_size // (bsz * q_len)
                    
                    if actual_hidden != expected_hidden and actual_hidden > 0:
                        fixed_shape = (bsz, q_len, actual_hidden)
                        print(f"  Fixing view: {shape} -> {fixed_shape}")
                        return original_view(self, fixed_shape)
            raise
    
    torch.Tensor.reshape = patched_reshape
    torch.Tensor.view = patched_view
    
    model.to(device)
    return model


def load_tokenizer(model_id: str):
    """
    Load tokenizer for the model.
    
    NOTE: This automatically downloads the tokenizer files (tokenizer.json, config, etc.)
    if not already cached. Uses Hugging Face cache, so won't re-download if already present.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration for the analysis."""
    # Model settings
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    use_quantization = False  # Set to True for 4-bit/8-bit if needed
    quantization_bits = 8
    
    # Data collection
    max_samples_per_expert = 5000  # Buffer size to avoid OOM
    top_k = 2  # Top-k routing
    
    # Dataset
    dataset_name = "wikitext"  # or use custom data
    num_samples = 100  # Number of text samples to process
    
    # Layer selection
    target_layer_idx = 0  # Which MoE layer to analyze (0-31 for Mixtral)
    
    # Visualization
    pca_components = 2
    plot_dpi = 150
    output_dir = Path("results/router_centroid_analysis")


# ============================================================================
# Data Collection with Hooks
# ============================================================================

class RouterHook:
    """Hook to capture router inputs and expert selections."""
    
    def __init__(self, max_samples_per_expert: int = 5000):
        self.max_samples_per_expert = max_samples_per_expert
        self.reset()
    
    def reset(self):
        """Reset collected data."""
        # Store hidden states per expert: {expert_idx: [hidden_states]}
        self.hidden_states_by_expert = defaultdict(list)
        # Store router weights
        self.router_weights = None
        # Track how many samples collected per expert
        self.sample_counts = defaultdict(int)
    
    def forward_hook(self, module, input_tuple, output):
        """
        Hook function to capture router inputs and outputs.
        
        Args:
            module: The router/gate module
            input_tuple: Tuple containing (hidden_states, ...)
            output: Router logits [batch, seq_len, n_experts]
        """
        # Extract hidden states (input to router)
        if isinstance(input_tuple, tuple) and len(input_tuple) > 0:
            hidden_states = input_tuple[0]  # [batch, seq_len, hidden_dim]
        else:
            hidden_states = input_tuple
        
        # Get router weights
        if self.router_weights is None and hasattr(module, 'weight'):
            self.router_weights = module.weight.detach().cpu()  # [n_experts, hidden_dim]
        
        # Get router logits/output
        if isinstance(output, tuple):
            router_logits = output[0]  # [batch, seq_len, n_experts]
        else:
            router_logits = output
        
        # Compute probabilities and top-k
        probs = torch.softmax(router_logits, dim=-1)
        topk_vals, topk_indices = torch.topk(probs, k=min(Config.top_k, router_logits.shape[-1]), dim=-1)
        
        # Process each token
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_cpu = hidden_states.detach().cpu()
        
        for b in range(batch_size):
            for s in range(seq_len):
                # Get hidden state for this token
                hidden_state = hidden_states_cpu[b, s, :].numpy()  # [hidden_dim]
                
                # Get selected experts (top-k)
                selected_experts = topk_indices[b, s, :].cpu().tolist()
                
                # Store hidden state for each selected expert
                for expert_idx in selected_experts:
                    if self.sample_counts[expert_idx] < self.max_samples_per_expert:
                        self.hidden_states_by_expert[expert_idx].append(hidden_state)
                        self.sample_counts[expert_idx] += 1
    
    def get_collected_data(self) -> Dict[int, np.ndarray]:
        """
        Get collected hidden states as arrays.
        
        Returns:
            Dictionary mapping expert_idx -> array of shape [n_samples, hidden_dim]
        """
        result = {}
        for expert_idx, states_list in self.hidden_states_by_expert.items():
            if len(states_list) > 0:
                result[expert_idx] = np.array(states_list)
        return result


# ============================================================================
# Data Loading
# ============================================================================

def load_wikitext_samples(tokenizer, num_samples: int = 100, max_length: int = 128):
    """
    Load samples from Wikitext dataset.
    
    Args:
        tokenizer: Tokenizer for the model
        num_samples: Number of samples to load
        max_length: Maximum sequence length
        
    Returns:
        List of tokenized input_ids
    """
    try:
        from datasets import load_dataset
        
        print(f"Loading {num_samples} samples from Wikitext...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        
        texts = []
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            text = example.get("text", "")
            if len(text.strip()) > 50:  # Filter very short texts
                texts.append(text)
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        
        print(f"Loaded {len(texts)} text samples")
        return tokenized["input_ids"]
    
    except ImportError:
        print("⚠️  datasets library not available. Using dummy data instead.")
        # Generate dummy data
        vocab_size = len(tokenizer)
        dummy_inputs = torch.randint(0, vocab_size, (num_samples, max_length))
        return dummy_inputs


# ============================================================================
# Analysis Functions
# ============================================================================

def compute_empirical_centroids(hidden_states_by_expert: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    Compute empirical centroids (mean) for each expert.
    
    Args:
        hidden_states_by_expert: Dictionary mapping expert_idx -> [n_samples, hidden_dim]
        
    Returns:
        Dictionary mapping expert_idx -> centroid [hidden_dim]
    """
    centroids = {}
    for expert_idx, states in hidden_states_by_expert.items():
        if len(states) > 0:
            centroids[expert_idx] = np.mean(states, axis=0)
        else:
            centroids[expert_idx] = None
    
    return centroids


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score
    """
    # Normalize vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Cosine similarity
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    
    # Clamp to [-1, 1] for numerical stability
    return np.clip(similarity, -1.0, 1.0)


def analyze_router_centroids(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    target_layer_idx: int = 0,
) -> Dict:
    """
    Main analysis function.
    
    Args:
        model: Loaded MoE model
        tokenizer: Tokenizer
        device: Device to run on
        target_layer_idx: Which MoE layer to analyze
        
    Returns:
        Dictionary with analysis results
    """
    model.eval()
    
    # Find the router/gate module for the target layer
    router_module = None
    router_name = None
    
    for name, module in model.named_modules():
        if f"layers.{target_layer_idx}" in name:
            if "gate" in name.lower() or "router" in name.lower():
                if "weight" not in name.lower():  # Get the module, not the parameter
                    router_module = module
                    router_name = name
                    break
    
    if router_module is None:
        # Try alternative naming
        try:
            router_module = model.model.layers[target_layer_idx].block_sparse_moe.gate
            router_name = f"model.layers[{target_layer_idx}].block_sparse_moe.gate"
        except (AttributeError, IndexError) as e:
            raise RuntimeError(f"Could not find router module for layer {target_layer_idx}: {e}")
    
    print(f"Found router module: {router_name}")
    
    # Initialize hook
    hook_manager = RouterHook(max_samples_per_expert=Config.max_samples_per_expert)
    
    # Register hook
    hook_handle = router_module.register_forward_hook(hook_manager.forward_hook)
    
    try:
        # Load data
        print("\n" + "="*80)
        print("Collecting hidden states...")
        print("="*80)
        
        input_ids = load_wikitext_samples(tokenizer, num_samples=Config.num_samples)
        input_ids = input_ids.to(device)
        
        # Process in batches to avoid OOM
        batch_size = 4
        num_batches = (len(input_ids) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(input_ids))
                batch_inputs = input_ids[start_idx:end_idx]
                
                print(f"Processing batch {batch_idx + 1}/{num_batches}...", end="\r")
                
                # Forward pass
                try:
                    outputs = model(batch_inputs, output_router_logits=True)
                except Exception as e:
                    print(f"\n⚠️  Error in forward pass: {e}")
                    continue
        
        print(f"\n✓ Collected data for {len(hook_manager.hidden_states_by_expert)} experts")
        for expert_idx, count in sorted(hook_manager.sample_counts.items()):
            print(f"  Expert {expert_idx}: {count} samples")
    
    finally:
        # Remove hook
        hook_handle.remove()
    
    # Get collected data
    hidden_states_by_expert = hook_manager.get_collected_data()
    router_weights = hook_manager.router_weights.numpy() if hook_manager.router_weights is not None else None
    
    if router_weights is None:
        # Try to get router weights directly
        try:
            router_weights = router_module.weight.detach().cpu().numpy()
        except AttributeError:
            raise RuntimeError("Could not extract router weights")
    
    print(f"\nRouter weights shape: {router_weights.shape}")  # [n_experts, hidden_dim]
    
    # Compute empirical centroids
    print("\n" + "="*80)
    print("Computing empirical centroids...")
    print("="*80)
    
    centroids = compute_empirical_centroids(hidden_states_by_expert)
    
    # Compute cosine similarities
    print("\n" + "="*80)
    print("Computing Cosine Similarities...")
    print("="*80)
    
    similarities = {}
    for expert_idx in range(router_weights.shape[0]):
        router_weight = router_weights[expert_idx, :]  # [hidden_dim]
        
        if expert_idx in centroids and centroids[expert_idx] is not None:
            centroid = centroids[expert_idx]
            similarity = compute_cosine_similarity(router_weight, centroid)
            similarities[expert_idx] = similarity
            print(f"Expert {expert_idx:2d}: Cosine Similarity = {similarity:.4f} "
                  f"(n_samples={len(hidden_states_by_expert.get(expert_idx, []))})")
        else:
            similarities[expert_idx] = None
            print(f"Expert {expert_idx:2d}: No data collected")
    
    return {
        "hidden_states_by_expert": hidden_states_by_expert,
        "router_weights": router_weights,
        "centroids": centroids,
        "similarities": similarities,
        "sample_counts": dict(hook_manager.sample_counts),
        "layer_idx": target_layer_idx,
    }


# ============================================================================
# Visualization
# ============================================================================

def visualize_router_centroids(
    results: Dict,
    output_path: Path,
    pca_components: int = 2,
):
    """
    Visualize router weights and empirical centroids using PCA.
    
    Args:
        results: Analysis results dictionary
        output_path: Path to save the plot
        pca_components: Number of PCA components (2 for 2D plot)
    """
    hidden_states_by_expert = results["hidden_states_by_expert"]
    router_weights = results["router_weights"]
    centroids = results["centroids"]
    similarities = results["similarities"]
    
    # Collect all hidden states for PCA fitting
    all_states = []
    expert_labels = []
    
    for expert_idx, states in hidden_states_by_expert.items():
        all_states.append(states)
        expert_labels.extend([expert_idx] * len(states))
    
    if len(all_states) == 0:
        print("⚠️  No hidden states collected. Cannot create visualization.")
        return
    
    all_states_array = np.vstack(all_states)
    expert_labels = np.array(expert_labels)
    
    print(f"\nFitting PCA on {all_states_array.shape[0]} samples...")
    
    # Fit PCA
    pca = PCA(n_components=pca_components, random_state=42)
    all_states_2d = pca.fit_transform(all_states_array)
    
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Transform router weights and centroids
    router_weights_2d = pca.transform(router_weights)
    
    centroids_2d = {}
    for expert_idx, centroid in centroids.items():
        if centroid is not None:
            centroids_2d[expert_idx] = pca.transform(centroid.reshape(1, -1))[0]
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Hidden states colored by expert
    ax = axes[0]
    
    # Get unique experts and assign colors
    unique_experts = sorted(set(expert_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_experts)))
    color_map = {expert: colors[i % len(colors)] for i, expert in enumerate(unique_experts)}
    
    # Plot hidden states
    for expert_idx in unique_experts:
        mask = expert_labels == expert_idx
        if mask.sum() > 0:
            ax.scatter(
                all_states_2d[mask, 0],
                all_states_2d[mask, 1],
                c=[color_map[expert_idx]],
                label=f"Expert {expert_idx}",
                alpha=0.3,
                s=10,
            )
    
    # Overlay router weights
    for expert_idx in range(router_weights_2d.shape[0]):
        ax.scatter(
            router_weights_2d[expert_idx, 0],
            router_weights_2d[expert_idx, 1],
            c=[color_map.get(expert_idx, 'black')],
            marker='*',
            s=500,
            edgecolors='black',
            linewidths=2,
            label=f"Router {expert_idx}" if expert_idx < 3 else "",  # Only label first few
            zorder=10,
        )
    
    # Overlay empirical centroids
    for expert_idx, centroid_2d in centroids_2d.items():
        if expert_idx in color_map:
            ax.scatter(
                centroid_2d[0],
                centroid_2d[1],
                c=[color_map[expert_idx]],
                marker='X',
                s=500,
                edgecolors='red',
                linewidths=2,
                label=f"Centroid {expert_idx}" if expert_idx < 3 else "",  # Only label first few
                zorder=10,
            )
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax.set_title("Hidden States, Router Weights, and Empirical Centroids (PCA)", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cosine similarity bar chart
    ax2 = axes[1]
    
    expert_indices = sorted([k for k, v in similarities.items() if v is not None])
    similarity_values = [similarities[k] for k in expert_indices]
    
    bars = ax2.bar(
        expert_indices,
        similarity_values,
        color=[color_map.get(idx, 'gray') for idx in expert_indices],
        alpha=0.7,
        edgecolor='black',
        linewidth=1,
    )
    
    # Add value labels on bars
    for bar, val in zip(bars, similarity_values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01 if height >= 0 else height - 0.05,
            f'{val:.3f}',
            ha='center',
            va='bottom' if height >= 0 else 'top',
            fontsize=9,
        )
    
    ax2.set_xlabel("Expert Index", fontsize=12)
    ax2.set_ylabel("Cosine Similarity", fontsize=12)
    ax2.set_title("Router Weight vs Empirical Centroid Similarity", fontsize=14, fontweight='bold')
    ax2.set_ylim([-1.1, 1.1])
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=Config.plot_dpi, bbox_inches='tight')
    print(f"\n✓ Saved plot to {output_path}")
    
    plt.close()


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("ROUTER-CENTROID ANALYSIS")
    print("="*80)
    print(f"Model: {Config.model_id}")
    print(f"Target Layer: {Config.target_layer_idx}")
    print(f"Max samples per expert: {Config.max_samples_per_expert}")
    print(f"Top-K routing: {Config.top_k}")
    print()
    
    # Setup device
    device = pick_device()
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model {Config.model_id}...")
    model = load_model(Config.model_id, device)
    tokenizer = load_tokenizer(Config.model_id)
    
    # Run analysis
    results = analyze_router_centroids(
        model=model,
        tokenizer=tokenizer,
        device=device,
        target_layer_idx=Config.target_layer_idx,
    )
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    valid_similarities = {k: v for k, v in results["similarities"].items() if v is not None}
    if valid_similarities:
        mean_similarity = np.mean(list(valid_similarities.values()))
        std_similarity = np.std(list(valid_similarities.values()))
        min_similarity = min(valid_similarities.values())
        max_similarity = max(valid_similarities.values())
        
        print(f"\nCosine Similarity Statistics:")
        print(f"  Mean: {mean_similarity:.4f}")
        print(f"  Std:  {std_similarity:.4f}")
        print(f"  Min:  {min_similarity:.4f}")
        print(f"  Max:  {max_similarity:.4f}")
        print(f"  Experts with data: {len(valid_similarities)}/{len(results['similarities'])}")
    
    # Save results
    output_dir = Config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numerical results (without large arrays)
    results_summary = {
        "layer_idx": results["layer_idx"],
        "similarities": {str(k): float(v) if v is not None else None 
                         for k, v in results["similarities"].items()},
        "sample_counts": results["sample_counts"],
        "router_weights_shape": list(results["router_weights"].shape),
    }
    
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n✓ Saved results to {results_file}")
    
    # Create visualization
    plot_path = output_dir / "router_centroid_analysis.png"
    visualize_router_centroids(
        results=results,
        output_path=plot_path,
        pca_components=Config.pca_components,
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"  - results.json: Numerical results")
    print(f"  - router_centroid_analysis.png: Visualization")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Router-Centroid Analysis for MoE Models")
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="MoE layer index to analyze (default: 0)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of text samples to process (default: 100)",
    )
    parser.add_argument(
        "--max-samples-per-expert",
        type=int,
        default=5000,
        help="Maximum samples to collect per expert (default: 5000)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-K routing (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/router_centroid_analysis)",
    )
    
    args = parser.parse_args()
    
    # Update config
    Config.target_layer_idx = args.layer
    Config.num_samples = args.num_samples
    Config.max_samples_per_expert = args.max_samples_per_expert
    Config.top_k = args.top_k
    if args.output_dir:
        Config.output_dir = Path(args.output_dir)
    
    main()
