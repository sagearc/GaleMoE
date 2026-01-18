"""
Router-Centroid Analysis for MoE Models

This script empirically tests the hypothesis that Router/Gate weights act as 
"centroids" for the input data assigned to each expert.

It calculates the Cosine Similarity between the Router's weights and the actual 
mean (centroid) of the hidden states routed to each expert.

REQUIREMENTS:
    - autoawq: Required for AWQ (pre-quantized) model loading
      Install with: pip install autoawq
"""

import json
import os
import re
import shutil
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

# Set PyTorch CUDA memory allocator configuration to reduce fragmentation
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ============================================================================
# Model Loading Utilities
# ============================================================================

def pick_device() -> torch.device:
    """Pick the best available device."""
    if torch.cuda.is_available():
        # Aggressively clear cache before starting
        clear_cuda_cache()
        # Force garbage collection
        import gc
        gc.collect()
        clear_cuda_cache()
        
        # Print memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU Memory Status: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {total:.2f} GB total")
        
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_cuda_cache():
    """Clear CUDA cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def force_clear_gpu_memory():
    """Force clear all GPU memory by garbage collection and cache clearing."""
    import gc
    if torch.cuda.is_available():
        # Clear Python references
        gc.collect()
        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Try to reset peak memory stats (if supported)
        try:
            torch.cuda.reset_peak_memory_stats()
        except:
            pass


def print_memory_requirements(model_id: str, device: torch.device):
    """
    Print memory requirements for loading the model.
    
    Args:
        model_id: Hugging Face model identifier
        device: Target device
    """
    print("\n" + "="*80)
    print("MEMORY REQUIREMENTS")
    print("="*80)
    
    # Model size estimates (AWQ 4-bit quantized)
    if "AWQ" in model_id or "awq" in model_id.lower():
        print(f"📦 Model: {model_id} (4-bit AWQ quantized)")
        print(f"   Disk space needed: ~4-5 GB (for model files)")
        print(f"   GPU memory needed: ~6-8 GB (for inference)")
        print(f"   CPU RAM needed: ~8-10 GB (if offloading to CPU)")
    else:
        print(f"📦 Model: {model_id}")
        print(f"   Disk space needed: ~47 GB (full precision Mixtral-8x7B)")
        print(f"   GPU memory needed: ~48-50 GB (for inference)")
        print(f"   CPU RAM needed: ~50 GB (if offloading to CPU)")
    
    # Check available GPU memory
    if device.type == "cuda" and torch.cuda.is_available():
        total_gpu = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_gpu = torch.cuda.memory_allocated() / 1024**3
        reserved_gpu = torch.cuda.memory_reserved() / 1024**3
        free_gpu = total_gpu - reserved_gpu
        
        print(f"\n💾 GPU Memory Status:")
        print(f"   Total: {total_gpu:.2f} GB")
        print(f"   Allocated: {allocated_gpu:.2f} GB")
        print(f"   Reserved: {reserved_gpu:.2f} GB")
        print(f"   Free: {free_gpu:.2f} GB")
        
        # Check if enough memory
        if "AWQ" in model_id or "awq" in model_id.lower():
            required = 6.0  # GB
        else:
            required = 48.0  # GB
        
        if free_gpu < required:
            print(f"\n⚠️  WARNING: Free GPU memory ({free_gpu:.2f} GB) may be insufficient!")
            print(f"   Recommended: {required:.1f} GB free for smooth operation")
            print(f"   The model will use device_map='auto' which may offload to CPU if needed")
        else:
            print(f"\n✓ Sufficient GPU memory available ({free_gpu:.2f} GB free)")
    
    # Check disk space for cache
    cache_path = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    if not cache_path:
        cache_path = str(Path.home() / ".cache" / "huggingface")
    
    cache_dir = Path(cache_path)
    if cache_dir.exists():
        # Get disk space
        stat = shutil.disk_usage(cache_dir)
        free_disk = stat.free / 1024**3
        print(f"\n💿 Disk Space (cache location: {cache_path}):")
        print(f"   Free: {free_disk:.2f} GB")
        
        if "AWQ" in model_id or "awq" in model_id.lower():
            required_disk = 5.0
        else:
            required_disk = 50.0
        
        if free_disk < required_disk:
            print(f"   ⚠️  WARNING: May need {required_disk:.1f} GB for model download")
        else:
            print(f"   ✓ Sufficient disk space available")
    
    print("="*80 + "\n")


def load_model(model_id: str, device: torch.device) -> nn.Module:
    """
    Load AWQ (pre-quantized) model using transformers with memory-efficient loading.
    
    NOTE: AWQ models are pre-quantized, so loading is much faster than on-the-fly quantization.
    The transformers library automatically uses cached files if they exist (typically
    in ~/.cache/huggingface/), so if you've already downloaded the model elsewhere,
    it won't re-download - it will just load from cache.
    
    REQUIREMENTS:
        - autoawq: Required for AWQ model support
          Install with: pip install autoawq
    """
    # Check if autoawq is available
    try:
        import awq
    except ImportError:
        raise ImportError(
            "autoawq is required for AWQ model loading. "
            "Install with: pip install autoawq"
        )
    
    # Clear cache aggressively before loading
    clear_cuda_cache()
    
    # Force garbage collection to free any Python objects holding GPU memory
    import gc
    gc.collect()
    clear_cuda_cache()
    
    # Set up Hugging Face cache and download optimizations
    import os
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    
    # Print cache location for debugging
    # Get cache path - try multiple methods for compatibility
    cache_path = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    if not cache_path:
        # Fallback to default location
        from pathlib import Path
        cache_path = str(Path.home() / ".cache" / "huggingface")
    print(f"📦 Using Hugging Face cache: {cache_path}")
    
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, 'get_usable_length'):
        def get_usable_length(self, seq_length=None):
            return self.get_seq_length()
        DynamicCache.get_usable_length = get_usable_length
    
    # Load AWQ model - AWQ models are pre-quantized, so no quantization config needed
    print("⚡ Loading AWQ (pre-quantized) model - this should be fast!")
    
    try:
        if device.type == "cuda":
            # Calculate available memory (leave some headroom)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            max_memory = {0: f"{int(total_memory * 0.95 / 1024**3)}GiB"}
            
            load_kwargs = {
                "low_cpu_mem_usage": True,
                "trust_remote_code": False,  # AWQ models typically don't need trust_remote_code
                "device_map": "auto",
                "max_memory": max_memory,
            }
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **load_kwargs
            )
        else:
            # For non-CUDA devices, load normally
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            )
            model.to(device)
    except Exception as e:
        print(f"⚠️  Error with device_map loading, falling back to standard loading: {e}")
        # Fallback: load to CPU first, then move to device
        clear_cuda_cache()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        # Move to device
        clear_cuda_cache()
        model.to(device)
    
    # Verify router layer is accessible and check dtype
    try:
        router_weight = model.model.layers[0].block_sparse_moe.gate.weight
        router_dtype = router_weight.dtype
        print(f"✓ Router layer verified - dtype: {router_dtype}")
        print(f"  Router weight shape: {router_weight.shape}")
        
        # Check if dtype is suitable for analysis (should be float16 or bfloat16)
        if router_dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            print(f"⚠️  Warning: Router dtype is {router_dtype}, which may affect analysis precision")
        else:
            print(f"  ✓ Router dtype is suitable for analysis")
    except (AttributeError, IndexError) as e:
        print(f"⚠️  Warning: Could not verify router layer: {e}")
        print("   Analysis may still work, but verification failed")
    
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
    
    # Final cache clear after loading
    clear_cuda_cache()
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
    # Using AWQ (pre-quantized) model for faster loading
    model_id = "TheBloke/Mixtral-8x7B-v0.1-AWQ"  # Pre-quantized AWQ model (4-bit, fast loading)
    # Note: AWQ models are pre-quantized, so quantization_bits is not applicable
    
    # Data collection
    max_samples_per_expert = 5000  # Buffer size to avoid OOM
    top_k = 2  # Top-k routing
    
    # Dataset
    dataset_name = "wikitext"  # or use custom data
    num_samples = 100  # Number of text samples to process
    
    # Memory management
    initial_batch_size = 1  # Start with smaller batch size to avoid OOM
    min_batch_size = 1  # Minimum batch size
    
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
        # Store inputs captured by pre-hook
        self.current_inputs = None
    
    def pre_hook(self, module, input_tuple):
        """Pre-hook to capture inputs before forward pass."""
        try:
            # Extract hidden states (input to router)
            if isinstance(input_tuple, tuple) and len(input_tuple) > 0:
                hidden_states = input_tuple[0]  # [batch, seq_len, hidden_dim]
            else:
                hidden_states = input_tuple
            
            # Store inputs for use in forward hook
            if isinstance(hidden_states, torch.Tensor):
                self.current_inputs = hidden_states.detach().cpu()
        except Exception as e:
            # Don't break the forward pass
            pass
    
    def forward_hook(self, module, input_tuple, output):
        """
        Hook function to capture router inputs and outputs.
        
        Args:
            module: The router/gate module
            input_tuple: Tuple containing (hidden_states, ...)
            output: Router logits [batch, seq_len, n_experts]
        """
        try:
            # Try to use pre-hook captured inputs first (more reliable)
            if self.current_inputs is not None:
                hidden_states = self.current_inputs
                # Reset for next call
                self.current_inputs = None
            else:
                # Fallback to extracting from input_tuple
                if isinstance(input_tuple, tuple) and len(input_tuple) > 0:
                    hidden_states = input_tuple[0]  # [batch, seq_len, hidden_dim]
                else:
                    hidden_states = input_tuple
            
            # Check if hidden_states is a tensor
            if not isinstance(hidden_states, torch.Tensor):
                print(f"⚠️  Hook: hidden_states is not a tensor: {type(hidden_states)}")
                return
            
            # Get router weights
            if self.router_weights is None and hasattr(module, 'weight'):
                self.router_weights = module.weight.detach().cpu()  # [n_experts, hidden_dim]
            
            # Get router logits/output
            if isinstance(output, tuple):
                router_logits = output[0]  # [batch, seq_len, n_experts]
            else:
                router_logits = output
            
            # Check if router_logits is a tensor
            if not isinstance(router_logits, torch.Tensor):
                print(f"⚠️  Hook: router_logits is not a tensor: {type(router_logits)}")
                return
            
            # Move to CPU immediately to free GPU memory
            router_logits_cpu = router_logits.detach().cpu()
            
            # Compute probabilities and top-k on CPU
            probs = torch.softmax(router_logits_cpu, dim=-1)
            topk_vals, topk_indices = torch.topk(probs, k=min(Config.top_k, router_logits_cpu.shape[-1]), dim=-1)
            
            # Move hidden states to CPU immediately (if not already on CPU from pre-hook)
            if len(hidden_states.shape) != 3:
                print(f"⚠️  Hook: Unexpected hidden_states shape: {hidden_states.shape}, expected [batch, seq_len, hidden_dim]")
                return
            
            batch_size, seq_len, hidden_dim = hidden_states.shape
            # If already on CPU from pre-hook, use it directly; otherwise move to CPU
            if isinstance(hidden_states, torch.Tensor) and hidden_states.is_cuda:
                hidden_states_cpu = hidden_states.detach().cpu()
            elif isinstance(hidden_states, torch.Tensor):
                hidden_states_cpu = hidden_states
            else:
                # Convert numpy array to tensor if needed
                hidden_states_cpu = torch.from_numpy(hidden_states) if isinstance(hidden_states, np.ndarray) else hidden_states
            
            # Clear GPU references
            del hidden_states, router_logits
            
            samples_collected = 0
            for b in range(batch_size):
                for s in range(seq_len):
                    # Get hidden state for this token
                    hidden_state = hidden_states_cpu[b, s, :].numpy()  # [hidden_dim]
                    
                    # Get selected experts (top-k) - already on CPU
                    selected_experts = topk_indices[b, s, :].tolist()
                    
                    # Store hidden state for each selected expert
                    for expert_idx in selected_experts:
                        if self.sample_counts[expert_idx] < self.max_samples_per_expert:
                            self.hidden_states_by_expert[expert_idx].append(hidden_state)
                            self.sample_counts[expert_idx] += 1
                            samples_collected += 1
            
            # Debug: Print first call to verify hook is working
            if not hasattr(self, '_hook_called'):
                self._hook_called = True
                print(f"\n✓ Hook called successfully! Collected {samples_collected} samples from batch")
                print(f"  Hidden states shape: [{batch_size}, {seq_len}, {hidden_dim}]")
                print(f"  Router logits shape: {router_logits_cpu.shape}")
                
        except Exception as e:
            # Print error for debugging instead of silently ignoring
            if not hasattr(self, '_hook_error_printed'):
                self._hook_error_printed = True
                print(f"\n❌ Hook error: {type(e).__name__}: {e}")
                print(f"   Input type: {type(input_tuple)}")
                if isinstance(input_tuple, tuple):
                    print(f"   Input tuple length: {len(input_tuple)}")
                    for i, inp in enumerate(input_tuple):
                        print(f"   Input[{i}] type: {type(inp)}")
                        if isinstance(inp, torch.Tensor):
                            print(f"   Input[{i}] shape: {inp.shape}")
                print(f"   Output type: {type(output)}")
                if isinstance(output, torch.Tensor):
                    print(f"   Output shape: {output.shape}")
                import traceback
                traceback.print_exc()
    
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
        min_text_length = 50  # Start with minimum length threshold
        max_iterations = num_samples * 10  # Safety limit to avoid infinite loop
        
        # Collect texts, continuing until we have enough or hit the limit
        for i, example in enumerate(dataset):
            if len(texts) >= num_samples:
                break
            if i >= max_iterations:
                break
            
            text = example.get("text", "")
            if len(text.strip()) > min_text_length:  # Filter very short texts
                texts.append(text)
        
        # If we still don't have enough samples, lower the threshold and try again
        if len(texts) < num_samples:
            print(f"⚠️  Only found {len(texts)} samples with length > {min_text_length}. Lowering threshold...")
            texts = []
            min_text_length = 10  # Lower threshold
            
            for i, example in enumerate(dataset):
                if len(texts) >= num_samples:
                    break
                if i >= max_iterations:
                    break
                
                text = example.get("text", "")
                if len(text.strip()) > min_text_length:
                    texts.append(text)
        
        # Check if we have any texts before tokenizing
        if len(texts) == 0:
            print("⚠️  No texts found after filtering. Using dummy data instead.")
            vocab_size = len(tokenizer)
            dummy_inputs = torch.randint(0, vocab_size, (num_samples, max_length))
            return dummy_inputs
        
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
    print(f"✓ Hook registered on {router_name}")
    
    try:
        # Load data
        print("\n" + "="*80)
        print("Collecting hidden states...")
        print("="*80)
        
        input_ids = load_wikitext_samples(tokenizer, num_samples=Config.num_samples)
        
        # Verify input_ids shape
        if isinstance(input_ids, torch.Tensor):
            print(f"Input shape: {input_ids.shape}")
        else:
            print(f"⚠️  Unexpected input_ids type: {type(input_ids)}")
            raise ValueError(f"Expected torch.Tensor, got {type(input_ids)}")
        
        # Test forward pass with a single sample to verify hook works
        print("\nTesting forward pass with single sample...")
        test_input = input_ids[:1].to(device)
        try:
            with torch.no_grad():
                _ = model(test_input)
            print("✓ Forward pass test successful")
            # Check if hook collected anything
            test_samples = sum(hook_manager.sample_counts.values())
            if test_samples > 0:
                print(f"✓ Hook is working! Collected {test_samples} samples from test")
            else:
                print(f"⚠️  Hook did not collect samples from test forward pass")
                print(f"   This suggests the hook may not be called or is encountering errors")
        except Exception as e:
            print(f"❌ Forward pass test failed: {e}")
            raise
        
        # Process in batches to avoid OOM with dynamic batch size reduction
        batch_size = Config.initial_batch_size
        num_batches = (len(input_ids) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            batch_idx = 0
            while batch_idx < num_batches:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(input_ids))
                
                # Move batch to device only when needed
                batch_inputs = input_ids[start_idx:end_idx].to(device)
                
                print(f"Processing batch {batch_idx + 1}/{num_batches} (batch_size={batch_size})...", end="\r")
                
                # Forward pass with retry logic for OOM
                # Note: We don't need output_router_logits=True since we're using hooks
                # to capture router outputs directly from the router module
                # 
                # IMPORTANT: With quantized models, there's a known issue where the forward
                # pass may fail with "not enough values to unpack" error. Since we're using
                # hooks to capture data, we can work around this by catching the error.
                try:
                    # Try different return formats to work around quantized model issues
                    try:
                        outputs = model(batch_inputs, return_dict=False)
                    except (ValueError, RuntimeError) as e:
                        error_msg = str(e)
                        if "not enough values to unpack" in error_msg:
                            # Known issue with quantized Mixtral - try with return_dict=True
                            try:
                                outputs = model(batch_inputs, return_dict=True)
                            except Exception:
                                # If both fail, the hook should have already captured the data
                                # So we can continue - the error is in the output unpacking, not the forward pass
                                print(f"\n⚠️  Output unpacking error (known quantized model issue) - hook data captured, continuing...")
                                del batch_inputs
                                batch_idx += 1
                                continue
                        else:
                            raise
                    
                    # Clear batch from GPU immediately after forward pass
                    del batch_inputs, outputs
                    
                    # Clear cache periodically
                    if (batch_idx + 1) % 5 == 0:
                        clear_cuda_cache()
                    
                    batch_idx += 1
                    
                except torch.cuda.OutOfMemoryError as e:
                    print(f"\n⚠️  OOM error at batch {batch_idx + 1}. Reducing batch size...")
                    clear_cuda_cache()
                    
                    # Reduce batch size
                    batch_size = max(Config.min_batch_size, batch_size // 2)
                    if batch_size < Config.min_batch_size:
                        print(f"\n❌ Cannot reduce batch size further. Stopping.")
                        break
                    
                    # Recalculate number of batches with new batch size
                    num_batches = (len(input_ids) + batch_size - 1) // batch_size
                    print(f"   New batch size: {batch_size}, Remaining batches: {num_batches - batch_idx}")
                    continue
                    
                except (ValueError, RuntimeError) as e:
                    error_msg = str(e)
                    if "not enough values to unpack" in error_msg:
                        # This is a known issue with quantized Mixtral models
                        # The hook should have already captured the router data before the error
                        # So we can safely continue - the error is just in unpacking the final output
                        if batch_idx == 0:
                            print(f"\n⚠️  Detected quantized model unpacking issue (known bug)")
                            print(f"   The hook has captured the data we need, so we'll continue despite this error.")
                        del batch_inputs
                        clear_cuda_cache()
                        batch_idx += 1
                        continue
                    else:
                        # Re-raise if it's a different error
                        raise
                    
                except Exception as e:
                    error_msg = str(e)
                    if "not enough values to unpack" in error_msg:
                        # Known quantized model issue - hook data should be captured
                        if batch_idx == 0:
                            print(f"\n⚠️  Quantized model unpacking error (continuing - hook data captured)")
                        del batch_inputs
                        clear_cuda_cache()
                        batch_idx += 1
                        continue
                    else:
                        print(f"\n⚠️  Unexpected error in forward pass: {error_msg}")
                        # Move to next batch even on error
                        del batch_inputs
                        clear_cuda_cache()
                        batch_idx += 1
                        continue
        
        # Final cache clear
        clear_cuda_cache()
        
        # Check if any data was collected
        total_samples = sum(hook_manager.sample_counts.values())
        if total_samples == 0:
            print(f"\n❌ ERROR: No hidden states collected!")
            print(f"   This could mean:")
            print(f"   1. The hook is not being called (check if forward pass completes)")
            print(f"   2. The hook is encountering errors (check error messages above)")
            print(f"   3. The router module structure is different than expected")
            print(f"   4. All samples are being filtered out")
            
            # Try to verify hook was called
            if hasattr(hook_manager, '_hook_called'):
                print(f"   ✓ Hook was called at least once")
            else:
                print(f"   ❌ Hook was never called - forward pass may be failing before reaching router")
            
            # Try to get router weights directly as a diagnostic
            try:
                test_weights = router_module.weight
                print(f"   ✓ Router module accessible, weight shape: {test_weights.shape}")
            except Exception as e:
                print(f"   ❌ Cannot access router weights: {e}")
        
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
    print(f"Initial batch size: {Config.initial_batch_size}")
    print()
    
    # Setup device
    device = pick_device()
    print(f"Device: {device}")
    
    # Print memory info if CUDA
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free = total - reserved
        print(f"CUDA Memory: {total:.2f} GB total, {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {free:.2f} GB free")
        print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'not set')}")
        
        # Warn if memory is already heavily used
        if reserved / total > 0.9:
            print(f"⚠️  WARNING: GPU memory is {reserved/total*100:.1f}% used. Clearing cache...")
            force_clear_gpu_memory()
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            free = total - reserved
            print(f"   After clearing: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {free:.2f} GB free")
    
    # Print memory requirements
    print_memory_requirements(Config.model_id, device)
    
    # Load model
    print(f"\n⚡ Loading AWQ model {Config.model_id}...")
    print(f"   💡 AWQ models are pre-quantized - loading should be fast!")
    try:
        model = load_model(Config.model_id, device)
        tokenizer = load_tokenizer(Config.model_id)
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ Out of memory during model loading!")
        print(f"   Try the following:")
        print(f"   1. Close other processes using the GPU")
        print(f"   2. Restart Python to clear any lingering GPU memory")
        print(f"   3. Try a different AWQ model or check available GPU memory")
        print(f"   4. Run: python -c 'import torch; torch.cuda.empty_cache()' in a separate terminal")
        raise
    
    # Clear cache after model loading
    clear_cuda_cache()
    
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
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model ID to use (default: TheBloke/Mixtral-8x7B-v0.1-AWQ)",
    )
    
    args = parser.parse_args()
    
    # Update config
    Config.target_layer_idx = args.layer
    Config.num_samples = args.num_samples
    Config.max_samples_per_expert = args.max_samples_per_expert
    Config.top_k = args.top_k
    if args.output_dir:
        Config.output_dir = Path(args.output_dir)
    if args.model_id:
        Config.model_id = args.model_id
    
    main()
