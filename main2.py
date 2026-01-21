import os
import json
import pickle
import argparse
import copy
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure clear logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    svd_dir: str
    layer_idx: int
    num_experts: int = 8
    model_id: str = "mistralai/Mixtral-8x7B-v0.1"
    model_tag: str = "mistralai_Mixtral_8x7B_v0.1"
    output_file: str = "results_ablation.json"
    num_samples: int = 200
    seq_len: int = 512
    batch_size: int = 4
    seed: int = 42

# -----------------------------------------------------------------------------
# Math & Vector Utils
# -----------------------------------------------------------------------------
def load_single_expert_vector(filepath: str) -> torch.Tensor:
    """Loads the top singular vector from a pickle file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"SVD file not found: {filepath}")

    with open(filepath, "rb") as f:
        obj = pickle.load(f)

    # Handle different storage formats
    tensor = None
    if isinstance(obj, dict):
        for key in ["Vh", "vh", "V", "v"]:
            if key in obj:
                tensor = obj[key]
                break
    elif isinstance(obj, (tuple, list)) and len(obj) >= 3:
        tensor = obj[2]  # Assume (U, S, Vh)

    if tensor is None:
        raise ValueError(f"Could not extract vector from {filepath}")

    # Convert to torch and normalize
    t = torch.as_tensor(tensor, dtype=torch.float32, device="cpu")
    
    # If matrix, take top component
    if t.ndim == 2:
        # Heuristic: if [d, r] take col 0, if [r, d] take row 0
        t = t[:, 0] if t.shape[0] > t.shape[1] else t[0]

    return t / (t.norm() + 1e-12)

def make_orthogonal(v: torch.Tensor, seed: int) -> torch.Tensor:
    """Generates a random unit vector orthogonal to v."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn_like(v, generator=g)
    # Gram-Schmidt: r_orth = r - proj_v(r)
    r_orth = r - (torch.dot(r, v) * v)
    return r_orth / (r_orth.norm() + 1e-12)

def make_random(d_dim: int, seed: int) -> torch.Tensor:
    """Generates a random unit vector."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(d_dim, generator=g)
    return r / (r.norm() + 1e-12)

def project_out_vector(base_vector: torch.Tensor, remove_vector: torch.Tensor) -> torch.Tensor:
    """Removes the component of base_vector that is parallel to remove_vector."""
    # Ensure devices match
    if base_vector.device != remove_vector.device:
        remove_vector = remove_vector.to(base_vector.device, dtype=base_vector.dtype)
    
    dot = torch.dot(base_vector, remove_vector)
    return base_vector - (dot * remove_vector)

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
def prepare_wikitext_batches(
    tokenizer: AutoTokenizer, 
    num_samples: int, 
    seq_len: int, 
    batch_size: int
) -> List[torch.Tensor]:
    """Streams Wikipedia, tokenizes, and chunks into batches."""
    logger.info(f"Streaming Wikipedia (target: {num_samples} samples)...")
    
    # Set cache directory explicitly if not set (fixes "cached in None" error)
    from pathlib import Path
    if not os.environ.get("HF_DATASETS_CACHE") and not os.environ.get("HF_HOME"):
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
        logger.info(f"Set HF_DATASETS_CACHE to: {cache_dir}")
    
    # Load dataset with explicit cache_dir parameter
    cache_dir = os.environ.get("HF_DATASETS_CACHE") or str(Path.home() / ".cache" / "huggingface" / "datasets")
    try:
        ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, cache_dir=cache_dir)
    except Exception as e:
        logger.warning(f"Failed to load Wikipedia dataset: {e}")
        logger.info("Trying alternative: wikitext-2 (smaller, faster to load)...")
        # Fallback to wikitext-2 which is smaller and easier to load
        try:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True, cache_dir=cache_dir)
        except Exception as e2:
            logger.error(f"Failed to load wikitext-2: {e2}")
            raise RuntimeError(f"Could not load any dataset. Wikipedia error: {e}, Wikitext error: {e2}")
    
    texts = []
    for x in ds:
        if len(texts) >= num_samples: break
        if len(x["text"]) > 100: texts.append(x["text"])

    logger.info(f"Tokenizing {len(texts)} texts...")
    # Tokenize each text individually to avoid batching issues with different lengths
    all_token_ids = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", padding=False, truncation=False)
        all_token_ids.append(enc["input_ids"].squeeze(0))  # Remove batch dimension
    
    # Concatenate all token IDs into a single tensor
    if not all_token_ids:
        raise ValueError("No texts were tokenized. Check dataset loading.")
    ids = torch.cat(all_token_ids, dim=0)

    # Discard incomplete batch at the end
    n_tokens = (ids.numel() // seq_len) * seq_len
    if n_tokens == 0:
        raise ValueError("Not enough data fetched to create a single batch.")
        
    ids = ids[:n_tokens].view(-1, seq_len)
    
    batches = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]
    logger.info(f"Created {len(batches)} batches of shape [B, {seq_len}].")
    return batches

# -----------------------------------------------------------------------------
# Model Manager
# -----------------------------------------------------------------------------
class MixtralRouterManager:
    """
    Helper to locate, modify, and restore Mixtral router weights.
    Acts as a Context Manager to ensure weights are restored if code crashes.
    """
    def __init__(self, model: torch.nn.Module, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.gate_module = self._find_gate()
        self.original_weights = self.gate_module.weight.data.clone().cpu()
        self.device = self.gate_module.weight.device
        self.dtype = self.gate_module.weight.dtype

    def _find_gate(self) -> torch.nn.Module:
        # Mixtral structure: model.layers[i].block_sparse_moe.gate
        try:
            return self.model.model.layers[self.layer_idx].block_sparse_moe.gate
        except AttributeError:
            raise ValueError(f"Could not locate Mixtral gate at layer {self.layer_idx}")

    def apply_weights(self, new_weights: torch.Tensor):
        """Safely apply new weights to the model."""
        if new_weights.shape != self.original_weights.shape:
            raise ValueError(f"Shape mismatch: {new_weights.shape} vs {self.original_weights.shape}")
        self.gate_module.weight.data = new_weights.to(self.device, dtype=self.dtype)

    def restore(self):
        """Restore original weights."""
        logger.info("Restoring original router weights.")
        self.gate_module.weight.data = self.original_weights.to(self.device, dtype=self.dtype)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()

# -----------------------------------------------------------------------------
# Core Evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, batches: List[torch.Tensor]) -> float:
    model.eval()
    losses = []
    # Identify valid device for input (usually same as first layer)
    device = model.model.layers[0].self_attn.q_proj.weight.device

    for b in batches:
        b = b.to(device)
        # Mixtral forward pass
        out = model(input_ids=b, labels=b)
        losses.append(out.loss.item())
        
    return sum(losses) / len(losses) if losses else 0.0

# -----------------------------------------------------------------------------
# Main Experiment Logic
# -----------------------------------------------------------------------------
def run_ablation_experiment(cfg: ExperimentConfig):
    # 1. Setup Model & Data
    logger.info(f"Loading model: {cfg.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    
    batches = prepare_wikitext_batches(tokenizer, cfg.num_samples, cfg.seq_len, cfg.batch_size)

    # 2. Load Expert Vectors (CPU)
    logger.info("Loading cached expert SVD vectors...")
    expert_vectors_svd = {}
    for i in range(cfg.num_experts):
        fname = f"{cfg.model_tag}_layer{cfg.layer_idx}_expert{i}.pkl"
        path = os.path.join(cfg.svd_dir, fname)
        try:
            expert_vectors_svd[i] = load_single_expert_vector(path)
        except Exception as e:
            logger.warning(f"Skipping Expert {i} (Load failed: {e})")

    if not expert_vectors_svd:
        raise RuntimeError("No SVD vectors loaded. Check paths.")

    # 3. Run Experiments inside Context Manager
    results = {
        "config": vars(cfg),
        "results": {}
    }

    with MixtralRouterManager(model, cfg.layer_idx) as router:
        
        # A. Baseline
        logger.info("Evaluating Baseline...")
        base_loss = evaluate_loss(model, batches)
        results["baseline_loss"] = base_loss
        logger.info(f"Baseline Loss: {base_loss:.4f}")

        # B. Ablation Variations
        variations = ["svd", "orthogonal", "random"]
        
        for variant in variations:
            logger.info(f"--- Running Variant: Remove {variant.upper()} ---")
            
            # Start with a fresh copy of CPU weights to modify
            modified_weights = router.original_weights.clone() # [NumExperts, Dim]
            
            # Iterate over every expert that we have a vector for
            for exp_idx, v_svd in expert_vectors_svd.items():
                
                # Determine which vector to remove
                if variant == "svd":
                    v_remove = v_svd
                elif variant == "orthogonal":
                    v_remove = make_orthogonal(v_svd, seed=cfg.seed + exp_idx)
                elif variant == "random":
                    v_remove = make_random(v_svd.shape[0], seed=cfg.seed + exp_idx)
                
                # Project this vector out of the specific router row for this expert
                current_row = modified_weights[exp_idx]
                modified_weights[exp_idx] = project_out_vector(current_row, v_remove)

            # Apply modified weights to GPU model
            router.apply_weights(modified_weights)
            
            # Evaluate
            loss = evaluate_loss(model, batches)
            delta = loss - base_loss
            
            logger.info(f"Result [{variant}]: Loss={loss:.4f}, Delta={delta:+.4f}")
            results["results"][variant] = {"loss": loss, "delta": delta}

    # 4. Save
    logger.info(f"Saving results to {cfg.output_file}")
    with open(cfg.output_file, "w") as f:
        json.dump(results, f, indent=2)

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--svd_dir", type=str, required=True)
    parser.add_argument("--layer_idx", type=int, required=True)
    parser.add_argument("--output_file", type=str, default="ablation_results.json")
    parser.add_argument("--num_samples", type=int, default=200)
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        svd_dir=args.svd_dir,
        layer_idx=args.layer_idx,
        output_file=args.output_file,
        num_samples=args.num_samples
    )
    
    run_ablation_experiment(config)