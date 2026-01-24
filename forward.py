import types
from collections import defaultdict
from pathlib import Path
from functools import partial

import torch
import torch.nn.functional as F
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from safetensors.torch import save_file
from transformers import BitsAndBytesConfig
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.models.mixtral.modeling_mixtral import (
    MixtralBlockSparseTop2MLP,
    MixtralAttention,
    MixtralForCausalLM,
    MixtralSparseMoeBlock,
    MoeCausalLMOutputWithPast,
)
from tqdm import tqdm

from src.forward.patched_blocks import (
    CacheKey,
    patched_block_sparse_top2_mlp_forward,
    patched_sparse_moe_block_forward,
)

NUM_EXPERTS = 8
W_IDS = (1, 3)
LAYERS_TO_PATCH = [4, 16, 28]

BATCH_SIZE = 10000
N_BATCHES = 100

OUTPUT_DIR = "output"
WIKI_SEED = 42


def prepare_output_dir(output_dir: str) -> Path:
    """Prepare output directory structure."""
    output_dir: Path = Path(output_dir)
    output_dir.mkdir()

    for layer in LAYERS_TO_PATCH:
        for expert in range(NUM_EXPERTS):
            for w_id in W_IDS:
                group_dir: Path = output_dir / f"layer={layer:02}/expert={expert}/w={w_id}"
                group_dir.mkdir(parents=True)
    return output_dir


def load_wiki_dataset(seed: int) -> Dataset:
    """Load Wikipedia dataset and return it with only title column."""
    print("\nLoading Wikipedia dataset...")
    ds: Dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    ds = ds.remove_columns(["url", "text"])
    ds = ds.shuffle(seed=seed)
    return ds


def run_forward_pass(model: MixtralForCausalLM, tokenizer, batch_text):
    """Tokenize input, run forward pass, and track expert activations."""
    # Tokenize the input
    inputs = tokenizer(batch_text, return_tensors="pt", truncation=True, max_length=32, padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Forward pass (no gradients needed)
    with torch.no_grad():
        output: MoeCausalLMOutputWithPast = model(input_ids=input_ids, attention_mask=attention_mask, output_router_logits=False)
    
    return output, input_ids


def save_cache_to_disk(output_dir: Path, cache: dict[CacheKey, torch.Tensor], batch_id: int):
    """Save cached tensors to disk."""
    grouped_cache = defaultdict(dict)
    for key, weight in cache.items():
        group = f"layer={key.layer_idx:02}/expert={key.expert_id}/w={key.w_id}"
        tensor_prompt = f"{key.row_id}.{key.prompt}"
        grouped_cache[group][tensor_prompt] = weight
    
    for group, tensors in grouped_cache.items():
        group_dir: Path = output_dir / group
        file_path = group_dir / f"{batch_id:05}.safetensors"
        save_file(tensors, str(file_path), metadata={"batch_size": str(BATCH_SIZE), "wiki_seed": str(WIKI_SEED)})


def patch_loop(model: MixtralForCausalLM, loop: tqdm):
    """Patch the loop into each MixtralSparseMoeBlock for progress tracking."""
    def hook(module, input, layer_idx):
        loop.set_description(f"Forward pass [layer {1 + layer_idx:02}/32]")

    named_modules = dict(model.named_modules())
    for layer_idx in range(model.config.num_hidden_layers):
        layer_path = f"model.layers.{layer_idx}.self_attn"
        module = named_modules[layer_path]
        assert isinstance(module, MixtralAttention)
        h = partial(hook, layer_idx=layer_idx)
        module.register_forward_pre_hook(h)

if __name__ == "__main__":
    output_dir = prepare_output_dir(OUTPUT_DIR)
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = MixtralForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1",
        quantization_config=quantization_config,
        device_map="auto")

    model.eval()

    tokenizer = LlamaTokenizerFast.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Using device: {model.device}")
    print(f"Number of experts: {NUM_EXPERTS}")
    print(f"Layers to patch: {LAYERS_TO_PATCH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches: {N_BATCHES}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Wikipedia seed: {WIKI_SEED}")
    print()

    ds = load_wiki_dataset(seed=WIKI_SEED)
    loop = tqdm(ds.iter(BATCH_SIZE), total=N_BATCHES)
    patch_loop(model, loop)

    cache: dict[CacheKey, torch.Tensor] = {}
    row_idx_to_prompt = [None for _ in range(BATCH_SIZE)]
    named_modules = dict(model.named_modules())

    print("Patching MixtralSparseMoeBlock and MixtralBlockSparseTop2MLP forward methods...")
    for layer_idx in LAYERS_TO_PATCH:
        layer_path = f"model.layers.{layer_idx}.block_sparse_moe"
        module = named_modules[layer_path]
        assert isinstance(module, MixtralSparseMoeBlock)
        module.forward = types.MethodType(patched_sparse_moe_block_forward, module)
        module.patch_layer_idx = layer_idx

        for expert in range(NUM_EXPERTS):
            layer_path = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert}"
            module = named_modules[layer_path]
            assert isinstance(module, MixtralBlockSparseTop2MLP)
            module.forward = types.MethodType(patched_block_sparse_top2_mlp_forward, module)
            module.patch_expert_id = expert
            module.patch_layer_idx = layer_idx
            module.patch_cache = cache
            module.patch_row_idx_to_prompt = row_idx_to_prompt

    for batch_id, batch in  enumerate(loop):
        if batch_id >= N_BATCHES:
            break

        for i, (row_id, prompt) in enumerate(zip(batch["id"], batch["title"])):
            row_idx_to_prompt[i] = (row_id, prompt)

        output, input_ids = run_forward_pass(model, tokenizer, row_idx_to_prompt)

        loop.set_description(f"Saving batch to disk")
        # group by layer, expert, w_id and save to disk
        save_cache_to_disk(output_dir, cache, batch_id)

        print(f"Saved total of {len(cache)} tensors in batch {batch_id}\n")
        cache.clear()
        assert len(cache) == 0

        loop.set_description("Forward pass")
