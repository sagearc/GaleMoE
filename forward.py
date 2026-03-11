from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from safetensors.torch import save_file
from transformer_lens import HookedTransformer
from tqdm import tqdm

from src.forward.patched_blocks import (
    CacheKey,
    build_fwd_hooks,
    make_gate_logits_hook,
)

NUM_EXPERTS = 8
W_IDS = (1, 3)
LAYERS_TO_PATCH = [4, 16, 28]

BATCH_SIZE = 2000
N_BATCHES = 100

OUTPUT_DIR = "output"
WIKI_SEED = 42


def prepare_output_dir(output_dir: str) -> Path:
    """Prepare output directory structure."""
    output_dir: Path = Path(output_dir)
    output_dir.mkdir()

    for layer in LAYERS_TO_PATCH:
        router_dir: Path = output_dir / f"layer={layer:02}/router"
        router_dir.mkdir(parents=True)
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


def run_forward_pass(model: HookedTransformer, batch_text, fwd_hooks, batch_info) -> torch.Tensor:
    """Tokenize input, run forward pass with TransformerLens hooks."""
    inputs = model.tokenizer(batch_text, return_tensors="pt", truncation=True, max_length=32, padding=True)
    input_ids = inputs["input_ids"].to(model.cfg.device)
    attention_mask = inputs["attention_mask"].to(model.cfg.device)

    batch_info["seq_length"] = input_ids.shape[1]

    with torch.no_grad():
        model.run_with_hooks(input_ids, fwd_hooks=fwd_hooks, attention_mask=attention_mask)

    return input_ids


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


def save_router_logits_to_disk(router_logits: dict[int, torch.Tensor], output_dir: Path, batch, batch_id: int, batch_size: int, seq_length: int):
    """Save router logits for each layer to disk."""
    for layer_idx in LAYERS_TO_PATCH:
        file_path = output_dir / f"layer={layer_idx:02}/router/{batch_id:05}.safetensors"
        logits = router_logits[layer_idx]  # (batch_size * sequence_length, n_experts)
        logits = logits.view(batch_size, seq_length, NUM_EXPERTS)  # reshape to (batch_size, sequence_length, n_experts)
        tensors = {}
        for i, (row_id, prompt) in enumerate(zip(batch["id"], batch["title"])):
            tensor_name = f"{row_id}.{prompt}"
            tensors[tensor_name] = logits[i].cpu()
        save_file(tensors, str(file_path), metadata={"batch_size": str(BATCH_SIZE), "wiki_seed": str(WIKI_SEED)})


def patch_loop(model: HookedTransformer, loop: tqdm):
    """Register progress-tracking hooks on attention modules."""
    def hook(module, input, layer_idx):
        loop.set_description(f"Forward pass [layer {1 + layer_idx:02}/{model.cfg.n_layers}]")

    for layer_idx in range(model.cfg.n_layers):
        h = partial(hook, layer_idx=layer_idx)
        model.blocks[layer_idx].attn.register_forward_pre_hook(h)


def free_memory(activations_cache: dict[CacheKey, torch.Tensor]):
    """Free up unused memory."""
    for v in activations_cache.values():
        del v
    activations_cache.clear()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    output_dir = prepare_output_dir(OUTPUT_DIR)

    model = HookedTransformer.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1",
        dtype=torch.float16,
        device="cuda",
    )
    model.tokenizer.pad_token = model.tokenizer.eos_token

    print(f"Using device: {model.cfg.device}")
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
    routing_state = {}
    batch_info = {"seq_length": None}
    router_logits_store = {}

    fwd_hooks = build_fwd_hooks(
        layers_to_patch=LAYERS_TO_PATCH,
        num_experts=NUM_EXPERTS,
        cache=cache,
        routing_state=routing_state,
        batch_info=batch_info,
        row_idx_to_prompt=row_idx_to_prompt,
    )

    for layer_idx in LAYERS_TO_PATCH:
        model.blocks[layer_idx].mlp.W_gate.register_forward_hook(
            make_gate_logits_hook(layer_idx, router_logits_store)
        )

    for batch_id, batch in enumerate(loop):
        if batch_id >= N_BATCHES:
            break

        for i, (row_id, prompt) in enumerate(zip(batch["id"], batch["title"])):
            row_idx_to_prompt[i] = (row_id, prompt)

        input_ids = run_forward_pass(model, batch["title"], fwd_hooks, batch_info)
        batch_size, seq_length = input_ids.shape

        loop.set_description("Saving batch to disk")
        save_cache_to_disk(output_dir, cache, batch_id)
        save_router_logits_to_disk(router_logits=router_logits_store,
                                   output_dir=output_dir,
                                   batch=batch,
                                   batch_id=batch_id,
                                   batch_size=batch_size,
                                   seq_length=seq_length)

        print(f"Saved total of {len(cache)} tensors in batch {batch_id}\n")

        del input_ids
        free_memory(cache)
        routing_state.clear()
        for v in router_logits_store.values():
            del v
        router_logits_store.clear()

        loop.set_description("Forward pass")
