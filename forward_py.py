import types
from collections import defaultdict
from pathlib import Path
from functools import partial

import torch
import torch.nn.functional as F
import pandas as pd
from safetensors.torch import save_file
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import (
    MixtralBlockSparseTop2MLP,
    MixtralAttention,
    MixtralForCausalLM,
    MixtralSparseMoeBlock,
    MoeCausalLMOutputWithPast,
)
from tqdm import tqdm
import pandas as pd
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import (
    MixtralBlockSparseTop2MLP,
    MixtralAttention,
    MixtralForCausalLM,
    MixtralSparseMoeBlock,
    MoeCausalLMOutputWithPast,
)
from tqdm import tqdm
import pandas as pd

from src.forward.patched_blocks import (
    CacheKey,
    patched_block_sparse_top2_mlp_forward,
    patched_sparse_moe_block_forward,
)

NUM_EXPERTS = 8
W_IDS = (1, 3)
LAYERS_TO_PATCH = [i for i in range(32)]

OUTPUT_DIR = "output/common_words_1k"


def prepare_output_dir(output_dir: str) -> Path:
    """Prepare output directory structure."""
    output_dir: Path = Path(output_dir)
    output_dir.mkdir(parents=True)

    for layer in LAYERS_TO_PATCH:
        router_dir: Path = output_dir / f"layer={layer:02}"
        router_dir.mkdir()
        for expert in range(NUM_EXPERTS):
                group_dir: Path = output_dir / f"layer={layer:02}/expert={expert}"
                group_dir.mkdir()
    return output_dir


def run_forward_pass(model: MixtralForCausalLM, tokenizer, batch_text) -> tuple[MoeCausalLMOutputWithPast, torch.Tensor]:
    """Tokenize input, run forward pass, and track expert activations."""
    # Tokenize the input
    inputs = tokenizer(batch_text, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Forward pass (no gradients needed)
    with torch.no_grad():
        output: MoeCausalLMOutputWithPast = model(input_ids=input_ids, attention_mask=attention_mask, output_router_logits=True)    

    return output, input_ids


def save_cache_to_disk(output_dir: Path, cache: dict[CacheKey, torch.Tensor]):
    """Save cached tensors to disk."""
    grouped_cache = defaultdict(dict)
    for key, weight in cache.items():
        group = f"layer={key.layer_idx:02}/expert={key.expert_id}/w={key.w_id}"
        tensor_prompt = f"{key.row_id}.{key.prompt}"
        grouped_cache[group][tensor_prompt] = weight
    
    for group, tensors in grouped_cache.items():
        file_path: Path = (output_dir / group)
        file_path = file_path.with_suffix('.safetensors')
        save_file(tensors, str(file_path))


def save_router_logits_to_disk(router_logits: tuple[torch.FloatTensor], output_dir: Path, repo_ranks: list, prompts: list, batch_size: int, seq_length: int):
    """Save router logits for each layer to disk."""
    for layer_idx in LAYERS_TO_PATCH:
        file_path = (output_dir / f"layer={layer_idx:02}/router").with_suffix('.safetensors')
        logits = router_logits[layer_idx]  # (batch_size * sequence_length, n_experts)
        logits = logits.view(batch_size, seq_length, NUM_EXPERTS)  # reshape to (batch_size, sequence_length, n_experts)
        tensors = {}
        for i, (row_id, prompt) in enumerate(zip(repo_ranks, prompts)):
            tensor_name = f"{row_id}.{prompt}"
            tensors[tensor_name] = logits[i].cpu()
        save_file(tensors, str(file_path))


def patch_loop(model: MixtralForCausalLM, loop: tqdm):
    """Patch the loop into each MixtralSparseMoeBlock for progress tracking."""
    def hook(module, input, layer_idx):
        loop.set_description(f"Forward pass [layer {1 + layer_idx:02}/32]")
        loop.update((1 / len(LAYERS_TO_PATCH)) + loop.n)

    named_modules = dict(model.named_modules())
    for layer_idx in range(model.config.num_hidden_layers):
        layer_path = f"model.layers.{layer_idx}.self_attn"
        module = named_modules[layer_path]
        assert isinstance(module, MixtralAttention)
        h = partial(hook, layer_idx=layer_idx)
        module.register_forward_pre_hook(h)


def free_memory(activations_cache: dict[CacheKey, torch.Tensor]):
    """Free up unused memory."""
    for v in activations_cache.values():
        del v
    activations_cache.clear()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    output_dir = prepare_output_dir(OUTPUT_DIR)

    df_genesis = pd.read_csv('data/genesis.csv')
    df_genesis_by_chapter = df_genesis.groupby('chapter').apply(lambda g: g.sort_values('verse')['text'].tolist())
    df_genesis_by_chapter = df_genesis_by_chapter.apply(lambda verses: '\n'.join(verses))
    df = df_genesis_by_chapter.to_frame(name='text').reset_index()

    with open('common_words.txt', 'r') as f:
        lines = f.readlines()
    common_words = [line.strip() for line in lines if not line.startswith('#')][:1000]

    prompts = common_words
    repo_ranks = [i for i in range(len(prompts))]
    df = pd.DataFrame({'chapter': repo_ranks, 'text': prompts})

    # USE ALL EXPERTS FORWARD PASS:
    config = MixtralConfig.from_pretrained("mistralai/Mixtral-8x7B-v0.1",
                                           num_experts_per_tok=8)

    model = MixtralForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1",
        config=config,
        dtype=torch.bfloat16,
        device_map="auto")

    model.eval()

    tokenizer = LlamaTokenizerFast.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Using device: {model.device}")
    print(f"Number of experts: {NUM_EXPERTS}")
    print(f"Layers to patch: {LAYERS_TO_PATCH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    loop = tqdm(range(1), total=1)
    patch_loop(model, loop)

    cache: dict[CacheKey, torch.Tensor] = {}
    row_idx_to_prompt = [None for _ in range(len(df))]
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


    iter_loop = iter(loop)

    prompts = common_words
    repo_ranks = [i for i in range(len(prompts))]
    # repo_ranks = df['chapter'].tolist()[:10]
    # prompts = df['text'].tolist()[:10]

    for i, (row_id, prompt) in enumerate(zip(repo_ranks, prompts)):
        row_idx_to_prompt[i] = (row_id, prompt)

    output, input_ids = run_forward_pass(model, tokenizer, prompts)
    batch_size, seq_length = input_ids.shape

    loop.set_description(f"Saving batch to disk")
    # group by layer, expert, w_id and save to disk
    save_cache_to_disk(output_dir, cache)
    save_router_logits_to_disk(router_logits=output.router_logits,
                                output_dir=output_dir,
                                repo_ranks=repo_ranks,
                                prompts=prompts,
                                batch_size=batch_size,
                                seq_length=seq_length)

    print(f"Saved total of {len(cache)} tensors\n")

    del output
    del input_ids
    free_memory(cache)

    loop.set_description("Forward pass")
