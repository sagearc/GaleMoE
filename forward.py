import types
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from safetensors.torch import save_file
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.models.mixtral.modeling_mixtral import (
    MixtralBlockSparseTop2MLP,
    MixtralForCausalLM,
    MixtralSparseMoeBlock,
    MoeCausalLMOutputWithPast,
)

NUM_EXPERTS = 8
W_IDS = (1, 3)
LAYER_IDX = 20

BATCH_SIZE = 3
N_BATCHES = 10

OUTPUT_DIR = "output"
WIKI_SEED = 42

class Key(NamedTuple):
    layer_idx: int
    expert_id: int
    w_id: int
    prompt: int
    row_id: int


cache: dict[Key, torch.Tensor] = {}
row_idx_to_prompt = [None for _ in range(BATCH_SIZE)]


# Monkey-patched MixtralBlockSparseTop2MLP.forward method
def patched_block_sparse_top2_mlp_forward(self: MixtralBlockSparseTop2MLP, hidden_states: torch.Tensor, top_x: torch.Tensor, sequence_length: int):
    """ """
    print(f"Expert {self.galemoe_expert_id} - Top-x indices: {top_x}")

    W1x: torch.Tensor = self.w1(hidden_states)
    W3x: torch.Tensor = self.w3(hidden_states)

    is_last = ((top_x + 1) % sequence_length) == 0
    is_last_indices = torch.nonzero(is_last).squeeze(-1)

    # save to cache
    for i in is_last_indices:
        prompt_idx = (top_x[i] // sequence_length).item()
        row_id_in_dataset, prompt = row_idx_to_prompt[prompt_idx]

        key1 = Key(
            layer_idx=self.galemoe_layer_idx,
            expert_id=self.galemoe_expert_id,
            w_id=1,
            prompt=prompt,
            row_id=row_id_in_dataset,
        )
        cache[key1] = W1x[i].cpu()

        key3 = Key(
            layer_idx=self.galemoe_layer_idx,
            expert_id=self.galemoe_expert_id,
            w_id=3,
            prompt=prompt,
            row_id=row_id_in_dataset,
        )
        cache[key3] = W3x[i].cpu()

    current_hidden_states = self.act_fn(W1x) * W3x
    current_hidden_states = self.w2(current_hidden_states)
    return current_hidden_states


# Monkey-patched MixtralSparseMoeBlock.forward method
def patched_sparse_moe_block_forward(self: MixtralSparseMoeBlock, hidden_states: torch.Tensor):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(hidden_states=current_state, top_x=top_x, sequence_length=sequence_length) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


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
    
    print("\n---- Forward Pass ----")
    print(f"INPUT IDS:\n{input_ids}")
    print(f"\nInput shape: {input_ids.shape}\n")
    print(f"Batch texts: {batch_text}")
    
    # Forward pass (no gradients needed)
    with torch.no_grad():
        output: MoeCausalLMOutputWithPast = model(input_ids=input_ids, attention_mask=attention_mask, output_router_logits=False)
    
    return output, input_ids



if __name__ == "__main__":
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    model = MixtralForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", dtype=torch.bfloat16)
    model.eval()

    tokenizer = LlamaTokenizerFast.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    
    named_modules = dict(model.named_modules())

    print("Patching MixtralSparseMoeBlock and MixtralBlockSparseTop2MLP forward methods...")
    layers_to_patch = [0, 1, LAYER_IDX]
    for layer_idx in layers_to_patch:
        layer_path = f"model.layers.{layer_idx}.block_sparse_moe"
        module = named_modules[layer_path]
        assert isinstance(module, MixtralSparseMoeBlock)
        module.forward = types.MethodType(patched_sparse_moe_block_forward, module)
        module.galemoe_layer_idx = layer_idx

        for expert in range(NUM_EXPERTS):
            layer_path = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert}"
            module = named_modules[layer_path]
            assert isinstance(module, MixtralBlockSparseTop2MLP)
            module.forward = types.MethodType(patched_block_sparse_top2_mlp_forward, module)
            module.galemoe_expert_id = expert
            module.galemoe_layer_idx = layer_idx    

    ds = load_wiki_dataset(seed=WIKI_SEED)
    for batch_id, batch in enumerate(ds.iter(BATCH_SIZE)):
        if batch_id >= N_BATCHES:
            break

        print(f"\n=== Processing batch {batch_id} ===")
        for i, (row_id, prompt) in enumerate(zip(batch["id"], batch["title"])):
            row_idx_to_prompt[i] = (row_id, prompt)

        output, input_ids = run_forward_pass(
            model, tokenizer, row_idx_to_prompt
        )

        # group by layer, expert, w_id and save to disk

        grouped_cache = defaultdict(dict)
        for key, weight in cache.items():
            group = f"layer={key.layer_idx:02}/expert={key.expert_id}/w={key.w_id}"
            tensor_prompt = f"{key.row_id}.{key.prompt}"
            grouped_cache[group][tensor_prompt] = weight
        
        for group, tensors in grouped_cache.items():
            group_dir: Path = output_dir / group
            group_dir.mkdir(parents=True, exist_ok=True)
            file_path = group_dir / f"{batch_id:05}.safetensors"
            save_file(tensors, str(file_path), metadata={"batch_size": str(BATCH_SIZE), "wiki_seed": str(WIKI_SEED)})
            print(f"Saved {len(tensors)} tensors to {file_path}")

        print("\n---- Cached Weights ----")
        print(cache.popitem())
        print(len(cache), "weights cached.")
