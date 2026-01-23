import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from torch.utils.hooks import RemovableHandle
# from datasets import load_dataset
from typing import Dict, List, Set, NamedTuple, Optional
# from datasets.arrow_dataset import Dataset
import json
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MoeCausalLMOutputWithPast
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.models.mixtral.modeling_mixtral import MixtralBlockSparseTop2MLP, MixtralSparseMoeBlock
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import types


NUM_EXPERTS = 8
W_IDS = (1, 3)
LAYER_IDX = 20


class ExpertLayerInfo(NamedTuple):
    layer_idx: int
    expert_id: int
    weight_type: int
    path: str

# Monkey-patched MixtralBlockSparseTop2MLP.forward method
def patched_block_sparse_top2_mlp_forward(self: MixtralBlockSparseTop2MLP, hidden_states: torch.Tensor, top_x: torch.Tensor, sequence_length: int):
    """ """
    print(f"Expert {self.galemoe_expert_id} - Top-x indices: {top_x}")
    is_last = ((top_x + 1) % sequence_length) == 0
    hidden_states = hidden_states[0]
    W1x = self.w1(hidden_states)
    W3x = self.w3(hidden_states)

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


# def get_expert_layer_names(layer_idx: int = LAYER_IDX) -> List[ExpertLayerInfo]:
#     """Generate list of expert layer names to hook with metadata."""
#     expert_layer_info = []
#     for j in range(NUM_EXPERTS):
#         for k in W_IDS:
#             layer_path = f"model.layers.{layer_idx}.block_sparse_moe.experts.{j}.w{k}"
#             expert_metadata = ExpertLayerInfo(
#                 layer_idx=layer_idx,
#                 expert_id=j,
#                 weight_type=k,
#                 path=layer_path,
#             )
#             expert_layer_info.append(expert_metadata)
#     return expert_layer_info

# def load_wiki_dataset() -> Dataset:
#     """Load Wikipedia dataset and return it with only title column."""
#     print("\nLoading Wikipedia dataset...")
#     ds: Dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
#     ds = ds.remove_columns(["url", "text"])
#     ds = ds.shuffle(seed=42)
#     return ds


def run_forward_pass(model: MixtralForCausalLM, tokenizer, batch_text):
    """Tokenize input, run forward pass, and track expert activations."""
    # Tokenize the input
    inputs = tokenizer(batch_text, return_tensors="pt", truncation=True, max_length=32, padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    batch_size, seq_len = input_ids.shape
    print("\n---- Forward Pass ----")
    print(f"INPUT IDS:\n{input_ids}\n")
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Batch texts: {batch_text}")
    
    # Find actual last token position for each batch item (ignoring padding)
    last_token_positions = (attention_mask.sum(dim=1) - 1).cpu().tolist()
    print(f"Last token positions (non-padded): {last_token_positions}")
    
    # Forward pass (no gradients needed)
    with torch.no_grad():
        output: MoeCausalLMOutputWithPast = model(input_ids=input_ids, attention_mask=attention_mask, output_router_logits=True)
        # print router logits if needed
        print(f"Router logits captured during forward pass.")
        print(f"Output logits shape: {len(output.router_logits)} layers") 
        print(f"Output logits shape (layer 0): {output.router_logits[LAYER_IDX].shape}")
        curr_layer_logits = output.router_logits[LAYER_IDX]  # Shape: [batch_size, seq_len, num_experts]
        # How many tokens were routed to each expert by top 2 for each token
        top2_experts = curr_layer_logits.topk(2, dim=-1).indices
        expert_token_counts = top2_experts.flatten().bincount(minlength=NUM_EXPERTS)
        print(f"Token counts per expert at layer {LAYER_IDX}: {expert_token_counts.tolist()}")

        
    
    # Log which experts were activated
    print(f"\n{'='*80}")
    print(f"EXPERTS ACTIVATED (Layer {LAYER_IDX})")
    print(f"{'='*80}\n")
    
    return output, input_ids, batch_size, last_token_positions



if __name__ == "__main__":
    model = MixtralForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", dtype=torch.bfloat16)
    tokenizer = LlamaTokenizerFast.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

    config = model.config
    
    named_modules = dict(model.named_modules())

    print("Patching MixtralSparseMoeBlock and MixtralBlockSparseTop2MLP forward methods...")
    for layer_idx in [0, 1, LAYER_IDX]:
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


    
    # Set model to evaluation mode
    model.eval()
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # expert_layer_info = get_expert_layer_names(layer_idx=LAYER_IDX)
    
    # Load data and run forward pass
    # ds = load_wiki_dataset()
    # batch_text = [ds[i]["title"] for i in range(1000)]  # Process 3 samples
    batch_text = [
        "Artificial intelligence",
        "Brown fox",
        "OpenAI"
    ]
    output, input_ids, batch_size, last_token_positions = run_forward_pass(
        model, tokenizer, batch_text
    )
