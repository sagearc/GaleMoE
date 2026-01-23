import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from torch.utils.hooks import RemovableHandle
from datasets import load_dataset
from typing import Dict, List, Set, NamedTuple, Optional
from datasets.arrow_dataset import Dataset
import json
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MoeCausalLMOutputWithPast
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.models.mixtral.modeling_mixtral import MixtralBlockSparseTop2MLP, MixtralSparseMoeBlock
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import types


NUM_EXPERTS = 8
EXPERT_IDS = (1, 3)
LAYER_IDX = 20


class ExpertLayerInfo(NamedTuple):
    layer_idx: int
    expert_id: int
    weight_type: int
    path: str


def patched_moe_forward(self: MixtralSparseMoeBlock, hidden_states: torch.Tensor):
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
            is_last = ((top_x + 1) == self.num_experts)
            print(f"Processing Expert {expert_idx.item()} (is_last: {is_last.any().item()})")
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            print(f"----- Expert {expert_idx.item()} -----")
            print(f"Expert {expert_idx.item()} - Processing {top_x.shape[0]} tokens")
            print(f"Top-x indices: {top_x}")
            print(f"Token indices in batch: {idx}")
            print(f"Routing weights: {routing_weights[top_x, idx]}")
            print()
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


# NumPy structured dtype for ExpertLayerInfo
EXPERT_LAYER_INFO_DTYPE = np.dtype([
    ('layer_idx', np.int16),
    ('expert_id', np.int8),
    ('weight_type', np.int8),
])


def get_expert_layer_names(layer_idx: int = LAYER_IDX) -> List[ExpertLayerInfo]:
    """Generate list of expert layer names to hook with metadata."""
    expert_layer_info = []
    for j in range(NUM_EXPERTS):
        for k in EXPERT_IDS:
            layer_path = f"model.layers.{layer_idx}.block_sparse_moe.experts.{j}.w{k}"
            expert_metadata = ExpertLayerInfo(
                layer_idx=layer_idx,
                expert_id=j,
                weight_type=k,
                path=layer_path,
            )
            expert_layer_info.append(expert_metadata)
    return expert_layer_info


# Hook for expert weights that tracks which tokens are routed to it
def get_w_activation_hook(expert_layer_info: ExpertLayerInfo, activations: Dict[ExpertLayerInfo, torch.Tensor], token_routing):
    """Capture W1 and W3 activations and track token routing."""
    layer_idx = expert_layer_info.layer_idx
    expert_id = expert_layer_info.expert_id
    
    def hook(module, input, output: torch.Tensor):
        print("-----INPUT---------")
        #type
        print(f"type: {type(input)}")
        print(f"len: {len(input)}")
        print(f"input[0] shape: {input[0].shape}")
        # Store activation tensor
        # Shape: [num_tokens_routed_to_expert, hidden_dim]
        activations[expert_layer_info] = output.clone()
        
        # Track that this expert was used (tokens will be tracked via batch processing)
        if layer_idx not in token_routing:
            token_routing[layer_idx] = {}
        if expert_id not in token_routing[layer_idx]:
            token_routing[layer_idx][expert_id] = True
            
    return hook

def get_gate_logits_hook(layer_idx: int, token_routing):
    """Hook to capture gate logits and track token routing."""
    def hook(module, input, output: torch.Tensor):
        # Output shape (router logits): [batch_size * seq_len, num_experts]
        print(f"Gate Layer {layer_idx} - Output shape: {output.shape}")

        _, selected_experts = torch.topk(output, k=2, dim=-1)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=NUM_EXPERTS).permute(2, 1, 0)
        
        # Determine top-2 experts for each token
        top2_experts = output.topk(2, dim=-1).indices  # Shape: [batch_size, seq_len, 2]
        
        # Track which experts were used for this layer
        if layer_idx not in token_routing:
            token_routing[layer_idx] = {}
        
        batch_size, seq_len, _ = top2_experts.shape
        for b in range(batch_size):
            for s in range(seq_len):
                for expert_id in top2_experts[b, s].tolist():
                    if expert_id not in token_routing[layer_idx]:
                        token_routing[layer_idx][expert_id] = True
                    
    return hook


def setup_hooks(model: torch.nn.Module, expert_layer_info: List[ExpertLayerInfo]):
    """Register expert weight hooks only (no gate hook needed)."""
    activations: Dict[ExpertLayerInfo, torch.Tensor] = {}
    token_routing = {}
    handles = []
    
    named_modules = dict(model.named_modules())
    
    # Register expert weight hooks with metadata
    for info in expert_layer_info:
        target_layer = named_modules[info.path]
        hook = get_w_activation_hook(
            info,
            activations, 
            token_routing
        )
        
        act_name = f"layers.{info.layer_idx}.experts.{info.expert_id}.w{info.weight_type}"
        print(f"Registering hook on: {act_name}")
        
        handle = target_layer.register_forward_hook(hook)
        handles.append(handle)
    
    print(f"Registered {len(handles)} expert hooks.")
    
    return handles, activations, token_routing


def load_wiki_dataset() -> Dataset:
    """Load Wikipedia dataset and return it with only title column."""
    print("\nLoading Wikipedia dataset...")
    ds: Dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    ds = ds.remove_columns(["url", "text"])
    ds = ds.shuffle(seed=42)
    return ds


def run_forward_pass(model: MixtralForCausalLM, tokenizer, batch_text, activations, token_routing):
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
    print(f"{'='*80}")
    
    for layer_idx in sorted(token_routing.keys()):
        expert_ids = sorted(token_routing[layer_idx].keys())
        print(f"Layer {layer_idx}: Experts {expert_ids}")
    
    print(f"{'='*80}\n")
    
    return output, input_ids, batch_size, last_token_positions


def extract_last_token_activations(activations, token_routing, batch_text, layer_idx=LAYER_IDX):
    """Extract W1 and W3 activations for the batch.
    
    Returns: Dict with batch text and activations for each expert that was used
    """
    batch_activations = {
        "texts": batch_text,
        "layer": layer_idx,
        "expert_activations": {}
    }
    
    print(f"\nExtracting activations from {len(activations)} expert layers")
    
    # Iterate through all activations that were captured
    for info_key, activation in activations.items():
        if info_key.layer_idx == layer_idx:
            # Use string key for output
            act_name = f"layers.{info_key.layer_idx}.experts.{info_key.expert_id}.w{info_key.weight_type}"
            batch_activations["expert_activations"][act_name] = activation
            print(f"  {act_name}: shape {activation.shape}")
    
    return batch_activations



def save_activations(batch_activations, filename="activations.pt"):
    """Save batch activations to file."""
    # Convert to saveable format
    save_data = {
        "texts": batch_activations["texts"],
        "layer": batch_activations["layer"],
        "activations": {}
    }
    
    for expert_name, activation in batch_activations["expert_activations"].items():
        save_data["activations"][expert_name] = {
            "activation": activation.cpu(),
            "shape": list(activation.shape),
            "mean": activation.mean().item(),
            "std": activation.std().item(),
        }
    
    torch.save(save_data, filename)
    print(f"\nSaved activations to {filename}")
    
    # Also save a JSON summary
    summary = {
        "texts": batch_activations["texts"],
        "layer": batch_activations["layer"],
        "activations": {}
    }
    
    for expert_name, activation in batch_activations["expert_activations"].items():
        summary["activations"][expert_name] = {
            "shape": list(activation.shape),
            "mean": float(activation.mean().item()),
            "std": float(activation.std().item()),
        }
    
    with open(filename.replace(".pt", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved activation summary to {filename.replace('.pt', '_summary.json')}")



if __name__ == "__main__":
    model = MixtralForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", dtype=torch.bfloat16)
    tokenizer = LlamaTokenizerFast.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

    config = model.config
    
    # Monkey-patch all MixtralSparseMoeBlock forward methods
    for name, module in model.named_modules():
        if isinstance(module, MixtralSparseMoeBlock):
            # Bind the patched function as a method to this specific module instance
            module.forward = types.MethodType(patched_moe_forward, module)
            print(f"Patched forward method at {name}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    expert_layer_info = get_expert_layer_names(layer_idx=LAYER_IDX)
    handles, activations, token_routing = setup_hooks(model, expert_layer_info)
    
    # Load data and run forward pass
    ds = load_wiki_dataset()
    batch_text = [ds[i]["title"] for i in range(3)]  # Process 3 samples
    output, input_ids, batch_size, last_token_positions = run_forward_pass(
        model, tokenizer, batch_text, activations, token_routing
    )
    
    # Extract and save activations
    print("\n---- Extracting Activations ----")
    batch_activations = extract_last_token_activations(
        activations, token_routing, batch_text
    )
    
    # Display extracted activations
    print("\n---- Activation Summary ----")
    print(f"Texts: {batch_activations['texts']}")
    print(f"Layer: {batch_activations['layer']}")
    for expert_name, activation in batch_activations["expert_activations"].items():
        print(f"  {expert_name}: shape {activation.shape}, mean {activation.mean():.6f}, std {activation.std():.6f}")
    
    # Save activations
    save_activations(batch_activations)
    
    # Cleanup hooks
    for handle in handles:
        handle.remove()
