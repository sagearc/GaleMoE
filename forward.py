import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from torch.utils.hooks import RemovableHandle
from datasets import load_dataset
from typing import Dict, List, Set, NamedTuple
from datasets.arrow_dataset import Dataset
import json
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MoeCausalLMOutputWithPast
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
import numpy as np


NUM_EXPERTS = 8
EXPERT_IDS = (1, 3)
LAYER_IDX = 20


class ExpertLayerInfo(NamedTuple):
    layer_idx: int
    expert_id: int
    weight_type: int
    path: str

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
