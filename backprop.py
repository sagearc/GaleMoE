import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from torch.utils.hooks import RemovableHandle
from datasets import load_dataset
from typing import Dict, List, Set
import torch
from datasets.arrow_dataset import Dataset
import json


NUM_EXPERTS = 8
EXPERT_IDS = (1, 3)
LAYER_NAME = 20


def get_expert_layer_names(layer_name: int = LAYER_NAME):
    """Generate list of expert layer names to hook."""
    expert_layer_names = [
        f"model.layers.{layer_name}.block_sparse_moe.experts.{j}.w{k}" 
        for j in range(NUM_EXPERTS) 
        for k in EXPERT_IDS
    ]
    expert_layer_names.sort()
    return expert_layer_names

# Hook to capture gate routing decisions per batch item
def get_gate_hook(layer_idx, expert_routing_map):
    """Track which experts are selected for which tokens.
    expert_routing_map[layer_idx][(batch_idx, token_idx)] = {expert_ids}
    """
    def hook(module, input, output):
        # output is tuple: (final_hidden_states, router_logits)
        hidden_states = input[0]  # Input hidden states
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if len(output) > 1 and output[1] is not None:
            router_logits = output[1]  # Shape: [batch_size * seq_len, num_experts]
            
            # Reshape to [batch_size, seq_len, num_experts]
            router_logits_reshaped = router_logits.view(batch_size, seq_len, -1)
            
            # Get top-2 experts for each token
            top_k = torch.topk(router_logits_reshaped, k=2, dim=-1)
            
            # Store expert indices for each (batch_idx, token_idx)
            if layer_idx not in expert_routing_map:
                expert_routing_map[layer_idx] = {}
            
            for batch_idx in range(batch_size):
                for token_idx in range(seq_len):
                    expert_indices = top_k.indices[batch_idx, token_idx].tolist()
                    expert_routing_map[layer_idx][(batch_idx, token_idx)] = set(expert_indices)
    return hook


# Define the hook function for expert weights
def get_w_activation_hook(name, activations, activation_positions):
    """Track activations and their positions.
    activation_positions[name] tracks cumulative count of tokens processed.
    """
    def hook(module, input, output: torch.Tensor):
        # Detach and create a new tensor that requires grad
        output_detached = output.detach().requires_grad_(True)
        # Store activation tensor
        # Shape: [num_tokens_routed_to_expert, hidden_dim]
        activations[name] = output_detached
        # Track how many tokens have been processed by this expert so far
        if name not in activation_positions:
            activation_positions[name] = 0
        activation_positions[name] += output.shape[0]
    return hook


def setup_hooks(model, expert_layer_names, layer_name: int = LAYER_NAME):
    """Register gate and expert weight hooks."""
    activations = {}
    activation_positions = {}
    expert_routing_map = {}
    handles = []
    
    named_modules = dict(model.named_modules())
    
    # Register gate hook on block_sparse_moe to track expert routing
    moe_layer = named_modules[f"model.layers.{layer_name}.block_sparse_moe"]
    gate_hook = get_gate_hook(layer_name, expert_routing_map)
    gate_handle = moe_layer.register_forward_hook(gate_hook)
    handles.append(gate_handle)
    
    # Register expert weight hooks
    for target_layer_name in expert_layer_names:
        target_layer = named_modules[target_layer_name]
        clean_layer_name = target_layer_name.removeprefix("model.").replace("block_sparse_moe.", "")
        hook = get_w_activation_hook(clean_layer_name, activations, activation_positions)
        
        print(f"Registering hook on layer: {clean_layer_name}")
        
        handle = target_layer.register_forward_hook(hook)
        handles.append(handle)
    
    print(f"Registered hooks on {len(handles)} layers (gates + experts).")
    
    return handles, activations, activation_positions, expert_routing_map


def load_wiki_dataset() -> Dataset:
    """Load Wikipedia dataset and return it with only title column."""
    print("\nLoading Wikipedia dataset...")
    ds: Dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    ds = ds.remove_columns(["url", "text"])
    ds = ds.shuffle(seed=42)
    return ds


def run_forward_pass(model, tokenizer, batch_text, activations, expert_routing_map):
    """Tokenize input, run forward pass, and track expert activations per token."""
    # Tokenize the input
    inputs = tokenizer(batch_text, return_tensors="pt", truncation=True, max_length=32, padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    batch_size, seq_len = input_ids.shape
    print(f"Input shape: {input_ids.shape}")
    print(f"Batch texts: {batch_text}")
    
    # Forward pass
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Retain gradients for all activations
    for name, activation in activations.items():
        activation.retain_grad()
    
    # Log expert routing for last token of each batch item
    print(f"\n{'='*80}")
    print(f"EXPERT ROUTING FOR LAST TOKENS")
    print(f"{'='*80}")
    
    for batch_idx in range(batch_size):
        last_token_idx = seq_len - 1
        print(f"\nBatch {batch_idx}, Last Token (pos {last_token_idx}):")
        
        for layer_idx in sorted(expert_routing_map.keys()):
            key = (batch_idx, last_token_idx)
            if key in expert_routing_map[layer_idx]:
                expert_ids = expert_routing_map[layer_idx][key]
                print(f"  Layer {layer_idx}: Experts {sorted(expert_ids)}")
    
    print(f"{'='*80}\n")
    
    return output, input_ids, batch_size, seq_len


def extract_last_token_gradients(activations, expert_routing_map, batch_size, seq_len, layer_name=LAYER_NAME):
    """Extract gradients for last token of each batch item.
    
    Returns: Dict[batch_idx] -> Dict[expert_name] -> gradient tensor
    """
    last_token_gradients = {}
    
    # For MoE, we need to figure out which slice of the expert activation corresponds
    # to our last tokens. This is complex because experts process variable numbers of tokens.
    # For simplicity, we'll extract all gradients and annotate which experts were used.
    
    for batch_idx in range(batch_size):
        last_token_idx = seq_len - 1
        last_token_gradients[batch_idx] = {}
        
        # Get which experts were used for this last token
        key = (batch_idx, last_token_idx)
        if layer_name in expert_routing_map and key in expert_routing_map[layer_name]:
            expert_ids = expert_routing_map[layer_name][key]
            
            for expert_id in expert_ids:
                for weight_type in EXPERT_IDS:
                    act_name = f"layers.{layer_name}.experts.{expert_id}.w{weight_type}"
                    
                    if act_name in activations and activations[act_name].grad is not None:
                        # Store the full gradient (all tokens that went through this expert)
                        # Note: In MoE, multiple tokens share the same expert, so gradients are aggregated
                        grad = activations[act_name].grad
                        last_token_gradients[batch_idx][act_name] = grad.clone()
    
    return last_token_gradients


def save_gradients(last_token_gradients, filename="last_token_gradients.pt"):
    """Save last token gradients to file."""
    # Convert to saveable format
    save_data = {}
    for batch_idx, expert_grads in last_token_gradients.items():
        save_data[f"batch_{batch_idx}"] = {}
        for expert_name, grad in expert_grads.items():
            save_data[f"batch_{batch_idx}"][expert_name] = {
                "gradient": grad.cpu(),
                "shape": list(grad.shape),
                "mean": grad.mean().item(),
                "std": grad.std().item(),
            }
    
    torch.save(save_data, filename)
    print(f"\nSaved gradients to {filename}")
    
    # Also save a JSON summary
    summary = {}
    for batch_idx, expert_grads in last_token_gradients.items():
        summary[f"batch_{batch_idx}"] = {}
        for expert_name, grad in expert_grads.items():
            summary[f"batch_{batch_idx}"][expert_name] = {
                "shape": list(grad.shape),
                "mean": float(grad.mean().item()),
                "std": float(grad.std().item()),
            }
    
    with open(filename.replace(".pt", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved gradient summary to {filename.replace('.pt', '_summary.json')}")


def compute_loss(output, input_ids, top_k=50):
    """Compute cross-entropy loss against top-k word distribution."""
    # Get the last token's logits
    last_logit = output.logits[:, -1, :]  # Shape: [batch_size, vocab_size]
    
    # Get top-k predictions and their probabilities
    top_k_values, top_k_indices = torch.topk(last_logit, k=top_k, dim=-1)
    
    # Convert logits to probabilities and normalize
    top_k_probs = torch.softmax(top_k_values, dim=-1)  # Shape: [batch_size, top_k]
    
    # Create target distribution (uniform over top-k)
    target_dist = torch.ones_like(top_k_probs) / top_k  # Uniform distribution
    
    # Compute KL divergence loss (cross-entropy between distributions)
    # KL(target || pred) = sum(target * log(target/pred))
    log_probs = torch.log_softmax(top_k_values, dim=-1)
    loss = -torch.sum(target_dist * log_probs, dim=-1).mean()
    
    print(f"\nOutput logits shape: {output.logits.shape}")
    print(f"Top-{top_k} words: {top_k_indices[0, :10].tolist()}")
    print(f"Top-{top_k} probs: {top_k_probs[0, :5].tolist()}")
    print(f"Loss: {loss.item()}")
    
    return loss


def run_backward(loss, activations, expert_routing_map, batch_size, seq_len):
    """Run backward pass, extract last-token gradients, and save them."""
    loss.backward()
    
    print("\n---- Gradient Statistics (All Activations) ----")
    for name, activation in activations.items():
        if activation.grad is not None:
            print(f"{name}: grad shape {activation.grad.shape}, mean {activation.grad.mean().item():.6f}, std {activation.grad.std().item():.6f}")
    
    # Extract gradients for last token only
    print("\n---- Extracting Last Token Gradients ----")
    last_token_gradients = extract_last_token_gradients(
        activations, expert_routing_map, batch_size, seq_len
    )
    
    # Display extracted gradients
    print("\n---- Last Token Gradient Summary ----")
    for batch_idx, expert_grads in last_token_gradients.items():
        print(f"\nBatch {batch_idx}:")
        for expert_name, grad in expert_grads.items():
            print(f"  {expert_name}: shape {grad.shape}, mean {grad.mean():.6f}, std {grad.std():.6f}")
    
    # Save gradients
    save_gradients(last_token_gradients)
    
    return last_token_gradients


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Freeze all model weights
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze expert layers we want to track gradients for
    expert_layer_names = get_expert_layer_names(layer_name=LAYER_NAME)
    named_modules = dict(model.named_modules())
    for target_layer_name in expert_layer_names:
        target_layer = named_modules[target_layer_name]
        for param in target_layer.parameters():
            param.requires_grad = True
    
    handles, activations, activation_positions, expert_routing_map = setup_hooks(
        model, expert_layer_names, layer_name=LAYER_NAME
    )
    
    # Load data and run forward pass
    ds = load_wiki_dataset()
    batch_text = [ds[i]["title"] for i in range(3)]  # Process 3 samples
    output, input_ids, batch_size, seq_len = run_forward_pass(
        model, tokenizer, batch_text, activations, expert_routing_map
    )
    
    # Compute loss and backward pass
    loss = compute_loss(output, input_ids)
    last_token_gradients = run_backward(loss, activations, expert_routing_map, batch_size, seq_len)
    
    # Cleanup hooks
    for handle in handles:
        handle.remove()
