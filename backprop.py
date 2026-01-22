import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from torch.utils.hooks import RemovableHandle
from datasets import load_dataset
from typing import NamedTuple
import torch


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

# Hook to capture gate routing decisions
def get_gate_hook(layer_idx, last_token_expert_usage):
    def hook(module, input, output):
        # output is tuple: (final_hidden_states, router_logits)
        # router_logits shape: [batch_size, sequence_length, num_experts]
        if len(output) > 1 and output[1] is not None:
            router_logits = output[1]
            # Get the top-k experts for the last token
            last_token_logits = router_logits[:, -1, :]  # [batch, num_experts]
            top_k = torch.topk(last_token_logits, k=2, dim=-1)
            expert_indices = top_k.indices[0].tolist()  # Convert to list of expert IDs
            last_token_expert_usage[layer_idx] = set(expert_indices)
    return hook


# Define the hook function for expert weights
def get_w_activation_hook(name, activations):
    def hook(module, input, output: torch.Tensor):
        # Store activation without retaining grad yet
        # We'll selectively retain grad after we know which experts were used
        activations[name] = output
    return hook


def setup_hooks(model, expert_layer_names, layer_name: int = LAYER_NAME):
    """Register gate and expert weight hooks."""
    activations = {}
    last_token_expert_usage = {}
    handles = []
    
    named_modules = dict(model.named_modules())
    
    # Register gate hooks to track expert usage
    gate_layer = named_modules[f"model.layers.{layer_name}.block_sparse_moe"]
    gate_hook = get_gate_hook(layer_name, last_token_expert_usage)
    gate_handle = gate_layer.register_forward_hook(gate_hook)
    handles.append(gate_handle)
    
    # Register expert weight hooks
    for target_layer_name in expert_layer_names:
        target_layer = named_modules[target_layer_name]
        clean_layer_name = target_layer_name.removeprefix("model.").replace("block_sparse_moe.", "")
        hook = get_w_activation_hook(clean_layer_name, activations)
        
        print(f"Registering hook on layer: {clean_layer_name}")
        
        handle = target_layer.register_forward_hook(hook)
        handles.append(handle)
    
    print(f"Registered hooks on {len(handles)} layers (gates + experts).")
    
    return handles, activations, last_token_expert_usage


def load_sample_text():
    """Load WikiText dataset and return a sample."""
    print("\nLoading WikiText dataset...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en")
    
    # Find a non-empty sample
    sample_text = None
    for i in range(len(dataset)):
        if len(dataset[i]["text"].strip()) > 100:
            sample_text = dataset[i]["text"]
            break
    
    print(f"Sample text: {sample_text[:200]}...")
    return sample_text


def run_forward_pass(model, tokenizer, sample_text, activations, last_token_expert_usage):
    """Tokenize input, run forward pass, and retain gradients for relevant experts."""
    # Tokenize the input
    inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass with real inputs
    output = model(input_ids=input_ids, attention_mask=attention_mask, output_router_logits=True)
    
    # After forward pass, retain gradients for all expert activations
    print(f"\n--- Experts Used for Last Token ---")
    for layer_idx, expert_ids in last_token_expert_usage.items():
        print(f"Layer {layer_idx}: Experts {expert_ids}")
    
    # Retain gradients for all activations (all 14336 neurons per expert)
    print(f"\n--- Retaining Gradients for All Expert Activations ---")
    for name, activation in activations.items():
        activation.retain_grad()
        print(f"Retaining grad for {name}, shape: {activation.shape}")
    
    return output, input_ids


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


def run_backward(loss, activations):
    """Run backward pass and display results."""
    loss.backward()
    
    print("\n---- Activations Captured ----")
    for name, activation in activations.items():
        print(f"Layer: {name}, Activation shape: {activation.shape}")
    print(f"Total activations: {len(activations)}")
    
    # Print gradient statistics for each neuron
    print("\n---- Gradient Statistics ----")
    for name, activation in activations.items():
        if activation.grad is not None:
            print(f"{name}: grad shape {activation.grad.shape}, mean {activation.grad.mean().item():.6f}, std {activation.grad.std().item():.6f}")


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    
    # Freeze all model weights
    for param in model.parameters():
        param.requires_grad = False
    
    expert_layer_names = get_expert_layer_names(layer_name=LAYER_NAME)
    handles, activations, last_token_expert_usage = setup_hooks(
        model, expert_layer_names, layer_name=LAYER_NAME
    )
    
    # Load data and run forward pass
    sample_text = load_sample_text()
    output, input_ids = run_forward_pass(model, tokenizer, sample_text, activations, last_token_expert_usage)
    
    # Compute loss and backward pass
    loss = compute_loss(output, input_ids)
    run_backward(loss, activations)
    
    # Cleanup hooks
    for handle in handles:
        handle.remove()

