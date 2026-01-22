import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from torch.utils.hooks import RemovableHandle


NUM_LAYERS = 32
NUM_EXPERTS = 8
EXPERT_IDS = (1, 3)
LAYER_NAME = 20

model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", dtype=torch.bfloat16)
tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

# 1. Freeze all model weights
for param in model.parameters():
    param.requires_grad = False

if LAYER_NAME is not None:
    expert_layer_names = [f"model.layers.{LAYER_NAME}.block_sparse_moe.experts.{j}.w{k}" for j in range(NUM_EXPERTS) for k in EXPERT_IDS]
else:
    expert_layer_names = [f"model.layers.{i}.block_sparse_moe.experts.{j}.w{k}" for i in range(NUM_LAYERS) for j in range(NUM_EXPERTS) for k in EXPERT_IDS]

expert_layer_names.sort()

# Dictionary to store the activation tensors (where gradients will appear)
activations = {}

# 2. Define the hook function
def get_w_activation_hook(name):
    def hook(module, input, output: torch.Tensor):
        # 'output' is the result of self.w3(hidden_states)
        # We must call retain_grad() because it is an intermediate tensor
        output.retain_grad()
        activations[name] = output
    return hook

named_modules = dict(model.named_modules())
handles: list[RemovableHandle] = []
for target_layer_name in expert_layer_names:
    target_layer = named_modules[target_layer_name]

    clean_layer_name = target_layer_name.removeprefix("model.").replace("block_sparse_moe.", "")
    hook = get_w_activation_hook(clean_layer_name)

    print(f"Registering hook on layer: {clean_layer_name}")

    handle = target_layer.register_forward_hook(hook)
    handles.append(handle)

print(f"Registered hooks on {len(handles)} expert layers.")


# --- Running the Pass ---

# Ensure inputs require grad so the graph is built, 
# even though weights are frozen.
# (See "Crucial Warning" below if this part confuses you)
dummy_input = torch.randn(1, 3, 4096, requires_grad=True, dtype=torch.bfloat16).to(model.device)

# Forward pass
output = model(inputs_embeds=dummy_input) # specific call depends on your model signature
loss = output.logits.mean() # simple dummy loss

print("output logits shape:", output.logits.shape)
print("loss:", loss.item())

# Backward pass
loss.backward()

print("---- Activations Captured ----")
for name, activation in activations.items():
    print(f"Layer: {name}, Activation shape: {activation.shape}")
print("len:", len(activations))
