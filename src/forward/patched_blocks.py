# ruff: noqa
"""TransformerLens hook functions for capturing Mixtral MoE intermediate activations."""
from typing import NamedTuple

import torch
import torch.nn.functional as F


class CacheKey(NamedTuple):
    layer_idx: int
    expert_id: int
    w_id: int
    prompt: int
    row_id: int


def make_routing_hook(layer_idx: int, num_experts: int, routing_state: dict):
    """Captures expert routing masks from hook_expert_indices."""
    def hook_fn(selected_experts, hook):
        expert_mask = F.one_hot(selected_experts.long(), num_classes=num_experts).permute(2, 1, 0)
        routing_state[layer_idx] = expert_mask
        return selected_experts
    return hook_fn


def make_expert_activation_hook(
    layer_idx: int,
    expert_id: int,
    w_id: int,
    cache: dict,
    routing_state: dict,
    batch_info: dict,
    row_idx_to_prompt: list,
):
    """Captures expert intermediate activations for last tokens only.

    w_id=1 → hook_gate (W1x, gate projection)
    w_id=3 → hook_pre  (W3x, up projection)
    """
    def hook_fn(activation, hook):
        expert_mask = routing_state[layer_idx]
        seq_length = batch_info["seq_length"]

        idx, top_x = torch.where(expert_mask[expert_id])

        is_last = ((top_x + 1) % seq_length) == 0
        is_last_indices = torch.nonzero(is_last).squeeze(-1)

        for i in is_last_indices:
            prompt_idx = (top_x[i] // seq_length).item()
            row_id, prompt = row_idx_to_prompt[prompt_idx]
            key = CacheKey(layer_idx, expert_id, w_id, prompt, row_id)
            cache[key] = activation[i].cpu()

        return activation
    return hook_fn


def make_gate_logits_hook(layer_idx: int, router_logits_store: dict):
    """PyTorch forward hook on W_gate to capture pre-softmax router logits."""
    def hook_fn(module, input, output):
        router_logits_store[layer_idx] = output.detach()
    return hook_fn


def build_fwd_hooks(
    layers_to_patch: list[int],
    num_experts: int,
    cache: dict,
    routing_state: dict,
    batch_info: dict,
    row_idx_to_prompt: list,
) -> list[tuple]:
    """Build TransformerLens (hook_name, hook_fn) pairs for run_with_hooks."""
    hooks = []
    for layer_idx in layers_to_patch:
        hooks.append((
            f"blocks.{layer_idx}.mlp.hook_expert_indices",
            make_routing_hook(layer_idx, num_experts, routing_state),
        ))
        for expert in range(num_experts):
            hooks.append((
                f"blocks.{layer_idx}.mlp.experts.{expert}.hook_gate",
                make_expert_activation_hook(layer_idx, expert, 1, cache, routing_state, batch_info, row_idx_to_prompt),
            ))
            hooks.append((
                f"blocks.{layer_idx}.mlp.experts.{expert}.hook_pre",
                make_expert_activation_hook(layer_idx, expert, 3, cache, routing_state, batch_info, row_idx_to_prompt),
            ))
    return hooks
