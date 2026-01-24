# ruff: noqa
"""Monkey-patched forward methods for Mixtral sparse MoE blocks."""
from typing import NamedTuple

import torch
import torch.nn.functional as F
from transformers.models.mixtral.modeling_mixtral import (
    MixtralBlockSparseTop2MLP,
    MixtralSparseMoeBlock,
)


class CacheKey(NamedTuple):
    layer_idx: int
    expert_id: int
    w_id: int
    prompt: int
    row_id: int


# Monkey-patched MixtralBlockSparseTop2MLP.forward method
def patched_block_sparse_top2_mlp_forward(self: MixtralBlockSparseTop2MLP, hidden_states: torch.Tensor, top_x: torch.Tensor, sequence_length: int):
    """ """
    W1x: torch.Tensor = self.w1(hidden_states)
    W3x: torch.Tensor = self.w3(hidden_states)

    is_last = ((top_x + 1) % sequence_length) == 0
    is_last_indices = torch.nonzero(is_last).squeeze(-1)

    # save to cache
    for i in is_last_indices:
        prompt_idx = (top_x[i] // sequence_length).item()
        row_id_in_dataset, prompt = self.patch_row_idx_to_prompt[prompt_idx]

        key1 = CacheKey(
            layer_idx=self.patch_layer_idx,
            expert_id=self.patch_expert_id,
            w_id=1,
            prompt=prompt,
            row_id=row_id_in_dataset,
        )
        self.patch_cache[key1] = W1x[i].cpu()

        key3 = CacheKey(
            layer_idx=self.patch_layer_idx,
            expert_id=self.patch_expert_id,
            w_id=3,
            prompt=prompt,
            row_id=row_id_in_dataset,
        )
        self.patch_cache[key3] = W3x[i].cpu()

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
