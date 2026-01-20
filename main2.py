import os
import pickle
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

import os
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Tuple, Iterable, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


# ----------------------------
# Config
# ----------------------------

@dataclass
class MoELogConfig:
    model_id: str = "mistralai/Mixtral-8x7B-v0.1"
    num_layers: int = 32  # MoE layers to hook (Mixtral has 32)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512
    batch_size: int = 1  # keep 1 unless you also want batching logic for logs
    output_dir: str = "outputs"
    raw_jsonl_name: str = "moe_raw_token_logs.jsonl"
    agg_json_name: str = "moe_agg_expert_usage.json"
    print_debug: bool = False


# ----------------------------
# Hooking
# ----------------------------

def create_gate_hook(layer_idx: int, cfg: MoELogConfig):
    """
    Forward hook for a Mixtral gate module.
    Gate output is expected to be router logits: [B, T, E]
    """
    gate_outputs: List[Dict[str, Any]] = []

    def hook_fn(module, _inp, output):
        # output should be router logits
        # shape: (batch, seq_len, num_experts)
        if cfg.print_debug:
            print(f"[HOOK] Gate Layer {layer_idx} output shape: {tuple(output.shape)}")
            print(f"[HOOK] min={output.min().item():.4f} max={output.max().item():.4f} mean={output.mean().item():.4f}")

        gate_outputs.append(
            {
                "layer_idx": layer_idx,
                "router_logits": output.detach().cpu(),  # keep on CPU
                "timestamp": datetime.now().isoformat(),
            }
        )
        return output

    return hook_fn, gate_outputs


def register_gate_hooks(model: torch.nn.Module, cfg: MoELogConfig):
    """
    Register hooks for all MoE gate modules:
    model.layers.{i}.block_sparse_moe.gate
    """
    gate_module_names = [f"model.layers.{i}.block_sparse_moe.gate" for i in range(cfg.num_layers)]
    named_modules = dict(model.named_modules())

    hooks: List[torch.utils.hooks.RemovableHandle] = []
    all_gate_outputs: List[List[Dict[str, Any]]] = []

    for i, gate_name in enumerate(gate_module_names):
        module = named_modules.get(gate_name)
        if module is None:
            # Some models/layers might not exist if cfg.num_layers is larger than actual
            continue

        hook_fn, gate_outputs = create_gate_hook(i, cfg)
        hooks.append(module.register_forward_hook(hook_fn))
        all_gate_outputs.append(gate_outputs)

    return hooks, all_gate_outputs


# ----------------------------
# Logging + Aggregation
# ----------------------------

def compute_top2_from_router_logits(router_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    router_logits: [B, T, E]
    Returns:
      top2_idx:   [B, T, 2]
      top2_prob:  [B, T, 2]
    """
    probs = F.softmax(router_logits, dim=-1)
    top2_prob, top2_idx = torch.topk(probs, k=2, dim=-1)
    return top2_idx, top2_prob


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def append_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def default_agg():
    # agg[domain][layer_idx][expert_idx] = count
    return defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


def update_agg_counts(agg, domain: str, layer_idx: int, top2_idx: torch.Tensor):
    """
    top2_idx is [T, 2] for a single example
    increments counts for each expert appearance in top2
    """
    # flatten all top2 expert indices across tokens
    # shape [T*2]
    flat = top2_idx.reshape(-1).tolist()
    for e in flat:
        agg[domain][layer_idx][int(e)] += 1


def save_agg_json(agg, path: str):
    # make it JSON serializable
    serializable = {}
    for domain, layers in agg.items():
        serializable[domain] = {}
        for layer_idx, expert_counts in layers.items():
            serializable[domain][str(layer_idx)] = {str(e): int(c) for e, c in expert_counts.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


# ----------------------------
# Domain datasets (you replace this)
# ----------------------------

def iter_domain_examples() -> Iterable[Tuple[str, str, str]]:
    """
    Yields: (domain, example_id, text)
    Replace with your datasets (code/math/general).
    """
    examples = [
        ("general", "g0", "Hello my name is"),
        ("math", "m0", "Compute the derivative of x^2 + 3x."),
        ("code", "c0", "Write a Python function that checks if a number is prime."),
    ]
    for d, eid, txt in examples:
        yield d, eid, txt


# ----------------------------
# Main runner
# ----------------------------

def run_moe_logging(cfg: MoELogConfig):
    ensure_dir(cfg.output_dir)
    raw_path = os.path.join(cfg.output_dir, cfg.raw_jsonl_name)
    agg_path = os.path.join(cfg.output_dir, cfg.agg_json_name)

    # reset raw log file
    if os.path.exists(raw_path):
        os.remove(raw_path)

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.float16 if cfg.device.startswith("cuda") else None)
    model.eval().to(cfg.device)

    # Register gate hooks
    hooks, all_gate_outputs = register_gate_hooks(model, cfg)
    if cfg.print_debug:
        print(f"[INFO] Registered {len(hooks)} gate hooks")

    agg = default_agg()
    num_logged_examples = 0

    with torch.no_grad():
        for domain, example_id, text in iter_domain_examples():
            # Clear hook buffers (IMPORTANT: otherwise you accumulate across examples)
            for layer_buf in all_gate_outputs:
                layer_buf.clear()

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.max_length,
            ).to(cfg.device)

            input_ids = inputs["input_ids"][0].detach().cpu()  # [T]
            seq_len = input_ids.shape[0]

            # Forward pass (NOT generate) -> captures routing for the whole input sequence
            _ = model(**inputs)

            # Now process captured gate outputs
            # all_gate_outputs is list per-layer buffer, each should have 1 entry for this forward
            per_token_logs: List[Dict[str, Any]] = []

            # Build a dict layer_idx -> router_logits for the last forward
            layer_to_logits: Dict[int, torch.Tensor] = {}
            for layer_buf in all_gate_outputs:
                if not layer_buf:
                    continue
                entry = layer_buf[-1]  # last forward captured
                layer_idx = int(entry["layer_idx"])
                layer_to_logits[layer_idx] = entry["router_logits"]  # [B, T, E] on CPU

            for layer_idx, router_logits in sorted(layer_to_logits.items()):
                # [1, T, E]
                top2_idx, top2_prob = compute_top2_from_router_logits(router_logits)

                # single example
                top2_idx_ex = top2_idx[0]    # [T, 2]
                top2_prob_ex = top2_prob[0]  # [T, 2]

                # update aggregate
                update_agg_counts(agg, domain, layer_idx, top2_idx_ex)

                # raw per-token logging
                for pos in range(seq_len):
                    per_token_logs.append(
                        {
                            "domain": domain,
                            "example_id": example_id,
                            "layer_idx": layer_idx,
                            "token_pos": pos,
                            "token_id": int(input_ids[pos].item()),
                            "expert_top1": int(top2_idx_ex[pos, 0].item()),
                            "expert_top2": int(top2_idx_ex[pos, 1].item()),
                            "gate_p_top1": float(top2_prob_ex[pos, 0].item()),
                            "gate_p_top2": float(top2_prob_ex[pos, 1].item()),
                        }
                    )

            append_jsonl(raw_path, per_token_logs)
            num_logged_examples += 1

            # periodic flush of agg
            if (num_logged_examples % 10) == 0:
                save_agg_json(agg, agg_path)
                if cfg.print_debug:
                    print(f"[INFO] logged {num_logged_examples} examples")

    # final save
    save_agg_json(agg, agg_path)

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"[DONE] Raw logs saved to: {raw_path}")
    print(f"[DONE] Aggregated stats saved to: {agg_path}")
    print(f"[DONE] Total examples logged: {num_logged_examples}")


if __name__ == "__main__":
    cfg = MoELogConfig(
        model_id="mistralai/Mixtral-8x7B-v0.1",
        num_layers=2,          # set to 32 for full model; keep small for testing
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_length=128,
        output_dir="outputs",
        print_debug=False,
    )
    run_moe_logging(cfg)
    print("Done")