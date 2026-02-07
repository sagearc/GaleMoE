# GaleMoE

Tools and experiments for Mixtral-style MoE models: loading expert/router tensors, router interventions (project-out, vector interventions), and evaluation.

## Beginning

### Prerequisites

- **Python 3.12+**
- **Optional for quantization**: `bitsandbytes>=0.41.0` (for 8-bit/4-bit quantization to reduce memory usage)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install uv (if needed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Then restart your terminal, or source the env (sh, bash, zsh):
source $HOME/.local/bin/env
```

### Setup

```bash
# Clone and enter the repo
cd GaleMoE

# Install with uv
uv sync

# Or with pip
pip install -e .
```

**SVD cache:** For router intervention experiments (project-out, vector interventions), sync or generate precomputed SVD vectors if needed and pass `--svd_dir` to the run scripts.

### Quick start

- **Run router gate monitoring** (capture gate outputs over a forward pass):

  ```bash
  python main.py
  ```

- **Router intervention experiments** (project-out, vector interventions): see [Router intervention experiments](#router-intervention-experiments) below for full steps.

- **Load Mixtral router/expert metadata and tensors**: use the `BaseMoE` / `Mixtral8x7B` pattern below.

---

## Router intervention experiments

These experiments modify router (gate) weights (project out directions, zero, shuffle, etc.) and measure loss and token distributions. They require **precomputed SVD vectors** and a **GPU** (or multiple GPUs) with enough memory for the model. Run commands from the **repo root** so that `src` is on `PYTHONPATH`, or set `PYTHONPATH` to the repo root.

### What you need

1. **Python env** – Same as above (`uv sync` or `pip install -e .`).
2. **SVD cache** – A directory of pickle files, one per expert per layer, containing the top singular vectors (e.g. from your own SVD pipeline). File names must be:
   - `{model_tag}_layer{layer_idx}_expert{expert_idx}.pkl`
   - Example: `mistralai_Mixtral_8x7B_v0.1_layer0_expert0.pkl` … `expert7.pkl` for layer 0.
   - Each file can hold a single vector or a matrix (multiple vectors); the code uses `top_k` to take the first `k` vectors.
3. **GPU** – Mixtral-8x7B in bfloat16 is ~92 GiB. Default loading uses `device_map="auto"` so the model can span multiple GPUs or CPU offload. A single 44 GiB GPU is **not** enough for the full model on one device; do **not** use `--use-single-device` unless you have one very large GPU (e.g. 80 GiB+). Use `--target-layer-only-gpu` to put only the target layer on GPU and save memory.

### 1. Project-out experiment

Project out SVD (or orthogonal/random) directions from each expert's router row and compare loss to baseline, zero, and shuffle.

**Run from repo root (so `PYTHONPATH` includes the package):**

```bash
# Single top-k (default 1)
python -m src.experiments.router_interventions.run_project_out \
  --svd_dir /path/to/svd_cache \
  --layer_idx 0 \
  --output_file results_project_out.json

# Multiple top-k values (e.g. 1, 4, 16, 64, 128)
python -m src.experiments.router_interventions.run_project_out \
  --svd_dir /path/to/svd_cache \
  --layer_idx 0 \
  --top-k "1,4,16,64,128" \
  --output_file results_project_out.json

# Target layer only on GPU (saves memory on single 44 GiB GPU)
python -m src.experiments.router_interventions.run_project_out \
  --svd_dir /path/to/svd_cache \
  --layer_idx 0 \
  --target-layer-only-gpu

# Fewer samples for a quick run
python -m src.experiments.router_interventions.run_project_out \
  --svd_dir /path/to/svd_cache \
  --layer_idx 0 \
  --num_samples 10
```

**Important options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--svd_dir` | *(required)* | Directory with `{model_tag}_layer{N}_expert{i}.pkl` files. |
| `--layer_idx` | *(required)* | MoE layer index (0–31 for Mixtral). |
| `--top-k` | `1` | Comma-separated list of k values, e.g. `1,4,16,64,128`. |
| `--variations` | `svd,orthogonal,random,zero,shuffle` | Which interventions to run. |
| `--num_samples` | `200` | Number of samples for loss evaluation. |
| `--seq-len` | `512` | Sequence length. |
| `--batch-size` | `4` | Batch size (reduce if OOM). |
| `--dataset` | `wikitext` | `wikitext` or `text` (with `--text-file`). |
| `--target-layer-only-gpu` | off | Intelligent GPU memory allocation: fills GPU with as many layers as fit while **guaranteeing** the target layer is on GPU. Remaining layers offload to CPU. Optimal for single-GPU systems with limited memory (e.g., 44 GiB GPU with Mixtral loads ~12-15 layers). |
| `--use-single-device` | off | Load model on one GPU. Use only if you have ~92+ GiB VRAM; otherwise omit. |
| `--num-layers` | `32` | Total layers (for device_map building); 32 for Mixtral. |
| `--quantization` | None | Quantization: `8bit` (~4x memory reduction) or `4bit` (~8x reduction). May affect accuracy. |

**Output:** `results_project_out.json` (or `--output_file`) with:

- `config` – Run configuration.
- `baseline_loss` – Loss with original router weights.
- `results` – If a single `top_k` value: `{ "svd": { "loss", "delta" }, "orthogonal": ..., "random": ..., "zero": ..., "shuffle": ... }`.
- `by_k` – If multiple `top_k` values: `{ "1": { "svd": ..., "zero": ..., ... }, "4": { ... }, ... }`.

### 2. Vector interventions experiment

Apply project-out, inject, and subtract with SVD/orthogonal/random directions and compare loss plus token distribution metrics (KL/CE, confusion matrix).

```bash
python -m src.experiments.router_interventions.run_vector_interventions \
  --svd_dir /path/to/svd_cache \
  --layer_idx 0 \
  --output_file results_vector_interventions.json
```

Options include `--interventions`, `--variations`, `--top-k` (single value for this script), `--distribution-metric` (kl/ce), `--confusion-top-k`, `--num_samples`, `--dataset`, `--text-file`.

### 3. VS Code / launch.json

In `.vscode/launch.json` there are two configs:

- **Router: Project out** – Single run, layer 0, `num_samples=10`, output to `results_project_out.json`. Edit `args` to set `--svd_dir` (and optionally `--layer_idx`) to your paths. Uses `--target-layer-only-gpu` by default to save GPU memory.
- **Router: Project out (multiple k)** – Same but with `--top-k "1,4,16,64,128"` and layer 15. Also uses `--target-layer-only-gpu`.

Set `PYTHONPATH` in each config to your repo root (e.g. `/worxpace/repos/py/GaleMoE` or `"${workspaceFolder}"`). Run via Run and Debug (F5) and pick the desired config.

### 4. Device Mapping & Memory Strategy

#### Quantization (Recommended for <48GB GPU)

```bash
# 8-bit quantization: ~4x memory reduction (23GB for Mixtral-8x7B)
python -m src.experiments.router_interventions.run_project_out \
  --quantization 8bit \
  --svd_dir ./svd_cache \
  --layer_idx 15

# 4-bit quantization: ~8x memory reduction (12GB for Mixtral-8x7B)
python -m src.experiments.router_interventions.run_project_out \
  --quantization 4bit \
  --svd_dir ./svd_cache \
  --layer_idx 15
```

**Trade-offs:**
- 8-bit: Minimal accuracy loss (~1-2% perplexity increase), fits on 24GB GPUs
- 4-bit: Moderate accuracy loss (~3-5% perplexity increase), fits on 16GB GPUs

#### Intelligent Device Mapping (No Quantization)

- **`--target-layer-only-gpu` (Recommended for 44GB+ GPU):** Automatically calculates how many layers fit on your GPU and loads them, while **guaranteeing** the target intervention layer is on GPU (priority placement). Uses "sequential" fill: loads embeddings, norm, lm_head, target layer, then fills from layer 0, 1, 2... until GPU is full.
  - Example: On a 44GB GPU with Mixtral-8x7B (~2.8GB/layer), it may fit ~12-15 layers on GPU (including the target layer), with the rest on CPU.
  - This balances speed (target layer on GPU) with memory efficiency (offloads non-critical layers to CPU).

- **Default (`device_map="auto"`):** HuggingFace's automatic device mapping. May spread layers across multiple GPUs or offload more to CPU.

- **`--use-single-device`:** Forces entire model to a single GPU (no device map). Only works if your GPU has enough memory (~92GB for Mixtral-8x7B).

### 5. Troubleshooting

- **CUDA out of memory**
  - **Option 1 (Best):** Use `--quantization 8bit` to reduce memory by ~4x (23GB for Mixtral-8x7B)
  - **Option 2:** Use `--target-layer-only-gpu` to intelligently fill GPU with layers
  - **Option 3:** Reduce `--batch-size` (try 2 or 1), `--seq-len` (try 256), or `--num_samples`
  - Do **not** use `--use-single-device` on a single 44 GiB GPU; the full model needs ~92 GiB in bfloat16.

- **"Gate weights are still on meta device"**
  - The code defers reading the gate until after the first forward; if you still see this, try:
    - Running with `--target-layer-only-gpu` (recommended for single GPU, ensures target layer is materialized).
    - Running without `--use-single-device` (default).
    - If you have one very large GPU, try `--use-single-device` so the model loads without lazy (meta) tensors.

- **Logs:** The runner logs GPU memory before/after model load and after the first forward. Use these to see how much memory is in use.

---

## Mixtral8x7B example

### Define the mandatory parameters for the Mixtral8x7B model


```python
from pprint import pprint
from model_loader import BaseMoE


class Mixtral8x7B(BaseMoE):
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    n_experts = 8
    n_layers = 32

    expert_tensor_name_template = "model.layers.{layer}.block_sparse_moe.experts.{expert}.w1.weight"
    router_tensor_name_template = "model.layers.{layer}.block_sparse_moe.gate.weight"


model = Mixtral8x7B()
```

## Query router/expert tensors metadata from the Hugging Face Hub

### Expert tensors metadata

Each expert tensor is described by an instance of `TensorMetadata`, which includes methods to download and load the tensor.


```python
experts_metadata = model.get_experts_metdata(layer=10)

print("Expert tensors metadata for layer 10:")
pprint(experts_metadata)
```

    Expert tensors metadata for layer 10:
    [TensorMetadata(model_id='mistralai/Mixtral-8x7B-v0.1',
                    tensor_name='model.layers.10.block_sparse_moe.experts.0.w1.weight',
                    hf_filename='model-00006-of-00019.safetensors',
                    local_path=None),
    .
    .
    .
     TensorMetadata(model_id='mistralai/Mixtral-8x7B-v0.1',
                    tensor_name='model.layers.10.block_sparse_moe.experts.7.w1.weight',
                    hf_filename='model-00007-of-00019.safetensors',
                    local_path=None)]


### Router tensor metadata

```python
router_metadata = model.get_router_metadata(layer=10)

print("Router tensor metadata for layer 10:")
pprint(router_metadata)
```

    Router tensor metadata for layer 10:
    TensorMetadata(model_id='mistralai/Mixtral-8x7B-v0.1',
                   tensor_name='model.layers.10.block_sparse_moe.gate.weight',
                   hf_filename='model-00006-of-00019.safetensors',
                   local_path=None)


## Download the file containing the router tensor for layer 10


```python
router_metadata.download_file()
```




    '/Users/sagi/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0/model-00006-of-00019.safetensors'



Note that the `local_path` field will be populated after the first download.
Files are cached locally to avoid redundant downloads.


```python
pprint(router_metadata)
```

    TensorMetadata(model_id='mistralai/Mixtral-8x7B-v0.1',
                   tensor_name='model.layers.10.block_sparse_moe.gate.weight',
                   hf_filename='model-00006-of-00019.safetensors',
                   local_path='/Users/sagi/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0/model-00006-of-00019.safetensors')


## Load the router tensor into memory


```python
tensor = router_metadata.load()

print("Loaded router tensor shape:", tensor.shape)  # 8 experts

tensor
```

    Loaded router tensor shape: torch.Size([8, 4096])

    tensor([[-7.1049e-05,  4.4861e-03,  9.7656e-04,  ..., -1.1169e-02,
              6.5308e-03,  6.1989e-05],
            [-5.4626e-03,  6.6280e-05,  1.3199e-03,  ...,  6.8054e-03,
             -7.8735e-03, -2.8687e-03],
            [ 4.7302e-03,  1.1902e-03,  2.2888e-03,  ..., -9.8877e-03,
              6.2561e-03,  6.6528e-03],
            ...,
            [-2.2736e-03, -2.7008e-03,  1.7242e-03,  ...,  1.5137e-02,
             -4.0588e-03, -1.4114e-03],
            [ 8.9722e-03, -3.7689e-03, -4.9744e-03,  ..., -2.4872e-03,
             -9.4604e-03,  6.5918e-03],
            [ 7.3624e-04,  5.6076e-04,  1.1215e-03,  ...,  1.6174e-03,
              9.1553e-03, -5.2490e-03]], dtype=torch.bfloat16)


