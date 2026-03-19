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
   - **From expert w1 (SVD):** We compute SVD of each model’s expert w1; project-out is then applied on the router (gate) vector. Use `run_svd_from_expert` with the same loading as project-out:
     ```bash
     # Float model (writes to svd_cache/float/)
     python -m src.experiments.router_interventions.run_svd_from_expert \
       --cache-dir svd_cache --layer_idx 0 5 10 15
     # 8bit model (writes to svd_cache/8bit/; we dequantize to get directions; saved vectors are still float)
     python -m src.experiments.router_interventions.run_svd_from_expert \
       --cache-dir svd_cache --layer_idx 0 5 10 15 --quantization 8bit
     ```
     Then use `--svd_dir` pointing at the printed path (e.g. `svd_cache/float` or `svd_cache/8bit`) with `run_project_out` (and the same `--quantization` if used).
3. **GPU** – Mixtral-8x7B in bfloat16 is ~92 GiB. Use `--quantization 8bit` (or `4bit`) to fit on a single GPU; otherwise the model loads with `device_map="auto"` and can span multiple GPUs or CPU offload.

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

# With quantization (saves memory on single GPU)
python -m src.experiments.router_interventions.run_project_out \
  --svd_dir /path/to/svd_cache \
  --layer_idx 0 \
  --quantization 8bit

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
| `--seq-len` | `32` | Sequence length. Use **32 for wiki_titles** (like gate-hook); 64–512 for wikitext. |
| `--batch-size` | `64` | Batch size. With **wiki_titles** (seq_len=32) use 64–2000; with long seq reduce. |
| `--dataset` | `wiki_titles` | `wiki_titles` (Wikipedia/Wikitext titles, seq 32), `wikitext`, or `text` (with `--text-file`). |
| `--quantization` | None | Quantization: `8bit` (~4x memory reduction) or `4bit` (~8x reduction). Recommended for single-GPU; may affect accuracy. |
| `--output-dir` | None | If set, save results to this folder with an indicative name (e.g. `project_out_L13_k1-4-16_wiki_titles_q8bit.json`). Ignores `--output_file`. |

**Output:** `results_project_out.json` (or a file in `--output-dir` with an indicative name) with:

- `config` – Run configuration.
- `baseline_loss` – Loss with original router weights. With **wiki_titles**, baseline loss is often high (e.g. 7–8) because we only score the first few tokens of each short title; use **`--dataset wikitext`** (and e.g. `--seq-len 128`) for a lower, more natural baseline.
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

- **Router: Project out** – Single run, layer 0, wiki_titles, `--quantization 8bit`. Edit `args` to set `--svd_dir` and `--layer_idx` to your paths.
- **Router: Project out (multiple k)** – Same with `--top-k "1,4,16,64,128"` and layer 13, `--quantization 8bit`.

Set `PYTHONPATH` in each config to your repo root (e.g. `/worxpace/repos/py/GaleMoE` or `"${workspaceFolder}"`). Run via Run and Debug (F5) and pick the desired config.

**Plotting:** Use `notebooks/project_out_plots.ipynb` to plot **Loss Δ vs layer** (for a chosen k) and **Loss Δ vs k** (for a chosen layer). Save runs with `--output-dir results` so the notebook can load all `project_out_*.json` from that folder.

### 4. Memory: Quantization

```bash
# wiki_titles (seq_len=32, like gate-hook): allows large batch sizes
python -m src.experiments.router_interventions.run_project_out \
  --svd_dir ./svd_cache \
  --layer_idx 15 \
  --dataset wiki_titles \
  --seq-len 32 \
  --batch-size 64

# 8-bit quantization + wiki_titles: ~4x memory reduction, batch 64
python -m src.experiments.router_interventions.run_project_out \
  --quantization 8bit \
  --dataset wiki_titles \
  --seq-len 32 \
  --batch-size 64 \
  --svd_dir ./svd_cache \
  --layer_idx 15

# 4-bit quantization + wiki_titles
python -m src.experiments.router_interventions.run_project_out \
  --quantization 4bit \
  --dataset wiki_titles \
  --seq-len 32 \
  --batch-size 64 \
  --svd_dir ./svd_cache \
  --layer_idx 15
```

**Trade-offs:**
- 8-bit: Minimal accuracy loss (~1-2% perplexity increase), fits on 24GB GPUs
- 4-bit: Moderate accuracy loss (~3-5% perplexity increase), fits on 16GB GPUs

Without `--quantization`, the model loads in bfloat16 with `device_map="auto"` (can span multiple GPUs or CPU offload).

### 5. Choosing batch size

**Trade-off:** Larger batch → fewer forward passes (faster) but more GPU memory per forward. The runner logs GPU memory and time per phase so you can find the best batch size for your GPU.

**How to find it:**

1. **Use the logs** – The runner prints:
   - `After first forward (batches loaded): total=X GiB, allocated=Y GiB, free≈Z GiB`
   - `Baseline evaluation took T seconds`
   - From that you get: memory headroom (Z) and throughput (samples / T).

2. **Sweep** – Run the same command with different `--batch-size` (e.g. 16, 32, 64, 128). Pick the **largest batch size that does not OOM** and that gives acceptable time per evaluation. Example:
   ```bash
   # Try 64 first (default for wiki_titles + 8bit)
   --batch-size 64
   # If it OOMs, try 32 or 16. If memory is free and you want speed, try 128 or 256.
   ```

3. **Rough guidance:**
   - **wiki_titles + 8bit + seq_len 32:** often 64–256 on a 24 GiB GPU; 32–64 on 16 GiB.
   - **wikitext + long seq_len:** reduce batch size (e.g. 2–8) so activations fit.

4. **Optional:** Compare throughput: `total_samples = num_batches * batch_size`, then `samples_per_sec = total_samples / baseline_eval_time`. Prefer the batch size that maximizes samples/sec without OOM.

### 6. Troubleshooting

- **CUDA out of memory**
  - Use `--quantization 8bit` (or `4bit`) to reduce memory. With quantization, use **`--seq-len 32`** and **`--dataset wiki_titles`** for lower activation memory; reduce `--batch-size` (e.g. 1) or `--num_samples` if needed.

- **"Gate weights are still on meta device"**
  - The model is loaded with `device_map="auto"`; the gate is materialized on first forward. If you still see this, ensure you are not overriding the loader (e.g. use `--quantization 8bit` so the standard loader runs).

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


