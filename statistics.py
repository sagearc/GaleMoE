from safetensors.torch import load_file
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm


EXPERT_ID = 2
W_ID = 1
LAYER_ID = 4


if __name__ == "__main__":
    files = []
    for i in range(2):
        p = f"output_100/layer={LAYER_ID:02}/expert={EXPERT_ID}/w={W_ID}/{i:05}.safetensors"
        files.append(p)


    row_ids = []
    titles = []
    activations = torch.empty((14336, 0)).to("mps")


    for path in tqdm(files):
        tensors = []
        weights = load_file(path, device="mps")
        for k, v in weights.items():
            row_id, title = k.split(".", 1) # split left most ".":
            row_ids.append(row_id)
            titles.append(title)
            tensors.append(v.unsqueeze(1))  # (14336, 1)
        activations = torch.hstack([activations] + tensors)
    
    # for each row in activtions (14336), get the indices of the top 50 values
    top50 = torch.topk(activations, 50, dim=1)

    neurons_to_top50_titles = {}
    for neuron_id in range(activations.shape[0]):
        neuron_top_50_titles = [titles[i] for i in top50.indices[neuron_id]]
        neurons_to_top50_titles[neuron_id] = neuron_top_50_titles
    
    print()
