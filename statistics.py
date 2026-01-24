from safetensors.torch import load_file
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm
import numpy as np

EXPERT_ID = 7
W_ID = 1
LAYER_ID = 16


if __name__ == "__main__":
    files = []
    for i in range(100):
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
    
    # for each row in activtions (14336), get the indices of the top 20 values
    top20 = torch.topk(activations, 20, dim=1, sorted=True)
    top_20_indices_np = top20.indices.cpu().numpy()
    titles_np = np.array(titles)
    all_neurons_top_20_titles = titles_np[top_20_indices_np] # (14336, 20) array of titles


    # create np array mapping neuron_id to top20 titles
    df = pd.DataFrame(all_neurons_top_20_titles, columns=[f"top{i+1}" for i in range(20)])
    df.to_csv(f"neuron_top20_titles_layer{LAYER_ID}_expert{EXPERT_ID}_w{W_ID}.csv", index_label="neuron_id")
    print()
