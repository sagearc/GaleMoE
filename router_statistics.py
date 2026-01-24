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
        p = f"output_100/layer={LAYER_ID:02}/router/{i:05}.safetensors"
        files.append(p)


    row_ids = []
    titles = []
    scores = []


    for path in tqdm(files):
        tensors = []
        weights = load_file(path, device="mps")
        for k, v in weights.items():
            row_id, title = k.split(".", 1) # split left most ".":
            row_ids.append(row_id)
            titles.append(title)

            score = v[-1, EXPERT_ID]
            scores.append(score)
    
    scores_tensor = torch.tensor(scores)
    top_1000 = torch.topk(scores_tensor, 1000, dim=0, sorted=True)
    top_1000_indices_np = top_1000.indices.cpu().numpy()
    top_1000_scores_np = top_1000.values.cpu().numpy()

    titles_np = np.array(titles)
    top_1000_titles = titles_np[top_1000_indices_np]

    df = pd.DataFrame({
        "title": top_1000_titles,
        "score": top_1000_scores_np,
    })

    df.to_csv(f"router_statistics_layer{LAYER_ID}_expert{EXPERT_ID}_w{W_ID}.csv", index=False)