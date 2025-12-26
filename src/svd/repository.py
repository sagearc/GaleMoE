"""Repository for loading MoE weights from Hugging Face."""
import os
from collections import defaultdict
from typing import Dict, List, Optional

from safetensors import safe_open
from torch import Tensor

from src.svd.data_structures import LayerWeights


class ShardCache:
    """
    In-memory cache to avoid repeated calls to download_file() for the same file.
    download_file() already checks Hugging Face cache, but this avoids repeated
    cache lookups within the same Python session.
    """
    def __init__(self):
        self._local_paths: Dict[tuple[str, str], str] = {}

    def ensure_downloaded(self, meta) -> str:
        """
        Ensure file is downloaded. Uses in-memory cache to avoid repeated calls.
        download_file() handles Hugging Face cache checking automatically.
        """
        key = (meta.model_id, meta.hf_filename)
        if key not in self._local_paths:
            # download_file() will check Hugging Face cache first, then download if needed
            local_path = meta.download_file()
            self._local_paths[key] = local_path
        meta.local_path = self._local_paths[key]
        return meta.local_path


class MoEWeightsRepository:
    """Repository for loading MoE weights from Hugging Face Hub."""
    
    def __init__(self, moe, cache: Optional[ShardCache] = None):
        self.moe = moe
        self.cache = cache or ShardCache()

    def _load_many(self, metas: List) -> Dict[str, Tensor]:
        """
        Load a bunch of tensors that may live across multiple shards.
        Returns: {tensor_name: tensor}
        """
        by_file = defaultdict(list)
        for m in metas:
            by_file[m.hf_filename].append(m)

        loaded: Dict[str, Tensor] = {}
        for hf_filename, metas_in_file in by_file.items():
            # download once
            local_path = self.cache.ensure_downloaded(metas_in_file[0])
            for m in metas_in_file:
                m.local_path = local_path

            with safe_open(local_path, framework="pt") as f:
                for m in metas_in_file:
                    loaded[m.tensor_name] = f.get_tensor(m.tensor_name)

        return loaded

    def prefetch_layer(self, layer: int, max_workers: int = 4, max_retries: int = 3) -> None:
        """
        Prefetch (download) all files needed for a layer using snapshot_download.
        snapshot_download is faster and more efficient for downloading multiple files.
        """
        from huggingface_hub import hf_hub_download, snapshot_download
        
        router_meta = self.moe.get_router_metadata(layer)
        expert_metas = self.moe.get_experts_metdata(layer)

        # Get unique shard filenames needed for THIS layer
        all_metas = [router_meta] + expert_metas
        unique_shards = set()
        for meta in all_metas:
            unique_shards.add(meta.hf_filename)
        
        # Check which files are already cached
        needed_files = []
        for shard_filename in sorted(unique_shards):
            try:
                cached_path = hf_hub_download(
                    repo_id=self.moe.model_id,
                    filename=shard_filename,
                    local_files_only=True
                )
                if os.path.exists(cached_path):
                    file_size_mb = os.path.getsize(cached_path) / (1024 * 1024)
                    print(f"✓ Already cached: {shard_filename} ({file_size_mb:.1f} MB)")
                    continue
            except Exception:
                pass
            needed_files.append(shard_filename)
        
        if not needed_files:
            print("✓ All files already cached")
            return
        
        # Use snapshot_download for faster parallel downloads
        print(f"Downloading {len(needed_files)} file(s) using snapshot_download (faster)...")
        try:
            snapshot_dir = snapshot_download(
                repo_id=self.moe.model_id,
                allow_patterns=needed_files,  # Only download needed files
                local_files_only=False,
                max_workers=max_workers,  # Parallel downloads for speed
            )
            print(f"✓ Successfully downloaded files to: {snapshot_dir}")
        except Exception as e:
            print(f"✗ Failed to download files: {e}")
            raise

    def load_layer(self, layer: int) -> LayerWeights:
        """Load weights for a specific layer."""
        router_meta = self.moe.get_router_metadata(layer)
        expert_metas = self.moe.get_experts_metdata(layer)
        metas = [router_meta] + expert_metas

        loaded = self._load_many(metas)

        gate_w = loaded[router_meta.tensor_name]
        experts_w_in = [loaded[m.tensor_name] for m in expert_metas]

        # basic sanity checks
        if gate_w.ndim != 2 or gate_w.shape[0] != self.moe.n_experts:
            raise ValueError(f"Unexpected gate weight shape {tuple(gate_w.shape)} (expected [n_experts, d_model])")
        d_model = gate_w.shape[1]
        for i, w in enumerate(experts_w_in):
            if w.ndim != 2 or w.shape[1] != d_model:
                raise ValueError(f"Expert {i} unexpected shape {tuple(w.shape)} (expected [d_ff, d_model])")

        return LayerWeights(
            model_id=self.moe.model_id,
            layer=layer,
            gate_w=gate_w,
            experts_w_in=experts_w_in,
        )

