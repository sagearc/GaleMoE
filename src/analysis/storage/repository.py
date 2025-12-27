"""Repository for loading MoE weights from Hugging Face."""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional

from safetensors import safe_open
from torch import Tensor

from src.analysis.core.data_structures import LayerWeights
from src.models.model_loader import BaseMoE, TensorMetadata


class ShardCache:
    """In-memory cache to avoid repeated calls to download_file() for the same file.
    
    download_file() already checks Hugging Face cache, but this avoids repeated
    cache lookups within the same Python session.
    """
    
    def __init__(self) -> None:
        """Initialize an empty shard cache."""
        self._local_paths: Dict[tuple[str, str], str] = {}

    def ensure_downloaded(self, meta: TensorMetadata) -> str:
        """Ensure file is downloaded. Uses in-memory cache to avoid repeated calls.
        
        download_file() handles Hugging Face cache checking automatically.
        
        Args:
            meta: TensorMetadata object for the file to download
            
        Returns:
            Local path to the downloaded file
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
    
    def __init__(self, moe: BaseMoE, cache: Optional[ShardCache] = None) -> None:
        """Initialize the repository.
        
        Args:
            moe: BaseMoE model instance
            cache: Optional ShardCache instance (creates new one if not provided)
        """
        self.moe = moe
        self.cache = cache or ShardCache()

    def _load_many(self, metas: List[TensorMetadata]) -> Dict[str, Tensor]:
        """Load multiple tensors that may live across multiple shard files.
        
        Args:
            metas: List of TensorMetadata objects to load
            
        Returns:
            Dictionary mapping tensor names to loaded tensors
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

    def prefetch_layer(self, layer: int, max_workers: int = 16) -> None:
        """
        Prefetch (download) all files needed for a layer using optimized snapshot_download.
        
        Uses optimizations for faster downloads:
        - Higher max_workers for parallel downloads (default 16)
        - Enables hf_transfer if available (via HF_HUB_ENABLE_HF_TRANSFER env var)
        - Downloads automatically resume if interrupted (default behavior)
        
        Args:
            layer: Layer number to prefetch
            max_workers: Maximum number of parallel download workers (default: 16 for speed)
        """
        import os
        from huggingface_hub import hf_hub_download
        
        # Enable hf_transfer for faster downloads (if available)
        # This can provide 10-100x speedup for large files
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        
        # Additional optimizations for faster downloads
        # Increase timeout for large files
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")  # 10 minutes
        
        # Check if hf-transfer is actually available and print status
        hf_transfer_available = False
        hf_transfer_version = None
        try:
            import hf_transfer  # type: ignore[import-untyped]
            hf_transfer_available = True
            hf_transfer_version = getattr(hf_transfer, "__version__", "unknown")
        except ImportError:
            pass
        
        if hf_transfer_available:
            print(f"✓ Using hf-transfer for fast downloads (version: {hf_transfer_version})")
        else:
            print("⚠️  hf-transfer not installed. Install with: pip install hf-transfer")
            print("   Downloads will be slower without it.")
        
        router_meta = self.moe.get_router_metadata(layer)
        expert_metas = self.moe.get_experts_metadata(layer)

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
            except Exception:  # noqa: BLE001 - Expected to catch all exceptions here
                # File not in cache, will download below
                pass
            needed_files.append(shard_filename)
        
        if not needed_files:
            print("✓ All files already cached")
            return
        
        # Use parallel hf_hub_download instead of snapshot_download for better parallelism
        # snapshot_download with allow_patterns doesn't parallelize as well
        print(f"Downloading {len(needed_files)} file(s) in parallel...")
        print(f"  Using {max_workers} parallel workers (hf_transfer enabled if available)")
        
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def download_file(filename: str) -> tuple[str, str]:
                """Download a single file and return (filename, path)."""
                path = hf_hub_download(
                    repo_id=self.moe.model_id,
                    filename=filename,
                    local_files_only=False,
                )
                return (filename, path)
            
            # Download files in parallel
            downloaded = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all download tasks
                future_to_file = {
                    executor.submit(download_file, filename): filename 
                    for filename in needed_files
                }
                
                # Process completed downloads
                for future in as_completed(future_to_file):
                    filename = future_to_file[future]
                    try:
                        file_path = future.result()[1]
                        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        downloaded[filename] = file_path
                        print(f"✓ Downloaded: {filename} ({file_size_mb:.1f} MB)")
                    except Exception as e:
                        print(f"✗ Failed to download {filename}: {e}")
                        raise
            
            print(f"✓ Successfully downloaded {len(downloaded)} file(s)")
        except Exception as e:
            print(f"✗ Failed to download files: {e}")
            raise

    def load_layer(self, layer: int) -> LayerWeights:
        """Load weights for a specific layer.
        
        Args:
            layer: Layer number to load
            
        Returns:
            LayerWeights object containing router and expert weights
            
        Raises:
            ValueError: If weight shapes don't match expected dimensions
        """
        router_meta = self.moe.get_router_metadata(layer)
        expert_metas = self.moe.get_experts_metadata(layer)
        metas = [router_meta] + expert_metas

        loaded = self._load_many(metas)

        gate_w = loaded[router_meta.tensor_name]
        experts_w_in = [loaded[m.tensor_name] for m in expert_metas]

        # Validate weight shapes
        if gate_w.ndim != 2 or gate_w.shape[0] != self.moe.n_experts:
            raise ValueError(
                f"Unexpected gate weight shape {tuple(gate_w.shape)} "
                f"(expected [n_experts={self.moe.n_experts}, d_model])"
            )
        d_model = gate_w.shape[1]
        for i, w in enumerate(experts_w_in):
            if w.ndim != 2 or w.shape[1] != d_model:
                raise ValueError(
                    f"Expert {i} unexpected shape {tuple(w.shape)} "
                    f"(expected [d_ff, d_model={d_model}])"
                )

        return LayerWeights(
            model_id=self.moe.model_id,
            layer=layer,
            gate_w=gate_w,
            experts_w_in=experts_w_in,
        )

