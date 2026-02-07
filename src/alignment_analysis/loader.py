"""Weight loading from HuggingFace Hub for MoE models."""
from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import Tensor

from src.alignment_analysis.base import LayerWeights
from src.models.model_loader import BaseMoE, TensorMetadata


class ShardCache:
    """In-memory cache to avoid repeated calls to download_file() for the same file.
    
    download_file() already checks Hugging Face cache, but this avoids repeated
    cache lookups within the same Python session.
    """
    
    def __init__(self) -> None:
        self._local_paths: Dict[tuple[str, str], str] = {}

    def ensure_downloaded(self, meta: TensorMetadata) -> str:
        """Ensure file is downloaded. Uses in-memory cache to avoid repeated calls."""
        key = (meta.model_id, meta.hf_filename)
        if key not in self._local_paths:
            local_path = meta.download_file()
            self._local_paths[key] = local_path
        meta.local_path = self._local_paths[key]
        return meta.local_path


class MoEWeightsRepository:
    """Repository for loading MoE weights from Hugging Face Hub."""
    
    def __init__(self, moe: BaseMoE, cache: Optional[ShardCache] = None) -> None:
        self.moe = moe
        self.cache = cache or ShardCache()

    def _load_many(self, metas: List[TensorMetadata]) -> Dict[str, Tensor]:
        """Load multiple tensors that may live across multiple shard files."""
        by_file = defaultdict(list)
        for m in metas:
            by_file[m.hf_filename].append(m)

        loaded: Dict[str, Tensor] = {}
        for hf_filename, metas_in_file in by_file.items():
            local_path = self.cache.ensure_downloaded(metas_in_file[0])
            for m in metas_in_file:
                m.local_path = local_path

            with safe_open(local_path, framework="pt") as f:
                for m in metas_in_file:
                    loaded[m.tensor_name] = f.get_tensor(m.tensor_name)

        return loaded

    def prefetch_layer(self, layer: int, max_workers: int = 16) -> None:
        """Prefetch (download) all files needed for a layer using optimized parallel downloads.
        
        Args:
            layer: Layer number to prefetch
            max_workers: Maximum number of parallel download workers (default: 16)
        """
        # Enable hf_transfer for faster downloads if available
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
        
        # Check hf-transfer availability
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

        all_metas = [router_meta] + expert_metas
        unique_shards = set(meta.hf_filename for meta in all_metas)
        
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
        
        print(f"Downloading {len(needed_files)} file(s) in parallel...")
        print(f"  Using {max_workers} parallel workers")
        
        def download_file(filename: str) -> tuple[str, str]:
            path = hf_hub_download(
                repo_id=self.moe.model_id,
                filename=filename,
                local_files_only=False,
            )
            return (filename, path)
        
        downloaded = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(download_file, filename): filename 
                for filename in needed_files
            }
            
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

    def load_layer(self, layer: int) -> LayerWeights:
        """Load weights for a specific layer."""
        router_meta = self.moe.get_router_metadata(layer)
        expert_metas = self.moe.get_experts_metadata(layer)
        metas = [router_meta] + expert_metas

        loaded = self._load_many(metas)

        gate_w = loaded[router_meta.tensor_name]
        experts_w_in = [loaded[m.tensor_name] for m in expert_metas]

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


def check_download_config() -> dict:
    """Check the current download configuration and diagnose potential issues."""
    diagnostics = {
        "hf_transfer_enabled": False,
        "hf_transfer_installed": False,
        "environment_vars": {},
        "recommendations": [],
    }
    
    hf_transfer_env = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "")
    diagnostics["hf_transfer_enabled"] = hf_transfer_env.lower() in ("1", "true", "yes")
    diagnostics["environment_vars"]["HF_HUB_ENABLE_HF_TRANSFER"] = hf_transfer_env or "not set"
    
    try:
        import hf_transfer
        diagnostics["hf_transfer_installed"] = True
        diagnostics["hf_transfer_version"] = getattr(hf_transfer, "__version__", "unknown")
    except ImportError:
        diagnostics["hf_transfer_installed"] = False
        diagnostics["recommendations"].append(
            "Install hf-transfer for 10-100x faster downloads: pip install hf-transfer"
        )
    
    if diagnostics["hf_transfer_enabled"] and not diagnostics["hf_transfer_installed"]:
        diagnostics["recommendations"].append(
            "HF_HUB_ENABLE_HF_TRANSFER is set but hf-transfer is not installed."
        )
    
    if diagnostics["hf_transfer_installed"] and not diagnostics["hf_transfer_enabled"]:
        diagnostics["recommendations"].append(
            "hf-transfer is installed but not enabled. Set: export HF_HUB_ENABLE_HF_TRANSFER=1"
        )
    
    diagnostics["environment_vars"]["HF_HUB_DOWNLOAD_TIMEOUT"] = os.environ.get(
        "HF_HUB_DOWNLOAD_TIMEOUT", "not set"
    )
    
    return diagnostics


def print_download_diagnostics() -> None:
    """Print diagnostic information about download configuration."""
    print("=" * 60)
    print("Download Configuration Diagnostics")
    print("=" * 60)
    
    diag = check_download_config()
    
    print(f"\n✓ hf-transfer installed: {diag['hf_transfer_installed']}")
    if diag.get("hf_transfer_version"):
        print(f"  Version: {diag['hf_transfer_version']}")
    
    print(f"\n✓ hf-transfer enabled: {diag['hf_transfer_enabled']}")
    
    print("\nEnvironment Variables:")
    for key, value in diag["environment_vars"].items():
        print(f"  {key} = {value}")
    
    if diag["recommendations"]:
        print("\n⚠️  Recommendations:")
        for i, rec in enumerate(diag["recommendations"], 1):
            print(f"  {i}. {rec}")
    else:
        print("\n✓ Configuration looks good!")
    
    print("=" * 60)
