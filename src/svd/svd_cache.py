"""SVD cache for persisting decompositions across runs."""
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

from torch import Tensor


class SVDCache:
    """
    Cache for SVD decompositions across the entire network.
    Persists to disk so it can be reused across runs.
    """
    def __init__(self, cache_dir: str = "svd_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache: Dict[Tuple[str, int, int], Tensor] = {}
    
    def _get_cache_key(self, model_id: str, layer: int, expert: int) -> Tuple[str, int, int]:
        return (model_id, layer, expert)
    
    def _get_cache_file(self, model_id: str, layer: int, expert: int) -> Path:
        # Create safe filename from model_id
        safe_model_id = model_id.replace("/", "_").replace("-", "_")
        filename = f"{safe_model_id}_layer{layer}_expert{expert}.pkl"
        return self.cache_dir / filename
    
    def get(self, model_id: str, layer: int, expert: int, w_in: Tensor) -> Optional[Tensor]:
        """
        Get cached V matrix, or None if not cached.
        """
        key = self._get_cache_key(model_id, layer, expert)
        
        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Check disk cache
        cache_file = self._get_cache_file(model_id, layer, expert)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    V = pickle.load(f)
                # Store in memory cache for faster access
                self._memory_cache[key] = V
                return V
            except Exception as e:
                print(f"Warning: Failed to load cache for {cache_file}: {e}")
        
        return None
    
    def put(self, model_id: str, layer: int, expert: int, V: Tensor) -> None:
        """
        Cache V matrix to both memory and disk.
        """
        key = self._get_cache_key(model_id, layer, expert)
        
        # Store in memory cache
        self._memory_cache[key] = V
        
        # Store on disk
        cache_file = self._get_cache_file(model_id, layer, expert)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(V, f)
        except Exception as e:
            print(f"Warning: Failed to save cache to {cache_file}: {e}")
    
    def clear(self) -> None:
        """Clear memory cache (disk cache remains)."""
        self._memory_cache.clear()
    
    def clear_all(self) -> None:
        """Clear both memory and disk cache."""
        self._memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print(f"Cleared all SVD cache files from {self.cache_dir}")

