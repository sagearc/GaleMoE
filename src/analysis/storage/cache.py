"""Cache for SVD decompositions used in router-expert analysis."""
import subprocess
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

from torch import Tensor


def _get_project_root() -> Path:
    """Get the project root directory using git.
    
    Uses 'git rev-parse --show-toplevel' to find the git repository root,
    which should be the project root (GaleMoE/).
    
    Falls back to going up from current file if git is not available.
    """
    try:
        # Try to get git root
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        git_root = Path(result.stdout.strip())
        return git_root
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: assume this file is in src/analysis/storage/, go up 3 levels
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        return project_root


class SVDCache:
    """
    Cache for SVD decompositions across the entire network.
    Persists to disk so it can be reused across runs.
    
    Cache directory is always created relative to the project root (GaleMoE/),
    regardless of where the script is run from.
    """
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize SVD cache.
        
        Args:
            cache_dir: Optional cache directory path. If None, uses 'svd_cache' 
                      relative to project root (GaleMoE/). If relative path provided,
                      it's relative to project root. If absolute path provided, uses as-is.
        """
        if cache_dir is None:
            project_root = _get_project_root()
            self.cache_dir = project_root / "svd_cache"
        else:
            cache_path = Path(cache_dir)
            if cache_path.is_absolute():
                self.cache_dir = cache_path
            else:
                project_root = _get_project_root()
                self.cache_dir = project_root / cache_dir
        
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
        """Get cached V matrix, or None if not cached.
        
        Args:
            model_id: Model identifier
            layer: Layer number
            expert: Expert index
            w_in: Expert weight matrix (used for cache key, not validated)
            
        Returns:
            Cached V matrix if available, None otherwise
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
        """Cache V matrix to both memory and disk.
        
        Args:
            model_id: Model identifier
            layer: Layer number
            expert: Expert index
            V: V matrix (right singular vectors) to cache
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

