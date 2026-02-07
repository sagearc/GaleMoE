"""Shared utilities for the project."""
import re
import subprocess
from pathlib import Path
from typing import Optional


def sanitize_model_id(model_id: str) -> str:
    """Sanitize model ID for use in filenames.
    
    Replaces problematic characters, but keeps periods for version numbers.
    
    Args:
        model_id: Model identifier (e.g., 'mistralai/Mixtral-8x7B-v0.1')
        
    Returns:
        Sanitized string safe for filenames (e.g., 'mistralai_Mixtral_8x7B_v0.1')
    """
    # Replace only problematic characters, keep period and underscore
    return re.sub(r"[^\w.]", "_", model_id)


def get_project_root() -> Path:
    """Get the project root directory (git repository root).
    
    Returns:
        Path to the project root directory
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        # Fallback: assume we're in src/
        return Path(__file__).parent.parent


class OutputDir:
    """Resolves and holds an output directory path, creating it if needed."""

    def __init__(self, path: Path):
        self._path = path
        self._path.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    @classmethod
    def resolve(
        cls,
        output_dir: Optional[str] = None,
        default_subdir: str = "results",
    ) -> "OutputDir":
        """Resolve output_dir to an absolute Path relative to project root.

        Args:
            output_dir: None for default, or a path string (absolute or relative to project root).
            default_subdir: Subdirectory under project root when output_dir is None.

        Returns:
            OutputDir instance with the resolved path; directory is created.
        """
        if output_dir is None:
            project_root = get_project_root()
            resolved = project_root / default_subdir
        else:
            candidate = Path(output_dir)
            if candidate.is_absolute():
                resolved = candidate
            else:
                project_root = get_project_root()
                resolved = project_root / output_dir
        return cls(resolved)
