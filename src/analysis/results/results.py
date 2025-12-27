"""Object-oriented utilities for saving, loading, and comparing analysis results."""
import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.analysis.core.analyzer import AlignmentAnalyzer
from src.analysis.core.data_structures import AlignmentResult


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
        # Fallback: assume this file is in src/analysis/results/, go up 3 levels
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        return project_root


class ResultsManager:
    """Manages saving, loading, and comparing analysis results.
    
    Results directory is always created relative to the project root (GaleMoE/),
    regardless of where the script is run from.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize ResultsManager.
        
        Args:
            output_dir: Optional output directory path. If None, uses 'results' 
                      relative to project root (GaleMoE/). If relative path provided,
                      it's relative to project root. If absolute path provided, uses as-is.
        """
        if output_dir is None:
            project_root = _get_project_root()
            self.output_dir = project_root / "results"
        else:
            output_path = Path(output_dir)
            if output_path.is_absolute():
                self.output_dir = output_path
            else:
                project_root = _get_project_root()
                self.output_dir = project_root / output_dir
        
        self.output_dir.mkdir(exist_ok=True)
    
    def save(
        self,
        results: List[AlignmentResult],
        analyzer: AlignmentAnalyzer,
        layer: int,
        model_id: str,
    ) -> str:
        """
        Save results to a file with metadata for easy comparison across runs.
        
        Args:
            results: List of AlignmentResult objects
            analyzer: AlignmentAnalyzer used for analysis (any method)
            layer: Layer number analyzed
            model_id: Model identifier
            
        Returns:
            Path to saved file
        """
        # Create filename with metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method_name = analyzer.method_name
        
        # Get metadata from analyzer (handles optional fields gracefully)
        metadata_dict = analyzer.get_metadata()
        n_shuffles = metadata_dict.get("n_shuffles", "unknown")
        seed = metadata_dict.get("seed", "unknown")
        
        # Build filename with available metadata
        # Format: layer{layer}_{method}_k{min}-{max}_shuffles{n}_seed{s}_{timestamp}
        filename_parts = [f"layer{layer}", method_name]
        
        # Add k_list info (compact format for long lists)
        k_list = list(analyzer.k_list)
        if len(k_list) <= 5:
            k_str = "_".join(map(str, k_list))
            filename_parts.append(f"k{k_str}")
        else:
            # For long k lists, use range format
            k_str = f"k{k_list[0]}-{k_list[-1]}-{len(k_list)}"
            filename_parts.append(k_str)
        
        # Always include n_shuffles if available (critical parameter)
        if n_shuffles != "unknown":
            filename_parts.append(f"shuffles{n_shuffles}")
        else:
            filename_parts.append("shuffles-unknown")
        
        # Always include seed if available
        if seed != "unknown":
            filename_parts.append(f"seed{seed}")
        
        filename_parts.append(timestamp)
        filename = "_".join(filename_parts) + ".json"
        filepath = self.output_dir / filename
        
        # Prepare data with metadata
        data = {
            "metadata": {
                "model_id": model_id,
                "layer": layer,
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "n_experts": len(set(r.expert for r in results)),
                **metadata_dict,  # Include all method-specific metadata
            },
            "results": [asdict(r) for r in results]
        }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Results saved to: {filepath}")
        return str(filepath)
    
    def load(self, filepath: str) -> Tuple[dict, List[AlignmentResult]]:
        """
        Load results from a saved file.
        
        Args:
            filepath: Path to results JSON file
            
        Returns:
            (metadata_dict, list_of_results)
        """
        filepath = Path(filepath)
        if not filepath.is_absolute():
            # Try relative to output_dir first, then as-is
            potential_path = self.output_dir / filepath
            if potential_path.exists():
                filepath = potential_path
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metadata = data["metadata"]
        results = [AlignmentResult(**r) for r in data["results"]]
        
        return metadata, results
    
    def compare(self, result_files: List[str]) -> pd.DataFrame:
        """
        Compare multiple result files side by side.
        
        Args:
            result_files: List of paths to result JSON files
            
        Returns:
            DataFrame with comparison results
        """
        all_data = []
        for filepath in result_files:
            metadata, results = self.load(filepath)
            df = pd.DataFrame([asdict(r) for r in results])
            
            # Add metadata columns
            summary = df.groupby("k")[["align", "delta_vs_shuffle", "z_vs_shuffle", "effect_over_random"]].mean()
            summary = summary.reset_index()
            
            # Build run identifier with available metadata
            run_parts = [f"L{metadata['layer']}", f"k{metadata['k_list']}"]
            if "n_shuffles" in metadata:
                run_parts.append(f"s{metadata['n_shuffles']}")
            summary["run"] = "_".join(run_parts)
            summary["timestamp"] = metadata["timestamp"]
            
            all_data.append(summary)
        
        combined = pd.concat(all_data, ignore_index=True)
        print("\nComparison of runs:")
        print("=" * 80)
        print(combined.to_string(index=False))
        return combined
    
    def list_results(self, pattern: Optional[str] = None) -> List[Path]:
        """
        List all result files in the output directory.
        
        Args:
            pattern: Optional pattern to filter files (e.g., "layer10_*")
            
        Returns:
            List of result file paths
        """
        if pattern:
            files = list(self.output_dir.glob(pattern))
        else:
            files = list(self.output_dir.glob("*.json"))
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


# Backward compatibility functions
def save_results(
    results: List[AlignmentResult],
    analyzer: AlignmentAnalyzer,
    layer: int,
    model_id: str,
    output_dir: str = "results"
) -> str:
    """Backward compatibility wrapper for ResultsManager.save()."""
    manager = ResultsManager(output_dir)
    return manager.save(results, analyzer, layer, model_id)


def load_results(filepath: str) -> Tuple[dict, List[AlignmentResult]]:
    """Backward compatibility wrapper for ResultsManager.load()."""
    manager = ResultsManager()
    return manager.load(filepath)


def compare_runs(result_files: List[str]) -> None:
    """Backward compatibility wrapper for ResultsManager.compare()."""
    manager = ResultsManager()
    manager.compare(result_files)

