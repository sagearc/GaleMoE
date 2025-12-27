"""Object-oriented utilities for saving, loading, and comparing analysis results."""
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.analysis.core.analyzer import AlignmentAnalyzer
from src.analysis.core.data_structures import AlignmentResult


class ResultsManager:
    """Manages saving, loading, and comparing analysis results."""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize ResultsManager.
        
        Args:
            output_dir: Directory to save/load results from
        """
        self.output_dir = Path(output_dir)
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
        k_str = "_".join(map(str, analyzer.k_list))
        
        # Get metadata from analyzer (handles optional fields gracefully)
        metadata_dict = analyzer.get_metadata()
        n_shuffles = metadata_dict.get("n_shuffles", "unknown")
        seed = metadata_dict.get("seed", "unknown")
        
        # Build filename with available metadata
        filename_parts = [f"layer{layer}", method_name, f"k{k_str}"]
        if n_shuffles != "unknown":
            filename_parts.append(f"shuffles{n_shuffles}")
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

