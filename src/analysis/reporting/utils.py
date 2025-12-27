"""Shared utilities for reporting and visualization."""
from io import BytesIO
from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt


def build_run_id(metadata: dict) -> str:
    """Build a consistent run identifier from metadata.
    
    Format: L{layer}_s{n_shuffles}_{timestamp}
    
    Args:
        metadata: Metadata dictionary from results file
        
    Returns:
        Run identifier string
    """
    run_parts = [f"L{metadata.get('layer', '?')}"]
    if "n_shuffles" in metadata:
        run_parts.append(f"s{metadata['n_shuffles']}")
    # Add timestamp for uniqueness if available
    if "timestamp" in metadata:
        run_parts.append(metadata['timestamp'])
    elif "datetime" in metadata:
        # Extract timestamp from datetime if available
        datetime_str = metadata['datetime']
        if isinstance(datetime_str, str) and '_' in datetime_str:
            # Try to extract timestamp from ISO format
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                timestamp = dt.strftime("%Y%m%d_%H%M%S")
                run_parts.append(timestamp)
            except (ValueError, AttributeError):
                pass
    
    return "_".join(run_parts)


def get_layer_label(metadata: dict) -> str:
    """Get a simple layer label for legends (no timestamp).
    
    Format: L{layer}
    
    Args:
        metadata: Metadata dictionary from results file
        
    Returns:
        Layer label string (e.g., "L5")
    """
    return f"L{metadata.get('layer', '?')}"


def extract_layer_number(layer_label: str) -> int:
    """Extract layer number from label like 'L5' -> 5, 'L10' -> 10.
    
    Args:
        layer_label: Layer label string (e.g., 'L5', 'L10')
        
    Returns:
        Layer number as integer, or 999999 if cannot parse (to sort last)
    """
    try:
        # Remove 'L' prefix and convert to int
        return int(layer_label.lstrip('L'))
    except (ValueError, AttributeError):
        return 999999  # Sort unparseable labels last


def get_summary_columns(df) -> list:
    """Get standard summary columns for analysis.
    
    Args:
        df: DataFrame with analysis results
        
    Returns:
        List of column names for summary statistics
    """
    summary_cols = ["align", "delta_vs_shuffle", "z_vs_shuffle", "effect_over_random"]
    if 1 in df["k"].values:
        summary_cols.append("cos_squared")
    return summary_cols


def display_plot_in_jupyter(fig, close_fig: bool = True) -> None:
    """Display a matplotlib figure in Jupyter notebook, handling different backends.
    
    Args:
        fig: Matplotlib figure to display
        close_fig: Whether to close the figure after displaying
    """
    try:
        from IPython import get_ipython
        
        is_jupyter = get_ipython() is not None
        if is_jupyter:
            backend = matplotlib.get_backend()
            is_agg_backend = backend.lower() == 'agg'
            
            if is_agg_backend:
                # Agg backend doesn't display - use Image display
                from IPython.display import display, Image as IPImage
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                display(IPImage(data=buf.read()))
                buf.close()
                if close_fig:
                    plt.close(fig)
            else:
                # Non-Agg backend - try plt.show() (works with %matplotlib inline)
                try:
                    plt.show()
                    if close_fig:
                        plt.close(fig)
                except Exception:
                    # Fallback: convert to image and display
                    from IPython.display import display, Image as IPImage
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    display(IPImage(data=buf.read()))
                    buf.close()
                    if close_fig:
                        plt.close(fig)
        else:
            # Not in Jupyter, try to show, if fails just close
            try:
                plt.show()
            except Exception:
                pass
            if close_fig:
                plt.close(fig)
    except (ImportError, NameError):
        # Not in Jupyter, just close
        if close_fig:
            plt.close(fig)


def figure_to_base64(fig, dpi: int = 150) -> str:
    """Convert a matplotlib figure to base64-encoded PNG string.
    
    Args:
        fig: Matplotlib figure
        dpi: Resolution for saving
        
    Returns:
        Base64-encoded image string
    """
    import base64
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_data


def figure_to_bytes(fig, dpi: int = 150) -> bytes:
    """Convert a matplotlib figure to PNG bytes.
    
    Args:
        fig: Matplotlib figure
        dpi: Resolution for saving
        
    Returns:
        PNG image bytes
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    img_bytes = buf.read()
    buf.close()
    return img_bytes

