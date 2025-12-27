"""Utility to check download configuration and diagnose download performance."""
import os
import sys


def check_download_config() -> dict:
    """
    Check the current download configuration and diagnose potential issues.
    
    Returns:
        Dictionary with diagnostic information
    """
    diagnostics = {
        "hf_transfer_enabled": False,
        "hf_transfer_installed": False,
        "max_workers": None,
        "environment_vars": {},
        "recommendations": [],
    }
    
    # Check if HF_HUB_ENABLE_HF_TRANSFER is set
    hf_transfer_env = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "")
    diagnostics["hf_transfer_enabled"] = hf_transfer_env.lower() in ("1", "true", "yes")
    diagnostics["environment_vars"]["HF_HUB_ENABLE_HF_TRANSFER"] = hf_transfer_env or "not set"
    
    # Check if hf-transfer package is installed
    try:
        import hf_transfer
        diagnostics["hf_transfer_installed"] = True
        diagnostics["hf_transfer_version"] = getattr(hf_transfer, "__version__", "unknown")
    except ImportError:
        diagnostics["hf_transfer_installed"] = False
        diagnostics["recommendations"].append(
            "Install hf-transfer for 10-100x faster downloads: pip install hf-transfer"
        )
    
    # Check if hf-transfer is enabled but not installed
    if diagnostics["hf_transfer_enabled"] and not diagnostics["hf_transfer_installed"]:
        diagnostics["recommendations"].append(
            "HF_HUB_ENABLE_HF_TRANSFER is set but hf-transfer is not installed. "
            "Install it with: pip install hf-transfer"
        )
    
    # Check if hf-transfer is installed but not enabled
    if diagnostics["hf_transfer_installed"] and not diagnostics["hf_transfer_enabled"]:
        diagnostics["recommendations"].append(
            "hf-transfer is installed but not enabled. "
            "The code should enable it automatically, but you can set: "
            "export HF_HUB_ENABLE_HF_TRANSFER=1"
        )
    
    # Check other relevant environment variables
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
    print(f"  HF_HUB_ENABLE_HF_TRANSFER = {diag['environment_vars']['HF_HUB_ENABLE_HF_TRANSFER']}")
    
    print("\nEnvironment Variables:")
    for key, value in diag["environment_vars"].items():
        print(f"  {key} = {value}")
    
    if diag["recommendations"]:
        print("\n⚠️  Recommendations:")
        for i, rec in enumerate(diag["recommendations"], 1):
            print(f"  {i}. {rec}")
    else:
        print("\n✓ Configuration looks good!")
    
    print("\n" + "=" * 60)
    
    # Test actual download speed
    print("\nTo test download speed, the code will automatically:")
    print("  - Use up to 16 parallel workers")
    print("  - Enable hf-transfer if available")
    print("  - Resume interrupted downloads")
    print("\nIf downloads are still slow, check:")
    print("  1. Network connection speed")
    print("  2. Hugging Face Hub status")
    print("  3. Firewall/proxy settings")
    print("=" * 60)


if __name__ == "__main__":
    print_download_diagnostics()

