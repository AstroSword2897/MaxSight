#!/usr/bin/env python3
"""Find best_model.pt in checkpoints_color_blindness and convert to CoreML."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

def find_checkpoint():
    """Search common locations for checkpoints_color_blindness/best_model.pt."""
    import os
    home = Path.home()
    
    candidates = [
        # Check CHECKPOINTS_BASE env var
        os.environ.get("CHECKPOINTS_BASE"),
        # Local repo
        REPO / "checkpoints",
        REPO / "backups",
        # Google Drive locations
        home / "Google Drive" / "My Drive" / "MaxSight",
    ]
    
    # Add Google Drive cloud storage paths
    for gd_path in home.glob("Library/CloudStorage/GoogleDrive-*/My Drive/MaxSight"):
        candidates.append(gd_path)
    
    # Colab path
    candidates.append(Path("/content/drive/MyDrive/MaxSight"))
    
    for base in candidates:
        if not base:
            continue
        base = Path(base)
        if not base.exists():
            continue
        
        checkpoint_path = base / "checkpoints_color_blindness" / "best_model.pt"
        if checkpoint_path.exists():
            return checkpoint_path.resolve()
    
    return None

def main():
    checkpoint = find_checkpoint()
    
    if checkpoint is None:
        print("Could not find checkpoints_color_blindness/best_model.pt")
        print("\nSearched in:")
        print("  - CHECKPOINTS_BASE environment variable")
        print("  - <repo>/checkpoints")
        print("  - <repo>/backups")
        print("  - ~/Google Drive/My Drive/MaxSight")
        print("  - ~/Library/CloudStorage/GoogleDrive-*/My Drive/MaxSight")
        print("  - /content/drive/MyDrive/MaxSight (Colab)")
        print("\nPlease provide the full path to your .pt file:")
        print("  python scripts/convert_pt_to_coreml.py <path/to/best_model.pt>")
        return 1
    
    print(f"Found checkpoint: {checkpoint}")
    print(f"Converting to CoreML...\n")
    
    # Import and run the conversion script
    from scripts.convert_pt_to_coreml import main as convert_main
    import sys as sys_module
    
    # Set up argv to simulate command-line call
    original_argv = sys_module.argv
    sys_module.argv = ["convert_pt_to_coreml.py", str(checkpoint), "--condition", "color_blindness"]
    
    try:
        result = convert_main()
        return result
    finally:
        sys_module.argv = original_argv

if __name__ == "__main__":
    sys.exit(main())
