#!/usr/bin/env python3
"""Monitor Open Images V6 download progress."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def monitor_download():
    """Monitor download progress."""
    print("="*70)
    print("Open Images V6 Download Monitor")
    print("="*70)
    
    # Check FiftyOne download location.
    fo_dir = Path.home() / "fiftyone" / "open-images-v6" / "validation"
    expected_dir = ROOT / "datasets" / "open_images_v6" / "validation"
    
    print("\nMonitoring download progress...")
    print(f"  FiftyOne location: {fo_dir}")
    print(f"  Expected location: {expected_dir}")
    print("\nPress Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Check FiftyOne directory.
            if fo_dir.exists():
                img_count = len(list(fo_dir.rglob("*.jpg")))
                total_size = sum(f.stat().st_size for f in fo_dir.rglob("*.jpg") if f.is_file()) / (1024**2)  # MB.
                
                print(f"\r[{time.strftime('%H:%M:%S')}] Downloaded: {img_count:,} images ({total_size:.1f} MB)", end="", flush=True)
                
                # Check if download is complete (41,620 images expected)
                if img_count >= 41600:
                    print(f"\n\nOK Download appears complete! ({img_count:,} images)")
                    print("\n  Moving files to expected location...")
                    
                    # Move files.
                    expected_dir.mkdir(parents=True, exist_ok=True)
                    moved = 0
                    for img_path in fo_dir.rglob("*.jpg"):
                        rel_path = img_path.relative_to(fo_dir / "data")
                        if "data" in str(rel_path):
                            # Handle nested structure.
                            parts = rel_path.parts
                            if len(parts) > 1:
                                subdir = expected_dir / parts[0]
                                subdir.mkdir(exist_ok=True)
                                dest = subdir / parts[-1]
                            else:
                                dest = expected_dir / rel_path.name
                        else:
                            dest = expected_dir / rel_path.name
                        
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if not dest.exists():
                            img_path.rename(dest)
                            moved += 1
                    
                    print(f"  [ok] Moved {moved} images")
                    break
            else:
                print(f"\r[{time.strftime('%H:%M:%S')}] Waiting for download to start...", end="", flush=True)
            
            time.sleep(5)  # Check every 5 seconds.
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print("\nTo check manually:")
        print(f"  find {fo_dir} -name '*.jpg' | wc -l")
        print(f"  tail -f /tmp/open_images_download.log")


if __name__ == "__main__":
    monitor_download()





