#!/usr/bin/env python3
"""
Monitor COCO dataset download progress.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def format_size(size_bytes):
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def monitor_download():
    """Monitor download progress."""
    zip_path = Path("datasets/coco_raw/train2017.zip")
    expected_size = 18 * 1024**3  # 18GB
    
    if not zip_path.exists():
        print("train2017.zip not found. Download may not have started.")
        return
    
    print("Monitoring COCO train2017.zip download...")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_size = 0
    start_time = time.time()
    
    try:
        while True:
            if zip_path.exists():
                current_size = zip_path.stat().st_size
                progress = (current_size / expected_size) * 100 if expected_size > 0 else 0
                
                # Calculate download speed
                elapsed = time.time() - start_time
                if elapsed > 0:
                    speed = (current_size - last_size) / elapsed  # bytes per second
                    speed_str = format_size(speed) + "/s"
                    
                    # Estimate time remaining
                    remaining_bytes = expected_size - current_size
                    if speed > 0:
                        eta_seconds = remaining_bytes / speed
                        eta_minutes = eta_seconds / 60
                        eta_str = f"{eta_minutes:.1f} min" if eta_minutes < 60 else f"{eta_minutes/60:.1f} hours"
                    else:
                        eta_str = "calculating..."
                else:
                    speed_str = "calculating..."
                    eta_str = "calculating..."
                
                print(f"\rProgress: {format_size(current_size)} / {format_size(expected_size)} "
                      f"({progress:.1f}%) | Speed: {speed_str} | ETA: {eta_str}", end='', flush=True)
                
                last_size = current_size
                start_time = time.time()
                
                # Check if download is complete (file size matches expected)
                if current_size >= expected_size * 0.99:  # 99% threshold
                    print("\n\n✅ Download appears complete!")
                    print(f"Final size: {format_size(current_size)}")
                    break
            else:
                print("File not found. Waiting for download to start...")
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        if zip_path.exists():
            current_size = zip_path.stat().st_size
            progress = (current_size / expected_size) * 100 if expected_size > 0 else 0
            print(f"Current progress: {format_size(current_size)} ({progress:.1f}%)")


if __name__ == "__main__":
    monitor_download()

