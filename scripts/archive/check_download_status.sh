#!/bin/bash
# Quick status check for COCO download

cd "$(dirname "$0")/.."

ZIP_FILE="datasets/coco_raw/train2017.zip"

echo "=== COCO Download Status ==="
echo ""

# Check if file exists
if [ ! -f "$ZIP_FILE" ]; then
    echo "❌ File not found: $ZIP_FILE"
    echo "Run: ./scripts/resume_coco_download.sh"
    exit 1
fi

# Get file size
SIZE_BYTES=$(stat -f%z "$ZIP_FILE" 2>/dev/null || stat -c%s "$ZIP_FILE" 2>/dev/null)
SIZE_GB=$(echo "scale=2; $SIZE_BYTES / (1024^3)" | bc)
EXPECTED_GB=18
PROGRESS=$(echo "scale=1; 100 * $SIZE_GB / $EXPECTED_GB" | bc)

echo "📦 File: $ZIP_FILE"
echo "📊 Size: ${SIZE_GB} GB / ${EXPECTED_GB} GB"
echo "📈 Progress: ${PROGRESS}%"
echo ""

# Check if curl is running
if pgrep -f "curl.*train2017.zip" > /dev/null; then
    echo "✅ Download is ACTIVE (curl process running)"
    echo ""
    echo "Monitor with:"
    echo "  watch -n 5 'ls -lh $ZIP_FILE'"
    echo "  python scripts/monitor_coco_download.py"
else
    echo "⚠️  Download process NOT running"
    echo ""
    if (( $(echo "$SIZE_GB < $EXPECTED_GB" | bc -l) )); then
        echo "Resume download with:"
        echo "  ./scripts/resume_coco_download.sh"
    else
        echo "✅ Download appears complete! Verify with:"
        echo "  python scripts/download_coco.py --verify-only"
    fi
fi

