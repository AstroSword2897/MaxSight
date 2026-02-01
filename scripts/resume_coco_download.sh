#!/bin/bash
# Resume COCO train2017.zip download

cd "$(dirname "$0")/.."

ZIP_FILE="datasets/coco_raw/train2017.zip"
URL="http://images.cocodataset.org/zips/train2017.zip"

# Kill any existing curl processes for this download
pkill -f "curl.*train2017.zip" || true
sleep 2

# Check current file size
if [ -f "$ZIP_FILE" ]; then
    CURRENT_SIZE=$(stat -f%z "$ZIP_FILE" 2>/dev/null || stat -c%s "$ZIP_FILE" 2>/dev/null)
    CURRENT_GB=$(echo "scale=2; $CURRENT_SIZE / (1024^3)" | bc)
    echo "Resuming download from ${CURRENT_GB} GB..."
else
    echo "Starting new download..."
fi

# Resume download with curl
# -C - : Resume from where it left off
# -L : Follow redirects
# --progress-bar : Show progress
# --retry 3 : Retry on failure
# --max-time 3600 : 1 hour timeout per request (will retry)
# -o : Output file
curl -L -C - --progress-bar --retry 3 --max-time 3600 \
    -o "$ZIP_FILE" "$URL" &

CURL_PID=$!
echo "Download started (PID: $CURL_PID)"
echo "Monitor progress with: ls -lh $ZIP_FILE"
echo "Or: python scripts/monitor_coco_download.py"

