#!/bin/bash
# Setup rclone and upload datasets to Google Drive

set -e

PROJECT_DIR="/Users/nani/2026-Prototype"
DRIVE_FOLDER="MaxSight"

echo "=========================================="
echo "rclone Setup & Upload Script"
echo "=========================================="

# Check if rclone is installed
if ! command -v rclone &> /dev/null; then
    echo ""
    echo "📦 Installing rclone..."
    brew install rclone
    echo "✅ rclone installed"
else
    echo ""
    echo "✅ rclone is already installed"
    rclone version
fi

echo ""
echo "=========================================="
echo "Configure Google Drive"
echo "=========================================="
echo ""
echo "If you haven't configured rclone yet, run:"
echo "  rclone config"
echo ""
echo "Steps:"
echo "  1. Choose 'n' for new remote"
echo "  2. Name it 'gdrive'"
echo "  3. Choose 'drive' (Google Drive)"
echo "  4. Follow authentication steps"
echo ""
read -p "Have you configured rclone? (y/n): " configured

if [ "$configured" != "y" ]; then
    echo ""
    echo "Running rclone config..."
    rclone config
fi

echo ""
echo "=========================================="
echo "Check What to Upload"
echo "=========================================="

# Check datasets
echo ""
echo "📊 Local Datasets:"

# Open Images V6
if [ -d "$HOME/fiftyone/open-images-v6/validation" ]; then
    COUNT=$(find "$HOME/fiftyone/open-images-v6/validation" -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
    SIZE=$(du -sh "$HOME/fiftyone/open-images-v6/validation" 2>/dev/null | cut -f1)
    echo "  ✅ Open Images V6: $COUNT images ($SIZE)"
    OI6_SOURCE="$HOME/fiftyone/open-images-v6/validation"
elif [ -d "$PROJECT_DIR/datasets/open_images_v6/validation" ]; then
    COUNT=$(find "$PROJECT_DIR/datasets/open_images_v6/validation" -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
    SIZE=$(du -sh "$PROJECT_DIR/datasets/open_images_v6/validation" 2>/dev/null | cut -f1)
    echo "  ✅ Open Images V6: $COUNT images ($SIZE)"
    OI6_SOURCE="$PROJECT_DIR/datasets/open_images_v6"
else
    echo "  ⚠️  Open Images V6: Not found (run reorganize script first)"
    OI6_SOURCE=""
fi

# ADE20K
if [ -d "$PROJECT_DIR/datasets/ade20k" ]; then
    COUNT=$(find "$PROJECT_DIR/datasets/ade20k/images/validation" -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
    SIZE=$(du -sh "$PROJECT_DIR/datasets/ade20k" 2>/dev/null | cut -f1)
    echo "  ✅ ADE20K: $COUNT images ($SIZE)"
    ADE20K_SOURCE="$PROJECT_DIR/datasets/ade20k"
else
    echo "  ⚠️  ADE20K: Not found"
    ADE20K_SOURCE=""
fi

# Checkpoints
if [ -d "$PROJECT_DIR/checkpoints" ]; then
    SIZE=$(du -sh "$PROJECT_DIR/checkpoints" 2>/dev/null | cut -f1)
    COUNT=$(ls "$PROJECT_DIR/checkpoints"/*.pt 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✅ Checkpoints: $COUNT files ($SIZE)"
    CHECKPOINTS_SOURCE="$PROJECT_DIR/checkpoints"
else
    echo "  ⚠️  Checkpoints: Not found"
    CHECKPOINTS_SOURCE=""
fi

# Splits
if [ -d "$PROJECT_DIR/datasets/cleaned_splits" ]; then
    SIZE=$(du -sh "$PROJECT_DIR/datasets/cleaned_splits" 2>/dev/null | cut -f1)
    echo "  ✅ Dataset Splits: ($SIZE)"
    SPLITS_SOURCE="$PROJECT_DIR/datasets/cleaned_splits"
else
    echo "  ⚠️  Dataset Splits: Not found"
    SPLITS_SOURCE=""
fi

echo ""
echo "=========================================="
echo "Upload Options"
echo "=========================================="
echo ""
echo "What would you like to upload?"
echo ""
echo "1. All datasets (Open Images V6, ADE20K, Splits)"
echo "2. Checkpoints only"
echo "3. Everything (datasets + checkpoints)"
echo "4. Custom selection"
echo ""
read -p "Choose (1-4): " choice

case $choice in
    1)
        UPLOAD_DATASETS=true
        UPLOAD_CHECKPOINTS=false
        ;;
    2)
        UPLOAD_DATASETS=false
        UPLOAD_CHECKPOINTS=true
        ;;
    3)
        UPLOAD_DATASETS=true
        UPLOAD_CHECKPOINTS=true
        ;;
    4)
        echo ""
        read -p "Upload datasets? (y/n): " upload_ds
        read -p "Upload checkpoints? (y/n): " upload_ckpt
        UPLOAD_DATASETS=[ "$upload_ds" = "y" ]
        UPLOAD_CHECKPOINTS=[ "$upload_ckpt" = "y" ]
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Uploading to Google Drive"
echo "=========================================="
echo ""
echo "Destination: gdrive:$DRIVE_FOLDER/"
echo ""

# Upload datasets
if [ "$UPLOAD_DATASETS" = true ]; then
    echo "📤 Uploading datasets..."
    
    # Create base directory
    rclone mkdir "gdrive:$DRIVE_FOLDER/datasets" 2>/dev/null || true
    
    # Open Images V6
    if [ -n "$OI6_SOURCE" ]; then
        echo ""
        echo "  Uploading Open Images V6..."
        if [ -d "$HOME/fiftyone/open-images-v6/validation" ]; then
            # Need to reorganize first or upload from FiftyOne location
            echo "    ⚠️  Note: Uploading from FiftyOne location"
            echo "    Consider reorganizing first: python scripts/reorganize_open_images.py"
            rclone copy "$HOME/fiftyone/open-images-v6" "gdrive:$DRIVE_FOLDER/datasets/open_images_v6" --progress --transfers 4
        else
            rclone copy "$OI6_SOURCE" "gdrive:$DRIVE_FOLDER/datasets/open_images_v6" --progress --transfers 4
        fi
        echo "    ✅ Open Images V6 uploaded"
    fi
    
    # ADE20K
    if [ -n "$ADE20K_SOURCE" ]; then
        echo ""
        echo "  Uploading ADE20K..."
        rclone copy "$ADE20K_SOURCE" "gdrive:$DRIVE_FOLDER/datasets/ade20k" --progress --transfers 4
        echo "    ✅ ADE20K uploaded"
    fi
    
    # Splits
    if [ -n "$SPLITS_SOURCE" ]; then
        echo ""
        echo "  Uploading dataset splits..."
        rclone copy "$SPLITS_SOURCE" "gdrive:$DRIVE_FOLDER/datasets/cleaned_splits" --progress
        echo "    ✅ Splits uploaded"
    fi
fi

# Upload checkpoints
if [ "$UPLOAD_CHECKPOINTS" = true ]; then
    if [ -n "$CHECKPOINTS_SOURCE" ]; then
        echo ""
        echo "📤 Uploading checkpoints..."
        rclone copy "$CHECKPOINTS_SOURCE" "gdrive:$DRIVE_FOLDER/checkpoints" --progress
        echo "    ✅ Checkpoints uploaded"
    fi
fi

echo ""
echo "=========================================="
echo "Upload Complete!"
echo "=========================================="
echo ""
echo "📁 Files are now in: gdrive:$DRIVE_FOLDER/"
echo ""
echo "In Colab, access them at:"
echo "  /content/drive/MyDrive/$DRIVE_FOLDER/"
echo ""
