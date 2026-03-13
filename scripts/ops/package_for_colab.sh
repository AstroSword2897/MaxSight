#!/bin/bash
# Package MaxSight for Google Colab deployment

set -e

cd /Users/nani/2026-Prototype

echo "📦 Packaging MaxSight for Google Colab..."
echo ""

# Option 1: Code only (recommended)
echo "Creating code package (no images)..."
tar -czf maxsight_code.tar.gz \
  ml/ \
  scripts/ \
  datasets/cleaned_splits/ \
  requirements_colab.txt \
  --exclude="*.pyc" \
  --exclude="__pycache__" \
  --exclude="*.pth" \
  --exclude=".git" \
  --exclude="checkpoints"

SIZE=$(du -h maxsight_code.tar.gz | cut -f1)
echo "✅ Created: maxsight_code.tar.gz ($SIZE)"
echo ""

echo "📤 Next steps:"
echo "1. Upload maxsight_code.tar.gz to Google Drive"
echo "2. Upload MaxSight_Colab_Training.ipynb to Colab"
echo "3. Follow COLAB_SETUP_GUIDE.md"
echo ""
echo "Optional: Create full package with images (takes longer)"
echo "  tar -czf maxsight_full.tar.gz ml/ scripts/ datasets/"
