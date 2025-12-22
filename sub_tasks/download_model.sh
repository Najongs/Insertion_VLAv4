#!/bin/bash
# Download trained SmolVLA model from Hugging Face Hub

# Configuration
REPO_ID="Najongs/smolvla-insertion-vla"  # Change to the model you want to download
OUTPUT_DIR="downloads/model"
SAVE_CHECKPOINT="--save_checkpoint"  # Add this to also save as checkpoint format

echo "=========================================="
echo "SmolVLA Model Download"
echo "=========================================="
echo "Repository: $REPO_ID"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Add lerobot to PYTHONPATH
export PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH

# Run download script
python download_model.py \
    --repo_id "$REPO_ID" \
    --output_dir "$OUTPUT_DIR" \
    $SAVE_CHECKPOINT

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Download completed successfully!"
    echo "Model saved to: $OUTPUT_DIR"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Download failed!"
    echo "=========================================="
    exit 1
fi
