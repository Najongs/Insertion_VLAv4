#!/bin/bash
# Upload trained SmolVLA checkpoint to Hugging Face Hub

# Configuration
CHECKPOINT_PATH="/home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_vla_insertion_multigpu/checkpoints/checkpoint_step_0016000.pt"
REPO_ID="Najongs/smolvla-insertion-vla"  # Change this to your Hugging Face username
OUTPUT_DIR="outputs/hf_upload"
PRIVATE="--private"  # Remove this flag to make the model public

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

echo "=========================================="
echo "SmolVLA to Hugging Face Hub Upload"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Repository: $REPO_ID"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Add lerobot to PYTHONPATH
export PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH

# Check if HF token is set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable not set."
    echo "You can set it with: export HF_TOKEN=your_token_here"
    echo "Or the script will prompt you to login."
    echo ""
fi

# Run upload script
python upload_to_huggingface.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --repo_id "$REPO_ID" \
    --output_dir "$OUTPUT_DIR" \
    $PRIVATE

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Upload completed successfully!"
    echo "Model available at: https://huggingface.co/$REPO_ID"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Upload failed!"
    echo "=========================================="
    exit 1
fi
