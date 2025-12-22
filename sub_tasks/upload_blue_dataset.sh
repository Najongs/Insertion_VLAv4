#!/bin/bash
# Upload Blue Point dataset (10 episodes) to Hugging Face Hub

# Configuration
EPISODE_DIR="/home/najo/NAS/VLA/dataset/New_dataset2/Blue_point"
REPO_ID="Najongs/vla-insertion-blue-point"  # Change to your username
DATASET_NAME="VLA Insertion - Blue Point"
MAX_EPISODES=10
OUTPUT_DIR="outputs/dataset_upload_blue"
PRIVATE="--private"  # Remove to make dataset public

# Check if directory exists
if [ ! -d "$EPISODE_DIR" ]; then
    echo "Error: Episode directory not found at $EPISODE_DIR"
    exit 1
fi

echo "=========================================="
echo "VLA Dataset Upload - Blue Point"
echo "=========================================="
echo "Episode directory: $EPISODE_DIR"
echo "Repository: $REPO_ID"
echo "Max episodes: $MAX_EPISODES"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Add lerobot to PYTHONPATH
export PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH

# Check if HF token is set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable not set."
    echo "You can set it with: export HF_TOKEN=your_token_here"
    echo ""
fi

# Run upload script
python upload_dataset.py \
    --episode_dir "$EPISODE_DIR" \
    --max_episodes $MAX_EPISODES \
    --repo_id "$REPO_ID" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    $PRIVATE

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Upload completed successfully!"
    echo "Dataset available at: https://huggingface.co/datasets/$REPO_ID"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Upload failed!"
    echo "=========================================="
    exit 1
fi
