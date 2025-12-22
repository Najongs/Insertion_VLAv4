#!/bin/bash
# Upload all color datasets to Hugging Face Hub

BASE_DIR="/home/najo/NAS/VLA/dataset/New_dataset2"
USERNAME="Najongs"  # Change to your Hugging Face username
MAX_EPISODES=10
PRIVATE="--private"  # Remove to make datasets public

# Add lerobot to PYTHONPATH
export PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH

# Check if HF token is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable not set."
    echo "Set it with: export HF_TOKEN=your_token_here"
    exit 1
fi

echo "=========================================="
echo "Upload All VLA Insertion Datasets"
echo "=========================================="
echo "Base directory: $BASE_DIR"
echo "Username: $USERNAME"
echo "Max episodes per color: $MAX_EPISODES"
echo "=========================================="
echo ""

# Array of colors
COLORS=("Blue_point" "Green_point" "Red_point" "White_point" "Yellow_point" "Eye_trocar")

# Upload each color dataset
for COLOR in "${COLORS[@]}"; do
    EPISODE_DIR="$BASE_DIR/$COLOR"
    REPO_ID="$USERNAME/vla-insertion-${COLOR,,}"  # Convert to lowercase
    DATASET_NAME="VLA Insertion - $COLOR"
    OUTPUT_DIR="outputs/dataset_upload_${COLOR,,}"

    echo ""
    echo "=========================================="
    echo "Uploading: $COLOR"
    echo "=========================================="

    if [ ! -d "$EPISODE_DIR" ]; then
        echo "Warning: Directory not found: $EPISODE_DIR"
        echo "Skipping..."
        continue
    fi

    python upload_dataset.py \
        --episode_dir "$EPISODE_DIR" \
        --max_episodes $MAX_EPISODES \
        --repo_id "$REPO_ID" \
        --dataset_name "$DATASET_NAME" \
        --output_dir "$OUTPUT_DIR" \
        $PRIVATE

    if [ $? -eq 0 ]; then
        echo "✅ $COLOR uploaded successfully!"
    else
        echo "❌ $COLOR upload failed!"
    fi

    echo ""
done

echo "=========================================="
echo "All uploads completed!"
echo "=========================================="
echo ""
echo "Datasets available at:"
for COLOR in "${COLORS[@]}"; do
    REPO_ID="$USERNAME/vla-insertion-${COLOR,,}"
    echo "  - https://huggingface.co/datasets/$REPO_ID"
done
echo "=========================================="
