#!/bin/bash
# Upload all task datasets to Hugging Face Hub

BASE_DIR="/home/irom/NAS/VLA/Insertion_VLAv4"
USERNAME="Najongs"  # ⚠️ Change to your Hugging Face username
MAX_EPISODES=50  # Maximum episodes to upload per task
PRIVATE="--private"  # Remove to make datasets public

# Add lerobot to PYTHONPATH
export PYTHONPATH=/home/irom/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH

# Check if HF token is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable not set."
    echo "Set it with: export HF_TOKEN=your_token_here"
    exit 1
fi

echo "=========================================="
echo "Upload All VLA Task Datasets"
echo "=========================================="
echo "Base directory: $BASE_DIR"
echo "Username: $USERNAME"
echo "Max episodes per task: $MAX_EPISODES"
echo "=========================================="
echo ""

# Check if USERNAME is still default
if [ "$USERNAME" == "Najongs" ]; then
    echo "⚠️  ERROR: Please set your Hugging Face username in the script!"
    echo "Edit line 5: USERNAME=\"your_actual_username\""
    exit 1
fi

# Find all task directories (directories containing .h5 files)
echo "🔍 Searching for task directories with .h5 files..."
TASK_DIRS=()
UPLOADED_REPOS=()

# Look for directories with episode_*.h5 files
for dir in "$BASE_DIR"/*/ ; do
    if [ -d "$dir" ]; then
        # Check if directory contains .h5 files
        if ls "$dir"/*.h5 1> /dev/null 2>&1; then
            TASK_DIRS+=("$dir")
            TASK_NAME=$(basename "$dir")
            echo "  ✓ Found task: $TASK_NAME ($(ls "$dir"/*.h5 | wc -l) episodes)"
        fi
    fi
done

# Also check New_dataset/collected_data if exists
COLLECTED_DATA="$BASE_DIR/New_dataset/collected_data"
if [ -d "$COLLECTED_DATA" ] && ls "$COLLECTED_DATA"/*.h5 1> /dev/null 2>&1; then
    TASK_DIRS+=("$COLLECTED_DATA")
    echo "  ✓ Found task: collected_data ($(ls "$COLLECTED_DATA"/*.h5 | wc -l) episodes)"
fi

if [ ${#TASK_DIRS[@]} -eq 0 ]; then
    echo ""
    echo "❌ No task directories with .h5 files found!"
    echo "Expected structure:"
    echo "  $BASE_DIR/task_name/*.h5"
    echo "  or"
    echo "  $BASE_DIR/New_dataset/collected_data/*.h5"
    exit 1
fi

echo ""
echo "📦 Found ${#TASK_DIRS[@]} task(s) to upload"
echo ""

# Upload each task dataset
for EPISODE_DIR in "${TASK_DIRS[@]}"; do
    TASK_NAME=$(basename "$EPISODE_DIR")
    # Sanitize task name for repo ID (lowercase, replace spaces/underscores with hyphens)
    REPO_SUFFIX=$(echo "$TASK_NAME" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | tr ' ' '-')
    REPO_ID="$USERNAME/vla-meca-${REPO_SUFFIX}"
    DATASET_NAME="VLA MECA500 - $TASK_NAME"
    OUTPUT_DIR="outputs/dataset_upload_${REPO_SUFFIX}"

    echo ""
    echo "=========================================="
    echo "Uploading: $TASK_NAME"
    echo "=========================================="
    echo "Source: $EPISODE_DIR"
    echo "Target: $REPO_ID"

    NUM_EPISODES=$(ls "$EPISODE_DIR"/*.h5 2>/dev/null | wc -l)
    echo "Episodes found: $NUM_EPISODES"

    if [ $NUM_EPISODES -eq 0 ]; then
        echo "⚠️  No .h5 files found, skipping..."
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
        echo "✅ $TASK_NAME uploaded successfully!"
        UPLOADED_REPOS+=("$REPO_ID")
    else
        echo "❌ $TASK_NAME upload failed!"
    fi

    echo ""
done

echo "=========================================="
echo "All uploads completed!"
echo "=========================================="
echo ""
if [ ${#UPLOADED_REPOS[@]} -gt 0 ]; then
    echo "✅ Successfully uploaded ${#UPLOADED_REPOS[@]} dataset(s):"
    echo ""
    echo "Datasets available at:"
    for REPO in "${UPLOADED_REPOS[@]}"; do
        echo "  🔗 https://huggingface.co/datasets/$REPO"
    done
else
    echo "❌ No datasets were uploaded successfully"
fi
echo "=========================================="
