#!/bin/bash
# =============================================================================
# Background Training Script for SmolVLA on New Dataset
# =============================================================================
# This script runs training in the background with nohup
# Useful for long training runs that continue after logout
#
# Usage:
#   bash train_background.sh [single|multi]
#
# Examples:
#   bash train_background.sh single  # Single GPU background training
#   bash train_background.sh multi   # Multi-GPU background training
# =============================================================================

set -e  # Exit on error

# Change to script directory
cd "$(dirname "$0")"

# Default to single GPU
MODE=${1:-single}

echo "============================================="
echo "SmolVLA Background Training - $MODE GPU"
echo "============================================="

# Validate mode
if [[ "$MODE" != "single" && "$MODE" != "multi" ]]; then
    echo "ERROR: Invalid mode '$MODE'. Use 'single' or 'multi'"
    echo "Usage: bash train_background.sh [single|multi]"
    exit 1
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA is not available."
    exit 1
fi

echo ""
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Set up GPU configuration based on mode
if [ "$MODE" == "multi" ]; then
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        if [ $NUM_GPUS -gt 1 ]; then
            CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        else
            echo "WARNING: Only 1 GPU detected. Using single GPU mode."
            MODE="single"
            CUDA_VISIBLE_DEVICES=0
        fi
    fi
else
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        CUDA_VISIBLE_DEVICES=0
    fi
fi

export CUDA_VISIBLE_DEVICES
echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"
echo ""

# Training configuration
CONFIG_FILE="train_config_new_dataset.yaml"
BATCH_SIZE=1  # MUST be 1 (larger batches cause tensor size errors)
STEPS=82350   # 10 epochs (8235 samples * 10)
LR=0.0001

# Output directory
if [ "$MODE" == "multi" ]; then
    OUTPUT_DIR="outputs/train/smolvla_new_dataset_multigpu"
else
    OUTPUT_DIR="outputs/train/smolvla_new_dataset"
fi
mkdir -p "$OUTPUT_DIR"

# Create logs directory
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_bg_${TIMESTAMP}.log"
PID_FILE="$OUTPUT_DIR/train.pid"

echo "Configuration:"
echo "  Mode: $MODE GPU"
echo "  Config file: $CONFIG_FILE"
echo "  Batch size: $BATCH_SIZE"
echo "  Training steps: $STEPS"
echo "  Learning rate: $LR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Log file: $LOG_FILE"
echo "  PID file: $PID_FILE"
echo ""

# Confirm before starting
read -p "Start background training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Create temporary config for multi-GPU if needed
if [ "$MODE" == "multi" ]; then
    TEMP_CONFIG="/tmp/train_config_multigpu_${TIMESTAMP}.yaml"
    cp "$CONFIG_FILE" "$TEMP_CONFIG"

    # Update config to enable multi-GPU
    if command -v yq &> /dev/null; then
        yq eval '.policy.use_multi_gpu = true' -i "$TEMP_CONFIG"
        yq eval ".output_dir = \"$OUTPUT_DIR\"" -i "$TEMP_CONFIG"
    else
        sed -i 's/use_multi_gpu: false/use_multi_gpu: true/' "$TEMP_CONFIG"
        sed -i "s|output_dir:.*|output_dir: \"$OUTPUT_DIR\"|" "$TEMP_CONFIG"
    fi

    USE_CONFIG="$TEMP_CONFIG"
else
    USE_CONFIG="$CONFIG_FILE"
fi

echo ""
echo "Starting background training..."
echo ""

# Run training in background with nohup
nohup python -u train_smolvla_new_dataset.py \
    --config "$USE_CONFIG" \
    --batch_size $BATCH_SIZE \
    --steps $STEPS \
    --lr $LR \
    > "$LOG_FILE" 2>&1 &

# Save PID
TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"

echo "============================================="
echo "Training started in background!"
echo "============================================="
echo "PID: $TRAIN_PID"
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"
echo ""
echo "To monitor training:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check if training is running:"
echo "  ps aux | grep $TRAIN_PID"
echo ""
echo "To stop training:"
echo "  kill $TRAIN_PID"
echo "  # or: bash stop_training.sh"
echo ""
echo "To check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
