#!/bin/bash
# =============================================================================
# Multi-GPU Training Script for SmolVLA on New Dataset
# =============================================================================
# This script runs training on multiple GPUs using DataParallel
#
# Usage:
#   bash train_multi_gpu.sh
#
# To specify GPUs:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_multi_gpu.sh
# =============================================================================

set -e  # Exit on error

# Change to script directory
cd "$(dirname "$0")"

echo "============================================="
echo "SmolVLA Training - Multi-GPU"
echo "============================================="

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA is not available."
    exit 1
fi

echo ""
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Set GPUs to use (default: all available GPUs)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # Auto-detect number of GPUs
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ $NUM_GPUS -gt 1 ]; then
        # Use all GPUs
        CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
        export CUDA_VISIBLE_DEVICES
        echo "Using all available GPUs: $CUDA_VISIBLE_DEVICES"
    else
        echo "WARNING: Only 1 GPU detected. Multi-GPU training requires at least 2 GPUs."
        echo "Falling back to single GPU training..."
        export CUDA_VISIBLE_DEVICES=0
    fi
else
    echo "Using specified GPUs: $CUDA_VISIBLE_DEVICES"
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Number of GPUs: $NUM_GPUS"
echo ""

# Training configuration
CONFIG_FILE="train_config_new_dataset.yaml"
BATCH_SIZE=1   # MUST be 1 (larger batches cause tensor size errors)
STEPS=170830   # 10 epochs (17083 samples * 10) - VLM training needs more epochs
LR=0.0001

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create output directory
OUTPUT_DIR="outputs/train/smolvla_new_dataset_multigpu"
mkdir -p "$OUTPUT_DIR"

# Create logs directory
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

# Calculate effective batch size
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Per-GPU batch size: $BATCH_SIZE"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo "  Training steps: $STEPS"
echo "  Learning rate: $LR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Log file: $LOG_FILE"
echo ""

# Confirm before starting
read -p "Start training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "Starting multi-GPU training..."
echo "Press Ctrl+C to stop"
echo ""

# Create temporary config file with multi-GPU enabled
TEMP_CONFIG="/tmp/train_config_multigpu_${TIMESTAMP}.yaml"
cp "$CONFIG_FILE" "$TEMP_CONFIG"

# Update config to enable multi-GPU
if command -v yq &> /dev/null; then
    # If yq is available, use it to modify YAML
    yq eval '.policy.use_multi_gpu = true' -i "$TEMP_CONFIG"
    yq eval ".output_dir = \"$OUTPUT_DIR\"" -i "$TEMP_CONFIG"
else
    # Fallback: use sed (less reliable but works for simple cases)
    sed -i 's/use_multi_gpu: false/use_multi_gpu: true/' "$TEMP_CONFIG"
    sed -i "s|output_dir:.*|output_dir: \"$OUTPUT_DIR\"|" "$TEMP_CONFIG"
fi

echo "Multi-GPU config enabled in: $TEMP_CONFIG"
echo ""

# Run training with logging
python train_smolvla_new_dataset.py \
    --config "$TEMP_CONFIG" \
    --batch_size $BATCH_SIZE \
    --steps $STEPS \
    --lr $LR \
    2>&1 | tee "$LOG_FILE"

# Store exit code
EXIT_CODE=$?

# Clean up temporary config
rm -f "$TEMP_CONFIG"

# Check if training completed successfully
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================="
    echo "Training completed successfully!"
    echo "============================================="
    echo "Output saved to: $OUTPUT_DIR"
    echo "Log saved to: $LOG_FILE"
else
    echo ""
    echo "============================================="
    echo "Training failed with error code: $EXIT_CODE"
    echo "============================================="
    echo "Check log file: $LOG_FILE"
    exit $EXIT_CODE
fi
