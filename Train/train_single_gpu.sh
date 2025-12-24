#!/bin/bash
# =============================================================================
# Single GPU Training Script for SmolVLA on New Dataset
# =============================================================================
# This script runs training on a single GPU (GPU 0 by default)
#
# Usage:
#   bash train_single_gpu.sh
#
# To use a different GPU:
#   CUDA_VISIBLE_DEVICES=1 bash train_single_gpu.sh
# =============================================================================

set -e  # Exit on error

# Change to script directory
cd "$(dirname "$0")"

echo "============================================="
echo "SmolVLA Training - Single GPU"
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

# Set GPU to use (default: GPU 0)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "Using GPU: 0 (default)"
else
    echo "Using GPU: $CUDA_VISIBLE_DEVICES"
fi

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
OUTPUT_DIR="outputs/train/smolvla_new_dataset"
mkdir -p "$OUTPUT_DIR"

# Create logs directory
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Batch size: $BATCH_SIZE"
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
echo "Starting training..."
echo "Press Ctrl+C to stop"
echo ""

# Run training with logging
python train_smolvla_new_dataset.py \
    --config "$CONFIG_FILE" \
    --batch_size $BATCH_SIZE \
    --steps $STEPS \
    --lr $LR \
    2>&1 | tee "$LOG_FILE"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================="
    echo "Training completed successfully!"
    echo "============================================="
    echo "Output saved to: $OUTPUT_DIR"
    echo "Log saved to: $LOG_FILE"
else
    echo ""
    echo "============================================="
    echo "Training failed with error code: $?"
    echo "============================================="
    echo "Check log file: $LOG_FILE"
    exit 1
fi
