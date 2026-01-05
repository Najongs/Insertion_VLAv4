#!/bin/bash
# =============================================================================
# Multi-GPU Training Script for SmolVLA using torchrun (DDP)
# =============================================================================
# This script runs distributed training on multiple GPUs using
# DistributedDataParallel (DDP) via torchrun
#
# Usage:
#   bash train_ddp.sh
#
# To specify GPUs:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4 bash train_ddp.sh
# =============================================================================

set -e  # Exit on error

# Change to script directory
cd "$(dirname "$0")"

echo "============================================="
echo "SmolVLA Training - DDP (torchrun)"
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

# Set GPUs to use (force all 5 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4
export CUDA_VISIBLE_DEVICES

# Count number of GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $NUM_GPUS"
echo ""

# Training configuration
CONFIG_FILE="train_config_new_dataset.yaml"
BATCH_SIZE=64   # Per-GPU batch size (effective=40 with 5 GPUs)
STEPS=25000    # Total training steps (reduced for faster testing)
LR=0.0001

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create output directory
OUTPUT_DIR="outputs/train/smolvla_new_dataset_ddp"
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
echo "Starting DDP training with torchrun..."
echo "Press Ctrl+C to stop"
echo ""

# Run training with torchrun
PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train_smolvla_new_dataset.py \
    --config "$CONFIG_FILE" \
    --batch_size $BATCH_SIZE \
    --steps $STEPS \
    --lr $LR \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

# Store exit code
EXIT_CODE=$?

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
