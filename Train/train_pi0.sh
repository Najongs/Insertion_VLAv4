#!/bin/bash
# =============================================================================
# Multi-GPU Training Script for π0 (Pi Zero) using torchrun (DDP)
# =============================================================================
# π0 is Physical Intelligence's vision-language-action model
# Uses flow matching for action generation with language conditioning
#
# Usage:
#   bash train_pi0.sh
#
# To specify GPUs:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4 bash train_pi0.sh
# =============================================================================

set -e  # Exit on error

# Change to script directory
cd "$(dirname "$0")"

echo "============================================="
echo "π0 (Pi Zero) Training - Needle Insertion"
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
CONFIG_FILE="train_config_pi0.yaml"
BATCH_SIZE=8       # Per-GPU batch size (effective=40 with 5 GPUs)
STEPS=30000        # Total training steps (π0 typical)
LR=0.000025        # Learning rate (2.5e-5)

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create output directory
OUTPUT_DIR="outputs/train/pi0_needle_insertion"
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
echo "Starting training in 3 seconds... (Press Ctrl+C to cancel)"
sleep 3
echo ""
echo "Starting π0 DDP training with torchrun..."
echo "π0 uses flow matching for smooth action generation"
echo "Vision-language conditioning for better generalization"
echo "Press Ctrl+C to stop"
echo ""

# Run training with torchrun
PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train_pi0.py \
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
    echo "π0 training completed!"
    echo "============================================="
    echo "Output saved to: $OUTPUT_DIR"
    echo "Log saved to: $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "1. Evaluate model: cd ../Eval && python evaluate_pi0.py"
    echo "2. Run inference: cd ../Inference && python inference_pi0.py"
else
    echo ""
    echo "============================================="
    echo "Training failed with error code: $EXIT_CODE"
    echo "============================================="
    echo "Check log file: $LOG_FILE"
    exit $EXIT_CODE
fi
