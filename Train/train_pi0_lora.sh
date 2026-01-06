#!/bin/bash
# =============================================================================
# LoRA Training Script for π0 (Pi Zero) using torchrun (DDP)
# =============================================================================
# π0 is Physical Intelligence's vision-language-action model
# LoRA (Low-Rank Adaptation) enables parameter-efficient fine-tuning
#
# Memory Requirements:
# - Full Fine-Tuning: > 70 GB per GPU ❌ (OOM on RTX 3090)
# - LoRA Fine-Tuning: > 22.5 GB per GPU ✅ (fits on RTX 3090)
#
# Usage:
#   bash train_pi0_lora.sh
#
# To specify GPUs:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4 bash train_pi0_lora.sh
# =============================================================================

set -e  # Exit on error

# Change to script directory
cd "$(dirname "$0")"

echo "============================================="
echo "π0 (Pi Zero) LoRA Training - Needle Insertion"
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
CONFIG_FILE="train_config_pi0_lora.yaml"
BATCH_SIZE=4       # Per-GPU batch size (effective=20 with 5 GPUs)
STEPS=30000        # Total training steps (π0 typical)
LR=0.000025        # Learning rate (2.5e-5)

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create output directory
OUTPUT_DIR="outputs/train/pi0_lora_needle_insertion"
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
echo "LoRA Benefits:"
echo "  ✓ Only ~2-5% of parameters trained"
echo "  ✓ Much lower memory usage (> 22.5 GB vs > 70 GB)"
echo "  ✓ Faster training and smaller checkpoints"
echo "  ✓ Works on RTX 3090 GPUs!"
echo ""
echo "Starting training in 3 seconds... (Press Ctrl+C to cancel)"
sleep 3
echo ""
echo "Starting π0 LoRA DDP training with torchrun..."
echo "Parameter-efficient fine-tuning with adapter modules"
echo "Press Ctrl+C to stop"
echo ""

# Run training with torchrun
PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train_pi0_lora.py \
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
    echo "π0 LoRA training completed!"
    echo "============================================="
    echo "Output saved to: $OUTPUT_DIR"
    echo "Log saved to: $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "1. Merge LoRA adapters: python merge_lora_adapters.py"
    echo "2. Evaluate model: cd ../Eval && python evaluate_pi0.py"
    echo "3. Run inference: cd ../Inference && python inference_pi0.py"
else
    echo ""
    echo "============================================="
    echo "Training failed with error code: $EXIT_CODE"
    echo "============================================="
    echo "Check log file: $LOG_FILE"
    exit $EXIT_CODE
fi
