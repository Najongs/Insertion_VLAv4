#!/bin/bash
################################################################################
# Multi-GPU Training Script for SmolVLA on VLA Insertion Dataset
################################################################################

# Configuration
LEROBOT_PATH="/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src"
CONFIG_FILE="train_config.yaml"

# GPU Configuration
# Specify which GPUs to use (comma-separated, e.g., "0,1,2,3")
# Leave empty to use all available GPUs
export CUDA_VISIBLE_DEVICES="0,1,2,3,4"  # Change this to your GPU IDs

# Training parameters (optional overrides)
STEPS=20000              # Total training steps
BATCH_SIZE=8             # Batch size per GPU
LOG_FREQ=100             # Log frequency
SAVE_FREQ=2000           # Checkpoint save frequency

# Output directory
OUTPUT_DIR="outputs/train/smolvla_vla_insertion_multigpu"

################################################################################
# Script execution
################################################################################

echo "========================================="
echo "SmolVLA Multi-GPU Training"
echo "========================================="
echo ""
echo "Configuration:"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo "  Steps: $STEPS"
echo "  Batch size: $BATCH_SIZE (per GPU)"
echo "  Config: $CONFIG_FILE"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "========================================="
echo ""

# Count number of GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Number of GPUs: $NUM_GPUS"
echo ""

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
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
echo ""

# Run training with multi-GPU support
# Note: We use DataParallel which handles multi-GPU automatically
# when use_multi_gpu=true in config

PYTHONPATH=$LEROBOT_PATH python train_smolvla.py \
    --config $CONFIG_FILE \
    --steps $STEPS \
    --batch_size $BATCH_SIZE \
    --output_dir $OUTPUT_DIR \
    2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="
echo ""
echo "Logs saved to: training_*.log"
echo "Model saved to: $OUTPUT_DIR"
echo ""
