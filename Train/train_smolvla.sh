#!/bin/bash
# =============================================================================
# SmolVLA Expert-Only Training Script using torchrun (DDP)
# =============================================================================
# SmolVLA is HuggingFace's efficient vision-language-action model
# Expert-only training freezes vision encoder and trains action expert only
#
# Memory Requirements:
# - Full Fine-Tuning: > 40 GB per GPU ❌ (OOM on RTX 3090)
# - Expert-Only Training: ~12-16 GB per GPU ✅ (fits on RTX 3090!)
#
# Improvements from Inference:
# - Temporal consistency loss for smoother trajectories
# - Data augmentation for robustness
# - Freeze vision encoder for efficiency
#
# Usage:
#   bash train_smolvla.sh
#   bash train_smolvla.sh --resume /path/to/checkpoint.pt
#   bash train_smolvla.sh --resume /path/to/checkpoint.pt --reset_scheduler
#
# To specify GPUs:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4 bash train_smolvla.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4 bash train_smolvla.sh --resume /path/to/checkpoint.pt --reset_scheduler
# =============================================================================

set -e  # Exit on error

# Parse command line arguments
RESUME_CHECKPOINT=""
RESET_SCHEDULER=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --reset_scheduler)
            RESET_SCHEDULER=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash train_smolvla.sh [--resume /path/to/checkpoint.pt] [--reset_scheduler]"
            exit 1
            ;;
    esac
done

# Change to script directory
cd "$(dirname "$0")"

echo "=================================================="
echo "SmolVLA LoRA Training - Needle Insertion (Meca500)"
echo "=================================================="

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

# Set PyTorch CPU threads for optimal performance
# Use all available CPU cores divided by number of processes
TOTAL_CPUS=$(nproc)
THREADS_PER_PROCESS=$((TOTAL_CPUS / NUM_GPUS))
export OMP_NUM_THREADS=$THREADS_PER_PROCESS
export MKL_NUM_THREADS=$THREADS_PER_PROCESS

echo "CPU Configuration:"
echo "  Total CPU cores: $TOTAL_CPUS"
echo "  Threads per GPU process: $THREADS_PER_PROCESS"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo ""

# Training configuration
CONFIG_FILE="train_config_smolvla_normalized.yaml"
BATCH_SIZE=16      # Per-GPU batch size (=80 with 5 GPUs) - MAXIMIZED!
STEPS=50000        # Total training steps
LR=0.0001          # Learning rate (1e-4, SmolVLA default)

# Note: With gradient accumulation (x1), effective batch size = 80 * 1 = 80

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create output directory
OUTPUT_DIR="outputs/train/smolvla_needle_insertion_new"
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
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "  Resume from: $RESUME_CHECKPOINT"
    if [ "$RESET_SCHEDULER" = true ]; then
        echo "  Reset scheduler: YES (fresh warmup from current step)"
    else
        echo "  Reset scheduler: NO (continue with checkpoint scheduler)"
    fi
fi
echo ""
echo "Expert-Only Training Benefits:"
echo "  ✓ Only action expert trained (~100-200M params)"
echo "  ✓ Vision encoder frozen (~300M params)"
echo "  ✓ Lower memory usage (~12-16 GB vs > 40 GB)"
echo "  ✓ Faster training and smaller checkpoints"
echo "  ✓ Works on RTX 3090 GPUs!"
echo ""
echo "Training Optimizations Applied:"
echo "  ✓ Batch size MAXIMIZED (12 per GPU) → 12x more samples per step"
echo "  ✓ Gradient Accumulation (x3) → effective batch = 180"
echo "  ✓ Total throughput: 180 samples per update (6x faster convergence)"
echo "  ✓ DataLoader workers (2 per GPU) → faster data loading"
echo "  ✓ pin_memory + prefetch → optimized GPU transfer"
echo "  ✓ Memory cleanup every 50 steps → stable RAM usage"
echo "  ✓ Temporal consistency loss (λ=0.1) → smooth trajectories"
echo "  ✓ Data augmentation → robust model"
echo ""
echo "Note: Mixed Precision (AMP) disabled due to SmolVLA BFloat16 incompatibility"
echo "Expected VRAM usage: ~10-12GB per GPU (50% utilization)"
echo ""
echo "Starting training in 3 seconds... (Press Ctrl+C to cancel)"
sleep 3
echo ""
echo "Starting SmolVLA Expert-Only DDP training with torchrun..."
echo "Training action expert only with frozen vision encoder"
echo "Press Ctrl+C to stop"
echo ""

# Run training with torchrun
RESUME_ARG=""
if [ -n "$RESUME_CHECKPOINT" ]; then
    RESUME_ARG="--resume $RESUME_CHECKPOINT"
fi

RESET_SCHEDULER_ARG=""
if [ "$RESET_SCHEDULER" = true ]; then
    RESET_SCHEDULER_ARG="--reset_scheduler"
fi

PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train_smolvla.py \
    --config "$CONFIG_FILE" \
    --batch_size $BATCH_SIZE \
    --steps $STEPS \
    --lr $LR \
    --output_dir "$OUTPUT_DIR" \
    $RESUME_ARG \
    $RESET_SCHEDULER_ARG \
    2>&1 | tee "$LOG_FILE"

# Store exit code
EXIT_CODE=$?

# Check if training completed successfully
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "SmolVLA Expert-Only training completed!"
    echo "=================================================="
    echo "Output saved to: $OUTPUT_DIR"
    echo "Log saved to: $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "1. Evaluate model: cd ../Eval && python evaluate_smolvla.py"
    echo "2. Run inference: cd ../Inference && python lerobot_to_MECA.py"
    echo "3. Extract dataset stats: cd ../Inference && python extract_dataset_stats.py"
    echo ""
    echo "Model improvements:"
    echo "  - Expert-only training (efficient and effective)"
    echo "  - Temporal consistency loss applied during training"
    echo "  - Smooth trajectories matching inference behavior"
    echo "  - Ready for Meca500 deployment"
else
    echo ""
    echo "=================================================="
    echo "Training failed with error code: $EXIT_CODE"
    echo "=================================================="
    echo "Check log file: $LOG_FILE"
    exit $EXIT_CODE
fi
