#!/bin/bash
# Evaluation runner script for SmolVLA checkpoint

# Set paths
CHECKPOINT_PATH="/home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_vla_insertion_multigpu/checkpoints/checkpoint_step_0016000.pt"
CONFIG_PATH="eval_config.yaml"
OUTPUT_DIR="outputs/eval_step_16000"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found at $CONFIG_PATH"
    exit 1
fi

echo "=========================================="
echo "SmolVLA Checkpoint Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Add lerobot to PYTHONPATH
export PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH

# Run evaluation
python evaluate_smolvla.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --config "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Evaluation failed!"
    echo "=========================================="
    exit 1
fi
