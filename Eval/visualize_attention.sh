#!/bin/bash

# Visualize Vision Encoder Attention Maps for SmolVLA Model
# This script extracts and visualizes attention maps from a trained model

# Set Python path
export PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH

# Configuration
CHECKPOINT="/home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_needle_insertion_lora/checkpoints/checkpoint_step_4000.pt"
EPISODE="/home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar/260107/1_MIN/episode_20260107_134411.h5"
STATS="/home/najo/NAS/VLA/Insertion_VLAv4/Train/dataset_stats.yaml"
OUTPUT_DIR="/home/najo/NAS/VLA/Insertion_VLAv4/Eval/outputs/attention_maps"
TASK_INSTRUCTION="Insert needle into eye trocar"

# Processing options
NUM_FRAMES=30        # Number of frames to process
FRAME_SKIP=3         # Process every 3rd frame to speed up
CAMERAS="camera1 camera2 camera3"  # Which cameras to visualize

echo "Starting Attention Map Visualization..."
echo "Checkpoint: $CHECKPOINT"
echo "Episode: $EPISODE"
echo "Output: $OUTPUT_DIR"
echo "Processing $NUM_FRAMES frames (every ${FRAME_SKIP}th frame)"
echo ""

python3 /home/najo/NAS/VLA/Insertion_VLAv4/Eval/visualize_attention_maps.py \
    --checkpoint "$CHECKPOINT" \
    --episode "$EPISODE" \
    --stats "$STATS" \
    --output_dir "$OUTPUT_DIR" \
    --task_instruction "$TASK_INSTRUCTION" \
    --num_frames $NUM_FRAMES \
    --frame_skip $FRAME_SKIP \
    --cameras $CAMERAS \
    --device cuda

echo ""
echo "Done! Check results in: $OUTPUT_DIR"
