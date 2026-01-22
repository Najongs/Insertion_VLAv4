#!/bin/bash
# Training script for SmolVLA on Simulation Data
# Usage: bash train_smolvla_sim.sh

# Exit on error
set -e

echo "=================================================="
echo "SmolVLA Simulation Training"
echo "=================================================="
echo ""

# Set environment variables
export PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# Configuration
CONFIG_FILE="train_config_smolvla_sim.yaml"
NUM_GPUS=5

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  GPUs: $NUM_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  Dataset: Eye_trocar_sim (352 episodes)"
echo "  Domain Randomization: Applied during data collection"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if dataset stats file exists
if [ ! -f "dataset_stats_sim.yaml" ]; then
    echo "⚠️  Warning: dataset_stats_sim.yaml not found!"
    echo "    Please run compute_dataset_stats.py on simulation data first:"
    echo "    python compute_dataset_stats.py --dataset_path /home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar_sim --output dataset_stats_sim.yaml"
    echo ""
    read -p "Continue without normalization? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Starting training..."
echo ""

# Run training with torchrun (DDP)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    train_smolvla_sim.py \
    --config $CONFIG_FILE

echo ""
echo "=================================================="
echo "Training completed!"
echo "=================================================="
