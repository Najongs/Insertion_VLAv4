#!/bin/bash
# Compute dataset statistics for simulation data
# Usage: bash compute_dataset_stats_sim.sh

echo "=================================================="
echo "Computing Dataset Statistics for Simulation Data"
echo "=================================================="
echo ""

# Set Python path
export PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH

# Simulation dataset path
DATASET_PATH="/home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar_sim"
OUTPUT_FILE="dataset_stats_sim.yaml"

echo "Dataset path: $DATASET_PATH"
echo "Output file: $OUTPUT_FILE"
echo ""

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset directory not found: $DATASET_PATH"
    exit 1
fi

# Count total episodes
TOTAL_EPISODES=$(find "$DATASET_PATH" -name "*.h5" | wc -l)
echo "Total simulation episodes found: $TOTAL_EPISODES"
echo ""

if [ $TOTAL_EPISODES -eq 0 ]; then
    echo "❌ Error: No .h5 files found in dataset directory"
    exit 1
fi

echo "Starting statistics computation..."
echo "This may take a few minutes..."
echo ""

# Run statistics computation
python3 compute_dataset_stats.py \
    --dataset_path "$DATASET_PATH" \
    --output "$OUTPUT_FILE" \
    --use_ee_pose true \
    --use_qpos false

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Statistics computed successfully!"
    echo "=================================================="
    echo "Output saved to: $OUTPUT_FILE"
    echo ""
    echo "You can now run training with:"
    echo "  bash train_smolvla_sim.sh"
else
    echo ""
    echo "❌ Error: Statistics computation failed"
    exit 1
fi
