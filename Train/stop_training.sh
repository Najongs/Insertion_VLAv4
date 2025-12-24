#!/bin/bash
# =============================================================================
# Stop Training Script
# =============================================================================
# This script stops a background training process
#
# Usage:
#   bash stop_training.sh
# =============================================================================

set -e  # Exit on error

# Change to script directory
cd "$(dirname "$0")"

echo "============================================="
echo "Stop Training"
echo "============================================="

# Look for PID files
PID_FILES=(
    "outputs/train/smolvla_new_dataset/train.pid"
    "outputs/train/smolvla_new_dataset_multigpu/train.pid"
)

FOUND_PID=false

for PID_FILE in "${PID_FILES[@]}"; do
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")

        # Check if process is running
        if ps -p $PID > /dev/null 2>&1; then
            echo "Found running training process:"
            echo "  PID: $PID"
            echo "  PID file: $PID_FILE"
            echo ""

            # Confirm before killing
            read -p "Stop this training process? (y/n) " -n 1 -r
            echo ""

            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "Stopping training (PID: $PID)..."
                kill $PID

                # Wait for process to end
                echo "Waiting for process to terminate..."
                for i in {1..10}; do
                    if ! ps -p $PID > /dev/null 2>&1; then
                        break
                    fi
                    sleep 1
                done

                # Force kill if still running
                if ps -p $PID > /dev/null 2>&1; then
                    echo "Process still running, force killing..."
                    kill -9 $PID
                    sleep 1
                fi

                # Check if process is stopped
                if ! ps -p $PID > /dev/null 2>&1; then
                    echo "Training stopped successfully"
                    rm -f "$PID_FILE"
                else
                    echo "ERROR: Failed to stop training process"
                    exit 1
                fi
            else
                echo "Training not stopped"
            fi

            FOUND_PID=true
        else
            echo "Found stale PID file: $PID_FILE"
            echo "  PID: $PID (not running)"
            rm -f "$PID_FILE"
            echo "  Removed stale PID file"
            echo ""
        fi
    fi
done

if [ "$FOUND_PID" = false ]; then
    echo "No running training process found"
    echo ""
    echo "To manually check for training processes:"
    echo "  ps aux | grep train_smolvla_new_dataset"
fi

echo ""
