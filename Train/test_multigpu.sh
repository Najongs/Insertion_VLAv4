#!/bin/bash
# =============================================================================
# Multi-GPU Test Script for All Models
# =============================================================================
# Tests ACT, Diffusion, and SmolVLA with multi-GPU setup
# Quick validation (100 steps each)
#
# Usage:
#   bash test_multigpu.sh
# =============================================================================

set -e  # Exit on error

cd "$(dirname "$0")"

echo "============================================="
echo "Multi-GPU Model Testing"
echo "============================================="
echo ""

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA is not available."
    exit 1
fi

echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Set GPUs to use
CUDA_VISIBLE_DEVICES=0,1,2,3,4
export CUDA_VISIBLE_DEVICES

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $NUM_GPUS"
echo ""

# Test parameters
TEST_STEPS=100
TEST_BATCH_SIZE=4

# Create test output directory
TEST_OUTPUT_DIR="outputs/test_multigpu"
mkdir -p "$TEST_OUTPUT_DIR"

echo "============================================="
echo "Test 1: ACT Multi-GPU"
echo "============================================="
echo ""

if [ -f "train_config_act.yaml" ]; then
    echo "Testing ACT with $NUM_GPUS GPUs, $TEST_STEPS steps..."

    PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH \
    timeout 300 torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        train_act.py \
        --config train_config_act.yaml \
        --batch_size $TEST_BATCH_SIZE \
        --steps $TEST_STEPS \
        --output_dir "$TEST_OUTPUT_DIR/act" \
        2>&1 | tee "$TEST_OUTPUT_DIR/test_act.log"

    ACT_EXIT_CODE=$?

    if [ $ACT_EXIT_CODE -eq 0 ] || [ $ACT_EXIT_CODE -eq 124 ]; then
        echo "✅ ACT multi-GPU test PASSED"
        echo ""
    else
        echo "❌ ACT multi-GPU test FAILED (exit code: $ACT_EXIT_CODE)"
        echo "Check log: $TEST_OUTPUT_DIR/test_act.log"
        echo ""
    fi
else
    echo "⚠️ ACT config not found, skipping..."
    ACT_EXIT_CODE=0
    echo ""
fi

echo "============================================="
echo "Test 2: Diffusion Multi-GPU"
echo "============================================="
echo ""

if [ -f "train_config_diffusion.yaml" ]; then
    echo "Testing Diffusion with $NUM_GPUS GPUs, $TEST_STEPS steps..."

    PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH \
    timeout 300 torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        train_diffusion.py \
        --config train_config_diffusion.yaml \
        --batch_size $TEST_BATCH_SIZE \
        --steps $TEST_STEPS \
        --output_dir "$TEST_OUTPUT_DIR/diffusion" \
        2>&1 | tee "$TEST_OUTPUT_DIR/test_diffusion.log"

    DIFFUSION_EXIT_CODE=$?

    if [ $DIFFUSION_EXIT_CODE -eq 0 ] || [ $DIFFUSION_EXIT_CODE -eq 124 ]; then
        echo "✅ Diffusion multi-GPU test PASSED"
        echo ""
    else
        echo "❌ Diffusion multi-GPU test FAILED (exit code: $DIFFUSION_EXIT_CODE)"
        echo "Check log: $TEST_OUTPUT_DIR/test_diffusion.log"
        echo ""
    fi
else
    echo "⚠️ Diffusion config not found, skipping..."
    DIFFUSION_EXIT_CODE=0
    echo ""
fi

echo "============================================="
echo "Test 3: SmolVLA Multi-GPU"
echo "============================================="
echo ""

if [ -f "train_config_new_dataset.yaml" ]; then
    echo "Testing SmolVLA with $NUM_GPUS GPUs, $TEST_STEPS steps..."

    PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH \
    timeout 300 torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS \
        train_smolvla_new_dataset.py \
        --config train_config_new_dataset.yaml \
        --batch_size $TEST_BATCH_SIZE \
        --steps $TEST_STEPS \
        --output_dir "$TEST_OUTPUT_DIR/smolvla" \
        2>&1 | tee "$TEST_OUTPUT_DIR/test_smolvla.log"

    SMOLVLA_EXIT_CODE=$?

    if [ $SMOLVLA_EXIT_CODE -eq 0 ] || [ $SMOLVLA_EXIT_CODE -eq 124 ]; then
        echo "✅ SmolVLA multi-GPU test PASSED"
        echo ""
    else
        echo "❌ SmolVLA multi-GPU test FAILED (exit code: $SMOLVLA_EXIT_CODE)"
        echo "Check log: $TEST_OUTPUT_DIR/test_smolvla.log"
        echo ""
    fi
else
    echo "⚠️ SmolVLA config not found, skipping..."
    SMOLVLA_EXIT_CODE=0
    echo ""
fi

echo "============================================="
echo "Test Summary"
echo "============================================="
echo ""

TOTAL_TESTS=0
PASSED_TESTS=0

if [ -f "train_config_act.yaml" ]; then
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if [ $ACT_EXIT_CODE -eq 0 ] || [ $ACT_EXIT_CODE -eq 124 ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo "✅ ACT: PASSED"
    else
        echo "❌ ACT: FAILED"
    fi
fi

if [ -f "train_config_diffusion.yaml" ]; then
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if [ $DIFFUSION_EXIT_CODE -eq 0 ] || [ $DIFFUSION_EXIT_CODE -eq 124 ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo "✅ Diffusion: PASSED"
    else
        echo "❌ Diffusion: FAILED"
    fi
fi

if [ -f "train_config_new_dataset.yaml" ]; then
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if [ $SMOLVLA_EXIT_CODE -eq 0 ] || [ $SMOLVLA_EXIT_CODE -eq 124 ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo "✅ SmolVLA: PASSED"
    else
        echo "❌ SmolVLA: FAILED"
    fi
fi

echo ""
echo "Tests passed: $PASSED_TESTS / $TOTAL_TESTS"
echo ""

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    echo "🎉 All tests passed! Multi-GPU setup is working."
    exit 0
else
    echo "⚠️ Some tests failed. Check logs in: $TEST_OUTPUT_DIR"
    exit 1
fi
