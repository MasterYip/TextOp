#!/bin/bash

# Example: Quick test with small dataset
# This script demonstrates how to collect a small test dataset

echo "=========================================="
echo "G1 Dataset Collection - Quick Test"
echo "=========================================="

# Change to TextOpTracker directory
cd "$(dirname "$0")/../.." || exit

# Run with reduced parameters for quick testing
python scripts/data_collection/data_collection.py \
    --checkpoint logs/rsl_rl/tracking/exported/policy.pt \
    --motion_file Data10k-open \
    --output artifacts/test_dataset/motion.zarr \
    --num_envs 10 \
    --min_episode_length 100 \
    --len_to_save 5000 \
    --headless

echo ""
echo "=========================================="
echo "Verifying collected dataset..."
echo "=========================================="

# Verify the dataset
python scripts/data_collection/verify_dataset.py \
    artifacts/test_dataset/motion.zarr

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
