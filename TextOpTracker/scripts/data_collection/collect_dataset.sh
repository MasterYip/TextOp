#!/bin/bash

# G1 Dataset Collection Script
# This script collects training data from a trained TextOpTracker policy for diffuse_cloc

# Default values
CHECKPOINT="logs/rsl_rl/tracking/exported/policy.pt"
MOTION_FILE="Data10k-open"
OUTPUT="artifacts/g1_tracking_dataset/motion.zarr"
NUM_ENVS=100
MIN_EPISODE_LENGTH=300
LEN_TO_SAVE=500000
TASK="Isaac-TextOp-Tracking-G1-Direct-v0"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -m|--motion_file)
            MOTION_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --min_episode_length)
            MIN_EPISODE_LENGTH="$2"
            shift 2
            ;;
        --len_to_save)
            LEN_TO_SAVE="$2"
            shift 2
            ;;
        --visualize)
            VISUALIZE="--visualize"
            shift
            ;;
        --headless)
            HEADLESS="--headless"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "G1 Dataset Collection"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Motion file: $MOTION_FILE"
echo "Output: $OUTPUT"
echo "Number of environments: $NUM_ENVS"
echo "Minimum episode length: $MIN_EPISODE_LENGTH"
echo "Target timesteps: $LEN_TO_SAVE"
echo "=========================================="

# Run data collection
cd "$(dirname "$0")/../.." || exit

python -m TextOpTracker.scripts.data_collection.data_collection \
    --checkpoint "$CHECKPOINT" \
    --motion_file "$MOTION_FILE" \
    --output "$OUTPUT" \
    --num_envs "$NUM_ENVS" \
    --min_episode_length "$MIN_EPISODE_LENGTH" \
    --len_to_save "$LEN_TO_SAVE" \
    --task "$TASK" \
    $VISUALIZE \
    $HEADLESS

echo "=========================================="
echo "Data collection complete!"
echo "Dataset saved to: $OUTPUT"
echo "=========================================="
