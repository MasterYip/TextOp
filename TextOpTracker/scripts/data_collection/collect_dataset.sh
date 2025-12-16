#!/bin/bash

# G1 Dataset Collection Script
# This script collects training data from a trained TextOpTracker policy for diffuse_cloc

# Default values
CHECKPOINT="logs/rsl_rl/Pretrained/checkpoints/model_75000.pt"
MOTION_FILE="Data10k-open/homejrhangmr_dataset_pbhc_contact_maskACCADFemale1Walking_c3dB3-walk1_posespkl"
OUTPUT="artifacts/g1_tracking_dataset/motion.zarr"
NUM_ENVS=2048
MIN_EPISODE_LENGTH=500
LEN_TO_SAVE=500000
TASK="Tracking-Flat-G1-ProjGravObs-MNMLP-v0"
LOAD_PICKLE_CFG="--load_pickle_cfg"

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
        --load_pickle_cfg)
            LOAD_PICKLE_CFG="--load_pickle_cfg"
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

python scripts/data_collection/data_collection.py \
    --checkpoint "$CHECKPOINT" \
    --motion_file "$MOTION_FILE" \
    --output "$OUTPUT" \
    --num_envs "$NUM_ENVS" \
    --min_episode_length "$MIN_EPISODE_LENGTH" \
    --len_to_save "$LEN_TO_SAVE" \
    --task "$TASK" \
    $VISUALIZE \
    $HEADLESS \
    $LOAD_PICKLE_CFG

echo "=========================================="
echo "Data collection complete!"
echo "Dataset saved to: $OUTPUT"
echo "=========================================="
