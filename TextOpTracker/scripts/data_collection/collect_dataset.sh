#!/bin/bash

# G1 Dataset Collection Script with Hydra Configuration
# This script collects training data from a trained TextOpTracker policy for diffuse_cloc
# Uses Hydra for configuration management
#
# Collection Modes:
#   - standard (default): Random sampling with quality filters
#   - deterministic: M motions × N samples = M*N episodes for full coverage

# Default Hydra overrides (can be overridden by command line)
HYDRA_OVERRIDES=""

# Parse command line arguments and convert to Hydra overrides
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint)
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES checkpoint.path=$2"
            shift 2
            ;;
        -m|--motion_file)
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES motion.pattern=$2"
            shift 2
            ;;
        -o|--output_dir)
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES output.dir=$2"
            shift 2
            ;;
        --zarr_name)
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES output.zarr_name=$2"
            shift 2
            ;;
        --num_envs)
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES task.num_envs=$2"
            shift 2
            ;;
        --mode)
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES collection.mode=$2"
            shift 2
            ;;
        --samples_per_motion)
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES collection.samples_per_motion=$2"
            shift 2
            ;;
        --min_episode_length)
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES collection.min_episode_length=$2"
            shift 2
            ;;
        --len_to_save)
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES collection.len_to_save=$2"
            shift 2
            ;;
        --task)
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES task.name=$2"
            shift 2
            ;;
        --no_noise)
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES noise.enable=false"
            shift
            ;;
        --noise_sigma)
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES noise.sigma=$2"
            shift 2
            ;;
        --headless)
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES visualization.headless=true"
            shift
            ;;
        --visualize)
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES visualization.headless=false"
            shift
            ;;
        --load_pickle_cfg)
            HYDRA_OVERRIDES="$HYDRA_OVERRIDES checkpoint.load_pickle_cfg=true"
            shift
            ;;
        --config)
            # Use custom config file
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Collection Modes:"
            echo "  --mode standard        Random sampling with quality filters (default)"
            echo "  --mode deterministic   M×N sampling for full coverage"
            echo ""
            echo "Options:"
            echo "  -c, --checkpoint PATH          Path to checkpoint"
            echo "  -m, --motion_file PATTERN      Motion file pattern"
            echo "  -o, --output_dir DIR           Output directory"
            echo "  --zarr_name NAME               Zarr filename"
            echo "  --num_envs N                   Number of environments"
            echo "  --samples_per_motion N         Samples per motion (deterministic mode)"
            echo "  --min_episode_length N         Minimum episode length"
            echo "  --len_to_save N                Total timesteps to save (standard mode)"
            echo "  --task NAME                    Task name"
            echo "  --no_noise                     Disable noise injection"
            echo "  --noise_sigma SIGMA            OU noise sigma parameter"
            echo "  --headless                     Run headless"
            echo "  --visualize                    Enable visualization"
            echo "  --load_pickle_cfg              Load config from pickle"
            echo "  --config FILE                  Custom config file"
            echo "  -h, --help                     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "G1 Dataset Collection (Hydra-based)"
echo "=========================================="

# Run data collection with Hydra
cd "$(dirname "$0")/../.." || exit

if [ -n "$CONFIG_FILE" ]; then
    # Use custom config file
    python scripts/data_collection/data_collection.py \
        --config-name "$CONFIG_FILE" \
        $HYDRA_OVERRIDES
else
    # Use default data_collection.yaml
    python scripts/data_collection/data_collection.py \
        $HYDRA_OVERRIDES
fi

echo "=========================================="
echo "Data collection complete!"
echo "=========================================="
