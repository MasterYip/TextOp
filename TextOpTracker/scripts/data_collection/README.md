# G1 Dataset Collection for DiffuseCLOC Training

This directory contains scripts for collecting G1 robot training data from the TextOpTracker environment for use with DiffuseCLOC (diffuse_cloc).

## Overview

The data collection pipeline extracts robot state information from a trained tracking policy and saves it in zarr format compatible with the G1 offline dataset structure required by DiffuseCLOC.

## Data Format

The collected dataset follows the structure defined in `diffuse_cloc/diffusion_policy/dataset/offline_dataset.py` and `g1_offline_dataset.py`:

### Required Fields

- **act**: Joint position actions `[T, 29]`
- **body_ang_vel**: Body angular velocities `[T, 30, 3]`
- **body_lin_vel**: Body linear velocities `[T, 30, 3]`
- **body_pos**: Body positions `[T, 30, 3]`
- **body_rot**: Body rotations (quaternions) `[T, 30, 4]`
- **joint_pos**: Joint positions `[T, 29]`
- **joint_vel**: Joint velocities `[T, 29]`
- **root_pos**: Root (pelvis) position `[T, 3]`
- **root_rot**: Root rotation (quaternion) `[T, 4]`

Where:
- `T`: Number of timesteps
- 30 bodies: All bodies in the G1 robot (see `BODY_NAMES` in `offline_dataset.py`)
- 29 joints: All joints in the G1 robot (see `JOINT_NAMES` in `offline_dataset.py`)

## Scripts

### 1. `data_collection.py`

Main data collection script that:
- Loads a trained tracking policy
- Runs parallel environments
- Collects robot state data
- Applies quality filters
- Saves data in zarr format

### 2. `collect_dataset.sh`

Convenience bash script for running data collection with common parameters.

## Usage

### Basic Usage

```bash
# Using the shell script (recommended)
cd TextOpTracker
./scripts/data_collection/collect_dataset.sh \
    --checkpoint logs/rsl_rl/tracking/exported/policy.pt \
    --motion_file Data10k-open \
    --output artifacts/g1_tracking_dataset/motion.zarr
```

### Advanced Usage with Python Script

```bash
cd TextOpTracker

python scripts/data_collection/data_collection.py \
    --checkpoint logs/rsl_rl/tracking/exported/policy.pt \
    --motion_file Data10k-open \
    --output artifacts/g1_tracking_dataset/motion.zarr \
    --num_envs 100 \
    --min_episode_length 300 \
    --len_to_save 500000 \
    --headless
```

## Parameters

### Required Parameters

- `--checkpoint, -c`: Path to trained policy checkpoint (e.g., `logs/rsl_rl/tracking/exported/policy.pt`)
- `--motion_file, -m`: Motion file pattern in `artifacts/` directory (e.g., `Data10k-open`)
- `--output, -o`: Output path for zarr dataset (e.g., `artifacts/g1_tracking_dataset/motion.zarr`)

### Optional Parameters

- `--task, -t`: Task name (default: `Isaac-TextOp-Tracking-G1-Direct-v0`)
- `--num_envs`: Number of parallel environments (default: 100)
- `--min_episode_length`: Minimum episode length to keep for data quality (default: 300)
- `--min_mean_reward`: Minimum mean reward per episode to keep (alternative quality filter)
- `--len_to_save`: Total timesteps to save (default: 500000)
- `--max_episode_length`: Maximum steps per episode (default: 1000)
- `--chunk_length`: Chunk length for zarr file, -1 for auto (default: -1)
- `--seed`: Random seed (default: 42)
- `--headless`: Run without visualization
- `--visualize`: Enable visualization (useful for debugging)

## Quality Filtering

The data collection includes quality filtering to ensure high-quality training data:

1. **Episode Length Filter**: Only episodes longer than `min_episode_length` are saved
   - Default: 300 timesteps
   - Filters out episodes where the robot fails quickly

2. **Mean Reward Filter** (optional): Only episodes with mean reward above `min_mean_reward` are saved
   - Disabled by default
   - Use when you want to ensure successful tracking performance

These filters help ensure that only successful tracking demonstrations are included in the training dataset.

## Output

The script produces:

1. **Dataset file**: `{output}` (zarr format)
   - Contains all robot state data organized by episodes
   - Compressed using zstd for efficient storage

2. **Metadata file**: `{output_dir}/metadata.json`
   - Collection parameters
   - Statistics (episodes collected, saved, retention rate)
   - Creation timestamp

## Example Workflow

```bash
# 1. Train a tracking policy (see TextOpTracker documentation)
cd TextOpTracker
python scripts/rsl_rl/train.py --task Isaac-TextOp-Tracking-G1-Direct-v0

# 2. Collect dataset from trained policy
./scripts/data_collection/collect_dataset.sh \
    --checkpoint logs/rsl_rl/tracking_g1/exported/policy.pt \
    --motion_file Data10k-open \
    --output artifacts/g1_tracking_dataset/motion.zarr \
    --num_envs 100 \
    --min_episode_length 300 \
    --len_to_save 500000 \
    --headless

# 3. Use dataset for DiffuseCLOC training
cd ../diffuse_cloc
python train.py --config config_files/train_g1_dataset.yaml
```

## Data Statistics

After collection, check the metadata file for statistics:

```bash
cat artifacts/g1_tracking_dataset/metadata.json
```

Example output:
```json
{
  "task": "Isaac-TextOp-Tracking-G1-Direct-v0",
  "total_episodes_collected": 2000,
  "total_episodes_saved": 1800,
  "total_timesteps": 540000,
  "n_episodes": 1800,
  "episode_retention_rate": 90.0
}
```

## Troubleshooting

### Issue: No data being saved

- Check that `min_episode_length` is not too high
- Verify policy checkpoint path is correct
- Ensure motion files exist in `artifacts/{motion_file}/motion.npz`

### Issue: Low episode retention rate

- Decrease `min_episode_length`
- Remove or lower `min_mean_reward` filter
- Check policy performance

### Issue: Out of memory

- Decrease `num_envs`
- Decrease `len_to_save`
- Run data collection in multiple batches

## Integration with DiffuseCLOC

The collected dataset can be directly used with DiffuseCLOC training:

1. Update the dataset path in DiffuseCLOC config:
   ```yaml
   dataset:
     zarr_path: ../TextOpTracker/artifacts/g1_tracking_dataset/motion.zarr
     horizon: 16
     n_past_steps: 4
   ```

2. The dataset structure matches `G1_Dataset` or `G1_Dataset_EE` classes in `g1_offline_dataset.py`

3. Data will be automatically normalized using character-frame normalization during training

## NEW: Action Noise Injection (Following BeyondMimic)

### Overview

The data collection now supports **Ornstein-Uhlenbeck (OU) noise injection** to create more robust training data:

- **Purpose**: Perturb states during rollout to collect corrective actions
- **Benefit**: Improves policy robustness to disturbances and creates diverse training data
- **Implementation**: Temporally correlated OU noise instead of i.i.d. Gaussian

### OU Noise Formula

```
η_{t+1} = η_t + θ(μ - η_t)Δt + σ√Δt ε_t, where ε_t ~ N(0, I)
```

### Default Parameters (from BeyondMimic)

- `θ = 0.8`: Mean reversion rate
- `μ = 0.0`: Long-term mean
- `σ = 0.1`: Joint-wise noise scale
- `Δt = 1.0`: Time step

### Configuration

Noise parameters are configured in `data_collection.yaml`:

```yaml
noise:
  enable: true
  type: "ou"
  theta: 0.8
  mu: 0.0
  sigma: 0.1
  dt: 1.0
```

### Usage Examples

```bash
# Disable noise
python scripts/data_collection/data_collection.py noise.enable=false

# Adjust noise scale
python scripts/data_collection/data_collection.py noise.sigma=0.15

# Using shell script
bash scripts/data_collection/collect_dataset.sh --no_noise
bash scripts/data_collection/collect_dataset.sh --noise_sigma 0.15
```

## NEW: Hydra Configuration Management

The script now uses **Hydra** for flexible configuration instead of argparse:

### Benefits

- Hierarchical configuration with defaults and overrides
- Easy experiment tracking
- Type-safe configuration
- Composition of multiple config files

### Config Files

- **data_collection.yaml**: Main configuration file with all parameters
- **data_collection_no_noise.yaml**: Example variant without noise injection

### Override Syntax

```bash
# Single override
python scripts/data_collection/data_collection.py task.num_envs=4096

# Multiple overrides
python scripts/data_collection/data_collection.py \
    task.num_envs=4096 \
    noise.sigma=0.15 \
    collection.len_to_save=5000000

# Use custom config
python scripts/data_collection/data_collection.py \
    --config-name data_collection_no_noise
```

### Shell Script Integration

The `collect_dataset.sh` script automatically converts arguments to Hydra overrides:

```bash
bash scripts/data_collection/collect_dataset.sh \
    --checkpoint logs/rsl_rl/model.pt \
    --motion_file "Data10k-open" \
    --num_envs 2048 \
    --no_noise
```

## See Also

- `data_collection.yaml`: Main configuration file
- `data_collection_no_noise.yaml`: Example config without noise
- `diffuse_cloc/diffusion_policy/dataset/g1_offline_dataset.py`: Dataset loading and normalization
- `TextOpTracker/scripts/rsl_rl/play.py`: Policy evaluation and export

