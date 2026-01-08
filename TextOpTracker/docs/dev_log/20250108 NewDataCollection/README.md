# G1 Dataset Collection for DiffuseCLOC Training

This directory contains scripts for collecting G1 robot training data from the TextOpTracker environment for use with DiffuseCLOC (diffuse_cloc).

## Overview

The data collection pipeline extracts robot state information from a trained tracking policy and saves it in zarr format compatible with the G1 offline dataset structure required by DiffuseCLOC.

## Collection Modes

### Standard Mode (Default)

Random sampling with quality filters - the original behavior:
- Continuously samples episodes from random motions
- Applies quality filters (minimum length, minimum reward)
- Collects until target timestep count reached
- Good for diverse training data with best-performing demonstrations

### Deterministic Mode (NEW)

Full coverage M×N sampling strategy:
- **M motions** × **N samples per motion** = **M×N total episodes**
- Each motion sampled exactly N times from start to finish
- Failed episodes (terminated early) are **reassigned** and don't count as completed
- No random start position - always starts from beginning
- Ensures complete action distribution coverage

**Use cases:**
- Comprehensive dataset with full motion coverage
- Imitation learning requiring complete trajectories
- Balanced representation of all motions

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
- Applies quality filters (standard mode) or deterministic assignment (deterministic mode)
- Saves data in zarr format

### 2. `collect_dataset.sh`

Convenience bash script for running data collection with common parameters.

### 3. `commands_collection.py`

New command term specifically for deterministic M×N collection:
- Assigns each environment to a (motion_id, sample_id) task
- Tracks completion status
- Resets idle environments when all tasks done

## Usage

### Standard Mode (Random Sampling)

```bash
# Using the shell script
cd TextOpTracker
./scripts/data_collection/collect_dataset.sh \
    --checkpoint logs/rsl_rl/tracking/exported/policy.pt \
    --motion_file "Data10k-open/*" \
    --output_dir artifacts/g1_tracking_dataset \
    --mode standard \
    --len_to_save 1000000 \
    --min_episode_length 300 \
    --headless
```

### Deterministic Mode (M×N Sampling)

```bash
# Collect 100 motions × 5 samples = 500 episodes
cd TextOpTracker
./scripts/data_collection/collect_dataset.sh \
    --checkpoint logs/rsl_rl/tracking/exported/policy.pt \
    --motion_file "Data100-subset/*" \
    --output_dir artifacts/g1_deterministic_dataset \
    --mode deterministic \
    --samples_per_motion 5 \
    --num_envs 500 \
    --headless
```

**Important for deterministic mode:**
- Set `num_envs >= M × N` for best performance
- Each env gets assigned one task
- Progress tracked by completed tasks, not timesteps

### Advanced Usage with Python Script

```bash
cd TextOpTracker

python scripts/data_collection/data_collection.py \
    checkpoint.path=logs/rsl_rl/tracking/exported/policy.pt \
    motion.pattern="Data10k-open/*" \
    output.dir=artifacts/g1_tracking_dataset \
    collection.mode=deterministic \
    collection.samples_per_motion=3 \
    task.num_envs=300 \
    visualization.headless=true
```

## Parameters

### Core Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--checkpoint, -c` | path | Path to trained policy checkpoint | Required |
| `--motion_file, -m` | pattern | Motion file pattern in artifacts/ | Required |
| `--output_dir, -o` | path | Output directory for zarr dataset | Required |
| `--mode` | string | Collection mode: `standard` or `deterministic` | `standard` |

### Standard Mode Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--len_to_save` | int | Total timesteps to collect | 1000000 |
| `--min_episode_length` | int | Minimum episode length to keep | 300 |
| `--min_mean_reward` | float | Minimum mean reward per episode | None |

### Deterministic Mode Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--samples_per_motion` | int | N: number of samples per motion | 5 |
| `--num_envs` | int | Number of parallel environments | 1536 |

### Common Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--task, -t` | string | Task name | Auto-detect |
| `--zarr_name` | string | Zarr filename | motion.zarr |
| `--no_noise` | flag | Disable action noise injection | False |
| `--noise_sigma` | float | OU noise sigma parameter | 0.1 |
| `--headless` | flag | Run without visualization | False |
| `--seed` | int | Random seed | 42 |

## Quality Filtering (Standard Mode)

The standard mode includes quality filtering to ensure high-quality training data:

1. **Episode Length Filter**: Only episodes longer than `min_episode_length` are saved
   - Default: 300 timesteps
   - Filters out episodes where the robot fails quickly

2. **Mean Reward Filter** (optional): Only episodes with mean reward above `min_mean_reward` are saved
   - Disabled by default
   - Use when you want to ensure successful tracking performance

These filters help ensure that only successful tracking demonstrations are included in the training dataset.

## Deterministic Mode Details

### Task Assignment

```
M motions × N samples_per_motion = M×N tasks

Environment assignment:
- Env 0 → (Motion 0, Sample 0)
- Env 1 → (Motion 0, Sample 1)
- ...
- Env N-1 → (Motion 0, Sample N-1)
- Env N → (Motion 1, Sample 0)
- ...
```

### Completion Tracking

- Each task tracked as completed/incomplete
- Failed episode (terminated early) → **reassign same task**
- Successful episode (full motion) → **mark complete, assign next task**
- All tasks done → idle environments reset to default pose

### Output Guarantees

- Exactly M×N episodes collected (if all motions can be tracked)
- Each motion represented N times
- Full trajectories from start to end
- Balanced coverage across all motions

## Output

The script produces:

1. **Dataset file**: `{output_dir}/{zarr_name}` (zarr format)
   - Contains all robot state data organized by episodes
   - Compressed using zstd for efficient storage

2. **Metadata file**: `{output_dir}/metadata.json`
   - Collection parameters
   - Collection mode and mode-specific settings
   - Episode statistics
   - Creation timestamp

## Example Workflows

### Workflow 1: Standard Collection (Best Quality)

```bash
# Collect high-quality demonstrations with filters
./scripts/data_collection/collect_dataset.sh \
    -c logs/rsl_rl/tracking_g1/exported/policy.pt \
    -m "Data10k-open/*" \
    -o artifacts/g1_dataset_quality \
    --mode standard \
    --len_to_save 500000 \
    --min_episode_length 400 \
    --headless
```

### Workflow 2: Deterministic Collection (Full Coverage)

```bash
# Collect all 100 motions, 10 samples each = 1000 episodes
./scripts/data_collection/collect_dataset.sh \
    -c logs/rsl_rl/tracking_g1/exported/policy.pt \
    -m "Data100/*" \
    -o artifacts/g1_dataset_deterministic \
    --mode deterministic \
    --samples_per_motion 10 \
    --num_envs 1000 \
    --headless
```

### Workflow 3: Small Test Dataset

```bash
# Quick test: 10 motions × 2 samples = 20 episodes
./scripts/data_collection/collect_dataset.sh \
    -c logs/rsl_rl/tracking_g1/exported/policy.pt \
    -m "Data10-test/*" \
    -o artifacts/g1_dataset_test \
    --mode deterministic \
    --samples_per_motion 2 \
    --num_envs 20 \
    --visualize  # Enable visualization for debugging
```

## Data Statistics

After collection, check the metadata file for statistics:

```bash
cat artifacts/g1_tracking_dataset/metadata.json
```

### Standard Mode Metadata Example

```json
{
  "task": "Isaac-TextOp-Tracking-G1-Direct-v0",
  "collection_mode": "standard",
  "total_episodes_collected": 1200,
  "total_episodes_saved": 1080,
  "episode_retention_rate": 90.0,
  "total_timesteps": 502341,
  "min_episode_length": 300
}
```

### Deterministic Mode Metadata Example

```json
{
  "task": "Isaac-TextOp-Tracking-G1-Direct-v0",
  "collection_mode": "deterministic",
  "samples_per_motion": 5,
  "total_episodes_collected": 520,
  "total_episodes_saved": 500,
  "total_timesteps": 487234,
  "n_episodes": 500
}
```

## Troubleshooting

### Issue: No data being saved (Standard Mode)

- Check that `min_episode_length` is not too high
- Verify policy checkpoint path is correct
- Ensure motion files exist in `artifacts/{motion_file}/motion.npz`

### Issue: Collection stuck (Deterministic Mode)

- Some motions may be too difficult for the policy
- Check console for completion progress
- Reduce `samples_per_motion` for difficult motion sets
- Ensure `num_envs` >= M × N for optimal performance

### Issue: Low episode retention rate (Standard Mode)

- Decrease `min_episode_length`
- Remove or lower `min_mean_reward` filter
- Check policy performance

### Issue: Out of memory

- Decrease `num_envs`
- Decrease `samples_per_motion` (deterministic mode)
- Decrease `len_to_save` (standard mode)
- Run data collection in multiple batches

## Comparison: Standard vs Deterministic

| Aspect | Standard Mode | Deterministic Mode |
|--------|---------------|-------------------|
| **Sampling** | Random | Sequential M×N |
| **Starting Point** | Random in motion | Always frame 0 |
| **Target** | Timestep count | Episode count |
| **Failure Handling** | Filter out | Reassign task |
| **Coverage** | Biased to easy motions | Uniform coverage |
| **Best For** | Diverse, high-quality data | Complete representation |
| **Speed** | Variable | Predictable |

## Advanced Configuration

Create custom YAML configs for repeated use:

```yaml
# my_collection.yaml
defaults:
  - data_collection

collection:
  mode: deterministic
  samples_per_motion: 10

task:
  num_envs: 2000

motion:
  pattern: "MyCustomData/*"

noise:
  sigma: 0.15  # Higher noise for more diversity
```

Run with:
```bash
python scripts/data_collection/data_collection.py --config-name my_collection
```
