# Data Dyeing Modes

This document explains the two dyeing modes available in `data_dyeing.py` for encoding motion data with MotionCLIP.

## Overview

Data dyeing encodes collected motion trajectories into CLIP latent space using MotionCLIP. The latents are then stored alongside the original motion data for use in conditional diffusion policy training.

## Mode 1: Standard Dyeing (Default)

**When to use**: When you have arbitrary collected data without motion tracking, or when each episode contains unique motion patterns.

**How it works**:
1. Loads the collected zarr dataset
2. For each frame in the dataset:
   - Extracts a motion window centered at that frame
   - Encodes the window using MotionCLIP
   - Stores the resulting latent vector
3. Saves the dataset with added latent field

**Configuration**:
```yaml
input:
  zarr_path: "path/to/collected_data.zarr"
  dyeing_from_origin_motion: false  # Default mode
```

**Efficiency**: 
- For M motions × N samples = M×N episodes → Encodes M×N episodes
- Each frame gets independently encoded

## Mode 2: Origin Motion Dyeing (Efficient for Deterministic Collection)

**When to use**: When you used deterministic collection mode with motion tracking (M motions × N samples), and the original motion files are available.

**How it works**:
1. Loads the collected zarr dataset (for structure and motion_idx)
2. Loads original motion files from artifacts using glob pattern
3. For each unique motion:
   - Encodes the entire motion sequence once
   - Caches the latent sequence indexed by motion_idx
4. For each episode in dataset:
   - Reads its motion_idx
   - Retrieves cached latents for that motion
   - Handles length mismatches with linear interpolation
5. Saves the dataset with added latent field

**Configuration**:
```yaml
input:
  zarr_path: "path/to/collected_data.zarr"
  dyeing_from_origin_motion: true  # Enable origin motion mode
  motion_pattern: "selected_motions/*"  # Pattern to match motion folders
  motion_base_path: "../../artifacts"  # Base path for motion files
```

**Efficiency**:
- For M motions × N samples = M×N episodes → Encodes only M motions
- **Speedup**: ~N× faster when N is large (e.g., 50× for N=50)
- **Memory**: Requires storing M motion latent sequences in cache

## Requirements for Origin Motion Dyeing

1. **Motion Tracking**: Dataset must have been collected with `motion_idx` field
   - Enable in data_collection.yaml: `sort_by_motion_idx: true`
   - Automatically added when using deterministic collection mode

2. **Original Motion Files**: Motion npz files must be available
   - Located at: `{motion_base_path}/{motion_pattern}/motion.npz`
   - Example: `artifacts/selected_motions/motion_0/motion.npz`
   - Files should be in the same order as used during collection

3. **NPZ Format**: Each motion.npz should contain:
   - `body_pos_w`: [T, 30, 3] - body positions
   - `body_quat_w`: [T, 30, 4] - body quaternions  
   - `body_lin_vel_w`: [T, 30, 3] - linear velocities
   - `body_ang_vel_w`: [T, 30, 3] - angular velocities
   - `fps`: scalar - frames per second

## Length Mismatch Handling

When episode length doesn't match original motion length (due to early termination, different FPS, etc.):
- Automatically applies linear interpolation to align latent sequences
- Preserves semantic content while adapting to actual trajectory length
- Logs mismatches for verification

## Example Usage

### Standard Mode
```bash
cd TextOpTracker/scripts/data_dyeing
python data_dyeing.py \
    input.zarr_path=../../outputs/collected_data/dataset.zarr \
    output.zarr_path=../../outputs/dyed_data/dataset.zarr
```

### Origin Motion Mode
```bash
cd TextOpTracker/scripts/data_dyeing
python data_dyeing.py \
    input.zarr_path=../../outputs/collected_data/dataset.zarr \
    input.dyeing_from_origin_motion=true \
    input.motion_pattern="selected_motions/*" \
    input.motion_base_path=../../artifacts \
    output.zarr_path=../../outputs/dyed_data/dataset.zarr
```

## Performance Comparison

Example with 20 motions × 50 samples = 1000 episodes:

| Mode | Motions Encoded | Encoding Time | Memory Usage |
|------|----------------|---------------|--------------|
| Standard | 1000 episodes | ~50 minutes | Low |
| Origin | 20 motions | ~1 minute | Medium (20 motion caches) |

**Speedup**: 50× for this example

## Workflow Integration

### Complete Deterministic Collection + Dyeing Pipeline:

1. **Collect Data** (with motion tracking):
```bash
python data_collection.py \
    env.cfg.commands.motion_collection.deterministic_mode=true \
    env.cfg.commands.motion_collection.episodes_per_motion=50 \
    output.sort_by_motion_idx=true
```

2. **Dye Data** (from origin motions):
```bash
python data_dyeing.py \
    input.dyeing_from_origin_motion=true \
    input.motion_pattern="selected_motions/*"
```

3. **Train Policy**:
```bash
cd ../../diffuse_cloc
python train.py task.dataset.zarr_path=../TextOpTracker/outputs/dyed_data/dataset.zarr
```

## Troubleshooting

**"motion_idx not in cache"**: 
- Ensure motion files match the order used during collection
- Check that motion_pattern correctly matches all motion folders

**Length mismatches**:
- Expected when episodes terminate early or have FPS differences
- Automatically handled with interpolation
- Check logs to verify interpolation quality

**Memory errors**:
- Reduce batch_size in encoding config
- Process fewer motions at once by splitting dataset
