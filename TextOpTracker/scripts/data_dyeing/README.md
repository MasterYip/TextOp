# Data Dyeing for G1 Motion Dataset

This directory contains scripts for encoding collected G1 motion data into CLIP latent space using MotionCLIP.

## Overview

The data dyeing process:
1. Loads motion data collected from IsaacLab (zarr format)
2. Converts motion format to MotionCLIP input format
3. Encodes each frame's motion by sampling a window around it
4. Adds motion latent vectors (512-dim) to the dataset
5. Saves the enhanced dataset back to zarr

## Files

- `data_dyeing.py` - Main script for encoding motions
- `data_dyeing.yaml` - Configuration file
- `motion_converter.py` - Motion format conversion utilities
- `README.md` - This file

## Usage

### Basic Usage

```bash
python data_dyeing.py \
  input.zarr_path=../../artifacts/g1_152_obsnoise/motion.zarr \
  output.zarr_path=../../artifacts/g1_152_obsnoise/motion_dyed_text_t0.01.zarr \
  motionclip.checkpoint_path=../../../MotionCLIP/exps/g1-model-xyz-clip/checkpoint_0100.pth.tar \
  encoding.use_text_alignment=true
```

### Configuration Options

Edit `data_dyeing.yaml` or override via command line:

**Input/Output:**
- `input.zarr_path`: Path to input zarr dataset
- `output.zarr_path`: Path to output zarr dataset
- `output.latent_field_name`: Name of latent field (default: "motion_latent")
- `output.overwrite`: Whether to overwrite existing latent field

**MotionCLIP:**
- `motionclip.checkpoint_path`: Path to trained MotionCLIP checkpoint
- `motionclip.device`: Device to use ('cuda' or 'cpu')

**Encoding:**
- `encoding.window_size`: Window size for motion sequences (default: 60)
- `encoding.batch_size`: Batch size for processing (default: 128)
- `encoding.boundary_mode`: How to handle boundaries ('pad', 'truncate', 'mirror')
- `encoding.pose_rep`: Pose representation ('xyz', 'rot6d', 'posquat', 'posvel')

## Motion Format Conversion

The converter supports different pose representations:

### Input Format (IsaacLab)
```python
{
    'body_pos': [T, 30, 3],      # Body positions
    'body_rot': [T, 30, 4],      # Body rotations (quaternions)
    'body_lin_vel': [T, 30, 3],  # Body linear velocities
    'body_ang_vel': [T, 30, 3],  # Body angular velocities
    'joint_pos': [T, 29],        # Joint positions
    'joint_vel': [T, 29],        # Joint velocities
}
```

### Output Format (MotionCLIP)
```python
# Shape: [num_bodies, feat_dim, seq_len]

# xyz: [30, 3, 60]
# rot6d: [30, 6, 60]
# posquat: [30, 7, 60]  # pos(3) + quat(4)
# posvel: [30, 6, 60]   # pos(3) + vel(3)

# With translation: [31, feat_dim, 60]  # +1 for translation body
```

## Window Sampling

For each frame at index `i`, we sample a window of `window_size` frames centered at `i`:

```
Frame:  0  1  2  3  4  5  6  7  8  9 10 11
Window:       [------ i=5 ------]

For i=5 with window_size=6:
  Extract frames [2, 3, 4, 5, 6, 7]
  Encode to get latent for frame 5
```

### Boundary Handling

- **pad**: Pad with edge frames (recommended)
- **truncate**: Use shorter windows at boundaries
- **mirror**: Mirror frames at boundaries

## Example Workflow

### 1. Collect Data
```bash
cd ../data_collection
python data_collection.py
# Creates: outputs/collected_data/motion_dataset.zarr
```

### 2. Dye Data (Add Latents)
```bash
cd ../data_dyeing
python data_dyeing.py \
  input.zarr_path=../../outputs/collected_data/motion_dataset.zarr \
  output.zarr_path=../../outputs/collected_data/motion_dataset_dyed.zarr
```

```bash
python data_dyeing.py \
  input.zarr_path=../../artifacts/g1_multimotion_noise_median/motion.zarr \
  output.zarr_path=../../artifacts/g1_multimotion_noise_median/motion_dyed.zarr
```

### 3. Verify Output
```python
import zarr

# Load dyed dataset
root = zarr.open('../../outputs/collected_data/motion_dataset_dyed.zarr', mode='r')

print("Fields:", list(root.keys()))
# Fields: ['act', 'body_pos', 'body_rot', ..., 'motion_latent']

print("Latent shape:", root['motion_latent'].shape)
# Latent shape: (10000, 512)  # [T, latent_dim]

# Access latent for frame i
latent_i = root['motion_latent'][i]  # [512,]
```

## Motion Encoding Modes

The data dyeing script supports two encoding modes:

### 1. **Direct Motion Encoding** (Default)
- Encodes motions directly with MotionCLIP encoder
- Results in motion embeddings in motion latent space
- Fast and efficient

### 2. **Text-Aligned Motion Encoding** (New)
- Bridges the semantic gap between motion and text embeddings
- Process:
  1. Encode motion with MotionCLIP → motion latent
  2. Compute similarity with vast vocabulary of text descriptions
  3. Create weighted average of CLIP text embeddings based on similarity
- Results in embeddings aligned with CLIP text space
- Better for conditional models trained with text-based control

**When to use text-aligned encoding:**
- When using CLIP text embeddings for conditioning during evaluation
- When you need embeddings compatible with text-based control
- When bridging semantic gap is critical

**Configuration:**
```yaml
# In data_dyeing.yaml
encoding:
  use_text_alignment: true  # Enable text-aligned embeddings
  vocabulary_path: "test/categories.yaml"  # Vocabulary file
```

**Usage:**
```bash
# Create text-aligned embeddings
python data_dyeing.py encoding.use_text_alignment=true

# Regular motion embeddings (default)
python data_dyeing.py encoding.use_text_alignment=false
```

## Performance

**Encoding Speed:**
- GPU (RTX 4090): ~5000 frames/sec
- CPU: ~500 frames/sec

**Memory Usage:**
- Batch size 128: ~4 GB GPU memory
- Batch size 256: ~8 GB GPU memory

## Troubleshooting

### Out of Memory
Reduce `encoding.batch_size`:
```bash
python data_dyeing.py encoding.batch_size=64
```

### Shape Mismatch
Check that `pose_rep` matches your MotionCLIP model:
```bash
python data_dyeing.py encoding.pose_rep=posquat
```

### Checkpoint Not Found
Update checkpoint path:
```bash
python data_dyeing.py \
  motionclip.checkpoint_path=../../../MotionCLIP/exps/your-model/checkpoint.pth.tar
```

## Integration with Diffusion Policy

The dyed dataset can be used with diffusion_policy for training:

```python
from diffusion_policy.dataset.g1_offline_dataset import G1OfflineDataset

dataset = G1OfflineDataset(
    zarr_path='outputs/collected_data/motion_dataset_dyed.zarr',
    obs_keys=['body_pos', 'body_rot', 'motion_latent'],  # Use motion_latent!
    # ...
)
```

## Notes

- Latent vectors are computed per-frame using a sliding window
- Each frame gets its own 512-dim CLIP latent
- The window ensures temporal context for better encoding
- Encoding is deterministic (no randomness in model)
