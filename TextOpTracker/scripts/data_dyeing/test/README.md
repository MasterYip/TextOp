# Dyed Data Visualization

Visualization tools for motion data encoded with CLIP latent vectors.

## Features

1. **Real-time Motion-to-Text Visualization**
   - Displays G1 robot motion animation
   - Shows predicted text descriptions in real-time
   - Displays confidence scores for top predictions
   - Plots confidence trajectory over time

2. **t-SNE Trajectory Visualization**
   - Projects motion latents into 2D semantic space using t-SNE
   - Shows episode trajectory through the semantic space
   - Displays reference motion clusters for context
   - Color-codes trajectory by time progression

## Usage

### Basic Usage

```bash
cd TextOpTracker/scripts/data_dyeing/test

python dyed_data_vis.py \
    --zarr_path ../../../artifacts/g1_multimotion_noise_median/motion_dyed.zarr \
    --checkpoint ../../../../MotionCLIP/exps/g1-model-xyz/checkpoint_0100.pth.tar
```

### Advanced Options

```bash
python dyed_data_vis.py \
    --zarr_path <path_to_dyed_zarr> \
    --checkpoint <path_to_checkpoint> \
    --episode 5 \
    --fps 30 \
    --device cuda \
    --output_dir ./visualizations
```

### Arguments

- `--zarr_path`: Path to dyed zarr dataset (required)
- `--checkpoint`: Path to MotionCLIP checkpoint (required)
- `--config`: Path to config YAML file (optional, auto-detected from checkpoint dir)
- `--episode`: Episode index to visualize (default: 0)
- `--device`: Device to use - 'cuda' or 'cpu' (default: cuda)
- `--fps`: Frames per second for video output (default: 30)
- `--output_dir`: Output directory for visualizations (default: current directory)

## Output Files

The script generates two files per episode:

1. **`episode_<N>_realtime.gif`**: Animated visualization showing:
   - 3D G1 robot motion
   - Top-3 predicted text descriptions with confidence scores
   - Confidence trajectory plot

2. **`episode_<N>_tsne.png`**: Static visualization showing:
   - Reference motion clusters in t-SNE space
   - Episode trajectory through semantic space
   - Start (green star) and end (red X) markers
   - Color-coded trajectory by frame number

## Requirements

The script requires:
- Dyed dataset with `motion_latent` field (run `data_dyeing.py` first)
- MotionCLIP checkpoint (G1 model)
- Category definitions (categories_simple.yaml in MotionCLIP)
- AMASS dataset (for reference motions in t-SNE)

## Example Workflow

```bash
# 1. Collect motion data
cd ../data_collection
python data_collection.py

# 2. Dye the data with CLIP latents
cd ../data_dyeing
python data_dyeing.py \
    input.zarr_path=../../artifacts/g1_multimotion/motion.zarr \
    output.zarr_path=../../artifacts/g1_multimotion/motion_dyed.zarr

# 3. Visualize the dyed data
cd test
python dyed_data_vis.py \
    --zarr_path ../../../artifacts/g1_multimotion/motion_dyed.zarr \
    --checkpoint ../../../../MotionCLIP/exps/g1-model-xyz/checkpoint_0100.pth.tar \
    --episode 0
```

## Implementation Details

### Real-time Text Prediction

The visualization:
1. Loads pre-computed motion latents from the dyed dataset
2. Encodes vocabulary with CLIP text encoder
3. Computes cosine similarity between motion and text latents
4. Uses softmax to get confidence scores
5. Displays top-K predictions for each frame

### t-SNE Trajectory

The visualization:
1. Loads reference motions from AMASS dataset (using categories_simple.yaml)
2. Encodes reference motions with MotionCLIP encoder
3. Combines episode latents with reference latents
4. Applies t-SNE dimensionality reduction
5. Plots trajectory with reference clusters

### G1 Motion Rendering

Uses the same kinematic chain visualization as MotionCLIP:
- 30 body positions (pelvis, legs, torso, arms)
- 5 kinematic chains with different colors
- Centered on pelvis for clarity
- Ground trajectory for translation context

## Troubleshooting

### Missing motion_latent field

If you get `ValueError: Dataset does not contain 'motion_latent' field`:
- Run data dyeing first: `python data_dyeing.py ...`
- Check that output zarr has the motion_latent field

### Out of memory

If you run out of GPU memory:
- Use `--device cpu` to run on CPU
- Reduce the number of reference motions in categories_simple.yaml
- Process fewer episodes

### No reference motions found

If you see "Warning: No reference motions found":
- Check that AMASS dataset path is correct in config
- Verify categories_simple.yaml exists and has valid motion labels
- Ensure the dataset split ('all') is loaded correctly
