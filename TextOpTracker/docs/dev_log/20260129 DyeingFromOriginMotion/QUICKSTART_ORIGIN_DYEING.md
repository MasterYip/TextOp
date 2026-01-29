# Quick Start: Origin Motion Dyeing

## When to Use
Use origin motion dyeing when you have:
- ✅ Collected data with deterministic mode (M motions × N samples)
- ✅ Original motion npz files available in artifacts
- ✅ Dataset with `motion_idx` field tracking

## Setup Steps

### 1. Verify Dataset Has Motion Tracking
Check your zarr dataset has the `motion_idx` field:
```bash
python -c "
from diffusion_policy.dataset.replay_buffer import ReplayBuffer
buffer = ReplayBuffer.create_from_path('path/to/dataset.zarr', mode='r')
print('Fields:', list(buffer.data.keys()))
print('Has motion_idx:', 'motion_idx' in buffer.data)
"
```

### 2. Verify Motion Files Exist
Check your artifacts directory structure:
```
artifacts/
  selected_motions/
    motion_0/
      motion.npz
    motion_1/
      motion.npz
    ...
```

### 3. Configure data_dyeing.yaml
```yaml
input:
  zarr_path: "../../outputs/collected_data/dataset.zarr"
  dyeing_from_origin_motion: true
  motion_pattern: "selected_motions/*"
  motion_base_path: "../../artifacts"
  
output:
  zarr_path: "../../outputs/dyed_data/dataset.zarr"
```

### 4. Run Dyeing
```bash
cd TextOpTracker/scripts/data_dyeing
python data_dyeing.py
```

## Expected Output

```
Configuration:
...
================================================================================
Using ORIGIN MOTION DYEING mode
================================================================================
Loading original motion files from: ../../artifacts
Pattern: selected_motions/*

Loading dataset...
  Dataset: ../../outputs/collected_data/dataset.zarr
  Episodes: 1000
  Total frames: 50000
  Fields: ['action', 'obs', 'motion_idx', ...]

[2/4] Loading and encoding original motions...
  Found 20 motion files in ../../artifacts/selected_motions/
Encoding motions: 100%|████████| 20/20 [00:45<00:00,  2.25s/it]
  [0] motion_0: 120 frames @ 30.0 fps -> latents (120, 512)
  [1] motion_1: 150 frames @ 30.0 fps -> latents (150, 512)
  ...
  Cached 20 motion latents

[3/4] Attaching cached latents to dataset samples...
Attaching latents: 100%|████████| 1000/1000 [00:05<00:00, 200.0it/s]

  [INFO] Found 50 episodes with length mismatch:
  (Using linear interpolation to align latents)
    Episode 0 (motion 0): 118 frames vs 120 original
    Episode 1 (motion 0): 119 frames vs 120 original
    ...

[4/4] Saving dataset with latent field...
  Saving to: ../../outputs/dyed_data/dataset.zarr
  Latent field: 'motion_latent_clip' with shape (50000, 512)
  Dataset saved successfully!
  Total size: 1234.56 MB

================================================================================
Data Dyeing Complete!
================================================================================
Input dataset: ../../outputs/collected_data/dataset.zarr
Output dataset: ../../outputs/dyed_data/dataset.zarr
Total frames: 50000
Total episodes: 1000
Latent field: 'motion_latent_clip' with shape (50000, 512)
Latent dim: 512
================================================================================
```

## Performance Comparison

For 20 motions × 50 samples = 1000 episodes:

| Mode | Time | Motions Encoded |
|------|------|-----------------|
| Standard | ~50 min | 1000 episodes |
| **Origin** | **~1 min** | **20 motions** |

**Speedup: 50×**

## Troubleshooting

### "motion_idx field not found"
Your dataset wasn't collected with motion tracking. Re-collect with:
```yaml
output:
  sort_by_motion_idx: true
```

### "Found 0 motion files"
Check your `motion_pattern` and `motion_base_path`. The pattern should match your artifact structure.

### "motion_idx not in cache"
Motion files don't match collection order. Ensure motion files are in the same order as used during collection.

## Next Steps

After dyeing, use the dyed dataset for training:
```bash
cd ../../diffuse_cloc
python train.py \
    task.dataset.zarr_path=../TextOpTracker/outputs/dyed_data/dataset.zarr \
    train_args.use_cond=true
```
