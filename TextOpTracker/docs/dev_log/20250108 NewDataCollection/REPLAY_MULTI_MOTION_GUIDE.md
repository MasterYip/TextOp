# Multi-Motion Replay Guide

This guide explains how to use `replay_npz_multi.py` to visualize multiple motions simultaneously in Isaac Lab.

## Features

- **Multi-Motion Loading**: Load multiple motion files using glob patterns
- **One Environment Per Motion**: Each motion gets its own environment for easy comparison
- **Automatic Configuration**: Number of environments automatically matches number of motions
- **Motion Labels**: Console output shows which motion is in which environment
- **Synchronized Playback**: All motions play simultaneously with automatic looping

## Usage

### Basic Examples

```bash
# Replay all LAFAN motions
cd TextOpTracker
python scripts/replay_npz_multi.py --motion_pattern "Data10k-open/*"
python scripts/replay_npz_multi.py --motion_pattern "lafan_*"

# Replay all available motions
python scripts/replay_npz_multi.py --motion_pattern "*"

# Replay specific motion types
python scripts/replay_npz_multi.py --motion_pattern "walk_*"
python scripts/replay_npz_multi.py --motion_pattern "*_jump"

# Replay single motion (backward compatibility)
python scripts/replay_npz_multi.py --motion_pattern "lafan_walk_short"

# Limit number of environments
python scripts/replay_npz_multi.py --motion_pattern "*" --max_envs 10

# Run headless (no GUI)
python scripts/replay_npz_multi.py --motion_pattern "lafan_*" --headless
```

### Command-Line Arguments

```
--motion_pattern PATTERN    Glob pattern for motion files (e.g., 'lafan_*' or '*')
--motion_file FILE          [Deprecated] Single motion file name
--max_envs N                Maximum number of environments to create (default: 20)
--headless                  Run without GUI
--device DEVICE             Device to use (default: cuda:0)
```

## How It Works

### 1. Motion File Discovery

The script searches for motion files using the glob pattern:
```
./artifacts/{pattern}/motion.npz
```

For example, `--motion_pattern "lafan_*"` searches:
```
./artifacts/lafan_walk_short/motion.npz
./artifacts/lafan_run/motion.npz
./artifacts/lafan_dance/motion.npz
...
```

### 2. Environment Creation

- **Automatic Sizing**: Creates exactly one environment per motion file
- **Spatial Layout**: Environments are spaced 2.0 units apart in a grid
- **Independent Playback**: Each environment replays its own motion independently

### 3. Motion Playback

Each environment:
- Loads its assigned motion data using `MultiMotionLoader`
- Plays the motion at 50 fps (dt=0.02)
- Automatically loops when reaching the end
- Updates robot joint positions, velocities, and root state

### 4. Environment Labels

The console output shows which motion is in which environment:
```
[INFO] Loaded 5 motions:
  Env 0: lafan_dance (450 frames)
  Env 1: lafan_jump (120 frames)
  Env 2: lafan_run (280 frames)
  Env 3: lafan_walk (340 frames)
  Env 4: lafan_walk_short (180 frames)

[ENV 0] Motion: lafan_dance
[ENV 1] Motion: lafan_jump
[ENV 2] Motion: lafan_run
[ENV 3] Motion: lafan_walk
[ENV 4] Motion: lafan_walk_short
```

## Motion File Structure

Expected structure:
```
artifacts/
├── lafan_walk_short/
│   └── motion.npz
├── lafan_run/
│   └── motion.npz
├── lafan_dance/
│   └── motion.npz
...
```

Each `motion.npz` file should contain:
- `fps`: Frame rate (typically 50)
- `joint_pos`: Joint positions [T, num_joints]
- `joint_vel`: Joint velocities [T, num_joints]
- `body_pos_w`: Body positions [T, num_bodies, 3]
- `body_quat_w`: Body rotations [T, num_bodies, 4]
- `body_lin_vel_w`: Body linear velocities [T, num_bodies, 3]
- `body_ang_vel_w`: Body angular velocities [T, num_bodies, 3]

## Camera Control

- **Auto-Focus**: Camera automatically follows the first environment (env 0)
- **Manual Control**: Use Isaac Lab's built-in camera controls:
  - Mouse drag: Rotate view
  - Mouse scroll: Zoom
  - Arrow keys: Pan

## Performance Tips

### Limit Number of Environments

Rendering many environments can be slow. Use `--max_envs` to limit:
```bash
# Only show first 5 motions
python scripts/replay_npz_multi.py --motion_pattern "*" --max_envs 5
```

### Headless Mode

For faster playback without GUI (useful for debugging):
```bash
python scripts/replay_npz_multi.py --motion_pattern "lafan_*" --headless
```

### GPU Selection

Specify GPU device if you have multiple:
```bash
python scripts/replay_npz_multi.py --motion_pattern "*" --device cuda:1
```

## Comparison with Single Motion Replay

| Feature | Single Motion | Multi Motion |
|---------|--------------|--------------|
| Motion files | 1 | Multiple (via glob) |
| Environments | Fixed (10) | Auto (1 per motion) |
| Playback | Same motion in all envs | Different motion per env |
| Use case | Debug single motion | Compare multiple motions |

## Troubleshooting

### "No motion files found"
- Check that motion files exist in `./artifacts/`
- Verify glob pattern matches directory names
- Example: `ls ./artifacts/*/motion.npz`

### Too many environments / Out of memory
- Use `--max_envs` to limit number of environments
- Start with small number: `--max_envs 3`

### Motions not synchronized
- This is expected! Each motion has different length
- Shorter motions will loop more frequently
- This is useful for comparing motion durations

### Camera not following robot
- Camera follows first environment (env 0) only
- Manually adjust camera to view other environments

## Integration with Data Collection

This script uses the same `MultiMotionLoader` class as `data_collection.py`, ensuring:
- Consistent motion loading behavior
- Same data format and coordinate frames
- Easy verification of collected data quality

You can verify collected data by replaying the source motions:
```bash
# If you collected data from lafan_* motions
python scripts/replay_npz_multi.py --motion_pattern "lafan_*"
```

## Example Workflows

### Verify Motion Conversion
```bash
# After converting SMPL/AMASS data to G1 format
python scripts/replay_npz_multi.py --motion_pattern "*" --max_envs 5
```

### Compare Motion Types
```bash
# Compare all walking motions
python scripts/replay_npz_multi.py --motion_pattern "walk_*"

# Compare all running motions
python scripts/replay_npz_multi.py --motion_pattern "run_*"
```

### Quality Check Before Data Collection
```bash
# Preview motions that will be used for data collection
python scripts/replay_npz_multi.py --motion_pattern "lafan_*"

# Then collect data with same pattern
cd scripts/data_collection
python data_collection.py motion.pattern="lafan_*"
```

## Related Files

- `scripts/data_collection/data_collection.py` - Collects dataset using same motion loader
- `textop_tracker/tasks/tracking/mdp/commands_multi.py` - MultiMotionLoader implementation
- `artifacts/*/motion.npz` - Motion data files

## Tips

1. **Start Small**: Test with a few motions first (`--max_envs 3`)
2. **Use Patterns**: Take advantage of naming conventions (`lafan_*`, `walk_*`)
3. **Check Console**: Motion-to-environment mapping is printed at startup
4. **Spatial Awareness**: Environments are laid out in a grid, env 0 is at origin
5. **Loop Length**: Notice which motions are longer by observing loop frequency
