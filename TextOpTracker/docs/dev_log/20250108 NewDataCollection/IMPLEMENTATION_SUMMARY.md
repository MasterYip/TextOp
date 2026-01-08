# Data Collection Implementation Summary

## Overview

Successfully implemented a new **deterministic M×N data collection mode** alongside the existing standard random sampling mode. This provides complete motion coverage for imitation learning datasets.

## New Files Created

### 1. `commands_collection.py`
**Location**: `TextOpTracker/source/textop_tracker/textop_tracker/tasks/tracking/mdp/commands_collection.py`

**Purpose**: Specialized command term for deterministic M×N sampling

**Key Features**:
- `MotionCollectionCommand` class for deterministic sampling
- Tracks completion status per (motion_id, sample_id) task
- Assigns each environment to specific tasks
- Reassigns failed episodes (don't count as completed)
- Resets idle environments when all tasks complete
- Always starts from frame 0 (no random starting)

**Architecture**:
```python
class MotionCollectionCommand(CommandTerm):
    - task_completed: [M, N] boolean tensor tracking completion
    - env_task_assignment: maps env_id to task_id
    - env_is_idle: marks completed environments
    - _get_next_task(): finds incomplete tasks
    - _assign_task_to_env(): assigns motion/sample to env
    - _reset_to_idle(): default pose for finished envs
```

## Updated Files

### 2. `data_collection.py`
**Location**: `TextOpTracker/scripts/data_collection/data_collection.py`

**Changes**:
- Added collection mode support (`standard` vs `deterministic`)
- Mode detection from config: `cfg.collection.mode`
- Deterministic mode uses `MotionCollectionCommandCfg`
- Different progress tracking (episodes vs timesteps)
- Mode-specific stopping conditions
- Updated metadata with mode information

**Key Logic**:
```python
if collection_mode == "deterministic":
    # Use collection command
    # Track by episodes, not timesteps
    # Stop when M×N tasks complete
else:
    # Standard random sampling
    # Track by timesteps
    # Apply quality filters
```

### 3. `collect_dataset.sh`
**Location**: `TextOpTracker/scripts/data_collection/collect_dataset.sh`

**New Parameters**:
- `--mode`: Choose `standard` or `deterministic`
- `--samples_per_motion`: N samples per motion (deterministic only)
- `--help`: Comprehensive help message

**Example Usage**:
```bash
# Deterministic: 100 motions × 5 samples = 500 episodes
./collect_dataset.sh \
    -c policy.pt \
    -m "Data100/*" \
    -o output/ \
    --mode deterministic \
    --samples_per_motion 5
```

### 4. `data_collection.yaml`
**Location**: `TextOpTracker/scripts/data_collection/data_collection.yaml`

**New Configuration**:
```yaml
collection:
  mode: "standard"  # or "deterministic"
  
  # Standard mode
  len_to_save: 1000000
  min_episode_length: 300
  min_mean_reward: null
  
  # Deterministic mode
  samples_per_motion: 5  # N per motion
```

### 5. `README.md`
**Location**: `TextOpTracker/scripts/data_collection/README.md`

**Major Additions**:
- Collection Modes section explaining both modes
- Deterministic Mode Details with task assignment logic
- Comparison table: Standard vs Deterministic
- Example workflows for both modes
- Mode-specific troubleshooting
- Advanced configuration examples

## Feature Comparison

| Feature | Standard Mode | Deterministic Mode |
|---------|---------------|-------------------|
| **Sampling Strategy** | Random with filters | Sequential M×N assignment |
| **Start Position** | Random in motion | Always frame 0 |
| **Target Metric** | Timestep count | Episode count (M×N) |
| **Failed Episodes** | Filtered out | Reassigned (retry) |
| **Coverage** | Biased to easy motions | Uniform, complete |
| **Best Use Case** | Diverse, high-quality | Full representation |
| **Command Term** | `MotionCommand` | `MotionCollectionCommand` |

## Usage Examples

### Standard Mode (Original)
```bash
./collect_dataset.sh \
    -c policy.pt \
    -m "Data10k/*" \
    -o dataset/ \
    --mode standard \
    --len_to_save 1000000 \
    --min_episode_length 300
```

### Deterministic Mode (New)
```bash
./collect_dataset.sh \
    -c policy.pt \
    -m "Data100/*" \
    -o dataset/ \
    --mode deterministic \
    --samples_per_motion 10 \
    --num_envs 1000
```

## Implementation Highlights

### 1. Task Assignment Strategy
```
M motions × N samples = M×N total tasks

Env 0 → (Motion 0, Sample 0)
Env 1 → (Motion 0, Sample 1)
...
Env N → (Motion 1, Sample 0)
...
```

### 2. Completion Tracking
- Boolean tensor `task_completed[M, N]`
- Episode success = reached full motion length
- Failed episode → reassign same task
- Successful episode → mark complete, assign next

### 3. Idle Environment Handling
```python
def _reset_to_idle(env_ids):
    # Set to default pose
    # Prevent termination
    # Mark as idle
```

### 4. Progress Monitoring
```python
metrics["tasks_completed"] = completed_count
metrics["collection_progress"] = completed / total
```

## Testing Checklist

- [ ] Standard mode backward compatibility
- [ ] Deterministic mode M×N collection
- [ ] Failed episode reassignment
- [ ] Idle environment reset
- [ ] Metadata generation (both modes)
- [ ] Progress tracking accuracy
- [ ] Shell script parameter passing
- [ ] Hydra config overrides

## Benefits

1. **Complete Coverage**: Every motion sampled N times
2. **Balanced Dataset**: No bias toward easy motions
3. **Deterministic**: Reproducible data collection
4. **Failure Handling**: Automatic retry for failed episodes
5. **Backward Compatible**: Original mode still available
6. **Flexible**: Configure via YAML or CLI

## Future Enhancements

Potential improvements:
- [ ] Dynamic N per motion based on difficulty
- [ ] Resume partially completed collections
- [ ] Multi-GPU support for larger M×N
- [ ] Real-time collection progress visualization
- [ ] Automatic motion difficulty classification

## Files Modified Summary

```
TextOpTracker/
├── source/textop_tracker/textop_tracker/tasks/tracking/mdp/
│   └── commands_collection.py (NEW - 850 lines)
└── scripts/data_collection/
    ├── data_collection.py (UPDATED - added mode support)
    ├── collect_dataset.sh (UPDATED - new parameters)
    ├── data_collection.yaml (UPDATED - mode configs)
    └── README.md (UPDATED - comprehensive documentation)
```

## Documentation

All changes fully documented in:
- [README.md](README.md) - Complete usage guide
- [Command.md](../../docs/mdp/Command.md) - Technical documentation
- [data_collection.yaml](data_collection.yaml) - Configuration reference
- This summary - Implementation overview
