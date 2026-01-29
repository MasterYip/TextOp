# Body Index Remapping Fix

## Problem

**Critical Bug**: Body and joint indices are ordered differently between IsaacLab (TextOpTracker) and MotionCLIP, causing incorrect motion encoding and visualization.

### Root Cause

- **IsaacLab**: Returns bodies/joints in **alphabetical order** from URDF
  - Example: `['left_ankle_pitch_link', 'left_ankle_roll_link', 'left_elbow_link', ...]`
  
- **MotionCLIP**: Uses **semantic ordering** for G1 AMASS dataset
  - Example: `['pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', ...]`

### Impact

Without remapping:
- ❌ Motion latents encode wrong body positions
- ❌ Visualizations show incorrect skeleton structure
- ❌ Motion-to-text predictions are meaningless
- ❌ t-SNE trajectories don't match actual motion semantics

## Solution

Created automatic body/joint index remapping utilities:

### New Files

1. **`body_index_mapping.py`** - Core remapping logic
   - `create_body_index_mapping()` - Generate body index map
   - `create_dof_index_mapping()` - Generate joint index map
   - `remap_isaaclab_to_motionclip()` - Convert IsaacLab → MotionCLIP
   - `remap_motionclip_to_isaaclab()` - Convert MotionCLIP → IsaacLab
   - `print_mapping_info()` - Debug utility

2. **`test/test_body_mapping.py`** - Comprehensive test suite
   - Tests ordering correctness
   - Tests round-trip remapping
   - Verifies specific indices
   - Prints detailed mapping table

### Updated Files

1. **`motion_converter.py`**
   - Added automatic remapping in `convert()` method
   - Now accepts IsaacLab order, outputs MotionCLIP order
   - Updated docstrings to clarify ordering

2. **`test/visualize_utils.py`**
   - Added `auto_remap` parameter to `G1MotionVisualizer`
   - Automatically remaps from IsaacLab order when enabled

3. **`test/dyed_data_vis.py`**
   - Enabled `auto_remap=True` for visualizer
   - Imports remapping utilities

## Usage

### Data Dyeing (Automatic)

```python
# motion_converter.py automatically handles remapping
converter = G1MotionConverter(pose_rep='posquat')

# Input: IsaacLab order [T, 30, 3]
body_pos_isaaclab = buffer['body_pos'][:]

# Output: MotionCLIP order [30, feat_dim, T]
motion_tensor = converter.convert(
    body_pos=body_pos_isaaclab,  # IsaacLab order
    body_rot=buffer['body_rot'][:]  # IsaacLab order
)  # Returns MotionCLIP order
```

### Visualization (Automatic)

```python
# Visualizer with auto-remapping enabled
visualizer = G1MotionVisualizer(auto_remap=True)

# Input: IsaacLab order [T, 30, 3]
body_positions = episode_data['body_pos']  # IsaacLab order

# Automatically remaps to MotionCLIP order before rendering
visualizer.draw_frame(ax, body_positions, frame_idx)
```

### Manual Remapping

```python
from body_index_mapping import remap_isaaclab_to_motionclip

# Prepare data dict
data = {
    'body_pos': isaaclab_body_pos,  # [T, 30, 3] IsaacLab order
    'body_rot': isaaclab_body_rot,  # [T, 30, 4] IsaacLab order
    'joint_pos': isaaclab_joint_pos,  # [T, 29] IsaacLab order
}

# Remap to MotionCLIP order
motionclip_data = remap_isaaclab_to_motionclip(data)

# Now in MotionCLIP order
body_pos_mc = motionclip_data['body_pos']  # [T, 30, 3] MotionCLIP order
```

## Verification

### Run Tests

```bash
cd TextOpTracker/scripts/data_dyeing/test
python test_body_mapping.py
```

Expected output:
```
TEST 1: Body Name Ordering                          ✓ PASS
TEST 2: DOF Name Ordering                           ✓ PASS  
TEST 3: Remapping Round-Trip                        ✓ PASS
TEST 4: Specific Index Mapping                      ✓ PASS

Total: 4/4 tests passed
✓ All tests passed! Body index remapping is working correctly.
```

### Mapping Example

```
BODY INDEX MAPPING: IsaacLab -> MotionCLIP
================================================================
MotionCLIP Idx     IsaacLab Idx    Body Name
----------------------------------------------------------------
0                  15              pelvis
1                  1               left_hip_pitch_link
2                  2               left_hip_roll_link
3                  3               left_hip_yaw_link
4                  4               left_knee_link
5                  0               left_ankle_pitch_link
6                  7               left_ankle_roll_link
7                  16              right_hip_pitch_link
...
```

## Important Notes

### When Remapping is Applied

✅ **Automatically handled**:
- `motion_converter.py` - During motion encoding
- `visualize_utils.py` - During visualization (with `auto_remap=True`)
- `dyed_data_vis.py` - Uses auto-remapping visualizer

❌ **NOT needed**:
- Data stored in zarr (stays in IsaacLab order)
- Data collection (uses IsaacLab order)
- Replay buffer operations (uses IsaacLab order)

### Data Flow

```
Data Collection (IsaacLab order)
    ↓
Zarr Dataset (IsaacLab order)
    ↓
Motion Converter → [REMAP] → MotionCLIP Encoder (MotionCLIP order)
    ↓
Motion Latents (stored in zarr, no order dependency)
    ↓
Visualization → [REMAP] → G1MotionVisualizer (MotionCLIP order)
```

## References

- **MotionCLIP Ordering**: `MotionCLIP/src/datasets/g1_amass_utils.py`
  - `G1_BODY_NAMES` - Semantic body ordering
  - `G1_DOF_NAMES` - Semantic joint ordering

- **IsaacLab Ordering**: Alphabetical from URDF
  - Retrieved via `robot.data.body_pos_w`, `robot.data.joint_pos`
  - Determined by URDF link/joint definition order

## Troubleshooting

### "Skeleton looks wrong in visualization"

Check if auto_remap is enabled:
```python
visualizer = G1MotionVisualizer(auto_remap=True)  # ← Must be True!
```

### "Motion latents don't match motions"

Verify motion_converter is using remapping:
```python
from body_index_mapping import remap_isaaclab_to_motionclip
# Should be imported and used in motion_converter.py
```

### "Test failures in test_body_mapping.py"

The body/joint name lists may need updating if:
- G1 URDF changed
- MotionCLIP g1_amass_utils.py changed
- Check both files and update `body_index_mapping.py` accordingly
