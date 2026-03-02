# Data Selection Tool

## Key Features:

**1. Motion Replay with Pagination:**
- Displays motions in batches (default 100 per page)
- Handles large datasets efficiently
- Shows local env_id (0-99) mapped to global motion index

**2. Interactive Commands:**
- `select <id1> <id2> ...` - Select motions by env_id (space-separated)
- `deselect <id1> <id2> ...` - Deselect motions
- `list` - Show current page motions with selection status
- `selected` - Display all selected motions
- `next/prev` - Navigate pages
- `goto <page>` - Jump to specific page
- `page` - Show current page info
- `done` - Copy selected motions and save to YAML
- `quit` - Exit without saving
- `help` - Show command list

**3. File Management:**
- Copies selected motion directories to output folder
- Overrides existing motions with same name
- Saves selection list to `selected_motions.yaml` with metadata

## Usage:

```bash
cd /home/user/CodeSpace/Diffusion/TextOp/TextOpTracker
python ./scripts/data_selection/data_selection.py \
    --motion_pattern "Data10k-open/*" \
    --output_dir ./artifacts/selected_motions \
    --page_size 100
```

The script runs in a separate input thread, allowing you to interact via terminal while watching the motion replay. When you change pages, the simulation restarts with the new batch of motions.
