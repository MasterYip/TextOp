"""Interactive data selection tool for motion replay with pagination.

This script allows users to replay multiple motions and interactively select them
for copying to a new dataset folder.

.. code-block:: bash

    # Usage
    python data_selection.py --motion_pattern "Data10k-open/*" --output_dir ./artifacts/selected_motions
"""
"""Launch Isaac Sim Simulator first."""

import os
import argparse
import glob
import numpy as np
import torch
import threading
import shutil
import yaml
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Interactive motion selection with replay.")
parser.add_argument("--motion_pattern", type=str, required=True, help="Glob pattern for motion files (e.g., 'Data10k-open/*')")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for selected motions")
parser.add_argument("--page_size", type=int, default=100, help="Number of motions to display per page")
parser.add_argument("--max_envs", type=int, default=100, help="Maximum number of environments to display at once")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from textop_tracker.robots.g1 import G1_CYLINDER_CFG
from textop_tracker.tasks.tracking.mdp.commands_multi import MultiMotionLoader


@configclass
class DataSelectionSceneCfg(InteractiveSceneCfg):
    """Configuration for data selection scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionSelector:
    """Interactive motion selector with pagination."""
    
    def __init__(self, motion_files: list, page_size: int, output_dir: str):
        self.all_motion_files = motion_files
        self.page_size = page_size
        self.output_dir = Path(output_dir)
        self.current_page = 0
        self.total_pages = (len(motion_files) + page_size - 1) // page_size
        self.selected_motions = set()
        self.should_exit = False
        self.page_changed = False
        
        # Motion names for display
        self.motion_names = [Path(f).parent.name for f in motion_files]
        
        print("\n" + "="*80)
        print("INTERACTIVE MOTION SELECTOR")
        print("="*80)
        print(f"Total motions: {len(motion_files)}")
        print(f"Page size: {page_size}")
        print(f"Total pages: {self.total_pages}")
        print(f"Output directory: {self.output_dir}")
        print("="*80)
        
    def get_current_page_files(self):
        """Get motion files for current page."""
        start_idx = self.current_page * self.page_size
        end_idx = min(start_idx + self.page_size, len(self.all_motion_files))
        return self.all_motion_files[start_idx:end_idx], start_idx
    
    def get_global_env_id(self, local_env_id: int) -> int:
        """Convert local env_id to global index."""
        return self.current_page * self.page_size + local_env_id
    
    def print_help(self):
        """Print help message."""
        print("\n" + "="*80)
        print("COMMANDS:")
        print("="*80)
        print("  select <id1> <id2> ...    - Select motions by env_id (space-separated)")
        print("  deselect <id1> <id2> ...  - Deselect motions by env_id")
        print("  list                      - List currently displayed motions")
        print("  selected                  - Show all selected motions")
        print("  next                      - Go to next page")
        print("  prev                      - Go to previous page")
        print("  goto <page>               - Go to specific page (0-indexed)")
        print("  page                      - Show current page info")
        print("  done                      - Finish selection and copy files")
        print("  quit                      - Exit without saving")
        print("  help                      - Show this help")
        print("="*80 + "\n")
    
    def print_page_info(self):
        """Print current page information."""
        start_idx = self.current_page * self.page_size
        end_idx = min(start_idx + self.page_size, len(self.all_motion_files))
        print(f"\n[Page {self.current_page + 1}/{self.total_pages}] Showing motions {start_idx}-{end_idx-1} (env_id 0-{end_idx-start_idx-1})")
        print(f"Selected: {len(self.selected_motions)} motions total")
    
    def list_current_page(self):
        """List motions on current page."""
        _, start_idx = self.get_current_page_files()
        page_files, _ = self.get_current_page_files()
        
        print("\n" + "-"*80)
        print("CURRENT PAGE MOTIONS:")
        print("-"*80)
        for local_id, motion_file in enumerate(page_files):
            global_id = start_idx + local_id
            motion_name = Path(motion_file).parent.name
            selected_mark = "[✓]" if global_id in self.selected_motions else "[ ]"
            print(f"{selected_mark} env_id {local_id:3d} (global {global_id:4d}): {motion_name}")
        print("-"*80 + "\n")
    
    def show_selected(self):
        """Show all selected motions."""
        if not self.selected_motions:
            print("\n[INFO] No motions selected yet.\n")
            return
        
        print("\n" + "-"*80)
        print(f"SELECTED MOTIONS ({len(self.selected_motions)} total):")
        print("-"*80)
        for global_id in sorted(self.selected_motions):
            motion_name = self.motion_names[global_id]
            print(f"  {global_id:4d}: {motion_name}")
        print("-"*80 + "\n")
    
    def handle_command(self, command: str):
        """Handle user command."""
        parts = command.strip().split()
        if not parts:
            return
        
        cmd = parts[0].lower()
        
        if cmd == "help":
            self.print_help()
        
        elif cmd == "select":
            if len(parts) < 2:
                print("[ERROR] Usage: select <id1> <id2> ...")
                return
            
            for env_id_str in parts[1:]:
                try:
                    local_id = int(env_id_str)
                    global_id = self.get_global_env_id(local_id)
                    
                    if global_id >= len(self.all_motion_files):
                        print(f"[ERROR] env_id {local_id} is out of range for current page")
                        continue
                    
                    # Check if it's in current page
                    _, start_idx = self.get_current_page_files()
                    page_files, _ = self.get_current_page_files()
                    if local_id >= len(page_files):
                        print(f"[ERROR] env_id {local_id} is out of range for current page (max: {len(page_files)-1})")
                        continue
                    
                    self.selected_motions.add(global_id)
                    print(f"[✓] Selected: env_id {local_id} -> {self.motion_names[global_id]}")
                except ValueError:
                    print(f"[ERROR] Invalid env_id: {env_id_str}")
        
        elif cmd == "deselect":
            if len(parts) < 2:
                print("[ERROR] Usage: deselect <id1> <id2> ...")
                return
            
            for env_id_str in parts[1:]:
                try:
                    local_id = int(env_id_str)
                    global_id = self.get_global_env_id(local_id)
                    
                    if global_id in self.selected_motions:
                        self.selected_motions.remove(global_id)
                        print(f"[✗] Deselected: env_id {local_id} -> {self.motion_names[global_id]}")
                    else:
                        print(f"[INFO] env_id {local_id} was not selected")
                except ValueError:
                    print(f"[ERROR] Invalid env_id: {env_id_str}")
        
        elif cmd == "list":
            self.list_current_page()
        
        elif cmd == "selected":
            self.show_selected()
        
        elif cmd == "next":
            if self.current_page < self.total_pages - 1:
                self.current_page += 1
                self.page_changed = True
                self.print_page_info()
                print("[INFO] Moved to next page. Simulation will restart.")
            else:
                print("[INFO] Already on last page")
        
        elif cmd == "prev":
            if self.current_page > 0:
                self.current_page -= 1
                self.page_changed = True
                self.print_page_info()
                print("[INFO] Moved to previous page. Simulation will restart.")
            else:
                print("[INFO] Already on first page")
        
        elif cmd == "goto":
            if len(parts) < 2:
                print("[ERROR] Usage: goto <page_number>")
                return
            
            try:
                page_num = int(parts[1])
                if 0 <= page_num < self.total_pages:
                    self.current_page = page_num
                    self.page_changed = True
                    self.print_page_info()
                    print("[INFO] Moved to page. Simulation will restart.")
                else:
                    print(f"[ERROR] Page number must be between 0 and {self.total_pages - 1}")
            except ValueError:
                print("[ERROR] Invalid page number")
        
        elif cmd == "page":
            self.print_page_info()
        
        elif cmd == "done":
            self.finish_selection()
        
        elif cmd == "quit":
            print("\n[INFO] Exiting without saving...")
            self.should_exit = True
        
        else:
            print(f"[ERROR] Unknown command: {cmd}")
            print("Type 'help' for list of commands")
    
    def finish_selection(self):
        """Finish selection and copy files."""
        if not self.selected_motions:
            print("\n[WARNING] No motions selected!")
            confirm = input("Exit anyway? (yes/no): ")
            if confirm.lower() in ['yes', 'y']:
                self.should_exit = True
            return
        
        print("\n" + "="*80)
        print("FINISHING SELECTION")
        print("="*80)
        print(f"Total selected: {len(self.selected_motions)} motions")
        print(f"Output directory: {self.output_dir}")
        print("="*80)
        
        confirm = input("\nProceed with copying? (yes/no): ")
        if confirm.lower() not in ['yes', 'y']:
            print("[INFO] Cancelled.")
            return
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy selected motions
        copied_motions = []
        print("\n[INFO] Copying selected motions...")
        
        for global_id in sorted(self.selected_motions):
            motion_file = self.all_motion_files[global_id]
            motion_name = self.motion_names[global_id]
            
            # Source and destination paths
            src_dir = Path(motion_file).parent
            dst_dir = self.output_dir / motion_name
            
            # Copy entire motion directory
            if dst_dir.exists():
                print(f"  [OVERRIDE] {motion_name}")
                shutil.rmtree(dst_dir)
            else:
                print(f"  [COPY] {motion_name}")
            
            shutil.copytree(src_dir, dst_dir)
            copied_motions.append(motion_name)
        
        # Save motion list to YAML
        yaml_path = self.output_dir / "selected_motions.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump({
                'total_motions': len(copied_motions),
                'motions': copied_motions,
                'source_pattern': args_cli.motion_pattern,
            }, f, default_flow_style=False)
        
        print("\n" + "="*80)
        print("SELECTION COMPLETE")
        print("="*80)
        print(f"Copied: {len(copied_motions)} motions")
        print(f"Location: {self.output_dir}")
        print(f"Motion list: {yaml_path}")
        print("="*80 + "\n")
        
        self.should_exit = True
    
    def input_thread(self):
        """Thread for handling user input."""
        self.print_help()
        self.print_page_info()
        self.list_current_page()
        
        while not self.should_exit:
            try:
                command = input("\n> ")
                self.handle_command(command)
            except EOFError:
                print("\n[INFO] EOF detected, exiting...")
                self.should_exit = True
                break
            except KeyboardInterrupt:
                print("\n[INFO] Interrupted, exiting...")
                self.should_exit = True
                break


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, selector: MotionSelector):
    """Run the simulator with interactive selection."""
    # Extract scene entities
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()
    
    # Body indexes for tracking
    body_indexes = [0]
    
    # Start input thread
    input_thread = threading.Thread(target=selector.input_thread, daemon=True)
    input_thread.start()
    
    # Main simulation loop with page support
    while simulation_app.is_running() and not selector.should_exit:
        # Get current page files
        page_motion_files, start_idx = selector.get_current_page_files()
        num_envs = len(page_motion_files)
        
        if num_envs == 0:
            print("[ERROR] No motions on current page!")
            break
        
        # Load motions for current page
        motion = MultiMotionLoader(
            page_motion_files,
            body_indexes,
            sim.device,
        )
        
        # Initialize time steps
        time_steps = torch.zeros(num_envs, dtype=torch.long, device=sim.device)
        motion_indices = torch.arange(num_envs, dtype=torch.long, device=sim.device)
        
        # Reset page changed flag
        selector.page_changed = False
        
        # Render current page
        while simulation_app.is_running() and not selector.should_exit and not selector.page_changed:
            time_steps += 1
            
            # Reset environments that reached end
            for env_idx in range(num_envs):
                motion_idx = motion_indices[env_idx].item()
                if time_steps[env_idx] >= motion.file_lengths[motion_idx]:
                    time_steps[env_idx] = 0
            
            # Prepare batch data
            root_states = robot.data.default_root_state.clone()
            joint_pos_batch = torch.zeros(num_envs, robot.num_joints, device=sim.device)
            joint_vel_batch = torch.zeros(num_envs, robot.num_joints, device=sim.device)
            
            # Fetch motion data for each environment
            for env_idx in range(num_envs):
                motion_idx = motion_indices[env_idx].item()
                time_step = time_steps[env_idx].item()
                
                motion_data = motion.get_motion_data_batch(
                    motion_idx,
                    time_step,
                    time_step + 1
                )
                
                # Set root state
                root_states[env_idx, :3] = motion_data["body_pos_w"][0, 0] + scene.env_origins[env_idx]
                root_states[env_idx, 3:7] = motion_data["body_quat_w"][0, 0]
                root_states[env_idx, 7:10] = motion_data["body_lin_vel_w"][0, 0]
                root_states[env_idx, 10:] = motion_data["body_ang_vel_w"][0, 0]
                
                # Set joint state
                joint_pos_batch[env_idx] = motion_data["joint_pos"][0]
                joint_vel_batch[env_idx] = motion_data["joint_vel"][0]
            
            # Write states to simulation
            robot.write_root_state_to_sim(root_states)
            robot.write_joint_state_to_sim(joint_pos_batch, joint_vel_batch)
            scene.write_data_to_sim()
            sim.render()
            scene.update(sim_dt)
            
            # Camera follows first environment
            if time_steps[0] < 10:
                pos_lookat = root_states[0, :3].cpu().numpy()
                sim.set_camera_view(pos_lookat + np.array([3.0, 3.0, 1.5]), pos_lookat)


def main():
    # Find motion files using glob
    motion_files = glob.glob(str(Path("./artifacts") / args_cli.motion_pattern / "motion.npz"))
    
    if not motion_files:
        print(f"[ERROR] No motion files found matching pattern: {args_cli.motion_pattern}")
        print(f"[ERROR] Search path: ./artifacts/{args_cli.motion_pattern}/motion.npz")
        return
    
    # Sort for consistent ordering
    motion_files = sorted(motion_files)
    
    print("="*80)
    print(f"Interactive Motion Selection")
    print("="*80)
    print(f"Pattern: {args_cli.motion_pattern}")
    print(f"Found: {len(motion_files)} motion files")
    print(f"Page size: {args_cli.page_size}")
    print(f"Max display envs: {min(args_cli.page_size, args_cli.max_envs)}")
    print("="*80)
    
    # Use page_size but limit by max_envs
    display_size = min(args_cli.page_size, args_cli.max_envs)
    
    # Create motion selector
    selector = MotionSelector(motion_files, display_size, args_cli.output_dir)
    
    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)
    
    # Create scene with display_size envs
    scene_cfg = DataSelectionSceneCfg(num_envs=display_size, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    # Run the simulator
    run_simulator(sim, scene, selector)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    os._exit(0)
