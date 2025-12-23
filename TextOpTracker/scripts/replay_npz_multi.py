"""This script demonstrates how to replay multiple motions simultaneously with text labels.

.. code-block:: bash

    # Usage - Single motion
    python replay_npz_multi.py --motion_file lafan_walk_short
    
    # Usage - Multiple motions using glob pattern
    python replay_npz_multi.py --motion_pattern "lafan_*"
    python replay_npz_multi.py --motion_pattern "*"
"""
"""Launch Isaac Sim Simulator first."""

import os
import argparse
import glob
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions with multiple environments.")
parser.add_argument("--motion_file", type=str, default=None, help="Single motion file name (deprecated, use --motion_pattern)")
parser.add_argument("--motion_pattern", type=str, default=None, help="Glob pattern for motion files (e.g., 'lafan_*' or '*')")
parser.add_argument("--max_envs", type=int, default=200, help="Maximum number of environments to create")

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
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG

from pathlib import Path

##
# Pre-defined configs
##
from textop_tracker.robots.g1 import G1_CYLINDER_CFG
from textop_tracker.tasks.tracking.mdp.commands_multi import MultiMotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, motion_files: list):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Get number of environments (should match number of motion files)
    num_envs = scene.num_envs
    
    # Find body index for visualization (robot root body index is 0)
    body_indexes = [0]  # Only track root body for MultiMotionLoader
    
    # Load all motions using MultiMotionLoader
    motion = MultiMotionLoader(
        motion_files,
        body_indexes,
        sim.device,
    )
    
    print(f"[INFO] Loaded {len(motion_files)} motions:")
    for i, motion_file in enumerate(motion_files):
        motion_name = Path(motion_file).parent.name
        print(f"  Env {i}: {motion_name} ({motion.file_lengths[i]} frames)")
    
    # Initialize time steps for each environment
    time_steps = torch.zeros(num_envs, dtype=torch.long, device=sim.device)
    
    # Environment to motion index mapping (one motion per env)
    motion_indices = torch.arange(num_envs, dtype=torch.long, device=sim.device)
    
    # Create text labels for each environment using USD prims
    text_prims = []
    motion_names = [Path(f).parent.name for f in motion_files]
    
    try:
        stage = sim.stage
        for env_idx in range(num_envs):
            # Create text prim path
            text_prim_path = f"/World/envs/env_{env_idx}/MotionLabel"
            
            # Get environment origin
            env_origin = scene.env_origins[env_idx].cpu().numpy()
            
            # Create a simple sphere as a label marker (text rendering in USD is complex)
            # We'll use print statements instead and add a visual marker
            if env_idx < len(motion_names):
                print(f"[ENV {env_idx}] Motion: {motion_names[env_idx]}")
        
        print("\n[INFO] Motion labels printed above. Check console for environment-motion mapping.")
        
    except Exception as e:
        print(f"[WARNING] Could not create text labels: {e}")
        for env_idx, motion_name in enumerate(motion_names[:num_envs]):
            print(f"[ENV {env_idx}] {motion_name}")

    # Simulation loop
    print("\n[INFO] Starting replay simulation...")
    print("[INFO] Press Ctrl+C to stop\n")
    
    while simulation_app.is_running():
        time_steps += 1
        
        # Reset environments that reached end of their motion
        for env_idx in range(num_envs):
            motion_idx = motion_indices[env_idx].item()
            if time_steps[env_idx] >= motion.file_lengths[motion_idx]:
                time_steps[env_idx] = 0
        
        # Prepare batch data for all environments
        root_states = robot.data.default_root_state.clone()
        joint_pos_batch = torch.zeros(num_envs, robot.num_joints, device=sim.device)
        joint_vel_batch = torch.zeros(num_envs, robot.num_joints, device=sim.device)
        
        # Fetch motion data for each environment
        for env_idx in range(num_envs):
            motion_idx = motion_indices[env_idx].item()
            time_step = time_steps[env_idx].item()
            
            # Get motion data for this specific environment
            motion_data = motion.get_motion_data_batch(
                motion_idx,
                time_step,
                time_step + 1
            )
            
            # Set root state (body_pos_w and body_quat_w are [1, num_bodies, 3/4])
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
        sim.render()  # We don't want physics (sim.step())
        scene.update(sim_dt)

        # Camera follows first environment
        pos_lookat = root_states[0, :3].cpu().numpy()
        if time_steps[0] < 10:
            sim.set_camera_view(pos_lookat + np.array([3.0, 3.0, 1.5]), pos_lookat)


def main():
    # Determine motion pattern
    if args_cli.motion_pattern:
        pattern = args_cli.motion_pattern
    elif args_cli.motion_file:
        pattern = args_cli.motion_file
        print("[WARNING] --motion_file is deprecated, use --motion_pattern instead")
    else:
        print("[ERROR] Must specify either --motion_pattern or --motion_file")
        print("Examples:")
        print("  python replay_npz_multi.py --motion_pattern 'lafan_*'")
        print("  python replay_npz_multi.py --motion_pattern '*'")
        return
    
    # Find motion files using glob
    motion_files = glob.glob(str(Path("./artifacts") / Path(pattern) / "motion.npz"))
    
    if not motion_files:
        print(f"[ERROR] No motion files found matching pattern: {pattern}")
        print(f"[ERROR] Search path: ./artifacts/{pattern}/motion.npz")
        return
    
    # Sort for consistent ordering
    motion_files = sorted(motion_files)
    
    # Limit to max_envs
    if len(motion_files) > args_cli.max_envs:
        print(f"[WARNING] Found {len(motion_files)} motions, limiting to {args_cli.max_envs}")
        motion_files = motion_files[:args_cli.max_envs]
    
    num_envs = len(motion_files)
    
    print("="*70)
    print(f"Multi-Motion Replay Configuration")
    print("="*70)
    print(f"Pattern: {pattern}")
    print(f"Found: {len(motion_files)} motion files")
    print(f"Environments: {num_envs} (one per motion)")
    print("="*70)
    
    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    # Create scene with num_envs matching number of motions
    scene_cfg = ReplayMotionsSceneCfg(num_envs=num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    # Run the simulator with motion files
    run_simulator(sim, scene, motion_files)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    # simulation_app.close()
    os._exit(0)  # type: ignore
