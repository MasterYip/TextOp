"""This script demonstrates how to replay collected dataset episodes for quality verification.

.. code-block:: bash

    # Usage - Replay first 10 episodes
    python replay_dataset.py --zarr_path ../output/motion.zarr --num_episodes 10
    
    # Usage - Replay specific episodes
    python replay_dataset.py --zarr_path ../output/motion.zarr --episode_ids 0 5 10 15
    
    # Usage - Replay all episodes (up to max_envs limit)
    python replay_dataset.py --zarr_path ../output/motion.zarr
"""
"""Launch Isaac Sim Simulator first."""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay collected dataset episodes for quality verification.")
parser.add_argument("--zarr_path", type=str, required=True, help="Path to zarr dataset file")
parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes to replay (from start)")
parser.add_argument("--episode_ids", type=int, nargs='+', default=None, help="Specific episode IDs to replay")
parser.add_argument("--max_envs", type=int, default=50, help="Maximum number of environments to create")
parser.add_argument("--slow_motion", type=float, default=1.0, help="Slow motion factor (1.0 = normal speed, 0.5 = half speed)")

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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from replay_buffer import ReplayBuffer

##
# Pre-defined configs
##
from textop_tracker.robots.g1 import G1_CYLINDER_CFG


@configclass
class ReplayDatasetSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay dataset scene."""

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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, buffer: ReplayBuffer, episode_ids: list, slow_motion: float):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    
    # Get number of environments (should match number of episodes)
    num_envs = scene.num_envs
    
    # Get episode data
    episodes = []
    episode_lengths = []
    for ep_id in episode_ids:
        episode_data = buffer.get_episode(ep_id, copy=True)
        episodes.append(episode_data)
        # Get length from any data field
        ep_length = episode_data['root_pos'].shape[0]
        episode_lengths.append(ep_length)
    
    print(f"\n[INFO] Loaded {len(episodes)} episodes:")
    for i, (ep_id, ep_len) in enumerate(zip(episode_ids, episode_lengths)):
        print(f"  Env {i}: Episode {ep_id} ({ep_len} timesteps)")
    
    # Initialize time steps for each environment
    time_steps = torch.zeros(num_envs, dtype=torch.long, device=sim.device)
    
    # Print episode info
    print("\n[INFO] Episode-Environment Mapping:")
    for env_idx, ep_id in enumerate(episode_ids[:num_envs]):
        print(f"  [ENV {env_idx}] Episode {ep_id} ({episode_lengths[env_idx]} frames)")

    # Simulation loop
    print("\n[INFO] Starting replay simulation...")
    print(f"[INFO] Slow motion factor: {slow_motion}x")
    print("[INFO] Press Ctrl+C to stop\n")
    
    frame_counter = 0
    
    while simulation_app.is_running():
        # Apply slow motion by skipping frames
        if slow_motion < 1.0:
            if frame_counter % int(1.0 / slow_motion) != 0:
                frame_counter += 1
                sim.render()
                continue
        
        time_steps += 1
        
        # Reset environments that reached end of their episode
        for env_idx in range(num_envs):
            if time_steps[env_idx] >= episode_lengths[env_idx]:
                time_steps[env_idx] = 0
        
        # Prepare batch data for all environments
        root_states = robot.data.default_root_state.clone()
        joint_pos_batch = torch.zeros(num_envs, robot.num_joints, device=sim.device)
        joint_vel_batch = torch.zeros(num_envs, robot.num_joints, device=sim.device)
        
        # Fetch episode data for each environment
        for env_idx in range(num_envs):
            episode_data = episodes[env_idx]
            time_step = time_steps[env_idx].item()
            
            # Get data from zarr (convert to torch tensors)
            root_pos = torch.from_numpy(episode_data['root_pos'][time_step]).to(device=sim.device, dtype=torch.float32)
            root_rot = torch.from_numpy(episode_data['root_rot'][time_step]).to(device=sim.device, dtype=torch.float32)
            body_lin_vel = torch.from_numpy(episode_data['body_lin_vel'][time_step]).to(device=sim.device, dtype=torch.float32)
            body_ang_vel = torch.from_numpy(episode_data['body_ang_vel'][time_step]).to(device=sim.device, dtype=torch.float32)
            joint_pos = torch.from_numpy(episode_data['joint_pos'][time_step]).to(device=sim.device, dtype=torch.float32)
            joint_vel = torch.from_numpy(episode_data['joint_vel'][time_step]).to(device=sim.device, dtype=torch.float32)
            
            # Set root state (position, quaternion, linear velocity, angular velocity)
            # Root body velocities are body_lin_vel[0] and body_ang_vel[0]
            root_states[env_idx, :3] = root_pos + scene.env_origins[env_idx]
            root_states[env_idx, 3:7] = root_rot
            root_states[env_idx, 7:10] = body_lin_vel[0]  # Root linear velocity
            root_states[env_idx, 10:] = body_ang_vel[0]   # Root angular velocity
            
            # Set joint state
            joint_pos_batch[env_idx] = joint_pos
            joint_vel_batch[env_idx] = joint_vel

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
        
        frame_counter += 1


def main():
    # Check if zarr path exists
    zarr_path = args_cli.zarr_path
    if not os.path.exists(zarr_path):
        print(f"[ERROR] Zarr path does not exist: {zarr_path}")
        return
    
    # Load replay buffer
    print(f"[INFO] Loading dataset from: {zarr_path}")
    buffer = ReplayBuffer.create_from_path(zarr_path, mode='r')
    
    total_episodes = buffer.n_episodes
    total_timesteps = buffer.n_steps
    
    print(f"[INFO] Dataset contains {total_episodes} episodes, {total_timesteps} total timesteps")
    
    # Print episode lengths
    episode_lengths = buffer.episode_lengths
    print(f"[INFO] Episode length statistics:")
    print(f"  Min: {episode_lengths.min()} timesteps")
    print(f"  Max: {episode_lengths.max()} timesteps")
    print(f"  Mean: {episode_lengths.mean():.1f} timesteps")
    print(f"  Median: {np.median(episode_lengths):.1f} timesteps")
    
    # Determine which episodes to replay
    if args_cli.episode_ids is not None:
        # Use specific episode IDs
        episode_ids = args_cli.episode_ids
        # Validate episode IDs
        invalid_ids = [ep_id for ep_id in episode_ids if ep_id >= total_episodes or ep_id < 0]
        if invalid_ids:
            print(f"[ERROR] Invalid episode IDs: {invalid_ids}")
            print(f"[ERROR] Valid range: 0 to {total_episodes - 1}")
            return
    elif args_cli.num_episodes is not None:
        # Use first N episodes
        num_episodes = min(args_cli.num_episodes, total_episodes, args_cli.max_envs)
        episode_ids = list(range(num_episodes))
    else:
        # Use all episodes (up to max_envs)
        num_episodes = min(total_episodes, args_cli.max_envs)
        episode_ids = list(range(num_episodes))
    
    num_envs = len(episode_ids)
    
    # Limit to max_envs
    if num_envs > args_cli.max_envs:
        print(f"[WARNING] Requested {num_envs} episodes, limiting to {args_cli.max_envs}")
        episode_ids = episode_ids[:args_cli.max_envs]
        num_envs = args_cli.max_envs
    
    print("="*70)
    print(f"Dataset Replay Configuration")
    print("="*70)
    print(f"Zarr path: {zarr_path}")
    print(f"Total episodes in dataset: {total_episodes}")
    print(f"Episodes to replay: {num_envs}")
    print(f"Episode IDs: {episode_ids}")
    print(f"Slow motion: {args_cli.slow_motion}x")
    print("="*70)
    
    # Setup simulation
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    # Create scene with num_envs matching number of episodes
    scene_cfg = ReplayDatasetSceneCfg(num_envs=num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    # Run the simulator with episode data
    run_simulator(sim, scene, buffer, episode_ids, args_cli.slow_motion)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    os._exit(0)  # type: ignore
