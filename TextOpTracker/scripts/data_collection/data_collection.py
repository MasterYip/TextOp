"""
Script to collect G1 dataset for diffuse_cloc training from TextOpTracker environment.

This script collects data from a trained tracking policy and saves it in zarr format
compatible with g1_offline_dataset.py and offline_dataset.py.

Required data fields (matching OfflineDataset in offline_dataset.py):
    - act: joint position actions [T, 29]
    - body_ang_vel: body angular velocities [T, 30, 3]
    - body_lin_vel: body linear velocities [T, 30, 3]
    - body_pos: body positions [T, 30, 3]
    - body_rot: body rotations (quaternions) [T, 30, 4]
    - joint_pos: joint positions [T, 29]
    - joint_vel: joint velocities [T, 29]
    - root_pos: root position [T, 3]
    - root_rot: root rotation (quaternion) [T, 4]
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import Dict

import click
import numpy as np
import torch
from tqdm import tqdm

# Add paths for imports
ROOT_DIR = str(Path(__file__).parent)
sys.path.append(ROOT_DIR)

from replay_buffer import ReplayBuffer


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Collect G1 dataset from tracking environment")
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to save dataset (e.g., artifacts/g1_tracking_dataset/motion.zarr)"
    )
    parser.add_argument(
        "-c", "--checkpoint",
        required=True,
        help="Path to trained policy checkpoint"
    )
    parser.add_argument(
        "-m", "--motion_file",
        required=True,
        help="Motion file pattern (e.g., Data10k-open)"
    )
    parser.add_argument(
        "-t", "--task",
        default="Isaac-TextOp-Tracking-G1-Direct-v0",
        help="Task name"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=100,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--min_episode_length",
        type=int,
        default=300,
        help="Minimum episode length to keep (for data quality)"
    )
    parser.add_argument(
        "--min_mean_reward",
        type=float,
        default=None,
        help="Minimum mean reward per episode to keep (alternative quality filter)"
    )
    parser.add_argument(
        "--len_to_save",
        type=int,
        default=500000,
        help="Total number of timesteps to save"
    )
    parser.add_argument(
        "--max_episode_length",
        type=int,
        default=1000,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--chunk_length",
        type=int,
        default=-1,
        help="Chunk length for zarr file, -1 for auto"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization"
    )
    parser.add_argument(
        "--load_pickle_cfg",
        action="store_true",
        help="Load environment and agent config from pickle files instead of Hydra"
    )
    return parser


def extract_robot_state(env) -> Dict[str, np.ndarray]:
    """
    Extract robot state data from the environment.
    
    Returns dictionary with keys matching OfflineDataset requirements:
        - body_pos: [num_envs, 30, 3]
        - body_rot: [num_envs, 30, 4] 
        - body_lin_vel: [num_envs, 30, 3]
        - body_ang_vel: [num_envs, 30, 3]
        - joint_pos: [num_envs, 29]
        - joint_vel: [num_envs, 29]
        - root_pos: [num_envs, 3]
        - root_rot: [num_envs, 4]
    """
    robot = env.scene["robot"]
    command = env.command_manager.get_term("motion")
    
    # Get body data for all bodies (30 bodies for G1)
    # robot.data.body_pos_w and body_quat_w include all bodies
    body_pos_w = robot.data.body_pos_w.clone()  # [num_envs, num_bodies, 3]
    body_quat_w = robot.data.body_quat_w.clone()  # [num_envs, num_bodies, 4]
    body_lin_vel_w = robot.data.body_lin_vel_w.clone()  # [num_envs, num_bodies, 3]
    body_ang_vel_w = robot.data.body_ang_vel_w.clone()  # [num_envs, num_bodies, 3]
    
    # Remove environment origins to get world-frame positions
    env_origins = env.scene.env_origins  # [num_envs, 3]
    body_pos_w = body_pos_w - env_origins[:, None, :]
    
    # Get joint data (29 joints for G1)
    joint_pos = robot.data.joint_pos.clone()  # [num_envs, 29]
    joint_vel = robot.data.joint_vel.clone()  # [num_envs, 29]
    
    # Get root (pelvis) data - root is the first body (index 0)
    root_pos = body_pos_w[:, 0, :]  # [num_envs, 3]
    root_rot = body_quat_w[:, 0, :]  # [num_envs, 4]
    
    return {
        "body_pos": body_pos_w.cpu().numpy(),  # [num_envs, 30, 3]
        "body_rot": body_quat_w.cpu().numpy(),  # [num_envs, 30, 4]
        "body_lin_vel": body_lin_vel_w.cpu().numpy(),  # [num_envs, 30, 3]
        "body_ang_vel": body_ang_vel_w.cpu().numpy(),  # [num_envs, 30, 3]
        "joint_pos": joint_pos.cpu().numpy(),  # [num_envs, 29]
        "joint_vel": joint_vel.cpu().numpy(),  # [num_envs, 29]
        "root_pos": root_pos.cpu().numpy(),  # [num_envs, 3]
        "root_rot": root_rot.cpu().numpy(),  # [num_envs, 4]
    }


def collect_data(args):
    """Main data collection loop."""
    
    # Import Isaac Sim related modules
    from isaaclab.app import AppLauncher
    
    # Create app launcher
    app_launcher_args = argparse.Namespace(
        headless=args.headless or not args.visualize,
        livestream=False,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        enable_cameras=False,
    )
    app_launcher = AppLauncher(app_launcher_args)
    simulation_app = app_launcher.app
    
    import gymnasium as gym
    from rsl_rl.runners import OnPolicyRunner
    
    # Import tasks to register environment
    import textop_tracker.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import register_task_to_hydra
    from isaaclab.utils.io.pkl import load_pickle
    
    # Try to load config using Hydra first (like play.py), fallback to pickle
    if args.load_pickle_cfg:
        param_dir = Path(args.checkpoint).parent / "params"
        env_cfg = load_pickle(str(param_dir / "env.pkl"))
        agent_cfg = load_pickle(str(param_dir / "agent.pkl"))
        print(f"[INFO] Successfully loaded config from pickle files")
    else:
        env_cfg, agent_cfg = register_task_to_hydra(args.task, "rsl_rl_cfg_entry_point")
        print(f"[INFO] Successfully loaded config using Hydra")
        
    
    # Update environment config
    env_cfg.scene.num_envs = args.num_envs
    
    # Set motion files
    motion_files = glob.glob(str(Path("./artifacts") / Path(args.motion_file) / "motion.npz"))
    if not motion_files:
        raise FileNotFoundError(f"No motion.npz found in {Path('./artifacts') / Path(args.motion_file)}")
    env_cfg.commands.motion.motion_files = motion_files
    
    print(f"[INFO] Using {len(motion_files)} motion files")
    print(f"[INFO] Number of environments: {args.num_envs}")
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    
    # Create environment
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env_unwrapped = env.unwrapped
    
    # Load policy
    log_dir = os.path.dirname(args.checkpoint)
    wrapped_env = RslRlVecEnvWrapper(env)
    
    ppo_runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(args.checkpoint)
    policy = ppo_runner.get_inference_policy(device=env_unwrapped.device)
    
    # Initialize replay buffer
    buffer = ReplayBuffer.create_empty_numpy()
    
    # Storage for current episodes
    episode_data = {
        "act": [[] for _ in range(args.num_envs)],
        "body_pos": [[] for _ in range(args.num_envs)],
        "body_rot": [[] for _ in range(args.num_envs)],
        "body_lin_vel": [[] for _ in range(args.num_envs)],
        "body_ang_vel": [[] for _ in range(args.num_envs)],
        "joint_pos": [[] for _ in range(args.num_envs)],
        "joint_vel": [[] for _ in range(args.num_envs)],
        "root_pos": [[] for _ in range(args.num_envs)],
        "root_rot": [[] for _ in range(args.num_envs)],
    }
    
    # Track episode statistics for quality filtering
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    episode_rewards = np.zeros(args.num_envs, dtype=np.float32)
    
    # Statistics tracking
    total_saved_steps = 0
    total_episodes_collected = 0
    total_episodes_saved = 0
    
    # Reset environment
    obs, _ = wrapped_env.get_observations()
    
    print(f"[INFO] Starting data collection...")
    print(f"[INFO] Target: {args.len_to_save} timesteps")
    print(f"[INFO] Quality filter: min_episode_length={args.min_episode_length}")
    if args.min_mean_reward is not None:
        print(f"[INFO] Quality filter: min_mean_reward={args.min_mean_reward}")
    
    pbar = tqdm(total=args.len_to_save, desc="Collecting data")
    
    with torch.inference_mode():
        while total_saved_steps < args.len_to_save and simulation_app.is_running():
            # Get action from policy
            actions = policy(obs)
            
            # Extract robot state before step
            robot_state = extract_robot_state(env_unwrapped)
            
            # Store current state and action for all envs
            for env_idx in range(args.num_envs):
                episode_data["act"][env_idx].append(actions[env_idx].cpu().numpy())
                episode_data["body_pos"][env_idx].append(robot_state["body_pos"][env_idx])
                episode_data["body_rot"][env_idx].append(robot_state["body_rot"][env_idx])
                episode_data["body_lin_vel"][env_idx].append(robot_state["body_lin_vel"][env_idx])
                episode_data["body_ang_vel"][env_idx].append(robot_state["body_ang_vel"][env_idx])
                episode_data["joint_pos"][env_idx].append(robot_state["joint_pos"][env_idx])
                episode_data["joint_vel"][env_idx].append(robot_state["joint_vel"][env_idx])
                episode_data["root_pos"][env_idx].append(robot_state["root_pos"][env_idx])
                episode_data["root_rot"][env_idx].append(robot_state["root_rot"][env_idx])
            
            # Step environment
            obs, rewards, terminated, infos = wrapped_env.step(actions)
            dones = terminated
            
            # Update episode statistics
            episode_lengths += 1
            episode_rewards += rewards.cpu().numpy()
            
            # Check for finished episodes
            if dones.any():
                done_indices = torch.where(dones)[0].cpu().numpy()
                
                for env_idx in done_indices:
                    total_episodes_collected += 1
                    ep_length = episode_lengths[env_idx]
                    ep_reward = episode_rewards[env_idx]
                    mean_reward = ep_reward / ep_length if ep_length > 0 else 0
                    
                    # Apply quality filters
                    keep_episode = True
                    
                    # Filter by episode length
                    if ep_length < args.min_episode_length:
                        keep_episode = False
                    
                    # Filter by mean reward (if specified)
                    if args.min_mean_reward is not None and mean_reward < args.min_mean_reward:
                        keep_episode = False
                    
                    if keep_episode and len(episode_data["act"][env_idx]) > 0:
                        # Convert lists to numpy arrays (float64 for compatibility)
                        ep_data = {
                            "act": np.array(episode_data["act"][env_idx], dtype=np.float64),
                            "body_pos": np.array(episode_data["body_pos"][env_idx], dtype=np.float64),
                            "body_rot": np.array(episode_data["body_rot"][env_idx], dtype=np.float64),
                            "body_lin_vel": np.array(episode_data["body_lin_vel"][env_idx], dtype=np.float64),
                            "body_ang_vel": np.array(episode_data["body_ang_vel"][env_idx], dtype=np.float64),
                            "joint_pos": np.array(episode_data["joint_pos"][env_idx], dtype=np.float64),
                            "joint_vel": np.array(episode_data["joint_vel"][env_idx], dtype=np.float64),
                            "root_pos": np.array(episode_data["root_pos"][env_idx], dtype=np.float64),
                            "root_rot": np.array(episode_data["root_rot"][env_idx], dtype=np.float64),
                        }
                        
                        # Add episode to buffer
                        buffer.add_episode(ep_data)
                        total_saved_steps += ep_length
                        total_episodes_saved += 1
                        pbar.update(ep_length)
                    
                    # Reset episode data for this environment
                    for key in episode_data:
                        episode_data[key][env_idx] = []
                    episode_lengths[env_idx] = 0
                    episode_rewards[env_idx] = 0
            
            # Stop if we have enough data
            if total_saved_steps >= args.len_to_save:
                break
    
    pbar.close()
    
    # Print statistics
    print(f"\n[INFO] Data collection complete!")
    print(f"[INFO] Total episodes collected: {total_episodes_collected}")
    print(f"[INFO] Total episodes saved: {total_episodes_saved}")
    print(f"[INFO] Total timesteps saved: {total_saved_steps}")
    print(f"[INFO] Episode retention rate: {total_episodes_saved/max(total_episodes_collected,1)*100:.1f}%")
    
    # Save dataset
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Saving dataset to {args.output}")
    
    # Set chunk length
    chunk_length = args.chunk_length if args.chunk_length > 0 else None
    
    # Save with optimal compression for disk storage
    buffer.save_to_path(
        args.output,
        chunks={'act': None, 'body_pos': None, 'body_rot': None,
                'body_lin_vel': None, 'body_ang_vel': None,
                'joint_pos': None, 'joint_vel': None,
                'root_pos': None, 'root_rot': None} if chunk_length is None else {},
        compressors='disk',  # Use zstd compression for better disk storage
    )
    
    # Save metadata
    metadata = {
        'task': args.task,
        'checkpoint': args.checkpoint,
        'motion_file': args.motion_file,
        'num_envs': int(args.num_envs),
        'min_episode_length': int(args.min_episode_length),
        'min_mean_reward': float(args.min_mean_reward) if args.min_mean_reward is not None else None,
        'len_to_save': int(args.len_to_save),
        'total_episodes_collected': int(total_episodes_collected),
        'total_episodes_saved': int(total_episodes_saved),
        'total_timesteps': int(total_saved_steps),
        'n_episodes': int(buffer.n_episodes),
        'seed': int(args.seed),
        'creation_time': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    import json
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[INFO] Metadata saved to {metadata_path}")
    print(f"[INFO] Dataset contains {buffer.n_episodes} episodes")
    
    # Close environment and simulation
    env.close()
    simulation_app.close()
    
    return metadata


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    
    collect_data(args)
