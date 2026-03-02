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

Action Noise Injection:
    Following BeyondMimic, we inject temporally correlated OU noise during rollout:
    η_{t+1} = η_t + θ(μ - η_t)Δt + σ√Δt ε_t
    
    This creates state diversity and collects corrective actions for robustness.

Collection Modes:
    - standard: Random sampling with quality filters (original behavior)
    - deterministic: M motions × N samples = M*N episodes, full coverage
"""

import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from tqdm import tqdm

# Add paths for imports
ROOT_DIR = str(Path(__file__).parent)
sys.path.append(ROOT_DIR)

from diffusion_policy.utils.replay_buffer import ReplayBuffer
from diffusion_policy.utils.g1fk_torch import build_fk_calculator


class OUNoise:
    """
    Ornstein-Uhlenbeck noise generator for temporally correlated action perturbations.
    
    Following BeyondMimic: "Overdamped PD gains suppress high-frequency perturbations,
    limiting state diversity. OU noise produces temporally correlated action perturbations."
    
    Update formula: η_{t+1} = η_t + θ(μ - η_t)Δt + σ√Δt ε_t
    
    Args:
        action_dim: Dimension of action space (e.g., 29 for G1 joints)
        theta: Mean reversion rate (default: 0.8)
        mu: Long-term mean (default: 0.0)
        sigma: Joint-wise noise scale (default: 0.1)
        dt: Time step (default: 1.0)
        device: Torch device
    """
    
    def __init__(
        self,
        action_dim: int,
        theta: float = 0.8,
        mu: float = 0.0,
        sigma: float = 0.1,
        dt: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.action_dim = action_dim
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.device = device
        
        # Initialize noise state
        self.state = None
        
    def reset(self, batch_size: int):
        """Reset noise state to zero (or mu)."""
        self.state = torch.ones(batch_size, self.action_dim, device=self.device) * self.mu
        
    def sample(self) -> torch.Tensor:
        """
        Generate next noise sample using OU process.
        
        Returns:
            Noise tensor [batch_size, action_dim]
        """
        if self.state is None:
            raise RuntimeError("OUNoise must be reset before sampling")
        
        # ε_t ~ N(0, I)
        epsilon = torch.randn_like(self.state)
        
        # η_{t+1} = η_t + θ(μ - η_t)Δt + σ√Δt ε_t
        self.state = (
            self.state
            + self.theta * (self.mu - self.state) * self.dt
            + self.sigma * np.sqrt(self.dt) * epsilon
        )
        
        return self.state.clone()


@hydra.main(version_base=None, config_path=".", config_name="data_collection")
def main(cfg: DictConfig):
    """Main data collection entry point with Hydra configuration."""
    
    # Print configuration
    print("="*70)
    print("G1 Dataset Collection Configuration")
    print("="*70)
    print(OmegaConf.to_yaml(cfg))
    print("="*70)
    
    # Run data collection
    collect_data(cfg)


def collect_data(cfg: DictConfig):
    """Main data collection loop with OU noise injection."""
    
    # Import Isaac Sim related modules
    from isaaclab.app import AppLauncher
    import argparse
    
    # Create app launcher
    app_launcher_args = argparse.Namespace(
        headless=cfg.visualization.headless,
        livestream=False,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        enable_cameras=cfg.visualization.enable_cameras,
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
    if cfg.checkpoint.load_pickle_cfg:
        param_dir = Path(cfg.checkpoint.path).parent / "params"
        env_cfg = load_pickle(str(param_dir / "env.pkl"))
        agent_cfg = load_pickle(str(param_dir / "agent.pkl"))
        # env_cfg.scene.robot.spawn.fix_base = True  # Ensure robot base is fixed
        print(f"[INFO] Successfully loaded config from pickle files")
    else:
        env_cfg, agent_cfg = register_task_to_hydra(cfg.task.name, "rsl_rl_cfg_entry_point")
        print(f"[INFO] Successfully loaded config using Hydra")
        
    
    # Update environment config
    env_cfg.scene.num_envs = cfg.task.num_envs
    
    # Set motion files
    motion_files = glob.glob(str(Path("./artifacts") / Path(cfg.motion.pattern) / "motion.npz"))
    if not motion_files:
        raise FileNotFoundError(f"No motion.npz found in {Path('./artifacts') / Path(cfg.motion.pattern)}")
    
    # === Handle collection mode ===
    collection_mode = cfg.collection.get("mode", "standard")
    
    if collection_mode == "deterministic":
        # Deterministic M*N sampling mode
        samples_per_motion = cfg.collection.samples_per_motion
        print(f"[INFO] Using DETERMINISTIC collection mode")
        print(f"[INFO] {len(motion_files)} motions × {samples_per_motion} samples = {len(motion_files) * samples_per_motion} total episodes")
        
        # Need to use collection command term instead of standard motion command
        from textop_tracker.tasks.tracking.mdp.commands_collection import MotionCollectionCommandCfg
        
        # Replace motion command with collection command
        env_cfg.commands.motion = MotionCollectionCommandCfg(
            asset_name=env_cfg.commands.motion.asset_name,
            anchor_body_name=env_cfg.commands.motion.anchor_body_name,
            body_names=env_cfg.commands.motion.body_names,
            motion_files=motion_files,
            samples_per_motion=samples_per_motion,
            default_height=getattr(env_cfg.commands.motion, "default_height", 2.0),
            pose_range=getattr(env_cfg.commands.motion, "pose_range", {}),
            velocity_range=getattr(env_cfg.commands.motion, "velocity_range", {}),
            joint_position_range=getattr(env_cfg.commands.motion, "joint_position_range", (-0.52, 0.52)),
            future_steps=getattr(env_cfg.commands.motion, "future_steps", 1),
            resampling_time_range=(1.0e9, 1.0e9),
        )
        env_cfg.rewards = None
    else:
        # Standard random sampling mode (backward compatible)
        print(f"[INFO] Using STANDARD collection mode")
        if len(motion_files) > cfg.task.num_envs:
            print(f"[WARNING] Number of motion files ({len(motion_files)}) exceeds num_envs ({cfg.task.num_envs}), truncating for loading speed.")
            motion_files = motion_files[:cfg.task.num_envs]
        
        env_cfg.commands.motion.motion_files = motion_files
    
    print(f"[INFO] Using {len(motion_files)} motion files")
    print(f"[INFO] Number of environments: {cfg.task.num_envs}")
    print(f"[INFO] Loading checkpoint: {cfg.checkpoint.path}")
    
    # Print noise configuration
    if cfg.noise.enable:
        print(f"[INFO] Action noise injection ENABLED")
        print(f"[INFO] Noise type: {cfg.noise.type}")
        if cfg.noise.type == "ou":
            print(f"[INFO] OU parameters: θ={cfg.noise.theta}, μ={cfg.noise.mu}, σ={cfg.noise.sigma}, Δt={cfg.noise.dt}")
    else:
        print(f"[INFO] Action noise injection DISABLED")
    
    # Create environment
    env = gym.make(cfg.task.name, cfg=env_cfg, render_mode=None)
    env_unwrapped = env.unwrapped
    
    # Load policy
    log_dir = os.path.dirname(cfg.checkpoint.path)
    wrapped_env = RslRlVecEnvWrapper(env)
    
    ppo_runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(cfg.checkpoint.path)
    policy = ppo_runner.get_inference_policy(device=env_unwrapped.device)
    
    # Initialize noise generator if enabled
    noise_generator = None
    if cfg.noise.enable:
        action_dim = wrapped_env.num_actions
        if cfg.noise.type == "ou":
            noise_generator = OUNoise(
                action_dim=action_dim,
                theta=cfg.noise.theta,
                mu=cfg.noise.mu,
                sigma=cfg.noise.sigma,
                dt=cfg.noise.dt,
                device=env_unwrapped.device,
            )
            noise_generator.reset(cfg.task.num_envs)
            print(f"[INFO] Initialized OU noise generator for {action_dim} actions")
        else:
            raise ValueError(f"Unknown noise type: {cfg.noise.type}")
    
    # Initialize FK calculator if enabled
    fk_calculator = None
    if cfg.fk.use_fk:
        print(f"[INFO] Forward Kinematics ENABLED")
        print(f"[INFO] FK backend: {cfg.fk.fk_type}, device: {cfg.fk.fk_device}")
        fk_calculator = build_fk_calculator(
            fk_type=cfg.fk.fk_type,
            urdf_path=cfg.fk.urdf_path,
            device=cfg.fk.fk_device,
        )
        print(f"[INFO] FK calculator initialized")
    else:
        print(f"[INFO] Forward Kinematics DISABLED (using simulation body states)")
    
    # Initialize replay buffer
    buffer = ReplayBuffer.create_empty_numpy()
    
    # Storage for current episodes
    episode_data = {
        "act": [[] for _ in range(cfg.task.num_envs)],
        "body_pos": [[] for _ in range(cfg.task.num_envs)],
        "body_rot": [[] for _ in range(cfg.task.num_envs)],
        "body_lin_vel": [[] for _ in range(cfg.task.num_envs)],
        "body_ang_vel": [[] for _ in range(cfg.task.num_envs)],
        "joint_pos": [[] for _ in range(cfg.task.num_envs)],
        "joint_vel": [[] for _ in range(cfg.task.num_envs)],
        "root_pos": [[] for _ in range(cfg.task.num_envs)],
        "root_rot": [[] for _ in range(cfg.task.num_envs)],
        "motion_idx": [[] for _ in range(cfg.task.num_envs)],  # Track which motion file
    }
    
    # Track episode statistics for quality filtering
    episode_lengths = np.zeros(cfg.task.num_envs, dtype=np.int32)
    episode_rewards = np.zeros(cfg.task.num_envs, dtype=np.float32)
    
    # Track which envs just reset (to skip storing their first frame)
    env_just_reset = np.ones(cfg.task.num_envs, dtype=bool)  # All envs start as "just reset"
    
    # Statistics tracking
    total_saved_steps = 0
    total_episodes_collected = 0
    total_episodes_saved = 0
    
    # Reset environment
    obs, _ = wrapped_env.get_observations()
    
    print(f"[INFO] Starting data collection...")
    print(f"[INFO] Target: {cfg.collection.len_to_save} timesteps")
    print(f"[INFO] Quality filter: min_episode_length={cfg.collection.min_episode_length}")
    if cfg.collection.min_mean_reward is not None:
        print(f"[INFO] Quality filter: min_mean_reward={cfg.collection.min_mean_reward}")
    
    # === Deterministic mode: target is M*N episodes, not timesteps ===
    if collection_mode == "deterministic":
        total_tasks = len(motion_files) * samples_per_motion
        target_episodes = total_tasks
        print(f"[INFO] Deterministic mode: collecting {target_episodes} episodes")
        pbar = tqdm(total=target_episodes, desc="Collecting episodes")
    else:
        target_episodes = float('inf')  # No limit
        pbar = tqdm(total=cfg.collection.len_to_save, desc="Collecting data")

    with torch.inference_mode():
        while simulation_app.is_running():
            # Check stopping conditions
            if collection_mode == "deterministic":
                if total_episodes_saved >= target_episodes:
                    break
            else:
                if total_saved_steps >= cfg.collection.len_to_save:
                    break
            
            # Get action from policy
            actions = policy(obs)
            
            # Extract robot state with noise from privileged observations
            # The observation manager applies noise internally based on the ObsTerm config
            priv_obs = wrapped_env.env.observation_manager.compute_group("critic")
            
            # Reshape observations from flat vectors back to original shapes
            robot_state = {
                "body_pos": priv_obs["body_pos"].reshape(-1, 30, 3),      # [num_envs, 30, 3]
                "body_rot": priv_obs["body_rot"].reshape(-1, 30, 4),      # [num_envs, 30, 4]
                "body_lin_vel": priv_obs["body_lin_vel"].reshape(-1, 30, 3),  # [num_envs, 30, 3]
                "body_ang_vel": priv_obs["body_ang_vel"].reshape(-1, 30, 3),  # [num_envs, 30, 3]
                "joint_pos": priv_obs["joint_pos"],                       # [num_envs, 29]
                "joint_vel": priv_obs["joint_vel"],                       # [num_envs, 29]
                "root_pos": priv_obs["root_pos"],                         # [num_envs, 3]
                "root_rot": priv_obs["root_rot"],                         # [num_envs, 4]
                "motion_idx": priv_obs["motion_idx"].long(),              # [num_envs]
            }
            
            # Compute FK if enabled (replaces body_pos and body_lin_vel)
            if fk_calculator is not None:
                # Prepare FK inputs (batch mode)
                joint_pos_batch = robot_state["joint_pos"]  # [num_envs, 29]
                joint_vel_batch = robot_state["joint_vel"]  # [num_envs, 29]
                root_quat = robot_state["root_rot"]  # [num_envs, 4] in [w,x,y,z] format
                root_ang_vel = robot_state["body_ang_vel"][:, 0, :]  # [num_envs, 3] - pelvis angular vel
                root_lin_vel = robot_state["body_lin_vel"][:, 0, :]  # [num_envs, 3] - pelvis linear vel
                root_pos = robot_state["root_pos"]  # [num_envs, 3]
                
                # Compute FK (returns dict with torch tensors)
                fk_result = fk_calculator.compute_fk(
                    joint_pos=joint_pos_batch.cpu().numpy() if isinstance(joint_pos_batch, torch.Tensor) else joint_pos_batch,
                    joint_vel=joint_vel_batch.cpu().numpy() if isinstance(joint_vel_batch, torch.Tensor) else joint_vel_batch,
                    imu_pose=root_quat.cpu().numpy() if isinstance(root_quat, torch.Tensor) else root_quat,  # FK accepts quat [w,x,y,z]
                    imu_gyro=root_ang_vel.cpu().numpy() if isinstance(root_ang_vel, torch.Tensor) else root_ang_vel,
                    base_lin_vel=root_lin_vel.cpu().numpy() if isinstance(root_lin_vel, torch.Tensor) else root_lin_vel,
                    base_pos=root_pos.cpu().numpy() if isinstance(root_pos, torch.Tensor) else root_pos,
                    isaaclab_q_order=True,
                )
                
                # Replace body_pos and body_lin_vel with FK results
                # FK returns numpy arrays or torch tensors depending on batch size
                if isinstance(fk_result['body_pos'], torch.Tensor):
                    robot_state["body_pos"] = fk_result['body_pos']
                    robot_state["body_lin_vel"] = fk_result['body_lin_vel']
                else:
                    # Single sample returns numpy, convert to torch
                    robot_state["body_pos"] = torch.from_numpy(fk_result['body_pos']).to(robot_state["body_pos"].device)
                    robot_state["body_lin_vel"] = torch.from_numpy(fk_result['body_lin_vel']).to(robot_state["body_lin_vel"].device)
            
            # Convert to numpy for storage
            robot_state_np = {
                key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                for key, value in robot_state.items()
            }
            
            # Store current state and action for all envs (EXCEPT those that just reset)
            # This ensures we don't store the inconsistent first frame after reset
            for env_idx in range(cfg.task.num_envs):
                if not env_just_reset[env_idx]:  # Only store if env did NOT just reset
                    episode_data["act"][env_idx].append(actions[env_idx].cpu().numpy())
                    episode_data["body_pos"][env_idx].append(robot_state_np["body_pos"][env_idx])
                    episode_data["body_rot"][env_idx].append(robot_state_np["body_rot"][env_idx])
                    episode_data["body_lin_vel"][env_idx].append(robot_state_np["body_lin_vel"][env_idx])
                    episode_data["body_ang_vel"][env_idx].append(robot_state_np["body_ang_vel"][env_idx])
                    episode_data["joint_pos"][env_idx].append(robot_state_np["joint_pos"][env_idx])
                    episode_data["joint_vel"][env_idx].append(robot_state_np["joint_vel"][env_idx])
                    episode_data["root_pos"][env_idx].append(robot_state_np["root_pos"][env_idx])
                    episode_data["root_rot"][env_idx].append(robot_state_np["root_rot"][env_idx])
                    episode_data["motion_idx"][env_idx].append(robot_state_np["motion_idx"][env_idx])
                else:
                    # This env just reset, mark as no longer fresh after this iteration
                    env_just_reset[env_idx] = False
            
            # Apply action noise if enabled
            if noise_generator is not None:
                noise = noise_generator.sample()
                actions = actions + noise
                # Optionally clip actions to valid range
                # actions = torch.clamp(actions, -1.0, 1.0)

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
                    
                    if ep_length < 5: # Ignore 1-step episodes (caused by env init)
                        keep_episode = False
                        total_episodes_collected -= 1  # Don't count zero-length episodes
                        continue

                    # Filter out idle env timeouts (all tasks completed, env is just waiting)
                    if collection_mode == "deterministic":
                        # Check if env is idle from command manager metrics
                        if env_unwrapped.command_manager._terms["motion"].env_is_idle[env_idx] == -1:
                            # Collect this episode, mark as idle
                            env_unwrapped.command_manager._terms["motion"].env_is_idle[env_idx] = 1
                        elif env_unwrapped.command_manager._terms["motion"].env_is_idle[env_idx] == 1:
                            # Skip idle env timeouts
                            keep_episode = False
                            total_episodes_collected -= 1  # Don't count towards collected
                        # In deterministic mode, discard terminated (and not idle)
                        elif not infos["time_outs"][env_idx]:
                            keep_episode = False

                    # Filter by episode length: If is shorter than min length, discard
                    if ep_length < cfg.collection.min_episode_length:
                        keep_episode = False
                    
                    # Filter by mean reward (if specified)
                    if cfg.collection.min_mean_reward is not None and mean_reward < cfg.collection.min_mean_reward:
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
                            "motion_idx": np.array(episode_data["motion_idx"][env_idx], dtype=np.int64),  # [1] - scalar per episode
                        }
                        
                        # Add episode to buffer
                        buffer.add_episode(ep_data)
                        total_saved_steps += ep_length
                        total_episodes_saved += 1
                        
                        # Update progress bar based on mode
                        if collection_mode == "deterministic":
                            pbar.update(1)  # Count episodes
                        else:
                            pbar.update(ep_length)  # Count timesteps
                    
                    # Reset episode data for this environment
                    for key in episode_data:
                        episode_data[key][env_idx] = []
                    episode_lengths[env_idx] = 0
                    episode_rewards[env_idx] = 0
                    
                    # Mark this env as just reset (to skip storing first frame)
                    env_just_reset[env_idx] = True
                    
                    # Reset noise for finished environments if noise is enabled
                    if noise_generator is not None:
                        # Only reset noise for this specific environment
                        noise_generator.state[env_idx] = noise_generator.mu
            
    
    pbar.close()
    
    # Print statistics
    print(f"\n[INFO] Data collection complete!")
    print(f"[INFO] Total episodes collected: {total_episodes_collected}")
    print(f"[INFO] Total episodes saved: {total_episodes_saved}")
    print(f"[INFO] Total timesteps saved: {total_saved_steps}")
    print(f"[INFO] Episode retention rate: {total_episodes_saved/max(total_episodes_collected,1)*100:.1f}%")
    
    # Save dataset
    output_path = os.path.join(cfg.output.dir, cfg.output.zarr_name)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Saving dataset to {output_path}")
    
    # Sort episodes by motion_idx if requested (ensures motion1: eps0-N, motion2: eps(N+1)-2N, etc.)
    if cfg.output.get('sort_by_motion_idx', False):
        print(f"[INFO] Sorting {buffer.n_episodes} episodes by motion_idx...")
        
        # Extract motion_idx for each episode and create sorting index
        episode_motion_indices = []
        for ep_idx in range(buffer.n_episodes):
            episode = buffer.get_episode(ep_idx, copy=False)
            motion_idx = int(episode['motion_idx'][0])  # Get scalar value
            episode_motion_indices.append((ep_idx, motion_idx))
        
        # Sort by motion_idx (stable sort preserves order within same motion)
        sorted_episodes = sorted(episode_motion_indices, key=lambda x: x[1])
        
        # Create new buffer with sorted episodes
        sorted_buffer = ReplayBuffer.create_empty_numpy()
        for orig_idx, motion_idx in tqdm(sorted_episodes, desc="Sorting episodes"):
            episode_data = buffer.get_episode(orig_idx, copy=True)
            sorted_buffer.add_episode(episode_data)
        
        buffer = sorted_buffer
        print(f"[INFO] Episodes sorted by motion_idx (motion files in order)")
    
    # Set chunk length
    chunk_length = cfg.output.chunk_length if cfg.output.chunk_length > 0 else None
    
    # Save with optimal compression for disk storage
    buffer.save_to_path(
        output_path,
        chunks={'act': None, 'body_pos': None, 'body_rot': None,
                'body_lin_vel': None, 'body_ang_vel': None,
                'joint_pos': None, 'joint_vel': None,
                'root_pos': None, 'root_rot': None, 'motion_idx': None} if chunk_length is None else {},
        compressors=cfg.output.compressor,
    )
    
    # Save metadata
    metadata = {
        'task': cfg.task.name,
        'checkpoint': cfg.checkpoint.path,
        'motion_pattern': cfg.motion.pattern,
        'num_envs': int(cfg.task.num_envs),
        'collection_mode': collection_mode,
        'samples_per_motion': int(cfg.collection.samples_per_motion) if collection_mode == "deterministic" else None,
        'min_episode_length': int(cfg.collection.min_episode_length),
        'min_mean_reward': float(cfg.collection.min_mean_reward) if cfg.collection.min_mean_reward is not None else None,
        'len_to_save': int(cfg.collection.len_to_save),
        'total_episodes_collected': int(total_episodes_collected),
        'total_episodes_saved': int(total_episodes_saved),
        'total_timesteps': int(total_saved_steps),
        'n_episodes': int(buffer.n_episodes),
        'seed': int(cfg.task.seed),
        'noise_enabled': cfg.noise.enable,
        'noise_type': cfg.noise.type if cfg.noise.enable else None,
        'noise_params': {
            'theta': cfg.noise.theta,
            'mu': cfg.noise.mu,
            'sigma': cfg.noise.sigma,
            'dt': cfg.noise.dt,
        } if cfg.noise.enable and cfg.noise.type == "ou" else None,
        'fk_enabled': cfg.fk.use_fk,
        'fk_type': cfg.fk.fk_type if cfg.fk.use_fk else None,
        'fk_device': cfg.fk.fk_device if cfg.fk.use_fk else None,
        'creation_time': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
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
    main()
