"""
Deterministic Motion Collection Command for Data Collection

This module implements a deterministic sampling strategy for data collection:
- M motion files × N samples per motion = M*N total episodes
- Each environment is assigned to a specific (motion_id, sample_id) pair
- No resampling on failure - failed episodes don't count as completed
- When all assignments complete, idle envs reset to default pose

This ensures complete coverage of the action distribution for all motions.
"""

from __future__ import annotations

from copy import deepcopy
import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any, Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MultiMotionLoader:
    """
    Loads multiple motion files with different lengths.
    Each motion is stored independently without padding.
    """

    def __init__(self,
                 motion_files: list[str],
                 body_indexes: Sequence[int],
                 device: str = "cpu"):
        assert len(motion_files) > 0, "motion_files cannot be empty"
        self.num_files = len(motion_files)
        self._body_indexes = body_indexes
        self.device = device

        # Store each motion's data in lists (no padding)
        self.joint_pos_list = []
        self.joint_vel_list = []
        self._body_pos_w_list = []
        self._body_quat_w_list = []
        self._body_lin_vel_w_list = []
        self._body_ang_vel_w_list = []
        self.fps_list = []
        self.file_lengths = []

        for motion_file in motion_files:
            assert os.path.isfile(
                motion_file), f"Invalid file path: {motion_file}"
            data = np.load(motion_file)

            self.fps_list.append(data["fps"])

            jp = torch.tensor(data["joint_pos"],
                              dtype=torch.float32,
                              device=device)
            jv = torch.tensor(data["joint_vel"],
                              dtype=torch.float32,
                              device=device)
            bp = torch.tensor(data["body_pos_w"],
                              dtype=torch.float32,
                              device=device)
            bq = torch.tensor(data["body_quat_w"],
                              dtype=torch.float32,
                              device=device)
            blv = torch.tensor(data["body_lin_vel_w"],
                               dtype=torch.float32,
                               device=device)
            bav = torch.tensor(data["body_ang_vel_w"],
                               dtype=torch.float32,
                               device=device)

            self.joint_pos_list.append(jp)
            self.joint_vel_list.append(jv)
            self._body_pos_w_list.append(bp)
            self._body_quat_w_list.append(bq)
            self._body_lin_vel_w_list.append(blv)
            self._body_ang_vel_w_list.append(bav)
            self.file_lengths.append(jp.shape[0])

        self.file_lengths = torch.tensor(self.file_lengths,
                                         dtype=torch.long,
                                         device=self.device)
        self.fps = self.fps_list[0]

    def get_motion_data_batch(
            self, motion_idx: int, time_steps_start: torch.Tensor,
            time_steps_end: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get a time range of motion data for a specific motion."""
        time_steps_tensor = torch.arange(time_steps_start,
                                         time_steps_end,
                                         device=self.device)  # type: ignore
        time_steps_tensor = torch.clamp(time_steps_tensor,
                                        torch.tensor(0, device=self.device),
                                        self.file_lengths[motion_idx] - 1)

        return {
            "joint_pos":
            self.joint_pos_list[motion_idx][time_steps_tensor],
            "joint_vel":
            self.joint_vel_list[motion_idx][time_steps_tensor],
            "body_pos_w":
            self._body_pos_w_list[motion_idx][time_steps_tensor]
            [:, self._body_indexes],
            "body_quat_w":
            self._body_quat_w_list[motion_idx][time_steps_tensor]
            [:, self._body_indexes],
            "body_lin_vel_w":
            self._body_lin_vel_w_list[motion_idx][time_steps_tensor]
            [:, self._body_indexes],
            "body_ang_vel_w":
            self._body_ang_vel_w_list[motion_idx][time_steps_tensor]
            [:, self._body_indexes],
        }


class MotionCollectionCommand(CommandTerm):
    """
    Deterministic motion collection command for M*N sampling strategy.
    
    Key features:
    - Each env assigned to (motion_id, sample_id) pair
    - Failed episodes don't count as completed
    - Idle envs reset to default pose when all tasks done
    """
    
    cfg: MotionCollectionCommandCfg

    def __init__(self, cfg: MotionCollectionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(
            self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(
            self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(self.robot.find_bodies(
            self.cfg.body_names, preserve_order=True)[0],
                                         dtype=torch.long,
                                         device=self.device)

        self.motion = MultiMotionLoader(self.cfg.motion_files,
                                        self.body_indexes.tolist(),
                                        device=self.device)

        self.buffer_length: int = np.min(
            [env.max_episode_length,
             (self.motion.file_lengths.max().item())]
        ) + self.cfg.future_steps  # type: ignore

        # === Collection-specific tracking ===
        self.num_motions = len(self.cfg.motion_files)
        self.samples_per_motion = self.cfg.samples_per_motion
        self.total_tasks = self.num_motions * self.samples_per_motion
        
        # Track completion status: [M, N] - True if (motion_i, sample_j) completed
        self.task_completed = torch.zeros(self.num_motions, self.samples_per_motion,
                                         dtype=torch.bool, device=self.device)
        
        # Track which task each env is working on: [num_envs]
        # Stores linear index: motion_idx * samples_per_motion + sample_idx
        self.env_task_assignment = torch.full((self.num_envs,), -1,
                                             dtype=torch.long, device=self.device)
        
        # Track if env is idle (all tasks done)
        self.env_is_idle = torch.zeros(self.num_envs, dtype=torch.bool,
                                       device=self.device)
        
        # Motion and time tracking (same as original)
        self.time_steps = torch.zeros(self.num_envs,
                                      dtype=torch.long,
                                      device=self.device)
        self.motion_idx = torch.zeros(self.num_envs,
                                      dtype=torch.long,
                                      device=self.device)
        self.sample_idx = torch.zeros(self.num_envs,
                                      dtype=torch.long,
                                      device=self.device)
        self.motion_length = torch.zeros(self.num_envs,
                                         dtype=torch.long,
                                         device=self.device)
        self.buffer_start_time = torch.zeros(self.num_envs,
                                             dtype=torch.long,
                                             device=self.device)

        # Initialize buffers
        self._init_buffers()

        # Transform buffers
        self.body_pos_relative_w = torch.zeros(self.num_envs,
                                               len(cfg.body_names),
                                               3,
                                               device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs,
                                                len(cfg.body_names),
                                                4,
                                                device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Metrics
        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs,
                                                       device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs,
                                                       device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs,
                                                      device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs,
                                                      device=self.device)
        self.metrics["tasks_completed"] = torch.zeros(self.num_envs,
                                                      device=self.device)
        self.metrics["tasks_total"] = torch.zeros(self.num_envs,
                                                  device=self.device)
        self.metrics["collection_progress"] = torch.zeros(self.num_envs,
                                                         device=self.device)

    def _init_buffers(self):
        """Initialize buffers for trajectory data."""
        joint_dim = self.motion.joint_pos_list[0].shape[1]
        body_dim = len(self.cfg.body_names)

        self.joint_pos_buffer = torch.zeros(self.num_envs,
                                            self.buffer_length,
                                            joint_dim,
                                            device=self.device)
        self.joint_vel_buffer = torch.zeros(self.num_envs,
                                            self.buffer_length,
                                            joint_dim,
                                            device=self.device)
        self.body_pos_w_buffer = torch.zeros(self.num_envs,
                                             self.buffer_length,
                                             body_dim,
                                             3,
                                             device=self.device)
        self.body_quat_w_buffer = torch.zeros(self.num_envs,
                                              self.buffer_length,
                                              body_dim,
                                              4,
                                              device=self.device)
        self.body_lin_vel_w_buffer = torch.zeros(self.num_envs,
                                                 self.buffer_length,
                                                 body_dim,
                                                 3,
                                                 device=self.device)
        self.body_ang_vel_w_buffer = torch.zeros(self.num_envs,
                                                 self.buffer_length,
                                                 body_dim,
                                                 3,
                                                 device=self.device)
        self.body_quat_w_buffer[:, :, :, 0] = 1.0

    def _update_buffers(self, env_ids: Optional[torch.Tensor] = None):
        """Update buffers from motion data."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if len(env_ids) == 0:
            return

        for env_id in env_ids:
            motion_data = self.motion.get_motion_data_batch(
                int(self.motion_idx[env_id].item()),
                self.buffer_start_time[env_id],
                self.buffer_start_time[env_id] + self.buffer_length,
            )

            self.joint_pos_buffer[env_id] = motion_data["joint_pos"]
            self.joint_vel_buffer[env_id] = motion_data["joint_vel"]
            self.body_pos_w_buffer[env_id] = motion_data["body_pos_w"]
            self.body_quat_w_buffer[env_id] = motion_data["body_quat_w"]
            self.body_lin_vel_w_buffer[env_id] = motion_data["body_lin_vel_w"]
            self.body_ang_vel_w_buffer[env_id] = motion_data["body_ang_vel_w"]

    def _get_next_task(self) -> Optional[tuple[int, int]]:
        """Get next incomplete (motion_idx, sample_idx) task."""
        incomplete_mask = ~self.task_completed
        if not incomplete_mask.any():
            return None
        
        # Find first incomplete task
        incomplete_indices = torch.nonzero(incomplete_mask, as_tuple=False)
        if len(incomplete_indices) == 0:
            return None
        
        motion_idx, sample_idx = incomplete_indices[0].tolist()
        return (motion_idx, sample_idx)

    def _assign_task_to_env(self, env_id: int, motion_idx: int, sample_idx: int):
        """Assign a specific task to an environment."""
        self.motion_idx[env_id] = motion_idx
        self.sample_idx[env_id] = sample_idx
        self.motion_length[env_id] = self.motion.file_lengths[motion_idx]
        
        # Store task assignment
        task_id = motion_idx * self.samples_per_motion + sample_idx
        self.env_task_assignment[env_id] = task_id
        
        # Always start from beginning for deterministic collection
        self.time_steps[env_id] = 0
        self.buffer_start_time[env_id] = 0
        
        # Update buffers
        self._update_buffers(torch.tensor([env_id], device=self.device))

    def _reset_to_idle(self, env_ids: torch.Tensor):
        """Reset idle environments to default pose to avoid termination."""
        if len(env_ids) == 0:
            return
        
        self.env_is_idle[env_ids] = True
        
        # Set to default/zero pose
        joint_dim = self.joint_pos_buffer.shape[2]
        
        # Default joint positions (zeros or specified default)
        default_joint_pos = torch.zeros(len(env_ids), joint_dim,
                                       device=self.device)
        default_joint_vel = torch.zeros(len(env_ids), joint_dim,
                                       device=self.device)
        
        # Default root state (standing at env origin)
        default_root_pos = self._env.scene.env_origins[env_ids].clone()
        default_root_pos[:, 2] += self.cfg.default_height  # Add height offset
        
        default_root_quat = torch.zeros(len(env_ids), 4, device=self.device)
        default_root_quat[:, 0] = 1.0  # Identity quaternion
        
        default_root_lin_vel = torch.zeros(len(env_ids), 3, device=self.device)
        default_root_ang_vel = torch.zeros(len(env_ids), 3, device=self.device)
        
        # Write to simulation
        self.robot.write_joint_state_to_sim(default_joint_pos,
                                           default_joint_vel,
                                           env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([default_root_pos, default_root_quat,
                      default_root_lin_vel, default_root_ang_vel], dim=-1),
            env_ids=env_ids,
        )
        
        # Set buffers to maintain default state
        for i, env_id in enumerate(env_ids):
            self.joint_pos_buffer[env_id, :] = default_joint_pos[i]
            self.joint_vel_buffer[env_id, :] = default_joint_vel[i]

    # === Properties (same as original) ===
    
    @property
    def command(self) -> torch.Tensor:
        cmd = torch.cat([self.motion_joint_pos, self.motion_joint_vel], dim=1)
        return cmd

    @property
    def joint_pos(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.joint_pos_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices]

    @property
    def joint_vel(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.joint_vel_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices]

    @property
    def body_pos_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.body_pos_w_buffer[
            torch.arange(self.num_envs, device=self.device),
            buffer_indices] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.body_quat_w_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.body_lin_vel_w_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.body_ang_vel_w_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return (self.body_pos_w_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices,
            self.motion_anchor_body_index] + self._env.scene.env_origins)

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.body_quat_w_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices,
            self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.body_lin_vel_w_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices,
            self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        buffer_indices = torch.clamp(self.time_steps - self.buffer_start_time,
                                     0, self.buffer_length - 1)
        return self.body_ang_vel_w_buffer[
            torch.arange(self.num_envs, device=self.device), buffer_indices,
            self.motion_anchor_body_index]

    @property
    def motion_joint_pos(self) -> torch.Tensor:
        """Future N-step joint positions reference."""
        if self.cfg.future_steps <= 1:
            return self.joint_pos
        else:
            current_indices = torch.clamp(
                self.time_steps - self.buffer_start_time, 0,
                self.buffer_length - 1)
            future_indices = current_indices[:, None] + torch.arange(
                self.cfg.future_steps, device=self.device)[None, :]
            future_indices = torch.clamp(future_indices, 0,
                                         self.buffer_length - 1)
            return self.joint_pos_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None],
                future_indices].view(self.num_envs, -1)

    @property
    def motion_joint_vel(self) -> torch.Tensor:
        """Future N-step joint velocities reference."""
        if self.cfg.future_steps <= 1:
            return self.joint_vel
        else:
            current_indices = torch.clamp(
                self.time_steps - self.buffer_start_time, 0,
                self.buffer_length - 1)
            future_indices = current_indices[:, None] + torch.arange(
                self.cfg.future_steps, device=self.device)[None, :]
            future_indices = torch.clamp(future_indices, 0,
                                         self.buffer_length - 1)
            return self.joint_vel_buffer[
                torch.arange(self.num_envs, device=self.device)[:, None],
                future_indices].view(self.num_envs, -1)

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        """Update tracking metrics."""
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w -
                                                      self.robot_anchor_pos_w,
                                                      dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos -
                                                     self.robot_joint_pos,
                                                     dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel -
                                                     self.robot_joint_vel,
                                                     dim=-1)
        
        # Collection progress metrics
        completed_count = self.task_completed.sum().item()
        self.metrics["tasks_completed"][:] = completed_count
        self.metrics["tasks_total"][:] = self.total_tasks
        self.metrics["collection_progress"][:] = completed_count / max(self.total_tasks, 1)

    def _resample_command(self, env_ids: torch.Tensor):
        """
        Resample command for environments that need new tasks.
        
        Key logic:
        - On episode end, check if it was successful (full length)
        - If successful, mark task as completed and assign next task
        - If failed (terminated early), reassign same task
        - If no tasks left, set to idle
        """
        if len(env_ids) == 0:
            return

        for env_id in env_ids:
            env_id_item = env_id.item()
            
            # Check if episode completed successfully (reached full motion length)
            # Motion completes when time_steps >= motion_length
            was_successful = self.time_steps[env_id_item] >= self.motion_length[env_id_item]
            
            if was_successful and self.env_task_assignment[env_id_item] >= 0:
                # Mark current task as completed
                motion_id = self.motion_idx[env_id_item].item()
                sample_id = self.sample_idx[env_id_item].item()
                self.task_completed[motion_id, sample_id] = True
                
                print(f"[Collection] Completed: Motion {motion_id}, Sample {sample_id} "
                      f"({self.task_completed.sum().item()}/{self.total_tasks})")
            
            # Try to get next task
            next_task = self._get_next_task()
            
            if next_task is None:
                # All tasks completed - set to idle
                self._reset_to_idle(torch.tensor([env_id_item], device=self.device))
                continue
            
            # Assign new task (or reassign same task if it failed)
            motion_idx, sample_idx = next_task
            self._assign_task_to_env(env_id_item, motion_idx, sample_idx)
            
            # Apply randomization to initial state
            self._apply_initialization_randomization(torch.tensor([env_id_item],
                                                                  device=self.device))

    def _apply_initialization_randomization(self, env_ids: torch.Tensor):
        """Apply pose/velocity/joint randomization on episode start."""
        if len(env_ids) == 0:
            return
        
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        # Pose randomization
        range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0],
                                      ranges[:, 1], (len(env_ids), 6),
                                      device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3],
                                                 rand_samples[:, 4],
                                                 rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])

        # Velocity randomization
        range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0],
                                      ranges[:, 1], (len(env_ids), 6),
                                      device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        # Joint randomization
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range,
                                    joint_pos.shape,
                                    device=self.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(joint_pos[env_ids],
                                        soft_joint_pos_limits[:, :, 0],
                                        soft_joint_pos_limits[:, :, 1])

        # Write to simulation
        self.robot.write_joint_state_to_sim(joint_pos[env_ids],
                                           joint_vel[env_ids],
                                           env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([
                root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids],
                root_ang_vel[env_ids]
            ], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        """Update command each step."""
        # Only advance time for non-idle envs
        active_mask = ~self.env_is_idle
        self.time_steps[active_mask] += 1
        
        # Find envs that reached motion end
        env_ids = torch.where(self.time_steps >= self.motion_length)[0]
        self._resample_command(env_ids)

        # Transform motion reference to robot frame (same as original)
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:,
                                                              None, :].repeat(
                                                                  1,
                                                                  len(self.cfg.
                                                                      body_names
                                                                      ), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(
            quat_mul(robot_anchor_quat_w_repeat,
                     quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization (simplified)."""
        pass

    def _debug_vis_callback(self, event):
        """Debug visualization callback (simplified)."""
        pass


@configclass
class MotionCollectionCommandCfg(CommandTermCfg):
    """Configuration for motion collection command."""

    class_type: type = MotionCollectionCommand

    asset_name: str = MISSING
    motion_files: list[str] = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    # Collection-specific
    samples_per_motion: int = 1  # N samples per motion
    default_height: float = 0.98  # Default height for idle pose

    # Randomization ranges
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # Future steps for N-step lookahead
    future_steps: int = 1
