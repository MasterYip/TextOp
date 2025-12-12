from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import sys
from pathlib import Path

# Add diffuse_cloc to path for importing trajectory utilities
DIFFUSE_CLOC_PATH = str(Path(__file__).parent.parent.parent.parent.parent.parent.parent / "diffuse_cloc")
if DIFFUSE_CLOC_PATH not in sys.path:
    sys.path.append(DIFFUSE_CLOC_PATH)

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from textop_tracker.tasks.tracking.mdp import MotionCommand
from isaaclab.envs.mdp.observations import SceneEntityCfg
from isaaclab.envs.mdp.observations import Articulation

# Import trajectory utilities from diffuse_cloc for state normalization
from diffusion_policy.utils.traj_utils import (
    quat_from_euler_xyz, get_euler_xyz, quat_mul, quat_rotate,
    box_minus, quat_rotate_inverse, box_plus, get_yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def joint_vel_rel_no_ankle(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel_rel = asset.data.joint_vel[:, :] - asset.data.default_joint_vel[:, :]
    # ankle_id  = asset_cfg.joint_ids
    print(joint_vel_rel[:, asset_cfg.joint_ids].norm(dim=-1).mean())
    # breakpoint()
    joint_vel_rel[:, asset_cfg.joint_ids] = 0
    return joint_vel_rel


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future N-step motion anchor position in body frame, flattened to vector."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    # Reshape future data: (num_envs, future_steps, 3) and (num_envs, future_steps, 4)
    future_anchor_pos_w = command.motion_anchor_pos.view(env.num_envs, -1, 3)
    future_anchor_quat_w = command.motion_anchor_quat.view(env.num_envs, -1, 4)

    # Expand robot anchor for broadcasting: (num_envs, future_steps, 3) and (num_envs, future_steps, 4)
    robot_anchor_pos_w_exp = command.robot_anchor_pos_w[:, None, :].expand(-1, future_anchor_pos_w.shape[1], -1)
    robot_anchor_quat_w_exp = command.robot_anchor_quat_w[:, None, :].expand(-1, future_anchor_quat_w.shape[1], -1)

    # Transform all future steps at once
    pos_b, _ = subtract_frame_transforms(
        robot_anchor_pos_w_exp,
        robot_anchor_quat_w_exp,
        future_anchor_pos_w,
        future_anchor_quat_w,
    )
    return pos_b.view(env.num_envs, -1)


def motion_anchor_ori_b_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future N-step motion anchor orientation in body frame, flattened to vector."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    # Reshape future data: (num_envs, future_steps, 3) and (num_envs, future_steps, 4)
    future_anchor_pos_w = command.motion_anchor_pos.view(env.num_envs, -1, 3)
    future_anchor_quat_w = command.motion_anchor_quat.view(env.num_envs, -1, 4)

    # Expand robot anchor for broadcasting: (num_envs, future_steps, 3) and (num_envs, future_steps, 4)
    robot_anchor_pos_w_exp = command.robot_anchor_pos_w[:, None, :].expand(-1, future_anchor_pos_w.shape[1], -1)
    robot_anchor_quat_w_exp = command.robot_anchor_quat_w[:, None, :].expand(-1, future_anchor_quat_w.shape[1], -1)

    # Transform all future steps at once
    pos_b, ori_b = subtract_frame_transforms(
        robot_anchor_pos_w_exp,
        robot_anchor_quat_w_exp,
        future_anchor_pos_w,
        future_anchor_quat_w,
    )

    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def diffusion_state_observation(env: ManagerBasedEnv) -> torch.Tensor:
    """
    Extract and normalize robot state for diffusion policy matching G1_Dataset format.
    
    Returns normalized observation matching data collection format:
        - body_pos_local: [num_envs, 30, 3] -> [num_envs, 90]
        - body_lin_vel_local: [num_envs, 30, 3] -> [num_envs, 90]
        - root_pos_local: [num_envs, 3]
        - root_rot_local: [num_envs, 3]
        - root_lin_vel_local: [num_envs, 3]
        - root_ang_vel_local: [num_envs, 3]
    
    Total: 192 dimensions (matching G1_Dataset normalization)
    """
    robot = env.scene["robot"]
    
    # Get body data for all bodies (30 bodies for G1)
    body_pos_w = robot.data.body_pos_w.clone()  # [num_envs, 30, 3]
    body_quat_w = robot.data.body_quat_w.clone()  # [num_envs, 30, 4]
    body_lin_vel_w = robot.data.body_lin_vel_w.clone()  # [num_envs, 30, 3]
    body_ang_vel_w = robot.data.body_ang_vel_w.clone()  # [num_envs, 30, 3]
    
    # Remove environment origins to get world-frame positions
    env_origins = env.scene.env_origins  # [num_envs, 3]
    body_pos_w = body_pos_w - env_origins[:, None, :]
    
    # Get root data (pelvis is first body, index 0)
    root_pos = body_pos_w[:, 0, :]  # [num_envs, 3]
    root_quat = body_quat_w[:, 0, :]  # [num_envs, 4]
    root_lin_vel = body_lin_vel_w[:, 0, :]  # [num_envs, 3]
    root_ang_vel = body_ang_vel_w[:, 0, :]  # [num_envs, 3]
    
    # Add time dimension for normalization (single timestep, current frame)
    # Shape becomes [B, 1, ...] for compatibility with G1Dataset normalization
    B = root_pos.shape[0]
    body_pos = body_pos_w.unsqueeze(1)  # [B, 1, 30, 3]
    root_pos_frame = root_pos.unsqueeze(1)  # [B, 1, 3]
    root_rot_frame = root_quat.unsqueeze(1)  # [B, 1, 4]
    body_lin_vel = body_lin_vel_w.unsqueeze(1)  # [B, 1, 30, 3]
    body_ang_vel = body_ang_vel_w.unsqueeze(1)  # [B, 1, 30, 3]
    
    # Compute yaw frame (gravity-aligned rotation)
    roll, pitch, yaw = get_euler_xyz(root_rot_frame.reshape(-1, 4))
    yaw_quat = quat_from_euler_xyz(roll * 0, pitch * 0, yaw).reshape(B, 1, 4)
    
    # Normalize body positions (remove root translation and rotate to yaw frame)
    J = body_pos.shape[2]  # 30 bodies
    body_pos_local = body_pos.clone()
    body_pos_local[:, :, :, :2] -= root_pos_frame[:, :, None, :2]  # Remove XY translation
    body_pos_local = quat_rotate_inverse(
        yaw_quat[:, :, None, :].repeat(1, 1, J, 1).reshape(-1, 4),
        body_pos_local.reshape(-1, 3),
    ).reshape(B, 1, J, 3)
    
    # Normalize body linear velocities (remove root velocity and rotate to yaw frame)
    body_lin_vel_local = body_lin_vel.clone() - root_lin_vel.unsqueeze(1).unsqueeze(2)
    body_lin_vel_local = quat_rotate_inverse(
        yaw_quat[:, :, None, :].repeat(1, 1, J, 1).reshape(-1, 4),
        body_lin_vel_local.reshape(-1, 3),
    ).reshape(B, 1, J, 3)
    
    # Normalize root position (relative to current position, rotated to yaw frame)
    root_pos_local = torch.zeros_like(root_pos_frame)  # Current frame is origin
    
    # Normalize root rotation (relative to yaw orientation)
    root_rot_local = box_minus(
        root_rot_frame.reshape(-1, 4),
        yaw_quat.reshape(-1, 4)
    ).reshape(B, 1, 3)
    
    # Normalize root linear velocity (rotate to yaw frame)
    root_lin_vel_local = quat_rotate_inverse(
        yaw_quat.reshape(-1, 4),
        root_lin_vel.unsqueeze(1).reshape(-1, 3)
    ).reshape(B, 1, 3)
    
    # Normalize root angular velocity (rotate to yaw frame)
    root_ang_vel_local = quat_rotate_inverse(
        yaw_quat.reshape(-1, 4),
        root_ang_vel.unsqueeze(1).reshape(-1, 3)
    ).reshape(B, 1, 3)
    
    # Concatenate all components (removing time dimension for single-step observation)
    obs = torch.cat([
        body_pos_local.reshape(B, 1, -1),  # [B, 1, 90]
        body_lin_vel_local.reshape(B, 1, -1),  # [B, 1, 90]
        root_pos_local.reshape(B, 1, -1),  # [B, 1, 3]
        root_rot_local.reshape(B, 1, -1),  # [B, 1, 3]
        root_lin_vel_local.reshape(B, 1, -1),  # [B, 1, 3]
        root_ang_vel_local.reshape(B, 1, -1),  # [B, 1, 3]
    ], dim=-1)  # [B, 1, 192]
    
    # Remove time dimension to get [B, 192]
    return obs.squeeze(1)
