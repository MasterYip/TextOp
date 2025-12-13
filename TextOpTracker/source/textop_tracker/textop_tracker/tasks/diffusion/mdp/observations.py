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

from textop_tracker.tasks.diffusion.mdp import MotionCommand
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


def extract_robot_state(env: ManagerBasedEnv) -> dict:
    """
    Extract raw robot state data from the environment.
    
    This function provides the raw data in the same format used during data collection.
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
        "body_pos": body_pos_w,  # [num_envs, 30, 3]
        "body_rot": body_quat_w,  # [num_envs, 30, 4]
        "body_lin_vel": body_lin_vel_w,  # [num_envs, 30, 3]
        "body_ang_vel": body_ang_vel_w,  # [num_envs, 30, 3]
        "joint_pos": joint_pos,  # [num_envs, 29]
        "joint_vel": joint_vel,  # [num_envs, 29]
        "root_pos": root_pos,  # [num_envs, 3]
        "root_rot": root_rot,  # [num_envs, 4]
    }


def diffusion_state_observation(env: ManagerBasedEnv) -> torch.Tensor:
    """
    Extract and normalize robot state for diffusion policy matching G1_Dataset format.
    
    This function ensures data consistency by:
    1. Using extract_robot_state() to get raw data (same as data collection)
    2. Applying G1_Dataset.state_normalize() (same as training)
    
    Returns normalized observation matching G1_Dataset format:
        - body_pos_local: [num_envs, 30, 3] -> [num_envs, 90]
        - body_lin_vel_local: [num_envs, 30, 3] -> [num_envs, 90]
        - root_pos_local: [num_envs, 3]
        - root_rot_local: [num_envs, 3]
        - root_lin_vel_local: [num_envs, 3]
        - root_ang_vel_local: [num_envs, 3]
    
    Total: 192 dimensions (matching G1_Dataset normalization)
    """
    # Import G1_Dataset for normalization (same as training)
    from diffusion_policy.dataset.g1_offline_dataset import G1_Dataset
    
    # Step 1: Extract raw robot state (same as data collection)
    robot_state = extract_robot_state(env)
    
    # Step 2: Prepare data for G1_Dataset.state_normalize
    # Add time dimension (single timestep, current frame)
    B = robot_state["root_pos"].shape[0]
    
    # Reshape data to match expected input format [B, H, ...] where H=1 for single timestep
    root_pos_frame = robot_state["root_pos"].unsqueeze(1)  # [B, 1, 3]
    root_rot_frame = robot_state["root_rot"].unsqueeze(1)  # [B, 1, 4]
    body_pos = robot_state["body_pos"].unsqueeze(1)  # [B, 1, 30, 3]
    body_rot = robot_state["body_rot"].unsqueeze(1)  # [B, 1, 30, 4]
    body_lin_vel = robot_state["body_lin_vel"].unsqueeze(1)  # [B, 1, 30, 3]
    body_ang_vel = robot_state["body_ang_vel"].unsqueeze(1)  # [B, 1, 30, 3]
    joint_pos = robot_state["joint_pos"].unsqueeze(1)  # [B, 1, 29]
    
    # Step 3: Apply G1_Dataset normalization (same as training)
    # nominal_frame_idx=0 since we only have one timestep (current frame)
    # ee_idxs is not used in G1_Dataset.state_normalize (only in G1_Dataset_EE)
    obs_normalized = G1_Dataset.state_normalize(
        root_pos_frame=root_pos_frame,
        root_rot_frame=root_rot_frame,
        body_pos=body_pos,
        body_rot=body_rot,
        body_lin_vel=body_lin_vel,
        body_ang_vel=body_ang_vel,
        nominal_frame_idx=0,  # Current frame is the reference
        ee_idxs=G1_Dataset.ee_idxs(),
        joint_pos=joint_pos,
        return_raw=False,
    )  # Returns [B, 1, 192]
    
    # Remove time dimension to get [B, 192]
    return obs_normalized.squeeze(1)
