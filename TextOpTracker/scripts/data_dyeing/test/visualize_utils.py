"""
Visualization utilities for G1 motion data.

Note: Expects body positions in MotionCLIP ordering (semantic).
If you have IsaacLab ordering (alphabetical), use body_index_mapping.remap_isaaclab_to_motionclip first.
"""

import numpy as np
from body_index_mapping import remap_isaaclab_to_motionclip


# G1 Robot kinematic chain (30 bodies)
g1_kinematic_chain = [
    [0, 1, 2, 3, 4, 5, 6],        # Left leg: pelvis -> left foot
    [0, 7, 8, 9, 10, 11, 12],     # Right leg: pelvis -> right foot
    [0, 13, 14, 15],              # Torso: pelvis -> torso
    [15, 16, 17, 18, 19, 20, 21, 22],  # Left arm: torso -> left wrist
    [15, 23, 24, 25, 26, 27, 28, 29],  # Right arm: torso -> right wrist
]

# Colors for each kinematic chain
colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]


class G1MotionVisualizer:
    """
    Visualizer for G1 robot motion using matplotlib 3D.
    
    Note: Expects body positions in MotionCLIP ordering (semantic).
    Set auto_remap=True to automatically convert from IsaacLab ordering.
    """
    
    def __init__(self, auto_remap=False):
        """
        Args:
            auto_remap: If True, automatically remap from IsaacLab to MotionCLIP ordering
        """
        self.kinematic_tree = g1_kinematic_chain
        self.colors = colors_blue
        self.auto_remap = auto_remap
    
    def draw_frame(self, ax, body_positions, frame_idx):
        """
        Draw a single frame of G1 motion.
        
        Args:
            ax: Matplotlib 3D axes
            body_positions: Body positions array [T, 30, 3]
            frame_idx: Frame index to draw
        
        Note:
            If auto_remap=True, body_positions is assumed to be in IsaacLab order
            and will be automatically remapped to MotionCLIP order.
        """
        # Remap if needed
        if self.auto_remap:
            data_dict = {'body_pos': body_positions}
            data_dict = remap_isaaclab_to_motionclip(data_dict)
            body_positions = data_dict['body_pos']
        # Extract positions for current frame
        positions_world = body_positions[frame_idx]  # (30, 3)
        
        # Center pelvis to origin for clearer articulation view
        pelvis = positions_world[0:1]
        positions = positions_world - pelvis  # centered skeleton
        
        # Draw kinematic chains
        for chain, color in zip(self.kinematic_tree, self.colors):
            chain_pos = positions[chain]  # (len(chain), 3)
            ax.plot(chain_pos[:, 0], chain_pos[:, 1], chain_pos[:, 2],
                   linewidth=4.0, color=color)
        
        # Draw joints as scatter points
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='black', s=20, alpha=0.6)
        
        # Highlight pelvis (root)
        ax.scatter(positions[0:1, 0], positions[0:1, 1], positions[0:1, 2],
                  c='red', s=40)
        
        # Draw ground reference using pelvis trajectory
        if frame_idx > 0:
            pelvis_traj = body_positions[:frame_idx+1, 0, :]  # (frame_idx+1, 3)
            pelvis_traj_rel = pelvis_traj - pelvis  # relative to current pelvis
            ax.plot(pelvis_traj_rel[:, 0],
                   pelvis_traj_rel[:, 1],
                   pelvis_traj_rel[:, 2],
                   color='gray', linewidth=2.0, alpha=0.6)
