"""Configuration for G1 robot with diffusion policy deployment."""

from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

from textop_tracker.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from textop_tracker.tasks.diffusion.tracking_env_cfg import TrackingEnvCfg
import textop_tracker.tasks.diffusion.mdp as mdp


@configclass
class G1DiffusionEnvCfg(TrackingEnvCfg):
    """Configuration for G1 with diffusion policy - matches data collection format."""
    
    @configclass
    class DiffusionObservationsCfg:
        """Observations for diffusion policy - single observation matching G1_Dataset format."""
        
        @configclass
        class PolicyCfg(ObsGroup):
            """Policy observations - 192-dim normalized state."""
            diffusion_state = ObsTerm(func=mdp.diffusion_state_observation)
            
            def __post_init__(self):
                self.enable_corruption = False  # No noise for diffusion deployment
                self.concatenate_terms = True
        
        policy: PolicyCfg = PolicyCfg()
    
    observations: DiffusionObservationsCfg = DiffusionObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        
        # Set robot configuration
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        
        # Configure motion command
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        
        # Disable domain randomization for clean evaluation
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
