"""Configuration for G1 robot with diffusion policy deployment."""

from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from textop_tracker.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from textop_tracker.tasks.tracking.tracking_env_cfg import TrackingEnvCfg, ProjGravObservationsCfg
import textop_tracker.tasks.tracking.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm

@configclass
class G1DiffusionDataCollectionEnvCfg(TrackingEnvCfg):
    """Configuration for G1 data collection - uses RL policy for inference, priv_obs for data."""
    
    @configclass
    class DataCollectionObservationsCfg:
        """Observations for data collection."""
        
        @configclass
        class PolicyCfg(ObsGroup):
            """Policy observations for RL inference (ProjGrav observations)."""
            # Projected gravity observations (for trained RL policy)
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
            projected_gravity = ObsTerm(func=mdp.projected_gravity)
            velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            actions = ObsTerm(func=mdp.last_action)
            motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
            motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
            # Future anchor observations
            motion_anchor_pos_b_future = ObsTerm(func=mdp.motion_anchor_pos_b_future, params={"command_name": "motion"})
            motion_anchor_ori_b_future = ObsTerm(func=mdp.motion_anchor_ori_b_future, params={"command_name": "motion"})
            
            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True
        
        @configclass
        class PrivilegedCfg(ObsGroup):
            """Privileged observations for data collection - raw robot state with noise."""
            # Raw robot state components with configurable noise for data collection
            body_pos = ObsTerm(
                func=mdp.robot_body_pos_all,
                noise=Unoise(n_min=-0.01, n_max=0.01)  # ±1cm noise on positions
            )
            body_rot = ObsTerm(
                func=mdp.robot_body_rot_all,
                noise=Unoise(n_min=-0.02, n_max=0.02)  # Small noise on quaternions
            )
            body_lin_vel = ObsTerm(
                func=mdp.robot_body_lin_vel_all,
                noise=Unoise(n_min=-0.05, n_max=0.05)  # ±5cm/s noise on velocities
            )
            body_ang_vel = ObsTerm(
                func=mdp.robot_body_ang_vel_all,
                noise=Unoise(n_min=-0.05, n_max=0.05)  # ±0.05 rad/s noise on angular velocities
            )
            joint_pos = ObsTerm(
                func=mdp.robot_joint_pos_all,
                noise=Unoise(n_min=-0.01, n_max=0.01)  # ±0.01 rad noise on joint positions
            )
            joint_vel = ObsTerm(
                func=mdp.robot_joint_vel_all,
                noise=Unoise(n_min=-0.05, n_max=0.05)  # ±0.05 rad/s noise on joint velocities
            )
            root_pos = ObsTerm(
                func=mdp.robot_root_pos,
                noise=Unoise(n_min=-0.01, n_max=0.01)  # ±1cm noise on root position
            )
            root_rot = ObsTerm(
                func=mdp.robot_root_rot,
                noise=Unoise(n_min=-0.02, n_max=0.02)  # Small noise on root quaternion
            )
            motion_idx = ObsTerm(
                func=mdp.robot_motion_idx
            )
            
            def __post_init__(self):
                self.enable_corruption = True  # Enable noise for data collection
                self.concatenate_terms = False  # Keep as dict for easy access
        
        policy: PolicyCfg = PolicyCfg()
        critic: PrivilegedCfg = PrivilegedCfg()  # Use critic group for privileged obs
    
    observations: DataCollectionObservationsCfg = DataCollectionObservationsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        
        # Set robot configuration (same as G1FlatProjGravObsEnvCfg)
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        
        # Align reward weights and pfail_thresholds to match pretrained checkpoint
        self.rewards.feet_slide.weight = -0.3
        self.rewards.soft_landing.weight = -0.0003
        self.rewards.overspeed.weight = -1.0
        self.rewards.overeffort.weight = -1.0
        
        self.rewards.feet_slide.params["pfail_threshold"] = 1.0
        self.rewards.soft_landing.params["pfail_threshold"] = 1.0
        self.rewards.overspeed.params["pfail_threshold"] = 1.0
        self.rewards.overeffort.params["pfail_threshold"] = 1.0
        
        # Set anchor body to pelvis (same as ProjGravObs config)
        self.commands.motion.anchor_body_name = "pelvis"
        
        # Enable motion end reset for data collection
        self.commands.motion.motion_end_reset = True
        
        # Configure body names for motion tracking
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
        
        # Disable domain randomization for clean data collection
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None


@configclass
class G1DiffusionEnvCfg(TrackingEnvCfg):
    """Configuration for G1 with diffusion policy deployment - evaluation only."""
    
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
        
        # # Set a dummy motion file - required for MotionCommand initialization
        # # but not actually used during diffusion policy execution (policy generates its own actions)
        # import glob
        # from pathlib import Path
        # ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent.parent.parent
        # motion_files = glob.glob(str(ROOT_DIR / "artifacts" / "Data10k-open" / "*" / "motion.npz"))
        # if motion_files:
        #     self.commands.motion.motion_files = [motion_files[0]]
        # else:
        #     raise FileNotFoundError(f"No motion.npz found in {ROOT_DIR / 'artifacts' / 'Data10k-open'}")
        
        # Placeholder motion file path (not used during diffusion policy execution)
        self.commands.motion.motion_files = ["/home/user/CodeSpace/Diffusion/TextOp/TextOpTracker/artifacts/Data10k-open/homejrhangmr_dataset_pbhc_contact_maskACCADFemale1General_c3dA1-Stand_posespkl/motion.npz"]
        self.commands.motion.freeze_motion = True
        # Disable domain randomization for clean evaluation
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.episode_length_s = 200.0

        self.terminations = self.terminations.replace(
            anchor_pos=DoneTerm(func=mdp.bad_anchor_pos_z_only,
                                params={
                                    "command_name": "motion",
                                    "threshold": 0.5
                                }),
            anchor_ori=None,
            ee_body_pos=None,
            motion_end=None,
        )

        # self.scene.robot.spawn.fix_base = True

        self.viewer.eye = (3.0, 0.0, 0.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # self.viewer.origin_type = "env"
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
