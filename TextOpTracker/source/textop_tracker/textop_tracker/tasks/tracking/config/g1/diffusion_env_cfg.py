"""Configuration for G1 robot with diffusion policy deployment."""

from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

from textop_tracker.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from textop_tracker.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
import textop_tracker.tasks.tracking.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
import textop_tracker.tasks.tracking.mdp as mdp

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
        self.commands.motion.motion_files = ["/home/user/CodeSpace/HumanoidCtrl/TextOp/TextOpTracker/artifacts/Data10k-open/homejrhangmr_dataset_pbhc_contact_maskACCADFemale1General_c3dA1-Stand_posespkl/motion.npz"]
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
