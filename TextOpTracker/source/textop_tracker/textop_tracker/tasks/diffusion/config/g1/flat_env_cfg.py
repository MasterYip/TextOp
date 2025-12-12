from isaaclab.utils import configclass

from textop_tracker.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from textop_tracker.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from textop_tracker.tasks.tracking.tracking_env_cfg import TrackingEnvCfg, PrivPrivObservationsCfg, PropPropObservationsCfg, NoisePrivObservationsCfg, ProjGravObservationsCfg


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
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


@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatProjGravObsEnvCfg(G1FlatEnvCfg):
    observations: ProjGravObservationsCfg = ProjGravObservationsCfg()


@configclass
class G1FlatProjGravObsEnvCfg_LargeHand(G1FlatProjGravObsEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        from textop_tracker.robots.g1 import sim_utils, ASSET_DIR

        self.scene.robot = G1_CYLINDER_CFG.replace(spawn=sim_utils.UrdfFileCfg(
            fix_base=False,
            replace_cylinders_with_capsules=True,
            asset_path=
            f"{ASSET_DIR}/unitree_description/urdf/g1/main_largehand.urdf",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0, damping=0)),
        )).replace(prim_path="{ENV_REGEX_NS}/Robot")

    ...


@configclass
class G1FlatProjGravObsEnvCfg_LargeHandHeavy(G1FlatProjGravObsEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        from textop_tracker.robots.g1 import sim_utils, ASSET_DIR

        self.scene.robot = G1_CYLINDER_CFG.replace(spawn=sim_utils.UrdfFileCfg(
            fix_base=False,
            replace_cylinders_with_capsules=True,
            asset_path=
            f"{ASSET_DIR}/unitree_description/urdf/g1/main_largehand_heavy.urdf",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0, damping=0)),
        )).replace(prim_path="{ENV_REGEX_NS}/Robot")

    ...

    ...


@configclass
class G1FlatPropPropObsEnvCfg(G1FlatEnvCfg):
    observations: PropPropObservationsCfg = PropPropObservationsCfg()


@configclass
class G1FlatPrivPrivObsEnvCfg(G1FlatEnvCfg):
    observations: PrivPrivObservationsCfg = PrivPrivObservationsCfg()


@configclass
class G1FlatNoisePrivObsEnvCfg(G1FlatEnvCfg):
    observations: NoisePrivObservationsCfg = NoisePrivObservationsCfg()


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
