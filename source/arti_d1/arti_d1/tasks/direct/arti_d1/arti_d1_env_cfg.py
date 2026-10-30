import os

from arti_d1.robots import ISAACLAB_ASSETS_DIR, ARTI_D1_CFG
from arti_d1.robots.arti_d1 import JOINT_DAMPING, JOINT_EFFORT_LIMIT, JOINT_STIFFNESS, JOINT_VELOCITY_LIMIT

from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import PvaCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg, preset


# 5 joints per leg (hip yaw / roll / pitch, knee, ankle) + 2 per arm (arm_m8, arm_m8crank_plate), like BDX.
ARTI_D1_NUM_JOINTS = 14
ARTI_D1_AMP_OBSERVATION_SPACE = 1 + 6 + 3 + 3 + ARTI_D1_NUM_JOINTS * 6 + ARTI_D1_NUM_JOINTS + 3 + 3
# Trailing `+ ARTI_D1_NUM_JOINTS` is the previous (clamped) action, fed back to the policy.
ARTI_D1_POLICY_OBSERVATION_SPACE = 6 + 3 + ARTI_D1_NUM_JOINTS * 6 + ARTI_D1_NUM_JOINTS + 3 + 3 + ARTI_D1_NUM_JOINTS
# root height (1) + true root lin/ang velocity in the heading frame (3 + 3)
# + averaged planar velocity and yaw rate used by the reward (3) + command tracking error (3).
ARTI_D1_CRITIC_PRIVILEGED_SPACE = 1 + 3 + 3 + 3 + 3
ARTI_D1_ACTION_SCALE_MARGIN = 1.2
# The reference only uses +-0.065 rad of hip yaw; a PD target that small cannot overcome the stance-foot twist.
ARTI_D1_HIP_YAW_ACTION_SCALE_MARGIN = 2.0
# Distance from the default pose (first reference frame) to the min / max over every reference clip
# (scripts/tuning/reference_stats.py, 1.36 s gait, 22 mm swing, hip_roll 10 deg inward stance). The arm plate only rises (lateral steps), so its negative range is
# 0 (floored by action_scale_min).
ARTI_D1_ACTION_NEGATIVE_SCALE = {
    "left_hip_yaw": 0.06259 * ARTI_D1_HIP_YAW_ACTION_SCALE_MARGIN,
    "left_hip_roll": 0.13403 * ARTI_D1_ACTION_SCALE_MARGIN,
    "left_hip_pitch": 0.33240 * ARTI_D1_ACTION_SCALE_MARGIN,
    "left_knee": 0.34321 * ARTI_D1_ACTION_SCALE_MARGIN,
    "left_ankle": 0.31999 * ARTI_D1_ACTION_SCALE_MARGIN,
    "right_hip_yaw": 0.06259 * ARTI_D1_HIP_YAW_ACTION_SCALE_MARGIN,
    "right_hip_roll": 0.19130 * ARTI_D1_ACTION_SCALE_MARGIN,
    "right_hip_pitch": 0.33299 * ARTI_D1_ACTION_SCALE_MARGIN,
    "right_knee": 0.36214 * ARTI_D1_ACTION_SCALE_MARGIN,
    "right_ankle": 0.32033 * ARTI_D1_ACTION_SCALE_MARGIN,
    "left_arm_m8": 0.52203 * ARTI_D1_ACTION_SCALE_MARGIN,
    "left_arm_m8crank_plate": 0.0,
    "right_arm_m8": 0.52203 * ARTI_D1_ACTION_SCALE_MARGIN,
    "right_arm_m8crank_plate": 0.0,
}
ARTI_D1_ACTION_POSITIVE_SCALE = {
    "left_hip_yaw": 0.06259 * ARTI_D1_HIP_YAW_ACTION_SCALE_MARGIN,
    "left_hip_roll": 0.17924 * ARTI_D1_ACTION_SCALE_MARGIN,
    "left_hip_pitch": 0.34435 * ARTI_D1_ACTION_SCALE_MARGIN,
    "left_knee": 0.16855 * ARTI_D1_ACTION_SCALE_MARGIN,
    "left_ankle": 0.33688 * ARTI_D1_ACTION_SCALE_MARGIN,
    "right_hip_yaw": 0.06259 * ARTI_D1_HIP_YAW_ACTION_SCALE_MARGIN,
    "right_hip_roll": 0.13592 * ARTI_D1_ACTION_SCALE_MARGIN,
    "right_hip_pitch": 0.39544 * ARTI_D1_ACTION_SCALE_MARGIN,
    "right_knee": 0.16866 * ARTI_D1_ACTION_SCALE_MARGIN,
    "right_ankle": 0.38652 * ARTI_D1_ACTION_SCALE_MARGIN,
    "left_arm_m8": 0.26180 * ARTI_D1_ACTION_SCALE_MARGIN,
    "left_arm_m8crank_plate": 0.36542 * ARTI_D1_ACTION_SCALE_MARGIN,
    "right_arm_m8": 0.26180 * ARTI_D1_ACTION_SCALE_MARGIN,
    "right_arm_m8crank_plate": 0.36542 * ARTI_D1_ACTION_SCALE_MARGIN,
}
# Dynamixel position-mode model (robots/arti_d1.py): legs XH540-W270, arms XM430-W350, factory Position P Gain 800.
ARTI_D1_JOINT_STIFFNESS = JOINT_STIFFNESS
ARTI_D1_JOINT_DAMPING = JOINT_DAMPING
ARTI_D1_JOINT_EFFORT_LIMIT = JOINT_EFFORT_LIMIT
ARTI_D1_JOINT_VELOCITY_LIMIT = JOINT_VELOCITY_LIMIT
# trunk_frame in arti_d1.urdf. The reference walk_com_height (0.2585) is this plus the 0.01225 m pelvis origin height.
ARTI_D1_TRUNK_OFFSET_FROM_PELVIS = (0.00964957, 0.0, 0.246248)
# IMU mount (Xsens MTi, humanoid_controller sensor.yaml): assumed at the trunk frame until the real mount is measured.
ARTI_D1_IMU_OFFSET_FROM_TRUNK = (0.0, 0.0, 0.0)
ARTI_D1_IMU_OFFSET_FROM_PELVIS = (
    ARTI_D1_TRUNK_OFFSET_FROM_PELVIS[0] + ARTI_D1_IMU_OFFSET_FROM_TRUNK[0],
    ARTI_D1_TRUNK_OFFSET_FROM_PELVIS[1] + ARTI_D1_IMU_OFFSET_FROM_TRUNK[1],
    ARTI_D1_TRUNK_OFFSET_FROM_PELVIS[2] + ARTI_D1_IMU_OFFSET_FROM_TRUNK[2],
)
# left_foot_frame / right_foot_frame in arti_d1.urdf (sole centre in the foot link; the legs are mirrored in y).
ARTI_D1_LEFT_FOOT_OFFSET_FROM_FOOT_LINK = (0.0135599213, -0.024, -0.0593316689)
ARTI_D1_RIGHT_FOOT_OFFSET_FROM_FOOT_LINK = (0.0135599213, 0.024, -0.0593316765)


# Reflected rotor inertia (kg*m^2) per joint type, added to the joint (high-ratio Dynamixel gearheads). The env's
# actuator model writes it (x a per-env random scale) at start-up (ArtiD1AmpEnvCfg.actuator_armature).
ARTI_D1_ACTUATOR_ARMATURE = {
    "hip_yaw": 0.005, "hip_roll": 0.005, "hip_pitch": 0.005, "knee": 0.005, "ankle": 0.005,
    "arm_m8": 0.002, "arm_m8crank_plate": 0.002,
}
# SolverKamino sizes each DoF's dynamics at build time and raises if armature later crosses zero ("recreate
# SolverKamino"), which the start-up armature write does on this USD (armature 0). Under newton_kamino the nominal
# armature is therefore authored at build time as well; every other backend keeps None (USD value, then the env write).
ARTI_D1_KAMINO_BUILD_ARMATURE = {
    f"{side}_{joint}": value for side in ("left", "right") for joint, value in ARTI_D1_ACTUATOR_ARMATURE.items()
}


@configclass
class ArtiD1PhysicsCfg(PresetCfg):
    """Physics backend presets, selected with physics=<name> (default: isaacsim_physx, the 2.x settings)."""

    # Exactly the Isaac Lab 2.x sim_utils.PhysxCfg(solver_type=1, bounce_threshold_velocity=0.2).
    isaacsim_physx: PhysxCfg = PhysxCfg(solver_type=1, bounce_threshold_velocity=0.2)
    # OvPhysxCfg exposes neither solver_type nor bounce_threshold_velocity: runs on its own defaults.
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    # Solver: core/locomotion/humanoid profile (biped on a plane); sim.dt is already 1/240, so one substep.
    # Contact: core/velocity legged profile, the ARTI-H1 port's values as the starting point (Newton defaults ke 2.5e3,
    # kd 100 let the soles sink); re-checked for ARTI-D1 with scripts/porting/compare_backends.py (PORTING_NOTES).
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=80,
            nconmax=25,
            cone="pyramidal",
            integrator="implicitfast",
            impratio=1,
        ),
        num_substeps=1,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )
    # Stub for the closed-loop variants (core/velocity pattern); not tuned.
    newton_kamino: NewtonCfg = NewtonCfg(solver_cfg=KaminoPADMMSolverCfg(max_contacts_per_world=64))
    default: PhysxCfg = isaacsim_physx


@configclass
class ArtiD1SceneCfg(InteractiveSceneCfg):
    """Robot, ground, IMU and light, cloned by the scene (Isaac Lab 3.x replaces _setup_scene cloning)."""

    robot: ArticulationCfg = ARTI_D1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=ARTI_D1_JOINT_STIFFNESS,
                damping=ARTI_D1_JOINT_DAMPING,
                # 2.x effort_limit (implicit: rated and solver limit) / velocity_limit_sim, under the 3.x names
                actuator_effort_limit=ARTI_D1_JOINT_EFFORT_LIMIT,
                joint_effort_limit=ARTI_D1_JOINT_EFFORT_LIMIT,
                joint_velocity_limit=ARTI_D1_JOINT_VELOCITY_LIMIT,
                armature=preset(default=None, newton_kamino=ARTI_D1_KAMINO_BUILD_ARMATURE),
            ),
        },
    )

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # MTi-3 sensor (the 2.x full-state Imu is Pva in 3.x). Filled in from the env cfg imu_* fields by
    # ArtiD1AmpEnv.__init__ so that CLI overrides of those fields still take effect; None = no sensor.
    mti3_imu: PvaCfg | None = None

    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )


@configclass
class ArtiD1EventCfg:
    """Domain randomization. Sampled once per environment at startup so the implicit-actuator
    and physics-material writes (CPU-side) never run inside the reset path."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # Combined with the plane's 1.0 via "average" -> effective 0.7..1.2.
            "static_friction_range": (0.4, 1.4),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            # Sampled independently, so dynamic can otherwise exceed static (unphysical).
            "make_consistent": True,
        },
    )

    add_pelvis_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # Pelvis carries 3.69 kg of the ~5.8 kg total; +-0.3 kg covers battery / payload drift.
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-0.3, 0.3),
            "operation": "add",
            "recompute_inertia": True,
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            # Wide: the Dynamixel gains are a datasheet model, not identified yet.
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.6, 1.3),
            "operation": "scale",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            # Walking speed is ~0.04-0.08 m/s; keep the push small against a full averaging window.
            "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.2, 0.2)},
        },
    )


@configclass
class ArtiD1AmpEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 15.0

    action_space = ARTI_D1_NUM_JOINTS
    # + gait clock (sin, cos) + commands (3, always last)
    observation_space = ARTI_D1_POLICY_OBSERVATION_SPACE + 2 + 3
    # Asymmetric critic: policy observation + privileged terms (see ArtiD1AmpEnv._get_critic_state). The tracking
    # reward uses an averaged true root velocity the policy (IMU only) cannot observe.
    state_space = ARTI_D1_POLICY_OBSERVATION_SPACE + 2 + 3 + ARTI_D1_CRITIC_PRIVILEGED_SPACE
    num_amp_observations = 2
    amp_observation_space = ARTI_D1_AMP_OBSERVATION_SPACE
    action_negative_scale = ARTI_D1_ACTION_NEGATIVE_SCALE
    action_positive_scale = ARTI_D1_ACTION_POSITIVE_SCALE
    # If set (rad), overrides the reference-derived scales above with the same +-range on every joint (bdx uses 0.6).
    # ARTI-H1: fresh runs with a uniform range skated; use it only on a warm start once the gait exists.
    uniform_action_scale = 0.0
    # Hard +-limit (rad) on both hip_roll joints in sim; 0 keeps the URDF limits (D1: -18 / +12 deg, mirrored).
    hip_roll_limit = 0.0
    # True = overwrite the sim joint limits with real_joint_limit_table (per joint type, both sides); False = URDF limits.
    # D1 uses the URDF limits (hip_roll is asymmetric and mirrored, so it cannot be one entry per joint type).
    real_joint_limits = False
    real_joint_limit_table = {}
    # Joint model applied every physics step in ArtiD1AmpEnv._apply_actuator_model, per joint type (name without the
    # left_ / right_ prefix); 0 disables a term. Dynamixel defaults, not identified on the robot yet.
    actuator_model = True
    # Command transport delay (ms): one Dynamixel bus (sync write + sync read of 14 servos at 1 Mbps, 8 ms cycle).
    actuator_delay_ms = {
        "hip_yaw": 10.0, "hip_roll": 10.0, "hip_pitch": 10.0, "knee": 10.0, "ankle": 10.0,
        "arm_m8": 10.0, "arm_m8crank_plate": 10.0,
    }
    # Reflected rotor inertia (kg*m^2) added to the joint (high-ratio Dynamixel gearheads).
    actuator_armature = dict(ARTI_D1_ACTUATOR_ARMATURE)
    # Coulomb friction with stiction (Nm), modelled implicitly: the PD target is shifted by coulomb / kp against the
    # motion, and frozen where the joint stuck while |kp * error| <= coulomb.
    actuator_coulomb = {
        "hip_yaw": 0.1, "hip_roll": 0.1, "hip_pitch": 0.1, "knee": 0.1, "ankle": 0.1,
        "arm_m8": 0.03, "arm_m8crank_plate": 0.03,
    }
    actuator_stick_velocity = 0.02  # rad/s
    # Per-env randomisation at start-up: delay + uniform(lo, hi) ms, armature and coulomb x uniform(lo, hi).
    actuator_delay_jitter_ms = (-6.0, 8.0)
    actuator_armature_scale_range = (0.3, 1.5)
    actuator_coulomb_scale_range = (0.3, 1.5)
    # Gaussian noise (rad/s) on the policy joint velocity obs; 0 = off.
    policy_joint_vel_noise_std = 0.3
    # Per control step probability that the joint obs repeat the previous step (missed bus reply); 0 = off.
    joint_obs_stale_prob = 0.05
    # Joints whose obs can go stale together (substring match): every Dynamixel shares one bus.
    joint_obs_stale_keywords = ("",)
    # Per-episode constant offset (rad, +-) on the policy joint position obs: assembly / zero offset error.
    joint_obs_offset_range = 0.02
    # Minimum +-range per side on top of the reference-derived scales (used when uniform_action_scale = 0).
    # ARTI-H1: with the plain reference scales the stance knee sagged and the other foot never lifted.
    action_scale_min = 0.25
    hip_yaw_action_scale_min = 0.10

    # Min / max of the recomputed (1.36 s averaged) reference commands plus ~3% margin
    # (scripts/tuning/reference_stats.py: x +-0.0519, y +-0.0209, yaw +-0.1198; feet 0.1354 m apart).
    command_linear_x_range = (-0.0534, 0.0534)
    command_linear_y_range = (-0.0215, 0.0215)
    command_yaw_range = (-0.1234, 0.1234)
    command_linear_velocity_scale = 0.5
    command_angular_velocity_scale = 0.25
    sample_commands_from_reference = True
    recompute_motion_commands = True
    use_average_velocities = True
    # One gait period (1.36 s): a shorter window leaves gait-frequency ripple in the tracked velocity.
    average_velocity_window_s = 1.36
    # Small relative to the command magnitude (|mean| x 0.028 m/s, yaw 0.058 rad/s), so standing still under a moving
    # command earns little of the tracking reward (the 1.25 s gait used 0.002 / 0.003 for |mean| x 0.042, yaw 0.061).
    linear_velocity_tracking_sigma = 0.0015
    angular_velocity_tracking_sigma = 0.003
    linear_velocity_xy_reward_scale = 0.5
    angular_velocity_z_reward_scale = 0.5
    # Regularization terms are NOT scaled by step_dt (see _get_rewards); these are the per-step magnitudes directly.
    # Size them against the exploration noise floor, not against a converged gait (ARTI-H1 lesson): a penalty paid on
    # the action noise alone drives log_std to its floor before the policy ever lifts a foot.
    # Torque term -scale * sum_j (tau_j / effort_limit_j)^2 and power term -scale * sum_j |tau_j * qd_j| (W), both
    # ramped in by the torque curriculum. The Dynamixel back-EMF damping (~3.4 Nm s/rad on the legs) is part of the
    # applied torque, so the scale is lower than on ARTI-H1 (-2.0).
    torque_reward_scale = -0.5
    joint_power_reward_scale = -0.01
    joint_effort_limit = ARTI_D1_JOINT_EFFORT_LIMIT
    # Smoothness: -scale * sum_j (qd_j / ref_rms_j)^2 (reference speed rms per joint from the clips, floored).
    joint_vel_reward_scale = -0.015
    joint_vel_rms_floor = 0.17  # rad/s (~10 deg/s), keeps hip yaw and the constant arm plate from dominating
    # -scale * sum qdd^2 (rad/s^2). Accelerations scale with 1/period^2: the 2.4 s gait used -2e-6, the 0.72 s -5e-7.
    joint_acc_reward_scale = -4.2e-7
    # Added to common_step_counter by every curriculum (support fade, torque, regularization), so a warm start
    # continues the schedules instead of restarting them at step 0 (set to the checkpoint step).
    curriculum_step_offset = 0
    # Fraction of the torque/power scales applied from step 0, ramped linearly to 1.0 between the two env-step counts.
    torque_curriculum_initial_fraction = 0.0
    torque_curriculum_steps = (100_000, 500_000)
    # Same ramp for action_rate, action-energy symmetry, base_flat / base_ang_vel / base_sway and the
    # stagger / step / knee / swing symmetry penalties (see _get_rewards).
    regularization_curriculum_initial_fraction = 0.0
    regularization_curriculum_steps = (60_000, 300_000)
    # Residual action on a nominal gait: PD target = default + nominal(clock phase) + scale * action, nominal = mean
    # reference joint offset per phase bin over all clips (stepping in place with the reference foot lift).
    # ARTI-H1: plain actions never lifted a foot. Deployment must add the same table (pt_to_onnx.py).
    nominal_gait_scale = 1.0
    nominal_gait_bins = 64
    # Sag/lag-compensated command table (bins x joints, robot joint order) from scripts/tuning/ilc_nominal.py.
    nominal_gait_table_file = os.path.join(ISAACLAB_ASSETS_DIR, "motions", "arti_d1_nominal_gait_command.npy")
    # Balance "training wheels": pelvis torque kp * sin(tilt) - kd * roll/pitch rate (Nm), full strength until the
    # first step count, then faded linearly to 0. Tipping moment of the whole robot at 0.2 rad is ~2.9 Nm (5.8 kg,
    # CoM 0.255 m). Set kp = 0 to disable (e.g. for play / eval / warm-started fine-tuning).
    balance_support_kp = 16.0
    balance_support_kd = 0.8
    balance_support_fade_steps = (150_000, 600_000)
    action_rate_reward_scale = -0.5
    alive_reward_scale = 0.1
    motion_time_scale = 1.0
    cycle_reference_motions = False
    # Penalize only the rolling left/right action-energy imbalance (permits the half-cycle phase offset of walking).
    symmetry_reward_scale = -0.05
    symmetry_window_s = 1.36
    # Non-saturating reward for moving along each commanded planar axis: clamp(v_a / cmd_a, -1, 1) averaged over the
    # axes with |cmd_a| > min_command, so standing still scores 0 under a moving command.
    signed_progress_reward_scale = 0.1
    signed_progress_min_command = 0.01
    # Seconds between mid-episode command switches (uniform range); (0, 0) disables.
    command_resample_interval_s = (4.0, 8.0)
    # Sole height below which a foot counts as in contact (the asset has no contact sensor).
    foot_contact_height = 0.005
    # Paid per foot at touchdown: max(clamp(air_time, max) - target, 0), only under a moving command.
    # Reference single support is 0.55 s.
    feet_air_time_reward_scale = 5.0
    feet_air_time_target_s = 0.0
    feet_air_time_max_s = 0.65
    # (stance_fraction_left - stance_fraction_right)^2 over a rolling window, under a moving command (off).
    contact_symmetry_reward_scale = 0.0
    contact_symmetry_window_s = 1.36
    # Gait clock (sin/cos in the policy observation), following the reference: phase_rate 0.73593 /s (1.359 s cycle,
    # 0.55 s single support), phase 0 = left single support start. From the reference foot contacts (reference_stats.py):
    # right foot swings over [0.0055, 0.4103), left over [0.5072, 0.9120), half-sine swing height peaking at
    # walk_foot_height 0.022 m. It is reset to the reference gait_phase of the RSI frame, so reference state and clock
    # agree at reset.
    gait_phase_rate = 0.73593
    # Standing: under a zero command (max |vx|, |vy|, |wz| <= stand_command_threshold) the clock runs on only to the
    # next rest phase (mid double support) and stays there, so the nominal stepping holds a two-feet stance; it starts
    # again from there with the next moving command. Empty = the clock always runs (stepping in place at zero command).
    gait_rest_phases = (0.4588, 0.9588)
    stand_command_threshold = 0.003
    # Standing (zero command) rewards: both soles down, and -scale * sum(joint_vel^2).
    stand_feet_contact_reward_scale = 0.5
    stand_joint_vel_reward_scale = 0.01
    # -scale * (|yaw rate| + |planar base speed|) while standing.
    stand_base_motion_reward_scale = 5.0
    gait_swing_start_phase = (0.5072, 0.0055)  # (left, right)
    gait_swing_duration_phase = 0.4048
    gait_swing_height = 0.022
    # 0.011 (0.008 at the 15 mm swing): a 5 mm lift already earns ~0.2, so a dragged foot sees a gradient toward lifting.
    gait_swing_height_sigma = 0.011
    # Fraction of feet whose contact state matches the clock (stance foot down, swing foot up).
    gait_contact_reward_scale = 1.0
    # exp(-(z - z_ref)^2 / sigma^2) averaged over the feet that should be swinging.
    gait_swing_height_reward_scale = 1.0
    # Share of the swing-height term paid linearly in sole height (clamped at gait_swing_height), rest = exp kernel.
    gait_swing_lift_weight = 0.5
    # -scale per foot that is still on the ground while the clock says it should swing (moving commands only).
    gait_swing_contact_penalty_scale = 1.0
    # |cycle-averaged x_right - x_left| in the heading frame (L1).
    foot_stagger_reward_scale = -8.0
    # Per touchdown: -scale * |step_k - step_(k-1)| (m).
    step_symmetry_reward_scale = -150.0
    # Sole contact patch half-size (m): the flat bottom face of the foot mesh (x +-0.040, y +-0.0173 about the sole
    # frame); the sole edge above it is a ~10 mm fillet out to x +-0.0525, y +-0.0295.
    foot_sole_half_length = 0.040
    foot_sole_half_width = 0.0173
    # -scale * sum_feet sin^2(sole tilt). Reference swing and stance soles are flat.
    foot_flat_reward_scale = -30.0
    # -scale * sum over feet off the ground of (roll rate^2 + pitch rate^2) (rad/s).
    swing_foot_ang_vel_reward_scale = -0.05
    # Knee = shin link origin. Reference knee lift (swing peak - stance) is 22.5 / 23.5 mm (22 mm swing, 10 deg hip_roll stance)
    # (scripts/tuning/reference_knee_lift.py).
    left_knee_body = "left_shin"
    right_knee_body = "right_shin"
    knee_lift_target = 0.0230
    # Per touchdown: scale * |swing_k - swing_(k-1)| (s).
    swing_symmetry_reward_scale = -20.0
    knee_lift_sigma = 0.008
    # Per touchdown: -scale * |lift_k - lift_(k-1)| (m).
    knee_symmetry_reward_scale = -300.0
    # Per touchdown: -scale * |ankle range of this stride - other leg's last stride range| (rad).
    ankle_pitch_symmetry_reward_scale = -20.0
    ankle_pitch_joint_names = ("left_ankle", "right_ankle")
    # Per touchdown: scale * exp(-((lift - target) / sigma)^2).
    knee_lift_reward_scale = 2.0
    # Pelvis level: -scale * |projected gravity xy|^2 (= sin^2 tilt). Reference roll/pitch = 0.
    base_flat_reward_scale = -60.0
    # -scale * (roll_rate^2 + pitch_rate^2) in rad/s.
    base_ang_vel_xy_reward_scale = -0.5
    # -scale * (lateral vel - its window mean)^2 in m/s.
    base_lateral_sway_reward_scale = -10.0
    # Sum over stance feet of |v_xy|^2 at the sole (skating penalty).
    foot_slip_reward_scale = -8.0

    early_termination = True
    # Trunk frame height (reference 0.255 m).
    termination_height = 0.17
    # Trunk tilt from vertical, in radians. Height alone lets the robot survive in heavily tipped poses.
    termination_max_tilt_rad = 0.8
    reference_body = "pelvis"
    left_toe_body = "left_foot_link"
    right_toe_body = "right_foot_link"
    left_toe_body_offset = ARTI_D1_LEFT_FOOT_OFFSET_FROM_FOOT_LINK
    right_toe_body_offset = ARTI_D1_RIGHT_FOOT_OFFSET_FROM_FOOT_LINK
    termination_body = "pelvis"
    termination_body_offset = ARTI_D1_TRUNK_OFFSET_FROM_PELVIS
    # "random" enables reference state initialization (RSI): episodes start at a random
    # time in the clip. "random_start" pins every reset to t=0 of the clip.
    reset_strategy = "random"
    # Fraction of RSI resets forced to t=0 of the clip (standstill start).
    reset_standstill_probability = 0.2
    use_isaac_imu_sensor = True
    imu_backend = "isaaclab"
    imu_body = "pelvis"
    imu_sensor_name = "mti3_imu"
    imu_prim_path = "/World/envs/env_.*/Robot/pelvis"
    imu_parent_path_template = "/World/envs/env_{env_id}/Robot/{body}"
    imu_sensor_period_s = 0.0
    imu_linear_acceleration_filter_size = 1
    imu_angular_velocity_filter_size = 1
    imu_orientation_filter_size = 1
    imu_translation = ARTI_D1_IMU_OFFSET_FROM_PELVIS
    imu_orientation_wxyz = (1.0, 0.0, 0.0, 0.0)
    imu_read_gravity = True
    # Xsens MTi model (tasks/utils/mti3_imu.py): datasheet white noise, bias random walk, transport delay.
    imu_latency_s = 0.01  # -> 1 control step at 60 Hz
    mti3_gyro_bias_walk_std_rad_s = 0.0005
    mti3_accel_bias_walk_std_m_s2 = 0.002
    mti3_gyro_noise_std_rad_s = 0.01
    mti3_gyro_bias_range_rad_s = (-0.01, 0.01)
    imu_gyro_extra_delay_steps = 1
    # Per-episode constant IMU roll / pitch mounting error (rad, +-); 0 = off.
    mti3_mount_tilt_range_rad = 0.02
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=decimation,
        physics=ArtiD1PhysicsCfg(),
    )

    # robot, terrain, IMU and light live in ArtiD1SceneCfg. 2.x replicate_physics=True / clone_in_fabric=False:
    # the first is deprecated and ignored in 3.x, the second was removed.
    scene: ArtiD1SceneCfg = ArtiD1SceneCfg(num_envs=4096, env_spacing=1.0)

    events: ArtiD1EventCfg = ArtiD1EventCfg()

    dt = 1 / 240 * 4
    motion_folder_path = os.path.join(ISAACLAB_ASSETS_DIR, "motions", "arti_d1")
ArtiD1EnvCfg = ArtiD1AmpEnvCfg
