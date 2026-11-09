"""ARTI-D1 velocity-tracking walk, ported from pollen-robotics/microduck_rl (Mjlab-Velocity-Flat-MicroDuck on mjlab 1.3.0).

No reference motion, no gait clock, no nominal gait table: plain velocity-command tracking with the microduck reward
stack (mjlab velocity base + microduck overrides), observation layout, noise / delays and domain randomization. Kept
from the ARTI-D1 AMP task: robot USD, Dynamixel joint order, PD gains and the per-physics-step Dynamixel model (bus delay,
Coulomb friction with stiction, voltage saturation) that stands in for microduck's BAM actuator model.

Rescaled for ARTI-D1 (microduck is 25 cm / 0.8 kg with XL330s; ARTI-D1 is ~0.5 m / 5.8 kg with XH540s at 3.14 rad/s):
command ranges, tracking stds, swing height targets, air-time window and push size. Every rescaled value says from what.
Arms: held at 0 (the deployed controller fixes them at 0, humanoid_controller robot.yaml fixed_joints); the policy has
10 leg outputs and sees only the leg joints (Dynamixel order with the arms removed).
Linear velocity (command tracking, critic obs) is taken at the trunk point, not the pelvis link origin (foot level).
"""

import math

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from .arti_d1_env_cfg import ARTI_D1_ACTUATOR_ARMATURE, ARTI_D1_JOINT_EFFORT_LIMIT, ArtiD1PhysicsCfg, ArtiD1SceneCfg

ARTI_D1_VEL_NUM_JOINTS = 10  # leg joints
# actor: base ang vel (3) + projected gravity (3) + leg joint pos rel (10) + leg joint vel (10) + last action (10) + command (3)
ARTI_D1_VEL_POLICY_OBS = 3 + 3 + 10 + 10 + 10 + 3
# critic: actor terms without noise / delay / bias + base lin vel (3) + foot height (2) + air time (2) + contact (2)
ARTI_D1_VEL_CRITIC_OBS = ARTI_D1_VEL_POLICY_OBS + 3 + 2 + 2 + 2


@configclass
class ArtiD1VelEventCfg:
    """Startup domain randomization (microduck: foot friction, mass/inertia, encoder bias; IMU tilt and CoM are in the env)."""

    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
            # plane 1.0 combined by "average": effective 0.7 .. 1.3 = microduck foot_friction (0.7, 1.3)
            "static_friction_range": (0.4, 1.6),
            "dynamic_friction_range": (0.4, 1.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # microduck MASS_INERTIA_RANDOMIZATION_RANGE (0.95, 1.05) on mass and inertia together
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            # microduck COM_RANDOMIZATION_RANGE +-3 mm (ramped to +-8 mm); per env at start-up here
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "com_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.005, 0.005)},
        },
    )


@configclass
class ArtiD1VelEnvCfg(DirectRLEnvCfg):
    # 60 Hz policy (humanoid_controller ARTI-D1 period), 240 Hz physics. microduck: 50 Hz.
    decimation = 4
    episode_length_s = 20.0  # mjlab velocity default
    action_space = ARTI_D1_VEL_NUM_JOINTS
    observation_space = ARTI_D1_VEL_POLICY_OBS
    state_space = ARTI_D1_VEL_CRITIC_OBS

    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=decimation, physics=ArtiD1PhysicsCfg())
    scene: ArtiD1SceneCfg = ArtiD1SceneCfg(num_envs=4096, env_spacing=1.0)
    events: ArtiD1VelEventCfg = ArtiD1VelEventCfg()

    # ---- actions: target = default + scale * a (mjlab JointPositionAction, use_default_offset). microduck scale 1.0 on
    # low-kp XL330s; ARTI-D1 XH540 kp ~49 Nm/rad tracks closely, 0.5 = mjlab's humanoid default.
    action_scale = 0.5
    # Arms held at this target (rad); no policy output.
    arm_hold_target = {"left_arm_m8": 0.0, "right_arm_m8": 0.0, "left_arm_m8crank_plate": 0.0, "right_arm_m8crank_plate": 0.0}

    # ---- Dynamixel joint model, every physics step (same as ArtiD1AmpEnvCfg; BAM stand-in)
    actuator_delay_ms = 10.0
    actuator_delay_jitter_ms = (-6.0, 8.0)
    actuator_armature = dict(ARTI_D1_ACTUATOR_ARMATURE)
    actuator_armature_scale_range = (0.9, 1.1)  # microduck ARMATURE_RANDOMIZATION_RANGE
    actuator_coulomb = {
        "hip_yaw": 0.1, "hip_roll": 0.1, "hip_pitch": 0.1, "knee": 0.1, "ankle": 0.1,
        "arm_m8": 0.03, "arm_m8crank_plate": 0.03,
    }
    actuator_coulomb_scale_range = (0.9, 1.1)  # microduck JOINT_FRICTION_RANDOMIZATION_RANGE
    actuator_stick_velocity = 0.02
    joint_effort_limit = ARTI_D1_JOINT_EFFORT_LIMIT

    # ---- commands (microduck UniformVelocityCommand, heading off, fixed ranges, turn-in-place bucket)
    # microduck lin x +-0.4, y +-0.3, yaw +-1.0 for a 25 cm robot (~1.6 / 1.2 leg lengths per s); ARTI-D1 legs are
    # ~0.26 m with servos at 3.14 rad/s.
    command_lin_x_range = (-0.12, 0.15)
    command_lin_y_range = (-0.08, 0.08)
    command_yaw_range = (-0.4, 0.4)
    command_resample_s = (3.0, 8.0)
    turn_in_place_fraction = 0.15
    lateral_fraction = 0.15  # pure sideways commands, |vy| in [0.4, 1] x range max
    # standing (exactly zero command) env fraction by env step (microduck standing_envs curriculum, iter x 24)
    standing_stages = ((0, 0.02), (500 * 24, 0.05), (750 * 24, 0.1), (1000 * 24, 0.15), (1500 * 24, 0.2), (2000 * 24, 0.25))
    command_threshold = 0.01  # microduck command_threshold for gait terms / pose walking_threshold

    # ---- rewards (weights x step_dt, like mjlab's RewardManager)
    track_lin_vel_weight = 2.0
    track_lin_vel_std = 0.08  # m/s; microduck sqrt(0.1)=0.32 for +-0.4 commands, scaled by the command range (v3/v4: 0.06, no tracking learned)
    # v3: extra lateral term (v1/v2 never learned lin y: a 0.06 m/s miss still paid exp(-.56)=.57 of the joint xy term)
    track_lat_vel_weight = 1.0  # v8 lateral fine-tune from v7 cont1 240k; v6/v7 off; v3-v5 1.0 (from scratch, never tracked)
    track_lat_vel_std = 0.06  # v3/v4: 0.03
    # v3: alive bonus (per s) + one-off termination penalty (seed 42 v2 run collapsed into "fall early" optimum)
    alive_weight = 0.0  # v6 off (v3-v5: 1.0)
    termination_weight = 0.0  # v6 off (v3-v5: -10)
    # v4: swing-foot xy speed penalty (excess over the limit, squared) and left/right symmetry loss on the policy mean
    swing_speed_limit = 0.5  # m/s
    swing_speed_weight = -1.0  # v7 / v7 cont1 (fine-tune from v6 240k); v6 / v6 cont1 0; v4/v5 -5 (never tracked, collapsed)
    symmetry_loss_coef = 0.2  # scripts/skrl/train.py; 0 = off (v4/v5: 0.5)
    # both start at iter 500 (from the start they made the stand-up phase collapse on 2 of 3 seeds)
    symmetry_kick_start_step = 500 * 24
    track_ang_vel_weight = 2.0
    track_ang_vel_std = 0.28  # rad/s; microduck sqrt(0.5)=0.71 for +-1.0
    upright_weight = 2.0
    upright_std = math.sqrt(0.05)
    pose_weight = 1.0
    # per-joint std (rad) of exp(-mean((q - q0)^2 / std^2)); microduck values (hip_roll walking 0.05 -> 0.1: ARTI-D1 shifts
    # its weight with hip_roll, the demo uses 15 deg of it)
    pose_std_standing = {"hip_yaw": 0.1, "hip_roll": 0.05, "hip_pitch": 0.15, "knee": 0.15, "ankle": 0.1}
    pose_std_walking = {"hip_yaw": 0.3, "hip_roll": 0.1, "hip_pitch": 0.4, "knee": 0.4, "ankle": 0.25}
    body_ang_vel_weight = -0.05
    dof_pos_limits_weight = -1.0
    action_rate_stages = ((0, -0.1), (500 * 24, -0.2), (750 * 24, -0.4), (1000 * 24, -0.6), (1250 * 24, -0.8), (1500 * 24, -1.0))
    air_time_weight = 3.0
    # microduck window [0.125, 0.300] s; ARTI-D1 swings are slower (servo speed), policy swings measured 0.25-0.32 s
    air_time_min = 0.15
    air_time_max = 0.45
    foot_clearance_weight = -2.0
    foot_swing_height_weight = -0.25
    # microduck 0.02 m on a 25 cm robot; the ARTI-D1 demo swings 28 mm
    swing_target_height = 0.03
    foot_slip_weight = -0.1

    # ---- feet (sole frame and contact from the lowest sole corner, as in the AMP task; no contact sensor on the USD)
    left_foot_body = "left_foot_link"
    right_foot_body = "right_foot_link"
    left_foot_offset = (0.0135599213, -0.024, -0.0593316689)
    right_foot_offset = (0.0135599213, 0.024, -0.0593316765)
    foot_sole_half_length = 0.040
    foot_sole_half_width = 0.0173
    contact_height = 0.006  # m, lowest sole corner (stance ~3.4 mm on flat ground)

    # ---- observation noise / delays / bias (microduck)
    noise_ang_vel = 0.03
    noise_gravity = 0.01
    noise_joint_pos = 0.001
    noise_joint_vel = 0.25
    imu_delay_max_steps = 1  # base ang vel / gravity lag 0..1 step, resampled per episode
    joint_vel_delay_steps = 1  # Dynamixel present velocity is ~1 control period old
    encoder_bias_range = 0.015  # rad, per env, actor joint pos only
    imu_misalign_deg = 6.0  # random-axis IMU mounting rotation per env, actor only

    # ---- events
    reset_xy_range = 0.5
    # pelvis origin z with the soles on the ground (ARTI_D1_CFG init pos; the USD pelvis frame sits at foot level) + 5 mm drop
    reset_height = 0.00445 + 0.005
    push_interval_s = (3.0, 6.0)
    push_velocity = 0.25  # m/s; microduck +-0.3 = 0.75 x its max speed (0.15 run: stand-push failed at 0.4 m/s)

    # ---- terminations
    fell_over_angle = math.radians(70.0)
    # trunk frame (pelvis + ARTI_D1_TRUNK_OFFSET_FROM_PELVIS, 0.2585 m when standing) below this = fallen
    trunk_offset = (0.00964957, 0.0, 0.254045)
    min_trunk_height = 0.15
    lin_vel_at_trunk = True  # False = pelvis link origin (v1 behaviour)

    curriculum_step_offset = 0
