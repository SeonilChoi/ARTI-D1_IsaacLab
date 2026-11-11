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
import os

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from .arti_d1_env_cfg import ARTI_D1_ACTUATOR_ARMATURE, ARTI_D1_JOINT_EFFORT_LIMIT, ArtiD1PhysicsCfg, ArtiD1SceneCfg

ARTI_D1_VEL_NUM_JOINTS = 10  # leg joints
# v10 (Open Duck Mini): the actor sees the last N actions (newest first), so it can tell which targets are still in
# flight under the actuator delay. v1-v9: 1 (39-dim obs; set 1 + the v9 delay settings to evaluate those checkpoints).
# Run profile (env var ARTI_D1_VEL_PROFILE, read at import; set it for train / eval / video / export of a run):
#   v10 (default): Open Duck style, 3-action history, action_scale 0.25, per-step 25-65 ms actuator + 0-2 step IMU delay
#   v11: v9 setup (1 action, scale 0.5, per-env delay, IMU 0/1 per episode) + the servo model fitted to the real
#        robot (scripts/tuning/fit_servo_model.py on sysid_response2 2026-11-10, Return Delay Time 0, 10 ms bus):
#        delay ~22 ms, kp x1.5, kd x1.25, armature x3, Coulomb 0.06 Nm (fit: mean normalised RMS .055 vs .11-.27 before)
ARTI_D1_VEL_PROFILE = os.environ.get("ARTI_D1_VEL_PROFILE", "v10")
#   v12: v11 + joint play (backlash, total 1-3 deg per env / leg joint, user 2026-11-10) + torso ang vel penalty x4
#   v13: v12 + the real mass (robot weighed 5.34 kg vs URDF 5.79: pelvis 3.69 -> 3.24 kg, +-10%), pelvis CoM +-15 mm,
#        TPU-like compliant foot / ground contact (user 2026-11-10: soles are slightly elastic TPU)
#   v14: v12 + "잔발" (user 2026-11-11: short quick steps like the shuffle reference, learned the microduck way, no
#        reference): air-time window 0.15-0.45 -> 0.06-0.22 s, swing target 30 -> 12 mm
#   v15: v14 + air-time reward only in single stance (v14 ran with 12 % both-feet-up flight)
#   v16: v15 + forward / backward tracking weight 2 -> 3 (v15 240k: fwd .059 of .12, back -.065 of -.10)
_PROFILES = ("v10", "v11", "v12", "v13", "v14", "v15", "v16")
if ARTI_D1_VEL_PROFILE not in _PROFILES:
    raise ValueError(f"ARTI_D1_VEL_PROFILE must be one of {_PROFILES}, got {ARTI_D1_VEL_PROFILE}")
_V13 = ARTI_D1_VEL_PROFILE == "v13"
_V16 = ARTI_D1_VEL_PROFILE == "v16"
_V15 = ARTI_D1_VEL_PROFILE in ("v15", "v16")  # v16 builds on v15
_V14 = ARTI_D1_VEL_PROFILE in ("v14", "v15", "v16")  # v15 builds on v14
_V12 = ARTI_D1_VEL_PROFILE in ("v12", "v13", "v14", "v15", "v16")  # v13 / v14 build on v12
_V11 = ARTI_D1_VEL_PROFILE in ("v11", "v12", "v13", "v14", "v15", "v16")  # v12 builds on v11
ARTI_D1_VEL_ACTION_HISTORY = 1 if _V11 else 3
# actor: base ang vel (3) + projected gravity (3) + leg joint pos rel (10) + leg joint vel (10) + last N actions (10 N)
# + command (3, always last)
ARTI_D1_VEL_POLICY_OBS = 3 + 3 + 10 + 10 + 10 * ARTI_D1_VEL_ACTION_HISTORY + 3
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
            "com_range": {"x": (-0.015, 0.015), "y": (-0.015, 0.015), "z": (-0.015, 0.015)} if _V13
            else {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.005, 0.005)},
        },
    )

    # v13: real pelvis mass. URDF total 5.79 kg, robot weighed 5.34 kg -> pelvis 3.69 - 0.45 = 3.24 kg, +-10% per env
    # (applied after body_mass, absolute)
    pelvis_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (2.92, 3.56),
            "operation": "abs",
            "recompute_inertia": True,
        },
    ) if _V13 else None


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
    # v10 0.25 (Open Duck Mini action_scale; less aggressive targets); v1-v9 0.5
    action_scale = 0.5 if _V11 else 0.25
    action_history_steps = ARTI_D1_VEL_ACTION_HISTORY
    # Arms held at this target (rad); no policy output.
    arm_hold_target = {"left_arm_m8": 0.0, "right_arm_m8": 0.0, "left_arm_m8crank_plate": 0.0, "right_arm_m8crank_plate": 0.0}

    # ---- Dynamixel joint model, every physics step (same as ArtiD1AmpEnvCfg; BAM stand-in)
    # v9 (real robot 11-10: action -> joint lag 85-117 ms vs 50-67 ms in sim at 10 ms, robot shook): 40 ms, 25-65 ms per env.
    # v1-v8: 10.0 with (-6, 8)
    actuator_delay_ms = 22.0 if _V11 else 40.0  # v11: fitted 21-25 ms
    actuator_delay_jitter_ms = (-7.0, 18.0) if _V11 else (-15.0, 25.0)  # v11: 15-40 ms per env
    # v10 (Open Duck Mini: action delay drawn every control step): True = a new delay per env every control step,
    # False = one per env at start-up (v1-v9)
    actuator_delay_per_step = not _V11
    actuator_armature = {k: v * (3.0 if _V11 else 1.0) for k, v in ARTI_D1_ACTUATOR_ARMATURE.items()}
    # PD gain scales on the robot actuators (applied in __post_init__); v11 fitted kp x1.5, kd x1.25
    actuator_kp_scale = 1.5 if _V11 else 1.0
    actuator_kd_scale = 1.25 if _V11 else 1.0
    # Joint play (backlash) on the leg joints: total play in deg, drawn per env and joint; inside +-play/2 of the target
    # the servo gives no position torque (dead band on the PD error). (0, 0) = off. v12: 1-3 deg (user estimate).
    actuator_backlash_deg = (1.0, 3.0) if _V12 else (0.0, 0.0)
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
    track_lin_vel_weight = 3.0 if _V16 else 2.0
    track_lin_vel_std = 0.08  # m/s; microduck sqrt(0.1)=0.32 for +-0.4 commands, scaled by the command range (v3/v4: 0.06, no tracking learned)
    # v3: extra lateral term (v1/v2 never learned lin y: a 0.06 m/s miss still paid exp(-.56)=.57 of the joint xy term)
    track_lat_vel_weight = 0.0  # off (= v7 cont1 deployed policy); v8 1.0 fine-tune from v7c1 240k and v3-v5 1.0 from scratch never tracked lateral
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
    body_ang_vel_weight = -0.2 if _V12 else -0.05  # v12 x4 (real torso rocked 6x sim)
    dof_pos_limits_weight = -1.0
    action_rate_stages = ((0, -0.1), (500 * 24, -0.2), (750 * 24, -0.4), (1000 * 24, -0.6), (1250 * 24, -0.8), (1500 * 24, -1.0))
    air_time_weight = 3.0
    # microduck window [0.125, 0.300] s; ARTI-D1 swings are slower (servo speed), policy swings measured 0.25-0.32 s
    air_time_min = 0.06 if _V14 else 0.15  # v14: quick short swings (shuffle reference single support 0.18 s)
    air_time_max = 0.22 if _V14 else 0.45
    air_time_single_stance = _V15
    foot_clearance_weight = -2.0
    foot_swing_height_weight = -0.25
    # microduck 0.02 m on a 25 cm robot; the ARTI-D1 demo swings 28 mm
    swing_target_height = 0.012 if _V14 else 0.03  # v14: low shuffle swing (reference 10 mm)
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
    imu_delay_max_steps = 1 if _V11 else 2  # base ang vel / gravity lag 0..N steps (v1-v9: 1, per episode)
    imu_delay_per_step = not _V11  # v10 (Open Duck Mini 0-3 steps every step); False = drawn per episode
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

    def __post_init__(self):
        if _V13:  # TPU-like soles: compliant contact on the ground material (~1.3 mm under the full weight on one point)
            self.scene.terrain.physics_material.compliant_contact_stiffness = 4.0e4  # N/m
            self.scene.terrain.physics_material.compliant_contact_damping = 400.0  # N s/m
        if _V11:  # fitted Coulomb friction on the legs
            for k in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"):
                self.actuator_coulomb[k] = 0.06
        if self.actuator_kp_scale != 1.0 or self.actuator_kd_scale != 1.0:
            for act in self.scene.robot.actuators.values():
                act.stiffness = {k: v * self.actuator_kp_scale for k, v in act.stiffness.items()}
                act.damping = {k: v * self.actuator_kd_scale for k, v in act.damping.items()}
