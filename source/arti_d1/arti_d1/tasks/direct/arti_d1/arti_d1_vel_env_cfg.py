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
#   v17: v16 + the v13 realism (real pelvis mass 3.24 kg +-10 %, pelvis CoM +-15 mm, TPU-like compliant contact) for the
#        real robot (v16 converged: 138k fwd .076 / back -.080 / turn .276, 0/64 falls)
#   v18: v17 + training pushes 0.25 -> 0.4 m/s (v17 standpush falls only at 0.4 / 0.5 m/s: 120k 1/8, 4/8)
#   v19: v18 + lateral velocity tracking reward 1.0 (v18: side .014 of .06; play video env falls ~1.5 s into a slow
#        .016 m/s side step on the realistic model, with or without pushes)
#   v20: v18 + lateral tracking 0.3 (v19 with 1.0 fixed the play-video side-step fall but halved turning, .265 -> .115,
#        doubled torso roll and lost the shuffle: 3.2 touchdowns/s, both feet down 33 %)
#   v21: v20 + lateral tracking 0.3 -> 0.5 (v20 converged: side .018-.019 of .06, no side-step fall, turn kept)
#   v24: v23 + trunk shake (real2sim 2026-11-13: v23 on the real robot stands buzz-free but rocks at ~3.5 Hz while
#        stepping, trunk roll / yaw rate 3x / 2.5x the sim for the same replayed joint motion; no friction / contact /
#        play / compliance / mass setting reproduces it, a 3-6 Hz pelvis torque of ~10 Nm roll does)
#   v25: v23 + flat-sole default pose (user 2026-11-13): the old stance (hip_roll -/+0.0247, legs 10 deg inward, no ankle
#        roll joint) stands each sole 10 deg on its edge (URDF FK); hip_roll +/-0.15 = legs vertical, soles flat, feet
#        0.202 m apart (was 0.136), pelvis 8.7 mm lower. Edge stance = a passive roll / yaw pivot on the TPU sole edge,
#        the likely cause of the real 3.5 Hz roll / yaw rocking (v24 models it as a torque instead)
#   v26: v25 + real2sim from the v25 real walk (2026-11-13 112410): replay matches the real joint response only with
#        the leg kd x0.7 (pos/q_des 3-7 Hz .52 vs real .53; x1.0 .40), the remaining trunk roll gap (2.7x) is covered by
#        the v24 trunk shake. Closed loop v25 96k with kd x0.7 + shake 8 Nm = the real rocking (gyro .87/.47/.70, 6/64
#        falls). Training: kd x U(0.5, 0.9), shake U(0, 10) Nm
#   v27 (2026-11-13): v26 78k kept the forward walk and cut the real-repro falls 6/64 -> 1/64, but backward fell to
#        -.016 m/s (v25 -.067) and turn to .10 rad/s (.28): kd x U(0.65, 1.0), shake U(0, 6) Nm
#   v28 (2026-11-13): real v27 60k walk (161844): roll 3-7 Hz gyro 1.05 -> .70 (yaw .57 -> .35) vs v25, but pitch .26 -> .45;
#        replay again fits kd x0.7 (pos/q_des .51-.54 vs real .54). Closed loop v27 60k, kd .7 + shake 10 Nm axes (1, .5, .2)
#        = the real walk (gyro .90/.82/.71 vs real .85/.70/.57, 12/64 falls). From v27 60k: shake U(0, 8) Nm, axes (1, .55, .2)
#   v29 (2026-11-13): v28 60k / 114k: real-repro falls 12 -> 4-6 /64 but back -.015, turn .12-.20 (v26 again): the
#        shake-driven trunk rate made the body_ang_vel penalty pay for walking -> body_ang_vel -0.2 -> -0.05, turn tracking 2 -> 3
#   v30 (2026-11-13): v27 + hip_yaw encoder bias U(+-0.10) rad. Real v27 60k (161844) turned CW at .41 / .52 rad/s on
#        forward / back and -.32 on a +.29 turn command (= no forward / back progress, user); leg odometry of the real stride
#        .039 / -.035 m/s = sim. Sim v27 60k with both hip_yaw encoders +0.06 rad: wz -.25 / -.29, turn +.3 -> -.03 (same)
#   v31 (2026-11-14): v30 + randomization curriculum (adr_*): start narrow at the identified real values (kd x.65-.75,
#        shake 6-8 Nm, gyro hf k 1.5-2.5, hip_yaw +-.03, push .25) and widen to kd .5-.95, shake 0-10, k .5-3,
#        hip_yaw +-.10, push .4 only while falls / tracking stay good. From v30 60k.
_PROFILES = ("v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31")
if ARTI_D1_VEL_PROFILE not in _PROFILES:
    raise ValueError(f"ARTI_D1_VEL_PROFILE must be one of {_PROFILES}, got {ARTI_D1_VEL_PROFILE}")
_V21 = ARTI_D1_VEL_PROFILE == "v21"
_V26 = ARTI_D1_VEL_PROFILE in ("v26", "v27", "v28", "v29", "v30", "v31")  # v27 = v26 with milder kd / shake randomisation, v28 builds on v27
_V27 = ARTI_D1_VEL_PROFILE in ("v27", "v28", "v29", "v30", "v31")
_V30 = ARTI_D1_VEL_PROFILE in ("v30", "v31")
_V31 = ARTI_D1_VEL_PROFILE == "v31"
_V28 = ARTI_D1_VEL_PROFILE in ("v28", "v29")
_V29 = ARTI_D1_VEL_PROFILE == "v29"
_V25 = ARTI_D1_VEL_PROFILE in ("v25", "v26", "v27", "v28", "v29", "v30", "v31")  # v26 builds on v25
_V24 = ARTI_D1_VEL_PROFILE == "v24"
_V23 = ARTI_D1_VEL_PROFILE in ("v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31")  # v24 / v25 build on v23
_V22 = ARTI_D1_VEL_PROFILE in ("v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31")  # v23 builds on v22
_V20 = ARTI_D1_VEL_PROFILE in ("v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31")  # v21 / v22 build on v20
_V19 = ARTI_D1_VEL_PROFILE == "v19"
_V18 = ARTI_D1_VEL_PROFILE in ("v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31")  # v19 builds on v18
_V13 = ARTI_D1_VEL_PROFILE in ("v13", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31")  # v17 adds the v13 realism
_V16 = ARTI_D1_VEL_PROFILE in ("v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31")  # v17 builds on v16
_V15 = ARTI_D1_VEL_PROFILE in ("v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31")  # v16 builds on v15
_V14 = ARTI_D1_VEL_PROFILE in ("v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31")  # v15 builds on v14
_V12 = ARTI_D1_VEL_PROFILE in ("v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31")  # v13 / v14 build on v12
_V11 = ARTI_D1_VEL_PROFILE in ("v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31")  # v12 builds on v11
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
    command_lin_y_range = (0.0, 0.0) if _V26 else (-0.08, 0.08)  # v26: no sideways (user 2026-11-13: forward / back / turn only)
    command_yaw_range = (-0.4, 0.4)
    command_resample_s = (3.0, 8.0)
    turn_in_place_fraction = 0.15
    lateral_fraction = 0.0 if _V26 else 0.15  # pure sideways commands, |vy| in [0.4, 1] x range max (v26: none)
    # standing (exactly zero command) env fraction by env step (microduck standing_envs curriculum, iter x 24)
    standing_stages = ((0, 0.02), (500 * 24, 0.05), (750 * 24, 0.1), (1000 * 24, 0.15), (1500 * 24, 0.2), (2000 * 24, 0.25))
    command_threshold = 0.01  # microduck command_threshold for gait terms / pose walking_threshold

    # ---- rewards (weights x step_dt, like mjlab's RewardManager)
    track_lin_vel_weight = 3.0 if _V16 else 2.0
    track_lin_vel_std = 0.08  # m/s; microduck sqrt(0.1)=0.32 for +-0.4 commands, scaled by the command range (v3/v4: 0.06, no tracking learned)
    # v3: extra lateral term (v1/v2 never learned lin y: a 0.06 m/s miss still paid exp(-.56)=.57 of the joint xy term)
    track_lat_vel_weight = 1.0 if _V19 else 0.5 if _V21 else 0.3 if _V20 else 0.0  # v19: 1.0, v20: 0.3, v21: 0.5; off (= v7 cont1 deployed policy); v8 1.0 fine-tune from v7c1 240k and v3-v5 1.0 from scratch never tracked lateral
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
    track_ang_vel_weight = 3.0 if _V29 else 2.0
    track_ang_vel_std = 0.28  # rad/s; microduck sqrt(0.5)=0.71 for +-1.0
    upright_weight = 2.0
    upright_std = math.sqrt(0.05)
    pose_weight = 1.0
    # per-joint std (rad) of exp(-mean((q - q0)^2 / std^2)); microduck values (hip_roll walking 0.05 -> 0.1: ARTI-D1 shifts
    # its weight with hip_roll, the demo uses 15 deg of it)
    pose_std_standing = {"hip_yaw": 0.1, "hip_roll": 0.05, "hip_pitch": 0.15, "knee": 0.15, "ankle": 0.1}
    pose_std_walking = {"hip_yaw": 0.3, "hip_roll": 0.1, "hip_pitch": 0.4, "knee": 0.4, "ankle": 0.25}
    body_ang_vel_weight = -0.05 if _V29 else -0.2 if _V12 else -0.05  # v12 x4 (real torso rocked 6x sim)
    dof_pos_limits_weight = -1.0
    action_rate_stages = ((0, -0.1), (500 * 24, -0.2), (750 * 24, -0.4), (1000 * 24, -0.6), (1250 * 24, -0.8), (1500 * 24, -1.0))
    air_time_weight = 4.0 if _V23 else 3.0  # v23: win back the v20 shuffle under the v22 gyro vibration model
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
    encoder_offset_fixed = ()  # eval what-if: fixed leg encoder offsets (rad, leg action order), e.g. hip_yaw calibration error
    encoder_bias_hip_yaw = 0.10 if _V30 else 0.0  # v30: per-env hip_yaw encoder bias U(+-0.10) rad (real: steady CW yaw .4-.5 rad/s = both hip_yaw zeros off ~0.06-0.1)
    # v22 real2sim (2026-11-12): replaying the real robot's actions open loop gives the same joint motion in sim (7-13 Hz
    # rms .0106 vs .0108 rad) but 2.4x less trunk gyro (.25 vs .60 rad/s): flex / play outside the encoders and the IMU
    # mount. The policy fed that back (gyro ~50 % of its 10 Hz action) -> ~10 Hz standing buzz on the robot (v12, v20).
    # Model: gyro seen = gyro + k * highpass(gyro, gyro_hf_cutoff_hz), k ~ U(range) per env and episode.
    gyro_hf_gain_range = (0.5, 3.0) if _V22 else (0.0, 0.0)  # real fit k ~1.5-2.5 (replay + closed-loop buzz)
    gyro_hf_cutoff_hz = 3.0
    # series compliance outside the encoder per joint type: (lo, hi) of the effective kp fraction s per env; the encoder
    # reads the motor side (real2sim v24, 2026-11-13: real walking replay, same hip roll encoder motion, 3x trunk roll)
    actuator_series_scale = {}
    # joint play (actuator_backlash_deg) between the encoder and the link (frames / horns) instead of inside the servo
    backlash_outside_encoder = False
    actuator_gain_whatif = (1.0, 1.0)  # (kp, kd) multipliers on the leg joints at env start (eval what-ifs)
    actuator_kd_random = (0.5, 0.95) if _V31 else (0.65, 1.0) if _V27 else (0.5, 0.9) if _V26 else (1.0, 1.0)  # per-env multiplier on the leg kd (on top of actuator_kd_scale)
    # trunk shake: per episode a pelvis torque amp * axes * U(0.5, 1) * sin(2 pi f t + phase), f in trunk_shake_freq_hz,
    # body frame, only while the command norm > trunk_shake_cmd_min (the real rocking comes with stepping)
    trunk_shake_torque = (0.0, 10.0) if _V31 else (0.0, 8.0) if _V28 else (0.0, 6.0) if _V27 else (0.0, 10.0) if (_V24 or _V26) else (0.0, 0.0)  # Nm amplitude range (replay fit ~8-10 Nm roll)
    trunk_shake_freq_hz = (3.0, 6.0)
    trunk_shake_axes = (1.0, 0.55, 0.2) if _V28 else (1.0, 0.15, 0.3)  # roll, pitch, yaw weights (real walking replay: roll 3x, yaw 2.5x, pitch 1x)
    trunk_shake_walk_only = True
    trunk_shake_cmd_min = 0.015
    imu_misalign_deg = 6.0  # random-axis IMU mounting rotation per env, actor only

    # ---- events
    reset_xy_range = 0.5
    # pelvis origin z with the soles on the ground (ARTI_D1_CFG init pos; the USD pelvis frame sits at foot level) + 5 mm drop
    reset_height = (-0.0042 if _V25 else 0.00445) + 0.005  # pelvis origin over the sole at the default pose
    push_interval_s = (3.0, 6.0)
    push_velocity = 0.4 if _V18 else 0.25  # v18: 0.4; m/s; microduck +-0.3 = 0.75 x its max speed (0.15 run: stand-push failed at 0.4 m/s)

    # ---- terminations
    fell_over_angle = math.radians(70.0)
    # trunk frame (pelvis + ARTI_D1_TRUNK_OFFSET_FROM_PELVIS, 0.2585 m when standing) below this = fallen
    trunk_offset = (0.00964957, 0.0, 0.254045)
    min_trunk_height = 0.15
    lin_vel_at_trunk = True  # False = pelvis link origin (v1 behaviour)

    curriculum_step_offset = 0

    # ---- v31 domain randomization curriculum (ADR-like, user 2026-11-14: "as real as possible, then widen slowly").
    # One level L in [0, 1] per run: each range = start + L * (end - start), start = narrow around the identified real
    # values, end = the wide ranges above (used as-is with adr_enable False, e.g. in eval_vel). Drawn per env at reset.
    # Every adr_window_steps env steps: L += adr_step if falls per env-episode (20 s) <= adr_fall_up and the mean
    # track_lin / track_ang rewards >= adr_lin_min / adr_ang_min; L -= adr_step if falls >= adr_fall_down.
    adr_enable = _V31
    adr_level_init = 0.0
    adr_kd = ((0.65, 0.75), (0.5, 0.95))  # leg kd multiplier (real replay fit x0.7)
    # v31c: shake starts low (0-4 Nm): a constant 6-8 Nm start (v31b) cost backward (-.015 nominal) and it never came back
    adr_shake = ((0.0, 4.0), (0.0, 10.0))  # trunk shake Nm (real closed-loop fit ~8; a stand-in torque, not identified physics)
    adr_gyro_hf = ((1.5, 2.5), (0.5, 3.0))  # gyro high-pass gain (real fit 1.5-2.5)
    # v31b: hip_yaw stays wide from the start - the real zero error is unknown (v31 30k at +-.03 lost the v30 drift fix:
    # real-repro wz -.07 / -.095, turn .064)
    adr_hip_yaw = (0.10, 0.10)  # hip_yaw encoder bias half-range (rad) around adr_hip_yaw_center
    adr_hip_yaw_center = (0.0, 0.0)  # measured left / right hip_yaw zero error once known (leg order 0 / 1)
    adr_push = (0.25, 0.4)  # push velocity m/s
    adr_window_steps = 4800
    adr_step = 0.05
    adr_fall_up = 0.05
    adr_fall_down = 0.15
    adr_lin_min = 0.70  # absolute gate (used when adr_calib_windows = 0)
    adr_ang_min = 0.52
    # v31b: gate = keep adr_keep_frac of the best track_lin and yaw-only tracking (unweighted exp, roll / pitch rate excluded:
    # the forced shake caps r_ang) reached during the first adr_calib_windows windows at the start level
    adr_calib_windows = 8
    adr_keep_frac = 0.95
    adr_drop_frac = 0.85  # v31c: narrow when backward / turn tracking falls below this fraction of its reference

    def __post_init__(self):
        if _V25:  # flat-sole stance: legs vertical
            self.scene.robot.init_state.joint_pos = {
                **self.scene.robot.init_state.joint_pos, "left_hip_roll": 0.15, "right_hip_roll": -0.15}
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
