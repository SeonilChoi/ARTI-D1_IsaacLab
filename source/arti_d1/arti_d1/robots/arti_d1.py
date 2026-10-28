from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ISAACLAB_ASSETS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "assets")

LEG_JOINTS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
ARM_JOINTS = ("arm_m8", "arm_m8crank_plate")
JOINT_NAMES = tuple(f"{side}_{j}" for side in ("left", "right") for j in LEG_JOINTS + ARM_JOINTS)

# Standing pose = first frame of every reference clip (motion_generator arti_d1, 1.36 s gait: CoM 0.255 m, feet 0.17 m
# apart). The legs are mirrored in the URDF (no pi rotation like ARTI-H1), so the hip_roll signs are opposite
# left / right. The arm plate sits at its -13 deg rest angle (= the URDF lower limit).
JOINT_INIT_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.08412,
    "left_hip_pitch": -0.00549,
    "left_knee": 0.07658,
    "left_ankle": -0.07109,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.08412,
    "right_hip_pitch": -0.00549,
    "right_knee": 0.07658,
    "right_ankle": -0.07108,
    "left_arm_m8": -0.00157,
    "left_arm_m8crank_plate": -0.2268,  # reference -0.22689 = URDF lower limit; kept just inside it
    "right_arm_m8": -0.00157,
    "right_arm_m8crank_plate": -0.2268,
}

# Dynamixel position-mode model. Legs: XH540-W270 (12 V stall 10.6 Nm, no load 30 rpm = 3.14 rad/s), arms: XM430-W350
# (12 V stall 4.1 Nm, no load 46 rpm = 4.82 rad/s); the URDF effort / velocity limits are these datasheet values.
# The servo applies PWM = P / 128 * error[pulse] (885 = full voltage = stall torque at zero speed), and the DC motor
# back-EMF removes stall / no_load_speed per rad/s, so as a joint PD:
#   kp = P / 128 * 4096 / (2 pi) / 885 * stall      (Nm/rad)
#   kd = stall / no_load_speed (+ the D gain term)  (Nm s/rad)
# P = 800 is the factory Position P Gain of both models; humanoid_controller motor.yaml 'kp' writes it to the servo.
# First-principles values, not identified on the robot yet: run a sysid before trusting them for sim-to-real.
DYNAMIXEL_POSITION_P_GAIN = 800.0
DYNAMIXEL_MOTORS = {
    "x540": {"stall_torque": 10.6, "no_load_speed": 3.14},
    "x430": {"stall_torque": 4.1, "no_load_speed": 4.82},
}


def joint_motor(joint_name: str) -> dict:
    return DYNAMIXEL_MOTORS["x430" if "_arm_" in joint_name else "x540"]


def dynamixel_kp(joint_name: str, p_gain: float = DYNAMIXEL_POSITION_P_GAIN) -> float:
    return p_gain / 128.0 * 4096.0 / (2.0 * 3.141592653589793) / 885.0 * joint_motor(joint_name)["stall_torque"]


def dynamixel_kd(joint_name: str) -> float:
    motor = joint_motor(joint_name)
    return motor["stall_torque"] / motor["no_load_speed"]


JOINT_STIFFNESS = {name: round(dynamixel_kp(name), 3) for name in JOINT_NAMES}
JOINT_DAMPING = {name: round(dynamixel_kd(name), 3) for name in JOINT_NAMES}
JOINT_EFFORT_LIMIT = {name: joint_motor(name)["stall_torque"] for name in JOINT_NAMES}
# Capped at the 12 V no-load speed. The reference gait (1.36 s cycle, 22 mm swing, steps <= 4.5 cm, feet 0.17 m apart)
# is the highest swing that stays below it (leg joints <= 3.03 rad/s) with hip_roll in -18 / +12 deg (was 1.25 s / 15 mm);
# the old 0.72 s gait needed up to 8.6 rad/s. Slower gaits cannot shift the CoM over the stance foot within that range.
JOINT_VELOCITY_LIMIT = {name: joint_motor(name)["no_load_speed"] for name in JOINT_NAMES}


ARTI_D1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DIR}/models/arti_d1/arti_d1/arti_d1.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Reference pelvis origin height with the soles on the ground (root z of every clip).
        pos=(0.0, 0.0, 0.00875),
        joint_pos=JOINT_INIT_POS,
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=JOINT_STIFFNESS,
            damping=JOINT_DAMPING,
            effort_limit=JOINT_EFFORT_LIMIT,
            velocity_limit_sim=JOINT_VELOCITY_LIMIT,
        ),
    },
)
