from __future__ import annotations

import os

from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxCollisionCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ISAACLAB_ASSETS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "assets")

# configuration/arti_d1_base.usd references /visuals/<body> in arti_d1_physics.usd for these frame links, but the physics
# layer has no such prims (the links carry no mesh). Kit/PhysX only warns about the unresolved arcs; Newton's USD
# importer raises on any stage composition error. The arcs are authored inside the asset's own layer stack, so no
# override on the scene stage can remove them: they are stripped from the in-memory base layer (never saved, the file
# is unchanged). Removing an unresolved reference composes the same (empty) prims, on every backend.
_DANGLING_VISUAL_REFERENCE_BODIES = ("trunk",)
_patched_layers = []  # keeps the edited layers alive in the USD layer registry for the whole process


def _strip_dangling_visual_references(usd_path: str) -> None:
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(os.path.join(os.path.dirname(usd_path), "configuration", "arti_d1_base.usd"))
    if layer is None or layer in _patched_layers:
        return
    for body in _DANGLING_VISUAL_REFERENCE_BODIES:
        spec = layer.GetPrimAtPath(f"/arti_d1/{body}/visuals")
        if spec is not None:
            spec.referenceList.ClearEdits()
    _patched_layers.append(layer)


# The URDF importer put CollisionAPI + MeshCollisionAPI(convexHull) on each collider's "node_STL_BINARY_" Xform, whose
# single child "mesh" holds the geometry. PhysX builds the hull from the Xform's child; Newton only turns gprims into
# shapes ("CollisionAPI applied to an unknown UsdGeomGPrim type"), leaving the robot without collision shapes. For
# Newton the same collision APIs and approximation are added to that child mesh, in memory only (file unchanged).
_COLLISION_SCHEMAS = ("PhysicsCollisionAPI", "PhysicsMeshCollisionAPI")


def _add_newton_mesh_colliders(usd_path: str) -> None:
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(os.path.join(os.path.dirname(usd_path), "configuration", "arti_d1_physics.usd"))
    if layer is None or layer in _patched_layers:
        return
    colliders = []
    layer.Traverse("/colliders", lambda path: colliders.append(path) if path.IsPrimPath() else None)
    for path in colliders:
        spec = layer.GetPrimAtPath(path)
        if spec.name != "node_STL_BINARY_" or "PhysicsCollisionAPI" not in spec.GetInfo("apiSchemas").GetAddedOrExplicitItems():
            continue
        mesh = layer.GetPrimAtPath(path.AppendChild("mesh")) or Sdf.PrimSpec(spec, "mesh", Sdf.SpecifierOver)
        mesh.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=list(_COLLISION_SCHEMAS)))
        approximation = spec.attributes.get("physics:approximation")
        if approximation is not None:
            attr = Sdf.AttributeSpec(mesh, "physics:approximation", Sdf.ValueTypeNames.Token, Sdf.VariabilityUniform)
            attr.default = approximation.default
    _patched_layers.append(layer)


def _physics_is_newton() -> bool:
    from isaaclab.sim import SimulationContext

    sim = SimulationContext.instance()
    return sim is not None and type(sim.cfg.physics).__module__.startswith("isaaclab_newton")


def spawn_arti_d1_usd(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """``spawn_from_usd`` after the in-memory asset fixes above (Newton colliders only on the Newton backend)."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd

    _strip_dangling_visual_references(cfg.usd_path)
    if _physics_is_newton():
        _add_newton_mesh_colliders(cfg.usd_path)
    return spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)


LEG_JOINTS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
ARM_JOINTS = ("arm_m8", "arm_m8crank_plate")
JOINT_NAMES = tuple(f"{side}_{j}" for side in ("left", "right") for j in LEG_JOINTS + ARM_JOINTS)

# Policy / action / observation joint order = Dynamixel IDs 1-14 (humanoid_controller config/robot/ARTI-D1,
# kPolicyJointNames) = the Isaac Sim PhysX articulation order the deployed ONNX policies were trained in.
# Pinned on every physics backend through ArticulationCfg.joint_ordering (Newton/MJWarp traverses the tree in a
# different order); ArtiD1AmpEnv asserts it at start-up.
ARTI_D1_POLICY_JOINT_ORDER = (
    "left_arm_m8",  # 1
    "left_hip_yaw",  # 2
    "right_arm_m8",  # 3
    "right_hip_yaw",  # 4
    "left_arm_m8crank_plate",  # 5
    "left_hip_roll",  # 6
    "right_arm_m8crank_plate",  # 7
    "right_hip_roll",  # 8
    "left_hip_pitch",  # 9
    "right_hip_pitch",  # 10
    "left_knee",  # 11
    "right_knee",  # 12
    "left_ankle",  # 13
    "right_ankle",  # 14
)

# Standing pose = first frame of every reference clip (motion_generator arti_d1, 1.36 s gait: CoM 0.2585 m, feet 0.1354 m
# apart = hip_roll 10 deg inward). The legs are mirrored in the URDF (no pi rotation like ARTI-H1), so the hip_roll
# signs are opposite
# left / right. The arm plate sits at its -13 deg rest angle (= the URDF lower limit).
JOINT_INIT_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.17466,
    "left_hip_pitch": -0.00293,
    "left_knee": 0.07136,
    "left_ankle": -0.06844,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.17466,
    "right_hip_pitch": -0.00293,
    "right_knee": 0.07136,
    "right_ankle": -0.06844,
    "left_arm_m8": -0.00048,
    "left_arm_m8crank_plate": -0.2268,  # reference -0.22689 = URDF lower limit; kept just inside it
    "right_arm_m8": -0.00048,
    "right_arm_m8crank_plate": -0.2268,
}

# Isaac Sim PhysX body order (breadth-first), recorded from the 3.x PhysX run (scripts/porting/check_amp_consistency.py).
# The env resolves bodies by name, so this is not policy-facing; it keeps body tensors identical across backends
# (the Isaac Lab ordering docs say to set joint_ordering and body_ordering together).
ARTI_D1_BODY_ORDER = (
    "pelvis",
    "left_arm_m8_crank",
    "left_hip",
    "right_arm_m8_crank",
    "right_hip",
    "trunk",
    "left_arm_plate",
    "left_roll_to_pitch",
    "right_arm_plate",
    "right_roll_to_pitch",
    "left_thigh",
    "right_thigh",
    "left_shin",
    "right_shin",
    "left_foot_link",
    "right_foot_link",
)

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
# the 10 deg inward hip_roll stance (feet 0.1354 m) reaches 3.3 rad/s knee in 0.15 % of reference frames;
# the old 0.72 s gait needed up to 8.6 rad/s. Slower gaits cannot shift the CoM over the stance foot within that range.
JOINT_VELOCITY_LIMIT = {name: joint_motor(name)["no_load_speed"] for name in JOINT_NAMES}


ARTI_D1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        func=spawn_arti_d1_usd,
        usd_path=f"{ISAACLAB_ASSETS_DIR}/models/arti_d1/arti_d1/arti_d1.usd",
        # Isaac Lab 3.x schema fragments: the PhysX values are the 2.x RigidBody/Collision/ArticulationRootPropertiesCfg
        # values unchanged; the Newton fragment only mirrors enabled_self_collisions=False.
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        collision_props=PhysxCollisionCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
        copy_from_source=True,
    ),
    # Public (policy) order on every backend; on Isaac Sim PhysX it equals the backend order, so no map is installed.
    joint_ordering=ARTI_D1_POLICY_JOINT_ORDER,
    body_ordering=ARTI_D1_BODY_ORDER,
    init_state=ArticulationCfg.InitialStateCfg(
        # Reference pelvis origin height with the soles on the ground (root z of every clip).
        pos=(0.0, 0.0, 0.01225),
        joint_pos=JOINT_INIT_POS,
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=JOINT_STIFFNESS,
            damping=JOINT_DAMPING,
            # 2.x effort_limit (implicit: rated and solver limit) / velocity_limit_sim, under the 3.x names
            actuator_effort_limit=JOINT_EFFORT_LIMIT,
            joint_effort_limit=JOINT_EFFORT_LIMIT,
            joint_velocity_limit=JOINT_VELOCITY_LIMIT,
        ),
    },
)
