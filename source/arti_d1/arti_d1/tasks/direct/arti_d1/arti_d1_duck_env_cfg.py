"""ARTI-D1 velocity walk trained the Open Duck Mini v2 way (apirrone/Open_Duck_Playground joystick.py), ARTI-D1 sized.

What is taken from Open Duck (ctrl 50 Hz there, 60 Hz here):
  * imitation reward towards a placo reference gait looked up by command and gait phase (reference: motion_generator
    config/arti_d1/gait_duck.json sweep -> assets/motions/arti_d1_duck_reference.npz), phase (cos, sin) in the obs
  * reward set and weights: tracking lin 2.5 / ang 6.0 (exp(-err^2 / sigma)), torques, action rate -0.5,
    stand still -0.2, alive 20, imitation 1.0 (joint pos 15, joint vel 1e-3, base vel / ang vel, contacts);
    total clipped at >= 0
  * actor obs: gyro, gravity, joint pos - default, joint vel x 0.05, last 3 actions, motor targets, phase, command
  * action_scale 0.25, motor target rate limit, 10 % zero commands, pushes, noise levels, domain randomisation
    (mass +-10 %, CoM, kp +-10 %, joint offsets +-0.03 rad, friction), backlash (here: joint play)
What is ARTI-D1 specific: the servo model fitted to this robot (v11), joint play 1-3 deg (v12), real pelvis mass and
TPU-like contact (v13), sigmas / ranges scaled to the ARTI-D1 reference command coverage, no foot contact sensors on the
robot (contacts only in the critic).
"""

import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from .arti_d1_env_cfg import ARTI_D1_ACTUATOR_ARMATURE
from .arti_d1_vel_env_cfg import ArtiD1VelEnvCfg, ArtiD1VelEventCfg

DUCK_ACTION_HISTORY = 3
# gyro 3 + gravity 3 + joint pos 10 + joint vel 10 + 3 actions 30 + motor targets 10 + phase 2 + command 3 (last)
ARTI_D1_DUCK_POLICY_OBS = 3 + 3 + 10 + 10 + 10 * DUCK_ACTION_HISTORY + 10 + 2 + 3
# + lin vel 3, foot height 2, air time 2, contact 2, reference joint pos 10, reference contact 2
ARTI_D1_DUCK_CRITIC_OBS = ARTI_D1_DUCK_POLICY_OBS + 3 + 2 + 2 + 2 + 10 + 2

_REFERENCE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "motions", "arti_d1_quick_reference.npz")
# duck v3 (2026-11-11): quick short-step gait (motion_generator gait_quick.json: single support 0.25 s, period 0.6 s,
# foot 15 mm); duck v1 / v2 used arti_d1_duck_reference.npz (0.45 s, period 1.08 s, 25 mm: feet dragged)


@configclass
class ArtiD1DuckEventCfg(ArtiD1VelEventCfg):
    def __post_init__(self):
        # Open Duck randomize.py: floor friction 0.5-1.0 (combined "average" with the plane's 1.0 -> foot 0.0-1.0)
        self.foot_friction.params["static_friction_range"] = (0.0, 1.0)
        self.foot_friction.params["dynamic_friction_range"] = (0.0, 1.0)
        # body masses x0.9-1.1
        self.body_mass.params["mass_distribution_params"] = (0.9, 1.1)
        # torso CoM: Open Duck +-5 cm on a 42 cm robot; +-3 cm here
        self.base_com.params["com_range"] = {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)}

    # real pelvis mass (robot 5.34 kg, URDF 5.79): 3.24 kg +-10 % (Open Duck: torso +-0.1 kg)
    pelvis_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (2.92, 3.56),
            "operation": "abs",
            "recompute_inertia": True,
        },
    )


@configclass
class ArtiD1DuckEnvCfg(ArtiD1VelEnvCfg):
    events: ArtiD1DuckEventCfg = ArtiD1DuckEventCfg()
    observation_space = ARTI_D1_DUCK_POLICY_OBS
    state_space = ARTI_D1_DUCK_CRITIC_OBS
    episode_length_s = 20.0  # Open Duck 1000 steps at 50 Hz

    reference_file = _REFERENCE
    reference_knn = 4  # inverse-distance blend of the nearest reference clips in command space

    # ---- actions (Open Duck)
    action_scale = 0.25
    action_history_steps = DUCK_ACTION_HISTORY
    max_motor_velocity = 4.8  # rad/s target rate limit (Open Duck 5.24 for the STS3215; XH540-W270 no-load 4.8)

    # ---- servo model: v11 fit to this robot + v12 joint play
    actuator_delay_ms = 22.0
    actuator_delay_jitter_ms = (-7.0, 18.0)
    actuator_delay_per_step = False
    actuator_armature = {k: v * 3.0 for k, v in ARTI_D1_ACTUATOR_ARMATURE.items()}
    actuator_kp_scale = 1.5
    actuator_kd_scale = 1.25
    actuator_kp_random = (0.9, 1.1)  # Open Duck kp x0.9-1.1 per env
    actuator_backlash_deg = (1.0, 3.0)

    # ---- commands: inside the reference coverage (quick: vx -0.060..0.071, vy +-0.07, wz +-0.33)
    command_lin_x_range = (-0.05, 0.07)
    command_lin_y_range = (-0.05, 0.05)
    command_yaw_range = (-0.33, 0.33)
    command_resample_s = (6.0, 10.0)  # Open Duck: every 500 steps (10 s)
    zero_command_fraction = 0.2  # duck v4 (v3: 0.1, Open Duck)

    # ---- observation noise, uniform +- (duck v3: ARTI-D1 sensor levels; Open Duck gyro .1, gravity .1, joint pos
    # .03-.08, joint vel 2.5 made duck v2 lurch)
    noise_ang_vel = 0.05
    noise_gravity = 0.02
    noise_joint_pos_by_type = {"hip_yaw": 0.01, "hip_roll": 0.01, "hip_pitch": 0.01, "knee": 0.015, "ankle": 0.02}
    noise_joint_vel = 0.5  # rad/s before the 0.05 scale
    joint_vel_obs_scale = 0.05
    encoder_bias_range = 0.03  # Open Duck qpos0 +-0.03
    imu_delay_max_steps = 2
    imu_delay_per_step = True

    # ---- pushes (Open Duck 0.1-1.0 m/s every 5-10 s on a ~2 kg robot)
    push_interval_s = (5.0, 10.0)
    push_velocity = 0.25  # duck v3 (v2: 0.5)

    # ---- rewards (Open Duck scales; sigmas scaled to the ARTI-D1 command coverage)
    duck_tracking_lin_vel = 2.5
    duck_tracking_ang_vel = 6.0
    # duck v4: tracking on a 0.3 s low-passed velocity with wider sigmas (v3: instantaneous, .0022 / .0011 -> the term
    # paid ~0 and the policy kept a +.02-.03 m/s forward drift on stand / back commands)
    duck_tracking_filter_s = 0.3
    duck_tracking_sigma_lin = 0.006  # err .05 m/s -> .66
    duck_tracking_sigma_ang = 0.004  # err .1 rad/s -> .08, .05 -> .54
    duck_tracking_y_tol = 0.03  # Open Duck 0.1 for +-0.2 m/s
    duck_torques = -7.5e-5  # Open Duck -1e-3 with ~2.9 Nm servos -> x (2.9 / 10.6)^2
    duck_action_rate = -0.5
    duck_stand_still = -1.0  # duck v4 (v3: -0.2)
    duck_ang_vel_xy = -0.5  # duck v4: trunk roll / pitch rate^2 (v3 torso gyro std 1.1-1.3 rad/s, v12 .17)
    duck_alive = 20.0
    duck_imitation = 1.0
    duck_imitation_joint_pos = 15.0
    duck_imitation_joint_vel = 1.0e-3
    duck_command_eps = 0.01

    def __post_init__(self):
        for k in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"):
            self.actuator_coulomb[k] = 0.06  # v11 fit
        for act in self.scene.robot.actuators.values():
            act.stiffness = {k: v * self.actuator_kp_scale for k, v in act.stiffness.items()}
            act.damping = {k: v * self.actuator_kd_scale for k, v in act.damping.items()}
        # TPU-like soles (v13)
        self.scene.terrain.physics_material.compliant_contact_stiffness = 4.0e4
        self.scene.terrain.physics_material.compliant_contact_damping = 400.0
