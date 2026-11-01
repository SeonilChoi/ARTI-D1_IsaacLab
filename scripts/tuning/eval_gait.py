"""Gait diagnostics for the arti_d1 AMP task (tuning tool, not part of training).

--mode reference: open-loop PD replay. Every env starts in the t=0 reference state of one clip and
    receives that clip's joint positions as PD targets each step (the same action mapping the policy
    uses). Compares the achieved heading-frame velocity with the clip's command, i.e. how much of the
    commanded speed is physically reachable with the current gains / friction / geometry.
--mode policy: deterministic policy from a checkpoint under fixed commands, reporting the achieved
    velocity per command.

    python scripts/tuning/eval_gait.py --mode reference
    python scripts/tuning/eval_gait.py --mode policy --checkpoint <agent.pt>
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["reference", "policy", "sequence", "standpush", "score"], required=True)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=96)
parser.add_argument("--seconds", type=float, default=14.0)
parser.add_argument("--warmup", type=float, default=3.0)
parser.add_argument("--kp_scale", type=float, default=1.0, help="multiply PD stiffness (damping by sqrt) - physics diagnosis")
parser.add_argument("--support_kp", type=float, default=0.0, help="training balance-support torque gain (0 = off)")
parser.add_argument("--no_randomization", action="store_true", help="disable startup domain randomization")
parser.add_argument("--uniform_action_scale", type=float, default=0.6, help="must match how the checkpoint was trained")
parser.add_argument("--motion_folder", type=str, default=None, help="reference motion set the checkpoint was trained on")
parser.add_argument("--nominal_table", type=str, default=None, help="nominal gait table the checkpoint was trained with")
parser.add_argument("--hip_roll_limit", type=float, default=None, help="hard +-limit (rad) on hip_roll, as in training")
parser.add_argument("--reset", choices=("standstill", "rsi"), default="standstill", help="sequence mode start state")
parser.add_argument("--train_step", type=int, default=0,
                    help="score mode: evaluate as at this training step (balance support / curriculum fractions); 0 = support off")
parser.add_argument("--physics", type=str, default="isaacsim_physx", help="physics preset: isaacsim_physx | newton_mjwarp")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

import arti_d1.tasks  # noqa: F401

TASK = "Template-Arti-D1-Amp-Direct-v0"
COMMANDS = [  # 28 mm / 1.625 s ranges: x +-0.0428, y +-0.0173, yaw +-0.0987
    ("forward", (0.036, 0.0, 0.0)),
    ("forward_fast", (0.042, 0.0, 0.0)),
    ("backward", (-0.036, 0.0, 0.0)),
    ("left", (0.0, 0.016, 0.0)),
    ("right", (0.0, -0.016, 0.0)),
    ("turn_left", (0.0, 0.0, 0.095)),
]

SEQ_DURATION_S = 5.0
# Commands inside the training range (22 mm, 10 deg hip_roll stance reference: x .0534 y .0215 yaw .1234) plus standing (zero command) segments; same as play.py.
# Inside the 28 mm / 1.625 s command ranges (x +-0.0428, y +-0.0173, yaw +-0.0987); the 22 mm / 1.36 s set used
# forward .05, side .02, backward -.045, diagonal (.045, -.018), turn .12.
PLAY_SEQUENCE = [
    ("forward", (0.040, 0.0, 0.0)),
    ("stand", (0.0, 0.0, 0.0)),
    ("side_left", (0.0, 0.016, 0.0)),
    ("backward", (-0.040, 0.0, 0.0)),
    ("stand", (0.0, 0.0, 0.0)),
    ("diagonal_right", (0.038, -0.015, 0.0)),
    ("turn_left", (0.0, 0.0, 0.095)),
    ("stand", (0.0, 0.0, 0.0)),
]


# --mode standpush: env e is pushed once, STANDPUSH_SETTLE_S after a standstill start under a zero command, with
# speed STANDPUSH_SPEEDS[(e // 8) % 4] (m/s, planar root velocity added) toward heading 45 deg x (e % 8) (0 = forward).
STANDPUSH_SPEEDS = (0.20, 0.30, 0.40, 0.50)
STANDPUSH_SETTLE_S = 3.0
STANDPUSH_OBSERVE_S = 4.0


class StepMeter:
    """Step length at each touchdown: (touching foot - other foot) along the heading, per foot."""

    def __init__(self, uenv, height=0.006):
        self.uenv = uenv
        self.height = height
        self.prev = None
        self.steps = {0: [[] for _ in range(uenv.num_envs)], 1: [[] for _ in range(uenv.num_envs)]}
        self.max_z = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        names = uenv.robot.data.body_names
        # foot links + sole-centre offsets (cfg left/right_toe_body_offset); corners from cfg foot_sole_half_length / width
        self.sole_ids = [uenv.left_toe_body_index, uenv.right_toe_body_index]
        self.sole_offsets = [uenv.left_toe_body_offset[0], uenv.right_toe_body_offset[0]]
        hl, hw = uenv.cfg.foot_sole_half_length, uenv.cfg.foot_sole_half_width
        self.corners = torch.tensor([[sx * hl, sy * hw, 0.0] for sx in (-1, 1) for sy in (-1, 1)], device=uenv.device)
        self.tilt_sum = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.swing_n = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.corner_sum = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.edge_drag = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.knee_ids = [names.index("left_shin"), names.index("right_shin")]
        self.knee_swing_max = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.knee_peaks = {0: [], 1: []}
        self.knee_stance_sum = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.knee_stance_n = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.rp_min = torch.full((uenv.num_envs, 2), 1e9, device=uenv.device)
        self.rp_max = torch.full((uenv.num_envs, 2), -1e9, device=uenv.device)
        self.rate_sq = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.vy_list = []
        self.air = torch.zeros((uenv.num_envs, 2), device=uenv.device)
        self.n = 0

    def feet(self):
        u = self.uenv
        l = u._get_offset_body_position(u.left_toe_body_index, u.left_toe_body_offset, u.has_left_toe_body_offset)
        r = u._get_offset_body_position(u.right_toe_body_index, u.right_toe_body_offset, u.has_right_toe_body_offset)
        return torch.stack((l, r), dim=1)

    def update(self, active=None):
        u = self.uenv
        feet = self.feet()
        contact = feet[..., 2] < self.height
        self.max_z = torch.maximum(self.max_z, feet[..., 2])
        self.air += (feet[..., 2] > 0.01).float()
        from isaaclab.utils.math import quat_apply, quat_apply_inverse, euler_xyz_from_quat, quat_from_euler_xyz
        d = u.robot.data
        pq = d.body_quat_w.torch[:, u.ref_body_index]
        r_, p_, y_ = euler_xyz_from_quat(pq)
        hq = quat_from_euler_xyz(torch.zeros_like(y_), torch.zeros_like(y_), y_)
        # pelvis roll / pitch from the world up axis in the pelvis frame
        up = quat_apply_inverse(pq, torch.tensor([[0.0, 0.0, 1.0]], device=u.device).expand(u.num_envs, -1))
        roll = torch.rad2deg(torch.atan2(up[:, 1], up[:, 2])); pitch = torch.rad2deg(torch.atan2(-up[:, 0], up[:, 2]))
        rp = torch.stack((roll, pitch), dim=-1)
        self.rp_min = torch.minimum(self.rp_min, rp); self.rp_max = torch.maximum(self.rp_max, rp)
        self.min_upz = torch.minimum(getattr(self, "min_upz", up[:, 2]), up[:, 2])  # most tilted pelvis per env
        self.max_angvel = torch.maximum(getattr(self, "max_angvel", torch.zeros_like(up[:, 2])),
                                        torch.linalg.norm(d.body_ang_vel_w.torch[:, u.ref_body_index], dim=-1))
        w = quat_apply_inverse(hq, d.body_ang_vel_w.torch[:, u.ref_body_index])
        self.rate_sq += torch.rad2deg(w[:, :2]) ** 2
        v = quat_apply_inverse(hq, d.body_lin_vel_w.torch[:, u.ref_body_index])
        self.vy_list.append(v[:, 1].clone())
        for fi, bi in enumerate(self.sole_ids):
            q = d.body_quat_w.torch[:, bi]
            up = quat_apply(q, torch.tensor([[0.0, 0.0, 1.0]], device=u.device).expand(u.num_envs, -1))
            tilt = torch.rad2deg(torch.acos(torch.clamp(up[:, 2], -1.0, 1.0)))
            cz = torch.stack([ (d.body_pos_w.torch[:, bi] + quat_apply(q, (self.sole_offsets[fi] + c).expand(u.num_envs, -1)))[:, 2] for c in self.corners ], dim=-1).min(dim=-1).values
            swing = feet[:, fi, 2] > 0.01
            self.tilt_sum[:, fi] += tilt * swing
            self.corner_sum[:, fi] += cz * swing
            self.swing_n[:, fi] += swing.float()
            # centre lifted (> 8 mm) while the lowest sole corner is still on the ground (< 3 mm): edge dragging
            self.edge_drag[:, fi] += ((feet[:, fi, 2] > 0.008) & (cz < 0.003)).float()
            kz = d.body_pos_w.torch[:, self.knee_ids[fi], 2]
            self.knee_swing_max[:, fi] = torch.where(contact[:, fi], self.knee_swing_max[:, fi], torch.maximum(self.knee_swing_max[:, fi], kz))
            self.knee_stance_sum[:, fi] += kz * contact[:, fi]
            self.knee_stance_n[:, fi] += contact[:, fi].float()
            if self.prev is not None:
                td = contact[:, fi] & ~self.prev[:, fi]
                for e in torch.nonzero(td).flatten().tolist():
                    if self.knee_swing_max[e, fi] > 0:
                        self.knee_peaks[fi].append(self.knee_swing_max[e, fi].item())
                self.knee_swing_max[:, fi] = torch.where(td, torch.zeros_like(kz), self.knee_swing_max[:, fi])
        self.n += 1
        if self.prev is not None:
            q = u.robot.data.body_quat_w.torch[:, u.ref_body_index]
            x, y, z, w = q.unbind(-1)  # Isaac Lab 3.x: (x, y, z, w)
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            head = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
            for foot in (0, 1):
                td = contact[:, foot] & ~self.prev[:, foot]
                if active is not None:
                    td = td & active
                step = ((feet[:, foot, :2] - feet[:, 1 - foot, :2]) * head).sum(-1)
                for e in torch.nonzero(td).flatten().tolist():
                    self.steps[foot][e].append(step[e].item())
        self.prev = contact

    def summary(self, env_mask):
        out = []
        for foot in (0, 1):
            vals = [v for e in range(self.uenv.num_envs) if env_mask[e] for v in self.steps[foot][e][1:]]
            out.append((float(np.mean(vals)) if vals else float("nan"), len(vals)))
        return out


def make_env():
    cfg = resolve_task_config(TASK, None, overrides=[f"physics={args_cli.physics}"])[0]
    cfg.scene.num_envs = args_cli.num_envs
    cfg.early_termination = False
    if hasattr(cfg, "balance_support_kp"):
        cfg.balance_support_kp = args_cli.support_kp  # training-only support torque, off unless diagnosing
    if args_cli.kp_scale != 1.0:
        act = cfg.scene.robot.actuators["body"]
        act.stiffness = {k: v * args_cli.kp_scale for k, v in act.stiffness.items()}
        act.damping = {k: v * args_cli.kp_scale ** 0.5 for k, v in act.damping.items()}
    if args_cli.no_randomization:
        cfg.events.physics_material = None
        cfg.events.add_pelvis_mass = None
        cfg.events.actuator_gains = None
    if hasattr(cfg, "uniform_action_scale"):
        cfg.uniform_action_scale = args_cli.uniform_action_scale
    if args_cli.motion_folder:
        cfg.motion_folder_path = args_cli.motion_folder
    if args_cli.nominal_table:
        cfg.nominal_gait_table_file = args_cli.nominal_table
    if args_cli.hip_roll_limit:
        cfg.hip_roll_limit = args_cli.hip_roll_limit
    if hasattr(cfg, "command_resample_interval_s"):
        cfg.command_resample_interval_s = (0.0, 0.0)  # the eval drives the commands itself
    cfg.episode_length_s = args_cli.seconds + 5.0
    if args_cli.mode == "score":
        # Same sequence and reset as --mode sequence; --train_step sets the curriculum (balance support) the policy had then.
        cfg.episode_length_s = SEQ_DURATION_S * len(PLAY_SEQUENCE) + 1.0
        cfg.reset_standstill_probability = 1.0
        if args_cli.train_step > 0:
            cfg.curriculum_step_offset = args_cli.train_step
            cfg.balance_support_kp = type(cfg)().balance_support_kp
    if args_cli.mode == "sequence":
        # Same conditions as scripts/skrl/play.py: trained reset strategy, pushes on, 30 s episode.
        cfg.episode_length_s = SEQ_DURATION_S * len(PLAY_SEQUENCE) + 1.0
        if hasattr(cfg, "reset_standstill_probability"):
            # standstill: every env starts from the clip's first frame (double-support standstill), as the robot does.
            # rsi (the old default, eval logs before 10-31 11:30): random clip times whose motion often disagrees with
            # the forced eval command, so most "falls" happened 0.4-0.9 s after the reset.
            cfg.reset_standstill_probability = 1.0 if args_cli.reset == "standstill" else 0.0
    elif args_cli.mode == "standpush":
        # Standstill start, zero command, one planar push per env from the eval itself (no random pushes).
        cfg.events.push_robot = None
        cfg.stand_push_velocity = (0.0, 0.0)
        cfg.reset_standstill_probability = 1.0
        cfg.episode_length_s = STANDPUSH_SETTLE_S + STANDPUSH_OBSERVE_S + 2.0
    else:
        cfg.events.push_robot = None
        cfg.reset_strategy = "random_start" if args_cli.mode == "policy" else "default"
    env = gym.make(TASK, cfg=cfg)
    return env, env.unwrapped


ROM_JOINTS = ("hip_roll", "hip_yaw", "hip_pitch", "knee", "ankle", "arm_m8")


def print_rom(tag, names, jmin, jmax, vsq, n, mask=None):
    """Per-joint peak-to-peak range (deg) and joint velocity rms (deg/s), L/R, averaged over envs in mask."""
    pp = torch.rad2deg(jmax - jmin)
    vr = torch.rad2deg(torch.sqrt(vsq / max(n, 1)))
    if mask is not None:
        pp, vr = pp[mask], vr[mask]
    pp, vr = pp.mean(0).tolist(), vr.mean(0).tolist()
    parts = []
    for jn in ROM_JOINTS:
        li = [i for i, x in enumerate(names) if x.startswith("left_") and x.endswith(jn)]
        ri = [i for i, x in enumerate(names) if x.startswith("right_") and x.endswith(jn)]
        if li and ri:
            parts.append(f"{jn} {pp[li[0]]:.1f}/{pp[ri[0]]:.1f} ({vr[li[0]]:.0f}/{vr[ri[0]]:.0f})")
    print(f"       ROM p-p deg L/R (vel rms deg/s) [{tag}]: " + " | ".join(parts))


def print_torque(tag, names, tsq, pw, n, mask=None):
    """Per-joint torque rms (Nm) L/R and mean total mechanical power sum|tau*qd| (W)."""
    tr = torch.sqrt(tsq / max(n, 1))
    p = pw / max(n, 1)
    if mask is not None:
        tr, p = tr[mask], p[mask]
    tr = tr.mean(0).tolist()
    parts = []
    for jn in ROM_JOINTS:
        li = [i for i, x in enumerate(names) if x.startswith("left_") and x.endswith(jn)]
        ri = [i for i, x in enumerate(names) if x.startswith("right_") and x.endswith(jn)]
        if li and ri:
            parts.append(f"{jn} {tr[li[0]]:.2f}/{tr[ri[0]]:.2f}")
    allr = math.sqrt(sum(v * v for v in tr) / len(tr))
    print(f"       TORQUE rms Nm L/R [{tag}]: " + " | ".join(parts) + f" || all {allr:.2f} Nm, power {p.mean().item():.2f} W")


def accumulate_torque(uenv, tsq, pw):
    tau = uenv.robot.actuators.applied_effort.torch
    tsq += tau ** 2
    pw += (tau * uenv.robot.data.joint_vel.torch).abs().sum(-1)


def heading_velocity(uenv):
    idx = uenv.ref_body_index
    quat_xyzw = uenv.robot.data.body_quat_w.torch[:, idx]
    _, lin, ang = uenv.motion_loader.normalize_heading_observation(
        quat_xyzw, uenv.robot.data.body_lin_vel_w.torch[:, idx], uenv.robot.data.body_ang_vel_w.torch[:, idx]
    )
    return torch.cat((lin[:, :2], ang[:, 2:3]), dim=-1)


def trunk_height(uenv):
    # Same rule as the env termination: the trunk offset rotates with the pelvis, and a tilt past
    # termination_max_tilt_rad counts as a fall. The old unrotated offset (+0.15 m) kept a pelvis lying at 0.09 m
    # above the 0.2 m threshold, so fallen / flipped D1 robots were reported as fell=0.
    from isaaclab.utils.math import quat_apply
    d = uenv.robot.data
    q = d.body_quat_w.torch[:, uenv.termination_body_index]
    z = (d.body_pos_w.torch[:, uenv.termination_body_index] + quat_apply(q, uenv.termination_body_offset.expand(uenv.num_envs, -1)))[:, 2]
    if uenv.termination_min_up_z is not None:
        up_z = quat_apply(q, uenv.up_axis_b.expand(uenv.num_envs, -1))[:, 2]
        z = torch.where(up_z < uenv.termination_min_up_z, torch.zeros_like(z), z)
    return z


def write_reference_state(uenv, motion_ids, t):
    env_ids = torch.arange(uenv.num_envs, device=uenv.device)
    rp, rq, rv, rw, jp, jv, _, _ = uenv.motion_loader.sample(
        uenv.num_envs, motion_ids=motion_ids, times=np.full(uenv.num_envs, t), return_full_state=True
    )
    root = uenv._default_root_state(env_ids)
    root[:, :3] = rp + uenv.scene.env_origins
    root[:, 3:7] = rq  # motion loader and Isaac Lab 3.x are both (x, y, z, w)
    root[:, 7:10] = rv
    root[:, 10:13] = rw
    uenv.robot.write_root_link_pose_to_sim_index(root_pose=root[:, :7], env_ids=env_ids)
    uenv.robot.write_root_com_velocity_to_sim_index(root_velocity=root[:, 7:], env_ids=env_ids)
    uenv.robot.write_joint_position_to_sim_index(position=jp, env_ids=env_ids)
    uenv.robot.write_joint_velocity_to_sim_index(velocity=jv, env_ids=env_ids)


def reference_actions(uenv, motion_ids, t):
    jp = uenv.motion_loader.sample(
        uenv.num_envs, motion_ids=motion_ids, times=np.full(uenv.num_envs, t), return_full_state=True
    )[4]
    delta = jp - uenv.robot.data.default_joint_pos.torch
    scale = torch.where(delta >= 0.0, uenv.action_positive_scale, uenv.action_negative_scale)
    return delta / scale


class Scaler(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.mean = sd["running_mean"].float()
        self.std = torch.sqrt(sd["running_variance"].float()) + 1e-8

    def forward(self, x):
        return torch.clamp((x - self.mean) / self.std, -5.0, 5.0)


def load_policy(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["policy"]
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("net_container.") and k.endswith(".weight")})
    has_out = "output_layer.weight" in sd
    layers = []
    for n, i in enumerate(idx):
        w, b = sd[f"net_container.{i}.weight"], sd[f"net_container.{i}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if n < len(idx) - 1 or has_out:
            layers.append(nn.ReLU())
    if has_out:
        out = nn.Linear(sd["output_layer.weight"].shape[1], sd["output_layer.weight"].shape[0])
        out.weight.data.copy_(sd["output_layer.weight"])
        out.bias.data.copy_(sd["output_layer.bias"])
        layers.append(out)
    actor = nn.Sequential(*layers).to(device).eval()
    pre = ck.get("observation_preprocessor") or ck.get("state_preprocessor")
    scaler = Scaler({k: v.to(device) for k, v in pre.items()}) if pre else nn.Identity()
    return lambda obs: actor(scaler(obs))


SCORE_SKIP_S = 1.0  # transition after each command change, not scored


def run_score(env, uenv, obs):
    """Walking-performance scorecard over PLAY_SEQUENCE: stability, tracking, smoothness, gait quality, effort.

    Every metric is accumulated only over env-steps before that env's first fall, separately for walking and
    standing segments, so a policy that falls early is not rewarded with fewer bad samples (falls are reported).
    """
    from isaaclab.utils.math import quat_apply_inverse, euler_xyz_from_quat, quat_from_euler_xyz
    policy = load_policy(args_cli.checkpoint, uenv.device)
    n, dev, dt = uenv.num_envs, uenv.device, uenv.step_dt
    d = uenv.robot.data
    seg_steps = int(SEQ_DURATION_S / dt)
    skip = int(SCORE_SKIP_S / dt)
    table = torch.tensor([c for _, c in PLAY_SEQUENCE], dtype=torch.float32, device=dev)
    feet_ids = [uenv.left_toe_body_index, uenv.right_toe_body_index]
    mass = d.body_mass.torch.to(dev).reshape(n, -1).sum(-1)
    acc = {}  # name -> (sum per env, count per env), per kind "walk" / "stand"

    def add(kind, name, value, mask):
        s, c = acc.setdefault((kind, name), (torch.zeros(n, device=dev), torch.zeros(n, device=dev)))
        s += torch.where(mask, value, torch.zeros_like(value)); c += mask.float()

    events = {}  # per-event samples (touchdown impact, swing height, swing time) as python lists

    def add_events(kind, name, value, mask):
        events.setdefault((kind, name), []).extend(value[mask].tolist())

    fall_t = torch.full((n,), -1.0, device=dev)
    min_h = torch.full((n,), 1e9, device=dev)
    prev_a = prev_a2 = prev_qd = prev_v = None
    prev_contact = torch.ones((n, 2), dtype=torch.bool, device=dev)
    prev_vz = torch.zeros((n, 2), device=dev)
    swing_t = torch.zeros((n, 2), device=dev); swing_h = torch.zeros((n, 2), device=dev)
    up_w = torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(n, -1)
    seg_hv = torch.zeros((n, 3), device=dev); seg_c = torch.zeros(n, device=dev)
    with torch.inference_mode():
        for k in range(seg_steps * len(PLAY_SEQUENCE)):
            j = k // seg_steps
            cmd = table[j].expand(n, -1)
            kind = "stand" if PLAY_SEQUENCE[j][0] == "stand" else "walk"
            uenv.commands[:] = cmd
            pobs = obs["policy"].clone()
            pobs[:, -3:] = cmd * uenv.command_scale
            a = torch.clamp(policy(pobs), -1.0, 1.0)
            obs, _, _, _, _ = env.step(a)
            min_h = torch.minimum(min_h, trunk_height(uenv))
            fall_t = torch.where((min_h < 0.2) & (fall_t < 0), torch.full_like(fall_t, k * dt), fall_t)
            alive = fall_t < 0
            m = alive & (k % seg_steps >= skip)
            pq = d.body_quat_w.torch[:, uenv.ref_body_index]
            _, _, yaw = euler_xyz_from_quat(pq)
            hq = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
            v_h = quat_apply_inverse(hq, d.body_lin_vel_w.torch[:, uenv.ref_body_index])
            w_h = quat_apply_inverse(hq, d.body_ang_vel_w.torch[:, uenv.ref_body_index])
            up = quat_apply_inverse(pq, up_w)
            roll = torch.rad2deg(torch.atan2(up[:, 1], up[:, 2])); pitch = torch.rad2deg(torch.atan2(-up[:, 0], up[:, 2]))
            qd, tau = d.joint_vel.torch, uenv.robot.actuators.applied_effort.torch
            hv = heading_velocity(uenv)
            # stability
            add(kind, "roll_sq", roll ** 2, m); add(kind, "pitch_sq", pitch ** 2, m)
            add(kind, "rollrate_sq", torch.rad2deg(w_h[:, 0]) ** 2, m); add(kind, "pitchrate_sq", torch.rad2deg(w_h[:, 1]) ** 2, m)
            # tracking (heading frame), x / y in mm/s, yaw in deg/s
            # tracking: segment-mean velocity vs command (the gait's own oscillation is the sway term below)
            seg_hv += torch.where(m.unsqueeze(-1), hv, torch.zeros_like(hv)); seg_c += m.float()
            if k % seg_steps == seg_steps - 1:
                ok = alive & (seg_c > 0)
                err = seg_hv / seg_c.clamp_min(1.0).unsqueeze(-1) - cmd
                add(kind, "err_x_sq", (err[:, 0] * 1000) ** 2, ok); add(kind, "err_y_sq", (err[:, 1] * 1000) ** 2, ok)
                add(kind, "err_yaw_sq", torch.rad2deg(err[:, 2]) ** 2, ok)
                seg_hv.zero_(); seg_c.zero_()
            add(kind, "sway_sq", (v_h[:, 1] * 1000) ** 2, m)
            # smoothness
            if prev_a is not None:
                add(kind, "action_rate_sq", ((a - prev_a) ** 2).mean(-1), m)
                add(kind, "joint_acc_sq", (((qd - prev_qd) / dt) ** 2).mean(-1), m)
                pa = (v_h - prev_v) / dt
                add(kind, "pelvis_acc_y_sq", pa[:, 1] ** 2, m); add(kind, "pelvis_acc_z_sq", pa[:, 2] ** 2, m)
                if prev_a2 is not None:
                    add(kind, "action_jerk_sq", ((a - 2 * prev_a + prev_a2) ** 2).mean(-1), m)
            prev_a2, prev_a, prev_qd, prev_v = prev_a, a.clone(), qd.clone(), v_h.clone()
            # effort
            power = (tau * qd).abs().sum(-1)
            add(kind, "torque_sq", (tau ** 2).mean(-1), m); add(kind, "power", power, m)
            add(kind, "cot_num", power, m); add(kind, "cot_den", mass * 9.81 * torch.linalg.norm(v_h[:, :2], dim=-1), m)
            # feet: sole height contact, slip, touchdown impact, swing height / time, stance width
            fz = torch.stack([uenv._get_offset_body_position(uenv.left_toe_body_index, uenv.left_toe_body_offset, uenv.has_left_toe_body_offset),
                              uenv._get_offset_body_position(uenv.right_toe_body_index, uenv.right_toe_body_offset, uenv.has_right_toe_body_offset)], dim=1)
            contact = fz[..., 2] < 0.006
            fv = d.body_lin_vel_w.torch[:, feet_ids]
            slip = torch.linalg.norm(fv[..., :2], dim=-1) * 1000
            for fi in range(2):
                add(kind, "slip", slip[:, fi], m & contact[:, fi])
                td = contact[:, fi] & ~prev_contact[:, fi]
                add_events(kind, "impact", -prev_vz[:, fi] * 1000, m & td & (swing_t[:, fi] >= 3 * dt))
                add_events(kind, "swing_h", swing_h[:, fi] * 1000, m & td & (swing_t[:, fi] >= 3 * dt))
                add_events(kind, "swing_t_" + "LR"[fi], swing_t[:, fi], m & td & (swing_t[:, fi] >= 3 * dt))
                add(kind, "lift", (~contact[:, fi] & prev_contact[:, fi]).float(), m)
            swing_t = torch.where(contact, torch.zeros_like(swing_t), swing_t + dt)
            swing_h = torch.where(contact, torch.zeros_like(swing_h), torch.maximum(swing_h, fz[..., 2]))
            prev_contact, prev_vz = contact, fv[..., 2].clone()
            lr = fz[:, 0] - fz[:, 1]
            add(kind, "width", (-torch.sin(yaw) * lr[:, 0] + torch.cos(yaw) * lr[:, 1]) * 1000, m)
            add(kind, "air", (~contact).float().mean(-1), m)

    def mean(kind, name, rms=False):
        s, c = acc[(kind, name)]
        ok = c > 0
        if not bool(ok.any()):
            return float("nan")
        v = (s[ok] / c[ok])
        return v.sqrt().mean().item() if rms else v.mean().item()

    def ev(kind, name, q=None):
        x = events.get((kind, name), [])
        if not x:
            return float("nan")
        return float(np.mean(x)) if q is None else float(np.quantile(x, q))

    fell = fall_t >= 0
    total = seg_steps * len(PLAY_SEQUENCE) * dt
    alive_t = torch.where(fell, fall_t, torch.full_like(fall_t, total))
    print(f"[SCORE] checkpoint={args_cli.checkpoint} physics={args_cli.physics} envs={n} train_step={args_cli.train_step} "
          f"(balance support {'as in training' if args_cli.train_step else 'off'})")
    print(f"[SCORE] stability  falls {int(fell.sum())}/{n} | mean time up {alive_t.mean().item():.1f}/{total:.0f} s | "
          f"falls per env-minute {fell.sum().item() / (alive_t.sum().item() / 60.0):.2f}")
    for kind in ("walk", "stand"):
        if (kind, "roll_sq") not in acc:
            continue
        cot = mean(kind, "cot_num") / max(mean(kind, "cot_den"), 1e-9)
        tl, tr = ev(kind, "swing_t_L"), ev(kind, "swing_t_R")
        print(f"[SCORE] {kind:5s} stability   roll rms {mean(kind, 'roll_sq', True):.2f} pitch rms {mean(kind, 'pitch_sq', True):.2f} deg | "
              f"roll-rate rms {mean(kind, 'rollrate_sq', True):.1f} pitch-rate rms {mean(kind, 'pitchrate_sq', True):.1f} deg/s")
        print(f"[SCORE] {kind:5s} tracking    err rms x {mean(kind, 'err_x_sq', True):.1f} y {mean(kind, 'err_y_sq', True):.1f} mm/s "
              f"yaw {mean(kind, 'err_yaw_sq', True):.2f} deg/s (segment mean vs command) | lateral sway vel rms {mean(kind, 'sway_sq', True):.0f} mm/s")
        print(f"[SCORE] {kind:5s} smoothness  action rate rms {mean(kind, 'action_rate_sq', True):.4f} jerk rms {mean(kind, 'action_jerk_sq', True):.4f} | "
              f"joint acc rms {mean(kind, 'joint_acc_sq', True):.1f} rad/s2 | pelvis acc rms y {mean(kind, 'pelvis_acc_y_sq', True):.2f} "
              f"z {mean(kind, 'pelvis_acc_z_sq', True):.2f} m/s2")
        print(f"[SCORE] {kind:5s} feet        touchdown speed mean {ev(kind, 'impact'):.0f} p95 {ev(kind, 'impact', 0.95):.0f} mm/s | "
              f"stance slip {mean(kind, 'slip'):.1f} mm/s | swing height {ev(kind, 'swing_h'):.1f} mm | "
              f"swing time L {tl:.3f} R {tr:.3f} s (sym {min(tl, tr) / max(tl, tr):.2f}) | lifts/s {mean(kind, 'lift') * 2 / dt:.2f} | "
              f"air {mean(kind, 'air'):.2f} | width {mean(kind, 'width'):.1f} mm")
        print(f"[SCORE] {kind:5s} effort      torque rms {mean(kind, 'torque_sq', True):.3f} Nm | power {mean(kind, 'power'):.2f} W"
              + (f" | CoT {cot:.1f}" if kind == "walk" else ""))


def main():
    env, uenv = make_env()
    obs, _ = env.reset()
    dt = uenv.step_dt
    steps = int(args_cli.seconds / dt)
    warm = int(args_cli.warmup / dt)
    vel_sum = torch.zeros((uenv.num_envs, 3), device=uenv.device)
    min_h = torch.full((uenv.num_envs,), 1e9, device=uenv.device)

    if args_cli.mode == "reference":
        motion_ids = np.arange(uenv.num_envs) % uenv.motion_loader._n_motions
        target = uenv.motion_loader.get_motion_commands(motion_ids, use_average=True)
        write_reference_state(uenv, motion_ids, 0.0)
        nj = len(uenv.robot.data.joint_names)
        jmin = torch.full((uenv.num_envs, nj), 1e9, device=uenv.device); jmax = -jmin.clone()
        vsq = torch.zeros((uenv.num_envs, nj), device=uenv.device); nr = 0
        tsq = torch.zeros((uenv.num_envs, nj), device=uenv.device); pw = torch.zeros(uenv.num_envs, device=uenv.device)
        ref_meter = StepMeter(uenv)
        with torch.inference_mode():
            for k in range(steps):
                acts = reference_actions(uenv, motion_ids, (k + 1) * dt)
                env.step(acts)
                if k % 30 == 0 and k < 240:
                    d = uenv.robot.data
                    ref = uenv.motion_loader.sample(uenv.num_envs, motion_ids=motion_ids,
                                                    times=np.full(uenv.num_envs, (k + 1) * dt), return_full_state=True)
                    print(f"[DBG] t={(k+1)*dt:.2f} pelvis_w={[round(v,3) for v in (d.body_pos_w.torch[0, uenv.ref_body_index]-uenv.scene.env_origins[0]).tolist()]} "
                          f"ref_root={[round(v,3) for v in ref[0][0].tolist()]}")
                    print(f"      act={[round(v,2) for v in acts[0].tolist()]}")
                    print(f"      q_ref={[round(v,3) for v in ref[4][0].tolist()]}")
                    print(f"      q_sim={[round(v,3) for v in d.joint_pos.torch[0].tolist()]}")
                    print(f"      q_tgt={[round(v,3) for v in uenv.robot.actuators.target_command.position.torch[0].tolist()]}")
                    lt = d.body_pos_w.torch[0, uenv.left_toe_body_index, 2].item(); rt = d.body_pos_w.torch[0, uenv.right_toe_body_index, 2].item()
                    print(f"      footlink_z L={lt:.3f} R={rt:.3f} names={d.joint_names if k==0 else ''}")
                if k >= warm:
                    ref_meter.update()
                    vel_sum += heading_velocity(uenv)
                    qj = uenv.robot.data.joint_pos.torch; jmin = torch.minimum(jmin, qj); jmax = torch.maximum(jmax, qj)
                    vsq += uenv.robot.data.joint_vel.torch ** 2; nr += 1
                    accumulate_torque(uenv, tsq, pw)
                min_h = torch.minimum(min_h, trunk_height(uenv))
        vel = vel_sum / (steps - warm)
        fell = min_h < 0.2
        ok = ~fell
        err = vel - target
        planar = torch.linalg.norm(target[:, :2], dim=-1)
        ratio = (vel[:, :2] * target[:, :2]).sum(-1) / planar.clamp_min(1e-6) ** 2
        big = planar > 0.03
        lin_r = torch.exp(-(err[:, :2] ** 2).sum(-1) / uenv.cfg.linear_velocity_tracking_sigma)
        ang_r = torch.exp(-(err[:, 2] ** 2) / uenv.cfg.angular_velocity_tracking_sigma)
        print(f"[EVAL-REF] clips={uenv.num_envs} fell={int(fell.sum())}")
        print(f"[EVAL-REF] speed ratio (projection on cmd, |cmd|>0.03, standing): "
              f"mean {ratio[big & ok].mean():.3f} median {ratio[big & ok].median():.3f}")
        print(f"[EVAL-REF] exp tracking terms (standing): lin {lin_r[ok].mean():.3f} ang {ang_r[ok].mean():.3f}")
        print(f"[EVAL-REF] mean |err| x {err[ok, 0].abs().mean():.4f} y {err[ok, 1].abs().mean():.4f} yaw {err[ok, 2].abs().mean():.4f}")
        fwd = ok & (target[:, 0] > 0.035) & (target[:, 1].abs() < 0.01) & (target[:, 2].abs() < 0.02)
        print(f"[EVAL-REF] forward clips n={int(fwd.sum())} mean cmd {target[fwd].mean(0).tolist() if bool(fwd.any()) else None}")
        if bool(fwd.any()):
            print_rom('reference forward', uenv.robot.data.joint_names, jmin, jmax, vsq, nr, fwd)
        print_rom('reference all', uenv.robot.data.joint_names, jmin, jmax, vsq, nr, ok)
        print_torque('reference all', uenv.robot.data.joint_names, tsq, pw, nr, ok)
        kst = (ref_meter.knee_stance_sum / ref_meter.knee_stance_n.clamp_min(1.0))[ok].mean(0).tolist()
        kpk = [float(np.mean(ref_meter.knee_peaks[i])) if ref_meter.knee_peaks[i] else float("nan") for i in (0, 1)]
        mz = ref_meter.max_z[ok].mean(0).tolist()
        print(f"[EVAL-REF] knee lift (peak - stance) L {(kpk[0]-kst[0])*1000:.1f} R {(kpk[1]-kst[1])*1000:.1f} mm | foot max height L {mz[0]*1000:.1f} R {mz[1]*1000:.1f} mm")
        for i in range(min(12, uenv.num_envs)):
            print(f"  clip{motion_ids[i]:4d} cmd {target[i].tolist()} got {[round(v, 4) for v in vel[i].tolist()]} fell={bool(fell[i])}")
    elif args_cli.mode == "policy":
        policy = load_policy(args_cli.checkpoint, uenv.device)
        cmd_ids = torch.arange(uenv.num_envs, device=uenv.device) % len(COMMANDS)
        table = torch.tensor([c for _, c in COMMANDS], dtype=torch.float32, device=uenv.device)
        cmds = table[cmd_ids]
        meter = StepMeter(uenv)
        with torch.inference_mode():
            for k in range(steps):
                uenv.commands[:] = cmds
                pobs = obs["policy"].clone()
                pobs[:, -3:] = cmds * uenv.command_scale
                obs, _, _, _, _ = env.step(torch.clamp(policy(pobs), -1.0, 1.0))
                if k >= warm:
                    vel_sum += heading_velocity(uenv)
                    meter.update()
                min_h = torch.minimum(min_h, trunk_height(uenv))
        vel = vel_sum / (steps - warm)
        fell = min_h < 0.2
        print(f"[EVAL-POL] checkpoint={args_cli.checkpoint}")
        for j, (name, c) in enumerate(COMMANDS):
            m = (cmd_ids == j).tolist()
            (ls, ln), (rs, rn) = meter.summary(m)
            print(f"  steps {name:13s} left {ls:+.4f} (n={ln})  right {rs:+.4f} (n={rn})")
            mt = torch.tensor(m, device=uenv.device)
            mz = meter.max_z[mt].mean(0).tolist(); air = (meter.air[mt] / max(meter.n, 1)).mean(0).tolist()
            print(f"        foot max height L {mz[0]*1000:.1f} mm R {mz[1]*1000:.1f} mm | time >10mm L {air[0]:.2f} R {air[1]:.2f}")
        for j, (name, c) in enumerate(COMMANDS):
            m = cmd_ids == j
            v = vel[m & ~fell].mean(0).tolist() if bool((m & ~fell).any()) else [float("nan")] * 3
            print(f"  {name:13s} cmd {c} got ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}) fell {int((m & fell).sum())}/{int(m.sum())}")
    if args_cli.mode == "sequence":
        policy = load_policy(args_cli.checkpoint, uenv.device)
        seg_steps = int(SEQ_DURATION_S / dt)
        meter = StepMeter(uenv)
        seg_vel = torch.zeros((len(PLAY_SEQUENCE), uenv.num_envs, 3), device=uenv.device)
        seg_n = [0] * len(PLAY_SEQUENCE)
        seg_steps_lr = []
        seg_rom = []
        seg_torque = []
        all_tau, all_vel = [], []  # |tau|, |qd| per measured step, for motor sizing (p99 / max)
        seg_feet = []
        seg_tilt = []
        seg_knee = []
        seg_base = []
        seg_joint = torch.zeros((len(PLAY_SEQUENCE), len(uenv.robot.data.joint_names)), device=uenv.device)
        seg_footyaw = torch.zeros((len(PLAY_SEQUENCE), 2), device=uenv.device)
        seg_width = torch.zeros(len(PLAY_SEQUENCE), device=uenv.device)  # lateral sole distance in the pelvis heading frame
        table = torch.tensor([c for _, c in PLAY_SEQUENCE], dtype=torch.float32, device=uenv.device)
        fall_t = torch.full((uenv.num_envs,), -1.0, device=uenv.device)  # time of the first fall per env (s)
        with torch.inference_mode():
            for k in range(seg_steps * len(PLAY_SEQUENCE)):
                j = k // seg_steps
                if k % seg_steps == 0:
                    meter = StepMeter(uenv)
                    nj = len(uenv.robot.data.joint_names)
                    jmin = torch.full((uenv.num_envs, nj), 1e9, device=uenv.device); jmax = -jmin.clone()
                    vsq = torch.zeros((uenv.num_envs, nj), device=uenv.device); nr = 0
                    tsq = torch.zeros((uenv.num_envs, nj), device=uenv.device); pw = torch.zeros(uenv.num_envs, device=uenv.device)
                cmds = table[j].expand(uenv.num_envs, -1)
                uenv.commands[:] = cmds
                pobs = obs["policy"].clone()
                pobs[:, -3:] = cmds * uenv.command_scale
                raw = policy(pobs)
                if os.environ.get("EVAL_FALL_DEBUG") and k in (0, 1, 2, 5, 10, 15, 20, 25, 30, 60, 120):
                    print(f"[RAW] t {k * dt:.2f} |raw| max {raw.abs().max().item():.2f} mean {raw.abs().mean().item():.2f} "
                          f"frac |raw|>1 {(raw.abs() > 1).float().mean().item():.2f} | |obs| max {pobs.abs().max().item():.1f} "
                          f"(idx {int(pobs.abs().max(0).values.argmax())})")
                obs, _, _, _, _ = env.step(torch.clamp(raw, -1.0, 1.0))
                up_env = (fall_t < 0).float().unsqueeze(-1)  # envs that have not fallen yet
                sat = torch.stack((((raw.abs() > 1.0).float() * up_env).sum(0), up_env.sum(0).expand(raw.shape[1])))
                raw_sat = raw_sat + sat if k else sat
                if k == seg_steps * len(PLAY_SEQUENCE) - 1:  # per-joint share of standing env-steps with |raw| > 1
                    frac = (raw_sat[0] / raw_sat[1].clamp_min(1.0)).tolist()
                    print("[RAW-SAT] " + " ".join(f"{n}:{v:.2f}" for n, v in zip(uenv.robot.data.joint_names, frac)))
                if os.environ.get("EVAL_STEP_DEBUG"):  # env 0 liftoff / touchdown order with the clock phase
                    up = (meter.feet()[0, :, 2] > 0.008).tolist()
                    for fi, side in enumerate("LR"):
                        if k > 0 and up[fi] != step_up[fi]:
                            print(f"[STEP] t {k * dt:6.2f} {PLAY_SEQUENCE[j][0]:14s} {side} {'lift' if up[fi] else 'down'} "
                                  f"phase {uenv.gait_phase[0].item():.3f}")
                    step_up = up
                if k % seg_steps >= int(2.0 / dt):
                    seg_vel[j] += heading_velocity(uenv)
                    seg_n[j] += 1
                    meter.update()
                    d = uenv.robot.data
                    names = d.joint_names
                    seg_joint[j] += d.joint_pos.torch.mean(0)
                    jmin = torch.minimum(jmin, d.joint_pos.torch); jmax = torch.maximum(jmax, d.joint_pos.torch)
                    vsq += d.joint_vel.torch ** 2; nr += 1
                    accumulate_torque(uenv, tsq, pw)
                    all_tau.append(uenv.robot.actuators.applied_effort.torch.abs().clone()); all_vel.append(d.joint_vel.torch.abs().clone())
                    def yaw_of(q):
                        x, y, z, w = q.unbind(-1)  # Isaac Lab 3.x: (x, y, z, w)
                        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                    py = yaw_of(d.body_quat_w.torch[:, uenv.ref_body_index])
                    lr = meter.feet()[:, 0] - meter.feet()[:, 1]
                    seg_width[j] += (-torch.sin(py) * lr[:, 0] + torch.cos(py) * lr[:, 1]).mean()
                    for fi, bi in enumerate((uenv.left_toe_body_index, uenv.right_toe_body_index)):
                        dy = torch.remainder(yaw_of(d.body_quat_w.torch[:, bi]) - py + math.pi, 2 * math.pi) - math.pi
                        seg_footyaw[j, fi] += dy.mean()
                min_h = torch.minimum(min_h, trunk_height(uenv))
                new_fall = (min_h < 0.2) & (fall_t < 0)
                if bool(new_fall.any()) and os.environ.get("EVAL_FALL_DEBUG"):
                    dd = uenv.robot.data; e = int(torch.nonzero(new_fall)[0])
                    from isaaclab.utils.math import quat_apply as _qa
                    qq = dd.body_quat_w.torch[e:e + 1, uenv.termination_body_index]
                    print(f"[FALL] t {k * dt:.2f} env {e} pelvis z {dd.body_pos_w.torch[e, uenv.termination_body_index, 2].item():.3f} "
                          f"up_z {_qa(qq, uenv.up_axis_b[:1] if uenv.up_axis_b.dim() > 1 else uenv.up_axis_b[None])[0, 2].item():.3f} "
                          f"lin vel {[round(v, 2) for v in dd.root_lin_vel_w.torch[e].tolist()]} ang vel {[round(v, 2) for v in dd.root_ang_vel_w.torch[e].tolist()]} "
                          f"action {[round(v, 2) for v in uenv.actions[e].tolist()]}")
                fall_t = torch.where(new_fall, torch.full_like(fall_t, k * dt), fall_t)
                if k % seg_steps == seg_steps - 1:
                    seg_steps_lr.append(meter.summary([True] * uenv.num_envs))
                    seg_rom.append((jmin.clone(), jmax.clone(), vsq.clone(), nr))
                    seg_torque.append((tsq.clone(), pw.clone(), nr))
                    seg_feet.append((meter.max_z.mean(0).tolist(), (meter.air / max(meter.n, 1)).mean(0).tolist()))
                    kst = (meter.knee_stance_sum / meter.knee_stance_n.clamp_min(1.0)).mean(0).tolist()
                    kpk = [float(np.mean(meter.knee_peaks[i])) if meter.knee_peaks[i] else float("nan") for i in (0, 1)]
                    seg_knee.append((kpk, kst))
                    vy = torch.stack(meter.vy_list, dim=0)
                    rpr_env = meter.rp_max - meter.rp_min
                    print(f"[EVAL-SEQ] seg {len(seg_base) + 1} roll range per env: median {rpr_env[:, 0].median().item():.2f} "
                          f"max {rpr_env[:, 0].max().item():.2f} (env {int(rpr_env[:, 0].argmax())}) | envs > 30 deg "
                          f"{int((rpr_env[:, 0] > 30.0).sum())} | min height {min_h.min().item():.3f} m | pelvis up_z min "
                          f"{meter.min_upz.min().item():.2f}, envs < 0.5: {int((meter.min_upz < 0.5).sum())} | max pelvis ang vel "
                          f"{meter.max_angvel.max().item():.1f} rad/s (median {meter.max_angvel.median().item():.1f})")
                    seg_base.append((rpr_env.mean(0).tolist(),
                                     torch.sqrt(meter.rate_sq / max(meter.n, 1)).mean(0).tolist(),
                                     (vy - vy.mean(0, keepdim=True)).std(0).mean().item()))
                    sn = meter.swing_n.clamp_min(1.0)
                    seg_tilt.append(((meter.tilt_sum / sn).mean(0).tolist(), (meter.corner_sum / sn).mean(0).tolist(),
                                     (meter.edge_drag / max(meter.n, 1)).mean(0).tolist()))
        fell = min_h < 0.2
        print(f"[EVAL-SEQ] checkpoint={args_cli.checkpoint} envs={uenv.num_envs} fell={int(fell.sum())}")
        if bool(fell.any()):
            print(f"[EVAL-SEQ] fall times s: {sorted(round(v, 2) for v in fall_t[fell].tolist())}")
        for j, (name, c) in enumerate(PLAY_SEQUENCE):
            v = (seg_vel[j] / max(seg_n[j], 1))[~fell].mean(0).tolist()
            (ls, ln), (rs, rn) = seg_steps_lr[j]
            print(f"  {j+1}. {name:15s} cmd {c} got ({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f})  step L {ls:+.4f} R {rs:+.4f}")
            mz, air = seg_feet[j]
            tl, cz, ed = seg_tilt[j]
            kpk, kst = seg_knee[j]
            rpr, rate, vyo = seg_base[j]
            print(f"       base roll range {rpr[0]:.2f} pitch range {rpr[1]:.2f} deg | roll-rate rms {rate[0]:.1f} pitch-rate rms {rate[1]:.1f} deg/s | lateral vel osc std {vyo*1000:.1f} mm/s")
            print(f"       knee peak L {kpk[0]*1000:.1f} R {kpk[1]*1000:.1f} mm | knee lift (peak - stance) L {(kpk[0]-kst[0])*1000:.1f} R {(kpk[1]-kst[1])*1000:.1f} mm")
            print(f"       swing sole tilt L {tl[0]:.1f} R {tl[1]:.1f} deg | lowest corner in swing L {cz[0]*1000:.1f} R {cz[1]*1000:.1f} mm | edge-drag time L {ed[0]:.2f} R {ed[1]:.2f}")
            print(f"       foot max height L {mz[0]*1000:.1f} mm R {mz[1]*1000:.1f} mm | time >10mm L {air[0]:.2f} R {air[1]:.2f}")
            print_rom(name, uenv.robot.data.joint_names, *seg_rom[j], ~fell)
            print_torque(name, uenv.robot.data.joint_names, *seg_torque[j], ~fell)
            jn = uenv.robot.data.joint_names; jm = (seg_joint[j] / max(seg_n[j], 1)).tolist()
            fy = (seg_footyaw[j] / max(seg_n[j], 1)).tolist()
            print(f"       hip_roll mean L {jm[jn.index('left_hip_roll')]:+.4f} R {jm[jn.index('right_hip_roll')]:+.4f} rad "
                  f"(default -/+0.0247) | sole width {seg_width[j].item() / max(seg_n[j], 1) * 1000:.1f} mm (reference 135)")
            print(f"       hip_yaw mean L {jm[jn.index('left_hip_yaw')]:+.4f} R {jm[jn.index('right_hip_yaw')]:+.4f} rad | "
                  f"foot yaw rel. pelvis L {math.degrees(fy[0]):+.2f} deg R {math.degrees(fy[1]):+.2f} deg")
        # Motor sizing over the whole sequence (non-fallen envs): torque rms / p99 / max (Nm), speed rms / p99 / max (rpm).
        tau = torch.stack(all_tau)[:, ~fell]; vel = torch.stack(all_vel)[:, ~fell]
        tau = tau.reshape(-1, tau.shape[-1]); vel = vel.reshape(-1, vel.shape[-1])
        rpm = 60.0 / (2 * math.pi)
        print("[MOTOR] joint              torque Nm rms / p99 / max      speed rpm rms / p99 / max   (deg/s max)")
        for i, jn in enumerate(uenv.robot.data.joint_names):
            t, v = tau[:, i], vel[:, i]
            print(f"[MOTOR] {jn:18s} {t.pow(2).mean().sqrt():6.2f} / {torch.quantile(t, 0.99):6.2f} / {t.max():6.2f}"
                  f"      {v.pow(2).mean().sqrt()*rpm:6.1f} / {torch.quantile(v, 0.99)*rpm:6.1f} / {v.max()*rpm:6.1f}   ({math.degrees(v.max()):6.0f})")
        # Dynamixel servos (robots/arti_d1.py: legs XH540-W270, arms XM430-W350): check each (speed, torque) sample against
        # the linear torque-speed line from the stall torque to 0 Nm at the no-load speed.
        from arti_d1.robots.arti_d1 import joint_motor
        for i, jn in enumerate(uenv.robot.data.joint_names):
            motor = joint_motor(jn)
            t, v = tau[:, i].abs(), vel[:, i].abs()
            avail = motor["stall_torque"] * (1.0 - v / motor["no_load_speed"]).clamp_min(0.0)
            over = t > avail
            print(f"[DXL] {jn:24s} time > no-load speed {100*(v > motor['no_load_speed']).float().mean():5.2f}%"
                  f"  outside T-w line {100*over.float().mean():5.2f}%  worst excess {(t - avail).max():5.2f} Nm")
    if args_cli.mode == "score":
        run_score(env, uenv, obs)
    if args_cli.mode == "standpush":
        policy = load_policy(args_cli.checkpoint, uenv.device)
        n = uenv.num_envs
        ids = torch.arange(n, device=uenv.device)
        speed = torch.tensor(STANDPUSH_SPEEDS, device=uenv.device)[(ids // 8) % len(STANDPUSH_SPEEDS)]
        heading = (ids % 8).float() * (math.pi / 4.0)
        settle, total = int(STANDPUSH_SETTLE_S / dt), int((STANDPUSH_SETTLE_S + STANDPUSH_OBSERVE_S) / dt)
        meter = StepMeter(uenv)
        fell_before = torch.zeros(n, dtype=torch.bool, device=uenv.device)
        lifts = torch.zeros(n, device=uenv.device)
        settle_t = torch.full((n,), -1.0, device=uenv.device)  # time after the push when both soles are down for good
        start_xy = None
        max_disp = torch.zeros(n, device=uenv.device)
        prev_up = None
        zero = torch.zeros((n, 3), device=uenv.device)
        with torch.inference_mode():
            for k in range(total):
                uenv.commands[:] = zero
                pobs = obs["policy"].clone()
                pobs[:, -3:] = 0.0
                obs, _, _, _, _ = env.step(torch.clamp(policy(pobs), -1.0, 1.0))
                min_h = torch.minimum(min_h, trunk_height(uenv))
                if k == settle - 1:
                    fell_before = min_h < 0.2
                    yaw_q = uenv.robot.data.body_quat_w.torch[:, uenv.ref_body_index]
                    x, y, z, w = yaw_q.unbind(-1)
                    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                    vel = uenv.robot.data.root_vel_w.torch.clone()
                    vel[:, 0] += speed * torch.cos(yaw + heading)
                    vel[:, 1] += speed * torch.sin(yaw + heading)
                    uenv.robot.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
                    start_xy = uenv.robot.data.body_pos_w.torch[:, uenv.ref_body_index, :2].clone()
                    if os.environ.get("EVAL_PUSH_DEBUG"):
                        print(f"[PUSH-DBG] written {vel[31, :3].tolist()} read back {uenv.robot.data.root_vel_w.torch[31, :3].tolist()}")
                if os.environ.get("EVAL_PUSH_DEBUG") and settle <= k < settle + 5:
                    print(f"[PUSH-DBG] k {k - settle} root vel {[round(v, 3) for v in uenv.robot.data.root_vel_w.torch[31, :3].tolist()]}")
                if k >= settle:
                    up = meter.feet()[..., 2] > 0.008
                    if prev_up is not None:
                        lifts += (up & ~prev_up).float().sum(-1)
                    prev_up = up
                    down = ~up.any(-1)
                    t = (k - settle) * dt
                    settle_t = torch.where(down & (settle_t < 0), torch.full_like(settle_t, t), settle_t)
                    settle_t = torch.where(~down, torch.full_like(settle_t, -1.0), settle_t)
                    disp = torch.linalg.norm(uenv.robot.data.body_pos_w.torch[:, uenv.ref_body_index, :2] - start_xy, dim=-1)
                    max_disp = torch.maximum(max_disp, disp)
        fell = (min_h < 0.2) & ~fell_before
        print(f"[EVAL-PUSH] checkpoint={args_cli.checkpoint} envs={n} fell before the push={int(fell_before.sum())}")
        names = ["fwd", "fwd-left", "left", "back-left", "back", "back-right", "right", "fwd-right"]
        for i, v in enumerate(STANDPUSH_SPEEDS):
            m = (speed == v) & ~fell_before
            if not bool(m.any()):
                continue
            ok = m & ~fell
            dirs = ",".join(names[int(e) % 8] for e in torch.nonzero(m & fell).flatten().tolist())
            print(f"[EVAL-PUSH] push {v:.2f} m/s: fell {int((m & fell).sum())}/{int(m.sum())} ({dirs or '-'}) | survivors: "
                  f"foot lifts {lifts[ok].mean().item() if bool(ok.any()) else float('nan'):.1f}, "
                  f"settled after {torch.where(settle_t < 0, torch.full_like(settle_t, STANDPUSH_OBSERVE_S), settle_t)[ok].mean().item() if bool(ok.any()) else float('nan'):.2f} s, "
                  f"max drift {max_disp[ok].mean().item() * 1000 if bool(ok.any()) else float('nan'):.0f} mm")
        print(f"[EVAL-PUSH] total fell {int(fell.sum())}/{int((~fell_before).sum())}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        simulation_app.close()
