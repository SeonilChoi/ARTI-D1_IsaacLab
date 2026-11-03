"""Physics check before training the velocity task (microduck AGENTS.md step 2): with zero actions (= default pose
targets, arms at 0) every env must stay upright from the reset; prints tilt, trunk height, sole contact and corner heights.

    python scripts/vel/stand_check.py --num_envs 64 --seconds 3
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--physics", type=str, default="isaacsim_physx")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import math

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import resolve_task_config

import arti_d1.tasks  # noqa: F401

TASK = "Template-Arti-D1-Vel-Direct-v0"
cfg = resolve_task_config(TASK, None, overrides=[f"physics={args_cli.physics}"])[0]
cfg.scene.num_envs = args_cli.num_envs
cfg.push_velocity = 0.0
cfg.push_interval_s = (1e9, 1e9)
cfg.episode_length_s = args_cli.seconds + 5.0
env = gym.make(TASK, cfg=cfg)
u = env.unwrapped
env.reset()
act = torch.zeros(u.num_envs, u.cfg.action_space, device=u.device)
steps = int(args_cli.seconds / u.step_dt)
max_tilt = torch.zeros(u.num_envs, device=u.device)
fell = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
for i in range(steps):
    _, _, term, _, _ = env.step(act)
    fell |= term
    quat = u.robot.data.body_quat_w.torch[:, u.pelvis_id]
    up = quat_apply(quat, u.up_b)[:, 2]
    max_tilt = torch.maximum(max_tilt, torch.rad2deg(torch.acos(up.clamp(-1, 1))))
    if i in (0, int(0.5 / u.step_dt), steps - 1):
        _, vel, h = u._feet()
        trunk_z = u.robot.data.body_pos_w.torch[:, u.pelvis_id, 2] + quat_apply(quat, u.trunk_offset)[:, 2]
        print(f"[STAND] t={i * u.step_dt:4.2f}s sole corner min height L/R mean {h[:, 0].mean() * 1000:.1f}/"
              f"{h[:, 1].mean() * 1000:.1f} mm, contact {(h < u.cfg.contact_height).float().mean().item():.2f}, "
              f"trunk z {trunk_z.mean():.4f} m, foot speed {torch.linalg.norm(vel, dim=-1).mean() * 1000:.1f} mm/s", flush=True)
print(f"[STAND] {args_cli.seconds:.1f} s zero action: fell {int(fell.sum())}/{u.num_envs}, max tilt median "
      f"{max_tilt.median():.2f} max {max_tilt.max():.2f} deg", flush=True)
env.close()
simulation_app.close()
