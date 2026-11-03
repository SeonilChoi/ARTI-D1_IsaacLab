"""Evaluation of Template-Arti-D1-Vel-Direct-v0 checkpoints (velocity walk after microduck_rl).

--mode score: deterministic policy over an 8 x 5 s command sequence (pushes off). Per env until its first fall:
    falls, velocity tracking per segment, gait (touchdowns, swing time, swing peak height, touchdown speed, stance slip,
    trailing-foot slide during double support, lift delay), joint ROM while walking forward, standing foot lifts / drift.
--mode standpush: zero command, settle 2 s, one planar push per env (0.2 / 0.3 / 0.4 / 0.5 m/s, 8 envs each), observe 3 s.

    python scripts/vel/eval_vel.py --mode score --checkpoint <agent.pt> --num_envs 64
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("score", "standpush"), required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--physics", type=str, default="isaacsim_physx")
parser.add_argument("--train_step", type=int, default=0, help="curriculum step the checkpoint was at (logging only)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab_tasks.utils import resolve_task_config

import arti_d1.tasks  # noqa: F401

TASK = "Template-Arti-D1-Vel-Direct-v0"
SEG_S = 5.0
SEQUENCE = (
    ("forward", (0.12, 0.0, 0.0)),
    ("stand", (0.0, 0.0, 0.0)),
    ("side_left", (0.0, 0.06, 0.0)),
    ("backward", (-0.10, 0.0, 0.0)),
    ("stand", (0.0, 0.0, 0.0)),
    ("diagonal_right", (0.10, -0.05, 0.0)),
    ("turn_left", (0.0, 0.0, 0.3)),
    ("stand", (0.0, 0.0, 0.0)),
)
SKIP_S = 1.0  # transition after each command change, not scored
PUSHES = (0.2, 0.3, 0.4, 0.5)
SETTLE_S, OBSERVE_S = 2.0, 3.0


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
            layers.append(nn.ELU())
    if has_out:
        out = nn.Linear(sd["output_layer.weight"].shape[1], sd["output_layer.weight"].shape[0])
        out.weight.data.copy_(sd["output_layer.weight"])
        out.bias.data.copy_(sd["output_layer.bias"])
        layers.append(out)
    actor = nn.Sequential(*layers).to(device).eval()
    pre = ck.get("observation_preprocessor")
    scaler = Scaler({k: v.to(device) for k, v in pre.items()}) if pre else nn.Identity()
    return lambda obs: actor(scaler(obs))


def make_env(seconds):
    cfg = resolve_task_config(TASK, None, overrides=[f"physics={args_cli.physics}"])[0]
    cfg.scene.num_envs = args_cli.num_envs
    cfg.push_velocity = 0.0
    cfg.push_interval_s = (1e9, 1e9)
    cfg.command_resample_s = (1e9, 1e9)
    cfg.episode_length_s = seconds + 2.0
    cfg.reset_xy_range = 0.0
    env = gym.make(TASK, cfg=cfg)
    return env, env.unwrapped


def q(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def run_score(env, u, policy):
    n, dt = u.num_envs, u.step_dt
    steps = int(SEG_S * len(SEQUENCE) / dt)
    obs, _ = env.reset()
    alive = torch.ones(n, dtype=torch.bool, device=u.device)
    fall_t = []
    rec = {k: [] for k in ("seg", "alive", "v", "w", "sole", "fv", "h", "contact", "q", "yaw")}
    for i in range(steps):
        t = i * dt
        s = min(int(t / SEG_S), len(SEQUENCE) - 1)
        u.commands[:] = torch.tensor(SEQUENCE[s][1], device=u.device)
        o = obs["policy"].clone()
        o[:, -3:] = u.commands
        with torch.no_grad():
            act = policy(o)
        obs, _, term, _, _ = env.step(act)
        for e in torch.nonzero(term & alive).flatten().tolist():
            fall_t.append(round(t, 2))
        alive &= ~term
        quat, lin_b, ang_b, _ = u._base()
        sole, fv, h = u._feet()
        fwd = quat_apply(quat, torch.tensor([1.0, 0.0, 0.0], device=u.device).expand(n, 3))
        rec["seg"].append(s)
        rec["alive"].append(alive.cpu().numpy().copy())
        rec["v"].append(lin_b.cpu().numpy())
        rec["w"].append(ang_b.cpu().numpy())
        rec["sole"].append(sole.cpu().numpy())
        rec["fv"].append(fv.cpu().numpy())
        rec["h"].append(h.cpu().numpy())
        rec["contact"].append((h < u.cfg.contact_height).cpu().numpy())
        rec["q"].append(u.robot.data.joint_pos.torch.cpu().numpy())
        rec["yaw"].append(torch.atan2(fwd[:, 1], fwd[:, 0]).cpu().numpy())
    R = {k: np.asarray(v) for k, v in rec.items()}
    seg, al = R["seg"], R["alive"]
    tseg = (np.arange(len(seg)) * dt) % SEG_S
    print(f"[VEL-SCORE] checkpoint={args_cli.checkpoint} envs={n} train_step={args_cli.train_step}")
    print(f"[VEL-SCORE] falls {len(fall_t)}/{n} at {sorted(fall_t)}")
    for s, (name, cmd) in enumerate(SEQUENCE):
        m = (seg == s) & (tseg >= SKIP_S)
        rows = np.nonzero(m)[0]
        mask = al[rows]
        v = R["v"][rows][mask]
        w = R["w"][rows][mask]
        if len(v) == 0:
            continue
        print(f"[VEL-SCORE] seg {s + 1} {name:15s} cmd ({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) got vx {v[:, 0].mean():+.3f} "
              f"vy {v[:, 1].mean():+.3f} m/s wz {w[:, 2].mean():+.3f} rad/s")

    # gait events per env (walking segments only, alive, after the transition)
    walk = np.array([np.abs(np.array(SEQUENCE[s][1])).sum() > 0 for s in seg])
    td_speed, swing_t, peak, slip, ds_slide, lift_delay = [], [], [], [], [], []
    n_td, walk_time, stand_lifts, stand_time = 0, 0.0, 0, 0.0
    C, H, FV, P = R["contact"], R["h"], R["fv"], R["sole"]
    for e in range(n):
        ok = al[:, e] & (tseg >= SKIP_S)
        for f in (0, 1):
            k = 1
            while k < len(seg):
                if ok[k] and walk[k] and C[k - 1, e, f] and not C[k, e, f]:  # lift-off
                    j = k
                    while j < len(seg) and not C[j, e, f]:
                        j += 1
                    if j < len(seg) and ok[j]:
                        swing_t.append((j - k) * dt)
                        peak.append(H[k:j, e, f].max() * 1000)
                        td_speed.append(-FV[j - 1, e, f, 2] * 1000)
                        n_td += 1
                    k = j
                k += 1
        both = C[:, e, 0] & C[:, e, 1]
        k = 1
        while k < len(seg):  # double-support intervals: xy slide of the foot that was already down (trailing)
            if ok[k] and walk[k] and both[k] and not both[k - 1]:
                landing = 0 if not C[k - 1, e, 0] else 1
                trail = 1 - landing
                j = k
                while j < len(seg) and both[j]:
                    j += 1
                if j < len(seg) and ok[j - 1]:
                    ds_slide.append(np.linalg.norm(P[j - 1, e, trail, :2] - P[k, e, trail, :2]) * 1000)
                    lift_delay.append((j - k) * dt)
                k = j
            k += 1
        stance = C[:, e] & ok[:, None] & walk[:, None]
        slip.extend((np.linalg.norm(FV[:, e, :, :2], axis=-1)[stance] * 1000).tolist())
        walk_time += (ok & walk).sum() * dt
        st = ok & ~walk
        stand_time += st.sum() * dt
        for f in (0, 1):
            stand_lifts += int(np.sum(st[1:] & C[:-1, e, f] & ~C[1:, e, f]))
    print(f"[VEL-SCORE] walk feet: touchdowns {n_td / max(walk_time, 1e-6):.2f}/s per env | swing time median "
          f"{q(swing_t, 50):.2f} s | swing peak height median {q(peak, 50):.1f} p10 {q(peak, 10):.1f} mm | touchdown "
          f"speed median {q(td_speed, 50):.0f} p90 {q(td_speed, 90):.0f} mm/s | stance slip mean {np.mean(slip):.1f} mm/s")
    print(f"[VEL-SCORE] drag: double-support trailing-foot slide median {q(ds_slide, 50):.1f} p90 {q(ds_slide, 90):.1f} mm "
          f"(n={len(ds_slide)}) | double support median {q(lift_delay, 50):.2f} s")
    print(f"[VEL-SCORE] stand: foot lifts {stand_lifts / max(stand_time, 1e-6):.2f}/s per env")
    names = list(u.robot.data.joint_names)
    m = (seg == 0) & (tseg >= SKIP_S)
    Q = R["q"][m]
    A = al[m]
    parts = []
    for jt in ("hip_roll", "hip_yaw", "hip_pitch", "knee", "ankle"):
        vals = []
        for side in ("left", "right"):
            ji = names.index(f"{side}_{jt}")
            rom = [np.ptp(Q[A[:, e], ji]) for e in range(n) if A[:, e].sum() > 10]
            vals.append(math.degrees(np.median(rom)) if rom else float("nan"))
        parts.append(f"{jt} {vals[0]:.1f}/{vals[1]:.1f}")
    print("[VEL-SCORE] ROM p-p deg L/R forward: " + " | ".join(parts))


def run_standpush(env, u, policy):
    n, dt = u.num_envs, u.step_dt
    obs, _ = env.reset()
    u.commands[:] = 0.0
    alive = torch.ones(n, dtype=torch.bool, device=u.device)
    fell_before = 0
    group = torch.arange(n, device=u.device) % len(PUSHES)
    speed = torch.tensor(PUSHES, device=u.device)[group]
    heading = torch.rand(n, device=u.device) * 2 * math.pi
    lifts = torch.zeros(n, device=u.device)
    last_c = None
    for i in range(int((SETTLE_S + OBSERVE_S) / dt)):
        if i == int(SETTLE_S / dt):
            fell_before = int((~alive).sum())
            vel = u.robot.data.root_vel_w.torch.clone()
            vel[:, 0] += speed * torch.cos(heading)
            vel[:, 1] += speed * torch.sin(heading)
            u.robot.write_root_velocity_to_sim_index(root_velocity=vel)
        o = obs["policy"].clone()
        o[:, -3:] = 0.0
        with torch.no_grad():
            obs, _, term, _, _ = env.step(policy(o))
        alive &= ~term
        _, _, h = u._feet()
        c = h < u.cfg.contact_height
        if last_c is not None and i > SETTLE_S / dt:
            lifts += (last_c & ~c).float().sum(dim=1) * alive.float()
        last_c = c
    print(f"[VEL-PUSH] checkpoint={args_cli.checkpoint} envs={n} fell before the push={fell_before}")
    for g, s in enumerate(PUSHES):
        m = group == g
        print(f"[VEL-PUSH] push {s:.2f} m/s: fell {int((~alive & m).sum())}/{int(m.sum())} | survivors foot lifts "
              f"{lifts[m & alive].mean().item() if (m & alive).any() else float('nan'):.1f}")
    print(f"[VEL-PUSH] total fell {int((~alive).sum())}/{n}")


device = "cuda:0"
policy = load_policy(args_cli.checkpoint, device)
if args_cli.mode == "score":
    env, u = make_env(SEG_S * len(SEQUENCE))
    run_score(env, u, policy)
else:
    env, u = make_env(SETTLE_S + OBSERVE_S)
    run_standpush(env, u, policy)
env.close()
simulation_app.close()
