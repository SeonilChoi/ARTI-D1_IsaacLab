"""Evaluation of Template-Arti-D1-Vel-Direct-v0 checkpoints (velocity walk after microduck_rl).

--mode score: deterministic policy over an 8 x 5 s command sequence (pushes off). Per env until its first fall:
    falls, velocity tracking per segment, gait (touchdowns, swing time, swing peak height, touchdown speed, stance slip,
    trailing-foot slide during double support, lift delay), joint ROM while walking forward, standing foot lifts / drift.
--mode standpush: zero command, settle 2 s, one planar push per env (0.2 / 0.3 / 0.4 / 0.5 m/s, 8 envs each), observe 3 s.

    python scripts/vel/eval_vel.py --mode score --checkpoint <agent.pt> --num_envs 64
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("score", "standpush", "lag", "replay"), required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--physics", type=str, default="isaacsim_physx")
parser.add_argument("--train_step", type=int, default=0, help="curriculum step the checkpoint was at (logging only)")
parser.add_argument("--task", type=str, default="Template-Arti-D1-Vel-Direct-v0")
parser.add_argument("--cmd_scale", type=float, default=1.0, help="scale the scored command sequence (duck task: 0.5)")
parser.add_argument("--real_csv", type=str, default="", help="replay: humanoid_controller log whose actions are replayed")
parser.add_argument("--t0", type=float, default=0.0, help="replay: start time in the real log (s)")
parser.add_argument("--t1", type=float, default=1e9, help="replay: end time in the real log (s)")
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

TASK = args_cli.task
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
SEQUENCE = tuple((nm, tuple(c * args_cli.cmd_scale for c in cmd)) for nm, cmd in SEQUENCE)
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
    if path.endswith(".onnx"):
        # deployment check: the exported model (scripts/vel/pt_to_onnx_vel.py) on CPU, its "actions" output drives the env
        import onnxruntime as ort

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

        def run(obs):
            out = [sess.run(["actions"], {"obs": o[None]})[0][0] for o in obs.detach().cpu().numpy()]
            return torch.from_numpy(np.stack(out)).to(device)

        return run
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
    # EVAL_CFG="actuator_delay_ms=40;actuator_delay_jitter_ms=(0.0,10.0)": override env cfg fields (sim2real what-ifs)
    for kv in filter(None, os.environ.get("EVAL_CFG", "").split(";")):
        k, v = kv.split("=", 1)
        *path, leaf = k.strip().split(".")
        obj = cfg
        for a in path:
            obj = getattr(obj, a)
        setattr(obj, leaf, eval(v))
        print(f"[EVAL_CFG] {k.strip()} = {getattr(obj, leaf)}")
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
    per_foot = {f: {"swing": [], "peak": [], "td": [], "vmax": [], "reach": []} for f in (0, 1)}
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
                        pf = per_foot[f]
                        pf["swing"].append((j - k) * dt)
                        pf["peak"].append(H[k:j, e, f].max() * 1000)
                        pf["td"].append(-FV[j - 1, e, f, 2] * 1000)
                        pf["vmax"].append(np.linalg.norm(FV[k:j, e, f, :2], axis=-1).max() * 1000)
                        pf["reach"].append(np.linalg.norm(P[j - 1, e, f, :2] - P[k - 1, e, f, :2]) * 1000)
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
                    ds_slide.append(np.linalg.norm(P[j - 1, e, trail, :2] - P[k - 1, e, trail, :2]) * 1000)
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
    for f, side in ((0, "L"), (1, "R")):
        pf = per_foot[f]
        print(f"[VEL-SCORE] foot {side}: swings {len(pf['swing'])} | swing {q(pf['swing'], 50):.2f} s | peak {q(pf['peak'], 50):.1f} mm"
              f" | touchdown {q(pf['td'], 50):.0f} p90 {q(pf['td'], 90):.0f} mm/s | swing max xy speed {q(pf['vmax'], 50):.0f}"
              f" p90 {q(pf['vmax'], 90):.0f} mm/s | step length {q(pf['reach'], 50):.1f} mm")
    print(f"[VEL-SCORE] drag: double-support trailing-foot slide median {q(ds_slide, 50):.1f} p90 {q(ds_slide, 90):.1f} mm "
          f"(n={len(ds_slide)}) | double support median {q(lift_delay, 50):.2f} s")
    wk = al & (tseg >= SKIP_S)[:, None] & walk[:, None]
    both_frac = (C[..., 0] & C[..., 1])[wk].mean() if wk.any() else float("nan")
    none_frac = (~C[..., 0] & ~C[..., 1])[wk].mean() if wk.any() else float("nan")
    flips = np.abs(np.diff(C.astype(int), axis=0)).sum(-1)[wk[1:]].mean() / dt
    print(f"[VEL-SCORE] contact while walking: both down {both_frac:.2f}, both up {none_frac:.2f} of the time, "
          f"contact changes {flips:.1f}/s per env")
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
            rom = [np.ptp(Q[A[:, e], e, ji]) for e in range(n) if A[:, e].sum() > 10]
            vals.append(math.degrees(np.median(rom)) if rom else float("nan"))
        parts.append(f"{jt} {vals[0]:.1f}/{vals[1]:.1f}")
    print("[VEL-SCORE] ROM p-p deg L/R forward: " + " | ".join(parts))
    # torso angular velocity spread per env (pelvis frame, = the real IMU gyro std the robot logs give), forward / turn
    for s_name in ("forward", "turn_left"):
        s = [i for i, (nm, _) in enumerate(SEQUENCE) if nm == s_name][0]
        m = (seg == s) & (tseg >= SKIP_S)
        W = R["w"][m]
        stds = [W[al[m][:, e], e].std(0) for e in range(n) if al[m][:, e].sum() > 10]
        if stds:
            sd = np.median(np.array(stds), axis=0)
            print(f"[VEL-SCORE] torso gyro std {s_name}: x {sd[0]:.3f} y {sd[1]:.3f} z {sd[2]:.3f} rad/s")


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
    gyro_log, act_log = [], []
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
            a = policy(o)
            obs, _, term, _, _ = env.step(a)
        gyro_log.append(o[:, 0:3].cpu())
        act_log.append(a.cpu())
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
    # standing buzz (real ARTI-D1 v20: ~10 Hz trunk / action limit cycle at zero command, gyro std ~1 rad/s): observed
    # gyro (obs, with noise) and actions over the settle window before the push and the last 1.5 s after it
    G, A = torch.stack(gyro_log, 1).numpy(), torch.stack(act_log, 1).numpy()  # [env, step, k]
    al = alive.cpu().numpy()
    for name, sl in (("settle", slice(int(0.5 / dt), int(SETTLE_S / dt))), ("after push", slice(-int(1.5 / dt), None))):
        g, a = G[al, sl], A[al, sl]
        f = np.fft.rfftfreq(g.shape[1], dt)
        def hf(x):
            P = np.abs(np.fft.rfft(x - x.mean(1, keepdims=True), axis=1)) ** 2
            return float((P[:, (f >= 7) & (f < 13)].sum(1) / np.maximum(P[:, f > 0.3].sum(1), 1e-12)).mean())
        print(f"[VEL-PUSH] buzz {name}: gyro std x {g[..., 0].std(1).mean():.3f} y {g[..., 1].std(1).mean():.3f} "
              f"z {g[..., 2].std(1).mean():.3f} rad/s (max env x {g[..., 0].std(1).max():.3f}) | 7-13 Hz share gyro "
              f"{hf(g[..., 0]):.2f} action {hf(a):.2f} | action step |da| {np.abs(np.diff(a, axis=1)).mean():.3f}")


LAG_S = 8.0
LEG_ORDER = ("left_hip_yaw", "right_hip_yaw", "left_hip_roll", "right_hip_roll", "left_hip_pitch", "right_hip_pitch",
             "left_knee", "right_knee", "left_ankle", "right_ankle")


def band_rms(x, dt, lo, hi):
    """rms of the lo-hi Hz band of x [..., time, k] (mean over k)."""
    X = np.fft.rfft(x - x.mean(-2, keepdims=True), axis=-2)
    f = np.fft.rfftfreq(x.shape[-2], dt)
    X[..., (f < lo) | (f >= hi), :] = 0
    return np.sqrt((np.fft.irfft(X, n=x.shape[-2], axis=-2) ** 2).mean(axis=(-2, -1)))


def run_replay(env, u, real):
    """real2sim: replay a real log's policy actions open loop (every env the same sequence; per-env randomised servo
    model / joint play / masses as in training) from the default standing pose and compare the trunk gyro and joint
    position response with the real one, per frequency band."""
    n, dt = u.num_envs, u.step_dt
    env.reset()
    u.commands[:] = 0.0
    A = torch.tensor(real[["action_" + j for j in LEG_ORDER]].values, dtype=torch.float32, device=u.device)
    zero = torch.zeros(n, A.shape[1], device=u.device)
    for _ in range(int(1.0 / dt)):  # settle on the default pose
        env.step(zero)
    g_log, q_log, alive = [], [], torch.ones(n, dtype=torch.bool, device=u.device)
    for k in range(len(A)):
        obs, _, term, _, _ = env.step(A[k][None].expand(n, -1))
        alive &= ~term
        g_log.append(obs["policy"][:, 0:3].cpu())  # the gyro the policy sees (v22: + structural vibration model)
        q_log.append((u.robot.data.joint_pos.torch - u.default_joint_pos)[:, u.leg_ids].cpu())
    G, Q = torch.stack(g_log, 1).numpy(), torch.stack(q_log, 1).numpy()
    al = alive.cpu().numpy()
    rg = real[["obs_0", "obs_1", "obs_2"]].values[None]
    rq = real[[f"obs_{i}" for i in range(6, 16)]].values[None]
    rqd = real[["q_des_" + j for j in LEG_ORDER]].values[None]
    sqd = (u.default_joint_pos[0, u.leg_ids].cpu().numpy() + u.cfg.action_scale * A.cpu().numpy())[None]
    print(f"[REPLAY] {len(A)} steps, envs alive {int(al.sum())}/{n}")
    for lo, hi in ((0.3, 3.0), (3.0, 7.0), (7.0, 13.0), (13.0, 30.0)):
        sg, rg_ = band_rms(G[al], dt, lo, hi), band_rms(rg, dt, lo, hi)[0]
        sq, rq_ = band_rms(Q[al], dt, lo, hi), band_rms(rq, dt, lo, hi)[0]
        tq_s = sq / np.maximum(band_rms(sqd, dt, lo, hi)[0], 1e-9)
        tq_r = rq_ / max(band_rms(rqd, dt, lo, hi)[0], 1e-9)
        print(f"[REPLAY] {lo:4.1f}-{hi:4.1f} Hz: gyro rms real {rg_:.3f} sim {sg.mean():.3f} (p10 {np.percentile(sg, 10):.3f} "
              f"p90 {np.percentile(sg, 90):.3f}) | joint pos rms real {rq_:.4f} sim {sq.mean():.4f} | pos/q_des real "
              f"{tq_r:.2f} sim {tq_s.mean():.2f}")




def run_lag(env, u, policy):
    """Servo response lag as the real logs show it: steps between the policy joint target (q_des) and the measured
    joint position at peak cross-correlation, walking forward 0.08 m/s (real ARTI-D1 2026-11-10/12: 4-5 steps)."""
    n = u.num_envs
    obs, _ = env.reset()
    qd_log, q_log = [], []
    for i in range(int(LAG_S / u.step_dt)):
        o = obs["policy"].clone()
        o[:, -3:] = torch.tensor([0.08, 0.0, 0.0], device=u.device)
        u.commands[:] = o[:, -3:]
        with torch.no_grad():
            obs, _, _, _, _ = env.step(policy(o))
        if i * u.step_dt > 2.0:
            qd_log.append(u.joint_target[:, u.leg_mask].cpu())
            q_log.append(u.robot.data.joint_pos.torch[:, u.leg_mask].cpu() if hasattr(u.robot.data.joint_pos, "torch")
                         else u.robot.data.joint_pos[:, u.leg_mask].cpu())
    QD, Q = torch.stack(qd_log, 1).numpy(), torch.stack(q_log, 1).numpy()
    lags = np.zeros((n, QD.shape[2]), int)
    gains = np.zeros((n, QD.shape[2]))
    for e in range(n):
        for j in range(QD.shape[2]):
            a, b = QD[e, :, j] - QD[e, :, j].mean(), Q[e, :, j] - Q[e, :, j].mean()
            cc = [np.corrcoef(a[: len(a) - k], b[k:])[0, 1] for k in range(9)]
            lags[e, j] = int(np.nanargmax(cc))
            gains[e, j] = np.polyfit(a[: len(a) - lags[e, j]], b[lags[e, j]:], 1)[0]
    print(f"[VEL-LAG] q_des->pos lag steps per leg joint (median over envs): {np.median(lags, 0).tolist()} "
          f"mean {lags.mean():.2f} | gain {np.median(gains, 0).round(2).tolist()}")


device = "cuda:0"
policy = load_policy(args_cli.checkpoint, device) if args_cli.mode != "replay" else None
if args_cli.mode == "score":
    env, u = make_env(SEG_S * len(SEQUENCE))
    run_score(env, u, policy)
elif args_cli.mode == "replay":
    import pandas as pd

    real = pd.read_csv(args_cli.real_csv)
    real = real[(real.t >= args_cli.t0) & (real.t < args_cli.t1)].reset_index(drop=True)
    env, u = make_env(len(real) * 0.0167 + 3.0)
    run_replay(env, u, real)
elif args_cli.mode == "lag":
    env, u = make_env(LAG_S)
    run_lag(env, u, policy)
else:
    env, u = make_env(SETTLE_S + OBSERVE_S)
    run_standpush(env, u, policy)
env.close()
simulation_app.close()
