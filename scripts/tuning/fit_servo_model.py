"""BAM-style fit of the sim servo model (ArtiD1VelEnv._apply_action + PhysX PD) to real ARTI-D1 sysid logs.

Every env is one parameter set (delay, kp scale, kd scale, armature scale, Coulomb friction); the robot hangs on a fixed
base like the real sysid run. Each real test (example_sysid step / chirp CSV) is replayed: the real command trace drives
the tested joint every physics step (240 Hz), the others hold the default pose. Cost = RMS(sim - real position) of the
tested joint, normalised per test by the RMS of the real motion, averaged over the tests.

    python scripts/tuning/fit_servo_model.py --logs <sysid_response2 dir> [--tests 02 03 04 08 09 10]
"""

import argparse
import glob
import itertools
import os
import re

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--logs", required=True)
p.add_argument("--tests", nargs="+", default=["02", "03", "04", "08", "09", "10"])
p.add_argument("--delay_steps", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])  # physics steps of 1/240 s
p.add_argument("--kp", type=float, nargs="+", default=[0.4, 0.6, 0.8, 1.0, 1.3])
p.add_argument("--kd", type=float, nargs="+", default=[0.5, 1.0, 2.0])
p.add_argument("--armature", type=float, nargs="+", default=[1.0, 3.0, 6.0])
p.add_argument("--coulomb", type=float, nargs="+", default=[0.05, 0.15])
p.add_argument("--out", default="")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import arti_d1.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

T = "Template-Arti-D1-Vel-Direct-v0"
grid = list(itertools.product(a.delay_steps, a.kp, a.kd, a.armature, a.coulomb))
cfg = resolve_task_config(T, None, overrides=["physics=isaacsim_physx"])[0]
cfg.scene.num_envs = len(grid)
cfg.scene.env_spacing = 1.0
cfg.push_velocity = 0.0
cfg.push_interval_s = (1e9, 1e9)
cfg.command_resample_s = (1e9, 1e9)
cfg.episode_length_s = 1e6
cfg.reset_xy_range = 0.0
cfg.reset_height = 0.35
cfg.min_trunk_height = -1.0
cfg.fell_over_angle = 3.2
cfg.scene.robot.spawn.fix_root_link = True
cfg.actuator_delay_per_step = False
cfg.actuator_delay_ms = 30.0
cfg.actuator_delay_jitter_ms = (-30.0, 30.0)  # buffer for up to 60 ms; the delay is set per env below
cfg.actuator_armature_scale_range = (1.0, 1.0)
cfg.actuator_coulomb_scale_range = (1.0, 1.0)
env = gym.make(T, cfg=cfg)
u = env.unwrapped
dev, n = u.device, u.num_envs
names = list(u.robot.data.joint_names)
nj = len(names)
G = torch.tensor(grid, device=dev, dtype=torch.float32)
base_kp = u.robot.data.joint_stiffness.torch[0].clone()
base_kd = u.robot.data.joint_damping.torch[0].clone()
base_arm = torch.tensor([cfg.actuator_armature.get(k, 0.0) for k in
                         [next(t for t in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle", "arm_m8crank_plate", "arm_m8")
                               if j.endswith(t)) for j in names]], device=dev)
base_coul = u.act_coulomb[0].clone()
print(f"[FIT] {n} parameter sets, base kp {base_kp.tolist()}", flush=True)


def apply_params():
    u.robot.write_joint_stiffness_to_sim_index(stiffness=base_kp[None] * G[:, 1:2])
    u.robot.write_joint_damping_to_sim_index(damping=base_kd[None] * G[:, 2:3])
    u.robot.write_joint_armature_to_sim_index(armature=base_arm[None] * G[:, 3:4])
    u.act_coulomb[:] = (base_coul > 0).float()[None] * G[:, 4:5]
    u.act_delay_steps = G[:, 0:1].long().expand(n, nj).contiguous()


def run_test(path):
    head = open(path).readline()
    j = re.search(r"joint=(\w+)", head).group(1)
    d = pd.read_csv(path, comment="#")
    t_real, cmd, pos = d.t.values, d[f"{j}_cmd"].values, d[f"{j}_pos"].values
    jid = names.index(j)
    env.reset()
    apply_params()
    u.act_buffer[:] = u.default_joint_pos[None]
    dt = u.physics_dt
    # settle 1 s at the default pose
    for _ in range(int(1.0 / dt)):
        u.joint_target = u.default_joint_pos.clone()
        u._apply_action(); u.scene.write_data_to_sim(); u.sim.step(render=False); u.scene.update(dt=dt)
    steps = int(t_real[-1] / dt)
    sim_t = np.arange(steps) * dt
    trace = np.zeros((steps, n), dtype=np.float32)
    offset = float(u.default_joint_pos[0, jid]) - cmd[0]  # real hold pose vs sim default (same legs, ~0)
    for s in range(steps):
        k = min(np.searchsorted(t_real, sim_t[s], side="right") - 1, len(cmd) - 1)
        tgt = u.default_joint_pos.clone()
        tgt[:, jid] = float(cmd[max(k, 0)]) + offset
        u.joint_target = tgt
        u._apply_action(); u.scene.write_data_to_sim(); u.sim.step(render=False); u.scene.update(dt=dt)
        trace[s] = (u.robot.data.joint_pos.torch[:, jid] - offset).cpu().numpy()
    sim_at_real = np.stack([np.interp(t_real, sim_t, trace[:, e]) for e in range(n)], axis=1)
    err = sim_at_real - pos[:, None]
    scale = np.sqrt(np.mean((pos - pos.mean()) ** 2)) + 1e-6
    return j, np.sqrt(np.mean(err ** 2, axis=0)) / scale, sim_at_real, t_real, cmd, pos


costs, names_run, keep = [], [], {}
for tid in a.tests:
    f = sorted(glob.glob(os.path.join(a.logs, f"{tid}_*.csv")))[0]
    j, c, sim, t_real, cmd, pos = run_test(f)
    costs.append(c)
    names_run.append(os.path.basename(f))
    keep[os.path.basename(f)] = (sim, t_real, cmd, pos)
    b = int(np.argmin(c))
    print(f"[FIT] {os.path.basename(f):28s} best here {grid[b]} cost {c[b]:.3f} | base-like {c[0]:.3f}", flush=True)
C = np.mean(costs, axis=0)
order = np.argsort(C)
print("[FIT] delay_steps(1/240 s), kp_scale, kd_scale, armature_scale, coulomb(Nm) -> mean normalised RMS")
for r in order[:10]:
    print(f"[FIT]   {grid[r]}  {C[r]:.3f}  per test {[round(float(c[r]), 3) for c in costs]}")
ref = [i for i, g in enumerate(grid) if g[1] == 1.0 and g[2] == 1.0 and g[3] == 1.0]
for r in ref:
    if abs(grid[r][4] - 0.15) < 1e-6 or len(a.coulomb) == 1:
        print(f"[FIT] current-model-like {grid[r]}  {C[r]:.3f}")
if a.out:
    best = order[0]
    np.savez(a.out, grid=np.array(grid), cost=C, best=best,
             **{k + "_sim": v[0][:, best] for k, v in keep.items()},
             **{k + "_t": v[1] for k, v in keep.items()}, **{k + "_cmd": v[2] for k, v in keep.items()},
             **{k + "_pos": v[3] for k, v in keep.items()})
env.close()
app.close()
