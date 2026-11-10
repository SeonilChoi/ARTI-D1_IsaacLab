"""Hung-robot (fixed base) step test of the sim actuator model, same as example_sysid 'step' (+-0.1 rad, 1 s holds)."""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--delay", type=float, default=40.0); p.add_argument("--jitter", type=float, nargs=2, default=(-15.0, 25.0))
p.add_argument("--per_step", type=int, default=1); p.add_argument("--kp_scale", type=float, default=1.0); p.add_argument("--kd_scale", type=float, default=1.0)
p.add_argument("--armature_scale", type=float, default=1.0); p.add_argument("--out", default="")
AppLauncher.add_app_launcher_args(p); a = p.parse_args(); app = AppLauncher(a).app
import gymnasium as gym, numpy as np, torch
import isaaclab_tasks  # noqa
import arti_d1.tasks  # noqa
from isaaclab_tasks.utils import resolve_task_config
T = "Template-Arti-D1-Vel-Direct-v0"
cfg = resolve_task_config(T, None, overrides=["physics=isaacsim_physx"])[0]
cfg.scene.num_envs = 8; cfg.push_velocity = 0.0; cfg.push_interval_s = (1e9, 1e9); cfg.command_resample_s = (1e9, 1e9)
cfg.episode_length_s = 60.0; cfg.reset_xy_range = 0.0; cfg.reset_height = 0.35; cfg.min_trunk_height = -1.0; cfg.fell_over_angle = 3.2
cfg.scene.robot.spawn.fix_root_link = True
cfg.actuator_delay_ms = a.delay; cfg.actuator_delay_jitter_ms = tuple(a.jitter); cfg.actuator_delay_per_step = bool(a.per_step)
cfg.actuator_armature = {k: v * a.armature_scale for k, v in cfg.actuator_armature.items()}
for act in cfg.scene.robot.actuators.values():
    act.stiffness = {k: v * a.kp_scale for k, v in act.stiffness.items()}
    act.damping = {k: v * a.kd_scale for k, v in act.damping.items()}
env = gym.make(T, cfg=cfg); u = env.unwrapped
obs, _ = env.reset()
names = list(u.robot.data.joint_names); legs = [names[i] for i in u.leg_ids.tolist()]
dt = u.step_dt; res = {}
for j in ("left_knee", "left_hip_pitch", "left_ankle", "left_hip_roll", "left_hip_yaw"):
    k = legs.index(j); jid = names.index(j)
    act = torch.zeros(u.num_envs, len(legs), device=u.device)
    for _ in range(int(1.0 / dt)): env.step(act)
    q0 = u.robot.data.joint_pos.torch[:, jid].clone()
    t63s, deads = [], []
    for s in (0.1, -0.1, 0.1, -0.1):
        act[:, k] = s / cfg.action_scale
        start = u.robot.data.joint_pos.torch[:, jid].clone(); target = q0 + s
        trace = []
        for i in range(int(1.0 / dt)):
            env.step(act); trace.append(((u.robot.data.joint_pos.torch[:, jid] - start) / (target - start)).cpu().numpy())
        tr = np.array(trace); tt = (np.arange(len(tr)) + 1) * dt
        for e in range(tr.shape[1]):
            c = tr[:, e]
            if (c > 0.632).any(): t63s.append(np.interp(0.632, c[: np.argmax(c > 0.632) + 1], tt[: np.argmax(c > 0.632) + 1]))
            if (c > 0.05).any(): deads.append(np.interp(0.05, c[: np.argmax(c > 0.05) + 1], tt[: np.argmax(c > 0.05) + 1]))
    act[:, k] = 0.0
    res[j] = (np.median(deads) * 1000, np.median(t63s) * 1000)
    print(f"[SIMSTEP] {j:16s} dead(5%) {res[j][0]:5.0f} ms  t63 {res[j][1]:5.0f} ms", flush=True)
env.close(); app.close()
