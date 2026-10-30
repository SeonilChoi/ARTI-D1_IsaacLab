"""Deterministic backend A/B: zero policy action (PD target = default + nominal gait table), training balance support,
reset to the reference standstill frame, no domain randomization. Prints contact / tracking statistics.

    python scripts/porting/compare_backends.py physics=isaacsim_physx
    python scripts/porting/compare_backends.py physics=newton_mjwarp
"""

import argparse
import sys

import warp as wp

wp.config.enable_backward = False

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg, setup_preset_cli  # noqa: E402

import arti_d1.tasks  # noqa: F401, E402

TASK = "Template-Arti-D1-Amp-Direct-v0"
parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=4.0)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--hang", action="store_true", help="fixed base 0.6 m up (feet in the air): joint dynamics without contact")
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser, sys.argv[1:])
cfg = parse_env_cfg(TASK, num_envs=args.num_envs, overrides=hydra_args)
cfg.seed = 0
cfg.events = None
cfg.reset_strategy = "random_start"  # every env starts at t=0 of a clip: double-support standstill
cfg.early_termination = False
cfg.actuator_delay_jitter_ms = (0.0, 0.0)
cfg.actuator_armature_scale_range = (1.0, 1.0)
cfg.actuator_coulomb_scale_range = (1.0, 1.0)
cfg.command_resample_interval_s = (0.0, 0.0)
if args.hang:
    cfg.scene.robot.spawn.fix_root_link = True  # 3.x: spawner-level flag
    cfg.balance_support_kp = 0.0
with launch_simulation(cfg, args):
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    env.reset()
    if args.hang:
        root = u._default_root_state(torch.arange(u.num_envs, device=u.device))
        root[:, :3] = u.scene.env_origins + torch.tensor([0.0, 0.0, 0.6], device=u.device)
        u.robot.write_root_pose_to_sim_index(root_pose=root[:, :7])
    d = u.robot.data
    names = d.joint_names
    for j in ("left_arm_m8", "left_arm_m8crank_plate", "left_hip_roll", "left_knee", "left_ankle"):
        i = names.index(j)
        print(
            f"[AB] {j}: armature {d.joint_armature.torch[0, i].item():.4f} | limits "
            f"{[round(v, 4) for v in d.joint_pos_limits.torch[0, i].tolist()]} | soft "
            f"{[round(v, 4) for v in d.soft_joint_pos_limits.torch[0, i].tolist()]} | kp {d.joint_stiffness.torch[0, i].item():.2f}"
            f" kd {d.joint_damping.torch[0, i].item():.3f} | vel limit {d.joint_vel_limits.torch[0, i].item():.1f}"
            f" effort limit {d.joint_effort_limits.torch[0, i].item():.1f}"
        )
    print(f"[AB] total mass {d.body_mass.torch[0].sum().item():.3f} kg")
    for b in ("trunk", "left_arm_m8_crank"):
        i = d.body_names.index(b)
        print(f"[AB] {b}: mass {d.body_mass.torch[0, i].item():.3e} kg, inertia diag "
              f"{[f'{v:.3e}' for v in d.body_inertia.torch[0, i].reshape(3, 3).diagonal().tolist()]}")
    vel_limit = d.joint_vel_limits.torch.clone()  # Dynamixel no-load speed; MJWarp does not enforce it (Q1)
    zero = torch.zeros(u.num_envs, u.cfg.action_space, device=u.device)
    steps = int(args.seconds / u.step_dt)
    pelvis_z, sole_min, sole_max, track_err, qd_rms, tilt, over_vel, fallen = [], [], [], [], [], [], [], 0
    groups = {g: [i for i, j in enumerate(names) if j.endswith(g)] for g in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")}
    torque_sq = {g: [] for g in groups}
    foot_ids = (u.left_toe_body_index, u.right_toe_body_index)
    stance_yaw_rate_sq, stance_n = 0.0, 0
    for k in range(steps):
        _, _, term, trunc, _ = env.step(zero)
        d = u.robot.data
        feet = torch.stack(
            [
                u._get_offset_body_position(i, o, True)[:, 2]
                for i, o in ((u.left_toe_body_index, u.left_toe_body_offset), (u.right_toe_body_index, u.right_toe_body_offset))
            ],
            dim=-1,
        )
        pelvis_z.append(d.body_pos_w.torch[:, u.ref_body_index, 2].mean().item())
        sole_min.append(feet.min().item())
        sole_max.append(feet.max(dim=0).values.mean().item())
        track_err.append((u.desired_joint_target - d.joint_pos.torch).abs().mean().item())
        over_vel.append((d.joint_vel.torch.abs() > vel_limit).float().mean().item())
        qd_rms.append(torch.sqrt((d.joint_vel.torch**2).mean()).item())
        tilt.append(torch.rad2deg(torch.acos((-d.projected_gravity_b.torch[:, 2]).clamp(-1, 1))).mean().item())
        fallen += int(term.sum().item())
        tau = u.robot.actuators.applied_effort.torch
        for g, idx in groups.items():
            torque_sq[g].append((tau[:, idx] ** 2).mean().item())
        # stance foot spin: |yaw rate| of a foot whose sole is within 3 mm of the ground (torsional friction probe)
        wz = d.body_ang_vel_w.torch[:, list(foot_ids), 2]
        stance = feet < 0.003
        stance_yaw_rate_sq += float((wz[stance] ** 2).sum().item())
        stance_n += int(stance.sum().item())
    n = len(pelvis_z)

    def mean(x, a=0):
        x = x[a:]
        return sum(x) / len(x)

    print(f"[AB] steps {n} | phase gait_rate {cfg.gait_phase_rate} | first step pelvis z {pelvis_z[0]:.4f}")
    print(f"[AB] pelvis z mean {mean(pelvis_z):.4f} m (last 2 s {mean(pelvis_z, n // 2):.4f})")
    print(f"[AB] min sole z over run {min(sole_min) * 1000:.2f} mm (penetration < 0), mean of per-step highest sole {mean(sole_max) * 1000:.2f} mm")
    print(f"[AB] |PD target - q| mean {mean(track_err):.4f} rad, joint vel rms {mean(qd_rms):.3f} rad/s, pelvis tilt {mean(tilt):.2f} deg")
    print(f"[AB] joint samples above the no-load speed {mean(over_vel) * 100:.3f} %, max |qd| / limit "
          f"{(d.joint_vel.torch.abs() / vel_limit).max().item():.2f} (last step)")
    print("[AB] applied torque rms Nm: " + " | ".join(f"{g} {mean(v) ** 0.5:.3f}" for g, v in torque_sq.items()))
    print(f"[AB] stance foot yaw rate rms {(stance_yaw_rate_sq / max(stance_n, 1)) ** 0.5 * 57.2958:.2f} deg/s ({stance_n} samples)")
    print(f"[AB] terminations {fallen}, finite {all(map(lambda v: v == v, pelvis_z + track_err))}")
    env.close()
