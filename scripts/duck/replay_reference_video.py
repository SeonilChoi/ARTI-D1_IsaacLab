"""Replay placo reference clips (assets/motions/<motion_set>/<i>.json) on the ARTI-D1 in Isaac Sim and record a video.

Kinematic replay like scripts/replay_demo.py (root pose / velocity and every joint written from the clip each control
step, the sim then steps), recorded with the same Isaac Lab 3.x video path as scripts/skrl/play.py
(apply_video_recording). Clips play one after another, each restarted at the env origin.

    python scripts/duck/replay_reference_video.py --motion_set arti_d1_duck --clips 187 12 89 97 --seconds 5 \
        --video newton_gl --out logs/tuning/videos/duck_reference_isaac.mp4
"""

import argparse
import contextlib
import glob
import json
import os
import shutil
import sys

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.entrypoints.common import (
    add_common_play_args,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    enable_cameras_for_video,
    pre_launch_video_config,
    set_hydra_args,
)
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

TASK = "Template-Arti-D1-Duck-Direct-v0"
CAMERAS = {"front": ((1.6, 0.0, 0.45), (0.0, 0.0, 0.12)), "side": ((0.3, -2.0, 0.6), (0.3, 0.0, 0.06)),
           "close": ((0.75, -0.45, 0.35), (0.0, 0.0, 0.15))}


def parse():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--motion_set", default="arti_d1_duck")
    p.add_argument("--clips", type=int, nargs="+", required=True)
    p.add_argument("--seconds", type=float, default=5.0, help="seconds of each clip (from its periodic start)")
    p.add_argument("--out", required=True)
    p.add_argument("--video_view", choices=tuple(CAMERAS), default="front")
    p.add_argument("--video_width", type=int, default=1280)
    p.add_argument("--video_height", type=int, default=720)
    add_common_play_args(p, agent_default=None, agent_help="unused")  # --video <source>, --video_length (as play.py)
    add_launcher_args(p)
    args, hydra = setup_preset_cli(p, sys.argv[1:])
    args.task = TASK
    args.num_envs = 1
    enable_cameras_for_video(args)
    set_hydra_args(hydra)
    return args


def copy_video(log_dir, out):
    mp4 = sorted(glob.glob(os.path.join(log_dir, "videos", "replay", "*.mp4")), key=os.path.getsize)
    if not mp4:
        print(f"[REPLAY] no video written under {log_dir}/videos/replay", flush=True)
        return
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    shutil.copy(mp4[-1], out)
    print(f"[REPLAY] saved {out}", flush=True)


def main():
    args = parse()
    import arti_d1.tasks  # noqa: F401
    import numpy as np
    import torch
    from arti_d1.robots import ISAACLAB_ASSETS_DIR

    env_cfg, _ = resolve_task_config(TASK, None, play_mode=False)
    env_cfg.scene.num_envs = 1
    env_cfg.push_velocity = 0.0
    env_cfg.push_interval_s = (1e9, 1e9)
    env_cfg.episode_length_s = 1e6
    env_cfg.reset_xy_range = 0.0
    env_cfg.fell_over_angle = 3.2
    env_cfg.min_trunk_height = -1.0
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.asset_name = None
    env_cfg.viewer.eye, env_cfg.viewer.lookat = CAMERAS[args.video_view]
    env_cfg.viewer.resolution = (args.video_width, args.video_height)

    clips = []
    for i in args.clips:
        d = json.load(open(os.path.join(ISAACLAB_ASSETS_DIR, "motions", args.motion_set, f"{i}.json")))
        off, size = d["frame_offset"][0], d["frame_size"][0]
        fr = np.array(d["frames"])
        start = int(np.argmax(fr[:, off["periodic_valid"]] > 0.5))
        n = int(round(args.seconds * d["fps"]))
        fr = fr[start:start + n]
        clips.append((i, d, fr, off, size))
    total = sum(len(c[2]) for c in clips)
    args.video_length = total + 5

    log_dir = os.path.abspath(os.path.join("logs", "tuning", "replay_reference"))
    os.makedirs(log_dir, exist_ok=True)
    pre_launch_video_config(env_cfg, args)
    with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
        env_cfg.log_dir = log_dir
        shutil.rmtree(os.path.join(log_dir, "videos"), ignore_errors=True)
        apply_video_recording(env_cfg, log_dir, args, subdir="replay")
        cleanup.callback(lambda: copy_video(log_dir, args.out))  # runs after close_env (LIFO): the mp4 is complete then
        env = create_isaaclab_env(TASK, env_cfg, args, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        u = env.unwrapped
        names = list(u.robot.data.joint_names)
        dev = u.device
        env.reset()
        ids = torch.arange(1, device=dev)
        zero = torch.zeros(1, u.cfg.action_space, device=dev)
        for i, d, fr, off, size in clips:
            g = d["gait_parameters"]
            print(f"[REPLAY] clip {i}: dx {g['dx']:+.2f} dy {g['dy']:+.2f} dth {g['dth']:+.2f}, {len(fr)} frames", flush=True)
            col = [d["joints"].index(nm) for nm in names]
            p0 = fr[0, off["root_position"]:off["root_position"] + 3].copy()
            for row in fr:
                t = lambda k: torch.tensor(row[off[k]:off[k] + size[k]], device=dev, dtype=torch.float32)[None]
                pos = t("root_position")
                pos[0, :2] -= torch.tensor(p0[:2], device=dev, dtype=torch.float32)
                pose = torch.cat((pos + u.scene.env_origins[ids], t("root_orientation_quat")), dim=-1)  # quat x y z w
                vel = torch.cat((t("world_linear_velocity"), t("world_angular_velocity")), dim=-1)
                q = t("joints_positions")[:, col]
                qd = t("joints_velocities")[:, col]
                u.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
                u.robot.write_root_com_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
                u.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
                u.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=ids)
                env.step(zero)


if __name__ == "__main__":
    main()
