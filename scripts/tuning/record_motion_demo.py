"""Record a video of reference motion clips replayed kinematically (root + joints written every step, no policy).

    python scripts/tuning/record_motion_demo.py --clips 681,408,52,375,795 --seconds 8 \
        --out logs/tuning/videos/motion_demo.mp4 physics=newton_mjwarp --video newton_gl

Each clip plays for --seconds from its first frame, one after another, on env 0 (the play.py camera follows env 0).
Keep clips x seconds x 60 <= 2000 frames. The PD targets are the reference joint positions too, so the one physics step between two writes barely moves it.
Isaac Lab 3.x replacement for replay_demo.py --video (removed in the port).
"""

import argparse
import contextlib
import glob
import os
import shutil
import sys
import tempfile

import torch

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

import arti_d1.tasks  # noqa: F401
from arti_d1.robots import ISAACLAB_ASSETS_DIR

TASK = "Template-Arti-D1-Amp-Direct-v0"
parser = argparse.ArgumentParser(description="Replay reference clips kinematically and record a video.")
add_common_play_args(parser, agent_default=None, agent_help="unused")
parser.add_argument("--motion_set", type=str, default="arti_d1", help="Folder under assets/motions.")
parser.add_argument("--motion_dir", type=str, default=None, help="Clip folder (overrides --motion_set).")
parser.add_argument("--clips", type=str, required=True, help="Comma-separated clip indices, played in order.")
parser.add_argument("--seconds", type=float, default=8.0, help="Seconds per clip.")
parser.add_argument("--out", type=str, required=True, help="Output mp4 path.")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
enable_cameras_for_video(args_cli)
set_hydra_args(hydra_args)

clips = [int(c) for c in args_cli.clips.split(",")]
motion_dir = tempfile.mkdtemp(prefix="arti_d1_motion_demo_")
for k, index in enumerate(clips):  # the loader sorts by number: k is the motion id
    src_dir = args_cli.motion_dir or os.path.join(ISAACLAB_ASSETS_DIR, "motions", args_cli.motion_set)
    shutil.copy2(os.path.join(src_dir, f"{index}.json"),
                 os.path.join(motion_dir, f"{k}.json"))

env_cfg, _ = resolve_task_config(TASK, None, overrides=hydra_args)
env_cfg.scene.num_envs = 1
env_cfg.motion_folder_path = motion_dir
env_cfg.reset_strategy = "default"
env_cfg.early_termination = False
env_cfg.episode_length_s = args_cli.seconds * len(clips) + 10.0
env_cfg.nominal_gait_scale = 0.0
env_cfg.balance_support_kp = 0.0
env_cfg.uniform_action_scale = 0.0
env_cfg.events = None
env_cfg.viewer.origin_type = "env"
env_cfg.viewer.env_index = 0
env_cfg.viewer.asset_name = None
env_cfg.viewer.eye = (0.15, -0.9, 0.3)  # closer than play.py: one robot, legs readable
env_cfg.viewer.lookat = (0.1, 0.0, 0.13)
env_cfg.viewer.resolution = (1280, 720)
steps_per_clip = int(round(args_cli.seconds / (env_cfg.sim.dt * env_cfg.decimation)))
args_cli.video_length = steps_per_clip * len(clips)  # the recorder still caps a clip at 2000 frames
pre_launch_video_config(env_cfg, args_cli)

with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
    log_dir = tempfile.mkdtemp(prefix="arti_d1_motion_demo_video_")
    apply_video_recording(env_cfg, log_dir, args_cli, subdir="demo")
    env = create_isaaclab_env(TASK, env_cfg, args_cli, convert_marl_to_single_agent=False)
    cleanup.callback(lambda: close_env(env))
    u = env.unwrapped
    ids = torch.arange(u.num_envs, device=u.device)
    env.reset()
    dt = u.step_dt
    with torch.inference_mode():
        for k, index in enumerate(clips):
            print(f"[DEMO] clip {index} ({k + 1}/{len(clips)})")
            for step in range(steps_per_clip):
                motion_ids = torch.full((u.num_envs,), k, dtype=torch.long).numpy()
                times = torch.full((u.num_envs,), step * dt, dtype=torch.float64).numpy()
                pos, quat, lin, ang, q, qd, _, _ = u.motion_loader.sample(
                    u.num_envs, motion_ids=motion_ids, times=times, return_full_state=True
                )
                root = u._default_root_state(ids)
                root[:, :3] = pos + u.scene.env_origins[ids]
                root[:, 3:7] = quat  # (x, y, z, w)
                root[:, 7:10] = lin
                root[:, 10:13] = ang
                u.robot.write_root_link_pose_to_sim_index(root_pose=root[:, :7], env_ids=ids)
                u.robot.write_root_com_velocity_to_sim_index(root_velocity=root[:, 7:], env_ids=ids)
                u.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
                u.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=ids)
                delta = q - u.robot.data.default_joint_pos.torch
                scale = torch.where(delta >= 0.0, u.action_positive_scale, u.action_negative_scale)
                env.step(torch.clamp(delta / scale, -1.0, 1.0))
mp4 = sorted(glob.glob(os.path.join(log_dir, "videos", "demo", "*.mp4")), key=os.path.getsize)
if mp4:
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    shutil.copy2(mp4[-1], args_cli.out)
    print(f"[DEMO] saved {args_cli.out}")
else:
    print(f"[DEMO] no video in {log_dir}")
shutil.rmtree(motion_dir, ignore_errors=True)
