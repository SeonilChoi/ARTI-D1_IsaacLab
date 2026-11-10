"""ARTI-D1 velocity walk the Open Duck Mini v2 way (see arti_d1_duck_env_cfg.py). Built on ArtiD1VelEnv (scene, servo
model with delay / Coulomb / joint play, pushes, feet helpers, terminations); replaces commands, observations and
rewards with Open Duck's, plus a placo reference gait looked up by command and phase."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.utils.math import quat_apply

from .arti_d1_duck_env_cfg import ArtiD1DuckEnvCfg
from .arti_d1_vel_env import ArtiD1VelEnv, _joint_type


class ArtiD1DuckEnv(ArtiD1VelEnv):
    cfg: ArtiD1DuckEnvCfg

    def __init__(self, cfg: ArtiD1DuckEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev, n = self.device, self.num_envs
        names = list(self.robot.data.joint_names)
        nl = len(self.leg_ids)

        ref = np.load(cfg.reference_file)
        legs = [str(x) for x in ref["legs"]]
        if legs != [names[i] for i in self.leg_ids.tolist()]:
            raise RuntimeError(f"reference leg order {legs} != env {[names[i] for i in self.leg_ids.tolist()]}")
        self.ref_keys = torch.tensor(ref["keys"], device=dev, dtype=torch.float32)  # (N, 3) vx vy wz
        self.ref_table = torch.tensor(ref["table"], device=dev, dtype=torch.float32)  # (N, B, 28)
        self.ref_period = float(ref["period"])
        self.ref_bins = int(ref["bins"])
        span = torch.tensor([cfg.command_lin_x_range[1] - cfg.command_lin_x_range[0],
                             cfg.command_lin_y_range[1] - cfg.command_lin_y_range[0],
                             cfg.command_yaw_range[1] - cfg.command_yaw_range[0]], device=dev)
        self.ref_scale = 1.0 / span
        self.phase = torch.zeros(n, device=dev)
        self.ref = torch.zeros(n, self.ref_table.shape[-1], device=dev)
        self.motor_targets = self.default_joint_pos[:, self.leg_ids].clone()
        # per-env kp factor (Open Duck kp x0.9-1.1)
        lo, hi = cfg.actuator_kp_random
        kp = self.robot.data.joint_stiffness.torch.clone()
        kp[:, self.leg_ids] *= torch.empty(n, 1, device=dev).uniform_(lo, hi)
        self.robot.write_joint_stiffness_to_sim_index(stiffness=kp)
        self.noise_q = torch.tensor([cfg.noise_joint_pos_by_type[_joint_type(names[i])] for i in self.leg_ids.tolist()],
                                    device=dev)

        # left/right mirror for the symmetry loss: obs = gyro, gravity, q, qd, 3 actions, motor targets, phase, cmd
        sym = self.symmetry
        perm, sign = sym["act_perm"].tolist(), sym["act_sign"].tolist()
        o_perm = list(range(6))
        o_sign = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        blocks = 2 + cfg.action_history_steps + 1  # q, qd, actions..., motor targets
        for b in range(blocks):
            base = 6 + b * nl
            o_perm += [base + i for i in perm]
            o_sign += sign
        p0 = 6 + blocks * nl
        o_perm += [p0, p0 + 1]  # phase cos / sin: the mirror is half a cycle later -> both change sign
        o_sign += [-1.0, -1.0]
        o_perm += [p0 + 2, p0 + 3, p0 + 4]
        o_sign += [1.0, -1.0, -1.0]
        if len(o_perm) != cfg.observation_space:
            raise RuntimeError(f"mirror map {len(o_perm)} != obs {cfg.observation_space}")
        sym["obs_perm"] = torch.tensor(o_perm, device=dev)
        sym["obs_sign"] = torch.tensor(o_sign, device=dev)
        print(f"[ARTI-D1 DUCK] reference {self.ref_keys.shape[0]} clips x {self.ref_bins} bins, period "
              f"{self.ref_period:.3f} s; obs {cfg.observation_space}, critic {cfg.state_space}", flush=True)

    # ------------------------------------------------------------------ commands / reference
    def _resample_commands(self, ids: torch.Tensor):
        k, dev = len(ids), self.device
        cmd = torch.empty(k, 3, device=dev)
        cmd[:, 0].uniform_(*self.cfg.command_lin_x_range)
        cmd[:, 1].uniform_(*self.cfg.command_lin_y_range)
        cmd[:, 2].uniform_(*self.cfg.command_yaw_range)
        cmd[torch.rand(k, device=dev) < self.cfg.zero_command_fraction] = 0.0
        self.commands[ids] = cmd
        self.cmd_timer[ids] = torch.empty(k, device=dev).uniform_(*self.cfg.command_resample_s)

    def _reference(self) -> torch.Tensor:
        """Reference frame (n, 28) at the current command and phase: inverse-distance blend of the nearest clips."""
        d = torch.cdist(self.commands * self.ref_scale, self.ref_keys * self.ref_scale)  # (n, N)
        dist, idx = torch.topk(d, self.cfg.reference_knn, dim=1, largest=False)
        w = 1.0 / (dist + 1e-3)
        w = w / w.sum(dim=1, keepdim=True)
        x = self.phase * self.ref_bins
        b0 = torch.floor(x).long() % self.ref_bins
        b1 = (b0 + 1) % self.ref_bins
        f = (x - torch.floor(x)).unsqueeze(-1)
        r0 = self.ref_table[idx, b0.unsqueeze(1).expand_as(idx)]  # (n, k, 28)
        r1 = self.ref_table[idx, b1.unsqueeze(1).expand_as(idx)]
        r = (1.0 - f.unsqueeze(1)) * r0 + f.unsqueeze(1) * r1
        return (w.unsqueeze(-1) * r).sum(dim=1)

    def _moving(self) -> torch.Tensor:
        return torch.linalg.norm(self.commands, dim=-1) > self.cfg.duck_command_eps

    # ------------------------------------------------------------------ actions
    def _pre_physics_step(self, actions: torch.Tensor):
        self.phase = (self.phase + self.step_dt / self.ref_period) % 1.0
        # brax PPO (Open Duck) samples tanh-squashed actions in (-1, 1); skrl's Gaussian is unbounded. Without this the
        # policy mean drifted to 1e12 (duck v1 collapse at 25k): the >= 0 reward clip hid the action-rate penalty.
        actions = torch.clamp(actions, -1.0, 1.0)
        super()._pre_physics_step(actions)  # history, commands, pushes, target = default + scale * a
        # Open Duck motor speed limit on the targets (rad per control step)
        step = self.cfg.max_motor_velocity * self.step_dt
        legs = self.joint_target[:, self.leg_ids]
        legs = torch.max(torch.min(legs, self.motor_targets + step), self.motor_targets - step)
        self.joint_target[:, self.leg_ids] = legs
        self.motor_targets = legs.clone()
        self.ref = self._reference()

    # ------------------------------------------------------------------ observations
    def _get_observations(self) -> dict:
        _, lin_b, ang_b, grav_b = self._base()
        q = self.robot.data.joint_pos.torch
        qd = self.robot.data.joint_vel.torch[:, self.leg_ids]
        q_rel = (q - self.default_joint_pos)[:, self.leg_ids]
        dev = self.device
        imu = torch.cat((quat_apply(self.imu_misalign, ang_b), quat_apply(self.imu_misalign, grav_b)), dim=-1)
        self.imu_hist = torch.roll(self.imu_hist, 1, dims=1)
        self.imu_hist[:, 0] = imu
        lag = torch.randint(0, self.cfg.imu_delay_max_steps + 1, (self.num_envs,), device=dev) \
            if self.cfg.imu_delay_per_step else self.imu_lag
        imu_seen = self.imu_hist[torch.arange(self.num_envs, device=dev), lag]
        jv_seen = self.prev_joint_vel if self.cfg.joint_vel_delay_steps > 0 else qd
        self.prev_joint_vel = qd.clone()

        def u(x, a):
            return x + (torch.rand_like(x) * 2.0 - 1.0) * a

        c, s = torch.cos(2 * math.pi * self.phase), torch.sin(2 * math.pi * self.phase)
        phase = torch.stack((c, s), dim=-1)
        mt = self.motor_targets - self.default_joint_pos[:, self.leg_ids]
        vs = self.cfg.joint_vel_obs_scale
        policy = torch.cat(
            (
                u(imu_seen[:, :3], self.cfg.noise_ang_vel),
                u(imu_seen[:, 3:], self.cfg.noise_gravity),
                u(q_rel + self.encoder_bias, self.noise_q),
                u(jv_seen, self.cfg.noise_joint_vel) * vs,
                self.act_hist.flatten(1),
                mt,
                phase,
                self.commands,
            ),
            dim=-1,
        )
        _, _, height = self._feet()
        contact = (height < self.cfg.contact_height).float()
        critic = torch.cat(
            (ang_b, grav_b, q_rel, qd * vs, self.act_hist.flatten(1), mt, phase, self.commands, lin_b, height,
             self.air_time, contact, self.ref[:, 0:10] - self.default_joint_pos[:, self.leg_ids], self.ref[:, 20:22]),
            dim=-1,
        )
        return {"policy": policy, "critic": critic}

    # ------------------------------------------------------------------ rewards
    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        quat, lin_b, ang_b, _ = self._base()
        q = self.robot.data.joint_pos.torch
        qd_all = self.robot.data.joint_vel.torch
        ql, qdl = q[:, self.leg_ids], qd_all[:, self.leg_ids]
        cmd = self.commands
        moving = self._moving().float()

        # tracking (Open Duck rewards.py): x squared, y with a tolerance; yaw rate
        ex = torch.square(cmd[:, 0] - lin_b[:, 0])
        ey = torch.square((torch.abs(lin_b[:, 1] - cmd[:, 1]) - cfg.duck_tracking_y_tol).clamp_min(0.0))
        r_lin = torch.exp(-(ex + ey) / cfg.duck_tracking_sigma_lin)
        r_ang = torch.exp(-torch.square(cmd[:, 2] - ang_b[:, 2]) / cfg.duck_tracking_sigma_ang)

        # torques of the drive (PD on the delayed / played target, as applied)
        kp = self.robot.data.joint_stiffness.torch
        kd = self.robot.data.joint_damping.torch
        tau = (kp * (self.joint_target - q) - kd * qd_all).clamp(-self.effort_limit, self.effort_limit)[:, self.leg_ids]
        c_tau = torch.sum(torch.square(tau), dim=1)
        c_rate = torch.sum(torch.square(self.actions - self.prev_actions), dim=1)
        c_stand = (torch.sum(torch.abs(ql - self.default_joint_pos[:, self.leg_ids]), dim=1)
                   + torch.sum(torch.abs(qdl), dim=1)) * (1.0 - moving)

        # imitation (Open Duck custom_rewards.py), heading-frame velocities
        ref = self.ref
        yaw_q = quat.clone()
        fwd = quat_apply(quat, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3))
        yaw = torch.atan2(fwd[:, 1], fwd[:, 0])
        lin_w = self.robot.data.body_lin_vel_w.torch[:, self.pelvis_id]
        ang_w = self.robot.data.body_ang_vel_w.torch[:, self.pelvis_id]
        cy, sy = torch.cos(-yaw), torch.sin(-yaw)

        def heading(v):
            return torch.stack((cy * v[:, 0] - sy * v[:, 1], sy * v[:, 0] + cy * v[:, 1], v[:, 2]), dim=-1)

        lin_h, ang_h = heading(lin_w), heading(ang_w)
        _, _, height = self._feet()
        contact = (height < cfg.contact_height).float()
        im = (torch.exp(-8.0 * torch.sum(torch.square(lin_h[:, :2] - ref[:, 22:24]), dim=1))
              + torch.exp(-8.0 * torch.square(lin_h[:, 2] - ref[:, 24]))
              + 0.5 * torch.exp(-2.0 * torch.sum(torch.square(ang_h[:, :2] - ref[:, 25:27]), dim=1))
              + 0.5 * torch.exp(-2.0 * torch.square(ang_h[:, 2] - ref[:, 27]))
              - cfg.duck_imitation_joint_pos * torch.sum(torch.square(ql - ref[:, 0:10]), dim=1)
              - cfg.duck_imitation_joint_vel * torch.sum(torch.square(qdl - ref[:, 10:20]), dim=1)
              + torch.sum((contact == ref[:, 20:22]).float(), dim=1)) * moving
        del yaw_q

        terms = {
            "tracking_lin_vel": cfg.duck_tracking_lin_vel * r_lin,
            "tracking_ang_vel": cfg.duck_tracking_ang_vel * r_ang,
            "torques": cfg.duck_torques * c_tau,
            "action_rate": cfg.duck_action_rate * c_rate,
            "stand_still": cfg.duck_stand_still * c_stand,
            "alive": cfg.duck_alive * torch.ones_like(r_lin),
            "imitation": cfg.duck_imitation * im,
        }
        reward = torch.clamp(sum(terms.values()) * self.step_dt, min=0.0)

        # gait bookkeeping (critic air time / contact) as in the base env
        first = (self.air_time > 0.0) & (contact > 0.5)
        self.air_time = torch.where(contact > 0.5, torch.zeros_like(self.air_time), self.air_time + self.step_dt)
        del first
        log = {f"Reward/{k}": (v * self.step_dt).mean() for k, v in terms.items()}
        log.update({
            "Duck/imitation_joint_err_deg": torch.rad2deg(torch.sqrt(torch.mean(torch.square(ql - ref[:, 0:10]), dim=1))).mean(),
            "Duck/contact_match": (contact == ref[:, 20:22]).float().mean(),
            "Duck/lin_vel_err": torch.abs(cmd[:, 0] - lin_b[:, 0]).mean(),
            "Duck/yaw_rate_err": torch.abs(cmd[:, 2] - ang_b[:, 2]).mean(),
            "Duck/moving": moving.mean(),
        })
        self.extras["log"] = log
        return reward

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if hasattr(self, "phase"):
            self.phase[env_ids] = 0.0
            self.motor_targets[env_ids] = self.default_joint_pos[env_ids][:, self.leg_ids]
            self.air_time[env_ids] = 0.0
