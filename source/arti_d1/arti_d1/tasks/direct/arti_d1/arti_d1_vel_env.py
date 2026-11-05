"""ARTI-D1 velocity-tracking walk after microduck_rl (see arti_d1_vel_env_cfg.py for the recipe and what was rescaled)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_angle_axis

from arti_d1.robots.arti_d1 import ARTI_D1_POLICY_JOINT_ORDER

from .arti_d1_vel_env_cfg import ArtiD1VelEnvCfg

LEG_TYPES = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")


def _joint_type(name: str) -> str:
    return name.split("_", 1)[1]


class ArtiD1VelEnv(DirectRLEnv):
    cfg: ArtiD1VelEnvCfg

    def __init__(self, cfg: ArtiD1VelEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.scene.mti3_imu = None
        super().__init__(cfg, render_mode, **kwargs)
        names = list(self.robot.data.joint_names)
        if names != list(ARTI_D1_POLICY_JOINT_ORDER):
            raise RuntimeError(f"joint order {names} != Dynamixel order {list(ARTI_D1_POLICY_JOINT_ORDER)}")
        dev, n, nj = self.device, self.num_envs, len(names)

        def per_type(values: dict, default: float = 0.0) -> torch.Tensor:
            return torch.tensor([values.get(_joint_type(j), default) for j in names], device=dev, dtype=torch.float32)

        self.default_joint_pos = self.robot.data.default_joint_pos.torch.clone()
        self.leg_mask = torch.tensor([_joint_type(j) in LEG_TYPES for j in names], device=dev)
        self.leg_ids = torch.nonzero(self.leg_mask).flatten()
        nl = len(self.leg_ids)
        if self.cfg.action_space != nl:
            raise RuntimeError(f"action_space {self.cfg.action_space} != {nl} leg joints")
        # arms: no policy output, held at the hold target
        self.arm_ids = torch.tensor([names.index(j) for j in self.cfg.arm_hold_target], device=dev, dtype=torch.long)
        self.arm_hold = torch.tensor(list(self.cfg.arm_hold_target.values()), device=dev, dtype=torch.float32)
        self.default_joint_pos[:, self.arm_ids] = self.arm_hold
        limits = self.robot.data.soft_joint_pos_limits.torch
        self.q_lower, self.q_upper = limits[..., 0], limits[..., 1]

        # Dynamixel joint model (as ArtiD1AmpEnv._init_actuator_model)
        sim_dt = self.cfg.sim.dt
        lo, hi = self.cfg.actuator_delay_jitter_ms
        delay_ms = self.cfg.actuator_delay_ms + torch.empty(n, 1, device=dev).uniform_(lo, hi).expand(n, nj)
        self.act_delay_steps = torch.clamp(torch.round(delay_ms.clamp_min(0.0) / 1000.0 / sim_dt), min=0).long()
        self.act_buffer = torch.zeros(int(self.act_delay_steps.max().item()) + 1, n, nj, device=dev)
        self.act_head = 0
        lo, hi = self.cfg.actuator_armature_scale_range
        armature = per_type(self.cfg.actuator_armature)[None, :] * torch.empty(n, nj, device=dev).uniform_(lo, hi)
        self.robot.write_joint_armature_to_sim_index(armature=armature)
        lo, hi = self.cfg.actuator_coulomb_scale_range
        self.act_coulomb = per_type(self.cfg.actuator_coulomb)[None, :] * torch.empty(n, nj, device=dev).uniform_(lo, hi)
        self.act_stuck = torch.zeros(n, nj, dtype=torch.bool, device=dev)
        self.act_q_stick = torch.zeros(n, nj, device=dev)
        self.effort_limit = torch.tensor([self.cfg.joint_effort_limit[j] for j in names], device=dev)

        # pose reward stds over the leg joints
        self.pose_std_stand = per_type(self.cfg.pose_std_standing, 1.0)[self.leg_ids]
        self.pose_std_walk = per_type(self.cfg.pose_std_walking, 1.0)[self.leg_ids]

        bodies = list(self.robot.data.body_names)
        self.pelvis_id = bodies.index("pelvis")
        self.foot_ids = [bodies.index(self.cfg.left_foot_body), bodies.index(self.cfg.right_foot_body)]
        self.foot_offsets = torch.tensor([self.cfg.left_foot_offset, self.cfg.right_foot_offset], device=dev)
        self.sole_corners = torch.tensor(
            [[sx * self.cfg.foot_sole_half_length, sy * self.cfg.foot_sole_half_width, 0.0] for sx in (-1, 1) for sy in (-1, 1)],
            device=dev,
        )
        self.trunk_offset = torch.tensor(self.cfg.trunk_offset, device=dev).expand(n, 3)
        self.gravity_w = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(n, 3)
        self.up_b = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)

        self.actions = torch.zeros(n, nl, device=dev)
        self.prev_actions = torch.zeros(n, nl, device=dev)
        self.commands = torch.zeros(n, 3, device=dev)
        self.cmd_timer = torch.zeros(n, device=dev)
        self.push_timer = torch.zeros(n, device=dev)
        self.air_time = torch.zeros(n, 2, device=dev)
        self.contact_time = torch.zeros(n, 2, device=dev)
        self.last_contact = torch.ones(n, 2, dtype=torch.bool, device=dev)
        self.peak_height = torch.zeros(n, 2, device=dev)

        # observation corruption state (actor only)
        self.encoder_bias = torch.empty(n, nl, device=dev).uniform_(-self.cfg.encoder_bias_range, self.cfg.encoder_bias_range)
        axis = torch.nn.functional.normalize(torch.randn(n, 3, device=dev), dim=-1)
        angle = torch.empty(n, device=dev).uniform_(0.0, math.radians(self.cfg.imu_misalign_deg))
        self.imu_misalign = quat_from_angle_axis(angle, axis)
        self.imu_lag = torch.zeros(n, dtype=torch.long, device=dev)
        self.prev_imu = torch.zeros(n, 6, device=dev)
        self.prev_joint_vel = torch.zeros(n, nl, device=dev)

        # scripts/skrl/play.py: it writes commands * command_scale into the last 3 obs and plays this sequence (5 s each)
        self.command_scale = torch.ones(3, device=dev)
        self.play_command_sequence = (
            ("forward", (0.12, 0.0, 0.0)),
            ("stand", (0.0, 0.0, 0.0)),
            ("side_left", (0.0, 0.06, 0.0)),
            ("backward", (-0.10, 0.0, 0.0)),
            ("stand", (0.0, 0.0, 0.0)),
            ("diagonal_right", (0.10, -0.05, 0.0)),
            ("turn_left", (0.0, 0.0, 0.3)),
            ("stand", (0.0, 0.0, 0.0)),
        )
        self._log: dict[str, torch.Tensor] = {}

        # left/right mirror of the policy obs / actions (scripts/skrl/train.py symmetry loss). Leg order is Dynamixel order
        # without the arms: (L,R) hip_yaw, hip_roll, hip_pitch, knee, ankle. Mirror = swap L/R, negate yaw and roll
        # (same-signed joint motion moves both feet the same world direction, sysid expect_positive).
        leg_names = [names[i] for i in self.leg_ids.tolist()]
        perm, sign = [], []
        for j in leg_names:
            other = j.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
            perm.append(leg_names.index(other))
            sign.append(-1.0 if _joint_type(j) in ("hip_yaw", "hip_roll") else 1.0)
        nl_perm = torch.tensor(perm, device=dev)
        nl_sign = torch.tensor(sign, device=dev)
        o_perm = list(range(6))
        o_sign = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]  # ang vel (wx, wy, wz) -> (-wx, wy, -wz); gravity (gx, gy, gz) -> (gx, -gy, gz)
        for base in (6, 6 + nl, 6 + 2 * nl):  # joint pos, joint vel, last action
            o_perm += [base + i for i in perm]
            o_sign += sign
        o_perm += [6 + 3 * nl, 6 + 3 * nl + 1, 6 + 3 * nl + 2]
        o_sign += [1.0, -1.0, -1.0]  # command (vx, vy, wz)
        self.symmetry = {
            "obs_perm": torch.tensor(o_perm, device=dev), "obs_sign": torch.tensor(o_sign, device=dev),
            "act_perm": nl_perm, "act_sign": nl_sign, "coef": self.cfg.symmetry_loss_coef,
            "start_step": self.cfg.symmetry_kick_start_step, "env": self,
        }
        print(f"[ARTI-D1 VEL] {n} envs, policy obs {self.cfg.observation_space}, critic {self.cfg.state_space}, "
              f"{nl} leg actions, scale {self.cfg.action_scale}, arms held at {self.cfg.arm_hold_target}", flush=True)

    # ------------------------------------------------------------------ scene / actions
    def _setup_scene(self):
        self.robot: Articulation = self.scene["robot"]

    def _step(self) -> int:
        return self.common_step_counter + self.cfg.curriculum_step_offset

    @staticmethod
    def _stage(stages, step: int) -> float:
        value = stages[0][1]
        for s, v in stages:
            if step >= s:
                value = v
        return value

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions.copy_(self.actions)
        self.actions = actions.clone()
        self.cmd_timer -= self.step_dt
        ids = torch.nonzero(self.cmd_timer <= 0.0).flatten()
        if len(ids):
            self._resample_commands(ids)
        self.push_timer -= self.step_dt
        ids = torch.nonzero(self.push_timer <= 0.0).flatten()
        if len(ids):
            vel = self.robot.data.root_vel_w.torch[ids].clone()
            vel[:, :2] += torch.empty(len(ids), 2, device=self.device).uniform_(-self.cfg.push_velocity, self.cfg.push_velocity)
            self.robot.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
            self.push_timer[ids] = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.push_interval_s)
        target = self.default_joint_pos.clone()
        target[:, self.leg_ids] += self.cfg.action_scale * self.actions
        self.joint_target = torch.clamp(target, self.q_lower, self.q_upper)

    def _apply_action(self):
        # per physics step: bus delay, Coulomb friction with stiction, voltage saturation of the drive term
        steps = self.act_buffer.shape[0]
        self.act_head = (self.act_head + 1) % steps
        self.act_buffer[self.act_head] = self.joint_target
        index = (self.act_head - self.act_delay_steps) % steps
        delayed = torch.gather(self.act_buffer.permute(1, 2, 0), 2, index.unsqueeze(-1)).squeeze(-1)
        q = self.robot.data.joint_pos.torch
        qd = self.robot.data.joint_vel.torch
        kp = self.robot.data.joint_stiffness.torch.clamp_min(1e-3)
        err = delayed - q
        moving = qd.abs() > self.cfg.actuator_stick_velocity
        stick = ~moving & (kp * err.abs() <= self.act_coulomb)
        self.act_q_stick = torch.where(stick & ~self.act_stuck, q, self.act_q_stick)
        self.act_stuck = stick
        direction = torch.where(moving, torch.sign(qd), torch.sign(err))
        target = torch.where(stick, self.act_q_stick, delayed - direction * self.act_coulomb / kp)
        reach = self.effort_limit / kp
        target = q + torch.clamp(target - q, -reach, reach)
        self.robot.actuators.target_command.set_position_index(value=target)

    # ------------------------------------------------------------------ commands
    def _resample_commands(self, ids: torch.Tensor):
        k, dev = len(ids), self.device
        cmd = torch.empty(k, 3, device=dev)
        cmd[:, 0].uniform_(*self.cfg.command_lin_x_range)
        cmd[:, 1].uniform_(*self.cfg.command_lin_y_range)
        cmd[:, 2].uniform_(*self.cfg.command_yaw_range)
        r = torch.rand(k, device=dev)
        stand_frac = self._stage(self.cfg.standing_stages, self._step())
        turn = (r >= stand_frac) & (r < stand_frac + self.cfg.turn_in_place_fraction)
        ymax = max(abs(self.cfg.command_yaw_range[0]), abs(self.cfg.command_yaw_range[1]))
        spin = torch.empty(k, device=dev).uniform_(0.4 * ymax, ymax) * torch.where(torch.rand(k, device=dev) < 0.5, -1.0, 1.0)
        cmd[turn, :2] = 0.0
        cmd[turn, 2] = spin[turn]
        # pure sideways bucket (microduck has none; ARTI-D1 did not learn lin y from the uniform draw)
        lo = stand_frac + self.cfg.turn_in_place_fraction
        side = (r >= lo) & (r < lo + self.cfg.lateral_fraction)
        ymax_l = max(abs(self.cfg.command_lin_y_range[0]), abs(self.cfg.command_lin_y_range[1]))
        sway = torch.empty(k, device=dev).uniform_(0.4 * ymax_l, ymax_l) * torch.where(torch.rand(k, device=dev) < 0.5, -1.0, 1.0)
        cmd[side, 0] = 0.0
        cmd[side, 1] = sway[side]
        cmd[side, 2] = 0.0
        cmd[r < stand_frac] = 0.0
        self.commands[ids] = cmd
        self.cmd_timer[ids] = torch.empty(k, device=dev).uniform_(*self.cfg.command_resample_s)

    def _moving(self) -> torch.Tensor:
        return (torch.linalg.norm(self.commands[:, :2], dim=-1) + self.commands[:, 2].abs()) > self.cfg.command_threshold

    # ------------------------------------------------------------------ state helpers
    def _feet(self):
        """Sole-centre position / velocity (world) and lowest sole-corner height per foot."""
        pos = self.robot.data.body_pos_w.torch[:, self.foot_ids]
        quat = self.robot.data.body_quat_w.torch[:, self.foot_ids]
        lin = self.robot.data.body_lin_vel_w.torch[:, self.foot_ids]
        ang = self.robot.data.body_ang_vel_w.torch[:, self.foot_ids]
        n = self.num_envs
        off = quat_apply(quat.reshape(-1, 4), self.foot_offsets.unsqueeze(0).expand(n, -1, -1).reshape(-1, 3)).view(n, 2, 3)
        sole = pos + off
        vel = lin + torch.cross(ang, off, dim=-1)
        corners = quat_apply(
            quat.unsqueeze(2).expand(-1, -1, 4, -1).reshape(-1, 4),
            (self.foot_offsets.view(1, 2, 1, 3) + self.sole_corners.view(1, 1, 4, 3)).expand(n, -1, -1, -1).reshape(-1, 3),
        ).view(n, 2, 4, 3)
        height = (pos[..., 2:3] + corners[..., 2]).min(dim=-1).values
        return sole, vel, height

    def _base(self):
        # linear velocity of the trunk point (pelvis + trunk_offset; the pelvis link origin sits at foot level, where
        # leg swing / body roll add velocity the trunk does not have). microduck: trunk_base site.
        quat = self.robot.data.body_quat_w.torch[:, self.pelvis_id]
        ang_w = self.robot.data.body_ang_vel_w.torch[:, self.pelvis_id]
        lin_w = self.robot.data.body_lin_vel_w.torch[:, self.pelvis_id]
        if self.cfg.lin_vel_at_trunk:
            lin_w = lin_w + torch.cross(ang_w, quat_apply(quat, self.trunk_offset), dim=-1)
        lin_b = quat_apply_inverse(quat, lin_w)
        ang_b = quat_apply_inverse(quat, ang_w)
        grav_b = quat_apply_inverse(quat, self.gravity_w)
        return quat, lin_b, ang_b, grav_b

    # ------------------------------------------------------------------ observations
    def _get_observations(self) -> dict:
        _, lin_b, ang_b, grav_b = self._base()
        q = self.robot.data.joint_pos.torch
        qd = self.robot.data.joint_vel.torch
        q_rel = (q - self.default_joint_pos)[:, self.leg_ids]
        qd = qd[:, self.leg_ids]
        dev = self.device

        # actor: IMU misalignment, 0..1 step lag, noise; encoder bias; joint vel one step old; noise
        imu = torch.cat((quat_apply(self.imu_misalign, ang_b), quat_apply(self.imu_misalign, grav_b)), dim=-1)
        imu_seen = torch.where((self.imu_lag > 0).unsqueeze(-1), self.prev_imu, imu)
        self.prev_imu = imu
        jv_seen = self.prev_joint_vel if self.cfg.joint_vel_delay_steps > 0 else qd
        self.prev_joint_vel = qd.clone()

        def u(x, a):
            return x + (torch.rand_like(x) * 2.0 - 1.0) * a

        policy = torch.cat(
            (
                u(imu_seen[:, :3], self.cfg.noise_ang_vel),
                u(imu_seen[:, 3:], self.cfg.noise_gravity),
                u(q_rel + self.encoder_bias, self.cfg.noise_joint_pos),
                u(jv_seen, self.cfg.noise_joint_vel),
                self.actions,
                self.commands,
            ),
            dim=-1,
        )
        _, _, height = self._feet()
        contact = (height < self.cfg.contact_height).float()
        critic = torch.cat(
            (ang_b, grav_b, q_rel, qd, self.actions, self.commands, lin_b, height, self.air_time, contact), dim=-1
        )
        return {"policy": policy, "critic": critic}

    # ------------------------------------------------------------------ rewards
    def _get_rewards(self) -> torch.Tensor:
        cfg, step = self.cfg, self._step()
        quat, lin_b, ang_b, grav_b = self._base()
        moving = self._moving().float()
        sole, foot_vel, height = self._feet()
        contact = height < cfg.contact_height

        # contact timers (mjlab ContactSensor track_air_time)
        first_contact = contact & ~self.last_contact
        self.air_time = torch.where(contact, torch.zeros_like(self.air_time), self.air_time + self.step_dt)
        self.contact_time = torch.where(contact, self.contact_time + self.step_dt, torch.zeros_like(self.contact_time))
        self.last_contact = contact

        lin_err = torch.sum(torch.square(self.commands[:, :2] - lin_b[:, :2]), dim=1) + torch.square(lin_b[:, 2])
        r_lin = torch.exp(-lin_err / cfg.track_lin_vel_std**2)
        r_lat = torch.exp(-torch.square(self.commands[:, 1] - lin_b[:, 1]) / cfg.track_lat_vel_std**2)
        died = self.reset_terminated.float()
        ang_err = torch.square(self.commands[:, 2] - ang_b[:, 2]) + torch.sum(torch.square(ang_b[:, :2]), dim=1)
        r_ang = torch.exp(-ang_err / cfg.track_ang_vel_std**2)
        r_up = torch.exp(-torch.sum(torch.square(grav_b[:, :2]), dim=1) / cfg.upright_std**2)

        q = self.robot.data.joint_pos.torch
        err2 = torch.square(q[:, self.leg_ids] - self.default_joint_pos[:, self.leg_ids])
        std = torch.where(moving.bool().unsqueeze(-1), self.pose_std_walk, self.pose_std_stand)
        r_pose = torch.exp(-torch.mean(err2 / std**2, dim=1))

        c_ang = torch.sum(torch.square(self.robot.data.body_ang_vel_w.torch[:, self.pelvis_id, :2]), dim=1)
        c_lim = torch.sum(
            (self.q_lower - q).clamp_min(0.0)[:, self.leg_ids] + (q - self.q_upper).clamp_min(0.0)[:, self.leg_ids], dim=1
        )
        c_rate = torch.sum(torch.square(self.actions - self.prev_actions), dim=1)

        in_range = (self.air_time > cfg.air_time_min) & (self.air_time < cfg.air_time_max)
        r_air = torch.sum(in_range.float(), dim=1) * moving

        speed_xy = torch.linalg.norm(foot_vel[..., :2], dim=-1)
        c_clear = torch.sum(torch.abs(height - cfg.swing_target_height) * speed_xy, dim=1) * moving

        self.peak_height = torch.where(~contact, torch.maximum(self.peak_height, height), self.peak_height)
        swing_err = torch.square(self.peak_height / cfg.swing_target_height - 1.0)
        c_swing = torch.sum(swing_err * first_contact.float(), dim=1) * moving
        landed_peak = torch.sum(self.peak_height * first_contact.float()) / first_contact.float().sum().clamp_min(1.0)
        self.peak_height = torch.where(first_contact, torch.zeros_like(self.peak_height), self.peak_height)

        c_slip = torch.sum(torch.square(speed_xy) * contact.float(), dim=1) * moving
        # v4: swing-foot horizontal speed above the limit (right-foot kick at ~1.5 m/s vs left 0.8 m/s in v2)
        c_kick = torch.sum(torch.square((speed_xy - cfg.swing_speed_limit).clamp_min(0.0)) * (~contact).float(), dim=1)

        w_rate = self._stage(cfg.action_rate_stages, step)
        terms = {
            "track_lin_vel": cfg.track_lin_vel_weight * r_lin,
            "track_ang_vel": cfg.track_ang_vel_weight * r_ang,
            "track_lat_vel": cfg.track_lat_vel_weight * r_lat,
            "alive": cfg.alive_weight * (1.0 - died),
            # one-off on the falling step: weight / step_dt so that x step_dt below gives the raw weight
            "termination": cfg.termination_weight / self.step_dt * died,
            "upright": cfg.upright_weight * r_up,
            "pose": cfg.pose_weight * r_pose,
            "body_ang_vel": cfg.body_ang_vel_weight * c_ang,
            "dof_pos_limits": cfg.dof_pos_limits_weight * c_lim,
            "action_rate": w_rate * c_rate,
            "air_time": cfg.air_time_weight * r_air,
            "foot_clearance": cfg.foot_clearance_weight * c_clear,
            "foot_swing_height": cfg.foot_swing_height_weight * c_swing,
            "foot_slip": cfg.foot_slip_weight * c_slip,
            "swing_speed": (cfg.swing_speed_weight if step >= cfg.symmetry_kick_start_step else 0.0) * c_kick,
        }
        reward = torch.zeros(self.num_envs, device=self.device)
        log = {}
        for k, v in terms.items():
            reward += v * self.step_dt
            log[f"Reward/{k}"] = v.mean()
        slip_speed = torch.sum(speed_xy * contact.float()) / contact.float().sum().clamp_min(1.0)
        roll = torch.atan2(grav_b[:, 1], -grav_b[:, 2])
        log.update({
            "Reward/total": reward.mean() / self.step_dt,
            "Curriculum/action_rate_weight": torch.tensor(w_rate),
            "Curriculum/standing_fraction": torch.tensor(self._stage(cfg.standing_stages, step)),
            "Gait/lin_vel_err": torch.sqrt(torch.sum(torch.square(self.commands[:, :2] - lin_b[:, :2]), dim=1)).mean(),
            "Gait/yaw_rate_err": (self.commands[:, 2] - ang_b[:, 2]).abs().mean(),
            "Gait/air_ratio_left": 1.0 - contact[:, 0].float().mean(),
            "Gait/air_ratio_right": 1.0 - contact[:, 1].float().mean(),
            "Gait/peak_height_at_landing_mm": landed_peak * 1000.0,
            "Gait/slip_speed_mm_s": slip_speed * 1000.0,
            "Gait/base_roll_abs_deg": torch.rad2deg(roll.abs()).mean(),
            "Gait/standing_now": (1.0 - moving).mean(),
        })
        self.extras["log"] = log
        return reward

    # ------------------------------------------------------------------ dones / reset
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        quat = self.robot.data.body_quat_w.torch[:, self.pelvis_id]
        up_z = quat_apply(quat, self.up_b)[:, 2]
        trunk_z = self.robot.data.body_pos_w.torch[:, self.pelvis_id, 2] + quat_apply(quat, self.trunk_offset)[:, 2]
        died = (up_z < math.cos(self.cfg.fell_over_angle)) | (trunk_z < self.cfg.min_trunk_height)
        died |= ~torch.isfinite(self.robot.data.joint_pos.torch).all(dim=-1)
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        k, dev = len(env_ids), self.device

        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        pose[:, :2] += torch.empty(k, 2, device=dev).uniform_(-self.cfg.reset_xy_range, self.cfg.reset_xy_range)
        pose[:, 2] = self.scene.env_origins[env_ids, 2] + self.cfg.reset_height
        yaw = torch.empty(k, device=dev).uniform_(-math.pi, math.pi)
        z = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(k, 3)
        pose[:, 3:7] = quat_from_angle_axis(yaw, z)
        joint_pos = self.default_joint_pos[env_ids].clone()
        self.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros(k, 6, device=dev), env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=env_ids)

        self.act_buffer[:, env_ids] = joint_pos.unsqueeze(0)
        self.act_stuck[env_ids] = False
        self.joint_target = getattr(self, "joint_target", self.default_joint_pos.clone())
        self.joint_target[env_ids] = joint_pos
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.air_time[env_ids] = 0.0
        self.contact_time[env_ids] = 0.0
        self.last_contact[env_ids] = True
        self.peak_height[env_ids] = 0.0
        self.prev_imu[env_ids] = 0.0
        self.prev_imu[env_ids, 5] = -1.0
        self.prev_joint_vel[env_ids] = 0.0
        self.imu_lag[env_ids] = torch.randint(0, self.cfg.imu_delay_max_steps + 1, (k,), device=dev)
        self._resample_commands(env_ids)
        self.push_timer[env_ids] = torch.empty(k, device=dev).uniform_(*self.cfg.push_interval_s)
