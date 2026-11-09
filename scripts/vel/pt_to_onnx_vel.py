"""Export an ARTI-D1 velocity-walk checkpoint (Template-Arti-D1-Vel-Direct-v0) to ONNX for humanoid_controller.

    python scripts/vel/pt_to_onnx_vel.py --checkpoint <run>/checkpoints/agent_N.pt [--output agent_vel.onnx]

ONNX input "obs" [1, 39] (ArtiD1VelEnv._get_observations, actor part, no noise):
    [0:3]   pelvis angular velocity, pelvis frame (rad/s)
    [3:6]   projected gravity, pelvis frame (unit vector, (0, 0, -1) when upright)
    [6:16]  leg joint position - default pose (rad), LEG order below
    [16:26] leg joint velocity (rad/s), LEG order
    [26:36] previous raw actions (the "actions" output of the last step), LEG order
    [36:39] command (vx m/s, vy m/s, yaw rate rad/s), unscaled
LEG order = Dynamixel order without the arms: left_hip_yaw, right_hip_yaw, left_hip_roll, right_hip_roll,
left_hip_pitch, right_hip_pitch, left_knee, right_knee, left_ankle, right_ankle.

Outputs:
    "actions" [1, 10]: raw policy mean (not clipped, the env does not clip either); feed back as obs[26:36].
    "q_des"   [1, 14]: joint targets, ISAAC_JOINT_NAMES (Dynamixel) order = clamp(default + action_scale * a, URDF
                       limits) on the legs, arm_hold_target on the arms (ArtiD1VelEnv._pre_physics_step).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

_spec = importlib.util.spec_from_file_location("p2o", Path(__file__).resolve().parents[1] / "pt_to_onnx.py")
p2o = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2o)

LEG_TYPES = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")
LEG_IDS = [i for i, n in enumerate(p2o.ISAAC_JOINT_NAMES) if any(n.endswith(t) for t in LEG_TYPES)]
LEG_NAMES = [p2o.ISAAC_JOINT_NAMES[i] for i in LEG_IDS]


def _env_yaml_dict(checkpoint_path: Path, key: str) -> dict[str, float]:
    """Top-level `key:` mapping of name: float in params/env.yaml ({} if absent)."""
    env_yaml = checkpoint_path.parents[1] / "params" / "env.yaml"
    if not env_yaml.is_file():
        return {}
    lines = env_yaml.read_text(encoding="utf-8").splitlines()
    out: dict[str, float] = {}
    for k, line in enumerate(lines):
        if line.rstrip() == f"{key}:":
            for nxt in lines[k + 1:]:
                if not nxt.startswith(" ") or ":" not in nxt:
                    break
                name, value = nxt.strip().split(":", 1)
                out[name.strip()] = float(value)
            break
    return out


class VelOnnxPolicy(nn.Module):
    def __init__(self, actor: nn.Module, normalizer: nn.Module | None, default_pos: list[float], action_scale: float,
                 lower: list[float], upper: list[float]):
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer if normalizer is not None else nn.Identity()
        self.action_scale = action_scale
        self.register_buffer("default_joint_pos", torch.tensor(default_pos, dtype=torch.float32))
        self.register_buffer("joint_lower", torch.tensor(lower, dtype=torch.float32))
        self.register_buffer("joint_upper", torch.tensor(upper, dtype=torch.float32))
        leg_scale = torch.zeros(len(LEG_IDS), len(p2o.ISAAC_JOINT_NAMES))
        for r, c in enumerate(LEG_IDS):
            leg_scale[r, c] = action_scale
        self.register_buffer("leg_scale", leg_scale)  # actions [.., 10] @ leg_scale -> per-joint offset [.., 14]

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actor(self.normalizer(obs))
        q_des = self.default_joint_pos + actions @ self.leg_scale
        return actions, torch.clamp(q_des, self.joint_lower, self.joint_upper)


def build(checkpoint_path: Path) -> VelOnnxPolicy:
    checkpoint = p2o._torch_load(checkpoint_path, map_location="cpu")
    actor = p2o.reconstruct_actor(checkpoint["policy"], p2o.infer_activation(checkpoint_path, None))
    normalizer = p2o.load_preprocessor(checkpoint)
    yaml_pos = p2o._env_yaml_joint_pos(checkpoint_path)
    if not yaml_pos:
        raise RuntimeError(f"no init_state.joint_pos in {checkpoint_path.parents[1] / 'params/env.yaml'}")
    arm_hold = _env_yaml_dict(checkpoint_path, "arm_hold_target")
    if not arm_hold:
        raise RuntimeError("no arm_hold_target in params/env.yaml (not a velocity-walk run?)")
    by_name = {**yaml_pos, **arm_hold}  # env: default_joint_pos[:, arm_ids] = arm_hold
    default = [by_name[n] for n in p2o.ISAAC_JOINT_NAMES]
    scale = p2o._env_yaml_float(checkpoint_path, "action_scale", 0.0)
    if scale <= 0.0:
        raise RuntimeError("no action_scale in params/env.yaml")
    # soft_joint_pos_limit_factor 1.0 (params/env.yaml) -> the URDF limits
    lower, upper = p2o._to_isaac_order(p2o.JOINT_LOWER), p2o._to_isaac_order(p2o.JOINT_UPPER)
    model = VelOnnxPolicy(actor, normalizer, default, scale, lower, upper).eval()
    obs_dim = next(actor.parameters()).shape[1]
    act_dim = actor[-1].out_features
    if obs_dim != 39 or act_dim != len(LEG_IDS):
        raise RuntimeError(f"policy MLP {obs_dim} -> {act_dim}, expected 39 -> {len(LEG_IDS)}")
    print(f"[INFO] action_scale {scale}, arms held at {arm_hold}")
    print(f"[INFO] default pose (Dynamixel order): {dict(zip(p2o.ISAAC_JOINT_NAMES, default))}")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="defaults to <checkpoint>.onnx")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()
    ckpt = args.checkpoint.resolve()
    out = (args.output or ckpt.with_suffix(".onnx")).resolve()
    model = build(ckpt)
    dummy = torch.zeros(1, 39)
    torch.onnx.export(model, dummy, str(out), input_names=["obs"], output_names=["actions", "q_des"],
                      opset_version=args.opset, dynamo=False)
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        obs = rng.normal(size=(1, 39)).astype(np.float32)
        a, q = sess.run(["actions", "q_des"], {"obs": obs})
        with torch.no_grad():
            ta, tq = model(torch.from_numpy(obs))
        worst = max(worst, float(abs(a - ta.numpy()).max()), float(abs(q - tq.numpy()).max()))
    print(f"[VERIFY] 200 random obs: max |torch - onnx| = {worst:.2e}")
    print(f"[INFO] leg order (obs joint terms, actions): {LEG_NAMES}")
    print(f"[INFO] q_des order: {p2o.ISAAC_JOINT_NAMES}")
    print(f"[INFO] wrote {out}")
    return 0 if worst < 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main())
