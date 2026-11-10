"""Phase-binned reference table from the Open Duck style placo sweep (motion_generator scripts/sweep_motions.py).

    python scripts/duck/build_duck_reference.py --clips source/arti_d1/arti_d1/assets/motions/arti_d1_duck \
        --out source/arti_d1/arti_d1/assets/motions/arti_d1_duck_reference.npz

For every clip: the periodic part (periodic_valid) is binned by gait_phase into --bins bins and averaged. Stored per
clip and bin (leg joints in Isaac / Dynamixel order without the arms):
    q (10) joint pos rad, qd (10) joint vel rad/s, contact (2) left/right, v (3) base linear velocity in the heading
    frame, w (3) base angular velocity in the heading frame.
Key per clip: its measured average (vx, vy, wz) (the env looks the reference up by command). Also period (s).
"""

import argparse
import glob
import json
import os

import numpy as np

LEGS = ["left_hip_yaw", "right_hip_yaw", "left_hip_roll", "right_hip_roll", "left_hip_pitch", "right_hip_pitch",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]


def yaw_of(q_xyzw):
    x, y, z, w = q_xyzw.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def to_heading(vec, yaw):
    c, s = np.cos(-yaw), np.sin(-yaw)
    out = vec.copy()
    out[:, 0] = c * vec[:, 0] - s * vec[:, 1]
    out[:, 1] = s * vec[:, 0] + c * vec[:, 1]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bins", type=int, default=100)
    a = p.parse_args()
    files = sorted(glob.glob(os.path.join(a.clips, "[0-9]*.json")), key=lambda f: int(os.path.basename(f)[:-5]))
    keys, tables, periods = [], [], []
    for f in files:
        d = json.load(open(f))
        off, size = d["frame_offset"][0], d["frame_size"][0]
        sl = lambda k: slice(off[k], off[k] + size[k])
        fr = np.array(d["frames"])
        fr = fr[fr[:, off["periodic_valid"]] > 0.5]
        idx = [d["joints"].index(n) for n in LEGS]
        q = fr[:, sl("joints_positions")][:, idx]
        qd = fr[:, sl("joints_velocities")][:, idx]
        contact = fr[:, sl("foot_contacts")]
        yaw = yaw_of(fr[:, sl("root_orientation_quat")])
        v = to_heading(fr[:, sl("world_linear_velocity")], yaw)
        w = to_heading(fr[:, sl("world_angular_velocity")], yaw)
        phase = fr[:, off["gait_phase"]] % 1.0
        b = np.minimum((phase * a.bins).astype(int), a.bins - 1)
        rows = np.concatenate([q, qd, contact, v, w], axis=1)
        table = np.zeros((a.bins, rows.shape[1]))
        for k in range(a.bins):
            m = b == k
            if not m.any():
                raise RuntimeError(f"{f}: phase bin {k} empty (too few periodic frames for {a.bins} bins)")
            table[k] = rows[m].mean(0)
        table[:, 20:22] = (table[:, 20:22] > 0.5).astype(float)  # contacts back to 0 / 1
        keys.append([d["x_linear_velocity"], d["y_linear_velocity"], d["z_angular_velocity"]])
        tables.append(table)
        periods.append(d["gait_parameters"]["period"])
    periods = np.array(periods)
    if np.ptp(periods) > 1e-6:
        raise RuntimeError(f"clips have different periods {np.unique(periods)}")
    keys = np.array(keys)
    np.savez(a.out, keys=keys, table=np.array(tables, dtype=np.float32), period=periods[0], bins=a.bins,
             legs=np.array(LEGS), layout=np.array(["q:0:10", "qd:10:20", "contact:20:22", "v:22:25", "w:25:28"]))
    print(f"{len(files)} clips, period {periods[0]:.3f} s, {a.bins} bins -> {a.out}")
    print(f"command coverage vx {keys[:, 0].min():+.3f}..{keys[:, 0].max():+.3f}  vy {keys[:, 1].min():+.3f}.."
          f"{keys[:, 1].max():+.3f}  wz {keys[:, 2].min():+.3f}..{keys[:, 2].max():+.3f}")
    t = np.array(tables)
    print(f"max |qd| per leg joint: {np.abs(t[:, :, 10:20]).max((0, 1)).round(2)}")


if __name__ == "__main__":
    main()
