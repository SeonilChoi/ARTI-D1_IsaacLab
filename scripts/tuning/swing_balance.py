"""Single-support balance: does the body tip toward the swing foot (so the foot has to come down fast)?

python swing_balance.py <trace.npz from eval_gait --mode score --dump> [motion_dir]

Per swing (one foot off, the other planted; forward-walking only) for the policy trace and the demo clips:
  off   pelvis lateral offset from the stance foot, + = toward the swing side (mm)
  xcom  extrapolated CoM offset  y + vy / w0 (w0 = sqrt(g / 0.2585 m)); > 0 means the body is falling toward the
        swing foot and cannot stay on the stance leg
  roll rate toward the swing side (deg/s), late swing (last third) and at touchdown
  swing time, touchdown foot speed
The pelvis stands in for the CoM in both, so the numbers compare like with like.
"""
import glob
import json
import os
import sys

import numpy as np

W0 = np.sqrt(9.81 / 0.2585)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def swings(t_dt, root, vy, roll_rate, feet, feet_vz, contact, yaw):
    """root/feet in world xy(z); returns a list of per-swing dicts. Arrays are [T] / [T,2] / [T,2,3]."""
    out = []
    T = len(contact)
    c, s = np.cos(yaw), np.sin(yaw)
    lat = lambda p, i: -s[i] * p[..., 0] + c[i] * p[..., 1]  # heading-frame lateral coordinate at steps i
    for sw in (0, 1):
        st = 1 - sw
        sign = 1.0 if sw == 0 else -1.0  # left foot swinging -> swing side is +y
        k = 1
        while k < T:
            if contact[k - 1, sw] and not contact[k, sw] and contact[k, st]:
                e = k
                while e < T and not contact[e, sw]:
                    e += 1
                if e >= T:
                    break
                ss = np.arange(k, e)
                ss = ss[contact[ss, st]]  # single support only
                if len(ss) >= 3 and (e - k) * t_dt < 1.2:
                    off = sign * (lat(root[ss], ss) - lat(feet[ss, st], ss))
                    xcom = off + sign * vy[ss] / W0
                    rr = -sign * roll_rate[ss]  # +roll about x tilts the body toward -y
                    late = ss[int(len(ss) * 2 / 3):]
                    out.append(dict(
                        t=(e - k) * t_dt, off=off.mean() * 1000, off_end=off[-1] * 1000,
                        xcom_mid=xcom[len(xcom) // 2] * 1000, xcom_end=xcom[-1] * 1000,
                        xcom_pos=(xcom > 0).mean(),
                        rr_late=(-sign * roll_rate[late]).mean(), rr_td=rr[-1],
                        td=-feet_vz[e - 1, sw] * 1000))
                k = e
            k += 1
    return out


def report(tag, sw):
    if not sw:
        print(f"{tag}: no swings"); return
    a = {k: np.array([d[k] for d in sw]) for k in sw[0]}
    med = lambda k: np.median(a[k])
    print(f"{tag}: n={len(sw)} swing {med('t'):.2f} s | touchdown {med('td'):.0f} mm/s | pelvis off stance "
          f"(+ = swing side) mean {med('off'):+.1f} end {med('off_end'):+.1f} mm | XCoM mid {med('xcom_mid'):+.1f} "
          f"end {med('xcom_end'):+.1f} mm, time with XCoM on swing side {med('xcom_pos')*100:.0f}% | roll rate to "
          f"swing side late {med('rr_late'):+.1f} at touchdown {med('rr_td'):+.1f} deg/s")
    if len(sw) > 10:
        r = np.corrcoef(a["xcom_mid"], a["td"])[0, 1]; r2 = np.corrcoef(a["xcom_mid"], a["t"])[0, 1]
        r3 = np.corrcoef(a["rr_late"], a["td"])[0, 1]
        print(f"   per-swing correlation: XCoM mid vs touchdown speed {r:+.2f}, vs swing time {r2:+.2f}; "
              f"late roll rate vs touchdown speed {r3:+.2f}")


# policy trace: forward segment (0), after 1 s, alive envs
z = np.load(sys.argv[1])
dt = float(z["dt"]); seg = z["seg"]
idx = np.nonzero(seg == 0)[0][int(1.0 / dt):]
pol = []
for e in range(z["root"].shape[1]):
    ok = idx[z["alive"][idx, e]]
    if len(ok) < 60:
        continue
    yaw = z["yaw"][ok, e]
    # heading-frame v_h / w_h are already rotated by yaw: v_h[...,1] lateral velocity, w_h[...,0] roll rate
    pol += swings(dt, z["root"][ok, e], z["v_h"][ok, e, 1], np.rad2deg(z["w_h"][ok, e, 0]), z["feet"][ok, e],
                  z["feet_v"][ok, e, :, 2], z["contact"][ok, e], yaw)
report("POLICY forward", pol)

# demo clips: forward commands
motion_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "source", "arti_d1", "arti_d1", "assets", "motions", "arti_d1")
demo = []
for path in glob.glob(os.path.join(motion_dir, "*.json")):
    m = json.load(open(path))
    g = m["gait_parameters"]
    if not (g["dx"] >= 0.035 and abs(g["dy"]) < 0.01 and abs(g["dth"]) < 0.02):
        continue
    off, size = m["frame_offset"][0], m["frame_size"][0]
    f = np.asarray(m["frames"], dtype=np.float64)
    F = lambda n: f[:, off[n]: off[n] + size[n]]
    valid = F("periodic_valid")[:, 0] > 0.5
    rq = F("root_orientation_quat")  # x y z w
    x, y, zq, w = rq.T
    yaw = np.arctan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq))
    c, s = np.cos(yaw), np.sin(yaw)
    v = F("world_linear_velocity"); vy = -s * v[:, 0] + c * v[:, 1]
    av = F("world_angular_velocity"); rr = np.rad2deg(c * av[:, 0] + s * av[:, 1])
    feet = np.stack((F("left_toe_position"), F("right_toe_position")), 1)
    fvz = np.stack((F("left_toe_linear_velocity")[:, 2], F("right_toe_linear_velocity")[:, 2]), 1)
    contact = F("foot_contacts") > 0.5
    sl = np.nonzero(valid)[0]
    demo += swings(m["frame_duration"], F("root_position")[sl], vy[sl], rr[sl], feet[sl], fvz[sl], contact[sl], yaw[sl])
report("DEMO   forward", demo)
