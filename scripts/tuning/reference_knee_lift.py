"""Reference knee lift per swing (knee = shin link origin, swing peak height - height just before lift-off), from URDF
forward kinematics of the clips (numpy only). Sets env cfg knee_lift_target.

    python scripts/tuning/reference_knee_lift.py [motion_dir] [max_clips]
"""
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF = os.path.join(ROOT, "source/arti_d1/arti_d1/assets/models/arti_d1/arti_d1.urdf")
motion_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "source/arti_d1/arti_d1/assets/motions/arti_d1")
max_clips = int(sys.argv[2]) if len(sys.argv) > 2 else 200

joints = {j.get("name"): j for j in ET.parse(URDF).getroot().findall("joint")}


def chain(link):
    names = []
    while link != "pelvis":
        j = [x for x in joints.values() if x.find("child").get("link") == link][0]
        names.append(j.get("name"))
        link = j.find("parent").get("link")
    return names[::-1]


def axang(a, t):
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def quat_xyzw_to_R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def knee_pos(side, q):
    """Shin link origin in the pelvis frame (all URDF joint rpy are 0)."""
    p, R = np.zeros(3), np.eye(3)
    for n in chain(f"{side}_shin"):
        j = joints[n]
        p = p + R @ np.array([float(v) for v in j.find("origin").get("xyz").split()])
        R = R @ axang(np.array([float(v) for v in j.find("axis").get("xyz").split()]), q[n])
    return p


files = sorted(glob.glob(os.path.join(motion_dir, "*.json")), key=lambda f: int(os.path.basename(f)[:-5]))[:max_clips]
lifts = {"left": [], "right": []}
for path in files:
    m = json.load(open(path))
    off, names = m["frame_offset"][0], m["joints"]
    f = np.asarray(m["frames"])
    if "periodic_valid" in off and not (f[:, off["periodic_valid"]] > 0.5).any():
        continue  # standing clip
    q = f[:, off["joints_positions"]:off["joints_positions"] + len(names)]
    for k, side in enumerate(("left", "right")):
        c = f[:, off["foot_contacts"] + k] > 0.5
        z = np.array([(f[i, 0:3] + quat_xyzw_to_R(f[i, 3:7]) @ knee_pos(side, dict(zip(names, q[i]))))[2]
                      for i in range(len(f))])
        land = np.nonzero(~c[:-1] & c[1:])[0] + 1
        for i0 in np.nonzero(c[:-1] & ~c[1:])[0] + 1:
            after = land[land > i0]
            if len(after) and f[i0, off["periodic_valid"]] > 0.5:
                lifts[side].append(z[i0:after[0]].max() - z[i0 - 1])
print(f"{len(files)} clips: knee lift left {np.mean(lifts['left']) * 1000:.1f} mm right {np.mean(lifts['right']) * 1000:.1f} mm")
