# Usage (motion_generator venv, has pinocchio + matplotlib): python scripts/duck/render_reference.py <urdf> <out.mp4> <ffmpeg> <clip.json>...
"""Stick-figure render of placo reference clips (URDF FK with pinocchio), front + side views, one row per clip."""
import json, sys, numpy as np, pinocchio as pin
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
urdf, out, ffmpeg = sys.argv[1], sys.argv[2], sys.argv[3]
clips = sys.argv[4:]
plt.rcParams["animation.ffmpeg_path"] = ffmpeg
m = pin.buildModelFromUrdf(urdf); d = m.createData()
chain = ["pelvis", "left_thigh", "left_foot_link"]
links = [f.name for f in m.frames if f.type == pin.FrameType.BODY]
def leg_frames(side):
    return ["pelvis"] + [f"{side}_{k}" for k in ("hip", "thigh", "shin", "foot_link")]
L, R = leg_frames("left"), leg_frames("right")
SOLE = {"left": np.array([0.0135599213, -0.024, -0.0593316689]), "right": np.array([0.0135599213, 0.024, -0.0593316765])}
data = []
for c in clips:
    j = json.load(open(c)); off = j["frame_offset"][0]; fr = np.array(j["frames"])
    names = j["joints"]; data.append((c, j, fr, off, names))
T = min(len(x[2]) for x in data); T = min(T, 600)
fig, axes = plt.subplots(len(data), 2, figsize=(10, 3.6 * len(data)))
axes = np.atleast_2d(axes)
def pts(fr_row, off, names):
    q = pin.neutral(m)
    for k, n in enumerate(names):
        jid = m.getJointId(n)
        if jid < m.njoints: q[m.joints[jid].idx_q] = fr_row[off["joints_positions"] + k]
    pin.framesForwardKinematics(m, d, q)
    quat = fr_row[off["root_orientation_quat"]:off["root_orientation_quat"] + 4]  # x y z w
    Rw = pin.Quaternion(quat[3], quat[0], quat[1], quat[2]).toRotationMatrix()
    p0 = fr_row[off["root_position"]:off["root_position"] + 3]
    P = lambda n: p0 + Rw @ d.oMf[m.getFrameId(n)].translation
    sole = lambda side: p0 + Rw @ (d.oMf[m.getFrameId(side + "_foot_link")].translation + d.oMf[m.getFrameId(side + "_foot_link")].rotation @ SOLE[side])
    Lp = np.array([P(n) for n in L] + [sole("left")]); Rp = np.array([P(n) for n in R] + [sole("right")])
    return [Lp, Rp, P("pelvis"), P("trunk")]
writer = FFMpegWriter(fps=30)
with writer.saving(fig, out, dpi=80):
    for t in range(0, T, 2):
        for r, (c, j, fr, off, names) in enumerate(data):
            Lp, Rp, pel, tr = pts(fr[t], off, names)
            g = j["gait_parameters"]
            for col, (a, b, lab) in enumerate(((1, 2, "front (y-z)"), (0, 2, "side (x-z)"))):
                ax = axes[r, col]; ax.cla()
                cx = pel[a]
                ax.plot(Lp[:, a], Lp[:, b], "o-", color="tab:blue", lw=2, ms=3)
                ax.plot(Rp[:, a], Rp[:, b], "o-", color="tab:red", lw=2, ms=3)
                ax.plot([pel[a], tr[a]], [pel[b], tr[b]], "k-", lw=4)
                ax.axhline(0, color="gray", lw=1)
                ax.set_xlim(cx - 0.25, cx + 0.25); ax.set_ylim(-0.02, 0.32); ax.set_aspect("equal")
                ax.set_title(f"dx {g['dx']:+.2f} dy {g['dy']:+.2f} dth {g['dth']:+.2f} | {lab} | t {t/60:.1f}s", fontsize=8)
                ax.tick_params(labelsize=6)
        writer.grab_frame()
print("saved", out)
