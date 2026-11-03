"""Trailing-foot drag around touchdown (forward walking) from an eval_gait --dump trace:
python drag.py <trace.npz> [...]
After each touchdown of foot A, follow the other foot B for 0.4 s while B's sole centre is near the ground (< 10 mm):
  slide   horizontal distance B travels while near the ground before it is clearly lifted (> 10 mm) (mm)
  speed   mean horizontal speed of B while near the ground and moving (> 30 mm/s) (mm/s)
  late    time from A's touchdown until B is clearly lifted (s)
Demo reference: double support ~0.08 s per step (clock), B lifts straight up (no slide).
"""
import sys
import numpy as np
for path in sys.argv[1:]:
    z = np.load(path); dt = float(z["dt"]); idx = np.nonzero(z["seg"] == 0)[0][int(1.0 / dt):]
    rows = []
    for e in range(z["root"].shape[1]):
        ok = idx[z["alive"][idx, e]]
        c = z["contact"][ok, e]; fz = z["feet"][ok, e, :, 2]; fv = z["feet_v"][ok, e]
        W = int(0.4 / dt)
        for a in (0, 1):
            b = 1 - a
            for k in range(1, len(ok) - W):
                if c[k, a] and not c[k - 1, a] and c[k, b]:
                    seg = np.arange(k, k + W)
                    up = np.nonzero(fz[seg, b] > 0.010)[0]
                    end = up[0] if len(up) else W
                    near = seg[:end]
                    hv = np.linalg.norm(fv[near, b, :2], axis=-1)
                    rows.append((hv.sum() * dt * 1000, hv[hv > 0.03].mean() * 1000 if (hv > 0.03).any() else 0.0, end * dt))
    a = np.array(rows)
    print(f"{path}: n={len(a)} | slide before lift median {np.median(a[:,0]):.1f} p90 {np.percentile(a[:,0],90):.1f} mm "
          f"| speed while near ground {np.median(a[:,1]):.0f} mm/s | lift delay after other foot's touchdown median "
          f"{np.median(a[:,2]):.2f} s p90 {np.percentile(a[:,2],90):.2f}")
