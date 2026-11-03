"""Landing quality per touchdown (forward walking) from an eval_gait --dump trace:
python landing.py <trace.npz> [...]
  bounce   foot lifts again (> 1 cm... sole contact lost) within 0.25 s of touchdown
  short    contact lasts < 0.15 s
  roll     peak |pelvis roll rate| in the 0.2 s after touchdown (deg/s)
  slip     mean horizontal foot speed in the 0.1 s after touchdown (mm/s)
  impact   foot vertical speed just before touchdown (mm/s)
"""
import sys
import numpy as np
for path in sys.argv[1:]:
    z = np.load(path); dt = float(z["dt"]); idx = np.nonzero(z["seg"] == 0)[0][int(1.0 / dt):]
    rows = []
    for e in range(z["root"].shape[1]):
        ok = idx[z["alive"][idx, e]]
        c = z["contact"][ok, e]; fv = z["feet_v"][ok, e]; rr = np.rad2deg(z["w_h"][ok, e, 0])
        T = len(ok)
        for f in (0, 1):
            for k in range(1, T - int(0.3 / dt)):
                if c[k, f] and not c[k - 1, f]:
                    w = int(0.25 / dt); nxt = np.nonzero(~c[k:k + w, f])[0]
                    first_off = nxt[0] if len(nxt) else w
                    rows.append((first_off < w, first_off * dt < 0.15, np.abs(rr[k:k + int(0.2 / dt)]).max(),
                                 np.linalg.norm(fv[k:k + int(0.1 / dt), f, :2], axis=-1).mean() * 1000, -fv[k - 1, f, 2] * 1000))
    a = np.array(rows, dtype=float)
    lifts = len(a) / (len(idx) * dt * z["root"].shape[1])
    print(f"{path}: touchdowns {len(a)} ({lifts:.2f}/s per env, 2 feet; clock 1.23/s) | bounce<0.25s {a[:,0].mean()*100:.0f}% "
          f"| contact<0.15s {a[:,1].mean()*100:.0f}% | roll-rate peak after td median {np.median(a[:,2]):.0f} p90 {np.percentile(a[:,2],90):.0f} deg/s "
          f"| slip after td {np.median(a[:,3]):.0f} mm/s | impact median {np.median(a[:,4]):.0f} p90 {np.percentile(a[:,4],90):.0f} mm/s")
