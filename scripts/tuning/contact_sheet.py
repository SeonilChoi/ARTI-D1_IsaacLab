"""Frame grid of a play video for a quick visual check:
python contact_sheet.py <file.mp4> <out.png> [t0 t1 ...]   (seconds; default covers start-up and every play segment)

2026-11-02: the 920k A/B recheck called A better from numbers while its video fell at 0.5 s; the blank check alone
does not show that, so every recorded video gets a sheet that is actually looked at.
"""
import sys

import cv2
import imageio.v2 as iio
import numpy as np

path, out = sys.argv[1], sys.argv[2]
times = [float(t) for t in sys.argv[3:]] or [0, 0.5, 1, 1.5, 2, 3, 5, 8, 12, 17, 22, 27, 30, 33]
reader = iio.get_reader(path)
fps = reader.get_meta_data().get("fps", 60.0)
want = {int(round(t * fps)): t for t in times}
tiles = []
for i, frame in enumerate(reader):
    if i in want:
        tile = cv2.resize(np.asarray(frame), (426, 240))
        cv2.putText(tile, f"{i / fps:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        tiles.append(tile)
    if i > max(want):
        break
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
iio.imwrite(out, np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]))
print(f"saved {out} ({len([t for t in tiles if t.any()])} frames, fps {fps:g})")
