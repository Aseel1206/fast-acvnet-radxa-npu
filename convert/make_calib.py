#!/usr/bin/env python3
"""Build a calibration set (left/right PNGs + dataset0/1.txt) from side-by-side stereo images (left | right).

  python convert/make_calib.py --images my_stereo_frames --size 256x192 --n 47 --out work/calib \
         --host-root /home/me/work --container-root /workspace

The dataset txt files contain container paths (host-root is replaced by container-root).
Use frames that look like what the model will see in practice (50 or so, varied scenes/lighting).
"""
import argparse, glob, os
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True, help="folder of side-by-side stereo images (jpg/png)")
ap.add_argument("--size", default="256x192")
ap.add_argument("--n", type=int, default=47)
ap.add_argument("--out", required=True)
ap.add_argument("--host-root", required=True, help="host directory that is mounted into the container")
ap.add_argument("--container-root", default="/workspace", help="where host-root is mounted in the container")
a = ap.parse_args()
W, H = map(int, a.size.lower().split("x"))
files = sorted(sum([glob.glob(os.path.join(a.images, e)) for e in ("*.jpg", "*.jpeg", "*.png")], []))
assert files, "no images found"
pick = [files[i] for i in np.linspace(0, len(files) - 1, min(a.n, len(files))).astype(int)]
os.makedirs(a.out, exist_ok=True)
l0, r0 = [], []
k = 0
for f in pick:
    im = cv2.imread(f)
    if im is None or im.shape[1] < 2 * 64:      # unreadable or not a side-by-side pair
        print("skipping", f)
        continue
    w = im.shape[1] // 2
    for side, img, lst in (("left", im[:, :w], l0), ("right", im[:, w:], r0)):
        p = os.path.abspath(os.path.join(a.out, f"{side}_{k:03d}.png"))
        cv2.imwrite(p, cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA))
        lst.append(p.replace(os.path.abspath(a.host_root), a.container_root, 1))
    k += 1
assert k > 0, "no usable images"
open(os.path.join(a.out, "dataset0.txt"), "w").write("\n".join(l0) + "\n")
open(os.path.join(a.out, "dataset1.txt"), "w").write("\n".join(r0) + "\n")
print(k, "pairs ->", a.out)
