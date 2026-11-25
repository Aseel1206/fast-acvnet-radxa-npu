#!/usr/bin/env python3
"""Compare NPU outputs with the float ONNX reference.
  python runtime/compare_eval.py --io io --out out --tag pcq --size 256x192
<out>/NN_<tag>.bin are the float32 disparity maps written by nb_run -o (one per input pair)."""
import argparse, glob, os
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--io", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--size", default="256x192")
a = ap.parse_args()
W, H = map(int, a.size.lower().split("x"))
rows = []
for f in sorted(glob.glob(f"{a.io}/*_ref.npy")):
    i = os.path.basename(f)[:2]
    ref = np.load(f); npu = np.fromfile(f"{a.out}/{i}_{a.tag}.bin", np.float32).reshape(H, W)
    d = np.abs(npu - ref)
    rows.append((d.mean(), np.median(d), (d < 0.5).mean() * 100, (d < 1).mean() * 100, np.corrcoef(npu.ravel(), ref.ravel())[0, 1]))
    print(f"{i}: EPE {rows[-1][0]:.3f}px  median {rows[-1][1]:.3f}  <0.5px {rows[-1][2]:.1f}%  <1px {rows[-1][3]:.1f}%  corr {rows[-1][4]:.3f}")
m = np.array(rows).mean(0)
print(f"MEAN  EPE {m[0]:.3f}px  median {m[1]:.3f}  <0.5px {m[2]:.1f}%  <1px {m[3]:.1f}%  corr {m[4]:.3f}")
