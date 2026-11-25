#!/usr/bin/env python3
"""Run a Fast-ACVNet+ .nb on a stereo pair on the Radxa and save a disparity picture (and optional depth).

  python3 runtime/run_stereo.py --model models/fastacv_plus_md64_k4_256x192_int8pcq.nb --size 256x192 \
          --pair frame.jpg                      # side-by-side image (left | right)
  python3 runtime/run_stereo.py --model ... --size 256x192 --left l.png --right r.png --out result

Needs: OpenCV + numpy (python3) and the built `nb_bench` (runtime/build_on_board.sh).
Images must be rectified. Writes <out>_disp.npy (float32 disparity, pixels of the model input),
<out>_vis.png (left | colorised disparity) and, if --fx/--baseline are given, prints depth statistics.
"""
import argparse, os, re, subprocess, sys, tempfile
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help=".nb file")
ap.add_argument("--size", default=None, help="model input WxH (default: parsed from the file name, e.g. 256x192)")
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--pair", help="side-by-side stereo image (left | right)")
g.add_argument("--left", help="left image (use with --right)")
ap.add_argument("--right")
ap.add_argument("--out", default="result", help="output prefix")
ap.add_argument("--nb-bench", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "nb_bench"))
ap.add_argument("--awnn", default=os.environ.get("AWNN", os.path.expanduser("~/awnn_inference")), help="dir with install/lib (VIPLite libs)")
ap.add_argument("--runs", type=int, default=10, help="timing runs")
ap.add_argument("--fx", type=float, help="focal length in pixels of the ORIGINAL image width (rectified)")
ap.add_argument("--baseline", type=float, help="stereo baseline in metres")
a = ap.parse_args()

m = re.search(r"(\d+)x(\d+)", a.size or os.path.basename(a.model))
if not m:
    sys.exit("cannot determine model size, pass --size WxH")
W, H = int(m.group(1)), int(m.group(2))

if a.pair:
    im = cv2.imread(a.pair)
    if im is None:
        sys.exit(f"cannot read {a.pair}")
    L, R = im[:, : im.shape[1] // 2], im[:, im.shape[1] // 2:]
else:
    if not a.right:
        sys.exit("--left needs --right")
    L, R = cv2.imread(a.left), cv2.imread(a.right)
    if L is None or R is None:
        sys.exit("cannot read left/right image")
orig_w = L.shape[1]

with tempfile.TemporaryDirectory() as td:
    paths = []
    for name, img in (("l", L), ("r", R)):
        rgb = cv2.cvtColor(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)   # raw uint8 RGB HWC
        p = os.path.join(td, name + ".bin"); rgb.tofile(p); paths.append(p)
    outbin = os.path.join(td, "disp.bin")
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(a.awnn, "install/lib") + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    r = subprocess.run([a.nb_bench, "-m", a.model, "-n", str(a.runs), "-i0", paths[0], "-i1", paths[1], "-o", outbin],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0 or not os.path.exists(outbin):
        sys.exit("nb_bench failed:\n" + r.stdout[-800:] + r.stderr[-800:])
    for line in r.stdout.splitlines():
        if line.startswith("run "):
            print("NPU", line.strip())
    disp = np.fromfile(outbin, np.float32).reshape(H, W)

np.save(a.out + "_disp.npy", disp)
vmax = float(max(np.percentile(disp, 99), 1.0))
col = cv2.applyColorMap(np.clip(disp / vmax * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
left_small = cv2.resize(L, (W, H), interpolation=cv2.INTER_AREA)
cv2.imwrite(a.out + "_vis.png", np.hstack([left_small, col]))
print(f"disparity: min {disp.min():.2f}  median {np.median(disp):.2f}  max {disp.max():.2f} px (model input width {W}); colour scale 0-{vmax:.1f}")
if a.fx and a.baseline:
    z = a.fx * (W / orig_w) * a.baseline / np.maximum(disp, 0.05)
    c = z[H // 3: 2 * H // 3, W // 3: 2 * W // 3]
    print(f"depth: centre-region median {np.median(c):.2f} m, 10th percentile {np.percentile(c, 10):.2f} m, whole-image min {z.min():.2f} m")
print("wrote", a.out + "_disp.npy", a.out + "_vis.png")
