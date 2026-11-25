#!/usr/bin/env python3
"""Prepare raw NPU inputs and float-ONNX reference disparities from side-by-side stereo images (left | right).

  python runtime/prep_inputs.py --images stereo_frames --onnx work/fastacv.onnx --size 256x192 --out io

Writes io/NN_l.bin, io/NN_r.bin (uint8 RGB, HWC: exactly what the .nb expects), io/NN_ref.npy (float ONNX output).
The ONNX reference uses ImageNet normalisation in float; the .nb does the same normalisation on the NPU.
"""
import argparse, glob, os
import cv2, numpy as np, onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True); ap.add_argument("--onnx", required=True)
ap.add_argument("--size", default="256x192"); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=9)
a = ap.parse_args()
W, H = map(int, a.size.lower().split("x"))
os.makedirs(a.out, exist_ok=True)
sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
files = sorted(sum([glob.glob(os.path.join(a.images, e)) for e in ("*.jpg", "*.jpeg", "*.png")], []))[: a.n]
for i, f in enumerate(files):
    im = cv2.imread(f); w = im.shape[1] // 2
    rgb = [cv2.cvtColor(cv2.resize(x, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for x in (im[:, :w], im[:, w:])]
    for side, x in zip("lr", rgb):
        x.tofile(f"{a.out}/{i:02d}_{side}.bin")
    f32 = [(((x.astype(np.float32) / 255) - mean) / std).transpose(2, 0, 1)[None].astype(np.float32) for x in rgb]
    np.save(f"{a.out}/{i:02d}_ref.npy", sess.run(None, {"left_image": f32[0], "right_image": f32[1]})[0].squeeze())
print(len(files), "pairs written to", a.out)
