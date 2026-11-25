#!/bin/bash
# Inside the container.  usage: prep_meta.sh <workdir> <calib_dir>
#   <workdir> contains fastacv.onnx (patched) ;  <calib_dir> contains dataset0.txt / dataset1.txt (see make_calib.py)
# Imports the ONNX into Acuity and writes the input/post-process metadata:
#   - raw uint8 RGB in, ImageNet mean/std applied on the NPU (add_preproc_node)
#   - float32 output (add_postproc_node)
set -e
source "$(dirname "$0")/env.sh"
WORK=$1; CALIB=$2
cd "$WORK"
$PEG import onnx --model fastacv.onnx --output-model fastacv.json --output-data fastacv.data 2>&1 \
  | sed 's/\x1b\[[0-9;]*m//g' | grep -E "^(E|W) |Error\(|Traceback|rror:" | grep -v -E "cuda|libcu" || true
$PEG generate inputmeta --model fastacv.json --separated-database --input-meta-output fastacv_inputmeta.yml > /dev/null 2>&1
$PEG generate postprocess-file --model fastacv.json --postprocess-file-output fastacv_postprocess_file.yml > /dev/null 2>&1
CALIB="$CALIB" python3 - <<'PY'
import os, re
C = os.environ["CALIB"]
s = open("fastacv_inputmeta.yml").read()
s = s.replace("path: dataset0.txt", f"path: {C}/dataset0.txt").replace("path: dataset1.txt", f"path: {C}/dataset1.txt")
s = s.replace("reverse_channel: true", "reverse_channel: false")
# ImageNet normalisation: (x - mean) * scale, scale = 1/(255*std)
s = re.sub(r"mean:\n        - 0\n        - 0\n        - 0", "mean:\n        - 123.675\n        - 116.28\n        - 103.53", s)
s = re.sub(r"scale:\n        - 1.0\n        - 1.0\n        - 1.0", "scale:\n        - 0.017125\n        - 0.017507\n        - 0.017429", s)
s = s.replace("add_preproc_node: false", "add_preproc_node: true")
open("fastacv_inputmeta.yml", "w").write(s)
p = open("fastacv_postprocess_file.yml").read().replace("add_postproc_node: false", "add_postproc_node: true")
open("fastacv_postprocess_file.yml", "w").write(p)
print("metadata ready")
PY
chmod -R a+rwX "$WORK"
