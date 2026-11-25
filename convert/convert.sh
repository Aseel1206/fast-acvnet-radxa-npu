#!/bin/bash
# Host-side driver: ONNX export -> onnxsim -> patch -> (docker) Acuity import / quantize / NBG export.
#
#   convert/convert.sh --upstream ~/Fast-ACVNet --ckpt sceneflow.ckpt --calib-images ~/stereo_frames \
#       --mount ~/npu_work --size 256x192 --maxdisp 64 --q "pcq i16"
#
# --mount is a host directory that is mounted at /workspace in the container (e.g.
#   docker run --ipc=host -itd -v ~/npu_work:/workspace --name allwinner ubuntu-npu:v2.0.10.1 /bin/bash).
# This repo must live under that directory, or be copied into it (the container runs convert/*.sh).
# Host python needs: torch onnx onnxsim onnxruntime opencv-python numpy.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SIZE=256x192; MAXDISP=64; TOPK=4; Q="pcq"; CONTAINER=allwinner_v2; SG=0
while [ $# -gt 0 ]; do case $1 in
  --upstream) UP=$2; shift 2;; --ckpt) CKPT=$2; shift 2;; --calib-images) IMGS=$2; shift 2;;
  --mount) MOUNT=$(realpath "$2"); shift 2;; --size) SIZE=$2; shift 2;; --maxdisp) MAXDISP=$2; shift 2;;
  --topk) TOPK=$2; shift 2;; --q) Q=$2; shift 2;; --container) CONTAINER=$2; shift 2;;
  --sg-docker) SG=1; shift;; *) echo "unknown arg $1"; exit 1;; esac; done
: "${UP:?--upstream}" "${CKPT:?--ckpt}" "${IMGS:?--calib-images}" "${MOUNT:?--mount}"
NAME=md${MAXDISP}_k${TOPK}_${SIZE}
WORK=$MOUNT/work_$NAME; mkdir -p "$WORK"
HERE_C=/workspace${HERE#$MOUNT}                      # this repo's convert/ dir as seen inside the container
case "$HERE" in $MOUNT/*) ;; *) echo "this repo must be inside --mount ($MOUNT)"; exit 1;; esac

echo "== 1/5 export ONNX";            python3 "$HERE/export_npu_onnx.py" --upstream "$UP" --ckpt "$CKPT" --maxdisp $MAXDISP --topk $TOPK --size $SIZE --out "$WORK/fastacv_raw.onnx"
echo "== 2/5 onnxsim";                python3 -m onnxsim "$WORK/fastacv_raw.onnx" "$WORK/fastacv_sim.onnx" > /dev/null
echo "== 3/5 patch for Acuity";       python3 "$HERE/onnx_patch.py" "$WORK/fastacv_sim.onnx" "$WORK/fastacv.onnx"
echo "== 4/5 calibration set";        python3 "$HERE/make_calib.py" --images "$IMGS" --size $SIZE --out "$WORK/calib" --host-root "$MOUNT"
WORK_C=/workspace/work_$NAME
echo "== 5/5 Acuity (container $CONTAINER)"
dk() { if [ "$SG" = 1 ]; then sg docker -c "docker $*"; else docker "$@"; fi; }   # --sg-docker: user not (yet) in the docker group
dk start $CONTAINER > /dev/null 2>&1 || true
dk exec $CONTAINER bash $HERE_C/prep_meta.sh $WORK_C $WORK_C/calib
for q in $Q; do dk exec $CONTAINER bash $HERE_C/quantize_export.sh $WORK_C $q; done
for q in $Q; do
  [ -f "$WORK/out_${q}_nbg_unify/network_binary.nb" ] && cp "$WORK/out_${q}_nbg_unify/network_binary.nb" "$WORK/fastacv_plus_${NAME}_$q.nb" && echo "-> $WORK/fastacv_plus_${NAME}_$q.nb"
done
