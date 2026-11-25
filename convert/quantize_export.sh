#!/bin/bash
# Inside the container.  usage: quantize_export.sh <workdir> <pcq|u8|i16>
#   pcq = per-channel int8 (recommended: as fast as uint8, much more accurate), u8 = uint8, i16 = int16 (near-lossless, ~1.5x slower)
# Output: <workdir>/out_<q>_nbg_unify/network_binary.nb
source "$(dirname "$0")/env.sh"
WORK=$1; Q=$2
case $Q in
  pcq) QZ=perchannel_symmetric_affine; QT=int8 ;;
  u8)  QZ=asymmetric_affine;           QT=uint8 ;;
  i16) QZ=dynamic_fixed_point;         QT=int16 ;;
  *) echo "quantizer must be pcq|u8|i16"; exit 1 ;;
esac
cd "$WORK"
$PEG quantize --model fastacv.json --model-data fastacv.data --device CPU \
  --with-input-meta fastacv_inputmeta.yml --iterations 47 --rebuild \
  --model-quantize fastacv_$Q.quantize --quantizer $QZ --qtype $QT > quantize_$Q.log 2>&1
$PEG export ovxlib --model fastacv.json --model-data fastacv.data --model-quantize fastacv_$Q.quantize \
  --with-input-meta fastacv_inputmeta.yml --postprocess-file fastacv_postprocess_file.yml \
  --dtype quantized --pack-nbg-unify --optimize $NPU_TARGET \
  --viv-sdk $VIVANTE_SDK_DIR --output-path "$WORK/out_$Q/net" > export_$Q.log 2>&1
chmod -R a+rwX "$WORK"
if [ -f "$WORK/out_${Q}_nbg_unify/network_binary.nb" ]; then
  echo "OK  $WORK/out_${Q}_nbg_unify/network_binary.nb"
else
  echo "FAILED $Q - see $WORK/export_$Q.log"; sed 's/\x1b\[[0-9;]*m//g' export_$Q.log | grep -E "^E |Fatal" | head -5; exit 1
fi
