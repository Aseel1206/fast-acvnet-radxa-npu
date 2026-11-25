# Source this INSIDE the Allwinner NPU docker container (ubuntu-npu:v2.0.x).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VIVANTE_SDK_DIR=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator
export LD_LIBRARY_PATH=$VIVANTE_SDK_DIR/lib:$VIVANTE_SDK_DIR/lib/x64_linux:/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/common/lib:$LD_LIBRARY_PATH
# pegasus_patched.py stubs the missing `torch` module (the container has none) and then runs Acuity's pegasus.py
export PEG="python3 $HERE/pegasus_patched.py"
# A733 (Radxa Cubie A7A) NPU target. Other targets give empty .nb files or "Fatal model generation error".
export NPU_TARGET=VIP9000NANODI_PLUS_PID0X1000003B
