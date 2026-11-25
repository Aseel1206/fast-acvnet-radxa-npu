#!/bin/bash
# On the Radxa. Builds ./nb_run against your copy of the Allwinner VIPLite SDK.
# From the AWNPU model zoo (see README):
#   NPU_INC=<zoo>/common/npuruntime/include  NPU_LIB=<zoo>/common/npuruntime/lib_linux_aarch64/A733  ./build_on_board.sh
set -e
: "${NPU_INC:?set NPU_INC to the directory containing vip_lite.h}" "${NPU_LIB:?set NPU_LIB to the directory containing libNBGlinker.so and libVIPhal.so}"
g++ -O2 -std=c++14 -pthread "$(dirname "$0")/nb_run.cpp" -I "$NPU_INC" -L "$NPU_LIB" -lNBGlinker -lVIPhal -Wl,--disable-new-dtags -Wl,-rpath,"$NPU_LIB" -o "$(dirname "$0")/nb_run"
echo "built $(dirname "$0")/nb_run"
