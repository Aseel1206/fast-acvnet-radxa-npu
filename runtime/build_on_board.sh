#!/bin/bash
# On the Radxa. Needs g++ and a VIPLite wrapper that provides awnn_inference/session.h (see README, "Runtime").
#   AWNN=~/awnn_inference ./build_on_board.sh
AWNN=${AWNN:-$HOME/awnn_inference}
g++ -O2 -std=c++14 -pthread nb_bench.cpp -I "$AWNN/include" -L "$AWNN/install/lib" \
    -lawnn_inference -lNBGlinker -lVIPhal -Wl,-rpath,"$AWNN/install/lib" -o nb_bench && echo built nb_bench
