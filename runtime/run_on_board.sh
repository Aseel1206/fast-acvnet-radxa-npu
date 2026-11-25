#!/bin/bash
# On the Radxa. Runs a .nb on every pair in <io_dir> (made by prep_inputs.py) and saves float32 disparity maps.
#   ./run_on_board.sh model.nb <io_dir> <tag> [runs]    ->  out/NN_<tag>.bin
cd "$(dirname "$0")"
NB=$1; IO=$2; TAG=$3; N=${4:-20}
mkdir -p out
for l in "$IO"/*_l.bin; do
  i=$(basename "$l" _l.bin)
  ./nb_run -m "$NB" -n "$N" -i0 "$l" -i1 "$IO/${i}_r.bin" -o "out/${i}_$TAG.bin" 2>&1 | grep -E "^run "
done
