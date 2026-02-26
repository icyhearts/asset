#!/bin/bash
# set -x
FNAME="$(date +%Y_%m_%d).txt"
DSTR=$(date +%Y_%m_%d___%H_%M_%S)
TSTR=$(nvidia-smi | egrep  "\/[ 0-9]+W" | sed -r 's/ +\/ +/\//g' | sed 's/|//g' | awk '{print $2,$4,$5,$6}' | xargs | sed 's/% /%|/g')
OUT="${DSTR}|${TSTR}"
echo $OUT >>/mast/gpu_temperature/${FNAME}
