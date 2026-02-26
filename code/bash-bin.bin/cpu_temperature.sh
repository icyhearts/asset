#!/bin/bash
# set -x
FNAME="$(date +%Y_%m_%d).txt"
DSTR=$(date +%Y_%m_%d___%H_%M_%S)
TSTR=$(sensors | grep Core | sort | awk -F"(" '{print $1}'  | xargs  | sed 's/+//g' | sed  's/\.0//g' | sed 's/°C/|/g')
OUT="${DSTR}|${TSTR}"
echo $OUT >>/mast/cpu_temperature/${FNAME}
