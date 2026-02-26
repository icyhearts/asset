#!/bin/bash
# awk: https://stackoverflow.com/questions/9394408/pad-all-lines-with-spaces-to-a-fixed-width-in-vim-or-using-sed-awk-etc
# cut: https://unix.stackexchange.com/questions/446953/trim-lines-to-a-specific-length
set -x
LENGTH=""
IFNAME=""
OFNAME=""
INTERFILE=/tmp/xxyy9527.txt

help() {
  echo "usage xx.sh -l length -i inpupt -o output"
}

while getopts "l:i:o:" OPTION; do
  case $OPTION in
  l)
     LENGTH=$OPTARG
     ;;
  i)
     IFNAME=$OPTARG
     ;;
  o)
     OFNAME=$OPTARG
     ;;
  *)
    help
    exit 2
    ;;
  esac
done
if [[ -z "$LENGTH" || -z "$IFNAME" || -z "$OFNAME" ]]; then
    help
    exit 2
fi

echo "pad $IFNAME to $OFNAME with lenght=$LENGTH"

# It is impossible to pass LENGTH to awk, because awk use '' instead of "", '' will not parse variable,
# So, we first use awk get a file with 200 width
# then use cut to cut it to LENGTH
awk '{printf "%-200s\n", $0}' ${IFNAME}  >${INTERFILE}
cut -c -${LENGTH} ${INTERFILE} > ${OFNAME}

set +x
