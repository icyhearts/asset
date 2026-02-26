#!/bin/bash
PATTERN=""
FILE=""

while getopts "p:f:" OPTION; do
  case $OPTION in
  p)
     PATTERN=$OPTARG
     ;;
  f)
     FILE=$OPTARG
     ;;
  *)
    "echo usage xx.sh -p pattern -f file/dir"
    exit 2
    ;;
  esac
done
if [[ -z "$PATTERN" ]]; then
    echo "usage xx.sh -p pattern -f file/dir"
    exit 2
fi

if [[ -z "$FILE" ]]; then
    echo "usage xx.sh -p pattern -f file/dir"
    exit 2
fi

# CMT_DEBUG set -x
egrep --color -I "(make_unique|make_shared|new) *<? *${PATTERN} *>? *" -n -r ${FILE}
set +x
