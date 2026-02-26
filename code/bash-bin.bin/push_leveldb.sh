#!/bin/bash
set -x
cd /home/muice/packageLinux/learnPurpose/leveldb/

ZERO=$(git diff >a.txt; wc -c a.txt  | awk '{print $1}')

if [[ "$ZERO" != "0" ]]; then
  bash gitadd.sh
  git commit -m "commited on `date +%Y-%m-%d___%H-%M-%S`"
  git push -f pc_host pchost-like 
  git push -f workpc pchost-like
fi
