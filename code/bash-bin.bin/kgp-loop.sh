#!/usr/bin/env bash
set -euo pipefail
#set -x
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <gpu_id_list> <protected_user>"
  echo "Example: $0 \"1,3,5\" like"
  exit 1
fi

while true;
do
  /bin/bash /share_data/users/like/bash-bin/bin/kgp.sh $1 $2
  sleep 1
done
