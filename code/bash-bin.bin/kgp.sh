#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <gpu_id_list> <protected_user>"
  echo "Example: $0 \"1,3,5\" like"
  exit 1
fi

GPU_LIST="$1"
PROTECTED_USER="$2"

IFS=',' read -ra GPU_IDS <<< "$GPU_LIST"

# --------------------------------------------------
# 1. GPU index -> GPU UUID 映射
# --------------------------------------------------
declare -A GPU_UUID_BY_INDEX

while IFS=',' read -r index uuid _; do
  index=$(echo "$index" | xargs)
  uuid=$(echo "$uuid" | xargs)
  GPU_UUID_BY_INDEX["$index"]="$uuid"
done < <(
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits
)

# --------------------------------------------------
# 2. GPU UUID -> PID 列表
# --------------------------------------------------
mapfile -t COMPUTE_APPS < <(
  nvidia-smi --query-compute-apps=gpu_uuid,pid \
             --format=csv,noheader,nounits
)

# --------------------------------------------------
# 3. 主逻辑
# --------------------------------------------------
for gpu_index in "${GPU_IDS[@]}"; do
  gpu_uuid="${GPU_UUID_BY_INDEX[$gpu_index]:-}"

  [[ -z "$gpu_uuid" ]] && continue

  for line in "${COMPUTE_APPS[@]}"; do
    app_uuid=$(echo "$line" | awk -F',' '{print $1}' | xargs)
    pid=$(echo "$line" | awk -F',' '{print $2}' | xargs)

    [[ "$app_uuid" != "$gpu_uuid" ]] && continue

    owner=$(ps -o user= -p "$pid" 2>/dev/null || true)
    [[ -z "$owner" ]] && continue

    if [[ "$owner" == "$PROTECTED_USER" ]]; then
      echo "[SKIP] GPU $gpu_index PID $pid owned by $owner"
      continue
    fi

    echo "[KILL] GPU $gpu_index PID $pid owned by $owner"
    kill -30 "$pid"
  done
done

