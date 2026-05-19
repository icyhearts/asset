set -x
#!/usr/bin/env bash

# Usage: ./kill_vllm.sh <username>

USER_NAME="$1"

if [[ -z "$USER_NAME" ]]; then
    echo "Usage: $0 <username>"
    exit 1
fi

declare -a PIDS=()

###############################################################################
# 1. Find processes containing "VLLM::" and owned by the user
###############################################################################
mapfile -t vllm_colon_pids < <(ps -u "$USER_NAME" -o pid= -o cmd= \
    | grep "VLLM::" | grep -v grep | awk '{print $1}')

PIDS+=("${vllm_colon_pids[@]}")

###############################################################################
# 2. Find processes containing "vllm serve" and owned by the user
###############################################################################
mapfile -t vllm_serve_pids < <(ps -u "$USER_NAME" -o pid= -o cmd= \
    | grep "vllm serve" | grep -v grep | awk '{print $1}')

PIDS+=("${vllm_serve_pids[@]}")

###############################################################################
# 3. For each vllm serve PID, collect its child processes recursively
###############################################################################
for pid in "${vllm_serve_pids[@]}"; do
    # Get all child PIDs recursively
    mapfile -t children < <(pgrep -P "$pid" -d ' ' -f || true)
    PIDS+=("${children[@]}")
done

###############################################################################
# 4. Remove duplicates
###############################################################################
unique_pids=($(printf "%s\n" "${PIDS[@]}" | sort -u))

###############################################################################
# 5. Kill all collected PIDs
###############################################################################
echo "Killing PIDs: ${unique_pids[*]}"

if [ "${#unique_pids[@]}" -gt "0" ]; then
    kill -9 "${unique_pids[@]}"
fi

echo "Done."
