#!/usr/bin/env bash

set -o nounset
set -o pipefail

nvidia_smi_bin="${NVIDIA_SMI_BIN:-$(command -v nvidia-smi || true)}"

if [[ -z "${nvidia_smi_bin}" ]]; then
    printf 'nvidia-smi-user.sh: nvidia-smi was not found in PATH\n' >&2
    exit 127
fi

# Non-default modes can stream forever or change the output format. Preserve
# their native behavior instead of appending a misleading process snapshot.
if (( $# > 0 )); then
    exec "${nvidia_smi_bin}" "$@"
fi

nvidia_smi_output="$("${nvidia_smi_bin}")"
nvidia_smi_status=$?
printf '%s\n' "${nvidia_smi_output}"

if (( nvidia_smi_status != 0 )); then
    exit "${nvidia_smi_status}"
fi

# Find PID/type pairs in NVIDIA's process table. This works with both the
# legacy layout and the newer layout containing GI/CI columns.
mapfile -t gpu_processes < <(
    LC_ALL=C awk '
        /^\| Processes:/ {
            in_process_table = 1
            next
        }

        !in_process_table || !/^\|/ {
            next
        }

        {
            for (i = 2; i < NF; i++) {
                if ($i ~ /^[0-9]+$/ && $(i + 1) ~ /^(C|G|C\+G|M|M\+C)$/) {
                    print $2 "\t" $i
                    break
                }
            }
        }
    ' <<< "${nvidia_smi_output}"
)

lookup_username() {
    local pid="$1"
    local uid
    local passwd_entry

    if ! uid="$(ps -o uid= -p "${pid}" 2>/dev/null)"; then
        if [[ -d "/proc/${pid}" ]]; then
            printf '<unknown>'
        else
            printf '<exited>'
        fi
        return
    fi

    uid="${uid//[[:space:]]/}"
    if [[ -z "${uid}" ]]; then
        printf '<exited>'
        return
    fi

    if passwd_entry="$(getent passwd "${uid}" 2>/dev/null)" && [[ -n "${passwd_entry}" ]]; then
        printf '%s' "${passwd_entry%%:*}"
    else
        printf '%s' "${uid}"
    fi
}

printf '\nGPU process users:\n'
if (( ${#gpu_processes[@]} == 0 )); then
    printf '  (no running GPU processes)\n'
    exit 0
fi

printf '%-5s %-12s %s\n' 'GPU' 'PID' 'USERNAME'
printf '%-5s %-12s %s\n' '---' '---' '--------'
for gpu_process in "${gpu_processes[@]}"; do
    IFS=$'\t' read -r gpu pid <<< "${gpu_process}"
    username="$(lookup_username "${pid}")"
    printf '%-5s %-12s %s\n' "${gpu}" "${pid}" "${username}"
done
