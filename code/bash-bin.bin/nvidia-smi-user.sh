#!/usr/bin/env bash

set -o nounset
set -o pipefail

log_file="${1:-/tmp/.smi.txt}"

if (( $# > 1 )); then
    printf 'Usage: %s [log_file]\n' "$0" >&2
    exit 2
fi

nvidia_smi_bin="${NVIDIA_SMI_BIN:-$(command -v nvidia-smi || true)}"

if [[ -z "${nvidia_smi_bin}" ]]; then
    printf 'nvidia-smi-user.sh: nvidia-smi was not found in PATH\n' >&2
    exit 127
fi

python_bin="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"

if [[ -z "${python_bin}" ]]; then
    printf 'nvidia-smi-user.sh: python3 (or python) was not found in PATH\n' >&2
    exit 127
fi

log_parent=$(dirname -- "${log_file}")
if [[ ! -d "${log_parent}" ]] && ! mkdir -p -- "${log_parent}"; then
    printf 'nvidia-smi-user.sh: cannot create log directory %s\n' "${log_parent}" >&2
    exit 1
fi

# Capture the complete command output first. The file is then rewritten with
# the annotated process table and the /proc snapshot below.
"${nvidia_smi_bin}" >"${log_file}" 2>&1
nvidia_smi_status=$?

if (( nvidia_smi_status != 0 )); then
    cat -- "${log_file}"
    exit "${nvidia_smi_status}"
fi

"${python_bin}" - "${log_file}" <<'PY'
from __future__ import annotations

import datetime as _datetime
import os
import pwd
import re
import shlex
import sys
import time


LOG_FILE = sys.argv[1]
PID_TYPE_RE = re.compile(
    r"(?P<pid>[0-9]+)(?P<gap>\s+)"
    r"(?P<type>C\+G|M\+C|C|G|M)(?P<after>\s+)"
)


def split_line(line: str) -> tuple[str, str]:
    body = line.rstrip("\r\n")
    return body, line[len(body) :]


def is_border(body: str) -> bool:
    return body.startswith("+") and body.endswith("+")


def is_table_line(body: str) -> bool:
    return is_border(body) or (body.startswith("|") and body.endswith("|"))


def fill_character(body: str) -> str:
    if is_border(body):
        return "-"
    interior = body[1:-1]
    if interior and set(interior) <= {"="}:
        return "="
    return " "


def insert_columns(body: str, position: int, count: int) -> str:
    if count <= 0:
        return body
    position = min(max(position, 1), max(1, len(body) - 1))
    return body[:position] + fill_character(body) * count + body[position:]


def pad_table_line(body: str, width: int) -> str:
    if not is_table_line(body) or len(body) >= width:
        return body
    return insert_columns(body, len(body) - 1, width - len(body))


def read_proc_status(pid: int) -> dict[str, str] | None:
    values: dict[str, str] = {}
    try:
        with open(
            f"/proc/{pid}/status", encoding="utf-8", errors="replace"
        ) as stream:
            for line in stream:
                key, separator, value = line.partition(":")
                if separator:
                    values[key] = value.strip()
    except OSError:
        return None
    return values


def one_line(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def read_command_arguments(pid: int, fallback: str) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as stream:
            arguments = [
                argument.decode("utf-8", errors="replace")
                for argument in stream.read().split(b"\0")
                if argument
            ]
    except OSError:
        arguments = []

    if arguments:
        return one_line(shlex.join(arguments))
    return one_line(fallback or "<unavailable>")


def read_link(path: str, fallback: str) -> str:
    try:
        return one_line(os.readlink(path))
    except OSError:
        return fallback


def read_boot_time() -> float | None:
    try:
        with open("/proc/stat", encoding="ascii", errors="replace") as stream:
            for line in stream:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except (OSError, IndexError, ValueError):
        pass
    return None


def read_start_epoch(pid: int, boot_time: float | None) -> float | None:
    if boot_time is None:
        return None
    try:
        with open(
            f"/proc/{pid}/stat", encoding="utf-8", errors="replace"
        ) as stream:
            stat_line = stream.read()
        # The comm field can contain spaces and parentheses. The final ')'
        # before the state field is safer than a whitespace split.
        comm_end = stat_line.rfind(")")
        if comm_end < 0:
            return None
        fields_after_comm = stat_line[comm_end + 2 :].split()
        start_ticks = int(fields_after_comm[19])  # /proc stat field 22
        return boot_time + start_ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError, OverflowError):
        return None


def format_start_time(start_epoch: float | None) -> str:
    if start_epoch is None:
        return "<unavailable>"
    try:
        return _datetime.datetime.fromtimestamp(start_epoch).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S %z"
        )
    except (OSError, OverflowError, ValueError):
        return "<unavailable>"


def format_duration(start_epoch: float | None) -> str:
    if start_epoch is None:
        return "<unavailable>"
    elapsed = max(0, int(time.time() - start_epoch))
    days, remainder = divmod(elapsed, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def unavailable_process(state: str) -> dict[str, str]:
    return {
        "ppid": state,
        "uid": state,
        "username": state,
        "command": state,
        "arguments": state,
        "start_time": state,
        "duration": state,
        "cwd": state,
    }


def process_info(pid: int, boot_time: float | None) -> dict[str, str]:
    status = read_proc_status(pid)
    if status is None:
        return unavailable_process(
            "<unknown>" if os.path.isdir(f"/proc/{pid}") else "<exited>"
        )

    ppid = status.get("PPid", "<unknown>")
    uid_values = status.get("Uid", "").split()
    uid = uid_values[1] if len(uid_values) > 1 else (
        uid_values[0] if uid_values else "<unknown>"
    )
    try:
        username = pwd.getpwuid(int(uid)).pw_name
    except (KeyError, ValueError, OverflowError):
        username = uid

    comm = status.get("Name", "")
    start_epoch = read_start_epoch(pid, boot_time)
    executable = read_link(f"/proc/{pid}/exe", "<unavailable>")
    arguments = read_command_arguments(pid, comm)
    if executable == "<unavailable>" and arguments not in {
        "<unavailable>",
        one_line(comm),
    }:
        try:
            executable = shlex.split(arguments)[0]
        except (IndexError, ValueError):
            pass

    command = executable if executable != "<unavailable>" else one_line(
        comm or executable
    )
    return {
        "ppid": ppid,
        "uid": uid,
        "username": username,
        "command": command,
        "arguments": arguments,
        "start_time": format_start_time(start_epoch),
        "duration": format_duration(start_epoch),
        "cwd": read_link(f"/proc/{pid}/cwd", "<unavailable>"),
    }


def find_process_block(lines: list[str]) -> tuple[int | None, int]:
    start = None
    for index, line in enumerate(lines):
        body, _ = split_line(line)
        if re.search(r"\|\s*Processes:\s*", body):
            start = index
            break
    if start is None:
        return None, len(lines)

    end = len(lines)
    for index in range(start + 1, len(lines)):
        body, _ = split_line(lines[index])
        if is_border(body):
            end = index + 1
            break
    return start, end


def annotate_log() -> None:
    with open(
        LOG_FILE, "r", encoding="utf-8", errors="replace", newline=""
    ) as stream:
        lines = stream.read().splitlines(keepends=True)

    process_start, process_end = find_process_block(lines)
    rows: list[dict[str, int]] = []
    if process_start is not None:
        for index in range(process_start + 1, process_end):
            body, _ = split_line(lines[index])
            if not body.startswith("|"):
                continue
            match = PID_TYPE_RE.search(body)
            if match is None:
                continue
            rows.append(
                {
                    "index": index,
                    "pid": int(match.group("pid")),
                    "pid_end": match.end("pid"),
                    "pid_start": match.start("pid"),
                    "type_start": match.start("type"),
                }
            )

    pids: list[int] = []
    for row in rows:
        if row["pid"] not in pids:
            pids.append(row["pid"])

    boot_time = read_boot_time()
    details = {pid: process_info(pid, boot_time) for pid in pids}

    shift = 0
    if process_start is not None and rows:
        old_type_start = min(row["type_start"] for row in rows)
        new_type_start = max(
            old_type_start,
            *(
                row["pid_start"]
                + len(f"{row['pid']},{details[row['pid']]['username']}")
                + row["type_start"]
                - row["pid_end"]
                for row in rows
            ),
        )
        shift = new_type_start - old_type_start

        header_insertion = old_type_start
        for index in range(process_start + 1, process_end):
            body, _ = split_line(lines[index])
            header_match = re.search(r"\bType\b", body)
            if header_match:
                header_insertion = header_match.start()
                break

        rows_by_index = {row["index"]: row for row in rows}
        for index in range(process_start + 1, process_end):
            body, ending = split_line(lines[index])
            row = rows_by_index.get(index)
            if row is None:
                lines[index] = insert_columns(body, header_insertion, shift) + ending
                continue

            display_pid = f"{row['pid']},{details[row['pid']]['username']}"
            gap = max(1, new_type_start - row["pid_start"] - len(display_pid))
            lines[index] = (
                body[: row["pid_start"]]
                + display_pid
                + " " * gap
                + body[row["type_start"] :]
                + ending
            )

    original_table_width = max(
        (len(split_line(line)[0]) for line in lines if is_table_line(split_line(line)[0])),
        default=0,
    )
    target_table_width = original_table_width
    if shift:
        # Process rows and headers have already grown. Add the same columns to
        # the rest of nvidia-smi's box so every right border remains aligned.
        unmodified_widths = [
            len(split_line(line)[0])
            for index, line in enumerate(lines)
            if is_table_line(split_line(line)[0])
            and not (
                process_start is not None and process_start < index < process_end
            )
        ]
        if unmodified_widths:
            target_table_width = max(target_table_width, max(unmodified_widths) + shift)

    for index, line in enumerate(lines):
        body, ending = split_line(line)
        lines[index] = pad_table_line(body, target_table_width) + ending

    if lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += "\n"
    for pid in pids:
        info = details[pid]
        lines.append(
            ",".join(
                [
                    str(pid),
                    info["ppid"],
                    info["uid"],
                    info["username"],
                    info["command"],
                    info["arguments"],
                    info["start_time"],
                    info["duration"],
                    info["cwd"],
                ]
            )
            + "\n"
        )

    with open(LOG_FILE, "w", encoding="utf-8", newline="") as stream:
        stream.writelines(lines)


annotate_log()
PY
processing_status=$?

cat -- "${log_file}"
exit "${processing_status}"
