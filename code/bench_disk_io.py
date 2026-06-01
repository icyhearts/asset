#!/usr/bin/env python3
"""Benchmark disk IO: read first byte of every non-empty file under a directory."""

import os
import sys
import time

sum_result = 0


def traverse_and_read(root_dir: str) -> None:
    global sum_result
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                fsize = os.path.getsize(fpath)
            except OSError:
                continue
            if fsize > 0:
                try:
                    with open(fpath, "rb") as f:
                        first_byte = f.read(1)
                    sum_result += int.from_bytes(first_byte, byteorder="little")
                except OSError:
                    continue


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory>", file=sys.stderr)
        sys.exit(1)

    root_dir = sys.argv[1]
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()
    traverse_and_read(root_dir)
    elapsed = time.perf_counter() - t0

    print(f"{elapsed:.6f}")
    print(sum_result)


if __name__ == "__main__":
    main()
