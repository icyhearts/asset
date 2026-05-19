#!/bin/bash

src_dir="$1"
dst_dir="$2"
suffix="$3"

if [ -z "$src_dir" ] || [ -z "$dst_dir" ]; then
    echo "Usage: $0 <src_dir> <dst_dir> [suffix]"
    exit 1
fi

mkdir -p "$dst_dir"

for f in "$src_dir"/*; do
    [ -f "$f" ] || continue
    if [ -z "$suffix" ] || [[ "$f" == *"$suffix" ]]; then
        ln -sf "$(realpath "$f")" "$dst_dir/$(basename "$f")"
    fi
done
