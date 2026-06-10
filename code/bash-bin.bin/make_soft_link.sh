#!/bin/bash

# Create symbolic links for files in a source directory.
#
# Usage:
#   make_soft_link.sh <src_dir> <dst_dir> [suffix]
#
# Arguments:
#   src_dir  Directory containing files to link from.
#   dst_dir  Directory where symlinks will be created. It is created if missing.
#   suffix   Optional filter. When set, only files whose full path ends with
#            this suffix are linked, for example ".so" or ".py".
src_dir="$1"
dst_dir="$2"
suffix="$3"

# Require source and destination directories. The suffix is optional.
if [ -z "$src_dir" ] || [ -z "$dst_dir" ]; then
    echo "Usage: $0 <src_dir> <dst_dir> [suffix]"
    exit 1
fi

# Ensure the destination directory exists before creating links.
mkdir -p "$dst_dir"

# Iterate over direct children of src_dir. Only regular files are processed;
# subdirectories and other file types are skipped.
for f in "$src_dir"/*; do
    [ -f "$f" ] || continue
    # If suffix is empty, link every regular file. Otherwise link only files
    # whose path ends with the requested suffix.
    if [ -z "$suffix" ] || [[ "$f" == *"$suffix" ]]; then
        # Create or replace a symlink in dst_dir using the same basename.
        # realpath makes the symlink point to an absolute source path.
        ln -sf "$(realpath "$f")" "$dst_dir/$(basename "$f")"
    fi
done
