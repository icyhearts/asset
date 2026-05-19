#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 src_dir dst_dir" >&2
}

if [[ $# -ne 2 ]]; then
    usage
    exit 2
fi

src_arg=$1
dst_arg=$2

if [[ ! -d "$src_arg" ]]; then
    echo "Error: src_dir is not a directory: $src_arg" >&2
    exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
    echo "Error: rsync is required" >&2
    exit 1
fi

if ! command -v realpath >/dev/null 2>&1; then
    echo "Error: realpath is required" >&2
    exit 1
fi

src_dir=$(realpath "$src_arg")
mkdir -p "$dst_arg"
dst_dir=$(realpath "$dst_arg")

if [[ "$src_dir" == "$dst_dir" ]]; then
    echo "Error: src_dir and dst_dir must be different" >&2
    exit 1
fi

case "$dst_dir"/ in
    "$src_dir"/*)
        echo "Error: dst_dir must not be inside src_dir: $dst_dir" >&2
        exit 1
        ;;
esac

# Copy all non-safetensors content in archive mode. rsync -a preserves
# symlinks as symlinks and does not follow them.
rsync -a --exclude='*.safetensors' "$src_dir"/ "$dst_dir"/

# Link model weight shards instead of copying them.
while IFS= read -r -d '' src_file; do
    rel_path=${src_file#"$src_dir"/}
    dst_file=$dst_dir/$rel_path
    dst_parent=$(dirname "$dst_file")

    mkdir -p "$dst_parent"
    rm -f "$dst_file"

    link_target=$(realpath --relative-to="$dst_parent" "$src_file")
    ln -s "$link_target" "$dst_file"
done < <(find "$src_dir" -type f -name '*.safetensors' -print0)
