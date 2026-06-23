#!/usr/bin/env python3
"""Print sha256 values for files with a chosen suffix in DeepSeek-V4-Flash.

This reads Hugging Face Hub repository metadata only. It does not download the
actual model files.
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "deepseek-ai/DeepSeek-V4-Flash"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Get sha256 values for files matching a filename suffix from Hugging "
      "Face Hub metadata without downloading the files."
    )
  )
  parser.add_argument(
    "--repo-id",
    default=DEFAULT_REPO_ID,
    help=f"Hugging Face model repo id. Default: {DEFAULT_REPO_ID}",
  )
  parser.add_argument(
    "--revision",
    default=None,
    help="Optional repo revision, branch, tag, or commit hash.",
  )
  parser.add_argument(
    "--token",
    default=None,
    help="Optional Hugging Face token. If omitted, cached login token may be used.",
  )
  parser.add_argument(
    "--timeout",
    type=float,
    default=30.0,
    help="Hub API timeout in seconds. Default: 30.",
  )
  parser.add_argument(
    "--file-suffix",
    required=True,
    help=(
      "Only print files whose repo filename ends with this suffix. "
      "Example: .safetensors"
    ),
  )
  parser.add_argument(
    "--with-filenames",
    action="store_true",
    help="Print '<sha256>  <filename>' instead of only sha256.",
  )
  args = parser.parse_args()
  if not args.file_suffix:
    parser.error("--file-suffix must not be empty")
  return args


def iter_file_sha256_by_suffix(
  repo_id: str,
  revision: str | None,
  token: str | None,
  timeout: float,
  file_suffix: str,
):
  api = HfApi(token=token)
  info = api.model_info(
    repo_id,
    revision=revision,
    files_metadata=True,
    timeout=timeout,
    token=token,
  )

  siblings = sorted(info.siblings or [], key=lambda sibling: sibling.rfilename)
  for sibling in siblings:
    filename = sibling.rfilename
    if not filename.endswith(file_suffix):
      continue

    lfs = sibling.lfs
    if lfs is None or not lfs.sha256:
      continue

    yield filename, lfs.sha256


def main() -> int:
  args = parse_args()

  count = 0
  for filename, sha256 in iter_file_sha256_by_suffix(
    args.repo_id,
    args.revision,
    args.token,
    args.timeout,
    args.file_suffix,
  ):
    count += 1
    if args.with_filenames:
      print(f"{sha256}  {filename}")
    else:
      print(sha256)

  if count == 0:
    print(
      f"No sha256 metadata found for files ending with "
      f"{args.file_suffix!r} in {args.repo_id}",
      file=sys.stderr,
    )
    return 0

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
