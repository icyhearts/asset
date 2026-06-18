#!/usr/bin/env python3
"""Compare SGLang and vLLM saved all-reduce outputs bitwise."""

import argparse

import torch
from safetensors.torch import load_file


SGLANG_OUTPUT = "/data/like/temp/sglang.safetensors"
VLLM_OUTPUT = "/data/like/temp/vllm.safetensors"
KEY = "rank_0_all_reduce"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang", default=SGLANG_OUTPUT)
    parser.add_argument("--vllm", default=VLLM_OUTPUT)
    parser.add_argument("--key", default=KEY)
    return parser.parse_args()


def _as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().view(torch.uint8)


def main() -> None:
    args = parse_args()
    sglang = load_file(args.sglang)[args.key]
    vllm = load_file(args.vllm)[args.key]

    same_shape = tuple(sglang.shape) == tuple(vllm.shape)
    same_dtype = sglang.dtype == vllm.dtype
    bitwise_equal = same_shape and same_dtype and torch.equal(
        _as_bytes(sglang), _as_bytes(vllm)
    )

    print(f"sglang: shape={tuple(sglang.shape)} dtype={sglang.dtype}")
    print(f"vllm:   shape={tuple(vllm.shape)} dtype={vllm.dtype}")
    print(f"bitwise_equal={bitwise_equal}")

    if bitwise_equal:
        return

    if same_shape:
        numeric_diff = (sglang.to(torch.float32) - vllm.to(torch.float32)).abs()
        mismatch = _as_bytes(sglang) != _as_bytes(vllm)
        print(f"mismatched_bytes={int(mismatch.sum().item())}")
        print(f"max_abs_diff={float(numeric_diff.max().item())}")
        value_mismatch = sglang != vllm
        if bool(value_mismatch.any().item()):
            first_flat = int(value_mismatch.flatten().nonzero()[0].item())
            index = tuple(torch.unravel_index(torch.tensor(first_flat), sglang.shape))
            index = tuple(int(i.item()) for i in index)
            print(
                "first_value_mismatch="
                f"index={index} sglang={sglang[index].item()} vllm={vllm[index].item()}"
            )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
