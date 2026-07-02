#!/usr/bin/env python3
"""Test SIMO downcast/upcast kernels for MX-format quantization.

Reads a JSON quant config, extracts the weight spec, and measures
the reconstruction quality (cosine similarity, L2 relative norm) of a
quantize-dequantize round-trip on a random bf16 tensor.

Usage:
    python test_simo_downcast_upcast.py
    python test_simo_downcast_upcast.py -c path/to/config.json
    python test_simo_downcast_upcast.py -s 2048
    python test_simo_downcast_upcast.py -m 4096 -k 14336
"""

import argparse
import json
import os
import sys
import torch


# ---- path setup ----

_conda_env = os.getenv("CONDA_PREFIX", "/data/like/miniconda3/envs/simo_sglang")
_simo_src = os.path.join(_conda_env, "lib", "python3.12", "site-packages")
if os.path.isdir(_simo_src) and _simo_src not in sys.path:
    sys.path.insert(0, _simo_src)

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from simo.extensions.sglang_simo.quantization.quantization import (
    parse_quantize_spec,
    get_downcast_kernel,
    get_upcast_kernel,
)

DEFAULT_CONFIG = os.path.join(
    _repo_root,
    "simo/extensions/sglang_simo/example/simo_quantization_config",
    "online_quantization/quant_config_w6a6_mxfp.json",
)


# ---- helpers ----

def _load_weight_spec(config_path: str) -> dict:
    with open(config_path) as f:
        full = json.load(f)

    config = full.get("quantization_config", full)
    module_configs = config["module_configs"]
    if not module_configs:
        raise ValueError("module_configs is empty")

    return module_configs[0]["weight"]


def _parse_shape(args: argparse.Namespace) -> tuple[int, int]:
    """Return (M, K) from CLI args.  -s sets both; -m/-k override individually."""
    if args.m is not None and args.k is not None:
        return args.m, args.k
    size = args.size
    return size, size


def _bytes(t: torch.Tensor) -> int:
    return t.element_size() * t.numel()


# ---- main ----

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test SIMO downcast/upcast kernel reconstruction quality"
    )
    parser.add_argument(
        "-c", "--config",
        default=DEFAULT_CONFIG,
        help="Path to the quant config JSON (default: %%🪝(default)s)",
    )
    parser.add_argument(
        "-s", "--size",
        type=int, default=4096,
        help="Square matrix size M=K (default: 4096)",
    )
    parser.add_argument(
        "-m", "--rows",
        type=int, default=None,
        dest="m",
        help="Matrix rows (overrides -s for M)",
    )
    parser.add_argument(
        "-k", "--cols",
        type=int, default=None,
        dest="k",
        help="Matrix columns (overrides -s for K)",
    )

    args = parser.parse_args()
    M, K = _parse_shape(args)

    print(f"Config:  {args.config}")
    print(f"Matrix:  {M} x {K}")

    # 1. Parse quant spec
    spec_dict = _load_weight_spec(args.config)
    spec = parse_quantize_spec(spec_dict)
    print(f"Spec:    {spec}")

    # 2. Get kernels
    downcast = get_downcast_kernel(spec)
    upcast = get_upcast_kernel(spec)

    # 3. Create random bf16 tensor
    X = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    print(f"\nInput:   shape={tuple(X.shape)}  dtype={X.dtype}")

    # 4. Quantize
    X_packed, scale = downcast(X)
    ratio = _bytes(X) / (_bytes(X_packed) + _bytes(scale))
    print(f"Packed:  shape={tuple(X_packed.shape)}  dtype={X_packed.dtype}")
    print(f"Scale:   shape={tuple(scale.shape)}  dtype={scale.dtype}")
    print(f"Compression ratio: {ratio:.2f}x")

    # 5. Dequantize
    qdq_X = upcast(X_packed, scale, torch.bfloat16)

    # 6. Metrics
    X_f32 = X.float()
    qdq_f32 = qdq_X.float()

    cosine = torch.nn.functional.cosine_similarity(
        X_f32.reshape(-1), qdq_f32.reshape(-1), dim=0
    )
    l2_rel = torch.norm(X_f32 - qdq_f32) / (torch.norm(X_f32) + 1e-8)
    abs_err = (X_f32 - qdq_f32).abs()

    print(f"\n=== Reconstruction Quality ===")
    print(f"Cosine similarity:   {cosine.item():.8f}")
    print(f"L2 relative norm:    {l2_rel.item():.8f}")
    print(f"Max absolute error:  {abs_err.max().item():.6f}")
    print(f"Mean absolute error: {abs_err.mean().item():.6f}")


if __name__ == "__main__":
    main()
