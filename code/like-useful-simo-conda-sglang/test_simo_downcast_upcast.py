#!/usr/bin/env python3
"""Test SIMO downcast/upcast kernels for mxfp6_e2m3 quantization.

Reads a JSON quant config, extracts the weight spec, and measures
the reconstruction quality (cosine similarity, L2 relative norm) of an
mxfp6_e2m3 quantize-dequantize round-trip on a random bf16 tensor.
"""

import json
import os
import sys
import torch

# Ensure the SIMO package can be imported
_conda_env = os.getenv("CONDA_PREFIX", "/data/like/miniconda3/envs/simo_sglang")
_simo_src = os.path.join(_conda_env, "lib", "python3.12", "site-packages")
if os.path.isdir(_simo_src) and _simo_src not in sys.path:
    sys.path.insert(0, _simo_src)

# Alternative: if running from repo root
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from simo.extensions.sglang_simo.quantization.quantization import (
    parse_quantize_spec,
    get_downcast_kernel,
    get_upcast_kernel,
)


def _load_weight_spec(config_path: str) -> dict:
    """Extract the weight quant spec dict from a SIMO quant config JSON."""
    with open(config_path) as f:
        full = json.load(f)

    config = full.get("quantization_config", full)
    module_configs = config["module_configs"]
    if not module_configs:
        raise ValueError("module_configs is empty")

    # Take the weight spec from the first module_config entry
    weight_cfg = module_configs[0]["weight"]
    return weight_cfg


def main() -> None:
    config_path = os.path.join(
        _repo_root,
        "simo/extensions/sglang_simo/example/simo_quantization_config",
        "online_quantization/quant_config_w6a6_mxfp.json",
    )
    print(f"Loading config from: {config_path}")

    # 1. Parse the quant spec
    weight_spec_dict = _load_weight_spec(config_path)
    print(f"Weight spec dict: {weight_spec_dict}")

    spec = parse_quantize_spec(weight_spec_dict)
    print(f"Parsed spec: {spec}")

    # 2. Get downcast / upcast kernels
    downcast = get_downcast_kernel(spec)
    upcast = get_upcast_kernel(spec)
    print(f"Downcast kernel: {downcast}")
    print(f"Upcast kernel:   {upcast}")

    # 3. Create a random bf16 CUDA tensor
    M, K = 4096, 4096
    X = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    print(f"\nInput  tensor: shape={tuple(X.shape)}, dtype={X.dtype}, device={X.device}")

    # 4. Quantize (downcast)
    X_packed, scale = downcast(X)
    print(f"Packed tensor:  shape={tuple(X_packed.shape)}, dtype={X_packed.dtype}")
    print(f"Scale tensor:   shape={tuple(scale.shape)}, dtype={scale.dtype}")
    print(f"Compression ratio: {X.element_size() * X.numel() / (X_packed.element_size() * X_packed.numel() + scale.element_size() * scale.numel()):.2f}x")

    # 5. Dequantize (upcast)
    qdq_X = upcast(X_packed, scale, torch.bfloat16)
    print(f"Output tensor:  shape={tuple(qdq_X.shape)}, dtype={qdq_X.dtype}")

    # 6. Compute quality metrics
    X_f32 = X.float()
    qdq_X_f32 = qdq_X.float()

    # Cosine similarity (flat)
    cosine = torch.nn.functional.cosine_similarity(
        X_f32.reshape(-1), qdq_X_f32.reshape(-1), dim=0
    )
    # L2 relative norm
    l2_rel = torch.norm(X_f32 - qdq_X_f32) / (torch.norm(X_f32) + 1e-8)

    print(f"\n=== Reconstruction Quality ===")
    print(f"Cosine similarity:  {cosine.item():.8f}")
    print(f"L2 relative norm:   {l2_rel.item():.8f}")

    # Extra stats
    abs_err = (X_f32 - qdq_X_f32).abs()
    print(f"Max absolute error: {abs_err.max().item():.6f}")
    print(f"Mean absolute error:{abs_err.mean().item():.6f}")


if __name__ == "__main__":
    main()
