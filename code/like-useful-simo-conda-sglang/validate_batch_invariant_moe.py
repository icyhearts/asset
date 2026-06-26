#!/usr/bin/env python3
"""
Validate whether MoE routing and DeepSeek Router Gate non-determinism is the
primary source of gsm8k score differences with vs without enable_deterministic_inference.

Background:
  - Llama (dense, no MoE) without deterministic inference: score range 0.0129
    (0.7672 to 0.7801), even with KV cache from 267K down to 6K tokens.
  - DeepSeek-V2-Lite (MoE) without deterministic inference: score range ~0.0205
    (0.6497 to 0.6702), larger than Llama despite both using triton attention.

  Hypothesis: MoE routing non-determinism is the PRIMARY cause of the larger
  DeepSeek score differences. The router selects different top-k experts for
  the same token when batch composition changes → different experts compute
  the output → completely different result.

Tests:
  A: DeepSeek Router Gate — F.linear vs optimized router_gemm
  B: MoE Router — tensorcore vs cudacore kernel outputs
  C: MoE fused config — auto-tuned vs fixed config
  D: End-to-end: same hidden_states, different MoE routing → different output
"""

import math, os, sys
import torch
import torch.nn.functional as F

SGLANG_SRC = "/data/like/package/sglang_kernel_src/python"
if SGLANG_SRC not in sys.path:
    sys.path.insert(0, SGLANG_SRC)

device = "cuda"

def max_abs_diff(a, b):
    return (torch.abs(a - b)).max().item()

def rbf(shape, seed=42, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, device="cuda", generator=g, dtype=torch.float32)*scale).bfloat16()

def rbf_f32(shape, seed=42, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(shape, device="cuda", generator=g, dtype=torch.float32) * scale

# ============================================================
# TEST A: DeepSeek Router Gate — F.linear vs optimized router_gemm
#   (deepseek_v2.py:440-441 vs 454-485)
# ============================================================
print("=" * 80)
print("TEST A: DeepSeek Router Gate — F.linear vs optimized router_gemm")
print("  Does the deterministic F.linear path produce different logits from")
print("  the optimized router_gemm path for the same hidden_states and weight?")
print("=" * 80)

HAS_FLASHINFER_ROUTER = False  # init
try:
    from sgl_kernel import dsv3_router_gemm
    HAS_DSV3_ROUTER = True
except ImportError:
    print("  dsv3_router_gemm not available, trying flashinfer variant...")
    HAS_DSV3_ROUTER = False
    try:
        from sglang.srt.layers.flashinfer_router_gemm import flashinfer_dsv3_router_gemm
        HAS_FLASHINFER_ROUTER = True
    except ImportError:
        print("  No router gemm kernels available.")
        HAS_FLASHINFER_ROUTER = False

# Test configurations matching DeepSeek-V2-Lite router gate
# DeepSeek-V2-Lite: hidden_dim=2048, num_experts=64
HIDDEN_DIM = 2048
NUM_EXPERTS = 64

def test_router_gemm(batch_sizes, hidden_dim, num_experts, label=""):
    """Compare F.linear vs router_gemm for given batch sizes."""
    W = rbf((num_experts, hidden_dim), seed=42)
    print(f"\n  [{label}] hidden={hidden_dim} experts={num_experts}:")
    for bs in batch_sizes:
        x = rbf((bs, hidden_dim), seed=hash((bs, 100)))
        logits_linear = F.linear(x, W, None)
        logits_float32 = logits_linear.float()

        if HAS_DSV3_ROUTER:
            # Note: dsv3_router_gemm is hardcoded for hidden_dim=7168 (V3 only)
            if hidden_dim == 7168:
                logits_gemm = dsv3_router_gemm(x, W, out_dtype=torch.float32)
            else:
                logits_gemm = None  # skip comparison for non-V3 dims
        elif HAS_FLASHINFER_ROUTER:
            logits_gemm = torch.empty(bs, num_experts, device=device, dtype=torch.float32)
            flashinfer_dsv3_router_gemm(logits_gemm, x, W)
        else:
            logits_gemm = None

        mad = 0.0
        exact = True
        topk_match = True
        n_diff = 0
        n_top2_partial = 0
        if logits_gemm is not None:
            mad = max_abs_diff(logits_float32, logits_gemm)
            exact = torch.equal(logits_float32, logits_gemm)

            # Check if argmax (top-1 expert selection) differs
            top1_linear = logits_float32.argmax(dim=-1)
            top1_gemm = logits_gemm.argmax(dim=-1)
            topk_match = torch.equal(top1_linear, top1_gemm)

            # Check how many sequences have different top-1
            n_diff = (top1_linear != top1_gemm).sum().item()

            # For top-2 (if used), check match
            _, top2_linear = logits_float32.topk(2, dim=-1)
            _, top2_gemm = logits_gemm.topk(2, dim=-1)

            for i in range(bs):
                if set(top2_linear[i].tolist()) != set(top2_gemm[i].tolist()):
                    n_top2_partial += 1

        status = ""
        if logits_gemm is None:
            status = " [skip: kernel not available]"
        elif not exact:
            status += f" max_diff={mad:.6e}"
        if not topk_match and logits_gemm is not None:
            status += f" top1_diff={n_diff}/{bs}"
        if n_top2_partial > 0:
            status += f" top2_diff={n_top2_partial}/{bs}"
        if not status:
            status = " identical"

        print(f"    bs={bs:4d}  max_abs={mad:.6e}  top1_match={topk_match}{status}")

# Test with DeepSeek-V2-Lite dimensions
test_router_gemm([1, 4, 8, 16, 32, 64, 128, 256], HIDDEN_DIM, NUM_EXPERTS, "DSV2-Lite")

# Test with larger expert counts (V3 — hidden_dim must be 7168, max batch=16 for the kernel)
test_router_gemm([1, 4, 8, 16], 7168, 256, "DSV3 (hidden=7168)")

# ============================================================
# TEST B: MoE Router — tensorcore vs cudacore
#   (router.py:364-389)
# ============================================================
print("\n" + "=" * 80)
print("TEST B: MoE Router — tensorcore vs cudacore kernel")
print("  The deterministic path forces cudacore; non-deterministic uses tensorcore")
print("  for large batches. Do they produce different top-k selections?")
print("=" * 80)

try:
    from sglang.srt.layers.moe.router import (
        fused_moe_router_tensorcore,
        fused_moe_router_cudacore,
    )
    HAS_ROUTER_KERNELS = True
except ImportError as e:
    print(f"  MoE router kernels not importable: {e}")
    HAS_ROUTER_KERNELS = False

if HAS_ROUTER_KERNELS:
    print("\n  Comparing router_tensorcore vs router_cudacore...")

    for hidden_dim in [2048]:
        for num_experts in [64, 256]:
            W = rbf((num_experts, hidden_dim), seed=100)

            for bs in [1, 4, 16, 64, 128, 256, 512]:
                # Ensure hidden_dim % BLOCK_SIZE_K == 0 for tensorcore path
                if hidden_dim % 64 != 0:
                    continue
                x = rbf((bs, hidden_dim), seed=hash((bs, num_experts, 200)))

                # tensorcore path (non-deterministic)
                if bs >= 512 or num_experts > 8:
                    logits_tc = fused_moe_router_tensorcore(
                        x=x, router_weight=W, topk=2, moe_softcapping=0.0,
                        BLOCK_SIZE_M=32, BLOCK_SIZE_N=max(num_experts, 16),
                        BLOCK_SIZE_K=256 if num_experts < 256 else 64,
                    )
                else:
                    logits_tc = None

                # cudacore path (deterministic, always available)
                logits_cc = fused_moe_router_cudacore(
                    x=x, router_weight=W, topk=2, moe_softcapping=0.0,
                )

                if logits_tc is not None:
                    # tensorcore returns (topk_weights, topk_ids), same as cudacore
                    tc_weights, tc_ids = logits_tc
                    cc_weights, cc_ids = logits_cc
                    mad_ids = (tc_ids != cc_ids).sum().item()
                    mad_w = max_abs_diff(tc_weights, cc_weights)
                    top1_diff = (tc_ids[:, 0] != cc_ids[:, 0]).sum().item()

                    n_top2 = 0
                    for i in range(bs):
                        if set(tc_ids[i].tolist()) != set(cc_ids[i].tolist()):
                            n_top2 += 1

                    status = ""
                    if mad_ids > 0:
                        status += f" topk_id_diff={mad_ids}/{bs*tc_ids.shape[1]}"
                    if mad_w > 0:
                        status += f" weight_max_diff={mad_w:.6e}"
                    if not status:
                        status = " identical"

                    print(f"    h={hidden_dim} E={num_experts:3d} bs={bs:4d}{status}")
                else:
                    print(f"    h={hidden_dim} E={num_experts:3d} bs={bs:4d}  (tensorcore not applicable, cudacore only)")
else:
    print("  Skipping — MoE router kernels not importable")

# ============================================================
# TEST C: Impact of expert selection difference — simulate
# ============================================================
print("\n" + "=" * 80)
print("TEST C: Impact of different expert selection")
print("  If the same token is routed to different experts, the output changes")
print("  completely. Simulate this to quantify the magnitude.")
print("=" * 80)

# For DSV2-Lite: 64 experts, 2 shared + 6 routed, hidden=2048, intermediate=2048*2.4≈4915
# Simplified: each expert is an FFN (two linear layers with activation)
# If different experts are selected, output differs by ~ the weight norm
intermediate_size = 2048 * 3  # rough

# Simulate two different MoE expert FFNs
W1_A = rbf_f32((intermediate_size, 2048), seed=100, scale=0.02)
W2_A = rbf_f32((2048, intermediate_size), seed=101, scale=0.02)
W1_B = rbf_f32((intermediate_size, 2048), seed=200, scale=0.02)
W2_B = rbf_f32((2048, intermediate_size), seed=201, scale=0.02)

x = rbf_f32((1, 2048), seed=42)
out_A = F.silu(x @ W1_A.T) @ W2_A.T
out_B = F.silu(x @ W1_B.T) @ W2_B.T

diff = max_abs_diff(out_A, out_B)
print(f"  Same input routed to different expert:")
print(f"    Expert A output norm: {out_A.norm().item():.4f}")
print(f"    Expert B output norm: {out_B.norm().item():.4f}")
print(f"    max_abs_diff:         {diff:.6e}")
print(f"    This is {diff/x.abs().max().item():.2f}x the input magnitude")
print(f"  ** Different expert → completely different output **")

# ============================================================
# TEST D: Fused MoE Triton config — auto vs fixed
# ============================================================
print("\n" + "=" * 80)
print("TEST D: Fused MoE config — auto-tuned vs deterministic fixed config")
print("  The deterministic path uses fixed {M:64 N:64 K:32 GM:8}.")
print("  Auto-tuning selects different configs for different batch sizes.")
print("  Do they produce different outputs?")
print("=" * 80)

try:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        get_default_config,
    )
    HAS_MOE_CONFIG = True
except ImportError as e:
    print(f"  MoE config not importable: {e}")
    HAS_MOE_CONFIG = False

if HAS_MOE_CONFIG:
    deterministic_config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
                            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}
    print(f"  Deterministic config (enable_deterministic_inference=True): {deterministic_config}")
    # Show non-deterministic configs from the source code (fused_moe_triton_config.py:164-208)
    non_det_configs = {
        "fp8_w8a8 (M<=E)": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1},
        "fp8_w8a8 (M>E)": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32},
        "other (M<=E)": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1},
        "other default": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
    }
    print(f"  Non-deterministic configs (by batch size):")
    for label, cfg in non_det_configs.items():
        same = "**same**" if cfg == deterministic_config else "DIFFERENT"
        print(f"    {label:25s}: {cfg}  {same}")
    print(f"  ** The \"other default\" non-det config matches the deterministic one.")
    print(f"  ** fp8 configs show LARGER tile sizes → different accumulation order.")
else:
    print("  Skipping — MoE config not importable")

# ============================================================
# TEST E: Batch-dependent — same hidden_states, different batch → different routing?
# ============================================================
print("\n" + "=" * 80)
print("TEST E: Batch-dependent expert selection")
print("  Does the same token get different expert assignments when processed")
print("  alone vs in a larger batch?")
print("=" * 80)

# Simulate: one target token, processed alone (bs=1) vs in batch (bs=64)
# Same router weights, but different batch sizes may trigger different
# router_gemm kernels (F.linear vs dsv3_router_gemm)

W = rbf((NUM_EXPERTS, HIDDEN_DIM), seed=42)
x_target = rbf((1, HIDDEN_DIM), seed=999)

# Alone
logits_alone = F.linear(x_target, W, None).float()
top1_alone = logits_alone.argmax(dim=-1)
_, top2_alone = logits_alone.topk(2, dim=-1)

# In batch of 64 (with other random tokens mixed in)
other_tokens = rbf((63, HIDDEN_DIM), seed=hash((64, 300)))
x_batch = torch.cat([x_target, other_tokens], dim=0)

# The router operates on the full batch
# NOTE: dsv3_router_gemm requires hidden_dim=7168 (hardcoded for V3).
# For DSV2-Lite (hidden_dim=2048), only F.linear is available.
HAS_ROUTER_GEMM_FOR_THIS_DIM = HAS_DSV3_ROUTER and HIDDEN_DIM == 7168
if HAS_ROUTER_GEMM_FOR_THIS_DIM:
    logits_all_gemm = dsv3_router_gemm(x_batch, W, out_dtype=torch.float32)
elif HAS_FLASHINFER_ROUTER:
    logits_all_gemm = torch.empty(64, NUM_EXPERTS, device=device, dtype=torch.float32)
    flashinfer_dsv3_router_gemm(logits_all_gemm, x_batch, W)
else:
    logits_all_gemm = F.linear(x_batch, W, None).float()

logits_target_in_batch = logits_all_gemm[0:1]
logits_all_linear = F.linear(x_batch, W, None).float()
logits_target_linear_in_batch = logits_all_linear[0:1]

print(f"\n  Target token alone:")
print(f"    F.linear logits:     {logits_alone[0, :6].tolist()}...")
print(f"    top-1 expert:        {top1_alone.item()}")
print(f"    top-2 experts:       {top2_alone[0].tolist()}")

print(f"\n  Target token in batch of 64:")
print(f"    F.linear logits:     {logits_target_linear_in_batch[0, :6].tolist()}...")

if HAS_ROUTER_GEMM_FOR_THIS_DIM or HAS_FLASHINFER_ROUTER:
    print(f"    router_gemm logits:  {logits_target_in_batch[0, :6].tolist()}...")

    # Compare F.linear alone vs F.linear in batch (should be identical — same input)
    print(f"\n  F.linear(alone) vs F.linear(in_batch):")
    print(f"    max_abs_diff: {max_abs_diff(logits_alone, logits_target_linear_in_batch):.6e}")
    print(f"    exact_match:  {torch.equal(logits_alone, logits_target_linear_in_batch)}")

    # Compare F.linear alone vs router_gemm in batch
    print(f"\n  F.linear(alone) vs router_gemm(in_batch):")
    diff_alone_vs_gemm = max_abs_diff(logits_alone, logits_target_in_batch)
    top1_gemm = logits_target_in_batch.argmax(dim=-1)
    _, top2_gemm = logits_target_in_batch.topk(2, dim=-1)
    print(f"    max_abs_diff:       {diff_alone_vs_gemm:.6e}")
    print(f"    exact_match:        {torch.equal(logits_alone, logits_target_in_batch)}")
    print(f"    top-1 (router_gemm): {top1_gemm.item()}")
    print(f"    top-2 (router_gemm): {top2_gemm[0].tolist()}")

    expert_changed = top1_alone.item() != top1_gemm.item()
    if expert_changed:
        print(f"    ** EXPERT SELECTION CHANGED! F.linear→{top1_alone.item()} vs router_gemm→{top1_gemm.item()} **")
    else:
        print(f"    Same top-1 expert, but check top-2:")
        t2a = set(top2_alone[0].tolist())
        t2b = set(top2_gemm[0].tolist())
        if t2a != t2b:
            print(f"    ** TOP-2 EXPERTS CHANGED! {t2a} vs {t2b} **")
        else:
            print(f"    top-2 also same: {t2a}")
else:
    print(f"  No router_gemm kernels available. Using F.linear only.")
    print(f"  F.linear(alone) vs F.linear(in_batch):")
    print(f"    max_abs_diff: {max_abs_diff(logits_alone, logits_target_linear_in_batch):.6e}")
    print(f"    exact_match:  {torch.equal(logits_alone, logits_target_linear_in_batch)}")

# ============================================================
# CONCLUSION
# ============================================================
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
