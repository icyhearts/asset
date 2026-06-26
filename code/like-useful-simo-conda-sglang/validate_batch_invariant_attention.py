#!/usr/bin/env python3
"""
Validate whether attention KV split determinism is the primary source of
gsm8k score differences with vs without enable_deterministic_inference.

Key hypothesis:
  Without deterministic inference, get_num_kv_splits() (triton_backend.py:238-287)
  uses a batch-dependent dynamic Triton kernel that can assign DIFFERENT numbers
  of KV splits to the same sequence length depending on batch composition.
  Different split counts → different chunk boundaries → different FP accumulation
  order in the two-stage flash-decoding → different attention output.

Tests:
  A: Does different num_kv_splits (same Q/K/V) produce different output?
  B: Does get_num_kv_splits produce batch-dependent results without deterministic mode?
  C: End-to-end: same request in different batches → different output?
  D: End-to-end with deterministic: same request in different batches → same output?
"""

import os, sys
import torch

# Setup
SGLANG_SRC = "/data/like/package/sglang_kernel_src/python"
if SGLANG_SRC not in sys.path:
    sys.path.insert(0, SGLANG_SRC)

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd,
)

# ---------- helpers ----------
def max_abs_diff(a, b):
    return (torch.abs(a - b)).max().item()

def describe(a, b, tag=""):
    exact = torch.equal(a, b)
    mad = max_abs_diff(a, b)
    print(f"  [{tag}] exact={exact}  max_abs_diff={mad:.6e}")

def rbf(shape, seed=42, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, device="cuda", generator=g, dtype=torch.float32)*scale).bfloat16()

# ============================================================
# Configuration
# ============================================================
device = "cuda"
num_heads = 8
head_dim = 128
seq_len = 512        # representative decode sequence length
bs = 4               # batch size for multi-seq tests

# ============================================================
# TEST A: Same Q/K/V, different num_kv_splits → different output?
# ============================================================
print("=" * 80)
print("TEST A: Same Q/K/V with different num_kv_splits → different output?")
print("=" * 80)

max_kv_splits = 8
Q = rbf((bs, num_heads, head_dim), seed=1)
K = rbf((bs * seq_len, num_heads, head_dim), seed=2)
V = rbf((bs * seq_len, num_heads, head_dim), seed=3)

kv_indptr = torch.tensor([0, seq_len, 2*seq_len, 3*seq_len, 4*seq_len],
                          dtype=torch.int32, device=device)
kv_indices = torch.arange(bs * seq_len, dtype=torch.int64, device=device)

o = torch.empty(bs, num_heads, head_dim, dtype=torch.bfloat16, device=device)
attn_logits = torch.empty(bs, num_heads, max_kv_splits, head_dim,
                           dtype=torch.float32, device=device)
attn_lse = torch.empty(bs, num_heads, max_kv_splits,
                        dtype=torch.float32, device=device)

results = {}
for num_splits_val in [1, 2, 4, 8]:
    num_kv_splits = torch.full((bs,), num_splits_val, dtype=torch.int32, device=device)
    o.zero_()
    attn_logits.zero_()
    attn_lse.zero_()

    decode_attention_fwd(
        Q, K, V, o, kv_indptr, kv_indices,
        attn_logits, attn_lse, num_kv_splits, max_kv_splits,
        sm_scale=head_dim ** -0.5, k_scale=1.0, v_scale=1.0,
        logit_cap=0, sinks=None, xai_temperature_len=0,
        has_mla=False, use_pdl=False,
    )
    results[num_splits_val] = o.clone()
    torch.cuda.synchronize()

print("Comparing outputs with different num_kv_splits (same Q/K/V):")
for s1 in [1, 2, 4, 8]:
    for s2 in [1, 2, 4, 8]:
        if s1 >= s2:
            continue
        describe(results[s1], results[s2], f"splits={s1} vs splits={s2}")

# Also show diff magnitude for each pair
print("\nPairwise max_abs_diff matrix:")
pairs = [1, 2, 4, 8]
header = "        " + " ".join(f"splits={s:1d}" for s in pairs)
print(header)
for s1 in pairs:
    row = f"splits={s1:1d}  "
    for s2 in pairs:
        row += f"{max_abs_diff(results[s1], results[s2]):11.3e} "
    print(row)

# ============================================================
# TEST B: Non-deterministic vs deterministic num_kv_splits
#   Compare get_num_kv_splits behavior to understand the mechanism.
#   We simulate what the Triton kernel does: compute splits based
#   on batch layout (device_core_count, num_heads, num_seq) vs
#   fixed split_tile_size.
# ============================================================
print("\n" + "=" * 80)
print("TEST B: Simulated num_kv_splits — deterministic vs dynamic")
print("=" * 80)

# Deterministic formula (triton_backend.py:268-271)
def get_splits_deterministic(seq_lens, split_tile_size=256):
    return (seq_lens + split_tile_size - 1) // split_tile_size

# Dynamic formula simulates what get_num_kv_splits_triton roughly does:
# It considers device_core_count and divides work across available SMs.
# A simplified proxy: split count ≈ device_cores / (num_seq * num_heads_factor)
# For SM90 (H100) with 132 SMs, the actual logic is more complex but the
# key point is it's batch-dependent.
def get_splits_dynamic_sim(seq_lens, num_seq, num_heads, max_splits, device_cores=132):
    """Simplified simulation of the dynamic Triton kernel logic.

    The real kernel (triton_backend.py:1434) computes:
      kv_chunk_size_1 = max_seq_len / max_kv_splits
      total_kv_tasks = per-head kv tasks across all sequences
      num_kv_chunks = maximum chunks that can be allocated
      kv_chunk_size_2 = total_kv_len / num_kv_chunks
      For each seq: splits = max(seq_len/kv_chunk_size_1, seq_len/kv_chunk_size_2)
    """
    bs = len(seq_lens)
    max_seq_len = seq_lens.max().item()
    total_kv_len = seq_lens.sum().item()

    # chunk_size_1: uniform distribution
    kv_chunk_size_1 = max_seq_len / max_splits

    # chunk_size_2: based on device core count
    # grid = num_heads * num_seq  (each head per sequence is a task)
    token_grid = num_heads * num_seq
    # available_sms factor: device_cores - reserved_for_other_work
    available_kv_sms = max(1, device_cores - 8)  # reserve ~8 SMs
    # how many KV chunks total
    num_kv_chunks = max(1, min(max_splits * token_grid,
                               (available_kv_sms // min(token_grid, 8)) * max_splits))
    kv_chunk_size_2 = total_kv_len / num_kv_chunks

    splits = torch.zeros(bs, dtype=torch.int32)
    for i in range(bs):
        s1 = max(1, int(seq_lens[i].item() / kv_chunk_size_1))
        s2 = max(1, int(seq_lens[i].item() / kv_chunk_size_2)) if kv_chunk_size_2 > 0 else 1
        splits[i] = min(max_splits, max(s1, s2))
    return splits

# Test: show that different batch compositions can give different splits
# for the SAME sequence length
test_seq_len = 2048
print(f"\nTarget seq_len = {test_seq_len}, max_splits = 16")

# Scenario 1: Small batch (e.g., large KV cache, few concurrent requests)
seq_lens_small = torch.tensor([test_seq_len, 1024], dtype=torch.int32)
det_small = get_splits_deterministic(seq_lens_small, 256)
dyn_small = get_splits_dynamic_sim(seq_lens_small, len(seq_lens_small), num_heads, 16)
print(f"  Small batch (2 seqs): seq_lens={seq_lens_small.tolist()}")
print(f"    deterministic splits: {det_small.tolist()}")
print(f"    dynamic     splits: {dyn_small.tolist()}")

# Scenario 2: Large batch (e.g., small KV cache, many concurrent requests)
seq_lens_large = torch.tensor([test_seq_len, 1024, 1500, 800, 600, 1200, 1800, 2048],
                               dtype=torch.int32)
det_large = get_splits_deterministic(seq_lens_large, 256)
dyn_large = get_splits_dynamic_sim(seq_lens_large, len(seq_lens_large), num_heads, 16)
print(f"\n  Large batch (8 seqs): seq_lens={seq_lens_large.tolist()}")
print(f"    deterministic splits: {det_large.tolist()}")
print(f"    dynamic     splits: {dyn_large.tolist()}")

# Key comparison: same seq_len in different batch → different splits?
if dyn_small[0] != dyn_large[0]:
    print(f"\n  ** seq_len={test_seq_len} gets different splits: {dyn_small[0].item()} vs {dyn_large[0].item()}")
    print(f"  ** This confirms batch-dependent KV split assignment!")
    print(f"  ** Deterministic always gives: {det_small[0].item()} (seq_len only)")
else:
    print(f"\n  seq_len={test_seq_len} same splits in both: {dyn_small[0].item()}")

# ============================================================
# TEST C: End-to-end — same request in different batches
#   A sequence with length L is processed:
#     Once alone (batch of 1)
#     Once mixed with other sequences (batch of N)
#   Does its attention output differ?
# ============================================================
print("\n" + "=" * 80)
print("TEST C: End-to-end — same sequence in different batch compositions")
print("=" * 80)

target_seq_len = 2048
companion_lens = [512, 1024, 1500]  # other sequences mixed in
max_splits = 16

# Setup: single sequence's Q/K/V
Q_single = rbf((1, num_heads, head_dim), seed=100)
K_single = rbf((target_seq_len, num_heads, head_dim), seed=200)
V_single = rbf((target_seq_len, num_heads, head_dim), seed=300)

# Shared buffer: Q, K, V for all sequences together
Q_all = torch.cat([Q_single] + [rbf((1, num_heads, head_dim), seed=400+i) for i in range(len(companion_lens))], dim=0)
K_all = torch.cat([K_single] + [rbf((cl, num_heads, head_dim), seed=500+i) for i, cl in enumerate(companion_lens)], dim=0)
V_all = torch.cat([V_single] + [rbf((cl, num_heads, head_dim), seed=600+i) for i, cl in enumerate(companion_lens)], dim=0)

total_bs = 1 + len(companion_lens)
seq_lens_all = [target_seq_len] + companion_lens
cumsum = [0]
for sl in seq_lens_all:
    cumsum.append(cumsum[-1] + sl)
kv_indptr_all = torch.tensor(cumsum, dtype=torch.int32, device=device)
kv_indices_all = torch.arange(cumsum[-1], dtype=torch.int64, device=device)

# --- Run alone (batch=1) ---
kv_indptr_alone = torch.tensor([0, target_seq_len], dtype=torch.int32, device=device)
kv_indices_alone = torch.arange(target_seq_len, dtype=torch.int64, device=device)

o_alone = torch.empty(1, num_heads, head_dim, dtype=torch.bfloat16, device=device)
al_alone = torch.empty(1, num_heads, max_splits, head_dim, dtype=torch.float32, device=device)
lse_alone = torch.empty(1, num_heads, max_splits, dtype=torch.float32, device=device)

# Non-deterministic simulation: use dynamic split for alone case
dyn_split_alone = get_splits_dynamic_sim(
    torch.tensor([target_seq_len], dtype=torch.int32), 1, num_heads, max_splits
)
det_split_alone = get_splits_deterministic(
    torch.tensor([target_seq_len], dtype=torch.int32), 256
)

# --- Run in full batch ---
o_full = torch.empty(total_bs, num_heads, head_dim, dtype=torch.bfloat16, device=device)
al_full = torch.empty(total_bs, num_heads, max_splits, head_dim, dtype=torch.float32, device=device)
lse_full = torch.empty(total_bs, num_heads, max_splits, dtype=torch.float32, device=device)

dyn_split_full = get_splits_dynamic_sim(
    torch.tensor(seq_lens_all, dtype=torch.int32), total_bs, num_heads, max_splits
)
det_split_full = get_splits_deterministic(
    torch.tensor(seq_lens_all, dtype=torch.int32), 256
)

print(f"seq_lens: {seq_lens_all}")
print(f"Deterministic splits:  alone={det_split_alone.tolist()}  full_batch={det_split_full.tolist()}")
print(f"Dynamic splits:        alone={dyn_split_alone.tolist()}  full_batch={dyn_split_full.tolist()}")
print(f"Target seq_len={target_seq_len}: det_split={det_split_alone[0].item()} dyn_alone={dyn_split_alone[0].item()} dyn_full={dyn_split_full[0].item()}")

# --- Run with dynamic splits (non-deterministic simulation) ---
print("\n--- With DYNAMIC splits (simulating non-deterministic mode) ---")
nks_alone = torch.tensor(dyn_split_alone, dtype=torch.int32, device=device)
decode_attention_fwd(
    Q_single, K_single, V_single, o_alone,
    kv_indptr_alone, kv_indices_alone,
    al_alone, lse_alone, nks_alone, max_splits,
    sm_scale=head_dim**-0.5, k_scale=1.0, v_scale=1.0,
    logit_cap=0, sinks=None, xai_temperature_len=0,
    has_mla=False, use_pdl=False,
)

nks_full = torch.tensor(dyn_split_full, dtype=torch.int32, device=device)
decode_attention_fwd(
    Q_all, K_all, V_all, o_full,
    kv_indptr_all, kv_indices_all,
    al_full, lse_full, nks_full, max_splits,
    sm_scale=head_dim**-0.5, k_scale=1.0, v_scale=1.0,
    logit_cap=0, sinks=None, xai_temperature_len=0,
    has_mla=False, use_pdl=False,
)

torch.cuda.synchronize()

# Extract the target sequence's output from the full batch
o_target_from_full = o_full[0:1]  # first sequence is our target
describe(o_alone, o_target_from_full, "dynamic: alone vs in-batch")

# --- Run with deterministic splits ---
print("\n--- With DETERMINISTIC splits (simulating deterministic mode) ---")
o_alone_det = torch.empty(1, num_heads, head_dim, dtype=torch.bfloat16, device=device)
nks_alone = torch.tensor(det_split_alone, dtype=torch.int32, device=device)
decode_attention_fwd(
    Q_single, K_single, V_single, o_alone_det,
    kv_indptr_alone, kv_indices_alone,
    al_alone, lse_alone, nks_alone, max_splits,
    sm_scale=head_dim**-0.5, k_scale=1.0, v_scale=1.0,
    logit_cap=0, sinks=None, xai_temperature_len=0,
    has_mla=False, use_pdl=False,
)

o_full_det = torch.empty(total_bs, num_heads, head_dim, dtype=torch.bfloat16, device=device)
nks_full = torch.tensor(det_split_full, dtype=torch.int32, device=device)
decode_attention_fwd(
    Q_all, K_all, V_all, o_full_det,
    kv_indptr_all, kv_indices_all,
    al_full, lse_full, nks_full, max_splits,
    sm_scale=head_dim**-0.5, k_scale=1.0, v_scale=1.0,
    logit_cap=0, sinks=None, xai_temperature_len=0,
    has_mla=False, use_pdl=False,
)

torch.cuda.synchronize()

o_target_det = o_full_det[0:1]
describe(o_alone_det, o_target_det, "deterministic: alone vs in-batch")

# --- Cross-compare: dynamic alone vs deterministic alone ---
print("\n--- Cross-comparison ---")
describe(o_alone, o_alone_det, "alone: dynamic vs deterministic splits")

# ============================================================
# TEST D: Repeated runs with dynamic splits vs fixed splits
#   Show that with dynamic splits, the same request's output
#   depends on batch companions, while with fixed splits it doesn't.
# ============================================================
print("\n" + "=" * 80)
print("TEST D: Batch-companion-dependent output (dynamic) vs independent (deterministic)")
print("=" * 80)

# Create 3 different batch compositions, all containing our target sequence
target_sl = 2048
batch_configs = [
    ([target_sl], "alone"),
    ([target_sl, 512], "+short"),
    ([target_sl, 512, 1024, 1500, 800], "+4 companions"),
]

# Target sequence's fixed Q/K/V
Q_tgt = rbf((1, num_heads, head_dim), seed=700)
K_tgt = rbf((target_sl, num_heads, head_dim), seed=800)
V_tgt = rbf((target_sl, num_heads, head_dim), seed=900)

outputs_dynamic = []
outputs_deterministic = []

for seq_lens, label in batch_configs:
    bs_cfg = len(seq_lens)
    total_kv = sum(seq_lens)

    # Build Q, K, V with target as first sequence
    Q_cfg = [Q_tgt] + [rbf((1, num_heads, head_dim), seed=1000+hash((label, i)))
                        for i in range(bs_cfg-1)]
    K_cfg = [K_tgt] + [rbf((sl, num_heads, head_dim), seed=1100+hash((label, i)))
                        for i, sl in enumerate(seq_lens[1:])]
    V_cfg = [V_tgt] + [rbf((sl, num_heads, head_dim), seed=1200+hash((label, i)))
                        for i, sl in enumerate(seq_lens[1:])]

    Q_all = torch.cat(Q_cfg, dim=0)
    K_all = torch.cat(K_cfg, dim=0)
    V_all = torch.cat(V_cfg, dim=0)

    cum = [0]
    for sl in seq_lens:
        cum.append(cum[-1]+sl)
    kv_indptr = torch.tensor(cum, dtype=torch.int32, device=device)
    kv_indices = torch.arange(total_kv, dtype=torch.int64, device=device)

    # Dynamic splits
    dyn = get_splits_dynamic_sim(torch.tensor(seq_lens, dtype=torch.int32), bs_cfg, num_heads, 16)
    det = get_splits_deterministic(torch.tensor(seq_lens, dtype=torch.int32), 256)

    o_dyn = torch.empty(bs_cfg, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    al = torch.empty(bs_cfg, num_heads, 16, head_dim, dtype=torch.float32, device=device)
    lse = torch.empty(bs_cfg, num_heads, 16, dtype=torch.float32, device=device)

    nks = torch.tensor(dyn, dtype=torch.int32, device=device)
    decode_attention_fwd(Q_all, K_all, V_all, o_dyn, kv_indptr, kv_indices,
                         al, lse, nks, 16,
                         sm_scale=head_dim**-0.5, k_scale=1.0, v_scale=1.0,
                         logit_cap=0, sinks=None, xai_temperature_len=0,
                         has_mla=False, use_pdl=False)
    outputs_dynamic.append((label, o_dyn[0:1].clone(), dyn[0].item()))

    o_det = torch.empty(bs_cfg, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    nks = torch.tensor(det, dtype=torch.int32, device=device)
    decode_attention_fwd(Q_all, K_all, V_all, o_det, kv_indptr, kv_indices,
                         al, lse, nks, 16,
                         sm_scale=head_dim**-0.5, k_scale=1.0, v_scale=1.0,
                         logit_cap=0, sinks=None, xai_temperature_len=0,
                         has_mla=False, use_pdl=False)
    outputs_deterministic.append((label, o_det[0:1].clone(), det[0].item()))

    torch.cuda.synchronize()

# Show results: dynamic mode
print("\nDynamic splits (non-deterministic simulation):")
for label, out, splits in outputs_dynamic:
    print(f"  {label:15s} num_splits_for_target={splits}")
for i in range(len(outputs_dynamic)):
    for j in range(i+1, len(outputs_dynamic)):
        l1, o1, s1 = outputs_dynamic[i]
        l2, o2, s2 = outputs_dynamic[j]
        describe(o1, o2, f"dynamic: {l1} vs {l2}")
        if not torch.equal(o1, o2):
            print(f"    ** BATCH-DEPENDENT! {l1}(splits={s1}) != {l2}(splits={s2}) **")

# Show results: deterministic mode
print("\nDeterministic splits:")
for label, out, splits in outputs_deterministic:
    print(f"  {label:15s} num_splits_for_target={splits}")
for i in range(len(outputs_deterministic)):
    for j in range(i+1, len(outputs_deterministic)):
        l1, o1, s1 = outputs_deterministic[i]
        l2, o2, s2 = outputs_deterministic[j]
        describe(o1, o2, f"deterministic: {l1} vs {l2}")

# ============================================================
# TEST E: Magnitude analysis — how much does the difference matter?
# ============================================================
print("\n" + "=" * 80)
print("TEST E: Can single-token attention difference flip the argmax?")
print("=" * 80)

# For one of the differing cases, check if the max element differs
for i in range(len(outputs_dynamic)):
    for j in range(i+1, len(outputs_dynamic)):
        l1, o1, s1 = outputs_dynamic[i]
        l2, o2, s2 = outputs_dynamic[j]
        if not torch.equal(o1, o2):
            diff = o1.float() - o2.float()
            max_diff_idx = diff.abs().argmax()
            flat_idx = max_diff_idx.item()
            print(f"\n  {l1} vs {l2}:")
            print(f"    max element diff: {diff.abs().max().item():.6e}")
            print(f"    at index {flat_idx}: val1={o1.float().flatten()[flat_idx].item():.6f}  val2={o2.float().flatten()[flat_idx].item():.6f}")

            # Check if the argmax position differs
            am1 = o1.float().reshape(-1).argmax()
            am2 = o2.float().reshape(-1).argmax()
            if am1 != am2:
                print(f"    ** ARGMAX DIFFERS! argmax1={am1.item()} argmax2={am2.item()} **")
                print(f"    This proves KV split non-determinism can change model output tokens.")
            else:
                print(f"    argmax same: {am1.item()}")

# ============================================================
# CONCLUSION
# ============================================================
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
