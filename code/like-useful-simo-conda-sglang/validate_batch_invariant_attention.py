#!/usr/bin/env python3
"""
Validate whether attention KV split determinism is the primary source of
gsm8k score differences with vs without enable_deterministic_inference.

Core mechanism:
  The decode attention uses a two-stage flash-decoding algorithm:
    Stage 1: Split KV sequence into num_kv_splits chunks, compute partial attn per chunk
    Stage 2: Merge all partial results via online softmax reduction

  Without deterministic inference (triton_backend.py:278-287):
    get_num_kv_splits_triton kernel dynamically computes num_kv_splits based on
    batch composition (num_seq, num_group, num_heads, device_core_count).
    Same seq_len in different batches → different num_kv_splits.

  With deterministic inference (triton_backend.py:268-271):
    num_kv_splits = (seq_len + split_tile_size - 1) // split_tile_size
    Pure function of seq_len only, independent of batch composition.

Tests:
  A: Different num_kv_splits for same Q/K/V → different output?
  B: Accurate simulation of dynamic vs deterministic split assignment
  C: End-to-end: same request in different batches → attention output differs?
"""

import math, os, sys
import torch

SGLANG_SRC = "/data/like/package/sglang_kernel_src/python"
if SGLANG_SRC not in sys.path:
    sys.path.insert(0, SGLANG_SRC)
from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd

device = "cuda"

def max_abs_diff(a, b):
    return (torch.abs(a - b)).max().item()

def rbf(shape, seed=42, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, device="cuda", generator=g, dtype=torch.float32)*scale).bfloat16()

# ============================================================
# Accurate Python reimplementation of get_num_kv_splits_triton
# (matching triton_backend.py:1435-1483)
# ============================================================
def get_num_kv_splits_dynamic(seq_lens, num_heads, num_kv_heads, max_kv_splits,
                                device_core_count=132, num_group=1):
    """
    Accurate reimplementation of the Triton kernel get_num_kv_splits_triton
    (triton_backend.py:1435-1483).

    The kernel uses TWO methods to compute split counts and takes the max:
      Method 1: based on max_seq_len / min_seq_len ratio (uniform across seqs)
      Method 2: based on device_cores / token_grid (depends on num_seq)
    """
    bs = len(seq_lens)
    max_seq_len = max(seq_lens)
    min_seq_len = min(seq_lens)

    # Method 1: uniform based on length ratios
    if max_seq_len * 8 < min_seq_len * 10:  # within 80% ratio
        min_seq_len = max_seq_len
    max_kv_splits_1 = min(math.ceil(max_seq_len / min_seq_len), max_kv_splits)
    kv_chunk_size_1 = math.ceil(max_seq_len / max_kv_splits_1)

    # Method 2: based on device core count / token grid
    ext_seq_len = max_seq_len / 64.0
    ext_device_cores = int(device_core_count * max(math.log2(ext_seq_len), 1.0))

    num_kv_group = num_heads // num_kv_heads
    if num_kv_group == 1:
        token_grid = bs * num_group * num_heads
    else:
        block_h = min(16, num_kv_group)
        token_grid = bs * num_group * math.ceil(num_heads / block_h)

    max_kv_splits_2 = min(math.ceil(ext_device_cores / max(token_grid, 1)), max_kv_splits)
    kv_chunk_size_2 = math.ceil(max_seq_len / max(max_kv_splits_2, 1))

    # For each sequence: max of both methods
    result = []
    for sl in seq_lens:
        s1 = math.ceil(sl / kv_chunk_size_1)
        s2 = math.ceil(sl / kv_chunk_size_2) if kv_chunk_size_2 > 0 else 1
        result.append(max(1, min(max(s1, s2), max_kv_splits)))
    return result

def get_num_kv_splits_deterministic(seq_lens, split_tile_size=256):
    return [max(1, (sl + split_tile_size - 1) // split_tile_size) for sl in seq_lens]

# ============================================================
# TEST A: Same Q/K/V, different num_kv_splits → different output?
# ============================================================
print("=" * 80)
print("TEST A: Same Q/K/V with different num_kv_splits → different attention output?")
print("  This isolates the KV-split mechanism from batch composition effects.")
print("=" * 80)

num_heads, head_dim = 32, 128
seq_len = 1024  # typical decode length
max_splits = 16
nks_values = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16]

Q = rbf((1, num_heads, head_dim), seed=100)
K = rbf((seq_len, num_heads, head_dim), seed=200)
V = rbf((seq_len, num_heads, head_dim), seed=300)
kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
kv_indices = torch.arange(seq_len, dtype=torch.int64, device=device)

results = {}
for nkv in nks_values:
    o = torch.empty(1, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    al = torch.empty(1, num_heads, max_splits, head_dim, dtype=torch.float32, device=device)
    lse = torch.empty(1, num_heads, max_splits, dtype=torch.float32, device=device)
    nks = torch.full((1,), nkv, dtype=torch.int32, device=device)

    decode_attention_fwd(Q, K, V, o, kv_indptr, kv_indices,
                         al, lse, nks, max_splits,
                         sm_scale=head_dim**-0.5, k_scale=1.0, v_scale=1.0,
                         logit_cap=0, sinks=None, xai_temperature_len=0,
                         has_mla=False, use_pdl=False)
    results[nkv] = o.clone()

torch.cuda.synchronize()

print(f"seq_len={seq_len} num_heads={num_heads} head_dim={head_dim} max_splits={max_splits}")
print("\nPairwise max_abs_diff matrix:")
header = "        " + "".join(f" nkv={n:2d}" for n in nks_values)
print(header)
for n1 in nks_values:
    row = f"nkv={n1:2d}"
    for n2 in nks_values:
        row += f"  {max_abs_diff(results[n1], results[n2]):8.2e}"
    print(row)

n_different = sum(1 for n1 in nks_values for n2 in nks_values
                  if n1 < n2 and not torch.equal(results[n1], results[n2]))
print(f"\n{n_different} out of {len(nks_values)*(len(nks_values)-1)//2} pairs differ")
if n_different > 0:
    max_diff = max(max_abs_diff(results[n1], results[n2])
                   for n1 in nks_values for n2 in nks_values if n1 < n2)
    print(f"Max difference across all pairs: {max_diff:.6e}")
    print("** CONFIRMED: different num_kv_splits → different attention output **")
else:
    print("All pairs identical for this config")

# ============================================================
# TEST B: Accurate simulation — dynamic vs deterministic splits
# ============================================================
print("\n" + "=" * 80)
print("TEST B: Accurate simulation of get_num_kv_splits — dynamic vs deterministic")
print("  Reimplementation matching triton_backend.py:1435-1483")
print("=" * 80)

# Use realistic DeepSeek-V2-Lite numbers
num_heads_test = 16
num_kv_heads_test = 16  # 1:1 for Lite (non-GQA)
device_cores = 132  # H100

test_target_sl = 2048
scenarios = [
    ([test_target_sl], "alone"),
    ([test_target_sl, 1024], "batch_2"),
    ([test_target_sl, 512, 1024, 1500], "batch_4"),
    ([test_target_sl, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800], "batch_10"),
    ([test_target_sl, 1800, 1600, 1400, 800, 600, 400, 200, 120, 80], "batch_10_v2"),
]

print(f"Target seq_len={test_target_sl}  device_cores={device_cores}")
print(f"num_heads={num_heads_test}  num_kv_heads={num_kv_heads_test}  max_splits=16\n")

any_batch_dep = False
for seq_lens, label in scenarios:
    det = get_num_kv_splits_deterministic(seq_lens, 256)
    dyn = get_num_kv_splits_dynamic(seq_lens, num_heads_test, num_kv_heads_test, 16, device_cores)
    target_det = det[0]
    target_dyn = dyn[0]
    diff_mark = ""
    # Check if the same seq_len gets different splits in different batches
    if scenarios[0][0] != seq_lens:  # not the alone case
        first_dyn = get_num_kv_splits_dynamic(scenarios[0][0], num_heads_test, num_kv_heads_test, 16, device_cores)[0]
        if first_dyn != target_dyn:
            diff_mark = f" <-- DIFFERENT from alone case ({first_dyn})!"
            any_batch_dep = True
    print(f"  {label:15s} det={target_det:2d}  dyn={target_dyn:2d}{diff_mark}")

if any_batch_dep:
    print(f"\n** CONFIRMED: dynamic num_kv_splits is batch-dependent for seq_len={test_target_sl}")
else:
    print(f"\nseq_len={test_target_sl}: dynamic splits same across all test batches")

# Try wider range of seq_lens to find batch-dependent cases
print(f"\nSearching for batch-dependent cases across more configurations...")
found_any = False
for target_sl in [512, 1024, 2048, 4096, 8192]:
    for bs in [1, 2, 4, 8, 16]:
        companion_lens = [max(32, target_sl - i*128) for i in range(1, bs)]
        seq_lens = [target_sl] + companion_lens
        if bs == 1:
            alone_dyn = get_num_kv_splits_dynamic(seq_lens, num_heads_test, num_kv_heads_test, 16, device_cores)[0]
            continue
        batch_dyn = get_num_kv_splits_dynamic(seq_lens, num_heads_test, num_kv_heads_test, 16, device_cores)[0]
        if alone_dyn != batch_dyn:
            if not found_any:
                print(f"  target_sl={'alone':>5s}  {'batch N':>5s}  {'seq_lens':>40s}")
                found_any = True
            print(f"  target_sl={alone_dyn:5d}  {batch_dyn:5d}  {str(seq_lens[:5]):>40s}")
            break  # one example per target_sl is enough
    if bs == 1:
        # reset for next target_sl
        pass

if not found_any:
    print("  No batch-dependent cases found with these parameters.")
    print("  The dynamic Triton kernel is quite stable for moderate batch sizes.")
    print("  But the PRINCIPLE holds: it IS batch-dependent by design.")

# ============================================================
# TEST C: End-to-end — construct a scenario where splits differ
# ============================================================
print("\n" + "=" * 80)
print("TEST C: Prove that different num_kv_splits changes attention output")
print("  End-to-end: run decode_attention_fwd with deliberately different splits")
print("=" * 80)

# We know from TEST A that for seq_len=1024 with 32 heads, different splits
# produce different outputs. Let's verify this for a larger, more realistic config
# that matches typical decode scenarios.

num_heads_c = 16
head_dim_c = 128
seq_len_c = 4096
max_splits_c = 32

# Simulate two scenarios where the same request could get different splits:
# Scenario 1: small batch (num_seq=1), more splits per sequence
# Scenario 2: large batch (num_seq=8), fewer splits per sequence

# Compute realistic split counts for both
sl_alone = [seq_len_c]
sl_mixed = [seq_len_c, 512, 1024, 1500, 800, 600, 1200, 1800]

dyn_alone = get_num_kv_splits_dynamic(sl_alone, num_heads_c, num_heads_c, max_splits_c, 132)
dyn_mixed = get_num_kv_splits_dynamic(sl_mixed, num_heads_c, num_heads_c, max_splits_c, 132)

det_alone = get_num_kv_splits_deterministic(sl_alone, 256)
det_mixed = get_num_kv_splits_deterministic(sl_mixed, 256)

print(f"seq_len={seq_len_c} num_heads={num_heads_c} head_dim={head_dim_c} max_splits={max_splits_c}")
print(f"  Deterministic splits: alone={det_alone[0]} mixed={det_mixed[0]}  (same: {det_alone[0]==det_mixed[0]})")
print(f"  Dynamic splits:      alone={dyn_alone[0]} mixed={dyn_mixed[0]}  (same: {dyn_alone[0]==dyn_mixed[0]})")

# Run attention with the split counts we got
Q_c = rbf((1, num_heads_c, head_dim_c), seed=500)
K_c = rbf((seq_len_c, num_heads_c, head_dim_c), seed=600)
V_c = rbf((seq_len_c, num_heads_c, head_dim_c), seed=700)
kv_indptr_c = torch.tensor([0, seq_len_c], dtype=torch.int32, device=device)
kv_indices_c = torch.arange(seq_len_c, dtype=torch.int64, device=device)

results_c = {}
for mode, splits_val in [("det_alone", det_alone[0]), ("det_mixed", det_mixed[0]),
                          ("dyn_alone", dyn_alone[0]), ("dyn_mixed", dyn_mixed[0])]:
    o = torch.empty(1, num_heads_c, head_dim_c, dtype=torch.bfloat16, device=device)
    al = torch.empty(1, num_heads_c, max_splits_c, head_dim_c, dtype=torch.float32, device=device)
    lse = torch.empty(1, num_heads_c, max_splits_c, dtype=torch.float32, device=device)
    nks = torch.full((1,), splits_val, dtype=torch.int32, device=device)

    decode_attention_fwd(Q_c, K_c, V_c, o, kv_indptr_c, kv_indices_c,
                         al, lse, nks, max_splits_c,
                         sm_scale=head_dim_c**-0.5, k_scale=1.0, v_scale=1.0,
                         logit_cap=0, sinks=None, xai_temperature_len=0,
                         has_mla=False, use_pdl=False)
    results_c[mode] = o.clone()

torch.cuda.synchronize()

print(f"\nAttention output comparisons:")
print(f"  det_alone vs det_mixed (same splits={det_alone[0]}):  "
      f"equal={torch.equal(results_c['det_alone'], results_c['det_mixed'])}  "
      f"max_abs={max_abs_diff(results_c['det_alone'], results_c['det_mixed']):.6e}")

dyn_same = dyn_alone[0] == dyn_mixed[0]
print(f"  dyn_alone vs dyn_mixed (splits={dyn_alone[0]} vs {dyn_mixed[0]}, same={dyn_same}):  "
      f"equal={torch.equal(results_c['dyn_alone'], results_c['dyn_mixed'])}  "
      f"max_abs={max_abs_diff(results_c['dyn_alone'], results_c['dyn_mixed']):.6e}")

print(f"  det_alone vs dyn_alone (splits={det_alone[0]} vs {dyn_alone[0]}):  "
      f"equal={torch.equal(results_c['det_alone'], results_c['dyn_alone'])}  "
      f"max_abs={max_abs_diff(results_c['det_alone'], results_c['dyn_alone']):.6e}")

# Check if argmax differs
for label1, label2 in [("det_alone", "dyn_alone"), ("dyn_alone", "dyn_mixed")]:
    if not torch.equal(results_c[label1], results_c[label2]):
        o1 = results_c[label1].float()
        o2 = results_c[label2].float()
        am1 = o1.reshape(-1).argmax()
        am2 = o2.reshape(-1).argmax()
        diff = o1 - o2
        print(f"\n  {label1} vs {label2}:")
        print(f"    max element diff: {diff.abs().max().item():.6e}")
        print(f"    argmax1={am1.item()} argmax2={am2.item()} {'DIFFERS!' if am1 != am2 else 'same'}")

# ============================================================
# TEST D: Multi-layer cumulative effect — attention output fed to next layer
# ============================================================
print("\n" + "=" * 80)
print("TEST D: Cumulative effect — attention diff amplified through 2 layers")
print("  Simulate attention output fed to next layer's matmul to show amplification")
print("=" * 80)

# If attention outputs differ, feed both through a typical FFN projection
# to see if differences amplify
if not torch.equal(results_c['det_alone'], results_c['dyn_alone']):
    attn_det = results_c['det_alone'].float()    # [1, 16, 128]
    attn_dyn = results_c['dyn_alone'].float()

    # Simulate next-layer projection: attn_out @ W_proj.T
    W_proj = rbf((head_dim_c, 2048), seed=900).float()  # projects to hidden_dim=2048

    # Compute projection
    proj_det = attn_det.reshape(1, -1) @ W_proj   # [1, 2048]
    proj_dyn = attn_dyn.reshape(1, -1) @ W_proj

    print(f"  Attention output difference: max_abs={max_abs_diff(attn_det, attn_dyn):.6e}")
    print(f"  After FFN projection:        max_abs={max_abs_diff(proj_det, proj_dyn):.6e}")

    # Simulate rms_norm
    w_norm = rbf((2048,), seed=950).float()
    norm_det = torch.nn.functional.rms_norm(proj_det, [2048], weight=w_norm)
    norm_dyn = torch.nn.functional.rms_norm(proj_dyn, [2048], weight=w_norm)
    print(f"  After RMS norm:              max_abs={max_abs_diff(norm_det, norm_dyn):.6e}")
    print(f"  Amplification factor:        {max_abs_diff(norm_det, norm_dyn)/max(1e-12, max_abs_diff(attn_det, attn_dyn)):.2f}x")

    # Argmax check after 1 layer of processing
    am_norm_det = norm_det.reshape(-1).argmax()
    am_norm_dyn = norm_dyn.reshape(-1).argmax()
    print(f"  After 1 more layer argmax:   det={am_norm_det.item()} dyn={am_norm_dyn.item()} "
          f"{'DIFFERS!' if am_norm_det != am_norm_dyn else 'same'}")

# ============================================================
# TEST E: Sweep across many seq_lens to find where splits differ
# ============================================================
print("\n" + "=" * 80)
print("TEST E: Sweep — which seq_lens cause split-count divergence?")
print("=" * 80)

diverged = []
for sl in [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]:
    alone_sl = [sl]
    mixed_sl = [sl, 256, 512, 1024, 2048, 100, 300, 700, 1500]
    dyn_a = get_num_kv_splits_dynamic(alone_sl, 16, 16, 32, 132)[0]
    dyn_m = get_num_kv_splits_dynamic(mixed_sl, 16, 16, 32, 132)[0]
    det_a = get_num_kv_splits_deterministic(alone_sl, 256)[0]
    det_m = get_num_kv_splits_deterministic(mixed_sl, 256)[0]
    mark = ""
    if dyn_a != dyn_m:
        mark = " <-- batch-dependent!"
        diverged.append(sl)
    print(f"  sl={sl:5d}  det(alone={det_a:2d} mixed={det_m:2d})  dyn(alone={dyn_a:2d} mixed={dyn_m:2d}){mark}")

print(f"\n  Batch-dependent seq_lens: {diverged if diverged else 'none'}")
print(f"  Deterministic: ALWAYS same (seq_len-based)")

# ============================================================
# CONCLUSION
# ============================================================
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
