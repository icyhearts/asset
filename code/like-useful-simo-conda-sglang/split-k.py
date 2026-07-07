import os
import sys
import torch
import math

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
from sglang.srt.batch_invariant_ops.batch_invariant_ops import matmul_persistent

def rbf(shape, seed=42, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, device="cuda", generator=g, dtype=torch.float32)*scale).bfloat16()

# ---------- helpers ----------

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

K, N = 10944, 4096  # typical SGLang linear-layer shape (up-proj)
test_configs = [
    (6, 256),      # small batch, padded to 256
]

for row_count, total_M in test_configs:
    # Same target rows and weight matrix for both paths
    A_rows = rbf((row_count, K), seed=42)
    B_fixed = rbf((K, N), seed=43)

    A_pad = rbf((total_M - row_count, K), seed=44)
    A_large = torch.cat([A_rows, A_pad], dim=0)

    # Small batch: only the target rows
    out_small = torch.mm(A_rows, B_fixed)

    out_small_mp = matmul_persistent(A_rows, B_fixed)

    out_large_full = torch.mm(A_large, B_fixed)
    out_large_full_mp = matmul_persistent(A_large, B_fixed)
