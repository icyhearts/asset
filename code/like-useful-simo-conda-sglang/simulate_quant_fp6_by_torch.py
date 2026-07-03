#!/usr/bin/env python3
"""PyTorch equivalent of the MXFP6_E2M3 quantization kernel.

Converts lines 438-488 of simo/ops/kernels/downcast/_downcast_to_mxfmt.py
into pure PyTorch operations.

The kernel quantizes FP32 values to MXFP6_E2M3 (2 exponent bits + 3 mantissa bits
= 5 magnitude + 1 sign = 6 bits total), using block-wise shared exponent (E8M0 scale).

MXFP6_E2M3 magnitude bit table (5 bits):
  Bits   Biased Exp   Value (× 2^block_scale)
  00000      0         ±0
  00001      0         Denormal: 0.125 = 1/8
  00010      0         Denormal: 0.250 = 2/8
  00011      0         Denormal: 0.375 = 3/8
  00100      0         Denormal: 0.500 = 4/8
  00101      0         Denormal: 0.625 = 5/8
  00110      0         Denormal: 0.750 = 6/8
  00111      0         Denormal: 0.875 = 7/8
  01000      1         Normal:   1.000
  01001      1         Normal:   1.125
  01010      1         Normal:   1.250
  01011      1         Normal:   1.375
  01100      1         Normal:   1.500
  01101      1         Normal:   1.625
  01110      1         Normal:   1.750
  01111      1         Normal:   1.875
  10000      2         Normal:   2.000
  10001      2         Normal:   2.250
  10010      2         Normal:   2.500
  10011      2         Normal:   2.750
  10100      2         Normal:   3.000
  10101      2         Normal:   3.250
  10110      2         Normal:   3.500
  10111      2         Normal:   3.750
  11000      3         Normal:   4.000
  11001      3         Normal:   4.500
  11010      3         Normal:   5.000
  11011      3         Normal:   5.500
  11100      3         Normal:   6.000
  11101      3         Normal:   6.500
  11110      3         Normal:   7.000
  11111      -         Saturation: 7.500

Usage:
    python like-useful/simulate_quant_fp6_by_torch.py
"""

import torch


# ---------------------------------------------------------------------------
# MXFP6_E2M3 format constants
# ---------------------------------------------------------------------------
EBITS = 2
MBITS = 3
MAGNITUDE_BITS = EBITS + MBITS  # = 5
MAX_NORMAL = 7.5
MIN_NORMAL = 1.0
SATURATED_VAL = (1 << MAGNITUDE_BITS) - 1  # = 31

F32_MBITS = 23
F32_EXP_BIAS = 127
EXP_BIAS = (1 << (EBITS - 1)) - 1  # = 1

SIGN_MASK = 1 << MAGNITUDE_BITS  # = 0x20
SHIFT_AMOUNT = 31 - MAGNITUDE_BITS  # = 26


# ---------------------------------------------------------------------------
# Pure PyTorch (no CUDA) quantizer — equivalent of lines 438-488
# ---------------------------------------------------------------------------
def quantize_mxfp6_e2m3(x: torch.Tensor) -> torch.Tensor:
    """Quantize FP32 tensor to MXFP6_E2M3 (6-bit) unsigned magnitude + sign bit.

    Returns uint8 tensor where each element is [sign bit(1b)][magnitude(5b)]
    with upper 2 bits set to 0.
    """
    # --- Get sign and absolute value ---
    # Truncate x to fp32 (ensures fp32 bit patterns, not bf16 truncated)
    x = x.to(torch.float32)
    x_u32 = x.view(torch.int32)
    signs = x_u32 & 0x80000000  # extract sign bit (int32)

    x_abs = x.abs()

    # --- Create masks ---
    saturate_mask = x_abs >= MAX_NORMAL
    zero_mask = x_abs == 0.0
    denormal_mask = (x_abs < MIN_NORMAL) & ~saturate_mask & ~zero_mask
    normal_mask = ~saturate_mask & ~denormal_mask & ~zero_mask

    # --- Branch 2: Denormal ---
    # denorm_exp = (127 - 1) + (23 - 3) + 1 = 126 + 20 + 1 = 147
    # denorm_mask_int = 147 << 23 = 0x4B000000 (FP32 representation of 2^20)
    denorm_exp = (F32_EXP_BIAS - EXP_BIAS) + (F32_MBITS - MBITS) + 1
    denorm_mask_int = denorm_exp << F32_MBITS  # = 147 << 23
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(torch.float32)

    # x_pos + 2^20 → float add aligns ULP, then bitcast back to int
    denormal_x = (x_abs + denorm_mask_float).view(torch.int32)
    denormal_x = denormal_x - denorm_mask_int  # remove the bias

    # --- Branch 3: Normal ---
    # magic_adder = (1 << (23 - 3 - 1)) - 1 = (1 << 19) - 1 = 0x7FFFF
    magic_adder = (1 << (F32_MBITS - MBITS - 1)) - 1
    # val_to_add = (1 - 127) << 23 + magic_adder = -126 * 2^23 + 0x7FFFF
    val_to_add = ((EXP_BIAS - F32_EXP_BIAS) << F32_MBITS) + magic_adder

    x_abs_u32 = x_abs.view(torch.int32)
    mant_odd = (x_abs_u32 >> (F32_MBITS - MBITS)) & 1  # bit 20 of the mantissa

    normal_x_u32 = x_abs_u32 + val_to_add  # uint32 add: bias adjust + round
    normal_x_u32 = normal_x_u32 + mant_odd  # tie-to-even correction
    normal_vals = normal_x_u32 >> (F32_MBITS - MBITS)  # shift right by 20

    # --- Combine branches ---
    mx_val = torch.where(zero_mask, torch.zeros_like(normal_vals), normal_vals)
    mx_val = torch.where(saturate_mask,
                        torch.full_like(mx_val, SATURATED_VAL),
                        mx_val)
    mx_val = torch.where(denormal_mask, denormal_x, mx_val)

    # --- Add sign bit ---
    # Triton does: final_sign_bit = (signs >> 26) & 0x20
    # On CPU, torch.uint32 doesn't support right-shift, so use alternative:
    # signs is 0x80000000 for negative values, 0 for positive/zero
    sign_bit = (signs != 0).to(torch.int32)          # 1 for neg, 0 for non-neg
    final_sign_bit = (sign_bit << MAGNITUDE_BITS) & SIGN_MASK  # 0x20 or 0x00

    quant_values = mx_val.to(torch.uint8) | final_sign_bit.to(torch.uint8)

    return quant_values


# ---------------------------------------------------------------------------
# Dequantizer: convert 6-bit quantized value back to FP32
# ---------------------------------------------------------------------------
def dequantize_mxfp6_e2m3(quant_values: torch.Tensor, block_scale: torch.Tensor = None) -> torch.Tensor:
    """Convert MXFP6_E2M3 quantized values back to FP32.

    Args:
        quant_values: uint8 tensor, each element = [sign(1b)][magnitude(5b)]
        block_scale: optional float32 scale (E8M0 exponent applied as 2^exp)

    Returns:
        FP32 float tensor
    """
    q = quant_values.to(torch.int32)
    sign_bit = (q & SIGN_MASK) != 0
    magnitude = q & (SATURATED_VAL)  # lower 5 bits

    # Decode the 5 magnitude bits
    exp = magnitude >> MBITS  # upper 2 bits = exponent
    mant = magnitude & ((1 << MBITS) - 1)  # lower 3 bits = mantissa

    # Denormal: exp = 0, value = mant * 2^-3 = mant * 0.125
    denormal_val = mant.to(torch.float32) * 0.125
    # Normal: exp > 0, value = 2^(exp-1) * (1 + mant/8)
    normal_val = (2.0 ** (exp.to(torch.float32) - 1.0)) * (1.0 + mant.to(torch.float32) / 8.0)

    values = torch.where(exp == 0, denormal_val, normal_val)
    values = torch.where(sign_bit, -values, values)

    if block_scale is not None:
        values = values * (2.0 ** block_scale.to(torch.float32))

    return values


# ===========================================================================
# Comprehensive test inputs
# ===========================================================================
def run_tests():
    print("=" * 80)
    print("MXFP6_E2M3  Quantization Kernel — PyTorch Equivalent")
    print("=" * 80)

    # Build test inputs covering:
    #   1. All 5 magnitude bits (0b00000 .. 0b11111)
    #   2. FP32 normal / denormal / Inf / NaN / 0.0
    #   3. Boundary values (exactly at 0.0, 1.0, 7.5)
    #   4. Round-to-nearest-even tie cases

    # Assume block_scale = 1.0 (no scaling) for simplicity
    block_scale = torch.tensor([0.0])  # E8M0 exponent 0 => scale = 2^0 = 1

    test_inputs = [
        # ---- FP32 special values ----
        (0.0,          "Zero (0.0)"),
        (-0.0,         "Negative zero (-0.0)"),
        (float('inf'), "Positive Infinity"),
        (-float('inf'),"Negative Infinity"),
        (float('nan'), "NaN"),

        # ---- Denormal range (0 < x < 1.0) ----
        # Exact representable denormals
        (0.125,        "Denormal: 0.125 = 1/8   (mag=0b00001)"),
        (0.25,         "Denormal: 0.250 = 2/8   (mag=0b00010)"),
        (0.375,        "Denormal: 0.375 = 3/8   (mag=0b00011)"),
        (0.5,          "Denormal: 0.500 = 4/8   (mag=0b00100)"),
        (0.625,        "Denormal: 0.625 = 5/8   (mag=0b00101)"),
        (0.75,         "Denormal: 0.750 = 6/8   (mag=0b00110)"),
        (0.875,        "Denormal: 0.875 = 7/8   (mag=0b00111)"),
        # Non-exact denormals
        (0.1,          "Denormal: 0.1 → round to 0.125"),
        (0.3,          "Denormal: 0.3 → round to 0.250"),

        # ---- Normal range (1.0 <= x < 7.5) ----
        # exp=1 normals (0b01000 .. 0b01111)
        (1.0,          "Normal: 1.000  (exp=1, mag=0b01000)"),
        (1.125,        "Normal: 1.125  (exp=1, mag=0b01001)"),
        (1.25,         "Normal: 1.250  (exp=1, mag=0b01010)"),
        (1.375,        "Normal: 1.375  (exp=1, mag=0b01011)"),
        (1.5,          "Normal: 1.500  (exp=1, mag=0b01100)"),
        (1.625,        "Normal: 1.625  (exp=1, mag=0b01101)"),
        (1.75,         "Normal: 1.750  (exp=1, mag=0b01110)"),
        (1.875,        "Normal: 1.875  (exp=1, mag=0b01111)"),
        # exp=2 normals (0b10000 .. 0b10111)
        (2.0,          "Normal: 2.000  (exp=2, mag=0b10000)"),
        (2.25,         "Normal: 2.250  (exp=2, mag=0b10001)"),
        (2.5,          "Normal: 2.500  (exp=2, mag=0b10010)"),
        (2.75,         "Normal: 2.750  (exp=2, mag=0b10011)"),
        (3.0,          "Normal: 3.000  (exp=2, mag=0b10100)"),
        (3.25,         "Normal: 3.250  (exp=2, mag=0b10101)"),
        (3.5,          "Normal: 3.500  (exp=2, mag=0b10110)"),
        (3.75,         "Normal: 3.750  (exp=2, mag=0b10111)"),
        # exp=3 normals (0b11000 .. 0b11110)
        (4.0,          "Normal: 4.000  (exp=3, mag=0b11000)"),
        (4.5,          "Normal: 4.500  (exp=3, mag=0b11001)"),
        (5.0,          "Normal: 5.000  (exp=3, mag=0b11010)"),
        (5.5,          "Normal: 5.500  (exp=3, mag=0b11011)"),
        (6.0,          "Normal: 6.000  (exp=3, mag=0b11100)"),
        (6.5,          "Normal: 6.500  (exp=3, mag=0b11101)"),
        (7.0,          "Normal: 7.000  (exp=3, mag=0b11110)"),

        # ---- Round-to-nearest-even cases (ties) ----
        (1.0625,       "Normal tie: 1.0625 → round to 1.000 (1.000 gap=0.0625, 1.125 gap=0.0625, mant LSB=0→even)"),
        (1.1875,       "Normal tie: 1.1875 → round to 1.250 (1.125 gap=0.0625, 1.250 gap=0.0625, mant LSB of 1.250=0→even)"),
        (2.375,        "Normal tie: 2.375 → round to 2.250 or 2.500 (test tie-to-even)"),

        # ---- Saturate range (x >= 7.5) ----
        (7.5,          "Saturate: 7.5  (max representable, mag=0b11111)"),
        (8.0,          "Saturate: 8.0  → 7.5"),
        (10.0,         "Saturate: 10.0 → 7.5"),
        (100.0,        "Saturate: 100.0 → 7.5"),

        # ---- Negative values ----
        (-1.5,         "Negative: -1.5"),
        (-6.0,         "Negative: -6.0"),
        (-7.5,         "Negative: -7.5"),
        (-10.0,        "Saturate negative: -10.0 → -7.5"),

        # ---- Near boundary values ----
        (0.999,        "Near min_normal (below): 0.999 → denormal"),
        (1.001,        "Near min_normal (above): 1.001 → normal 1.0"),
        (7.499,        "Near max_normal (below): 7.499 → 7.0 or 7.5?"),
        (7.501,        "Near max_normal (above): 7.501 → saturate 7.5"),

        # ---- Small subnormals ----
        (0.0625,       "Very small denormal: quantizes to 0 (mant=0, <0.0625)"),
        (0.01,         "Very small denormal: 0.01 → 0"),
    ]

    # Combine into a single tensor
    values = torch.tensor([t[0] for t in test_inputs], dtype=torch.float32)
    labels = [t[1] for t in test_inputs]

    # Quantize
    quant = quantize_mxfp6_e2m3(values)

    # Dequantize
    dequant = dequantize_mxfp6_e2m3(quant, block_scale.expand(len(values)))

    # Display results
    print(f"\n{'#':>3s}  {'Input':>12s}  {'Quant(hex)':>10s}  {'Quant(bin)':>10s}  {'Dequant':>12s}  {'Error':>12s}  Description")
    print("-" * 120)

    for i, (val, label) in enumerate(zip(values, test_inputs)):
        v = val.item()
        q = quant[i].item()
        dq = dequant[i].item()

        # NaN cases
        if v != v:  # NaN
            q_str = f"0x{q:02x}"
            b_str = f"0b{q:08b}"
            dq_str = f"{dq:>12g}"
            err_str = "N/A"
        elif abs(v) == float('inf'):
            q_str = f"0x{q:02x}"
            b_str = f"0b{q:08b}"
            dq_str = f"{dq:>12g}"
            err_str = "N/A"
        else:
            q_str = f"0x{q:02x}"
            b_str = f"0b{q:08b}"
            dq_str = f"{dq:>12.6f}"
            err_str = f"{abs(v - dq):>12.6e}"

        print(f"{i:3d}  {v:>12g}  {q_str:>10s}  {b_str:>10s}  {dq_str:>12s}  {err_str:>12s}  {label[1]}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary: All 32 magnitude values (0b00000 .. 0b11111) of MXFP6_E2M3")
    print("have been covered by the test inputs above.")
    print("=" * 80)

    # ---- Verify quantization mapping for exact representable values ----
    print("\nVerification: Exact representable MXFP6_E2M3 values (unsigned magnitude part)")
    print("-" * 80)
    print(f"{'Magnitude':>12s}  {'Expected FP32':>14s}  {'Test input v':>14s}  {'Dequant v':>14s}  {'Match':>8s}")
    print("-" * 80)

    exact_cases = [
        (0, 0.0, "0"),
        (1, 0.125, "1/8"),
        (2, 0.250, "2/8"),
        (3, 0.375, "3/8"),
        (4, 0.500, "4/8"),
        (5, 0.625, "5/8"),
        (6, 0.750, "6/8"),
        (7, 0.875, "7/8"),
        (8, 1.000, "1 * 2^0 = 1.0"),
        (9, 1.125, "1.125 * 2^0"),
        (10, 1.250, "1.250 * 2^0"),
        (11, 1.375, "1.375 * 2^0"),
        (12, 1.500, "1.500 * 2^0"),
        (13, 1.625, "1.625 * 2^0"),
        (14, 1.750, "1.750 * 2^0"),
        (15, 1.875, "1.875 * 2^0"),
        (16, 2.000, "1 * 2^1 = 2.0"),
        (17, 2.250, "1.125 * 2^1"),
        (18, 2.500, "1.250 * 2^1"),
        (19, 2.750, "1.375 * 2^1"),
        (20, 3.000, "1.500 * 2^1"),
        (21, 3.250, "1.625 * 2^1"),
        (22, 3.500, "1.750 * 2^1"),
        (23, 3.750, "1.875 * 2^1"),
        (24, 4.000, "1 * 2^2 = 4.0"),
        (25, 4.500, "1.125 * 2^2"),
        (26, 5.000, "1.250 * 2^2"),
        (27, 5.500, "1.375 * 2^2"),
        (28, 6.000, "1.500 * 2^2"),
        (29, 6.500, "1.625 * 2^2"),
        (30, 7.000, "1.750 * 2^2"),
        (31, 7.500, "Saturation [sic]"),  # actually this is the overflow all-ones value
    ]

    for mag, expected_val, note in exact_cases:
        # Find the corresponding test result by rounding to the nearest input
        # Note: all exact representable values are in our test inputs
        test_v = torch.tensor(expected_val, dtype=torch.float32)
        q = quantize_mxfp6_e2m3(test_v)
        dq = dequantize_mxfp6_e2m3(q, torch.tensor([0.0])).item()
        match = "YES" if abs(dq - expected_val) < 1e-6 or (mag == 31 and dq == 7.5) else "NO"
        print(f"  {mag:>3d} (0b{mag:05b})  {expected_val:>14.6f}  {test_v.item():>14.6f}  {dq:>14.6f}  {match:>8s}  ({note})")


if __name__ == "__main__":
    run_tests()
