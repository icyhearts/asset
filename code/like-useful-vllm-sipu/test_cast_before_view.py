#!/usr/bin/env python3
"""Show why MX E8M0 scales must be cast to float32 before ``view(int32)``.

The old implementation from before simo commit
83dbdd1bc048810ba6a5206cad2c250d18cc96d4 used::

    (blocked_scale.view(torch.int32) >> 23).to(torch.uint8)

This script calls the torch implementation directly, reconstructs the scale
tensor immediately before serialization, and compares that old expression
with the fixed cast-first expression.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from simo.ops.formats.mx.quant import _downcast_to_mxfmt_torch
from simo.ops.formats.mx.scale import (
  ObserverMode,
  ScaleModeEnum,
  calculate_mx_scale,
)
from simo.quantization import dtypes
from simo.quantization.utils import transform_to_block_wise


@dataclass(frozen=True)
class Case:
  name: str
  shape: tuple[int, int]
  input_dtype: torch.dtype
  scale_mode: ScaleModeEnum
  block_size: int = 32
  expect_legacy_error: bool = False
  expect_legacy_shape_bug: bool = False


def _make_input(
  shape: tuple[int, int], dtype: torch.dtype, block_size: int
) -> torch.Tensor:
  # Give neighboring blocks different powers of two.  This makes both the
  # missing elements and the corrupted legacy bit patterns visible in output.
  rows = torch.arange(shape[0], dtype=torch.float32).unsqueeze(1)
  cols = torch.arange(shape[1], dtype=torch.float32).unsqueeze(0)
  block_ids = torch.floor(cols / block_size)
  exponents = torch.remainder(rows + block_ids, 8.0) - 4.0
  within_block = 1.0 + torch.remainder(cols, block_size) / block_size
  values = within_block * torch.pow(2.0, exponents)
  return values.to(dtype)


def _scale_before_serialization(
  src_tensor: torch.Tensor,
  block_size: int,
  scale_mode: ScaleModeEnum,
) -> torch.Tensor:
  """Reproduce ``scale_bw.squeeze(-1)`` just before the changed line."""
  blocked = transform_to_block_wise(src_tensor.contiguous(), block_size, axis=-1)
  scale_bw = calculate_mx_scale(
    blocked,
    dtypes.mxfp4_e2m1,
    scale_mode,
  )
  return scale_bw.squeeze(-1)


def _legacy_serialize(blocked_scale: torch.Tensor) -> torch.Tensor:
  """The expression used before commit 83dbdd1."""
  return (blocked_scale.view(torch.int32) >> 23).to(torch.uint8)


def _fixed_serialize(blocked_scale: torch.Tensor) -> torch.Tensor:
  """The expression introduced by commit 83dbdd1."""
  return (blocked_scale.to(torch.float32).view(torch.int32) >> 23).to(torch.uint8)


def _call_downcast(
  src_tensor: torch.Tensor,
  scale_mode: ScaleModeEnum,
  block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  # This is intentionally the low-level torch implementation, rather than
  # torch.ops.simo.downcast_to_mxfmt or the Triton implementation.
  # Some simo revisions decorate this function with torch.compile; unwrap only
  # that decorator so the experiment still exercises the same Python body.
  downcast_impl = getattr(_downcast_to_mxfmt_torch, "__wrapped__", _downcast_to_mxfmt_torch)
  return downcast_impl(
    src_tensor=src_tensor,
    dtype=dtypes.mxfp4_e2m1,
    axis=-1,
    block_size=block_size,
    observer_mode=ObserverMode.ABS_MAX_OBSERVER_MODE,
    quant_scale_rounding_mode=scale_mode,
  )


def _preview(tensor: torch.Tensor, limit: int = 8) -> list[object]:
  return tensor.reshape(-1)[:limit].tolist()


def run_case(case: Case) -> None:
  src = _make_input(case.shape, case.input_dtype, case.block_size)
  blocked_scale = _scale_before_serialization(src, case.block_size, case.scale_mode)

  quantized, returned_scale = _call_downcast(src, case.scale_mode, case.block_size)
  expected_scale = _fixed_serialize(blocked_scale)

  assert returned_scale.dtype == torch.uint8
  assert tuple(returned_scale.shape) == tuple(blocked_scale.shape)
  torch.testing.assert_close(returned_scale, expected_scale, rtol=0, atol=0)

  print(f"\nCASE: {case.name}")
  print(f"  input:          dtype={src.dtype}, shape={tuple(src.shape)}")
  print(
    "  blocked_scale:  "
    f"dtype={blocked_scale.dtype}, shape={tuple(blocked_scale.shape)}, "
    f"values={_preview(blocked_scale)}"
  )
  print(
    "  fixed result:   "
    f"quantized_shape={tuple(quantized.shape)}, "
    f"scale_dtype={returned_scale.dtype}, scale_shape={tuple(returned_scale.shape)}, "
    f"values={_preview(returned_scale)}"
  )

  try:
    legacy_scale = _legacy_serialize(blocked_scale)
  except RuntimeError as error:
    print(f"  legacy result:  ERROR ({type(error).__name__}: {error})")
    assert case.expect_legacy_error, "unexpected legacy view error"
    return

  print(
    "  legacy result:  "
    f"dtype={legacy_scale.dtype}, shape={tuple(legacy_scale.shape)}, "
    f"values={_preview(legacy_scale)}"
  )
  if case.expect_legacy_shape_bug:
    assert tuple(legacy_scale.shape) != tuple(returned_scale.shape)
    assert legacy_scale.numel() != returned_scale.numel()
    print("  verdict:        old view loses scale elements; fixed result keeps all blocks")
  else:
    torch.testing.assert_close(legacy_scale, returned_scale, rtol=0, atol=0)
    print("  verdict:        legacy and fixed serialization agree for this control case")


def main() -> int:
  cases = (
    Case(
      "float32 + E8M0_SIPU (control)",
      shape=(64, 64),
      input_dtype=torch.float32,
      scale_mode=ScaleModeEnum.E8M0_SIPU,
    ),
    Case(
      "bf16 + E8M0_SIPU (512x512 trigger)",
      shape=(512, 512),
      input_dtype=torch.bfloat16,
      scale_mode=ScaleModeEnum.E8M0_SIPU,
      expect_legacy_shape_bug=True,
    ),
    Case(
      "fp16 + E8M0_RCEIL (512x512 trigger)",
      shape=(512, 512),
      input_dtype=torch.float16,
      scale_mode=ScaleModeEnum.E8M0_RCEIL,
      expect_legacy_shape_bug=True,
    ),
    Case(
      "bf16 + E8M0_FLOOR (scale helper promotes to float32)",
      shape=(64, 64),
      input_dtype=torch.bfloat16,
      scale_mode=ScaleModeEnum.E8M0_FLOOR,
    ),
    Case(
      "bf16 + E8M0_SIPU (one block; old view cannot reinterpret)",
      shape=(1, 32),
      input_dtype=torch.bfloat16,
      scale_mode=ScaleModeEnum.E8M0_SIPU,
      expect_legacy_error=True,
    ),
  )

  for case in cases:
    run_case(case)

  print("\nAll cast-before-view examples passed.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
