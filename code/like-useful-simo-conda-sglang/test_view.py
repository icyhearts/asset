
from __future__ import annotations

from dataclasses import dataclass

import torch
from simo.ops.formats.mx.quant import _downcast_to_mxfmt_torch, _upcast_from_mxfmt_torch
from simo.ops.formats.mx.scale import (
  ObserverMode,
  ScaleModeEnum,
  calculate_mx_scale,
)
from simo.quantization import dtypes
from simo.quantization.utils import transform_to_block_wise

