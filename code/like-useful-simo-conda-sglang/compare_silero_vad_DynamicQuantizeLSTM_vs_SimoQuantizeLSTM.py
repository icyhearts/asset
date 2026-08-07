#!/usr/bin/env python3
"""Compare ORT and SIMO LSTM quantization with real Silero VAD weights.

The source model is constant-folded first. Every nested LSTM is then copied to
an otherwise empty standalone model, preserving its real W/R/B initializers and
attributes. The script compares these execution paths with identical inputs:

* float LSTM on CUDA (the baseline),
* float LSTM on CPU (a backend-difference control),
* ORT DynamicQuantizeLSTM, QInt8 per-channel, on CPU, and
* com.simo::SimoQuantizeLSTM with the supplied SIMO config on CUDA.

Both independent same-input cases and a closed-loop streaming rollout are run.
The latter feeds each implementation's Y_h/Y_c back into its next invocation,
matching Silero VAD's recurrent-state usage. A SIMO Torch reference independently
rolls out the same recurrence and keeps the cell state in floating point.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterator, Sequence

import numpy as np
import onnx
from onnx import AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto
from onnx import numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

from simo.onnx.api import quantize as simo_quantize
from simo.onnx.runtime import get_custom_ops_library_path, register_custom_ops


DEFAULT_INPUT_MODEL = Path(
  "/share/users/like/package/jdjv/silero_vad_clean/onnx_float_baseline/silero_vad.onnx"
)
DEFAULT_CONFIG = Path(
  "/share/users/like/package/jdjv/quant_schema_only_lstm/"
  "w8a8_int8/w_int8_per_channel_a_int8_per_channel.json"
)
DEFAULT_OUTPUT_DIR = Path("temp/compare-silero-vad-lstm")
OUTPUT_NAMES = ("Y", "Y_h", "Y_c")
SIMO_REFERENCE_ROLLOUT_RELATIVE_L2_TOLERANCE = 1e-2


@dataclass(frozen=True)
class LSTMRecord:
  index: int
  graph_path: str
  node: NodeProto
  weight: np.ndarray
  recurrent: np.ndarray
  bias: np.ndarray | None

  @property
  def direction(self) -> str:
    value = _node_attributes(self.node).get("direction", b"forward")
    return value.decode() if isinstance(value, bytes) else str(value)

  @property
  def num_directions(self) -> int:
    return 2 if self.direction == "bidirectional" else 1

  @property
  def hidden_size(self) -> int:
    return int(_node_attributes(self.node)["hidden_size"])

  @property
  def input_size(self) -> int:
    return int(self.weight.shape[2])

  @property
  def label(self) -> str:
    return self.node.name or f"LSTM_{self.index}"


@dataclass(frozen=True)
class ModelPaths:
  float_model: Path
  dynamic_model: Path
  simo_model: Path


@dataclass(frozen=True)
class ErrorMetrics:
  cosine_similarity: float
  relative_l2_error: float
  mean_absolute_error: float
  max_absolute_error: float
  reference_l2_norm: float


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Compare DynamicQuantizeLSTM and SimoQuantizeLSTM on real Silero weights."
  )
  parser.add_argument(
    "--input-model",
    type=Path,
    default=DEFAULT_INPUT_MODEL,
    help=f"source Silero ONNX model (default: {DEFAULT_INPUT_MODEL})",
  )
  parser.add_argument(
    "--config",
    type=Path,
    default=DEFAULT_CONFIG,
    help=f"SIMO LSTM-only quantization config (default: {DEFAULT_CONFIG})",
  )
  parser.add_argument(
    "--custom-op-library",
    type=Path,
    default=None,
    help="Simo custom-op library (default: simo.onnx runtime resolution)",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR,
    help=f"model and JSON output directory (default: {DEFAULT_OUTPUT_DIR})",
  )
  parser.add_argument("--seed", type=int, default=20260805)
  parser.add_argument("--cases", type=int, default=16)
  parser.add_argument("--rollout-steps", type=int, default=256)
  parser.add_argument("--sequence-length", type=int, default=1)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--x-std", type=float, default=0.5)
  parser.add_argument(
    "--rollout-x-stds",
    type=float,
    nargs="+",
    default=[0.01, 0.1, 0.5],
    help="random X standard deviations used for closed-loop streaming rollouts",
  )
  parser.add_argument("--h-std", type=float, default=0.25)
  parser.add_argument("--c-std", type=float, default=0.5)
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="replace artifacts previously produced in the output directory",
  )
  args = parser.parse_args()
  for name in ("cases", "rollout_steps", "sequence_length", "batch_size"):
    if getattr(args, name) <= 0:
      parser.error(f"--{name.replace('_', '-')} must be positive")
  for name in ("x_std", "h_std", "c_std"):
    value = getattr(args, name)
    if not math.isfinite(value) or value < 0:
      parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
  if any(not math.isfinite(value) or value < 0 for value in args.rollout_x_stds):
    parser.error("--rollout-x-stds values must be finite and non-negative")
  return args


def _node_attributes(node: NodeProto) -> dict[str, Any]:
  return {
    attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute
  }


def _iter_subgraphs(node: NodeProto) -> Iterator[tuple[str, GraphProto]]:
  for attribute in node.attribute:
    if attribute.type == AttributeProto.GRAPH:
      yield attribute.name, attribute.g
    elif attribute.type == AttributeProto.GRAPHS:
      for index, graph in enumerate(attribute.graphs):
        yield f"{attribute.name}[{index}]", graph


def _constant_tensor(node: NodeProto) -> TensorProto | None:
  if node.op_type != "Constant":
    return None
  return next(
    (attribute.t for attribute in node.attribute if attribute.type == AttributeProto.TENSOR),
    None,
  )


def _tensor_array(constants: dict[str, TensorProto], name: str, role: str) -> np.ndarray:
  tensor = constants.get(name)
  if tensor is None:
    raise RuntimeError(f"LSTM {role} input {name!r} is not a folded constant")
  return np.asarray(numpy_helper.to_array(tensor)).copy()


def collect_lstm_records(model: ModelProto) -> list[LSTMRecord]:
  records: list[LSTMRecord] = []

  def visit_graph(
    graph: GraphProto,
    graph_path: str,
    inherited_constants: dict[str, TensorProto],
  ) -> None:
    constants = dict(inherited_constants)
    constants.update({tensor.name: tensor for tensor in graph.initializer})
    for node in graph.node:
      tensor = _constant_tensor(node)
      if tensor is not None:
        constants.update({output: tensor for output in node.output if output})

    for node in graph.node:
      if node.domain in ("", "ai.onnx") and node.op_type == "LSTM":
        _validate_source_lstm(node)
        weight = _tensor_array(constants, node.input[1], "W")
        recurrent = _tensor_array(constants, node.input[2], "R")
        bias = (
          _tensor_array(constants, node.input[3], "B")
          if len(node.input) > 3 and node.input[3]
          else None
        )
        record = LSTMRecord(
          index=len(records),
          graph_path=graph_path,
          node=node,
          weight=weight,
          recurrent=recurrent,
          bias=bias,
        )
        _validate_record(record)
        records.append(record)

      node_label = node.name or node.op_type
      for attribute_name, subgraph in _iter_subgraphs(node):
        visit_graph(
          subgraph,
          f"{graph_path}/{node_label}:{attribute_name}",
          constants,
        )

  visit_graph(model.graph, "main", {})
  return records


def _validate_source_lstm(node: NodeProto) -> None:
  if len(node.input) < 3 or not node.input[1] or not node.input[2]:
    raise RuntimeError(f"{node.name or '<unnamed LSTM>'}: W and R are required")
  if len(node.input) > 4 and node.input[4]:
    raise RuntimeError(f"{node.name or '<unnamed LSTM>'}: sequence_lens is unsupported")
  if len(node.input) > 7 and node.input[7]:
    raise RuntimeError(f"{node.name or '<unnamed LSTM>'}: peephole P is unsupported")
  attrs = _node_attributes(node)
  if int(attrs.get("layout", 0)) != 0:
    raise RuntimeError(f"{node.name or '<unnamed LSTM>'}: only layout=0 is supported")


def _validate_record(record: LSTMRecord) -> None:
  if record.direction not in {"forward", "reverse", "bidirectional"}:
    raise RuntimeError(f"{record.label}: unsupported direction {record.direction!r}")
  if record.weight.dtype != np.float32 or record.recurrent.dtype != np.float32:
    raise RuntimeError(
      f"{record.label}: this comparison expects float32 W/R, got "
      f"{record.weight.dtype}/{record.recurrent.dtype}"
    )
  directions = record.num_directions
  hidden = record.hidden_size
  expected_w = (directions, 4 * hidden, record.input_size)
  expected_r = (directions, 4 * hidden, hidden)
  if record.weight.shape != expected_w or record.recurrent.shape != expected_r:
    raise RuntimeError(
      f"{record.label}: invalid W/R shapes {record.weight.shape}/{record.recurrent.shape}; "
      f"expected {expected_w}/{expected_r}"
    )
  if record.bias is not None:
    expected_b = (directions, 8 * hidden)
    if record.bias.dtype != np.float32 or record.bias.shape != expected_b:
      raise RuntimeError(
        f"{record.label}: invalid B dtype/shape {record.bias.dtype}/{record.bias.shape}; "
        f"expected float32/{expected_b}"
      )


def _fold_model(input_path: Path, output_path: Path) -> None:
  options = ort.SessionOptions()
  options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
  options.optimized_model_filepath = str(output_path)
  ort.InferenceSession(str(input_path), sess_options=options, providers=["CPUExecutionProvider"])
  if not output_path.is_file():
    raise RuntimeError(f"ONNX Runtime did not write the folded model: {output_path}")


def _safe_stem(record: LSTMRecord) -> str:
  leaf = record.label.rsplit("/", 1)[-1]
  leaf = re.sub(r"[^A-Za-z0-9_.-]+", "_", leaf).strip("._") or "lstm"
  digest = hashlib.sha256(f"{record.graph_path}/{record.label}".encode()).hexdigest()[:8]
  return f"lstm_{record.index:02d}_{leaf[:48]}_{digest}"


def _standalone_model(record: LSTMRecord, source_model: ModelProto) -> ModelProto:
  node = NodeProto()
  node.CopyFrom(record.node)
  node.name = f"standalone_lstm_{record.index:02d}"
  del node.input[:]
  node.input.extend([
    "X",
    "W",
    "R",
    "B" if record.bias is not None else "",
    "",
    "initial_h",
    "initial_c",
  ])
  del node.output[:]
  node.output.extend(OUTPUT_NAMES)

  directions = record.num_directions
  hidden = record.hidden_size
  tensor_type = TensorProto.FLOAT
  inputs = [
    onnx.helper.make_tensor_value_info(
      "X", tensor_type, ["sequence_length", "batch_size", record.input_size]
    ),
    onnx.helper.make_tensor_value_info(
      "initial_h", tensor_type, [directions, "batch_size", hidden]
    ),
    onnx.helper.make_tensor_value_info(
      "initial_c", tensor_type, [directions, "batch_size", hidden]
    ),
  ]
  outputs = [
    onnx.helper.make_tensor_value_info(
      "Y", tensor_type, ["sequence_length", directions, "batch_size", hidden]
    ),
    onnx.helper.make_tensor_value_info("Y_h", tensor_type, [directions, "batch_size", hidden]),
    onnx.helper.make_tensor_value_info("Y_c", tensor_type, [directions, "batch_size", hidden]),
  ]
  initializers = [
    numpy_helper.from_array(record.weight, "W"),
    numpy_helper.from_array(record.recurrent, "R"),
  ]
  if record.bias is not None:
    initializers.append(numpy_helper.from_array(record.bias, "B"))

  graph = onnx.helper.make_graph(
    [node],
    f"standalone_silero_lstm_{record.index:02d}",
    inputs,
    outputs,
    initializers,
  )
  main_opset = next(
    (item.version for item in source_model.opset_import if item.domain in ("", "ai.onnx")),
    16,
  )
  model = onnx.helper.make_model(
    graph,
    opset_imports=[onnx.helper.make_operatorsetid("", main_opset)],
    producer_name="compare_silero_vad_DynamicQuantizeLSTM_vs_SimoQuantizeLSTM.py",
  )
  model.ir_version = source_model.ir_version
  onnx.checker.check_model(model)
  return model


def _quantize_models(
  float_path: Path,
  paths: ModelPaths,
  config_path: Path,
) -> None:
  quantize_dynamic(
    model_input=float_path,
    model_output=paths.dynamic_model,
    op_types_to_quantize=["LSTM"],
    per_channel=True,
    reduce_range=False,
    weight_type=QuantType.QInt8,
  )
  simo_quantize(
    float_path,
    config_path,
    output_path=paths.simo_model,
    simplify=False,
  )
  _validate_operator_model(paths.dynamic_model, "com.microsoft", "DynamicQuantizeLSTM")
  _validate_operator_model(paths.simo_model, "com.simo", "SimoQuantizeLSTM")


def _validate_operator_model(path: Path, domain: str, op_type: str) -> None:
  model = onnx.load(path, load_external_data=True)
  onnx.checker.check_model(model)
  matches = [node for node in model.graph.node if node.domain == domain and node.op_type == op_type]
  if len(model.graph.node) != 1 or len(matches) != 1:
    nodes = [(node.domain, node.op_type) for node in model.graph.node]
    raise RuntimeError(f"{path.name}: expected only {domain}::{op_type}, found {nodes}")


def _strict_cuda_options(custom_op_library: Path | None = None) -> ort.SessionOptions:
  options = ort.SessionOptions()
  options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
  options.log_severity_level = 3
  if custom_op_library is not None:
    register_custom_ops(options, custom_op_library)
  return options


def _create_sessions(paths: ModelPaths, custom_op_library: Path) -> dict[str, ort.InferenceSession]:
  sessions = {
    "float_cuda": ort.InferenceSession(
      str(paths.float_model),
      sess_options=_strict_cuda_options(),
      providers=["CUDAExecutionProvider"],
    ),
    "float_cpu": ort.InferenceSession(
      str(paths.float_model),
      providers=["CPUExecutionProvider"],
    ),
    "dynamic_cpu": ort.InferenceSession(
      str(paths.dynamic_model),
      providers=["CPUExecutionProvider"],
    ),
    "simo_cuda": ort.InferenceSession(
      str(paths.simo_model),
      sess_options=_strict_cuda_options(custom_op_library),
      providers=["CUDAExecutionProvider"],
    ),
  }
  expected_primary = {
    "float_cuda": "CUDAExecutionProvider",
    "float_cpu": "CPUExecutionProvider",
    "dynamic_cpu": "CPUExecutionProvider",
    "simo_cuda": "CUDAExecutionProvider",
  }
  for name, session in sessions.items():
    actual = session.get_providers()
    if not actual or actual[0] != expected_primary[name]:
      raise RuntimeError(
        f"{name}: expected primary provider {expected_primary[name]}, got {actual}"
      )
  return sessions


def _run(session: ort.InferenceSession, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
  outputs = session.run(list(OUTPUT_NAMES), feeds)
  if len(outputs) != len(OUTPUT_NAMES):
    raise RuntimeError(f"expected {len(OUTPUT_NAMES)} outputs, got {len(outputs)}")
  for name, output in zip(OUTPUT_NAMES, outputs, strict=True):
    if output.dtype != np.float32 or not np.isfinite(output).all():
      raise RuntimeError(f"{name}: invalid output dtype or non-finite values")
  return outputs


def _metrics(reference: np.ndarray, actual: np.ndarray) -> ErrorMetrics:
  reference64 = np.asarray(reference, dtype=np.float64).ravel()
  actual64 = np.asarray(actual, dtype=np.float64).ravel()
  if reference64.shape != actual64.shape:
    raise RuntimeError(f"metric shape mismatch: {reference64.shape} vs {actual64.shape}")
  delta = actual64 - reference64
  reference_norm = float(np.linalg.norm(reference64))
  actual_norm = float(np.linalg.norm(actual64))
  denominator = reference_norm * actual_norm
  if denominator <= 1e-24:
    cosine = 1.0 if np.linalg.norm(delta) <= 1e-12 else 0.0
  else:
    cosine = float(np.dot(reference64, actual64) / denominator)
  return ErrorMetrics(
    cosine_similarity=cosine,
    relative_l2_error=float(np.linalg.norm(delta) / max(reference_norm, 1e-12)),
    mean_absolute_error=float(np.mean(np.abs(delta))) if delta.size else 0.0,
    max_absolute_error=float(np.max(np.abs(delta))) if delta.size else 0.0,
    reference_l2_norm=reference_norm,
  )


def _comparison_metrics(
  reference_runs: Sequence[Sequence[np.ndarray]],
  candidate_runs: Sequence[Sequence[np.ndarray]],
) -> dict[str, dict[str, float]]:
  if len(reference_runs) != len(candidate_runs):
    raise RuntimeError("reference and candidate run counts differ")
  result: dict[str, dict[str, float]] = {}
  combined_reference = []
  combined_candidate = []
  for output_index, output_name in enumerate(OUTPUT_NAMES):
    reference = np.concatenate([
      np.asarray(outputs[output_index]).ravel() for outputs in reference_runs
    ])
    candidate = np.concatenate([
      np.asarray(outputs[output_index]).ravel() for outputs in candidate_runs
    ])
    result[output_name] = asdict(_metrics(reference, candidate))
    combined_reference.append(reference)
    combined_candidate.append(candidate)
  result["combined"] = asdict(
    _metrics(np.concatenate(combined_reference), np.concatenate(combined_candidate))
  )
  return result


def _random_feeds(
  record: LSTMRecord,
  rng: np.random.Generator,
  sequence_length: int,
  batch_size: int,
  x_std: float,
  h_std: float,
  c_std: float,
  *,
  zero_state: bool,
) -> dict[str, np.ndarray]:
  state_shape = (record.num_directions, batch_size, record.hidden_size)
  h = np.zeros(state_shape, dtype=np.float32)
  c = np.zeros(state_shape, dtype=np.float32)
  if not zero_state:
    h = rng.normal(0.0, h_std, state_shape).astype(np.float32)
    c = rng.normal(0.0, c_std, state_shape).astype(np.float32)
  return {
    "X": rng.normal(
      0.0,
      x_std,
      (sequence_length, batch_size, record.input_size),
    ).astype(np.float32),
    "initial_h": h,
    "initial_c": c,
  }


def _run_same_input_cases(
  record: LSTMRecord,
  sessions: dict[str, ort.InferenceSession],
  rng: np.random.Generator,
  *,
  cases: int,
  sequence_length: int,
  batch_size: int,
  x_std: float,
  h_std: float,
  c_std: float,
  zero_state: bool,
) -> dict[str, Any]:
  runs = {name: [] for name in sessions}
  for _ in range(cases):
    feeds = _random_feeds(
      record,
      rng,
      sequence_length,
      batch_size,
      x_std,
      h_std,
      c_std,
      zero_state=zero_state,
    )
    for name, session in sessions.items():
      runs[name].append(_run(session, feeds))
  baseline = runs["float_cuda"]
  return {
    "case_count": cases,
    "zero_state": zero_state,
    "comparisons": {
      name: _comparison_metrics(baseline, outputs)
      for name, outputs in runs.items()
      if name != "float_cuda"
    },
  }


class _SimoTorchReference:
  """ATen/Triton reference for the exact SIMO LSTM recurrence."""

  def __init__(self, record: LSTMRecord, simo_model_path: Path):
    import torch
    from simo.ops.flex_api import (
      per_channel_downcast_to_fp8_or_int8,
      per_channel_upcast,
    )
    from simo.quantization.dtypes import int8

    model = onnx.load(simo_model_path, load_external_data=True)
    node = _single_node(model, "com.simo", "SimoQuantizeLSTM")
    attrs = _node_attributes(node)
    expected_attrs = {
      "input_dtype": b"int8",
      "input_granularity": b"per_channel",
      "input_axis": 0,
      "input_quant_min": -128.0,
      "input_quant_max": 127.0,
    }
    mismatches = {
      name: (attrs.get(name), expected)
      for name, expected in expected_attrs.items()
      if attrs.get(name) != expected
    }
    if mismatches:
      raise RuntimeError(
        "the SIMO Torch reference only supports the requested INT8 axis=0 config; "
        f"attribute mismatches: {mismatches}"
      )

    weight, _ = _dequantize_simo_weight(
      model,
      node,
      quantized_input=1,
      scale_input=3,
      num_directions=record.num_directions,
      hidden_size=record.hidden_size,
    )
    recurrent, _ = _dequantize_simo_weight(
      model,
      node,
      quantized_input=2,
      scale_input=4,
      num_directions=record.num_directions,
      hidden_size=record.hidden_size,
    )
    bias = (
      record.bias
      if record.bias is not None
      else np.zeros((record.num_directions, 8 * record.hidden_size), dtype=np.float32)
    )
    fused_bias = bias[:, : 4 * record.hidden_size] + bias[:, 4 * record.hidden_size :]

    self._torch = torch
    self._downcast = per_channel_downcast_to_fp8_or_int8
    self._upcast = per_channel_upcast
    self._int8 = int8
    self._record = record
    self._weight = torch.from_numpy(weight).to(device="cuda", dtype=torch.float32)
    self._recurrent = torch.from_numpy(recurrent).to(device="cuda", dtype=torch.float32)
    self._bias = torch.from_numpy(fused_bias).to(device="cuda", dtype=torch.float32)

  def _qdq(self, tensor):
    quantized, scale = self._downcast(
      tensor,
      self._int8,
      -128.0,
      127.0,
      0,
    )
    return self._upcast(quantized, scale, self._torch.float32)

  def run(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    torch = self._torch
    record = self._record
    hidden = record.hidden_size
    with torch.inference_mode():
      x = torch.from_numpy(feeds["X"]).to(device="cuda", dtype=torch.float32)
      initial_h = torch.from_numpy(feeds["initial_h"]).to(device="cuda", dtype=torch.float32)
      initial_c = torch.from_numpy(feeds["initial_c"]).to(device="cuda", dtype=torch.float32)
      sequence_length, batch_size, _ = x.shape
      y = torch.empty(
        (sequence_length, record.num_directions, batch_size, hidden),
        device="cuda",
        dtype=torch.float32,
      )
      y_h = torch.empty_like(initial_h)
      y_c = torch.empty_like(initial_c)

      for direction in range(record.num_directions):
        reverse = record.direction == "reverse" or (
          record.direction == "bidirectional" and direction == 1
        )
        h = initial_h[direction].contiguous()
        c = initial_c[direction].contiguous()
        for step in range(sequence_length):
          time = sequence_length - 1 - step if reverse else step
          x_qdq = self._qdq(x[time].contiguous())
          h_qdq = self._qdq(h)
          gates = []
          for gate in range(4):
            start = gate * hidden
            end = (gate + 1) * hidden
            gates.append(
              x_qdq @ self._weight[direction, start:end].T
              + h_qdq @ self._recurrent[direction, start:end].T
              + self._bias[direction, start:end]
            )
          input_gate = torch.sigmoid(gates[0])
          output_gate = torch.sigmoid(gates[1])
          forget_gate = torch.sigmoid(gates[2])
          cell_gate = torch.tanh(gates[3])
          c = forget_gate * c + input_gate * cell_gate
          h = output_gate * torch.tanh(c)
          y[time, direction].copy_(h)
        y_h[direction].copy_(h)
        y_c[direction].copy_(c)

      return [tensor.cpu().numpy() for tensor in (y, y_h, y_c)]


def _run_closed_loop_rollout(
  record: LSTMRecord,
  sessions: dict[str, ort.InferenceSession],
  simo_model_path: Path,
  rng: np.random.Generator,
  *,
  steps: int,
  sequence_length: int,
  batch_size: int,
  x_std: float,
) -> dict[str, Any]:
  state_shape = (record.num_directions, batch_size, record.hidden_size)
  reference = _SimoTorchReference(record, simo_model_path)
  reference_name = "simo_torch_reference"
  all_names = [*sessions, reference_name]
  states = {
    name: (
      np.zeros(state_shape, dtype=np.float32),
      np.zeros(state_shape, dtype=np.float32),
    )
    for name in all_names
  }
  runs = {name: [] for name in all_names}
  per_step_relative_l2 = {name: [] for name in all_names if name != "float_cuda"}
  for _ in range(steps):
    x = rng.normal(
      0.0,
      x_std,
      (sequence_length, batch_size, record.input_size),
    ).astype(np.float32)
    for name, session in sessions.items():
      h, c = states[name]
      outputs = _run(session, {"X": x, "initial_h": h, "initial_c": c})
      runs[name].append(outputs)
      states[name] = (outputs[1], outputs[2])
    h, c = states[reference_name]
    outputs = reference.run({"X": x, "initial_h": h, "initial_c": c})
    runs[reference_name].append(outputs)
    states[reference_name] = (outputs[1], outputs[2])
    baseline_step = np.concatenate([output.ravel() for output in runs["float_cuda"][-1]])
    for name in per_step_relative_l2:
      actual = np.concatenate([output.ravel() for output in runs[name][-1]])
      per_step_relative_l2[name].append(_metrics(baseline_step, actual).relative_l2_error)

  baseline = runs["float_cuda"]
  comparisons = {
    name: _comparison_metrics(baseline, outputs)
    for name, outputs in runs.items()
    if name != "float_cuda"
  }
  final_step = {
    name: _comparison_metrics([baseline[-1]], [outputs[-1]])
    for name, outputs in runs.items()
    if name != "float_cuda"
  }
  operator_reference_check = _comparison_metrics(runs["simo_cuda"], runs[reference_name])
  if (
    operator_reference_check["combined"]["relative_l2_error"]
    > SIMO_REFERENCE_ROLLOUT_RELATIVE_L2_TOLERANCE
  ):
    raise RuntimeError(
      "SIMO Torch reference does not match SimoQuantizeLSTM: "
      f"{operator_reference_check['combined']}"
    )
  return {
    "steps": steps,
    "initial_state": "zeros",
    "comparisons": comparisons,
    "final_step": final_step,
    "combined_relative_l2_by_step": per_step_relative_l2,
    "simo_operator_vs_torch_reference": operator_reference_check,
  }


def _initializer_map(model: ModelProto) -> dict[str, TensorProto]:
  return {tensor.name: tensor for tensor in model.graph.initializer}


def _single_node(model: ModelProto, domain: str, op_type: str) -> NodeProto:
  nodes = [node for node in model.graph.node if node.domain == domain and node.op_type == op_type]
  if len(nodes) != 1:
    raise RuntimeError(f"expected one {domain}::{op_type}, got {len(nodes)}")
  return nodes[0]


def _dequantize_dynamic_weight(
  model: ModelProto,
  node: NodeProto,
  *,
  weight_input: int,
  scale_input: int,
  zero_point_input: int,
) -> tuple[np.ndarray, dict[str, Any]]:
  initializers = _initializer_map(model)
  quantized = np.asarray(numpy_helper.to_array(initializers[node.input[weight_input]]))
  scale = np.asarray(numpy_helper.to_array(initializers[node.input[scale_input]]), dtype=np.float32)
  zero_point = np.asarray(numpy_helper.to_array(initializers[node.input[zero_point_input]]))
  if quantized.ndim != 3 or scale.ndim != 2 or zero_point.shape != scale.shape:
    raise RuntimeError(
      f"unexpected DynamicQuantizeLSTM carrier shapes: {quantized.shape}, "
      f"{scale.shape}, {zero_point.shape}"
    )
  # ORT stores quantized LSTM weights as [D, K, 4H]. Scale/ZP are [D, 4H].
  dequantized = (
    (quantized.astype(np.float32) - zero_point[:, None, :].astype(np.float32)) * scale[:, None, :]
  ).transpose(0, 2, 1)
  details = {
    "carrier_dtype": str(quantized.dtype),
    "carrier_shape": list(quantized.shape),
    "scale_shape": list(scale.shape),
    "code_min": int(quantized.min()),
    "code_max": int(quantized.max()),
  }
  return dequantized, details


def _decode_simo_scale(carrier: np.ndarray) -> np.ndarray:
  carrier = np.ascontiguousarray(carrier)
  if carrier.dtype != np.uint8 or carrier.shape[-1] % np.dtype(np.float32).itemsize:
    raise RuntimeError(
      f"unexpected SIMO scale carrier dtype/shape: {carrier.dtype}/{carrier.shape}"
    )
  return carrier.view(np.float32)


def _dequantize_simo_weight(
  model: ModelProto,
  node: NodeProto,
  *,
  quantized_input: int,
  scale_input: int,
  num_directions: int,
  hidden_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
  initializers = _initializer_map(model)
  carrier = np.asarray(numpy_helper.to_array(initializers[node.input[quantized_input]]))
  scale_carrier = np.asarray(numpy_helper.to_array(initializers[node.input[scale_input]]))
  quantized = np.ascontiguousarray(carrier).view(np.int8)
  scale = _decode_simo_scale(scale_carrier)
  expected_gates = num_directions * 4
  if (
    quantized.ndim != 3
    or quantized.shape[0] != expected_gates
    or quantized.shape[1] != hidden_size
    or scale.shape != (expected_gates, hidden_size, 1)
  ):
    raise RuntimeError(
      f"unexpected SIMO carrier shapes: quantized={quantized.shape}, scale={scale.shape}"
    )
  dequantized_gates = quantized.astype(np.float32) * scale
  dequantized = dequantized_gates.reshape(
    num_directions,
    4 * hidden_size,
    quantized.shape[2],
  )
  details = {
    "carrier_dtype": str(carrier.dtype),
    "carrier_shape": list(carrier.shape),
    "scale_carrier_shape": list(scale_carrier.shape),
    "decoded_scale_shape": list(scale.shape),
    "code_min": int(quantized.min()),
    "code_max": int(quantized.max()),
  }
  return dequantized, details


def _weight_diagnostics(record: LSTMRecord, paths: ModelPaths) -> dict[str, Any]:
  dynamic_model = onnx.load(paths.dynamic_model, load_external_data=True)
  dynamic_node = _single_node(dynamic_model, "com.microsoft", "DynamicQuantizeLSTM")
  simo_model = onnx.load(paths.simo_model, load_external_data=True)
  simo_node = _single_node(simo_model, "com.simo", "SimoQuantizeLSTM")

  dynamic_w, dynamic_w_details = _dequantize_dynamic_weight(
    dynamic_model,
    dynamic_node,
    weight_input=1,
    scale_input=8,
    zero_point_input=9,
  )
  dynamic_r, dynamic_r_details = _dequantize_dynamic_weight(
    dynamic_model,
    dynamic_node,
    weight_input=2,
    scale_input=10,
    zero_point_input=11,
  )
  simo_w, simo_w_details = _dequantize_simo_weight(
    simo_model,
    simo_node,
    quantized_input=1,
    scale_input=3,
    num_directions=record.num_directions,
    hidden_size=record.hidden_size,
  )
  simo_r, simo_r_details = _dequantize_simo_weight(
    simo_model,
    simo_node,
    quantized_input=2,
    scale_input=4,
    num_directions=record.num_directions,
    hidden_size=record.hidden_size,
  )

  def compare(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    return asdict(_metrics(reference, actual))

  return {
    "W": {
      "dynamic_vs_float": compare(record.weight, dynamic_w),
      "simo_vs_float": compare(record.weight, simo_w),
      "simo_vs_dynamic": compare(dynamic_w, simo_w),
      "dynamic_carrier": dynamic_w_details,
      "simo_carrier": simo_w_details,
    },
    "R": {
      "dynamic_vs_float": compare(record.recurrent, dynamic_r),
      "simo_vs_float": compare(record.recurrent, simo_r),
      "simo_vs_dynamic": compare(dynamic_r, simo_r),
      "dynamic_carrier": dynamic_r_details,
      "simo_carrier": simo_r_details,
    },
  }


def _winner(comparisons: dict[str, Any]) -> str:
  dynamic = comparisons["dynamic_cpu"]["combined"]["relative_l2_error"]
  simo = comparisons["simo_cuda"]["combined"]["relative_l2_error"]
  if math.isclose(dynamic, simo, rel_tol=1e-6, abs_tol=1e-12):
    return "tie"
  return "dynamic_cpu" if dynamic < simo else "simo_cuda"


def _print_comparison(title: str, section: dict[str, Any]) -> None:
  print(f"  {title}")
  print("    candidate                cosine_similarity   relative_l2_error   max_absolute_error")
  names = ["float_cpu", "dynamic_cpu", "simo_cuda"]
  if "simo_torch_reference" in section["comparisons"]:
    names.append("simo_torch_reference")
  for name in names:
    metric = section["comparisons"][name]["combined"]
    print(
      f"    {name:<24} {metric['cosine_similarity']:>17.9f} "
      f"{metric['relative_l2_error']:>19.9e} {metric['max_absolute_error']:>20.9e}"
    )
  print(f"    quantized winner by combined relative L2: {_winner(section['comparisons'])}")


def _preflight(args: argparse.Namespace, custom_op_library: Path) -> tuple[Path, Path, Path]:
  input_path = args.input_model.expanduser().resolve()
  config_path = args.config.expanduser().resolve()
  output_dir = args.output_dir.expanduser().resolve()
  for label, path in (
    ("input model", input_path),
    ("SIMO config", config_path),
    ("custom-op library", custom_op_library),
  ):
    if not path.is_file():
      raise FileNotFoundError(f"{label} does not exist: {path}")
  if output_dir.exists() and not output_dir.is_dir():
    raise NotADirectoryError(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  protected = [output_dir / "silero_vad.const_folded.onnx", output_dir / "comparison.json"]
  existing = [path for path in protected if path.exists()]
  if existing and not args.overwrite:
    raise FileExistsError(
      f"output already exists: {existing[0]}; pass --overwrite to replace generated artifacts"
    )
  return input_path, config_path, output_dir


def main() -> int:
  args = _parse_args()
  custom_op_library = (
    args.custom_op_library.expanduser().resolve()
    if args.custom_op_library is not None
    else get_custom_ops_library_path().resolve()
  )
  input_path, config_path, output_dir = _preflight(args, custom_op_library)
  folded_path = output_dir / "silero_vad.const_folded.onnx"

  print(f"Input model: {input_path}")
  print(f"SIMO config: {config_path}")
  print(f"Custom-op library: {custom_op_library}")
  print(f"Output directory: {output_dir}")
  _fold_model(input_path, folded_path)
  folded_model = onnx.load(folded_path, load_external_data=True)
  onnx.checker.check_model(folded_model)
  records = collect_lstm_records(folded_model)
  if not records:
    raise RuntimeError("the constant-folded model does not contain an LSTM")
  print(f"Constant-folded nested LSTMs: {len(records)}")

  report: dict[str, Any] = {
    "environment": {
      "onnx_version": onnx.__version__,
      "onnxruntime_version": ort.__version__,
      "available_providers": ort.get_available_providers(),
      "input_model": str(input_path),
      "folded_model": str(folded_path),
      "simo_config": str(config_path),
      "custom_op_library": str(custom_op_library),
    },
    "parameters": {
      "seed": args.seed,
      "cases": args.cases,
      "rollout_steps": args.rollout_steps,
      "sequence_length": args.sequence_length,
      "batch_size": args.batch_size,
      "x_std": args.x_std,
      "rollout_x_stds": args.rollout_x_stds,
      "h_std": args.h_std,
      "c_std": args.c_std,
    },
    "lstm_count": len(records),
    "lstms": [],
  }

  for record in records:
    stem = _safe_stem(record)
    paths = ModelPaths(
      float_model=output_dir / f"{stem}.float.onnx",
      dynamic_model=output_dir / f"{stem}.dynamic_qint8_per_channel.onnx",
      simo_model=output_dir / f"{stem}.simo_int8_per_channel.onnx",
    )
    if not args.overwrite:
      existing = [path for path in asdict(paths).values() if Path(path).exists()]
      if existing:
        raise FileExistsError(f"output already exists: {existing[0]}; pass --overwrite")

    standalone = _standalone_model(record, folded_model)
    onnx.save(standalone, paths.float_model)
    _quantize_models(paths.float_model, paths, config_path)
    sessions = _create_sessions(paths, custom_op_library)
    rng = np.random.default_rng(args.seed + record.index * 1009)

    print(f"\nLSTM {record.index}: {record.label}")
    print(f"  graph: {record.graph_path}")
    print(
      f"  direction={record.direction}, W={record.weight.shape}, "
      f"R={record.recurrent.shape}, B={None if record.bias is None else record.bias.shape}"
    )
    weight_diagnostics = _weight_diagnostics(record, paths)
    for role in ("W", "R"):
      dynamic_error = weight_diagnostics[role]["dynamic_vs_float"]["relative_l2_error"]
      simo_error = weight_diagnostics[role]["simo_vs_float"]["relative_l2_error"]
      print(f"  {role} dequant relative L2: Dynamic={dynamic_error:.9e}, SIMO={simo_error:.9e}")

    zero_state = _run_same_input_cases(
      record,
      sessions,
      rng,
      cases=args.cases,
      sequence_length=args.sequence_length,
      batch_size=args.batch_size,
      x_std=args.x_std,
      h_std=args.h_std,
      c_std=args.c_std,
      zero_state=True,
    )
    random_state = _run_same_input_cases(
      record,
      sessions,
      rng,
      cases=args.cases,
      sequence_length=args.sequence_length,
      batch_size=args.batch_size,
      x_std=args.x_std,
      h_std=args.h_std,
      c_std=args.c_std,
      zero_state=False,
    )
    _print_comparison("same input, zero initial state", zero_state)
    _print_comparison("same input, random initial state", random_state)
    rollouts = {}
    for scale_index, rollout_x_std in enumerate(args.rollout_x_stds):
      rollout_rng = np.random.default_rng(args.seed + record.index * 1009 + 100_000 + scale_index)
      rollout = _run_closed_loop_rollout(
        record,
        sessions,
        paths.simo_model,
        rollout_rng,
        steps=args.rollout_steps,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        x_std=rollout_x_std,
      )
      key = format(rollout_x_std, ".12g")
      rollouts[key] = rollout
      _print_comparison(
        f"closed-loop recurrent rollout (X std={rollout_x_std:g})",
        rollout,
      )

    report["lstms"].append({
      "index": record.index,
      "name": record.label,
      "graph_path": record.graph_path,
      "direction": record.direction,
      "input_size": record.input_size,
      "hidden_size": record.hidden_size,
      "num_directions": record.num_directions,
      "weight_sha256": hashlib.sha256(record.weight.tobytes()).hexdigest(),
      "recurrent_sha256": hashlib.sha256(record.recurrent.tobytes()).hexdigest(),
      "models": {key: str(value) for key, value in asdict(paths).items()},
      "providers": {name: session.get_providers() for name, session in sessions.items()},
      "weight_diagnostics": weight_diagnostics,
      "same_input_zero_state": zero_state,
      "same_input_random_state": random_state,
      "closed_loop_rollouts": rollouts,
    })
    del sessions

  result_path = output_dir / "comparison.json"
  result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"\nDetailed JSON report: {result_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
