#!/usr/bin/env python3
"""Constant-fold an ONNX model, then dynamically quantize nested LSTMs.

The script writes one folded model plus the Cartesian product of the
schema-legal DynamicQuantizeLSTM weight types and per-channel modes. Only LSTM
nodes are quantized, including LSTMs inside control-flow subgraphs.

Examples:
  python like-useful/test-const-fold-quantize-lstm.py \
    --output-dir temp/silero-vad-dynamic-lstm

  python like-useful/test-const-fold-quantize-lstm.py model.onnx \
    --output-dir output --overwrite
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import product
import os
from pathlib import Path
import tempfile
from typing import Iterator

import onnx
from onnx import AttributeProto, GraphProto, ModelProto, TensorProto
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


DEFAULT_INPUT = Path(
  "/share/users/like/package/jdjv/silero_vad_clean/onnx_float_baseline/silero_vad.onnx"
)
PER_CHANNEL_VALUES = (False, True)

# DynamicQuantizeLSTM's T2 schema permits only int8 and uint8. ORT 1.27's
# Python LSTM quantizer currently emits int8 for both requested types; the
# validation report below makes that implementation detail visible.
LSTM_WEIGHT_TYPES = (QuantType.QInt8, QuantType.QUInt8)
LSTM_PARAMETER_INPUTS = ((1, "W"), (2, "R"), (3, "B"))
QUANTIZED_LSTM_INPUTS = {
  "W": 1,
  "R": 2,
  "W_scale": 8,
  "W_zero_point": 9,
  "R_scale": 10,
  "R_zero_point": 11,
}
EIGHT_BIT_TYPES = {TensorProto.INT8, TensorProto.UINT8}


@dataclass(frozen=True)
class ConstantSource:
  kind: str
  owner_graph: str
  shape: tuple[int, ...] | None = None
  dtype: str | None = None


@dataclass(frozen=True)
class LSTMRecord:
  name: str
  graph_path: str
  inputs: dict[int, tuple[str, ConstantSource | None]]

  def has_constant_inputs(self, indices: tuple[int, ...]) -> bool:
    return all(self.inputs[index][1] is not None for index in indices)


@dataclass(frozen=True)
class DynamicLSTMRecord:
  name: str
  graph_path: str
  weight_dtype: str
  recurrent_dtype: str
  weight_shape: tuple[int, ...]
  recurrent_shape: tuple[int, ...]
  weight_scale_shape: tuple[int, ...]
  recurrent_scale_shape: tuple[int, ...]


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Constant-fold nested LSTM weights, then run LSTM-only ONNX Runtime "
      "dynamic quantization for every per-channel/weight-type combination."
    )
  )
  parser.add_argument(
    "input",
    nargs="?",
    type=Path,
    default=DEFAULT_INPUT,
    help=f"input ONNX model (default: {DEFAULT_INPUT})",
  )
  parser.add_argument(
    "-o",
    "--output-dir",
    type=Path,
    required=True,
    help="directory for the folded model and all quantized models",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="replace this script's existing output files",
  )
  return parser.parse_args()


def _iter_subgraphs(node: onnx.NodeProto) -> Iterator[tuple[str, GraphProto]]:
  for attribute in node.attribute:
    if attribute.type == AttributeProto.GRAPH:
      yield attribute.name, attribute.g
    elif attribute.type == AttributeProto.GRAPHS:
      for index, graph in enumerate(attribute.graphs):
        yield f"{attribute.name}[{index}]", graph


def _tensor_source(tensor: TensorProto, graph_path: str, kind: str) -> ConstantSource:
  return ConstantSource(
    kind=kind,
    owner_graph=graph_path,
    shape=tuple(tensor.dims),
    dtype=TensorProto.DataType.Name(tensor.data_type),
  )


def _constant_node_source(node: onnx.NodeProto, graph_path: str) -> ConstantSource:
  tensor = next(
    (attribute.t for attribute in node.attribute if attribute.type == AttributeProto.TENSOR),
    None,
  )
  if tensor is None:
    return ConstantSource(kind="Constant node", owner_graph=graph_path)
  return _tensor_source(tensor, graph_path, "Constant node")


def collect_lstm_records(model: ModelProto) -> list[LSTMRecord]:
  records: list[LSTMRecord] = []

  def visit_graph(
    graph: GraphProto,
    graph_path: str,
    inherited_constants: dict[str, ConstantSource],
  ) -> None:
    constants = dict(inherited_constants)
    constants.update({
      tensor.name: _tensor_source(tensor, graph_path, "initializer") for tensor in graph.initializer
    })
    producers = {
      output_name: node for node in graph.node for output_name in node.output if output_name
    }
    for node in graph.node:
      if node.op_type == "Constant":
        source = _constant_node_source(node, graph_path)
        constants.update({name: source for name in node.output if name})

    for node in graph.node:
      if node.domain in ("", "ai.onnx") and node.op_type == "LSTM":
        parameter_inputs: dict[int, tuple[str, ConstantSource | None]] = {}
        for index, _label in LSTM_PARAMETER_INPUTS:
          name = node.input[index] if index < len(node.input) else ""
          source = constants.get(name)
          if source is None and name in producers and producers[name].op_type == "Constant":
            source = _constant_node_source(producers[name], graph_path)
          parameter_inputs[index] = (name, source)
        records.append(
          LSTMRecord(
            name=node.name or "<unnamed LSTM>",
            graph_path=graph_path,
            inputs=parameter_inputs,
          )
        )

      node_name = node.name or node.op_type
      for attribute_name, subgraph in _iter_subgraphs(node):
        visit_graph(
          subgraph,
          f"{graph_path}/{node_name}:{attribute_name}",
          constants,
        )

  visit_graph(model.graph, "main", {})
  return records


def _describe_input(record: LSTMRecord, index: int, label: str) -> str:
  name, source = record.inputs[index]
  if source is None:
    return f"{label} ({name!r}): non-constant"
  shape = list(source.shape) if source.shape is not None else "unknown"
  return f"{label} ({name!r}): {source.kind}, shape={shape}, dtype={source.dtype or 'unknown'}"


def print_lstm_report(title: str, records: list[LSTMRecord]) -> None:
  required = tuple(index for index, _label in LSTM_PARAMETER_INPUTS)
  constant_count = sum(record.has_constant_inputs(required) for record in records)
  print(f"\n{title}")
  print(f"LSTM count: {len(records)}")
  print(f"LSTMs with constant W/R/B: {constant_count}/{len(records)}")
  for record in records:
    print(f"- {record.name}")
    print(f"  graph: {record.graph_path}")
    for index, label in LSTM_PARAMETER_INPUTS:
      print(f"  {_describe_input(record, index, label)}")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _fold_with_onnxruntime(input_path: Path, output_path: Path) -> None:
  options = ort.SessionOptions()
  options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
  options.optimized_model_filepath = str(output_path)
  ort.InferenceSession(
    str(input_path),
    sess_options=options,
    providers=["CPUExecutionProvider"],
  )
  if not output_path.is_file():
    raise RuntimeError(f"ONNX Runtime did not create the folded model: {output_path}")


def _validate_session(model_path: Path) -> None:
  options = ort.SessionOptions()
  options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
  ort.InferenceSession(
    str(model_path),
    sess_options=options,
    providers=["CPUExecutionProvider"],
  )


def _validate_folded_model(
  input_records: list[LSTMRecord],
  folded_path: Path,
) -> tuple[ModelProto, list[LSTMRecord]]:
  folded_model = onnx.load(folded_path, load_external_data=True)
  onnx.checker.check_model(folded_model)
  _validate_session(folded_path)

  folded_records = collect_lstm_records(folded_model)
  if len(folded_records) != len(input_records):
    raise RuntimeError(
      f"LSTM count changed during folding: {len(input_records)} -> {len(folded_records)}"
    )
  required = tuple(index for index, _label in LSTM_PARAMETER_INPUTS)
  nonconstant = [
    record.name for record in folded_records if not record.has_constant_inputs(required)
  ]
  if nonconstant:
    raise RuntimeError(
      "constant folding did not produce direct constant W/R/B inputs for: " + ", ".join(nonconstant)
    )
  return folded_model, folded_records


def _require_initializer(
  initializers: dict[str, TensorProto],
  node: onnx.NodeProto,
  input_label: str,
  graph_path: str,
) -> TensorProto:
  input_index = QUANTIZED_LSTM_INPUTS[input_label]
  name = node.input[input_index] if input_index < len(node.input) else ""
  tensor = initializers.get(name)
  if tensor is None:
    raise RuntimeError(
      f"{graph_path}/{node.name or node.op_type}: {input_label} input {name!r} "
      "is not a visible initializer"
    )
  return tensor


def collect_dynamic_lstm_records(
  model: ModelProto,
  per_channel: bool,
) -> list[DynamicLSTMRecord]:
  records: list[DynamicLSTMRecord] = []

  def visit_graph(
    graph: GraphProto,
    graph_path: str,
    inherited_initializers: dict[str, TensorProto],
  ) -> None:
    initializers = dict(inherited_initializers)
    initializers.update({tensor.name: tensor for tensor in graph.initializer})

    for node in graph.node:
      if node.domain == "com.microsoft" and node.op_type == "DynamicQuantizeLSTM":
        tensors = {
          label: _require_initializer(initializers, node, label, graph_path)
          for label in QUANTIZED_LSTM_INPUTS
        }
        weight = tensors["W"]
        recurrent = tensors["R"]
        if len(weight.dims) != 3 or len(recurrent.dims) != 3:
          raise RuntimeError(
            f"{node.name}: expected rank-3 quantized W/R, got "
            f"{list(weight.dims)} and {list(recurrent.dims)}"
          )
        if weight.data_type not in EIGHT_BIT_TYPES or recurrent.data_type not in EIGHT_BIT_TYPES:
          raise RuntimeError(
            f"{node.name}: DynamicQuantizeLSTM W/R must be int8 or uint8, got "
            f"{TensorProto.DataType.Name(weight.data_type)} and "
            f"{TensorProto.DataType.Name(recurrent.data_type)}"
          )
        if weight.data_type != tensors["W_zero_point"].data_type:
          raise RuntimeError(f"{node.name}: W and W_zero_point dtypes differ")
        if recurrent.data_type != tensors["R_zero_point"].data_type:
          raise RuntimeError(f"{node.name}: R and R_zero_point dtypes differ")
        if tensors["W_scale"].data_type != TensorProto.FLOAT:
          raise RuntimeError(f"{node.name}: W_scale is not FLOAT")
        if tensors["R_scale"].data_type != TensorProto.FLOAT:
          raise RuntimeError(f"{node.name}: R_scale is not FLOAT")

        num_directions = weight.dims[0]
        gate_width = weight.dims[2]
        if recurrent.dims[0] != num_directions or recurrent.dims[2] != gate_width:
          raise RuntimeError(f"{node.name}: W/R direction or gate dimensions differ")
        expected_scale_shape = (num_directions, gate_width) if per_channel else (num_directions,)
        for scale_label in ("W_scale", "W_zero_point", "R_scale", "R_zero_point"):
          actual_shape = tuple(tensors[scale_label].dims)
          if actual_shape != expected_scale_shape:
            raise RuntimeError(
              f"{node.name}: {scale_label} shape {actual_shape} does not match "
              f"expected {expected_scale_shape} for per_channel={per_channel}"
            )

        records.append(
          DynamicLSTMRecord(
            name=node.name or "<unnamed DynamicQuantizeLSTM>",
            graph_path=graph_path,
            weight_dtype=TensorProto.DataType.Name(weight.data_type),
            recurrent_dtype=TensorProto.DataType.Name(recurrent.data_type),
            weight_shape=tuple(weight.dims),
            recurrent_shape=tuple(recurrent.dims),
            weight_scale_shape=tuple(tensors["W_scale"].dims),
            recurrent_scale_shape=tuple(tensors["R_scale"].dims),
          )
        )

      node_name = node.name or node.op_type
      for attribute_name, subgraph in _iter_subgraphs(node):
        visit_graph(
          subgraph,
          f"{graph_path}/{node_name}:{attribute_name}",
          initializers,
        )

  visit_graph(model.graph, "main", {})
  return records


def _validate_quantized_model(
  model_path: Path,
  expected_lstm_count: int,
  per_channel: bool,
  requested_weight_type: QuantType,
) -> list[DynamicLSTMRecord]:
  model = onnx.load(model_path, load_external_data=True)
  onnx.checker.check_model(model)
  _validate_session(model_path)

  remaining_lstm = collect_lstm_records(model)
  dynamic_records = collect_dynamic_lstm_records(model, per_channel)
  if remaining_lstm:
    raise RuntimeError(
      f"{model_path.name}: {len(remaining_lstm)} standard LSTM nodes were not quantized"
    )
  if len(dynamic_records) != expected_lstm_count:
    raise RuntimeError(
      f"{model_path.name}: expected {expected_lstm_count} DynamicQuantizeLSTM nodes, "
      f"found {len(dynamic_records)}"
    )

  actual_types = sorted(
    {record.weight_dtype for record in dynamic_records}
    | {record.recurrent_dtype for record in dynamic_records}
  )
  requested_dtype = TensorProto.DataType.Name(requested_weight_type.tensor_type)
  if actual_types != [requested_dtype]:
    print(
      "WARNING: "
      f"requested weight_type={requested_weight_type.name}, but ORT emitted "
      f"DynamicQuantizeLSTM W/R types {actual_types}"
    )
  return dynamic_records


def _output_names(input_path: Path) -> tuple[str, dict[tuple[bool, QuantType], str]]:
  stem = input_path.stem
  folded_name = f"{stem}.const_folded.onnx"
  quantized_names = {
    (per_channel, weight_type): (
      f"{stem}.const_folded.dynamic_lstm."
      f"per_channel_{str(per_channel).lower()}."
      f"weight_{weight_type.name.lower()}.onnx"
    )
    for per_channel, weight_type in product(PER_CHANNEL_VALUES, LSTM_WEIGHT_TYPES)
  }
  return folded_name, quantized_names


def _validate_paths(
  input_path: Path,
  output_dir: Path,
  output_names: list[str],
  overwrite: bool,
) -> None:
  if not input_path.is_file():
    raise FileNotFoundError(f"input model does not exist: {input_path}")
  if output_dir.exists() and not output_dir.is_dir():
    raise NotADirectoryError(f"output path is not a directory: {output_dir}")

  final_paths = [output_dir / name for name in output_names]
  if any(path.resolve(strict=False) == input_path for path in final_paths):
    raise ValueError("refusing to overwrite the input ONNX model")
  existing = [path for path in final_paths if path.exists()]
  if existing and not overwrite:
    formatted = "\n".join(f"  {path}" for path in existing)
    raise FileExistsError(
      f"output files already exist; pass --overwrite to replace them:\n{formatted}"
    )


def main() -> int:
  args = _parse_args()
  input_path = args.input.expanduser().resolve()
  output_dir = args.output_dir.expanduser().resolve()
  folded_name, quantized_names = _output_names(input_path)
  all_output_names = [folded_name, *quantized_names.values()]
  _validate_paths(input_path, output_dir, all_output_names, args.overwrite)
  output_dir.mkdir(parents=True, exist_ok=True)

  input_hash = _sha256(input_path)
  input_model = onnx.load(input_path, load_external_data=True)
  onnx.checker.check_model(input_model)
  input_records = collect_lstm_records(input_model)
  if not input_records:
    raise RuntimeError("the input model does not contain any LSTM nodes")
  print_lstm_report("Before constant folding", input_records)

  staged_paths: dict[str, Path] = {}
  quantized_reports: dict[tuple[bool, QuantType], list[DynamicLSTMRecord]] = {}
  with tempfile.TemporaryDirectory(
    dir=output_dir,
    prefix=f".{input_path.stem}.const-fold-quantize.",
  ) as temporary_dir:
    staging_dir = Path(temporary_dir)
    staged_folded = staging_dir / folded_name
    _fold_with_onnxruntime(input_path, staged_folded)
    _folded_model, folded_records = _validate_folded_model(input_records, staged_folded)
    print_lstm_report("After constant folding", folded_records)
    staged_paths[folded_name] = staged_folded

    print("\nDynamic quantization Cartesian product")
    for per_channel, weight_type in product(PER_CHANNEL_VALUES, LSTM_WEIGHT_TYPES):
      output_name = quantized_names[(per_channel, weight_type)]
      staged_output = staging_dir / output_name
      print(f"- per_channel={per_channel}, requested_weight_type={weight_type.name}")
      quantize_dynamic(
        model_input=staged_folded,
        model_output=staged_output,
        op_types_to_quantize=["LSTM"],
        per_channel=per_channel,
        reduce_range=False,
        weight_type=weight_type,
        extra_options={"EnableSubgraph": True},
      )
      records = _validate_quantized_model(
        staged_output,
        expected_lstm_count=len(folded_records),
        per_channel=per_channel,
        requested_weight_type=weight_type,
      )
      actual_types = sorted({record.weight_dtype for record in records})
      scale_shapes = sorted({record.weight_scale_shape for record in records})
      print(
        f"  DynamicQuantizeLSTM={len(records)}, actual_W_types={actual_types}, "
        f"W_scale_shapes={scale_shapes}"
      )
      staged_paths[output_name] = staged_output
      quantized_reports[(per_channel, weight_type)] = records

    if _sha256(input_path) != input_hash:
      raise RuntimeError("the input ONNX model changed during processing")
    if not args.overwrite:
      appeared = [output_dir / name for name in all_output_names if (output_dir / name).exists()]
      if appeared:
        raise FileExistsError(f"output appeared during processing: {appeared[0]}")

    for output_name in all_output_names:
      os.replace(staged_paths[output_name], output_dir / output_name)

  print("\nCreated ONNX files")
  print(f"- folded: {output_dir / folded_name}")
  for combination in product(PER_CHANNEL_VALUES, LSTM_WEIGHT_TYPES):
    records = quantized_reports[combination]
    print(
      f"- quantized: {output_dir / quantized_names[combination]} "
      f"(nested DynamicQuantizeLSTM={len(records)})"
    )
  print(f"Input SHA256 (unchanged): {input_hash}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
