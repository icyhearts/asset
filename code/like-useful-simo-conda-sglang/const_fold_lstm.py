#!/usr/bin/env python3
"""Fold constant LSTM parameter chains in a nested ONNX model.

The output is always written to a different path. ONNX Runtime's basic graph
optimizer performs the folding recursively in control-flow subgraphs. The
script then verifies that every LSTM has direct constant W, R, and B inputs.

Usage:
  python like-useful/const_fold_lstm.py
  python like-useful/const_fold_lstm.py input.onnx -o output.onnx
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Iterator

import onnx
from onnx import AttributeProto, GraphProto, ModelProto, TensorProto
import onnxruntime as ort


DEFAULT_INPUT = Path(
  "/share_data/users/tangdehua/project/jdjv/silero_vad_clean/"
  "onnx_float_baseline/silero_vad.onnx"
)
DEFAULT_OUTPUT = Path(__file__).parent / "silero_vad_const_folded.onnx"

# ONNX LSTM inputs are zero-based: X=0, W=1, R=2, B=3. In one-based
# terminology, W and R are inputs 2 and 3. Requiring B to be constant as well
# also covers callers that use zero-based "input 2/input 3" to mean R and B.
LSTM_PARAMETER_INPUTS = ((1, "W"), (2, "R"), (3, "B"))


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


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Fold constant LSTM weights in all nested ONNX subgraphs."
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
    "--output",
    type=Path,
    default=DEFAULT_OUTPUT,
    help=f"new output path (default: {DEFAULT_OUTPUT})",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="replace an existing output file; the input path is still protected",
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
      tensor.name: _tensor_source(tensor, graph_path, "initializer")
      for tensor in graph.initializer
    })

    producers = {
      output_name: node
      for node in graph.node
      for output_name in node.output
      if output_name
    }
    for node in graph.node:
      if node.op_type == "Constant":
        source = _constant_node_source(node, graph_path)
        constants.update({name: source for name in node.output if name})

    for node in graph.node:
      if node.op_type == "LSTM":
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
  one_based_index = index + 1
  if source is None:
    return f"input {one_based_index} ({label}, node.input[{index}]={name!r}): non-constant"
  shape = list(source.shape) if source.shape is not None else "unknown"
  return (
    f"input {one_based_index} ({label}, node.input[{index}]={name!r}): "
    f"{source.kind}, shape={shape}, dtype={source.dtype or 'unknown'}"
  )


def print_lstm_report(title: str, records: list[LSTMRecord]) -> None:
  required = tuple(index for index, _label in LSTM_PARAMETER_INPUTS)
  constant_records = [record for record in records if record.has_constant_inputs(required)]
  print(f"\n{title}")
  print(f"LSTM count: {len(records)}")
  print(f"LSTMs with constant W/R/B: {len(constant_records)}/{len(records)}")
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


def _validate_paths(input_path: Path, output_path: Path, overwrite: bool) -> None:
  if not input_path.is_file():
    raise FileNotFoundError(f"input model does not exist: {input_path}")
  if input_path.resolve() == output_path.resolve(strict=False):
    raise ValueError("refusing to overwrite the input ONNX model")
  if output_path.exists() and not overwrite:
    raise FileExistsError(
      f"output already exists: {output_path}; pass --overwrite to replace it"
    )


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
    raise RuntimeError(f"ONNX Runtime did not create the optimized model: {output_path}")


def _validate_folded_model(
  input_records: list[LSTMRecord],
  folded_path: Path,
) -> tuple[ModelProto, list[LSTMRecord]]:
  folded_model = onnx.load(folded_path, load_external_data=True)
  onnx.checker.check_model(folded_model)

  validation_options = ort.SessionOptions()
  validation_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
  ort.InferenceSession(
    str(folded_path),
    sess_options=validation_options,
    providers=["CPUExecutionProvider"],
  )

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
      "constant folding did not produce direct constant W/R/B inputs for: "
      + ", ".join(nonconstant)
    )
  return folded_model, folded_records


def main() -> int:
  args = _parse_args()
  input_path = args.input.expanduser().resolve()
  output_path = args.output.expanduser().absolute()
  _validate_paths(input_path, output_path, args.overwrite)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  input_hash = _sha256(input_path)
  input_model = onnx.load(input_path, load_external_data=True)
  input_records = collect_lstm_records(input_model)
  if not input_records:
    raise RuntimeError("the input model does not contain any LSTM nodes")
  print_lstm_report("Before constant folding", input_records)

  descriptor, temporary_name = tempfile.mkstemp(
    dir=output_path.parent,
    prefix=f".{output_path.stem}.",
    suffix=".onnx",
  )
  os.close(descriptor)
  temporary_path = Path(temporary_name)
  temporary_path.unlink()
  try:
    _fold_with_onnxruntime(input_path, temporary_path)
    _folded_model, folded_records = _validate_folded_model(input_records, temporary_path)
    if _sha256(input_path) != input_hash:
      raise RuntimeError("the input ONNX model changed while folding")
    if output_path.exists() and not args.overwrite:
      raise FileExistsError(f"output appeared while folding: {output_path}")
    os.replace(temporary_path, output_path)
  finally:
    temporary_path.unlink(missing_ok=True)

  print_lstm_report("After constant folding", folded_records)
  print("\nLSTM names with constant input 2 (W) and input 3 (R):")
  for record in folded_records:
    if record.has_constant_inputs((1, 2)):
      print(record.name)
  print(f"\nInput SHA256 (unchanged): {input_hash}")
  print(f"Saved folded model: {output_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
