from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


INPUT_SIZE = 10
HIDDEN_SIZE = 20
NUM_LAYERS = 1
NUM_DIRECTIONS = 2
ONNX_PATH = Path("temp/test-lstm-bidirectional.onnx")
EXPORT_SHAPE = (5, 3)
VERIFY_SHAPES = (
  (5, 3),
  (7, 2),
  (3, 4),
)


class LSTMExportWrapper(nn.Module):
  def __init__(self, lstm: nn.LSTM):
    super().__init__()
    self.lstm = lstm

  def forward(self, input_tensor, h0, c0):
    output, (hn, cn) = self.lstm(input_tensor, (h0, c0))
    return output, hn, cn


def make_inputs(seq_len: int, batch_size: int, generator: torch.Generator):
  input_tensor = torch.randn(seq_len, batch_size, INPUT_SIZE, generator=generator)
  state_shape = (NUM_LAYERS * NUM_DIRECTIONS, batch_size, HIDDEN_SIZE)
  h0 = torch.randn(*state_shape, generator=generator)
  c0 = torch.randn(*state_shape, generator=generator)
  return input_tensor, h0, c0


def _manual_lstm_direction(
  input_tensor: torch.Tensor,
  initial_h: torch.Tensor,
  initial_c: torch.Tensor,
  weight_ih: torch.Tensor,
  weight_hh: torch.Tensor,
  bias_ih: torch.Tensor,
  bias_hh: torch.Tensor,
  *,
  reverse: bool,
):
  w_ii, w_if, w_ig, w_io = weight_ih.chunk(4, dim=0)
  w_hi, w_hf, w_hg, w_ho = weight_hh.chunk(4, dim=0)
  b_ii, b_if, b_ig, b_io = bias_ih.chunk(4, dim=0)
  b_hi, b_hf, b_hg, b_ho = bias_hh.chunk(4, dim=0)

  h_t = initial_h
  c_t = initial_c
  output_steps = []
  if reverse:
    time_indices = range(input_tensor.size(0) - 1, -1, -1)
  else:
    time_indices = range(input_tensor.size(0))

  for time_index in time_indices:
    x_t = input_tensor[time_index]
    i_t = torch.sigmoid(
      torch.matmul(x_t, w_ii.T) + b_ii
      + torch.matmul(h_t, w_hi.T) + b_hi
    )
    f_t = torch.sigmoid(
      torch.matmul(x_t, w_if.T) + b_if
      + torch.matmul(h_t, w_hf.T) + b_hf
    )
    g_t = torch.tanh(
      torch.matmul(x_t, w_ig.T) + b_ig
      + torch.matmul(h_t, w_hg.T) + b_hg
    )
    o_t = torch.sigmoid(
      torch.matmul(x_t, w_io.T) + b_io
      + torch.matmul(h_t, w_ho.T) + b_ho
    )
    c_t = f_t * c_t + i_t * g_t
    h_t = o_t * torch.tanh(c_t)
    output_steps.append(h_t)

  # Reverse recurrence visits L-1 ... 0. Restore output to the original
  # sequence positions before concatenating it with the forward direction.
  if reverse:
    output_steps.reverse()

  output = torch.stack(output_steps, dim=0)
  return output, h_t, c_t


def manual_lstm_forward(
  lstm: nn.LSTM,
  input_tensor: torch.Tensor,
  h0: torch.Tensor,
  c0: torch.Tensor,
):
  """Evaluate this script's bidirectional LSTM from its gate equations."""
  if lstm.num_layers != 1:
    raise ValueError("manual_lstm_forward only supports num_layers=1")
  if not lstm.bidirectional:
    raise ValueError("manual_lstm_forward expects a bidirectional LSTM")
  if lstm.batch_first:
    raise ValueError("manual_lstm_forward expects (seq_len, batch, input_size)")
  if lstm.proj_size != 0:
    raise ValueError("manual_lstm_forward does not support LSTM projections")
  if not lstm.bias:
    raise ValueError("manual_lstm_forward expects both LSTM bias tensors")

  # Each direction owns four registered parameters. Every parameter stores the
  # input, forget, cell, and output gate values concatenated in I/F/G/O order.
  forward = _manual_lstm_direction(
    input_tensor,
    h0[0],
    c0[0],
    lstm.weight_ih_l0,
    lstm.weight_hh_l0,
    lstm.bias_ih_l0,
    lstm.bias_hh_l0,
    reverse=False,
  )
  backward = _manual_lstm_direction(
    input_tensor,
    h0[1],
    c0[1],
    lstm.weight_ih_l0_reverse,
    lstm.weight_hh_l0_reverse,
    lstm.bias_ih_l0_reverse,
    lstm.bias_hh_l0_reverse,
    reverse=True,
  )

  forward_output, forward_hn, forward_cn = forward
  backward_output, backward_hn, backward_cn = backward
  output = torch.cat((forward_output, backward_output), dim=2)
  hn = torch.stack((forward_hn, backward_hn), dim=0)
  cn = torch.stack((forward_cn, backward_cn), dim=0)
  return output, hn, cn


def print_lstm_parameter_shapes(lstm: nn.LSTM) -> None:
  parameter_names = (
    "weight_ih_l0",
    "weight_hh_l0",
    "bias_ih_l0",
    "bias_hh_l0",
    "weight_ih_l0_reverse",
    "weight_hh_l0_reverse",
    "bias_ih_l0_reverse",
    "bias_hh_l0_reverse",
  )
  print("Bidirectional LSTM parameters (I/F/G/O concatenation):")
  for name in parameter_names:
    parameter = getattr(lstm, name)
    print(f"  {name}: shape={tuple(parameter.shape)}")


def export_dynamic_onnx(model: nn.Module, example_inputs) -> None:
  ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
  with torch.no_grad():
    torch.onnx.export(
      model,
      example_inputs,
      ONNX_PATH,
      input_names=["input", "h0", "c0"],
      output_names=["output", "hn", "cn"],
      opset_version=18,
      dynamo=False,
      dynamic_axes={
        "input": {0: "seq_len", 1: "batch"},
        "h0": {1: "batch"},
        "c0": {1: "batch"},
        "output": {0: "seq_len", 1: "batch"},
        "hn": {1: "batch"},
        "cn": {1: "batch"},
      },
    )

  onnx_model = onnx.load(ONNX_PATH)
  onnx.checker.check_model(onnx_model)
  op_types = [node.op_type for node in onnx_model.graph.node]
  lstm_nodes = [node for node in onnx_model.graph.node if node.op_type == "LSTM"]
  assert len(lstm_nodes) == 1, f"Expected one ONNX LSTM node, got {op_types}"
  direction_attributes = {
    attribute.name: onnx.helper.get_attribute_value(attribute)
    for attribute in lstm_nodes[0].attribute
  }
  assert direction_attributes.get("direction") == b"bidirectional", (
    f"Expected direction=bidirectional, got {direction_attributes}"
  )
  print(
    f"exported={ONNX_PATH} direction=bidirectional ops={op_types}"
  )


def verify_dynamic_metadata(session: ort.InferenceSession) -> None:
  input_shapes = {value.name: value.shape for value in session.get_inputs()}
  output_shapes = {value.name: value.shape for value in session.get_outputs()}
  assert input_shapes == {
    "input": ["seq_len", "batch", INPUT_SIZE],
    "h0": [NUM_LAYERS * NUM_DIRECTIONS, "batch", HIDDEN_SIZE],
    "c0": [NUM_LAYERS * NUM_DIRECTIONS, "batch", HIDDEN_SIZE],
  }
  assert output_shapes == {
    "output": ["seq_len", "batch", NUM_DIRECTIONS * HIDDEN_SIZE],
    "hn": [NUM_LAYERS * NUM_DIRECTIONS, "batch", HIDDEN_SIZE],
    "cn": [NUM_LAYERS * NUM_DIRECTIONS, "batch", HIDDEN_SIZE],
  }
  print(f"dynamic_inputs={input_shapes}")
  print(f"dynamic_outputs={output_shapes}")


def verify_case(
  model: nn.Module,
  session: ort.InferenceSession,
  seq_len: int,
  batch_size: int,
  generator: torch.Generator,
) -> None:
  input_tensor, h0, c0 = make_inputs(seq_len, batch_size, generator)

  with torch.no_grad():
    expected = model(input_tensor, h0, c0)
    manual = manual_lstm_forward(model.lstm, input_tensor, h0, c0)

  print(f"case seq_len={seq_len} batch={batch_size}")
  for name, manual_tensor, expected_tensor in zip(
    ("output", "hn", "cn"), manual, expected, strict=True
  ):
    torch.testing.assert_close(
      manual_tensor, expected_tensor, rtol=1e-5, atol=1e-6
    )
    max_abs_diff = float(torch.max(torch.abs(manual_tensor - expected_tensor)))
    print(
      f"  manual vs PyTorch {name}: shape={tuple(manual_tensor.shape)} "
      f"max_abs_diff={max_abs_diff:.9e}"
    )

  actual = session.run(
    ["output", "hn", "cn"],
    {
      "input": input_tensor.numpy(),
      "h0": h0.numpy(),
      "c0": c0.numpy(),
    },
  )

  for name, actual_value, expected_tensor in zip(
    ("output", "hn", "cn"), actual, expected, strict=True
  ):
    expected_value = expected_tensor.numpy()
    assert actual_value.shape == expected_value.shape, (
      f"{name} shape mismatch: ONNX Runtime {actual_value.shape}, "
      f"PyTorch {expected_value.shape}"
    )
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-5, atol=1e-6)
    max_abs_diff = float(np.max(np.abs(actual_value - expected_value)))
    print(
      f"  ONNX Runtime vs PyTorch {name}: shape={actual_value.shape} "
      f"max_abs_diff={max_abs_diff:.9e}"
    )


def main() -> None:
  torch.manual_seed(20260720)
  input_generator = torch.Generator().manual_seed(20260721)
  model = LSTMExportWrapper(
    nn.LSTM(
      INPUT_SIZE,
      HIDDEN_SIZE,
      num_layers=NUM_LAYERS,
      bidirectional=True,
    )
  ).eval()
  print_lstm_parameter_shapes(model.lstm)

  example_inputs = make_inputs(*EXPORT_SHAPE, input_generator)
  export_dynamic_onnx(model, example_inputs)

  session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
  verify_dynamic_metadata(session)
  for seq_len, batch_size in VERIFY_SHAPES:
    verify_case(model, session, seq_len, batch_size, input_generator)

  print(
    "PASS: manual bidirectional LSTM matches PyTorch and ONNX Runtime "
    "matches PyTorch "
    f"for {len(VERIFY_SHAPES)} dynamic input shapes"
  )


if __name__ == "__main__":
  main()
