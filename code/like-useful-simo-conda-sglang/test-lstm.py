from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


INPUT_SIZE = 10
HIDDEN_SIZE = 20
NUM_LAYERS = 1
NUM_DIRECTIONS = 1
ONNX_PATH = Path("temp/test-lstm.onnx")
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
  assert "LSTM" in op_types, f"Expected an ONNX LSTM node, got {op_types}"
  print(f"exported={ONNX_PATH} ops={op_types}")


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

  actual = session.run(
    ["output", "hn", "cn"],
    {
      "input": input_tensor.numpy(),
      "h0": h0.numpy(),
      "c0": c0.numpy(),
    },
  )

  print(f"case seq_len={seq_len} batch={batch_size}")
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
    print(f"  {name}: shape={actual_value.shape} max_abs_diff={max_abs_diff:.9e}")


def main() -> None:
  torch.manual_seed(20260720)
  input_generator = torch.Generator().manual_seed(20260721)
  model = LSTMExportWrapper(
    nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, num_layers=NUM_LAYERS)
  ).eval()

  example_inputs = make_inputs(*EXPORT_SHAPE, input_generator)
  export_dynamic_onnx(model, example_inputs)

  session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
  verify_dynamic_metadata(session)
  for seq_len, batch_size in VERIFY_SHAPES:
    verify_case(model, session, seq_len, batch_size, input_generator)

  print(f"PASS: verified {len(VERIFY_SHAPES)} dynamic LSTM input shapes")


if __name__ == "__main__":
  main()
