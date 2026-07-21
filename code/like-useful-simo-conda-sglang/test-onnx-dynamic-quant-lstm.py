import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


INPUT_SIZE = 10
HIDDEN_SIZE = 20
NUM_LAYERS = 1
MODEL_FP32 = Path("temp/test-lstm.onnx")
MODEL_QUANT = Path("temp/test-lstm-quant.onnx")
VERIFY_SHAPES = (
  (5, 3),
  (7, 2),
  (3, 4),
)


def quantize_model() -> None:
  if not MODEL_FP32.exists():
    raise FileNotFoundError(
      f"{MODEL_FP32} does not exist; run like-useful/test-lstm.py first"
    )

  MODEL_QUANT.parent.mkdir(parents=True, exist_ok=True)
  quantize_dynamic(
    model_input=MODEL_FP32,
    model_output=MODEL_QUANT,
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
  )

  model = onnx.load(MODEL_QUANT)
  onnx.checker.check_model(model)
  operators = [(node.domain or "ai.onnx", node.op_type) for node in model.graph.node]
  assert ("com.microsoft", "DynamicQuantizeLSTM") in operators, operators
  print(f"quantized_model={MODEL_QUANT}")
  print(f"quantized_operators={operators}")


def make_input_cases() -> list[tuple[int, int, dict[str, np.ndarray]]]:
  rng = np.random.default_rng(20260720)
  cases = []
  for seq_len, batch_size in VERIFY_SHAPES:
    state_shape = (NUM_LAYERS, batch_size, HIDDEN_SIZE)
    feeds = {
      "input": rng.standard_normal(
        (seq_len, batch_size, INPUT_SIZE), dtype=np.float32
      ),
      "h0": rng.standard_normal(state_shape, dtype=np.float32),
      "c0": rng.standard_normal(state_shape, dtype=np.float32),
    }
    cases.append((seq_len, batch_size, feeds))
  return cases


def run_cases(
  label: str,
  session: ort.InferenceSession,
  cases: list[tuple[int, int, dict[str, np.ndarray]]],
) -> list[list[np.ndarray]]:
  print(f"{label}_session_providers={session.get_providers()}")
  all_outputs = []
  for seq_len, batch_size, feeds in cases:
    outputs = session.run(["output", "hn", "cn"], feeds)
    expected_shapes = (
      (seq_len, batch_size, HIDDEN_SIZE),
      (NUM_LAYERS, batch_size, HIDDEN_SIZE),
      (NUM_LAYERS, batch_size, HIDDEN_SIZE),
    )
    for name, value, expected_shape in zip(
      ("output", "hn", "cn"), outputs, expected_shapes, strict=True
    ):
      assert value.shape == expected_shape, (
        f"{label} {name}: expected shape {expected_shape}, got {value.shape}"
      )
      assert np.isfinite(value).all(), f"{label} {name} contains NaN or Inf"
    print(
      f"{label}: seq_len={seq_len} batch={batch_size} "
      f"output_shapes={[value.shape for value in outputs]}"
    )
    all_outputs.append(outputs)
  return all_outputs


def compare_outputs(
  expected: list[list[np.ndarray]], actual: list[list[np.ndarray]]
) -> None:
  for case_index, (expected_case, actual_case) in enumerate(
    zip(expected, actual, strict=True)
  ):
    for name, expected_value, actual_value in zip(
      ("output", "hn", "cn"), expected_case, actual_case, strict=True
    ):
      np.testing.assert_allclose(actual_value, expected_value, rtol=1e-5, atol=1e-6)
      max_abs_diff = float(np.max(np.abs(actual_value - expected_value)))
      print(f"CPU_vs_CUDA case={case_index} {name} max_abs_diff={max_abs_diff:.9e}")


def read_profile_assignments(profile_path: Path) -> list[tuple[str, str]]:
  events = json.loads(profile_path.read_text(encoding="utf-8"))
  assignments = {
    (event.get("args", {}).get("op_name"), event.get("args", {}).get("provider"))
    for event in events
    if event.get("args", {}).get("op_name")
    and event.get("args", {}).get("provider")
  }
  profile_path.unlink(missing_ok=True)
  return sorted(assignments)


def try_cuda(
  cases: list[tuple[int, int, dict[str, np.ndarray]]],
  cpu_outputs: list[list[np.ndarray]],
) -> tuple[bool, str]:
  if "CUDAExecutionProvider" not in ort.get_available_providers():
    return False, "CUDAExecutionProvider is not included in this onnxruntime build"

  # The pip-installed NVIDIA libraries are outside the default loader search path.
  # ONNX Runtime provides this helper to preload its matching CUDA/cuDNN libraries.
  try:
    ort.preload_dlls(directory="")
  except Exception as error:
    return False, f"failed to preload CUDA/cuDNN libraries: {error}"

  profile_prefix = MODEL_QUANT.parent / "test-lstm-quant-cuda-profile"
  session_options = ort.SessionOptions()
  session_options.enable_profiling = True
  session_options.profile_file_prefix = str(profile_prefix)
  cuda_session = ort.InferenceSession(
    str(MODEL_QUANT),
    sess_options=session_options,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
  )
  active_providers = cuda_session.get_providers()
  if "CUDAExecutionProvider" not in active_providers:
    cuda_session.end_profiling()
    return False, f"CUDA EP failed to load; active providers are {active_providers}"

  cuda_outputs = run_cases("CUDA_with_CPU_fallback", cuda_session, cases)
  compare_outputs(cpu_outputs, cuda_outputs)
  profile_path = Path(cuda_session.end_profiling())
  assignments = read_profile_assignments(profile_path)
  print(f"CUDA_profile_assignments={assignments}")

  quant_lstm_providers = {
    provider
    for operator, provider in assignments
    if operator == "DynamicQuantizeLSTM"
  }
  if quant_lstm_providers != {"CUDAExecutionProvider"}:
    reason = (
      "com.microsoft::DynamicQuantizeLSTM has no CUDA EP kernel in this "
      f"ONNX Runtime build; profiling assigned it to {sorted(quant_lstm_providers)}"
    )
  else:
    reason = ""

  strict_options = ort.SessionOptions()
  strict_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
  try:
    strict_session = ort.InferenceSession(
      str(MODEL_QUANT),
      sess_options=strict_options,
      providers=["CUDAExecutionProvider"],
    )
    strict_session.disable_fallback()
    strict_outputs = run_cases("CUDA_only", strict_session, cases)
    compare_outputs(cpu_outputs, strict_outputs)
  except Exception as error:
    strict_reason = f"strict CUDA session failed: {error}"
    return False, f"{reason}; {strict_reason}" if reason else strict_reason

  return True, "all quantized LSTM nodes ran with CPU fallback disabled"


def main() -> None:
  quantize_model()
  cases = make_input_cases()

  cpu_session = ort.InferenceSession(
    str(MODEL_QUANT), providers=["CPUExecutionProvider"]
  )
  cpu_outputs = run_cases("CPU", cpu_session, cases)
  print(f"PASS: CPU verified {len(cases)} dynamic LSTM input shapes")

  cuda_succeeded, cuda_detail = try_cuda(cases, cpu_outputs)
  if cuda_succeeded:
    print(f"PASS: CUDA-only inference succeeded: {cuda_detail}")
  else:
    print(f"CUDA-only inference is unavailable: {cuda_detail}")


if __name__ == "__main__":
  main()
