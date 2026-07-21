import json
import os
import pathlib

import numpy as np
import onnx
import pytest

from simo.onnx.onnx_quant import insert_qdq_nodes
from simo.onnx.runtime import get_custom_ops_library_path
from simo.quantization.config import parse_quantize_spec

try:
  import onnxruntime as ort
except ImportError:
  ort = None

try:
  from onnxruntime.quantization.qdq_loss_debug import (
    compute_signal_to_quantization_noice_ratio as _ort_compute_sqnr,
  )
except (ImportError, AttributeError):
  _ort_compute_sqnr = None

try:
  import torch
  from simo.ops.flex_api import (
    per_block_downcast_to_fp8_or_int8,
    per_block_upcast,
    per_channel_downcast_to_fp8_or_int8,
    per_channel_upcast,
    per_group_downcast_to_fp8_or_int8_or_int4,
    per_group_upcast,
  )
  from simo.ops.fake_quant import fake_quantize_mx
  from simo.ops.formats.flexpoint.scale import ScaleModeEnum as FlexScaleModeEnum
  from simo.ops.formats.mx.scale import ScaleModeEnum
  from simo.ops.kernels.mx_trition_api import _downcast_to_mxfmt_triton, _upcast_from_mxfmt_triton
  from simo.quantization.dtypes import as_dtype
except ImportError:
  torch = None
  fake_quantize_mx = None
  per_block_downcast_to_fp8_or_int8 = None
  per_block_upcast = None
  per_channel_downcast_to_fp8_or_int8 = None
  per_channel_upcast = None
  per_group_downcast_to_fp8_or_int8_or_int4 = None
  per_group_upcast = None
  FlexScaleModeEnum = None
  ScaleModeEnum = None
  _downcast_to_mxfmt_triton = None
  _upcast_from_mxfmt_triton = None
  as_dtype = None


def _session_options_with_simo_plugin():
  if ort is None:
    pytest.skip("onnxruntime is not available")
  if "CUDAExecutionProvider" not in ort.get_available_providers():
    pytest.skip("CUDAExecutionProvider is not available")
  if torch is None or not torch.cuda.is_available():
    pytest.skip("No CUDA device is available")

  lib = os.environ.get("SIMO_ONNX_CUSTOM_OPS_LIBRARY")
  if lib:
    lib_path = pathlib.Path(lib)
  else:
    try:
      lib_path = get_custom_ops_library_path()
    except RuntimeError as exc:
      pytest.skip(str(exc))

  if not lib_path.exists():
    pytest.skip(f"SIMO ONNX custom ops library does not exist: {lib_path}")

  options = ort.SessionOptions()
  try:
    options.register_custom_ops_library(str(lib_path))
  except Exception as exc:
    if "incompatible ONNX Runtime API version" in str(exc):
      pytest.skip(str(exc))
    if "failed to load SIMO ONNX Triton" in str(exc):
      pytest.skip(str(exc))
    raise
  return options


def compute_sqnr(float_tensor, qdq_tensor):
  if _ort_compute_sqnr is not None:
    return _ort_compute_sqnr(float_tensor, qdq_tensor)

  signal = np.asarray(float_tensor, dtype=np.float64)
  noise = signal - np.asarray(qdq_tensor, dtype=np.float64)
  signal_power = np.sum(signal * signal)
  noise_power = np.sum(noise * noise)

  if noise_power == 0:
    return np.inf
  if signal_power == 0:
    return -np.inf
  return 10 * np.log10(signal_power / noise_power)


def _semantic_attrs(dtype, granularity, **attrs):
  spec_data = {"dtype": dtype}
  if "scale_mode" in attrs and attrs["scale_mode"] is not None:
    spec_data["scale_mode"] = attrs["scale_mode"]
  elif dtype == "nvfp4_e2m1":
    spec_data["scale_mode"] = "e4m3"
  if "observer_mode" in attrs and attrs["observer_mode"] is not None:
    spec_data["observer_mode"] = attrs["observer_mode"]
  if granularity == "per_channel":
    spec_data["axis"] = attrs.get("axis", 0)
  elif granularity == "per_group":
    spec_data["axis"] = attrs.get("axis", -1)
    if "group_size" in attrs:
      spec_data["group_size"] = attrs["group_size"]
  elif granularity == "per_block":
    spec_data["axis"] = attrs.get("axes", [0, 1])
    if "group_size" in attrs:
      spec_data["group_size"] = attrs["group_size"]
  if "block_size" in attrs:
    spec_data["block_size"] = attrs["block_size"]
  spec = parse_quantize_spec(spec_data)

  return {
    "scale_mode": getattr(spec, "scale_mode", "fp32"),
    "observer_mode": getattr(spec, "observer_mode", "abs_max"),
    "narrow_range": int(bool(getattr(spec, "narrow_range", True))),
    "group_size": int(getattr(spec, "group_size", None) or 1),
    "block_size": int(getattr(spec, "block_size", None) or 1),
  }


def _reference_mx_qdq(dtype, tensor, block_size, scale_mode=None, *, return_scale_dtype=False):
  if torch is None:
    pytest.skip("torch is not available")
  if scale_mode is None:
    scale_mode = ScaleModeEnum.E4M3 if dtype == "nvfp4_e2m1" else ScaleModeEnum.E8M0_FLOOR
  torch_tensor = torch.from_numpy(tensor).cuda()
  q_tensor, scale = _downcast_to_mxfmt_triton(
    torch_tensor,
    as_dtype(dtype),
    axis=-1,
    block_size=block_size,
    quant_scale_rounding_mode=scale_mode,
  )
  scale_dtype = scale.dtype
  output = torch.empty_like(torch_tensor)
  _upcast_from_mxfmt_triton(q_tensor, scale, output, as_dtype(dtype), axis=-1)
  torch.cuda.synchronize()
  if return_scale_dtype:
    return output.cpu().numpy(), scale_dtype
  return output.cpu().numpy()


def _reference_mx_qdq_with_last_dim_padding(dtype, tensor, block_size, scale_mode=None):
  pad = (-tensor.shape[-1]) % block_size
  rank2 = tensor.reshape(-1, tensor.shape[-1])
  if pad:
    rank2 = np.pad(rank2, ((0, 0), (0, pad)))
  output = _reference_mx_qdq(dtype, rank2.astype(np.float32, copy=False), block_size, scale_mode)
  return output[:, : tensor.shape[-1]].reshape(tensor.shape)


def _reference_fake_quant_mx(dtype, tensor, block_size, scale_mode):
  if torch is None or fake_quantize_mx is None:
    pytest.skip("torch fake quant is not available")
  torch_tensor = torch.from_numpy(tensor).cuda()
  output = fake_quantize_mx(
    torch_tensor,
    as_dtype(dtype),
    axis=-1,
    block_size=block_size,
    scale_mode=scale_mode,
  )
  torch.cuda.synchronize()
  return output.cpu().numpy()


def _reference_fp8_per_group_qdq(tensor, group_size):
  if torch is None:
    pytest.skip("torch is not available")
  torch_tensor = torch.from_numpy(tensor).cuda()
  q_tensor, scale = per_group_downcast_to_fp8_or_int8_or_int4(
    torch_tensor,
    as_dtype("fp8_e4m3"),
    axis=-1,
    group_size=group_size,
    scale_mode=FlexScaleModeEnum.FP32,
  )
  output = per_group_upcast(q_tensor, scale, torch.float32, group_size=group_size)
  torch.cuda.synchronize()
  return output.cpu().numpy()


def _reference_fp8_per_block_qdq(tensor, group_size):
  if torch is None:
    pytest.skip("torch is not available")
  torch_tensor = torch.from_numpy(tensor).cuda()
  q_tensor, scale = per_block_downcast_to_fp8_or_int8(
    torch_tensor,
    as_dtype("fp8_e4m3"),
    axis=[0, 1],
    group_size=group_size,
    scale_mode=FlexScaleModeEnum.FP32,
  )
  output = per_block_upcast(q_tensor, scale, torch.float32, group_size=group_size)
  torch.cuda.synchronize()
  return output.cpu().numpy()


def _reference_int8_per_block_qdq(tensor, group_size):
  if torch is None:
    pytest.skip("torch is not available")
  torch_tensor = torch.from_numpy(tensor).cuda()
  q_tensor, scale = per_block_downcast_to_fp8_or_int8(
    torch_tensor,
    as_dtype("int8"),
    axis=[0, 1],
    group_size=group_size,
    scale_mode=FlexScaleModeEnum.FP32,
  )
  output = per_block_upcast(q_tensor, scale, torch.float32, group_size=group_size)
  torch.cuda.synchronize()
  return output.cpu().numpy()


def _reference_fp8_per_channel_qdq(tensor):
  if torch is None:
    pytest.skip("torch is not available")
  torch_tensor = torch.from_numpy(tensor).cuda()
  q_tensor, scale = per_channel_downcast_to_fp8_or_int8(
    torch_tensor,
    as_dtype("fp8_e4m3"),
    axis=0,
    scale_mode=FlexScaleModeEnum.FP32,
  )
  output = per_channel_upcast(q_tensor, scale, torch.float32)
  torch.cuda.synchronize()
  return output.cpu().numpy()


def _reference_int8_per_channel_qdq(tensor):
  if torch is None:
    pytest.skip("torch is not available")
  torch_tensor = torch.from_numpy(tensor).cuda()
  q_tensor, scale = per_channel_downcast_to_fp8_or_int8(
    torch_tensor,
    as_dtype("int8"),
    axis=0,
    scale_mode=FlexScaleModeEnum.FP32,
  )
  output = per_channel_upcast(q_tensor, scale, torch.float32)
  torch.cuda.synchronize()
  return output.cpu().numpy()


def _reference_int4_per_group_dequant(q_uint8, scale_uint8, logical_shape, group_size):
  if torch is None:
    pytest.skip("torch is not available")
  q_tensor = torch.from_numpy(q_uint8.view(np.int32)).cuda()
  scale_tensor = torch.from_numpy(scale_uint8.view(np.float32)).cuda()
  output = per_group_upcast(
    q_tensor, scale_tensor, torch.float32, group_size=group_size, quant_dtype=as_dtype("int4")
  )
  torch.cuda.synchronize()
  return output.cpu().numpy().reshape(logical_shape)


def _require_simo_native_reference():
  try:
    import simo._C  # noqa: F401
  except ModuleNotFoundError as exc:
    if exc.name != "simo._C":
      raise
    pytest.skip("simo._C is required for this numerical reference")


def _insert_qdq_nodes_with_native_weight_quant(model, config_path):
  _require_simo_native_reference()
  return insert_qdq_nodes(model, config_path)


def create_simo_activation_matching(qdq_activations, float_activations=None):
  pre_suffix = "_SimoQuantInput"
  post_suffix = "_SimoDequantOutput"
  matches = {}

  for name, pre_qdq in qdq_activations.items():
    if not name.endswith(pre_suffix):
      continue

    base_name = name[: -len(pre_suffix)]
    post_name = f"{base_name}{post_suffix}"
    if post_name not in qdq_activations:
      continue

    match = {
      "pre_qdq": pre_qdq,
      "post_qdq": qdq_activations[post_name],
    }
    if float_activations is not None and base_name in float_activations:
      match["float"] = float_activations[base_name]
    matches[base_name] = match

  return matches


def test_simo_activation_matching_pairs_dynamic_qdq_names():
  pre = np.array([1.0, 2.0], dtype=np.float32)
  post = np.array([1.0, 1.9], dtype=np.float32)
  floating = np.array([1.0, 2.1], dtype=np.float32)

  matches = create_simo_activation_matching(
    {
      "layer_0_SimoQuantInput": pre,
      "layer_0_SimoDequantOutput": post,
      "layer_1_SimoQuantInput": np.array([3.0], dtype=np.float32),
      "unrelated": np.array([4.0], dtype=np.float32),
    },
    {"layer_0": floating},
  )

  assert list(matches) == ["layer_0"]
  assert matches["layer_0"]["pre_qdq"] is pre
  assert matches["layer_0"]["post_qdq"] is post
  assert matches["layer_0"]["float"] is floating
  assert np.isfinite(compute_sqnr(pre, post))


def _tiny_simo_qdq_model(tmp_path, shape, dtype="mxint8", block_size=32, scale_mode=None):
  options = _session_options_with_simo_plugin()
  input_name = "input"
  q_name = "input_SimoQuantInput"
  output_name = "input_SimoDequantOutput"

  graph = onnx.helper.make_graph(
    [
      onnx.helper.make_node(
        "Quantize",
        [input_name],
        [q_name, "input_SimoScale"],
        domain="com.simo",
        dtype=dtype,
        granularity="per_group",
        **_semantic_attrs(dtype, "per_group", block_size=block_size, scale_mode=scale_mode),
      ),
      onnx.helper.make_node(
        "Dequantize",
        [q_name, "input_SimoScale"],
        [output_name],
        domain="com.simo",
        dtype=dtype,
        granularity="per_group",
        **_semantic_attrs(dtype, "per_group", block_size=block_size, scale_mode=scale_mode),
      ),
    ],
    "tiny_simo_qdq",
    [onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, shape)],
    [onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, shape)],
  )
  model = onnx.helper.make_model(
    graph,
    opset_imports=[
      onnx.helper.make_operatorsetid("", 18),
      onnx.helper.make_operatorsetid("com.simo", 1),
    ],
  )
  model_path = tmp_path / "tiny_simo_qdq.onnx"
  onnx.save(model, model_path)
  session = ort.InferenceSession(
    str(model_path),
    sess_options=options,
    providers=["CUDAExecutionProvider"],
  )
  return session, input_name, output_name


def _dynamic_conv_activation_qdq_model(tmp_path):
  options = _session_options_with_simo_plugin()
  weight = (np.arange(128, dtype=np.float32).reshape(4, 32, 1, 1) - 64.0) / 32.0
  graph = onnx.helper.make_graph(
    [onnx.helper.make_node("Conv", ["X", "W"], ["Y"], name="conv")],
    "dynamic_conv_activation_qdq",
    [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["N", 32, "H", "W"])],
    [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, ["N", 4, "H", "W"])],
    [onnx.numpy_helper.from_array(weight, "W")],
  )
  model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_operatorsetid("", 18)])
  orig_model_path = tmp_path / "dynamic_conv_activation_orig.onnx"
  onnx.save(model, orig_model_path)
  config_path = tmp_path / "dynamic_conv_quant_config.json"
  config_path.write_text(
    json.dumps({
      "quantization_config": {
        "quant_algo": "onnx_quant",
        "quant_method": "simo",
        "module_configs": [
          {
            "targets_op_types": ["Conv"],
            "input": {"dtype": "mxint8", "axis": 1},
            "weight": {"dtype": "mxint8", "axis": 1, "block_size": 32},
          }
        ],
      }
    })
  )
  model_with_qdq = _insert_qdq_nodes_with_native_weight_quant(model, config_path)
  assert any(
    node.domain == "com.simo" and node.op_type == "Quantize" for node in model_with_qdq.graph.node
  )
  model_path = tmp_path / "dynamic_conv_activation_qdq.onnx"
  onnx.save(model_with_qdq, model_path)
  session = ort.InferenceSession(
    str(model_path),
    sess_options=options,
    providers=["CUDAExecutionProvider"],
  )
  return session


def _qdq_unaligned_matmul_activation_qdq_model(tmp_path):
  options = _session_options_with_simo_plugin()
  weight = (np.arange(72, dtype=np.float32).reshape(18, 4) - 36.0) / 64.0
  graph = onnx.helper.make_graph(
    [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"], name="matmul")],
    "unaligned_matmul_activation_qdq",
    [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 3, 18])],
    [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 3, 4])],
    [onnx.numpy_helper.from_array(weight, "W")],
  )
  model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_operatorsetid("", 18)])
  orig_model_path = tmp_path / "unaligned_matmul_activation_orig.onnx"
  onnx.save(model, orig_model_path)
  config_path = tmp_path / "unaligned_matmul_quant_config.json"
  config_path.write_text(
    json.dumps({
      "quantization_config": {
        "quant_algo": "onnx_quant",
        "quant_method": "simo",
        "module_configs": [
          {
            "targets_op_types": ["MatMul"],
            "input": {"dtype": "mxint8"},
            "weight": {"dtype": "fp8_e4m3", "axis": [0, 1], "group_size": 128},
          }
        ],
      }
    })
  )
  model_with_qdq = _insert_qdq_nodes_with_native_weight_quant(model, config_path)
  model_with_qdq.graph.output.append(
    onnx.helper.make_tensor_value_info(
      "matmul_input_simo_restore", onnx.TensorProto.FLOAT, [2, 3, 18]
    )
  )
  model_path = tmp_path / "unaligned_matmul_activation_qdq.onnx"
  onnx.save(model_with_qdq, model_path)
  session = ort.InferenceSession(
    str(model_path),
    sess_options=options,
    providers=["CUDAExecutionProvider"],
  )
  return session


def _qdq_fp8_per_group_padded_rank2_slice_model(tmp_path):
  options = _session_options_with_simo_plugin()
  options.enable_mem_pattern = False
  options.enable_mem_reuse = False
  weight = (np.arange(1280, dtype=np.float32).reshape(320, 4) - 640.0) / 640.0
  graph = onnx.helper.make_graph(
    [onnx.helper.make_node("MatMul", ["X", "W"], ["Y"], name="matmul")],
    "fp8_per_group_padded_rank2_slice",
    [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["N", 320])],
    [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, ["N", 4])],
    [onnx.numpy_helper.from_array(weight, "W")],
  )
  model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_operatorsetid("", 18)])
  config_path = tmp_path / "fp8_per_group_padded_rank2_quant_config.json"
  config_path.write_text(
    json.dumps({
      "quantization_config": {
        "quant_algo": "onnx_quant",
        "quant_method": "simo",
        "module_configs": [
          {
            "targets_op_types": ["MatMul"],
            "input": {"dtype": "fp8_e4m3", "axis": 1, "group_size": 128},
            "weight": {"dtype": "fp8_e4m3", "axis": [0, 1], "group_size": 128},
          }
        ],
      }
    })
  )
  model_with_qdq = _insert_qdq_nodes_with_native_weight_quant(model, config_path)
  initializers = {init.name: init for init in model_with_qdq.graph.initializer}
  assert onnx.numpy_helper.to_array(initializers["matmul_input_simo_unpad_axes"]).tolist() == [1]
  assert onnx.numpy_helper.to_array(initializers["matmul_input_simo_unpad_ends"]).tolist() == [320]
  model_with_qdq.graph.output.append(
    onnx.helper.make_tensor_value_info(
      "matmul_input_simo_restore", onnx.TensorProto.FLOAT, ["N", 320]
    )
  )
  model_path = tmp_path / "fp8_per_group_padded_rank2_slice.onnx"
  onnx.save(model_with_qdq, model_path)
  session = ort.InferenceSession(
    str(model_path),
    sess_options=options,
    providers=["CUDAExecutionProvider"],
  )
  return session


def _tiny_simo_fp8_per_group_qdq_model(tmp_path, shape, group_size=128):
  options = _session_options_with_simo_plugin()
  input_name = "input"
  q_name = "input_SimoQuantInput"
  output_name = "input_SimoDequantOutput"

  graph = onnx.helper.make_graph(
    [
      onnx.helper.make_node(
        "Quantize",
        [input_name],
        [q_name, "input_SimoScale"],
        domain="com.simo",
        dtype="fp8_e4m3",
        granularity="per_group",
        axis=-1,
        **_semantic_attrs("fp8_e4m3", "per_group", group_size=group_size),
      ),
      onnx.helper.make_node(
        "Dequantize",
        [q_name, "input_SimoScale"],
        [output_name],
        domain="com.simo",
        dtype="fp8_e4m3",
        granularity="per_group",
        axis=-1,
        **_semantic_attrs("fp8_e4m3", "per_group", group_size=group_size),
      ),
    ],
    "tiny_simo_fp8_per_group_qdq",
    [onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, shape)],
    [onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, shape)],
  )
  model = onnx.helper.make_model(
    graph,
    opset_imports=[
      onnx.helper.make_operatorsetid("", 18),
      onnx.helper.make_operatorsetid("com.simo", 1),
    ],
  )
  model_path = tmp_path / "tiny_simo_fp8_per_group_qdq.onnx"
  onnx.save(model, model_path)
  session = ort.InferenceSession(
    str(model_path),
    sess_options=options,
    providers=["CUDAExecutionProvider"],
  )
  return session, input_name, output_name


def _tiny_simo_fp8_per_block_qdq_model(tmp_path, shape, group_size=128):
  options = _session_options_with_simo_plugin()
  input_name = "input"
  q_name = "input_SimoQuantInput"
  output_name = "input_SimoDequantOutput"

  graph = onnx.helper.make_graph(
    [
      onnx.helper.make_node(
        "Quantize",
        [input_name],
        [q_name, "input_SimoScale"],
        domain="com.simo",
        dtype="fp8_e4m3",
        granularity="per_block",
        axes=[0, 1],
        **_semantic_attrs("fp8_e4m3", "per_block", group_size=group_size),
      ),
      onnx.helper.make_node(
        "Dequantize",
        [q_name, "input_SimoScale"],
        [output_name],
        domain="com.simo",
        dtype="fp8_e4m3",
        granularity="per_block",
        axes=[0, 1],
        **_semantic_attrs("fp8_e4m3", "per_block", group_size=group_size),
      ),
    ],
    "tiny_simo_fp8_per_block_qdq",
    [onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, shape)],
    [onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, shape)],
  )
  model = onnx.helper.make_model(
    graph,
    opset_imports=[
      onnx.helper.make_operatorsetid("", 18),
      onnx.helper.make_operatorsetid("com.simo", 1),
    ],
  )
  model_path = tmp_path / "tiny_simo_fp8_per_block_qdq.onnx"
  onnx.save(model, model_path)
  session = ort.InferenceSession(
    str(model_path),
    sess_options=options,
    providers=["CUDAExecutionProvider"],
  )
  return session, input_name, output_name


def _tiny_simo_int8_per_block_qdq_model(tmp_path, shape, group_size=128):
  options = _session_options_with_simo_plugin()
  input_name = "input"
  q_name = "input_SimoQuantInput"
  output_name = "input_SimoDequantOutput"

  graph = onnx.helper.make_graph(
    [
      onnx.helper.make_node(
        "Quantize",
        [input_name],
        [q_name, "input_SimoScale"],
        domain="com.simo",
        dtype="int8",
        granularity="per_block",
        axes=[0, 1],
        **_semantic_attrs("int8", "per_block", group_size=group_size),
      ),
      onnx.helper.make_node(
        "Dequantize",
        [q_name, "input_SimoScale"],
        [output_name],
        domain="com.simo",
        dtype="int8",
        granularity="per_block",
        axes=[0, 1],
        **_semantic_attrs("int8", "per_block", group_size=group_size),
      ),
    ],
    "tiny_simo_int8_per_block_qdq",
    [onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, shape)],
    [onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, shape)],
  )
  model = onnx.helper.make_model(
    graph,
    opset_imports=[
      onnx.helper.make_operatorsetid("", 18),
      onnx.helper.make_operatorsetid("com.simo", 1),
    ],
  )
  model_path = tmp_path / "tiny_simo_int8_per_block_qdq.onnx"
  onnx.save(model, model_path)
  session = ort.InferenceSession(
    str(model_path),
    sess_options=options,
    providers=["CUDAExecutionProvider"],
  )
  return session, input_name, output_name


def _tiny_simo_fp8_per_channel_qdq_model(tmp_path, shape):
  options = _session_options_with_simo_plugin()
  input_name = "input"
  q_name = "input_SimoQuantInput"
  output_name = "input_SimoDequantOutput"

  graph = onnx.helper.make_graph(
    [
      onnx.helper.make_node(
        "Quantize",
        [input_name],
        [q_name, "input_SimoScale"],
        domain="com.simo",
        dtype="fp8_e4m3",
        granularity="per_channel",
        axis=0,
        **_semantic_attrs("fp8_e4m3", "per_channel"),
      ),
      onnx.helper.make_node(
        "Dequantize",
        [q_name, "input_SimoScale"],
        [output_name],
        domain="com.simo",
        dtype="fp8_e4m3",
        granularity="per_channel",
        axis=0,
        **_semantic_attrs("fp8_e4m3", "per_channel"),
      ),
    ],
    "tiny_simo_fp8_per_channel_qdq",
    [onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, shape)],
    [onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, shape)],
  )
  model = onnx.helper.make_model(
    graph,
    opset_imports=[
      onnx.helper.make_operatorsetid("", 18),
      onnx.helper.make_operatorsetid("com.simo", 1),
    ],
  )
  model_path = tmp_path / "tiny_simo_fp8_per_channel_qdq.onnx"
  onnx.save(model, model_path)
  session = ort.InferenceSession(
    str(model_path),
    sess_options=options,
    providers=["CUDAExecutionProvider"],
  )
  return session, input_name, output_name


def _tiny_simo_int8_per_channel_qdq_model(tmp_path, shape):
  options = _session_options_with_simo_plugin()
  input_name = "input"
  q_name = "input_SimoQuantInput"
  output_name = "input_SimoDequantOutput"

  graph = onnx.helper.make_graph(
    [
      onnx.helper.make_node(
        "Quantize",
        [input_name],
        [q_name, "input_SimoScale"],
        domain="com.simo",
        dtype="int8",
        granularity="per_channel",
        axis=0,
        **_semantic_attrs("int8", "per_channel"),
      ),
      onnx.helper.make_node(
        "Dequantize",
        [q_name, "input_SimoScale"],
        [output_name],
        domain="com.simo",
        dtype="int8",
        granularity="per_channel",
        axis=0,
        **_semantic_attrs("int8", "per_channel"),
      ),
    ],
    "tiny_simo_int8_per_channel_qdq",
    [onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, shape)],
    [onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, shape)],
  )
  model = onnx.helper.make_model(
    graph,
    opset_imports=[
      onnx.helper.make_operatorsetid("", 18),
      onnx.helper.make_operatorsetid("com.simo", 1),
    ],
  )
  model_path = tmp_path / "tiny_simo_int8_per_channel_qdq.onnx"
  onnx.save(model, model_path)
  session = ort.InferenceSession(
    str(model_path),
    sess_options=options,
    providers=["CUDAExecutionProvider"],
  )
  return session, input_name, output_name


def _tiny_simo_int4_per_group_dequant_model(tmp_path, logical_shape, group_size=32):
  options = _session_options_with_simo_plugin()
  q_name = "Q"
  scale_name = "S"
  output_name = "Y"
  packed_shape = [logical_shape[0], logical_shape[1] // 8 * 4]
  scale_shape = [logical_shape[0], logical_shape[1] // group_size * 4]

  graph = onnx.helper.make_graph(
    [
      onnx.helper.make_node(
        "Dequantize",
        [q_name, scale_name],
        [output_name],
        domain="com.simo",
        dtype="int4",
        granularity="per_group",
        axis=-1,
        **_semantic_attrs("int4", "per_group", group_size=group_size),
      ),
    ],
    "tiny_simo_int4_per_group_dequant",
    [
      onnx.helper.make_tensor_value_info(q_name, onnx.TensorProto.UINT8, packed_shape),
      onnx.helper.make_tensor_value_info(scale_name, onnx.TensorProto.UINT8, scale_shape),
    ],
    [onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, logical_shape)],
  )
  model = onnx.helper.make_model(
    graph,
    opset_imports=[
      onnx.helper.make_operatorsetid("", 18),
      onnx.helper.make_operatorsetid("com.simo", 1),
    ],
  )
  model_path = tmp_path / "tiny_simo_int4_per_group_dequant.onnx"
  onnx.save(model, model_path)
  session = ort.InferenceSession(
    str(model_path),
    sess_options=options,
    providers=["CUDAExecutionProvider"],
  )
  return session, q_name, scale_name, output_name


def test_simo_custom_qdq_plugin_rejects_unsupported_mxint8_shape(tmp_path):
  session, input_name, output_name = _tiny_simo_qdq_model(tmp_path, [2, 2])
  tensor = np.array([[-1.0, 0.0], [0.5, 1.0]], dtype=np.float32)

  with pytest.raises(Exception, match="K % 32 == 0"):
    session.run([output_name], {input_name: tensor})


@pytest.mark.parametrize(
  ("dtype", "block_size"),
  [
    ("mxint8", 32),
    ("mxfp8_e5m2", 32),
    ("mxfp8_e4m3", 32),
    ("mxfp6_e3m2", 32),
    ("mxfp6_e2m3", 32),
    ("mxfp4_e2m1", 32),
    ("nvfp4_e2m1", 16),
  ],
)
def test_simo_custom_qdq_plugin_runs_mx_qdq_tiny_tensor(tmp_path, dtype, block_size):
  session, input_name, output_name = _tiny_simo_qdq_model(
    tmp_path,
    [128, 32],
    dtype=dtype,
    block_size=block_size,
  )
  tensor = np.linspace(-4.0, 4.0, 128 * 32, dtype=np.float32).reshape(128, 32)

  (actual,) = session.run([output_name], {input_name: tensor})

  assert actual.shape == tensor.shape
  assert actual.dtype == np.float32
  assert np.isfinite(compute_sqnr(tensor, actual))
  np.testing.assert_allclose(
    actual,
    _reference_mx_qdq(dtype, tensor, block_size),
    rtol=1e-3,
    atol=1e-3,
  )


@pytest.mark.parametrize(
  "dtype",
  ["mxint8", "mxfp8_e5m2", "mxfp8_e4m3", "mxfp6_e3m2", "mxfp6_e2m3", "mxfp4_e2m1"],
)
def test_simo_custom_qdq_plugin_runs_mx_e8m0_sipu_tiny_tensor(tmp_path, dtype):
  block_size = 32
  session, input_name, output_name = _tiny_simo_qdq_model(
    tmp_path,
    [4, block_size],
    dtype=dtype,
    block_size=block_size,
    scale_mode="e8m0_sipu",
  )
  tensor = np.zeros((4, block_size), dtype=np.float32)
  tensor[:, 0] = np.array([1.9, 3.7, 7.5, 15.0], dtype=np.float32)
  tensor[:, 1] = -tensor[:, 0] / 3.0

  (actual,) = session.run([output_name], {input_name: tensor})
  expected, scale_dtype = _reference_mx_qdq(
    dtype,
    tensor,
    block_size,
    ScaleModeEnum.E8M0_SIPU,
    return_scale_dtype=True,
  )

  assert scale_dtype == torch.uint8

  np.testing.assert_allclose(
    actual,
    expected,
    rtol=1e-3,
    atol=1e-3,
  )


def test_simo_custom_qdq_plugin_matches_torch_fake_quant_for_kws_mxfp8_e5m2_activation(tmp_path):
  block_size = 32
  session, input_name, output_name = _tiny_simo_qdq_model(
    tmp_path,
    [4, 128],
    dtype="mxfp8_e5m2",
    block_size=block_size,
    scale_mode="e8m0_sipu",
  )
  tensor = np.linspace(-0.75, 0.75, 4 * 128, dtype=np.float32).reshape(4, 128)
  tensor[2, 106] = -0.0361
  tensor[3, 120] = 0.5

  (actual,) = session.run([output_name], {input_name: tensor})
  expected = _reference_fake_quant_mx(
    "mxfp8_e5m2",
    tensor,
    block_size,
    ScaleModeEnum.E8M0_SIPU,
  )

  np.testing.assert_allclose(
    actual,
    expected,
    rtol=1e-3,
    atol=1e-3,
  )


def test_dynamic_conv_activation_qdq_runs_with_symbolic_batch_and_spatial_dims(tmp_path):
  session = _dynamic_conv_activation_qdq_model(tmp_path)
  tensor = np.linspace(-1.0, 1.0, 2 * 32 * 5 * 7, dtype=np.float32).reshape(2, 32, 5, 7)

  (actual,) = session.run(["Y"], {"X": tensor})

  assert actual.shape == (2, 4, 5, 7)
  assert actual.dtype == np.float32
  assert np.isfinite(actual).all()


def test_unaligned_matmul_activation_qdq_matches_simo_torch_with_padding(tmp_path):
  session = _qdq_unaligned_matmul_activation_qdq_model(tmp_path)
  tensor = np.linspace(-3.0, 3.0, 2 * 3 * 18, dtype=np.float32).reshape(2, 3, 18)

  (actual,) = session.run(["matmul_input_simo_restore"], {"X": tensor})

  assert actual.shape == tensor.shape
  np.testing.assert_allclose(
    actual,
    _reference_mx_qdq_with_last_dim_padding("mxint8", tensor, 32),
    rtol=1e-3,
    atol=1e-3,
  )


def test_simo_custom_qdq_plugin_loads_fp8_per_group_padded_rank2_slice(tmp_path):
  session = _qdq_fp8_per_group_padded_rank2_slice_model(tmp_path)
  tensor = np.linspace(-3.0, 3.0, 2 * 320, dtype=np.float32).reshape(2, 320)

  (actual,) = session.run(["matmul_input_simo_restore"], {"X": tensor})

  assert actual.shape == tensor.shape
  assert actual.dtype == np.float32
  assert np.isfinite(actual).all()


def test_simo_custom_qdq_plugin_runs_fp8_per_group_tiny_tensor(tmp_path):
  group_size = 128
  session, input_name, output_name = _tiny_simo_fp8_per_group_qdq_model(
    tmp_path,
    [32, group_size],
    group_size=group_size,
  )
  tensor = np.linspace(-8.0, 8.0, 32 * group_size, dtype=np.float32).reshape(32, group_size)

  (actual,) = session.run([output_name], {input_name: tensor})

  assert actual.shape == tensor.shape
  assert actual.dtype == np.float32
  assert np.isfinite(compute_sqnr(tensor, actual))
  _require_simo_native_reference()
  np.testing.assert_allclose(
    actual,
    _reference_fp8_per_group_qdq(tensor, group_size),
    rtol=1e-3,
    atol=1e-3,
  )


def test_simo_custom_qdq_plugin_runs_fp8_per_block_tiny_tensor(tmp_path):
  group_size = 128
  session, input_name, output_name = _tiny_simo_fp8_per_block_qdq_model(
    tmp_path,
    [130, 129],
    group_size=group_size,
  )
  tensor = np.linspace(-8.0, 8.0, 130 * 129, dtype=np.float32).reshape(130, 129)

  (actual,) = session.run([output_name], {input_name: tensor})

  assert actual.shape == tensor.shape
  assert actual.dtype == np.float32
  assert np.isfinite(compute_sqnr(tensor, actual))
  _require_simo_native_reference()
  np.testing.assert_allclose(
    actual,
    _reference_fp8_per_block_qdq(tensor, group_size),
    rtol=1e-3,
    atol=1e-3,
  )


def test_simo_custom_qdq_plugin_runs_int8_per_block_tiny_tensor(tmp_path):
  group_size = 128
  session, input_name, output_name = _tiny_simo_int8_per_block_qdq_model(
    tmp_path,
    [130, 129],
    group_size=group_size,
  )
  tensor = np.linspace(-8.0, 8.0, 130 * 129, dtype=np.float32).reshape(130, 129)

  (actual,) = session.run([output_name], {input_name: tensor})

  assert actual.shape == tensor.shape
  assert actual.dtype == np.float32
  assert np.isfinite(compute_sqnr(tensor, actual))
  _require_simo_native_reference()
  np.testing.assert_allclose(
    actual,
    _reference_int8_per_block_qdq(tensor, group_size),
    rtol=1e-3,
    atol=1e-3,
  )


def test_simo_custom_qdq_plugin_runs_fp8_per_channel_tiny_tensor(tmp_path):
  session, input_name, output_name = _tiny_simo_fp8_per_channel_qdq_model(tmp_path, [33, 65])
  tensor = np.linspace(-8.0, 8.0, 33 * 65, dtype=np.float32).reshape(33, 65)

  (actual,) = session.run([output_name], {input_name: tensor})

  assert actual.shape == tensor.shape
  assert actual.dtype == np.float32
  assert np.isfinite(compute_sqnr(tensor, actual))
  _require_simo_native_reference()
  np.testing.assert_allclose(
    actual,
    _reference_fp8_per_channel_qdq(tensor),
    rtol=1e-3,
    atol=1e-3,
  )


def test_simo_custom_qdq_plugin_runs_int8_per_channel_tiny_tensor(tmp_path):
  session, input_name, output_name = _tiny_simo_int8_per_channel_qdq_model(tmp_path, [33, 65])
  tensor = np.linspace(-8.0, 8.0, 33 * 65, dtype=np.float32).reshape(33, 65)

  (actual,) = session.run([output_name], {input_name: tensor})

  assert actual.shape == tensor.shape
  assert actual.dtype == np.float32
  assert np.isfinite(compute_sqnr(tensor, actual))
  _require_simo_native_reference()
  np.testing.assert_allclose(
    actual,
    _reference_int8_per_channel_qdq(tensor),
    rtol=1e-3,
    atol=1e-3,
  )


def test_simo_custom_dequant_plugin_runs_int4_per_group_tiny_tensor(tmp_path):
  group_size = 32
  logical_shape = [3, 64]
  session, q_name, scale_name, output_name = _tiny_simo_int4_per_group_dequant_model(
    tmp_path,
    logical_shape,
    group_size=group_size,
  )
  q_i32 = np.arange(logical_shape[0] * logical_shape[1] // 8, dtype=np.int32).reshape(
    logical_shape[0], logical_shape[1] // 8
  )
  scale = np.linspace(
    0.01, 0.03, logical_shape[0] * (logical_shape[1] // group_size), dtype=np.float32
  ).reshape(logical_shape[0], logical_shape[1] // group_size)
  q_uint8 = q_i32.view(np.uint8)
  scale_uint8 = scale.view(np.uint8)

  (actual,) = session.run([output_name], {q_name: q_uint8, scale_name: scale_uint8})

  assert actual.shape == tuple(logical_shape)
  assert actual.dtype == np.float32
  _require_simo_native_reference()
  np.testing.assert_allclose(
    actual,
    _reference_int4_per_group_dequant(q_uint8, scale_uint8, logical_shape, group_size),
    rtol=1e-3,
    atol=1e-3,
  )
temp_path = pathlib.Path("temp/test_dynamic_qdq_runtime_debug-debug")
test_unaligned_matmul_activation_qdq_matches_simo_torch_with_padding(temp_path)
test_dynamic_conv_activation_qdq_runs_with_symbolic_batch_and_spatial_dims(temp_path)
#for (dtype, block_size) in [("mxint8", 32), ("mxfp8_e5m2", 32), ("mxfp8_e4m3", 32), ("mxfp6_e3m2", 32), ("mxfp6_e2m3", 32), ("mxfp4_e2m1", 32), ("nvfp4_e2m1", 16)]:
#  test_simo_custom_qdq_plugin_runs_mx_qdq_tiny_tensor(temp_path, dtype, block_size)
#  print("========================================")
#
#
