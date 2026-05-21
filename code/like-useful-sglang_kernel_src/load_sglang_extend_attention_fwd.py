
from simo.extensions.sglang_simo.layers.attention.triton_ops.extend_attention import extend_attention_fwd
from simo.extensions.sglang_simo.quantization.quantization import parse_quantize_spec
import torch
import json
from safetensors.torch import load_file
if True:
    if True:
      time_prefix = "1776246917.2021449" # 不知道是什么格式的
      save_dir = "temp/prepare_data_simo_sglang_extend_attention_fwd/"

      safe_tensor_path = f"{save_dir}/simo_sglang_extend_attention_fwd.{time_prefix}.safetensors"
      args_json_path = f"{save_dir}/non_tensor_args.{time_prefix}.json"

      layer_dict_json_path = f"{save_dir}/layer_dict.{time_prefix}.json"

      data_dict = load_file(safe_tensor_path, device="cuda:0")

      q_extend=data_dict["q_extend"]
      k_extend=data_dict["k_extend"]
      v_extend=data_dict["v_extend"]
      o_extend=data_dict["o_extend"]
      k_buffer=data_dict["k_buffer"]
      v_buffer=data_dict["v_buffer"]
      qo_indptr=data_dict["qo_indptr"]
      kv_indptr=data_dict["kv_indptr"]
      kv_indices=data_dict["kv_indices"]

      with open(args_json_path, 'r') as fp:
        non_tensor_args = json.load(fp)

      custom_mask=non_tensor_args["custom_mask"]
      is_causal=non_tensor_args["is_causal"]
      max_len_extend=non_tensor_args["max_len_extend"]
      sm_scale=non_tensor_args["sm_scale"]
      logit_cap=non_tensor_args["logit_cap"]
      skip_prefix_custom_mask=non_tensor_args["skip_prefix_custom_mask"]
      sliding_window_size=non_tensor_args["sliding_window_size"]
      sinks=non_tensor_args["sinks"]
      window_kv_offsets=non_tensor_args["window_kv_offsets"]
      xai_temperature_len=non_tensor_args["xai_temperature_len"]
      mask_indptr=non_tensor_args["mask_indptr"]

      with open(layer_dict_json_path, 'r') as fp:
        layer_dict = json.load(fp)

      layer = torch.nn.Linear(2,3)
      layer.qk_head_dim=layer_dict["qk_head_dim"]
      layer.v_head_dim=layer_dict["v_head_dim"]
      layer.packed_head_size=layer_dict["packed_head_size"]
      layer.scale_head_size=layer_dict["scale_head_size"]
      layer.kv_cache_quant_spec=  parse_quantize_spec(layer_dict["kv_cache_quant_spec"])
      extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            is_causal,
            mask_indptr,
            max_len_extend,
            sm_scale=sm_scale,
            logit_cap=logit_cap,
            skip_prefix_custom_mask=skip_prefix_custom_mask,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=xai_temperature_len,
            layer=layer,
            )

