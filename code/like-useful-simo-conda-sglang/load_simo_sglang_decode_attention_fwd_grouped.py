from simo.extensions.sglang_simo.layers.attention.triton_ops.decode_attention import decode_attention_fwd_grouped
from simo.extensions.sglang_simo.quantization.quantization import parse_quantize_spec
import torch
import json
from safetensors.torch import load_file
if True:
    if True:
      time_prefix = "1775983258.1490386"
      save_dir = "temp/debug_triton_decode_attention_fwd_grouped/"
      safe_tensor_path = f"{save_dir}/decode_attention_fwd_grouped.{time_prefix}.safetensors"
      args_json_path = f"{save_dir}/non_tensor_args.{time_prefix}.json"
      layer_dict_json_path = f"{save_dir}/layer_dict.{time_prefix}.json"
      data_dict = load_file(safe_tensor_path, device="cuda:0")

      q=data_dict["q"]
      k_buffer=data_dict["k_buffer"]
      v_buffer=data_dict["v_buffer"]
      o=data_dict["o"]
      kv_indptr=data_dict["kv_indptr"]
      kv_indices=data_dict["kv_indices"]
      attn_logits=data_dict["attn_logits"]
      attn_lse=data_dict["attn_lse"]
      num_kv_splits=data_dict["num_kv_splits"]

      with open(args_json_path, 'r') as fp:
        non_tensor_args = json.load(fp)

      max_kv_splits=non_tensor_args["max_kv_splits"]
      sm_scale=non_tensor_args["sm_scale"]
      logit_cap=non_tensor_args["logit_cap"]
      sinks=non_tensor_args["sinks"]
      xai_temperature_len=non_tensor_args["xai_temperature_len"]

      with open(layer_dict_json_path, 'r') as fp:
        layer_dict = json.load(fp)

      layer = torch.nn.Linear(2,3)
      layer.qk_head_dim=layer_dict["qk_head_dim"]
      layer.v_head_dim=layer_dict["v_head_dim"]
      layer.packed_head_size=layer_dict["packed_head_size"]
      layer.scale_head_size=layer_dict["scale_head_size"]
      layer.kv_cache_quant_spec=  parse_quantize_spec(layer_dict["kv_cache_quant_spec"])
      decode_attention_fwd_grouped(
          q,
          k_buffer,
          v_buffer,
          o,
          kv_indptr,
          kv_indices,
          attn_logits,
          attn_lse,
          num_kv_splits,
          max_kv_splits,
          sm_scale,
          logit_cap=logit_cap,
          sinks=sinks,
          xai_temperature_len=xai_temperature_len,
          layer=layer,
      )

