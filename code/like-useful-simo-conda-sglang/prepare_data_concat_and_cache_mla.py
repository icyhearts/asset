#if True:
#    import os
#    import time
#    import json
#    if True:
#      from safetensors.torch import save_file
#      data_dict = {
#          "q":q.contiguous(),
#          "k_buffer": k_buffer.contiguous(),
#          "v_buffer": v_buffer.contiguous(),
#          "o": o.contiguous(),
#          "kv_indptr": kv_indptr.contiguous(),
#          "kv_indices": kv_indices.contiguous(),
#          "attn_logits": attn_logits.contiguous(),
#          "attn_lse": attn_lse.contiguous(),
#          "num_kv_splits": num_kv_splits,
#          }
#      non_tensor_args = {
#          "max_kv_splits": max_kv_splits,
#          "sm_scale": sm_scale,
#          "logit_cap": logit_cap,
#          "sinks": sinks,
#          "xai_temperature_len": xai_temperature_len,
#          "stride_info": {
#            "q":q.stride(),
#            "k_buffer": k_buffer.stride(),
#            "v_buffer": v_buffer.stride(),
#            "o": o.stride(),
#            "kv_indptr": kv_indptr.stride(),
#            "kv_indices": kv_indices.stride(),
#            "attn_logits": attn_logits.stride(),
#            "attn_lse": attn_lse.stride(),
#            "num_kv_splits": num_kv_splits.stride(),
#            },
#          "is_contiguous_info": {
#            "q":q.is_contiguous(),
#            "k_buffer": k_buffer.is_contiguous(),
#            "v_buffer": v_buffer.is_contiguous(),
#            "o": o.is_contiguous(),
#            "kv_indptr": kv_indptr.is_contiguous(),
#            "kv_indices": kv_indices.is_contiguous(),
#            "attn_logits": attn_logits.is_contiguous(),
#            "attn_lse": attn_lse.is_contiguous(),
#            "num_kv_splits": num_kv_splits.is_contiguous(),
#            }
#          }
#      layer_dict = {
#          "qk_head_dim": layer.qk_head_dim,
#          "v_head_dim": layer.v_head_dim,
#          "packed_head_size": layer.packed_head_size,
#          "scale_head_size": layer.scale_head_size,
#          "kv_cache_quant_spec": layer.kv_cache_quant_spec.to_dict(),
#          }
#      time_prefix = time.time()
#      save_dir = "temp/debug_triton_decode_attention_fwd_grouped/"
#      os.makedirs(save_dir, exist_ok=True)
#      safe_tensor_path = f"{save_dir}/decode_attention_fwd_grouped.{time_prefix}.safetensors"
#      save_file(data_dict, safe_tensor_path)
#
#      args_json_path = f"{save_dir}/non_tensor_args.{time_prefix}.json"
#      with open(args_json_path, 'w') as fp:
#        json.dump(non_tensor_args, fp)
#
#      layer_dict_json_path = f"{save_dir}/layer_dict.{time_prefix}.json"
#      with open(layer_dict_json_path, 'w') as fp:
#        json.dump(layer_dict, fp)

##
if True:
    import os
    import time
    import json
    if True:
      from safetensors.torch import save_file
      data_dict = {
        "kv_c": kv_c.contiguous(),
        "k_pe": k_pe.contiguous(),
        "kv_cache": kv_cache.contiguous(),
        "slot_mapping": slot_mapping.contiguous(),
          }
      layer_dict = kv_cache_quant_spec.to_dict()
      time_prefix = time.time()
      save_dir = "temp/debug_kv_cache__concat_and_cache_mla/"
      os.makedirs(save_dir, exist_ok=True)
      safe_tensor_path = f"{save_dir}/concat_and_cache_mla.{time_prefix}.safetensors"
      save_file(data_dict, safe_tensor_path)

      args_json_path = f"{save_dir}/kv_cache_quant_spec.{time_prefix}.json"
      with open(args_json_path, 'w') as fp:
        json.dump(layer_dict, fp)
