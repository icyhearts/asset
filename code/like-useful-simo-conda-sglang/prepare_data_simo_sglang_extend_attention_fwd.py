if True:
    # like_debug {
    import os
    import time
    import json
    if True:
      from safetensors.torch import save_file
      data_dict = {
          "q_extend":q_extend.contiguous(),
          "k_extend":k_extend.contiguous(),
          "v_extend":v_extend.contiguous(),
          "o_extend":o_extend.contiguous(),
          "k_buffer":k_buffer.contiguous(),
          "v_buffer":v_buffer.contiguous(),
          "qo_indptr":qo_indptr.contiguous(),
          "kv_indptr":kv_indptr.contiguous(),
          "kv_indices":kv_indices.contiguous(),
          }
      non_tensor_args = {
          "custom_mask":custom_mask,
          "is_causal":is_causal,
          "max_len_extend":max_len_extend,
          "sm_scale":sm_scale,
          "logit_cap":logit_cap,
          "skip_prefix_custom_mask":skip_prefix_custom_mask,
          "sliding_window_size":sliding_window_size,
          "sinks":sinks,
          "window_kv_offsets":window_kv_offsets,
          "xai_temperature_len": xai_temperature_len,
          "mask_indptr":mask_indptr,
          }
      layer_dict = {
          "qk_head_dim": layer.qk_head_dim,
          "v_head_dim": layer.v_head_dim,
          "packed_head_size": layer.packed_head_size,
          "scale_head_size": layer.scale_head_size,
          "kv_cache_quant_spec": layer.kv_cache_quant_spec.to_dict(),
          }
      time_prefix = time.time()
      save_dir = "temp/prepare_data_simo_sglang_extend_attention_fwd/"
      os.makedirs(save_dir, exist_ok=True)
      safe_tensor_path = f"{save_dir}/simo_sglang_extend_attention_fwd.{time_prefix}.safetensors"
      save_file(data_dict, safe_tensor_path)

      args_json_path = f"{save_dir}/non_tensor_args.{time_prefix}.json"
      with open(args_json_path, 'w') as fp:
        json.dump(non_tensor_args, fp)

      layer_dict_json_path = f"{save_dir}/layer_dict.{time_prefix}.json"
      with open(layer_dict_json_path, 'w') as fp:
        json.dump(layer_dict, fp)

    # like_debug }
