
if True:
    import os
    import time
    import json
    if True:
      from safetensors.torch import save_file
      data_dict = {
          "q":q.contiguous(),
          "k_buffer":k_buffer.contiguous(),
          "v_buffer":v_buffer.contiguous(),
          "o":o.contiguous(),
          "kv_indptr":kv_indptr.contiguous(),
          "kv_indices":kv_indices.contiguous(),
          "attn_logits":attn_logits.contiguous(),
          "attn_lse":attn_lse.contiguous(),
          "num_kv_splits":num_kv_splits.contiguous(),
          }
      non_tensor_args = {
          "max_kv_splits":max_kv_splits,
              "sm_scale":sm_scale,
              "logit_cap":logit_cap,
              "sinks":sinks,
              "xai_temperature_len":xai_temperature_len,
          }
      time_prefix = time.time()
      save_dir = "../sglang_kernel_src/temp/prepare_data_sgl_decode_attention_fwd/"
      os.makedirs(save_dir, exist_ok=True)
      safe_tensor_path = f"{save_dir}/prepare_data_sgl_decode_attention_fwd.{time_prefix}.safetensors"
      save_file(data_dict, safe_tensor_path)

      args_json_path = f"{save_dir}/non_tensor_args.{time_prefix}.json"
      with open(args_json_path, 'w') as fp:
        json.dump(non_tensor_args, fp)

