if True:
    # like_debug {
    import os
    import time
    import json
    if True:
      from safetensors.torch import save_file
      data_dict = {
              "kv_buffer":kv_buffer.contiguous(),
              "loc":loc.contiguous(),
              "cache_k_nope":cache_k_nope.contiguous(),
              "cache_k_rope":cache_k_rope.contiguous(),
          }
      time_prefix = time.time()
      save_dir = "temp/prepare_data_sgl_src_set_mla_kv_buffer_triton/"
      os.makedirs(save_dir, exist_ok=True)
      safe_tensor_path = f"{save_dir}/prepare_data_sgl_src_set_mla_kv_buffer_triton.{time_prefix}.safetensors"
      save_file(data_dict, safe_tensor_path)


    # like_debug }
