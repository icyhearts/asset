

import torch
import triton
from safetensors.torch import load_file
from sglang.srt.layers.attention.triton_ops.decode_attention import _fwd_grouped_kernel_stage1
time_prefix=1776592047.5375593
save_dir = "temp/prepare_data_sgl_src_set_mla_kv_buffer_triton/"
input_file_path = f"{save_dir}/prepare_data_sgl_src_set_mla_kv_buffer_triton.{time_prefix}.safetensors"
data_dict = load_file(input_file_path, device="cuda:0")
for key, tensor in data_dict.items():
    MiB = tensor.numel() * tensor.element_size()/1024**2
    KiB = tensor.numel() * tensor.element_size()/1024
    print(f"Tensor '{key}' is on device: {tensor.device}, dtype:{tensor.dtype}, shape:{tensor.shape}, KiB:{KiB}, MiB:{MiB}")
#####

kv_buffer=data_dict["kv_buffer"]
loc=data_dict["loc"]
cache_k_nope=data_dict["cache_k_nope"]
cache_k_rope=data_dict["cache_k_rope"]
from sglang.srt.mem_cache.utils import set_mla_kv_buffer_triton
set_mla_kv_buffer_triton(
    kv_buffer,
    loc,
    cache_k_nope,
    cache_k_rope)
