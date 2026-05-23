import torch
import json
import triton
from safetensors.torch import load_file
from sglang.srt.layers.attention.triton_ops.decode_attention import _fwd_grouped_kernel_stage1
#time_prefix="1776764594.4215453"  # llama
time_prefix="1779268229.8098829"  # dsv2 lite

save_dir = "../sglang_kernel_src/temp/prepare_data_sgl_decode_attention_fwd/"

safe_tensor_path = f"{save_dir}/prepare_data_sgl_decode_attention_fwd.{time_prefix}.safetensors"
data_dict = load_file(safe_tensor_path, device="cuda:0")
for key, tensor in data_dict.items():
    MiB = tensor.numel() * tensor.element_size()/1024**2
    KiB = tensor.numel() * tensor.element_size()/1024
    print(f"Tensor '{key}' is on device: {tensor.device}, dtype:{tensor.dtype}, shape:{tensor.shape}, KiB:{KiB}, MiB:{MiB}")

##
q=data_dict["q"]
k_buffer=data_dict["k_buffer"]
v_buffer=data_dict["v_buffer"]
o=data_dict["o"]
kv_indptr=data_dict["kv_indptr"]
kv_indices=data_dict["kv_indices"]
attn_logits=data_dict["attn_logits"]
attn_lse=data_dict["attn_lse"]
num_kv_splits=data_dict["num_kv_splits"]

#####
args_json_path = f"{save_dir}/non_tensor_args.{time_prefix}.json"
with open(args_json_path, 'r') as fp:
  non_tensor_args = json.load(fp)

max_kv_splits=non_tensor_args["max_kv_splits"]
sm_scale=non_tensor_args["sm_scale"]
logit_cap=non_tensor_args["logit_cap"]
sinks=non_tensor_args["sinks"]
xai_temperature_len=non_tensor_args["xai_temperature_len"]
###
from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd
decode_attention_fwd(
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
    xai_temperature_len=xai_temperature_len
)
