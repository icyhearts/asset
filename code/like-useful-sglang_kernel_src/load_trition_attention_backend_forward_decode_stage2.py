

import torch
import triton
from safetensors.torch import load_file
from sglang.srt.layers.attention.triton_ops.decode_attention import  _fwd_kernel_stage2
input_file_path = "temp/decode_forward_stage2_triton_input.safetensors"
data_dict = load_file(input_file_path, device="cuda:0")
for key, tensor in data_dict.items():
    MiB = tensor.numel() * tensor.element_size()/1024**2
    KiB = tensor.numel() * tensor.element_size()/1024
    print(f"Tensor '{key}' is on device: {tensor.device}, dtype:{tensor.dtype}, shape:{tensor.shape}, KiB:{KiB}, MiB:{MiB}")
##
logits=data_dict["logits"]
lse=data_dict["lse"]
o=data_dict["o"]
kv_indptr=data_dict["kv_indptr"]
num_kv_splits=data_dict["num_kv_splits"]

extra_kargs={}
HAS_SINK=False
batch, head_num = (1, 32) 
Lv = 128
grid = (batch, head_num)
BLOCK_DV=128
_MIN_BLOCK_KV=32
sinks=None
MAX_KV_SPLITS=8
_fwd_kernel_stage2[grid](
        logits,
        lse,
        o,
        kv_indptr,
        num_kv_splits,
        sinks,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        HAS_SINK=HAS_SINK,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )
