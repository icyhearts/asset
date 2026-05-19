
import torch
import triton
from safetensors.torch import load_file
from sglang.srt.layers.attention.triton_ops.decode_attention import _fwd_grouped_kernel_stage1
input_file_path = "temp/decode_forward_stage1_triton_input.safetensors"
data_dict = load_file(input_file_path, device="cuda:0")
for key, tensor in data_dict.items():
    MiB = tensor.numel() * tensor.element_size()/1024**2
    KiB = tensor.numel() * tensor.element_size()/1024
    print(f"Tensor '{key}' is on device: {tensor.device}, dtype:{tensor.dtype}, shape:{tensor.shape}, KiB:{KiB}, MiB:{MiB}")
#####

q=data_dict["q"]
k_buffer=data_dict["k_buffer"]
v_buffer=data_dict["v_buffer"]
kv_indptr=data_dict["kv_indptr"]
kv_indices=data_dict["kv_indices"]
att_out=data_dict["att_out"]
att_lse=data_dict["att_lse"]
num_kv_splits=data_dict["num_kv_splits"]


##
print(f"kv_indptr:{kv_indptr}")
print(f"kv_indices:{kv_indices}")
print(f"num_kv_splits:{num_kv_splits}")
##
sm_scale=0.08838834764831845 
BLOCK_DMODEL=128
batch, head_num = q.shape[0], q.shape[1]
kv_group_num = q.shape[1] // k_buffer.shape[1]
BLOCK_DPE=0
BLOCK_DV=128
BLOCK=32
BLOCK_H=16
_MIN_BLOCK_KV=32
logit_cap=0.0
xai_temperature_len=-1
num_stages=2
Lk = k_buffer.shape[-1]
extra_kargs={}
Lv = v_buffer.shape[-1]
MAX_KV_SPLITS=8
grid = (batch, triton.cdiv(head_num, min(BLOCK_H, kv_group_num)), MAX_KV_SPLITS)
###
_fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
    )
