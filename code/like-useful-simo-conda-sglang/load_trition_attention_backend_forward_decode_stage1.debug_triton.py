import json
import torch
import triton
from safetensors.torch import load_file
from simo.extensions.sglang_simo.layers.attention.triton_ops.decode_attention  import _fwd_grouped_kernel_stage1

save_dir = "temp/debug_triton/"
#time_prefix="1775815542.3402684"
#time_prefix="1775817432.295112" # work
time_prefix="1775817431.3377056" # checking
time_prefix="1775817430.4585114"
#time_prefix="1775818880.3736513"
time_prefix="1776075883.886032"
print(f"time_prefix:{time_prefix}")

###
safe_tensor_path = f"{save_dir}/decode_forward_stage1_triton_input.{time_prefix}.safetensors"
data_dict = load_file(safe_tensor_path, device="cuda:0")

args_json_path = f"{save_dir}/non_tensor_args.{time_prefix}.json"
with open(args_json_path) as fp:
  non_tensor_args = json.load( fp)

extra_args_json_path = f"{save_dir}/extra_args.{time_prefix}.json"
with open(extra_args_json_path) as fp:
  extra_kargs = json.load(fp)
###

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
sm_scale=non_tensor_args.get('sm_scale')
BLOCK_DMODEL=non_tensor_args.get('BLOCK_DMODEL')
batch, head_num = q.shape[0], q.shape[1]
kv_group_num = non_tensor_args.get('kv_group_num')
BLOCK_DPE=non_tensor_args.get('BLOCK_DPE')
BLOCK_DV=non_tensor_args.get('BLOCK_DV')
BLOCK=non_tensor_args.get('BLOCK')
BLOCK_H=non_tensor_args.get('BLOCK_H')
_MIN_BLOCK_KV=non_tensor_args.get('_MIN_BLOCK_KV')
logit_cap=non_tensor_args.get('logit_cap')
xai_temperature_len=non_tensor_args.get('xai_temperature_len')
num_stages=non_tensor_args.get('num_stages')
Lk =non_tensor_args.get('Lk')
Lv =non_tensor_args.get('Lv')
MAX_KV_SPLITS=8


###
grid = (batch, triton.cdiv(head_num, min(BLOCK_H, kv_group_num)), MAX_KV_SPLITS)
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
###
o = torch.empty_like(q)
attn_logits = att_out
attn_lse = att_lse
max_kv_splits = MAX_KV_SPLITS
sinks=None
from sglang.srt.layers.attention.triton_ops.decode_attention import  _decode_softmax_reducev_fwd
_decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_buffer,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        sinks,
    )

