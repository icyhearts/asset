
import torch
import triton
from safetensors.torch import load_file
from sglang.srt.layers.attention.triton_ops.extend_attention import _fwd_kernel



input_file_path = "temp/extend_forward_triton_input.safetensors"

# Load the file directly to the first available CUDA device (cuda:0)
# Use "cuda:N" for a specific GPU device N (e.g., "cuda:1")
data_dict = load_file(input_file_path, device="cuda:0")

# Verify the device of the loaded data_dict
for key, tensor in data_dict.items():
    MiB = tensor.numel() * tensor.element_size()/1024**2
    KiB = tensor.numel() * tensor.element_size()/1024
    print(f"Tensor '{key}' is on device: {tensor.device}, dtype:{tensor.dtype}, shape:{tensor.shape}, KiB:{KiB}, MiB:{MiB}")


q_extend=data_dict["q_extend"]
k_extend=data_dict["k_extend"]
v_extend=data_dict["v_extend"]
o_extend=data_dict["o_extend"]
k_buffer=data_dict["k_buffer"]
v_buffer=data_dict["v_buffer"]
qo_indptr=data_dict["qo_indptr"]
kv_indptr=data_dict["kv_indptr"]
kv_indices=data_dict["kv_indices"]

print(f"kv_indices:{kv_indices}")
print(f"kv_indptr:{kv_indptr}")
print(f"qo_indptr:{qo_indptr}")

custom_mask=None
mask_indptr=None
sinks=None
window_kv_offsets=None
Lq, Lk, Lv = ( q_extend.shape[-1], k_extend.shape[-1], v_extend.shape[-1],)
sm_scale = 1.0 / (Lq**0.5)
kv_group_num = q_extend.shape[1] // k_extend.shape[1]
sliding_window_size=-1
logit_cap=0.0
xai_temperature_len=-1
BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = (128, 0, 128, 128, 64, 8)
USE_CUSTOM_MASK = custom_mask is not None #custom_mask=none
is_causal=True
SKIP_PREFIX_CUSTOM_MASK = True
HAS_SINK = sinks is not None #sinks=None
_is_hip=False
num_stages = 1
extra_kargs = {}
max_len_extend=2



batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))




_fwd_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        sinks,
        window_kv_offsets,
        sm_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        SLIDING_WINDOW_SIZE=sliding_window_size,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        Lq=Lq,
        Lv=Lv,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        IS_CAUSAL=is_causal,
        SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,
        HAS_SINK=HAS_SINK,
        STORE_TRANSPOSE=_is_hip,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kargs,
    )
