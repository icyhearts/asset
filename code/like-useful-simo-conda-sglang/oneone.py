
from safetensors.torch import load_file
import os
import re
from safetensors.torch import load_file
from pathlib import Path
import torch
import torch.nn.functional as F


def cosine_similarity(a, b, dim=0):
    return F.cosine_similarity(a.float(), b.float(), dim=dim)

def relative_l2_error(x: torch.Tensor, x_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # 转为 float32 以保证计算精度
    x_f32 = x.float()
    x_hat_f32 = x_hat.float()

    # torch.norm 会计算整个张量的 L2 范数
    diff_norm = torch.norm(x_f32 - x_hat_f32)
    orig_norm = torch.norm(x_f32)

    return diff_norm / (orig_norm + eps)

def describe(tensor1, tensor2, name1, name2):
  cos_sim = cosine_similarity(tensor1.reshape(-1).float(), tensor2.reshape(-1).float())
  l2_err = relative_l2_error(tensor1.reshape(-1).float(), tensor2.reshape(-1).float())
  byte_diff = (tensor1.view(torch.uint8) - tensor2.view(torch.uint8)).max()
  print(f"{name1} vs {name2}, cosine_similarity:{cos_sim}, relative_l2_error:{l2_err}, byte_diff:{byte_diff}")

sgl_directory = "/data/like/temp/sgl_safe_tensor_batch_invariant_triton/"
ref_dir = "/data/like/temp/sgl_safe_tensor_batch_invariant/"

row_parallel_quant_method_out__safetensor = 'rank-0-prefix-model.layers.0.self_attn.o_proj-forwardcount-0-1.safetensors'
row_parallel_quant_method_out__safetensor_sgl = load_file(sgl_directory + row_parallel_quant_method_out__safetensor, device="cuda:0")['row_parallel_quant_method_out']
row_parallel_quant_method_out__safetensor_ref = load_file(ref_dir + row_parallel_quant_method_out__safetensor, device="cuda:0")['row_parallel_quant_method_out']


act_out_safe_tensor_name = "rank-0-prefix-model.layers.0.mlp-forwardcount-0-0.safetensors"

act_out_tensor = load_file(sgl_directory + act_out_safe_tensor_name, device="cuda:0")['in_v2mlp_2_act_fn_out']

W_dict = load_file("/data/like/hf-models/DeepSeek-V2-Lite-Chat-16B_A2.4B/model-00001-of-000004.safetensors", device="cpu")
for k in W_dict.keys():
  if "layers.0.mlp.down_proj" in k:
    print(f"found k:{k}")
    layers_0_mlp_down_proj_w = W_dict.get(k).to(act_out_tensor.device)
    print(layers_0_mlp_down_proj_w.shape)

from sglang.srt.batch_invariant_ops.batch_invariant_ops import matmul_persistent

torch_out = torch.mm(act_out_tensor, layers_0_mlp_down_proj_w.T)
sgl_out = matmul_persistent(act_out_tensor, layers_0_mlp_down_proj_w.T)

byte_diff = (torch_out.view(torch.uint8) - sgl_out.view(torch.uint8)).max()




torch_out_f32 = torch.mm(act_out_tensor.float(), layers_0_mlp_down_proj_w.T.float())
sgl_out_f32 = matmul_persistent(act_out_tensor.float(), layers_0_mlp_down_proj_w.T.float())

