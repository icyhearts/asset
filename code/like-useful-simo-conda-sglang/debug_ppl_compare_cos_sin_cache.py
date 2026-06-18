from pathlib import Path

import os
# 方法一：使用 os.listdir
# 
#sgl_directory = "/data/like/temp/qdqx_2026_05_05___18_51_05_safetensors-online/"
#ref_dir = "/data/like/temp/qdqx_2026_05_05___18_51_05_safetensors-xuhaifeng/"
#print(full_paths)
#print(len(full_paths))

import torch
import torch.nn.functional as F

def cosine_similarity(a, b, dim=0):
    """
    计算两个张量的余弦相似度。

    参数:
        a (torch.Tensor): 第一个张量
        b (torch.Tensor): 第二个张量
        dim (int): 计算相似度的维度，默认为0

    返回:
        torch.Tensor: 余弦相似度值
    """
    # 方法一：使用 torch.nn.functional.cosine_similarity（推荐）
    return F.cosine_similarity(a.float(), b.float(), dim=dim)


def relative_l2_error(x: torch.Tensor, x_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算两个张量的相对 L2 误差：
        ||x - x_hat||_2 / (||x||_2 + eps)

    参数:
        x: 原始张量 (bf16 或其他类型)
        x_hat: 重建张量 (bf16 或其他类型)
        eps: 防止除零的小常数
    返回:
        一个标量张量 (float32)
    """
    # 转为 float32 以保证计算精度
    x_f32 = x.float()
    x_hat_f32 = x_hat.float()

    # torch.norm 会计算整个张量的 L2 范数
    diff_norm = torch.norm(x_f32 - x_hat_f32)
    orig_norm = torch.norm(x_f32)

    return diff_norm / (orig_norm + eps)

from safetensors.torch import load_file
#     /data/like/temp/cos_sin_cache-xuhaifeng.safetensors
# original fp32 cos_sin_cache
#sgl_cos_sin_safe_tensor_path = "/data/like/temp/cos_sin_cache-online.safetensors"
# force to vllm
sgl_cos_sin_safe_tensor_path = "/data/like/temp/force_vllm_cos.safetensors"
vlm_cos_sin_safe_tensor_path = "/data/like/temp/cos_sin_cache-vllm_lm_eval.safetensors"

xhf_cos_sin_safe_tensor_path = "/data/like/temp/cos_sin_cache-xuhaifeng.safetensors" 

sgl_data_dict = load_file(sgl_cos_sin_safe_tensor_path, device="cuda:0")
vlm_data_dict = load_file(vlm_cos_sin_safe_tensor_path, device="cuda:0")
xhf_data_dict = load_file(xhf_cos_sin_safe_tensor_path, device="cuda:0")

sgl_cos_sin_safe_tensor = list(sgl_data_dict.values())[0]
vlm_cos_sin_safe_tensor = list(vlm_data_dict.values())[0]
xhf_cos_sin_safe_tensor = list(xhf_data_dict.values())[0]

low_shape = vlm_cos_sin_safe_tensor.shape[0]

sgl_cos_sin_safe_tensor_slice = sgl_cos_sin_safe_tensor[:low_shape, :]

sgl_vlm_cos = cosine_similarity(sgl_cos_sin_safe_tensor_slice.reshape(-1), vlm_cos_sin_safe_tensor.reshape(-1))
sgl_vlm_l2 = relative_l2_error(sgl_cos_sin_safe_tensor_slice.reshape(-1), vlm_cos_sin_safe_tensor.reshape(-1))
bin_diff =  (sgl_cos_sin_safe_tensor_slice.float().view(torch.uint8) - vlm_cos_sin_safe_tensor.float().view(torch.uint8)).max()
print(f"sgl_vlm_cos:{sgl_vlm_cos}, sgl_vlm_l2:{sgl_vlm_l2}, byte diff max:{bin_diff}")

xhf_cos_sin_safe_tensor_slice =  xhf_cos_sin_safe_tensor[:low_shape, :]

xhf_vlm_cos = cosine_similarity(xhf_cos_sin_safe_tensor_slice.reshape(-1), vlm_cos_sin_safe_tensor.reshape(-1))
xhf_vlm_l2 = relative_l2_error(xhf_cos_sin_safe_tensor_slice.reshape(-1), vlm_cos_sin_safe_tensor.reshape(-1))
bin_diff =  (xhf_cos_sin_safe_tensor_slice.float().view(torch.uint8) - vlm_cos_sin_safe_tensor.float().view(torch.uint8)).max()
print(f"xhf_vlm_cos:{xhf_vlm_cos}, xhf_vlm_l2:{xhf_vlm_l2}, byte diff max:{bin_diff}")

