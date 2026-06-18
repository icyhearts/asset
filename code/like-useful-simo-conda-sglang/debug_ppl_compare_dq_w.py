import os
from pathlib import Path

# 方法一：使用 os.listdir
# 
sgl_directory = "/data/like/temp/2026_04_30___17_00_14-online-quant/sgl/"
ref_dir = "/data/like/temp/2026_04_30___17_00_14-xuhaifeng/sgl/"
#  


full_paths = [str(f) for f in Path(sgl_directory).glob("*.safetensors")]
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

for sgl_path in full_paths:
  sgl_safe_tensor_path = sgl_path
  base_name = os.path.basename(sgl_path)
  
  sgl_data_dict = load_file(sgl_safe_tensor_path, device="cuda:0")
  sgl_tensor = sgl_data_dict["dq_w"]

  ref_safe_tensor_path = os.path.join(ref_dir, base_name)
  ref_data_dict = load_file(ref_safe_tensor_path, device="cuda:0")
  ref_tensor = ref_data_dict["dq_w"]
  cosine_sim = cosine_similarity(sgl_tensor.reshape(-1), ref_tensor.reshape(-1))
  abs_ratio = sgl_tensor.abs().max() / ref_tensor.abs().max()
  l2_norm = relative_l2_error(sgl_tensor, ref_tensor)
  print(f"name:{base_name}, cosine_sim:{cosine_sim}, abs_ratio:{abs_ratio}, l2_norm:{l2_norm}, sgl_tensor.max:{sgl_tensor.abs().max()}, ref_tensor.max:{ref_tensor.abs().max()}")
