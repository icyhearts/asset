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

#sgl_directory = "/data/like/temp/qdqx_2026_05_09___16_46_17_safetensors-online/"
#ref_dir = "/data/like/temp/qdqx_2026_05_09___16_46_17_safetensors-vllm/"

sgl_directory = "/data/like/temp/qdqx_2026_05_11___11_45_27_safetensors-online"
ref_dir = "/data/like/temp/qdqx_2026_05_11___11_45_27_safetensors-vllm"

sgl_directory = "/data/like/temp/qdqx_2026_05_11___11_45_27_safetensors-online"
ref_dir = "/data/like/temp/qdqx_2026_05_11___11_45_27_safetensors-vllm"


sgl_directory = "/data/like/temp/sgl_safe_tensor_batch_invariant_triton"
ref_dir = "/data/like/temp/sgl_safe_tensor_batch_invariant"

full_paths = [str(f) for f in Path(sgl_directory).glob("*.safetensors")]

## all about compare all reduce
def compare_all_reduce(sgl_path, vlm_path):
  sgl_data_dict = load_file(sgl_path, device="cuda:0")
  ref_data_dict = load_file(vlm_path, device="cuda:0")
  data_dict_keys = sgl_data_dict.keys()
  for name in data_dict_keys:
    sgl_tensor = sgl_data_dict[name]
    ref_tensor = ref_data_dict[name]
    cosine_sim = cosine_similarity(sgl_tensor.reshape(-1), ref_tensor.reshape(-1))
    abs_ratio = sgl_tensor.abs().max() / ref_tensor.abs().max()
    l2_norm = relative_l2_error(sgl_tensor, ref_tensor)
    byte_diff_max = (sgl_tensor.view(torch.uint8) - ref_tensor.view(torch.uint8)).max().item()
    print(f"compare all reduce,  sgl:{sgl_path}, vlm:{vlm_path} data_dict_key:{name}, cosine_sim:{cosine_sim}, abs_ratio:{abs_ratio}, l2_norm:{l2_norm}, sgl_tensor.max:{sgl_tensor.abs().max()}, ref_tensor.max:{ref_tensor.abs().max()}, byte_diff:{byte_diff_max}")

# 1: this is custom all reduce
#sgl_path="/data/like/temp/qdqx_2026_05_09___16_46_17_safetensors-online---compare-all-reduce/validate_all_reduce_rank-0.sglang.safetensors"
#vlm_path="/data/like/temp/qdqx_2026_05_09___16_46_17_safetensors-online---compare-all-reduce/validate_all_reduce_rank-0.vllm.safetensors"
# 2: this is all torch.distributed.all_reduce inside framework
#for my_tp_rank in range(0,8):
#  sgl_path=f"/data/like/temp/qdqx_2026_05_09___16_46_17_safetensors-online-compare-all-reduce-out/validate_all_reduce_rank-{my_tp_rank}.sglang.safetensors"
#  vlm_path=f"/data/like/temp/qdqx_2026_05_09___16_46_17_safetensors-online-compare-all-reduce-out/validate_all_reduce_rank-{my_tp_rank}.vllm.safetensors"
#  # compare_all_reduce(sgl_path, vlm_path)

# 3: 独立脚本, torchrun like-useful/torch_all_reduce.py
#sgl_path="/data/like/temp/torch.distributed.all_reduce.sglang.safetensors"
#vlm_path="/data/like/temp/torch.distributed.all_reduce.vllm.safetensors"
# 再比 vllm torch.distributed.all_reduce vs vs torchrun
#sgl_path="/data/like/temp/qdqx_2026_05_09___16_46_17_safetensors-online-compare-all-reduce-out/validate_all_reduce_rank-0.vllm.safetensors"
#vlm_path="/data/like/temp/torch.distributed.all_reduce.vllm.safetensors"

## end of all about compare all reduce

def sort_safetensor_files(file_list):
    """
    对符合命名规范的文件名列表进行排序。
    排序优先级（升序）：forwardcount > rank > layers
    rank-0-prefix-model-forwardcount-20-0.safetensors
    """
    pattern = re.compile(
        r'.*/rank_(\d+)_prefix_model\.layers\.(\d+)\..*?_forwardcount_(\d+)\.safetensors'
    )

    def sort_key(filename):
      splits = os.path.basename(filename).replace(".safetensors", "").split("-")
      fwd = int(splits[5])
      module_name_list = splits[3].split(".")[3:]
      module_name_merge = ".".join(module_name_list)
      inner_stage = int(splits[6]) # inner stage: inside attentoin.forward, RoPe+attention is in two stage, so save two file
      rk = int(splits[1])
      if len(splits[3].split(".")) >= 3:
        ly = int(splits[3].split(".")[2])
      else:
        ly = 0
      return  rk, fwd, ly, module_name_merge, inner_stage
      #return fwd, ly, rk, inner_stage

    return sorted(file_list, key=sort_key)
sorted_full_paths = sort_safetensor_files(full_paths)
####

for sgl_path in sorted_full_paths:
  sgl_safe_tensor_path = sgl_path
  base_name = os.path.basename(sgl_path)
  ref_safe_tensor_path = os.path.join(ref_dir, base_name)
  if not os.path.exists(ref_safe_tensor_path):
    continue
  
  sgl_data_dict = load_file(sgl_safe_tensor_path, device="cuda:0")
  ref_data_dict = load_file(ref_safe_tensor_path, device="cuda:0")
  ###
  #data_dict_keys = ["input_2d", "qdq_x"]
  data_dict_keys = sgl_data_dict.keys()
  for name in data_dict_keys:
    sgl_tensor = sgl_data_dict[name]
    ref_tensor = ref_data_dict[name]
    if "contiguous" in name and sgl_tensor.dtype == torch.bool and sgl_tensor.numel() == 1:
      print(f"name:{base_name}, data_dict_key:{name}, sgl_tensor:{sgl_tensor.item()}, ref_tensor{ref_tensor.item()}")
      continue
    cosine_sim = cosine_similarity(sgl_tensor.reshape(-1), ref_tensor.reshape(-1))
    abs_ratio = sgl_tensor.abs().max() / ref_tensor.abs().max()
    l2_norm = relative_l2_error(sgl_tensor, ref_tensor)
    dtype1 = sgl_tensor.dtype
    dtype2 = ref_tensor.dtype

    shape1 = sgl_tensor.shape
    shape2 = ref_tensor.shape

    print(f"name:{base_name}, data_dict_key:{name}, dtype1:{dtype1}, dtype2:{dtype2}, shape1:{shape1}, shape2:{shape2}")
    if (dtype1 == torch.int64 and dtype2 == torch.int32) or  (dtype2 == torch.int64 and dtype1 == torch.int32):
      ref_tensor = ref_tensor.to(dtype1)
    byte_diff_max = (sgl_tensor.view(torch.uint8) - ref_tensor.view(torch.uint8)).max().item()
    print(f"name:{base_name}, data_dict_key:{name}, cosine_sim:{cosine_sim}, abs_ratio:{abs_ratio}, l2_norm:{l2_norm}, sgl_tensor.max:{sgl_tensor.abs().max()}, ref_tensor.max:{ref_tensor.abs().max()}, byte_diff:{byte_diff_max}")
