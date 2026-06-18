
import torch.nn.functional as F
from safetensors.torch import load_file
sgl_path="/data/like/temp/sgl_safe_tensor_only_input_id_2026_06_13___15_44_11/rank-0-prefix-model-forwardcount-0-0.safetensors"
vlm_path="/data/like/temp/vllm_safe_tensor_only_input_id_2026_06_13___15_44_11/rank-0-prefix-model-forwardcount-0-0.safetensors"
sgl_data_dict = load_file(sgl_path, device="cuda:0")
ref_data_dict = load_file(vlm_path, device="cuda:0")


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

if True:
  data_dict_keys = sgl_data_dict.keys()
  for name in data_dict_keys:
    sgl_tensor = sgl_data_dict[name]
    ref_tensor = ref_data_dict[name]
    cosine_sim = cosine_similarity(sgl_tensor.reshape(-1), ref_tensor.reshape(-1))
    abs_ratio = sgl_tensor.abs().max() / ref_tensor.abs().max()
    l2_norm = relative_l2_error(sgl_tensor, ref_tensor)
    dtype1 = sgl_tensor.dtype
    dtype2 = ref_tensor.dtype

    shape1 = sgl_tensor.shape
    shape2 = ref_tensor.shape

    print(f" data_dict_key:{name}, dtype1:{dtype1}, dtype2:{dtype2}, shape1:{shape1}, shape2:{shape2}")
    if (dtype1 == torch.int64 and dtype2 == torch.int32) or  (dtype2 == torch.int64 and dtype1 == torch.int32):
      ref_tensor = ref_tensor.to(dtype1)
    byte_diff_max = (sgl_tensor.view(torch.uint8) - ref_tensor.view(torch.uint8)).max().item()
    print(f" data_dict_key:{name}, cosine_sim:{cosine_sim}, abs_ratio:{abs_ratio}, l2_norm:{l2_norm}, sgl_tensor.max:{sgl_tensor.abs().max()}, ref_tensor.max:{ref_tensor.abs().max()}, byte_diff:{byte_diff_max}")
