import torch
import hpc

num_tokens = 1024
num_group, n, k = 8, 4096, 4096
x = torch.randn((num_tokens, k), dtype=torch.float, device="cuda").to(torch.float8_e4m3fn)
w = torch.randn((num_group, n, k), dtype=torch.float, device="cuda").to(torch.float8_e4m3fn)
scale = torch.full((num_group,), 1.0, dtype=torch.float, device="cuda")

num_tokens_per_group = torch.full((num_group,), 8, dtype=torch.int32, device="cuda")
cu_num_tokens_per_group = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), num_tokens_per_group]), dim=0).to(torch.int32)

output = hpc.group_gemm_pertensor_fp8(
    x, w, num_tokens_per_group, cu_num_tokens_per_group, scale,
)
