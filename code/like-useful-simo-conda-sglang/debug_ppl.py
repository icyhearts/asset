
# 在两个环境中分别运行，对比结果
import torch
from simo.ops.mx_api import downcast_to_mxfmt, upcast_from_mxfmt
from simo.quantization.dtypes import as_dtype

# 固定输入
torch.manual_seed(42)
x = torch.randn(128, 1024, dtype=torch.bfloat16, device='cuda')

# mxfp4 downcast + upcast
q, s = downcast_to_mxfmt(x, as_dtype('mxfp4_e2m1'), axis=-1, block_size=32)
dq = upcast_from_mxfmt(q, s, as_dtype('mxfp4_e2m1'), torch.bfloat16, axis=-1)

print(f"q checksum: {q.sum().item()}")
print(f"s checksum: {s.float().sum().item()}")
print(f"dq checksum: {dq.sum().item()}")
print(f"dq abs mean: {dq.abs().mean().item()}")
print(f"error abs mean: {(x.float() - dq.float()).abs().mean().item()}")
