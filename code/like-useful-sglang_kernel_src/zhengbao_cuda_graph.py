import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDecoderLayer(nn.Module):
    def __init__(self, dim=256, n_heads=32):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def attention(self, x):
        B, D = x.shape
        qkv = self.qkv(x)  # [B, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, self.n_heads, self.head_dim)
        k = k.view(B, self.n_heads, self.head_dim)
        v = v.view(B, self.n_heads, self.head_dim)

        att = torch.einsum("bhd,bhd->bh", q, k) / (self.head_dim ** 0.5)
        att = F.softmax(att, dim=-1)  # [B, heads]

        out = (att.unsqueeze(-1) * v).reshape(B, -1)
        return self.out_proj(out)

    def forward(self, x):
        h = x + self.attention(self.norm1(x))
        h = h + self.fc2(F.gelu(self.fc1(self.norm2(h))))
        return h

def decode_step_eager(model, x):
    return model(x)

import time

# dim = 256 这样隐藏层大小小一点，显存不是瓶颈，更容易看出性能提升
# 如果dim = 4096，显存是瓶颈，使用CUDA Graph对性能提升不明显，可能只有5%左右的提升
model = SimpleDecoderLayer(dim=256, n_heads=8).cuda().eval()
B = 1 # 为了模拟 LLM decode 阶段“一次生成 1 个 token”的输入模式
x = torch.randn(B, 256, device="cuda")

# Warmup
for _ in range(10):
    model(x)

torch.cuda.synchronize()
t0 = time.time()

for _ in range(500):
    y = decode_step_eager(model, x)

torch.cuda.synchronize()
t1 = time.time()

print("Eager decode avg time:", (t1 - t0) / 500 * 1000, "ms")


# 1. static buffers
static_x = torch.empty_like(x)
static_y = torch.empty_like(x)

g = torch.cuda.CUDAGraph()

# 2. Warmup
for _ in range(3):
    model(x)

# 3. Capture
static_x.copy_(x)
with torch.cuda.graph(g):
    static_y.copy_(model(static_x))

# 4. Replay benchmark
torch.cuda.synchronize()
t0 = time.time()

for _ in range(500):
    static_x.copy_(x) # 输入更新，但地址不变
    g.replay()  # 整个 decode step 以 Graph 方式重放

torch.cuda.synchronize()
t1 = time.time()

print("CUDA Graph decode avg time:", (t1 - t0) / 500 * 1000, "ms")
