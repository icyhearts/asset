# w_kc 初始化分析

## 问题背景

在测试 DeepSeekV2 模型使用 SIMO int4 量化时��遇到错误：
```
RuntimeError: expected scalar type BFloat16 but Found Int
```

错误发生在 `forward_absorb_prepare` 中的 `torch.bmm(q_nope.transpose(0, 1), self.w_kc)` 调用，因为 `w_kc` 是 int 类型而不是 bfloat16。

## w_kc 初始化流程

在 DeepSeekV2 模型的 `post_load_weights` 方法中（`deepseek_v2.py` 第 3595-3644 行），`w_kc` 的初始化流程如下：

```python
# 1. 获取 kv_b_proj 的权重
w = self_attn.kv_b_proj.weight

# 2. 根据量化类型处理
if 有 weight_scale (block-wise):
    if use_deep_gemm_bmm:
        block_scale = weight_scale
    else:
        w = block_quant_dequant(weight, weight_scale, weight_block_size, torch.bfloat16)
else:
    w, scale = channel_quant_to_tensor_quant(weight, weight_scale)

# 3. int8 特殊处理
if w.dtype == torch.int8:
    if weight_block_size is not None:  # block-wise int8
        w = int8_block_dequant(weight, weight_scale, weight_block_size).to(torch.bfloat16)
    else:  # channel-wise int8
        w = w.to(torch.bfloat16) * self_attn.kv_b_proj.weight_scale

# 4. 分割出 w_kc 和 w_vc
w_kc, w_vc = w.unflatten(0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)).split(
    [self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1
)

# 5. 赋值
self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
```

## int4 量化的问题

关键在于第 3 步的检查条件：`if w.dtype == torch.int8:`

| 量化类型 | 打包后 dtype | 代码检查条件 | 结果 |
|----------|-------------|-------------|------|
| int8 | `torch.int8` | `w.dtype == torch.int8` ✅ | 被解量化 |
| int4 | `torch.int32` | `w.dtype == torch.int8` ❌ | 保持 int32 类型 |

**验证 int4 打包后的 dtype**：
```python
import torch
from simo.extensions.sglang_simo.quantization.quantization import get_downcast_kernel, parse_quantize_spec

weight_spec = parse_quantize_spec({'dtype': 'int4', 'axis': -1, 'group_size': 32})
downcast_kernel = get_downcast_kernel(weight_spec, 0)

test_weight = torch.randn(128, 256, dtype=torch.bfloat16)
packed_weight, scale = downcast_kernel(test_weight)

print(f'Original dtype: {test_weight.dtype}')  # torch.bfloat16
print(f'Packed dtype: {packed_weight.dtype}')  # torch.int32
print(f'Packed shape: {packed_weight.shape}')    # torch.Size([128, 32]) - 形状减半
```

**输出**：
```
Original dtype: torch.bfloat16
Packed dtype: torch.int32
Packed shape: torch.Size([128, 32])
```

## 结论

int4 量化后，权重被打包存储为 `torch.int32` 类型（2个int4值打包进1个int32），而不是 `torch.int8`。DeepSeekV2 的 `post_load_weights` 方法只检查 `w.dtype == torch.int8`，不会处理 int32 类型的权重，导致：

1. `w_kc` 保持为 int32 类型
2. 后续的 `torch.bmm(q_nope, self.w_kc)` 期望 bfloat16 类型
3. 报错：`RuntimeError: expected scalar type BFloat16 but Found Int`

## 解决方案

### 方案 1：排除 kv_b_proj（临时方案）

在量化配置文件中排除 `kv_b_proj`：
```json
{
    "excludes": [
        "lm_head",
        "re:.*kv_b_proj"
    ]
}
```

这样 `kv_b_proj` 不会被量化，`w_kc` 保持为 bfloat16 类型。

### 方案 2：修改 sglang 代码（根本方案）

在 DeepSeekV2 的 `post_load_weights` 中添加 int4 解量化支持：
```python
if w.dtype == torch.int32:  # int4 packed
    # 需要实现 int4 解包和解量化
    w = int4_unpack_and_dequant(w, weight_scale, group_size).to(torch.bfloat16)
```

这需要修改 sglang 源码 `deepseek_v2.py` 中的 `post_load_weights` 方法。

## 相关代码位置

- DeepSeekV2 `post_load_weights`: `/softhome/like/package/h100/package/sglang_kernel_src/python/sglang/srt/models/deepseek_v2.py:3595-3696`
- `w_kc` 使用位置: `deepseek_v2.py:1893` (`q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)`)
