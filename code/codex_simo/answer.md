
# vLLM SIMO KV Cache 量化：reshape_and_cache 与 unified_attention 分析

## 问题 1：`reshape_and_cache_kernel_flash` 中 `_compute_and_pack_mxfmt` 对 MXINT8 返回什么？

### 调用上下文

在 `reshape_and_cache_kernel_flash` 中（`triton_reshape_and_cache_flash.py:205-220`）：

```python
packed_key, key_scale = _compute_and_pack_mxfmt(
    tl.reshape(key_tile, (1, TILE_SIZE)),   # 输入 shape: (1, TILE_SIZE)
    MX_FORMAT_ID=MX_FORMAT_ID,              # MXINT8 = 1
    ...
    MX_BLOCK_SIZE=TILE_SIZE,
)
```

输入 `src_tensor` 的 shape 为 `(1, TILE_SIZE)`，其中 `TILE_SIZE` 是量化分块大小（典型值为 32）。

### `_compute_and_pack_mxfmt` 内部逻辑（MXINT8 路径）

**关键常量**（`_downcast_to_mxfmt.py:251-253`）：
- `BLOCK_SIZE_OUT_DIM = src_tensor.shape[0] = 1`
- `BLOCK_SIZE_QUANT_DIM = src_tensor.shape[1] = TILE_SIZE`（例如 32）
- `BLOCK_SIZE_QUANT_MX_SCALE = BLOCK_SIZE_QUANT_DIM // MX_BLOCK_SIZE = TILE_SIZE // TILE_SIZE = 1`

**量化过程**（`_downcast_to_mxfmt.py:380-420`）：
1. 计算 E8M0 scale：`quant_scale = (1 / dequant_scale_rounded) * (2**6.0)` — 乘以 `2^6` 是 MXINT8 特有的，将 scale 的精度移到整数范围
2. 量化：`quant_tensor = f32_tensor * quant_scale`
3. 取整并 clamp：`rint(quant_tensor)` 然后 `clamp(-127, 127)`
4. 转换为 `tl.int8`

**打包**（`_downcast_to_mxfmt.py:526-527`）：
```python
# MXINT8 不是 FP6 也不是 4-bit，直接走 else 分支
pack_tensor = quant_values  # 不做额外打包
```

**Scale tensor**（`_downcast_to_mxfmt.py:401-404`）：
```python
quant_scale_exponent = quant_scale_exponent.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE])
scale_tensor = (quant_scale_exponent >> 23).to(tl.uint8)  # 提取 E8M0 指数部分
```

### 返回值总结

| 返回值 | Shape | Data Type | 说明 |
|--------|-------|-----------|------|
| `packed_key` | `(1, TILE_SIZE)` 即 `(1, 32)` | `tl.int8` | 量化后的 INT8 值，范围 [-127, 127]，每个元素直接对应原始数据一个元素，无需打包 |
| `key_scale` | `(1, 1)` | `tl.uint8` | E8M0 格式的 scale，每个 MX block (32 个元素) 共享一个 scale，存储为 biased exponent（uint8） |

**具体来说**：
- **`packed_key`**：shape `(1, TILE_SIZE)`，dtype `tl.int8`。因为 MXINT8 是 8-bit 整数格式，每个量化值恰好占一个字节，所以不需要像 FP4（2个值压成1字节）或 FP6（4个值压成3字节）那样打包。
- **`key_scale`**：shape `(1, TILE_SIZE // MX_BLOCK_SIZE)` = `(1, 1)`（当 TILE_SIZE == MX_BLOCK_SIZE == 32 时），dtype `tl.uint8`。这是 E8M0 格式的 biased exponent，表示一个 2 的幂 dequant_scale。

### 存储流程

在 `reshape_and_cache_kernel_flash` 中存储时（line 265-279）：

```python
# MXINT8 不是 IS_FP6 也不是 IS_4BIT
k_packed = tl.reshape(packed_key, (PACKED_TILE_SIZE,))  # PACKED_TILE_SIZE = TILE_SIZE = 32
# 因为不是 4-bit，需要 bitcast 到 uint8
k_packed = k_packed.to(tl.uint8, bitcast=True)  # int8 -> uint8 bitcast
# 存储到 key_cache_ptr (uint8 buffer)
```

Scale 存储（line 281-286）：

```python
k_s_u8 = tl.reshape(key_scale, (1,)).to(tl.uint8, bitcast=True)  # 已经是 uint8，bitcast 是 no-op
# 存储到 scale plane
```

---

## 问题 2：`kernel_unified_attention_3d` 如何加载 key/key_scale，Q*K 矩阵乘法怎么做？

### Key 和 Key Scale 的加载

在 `kernel_unified_attention_3d`（`triton_unified_attention.py:793-841`）中，当 `SIMO_QUANT=True` 时：

#### (a) 计算地址偏移

```python
# line 800-805: K Cache 的 planar layout 地址
slot_base_k_offset = (
    physical_block_idx * stride_k_cache_0 + (seq_offset % BLOCK_SIZE) * stride_k_cache_1
)[:, None]
k_packed_offset = slot_base_k_offset + kv_head_idx * PACKED_HEAD_SIZE
k_scales_offset = slot_base_k_offset + SCALE_PLANE_OFFSET + kv_head_idx * SCALE_HEAD_SIZE
```

**Planar 布局**：每个 token slot 中，先存所有 head 的 packed data，再存所有 head 的 scale data：
```
| head0_packed | head1_packed | ... | headN_packed | head0_scale | head1_scale | ... | headN_scale |
|<---------- SCALE_PLANE_OFFSET (= num_kv_heads * PACKED_HEAD_SIZE) ---------->|
```

#### (b) 加载 packed data（非FP6路径）

```python
# line 821-826
packed_offs_d = tl.arange(0, PACKED_HEAD_SIZE_PADDED)
packed_mask = row_mask & (packed_offs_d[None, :] < PACKED_HEAD_SIZE)
k_packed = tl.load(
    key_cache_ptr + k_packed_offset + packed_offs_d[None, :], mask=packed_mask, other=0
)
```

`k_packed` 的 shape：`(TILE_SIZE, PACKED_HEAD_SIZE_PADDED)`
- `TILE_SIZE` = 序列方向的 tile 大小（32）
- `PACKED_HEAD_SIZE_PADDED` = `next_power_of_2(PACKED_HEAD_SIZE)`
- 对于 MXINT8，`PACKED_HEAD_SIZE = head_dim`（例如 128），`PACKED_HEAD_SIZE_PADDED = 128`
- dtype = `tl.uint8`（因为 cache 是 uint8）

#### (c) 加载 scale

```python
# line 832-838
k_scales = tl.load(
    key_cache_ptr + k_scales_offset + scale_offs_d[None, :], mask=scales_mask, other=0
)
```

`k_scales` 的 shape：`(TILE_SIZE, SCALE_HEAD_SIZE_PADDED)`
- `SCALE_HEAD_SIZE_PADDED` = `next_power_of_2(SCALE_HEAD_SIZE)`
- 对于 MXINT8 (block_size=32, head_dim=128)，`SCALE_HEAD_SIZE = 128/32 = 4`，`SCALE_HEAD_SIZE_PADDED = 4`
- dtype = `tl.uint8`（E8M0 biased exponent）

### Q*K 矩阵乘法方式

在 `kernel_unified_attention_3d` 的 line 936-952：

```python
S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
if MX_FORMAT_ID == MXFP4_E2M1:
    # MXFP4: 使用 tl.dot_scaled（硬件加速）
    ...
elif MX_FORMAT_ID == MXFP8_E4M3:
    # MXFP8 E4M3: 使用 tl.dot_scaled（硬件加速）
    S += scale * tl.dot_scaled(Q, None, "bf16", k_packed.T, k_scales, "e4m3", fast_math=True)
elif MX_FORMAT_ID == MXFP8_E5M2:
    # MXFP8 E5M2: 使用 tl.dot_scaled（硬件加速）
    S += scale * tl.dot_scaled(Q, None, "bf16", k_packed.T, k_scales, "e5m2", fast_math=True)
elif MX_FORMAT_ID > 0:
    # MXFP6, MXINT8, NVFP4: 软件反量化 + tl.dot
    K = tl.trans(_unpack_and_dequant_mxfmt(k_packed, k_scales, MX_FORMAT_ID))
    S += scale * tl.dot(Q, K)
else:
    S += scale * tl.dot(Q, K)
```

### 对于 MXINT8 (`MX_FORMAT_ID == 1`)：使用**软件反量化 + `tl.dot`**

MXINT8 命中 `elif MX_FORMAT_ID > 0` 分支（line 947-950），因为它不是 MXFP4 (6)、MXFP8_E4M3 (3)、MXFP8_E5M2 (2) 中的任何一个。

具体流程：

1. **软件反量化**：`_unpack_and_dequant_mxfmt(k_packed, k_scales, MX_FORMAT_ID=MXINT8)`
   - `k_packed` (uint8) 先 bitcast 到 `tl.int8`，再转为 `tl.bfloat16`
   - `k_scales` (uint8 E8M0) 左移 7 位变成 bf16 的 biased exponent
   - 乘以 `2^(-6)` 的 INT8 修正因子
   - 最终输出 bf16 的反量化后的 K tensor
   - 返回 shape：`(TILE_SIZE, PACKED_HEAD_SIZE_PADDED)` = `(32, 128)`，dtype = `tl.bfloat16`

2. **转置**：`tl.trans(...)` 得到 `K` shape `(128, 32)` = `(HEAD_SIZE, TILE_SIZE)`

3. **矩阵乘法**：`tl.dot(Q, K)` 其中
   - `Q`: shape `(BLOCK_M, HEAD_SIZE_PADDED)` = `(16, 128)`，dtype = `tl.bfloat16`
   - `K`: shape `(HEAD_SIZE_PADDED, TILE_SIZE)` = `(128, 32)`，dtype = `tl.bfloat16`
   - 输出 `S`: shape `(BLOCK_M, TILE_SIZE)` = `(16, 32)`，dtype = `tl.float32`

### V 的处理方式

V 也是同样的软件反量化路径（line 1012-1014）：

```python
if MX_FORMAT_ID > 0:
    V = _unpack_and_dequant_mxfmt(v_packed, v_scales, MX_FORMAT_ID)
acc += tl.dot(P.to(V.dtype), V)
```

V 不需要转置，因为 `P` shape 是 `(BLOCK_M, TILE_SIZE)`，`V` shape 是 `(TILE_SIZE, HEAD_SIZE_PADDED)`。

### 总结对比表

| 格式 | `MX_FORMAT_ID` | Q*K 计算方式 | 是否使用 `tl.dot_scaled` |
|------|----------------|-------------|-------------------------|
| MXFP4_E2M1 | 6 | `tl.dot_scaled`（硬件 MX 加速） | **是** |
| MXFP8_E4M3 | 3 | `tl.dot_scaled`（硬件 MX 加速） | **是** |
| MXFP8_E5M2 | 2 | `tl.dot_scaled`（硬件 MX 加速） | **是** |
| **MXINT8** | **1** | **软件反量化 → `tl.dot`** | **否** |
| MXFP6_E3M2 | 4 | 软件反量化 → `tl.dot` | 否 |
| MXFP6_E2M3 | 5 | 软件反量化 → `tl.dot` | 否 |
| NVFP4_E2M1 | 7 | 软件反量化 → `tl.dot` | 否 |

原因：`tl.dot_scaled` 是 Triton 对 NVIDIA Blackwell/Hopper 硬件 MX 指令的封装，仅支持 `"e2m1"`、`"e4m3"`、`"e5m2"` 三种 RHS 格式。MXINT8、MXFP6、NVFP4 没有对应的硬件 MX 指令，必须走软件反量化后再用标准 `tl.dot`（bf16 matmul）。

## 2026-04-27 sglang_simo vs vllm_simo W4A4/W8A8 MXFP PPL 和分布分析

### 结论

这次 w4a4 mxfp 下 sglang_simo 的 `word_perplexity` 明显差于 vllm_simo，主要不是 `SIMOLinearMethod.apply` 里的权重量化/反量化幅值不一致导致的。四份带均值日志显示，同一位宽下 sglang 和 vllm 的 `dq_w_abs_mean/max` 完全一致，`qdq_x` 和 `output` 的整体绝对值分布也基本一致。

真正可见的差异首先出现在 `self_attn.o_proj` 的输入，也就是 attention 输出进入 `o_proj` 之前。w4a4 下 sglang 的 `o_proj` `qdq_x_abs_mean` 比 vllm 高约 38%，`o_proj` `output_abs_mean` 高约 35%；而 qkv、gate_up、down 的均值基本对齐。w8a8 下也有同方向差异，但只有 13%左右，W8A8 精度足够，最终 PPL 仍基本一致。

所以更合理的解释是：W4A4 MXFP 对 attention 输出的数值误差极其敏感，sglang 使用的 attention 执行路径和 vllm 不同，w4a4 的 q/k/v 与 v 再经过 attention 后在 `o_proj` 输入处被放大；之后 `o_proj` 又做 A4 量化，误差继续传播，最终 PPL 从 vllm 的约 10 放大到 sglang 的约 27。W8A8 MXFP 量化误差小很多，同样的 backend 差异没有造成明显 PPL 退化。

### PPL 对比

新日志结果：

| 配置 | sglang_simo | vllm_simo |
|---|---:|---:|
| no quant | 未在新日志中测 | 未在新日志中测 |
| w8a8 mxfp | 2.9432 | 2.9707 |
| w4a4 mxfp | 27.6423 | 10.2593 |

旧日志结果：

| 配置 | sglang_simo | vllm_simo |
|---|---:|---:|
| no quant | 2.8325 | 2.8616 |
| w8a8 mxfp | 2.9432 | 2.9703 |
| w8a8 mxint | 2.8423 | 2.8737 |
| w8a8 fp8 per channel | 2.8905 | 2.9224 |
| w8a8 int8 per block | 3.0725 | 2.9668 |
| w6a6 mxfp | 2.9301 | 2.9669 |
| w4a16 int4 per group | 3.7414 | 4.0577 |
| w4a4 nvfp | 4.4210 | 4.1701 |
| w4a16 nvfp4 per group | 3.9018 | 3.6270 |
| w4a4 mxfp | 27.6423 | 10.2393 |

旧日志支持同一个判断：多数格式 sglang/vllm PPL 接近，只有 W4A4 MXFP 差异巨大。W4A4 MXFP 是同时量化权重和激活到 E2M1 的最低精度组合，且没有 NVFP 的额外全局归一化；W4A16、W8A8、W6A6 或 NVFP 都没有暴露出同等程度的问题。

### 代码层面

`simo/extensions/sglang_simo/quantization/quantization.py` 的 `SIMOLinearMethod.apply` 和 `simo/extensions/vllm_simo/quantization/quantization_method.py` 的 `SIMOLinearMethod.apply` 主流程一致：`input` flatten 后先 `input_downcast_kernel`，再 `input_upcast_kernel` 得到 `qdq_x`；权重用 `weight_upcast_kernel` 得到 `dq_w`；最后 `torch.matmul(qdq_x, dq_w.T)`。

vllm_simo 版本额外有 per-block 多 shard padding/slicing 逻辑，但这段只对 `QuantizeGranularity.PER_BLOCK` 生效；`QuantizeSpecMX` 在 `get_quantize_granularity` 里固定返回 `PER_GROUP`，所以 w4a4/w8a8 MXFP 不会走这段逻辑。这不是本次差异的主因。

配置上有一个不一致：sglang w4a4 mxfp 配置里有 `"per_quant_opt": "online_down_proj_rotation"`，vllm w4a4 mxfp 配置是 `null`。但当前 sglang_simo 的 `get_quant_method()` 没有使用 `per_quant_opt` 去打开 Hadamard/down_proj rotation，`get_quant_method_by_target_spec()` 也总是用默认 `hadamard=False` 创建 `SIMOLinearMethod`。因此这个配置项在当前 sglang 路径里基本没有实际效果，不足以解释本次 PPL 差异。不过后续如果补齐 sglang 的 rotation 支持，需要先把两边配置对齐再比较。

### 均值和最大值差异

w8a8 mxfp 聚合结果：

| 指标 | sglang | vllm | 说明 |
|---|---:|---:|---|
| overall `qdq_x_abs_mean` | 0.06803 | 0.06772 | 基本一致 |
| overall `dq_w_abs_mean` | 0.009911 | 0.009911 | 完全一致 |
| overall `output_abs_mean` | 0.23731 | 0.23236 | sglang 高约 2.1% |
| overall `qdq_x_abs_max` | 288 | 288 | 一致 |
| overall `dq_w_abs_max` | 112 | 112 | 一致 |
| overall `output_abs_max` | 270 | 270 | 一致 |
| `o_proj qdq_x_abs_mean` | 0.01280 | 0.01125 | sglang 高约 13.7% |
| `o_proj output_abs_mean` | 0.00523 | 0.00460 | sglang 高约 13.6% |

w4a4 mxfp 聚合结果：

| 指标 | sglang | vllm | 说明 |
|---|---:|---:|---|
| overall `qdq_x_abs_mean` | 0.06713 | 0.06685 | 基本一致 |
| overall `dq_w_abs_mean` | 0.009741 | 0.009741 | 完全一致 |
| overall `output_abs_mean` | 0.22532 | 0.22463 | 基本一致 |
| overall `dq_w_abs_max` | 96 | 96 | 一致 |
| overall `output_abs_max` | 408 | 408 | 一致 |
| `qkv_proj output_abs_mean` | 0.73305 | 0.72980 | 基本一致 |
| `gate_up_proj output_abs_mean` | 0.14904 | 0.15128 | 基本一致 |
| `down_proj output_abs_mean` | 0.01191 | 0.01204 | 基本一致 |
| `o_proj qdq_x_abs_mean` | 0.02104 | 0.01520 | sglang 高约 38.4% |
| `o_proj output_abs_mean` | 0.00728 | 0.00539 | sglang 高约 35.0% |

因此，单看你打印的绝对值 mean/max，不能解释 PPL 从 10 到 27 的差距；这些统计量会隐藏符号、channel 顺序、attention softmax 后的逐 token 误差以及输出方向误差。它们能说明的是：权重量化幅度一致，线性层整体幅值没有崩；差异更可能发生在线性层之间，尤其是 attention 输出到 `o_proj` 输入这一段。

### 建议的下一步验证

1. 在 `qkv_proj` 输出后、attention 输出后、`o_proj` 输入前分别 dump 小 batch 的 signed tensor，比较 sglang/vllm 的 cosine similarity、max relative error 和 top-k token logit 差异。只看 abs mean/max 不够。
2. 固定同一条 prompt、同一批次、同一 context 长度，关闭会改变调度/分块的选项，再比较每层 `o_proj` 输入误差在哪一层开始放大。
3. 让 sglang w4a4 mxfp 显式尝试非 `fa3` attention backend（如果当前环境支持），看 PPL 是否向 vllm 的 10.x 靠近。
4. 把 sglang w4 配置里的 `per_quant_opt` 临时改为 `null` 再跑一次。按当前代码判断它不应影响结果；如果结果变化，说明实际运行路径还有额外 patch 读取了该字段，需要继续查注册/monkey patch。

## kgp shell 脚本 C++ 重写

项目已放在 `/softhome/like/asset/code/cpp_guard`，包含一个 C++ 源文件 `kgp.cpp` 和 `CMakeLists.txt`，生成的 ELF 名字是 `kgp`。

命令行同时覆盖原来两个脚本的能力：

```bash
# 等价于原 kgp.sh 的单次扫描/kill
kgp "1,3,5" like 11
kgp --once "1,3,5" like 11

# 等价于原 kgp-loop.sh 的每秒循环执行，并新增 sig_num 参数
kgp --loop "1,3,5" like 11
```

实现逻辑和 shell 版一致：

1. 调用 `nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits`，建立 GPU index 到 GPU UUID 的映射。
2. 调用 `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits`，获取每个 compute 进程所在 GPU UUID 和 PID。
3. 对命令行给出的 GPU index 列表逐个匹配对应 UUID。
4. 对匹配到的 PID 调用 `ps -o user= -p <pid>` 查询 owner。
5. 如果 owner 等于 `protected_user`，跳过；否则打印 `[KILL] GPU ... PID ... owned by ...`，再执行 `kill -<sig_num> <pid>`。

代码没有使用三方库。C++ 侧只使用标准库做参数解析、字符串处理、临时文件读取和循环 sleep；外部系统信息仍通过允许的系统命令 `nvidia-smi`、`ps`、`kill` 获取或执行。为了保持 `kill` 信号可配置，`sig_num` 会校验为正整数，然后拼到 `kill -<sig_num>` 命令里。

构建方式：

```bash
cd /softhome/like/asset/code/cpp_guard
cmake -S . -B build
cmake --build build
./build/kgp --help
```
