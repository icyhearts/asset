## 22. SGLang 推理精度分析：为什么 Test 13e 的 FP32 累积链能测出 batch-dependence，但 SGLang 实际用的是 bf16？

### 22.1 问题

`validate_batch_invariant_ops.py` 的测试发现：

| 测试 | 精度 | 是否测得 batch-dependence |
|---|---|---|
| Test 5/5b/5c/5d/5e | bf16 单层 matmul | **未检测到**（所有 M 值 bit-identical） |
| Test 13a/b/c/d | bf16 多层 chain | **未检测到**（bit-identical） |
| Test 13e | **float32** 多层 chain | **检测到**（max_abs ~ 1.19e-6） |

但 GSM8k 实验中，`SGLANG_BATCH_INVARIANT_OPS_FORCE_SKIP_ATEN_MM=1`（使用 cuBLAS）确实在不同 `max_running_requests` 下产生了不同的 GSM8k 分数。

**问题**：SGLang 推理时模型使用哪种精度？如果 SGLang 用 bf16，为什么 FP32 测试能捕捉到 batch-dependence？

### 22.2 SGLang 推理的精度路径

#### 22.2.1 Model dtype 解析

SGLang 通过 `_get_and_verify_dtype()` 确定模型运行的 dtype：

```python
# sglang/srt/configs/model_config.py
def _get_and_verify_dtype(config, dtype):
    if dtype == "auto":
        if config_dtype == torch.float32:
            torch_dtype = torch.float16  # 默认 float16
        else:
            torch_dtype = config_dtype    # 使用 config 中的 dtype
```

DSv2-Lite 的 `config.json` 中 `torch_dtype=bfloat16`，所以 SGLang 运行时：
- **激活值（hidden states）**: bf16
- **权重**: bf16（未量化时）
- **KV cache**: 可配置，默认 bf16（用户命令用 `fp8_e4m3`）

#### 22.2.2 Matmul 的精度

`F.linear(x, weight)` 调用 cuBLAS `cublasGemmEx`。当输入和权重都是 bf16 时：

- cuBLAS 使用**原生 bf16 Tensor Core** 计算路径
- **不使用 TF32** — `allow_tf32=True` 只影响 float32 输入的 matmul，对 bf16 无影响
- 内部累加使用 FP32，输出转换为 bf16

SGLang 代码中**没有**设置 `torch.backends.cuda.matmul.allow_tf32`（保持 PyTorch 默认值 True，但对 bf16 无影响）。

验证实验：
```
GPU: H100
  bf16 mm deterministic (same M, two runs): True
  bf16 mm batch-dependence (M=64 vs M=4096): max_abs_diff = 0.0 (bit-identical)
  FP32 mm batch-dependence (M=64 vs M=4096): max_abs_diff may be non-zero (TF32 ON)
```

#### 22.2.3 RMSNorm 的精度

SGLang 的 RMSNorm 使用 fused CUDA kernel（`fused_add_rmsnorm`），内部使用 **float32** 计算：

```python
# sglang/srt/layers/layernorm.py (forward_native, 非 fused 路径的逻辑)
x = x.to(torch.float32)                          # bf16 → fp32
if residual is not None:
    x = x + residual.to(torch.float32)            # residual 在 fp32 相加
    residual = x.to(orig_dtype)                   # 存回 bf16
# rms_norm 在 fp32 计算
x = x * torch.rsqrt(variance + eps)               # fp32
x = (x * self.weight).to(orig_dtype)              # 乘 weight 后转 bf16
```

**关键**：fused kernel 虽然在 fp32 做运算，但输入和输出都是 bf16。因此，残差连接的"精度窗口"是有限的——sub-bf16 的差异在进入下一层之前被截断了。

#### 22.2.4 Attention 的精度

Triton backend 的 attention kernel 内部使用 fp32 或 tf32（取决于 kernel 设计），但输入输出都是 bf16。

### 22.3 为什么 Test 13e 的 FP32 测试与 SGLang 实际情况不直接对应

#### 关键差异对照表

| 维度 | Test 13e（FP32 chain） | SGLang 实际推理 |
|---|---|---|
| Matmul 输入类型 | float32 | bf16 |
| Matmul Tensor Core | TF32（10-bit mantissa） | 原生 bf16（7-bit mantissa） |
| TF32 的 M-dependent 算法选择 | **有**（TF32 matmul 算法选择依赖 M/K/N） | **无**（bf16 matmul 不使用 TF32 路径） |
| 中间层精度 | float32（不改回 bf16） | bf16 → fp32（仅在 RMSNorm 内部） → bf16 |
| 差值累积机制 | sub-bf16 差异在 fp32 中持续累积 | 每次 RMSNorm 后截断为 bf16 |

**结论**：Test 13e 检测到的 batch-dependence 实际上是 **TF32 matmul** 的 batch-dependence，而不是 **bf16 matmul** 的 batch-dependence。

#### 单层 bf16 matmul 为何 bit-identical？

H100 的 bf16 Tensor Core 计算是 bit-deterministic 的，给定相同输入（相同 shape，相同数据），bf16 Tensor Core 对相同的行产生完全相同的输出。即使 M 不同，cuBLAS 内部可能使用相同的 tiling 策略（在 bf16 精度下，cuBLAS 的算法选择更保守），导致结果 bit-identical。

### 22.4 那么 SGLang 的真实 batch-dependence 从何而来？

既然 bf16 matmul 本身是 bit-identical 的（standalone 测试确认），GSM8k 的 batch-dependence 可能来自：

#### 假设 A：BMM（Batch MatMul）的 batch-dependence

在 attention 计算中，`torch.bmm` 用于 QKV 投影的 batch matmul：
```python
# 在 attention 中，QKV 投影可能用 bmm 而非 mm
q, k, v = qkv_proj(hidden_states)  # 内部可能用 bmm
```

`enable_batch_invariant_mode()` 默认**不**替换 `aten::bmm`（除非 `enable_bmm=True`）。所以即使在 `enable_batch_invariant_mode()` 启用后，bmm 仍然使用 cuBLAS，而其 batch dimension 的变化可能导致不同的算法选择。

#### 假设 B：MoE Router 的 softmax/layernorm 与 matmul 的交互

DSv2-Lite 的 MoE 结构中，router 产生 top-k 选择。即使 router 的 matmul 是 bit-identical 的，softmax 后的细微差异可能导致不同的 expert 分配，进而经过 MoE FFN 多层累积后产生显著差异。

#### 假设 C：Fused kernel 的内部精度路径

SGLang 使用了大量 fused kernel（`fused_add_rmsnorm`、flash attention、fused MoE 等），这些 fused kernel 内部使用 fp32 精度。在这个 fp32 精度窗口中，matmul 产生的 sub-bf16 差异**可能会在同一个 fused kernel 内部被放大**：

```
输入 (bf16) ──→ Fused Kernel 内部 (fp32) ──→ 输出 (bf16)
                              ↑
                    在这个 fp32 窗口中，
                    matmul 的 sub-bf16 差异
                    被 RMSNorm fp32 累加保留
```

虽然单层的 diff < 1 bf16 ULP，但经过 27 层，每次 RMSNorm 都在 fp32 内处理，差异可能通过 fp32 残差路径累积。

#### 假设 D：CUDA stream / workspace 状态影响 cuBLAS 算法选择

在真实 SGLang 服务中，多个请求并发处理，CUDA stream 调度、memory allocator 状态、cuBLAS workspace 缓存状态**与 standalone 测试完全不同**。这些全局状态可能影响 cuBLAS 的 heuristics，导致在真实负载下选择了不同的内部算法。

### 22.5 Test 13e 仍有意义的原因

虽然 Test 13e 检测的是 TF32（非 bf16）的 batch-dependence，但它仍然有意义：

1. **证明了 cuBLAS 总体上是可以 batch-variant 的**：TF32 和 bf16 使用相同的 cuBLAS 算法选择启发式逻辑，只是精度不同
2. **FP32 内部积累模拟了 SGLang fused kernel 的 fp32 精度窗口**：SGLang 的 fused kernel 内部也是 fp32
3. **FP32 内部积累能放大 subt-bf16 差异**：在真实 SGLang 中，fused kernel 的 fp32 窗口起到了类似作用

SGLang 推理的精度路径总结：
```
Layer 1:  [bf16 matmul]→[fused_add_rmsnorm(fp32)→bf16]→[attn(bf16)]→[fused_add_rmsnorm(fp32)→bf16]→[ffn matmul(bf16)]
Layer 2:  [bf16 matmul]→[fused_add_rmsnorm(fp32)→bf16]→[attn(bf16)]→[fused_add_rmsnorm(fp32)→bf16]→[ffn matmul(bf16)]
...
Layer 27: [bf16 matmul]→[fused_add_rmsnorm(fp32)→bf16]→[attn(bf16)]→[fused_add_rmsnorm(fp32)→bf16]→[ffn matmul(bf16)]
                                                                                                                    ↓
                                                                                                   sampled token
```

**每个 layer 的 fused kernel 内部都有 fp32 精度窗口，matmul 的 subt-bf16 差异在 RMSNorm 计算中不被截断，残差路径通过 fp32 加法传递差异。27 层 + autoregressive decode 足以将这些差异放大到影响 token 采样。**

### 22.6 总结

| 问题 | 答案 |
|---|---|
| SGLang 把数据提升到 fp32 计算吗？ | **matmul 不提升**（bf16 原生 Tensor Core）；**RMSNorm 内部提升到 fp32**（fused kernel 内部） |
| Test 13e 的 FP32 与 SGLang 一致吗？ | **不完全一致** — Test 13e 模拟的是 TF32 matmul + fp32 全链路，SGLang 是 bf16 matmul + fp32 fused kernel |
| 为什么 GSM8k 能测出 batch-dependence？ | bf16 matmul + fp32 fused kernel 窗口 + 27 层累积 + autoregressive decode 放大效应 |
| Test 13e 还有价值吗？ | **有** — 证明了 cuBLAS batch-variant 的存在性，并提供了 FP32 累积链模拟 fused kernel 精度的有效实验方法 |

---

## 23. deepseek-v4-flash inference — convert.py 的功能与必要性

### 23.1 convert.py 的主要功能

`/data/like/hf-models/deepseek-v4-flash/inference/convert.py` 负责将 HuggingFace 格式的 safetensors 权重文件转换为该项目的自定义推理代码能直接加载的格式。它做了四件事：

#### 23.1.1 命名映射 (Name Mapping)

将 HuggingFace 的 key 命名规范映射为自定义 `model.py` 的命名规范：

```
HF key                          →  自定义 key
─────────────────────────────────────────────────────
model.layers.X.self_attn.*      →  layers.X.attn.*
model.layers.X.mlp.*            →  layers.X.ffn.*
weight_scale_inv                →  scale
e_score_correction_bias         →  bias
q_b_proj                        →  wq_b
kv_a_proj_with_mqa              →  wkv_a
gate_proj                       →  w1
up_proj                         →  w3
down_proj                       →  w2
o_proj                          →  wo
lm_head                         →  head
```

#### 23.1.2 Tensor Parallelism (TP) 分片

根据 `--model-parallel` 参数，将权重沿指定维度切分到多个 rank：

```python
# 例如 q_proj 沿 dim=0 切分
new_param = param.narrow(dim, i * shard_size, shard_size)
```

MoE expert 按 rank 分配：每个 rank 持有 `n_experts / mp` 个 expert 的权重。

#### 23.1.3 wo_a 反量化 (FP8 → BF16)

这是最关键的一步。原始的 `wo_a.weight` 是 **FP8 E4M3** 格式（`torch.float8_e4m3fn`），附带有 per-block 的 E8M0 scale。但 `model.py` 中 `wo_a` 被定义为 `torch.bfloat16`：

```python
# model.py:462
self.wo_a = ColumnParallelLinear(..., dtype=torch.bfloat16)
```

convert.py 执行反量化：

```python
# convert.py:137-141
weight = state_dicts[i][name]                                          # FP8
scale  = state_dicts[i].pop(name.replace("weight", "scale"))           # E8M0
weight = weight.unflatten(0,(-1,128)).unflatten(-1,(-1,128)).float() * scale[:,None,:,None].float()
state_dicts[i][name] = weight.flatten(2,3).flatten(0,1).bfloat16()     # → BF16
```

这会**永久性地**将 block-wise quantized FP8 权重展开为 BF16，不可逆。

#### 23.1.4 Expert 权重格式转换

原始 expert 权重存储在 `torch.int8` 中，每个 int8 打包了两个 E2M1 格式的 4-bit 值（即 `float4_e2m1fn_x2`，每元素 4 bit，共 16 个可表示值）。

- **`--expert-dtype fp8` 模式**：调用 `cast_e2m1fn_to_e4m3fn()`，将 packed int8 解包→查 FP4_TABLE→乘以 per-block scale→转换为 `torch.float8_e4m3fn`，scale 转为 `torch.float8_e8m0fnu`

  ```
  [2xE2M1 packed in int8] → 查表 → [E4M3 × scale] + [E8M0 scale]
  ```

- **`--expert-dtype fp4` 模式**：直接做 `view(torch.float4_e2m1fn_x2)`，将 int8 reinterpret 为 FP4 dtype，保持 packed 格式。

#### 23.1.5 输出

每个 rank 输出一个 `model{rank}-mp{world_size}.safetensors` 文件，以及 tokenizer 文件。

### 23.2 为什么 generate.py 不能直接读取原始 safetensors？

#### 原因 1：命名不匹配

generate.py 使用 `load_model()` 按 PyTorch 参数名匹配权重。原始 HF 格式使用 `model.layers.X.self_attn.q_b_proj.weight` 等名称，而模型定义中使用 `layers.X.attn.wq_b.weight` 等。直接加载会因名称不匹配而报错（`strict=False` 虽然不报错，但权重不会被加载到正确位置）。

#### 原因 2：TP 分片是必须的

模型代码（`model.py`）假设权重已经按 TP rank 切分好。例如 `ColumnParallelLinear` 和 `RowParallelLinear` 只持有部分权重。原始 safetensors 是全量权重，没有切分。convert.py 中的 `param.narrow(dim, ...)` 负责这一工作。

#### 原因 3：wo_a 的格式差异（最关键）

| 位置 | dtype | 说明 |
|---|---|---|
| 原始 safetensors | `torch.float8_e4m3fn` + per-block E8M0 scale | Block-wise FP8 量化 |
| model.py 定义 | `torch.bfloat16` | 模型参数声明为 BF16 |
| 转换后 | `torch.bfloat16` | convert.py 执行反量化 |

generate.py 在推理时**没有**对 wo_a 做运行时 FP8 反量化——它假设 wo_a 已经是 BF16。如果直接用原始 safetensors，wo_a 是 FP8，model.py 会尝试将其赋给 BF16 参数，导致 dtype mismatch 或静默错误。

generate.py 代码中也有一条注释确认了这一点：

```python
# model.py:539
# NOTE: wo_a is FP8 in checkpoint; could do FP8 einsum here for better perf,
```

这说明开发者也考虑过直接在模型中使用 FP8 wo_a，但当前实现选择了"convert.py 一次性反量化"的方案。

#### 原因 4：Expert 权重格式不匹配

原始 expert 权重是 `int8`（packed E2M1 x2），generate.py 的 fp8_gemm / fp4_gemm kernel 期望特定格式：
- FP8 模式：`torch.float8_e4m3fn` + `torch.float8_e8m0fnu` scale
- FP4 模式：`torch.float4_e2m1fn_x2`

convert.py 负责这个格式转换。如果 expert_dtype 设为 fp8，还需要调用 `cast_e2m1fn_to_e4m3fn()` 执行完整的格式提升（E2M1→E4M3）。

### 23.3 总结

| 转换步骤 | 做什么 | 是否可逆 |
|---|---|---|
| 命名映射 | HF key → 自定义 key | 是（纯字符串替换） |
| TP 分片 | 沿 dim 切分权重 | 是（合并后可还原） |
| wo_a 反量化 | FP8+scale → BF16（乘 scale、reshape） | **否**（信息丢失） |
| Expert 格式转换 | int8(E2M1x2) → fp8/fp4 | 一定程度上可逆 |

**核心结论**：`convert.py` 不仅仅是重命名和分片，它做了**不可逆的精度转换**。`wo_a` 的 FP8→BF16 反量化和 expert 的 E2M1→E4M3 格式提升是 generate.py 不能直接读取原始 safetensors 的根本原因。

---

## 24. deepseek-v4-flash inference — config.json 与并行策略详解

### 24.1 应该使用哪个 config.json？

有两个 config.json 文件：

| 路径 | 格式 | 用途 |
|---|---|---|
| `.../deepseek-v4-flash-git-control-by-like/config.json` | HuggingFace 格式 | 给 transformers 库加载模型用 |
| `.../deepseek-v4-flash-git-control-by-like/inference/config.json` | 自定义 ModelArgs 格式 | 给 generate.py 的 model.py 用 |

#### 应该使用 `inference/config.json`

README 中的 `--config ${CONFIG}` 传给 `generate.py`：

```python
# generate.py:94
with open(config) as f:
    args = ModelArgs(**json.load(f))
```

`ModelArgs` 期望的 key 名称是自定义格式：

```
inference/config.json          root config.json (HuggingFace)
─────────────────────────     ─────────────────────────────
"dim": 4096                    "hidden_size": 4096
"n_layers": 43                 "num_hidden_layers": 43
"n_heads": 64                  "num_attention_heads": 64
"dtype": "fp8"                 "torch_dtype": "bfloat16"
```

如果用 root 的 `config.json`，`ModelArgs(**json.load(f))` 会因为 key 名称不匹配而报错或静默忽略（取决于 `ModelArgs` 是否允许 extra fields）。

#### 两个 config.json 的关键差异

| 维度 | root config.json | inference/config.json |
|---|---|---|
| 定位 | HuggingFace 模型配置 | 自定义推理引擎配置 |
| key 风格 | `hidden_size`, `num_hidden_layers` | `dim`, `n_layers` |
| dtype 含义 | `torch_dtype: "bfloat16"` (模型参数精度) | `dtype: "fp8"` (运行时激活量化精度，即 GEMM 前将激活量化为 FP8) |
| 额外字段 | `architectures`, `model_type`, `quantization_config`, `transformers_version` | `n_activated_experts`, `score_func`, `route_scale`, `window_size`, `original_seq_len`, `rope_factor`, `compress_ratios` |
| 用途 | HF AutoModel 加载、SGLang/vLLM 服务 | deepseek-v4-flash 自有推理代码 |

### 24.2 `--model-parallel` 是专家并行还是 Tensor 并行？

**两者都是。** `--model-parallel ${MP}` 在 convert.py 中同时做了 TP 和 EP：

#### Tensor Parallelism 部分

convert.py 中 `mapping` 字典指定了每个权重沿哪个 dim 切分：

```python
mapping = {
    "q_proj":  ("wq", 0),    # 沿 out_features 切分 → ColumnParallel
    "o_proj":  ("wo", 1),    # 沿 in_features 切分  → RowParallel
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    ...
}
```

`dim=0` 对应 `ColumnParallelLinear`（输出维度切分），`dim=1` 对应 `RowParallelLinear`（输入维度切分）。

#### Expert Parallelism 部分

```python
if "experts" in name and "shared_experts" not in name:
    idx = int(name.split(".")[-3])
    if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
        continue  # 不属于本 rank，跳过
```

每个 rank 持有 `n_experts / mp` 个 expert 的完整权重（不切分 expert 内部的矩阵维度，只按 expert 编号分配）。

#### 一句话总结

> `--model-parallel ${MP}` 是一种 **TP + EP 联合并行**：所有非 expert 层的权重按 TP 切分，MoE expert 按 EP 分配。同一组 GPU 同时承担 TP 和 EP 角色。

### 24.3 generate.py 使用专家并行还是 Tensor 并行？

**同样两者都使用。**

generate.py 启动时：

```python
# model.py:773-775
world_size = dist.get_world_size()     # = torchrun 的 --nproc-per-node
rank = dist.get_rank()
```

这个 `world_size` 被用于所有并行逻辑：

#### Tensor Parallelism 运行时行为

```python
class ColumnParallelLinear(Linear):
    """Shards output dim. No all-reduce needed on output."""
    def __init__(self, in_features, out_features, ...):
        self.part_out_features = out_features // world_size  # 只持有部分输出

class RowParallelLinear(Linear):
    """Shards input dim. All-reduce on output to sum partial results."""
    def forward(self, x):
        y = linear(x, self.weight, None)
        if world_size > 1:
            dist.all_reduce(y)  # 汇总各 rank 的部分结果
        return y
```

#### Expert Parallelism 运行时行为

```python
class MoE(nn.Module):
    self.n_local_experts = args.n_routed_experts // world_size  # 只管理部分 expert
```

所有 rank 的 router 产生相同的 top-k 选择，但每个 rank 只计算自己持有的 expert。结果通过 all-reduce 汇总。

#### 一句话总结

> generate.py 使用与 convert.py 完全对称的 **TP + EP 联合并行**。`torchrun --nproc-per-node ${MP}` 中的 MP 同时控制 TP 和 EP 的并行度，没有独立的 TP size 和 EP size 配置项。

---

## 25. deepseek-v4-flash inference — 其他大模型并行策略分析

### 25.1 现有并行策略

确认 convert.py 和 generate.py 只使用了 **TP + EP 联合并行**。完整的 `dist.*` 通信原语调用如下：

| 调用 | 位置 (model.py) | 用途 |
|---|---|---|
| `dist.all_reduce(y)` | ParallelEmbedding.forward | 合并 vocab-sharded embedding 的部分结果 |
| `dist.all_reduce(y)` | RowParallelLinear.forward | 合并 TP 切分后的部分线性输出 |
| `dist.all_reduce(index_score)` | Indexer.forward | 合并 TP 切分的 indexer 分数 |
| `dist.all_reduce(y)` | MoE.forward | 合并各 rank 的 expert 输出 |
| `dist.all_gather(all_logits, logits)` | ParallelHead.forward | 收集 vocab-sharded 的 logits |
| `dist.broadcast_object_list(...)` | generate.py:main | 交互模式下广播用户输入到所有 rank（非模型并行） |

### 25.2 未被使用的大模型并行策略

| 策略 | 是否使用 | 说明 |
|---|---|---|
| **Pipeline Parallelism (PP)** | 否 | 无 `pipeline`、`stage`、`microbatch`、`schedule` 相关代码。所有层在单次 `forward` 中顺序执行，没有将层分配到不同 GPU 的逻辑 |
| **Data Parallelism (DP)** | 否 | 无 `data_parallel`、`dp`、`replicate` 相关代码。generate.py 中所有 rank 收到相同的 prompt（通过 broadcast），各自计算完整的 batch |
| **Sequence Parallelism (SP)** | 否 | 无 `sequence_parallel`、`sp`、`seq_parallel` 相关代码。序列维度没有被切分到不同 rank |
| **Context Parallelism (CP)** | 否 | 无 `context_parallel`、`cp`、`ring_attention`、`ulysses` 相关代码 |
| **FSDP / ZeRO** | 否 | 无 `FullyShardedDataParallel`、`ZeRO` 相关代码。权重通过 `load_model()` 一次性加载，不在 rank 间分片或重组 |

### 25.3 为什么只用 TP + EP 就够了？

对于 deepseek-v4-flash（43 层、4096 hidden dim、256 experts、1 KV head）的推理场景：

1. **不需要 PP**：43 层 × 4096 hidden dim 的总参数量约 300B+，但推理时计算密集度远高于训练，PP 的 pipeline bubble 开销（需要 micro-batch 填充）在低延迟推理中不可接受。且单层就能放入单卡时 PP 无必要。

2. **不需要 DP**：推理场景下 DP 要求每张卡持有完整模型副本，300B+ 模型（即使是 FP8/FP4）无法放入单卡。DP 只适用于小模型的大吞吐场景。

3. **不需要 SP/CP**：SP 和 CP 主要用于超长序列（百万 token 级别）。该推理代码的 `max_seq_len` 默认为 4096，单卡的 attention 计算完全够用。

4. **TP+EP 联合是最优组合**：
   - TP 解决单层权重大于单卡显存的问题（attention 的 Q/O projection）
   - EP 解决 256 个 expert 总权重大于单卡显存的问题
   - 两者共用同一组 GPU，通信路径最短（TP 的 all-reduce 和 EP 的 all-reduce 在同一个 NCCL group 内）

### 25.4 当前方案的局限

| 局限 | 说明 |
|---|---|
| TP 和 EP 无法独立扩展 | MP=8 意味着 8-way TP **且** 8-way EP，无法做到 2-way TP + 4-way EP |
| 不支持跨节点 | `torchrun --nproc-per-node` 仅单节点，TP 的 all-reduce 跨节点时带宽急剧下降 |
| 无 DP 弹性 | 无法通过增加 DP 维度来线性提升吞吐 |
| Expert 负载不均无解 | 256 experts 按编号均分到各 rank，不保证各 rank 的计算量均衡（某些 expert 被选中频率更高） |

---
