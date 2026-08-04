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

## 26. 为什么没有 all-to-all 通信？

标准 MoE 推理的专家并行通常需要 **两次 all-to-all**：第一次将 token 发送到持有对应 expert 的 rank，计算后再 send back。流程为：

```
标准 EP 流程:
  Router → all-to-all(dispatch tokens) → 各 rank 计算本地 expert → all-to-all(combine results)
```

但 deepseek-v4-flash 的 MoE 实现使用 `all_reduce` 而不是 `all_to_all`：

```python
# model.py:629-644 (MoE.forward)
def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    x = x.view(-1, self.dim)
    weights, indices = self.gate(x, input_ids.flatten())  # 所有 rank 独立路由所有 token
    y = torch.zeros_like(x, dtype=torch.float32)
    counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
    for i in range(self.experts_start_idx, self.experts_end_idx):  # 只遍历本地 expert
        if counts[i] == 0:
            continue
        expert = self.experts[i]
        idx, top = torch.where(indices == i)
        y[idx] += expert(x[idx], weights[idx, top, None])   # 用全量 token 中的匹配子集计算
    if world_size > 1:
        dist.all_reduce(y)                                   # ← all_reduce, 不是 all-to-all
    y += self.shared_experts(x)
    return y.type_as(x).view(shape)
```

### 关键差异：每个 rank 持有全量 token

标准 EP 中，token 分布在不同的 DP rank 上，需要 all-to-all 搬运。但在这个架构中，**因为 TP 的存在，所有 rank 持有相同的全量 token**：

```
标准 EP+DP:                          deepseek-v4-flash TP+EP:
┌──────────┐  ┌──────────┐          ┌──────────┐  ┌──────────┐
│ Rank 0   │  │ Rank 1   │          │ Rank 0   │  │ Rank 1   │
│ token A,B│  │ token C,D│          │token A,B │  │token A,B │  ← 相同 token!
│ expert 0 │  │ expert 1 │          │expert 0-3│  │expert 4-7│
└──────────┘  └──────────┘          └──────────┘  └──────────┘
      ↓ all-to-all                       ↓ all_reduce
  dispatch A→0, B→1                 Rank0: expert0-3(A,B) → y₀
  compute, then send back           Rank1: expert4-7(A,B) → y₁
                                          y = y₀ + y₁ (all_reduce)
```

### 为什么 all_reduce 可行？

每个 rank 的 `y` 初始化为 0（float32 零张量），遍历本地 expert 时只修改**属于该 expert 的 token 位置**。对于不归本地 expert 处理的 token-expert 组合，`y` 保持为 0。`all_reduce` 将所有 rank 的部分结果求和，得到完整的 MoE 输出。

```
Rank 0 (expert 0-3):    y = [exp0(token_A), exp1(token_B), 0, 0, ...]
Rank 1 (expert 4-7):    y = [0,              0,             exp4(token_A), exp5(token_B), ...]
all_reduce(y):          y = [exp0(A)+0, exp1(B)+0, 0+exp4(A), 0+exp5(B), ...]  ← 正确!
```

### 这种方案的优劣

| | all-to-all 方案（标准 EP） | all_reduce 方案（本实现） |
|---|---|---|
| token 数据量 | 只搬运被路由的 token | 每个 rank 处理全量 token |
| 通信模式 | 2× all-to-all（dispatch + combine） | 1× all-reduce |
| 通信复杂度 | 依赖 token-to-expert 分布 | 固定 = hidden_dim × num_tokens |
| 计算浪费 | 无（只算本地需处理的 token） | 有（rank 持有全量 token 但只算本地 expert） |
| 实现复杂度 | 高（需要管理 dispatch/combine 的索引和 buffer） | 低（标准 all_reduce） |
| 适合场景 | 大 batch / 训练（token 多，all-to-all 收益大） | 小 batch / 推理（token 少，全量遍历开销可接受） |

### 为什么推理场景下 all_reduce 方案更合适？

1. **推理 batch 小**：单次请求只有几十到几百个 token，全量遍历的额外计算开销微乎其微
2. **实现简单**：不需要管理复杂的 all-to-all 索引映射和 buffer 分配
3. **TP 天然提供全量 token**：TP 架构中所有 rank 本来就持有完整 hidden states（ColumnParallel 输出后再 RowParallel），不需要额外广播
4. **all_reduce 被高度优化**：NCCL 的 all_reduce 在节点内（NVLink）极为高效，而 all-to-all 即使在同一节点也有额外的拓扑调度开销

**结论**：deepseek-v4-flash 的推理代码不需要 all-to-all，因为它用 **全量 token + all_reduce** 替代了 **token dispatch + all-to-all**。这是 TP 架构下推理专用 MoE 的一种常见优化——用少量冗余计算换取通信简洁性。

---

## 27. 为什么 K=10944 时 cuBLAS 单层 mm 直接测出 batch-dependence，而 matmul_persistent 始终 batch-invariant？

### 27.1 实验现象

Test 5 使用 DeepSeek-V2-Lite 真实 FFN 层参数（K=10944, N=4096, bf16, TF32=ON）：

```
torch.mm:          rows=   6  M_small=   6  M_large=   256  equal=False  max_abs=1.0  <-- DIFF!
torch.mm:          rows=  64  M_small=  64  M_large=   256  equal=False  max_abs=2.0  <-- DIFF!
matmul_persistent: rows=   6  M_small=   6  M_large=   256  equal=True   max_abs=0.0  (batch-invariant)
matmul_persistent: rows=  64  M_small=  64  M_large=   256  equal=True   max_abs=0.0  (batch-invariant)
```

K=2048 时 cuBLAS 是 bit-identical，K=10944 时同样代码直接出 DIFF。为什么？

### 27.2 cuBLAS 为什么在 K=10944 时产生 batch-dependence

cuBLAS 内部使用 **启发式算法选择器（heuristics）**，根据 M、K、N 三个维度决定：
- 沿 K 维度的 tile 大小（`K_tile`）
- 沿 M 维度的 tile 大小（`M_tile`）
- 线程块级别的 reduction 策略

**浮点加法不满足结合律**：
```
(a + b) + c ≠ a + (b + c)    在有限精度下
```

不同的 K-tile 意味着不同的累加顺序，产生不同的舍入误差。

```
K=2048, K_tile=128:  累加顺序固定为 [0:128]+[128:256]+...+[1920:2048]
K=10944, K_tile 根据 M 动态选择:
  M=6 (小 batch):  cuBLAS 可能选 K_tile=256, M_tile=4
  M=64+:          cuBLAS 可能选 K_tile=128, M_tile=64

  不同的 K_tile 导致不同的累加顺序 → 不同结果
```

**TF32 加剧了差异**：TF32 只有 10-bit mantissa（vs FP32 的 23-bit），非结合性引起的误差被放大。

**为什么 K=2048 是 bit-identical？** 因为 K=2048 恰好是 cuBLAS 内部 tiling 粒度（通常为 128 或 256）的倍数，且 K 维度不够大，cuBLAS 的启发式算法对所有 M 值选择了相同的 tiling 策略。K=10944 超过了一个阈值，触发了 M-dependent 的算法选择。

### 27.3 matmul_persistent 如何做到 batch-invariant

matmul_persistent 是一个 Triton persistent kernel，通过以下 5 个设计决策**消除所有非确定性源**：

#### 1. 固定 tiling —— 无 autotuning

```python
configs = {
    torch.bfloat16: {
        "BLOCK_SIZE_M": 128,   # 固定 128 行
        "BLOCK_SIZE_N": 128,   # 固定 128 列
        "BLOCK_SIZE_K": 64,    # 固定 64 个 K 元素
        "num_stages": 3,
        "num_warps": 8,
    },
}
```

所有维度都硬编码。没有 `@triton.autotune`，不根据 M/K/N 动态选择 tile size。

#### 2. 固定 grid —— grid size 不随 M 变化

```python
def grid(META):
    return (min(NUM_SMS, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)),)
```

`grid = min(NUM_SMS, num_tiles)`。对于 K=10944 的实际场景，`num_tiles` 始终远大于 `NUM_SMS`（132 个 SM），所以 grid 始终固定在 `NUM_SMS`，与 M 无关。

#### 3. 顺序 K-reduction —— 累加顺序固定

```python
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for ki in range(k_tiles):
    a = tl.load(a_ptrs, mask=...)
    b = tl.load(b_ptrs, mask=...)
    accumulator = tl.dot(a, b, accumulator)  # FMA: accum += a @ b
```

K 循环从 `ki=0` 到 `ki=k_tiles-1` 顺序遍历，每次累加一个 K_tile。顺序完全固定，不随 M 变化。累加器是 **float32**，精度足够。

#### 4. Persistent striding —— 确定性 tile 调度

```python
for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
    pid_m, pid_n = _compute_pid(tile_id, ...)
    # 计算 tile (pid_m, pid_n)
```

每个 thread block 处理 `[start_pid, num_tiles, step=NUM_SMS)` 的 tile 序列，调度顺序完全确定。

#### 5. 无算法选择 —— 无条件分支

matmul_persistent 内部没有 `if M > threshold then use_strategy_A else strategy_B` 这类条件。相同的代码路径对所有输入执行。

### 27.4 对比总结

| 维度 | cuBLAS (torch.mm) | matmul_persistent (Triton) |
|---|---|---|
| Tiling | 动态启发式，M/K/N-dependent | 硬编码，固定 BLOCK_SIZE_{M,N,K} |
| Grid size | cuBLAS 内部决定，M-dependent | 固定 NUM_SMS（对实际场景） |
| K-reduction | tile size 和 order 取决于算法选择 | 固定 BLOCK_SIZE_K=64，ki 顺序递增 |
| 算法选择 | 有 — 启发式 + workspace 状态 | **无** — 无条件分支 |
| 累加器精度 | TF32（10-bit mantissa） | float32（23-bit mantissa） |
| 批量下的确定性 | batch-variant（M 不同 → 算法不同 → 结果不同） | batch-invariant（通过设计保证） |

### 27.5 一句话总结

> cuBLAS 为了性能针对不同 M/K/N 选择不同的内部 tiling 策略，不同策略的累加顺序不同，浮点非结合性 + TF32 低精度导致结果差异。matmul_persistent 通过**固定 tiling + 固定 grid + 顺序 K-reduction + 无算法选择 + float32 累加器**消除了所有非确定性源。

---

## 28. nsys profile 分析：torch.mm vs matmul_persistent 的 CUDA kernel 层级对比

### 28.1 实验设置

```python
# like-useful/split-k.py
K, N = 10944, 4096
test_configs = [(6, 256)]  # M=6, bf16, TF32=ON

out_small   = torch.mm(A_rows, B_fixed)             # cuBLAS
out_small_mp = matmul_persistent(A_rows, B_fixed)    # Triton persistent
```

nsys profile 结果从 `temp/cuda_gpu_trace.csv` 中提取。

### 28.2 torch.mm 调用的 CUDA kernel（2 个）

#### Kernel 1: cuBLAS Split-K Matmul

```
Name:  nvjet_tst_64x8_64x16_2x1_v_bz_splitK_NNT
Grid:  (2, 64, 1)    — 128 thread blocks (2 split-K × 64 output tiles)
Block: (384, 1, 1)   — 384 threads per block
Time:  37,856 ns
```

名称解码：
- `nvjet` = NVIDIA JIT-compiled kernel（cuBLAS 运行时编译的 kernel）
- `tst` = Tensor Core 指令
- `64x8` / `64x16` = MMA（Matrix Multiply-Accumulate）Tile 尺寸，由 cuBLAS 启发式选择
- `2x1` = Split-K 因子为 2（K 维度被切分为 2 份并行计算）
- `v_bz` = kernel variant
- `splitK` = **Split-K 策略**：K 维度被拆分为多个 chunk，各 chunk 独立计算部分和，最后 reduce
- `NNT` = A Not Transposed, B Not Transposed（也可能第三位指输出类型）

**Split-K 策略详解**：

```
常规 matmul:
  K=10944 → 每个 thread block 顺序处理全量 K → 单次结果

Split-K matmul (此例 split=2):
  K=10944 → 拆为 K1=5472, K2=5472
  Thread block 0-63 算 K1 部分，block 64-127 算 K2 部分
  各 block 产生 partial sum → 需要 reduction kernel 合并
```

Grid(2, 64) 表示：2 个 split-K 分组 × 64 个输出 tile = 128 thread blocks，每个 block 只处理一半的 K。

#### Kernel 2: Split-K Reduction

```
Name:  cublasLt::splitKreduce_kernel
Grid:  (128, 1, 1)  — 128 thread blocks（与 splitK 的 block 数对应）
Block: (32, 16, 1)  — 512 threads per block
Time:  1,984 ns
```

将 128 个 partial sum（来自 2 split × 64 tiles）合并为最终结果。Grid 数量与 splitK matmul 的 block 总数一致。

### 28.3 matmul_persistent 调用的 CUDA kernel（1 个）

```
Name:  matmul_kernel_persistent
Grid:  (32, 1, 1)   — 32 thread blocks（min(NUM_SMS=132, num_tiles=32)）
Block: (256, 1, 1)  — 256 threads per block
Time:  246,401 ns
```

- `num_tiles = ceil(M/128) * ceil(N/128) = ceil(6/128) * ceil(4096/128) = 1 * 32 = 32`
- 32 < 132(SMs)，所以 grid = 32
- 每个 block 处理 1 个 tile（无 striding）
- 单 kernel 完成全部计算，无 split/reduce 分离

### 28.4 关键对比

| 维度 | torch.mm (cuBLAS) | matmul_persistent (Triton) |
|---|---|---|
| Kernel 数量 | **2**（splitK matmul + reduce） | **1**（persistent kernel） |
| MMA tile 选择 | cuBLAS 启发式动态选择（`64x8`/`64x16`） | 硬编码 `BLOCK_SIZE_K=64`（固定） |
| K 维度策略 | **Split-K**（拆 2 份并行 → reduce 合并） | 顺序遍历全量 K（无 split） |
| Grid 决定因素 | cuBLAS 内部算法决定（128 blocks） | `min(NUM_SMS, ceil(M/128)*ceil(N/128))` |
| Kernel 命名 | `nvjet_tst_..._splitK_...`（JIT 编译，名称编码策略） | `matmul_kernel_persistent`（Triton 编译） |
| 确定性 | 否 — 启发式选择可能因 M 不同而选不同 split-K 因子或 tile 尺寸 | **是** — 固定策略，无动态选择 |

### 28.5 Split-K 是 batch-dependence 的直接证据

cuBLAS 对这个 (M=6, K=10944, N=4096) 问题选择了 **split-K 因子 2**。当 M 变为 64 或更大时，cuBLAS 的启发式可能选择完全不同的 split-K 因子（如 1，即不用 split-K）或不同的 MMA tile 尺寸。

不同的 split-K 因子意味着：
- 不同的 K 维度分组 → 不同的累加顺序
- 不同的 partial sum 数量 → 不同的 reduction tree 结构
- 不同的 rounding 误差累积路径

**这直接解释了为什么 torch.mm 在 K=10944 时产生 batch-variance 而 K=2048 时不产生**：K=10944 时 cuBLAS 启用了 M-dependent 的 split-K 策略，而 K=2048 时没有（可能所有 M 都使用统一的 non-split 路径）。

### 28.6 一句话总结

> nsys profile 揭示了 cuBLAS 的 `torch.mm` 使用了 **split-K 策略**（`nvjet_tst_..._splitK_NNT` + `splitKreduce_kernel`，共 2 个 kernel），其 split 因子和 MMA tile 尺寸由启发式动态选择，不同 M 值可能导致不同选择 → batch-variance。而 `matmul_persistent` 只用 1 个固定 kernel，K 维度顺序遍历，无 split/reduce 分离 → batch-invariant。

---

## 29. 小 M vs 大 M：cuBLAS 和 matmul_persistent 的行为对比

### 29.1 实验设置

```python
# like-useful/split-k.py  — 同一脚本中连续执行 4 个 matmul
K, N = 10944, 4096

A_rows = rbf((6, K))         # M=6
B_fixed = rbf((K, N))
A_large = cat([A_rows, rbf((250, K))])  # M=256

out_small       = torch.mm(A_rows, B_fixed)       # ① cuBLAS M=6
out_small_mp    = matmul_persistent(A_rows, B_fixed)  # ② Triton M=6
out_large_full  = torch.mm(A_large, B_fixed)       # ③ cuBLAS M=256
out_large_full_mp = matmul_persistent(A_large, B_fixed)  # ④ Triton M=256
```

从同一个 `cuda_gpu_trace.csv` 中提取 4 个 matmul 的 kernel：

### 29.2 cuBLAS：M=6 vs M=256 策略完全不同

| | M=6（小 batch） | M=256（大 batch） |
|---|---|---|
| **Kernel 名称** | `nvjet_tst_64x8_64x16_2x1_v_bz_splitK_NNT` | `nvjet_tst_128x64_64x8_1x2_h_bz_NNT` |
| **Grid** | (2, 64, 1) — 128 blocks | (2, 64, 1) — 128 blocks |
| **Block** | (384, 1, 1) | (384, 1, 1) |
| **MMA tile** | 64×8, 64×16（小 tile） | 128×64, 64×8（大 M tile + 小 K tile） |
| **Split-K** | **是**（名称含 `splitK`） | **否**（名称不含 `splitK`） |
| **Split-K 因子** | 2（Grid.X = 2） | —（不使用 split-K） |
| **Reduce kernel** | `splitKreduce_kernel` Grid=(128), Block=(32,16) | **无** — 不需要 reduce |
| **kernel variant** | `v_bz` | `h_bz` |
| **总耗时** | 39,009 + 1,984 = 40,993 ns | 45,216 ns（单 kernel） |
| **K 维度处理** | 拆 2 份并行，reduce 合并 | 完整 K 直接处理 |

#### 策略选择逻辑推演

cuBLAS 的启发式根据 M 的大小做出不同决策：

```
M=6（极小 batch）:
  K=10944 很大，M=6 极小
  → 每个 thread block 的工作沿着 M 方向太少（6 行）
  → 启用 split-K：把 K 拆成 2 份，增加并行度
  → 用较小的 MMA tile（64×8）适配较少的 M 方向工作量
  → variant: v_bz（"vertical" 优化？）

M=256（中等 batch）:
  K=10944，M=256
  → M 方向有足够工作量（256 行），无需 split-K
  → 直接用更大的 MMA tile（128×64）沿 M 方向
  → K 方向顺序遍历，单 kernel 完成
  → variant: h_bz（"horizontal" 优化？）
```

**关键结论**：cuBLAS 对**同一个 K=10944, N=4096 问题**，仅因为 M 从 6 变为 256，就选择了完全不同的策略——不同的 MMA tile 尺寸、不同的 kernel variant、是否启用 split-K。这直接导致了数值上的 batch-dependence。

### 29.3 matmul_persistent：M=6 vs M=256 策略一致

| | M=6（小 batch） | M=256（大 batch） |
|---|---|---|
| **Kernel 名称** | `matmul_kernel_persistent` | `matmul_kernel_persistent` |
| **Grid** | (32, 1, 1) | (64, 1, 1) |
| **Block** | (256, 1, 1) | (256, 1, 1) |
| **Registers** | 124 | 124 |
| **Static SMem** | 0.000 MB | 0.000 MB |
| **Dynamic SMem** | 0.082 MB | 0.082 MB |
| **BLOCK_SIZE_*** | 128/128/64（硬编码） | 128/128/64（硬编码，完全相同） |
| **K-reduction** | `for ki in range(k_tiles)` 顺序 | `for ki in range(k_tiles)` 顺序（完全相同） |
| **Accumulator** | float32 | float32 |
| **策略变化** | **无** | **无** |
| **总耗时** | 247,937 ns | 203,233 ns |

Grid 从 32 变为 64 的原因：

```
M=6:  num_tiles = ceil(6/128)   * ceil(4096/128) = 1 * 32 = 32
      grid = min(132, 32) = 32

M=256: num_tiles = ceil(256/128) * ceil(4096/128) = 2 * 32 = 64
       grid = min(132, 64) = 64
```

Grid 大小完全由确定性公式决定，不是启发式选择。Grid 增大只意味着更多 thread block 并行工作，**不影响每个 tile 内部的 K-reduction 顺序**。

### 29.4 对比总结图

```
cuBLAS (torch.mm):
  M=6  ──→ splitK=2, MMA=64x8, variant=v_bz, reduce kernel ──→ 结果 A
  M=256 ──→ splitK=NONE, MMA=128x64, variant=h_bz, 无 reduce ──→ 结果 B
  ↑                                                            ↑
  完全不同的策略 → 不同的累加顺序 → 结果不同 (batch-variant)

matmul_persistent:
  M=6  ──→ grid=32, BLOCK=128x128x64, 顺序 K-reduction ──→ 结果 A
  M=256 ──→ grid=64, BLOCK=128x128x64, 顺序 K-reduction ──→ 结果 A
  ↑                              ↑
  相同策略，仅有 grid 不同 → 相同累加顺序 → 结果相同 (batch-invariant)
```

### 29.5 一句话总结

> cuBLAS 对 M=6 和 M=256 使用了**完全不同的 kernel**（`splitK` vs 非 splitK，`64x8` vs `128x64` MMA tile，`v_bz` vs `h_bz` variant），策略变化导致不同的累加路径 → batch-dependence。matmul_persistent 只用**同一个 kernel**，Grid 变化（32→64）仅改变并行度，内部 K-reduction 顺序不变 → batch-invariant。

---

## 30. lm-eval 跑 GSM8K 时保存每道题、标准答案、模型输出和错题

当前环境里的 `lm-eval` 是 `lm_eval` 0.4.9.2。要让输出结果包含每个 GSM8K 样本，需要在原来的评测命令后面加：

```bash
--log_samples --output_path <输出目录>
```

也就是说，把原来的命令：

```bash
/data/like/miniconda3/envs/simo_sglang/bin/lm-eval \
  ...你的原有 model/model_args 参数... \
  --tasks gsm8k
```

改成：

```bash
OUT=like-useful/lm_eval_gsm8k_$(date +%Y%m%d_%H%M%S)

/data/like/miniconda3/envs/simo_sglang/bin/lm-eval \
  ...你的原有 model/model_args 参数... \
  --tasks gsm8k \
  --log_samples \
  --output_path "$OUT"
```

新版 CLI 也可以显式写 `run`，等价：

```bash
/data/like/miniconda3/envs/simo_sglang/bin/lm-eval run \
  ...你的原有 model/model_args 参数... \
  --tasks gsm8k \
  --log_samples \
  --output_path "$OUT"
```

注意：`--log_samples` 必须配合 `--output_path`。只用 `--write_out` 不够，`--write_out` 主要打印前几个 prompt，不会保存完整每题结果。也不要用 `--predict_only` 来分析错题，因为它会跳过 metric 计算；分析“哪题错了”需要保留 `exact_match`。

评测结束后，samples 文件通常在：

```bash
find "$OUT" -name 'samples_gsm8k_*.jsonl' -print
```

如果 `--output_path` 是目录，lm-eval 会在目录下再建一个模型名子目录，里面会有：

```text
results_<timestamp>.json
samples_gsm8k_<timestamp>.jsonl
```

每行 JSONL 是一个样本记录，关键字段是：

```text
doc_id              GSM8K test 集样本 id
doc.question        原始问题
doc.answer          GSM8K 标准解答，末尾通常有 #### final_answer
target              lm-eval 的 target；对 gsm8k 默认基本就是标准 answer 字符串
resps               模型原始生成
filtered_resps      经过 regex 抽取后的答案
filter              使用哪个抽取规则
exact_match         这个样本在该 filter 下是否答对，通常 1/0 或 true/false
```

GSM8K 默认配置有两个 filter：

```text
strict-match       要求模型输出里有类似 #### 42 的格式
flexible-extract   更宽松，从输出里抽取数字
```

因此同一道题通常会在 `samples_gsm8k_*.jsonl` 里出现两条记录：一条 `filter=="strict-match"`，一条 `filter=="flexible-extract"`。你要按最终结果表里关注的那一列选择对应 filter。一般排查模型实际答错了哪些题，`flexible-extract` 更直观；如果你关心严格格式是否符合，就看 `strict-match`。

列出所有题目、标准答案、模型原始输出和抽取答案：

```bash
SAMPLE=$(find "$OUT" -name 'samples_gsm8k_*.jsonl' | head -n 1)

jq -r '
  select(.filter == "flexible-extract") |
  "doc_id=\(.doc_id)\nQ: \(.doc.question)\nGOLD: \(.doc.answer)\nMODEL_RAW: \(.resps[0][0])\nEXTRACTED: \(.filtered_resps[0])\nexact_match=\(.exact_match)\n---"
' "$SAMPLE" > "$OUT/gsm8k_all_samples.txt"
```

只筛出错题：

```bash
jq -r '
  select(.filter == "flexible-extract" and (.exact_match == 0 or .exact_match == false)) |
  "doc_id=\(.doc_id)\nQ: \(.doc.question)\nGOLD: \(.doc.answer)\nMODEL_RAW: \(.resps[0][0])\nEXTRACTED: \(.filtered_resps[0])\n---"
' "$SAMPLE" > "$OUT/gsm8k_wrong_samples.txt"
```

如果你要分析 `strict-match` 下的错题，把上面命令里的：

```jq
select(.filter == "flexible-extract" ...)
```

改成：

```jq
select(.filter == "strict-match" ...)
```

如果只想抽取标准答案的最终数字，可以从 `doc.answer` 里的 `####` 后面取：

```bash
jq -r '
  select(.filter == "flexible-extract" and (.exact_match == 0 or .exact_match == false)) |
  (.doc.answer | capture("#### (?<gold>.*)$").gold) as $gold |
  "doc_id=\(.doc_id)\nQ: \(.doc.question)\nGOLD_FINAL: \($gold)\nMODEL_RAW: \(.resps[0][0])\nEXTRACTED: \(.filtered_resps[0])\n---"
' "$SAMPLE" > "$OUT/gsm8k_wrong_final_answer.txt"
```

如果你拿到错题 `doc_id` 后想单独重跑几道题，可以用 `--samples`，例如只跑第 12、345、678 题：

```bash
/data/like/miniconda3/envs/simo_sglang/bin/lm-eval \
  ...你的原有 model/model_args 参数... \
  --tasks gsm8k \
  --samples '{"gsm8k":[12,345,678]}' \
  --log_samples \
  --output_path "$OUT/rerun_selected"
```

`--samples` 和 `--limit` 不能同时用。

---

## 31. 2026-07-11 GSM8K 复测日志检查及参考分数对比

输入日志：

```text
temp/llm_eval_online_quant.sh.MAX_RUNNING_REQUESTS_128_CUDA_GRAPH_MAX_BS_128_ADD_BOS_TOKEN_true__TASKS_gsm8k__CUDA_VISIBLE_DEVICES_7.log.2026_07_11___11_36_29
```

### 31.1 日志完整性、OOM 和 crash

- 评测已经结束，日志不再写入，也没有残留的评测或 SGLang 服务进程。
- 没有 `OutOfMemoryError`、`CUDA out of memory` 或其他 OOM。
- 没有 SGLang scheduler、server 或 `lm-eval` 的运行期 crash、CUDA error、SIGKILL、segfault 或结果缺失。
- 脚本预期运行 `2 * (1 个无量化 + 13 个权重量化 + 7 个 KV cache 量化) = 42` 项；日志中 42 项全部有完整的 GSM8K 结果，覆盖 42/42，没有缺失或重复。
- 每项成功输出结果并关闭服务后，都出现一次 Python `resource_tracker`/loky 清理期 `KeyError('/loky-...')`，共 42 次。这是退出清理辅助进程的异常/资源泄漏告警，不是模型推理或评测 crash，不影响已经输出的分数。
- 脚本只有 `set -e`，没有 `set -o pipefail`；由于命令使用 `lm-eval ... | tee ...`，未来若 `lm-eval` 失败，退出码可能被 `tee` 掩盖。因此提取聚合日志时仍应像本次一样校验预期结果数量。

### 31.2 对比口径

- 使用 lm-eval 表格中的 `flexible-extract / exact_match`，不使用下一行 `strict-match`。
- 日志原始值乘以 100，并保留两位小数后，与 `tests/sglang_simo/references_accuracy/gsm8k.yaml` 比较。
- `差值 = 本次分数 - 参考分数`，单位是百分点（pp）。
- 为避免把单次评测约 1.1-1.4 pp 的 stderr 直接当成明显回归，这里将 `|差值| >= 3.00 pp` 定义为“较大差别”；`2.00 <= |差值| < 3.00 pp` 标记为“需关注”；其余标记为“否”。
- 权重量化名称对应 `quant_config_<quant_algo>.json`。KV cache 的 `fp8_per_group_64`、`int8_per_group_64` 分别对应日志中的 `quant_config_kvquant_fp8_per_group.json`、`quant_config_kvquant_int8_per_group.json`。

### 31.3 Llama-3.1-8B-Instruct

| 类型 | 配置 | 参考 | 本次 | 差值(pp) | 较大差别 |
|---|---|---:|---:|---:|---|
| 无量化 | baseline | 78.01 | 77.63 | -0.38 | 否 |
| 权重 | w8a8_fp8_per_block | 77.48 | 76.95 | -0.53 | 否 |
| 权重 | w4a16_int4_per_group | 72.71 | 72.93 | +0.22 | 否 |
| 权重 | w8a8_int8_per_block | 77.79 | 77.26 | -0.53 | 否 |
| 权重 | w8a8_fp8_per_channel | 76.35 | 77.71 | +1.36 | 否 |
| 权重 | w8a8_int8_per_channel | 77.94 | 75.59 | -2.35 | 需关注 |
| 权重 | w8a8_mxint | 78.17 | 77.48 | -0.69 | 否 |
| 权重 | w8a8_mxfp | 76.95 | 77.03 | +0.08 | 否 |
| 权重 | w6a6_mxfp | 77.63 | 76.35 | -1.28 | 否 |
| 权重 | w4a4_mxfp | 47.92 | 47.61 | -0.31 | 否 |
| 权重 | w4a16_nvfp4_per_group | 72.48 | 73.46 | +0.98 | 否 |
| 权重 | w4a16_nvfp4_per_group_4_over_6 | 73.24 | 74.00 | +0.76 | 否 |
| 权重 | w4a4_nvfp | 69.37 | 69.07 | -0.30 | 否 |
| 权重 | w4a4_nvfp_4_over_6 | 70.36 | 70.13 | -0.23 | 否 |
| KV cache | mxfp8 | 78.92 | 76.72 | -2.20 | 需关注 |
| KV cache | mxfp4 | 68.61 | 69.90 | +1.29 | 否 |
| KV cache | mxfp6 | 77.94 | 77.79 | -0.15 | 否 |
| KV cache | mxint8 | 78.32 | 77.94 | -0.38 | 否 |
| KV cache | fp8_per_group_64 | 77.33 | 76.95 | -0.38 | 否 |
| KV cache | int8_per_group_64 | 78.01 | 77.10 | -0.91 | 否 |
| KV cache | nvfp4 | 75.13 | 76.57 | +1.44 | 否 |

Llama 没有达到 `3.00 pp` 的“较大差别”项。需要关注两项下降：权重 `w8a8_int8_per_channel` 为 `-2.35 pp`，KV cache `mxfp8` 为 `-2.20 pp`。其余 19 项均小于 2 pp，Llama 全部 21 项的平均绝对差约为 `0.80 pp`。

### 31.4 DeepSeek-V2-Lite-Chat-16B_A2.4B

| 类型 | 配置 | 参考 | 本次 | 差值(pp) | 较大差别 |
|---|---|---:|---:|---:|---|
| 无量化 | baseline | 67.10 | 66.03 | -1.07 | 否 |
| 权重 | w8a8_fp8_per_block | 64.06 | 64.97 | +0.91 | 否 |
| 权重 | w4a16_int4_per_group | 59.44 | 58.68 | -0.76 | 否 |
| 权重 | w8a8_int8_per_block | 66.49 | 66.64 | +0.15 | 否 |
| 权重 | w8a8_fp8_per_channel | 58.15 | 65.50 | +7.35 | 较大 |
| 权重 | w8a8_int8_per_channel | 63.61 | 64.06 | +0.45 | 否 |
| 权重 | w8a8_mxint | 66.19 | 65.28 | -0.91 | 否 |
| 权重 | w8a8_mxfp | 63.91 | 64.90 | +0.99 | 否 |
| 权重 | w6a6_mxfp | 63.99 | 64.37 | +0.38 | 否 |
| 权重 | w4a4_mxfp | 35.48 | 38.51 | +3.03 | 较大 |
| 权重 | w4a16_nvfp4_per_group | 62.47 | 63.84 | +1.37 | 否 |
| 权重 | w4a16_nvfp4_per_group_4_over_6 | 61.94 | 61.71 | -0.23 | 否 |
| 权重 | w4a4_nvfp | 56.79 | 56.18 | -0.61 | 否 |
| 权重 | w4a4_nvfp_4_over_6 | 56.63 | 60.20 | +3.57 | 较大 |
| KV cache | mxfp8 | 66.79 | 66.03 | -0.76 | 否 |
| KV cache | mxfp4 | 29.95 | 31.39 | +1.44 | 否 |
| KV cache | mxfp6 | 65.81 | 64.37 | -1.44 | 否 |
| KV cache | mxint8 | 66.19 | 66.03 | -0.16 | 否 |
| KV cache | fp8_per_group_64 | 67.10 | 66.03 | -1.07 | 否 |
| KV cache | int8_per_group_64 | 66.03 | 66.26 | +0.23 | 否 |
| KV cache | nvfp4 | 48.29 | 47.08 | -1.21 | 否 |

DeepSeek 有 3 项达到“较大差别”，并且都是分数上升：权重 `w8a8_fp8_per_channel` 为 `+7.35 pp`，`w4a4_nvfp_4_over_6` 为 `+3.57 pp`，`w4a4_mxfp` 为 `+3.03 pp`。其余 18 项均小于 2 pp；DeepSeek 全部 21 项的平均绝对差约为 `1.34 pp`。

### 31.5 总结

42 项的总体平均绝对差约为 `1.07 pp`。按上述 `3.00 pp` 阈值，只有 3/42 项存在较大差别，均为 DeepSeek 权重量化且本次分数更高；另有 2/42 项需要关注，都是 Llama 的下降。其余 37/42 项没有较大差别。

本次日志提取并按参考文件顺序重排后的完整分数已写入：

```text
temp/gsm8k-fix.yaml
```

---

## 32. 2026-07-11 MMLU 复测日志检查及参考分数对比

输入日志：

```text
temp/llm_eval_online_quant.sh.MAX_RUNNING_REQUESTS_128_CUDA_GRAPH_MAX_BS_128_ADD_BOS_TOKEN_true__TASKS_mmlu__CUDA_VISIBLE_DEVICES_6.log.2026_07_11___11_33_06
```

### 32.1 日志完整性、OOM 和 crash

- 评测已经结束，日志不再写入，也没有残留的评测或 SGLang 服务进程。
- 没有 `OutOfMemoryError`、`CUDA out of memory` 或其他 OOM。
- 没有 SGLang scheduler、server 或 `lm-eval` 的运行期 crash、CUDA/NCCL error、SIGKILL、segfault 或结果缺失。
- 预期的 42 项评测全部完成：42 条 `lm-eval` 启动记录、42 个结果块、42 次 loglikelihood 100% 完成，没有缺失或重复配置。
- 每轮结果包含 `Tasks` 表和 `Groups` 表，两张表各打印一次相同的聚合 `mmlu` 分数，因此日志有 84 条聚合行。提取时每轮只计一次，并验证同轮两个值相等。
- 与 GSM8K 日志相同，每项成功出分并关闭服务后都有一次 `resource_tracker`/loky 清理期 `KeyError('/loky-...')`，共 42 次。这是退出清理辅助进程的异常/潜在资源泄漏告警，不是模型评测 crash，不影响分数。
- 本次只使用文件名结尾为 `2026_07_11___11_33_06` 的新日志；没有混用目录中存在 OOM 和旧 `page_size` 错误的 `2026_07_10___21_39_32` 旧日志。

### 32.2 对比口径

- 从每轮 `Tasks` 表中取第一列去空白后精确等于 `mmlu`、Metric 为 `acc` 的聚合行，不取四个 category 或具体学科行，也不自行重新平均。
- 日志原始 Value 乘以 100，并保留两位小数后，与 `tests/sglang_simo/references_accuracy/mmlu.yaml` 比较。
- `差值 = 本次分数 - 参考分数`，单位是百分点（pp）。
- 本次 MMLU 汇总分数的 stderr 约为 `0.37-0.41 pp`。这里将 `|差值| >= 1.00 pp` 定义为“较大差别”，`0.50 <= |差值| < 1.00 pp` 标记为“需关注”，其余标记为“否”。
- 权重量化和 KV cache 配置名称映射规则与上一节 GSM8K 相同。

### 32.3 Llama-3.1-8B-Instruct

| 类型 | 配置 | 参考 | 本次 | 差值(pp) | 较大差别 |
|---|---|---:|---:|---:|---|
| 无量化 | baseline | 68.15 | 68.47 | +0.32 | 否 |
| 权重 | w8a8_fp8_per_block | 67.81 | 68.18 | +0.37 | 否 |
| 权重 | w4a16_int4_per_group | 65.96 | 66.22 | +0.26 | 否 |
| 权重 | w8a8_int8_per_block | 67.87 | 68.17 | +0.30 | 否 |
| 权重 | w8a8_fp8_per_channel | 67.50 | 67.89 | +0.39 | 否 |
| 权重 | w8a8_int8_per_channel | 67.44 | 67.73 | +0.29 | 否 |
| 权重 | w8a8_mxint | 68.07 | 68.20 | +0.13 | 否 |
| 权重 | w8a8_mxfp | 67.51 | 67.67 | +0.16 | 否 |
| 权重 | w6a6_mxfp | 67.54 | 68.07 | +0.53 | 需关注 |
| 权重 | w4a4_mxfp | 60.54 | 57.99 | -2.55 | 较大 |
| 权重 | w4a16_nvfp4_per_group | 65.99 | 66.10 | +0.11 | 否 |
| 权重 | w4a16_nvfp4_per_group_4_over_6 | 66.01 | 66.53 | +0.52 | 需关注 |
| 权重 | w4a4_nvfp | 64.21 | 64.13 | -0.08 | 否 |
| 权重 | w4a4_nvfp_4_over_6 | 64.59 | 64.45 | -0.14 | 否 |
| KV cache | mxfp8 | 68.09 | 68.27 | +0.18 | 否 |
| KV cache | mxfp4 | 68.09 | 68.27 | +0.18 | 否 |
| KV cache | mxfp6 | 68.09 | 68.27 | +0.18 | 否 |
| KV cache | mxint8 | 68.09 | 68.27 | +0.18 | 否 |
| KV cache | fp8_per_group_64 | 68.09 | 68.27 | +0.18 | 否 |
| KV cache | int8_per_group_64 | 68.09 | 68.27 | +0.18 | 否 |
| KV cache | nvfp4 | 68.09 | 68.27 | +0.18 | 否 |

Llama 有 1 项较大差别：权重 `w4a4_mxfp` 从 `60.54` 降至 `57.99`，差值 `-2.55 pp`。另有两项小幅上升需要关注：`w6a6_mxfp` 为 `+0.53 pp`，`w4a16_nvfp4_per_group_4_over_6` 为 `+0.52 pp`。全部 21 项平均绝对差约为 `0.35 pp`。

### 32.4 DeepSeek-V2-Lite-Chat-16B_A2.4B

| 类型 | 配置 | 参考 | 本次 | 差值(pp) | 较大差别 |
|---|---|---:|---:|---:|---|
| 无量化 | baseline | 56.78 | 56.72 | -0.06 | 否 |
| 权重 | w8a8_fp8_per_block | 56.87 | 56.52 | -0.35 | 否 |
| 权重 | w4a16_int4_per_group | 55.18 | 54.62 | -0.56 | 需关注 |
| 权重 | w8a8_int8_per_block | 56.71 | 56.55 | -0.16 | 否 |
| 权重 | w8a8_fp8_per_channel | 56.38 | 55.94 | -0.44 | 否 |
| 权重 | w8a8_int8_per_channel | 56.05 | 55.88 | -0.17 | 否 |
| 权重 | w8a8_mxint | 56.73 | 56.56 | -0.17 | 否 |
| 权重 | w8a8_mxfp | 56.85 | 55.86 | -0.99 | 需关注 |
| 权重 | w6a6_mxfp | 56.64 | 55.93 | -0.71 | 需关注 |
| 权重 | w4a4_mxfp | 49.78 | 48.66 | -1.12 | 较大 |
| 权重 | w4a16_nvfp4_per_group | 54.52 | 54.03 | -0.49 | 否 |
| 权重 | w4a16_nvfp4_per_group_4_over_6 | 54.87 | 54.69 | -0.18 | 否 |
| 权重 | w4a4_nvfp | 53.35 | 53.06 | -0.29 | 否 |
| 权重 | w4a4_nvfp_4_over_6 | 53.57 | 52.98 | -0.59 | 需关注 |
| KV cache | mxfp8 | 56.60 | 56.76 | +0.16 | 否 |
| KV cache | mxfp4 | 56.60 | 56.76 | +0.16 | 否 |
| KV cache | mxfp6 | 56.60 | 56.76 | +0.16 | 否 |
| KV cache | mxint8 | 56.60 | 56.76 | +0.16 | 否 |
| KV cache | fp8_per_group_64 | 56.60 | 56.76 | +0.16 | 否 |
| KV cache | int8_per_group_64 | 56.60 | 56.76 | +0.16 | 否 |
| KV cache | nvfp4 | 56.60 | 56.76 | +0.16 | 否 |

DeepSeek 有 1 项较大差别：权重 `w4a4_mxfp` 从 `49.78` 降至 `48.66`，差值 `-1.12 pp`。另有四项下降需要关注：`w8a8_mxfp` 为 `-0.99 pp`、`w6a6_mxfp` 为 `-0.71 pp`、`w4a4_nvfp_4_over_6` 为 `-0.59 pp`、`w4a16_int4_per_group` 为 `-0.56 pp`。全部 21 项平均绝对差约为 `0.35 pp`。

### 32.5 总结

42 项总体平均绝对差约为 `0.35 pp`。按 `1.00 pp` 阈值，有 2/42 项存在较大差别，都是权重 `w4a4_mxfp` 的下降：Llama `-2.55 pp`，DeepSeek `-1.12 pp`。另有 6/42 项处于 `0.50-0.99 pp` 的需关注区间，其余 34/42 项差异小于 `0.50 pp`。所有 KV cache 配置相对参考文件都只变化 `+0.16` 或 `+0.18 pp`，没有较大差别。

本次日志提取并按参考文件顺序重排后的完整分数已写入：

```text
temp/mmlu-fix.yaml
```

---

## 33. MMLU 与 GSM8K 结束时间差异分析

### 33.1 结论

这次运行中，MMLU 的评测工作量确实明显大于 GSM8K，但差异不只是 GPU 计算量：MMLU 还需要为 57 个子任务反复构造上下文、分词并生成大量 `loglikelihood` 请求。两个脚本都遍历 2 个模型和 21 种配置，共完成 42 轮评测，因此配置轮数相同。

按日志文件名中的启动时间和文件最后修改时间计算：

| 任务 | 启动时间 | 日志结束时间 | 总运行时长 |
|---|---|---|---:|
| MMLU（GPU 6） | 11:33:06 | 19:13:50.879 | 7:40:44.9 |
| GSM8K（GPU 7） | 11:36:29 | 14:34:24.407 | 2:57:55.4 |

- MMLU 的总运行时长约为 GSM8K 的 `2.59` 倍，多运行约 `4:42:49.5`。
- 因为 MMLU 早启动约 `3:23`，所以日志文件的实际结束时刻比 GSM8K 晚约 `4:39:26.5`。
- 两份日志都完整跑完 42 轮，没有 OOM、crash、重试或缺失结果，因此不是失败重试把 MMLU 拖慢。

### 33.2 两个任务的实际工作量不同

当前安装的 `lm-eval` 任务定义和日志显示：

| 任务 | 评测方式 | 每轮数据规模 | 每轮 lm-eval 请求数 |
|---|---|---:|---:|
| MMLU | 57 个子任务，0-shot，多选题；每题 4 个选项分别做 `loglikelihood` | 14,042 题 | 56,168 |
| GSM8K | 单任务，5-shot，使用 `generate_until` 生成推理和答案 | 1,319 题 | 1,319 |

MMLU 每轮的请求条数是 GSM8K 的约 `42.6` 倍。但不能据此推断运行时间也应是 42.6 倍：

- MMLU 的单个请求主要是短 continuation 的 likelihood 计算，可以大量合批，工作以 prompt/prefill 和 logprob 计算为主。
- GSM8K 的请求数少，但每条都有较长的 5-shot prompt，并且需要逐 token 自回归生成推理过程；decode 串行依赖更强，单条请求明显更贵。
- SGLang scheduler 在每轮开始时报告的待处理输入 token，MMLU 约为 `6.8M-7.2M`，GSM8K 约为 `1.15M-1.34M`，中位比例约 `5.6` 倍；GSM8K 此后还要生成输出 token。

因此，更准确的说法是：MMLU 的题目数、输入 token 和请求准备量大很多；GSM8K 单请求的生成成本更高，所以总耗时比例被压缩到约 2.6 倍，而不是请求条数对应的 42.6 倍。

### 33.3 42 轮评测的耗时拆分

以下按每轮日志标记累计。`Tree cache initialized` 作为引擎初始化完成的稳定代理；“请求和结果”包含请求执行、指标聚合以及结果写出。

| 阶段（42 轮累计） | MMLU | GSM8K | MMLU 相对增加 |
|---|---:|---:|---:|
| 模型/服务初始化及 CUDA Graph | 0:35:19 | 0:36:10 | -0:00:51 |
| 任务、上下文和请求准备 | 1:28:30 | 0:10:48 | +1:17:42 |
| 请求执行、聚合和结果写出 | 5:02:30 | 1:45:05 | +3:17:25 |
| 41 次轮间清理和重启间隔 | 0:33:46 | 0:25:08 | +0:08:38 |
| 首轮开始到末轮结果 | 7:40:05 | 2:57:11 | +4:42:54 |

分解结果表明：

- 两者模型加载、服务初始化和 CUDA Graph 总时间几乎相同，MMLU 甚至少 51 秒，所以模型启动不是主因。
- MMLU 在任务/上下文准备上多用约 `1:17:42`。MMLU 每轮需要处理 57 个子任务和 56,168 个请求，而 GSM8K 每轮只有一个任务和 1,319 个请求。
- MMLU 在请求执行到结果写出阶段多用约 `3:17:25`，这是总差异中最大的一部分。只看 tqdm 所覆盖的实际请求阶段，MMLU 累计约 `4:17:47`，GSM8K 约 `1:18:07`，比例约 `3.30` 倍。
- 每轮从任务启动到出结果，MMLU 的中位数为 `8:45`、均值为 `10:09`；GSM8K 的中位数为 `3:16.5`、均值为 `3:37`。

### 33.4 GPU 6 和 GPU 7 不是主要原因

当前 GPU 6 和 GPU 7 都是 `NVIDIA H100 80GB HBM3`，两份日志也都使用 SM90。对应 42 轮的 SGLang 服务参数和模型配置一致，服务初始化累计时间也分别只有 `35:19` 和 `36:10`，没有证据表明 GPU 6 明显慢于 GPU 7。

两个作业在 GSM8K 结束前并发运行，可能存在少量 CPU、磁盘或系统调度干扰，但这不足以解释近 4 小时 43 分钟的运行时长差异。差异随任务准备量和请求执行量稳定出现，主要原因仍然是任务工作负载不同。

不同量化 kernel 的速度差异也会放大总时间。例如 DeepSeek 的 `w4a16_int4_per_group` 是两个任务中共同最慢的一轮：MMLU 请求进度耗时约 `28:43`，GSM8K 约 `9:28`。这说明是该配置在不同任务负载下都较慢，而不是某个 GPU 偶发卡死。

综上，这次 MMLU 比 GSM8K 慢很多是正常的工作量差异：约 1 小时 18 分钟来自额外的任务/请求准备，约 3 小时 17 分钟来自额外的请求计算和结果处理。它不是 OOM、crash、重试或 GPU id 不同造成的。不过 `2.59` 倍是本次模型、量化配置和 lm-eval 参数下的实测比例，不应当视为所有 MMLU/GSM8K 运行的固定比例。

---

## 34. `self._kv_buffer_descs = self._build_kv_buffer_descs()` 的作用

### 34.1 上下文

这行代码出现在 `SIMOMHATokenToKVPool._create_buffers()` 中（`simo/extensions/sglang_simo/mem_cache/memory_pool.py:209-210`）：

```python
# Override store_dtype to uint8 since buffers are quantized
self.store_dtype = torch.uint8
if hasattr(self, "_build_kv_buffer_descs"):
    self._kv_buffer_descs = self._build_kv_buffer_descs()
```

它位于 SIMO 子类重写的 `_create_buffers` 末尾，在 `self.store_dtype` 被设为 `torch.uint8` **之后**，在构造 `k_data_ptrs` / `v_data_ptrs` **之前**。

### 34.2 `_build_kv_buffer_descs()` 做了什么

该方法定义在 SGLang 的 `KVCache` 基类中（`sglang/srt/mem_cache/memory_pool.py:1593`），构建一个 `KvBufferDesc` 对象的列表，覆盖所有层的 k buffer 和 v buffer，顺序为 `k0, k1, ..., k(L-1), v0, v1, ..., v(L-1)`。

核心逻辑：

```python
def _build_kv_buffer_descs(self):
    itemsize = self.store_dtype.itemsize           # ← 对 SIMO 是 torch.uint8 → 1 byte
    if getattr(self, "k_buffer", None) and getattr(self, "v_buffer", None):
        k_shape = tuple(self.k_buffer[0].shape)    # ← 从实际张量获取形状
        v_shape = tuple(self.v_buffer[0].shape)
    else:
        k_shape, v_shape = self._kv_buffer_shapes()  # ← 回退到参数推导

    num_slots = self.size + self.page_size
    tokens_per_row = (
        self.page_size if k_shape[0] * self.page_size == num_slots else 1
    )
    descs = []
    for prefix, shape in (("k", k_shape), ("v", v_shape)):
        row_bytes = int(np.prod(shape[1:])) * itemsize
        for layer in range(self.layer_num):
            descs.append(KvBufferDesc(
                f"{prefix}{layer}", shape,
                row_bytes=row_bytes,
                tokens_per_row=tokens_per_row,
            ))
    return descs
```

每个 `KvBufferDesc` 是一个轻量描述符，包含 4 个字段：

| 字段 | 含义 | SIMO 场景下的值（以 k buffer 为例） |
|---|---|---|
| `name` | 描述符名称 | `"k0"`, `"k1"`, ... |
| `shape` | 张量形状 | `(size + page_size, head_num, k_combined_head_size)` |
| `row_bytes` | 每行（首维度的一行）的字节数 | `head_num × k_combined_head_size × 1`（uint8） |
| `tokens_per_row` | 每行包含的 token 数 | NHD 布局下为 `1`；HND 布局下为 `page_size` |

### 34.3 为什么 SIMO 需要在 `store_dtype = torch.uint8` 之后重新调用

SIMO 子类在 `_create_buffers` 中做了两件改变缓冲区布局的事：

1. **分配 uint8 缓冲区**：k_buffer 和 v_buffer 是 `torch.uint8` 张量，形状为 `[size+page_size, head_num, k_combined_head_size]`，而非父类的 float16 张量
2. **修改 `store_dtype`**：设置为 `torch.uint8`

父类 `_create_buffers` 结束时也会调用 `self._build_kv_buffer_descs()`，但那时 `store_dtype` 还是 `torch.bfloat16` 或 `torch.float16`。SIMO 必须在修改 `store_dtype` 之后重新调用，否则：

- `itemsize` 会错误地等于 2 bytes（bf16/fp16）而非 1 byte（uint8）
- `row_bytes` = `np.prod(shape[1:]) × itemsize` 会被高估一倍

这会导致下游消费者（见 34.4）对缓冲区的字节跨度计算错误，进而导致 KV cache 数据传输、内存注册等操作读取或写入错误大小。

### 34.4 `_kv_buffer_descs` 的下游消费者

`_kv_buffer_descs` 列表在 SGLang 中有 **两个核心消费点**：

#### 消费点 1：PD（Prefill-Decode）分离的 KV 传输 — `get_contiguous_buf_infos()`

```python
def get_contiguous_buf_infos(self):
    tensors = self._pd_registerable_tensors()      # ← 顺序需与 descs 一致
    ptrs = [t.data_ptr() for t in tensors]
    lens = [d.final_span_bytes(self.size, self.page_size) for d in self._kv_buffer_descs]
    item_lens = [d.item_len_bytes(self.page_size) for d in self._kv_buffer_descs]
    return ptrs, lens, item_lens
```

在 PD 分离模式下，prefill server 需要将计算出的 KV cache 传输给 decode server。`get_contiguous_buf_infos` 利用 `_kv_buffer_descs` 计算：
- `lens`: 每个 buffer 需要传输的总字节数
- `item_lens`: 每个 page 对应的字节块大小

如果 SIMO 没有正确设置 `store_dtype = uint8` 并重建描述符，这里计算的 `lens` 和 `item_lens` 会是实际值的 **2 倍**，导致传输错误。

#### 消费点 2：CUDA-VMM Post-Capture 内存管理 — `_alloc_post_capture_buffers()`

```python
def _alloc_post_capture_buffers(self):
    self._post_capture_owner = KvVmmBufferOwner(
        store_dtype=self.store_dtype,
        page_size=self.page_size,
        reserved_num_tokens=self.size,
        buffer_descs=self._build_kv_buffer_descs(),   # ← 传入描述符
    )
```

在 CUDA Graph capture 后的动态 resize 场景下，`KvVmmBufferOwner` 使用描述符来确定每个 buffer 的字节大小，驱动 CUDA Virtual Memory Management 的内存分配。错误的 `itemsize` 会导致分配过多或过少的内存。

### 34.5 总结

| 问题 | 答案 |
|---|---|
| `_build_kv_buffer_descs()` 做了什么？ | 为所有层的 k buffer 和 v buffer 构建 `KvBufferDesc` 列表，记录每个 buffer 的**名称、形状、每行字节数、每行 token 数** |
| 为什么用 `hasattr` 检查？ | 旧版本 SGLang 的 `KVCache` 可能没有这个方法，做了向前兼容 |
| 为什么要在 `store_dtype = torch.uint8` 之后调用？ | `itemsize` 依赖 `store_dtype`。SIMO 将 dtype 从 float16 改成 uint8，必须用新 dtype 重建描述符，否则字节跨度计算会差一倍 |
| 不调用有什么后果？ | PD 分离模式下 KV 传输的 `lens`/`item_lens` 会被高估一倍；CUDA-VMM 分配的大小不正确。SIMO 在 `_create_buffers` 中重建了 uint8 缓冲区，如果不重建描述符，描述符仍然反映旧的 float16 buffer 形状，导致后续数据传输或内存操作出现 size mismatch |

**一句话总结**：`self._kv_buffer_descs = self._build_kv_buffer_descs()` 在 SIMO 将 KV buffer 改为 uint8 量化存储后，重新构建缓冲区描述符列表，确保下游的 KV 传输（PD 分离）和 CUDA VMM 内存管理中字节跨度的计算基于正确的 `store_dtype`（uint8）和 buffer 形状（packed + scale 格式）。

---

## 35. DynamicQuantizeLSTM::Compute 计算过程详解（seq_len=5, batch_size=3）

### 35.0 前置背景

#### 35.0.1 调用链

`test-onnx-dynamic-quant-lstm.py` 第 189 行执行：

```python
cpu_outputs = run_cases("CPU", cpu_session, cases)
```

遍历 `cases` 时，第一个 case 是 `seq_len=5, batch_size=3`。`session.run()` 触发 ONNX Runtime 执行 DynamicQuantizeLSTM 算子，其调用链如下：

```
ort.InferenceSession.run("output", "hn", "cn")
  → DynamicQuantizeLSTM::Compute             (dynamic_quantize_lstm.cc:174)
    → LSTMBase::ComputeImpl<float, uint8_t>  (lstm_base.cc:22)
      → UniDirectionalLstm<float>::Compute<uint8_t>  (uni_directional_lstm.cc:626)
        → UniDirectionalLstm<float>::ComputeImpl<uint8_t>  (uni_directional_lstm.cc:228)
          → GateComputations                  (uni_directional_lstm.cc:463)
          → ComputeGemm                       (rnn_helpers.cc:247, for uint8_t weights)
```

#### 35.0.2 输入张量的具体形状与数值含义

| 输入 | 形状 | dtype | 说明 |
|---|---|---|---|
| `X` (input) | `[5, 3, 10]` | float32 | 5 个时间步 × 3 个 batch 元素 × 10 维特征 |
| `W` (input weights) | `[1, 10, 80]` | **uint8** (量化后) | num_directions=1, input_size=10, 4×hidden_size=80 |
| `R` (recurrence weights) | `[1, 20, 80]` | **uint8** (量化后) | num_directions=1, hidden_size=20, 4×hidden_size=80 |
| `B` (bias) | `[1, 160]` | float32 | W 偏置(80) + R 偏置(80) = 160 |
| `h0` (initial_h) | `[1, 3, 20]` | float32 | 初始隐藏状态 |
| `c0` (initial_c) | `[1, 3, 20]` | float32 | 初始细胞状态 |
| `w_scale` | `[1, 80]` | float32 | W 权重的 per-channel scale（80 列各有一个 scale） |
| `w_zp` | `[1, 80]` | uint8 | W 权重的 per-channel zero_point |
| `r_scale` | `[1, 80]` | float32 | R 权重的 per-channel scale |
| `r_zp` | `[1, 80]` | uint8 | R 权重的 per-channel zero_point |

注意：80 = 4 × 20 = hidden_size × 4（4 个门：Input, Output, Forget, Cell）。

输出：

| 输出 | 形状 | dtype | 说明 |
|---|---|---|---|
| `output` (Y) | `[5, 3, 20]` | float32 | 每个时间步的隐藏状态 `H_t` |
| `hn` (Y_h) | `[1, 3, 20]` | float32 | 最后一个时间步的隐藏状态 = `output[4, :, :]` |
| `cn` (Y_c) | `[1, 3, 20]` | float32 | 最后一个时间步的细胞状态 |

#### 35.0.3 Weights 预打包 (PrePack)

在第一次推理之前，`DynamicQuantizeLSTM::PrePack` 已经将 uint8 的 `W` 和 `R` 通过 `MlasGemmPackB` 预打包成 MLAS 内部格式（`packed_W_`, `packed_R_`），后续 GEMM 直接使用打包好的格式以获得更高性能。

#### 35.0.4 量化参数总结

量化公式（在 `ComputeGemm` 中完成）：

```
Y_fp32 = (a_scale × w_scale) × [ (A_uint8 - a_zp) × (W_uint8 - w_zp)^T ] + β × C_prev
```

- 激活 A (float32) → 动态量化 → A_uint8 + a_scale + a_zp（**每次 GEMM 前动态计算**）
- 权重 W/R 已经预量化为 uint8 + per-channel scale + per-channel zero_point
- GEMM 在 int32 中累积，最后乘以 scale 转回 float32

---

### 35.1 第一阶段：DynamicQuantizeLSTM::Compute — 参数准备

**源码：** `dynamic_quantize_lstm.cc:174-251`

```
1. 获取打包后的权重 buffer：
   - packed_W_.buffer_ 非空 → W = nullptr（使用预打包数据）
   - packed_R_.buffer_ 非空 → R = nullptr（使用预打包数据）

2. 获取 scale 和 zero_point：
   - w_scale: [1, 80]，w_zp: [1, 80]   (per-channel)
   - r_scale: [1, 80]，r_zp: [1, 80]   (per-channel)

3. 验证 scale/zp 形状 (WeightCheck 宏)：
   - W_scale_shape[0] == 1, W_scale_shape[1] == 80 == 4*hidden_size ✓
   - 同理验证 R_scale, W_zp, R_zp

4. 确定 signed/unsigned：
   - is_W_signed = packed_W_.is_W_signed_ → uint8 量化 → false（unsigned）
   - is_R_signed = packed_R_.is_R_signed_ → false

5. 验证非对称量化的 zero_point (ZeroPointCheck 宏)：
   - 对于 unsigned 权重，所有 80 列的 zp 必须相等（对 uint8 常量 zero_point 检查）
   - 对于 signed 权重，zp 必须全为 0（对称量化）

6. 构建量化参数对象：
   QuantizationParameter quant_para_W_1(w_scale.Data, w_zp.Data, is_W_signed=false, scale_size=80)
   QuantizationParameter quant_para_R_1(r_scale.Data, r_zp.Data, is_R_signed=false, scale_size=80)

7. 计算每个方向的权重跨度：
   W_size_per_direction = 10 * 80 = 800
   R_size_per_direction = 20 * 80 = 1600

8. 构建 GemmWeights 对象：
   GemmWeights<uint8_t> W_1(0,  nullptr, W_size_per_direction=800,  packed_W_, &quant_para_W_1)
   GemmWeights<uint8_t> R_1(0,  nullptr, R_size_per_direction=1600, packed_R_, &quant_para_R_1)

   由于 packed_W_.buffer_ 非空，Init() 中：
     is_prepacked_ = true
     buffer_ = static_cast<uint8_t*>(packed_W_.buffer_.get()) + 800 * 0
            = packed_W_.buffer_ 起始地址（direction 0）

9. 单向 LSTM → 不需要 W_2, R_2，直接调用：
   LSTMBase::ComputeImpl<float, uint8_t>(context, W_1, W_2, R_1, R_2)
```

---

### 35.2 第二阶段：LSTMBase::ComputeImpl<float, uint8_t> — 输入/输出张量提取

**源码：** `lstm_base.cc:22-178`

```
1. 提取输入：
   X:     [5, 3, 10] → seq_length=5, batch_size=3, input_size=10
   B:     [1, 160]   → bias，每方向 8*20=160
   h0:    [1, 3, 20] → 初始隐藏状态
   c0:    [1, 3, 20] → 初始细胞状态
   P:     nullptr    → 无 peephole 权重

2. 分配输出：
   Y:   [5, 1, 3, 20]  → 完整的输出序列
   Y_h: [1, 3, 20]     → 最终隐藏状态 hn
   Y_c: [1, 3, 20]     → 最终细胞状态 cn

3. 按方向拆分：
   direction_ == kForward → 单向
   bias_1 = bias[0:160]
   initial_hidden_1 = initial_h[0:60]   (1*3*20=60)
   initial_cell_1 = initial_c[0:60]

4. 创建 UniDirectionalLstm<float> 对象（单向）：
   UniDirectionalLstm<float> fw(
     seq_length=5, batch_size=3, input_size=10, hidden_size=20,
     direction=kForward, ...
   )

5. 调用 fw.Compute(input, seq_lens, num_directions=1, W_1, R_1, output_1, hidden_output_1, last_cell_1)
```

---

### 35.3 第三阶段：UniDirectionalLstm::ComputeImpl<uint8_t> — 核心计算

**源码：** `uni_directional_lstm.cc:228-457`

这是 LSTM 计算的核心，下面逐步分解。

#### 35.3.1 初始化

```
seq_length_ = 5, batch_size_ = 3, input_size_ = 10, hidden_size_ = 20
direction_ = kForward
hidden_size_x4 = 80
total_rows = max_sequence_length * batch_size = 5 * 3 = 15

output_iofc_ 缓冲区（预先分配）:
  逻辑形状: {hidden_size_, 4, batch_size_, seq_length_} = {20, 4, 3, 5}
  总元素数: 20 * 4 * 3 * 5 = 1200
  用途: 按时间步顺序存储每个 step 的 X*W + H*R 结果

分配量化缓冲区:
  quantized_input_or_a_:  大小 max(5*3*10, 3*20) = max(150, 60) = 150 个 uint8
  quantized_C_buffer_:    大小 3 * 4 * 20 = 240 个 int32
```

#### 35.3.2 步骤 A：批量 GEMM — 一次性计算所有时间步的 X*W

```cpp
// uni_directional_lstm.cc:287-293
float alpha = 1.0f;
float beta = 0.0f;   // 第一次 GEMM: 清零输出

ComputeGemm(total_rows=15, hidden_size_x4=80, input_size=10,
            alpha, inputs,  // A: X[15, 10]
            input_weights,  // B: W[10, 80]  (uint8, pre-packed)
            beta,
            C: output_iofc_, ldc=80,  // C: output_iofc_[15, 80]
            quantized_input_or_a_,
            quantized_C_buffer_,
            thread_pool, ...);
```

**这次 GEMM 的量化内部流程**（`rnn_helpers.cc:247-317`，`GemmWeights<uint8_t>` 重载）：

```
1. 动态量化 A 矩阵 (X[15, 10]):
   a. GetQuantizationParameter(A, 150, a_scale, a_zero_point, thread_pool)
      → 在 150 个 float 元素中找到 min/max
      → 计算 a_scale = (max - min) / 255
      → 计算 a_zero_point = round(-min / a_scale)

   b. ParQuantizeLinearStd(A, quantized_A_buffer, 150, a_scale, a_zp)
      → 将 150 个 float 元素量化为 uint8
      → A_uint8[i] = clip(round(A[i]/a_scale) + a_zp, 0, 255)

2. 计算 scale_multiplier (per-column, 80 个):
   对于每列 s (0..79):
     scale_multiplier[s] = a_scale * w_scale[s]

3. beta = 0.0 → 使用 MLAS_QGEMM_OUTPUT_MODE::ZeroMode
   → 输出 C 中先存 int32 累积结果，再乘以 scale 转回 float

4. 调用 MlasGemm:
   输入:  A_uint8 [15, 10]
   权重:  W_uint8 [10, 80] (pre-packed)
   输出:  C[15, 80] = Σ(A_uint8 - a_zp) × (W_uint8 - w_zp) × (a_scale × w_scale)
```

**结果：**

```
output_iofc_ = X_mat[15, 10] × W[10, 80]^T

存储布局（一维展开）:
  output_iofc_[0..239]    = step=0 的 3 个 batch × 80：X[0,0:3,:] * W^T
  output_iofc_[240..479]  = step=1 的 3 个 batch × 80：X[1,0:3,:] * W^T
  output_iofc_[480..719]  = step=2 的 3 个 batch × 80：X[2,0:3,:] * W^T
  output_iofc_[720..959]  = step=3 的 3 个 batch × 80：X[3,0:3,:] * W^T
  output_iofc_[960..1199] = step=4 的 3 个 batch × 80：X[4,0:3,:] * W^T

每 80 列的组织 (4 个门，每个门 20 维):
  offset 0..19:   Input Gate  pre-activation (i_gate)
  offset 20..39:  Output Gate pre-activation (o_gate)
  offset 40..59:  Forget Gate pre-activation (f_gate)
  offset 60..79:  Cell Gate   pre-activation (c_gate, block input)
```

设置 `beta = 1.0f`，后续 `ComputeGemm` 调用将**累加**到已有数据（不再清零）。

#### 35.3.3 步骤 B：确定并行策略

```cpp
// uni_directional_lstm.cc:306-311
num_seq_to_compute = batch_size_;   // = 3 (batch_parallel_ 设为 true 但只有一个线程时保持 3)
```

假设单线程场景或 `num_threads_=1` → `batch_parallel_=true` 但 `num_seq_to_compute=3`，所以 `sequences_calculator(0, ttp)` 一次处理全部 3 个 batch 元素。

#### 35.3.4 步骤 C：时间步循环 (sequences_calculator lambda)

这是最核心的部分。Lambda 进入循环 `for (int step = 0; step < 5; step++)`。

---

##### **Token 0 (step=0)**

```
1. 获取当前时间步在 output_iofc_ 中的位置:
   step_out_IOFC = output_iofc_.begin() + (0 * 3 + 0) * 80 = output_iofc_.begin()
   → 指向 X[0,:]*W 的结果: 3 行 × 80 列

2. 循环 GEMM: H_{t-1} × R → 累加到 X_t*W:
   previous_state = batched_hidden0_ + 0*20  → 指向 h0[0,0:3,0:20]

   ComputeGemm(num_seq=3, N=80, K=20,
               alpha=1.0,
               A: previous_state[3, 20],  // H_{t-1} = h0 (初始隐藏状态)
               recurrent_weights: R[20, 80],  // 量化 uint8, pre-packed
               beta=1.0,                 // ← 累加模式!
               C: step_out_IOFC[3, 80],   // ← 已有 X_t*W
               ldc=80,
               quantized_A_buffer: quantized_input_or_a_[0..59],   // 3*20
               quantized_C_buffer: quantized_C_buffer_[0..239]);   // 3*80

   GEMM 量化内部流程:
   a. 动态量化 A (H_{t-1}[3, 20]):
      a_scale = (H_max - H_min) / 255
      A_uint8[i] = clip(round(H[i]/a_scale) + a_zp, 0, 255)

   b. 计算 per-column scale_multiplier[80]:
      scale_multiplier[s] = a_scale * r_scale[s]

   c. beta = 1.0 → AccumulateMode:
      C_temp[3, 80] = int32_matmul   (写入 quantized_C_buffer)
      C[3, 80] += C_temp × scale     (累加到原有的 X_t*W 上)

   最终 step_out_IOFC[3, 80] = X_t*W + H_{t-1}*R  (对于 step=0 的所有 3 个 batch)

3. 门计算 (GateComputations):
   batch_output = outputs[0 * 3 * 20] = outputs[0, :, :]  (第一个时间步的输出)

   对每个 batch 元素 b = 0, 1, 2:

   a. 提取 4 个门 (LSTM 的 ONNX 门序为 I, O, F, C):
      pi = step_out_IOFC[b*80 + 0..19]   (Input gate)
      po = step_out_IOFC[b*80 + 20..39]  (Output gate)
      pf = step_out_IOFC[b*80 + 40..59]  (Forget gate)
      pc = step_out_IOFC[b*80 + 60..79]  (Cell gate / block input)

   b. Input Gate:
      pi[j] = clip(pi[j] + bias_WRi[j], -clip_, clip_)  // 加偏置并裁剪
      pi[j] = sigmoid(pi[j])                              // sigmoid 激活

   c. Forget Gate:
      pf[j] = clip(pf[j] + bias_WRf[j], -clip_, clip_)
      pf[j] = sigmoid(pf[j])

   d. Cell Gate:
      pc[j] = clip(pc[j] + bias_WRc[j], -clip_, clip_)
      pc[j] = tanh(pc[j])

   e. 更新细胞状态 C_t (in-place):
      C_prev → C_t
      C_t[j] = C_prev[j] * pf[j] + pi[j] * pc[j]

   f. Output Gate:
      po[j] = clip(po[j] + bias_WRo[j], -clip_, clip_)
      po[j] = sigmoid(po[j])

   g. 计算隐藏状态 H_t:
      C_t_tanh[j] = tanh(C_t[j])
      H_t[j] = po[j] * C_t_tanh[j]

   h. 将 H_t 写入输出:
      batch_output[b*20 + j] = H_t[j]
      → 即 outputs[0, b, j]

4. 更新 previous_state = batch_output + 0*20
   → 指向 outputs[0, 0, 0] (step=0 的 H_t)

   current state after step 0:
     C_prev (batched_internal_memory_prev_): C_0  (in-place updated)
     batched_internal_memory_clipped_[b, :]: tanh(C_0)
     H_0: outputs[0, :, :]
```

##### **Token 1 (step=1)**

```
1. step_out_IOFC = output_iofc_.begin() + (1 * 3 + 0) * 80 = output_iofc_.begin() + 240
   → 指向 X[1,:]*W 的结果 (已在上面的批量 GEMM 中算好)

2. 循环 GEMM: H_0 × R 累加到 X_1*W:
   previous_state = outputs[0, :, :] (上一步的 H_0)

   ComputeGemm(3, 80, 20, alpha=1.0,
               A: H_0[3, 20],         // 上一步的隐藏状态
               recurrent_weights: R,
               beta=1.0,
               C: step_out_IOFC[3, 80])  // 累加到 X_1*W

   结果: step_out_IOFC = X_1*W + H_0*R

3. 门计算: 同上 → 生成 H_1, C_1
   写入 outputs[1, :, :]

4. previous_state = outputs[1, :, :]
```

##### **Token 2 (step=2)**

```
1. step_out_IOFC = output_iofc_.begin() + (2 * 3 + 0) * 80 = output_iofc_.begin() + 480

2. 循环 GEMM: H_1 × R 累加到 X_2*W → 结果写回 step_out_IOFC[3, 80]

3. 门计算 → H_2, C_2 → outputs[2, :, :]

4. previous_state = outputs[2, :, :]
```

##### **Token 3 (step=3)**

```
1. step_out_IOFC = output_iofc_.begin() + (3 * 3 + 0) * 80 = output_iofc_.begin() + 720

2. 循环 GEMM: H_2 × R 累加 → step_out_IOFC[3, 80] = X_3*W + H_2*R

3. 门计算 → H_3, C_3 → outputs[3, :, :]

4. previous_state = outputs[3, :, :]
```

##### **Token 4 (step=4)**

```
1. step_out_IOFC = output_iofc_.begin() + (4 * 3 + 0) * 80 = output_iofc_.begin() + 960

2. 循环 GEMM: H_3 × R 累加 → step_out_IOFC[3, 80] = X_4*W + H_3*R

3. 门计算 → H_4, C_4 → outputs[4, :, :]
```

循环结束（step=5 不满足 step < max_sequence_length=5）。

#### 35.3.5 步骤 D：后处理

```
对于每个 batch 元素 i = 0, 1, 2:
  seq_len = 5 (因为 sequence_lengths 未提供，默认全部 = seq_length_)

  // 复制最后一个时间步的输出到 final_hidden_state (hn)
  src = outputs[(5-1) * 3 * 20 + i * 20]  = outputs[4, i, :]
  dst = final_hidden_state[i * 20]
  gsl::copy(src, dst)  → Y_h[0, i, :] = H_4[i, :]

// final_cell_state 已在时间步循环中维护:
//   Y_c[0, i, :] = C_4[i, :]

// max_sequence_length=5 == seq_length_=5 → 不需要 zero-padding
```

---

### 35.4 计算全景图（数据流与内存布局）

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Phase A: 批量 GEMM                            │
│                                                                      │
│  X[5,3,10] ──reshape──> X_mat[15,10]                                 │
│                           │                                          │
│                    ┌──────┴──────┐                                    │
│                    │ Quantize A  │  动态量化: float → uint8           │
│                    └──────┬──────┘                                    │
│                           ▼                                          │
│       ┌───────────────────────────────────┐                           │
│       │ MlasGemm (uint8 GEMM)             │                           │
│       │  A_uint8[15,10] × W_uint8[10,80]  │  W 已预量化 + 预打包     │
│       │  output = int32* + dequant         │                           │
│       └───────────────┬───────────────────┘                           │
│                       ▼                                              │
│  output_iofc_[15, 80]  ← X[t]*W 的结果 (全部 5 步一次性算完)           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Phase B: 时间步循环 (5 步)                        │
│                                                                      │
│  Step 0:                                                             │
│    ┌─────────┐    ┌──────────────┐    ┌─────────────┐               │
│    │ h0/c0   │───>│ H0*R + X0*W  │───>│ GateCompute │───> H0, C0   │
│    │ [3,20]  │    │ (量化 GEMM)   │    │ I/O/F/C 门  │     [3,20]    │
│    └─────────┘    └──────────────┘    └─────────────┘               │
│                                                                      │
│  Step 1:                                                             │
│    ┌─────────┐    ┌──────────────┐    ┌─────────────┐               │
│    │ H0, C0  │───>│ H1*R + X1*W  │───>│ GateCompute │───> H1, C1   │
│    │ [3,20]  │    │ (累加模式)    │    │             │               │
│    └─────────┘    └──────────────┘    └─────────────┘               │
│                                                                      │
│  Step 2-4: 同上, 依次计算 H2-H4, C2-C4                                │
│                                                                      │
│  每步的量化 GEMM 都动态量化 H_{t-1}，配合预量化的 R 权重做 int GEMM   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  Phase C: 输出                                        │
│                                                                      │
│  outputs (Y)   = [5, 3, 20]  ← 堆叠 H0, H1, H2, H3, H4              │
│  hn (Y_h)      = [1, 3, 20]  ← outputs[4, :, :]  = H4               │
│  cn (Y_c)      = [1, 3, 20]  ← C4 (最后一步的细胞状态)               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 35.5 量化 GEMM 调用次数统计

对于 seq_len=5, batch_size=3 的 case：

| GEMM 调用 | 矩阵维度 | 次数 | 说明 |
|---|---|---|---|
| 批量 X*W | [15, 10] × [10, 80] | **1** | 一次性计算所有 5 步的 X_t*W |
| 循环 H*R | [3, 20] × [20, 80] | **5** | 每步一次，累加到对应的 output_iofc_ 区块 |
| **总计** | | **6** 次 uint8 量化 GEMM | |

注意：是 **6 次** 量化 GEMM，不是 5 次。因为使用了两阶段策略——先用 1 次**大 GEMM** (M=15) 算完所有 X*W，然后在每步循环中用 **小 GEMM** (M=3) 累加上 H*R。这比每步都做 [3,10]×[10,80] + [3,20]×[20,80] 两个独立 GEMM 更高效（利用了更大的 M 维度的计算强度）。

### 35.6 关键代码引用

| 步骤 | 文件:行号 | 说明 |
|---|---|---|
| 量化参数构建与校验 | `dynamic_quantize_lstm.cc:190-231` | 验证 W_scale/zp, R_scale/zp 的形状和内容 |
| GemmWeights 初始化 | `dynamic_quantize_lstm.cc:230-231` | 构建 int8/uint8 权重的 GemmWeights 包装 |
| ComputeImpl 调用 | `dynamic_quantize_lstm.cc:250` | `LSTMBase::ComputeImpl<float, uint8_t>` |
| LSTMBase 输入提取 | `lstm_base.cc:32-46` | X, B, h0, c0 的形状解析 |
| UniDirectionalLstm 创建 | `lstm_base.cc:161-164` | 单向 LSTM 的构造和 Compute 调用 |
| 批量 X*W GEMM | `uni_directional_lstm.cc:287-293` | 一次性算完所有时间步的 X*W，beta=0 清零 |
| H*R 循环 GEMM | `uni_directional_lstm.cc:342-349` | 每步累加到对应的 output_iofc_ 区块，beta=1 |
| 门计算 | `uni_directional_lstm.cc:463-601` | 每步的 I/O/F/C 门 + C_t 更新 + H_t 计算 |
| 最终 hidden 复制 | `uni_directional_lstm.cc:415-427` | H_last → Y_h (hn 输出) |
| 量化 GEMM (uint8) | `rnn_helpers.cc:247-317` | A 动态量化 + scale 计算 + MlasGemm |
| 激活动态量化 | `rnn_helpers.cc:275-277` | GetQuantizationParameter + ParQuantizeLinearStd |

### 35.7 量化精度路径总结

```
每次量化 GEMM 的数据流:
┌─────────────────────────────────────────────────────────────────┐
│ 1. 动态量化 A (float32 → uint8)                                 │
│    A_fp32[i] → A_uint8[i] = clip(round(A_fp32[i]/a_scale)+a_zp) │
│                                                                 │
│ 2. GEMM 累积 (int32)                                            │
│    C_int32[m,n] = Σ (A_uint8[m,k] - a_zp) × (B_uint8[k,n]      │
│                          - b_zp)                                │
│                                                                 │
│ 3. 反量化输出 (int32 → float32)                                 │
│    C_fp32[m,n] = C_int32[m,n] × (a_scale × b_scale[n])         │
│                 + (beta × C_prev_fp32[m,n])                     │
│                                                                 │
│ 注意: int32 累加器有足够精度，不会溢出                           │
└─────────────────────────────────────────────────────────────────┘
```

整个 LSTM 的计算链路中：
- **权重端**：W 和 R 在 PrePack 阶段已静态量化为 uint8（离线），scale/zp 固定
- **激活端**：每次 GEMM 调用**动态**量化输入激活 (X 或 H_{t-1}) 为 uint8（在线）
- **中间计算**：GEMM 在 int32 中累积，确保没有量化误差在累积过程中被放大
- **输出**：反量化回 float32，门函数（sigmoid, tanh）在 float32 精度下计算

---

## 36. ONNX Runtime 三种自定义算子实现与注册方式对比

### 36.1 三种方式总览

| 维度 | 方式 A: LiteCustomOp (Struct) | 方式 B: CustomOpBase (Legacy) | 方式 C: ONNX_OPERATOR_TYPED_KERNEL_EX (Internal) |
|---|---|---|---|
| **文件** | `simo_qdq_ops.cc` | `simo_qdq_cpu_ops.cc` | `dynamic_quantize_lstm.cc` |
| **算子名称** | `com.simo::Dequantize` (v2) | `com.simo::Dequantize CPU v1` | `com.microsoft::DynamicQuantizeLSTM` |
| **Compute 签名** | `Ort::Status Compute(OrtKernelContext&, const Tensor<uint8_t>&, const Tensor<uint8_t>&, Tensor<T>&)` | `void Compute(OrtKernelContext* context)` | `Status Compute(OpKernelContext* context) const` |
| **基类/基础设施** | `Ort::Custom::OrtLiteCustomStruct` | `Ort::CustomOpBase<Op, Kernel>` | `onnxruntime::OpKernel` |
| **注册方式** | `Ort::Custom::CreateLiteCustomOp<DequantizeCustomOp<T>>()` + `domain.Add()` | 继承 `CustomOpBase`，`static` 实例化 + `domain.Add()` | `ONNX_OPERATOR_TYPED_KERNEL_EX` 宏 |
| **Schema 定义** | **自动推断**（从 Compute 的强类型参数） | **手动实现**（GetName, GetInputType, ...） | **手动实现**（KernelDefBuilder + TypeConstraint） |
| **构建方式** | external（独立 `.so`/`.dll` 或编译进 binary） | external（独立 `.so`/`.dll` 或编译进 binary） | **internal**（必须编译进 ONNX Runtime 源码树） |

### 36.2 方式 C：`ONNX_OPERATOR_TYPED_KERNEL_EX` — Internal Kernel（不推荐用于自定义算子）

```cpp
// dynamic_quantize_lstm.cc
Status DynamicQuantizeLSTM::Compute(OpKernelContext* context) const {
    const Tensor* W = context->Input<Tensor>(1);
    const Tensor* R = context->Input<Tensor>(2);
    // ...
    return LSTMBase::ComputeImpl<float, uint8_t>(*context, W_1, W_2, R_1, R_2);
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DynamicQuantizeLSTM,
    kMSDomain, 1, float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(),
                               DataTypeImpl::GetTensorType<int8_t>()}),
    DynamicQuantizeLSTM);
```

**这是 ONNX Runtime 内部 kernel 的注册方式**，不是给外部用户使用的。关键特征：

1. **必须与 ONNX Runtime 一起编译**：代码放在 `onnxruntime/contrib_ops/cpu/` 等目录下，是 ORT 源码的一部分
2. **使用 ORT 内部类型系统**：`OpKernelContext*`、`Tensor`、`TensorShape` 等都是 ORT 内部类，不是 C API
3. **使用 ORT 内部内存管理**：可以直接使用 `IAllocator`、`AllocatorPtr` 等
4. **Schema 通过宏定义**：`KernelDefBuilder().TypeConstraint(...)` 手动声明输入输出类型
5. **生命周期由 ORT 内核管理**：不需要手动 new/delete kernel 实例
6. **可以访问 ORT 内部功能**：如 PrePack、SharedPrePackedBuffers、ThreadPool 等

**这是"内置算子"的方式，不是"自定义算子"的方式。** 如果你的代码在 ORT 仓库外编译（例如 SIMO 的 `simo_qdq_ops.cc`），就不能使用这种方式。

### 36.3 方式 B：`CustomOpBase` — Legacy External Custom Op（不推荐新代码使用）

```cpp
// 定义 Op 元信息
template <typename T>
class DequantizeCpuDirectOp
    : public Ort::CustomOpBase<DequantizeCpuDirectOp<T>,
                                DequantizeCpuDirectKernel<T>, false> {
 public:
  DequantizeCpuDirectOp() {
    this->version = kOrtApiVersion;
    this->start_ver_ = 2;
    this->end_ver_ = 2;
  }
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return new DequantizeCpuDirectKernel<T>(api, info);  // 手动 new
  }
  const char* GetName() const { return "Dequantize"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 1; }          // 手动声明
  ONNXTensorElementDataType GetInputType(size_t) const { return IoDtype<T>::onnx_type; }
  size_t GetOutputTypeCount() const { return 2; }         // 手动声明
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8; }
};

// 定义 Kernel（实际的 Compute 逻辑）
template <typename T>
class DequantizeCpuDirectKernel {
  void Compute(OrtKernelContext* context) {
    // 手动通过 C API 获取输入
    const OrtValue* value = nullptr;
    Ort::GetApi().KernelContext_GetInput(context, 0, &value);
    // ...
  }
};

// 注册：static 实例 + domain.Add
static const DequantizeCpuDirectOp<float> dequantize_fp32;
domain.Add(const_cast<DequantizeCpuDirectOp<float>*>(&dequantize_fp32));
```

**这是 ORT 的第一代外部自定义算子 API**，基于 `OrtCustomOp` C 结构体。关键特征：

1. **必须手动实现 schema 方法**：`GetName()`, `GetInputTypeCount()`, `GetInputType()`, `GetOutputTypeCount()`, `GetOutputType()` 等全部要手写
2. **必须手动实现 kernel 生命周期**：`CreateKernel()` 中 `new`，`KernelDestroy` 中 `delete`
3. **原始 C API 访问数据**：`Compute(OrtKernelContext* context)` 中通过 `Ort::GetApi().KernelContext_GetInput(...)` 等 C API 函数访问输入输出
4. **Op 必须是 static/全局对象**：`static const DequantizeCpuDirectOp<float> dequantize_fp32;`
5. **模板参数 `WithStatus=false`**：第 3 个模板参数控制 Compute 是否返回 `Ort::Status`（false → `void`）
6. **可选的 shape inference**：通过 `static OrtStatusPtr InferOutputShape(Ort::ShapeInferContext&)` 静态方法

**优点**：可以完全控制所有细节，EP 类型、版本号、input/output characteristic 等。

**缺点**：
- **大量样板代码**：每种 dtype 都要手动写 `GetInputType()` / `GetOutputType()`
- **手动内存管理**：kernel 的 new/delete
- **错误处理不统一**：`void Compute()` 模式下只能用异常或提前设置状态
- **类型不安全**：通过 `KernelContext_GetInput(context, index, &value)` 手动按索引获取，容易写错索引

### 36.4 方式 A：`OrtLiteCustomOp` (Struct-as-op) — 推荐方式

```cpp
// 一个 struct 包含 constructor + Compute method
template <typename T>
class DequantizeCustomOp {
 public:
  DequantizeCustomOp(const OrtApi*, const OrtKernelInfo* info) {
    Ort::ConstKernelInfo kernel_info{info};
    spec_ = SpecFromKernelInfo(QdqOp::kDequantize, IoDtype<T>::name, kernel_info);
  }

  // 静态 shape inference
  static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
    return DequantizeShape(IoDtype<T>::name, IoDtype<T>::onnx_type, ctx);
  }

  // Compute 直接接收强类型参数
  Ort::Status Compute(
      OrtKernelContext& context,
      const Tensor<uint8_t>& quantized,   // 输入0: uint8 tensor (自动解开)
      const Tensor<uint8_t>& scale,       // 输入1: uint8 tensor (自动解开)
      Tensor<T>& output) {                // 输出0: T tensor (自动分配)
    // 直接使用 .Shape(), .Data() 等
    const auto shape = quantized.Shape();
    auto* output_data = output.Allocate({outer_dim, quant_dim});
    // ...
  }

 private:
  const QdqRuntimeSpec* spec_ = nullptr;
};

// 注册：一行代码
static const std::array<std::unique_ptr<Ort::Custom::OrtLiteCustomOp>, 3> dequantize = {
    std::unique_ptr<Ort::Custom::OrtLiteCustomOp>{
        Ort::Custom::CreateLiteCustomOp<DequantizeCustomOp<float>>(
            "Dequantize", "CUDAExecutionProvider", 2)},
    // ... 其他 dtype
};
for (const auto& op : dequantize) {
    op->version = kOrtApiVersion;
    domain.Add(op.get());
}
```

**这是 ORT 推荐的第二代外部自定义算子 API**（也称为 Lite Custom Op / V2 Custom Op）。关键特征：

1. **Schema 自动推断**：输入输出类型从 `Compute()` 的参数类型自动推导
   - `const Tensor<uint8_t>&` → 输入，uint8
   - `Tensor<T>&` → 输出，`T`
   - `std::optional<Tensor<T>>` → 可选输入
2. **强类型 tensor 访问**：`Ort::Custom::Tensor<T>` 提供 `.Data()`, `.Shape()`, `.Allocate()` 等方法
3. **状态保持**：struct 的成员变量在 kernel 生命周期内保持（`spec_` 等）
4. **Constructor 接收 `OrtKernelInfo*`**：可以在构造时读取 op 属性
5. **可选 shape inference**：通过同名静态方法 `InferOutputShape`
6. **版本号通过注册参数指定**：`CreateLiteCustomOp("Dequantize", "CUDAExecutionProvider", 2)` — 第 3 个参数是 `start_ver`

**LiteCustomOp 还有另一个子模式：Function-as-op (`OrtLiteCustomFunc`)**：

```cpp
// 如果不需要状态，直接用函数
void Filter(const Ort::Custom::Tensor<float>& in, Ort::Custom::Tensor<float>& out) {
    // ...
}

Ort::Custom::CreateLiteCustomOp("Filter", "CPUExecutionProvider", Filter);
```

这对无状态操作更加简洁。

### 36.5 三种方式的函数签名为什么不同

```cpp
// 方式 A: 强类型封装，按参数位置自动映射到 input/output
Ort::Status Compute(OrtKernelContext& context,
                   const Tensor<uint8_t>& quantized,  // 第0个输入 → 自动 get_input(0)
                   const Tensor<uint8_t>& scale,      // 第1个输入 → 自动 get_input(1)
                   Tensor<T>& output);                // 第0个输出 → 自动 get_output(0)

// 方式 B: 裸 C API，手动按索引获取
void Compute(OrtKernelContext* context) {
    KernelContext_GetInput(context, 0, &value);   // 手动指定索引 0
    KernelContext_GetInput(context, 1, &value);   // 手动指定索引 1
    KernelContext_GetOutput(context, 0, ...);     // 手动指定索引 0
}

// 方式 C: ORT 内部框架，使用 ORT 内部 Tensor 类
Status Compute(OpKernelContext* context) const {
    context->Input<Tensor>(1);    // 手动指定索引
    context->Output(0, dims);     // 手动指定索引
}
```

**方式 A 的参数都是 `Tensor<T>` 的自动包装**：`OrtLiteCustomStruct` 内部会：
1. 解析 `Compute` 的签名，统计 input/output 数量和类型
2. 在 `KernelCompute` lambda 中，按参数位置创建 `Ort::Custom::Tensor<T>` 实例，并传递给 `Compute`
3. `const Tensor<T>&` = 输入；`Tensor<T>&` / `Tensor<T>*` = 输出

**方式 B 需要手动调用 C API**：所有输入输出通过 `OrtKernelContext*` 和 ORT C API 手动管理。

**方式 C 使用 ORT 内部 API**：`OpKernelContext*` 不是 C API，是 ORT 内部的 C++ 类。

### 36.6 三种注册方式的本质区别

```
┌────────────────────────────────────────────────────────────────┐
│                    注册方式的本质区别                           │
├──────────────┬─────────────────┬───────────────────────────────┤
│ 方式 A       │ CreateLiteCustom │ 底层创建一个 OrtLiteCustomOp │
│ (推荐)       │ Op<Struct>()     │ 对象，继承 OrtCustomOp C 结构│
│              │ + domain.Add()   │ 体。Schema 自动推断。        │
├──────────────┼─────────────────┼───────────────────────────────┤
│ 方式 B       │ 继承 CustomOpBase │ 底层也是一个 OrtCustomOp C   │
│ (旧，不推荐) │ static 实例     │ 结构体。Schema 手动实现。     │
│              │ + domain.Add()   │                               │
├──────────────┼─────────────────┼───────────────────────────────┤
│ 方式 C       │ ONNX_OPERATOR_   │ 底层注册到 ORT 内部的         │
│ (仅内部)     │ TYPED_KERNEL_EX  │ KernelRegistry。编译进 ORT    │
│              │ 宏              │ binary。                      │
└──────────────┴─────────────────┴───────────────────────────────┘
```

### 36.7 推荐建议

| 场景 | 推荐方式 | 理由 |
|---|---|---|
| **新的外部自定义算子**（独立 `.so`/编译进 host binary） | **方式 A** (`CreateLiteCustomOp`) | Schema 自动推断、强类型、少样板代码 |
| **已有大量 CustomOpBase 代码** | 保持方式 B，逐步迁移到方式 A | 避免一次性重写风险 |
| **需要极端性能优化（如自定义 PrePack）** | **方式 C** (`OpKernel`) | 只有内部 kernel 能访问 PrePack/ThreadPool 等内部 API |
| **集成到 ONNX Runtime 官方仓库** | **方式 C** (`ONNX_OPERATOR_TYPED_KERNEL_EX`) | 官方贡献的唯一方式 |
| **无状态简单操作** | **方式 A 子模式** (`OrtLiteCustomFunc` — function-as-op) | 比 struct-as-op 更简洁 |

**方式 A 内部还有一个小坑需要注意**：

SIMO 的 `DequantizeCustomOp` 使用的是 `OrtLiteCustomStruct`（struct-as-op），其 `Compute` 方法第一个参数是 `OrtKernelContext& context`。这个 `context` 参数在**纯 `Tensor<T>` 参数**的情况下是多余的——只有在需要手动访问额外上下文时才有用。如果所有输入输出都通过强类型参数传递，可以省略 `OrtKernelContext&` 参数。

例如，官方示例中的纯函数式写法：

```cpp
// 不需要 context 参数的精简版本
void Compute(const Ort::Custom::Tensor<float>& in,
             Ort::Custom::Tensor<float>& out) { ... }
Ort::Status Compute(const Ort::Custom::Tensor<uint8_t>& in,
                    Ort::Custom::Tensor<float>& out) { ... return Ort::Status{nullptr}; }
```

但 SIMO 的 `DequantizeCustomOp::Compute` 保留了 `OrtKernelContext& context` 参数，因为它需要把 context 传给 `LaunchQdqKernel` 以获取 CUDA stream。

### 36.8 一句话总结

> **方式 C（`ONNX_OPERATOR_TYPED_KERNEL_EX`）不是"自定义算子"而是"内置算子"的实现方式，只能用于编译进 ONNX Runtime 源码树的代码。方式 A（`CreateLiteCustomOp`）和方式 B（`CustomOpBase`）都是外部自定义算子的 API，底层都是 `OrtCustomOp` C 结构体，但方式 A 是 ORT 官方推荐的现代方式——它通过模板元编程自动从 C++ 函数签名推断 schema，消除了方式 B 需要手写的大量样板代码。**
