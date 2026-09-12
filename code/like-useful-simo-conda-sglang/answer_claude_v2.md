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

## 37. onnxruntime CUDA LSTM 的底层实现，以及 SimoQuantizeLSTM 摆脱 torch 依赖的可行性分析

### 37.1 问题一：onnxruntime CUDAExecutionProvider 的浮点 LSTM 是调用 cudnn，还是有自己的 CUDA kernel？

**结论：直接调用 cuDNN，没有自己的 LSTM CUDA kernel。**

证据链如下：

1. **注册的是 cuDNN 基类 kernel**
   `onnxruntime/core/providers/cuda/rnn/lstm.cc:51-61` 用 `REGISTER_KERNEL_VERSIONED_TYPED(float)` / `REGISTER_KERNEL_TYPED(float)` 注册了 `float`/`double`/`MLFloat16` 三种浮点类型的 LSTM，但 kernel 类就是 `LSTM<T>`。

2. **LSTM kernel 是 `CudnnRnnBase` 的子类**
   `onnxruntime/core/providers/cuda/rnn/lstm.h:12-15` 定义 `class LSTM final : public CudnnRnnBase<T>`，构造函数里 `SetRNNMode(CUDNN_LSTM)`。它只处理 ONNX 的 `W[iofc]`/`R[iofc]` 布局到 cuDNN linLayerID 的映射（`lstm.h:24-27`），并把权重交给 `CacheCudnnRnnWeights`（`lstm.h:31`）。

3. **真正的计算在 cuDNN 调用里**
   `onnxruntime/core/providers/cuda/rnn/cudnn_rnn_base.cc:184` 的 `CudnnRnnBase<T>::ComputeInternal()` 做了所有事：
   - `cudnn_rnn_base.cc:351` 调用 `cudnnRNNForward(GetCudnnHandle(ctx), rnn_desc, CUDNN_FWD_MODE_INFERENCE, ...)` —— **这是推理时的核心计算**；
   - `cudnn_rnn_base.cc:306-316` 通过 `CudnnRNN::Set()` 构造 cuDNN RNN descriptor（`cudnn_rnn_base.h:52-67` 的 `cudnnSetRNNDescriptor_v8`）；
   - `cudnn_rnn_base.cc:322-327` 的 `ReorganizeWeights()` 把 ONNX 权重按 cuDNN 的 linLayerID 重排并拷入 cuDNN 权重 buffer（`cudnn_rnn_base.cc:34-46` 的 `cudnnGetRNNWeightParams` + `cudaMemcpyAsync`）。

4. **ONNX 算子文档本身也说明 LSTM 通常由 CuDNN 这类自定义实现支持**
   `onnx/docs/Operators.md:17334-17335` 明确写 "This operator is usually supported via some custom implementation such as CuDNN."

5. **ORT 里唯一的非 cuDNN LSTM 是 CPU 版本和 DynamicQuantizeLSTM**
   CUDA provider 目录下只有 `rnn/` 里这一个 LSTM 实现（`grep LSTM` 在 cuda provider 下除了 `lstm.cc/h`、`cudnn_rnn_base.*` 外只有 `cuda_execution_provider.cc` 的调度注册）。ORT 没有为 CUDA LSTM 写逐时间步的 CUDA kernel，而是把整个序列（含 `sequence_lens` 打包逻辑，见 `cudnn_rnn_base.cc:223-253`）一次交给 cuDNN。

> 附带说明：cuDNN LSTM 只注册了 float/double/float16（`lstm.cc:51-61`），**没有 bfloat16**。而 SIMO 的 SimoQuantizeLSTM 显式支持 bf16（见下文 37.2），这也是 SIMO 不能直接"换成 ORT 官方 LSTM kernel"的原因之一。

---

### 37.2 问题二：当前 SimoQuantizeLSTM 用了哪些 torch 算子？ORT 的 cuda 算子能否覆盖？

#### 37.2.1 SimoQuantizeLSTM 对 torch 的依赖点

文件 `simo/onnx/ort_plugin/simo_lstm_ops.cc`（978 行）：

**A. 头文件 / 编译期依赖（torch 强耦合的根源）**
- `simo_lstm_ops.cc:5-9` include `<ATen/ATen.h>`、`<ATen/ops/from_blob.h>`、`<c10/core/InferenceMode.h>`、`<c10/cuda/CUDAGuard.h>`、`<c10/cuda/CUDAStream.h>`；
- 模板参数携带 torch 类型：`simo_lstm_ops.cc:42` `at::kFloat`、`:50` `at::kHalf`、`:58` `at::kBFloat16`；
- 构建时显式链接 torch：`simo/onnx/ort_plugin/build_runtime.py:89-111` 传了 `-ltorch` `-ltorch_cpu` `-ltorch_cuda` `-lc10` `-lc10_cuda`，include 里还加了 `torch_include_paths()`（`build_runtime.py:87`）。

**B. 张量包装（避免拷贝、直接引用 ORT buffer）**
- `simo_lstm_ops.cc:276-283` `WrapOrtTensor()`：`at::from_blob(tensor.Data(), shape, options)` 把 ORT 输入 buffer 包成 torch tensor（零拷贝）；
- `simo_lstm_ops.cc:285-293` `WrapOrtOutput()`：先 `tensor.Allocate(shape)` 分配 ORT 输出，再 `at::from_blob` 包起来；
- 因此后面所有计算都能用 torch 算子直接写进 ORT 分配的输出 buffer。

**C. 权重反量化 DequantizeGate（`simo_lstm_ops.cc:467-515`）**
- `at::empty`（`:485`）分配 fp32/fp16/bf16 输出；
- `result.reshape(...)`（`:499`）、`result.slice(...)`（`:502`）、`result.permute(...)`（`:506`）、`result.contiguous()`（`:508`）做布局恢复；
- 注意核心反量化计算本身走的是 SIMO 自己的 Triton QDQ kernel（`:486-496` 的 `LaunchQdqKernelForRuntime`），**不是 torch**。

**D. 输入/隐状态量化-反量化 QuantizeDequantizeState（`simo_lstm_ops.cc:517-614`）**
- `input.contiguous()` / `input.transpose(...)`（`:529,:535,:542,:548`）；
- `at::zeros` + `copy_` 做 MX 格式的 block 对齐 padding（`:560-563`）；
- `at::empty` 分配 quantized/scale/dequantized buffer（`:578-580`）；
- 核心 Q/DQ 仍走 `LaunchQdqKernelForRuntime`（`:582-603`）。

**E. 主循环 ComputeImpl（`simo_lstm_ops.cc:616-758`）—— 真正吃 torch 的部分**
- `simo_lstm_ops.cc:670-673` 把 ORT 的 CUDA stream 转成 torch 外部 stream 并上 `CUDAStreamGuard` + `InferenceMode`；
- 每个 gate 的矩阵乘：`simo_lstm_ops.cc:739-740` `at::matmul(x_qdq, w_gates[d][g].transpose(0,1)) + at::matmul(h_qdq, r_gates[d][g].transpose(0,1))`；
- 加 bias：`simo_lstm_ops.cc:742` `gates[gate] + gate_bias[gate]`；
- 激活：`:746-749` `at::sigmoid` ×3、`at::tanh` ×2；
- 门融合：`:750` `c = forget_gate * c + input_gate * cell_gate`，`:751` `h = output_gate * at::tanh(c)`；
- 写回 ORT 输出：`:752` `y.select(...).copy_(h)`，`:754-755` `y_h/y_c.select(...).copy_(...)`。

> 一句话归纳：**matmul/sigmoid/tanh/elementwise(+,-,*)/copy_/transpose/contiguous/slice/select/reshape 全部依赖 torch；只有 QDQ 的量化/反量化计算用的是 SIMO 自己的 Triton kernel。**

#### 37.2.2 onnxruntime 的 cuda 算子能否满足这些需求？

理论上可以，而且 ORT 的 CUDA 算子覆盖得非常全：

| SimoQuantizeLSTM 用的 torch 算子 | ORT CUDA 对应算子 | ORT 实现 | fp32/fp16/bf16 支持 |
|---|---|---|---|
| `at::matmul` | `MatMul` | 走 cuBLAS：`onnxruntime/core/providers/cuda/math/matmul.cc:94` `ComputeInternal`，`matmul.cc:174`/`:345` `cublasGemmHelper`；有 tunable 变体（含 bf16）：`onnxruntime/core/providers/cuda/tunable/math/matmul.cc:105-108` | float/double/fp16/bf16（`matmul.cc:44-47`） |
| `at::sigmoid` / `at::tanh` | `Sigmoid` / `Tanh` | 手写 elementwise CUDA kernel：`activations.cc:80-81,84-85` 注册；`activations.cc:34-47` `UNARY_ACTIVATION_COMPUTE` 派发到 `activations_impl.cu` 的 `Impl_Sigmoid`/`Impl_Tanh` | float/double/fp16/**bf16**（`activations.cc:80-81` 用了 `UNARY_ACTIVATION_OP_HFD_WITH_BF16`） |
| `+` / `-` / `*`（`c = f*c + i*g` 等） | `Add` / `Sub` / `Mul` | 手写 elementwise CUDA kernel：`binary_elementwise_ops.cc:276-289` 注册；`binary_elementwise_ops.cc:127-145` `BINARY_ELEMENTWISE_COMPUTE` 派发；broadcast 逻辑在 `binary_elementwise_ops.h:141-178`（`Add`/`Sub`/`Mul` class） | float/double/fp16/bf16（`binary_elementwise_ops.cc:281-289` `_WITH_BF16`/`BWUZCSILHFD`） |
| `transpose` / `slice` / `select` / `reshape` | ONNX `Transpose`/`Slice` 等 | 这些属于"内存布局变换"，在 ORT 中要么是 `Transpose` 的 CUDA kernel（`cuda` 下有 transpose 实现），要么是纯 meta 操作 | — |

**关键结论**：
- 功能层面，ORT 的 `MatMul`/`Sigmoid`/`Tanh`/`Add`/`Sub`/`Mul` CUDA 算子**能覆盖** SimoQuantizeLSTM 在 fp32/fp16/bf16 下对 matmul/激活/elementwise 的需求；
- 但"能覆盖"**不等于"能直接调用"**。见 37.3 的工程约束，那才是真正决定能不能摆脱 torch 的地方。

---

### 37.3 问题三：能不能真正摆脱 torch？两种实现哪个更符合工程规范？

#### 37.3.1 现实约束：SimoQuantizeLSTM 是"外部自定义算子"，不是 ORT 内置 kernel

这是整个讨论的**决定性前提**：

1. **外部自定义算子无法访问 ORT 内部 kernel 类**
   ORT 的 CUDA `MatMul<T>`/`Sigmoid<T>`/`Add<T>` 等 kernel 类都继承 `onnxruntime::CudaKernel`（`cuda_rnn_base.h:81`、`binary_elementwise_ops.h:129`），它们只能在 `onnxruntime/core/providers/cuda/` 源码树内使用（方式 C 的内置注册）。SIMO 的插件是**在 ORT 仓库外编译的独立 `.so`**（见 `setup.py:106-110` `build_sm90_runtime`、`build_runtime.py:91-114` 的裸 g++ 链接），它拿到的是 `OrtKernelContext`（`simo_lstm_ops.cc:388`）而不是 `OpKernelContext`，**根本 new 不出 `MatMul<float>` 这种类，也没有 `ctx->GetCublasHandle()`**。

2. **ORT 提供的 C API 只有"在当前 stream 上 launch"的能力**
   自定义算子能拿到的全部 GPU 资源是 `Ort::GetApi().KernelContext_GetGPUComputeStream(&context, &stream)`（`onnxruntime_c_api.h:4662`，实际用法在 `simo_lstm_ops.cc:246-274` `CudaStreamAndDevice()`）。没有 cublas handle、没有 cudnn handle、没有 ORT allocator。

3. **因此"调用 onnxruntime 的 matmul/tanh/sigmoid/elementwise cuda 算子"现实上只有两条路：**
   - **路线 X：在 SimoQuantizeLSTM 内部实例化 ORT 内置 kernel** —— 做不到，类不可见、依赖不可用；
   - **路线 Y：自己调用 cuBLAS/cuDNN/CUDA runtime 重新实现 matmul/tanh/sigmoid/elementwise** —— 技术上可行（cuBLAS 的 `cublasGemmEx`、cuDNN 的激活、或仿照 `activations_impl.cu` 手写 elementwise kernel），但这本质上是"**重新实现 ORT 的 kernel**"，而不是"调用 ORT 的算子"。

#### 37.3.2 "调用 ORT 算子" vs "调用 torch 算子" 的工程对比

| 维度 | 调用 torch 算子（现状） | 调用 ORT 算子（理想） | 自己调 cuBLAS/cuDNN（路线 Y） |
|---|---|---|---|
| 可行性 | ✅ 已实现 | ❌ 外部插件无法实例化 ORT 内置 kernel | ✅ 可行，但要重写 kernel |
| 依赖 | 强依赖 libtorch（`build_runtime.py:89-111` 链接 5 个 torch 库） | 摆脱 torch，但和 ORT 内部深度耦合 | 摆脱 torch，只依赖 CUDA/cuBLAS/cuDNN |
| 数值一致性 | matmul 是 cuBLAS，和 ORT 一致；激活/elementwise 是 torch 自己的实现 | 和 ORT 逐 bit 一致 | 取决于实现精度（如 sigmoid 的精度处理 `activations_impl.cu:55-61`） |
| 维护性 | 插件和 torch 版本绑定 | 插件和 ORT 版本绑定 | 与 ORT 内部实现解耦，但要自己维护 kernel |
| 工程规范 | 依赖一个"比自己大一个数量级"的框架，只为一层 LSTM | 违反 ORT 的扩展边界 | 自定义算子推荐做法 |

#### 37.3.3 结论：哪种更符合工程规范？

**核心判断：SimoQuantizeLSTM 是"external custom op"，这个身份决定了"调用 onnxruntime 内置 cuda 算子"这条路在架构上是不成立的。** ORT 的扩展边界就是 C API + stream；凡是需要 ORT 内部句柄（cublas/cudnn/allocator/OpKernelContext）的调用，都不属于外部插件的合法能力范围。强行"调用 ORT 算子"等于把 ORT 内部实现当私有 API 用，这比依赖 torch 更不工程规范。

在"现状（torch）"与"路线 Y（自写 cuBLAS/cuDNN）"之间，工程规范角度的排序是：

1. **如果目标只是让 SimoQuantizeLSTM 工作、快速迭代**：**保持现状调用 torch**。理由：
   - QDQ 计算本来就是 SIMO 自己的 Triton kernel，只有 matmul/激活/elementwise 是 torch；
   - `at::matmul` 底层也是 cuBLAS（与 ORT 的 matmul 走同一条 `cublasGemmEx` 路径），数值上已经和 ORT 对齐；
   - torch 的 `sigmoid`/`tanh`/elementwise 是成熟、经过大规模验证的实现，正确性有保障；
   - 构建脚本已经固定依赖 torch（`setup.py:8,10-15`、`build_runtime.py:87-111`），移除它需要同时改 build 流程。

2. **如果目标是从依赖角度"去 torch 化"（比如运行环境没有 libtorch、或希望插件更轻量）**：更工程规范的做法是**路线 Y**——用 cuBLAS（`cublasGemmEx`）做 matmul、手写或调用 cuDNN 做 sigmoid/tanh/elementwise，并沿用现有的 `KernelContext_GetGPUComputeStream`（`simo_lstm_ops.cc:246-274`）在 ORT 的 stream 上 launch。这才是 external custom op 的"正统"依赖边界（只依赖 CUDA runtime / cuBLAS / cuDNN，不依赖任何推理框架）。代价是要自己维护这几个 kernel 的 fp32/fp16/bf16 三套实现（可参考 ORT `activations_impl.cu:55-61` 的 half 精度处理）。

3. **"调用 ORT 内部算子"**：**不建议**。它既没有摆脱 torch 的部署优势，又破坏了 ORT 的插件边界，还会把插件和 ORT 具体版本绑定死。

#### 37.3.4 一句话总结

> **onnxruntime 的 CUDA LSTM 是直接调用 `cudnnRNNForward`（`cudnn_rnn_base.cc:351`），没有自己的 LSTM CUDA kernel。SimoQuantizeLSTM 当前在 matmul/sigmoid/tanh/elementwise 上使用 torch（`simo_lstm_ops.cc:739-751`），且构建时硬链接 libtorch（`build_runtime.py:89-111`）。"调用 ORT 的 cuda 算子"对外部自定义算子而言架构上不可行——ORT 只对外暴露 stream（`onnxruntime_c_api.h:4662`），内置 kernel 的 cublas/cudnn 句柄对外不可见。真正可行的去 torch 化路线是自写 cuBLAS/cuDNN 实现（路线 Y），但那要自己维护三套 dtype 的 kernel；如果以工程规范和迭代效率为第一目标，继续调用 torch 反而更稳妥，因为 `at::matmul` 底层同样是 cuBLAS，数值上与 ORT 一致，且 QDQ 核心计算本来就是 SIMO 自己的 Triton kernel。**

---

## 38. sglang 提交 11d03eaeef：FlashInfer RMSNorm + FP8 量化融合（SM90/SM100/SM120）

> 行号说明：本节行号基于提交 `11d03eaeef` 自身（即 `git show 11d03eaeef:<path>`）。该提交之后仓库有改动，当前工作区行号有偏移（`layernorm.py` 偏移较大，如 `forward_with_per_tensor_quant_fusion` 从 858 漂到 905），定位时请以函数名为主。测试文件提交后未被修改，行号与工作区一致。

### 38.1 一句话核心功能

把 **RMSNorm（含 add-residual RMSNorm）+ 静态 per-tensor FP8 激活量化** 合并成一次 FlashInfer kernel 调用，从而在 norm 与下游 FP8 GEMM 之间**省掉一次独立的量化 kernel 启动**（以及激活的两趟 HBM 读写往返）。该融合在 **SM90 / SM100 / SM120** 上启用，覆盖两种 FP8 linear 实现：原生 `Fp8LinearMethod`（非 block / mxfp8 / marlin）与 compressed-tensors 的 `CompressedTensorsW8A8Fp8`（静态 per-tensor 输入 scale）。

```
提交前： RMSNorm kernel ──► bf16 激活 ──► static_quant_fp8 ──► fp8 激活 ──► FP8 GEMM
                          (HBM 写+读)        (HBM 写+读)
提交后： FlashInfer rmsnorm_quant ──► (fp8 激活, scale) ──► FP8 GEMM
```

### 38.2 改动清单（四个层面）

| 层面 | 文件 | 作用 |
|---|---|---|
| 生产者（norm） | `python/sglang/srt/layers/layernorm.py` | 新增融合 kernel 调用 + 判定下游能否吃 pre-quant |
| 接线（模型） | `python/sglang/srt/models/llama.py`、`qwen2.py`（+ eagle） | 把下游 linear 作为 `quant_linear` 传给 norm |
| 消费者（quant） | `python/sglang/srt/layers/quantization/fp8.py`、`compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py` | 识别 tuple 形式的预量化输入 |
| GEMM 调度 | `python/sglang/srt/layers/quantization/fp8_utils.py` | 跳过重复量化、传递 out_dtype、处理 scalar-A scale |

### 38.3 代码走读

#### 38.3.1 生产者：RMSNorm 侧（layernorm.py）

1. **能力探测**：`layernorm.py:58` 新增模块级开关 `_flashinfer_rmsnorm_quant_available`；`layernorm.py:89-99` 尝试 `from flashinfer.norm import rmsnorm_quant, fused_add_rmsnorm_quant`，导入失败则置 `False`——装不上就静默降级。

2. **判定下游是否可融合**：`_fp8_static_input_scale`（`layernorm.py:370`）——若传入的 linear 是「静态 per-tensor FP8」，返回它的 `input_scale`（要求 `numel()==1`），否则返回 `None`。内部由 `_is_static_per_tensor_fp8_linear`（`layernorm.py:393`）做类型判定：
   - 原生 `Fp8LinearMethod` 且非 block / mxfp8 / marlin；或
   - compressed-tensors 的 `CompressedTensorsW8A8Fp8` 且 `is_static_input_scheme`。
   `numel()==1` 的限制来自 flashinfer 该 kernel 只支持 per-tensor 量化。

3. **新入参 `quant_linear`**：`RMSNorm` 各 backend `forward_*` 都加了 `quant_linear: Optional[nn.Module] = None`（提交里加在 `layernorm.py:474/575/591/668/708/732/783/804`）。它由 `BaseFusedOp::forward`（`python/sglang/kernels/fused_op.py:630`）以 `**kwargs` 透传到具体 backend。

4. **调度点**：`RMSNorm::forward_cuda`（`layernorm.py:469`）在 `layernorm.py:508-518` 判断：
   ```python
   if (quant_linear is not None
       and not self.cast_x_before_out_mul
       and _flashinfer_rmsnorm_quant_available):
       scale = _fp8_static_input_scale(quant_linear)
       if scale is not None:
           return self.forward_with_per_tensor_quant_fusion(...)
   ```
   它被刻意放在 empty-input / `variance_size_override` / batch-invariant 等 guard **之后**——这些路径与融合 kernel 不兼容。

5. **融合实现**：`RMSNorm::forward_with_per_tensor_quant_fusion`（`layernorm.py:858`）：
   - 无 residual → `_flashinfer_rmsnorm_quant(out, x, weight, scale, eps)`；
   - 有 residual → `_flashinfer_fused_add_rmsnorm_quant(out, x, residual, weight, scale, eps)`（就地 `residual += x`，再 `out = quant(rmsnorm(residual) * w)`）；
   - 返回契约：无 residual 返回 `(fp8_out, scale, orig_dtype)`；有 residual 返回 `((fp8_out, scale, orig_dtype), residual_out)`。
   - `orig_dtype` 是关键：把「原本的激活 dtype」带下去，让下游 GEMM 输出回到模型 dtype 而非默认 bf16（见 38.3.4 与 38.5 的回归测试）。

#### 38.3.2 接线：模型层（llama.py / qwen2.py）

- `LlamaDecoderLayer::forward`（`llama.py:341`）：
  - `llama.py:352` / `llama.py:356`：`self.input_layernorm(..., quant_linear=self.self_attn.qkv_proj)`
  - `llama.py:366`：`self.post_attention_layernorm(..., quant_linear=self.mlp.gate_up_proj)`
- `Qwen2DecoderLayer::forward`（`qwen2.py:284`）：对应 `qwen2.py:295/299/309`。
- EAGLE 变体：layer 0 把 `input_layernorm` 换成恒等占位，签名同步改为 `lambda x, quant_linear=None: x`（`llama_eagle.py:53`、`qwen2_eagle.py:54`），否则新 kwarg 会 TypeError。

即：融合的「开关」由模型自己给——只有下游确实是静态 per-tensor FP8 linear 时才真的融合。

#### 38.3.3 消费者：FP8 linear 识别 tuple（fp8.py / compressed_tensors_w8a8_fp8.py）

- `Fp8LinearMethod::apply`（`fp8.py:957`）在 `fp8.py:1035` 新增分支：输入是 tuple 时取 `qx, x_scale = x[0], x[1]`、`out_dtype = x[2] if len(x) > 2 else None`，直接调 `apply_fp8_linear(..., input_scale=x_scale, pre_quant_output_dtype=out_dtype)`。
- `CompressedTensorsW8A8Fp8::apply_weights`（`compressed_tensors_w8a8_fp8.py:228`）在 `:234` 做同样的事，额外带 `compressed_tensor_quant=True`。

#### 38.3.4 GEMM 调度（fp8_utils.py）

`apply_fp8_linear`（`fp8_utils.py:1713`）是这套融合真正落地的地方：

1. **新参数** `pre_quant_output_dtype`（`fp8_utils.py:1724`）。
2. **识别预量化输入**（`fp8_utils.py:1745`）：`input_prequantized = input_2d.dtype in (float8_e4m3fn, float8_e4m3fnuz)`；随后 `output_dtype = pre_quant_output_dtype or torch.bfloat16`（`:1749-1752`）。**没有 dtype 提示时退回 bf16**——这正是下面回归测试要防的坑。
3. **跳过重复量化**（`:1765-1772`）：预量化分支直接 `qinput = input_2d`，复用调用方给的 per-tensor `input_scale`（`assert input_scale.numel() == 1`），不再走 `static_quant_fp8`。
4. **scalar-A scale 的原生支持**（`:1754-1763`）：新增 `channelwise_cutlass` / `use_cutlass_channelwise_gemm` / `native_scalar_a_scale`。当硬件是 SM90/SM100/SM120 时，per-tensor 的 A scale 可**原样（scalar）**交给 CUTLASS kernel；否则（不支持的 epilogue）需 `repeat` 成 per-row（`:1768-1771`、`:1807-1809`）。这解释了标题为何强调 SM90/SM100/SM120。
5. **统一 output_dtype**：所有 GEMM 分支（`fp8_scaled_mm`、`triton_scaled_mm`、aiter、`torch._scaled_mm`、padding 分支）都由 `input.dtype` 改为 `output_dtype`。

### 38.4 数据流（提交后）

```
   ┌────────────── RMSNorm::forward_cuda (layernorm.py:469) ──────────────┐
   │ quant_linear 非空 且 _fp8_static_input_scale() 命中                   │
   │      │                                                               │
   │      ├─ 否 ──► 原路径：norm → bf16 激活 ──► apply_fp8_linear 内部     │
   │      │                                    static_quant_fp8           │
   │      └─ 是 ──► forward_with_per_tensor_quant_fusion                  │
   │                    (layernorm.py:858)                                │
   └──────────────────────────────┬──────────────────────────────────────┘
                                  ▼
        FlashInfer rmsnorm_quant / fused_add_rmsnorm_quant
                                  ▼
      (fp8_act, input_scale, orig_dtype) ──tuple──► Fp8LinearMethod::apply
                                  ▼                        (fp8.py:1035)
        apply_fp8_linear(input_prequantized=True) (fp8_utils.py:1713)
                                  ▼
                  fp8_scaled_mm(out_dtype=orig_dtype)
```

### 38.5 测试用例：有，且覆盖较细

两个测试文件都在提交里新增/扩充，并以 `register_cuda_ci(... stage="base-b", runner_config="1-gpu-large")` 注册进 CI（**需要 GPU，无 GPU 会 `SkipTest`**）。

**A. `test/registered/layers/test_layernorm_fusion.py`（新增，142 行）**

| 用例（类::方法） | 行号 | 覆盖点 |
|---|---|---|
| `TestRMSNormFp8QuantFusion::_run_fusion_test` | 31 | 公共断言：输出 dtype=fp8、`s is scale`、`out_dtype==dtype`、shape、residual 正确性、反量化后 cos>0.99 且 rel_err<0.1 |
| `TestRMSNormFp8QuantFusion::test_rms_norm_fp8_quant_fusion` | 80 | 参数扫描 NUM_TOKENS×HIDDEN_SIZES×ADD_RESIDUAL×DTYPES（7/83/512 × 512/4096 × {无/有 residual} × {bf16/fp16}），对比 `forward_native` |
| `TestRMSNormFp8QuantFusion::test_forward_cuda_quant_linear_dispatch` | 95 | **调度正确性**：monkeypatch `_fp8_static_input_scale`，验证普通 norm 走融合；而 `variance_size_override` 与 `cast_x_before_out_mul`（HF 语义）**必须不融合** |

**B. `test/registered/quant/test_fp8_utils.py`（扩充 +268 行）**

| 用例（类::方法） | 行号 | 覆盖点 |
|---|---|---|
| `TestApplyFp8LinearScaleDispatch::test_native_scalar_a_static_prequant_and_dynamic_scale_shapes` | 65 | 逐个 mock `_is_sm90/100/120_supported`，断言静态预量化、compressed-tensor 静态、动态三条路径传给 GEMM 的 A-scale 形状：native scalar 时 `numel()==1`，预量化时 `is input_scale`，动态时 `(M,1)` |
| `TestApplyFp8LinearScaleDispatch::test_without_native_scalar_a_static_scale_is_repeated` | 142 | 非 SM90/100/120 时，scalar scale 被 `repeat` 成 `(M,1)` |
| `TestApplyFp8LinearScaleDispatch::test_linear_methods_forward_fused_scalar_tuple` | 179 | `Fp8LinearMethod::apply` 与 `CompressedTensorsW8A8Fp8::apply_weights` 能接收 fused tuple，并把 `input_scale` / `pre_quant_output_dtype` 正确透传 |
| `TestApplyFp8LinearPrequantOutputDtype::test_prequant_output_dtype` | 306（helper `_run` 245） | **回归测试**：预量化输入必须输出调用方给的 dtype（fp16 & bf16），缺省才回 bf16，并与非预量化路径数值对齐。注释点明动机：*FP16 模型里硬编码 bf16 会导致 attention 的 query/key dtype 不匹配* |

**C. 性能基准（非单测）**：`benchmark/kernels/bench_fused_rmsnorm_fp8_quant.py`（新增 185 行）——`run_unfused`(67) / `run_fused_default`(87) / `run_fused_cute`(91) 三种实现对比，`_check_correctness`(128) 做数值校验，`benchmark`(176) 出吞吐数据。

### 38.6 小结与边界

- **核心**：RMSNorm 与静态 per-tensor FP8 激活量化融合为单个 FlashInfer kernel，省一次 kernel 启动与两趟激活 HBM 往返；仅在 SM90/SM100/SM120 + flashinfer 可用 + 下游是静态 per-tensor FP8 linear 时生效，否则静默走原路径。
- **覆盖范围**：Llama / Qwen2（含 EAGLE）主干已接线；其它模型只要把 `quant_linear=` 传给 `RMSNorm` 即可复用。
- **不融合的情况**：`cast_x_before_out_mul`（HF 语义）、`variance_size_override`、非 per-tensor（block/mxfp8/marlin）FP8 linear、非静态输入 scale、flashinfer 不可用。
- **易踩的坑**：预量化路径若不带 `orig_dtype`，GEMM 默认 bf16，在 FP16 模型上会引发 attention q/k dtype 不匹配——这正是 `test_prequant_output_dtype` 守护的点。

> 环境备注：`sglang_sipu` 在容器 `sipu-dev` 内挂载为 `/share/users/like/package/sglang_sipu -> /sgl-workspace/sglang`（workdir 同），镜像 `harbor.siorigin.com/sglang-sipu/release:v0.5.18-sipu-dev-0.1.0`；包名仍为 `sglang`（editable 安装）。

---

## 39. `static_quant_fp8` 的 scale 是现场按 max/min 算的，还是从 safetensors 加载的？

**结论：从 safetensors（checkpoint）加载的，不是现场按 tensor 的 max/min 算的。** `static_quant_fp8` 本身只是一个「用给定 scale 做量化」的 kernel，它不做任何 reduce/max 运算，scale 是它的**入参**。

### 39.1 证据一：kernel 签名与函数体都不算 scale

`static_quant_fp8`（`python/sglang/kernels/ops/quantization/fp8_kernel.py:902`）签名是 `(x, x_s, repeat_scale)`，`x_s` 由外部传入。函数体（`fp8_kernel.py:894-895`）直接：

```python
y_s_inv = 1.0 / y_s
y_q = tl.clamp(y * y_s_inv, fp8_min, fp8_max).to(FP8_DTYPE)
```

只有乘法 + clamp，没有任何求 max/absmax 的操作。它对应的 triton kernel `_static_quant_fp8` 同理。

### 39.2 证据二：scale 通过 weight_loader 从 checkpoint 加载

- **原生 FP8**：`Fp8LinearMethod::create_weights`（`python/sglang/srt/layers/quantization/fp8.py:610-626`）中，只有当 `activation_scheme == "static"` 时才注册
  ```python
  layer.register_parameter("input_scale", PerTensorScaleParameter(..., weight_loader=weight_loader))
  ```
  `weight_loader` 就是 HF checkpoint 的加载器，会把 safetensors 里名为 `...input_scale` 的张量读进来。`activation_scheme == "dynamic"` 时注册的是 `input_scale = None`（`fp8.py:626`），根本不加载 scale。
- **compressed-tensors**：`CompressedTensorsW8A8Fp8::create_weights`（`python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py:137-143`）在 `is_static_input_scheme` 时同样用 `PerTensorScaleParameter(weight_loader=...)` 从 checkpoint 加载。

### 39.3 证据三：加载后有一次 `.max()` 归约（仍是 checkpoint 的值）

`process_weights_after_loading` 里会做 `layer.input_scale = Parameter(layer.input_scale.max(), ...)`（`fp8.py:946-948`；compressed-tensors 在 `compressed_tensors_w8a8_fp8.py:223-224`）。因为 checkpoint 可能按 shard/channel 存了多个 scale，这里取 max 归约成一个 per-tensor 标量——注意这是对**加载进来的值**做归约，不是重新按激活统计量计算。

### 39.4 对比：dynamic 才是现场算的

| 维度 | static | dynamic |
|---|---|---|
| scale 来源 | safetensors 加载（校准阶段算好并写入 checkpoint） | 推理时现场按激活统计量算 |
| 代码路径 | `input_scale` 非 None → `static_quant_fp8`（`fp8_utils.py:1951`） | `input_scale is None` → `sglang_per_token_quant_fp8(input_2d)`（`fp8_utils.py:1959`） |
| kernel 行为 | 只做 `x * (1/scale)` + clamp | `per_token_quant_fp8`（`python/sglang/kernels/ops/quantization/per_token_quant_fp8.py:43`），docstring 明写 "Dynamically quantize each row"，在 kernel 内对每行求 max |

即：quantization 术语里 static / dynamic 的差别，就是「激活 scale 是否在量化校准阶段预先确定并写进 checkpoint」。

### 39.5 与第 38 节的衔接

第 38 节里 `_fp8_static_input_scale`（`layernorm.py:370`）读的 `linear.input_scale`，正是这个从 safetensors 加载、再 `.max()` 归约后的 per-tensor 标量。RMSNorm + FP8 量化融合 kernel 只是**复用它**传给 FlashInfer，并不会重新计算——这也正是 `_fp8_static_input_scale` 要求 `input_scale.numel() == 1` 的原因。

---

## 40. Engine 的 `attention_backend="sipu"` 与 `device="sipu"` 分别作用在哪里？

### 40.1 `attention_backend="sipu"`：最终实现在哪

**调用链（Engine → 实现）**：

1. `Engine::__init__`（`python/sglang/srt/entrypoints/engine.py:232`）是 `def __init__(self, **kwargs)`，把 kwargs 原样构造成 `ServerArgs`（`engine.py:251`），所以 `attention_backend="sipu"` 直接落进 `ServerArgs`。
2. `ServerArgs.attention_backend` 字段（`python/sglang/srt/server_args.py:1701`）。`ServerArgs::_run_resolution_pipeline`（`server_args.py:3591`）在 `server_args.py:3670` 调用 `ServerArgs::_handle_sipu_backends`（定义在 `server_args.py:4390`）。
3. 运行时装配：`ModelRunner::init_attention_backends`（`python/sglang/srt/model_executor/model_runner.py:931`）
   → `resolve_attention_backend_strs`（`python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:158`）
   → `build_attention_backends`（`attention_backend_setup.py:69`）
   → `_build_resolved_backend`（`:181`）→ `_build_backend_from_str`（`:238`）→ `_build_full_attention_backend_from_str`（`:251`）
   → `ATTENTION_BACKENDS[backend_str](model_runner)`（`attention_backend_setup.py:257`）。
4. 注册表：`register_attention_backend`（`python/sglang/srt/layers/attention/attention_registry.py:34-39`）把名字写进 `ATTENTION_BACKENDS`（`:31`）；`"sipu"` 对应 `create_sipu_backend`（`attention_registry.py:133`）。

**最终实现**：`create_sipu_backend`（`attention_registry.py:133`）按「模型里是否存在 `indexer` 子模块」二选一：

- 有 `indexer`（DSA / 稀疏 MLA 类模型）→ `SIPUDSAAttnBackend`，实际类体是 `DeepseekSparseAttnBackend`（`python/sglang/srt/hardware_backend/sipu/attention/sipu_dsa_backend.py:244`，文件末尾 `:1618` 起了别名 `SIPUDSAAttnBackend = DeepseekSparseAttnBackend`）。
- 无 `indexer`（本次 smoke 用的 Llama-3.1-8B-4layer 就是这种）→ **`SIPUAttnBackend`**（`python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:95`）。

`SIPUAttnBackend` 的关键成员（相对路径 + 行号 + `类::方法`）：

| 成员 | 行号 | 作用 |
|---|---|---|
| `SIPUAttnBackend::__init__` | `.../sipu_flashattention_backend.py:113` | 持有 `model_runner.device`、`token_to_kv_pool`、`req_to_token`、`page_size` 等 |
| `SIPUAttnBackend::init_forward_metadata` | `:384` | 构造 `SIPUAttentionMetadata`（`:43`，含 `page_table` / `cache_seqlens_int32` / `cu_seqlens_q/k` / `scheduler_metadata`） |
| `SIPUAttnBackend::forward_extend` | `:798` | prefill / extend |
| `SIPUAttnBackend::forward_decode` | `:1297` | decode |
| `SIPUAttnBackend::init_cuda_graph_state` | `:1609` | CUDA graph 状态 |
| `SIPUAttentionMultiStepBackend` | `:2803` | 投机解码 draft worker 用 |

底层算子来自 `sgl_kernel`：`flash_attn_varlen_func` / `flash_attn_with_kvcache` / `merge_state_v2`（import 在 `sipu_flashattention_backend.py:33-37`）。也就是说，`attention_backend="sipu"` 最终落到 **sipu 定制版的 flash-attention backend**（把 FA3 风格的 metadata / page-table 逻辑接到 sipu 设备，底层调 sikernel 的 flash attention）。

### 40.2 `device="sipu"`：影响哪些 tensor 的分配

`ServerArgs.device`（`server_args.py:1214`）→ `ModelRunner.device = server_args.device`（`python/sglang/srt/model_executor/model_runner.py:309`）。

**结论：模型权重和 KV cache 都受它控制**，此外还有一批配套组件。

**(1) 模型权重 —— 受控**

`ModelRunner::load_model`（`model_runner.py:1061`）→ `load_model_with_memory_saver(device=self.device, ...)`（`model_runner.py:1111-1119`）→ `DeviceConfig(device, gpu_id)`（`python/sglang/srt/model_executor/model_runner_components/load_model_utils.py:305`）→ `DeviceConfig::__init__` 里 `self.device = torch.device(self.device_type)`（`python/sglang/srt/configs/device_config.py:17-23`；`"sipu"` 在 `SUPPORTED_DEVICES`，`device_config.py:10`）→ loader 用 `with torch.device(device_config.device):` 把权重直接建在 sipu 上（`python/sglang/srt/model_loader/loader.py:1664`、`:1792`、`:3175`、`:3332`、`:3639`）。

**(2) KV cache —— 受控**

`ModelRunner::init_kv_cache_configurator`（`model_runner.py:587-590`）把 `device=self.device` 传给 `KVCacheConfigurator`（`python/sglang/srt/mem_cache/kv_cache_configurator.py:213`），再由它传给各 pool 构造函数：

- `KVCacheConfigurator::_build_oot_mha_kv_pool`（`kv_cache_configurator.py:1175`，`device=self.device` 在 `:1186`）
- `KVCacheConfigurator::_build_mha_kv_pool`（`:1570`，`device=self.device` 在 `:1596`）
- `KVCacheConfigurator::_build_default_req_pool`（`:924`，`device=self.device` 在 `:943`，即 `req_to_token_pool`）

这些 pool 在 `python/sglang/srt/mem_cache/memory_pool.py` 内用 `device=device` / `device=self.device` 建 `torch.empty` / `torch.zeros`（例如 `memory_pool.py:540`、`:556`、`:582`、`:1266`、`:1273`、`:1300`）。

**(3) 其它同样受 `device` 影响的分配 / 行为**

- 当前设备切换：`torch.get_device_module(self.device).set_device(ps.gpu_id)`（`model_runner.py:386`）。
- 分布式后端选择：`_DEVICE_TO_DISTRIBUTED_BACKEND["sipu"] = "gloo"`（`python/sglang/srt/platforms/device_mixin.py:95`）。
- 其它组件：`WeightUpdater`（`model_runner.py:538-549`）、`NgramEmbeddingManager`（`:577-585`）、`RoutedExpertsCapturer`（`:1006-1021`）、indexer capturer（`:1023-1031`）、`init_torch_distributed`（`:1041-1050`）。
- 平台默认参数：`ServerArgs::_handle_sipu_backends`（`server_args.py:4390`）调 `set_default_server_args`（`python/sglang/srt/hardware_backend/sipu/utils.py:30-34`，目前只设 `page_size=32`），并把 prefill 的 `tc_compiler` 强制为 `"eager"`（`server_args.py:4396-4402`）。

**(4) 取值与自动探测**

不传 `device` 时由 `get_device()`（`python/sglang/srt/utils/common.py:916`）自动探测，只有 `is_sipu()`（`utils/common.py:194`，判据是 `hasattr(torch, "sipu")` 且 `torch.sipu.is_available()`）为真才返回 `"sipu"`（`utils/common.py:942-945`）；显式传参则跳过探测（`server_args.py:4265-4268`）。

### 40.3 一句话总结

- `attention_backend="sipu"`：经 `ServerArgs.attention_backend` → `ATTENTION_BACKENDS["sipu"]`（`attention_registry.py:133`）分发，无 `indexer` 的模型最终用 `SIPUAttnBackend`（`python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:95`），底层是 `sgl_kernel` 的 flash attention 算子。
- `device="sipu"`：经 `ModelRunner.device`（`model_runner.py:309`）扩散，**模型权重**（`DeviceConfig` + `with torch.device(...)`）和 **KV cache / req_to_token_pool**（`KVCacheConfigurator` → `memory_pool.py` 的 `torch.empty(device=...)`）都由它决定，分布式后端也随之选为 gloo。

> 附注（与上面两问无直接关系，但和本次 env 有关）：`temp/env-offlie-infer.sh` 里设了 `SGLANG_PLUGINS="mywhite"`。`Engine::__init__` 在构造 `ServerArgs` 前会调 `load_plugins()`（`engine.py:240`），而 `load_plugins_by_group`（`python/sglang/srt/plugins/__init__.py:35`）会用该白名单过滤（`:97-99`）。在该容器里 `entry_points(group="sglang.srt.plugins")` 只有 `sglang_simo_extensions -> simo.extensions.sglang_simo:register_simo_extensions` 一个，名字不等于 `mywhite`，因此会被跳过。若本意是启用 simo 扩展，白名单值可能需要改成 `sglang_simo_extensions`（或置空）。

---

## 41. 提交 11d03eaeef 真的减少 kernel launch 吗？量化产出的 scale 会写 global memory 吗？

> 行号基于当前工作区（HEAD `72f9ec9d7`）。第 38/39 节用的是提交 11d03eaeef 自身的行号，`fp8_utils.py` 有约 +130 行偏移；`fp8.py` 无偏移。

### 41.1 核心问题的直接回答

**要分两种情况，结论相反：**

| 路径 | 提交前 bf16→fp8 量化，产出的 scale 会写 global memory 吗 | 依据 |
|---|---|---|
| **静态 per-tensor**（本提交唯一覆盖的路径） | **不会 —— 严格说根本没有「产出的 scale」**。scale 是 checkpoint 加载的 per-tensor 常量，kernel 只**读**它。只有 `REPEAT_SCALE=True` 时才会把 `(M,1)` 的副本写回 global memory | `_static_quant_fp8`(`python/sglang/kernels/ops/quantization/fp8_kernel.py:851`)：`y_s = tl.load(y_s_ptr)`(`:889`) 读 scale；`tl.store(y_q_ptr + cols, ...)`(`:897`) 写 fp8；`if REPEAT_SCALE: tl.store(y_s_repeat_ptr, y_s)`(`:898-899`) 条件写 scale |
| **动态 per-token**（本提交**不**覆盖） | **会**。kernel 现场 reduce max 算出 scale 并 `output_s[token_id] = scale` 写回 global memory | `per_token_quant_fp8_warp_kernel`(`python/sglang/kernels/jit/csrc/gemm/per_token_quant_fp8.cuh:20`)：`:51` 算 `scale = reduce_max(...)/FP8_E4M3_MAX`，`:53` 写 `output_s[token_id]` |

**所以：你问的「产出的 scale 进 global memory」，在静态 per-tensor 这条路上本来就不存在「产出」——它是常量；真正进 global memory 的是 fp8 激活本身。而「现场算 scale 并写回 global memory」那种情况（动态 per-token），这个提交根本没碰。**

换句话说：`static_quant_fp8`(`fp8_kernel.py:902`) 这个函数名里的 "static" 就说明 scale 是外部给定（checkpoint）而非计算所得；它 `repeat_scale=False` 时直接把入参 scale 原样返回（`fp8_kernel.py:957`）。

### 41.2 kernel launch：3 → 2，省掉的是 bf16 往返，不是 fp8 往返

**提交前**（静态 per-tensor FP8 linear）：

```
1. RMSNorm kernel（sgl_kernel 的 rmsnorm / fused_add_rmsnorm）
      └─ 写 bf16 激活 ──► global memory
2. static_quant_fp8（triton，fp8_kernel.py:851）
      └─ 读 bf16，写 fp8 激活（+ 条件写 scale）
3. fp8_scaled_mm（cutlass，python/sglang/kernels/ops/gemm/__init__.py:86）
      └─ 读 fp8 激活 + scales
= 3 个 kernel
```

**提交后**：

```
1. rmsnorm_quant / fused_add_rmsnorm_quant（flashinfer，layernorm.py:957 / :949）
      └─ 读 bf16，写 fp8 激活
2. fp8_scaled_mm
      └─ 读 fp8 激活 + scales
= 2 个 kernel
```

所以：
- ✅ **少 1 次 kernel launch**（3→2），这是真实的。
- ✅ 少 1 次 **bf16 激活**的 global memory 写 + 读（norm 的输出不再落地）。
- ❌ **fp8 激活仍然要写 global memory 再被 GEMM 读**（量化产物仍然落地）。
- ❌ scale 仍然在 global memory 里（见 41.1），并没有被"省掉"。

### 41.3 你说对的部分：这个提交没有把 quant 融进 GEMM

你的设想「bf16→fp8 量化 + fp8 tensorcore MMA 放进一个 kernel，scale 留在 register/SMEM」在原理上确实更优——能同时省掉 fp8 激活的往返和 scale 的 global memory 访问。**但 sglang 现有的 GEMM 后端不做量化，所以这个提交做不到这件事：**

- `fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None)`（`python/sglang/kernels/ops/gemm/__init__.py:86`）要求 `mat_a` 已经是 fp8。
- 连 flashinfer 的 bmm 路径也是两步：`apply_fp8_linear_bmm_flashinfer`（`python/sglang/srt/layers/quantization/fp8_utils.py:1827`）先 `static_quant_fp8(...)`(`:1837`) 再 `flashinfer_bmm_fp8(qinput, ...)`(`:1838`)。
- 要把量化塞进 GEMM，需要给 CUTLASS/tensorcore GEMM 写 A 的 prologue（load bf16 → 乘 scale → 转 fp8 → 进 SMEM），那是改 GEMM kernel 本身，工作量和风险大得多。

这个提交选的是上游更容易的一刀：flashinfer 已经提供 `rmsnorm_quant` / `fused_add_rmsnorm_quant`，把 **norm 和 quant** 合成一个 kernel，与现有所有 GEMM 后端正交组合。它是「3 个 kernel 里合并掉 2 个」的**部分**优化，不是「量化+MMA 全融」的**完全**优化。

### 41.4 但"完全没有意义"不成立

- 对 decode（M 很小）场景，quant kernel 是 launch-latency 主导，少一次 launch 的收益是可测的；同时省掉一次 bf16 激活的 HBM 往返。
- 它还顺带修了 `pre_quant_output_dtype` 的 dtype 传播问题（见 38.5 的回归测试：FP16 模型上硬编码 bf16 会导致 attention q/k dtype 不匹配）。
- **真正的局限**：它只覆盖静态 per-tensor（`_fp8_static_input_scale` 要求 `input_scale.numel() == 1`，`python/sglang/srt/layers/layernorm.py:375`），所以对动态 per-token 零收益——那里 scale 必须现场算，除非把量化融进 GEMM（正是你设想的那条路）。
- 顺带一提，`fp8_utils.py:1915-1918` 的注释点出了另一条思路：compressed-tensors 路径在 `tc_compiler="inductor"` 时用纯 PyTorch 的 `(input * scale.reciprocal()).clamp().to(fp8)`，靠 inductor 把它和周围 RMSNorm/residual 融合掉，效果与这个提交类似（都是消除一次 launch，但 fp8 仍落地）。

### 41.5 一个容易忽略的细节：`repeat_scale`

在 SM90/SM100/SM120 之外（或 GEMM epilogue 不支持 scalar-A），`native_scalar_a_scale=False`（`fp8_utils.py:1898`），于是：

- **提交前**：`static_quant_fp8(..., repeat_scale=channelwise_cutlass and not native_scalar_a_scale)`（`fp8_utils.py:1954`），在 triton kernel 里**写一份 `(M,1)` 的 scale** 到 global memory。
- **提交后**：融合 kernel 不写 scale，但 `apply_fp8_linear` 的预量化分支里 `x_scale = input_scale.repeat(input_2d.shape[0]).view(-1, 1)`（`fp8_utils.py:1907`）——**又用一次 torch 算子把 `(M,1)` 物化出来**（`Tensor.repeat` 本身也是一次 kernel launch）。

所以在非 SM90/100/120 上，这个提交省下的 launch 数可能被 `.repeat()` 部分抵消。这也解释了提交标题为什么强调 **SM90 / SM100 / SM120**：只有这些平台 `native_scalar_a_scale=True`（`fp8_utils.py:1898-1900`），scalar-A 才能原样交给 CUTLASS，才真正省下 scale 的物化。

### 41.6 结论

1. **减少 kernel launch：真的减了，但只有 1 次（3→2）**，省掉的是 RMSNorm→quant 之间的 bf16 激活往返；**fp8 激活仍然往返 global memory**。
2. **提交前静态 per-tensor 的量化不写 scale**（scale 是 checkpoint 常量，只读）；**动态 per-token 才写 scale**，而那个路径本提交不覆盖。
3. 你的批评在「没把量化融进 GEMM、因此没能把 scale 留在 register/SMEM」这一点上成立；但「完全没有意义」不成立——省一次 launch + 省一次 bf16 往返 + 修 dtype 传播，对 decode 是有实际收益的，只是它不是最优融合。

---

## 42. `/share/liwang/` 空间占用统计（cache / temp / 数据集 / 模型权重）

### 42.1 背景

```
$ df -h /share
Filesystem            Size  Used Avail Use% Mounted on
10.97.128.245:/share   78T   78T  2.3G 100% /share     # NFS4.2, 已 100% 满
```

统计方法：`du -sh` / `du -h --max-depth=1`（NFS 上按 apparent size，含硬链接不去重）。本次统计范围 **仅 `/share/liwang/`**。

**`/share/liwang/` 自身合计约 8.3 TB**（占 78T 全盘的一小部分，说明 `/share` 满不全由 liwang 造成，但下面这些目录确实是 liwang 名下的大头）。

### 42.2 一级目录排行

| 目录 | 大小 | 类别 |
|---|---|---|
| `/share/liwang/VLM_projects` | **5.0T** | 项目/模型权重 |
| `/share/liwang/Projects` | **2.6T** | 项目/模型权重 |
| `/share/liwang/JD_evaluation` | **384G** | 项目/数据集/缓存 |
| `/share/liwang/envs` | **161G** | conda 环境 |
| `/share/liwang/Benchmark` | **101G** | 评测框架/缓存 |
| `/share/liwang/nim-cache-v2` | 30G | **cache** |
| `/share/liwang/nim-cache` | 16G | **cache** |
| `/share/liwang/.conda` | 15G | **cache**（pkgs 15G） |
| `/share/liwang/.cache` | 3.3G | **cache** |
| `/share/liwang/pip_cache` | 1.5G | **cache** |
| `/share/liwang/.trae-cn-server` | 1.4G | 工具 |
| `/share/liwang/tools` | 670M | 工具 |
| `/share/liwang/workspace-20260907-025142` | 79M | 临时工作区 |
| `/share/liwang/perf-test` | 39M | 测试 |
| `/share/liwang/.agents` | 4.3M | 工具 |
| `/share/liwang/xdg_cache` | 148K | **cache** |
| 其余（`=24`、`test`、`.bashrc`、`.vimrc`、`.npm`、`.claude` 等） | < 20K | — |

### 42.3 按类别归类

#### (1) 模型权重 / checkpoints —— 绝对大头（约 8.2T）

| 目录 | 大小 |
|---|---|
| `/share/liwang/VLM_projects/Griffon/GThinker/EasyR1/checkpoints_RL` | **2.3T** |
| `/share/liwang/VLM_projects/Griffon/GThinker/EasyR1/checkpoints_RL_0429` | **839G** |
| `/share/liwang/Projects/LlamaGen/checkpoints_vqvae` | **839G** |
| `/share/liwang/Projects/LlamaGen/results_tokenizer_image` | **839G**（中间产物） |
| `/share/liwang/VLM_projects/Griffon/GThinker/SFT_code/checkpoints-0324` | **814G** |
| `/share/liwang/VLM_projects/ml-fastvlm/checkpoints-distill` | **469G** |
| `/share/liwang/VLM_projects/Griffon/GThinker/SFT_code/checkpoints-0429-v2` | **302G** |
| `/share/liwang/VLM_projects/ml-fastvlm/checkpoints` | 109G |
| `/share/liwang/VLM_projects/ml-fastvlm/checkpoints-distill-liwang` | 102G |
| `/share/liwang/VLM_projects/Griffon/GThinker/SFT_code/checkpoints-0429-v2-1` | 55G |
| `/share/liwang/VLM_projects/Griffon/GThinker/SFT_code/checkpoints` | 20G |
| `/share/liwang/JD_evaluation/NavDP/checkpoints` | 521M |

> 层级：`VLM_projects/Griffon(4.3T)/GThinker(4.3T)` = `EasyR1(3.1T)` + `SFT_code(1.2T)`；`Projects/LlamaGen` 总 1.8T（其中 checkpoints_vqvae 839G）。

#### (2) 数据集 —— 相对不大（约 47G）

| 目录 | 大小 |
|---|---|
| `/share/liwang/JD_evaluation/datasets` | 42G |
| `/share/liwang/Projects/Dataset` | 3.8G |
| `/share/liwang/VLM_projects/Griffon/GThinker/EasyR1/split_datasets` | 922M |
| `/share/liwang/JD_evaluation/Mask2Former/datasets` | 2.1M |
| `Projects/JiT*/dataset`（3 个） | 各 4.0K |

#### (3) cache —— 约 115G

| 目录 | 大小 |
|---|---|
| `/share/liwang/Benchmark/.cache` | **28G** |
| `/share/liwang/JD_evaluation/JV-Adas/.sam3d-cache` | **21G** |
| `/share/liwang/nim-cache-v2` | **30G** |
| `/share/liwang/nim-cache` | **16G** |
| `/share/liwang/.conda/pkgs` | 15G |
| `/share/liwang/Benchmark/evalscope_main/.cache` | 14G |
| `/share/liwang/.cache`（含 `.cache/vllm/torch_compile_cache` 2.9G） | 3.3G |
| `/share/liwang/JD_evaluation/JV-Adas/.sam3-cache` | 2.4G |
| `/share/liwang/pip_cache` | 1.5G |
| `/share/liwang/xdg_cache` | 148K |
| `/share/liwang/Benchmark/.triton/cache` | 1.9M |

#### (4) temp / tmp —— **没有**

在 `/share/liwang/` 下 `find -maxdepth 3` 未发现名为 `tmp` / `temp` 的目录。最接近的是 `/share/liwang/workspace-20260907-025142`（79M，临时工作区，可确认后清理）。

#### (5) conda 环境（非 cache，但占空间）

`/share/liwang/envs` 161G，按大小：`navdp_isaaclab` 22G、`bench_vllm` 16G、`bench_sglang` 14G、`griffon` 13G、`navdp_rtx` 12G、`evalscope` 12G、`vllm` 11G、`vlmevalkit` 9.3G、`giga_brain_0` 9.0G、`jit` 8.6G、`llamagen` 7.9G、`fastvlm` 7.4G、`clip-eval` 6.0G、`sail7b` 5.5G、`torch_sipu` 4.8G、`torch_sipu_final` 4.2G 等。

### 42.4 完整钻取树（Top 分支）

```
/share/liwang/                                   ≈ 8.3T
├── VLM_projects/                                5.0T
│   ├── Griffon/                                 4.3T
│   │   ├── GThinker/                            4.3T
│   │   │   ├── EasyR1/                          3.1T
│   │   │   │   ├── checkpoints_RL/              2.3T   ← 模型权重
│   │   │   │   └── checkpoints_RL_0429/         839G   ← 模型权重
│   │   │   └── SFT_code/                        1.2T
│   │   │       ├── checkpoints-0324/            814G   ← 模型权重
│   │   │       ├── checkpoints-0429-v2/         302G   ← 模型权重
│   │   │       └── checkpoints-0429-v2-1/        55G
│   │   └── Griffon-G&R/                          19G
│   ├── ml-fastvlm/                              683G
│   │   ├── checkpoints-distill/                 469G   ← 模型权重
│   │   ├── checkpoints/                         109G
│   │   └── checkpoints-distill-liwang/          102G
│   ├── VLMEvalKit/                              4.9G
│   └── Qwen3-VL/                                4.3G
├── Projects/                                    2.6T
│   ├── LlamaGen/                                1.8T
│   │   ├── checkpoints_vqvae/                   839G   ← 模型权重
│   │   ├── results_tokenizer_image/             839G   ← 中间产物
│   │   └── imagenet_code_c2i_flip_ten_crop_105/  69G   ← 中间产物
│   ├── JiT_advanced/                            386G
│   ├── JiT_Advanced_HD/                         163G
│   ├── JiT_local_attention/                     147G
│   ├── pdf_to_images/                           111G   ← 中间产物
│   ├── test_torch_sipu/                17G
│   ├── JiT/                                      16G
│   └── Dataset/                                 3.8G   ← 数据集
├── JD_evaluation/                               384G
│   ├── JV-Adas/                                 281G
│   │   ├── .sam3d-cache/                         21G   ← cache
│   │   └── .sam3-cache/                         2.4G   ← cache
│   ├── datasets/                                 42G   ← 数据集
│   ├── JD_evaluation_clean/                      38G
│   ├── ultralytics/                             7.0G
│   ├── .agentos/                                6.7G   ← cache
│   └── NavDP/                                   3.0G
├── envs/                                        161G   ← conda 环境
├── Benchmark/                                   101G
│   ├── evalscope_main/                           36G（含 .cache 14G）
│   ├── .cache/                                   28G   ← cache
│   ├── evalscaope_main_v1.9.1/                   23G
│   ├── .agentos/                                 11G   ← cache
│   └── Perf_test/ venv/                          1.3G / 1.1G
├── nim-cache-v2/                                 30G   ← cache
├── nim-cache/                                    16G   ← cache
├── .conda/pkgs/                                  15G   ← cache
├── .cache/                                      3.3G   ← cache
├── pip_cache/                                   1.5G   ← cache
└── .trae-cn-server/ tools/ …                    1.4G / 670M
```

### 42.5 可清理候选（仅列出，**未执行任何删除**）

按「回收空间 / 风险」排序，供 liwang 确认：

| 候选 | 大小 | 说明 |
|---|---|---|
| `VLM_projects/Griffon/GThinker/EasyR1/checkpoints_RL_0429` | 839G | RL 旧版本 checkpoint，确认不再用可删 |
| `Projects/LlamaGen/checkpoints_vqvae` | 839G | VQVAE 权重，若已有备份可删 |
| `Projects/LlamaGen/results_tokenizer_image` | 839G | 看起来是 tokenizer 训练中间结果 |
| `VLM_projects/Griffon/GThinker/SFT_code/checkpoints-0324` | 814G | 3 月旧 checkpoint |
| `VLM_projects/ml-fastvlm/checkpoints-distill` | 469G | 蒸馏中间 checkpoint |
| `VLM_projects/Griffon/GThinker/SFT_code/checkpoints-0429-v2` | 302G | 旧版本 |
| `Projects/pdf_to_images` | 111G | 看起来是中间产物 |
| `nim-cache-v2` + `nim-cache` | 46G | NIM 模型缓存，可重建 |
| `Benchmark/.cache` | 28G | 评测缓存，可重建 |
| `JD_evaluation/JV-Adas/.sam3d-cache` | 21G | 模型缓存，可重建 |
| `.conda/pkgs` | 15G | `conda clean -a` 可回收 |
| `Benchmark/evalscope_main/.cache` | 14G | 可重建 |
| `.cache/vllm/torch_compile_cache` | 2.9G | torch.compile 缓存，可重建 |
| `pip_cache` | 1.5G | `pip cache purge` 可回收 |
| `workspace-20260907-025142` | 79M | 临时工作区 |

**注意**：`/share` 总盘 78T 已 100% 满，liwang 名下约 8.3T。清理 liwang 只能释放其中一部分；若目标是让 `/share` 恢复可用，还需排查其它用户的占用（本次未做，因为任务只指定了 `/share/liwang/`）。

### 42.6 备注

- 本次所有统计均为**只读操作**（`du` / `find`），未删除、未移动任何文件。
- NFS 上 `du` 很慢（大量小文件 + 网络 stat），单目录 5T 级遍历耗时数分钟到十几分钟。
- 未发现 `tmp` / `temp` 目录；最大的「缓存类」目录是 `Benchmark/.cache`(28G)、`JV-Adas/.sam3d-cache`(21G)、`nim-cache*`(46G)。

---

## 43. `/share/` 下 9 个用户 + `/share/users/` 的空间占用统计

### 43.0 背景与方法

```
$ df -h /share
Filesystem            Size  Used Avail Use% Mounted on
10.97.128.245:/share   78T   78T  2.3G 100% /share     # NFS4.2, 已 100% 满
```

统计方式：`du -sh` / `du -h --max-depth=1`；查找 cache/temp/tmp/数据集/权重类目录用 `find -maxdepth 3`。**全程只读**（`du` / `find`），未删除、未移动任何文件。

---

### 43.1 `/share/{yufne,gyzhou,weihongyang,guorui,pengkunfu,caolujing,mtang,huayicong,songzun}`

#### 总览（按大小排序）

| 用户目录 | 总大小 |
|---|---|
| `/share/yufne` | **8.7T** |
| `/share/weihongyang` | **4.0T** |
| `/share/guorui` | **3.7T** |
| `/share/pengkunfu` | **1.6T** |
| `/share/caolujing` | **1.6T** |
| `/share/mtang` | **1.4T** |
| `/share/huayicong` | **1.2T** |
| `/share/songzun` | **502G** |
| `/share/gyzhou` | 12K（基本为空） |
| **合计** | **≈ 21.7T** |

#### (A) 模型权重 / checkpoints

| 目录 | 大小 |
|---|---|
| `/share/guorui/投机解码训练/SpecForge` | **1.7T** |
| `/share/weihongyang/JDJV` | **1.2T** |
| `/share/weihongyang/ml-fastvlm/output` | **1.2T** |
| `/share/pengkunfu/JDJV/GaussianOcc` | **1.1T** |
| `/share/mtang/work/JD` | **430G** |
| `/share/mtang/work/Gen` | **353G** |
| `/share/guorui/SpecForge`（workspace 内） | 80G |
| `/share/huayicong/proj`（含 torch_sipu 73G、onnxruntime-ep 49G） | 359G（合计） |
| `/share/yufne/Moore-AnimateAnyone` | 308G |
| `/share/yufne/AnyEdit` | 279G |
| `/share/yufne/Open-AnimateAnyone` | 223G |
| `/share/yufne/Qwen3-VL-30B-A3B-Instruct` | 58G |
| `/share/weihongyang/ml-fastvlm/checkpoints` | 40G |

#### (B) 数据集

| 目录 | 大小 |
|---|---|
| `/share/yufne/UltraEdit` | **1.5T** |
| `/share/yufne/Senorita` | **1.5T** |
| `/share/yufne/JourneyDB` | **1.5T**（其 `data/` 1.5T） |
| `/share/yufne/ShareGPT4Video` | **1.4T** |
| `/share/yufne/Animate_project` | 758G |
| `/share/caolujing/data` | **587G**（`images` 245G、`imagenet` 151G、`vae-sd` 50G） |
| `/share/yufne/dataset` | 516G |
| `/share/yufne/LLaVA-Video-178K` | 465G |
| `/share/yufne/M4-Instruct-Data` | 219G |
| `/share/guorui/datasets` | 149G |
| `/share/weihongyang/JDJV-SIMO-Qwen3-VL` | 101G |
| `/share/yufne/pixmo-docs` | 52G |
| `/share/yufne/OmniVideo11B` | 47G |
| `/share/caolujing/work/guided-diffusion/datasets` | — |
| `/share/yufne/Bagel/data/interleave_datasets` | — |

#### (C) cache / temp / tmp

| 目录 | 大小 | 说明 |
|---|---|---|
| `/share/mtang/.cache` | **344G** | 本次 9 用户中最大的 cache（含 vllm / nvidia / nim / huggingface） |
| `/share/weihongyang/evalscope_latest` | **651G** | 评测工作副本（含缓存） |
| `/share/weihongyang/evalscope` | 115G | |
| `/share/guorui/home_cache` | 79G | 含 `tvm-ffi/` `sgl_kernel_jit_*`、`deep_gemm/`、`vllm/torch_compile_cache` |
| `/share/guorui/model_cache` | 71G | |
| `/share/guorui/hf_cache` | 60G | HuggingFace 数据集缓存 |
| `/share/guorui/.cache` | 29G | |
| `/share/guorui/conda_cache` | 12G | |
| `/share/guorui/.codex/cache`, `.codex/plugins/cache` | — | |
| `/share/songzun/uv-cache-sz` | 80G | uv 包缓存 |
| `/share/huayicong/.cache` | 20G | 含 `ccache/` |
| `/share/huayicong/.codex-local/*/tmp` | 867 个 tmp 目录 | 每个都是小目录，总计 ~6G（`find` 命中 867 处） |
| `/share/songzun/.codex/tmp`, `cosmos-*/__pycache__` | — | |
| `/share/weihongyang/tmp` | 145M | 含 `cutlass_python_cache`、`simo_debug_cache` |
| `/share/weihongyang/.cache_home` | 1.5G | |
| `/share/caolujing/.cache` | 2.4G | 含 `hub/checkpoints` |
| `/share/pengkunfu/.triton/cache`, `/share/pengkunfu/.cache` | 36M / 4K | |
| **temp/tmp 类** | — | `weihongyang/tmp`(145M)、`songzun/.codex/tmp`、`huayicong/.codex*/*/tmp`(867个)、`pengkunfu/tmp`(4K)、`guorui/.codex/tmp`、`mtang/work/tmp`(95M)、`mtang/.cache/nim/tmp`、`caolujing` 无 |

#### (D) conda 环境 / 解释器（非 cache，但占空间）

| 目录 | 大小 |
|---|---|
| `/share/weihongyang/miniconda3` | 379G |
| `/share/guorui/envs` | 100G |
| `/share/yufne/miniconda3` | 59G |
| `/share/pengkunfu/anaconda3` | 43G |
| `/share/caolujing/miniconda3` | 37G |
| `/share/huayicong/miniconda3` | 34G |
| `/share/mtang/miniforge3` | 27G |
| `/share/songzun/miniconda3` | 17G |

#### (E) 各用户完整明细（Top 分支）

```
/share/yufne  8.7T
├── UltraEdit/ 1.5T        数据集
├── Senorita/ 1.5T         数据集
├── JourneyDB/ 1.5T        数据集
├── ShareGPT4Video/ 1.4T   数据集
├── Animate_project/ 758G  模型权重
├── dataset/ 516G          数据集
├── LLaVA-Video-178K/ 465G 数据集
├── Moore-AnimateAnyone/ 308G
├── AnyEdit/ 279G
├── Open-AnimateAnyone/ 223G
├── M4-Instruct-Data/ 219G
├── Qwen3-VL-30B-A3B-Instruct/ 58G
├── miniconda3/ 59G
└── pixmo-docs/ 52G, OmniVideo11B/ 47G, …

/share/weihongyang  4.0T
├── ml-fastvlm/ 1.4T （output/ 1.2T, checkpoints/ 40G）
├── JDJV/ 1.2T             ← 模型权重
├── evalscope_latest/ 651G ← 评测工作副本/缓存
├── miniconda3/ 379G
├── evalscope/ 115G
├── JDJV-SIMO-Qwen3-VL/ 101G
├── JD_Project/ 59G, simo-Qwen3-VL/ 58G, LMUData/ 32G, .agentos/ 24G, tmp/ 145M

/share/guorui  3.7T
├── 投机解码训练/ 1.8T （SpecForge/ 1.7T, SpecForge_new/ 108G）
├── workspace/ 642G （早期注入/ 290G, 动态锚点/ 163G, sd/ 118G, tgl/ 65G）
├── 投机解码优化/ 173G
├── datasets/ 149G         ← 数据集
├── envs/ 100G
├── SpecForge/ 80G
├── home_cache/ 79G        ← cache（tvm-ffi/deep_gemm/torch_compile）
├── model_cache/ 71G       ← cache
├── qwen3-next/ 61G
├── hf_cache/ 60G          ← cache
├── claude/ 49G
├── .cache/ 29G, .codex/ 17G, DSA实现-backup/ 13G, conda_cache/ 12G

/share/pengkunfu  1.6T
├── JDJV/ 1.2T （GaussianOcc/ 1.1T, datasets/ 46G, OPUS/ 11G）
├── edge_llm_models/ 124G  ← 模型权重
├── vlmevalkit_selftest/ 117G （含 LMUData/datasets）
├── VLM_eval/ 45G
├── anaconda3/ 43G
├── silero_test/ 31G, tensorRT/ 6.5G, papers/ 895M

/share/caolujing  1.6T
├── work/ 919G （REPA/ 783G, DiT/ 134G）
├── data/ 587G （images/ 245G, imagenet/ 151G, vae-sd/ 50G）
├── miniconda3/ 37G, .cache/ 2.4G, arxivsearch/ 1.1G

/share/mtang  1.4T
├── work/ 996G （JD/ 430G, Gen/ 353G, LLM_Bench/ 193G, perf/ 20G, tmp/ 95M）
├── .cache/ 344G           ← 最大的 cache
├── miniforge3/ 27G, .devin-server/ 3.5G, .agentos/ 3.2G, .codex/ 952M

/share/huayicong  1.2T
├── proj/ 359G （torch_sipu/ 73G, onnxruntime-ep/ 49G, torch-branch-a/ 113G, …）
├── code/ 51G, miniconda3/ 34G, .cache/ 20G, .codex-local/ 5.9G
├── tensorrt-sdk-11.1-cuda12.9/ 4.7G, Qwen3-VL-2B-*-onnx-opset27/ 4.6G×2, tmp_migrated_20260805/ 1.5G

/share/songzun  502G
├── closed_sim_new/ 316G （cosmos-framework/ 87G, WorldEngine/ 61G, closed-sim/ 52G, cosmos-transfer2.5/ 42G, Tersim/ 20G, …）
├── uv-cache-sz/ 80G       ← cache
├── cosmos/ 22G, cosmos-transfer2.5/ 19G, miniconda3/ 17G, huggingface/ 15G
├── flashdreams/ 5.7G, TeraSim/ 4.0G, perf_ui/ 4.0G, huggingface/cache/ 
```

---

### 43.2 `/share/users/`

`/share/users/` 下共 60+ 个用户目录，**合计约 5.4T**。

#### 总览（Top 30，按大小排序）

| 目录 | 大小 | 目录 | 大小 |
|---|---|---|---|
| `zhouziyi` | **766G** | `xieyi` | 101G |
| `like` | **565G** | `al` | 97G |
| `bzhan` | **520G** | `hmx` | 87G |
| `aima` | **387G** | `lishuyuan` | 80G |
| `bokangz` | **384G** | `zhengbao` | 71G |
| `chenzhizhen` | **355G** | `byy` | 69G |
| `tangdehua` | **297G** | `lizi` | 37G |
| `yangrunlin` | **278G** | `txdrg` | 25G |
| `huyoufu` | **232G** | `guopengju` | 19G |
| `zhaosiwei` | 184G | `xiedebin` | 7.6G |
| `zlxu` | 172G | `sunzhongao` | 6.0G |
| `ziheng` | 169G | `kuangwanda` | 4.7G |
| `wangyu` | 160G | `zhubokang` | 4.3G |
| `yuliang` | 146G | `jiale` | 3.7G |
| `luman` | 121G | `xuzelong` | 3.4G |
| （其余 20+ 个 < 3G） | | | |

#### (A) 模型权重

| 目录 | 大小 |
|---|---|
| `/share/users/zhouziyi/models` | **442G** |
| `/share/users/chenzhizhen/Qwen3-Next-80B-A3B-Instruct` | **153G** |
| `/share/users/huyoufu/DeepSeek-V2-Lite-Chat` | 30G |
| `/share/users/huyoufu/Llama-3.2-1B-Instruct` | 7.3G |
| `/share/users/huyoufu/Meta-Llama-3-8B` | 602M |
| `/share/users/zlxu/model` | 16G |
| `/share/users/chenzhizhen/Qwen3.6` | 52G |

#### (B) 数据集

| 目录 | 大小 |
|---|---|
| `/share/users/zhaosiwei/transformer-pytorch/dataset` | — |
| `/share/users/chenzhizhen/dataset` | 8.3G |
| `/share/users/ziheng/data` | 37G |
| `/share/users/tangdehua/huggingface/datasets` | （含在 12G 内） |

> `/share/users/` 下数据集类目录整体不大，主要是代码仓库、conda 环境和缓存。

#### (C) cache / tmp / temp（`find -maxdepth 3` 命中，按大小）

| 目录 | 大小 |
|---|---|
| `/share/users/aima/cache`（含 `pip_cache`） | **69G** |
| `/share/users/like/.cache` | **39G** |
| `/share/users/wangyu/.cache` | **29G** |
| `/share/users/zhengbao/.pip_cache` | **17G** |
| `/share/users/tangdehua/huggingface` | 12G |
| `/share/users/al/triton_cacheexport` | **8.5G** |
| `/share/users/huyoufu/workspace_node38/uv-cache` | 5.4G |
| `/share/users/yuliang/.triton` | 5.2G |
| `/share/users/zhengbao/conda_pkgs` | 4.7G |
| `/share/users/like/huggingface_cache` | 3.5G |
| `/share/users/like/temp` | 2.6G |
| `/share/users/al/triton_cache` | 1.7G |
| `/share/users/zhengbao/tmp` | 1.4G |
| `/share/users/wangyu/.npm` | 1.4G |
| `/share/users/xieyi/cache`（含 `ccache`） | 932M |
| `/share/users/bzhan/tmp` | 669M |
| `/share/users/bzhan/.cache` | 296M |
| `/share/users/bzhan/torch_cache/torch_compile_cache` | 274M |
| `/share/users/like/package/temp`、`package/tmp`、`qemu_demo/temp` | — |
| `/share/users/like/.conda`、`xieyi/.conda`、`zhouziyi/.conda`、`backup/.conda` | — |
| `/share/users/zhouziyi/.cache`, `.npm/_cacache`, `.nvm/.cache`, `.claude/cache` | — |
| `/share/users/like/package/temp` / `tmp`（同 like） | — |
| `/share/users/zhaosiwei/tmp`, `.cache`, `vidur-servingsim/cache`, `hasp/tmp` | — |
| `/share/users/bokangz/tmp`（含 `hf-cache`, `flashinfer-cache`）, `probe-live/gocache`, `.cache` | — |
| `/share/users/wangyu/.claude/cache`, `.codex/tmp`, `.nvm/.cache`, `harness/cache`, `dcsm/.cache` | — |
| `/share/users/lishuyuan/temp`、`xieyi/code/tmp`、`al/tmp`、`nanhua/uv/cache` | — |

#### (D) 各用户 Top 分支明细

```
/share/users/like  565G
├── package/ 252G
├── miniconda3/ 99G
├── qemu_demo/ 96G
├── .cache/ 39G          ← cache
├── bench-io/ 26G
├── opt/ 23G
├── docker-image/ 5.5G, sipu_sdk_debug/ 4.1G, huggingface_cache/ 3.5G, temp/ 2.6G, build/ 2.2G

/share/users/aima  387G
├── workspace/ 297G （含 .cache）
├── cache/ 69G           ← cache（pip_cache）
├── miniconda3/ 17G
└── temp/ 1.3M

/share/users/bzhan  520G
├── vidur-e2e-combo-accuracy-20260729/ 196G
├── vidur-profile-replay-audit-20260722/ 66G
├── env/ 60G
├── gpu-frequency-diagnostic-20260728/ 29G
├── vidur-servingsim-origin-main-async-20260624/ 27G
├── dsv3l16_reprofile_*（3 个）/ 12–15G
└── tmp/ 669M, torch_cache/ 274M, .cache/ 296M

/share/users/chenzhizhen  355G
├── Qwen3-Next-80B-A3B-Instruct/ 153G   ← 模型权重
├── Qwen3.6/ 52G, .git/ 44G, qwen-runtime/ 30G, uv/ 19G, .cache/ 17G,
├── reproduce_kvcache/ 14G, marconi/ 13G, dataset/ 8.3G

/share/users/tangdehua  297G
├── project/ 229G
├── miniconda3/ 49G
├── huggingface/ 12G     ← cache
└── llm_wights/ 8.9G

/share/users/huyoufu  232G
├── qemu/ 66G, workspace_node38/ 52G（含 uv-cache 5.4G）, miniconda3/ 37G,
├── workspace_gpu/ 31G, DeepSeek-V2-Lite-Chat/ 30G, Llama-3.2-1B-Instruct/ 7.3G

/share/users/zhaosiwei  184G
├── .miniconda/ 43G, torch_sipu/ 26G, qemu/ 23G, pytorch/ 23G,
├── transformer-pytorch/ 22G, hasp-ws/ 16G, tracetto/ 14G, home/ 6.8G

/share/users/zlxu  172G
├── anaconda/ 62G, .agentos/ 31G, model/ 16G, mineru/ 14G, SimpleRAG/ 13G,
├── agent-os*/（多个）/ 2.7–8.6G, agentos-private-images/ 2.7G

/share/users/zhengbao  71G
├── miniconda3/ 29G, .pip_cache/ 17G ← cache, code/ 9.9G, conda_pkgs/ 4.7G ← cache,
├── huggingface/ 4.5G, conda_envs/ 3.8G, .pipcache/ 1.7G, tmp/ 1.4G

/share/users/xieyi  101G
├── anaconda3/ 48G, code/ 40G, downloads/ 9.1G, .codex/ 1.3G,
├── cache/ 932M（ccache）, to_be_deleted/ 767M, backup/ 730M, .tilelang/ 84M

/share/users/zhouziyi  766G   ← /share/users 下最大
├── models/ 442G           ← 模型权重
├── softwares/ 138G
├── projects/ 75G
├── experiments/ 64G
├── builds/ 43G
└── .vscode-server/ 2.3G, .npm/ 1.9G, .nvm/ 1.1G, papers/ 569M

/share/users/bokangz  384G
├── project/ 269G
├── anaconda3/ 71G
├── .cache/ 23G            ← cache
├── agent-os-private-images/ 17G
├── agent-os-private-deployment/ 4.1G
└── tmp/（含 hf-cache、flashinfer-cache）, probe-live/gocache

/share/users/yangrunlin  278G
├── learn/ 246G
├── tool-home/ 23G
├── codex-home/ 3.0G, arxiv/ 3.0G, .local/ 1.7G, quant/ 1.5G
└── .npm-stale-codex-20260909/ 260M, .agentos/ 191M

/share/users/ziheng  169G
├── codes/ 110G
├── data/ 37G              ← 数据集
├── software/ 22G
└── workspace-20260906-112706/ 480M, service_desk/ 440M

/share/users/byy  69G
├── workspace-202606121028/ 61G
├── .agentos/ 8.4G
└── workspace-202606121012/ 81M, .cache/ 832K
```

---

### 43.3 按类别的全局汇总

#### 🔴 模型权重 / checkpoints（最大头，约 10T+）

| 目录 | 大小 | 用户 |
|---|---|---|
| `weihongyang/JDJV` | 1.2T | weihongyang |
| `guorui/投机解码训练/SpecForge` | 1.7T | guorui |
| `weihongyang/ml-fastvlm/output` | 1.2T | weihongyang |
| `pengkunfu/JDJV/GaussianOcc` | 1.1T | pengkunfu |
| `mtang/work/JD` + `work/Gen` | 430G + 353G | mtang |
| `huayicong/proj` | 359G | huayicong |
| `yufne/Moore-AnimateAnyone` / `AnyEdit` / `Open-AnimateAnyone` | 308G / 279G / 223G | yufne |
| `chenzhizhen/Qwen3-Next-80B-A3B-Instruct` | 153G | chenzhizhen |

#### 🟠 数据集（约 8T）

| 目录 | 大小 | 用户 |
|---|---|---|
| `yufne/UltraEdit` / `Senorita` / `JourneyDB` | 各 1.5T | yufne |
| `yufne/ShareGPT4Video` | 1.4T | yufne |
| `caolujing/data` | 587G | caolujing |
| `yufne/dataset` | 516G | yufne |
| `yufne/LLaVA-Video-178K` | 465G | yufne |
| `yufne/M4-Instruct-Data` | 219G | yufne |
| `guorui/datasets` | 149G | guorui |
| `weihongyang/JDJV-SIMO-Qwen3-VL` | 101G | weihongyang |

#### 🟡 cache / temp / tmp（约 800G+）

| 目录 | 大小 | 用户 |
|---|---|---|
| `weihongyang/evalscope_latest` | 651G | weihongyang |
| `mtang/.cache` | 344G | mtang |
| `pengkunfu/edge_llm_models`（权重） | 124G | pengkunfu |
| `weihongyang/evalscope` | 115G | weihongyang |
| `songzun/uv-cache-sz` | 80G | songzun |
| `guorui/home_cache` / `model_cache` / `hf_cache` | 79G / 71G / 60G | guorui |
| `aima/cache`（`users/`） | 69G | aima |
| `like/.cache`（`users/`） | 39G | like |
| `guorui/.cache` | 29G | guorui |
| `wangyu/.cache`（`users/`） | 29G | wangyu |
| `huayicong/.cache` | 20G | huayicong |
| `zhengbao/.pip_cache`（`users/`） | 17G | zhengbao |
| `guorui/conda_cache` | 12G | guorui |
| `al/triton_cacheexport`（`users/`） | 8.5G | al |
| `guorui`.codex/cache`、`songzun/.codex/tmp`、`huayicong/.codex-local/*/tmp`（867 个） | — | — |

#### 🔵 conda 环境

`weihongyang/miniconda3` 379G、`guorui/envs` 100G、`yufne/miniconda3` 59G、`pengkunfu/anaconda3` 43G、`caolujing/miniconda3` 37G、`huayicong/miniconda3` 34G、`mtang/miniforge3` 27G、`songzun/miniconda3` 17G。

---

### 43.4 可清理候选（仅列出，**未执行任何删除**）

按「回收空间 / 风险」排序：

| 候选 | 大小 | 说明 |
|---|---|---|
| `guorui/投机解码训练/SpecForge` | 1.7T | 训练中间 checkpoint，确认后可删 |# done
| `yufne/UltraEdit`、`yufne/Senorita`、`yufne/JourneyDB` | 各 1.5T | 数据集，若可重新下载 |
| `yufne/ShareGPT4Video` | 1.4T | 数据集 |
| `weihongyang/JDJV` +
`pengkunfu/JDJV` | 1.2T + 1.2T | 同名项目，疑似重复，建议比对 | #done
| `weihongyang/ml-fastvlm/output` | 1.2T | 训练输出 |
| `pengkunfu/JDJV/GaussianOcc` | 1.1T | |
| `weihongyang/evalscope_latest` | 651G | 评测工作副本，可重建 |
| `caolujing/data/images` + `imagenet` | 245G + 151G | 数据集 |
| `mtang/.cache` | 344G | 可 `rm -rf` 后重建 |
| `guorui/home_cache`（tvm-ffi/deep_gemm/torch_compile） | 79G | 可重建 |
| `songzun/uv-cache-sz` | 80G | `uv cache clean` 可回收 |
| `guorui/model_cache` / `hf_cache` | 71G / 60G | HF 缓存，可重新下载 |
| `huayicong/.codex-local/*/tmp`（867 个） | ~6G | 大量陈旧 tmp |
| `weihongyang/evalscope_latest` 内的 `.cache` | — | |
| `users/` 下各类 `.cache` / `pip_cache` / `triton_cache` | `aima` 69G、`like` 39G、`wangyu` 29G、`zhengbao` 17G、`al` 8.5G 等 | 均可重建 |

**注意**：`/share` 总盘 78T 已 100% 满。本次统计的两个范围（9 个用户 ≈ 21.7T + `/share/users/` ≈ 5.4T）**合计约 27T**，其余空间在 `/share/` 下其它用户目录中，本次未覆盖。

### 43.5 备注

- 全部为**只读**操作（`du` / `find`），未删除、未移动任何文件。
- NFS 上遍历极慢（8.7T 级目录需十几分钟）。`/share/users/` 下**所有 60+ 个用户目录的总大小均已测出**，其中 `bokangz`、`yangrunlin`、`ziheng`、`byy` 等少数用户的二级明细为后补。
- `/share/huayicong/proj` 在两次测量间从 817G 变为 359G，推断期间有文件被删除（统计值是**当时快照**）。
- 9 个用户中最占空间的类别依次是：**数据集 ≈ 8T**、**模型权重/checkpoint ≈ 10T**、**cache/tmp ≈ 0.8T**、**conda 环境 ≈ 0.7T**。

---

## 44. 为什么 `du` 变小了、`df` 空间却没释放？—— 存储端快照（snapshot）钉住了已删数据

### 44.1 现象

```
$ du -sh /share/liwang/VLM_projects/Griffon/GThinker /share/liwang/VLM_projects/ml-fastvlm
43M   /share/liwang/VLM_projects/Griffon/GThinker      # 原本 4.3T
37M   /share/liwang/VLM_projects/ml-fastvlm           # 原本 683G

$ df -Th /share
10.97.128.245:/share nfs4   78T   77T  930G  99% /share   # 仍几乎 100%，没降下来
```

删了约 5T，`df` 却几乎不动。

### 44.2 根因

**`/share` 所在的 NAS 启用了存储端快照（snapshot）。被 `rm` 掉的数据仍被快照引用，存储无法回收这些数据块，所以 `df` 不降。**

这是 NetApp 风格的 `.snapshot` 目录，**在每一层目录下都可见**：

```
/share/.snapshot/
├── 2-hourly.2026-09-10_1015
├── 2-hourly.2026-09-10_1215
├── daily.2026-09-10_0010
├── weekly.2026-09-06_0015
├── hourly.2026-07-30_1605
└── hourly.2026-07-30_1705
```

（`/share/liwang/.snapshot`、`/share/liwang/VLM_projects/.snapshot`、`…/GThinker/.snapshot` 等各层同样存在。）

### 44.3 证据链

**(1) 同一路径：live 已空、快照仍在**

| 路径 | 大小 |
|---|---|
| **live**：`…/GThinker/EasyR1/checkpoints_RL` | **18M**（只剩 `checkpoint_tracker.json` / `experiment_log.jsonl` 等小文件，大权重已删） |
| **快照**：`…/GThinker/.snapshot/2-hourly.2026-09-10_1015/EasyR1/checkpoints_RL` | **2.3T**（`fastvlm-1.5B`、`fastvlm-1.5B-stabel`、`fastvlm-1.5B-stable-opt1`、`Qwen2.5_vl_7B` 都还在） |

**(2) 6 个快照全部仍含这份数据** → 只要还有任一快照引用这些块，存储就不回收：

| 快照 | 是否含 `checkpoints_RL/fastvlm-1.5B` |
|---|---|
| `hourly.2026-07-30_1605` | 有 |
| `hourly.2026-07-30_1705` | 有 |
| `weekly.2026-09-06_0015` | 有 |
| `daily.2026-09-10_0010` | 有 |
| `2-hourly.2026-09-10_1015` | 有 |
| `2-hourly.2026-09-10_1215` | 有 |

**(3) `df` 趋势稳定**：连续采样 3 次（间隔 20s）均为 `77T used / ~930G avail`，排除"删除仍在进行中"。

### 44.4 量化对比：删了 5.0T，只释放 0.93T

`/share/liwang` 删除前后（live 树）：

| 目录 | Sep 9 测得 | Sep 10 测得 | 变化 |
|---|---|---|---|
| `VLM_projects` | 5.0T | **28G** | **−4.97T** |
| `Benchmark` | 101G | 80G | −21G |
| 其余（nim-cache\*、.conda、.cache 等） | ~76G | ~66G | −10G |
| `Projects` / `JD_evaluation` / `envs` | 2.6T / 384G / 161G | 不变 | — |
| **合计** | **≈ 8.3T** | **≈ 3.3T** | **≈ −5.0T** |

对应的 `df`：

| 时间 | `/share` avail |
|---|---|
| Sep 8（最初） | 2.3G |
| Sep 10 | ~930G |
| **实际释放** | **≈ 0.93T** |

> **live 树删了 ≈ 5.0T，`df` 只释放了 ≈ 0.93T，差额 ≈ 4T 被快照钉住**（再叠加集群里其它用户的正常读写）。

### 44.5 机制

```
rm 删除文件
   │
   ├─ live 目录树：名字消失 ──► du 变小            ✅（看到的现象）
   │
   └─ 数据块：仍被 snapshot 引用 ──► 存储不能回收 ──► df 不变   ❌（看到的现象）
```

- `du` 只看 live 目录树，**不进入 `.snapshot`**（实测 `du -sh GThinker` = 43M，而其快照内容 ≥ 2.3T），所以目录"看起来很小"。
- `df` 报告的是**整个卷的真实占用**，包含被快照钉住的数据块，所以降不下来。
- **只要快照还在，`df` 就不会降；快照过期/被删后，空间才自动释放。**

### 44.6 怎么办

1. **找存储管理员删除相关快照**，或确认保留策略的轮转周期。
2. 只删最老的一两个快照通常就能释放绝大部分被钉住的空间——但要先确认这些快照没有恢复需求。
3. 最老的三个是 `hourly.2026-07-30_1605`、`hourly.2026-07-30_1705`、`weekly.2026-09-06_0015`；若保留策略要等它们自然过期，可能还要很久。
4. **自己重复 `rm` 是没用的**——快照存在期间，只会让 `du` 更小，`df` 纹丝不动。

### 44.7 排查中排除的其它原因

| 假设 | 检查 | 结论 |
|---|---|---|
| 删除后仍被进程持有（deleted-but-open） | gpu005、node33 上 `lsof +L1 \| grep /share` | 无 |
| NFS silly-rename 残留 | `find … -name '.nfs*'`（VLM_projects 全量） | 无 |
| 被移进回收站 | `/share/.Trash-*`（4 个）、各层 `*trash*` | 均为空/不存在 |
| 删除仍在进行 | gpu005/node33 上 `ps \| grep rm/shred`；`df` 连续采样 | 无进程，df 稳定 |
| 被 `mv` 到同文件系统别处 | 重新测 `/share/liwang` 顶层 | 总量确实 −5.0T，非移动 |

### 44.8 注意事项（坑）

- **`.snapshot` 让 `du` 与 `df` 天然不一致**：做空间统计时，`du` 反映的是 live 树，`df` 反映的是含快照的卷占用，两者对不上是正常的。
- 被清空目录的 `mtime` 显示为 **4 月/5 月**（如 `checkpoints_RL` = 2026-04-30、`fastvlm-1.5B` = 2026-04-08），而非删除当天；gpu005 和 node33 两节点 `stat` 结果一致，排除客户端 attr 缓存。这与普通客户端 `rm` 的行为不符，更像**存储侧操作（快照/管理员删除）留下的痕迹**；但不影响上面的结论——live 与快照的直接对比已确凿证明数据已从 live 树移除。
- 这次排查**全程只读**（`du` / `find` / `lsof` / `stat`），未删除、未修改任何快照或文件。

---

## 45. `scripts/ci/sipu_ci_exec.sh` 解析

> code base: `/share/users/like/package/sglang_sipu`

### 45.1 一句话

`sipu_ci_exec.sh` 是 **sipu 精度 CI 的宿主机入口**：先在宿主机上把 sgl-kernel 系列 wheel 编出来，再把「一个 case」或「一个 suite」分派到本机容器（archmodel）或 QEMU 客户机里执行，最后汇总评分。

脚本头注释（`scripts/ci/sipu_ci_exec.sh:2-12`）写明了两种用法：

```
sipu_ci_exec.sh --config-yaml <path> --launch-config <name> --test-case <name> [--platform qemu]   # 单 case
sipu_ci_exec.sh --test-suites nightly[,<suite>...] [--mode compile|graph]                          # suite
```

### 45.2 脚本分段讲解

**(1) 常量与路径**（`:15-30`）
`SCRIPT_DIR` / `SGLANG`（仓库根）/ `KERNEL`（`sgl-kernel-sipu` 子模块）/ `SUITES_YAML`；镜像 `IMAGE` 默认 `harbor.siorigin.com/sglang-sipu/release:v0.5.18-sipu-dev-0.1.0`；并导出 `SIPU_CI_DUMP` / `SIPU_CI_WHEEL_DIR` / `SIPU_CI_QUEUE_DIR` 等。

**(2) 参数解析** `sipu_ci_exec.sh` 内联的 `while` 循环（`:50-77`）
- `--config-yaml` / `--launch-config` / `--test-case` / `--timeout`（`:52-55`）→ 存进变量 **并追加进 `RUN_ARGS`**，原样透传给下层。
- `--test-suites`（`:56`）、`--mode eager|compile|graph`（`:57-63`）、`--platform archmodel|qemu`（`:64-69`）、`--tmp-sgl-kernel-sipu`（`:70-73`）。
- 互斥校验（`:79-89`）：`--test-suites` 与单 case 参数不能同时给；`--mode` 只能配 `--test-suites`。

**(3) 公共函数来自 `sipu_ci_single_case.sh`**（`source` 在 `:92`）
- `detect_current_platform`（`scripts/ci/sipu_ci_single_case.sh:10`）：有 `/dev/sipu` → `qemu`，否则 `host`。
- `setup_docker_args`（`sipu_ci_single_case.sh:34`）：拼 docker 参数（挂载 `$SGLANG:/sgl-workspace/sglang`、wheel 目录、CI_ROOT、`/share_data/*` 多个只读目录、`$HOME/.ssh`；qemu 时再加 `--device /dev/sipu/usipu*` 与 `SIRT_SHIM_NAME=ksipu`）。
- `run_container`（`:89`）、`reclaim_host_ownership`（`:104`，把容器写出的文件 chown 回宿主 UID/GID）。

**(4) 清理钩子** `cleanup`（`:96-103`）+ `trap cleanup EXIT`（`:104`）：退出时停 qemu、回收属主、删 `ci_queues`，并按 `SIPU_CI_KEEP_WHEELS` 决定是否删 `ci_wheels`。

**(5) 准备 sgl-kernel 子模块**（`:142-146`）：没给 `--tmp-sgl-kernel-sipu` 就走 `init_submodule`（`:115`），必要时 `git submodule update --init sgl-kernel-sipu`。

**(6) 编 wheel** `build_wheels`（`:122-140`）：起 `sipu-ci-builder` 容器，在 `$CONTAINER_KERNEL` 里 `source setup.sh` → `make clean` → `make install-kernels` → `pip wheel` 出 **5 个 wheel**（sgl-kernel 本体、`deepep`、`siorigin_triton_kernels`、`siorigin_tilelang_kernels`、`triton_ops`）。给了 `--tmp-sgl-kernel-sipu` 则整段跳过（`:148-150`）。

**(7) 分派** `dispatch`（`:182-190`）：`SIPU_CI_PLATFORM=qemu` → `run_on_qemu`（`:161`，`sipu_ci_qemu.sh start` 起虚拟机、ssh 进去跑 `sipu_ci_run.sh`）；否则 `run_local`（`:156`）→ 直接 `bash scripts/ci/sipu_ci_run.sh "$@"`。

**(8) suite 分支** `dispatch_suite`（`:192-215`）：调 `suite_raw`（`:152`）→ `python3 scripts/ci/sipu_ci_cases.py --suites-yaml ... --suite ...` 展开 suite；支持 `#include=` 递归子 suite、`#platform=` 覆盖平台。

**(9) 主入口**（`:217-228`）：有 `--test-suites` 就逐个 `dispatch_suite`；否则 `dispatch "${PLATFORM:-archmodel}" "${RUN_ARGS[@]}"`（`:226`），最后 `exit $overall`。

### 45.3 传入 `--config-yaml configs/qwen3/qwen3_0.6b_4layer.yaml --launch-config general_text_tp_custom_ar --test-case text-only` 会做什么

走的是**单 case**分支（没有 `--test-suites`），`PLATFORM` 未给 → 平台 `archmodel`（本机容器）。完整动作链：

**第 1 步：宿主机（`sipu_ci_exec.sh`）**
1. 解析出 `CONFIG_YAML=configs/qwen3/qwen3_0.6b_4layer.yaml`、`LAUNCH_CONFIG=general_text_tp_custom_ar`、`TEST_CASE=text-only`、`TIMEOUT=30m`（默认）。三个必填项齐全，校验通过（`:84-85`）。
2. `detect_current_platform`（`:93`）→ `host`。
3. 建 `ci_dumps` / `ci_wheels` / `ci_queues`（`:95`），挂 `trap cleanup EXIT`（`:104`）。
4. 没有 `--tmp-sgl-kernel-sipu` → `init_submodule`（`:145`）。
5. `setup_docker_args`（`:147`）→ `build_wheels`（`:149`）：**编 5 个 wheel** 到 `ci_wheels/`，然后 `reclaim_host_ownership`。
6. `dispatch archmodel --config-yaml ... --launch-config ... --test-case ...`（`:226`）→ `run_local`（`:158`）→ `bash scripts/ci/sipu_ci_run.sh <三个参数>`。

**第 2 步：`sipu_ci_run.sh`（仍在宿主机）**
7. 校验参数、检查 `$SIPU_CI_WHEEL_DIR/*.whl` 存在（`sipu_ci_run.sh:109`），否则直接报错退出。
8. 走单 case 分支（`:191-196`）：`export SIPU_CI_FAIL_FAST=1`，把下面这行写进 `ci_queues/cases`：
   ```
   qwen3|qwen3_0.6b_4layer|configs/qwen3/qwen3_0.6b_4layer.yaml|general_text_tp_custom_ar|text-only|30m
   ```
   （`family|model|config|launch|test_case|timeout`，`:193-195`），然后 `run_cases single 1`（`:196`）。
9. `run_cases`（`:136`）：`SIPU_CI_RUN_DIR=$CI_ROOT/single/<YYYYMMDD>`（`run_dir_for`，`:21`，默认 `CI_ROOT=$SGLANG/ci_logs`）；`parallel=1` → `SIPU_CI_LIVE_LOG=1`；用 `xargs -P 1 -I CASE scripts/ci/sipu_ci_single_case.sh CASE`（`:152`）跑这一条 case。

**第 3 步：`sipu_ci_single_case.sh` → 起容器**
10. `run_one_case`（`sipu_ci_single_case.sh:118`）拆出各字段，`cid = qwen3_0.6b_4layer_general_text_tp_custom_ar_text-only`，拼出容器内命令（`:126-130`）：
    ```
    /bin/bash /sgl-workspace/sglang/scripts/ci/sipu_ci_test.sh \
      --config-yaml configs/qwen3/qwen3_0.6b_4layer.yaml \
      --launch-config general_text_tp_custom_ar \
      --test-case text-only \
      --timeout 30m
    ```
11. `run_container`（`:89`）→ `docker run --name sipu-ci-<cid>-$$ <docker_args> $IMAGE bash -lc "<上面的命令>"`；跑完写 `ci_queues/status/<cid>`，并 `reclaim_host_ownership`。
12. 回到 `run_cases`：`python3 scripts/ci/sipu_ci_summary.py ...` 汇总打分（`:157`）。

**第 4 步：容器内（`sipu_ci_test.sh`）**
13. `source setup.sh` / `setup_triton.sh` / `setup_tilelang.sh`，`unset TORCH_DEVICE_BACKEND_AUTOLOAD`，设 `LD_LIBRARY_PATH`（`sipu_ci_test.sh:43-50`）。
14. 解析 `cfg`：相对路径拼成 `$SGLANG/test/srt/sipu/configs/qwen3/qwen3_0.6b_4layer.yaml`（`:52-53`）。
15. `case_uses_custom_deepep`（`:58`）读 `launch_configs.general_text_tp_custom_ar.test_env.text-only.SGLANG_SIPU_USE_CUSTOM_DEEPEP` → **本 config 没设** → 不装 deep_ep wheel，保留镜像自带版本（`:89-90`、`:101-104`）。
16. `pip install --force-reinstall --no-deps` 装 `$SIPU_CI_WHEEL_DIR/*.whl`（`:99-106`）。
17. `is_compile=0`（名字不以 `_compile` 结尾，`:122-123`），所以不设 tiled 环境、不做 compile contract 检查。
18. 找 CUDA golden：`$SIPU_CI_CUDA_DUMP/qwen3_0.6b_4layer/general_text_tp_custom_ar/text-only/cuda`，**不存在就 exit 1**（`:138-143`）；存在则在 `$SIPU_CI_DUMP/...` 下建同名 symlink（`:145-147`）。
19. 跑真正的 case（`:150-154`）：
    ```
    timeout 30m python3 -u test/srt/sipu/test_utils/run_test_job.py \
      --config-yaml <cfg> --launch-config general_text_tp_custom_ar --test-case text-only \
      --device sipu --dump-base $SIPU_CI_DUMP --log-base $SIPU_CI_RUN_DIR/logs
    ```
    日志 tee 到 `$SIPU_CI_RUN_DIR/logs/qwen3/qwen3_0.6b_4layer_general_text_tp_custom_ar_text-only_sipu.log`。
20. 比对（`:188-198`）：`python3 -u test/srt/sipu/analyse_utils/compare_cuda_sipu.py <cfg> --dump-base ... --launch-config ... --test-case ...`，结果 tee 到 `$SIPU_CI_RUN_DIR/compare/<cid>.log`。

**第 5 步：`run_test_job.py` 里这个 case 具体跑什么**
21. `main`（`test/srt/sipu/test_utils/run_test_job.py:482`）：
    - `launch_cfg = cfg["launch_configs"]["general_text_tp_custom_ar"]`（`:498`）；要求同时有 `cuda` 和 `sipu` 两个块（`:504`）——本 config 满足（`configs/qwen3/qwen3_0.6b_4layer.yaml:42-75`）。
    - `test_case` 必须在 `launch_cfg["tests"]` 里（`:508`）——本 config 是 `tests: [text-only]`（`qwen3_0.6b_4layer.yaml:44-45`）→ 通过。
    - `TEST_REGISTRY["text-only"]` → `run_text_only`（`:514`，registry 定义在 `:475-479`）。
    - 用时用 `_apply_test_env`（`:118`）把 `launch_configs.<name>.test_env.<case>` 里的键值写进 `os.environ`——本 launch config 会设
      **`SGLANG_SIPU_USE_CUSTOM_ALLREDUCE=1`**（`qwen3_0.6b_4layer.yaml:46-48`）。
22. 于是这个 case 的实际语义：用 **`sipu` 设备**、按 `sipu:` 块（`qwen3_0.6b_4layer.yaml:62-75`：`tp_size: 2`、`disable_overlap_schedule: true`、`gpu_id_step: 2`、`attention_backend: sipu`、`disable_cuda_graph: true` …）起引擎，跑 `text-only`（纯文本 prompt，`temperature=0`, `max_new_tokens=2`），**并打开 `SGLANG_SIPU_USE_CUSTOM_ALLREDUCE=1`**（这就是名字里 `tp_custom_ar` 的含义：TP + 自定义 all-reduce），dump 各层张量，再和 CUDA golden 比 `cos_sim` / `mean_atol`。

**判定门槛**（`scripts/ci/sipu_ci_summary.py` 模块 docstring）：eager 模式下要求 case 退出码 0，且每个可比对的 pass 满足 `cos_sim >= 0.999`、`mean_atol <= 0.05`；结果写 `<run-dir>/summary.tsv` 与 `summary.log`。

### 45.4 `--launch-config` 的合法值

**没有全局枚举——合法值由 `--config-yaml` 自己决定。** 校验点在 `test/srt/sipu/test_utils/run_test_job.py:498-503`：

```python
launch_cfg = cfg["launch_configs"].get(args.launch_config)
if not isinstance(launch_cfg, dict):
    raise ValueError(f"{args.config_yaml}: unknown launch_config {args.launch_config!r}. "
                     f"Available: {sorted(cfg['launch_configs'])}")
```

即：**合法值 = 该 YAML 里 `launch_configs` 的键**。另有两条附加约束：必须同时含 `cuda` 与 `sipu` 两个引擎块（`:504-507`）；`--test-case` 必须在 `launch_configs.<name>.tests` 列表里（`:508-513`）且在 `TEST_REGISTRY` 中（`:514-518`）。

**(a) 对本次的 `configs/qwen3/qwen3_0.6b_4layer.yaml`，只有 2 个合法值**（`qwen3_0.6b_4layer.yaml:14-75`）：

| `--launch-config` | 特点 |
|---|---|
| `general_text` | 基础文本路径，`tp_size` 未设（默认单卡） |
| `general_text_tp_custom_ar` | `tp_size: 2`、`gpu_id_step: 2`、`disable_overlap_schedule: true`，并设 `test_env.text-only.SGLANG_SIPU_USE_CUSTOM_ALLREDUCE=1` |

两个的 `tests` 都只有 `text-only`。

**(b) 全仓库范围出现过的名字共 13 个**（对 `test/srt/sipu/configs/*/*.yaml` 的键去重统计）：

| 名字 | 出现次数 | 含义 |
|---|---|---|
| `general_text` | 20 | 通用文本（dense）路径 |
| `triton_moe_text` | 18 | MoE + Triton 文本路径 |
| `deepep_deepgemm_text` | 17 | MoE + DeepEP + DeepGEMM 文本路径 |
| `general_vision` | 16 | 通用多模态（vision）路径 |
| `triton_moe_vision` | 10 | MoE + Triton 多模态 |
| `deepep_deepgemm_vision` | 5 | MoE + DeepEP + DeepGEMM 多模态 |
| `general_text_tp_siccl` | 1 | 文本 + TP + SiCCL |
| `general_text_tp_custom_ar` | 1 | 文本 + TP + 自定义 all-reduce（本次用的） |
| `general_text_compile` | 1 | 文本 + torch.compile |
| `deepep_deepgemm_text_compile` | 1 | MoE 路径 + torch.compile |
| `deepep_deepgemm_dp_ep` | 1 | DeepEP 的 DP/EP 变体 |
| `triton_text` | 1 | Triton 文本路径 |
| `triton_vision` | 1 | Triton 多模态路径 |

> 命名规律：`<attention_backend>_<moe_backend>_<modal>[_<并行/编译变体>]`，例如 `general_text`、`triton_moe_vision`、`deepep_deepgemm_text_compile`。

**(c) 命名后缀约定**：名字以 `_compile` 结尾时进入 compile 模式——`sipu_ci_test.sh:122-123` 用 `[[ "$LAUNCH_CONFIG" == *_compile ]]` 判定 `is_compile`，随后打开 tiled 环境变量并把用来找 CUDA golden 的名字去掉 `_compile` / `_graph` 后缀（`:125-134`）。所以「合法值」还包括各 config 里按这个约定命名的 `*_compile` 键（如 `general_text_compile`、`deepep_deepgemm_text_compile`）。

**(d) suite 路径下的取值**：`--test-suites` 走 `scripts/ci/sipu_ci_cases.py::main`（`:34`），从 `scripts/ci/sipu_ci_suites.yaml` 的 `suites.<name>.cases[].launch_config` 读（`:60-76`），已用到的就是上表里除 `general_text_tp_custom_ar` 之外的那些。当前 suites 有 4 个：`smoke`(3 例) / `single-rank`(77 例) / `distributed`(3 例, qemu) / `nightly`(include `single-rank`+`distributed`)。

### 45.5 附：相关可调项速查

| 参数 / 环境变量 | 作用 | 出处 |
|---|---|---|
| `--platform archmodel\|qemu` | 本机容器 or QEMU 客户机 | `sipu_ci_exec.sh:64-69`, `:185` |
| `--tmp-sgl-kernel-sipu <dir>` | 跳过子模块初始化 + 编 wheel，改用现成目录 | `:70-73`, `:142-150` |
| `--mode eager\|compile\|graph` | 仅配 `--test-suites` | `:57-63`, `:86-88` |
| `SIPU_CI_KEEP_WHEELS=1` | 退出时不删 `ci_wheels` | `:100-102` |
| `SIPU_CI_CUDA_DUMP` | CUDA golden 根目录 | `:27`，默认 `/share_data/sglang_sipu/accuracy_verify/$CI_USER` |
| `SIPU_CI_DUMP` | 本次 dump 根目录 | `:28`，默认 `$PWD/ci_dumps` |
| `SIPU_CI_WHEEL_DIR` | wheel 目录 | `:29`，默认 `$PWD/ci_wheels` |
| `SIPU_CI_RUN_DIR` | 日志/比对结果目录 | `sipu_ci_run.sh:21`，单 case 为 `$CI_ROOT/single/<日期>` |
| `SIPU_CI_IMAGE` | 容器镜像 | `:20`，默认 `harbor.siorigin.com/sglang-sipu/release:v0.5.18-sipu-dev-0.1.0` |

**`--test-case` 的合法值**（`run_test_job.py:475-479` 的 `TEST_REGISTRY`）只有 3 个：`text-only`、`vision-with-text`、`prefix-caching`（且还要出现在所选 launch-config 的 `tests` 里）。

---

## 46. `sipu_ci_exec.sh … --platform qemu` 会执行什么？

```
scripts/ci/sipu_ci_exec.sh \
  --config-yaml configs/deepseek/ds_v3_2layer.yaml \
  --launch-config deepep_deepgemm_dp_ep \
  --test-case text-only \
  --platform qemu
```

### 46.1 与 archmodel 路径的一句话差别

前 8 步（编 wheel）**完全相同**；差别只在**分派目标**：`--platform qemu` 不再在本机容器里跑 case，而是**在宿主机拉起一台 QEMU 虚拟机**（跑 sipu 的 cmodel/仿真设备），ssh 进 guest，在 guest 里再起容器执行 case，最后关掉虚拟机。

### 46.2 完整动作链

#### 第 1 步：宿主机 `sipu_ci_exec.sh`

1. 参数解析（`:50-77`）：`CONFIG_YAML` / `LAUNCH_CONFIG` / `TEST_CASE` 存入变量**并进 `RUN_ARGS`**；`--platform qemu` 只写入 `PLATFORM`（`:64-69`，**不**进 `RUN_ARGS`，所以 guest 侧收不到它）。
2. 互斥/必填校验（`:79-89`）：无 `--test-suites` → 三参数齐全 → `MODE=eager` → 通过。
3. `source sipu_ci_single_case.sh`（`:92`）；`detect_current_platform`（`:93`）→ **宿主上无 `/dev/sipu` → `SIPU_CURRENT_PLATFORM=host`**（`sipu_ci_single_case.sh:10-16`）。
4. `mkdir` 三个目录（`:95`），挂 `trap cleanup EXIT`（`:104`，`cleanup` 在 `:96-103`：**`sipu_ci_qemu.sh stop`** → reclaim 属主 → 删 queue → 按需删 wheel 目录）。
5. 无 `--tmp-sgl-kernel-sipu` → `init_submodule`（`:145`）。
6. `setup_docker_args`（`:147`）——注意此刻 `SIPU_CURRENT_PLATFORM=host`，所以 `sipu_ci_single_case.sh:65-86` 那段 qemu 专属的 `--device /dev/sipu/*`、`SIRT_SHIM_NAME=ksipu` **在宿主这一步不会加**（它是在 guest 里才加的）。
7. `build_wheels`（`:149` → `:122-140`）：起 `sipu-ci-builder` 容器，`source setup.sh` → `make clean` → `make install-kernels` → `pip wheel` 出 **5 个 wheel**（sgl-kernel / `deepep` / `siorigin_triton_kernels` / `siorigin_tilelang_kernels` / `triton_ops`）到 `$SIPU_CI_WHEEL_DIR`，随后 `reclaim_host_ownership`。
   - ⚠️ 本 case 的 `test_env` 里有 `SGLANG_SIPU_USE_CUSTOM_DEEPEP=1`，所以后面容器里**必须要能拿到 `deep_ep*.whl`** —— 正好由这一步产出。
8. 主入口（`:226`）：`dispatch "${PLATFORM:-archmodel}" "${RUN_ARGS[@]}"` → **`dispatch qemu --config-yaml … --launch-config … --test-case …`**。

#### 第 2 步：`dispatch` → `run_on_qemu`（仍在宿主机）

9. `dispatch`（`:182-190`）：`export SIPU_CI_PLATFORM=qemu`；`== qemu` → **`run_on_qemu`**（`:186`）。
10. `run_on_qemu`（`:161-180`）依序做四件事：
    - `bash scripts/ci/sipu_ci_qemu.sh start`（`:164`）
    - `qemu_port="$(cat "$qemu_run/ssh.port")"`（`:165`，`qemu_run` 默认 `$PWD/qemu_run`）
    - `bash scripts/ci/sipu_ci_qemu.sh ssh "<exports>; cd $SGLANG && bash scripts/ci/sipu_ci_run.sh <三参数>"`（`:167-177`）——把 `SIPU_CI_USER`/`SIPU_CI_CUDA_DUMP`/`SIPU_CI_DUMP`/`SIPU_CI_WHEEL_DIR`/`SIPU_CI_QUEUE_DIR`/`SIPU_CI_IMAGE`/`SIPU_CI_KEEP_DUMPS=1`/`SIPU_CI_CI_ROOT`/`SIPU_CI_TMP_SGL_KERNEL_SIPU` 通过 `printf %q` 转义后注入 guest。
    - `bash scripts/ci/sipu_ci_qemu.sh stop`（`:178`），`return $rc`。

#### 第 3 步：`sipu_ci_qemu.sh start`（宿主）

11. 若 guest 已可达（`load_ports && guest_ssh_ok`，`:178`）→ 直接 `prepare_guest` + `record_qemu_pid` 复用。
12. 否则：
    - `resolve_cmodel`（`:44-56`）：读 `sgl-kernel-sipu/sipu_sdk_path.txt` 第一行拿到 SDK setup 脚本 → `SI_CMODEL_ROOT=<其目录>/sipu1.5_cmodel`，要求 `bin/qemu-system-x86_64` 与 `sipu_cmodel_setup.sh` 存在。
    - `stage_qemu_dir`（`:159-175`）：要求 `$QEMU_SRC/run_qemu.sh`、`$QEMU_SRC/archmodel.toml`、`$QCOW2_SRC` 存在（默认 `SIPU_CI_QEMU_SRC=/share_data/sglang_sipu/qemu_ci`、`SIPU_CI_QCOW2=$QEMU_SRC/torch_sipu.overlay.qcow2`，实测这是个 **~30.6 GB** 的 overlay 镜像）；把它们拷进 `$PWD/qemu_run/`，并把 qcow2 拷/链接到 `$PWD/torch_sipu.overlay.qcow2`（`SIPU_CI_SKIP_QCOW2_COPY=1` 且目标已存在则跳过拷贝）。
    - 在 **12000–12999** 里挑两个空闲端口作为 SSH/GDB（`pick_free_port`，`:24-42`），写入 `$QEMU_RUN/ssh.port`、`gdb.port`。
    - 后台启动（`:193-200`）：`cd $QEMU_RUN && SI_CMODEL_ROOT=… UBUNTU_IMAGE_FILE=$QCOW2_DST SSH_PORT=… GDB_PORT=… sg kvm -c './run_qemu.sh'`，stdout/stderr 进 `$QEMU_RUN/qemu.log`。
      （`run_qemu.sh` 内部：`source $SI_CMODEL_ROOT/sipu_cmodel_setup.sh`，把 `archmodel.toml` 覆盖成 `sipu_num=4`，并挂 3 个 9p：`share_data`→宿主 `/share_data`、`sipu_sw`→宿主 `$QEMU_RUN`、`local_data`→宿主 `/local_data`。）
    - 每 5 秒探一次 SSH，最多 90 次（`:203-211`）；超时则打印 `qemu.log` 尾部并失败。
    - `prepare_guest`（`:76-100`）：`sshpass -p 123456 ssh -p $SSH_PORT root@127.0.0.1`（`:58-67`），在 guest 内 `mount -t 9p … share_data /share_data`、`local_data /local_data`，`modprobe sipu sipu_timeout=-1`，轮询等 `/dev/sipu` 出现（最多 20 s）。
    - `record_qemu_pid`（`:139`）：按「cmdline 含本 job 的 `hostfwd=tcp::<port>-:22` **且**含本 job 的 qcow2 路径」唯一匹配到 `qemu-system-x86_64` 进程，写 `$QEMU_RUN/qemu.pid`（注释明确写了**绝不 `pkill` by name**）。

#### 第 4 步：ssh 进 guest 跑 `sipu_ci_run.sh`

13. `sipu_ci_qemu.sh ssh "…"`（`:279-282`）→ `qemu_ssh` 执行那串命令。
14. guest 内 `sipu_ci_run.sh` 与 archmodel 路径**同一份脚本**，只是 `detect_current_platform`（`sipu_ci_single_case.sh:10`）在 guest 里因为 `/dev/sipu` 存在而判定 **`SIPU_CURRENT_PLATFORM=qemu`**。
15. 校验参数、检查 `$SIPU_CI_WHEEL_DIR/*.whl`（`sipu_ci_run.sh:109`）；走单 case 分支（`:191-196`）写队列行
    ```
    deepseek|ds_v3_2layer|configs/deepseek/ds_v3_2layer.yaml|deepep_deepgemm_dp_ep|text-only|30m
    ```
    然后 `run_cases single 1`。

#### 第 5 步：guest 内起容器（`sipu_ci_single_case.sh`）

16. `run_one_case`（`:118`）：这次 `setup_docker_args`（`:123` → `:34-87`）会因为 `SIPU_CURRENT_PLATFORM=qemu` 而**追加** qemu 专属参数（`:65-86`）：
    - `-e SIRT_SHIM_NAME=ksipu --net=host`、`-v /opt/siorigin:/opt/siorigin:ro`
    - `--device /dev/sipu/usipu-ctl`、`/dev/sipu/usipu{0,1,2,3}`（缺任一则直接报错退出）
    - 若 guest 有 `/dev/dri`，追加所有字符/块设备。
17. 拼出容器内命令（`:126-130`）并 `docker run`（`run_container`，`:89-100`）：
    ```
    /bin/bash /sgl-workspace/sglang/scripts/ci/sipu_ci_test.sh \
      --config-yaml configs/deepseek/ds_v3_2layer.yaml \
      --launch-config deepep_deepgemm_dp_ep \
      --test-case text-only --timeout 30m
    ```
    容器名 `sipu-ci-ds_v3_2layer_deepep_deepgemm_dp_ep_text-only-<pid>`，镜像即 `SIPU_CI_IMAGE`。
18. 退出码写 `ci_queues/status/<cid>`（`:139`），`reclaim_host_ownership`（`:137` → `:104-116`；`SIPU_CURRENT_PLATFORM==qemu` 时直接 return，因为 9p 不支持 chown）。

#### 第 6 步：容器内（`sipu_ci_test.sh`）—— 本 case 与上一个例子的**关键差异**

19. `source setup.sh` / `setup_triton.sh` / `setup_tilelang.sh`，`unset TORCH_DEVICE_BACKEND_AUTOLOAD`，设 `LD_LIBRARY_PATH`（`:43-50`）。
20. **`case_uses_custom_deepep`（`:58-78`）读 `launch_configs.deepep_deepgemm_dp_ep.test_env.text-only.SGLANG_SIPU_USE_CUSTOM_DEEPEP` = `'1'` → rc=0 → `install_deepep=1`**（`:86-88`，打印 `=== case uses SGLANG_SIPU_USE_CUSTOM_DEEPEP, install deep_ep wheel ===`）。
    - 并且**强制要求 `$SIPU_CI_WHEEL_DIR/deep_ep*.whl` 存在**，否则 `exit 1`（`:95-98`）。
    - 与上一个 qwen3 例子（该 env 未设 → 跳过 deep_ep）形成对比。
21. `pip install --force-reinstall --no-deps` 装全部 wheel（deep_ep 这次**不跳过**，`:99-106`），并打印 `sgl_kernel.__file__`。
22. `is_compile=0`（名字不以 `_compile` 结尾，`:122-123`）→ 不开 tiled 环境、不做 compile contract 校验。
23. CUDA golden：`$SIPU_CI_CUDA_DUMP/ds_v3_2layer/deepep_deepgemm_dp_ep/text-only/cuda` 必须存在，否则 `exit 1`（`:138-143`）；存在则 symlink 到 `$SIPU_CI_DUMP/…`（`:145-147`）。
24. 跑 case（`:150-154`）：
    ```
    timeout 30m python3 -u test/srt/sipu/test_utils/run_test_job.py \
      --config-yaml <cfg> --launch-config deepep_deepgemm_dp_ep --test-case text-only \
      --device sipu --dump-base $SIPU_CI_DUMP --log-base $SIPU_CI_RUN_DIR/logs
    ```
25. 比对（`:188-198`）：`python3 -u test/srt/sipu/analyse_utils/compare_cuda_sipu.py <cfg> --dump-base … --launch-config … --test-case …`，tee 到 `$SIPU_CI_RUN_DIR/compare/ds_v3_2layer_deepep_deepgemm_dp_ep_text-only.log`。

#### 第 7 步：`run_test_job.py` 里这个 case 的实际语义

26. `main`（`test/srt/sipu/test_utils/run_test_job.py:482`）：
    - `launch_cfg = cfg["launch_configs"]["deepep_deepgemm_dp_ep"]`（`:498`），含 `cuda` + `sipu` 两块（`:504`）✓；`test_case` 在 `tests: [text-only]`（`:508`）✓；`TEST_REGISTRY["text-only"] → run_text_only`（`:514`）。
    - `_prompts`（`:109-115`）优先取 **launch_config 自己的 `prompts`** —— 本 config 在该 launch_config 下**覆盖成了两条完全相同的 prompt**（`configs/deepseek/ds_v3_2layer.yaml:59-67`，注释说明是为了让 **DP attention=2 时没有空闲 rank**，绕开 DeepEP 在「某 rank dispatch 0 token」时的 wait-send 挂死）。
    - `_apply_test_env`（`:118-127`）把 `test_env.text-only` 写进 `os.environ`：**`SGLANG_SIPU_USE_CUSTOM_DEEPEP=1`** 与 **`SGLANG_SIPU_USE_SICCL=1`**（`ds_v3_2layer.yaml:68-71`）。
    - `_engine_kwargs`（`:57-74`）取 `launch_cfg["sipu"]` 深拷贝 + `model_path` + `device=sipu` + `forward_hooks`（`enable_tensor_dump: true`）。
      即引擎按 `sipu:` 块（`ds_v3_2layer.yaml:94-118`）起：**`tp_size=2`、`dp_size=2`、`ep_size=2`、`enable_dp_attention=true`、`gpu_id_step=2`、`disable_overlap_schedule=true`**、`moe_runner_backend=deep_gemm`、`moe_a2a_backend=deepep`、`deepep_mode=low_latency`、`disable_shared_experts_fusion=true`、`attention_backend=sipu`。
27. 因此这个 case 的语义是：在 **sipu 设备（QEMU cmodel，4 个 archmodel 实例）** 上用 **TP2 × DP2 × EP2 + DP attention** 跑一份 MoE 模型（`DeepSeek-V3-2layer`），强制启用**自定义 DeepEP** 与 **SiCCL**，跑两条相同 prompt，dump 张量后与 CUDA golden 比 `cos_sim`/`mean_atol`。

### 46.3 与 `--platform archmodel` 的差异对照

| 环节 | `archmodel`（默认） | `qemu` |
|---|---|---|
| 编 wheel | 宿主编 5 个 wheel | **相同** |
| 分派 | `dispatch archmodel` → `run_local`（`:156`） | `dispatch qemu` → `run_on_qemu`（`:161`） |
| 执行环境 | 宿主机上直接 `docker run` | 先在宿主起 QEMU 虚拟机，再 ssh 进 guest 起容器 |
| `SIPU_CURRENT_PLATFORM` | `host` | `qemu`（guest 内有 `/dev/sipu`） |
| docker `--device` | 不加 | 加 `/dev/sipu/usipu*`、`/dev/dri/*`，`SIRT_SHIM_NAME=ksipu`、`--net=host` |
| `reclaim_host_ownership` | 会 chown | 直接 return（9p 不支持 chown，`:107`） |
| 额外清理 | 无 | 退出时 `sipu_ci_qemu.sh stop`（`:97` 与 `:178`） |

### 46.4 前置条件（任一不满足即失败）

1. `sgl-kernel-sipu/sipu_sdk_path.txt` 指向的 SDK 目录下有 `sipu1.5_cmodel/bin/qemu-system-x86_64`（`qemu.sh:44-56`）。
2. `/share_data/sglang_sipu/qemu_ci/` 下有 `run_qemu.sh`、`archmodel.toml`、`torch_sipu.overlay.qcow2`（实测 qcow2 ≈ 30.6 GB，每次会**拷贝**到 `$PWD/`，磁盘要留够）。
3. 宿主有 `sshpass`（`qemu.sh:59-62`）与 KVM（`run_qemu.sh` 会检查 `vmx/svm`）。
4. 12000–12999 端口有空闲。
5. **CUDA golden 必须已存在**：`$SIPU_CI_CUDA_DUMP/ds_v3_2layer/deepep_deepgemm_dp_ep/text-only/cuda`，否则容器里 `exit 1`。
6. 因为本 case 带 `SGLANG_SIPU_USE_CUSTOM_DEEPEP=1`，`$SIPU_CI_WHEEL_DIR` 里必须有 `deep_ep*.whl`（`sipu_ci_test.sh:95-98`）。

### 46.5 结束与判定

- guest 内跑完 → `run_on_qemu` 调 `sipu_ci_qemu.sh stop`（`:178`）：先 `ssh poweroff`，20 s 不退再 `SIGTERM`、再 10 s 不退再 `SIGKILL`；每一步都重新校验「这个 pid 仍是本 job 的 qemu」才动手（`:227-273`）。
- 宿主 `exec.sh` 退出时 `cleanup`（`:96-103`）再 `qemu.sh stop` 一次（幂等）、`reclaim_host_ownership`、删 `ci_queues`、按 `SIPU_CI_KEEP_WHEELS` 决定是否删 `ci_wheels`。
- 判定同 eager 门槛：case 退出码 0 且每个可比 pass `cos_sim >= 0.999`、`mean_atol <= 0.05`（`scripts/ci/sipu_ci_summary.py` docstring），汇总写 `<run-dir>/summary.tsv` / `summary.log`；结果目录为 `$CI_ROOT/single/<YYYYMMDD>`。

> 补充：`sipu_ci_suites.yaml` 里的 `distributed` suite（3 个 case）就是 `platform: qemu`，所以这条路径也正是 suite 模式下 `distributed` 的执行方式。
---

## 47. sikernel `rms_norm` 全链路解析：从 `test_host.cpp` 到 device kernel

**代码位置**：`/share/users/like/package/sikernel/source/source_builtin/attention/rms_norm/`
**ISA 文档**：`/softhome/like/asset/code/isa/index.html`（Tile Core Extension 指令集手册）
**运行方式**：`source setup.sh` → `cd source/source_builtin/attention/rms_norm` → `bash build.sh` → `./build/test_host`

### 47.0 一句话结论

这个 kernel 是**一行一个 thread（thread-per-row）的 RMSNorm**：每个 thread 负责 `(batch_size, normalized_size)` 矩阵的一整行，分两趟（two-pass）处理——**Pass 1** 把整行的 `x²` 累加求和得到 `mean`、开方取倒数得到 `rsqrt`；**Pass 2** 重新读一遍这一行，乘 `rsqrt`、可选乘 `weight`，写回输出。

关键设计有两点：

1. **Tile 寄存器 + 1024 字节定长切片**。`normalized_size` 被切成 `1024/sizeof(T)` 个元素一块的 tile（f32→256 元素，f16/bf16→512 元素），每块用一条 `tld`/`tst` 搬进/搬出 tile 寄存器。所以 wrapper 里那条"`normalized_size * sizeof(T) % 1024 == 0`"的对齐要求不是随便定的——**它是 kernel 没有 tail 处理所直接推导出来的约束**。
2. **shared memory 当 cache 用**。Pass 1 读进来的行数据顺手存进 `__shared__`，Pass 2 优先从 shared 读，避免把整行从 global 再读一遍。每 thread 独占 256 KB，两个 thread 合计 512 KB，正好吃满一颗 PE 的 SRAM。

### 47.1 目录与构建

```
rms_norm/
├── CMakeLists.txt                  # 两个 scc 编译单元 + 一个 g++ 测试可执行文件
├── build.sh                        # cmake -B build ./ && cmake --build build
├── run.sh                          # rm -rf build && build.sh && ./build/test_host
├── kernel/
│   ├── rms_norm_kernel.su          # ★ host 侧 wrapper：校验 / dispatch / launch
│   ├── legacy_api.su               # 裸指针兼容层（旧 ABI）
│   ├── rms_norm_kernel_f32.hpp     # ★ device kernel：f32
│   ├── rms_norm_kernel_f16.hpp     # ★ device kernel：f16
│   └── rms_norm_kernel_bf16.hpp    # ★ device kernel：bf16（含 strided-input 版本）
└── test/
    └── test_host.cpp               # ★ host 测试 + golden 参考实现
```

**构建断点**：`CMakeLists.txt` 把 `rms_norm_kernel.su` + `legacy_api.su` 编成 `librms_norm.so`（`scc -arch=sipu_150`），再把 `test_host.cpp` 用**宿主 g++**编成可执行文件并链接这个 `.so`、`libsirt` 与 `libsi`（`CMakeLists.txt:52-56`）：

```cmake
scc_add_library(rms_norm SHARED kernel/rms_norm_kernel.su kernel/legacy_api.su)
scc_target_compile_definitions(rms_norm PRIVATE TARGET_SIPU_ARCH=${TARGET_SIPU_ARCH})
add_executable(test_host test/test_host.cpp)
target_link_libraries(test_host PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/build/librms_norm.so
    sipurt
    sipu)
```

实际编译命令（`build/CMakeFiles/rms_norm.dir/build.make:84`）：

```
scc -arch=sipu_150 --keep-dir build/SipuWork.rms_norm -c -fPIC -o .../rms_norm_kernel.su.o \
    -MD -MT ... -MF ... \
    -I<sikernel>/include -I<sikernel>/source/source_builtin/utils \
    -DTARGET_SIPU_ARCH=150 kernel/rms_norm_kernel.su
```

> **注意 `-arch` 的取名**：不是 `si150`，而是 `sipu_150`。`scc --list-gpu-arch` 给出的是 `sipu_150` / `sipu_160` / `sipu_170`；`sikernel/setup.sh:39-55` 用的又是裸数字 `150/160/170`。三套写法互不相通，详见第 51.1.1 节。

`.su` 是"含 device 代码的编译单元"的后缀：`scc` 会把 `__global__` 函数编成 fatbin 塞进 `.so`，host 侧的普通 C++ 函数编成普通 ELF 符号。`build/SipuWork.rms_norm/librms_norm_sipu_fatbin.llvm.asm` 就是 device 代码的反汇编，本文后面用它来对照 ISA。

### 47.2 调用链总览

```
test/test_host.cpp
│
├─ main(argc, argv)                                     test_host.cpp:main
│   ├─ [--emu-multi 路径] parse_multi_case_options → run_case(..., emu_test_mode=true)
│   ├─ run_case(16, 7168, 7168, false, false)           ←★ 默认路径：连续 bf16
│   └─ run_strided_case(33, 512)                        ←★ 默认路径：非连续 bf16
│
├─ run_case(batch_size, normalized_size, original_normalized_size, ...)
│   ├─ sikernel::tensor<bf16> input_tensor({batch_size, normalized_size})   ← 自动推 strides
│   ├─ sipuMalloc / sipuMemcpy(H2D)
│   ├─ rms_norm<bf16>(out, in, w, eps, orig_norm, w_opt=1, eps_opt=1)      ← host API
│   │   └─ rms_norm_launch<bf16>(..., launch_timestamp_ns=nullptr, kernel_elapsed_ms=nullptr)
│   │       ├─ check_rms_norm_tensor_metadata(...)    ← 20 余条 check(...)
│   │       ├─ is_contiguous_layout(input/output/weight)
│   │       └─ rms_norm_contiguous_launch<bf16>(...)   ← 三选一 dispatch
│   │           └─ __global__ rms_norm_bf16_kernel<<<grid, cluster, block, 0, stream>>>(...)
│   │                └─ 每个 thread 一行：Pass1 求 rsqrt → Pass2 缩放写回
│   ├─ sipuMemcpy(D2H)
│   └─ golden(...) + 逐元素相对误差比对（阈值 1%）
│
└─ run_strided_case(batch_size=33, normalized_size=512)
    ├─ input_tensor.strides = {2176, 1}   ← 唯一被手动改写 strides 的地方
    ├─ rms_norm<bf16>(...)
    │   └─ rms_norm_launch<bf16>(...)
    │       └─ rms_norm_bf16_strided_input_launch(...)  ← 命中非连续分支
    │           └─ __global__ rms_norm_bf16_strided_input_kernel<<<...>>>(...)
    └─ golden_strided_input(...) + 比对
```

一句话概括层次：**test_host（数值验证）→ rms_norm（公开 API）→ rms_norm_launch（校验+dispatch）→ xxx_launch（grid/block 配置）→ __global__ kernel（真正的算法）**。

### 47.3 第 0 层：`test/test_host.cpp`

#### 47.3.1 头部与测试形状

```cpp
#define K  7168      // padded size, align to 512
#define VK 7168      // valid size
#define M  16
#define DATATYPE sifmt::bfloat16
```

`K` 是**补齐后**的 normalized 维长度（注释说 align to 512；bf16 下 512 元素 = 1024 B，正好一个 tile），`VK` 是"有效"长度，`M=16` 是 batch。注意默认用例里 `K == VK == 7168`，所以"补齐"这条路径在默认测试中**没有被覆盖**——后面 47.10 会讲这里藏了一个语义陷阱。

`test_host.cpp` 自己**重新声明**了两个模板函数，而不是从 `sikernel.h` 拿声明：

```cpp
template <typename T>
void rms_norm(sikernel::tensor<T>& output, sikernel::tensor<T>& input, sikernel::tensor<T>& weight,
              const float eps, const int64_t original_normalized_size, int weight_opt, int eps_opt,
              sipuStream_t stream);
```

这是因为 `sikernel.h` 里那个模板只在 `.so` 里做了**显式实例化**（`kernel/rms_norm_kernel.su:263-265`），头文件里没有定义，所以测试只需要一份签名一致的外部声明，链接期由 `librms_norm.so` 提供符号。C++ 模板 + `extern` 显式实例化是这套代码的通用手法。

#### 47.3.2 `main` —— 三条入口

```cpp
int main(int argc, char* argv[]) {
    std::cout << "argc:" << argc << "\n";                 // ← 本地调试时加的
    const auto multi_options = sikernel::emu_test::parse_multi_case_options(argc, argv);
    if (multi_options.requested) {
        // ./test_host --emu-multi 33:7168 [--no-compare] [--repeat N]
        for (const std::string &shape : multi_options.case_args) {
            // 解析 "<case_size>:<hidden_size>"
            pass = run_case(case_size, hidden_size, hidden_size, /*emu_test_mode=*/true,
                            multi_options.no_compare, multi_options.repeat) && pass;
        }
        SIKERNEL_TEST_RETURN(pass);
    }
    if (argc != 1) { SIKERNEL_TEST_RETURN(false); }       // 默认路径要求无参数
    bool pass = run_case(M, K, VK, false, false);         // 连续用例
    pass = run_strided_case(33, 512) && pass;             // 非连续用例
    SIKERNEL_TEST_RETURN(pass);
}
```

`SIKERNEL_TEST_RETURN(cond)`（`include/sikernel_test.h:47`）负责打印 ASCII 艺术字 **PASS**/**FAIL** 并返回 `0` / `-1`——run.log 末尾那个八角星图案就是它打的。

`--emu-multi` 模式下走的是另一条路：`run_case(..., emu_test_mode=true, ...)` 会改用 `rms_norm_timed`，用 `sipuEvent` 量 kernel 时间、用 `host_timestamp_ns()` 量 launch 前后的 host 时间，并把结果 append 到 CSV（`dsv3_op_result_rms_norm.log`），供 EMU 性能测试脚本消费。`SIKERNEL_EMU_TEST_SKIP_CSV=1` 可以关掉写盘。

#### 47.3.3 `run_case` —— 连续路径

```cpp
bool run_case(int batch_size, int normalized_size, int original_normalized_size,
              bool emu_test_mode, bool no_compare, int repeat_count = 1) {
    const int weight_opt = 1;
    const int eps_opt = 1;
    const int case_size = emu_test_mode ? batch_size : 0;
    const std::string emu_test_kernel_name = "rms_norm_" + std::to_string(normalized_size);

    size_t nIn0 = normalized_size * batch_size;   // 输入元素数
    size_t nIn1 = normalized_size;                // weight 元素数
    size_t nOut = normalized_size * batch_size;   // 输出元素数
    ...
    DATATYPE* host_A = (DATATYPE*)malloc(sizeIn0);
    DATATYPE* host_B = (DATATYPE*)malloc(sizeIn1);
    DATATYPE* host_D = no_compare ? nullptr : (DATATYPE*)malloc(sizeOut);   // device 结果回读
    DATATYPE* host_G = no_compare ? nullptr : (DATATYPE*)malloc(sizeOut);   // golden

    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(-10.0f, 100.0f);   // 输入：[-10, 100)
    std::uniform_real_distribution<float> dist1(-1.0f, 1.0f);     // weight：[-1, 1)

    for (size_t i = 0; i < nIn0; i++) {
        host_A[i] = (DATATYPE)(dist(gen));
        if (i % normalized_size >= original_normalized_size)
            host_A[i] = (DATATYPE)(dist(gen));      // ← 两个分支完全相同，历史遗留
    }
    for (size_t i = 0; i < nIn1; i++) host_B[i] = (DATATYPE)(dist1(gen));
```

那个 `if` 的两条分支写法一模一样，推测原本是想给 padding 区域填 0（早期测试用 `K > VK` 验证 padding），后来改成了随机填充但没删掉。默认用例 `K == VK` 时这个分支永远不成立，属于无害的历史残留。

继续：

```cpp
    void *bo_A; void *bo_B; void *bo_D;
    sipuMalloc(&bo_A, sizeIn0);
    sipuMalloc(&bo_B, sizeIn1);
    sipuMalloc(&bo_D, sizeOut);
    sipuMemcpy(bo_A, host_A, sizeIn0, sipuMemcpyHostToDevice);
    sipuMemcpy(bo_B, host_B, sizeIn1, sipuMemcpyHostToDevice);

    sikernel::tensor<DATATYPE> input_tensor({batch_size, normalized_size});
    sikernel::tensor<DATATYPE> output_tensor({batch_size, normalized_size});
    sikernel::tensor<DATATYPE> weight_tensor({normalized_size});
    input_tensor.data  = bo_A;
    output_tensor.data = bo_D;
    weight_tensor.data = bo_B;
```

注意这里**没有手动设置 strides**——`sikernel::tensor<T>` 的 `initializer_list` 构造函数（`include/sikernel_tensor.h:65`）会自动按行主序推 `strides = {normalized_size, 1}`（weight 是 `{1}`）。所以这条路径天然是 contiguous。

```cpp
    const float eps = 1e-6;
    std::cout << "kernel to launch!" << std::endl;

    if (emu_test_mode) {
        // 用 rms_norm_timed 跑 repeat_count 次，丢弃第 0 次（warm-up）
        // 统计 kernel_elapsed_ms / api_launch_ms / launch_end_ms / total_host_ms
        const int measured_repeats = sikernel::emu_test::measured_repeat_count(repeat_count);
        for (int repeat = 0; repeat < repeat_count; ++repeat) {
            uint64_t launch_timestamp_ns = 0;
            float kernel_launch_ms = 0.0f;
            const uint64_t api_start_ns = sikernel::host_timestamp_ns();
            launch_timestamp_ns = api_start_ns;
            rms_norm_timed<DATATYPE>(output_tensor, input_tensor, weight_tensor, eps,
                                     original_normalized_size, weight_opt, eps_opt, nullptr,
                                     &launch_timestamp_ns, &kernel_launch_ms);
            const uint64_t api_end_ns = sikernel::host_timestamp_ns();
            if (sikernel::emu_test::repeat_counts_towards_average(repeat, repeat_count)) { ... }
        }
    } else {
        rms_norm<DATATYPE>(output_tensor, input_tensor, weight_tensor, eps,
                           original_normalized_size, weight_opt, eps_opt);   // stream 默认 nullptr
    }
    std::cout << "kernel return!" << std::endl;
```

`measured_repeat_count(n) = max(n-1, 1)`、`repeat_counts_towards_average(i, n) = (n == 1 || i > 0)`——即 `--repeat N` 时**第 0 次只做 warm-up，不计入平均**。

```cpp
    if (!no_compare) sipuMemcpy(host_D, bo_D, sizeOut, sipuMemcpyDeviceToHost);
    sipuFree(bo_A); sipuFree(bo_B); sipuFree(bo_D);

    int err = 0;
    if (!no_compare) {
        golden(host_G, host_A, host_B, eps, batch_size, normalized_size,
               original_normalized_size, weight_opt, eps_opt);
        for (int i = 0; i < nOut; i++) {
            float err_rate = fabs(((float)host_G[i] - (float)host_D[i]) / (float)host_G[i]);
            if (err_rate > 0.01f) { err = 1; std::cerr << "Error at index: " << i << ...; }
        }
    }
```

判据是**相对误差 < 1%**，逐元素检查。注意 `golden` 是**在 device 结果回读之后**才算的（先 Free 了 device 内存），这点和 `run_strided_case` 的顺序不同但无影响。

#### 47.3.4 `golden` —— 参考实现

```cpp
void golden(DATATYPE* pout, DATATYPE* pin, DATATYPE* weight, float eps,
            int batch_size, int normalized_size, int original_normalized_size,
            int weight_opt, int eps_opt) {
    for (int j = 0; j < batch_size; ++j) {
        float sum = 0;
        for (int i = 0; i < normalized_size; ++i)                 // ★ 求和走满 normalized_size
            sum += pin[j * normalized_size + i] * pin[j * normalized_size + i];
        float mean = sum / original_normalized_size;              // ★ 分母是 original_...
        if (eps_opt) mean += eps;
        float rsqrt = 1 / sqrt(mean);
        for (int i = 0; i < normalized_size; ++i) {
            float dout = rsqrt * pin[j * normalized_size + i];
            if (weight_opt) dout = dout * weight[i];
            pout[j * normalized_size + i] = dout;
        }
    }
}
```

这段就是 RMSNorm 的定义，逐字对应 device kernel：
`mean = Σx² / original_normalized_size`（+eps）→ `rsqrt = 1/sqrt(mean)` → `y = x * rsqrt * w`。

**注意 golden 用的是双精度积累（`float sum` 是 f32 标量、但顺序是纯串行）**，而 device 侧用 tile 向量并行累加、归约顺序完全不同——两者必然有浮点级差异，所以阈值放到 1%（bf16 输入本身只有 8 位尾数）。

#### 47.3.5 `run_strided_case` —— 非连续路径

```cpp
bool run_strided_case(int batch_size, int normalized_size) {
    const int input_stride0 = 2176;      // ★ 外部约定的行 pitch
    const int weight_opt = 1;
    const int eps_opt = 1;
    const float eps = 1e-6;

    size_t nIn0 = input_stride0 * batch_size;   // 注意：按 stride 分配，输入缓冲区更大
    size_t nIn1 = normalized_size;
    size_t nOut = normalized_size * batch_size;
    ...
    sikernel::tensor<DATATYPE> input_tensor({batch_size, normalized_size});
    sikernel::tensor<DATATYPE> output_tensor({batch_size, normalized_size});
    sikernel::tensor<DATATYPE> weight_tensor({normalized_size});
    input_tensor.data = bo_A;
    input_tensor.strides = {input_stride0, 1};   // ★ 唯一一处手动改 strides
    output_tensor.data = bo_D;
    weight_tensor.data = bo_B;
    ...
    rms_norm<DATATYPE>(output_tensor, input_tensor, weight_tensor, eps, normalized_size,
                       weight_opt, eps_opt, nullptr);
```

两个关键点：

1. `input_tensor` 的 `sizes` 是 `{33, 512}`、`strides` 是 `{2176, 1}`——**逻辑形状和物理布局不一致**，第 j 行的起始地址是 `bo_A + j*2176`。H2D 拷贝也按 `nIn0 = 2176 * 33` 拷全量。
2. `output_tensor` 保持默认 strides `{512, 1}`（连续），`weight` 是 `{1}`。

`golden_strided_input` 对应地按 `row = pin + j * input_stride0` 取行，其余与 `golden` 一致；注意它的 `mean = sum / normalized_size`（没有 `original_normalized_size` 参数，因为 strided 分支强制要求两者相等）。

`2176` 这个数字**在本仓库里 grep 不到定义**，只能确认它是**外部框架强加的行 pitch**（很可能是 KV cache / MLA 的物理布局）。可以观察到的算术关系是 `2176 = 2048 + 128`，对应 512 个 bf16（1024 B）有效数据 + 1024 B 额外间距，即行 pitch 4352 B；但这只是从数字反推，**具体来源需要查调用方（sglang/vllm 侧的 KV cache 分配逻辑）才能确认**。

从 kernel 角度看，它完全不关心这个数字的来源——只是照用 `input.strides[0]`。所以设计上把它写成了"只支持精确匹配 `2176` 的两个 shape"的**白名单**，而不是通用 strided 支持：通用化需要处理任意 `stride0` 下的边界/对齐，成本高且当时没有第二个用例，白名单是性价比最高的做法。

### 47.4 第 1 层：host API `rms_norm`

`kernel/rms_norm_kernel.su:250-256`：

```cpp
template<typename T>
void rms_norm(sikernel::tensor<T>& output, sikernel::tensor<T>& input, sikernel::tensor<T>& weight,
              const float eps, const int64_t original_normalized_size, int weight_opt, int eps_opt,
              sipuStream_t stream) {
    rms_norm_launch<T>(output, input, weight, eps, original_normalized_size, weight_opt, eps_opt,
                       stream, /*launch_timestamp_ns=*/nullptr, /*kernel_elapsed_ms=*/nullptr);
}
```

它只是 `rms_norm_launch` 的一层薄封装，把计时相关的两个可选出参置空。同文件 `242-248` 还有 `rms_norm_timed`，把这两个出参透传给 launch——**测试里的 EMU 计时路径就是靠它**。

文件末尾 `258-265` 是 6 条显式实例化（3 dtype × 2 API），这就是为什么头文件里只放声明也能链接成功。

参数语义（与 `include/sikernel.h:1265-1277` 的 doc comment 一致）：

| 参数 | 含义 | 约束 |
| --- | --- | --- |
| `output` | 输出，2D | shape 必须与 input 完全相同；`data != nullptr` |
| `input` | 输入，2D `(B, N)` | `sizes[1]*sizeof(T) % 1024 == 0` |
| `weight` | 缩放权重，1D | `weight_opt != 0` 时必填，长度 == `sizes[1]` |
| `eps` | 数值稳定项 | `eps_opt != 0` 时加到 mean 上 |
| `original_normalized_size` | 求 mean 时的**分母** | `0 < it <= normalized_size` |
| `weight_opt` / `eps_opt` | 开关 | 非 0 即启用 |
| `stream` | 启流 | 默认 `nullptr` |

另有 `legacy_api.su` 提供裸指针重载：拿到 `(void* out, void* in, void* w, B, N, origN, ...)` 后，构造三个 contiguous tensor 再转调新 API。

```cpp
template<typename T>
void rms_norm(const void* output, const void* input, const void* weight, const float eps,
              const int64_t batch_size, const int64_t normalized_size,
              const int64_t original_normalized_size, int weight_opt, int eps_opt,
              sipuStream_t stream) {
    sikernel::tensor<T> output_tensor({batch_size, normalized_size});
    sikernel::tensor<T> input_tensor({batch_size, normalized_size});
    sikernel::tensor<T> weight_tensor({normalized_size});
    output_tensor.data = const_cast<void*>(output);
    input_tensor.data  = const_cast<void*>(input);
    weight_tensor.data = const_cast<void*>(weight);
    rms_norm<T>(output_tensor, input_tensor, weight_tensor, eps, original_normalized_size,
                weight_opt, eps_opt, stream);
}
```

即老 ABI 只能表达连续布局——**非连续支持是新 API 独有的能力**。

### 47.5 第 2 层：`rms_norm_launch` —— 校验与 dispatch

`kernel/rms_norm_kernel.su:204-240`，整个 wrapper 的核心：

```cpp
template<typename T>
void rms_norm_launch(sikernel::tensor<T>& output, sikernel::tensor<T>& input, sikernel::tensor<T>& weight,
                     const float eps, const int64_t original_normalized_size, int weight_opt, int eps_opt,
                     sipuStream_t stream, uint64_t *launch_timestamp_ns, float *kernel_elapsed_ms) {
    check_rms_norm_tensor_metadata(output, input, weight, original_normalized_size, weight_opt);

    const int64_t batch_size     = input.sizes[0];
    const int64_t normalized_size = input.sizes[1];
    const bool weight_contiguous = weight_opt ? is_contiguous_layout(weight) : true;

    if (is_contiguous_layout(input) && is_contiguous_layout(output) && weight_contiguous) {
        rms_norm_contiguous_launch<T>(output.data, input.data, weight.data, eps, batch_size, normalized_size,
                                      original_normalized_size, weight_opt, eps_opt, stream,
                                      launch_timestamp_ns, kernel_elapsed_ms);
        return;
    }

    if constexpr (std::is_same_v<T, sifmt::bfloat16>) {
        const bool supported_bf16_shape  = (normalized_size == 512 || normalized_size == 1536) &&
                                            original_normalized_size == normalized_size;
        const bool supported_input_stride  = input.strides[0] == 2176 && input.strides[1] == 1;
        const bool supported_output_stride = output.strides[0] == normalized_size && output.strides[1] == 1;
        const bool supported_weight_stride = !weight_opt || weight.strides[0] == 1;

        if (supported_bf16_shape && supported_input_stride && supported_output_stride && supported_weight_stride) {
            rms_norm_bf16_strided_input_launch(output.data, input.data, weight.data, eps, batch_size,
                                               normalized_size, original_normalized_size, input.strides[0],
                                               weight_opt, eps_opt, stream, launch_timestamp_ns, kernel_elapsed_ms);
            return;
        }
    }

    sipu::check(false,
        "rms_norm non-contiguous tensors only support bf16 exact input shape/stride "
        "(num_tokens,512)/(2176,1) or (num_tokens,1536)/(2176,1), "
        "with contiguous matching output and weight", __FILE__, __LINE__);
}
```

逐段拆解：

**① metadata 校验**（`check_rms_norm_tensor_metadata`，`:51-79`）：

| # | 检查 | 备注 |
| --- | --- | --- |
| 1 | `output.data != nullptr` | |
| 2 | `input.data != nullptr` | |
| 3 | `weight_opt` 时 `weight.data != nullptr` | |
| 4 | `input.dim == 2` / `output.dim == 2` | 只支持 2D |
| 5 | `sizes.size() == 2 && strides.size() == 2` | 防御 `sizes`/`strides` 与 `dim` 不一致 |
| 6 | `input.sizes[0] > 0 && input.sizes[1] > 0` | batch_size / normalized_size 必须为正 |
| 7 | `output.sizes == input.sizes` | 输出必须与输入同形 |
| 8 | **`input.sizes[1] * sizeof(T) % 1024 == 0`** | ★ 1024 B 对齐——kernel 无 tail 的直接后果 |
| 9 | `0 < original_normalized_size <= input.sizes[1]` | 分母必须落在有效区间 |
| 10 | `weight.dim == 1`、`sizes.size()==1`、`weight.sizes[0] == input.sizes[1]` | weight 长度必须等于 normalized_size |

**② 连续性判定**（`is_contiguous_layout`，`:36-49`）：

```cpp
template<typename T>
bool is_contiguous_layout(const sikernel::Tensor<T>& tensor) {
    if (tensor.dim == 0 || tensor.sizes.size() != tensor.dim || tensor.strides.size() != tensor.dim)
        return false;
    int64_t expected_stride = 1;
    for (int64_t i = tensor.dim - 1; i >= 0; --i) {
        if (tensor.sizes[i] <= 0 || tensor.strides[i] != expected_stride) return false;
        expected_stride *= tensor.sizes[i];
    }
    return true;
}
```

从最后一维往前推"理想 stride"（`1, N, N*M, ...`），任一维对不上就不是连续。对 `{B,N}/{N,1}` 和 `{N}/{1}` 都成立。

注意 `sikernel_tensor.h` 里另有一个 `check_contiguous_tensor`，那个是**断言版**（不连续直接 `check(false)` 报错），wrapper 用的是这个**返回 bool** 的本地版本，因为它要做 dispatch 而不只是校验。

**③ 三分支 dispatch**：

```
                ┌─ input/output/weight 全 contiguous ──► rms_norm_contiguous_launch  (dtype 泛化)
非连续输入 ─────┼─ bf16 且命中 (N∈{512,1536}, stride=(2176,1),
                │   out=(N,1), w=(1,), origN==N)      ──► rms_norm_bf16_strided_input_launch
                └─ 其他                                ──► check(false) 直接报错
```

设计取舍很清楚：**通用性只给到连续布局**，非连续只在"有明确业务需求（KV cache 布局）的那两个精确 shape"上开洞，其余一律 fail-fast。README 里那句"非连续输入仅支持 bfloat16，并且 normalized 维度和 stride 只支持以下两个完全匹配的 case"就是这里。

`supported_output_stride` 用的是 `output.strides[0] == normalized_size`（而不是显式写 1）——因为 output 要求连续，行 stride 必然等于 normalized_size。`supported_weight_stride` 在 `weight_opt == 0` 时直接为真，此时 `weight` 根本没被读。

### 47.6 第 3 层：launch 配置 —— grid / block 怎么算

`rms_norm_contiguous_launch`（`:83-160`）和 `rms_norm_bf16_strided_input_launch`（`:162-202`）用**完全相同的** grid/block 推导：

```cpp
uint32_t block_dim = 1;
uint32_t cluster_dim = 1;
uint32_t grid_dim = 1;

if (batch_size >= 2)  block_dim = 2;                 // 每 block 2 个 thread
if (batch_size >= 32) grid_dim = 16;                 // 大批量：grid 封顶 16
else                  grid_dim = (batch_size + 1) / 2;  // 小批量：每 block 摊 2 行

dim3 grid{grid_dim};
dim3 cluster{cluster_dim};     // 恒为 1
dim3 block{block_dim};
```

由此得到的 `blockIdx.x * blockDim.x + threadIdx.x`（即 kernel 里的 `tid`）与总线程数：

| batch_size B | block_dim | grid_dim | 总线程 | 每线程处理行数 | 覆盖情况 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1 | 1 | `bid=0`，恰好 |
| 2 | 2 | 1 | 2 | 1 | `bid=0,1`，恰好 |
| 3 | 2 | 2 | 4 | ≤1 | 线程 3 空转 |
| 16（默认） | 2 | 8 | 16 | 1 | 恰好 |
| 32 | 2 | 16 | 32 | 1 | 恰好 |
| 33（strided 用例） | 2 | 16 | **32** | ≤2 | 线程 0 串行处理 `bid=0,32` |
| 100 | 2 | 16 | 32 | ≤4 | 线程循环 3~4 次 |

可以看到：
- `grid_dim` 在 B ≥ 32 时**封顶 16**（16 block × 2 thread = 32 线程），此后增加 batch **不再增加并行度**，而是让每个线程串行跑多行（kernel 里的 `for (bid = tid; bid < batch_size; bid += all_threads)`）。
- `run_strided_case(33, 512)` 正好落在"33 行 / 32 线程"这个**非整除**的边界上——第 0 号线程要跑 `bid=0` 和 `bid=32` 两行。这是有意挑选的边界测试。

**thread 数为什么是 2？** 从后面 device 侧的结构就能反推：

```cpp
__shared__ bfloat16_t shared_buff[2][shared_buffer_size];   // 第一维 = 2
...
tst_linear_share_m1(bf16_din, shared_buff[threadIdx.x], i*tile_size);
...
bf16_din = tld_linear_share_m1(shared_buff[threadIdx.x], offset);
```

shared memory 的容量是 512 KB（fatbin 元数据里 `.sram = 524288` 可以印证），被 `[2]` 均分成两块 **256 KB**，用 `threadIdx.x` 选块。所以 `block_dim = 2` 不是"为了 2 路并行"那么简单——**它同时也是 shared memory 的分配维数**，写成 2 是因为 256 KB 恰好能装下 256 个 tile（256 × 1024 B = 256 KB），这是个容量最优解。如果 block_dim 改成 4，每块只剩 128 KB，能缓存的 tile 数减半。

> **关于 grid/block 到硬件的映射**：device 侧拿到的 `blockIdx.x` / `blockDim.x` / `threadIdx.x` 实际上来自 CSR `tkernelidx`(0x230) 和 `tkerneldims`(0x240)（ISA 文档 §CSR列表）。反汇编里这段是：
>
> ```
> tcsrr.r.b64 t0, tkernelidx
> tcsrr.r.b64 t3, tkerneldims
> ... mul / add ...
> zext.w  t0, t1        # t1 = 线性 thread id
> ```
>
> 即编译器把三级索引 `(tbc, tb, thread)` 线性化成 `(f(tbc_dim_x)·tbc_idx_x + tb_idx_x)·tb_dim + thread_idx` 的形式。因为 `cluster_dim` 恒为 1，cluster 那一项恒为 0，**结果退化成我们熟悉的 `blockIdx.x * blockDim.x + threadIdx.x`**。ISA 文档只给出 `tkernelidx`/`tkerneldims` 各字段的位宽，没有给出 `tb`/`tbc` 的具体形状（2D？多少 thread？），所以这里只做"语义等价于 CUDA 式三维索引"的结论，具体位域请以 SDK 头文件为准。

### 47.7 第 4 层：device kernel 逐行讲解（重点）

先建立 tile 寄存器的语言，后面逐行讲解才不会卡住。

#### 47.7.0 前置：tile 寄存器速查表

这不是 CUDA，也不是普通 RVV。这是一套**tile 扩展**：除 RV 的标量寄存器（`x0-x31` / `f0-f31`）和向量寄存器（`v0-v31`，1024 bit）之外，还有一整排 **Tile 寄存器**，每个 8192 bit = **1024 字节**。

ISA 文档明确写了这几条：

> 每个 Tile Reg 的容量 = **8192-bit = 1024-Byte**（1KB）；`tilesize` 编码 m1/m2/m4/m8 对应 1024B/2048B/4096B/8192B。
> 一条指令中多个 Tile Reg 的操作数**不允许重叠**。
> 指令无法操作编号 ≥ `tregsize`(CSR) 的 tile reg，硬上限 160。

C 层的类型名和子访问是这么来的（`include/si_dev_apis/dtype/union.h`）：

```c
typedef union {
    struct { tfloat16m1_t m1e0; tfloat16m1_t m1e1; };
    tfloat16m2_t m2e0;
} tfloat16m2;                     // 2 KB = 2 个 m1

// tfloat32m2 同构：
typedef union {
    struct { tfloat32m1_t m1e0; tfloat32m1_t m1e1; };
    tfloat32m2_t m2e0;
} tfloat32m2;
```

即 `m2` 是**两个 `m1` 的并集**，`m1e0`/`m1e1` 是它的两个 1 KB 半区。kernel 里大量出现 `xxx.m1e0` / `xxx.m1e1`，就是在拆/合这 2 KB。

于是各 dtype 下一个 tile（1 KB）装多少元素：

| 类型 | 单 tile 元素数 | 类型说明 |
| --- | --- | --- |
| `tfloat32m1_t` | 1024 B / 4 B = **256** | 32 行 × 32 B 排布 |
| `tfloat16m1_t` | 1024 B / 2 B = **512** | 32 行 × 32 B 排布 |
| `tbfloat16m1_t` | 1024 B / 2 B = **512** | 同上 |
| `tfloat32m2_t` | **512** | = 2 个 m1 |

**这就是 `tile_size = 1024` 与 `norm_tile_num = norm_size/512`（bf16/f16）或 `norm_size/256`（f32）的由来**——`tile_size` 的单位是**字节**，`norm_tile_num` 是"一行切几块"：f32 每块 256 元素，bf16/f16 每块 512 元素。kernel 里所有 `offset` / `batch_offset` 也都是**字节偏移**（`bid*norm_size*sizeof(T)`）。

再记住几个在 kernel 里高频出现的 intrinsic（C 层 → 汇编的映射由反汇编实证，见 47.8）：

| C intrinsic | 汇编（本 kernel 反汇编实测） | 语义 |
| --- | --- | --- |
| `tld_linear_global_m1(ptr, off)` | `tld.trir.linear.u32.global` | 从 **global** 读 1 KB（=1 个 tile）到 tile 寄存器 |
| `tst_linear_global_m1(t, ptr, off)` | `tst.trir.linear.u32.global` | 把 tile 寄存器 1 KB 写回 **global** |
| `tst_linear_share_m1(t, ptr, off)` | `tst.trir.linear.u32.share` | 写 1 KB 到 **shared memory** |
| `tld_linear_share_m1(ptr, off)` | `tld.trir.linear.u32.share` | 从 **shared memory** 读 1 KB |
| `twait_store_share(n)` | `twait.i.store.share n` | 等待未完成的 share store 数 ≤ n（0 = 全部排空） |
| `tcvt_f32(tile, t_r32)` | `tcvt.tt.f32.bf16.r32` | 1 KB bf16(512) → 2 KB f32(512，拆成 m1e0/m1e1) |
| `tcvt_bf16(tile, t_r32)` | `tcvt.tt.bf16.f32.r32` | 2 KB f32 → 1 KB bf16 |
| `tmul(a, b)` / `tadd(a, b)` | `tmul.ttt.f32` / `tadd.ttt.f32` | tile 逐元素乘/加（无 fma，见 47.10） |
| `tmv_t_f32(v)` | `tmv.trr.broadcast` | 标量 → 广播填满整个 tile 寄存器 |
| `tmv_v_f32(tile, i)` | `tmv.vtr.e32` | 取 tile 里第 i 个 **128 B 块**（32 个 f32）到 RVV 向量寄存器（**128 B 的来历见 47.7.1 末尾的专门小节**） |
| `twait_store_share` | `twait.i.store.share` | 见上 |

`t_r32` 来自 `tile_vector.h`（`const unsigned t_r32 = 0x02;`），是 `tcvt` 的**shape / 布局属性**——ISA 文档里 matrix conversion 的语法是 `tcvt.tt.<dtype>.<atype>.<shape>`，其中 shape 编码 32 行 = `r32`、16 行 = `r16`、8 行 = `r8`。反汇编中确实生成成了 `.r32` 后缀，说明它选的是"32 行"的转换布局。

#### 47.7.1 `rms_norm_bf16_kernel` 逐行（`rms_norm_kernel_bf16.hpp:25-112`）

这是默认用例（`sifmt::bfloat16`，M=16，K=7168）实际执行的 kernel。**下面逐行讲解。**

---

**① 签名（line 25）**

```cpp
__global__ void rms_norm_bf16_kernel(sifmt::bfloat16* pout, sifmt::bfloat16* pin, sifmt::bfloat16* weight,
                                     float eps, int64_t batch_size, int64_t norm_size,
                                     int64_t original_norm_size, int weight_opt, int eps_opt)
```

| 参数 | 值（默认用例） | 说明 |
| --- | --- | --- |
| `pout` / `pin` | `bo_D` / `bo_A` | device 全局指针，行主序 `(16, 7168)` |
| `weight` | `bo_B` | 长度 7168 |
| `eps` | `1e-6f` | |
| `batch_size` | 16 | 逻辑行数，非 grid 维 |
| `norm_size` | 7168 | 补齐后的行长度 |
| `original_norm_size` | 7168 | **求 mean 的分母** |
| `weight_opt` / `eps_opt` | 1 / 1 | |

---

**② 局部常量与 tile 变量（line 29-45）**

```cpp
  const int tile_size = 1024;
  __andescore_fp_mode(BF16);
```

`tile_size = 1024` 是**字节**，即一个 tile 寄存器的大小。`__andescore_fp_mode(BF16)` 切换 core 的 16 位浮点解释模式。


> 这个 intrinsic **不在 ISA 文档里**（文档只描述指令，不描述这种 core 级模式开关）。从源码看（`__clang_nds_device_functions.h /share_data/sicx_sdk/release/latest/bin/nds64le-elf-newlib-v5d/lib/clang/20/include/__clang_siorigin_device_functions.h `）它的实现是：
> ```c
> enum AndesFpMode { FP16 = 0, BF16 = 1 };
> __device__ __attribute__((weak)) void __andescore_fp_mode(AndesFpMode mode) {
>     unsigned long long fp_mode_mask = ~0x3;
>     unsigned long long umisc_ctl = __nds__read_csr(NDS_UMISC_CTL) & fp_mode_mask;
>     if (mode == BF16) umisc_ctl |= 1;
>     __nds__write_csr(umisc_ctl, NDS_UMISC_CTL);
> }
> ```
> 即把 `NDS_UMISC_CTL` 的 bit[1:0] 置成 `01`(BF16) 或 `00`(FP16)。它是一个**弱符号**，所以 f32 kernel 不调用它也不会链接失败。注意它是 `mode` 而非"当前 dtype"——f32 kernel 里完全不出现，因为 f32 无需歧义。

```cpp
  tbfloat16m1_t bf16_din;      // 1 KB bf16 = 512 个输入元素
  tbfloat16m1_t bf16_wgt;      // 1 KB bf16 = 512 个 weight 元素
  tfloat32m2    f32_din;       // 2 KB f32 = 512 个，由 bf16_din 转换而来，m1e0/m1e1 各 256
  tfloat32m2    f32_wgt;       // 同上，weight 版
  tfloat32m2    f32_sq;        // 2 KB f32 = 512 个，存 x*x
  tfloat32m2    f32_sq_sum;    // 2 KB f32 = 512 个，跨 tile 的累加器（分两半 m1e0/m1e1）
  tfloat32m1_t  tile_rsqrt;    // 1 KB f32 = 256 个，装广播后的 1/sqrt(mean)
  float sum;                   // 标量归约结果
  float mean;                  // sum / original_norm_size (+eps)
  float sqrt;                  // fsqrt 结果
  tfloat32m2    f32_out;       // 2 KB f32 = 512 个，Pass 2 的输出
  tbfloat16m1_t bf16_out;      // 1 KB bf16 = 512 个，写回 global
```

整套变量就是一条流水线：`bf16 → f32(2KB) → 平方累加 → 归约成标量 → 广播回 tile → 乘 → f32 → bf16 → 写回`。

```cpp
  int64_t norm_tile_num = norm_size/512;                     // 7168/512 = 14
  const int shared_buffer_size = 512*1024/sizeof(bfloat16_t)/2;   // = 131072
  __shared__ bfloat16_t shared_buff[2][shared_buffer_size];
```

- `norm_tile_num = 14`：一行 7168 个 bf16，每个 tile 512 个，共 14 块。
- `shared_buffer_size = 512*1024 / 2 / 2 = 131072`（元素数）：512 KB SRAM / sizeof(bf16)=2 B / 2（两个 thread 平分）= 131072 个 bf16 = **262144 B = 256 KB**。
- `shared_buff[2][131072]` 总占用 `2 × 131072 × 2 B = 524288 B = 512 KB`，与 fatbin 元数据 `.sram = 524288` 完全一致。

**这里就埋下了缓存容量的定义**：每个 thread 能缓存 `131072 / 512 = 256` 个 tile（256 × 1 KB = 256 KB）。对 K=7168（14 个 tile）远远够用；但如果 `norm_size > 256*512 = 131072`，超出的部分就缓存不下、只能退回 global 重读（见 Pass 2 的 `if/else`）。

---

**③ 线程索引与主循环（line 46-48）**

```cpp
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int all_threads = gridDim.x * blockDim.x;
  for(long bid = tid; bid < batch_size; bid += all_threads){
```

标准的 **grid-stride loop**。默认用例 B=16：`block_dim=2, grid_dim=8` → `all_threads=16`，每个线程恰好处理 1 行。strided 用例 B=33：`all_threads=32`，线程 0 要跑 `bid=0` 和 `bid=32`。

**一个 thread 独占一行**是这个 kernel 的根本设计——正因如此，行内归约完全不需要跨线程通信（没有 `__syncthreads`、没有 warp shuffle、没有 shared memory reduce），只要 tile 内部和向量寄存器内部归约就够了。这也是为什么 shared memory 能放心地按 `[threadIdx.x]` 私有切分。

---

**④ 行首地址与累加器清零（line 49-51）**

```cpp
    int64_t batch_offset = bid*norm_size*sizeof(bfloat16_t);   // 字节偏移：bid*7168*2
    f32_sq_sum.m1e0 = tmv_t_f32(0U);   // 用 SpecialNumber 0 初始化
    f32_sq_sum.m1e1 = tmv_t_f32(0U);
```

`tmv_t_f32(0U)` 是"标量→tile 广播"（`tmv.trr.broadcast`），`0U` 是 `SpecialNumber::Zero` 的编码。反汇编里这段被展开成：

```
tmv.trr.broadcast  T0, s1, zero                # T0 = 全 0 的 1KB tile
tmv.ttrr.u128      T1, T0, 0, 0                # 拷贝 128B 块 0
tmv.ttrr.u128      T1, T0, 1, 1                # 拷贝 128B 块 1
... 共 8 条 ...                                 # 凑满 1KB (= 8 × 128B)
```

编译器把"1 KB 广播"实现成了 8 次 128 B 的 tile→tile 拷贝——这印证了 tile 内部以 **128 B** 为最小操作粒度（也正是 `tmv.vtr` / `tmv.ttr` 的粒度和 RVV 向量寄存器 1024 bit 的宽度）。

---

**⑤ Pass 1：累加平方和（line 52-64）**

```cpp
    for(int64_t i=0; i<norm_tile_num; ++i){                        // i = 0..13
      int64_t offset =  batch_offset + i*tile_size;                // 字节偏移
      bf16_din = tld_linear_global_m1((bfloat16_t*)pin, offset);   // ① 读 1KB (512 个 bf16)
      if(i<shared_buffer_size/512){                                // ② 顺手存 shared
        tst_linear_share_m1(bf16_din, shared_buff[threadIdx.x], i*tile_size);
      }

      f32_din.m2e0 = tcvt_f32(bf16_din, t_r32);                    // ③ 1KB bf16 -> 2KB f32
      f32_sq.m1e0 = tmul(f32_din.m1e0,f32_din.m1e0);               // ④ 平方（低半）
      f32_sq.m1e1 = tmul(f32_din.m1e1,f32_din.m1e1);               //    平方（高半）
      f32_sq_sum.m1e0 = tadd(f32_sq_sum.m1e0,f32_sq.m1e0);         // ⑤ 累加（低半）
      f32_sq_sum.m1e1 = tadd(f32_sq_sum.m1e1,f32_sq.m1e1);         //    累加（高半）
    }
```

逐句：

- **① `tld_linear_global_m1(pin, offset)`**：从 global 读 **1 KB = 512 个 bf16** 到 `bf16_din`。这是一个**异步的 tile 访存指令**，由 tile core 自己完成搬运，不占用 RV 标量流水线（这也是后面必须 `twait` 的原因）。
- **② `tst_linear_share_m1(bf16_din, shared_buff[threadIdx.x], i*tile_size)`**：把刚读到的 1 KB **同时**写进本线程的 shared 区、第 `i` 个 tile 槽位。注意偏移用的是**行内相对偏移** `i*tile_size`（不是 `offset`），因为缓存区是"每线程每行"复用的，所以行首偏移 `batch_offset` **不需要**加进去。这就是 Pass 2 能省一次 global 读的全部秘密。
  - 条件 `i < shared_buffer_size/512`（= `i < 256`）是缓存容量上限保护。当 `norm_tile_num > 256`（即 `norm_size > 131072`）时，前 256 个 tile 进缓存、后面的不缓存，Pass 2 再逐 tile 判断。
- **③ `tcvt_f32(bf16_din, t_r32)`**：1 KB bf16（512 个）→ 2 KB f32（512 个），结果装进 `tfloat32m2` 的 `m2e0`，`m1e0` 拿前 256 个、`m1e1` 拿后 256 个。反汇编是 `tcvt.tt.f32.bf16.r32 T10, T6`，与 ISA 文档 §Tile Conversion 中 `TCVT_F32_BF16`（输入 1×1024B → 输出 2×2048B）完全对应。
- **④⑤ `tmul` / `tadd`**：都是 `ttt` 形式（tile × tile → tile），在 f32 域做，避免 bf16 精度损失。分两次处理 `m1e0`/`m1e1` 是因为 `tadd`/`tmul` 的 m1 形式一次只吃 1 KB。

**这个循环走完后**，`f32_sq_sum.m1e0 + f32_sq_sum.m1e1` 里就存着整行 7168 个元素的平方和（分成 2×256 = 512 个部分和）。

---

**⑥ 归约到标量（line 65-78）**

```cpp
    f32_sq_sum.m1e0 = tadd(f32_sq_sum.m1e0,f32_sq_sum.m1e1);   // 两个 256 元素半区先合并
    vfloat32m1_t vsum_vec = tmv_v_f32(f32_sq_sum.m1e0, 0U);    // 取第 0 个 128B 块(32 个 f32)
    vfloat32m1_t zero_vec = __riscv_vfmv_v_f_f32m1(0.0f, 1);   // 初始 0
    vfloat32m1_t temp_vec;
    for (long i = 1; i < 8; ++i) {                             // 剩余 7 个 128B 块
        temp_vec =  tmv_v_f32(f32_sq_sum.m1e0, i);             // 取出第 i 个块
        vsum_vec =  __riscv_vfadd_vv_f32m1(temp_vec, vsum_vec, 32);  // 逐 lane 相加
    }
#ifdef RVV_UNORDER_REDUCE
    zero_vec = __riscv_vfredusum_vs_f32m1_f32m1(vsum_vec, zero_vec, 32);   // 乱序归约
#else
    zero_vec = __riscv_vfredosum_vs_f32m1_f32m1(vsum_vec, zero_vec, 32);   // 顺序归约
#endif
    sum =  __riscv_vfmv_f_s_f32m1_f32(zero_vec);               // 向量 -> 标量
```

这段是**两段式归约**：

1. **tile 内 → RVV 向量**：`tmv_v_f32(tile, i)` 把 tile 寄存器里第 `i` 个 **128 B 块**（32 个 f32）搬进一个 `vfloat32m1_t`（RVV，1024 bit）。一个 m1 tile = 1 KB = **8 个 128 B 块**，所以循环是 `i = 0..7`。反汇编确认：`tmv.vtr.e32 v9, T1, zero` / `tmv.vtr.e32 v10, T1, s2` … 配 7 条 `vfadd.vv`。
2. **RVV 向量内 → 标量**：先 8 个块**逐 lane 相加**（得到一个 32 lane 的部分和向量），再 `vfredosum` 做跨 lane 归约成 1 个数，最后 `vfmv.f.s` 把向量第 0 lane 搬到 `fa3`。

##### 为什么是 128 B？—— 这个数的完整来源

`tmv.vtr` 的第二个参数（`0U`、`i`）是**块序号**，不是字节偏移。为什么"一块"正好是 128 B，而不是 32 B、256 B？有四条独立证据交叉印证。

**块大小 128 B 与元素类型无关**（这一点实测确认）：f32 和 bf16 搬的**是同一个 128 B**，只是元素个数不同——

```asm
; tmv_v_f32(t, i)  ->  vsetvli a1, zero, e32, m1, ta, ma
;                      tmv.vtr.e32  v8, T0, a0         SEW=32 → 128/4 = 32 个 f32
; tmv_v_bf16(t, i) ->  vsetvli a1, zero, e16, m1, ta, ma
;                      tmv.vtr.e16  v8, T0, a0         SEW=16 → 128/2 = 64 个 bf16
```

两者指令里的 `rs2`（块序号 `a0`）语义完全一样，变的只是 `vsetvli` 设定的 `SEW`/`vl`。**"128 B"是 tile 侧的固定结构，与数据类型解耦**——这也是下面第 (2) 条算术能成立的原因。

**（1）ISA 手册的规定：`rs2` 索引的粒度是 128 B。**

手册在 move 类指令的字段说明里写明（`/softhome/like/asset/code/isa/index.html`，「Tile Tile to Vector」与字段表一节）：

> `rs2`：64-bit 整形标量寄存器编号，tile 寄存器的 index，**可按 32B 或 128Byte 维度索引**，粒度按照操作的元素宽度设计。……在操作 **128B** 时该值**大于等于 8 是未定义行为**；在操作 32B 时该值大于等于 32 是未定义行为。

以及 Tile→Vector 的语义：

> **该指令将 `rs2` 指定 index 的 128Byte 的 Tile 寄存器的数据 move 到 `vd` 指定的向量寄存器。**

"index ≥ 8 是未定义行为"这句话本身就是答案的一半——**只有 8 个块，所以块大小必然是 1024 B / 8 = 128 B**。内核里 `for (long i = 1; i < 8; ++i)` 的上界 8，正是从这个约束来的，不是随手写的。

**（2）算术自洽：128 B × 8 = 1024 B = 一个 tile 寄存器。**

```
1 个 tile reg        = 8192 bit = 1024 B        （第 47.7.0 节的速查表）
手册给的块数上限      = 8                          （index ≥ 8 未定义）
=> 块大小             = 1024 B / 8 = 128 B
```

**（3）RVV 侧宽度也正好是 128 B —— 这才是"能直接搬"的原因。**

这一步是**最关键的一环**：搬运的源块和目的寄存器必须等宽。这个平台的 **VLEN 是 1024 bit = 128 B**，编译器源码里写死了：

```c
// compiler-toolchain/llvm-project/llvm/lib/Target/RISCV/RISCVProcessors.td:1469-1471
def SiOrigin_SIPU150 : SiOriginSIPUModel<"sipu150", ...,
                                         [FeatureVendorXSOTile150G,
                                          // sipu150 has a vlen of 1024b
                                          FeatureStdExtZvl1024b]>;
```

注意 **150 是 1024b，而 160/170 是 256b**（同文件 `:1472-1479` 的注释："`sipu160` has a vlen of 256b"、":1478 同 170"）。**这个数字很重要，它决定了"一块 128 B 在 RVV 侧要占几个寄存器"**：

| | VLEN | 一个 RVV `m1` 寄存器 | 装下 128 B 需要 |
| --- | --- | --- | --- |
| **150**（本 kernel 用的） | 1024 bit = **128 B** | 128 B | **1 个 m1** |
| 160 / 170 | 256 bit = **32 B** | 32 B | **4 个 m1 → m4** |

**这一差别直接体现在函数名上**，而且是 150 与 160/170 的 `tmv` 接口不兼容的原因：

```c
// 150g（siorigin_tile_150g.h:9964）——没有粒度后缀，返回 m1
TILE_HEADER(tmv_vtr_f32) vfloat32m1_t tmv_v_f32(tfloat32m1_t src, unsigned long index);

// 160g（siorigin_tile_160g.h:7294 / :7334）——粒度写进名字，返回类型跟着变
TILE_HEADER(tmv_vtr_u128_f32) vfloat32m4_t tmv_v_u128_f32(tfloat32m1_t src, unsigned long index);
TILE_HEADER(tmv_vtr_u32_f32)  vfloat32m1_t tmv_v_u32_f32 (tfloat32m1_t src, unsigned long index);
```

**150 上之所以没有 `u32`/`u128` 前缀，是因为它的 VLEN 恰好就是 128 B，只有一种"能一次装完"的粒度**，不需要区分。我用 `-mcpu=` 实测过三种 target：

```asm
; -mcpu=sipu150  ->  tmv.vtr.e32        v8, T0, a0   (vfloat32m1_t,  128 B)
; -mcpu=sipu160  ->  tmv.vtr.u128.e32   v8, T0, a0   (vfloat32m4_t,  128 B)
;                              tmv.vtr.u32.e32    v8, T0, a0   (vfloat32m1_t,   32 B)
```

`u32`/`u128` 就是**粒度的字节数**，而 LMUL 自动选成"刚好等于该粒度"的那个。这也再次印证了第 (1) 条——ISA 说的"可按 32B 或 128Byte 维度索引"，两种粒度在硬件上都存在，只是 150 的头文件只暴露了 128 B 那一种（`grep tmv_vtr_u32 150g` 命中 0 次）。

**对本 kernel 的启示**：`for (i = 1; i < 8; ++i)` 这个上界、以及 `tmv_v_f32` 这个函数名，**都是 150 专属的**。要把这段归约移植到 160/170，函数名得换成 `tmv_v_u128_f32`（此时返回的是 `vfloat32m4_t`，后续 `vfadd` 也要换成 `m4`），否则根本编译不过——**不是悄悄变慢，是直接报 `use of undeclared identifier`**。

---

回到算术。在 150 上，f32 的情况下两边刚好都是 128 B = **32 个 f32**：

| | 宽度 | f32 元素数 |
| --- | --- | --- |
| tile 里的一块（128 B） | 128 B | 32 |
| 一个 RVV `vfloat32m1_t`（1×VLEN） | 128 B | 32 |

**两个 32 完全吻合，所以一条 `tmv.vtr.e32` 就能把一个块整块搬进一个 RVV 寄存器**，不需要循环、不需要掩码。这也解释了代码里那个看起来突兀的 `32`——`__riscv_vfadd_vv_f32m1(temp_vec, vsum_vec, 32)` 的 `vl=32` 就是"一块的 lane 数"，而 ISA 手册对 `tmv.vtr.e32` 的要求恰好也是 `vset(i)vl(i) SEW=32, vl=32, lmul=1`（实测反汇编里就是 `vsetvli a5, zero, e32, m1, ta, ma`，而且 `vfadd.vv` 前面**没有再插 `vsetvli`**——因为 `vlmax` 对 `e32,m1` 正好是 32，两个指令的向量配置天然一致，无需切换）。

**换个角度看第 (1) 条**：手册给的两种粒度，各自的上界乘出来是同一个数——`8 × 128 B = 32 × 32 B = 1024 B`，正好一个 tile 寄存器。**两种粒度殊途同归，这本身就是"tile = 1024 B"的又一次独立确认。**

**（4）反汇编实测。**

构建产物 `build/SipuWork.rms_norm/librms_norm_sipu_fatbin.llvm.asm` 里，索引寄存器的装载值一清二楚：

```asm
# —— 函数序言里，把"块序号 1..7"一次性装进 callee-saved 寄存器 ——
8040000000da: 4905        li  s2, 0x1
8040000000dc: 4989        li  s3, 0x2
8040000000de: 4a0d        li  s4, 0x3
8040000000e0: 4a91        li  s5, 0x4
8040000000e2: 4b15        li  s6, 0x5
8040000000e4: 4b99        li  s7, 0x6
8040000000e6: 4c1d        li  s8, 0x7        # ← 到 7 为止，共 8 块

# —— 归约处：1 条 vsetvli 打头，然后 8×tmv + 7×vfadd ——
8040000001e6: vsetvli a5, zero, e32, m1, ta, ma
8040000001ea: tmv.vtr.e32  v9,  T1, zero     # 块 0（常量 0 → 直接编成 x0）
8040000001f2: tmv.vtr.e32  v10, T1, s2       # 块 1
8040000001fa: vfadd.vv     v9,  v10, v9
8040000001fe: tmv.vtr.e32  v10, T1, s3       # 块 2
804000000206: vfadd.vv     v9,  v10, v9
...                                          # 一直到 s8（块 7）
80400000023a: tmv.vtr.e32  v10, T1, s8       # 块 7
804000000242: vfadd.vv     v9,  v10, v9
804000000246: vfredosum.vs v9,  v9,  v8
80400000024a: vfmv.f.s     fa3, v9
```

**7 条 `li`（1..7）+ 1 条隐含的 `0` = 8 个块**，与 `1024 B / 128 B = 8` 完全对上。寄存器分配也印证了"不变的部分提前算好"：编译器把 1~7 这七个常量在**函数序言**里一次装进 `s2`-`s8`（callee-saved，跨外层 `bid` 循环复用），循环体里只剩 `tmv.vtr` + `vfadd.vv` 两条指令。

**还有一处值得注意**：整段归约只有**开头一条 `vsetvli`**，后面 7 条 `vfadd.vv` 前面都没有再插——对 `e32, m1` 而言 `vlmax = VLEN/32 = 32`，正好等于代码里传的 `32`，所以向量配置一次设定后全程有效，没有多余的 `vsetvl` 切换开销。

**小结这条链**：

```
ISA: rs2 索引按 128B，且 index ≥ 8 未定义
        └─→ 隐含 8 个块
tile reg = 1024 B（8192 bit）
        └─→ 块大小 = 1024 / 8 = 128 B
VLEN(150) = 1024 bit = 128 B（编译器写死 zvl1024b）
        └─→ 1 个 m1 寄存器正好 = 1 块，一条 tmv.vtr 搬完
128 B / 4 B = 32 个 f32
        └─→ 循环上界 8、vl=32 全部自洽
（旁证：另一种粒度 32 B 的上界是 32 → 32 × 32 B = 1024 B，殊途同归）
（注意：VLEN 是 150 专属，160/170 为 256b，函数名与 LMUL 都不同）
```

**注意 `i` 的单位是"块"不是"字节"**，所以第二个参数写 `i` 而不是 `i*128`。这一点和内核里 `tld/tst` 那些按字节传 `offset` 的访存指令不同，容易混——判别方法是看手册里该字段写的是"index"还是"offset"。

`RVV_UNORDER_REDUCE` 宏切换 `vfredusum`（乱序，快，结果不保证顺序）与 `vfredosum`（有序，慢，结果确定）。**当前构建没有定义这个宏**（反汇编里是 `vfredosum.vs`），所以默认走顺序归约——这是为了**跨平台/跨版本位精确可复现**，代价是慢一点。

```cpp
    mean = sum/original_norm_size;
    if(eps_opt) mean += eps;
    asm volatile ( "fsqrt.s %0, %1" : "=f"(sqrt) : "f"(mean) );
    sqrt = 1.0/sqrt;
    tile_rsqrt = tmv_t_f32(sqrt);
```

- `mean = sum / original_norm_size`：★ 分母是 **`original_norm_size`（调用方给的"有效长度"）**，而分子是 `norm_size`（补齐后长度）个元素的和。默认用例两者相同。
- `mean += eps`：eps 加在 mean 上（不是加在方差的 `+eps` 意义上，但效果等价于 `rsqrt(x²/N + eps)`）。
- `fsqrt.s` 手写内联汇编 + `1.0/sqrt`：**没有用 `rsqrt` 近似指令**，而是精确开方再取倒数。反汇编是 `fsqrt.s fa3, fa3` 紧跟 `fdiv.s fa3, fa4, fa3`（`fa4` 是常量 1.0）。这是精度优先的选择——`rsqrt` 类近似指令在 1% 容差下其实也够，但这里没走捷径。
- `tmv_t_f32(sqrt)`：把这个**标量**广播填满整个 1 KB tile（256 个 f32 全等于 `1/sqrt(mean)`）。反汇编是 `fmv.x.w a5, fa3` + `tmv.trr.broadcast T2, a5, zero`（标量必须先搬到整型 GPR，这是 ISA 文档明确说的："浮点寄存器需先 move 到整型寄存器"）。

---

**⑦ 同步点（line 88）**

```cpp
    twait_store_share(0);
```

**这是整个 kernel 唯一的同步指令**，位置在 Pass 1 与 Pass 2 之间，语义是"等待未完成的 **share memory store** 数量降到 ≤ 0"，即**排空所有挂起的 shared 写入**。

为什么必须插在这里：`tst_linear_share_m1`（Pass 1 里发的）在 tile core 上是**异步执行**的，指令发射出去就返回；而 Pass 2 的 `tld_linear_share_m1` 要读同一块内存。如果没有这条 `twait`，Pass 2 可能读到尚未落盘的旧数据。反汇编里就是一条 `twait.i.store.share 0x0`。

注意这里**只需要等 store**：Pass 1 的 global load 结果是通过 tile 寄存器依赖（`tcvt` 直接消费 `bf16_din`）隐式串行化的，硬件自行保证；只有 shared 写这种"通过内存传递"的依赖需要显式 fence。

同时值得注意 **Pass 1 和 Pass 2 之间没有任何线程间同步**——因为每个线程只碰自己的行、自己的 shared 分区，天然无竞争。

---

**⑧ Pass 2：缩放并写回（line 89-109）**

```cpp
    for(int64_t i=0; i<norm_tile_num; ++i){
      int64_t offset = i*tile_size;                 // 行内字节偏移
      int64_t g_offset = batch_offset +offset;      // 全局字节偏移
      if(i<shared_buffer_size/512){
        bf16_din = tld_linear_share_m1(shared_buff[threadIdx.x], offset);   // 命中缓存
      }else{
        bf16_din = tld_linear_global_m1((bfloat16_t*)pin, g_offset);        // 回退 global
      }
      f32_din.m2e0 = tcvt_f32(bf16_din, t_r32);      // 再次 bf16 -> f32
      f32_out.m1e0 = tmul(f32_din.m1e0,tile_rsqrt);  // 乘 rsqrt（低半）
      f32_out.m1e1 = tmul(f32_din.m1e1,tile_rsqrt);  // 乘 rsqrt（高半）
      if(weight_opt){
        bf16_wgt =  tld_linear_global_m1((bfloat16_t*)weight, offset);      // 读 1KB weight
        f32_wgt.m2e0 = tcvt_f32(bf16_wgt, t_r32);
        f32_out.m1e0 = tmul(f32_out.m1e0,f32_wgt.m1e0);
        f32_out.m1e1 = tmul(f32_out.m1e1,f32_wgt.m1e1);
      }
      bf16_out = tcvt_bf16(f32_out.m2e0, t_r32);     // 2KB f32 -> 1KB bf16
      tst_linear_global_m1(bf16_out,(bfloat16_t*)pout, g_offset);   // 写回 global

    }
```

逐句：

- **`tile_rsqrt` 复用技巧**：`tile_rsqrt` 只有 1 KB（256 个 f32），而 `f32_din.m1e0`/`m1e1` 各是 1 KB。因为 `tile_rsqrt` 里 256 个值**全部相同**（广播出来的），所以拿它分别乘两半，等价于用 512 个相同的 rsqrt 去乘 512 个输入——一次性覆盖整行的一个 tile，**省掉了一半的标量广播开销**。
- **`tmul` 用的是精确乘法**：整条链 `bf16 → f32 → ×rsqrt → ×w → f32 → bf16` 全程在 f32 域算，只有出入口做 bf16 舍入。这保证了 bf16 输入下仍能满足 1% 判据。
- **`weight` 每次都从 global 重新读**（14 次），没有走 shared 缓存。weight 长度与 norm_size 相同，理论上也能缓存进 shared——但 shared 已被输入行占满（256 KB/线程），所以只缓存输入、不缓存 weight。
- **写回用 `tst_linear_global_m1`**（同样是异步），kernel 结束时由硬件保证 store 完成；这里**不需要**再补 `twait`，因为下一次循环的 `bid` 对应不同行、不同地址，无 RAW 依赖；跨 kernel 的可见性由 stream 的顺序语义保证。
- **`norm_tile_num` 在 Pass 2 里被复用**（而不是重新计算），保证两趟的切片完全一致。
- `#ifdef` 那段的 `printf` 注释掉了——调试残留。

**⑨ 收尾（line 111）**

外层 `for(bid...)` 结束，kernel 返回。

---

#### 47.7.2 `rms_norm_bf16_strided_input_kernel` 与连续版的差异

`rms_norm_kernel_bf16.hpp:114-201`。结构与连续版**逐行同构**，只有三处不同，全部围绕"输入的行 pitch ≠ 行长度"：

| 位置 | 连续版 | strided 版 |
| --- | --- | --- |
| 签名 | 无 `input_stride0` | 多了 `int64_t input_stride0` |
| 行基址 | `batch_offset = bid*norm_size*sizeof(bf16)` | `input_batch_offset = bid*input_stride0*sizeof(bf16)`（**输入用 stride**）<br>`output_batch_offset = bid*norm_size*sizeof(bf16)`（**输出仍连续**） |
| Pass 1 读 | `offset = batch_offset + i*tile_size` | `input_offset = input_batch_offset + i*tile_size` |
| Pass 1 存缓存 | `..., i*tile_size` | `..., offset`（= `i*tile_size`，**相对偏移，不受 stride 影响**） |
| Pass 2 读缓存 | `..., offset` | `..., offset`（同上） |
| Pass 2 回退路径 | `g_offset` | `input_offset` |
| Pass 2 写回 | `g_offset` | `output_offset = output_batch_offset + offset` |

**要点**：

1. **一套输入偏移、一套输出偏移**。输入按 `input_stride0`（=2176）跨行，输出按 `norm_size`（=512）跨行。这就是为什么 dispatch 里要单独检查 `output.strides[0] == normalized_size`。
2. **shared 缓存完全不受影响**。缓存写入用的是行内相对偏移 `offset`（每行都从 0 开始），因此无论 stride 多大，缓存布局都一样。这是该设计最优雅的一点——**加非连续支持没有让缓存逻辑变复杂一个字节**。
3. **其余部分（归约、rsqrt、weight、写回）逐字节相同**，包括 `mean = sum/original_norm_size` 和 `twait_store_share(0)` 的位置。
4. 因为 dispatch 强制 `original_normalized_size == normalized_size`，strided 版里 `original_norm_size` 恒等于 `norm_size`——但从代码看它仍然把两个参数分开传，保持与连续版一致（将来若要放开这个限制，kernel 不用改）。

对应实测：`run.log` 里 `strided kernel to launch, batch_size=33, normalized_size=512`，Pass 1 只跑 `norm_tile_num = 512/512 = 1` 次，缓存路径永远命中。

---

#### 47.7.3 `rms_norm_f16_kernel`（`rms_norm_kernel_f16.hpp:25-112`）

与 bf16 版**结构 100% 一致**，只有 dtype 替换：

| 项 | bf16 版 | f16 版 |
| --- | --- | --- |
| FP 模式 | `__andescore_fp_mode(BF16)` | `__andescore_fp_mode(FP16)` |
| tile 类型 | `tbfloat16m1_t` | `tfloat16m1_t` |
| 上行转换 | `tcvt_f32(bf16_din, t_r32)` | `tcvt_f32(f16_din, t_r32)` |
| 下行转换 | `tcvt_bf16(f32_out.m2e0, t_r32)` | `tcvt_f16(f32_out.m2e0, t_r32)` |
| `norm_tile_num` | `norm_size/512` | `norm_size/512` |
| `shared_buffer_size` | `512*1024/sizeof(bfloat16_t)/2` | `512*1024/sizeof(float16_t)/2` |
| 其余 | 完全相同 | |

f16 和 bf16 都是 2 字节，所以切块数、shared 容量、cache 槽位数完全一致。**这两个 kernel 除了 `__andescore_fp_mode` 与 tcvt 的方向后缀外，可以认为是同一份代码**——ISA 文档里也确认了 f16↔f32 与 bf16↔f32 的转换是两套独立的 uop（`TCVT_F32_F16` vs `TCVT_F32_BF16`），不能互相替代。

#### 47.7.4 `rms_norm_f32_kernel`（`rms_norm_kernel_f32.hpp:25-98`）

f32 版是唯一**结构不同**的：

| 项 | bf16 / f16 版 | f32 版 |
| --- | --- | --- |
| FP 模式 | 设 BF16 / FP16 | **不设**（f32 无歧义） |
| 输入 tile 类型 | `tbfloat16m1_t`（512 元素） | `tfloat32m1_t`（256 元素） |
| 上行转换 | 需要 `tcvt_f32` | **不需要**，本来就是 f32 |
| 累加器 | `tfloat32m2`（m1e0 + m1e1，512 个） | `tfloat32m1_t`（**只有 256 个，无 m1e1**） |
| 归约前合并 | `m1e0 = tadd(m1e0, m1e1)` | **无此步** |
| `norm_tile_num` | `norm_size/512` | `norm_size/256` |
| `shared_buffer_size` | `512*1024/2/2 = 131072` | `512*1024/4/2 = 65536` |
| 输出 | `tcvt_bf16` 后 1 KB 写回 | **直接** `tst_linear_global_m1(f32_out, ...)` |
| 运算 | `m1e0`/`m1e1` 各做一遍 | 单条 `tmul` / `tadd` |

f32 版的核心循环：

```cpp
    f32_sq_sum= tmv_t_f32(0U);
    for(int64_t i=0; i<norm_tile_num; ++i){
      int64_t offset =  batch_offset + i*tile_size;
      f32_din = tld_linear_global_m1(pin, offset);
      if(i<shared_buffer_size/256){                       // 65536/256 = 256 个槽位
        tst_linear_share_m1(f32_din, shared_buff[threadIdx.x], i*tile_size);
      }
      f32_sq = tmul(f32_din,f32_din);                     // 单条，无 m1e1
      f32_sq_sum = tadd(f32_sq_sum,f32_sq);
    }
    // 无 m1e0/m1e1 合并，直接进 8 路 RVV 归约
```

注意 cache 槽位数的表达式：f32 是 `shared_buffer_size/256`，bf16/f16 是 `shared_buffer_size/512`——分母就是"单 tile 元素数"，写法虽不一致但语义相同，都是 **256 个槽位**。

**一个反直觉的点**：f32 每个 tile 只装 256 个元素，输入数据量却是 bf16 的两倍，所以 **f32 kernel 的 global 流量反而是 bf16 的 2 倍**（同样的元素数、更宽的 dtype），且 tile 数翻倍（`norm_size/256` vs `norm_size/512`）。但省掉了两次 `tcvt`。这是典型的"带宽换算力"取舍。

#### 47.7.5 四个 kernel 差异总表

| | `rms_norm_f32_kernel` | `rms_norm_f16_kernel` | `rms_norm_bf16_kernel` | `..._bf16_strided_input_kernel` |
| --- | --- | --- | --- | --- |
| 文件 | `_f32.hpp:25` | `_f16.hpp:25` | `_bf16.hpp:25` | `_bf16.hpp:114` |
| 行数 | 74 | 88 | 88 | 88 |
| `__andescore_fp_mode` | — | `FP16` | `BF16` | `BF16` |
| 单 tile 元素 | 256 | 512 | 512 | 512 |
| `norm_tile_num` | `N/256` | `N/512` | `N/512` | `N/512` |
| tile 寄存器用量 | 8 | 19 | 19 | 19 |
| SRAM 用量 | 524288 | 524288 | 524288 | 524288 |
| 上行 tcvt | 免 | `tcvt_f32` | `tcvt_f32` | `tcvt_f32` |
| 下行 tcvt | 免 | `tcvt_f16` | `tcvt_bf16` | `tcvt_bf16` |
| 输入跨行步长 | `norm_size` | `norm_size` | `norm_size` | **`input_stride0`** |
| 输出跨行步长 | `norm_size` | `norm_size` | `norm_size` | `norm_size` |

（`tile 寄存器用量` / `SRAM 用量` 来自 `build/SipuWork.rms_norm/librms_norm_sipu_fatbin.llvm.resource`，是编译期静态资源核算：`.treg = 8/19`、`.sram = 524288`。注意 f32 只用 8 个 tile 寄存器——因为它只需同时驻留 3 个 m1 tile，而 16 位版本要驻留 m2 中间量。）

### 47.8 intrinsic → ISA 指令映射（反汇编实证）

下面左列是 kernel 源码里的 C intrinsic，右列是 `librms_norm_si_fatbin.llvm.asm` 中**实际生成的汇编**（以 f32 kernel 为例），中间是 ISA 手册里的语法原型。

| 源码 | 实测汇编 | ISA 手册原型 |
| --- | --- | --- |
| `tld_linear_global_m1(pin, off)` | `tld.trir.linear.u32.global T4, (a1), 0x0, s11` | `tld.trir.linear.u32.global.[m2..mf8].[tm] Td,(rs1),imm1,rs3` |
| `tst_linear_global_m1(t,pout,off)` | `tst.trir.linear.u32.global T3, (a0), 0x0, s10` | `tst.trir.linear.u32.global... Ts1,(rs1),imm1,rs3` |
| `tst_linear_share_m1(t,sh,off)` | `tst.trir.linear.u32.share T4, (s1), 0x0, s10` | `tst.trir.linear.u32.share...`（`tuop`=001） |
| `tld_linear_share_m1(sh,off)` | `tld.trir.linear.u32.share T5, (s1), 0x0, s11` | `tld.trir.linear.u32.share...` |
| `twait_store_share(0)` | `twait.i.store.share 0x0` | `twait.i.store.share cnt` |
| `tmv_t_f32(v)` | `tmv.trr.broadcast T2, a5, zero` | `tmv.trr.[broadcast] Td, rs1, rs2` |
| `tmv_v_f32(t,i)` | `tmv.vtr.e32 v9, T1, zero`（i 走 `s2..s8`） | `tmv.vtr.e32 vd, Ts1, rs2` |
| `tcvt_f32(bf16, t_r32)` | `tcvt.tt.f32.bf16.r32 T10, T6` | `tcvt.tt.<dtype>.<atype>.<shape>` |
| `tcvt_bf16(f32, t_r32)` | `tcvt.tt.bf16.f32.r32 ...` | 同上 |
| `tmul(a,b)` | `tmul.ttt.f32 T6, T4, T4` | `tmul.ttt.f32...[reuse].[neg].[tm] Td,Ts1,Ts2` |
| `tadd(a,b)` | `tadd.ttt.f32 T1, T1, T6` | `tadd.ttt.f32...` |
| `f32_sq_sum = tmv_t_f32(0U)` | `tmv.trr.broadcast T0,s1,zero` + 8×`tmv.ttrr.u128` | 标量→tile 广播；tile→tile 128B 拷贝 |
| `__riscv_vfredosum_*` | `vfredosum.vs v9, v9, v8` | （RVV 标准指令，非 tile 扩展） |
| `fsqrt.s` asm | `fsqrt.s fa3, fa3` + `fdiv.s fa3, fa4, fa3` | （RV 标量浮点） |

**助记符命名规则**（ISA 手册 §汇编指令命令方式）：`tld.` 是指令名，随后的 `trir`/`trii`/`trr` 描述操作数类型（`t`=tile reg、`r`=scalar GPR、`i`=immediate，按 `Td/rs1/imm/rs3` 顺序），`linear` 是 unit-stride 模式，`u32` 是 32 B 粒度，`global`/`share` 是访存空间，`[m2/m4/mf8...]` 是访问的 tile 数（**m1 省略不写**），`[tm]` 是 tile mask。

**128 B 粒度的证据链**：ISA 手册说 `tmv.vtr.e32` 是把 tile 里 **128 Byte** 的数据搬到 RVV 向量寄存器；反汇编里"清零 1 KB tile"被展开成 8 条 `tmv.ttrr.u128`（8 × 128 B = 1024 B）；`tmv.vtr` 的循环正好是 `i = 0..7`（8 × 32 个 f32 = 256 = 一个 m1）。三条独立证据互相印证 tile 内部以 128 B 为最小操作单元。

**grid/block 的 CSR 来源**：反汇编开头

```
tcsrr.r.b64 t0, tkernelidx      # 读 kernel 索引 CSR
tcsrr.r.b64 t3, tkerneldims     # 读 kernel 维度 CSR
slli/srli/mul/add ...           # 线性化
zext.w  t0, t1                  # 得到 tid
```

`tkernelidx`(0x230) / `tkerneldims`(0x240) 是 ISA 手册 §CSR列表 里的只读寄存器，由 CCS 在 kernel 启动前配置、kernel 用 `tcsrr` 读回。文档给出的 `tkernelidx` 位域是 `[3:0]=thread_idx, [15:4]=tb_idx_x, [23:16]=tb_idx_y, [31:24]=tb_idx_z, [47:32]=tbc_idx_x, ...`，`tkerneldims` 是 `[3:0]=tb_dim, [15:4]=tbc_dim_x, ...`。硬件层级是 **Chip → PEC(cluster) → PE**（`thwid` = 2 bit core_id / 2 bit pe_id / 3 bit cluster_id / 9 bit chip_id），全机最大线程规模在 Memory Barrier 一节给出：`256 chip × 16 cluster × 4 PE × 2 thread = 32768`——**每 PE 2 个 thread**，这正好解释了为什么 `block_dim` 的最大值是 2、为什么 `shared_buff[2][...]` 的第一维是 2：**它就是每 PE 的硬件线程数**。

### 47.9 shared memory 缓存机制小结

把三件事串起来看，能看出这块 512 KB SRAM 的设计意图：

| 事实 | 出处 |
| --- | --- |
| 每 PE 的 tile SRAM = 512 KB | fatbin 元数据 `.sram = 524288` |
| 每 PE = 2 个硬件线程 | ISA 文档 memory barrier 上限算式 `4 PE × 2 thread` |
| `__shared__ T shared_buff[2][512*1024/sizeof(T)/2]` | 源码，恰好 2 × 256 KB |
| 每线程缓存 256 个 tile（256 KB） | `shared_buffer_size / 单tile元素数 = 256` |
| 单行最多 256 个 tile → `norm_size ≤ 131072`（bf16）时全命中 | 推导 |

**缓存的有效性边界**：`norm_size ≤ 256 × 512 = 131072`（bf16/f16），或 `≤ 256 × 256 = 65536`（f32）时，**整行全部命中缓存，Pass 2 零 global 读**。典型 LLM 的 hidden size（4096 / 7168 / 8192）都远小于这个阈值，所以**实际线上场景 100% 命中**。超过阈值时，前 256 个 tile 命中、其余回退 global 重读——功能仍正确，只是退化为"两趟都读 global"。

**成本/收益**：Pass 1 多写一次 shared（1 KB/ tile），Pass 2 省一次 global 读（1 KB/tile）。因为 shared 带宽远高于 global、且写入与计算重叠，这是净赚的。代价是 512 KB SRAM 被这张 kernel 独占，**同 PE 上不能并发跑第二个 block 的同一 kernel**（occupancy = 1 block/PE）。

### 47.10 数值语义与三个容易踩的坑

#### 坑 1：`original_normalized_size` 的隐含契约——**padding 区必须填 0**

```cpp
mean = sum / original_norm_size;    // ← 分母是 original
// 而 sum 是 norm_size 个元素的平方和 ← ← 分子是 padded 长度
```

当 `norm_size > original_normalized_size`（存在 padding）时，求和**仍然覆盖 padding 区**，但分母只用有效长度：

```
mean = ( Σ_{i < norm_size} x_i² ) / original_norm_size       ← 实际行为
     = ( Σ_{i < origN}     x_i² + Σ_{padding} x_i² ) / origN
```

这本身**不是 bug，而是一个隐含契约**：只要调用方保证 padding 区**填 0**，`Σ_{padding} x_i²` 恒为 0，实际行为就等价于 `(Σ_{i<origN} x_i²) / origN`，即标准的 RMSNorm。这也正是"pad 到 512 元素对齐"这个做法的由来——**padded RMSNorm 是 LLM 推理里的常规技巧**（对齐到向量宽度/缓存行，再把多余位置零）。

**风险点在于契约是隐式的**：wrapper 只校验 `0 < original_normalized_size <= normalized_size`，既不检查、也不清零 padding。如果调用方拿一块复用过的、未清零的缓冲区当输入，padding 区的脏数据会被计入平方和、污染 mean。

`test_host.cpp` 里那段"两个分支写法一模一样"的 `if` 正是这个契约留下的痕迹：

```cpp
for (size_t i = 0; i < nIn0; i++) {
    host_A[i] = (DATATYPE)(dist(gen));
    if (i % normalized_size >= original_normalized_size)
        host_A[i] = (DATATYPE)(dist(gen));     // ← 原意大概是赋 0，现在两个分支一样
}
```

原意应当是"padding 区填 0 以验证契约"，后来改成了随机填充但没删掉判断。

**当前测试完全没有覆盖 `norm_size > origN` 的场景**：默认用例 `K == VK == 7168`；`--emu-multi` 路径也把同一个 `hidden_size` 当作 `case_size`/`normalized_size`/`original_normalized_size` 三个参数传（`run_case(case_size, hidden_size, hidden_size, ...)`）。所以这条 padding 路径只被代码逻辑覆盖、没有测试覆盖。如果生产中要依赖它，建议补一个 `normalized_size > original_normalized_size` 且 padding 填 0 的用例。

#### 坑 2：没有用 FMA，`x*x` 与 `Σ` 是两条独立指令

```cpp
f32_sq.m1e0 = tmul(f32_din.m1e0, f32_din.m1e0);
f32_sq_sum.m1e0 = tadd(f32_sq_sum.m1e0, f32_sq.m1e0);
```

反汇编里对应 `tmul.ttt.f32` + `tadd.ttt.f32` 两条（不是 `tfma`）。ISA 手册里有三操作数的 `tfma.tttt/tttr/ttti`（`TALU_FMA_FP32`），这里没用。影响：

- **吞吐**：每 tile 多一条指令；
- **精度**：`x²+acc` 分成两步，中间 `x²` 会被舍入一次，理论上比 FMA 差半个 ulp。

但归约本身是 f32、输入是 bf16（8 位尾数），这点差异远低于 1% 判据，所以不是问题。不过如果将来要提高精度/性能，把这一对换成 `tfma` 是第一个可以动的地方。

#### 坑 3：`vfredosum` vs `vfredusum` 的构建开关

```cpp
#ifdef RVV_UNORDER_REDUCE
    zero_vec = __riscv_vfredusum_vs_f32m1_f32m1(vsum_vec, zero_vec, 32);
#else
    zero_vec = __riscv_vfredosum_vs_f32m1_f32m1(vsum_vec, zero_vec, 32);
#endif
```

当前构建**未定义** `RVV_UNORDER_REDUCE`（反汇编是 `vfredosum.vs`），走**有序归约**。有序归约牺牲并行度换位精确可复现——对 EMU/CI 里"同一输入必须得到逐位相同输出"的比对场景是必要的。反过来说，**如果哪天有人打开这个宏，CI 里的逐位比对可能会挂**，而数值上没有任何问题。

### 47.11 实测：编译与运行

```
$ source setup.sh
$ cd source/source_builtin/attention/rms_norm
$ bash build.sh
$ ./build/test_host
```

`run.log` 输出（已实测复现）：

```
[SIRT] Library:0.4.2.3ec8ff9.Release @ /share_data/sicx_sdk/release/2609101917/lib/libsipu.so.0
argc:1
,batch_size:16,normalized_size:7168,original_normalized_size:7168,emu_test_mode:0,no_compare:0,repeat_count:1
[SIRT] Environment Shim: auto, detected Shim: swemusp, version: .../si1.5/2609080400/lib/libarchmodel.so
kernel to launch!
kernel return!
in golden function !!!
strided kernel to launch, batch_size=33, normalized_size=512
strided kernel return!
in strided golden function !!!
 ******      *       *****   *****          ← PASS 横幅（SIKERNEL_TEST_RETURN）
 *     *    * *      *       *
 ...
```

逐行对应：

| 输出 | 含义 |
| --- | --- |
| `[SIRT] Library ...` | 运行时（SIRT）加载 `libsipu.so`（`CMakeLists.txt:55` 的 `sipu`） |
| `argc:1` | 无参数 → 走默认用例 |
| `,batch_size:16,...` | `run_case` 开头的 `// like_debug` 调试打印（本地未提交改动），非必要输出 |
| `... detected Shim: swemusp` | 用 `swemusp` shim 跑 **archmodel（cmodel）**而非真实硬件 |
| `kernel to launch!` / `kernel return!` | `run_case` 里 launch 前后的打印 |
| `in golden function !!!` | `golden()` 被调用 → 说明**逐元素比对全部通过**（有失败会打 stderr 的 `Error at index:`） |
| `strided kernel to launch, batch_size=33, normalized_size=512` | `run_strided_case` 进入 |
| `in strided golden function !!!` | strided 用例也比对通过 |
| `****** * ***** *****` | PASS 横幅 |

**没有任何 `Error at index:` 输出**，两个用例（连续 bf16 7168、非连续 bf16 512/stride 2176）均通过。

**关于工作区里那两处 `// like_debug` 调试打印**：`main()` 开头和 `run_case()` 开头各有一行 `std::cout` 调试输出（`argc:...` 和 `,batch_size:...`），是未提交的本地改动。上面的实测就是在**带这两行打印的工作区原样**上跑通的，`bash build.sh` 返回 0、`./build/test_host` 返回 0。

> 本文早期版本曾记录这里有一处 `emu_test`（正确名 `emu_test_mode`）的编译错误 —— **该问题已修复**，当前工作区第 175 行是 `<< ",emu_test_mode:" << emu_test_mode << ...`，可以正常编译。上面 `run.log` 里多出来的 `,batch_size:16,...` 那一行就是这两处调试打印的产物。运行时若直接 `./build/test_host` 报 `cannot open shared object file: libsipu.so.0`，是因为没 `source setup.sh`——`LD_LIBRARY_PATH` 不跨 shell 保留，需在同一条命令里先 source。

`--emu-multi` 路径（EMU 性能测试用）：

```bash
SIKERNEL_EMU_TEST_SKIP_CSV=1 ./build/test_host --emu-multi 1:512
# 多 case：./build/test_host --emu-multi 16:7168 33:7168 --repeat 5
```

### 47.12 完整调用链时序（默认用例，M=16, K=7168, bf16）

```
main()
 ├─ parse_multi_case_options → requested=false（argc==1）
 ├─ run_case(16, 7168, 7168, emu_test_mode=false, no_compare=false)
 │   ├─ malloc host_A(16×7168 bf16) / host_B(7168) / host_D / host_G
 │   ├─ 随机填充：输入 U(-10,100)、weight U(-1,1)
 │   ├─ siMalloc bo_A / bo_B / bo_D
 │   ├─ siMemcpy H2D ×2
 │   ├─ tensor 构造：input{16,7168} strides{7168,1}; output 同; weight{7168} strides{1}
 │   ├─ rms_norm<bf16>(out, in, w, eps=1e-6, origN=7168, w_opt=1, eps_opt=1, stream=nullptr)
 │   │   └─ rms_norm_launch<bf16>(..., nullptr, nullptr)
 │   │       ├─ check_rms_norm_tensor_metadata → 10 类断言全过
 │   │       │   └─ 7168 × 2 B = 14336 B ≡ 0 (mod 1024) ✓
 │   │       ├─ is_contiguous_layout(input/output/weight) → 全 true
 │   │       └─ rms_norm_contiguous_launch<bf16>
 │   │           ├─ B=16: block_dim=2, grid_dim=(16+1)/2=8, cluster_dim=1
 │   │           └─ rms_norm_bf16_kernel<<<grid{8}, cluster{1}, block{2}, 0, nullptr>>>(...)
 │   │               ├─ all_threads = 8×2 = 16，tid ∈ [0,16)，每 thread 1 行
 │   │               ├─ 每 thread: norm_tile_num = 14 个 tile
 │   │               ├─ Pass 1: 14 × (tld 1KB → tst share → tcvt → tmul → tadd ×2)
 │   │               ├─ 归约: tadd(m1e0,m1e1) → 8×tmv.vtr+vfadd → vfredosum → sum
 │   │               ├─ mean=sum/7168 (+1e-6) → fsqrt.s → 1/x → tmv_t_f32 广播
 │   │               ├─ twait_store_share(0)
 │   │               └─ Pass 2: 14 × (tld share → tcvt → tmul rsqrt ×2 → tmul w ×2 → tcvt_bf16 → tst)
 │   ├─ siMemcpy D2H(bo_D → host_D)
 │   ├─ siFree ×3
 │   ├─ golden(host_G, ...)  → 串行参考实现
 │   └─ 逐元素 |gold-dev|/|gold| > 1% ? → 无输出，err=0
 │
 └─ run_strided_case(33, 512) && pass
     ├─ input_tensor.strides = {2176, 1}   ← 手动覆盖
     ├─ rms_norm<bf16>(...)
     │   └─ rms_norm_launch<bf16>
     │       ├─ is_contiguous_layout(input) → false（strides[0]=2176 ≠ 512）
     │       ├─ bf16 分支：N=512 ∈ {512,1536} ✓，origN==N ✓，in.strides=(2176,1) ✓
     │       │              out.strides[0]=512==N ✓，w.strides[0]=1 ✓
     │       └─ rms_norm_bf16_strided_input_launch
     │           └─ rms_norm_bf16_strided_input_kernel<<<grid{16}, cluster{1}, block{2}, 0, nullptr>>>
     │               ├─ all_threads = 32，B = 33 → 线程 0 跑 bid=0 和 bid=32
     │               ├─ 每行 norm_tile_num = 512/512 = 1
     │               └─ 输入偏移 bid×2176×2 B，输出偏移 bid×512×2 B
     └─ golden_strided_input(...) + 比对 → err=0
```

---

### 47.13 一句话总结这份 kernel 的设计哲学

**用最少的同步换最大的数据局部性**：一个 PE 的两个硬件线程各拿 256 KB SRAM 缓存自己负责的整行，Pass 1 读 global 顺手写 shared，Pass 2 从 shared 读，全程**只有一条 `twait_store_share(0)` 同步指令、零 `__syncthreads`、零跨线程归约**；代价是 `norm_size` 必须 1024 B 对齐（无 tail 处理）、occupancy 固定 1 block/PE、非连续布局只支持白名单里的两个精确 shape。对 LLM 推理里 hidden size 固定且对齐的场景，这是一笔非常划算的交易。

---

## 48. CSR 指令详解：从 RISC-V 标量 CSR 到 Tile Core CSR

**ISA 文档**：`/softhome/like/asset/code/isa/index.html` → 章节「Control Register Operation / CSR指令」
**实证来源**：`sikernel/source/source_builtin/attention/rms_norm/build/SipuWork.rms_norm/librms_norm_sipu_fatbin.llvm.asm`

第 47 节讲到 grid/block 坐标要从 CSR 里读出来，但没有展开 CSR 本身。这一节补齐。

### 48.1 什么是 CSR

**CSR = Control and Status Register**（控制状态寄存器）。严格说 CSR 是"一类寄存器"，CSR 指令是访问它们的那几条指令。

它和普通 `ld`/`st` 的根本区别有三条：

| 维度 | 普通内存 | CSR |
| --- | --- | --- |
| 是否在地址空间里 | 是，有虚拟/物理地址 | **否**，独立编号空间 |
| 用什么访问 | `ld` / `st` / `lw` / `sw` … | **专用 CSR 指令**，不能用访存指令 |
| 读写副作用 | 无（就是读写存储） | **常有**：读清、置位、清位、只写、写触发脉冲 |

第三条是关键。比如本文档 48.5 节要讲到的 `tend`，手册明确写「该寄存器不可读，只会发出一个脉冲信号通知 CCS」——它不是一块存储，而是一个**控制动作**。`tmask` 则是「kernel 配置一次，全局影响所有用到 mask 的指令」，写它等于改硬件行为，不是存一个数。

### 48.2 这个平台上有**两套**独立的 CSR 空间

这是读下面汇编时最容易踩的坑。rms_norm 的 fatbin 里两种 CSR 指令**并排出现**，前后只差几条指令：

```asm
804000000310: 813024f3    csrr  s1, umisc_ctl      ← RV 标量核的 CSR
804000000314: 98f1        andi  s1, s1, -0x4
804000000316: 81349073    csrw  umisc_ctl, s1      ← 清位后写回（f16 kernel 用）
80400000031a: 68c042fb    tcsrr.r.b64 t0, tkernelidx  ← Tile Core 的 CSR，另一条通道
804000000326: 69004e7b    tcsrr.r.b64 t3, tkerneldims
```

| | RV 标量核 CSR | Tile Core CSR |
| --- | --- | --- |
| 访问指令 | `CSRRW` / `CSRRS` / `CSRRC` + `i` 变体 | `TCSRR` / `TCSRW` / `TCSRWI` |
| 指令编码 | 标准 RISC-V，opcode `0x73`（SYSTEM） | ACE 自定义 opcode `0x7B` |
| 编址 | 12 bit CSR 号 | 9 bit `csr_addr[10:2]`，4 B 步进 |
| 谁写 | 软件（kernel 自己读写） | **CCS / Block Dispatch 配置**，kernel 只读 |
| 典型例子 | `umisc_ctl`（浮点模式）、`mstatus`、`mhartid` | `tkernelidx`、`tkerneldims`、`thwid`、`tmask`、`tend` |

**两张表不通用**：RV 的 `csrr` 读不到 `tkernelidx`，Tile 的 `tcsrr` 也读不到 `umisc_ctl`。两套空间、两套指令、各自独立编号。

### 48.3 RISC-V 标准 CSR 指令（6 条）

opcode 固定 `0x73`，由 `funct3` 区分操作：

| `funct3` | 指令 | 语义 |
| --- | --- | --- |
| 001 | `CSRRW` | 读旧值到 rd，同时把 rs1 写入 CSR |
| 010 | `CSRRS` | 读旧值到 rd，把 rs1 的 **1 位置位**到 CSR（`rs1=x0` 即只读） |
| 011 | `CSRRC` | 读旧值到 rd，把 rs1 的 **1 位清位**到 CSR |
| 101 | `CSRRWI` | 同 CSRRW，写 5 bit 立即数 |
| 110 | `CSRRSI` | 同 CSRRS，写 5 bit 立即数 |
| 111 | `CSRRCI` | 同 CSRRC，写 5 bit 立即数 |

两条常用伪指令：

- `csrr rd, csr` ≡ `csrrs rd, csr, x0` —— 只读不写
- `csrw csr, rs1` ≡ `csrrw x0, csr, rs1` —— 只写不读

**实证解码 1**：`csrr s1, umisc_ctl`，机器码 `0x813024f3`

| 位域 | 值 | 含义 |
| --- | --- | --- |
| `[6:0]` | `0b1110011` = `0x73` | SYSTEM opcode |
| `[11:7]` | 9 | rd = `s1` |
| `[14:12]` | `0b010` = 2 | funct3 = CSRRS |
| `[19:15]` | 0 | rs1 = `x0` → **不写，只读** |
| `[31:20]` | `0x813` | CSR 号 = `umisc_ctl` |

**实证解码 2**：`csrw umisc_ctl, s1`，机器码 `0x81349073`

| 位域 | 值 | 含义 |
| --- | --- | --- |
| `[6:0]` | `0x73` | SYSTEM opcode |
| `[11:7]` | 0 | rd = `x0` → **丢弃旧值** |
| `[14:12]` | 1 | funct3 = CSRRW |
| `[19:15]` | 9 | rs1 = `s1`（写入的数据） |
| `[31:20]` | `0x813` | CSR 号 = `umisc_ctl` |

对照源码 `__clang_nds_device_functions.h` 里的 `__andescore_fp_mode`，语义完全吻合——读 `umisc_ctl`、清掉 bit[1:0]、按需置 bf16 位、再写回：

```c
unsigned long long umisc_ctl = __nds__read_csr(NDS_UMISC_CTL) & ~0x3;
if (mode == BF16) umisc_ctl |= 1;
__nds__write_csr(umisc_ctl, NDS_UMISC_CTL);
```

汇编里的三行 `csrr` / `andi` / `csrw` 就是这个函数被内联的结果（**注意：RV 的 CSR 是软件自己读写的，和下面 Tile CSR 的"kernel 只读"完全不同**）。

### 48.4 Tile Core CSR 指令（3 条）

ISA 手册「Control Register Operation」一节给出的原型：

| 指令 | 语法 | 作用 |
| --- | --- | --- |
| `TCSRR` | `tcsrr.r.b32/b64 rd, csr_addr` | 把 CSR 原有值读到 rd |
| `TCSRW` | `tcsrw.r.b32/b64 rs1, csr_addr` | 把 rs1 的值写入 CSR |
| `TCSRWI` | `tcsrw.i.b32/b64 imm1, csr_addr` | 写 5 bit 立即数，未用到的其余 27 bit 补 0 |

三者共用同一套编码骨架（手册给出的 bitfield）：

| 位域 | TCSRR | TCSRW | TCSRWI |
| --- | --- | --- | --- |
| `[6:0]` `ACE_op` | `1111011` | `1111011` | `1111011` |
| `[11:7]` | `rd` | `00000` | `00000` |
| `[14:12]` `tuop` | `100` | `100` | `100` |
| `[19:15]` | `00000` | `rs1` | `imm_1` |
| `[28:20]` | `csr_addr[10:2]` | `csr_addr[10:2]` | `csr_addr[10:2]` |
| `[29]` `b64` | 0/1 | 0/1 | 0/1 |
| `[31:30]` `rw` | `01` | `10` | `00` |

**实证解码 3**：`tcsrr.r.b64 t0, tkernelidx`，机器码 `0x68c042fb`

| 位域 | 值 | 含义 |
| --- | --- | --- |
| `[6:0]` | `0b1111011` = `0x7B` | ACE 自定义 opcode |
| `[11:7]` | 5 | rd = `t0` |
| `[14:12]` | `0b100` = 4 | tuop = TCSRR |
| `[19:15]` | 0 | 保留 |
| `[28:20]` | `0b010001100` = `0x8C` | `csr_addr[10:2]` → **0x8C << 2 = 0x230 = `tkernelidx`** ✓ |
| `[29]` | 1 | `.b64` 模式 |
| `[31:30]` | `0b01` | rw = 读 |

三条语义细节：

- **`b64` 位**：0 = 按 32 bit 访问，1 = 按 64 bit 访问。手册明确：32 bit 模式下 0x4、0x8 地址都能访问；**64 bit 模式下地址必须 64 bit 对齐，访问 0x4 属于非法**。
- **吞吐**：读 CSR 是 cycle 0 发射命令、cycle 1 写回，throughput 0.5（两个 cycle 发一条读）；写 CSR 是 throughput 1（一个 cycle 一条）。
- **编址宽度**：`csr_addr[10:2]` 只有 9 bit，所以整个 Tile CSR 空间是 **0x000 ~ 0x3FC、按 4 B 步进**。这和手册正文那句「0x0-0x3fc 的 CSR 寄存器都要通过以下通路进行交互」互相印证。

手册还给出了完整的配置通路（原文）：

> 0x0-0x3fc 的 CSR 寄存器都要都要通过以下通路进行交互。**CCS 可通过 Block Dispatch 模块配置到 Tile CSR，之后向 RV Core 发起中断启动 kernel。** 同理，在 thread 完成后，**RV Core 写 tend 寄存器**，CSR 可发起结束信号发到 Block Dispatch，BD 返回 thread-done 信号给 CCS。

也就是说：**Tile CSR 的正常写入者是硬件（CCS/BD），不是 kernel 软件**。这直接决定了下面 48.6 的结论。

### 48.5 CSR 列表

手册「CSR列表」一节给出的完整清单：

| CSR_name | CSR_addr | 属性 | 说明 |
| --- | --- | --- | --- |
| `tmask` | 0x0 | RW | talu 使用的配置信息，由 kernel 配置，可全局配置所有用到 tmask bit 的指令的 mask |
| `tctrl` | 0x8 | RW | ALU 和 MMA 的 rounding / saturate / deform / MX OCP 信息，由 kernel 配置 |
| `tcmdvalid` | 0x100 | RO | 表示目前有可执行的 kernel。BD 发任务 cmdValid 为 1，`tend` 写值后 cmdValid 改为 0 |
| `tend` | 0x108 | **WO** | thread 完成任务需要写该寄存器，写非 0 可通知 CCS 该 thread 已经完成。**该寄存器不可读，只会发出一个脉冲信号** |
| `thwid` | 0x180 | RO | 2 bit core_id / 2 bit pe_id / 3 bit cluster_id / 9 bit chip_id |
| `tsynccnt` | 0x190 | RO | core_idle / mem_idle / smem_ld_cnt / smem_st_cnt / gmem_ld_cnt / cmt_group |
| `tsmembase` | 0x200 | RW | kernel 在 share memory 使用的起始地址，由 CCS 配置，256 B 对齐，**kernel 软件无需使用** |
| `tsmemsize` | 0x208 | RW | kernel 在 share memory 使用的地址空间范围，由 CCS 配置，256 B 对齐 |
| `tkerneladdr` | 0x210 | RO | kernel 启动 addr，由 CCS 配置 |
| `tparamaddr` | 0x220 | RO | kernel 启动需要的参数地址，由 CCS 配置，8 B 对齐 |
| `tparamsize` | 0x228 | RO | kernel 启动需要的参数 size，由 CCS 配置，8 B 对齐 |
| `tkernelidx` | 0x230 | RO | 含 tb/tbc 的 x/y/z 方向和 thread index，由 CCS 配置 |
| `tkerneldims` | 0x240 | RO | 含 tb/tbc 的 x/y/z 方向和 thread dimension，由 CCS 配置 |
| `tregbase` | 0x250 | RW | kernel 在 tile reg 使用的寄存器 offset，由 CCS 配置，256 B 对齐，**仅用于 debug** |
| `tregsize` | 0x254 | RW | kernel 在 tile reg 使用的寄存器数量，由 CCS 配置，**仅用于 debug** |
| `tprivatebase` | 0x260 | RO | kernel 在 global memory 上使用的 thread private 地址空间起始地址，256 B 对齐，**由 BD 配置** |
| `tprivatesize` | 0x268 | RO | kernel 在 global memory 上使用的 thread private 地址空间总容量，256 B 对齐，**由 BD 配置** |
| `ttbmap` | 0x270 | RO | 数据流模式下 tb 和 pe 的匹配表，由 CCS 配置 |

两个值得注意的属性组合：

- **`tkernelidx` / `tkerneldims` 都是 RO** —— 线程**无法伪造自己的身份**。这正是 `rms_norm_launch` 里敢用 `bid = tid; bid += all_threads` 做 grid-stride 循环、并假设各线程拿到的 `tid` 唯一且稳定的底气。
- **`tend` 是 WO 且不可读** —— 它不是存储，是一个"我干完了"的脉冲通知。所以读它没有意义，手册才特别注明「不可读」。

对照第 47 节：`rms_norm_kernel_bf16.hpp` 里没有出现过任何 `tcsrw`，线程结束时也不需要手动写 `tend`——`__global__` 函数的 epilogue 由编译器/运行时处理。

### 48.6 rms_norm 里为什么非读 CSR 不可

在 CUDA 里写 `blockIdx.x * blockDim.x + threadIdx.x` 看起来"免费"，是因为 SM 有专门的只读寄存器组。**这里没有等价物**：block/thread 的坐标是 CCS（Block Dispatch）在启动每个线程前，**逐线程**写进 `tkernelidx` / `tkerneldims` 这两个 CSR 的。所以编译器只能生成"读 CSR + 手工位运算"的代码。

以 f32 kernel 的 prologue 为例（`0x4e` ~ `0x84`，已逐条核对位运算）：

```asm
80400000004e: 68c042fb       tcsrr.r.b64 t0, tkernelidx     # t0 = 我的坐标
804000000052: 01029493       slli  s1, t0, 0x10
804000000056: 0304d313       srli  t1, s1, 0x30            # t1 = t0[47:32] = tbc_idx_x
80400000005a: 69004e7b       tcsrr.r.b64 t3, tkerneldims    # t3 = 网格形状
80400000005e: 030e1493       slli  s1, t3, 0x30
804000000062: 0344d393       srli  t2, s1, 0x34            # t2 = t3[15:4]  = tbc_dim_x
804000000066: 02638333       mul   t1, t2, t1             # t1 = tbc_dim_x * tbc_idx_x
80400000006a: 03029493       slli  s1, t0, 0x30
80400000006e: 90d1           srli  s1, s1, 0x34            # s1 = t0[15:4]  = tb_idx_x
804000000070: 9326           add   t1, t1, s1             # t1 = blockIdx.x
804000000072: 00fe7e13       andi  t3, t3, 0xf             # t3 = t3[3:0]   = tb_dim = blockDim.x
804000000076: 03c30333       mul   t1, t1, t3             # t1 *= blockDim.x
80400000007a: 00f2f493       andi  s1, t0, 0xf             # s1 = t0[3:0]   = thread_idx
80400000007e: 9326           add   t1, t1, s1
804000000080: 080302bb       zext.w t0, t1                 # tid = blockIdx.x*blockDim.x + threadIdx.x
```

紧随其后再读一次 `tkerneldims` 求 `all_threads`（`0x94` ~ `0xa4`）：

```asm
804000000094: 027e03b3       mul   t2, t3, t2             # blockDim.x * tbc_dim_x
804000000098: 690044fb       tcsrr.r.b64 s1, tkerneldims
80400000009c: 04c2           slli  s1, s1, 0x10
80400000009e: 90c1           srli  s1, s1, 0x30            # s1 = tkerneldims[47:32] = grid_dim_x
8040000000a0: 029383b3       mul   t2, t2, s1             # all_threads = grid_dim_x*tbc_dim_x*blockDim.x
```

两个 CSR 的位域定义（手册「CSR列表」，双方都是 64 bit）：

| bit | `tkernelidx` (0x230) | `tkerneldims` (0x240) |
| --- | --- | --- |
| `[3:0]` | `thread_idx` | `tb_dim` |
| `[15:4]` | `tb_idx_x` | `tbc_dim_x` |
| `[23:16]` | `tb_idx_y` | `tbc_dim_y` |
| `[31:24]` | `tb_idx_z` | `tbc_dim_z` |
| `[47:32]` | `tbc_idx_x` | `grid_dim_x` |
| `[55:48]` | `tbc_idx_y` | `grid_dim_y` |
| `[63:56]` | `tbc_idx_z` | `grid_dim_z` |

于是 kernel 里的三个符号量全部落到具体位域上：

| kernel 源码 | 由哪些位域合成 |
| --- | --- |
| `threadIdx.x` | `tkernelidx[3:0]` |
| `blockIdx.x` | `tkerneldims[15:4] * tkernelidx[47:32] + tkernelidx[15:4]` |
| `blockDim.x` | `tkerneldims[3:0]` |
| `gridDim.x` | `tkerneldims[47:32] * tkerneldims[15:4]`（两级 grid 展开） |

注意 grid 是**两级**的：`tbc`（thread block cluster）和 `tb`。`blockIdx.x` 是两级索引线性化后的结果（`tbc_dim_x * tbc_idx_x + tb_idx_x`），而 `gridDim.x = grid_dim_x * tbc_dim_x`。第 47.6 节里 `rms_norm_contiguous_launch` 算出的 `grid_dim = 16`、`block_dim = 2`，最终对应到硬件就是这四组字段的具体取值。

需要 `thwid`（0x180）才能区分 chip / cluster / PE 的算子——比如 `semaphore.h`、`cooperative_groups.h`、`util_device.h` 里的 `get_thwid_pe_id()` / `get_thwid_cluster_id()`——**不是** rms_norm 这种"线程-行一对一、零跨线程通信"的 kernel。

### 48.7 rms_norm fatbin 的 CSR 使用统计

对 `librms_norm_si_fatbin.llvm.asm` 全文做指令统计：

| 指令 | 出现次数 | 说明 |
| --- | --- | --- |
| `tcsrr.r.b64` | **52** | 其中读 `tkernelidx` 30 次、读 `tkerneldims` 22 次 |
| `tcsrw` / `tcsrwi` | **0** | kernel 不写任何 Tile CSR |
| `csrr` / `csrw`（RV 侧） | 6 / 6 | 全部是 `umisc_ctl`，即 3 处 `__andescore_fp_mode` 内联（f16、bf16、bf16_strided 各一次；f32 kernel 没有） |

三点观察：

1. **52 次 `tcsrr` 是"每个 kernel 每个线程重算一遍"的结果**——4 个 kernel，每个都有独立的 prologue，且 `tid` 与 `all_threads` 各要读一次 CSR（`grid-stride` 循环前算一次即可）。
2. **`tcsrw` = 0**，印证 48.5 的结论：Tile CSR 由硬件配置，kernel 只读。第 47 节的 kernel 里没有线程退出通知逻辑，也不需要。
3. **RV 侧那 6 条 `csrr`/`csrw` 全给了 `umisc_ctl`**，且都是"读—清 bit[1:0]—（按需置 1）—写回"的形态。f32 kernel 里没有，因为它不需要做 bf16 转换。这恰好和 47.7 节讲的 `__andescore_fp_mode(BF16)` / `(FP16)` 对上：**它改的是 RV 标量核的浮点模式 CSR，不是 Tile 的**。

### 48.8 一句话小结

**CSR 是"不在地址空间、要用专用指令访问、读写往往带副作用"的寄存器；这个平台上有两套互不相通的空间——RV 标量核的 `csrr/csrw`（软件自己读写，如 `umisc_ctl`）和 Tile Core 的 `tcsrr/tcsrw`（硬件配置、kernel 只读，如 `tkernelidx`/`tkerneldims`）。** rms_norm 之所以非读 CSR 不可，是因为 block/thread 坐标不像 CUDA 那样有专用寄存器，而是由 CCS 逐线程写进 `tkernelidx`/`tkerneldims`，kernel 只能用 `tcsrr.r.b64` 读回来再做位域拆解——这也解释了为什么每个 kernel 的 prologue 里都有一段"移位 + 乘法 + 加法"的样板代码。

---

## 49. `tbfloat16m1_t` 是什么、在哪里定义；以及 32 位浮点有没有 tile 类型

**SDK**：`/share_data/sicx_sdk/release/latest/`
**编译器**：`/share/users/like/package/compiler-toolchain/`
**ISA 文档**：`/softhome/like/asset/code/isa/index.html`
**源码出处**：`source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_bf16.hpp`

第 47 节逐行讲了 bf16 kernel，但把 `tbfloat16m1_t` 当成"天生就有的类型"用了过去。这一节回答两个问题：它到底在哪定义、`m1` 意味着什么；以及 32 位浮点在这个平台上有没有对应的 tile format。

### 49.0 一句话答案

- `tbfloat16m1_t` **不是** SDK 里的普通 `typedef`，而是 **clang 编译器内建的 tile 寄存器类型**（builtin type）。`tile_vector.h:123` 那行 `typedef __tile_bfloat16m1_t tbfloat16m1_t;` 只是把一个双下划线内建名**换个名字**，真正的定义在编译器的 `RISCVTileTypes.def:220`，是编译器前端认识的一等类型。
- 32 位浮点**有两个** tile 类型：`tfloat32m1_t`（真 f32/单精度）和 `ttfloat32m1_t`（TF32，19 位有效位）。**rms_norm 的 f32 kernel 用的是前者**（`tfloat32m1_t`），不是 TF32。

### 49.1 `tbfloat16m1_t` 的完整定义链

名字的构成是 `t` + `bfloat16` + `m1` + `_t`：

| 片段 | 含义 |
| --- | --- |
| `t` | **T**ile register（区别于 RVV 向量寄存器的 `v` 前缀） |
| `bfloat16` | 元素类型是 bfloat16 |
| `m1` | **LMUL = 1**，占 1 个 tile register |
| `_t` | C 类型命名惯例（`_t` 后缀） |

完整的定义链，从源码到编译器：

```
rms_norm_kernel_bf16.hpp:31        tbfloat16m1_t bf16_din;
        │
        ├─ 包含 <siorigin_tile.h>            ← 这里的 "siorigin_" 是占位符
        │       │
        │       ├─ #include <tile_vector.h>        ← 名字映射层
        │       │       tile_vector.h:123
        │       │       typedef __tile_bfloat16m1_t tbfloat16m1_t;
        │       │               │
        │       │               └─→ __tile_bfloat16m1_t 是 clang builtin type
        │       │                   编译器里的定义：
        │       │                   RISCVTileTypes.def:220
        │       │                   TILE_VECTOR_TYPE_FLOAT("__tile_bfloat16m1_t",
        │       │                       TileBFloat16m1, TileBFloat16m1Ty, 512, 16, true, false)
        │       │
        │       └─ #include "siorigin_tile_150g.h"  ← 指令原型层（arch=150 时）
        │               tbfloat16m1_t tld_linear_global_m1(const sifmt::bfloat16*, ...);
        │               ...（几千条 TILE_HEADER 声明）
        │
        └─ 这一层还依赖 #include <riscv_vector.h>（RVV 类型）和 "SiTe.hpp"（标量 sifmt 类型）
```

**关键点**：`tile_vector.h` 只是**别名层**。逐行看它做了什么：

```c
// tile_vector.h:120-128（节选）
typedef __tile_bfloat16mf8_t tbfloat16mf8_t;
typedef __tile_bfloat16mf4_t tbfloat16mf4_t;
typedef __tile_bfloat16mf2_t tbfloat16mf2_t;
typedef __tile_bfloat16m1_t  tbfloat16m1_t;    // ← 第 123 行
typedef __tile_bfloat16m2_t  tbfloat16m2_t;
...
```

左边是给用户写的短名，右边是 clang 认得的真名。**没有任何 `struct`/`union` 定义**——因为这些类型由编译器直接实现，不是库文件里的数据结构。这也解释了为什么写 `tbfloat16m1_t x;` 不需要链接任何东西。

### 49.2 编译器里的权威定义：`RISCVTileTypes.def`

打开 `/share/users/like/package/compiler-toolchain/llvm-project/clang/include/clang/Basic/RISCVTileTypes.def`，第 220 行：

```c
TILE_VECTOR_TYPE_FLOAT("__tile_bfloat16m1_t", TileBFloat16m1, TileBFloat16m1Ty, 512, 16, true, false)
```

对照文件开头宏的说明（`RISCVTileTypes.def:19-34`），各字段是：

| 位置 | 值 | 名称 | 含义 |
| --- | --- | --- | --- |
| 1 | `"__tile_bfloat16m1_t"` | Name | builtin type 的名字 |
| 2 | `TileBFloat16m1` | Id | 类型枚举 |
| 3 | `TileBFloat16m1Ty` | SingletonId | 全局单例 |
| 4 | **`512`** | **NumEls** | **元素个数** |
| 5 | **`16`** | **ElBits** | **每个元素 16 bit（SEW）** |
| 6 | `true` | IsBF | 是 bfloat16 |
| 7 | `false` | IsTF | 不是 tfloat32 |

**`NumEls = 512` 是理解 `m1` 的钥匙**：512 个元素 × 16 bit = 8192 bit = **1024 字节**。这正好是第 47 节讲的「一个 tile register 是 8192 bit / 1024 B」。所以 `m1` 的定义就是"一个满载的 tile register"。

宏展开后 `TILE_VECTOR_TYPE_FLOAT` 还有几个固定参数（`RISCVTileTypes.def:72-77`）：

```c
#define TILE_VECTOR_TYPE_FLOAT(Name, Id, SingletonId, NumEls, ElBits, IsBF, IsTF) \
  TILE_VECTOR_TYPE(Name, Id, SingletonId, NumEls, ElBits, 1, false, true,          \
                   IsBF, IsTF, false, false, false)
//                                                  ↑          ↑     ↑
//                                              IsSigned=false, IsFP=true, IsMX=false
```

所以 bf16 tile 的完整属性是：512 元素 / 16 bit / NF=1 / 有符号 / 浮点 / 是 BF16 / 非 TF32 / 非 MX。

### 49.3 同一族的所有类型：元素数表

把 `RISCVTileTypes.def:207-235` 里 32 位浮点和 16 位浮点全列出来，可以清楚看到 `NumEls` 的规律：

**bfloat16（`RISCVTileTypes.def:217-225`）—— 每元素 16 bit，`m1` = 512 个**

| 类型 | NumEls | ElBits | 字节数 |
| --- | --- | --- | --- |
| `__tile_bfloat16mf8_t` | 64 | 16 | 128 |
| `__tile_bfloat16mf4_t` | 128 | 16 | 256 |
| `__tile_bfloat16mf2_t` | 256 | 16 | 512 |
| **`__tile_bfloat16m1_t`** | **512** | **16** | **1024** |
| `__tile_bfloat16m2_t` | 1024 | 16 | 2048 |
| `__tile_bfloat16m4_t` | 2048 | 16 | 4096 |
| `__tile_bfloat16m8_t` | 4096 | 16 | 8192 |
| `__tile_bfloat16m16_t` | 8192 | 16 | 16384 |
| `__tile_bfloat16m32_t` | 16384 | 16 | 32768 |

规律：**`mf8` 的字节数 = 128 B，之后每级翻倍**。`mf8/mf4/mf2/m1/m2/m4/m8` 对应 128/256/512/1024/2048/4096/8192 字节。这也和第 47 节说的「tile 内部最小操作粒度 128 B」对上了——`mf8` 就是一个 128 B 的分数寄存器。

注意 `m16`/`m32` 的字节数（16 KB/32 KB）**超过了一个 tile register 的 1 KB 容量**，它们表示的是"多寄存器组"（占 16/32 个 tile reg），用于 MMA 的矩阵操作数。

### 49.4 第 47 节的 kernel 实际用了哪些

`rms_norm_kernel_bf16.hpp` 的类型使用统计：

| 类型 | 出现次数 | 用途 |
| --- | --- | --- |
| `tbfloat16m1_t` | 6 | `bf16_din` / `bf16_wgt` / `bf16_out`（输入、权重、输出） |
| `tfloat32m1_t` | 2 | `tile_rsqrt`（**注意：是 f32 不是 bf16**） |

其中 `bf16_din` 的诞生过程（`rms_norm_kernel_bf16.hpp:31`、`:49`）：

```c
tbfloat16m1_t bf16_din;                                  // 声明
bf16_din = tld_linear_global_m1(pin, offset);            // 从 global 加载，返回 tbfloat16m1_t
```

`tld_linear_global_m1` 的返回类型由**指针的标量类型**决定。编译器里同一个名字有多个重载（`__tile_inter.h:2330` 起）：

```c
tbfloat16m1_t tld_linear_global_m1(const sifmt::bfloat16 * baseAddr, ...);  // :2330
tfloat32m1_t  tld_linear_global_m1(const sifmt::float32  * baseAddr, ...);  // :2334
ttfloat32m1_t tld_linear_global_m1(const sifmt::tfloat32 * baseAddr, ...);  // :2346
```

这里是**引用 49.5 的关键**：三个重载分别把 `sifmt::bfloat16*` / `sifmt::float32*` / `sifmt::tfloat32*` 映射到 `tbfloat16m1_t` / `tfloat32m1_t` / `ttfloat32m1_t`。

底层的 builtin 层面完全一致——`RISCVTileTypes.def` 里 `__tile_float32m1_t` 和 `__tile_tfloat32m1_t` 的参数**逐位相同**：

```c
TILE_VECTOR_TYPE_FLOAT("__tile_float32m1_t",  TileFloat32m1,  TileFloat32m1Ty,  256, 32, false, false)  // :210
TILE_VECTOR_TYPE_FLOAT("__tile_tfloat32m1_t", TileTFloat32m1, TileTFloat32m1Ty, 256, 32, false, true)   // :230
//                                                                                        ↑     ↑
//                                                                                    IsBF=false, IsTF
```

区别**只有最后一个 `IsTF` 标志位**：`false` 是普通 f32，`true` 是 TF32。256 元素 × 32 bit = 8192 bit = 1 KB，两者一样大。

### 49.5 32 位浮点的两个 tile 类型：`tfloat32m1_t` vs `ttfloat32m1_t`

**问题：`sipu` 有没有定义 tfloat32 类型的 tile format？**
**答：有，而且有两个，语义完全不同。**

| | `tfloat32m1_t` | `ttfloat32m1_t` |
| --- | --- | --- |
| 名字前缀 | `t` + `float32` | `t` + `tfloat32`（双 t） |
| 编译器内建名 | `__tile_float32m1_t` | `__tile_tfloat32m1_t` |
| `IsTF` 标志 | `false` | **`true`** |
| 元素格式 | **IEEE 单精度 f32**（8 指数 / 23 尾数） | **TF32**（8 指数 / 10 尾数） |
| `NumEls` × `ElBits` | 256 × 32 = 8192 bit（1 KB） | 256 × 32 = 8192 bit（1 KB） |
| 存储宽度 | 32 bit | 32 bit（**值只用 19 bit**） |
| 支持的操作 | 全部（load/store/ALU/MMA） | **只有 load/store/MMA/GEMV** |
| 典型用途 | rms_norm f32 kernel | 矩阵乘的 A/B 操作数 |

**TF32 的位布局**（来自 SDK 的量化实现，`sifmt/quantize/define.hpp:73-77`）：

```c
constexpr uint8_t  EXP_BITS_TF32  = 8;
constexpr uint8_t  MANT_BITS_TF32 = 10;
constexpr uint32_t MASK_TF32_UI   = 0x7FFFF;      // 19 位
```

从读写函数可以反推出位序（`define.hpp:657-673`）：

```c
signTF32UI(a)  →  a >> 18                     // bit 18        = 符号
expTF32UI(a)   →  (a >> 10) & 0xFF            // bit 17:10     = 指数(8)
fracTF32UI(a)  →  a & 0x3FF                   // bit  9:0      = 尾数(10)
packToTF32UI(sign, exp, sig)
               →  (sign << 18) + (exp << 10) + sig
```

即 **TF32 是"右对齐的 19 位"**：`[18]` 符号、`[17:10]` 指数、`[9:0]` 尾数，存在一个 32 bit 容器里，高 13 位是 0。这和 NVIDIA 在 tensor core 里用的 TF32 **位序不同**（那边是左对齐、砍掉 f32 的低 13 位尾数），能不能互相直接搬数据要小心——**这个平台是右对齐 19 位，两边不能 memcpy 混用**。

`tf32_to_f32` 的实现也印证了这一点（`nonmx.hpp:697`）：

```c
uiA = a.v & 0x7FFFF;      // ← 只取低 19 位，高 13 位直接丢弃
```

**另一个关键差异：TF32 tile 没有逐元素算术。** 我把 arch-150 头文件里所有提到 `ttfloat32` 的 intrinsic 按前缀分类，结果只有四类：

| 前缀 | 数量 | 说明 |
| --- | --- | --- |
| `tld_*` | 若干 | 加载 |
| `tst_*` | 若干 | 存储 |
| `tmma_*` | 大量 | 矩阵乘（如 `tmma_ttt_f32_tf32_tf32_r8_m4`） |
| `tmva_*` | 若干 | 矩阵-向量乘（GEMV） |

`tmul` / `tadd` / `tsub` / `tcvt` / `tmv` / `tmax` / `tmin` / `trelu` **对 `ttfloat32m1_t` 一个都没有**。而 `tfloat32m1_t`（真 f32）全都有，例如：

```c
TILE_HEADER(tmul_ttt_f32) tfloat32m1_t tmul(tfloat32m1_t src1, tfloat32m1_t src2, uint32_t attr = 0);
TILE_HEADER(tadd_ttt_f32) tfloat32m1_t tadd(tfloat32m1_t src1, tfloat32m1_t src2, uint32_t attr = 0);
```

**为什么会这样**：TF32 在这个平台上是**为 MMA 的 A/B 操作数准备的存储格式**，不是通用的计算类型。乘法器直接吃 TF32 输入、产出 f32 累加（ISA 文档「运算模式」表里 `Index 0` 那行：`.tf32 .tf32 → .f32`，Compute Mode = Normal P0）。要做 TF32 的逐元素乘加，得先转成 f32。

**回到 rms_norm**：`rms_norm_kernel_f32.hpp:30-38` 声明的 `tfloat32m1_t f32_din / f32_wgt / f32_sq / f32_sq_sum / tile_rsqrt / f32_out` 全部是**真 f32**。这也符合它的算法需求——RMSNorm 要做 `x*x`、求和、`rsqrt`、逐元素缩放，这些在 TF32 上根本不存在。所以：

> **rms_norm 的 f32 kernel 走的是完整单精度路径，完全没有用到 TF32。**

### 49.6 名字容易混淆的三个层次

同一个"f32"在这个平台上出现在三个不同层次，写代码时容易串：

| 层次 | 类型名 | 定义位置 | 用途 |
| --- | --- | --- | --- |
| **标量**（C++ 类） | `sifmt::float32` / `sifmt::bfloat16` / `sifmt::tfloat32` | `SiTe/sifmt/sifmt_fp.hpp:277-280` | 指针类型、host 端数据 |
| | `float` / `bfloat16_t` / `tfloat32_t` | clang builtin（`TokenKinds.def:708`） | 编译器关键字类型 |
| **RVV 向量** | `vfloat32m1_t` | `riscv_vector.h` | RVV 标量核向量寄存器 |
| **Tile** | `tfloat32m1_t` / `tbfloat16m1_t` / `ttfloat32m1_t` | `tile_vector.h` + `RISCVTileTypes.def` | tile 寄存器 |

三者的对应关系靠**函数重载**串起来：传 `sifmt::float32*` 进 `tld_linear_global_m1`，出来的就是 `tfloat32m1_t`。`sifmt::float32` 本身的定义（`sifmt_fp.hpp:279`）：

```c
using float32 = SiFpBase<8, 23, uint32_t, sifmt::f32::toFloat, sifmt::f32::fromFloat<uint32_t>>;
//               ↑  ↑   ↑
//            指数8 尾数23 存储 uint32_t
```

而 `tfloat32`（`:280`）的宿主类型**也是 `uint32_t`**，只是指数尾数变成 `8, 10`：

```c
using tfloat32 = SiFpBase<8, 10, uint32_t, sifmt::tf32::toFloat, sifmt::tf32::fromFloat<uint32_t>>;
```

两者在 C++ 层面**字节数相同**（都是 4 字节），但语义不同，`void*` 互转是编译得过、结果错的经典陷阱。

### 49.7 小结

- `tbfloat16m1_t` 是 **clang builtin type** 的别名，不是库里的结构体。定义在编译器的 `RISCVTileTypes.def:220`，SDK 侧 `tile_vector.h:123` 只做 `__tile_bfloat16m1_t` → `tbfloat16m1_t` 的名字映射。
- 名字里的 `m1` 表示 LMUL=1，对应 **512 个 bf16 元素 = 1024 字节 = 一个 tile register**。`mf8`(128 B) 到 `m32` 的变化规律是字节数逐级翻倍。
- **32 位浮点有两个 tile 类型**：`tfloat32m1_t`（IEEE f32，`IsTF=false`）和 `ttfloat32m1_t`（TF32，`IsTF=true`）。两者都是 256 元素 × 32 bit = 1 KB。
- TF32 是**右对齐 19 位**（符号 1 / 指数 8 / 尾数 10），值不占满 32 bit。**只支持 load/store/MMA/GEMV，没有逐元素 ALU**——它是给矩阵乘准备的输入格式，不是通用计算类型。
- rms_norm 的 f32 kernel 用的是 `tfloat32m1_t`（真 f32），不是 TF32。对 RMSNorm 这种需要逐元素乘/加/开方的算法，TF32 在指令层面就不可用。

---

## 50. `tst_linear_share_m1` 的完整展开：从 kernel 源码到 `tst.trir.linear.u32.share` 机器码

**问题**：`rms_norm_kernel_bf16.hpp` 里的 `tst_linear_share_m1`，定义是在 `siorigin_tile_150g.h` 吗？`TILE_HEADER` 宏展开成什么样？包裹的 ISA 指令在哪里定义？

**SDK**：`/share_data/sicx_sdk/release/latest/`（→ 软链到 `2609101917`）
**编译器**：`/share/users/like/package/compiler-toolchain/`
**ISA 文档**：`/softhome/like/asset/code/isa/index.html`

### 50.0 一句话答案

| 问题 | 答案 |
| --- | --- |
| 定义在 `siorigin_tile_150g.h` 吗？ | **是**，`kernel/rms_norm_kernel_bf16.hpp` 走的是这个头（`siorigin_tile_150g.h:15532`）。同一个名字在 `siorigin_tile_inter.h:11942` 也有重载，但那个需要 `sifmt::bfloat16*`，kernel 传的是内建 `bfloat16_t*`，**不匹配** |
| `TILE_HEADER` 展开成什么？ | `static inline __device__ __attribute__((__always_inline__, __nodebug__)) __attribute__((clang_builtin_alias(__builtin_rvv_tst_trr_linear_share_m1)))` |
| ISA 指令在哪定义？ | **TableGen**：`compiler-toolchain/llvm-project/llvm/lib/Target/RISCV/RISCVInstrInfoXSOTile150GMemory.td:198`（`TST_TRIR_LINEAR_U32`）；ISA 手册对应章节是「Tile Linear Load/Store」 |
| 最终机器码 | `tst.trir.linear.u32.share T4, (s1), 0x0, s10` = `0x8204907b 91a0607b` |

整条链有**五层**，下面逐层拆。

### 50.1 第 0 层：调用点

`source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_bf16.hpp:56`

```c
if(i<shared_buffer_size/512){
  tst_linear_share_m1(bf16_din, shared_buff[threadIdx.x], i*tile_size);
}
```

三个实参：

| 实参 | 类型 | 含义 |
| --- | --- | --- |
| `bf16_din` | `tbfloat16m1_t`（`rms_norm_kernel_bf16.hpp:31`） | 源 tile 寄存器 |
| `shared_buff[threadIdx.x]` | `bfloat16_t*`（`rms_norm_kernel_bf16.hpp:45`） | 目标 share memory 基址 |
| `i*tile_size` | `int` | 字节偏移（`tile_size = 1024`） |

注意第二个实参的静态类型是**内建** `bfloat16_t`（来自 `siorigin_tile_150g.h`，因为 `shared_buff` 声明为 `__shared__ bfloat16_t[...]`），**不是** `sifmt::bfloat16`。这一点决定了后面选哪个重载。

`rms_norm_kernel_bf16.hpp:19-23` 的 include 顺序：

```c
#include <riscv_vector.h>
#include <stdint.h>
#include <siorigin_tile.h>     // ← 引入 tile_150g.h / tile_inter.h 的基础
#include "sipu.h"
#include "SiTe.hpp"                   // ← 引入 sifmt 类型 + tile_inter.h
```

### 50.2 第 1 层：`siorigin_tile.h` 的头文件分发

`/share_data/sicx_sdk/release/latest/bin/nds64le-elf-newlib-v5d/lib/clang/20/include/siorigin_tile.h`（全文件仅 30 行）：

```c
  7  #ifndef __SIORIGIN_TILE_H__          // （占位符还原后为对应前缀）
  8  #define __SIORIGIN_TILE_H__
  9
 10  #include <stdint.h>
 11  #include <riscv_vector.h>
 12
 13  #include <tile_vector.h>              // ① 类型别名层
 14
 15  #define __rvv_generic \
 16      static inline __device__ __attribute__((__always_inline__, __nodebug__))
 17  #if defined(__SIPU_ARCH__) && (__SIPU_ARCH__ >= 100)
 18  #define TILE_HEADER(T) __rvv_generic __attribute__((clang_builtin_alias(__builtin_rvv_##T)))
 19  #if defined(__riscv_xsotile150g)
 20  #include "siorigin_tile_150g.h"      // ② 指令原型层（arch=150）
 21  #endif
 22  #if defined(__riscv_xsotile160g)
 23  #include "siorigin_tile_160g.h"
 24  #endif
 25  #if defined(__riscv_xsotile170g)
 26  #include "siorigin_tile_170g.h"
 27  #endif
 28  #endif
```

（行号按该文件实际内容编号。）

`sikernel` 构建时 `-arch=sipu_150`（见第 47.1 节），编译器据此定义 `__riscv_xsotile150g`，所以走 `:20` 那条分支。

**`siorigin_tile.h:18` 是 `TILE_HEADER` 的定义处**：

```c
#define TILE_HEADER(T) __rvv_generic __attribute__((clang_builtin_alias(__builtin_rvv_##T)))
```

展开规则是 `##` 拼接——`TILE_HEADER(tst_trr_linear_share_m1)` 会拼出 `__builtin_rvv_tst_trr_linear_share_m1`。

### 50.3 第 2 层：`TILE_HEADER` 的两次展开

以 `siorigin_tile_150g.h:15532` 那行为例：

```c
TILE_HEADER(tst_trr_linear_share_m1) void tst_linear_share_m1(tbfloat16m1_t src, bfloat16_t * baseAddr, unsigned long offset, uint32_t attr = 0);
```

**第一次展开**——`TILE_HEADER(T)` → `__rvv_generic __attribute__((clang_builtin_alias(__builtin_rvv_##T)))`：

```c
__rvv_generic __attribute__((clang_builtin_alias(__builtin_rvv_tst_trr_linear_share_m1)))
void tst_linear_share_m1(tbfloat16m1_t src, bfloat16_t * baseAddr, unsigned long offset, uint32_t attr = 0);
```

**第二次展开**——`__rvv_generic` → 它的定义（`siorigin_tile.h:15-16`）：

```c
static inline __device__ __attribute__((__always_inline__, __nodebug__))
__attribute__((clang_builtin_alias(__builtin_rvv_tst_trr_linear_share_m1)))
void tst_linear_share_m1(tbfloat16m1_t src, bfloat16_t * baseAddr, unsigned long offset, uint32_t attr = 0);
```

**最终形态**：一个 `static inline __device__` 的函数，函数体由编译器从内建 `__builtin_rvv_tst_trr_linear_share_m1` 生成。三个 attribute 的作用：

| attribute | 作用 |
| --- | --- |
| `__always_inline__` + `static inline` | 强制内联，kernel 里不留函数调用 |
| `__nodebug__` | 不生成调试信息（避免为每个 intrinsic 建 DWARF 条目） |
| `clang_builtin_alias(...)` | **关键**：把这个函数名绑定到内建上，调用它等于直接调内建 |

`clang_builtin_alias` 是 clang 的通用机制：它让一个普通 C 函数名成为某个 builtin 的别名，这样 SDK 可以用人类可读的名字 + 正确的 C 函数签名，而不用让用户直接写 `__builtin_rvv_*`。

**`__device__` 在 host 编译时怎么处理**：`rms_norm_kernel.su` 里的 host 侧代码也会看到这个头。clang 对 `__device__` 函数在 host 侧的处理是"允许声明但不可调用"，所以 host 编译不会炸。

### 50.4 第 3 层：所选重载 —— 为什么是 `tile_150g.h` 而不是 `tile_inter.h`

同名函数在**两个**头文件里都有声明，签名只差第二个参数的类型：

| 头文件 | 行 | 声明中的第二个参数 |
| --- | --- | --- |
| `siorigin_tile_150g.h` | 15532 | `bfloat16_t * baseAddr` ← **内建类型** |
| `siorigin_tile_inter.h` | 11942 | `sifmt::bfloat16 * baseAddr` ← **类类型** |

两者都会被包含（`tile_150g.h` 由 `siorigin_tile.h:20` 引入；`tile_inter.h` 由 `SiTe/SiTe.hpp:29` 引入，见 `SiTe.hpp` 里 `#ifdef __SIPU_ARCH__` 包住的那两行）。

**选中的是 `tile_150g.h` 那个**，因为 kernel 传的 `shared_buff` 声明为 `__shared__ bfloat16_t [...]`（`rms_norm_kernel_bf16.hpp:45`）——`bfloat16_t` 是 clang 内建类型，精确匹配 `tile_150g.h:15532` 的签名。

`sifmt::bfloat16`（`SiTe/sifmt/sifmt_fp.hpp:278`）虽然定义了 `operator bfloat16_t()` 转换（`:138`），但**指针之间不做转换**——`bfloat16_t*` 无法隐式转成 `sifmt::bfloat16*` 来匹配另一个重载。所以 `tile_inter.h:11942` 那个版本在这个调用点不会被选中。

> 两个版本的差别不只是签名风格：`tile_inter.h` 用 `sifmt::` 标量类型是为了配合 SiTe 的高层算子库，`tile_150g.h` 用内建类型是 clang 生成的"裸"接口。**底层 builtin 是同一个**（都写 `TILE_HEADER(tst_trr_linear_share_m1)`），所以生成的汇编完全一致。

### 50.5 第 4 层：builtin 名 → LLVM intrinsic

`__builtin_rvv_tst_trr_linear_share_m1` 这个名字里的 `tst_trr_linear_share_m1` 一部分是编译器 TableGen **自动生成**的。生成规则在：

`compiler-toolchain/llvm-project/llvm/include/llvm/IR/IntrinsicsRISCVXSOTileFuncType.td`

`MemLinearConfig` 类（`:264-287`）负责拼名字：

```c
let MemNameStr   = !if(!eq(mem_idx, 0), "_GLOBAL", "_SHARE");
let MaskNameStr  = !if(!eq(tmask_idx, 0), "", "_TM");
let TileNameStr  = "_M"#!cond(!eq(tile_size, 1) : "1", ...);
let PseudoStr    = MemNameStr#remote_name#TileNameStr#MaskNameStr;
let IntrinsicStr = !tolower(PseudoStr);
```

对 `mem_idx=1`（share）、`tile_size=1`（m1）、`tmask_idx=0`：`PseudoStr = "_SHARE" + "" + "_M1" + ""`，`tolower` → `"_share_m1"`。

再接上操作名前缀（`IntrinsicsRISCVXSOTile.td:363-377` 的 `TStoreIntrinsic_m`）：

```c
multiclass TStoreIntrinsic_m {
  let IsLinear = true in {
    defm _trir_linear : TStoreLinearIntrinsic_m<0x14>;
  }
  ...
}
```

以及 clang 侧 `riscv_siorigin_xsotile_memory.td:299` 的 `defm tst_trr_linear : TStoreLinearBuiltin_m<"0sPeu", "tst_linear", ...>`，最终拼出：

```
tst  +  _trir_linear  +  _share_m1
```

即 builtin 名 `tst_trr_linear_share_m1`，LLVM intrinsic 名 `llvm.riscv.tst.trir.linear.share.m1.*`，汇编助记符 `tst.trir.linear.u32.share`。

**名字各段的含义**：

| 段 | 值 | 含义 |
| --- | --- | --- |
| `tst` | — | Tile STore |
| `trir` | `t`=tile, `r`=GPR, `i`=imm, `r`=GPR | 操作数类型：`Td`(tile), `rs1`(基址 GPR), `imm1`(偏移立即数), `rs3`(GPR) |
| `linear` | — | unit-stride（线性）访存模式 |
| `u32` | — | 32 字节粒度 |
| `share` | — | share memory 空间（对应 `mem_idx=1`）；另一种是 `global` |
| **`m1`** | LMUL=1 | **汇编里被省略**（见 50.7） |

同一份 fatbin 里也能看到对应的 `tst.trir.linear.u32.global`（写回 global memory 用）和 `tld.trir.linear.u32.share`（读回），名字规则一致。

### 50.6 第 5 层：ISA 指令的真身（TableGen）

builtin 只是编译器前端的名字；真正定义**指令编码**的地方在 TableGen：

**（a）指令类的实例化** —— `compiler-toolchain/llvm-project/llvm/lib/Target/RISCV/RISCVInstrInfoXSOTile150GMemory.td:198`

```c
let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
  defm TST_TRII_LINEAR_U32 : TMemLinear150G_m<T_StOp, uimm5>;
  defm TST_TRIR_LINEAR_U32 : TMemLinear150G_m<T_StOp, GPR>;     // ← 本 kernel 用的
  ...
}
```

`T_StOp` = `0b001000`（`RISCVInstrInfoXSOTile.td:17`）。`TMemLinear150G_m` 再展开成 `_GLOBAL` / `_SHARE` 两套（同文件 `:44-50`），`_SHARE` 带 `IsL2BStore = true`。

**（b）编码位域** —— `RISCVInstrFormatsXSOTile.td:110-137` 的 `RVInstTMemLinear`：

```c
  // Word 2
  let Inst{63}      = 0b1;
  let Inst{62 - 55} = td;         // 源 tile 寄存器
  let Inst{54 - 52} = tile_size;  // m1=0, m2=1, m4=2, m8=3, mf2=-1, mf4=-2, mf8=-3
  let Inst{51 - 47} = rs1;        // 基址
  let Inst{43}      = rmt;
  let Inst{42}      = order;
  let Inst{41}      = r_rsp;
  let Inst{40 - 39} = 0b00;
  // Word 1
  let Inst{31}      = off_enc;    // 1 = trir（rs3 是 GPR）
  let Inst{30 - 25} = memuop;     // T_StOp = 0b001000
  let Inst{24 - 20} = rs3;        // 第 4 个操作数
  let Inst{19 - 15} = imm1;       // 立即数偏移
  let Inst{11 - 10} = 0b00;
  let Inst{9}       = mode{24};
  let Inst{8 - 7}   = 0b00;
```

**（c）汇编名生成** —— 同文件 `:6-22`，`toAsmStr` 把 TableGen 名转成汇编名：

```c
class toAsmStr<string input> {
  string str = !tolower(!subst("_", ".", version_stripped));
}
```

`TST_TRIR_LINEAR_U32` → 小写、下划线转点 → `tst.trir.linear.u32`。再拼上 `_SHARE` 的 `toAsmStr` 结果 `.share`，得到 `tst.trir.linear.u32.share`。

**（d）`TILE_HEADER` 的名字从哪来** —— `toAsmStr` 的 `!subst("_", ".", ...)` 解释了汇编里的点号；而 builtin 名用的是 `IntrinsicStr = !tolower(PseudoStr)`（保留下划线）。**两者同源、不同格式**：

| | 分隔符 | 例子 |
| --- | --- | --- |
| builtin / intrinsic 名 | 下划线 | `tst_trr_linear_share_m1` |
| 汇编助记符 | 点号 | `tst.trir.linear.u32.share` |

### 50.7 实机验证：反汇编逐字段核对

`sikernel/source/source_builtin/attention/rms_norm/build/SipuWork.rms_norm/librms_norm_sipu_fatbin.llvm.asm` 里，bf16 kernel 的 `tst_linear_share_m1` 调用点生成了：

```asm
8040000001c4: 81b0607b 8205807b   tld.trir.linear.u32.global  T4, (a1), 0x0, s11
8040000001d0: 68c044fb           tcsrr.r.b64 s1, tkernelidx
8040000001d4: 88bd               andi  s1, s1, 0xf
8040000001d6: 04ca               slli  s1, s1, 0x12
8040000001d8: 01f4e4b3           or    s1, s1, t6
8040000001dc: 91a0607b 8204907b   tst.trir.linear.u32.share   T4, (s1), 0x0, s10
```

`0x1dc` 那条就是 `tst_linear_share_m1`。我把 64 bit 机器码 `0x8204907b91a0607b` 按 TableGen 的位域逐字段拆开验证：

| 位域 | 实测值 | 期望 | 对应 |
| --- | --- | --- | --- |
| `Inst{63}` | 1 | 1 | ACE 标志位 |
| `Inst{62:55}` | `0b100` = 4 | T4 | **`td` = T4 ← 源 tile `bf16_din`** |
| `Inst{54:52}` | 0 | `T_M1Op` = 0 | **tile_size = m1** |
| `Inst{51:47}` | `0b1001` = 9 | x9 = s1 | **`rs1` = 基址** |
| `Inst{43}` | 0 | 0 | rmt |
| `Inst{42}` | 0 | 0 | order |
| `Inst{41}` | 0 | 0 | r_rsp |
| `Inst{40:39}` | 0 | 0 | 保留 |
| `Inst{31}` | 1 | 1 | **`off_enc`=1 → trir** |
| `Inst{30:25}` | `0b1000` = 8 | `T_StOp` = `0b001000` = 8 | **`memuop` = 存操作** |
| `Inst{24:20}` | `0b11010` = 26 | x26 = s10 | **`rs3`** |
| `Inst{19:15}` | 0 | 0 | `imm1` = 0（偏移在 rs3/rs1 里算） |
| `Inst{11:10}` / `Inst{9}` / `Inst{8:7}` | 0 | 0 | 保留 / mode 位 / 保留 |

**15 个字段全部吻合**。

注意指令打印的**两个 32 bit 组顺序**：`91a0607b 8204907b` 中，**第一组是低 32 位（Word 1），第二组是高 32 位（Word 2）**——把 `0x8204907b` 当高位、`0x91a0607b` 当低位拼成 64 bit 才对得上 TableGen 的 `Inst{63:32}` / `Inst{31:0}` 划分。

### 50.8 关于 `m1` 在汇编里消失

注意生成的助记符是 `tst.trir.linear.u32.share`，**结尾没有 `.m1`**。这不是 bug，是设计：`toAsmStrWithType`（`IntrinsicsRISCVXSOTileFuncType.td:48-55`）里有一条替换规则：

```c
class toAsmStrWithType<string input> {
  defvar substs = [ ..., [".m1", ""], ];
  string str = !foldl(toAsmStr<input>.str, substs, acc, subst,
                      !subst(subst[0], subst[1], acc));
}
```

`.m1` 被显式替换成空串——**m1 是默认 LMUL，省略不写**；只有 m2/m4/m8/mf2/mf4/mf8 才会在助记符里出现后缀。

验证：把整份 fatbin 里所有 `tst.trir.linear.u32.share*` 去重，只有 `tst.trir.linear.u32.share` 一种形式——因为 rms_norm 只用了 m1。而编码里的 `Inst{54:52} = 0` 才是真正的 LMUL 信息（`T_M1Op = 0b000`，`RISCVInstrInfoXSOTile.td:21`）。

### 50.9 完整展开链一图

```
kernel/rms_norm_kernel_bf16.hpp:56
  tst_linear_share_m1(bf16_din, shared_buff[threadIdx.x], i*tile_size);
        │  实参类型: (tbfloat16m1_t, bfloat16_t*, int)
        │
        ├─ bfloat16_t 是 clang 内建 → 匹配 tile_150g.h 的重载
        │
        ▼
  siorigin_tile_150g.h:15532
  TILE_HEADER(tst_trr_linear_share_m1) void tst_linear_share_m1(
        tbfloat16m1_t src, bfloat16_t * baseAddr, unsigned long offset, uint32_t attr = 0);
        │
        │  TILE_HEADER 宏定义在 siorigin_tile.h:18
        ▼
  static inline __device__ __attribute__((__always_inline__, __nodebug__))
  __attribute__((clang_builtin_alias(__builtin_rvv_tst_trr_linear_share_m1)))
  void tst_linear_share_m1(...);
        │
        │  clang_builtin_alias 绑定到编译器内建
        ▼
  __builtin_rvv_tst_trr_linear_share_m1
        │
        │  名字由 TableGen 生成:
        │    IntrinsicsRISCVXSOTileFuncType.td:264  MemLinearConfig
        │      PseudoStr = "_SHARE" + "" + "_M1" + ""  → IntrinsicStr="_share_m1"
        │    riscv_siorigin_xsotile_memory.td:299  defm tst_trr_linear
        │      overloaded_name = "tst_linear"
        │    合成: tst + _trir_linear + _share_m1
        ▼
  llvm.riscv.tst.trir.linear.share.m1.*        (LLVM IR intrinsic)
        │
        │  指令定义 (CodeGen):
        │    RISCVInstrInfoXSOTile150GMemory.td:198
        │      defm TST_TRIR_LINEAR_U32 : TMemLinear150G_m<T_StOp, GPR>
        │        编码类 RVInstTMemLinear 在 RISCVInstrFormatsXSOTile.td:110
        ▼
  tst.trir.linear.u32.share  T4, (s1), 0x0, s10       (汇编助记符)
        │  名字: toAsmStr = tolower + "_"→"."  (IntrinsicsRISCVXSOTileFuncType.td:13)
        ▼
  0x8204907b 91a0607b                                  (64-bit 机器码)
```

### 50.10 小结

- `tst_linear_share_m1` 在 kernel 里**确实来自 `siorigin_tile_150g.h:15532`**。`siorigin_tile_inter.h:11942` 有同名重载，但因为参数是 `sifmt::bfloat16*`（类类型）而非内建 `bfloat16_t*`，**不会被选中**。
- `TILE_HEADER(T)` 定义为 `__rvv_generic __attribute__((clang_builtin_alias(__builtin_rvv_##T)))`（`siorigin_tile.h:18`），配合 `__rvv_generic`（`:15`）展开成 `static inline __device__ __attribute__((__always_inline__, __nodebug__)) __attribute__((clang_builtin_alias(__builtin_rvv_tst_trr_linear_share_m1)))`——**一个强制内联的设备端函数，函数体由编译器从内建生成**。
- 内建的**名字**由 TableGen 拼成（`MemLinearConfig.IntrinsicStr` → `"_share_m1"`，再前缀 `tst_linear`）；**指令编码**定义在 `RISCVInstrInfoXSOTile150GMemory.td:198`，位域在 `RISCVInstrFormatsXSOTile.td:110`。
- 最终汇编是 `tst.trir.linear.u32.share T4, (s1), 0x0, s10`，机器码 `0x8204907b 91a0607b`，**15 个编码字段已逐位核对全部吻合**。
- 助记符里看不到 `.m1` 是因为 `toAsmStrWithType` 显式把 `.m1` 替换为空——m1 是默认 LMUL；真正的 LMUL 编码在 `Inst{54:52}`。

---

## 51. 通用方法：从 `siorigin_tile_150g.h` 的函数名反查它的功能

**问题**：`siorigin_tile_150g.h` 里几千条 intrinsic 全无注释，`tst_linear_share_m1(...)` 这种名字只能靠猜。有没有**通用办法**，从一个函数名反推出它对应 ISA 手册里哪条指令、LLVM intrinsic 是什么、汇编助记符是什么？

**SDK**：`/share_data/sicx_sdk/release/latest/`（→ `2609101917`）
**编译器**：`/share/users/like/package/compiler-toolchain/`
**ISA 文档**：`/softhome/like/asset/code/isa/index.html`

### 51.0 结论先行：四种办法，按"是否需要动手"排序

| # | 办法 | 命令/位置 | 得到什么 | 成本 |
| --- | --- | --- | --- | --- |
| **1** | 直接编译探针 | 一条 `clang -S` | **汇编助记符**（注意 `-mcpu` 名字与 `-O2`） | 最低，10 秒 |
| **2** | 反汇编已构建产物 | `llvm-objcopy` + `llvm-objdump` | 汇编助记符 + 实际寻址代码 | 低，产物已在 |
| **3** | 查自动生成的编译器测试 | `grep` 测试目录 | 助记符 + 类型重载全表 | 最低，纯 grep |
| **4** | 读 TableGen 命名规则 | 编译器源码 | **名字每一段的含义**（唯一能给出"为什么"的办法） | 高，但一次学会终身受用 |

**没有注释不是疏漏，是设计**：这些名字是 TableGen 从结构化配置**机械生成**的（见 51.4）。所以名字本身就是一份编码过的文档——只要知道解码表，任何一条都能反查。

四条路对 `tst_linear_share_m1` 的答案一致：**`tst.trir.linear.u32.share`**，ISA 手册章节「Tile Unit-stride Share Memory Store」。

---

### 51.1 办法 1：直接编译一个探针文件（推荐首选）

**思路**：`siorigin_tile_150g.h` 里的函数都是 `clang_builtin_alias` 包装（见第 50 节），写一个最小函数调用它，让编译器把汇编打出来。

新建 `/tmp/probe.cpp`。**写得越朴素越好**——不需要 `__global__`，不需要 `__shared__`，不需要任何额外 flag：

```c
#include <tile_vector.h>
#include "siorigin_tile.h"

void probe(tbfloat16m1_t s, bfloat16_t *p, long off) {   // 偏移用变量，才能拿到 trir 形态
  tst_linear_share_m1(s, p, off);
}
```

编译命令（**`-O2` 是关键，见下面的坑**）：

```bash
SDK=/share_data/sicx_sdk/release/latest
CLANG=$SDK/bin/nds64le-elf-newlib-v5d/bin/clang
INC=$SDK/bin/nds64le-elf-newlib-v5d/lib/clang/20/include

$CLANG -S --target=riscv64 -mcpu=sipu150 -O2 -I$INC probe.cpp -o -
```

**实测输出**（整个函数就这么点）：

```asm
_Z5probeu19__tile_bfloat16m1_tPDF16bl:
.Lsipu.res.func0:
	tst.trir.linear.u32.share	T0, (a0), 0, a1   ← 目标
	ret
```

一行就拿到了助记符。再拿 `tst.trir.linear.u32.share` 去 ISA 文档里搜，就能定位到「Tile Unit-stride Share Memory Store」章节。

#### 51.1.1 四个必须注意的点（都是实测踩出来的）

**（1）`-mcpu` 的值是 `sipu150`，不是 `si150`。**

```
clang: error: unsupported argument 'si150' to option '-mcpu='
```

这是我最初失败的直接原因。**不要猜 CPU 名，让编译器自己报**（列表很长，直接 grep）：

```bash
$CLANG --target=riscv64 -print-supported-cpus | grep -i sipu
```

实测输出就是三行：

```
	sipu150
	sipu160
	sipu170
```

注意 `scc` 的 `-arch` 取值是同一套名字（`scc --list-gpu-arch` 输出 `sipu_150` / `sipu_160` / `sipu_170`，多一个下划线），**都不是 `si150`**。`sikernel` 自己的 `setup.sh:39-55` 用的是裸数字 `150/160/170`，写进 `SIARCH` 之类的环境变量。这三套写法互不相通，混用就报错。

**（2）`-mllvm` 的那两个 flag 名字也不是 `--si-xxx`。**

```
clang (LLVM option parsing): Unknown command line argument '--si-disable-reuse-legalize'.
clang (LLVM option parsing): Did you mean '--siorigin-disable-reuse-legalize'?
```

正确拼写是 `-mllvm --siorigin-disable-reuse-legalize` 和 `-mllvm --siorigin-default-threading-mode=single-thread`。

**不过更重要的是：这两个 flag 对反查助记符根本不需要。** 我做了对照实验，加上、去掉、甚至完全不写 `-mllvm`，`tst.trir.linear.u32.share` 一字不差。`-DLAZYLOAD` 同理——我把整个 SDK 头文件树和 `sikernel` 都 grep 了一遍，**`LAZYLOAD` 这个宏根本不存在**，加不加输出完全一样。这些都来自构建系统里为了别的目的（编译期、线程模式）加的开关，探针里请直接删掉，少一层出错面。

**（3）`__shared__` / `__global__` 在探针里是多余的——而且会误导你。**

这是最容易走偏的一点。我最初的推论是"第二个参数必须指向 shared memory 才能得到 `.share`"，**这是错的**。实测对照（同样传一个普通 `bfloat16_t *p` 参数）：

| 调用的函数 | 生成的助记符 |
| --- | --- |
| `tst_linear_share_m1(s, p, off)` | `tst.trir.linear.u32.share` |
| `tst_linear_global_m1(s, p, off)` | `tst.trir.linear.u32.global` |

**访存空间由函数名的 `share`/`global` 那一段决定，和指针的实际地址空间无关。** 用普通指针参数就能拿到 `.share`，不需要 `__shared__` 数组，也不需要 `__global__` 函数。

反过来，在 `--target=riscv64` 这个路径下写 `__shared__` / `__global__` 不但没用，还会被编译器忽略并告警：

```
warning: 'sipu_global' attribute ignored [-Wignored-attributes]
warning: 'sipu_shared' attribute ignored [-Wignored-attributes]
```

（这两个属性只在 `scc` 那套 `-x sipu` 设备编译路径下才有意义，是另一条产线。）**换句话说：`__shared__` 加与不加，助记符一样；但它会给你一个"我控制了地址空间"的错觉。** 探针里删掉。

**（4）`-O2` 不能省，`-fno-inline` 反而会污染输出。**

省略优化（默认 `-O0`）时，编译器会插入大量 tile 寄存器到栈的保存/恢复，输出里混进一堆噪声：

```asm
	tst.trii.linear.u32.global	T0, (a0), 0, 0   # 1024-byte Folded Spill  ← 噪声
	tld.trii.linear.u32.global	T0, (a0), 0, 0   # 1024-byte Folded Reload ← 噪声
	...（重复十几条，且全是 global，不是你要的）...
	tst.trir.linear.u32.share	T0, (a0), 0, a1   ← 真正那一行
```

这些 `1024-byte Folded Spill` 是 ABI 把 1 KB 的 tile 寄存器溢出到栈上，**和你的调用无关**。`-O2` 之后整个函数只剩两行，噪声全部消失。所以**别加 `-fno-inline`**（那是为了看别的才用的），想看干净输出就加 `-O2`。

#### 51.1.2 变体：只想要 LLVM intrinsic 名

把 `-S` 换成 `-S -emit-llvm`（同样加 `-O2`）：

```bash
$CLANG -S --target=riscv64 -mcpu=sipu150 -O2 -emit-llvm -I$INC probe.cpp -o -
```

**实测输出**：

```llvm
tail call void @llvm.riscv.tst.trir.linear.share.m1.tv512bf16.p0.i64.i64.i32(
    <[tile] 512 x bfloat> %0, ptr %1, i64 0, i64 %2, i32 0)
```

名字里的 `tv512bf16` 就是 `tbfloat16m1_t`（512 个 bfloat 元素的 tile 向量），后缀 `p0.i64.i64.i32` 是参数类型签名。**注意 `tst.trir` 这一段在 intrinsic 名里原样保留**——它对应的是 `IntrinsicsRISCVXSOTile.td:365` 的 `defm _trir_linear : TStoreLinearIntrinsic_m<0x14>`，和立即数版 `:368` 的 `defm _trii_linear : TStoreLinearIntrinsic_m<0x1C>` 是**两条不同的 LLVM intrinsic**。这一点在第 51.6 节展开。

---

### 51.2 办法 2：反汇编已构建的产物

如果算子已经构建过，`build/` 下就有 fatbin 对象，不用重新编译。

`sikernel/source/source_builtin/attention/rms_norm/build/` 里的产物分两层：

**（a）直接看已生成的 `.llvm.asm`**（最省事）。构建系统已经把它反汇编好了，路径在 `build/SipuWork.rms_norm/` 下（注意是**三个下划线**开头）：

```bash
cd <sikernel>
grep "tst.trir.linear.u32.share" \
  source/source_builtin/attention/rms_norm/build/SipuWork.rms_norm/librms_norm_sipu_fatbin.llvm.asm
```

**（b）自己从 `.su.o` 里重新提取**——这个配方来自 `mma_dte_tile_tensor/CMakeLists.txt:238-243` 的 `add_custom_command`（`OBJCOPY`/`OBJDUMP`/`ELFINFO` 在 `:215-217` 定义），是构建系统生成 `.llvm.asm` 的原始命令：

```bash
cd <sikernel>
BIN=/share_data/sicx_sdk/release/2609101917/bin/nds64le-elf-newlib-v5d/bin
OBJ=source/source_builtin/attention/rms_norm/build/SipuObj/rms_norm/kernel/rms_norm_kernel.su.o

# 0) 先确认段名（别手写，见下面的坑）
$BIN/llvm-readelf -S "$OBJ" | grep -i fatbin
#   [59] __sipu_fatbin   PROGBITS  ...  ← 这就是段名

# 1) 抽出 fatbin 段
$BIN/llvm-objcopy -O binary --only-section=__sipu_fatbin "$OBJ" /tmp/fb.bin

# 2) 反汇编
$BIN/llvm-objdump --mattr=+m,+c,+f,+a,+xsiorigin -zCDS /tmp/fb.bin
```

**实测输出**（与 `.llvm.asm` 完全一致）：

```
8040000001dc: 91a0607b 8204907b   tst.trir.linear.u32.share   T4, (s1), 0x0, s10
80400000027c: 91a0607b 8185007b   tst.trir.linear.u32.global  T3, (a0), 0x0, s10
804000000560: 91a0607b 8304907b   tst.trir.linear.u32.share   T6, (s1), 0x0, s10
```

**坑**：段名不是想当然的 `__si__fatbin`，而是 `__sipu_fatbin`（四个下划线开头，中间三个）——写错会得到 `not found in any input file`。**别手写，用上面第 0 步的 `llvm-readelf -S` 输出复制。** 同样的道理，`--mattr` 里的架构扩展名（`+xsiorigin`，即 `+xsotile150g`）也去 `llvm-readelf` 的段内容或者构建系统的 `add_custom_command` 里抄，别自己拼。

我实测跑了一遍完整配方，输出与 `.llvm.asm` 逐字节一致：

```
8040000001dc: 91a0607b 8204907b	tst.trir.linear.u32.share	T4, (s1), 0x0, s10
804000000560: 91a0607b 8304907b	tst.trir.linear.u32.share	T6, (s1), 0x0, s10
804000000926: 91a0607b 8304907b	tst.trir.linear.u32.share	T6, (s1), 0x0, s10
```

顺带一提，这个 64 bit 机器字 `0x8204907b91a0607b` 的**低 32 位是先打印的**（`91a0607b` 在前），解析位域时别弄反顺序——这一点第 50 节详细讲过。

---

### 51.3 办法 3：查编译器自带的映射表（纯 grep，零编译）

编译器仓库里有一批**自动生成**的测试，每一条 intrinsic 都写明了它生成的汇编。这就是官方维护的"函数名 → 助记符"对照表：

`compiler-toolchain/llvm-project/clang/test/CodeGen/RISCV/xsotile150g/autogenerated/`

共 **101 个**测试文件，按指令族命名（`xsotile_tld_linear_gpr_test.cpp`、`xsotile_tst_linear_imm_test.cpp`、`xsotile_tadd_3op_ttt_test.cpp`、…）。

直接 grep 函数名：

```bash
grep -n "tst_linear_share_m1" \
  compiler-toolchain/llvm-project/clang/test/CodeGen/RISCV/xsotile150g/
```

**实测输出**（`xsotile_tst_linear_imm_test.cpp:3048-3050`）：

```c
// CHECK: tst.trii.linear.u32.share
void __rvv_tst_trr_linear_share_m1(tint32m1_t op0, int32_t * op1) {
   return TILE_FUNC(tst_linear_share_m1)(op0, op1, 0);
}
```

`// CHECK:` 那行就是答案。

**注意**：这个测试里是 `tst.trii`（imm 版），而 rms_norm 内核里生成的是 `tst.trir`（GPR 版）——因为测试传的第三个参数是立即数 `0`，而内核传的是变量 `i*tile_size`。**同一个 C 函数名，参数是常量还是变量，会产生不同的助记符**。这一点如果只查测试文件会看漏，要结合办法 1/2 一起看。

---

### 51.4 办法 4：读懂命名规则（唯一能给出"为什么"的办法）

前三个办法都是"查表"，遇到表里没有的名字就卡住。真正通用的办法是理解**名字是怎么被机械拼出来的**——拼装规则在 TableGen 里，是可读的源码。

#### 51.4.1 名字的四段结构

`tst_linear_share_m1` 拆成四段，每段来源不同：

```
  tst      _linear      _share        _m1
   │           │            │           │
 指令族     访存模式     访存空间     LMUL
   │           │            │           │
riscv_si_    MemLinear    MemNameStr   TileNameStr
xsotile_     Config       ("_SHARE")   ("_M1")
memory.td    .td          同上          同上
   │           │            │           │
   └───────────┴────────────┴───────────┘
                    ↓
        PseudoStr = MemNameStr # remote_name # TileNameStr # MaskNameStr
        IntrinsicStr = !tolower(PseudoStr)
```

拼装核心在 `compiler-toolchain/llvm-project/llvm/include/llvm/IR/IntrinsicsRISCVXSOTileFuncType.td:264-290` 的 `MemLinearConfig`：

```c
class MemLinearConfig<int tile_size, int mem_idx, int tmask_idx,      // :264
                      string remote_name> : MemBasicConfig {
  let TileNameStr = "_M"#!cond(!eq(tile_size, 1) : "1", ...);         // :274
  let MemNameStr  = !if(!eq(mem_idx, 0), "_GLOBAL", "_SHARE");        // :282
  let MaskNameStr = !if(!eq(tmask_idx, 0), "", "_TM");                // :286
  let PseudoStr    = MemNameStr#remote_name#TileNameStr#MaskNameStr;  // :288
  let IntrinsicStr = !tolower(PseudoStr);                             // :289
}
```

对 `mem_idx=1`、`tile_size=1`、`tmask_idx=0`、`remote_name=""`：

```
PseudoStr = "_SHARE" + "" + "_M1" + ""  =  "_SHARE_M1"
IntrinsicStr = tolower("_SHARE_M1")     =  "_share_m1"
```

再前缀操作名（`riscv_siorigin_xsotile_memory.td:299` 的 `defm tst_trr_linear : TStoreLinearBuiltin_m<"0sPeu", "tst_linear", ...>`），合成 **`tst_linear` + `_share_m1` = `tst_linear_share_m1`**。

#### 51.4.2 查名字各段含义的"字典"

有了规则，就能反查任意一段。以下字典全部可从源码机械提取：

**（a）操作名前缀** —— 来自 `riscv_siorigin_xsotile_memory.td` 里的 `defm` 名：

| 前缀 | 含义 | 定义位置 |
| --- | --- | --- |
| `tld` | Tile LoaD | `xsotile_memory.td` |
| `tst` | Tile STore | `xsotile_memory.td` |
| `tmul` / `tadd` / `tsub` | tile 逐元素乘/加/减 | `riscv_siorigin_xsotile_alu.td` |
| `tcvt` | Tile ConVerT（类型转换） | `xsotile_alu.td` |
| `tmv` | Tile MoVe（tile↔RVV 搬移） | `riscv_siorigin_xsotile_mov.td` |
| `tmma` / `tmva` | Tile Matrix Multiply / MAtrix-vector | `riscv_siorigin_xsotile_mma.td` |
| `tacp` | Tile Asynchronous CoPy | `xsotile_memory.td` |
| `tcsrr` / `tcsrw` | Tile CSR 读/写 | `riscv_siorigin_xsotile_tcsr.td` |

**（b）访存模式** —— 决定 `_linear` / `_stride` / `_index` / `_blk` 这几段：

| 名字里出现 | ISA 章节 | 含义 |
| --- | --- | --- |
| `linear` | Tile Unit-stride Load/Store | 单位步长（连续） |
| `stride` | Tile Strided Load | 带步长 |
| `index` | Tile Indexed Load | 索引向量寻址（gather/scatter） |
| `blk` | Tile Block Load | 块状 |

**（c）访存空间** —— `MemNameStr`（`IntrinsicsRISCVXSOTileFuncType.td:282`）：

| 名字里出现 | `mem_idx` | 含义 |
| --- | --- | --- |
| `global` | 0 | global memory |
| `share` | 1 | **share memory**（kernel 的 `__shared__`） |

**（d）操作数类型 `trir` 这类字母串** —— `t`=tile reg，`r`=标量 GPR，`i`=立即数，按操作数顺序排列。例如 `trir` = `Ts1`(tile) + `rs1`(GPR) + `imm1`(imm) + `rs3`(GPR)。这个在 `IntrinsicsRISCVXSOTileFuncType.td` 的 `TLSU*ParamNames` 里定义。

**（e）元素类型字母** —— 这是**唯一需要查表**的一段（不是自解释的）。表在 `RISCVInstrInfoXSOTile.td` 的 `SiType` 定义里（`:66` 起的 `class SiType<string typename, string typecode, ...>`）。我把它机械提取出来：

| 字母 | 类型 | 字母 | 类型 |
| --- | --- | --- | --- |
| `c` | S8 | `m` | F8E4M3 |
| `x` | **F16** | `n` | F8E5M2 |
| `y` | **BF16** | `t` | **TF32** |
| `f` | **F32** | `g` | MXI4 |
| `i` | S32 | `e` | MXI8 |
| `l` | S64 | `a` | MXF8E5M2 |
| `h` | S4 | `b` | MXF8E4M3 |
| `j` | MXF6E3M2 | `k` | MXF6E2M3 |
| `p` | MXF4 | | |

（字母来自 `IntrinsicsRISCVXSOTileFuncType.td:401-405` 的 `Basic_type_common = ["i","c","x","y","f","m","n","t","l"]`；类型码来自 `RISCVInstrInfoXSOTile.td` 的 `SiType` 表。）

**（f）LMUL** —— `_m1` / `_m2` / `_m4` / `_m8` / `_mf2` / `_mf4` / `_mf8`，来自 `TileNameStr`（`IntrinsicsRISCVXSOTileFuncType.td:274-281`）。

**（g）`.tm` 后缀** —— tile mask，`tmask_idx=1` 时才有（`IntrinsicsRISCVXSOTileFuncType.td:286`）。

#### 51.4.3 从名字到 ISA 章节

名字段与 ISA 手册章节的对应是**一一映射**的。把 (b) 和 (c) 两段的取值组合起来，正好是手册里的章节标题：

| 名字片段 | ISA 手册章节名 |
| --- | --- |
| `tld_linear_global` | Tile Unit-stride Global Memory **Load** |
| `tst_linear_global` | Tile Unit-stride Global Memory **Store** |
| `tld_linear_share` | Tile Unit-stride Share Memory **Load** |
| **`tst_linear_share`** | **Tile Unit-stride Share Memory Store** ← `tst_linear_share_m1` 落在这 |
| `tld_stride_global` | Tile Strided Global Memory Load |
| `tst_index_share` | Tile Indexed Share Memory Store |
| `tst_blk_global` | Tile Block Global Memory Store |

（章节名清单可直接在 ISA 文档里 grep 到，例如搜 `Tile Unit-stride` 会返回上面这四类。）

---

### 51.5 以 `tst_linear_share_m1` 为例，走一遍四条路

| 办法 | 操作 | 结果 |
| --- | --- | --- |
| 1. 编译探针 | `clang -S --target=riscv64 -mcpu=sipu150 -O2 probe.cpp`（普通指针 + 变量偏移） | `tst.trir.linear.u32.share` |
| 2. 反汇编 | `llvm-objdump` on fatbin | `tst.trir.linear.u32.share T4, (s1), 0x0, s10` |
| 3. 查测试 | `grep tst_linear_share_m1 .../xsotile150g/` | `// CHECK: tst.trii.linear.u32.share`（**imm 版**，与内核的 trir 不同） |
| 4. 命名规则 | 查 TableGen | `tst`(存) + `linear`(连续) + `share`(共享内存) + `m1` |

**四条路的结果互相印证，唯一差异是办法 3 给的是 `trii`（测试里传字面量 `0`），内核实际是 `trir`（传变量 `i*tile_size`）——这不是矛盾，是同一函数的两种编码，判据见 51.6。**

**补充：办法 1 与办法 2 的交叉验证。** 我用办法 1 的探针配方直接编译了真实内核 `kernel/rms_norm_kernel.su`（用 `scc --dryrun` 拿到的原始 clang 命令），生成：

```asm
	tst.trir.linear.u32.share	T4, (s1), 0, s10
	tst.trir.linear.u32.share	T6, (s1), 0, s10
```

与办法 2 从 fatbin 反汇编出来的**完全一致**。两条独立路径互为印证，可以放心当作定论。

**名字逐段解释**：

| 段 | 取值 | 含义 | 来源 |
| --- | --- | --- | --- |
| `tst` | Tile STore | 写内存 | `xsotile_memory.td` 的 `defm tst_trr_linear` |
| `linear` | — | unit-stride（连续地址） | `TStoreLinearBuiltin_m` 的 `MemoryOpKind::LINEAR` |
| `share` | `mem_idx=1` | 写 **share memory** | `MemNameStr`（`IntrinsicsRISCVXSOTileFuncType.td:282`） |
| `m1` | LMUL=1 | 一个 tile register（1 KB） | `TileNameStr`（`:274`） |

**一句话功能**：把 tile 寄存器 `src` 里的 1 KB 数据，以单位步长写进 **share memory** 的 `baseAddr + offset` 处。

**ISA 手册对应**：`/softhome/like/asset/code/isa/index.html` → 搜 `tst.trir.linear.u32.share` → 章节「**Tile Unit-stride Share Memory Store**」，手册原型：

```
tst.trir.linear.u32.share.[m2/m4/m8/mf2/mf4/mf8].[tm]  Ts1, (rs1), imm1, rs3
```

注意手册原型里 `m1` 也是省略的（默认 LMUL），和实际生成的助记符一致。

---

### 51.6 一个补充：`trir` vs `trii` 由**偏移的数值**决定，不是函数名决定的

这是最容易踩的坑，也是我初稿写错、实测纠正过来的一处。同一个 `tst_linear_share_m1`，**第三个参数（偏移）的编译期取值**决定生成哪个助记符。

#### 51.6.1 表面规则

| 源码 | 实测助记符 | 编码 |
| --- | --- | --- |
| `tst_linear_share_m1(s, p, off)`（`off` 是变量） | **`tst.trir.linear.u32.share`** | `r` = register，偏移放在 GPR 里 |
| `tst_linear_share_m1(s, p, 0)`（字面量 0） | **`tst.trii.linear.u32.share`** | `i` = immediate，偏移编进指令 |
| `tst_linear_share_m1(s, p, 32768)` | **`tst.trir.linear.u32.share`** | 常量太大，装不进立即数 → 退回寄存器 |

**注意第三行**：光说"常量走 `trii`、变量走 `trir`"是**不准确的**。实测 `32768` 虽然是字面量，依然生成 `trir`。真正的判据不是"是不是常量"，而是"**这个常量能不能被立即数编码放下**"。

我实测扫了边界，实测结果如下：

| 偏移（字面量） | 助记符 |
| --- | --- |
| `0`、`32`、`64`、`512`、`1024`、`2048`、`31744`、`32736` | `tst.trii.…` |
| `1`、`2`、`4`、`8`、`16`、`33`、`1025` | `tst.trir.…` |
| `32768`、`32737`、`-32` | `tst.trir.…` |

规律是 **`offset >= 0` 且 `offset % 32 == 0` 且 `offset <= 32736`**（即 `1024*31 + 32*31`）时用 `trii`，否则一律 `trir`。

#### 51.6.2 它到底是怎么实现的

判据实现在编译器里，就一个函数——`compiler-toolchain/llvm-project/llvm/lib/Target/RISCV/RISCVISelDAGToDAG.cpp:3671-3697` 的 `decomposeTileOffset`：

```c
static std::optional<std::pair<uint64_t, uint64_t>>
decomposeTileOffset(int64_t Offset, bool Has32BOffset = true,
                    bool Has1024BOffset = true) {
  if (Offset < 0 || Offset % 32 != 0) {          // ← 负数、非 32 对齐：立即数编码放不下
    return std::nullopt;
  }
  ...
  if (Offset <= 1024 * 31 + 32 * 31) {           // ← 32736，上界
    return std::make_pair((Offset % 1024) / 32, Offset / 1024);
  }
  return std::nullopt;                            // ← 超界：退回 trir
}
```

**为什么是这个式子**：ISA 的立即数版把偏移拆成两个 5 bit 字段——`imm1`（32 字节为单位）和 `rs3` 位置上的 `imm2`（1024 字节为单位）。两个 5 bit 各自最大 31，所以总上界 `31*1024 + 31*32 = 31744 + 992 = 32736`。`Offset % 32 != 0` 时低位字段装不下，同样退回。

这个函数被两个 `ComplexPattern` 调用（`RISCVInstrInfoXSOTilePseudosMemory.td:892-896`）：`Tile32BAnd1024BOffset` 和 `Tile32BOffset`。它们挂在指令选择模式上（`RISCVInstrInfoXSOTile150GPseudosMemory.td:809` 的 `PatTStTRIR150G`）：

```c
// 150GPseudosMemory.td:820-833（节选）
def : Pat<(riscv_tile_store $td, $rs1, (uimm5:$imm1), $rs3, $mode),          ← trir 形态
          (inst $td, $rs1, $imm1, $rs3, $mode)>;
def : Pat<(riscv_tile_store $td, $rs1, (uimm5),
          (Tile32BAnd1024BOffset uimm5:$imm32, uimm5:$imm1024), $mode),       ← 拆成两个 imm
          (inst_trii $td, $rs1, $imm32, $imm1024, $mode)>;
```

**所以真相是：C 层只有一个函数、一条 builtin（`TILE_HEADER(tst_trr_linear_share_m1)`），到机器码层才由指令选择器根据偏移数值分流成两条指令。** 名字里的 `trir`/`trii` 是**后端**的区分，不是前端两个 builtin。

我初稿写的"`defm tst_trr_linear` 与 `defm tst_tri_linear` 是分开注册的两条 builtin"**是错的**——`riscv_siorigin_xsotile_memory.td` 里**只有 `defm tst_trr_linear`，没有 `tst_tri_linear`**（我 grep 过，命中数 0）。前端只有一条。

#### 51.6.3 对反查的实用影响

- **办法 3 查测试文件看到的多是 `trii`**（测试里传字面量 `0`），而 rms_norm 内核传的是 `i * tile_size`（变量），所以是 `trir`。**两个都对，是同一函数的两种编码**，别以为查错了。
- **想稳定复现内核的行为，就用变量偏移**；想复现测试文件的输出，就用 `0`。
- 如果 ISA 文档按 `trii`/`trir` 分章节，记得**你的代码是哪种取决于偏移的编译期取值**，不是取决于你调了哪个函数。

---

### 51.7 小结

- **首选办法 1**（编译探针）——一条命令、10 秒、结果最准，且能反映你自己代码的实际参数形态。探针写得越朴素越好：普通指针参数即可，**不要加 `__shared__`/`__global__`/`-DLAZYLOAD`/`-fno-inline`**，只要 `-O2`。
- **办法 2**（反汇编）适合已经构建过的算子，不用重编。两条路交叉验证过，结果一致。
- **办法 3**（查测试）纯 grep，适合快速浏览某个指令族的全部变体。
- **办法 4**（读 TableGen）是唯一能解释"为什么叫这个名字"的办法，也是遇到查不到的名字时唯一的出路。命名规则完全机械：`操作名 + 模式 + 空间 + LMUL`，各段的字典都能从 `IntrinsicsRISCVXSOTileFuncType.td` 和 `RISCVInstrInfoXSOTile.td` 里机械提取。
- 名字段到 ISA 手册章节是**一一映射**的：`tst_linear_share` → 「Tile Unit-stride Share Memory Store」，直接拿助记符去 ISA 文档里搜即可。
- 唯一不自解释的一段是**元素类型字母**（`c`=S8、`x`=F16、`y`=BF16、`f`=F32、`t`=TF32、`h`=S4…），这张表在 51.4.2(e) 里给出了。

**上手时最容易出错的两个地方**（都是本次实测踩出来的）：

1. **命令行**：`-mcpu` 的值是 `sipu150`（用 `-print-supported-cpus` 自己查），不是 `si150`；`-mllvm` 的 flag 是 `--siorigin-...`，不是 `--si-...`。**这两个 flag 和 `-DLAZYLOAD` 对反查都不需要，直接删掉。**
2. **语义**：访存空间（`.share`/`.global`）由**函数名**决定，不是由指针的地址空间决定；`trii`/`trir` 由**偏移能否装进立即数**决定（`offset >= 0 && offset % 32 == 0 && offset <= 32736`），不是由"常量还是变量"这个笼统说法决定。**别只看测试文件就下结论。**

---

## 52. 探针里 `tld_linear_global_m1` 被优化没了 —— 死代码消除（DCE）

**问题**：按第 51.1 节的办法写 `tld` 探针，编译出来只剩一条 `ret`：

```c
// /share/users/like/temp/probe.2.cpp
#include <tile_vector.h>
#include "siorigin_tile.h"

void probe(const bfloat16_t * baseAddr, unsigned long offset) {
  tbfloat16m1_t res = tld_linear_global_m1(baseAddr, offset);
}   // ← res 从没被用过
```

```bash
$CLANG -S --target=riscv64 -mcpu=sipu150 -O2 -I$INC probe.2.cpp -o -
```

```asm
_Z5probePKDF16bm:
.Lsipu.res.func0:
	ret                     ← 调用整条不见了
```

**答：是，被死代码消除（Dead Code Elimination）删掉了。** 而且是**完全正确**的行为——不是编译器 bug，也不是探针写错了语法。

### 52.0 一句话结论

- `tld` 系列 intrinsic 只带 `IntrReadMem` 语义，**没有** `IntrHasSideEffects`。当它的返回值没人用、结果对程序再无影响时，LLVM 有权把整条调用删掉。
- 反过来，`tst` 系列带 `IntrWriteMem`，**写内存本身就是可观测的副作用**，所以裸调用 `tst` 不会被删（这也是第 51.1 节的探针为什么一直好用——**它恰好是 store**）。
- **修法**：让结果逃逸。最小改法是再调一次 `tst` 把 `res` 存回去；或改用 `-O0`；或只想要 intrinsic 名时用 `-O0 -emit-llvm`。

### 52.1 实测证据：确实是 DCE，不是"没生成"

把优化等级扫一遍，代码条数随 `-O` 变化，这是 DCE 的典型特征：

| 编译选项 | 生成的 `tld`/`tst` 条数 |
| --- | --- |
| `-O0` | **6** |
| `-O1` | **0** |
| `-O2` | **0** |

同一份源码，`-O0` 下六条指令都在，`-O1` 起全部消失。**指令选择是正常发生的，是后端的优化遍把它删了。**

从 IR 层面看得更清楚——同样加 `-O2`，加不加 `-emit-llvm` 结果完全不同：

```bash
# -O0 -emit-llvm：call 在
$CLANG ... -O0 -emit-llvm probe.2.cpp -o - | grep call
  %8 = call <[tile] 512 x bfloat> @llvm.riscv.tld.trir.linear.global.m1.tv512bf16.p0.i64.i64.i32(
        ptr %6, i64 0, i64 %7, i32 0)

# -O2 -emit-llvm：call 没了，连参数都失去了名字
$CLANG ... -O2 -emit-llvm probe.2.cpp -o - | grep define
  define dso_local void @_Z5probePKDF16bm(ptr nocapture noundef readnone %0, i64 noundef %1) #0 {
                                           ^^^^^^^^^^^^^^^^^^^^^^^^
```

注意 `-O2` 那行里参数 `%0` 被标成了 **`readnone`**——连"这个指针被读过"都不再成立，因为读它的那条调用已经被删干净了。函数体是空的。

### 52.2 根本原因：load 与 store 的副作用属性不对称

这不是偶然，是 TableGen 里刻意区分的。两组 intrinsic 的属性定义在 `compiler-toolchain/llvm-project/llvm/include/llvm/IR/IntrinsicsRISCVXSOTile.td`：

**`tld` 用的 `TLoadAttrIntrinsic`（`:128-136`）**：

```c
class TLoadAttrIntrinsic<bits<26> scalar_num, bit has_pass_thru, list<LLVMType> rs2_ty>
    : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                            !listconcat(...),
                            [IntrReadMem, IntrArgMemOnly],          // ← 只有"读内存"
                            "", [SDNPMemOperand]>,
      TileIntrinsic {
```

**`tst` 用的 `TStoreAttrIntrinsic`（`:278-287`）**：

```c
class TStoreAttrIntrinsic<bits<26> scalar_num, list<LLVMType> rs2_ty>
    : DefaultAttrsIntrinsic<[],
                            !listconcat([llvm_anyvector_ty, llvm_anyptr_ty], ...),
                            [IntrWriteMem, IntrArgMemOnly],         // ← "写内存"
                            "", [SDNPMemOperand]>,
      TileIntrinsic {
```

两者都**没有** `IntrHasSideEffects`（这个 flag 在文件里只加给了 `:147`、`:291`、`:394` 等少数几条，**linear load 不在其中**）。区别只在于 `IntrReadMem` 与 `IntrWriteMem`：

| | 属性 | 结果没人用时 |
| --- | --- | --- |
| `tld`（load） | `IntrReadMem` + `IntrArgMemOnly` | **可删除**——读内存没有可观测效果 |
| `tst`（store） | `IntrWriteMem` + `IntrArgMemOnly` | **保留**——写内存是可观测副作用 |

编译出的 IR 属性行可以直接验证这一点。`-O0 -emit-llvm` 里，调用指令的注释行明明白白写着两个不同的 memory 语义：

```llvm
; probe.2.cpp（tld，load）
; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)   ← read

; s.cpp（tst，store）
; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)  ← write
```

这一串就是 `IntrReadMem`/`IntrWriteMem` + `IntrArgMemOnly` 的落地产物（`argmem` = `IntrArgMemOnly`，`read`/`write` = 对应的内存效果）。**"读"和"写"的差别，就是这里一条调用能不能被删的全部依据。**

**所以第 51.1 节的 `tst_linear_share_m1` 探针能直接出结果，纯粹是因为它是 store。** 换成任何 `tld`，只要返回值不用，在 `-O2` 下都会消失。这一点我在写第 51 节时没有意识到——**探针方法对 store 有效，对 load 需要额外一步**，现在补上。

### 52.3 四种修法（实测）

**修法 1（推荐）：让结果逃逸。** 再调一次 `tst` 把 `res` 存回去，两条指令都会保留：

```c
#include <tile_vector.h>
#include "siorigin_tile.h"

__shared__ bfloat16_t sbuf[512];

void probe(const bfloat16_t * baseAddr, unsigned long offset) {
  tbfloat16m1_t res = tld_linear_global_m1(baseAddr, offset);
  tst_linear_share_m1(res, sbuf, 0);        // ← 让 res 逃逸
}
```

实测 `-O2` 输出（两条都在，正好一次拿到 load 和 store 两个助记符）：

```asm
	tld.trir.linear.u32.global	T0, (a0), 0, a1
	lui	a0, %hi(sbuf)
	addi	a0, a0, %lo(sbuf)
	tst.trii.linear.u32.share	T0, (a0), 0, 0
```

（`tld` 那行末尾的 `a1` 就是 `offset` 变量，印证了第 51.6 节说的"变量偏移走 `trir`"。）

**修法 2：降优化等级到 `-O0`。** 直接有效，函数里六条指令都在：

```asm
	tld.trir.linear.u32.global	T0, (a0), 0, a1    ← 真正的那条
	tst.trii.linear.u32.global	T0, (a0), 0, 0     # 1024-byte Folded Spill
	tld.trii.linear.u32.global	T0, (a0), 0, 0     # 1024-byte Folded Reload
	tst.trii.linear.u32.global	T0, (a0), 0, 0
	tld.trii.linear.u32.global	T0, (a0), 0, 0     # 1024-byte Folded Reload
	tst.trii.linear.u32.global	T0, (a0), 0, 0
```

**代价**：后五条全是第 51.1.1(4) 节说的 ABI spill 噪声（`1024-byte Folded Spill`/`Reload`），而且**全是 `.global`、全是 `trii`**——和你要找的那条形态完全不同，很容易挑错。要自己从里面认出第一条。**能用修法 1 就别用这个。**

**修法 3：只要 LLVM intrinsic 名，就用 `-O0 -emit-llvm`。** 这一条很实用——`-emit-llvm` 发生在优化遍**之前**，所以即使在 `-O2` 下其实也……不，实测不行：

| 命令 | `llvm.riscv.tld` 调用是否还在 |
| --- | --- |
| `-O0 -emit-llvm` | **在** |
| `-O2 -emit-llvm` | **不在**（已被删） |

**`-emit-llvm` 只有在配合 `-O0` 时才能拿到名字**，`-O2 -emit-llvm` 一样会被 DCE 掉。这一点容易想当然，实测确认一下：

```bash
$CLANG -S --target=riscv64 -mcpu=sipu150 -O0 -emit-llvm -I$INC probe.2.cpp -o -
```

```llvm
call <[tile] 512 x bfloat> @llvm.riscv.tld.trir.linear.global.m1.tv512bf16.p0.i64.i64.i32(
    ptr %6, i64 0, i64 %7, i32 0)
```

**修法 4：`volatile` 无效。** 把指针参数写成 `volatile const bfloat16_t *` 也没用——实测 `-O2` 下依然为空。原因同上：`IntrReadMem` 语义已经声明了"只读"，`volatile` 管的是 C 语言层面的访问次序，管不到一个 intrinsic 的返回值有没有人用。

### 52.4 一条判断规则

以后写新探针时，先问一句：**这条 intrinsic 的结果会不会被丢掉？**

| intrinsic 家族 | 属性（定义处） | 裸调用在 `-O2` 下 |
| --- | --- | --- |
| `tld_*`（load） | `TLoadAttrIntrinsic`：`IntrReadMem`（`:134`） | **会被删**，必须让结果逃逸 |
| `tst_*`（store） | `TStoreAttrIntrinsic`：`IntrWriteMem`（`:282`） | 保留，可直接裸调 |
| `tmul` / `tadd` / `tcvt` 等 ALU | `TALUIntrinsic`：`IntrNoMem`（`:461`） | **会被删**——`tmul` 实测 `-O2` 下为空 |
| `tacp_*` 等带 `IntrHasSideEffects` 的 | 明确标了 side effects（`:394`、`:723`、`:794`…） | 保留 |

（`tmul` 用到 `TALUIntrinsic` 这条链路我不是只读源码推断的——实测 `tmul(a,b)` 结果丢掉后 `-O2` 输出为空，存回去才出现 `tmul.ttt.f32`，与属性表一致。）

**简单记法：凡是"算出一个值给你"的（load、算术），在 `-O2` 下都必须把值用掉；凡是"往外写东西"的（store、异步拷贝），裸调就安全。** 第 51.1 节推荐的探针模板恰好落在第二类，所以一直没暴露这个问题。

### 52.5 小结

- `tld_linear_global_m1` 没有被"优化成 ret"，而是**整条调用被 DCE 删掉了**——源码里 `res` 从未被使用，编译器删除它是完全正确的。
- 根因是 `tld` 只带 `IntrReadMem`（`IntrinsicsRISCVXSOTile.td:134`），而 `tst` 带 `IntrWriteMem`（`:282`）。**load 的结果无人使用时可以安全删除，store 不行。**
- 最省事的修法是**再调一次 `tst` 把结果存回去**；其次 `-O0`；要 intrinsic 名就用 `-O0 -emit-llvm`（**注意 `-O2 -emit-llvm` 也会被删**）。
- `volatile` 指针无效。
- 第 51.1 节的探针模板本身没错，但**只对 store 类 intrinsic 开箱即用**；探 `tld` 或 ALU 类时必须让结果逃逸。

---

## 53. `vfredosum` 这类 RVV intrinsic 声明在哪个头文件里？

**问题**：`rms_norm` device code 里用到 `__riscv_vfredosum_vs_f32m1_f32m1`、`__riscv_vfmv_v_f_f32m1`、`__riscv_vfadd_vv_f32m1` 这些 RVV intrinsic，它们的声明在哪个头文件？

`source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_f32.hpp:57-68`：

```c
vfloat32m1_t vsum_vec = tmv_v_f32(f32_sq_sum, 0U);
vfloat32m1_t zero_vec = __riscv_vfmv_v_f_f32m1(0.0f, 1);
vfloat32m1_t temp_vec;
for (long i = 1; i < 8; ++i) {
    temp_vec =  tmv_v_f32(f32_sq_sum, i);
    vsum_vec =  __riscv_vfadd_vv_f32m1(temp_vec, vsum_vec, 32);
}
zero_vec = __riscv_vfredosum_vs_f32m1_f32m1(vsum_vec, zero_vec, 32);
sum =  __riscv_vfmv_f_s_f32m1_f32(zero_vec);
```

**答：它们不在任何头文件里声明 —— 它们是 clang 的编译器内建函数（compiler builtins），由 `#pragma clang riscv intrinsic vector` 激活，在 Sema 做名字查找时动态生成声明。**

这个结论有点反直觉（毕竟源码写着 `#include <riscv_vector.h>`），所以下面把证据摆全。

### 53.0 一句话结论

- `#include <riscv_vector.h>` 拉进来的那份头文件，**只有 typedef、enum 和少量宏，一个 RVV intrinsic 的函数声明都没有**。
- 真正的"声明"由 clang 自己持有：TableGen 从 `riscv_vector.td` 生成一张 builtin 名字表编译进 clang 二进制，**名字查找命中时现场造一个 `FunctionDecl` 出来**。
- 头文件之所以必不可少，是因为它里面有一行 **`#pragma clang riscv intrinsic vector`**（`:21`）——**开这个开关才是关键，头文件内容本身几乎不做这件事**。
- 头文件路径：**`$($CLANG -print-resource-dir)/include/riscv_vector.h`**，靠 clang 的 resource dir 机制自动找到，**不需要任何 `-I`**。

### 53.1 先看"头文件里到底有什么"

头文件位置（`latest` 指向当天版本，`2609111951`）：

```
/share_data/sicx_sdk/release/latest/bin/nds64le-elf-newlib-v5d/lib/clang/20/include/riscv_vector.h
```

**全文 426 行**，构成如下：

| 内容 | 行 | 数量 |
| --- | --- | --- |
| `#pragma clang riscv intrinsic vector` | `:21` | **1 行（关键）** |
| `enum __RISCV_FRM` / `__RISCV_VXRM` | `:24`、`:90` | 2 个 |
| `#define __riscv_vsetvl*` / `__riscv_vsetvlmax*` / `__riscv_vlenb` | `:32-85` | **45 行宏** |
| `typedef __rvv_*_t v*_t;` | `:94-425` | **323 行 typedef** |

**423/426 行是类型别名和宏，函数声明 0 行。** 决定性的一步——在整个 clang include 树里搜这几个名字：

```bash
$ grep -rn "__riscv_vfredosum" $CLANG_INC/     # → 无输出
$ grep -rn "__riscv_vfadd_vv_f32m1" $CLANG_INC/  # → 无输出
$ grep -rn "__riscv_vfmv_v_f_f32m1" $CLANG_INC/  # → 无输出
```

**整个 `include/` 目录（含所有子目录）里，这些名字一次都没出现过。** 头文件里确实没有它们的声明。

作为对照，头文件里**真正**存在的 `__riscv_v*` 只有 45 行，全都是 `vsetvl` 系列的**宏**，例如：

```c
// riscv_vector.h:47
#define __riscv_vsetvl_e32m1(avl) __builtin_rvv_vsetvli((size_t)(avl), 2, 0)
```

注意它展开到 `__builtin_rvv_vsetvli`——**`__builtin_` 前缀暴露了底下的真身**。`vfredosum` 这类没有对应宏，是直接以名字暴露给用户的。

### 53.2 三个实验：定位"开关"到底是哪一行

光看头文件内容不能证明"是 pragma 在起作用"。直接做减法定量验证：

**实验 A —— 只给类型，不给 pragma、不 include（`t5.cpp`）：**

```c
typedef __rvv_float32m1_t vfloat32m1_t;
vfloat32m1_t f(vfloat32m1_t a, vfloat32m1_t b){
  return __riscv_vfredosum_vs_f32m1_f32m1(a,b,32);
}
```

```bash
$CLANG -S --target=riscv64 -mcpu=sipu150 -O2 t5.cpp -o -
```

```
t5.cpp:2:56: error: use of undeclared identifier '__riscv_vfredosum_vs_f32m1_f32m1'
```

→ **编译失败**。类型够用了（`__rvv_float32m1_t` 是 clang 内建类型，见 53.5），但名字不存在。

**实验 B —— 加上 pragma，依然不 include（`t6.cpp`）：**

```c
#pragma clang riscv intrinsic vector
typedef __rvv_float32m1_t vfloat32m1_t;
vfloat32m1_t f(vfloat32m1_t a, vfloat32m1_t b){
  return __riscv_vfredosum_vs_f32m1_f32m1(a,b,32);
}
```

```asm
	li	a0, 32
	vsetvli	zero, a0, e32, m1, ta, ma
	vfredosum.vs	v8, v8, v9
	ret
```

→ **编译通过，正常生成 `vfredosum.vs`。** 头文件一行没 include。

**实验 C —— `#include <riscv_vector.h>`，不写 pragma（`t4.cpp`）**：同样通过（因为头文件第 21 行替你把 pragma 写了）。

**三个实验合起来锁死了结论**：让 `__riscv_vfredosum_*` 从"未声明标识符"变成"可调用函数"的，**就是那一行 pragma**，跟头文件里的 323 行 typedef、45 行宏都没有关系。

### 53.3 头文件是怎么被找到的：resource dir 自动发现

这一点很实用——**`rms_norm` 的编译命令里没有任何指向 clang include 的 `-I`**。

`source/source_builtin/attention/rms_norm/build/CMakeFiles/rms_norm.dir/build.make:84`：

```
/share_data/sicx_sdk/release/2609101917/bin/scc -arch=sipu_150 --keep-dir ... \
  -c -fPIC -o .../rms_norm_kernel.su.o ... kernel/rms_norm_kernel.su \
  -I/share/users/like/package/sikernel/include \
  -I/share/users/like/package/sikernel/source/source_builtin/utils \
  -DTARGET_SIPU_ARCH=150
```

两个 `-I` 都是 sikernel 自己的目录，**没有一个是 clang 的 include**。那份 `riscv_vector.h` 是 clang 按 resource dir 约定自己找的：

```bash
$ $CLANG -print-resource-dir
/share_data/sicx_sdk/release/2609111951/bin/nds64le-elf-newlib-v5d/lib/clang/20
```

于是 `<riscv_vector.h>` 解析为 `<resource-dir>/include/riscv_vector.h`。两个实验印证：

| 命令 | 结果 |
| --- | --- |
| 不传任何 `-I`，直接编译 `t4.cpp` | **通过**，生成 `vfredosum.vs`（自动发现） |
| 加 `-nostdinc` | `fatal error: 'riscv_vector.h' file not found` |

编译产物里的 `.d` 依赖文件也坐实了实际路径：

```
$ grep riscv_vector build/_SipuObj/rms_norm/kernel/rms_norm_kernel.su.d
/share_data/sicx_sdk/release/2609101917/bin/nds64le-elf-newlib-v5d/lib/clang/20/include/riscv_vector.h
```

**推论**：换 SDK 版本时这个路径跟着变，**不要在脚本里硬编码**；需要时用 `$(CLANG -print-resource-dir)`。

同一个 resource dir 里还有一批同族文件，都是 `clang_tablegen` 生成的：

```
siorigin_tile.h              845 B      ← 总入口
siorigin_tile_150g.h         1.23 MB    ← 150 的全部 tile builtin
siorigin_tile_160g.h
siorigin_tile_170g.h
tile_vector.h                      9.7 KB     ← tile 类型 typedef + 常量
```

### 53.4 名字从哪来：TableGen 生成、编进 clang

既然头文件里没有，名字必然在编译器里。链路是这样的：

**（1）TableGen 里声明。** `compiler-toolchain/llvm-project/clang/include/clang/Basic/riscv_vector.td:2392`：

```c
// 14.3. Vector Single-Width Floating-Point Reduction Instructions
defm vfredusum : RVVFloatingReductionBuiltin;
defm vfredosum : RVVFloatingReductionBuiltin;
```

（带舍入模式的版本在 `:2384`，用 `RVVFloatingReductionBuiltinRoundingMode`。）这些 multiclass 定义在 `riscv_vector_common.td:707`：

```c
multiclass RVVFloatingReductionBuiltin {
  defm "" : RVVOutOp0BuiltinSet<NAME, "fd", [["vs", "vSv", "SvvSv"]]>;
  ...
}
```

`"vs"` 就是 `vfredosum` **`_vs`** 后缀的来源，`"fd"` 是 float/double 类型组合。

**（2）生成 `.inc`。** `clang/include/clang/Basic/CMakeLists.txt:187`：

```cmake
clang_tablegen(riscv_vector_builtin_prototypes.inc -gen-riscv-vector-builtin-prototypes
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../../../
  SOURCE riscv_vector.td
  TARGET ClangRISCVVectorBuiltinPrototypes)
```

Backend 实现在 `clang/utils/TableGen/RISCVVEmitter.cpp`（`TableGen.cpp:425` 注册 `gen-riscv-vector-builtin-prototypes`）。

**（3）编进 clang 二进制。** 直接对 clang 可执行文件做 `strings`：

```bash
$ strings -a $CLANG | grep vfredosum
vfredosum_vs            ← TableGen 里的记录名
vfredosum.vs            ← 汇编助记符
llvm.riscv.vfredosum    ← LLVM intrinsic 名
```

**这三个字符串就是全部答案**——名字表以字符串形式烧进了 clang，运行时按需匹配。

### 53.5 pragma 到 Sema 的完整链路

`#pragma clang riscv intrinsic vector` 只做一件事：**置一个 bool**。

**（1）解析 pragma。** `compiler-toolchain/llvm-project/clang/lib/Parse/ParsePragma.cpp:4163` `PragmaRISCVHandler::HandlePragma`，关键在 `:4193`：

```c
if (II->isStr("vector"))
  Actions.RISCV().DeclareRVVBuiltins = true;          // ← 就这一行
else if (II->isStr("sifive_vector"))
  Actions.RISCV().DeclareSiFiveVectorBuiltins = true;
```

这个 bool 声明在 `clang/include/clang/Sema/SemaRISCV.h:49`，默认 `false`。

**（2）名字查找时拦截。** `clang/lib/Sema/SemaLookup.cpp:951`：

```c
if (RISCV().DeclareRVVBuiltins || RISCV().DeclareSiFiveVectorBuiltins) {
  if (!RISCV().IntrinsicManager)
    RISCV().IntrinsicManager = CreateRISCVIntrinsicManager(*this);

  RISCV().IntrinsicManager->InitIntrinsicList();

  if (RISCV().IntrinsicManager->CreateIntrinsicIfFound(R, II, PP))
    return true;                                       // ← 找到了，当 builtin 处理
}
```

**这就是"头文件里找不到声明却能编译"的机制**：正常的标识符查找会失败，但 Sema 在查找流程里插了一段——**只要 pragma 开过，就先问一问 RVV builtin 表里有没有这个名字**。

**（3）动态造声明。** `clang/lib/Sema/SemaRISCV.cpp`：

- `:518` `RISCVIntrinsicManagerImpl::InitIntrinsicList()` —— 遍历全部 `RVVIntrinsicRecords`，按类型/LMUL 展开（`:522` 调 `ConstructRVVIntrinsics`）
- `:672` `RISCVIntrinsicManagerImpl::CreateIntrinsicIfFound(...)` —— 用当前查的那个名字去匹配，**命中就现场造一个 `FunctionDecl` 塞进查找结果**

**注意 `ConstructedRISCVVBuiltins` 这个幂等标志**（`:520`）——builtin 表是**懒构造**的，第一次真正用到某个 RVV 名字时才展开，不是每个 TU 都白付这个代价。

### 53.6 类型又是从哪来：`__rvv_*` 是 clang 内建类型

上面实验 A/B 都用了 `__rvv_float32m1_t` 却**没有 include 任何头文件**，说明它不是头文件 typedef。它定义在 `clang/include/clang/Basic/RISCVVTypes.def:151`：

```c
RVV_VECTOR_TYPE_FLOAT("__rvv_float32m1_t", RvvFloat32m1, RvvFloat32m1Ty, 2,  32, 1)
```

`.def` 文件经 X-macro 展开成 clang 的 `BuiltinType`。头文件 `riscv_vector.h:361` 只是给它起了个短名字：

```c
typedef __rvv_float32m1_t vfloat32m1_t;
```

**所以 `vfloat32m1_t`（能用、要 include）和 `__rvv_float32m1_t`（用的同一个类型、不用 include）是同一个东西的两个名字。** 写探针时用后者可以少 include 一个头——当然练手之外不建议。

### 53.7 对照：本项目里三种不同的"名字来源"机制

值得放在一起对比，因为 rms_norm kernel 里三种都用到了：

| 名字 | 机制 | 声明在哪 |
| --- | --- | --- |
| `__riscv_vfredosum_vs_f32m1_f32m1` | **pragma 激活的 compiler builtin** | 无头文件，Sema 动态生成（53.5） |
| `__riscv_vsetvl_e32m1` | **真·宏** | `riscv_vector.h:47`，展开到 `__builtin_rvv_vsetvli` |
| `tmv_v_f32` / `tld_linear_global_m1` / `tmul` | **`clang_builtin_alias`** | `siorigin_tile.h` + `siorigin_tile_150g.h` |

第三种（tile intrinsic，第 49–52 节反复打交道的那些）机制又不一样，它**确实是头文件声明的**，但不是普通函数——`siorigin_tile.h:15-18` 这几行宏是全部机关：

```c
#define __rvv_generic \
    static inline __device__ __attribute__((__always_inline__, __nodebug__))
#if defined(__SIPU_ARCH__) && (__SIPU_ARCH__ >= 100)
#define TILE_HEADER(T) __rvv_generic __attribute__((clang_builtin_alias(__builtin_rvv_##T)))
```

`siorigin_tile_150g.h:9964` 展开后就是：

```c
static inline __device__ __attribute__((__always_inline__, __nodebug__))
__attribute__((clang_builtin_alias(__builtin_rvv_tmv_vtr_f32)))
vfloat32m1_t tmv_v_f32(tfloat32m1_t src, unsigned long index);
```

`clang_builtin_alias` 把用户可见的 `tmv_v_f32` 绑到 `__builtin_rvv_tmv_vtr_f32` 上——**最后还是落到 clang 内建**。所以三种机制殊途同归，只是包装层次不同。

### 53.8 实操推论

1. **要查某个 RVV intrinsic 的精确签名，别翻 `riscv_vector.h`** ——里面没有。查 `compiler-toolchain/.../clang/Basic/riscv_vector.td`（声明处）和 `riscv_vector_common.td`（签名模板 `[["vs", "vSv", "SvvSv"]]`），或者直接编译一个探针让编译器报错告诉你签名。
2. **写探针/RVV 小例子时，`#pragma clang riscv intrinsic vector` 或 `#include <riscv_vector.h>` 二选一，必须有。** 只 include `siorigin_tile.h` 是不够的——它的 include 链里虽然间接带了 `riscv_vector.h`（`siorigin_tile.h:11`），但那只是因为 tile 头本身要用 `vbfloat16m1_t` 之类的 RVV 类型，**不能想当然认为"include 了 tile 头就能用 RVV intrinsic"**。
3. **头文件路径不要硬编码**，用 `$(CLANG -print-resource-dir)/include`。实测 `latest` 是软链（当前指向 `2609111951`），而 `build.make:84` 里写死的是 `2609101917`，两者并存。
4. **`__riscv_*` 是保留前缀**：写业务代码时如果出现"未声明标识符"但看着语法没问题，先想想是不是漏了 pragma/include；反过来，如果自己定义了以 `__riscv_` 开头的符号，很可能撞上 builtin 表。

---

## 54. `tcvt_f32` / `tcvt_bf16` 的第 2 个参数是什么？为什么两边都传 `t_r32`？

**问题**：`source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_bf16.hpp` 的 `rms_norm_bf16_kernel` 里，

```c
:59       f32_din.m2e0 = tcvt_f32(bf16_din, t_r32);          // Pass 1：bf16 -> f32
:97       f32_din.m2e0 = tcvt_f32(bf16_din, t_r32);          // Pass 2：bf16 -> f32
:102        f32_wgt.m2e0 = tcvt_f32(bf16_wgt, t_r32);        // Pass 2：weight bf16 -> f32
:106      bf16_out = tcvt_bf16(f32_out.m2e0, t_r32);         // Pass 2：f32 -> bf16
```

`tcvt_f32` 的第 2 个参数是什么含义？为什么传 `t_r32`？`tcvt_bf16` 的第 2 个参数又是什么含义？为什么**仍旧**传 `t_r32`？

**答：两个函数的第 2 个参数是同一个东西——形参名就叫 `attr`，是一个 `uint32_t` 位掩码，`t_r32` 是其中的"shape（行模式）"标志位，选择 tile 的矩阵排布方式（32 行）。两边都传 `t_r32`，是因为上行转换设定的是哪种排布，下行转换就必须用同一种排布把它逆回来。**

反直觉的地方在于：`tcvt` 是逐元素类型转换，按说跟"行怎么摆"无关。但实测表明**这个参数一字之差就会算错**，而且**只有两个方向的取值不一致时才错**。下面把证据摆全。

### 54.0 一句话结论

- 第 2 个参数 `attr` **不是数据类型、不是舍入模式**，而是**一整包转换属性的位掩码**——形状模式（`t_vec`/`t_r32`/`t_r16`/`t_r8`）只是其中最低 4 位，同一参数里还塞了舍入、饱和、reuse、取负、relu 等。
- `t_r32 = 0x02`（`tile_vector.h:15`），指向 ISA 里的 **`ashape` 字段：`00` = 32 rows(r32)**（另见 `01`=r16、`10`=r8）。
- 它决定**同一条 C++ 调用生成完全不同的指令**：传 `t_r32` → `tcvt.tt.f32.bf16.r32`（矩阵形式）；传 `t_vec` → `tcvt.tt.vec.f32.bf16.m8`（vector 形式）。**同一个 builtin，两种指令形态。**
- 上下行都传 `t_r32` 是**必须一致**的：实测把上下行**一起**换成 `t_r16`/`t_r8`/`t_vec`，结果照样正确；但**只换一个方向**，立刻 57212 个元素算错。
- `t_r32` 是本项目里的既定约定（全仓库 `tcvt` 调用点统计：`t_r32` 13577 次、`t_r8` 1100、`t_r16` 832、`t_vec` 267），因为 `tld_linear_*_m1` 取回的 1 KB linear tile 天然就是 32×32B 的 `r32` 排布。

### 54.1 参数的真身：形参名就叫 `attr`

先看声明。`tcvt_f32` 在 kernel 里的这个重载（`tbfloat16m1_t` → `tfloat32m2_t`）是：

```c
// $(CLANG -print-resource-dir)/include/siorigin_tile_150g.h:1080
TILE_HEADER(tcvt_tt_vec_f32_bf16_m8) tfloat32m2_t tcvt_f32(tbfloat16m1_t src, uint32_t attr);
```

对应的下行转换：

```c
// 同文件 :1032
TILE_HEADER(tcvt_tt_vec_bf16_f32_m8) tbfloat16m1_t tcvt_bf16(tfloat32m2_t src, uint32_t attr);
```

（另有 2 个 tile 形式的重载 `:548` / `:500`，签名是 `(tbfloat16m2_t src, uint32_t attr)` / `(tfloat32m4_t src, uint32_t attr)`，见 54.4。）

两个函数的第 2 个参数**形参名一模一样，都叫 `attr`**——从编译器视角看就是同一个 `uint32_t`，没有任何类型层面的区别。

`attr` 的取值不是随便定的，全部来自 `tile_vector.h` 的一组 `const unsigned`（`:14` 起，共 30 个）：

```c
// $(CLANG -print-resource-dir)/include/tile_vector.h:14-18
const unsigned t_vec = 0x01;      // ← vector 排布（1x1024B）
const unsigned t_r32 = 0x02;      // ← 32 rows，本项目默认
const unsigned t_r16 = 0x04;      // ← 16 rows
const unsigned t_r8  = 0x08;      // ← 8 rows
const unsigned t_rne = 0x10;      // ← 从这里开始是舍入模式
```

也就是说 `attr` 是一包**按位或**出来的标志：

| 位 | 常量 | 含义 | 本 kernel 用了吗 |
| --- | --- | --- | --- |
| bit0-3 | `t_vec` / `t_r32` / `t_r16` / `t_r8` | **shape 模式（互斥，四选一）** | ✅ 用了 `t_r32` |
| bit4-7 | `t_rne`/`t_rtz`/`t_rtn`/`t_rtp`（或 `t_rni`/`t_rzi`/`t_rmi`/`t_rpi`） | 舍入模式 | ❌ 省略，走默认 |
| bit16 | `t_sat` (0x10000) | 饱和（satfinite） | ❌ |
| bit17 | `t_relu` (0x20000) | ReLU | ❌ |
| bit8 | `t_reuse1` (0x100) | 寄存器 reuse | ❌ |
| bit12 | `t_neg1` (0x1000) | 取负 | ❌ |

**所以 `t_r32` 只是"这包标志里只填了 shape 字段、其余全 0"。** kernel 完全可以写 `tcvt_f32(bf16_din, t_r32 | t_rtz)` 之类，只是没必要。

顺带说明一点：`t_rne` 和 `t_rtz` 等**同占 bit4-7**（`t_rne=0x10`、`t_rtz=0x20`…），`t_rni`/`t_rzi` 等又和它们**共用同一批位**——这是给整数转换用的另一套舍入命名。位是复用的，具体解释取决于指令的 dtype 组合，所以位掩码校验是**逐指令类型**做的，不是全局统一（见 54.6）。

### 54.2 为什么是 `t_r32`：它是 tile 的"矩阵排布方式"

ISA 文档（`/softhome/like/asset/code/isa/index.html`）在 GEMM 一节里定义了这个字段，并明确说了它名字的来历：

> `ashape`
> a矩阵的shape 00代表32 rows(r32)模式 01代表16 rows(r16)模式 10代表8 rows(r8)模式

注意文档里的版本注记（对应 2025.1.8 那版）：

> mma中的shape字段改名为ashape。

**`r32` 就是 "32 rows"，`r16` = 16 rows，`r8` = 8 rows。** ISA 在 Tile Layout 一节解释了三种排布（`index.html` Tile Reg 排布方式）：

> 32x32B的数据排布模式是行(M)方向32B排布，再按列(K)方向排布32个32B。
> 16x64B的数据排布模式是行(M)方向32B排布，再按列(K)方向排布16个32B，再按行方向(M)方向排布2组16个32B。
> 8x128B的数据排布模式是行(M)方向32B排布，再按列(K)方向排布8个32B，再按行方向(M)方向排布4组8个32B。
> Vector Tile的数据排布模式是按列(K)方向排布1x1024B。

三者的**字节总量完全相同**（都是 1024 B），区别只在"多少行为一组、多宽算一行"：

| 模式 | 排布 | 对应 ISA 字节数 | 元素数（bf16, 2 B） |
| --- | --- | --- | --- |
| `t_r32` | 32×32B | 32×32 = **1024 B** | 512 |
| `t_r16` | 16×64B | 16×64 = **1024 B** | 512 |
| `t_r8` | 8×128B | 8×128 = **1024 B** | 512 |
| `t_vec` | 1×1024B | **1024 B** | 512 |

**这四个模式是同一块 1 KB 数据的四种"切法"。** 这也解释了为什么上下行必须配套——切法不同，元素在矩阵里的位置就不同。

**为什么项目默认选 `r32`？** 因为输入是 `tld_linear_global_m1`/`tld_linear_share_m1` 取回的 1 KB **linear** tile，而 `r32` 是与之配套的 1 KB 粒度排布；更直接的证据来自 MMA：`r32` 与 `tile_M = 32` 绑定（见下），rms_norm 的每个 tile 恰好是 32 行 × 32B。全仓库统计也印证了这是主流写法：

```
t_r32 : 13577      t_r8 : 1100      t_r16 : 832      t_vec : 267
```

更直接的旁证在 `mma_dte_tile_tensor` 里——那批 util 文件**按 shape 分文件**，文件名就是设计意图：

```
mma_dte_tiled_tensor_kernel_util_non_mx_r32.hpp   // 注释：“r32 (tile_M=32) specific ...”
mma_dte_tiled_tensor_kernel_util_non_mx_r16.hpp   // 注释：“r16 (tile_M=16) specific ...”
mma_dte_tiled_tensor_kernel_util_non_mx_r8.hpp    // 注释：“r8 (tile_M=8) specific ...”
```

每个文件里 `tcvt` 用的 shape 与文件名**严格一致**（r32 文件里 128 处全是 `t_r32`，r16 文件里 20 处全是 `t_r16`，r8 文件里 32 处全是 `t_r8`），而它们服务的 MMA 也正是对应的 `tile_M = 32/16/8`。**shape 是从 MMA 的 M 维一路传下来的，`tcvt` 只是跟着走。**

### 54.3 `attr` 如何改变生成的指令：实测对照表

这一点最容易踩坑——`attr` 不只是"填充字段"，它**直接切换指令形态**。用同一份源码、只改第 2 个参数，实测（`clang -S --target=riscv64 -mcpu=sipu150 -O2`）：

```c
extern "C" tfloat32m2_t F(tbfloat16m1_t s){ return tcvt_f32(s, <ATTR>); }
```

| `<ATTR>` | 生成的指令 | 形态 |
| --- | --- | --- |
| `t_r32` | `tcvt.tt.f32.bf16.r32 T2, T0` | **矩阵形式** |
| `t_r16` | `tcvt.tt.f32.bf16.r16 T2, T0` | 矩阵形式 |
| `t_r8` | `tcvt.tt.f32.bf16.r8 T2, T0` | 矩阵形式 |
| `t_vec` | `tcvt.tt.vec.f32.bf16.m8 T2, T0` | **vector 形式** |
| `0`（省略） | `tcvt.tt.vec.f32.bf16.m8 T2, T0` | vector 形式（默认补 `t_vec`） |

**同一个 C++ builtin，传 `t_r32` 出矩阵指令，传 `t_vec` 出 vector 指令。** 这跟 ISA 文档的说法完全吻合——矩阵转换要求"必须指定 r32/r16/r8"，而 vector 转换"无需指定 r32/r16/r8"：

> 该指令用于matrix tile format的conversion，该指令需要指定r32/r16/r8模式。
> tcvt.tt.dtype.atype.r32/r16/r8.[rtz/rtn/rtp/rzi/rmi/rpi].[satfinite].[reuse1].[neg1].[relu] Td, Ts1
>
> 该指令用于vector tile的conversion，该指令无需指定r32/r16/r8模式，使用1x1024B的layout排布工作。

注意这里有个**默认值陷阱**：`attr=0`（或者只写 `t_sat` 之类）并不是"没有 shape"，编译器会**自动补上 `t_vec`**（`SiOriginTileSuffixAttr.cpp:295` `Imm |= t_vec;`），直接给你一条 vector 形式指令。所以**想用矩阵形式就必须显式写 `t_r32`/`t_r16`/`t_r8`，不能省**。

从机器码层面看，shape 就是 32 位编码里的 bit18-19（用 `llvm-mc --show-encoding` 实测，编码小端读出后取位）：

| 助记符 | 编码 | bit18-19 解码 |
| --- | --- | --- |
| `tcvt.tt.f32.bf16.r32` | `7b 60 00 60 ...` | `00` |
| `tcvt.tt.f32.bf16.r16` | `7b 60 04 60 ...` | `01` |
| `tcvt.tt.f32.bf16.r8` | `7b 60 08 60 ...` | `10` |
| `tcvt.tt.vec...m8` | `7b e0 03 60 ...` | `00`（但 bit15-17 全 1，是另一个字段的 `vecen`） |

**与 ISA 的 `ashape` 编码表逐位对上**——ISA 文档里的字段图也是 `vecen / shape / relu / rnd / sat / reuse1 / neg1` 这一串。

### 54.4 为什么上行下行必须都用 `t_r32`（含反例）

这才是问题的核心。既然四个 shape 描述的是同一块 1 KB，为什么不能上面用 `t_r32`、下面用 `t_r16`？

**因为 `tcvt` 不是逐元素无关的搬运——shape 决定了数据在转换单元里怎么重排，下行必须精确逆回上行设定的那种排布。**

我把 `rms_norm_kernel_bf16.hpp` 里的 `t_r32` 按不同方式替换后，用官方测试程序跑（`bash build.sh && ./build/test_host`，测试判定阈值 `err_rate > 0.01`，见 `test/test_host.cpp:282`）：

| 改法 | `test_host` 退出码 | 报错元素数 | 结论 |
| --- | --- | --- | --- |
| **0. 原样（全 `t_r32`）** | 0 | 0 | ✅ 正确 |
| **1. 全部换成 `t_r16`** | 0 | 0 | ✅ **仍然正确** |
| **2. 全部换成 `t_r8`** | 0 | 0 | ✅ **仍然正确** |
| **3. 全部换成 `t_vec`** | 0 | 0 | ✅ **仍然正确** |
| **4. 只把上行换成 `t_r16`，下行留 `t_r32`** | **255** | **57212** | ❌ **算错** |
| **5. 只把下行换成 `t_r16`，上行留 `t_r32`** | **255** | **57212** | ❌ **算错** |
| **6. 再跑一遍原样** | 0 | 0 | ✅（可复现） |

这张表说明两件事：

**（1）测试是灵敏的，不是"改什么都不报错"。** 作为阳性对照，我把一行真正的数学改坏（`mean = sum/original_norm_size;` → `...*3.0f;`），立刻 **255 退出码 + 114688 个元素报错**。所以第 1-3 行"全换仍然通过"是真通过，不是测试失灵。

**（2）关键不在"用哪个 shape"，而在"两个方向是否一致"。**

- 全部换成同一个 shape → 上行的排布被下行**原样逆回**，中间夹着的 `tmul`/`tadd`/`tmv` 都是逐元素或与排布无关的归约，**净效果 = 恒等**，所以结果不变。
- 只换一个方向 → 上下行的排布**对不上**，`tcvt_bf16` 按 32×32B 去解释一个按 16×64B 写进去的缓冲区，元素落到错误位置 → 大面积算错。

反例的报错分布也印证是"排布错位"而不是数值精度问题——57212 ≈ 114688 的一半，且错误索引在**每个 32 元素块内均匀铺开**（`index mod 32` 从 0 到 31 各有约 1790 个错误，完全均匀）：

```
errors by (index mod 32):
    0 : 1791      8 : 1791     16 : 1783     24 : 1783
    1 : 1786      9 : 1786     17 : 1789     25 : 1789
   ...            ...          ...           ...
    7 : 1789     15 : 1789     23 : 1791     31 : 1791
```

**均匀铺开 = 整体移位/重排，而不是几个点上的舍入误差。** 如果是精度问题，应该是零星、集中在特定数值上的；这里的形态是典型的"数据搬错了地方"。

而且报错值本身也支持这个判断——例如 `index 8, result: -0.204102, gold: -0.144531`、`index 9, result: -0.609375, gold: -0.234375`，**`result` 和 `gold` 都是合法的 bf16 数值**（`-0.609375` 是 bf16 能精确表示的），只是"对不上号"。

**所以 kernel 里四处 `tcvt` 全部写 `t_r32` 的正确理由，不是"r32 是某种默认值"，而是"上下行必须锁定同一种排布"。** 写成全 `t_r16`/全 `t_r8` 在**功能上**同样正确（实测通过），但 `t_r32` 与 32 行的 tile 排布、以及与 `tile_M = 32` 的 MMA 配套，也是全仓库的既定约定，所以照写 `t_r32`。

### 54.5 那个 `.m2e0` / `.m1e0` 是怎么来的（与 54.1 呼应）

`tcvt_f32` 返回 `tfloat32m2_t`，而 kernel 里写的是 `f32_din.m2e0 = tcvt_f32(...)`。**`.m2e0` 不是 `tfloat32m2_t` 的成员**——直接对裸的 tile 类型写 `.m2e0` 会编译报错：

```
error: member reference base type 'tfloat32m2_t'
       (aka '__tile_float32m2_t') is not a structure or union
```

真正的成员来自 SDK 的 union 包装（`include/sipu_dev_apis/dtype/union.h:204`——§47 里已经讲过这套 union 包装）：

```c
typedef tfloat32m1_t tfloat32m1;                    // :203
typedef union                                       // :204
{
    struct
    {
        tfloat32m1_t m1e0;                          // :208
        tfloat32m1_t m1e1;                          // :209
    };
    tfloat32m2_t m2e0;                              // :211  ← kernel 用的就是它
} tfloat32m2;                                       // :212
```

**所以 kernel 里声明的 `tfloat32m2 f32_din;` 是 union，`.m2e0` 是它的 2 KB 全宽视图，`.m1e0`/`.m1e1` 是拆成两个 1 KB 半区的视图**（`tbfloat16m2` 同构，`union.h:104-117`）。`tcvt_f32(bf16_din, t_r32)` 返回 2 KB 的 `tfloat32m2_t`，直接赋给 `.m2e0` 整块落地；之后用 `.m1e0` / `.m1e1` 分别做乘加。**这和 `t_r32` 无关——`.m2e0` 是数据宽度（2 个 tile reg）的事，`t_r32` 是排布方式的事，两者正交。**

### 54.6 实操规则与坑

1. **第 2 个参数就是 `uint32_t attr`，位掩码。** 想加舍入/饱和/reuse 就按位或，例如 `t_r32 | t_rtz` → `tcvt.tt.f32.bf16.r32.rtz`。
2. **矩阵形式必须显式写 shape。** `attr=0` 或只写 `t_sat` 会被编译器**自动补成 `t_vec`**（`SiOriginTileSuffixAttr.cpp:295`；160G 的对应实现在 `:315` 一带），生成 vector 形式指令，而不是你想要的矩阵形式。
3. **上下行（或上游 `tcvt` 与下游 `tcvt`）的 shape 必须一致。** 这是实测出来的硬约束（54.4 表），不是风格问题。跨函数、跨 kernel 传递 tile 时尤其要注意。
4. **合法取值是逐指令校验的，不能想当然。** 同一个 `t_r32`，在这个重载上合法、在另一个上直接被拒：

   | 调用 | 传 `t_r32` 的结果 |
   | --- | --- |
   | `tcvt_f32(tbfloat16m1_t)` | ✅ → `tcvt.tt.f32.bf16.r32` |
   | `tcvt_f32(tbfloat16m2_t)` | ✅ → `tcvt.tt.f32.bf16.r32` |
   | `tcvt_f32(tbfloat16mf8_t)` | ❌ `Only Support t_vec,...,t_relu For TCVTVEC` |

   原因是**小到只占 1/8 个 tile reg 的重载走的是 vector 形式**（声明名 `tcvt_tt_vec_*_m1`），vector 形式不接受 r32/r16/r8；**占满 1 个或更多 tile reg 的重载**（`tcvt_tt_vec_*_m8` 及以上，以及 `tcvt_tt_*` tile 系列）才接受 shape 模式。校验规则在 `compiler-toolchain/llvm-project/llvm/lib/Support/SiOriginTileSuffixAttr.cpp` 的 `checkImmediate`（`:179`）。

5. **冲突的 shape 会直接报错**，不会静默取一个：
   ```
   t_r32 | t_r16   →  error: Invalid Shape Mode For Tile Convert
   t_r32 | t_vec   →  error: Only Support t_r32,t_r16,t_r8,... For TCVTTILE
   ```
   前者来自 `TCVTTileRanges`（`SiOriginTileSuffixAttr.h:810`，`{1,3, OneHot}` —— bit1..3 必须恰好置一位），后者来自 `CheckMask` 的白名单。
6. **省略 shape 时 `-S` 和 `-c` 行为不同**：`clang -S`（出汇编文本）会打印不带 shape 的 `tcvt.tt.f32.bf16`，但 `clang -c`（出目标文件）会报 `Invalid Shape Mode For Tile Convert`。**以 `-c` 为准**；`-S` 的输出不能直接拿去汇编，反过来汇编它也会报同样的错。

### 54.7 小结

- `tcvt_f32` / `tcvt_bf16` 的第 2 个参数都是 `uint32_t attr`，**一包按位或的属性**；`t_r32`（`= 0x02`）是其中的 shape 字段，含义是**"32 rows"矩阵排布**（ISA `ashape` 的 `00`）。
- 它决定生成**矩阵形式**（`tcvt.tt.<dtype>.<atype>.r32`）还是 **vector 形式**（`tcvt.tt.vec.<dtype>.<atype>.m8`）指令——**同一个 builtin 两种形态**。
- 两边都传 `t_r32`，是因为**上行设了哪种排布，下行就得用同一种逆回来**。实测：**全换成同一个 shape（r16/r8/vec）照样正确，只换一个方向就 57212 个元素算错**；而阳性对照（真改数学）会报 114688 个错，证明测试是灵敏的。
- `t_r32` 本身不是"默认值"（默认其实是 `t_vec`）。选它是因为它与 32 行的 tile 排布、以及与 `tile_M = 32` 的 MMA 配套，也是全仓库 13577 处调用点的既定约定——但**换成别的 shape 只要上下一致，功能上同样成立**。
- 这与 §47 / §53 里那些 `tcvt` 反汇编条目完全自洽：`tcvt.tt.f32.bf16.r32 T10, T6` 里的 `.r32` 后缀，就是这里这个 `t_r32` 一路传下去的结果。

---

## 55. SpecForge 训练权重 / checkpoint 在哪里

**问题**：把 `/share/guorui/投机解码训练/SpecForge` 里面的训练权重、checkpoint 找出来。

**答**：全量清单已经落到 `like-useful/specforge_checkpoints.txt`（人读）/ `.tsv`（机器读，可直接喂脚本）。**共 467 个 checkpoint，456.95 GB，分布在 201 个 run 目录、3 个存储位置。**

这一节先给结论和定位方法，再给清单本身。**清单文件才是交付物** —— 本节只讲怎么读它、以及扫的时候哪些坑必须知道。

### 55.0 一句话结论

| 项 | 值 |
| --- | --- |
| 权重文件 | **`<run>/epoch_<N>_step_<M>/model.safetensors`** |
| checkpoint 形态 | 目录名即 `epoch_N_step_M`，**没有 HF 的 `checkpoint-<step>` 形态**，也没有多分片/index |
| 训练状态 | 同目录 `training_state.pt` + `training_state_rank{0..7}.pt` + `rng_state_rank{0..7}.pt` |
| 模型结构 | 同目录 `config.json` + `<model_type>.py`（`auto_map` 指向的 modeling 文件，会被一起存下来） |
| 总量 | **467 个 / 456.95 GB / 201 个 run 目录** |
| 三个位置 | `outputs/`（现役）199 个 201.19 GB；`cache/checkpoint-trash/`（本机回收站）71 个 18.08 GB；`/data/guorui/checkpoint-trash/`（另一台盘）197 个 237.68 GB |

### 55.1 一个 checkpoint 目录里有什么

`source/source_builtin/...` 跟这个无关，这里直接看一个真实目录。以
`/share/guorui/投机解码训练/SpecForge/outputs/q3-06-static-one-gate-difference-pool-domino-b8-l5-1node-011-fromscratch-bs2-globalbs16-epoch2-gamma7-eval1k-20260911/epoch_1_step_140000/` 为例：

```
config.json                                   1.7 KB    ← 模型结构（HuggingFace 格式）
dflash.py                                   125   KB    ← auto_map 指向的 modeling 实现
model.safetensors                           269.7 MB   ← ★ 权重本体，只有这一个文件
static_non_pre_norm_one_gate_difference_pool_domino.py  6.6 KB  ← 同一个 modeling，另一份拷贝
training_state.pt                           435.7 MB   ← 优化器/EMA 等训练状态（单文件）
training_state_rank0.pt … rank3.pt          435.7 MB each ← per-rank 分片
training_state_rank4.pt … rank7.pt           29.7 KB each ← 该 rank 不持有参数时只剩空壳
rng_state_rank0.pt … rank7.pt                10.7 KB each ← 数据加载/采样随机数状态
```

三个要点：

1. **权重就是 `model.safetensors` 一个文件**，没有 `model-00001-of-00002` 那种分片，也没有 `*.safetensors.index.json`（全量扫描确认：**0 个 index 文件、0 个分片文件**）。
2. **`training_state.pt` 不是权重**，是恢复训练用的优化器状态，通常比权重还大（上面 435 MB vs 270 MB）。**要拿权重只取 `model.safetensors` 就够**，这一条直接决定了 456 GB 里有多少是"真权重"。
3. `training_state_rank*.pt` 大小参差（435 MB / 29.7 KB）是**分片训练的正常现象** —— 只有持有参数的分片才有内容，其余是空壳。别把 29.7 KB 的那几个当成"checkpoint 坏了"。

`config.json` 里 `auto_map` 指向的类，决定了同目录那个 `.py` 叫什么：

```json
"auto_map": {
  "AutoModel": "static_non_pre_norm_one_gate_difference_pool_domino.StaticNonPreNormOneGateDifferencePoolDominoDraftModel"
}
```

所以 **checkpoint 是"自包含"的**：拿 `config.json` + `model.safetensors` + 那个 `.py`，配 `trust_remote_code=True` 就能 `from_pretrained` 加载，不依赖 SpecForge 主仓库当时的代码版本。这一点在"训练代码改了、但想复现旧 checkpoint"时特别重要。

### 55.2 一种 checkpoint 命名，两种存放约定

**命名**：`epoch_<N>_step_<M>`，`M` 是**全局 step**（不随 epoch 归零）。上面例子 `epoch_1_step_140000` 就是第 1 个 epoch 结束、全局第 140000 步。

**存放**约定有两种，**这一点必须分清**：

| 位置 | 目录名 | 说明 |
| --- | --- | --- |
| `outputs/<run>/epoch_N_step_M/` | **无时间戳** | 现役，脚本按固定路径覆盖写，**每个 run 通常只留最新 1 个**（实测 177 个 run 里 165 个只有 1 个） |
| `checkpoint-trash/<run>/epoch_N_step_M-<YYYYMMDD-HHMMSS>/` | **带时间戳后缀** | 从 `outputs/` 淘汰下来的历史存档，**同一步的多个版本会因时间戳不同而并存** |

这个"只留最新"的规律**实测成立**：17 个 run 同时出现在 `outputs/`（各 1 个）和 trash 里，**每一个都是 `outputs/` 那个 step 号最大**。例如：

| run | `outputs/` 里留的 | trash 里最大的 |
| --- | --- | --- |
| `q3-06-static-one-gate-difference-pool-domino-b8-l5-1node-011-...` | `epoch_1_step_140000` | step 135000（27 个） |
| `...-sink-pre-gated-norm-b8-l5-1node-019-...` | `epoch_1_step_130000` | step 125000（25 个） |
| `...-sink-gated-norm-b8-l5-1node-011-...` | `epoch_0_step_55000` | step 50000（10 个） |

**所以规则是清楚的：`outputs/` = 最新那一个，trash = 之前的所有历史。** 想拿"某个 run 训练过程中的任意一步"，去 trash 里按 step 找。

`cache/checkpoint-trash/` 和 `/data/guorui/checkpoint-trash/` 里都是带时间戳的那种，例：

```
epoch_0_step_5000-20260815-032205     ← 存于 2026-08-15 03:22:05
epoch_0_step_10000-20260815-041028
```

**所以现在的 `outputs/` 只看到 1 个 checkpoint 的 run，它的历史版本很可能在 trash 里。** 反过来，trash 里 31 个 checkpoint 的 run（如 `qwen3-4b-residual-static-gate-pool-domino-shared-q-causal-block-...`），在 `outputs/` 里可能一个都不剩了。**找"某个 step 的权重"时两个地方都要看。**

### 55.3 三个存储位置的关系（扫的时候最容易踩的坑）

```
/share/guorui/投机解码训练/SpecForge/
├── outputs/                         199 个  201.19 GB   ← 现役
├── cache/checkpoint-trash/           71 个   18.08 GB   ← 本机回收站
└── outputs_del -> /data/guorui/SpecForge_outputs_del_20260728   ← 【悬空软链，目标不存在】

/data/guorui/
├── checkpoint-trash/                197 个  237.68 GB   ← 另一个盘的回收站
├── models/qwen3.5-4b/                                    ← 基座模型，不是训练产物
└── specforge-tmp/                                        ← 只有 triton/inductor 缓存，无权重
```

三个坑：

**坑 1：`outputs/` 里有 18 个悬空软链。** `find -L` 会静默跳过它们，`find` 不加 `-L` 又会对它们报错。实测：

```
symlinks OK=0 BROKEN=18
./smoke_kvbranch_20260910 -> /data/guorui/specforge-outputs/smoke_kvbranch_20260910
```

**全部 18 个都是断的**，因为软链目标是 `/data/guorui/specforge-outputs/`，而这个目录**不存在**（`ls` 报 `No such file or directory`）。所以 `outputs/` 里带软链的 run，**权重已经没了**。清单里这些 run 不会出现 —— 如果发现某个 run 在 `ls outputs/` 里看得到、清单里却没有，先 `[ -e "$d" ]` 测一下是不是悬空链。

**坑 2：两个 `checkpoint-trash` 是两棵不相交的树，别去重。** 名字一样，但：

```bash
$ stat -c '%d %i' /data/guorui/checkpoint-trash .../SpecForge/cache/checkpoint-trash
66312 330170370          # /data  盘
63 9290389181891343292   # /share 盘（NFS）
$ comm -12 t_data.txt t_cache.txt | wc -l
0
```

**`/data` 是本地盘（`/dev/md0p1`），`/share` 是 NFS（`10.97.128.245:/share`）—— 不同文件系统、不同 inode、run 名交集为 0。** 所以两边的 197 + 71 = 268 个要**全部计入**，不能因为目录名相同就以为是同一份副本去重。

**坑 3：`outputs_del` 是个死链。** 它指向 `/data/guorui/SpecForge_outputs_del_20260728`，**目标不存在**（`ls` 直接报错）。看名字是 2026-07-28 删除的一批输出，现在**已经不可恢复**。同理 `/data/guorui/specforge-outputs/` 也不存在 —— **这两个"删除归档"的位置没有任何权重**。

### 55.4 清单怎么用

**人读**：`like-useful/specforge_checkpoints.txt`（1734 行），三部分 —— 总量统计 → 按 run 名 A-Z 的逐 run 明细（每个 run 给出位置、数量、总大小、step 区间、最新写入时间，再逐条列 checkpoint）→ 按写入时间倒序的前 40 条。

**机器读**：`like-useful/specforge_checkpoints.tsv`（467 行 + 表头），列：

```
root  run  checkpoint  weights_path  size_bytes  size_gb  mtime  modified
```

`weights_path` 是**拼好的 `model.safetensors` 绝对路径**，直接拿去加载/拷贝。另外 `like-useful/specforge_checkpoints_raw.tsv` 是扫描原始记录（多两列：是否含 `dflash.py` / `config.json` / `training_state.pt`）。

常用查询：

```bash
# 某个 run 的所有 checkpoint
awk -F'\t' '$2=="<run名>"' like-useful/specforge_checkpoints.tsv

# 全部 456 GB 的权重路径列表（拿去 rsync/统计）
awk -F'\t' 'NR>1{print $4}' like-useful/specforge_checkpoints.tsv

# 只要最新的 20 个
awk -F'\t' 'NR>1{print $8"\t"$4}' like-useful/specforge_checkpoints.tsv | sort -r | head -20

# 某一步的权重（step 在目录名里，直接 grep）
grep "step_140000" like-useful/specforge_checkpoints.txt
```

### 55.5 复现扫描的命令

清单是扫出来的，不是猜的。核心一条：

```bash
find <root> -name "model.safetensors" -printf "%s\t%p\n" | sort -rn
```

三个 root 各扫一遍（`outputs`、`cache/checkpoint-trash`、`/data/guorui/checkpoint-trash`），`-printf` 同时拿大小和路径，`sort -rn` 让大文件排前面。**注意不要加 `-L`**，否则悬空软链被静默跳过、问题被掩盖；不加 `-L` 时软链本身就是个文件条目，`-name "model.safetensors"` 不会匹配到它，但**至少能看到 run 目录存在**。

### 55.6 统计画像

**按权重体积**（`model.safetensors` 单文件大小）：

| 桶 | 数量 | 典型对应 |
| --- | --- | --- |
| 1 – 1.5 GB | **323** | Qwen3-4B 类 draft 模型（含 target embedding / KV adapter 的大头） |
| < 0.5 GB | 111 | Qwen3-0.6B（`q3-06-*`）小 draft |
| 0.5 – 1 GB | 25 | 去掉词表后的瘦身版 |
| 1.5 – 2 GB | 4 | 带独立 LM head 的版本（`...-independent-lm-head-...` / `...-l2sp-lm-head-...`） |
| 2 – 3 GB | 3 | lowrank KV adapter r128 系列（`...-lowrank-kv-adapter-...-r128-...`） |
| > 3 GB | 1 | `qwen3.6-27b-domino-...`（3.63 GB，唯一的 27B） |

1–1.5 GB 占绝对多数（323/467 ≈ 69%），**说明绝大多数 run 训的是同一种体量的 Qwen3-4B draft**；体积异常的那几个（>1.5 GB）基本都是**额外挂了东西**：独立 LM head、低秩 KV adapter。查清单时体积是个好用的初筛维度。

**按基座模型**（仅 `outputs/`，199 个）：

| 前缀 | 数量 | 基座 |
| --- | --- | --- |
| `qwen3-4b-*` | 128 | Qwen3-4B |
| `q3-06-*` | 34 | Qwen3-0.6B |
| 其他 | 27 | 各类调试/临时 run |
| `qwen3.5-*` | 9 | Qwen3.5-4B |
| `qwen3.6-27b-*` | 1 | Qwen3.6-27B |

**命名里能读出实验维度**，值得记一下。典型：

```
q3-06-static-one-gate-difference-pool-domino-b8-l5-1node-011-fromscratch-bs2-globalbs16-epoch2-gamma7-eval1k-20260911
       │      │      │       │      │     │   │   │    │         │        │  │       │    │      │      └ 日期
       │      │      │       │      │     │   │   │    │         │        │  │       │    │      └ eval 每 1000 步
       │      │      │       │      │     │   │   │    │         │        │  │       │    └ γ=7
       │      │      │       │      │     │   │   │    │         │        │  │       └ 2 epochs
       │      │      │       │      │     │   │   │    │         │        └ global batch 16
       │      │      │       │      │     │   │   │    │         └ per-device bs 2
       │      │      │       │      │     │   │   │    └ 从零训（非 resume）
       │      │      │       │      │     │   │   └ 节点 011
       │      │      │       │      │     │   └ 1 node
       │      │      │       │      │     └ 第 5 层
       │      │      │       │      └ block size 8
       │      │      │       └ domino 投影器
       │      │      └ difference pool 聚合
       │      └ one gate
       └ static / 非 pre-norm
     └ Qwen3-0.6B
```

**`step` 号在同一个 run 里连续递增、跨 run 不可比** —— `epoch_1_step_140000` 是那个 run 自己的第 140000 步，不同 run 的 step 数因为 batch size / 节点数不同而不能横向比。要比进展就看 `epoch` 或百分比，别直接比 step。

### 55.7 小结

- **要权重**：`<run>/epoch_N_step_M/model.safetensors`，467 个 / 456.95 GB，清单在 `like-useful/specforge_checkpoints.txt`（人读）和 `.tsv`（机器读，含拼好的绝对路径）。
- **要接着训**：同目录 `training_state.pt`（+ `training_state_rank*.pt`）+ `rng_state_rank*.pt`。
- **要加载模型**：`config.json` + 那个 `.py`（`auto_map` 指向的 modeling 文件）与权重同目录存放，**checkpoint 自包含，不依赖 SpecForge 当时的代码版本**。
- **找历史版本去 `checkpoint-trash`**：`outputs/` 每个 run 通常只留最新 1 个，带时间戳后缀的历史存档在 `cache/checkpoint-trash/`（71 个）和 `/data/guorui/checkpoint-trash/`（197 个）。
- **`outputs/` 有 18 个悬空软链**（目标 `/data/guorui/specforge-outputs/` 不存在），这些 run 的权重**已经不在**；`outputs_del -> /data/guorui/SpecForge_outputs_del_20260728` 同样是死链。
- **两个 `checkpoint-trash` 分属不同文件系统（本地盘 vs NFS），run 名交集为 0，不要当副本去重。**
- 本次扫描时**没有训练在跑**（`ps` 无 `specforge`/`torchrun` 进程），最新 checkpoint 写于 **2026-09-12 06:04**。
