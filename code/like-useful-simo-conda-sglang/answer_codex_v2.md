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

## 34. RMSNorm：FlashInfer kernel 源码、batch invariance 与 nsys kernel 分析

分析对象实际是 nsys 脚本执行的：

```text
/share/users/like/package/sglang_kernel_src/like-useful/native.py
```

本次输入元数据为：

```text
小 batch input/residual: shape=(6, 2048), dtype=torch.bfloat16
大 batch input/residual: shape=(256, 2048), dtype=torch.bfloat16
input/residual stride:    (2048, 1), contiguous=True
weight:                   shape=(2048,), dtype=torch.bfloat16
GPU:                      NVIDIA H100 80GB HBM3, SM90
FlashInfer:               0.6.12
```

### 34.1 先给出核心结论

1. 本次 `sgl_kernel.fused_add_rmsnorm` 确实转发给 FlashInfer，并且实际执行的是 FlashInfer 0.6.12 的 CuTe DSL `FusedAddRMSNormKernel`。
2. 原脚本中 `torch.equal(input_fast, input_fast_large_slice)` 为 `False`，不是 kernel 缺少 batch invariance，而是测试代码在构造大 batch 前已经原地修改了小 batch。比较的实际是“计算一次”和“计算两次”的结果。
3. 从相同原始 prefix 分别构造小/大 batch、各调用一次后，`input` 和 `residual` 都是逐 bit 相等，`torch.equal` 均为 `True`。
4. nsys 中 `forward_native` 每次调用 11 个 GPU kernel；`fused_add_rmsnorm` 每次只调用 1 个 CuTe kernel。小/大 batch 使用完全相同的 fused kernel、block 和归约配置，只是 grid 从 2 增加到 64。

### 34.2 `sgl_kernel.fused_add_rmsnorm` 的实际调用链和 kernel 源码

本次调用链如下：

```text
sglang_kernel_src/like-useful/native.py:6,126
  -> sgl-kernel/python/sgl_kernel/__init__.py:35-47
  -> sgl-kernel/python/sgl_kernel/elementwise.py:125-162
  -> flashinfer/norm/__init__.py:238-276
  -> flashinfer/norm/kernels/fused_add_rmsnorm.py:1009-1038
  -> FusedAddRMSNormKernel.__call__ / kernel
```

各层的作用如下：

| 层 | 文件和关键行 | 行为 |
|---|---|---|
| SGL Kernel Python API | `/share/users/like/package/sglang_kernel_src/sgl-kernel/python/sgl_kernel/elementwise.py:125` | 定义 `fused_add_rmsnorm`；BF16/FP16、FlashInfer 可用且非 Dynamo tracing 时走 FlashInfer |
| FlashInfer API | `/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/flashinfer/norm/__init__.py:238` | 定义原地 API；本次 `_USE_CUDA_NORM=False`，在 `:274` 调用 CuTe 实现 |
| CuTe API/JIT cache | `/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/flashinfer/norm/kernels/fused_add_rmsnorm.py:875` 和 `:1009` | 按 dtype、hidden size、SM、PDL、contiguous 等参数编译并缓存 kernel；batch 维 `M` 是运行时参数 |
| 本次真正的 kernel | 同一文件 `:58-415` | `FusedAddRMSNormKernel` 的 CuTe DSL 源码；launch 在 `:187-216`，kernel body 在 `:218-415` |
| 行归约工具 | `/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/flashinfer/norm/utils.py:395-574` | warp、block/cluster 和多行 row-reduction 实现 |

`flashinfer.norm` 虽然写有 `@register_custom_op("flashinfer::fused_add_rmsnorm", mutates_args=...)`，但这个版本的 `flashinfer/utils.py:350-369` 因 dispatcher 开销而让装饰器直接返回 identity lambda。因此本次是普通 Python 调用 CuTe compiled callable，没有经过 `torch.ops.flashinfer` dispatcher。两次调用的编译缓存 key 都是 BF16、`H=2048`、SM90、PDL enabled、contiguous；`M=6/256` 是动态运行时参数，因此只编译一份 kernel。

核心计算位于 `fused_add_rmsnorm.py:381-411`：

```text
h = float(input) + float(residual)
residual = cast(h)                         # 原地写回
sum_sq = row_reduce_sum_multirow(h * h)
rstd = rsqrt(sum_sq / hidden_size + eps)
input = cast(h * rstd * weight)           # 原地写回
```

nsys 中的实际 kernel 名以如下字符串开头，这与上述 CuTe DSL 类名完全对应：

```text
kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnorm
FusedAddRMSNormKernel_...
```

还有两套容易混淆的 fallback 源码，但本次 report 没有执行它们：

- 设置 `FLASHINFER_USE_CUDA_NORM=1` 或 CuTe DSL 不可用时，FlashInfer CUDA JIT fallback 的 kernel 在 `/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/flashinfer/data/include/flashinfer/norm.cuh:386-514`，入口在 `flashinfer/data/csrc/norm.cu:121-149`。
- FlashInfer Python 包不可用、dtype 不支持或处于 Dynamo tracing 时，SGL Kernel 内部 fallback 入口在 `/share/users/like/package/sglang_kernel_src/sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.cu:24-58`；它再调用编译时包含的 FlashInfer `norm::FusedAddRMSNorm`。

### 34.3 为什么原脚本的 native 为 True，而 fused 为 False

#### `forward_native` 的测试输入构造正确

`native.py:91-95` 在任何 RMSNorm 调用之前，先从原始 tensor 分别构造小 batch 和大 batch：

```python
input_b_i = original_input.clone()
residual_b_i = original_residual.clone()
input_b_i_large = pad_to_need_row(input_b_i, 256)
residual_b_i_large = pad_to_need_row(residual_b_i, 256)
```

随后 `:97-105` 才对两份输入各调用一次 `forward_native`。所以两次调用的前 6 行输入完全相同。

此外，`forward_native` 在 `:68` 使用 `mean_batch_invariant`。其 Triton `mean_kernel` 位于 `batch_invariant_ops.py:438-485`：每个 program id 负责一个输出元素，固定使用 `BLOCK_SIZE=1024` 沿 hidden dimension 归约；launch 位于 `:560-576`，batch 只改变 program 数，不改变单行的归约顺序。

#### fused 测试在构造大 batch 前修改了小 batch

`fused_add_rmsnorm` 的 API 语义是原地修改两个实参：

```text
residual += input
input = rmsnorm(residual) * weight
```

但是原脚本的顺序是：

```python
# native.py:124-130，当前错误顺序
input_fast = original_input.clone()
residual_fast = original_residual.clone()
fused_add_rmsnorm(input_fast, residual_fast, weight, eps)  # 已原地修改

input_fast_large = pad_to_need_row(input_fast, 256)       # 使用第一次的输出
residual_fast_large = pad_to_need_row(residual_fast, 256) # 使用第一次的输出
fused_add_rmsnorm(input_fast_large, residual_fast_large, weight, eps)
```

因此大 batch 的前 6 行并不是第一次调用前的原始输入，而是第一次调用后的输出。第二次 fused 调用又对这些行计算一次。最后 `:135-136` 比较的是“计算一次”和“计算两次”，不是相同输入在 batch 6 与 batch 256 下的结果。

原错误测试中的差异非常大，也说明这不是浮点归约的末位误差：

| 输出 | 不同元素 | 比例 | 最大绝对差 |
|---|---:|---:|---:|
| `input` | 12,168 / 12,288 | 99.023% | 14.25 |
| `residual` | 12,279 / 12,288 | 99.927% | 35.46875 |

正确的测试顺序应当先用未修改的原始数据构造两份输入，再分别调用一次：

```python
input_fast = original_input.clone()
residual_fast = original_residual.clone()
input_fast_large = pad_to_need_row(original_input.clone(), 256)
residual_fast_large = pad_to_need_row(original_residual.clone(), 256)

fused_add_rmsnorm(input_fast, residual_fast, weight, eps)
fused_add_rmsnorm(input_fast_large, residual_fast_large, weight, eps)
```

修正输入构造后的实测结果是：

```text
torch.equal(input_fast, input_fast_large[:6])       = True
torch.equal(residual_fast, residual_fast_large[:6]) = True
input 不同元素数                                  = 0
residual 不同元素数                               = 0
```

补充扫描 `enable_pdl=False/True/None`，以及大 batch size `7, 8, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1024`，本例前 6 行也都逐 bit 相等。因此本次 `False` 应当明确归因于测试代码的原地修改，不应归因于 FlashInfer kernel。

#### 数学 batch 独立与逐 bit batch invariant 的区别

数学上逐行计算并不自动保证逐 bit batch invariant；如果 batch size 使 kernel 改变线程分工或浮点归约树，同一行仍可能出现末位差异。

本例的 CuTe fused kernel 对 hidden size 2048 固定使用：

```text
cluster_n       = 1
threads_per_row = 32
threads/block   = 128
rows/block      = 4
```

每行只在自己所在 CTA 内沿 2048 个 hidden 元素归约。batch 从 6 变为 256 时，只把 `grid.x` 从 `ceil(6/4)=2` 变为 `ceil(256/4)=64`，已有前 6 行的线程分工和归约顺序不变，所以这里能逐 bit 相等。

作为对照，若把 `forward_native` 的专用 `mean_batch_invariant` 换回普通 `torch.mean`，本样本实测前 6 行有 `1/12288` 个 BF16 输出元素不同，最大差 `0.001953125`；这才是“公式逐行独立，但归约配置随 batch 改变”导致的真实末位差异。

### 34.4 `forward_native` 小/大 batch 调用了哪些 kernel

CSV 中小 batch 对应数据行 15-25，大 batch对应数据行 26-36。每次调用都是相同顺序的 11 次 launch；下表中的 kernel 名省略了冗长的 ATen 模板参数，block 均为 `(128,1,1)`。

| 顺序/操作 | nsys kernel 名关键部分 | 小 batch grid / 时长 | 大 batch grid / 时长 |
|---:|---|---:|---:|
| 1-2. `x`、`residual` 转 FP32 | `unrolled_elementwise_kernel<...direct_copy_kernel_cuda...>`，2 次 | `24`, 2.656 / 2.240 us | `1024`, 3.680 / 2.912 us |
| 3. FP32 residual add | `vectorized_elementwise_kernel<4,...CUDAFunctor_add<float>...>` | `12`, 1.216 us | `512`, 1.920 us |
| 4. residual 转回 BF16 | `vectorized_elementwise_kernel<8,...bfloat16_copy_kernel_cuda...>` | `12`, 1.120 us | `512`, 1.632 us |
| 5. `x.pow(2)` | `vectorized_elementwise_kernel<4,...pow_tensor_scalar_kernel_impl...>` | `12`, 1.120 us | `512`, 1.632 us |
| 6. invariant mean | `mean_kernel` | `6`, 1.824 us | `256`, 1.952 us |
| 7. 加 epsilon | `vectorized_elementwise_kernel<4,...CUDAFunctorOnSelf_add<float>...>` | `1`, 1.152 us | `1`, 1.184 us |
| 8. `rsqrt` | `vectorized_elementwise_kernel<4,...rsqrt_kernel_cuda...>` | `1`, 1.248 us | `1`, 1.248 us |
| 9. `x * rstd` | `elementwise_kernel<128,2,...MulFunctor<float>...>` | `48`, 1.760 us | `2048`, 3.040 us |
| 10. 乘 weight | `elementwise_kernel<128,4,...MulFunctor<float>...>` | `24`, 3.616 us | `1024`, 4.384 us |
| 11. 输出转 BF16 | `vectorized_elementwise_kernel<8,...bfloat16_copy_kernel_cuda...>` | `12`, 1.184 us | `512`, 1.632 us |

汇总：

| 实现 | batch | 目标 kernel launch 数 | GPU kernel active time 合计 |
|---|---:|---:|---:|
| `forward_native` | 6 | 11 | 19.136 us |
| `forward_native` | 256 | 11 | 25.216 us |

大 batch 的 grid 通常按 `256/6` 的比例增加，例如 `mean_kernel` 从 6 变成 256；kernel 类型、顺序和 block 大小没有改变。

### 34.5 `fused_add_rmsnorm` 小/大 batch 调用了哪些 kernel

CSV 数据行 45 是小 batch fused 调用，数据行 50 是大 batch fused 调用。两次都只有一个完全相同的 CuTe kernel：

```text
kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnorm
FusedAddRMSNormKernel_object_at__tensorptrbf16...o20482048...
```

| batch | launch 数 | grid | block | Reg/Thread | Dynamic SMem | 时长 |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 1 | `(2,1,1)` | `(128,1,1)` | 124 | 0.033 MB | 2.272 us |
| 256 | 1 | `(64,1,1)` | `(128,1,1)` | 124 | 0.033 MB | 2.752 us |

这正好与每个 block 处理 4 行一致。动态 shared memory 的精确值是 32,784 bytes。batch size 没有触发另一个 kernel specialization，也没有改变 block、寄存器数或 shared memory；只增加了独立处理更多行的 block 数。

### 34.6 CSV 中哪些行不属于四次 RMSNorm 调用

- 数据行 9-10 是初始 tensor clone 的 D2D memcpy。
- 数据行 11-14 是为 native 大 batch 生成 padding 的 `randn` 和 `cat` kernel。
- 数据行 37-42 是 native 结果的两次 `torch.equal`，每次包含 compare、AND reduction 和 D2H copy。
- 数据行 43-44 是 fused 输入 clone 的 D2D memcpy。
- 数据行 46-49 是为 fused 大 batch 生成 padding 的 `randn` 和 `cat` kernel。
- 数据行 51-56 是 fused 结果的两次 `torch.equal`，不是额外的 fused kernel。

脚本没有 warmup、NVTX range 或 `cudaProfilerStart/Stop`，nsys 采集的是整个 Python 进程。首次 `mean_kernel` 和首次 CuTe fused kernel 前存在较长的主机/JIT 初始化空档，但它们不属于 GPU kernel active time；不能用 CSV 中“首个 kernel 到最后一个 kernel”的时间跨度代替 kernel 执行时长。

---

## 35. GSM8K `--log_samples` JSON 字段与 2638 行原因

分析文件：

```text
temp/gsm8k.nfs.2026_07_13___14_14_33/
  __data__like__hf-models__DeepSeek-V2-Lite-Chat-16B_A2.4B__/
  samples_gsm8k_2026-07-13T14-17-19.629002.jsonl
```

当前环境使用 `lm-eval 0.4.11`。该文件共有 2638 个完整 JSON object，每个 object 占一条以 LF 结尾的 JSONL 物理行；没有空行或拆成多行的 JSON object。

### 35.1 最重要的字段在哪里

对当前 GSM8K sample，可以直接按下面的路径理解：

| 内容 | JSON 路径 | 含义 |
|---|---|---|
| 数学问题 | `doc.question` | GSM8K test dataset 中当前题目的题面 |
| 标准完整答案 | `doc.answer` | 数据集给出的标准推理过程，最后用 `#### 数字` 标出标准最终答案 |
| 本次评测 target | `target` | `doc_to_target` 产生的标准答案；当前 YAML 配置下与 `doc.answer` 相同 |
| 实际发送的完整 prompt | `arguments.gen_args_0.arg_0` | 5 个 few-shot 示例，加上当前问题和末尾的 `Answer:`；其中会出现多道示例题，不要把它整体当成当前题面 |
| 生成参数 | `arguments.gen_args_0.arg_1` | `until`、`do_sample=false`、`temperature=0.0` 等 generation kwargs |
| 模型原始回答 | `resps[0][0]` | SGLang 返回的未经答案抽取的完整文本，包括模型推理过程和最终答案格式 |
| 从模型回答中抽取的最终答案 | `filtered_resps[0]` | 当前 `filter` pipeline 从原始回答提取出的字符串，真正送入 `exact_match` 的 prediction |
| 当前抽取规则 | `filter` | `strict-match` 或 `flexible-extract` |
| 使用的指标名称 | `metrics[0]` | 本文件固定为 `exact_match`；这里只是指标名称，不代表正确或错误 |
| 当前记录是否答对 | `exact_match` | `1.0` 表示该 filter 判定正确，`0.0` 表示错误 |

因此，查看一道题最常用的五个字段是：

```text
问题：         doc.question
标准最终答案： target 中最后的 "#### ..."
模型完整回答： resps[0][0]
模型抽取答案： filtered_resps[0]
正确或错误：   exact_match（1.0 / 0.0）
```

### 35.2 全部 top-level 字段解释

该文件所有记录都有相同的 12 个 top-level 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `doc_id` | integer | test split 中从 0 开始的题目编号，本文件范围是 0-1318 |
| `doc` | object | 原始 Hugging Face dataset record，包含 `question` 和 `answer` |
| `target` | string | `lm-eval` 根据 `doc_to_target: "{{answer}}"` 得到的标准 target |
| `arguments` | object | 本题构造出的模型 request 参数；`gen_args_0` 表示本题的第 0 个 request |
| `resps` | array of arrays | 每个 request 的全部原始响应；外层对应 request，内层对应 repeat。本任务每题一个 request 且 `repeats=1`，所以使用 `resps[0][0]` |
| `filtered_resps` | array | 当前 filter 处理后的响应。本任务只有一个 request，使用 `filtered_resps[0]` |
| `filter` | string | 生成当前 sample record 的答案抽取 pipeline 名称 |
| `metrics` | array | `process_results` 返回的指标名列表，本任务为 `["exact_match"]` |
| `exact_match` | number | 当前 filter 下该题的逐样本得分，只会是 `1.0` 或 `0.0` |
| `doc_hash` | string | 原始 `doc` JSON 的 SHA-256，用于审计数据是否变化 |
| `prompt_hash` | string | 实际 prompt，即 `arguments.gen_args_0.arg_0` 的 SHA-256 |
| `target_hash` | string | `target` 字符串的 SHA-256 |

三个 hash 字段不参与正确性判断，主要用于复现、去重和检查输入/target 是否发生变化。

一个记录可以简化理解为：

```json
{
  "doc_id": 0,
  "doc": {
    "question": "Janet's ducks ...?",
    "answer": "...\n#### 18"
  },
  "target": "...\n#### 18",
  "resps": [["模型的完整推理...\n#### 18"]],
  "filtered_resps": ["18"],
  "filter": "strict-match",
  "metrics": ["exact_match"],
  "exact_match": 1.0
}
```

### 35.3 标准答案和模型答案如何比较

GSM8K 配置位于：

```text
/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/
site-packages/lm_eval/tasks/gsm8k/gsm8k.yaml
```

该配置定义了两套独立的模型答案抽取方式：

#### `strict-match`

使用 YAML `:33-37` 的正则：

```regex
#### (\-?[0-9\.\,]+)
```

模型必须输出类似 `#### 18` 的格式，才能提取为 `18`。如果没有匹配，`filtered_resps[0]` 会是 `[invalid]`。

#### `flexible-extract`

使用 YAML `:38-43` 的更宽松正则，从回答中取最后一次数字/金额匹配。即使模型没有严格写出 `#### 数字`，只要回答末尾附近存在可识别的数字，仍可能抽取成功。

#### `exact_match`

`target` 原本包含完整标准推理，但 metric 配置会在 prediction 和 reference 两边删除：

```text
逗号 ,
美元符号 $
标准答案中最后一个 "#### " 之前的全部内容
末尾句点 .
```

所以标准 target `推理过程...\n#### 18` 最终按 `18` 比较。抽取后的模型答案与这个标准数字完全相同时，`exact_match=1.0`；否则是 `0.0`。

需要注意，同一份模型原始回答可能在两套 filter 下得到不同判定。`filter` 和 `exact_match` 必须一起看，不能只按 `doc_id` 找到任意一条就下结论。

### 35.4 正确、错误和格式差异示例

#### 正确示例：`doc_id=0`

问题：每天有 16 个蛋，吃掉 3 个、做松饼用 4 个，剩余蛋每个卖 2 美元。

```text
标准最终答案：18
模型原始回答结尾：#### 18
strict filtered_resps：["18"]
strict exact_match：1.0
flexible exact_match：1.0
```

模型计算 `(16 - 3 - 4) * 2 = 18`，数值和格式都正确。

#### 内容错误示例：`doc_id=5`

问题：16 个杯子，奇数位置杯子 5 美元，每第二个杯子是原价的 60%，应付多少钱。

```text
标准最终答案：8 * 5 + 8 * 3 = 64
模型最终答案：30
strict filtered_resps：["30"]
strict exact_match：0.0
flexible exact_match：0.0
```

模型错误地把“每第二个杯子打六折”理解成后续价格继续递减。这是答案内容错误，不是格式抽取问题。

#### 数值正确但 strict 格式错误：`doc_id=128`

模型的计算过程得到正确总价 880，但结尾只输出了空的 `#### `：

```text
strict filtered_resps：["[invalid]"]
strict exact_match：0.0

flexible filtered_resps：["880$."]
flexible exact_match：1.0
```

flexible pipeline 能从前面的推理文本中取到最后一个 `880$.`，metric 再删除 `$` 和末尾句点后与标准答案 `880` 相等。因此同一份原始模型回答在 strict 下错误、在 flexible 下正确。

### 35.5 为什么 1319 道题会有 2638 行

不是模型生成了 2638 道题，也不是每个 JSON object 被换行拆开。原因是 GSM8K YAML 同时定义了两个 filter pipeline：

```text
strict-match
flexible-extract
```

`lm-eval` 在生成完成后，对同一份模型原始响应分别运行两套 filter。`evaluator.py:623-666` 对每个 filter 再遍历全部 1319 个 doc，并为每个“题目 + filter”组合保存一条 sample：

```text
1319 道题 * 2 套 filter = 2638 条 JSONL record
```

具体排列为：

| JSONL 行号 | `doc_id` | `filter` |
|---:|---:|---|
| 1-1319 | 0-1318 | `strict-match` |
| 1320-2638 | 0-1318 | `flexible-extract` |

同一题的两条记录不是相邻的，而是相隔 1319 行：第 `n` 行和第 `n+1319` 行对应相同 `doc_id`。两条记录的 `doc`、`target`、`arguments`、`resps` 和三个 hash 完全相同；主要差异是 `filter`、`filtered_resps`，以及可能不同的 `exact_match`。

任务配置中的 `repeats=1`，所以这不是模型重复生成两次。2638 行不能合并作为一个分母计算官方准确率；每个 filter 单独聚合 1319 道题。

### 35.6 本次文件的正确/错误统计

| Filter | 正确 | 错误 | 准确率 | 日志显示 |
|---|---:|---:|---:|---:|
| `strict-match` | 854 | 465 | 64.7460% | 0.6475 |
| `flexible-extract` | 865 | 454 | 65.5800% | 0.6558 |

按题配对后：

- 854 题两套 filter 都判对。
- 454 题两套 filter 都判错。
- 11 题 `strict-match=0`、`flexible-extract=1`。
- 没有题目出现 `strict-match=1`、`flexible-extract=0`。

flexible 比 strict 多判对 11 题，提高约 `0.8340` 个百分点；这与日志最终打印的两行 GSM8K 得分完全一致。

字段生成逻辑位于 `lm_eval/evaluator.py:623-668`，JSONL 序列化位于 `lm_eval/loggers/evaluation_tracker.py:344-374`。writer 使用无缩进的 `json.dumps(...) + "\n"`，因此每个 sample object 写成一条 JSONL 行；字符串内部的换行保存为转义形式 `\n`，不会形成额外 sample 行。

---

## 36. `max_running_requests=146` 与 `16` 的 GSM8K 逐题对比

对比文件：

```text
旧运行（日志实际参数 max_running_requests=146）：
temp/gsm8k.nfs.2026_07_13___14_14_33/
  __data__like__hf-models__DeepSeek-V2-Lite-Chat-16B_A2.4B__/
  samples_gsm8k_2026-07-13T14-17-19.629002.jsonl

新运行（max_running_requests=16）：
temp/gsm8k.nfs.2026_07_13___14_25_02/
  __data__like__hf-models__DeepSeek-V2-Lite-Chat-16B_A2.4B__/
  samples_gsm8k_2026-07-13T14-29-17.016248.jsonl
```

虽然旧日志文件名也包含 `run-req.16`，但日志末尾打印的实际 `model_args` 明确是 `max_running_requests: 146`；本节以日志中的真实参数为准。

### 36.1 对比口径和输入一致性

两份文件都包含 2638 条合法 JSON 记录、1319 个唯一 `doc_id`，每题各有 `strict-match` 和 `flexible-extract` 两条记录。这里按 `(doc_id, filter)` 对齐，而不是直接比较两个文件的文本行。

1319 道题的以下字段全部一致：

```text
doc / question / 标准答案
target
arguments，包括完整 5-shot prompt 和 generation kwargs
doc_hash
prompt_hash
target_hash
```

因此两次运行的题目、few-shot prompt、标准答案和生成配置相同；影响执行的主要已知差异是 `max_running_requests=146` 与 `16`，它改变了 SGLang 的实际请求 batching 和调度。

### 36.2 总分变化

| Filter | 旧运行正确 | 新运行正确 | 净变化 | 旧得分 | 新得分 | 分数变化 |
|---|---:|---:|---:|---:|---:|---:|
| `strict-match` | 854 / 1319 | 860 / 1319 | +6 | 64.7460% | 65.2009% | +0.4549 pp |
| `flexible-extract` | 865 / 1319 | 871 / 1319 | +6 | 65.5800% | 66.0349% | +0.4549 pp |

日志四舍五入后分别为：

```text
max_running_requests=146: strict 0.6475, flexible 0.6558
max_running_requests=16:  strict 0.6520, flexible 0.6603
```

只看聚合分数，新运行似乎多答对 6 题。但逐题配对后可以看到，这不是“原来结果基本不变，额外答对 6 题”，而是大量改善和回退互相抵消后的净结果。

### 36.3 每个 filter 的正确/错误转移

| Filter | 两次都正确 | 两次都错误 | 旧错 -> 新对 | 旧对 -> 新错 | 发生翻转 |
|---|---:|---:|---:|---:|---:|
| `strict-match` | 785 | 390 | 75 | 69 | 144 |
| `flexible-extract` | 796 | 379 | 75 | 69 | 144 |

两套 filter 都是：

```text
75 道题从错误变正确
69 道题从正确变错误
净变化 = 75 - 69 = +6
```

单个 filter 有 `144/1319 = 10.9174%` 的题目发生正确/错误翻转。`+0.4549 pp` 的小幅净提升掩盖了明显的逐题结果变动，而且该提升小于日志给出的约 `1.3 pp` 标准误；不能据这一次运行断言 `max_running_requests=16` 系统性提高了准确率。

### 36.4 模型原始回答和抽取答案变化

每题只看一份原始 `resps`，结果如下：

| 对比项 | 相同 | 不同 | 不同比例 |
|---|---:|---:|---:|
| 模型完整原始回答 `resps[0][0]` | 410 | 909 | 68.9158% |
| strict 抽取答案 `filtered_resps[0]` | 1023 | 296 | 22.4412% |
| flexible 抽取答案 `filtered_resps[0]` | 1002 | 317 | 24.0334% |

909 道题的生成文本逐字不同。其中 152 道题至少有一个 filter 的正确/错误状态改变；另外 757 道题虽然生成文本改变，但两套 filter 的最终正确/错误状态没有改变。进一步拆分为：

```text
588 题：raw response 改变，但两套 filter 的抽取答案都不变
169 题：至少一套抽取答案改变，但正确/错误状态不变
152 题：至少一套 filter 的正确/错误状态改变
合计：909 题
```

这说明当前配置下，模型输出没有保持对 `max_running_requests` 的 output-level batch invariance。即使 `do_sample=false`、`temperature=0`，batch packing 改变后，GPU kernel 的浮点计算路径或归约顺序仍可能产生很小的 logit 差异；一旦某一步 greedy argmax 选择了不同 token，后续自回归文本就会继续分叉。

这两份结果只能证明“改变并发 batching 后输出发生变化”，不能只凭 sample 文件把原因唯一定位到 RMSNorm；要归因到某个算子，还需要逐层或逐 token 比较中间结果。

### 36.5 按题合并两套 filter 的状态

以下使用两位状态 `(strict, flexible)`：

```text
00 = 两套 filter 都判错
01 = strict 错、flexible 对
10 = strict 对、flexible 错
11 = 两套 filter 都判对
```

完整转移统计：

| 旧状态 -> 新状态 | 题数 | 含义 |
|---|---:|---|
| `00 -> 00` | 378 | 两次运行都判错 |
| `00 -> 01` | 5 | 新运行只有 flexible 判对 |
| `00 -> 10` | 1 | 新运行只有 strict 判对 |
| `00 -> 11` | 70 | 两套 filter 都从错变对 |
| `01 -> 00` | 3 | 原来 flexible 对，新运行两者都错 |
| `01 -> 01` | 4 | 两次都只有 flexible 判对 |
| `01 -> 11` | 4 | 新运行 strict 格式也正确 |
| `11 -> 00` | 66 | 两套 filter 都从对变错 |
| `11 -> 01` | 3 | 数值仍可被 flexible 判对，但 strict 回退 |
| `11 -> 11` | 785 | 两次运行都判对 |

共有 `152/1319 = 11.5239%` 的题目发生题级状态变化。其中 136 题两套 filter 同时翻转：70 题共同改善、66 题共同回退；另有 8 题只改变 strict，8 题只改变 flexible。

### 36.6 发生状态变化的完整 `doc_id`

#### 两套 filter 都从错变对：`00 -> 11`，70 题

```text
27, 28, 29, 31, 78, 90, 92, 152, 159, 162, 213, 243, 244, 256,
266, 273, 322, 323, 367, 383, 407, 417, 425, 437, 459, 471, 499,
526, 552, 583, 626, 635, 652, 654, 693, 711, 712, 725, 734, 750,
784, 785, 801, 811, 846, 852, 900, 923, 925, 971, 974, 986, 1002,
1040, 1057, 1064, 1067, 1071, 1092, 1120, 1139, 1175, 1185, 1187,
1198, 1203, 1213, 1256, 1267, 1315
```

#### 两套 filter 都从对变错：`11 -> 00`，66 题

```text
3, 21, 41, 47, 60, 66, 73, 77, 106, 132, 147, 151, 166, 167, 183,
203, 227, 234, 240, 297, 309, 314, 341, 371, 411, 434, 463, 467,
468, 474, 491, 529, 572, 623, 638, 660, 664, 722, 726, 751, 770,
803, 854, 886, 909, 941, 958, 991, 993, 996, 1026, 1045, 1046,
1053, 1055, 1118, 1131, 1144, 1152, 1244, 1247, 1255, 1265, 1295,
1310, 1314
```

#### 只有一套 filter 状态变化：16 题

```text
00 -> 01: 39, 299, 518, 692, 931
00 -> 10: 1001
01 -> 00: 576, 921, 1122
01 -> 11: 128, 829, 1208, 1317
11 -> 01: 282, 1047, 1313
```

### 36.7 翻转原因分类和代表题目

对 152 道状态变化题进一步核对标准答案、raw response 和抽取结果：

| 原因 | 题数 | 说明 |
|---|---:|---|
| 纯格式或 filter 边界 | 10 | 最终数值语义仍正确，得分变化来自 `####`、`$`、小数形式或时间字符串的抽取/精确匹配 |
| 数值、回答内容或输出截断变化 | 142 | 两次运行给出的最终数值不同，或者一边的回答在句子/算式中途结束 |

10 道纯格式/filter 题的 `doc_id` 是：

```text
128, 152, 282, 638, 829, 986, 1047, 1208, 1313, 1317
```

其中：

- `152`、`638`、`986` 是 `4`/`4.00`、`12`/`12.00`、`70`/`70.00` 等字符串 exact-match 差异。当前 metric 会删除 `$`、逗号和末尾句点，但不会把数值字符串统一转换成相同数值类型。
- `128`、`282`、`829`、`1047`、`1208`、`1313`、`1317` 是 `####` marker 缺失、marker 后出现 `$`，或只写 `The answer is: ...` 等格式差异。

其余 142 题属于最终数值、回答内容或截断变化：按标准答案的数值语义核对，74 题从错误内容变为正确内容，68 题从正确内容变为错误内容，净值仍为 `+6`。这里也不能把全部变化都称为“算术推理改变”：`299`、`459`、`474`、`518`、`576`、`693`、`931`、`1122`、`1295` 等题至少有一边的 raw response 明显结束在词句或算式中途。

#### 内容改善：`doc_id=27`，`00 -> 11`

问题是 60 天每天吃一份冰淇淋，每盒 15 份、每盒 4 美元，标准答案为 16 美元。

```text
max_running_requests=146:
  错误地认为只需 2 盒，输出 #### 8

max_running_requests=16:
  正确计算 60 / 15 = 4 盒，4 * 4 = 16，输出 #### 16
```

这是模型推理内容从错误变正确，不是 filter 格式差异。

#### 内容回退：`doc_id=3`，`11 -> 00`

问题是每次跑 3 个 60 米冲刺、每周跑 3 次，标准答案为 `3 * 60 * 3 = 540` 米。

```text
max_running_requests=146: 输出 #### 540，正确
max_running_requests=16:  只计算 60 * 3，输出 #### 180，错误
```

这是新运行漏掉“每次 3 个冲刺”，属于真实推理回退。

#### 数值改善但缺少 strict 格式：`doc_id=39`，`00 -> 01`

旧运行计算为 36，错误。新运行正确计算为 18，但回答末尾没有 `#### 18`：

```text
新 strict filtered_resps = ["[invalid]"]，exact_match=0
新 flexible filtered_resps = ["18"]，exact_match=1
```

#### strict 格式改善：`doc_id=128`，`01 -> 11`

两次运行的数学答案都是正确的 880：

```text
旧运行结尾只有空的 ####
  strict=[invalid]，flexible=880

新运行结尾为 #### 880
  strict=880，flexible=880
```

这里新增的 strict 正确不是数学能力变化，而是输出格式改善。

#### strict 格式回退：`doc_id=282`，`11 -> 01`

两次运行都算出正确答案 195，但新运行写成：

```text
#### $195
```

strict 正则要求 `#### ` 后立即出现数字，因此得到 `[invalid]`；flexible 仍提取 `$195` 并判对。这是格式回退，不是答案数值回退。

#### 两个 filter 的边界差异：`doc_id=1001`，`00 -> 10`

新运行的标准答案和模型答案都是下午 2 点，模型结尾为：

```text
#### 2:00 pm
```

strict 提取 `2`，与归一化后的标准答案相符，判为正确；flexible 取最后一个数字匹配 `00`，反而判错。这也说明 flexible 并不保证在每种输出格式下都优于 strict。

### 36.8 总结

将 `max_running_requests` 从 146 改为 16 后，两套官方分数都净增 6 题、约 `+0.4549 pp`。但逐题结果变化远大于净分数：909 题的完整生成文本不同，每套 filter 各有 75 题改善和 69 题回退，152 题至少有一个 filter 的正确/错误状态变化。

因此，更准确的结论不是“16 比 146 稳定地多答对 6 题”，而是“并发 batch 改变导致大量生成路径重新分叉，改善和回退大致抵消，本次恰好净增 6 题”。

---

## 37. `self._kv_buffer_descs = self._build_kv_buffer_descs()` 的作用

代码位于：

```text
simo/extensions/sglang_simo/mem_cache/memory_pool.py:209-210
```

先给出结论：这行代码不会创建 KV cache tensor，不会量化 K/V，也不会参与 attention kernel。它读取已经创建好的每层 K/V tensor，为它们生成一组 Python 侧的“buffer 字节布局描述符”，供新版 SGLang 计算每个 buffer 的总字节范围、每页传输字节数，以及 PD disaggregation 注册/传输 KV cache 时使用。

### 37.1 为什么 SIMO 需要补这一行

`SIMOMHATokenToKVPool` 继承自 SGLang 的 `MHATokenToKVPool`，但覆盖了 `_create_buffers()`，自己分配量化后的 uint8 K/V buffer：

```text
K: [size + page_size, head_num, k_packed_head_size + k_scale_head_size] uint8
V: [size + page_size, head_num, v_packed_head_size + v_scale_head_size] uint8
```

Python 调用父类 `MHATokenToKVPool.__init__()` 时，父类内部执行 `self._create_buffers()`。因为 `self` 实际是 `SIMOMHATokenToKVPool`，这里会动态分派到 SIMO 的 override，而不会进入父类自己的 `_create_buffers()`。

新版 SGLang 父类的正常 `_create_buffers()` 在完成 tensor 分配后会执行：

```python
self._kv_buffer_descs = self._build_kv_buffer_descs()
```

对应上游代码为：

```text
/share/users/like/package/sglang_kernel_src/python/sglang/srt/
mem_cache/memory_pool.py:1500-1523
```

SIMO 覆盖整个方法后也绕过了这一步。如果不在 SIMO override 中补回来，新版父类的其他继承方法会假定 `_kv_buffer_descs` 已存在，但实际没有初始化。

这段适配是在 SIMO 从旧 SGLang 升级到新 main 时加入的。新版 descriptor 机制来自 SGLang commit `2ad9a243f`（`Size KV pool after CUDA graph capture`）。旧版 SGLang 没有 `_build_kv_buffer_descs()`，所以 SIMO 使用兼容写法：

```python
if hasattr(self, "_build_kv_buffer_descs"):
    self._kv_buffer_descs = self._build_kv_buffer_descs()
```

- 新版 SGLang：继承方法存在，构建 descriptors。
- 旧版 SGLang：方法不存在，跳过，不会因兼容代码报错。

### 37.2 `_build_kv_buffer_descs()` 返回什么

实现位于 SGLang `memory_pool.py:1593-1624`。它返回 `list[KvBufferDesc]`，顺序严格为：

```text
k0, k1, ..., k(layer_num - 1),
v0, v1, ..., v(layer_num - 1)
```

因此 descriptor 数量是：

```text
2 * layer_num
```

每个 `KvBufferDesc` 包含四个字段：

| 字段 | 作用 |
|---|---|
| `name` | buffer 名称，例如 `k0`、`k1`、`v0` |
| `shape` | 对应真实 tensor 的完整 shape |
| `row_bytes` | tensor 第一维中一行占多少字节 |
| `tokens_per_row` | 一行表示多少 token；普通 NHD 为 1，page-major/HND 可为 `page_size` |

`KvBufferDesc` 类定义在 SGLang `memory_pool.py:1197-1229`。它基于以上字段提供：

| 方法 | 计算内容 |
|---|---|
| `reserved_span_bytes()` | 整个预留 tensor 的最大字节数 |
| `prefix_span_bytes(num_tokens)` | 让前 `num_tokens` 可用所需 backing 字节数 |
| `final_span_bytes(num_tokens, page_size)` | 当前 serving token 数加 padded page 后应暴露/注册的字节范围 |
| `item_len_bytes(page_size)` | 一页 KV cache 的传输 chunk 大小 |

### 37.3 对 SIMO uint8 量化 buffer，具体生成什么

SIMO 只支持普通 3-D NHD layout，所以一行就是一个 token slot：

```text
tokens_per_row = 1
```

SIMO 在调用 `_build_kv_buffer_descs()` 之前先执行：

```python
self.store_dtype = torch.uint8
```

因此 `itemsize=1`，descriptor 会从实际 tensor 得到：

```text
K shape = (size + page_size, head_num, k_combined_head_size)
V shape = (size + page_size, head_num, v_combined_head_size)

K row_bytes = head_num * k_combined_head_size
V row_bytes = head_num * v_combined_head_size
```

其中：

```text
k_combined_head_size = k_packed_head_size + k_scale_head_size
v_combined_head_size = v_packed_head_size + v_scale_head_size
```

descriptor 不再区分 packed data 和 scale data，而是把最后一维 `[packed bytes | scale bytes]` 当成一段连续、不可解释的 uint8 数据。量化格式及 packed/scale 的内部解释仍由 SIMO 写 cache kernel 和 attention kernel负责。

对于普通 NHD buffer：

```text
K final span = (size + page_size) * K row_bytes
V final span = (size + page_size) * V row_bytes

K page transfer bytes = page_size * K row_bytes
V page transfer bytes = page_size * V row_bytes
```

调用顺序非常重要。如果仍使用父类根据计算 dtype 设置的 `store_dtype`，例如 BF16 的 `itemsize=2`，descriptor 会把实际 uint8 buffer 的字节长度错误地放大两倍。现在先设置 `store_dtype=torch.uint8`，再构建 descriptors，得到的字节数才与真实 tensor 一致。

### 37.4 descriptors 在哪里被使用

父类的 `get_contiguous_buf_infos()` 位于 SGLang `memory_pool.py:1683-1697`，返回：

```text
ptrs:      每层 K/V tensor 的 data_ptr
lens:      每个 tensor 当前应注册的总字节范围
item_lens: 每个 tensor 中一页 KV cache 的字节数
```

其中 `lens` 和 `item_lens` 正是从 `self._kv_buffer_descs` 计算出来的。顺序必须与 `_pd_registerable_tensors()` 返回的 `k_buffer + v_buffer` 完全一致。

Prefill/Decode disaggregation 初始化会调用它：

```text
sglang/srt/disaggregation/prefill.py:160-176
sglang/srt/disaggregation/decode.py:405-437
```

然后将这些信息写入：

```text
kv_args.kv_data_ptrs
kv_args.kv_data_lens
kv_args.kv_item_lens
```

NIXL、Mooncake、MORI 等传输 backend 使用这些数据注册 GPU 内存，并按 token/page 计算 K/V 的地址和传输长度。

SGLang 通用实现还会用同一套 descriptors 驱动 CUDA VMM 的 post-capture KV backing：先预留较大的虚拟地址范围，CUDA Graph capture 后再按最终 token capacity backing 必要的物理显存。不过当前 SIMO 构造函数在 `memory_pool.py:93-94` 明确拒绝 `post_capture_active=True`，所以 SIMO 当前实际不会走这条 VMM 路径；对 SIMO 来说，保存 `_kv_buffer_descs` 的直接用途主要是继承的 PD KV transfer 接口。

如果当前运行没有启用 disaggregation，这组 metadata 通常不会进入普通 attention 的 K/V 写入和读取路径，因而不会改变正常单机推理结果。

### 37.5 与旁边几个字段的区别

SIMO `_create_buffers()` 后面还构建了：

```python
self.k_data_ptrs
self.v_data_ptrs
self.data_ptrs
self.data_strides
```

它们与 `_kv_buffer_descs` 的职责不同：

| 字段 | 位置/用途 |
|---|---|
| `k_buffer` / `v_buffer` | 真正保存量化 KV cache 的 GPU tensor |
| `data_ptrs` | GPU tensor 地址组成的 device tensor，供 CUDA/Triton copy kernel 使用 |
| `data_strides` | 每个 token slot 的字节 stride，供 GPU copy kernel 做地址计算 |
| `_kv_buffer_descs` | Python object 列表，描述完整 shape、行字节数和每页字节数，主要供 VMM/PD registration 与 transfer 使用 |

所以，这行代码的准确含义是：

```text
“SIMO 已经按自定义量化格式创建完每层 uint8 K/V buffer；
 现在为新版 SGLang 补齐与这些真实 buffer 匹配的字节布局元数据。”
```

### 37.6 如果删除会怎样

- 普通、不启用 disaggregation 的推理可能仍能运行，因为 attention 直接读取 `k_buffer`/`v_buffer`，不会读取 descriptors。
- 一旦调用继承的 `get_contiguous_buf_infos()`，会因 `self._kv_buffer_descs` 不存在而出现 `AttributeError`。
- 如果 descriptor 使用错误的 dtype、shape 或顺序，即使不立即报错，也会产生错误的注册长度、page stride 或传输 offset，导致 PD KV cache 少传、多传或地址错位。
- 对支持 post-capture VMM 的其他 SGLang pool，缺少或错误的 descriptor 还会导致虚拟地址 backing 范围错误；SIMO 当前已显式禁用该模式。

## 38. ONNX sm90 custom-op 动态库没有出现在源码目录及 pytest 失败原因

分析对象：

```text
代码：/share/users/like/package/simo_conda_sglang
环境：/share_data/users/like/miniconda3/envs/simo_sglang
安装日志：temp/dev.log
失败日志：temp/test_dynamic_qdq_runtime_debug.py.log.2026_07_14___16_11_58
```

### 38.1 结论

这次有三个需要分开的结论：

1. `pip install -e ".[dev]"` 不会编译 `libSimoOnnxCustomOps_sm90.so`，这是当前源码的预期设计，不是安装失败。安装阶段只构建 `simo._C`。
2. `libSimoOnnxCustomOps_sm90.so` 实际已经由 `runtime.py` 按需编译成功，只是产物位于用户 cache，而不在 `simo/onnx/` 源码目录。
3. pytest 的 21 个失败与“动态库不存在”无关。直接原因是 debug 测试创建的 ONNX custom-op 节点漏了必需属性 `narrow_range`；C++ 插件在 shape inference 中读取该属性时，ORT 报 `Attribute does not exist`。

本次实际找到的库为：

```text
/softhome/like/.cache/simo/onnx/
  sm90-py312-ort1.27.0-cuda13.0-08d729191a9f/
  libSimoOnnxCustomOps_sm90.so
```

文件大小为 `2408096` bytes，mtime 为 `2026-07-14 15:37:33 +0800`。调用公开解析函数得到的也是同一路径：

```text
get_custom_ops_library_path()
=> /softhome/like/.cache/simo/onnx/sm90-py312-ort1.27.0-cuda13.0-08d729191a9f/libSimoOnnxCustomOps_sm90.so
```

### 38.2 为什么 editable install 没有在包目录生成该 `.so`

`setup.py:82-90` 的 `ext_modules` 只声明了一个 extension：

```python
extension(
  "simo._C",
  sources,
  ...,
)
```

`setup.py:99-107` 的 `package_data` 也只包含 ONNX 插件的 `*.cc`、`*.h`、`*.lds` 和 ORT headers，没有包含或声明 `libSimoOnnxCustomOps_sm90.so`。

因此 `temp/dev.log:96-103` 中只有 editable wheel 的成功构建和安装：

```text
Building editable for simo ... done
Successfully built simo
Successfully installed simo-0.6.1.dev20260714+b229221
```

日志中没有 ONNX plugin build 命令是正常的。`--no-build-isolation` 只决定 pip 是否复用当前环境的构建依赖，不会把一个没有列入 `ext_modules` 的库额外编译出来。

另外，`[dev]` 只安装 `ruff`、`pytest`、`pytest-cov`。ONNX 相关依赖位于独立的 `[onnx]` extra。全新环境中运行 ONNX 测试应安装：

```bash
pip install -e ".[dev,onnx]" --no-build-isolation
```

当前环境已经有 `onnxruntime-gpu==1.27.0`，所以这不是本次失败原因。

### 38.3 `.so` 是在什么时候、怎样生成的

`simo/onnx/runtime.py:108-131` 的流程是：

```text
确认当前 GPU 是 sm90
  -> 查找包内预编译库
  -> 查找版本化 cache
  -> 两处都没有时调用 build_sm90_runtime(cache_path)
  -> 用文件锁避免多个进程同时构建
```

包内候选位置是：

```text
simo/onnx/ort_plugin/libSimoOnnxCustomOps_sm90.so
simo/onnx/libSimoOnnxCustomOps_sm90.so
```

editable source tree 中这两个位置都不存在，随后代码使用 cache。cache key 由以下内容组成：

```text
sm90 + Python 版本 + ORT 版本 + CUDA 版本 + 插件/内核源码 hash
```

`simo/onnx/ort_plugin/build_runtime.py:47-91` 先调用 `build_qdq_cubins.build(..., 90)` 生成嵌入式 sm90 cubin C++ 源码，再通过 `torch.utils.cpp_extension.load()` 编译：

```text
custom_op_library.cc
simo_qdq_ops.cc
triton_loader.cc
generated/embedded_qdq_kernels_sm90.cc
```

torch extension 的中间产物名是：

```text
build/SimoOnnxCustomOps_sm90.so
```

最后 `shutil.copy2()` 将它复制为 runtime 需要的：

```text
libSimoOnnxCustomOps_sm90.so
```

所以只在源码目录执行 `find simo -name libSimoOnnxCustomOps_sm90.so` 会误判为“没有编译”。应检查解析结果或 cache：

```bash
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
from simo.onnx.runtime import get_custom_ops_library_path
print(get_custom_ops_library_path())
PY
```

如需在不删除现有 cache 的情况下验证一次全新构建，可使用新的 cache 根目录：

```bash
SIMO_ONNX_RUNTIME_CACHE="$PWD/temp/simo-onnx-runtime-clean" \
SIMO_ONNX_RUNTIME_BUILD_VERBOSE=1 \
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
from simo.onnx.runtime import get_custom_ops_library_path
print(get_custom_ops_library_path())
PY
```

该 v1 runtime 明确只支持 compute capability `(9, 0)`。当前设备是 `NVIDIA H100 80GB HBM3`，capability 为 `(9, 0)`，满足构建和运行条件。

### 38.4 pytest 的直接失败原因

原日志结果为：

```text
21 failed, 4 passed, 14 warnings in 10.13s
```

21 个失败虽然覆盖多种 dtype/granularity，但异常完全相同：

```text
Load model ... failed:
Node () Op (Quantize) Attribute does not exist.
```

最后一个纯 Dequantize 用例对应：

```text
Node () Op (Dequantize) Attribute does not exist.
```

失败发生在 `ort.InferenceSession(...)` 创建阶段，即 custom-op shape inference 阶段，还没有运行 Triton/CUDA 数值 kernel。

`simo/onnx/ort_plugin/simo_qdq_ops.cc:58-75` 对每个 custom op 都无条件读取六个语义属性：

```cpp
dtype
granularity
scale_mode
group_size
block_size
narrow_range
```

但 `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py` 原来的 `_semantic_attrs()` 只返回：

```python
scale_mode
observer_mode
group_size
block_size
```

各模型构造函数另外显式传入了 `dtype` 和 `granularity`，唯独没有任何地方传 `narrow_range`。因此下面这行读取失败：

```cpp
const auto narrow_range = ctx.GetAttrInt("narrow_range") != 0;
```

这也解释了为什么三个通过真实 SIMO 导出路径创建图的 runtime 用例能够通过：生产代码 `simo/onnx/onnx_quant.py:1040-1084` 的 `_qdq_attrs()` 本来就会写入：

```python
"narrow_range": int(bool(normalized.get("narrow_range", True)))
```

问题是 debug 测试 helper 在最近的 semantic-attribute 重构中没有同步完整属性契约，不是 production exporter 漏属性。

### 38.5 是否与 `libSimoOnnxCustomOps_sm90.so` 有关

与该库的“代码路径”有关，但与该库“未生成/未加载”无关：

- 抛错代码确实在该 `.so` 内，是插件读取 ONNX 节点属性时报出的。
- `.so` 已存在、已被 ORT 注册，ORT 才会进入 SIMO custom-op 的 shape inference。
- 若默认构建路径抛 `RuntimeError`，测试 helper 会在 `test_dynamic_qdq_runtime_debug.py:68-71` 将用例标记为 skip，而不是产生当前 21 个 FAIL。
- 若文件路径不存在，测试也会在 `:73-74` skip。
- 当前异常是 ORT 已识别 `com.simo::Quantize/Dequantize` 后的属性错误，因此不能通过重新执行 `pip install -e` 修复。

### 38.6 ORT API 28/27 提示是不是根因

原失败日志还反复出现：

```text
The requested API version [28] is not available, only API versions [1, 27]
are supported in this build. Current ORT Version is: 1.27.0
```

原因是仓库 vendored 的 `onnxruntime_c_api.h` 定义 `ORT_API_VERSION 28`，而环境安装的是 ORT `1.27.0`。不过 `custom_op_library.cc:23-45` 已实现向下回退：先请求 28，失败后循环请求 27，成功后用 27 注册 custom ops。

所以该提示是版本探测产生的 stderr 噪声，不是本次 `Attribute does not exist` 的直接原因。最直接的证明是补齐 `narrow_range` 后，同一个 ORT 1.27.0、同一个缓存 `.so` 已通过全部 25 个测试。

它仍然提示依赖契约不够清晰。长期可以二选一：

- 将 `onnxruntime-gpu` 依赖锁定到与 vendored API 28 headers 一致的 runtime；
- 或 vendor 与最低受支持 ORT 一致的 headers，并保留有测试覆盖的版本回退。

这属于后续的版本治理，不需要作为这次 pytest 修复的前置条件。

### 38.7 已实施的修复

在 `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py` 的 `_semantic_attrs()` 中补上与 production `_qdq_attrs()` 相同的默认属性：

```diff
 return {
   "scale_mode": getattr(spec, "scale_mode", "fp32"),
   "observer_mode": getattr(spec, "observer_mode", "abs_max"),
+  "narrow_range": int(bool(getattr(spec, "narrow_range", True))),
   "group_size": int(getattr(spec, "group_size", None) or 1),
   "block_size": int(getattr(spec, "block_size", None) or 1),
 }
```

这里选择修测试 helper，而不是让 C++ 对缺失属性静默使用默认值，原因是：

- production exporter 已明确输出完整、自描述的 semantic attrs；
- runtime resolver 需要 `narrow_range` 区分 int8 full range 与 narrow range 的不同 cubin；
- 测试应该构造与生产导出一致的 ONNX contract。

### 38.8 修复验证

先运行原先失败的 7 个 MX 参数化用例：

```text
7 passed, 14 warnings in 10.54s
```

再运行完整文件并保留仓库默认 coverage 参数：

```bash
/share_data/users/like/miniconda3/envs/simo_sglang/bin/pytest \
  simo/onnx/tests/test_dynamic_qdq_runtime_debug.py
```

结果：

```text
25 passed, 14 warnings in 10.82s
```

修复后的完整日志保存于：

```text
temp/test_dynamic_qdq_runtime_debug.py.log.codex_fixed
```

14 个 warning 全部是 `torch.jit.script_method` 的弃用提示，与 ONNX custom-op 失败无关。

## 39. `kws_simo_quant/test_quant_onnx.sh` 的完整执行流程

分析对象：

```text
Shell 入口：/share/users/like/package/jdjv/kws_simo_quant/test_quant_onnx.sh
Python 调度器：/share/users/like/package/jdjv/kws_simo_quant/scripts/run_kws_onnx_float_sharded.py
单 shard 评测：/share/users/like/package/jdjv/kws_simo_quant/scripts/evaluate_kws_onnx_float_manifest.py
SIMO 环境：/share_data/users/like/miniconda3/envs/simo_sglang
SIMO 源码：/share/users/like/package/simo_conda_sglang
```

### 39.1 一句话概括

该脚本不是简单地对一个 ONNX 文件跑一次量化。它是一个 KWS 量化精度批量实验入口：

```text
11 套量化配置依次执行
  -> 每套配置重写 encoder/decoder/joiner 三个浮点 ONNX
  -> 注册 SIMO sm90 ONNX custom-op 库
  -> 将 31562 条 KWS manifest 数据轮转分成 32 个 shard
  -> 最多启动 8 个子进程，在 GPU 0-7 上做 ONNXRuntime 推理
  -> 合并逐条检测结果
  -> 计算 precision/recall/F1/false alarm 等全局指标
```

默认完整运行会做：

```text
11 * 3 = 33 次 ONNX 模型改写
11 * 32 = 352 个 shard 评测子进程
31562 * 11 = 347182 条 utterance 评测
```

11 套配置之间是串行的；同一套配置内部的 32 个 shard 才会并发。

### 39.2 它不会自动进入指定 conda 环境

`test_quant_onnx.sh:7` 是：

```bash
PY=${PY:-python}
```

这表示它直接使用当前 `PATH` 中的 `python`，不会执行 `conda activate`，也没有硬编码用户给出的环境。要保证使用：

```text
/share_data/users/like/miniconda3/envs/simo_sglang
```

建议显式运行：

```bash
cd /share/users/like/package/jdjv/kws_simo_quant
PY=/share_data/users/like/miniconda3/envs/simo_sglang/bin/python \
  bash test_quant_onnx.sh
```

该环境中的 SIMO 应是对以下源码的 editable install：

```text
/share/users/like/package/simo_conda_sglang
```

脚本本身不设置 `PYTHONPATH`，所以实际导入哪份 `simo` 取决于 `PY` 对应环境中安装的包。

脚本也不会自动 `cd` 到自身目录。第 4 行只是注释，不会执行。因为 `SCRIPT`、`MODEL_DIR` 和默认输出目录都是相对路径，所以从其他目录直接执行绝对路径很可能在第 34 行的：

```bash
test -d "$MODEL_DIR"
```

处直接退出。应先进入 `kws_simo_quant` 根目录，或把相关变量全部设为绝对路径。

### 39.3 Shell 安全选项

开头的：

```bash
set -euo pipefail
```

含义是：

- `-e`：目录检查、配置检查或任意一套 Python 评测失败后，停止后续配置。
- `-u`：读取未定义变量时报错；脚本对可选变量基本都使用了默认值保护。
- `pipefail`：管道中任意命令失败都视为失败；当前脚本没有复杂管道，但这是防御性设置。

因此它不是“某套失败后继续跑剩余配置”的容错批处理。

### 39.4 默认参数

Shell 层默认值如下：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `PY` | `python` | Python 解释器 |
| `SCRIPT` | `scripts/run_kws_onnx_float_sharded.py` | Python 调度入口 |
| `MODEL_DIR` | `onnx_float_baseline` | 浮点 KWS ONNX 目录 |
| `DEVICES` | `0,1,2,3,4,5,6,7` | 可用的 8 张物理 GPU |
| `NUM_SHARDS` | `32` | manifest 分片数 |
| `WORKERS` | `8` | 同时运行的 shard 子进程上限 |
| `PROVIDER` | `CUDAExecutionProvider` | ONNX Runtime execution provider |
| `CUDA_USE_TF32` | `0` | 默认关闭 CUDA EP 的 TF32 |
| `FORCE` | `--no-skip-existing` | 默认无视已有结果并重跑 shard |
| `CONFIG_DIR` | `/share_data/mtang/simo_quant_config` | 量化配置根目录 |
| `OUTPUT_ROOT` | `results_0708` | 多配置模式的输出根目录 |
| `CUSTOM_OP_LIBRARY` | 空 | 让 Python/SIMO 自动解析 custom-op 库 |

Python 调度器还有一组 shell 没有显式传入的默认值：

```text
num_threads=1
tail_padding=0.66 秒
batch_size=64
max_active_paths=4
keywords_score=1.0
keywords_threshold=0.25
num_trailing_blanks=1
```

设置同名环境变量不会改变这些 Python 默认值，因为 shell 没有读取并转发它们。若要修改，应直接调用 Python 入口并传对应命令行参数，或扩展 shell 脚本。

### 39.5 默认会跑哪 11 套配置

`DEFAULT_CONFIGS` 的每个元素采用：

```text
相对 JSON 路径|输出目录短名称
```

格式。默认配置为：

| 输出短名称 | 主要量化格式 |
|---|---|
| `w4a4_mxfp4_e2m1` | weight/activation 都使用 MXFP4 E2M1，`e8m0_sipu` scale |
| `w6a6_mxfp6_e2m3` | weight/activation 都使用 MXFP6 E2M3 |
| `w6a6_mxfp6_e3m2` | weight/activation 都使用 MXFP6 E3M2 |
| `w8a8_cnn_wpc_apt_transfomer_wpc_apc` | int8；CNN 与 Linear 使用配置中指定的 per-tensor/per-channel 组合 |
| `w8a8_fp8_block` | FP8 E4M3；activation 一维 group，weight 二维 block，group size 128 |
| `w8a8_int8_pc_pc` | int8 weight per-channel、activation per-channel |
| `w8a8_int8_pc_pt` | int8 weight per-channel、activation per-tensor |
| `w8a8_int8_pt_pt` | int8 weight/activation 都是 per-tensor |
| `w8a8_mxfp8_e4m3` | MXFP8 E4M3 |
| `w8a8_mxfp8_e5m2` | MXFP8 E5M2 |
| `w8a8_mxint8` | MXINT8 |

这里以 shell 数组为准，实际是 11 套；项目 README 中“All ten SIMO configs”的描述已经过期。

配置中的 `targets` 使用 PyTorch 模块名，例如 `Conv2d`、`Conv3d`、`Linear`。Python 调度器会在量化前补充 ONNX op 映射：

```text
Conv1d/Conv2d/Conv3d -> Conv
Linear               -> MatMul, Gemm
```

并把规范化配置写到每套结果目录的 `onnx_quant_config.json`。原始 `targets` 和 `excludes` 仍会保留。

### 39.6 单配置和多配置模式

如果没有设置 `QUANT_CONFIG`，脚本遍历全部 11 项。以第一项为例，输出目录为：

```text
results_0708/onnx_quant_w4a4_mxfp4_e2m1_32shards
```

通用命名规则是：

```text
$OUTPUT_ROOT/onnx_quant_<短名称>_${NUM_SHARDS}shards
```

如果设置了 `QUANT_CONFIG`，则只运行一套：

```bash
QUANT_CONFIG=w8a8/w_mxint8_a_mxint8.json \
OUTPUT_DIR=results/mxint8_test \
PY=/share_data/users/like/miniconda3/envs/simo_sglang/bin/python \
bash test_quant_onnx.sh
```

相对配置路径会拼到 `CONFIG_DIR` 下；以 `/` 开头的绝对路径直接使用。单配置模式的默认输出不是 `OUTPUT_ROOT`，而是：

```text
results/onnx_quant_32shards
```

### 39.7 每套配置首先怎样量化三个 ONNX

Shell 的 `run_one_config()` 调用 Python 后，`prepare_model()` 先完成模型改写，再启动任何 shard。

输入目录必须包含：

```text
encoder-epoch-13-avg-2-chunk-16-left-64.onnx
decoder-epoch-13-avg-2-chunk-16-left-64.onnx
joiner-epoch-13-avg-2-chunk-16-left-64.onnx
tokens.txt
```

对每个 ONNX，代码调用：

```python
rewritten = insert_qdq_nodes(source_model, config_path)
onnx.save(rewritten, quantized_model_path)
```

SIMO 不是把整个模型替换成一个全新的推理引擎，而是在原始 `Conv`、`MatMul`、`Gemm` 周围插入 `com.simo` domain 的 QDQ custom ops：

```text
浮点 activation -> Quantize -> uint8 packed data + scale -> Dequantize -> 原始计算 op
离线量化 weight initializer + scale -> Dequantize -----------------------> 原始计算 op
```

所以一个完整插入点通常对应 3 个 SIMO 节点：1 个 activation `Quantize` 和 2 个 `Dequantize`（activation、weight 各一个）。当前 W4A4 产物实测为：

```text
encoder: Quantize=140, Dequantize=280
decoder: Quantize=2,   Dequantize=4
joiner:  Quantize=1,   Dequantize=2
```

日志中 encoder 的 `targets=170, inserted=140, skipped=30` 表示识别出 170 个候选计算节点，其中 30 个 MatMul 的 weight 是动态输入而非静态 initializer，因 `dynamic_weight` 被跳过；不是 30 个 shard 失败。

量化模型和 `tokens.txt` 最后保存到：

```text
<output_dir>/quantized_model_dir/
```

### 39.8 custom-op 动态库如何取得和使用

当 shell 没有设置 `CUSTOM_OP_LIBRARY` 时，Python 按以下顺序解析：

```text
--custom-op-library 参数
  -> SIMO_ONNX_CUSTOM_OPS_LIBRARY 环境变量
  -> simo.onnx.get_custom_ops_library_path()
```

最后一条会使用或按需构建上一节解释的：

```text
libSimoOnnxCustomOps_sm90.so
```

当前 `simo_sglang` 环境解析到的路径是：

```text
/softhome/like/.cache/simo/onnx/
sm90-py312-ort1.27.0-cuda13.0-08d729191a9f/
libSimoOnnxCustomOps_sm90.so
```

父调度器把该路径传给所有 shard 的 `--custom-op-lib`。KWS runtime 创建一个 `OrtSessionOptions`，调用：

```python
so.register_custom_ops_library(str(custom_op_lib))
```

然后用同一组选项创建 encoder、decoder、joiner 三个 ORT session。没有该库，ORT 无法执行插入的 `com.simo::Quantize/Dequantize`。

只要使用量化配置，即使 `CUSTOM_OP_LIBRARY` 为空也不代表“不使用 custom ops”；它表示让 SIMO 自动找到或构建库。

### 39.9 manifest 如何切成 32 个 shard

默认 manifest 是：

```text
/share_data/mtang/work/JD/test/kws/data/open_commands_bilingual_kws_eval/small/
manifest_with_filtered_long_negatives.csv
```

该文件当前有 31562 条数据行。每条至少包含：

```text
id, group, label, expected_keyword, audio_path
```

还可包含 `begin_time/end_time`，用于只读取一个音频片段。

分片不是把文件切成 32 个连续区间，而是：

```python
rows = rows[shard_index::num_shards]
```

即 shard 0 处理原始行 `0, 32, 64, ...`，shard 1 处理 `1, 33, 65, ...`。31562 条数据分成 32 份后，前 10 个 shard 各有 987 条，其余 22 个各有 986 条。

若设置 `MAX_ITEMS=N`，先截取 manifest 的前 N 条，再分 shard；它表示所有 shard 合计最多 N 条，不是每个 shard 各 N 条。

### 39.10 GPU 和 worker 调度

父进程构建 32 个 task，GPU 采用：

```python
device = devices[shard_index % len(devices)]
```

所以静态映射为：

```text
shard 0,8,16,24  -> GPU 0
shard 1,9,17,25  -> GPU 1
...
shard 7,15,23,31 -> GPU 7
```

每个 task 启动独立 Python 子进程，并设置：

```bash
CUDA_VISIBLE_DEVICES=<该 task 的 device>
```

子进程内通常只看到一张逻辑 GPU 0。父进程使用 `ThreadPoolExecutor(max_workers=8)` 同时管理最多 8 个子进程。

需要注意：该实现只限制“总并发数为 8”，没有按 GPU 建立独立队列或锁。开始时前 8 个 task 恰好一张 GPU 一个；之后任意 worker 完成都会领取下一个 shard。例如 GPU 1 的 shard 1 先完成后，下一项 shard 8 固定映射 GPU 0，而 GPU 0 的 shard 0 可能尚未结束，于是同一张 GPU 可能短时间并发两个进程。`WORKERS == GPU 数量` 并不能严格保证一卡一进程。

### 39.11 每个 shard 内实际做什么

每个 shard 调用 `evaluate_kws_onnx_float_manifest.py`。执行流程为：

1. 先 import `torch`，使 PyTorch wheel 携带的 CUDA/cuDNN 动态库对后续 ONNX Runtime CUDA EP 可见。
2. 有 custom op 时关闭 ORT `enable_mem_pattern` 和 `enable_mem_reuse`。
3. 默认把 CUDA EP 的 `use_tf32` 设为 `0`，减少不同量化实验间的额外数值变量。
4. 注册 SIMO custom-op 库并创建 encoder、decoder、joiner 三个 CUDA ORT session。
5. 检查三个 session 的 active providers 中都确实包含 `CUDAExecutionProvider`，防止静默回落 CPU。
6. 读取 16 kHz 音频或 manifest 指定片段；双声道会平均成单声道。
7. 在音频尾部补 0.66 秒零值，提取 80 维 Kaldi fbank；frame length 25 ms、frame shift 10 ms。
8. 以最多 64 条音频为一个 batch，运行 streaming Zipformer encoder/decoder/joiner。
9. 使用最多 4 条 active paths 和 keyword context graph 做关键词解码；一次命中后重置该 stream 的模型状态并继续检测后续事件。
10. 为每条语音记录检测关键词、第一条预测、检测次数、时长和分类结果。

关键词定义来自：

```text
/share_data/mtang/work/JD/test/kws/data/open_commands_bilingual_kws_eval/small/keywords.txt
```

当前文件有 40 行关键词规则。

### 39.12 outcome 和指标怎样计算

每条数据先分类：

| 数据类型 | 检测情况 | outcome |
|---|---|---|
| target | 没检测到 | `miss` |
| target | 检测列表包含正确关键词，且总检测数为 1 | `hit` |
| target | 检测列表包含正确关键词，且总检测数大于 1 | `hit_with_extra` |
| target | 检测到了其他关键词 | `wrong_keyword` |
| non-target | 有任意检测 | `false_alarm` |
| non-target | 无检测 | `correct_reject` |

核心指标为：

```text
precision = target_hits /
            (target_hits + nontarget_false_alarm_events + target_wrong_keyword)

recall = target_hits / target_utterances

F1 = 2 * precision * recall / (precision + recall)
```

此外还计算：

```text
collapsed_accuracy
nontarget_false_alarm_rate
false_alarms_per_hour_on_nontarget
每个 group 的统计
每个 keyword 的 recall
false alarm keyword 分布
top 100 confusion
```

### 39.13 输出目录包含什么

每套配置的目录结构为：

```text
<output_dir>/
  onnx_quant_config.json
  quantized_model_dir/
    encoder-*.onnx
    decoder-*.onnx
    joiner-*.onnx
    tokens.txt
  run_metadata.json
  shard_summary.json
  shards/
    shard_000/
      run.log
      utterance_results.csv
      metrics.json
    ...
    shard_031/
      run.log
      utterance_results.csv
      metrics.json
  utterance_results.csv
  metrics.json
```

其中：

- `run_metadata.json` 保存命令、设备、provider、TF32、manifest 和模型路径。
- `shard_summary.json` 每完成一个 future 就更新一次，可用于判断哪些 shard 成功或失败。
- 每个 shard 的 `run.log` 保存其完整子命令、stdout/stderr、开始结束时间和 return code。
- shard `utterance_results.csv` 是逐条结果；shard `metrics.json` 是局部统计。
- 根目录 `utterance_results.csv` 是全部 shard CSV 的合并结果。
- 根目录 `metrics.json` 是在全部 31562 条合并数据上重新调用 `summarize()` 得到的全局结果，不能简单平均 32 个 shard 的 precision/recall。

父进程按 shard index 依次拼接各 shard 的行，因此根 CSV 的顺序是“shard 0 全部行、shard 1 全部行……”而不是恢复原 manifest 顺序；这不影响聚合指标，但按行和原 manifest 对比时应以 `id` 等字段关联。

任一 shard 返回非零时，父进程会等待已提交任务收集完结果后返回 1，并且不会生成新的根级聚合 CSV/metrics；已完成 shard 的文件和增量 `shard_summary.json` 会保留。

### 39.14 重跑、续跑和 dry run 的实际语义

默认：

```bash
FORCE=--no-skip-existing
```

所以即使 shard 已有 `metrics.json` 和 `utterance_results.csv`，也会重新执行并覆盖日志/结果。变量名 `FORCE` 表示强制重跑。

要利用已有完整 shard 续跑，应显式使用：

```bash
FORCE=--skip-existing
```

不能用 `FORCE=`，因为 `${FORCE:---no-skip-existing}` 会把空字符串也替换回默认的 `--no-skip-existing`。

skip 条件要求 shard 的两个文件同时存在：

```text
metrics.json
utterance_results.csv
```

续跑不会验证这些文件对应的量化配置、SIMO 版本或模型 hash。因此配置或代码变化后必须使用新输出目录，或强制重跑，否则可能把旧 shard 和新 shard 混合进同一个全局结果。

`DRY_RUN=1` 会生成 task 命令和 metadata，但不改写 ONNX、也不启动 shard 推理。它仍会解析并检查 custom-op 库，所以在 cache 为空时仍可能触发一次 runtime `.so` 构建。

### 39.15 几个实际使用风险

1. **工作目录依赖**：脚本没有 `cd`，必须从 `kws_simo_quant` 根目录运行。
2. **环境依赖**：必须显式激活 conda 环境或设置绝对 `PY`；脚本不会自动使用 `simo_sglang`。
3. **默认覆盖结果**：`--no-skip-existing` 会覆盖同名 shard 输出。
4. **GPU 可能过订阅**：全局 worker pool 没有 per-device 锁，后续 task 可能与同 GPU 上尚未完成的 task 重叠。
5. **不要在多配置模式共用 `QUANTIZED_MODEL_DIR`**：若设置一个全局固定目录，11 套配置会依次覆盖同一组三个量化模型；各次已经完成的指标仍来自当时模型，但最终目录只保留最后写入的模型。
6. **输出目录不是内容寻址的**：目录名不包含 SIMO source hash、custom-op cache key 或配置内容 hash，中断重跑时需人工保证一致性。

### 39.16 推荐的 smoke test

在完整 11 套、31562 条数据实验前，可先只跑一套配置和少量样本：

```bash
cd /share/users/like/package/jdjv/kws_simo_quant

PY=/share_data/users/like/miniconda3/envs/simo_sglang/bin/python \
QUANT_CONFIG=w8a8/w_mxint8_a_mxint8.json \
OUTPUT_DIR=temp/mxint8_smoke \
MAX_ITEMS=128 \
NUM_SHARDS=8 \
WORKERS=8 \
bash test_quant_onnx.sh
```

确认 `temp/mxint8_smoke/metrics.json`、8 个 shard 和各 shard `run.log` 后，再使用独立的新输出目录启动完整实验。

## 40. `build_sm90_runtime()` 的构建原理，以及 nvcc/NVRTC 是否参与

分析环境：

```text
Conda：/share_data/users/like/miniconda3/envs/simo_sglang
SIMO：/share/users/like/package/simo_conda_sglang
入口：simo/onnx/ort_plugin/build_runtime.py:47-91
PyTorch：2.11.0+cu130
Triton：3.6.0
目标 GPU：sm90（实际 ptxas target 为 sm_90a）
```

### 40.1 先给出结论

`build_sm90_runtime()` 包含两条不同的编译链：

```text
Triton Python kernels
  -> Triton/MLIR/LLVM
  -> PTX
  -> ptxas
  -> 33 份 sm_90a cubin
  -> 嵌入 generated C++ 文件

host C++ ORT plugin
  -> Ninja
  -> c++ 编译器
  -> 链接 libcuda
  -> SimoOnnxCustomOps_sm90.so
  -> 复制成 libSimoOnnxCustomOps_sm90.so
```

各工具是否参与：

| 工具/接口 | 是否用于这条构建链 | 说明 |
|---|---|---|
| `nvcc` | **否** | 没有 `.cu` source，`cpp_extension.load(with_cuda=False)`，Ninja 中无 nvcc rule |
| NVRTC 编译 API | **否** | 没有调用 `nvrtcCreateProgram/nvrtcCompileProgram`；Triton 用 LLVM 生成 PTX，再调用 `ptxas` |
| `ptxas` | **是** | 冷 Triton cache 时汇编 PTX；warm cache 新进程仍可能执行 `--version` 探测 |
| `c++` | **是** | 编译四个 host `.cc` 并链接共享库 |
| Ninja | **是** | 由 `torch.utils.cpp_extension.load()` 生成并驱动 host build |
| CUDA Driver API | **是** | build 时链接 `libcuda`；运行时加载 cubin 并 launch kernel |
| CUDA Runtime API / `cudart` | **不是核心路径** | custom-op loader 直接使用 Driver API，而不是 `<<<>>>` launch 或 Runtime API |

有一个容易造成误判的细节：Python 进程在 `import torch` 时会加载 `libnvrtc.so`，但这只是 PyTorch 的传递依赖，不表示 `build_sm90_runtime()` 使用 NVRTC 编译 kernel。后面会给出实际 trace 证据。

### 40.2 谁会调用 `build_sm90_runtime()`

普通用户通常不是直接调用该函数，而是调用：

```python
from simo.onnx import get_custom_ops_library_path
```

`simo/onnx/runtime.py:108-131` 的顺序是：

```text
检查当前可见 GPU 必须是 capability (9, 0)
  -> 查找 wheel/source package 内预编译库
  -> 根据环境和源码计算 cache path
  -> cache 中已有 .so 就直接返回
  -> 否则获取 flock 文件锁
  -> 再检查一次 cache
  -> 调用 build_sm90_runtime(cache_path)
```

文件锁防止多个 Python/pytest/KWS 进程同时生成同一个 runtime。

cache key 当前包含：

```text
sm90
Python major/minor
ONNX Runtime package version
torch.version.cuda
SIMO plugin、ORT headers 和 Triton kernel source hash
```

当前 cache 产物为：

```text
/softhome/like/.cache/simo/onnx/
sm90-py312-ort1.27.0-cuda13.0-08d729191a9f/
libSimoOnnxCustomOps_sm90.so
```

注意：直接调用 `build_sm90_runtime(output)` 本身没有 `_ensure_sm90()`；真实 GPU 检查位于上层 `get_custom_ops_library_path()`。因为 Triton target 被显式写成 arch 90，底层理论上可以做离线 cross-compile，但标准公共路径仍要求当前机器有可见 sm90 GPU。

### 40.3 第一步：准备 build 目录

`build_runtime.py:47-55` 将传入路径转成 `Path`，并创建：

```text
<output parent>/
  build/
    generated/
```

随后执行：

```python
generated = build_qdq_cubins.build(generated_dir, 90)
```

这一句完成所有 GPU device kernel 的提前编译，是整个构建中耗时和生成文件体积最大的阶段。

### 40.4 第二步：Triton kernel specialization

`build_qdq_cubins.py:71-75` 没有按常见的 `kernel[grid](...)` 方式在 GPU 上首次运行时 JIT，而是显式调用 Triton compiler：

```python
target = GPUTarget("cuda", 90, 32)
triton_compile(
  ASTSource(source, signature, constants),
  target=target,
  options={"num_warps": 8},
)
```

这里各字段含义是：

- `source`：一个 `@triton.jit` kernel。
- `signature`：11 个 SIMO kernel 显式参数的 pointer/integer 类型。
- `constants`：dtype、block size、rounding mode、pack mode 等 `tl.constexpr` specialization。
- `GPUTarget("cuda", 90, 32)`：CUDA backend、compute capability 9.0、warp size 32。
- `num_warps=8`：每个 Triton program 使用 8 warps，即通常 256 threads。

11 个显式参数依次是：

```text
q, scale, tensor,
outer_dim, quant_dim, packed_dim, scale_cols,
group_size, block_size,
reserved0, reserved1
```

Triton 3.6 生成 PTX entry 时还会在末尾附加 global scratch 和 profile scratch 两个 ABI pointer，所以最终 cubin entry 有 13 个参数。`simo_qdq_ops.cc` 相应固定准备 13 个 host arguments，最后两个是 `triton_scratch0/1`。当前这批 kernel 的：

```text
global_scratch_size = 0
profile_scratch_size = 0
```

因此 host 传入的两个 scratch pointer 都是空指针，但保留这两个位置可确保所有 kernel 使用同一 Triton 3.6 launch ABI。

`build()` 为不同语义组合分别 specialization，包括：

```text
MXINT8/MXFP8/MXFP6/MXFP4/NVFP4
Quantize 与 Dequantize
e8m0_floor/e8m0_sipu/e4m3 scale mode
per-group/per-block/per-channel
int8 full range 与 narrow range
fp8_e4m3/int8/int4 的不同 layout
```

当前生成文件中有 33 个 `EmbeddedQdqKernel`，即 33 份独立 cubin。这样 runtime 不需要再根据用户配置即时编译，只需从预编译表中选择匹配项。

### 40.5 Triton 内部怎样生成 cubin

Triton 3.6.0 CUDA backend 的实际 pipeline 是：

```text
Python @triton.jit AST
  -> TTIR
  -> Triton GPU IR (TTGIR)
  -> LLVM IR / NVVM lowering
  -> LLVM NVPTX backend 生成 PTX 文本
  -> 外部 ptxas 把 PTX 汇编为 cubin
```

关键点是 PTX 由 Triton 自带的 MLIR/LLVM backend 生成，不是由 nvcc 或 NVRTC 从 CUDA C++ 生成。

当前环境的 Triton backend 在：

```text
/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/
site-packages/triton/backends/nvidia/compiler.py
```

其中 `make_ptx()` 调用：

```python
llvm.translate_to_asm(...)
```

而 `make_cubin()` 明确构造并执行：

```python
ptxas_cmd = [
  ptxas,
  "-lineinfo",
  "-v",
  "--gpu-name=sm_90a",
  input_ptx,
  "-o",
  output_cubin,
]
subprocess.run(ptxas_cmd, check=True, ...)
```

虽然 SIMO API 和文件名写的是 `sm90`，Triton 对 capability 90 添加 `a` suffix，实际汇编目标是 `sm_90a`。

当前 Triton 使用的不是 `$PATH` 中的 `/usr/local/cuda-*/bin/ptxas`，而是 wheel 内置工具：

```text
/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/
site-packages/triton/backends/nvidia/bin/ptxas
```

其版本为：

```text
Cuda compilation tools, release 12.8, V12.8.93
```

所以 `torch.version.cuda == 13.0` 不代表 Triton 一定调用 CUDA 13.0 toolkit 中的 ptxas；Triton wheel 有自己的 tool selection。

如果同一 specialization 已命中 `TRITON_CACHE_DIR`，Triton 可以直接读取 cubin cache，不必再次运行“PTX 输入到 cubin 输出”的 assembler 命令。新 Python 进程为了计算 compiler/cache hash，仍可能运行一次 `ptxas --version`。`build_qdq_cubins.py:762` 默认将该 cache 放到：

```text
build/generated/.triton_cache
```

外部预先设置的 `TRITON_CACHE_DIR` 会覆盖这个默认位置，因为代码使用的是 `os.environ.setdefault()`。

### 40.6 第三步：把 cubin 变成 C++ byte arrays

每个 `triton_compile()` 结果包含：

```python
compiled.asm["cubin"]
compiled.metadata.name
compiled.metadata.shared
```

`_entry()` 提取：

- cubin 原始 bytes；
- cubin 中的 kernel symbol name；
- dynamic shared memory bytes；
- dtype/granularity/scale mode/narrow range 等匹配条件；
- packed/logical shape ratio；
- launch grid layout。

`_write_embedded_source()` 将每份 cubin 展开成：

```cpp
alignas(16) const unsigned char kCubin_xxx[] = {
  0x7f, 0x45, 0x4c, 0x46, ...
};
```

并生成对应的：

```cpp
EmbeddedQdqKernel
QdqRuntimeSpec
ResolveQdqRuntimeSpecSm90(...)
```

最终文件为：

```text
build/generated/embedded_qdq_kernels_sm90.cc
```

当前 cache 中该文件约 14 MB。它包含真正的 ELF cubin bytes，而不是 PTX 字符串，因此最终 runtime 不需要 NVRTC，也不依赖 CUDA Driver 的 PTX JIT。

### 40.7 第四步：准备 host C++ plugin

GPU cubin 生成后，`build_runtime.py:57-62` 组合四个 host sources：

```text
custom_op_library.cc
simo_qdq_ops.cc
triton_loader.cc
generated/embedded_qdq_kernels_sm90.cc
```

职责分别是：

| source | 作用 |
|---|---|
| `custom_op_library.cc` | 导出 ORT 要求的 `RegisterCustomOps`，创建 `com.simo` domain |
| `simo_qdq_ops.cc` | 定义 Quantize/Dequantize schema、shape inference、参数校验和 launch 参数 |
| `triton_loader.cc` | 使用 CUDA Driver API 加载/查找/启动内嵌 cubin |
| generated `.cc` | 保存 33 个 cubin byte arrays 和 semantic resolver table |

include path 包括：

```text
SIMO plugin headers
vendored ONNX Runtime C/C++ headers
PyTorch C++ headers
CUDA headers
```

CUDA headers来自 `torch.utils.cpp_extension.include_paths("cuda")`，因此取决于 build 当时的 `CUDA_HOME`。这和 Triton wheel 自己选择哪个 ptxas 是两套独立机制。

### 40.8 为什么它链接 `libcuda` 而不是使用 nvcc

`_cuda_library_paths()` 会从以下来源寻找包含 `libcuda.so` 或 `libcuda.so.1` 的目录：

```text
torch cpp_extension library_paths("cuda")
SIMO_CUDA_DRIVER_LIBRARY_PATH
LIBRARY_PATH
LD_LIBRARY_PATH
```

链接参数最终使用：

```text
-lcuda
```

或在只有 versioned driver library 时使用：

```text
-l:libcuda.so.1
```

这是普通 host C++ 链接 CUDA Driver API 的方式。只要代码没有 CUDA device syntax 和 kernel launch syntax，普通 `c++` 编译器就可以编译：

```cpp
cuModuleLoadData(...)
cuModuleGetFunction(...)
cuLaunchKernel(...)
```

这些函数只需要 `cuda.h` 声明以及链接 `libcuda`，不需要 nvcc。

### 40.9 第五步：`torch.utils.cpp_extension.load()`

host build 的关键调用是：

```python
load(
  name="SimoOnnxCustomOps_sm90",
  sources=[四个 .cc],
  extra_cflags=["-std=c++17", "-O3"],
  extra_include_paths=[...],
  extra_ldflags=[..., "-lcuda", version_script],
  build_directory=build_dir,
  with_cuda=False,
  is_python_module=False,
)
```

参数含义：

- `with_cuda=False`：不要生成 CUDA/nvcc compilation rules。
- `is_python_module=False`：产物是普通 shared library，不要求 `PyInit_*` Python extension entry point。
- `build_directory`：Ninja 文件、object 和中间 `.so` 都写在 runtime cache 下。
- `SIMO_ONNX_RUNTIME_BUILD_VERBOSE=1`：显示 cpp_extension/Ninja host build 详情。

当前实际 `build.ninja` 的核心内容是：

```ninja
cxx = c++

rule compile
  command = $cxx ... -c $in -o $out

rule link
  command = $cxx $in $ldflags -o $out

build custom_op_library.o: compile custom_op_library.cc
build simo_qdq_ops.o: compile simo_qdq_ops.cc
build triton_loader.o: compile triton_loader.cc
build embedded_qdq_kernels_sm90.o: compile embedded_qdq_kernels_sm90.cc
build SimoOnnxCustomOps_sm90.so: link ...
```

其中没有：

```text
nvcc = ...
rule cuda_compile
任何 .cu source
```

这直接证明 host plugin 阶段没有调用 nvcc。

linker 还使用：

```text
-Wl,--version-script=custom_op_library.lds
```

使最终插件只公开 ORT 所需的 `RegisterCustomOps`，隐藏其他 C++ symbols，降低 ABI 冲突风险。

### 40.10 最终文件名为什么变化

`cpp_extension.load()` 先生成：

```text
build/SimoOnnxCustomOps_sm90.so
```

然后 `shutil.copy2()` 将它复制到调用者要求的：

```text
libSimoOnnxCustomOps_sm90.so
```

因此 build 子目录中没有 `lib` prefix，而 runtime cache 的公开产物有 `lib` prefix；它们内容和时间戳相同。

### 40.11 运行时如何执行内嵌 cubin

ONNX Runtime 加载 `.so` 后调用 `RegisterCustomOps`，插件注册：

```text
com.simo::Quantize
com.simo::Dequantize
```

模型 session 初始化时，semantic resolver 根据节点的：

```text
dtype
granularity
scale_mode
group_size
block_size
narrow_range
```

选出对应 `QdqRuntimeSpec`。

实际执行时，`triton_loader.cc` 使用：

```cpp
cuModuleLoadData(&module, embedded.cubin)
cuModuleGetFunction(&function, module, embedded.symbol_name)
cuLaunchKernel(function, ..., ort_cuda_stream, ...)
```

module/function 按 CUDA context 缓存，同一个 context 不重复 load。launch 使用 ORT 提供的 CUDA stream，因此 custom op 和 ORT 图中的其他 CUDA op 保持正确的 stream ordering。

这里传给 `cuModuleLoadData` 的是 cubin ELF bytes，不是 CUDA C++ source，也不是 PTX source。运行时没有 Triton JIT、nvcc 或 NVRTC 编译步骤。

最终 `.so` 的 dynamic dependencies 实测包含：

```text
libcuda.so.1
libc10.so
libtorch_cpu.so
libstdc++.so.6
libgcc_s.so.1
libc.so.6
```

没有 `libnvrtc.so`。动态未解析 CUDA symbols 包括：

```text
cuModuleLoadData
cuModuleGetFunction
cuLaunchKernel
cuModuleUnload
```

这与 Driver API loader 的设计一致。

### 40.12 `nvcc` 到底会不会被调用

当前实现和当前环境下，答案是：**不会**。

理由有四个：

1. 输入 source 全部是 `.cc`，没有 `.cu`。
2. `cpp_extension.load()` 显式传入 `with_cuda=False`。
3. 实际 `build.ninja` 只有 `cxx = c++`。
4. 单 Triton kernel 的 `strace -f -e execve` 只看到 Triton bundled `ptxas`，没有 nvcc process。

机器上是否安装了 `/usr/local/cuda-*/bin/nvcc` 与这个结论无关；“可执行文件存在”不等于构建脚本调用它。

只有未来出现以下变化时才可能引入 nvcc：

```text
sources 中加入 .cu
with_cuda 改成 True/自动推断为 True
显式 subprocess 调用 nvcc
人为把 CXX 环境变量设置为 nvcc
```

当前代码均未这样做。

### 40.13 NVRTC 到底会不会被调用

如果“调用 NVRTC”指使用 NVRTC compiler API 编译 CUDA C++，答案也是：**不会**。

这条路径不存在：

```text
CUDA C++ source
  -> nvrtcCreateProgram
  -> nvrtcCompileProgram
  -> nvrtcGetPTX
```

实际路径是：

```text
Triton AST/IR
  -> Triton MLIR/LLVM
  -> PTX
  -> ptxas
  -> cubin
```

代码中没有 `nvrtc*` API 调用，最终 `.so` 也不依赖 `libnvrtc`。

但系统调用跟踪确实能看到：

```text
openat(.../nvidia/cuda_nvrtc/lib/libnvrtc.so.12, ...)
openat(.../nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so.12.8, ...)
```

原因是 `build_runtime.py` 导入 `torch.utils.cpp_extension`，而 `import torch` 会加载 `libcaffe2_nvrtc.so` 及其 NVRTC dependencies。用一个只执行：

```python
import torch
```

的基线进程也能观察到完全相同的 `libnvrtc.so` load。因此准确表述应是：

```text
Python/PyTorch 进程可能加载 NVRTC shared library；
build_sm90_runtime 的 kernel 编译逻辑不调用 NVRTC compiler。
```

不能仅凭 `lsof`、`/proc/<pid>/maps` 或 `strace openat` 中出现 `libnvrtc.so`，就判断 kernel 是 NVRTC 编译的；应看是否调用 `nvrtcCompileProgram` 以及实际 compiler pipeline。

### 40.14 实际 trace 证据

对一个全新 Triton cache，仅编译 `mxint8` Quantize specialization，结果为：

```text
kernel_symbol = _downcast_to_mxfmt
cubin_bytes = 83416
```

关键 `execve` 为：

```text
.../triton/backends/nvidia/bin/ptxas --version

.../triton/backends/nvidia/bin/ptxas \
  -lineinfo -v \
  --gpu-name=sm_90a \
  /tmp/<temporary>.ptx \
  -o /tmp/<temporary>.ptx.o
```

trace 中没有 nvcc executable。`libnvrtc.so` 的 open 行在只 import torch 的 baseline 中同样存在。

### 40.15 toolchain 和 cache 的注意事项

这个构建同时涉及几套版本来源：

```text
torch.version.cuda                 -> outer runtime cache key 中的 cuda13.0
CUDA_HOME/include_paths("cuda")   -> host C++ 使用的 CUDA headers
Triton wheel bundled ptxas        -> PTX 到 cubin 的 assembler
系统 NVIDIA driver/libcuda        -> plugin link 和 runtime module load/launch
```

它们不是同一个概念，也不一定来自同一目录。当前环境就是：

```text
PyTorch CUDA version: 13.0
Triton bundled ptxas: CUDA 12.8.93
runtime target: sm_90a
```

外层 `_runtime_cache_key()` 没有显式包含：

```text
Triton package version
ptxas version/path
host C++ compiler version
CUDA_HOME/header version
```

因此只升级 Triton、ptxas 或 host toolchain，而 Python/ORT/torch CUDA/source hash 都不变时，可能继续复用旧 `.so`。需要验证新 toolchain 时，最稳妥的方法是设置新的 cache 根目录，而不是依赖旧 key：

```bash
cd /share/users/like/package/simo_conda_sglang

SIMO_ONNX_RUNTIME_CACHE="$PWD/temp/simo-onnx-runtime-toolchain-check" \
SIMO_ONNX_RUNTIME_BUILD_VERBOSE=1 \
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
from simo.onnx.runtime import get_custom_ops_library_path
print(get_custom_ops_library_path())
PY
```

### 40.16 最终总结

`build_sm90_runtime()` 的核心不是“用 nvcc 编译一个 CUDAExtension”，而是：

```text
先由 Triton compiler 离线生成所有需要的 sm_90a cubin；
再把 cubin 当作普通 byte arrays 编进一个 host C++ ORT plugin；
运行时用 CUDA Driver API 从内存加载 cubin，并在 ORT stream 上 launch。
```

所以最简短而准确的回答是：

```text
nvcc：不调用。
NVRTC compiler：不调用。
ptxas：冷 Triton cache 构建时调用。
c++/Ninja：host shared library 构建时调用。
libcuda Driver API：链接并在运行时调用。
libnvrtc.so：可能因 import torch 被加载，但不参与本函数的 kernel 编译。
```

## 41. `test_simo_custom_qdq_plugin_runs_mx_qdq_tiny_tensor` 与 MXINT8 QDQ kernel 调用链

以下分析以参数组合：

```text
dtype = "mxint8"
block_size = 32
```

为例。这里必须先区分三层“实现”：

1. ONNX 图中的 `com.simo::Quantize` / `com.simo::Dequantize` 节点只是算子声明；
2. ORT custom op 的 C++ 适配实现分别是 `SimoQuantizeOp` / `SimoDequantizeOp`；
3. 真正执行量化数学的 GPU kernel 是 Triton kernel，它在构建 `.so` 时被离线编译为 cubin 并作为字节数组嵌入 `.so`。

因此，`onnx.helper.make_node()` 不会在构图时直接调用 CUDA kernel。真正的 kernel launch 发生在 `session.run()` 时。

### 41.1 这个测试整体在做什么

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:815-826` 的 `pytest.mark.parametrize()` 定义了 7 组 `(dtype, block_size)`，其中第一组是 `("mxint8", 32)`。

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:827-846` 的 `test_simo_custom_qdq_plugin_runs_mx_qdq_tiny_tensor()` 对每组参数执行以下流程：

1. `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:828-833` 调用 `_tiny_simo_qdq_model()`，构造输入 shape 为 `[128, 32]` 的 QDQ ONNX 模型和 CUDA ORT session；
2. `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:834` 构造 shape 为 `[128, 32]`、值域为 `[-4, 4]` 的 `float32` 输入；
3. `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:836` 的 `test_simo_custom_qdq_plugin_runs_mx_qdq_tiny_tensor()` 调用 `session.run()`，此处才真正执行 Quantize 和 Dequantize；
4. `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:838-840` 检查输出 shape、dtype 和 SQNR；
5. `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:841-846` 将 ORT plugin 输出与 `_reference_mx_qdq()` 的 PyTorch/Triton 路径结果进行比较。

需要注意，`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:135-154` 的 `_reference_mx_qdq()` 最终也使用同一套基础 Triton QDQ 算法。因此这个测试主要验证：

```text
ONNX 属性 -> ORT custom-op 匹配 -> shape/type 推导 -> C++/Triton ABI
-> 嵌入 cubin 的选择和 launch -> 数值结果
```

它不是用一套完全独立的数学实现验证另一套数学实现。

### 41.2 为什么 `tmp_path` 不在 `parametrize` 中也会有值

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:827` 的函数签名是：

```python
def test_simo_custom_qdq_plugin_runs_mx_qdq_tiny_tensor(tmp_path, dtype, block_size):
```

这里有两种不同的参数来源：

| 参数 | 值的来源 |
|---|---|
| `dtype` | `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:815-826` 的 `pytest.mark.parametrize()` |
| `block_size` | `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:815-826` 的 `pytest.mark.parametrize()` |
| `tmp_path` | pytest 内置的同名 fixture |

pytest 收集测试时会检查测试函数参数名。`dtype` 和 `block_size` 已由参数化 callspec 提供；剩余的 `tmp_path` 会按名字从 fixture 注册表中解析。`tmp_path` 是 pytest 自带的、`function` scope 的 fixture，不需要在 SIMO code base 中定义，也不应该加入 `parametrize`。

对本测试，pytest 会收集出 7 个独立 item，例如：

```text
test_simo_custom_qdq_plugin_runs_mx_qdq_tiny_tensor[mxint8-32]
test_simo_custom_qdq_plugin_runs_mx_qdq_tiny_tensor[mxfp8_e5m2-32]
...
test_simo_custom_qdq_plugin_runs_mx_qdq_tiny_tensor[nvfp4_e2m1-16]
```

每个 item setup 时，pytest 的 `tmp_path` fixture 都会创建一个本次 item 独占的 `pathlib.Path`。通常类似：

```text
/tmp/pytest-of-<user>/pytest-<run-number>/test_simo_custom_qdq_plugin_ru<item-number>/
```

确切目录名由 pytest 版本、当前 test node id 和临时目录编号决定，不能在测试中假设固定值；使用 `pytest --basetemp=<dir>` 可以改变临时目录根路径。

该 Path 在 `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:828-830` 被传给 `_tiny_simo_qdq_model()`，随后 `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:370-371` 的 `_tiny_simo_qdq_model()` 将模型保存为：

```python
model_path = tmp_path / "tiny_simo_qdq.onnx"
onnx.save(model, model_path)
```

所以 7 个参数化 item 不会争用同一个 `tiny_simo_qdq.onnx`。简言之：

```text
parametrize 负责 dtype/block_size；
fixture 注入负责 tmp_path；
两者可以同时出现在测试函数参数列表中。
```

### 41.3 `mxint8/32` 最终写入 ONNX 节点的属性

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:332-377` 的 `_tiny_simo_qdq_model()` 创建两个 `com.simo` domain 节点。

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:340-348` 的 `_tiny_simo_qdq_model()` 创建 Quantize：

```text
com.simo::Quantize
input  = ["input"]
output = ["input_SimoQuantInput", "input_SimoScale"]
```

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:349-357` 的 `_tiny_simo_qdq_model()` 创建 Dequantize：

```text
com.simo::Dequantize
input  = ["input_SimoQuantInput", "input_SimoScale"]
output = ["input_SimoDequantOutput"]
```

二者都调用 `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:104-132` 的 `_semantic_attrs()`。该函数在 `:104-124` 组装配置并调用 `parse_quantize_spec()`，然后在 `:126-132` 把规范化后的语义属性返回给 ONNX node。

`simo/quantization/config.py:654-688` 的 `parse_quantize_spec()` 根据 `dtype="mxint8"` 选择 `QuantizeSpecMX`。`simo/quantization/config.py:415-457` 的 `QuantizeSpecMX` 给出默认值：

```text
scale_mode   = e8m0_floor
observer_mode = abs_max
block_size   = 32
axis         = -1
is_dynamic   = false
```

`simo/quantization/config.py:459-468` 的 `QuantizeSpecMX.sync_group_size_with_block_size()` 再把未显式设置的 `group_size` 同步为 `32`。

因此两个 ONNX 节点的关键属性最终都是：

```text
dtype         = "mxint8"
granularity   = "per_group"
scale_mode    = "e8m0_floor"
observer_mode = "abs_max"
narrow_range  = 1
group_size    = 32
block_size    = 32
```

`axis=-1` 没有作为当前 custom-op ABI 的 node attribute 写出；当前 v1 plugin 本身限定 contiguous rank-2 tensor，并把第二维作为 quantization dimension。

### 41.4 `make_node()` 为什么能找到 SIMO kernel

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:56-85` 的 `_session_options_with_simo_plugin()` 先准备 ORT session options：

1. `:64-71` 读取 `SIMO_ONNX_CUSTOM_OPS_LIBRARY`，未设置时调用 `get_custom_ops_library_path()`；
2. `:76-78` 创建 `ort.SessionOptions()` 并调用 `register_custom_ops_library()` 加载 `.so`。

`simo/onnx/runtime.py:108-131` 的 `get_custom_ops_library_path()` 优先返回随包提供或 cache 中的 `libSimoOnnxCustomOps_sm90.so`；没有时在 `:120-128` 调用 `build_sm90_runtime()` 构建它。

加载 `.so` 后，ORT 调用其导出入口。`simo/onnx/ort_plugin/custom_op_library.cc:23-50` 的 `RegisterCustomOps()`：

1. 在 `:39` 初始化 ORT C++ API；
2. 在 `:41` 创建 `com.simo` custom domain；
3. 在 `:42` 调用 `RegisterQdqOps()`；
4. 在 `:43-45` 将 domain 加入 session options，并保持其生命周期。

`simo/onnx/ort_plugin/simo_qdq_ops.cc:504-515` 的 `RegisterQdqOps()` 完成名字到 C++ 实现的绑定：

```text
(domain="com.simo", op="Quantize", provider="CUDAExecutionProvider")
    -> SimoQuantizeOp

(domain="com.simo", op="Dequantize", provider="CUDAExecutionProvider")
    -> SimoDequantizeOp
```

因此 `make_node("Quantize", domain="com.simo")` 找到实现的关键是 `(domain, op name, opset/provider)` 注册匹配，不是 Python 函数名匹配。

### 41.5 构建阶段：实际 GPU kernel 如何进入 `.so`

Quantize/Dequantize GPU kernel 不是运行 `session.run()` 时才从 Python JIT 编译，也不是一份单独登记的 `.cu` 文件。构建链如下：

```text
simo/onnx/ort_plugin/build_runtime.py:47-91
build_sm90_runtime()
  -> simo/onnx/ort_plugin/build_qdq_cubins.py:758-807
     build(output_dir, arch=90)
       -> _compile_quant("mxint8", 90)
       -> _compile_dequant("mxint8", 90)
       -> _entry(...)
       -> _write_embedded_source(...)
  -> 将生成的 embedded_qdq_kernels_sm90.cc 与 C++ plugin 一起编入 .so
```

`simo/onnx/ort_plugin/build_runtime.py:47-91` 的 `build_sm90_runtime()` 在 `:55` 调用 `build_qdq_cubins.build()`，在 `:57-62` 把生成的 C++ 源文件加入 plugin sources，最后在 `:77-90` 构建和复制 shared library。

#### Quantize cubin 的生成

`simo/onnx/ort_plugin/build_qdq_cubins.py:350-366` 的 `_compile_quant()` 用以下 compile-time 常量实例化通用 kernel：

```text
BLOCK_SIZE_OUT_DIM       = 128
BLOCK_SIZE_QUANT_DIM     = 32
MX_FORMAT_ID             = 1          # MXINT8
HADAMARD_TRANSFORM_SIZE  = 0
OBSERVER_MODE            = 1          # abs_max
QUANT_SCALE_ROUNDING_MODE = 1         # e8m0_floor
MX_QUANT_DIM             = 32
PACK_MODE                = 1
num_warps                = 8
```

`simo/onnx/ort_plugin/build_qdq_cubins.py:119-161` 的 `_downcast_to_mxfmt()` 是为了固定 host ABI 而增加的 Triton wrapper；它在 `:141-161` 调用真正的基础 kernel `simo/ops/kernels/downcast/_downcast_to_mxfmt.py:525-694` 的 `_downcast_to_mxfmt()`。

#### Dequantize cubin 的生成

`simo/onnx/ort_plugin/build_qdq_cubins.py:369-381` 的 `_compile_dequant()` 使用 `BLOCK_SIZE_OUT_DIM=128`、`BLOCK_SIZE_QUANT_DIM=32`、`MX_FORMAT_ID=1` 和 `PACK_MODE=1` 实例化 kernel。

`simo/onnx/ort_plugin/build_qdq_cubins.py:164-198` 的 `_upcast_from_mxfmt()` 是 host ABI wrapper；它在 `:182-198` 调用真正的基础 kernel `simo/ops/kernels/upcast/_upcast_from_mxfmt.py:205-331` 的 `_upcast_from_mxfmt()`。

`simo/onnx/ort_plugin/build_qdq_cubins.py:71-75` 的 `_compile()` 调用 Triton compiler，返回的 `compiled.asm["cubin"]` 在 `simo/onnx/ort_plugin/build_qdq_cubins.py:629-670` 的 `_entry()` 中保存到 `EmbeddedEntry`。

`simo/onnx/ort_plugin/build_qdq_cubins.py:758-792` 的 `build()` 为 MXINT8 建立两项：

```text
logical_name = simo_quantize_mxint8
symbol_name  = _downcast_to_mxfmt

logical_name = simo_dequantize_mxint8
symbol_name  = _upcast_from_mxfmt
```

`simo/onnx/ort_plugin/build_qdq_cubins.py:684-755` 的 `_write_embedded_source()` 把 cubin 写成 `const unsigned char[]`，同时生成 `EmbeddedQdqKernel`、`QdqRuntimeSpec` 和 `ResolveQdqRuntimeSpecSm90()`。

相关 C++ 数据结构位于 `simo/onnx/ort_plugin/embedded_qdq_kernels.h:9-59`：

```text
EmbeddedQdqKernel = cubin bytes + cubin size + kernel symbol + shared memory
QdqRuntimeSpec    = kernel + block/packing/scale/grid 元数据
```

这里的关键点是：运行时 C++ 不会回调这些 Python 函数；这些 Python/Triton 函数已经在构建期被编译成 cubin，运行时执行的是嵌入 `.so` 的 GPU machine code。

### 41.6 `mxint8/32` 对应的 runtime spec、shape 和 grid

`simo/onnx/ort_plugin/build_qdq_cubins.py:563-600` 的 `_packed_ratio()`、`_logical_ratio()`、`_scale_layout()` 和 `_grid_layout()` 为 MXINT8 产生：

```text
packed ratio  = 1 / 1
logical ratio = 1 / 1
scale layout  = kScaleMx
grid layout   = kGridMx2d
block_out_dim = block_size * 4 = 128
```

`simo/onnx/ort_plugin/build_qdq_cubins.py:603-670` 的 `_condition()` / `_entry()` 生成属性匹配条件。普通 MXINT8 Quantize entry 要求：

```text
op=Quantize, dtype=mxint8, granularity=per_group,
group_size=32, block_size=32, scale_mode != e8m0_sipu
```

Dequantize entry 要求相同 dtype、granularity、group/block size；其结果不需要按 Quantize 的 scale rounding mode 再区分。当前普通 MX entry 的匹配条件没有使用 `narrow_range`，所以 node 中的 `narrow_range=1` 不会改变这个 entry 的选择。

session 初始化时：

1. `simo/onnx/ort_plugin/simo_qdq_ops.cc:58-83` 的 `SpecFromShapeAttrs()` / `SpecFromKernelInfo()` 读取 node attributes；
2. 二者经 `simo/onnx/ort_plugin/simo_qdq_ops.cc:46-56` 的 `ResolveSpec()` 调用生成的 `ResolveQdqRuntimeSpecSm90()`；
3. 返回上述 `simo_quantize_mxint8` 或 `simo_dequantize_mxint8` spec。

`simo/onnx/ort_plugin/simo_qdq_ops.cc:215-260` 的 `QuantizeShape()` 和 `simo/onnx/ort_plugin/simo_qdq_ops.cc:263-287` 的 `DequantizeShape()` 推导出：

```text
Quantize input:   float32 [128, 32]
Quantize q:       uint8   [128, 32]   # MXINT8 原始 bit pattern 容器
Quantize scale:   uint8   [128, 1]    # 每行每 32 个数一个 E8M0 scale
Dequantize output:float32 [128, 32]
```

`simo/onnx/ort_plugin/simo_qdq_ops.cc:101-138` 的 `ScaleCols()` / `ScaleShape()` 计算：

```text
scale_cols = quant_dim / block_size = 32 / 32 = 1
scale_shape = [outer_dim, scale_cols] = [128, 1]
```

`simo/onnx/ort_plugin/simo_qdq_ops.cc:141-167` 的 `LaunchGrid()` 对 `kGridMx2d` 计算：

```text
grid_x = ceil(128 / 128) = 1
grid_y = ceil(32 / 32)   = 1
block  = (256, 1, 1)
```

### 41.7 Quantize 节点的运行时调用链

完整运行时链路是：

```text
simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:836
test_simo_custom_qdq_plugin_runs_mx_qdq_tiny_tensor(): session.run()
  -> ORT 根据 com.simo + Quantize + CUDAExecutionProvider 找到 custom op
  -> simo/onnx/ort_plugin/simo_qdq_ops.cc:292-295
     SimoQuantizeOp::SimoQuantizeOp(): 解析属性并选中 simo_quantize_mxint8 spec
  -> simo/onnx/ort_plugin/simo_qdq_ops.cc:301-386
     SimoQuantizeOp::Compute()
  -> simo/onnx/ort_plugin/triton_loader.cc:62-108
     TritonLoader::Launch()
  -> 嵌入 cubin 中的 _downcast_to_mxfmt
  -> 构建来源：simo/onnx/ort_plugin/build_qdq_cubins.py:119-161
     _downcast_to_mxfmt() ABI wrapper
  -> 算法来源：simo/ops/kernels/downcast/_downcast_to_mxfmt.py:525-694
     _downcast_to_mxfmt()
  -> simo/ops/kernels/downcast/_downcast_to_mxfmt.py:234-522
     _compute_and_pack_mxfmt()
```

`simo/onnx/ort_plugin/simo_qdq_ops.cc:301-386` 的 `SimoQuantizeOp::Compute()` 具体负责：

1. 在 `:306-323` 检查 rank-2、K 维和 packing 条件；
2. 在 `:323-327` 分配 `[128,32]` q tensor 和 `[128,1]` scale tensor；
3. 在 `:331-343` 验证输入输出是 CUDA device pointer；
4. 在 `:350-364` 组装 Triton host ABI 参数；
5. 在 `:365-382` 计算 grid，并调用 `TritonLoader::Launch()`，固定 thread block 为 256 threads。

第一次在当前 CUDA context 使用该 cubin 时，`simo/onnx/ort_plugin/triton_loader.cc:28-60` 的 `TritonLoader::EnsureKernelLoadedLocked()` 在 `:48-58` 调用：

```text
cuModuleLoadData(cubin bytes)
cuModuleGetFunction("_downcast_to_mxfmt")
```

随后 `simo/onnx/ort_plugin/triton_loader.cc:62-108` 的 `TritonLoader::Launch()` 在 `:94-107` 用 `cuLaunchKernel()` 将 kernel 提交到 ORT 提供的 CUDA stream。后续调用会复用该 CUDA context 下缓存的 module/function。

### 41.8 Quantize 的真正数学实现在哪里

Quantize 的基础 GPU 实现位于：

```text
simo/ops/kernels/downcast/_downcast_to_mxfmt.py:525-694
_downcast_to_mxfmt()
```

该 launcher kernel 在 `simo/ops/kernels/downcast/_downcast_to_mxfmt.py:662-672` 调用纯计算函数：

```text
simo/ops/kernels/downcast/_downcast_to_mxfmt.py:234-522
_compute_and_pack_mxfmt()
```

对一行中的 32 个值组成的 block，它执行：

1. `simo/ops/kernels/downcast/_downcast_to_mxfmt.py:251-264` 的 `_compute_and_pack_mxfmt()` 将 tile reshape 成 32 元素 block，并以 `abs_max` 求 `amax`；
2. `simo/ops/kernels/downcast/_downcast_to_mxfmt.py:283-292` 按 `e8m0_floor` 计算共享指数。`simo/ops/kernels/downcast/_downcast_to_mxfmt.py:37-61` 的 `_get_mx_format_info()` 给 MXINT8 的 `max_quant_exp=0`，因此可写成：

```text
e = clamp(floor(log2(amax)), -127, 127)
S = 2^e
```

3. `simo/ops/kernels/downcast/_downcast_to_mxfmt.py:369-380` 的 `_compute_and_pack_mxfmt()` 形成 `quant_scale=(1/S)*2^6`，即先计算 `x * 64 / S`；
4. `simo/ops/kernels/downcast/_downcast_to_mxfmt.py:387-397` 将 `S` 的 biased exponent 保存为一个 `uint8` E8M0 scale；
5. `simo/ops/kernels/downcast/_downcast_to_mxfmt.py:399-413` 对 `x * 64 / S` 做 round-to-nearest-even，clamp 到 `[-127,127]`，再转换为 `int8`；
6. `simo/ops/kernels/downcast/_downcast_to_mxfmt.py:484-522` 不对 MXINT8 额外压缩，每个逻辑值仍占一个 byte；
7. `simo/ops/kernels/downcast/_downcast_to_mxfmt.py:674-694` 的 `_downcast_to_mxfmt()` 把 scale 和 q bytes 写回 global memory。

忽略异常值和指数边界的特殊情况，核心公式是：

```text
q = int8(clamp(RNE(x * 64 / S), -127, 127))
```

ORT shape inference 把 q 声明成 `uint8`，而 Triton 计算值是 `int8`。这里 `uint8` 是 ORT 图中的“一字节原始 bit pattern 容器”：负 `int8` 的 two's-complement bits 原样保存。`simo/onnx/ort_plugin/build_qdq_cubins.py:103-110` 的 `_tensor_signature()` 明确为 MXINT8 cubin 使用 `*i8` pointer signature。

### 41.9 Dequantize 节点的运行时调用链

由于 Dequantize 消费 Quantize 的两个输出，ORT 在 Quantize kernel 完成并满足同一 stream 的依赖后执行：

```text
simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:836
test_simo_custom_qdq_plugin_runs_mx_qdq_tiny_tensor(): session.run()
  -> ORT 根据 com.simo + Dequantize + CUDAExecutionProvider 找到 custom op
  -> simo/onnx/ort_plugin/simo_qdq_ops.cc:395-398
     SimoDequantizeOp::SimoDequantizeOp(): 解析属性并选中 simo_dequantize_mxint8 spec
  -> simo/onnx/ort_plugin/simo_qdq_ops.cc:404-495
     SimoDequantizeOp::Compute()
  -> simo/onnx/ort_plugin/triton_loader.cc:62-108
     TritonLoader::Launch()
  -> 嵌入 cubin 中的 _upcast_from_mxfmt
  -> 构建来源：simo/onnx/ort_plugin/build_qdq_cubins.py:164-198
     _upcast_from_mxfmt() ABI wrapper
  -> 算法来源：simo/ops/kernels/upcast/_upcast_from_mxfmt.py:205-331
     _upcast_from_mxfmt()
  -> simo/ops/kernels/upcast/_upcast_from_mxfmt.py:18-202
     _unpack_and_dequant_mxfmt()
```

`simo/onnx/ort_plugin/simo_qdq_ops.cc:404-495` 的 `SimoDequantizeOp::Compute()` 具体负责：

1. 在 `:409-428` 从 `[128,32]` packed shape 还原 logical shape 并计算 scale shape；
2. 在 `:429-434` 验证 scale 必须为 `[128,1]`，分配 `[128,32]` float output；
3. 在 `:439-452` 验证 CUDA pointers；
4. 在 `:459-473` 组装与 Quantize 相同布局的 host ABI 参数；
5. 在 `:474-491` 以 grid `(1,1)`、block `(256,1,1)` 调用 `TritonLoader::Launch()`。

第一次加载时，`simo/onnx/ort_plugin/triton_loader.cc:28-60` 的 `TritonLoader::EnsureKernelLoadedLocked()` 从 Dequantize cubin 查找 `_upcast_from_mxfmt`，再由 `simo/onnx/ort_plugin/triton_loader.cc:62-108` 的 `TritonLoader::Launch()` 通过 `cuLaunchKernel()` 启动。

### 41.10 Dequantize 的真正数学实现在哪里

Dequantize 的基础 GPU 实现位于：

```text
simo/ops/kernels/upcast/_upcast_from_mxfmt.py:205-331
_upcast_from_mxfmt()
```

其处理过程是：

1. `simo/ops/kernels/upcast/_upcast_from_mxfmt.py:245-316` 的 `_upcast_from_mxfmt()` 读取 q bytes 和每 32 元素一个的 scale byte；
2. `simo/ops/kernels/upcast/_upcast_from_mxfmt.py:318-325` 调用 `_unpack_and_dequant_mxfmt()`；
3. `simo/ops/kernels/upcast/_upcast_from_mxfmt.py:140-149` 的 `_unpack_and_dequant_mxfmt()` 将 q 的 byte bitcast 回有符号 `int8`，再转换成输出浮点类型；
4. `simo/ops/kernels/upcast/_upcast_from_mxfmt.py:174-187` 从 E8M0 byte 恢复 `S=2^e`，reshape 后广播到对应的 32 个元素；
5. `simo/ops/kernels/upcast/_upcast_from_mxfmt.py:189-200` 对 MXINT8 乘上隐含因子 `2^-6`，再与 q 相乘；
6. `simo/ops/kernels/upcast/_upcast_from_mxfmt.py:326-331` 的 `_upcast_from_mxfmt()` 将 `float32` 结果写回输出。

对应核心公式是：

```text
x_dequant = float(int8(q_bits)) * S * 2^-6
          = float(int8(q_bits)) * S / 64
```

这与 Quantize 中的 `x * 64 / S` 成对。

### 41.11 测试 reference 路径与 plugin 路径的关系

plugin 路径执行完后，`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:841-846` 的 `test_simo_custom_qdq_plugin_runs_mx_qdq_tiny_tensor()` 调用 `_reference_mx_qdq()`。

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:135-154` 的 `_reference_mx_qdq()`：

1. 在 `:140-147` 调用 `_downcast_to_mxfmt_triton()`；
2. 在 `:149-150` 调用 `_upcast_from_mxfmt_triton()`；
3. 在 `:151-154` 同步 GPU 并返回 NumPy output。

`simo/ops/kernels/mx_trition_api.py:84-195` 的 `_downcast_to_mxfmt_triton()` 在 `:166-191` 直接 launch `simo/ops/kernels/downcast/_downcast_to_mxfmt.py:525-694` 的 `_downcast_to_mxfmt()`。

`simo/ops/kernels/mx_trition_api.py:198-276` 的 `_upcast_from_mxfmt_triton()` 在 `:259-276` 直接 launch `simo/ops/kernels/upcast/_upcast_from_mxfmt.py:205-331` 的 `_upcast_from_mxfmt()`。

两条路径的区别是：

```text
reference:
Python/PyTorch 分配 tensor -> Triton Python launcher -> 基础 Triton kernel

ORT plugin:
ORT 分配 tensor -> C++ custom op -> CUDA Driver API -> 构建期嵌入的同源 Triton cubin
```

因此 `assert_allclose(rtol=1e-3, atol=1e-3)` 通过，说明 C++ plugin 选择了正确的 MXINT8/32 cubin，并且 shape、scale layout、pointer ABI、grid、stream 和数值语义都与 Python/Triton reference 一致。

### 41.12 最简调用链总结

Quantize：

```text
make_node("Quantize", domain="com.simo")
-> RegisterCustomOps()
-> RegisterQdqOps(): "Quantize" -> SimoQuantizeOp
-> SimoQuantizeOp::Compute()
-> TritonLoader::Launch()
-> cuModuleLoadData()/cuModuleGetFunction()
-> cuLaunchKernel("_downcast_to_mxfmt")
-> _downcast_to_mxfmt()
-> _compute_and_pack_mxfmt()
```

Dequantize：

```text
make_node("Dequantize", domain="com.simo")
-> RegisterCustomOps()
-> RegisterQdqOps(): "Dequantize" -> SimoDequantizeOp
-> SimoDequantizeOp::Compute()
-> TritonLoader::Launch()
-> cuModuleLoadData()/cuModuleGetFunction()
-> cuLaunchKernel("_upcast_from_mxfmt")
-> _upcast_from_mxfmt()
-> _unpack_and_dequant_mxfmt()
```

两个真正的核心算法位置分别是：

```text
Quantize:
simo/ops/kernels/downcast/_downcast_to_mxfmt.py:234-522
_compute_and_pack_mxfmt()

Dequantize:
simo/ops/kernels/upcast/_upcast_from_mxfmt.py:18-202
_unpack_and_dequant_mxfmt()
```

## 42. `RegisterCustomOps` 是谁调用的，为什么普通函数会被自动调用

### 42.1 直接结论

`simo/onnx/ort_plugin/custom_op_library.cc:23-24` 的 `RegisterCustomOps()` 不是由 SIMO 的某个 C++ 调用点显式调用的。它是 **ONNX Runtime custom-op shared-library ABI 规定的入口函数**。

在当前代码路径中，真正的调用者是 ONNX Runtime 核心（通常位于外部的 `libonnxruntime.so`）：

```text
Python/用户代码调用 SessionOptions.register_custom_ops_library(path)
  -> ONNX Runtime 的 RegisterCustomOpsLibrary_V2
  -> ORT 动态加载 .so（Linux 下相当于 dlopen）
  -> ORT 按固定名字查找 RegisterCustomOps（Linux 下相当于 dlsym）
  -> ORT 通过函数指针调用 RegisterCustomOps(options, api)
```

所以“普通函数会被调用”的原因不是 C++ 编译器的特殊规则，而是：

```text
固定的导出符号名 + 固定的 C ABI 函数签名
```

共同构成了 ORT 与 custom-op `.so` 之间的插件协议。

### 42.2 本项目中谁触发了这次调用

#### 测试路径

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:56-85` 的 `_session_options_with_simo_plugin()` 在 `:76-78` 执行：

```python
options = ort.SessionOptions()
options.register_custom_ops_library(str(lib_path))
```

这行 Python 调用是当前测试中最直接的触发点。它发生在 `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:372-376` 的 `ort.InferenceSession(...)` 之前，因此 `RegisterCustomOps()` 是“注册 custom-op library”阶段调用的，不是 `session.run()` 执行 Quantize 时才调用的。

#### SIMO public API 路径

如果用户使用 SIMO 的 public API：

1. `simo/onnx/runtime.py:134-137` 的 `register_custom_ops()` 调用 `sess_options.register_custom_ops_library(str(path))`；
2. `simo/onnx/runtime.py:140-150` 的 `create_session()` 在 `:149` 调用 `register_custom_ops(options)`，然后在 `:150` 创建 `ort.InferenceSession`。

因此 public API 的调用链是：

```text
simo/onnx/runtime.py:140-150 create_session()
  -> simo/onnx/runtime.py:134-137 register_custom_ops()
  -> onnxruntime Python binding: SessionOptions.register_custom_ops_library()
  -> ONNX Runtime C API: RegisterCustomOpsLibrary_V2()
  -> custom_op_library.cc:RegisterCustomOps()
```

Python binding 的实现位于当前 SIMO code base 之外，但 SIMO 随附的 ORT C API 头文件把这个 ABI 行为写得很明确：

`simo/onnx/ort_plugin/include/onnxruntime/core/session/onnxruntime_c_api.h:5353-5377` 对 `RegisterCustomOpsLibrary_V2` 的契约是：

1. 加载指定的 `.dll` / `.so`；
2. 查找名字为 `RegisterCustomOps` 的入口；
3. 把 `OrtSessionOptions*` 和 `OrtApiBase*` 传给它。

旧版 `RegisterCustomOpsLibrary` 的说明在 `simo/onnx/ort_plugin/include/onnxruntime/core/session/onnxruntime_c_api.h:1871-1893`，同样明确写着要查找并调用这个入口。

如果通过 C++ wrapper 调用，`simo/onnx/ort_plugin/include/onnxruntime/core/session/onnxruntime_cxx_inline.h:2063-2072` 的 `SessionOptionsImpl::RegisterCustomOpsLibrary()` 在 `:2071` 调用 `GetApi().RegisterCustomOpsLibrary_V2(...)`。Python binding 走的是其自己的封装，但最终使用同一个 ORT API 语义。

### 42.3 `RegisterCustomOps` 必须具备哪些特征

#### 1. 符号名必须精确是 `RegisterCustomOps`

`simo/onnx/ort_plugin/custom_op_library.cc:23-24` 的声明使用了：

```cpp
extern "C" ORT_EXPORT OrtStatus* ORT_API_CALL
RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api)
```

其中 `extern "C"` 禁止 C++ name mangling。没有它，C++ 编译器可能把函数名变成类似 `_Z18RegisterCustomOps...`，ORT 用字符串 `RegisterCustomOps` 查找时就找不到。

#### 2. 必须导出到动态库的动态符号表

`simo/onnx/ort_plugin/include/onnxruntime/core/session/onnxruntime_c_api.h:88-115` 定义了平台相关宏：

```text
Linux/GCC: ORT_EXPORT = __attribute__((visibility("default")))
Linux/GCC: ORT_API_CALL 为空
Windows:   ORT_API_CALL = __stdcall
```

因此 `ORT_EXPORT` 让 Linux 下的符号具有默认可见性；`ORT_API_CALL` 保证跨平台调用约定一致。

此外，`simo/onnx/ort_plugin/custom_op_library.lds:1-6` 的 linker version script 只把 `RegisterCustomOps` 放入 global exports，其余符号放入 local：

```text
global:
  RegisterCustomOps;
local:
  *;
```

`simo/onnx/ort_plugin/build_runtime.py:70-88` 的 `build_sm90_runtime()` 在 `:74` 把这个 `.lds` 作为 `-Wl,--version-script=...` 传给 host shared-library linker。也就是说，`ORT_EXPORT` 和 version script 两层共同保证 ORT 能从 `.so` 的动态符号表找到入口。

当前环境实际构建出的库可以用 `nm -D` 验证，导出的符号是：

```text
RegisterCustomOps@@VERS_1.0.0
```

#### 3. 函数签名必须符合 ORT typedef

`simo/onnx/ort_plugin/include/onnxruntime/core/session/onnxruntime_c_api.h:1155-1156` 定义了 `RegisterCustomOpsFn`：

```cpp
typedef OrtStatus*(ORT_API_CALL* RegisterCustomOpsFn)(
    OrtSessionOptions* options, const OrtApiBase* api);
```

`simo/onnx/ort_plugin/custom_op_library.cc:23-24` 的函数正好符合这个签名。ORT 查找到符号后会把它当成这个函数指针类型调用；它不是按 C++ 类型反射或按函数体内容识别的。

#### 4. 返回值必须遵守 `OrtStatus*` 协议

`simo/onnx/ort_plugin/custom_op_library.cc:40-50` 的 `RegisterCustomOps()`：

1. 成功时 `:46` 返回 `nullptr`；
2. C++ exception 在 `:47-50` 转换成 `OrtStatus*` 返回给 ORT；
3. `:25-38` 先尝试当前 `ORT_API_VERSION`，再向下兼容寻找旧版本的 `OrtApi`。

因此 ORT 不需要知道 SIMO 内部的 `RegisterQdqOps()`；它只需要知道这个入口的名字、签名和错误返回约定。

### 42.4 ORT 调用 `RegisterCustomOps` 时，函数内部做了什么

`simo/onnx/ort_plugin/custom_op_library.cc:23-51` 的 `RegisterCustomOps()` 是一个很薄的 adapter，调用链如下：

```text
RegisterCustomOps(options, api)
  -> :25-38 选择兼容的 OrtApi 版本
  -> :39 Ort::InitApi(ort_api)
  -> :41 创建 domain "com.simo"
  -> :42 simo::onnx::RegisterQdqOps(domain, ort_api_version)
  -> :43-44 domain 加入传入的 OrtSessionOptions
  -> :45 KeepDomainAlive(std::move(domain))
```

`simo/onnx/ort_plugin/simo_qdq_ops.cc:504-515` 的 `RegisterQdqOps()` 才会把两个算子加入 domain：

```text
"Quantize"   + "CUDAExecutionProvider" -> SimoQuantizeOp
"Dequantize" + "CUDAExecutionProvider" -> SimoDequantizeOp
```

`simo/onnx/ort_plugin/custom_op_library.cc:14-19` 的 `KeepDomainAlive()` 把 `Ort::CustomOpDomain` 保存到静态 vector 中。这样做不是为了让 ORT 找到 `RegisterCustomOps`，而是为了满足 ORT custom-domain 生命周期要求：

`simo/onnx/ort_plugin/include/onnxruntime/core/session/onnxruntime_c_api.h:1855-1859` 说明，加入 session options 的 custom-op domain 在所有使用它的 session 释放前不能被删除。

### 42.5 为什么 SIMO 源码中搜索不到“调用 RegisterCustomOps”的地方

搜索结果中只有定义：

```text
simo/onnx/ort_plugin/custom_op_library.cc:24:RegisterCustomOps(...)
```

这是预期行为，因为调用点在外部 ORT 二进制内部，不在 SIMO repository 中。SIMO repository 中能看到的是：

```text
simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:78
  options.register_custom_ops_library(...)

simo/onnx/runtime.py:136
  sess_options.register_custom_ops_library(...)

simo/onnx/ort_plugin/include/onnxruntime/core/session/onnxruntime_cxx_inline.h:2071
  GetApi().RegisterCustomOpsLibrary_V2(...)
```

之后的 `dlopen`、`dlsym` 和函数指针调用由 `libonnxruntime.so` 完成，所以不会在 `rg` 搜索 SIMO `.cc` 时出现一个普通的 `RegisterCustomOps(...)` 调用表达式。

### 42.6 它不是 C++ 静态初始化函数，也不是按 ONNX 节点名自动执行的函数

需要区分三个时间点：

```text
加载 .so 本身
  -> 由动态链接器加载 ELF；不等于调用 RegisterCustomOps

register_custom_ops_library(path)
  -> ORT 找到并调用 RegisterCustomOps；完成 custom-op 注册

InferenceSession.run()
  -> 执行已经注册的 Quantize/Dequantize kernel
```

`RegisterCustomOps()` 不会因为 `.so` 被 `dlopen` 就依靠 C++ 静态初始化自动执行；它是 ORT 显式通过导出符号查找后调用的插件入口。它也不会针对每一个 ONNX node 重复调用；node 的 `com.simo::Quantize` / `com.simo::Dequantize` 匹配发生在注册完成之后。

### 42.7 另一个 ORT 机制：按函数名注册，但 SIMO 当前没有使用

`simo/onnx/ort_plugin/include/onnxruntime/core/session/onnxruntime_c_api.h:5379-5409` 描述了 `RegisterCustomOpsUsingFunction()`：

```text
给 ORT 一个 registration_func_name；
ORT 在已经链接/加载的库中搜索该名字并调用它。
```

SIMO 选择的是 `RegisterCustomOpsLibrary_V2()` 的“给路径、由 ORT 加载并管理生命周期”模式，而不是 `RegisterCustomOpsUsingFunction()` 的“库已由应用加载、按名字搜索”模式。两种模式都依赖同一个 `RegisterCustomOpsFn` 签名，但当前 SIMO 的触发点是 `register_custom_ops_library(path)`。

### 42.8 完整调用图与故障含义

```text
simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:56-85
_session_options_with_simo_plugin()
  -> options.register_custom_ops_library(path)
  -> ONNX Runtime RegisterCustomOpsLibrary_V2
  -> dlopen(path)
  -> dlsym("RegisterCustomOps")
  -> RegisterCustomOps(options, api)
  -> simo/onnx/ort_plugin/custom_op_library.cc:41-45
     创建并加入 com.simo domain
  -> simo/onnx/ort_plugin/simo_qdq_ops.cc:504-515
     注册 Quantize/Dequantize
  -> 后续 InferenceSession/session.run 才执行具体 op kernel
```

如果出现“找不到 `RegisterCustomOps`”类错误，优先检查以下几项：

1. `simo/onnx/ort_plugin/custom_op_library.cc:23-24` 是否保留 `extern "C"`；
2. `simo/onnx/ort_plugin/custom_op_library.cc:23` 是否有 `ORT_EXPORT`；
3. `simo/onnx/ort_plugin/custom_op_library.lds:1-6` 是否导出该精确名字；
4. `simo/onnx/ort_plugin/build_runtime.py:70-88` 是否真的把 version script 传给 linker；
5. 返回签名是否仍匹配 `simo/onnx/ort_plugin/include/onnxruntime/core/session/onnxruntime_c_api.h:1155-1156`。

最核心的一句话是：

```text
RegisterCustomOps 不是“普通函数碰巧被调用”，而是 ORT custom-op ABI
规定的动态库入口；ORT 通过导出的精确 C 符号名和固定函数签名找到并调用它。
```

## 43. `_qdq_unaligned_matmul_activation_qdq_model` 如何改写并量化 ONNX `MatMul`

### 43.1 先给结论

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:421-461` 的
`_qdq_unaligned_matmul_activation_qdq_model()` 构造一个 K 维为 18 的 `MatMul`，然后调用
SIMO 的 `insert_qdq_nodes()` 对它做 QDQ 图改写。

这里的“量化 MatMul”准确含义是：

1. 对第一个输入 `X` 插入**运行时动态 MXINT8 Quantize + Dequantize**；
2. 对常量权重 `W` 做**模型转换期 FP8 E4M3 per-block 离线量化**，在 ONNX 中保存量化权重和 scale，再在运行时插入 Dequantize；
3. 原来的标准 ONNX `MatMul` 节点仍然存在，输入被改接到两个反量化后的 `float32` tensor；
4. 它**没有**把节点替换成 `QLinearMatMul`、整数 GEMM、FP8 GEMM 或 SIMO 自定义 MatMul kernel。

所以最终矩阵乘法仍可写成：

```text
Y_quant = DQ_activation(Q_activation(X))
          @ transpose(DQ_weight(Q_weight(transpose(W))))
```

这里真正送入 `ai.onnx::MatMul` 的两边都是 `float32`。低精度误差由前面的 Q/DQ 引入，
而不是由低精度乘法累加 kernel 引入。

### 43.2 原始 ONNX 图和量化配置

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:421-430`，函数
`_qdq_unaligned_matmul_activation_qdq_model()` 创建的原始图只有一个计算节点：

```text
X: float32 [2, 3, 18] ----\
                              MatMul(name="matmul") -> Y: float32 [2, 3, 4]
W: float32 [18, 4] -------/
```

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:423` 创建确定性的权重 `W`；
`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:425` 用输入 `X`、`W` 创建标准域的
`MatMul`；`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:427-429` 分别声明输入、输出和
常量 initializer 的形状。因此这个 MatMul 的收缩维 K 是 18：

```text
[2, 3, 18] @ [18, 4] -> [2, 3, 4]
```

量化配置位于 `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:432-447`：

```json
{
  "targets_op_types": ["MatMul"],
  "input":  {"dtype": "mxint8"},
  "weight": {"dtype": "fp8_e4m3", "axis": [0, 1], "group_size": 128}
}
```

各字段的实际含义是：

- `targets_op_types=["MatMul"]`：只匹配 `MatMul`；配置归一化发生在
  `simo/onnx/onnx_quant.py:511-526` 的 `_normalize_module_config()`。
- `input.dtype=mxint8`：`simo/quantization/config.py:415-468` 的
  `QuantizeSpecMX` 默认 `axis=-1`、`block_size=32`，并由
  `sync_group_size_with_block_size()` 把 `group_size` 同步为 32。
- 输入配置没有显式写 `is_dynamic`，但
  `simo/onnx/onnx_quant.py:519-525` 的 `_normalize_module_config()` 会为输入补上
  `is_dynamic=True`，所以激活 scale 在每次推理时根据实际输入计算。
- `weight.dtype=fp8_e4m3, axis=[0,1], group_size=128`：
  `simo/quantization/config.py:215-237` 的 `get_quantize_granularity()` 把“两条 axis +
  group_size”的组合判定为 `per_block`，即二维 128 x 128 block 共用一个 scale。

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:280-282` 的
`_insert_qdq_nodes_with_native_weight_quant()` 先要求 `simo._C` 可用，再进入
`simo/onnx/onnx_quant.py:469-482` 的 `insert_qdq_nodes()`。后者加载配置、加入
`com.simo` opset，并调用 `_insert_qdq_in_graph()` 改写图。

### 43.3 改写后的完整节点图

实际生成的 `unaligned_matmul_activation_qdq.onnx` 节点顺序如下：

```text
激活分支：
X [2,3,18]
  -> Reshape                         [6,18]
  -> Pad(last dim +14)               [6,32]
  -> com.simo::Quantize(mxint8)      q=[6,32], scale=[6,1]
  -> com.simo::Dequantize(mxint8)    float=[6,32]
  -> Slice(axis=1, end=18)           [6,18]
  -> Reshape                         [2,3,18]
  -> matmul 的 input[0]

权重分支：
原始 W [18,4]
  -> 转换期先按逻辑权重 W^T=[4,18] 离线 FP8 per-block 量化
     （[4,18] 是一个不完整的 128x128 边界 block；kernel 用 mask 处理，不做物理 padding）
  -> initializer matmul_W_simo_q     uint8 carrier [4,18]  （正确，不是 [128,128]）
  -> initializer matmul_W_simo_scale uint8 carrier [1,4]   （逻辑上是 float32 [1,1]）
  -> com.simo::Dequantize            float=[4,18]
  -> Transpose                       float=[18,4]
  -> matmul 的 input[1]

最终：
ai.onnx::MatMul([2,3,18], [18,4]) -> Y [2,3,4]
```

节点名及连接关系为：

```text
matmul_input_simo_reshape
  -> matmul_input_simo_pad
  -> matmul_input_simo_quant
  -> matmul_input_simo_dequant
  -> matmul_input_simo_unpad
  -> matmul_input_simo_restore
  -> matmul(input[0])

matmul_weight_simo_dequant
  -> matmul_weight_simo_transpose
  -> matmul(input[1])
```

原始节点本身只发生两项连接修改：

```text
改写前：MatMul.input = ["X", "W"]
改写后：MatMul.input = [
  "matmul_input_simo_restore",
  "matmul_weight_simo_weight_dq_transpose"
]
```

它的 `op_type="MatMul"`、默认 ONNX domain、名字 `matmul` 和输出 `Y` 均未改变。

### 43.4 激活 `X` 的 MXINT8 动态 QDQ 如何实现

#### 第一步：为 rank-3 MatMul 输入制订 rank-2 QDQ 方案

`simo/onnx/onnx_quant.py:824-922` 的 `_plan_activation_qdq()` 负责制订计划。
对已知形状 `[2,3,18]`：

1. `simo/onnx/onnx_quant.py:906-911` 对 `MatMul/Gemm` 固定选择最后一维作为量化轴；
2. `simo/onnx/onnx_quant.py:1113-1118` 的 `_rank2_quant_group()` 对 MXINT8 返回 32；
3. `simo/onnx/layout_utils.py:54-91` 的 `build_layout_recipe()` 得到：
   `axis=2`、`original_k=18`、`align_to=32`、`pad_len=14`、`padded_k=32`；
4. 最后一维已经是量化轴，因此不需要 `Transpose`，但 rank-3 输入需要展平为 custom op
   所支持的 rank-2 布局。

#### 第二步：展平为 `[6,18]`

`simo/onnx/onnx_quant.py:121-151` 的 `ONNXTensorTransformer.activation_qdq()` 调用
`simo/onnx/onnx_quant.py:153-223` 的 `forward_rank2()`。

`forward_rank2()` 创建：

```text
matmul_input_simo_qdq_shape     = [-1, 18]
matmul_input_simo_restore_shape = [2, 3, 18]
Reshape(X, [-1,18])             -> [6,18]
```

必须先展平，是因为 `simo/onnx/ort_plugin/simo_qdq_ops.cc:215-260` 的
`QuantizeShape()` 和 `simo/onnx/ort_plugin/simo_qdq_ops.cc:301-317` 的
`SimoQuantizeOp::Compute()` 都要求 custom Quantize 的输入是连续 rank-2 tensor。

#### 第三步：把 K=18 补零到 K=32

`simo/onnx/onnx_quant.py:225-280` 的 `ONNXTensorTransformer.rank2_qdq()` 检测到
`18 % 32 != 0`，创建以下常量和节点：

```text
pads   = [0, 0, 0, 14]  # rank-2 Pad: 前两项是 begin，后两项是 end
starts = [0]
ends   = [18]
axes   = [1]
steps  = [1]

[6,18] --Pad--> [6,32] --Q--> --DQ--> [6,32] --Slice--> [6,18]
```

补零并非为了改变 MatMul 的数学 K，而是满足 MX kernel 的输入契约。
`simo/onnx/ort_plugin/simo_qdq_ops.cc:121-130` 的 `ValidateQuantDim()` 明确要求 MX
`quant_dim % block_size == 0`。反量化后立即切掉 14 个补零元素，所以送给 MatMul 的 K
仍为 18。

#### 第四步：插入 `com.simo::Quantize/Dequantize`

`simo/onnx/onnx_quant.py:949-974` 的 `_create_qdq_nodes()` 创建两个 custom op：

```text
com.simo::Quantize(
  dtype="mxint8", granularity="per_group",
  axis=-1, group_size=32, block_size=32,
  scale_mode="e8m0_floor", observer_mode="abs_max"
)

com.simo::Dequantize(同一组语义属性)
```

对 `[6,32]` 输入，每一行恰好一个 32 元素 MX block，所以 Quantize 产生 6 组 scale。
量化值和 scale 通过 ONNX custom-op ABI 以 `uint8` tensor 承载；Dequantize 再恢复成
`float32 [6,32]`。

#### 第五步：恢复原始形状并改接 MatMul

`simo/onnx/onnx_quant.py:282-306` 的 `restore_from_rank2()` 把切片后的 `[6,18]`
恢复为 `[2,3,18]`。随后
`simo/onnx/onnx_quant.py:783-821` 的 `_create_activation_qdq_nodes()` 在第 820 行把
`matmul.input[0]` 从 `X` 改成 `matmul_input_simo_restore`。

因此激活侧实现的是：

```text
X_qdq = reshape_back(
          slice_K18(
            DQ_mxint8(Q_mxint8(pad_K32(reshape_2d(X))))
          )
        )
```

### 43.5 权重 `W` 的 FP8 per-block 量化如何实现

权重侧与激活侧不同：权重是 initializer，因此 Quantize 在模型转换期执行，生成后的
ONNX 图中只有权重 `Dequantize`，没有权重 `Quantize` 节点。

#### 第一步：把 MatMul 权重转成量化所用的逻辑布局

`simo/onnx/weight_quant.py:32-48` 的 `logical_weight_view()` 对 `MatMul` 返回 `weight.T`：

```text
原始 ONNX 权重 W: [18,4]
量化逻辑权重 W^T: [4,18]
```

这样逻辑布局的第一维对应输出通道，之后再用 ONNX `Transpose` 恢复 MatMul 所需的
`[18,4]` 布局。

#### 第二步：转换期在 CUDA 上离线量化

调用链为：

```text
simo/onnx/onnx_quant.py:699-757
_find_qdq_target()
  -> simo/onnx/weight_quant.py:51-69
     quantize_weight_array()
  -> simo/quantization/kernels.py:21-92
     get_downcast_kernel()
  -> simo/quantization/kernels.py:55-63
     torch.ops.simo.per_block_downcast_to_fp8_or_int8(...)
  -> simo/ops/flex_api.py:74-106
     per_block_downcast_to_fp8_or_int8_cuda_impl()
  -> simo/ops/kernels/downcast/_downcast_to_flexpoint.py:112-164
     per_block_downcast_to_fp8_or_int8_triton()
  -> simo/ops/kernels/downcast/_downcast_to_flexpoint.py:167-211
     _per_block_quant_fp8_or_int8_kernel
```

`simo/onnx/weight_quant.py:57-68` 的 `quantize_weight_array()` 把逻辑权重复制成 CUDA
`float32` tensor，并调用由配置选择出的 downcast kernel。对于 `[4,18]` 和
`group_size=128`，二维 tensor 整体落在唯一一个 128 x 128 边界 block 内，因此只计算
一个 FP32 scale。这里的 block 是 scale 的逻辑分组，不要求把存储形状扩展为
`[128,128]`：`simo/ops/kernels/downcast/_downcast_to_flexpoint.py:128-134` 的
`per_block_downcast_to_fp8_or_int8_triton()` 用 `ceil(4/128) x ceil(18/128) = 1 x 1`
计算 scale 形状；`simo/ops/kernels/downcast/_downcast_to_flexpoint.py:190-211` 的
`_per_block_quant_fp8_or_int8_kernel` 使用 `row_offsets < rows` 和 `col_offsets < cols`
的 mask 读写边界 block。因此量化值仍保持 `[4,18]`，没有权重 `Pad/Slice` 节点。

生成的两个 initializer 是：

```text
matmul_W_simo_q:     uint8 [4,18]  # FP8 E4M3 位模式的 byte carrier
matmul_W_simo_scale: uint8 [1,4]   # 一个 FP32 scale 的 4 个原始字节
```

上面两个形状都是正确的。scale 的数学/逻辑形状是
`[ceil(4/128), ceil(18/128)] = [1,1] float32`；它在 ONNX initializer 中看起来是
`[1,4] uint8`，是因为
`simo/onnx/weight_quant.py:137-163` 的 `_make_quantized_weight()` / `_as_uint8_numpy()`
把非 `uint8` tensor 统一 view 成 byte carrier。运行时
`simo/onnx/ort_plugin/simo_qdq_ops.cc:101-119` 的 `ScaleCols()` 和
`ScaleByteCols()` 会按 `sizeof(float)` 解释 flex scale，所以一个 scale 对应 4 个字节。

#### 第三步：在 ONNX 中插入权重 Dequantize 和 Transpose

`simo/onnx/onnx_quant.py:977-1031` 的 `_create_weight_dq_nodes()`：

1. 把量化权重和 scale 加入 `graph.initializer`（第 1007-1010 行）；
2. 创建 `matmul_weight_simo_dequant`，输出 `float32 [4,18]`（第 1017-1025 行）；
3. `simo/onnx/onnx_quant.py:1034-1037` 的 `_needs_weight_transpose()` 对 MatMul 返回
   `True`，因此再创建 `Transpose(perm=[1,0])`，恢复成 `[18,4]`；
4. 在 `simo/onnx/onnx_quant.py:1011` 把 `matmul.input[1]` 改接到上述输出。

权重 Dequantize 的关键属性为：

```text
dtype="fp8_e4m3"
granularity="per_block"
axes=[0,1]
group_size=128
original_shape=[18,4]
logical_shape=[4,18]
```

原始 `W` 已不再被任何节点使用，最后由
`simo/onnx/onnx_quant.py:1172-1198` 的 `_remove_unused_values()` 从 initializer 列表删除。

### 43.6 推理时哪些 kernel 真正执行

`simo/onnx/onnx_quant.py:608-669` 的 `_insert_qdq_in_graph()` 只是构造 ONNX 图；真正运行
`com.simo::Quantize/Dequantize` 的是 ORT custom-op plugin：

```text
simo/onnx/ort_plugin/simo_qdq_ops.cc:504-515
RegisterQdqOps()
  -> 注册 SimoQuantizeOp / SimoDequantizeOp 到 CUDAExecutionProvider

激活 Quantize：
simo/onnx/ort_plugin/simo_qdq_ops.cc:290-391
SimoQuantizeOp::Compute()
  -> ResolveSpec(mxint8, per_group, group=32, block=32)
  -> ResolveQdqRuntimeSpecSm90()
  -> TritonLoader::Launch()
  -> 内嵌 simo_quantize_mxint8 cubin

激活 Dequantize：
simo/onnx/ort_plugin/simo_qdq_ops.cc:393-500
SimoDequantizeOp::Compute()
  -> 内嵌 simo_dequantize_mxint8 cubin

权重 Dequantize：
SimoDequantizeOp::Compute()
  -> ResolveSpec(fp8_e4m3, per_block, group=128)
  -> 内嵌 simo_dequantize_fp8_e4m3_per_block cubin
```

MXINT8 cubin 的构建入口在
`simo/onnx/ort_plugin/build_qdq_cubins.py:758-807` 的 `build()`；其 Quantize/Dequantize
Triton wrapper 分别由 `simo/onnx/ort_plugin/build_qdq_cubins.py:350-381` 的
`_compile_quant()` / `_compile_dequant()` 编译。FP8 per-block Dequantize 由
`simo/onnx/ort_plugin/build_qdq_cubins.py:442-453` 的 `_compile_per_block_dequant()` 生成，
并在 `simo/onnx/ort_plugin/build_qdq_cubins.py:843-884` 的 `build()` 中加入运行时表。

`simo/onnx/ort_plugin/triton_loader.cc:28-59` 的
`TritonLoader::EnsureKernelLoadedLocked()` 用 CUDA Driver API 加载内嵌 cubin 并取得函数；
`simo/onnx/ort_plugin/triton_loader.cc:62-108` 的 `TritonLoader::Launch()` 最终调用
`cuLaunchKernel()`。

这些 kernel 只负责 Q/DQ。最后的 `ai.onnx::MatMul` 仍由 ONNX Runtime 的
`CUDAExecutionProvider` 处理，不经过 `RegisterQdqOps()`，也没有 SIMO MatMul kernel
调用链。

### 43.7 为什么是“unaligned”，padding 是否改变结果

“unaligned”特指激活 K=18 没有按 MXINT8 block size 32 对齐：

```text
pad_len = (-18) % 32 = 14
```

补入的 14 个零只参与最后一个 MX block 的 scale 计算。因为其绝对值为 0，不会增大该
block 的 `abs_max`；Q/DQ 后又被 `Slice` 删除，所以 MatMul 仍接收 18 个 K 元素。
padding 的作用是满足 kernel 内存布局和 block 契约，而不是把原 MatMul 改成 K=32。

需要注意，权重配置是普通 FP8 E4M3 per-block，而不是 MX 格式；其 per-block kernel
支持边界 mask，因此 `[4,18]` 不需要补成 `[128,128]`。这也是生成图中只有激活分支有
`Pad/Slice`，权重分支没有的原因。

### 43.8 这个 pytest 实际验证了什么，没有验证什么

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:448-461` 的 helper 在改写后：

1. 把中间结果 `matmul_input_simo_restore` 额外声明为 graph output；
2. 保存 ONNX；
3. 使用注册了 SIMO plugin 的 `SessionOptions` 和 `CUDAExecutionProvider` 创建 session。

对应测试是
`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:925-937` 的
`test_unaligned_matmul_activation_qdq_matches_simo_torch_with_padding()`。它运行时只取：

```python
session.run(["matmul_input_simo_restore"], {"X": tensor})
```

然后与 `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:157-163` 的
`_reference_mx_qdq_with_last_dim_padding()` 比较。reference 做的正是：

```text
[2,3,18] -> reshape [6,18] -> pad [6,32]
-> SIMO Torch MXINT8 Q/DQ -> slice [:,:18] -> reshape [2,3,18]
```

因此这个测试直接验证的是：

- rank-3 到 rank-2 的适配；
- K=18 到 K=32 的 padding；
- MXINT8 activation Quantize/Dequantize kernel；
- unpad 和原始形状恢复；
- ORT custom-op 动态库能够加载并创建 session。

它**没有读取或比较 `Y`**，因此不能把该测试视为 MatMul 最终输出、权重 FP8 数值或真正
低精度 GEMM 的数值正确性测试。session 创建会检查整张图和 custom-op 配置是否合法，
但当前断言只覆盖 `matmul_input_simo_restore`。若要验证完整结果，应额外读取 `Y`，并与
“MXINT8 QDQ 后的 X @ FP8 per-block QDQ 后的 W”参考结果比较。

### 43.9 一句话总结

```text
_qdq_unaligned_matmul_activation_qdq_model() 保留标准浮点 MatMul，
在 X 前插入带 K=18 -> 32 -> 18 适配的动态 MXINT8 QDQ，
在 W 前插入“离线 FP8 per-block 量化 initializer + 运行时 Dequantize + Transpose”；
它实现的是 QDQ fake-quantized MatMul 图，而不是真正用低精度输入执行的量化 MatMul kernel。
```

## 44. `simo/onnx/onnx_quant.py` 中 `_name_nodes()` 的功能

### 44.1 结论

`simo/onnx/onnx_quant.py:589-605` 的 `_name_nodes(graph)` 用于保证当前
`GraphProto` 中的每一个原始节点都有一个**非空且不重复**的 `node.name`：

- 已有非空且首次出现的名称保持不变；
- 没有名称的节点，以 `node.op_type` 为前缀生成名称，例如 `MatMul_1`；
- 后续出现的重名节点，以原名称为前缀生成名称，例如第二个 `encoder` 变成
  `encoder_1`；
- 生成名称会避开图中所有原有节点名称，包括尚未遍历到的节点名称。

它不改变 `op_type`、节点输入输出、initializer、图拓扑或计算结果，只修改
`NodeProto.name`。它的主要目的不是满足 MatMul 或 ONNX Runtime 的计算要求，而是为后续
SIMO QDQ 图改写提供稳定、唯一的命名前缀。

### 44.2 在量化流程中的调用位置

入口调用链是：

```text
simo/onnx/onnx_quant.py:469-482
insert_qdq_nodes()
  -> simo/onnx/onnx_quant.py:608-669
     _insert_qdq_in_graph()
       -> simo/onnx/onnx_quant.py:617
          _name_nodes(graph)
       -> 遍历并量化 graph.node
```

`simo/onnx/onnx_quant.py:608-627` 的 `_insert_qdq_in_graph()` 在收集权重、创建
`ONNXTensorTransformer` 和遍历节点之前调用 `_name_nodes()`。因此后续所有目标节点都已经
具有非空唯一名称。

### 44.3 逐行解释实现

源码位于 `simo/onnx/onnx_quant.py:589-605`，函数 `_name_nodes()`：

```python
def _name_nodes(graph: GraphProto) -> None:
  counts: dict[str, int] = {}
  reserved = {node.name for node in graph.node if node.name}
  seen: set[str] = set()
  for node in graph.node:
    if node.name and node.name not in seen:
      seen.add(node.name)
      continue
    base = node.name or node.op_type
    while True:
      counts[base] = counts.get(base, 0) + 1
      candidate = f"{base}_{counts[base]}"
      if candidate not in reserved and candidate not in seen:
        node.name = candidate
        reserved.add(candidate)
        seen.add(candidate)
        break
```

#### `counts`：记录每个前缀尝试到哪个序号

`simo/onnx/onnx_quant.py:590`：

```python
counts: dict[str, int] = {}
```

`counts` 按 `base` 分别计数。例如未命名的 `MatMul` 使用 `base="MatMul"`，重名的
`encoder` 使用 `base="encoder"`。序号从 1 开始，所以生成形式是 `MatMul_1`、
`MatMul_2` 或 `encoder_1`，不会生成无后缀的 `MatMul`。

#### `reserved`：预先保留图中全部原有非空名称

`simo/onnx/onnx_quant.py:591`：

```python
reserved = {node.name for node in graph.node if node.name}
```

这里不是一边遍历一边收集，而是在修改任何节点前扫描整个 `graph.node`。这样可以避免给
前面的未命名节点生成一个名称，却与后面原本已经存在的节点名称冲突。

例如图中第一个节点未命名，后面的节点已经叫 `MatMul_1`：

```text
若不预扫描：第一个节点可能抢占 MatMul_1
实际实现：  MatMul_1 已在 reserved 中，第一个节点改用 MatMul_2
```

#### `seen`：记录遍历过程中已经接受的名称

`simo/onnx/onnx_quant.py:592-596`：

```python
seen: set[str] = set()
if node.name and node.name not in seen:
  seen.add(node.name)
  continue
```

如果非空名称第一次出现，函数原样保留并加入 `seen`。如果同一名称再次出现，条件失败，
后一个重名节点进入重命名流程。因此重复名称的处理原则是：

```text
第一个保留原名，第二个及后续节点改名。
```

#### `base`：空名称用 op type，重名节点用原名称

`simo/onnx/onnx_quant.py:597`：

```python
base = node.name or node.op_type
```

两种情况分别为：

```text
node.name == ""       -> base = node.op_type，例如 "MatMul"
node.name == "layer" 但重名 -> base = "layer"
```

#### `while`：跳过所有已占用候选名称

`simo/onnx/onnx_quant.py:598-605` 的循环不断递增当前 `base` 的序号，直到候选名称既不在
`reserved` 中，也不在 `seen` 中。找到候选名称后：

1. 写入 `node.name`；
2. 加入 `reserved`，后续新名称不能再使用它；
3. 加入 `seen`，表示遍历中已经接受该名称；
4. `break` 结束当前节点的重命名。

### 44.4 一个同时包含空名称、重名和预留名称的例子

假设原始节点按下列顺序排列：

```text
序号  op_type  node.name
0     MatMul   ""
1     Add      "keep"
2     Relu     "keep"
3     MatMul   "MatMul_1"
4     MatMul   ""
```

`simo/onnx/onnx_quant.py:591` 首先得到：

```text
reserved = {"keep", "MatMul_1"}
```

运行 `simo/onnx/onnx_quant.py:589-605` 的 `_name_nodes()` 后：

```text
序号  op_type  修改后的 node.name  原因
0     MatMul   "MatMul_2"          MatMul_1 已被后面的原名预留
1     Add      "keep"              keep 首次出现，保留
2     Relu     "keep_1"            keep 重复，生成新名称
3     MatMul   "MatMul_1"          原有名称首次出现，保留
4     MatMul   "MatMul_3"          MatMul_1、MatMul_2 都已占用
```

相同图再次调用 `_name_nodes()` 时所有名称已经非空且唯一，因此不会再改变；也就是说，
在输入图未变化的前提下，该函数是幂等的。

### 44.5 为什么 QDQ 改写依赖唯一的 `node.name`

`simo/onnx/onnx_quant.py:699-757` 的 `_find_qdq_target()` 在第 743-756 行创建
`OpTarget`，并在第 746 行保存：

```python
node_name=node.name
```

此后大量新名称直接由 `target.node_name` 派生。

激活 QDQ 名称由 `simo/onnx/onnx_quant.py:824-835` 的
`_plan_activation_qdq()` 生成：

```text
{node_name}_SimoQuantInput
{node_name}_SimoScale
{node_name}_SimoDequantOutput
{node_name}_input_simo
```

`simo/onnx/onnx_quant.py:949-974` 的 `_create_qdq_nodes()` 又生成：

```text
{node_name}_input_simo_quant
{node_name}_input_simo_dequant
```

权重名称由 `simo/onnx/onnx_quant.py:977-1025` 的 `_create_weight_dq_nodes()` 生成：

```text
{node_name}_{weight_name}_simo_q
{node_name}_{weight_name}_simo_scale
{node_name}_weight_simo_weight_dq
{node_name}_weight_simo_dequant
```

如果两个 MatMul 都没有名称，且不先执行 `_name_nodes()`，两者的 `node_name` 都是空字符
串；如果两个节点原本重名，它们的派生前缀也相同。结果可能是：

- 两组 Q/DQ 产生相同的 tensor output 名称，破坏 ONNX 的单一赋值关系；
- 两组量化权重 initializer 同名；
- custom Q/DQ 节点同名，调试输出无法区分目标层；
- 日志和 exclude 配置无法可靠定位具体节点。

因此 `_name_nodes()` 不参与量化数值计算，但它是后续 QDQ 图改写能够安全命名的前置
条件。

### 44.6 对 exclude 和日志的影响

`simo/onnx/onnx_quant.py:765-773` 的 `_matched_exclude()` 会同时检查：

```python
[node.name, *node.input, *node.output]
```

因为 `_name_nodes()` 在 `_find_qdq_target()` 前执行，原本未命名的节点也可以通过生成的
名称（例如 `MatMul_1`）参与 exclude 匹配。不过这种自动名称取决于节点顺序；模型结构
发生变化时序号可能变化，因此长期配置通常使用稳定的导出名称或输入/输出 tensor 名更
稳妥。

`simo/onnx/onnx_quant.py:650-663` 的 `_insert_qdq_in_graph()` 日志也使用
`target.node_name`。统一命名后，日志中的 `node=...` 不会为空，并且可以区分多个同类型
节点。

### 44.7 主图和子图如何处理

`simo/onnx/onnx_quant.py:672-684` 的 `_insert_qdq_in_subgraphs()` 对节点属性中的单个
`GRAPH` 或多个 `GRAPHS` 递归调用 `_insert_qdq_in_graph()`。每次进入
`_insert_qdq_in_graph()`，都会在 `simo/onnx/onnx_quant.py:617` 再调用一次
`_name_nodes()`。

因此：

- 主图会单独命名；
- `If`、`Loop` 等节点携带的子图也会单独命名；
- 唯一性范围是**当前 `GraphProto` 内部**，不是整个 ModelProto 的全局命名空间；
- 不同子图中可以各自存在 `MatMul_1`，因为它们位于不同图作用域。

### 44.8 与 `GraphBuilder._unique_name()` 的区别

这两个机制容易混淆，但职责不同。

`simo/onnx/onnx_quant.py:589-605` 的 `_name_nodes()`：

- 处理进入量化流程前已经存在的 `graph.node`；
- 只保证 `node.name` 非空且在当前图内不重复；
- 不重命名 tensor、graph input/output 或 initializer。

`simo/onnx/onnx_quant.py:84-114` 的 `GraphBuilder._unique_name()`：

- 为图改写过程中新增的 adapter 节点的 `node.name` 和 int64 常量 initializer 名生成
  不冲突的名称；
- `simo/onnx/onnx_quant.py:458-466` 的 `_graph_used_names()` 会收集 graph
  input/output、value info、initializer、节点名以及节点输入输出 tensor 名；
- 因而它防止传给 `_unique_name()` 的新节点名/initializer 名与更广泛的现有图名称冲突；
- `simo/onnx/onnx_quant.py:97-102` 的 `GraphBuilder.node()` 不会改写调用者传入的 output
  tensor 名，只会把这些 output 名加入 `_used_names`，保护后续生成的节点名或常量名不再
  占用它们。

简单说：

```text
_name_nodes()              先整理原始节点身份
GraphBuilder._unique_name() 再保护改写期间新建对象的名称
```

### 44.9 不会做的事情

`_name_nodes()` 不会：

- 改变 `node.op_type` 或 domain；
- 改变 `node.input`、`node.output` 或 tensor 名称；
- 根据模块层级恢复 PyTorch 原始模块名；
- 给不同 GraphProto 建立模型级全局唯一名称；
- 插入 Quantize/Dequantize；
- 判断节点是否应该量化。

节点是否是量化目标由 `simo/onnx/onnx_quant.py:699-757` 的 `_find_qdq_target()` 和
配置共同决定。`_name_nodes()` 只解决后续图改写所需的节点身份与命名前缀问题。

### 44.10 一句话总结

```text
_name_nodes() 在每个 ONNX GraphProto 内保留首次出现的有效节点名，
为未命名节点和后续重名节点分配不冲突的“base_序号”名称，
从而保证 SIMO 后续按 node.name 派生的激活 QDQ、权重 initializer、adapter 和日志名称
能够稳定地区分不同量化目标；它本身不改变模型的计算语义。
```

## 45. `simo/onnx/onnx_quant.py` 中 `_find_qdq_target()` 的功能

### 45.1 结论

`simo/onnx/onnx_quant.py:699-757` 的 `_find_qdq_target()` 对一个 ONNX 节点执行两类工作：

1. **筛选候选节点**：检查算子类型、量化配置、exclude 规则以及权重是否为静态常量；
2. **准备量化目标**：读取并立即离线量化权重，然后把后续插入 Q/DQ 所需的全部信息封装
   成 `OpTarget`。

返回值有两种：

```text
OpTarget  -> 当前节点满足条件，后续应插入 activation QDQ 和/或 weight DQ
None      -> 当前节点不处理，调用者把原节点原样保留
```

函数名虽然是 `_find_qdq_target`，但它不是一个只返回 `True/False` 的轻量 predicate。
成功路径会调用 CUDA 权重量化 kernel，生成 `q` 和 `scale`，因此具有实际计算开销，也可能
抛出 CUDA 或未实现格式异常。

### 45.2 函数签名、参数和返回值

`simo/onnx/onnx_quant.py:699-705`，函数 `_find_qdq_target()`：

```python
def _find_qdq_target(
  graph: GraphProto,
  node: NodeProto,
  initializers: dict[str, TensorProto],
  config: QuantizeConfig,
  stats: QdqStats,
) -> OpTarget | None:
```

各参数含义如下：

- `graph`：当前节点所在的 `GraphProto`；筛选阶段不修改它，成功后存入 `OpTarget.graph`，
  供后续加入量化 initializer 和节点。
- `node`：当前待判断的 `NodeProto`；成功后保存原对象引用，后续 QDQ 插入代码会改写它的
  activation/weight input。
- `initializers`：名称到 `TensorProto` 的静态权重映射。这个参数名容易误解，调用点传入的
  不仅是 `graph.initializer`，也包含标准 ONNX `Constant` 节点产生的 tensor。
- `config`：已经解析和校验过的 `QuantizeConfig`，包含 `module_configs` 和 `excludes`。
- `stats`：整个转换过程共享的 `QdqStats`，用于记录候选、跳过原因和算子类型。
- 返回 `OpTarget` 表示成功；返回 `None` 表示跳过或不匹配。

### 45.3 调用上下文

调用链位于：

```text
simo/onnx/onnx_quant.py:469-482
insert_qdq_nodes()
  -> simo/onnx/onnx_quant.py:608-669
     _insert_qdq_in_graph()
       -> simo/onnx/onnx_quant.py:631
          _find_qdq_target(...)
```

`simo/onnx/onnx_quant.py:608-627` 的 `_insert_qdq_in_graph()` 在调用前准备：

```python
initializers = {init.name: init for init in graph.initializer}
constant_tensors = _constant_tensors(graph)
available_weights = {**constant_tensors, **initializers}
```

其中 `simo/onnx/onnx_quant.py:687-696` 的 `_constant_tensors()` 只收集：

- 默认 ONNX domain；
- `op_type == "Constant"`；
- 恰好一个 output；
- `value` 属性确实是 `TensorProto`。

最终传给 `_find_qdq_target()` 的是 `available_weights`。因此它支持两种直接静态权重：

```text
graph.initializer 中的 W
Constant(value=TensorProto) 节点直接输出的 W
```

而由其他节点在运行时计算出的权重不在这个映射中。

### 45.4 完整决策流程

可以把 `simo/onnx/onnx_quant.py:699-757` 的 `_find_qdq_target()` 概括为：

```text
当前 node
  |
  |-- op_type 不属于 MatMul/Gemm/Conv --------------------------> None
  |
  |-- 找不到匹配的 module_config -------------------------------> None
  |
  |-- stats.targets += 1
  |
  |-- 命中 excludes --------------------------------------------> 记录 excluded_by_config -> None
  |
  |-- 少于两个输入，或 input[1] 不是直接静态权重 -------------> 记录 dynamic_weight -> None
  |
  |-- TensorProto -> NumPy，读取 node attributes
  |
  |-- quantize_weight_array(...)
  |      |
  |      `-- ValueError ----------------------------------------> 记录 weight_quant_error -> None
  |
  `-- 封装并返回 OpTarget
```

下面逐项说明。

### 45.5 第一道筛选：只接受支持的 op type

`simo/onnx/onnx_quant.py:25-33` 定义：

```python
SUPPORTED_QDQ_OPS = {"MatMul", "Gemm", "Conv"}
```

`simo/onnx/onnx_quant.py:706-707` 的 `_find_qdq_target()` 首先检查：

```python
if node.op_type not in SUPPORTED_QDQ_OPS:
  return None
```

因此 `Add`、`Relu`、`LayerNormalization` 等节点立即返回 `None`，不会增加
`stats.targets`，也不会留下 skipped 记录。

当前检查只比较 `node.op_type`，没有检查 `node.domain`。也就是说代码假设匹配到的是标准
ONNX `MatMul/Gemm/Conv`；函数本身不会额外拒绝自定义 domain 下恰好同名的 op。

### 45.6 第二道筛选：查找匹配的 module config

`simo/onnx/onnx_quant.py:708-710` 调用
`simo/onnx/onnx_quant.py:776-780` 的 `_get_module_config()`：

```python
for module_config in config.module_configs:
  if node.op_type in module_config.targets:
    return module_config
return None
```

含义是：

- 只按 `node.op_type` 与 `module_config.targets` 精确匹配；
- 返回列表中第一个匹配配置，后续同类型配置不会继续检查；
- 没有匹配配置时返回 `None`，也不计为 skipped target。

配置归一化发生在 `simo/onnx/onnx_quant.py:511-526` 的
`_normalize_module_config()`。例如 `targets_op_types=["Linear"]` 会通过
`TARGET_OP_ALIASES` 展开为 `MatMul` 和 `Gemm`；这里的 `_find_qdq_target()` 最终看到的仍是
规范化后的 op type 列表。

### 45.7 `stats.targets` 的准确含义

通过“支持的 op type + module config 匹配”后，
`simo/onnx/onnx_quant.py:711` 执行：

```python
stats.targets += 1
```

这发生在 exclude、静态权重和权重量化成功检查之前。因此：

```text
targets = 配置选中的候选节点数
inserted = 最终成功插入 QDQ 的节点数
skipped = 命中配置但因后续条件被跳过的节点数
```

所以一个被 exclude 的 MatMul 可以产生 `targets=1, inserted=0, skipped=1`。
`simo/onnx/tests/test_qdq_utils.py:1175-1205` 的
`test_insert_qdq_nodes_excludes_config_pattern_and_logs()` 正在验证这个统计口径。

### 45.8 第三道筛选：exclude 规则

`simo/onnx/onnx_quant.py:713-723` 的 `_find_qdq_target()` 调用
`simo/onnx/onnx_quant.py:765-773` 的 `_matched_exclude()`。

`_matched_exclude()` 对以下名称逐一匹配：

```python
[node.name, *node.input, *node.output]
```

每个非空 exclude 字符串支持三种命中方式：

```text
name == exclude                     精确匹配
fnmatch.fnmatchcase(name, exclude)  shell wildcard 匹配
exclude in name                     子字符串匹配
```

命中后 `_find_qdq_target()`：

1. 调用 `simo/onnx/onnx_quant.py:760-762` 的 `_record_skip()`，记录
   `skipped_by_reason["excluded_by_config"]` 和对应 op type；
2. 把实际命中的 exclude 加入 `stats.matched_excludes`；
3. 输出包含 node、op 和 exclude 的 info 日志；
4. 返回 `None`。

`simo/onnx/tests/test_qdq_utils.py:1168-1172` 的
`test_insert_qdq_nodes_excludes_named_node()` 验证按节点名跳过；
`simo/onnx/tests/test_qdq_utils.py:1175-1205` 的
`test_insert_qdq_nodes_excludes_config_pattern_and_logs()` 验证 `lm_head` 子字符串匹配和日志。

### 45.9 第四道筛选：权重必须是直接静态 tensor

`simo/onnx/onnx_quant.py:725-728` 检查：

```python
if len(node.input) < 2 or node.input[1] not in initializers:
  _record_skip(stats, node, "dynamic_weight")
  ...
  return None
```

对当前支持的三个算子，代码固定采用：

```text
input[0] = activation
input[1] = weight
```

只有第二个输入名称存在于前面构造的 `available_weights` 时，才继续处理。下列情况会以
`dynamic_weight` 跳过：

- 节点少于两个输入；
- `W` 是 graph input；
- `W` 是另一个运行时计算节点的输出；
- `W` 虽然数值可能不变，但没有表示成 initializer 或直接 Constant TensorProto。

`simo/onnx/tests/test_qdq_utils.py:915-928` 的
`test_insert_qdq_nodes_skips_dynamic_weight_matmul()` 构造 `X`、`W` 都是 graph input 的
MatMul，验证不会插入任何 `com.simo` 节点，原始输入 `['X','W']` 保持不变。

另一方面，`simo/onnx/tests/test_qdq_utils.py:866-885` 的
`test_insert_qdq_nodes_inserts_if_subgraph_constant_conv_weight()` 验证子图中的直接
`Constant` 权重能够被识别和量化。

### 45.10 读取并离线量化权重

通过前四道筛选后，`simo/onnx/onnx_quant.py:730-733` 执行：

```python
weight = numpy_helper.to_array(initializers[node.input[1]])
attrs = _node_attrs(node)
quantized = quantize_weight_array(node.op_type, weight, attrs, module_config.weight)
```

其中：

- `simo/onnx/onnx_quant.py:1127-1128` 的 `_node_attrs()` 把 NodeProto attributes 转为
  Python 字典；例如 Gemm 的 `transB`、Conv 的 `group` 等会影响权重逻辑布局。
- `simo/onnx/weight_quant.py:32-48` 的 `logical_weight_view()` 根据 op type 解释权重：
  Conv 保持 `[O,I/group,...]`，Gemm 根据 `transB` 决定是否转置，MatMul 使用 `W.T`。
- `simo/onnx/weight_quant.py:51-69` 的 `quantize_weight_array()` 准备量化布局，创建 CUDA
  `float32` tensor，调用对应 downcast kernel，并返回 `QuantizedWeight`。

因此 `_find_qdq_target()` 的成功路径已经完成权重量化，而不是只保存“以后再量化”的配置。
最终返回的 `QuantizedWeight.q`、`QuantizedWeight.scale` 都已经是可写入 ONNX initializer
的 NumPy byte carrier。

### 45.11 `ValueError` 是软跳过，其他异常继续抛出

`simo/onnx/onnx_quant.py:732-742` 只捕获 `quantize_weight_array()` 抛出的
`ValueError`：

```python
except ValueError as exc:
  _record_skip(stats, node, "weight_quant_error")
  logger.warning(...)
  return None
```

典型 `ValueError` 包括权重 rank 不符合算子要求、量化 axis/layout 不支持等。该节点会原样
保留，统计为 `weight_quant_error`，转换继续处理其他节点。

但函数不会吞掉所有异常：

- `simo/onnx/weight_quant.py:54-55` 的 `quantize_weight_array()` 对 `mxint4` 抛出的
  `NotImplementedError` 不会被捕获；
- `simo/onnx/weight_quant.py:166-170` 的 `_float32_cuda_tensor()` 在 CUDA 不可用时抛出的
  `RuntimeError` 不会被捕获；
- CUDA kernel 或其他非 `ValueError` 异常也会向上传播，使整个转换失败。

这个区别是有意的行为边界：不适合某个节点布局的 `ValueError` 可以按节点跳过；运行环境
缺失或实现尚不存在则不会伪装成一次普通 skip。

### 45.12 成功时返回的 `OpTarget` 包含什么

`simo/onnx/onnx_quant.py:39-53` 定义 frozen dataclass `OpTarget`；
`simo/onnx/onnx_quant.py:743-757` 的 `_find_qdq_target()` 填充以下字段：

| 字段 | 来源与用途 |
|---|---|
| `graph` | 当前 GraphProto，后续加入 initializer |
| `node` | 原 NodeProto，后续改写 input |
| `node_name` | `node.name`，作为所有新增名称的前缀 |
| `op_type` | `MatMul`、`Gemm` 或 `Conv` |
| `activation_input_index` | 固定为 0 |
| `weight_input_index` | 固定为 1 |
| `weight_name` | 原始 `node.input[1]` 名称 |
| `weight_shape` | `quantized.original_shape`，即 ONNX 原权重形状 |
| `logical_shape` | op-specific 逻辑权重形状，可能与原形状不同 |
| `attrs` | `_node_attrs(node)` 的结果 |
| `input_spec` | module config 的 input 量化配置，可以为 `None` |
| `weight_spec` | module config 的 weight 量化配置 |
| `quantized_weight` | 已生成的 q、scale 和布局恢复元数据 |

`activation_input_index=0` 和 `weight_input_index=1` 不是从 schema 动态推断的，而是当前
`MatMul/Gemm/Conv` 路径的固定约定。

### 45.13 返回 `OpTarget` 后发生什么

回到 `simo/onnx/onnx_quant.py:629-665` 的 `_insert_qdq_in_graph()`：

```python
target = _find_qdq_target(...)
if target is None:
  new_nodes.append(node)
  continue
```

返回 `None` 时，原节点直接加入 `new_nodes`，不插入 SIMO Q/DQ。

返回 `OpTarget` 时：

1. 如果 `target.input_spec` 非空，调用
   `simo/onnx/onnx_quant.py:783-821` 的 `_create_activation_qdq_nodes()`，并把
   `node.input[0]` 改接到 activation DQ 输出；
2. 始终调用 `simo/onnx/onnx_quant.py:977-1031` 的 `_create_weight_dq_nodes()`，把已经
   量化的 q/scale 写入 initializer，创建 weight Dequantize，并改写 `node.input[1]`；
3. 更新 `stats.inserted`、`stats.simo_nodes` 和 `inserted_by_op`；
4. 按“activation adapter/QDQ、weight DQ/adapter、原节点”的顺序加入新图。

这也解释了 weight-only 配置：`input_spec=None` 并不会使 `_find_qdq_target()` 返回
`None`，只是调用者跳过 activation QDQ。`simo/onnx/tests/test_qdq_utils.py:1077-1092` 的
`test_insert_qdq_nodes_weight_only_config_skips_activation_qdq()` 验证最终只出现一个权重
`Dequantize`。

### 45.14 五类 `None` 返回的区别

| 条件 | 是否 `targets += 1` | 是否记录 skipped reason | 原因 |
|---|---:|---|---|
| op type 不在支持集合 | 否 | 否 | 与本量化 pass 无关 |
| 没有匹配 module config | 否 | 否 | 配置未选中 |
| 命中 excludes | 是 | `excluded_by_config` | 用户显式排除 |
| 权重不是直接静态 tensor | 是 | `dynamic_weight` | 当前实现只做静态权重量化 |
| 权重量化抛出 `ValueError` | 是 | `weight_quant_error` | 当前节点形状/布局不支持 |

前三种“有 reason 的 skip”都通过 `simo/onnx/onnx_quant.py:760-762` 的
`_record_skip()` 同时增加 `skipped_by_reason` 和 `skipped_by_op[node.op_type]`。

### 45.15 这个函数不会做什么

`_find_qdq_target()` 本身不会：

- 把 Q/DQ 节点加入 `graph.node`；
- 改写原节点的 input/output；
- 删除原始权重 initializer；
- 量化动态权重；
- 处理 bias 或 output 量化配置；
- 检查 activation shape 是否适合 custom QDQ runtime；
- 更新 `stats.inserted`。

这些工作分别由 `_create_activation_qdq_nodes()`、`_create_weight_dq_nodes()`、
`_remove_unused_values()` 和 `_insert_qdq_in_graph()` 后续完成。它的职责边界是：完成目标资格
判断、准备静态量化权重，并把后续改图所需上下文打包返回。

### 45.16 一句话总结

```text
_find_qdq_target() 是 SIMO ONNX QDQ pass 的“候选筛选器 + 静态权重预处理器”：
它只接受配置选中的 MatMul/Gemm/Conv，应用 exclude 和静态权重约束，
立即离线量化合格节点的权重，再返回包含节点、配置、形状、属性、q/scale 的 OpTarget；
返回 None 时调用者原样保留节点，只有已命中的候选 skip 才写入跳过统计。
```

## 46. `simo/onnx/onnx_quant.py` 中 `_create_activation_qdq_nodes()` 的功能

### 46.1 结论

`simo/onnx/onnx_quant.py:783-821` 的 `_create_activation_qdq_nodes()` 是 SIMO ONNX
激活 QDQ 图改写的**编排函数**。它本身不计算量化值，而是完成三件事：

1. 调用 `_plan_activation_qdq()`，根据算子、量化配置和输入形状制订激活 QDQ 布局计划；
2. 按计划创建 `com.simo::Quantize/Dequantize`，必要时同时创建
   `Shape/Transpose/Reshape/Pad/Slice` 等标准 ONNX adapter 节点；
3. 把原始 `MatMul/Gemm/Conv` 的 activation input 改接到反量化并恢复布局后的输出。

可以概括为：

```text
原始：X -----------------------------------------> target op

改写：X -> [layout adapters] -> Quantize -> Dequantize
          -> [inverse adapters] -----------------> target op
```

函数返回的是需要插到原算子前面的 `list[NodeProto]`。它会改写原节点 input，但不会自行把
这些新节点追加到 `graph.node`；真正合并节点列表的是调用者 `_insert_qdq_in_graph()`。
不过 transformer 构造 reshape/pad/slice 参数时，会通过
`simo/onnx/onnx_quant.py:90-95` 的 `GraphBuilder.const_i64()` 直接把所需 int64 常量加入
`graph.initializer`。

### 46.2 函数签名和参数

`simo/onnx/onnx_quant.py:783-789`，函数 `_create_activation_qdq_nodes()`：

```python
def _create_activation_qdq_nodes(
  target: OpTarget,
  spec: QuantizeSpecType,
  value_shapes: dict[str, tuple[int, ...]],
  value_shape_signatures: dict[str, tuple[int | str | None, ...]],
  transformer: ONNXTensorTransformer,
) -> list[NodeProto]:
```

参数含义：

- `target`：`simo/onnx/onnx_quant.py:39-53` 的 `OpTarget`，包含原始节点、activation
  input index、节点名、算子类型和权重逻辑形状等信息。
- `spec`：该模块的 input 量化配置，例如 MXINT8、FP8 per-group 或 INT8 per-tensor。
- `value_shapes`：只包含所有维度均为静态整数的 shape。
- `value_shape_signatures`：保留 `int`、符号维字符串和未知维 `None` 的 shape signature。
- `transformer`：当前 graph 共享的 `ONNXTensorTransformer`，用于生成布局变换、padding、
  Q/DQ 和恢复节点，并管理新增名称。

`simo/onnx/onnx_quant.py:1131-1170` 的 `_extract_shapes()`、`_value_shapes()`、
`_value_shape_signatures()`、`_value_shape()` 和 `_value_shape_signature()` 负责构造两类 shape
映射：

```text
完全静态 [2,3,18]      -> value_shapes 和 value_shape_signatures 都有
带符号 [N,3,H,W]       -> 只在 value_shape_signatures 中保留
没有 shape 信息         -> 两个映射中都没有
```

### 46.3 谁调用它

调用链为：

```text
simo/onnx/onnx_quant.py:469-482
insert_qdq_nodes()
  -> simo/onnx/onnx_quant.py:608-669
     _insert_qdq_in_graph()
       -> simo/onnx/onnx_quant.py:699-757
          _find_qdq_target()
       -> simo/onnx/onnx_quant.py:636-641
          _create_activation_qdq_nodes()
```

`simo/onnx/onnx_quant.py:636-641` 的 `_insert_qdq_in_graph()` 只有在
`target.input_spec` 非空时才调用该函数：

```python
activation_nodes = (
  _create_activation_qdq_nodes(...)
  if target.input_spec
  else []
)
```

因此 weight-only 配置不会进入 `_create_activation_qdq_nodes()`。
`simo/onnx/tests/test_qdq_utils.py:1077-1092` 的
`test_insert_qdq_nodes_weight_only_config_skips_activation_qdq()` 验证此时 activation input
仍是 `X`，图中只有 weight `Dequantize`。

### 46.4 三条执行分支

`simo/onnx/onnx_quant.py:790-821` 的控制流可以写成：

```text
plan = _plan_activation_qdq(...)
  |
  |-- plan is None
  |     `-> return []，不改 activation input
  |
  |-- plan.use_transformer == False
  |     |-> 直接创建 Quantize + Dequantize
  |     |-> target.input[activation_index] = plan.dq_out
  |     `-> 返回两个节点
  |
  `-- plan.use_transformer == True
        |-> transformer.activation_qdq(...) 创建 adapter + QDQ + inverse adapter
        |-> target.input[activation_index] = transformer 返回的最终 output
        `-> 返回完整节点列表
```

#### 分支一：planner 返回 `None`

`simo/onnx/onnx_quant.py:790-792`：

```python
plan = _plan_activation_qdq(...)
if plan is None:
  return []
```

当前主要发生在 Conv activation 缺少足够 rank 信息时。
`simo/onnx/onnx_quant.py:864-880` 的 `_plan_activation_qdq()` 尝试从 activation signature
和 weight rank 推断 Conv 布局；仍不能构造 rank-N recipe 时返回 `None`。

这里需要区分“跳过 activation QDQ”和“跳过整个量化 target”：

- `_create_activation_qdq_nodes()` 返回空列表；
- 原 Conv 的 `input[0]` 不变；
- `_insert_qdq_in_graph()` 仍会调用 `_create_weight_dq_nodes()`；
- 该 target 仍可插入 weight DQ，并计入 `stats.inserted`。

`simo/onnx/tests/test_activation_qdq_plan.py:91-99` 的
`test_plan_activation_qdq_skips_conv_without_rank_information()` 验证 planner 返回 `None`。

#### 分支二：直接 Q/DQ，不使用 transformer

`simo/onnx/onnx_quant.py:794-803`：

```python
if not plan.use_transformer:
  target.node.input[target.activation_input_index] = plan.dq_out
  return _create_qdq_nodes(...)
```

该路径不创建布局 adapter，直接生成：

```text
plan.x_name
  -> com.simo::Quantize
  -> com.simo::Dequantize(plan.dq_out)
  -> target activation input
```

`simo/onnx/onnx_quant.py:949-974` 的 `_create_qdq_nodes()` 创建这两个 custom op。

在当前正常入口中，`_find_qdq_target()` 只产生 `MatMul/Gemm/Conv` target：

- 未知 shape 的 MatMul/Gemm 会使用 transformer，并从权重推断 K；
- 未知 shape 的 Conv 要么构造动态 recipe，要么返回 `None`；
- 已知 shape 的三类算子也都使用 transformer。

所以 `use_transformer=False` 当前基本是为未来扩展算子保留的兜底分支，而不是现有三种
target 的常见路径。

#### 分支三：使用 transformer

`simo/onnx/onnx_quant.py:805-821` 调用
`simo/onnx/onnx_quant.py:121-151` 的 `ONNXTensorTransformer.activation_qdq()`，将 plan
中的所有布局参数传入。transformer 返回：

```python
nodes, output
```

- `nodes`：按执行顺序排列的 adapter、Q/DQ 和恢复节点；
- `output`：完成 DQ、去 padding 和恢复原布局后的最终 tensor 名。

第 820 行随后执行：

```python
target.node.input[target.activation_input_index] = output
```

对当前支持算子，`activation_input_index` 由
`simo/onnx/onnx_quant.py:743-756` 的 `_find_qdq_target()` 固定设置为 0，所以改写的是
`MatMul/Gemm/Conv` 的第一个输入。

### 46.5 `ActivationQdqPlan` 中各字段的作用

`simo/onnx/onnx_quant.py:56-70` 定义 `ActivationQdqPlan`：

| 字段 | 含义 |
|---|---|
| `x_name` | 原 activation tensor 名 |
| `q_out` | Quantize 的量化数据输出名 |
| `scale_out` | Quantize 的 scale 输出名 |
| `dq_out` | Dequantize 后的 rank-2/去 padding 输出名 |
| `attrs` | dtype、granularity、axis、group/block size、scale mode 等 custom-op 属性 |
| `prefix` | adapter 节点和 tensor 的统一命名前缀 |
| `quant_group` | rank-2 最后一维需要满足的量化分组对齐值 |
| `use_transformer` | 是否需要通过 transformer 编排布局 |
| `recipe` | transpose、flatten、padding 和逆变换所需的 `LayoutRecipe` |
| `k` | 可从权重或静态 shape 推断出的 activation K |
| `flatten_to_row` | 是否把整个 tensor 降成 `[1,-1]` |
| `already_rank2` | 输入是否已经是 custom op 接受的 rank-2 |
| `restore` | QDQ 后是否恢复原始 rank/layout |

`simo/onnx/onnx_quant.py:830-836` 的 `_plan_activation_qdq()` 首先生成统一名称：

```text
x_name    = target.node.input[activation_input_index]
q_out     = {node_name}_SimoQuantInput
scale_out = {node_name}_SimoScale
dq_out    = {node_name}_SimoDequantOutput
prefix    = {node_name}_input_simo
```

并通过 `_simo_attrs()` 生成 semantic QDQ 属性，通过 `_rank2_quant_group()` 计算是否需要
对齐/padding。

### 46.6 planner 如何选择布局方案

虽然 `_create_activation_qdq_nodes()` 只有 39 行，实际节点形态由
`simo/onnx/onnx_quant.py:824-922` 的 `_plan_activation_qdq()` 决定。

#### 情况 A：INT8/FP8 per-tensor 降成单行 QDQ

`simo/onnx/onnx_quant.py:838-848` 检测 `_uses_single_row_qdq(spec)`。
`simo/onnx/onnx_quant.py:1095-1110` 的 `_uses_single_row_qdq()` 和
`_single_row_qdq_attrs()` 将 INT8/FP8 E4M3 per-tensor 配置转换为 runtime 已支持的：

```text
Reshape(original, [1,-1])
-> Quantize(granularity=per_channel, axis=0)
-> Dequantize
-> Reshape(original shape)
```

因为只有一行，axis 0 per-channel 只生成一个 scale，语义等价于 per-tensor。
`simo/onnx/tests/test_qdq_utils.py:1322-1382` 的
`test_insert_qdq_nodes_lowers_int8_per_tensor_activation_to_single_row_qdq()` 和
`test_insert_qdq_nodes_lowers_fp8_per_tensor_activation_to_single_row_qdq()` 验证 `[1,-1]`、
`axis=0` 和恢复原形状。

#### 情况 B：未知 shape 的 MatMul/Gemm

`simo/onnx/onnx_quant.py:850-863` 在 `value_shapes` 中找不到完全静态 shape 时，对
MatMul/Gemm 仍构造 transformer plan：

1. `simo/onnx/onnx_quant.py:1121-1124` 的 `_linear_activation_k()` 从已量化权重的
   `target.logical_shape[1]` 推断 activation K；
2. 使用 `_rank2_axis_attrs(spec)` 把 custom op 的量化轴设为 rank-2 axis 1；
3. transformer 在 ONNX 图中动态读取原 shape，reshape 为 `[-1,K]`，QDQ 后再恢复。

`simo/onnx/tests/test_qdq_utils.py:559-589` 的
`test_insert_qdq_nodes_flattens_and_pads_unknown_rank_matmul_activation()` 验证未知 rank 输入仍能
根据 K=18 生成 reshape、padding、QDQ、slice 和 restore。

#### 情况 C：动态或符号 shape 的 Conv

`simo/onnx/onnx_quant.py:864-878`：

1. `_conv_activation_shape_signature()` 可根据 Conv weight rank 补出 activation rank 和
   channel 数；
2. `_can_dynamic_rank_n_qdq()` 检查 signature 与 axis；
3. `build_layout_recipe()` 生成把目标 axis 移到最后一维的 recipe；
4. transformer 在运行时通过 `Shape/Gather/Concat/Neg/Mod` 等节点计算 reshape 和动态
   padding。

相关辅助函数位于 `simo/onnx/onnx_quant.py:925-946` 的
`_can_dynamic_rank_n_qdq()` 和 `_conv_activation_shape_signature()`。

`simo/onnx/tests/test_qdq_utils.py:782-841` 的
`test_insert_qdq_nodes_inserts_dynamic_conv_activation_through_2d_qdq()` 和
`test_insert_qdq_nodes_inserts_dynamic_conv_activation_with_dynamic_axis_padding()` 分别验证动态
Conv 的 axis-to-last 变换与动态 padding。

#### 情况 D：已知 rank-1/rank-2 shape

`simo/onnx/onnx_quant.py:892-904` 设置：

```text
rank-2 -> already_rank2=True，直接进入 rank2_qdq()
rank-1 -> 由 transformer 转成 rank-2
两者 -> restore=False
```

对常见 rank-2 activation，不会创建无意义的 reshape/restore；如果 K 不对齐，仍可在
rank-2 上创建 `Pad -> QDQ -> Slice`。
`simo/onnx/tests/test_activation_qdq_plan.py:30-43` 的
`test_plan_activation_qdq_keeps_static_rank2_direct()` 验证 `already_rank2=True` 和
`restore=False`；`simo/onnx/tests/test_qdq_utils.py:477-518` 的
`test_insert_qdq_nodes_pads_unaligned_rank2_matmul_activation()` 验证 K=18 的 rank-2 输入只做
padding/QDQ/unpadding。

#### 情况 E：已知 rank 大于 2

`simo/onnx/onnx_quant.py:906-922`：

- MatMul/Gemm 总是把最后一维作为逻辑 K；
- Conv 使用配置中的 activation axis；
- `build_layout_recipe()` 负责必要的 axis-to-last transpose；
- QDQ 在连续 rank-2 `[outer_dim, quant_dim]` 上执行；
- 最后 reshape 并 inverse-transpose 回原布局。

`simo/onnx/tests/test_qdq_utils.py:438-474` 的
`test_insert_qdq_nodes_inserts_rank3_matmul_activation_through_2d_qdq()` 验证 rank-3 MatMul
经两个 Reshape 穿过 rank-2 QDQ；
`simo/onnx/tests/test_activation_qdq_plan.py:62-74` 的
`test_plan_activation_qdq_builds_conv_rank_n_recipe_on_channel_axis()` 验证 NCHW Conv 的
channel axis 1 生成 `perm=[0,2,3,1]`。

### 46.7 transformer 实际生成哪些节点

`simo/onnx/onnx_quant.py:121-151` 的 `ONNXTensorTransformer.activation_qdq()` 分三步：

```text
forward_rank2()
  -> rank2_qdq()
  -> restore_from_rank2()
```

#### `forward_rank2()`：把任意布局转换成连续 rank-2

`simo/onnx/onnx_quant.py:153-223` 的 `ONNXTensorTransformer.forward_rank2()` 可生成：

- `Transpose`：把配置的量化 axis 移到最后；
- `Shape`：保存运行时原始/转置 shape；
- `Gather + Concat`：在 K 也动态时构造 `[-1,K]`；
- `Reshape`：把输入变成 `[outer_dim,K]`；
- 单行模式下用 `[1,-1]`。

#### `rank2_qdq()`：插入 Q/DQ，并在必要时补齐 K

`simo/onnx/onnx_quant.py:225-280` 的 `ONNXTensorTransformer.rank2_qdq()`：

- K 已对齐、`quant_group<=1` 或无需对齐时，直接创建 Q/DQ；
- 静态 K 未对齐时，创建固定 `Pad -> QDQ -> Slice`；
- 动态 K 未对齐时，用 `Neg + Mod + Concat` 动态计算 pad length，再执行
  `Pad -> QDQ -> Slice`。

真正的两个 custom op 由 `simo/onnx/onnx_quant.py:949-974` 的 `_create_qdq_nodes()`
创建，domain 是 `com.simo`，并携带 plan 中的 dtype、granularity、axis、scale mode、
group size 和 block size 等属性。

#### `restore_from_rank2()`：恢复原 rank 和 axis

`simo/onnx/onnx_quant.py:282-306` 的 `ONNXTensorTransformer.restore_from_rank2()`：

1. `Reshape` 回保存的 shape；
2. 如果此前移动过 axis，再用 `recipe.inverse_perm` 执行逆 `Transpose`；
3. 返回最终 tensor 名给 `_create_activation_qdq_nodes()` 改接原算子。

### 46.8 具体例子：rank-3、K=18 的 MXINT8 MatMul

以 `X=[2,3,18]`、MXINT8 block size 32 为例，
`_create_activation_qdq_nodes()` 最终返回：

```text
Reshape [2,3,18] -> [6,18]
Pad K: 18 -> 32
com.simo::Quantize(mxint8) -> q + scale
com.simo::Dequantize       -> float [6,32]
Slice K: 32 -> 18
Reshape [6,18] -> [2,3,18]
```

然后 `simo/onnx/onnx_quant.py:820` 将：

```text
MatMul.input[0]: "X" -> "matmul_input_simo_restore"
```

这里 MatMul 仍接收 float32，只是数值已经经历 MXINT8 Q/DQ。该完整实例在本文第 43 节有
进一步的节点名和 kernel 调用链说明。

### 46.9 返回节点的顺序和图合并

transformer 返回的列表顺序由 `simo/onnx/onnx_quant.py:121-151` 的
`ONNXTensorTransformer.activation_qdq()` 保证：

```text
pre_nodes -> qdq_nodes -> restore_nodes
```

`_create_activation_qdq_nodes()` 把该 NodeProto 列表返回给调用者；与此同时，构图所需的
int64 shape/pad/slice 常量已经由 `GraphBuilder.const_i64()` 写入 `graph.initializer`。随后
`simo/onnx/onnx_quant.py:643-665` 的 `_insert_qdq_in_graph()` 创建 weight DQ，并按以下顺序
写入 `new_nodes`：

```text
activation_nodes -> weight_nodes -> original target node
```

最终在 `simo/onnx/onnx_quant.py:667-669` 替换 `graph.node` 并清理未使用值。

### 46.10 它不执行运行时量化

`_create_activation_qdq_nodes()` 运行在 ONNX 模型转换阶段。它只构造 NodeProto、必要的
int64 initializer 并改写连接，不会读取实际 activation，也不会计算动态 scale。

推理时：

```text
com.simo::Quantize   根据当前 activation 计算 q 和 scale
com.simo::Dequantize 根据 q 和 scale 恢复 float32
```

custom-op runtime 位于 `simo/onnx/ort_plugin/simo_qdq_ops.cc:290-391` 的
`SimoQuantizeOp::Compute()` 和 `simo/onnx/ort_plugin/simo_qdq_ops.cc:393-500` 的
`SimoDequantizeOp::Compute()`。因此激活量化是运行时动态量化，和
`_find_qdq_target()` 在转换期完成的静态权重量化不同。

### 46.11 异常和统计边界

`simo/onnx/onnx_quant.py:783-821` 的 `_create_activation_qdq_nodes()` 没有 `try/except`。
planner 或 transformer 发现 axis、rank、layout 不合法时，异常会直接向上传播，不会记录为
`weight_quant_error` 或其他 skip reason。

例如 `simo/onnx/tests/test_qdq_utils.py:767-779` 的
`test_insert_qdq_nodes_rejects_conv_rank_n_multi_axis_qdq()` 验证 Conv rank-N activation 配置
多个 axis 时抛出 `ValueError`。

该函数本身也不更新 `QdqStats`。`simo_nodes`、`inserted` 和 `inserted_by_op` 由
`simo/onnx/onnx_quant.py:643-649` 的 `_insert_qdq_in_graph()` 在 activation 和 weight 节点
都创建后统一统计。

### 46.12 一句话总结

```text
_create_activation_qdq_nodes() 是“plan -> materialize -> rewire”编排层：
它让 _plan_activation_qdq() 决定激活布局方案，让 ONNXTensorTransformer 把方案具体化为
transpose/reshape/pad/QDQ/slice/restore 节点，再把原算子的 input[0] 改接到最终 DQ 输出；
它只改写 ONNX 图，不在转换期计算 activation 量化值。
```

## 47. 解决 `like-useful/test-lstm.py` 导出 PyTorch LSTM 到 ONNX 的错误

### 47.1 结论

`like-useful/test-lstm.py:26-32` 的 `torch.onnx.export()` 把输入写成：

```python
(input, h0, c0)
```

但 `torch.nn.LSTM.forward()` 的调用签名是：

```python
forward(input, hx=None)
```

其中 `hx` 本身必须是二元组 `(h0, c0)`。因此 exporter 的正确参数树应为：

```python
(input, (h0, c0))
```

原脚本等价于让 exporter 调用：

```python
rnn(input, h0, c0)  # 三个位置参数，错误
```

正确调用是：

```python
rnn(input, (h0, c0))  # 两个位置参数，第二个参数是 tuple
```

这就是日志中 `TypeError: too many positional arguments` 的直接原因，与 SIMO、权重 shape、
ONNX `LSTM` 算子是否存在、CUDA 或 `h0/c0` 数值均无关。

在当前 conda 环境中实测版本为：

```text
Python      3.12.12
PyTorch     2.11.0+cu130
ONNX        1.22.0
ONNX Script 0.7.1
```

### 47.2 原脚本中哪里不一致

`like-useful/test-lstm.py:19` 的顶层代码已经使用了正确的 PyTorch 调用方式：

```python
output, (hn, cn) = rnn(input, (h0, c0))
```

因此普通 PyTorch forward 能成功，`temp/test-lstm.log:3-5` 也打印出正确 shape：

```text
input  [5,3,10]
h0/c0  [1,3,20]
output [5,3,20]
hn/cn  [1,3,20]
```

但 `like-useful/test-lstm.py:26-31` 的 `torch.onnx.export()` 又把同一组输入展平成：

```python
torch.onnx.export(
    rnn,
    (input, h0, c0),  # 错误：破坏了 LSTM.forward 的 hx tuple 结构
    ...,
    dynamo=True,
)
```

这两处必须保持相同的 Python 参数树结构。`input_names=["input","h0","c0"]` 有三个名字
并不表示 `forward()` 必须接收三个顶层参数；exporter 会把嵌套参数树中的三个 tensor leaf
展平成三个 ONNX graph input。

正确关系是：

```text
Python 调用参数： (input, (h0, c0))
                         |    |   |
ONNX tensor leaf：     input  h0  c0
ONNX input_names：    input  h0  c0
```

### 47.3 为什么错误发生在 ONNX 转换之前

`temp/test-lstm.log:6-9` 显示新 exporter 先后尝试：

```text
torch.export.export(..., strict=False) -> failed
torch.export.export(..., strict=True)  -> failed
```

`temp/test-lstm.log:50-61` 的核心调用是：

```text
make_fake_inputs()
  -> _combine_args()
  -> signature.bind(*args, **kwargs)
  -> TypeError: too many positional arguments
```

也就是说失败发生在 `torch.export` 用 Python signature 绑定 example inputs 的阶段，还没有
进入 ATen decomposition、ONNX translation 或 ONNX checker。

`temp/test-lstm.log:79-88` 中的 `TorchExportError` 是外层包装错误，真正的 root cause 是其
`Exception summary` 中的：

```text
TypeError: too many positional arguments
```

由于第一阶段就失败，原命令不会生成 `temp/test-lstm.onnx`。

### 47.4 最小修复：保持 `dynamo=True`

如果只解决当前 `too many positional arguments`，对
`like-useful/test-lstm.py:26-32` 做以下最小修改即可：

```diff
 torch.onnx.export(
     rnn,
-    (input, h0, c0),
+    (input, (h0, c0)),
     lstm_onnx_export_path,
     input_names=["input", "h0", "c0"],
     dynamo=True,
 )
```

在指定环境中，这个修改能让 `torch.onnx.export()` 完成全部步骤：

```text
Obtain model graph                 success
Run decompositions                success
Translate the graph into ONNX     success
Optimize the ONNX graph           success
```

因此参数嵌套修复是主错误的必要且充分修复。

### 47.5 为什么本环境更推荐暂时使用 `dynamo=False`

虽然上面的 `dynamo=True` 能导出，但本环境 PyTorch 2.11.0 新 exporter 的 LSTM 产物存在一
个独立问题：

```text
PyTorch 实际 hn/cn             [1,3,20]
ONNX LSTM runtime 实际 hn/cn   [1,3,20]
导出文件声明的 output metadata [1,1,3,20]
```

实测现象：

- `onnx.checker.check_model()` 能通过，因为 checker 不执行完整 shape inference；
- ONNX Runtime 加载时报告 source rank 3 与 target rank 4 不一致，然后采用 lenient merge；
- `onnx.shape_inference.infer_shapes(..., strict_mode=True)` 会明确失败；
- 关闭 `optimize`、去掉 `output_names` 或增加 flat-input wrapper 都不能修复该 metadata。

这是 `dynamo=True` 的后续 shape metadata 问题，不是原日志中参数过多错误的原因。

对当前 PyTorch 2.11 环境，如果后续 SIMO/ONNX 工具需要可靠的静态 shape，推荐暂时使用
TorchScript-based legacy exporter：

```python
dynamo=False
```

legacy exporter 已被 PyTorch 标记为 deprecated，因此这是当前 LSTM 导出兼容性 workaround，
不是长期要求。升级 PyTorch 后应重新验证新 exporter 的 `hn/cn` shape inference，再决定
是否恢复 `dynamo=True`。

### 47.6 当前环境推荐的完整静态 shape 修复

建议将 `like-useful/test-lstm.py:9-32` 中的导出准备和调用整理为：

```python
rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
rnn.eval()

output, (hn, cn) = rnn(input, (h0, c0))

with torch.no_grad():
  torch.onnx.export(
    rnn,
    (input, (h0, c0)),
    "temp/test-lstm.onnx",
    input_names=["input", "h0", "c0"],
    output_names=["output", "hn", "cn"],
    opset_version=18,
    dynamo=False,
  )
```

各修改的作用：

| 修改 | 作用 |
|---|---|
| `rnn.eval()` | 消除 training-mode warning，固定推理语义 |
| `(input, (h0, c0))` | 匹配 `LSTM.forward(input, hx)` 的参数树，解决主错误 |
| `output_names` | 明确三个 ONNX 输出名 |
| `opset_version=18` | 明确当前项目使用的 ONNX opset，不依赖 exporter 默认值 |
| `dynamo=False` | 绕过本环境新 exporter 的 `hn/cn` rank metadata 问题 |
| `torch.no_grad()` | 避免不必要的 autograd 状态；不是主错误修复条件 |

`temp/test-lstm.log:1-2` 的 training-mode `UserWarning` 不是导致导出失败的异常；
`rnn.eval()` 用于解决这条警告，参数 tuple 才用于解决 `TypeError`。

### 47.7 如果必须保留三个平坦的 Python 参数

如果调用方必须继续使用 `(input, h0, c0)`，需要让被导出的 module 本身也具有三个参数的
`forward()`。可以增加 wrapper：

```python
class LSTMExportWrapper(nn.Module):
  def __init__(self, lstm):
    super().__init__()
    self.lstm = lstm

  def forward(self, input, h0, c0):
    output, (hn, cn) = self.lstm(input, (h0, c0))
    return output, hn, cn


model = LSTMExportWrapper(rnn).eval()
torch.onnx.export(
  model,
  (input, h0, c0),
  "temp/test-lstm.onnx",
  input_names=["input", "h0", "c0"],
  output_names=["output", "hn", "cn"],
  opset_version=18,
  dynamo=False,
)
```

wrapper 的 `LSTMExportWrapper.forward()` 接收三个平坦参数，再在内部恢复 LSTM 所需的
`(h0,c0)` tuple；同时把 PyTorch 的嵌套返回值 `(output, (hn,cn))` 展平成三个输出。

对于当前只有一个导出脚本的情况，直接改为 `(input,(h0,c0))` 更简单；wrapper 适用于必须
维持平坦调用 API 的工程代码。

### 47.8 shape 为什么本来就是正确的

`like-useful/test-lstm.py:1-17` 配置：

```text
num_layers     = 1
num_directions = 1
seq_len        = 5
batch          = 3
input_size     = 10
hidden_size    = 20
batch_first    = False（nn.LSTM 默认值）
```

所以 `like-useful/test-lstm.py:14-17` 创建的 shape 正确：

```text
input = [seq_len, batch, input_size]                       = [5,3,10]
h0    = [num_layers * num_directions, batch, hidden_size] = [1,3,20]
c0    = [num_layers * num_directions, batch, hidden_size] = [1,3,20]
```

`like-useful/test-lstm.py:19-20` 的 `nn.LSTM.forward()` 输出也符合：

```text
output = [seq_len, batch, num_directions * hidden_size] = [5,3,20]
hn     = [num_layers * num_directions, batch, hidden]   = [1,3,20]
cn     = [num_layers * num_directions, batch, hidden]   = [1,3,20]
```

因此不需要修改 `h0/c0` 的维度，也不需要把它们 `squeeze`、`cat` 或改为两个额外 LSTM
位置参数。

### 47.9 可选：支持动态 sequence length 和 batch

如果导出的模型需要接受不同的 sequence length 和 batch size，在推荐的
`dynamo=False` 调用中加入：

```python
dynamic_axes={
  "input": {0: "seq_len", 1: "batch"},
  "h0": {1: "batch"},
  "c0": {1: "batch"},
  "output": {0: "seq_len", 1: "batch"},
  "hn": {1: "batch"},
  "cn": {1: "batch"},
},
```

注意所有与 batch 相关的输入和输出必须使用同一个符号 `batch`；仅把 `input` 标成动态而
保持 `h0/c0` 静态，会导致换 batch 时状态 shape 不匹配。

指定环境中已经验证：用 `[5,3,10]` 导出后，以 `[7,2,10]` 的新输入和
`[1,2,20]` 的新 `h0/c0` 在 ONNX Runtime CPUExecutionProvider 上运行成功。

### 47.10 验证结果

对“嵌套参数 + `dynamo=False` + opset 18”的推荐方案执行了以下验证：

1. `torch.onnx.export()` 成功生成模型；
2. `onnx.checker.check_model()` 通过；
3. ONNX 图包含一个标准 domain 的 `LSTM` 节点；
4. graph input shape 为 `input=[5,3,10]`、`h0/c0=[1,3,20]`；
5. graph output metadata 为 `output=[5,3,20]`、`hn/cn=[1,3,20]`；
6. ONNX Runtime CPU 数值与 PyTorch 对齐。

静态输入下的最大绝对误差：

```text
output 1.1920929e-07
hn     4.4703484e-08
cn     8.0093741e-08
```

动态 `[seq_len=7,batch=2]` 输入下的最大绝对误差：

```text
output 8.9406967e-08
hn     5.9604645e-08
cn     5.9604645e-08
```

### 47.11 最小修复与推荐修复的选择

```text
只想解决当前 TypeError：
  (input,h0,c0) -> (input,(h0,c0))，可继续 dynamo=True

需要当前环境生成 shape metadata 干净、便于后续 SIMO 处理的 ONNX：
  (input,(h0,c0)) + rnn.eval() + output_names + opset 18 + dynamo=False

必须保留三个平坦 Python 输入：
  使用 LSTMExportWrapper.forward(input,h0,c0)
```

### 47.12 一句话总结

```text
错误不是 LSTM 无法导出，而是 exporter example args 没有保持 LSTM.forward 的嵌套 hx 结构：
把 like-useful/test-lstm.py:28 的 (input,h0,c0) 改为 (input,(h0,c0)) 即可解决 TypeError；
在当前 PyTorch 2.11 环境，为避免 dynamo exporter 的 hn/cn rank metadata 问题，
推荐暂用 dynamo=False，并用 ONNX checker + ONNX Runtime 数值对比验证产物。
```

## 48. MatMul 的第二个输入不是常量时，SIMO ONNX QDQ 如何处理

### 48.1 直接结论

当 ONNX `MatMul` 的第二个输入 `node.input[1]` 不是 SIMO 能直接读取的静态权重时，
`simo/onnx/onnx_quant.py:699-757` 的 `_find_qdq_target()` 会把该节点记录为
`dynamic_weight`，然后返回 `None`。

返回 `None` 的结果是：

- **不会进行权重量化**；
- **不会生成量化权重 q/scale initializer**；
- **不会插入 weight `com.simo::Dequantize`**；
- 即使配置中包含 input quantization，**也不会插入 activation Quantize/Dequantize**；
- 原始 `MatMul.input == [X,W]` 保持不变；
- MatMul 仍作为普通浮点 ONNX 节点运行。

所以答案不是“动态权重只跳过 weight DQ、仍量化 activation”，而是：

```text
第二个输入不是直接静态权重 -> 整个 MatMul target 跳过 QDQ 改写
```

### 48.2 SIMO 如何定义“可量化的常量权重”

`simo/onnx/onnx_quant.py:608-627` 的 `_insert_qdq_in_graph()` 先构造两个映射：

```python
initializers = {init.name: init for init in graph.initializer}
constant_tensors = _constant_tensors(graph)
available_weights = {**constant_tensors, **initializers}
```

随后把 `available_weights` 传给 `_find_qdq_target()`。因此当前 pass 认为以下两类权重是可
直接离线量化的静态权重：

1. 位于 `graph.initializer` 中的 `TensorProto`；
2. 默认 ONNX domain 的直接 `Constant(value=TensorProto)` 节点输出。

第二类由 `simo/onnx/onnx_quant.py:687-696` 的 `_constant_tensors()` 收集。该函数要求：

```text
node.domain 为空
node.op_type == "Constant"
len(node.output) == 1
value attribute 是 TensorProto
```

这里的“常量”不是泛指理论上可常量折叠的表达式。以下第二输入仍不在
`available_weights` 中，会按 dynamic weight 跳过：

- graph input `W`；
- 另一个运行时算子的输出；
- `Transpose(Constant)`、`Cast(Constant)`、`Add(Constant,Constant)` 等尚未折叠的输出；
- 已有 `Dequantize` 节点的输出；
- 不使用 `value=TensorProto` 表达的其他 Constant 形式。

如果某个权重子图事实上恒定，需要在运行 SIMO QDQ pass 前先做 constant folding，或者把
结果直接转成 initializer/受支持的 Constant TensorProto。

### 48.3 动态权重判断发生在哪里

`simo/onnx/onnx_quant.py:699-728` 的 `_find_qdq_target()` 前半段按以下顺序判断：

```text
op_type 是否为 MatMul/Gemm/Conv
  -> 是否命中 module config
  -> stats.targets += 1
  -> 是否被 excludes 排除
  -> input[1] 是否存在于静态权重映射
```

动态权重的关键代码位于
`simo/onnx/onnx_quant.py:725-728`，函数 `_find_qdq_target()`：

```python
if len(node.input) < 2 or node.input[1] not in initializers:
  _record_skip(stats, node, "dynamic_weight")
  logger.debug("skip QDQ node=%s op=%s reason=dynamic_weight", node.name, node.op_type)
  return None
```

这里参数名虽然叫 `initializers`，但调用方实际传入的是上一节说明的
`available_weights`，即 graph initializer 和直接 Constant TensorProto 的合并映射。

对于典型动态 MatMul：

```text
graph.input = [X, W]
graph.initializer = []
MatMul.input = [X, W]
```

`W not in available_weights`，因此命中上述分支。

### 48.4 为什么不会调用权重量化

只有通过动态权重检查后，`simo/onnx/onnx_quant.py:730-733` 的
`_find_qdq_target()` 才会执行：

```python
weight = numpy_helper.to_array(initializers[node.input[1]])
attrs = _node_attrs(node)
quantized = quantize_weight_array(node.op_type, weight, attrs, module_config.weight)
```

动态权重分支已经在第 728 行返回，所以：

- `numpy_helper.to_array()` 不会执行；
- `simo/onnx/weight_quant.py:51-69` 的 `quantize_weight_array()` 不会执行；
- 不会调用任何 SIMO/PyTorch CUDA downcast kernel；
- 不会创建 `QuantizedWeight`；
- `_find_qdq_target()` 不会构造 `OpTarget`。

这是必要的：模型转换阶段没有动态 `W` 的实际数值，无法离线计算 q 和 scale。

### 48.5 为什么 weight Dequantize 和 activation QDQ 都不会插入

调用点位于 `simo/onnx/onnx_quant.py:629-665` 的 `_insert_qdq_in_graph()`：

```python
target = _find_qdq_target(...)
if target is None:
  new_nodes.append(node)
  continue
```

动态权重返回 `None` 后，代码在第 634 行 `continue`。因此不会执行后面的：

```text
simo/onnx/onnx_quant.py:636-641
_create_activation_qdq_nodes()

simo/onnx/onnx_quant.py:643
_create_weight_dq_nodes()
```

也就是说，跳过发生在 activation QDQ 和 weight DQ 创建之前。

如果 target 成功，`simo/onnx/onnx_quant.py:977-1031` 的 `_create_weight_dq_nodes()` 才会：

1. 从 `target.quantized_weight` 读取 q 和 scale；
2. 在第 1007-1010 行加入 q/scale initializer；
3. 在第 1017-1025 行创建 `com.simo::Dequantize`；
4. 在第 1011 行改写 `node.input[1]`。

动态权重没有 `OpTarget.quantized_weight`，所以这条路径完全不会进入。

改写前后可表示为：

```text
改写前：
X (graph input) ----\
                     MatMul -> Y
W (graph input) ----/

SIMO QDQ pass 后：
X (graph input) ----\
                     MatMul -> Y    # 完全不变
W (graph input) ----/
```

不会变成：

```text
W -> Quantize -> Dequantize -> MatMul
```

当前实现没有动态 weight Quantize/DQ 插入逻辑。

### 48.6 统计和日志如何记录

如果 MatMul 是受支持 op 且命中了 module config，
`simo/onnx/onnx_quant.py:706-711` 的 `_find_qdq_target()` 已经先执行
`stats.targets += 1`，然后才检查动态权重。

`simo/onnx/onnx_quant.py:760-762` 的 `_record_skip()` 会更新：

```text
stats.skipped_by_reason["dynamic_weight"] += 1
stats.skipped_by_op["MatMul"] += 1
```

但以下统计不会增加：

```text
stats.inserted
stats.simo_nodes
stats.inserted_by_op["MatMul"]
```

因为这些只在 `_find_qdq_target()` 成功返回 `OpTarget` 后，由
`simo/onnx/onnx_quant.py:643-649` 的 `_insert_qdq_in_graph()` 更新。

动态权重消息使用 `logger.debug()`，默认 info 日志级别下可能看不到逐节点消息；最终汇总中
仍会包含 `skipped_by_reason={'dynamic_weight': ...}`。

### 48.7 明确的单元测试例子在哪里

有动态 MatMul weight 的明确例子，但不在用户指定的 runtime debug 文件中，而在：

```text
simo/onnx/tests/test_qdq_utils.py:915-928
test_insert_qdq_nodes_skips_dynamic_weight_matmul()
```

该测试在 `simo/onnx/tests/test_qdq_utils.py:916-923` 构造：

```python
graph = helper.make_graph(
  [helper.make_node("MatMul", ["X", "W"], ["Y"], name="dynamic_matmul")],
  "dynamic_weight_graph",
  [_value("X", [2, 4]), _value("W", [4, 3])],
  [_value("Y", [2, 3])],
  [],  # 没有 initializer
)
```

这里 `X` 和 `W` 都是 graph input，initializer 列表为空。调用
`insert_qdq_nodes()` 后，`simo/onnx/tests/test_qdq_utils.py:925-928` 断言：

```python
assert _nodes(model_with_qdq, domain="com.simo") == []
assert _nodes(model_with_qdq, "MatMul")[0].input == ["X", "W"]
```

第一个断言证明 activation Quantize、activation Dequantize、weight Dequantize 一个都没有；
第二个断言证明原 MatMul 两个输入未被改写。

已在指定 conda 环境运行该测试，结果为：

```text
1 passed
```

### 48.8 `test_dynamic_qdq_runtime_debug.py` 中有没有这种例子

没有。

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py` 中只有两个 `MatMul` 模型构造 helper：

1. `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:421-461` 的
   `_qdq_unaligned_matmul_activation_qdq_model()`：
   - 第 423 行创建 NumPy `weight`；
   - 第 425 行创建 `MatMul(["X","W"])`；
   - 第 429 行通过 `onnx.numpy_helper.from_array(weight,"W")` 把 `W` 放入
     graph initializer。

2. `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:464-509` 的
   `_qdq_fp8_per_group_padded_rank2_slice_model()`：
   - 第 468 行创建 NumPy `weight`；
   - 第 470 行创建 `MatMul(["X","W"])`；
   - 第 474 行同样把 `W` 放入 graph initializer。

所以这两个 MatMul 的第二输入都是静态常量，都会进入权重量化和 weight Dequantize 路径。

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:380-418` 的
`_dynamic_conv_activation_qdq_model()` 名称中虽然有 `dynamic`，但动态的是第 386-387 行的
batch/spatial activation shape `N/H/W`；其 Conv 权重仍在第 382 行创建，并在第 388 行作为
initializer 加入图中，不是动态 weight 用例。

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py:332-377` 的
`_tiny_simo_qdq_model()` 直接构造 `com.simo::Quantize/Dequantize`，根本没有 MatMul 或第二个
权重输入，也不能算动态 MatMul weight 示例。

因此准确结论是：

```text
test_dynamic_qdq_runtime_debug.py：没有动态第二输入的 MatMul 例子
test_qdq_utils.py：有，并验证整个 MatMul QDQ 改写被跳过
```

### 48.9 如果希望动态 W 也量化，需要什么改动

当前代码不能通过配置开启动态 weight quantization。要支持它，至少需要新增与现有静态路径
不同的实现：

1. `_find_qdq_target()` 不能再直接把动态 `input[1]` 返回为 `None`；
2. 需要为 weight 在运行时插入 `Quantize -> Dequantize`，而不是转换期生成 q/scale
   initializer；
3. 需要处理 MatMul 权重逻辑布局、transpose、rank-2 ABI 和 MX 对齐/padding；
4. 需要定义动态 weight scale 的计算粒度和生命周期；
5. 需要新增 ONNX graph rewrite、custom-op runtime 和数值/性能测试。

这不是把 `_create_weight_dq_nodes()` 直接用于动态 tensor 就能完成，因为
`simo/onnx/onnx_quant.py:980-985` 的 `_create_weight_dq_nodes()` 一开始就要求已有的
`target.quantized_weight.q` 和 `target.quantized_weight.scale`。

如果权重实际恒定，工程上更简单的解决方法是先做 constant folding，把它变成 initializer，
再调用 `insert_qdq_nodes()`。

### 48.10 一句话总结

```text
simo/onnx/onnx_quant.py:725-728 的 _find_qdq_target() 要求 MatMul.input[1]
必须能在 initializer/直接 Constant TensorProto 映射中找到；否则记录 dynamic_weight 并返回 None。
调用者随后原样保留 MatMul，因此既不离线量化权重、不插 weight Dequantize，也不插 activation QDQ。
test_dynamic_qdq_runtime_debug.py 没有这种例子；明确覆盖位于
simo/onnx/tests/test_qdq_utils.py:915-928 的 test_insert_qdq_nodes_skips_dynamic_weight_matmul()。
```

## 49. `quantize_dynamic` 后的 LSTM 能否由 ONNX Runtime CUDA 执行

### 49.1 修改内容

已修改 `like-useful/test-onnx-dynamic-quant-lstm.py`，脚本现在按以下顺序执行：

1. `like-useful/test-onnx-dynamic-quant-lstm.py:22-42` 的 `quantize_model()` 调用
   `quantize_dynamic()`，再用 ONNX checker 检查量化模型；
2. `like-useful/test-onnx-dynamic-quant-lstm.py:181-189` 的 `main()` 显式创建
   `CPUExecutionProvider` session，先完成 CPU 推理；
3. `like-useful/test-onnx-dynamic-quant-lstm.py:116-178` 的 `try_cuda()` 再创建
   CUDA session，运行相同输入并采集 ONNX Runtime profiling；
4. 最后创建禁止 CPU fallback 的严格 CUDA session，判断核心量化算子是否真的能够在
   CUDA 上执行。

输入由 `like-useful/test-onnx-dynamic-quant-lstm.py:45-58` 的
`make_input_cases()` 生成，覆盖三组动态 shape：

```text
seq_len=5, batch=3
seq_len=7, batch=2
seq_len=3, batch=4
```

`like-useful/test-onnx-dynamic-quant-lstm.py:61-87` 的 `run_cases()` 对每组输入检查
`output/hn/cn` 的 shape，并检查结果中不存在 NaN/Inf。CPU 和 CUDA 混合 session 的结果还会由
`like-useful/test-onnx-dynamic-quant-lstm.py:90-101` 的 `compare_outputs()` 逐个比较。

### 49.2 量化后不再是标准 ONNX `LSTM`

实测量化模型 `temp/test-lstm-quant.onnx` 的节点为：

```text
ai.onnx::Constant
com.microsoft::DynamicQuantizeLSTM
ai.onnx::Squeeze
```

也就是说，`quantize_dynamic()` 把原来的 `ai.onnx::LSTM` 替换成了 ONNX Runtime contrib
算子 `com.microsoft::DynamicQuantizeLSTM`。`like-useful/test-onnx-dynamic-quant-lstm.py:37-40`
也显式检查了这个图结构，避免误以为测试的仍是浮点 `LSTM`。

### 49.3 CPU 运行结果

指定环境中的 ONNX Runtime 版本为 `onnxruntime-gpu 1.27.0`。显式使用
`CPUExecutionProvider` 后，三组动态 shape 均成功运行：

```text
CPU_session_providers=['CPUExecutionProvider']
CPU: seq_len=5 batch=3 output_shapes=[(5, 3, 20), (1, 3, 20), (1, 3, 20)]
CPU: seq_len=7 batch=2 output_shapes=[(7, 2, 20), (1, 2, 20), (1, 2, 20)]
CPU: seq_len=3 batch=4 output_shapes=[(3, 4, 20), (1, 4, 20), (1, 4, 20)]
PASS: CPU verified 3 dynamic LSTM input shapes
```

因此量化模型本身有效，动态 `sequence length` 和 `batch` 也不是本次 CUDA 失败的原因。

### 49.4 第一个 CUDA 环境问题：cuDNN 不在默认动态库搜索路径

如果直接创建 CUDA session，当前环境会报告：

```text
Failed to load library .../libonnxruntime_providers_cuda.so with error:
libcudnn.so.9: cannot open shared object file: No such file or directory
```

这并不表示 conda 环境没有安装 cuDNN。实际文件位于：

```text
<conda-env>/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9
```

只是该目录不在进程默认的动态库搜索路径中。脚本在 CPU 验证之后，于
`like-useful/test-onnx-dynamic-quant-lstm.py:123-128` 的 `try_cuda()` 中调用：

```python
ort.preload_dlls(directory="")
```

这是 ONNX Runtime 提供的预加载接口，会从 pip 安装的 `nvidia` site-packages 中加载匹配的
CUDA/cuDNN 动态库。调用后 CUDA EP 能够正常创建，session 的 provider 为：

```text
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

也可以在启动 Python 前设置对应的 `LD_LIBRARY_PATH`，但只解决动态库加载问题，不解决下一节的
算子支持问题。

### 49.5 CUDA session 能运行不等于量化 LSTM 在 CUDA 上运行

`providers=['CUDAExecutionProvider', 'CPUExecutionProvider']` 允许 CPU fallback。只检查
`session.run()` 成功会产生误判：CUDA EP 不支持的节点会自动交给 CPU EP。

`like-useful/test-onnx-dynamic-quant-lstm.py:130-148` 的 `try_cuda()` 开启 profiling，
`like-useful/test-onnx-dynamic-quant-lstm.py:104-113` 的 `read_profile_assignments()` 读取每个
kernel 的实际 provider。实测结果是：

```text
CUDA_profile_assignments=[
  ('DynamicQuantizeLSTM', 'CPUExecutionProvider'),
  ('MemcpyFromHost', 'CUDAExecutionProvider'),
  ('Squeeze', 'CUDAExecutionProvider')
]
```

因此：

- CUDA EP 已经成功加载；
- 图尾部的 `Squeeze` 确实在 CUDA 上运行；
- 核心 `com.microsoft::DynamicQuantizeLSTM` 仍然在 CPU 上运行；
- CPU 与这个混合 session 的结果完全一致，实测所有输出的 `max_abs_diff` 都是 0。

### 49.6 严格 CUDA 验证为什么失败

`like-useful/test-onnx-dynamic-quant-lstm.py:163-176` 的 `try_cuda()` 使用：

```python
strict_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
providers=["CUDAExecutionProvider"]
```

禁止 CPU fallback 后，session 初始化立即失败：

```text
This session contains graph nodes that are assigned to the default CPU EP,
but fallback to CPU EP has been explicitly disabled by the user.
```

这和 profiling 的结论相互印证：当前 `onnxruntime-gpu 1.27.0` 构建没有可用于该模型中
`com.microsoft::DynamicQuantizeLSTM` 的 CUDA EP kernel。它是 CPU 实现的 contrib 算子；仅仅把
`CUDAExecutionProvider` 放在 provider 列表首位不会把它转换成 CUDA 实现。

### 49.7 如何处理

当前模型不能作为“全 CUDA 动态量化 LSTM”运行。可行选择是：

1. 需要使用 `quantize_dynamic()` 生成的 int8 LSTM 时，明确使用 CPU EP；这是当前模型的实际
   支持路径。
2. 必须在 GPU 上执行时，使用未被改写成 `DynamicQuantizeLSTM` 的浮点 FP32/FP16 LSTM，并单独
   验证所有节点的 CUDA provider 分配。
3. 必须同时满足 int8 LSTM 和 CUDA 时，需要换用具有对应 GPU kernel 的推理后端/图表示，或者
   实现并注册支持相同语义的 CUDA custom op；安装 cuDNN 或调整 `LD_LIBRARY_PATH` 本身不够。

验证命令：

```bash
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python \
  like-useful/test-onnx-dynamic-quant-lstm.py \
  > temp/test-onnx-dynamic-quant-lstm.log 2>&1
```

脚本退出码为 0；CPU 与允许 fallback 的混合 session 均完成三组输入验证，日志最终明确报告
`CUDA-only inference is unavailable`。这里保留退出码 0 是因为脚本已经完成预期诊断，并把
当前 ORT 不支持严格 CUDA 执行作为检测结果输出，而不是把 CPU 成功路径误报为测试失败。

## 50. ONNX Runtime `DynamicQuantizeLSTM` 是真 int8 计算还是 QDQ 模拟

本节中 ONNX Runtime 路径均相对于 code base：

```text
/softhome/like/package/onnxruntime
```

SIMO 路径均相对于 code base：

```text
/share/users/like/package/simo_conda_sglang
```

### 50.1 直接结论

1. **是的，`DynamicQuantizeLSTM` 的 CPU kernel 入口实现在
   `onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc`。**
2. 但完整实现不只在这一个文件中：该文件负责 operator、量化权重 prepack、量化参数读取和
   kernel 注册；LSTM 循环在 `onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc`，真正的
   动态量化与量化 GEMM 在 `onnxruntime/core/providers/cpu/rnn/rnn_helpers.cc`，底层使用 MLAS。
3. 它不是 SIMO 当前 MatMul 那种显式 `Quantize -> Dequantize -> MatMul` 图结构，而是
   **融合型量化算子**：图中只有一个 `com.microsoft::DynamicQuantizeLSTM`，量化、int8 GEMM 和
   反量化都封装在它的 CPU kernel 内部。
4. 它的两个主要矩阵乘确实使用 8-bit 整数输入做 GEMM。对当前
   `weight_type=QInt8` 模型，准确类型是 **UINT8 activation x INT8 weight，INT32 accumulate**，
   随后乘 scale 写回 FP32。通常可以简称为 int8 GEMM，但并不是严格的 INT8 x INT8。
5. 它不是端到端全 int8 LSTM：bias、gate 激活、cell state、hidden state以及 operator 输入/输出
   都是 FP32。

### 50.2 为什么该算子不属于标准 ONNX

在 ONNX code base `/softhome/like/package/onnx` 中搜索不到 `DynamicQuantizeLSTM`。它的 domain 是
`com.microsoft`，属于 ONNX Runtime contrib op，而不是 ONNX 标准算子。

schema 位于
`onnxruntime/core/graph/contrib_ops/quantization_defs.cc:657-758` 的
`ONNX_MS_OPERATOR_SET_SCHEMA(DynamicQuantizeLSTM, 1, ...)`：

- `onnxruntime/core/graph/contrib_ops/quantization_defs.cc:688-691` 定义 `X` 为输入 sequence；
- `onnxruntime/core/graph/contrib_ops/quantization_defs.cc:692-701` 定义量化权重 `W`、`R`；
- `onnxruntime/core/graph/contrib_ops/quantization_defs.cc:727-742` 定义 `W/R` 的 scale 和
  zero-point；
- `onnxruntime/core/graph/contrib_ops/quantization_defs.cc:755-757` 明确规定 `X` 和输出为
  `tensor(float)`，`W/R` 为 `tensor(uint8)` 或 `tensor(int8)`。

所以从 operator ABI 就可以看出：

```text
外部 X、H、C、Y：FP32
静态 W、R：INT8/UINT8
activation 的动态量化：发生在 kernel 内部
```

### 50.3 `quantize_dynamic()` 如何生成该节点

`onnxruntime/python/tools/quantization/quantize.py:875-977` 的 `quantize_dynamic()` 做了两项关键
配置：

- `onnxruntime/python/tools/quantization/quantize.py:937-940` 选择
  `QuantizationMode.IntegerOps`，不是 `QDQQuantizer`；
- `onnxruntime/python/tools/quantization/quantize.py:961-974` 创建 `ONNXQuantizer`，其中第 968 行
  明确写着动态 activation 只支持 `QUInt8`。

`onnxruntime/python/tools/quantization/registry.py:35-38` 把标准 `LSTM` 映射到
`LSTMQuant`。随后
`onnxruntime/python/tools/quantization/operators/lstm.py:17-117` 的
`LSTMQuant.quantize()` 完成改图：

1. `onnxruntime/python/tools/quantization/operators/lstm.py:26-38` 要求 `W`、`R` 是可量化的
   initializer，且均为 rank 3；
2. `onnxruntime/python/tools/quantization/operators/lstm.py:43-58` 对 `W` 和 `R` 做离线
   per-channel INT8 量化；
3. `onnxruntime/python/tools/quantization/operators/lstm.py:63-77` 将量化权重从标准 LSTM 布局
   转成该 contrib kernel 需要的布局；
4. `onnxruntime/python/tools/quantization/operators/lstm.py:84-88` 把 per-channel scale 和
   zero-point 整理成 `[num_directions, 4 * hidden_size]`；
5. `onnxruntime/python/tools/quantization/operators/lstm.py:90-116` 组合 float 输入、int8 权重、
   scale、zero-point，创建 `com.microsoft::DynamicQuantizeLSTM`。

当前 `temp/test-lstm-quant.onnx` 的实际 initializer 也与此一致：

```text
W_quantized: INT8, shape=[1, 10, 80]
R_quantized: INT8, shape=[1, 20, 80]
W_scale:     FLOAT, shape=[1, 80]
R_scale:     FLOAT, shape=[1, 80]
W/R zero point: INT8，全为 0
bias: FLOAT
```

这里 `hidden_size=20`，所以 `4 * hidden_size=80`。`per_channel=True` 因而为 80 个 gate/output
channel 分别保存 weight scale。

量化后的 graph 没有为 LSTM 展开显式 `QuantizeLinear/DequantizeLinear` 节点：

```text
FP32 X, FP32 h0, FP32 c0
            |
            v
com.microsoft::DynamicQuantizeLSTM
            |
            v
FP32 Y, FP32 Y_h, FP32 Y_c
```

这是 operator-oriented/fused dynamic quantization，而不是 graph-level QDQ 表示。

### 50.4 CPU kernel 的完整调用链

#### 50.4.1 kernel 入口和注册

`onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc:11-39` 定义
`DynamicQuantizeLSTM`，继承 `OpKernel` 和 `LSTMBase`。

`onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc:238-248` 使用
`ONNX_OPERATOR_TYPED_KERNEL_EX` 注册 kernel，并明确指定：

```cpp
kCpuExecutionProvider
```

`onnxruntime/contrib_ops/cpu/cpu_contrib_kernels.cc:111` 声明该 CPU kernel，
`onnxruntime/contrib_ops/cpu/cpu_contrib_kernels.cc:279` 将它加入 CPU contrib kernel registry。
在 `onnxruntime/contrib_ops/cuda/` 和 `onnxruntime/core/providers/cuda/` 中没有对应
`DynamicQuantizeLSTM` 注册，这也解释了上一节 profiling 中该节点只能分配给 CPU EP。

#### 50.4.2 权重保持 8-bit，并进行 MLAS prepack

`onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc:41-87` 的
`DynamicQuantizeLSTM::TryPackWeights()`：

- 第 48-51 行读取 8-bit `W/R` 的 `[K, N]`；
- 第 57 行记录 weight 是 signed INT8 还是 UINT8；
- 第 58 行调用 `MlasGemmPackBSize()`；
- 第 78-83 行直接把 8-bit weight 传给 `MlasGemmPackB()`。

这里的 prepack 是为 MLAS integer GEMM 调整布局，不是先把权重反量化成 float。
`onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc:94-115` 的
`DynamicQuantizeLSTM::PrePack()` 分别处理 input weight `W` 和 recurrent weight `R`。

#### 50.4.3 读取量化参数并进入量化版 LSTM

`onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc:166-235` 的
`DynamicQuantizeLSTM::Compute()`：

- 第 166-178 行取得 int8 `W/R` 以及对应 scale/zero-point；
- 第 190-206 行构造 `QuantizationParameter`；
- 第 208-216 行构造 `GemmWeights<uint8_t>`。这里模板存储类型写成 `uint8_t`，实际 signedness
  由 `is_W_signed/is_R_signed` 单独传递，所以仍可表示 INT8 weight；
- 第 235 行调用 `LSTMBase::ComputeImpl<float, uint8_t>()`。

模板参数已经表达了其混合精度边界：LSTM state/input/output 是 float，GEMM weight 是 8 bit。

`onnxruntime/core/providers/cpu/rnn/lstm_base.cc:22-27` 的 `LSTMBase::ComputeImpl()` 接收
`GemmWeights<WeightT>`；第 146-167 行创建 `UniDirectionalLstm<float>` 并调用其 `Compute()`。

#### 50.4.4 两个矩阵乘都进入量化 GEMM

`onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc:228-457` 的
`UniDirectionalLstm<T>::ComputeImpl()` 执行 LSTM 主循环：

1. **输入投影 `X * W`**：
   `onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc:284-293` 对所有有效 timestep 的
   `X` 调用 `ComputeGemm()`，结果写入 float `output_iofc`。
2. **循环投影 `H[t-1] * R`**：
   `onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc:332-349` 在每个 timestep 再调用
   `ComputeGemm()`，并把结果累加到已有的 `X * W` 结果。
3. **门函数和状态更新**：
   `onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc:463-588` 的
   `UniDirectionalLstm<T>::GateComputations()` 对 FP32 gate buffer 执行 bias、clip、sigmoid、tanh、
   cell state 和 hidden state 更新。

因此被量化的是 LSTM 中计算量最大的两个 affine/GEMM 部分，不是 sigmoid、tanh 和 state 更新。

### 50.5 `ComputeGemm()` 为什么能证明是真正的整数 GEMM

量化 overload 位于 `onnxruntime/core/providers/cpu/rnn/rnn_helpers.cc:247-317` 的
`rnn::detail::ComputeGemm()`：

1. `onnxruntime/core/providers/cpu/rnn/rnn_helpers.cc:271-276` 调用
   `GetQuantizationParameter()` 和 `ParQuantizeLinearStd()`，把本次 float activation `A`
   **实际写入 UINT8 buffer**；
2. `onnxruntime/core/util/qmath.h:50-109` 的 `GetQuantizationParameter()` 根据本次输入的 min/max
   动态计算 `a_scale` 与 `a_zero_point`；
3. `onnxruntime/core/util/qmath.h:122-135` 的 `ParQuantizeLinearStd()` 调用
   `MlasQuantizeLinear()` 生成 8-bit activation；
4. `onnxruntime/core/providers/cpu/rnn/rnn_helpers.cc:281-296` 计算
   `a_scale * weight_scale`，并创建 `MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR`；
5. `onnxruntime/core/providers/cpu/rnn/rnn_helpers.cc:298-316` 把量化后的 A、8-bit B、双方
   zero-point 和 int32 C buffer 交给 `MlasGemm()`。

MLAS 的 ABI 进一步证明累加类型：

- `onnxruntime/core/mlas/inc/mlas.h:613-633` 的 `MLAS_GEMM_QUANT_*_PARAMS` 定义 A/B 的
  signedness，并把 C 定义为 `int32_t*`；
- `onnxruntime/core/mlas/inc/mlas.h:540-582` 的 output processor 接收 `const int32_t* C`；
- `onnxruntime/core/mlas/lib/qpostprocessor.cpp:103-118` 的
  `MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR::ProcessImpl()` 明确说明把 C 转换回 floating point；
- `onnxruntime/core/mlas/lib/qpostprocessor.cpp:161-183` 将 int32 accumulator 转成 float，并乘
  scale；recurrent GEMM 使用 accumulate mode 时，再加到已有的 float `X * W` 结果。

对应的数学过程可简化为：

```text
A_q = quantize_uint8(A_fp32, a_scale, a_zero_point)       # 每次运行动态计算
B_q = quantize_int8(B_fp32, b_scale, b_zero_point)        # 模型转换时已完成

C_int32[m,n] = sum_k(
  (A_q[m,k] - a_zero_point) * (B_q[k,n] - b_zero_point)
)

C_fp32[m,n] = C_int32[m,n] * a_scale * b_scale[n]
```

当前 QInt8 模型的 `b_zero_point=0`；per-channel 变化的是 `b_scale[n]`。输入 `X * W` 的
activation 参数针对整次输入 GEMM 动态计算，循环中的 `H[t-1] * R` 则在每次 recurrent GEMM
调用时根据当时的 hidden-state 数据重新计算。

对当前模型：

```text
X * W:       UINT8 x INT8 -> INT32 -> FP32
H[t-1] * R:  UINT8 x INT8 -> INT32 -> FP32，并累加到 X * W
gate/state:  FP32
```

所以答案是：**矩阵乘核心是真正的量化整数 GEMM，不是先把 W 反量化成 FP32 后再调用浮点
MatMul。** 但它也不是端到端全 int8，正确描述是“动态 activation 量化 + 静态 weight 量化的
混合精度 LSTM”。

### 50.6 与 SIMO MatMul QDQ 的区别

SIMO 当前路径是显式 QDQ graph：

`simo/onnx/onnx_quant.py:608-669` 的 `_insert_qdq_in_graph()` 在找到 MatMul target 后：

- 第 636-642 行创建 activation QDQ；
- 第 643 行创建 weight DQ；
- 第 665 行仍把原始标准 `MatMul` 节点追加回 graph。

activation 路径由 `simo/onnx/onnx_quant.py:949-974` 的 `_create_qdq_nodes()` 创建：

```text
X(float) -> com.simo::Quantize -> packed q/scale
         -> com.simo::Dequantize -> X_dequant(float)
```

weight 路径由 `simo/onnx/onnx_quant.py:977-1031` 的 `_create_weight_dq_nodes()` 创建：量化后的
weight 和 scale 作为 initializer 写入 graph，第 1018-1025 行插入
`com.simo::Dequantize`，然后标准 `MatMul` 使用反量化输出。

custom-op ABI 也明确了数据类型：

- `simo/onnx/ort_plugin/simo_qdq_ops.cc:301-305` 的 `SimoQuantizeOp::Compute()` 接收
  `Tensor<float>`，输出 packed `Tensor<uint8_t>`；
- `simo/onnx/ort_plugin/simo_qdq_ops.cc:404-408` 的 `SimoDequantizeOp::Compute()` 接收 packed
  `uint8`，输出 `Tensor<float>`；
- `simo/onnx/ort_plugin/simo_qdq_ops.cc:504-510` 的 `RegisterQdqOps()` 把 Quantize 和
  Dequantize 注册为 CUDA custom op。

因此 SIMO 当前 MatMul 的数据流是：

```text
X_fp32 -> Quantize -> X_q -> Dequantize -> X_dq_fp32 ---+
                                                        +-> ai.onnx::MatMul -> FP32 output
W_fp32 --offline quantize--> W_q -> Dequantize -> W_dq_fp32 ---+
```

在当前实现中，没有把这组 `com.simo::Quantize/Dequantize + ai.onnx::MatMul` 融合成一个直接消费
packed q/scale 的整数 MatMul kernel。由于标准 MatMul 接收到的两个输入均为 float，它执行的是
**浮点 MatMul**。SIMO QDQ 确实产生量化误差、保存量化 weight，并真实执行 Q/DQ CUDA kernel，
但它当前不是 int8 MatMul 加速路径。

需要强调：**QDQ 是一种图表示，不天然等于“假量化”。** 如果某个 execution provider 能识别
QDQ pattern 并把它融合成量化 kernel，底层同样可以是真正的整数计算。但 SIMO 这里使用
`com.simo` 自定义 Q/DQ，当前代码没有对应的 MatMul fusion，所以不能仅凭 graph 中出现 QDQ 就
声称 MatMul 已经使用 int8 arithmetic。

### 50.7 对照总结

| 项目 | ONNX Runtime `DynamicQuantizeLSTM` | SIMO 当前 MatMul QDQ |
|---|---|---|
| graph 表示 | 单个融合型 `com.microsoft` 算子 | 显式 `Quantize -> Dequantize -> MatMul` |
| weight | 转换期量化为 INT8/UINT8 | 转换期量化并保存 packed q/scale |
| activation | kernel 内运行时动态量化为 UINT8 | 自定义 Quantize 后立即 Dequantize 为 float |
| MatMul/GEMM 输入 | UINT8 activation、INT8 weight（当前模型） | 两个输入均为反量化后的 float |
| 累加 | INT32 | FP32 |
| GEMM 后 | 乘 scale 转回 FP32 | 已直接得到 FP32 |
| gate/state | FP32 | 不适用 |
| 当前 EP | CPU EP | Q/DQ 在 CUDA，标准 MatMul 处理 float |

一句话总结：

```text
DynamicQuantizeLSTM 不是 SIMO 当前的“QDQ 后再做浮点 MatMul”；它把动态量化和两个 MLAS
整数 GEMM 融合在 CPU kernel 内。当前 QInt8 模型的 GEMM 是 U8 x S8 -> S32，再缩放回 FP32；
因此 GEMM 是真量化计算，但整个 LSTM 不是全 int8。
```

## 51. 谁在 2026-07-20 升级了 `nvidia-fabricmanager`

### 51.1 结论

发起这次升级的用户是 **`scbjtfy`（UID 1000）**。

需要区分两个身份：

```text
发起/请求升级的登录用户：scbjtfy (1000)
实际执行 dpkg 的有效用户：root
```

APT/dpkg 必须以 root 完成安装，所以 `dpkg.log` 本身只记录 root 权限下的包状态变化；但
`apt/history.log` 的 `Requested-By`、`auth.log` 的 SSH/sudo 会话和时间窗口三者一致地指向
`scbjtfy`。

### 51.2 直接证据：APT history

`/data/like/temp/log/apt/history.log:185-190` 是完整的同一笔事务：

```text
Start-Date: 2026-07-20  11:31:04
Commandline: apt upgrade
Requested-By: scbjtfy (1000)
...
nvidia-fabricmanager:amd64 (590.48.01-0ubuntu1, 610.43.02-1ubuntu1)
End-Date: 2026-07-20  11:37:31
```

这里的 `Commandline` 不是 `/usr/bin/unattended-upgrade`，而是交互式/显式执行的
`apt upgrade`；`Requested-By` 是 APT 为这笔事务记录的请求者。

### 51.3 SSH 与 sudo 会话的时间关联

`/data/like/temp/log/auth.log:2067-2072` 显示：

```text
11:16:40  scbjtfy 从 172.16.3.240 登录，SSH session 17399
11:16:46  scbjtfy 执行 sudo，打开 TTY pts/81 的 /usr/bin/bash
           USER=root，root sudo session opened by scbjtfy(uid=1000)
```

该 root shell 在整个 APT 事务期间保持打开，直到
`/data/like/temp/log/auth.log:2097-2102`：

```text
11:38:21  scbjtfy 断开 SSH
11:38:21  scbjtfy 的 session 17399 和对应 root sudo session 关闭
```

时间关系为：

```text
11:16:40  scbjtfy SSH 登录
11:16:46  scbjtfy -> sudo /usr/bin/bash (root)
11:31:04  apt upgrade 开始
11:32:20  nvidia-fabricmanager 开始 unpack/upgrade
11:32:55  nvidia-fabricmanager 610.43.02 配置完成
11:37:31  apt 事务结束
11:38:21  scbjtfy SSH/root 会话关闭
```

独立的二进制登录记录 `/data/like/temp/log/wtmp` 也显示同一条会话：

```text
scbjtfy  pts/81  172.16.3.240  Mon Jul 20 11:16 - 11:38  (00:21)
```

这比单独看 `dpkg.log` 更重要：APT 事务完全落在 `scbjtfy` 打开的 root shell 生命周期内。

### 51.4 dpkg 记录的实际包变化

`/data/like/temp/log/dpkg.log:1219-1223` 记录 unpack 阶段：

```text
2026-07-20 11:32:20 upgrade nvidia-fabricmanager:amd64 \
  590.48.01-0ubuntu1 610.43.02-1ubuntu1
```

`/data/like/temp/log/dpkg.log:1429-1432` 记录 configure 阶段并最终变为：

```text
status installed nvidia-fabricmanager:amd64 610.43.02-1ubuntu1
```

这不是只升级了 Fabric Manager 一个包。APT history 同一行事务还包含
`nvidia-driver`、`nvidia-dkms`、`nvidia-firmware`、`cuda-drivers`、`nvidia-kernel-common`、
`libnvidia-*` 以及 CUDA/NCCL、内核和 Docker 等大量升级；Fabric Manager 是这次
`apt upgrade` 的一个成员。

### 51.5 apt term log 与 system log 的交叉验证

`/data/like/temp/log/apt/term.log:378` 显示 dpkg 事务在 11:31:04 开始，
`/data/like/temp/log/apt/term.log:1021` 显示在 11:37:31 结束，与 history 完全一致。

升级完成后，`/data/like/temp/log/syslog:149737-149803` 显示 systemd 在 11:38:17 停止并重新
启动 NVIDIA Fabric Manager；这属于包配置/服务重启的后处理，不是另一个用户再次执行升级。

### 51.6 排除 unattended-upgrades、cron 和其他登录用户

1. **不是 unattended-upgrades。**
   `/data/like/temp/log/unattended-upgrades/unattended-upgrades.log:112-116` 显示
   2026-07-20 06:02:25 启动检查，并在 06:02:27 明确记录：

   ```text
   No packages found that can be upgraded unattended and no pending auto-removals
   ```

   11:31 的 history 也写的是 `Commandline: apt upgrade`，不是 unattended 命令。
2. **不是 cron 触发的 apt。**
   `auth.log` 在 11:30 只看到 `like` 的 cron session，在 11:35 只看到 root 的常规
   `debian-sa1` cron session；`syslog:146638` 和 `syslog:147965` 进一步显示这两个时刻的
   命令分别是代码/历史同步和 `debian-sa1`，没有 apt/dpkg 命令。APT history 仍明确标为
   `Requested-By: scbjtfy`。
3. **不是同时登录的 `wangzelin`。**
   `/data/like/temp/log/auth.log:2064-2066` 只记录 `wangzelin` 在 11:16:19 登录；在该时间窗口
   没有其 sudo root session 或 apt command。紧接着的 `scbjtfy` 登录、sudo root shell 和
   APT `Requested-By` 才构成完整链路。
4. **11:38 的 PackageKit/systemd 活动不是发起者。**
   这些是包安装结束后的服务重载/重启日志；它们没有改变 APT history 中已经记录的请求用户。

### 51.7 升级后的 NVIDIA 版本不一致

此次升级还解释了日志中随后出现的 GPU 问题。`/data/like/temp/log/syslog:150088-150095` 和
`/data/like/temp/log/fabricmanager.log:28740-28762` 显示：

```text
Fabric Manager version: 610.43.02
GPU driver interface version: 610.43.02
running driver version: 590.48.01
fabric manager ... don't match with driver version 590.48.01
systemd ... nvidia-fabricmanager.service: Failed to start
```

也就是说，`scbjtfy` 发起的 `apt upgrade` 将用户态/服务侧 Fabric Manager 和多个 NVIDIA
包升级到了 610.43.02，但当时加载的内核/运行时 NVIDIA driver 仍为 590.48.01，导致版本不匹配。
这与当前日志中的 `nvidia-smi`/Fabric Manager 异常相符；问题不是 dpkg 没有安装成功，而是
驱动组件没有保持同一版本族。

### 51.8 证据强度与限制

结论的最强证据是 `apt/history.log:186-187` 的 `Commandline + Requested-By`，并由
`auth.log:2067-2072` 和 `auth.log:2097-2100` 的同一 SSH/root 会话闭合验证。

`auth.log` 没有记录交互式 root shell 内每一条按键命令，因此不能从日志中还原
`scbjtfy` 输入 `apt upgrade` 的具体 shell 命令行；但 APT 自己记录的 `Requested-By` 已明确给出
请求者，且时间、TTY、SSH 来源和 root session 全部吻合。因而可以确定：

```text
升级请求者：scbjtfy
dpkg 执行者：root（由 scbjtfy 的 sudo root shell 发起）
```

## 52. `nvidia-fabricmanager` 是直接指定升级，还是依赖被动升级

### 52.1 结论先行

准确的分类是：

```text
不是：apt install/apt upgrade nvidia-fabricmanager 这种命令行直接指定
不是：作为其他新安装包的 Depends 被动拉入
是：已经安装且被标记为 manual 的 nvidia-fabricmanager，
    被无包参数的 apt upgrade 纳入批量升级
```

因此“被动升级”如果指“用户没有在命令行点名，`apt upgrade` 自动把所有可升级的已安装包
一起处理”，这个说法成立；如果指“它是某个其他包的依赖、因此被新安装”，则不成立。

### 52.2 APT history 的 `Upgrade` 与 `Install` 区分

`/data/like/temp/log/apt/history.log:185-190` 中命令是：

```text
Commandline: apt upgrade
```

这条命令没有包名参数。该事务把包分成两类：

- `Install:`（第 188 行）列出新安装的包，并明确标注了很多 `automatic`；
- `Upgrade:`（第 189 行）列出已有包的旧版本到新版本替换。

`nvidia-fabricmanager:amd64 (590.48.01-0ubuntu1, 610.43.02-1ubuntu1)` 位于
`Upgrade:` 列表，不在 `Install:` 列表。这说明 590.48.01 在事务开始前已经安装，APT 只是
把它升级到仓库中的 610.43.02；不是因为本次安装另一个包才首次安装 Fabric Manager。

`/data/like/temp/log/apt/term.log:580-581` 也显示：

```text
Unpacking nvidia-fabricmanager (610.43.02-1ubuntu1) over (590.48.01-0ubuntu1) ...
```

关键词 `over` 表明这是 upgrade/replace，不是 fresh install。

### 52.3 依赖关系检查

对本次事务使用的两个版本查询包元数据，结果是：

```text
nvidia-fabricmanager 590.48.01-0ubuntu1:
  Depends: libc6, zlib1g

nvidia-fabricmanager 610.43.02-1ubuntu1:
  Depends: libc6, zlib1g
  Suggests: libnvidia-compute

cuda-drivers 590.48.01-0ubuntu1:
  Depends: nvidia-driver (>= 590.48.01)

cuda-drivers 610.43.02-1ubuntu1:
  Depends: nvidia-driver (>= 610.43.02)
```

当前安装状态在 `/var/lib/dpkg/status:22370-22383`（`nvidia-driver`）和
`/var/lib/dpkg/status:22395-22407`（`nvidia-fabricmanager`）中也一致：

- `nvidia-driver` 依赖一组 `libnvidia-*`、`nvidia-dkms` 和 kernel 包，但没有
  `Depends: nvidia-fabricmanager`；
- `cuda-drivers` 依赖 `nvidia-driver`，没有反向把 `nvidia-fabricmanager`列为依赖；
- `nvidia-fabricmanager`自身只依赖 libc/zlib，并不依赖 `nvidia-driver`。

所以不能把这次 FM 升级解释为“升级 `nvidia-driver` 时由 Depends 自动安装/升级 FM”。两个包
属于同一 NVIDIA/CUDA 软件家族、版本号同步，是仓库发布和批量升级的关联，但不是当前包元数据
中的依赖边。

### 52.4 APT 的手动/自动标记

当前系统的 APT 标记进一步支持这个结论：

```text
apt-mark showmanual:
  cuda-drivers
  nvidia-fabricmanager

apt-mark showauto:
  nvidia-dkms
  nvidia-driver
  nvidia-firmware
  nvidia-kernel-common
  nvidia-kernel-source
  nvidia-settings
  ...
```

`nvidia-fabricmanager` 在 `showmanual` 中，而不在 `showauto` 中，表示 APT 当前把它视为
手动安装的顶层包，而不是某个依赖链自动引入的包。这个标记不能还原最初是哪一次安装命令，
但足以说明本次事务中它不是新依赖的自动安装项。

### 52.5 为什么它和 NVIDIA driver 一起升级

`apt/history.log:189` 同时列出了：

```text
cuda-drivers:       590.48.01 -> 610.43.02
nvidia-driver:      590.48.01 -> 610.43.02
nvidia-dkms:        590.48.01 -> 610.43.02
nvidia-fabricmanager: 590.48.01 -> 610.43.02
```

`apt upgrade` 会枚举所有当前已安装、且仓库有可升级候选版本的包；因此同一 NVIDIA 发布批次
中的多个包会在同一事务被选中。`/data/like/temp/log/apt/term.log:432-479` 先 unpack
driver/CUDA 相关包，`580-581` 再 unpack Fabric Manager，这只是 dpkg 的事务顺序，不能据此
推断“前一个包触发了后一个包”。真正的选择依据已经由 history 的 `Upgrade:` 列表表达：它们
都是已安装包的版本升级。

### 52.6 最准确的一句话

```text
scbjtfy 执行的是无包参数的 `apt upgrade`，没有直接指定
`nvidia-fabricmanager`；但 `nvidia-fabricmanager` 在升级前已经是手动安装的包，
且不属于 nvidia-driver/cuda-drivers 的 Depends。APT 因为它有新的 610.43.02 候选版本，
把它作为批量升级的一员升级了。它是“未点名但已安装包的自动纳入”，不是“依赖导致的新安装”。
```

仅凭 `apt/history.log` 不能进一步证明用户当时是否主观上想升级 NVIDIA；它只能证明执行了
全局 `apt upgrade`，而 APT 将包括 Fabric Manager 在内的所有可升级包一起处理。

## 53. conda 环境中的 `onnxruntime-gpu` 是否由本地 ONNX Runtime 源码编译

分析对象：

```text
conda 环境：/share_data/users/like/miniconda3/envs/simo_sglang
本地 ONNX Runtime checkout：/softhome/like/package/onnxruntime
已安装包：onnxruntime-gpu==1.27.0
```

### 53.1 结论

**当前环境中的 `onnxruntime-gpu` 不是从 `/softhome/like/package/onnxruntime` 这个目录直接
编译安装的，而是 pip 安装的、已经由发布方编译好的 Linux binary wheel。**

这里的“不是本地编译”需要准确理解：wheel 中确实包含
`libonnxruntime.so.1.27.0`、`libonnxruntime_providers_cuda.so` 和
`libonnxruntime_providers_tensorrt.so` 等本地机器上的原生 `.so` 文件；但是这些 `.so` 已经
在 wheel 发布前编译好，当前 `pip install` 只是下载并解压它们，没有调用
`/softhome/like/package/onnxruntime` 的 CMake/C++/CUDA 构建流程。

### 53.2 直接证据：安装目录中的 wheel 与 pip cache 完全对应

`pip show` 报告的安装位置是：

```text
/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages
```

该目录下的
`onnxruntime_gpu-1.27.0.dist-info/WHEEL` 包含：

```text
Wheel-Version: 1.0
Generator: setuptools (82.0.1)
Tag: cp312-cp312-manylinux_2_27_x86_64
Tag: cp312-cp312-manylinux_2_28_x86_64
```

`Tag` 是 CPython 3.12 的 manylinux binary-wheel 标签；它能说明安装结果不是 editable
目录链接，但标签本身不能单独证明 wheel 是官方构建还是某人从源码构建的。
同一目录的 `INSTALLER` 内容为 `pip`，并且没有 `direct_url.json`。如果是
`pip install /softhome/like/package/onnxruntime`、`pip install -e /softhome/like/package/onnxruntime`
或直接以本地路径安装 wheel，pip 通常会在 `.dist-info/direct_url.json` 中记录本地路径；当前
安装没有这个记录。这个事实是辅助证据，最终结论由下面的索引记录和 wheel hash 确定。

更强的证据来自 pip 缓存：

```text
/share_data/users/like/.cache/pip/http-v2/f/c/f/7/1/
  fcf71a1333a4ffe06a1ec64c02815ebc20be0bd0d04d508d8c691923.body
```

这个 `.body` 是一个大小为 `220305746` bytes 的 ZIP wheel。pip 索引缓存
`/share_data/users/like/.cache/pip/http-v2/c/e/6/0/3/ce603105b286dc4c750f96738a23d53ba802cb4af27f02ac6b4ba5ed.body`
列出了完全相同的文件：

```text
filename: onnxruntime_gpu-1.27.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
size:     220305746
sha256:   b2b8d6afabf3c23c2c698639a16388816dc7298f388a9ddddf504b283990ea01
```

对缓存 wheel 计算得到的 SHA256 也是：

```text
b2b8d6afabf3c23c2c698639a16388816dc7298f388a9ddddf504b283990ea01
```

并且缓存 wheel 与已安装目录中的关键文件逐字节相同，包括：

```text
onnxruntime/capi/libonnxruntime.so.1.27.0
onnxruntime/capi/libonnxruntime_providers_cuda.so
onnxruntime/capi/build_and_package_info.py
onnxruntime_gpu-1.27.0.dist-info/METADATA
```

当前 pip 配置（`pip config list -v`）的索引为：

```text
global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'
```

因此可以把本次安装链路写成：

```text
pip index/mirror
  -> onnxruntime_gpu-1.27.0-cp312-...manylinux...whl
  -> /share_data/users/like/miniconda3/envs/simo_sglang/.../site-packages
```

这不是从 `/softhome/like/package/onnxruntime` 读取源码后再生成 wheel 的链路。

### 53.3 为什么本地源码版本也显示 1.27.0

`/softhome/like/package/onnxruntime/VERSION_NUMBER` 是 `1.27.0`，git checkout 当前为
`v1.27.0`。这只能说明本地 checkout 与已安装 wheel 属于同一个上游 release，不能证明 wheel
就是由这个 checkout 构建的。官方发布 wheel 当然也会使用同一个版本号。

本地源码的 `setup.py:727-730` 从 `VERSION_NUMBER` 读取版本号，
`setup.py:823-854` 的 `save_build_and_package_info()` 在构建时生成
`onnxruntime/capi/build_and_package_info.py`；因此已安装文件中看到：

```python
package_name = 'onnxruntime-gpu'
__version__ = '1.27.0'
cuda_version = '13.0'
```

只能证明 wheel 是 CUDA 13.0 的 GPU 包，不能作为“由本地 checkout 编译”的证明。相同的文件
也会出现在官方发布的 wheel 中。

### 53.4 它和 SIMO 本地 ONNX Runtime 代码的关系

`simo/pyproject.toml:35-40` 把 `onnxruntime-gpu` 放在 `[onnx]` optional extra 中；它不是
`[dev]` extra 的依赖。因此 `pip install -e ".[dev]"` 本身不会从
`/softhome/like/package/onnxruntime` 编译或安装 ONNX Runtime。需要 ONNX 功能时，通常是单独
安装 `onnxruntime-gpu`，或者安装 `.[onnx]`。

SIMO 只把已安装的 ORT 当作运行时：

- `simo/onnx/runtime.py:18-25` 读取已安装 `onnxruntime-gpu` 的版本，并把它放入 custom-op
  runtime cache key；
- `simo/onnx/runtime.py:146-150` 导入 `onnxruntime` Python 模块并创建 `InferenceSession`；
- `simo/onnx/ort_plugin/build_runtime.py:47-91` 使用
  `torch.utils.cpp_extension.load()` 编译的是 SIMO 自己的
  `libSimoOnnxCustomOps_sm90.so`，不是 ONNX Runtime 核心库；
- `simo/onnx/ort_plugin/build_runtime.py:13-14` 和 `:63-68` 使用 SIMO 包内的 ORT API
  headers 以及 CUDA include 路径，说明 custom op 是针对 ORT ABI 编译的，但不会自动调用
  `/softhome/like/package/onnxruntime` 的源码构建系统。

所以两者的关系是：

```text
pip wheel 提供 ONNX Runtime 的 Python API、ORT 核心 .so 和 CUDA/TensorRT EP
                         |
                         +-- SIMO 在其 ABI/header 之上另外编译 custom-op .so
```

### 53.5 什么时候才算“从这个本地目录编译”

必须能看到明确的本地源码安装或 wheel 构建命令，例如：

```bash
pip install -e /softhome/like/package/onnxruntime
pip install /softhome/like/package/onnxruntime/dist/onnxruntime_gpu-*.whl
```

或者先在该 checkout 中按 ONNX Runtime 的 `tools/ci_build/build.py` 流程执行
`--use_cuda --build --build_wheel`，再把生成的 wheel 路径传给 pip。此类安装应能在 pip
报告或 `.dist-info/direct_url.json` 中留下本地路径线索；当前环境没有这类线索，反而有上述
官方索引 wheel 的 hash 证据。

### 53.6 最终判断

```text
当前 onnxruntime-gpu==1.27.0：是 pip 从配置的 PyPI 镜像取得的预编译 wheel；
不是 pip 从 /softhome/like/package/onnxruntime checkout 现场编译出来的。

本地 checkout 的 v1.27.0 版本与 wheel 版本相同，只表示源码版本对齐；
SIMO 的 build_runtime.py 后续会使用 ORT 的 ABI/header 编译 SIMO custom op，
但不会因此反向编译或替换 onnxruntime-gpu 本身。
```

## 54. 本地 ONNX Runtime 源码能否构建 `onnxruntime-gpu` wheel 和 `CUDAExecutionProvider`

本节中的源码路径均相对于 ONNX Runtime code base：

```text
/softhome/like/package/onnxruntime
```

### 54.1 结论

**可以。** 这份代码是完整的 ONNX Runtime `v1.27.0` checkout，具备两项能力：

1. 使用 `--use_cuda --build_wheel` 构建包名为 `onnxruntime-gpu` 的 Python binary wheel；
2. 编译并打包真正的 `CUDAExecutionProvider`，包括 CUDA kernels、provider factory、Python/C
   API 注册入口以及 Linux 下的 `libonnxruntime_providers_cuda.so`。

但“源码具备能力”和“当前目录已经生成产物”是两件事。当前 checkout 中实测：

```text
onnxruntime_gpu-*.whl 数量：             0
libonnxruntime_providers_cuda.so 数量：  0
CMakeCache.txt 数量：                    0
```

所以第 53 节“当前 conda 环境中的 wheel 不是由该目录构建”与本节结论不矛盾：该源码**能够**
构建 GPU wheel，只是此前并没有用它构建当前已安装的 wheel。

### 54.2 从构建参数到 `onnxruntime-gpu` 包名的调用链

#### 第一步：构建入口显式支持 CUDA 和 wheel

`tools/ci_build/build_args.py:593-600` 的 `add_python_binding_args()` 定义：

```text
--enable_pybind
--build_wheel
--wheel_name_suffix
```

`tools/ci_build/build_args.py:641-648` 的 `add_execution_provider_args()` 定义：

```text
--use_cuda
--cuda_version
--cuda_home
--cudnn_home
```

因此 CUDA EP 和 Python wheel 都是该仓库正式构建入口的一部分，不是外部补丁。

#### 第二步：`--use_cuda` 转成 CMake CUDA 开关

`tools/ci_build/build.py:713-727` 的 `generate_build_tree()` 在 `use_cuda` 为真时：

```text
-DCMAKE_CUDA_COMPILER=<cuda_home>/bin/nvcc
-Donnxruntime_USE_CUDA=ON
-Donnxruntime_CUDA_VERSION=<cuda_version>
-Donnxruntime_CUDA_HOME=<cuda_home>
-Donnxruntime_CUDNN_HOME=<cudnn_home>
```

`tools/ci_build/build.py:2308-2321` 的 `main()` 还会因为 `--build_wheel` 自动启用 pybind，并为
非 training Python wheel 启用 shared-library build。

#### 第三步：CMake 真正编译 CUDA provider

`cmake/CMakeLists.txt:72-79` 声明 `onnxruntime_USE_CUDA` 和 CUDA EP 相关选项。
`cmake/CMakeLists.txt:742-769` 在该开关启用时依次执行：

```text
setup_cuda_compiler()
setup_cuda_architectures()
enable_language(CUDA)
添加 USE_CUDA=1
把 cuda 加入 ONNXRUNTIME_PROVIDER_NAMES
```

`cmake/CMakeLists.txt:1466-1474` 随后通过 `find_package(CUDAToolkit REQUIRED)` 找 CUDA
toolkit。`cmake/onnxruntime_providers.cmake:65-67` 和 `:116-119` 将
`onnxruntime_providers_cuda.cmake` 加入构建。

`cmake/onnxruntime_providers_cuda.cmake:35-66` 收集
`onnxruntime/core/providers/cuda/` 和 `onnxruntime/contrib_ops/cuda/` 下的 `.cc`、`.cu`、
`.cuh` 源码；该源码目录当前共有 382 个文件。随后
`cmake/onnxruntime_providers_cuda.cmake:169-193` 创建 shared-library target：

```text
onnxruntime_providers_cuda
```

Linux 产物就是：

```text
libonnxruntime_providers_cuda.so
```

#### 第四步：CUDA provider 被放入 Python wheel

`cmake/onnxruntime_python.cmake:1085-1092` 在 `onnxruntime_USE_CUDA` 为真时，把
`onnxruntime_providers_cuda` 和 `onnxruntime_providers_shared` 复制到 wheel staging
目录的 `onnxruntime/capi/`。

`setup.py:349-400` 将 Linux 文件名设为 `libonnxruntime_providers_cuda.so`，并把它加入
需要打包的 native libraries。

最后，`tools/ci_build/build.py:1924-2004` 的 `build_python_wheel()` 在 `use_cuda=True` 时
调用：

```text
python setup.py bdist_wheel --wheel_name_suffix=gpu --cuda_version=<version>
```

`setup.py:797-800` 再把基础包名 `onnxruntime` 改成：

```text
onnxruntime-gpu
```

因此完整链路是：

```text
build.sh --use_cuda --build_wheel
  -> build.py: generate_build_tree()
  -> CMake: onnxruntime_USE_CUDA=ON
  -> nvcc 编译 CUDA kernels
  -> 生成 libonnxruntime_providers_cuda.so
  -> build.py: build_python_wheel()
  -> setup.py --wheel_name_suffix=gpu
  -> onnxruntime_gpu-1.27.0-<python>-<platform>.whl
```

### 54.3 这份源码是否真的实现了 `CUDAExecutionProvider`

答案也是肯定的，不只是“wheel 名字带 gpu”。关键实现链路如下：

1. `onnxruntime/core/providers/cuda/cuda_execution_provider.cc:327-341` 的
   `CUDAExecutionProvider::CUDAExecutionProvider()` 创建名为
   `kCudaExecutionProvider` 的 GPU provider，调用 `cudaSetDevice()`、
   `cudaDeviceSynchronize()` 和 `cudaGetDeviceProperties()` 初始化目标设备。
2. `onnxruntime/core/providers/cuda/cuda_execution_provider.cc:551-630` 开始声明并注册 CUDA
   kernels；例如 `Gemm`、`MatMul`、`MatMulInteger` 都有 CUDA kernel registration。
3. `onnxruntime/core/providers/cuda/cuda_execution_provider.cc:3188-3204` 的
   `InitializeRegistry()`/`CUDAExecutionProvider::GetKernelRegistry()` 建立并返回 CUDA kernel
   registry。
4. `onnxruntime/core/providers/cuda/cuda_provider_factory.cc:41-54` 的
   `CUDAProviderFactory::CreateProvider()` 实际构造 `CUDAExecutionProvider`；同文件
   `:179-180` 从 provider options 创建 factory。
5. `onnxruntime/python/onnxruntime_pybind_state.cc:1075-1091` 在 Python 请求
   `CUDAExecutionProvider` 时加载 CUDA provider 信息并创建 factory。
6. `onnxruntime/core/session/provider_bridge_ort.cc:2648-2687` 提供 C API
   `OrtSessionOptionsAppendExecutionProvider_CUDA()`，最终把 CUDA factory 放入 session
   options。

所以源码生成的 GPU wheel 具备完整的 Python 调用路径：

```text
InferenceSession(..., providers=["CUDAExecutionProvider"])
  -> Python pybind provider selection
  -> CUDAProviderFactory
  -> CUDAExecutionProvider
  -> CUDA kernel registry
  -> CUDA/cuDNN/cuBLAS 等运行时
```

### 54.4 “提供 CUDA EP”不等于“所有 ONNX 节点都在 CUDA 上运行”

`CUDAExecutionProvider` 只接管它具有匹配 kernel、opset、dtype、shape/attribute 约束的节点。
`onnxruntime/core/providers/cuda/cuda_execution_provider.cc:3386-3448` 的
`CUDAExecutionProvider::GetCapability()` 会逐节点查询 CUDA kernel；找不到 kernel 时直接跳过。
同函数 `:3451-3487` 还会检查特定算子的属性限制，不满足时允许回退到 CPU。

因此：

- 普通 `MatMul`、`Add`、`Conv` 等受支持节点可以分配给 CUDA EP；
- 不受支持的节点默认交给 `CPUExecutionProvider`；
- 禁用 CPU fallback 后，只要图中还存在未被 CUDA EP 接管的节点，session 初始化就会失败；
- 前面分析过的 `com.microsoft::DynamicQuantizeLSTM` 没有对应 CUDA kernel，即使安装了
  `onnxruntime-gpu` 或自己编译 CUDA EP，它仍不会自动变成 CUDA 算子。

判断一个模型是否真正使用 CUDA，不能只看 `ort.get_available_providers()` 或
`session.get_providers()`；还需要使用 profiling 查看每个 kernel 的 provider，或者禁用 CPU
fallback 后运行一个已知由 CUDA 支持的模型。

### 54.5 当前机器是否具备本地构建条件

本次只检查了工具链，没有执行耗时很长的完整 ORT CUDA build。当前可见条件如下：

```text
Python:       3.12.12
GCC/G++:      11.4.0
CMake:        4.2.0
Ninja:        1.13.0
/usr/local/cuda -> /usr/local/cuda-13.1
nvcc:         13.1.115
cuDNN:        9.19.0（位于 conda site-packages/nvidia/cudnn）
GPU:          8 x NVIDIA H100, compute capability 9.0
driver:       590.48.01
```

这与源码要求基本匹配：

- `cmake/external/cuda_configuration.cmake:57-60` 要求 CUDA 至少为 12.0；
- 同文件 `:62-73` 有 CUDA 13+ 的专门处理；
- 同文件 `:129-136` 的 CUDA 12.8/13 默认架构列表都包含 `sm90`；
- `cmake/external/cuDNN.cmake:3-20` 会在显式 `CUDNN_PATH`、Python
  `site-packages/nvidia/cudnn` 或 CUDA toolkit 路径中查找 cuDNN header/library；
- `cmake/CMakeLists.txt:274-275` 要求 GCC 至少为 11.1，当前 GCC 11.4 满足。

但是当前 shell 的 `PATH`/PyTorch 自动探测优先得到的是 `/usr/local/cuda-12.8`，系统
`ldconfig` 默认还能看到 cuDNN 8；当前**已安装**的官方 wheel 则记录 CUDA 13.0，并链接
cuDNN 9。`v1.27.0` 源码并非只能用 CUDA 13 构建；如果目标是与当前 wheel 的 major runtime
stack 对齐，构建时应显式传入 CUDA 13.1 和 conda 中 cuDNN 9 的路径，不能依赖自动探测。

此外，CUDA EP 是大型构建，默认 CUDA 13 架构列表包含多个 GPU 架构。当前机器只有 H100 时，
显式设置 `CMAKE_CUDA_ARCHITECTURES=90` 可以显著减少编译时间和 wheel 体积。完整构建仍可能受
Python build dependencies、外部 CMake 依赖下载、内存、磁盘空间及编译器兼容性影响，因此在
没有真正完成一次 build 前，只能判断“源码和主要工具链具备能力”，不能宣称本地构建已经成功。

### 54.6 针对当前机器的参考构建命令

以下命令没有在本次分析中执行，用于从该 checkout 构建仅面向 H100/sm90 的本地 GPU wheel：

```bash
ORT_ROOT=/softhome/like/package/onnxruntime
CONDA_ENV=/share_data/users/like/miniconda3/envs/simo_sglang
CUDA_HOME=/usr/local/cuda
CUDNN_HOME="$CONDA_ENV/lib/python3.12/site-packages/nvidia/cudnn"

cd "$ORT_ROOT"
PATH="$CONDA_ENV/bin:$PATH" ./build.sh \
  --config Release \
  --update \
  --build \
  --build_wheel \
  --skip_tests \
  --cmake_generator Ninja \
  --parallel 8 \
  --nvcc_threads 1 \
  --use_cuda \
  --cuda_home "$CUDA_HOME" \
  --cudnn_home "$CUDNN_HOME" \
  --cuda_version 13.1 \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=90
```

预期 wheel 位于：

```text
/softhome/like/package/onnxruntime/build/Linux/Release/dist/
  onnxruntime_gpu-1.27.0-cp312-cp312-linux_x86_64.whl
```

普通本地构建通常得到 `linux_x86_64` wheel。要生成类似 PyPI 的 portable manylinux wheel，
还需要在相应 manylinux 构建环境中设置 `AUDITWHEEL_PLAT` 并执行 auditwheel repair；对应逻辑在
`setup.py:122` 和 `setup.py:315-334`。即使版本相同，本地 wheel 也不会与官方 wheel
字节完全一致，CUDA/cuDNN、编译器、GPU architectures、TensorRT 开关和 manylinux 环境都会
影响产物。

建议在独立 conda/venv 中安装和验证本地 wheel，避免覆盖当前 SIMO 环境中已工作的官方包。
最低限度应检查：

```bash
python -m pip install --no-deps /path/to/onnxruntime_gpu-1.27.0-*.whl
python -c 'import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())'
```

随后使用一个 CUDA 支持的模型、禁止 CPU fallback 并查看 profiling，才能证明生成的 wheel
不仅包含 provider 名称，而且 CUDA provider 动态库及实际 kernel 都能加载运行。

### 54.7 当前已安装 wheel 的 CUDA 运行验证

当前已安装的预编译 wheel 报告：

```text
ort.__version__ = 1.27.0
ort.get_available_providers() =
  ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
ort.get_device() = 'GPU'
```

直接创建 CUDA session 时，因为 pip 安装的 cuDNN 9 目录不在默认动态库搜索路径，首先出现：

```text
libcudnn.so.9: cannot open shared object file
```

调用 `ort.preload_dlls(directory="")` 后，使用 `session.disable_cpu_ep_fallback=1` 创建仅请求
`CUDAExecutionProvider` 的 `Add` 模型并运行成功，输出与 NumPy 结果完全相同。这证明当前机器
的 CUDA 13/cuDNN 9/driver 运行链在正确预加载后可用；但它验证的是当前安装的官方 wheel，
不是尚未执行的本地源码 build。

### 54.8 最终回答

```text
问题 1：/softhome/like/package/onnxruntime 能否编译 onnxruntime-gpu wheel？
回答：能。正式链路是 --use_cuda + --build_wheel，最终由 setup.py 生成
      onnxruntime_gpu-1.27.0-*.whl，并打包 libonnxruntime_providers_cuda.so。

问题 2：这份代码能否提供 CUDAExecutionProvider？
回答：能。源码包含 CUDA EP、kernel registry、provider factory、Python/C API 注册和
      CUDA provider shared library 构建逻辑。

边界：当前 checkout 尚未实际构建；CUDA EP 只执行其支持的节点，不保证任意模型或任意
      contrib op 都能在 CUDA 上运行。
```

## 55. `/share/users/like/temp/gtest.png` 图片内容

这张图片不是 Google Test（gtest）的运行界面、测试日志或错误截图。文件名虽然是
`gtest.png`，但文件名与实际画面内容没有直接关系。

图片主体是《龙珠》中的少年孙悟空动漫立绘，主要视觉特征包括：

- 黑色尖刺状头发，人物面向前方并摆出握拳的战斗姿势；
- 身穿橙红色龟仙流武道服，系黑色腰带；
- 胸前圆形徽章中有“龟（亀）”字；
- 手腕佩戴蓝色护腕，脚穿蓝白色鞋子；
- 身后可以看到少年孙悟空的棕色尾巴；
- 没有测试结果、终端文字、源代码或其他 gtest 相关信息。

文件属性为：

```text
路径：/share/users/like/temp/gtest.png
格式：PNG
尺寸：655 x 1220
颜色模式：8-bit RGBA
背景：透明（alpha 范围为 0～255，包含完全透明像素）
```

因此，这个文件可以概括为一张**透明背景的少年孙悟空全身人物素材图**，而不是一张
gtest 测试图片。

## 56. 在北京住建委网站查询期房项目网签情况

### 56.1 查询项目整体销售/网签状态的具体路径

截至 2026-07-22，推荐使用项目公示入口，而不是先进入“期房合同信息查询”：

1. 打开北京住建委官网：<https://zjw.beijing.gov.cn/>。
2. 选择“房屋”业务下的“房地产开发管理”。也可以直接打开：
   <https://zjw.beijing.gov.cn/bjjs/fdckfgl/index.shtml>。
3. 在“预售许可”区域找到“已办理预售许可项目公示”（页面标题为“项目信息公示”），点击“查询”。
   直接入口通常为：
   <https://bjjs.zjw.beijing.gov.cn/eportal/ui?isTrue=1&pageId=307670>。
4. 在查询条件中填写或选择：
   - “项目名称”：优先使用预售许可证或住建委备案中的项目名称；
   - “开发单位”：可用开发商全称缩小结果范围；
   - “项目地址”和“所属区县”：用于排除同名项目；
   - “现房/期房”：选择“期房”；
   - “预售证号”：已知时可直接使用。
5. 点击“查询”，在结果中确认项目名称、坐落、开发单位和预售证信息，进入该项目的“详细信息”。
6. 在项目详情中选择对应的预售楼栋，打开“楼盘表”。楼盘表按楼栋、楼层和房号列出每套房的公示状态；逐套查看即可判断该项目哪些房屋已签约、已预订或仍显示为未签约/可售。

住建委的“房地产开发管理”页面明确说明，该数据库可按“项目名称”“开发单位”“项目地址”
“现房/期房”“预售证号”“所属区县”查询新建商品房及已办理预售许可项目。若项目在售楼处使用的
营销名查不到，应改用预售许可证上的备案项目名，或结合开发单位、地址查询。

### 56.2 如何从楼盘表判断“网签”

楼盘表中的文字或图例应以页面当前显示为准，常见含义如下：

| 页面状态 | 含义 | 是否计入已网签 |
| --- | --- | --- |
| 已签约、已预售，或已联机备案 | 已提交商品房预售合同网上签约/备案流程 | 是，按页面的具体标签统计 |
| 已预订、已认购 | 只完成认购书或网上预订，尚未完成预售合同网签 | 不应直接当作已网签 |
| 未签约、可售 | 当前公示中没有网签标识 | 否，但仍应核对是否存在限售、抵押或开发企业更新延迟 |
| 不可售、抵押等 | 当前不能按普通可售房源交易 | 不作为已网签房源统计 |

北京住建委公开的预售管理说明要求楼盘表公示网上已预订、已签约房屋；预售合同网上签约后，
楼盘表会标识该套房屋已预售。部分页面使用颜色图例，颜色可能随页面版本变化，因此应以楼盘表
旁边的文字图例为准，不要只凭红色、粉色或褐色等颜色推断。公开数据是交易管理系统的公示快照，
合同撤销、换房、备案变更或开发企业尚未更新时，页面状态可能与销售现场口径不同。

### 56.3 查询某一份期房合同的另一条路径

如果问题不是“这个项目卖了多少套”，而是核验某一位购房人的具体合同，则走：

1. 官网首页 → “房屋” → “房地产交易” → “网签合同”；或直接打开
   <https://zjw.beijing.gov.cn/bjjs/fdcjy/wqht/index.shtml>。
2. 找到“期房合同网上签约查询”，进入“期房合同信息查询”；也可从查询中心
   <https://zjw.beijing.gov.cn/bjjs/cxzx29/index.shtml> 选择“房地产交易 → 期房合同信息查询”。
3. 按页面要求填写合同号（或合同编码）、买方名称、证件号码、网签密码和验证码。

这条查询需要合同号、购房人身份信息和签约时设置的密码，不能用来仅凭项目名称查看全项目的
销售套数；项目整体情况应使用 56.1 的“项目信息公示 → 楼盘表”。

### 56.4 结果解释和核对建议

- “已预订”不等于“已网签”。北京住建委公布的流程是先认购/预订，再在规定期限内签订商品房预售合同；合同网上提交并联机备案后才属于合同网签/备案状态。
- 统计时应明确口径：通常只统计楼盘表标为“已签约/已预售/已联机备案”的房屋，另行列出“已预订/已认购”，不要把两者相加后称为网签套数。
- 页面若显示项目已取得预售许可但没有可售楼栋，可能是已售完、部分楼栋尚未公示、项目状态已变更，或查询条件使用了营销名而非备案名；可用预售证号、开发商全称和区县重新查询。
- 如需对合同效力、退房或备案解除作正式证明，应以购房人通过合同号和密码查询到的合同/联机备案信息，以及开发企业和区住建部门出具的材料为准，网页楼盘表只适合作为公开信息核对。

相关官方入口：

- 北京住建委“房地产开发管理”：<https://zjw.beijing.gov.cn/bjjs/fdckfgl/index.shtml>
- 北京住建委“网签合同”：<https://zjw.beijing.gov.cn/bjjs/fdcjy/wqht/index.shtml>
- 北京住建委“查询中心”：<https://zjw.beijing.gov.cn/bjjs/cxzx29/index.shtml>
- 北京住建委“项目信息公示”动态查询：<https://bjjs.zjw.beijing.gov.cn/eportal/ui?isTrue=1&pageId=307670>

## 57. 提交 `d19622e34e8cd86d7d6e2194f9539a7906049856` 修改说明

### 57.1 提交基本信息和总体结论

```text
commit:  d19622e34e8cd86d7d6e2194f9539a7906049856
parent:  ee319fc686ecdea40b82f952376d433734e10d58
author:  haifeng <hfxu@siorigin.com>
time:    2026-07-21 08:10:45 UTC（北京时间 2026-07-21 16:10:45）
subject: [feat] Add ONNX accuracy debug compute output comparison (model-opt/simo!190)
规模:    10 files changed, 2549 insertions(+), 1 deletion(-)
```

这个提交包含两个相互配合的功能：

1. 新增 ONNX 精度调试工具：将两个 ONNX 模型中 `MatMul`、`Gemm`、`Conv` 节点的内部输出
   暂时暴露为 graph output，在相同输入下分别用 ONNX Runtime 执行，再生成逐节点误差报告。
2. 新增 `com.simo::Quantize` 和 `com.simo::Dequantize` 的 CPU custom-op 实现，使带 SIMO Q/DQ
   节点的量化 ONNX 模型可以选择 `CPUExecutionProvider` 执行，方便与浮点参考模型做精度对比。

它**没有修改** `simo/onnx/onnx_quant.py` 的量化图转换流程，也没有改变 MatMul/Conv 本身的
量化策略；它增加的是运行、采集中间结果和比较误差所需的调试基础设施。

提交新增 6 个文件、修改 4 个文件：

| 文件 | 变化 | 作用 |
| --- | ---: | --- |
| `examples/accuracy_debug/run_compare_onnx.py` | `+145` | ONNX 模型比较命令行入口 |
| `examples/accuracy_debug/run_compare_onnx.sh` | `+44` | 用环境变量封装 CLI 的示例脚本 |
| `simo/accuracy_debug/onnx_runner.py` | `+409` | ONNX 扫描、插桩、执行、对齐和比较的核心实现 |
| `simo/accuracy_debug/__init__.py` | `+8` | 导出新的 ONNX accuracy-debug API |
| `simo/onnx/ort_plugin/simo_qdq_cpu_ops.cc` | `+1224` | SIMO Quantize/Dequantize CPU kernel |
| `simo/onnx/ort_plugin/build_runtime.py` | `+1` | 将 CPU Q/DQ 源文件加入 custom-op 共享库构建 |
| `simo/onnx/ort_plugin/custom_op_library.cc` | `+8/-1` | 根据环境变量选择注册 CPU 或原 CUDA Q/DQ op |
| `simo/onnx/ort_plugin/simo_qdq_ops.h` | `+1` | 声明 `RegisterCpuQdqOps()` |
| `tests/simo_quant/test_accuracy_debug_onnx.py` | `+187` | ONNX 精度调试功能测试 |
| `simo/onnx/tests/test_cpu_qdq_custom_ops.py` | `+522` | CPU Q/DQ 编译、运行和格式一致性测试 |

### 57.2 ONNX 中间计算结果比较的实现

#### 57.2.1 查找需要比较的节点

`simo/accuracy_debug/onnx_runner.py:23` 的 `DEFAULT_COMPUTE_OP_TYPES` 默认只包含：

```text
MatMul, Gemm, Conv
```

`simo/accuracy_debug/onnx_runner.py:60` 的 `find_onnx_compute_outputs()` 遍历 ONNX graph：

- 只选择标准 ONNX domain，即 `node.domain` 为空的节点；
- 按 `op_types` 过滤算子类型；
- 按节点名称应用 `include`/`exclude` glob；
- 节点没有名称时生成 `MatMul_<graph index>` 形式的调试名称；
- 记录每个非空输出的 tensor name、输出序号和用于报告对齐的 capture name。

因此，即使自定义 domain 中也有一个叫 `MatMul` 的节点，它也不会被这个函数直接选中。这一点由
`tests/simo_quant/test_accuracy_debug_onnx.py:70` 的
`test_find_onnx_compute_outputs_filters_standard_domain_and_globs()` 覆盖。

#### 57.2.2 把中间 tensor 临时变成 graph output

ONNX Runtime 通常只返回原模型声明的 graph output。为取得中间节点结果，
`simo/accuracy_debug/onnx_runner.py:119` 的 `_instrument_model_outputs()` 会：

1. 深拷贝模型，不原地修改输入模型；
2. 尝试调用 ONNX shape inference；
3. 查找目标 tensor 对应的 `ValueInfoProto`；
4. 将目标 tensor 追加到 `model.graph.output`；
5. 找不到类型/shape 信息时，退化为未知 shape 的 FLOAT output。

插桩模型只以序列化内存数据传给 ORT，不会覆盖磁盘上的原 ONNX 文件。

#### 57.2.3 输入和 ONNX Runtime session

`simo/accuracy_debug/onnx_runner.py:172` 的 `normalize_onnx_inputs()` 支持三种输入：

- Python `dict[str, numpy.ndarray/torch.Tensor/array-like]`；
- `.npz` 文件；
- `torch.save()` 保存的 mapping，既可直接保存输入字典，也可保存 `{"inputs": {...}}`。

Torch tensor 会先 `detach().cpu().numpy()`。`simo/accuracy_debug/onnx_runner.py:147` 的
`_session_options()` 可以设置 ORT graph optimization level，并按需调用
`simo.onnx.runtime.register_custom_ops()`。`simo/accuracy_debug/onnx_runner.py:189` 的
`_create_session()` 分别为参考模型和实际模型创建 ORT session。

默认 providers 为：

```text
CUDAExecutionProvider, CPUExecutionProvider
```

默认 optimization level 是 `disable`。精度调试时关闭图优化有助于保留原始计算节点和中间输出；
也可以显式选择 `basic`、`extended` 或 `all`。

#### 57.2.4 采集、对齐和比较

`simo/accuracy_debug/onnx_runner.py:211` 的 `collect_onnx_compute_outputs()` 完成单个模型的流程：

```text
选择节点输出
  -> 插桩为 graph output
  -> 创建 ORT session
  -> session.run(output_names, feeds)
  -> NumPy 转 CPU torch.Tensor
  -> 生成 TensorSummary/CaptureResult
```

`simo/accuracy_debug/onnx_runner.py:308` 的 `compare_onnx_models()` 对参考模型和实际模型分别执行
上述流程，然后复用原有 `compare_capture_results()` 和 `write_report()`。比较指标包括 MSE、RMSE、
MAE、最大绝对/相对误差、cosine similarity、SQNR；若节点被标为 `class_logits`，还会计算 sigmoid
分数误差、阈值翻转率和 top-k overlap。

该函数提供两种对齐方式：

- `align_by="name"`：默认方式，以节点 capture name 对齐。名称不同的节点不会产生比较项；底层
  `simo/accuracy_debug/comparator.py:21` 的 `compare_capture_results()` 会跳过实际模型中不存在的名称，
  不会报 missing-node 错误。
- `align_by="order"`：`simo/accuracy_debug/onnx_runner.py:279` 的 `_align_outputs_by_order()` 按 graph
  拓扑顺序配对，要求两侧被选输出数量相同，且每一对 `op_type` 相同；实际模型节点名可以不同，
  报告统一使用参考模型名称。数量或 op type 不匹配会立即抛出 `ValueError`。

比较模式会通过 `simo/accuracy_debug/onnx_runner.py:261` 的 `_force_tensor_mode()` 强制保存 tensor，
因为只有 summary 无法计算逐元素误差。若提供 `output_dir`，会写出：

```text
summary.json
anomaly.json
layer_metrics.json
summary.md
```

`simo/accuracy_debug/onnx_runner.py:372` 的 `scan_onnx_model()` 则只扫描一个模型，适合检查 NaN、Inf
等异常，不进行双模型比较。

#### 57.2.5 Python API 导出

`simo/accuracy_debug/__init__.py:5-9` 和 `simo/accuracy_debug/__init__.py:17-30` 新增公开导出：

```python
collect_onnx_compute_outputs
compare_onnx_models
scan_onnx_model
```

低层的 `find_onnx_compute_outputs()` 和 `normalize_onnx_inputs()` 仍可从
`simo.accuracy_debug.onnx_runner` 直接导入。

### 57.3 命令行和示例脚本

`examples/accuracy_debug/run_compare_onnx.py:31` 的 `main()` 新增 CLI，主要参数包括：

- `--ref-model`、`--actual-model`、`--input`、`--output-dir`；
- `--include`、`--exclude` 和可重复的 `--op-type`；
- `--align-by name|order`；
- 公共或独立的 `--providers`、`--ref-providers`、`--actual-providers`；
- `--register-simo-ops-for-ref/actual` 和 `--simo-custom-ops-library`；
- `--optimization-level`、`--top-k`、`--eps`；
- class-logits 的 semantic pattern、阈值和 top-k 参数。

`examples/accuracy_debug/run_compare_onnx.sh:4-44` 用环境变量构造这些参数。比较普通 FP ONNX 和
SIMO 量化 ONNX，并让量化模型的 Q/DQ 在 CPU 上运行，可使用：

```bash
SIMO_ONNX_QDQ_PROVIDER=CPU \
python examples/accuracy_debug/run_compare_onnx.py \
  --ref-model /path/ref.onnx \
  --actual-model /path/quantized.onnx \
  --input /path/inputs.npz \
  --output-dir /path/report \
  --providers CPUExecutionProvider \
  --register-simo-ops-for-actual \
  --align-by name
```

若两个模型量化前后节点名发生变化，应考虑 `--align-by order`，但必须确认两侧选中的计算节点数量
和 op-type 顺序确实一一对应，否则“拓扑顺序相同”并不自动等于“语义上是同一个层”。

### 57.4 SIMO Q/DQ CPU custom op

#### 57.4.1 构建和注册方式

`simo/onnx/ort_plugin/build_runtime.py:70-76` 的 `build_sm90_runtime()` 将新增的
`simo_qdq_cpu_ops.cc` 加入 `libSimoOnnxCustomOps_sm90.so`。共享库仍同时包含原 CUDA Q/DQ、Triton
loader 和生成的 cubin，因此：

- CPU Q/DQ 不是一个独立 `.so`；
- 库名仍然是 `libSimoOnnxCustomOps_sm90.so`；
- 从源码调用 `build_sm90_runtime()` 时仍会构建 SM90 cubin、包含 CUDA 头文件并链接 CUDA driver，
  “运行 Q/DQ 用 CPU”不等于“构建过程不需要 CUDA”。

`simo/onnx/ort_plugin/simo_qdq_ops.h:13-14` 增加 `RegisterCpuQdqOps()` 声明。
`simo/onnx/ort_plugin/custom_op_library.cc:13` 的 `QdqDomain()` 在第一次创建静态 `com.simo` domain 时
读取环境变量：

```text
SIMO_ONNX_QDQ_PROVIDER=CPU  -> RegisterCpuQdqOps(domain)
其他值或未设置             -> RegisterQdqOps(domain)，保留原 CUDA 路径
```

判断是大小写敏感的精确字符串 `CPU`，而且 domain 是进程内静态对象，所以环境变量必须在该进程
第一次调用 `register_custom_ops()` 之前设置，最稳妥的方式是在启动 Python 前设置。第一次注册后
再修改变量，不会重建 domain。

#### 57.4.2 CPU kernel 的输入输出和限制

`simo/onnx/ort_plugin/simo_qdq_cpu_ops.cc:91-128` 的 `IoDtype` 支持 Q/DQ 外部浮点类型：

```text
fp32, fp16, bf16
```

`simo/onnx/ort_plugin/simo_qdq_cpu_ops.cc:1138` 的 `QuantizeCpuDirectOp`：

```text
一个 fp32/fp16/bf16 输入 -> quantized UINT8 + scale UINT8
Execution Provider       -> CPUExecutionProvider
com.simo opset           -> version 2
```

`simo/onnx/ort_plugin/simo_qdq_cpu_ops.cc:1173` 的 `DequantizeCpuDirectOp` 执行反方向转换。
这里 scale 的 ONNX carrier 类型是 `UINT8`；FP32 scale 以 4 个原始字节存储，并不代表 scale 的
数学类型是整数。

`simo/onnx/ort_plugin/simo_qdq_cpu_ops.cc:160` 的 `ResolveSpec()` 先复用嵌入式 SM90 semantic
spec 的布局元数据，未命中时再构造 CPU fallback spec。这个提交的参数化端到端测试明确验证了
以下组合；该表表示**已验证集合**，不应扩大解释成所有 resolver 可能返回组合的稳定公共契约：

| 量化 family | dtype | granularity | scale mode/典型 block |
| --- | --- | --- | --- |
| MX | `mxfp8_e4m3`, `mxfp8_e5m2` | `per_group` | `e8m0_floor`, block 32 |
| MX | `mxfp6_e2m3`, `mxfp6_e3m2` | `per_group` | `e8m0_floor`, block 32 |
| MX | `mxfp4_e2m1` | `per_group` | `e8m0_floor`, block 32 |
| NV | `nvfp4_e2m1` | `per_group` | `e4m3`, block 16 |
| MX integer | `mxint8`, `mxint4` | `per_group` | `e8m0_floor`, block 32 |
| Flex | `fp8_e4m3`, `int8` | `per_group`, `per_channel`, `per_block` | `fp32`, test block 32 |

主要边界条件：

- `simo/onnx/ort_plugin/simo_qdq_cpu_ops.cc:342` 的 `DequantizeShape()` 和
  `:373` 的 `QuantizeShape()` 只接受 rank-2 tensor；kernel 也要求 contiguous rank-2 输入。
- `simo/onnx/ort_plugin/simo_qdq_cpu_ops.cc:257` 的 `SpecFromKernelInfo()` 要求 per-channel 的
  canonical `axis=0`。
- MX 最后一维 K 必须能被 `block_size` 整除；4-bit、6-bit 格式还必须满足相应 packed K 比例。
- 该实现只补充 Quantize/Dequantize CPU kernel，不是所有 SIMO custom op 的 CPU 实现。

#### 57.4.3 实际量化/反量化计算

`simo/onnx/ort_plugin/simo_qdq_cpu_ops.cc:565-647` 实现格式最大值和 scale 计算；
`:667-705` 将归一化浮点数编码为 MX/Flex 数据；`:711-759` 实现 4-bit 和 6-bit packing；
`:761-810` 解码 packed 数据。

`simo/onnx/ort_plugin/simo_qdq_cpu_ops.cc:819` 的 `DequantizeCpu()` 遍历二维 tensor，根据 MX、
per-group、per-channel 或 per-block 布局找到对应 scale，然后计算：

```text
output = decoded_quantized_value * scale
```

`simo/onnx/ort_plugin/simo_qdq_cpu_ops.cc:891` 的 `QuantizeCpu()` 在各 scale block 内计算有限值的
`amax`，生成 scale，再除以 scale、舍入/截断到目标格式并完成 packing。`:1001` 和 `:1072` 的
kernel 类负责校验输入 shape、计算 packed/scale shape、分配 ORT 输出并调用上述 CPU 循环。

最后，`simo/onnx/ort_plugin/simo_qdq_cpu_ops.cc:1209` 的 `RegisterCpuQdqOps()` 为 fp32、fp16、bf16
分别注册 Quantize 和 Dequantize，共 6 个 custom-op type specialization。

### 57.5 新增测试及本机验证结果

`tests/simo_quant/test_accuracy_debug_onnx.py:70-187` 新增 6 项测试，覆盖：

- 标准 domain 和 glob 过滤；
- MatMul/Conv 中间结果及 summary 采集；
- 按名称比较并写出 JSON/Markdown report；
- 不同节点名按拓扑顺序对齐；
- 输出数量不匹配时报错；
- `.npz` 与 torch-saved mapping 输入。

`simo/onnx/tests/test_cpu_qdq_custom_ops.py:30-522` 新增 21 项展开后的测试，覆盖：

- 注册源码检查和 C++17 syntax-only 编译；
- `mxfp6_e2m3` Q/DQ 共享库端到端运行；
- 14 种 dtype/granularity 组合与 PyTorch reference 的逐元素一致性；
- FP6 packing、FP8/MX/NVFP4 carrier 和 scale dtype；
- 必要时查找或即时构建 `libSimoOnnxCustomOps_sm90.so`。

在 `/share_data/users/like/miniconda3/envs/simo_sglang/` 中对当前代码实际执行：

```text
python -m pytest -q tests/simo_quant/test_accuracy_debug_onnx.py
结果：6 passed in 3.03s

python -m pytest -q simo/onnx/tests/test_cpu_qdq_custom_ops.py
结果：21 passed, 14 warnings in 81.12s
```

14 个 warning 均来自 PyTorch `torch.jit.script_method` deprecation，不是本提交测试失败。

### 57.6 最终概括

```text
这个提交解决的问题：
  给定浮点参考 ONNX 和 SIMO 量化 ONNX，能够抓取两者 MatMul/Gemm/Conv 的中间输出，
  按节点名或计算顺序比较误差，并生成逐层精度报告。

为什么同时加入 CPU Q/DQ：
  量化 ONNX 含 com.simo::Quantize/Dequantize；没有 CPU kernel 时，CPUExecutionProvider
  无法加载/执行这些节点。新增 CPU kernel 后，可以在不执行 CUDA Q/DQ kernel 的情况下
  跑量化模型，便于确定误差来自哪一层。

没有做什么：
  没有修改 ONNX 量化插入策略，没有把 MatMul/Conv 自定义实现改成 CPU，也没有让所有
  SIMO custom op 都支持 CPU。CPU Q/DQ 当前有 rank-2、格式、granularity 和 axis 限制。
```

## 58. 提交 `ee319fc686ecdea40b82f952376d433734e10d58` 做了哪些重构

### 58.1 提交范围和总体结论

提交信息为：

```text
commit:  ee319fc686ecdea40b82f952376d433734e10d58
parent:  ea99510019e20ae1d982aebc65b2ad5d6fa188e8
author:  dehua <hchu@siorigin.com>
date:    2026-07-17 08:30:34 +0000
subject: refactor_onnx (model-opt/simo!188)
```

规模为 66 个文件、5022 行新增、4382 行删除，其中 4 个文件新增、46 个修改、16 个删除。
它不是单纯的函数改名或格式整理，而是一次 **SIMO ONNX QDQ v2 的端到端重构**：

```text
旧链路：
ONNX protobuf 手工改图
  -> v1 QDQ 属性
  -> 运行时检查源码/hash/cache
  -> 缺少 .so 时现场编译
  -> ORT 自定义算子调用内嵌 cubin

新链路：
ONNX GraphSurgeon 统一改图
  -> v2 语义属性 + 严格配置/类型检查
  -> 声明式 Triton kernel spec 在构建期 AOT 编译
  -> cubin 嵌入 libSimoOnnxCustomOps_sm90.so
  -> .so 随 wheel 安装，运行阶段只加载、不再编译
```

以下路径和行号均以该提交自身的文件快照为准，而不是后续 `HEAD` 的行号。

### 58.2 ONNX 公共接口和图改写器重构

#### 1. `insert_qdq_nodes` 被替换为 `apply_qdq_quantization`

- `simo/onnx/api.py:15-25` 的 `quantize()` 改为调用
  `simo/onnx/onnx_quant.py:238-345` 的 `apply_qdq_quantization()`。
- 配置参数从只接受 JSON 路径，扩展为 `str | Path | Mapping | QuantizeConfig`。
- 新增 `simplify=False` 关键字参数；显式开启时才调用
  `simo/onnx/onnx_quant.py:437-449` 的 `simplify_onnx_model()`，并禁止 ONNXSlim 把
  `Gemm` 融合掉。简化失败会记录 warning 后使用原模型。
- 公开的 `simo.onnx.quantize()` 名字保持不变，但内部的
  `insert_qdq_nodes`、`rewrite_dynamic_qdq` 已删除。因此直接导入这些旧内部函数的代码会失效。

#### 2. 从手写 protobuf 编辑器切换到 ONNX GraphSurgeon

旧代码使用 `GraphBuilder`、`ONNXTensorTransformer`、`ActivationQdqPlan`、`OpTarget` 等多层中间
对象，手动维护 `GraphProto.node`、initializer、shape 和名称。新代码引入
`onnx-graphsurgeon`，由 `simo/onnx/onnx_quant.py:172-235` 的 `SIMOGraphEditor` 统一创建
常量、节点、输出 dtype 和唯一名称；主图与递归子图在 `apply_qdq_quantization()` 中采用同一条
处理路径。

这带来几个具体变化：

- 不再先执行 `_name_nodes()` 重命名原模型所有匿名或重名节点，只保证新生成的节点和 tensor
  名称唯一，减少对原图的无关修改。
- 使用 `graph.cleanup().toposort()` 清理和排序改写后的图。
- 能处理子图中的 Constant，以及子图捕获外层 initializer 的情况。
- 忽略自定义 domain 中恰好也叫 `MatMul/Gemm/Conv` 的节点，只处理标准 ONNX domain。
- 输入模型先深拷贝，改写结果再写回模型外壳；无任何插入时直接返回原始内容，避免 GraphSurgeon
  round-trip 改变一个本应 no-op 的模型。

GraphSurgeon 不能无损保留所有 protobuf 字段，因此新增
`simo/onnx/onnx_quant.py:348-434` 的 `validate_graphsurgeon_conversion()`：遇到 sparse
initializer、未加载的 external tensor、graph/value/node/tensor metadata、不能保留的 attribute
类型等情况会明确拒绝，而不是静默丢字段。

#### 3. 目标选择和错误处理更严格

`simo/onnx/onnx_quant.py:625-736` 的 `prepare_quantization_target()` 集中负责匹配目标、排除项、
常量权重、dtype 和离线权重量化：

- 仍然只量化第二个输入是 initializer/Constant 的 `MatMul/Gemm/Conv`；动态权重记录
  `dynamic_weight` 并跳过。
- 多条 module config 同时匹配时由“第一条生效”改成“最后一条生效”，便于前面写通配默认值、
  后面写更具体的覆盖规则。
- 被选中的静态权重如果量化失败，不再记录 warning 后静默跳过，而是抛出带节点名和算子名的
  `ValueError`，防止输出模型少量化节点却不易察觉。
- 检查 activation、weight、bias、output 的模型计算 dtype 是否一致。
- 根据模型计算 dtype 选择 FP32、FP16 或 BF16 的 Dequantize 自定义算子，并检查相应 ONNX
  opset 是否足够。

### 58.3 配置格式升级为严格的 QDQ v2 配置

`simo/onnx/onnx_quant.py:452-579` 的 `load_quantization_config()` 不再兼容旧包装格式，而是要求
直接使用 `QuantizeConfig` 字段：

```json
{
  "module_configs": [
    {
      "targets": ["MatMul"],
      "input": {"dtype": "mxint8", "axis": -1},
      "weight": {"dtype": "int8", "axis": 0}
    }
  ]
}
```

主要兼容性变化为：

- 拒绝顶层 `quantization_config`、`quant_method`、`quant_algo`。
- 拒绝 module 中的 `targets_op_types`，统一改为 `targets`。
- 支持 `Linear -> MatMul/Gemm`、`Conv2d/Conv3d -> Conv` 别名和 `"*"` 通配目标。
- 严格拒绝未知字段；不再因 Pydantic 默认忽略字段而把拼写错误吞掉。
- 当前 v2 只支持 `algorithm.name=naive`、静态 weight、动态 input、`abs_max`、对称量化和
  `half_even`；不支持 output/bias 量化、zero point 和 `pre_quant_opt`。
- `simo/onnx/onnx_quant.py:582-622` 的 `validate_quantization_spec()` 把实际 runtime 支持的
  dtype、granularity、group/block size、scale mode 组合变成显式白名单。

原来 `simo/onnx/quant_schema/` 下的 10 个示例 JSON 全部删除，提交中没有一对一替代文件；旧配置
需要改成上述直接 `module_configs + targets` 格式。

### 58.4 activation 和 weight 布局逻辑重构

#### 1. 删除大而全的 `LayoutRecipe`

`simo/onnx/layout_utils.py` 从 96 行缩减到 36 行，删除 `LayoutRecipe`、padding 状态和 NumPy
执行逻辑，只保留四个纯 permutation 工具：

```text
move_axis_to_last_permutation
move_axis_to_first_permutation
invert_permutation
compose_permutations
```

padding、reshape、unpad 等决策移到真正拥有上下文的 activation/weight 处理代码中，避免同一套
布局状态在多个对象间重复表达。

#### 2. `QuantizedWeight` 改成“载荷 + 恢复步骤”描述

`simo/onnx/weight_quant.py:22-33` 的 `QuantizedWeight` 从 `q/scale/logical_shape/LayoutRecipe`
改为：

```text
quantized_values     实际量化字节载荷
scale_values         scale 字节载荷
source_shape         原始 ONNX 权重形状
dequantized_shape    custom DQ 输出的规范 rank-2 形状
dequantize_spec      DQ 节点真正采用的规范配置
restore_shape        可选 Reshape
unpad_last_axis_to   可选 Slice
output_permutation   可选 Transpose
```

`simo/onnx/weight_quant.py:55-155` 的 `quantize_weight_array()` 先把 MatMul/Gemm/Conv 权重转换到
逻辑视图，再规范化为 runtime 接受的 rank-2 布局。MX 权重按 block size padding；per-channel
把量化轴移到第 0 维；per-tensor 降成单行；最后组合量化布局逆变换与算子存储转置，减少多余
Transpose。所有 quantized/scale initializer 最终都转换成连续的 `uint8` carrier，具体 dtype
由 QDQ 属性和内核解释。

`simo/onnx/onnx_quant.py:739-1008` 的 `insert_activation_qdq()` 对 activation 做同样的规范化：

- rank-N tensor 通过 Transpose/Flatten 变成连续 rank-2；
- MX 的 K 不对齐时支持静态或动态 padding，并在 DQ 后 Slice、Reshape、Transpose 恢复；
- per-channel 在 runtime 侧统一为 `axis=0`；
- FP8/INT8 per-tensor 复用单行 per-channel 内核；
- 非 MX 的 per-group kernel 本身支持 tail，不再无条件插入 Pad/Slice。

`simo/onnx/onnx_quant.py:1011-1073` 的 `insert_weight_dequantization()` 只消费
`QuantizedWeight` 描述并依次生成 DQ、Reshape、Slice、Transpose，职责比旧
`_create_weight_dq_nodes()` 更清晰。

### 58.5 QDQ 自定义算子升级到 opset v2

`simo/onnx/onnx_quant.py:35-36` 把 `com.simo` opset 从 1 升到 2。v2 节点只携带 runtime
真正需要的语义属性；`simo/onnx/onnx_quant.py:1076-1100` 的 `build_qdq_attributes()` 生成：

```text
dtype, granularity, scale_mode, group_size, block_size,
axis/axes，以及整数格式的 quant_min/quant_max
```

旧的 `observer_mode` 和布尔 `narrow_range` 不再进入 ONNX runtime ABI。整数范围直接编码为
`quant_min/quant_max`，例如 INT8 full range 为 `[-128, 127]`、narrow range 为
`[-127, 127]`；INT4 对应 `[-8, 7]` 或 `[-7, 7]`。

`simo/onnx/ort_plugin/simo_qdq_ops.cc` 的主要重构包括：

- `:30-52` 用 `IoDtype<T>` 为 FP32、FP16、BF16 建立类型映射。
- `:74-158` 根据 op、I/O dtype、quant dtype、granularity、scale mode、尺寸和数值范围精确选择
  AOT kernel，不再依赖宽松的运行时分支。
- `:263-380` 把各类 kernel 的 host 参数组装和 grid 计算集中到 `LaunchQdqKernel()`，并显式
  附加 Triton 3.6 的 global/profile scratch 两个 ABI 参数。
- `:385-443` 统一 rank-2 Q/DQ shape inference；`:447-569` 用模板化的
  `QuantizeCustomOp<T>` / `DequantizeCustomOp<T>` 实现计算。
- `:573-605` 注册一个 Quantize 名称的三种输入类型，以及
  `Dequantize`、`DequantizeFloat16`、`DequantizeBFloat16` 三种输出 wrapper。

同时删除了四个复制进仓库的 ONNX Runtime 私有 provider/resource 头文件，custom op 改用公开的
Lite Custom Op API，并通过 `KernelContext_GetGPUComputeStream` 获取 ORT CUDA stream。这样减少了
对 ORT 内部 C++ 实现的耦合；`simo/onnx/ort_plugin/simo_qdq_ops.h:11` 将使用的 ORT API 固定为
17。

`simo/onnx/ort_plugin/triton_loader.cc:21-40` 新增 `ScopedCudaContext`，kernel launch 前使用
`cuCtxPushCurrent`，结束后 `cuCtxPopCurrent`，避免旧代码 `cuCtxSetCurrent` 永久改变调用线程的
CUDA context。

### 58.6 Triton AOT 内核构建改成声明式单一来源

旧 `build_qdq_cubins.py` 在文件内部包装或复制了多套 Triton kernel，并用大量 `_compile_*`、
`_condition()`、`_entry()` 函数组装结果。新实现直接导入 `simo.ops.kernels` 中的生产 kernel：

- `simo/onnx/ort_plugin/build_qdq_cubins.py:82-97` 的 `QdqKernelBuildSpec` 同时描述 kernel、
  signature、constexpr、op/dtype/granularity、I/O dtype、scale mode 和 quant range。
- `:142-406` 的 `create_kernel_specs()` 用一个声明式表生成所有变体，包括 FP32/FP16/BF16、
  MX、NVFP4、FP8、INT8 full/narrow range、INT4 DQ、per-group/per-block/per-channel。
- 提交中的测试确认生成 99 个唯一编译 spec，对应 126 个唯一语义匹配组合。
- `:99-139` 的 `validate_kernel_abi()` 校验 `num_warps=8`、warp size 32、scratch size 为 0，
  并解析 PTX 验证参数个数等于显式 signature 加 Triton 3.6 的两个 scratch 参数。
- `:409-542` 把 cubin 字节、符号名、shared memory、grid/shape 元数据和精确 resolver 一并生成
  到 `embedded_qdq_kernels_sm90.cc`；不生成外部 cubin 文件或 manifest。

这一变化消除了“ONNX runtime builder 自己维护一份 Triton wrapper 逻辑”和“PyTorch 执行路径
维护另一份 kernel”之间的漂移风险。

### 58.7 `.so` 从运行时现场编译改成安装/构建时产物

这是运行方式上最重要的重构。

#### 旧行为

旧 `simo/onnx/runtime.py` 会：

1. 检查当前 GPU 是否为 sm90；
2. 对 Python、ORT、CUDA、源码和 Triton kernel 源码计算 cache key；
3. 在 `~/.cache/simo/onnx/` 上锁；
4. 找不到已打包 `.so` 时调用 `build_sm90_runtime()` 现场构建。

#### 新行为

- `setup.py:97-110` 的 `SimoBuildExtension.run()` 在 `pip install .`、`pip install -e .` 或 wheel
  build 阶段调用 `build_sm90_runtime()`，并把
  `libSimoOnnxCustomOps_sm90.so` 写入目标 package。
- `setup.py:117-122` 让 wheel 只携带最终 `.so`，源码/header 不进入已安装 wheel；新增
  `MANIFEST.in` 保证构建 sdist 时这些源文件仍存在。
- 新增 `simo/onnx/ort_plugin/build_wheel.sh`，统一执行
  `python -m pip wheel --no-build-isolation --no-cache-dir --no-deps .`。
- `simo/onnx/ort_plugin/build_runtime.py:48-99` 不再调用
  `torch.utils.cpp_extension.load()`；它在临时目录 AOT 编译 Triton cubin，然后直接调用系统
  C++17 编译器链接 CUDA Driver API，产出 `.so` 后原子移动到目标路径。
- custom-op `.so` 这一段使用 Triton compiler + host `c++`，不直接调用 `nvcc`；但同一安装过程
  中主 `simo._C` CUDA extension 仍由 PyTorch `CUDAExtension`/nvcc 构建，并在
  `setup.py:64` 固定 `sm_90` gencode。
- `simo/onnx/runtime.py:11-26` 的 `get_custom_ops_library_path()` 现在只返回 wheel 中的固定路径，
  或读取 `SIMO_ONNX_CUSTOM_OPS_LIBRARY` 覆盖值；若 `.so` 不存在则直接提示重新安装，不再创建
  runtime cache，也不再运行编译器。

结果是构建环境必须有 C++17、CUDA headers、Triton 和 CUDA driver link stub/library；而正确安装
的 wheel 在运行环境不再需要编译器、Triton 源码或可写 cache。

还有一个调用约定变化：`simo/onnx/runtime.py:35-49` 的 `create_session()` 只有在自己创建
`SessionOptions` 时才自动注册 SIMO custom ops；调用者传入自定义 `sess_options` 时必须先显式
调用 `register_custom_ops(options)`。此外它新增了直接接收 `onnx.ModelProto` 并序列化的能力。

### 58.8 整数范围和 rounding 在整个代码库中统一

该提交虽以 `refactor_onnx` 命名，但同时重构了通用量化内核，因为 ONNX AOT runtime 必须与
PyTorch/SGLang/vLLM 的参考结果使用完全一致的数值语义：

- `simo/quantization/dtypes.py:115-118` 新增 `INT_DType.quant_range()`。
- `simo/quantization/config.py:354-406` 的 `QuantizeSpecInt` 冻结 `dtype/narrow_range`，构造时
  解析并缓存 `_quant_range`；FP/MX spec 也提供 `quant_range` property。
- observer、quantizer、`simo/quantization/kernels.py:21-96`、
  `simo/ops/flex_api.py:44-55` 和所有 downcast operator 不再各自接收/推断 `narrow_range`，而是
  显式传递并校验 `quant_min/quant_max`。
- scale divisor 统一为 `(quant_max - quant_min) / 2`：INT8 full/narrow 分别为 127.5/127，
  INT4 full/narrow 分别为 7.5/7。
- Torch 参考路径使用 `torch.round`，Triton 整数路径使用 `rint`，统一为 half-even rounding。
- KV cache、attention、GEMM、fused MoE 等调用方改为传完整 quant spec；fused MoE 也显式接收
  min/max，避免自身重新推断范围。

因此，这部分不是只为 ONNX 改接口，而是修复不同执行后端对 full range、narrow range、scale
和舍入方式可能不一致的问题。

### 58.9 依赖、测试和删除内容

`pyproject.toml:35-40` 的 ONNX 可选依赖改为：

```text
onnx>=1.19,<1.23
onnx-graphsurgeon>=0.6.1,<0.7
onnxslim>=0.1.84,<0.2
onnxruntime-gpu>=1.24,<1.28
```

即新增 GraphSurgeon，删除 onnxscript，并给 ONNX、ONNXSlim、ORT 添加明确兼容范围。

测试重构不是简单删测试：

- 删除 `test_activation_qdq_plan.py` 和 `test_qdq_runtime_contract.py` 两个旧白盒/源码字符串测试。
- 新增 `test_weight_quant.py`，覆盖 per-channel 布局、BF16 离线转换和 INT8/INT4 full/narrow range。
- 新增 `test_fused_moe_narrow_range.py`，验证 fused MoE 与通用 downcast 数值一致。
- `test_qdq_utils.py` 从 1582 行扩大到 2617 行，增加 GraphSurgeon 保真限制、子图、dtype、opset、
  配置优先级、no-op 字节保持、动态 shape/padding、carrier dtype 等测试。
- `test_dynamic_qdq_runtime_debug.py` 增加 FP16/BF16、整数范围、half-even、非法 ABI 组合和多种
  runtime kernel 的真实 CUDA 执行验证。
- `test_qdq_cubin_build.py:88-159` 检查 spec 唯一性、支持矩阵和 Triton/PTX ABI，而不是只比较
  生成源码是否包含某些字符串。

### 58.10 最终归纳和迁移影响

这次提交可以概括为五个核心重构：

1. **图层**：protobuf 手工改图改为 GraphSurgeon 图编辑，并增强子图、动态 shape、dtype 和模型
   保真检查。
2. **配置/ABI 层**：启用 QDQ v2 严格语义配置，用明确的 dtype、granularity、layout 和数值范围
   选择 kernel。
3. **kernel 层**：用声明式 `QdqKernelBuildSpec` 直接 AOT 编译生产 Triton kernel，校验 Triton
   ABI 并把 cubin/resolver 嵌入 `.so`。
4. **交付层**：从运行时现场构建/cache 改成安装或 wheel 构建时生成 sm90 `.so`，运行时只加载。
5. **数值层**：把 full/narrow integer range、scale divisor 和 half-even rounding 统一到 ONNX、
   PyTorch、Triton、attention、GEMM 和 MoE 路径。

它同时包含明确的 breaking changes：旧 `insert_qdq_nodes` 内部 API、旧 JSON wrapper、
`targets_op_types`、运行时自动编译和传入 `sess_options` 后的自动注册都不再兼容。迁移时至少需要：

```text
旧配置 targets_op_types -> targets
删除 quantization_config/quant_method/quant_algo 外层
使用 simo.onnx.quantize() 或 apply_qdq_quantization()
重新 pip install -e . / 构建 wheel，确保 .so 在 package 内
自定义 SessionOptions 时先调用 register_custom_ops(options)
确认目标 GPU 为 sm90，且 ONNX/ORT 版本落在新的依赖范围
```

## 59. MatMul/Gemm 权重量化中的布局变换

核心结论是：量化前的 transpose 是把权重整理成量化内核需要的规范布局，Dequantize 后的
transpose 是把结果还原成原 ONNX 算子期望的存储布局。它们处理的是“同一份权重如何排布”，
不会改变 MatMul/Gemm 的乘法定义；真正可能引入数值误差的是量化和反量化本身，而不是这些可逆的
布局变换。

### 59.1 原始存储布局与量化逻辑布局不是一回事

对线性层，量化代码统一把二维逻辑权重看成 `[N, K]`：一行通常对应一个输出通道，最后一维是
乘法的归约维 K。ONNX 算子实际保存的 B 不一定是这个方向：

| 算子情况 | ONNX 中 B 的原始形状 | 量化使用的逻辑视图 | 从逻辑视图恢复原始 B 的基础排列 |
| --- | --- | --- | --- |
| `MatMul` | `[K, N]` | `B.T`，即 `[N, K]` | `(1, 0)` |
| `Gemm(transB=0)` | `[K, N]` | `B.T`，即 `[N, K]` | `(1, 0)` |
| `Gemm(transB=1)` | `[N, K]`，算子内部使用 `B.T` | `B`，即 `[N, K]` | identity |

这正是 `simo/onnx/weight_quant.py:36-50` 中 `logical_weight_view()` 的行为：`MatMul` 总是返回
`weight.T`；`Gemm` 在 `transB=0` 时返回 `weight.T`，在 `transB=1` 时直接返回 `weight`。
`simo/onnx/weight_quant.py:61-68` 随后用相同条件建立 `storage_permutation`，记录最终怎样回到算子原来
接收的 B 布局。

以 `MatMul` 为例，原图计算 `A[M,K] @ B[K,N]`。离线量化先处理 `B.T[N,K]`，是因为量化器更适合
按“输出通道为行、K 为最后一维”的方式生成量化值和 scale；运行时 Dequantize 得到近似的
`B.T[N,K]` 后，再转置为 `[K,N]` 交给原 MatMul。MatMul 仍执行原来的 `A @ B`。

需要注意，权重是 initializer，所以这里的量化前 transpose/reshape/pad 发生在模型转换期的 NumPy
数组上，不是在最终 ONNX 图里插入一个运行时 Quantize 前的 Transpose。最终图中的权重路径只有量化
常量、Dequantize 和必要的恢复节点。

### 59.2 为什么量化器需要规范的二维布局

`simo/onnx/weight_quant.py:55-125` 的 `quantize_weight_array()` 不只是简单执行一次 `.T`，而是根据
量化粒度把逻辑权重规范成内核接受的连续 rank-2 输入：

- per-channel 在 `:104-115` 把配置的 channel axis 移到第 0 维，再把其余维度展平。这样每一行对应
  一个 channel，scale 的行数和 channel 数严格一致，传给内核的规范配置也统一为 `axis=0`。
- per-group 要求二维逻辑权重沿最后一维分组；`:119-125` 会拒绝不是最后一维的 group axis。这样每行
  都能沿连续的 K 维按 `group_size` 划分，scale 形状和 kernel 索引规则一致。
- MX/flex 的 block 量化在 `:75-94` 把配置轴移到最后一维，按 `block_size` 补齐尾部，再展平为
  `[rows, padded_K]`。这保证每个 block 完整且连续，并记录原 K，供 DQ 后去 padding。
- FP8/INT8 per-tensor 在 `:95-103` 展平成单行；per-block 等其余二维 flex 路径则保留规范 rank-2
  布局。它们共同满足 custom kernel 对维度、连续性、scale 形状和 block/group 索引的约定。

因此 transpose 的目的不是适配 MatMul/Gemm 本身，而是给不同量化方式提供统一、可预测的“行”和
“最后一维”语义。reshape 负责得到内核 ABI 使用的二维形状，pad 负责满足 MX block size 对齐；这些
都是量化计算的临时布局要求。

### 59.3 Dequantize 后如何恢复原布局

`quantize_weight_array()` 把恢复信息写入 `QuantizedWeight`：`restore_shape` 保存展平前形状，
`unpad_last_axis_to` 保存 padding 前的 K，`output_permutation` 保存最后的轴排列。对发生过轴移动的
情况，最终排列等价于：

```text
规范量化布局 --逆量化布局排列--> 逻辑权重 [N,K]
             --存储排列----------> 原始 ONNX B 布局
```

代码在 `simo/onnx/weight_quant.py:93-115` 用 `compose_permutations()` 把这两步合成一个
`output_permutation`；如果合成结果是 identity，`:127-129` 就把它消掉。因此：

- `MatMul` 或 `Gemm(transB=0)` 通常需要恢复 B 的存储方向，但这个 transpose 可能和量化轴移动的
  逆 transpose 抵消；
- `Gemm(transB=1)` 不需要额外的“存储方向”转置，但若量化时移动过轴，仍可能需要恢复布局的
  transpose。不能简单理解为 `transB=1` 时权重路径永远没有 Transpose。

`simo/onnx/onnx_quant.py:1011-1073` 的 `insert_weight_dequantization()` 严格按
`Dequantize -> Reshape -> Slice(unpad) -> Transpose(output_permutation)` 的顺序生成图，并把最终结果
接回 `node.inputs[1]`。所以 MatMul/Gemm 看到的形状、轴顺序以及保留的 `transB` 属性都与原图一致。

例如未对齐的 MatMul MX 权重 `B[18,4]` 会在转换期经历：

```text
B[18,4] -> B.T[4,18] -> pad [4,32] -> 离线 Quantize
运行时 Dequantize [4,32] -> Slice [4,18] -> Transpose [18,4] -> MatMul
```

`simo/onnx/tests/test_qdq_utils.py:576-605` 验证了 DQ 的 `logical_shape == [4,32]`、Slice 恢复到
18，以及 Slice 后才执行权重 Transpose；`:1436-1467` 也直接断言量化元数据中的
`unpad_last_axis_to == 18` 和 `output_permutation == (1,0)`。

### 59.4 `transA` 与 `transB` 分别影响什么

Gemm 的语义是 `Y = alpha * op(A) @ op(B) + beta * C`。因此两个属性不能混在一起解释：

| 属性 | 权重 B 路径 | 激活 A 路径 | 对恢复 Transpose 的影响 |
| --- | --- | --- | --- |
| `transB` | 直接决定 `logical_weight_view()` 是 `B.T` 还是 `B`，并决定基础 `storage_permutation` | 当允许从权重形状推断激活 K 时，决定原始 B 的哪一维是 K | 直接参与权重 `output_permutation`；实际节点还取决于它与量化布局逆排列合成后是否为 identity |
| `transA` | 不改变 B，不参与 `logical_weight_view()`、权重离线量化或权重 `output_permutation` | 把配置针对逻辑 `op(A)` 的轴映射回原始 A 的轴，影响 K 的取得、QDQ 前的规范化排列以及 DQ 后的逆排列 | 只直接影响激活路径的前后 Transpose；对权重 Dequantize 分支没有直接作用 |

具体来说，`simo/onnx/onnx_quant.py:774-823` 的 `insert_activation_qdq()` 把配置轴解释为逻辑
`op(A)` 的轴。对 Gemm，`transA=1` 时用 `raw_axis = 1 - logical_axis` 映射回原始二维 A；per-channel
把该轴移到第 0 维，per-group/MX 则把它移到最后一维。`:894-1007` 在 Quantize/Dequantize 前插入
规范化 Transpose，并在 DQ、去 padding 和恢复 shape 后插入其逆排列，最后才接回 `node.inputs[0]`。

K 维推断也体现了两者的分工。`simo/onnx/onnx_quant.py:817-844` 优先按照 `transA` 映射后的原始 A
轴读取静态 K；`:881-892` 只在 `Gemm(transA=0)` 时允许从权重补充推断 K，并再用 `transB` 选择
`source_shape[-2]` 或 `source_shape[-1]`。也就是说，`transA` 决定怎样解释 A，`transB` 决定怎样
解释 B；二者可能共同参与 K 的确定，但只有 `transB` 改变权重的逻辑视图。

现有测试给出了直接的行为依据：

- `simo/onnx/tests/test_weight_quant.py:26-82` 同时覆盖 MatMul、默认 Gemm 和
  `Gemm(transB=1)` 的 canonicalization、rank-2 形状与最终排列；其中 `transB=1`、axis 0 的案例
  明确断言 `output_permutation is None`。
- `simo/onnx/tests/test_qdq_utils.py:637-647` 验证量化后 Gemm 仍保留 `transB=1` 和原 bias；恢复后的
  权重由 Dequantize 经 Reshape 直接接入 B，不需要额外的存储方向 Transpose。
- `simo/onnx/tests/test_qdq_utils.py:1285-1311` 验证 `transA=1` 时逻辑 per-channel axis 映射到原始
  A 后，Q 前和 DQ 后各出现一个 `(1,0)` Transpose。
- `simo/onnx/tests/test_qdq_utils.py:650-681` 验证 `transA=1` 的 MX 激活路径使用逻辑 K=18 做
  padding/unpadding，并在两侧使用互逆的 `(1,0)` 排列。

最终可以简化为一句话：`transB` 管 B 的权重解释与权重恢复，`transA` 管 A 的激活解释与激活恢复；
权重 DQ 后是否真的插入 Transpose，以合成后的 `output_permutation` 是否为 identity 为准。

## 60. `validate_graphsurgeon_conversion()` 的功能

### 60.1 核心结论

`simo/onnx/onnx_quant.py:348-434` 的 `validate_graphsurgeon_conversion(model)` 是一个
**ONNX GraphSurgeon 无损转换的前置兼容性检查器**。它在模型进入
`gs.import_onnx(...) -> 图编辑 -> gs.export_onnx(...)` 之前，检查图中是否含有当前
`onnx-graphsurgeon` 不能可靠保留的 protobuf 结构。

它的策略是“不能保证保留就提前失败”：

- 所有检查通过时返回 `None`，并且不修改传入的 `ModelProto`；
- 遇到第一个不支持的结构时抛出 `NotImplementedError`；
- 这样可避免量化本身成功，但 GraphSurgeon 导入/导出后静默丢失 metadata、类型信息、属性或
  initializer 表示等模型内容。

因此，这个函数名中的 `conversion` 指的是 **ONNX protobuf 与 GraphSurgeon 内部图表示之间的
往返转换**，不是检查量化数值是否正确。

### 60.2 它在量化流程中的位置

`apply_qdq_quantization()` 中的关键顺序如下：

```text
输入 ModelProto/模型路径
  -> 深拷贝模型
  -> validate_graphsurgeon_conversion(model)
  -> 可选 ONNXSlim simplify
  -> validate_graphsurgeon_conversion(simplified_model)
  -> gs.import_onnx
  -> 插入 QDQ、cleanup、toposort
  -> gs.export_onnx
  -> 将编辑后的 graph 和 opset_import 写回模型外壳
```

对应代码是 `simo/onnx/onnx_quant.py:251-265` 和 `:329-334`。原模型在 `:251-252` 先被深拷贝，
普通路径在 `:253` 检查一次；若 `simplify=True`，简化结果还会在 `:258` 再检查一次。这保证原图和
简化器产生的新图都满足 GraphSurgeon 的转换约束。

GraphSurgeon 实际处理的是 `:260-264` 用 `graph`、`opset_import` 和 `ir_version` 临时构造的
working model。模型外层的 `domain`、`model_version`、`doc_string`、`metadata_props`、
`training_info`、`functions` 和 `configuration` 不经过这次 GraphSurgeon round-trip；导出后只把
新 `graph` 和 `opset_import` 写回原有模型外壳。因此本函数主要检查 `GraphProto` 及其内部对象，
而不是拒绝这些可由外壳直接保留的 `ModelProto` 字段。

### 60.3 具体检查了什么

| 检查对象 | 被拒绝的内容 | 代码位置 |
| --- | --- | --- |
| 图 `GraphProto` | `metadata_props`、`quantization_annotation` | `:352-355` |
| 图 initializer | `sparse_initializer` | `:356-359` |
| 图输入与 initializer | 同一个名称同时是 graph input 和 initializer，即 initializer-backed graph input | `:361-365` |
| `value_info` | 与 graph input、output 或 initializer 重名；或者没有被图输入、输出、initializer、节点输入/输出引用 | `:367-378` |
| 输入、输出和 `value_info` | `doc_string`、`metadata_props`，以及 tensor 以外的 value type，例如 sequence、map、optional 或 sparse tensor type | `:384-389` |
| 类型和维度 | `TypeProto.denotation` 或任意 shape dimension 的 `denotation` | `:390-398` |
| 节点 `NodeProto` | `doc_string`、`metadata_props`、`overload`、`device_configurations` | `:400-404` |
| 节点属性 | `doc_string`、`ref_attr_name`、不支持的属性类型，以及空的 `FLOATS`/`INTS`/`STRINGS` 重复属性 | `:405-420` |
| 普通张量 | `doc_string`、`metadata_props`、`segment`，或者仍使用 `data_location=EXTERNAL` 的外部数据 | `:426-433` |

属性类型采用白名单。`simo/onnx/onnx_quant.py:53-62` 只允许：

```text
FLOAT, INT, STRING, TENSOR, GRAPH, FLOATS, INTS, STRINGS
```

所以 `TENSORS`、`GRAPHS`、`SPARSE_TENSOR`、`TYPE_PROTO` 及其重复形式等属性会被拒绝。即使类型在
白名单内，空的 `FLOATS`、`INTS` 或 `STRINGS` 也会被拒绝，因为 GraphSurgeon 往返时不能可靠区分
并保留这种带显式类型的空序列属性。

### 60.4 如何遍历子图和张量

函数用 `graphs = [model.graph]` 作为栈，从主图开始检查。遇到受支持的单个 `GRAPH` 属性时，
把 `attr.g` 加入栈，因此 `If`、`Loop` 等节点属性中的嵌套子图也会递归执行相同检查。`GRAPHS`
属性本身不在白名单中，会直接失败，而不会递归处理。

需要检查的张量先包含当前图的普通 initializer；遇到 `TENSOR` 属性时，再把属性中的 `attr.t`
加入列表。随后统一检查这些张量的 metadata、segment 和 external-data 状态。也就是说，检查范围
不只包括主图权重，还包括嵌套子图的 initializer 和节点携带的 tensor attribute。

`data_location=EXTERNAL` 的张量会收到
`external tensor ... must be loaded before graph editing`。这并不是禁止模型最初使用 ONNX external
data 格式，而是要求在交给 GraphSurgeon 编辑前，外部数据已经加载并物化到 `TensorProto` 中，
否则导入/导出无法保证权重数据仍然完整。

### 60.5 它不负责什么

这个函数不是 `onnx.checker.check_model()` 的替代品。它不会：

- 校验节点是否满足 ONNX schema、opset 或拓扑规则；
- 做 shape inference 或验证张量维度能否进行 MatMul/Gemm/Conv；
- 检查量化配置、量化 dtype、scale 或 QDQ 数值正确性；
- 证明 GraphSurgeon 对所有普通字段都绝对无损。

它只编码当前实现已经知道的、GraphSurgeon 无法保留或表示不稳定的情况。因此抛出
`NotImplementedError` 表示“该 ONNX 表示目前不支持走这条图编辑链路”，并不表示输入一定是非法
ONNX 模型。

测试也体现了这条边界：

- `simo/onnx/tests/test_qdq_utils.py:1178-1193` 验证 sparse initializer 会被拒绝；
- `:1988-1993` 验证带节点 `doc_string` 的图会在转换前失败；
- `:1996-2008` 验证 `GRAPHS` 属性类型会被拒绝；
- `:2011-2020` 验证显式空的重复属性会被拒绝；
- `:2023-2061` 验证模型外壳字段会被保留，且源 `ModelProto` 不会被原地修改。

一句话概括：`validate_graphsurgeon_conversion()` 是 QDQ 图改写前的防丢数据护栏，负责确认
GraphSurgeon 即将接管的图结构处于当前代码支持的可往返表示范围内；不满足时宁可明确报错，也不
生成一个字段已被静默丢弃的量化模型。

## 61. 静态 MXINT8 MatMul QDQ 节点的插入位置

### 61.1 本节对应的具体案例

这里严格以 `main()` 中的 `("static", "mxint8", [2, 3, 18])` 为准。
`like-useful/test_dynamic_qdq_runtime_debug-debug.py:1346-1369` 将该元组依次解包为：

```text
shape_id   = "static"
dtype      = "mxint8"
input_shape = [2, 3, 18]
```

然后它调用 `like-useful/test_dynamic_qdq_runtime_debug-debug.py:1166-1182` 的
`test_unaligned_matmul_activation_qdq_matches_simo_torch_with_padding()`。测试把激活配置转换为
`{"dtype": "mxint8"}`，运行模型，并把 QDQ 后的激活与 SIMO Torch 参考结果做数值比较。

建图函数位于 `like-useful/test_dynamic_qdq_runtime_debug-debug.py:441-481`。它创建：

```text
X: float32 [2,3,18] @ W: float32 [18,4] -> Y: float32 [2,3,4]
```

同一处配置了两条不同的量化路径：激活 `X` 使用 MXINT8；权重 `W` 使用
`fp8_e4m3, axis=[0,1], group_size=128`，即 FP8 E4M3 per-block。下面分别说明两条路径。

### 61.2 激活路径的七类节点

对于 rank-3 的 `X[2,3,18]`，最后一维 K 已经是量化轴。实现先把前两维合并成 token 行，把
K=18 补到 MXINT8 block size 32，完成 QDQ 后再切回 K=18 并恢复原形状。

| 算子 | 创建位置 | 导出的节点名 | 输入 -> 输出 | 用途 |
| --- | --- | --- | --- | --- |
| `Shape` | `simo/onnx/onnx_quant.py:906-907` | `matmul_input_simo_shape` | `X [2,3,18] -> int64 [3]`，值为 `[2,3,18]` | 保存原始形状，供末尾 `Reshape` 使用 |
| `Flatten` | `simo/onnx/onnx_quant.py:908-915` | `matmul_input_simo_flatten` | `X [2,3,18] -> matmul_input_simo_rank2 [6,18]` | `axis=-1`，把 `2*3` 合并为 6 行 |
| `Pad` | `simo/onnx/onnx_quant.py:932-961` | `matmul_input_simo_pad` | `[6,18] -> matmul_input_simo_pad [6,32]` | 使用 `pads=[0,0,0,14]`，仅在 K 维右侧补 14 个零 |
| `Quantize` | `simo/onnx/onnx_quant.py:963-972` | `matmul_input_simo_quant` | `[6,32] -> matmul_SimoQuantInput uint8 [6,32]` 和 `matmul_SimoScale uint8 [6,1]` | 运行时按每行一个 32 元素 block 生成 MXINT8 数据与 scale |
| `Dequantize` | `simo/onnx/onnx_quant.py:973-984` | `matmul_input_simo_dequant` | 两个 UINT8 carrier -> `matmul_SimoDequantOutput_padded float32 [6,32]` | 按相同 MXINT8 属性恢复浮点值 |
| `Slice` | `simo/onnx/onnx_quant.py:986-992` | `matmul_input_simo_unpad` | `[6,32] -> matmul_SimoDequantOutput [6,18]` | 使用 `starts=[0], ends=[18], axes=[1]` 去掉补齐区 |
| `Reshape` | `simo/onnx/onnx_quant.py:993-999` | `matmul_input_simo_restore` | `[6,18] + Shape结果 -> [2,3,18]` | 恢复 MatMul 原始 A 输入布局 |

`Quantize` 和 `Dequantize` 的实际属性都是
`dtype="mxint8", granularity="per_group", axis=1, group_size=32, block_size=32,
scale_mode="e8m0_floor"`。MX 配置的默认 block size 及其 `group_size` 同步逻辑位于
`simo/quantization/config.py:450-490`；这里的 `per_group` 描述的是 custom QDQ 算子的运行时属性，
每个 `[1,32]` 行块共享一个 scale。

这个静态案例的激活分支总共只有一个 `Shape`。`simo/onnx/onnx_quant.py:834-837` 可以直接从
静态输入形状读取 `known_k=18`，所以 `needs_runtime_k` 为假，不会进入
`simo/onnx/onnx_quant.py:916-930` 的额外 `Shape + Gather` 动态 K 分支。唯一的 `Shape` 只是保存
`[2,3,18]`，并不用于求 K。

源码在 `simo/onnx/onnx_quant.py:906-915` 先创建 `Shape`、再创建 `Flatten`。两者都只依赖 `X`，
彼此没有先后数据依赖；`simo/onnx/onnx_quant.py:329-330` 的 cleanup、拓扑排序和导出可以改变它们
在 ONNX 节点列表中的排列。本次实际导出中是 `Flatten` 在前、`Shape` 在后，这不表示多出或漏掉了
任何节点。

### 61.3 权重如何成为 UINT8 carrier initializer

权重量化发生在模型转换期，而不是运行时图中的 `Quantize` 节点中：

1. `simo/onnx/weight_quant.py:61-68` 对 MatMul 的原始 `W[18,4]` 取逻辑视图 `W.T[4,18]`，同时把
   最终恢复排列记录为 `(1,0)`。
2. 该 FP8 配置的两个轴和 `group_size=128` 被判定为 per-block；判定代码位于
   `simo/quantization/config.py:229-237`。二维逻辑权重直接以连续 `[4,18]` 送入量化，不做图级
   padding，相关分支位于 `simo/onnx/weight_quant.py:116-125`。
3. `simo/onnx/weight_quant.py:136-138` 选择并调用 CUDA downcast kernel，得到 FP8 量化权重和
   FP32 scale。对 `[4,18]` 和 128 x 128 block，逻辑 scale 形状是 `[1,1]`。
4. `simo/onnx/weight_quant.py:139-144` 把量化权重和 scale 都按原始字节解释为 `torch.uint8`。
   因此一个 FP32 scale 的 4 个字节表现为 `[1,4]`，并不是四个独立 scale。
5. `simo/onnx/weight_quant.py:146-154` 将两个 carrier 及恢复元数据保存到 `QuantizedWeight`。

插图时，`simo/onnx/onnx_quant.py:1020-1025` 分别创建
`matmul_W_simo_q` 和 `matmul_W_simo_scale`；底层的 `gs.Constant` 创建点是
`simo/onnx/onnx_quant.py:195-196`。GraphSurgeon 导出后，它们是以下两个 ONNX initializer：

| initializer | ONNX 数据类型 | 导出形状 | 实际含义 |
| --- | --- | --- | --- |
| `matmul_W_simo_q` | `UINT8` | `[4,18]` | FP8 E4M3 权重的逐字节 carrier |
| `matmul_W_simo_scale` | `UINT8` | `[1,4]` | 逻辑 FP32 scale `[1,1]` 的四个字节 |

所以图中所谓的 weight load 和 scale uint8 load 实际是读取常量 initializer；它们不是 ONNX
`Load` 算子，导出的图中也没有名为 `Load` 的节点。

### 61.4 权重侧实际插入的两个算子

| 算子 | 创建位置 | 导出的节点名 | 输入 -> 输出 | 用途 |
| --- | --- | --- | --- | --- |
| `Dequantize` | `simo/onnx/onnx_quant.py:1031-1039` | `matmul_weight_simo_dequant` | 两个 UINT8 initializer -> `matmul_weight_simo_weight_dq float32 [4,18]` | 按 `fp8_e4m3/per_block` 解释 carrier；属性包含 `original_shape=[18,4]`、`logical_shape=[4,18]`、`axes=[0,1]`、`group_size=128` |
| `Transpose` | `simo/onnx/onnx_quant.py:1065-1072` | `matmul_weight_simo_transpose` | `[4,18] -> matmul_weight_simo_transpose [18,4]` | 使用 `perm=[1,0]` 恢复 MatMul 所需的原始 B 布局 |

`simo/onnx/onnx_quant.py:1073` 最后把 `matmul_weight_simo_transpose` 接到原 `matmul` 的第二个输入。

这条权重分支没有 `Reshape`、`Pad` 或 `Slice`。它本来就是二维 `[4,18]`，也没有设置
`restore_shape` 或 `unpad_last_axis_to`。FP8 per-block CUDA 路径按向上取整得到边界 block 的 scale
形状，见 `simo/ops/kernels/downcast/_downcast_to_flexpoint.py:145-151`；kernel 在
`simo/ops/kernels/downcast/_downcast_to_flexpoint.py:206-225` 对超出 4 行或 18 列的元素使用边界
mask，因此不需要为了 128 x 128 block 在 ONNX 权重分支中物理补齐。这个案例只需最后的 `(1,0)`
恢复排列。

### 61.5 完整数据流

```text
激活 A：
X float32 [2,3,18]
  +-> Shape -> int64 [3]，值 [2,3,18] ---------------------------+
  |                                                               |
  +-> Flatten(axis=-1) [6,18]                                    |
      -> Pad(+14 on K) [6,32]                                    |
      -> com.simo::Quantize                                      |
           q uint8 [6,32], scale uint8 [6,1]                     |
      -> com.simo::Dequantize float32 [6,32]                     |
      -> Slice(K<18) float32 [6,18]                              |
      -> Reshape <------------------------------------------------+
           float32 [2,3,18]
      -> matmul.input[0]

权重 B：
W float32 [18,4]
  -> 转换期 W.T [4,18]
  -> FP8 E4M3 per-block CUDA downcast
  -> matmul_W_simo_q uint8 initializer [4,18] --------+
     matmul_W_simo_scale uint8 initializer [1,4] -----+-> com.simo::Dequantize
                                                          float32 [4,18]
                                                       -> Transpose(perm=[1,0])
                                                          float32 [18,4]
                                                       -> matmul.input[1]

MatMul: [2,3,18] @ [18,4] -> Y float32 [2,3,4]
```

第 43 节描述的是重构前的实现，其中的 `Reshape` 节点名、旧辅助函数和部分代码行号不再对应当前
源码。本节的节点名称、连接、形状及所有“仓库相对路径 + 行号”均以当前实现和上述静态导出模型为准。

## 62. SGLang W4A16 在 `temperature=0` 时未输出 `Paris` 的原因

### 62.1 结论

这次失败的直接原因不是随机采样，也没有证据表明 SGLang 的 INT4 解包或 greedy sampler 整体算错。
直接触发点是：在这条包含两种互相竞争意图的长提示上，W4A16 量化误差和 SGLang 的 DeepSeek/MLA
执行图共同改变了首 token 的 logits 排名，使填空 token `" _____"` 略高于 `" Paris"`。当
`temperature=0` 时 SGLang 执行确定性 argmax，于是稳定选择填空 token；自回归生成再把首 token
的差别扩展为整段不同文本。

受控 top-logprobs 直接证明了这个判断：

| 运行方式 | 长提示首选 | `log P(" Paris")` | `log P(" _____")` | 两者差值 |
| --- | --- | ---: | ---: | ---: |
| SGLang BF16，默认 attention backend | `" Paris"` | -1.119431 | -1.619431 | `Paris` 高 0.500 |
| SGLang W4A16，默认 attention backend | `" _____"` | -1.335353 | -1.210353 | 填空高 0.125 |
| SGLang W4A16，显式 `attention_backend="triton"` | `" _____"` | -1.493297 | -1.118297 | 填空高 0.375 |
| vLLM W4A16，显式 `temperature=0` | `" Paris"` | -1.099223 | -1.974223 | `Paris` 高 0.875 |

相反，对没有面试指令前缀的短提示 `The capital of France is`，SGLang BF16 和 W4A16 都选择
`Paris`，其首 token logprob 分别为 `-0.352196` 和 `-0.358543`，非常接近。这排除了 tokenizer、
`temperature=0` 实现、INT4 权重基本装载流程或整个模型已经损坏等解释。

因此应把这次现象归类为“脆弱的单样本精确文本断言被量化后的近邻 token 翻转触发”，而不是仅凭
这一条生成就认定 SGLang W4A16 存在系统性精度错误。

### 62.2 原日志实际说明了什么

SGLang 单配置日志在
`temp/smoke.single.txt.2026_07_23___17_27_40:193-205` 中输出：

```text
Generated text:  ________.

1. Can you describe your teaching philosophy and how it
```

这不是乱码、空输出或重复异常。它把 `The capital of France is` 处理成了问卷中的填空题，然后继续
响应长前缀要求的面试问题。原提示在
`tests/sglang_simo/e2e_test/test_basic_generate_single.py:165-180` 先要求“起草 10-15 个教师面试问题”，
又要求“fulfill the following paragraph”，最后才拼接事实补全短句。这两个目标存在明显歧义，
`" Paris"` 和 `" _____"` 都是上下文上合理的高分开头。

全量日志也不支持“W4A16 运行链路崩坏”：`temp/smoke.txt:9-23` 的 13 个量化配置中有 11 个通过，
只有 W4A16 INT4 和 W4A4 NVFP 在同一个 `Paris` 文本断言上失败；最终汇总见
`temp/smoke.txt:548-551`。失败配置恰好更容易改变模型 logits，但都成功完成加载、prefill、decode
和 detokenize。

此外，这个配置没有启用 KV cache 量化。配置中的 `kv_cache_quant_algo` 为 `null`，见
`simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w4a16_int4_per_group.json:22-25`；
运行时也明确回退到标准 BF16 KV cache，见
`temp/smoke.single.txt.2026_07_23___17_27_40:179-182`。所以本次首 token 分叉不能归因于 KV cache
量化。

### 62.3 `temperature=0` 为什么没有保证两个框架输出相同

SGLang 在
`sglang_kernel_src/python/sglang/srt/sampling/sampling_params.py:168-172` 把接近零的 temperature
正规化为 `top_k=1`，也就是 greedy argmax。该行为本身符合预期：

```text
next_token = argmax(logits)
```

它保证的是“给定同一组 logits 时不随机抽样”，不保证 BF16、W4A16、不同 attention backend 或不同
框架产生完全相同的 logits。量化前若 `Paris` 只领先另一个候选很小的 margin，INT4 per-group 舍入误差
经过 27 层传播后就可能交换二者顺序。`temperature=0` 不会消除这个误差，反而会把交换后的第一名
稳定选中。由于后续每一步又以已经不同的 token 为输入，文本会快速完全分叉。

这里 `top_p=0.95` 也不是决定因素。`top_k=1` 后候选集合只有 argmax token，top-p 不会把第二名
`Paris` 再选回来。

W4A16 的实际数值路径也与这一解释一致：

- 配置在
  `simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w4a16_int4_per_group.json:10-20`
  指定 INT4、`axis=-1`、`group_size=32`，没有激活量化配置，因此激活保持 BF16。
- `simo/extensions/sglang_simo/quantization/quantization.py:931-988` 在加载浮点 checkpoint 时对每个命中的
  权重执行 downcast，并保存 packed weight 与 scale。
- `simo/extensions/sglang_simo/quantization/quantization.py:1042-1096` 为 packed weight 和
  `GroupQuantScaleParameter` 分配参数。
- `simo/extensions/sglang_simo/quantization/quantization.py:1120-1124` 在 forward 中恢复 BF16 权重，随后
  `simo/extensions/sglang_simo/quantization/quantization.py:1194-1197` 执行矩阵乘。

这是一条真实的 weight-only INT4 QDQ 路径，输出本来就不要求逐 bit 等于 BF16。

### 62.4 原来的 vLLM 结果不是等条件对照

首先，用户给出的 vLLM 日志实际没有使用 `temperature=0`。SGLang 测试在
`tests/sglang_simo/e2e_test/test_basic_generate_single.py:163` 显式设置 0；vLLM 测试却在
`tests/vllm_simo/e2e_test/test_basic_generate_single.py:189` 调用 `llm.get_default_sampling_params()`。
日志 `temp/vllm.test_basic_generate_single.py.2026_07_23___17_50_21:363-368` 明确记录模型
`generation_config.json` 把默认值覆盖成 `temperature=0.3, top_p=0.95`，随后输出 `Paris`。

本次分析额外用 `vllm.SamplingParams(temperature=0.0, top_p=0.95, max_tokens=16)` 重跑后，vLLM
仍然选择 `Paris`，所以采样参数不一致不是 vLLM 成功的唯一原因，但原日志不能直接作为
`temperature=0` 的 A/B 证据。

其次，两份名为 W4A16 的 JSON 只有数值格式相同，量化目标并不相同：

| 项目 | SGLang | vLLM |
| --- | --- | --- |
| 配置位置 | `simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w4a16_int4_per_group.json:1-27` | `simo/extensions/vllm_simo/example/simo_quantization_config/online_quantization/quant_config_w4a16_int4_per_group.json:1-24` |
| targets | 所有 `Linear` | `*attn*`、`*mlp*`、`*ffn*` |
| excludes | `lm_head`、`re:.*kv_b_proj` | 空 |
| `ReplicatedLinear` | SGLang 会量化 | vLLM 强制保持非量化 |

SGLang 的选择逻辑见 `simo/extensions/sglang_simo/quantization/quantization.py:805-849`；原日志
`temp/smoke.single.txt.2026_07_23___17_27_40:40-172` 显示每层量化了 `q_proj`、
`kv_a_proj_with_mqa` 和 `o_proj`，没有量化 `kv_b_proj`。vLLM 在
`simo/extensions/vllm_simo/quantization/quantization_config.py:253-271` 明确跳过所有
`ReplicatedLinear`，因此不量化 `kv_a_proj_with_mqa`；其原日志则显示它量化了 `kv_b_proj`。

这不是简单的 JSON 写法差别。SGLang MLA 的 post-load 路径在
`sglang_kernel_src/python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:508-628`
只专门处理 AWQ、FP8 和 INT8 的 `kv_b_proj`，没有 SIMO packed INT4 分支；随后把处理后的权重拆成
`w_kc/w_vc`。普通路径又在
`sglang_kernel_src/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:505-529`
直接把 `w_kc` 传给 BMM。受控实验去掉 `kv_b_proj` exclusion 后，packed INT4 最终进入该 BMM，得到
`expected scalar type BFloat16 but found Int`。因此当前 SGLang 配置排除 `kv_b_proj` 是必要的架构约束。

vLLM 的 SIMO MLA 则显式识别 `kv_b_proj.quant_method`，见
`simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:291-306`，并在 prefill 中通过正常的
`self.kv_b_proj(...)` 调用量化 Linear，见
`simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:368-377`。所以两边当前不能只换一个 JSON
就获得完全相同的 DeepSeek MLA 权重执行图。

最后，attention 参数也没有按测试名称生效。SGLang 测试接收 `attention_backend`，但
`tests/sglang_simo/e2e_test/test_basic_generate_single.py:135-161` 构造 Engine 时没有传它；原日志实际
使用默认 `fa3`。vLLM 则在 `tests/vllm_simo/e2e_test/test_basic_generate_single.py:178-187` 明确传入
`TRITON_MLA`。受控实验给 SGLang 显式传 `triton` 后仍选择填空 token，因此这也是对照混杂项，而不是
单独的根因或修复。

### 62.5 受控排除结果

本次使用同一模型目录、单请求、`max_new_tokens=16` 和显式 greedy 参数完成了以下检查：

| 对照 | 结果 | 说明 |
| --- | --- | --- |
| SGLang BF16 + 原长提示 | ` Paris...` | BF16 基线通过原断言 |
| SGLang W4A16 + 原长提示 | ` ________...` | 复现问题 |
| SGLang BF16 + 纯事实短提示 | ` Paris...` | 通过 |
| SGLang W4A16 + 纯事实短提示 | ` Paris...` | 通过，且与 BF16 的 `Paris` logprob 几乎相同 |
| SGLang W4A16，额外排除 `kv_a` | ` ________...` | 不是单个 `kv_a` 量化导致 |
| vLLM W4A16，显式 `temperature=0` | ` Paris...` | 原日志的 0.3 不是唯一差别 |
| 两边都只量化 `q/o/MLP/MoE` | SGLang 填空，vLLM `Paris` | 量化覆盖差异不是唯一原因 |
| SGLang 显式 `triton` | ` ________...` | 漏传 backend 不是唯一原因 |
| vLLM 把 prefill 上限从 32 改为 8192 | ` Paris...` | chunked prefill 不是唯一原因 |

这些结果把可支持的结论限定得很清楚：首 token 的框架差异是真实的，但它来自量化舍入、DeepSeek
模型实现、MLA/Linear/MoE 执行路径等数值差异的累积；现有证据不能把它归因于某一个已经证明错误的
SGLang kernel。特别是，匹配量化层集合、attention backend 或 prefill 分块中的任意单项，都没有让
两边 logits 完全一致。

任务级参考结果也不呈现整体损坏。DeepSeek W4A16 的 MMLU 参考分数在 SGLang/vLLM 中分别为
`54.62/55.02`，见 `tests/sglang_simo/references_accuracy/mmlu.yaml:44-50` 和
`tests/vllm_simo/references_accuracy/mmlu.yaml:54-81`；GSM8K 分别为 `58.68/57.32`，见
`tests/sglang_simo/references_accuracy/gsm8k.yaml:43-49` 和
`tests/vllm_simo/references_accuracy/gsm8k.yaml:54-81`。这比单条开放式生成的 exact substring 更能
反映量化模型总体精度。

### 62.6 测试应如何修改

当前断言位于 `tests/sglang_simo/e2e_test/test_basic_generate_single.py:177-188`，建议按测试目标拆分：

1. 基础 smoke test 直接使用无歧义短提示 `The capital of France is`，不要拼接教师面试长前缀。
   本次实测 SGLang W4A16 对该短提示会输出 `Paris`。
2. 两个框架都显式构造相同的 greedy 参数，不要在 vLLM 中调用
   `get_default_sampling_params()`。同时统一 `max_tokens/max_new_tokens`、stop 条件和 prompt token ids。
3. 修复 SGLang 测试中未传递 `attention_backend` 的问题，至少让测试名、参数和实际 backend 一致；
   但不要把它描述成本次输出分叉的充分修复，因为实测显式 `triton` 仍会选择填空。
4. 如果目标是比较 SGLang 与 vLLM 的量化数值，先明确共同量化层。当前 SGLang 不能直接把 SIMO
   INT4 `kv_b_proj` 接入 MLA absorb BMM；在补齐该支持前，应在两边都排除 `kv_a/kv_b`，并记录这不是
   原始两份配置的默认语义。
5. 不要用一条自然语言生成的 `"Paris" in text` 作为量化正确性的唯一判据。更合适的层次是：
   smoke 检查非空输出和无运行期错误；数值测试比较指定 token 的 rank/logprob 或中间张量；精度门禁使用
   MMLU、GSM8K 等多样本指标及允许阈值。

一句话概括：SGLang 并不是因为 `temperature=0` 而“失去 Paris”；它是在一个 `Paris` 与填空符号
本就接近的歧义提示上，经 W4A16 和框架执行路径扰动后把填空符号排到了第一名，而 greedy decoding
忠实且确定性地放大了这个首 token 选择。

## 63. Llama-3.1-8B-Instruct W4A16 复测及双模型稳定输出 Paris 的设置

### 63.1 直接结论

使用用户指定的模型目录
`/data/like/hf-models/Llama3.1-8B-Instruct/`，以及同一份
`simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w4a16_int4_per_group.json:1-27`
启动 SGLang SIMO 后，原测试的“教师面试长前缀 + `The capital of France is`”确实能够让 Llama
以 `Paris` 作为首个生成 token：

```text
 Paris. This is a statement of fact, not a question. I will not ...
```

该请求连续执行 3 次结果相同。首 token 是 id `12366`，解码文本为带前导空格的 `" Paris"`；
它的 logprob 为 `-0.2093`，第二名为 `-3.0843`，对应约 `2.875` 的 logit 间隔。因此，对问题
“Llama-3.1-8B-Instruct 用同样 W4A16 时，SGLang SIMO 还能否得到 Paris 作为首 token”，答案是
**可以**。

不过，这不能说明原长提示本身适合作为跨模型测试。相同输入在 DeepSeek-V2-Lite-Chat W4A16
上的第一名仍是填空 token `" ________"`，它只比 `" Paris"` 高 `0.375` logit。Llama 与
DeepSeek 对一段相互冲突的续写上下文作出不同选择是正常的模型行为，不应通过调整采样参数来掩盖。

### 63.2 受控复测结果

本次两个模型均使用上述 W4A16 INT4 per-group 配置、`temperature=0`、`top_k=1`、
`top_p=1.0` 和 `max_new_tokens=16`。`temperature=0` 在 SGLang 内部本来也会归一化为
`top_k=1`，见
`sglang_kernel_src/python/sglang/srt/sampling/sampling_params.py:168-172`。

| 输入方式 | Llama-3.1-8B-Instruct | DeepSeek-V2-Lite-Chat |
| --- | --- | --- |
| 原教师面试长前缀 | 3/3 均以 ` Paris` 开头 | 3/3 均以 ` ________` 开头 |
| 裸字符串 `The capital of France is` | 3/3 均以 ` a city...` 开头，稍后含 `Paris` | 3/3 均以 ` Paris` 开头 |
| 各自 chat template + 明确单词回答指令 | 3/3 均为 `Paris.` | 3/3 均为 ` Paris` |
| 同一 chat 请求组成 4 请求 batch | 4/4 均为 `Paris.` | 4/4 均为 ` Paris` |

Llama 裸短提示的首选 `" a"` 只比第二名 `" Paris"` 高 `0.25` logit；所以“去掉长前缀”
虽然能让输出中出现 `Paris`，却不能在 Llama 上保证它是首 token。真正适合两个 instruct/chat 模型
的共同输入方式，是让各自 tokenizer 用自己的 chat template 格式化一条无歧义问题：

```text
What is the capital of France? Answer with exactly one word.
```

此时 Llama 的首 token 是 id `60704`、文本 `"Paris"`，相对第二名约高 `12.25` logits；
DeepSeek 的首 token 是 id `8913`、文本 `" Paris"`，相对第二名约高 `4.875` logits。这两个间隔
都远大于原 DeepSeek 长提示中只有 `0.375` 的反向间隔，因而对量化舍入和执行顺序的小扰动更稳健。

### 63.3 推荐的自然生成设置

两个模型可以共用同一套 Engine 构造和采样代码，但 **prompt 必须分别由各自 tokenizer 的 chat
template 生成**。不要自己拼一个看似通用的 `User:`/`Assistant:` 字符串，也不要把与问题无关的教师
面试前缀保留下来。

下面的设置可直接用于这两个目录：

```python
import json

import sglang as sgl
from transformers import AutoTokenizer

QUANT_CONFIG = (
    "/share/users/like/package/simo_conda_sglang/simo/extensions/sglang_simo/"
    "example/simo_quantization_config/online_quantization/"
    "quant_config_w4a16_int4_per_group.json"
)


def run_one(model_path: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "What is the capital of France? "
                    "Answer with exactly one word."
                ),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    engine = sgl.Engine(
        model_path=model_path,
        quantization="simo",
        json_model_override_args=json.dumps(
            {"quantization_config_file": QUANT_CONFIG}
        ),
        mem_fraction_static=0.5,
        log_level="info",
    )
    try:
        output = engine.generate(
            [prompt],
            {
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "max_new_tokens": 16,
            },
        )[0]["text"]
    finally:
        engine.shutdown()

    answer = output.strip().rstrip(".")
    assert answer == "Paris", repr(output)
    return output


models = [
    (
        "/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/"
        "DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights"
    ),
    "/data/like/hf-models/Llama3.1-8B-Instruct",
]
for model in models:
    print(model, repr(run_one(model)))
```

为避免两个 Engine 的显存占用叠加，上面的 `finally` 会在切换模型前释放当前 Engine。
实际测试若只运行一个模型，可以沿用
`tests/sglang_simo/e2e_test/test_basic_generate_single.py:150-161` 的 Engine 生命周期，只替换
`tests/sglang_simo/e2e_test/test_basic_generate_single.py:163-188` 中的采样参数、prompt 构造和断言。

W4A16 配置本身不需要为 Llama 另写一份。它量化 `Linear` 的权重为 INT4、沿 `axis=-1` 每 32
元素一组，并排除 `lm_head` 和 `kv_b_proj`，见
`simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w4a16_int4_per_group.json:6-23`。
Llama 没有 `kv_b_proj`，该正则不会额外影响它；DeepSeek 则必须保留这个 exclusion，原因见第
62.4 节对 SGLang MLA 路径的分析。

这里显式写 `top_k=1` 是为了让测试意图清晰，虽然 `temperature=0` 已经会触发同一 greedy
归一化。`top_p=1.0` 避免引入额外过滤。对输出先 `strip()` 再去掉句号，是因为两个 tokenizer
生成的首 token 空格形式不同，而模型给一词答案加句号也是合法行为。

### 63.4 为什么不把 deterministic inference 当作修复

`enable_deterministic_inference=True` 的定义是启用 batch-invariant operators，见
`sglang_kernel_src/python/sglang/srt/server_args.py:2569-2575`；它还会调整 sampling backend、
attention backend 等实现，见 `sglang_kernel_src/python/sglang/srt/server_args.py:6025-6095`。
它解决的是同一输入在不同 batch 组织下的数值一致性问题，并不规定模型必须选择哪个语义答案。

实测 DeepSeek W4A16 在该模式下运行原长提示三次，首次冷请求为填空，后续两个复用前缀缓存的请求为
`Paris`；最后一次记录到 `Paris` 与填空 token 的 logprob 恰好同为 `-1.2311`。这再次表明原长
提示处在决策边界附近，也说明 batch-invariant operators 不等于跨冷/缓存执行状态强制得到某个语义答案。
deterministic inference 不能替代明确 prompt。推荐的 chat prompt 在不开启该选项时已经通过 3 次串行
和 4 请求 batch 检查，因此不需要为了这个 smoke test 打开它。

### 63.5 如果协议要求强制得到 Paris

如果测试目标只是确认“请求经过 SGLang、量化模型和 detokenizer 后能返回指定字符串”，而不是检查模型
是否知道法国首都，可以使用 grammar 约束。SGLang 的 `regex` 参数声明在
`sglang_kernel_src/python/sglang/srt/sampling/sampling_params.py:102-104`。对原始长提示，下面的设置
在两个模型上均实测只生成 `" Paris"`：

```python
forced = engine.generate(
    [long_prefix + "The capital of France is"],
    {
        "temperature": 0.0,
        "top_k": 1,
        "max_new_tokens": 2,
        "regex": " Paris",
    },
)[0]["text"]
assert forced == " Paris"
```

也可以对首 token 加很大的正 bias。不要硬编码跨模型 token id，应由当前 tokenizer 计算：

```python
paris_ids = tokenizer.encode(" Paris", add_special_tokens=False)
assert len(paris_ids) == 1

forced = engine.generate(
    [long_prefix + "The capital of France is"],
    {
        "temperature": 0.0,
        "top_k": 1,
        "max_new_tokens": 1,
        "logit_bias": {str(paris_ids[0]): 100.0},
    },
)[0]["text"]
assert forced == " Paris"
```

本次两个 id 分别是 DeepSeek 的 `8913` 和 Llama 的 `12366`。SGLang 在
`sglang_kernel_src/python/sglang/srt/sampling/sampling_params.py:112-113` 接收 `logit_bias`，在
`sglang_kernel_src/python/sglang/srt/sampling/sampling_batch_info.py:125-131` 构造逐请求 bias，并在
`sglang_kernel_src/python/sglang/srt/sampling/sampling_batch_info.py:282-283` 加到 logits 上。

`regex` 会从允许的字符序列上作硬约束；有限的 `+100` bias 是本次模型上足够大的实用强制值，但不是
与模型 logits 无关的数学保证。更重要的是，这两种方式都把正确答案直接写进了解码约束，因而只能用于
接口/链路 smoke test，**不能**再把成功输出 `Paris` 当作 W4A16 数值正确或模型知识正确的证据。

最终建议按测试目的选择：事实回答 smoke test 使用“各自 chat template + 明确问题 + greedy”，这是
两个模型自然且共同稳定的设置；协议层 exact-output test 才使用 `regex`；量化正确性仍应使用第 62.6
节所述的 token rank/logprob、中间张量和多样本精度指标，而不是依靠一条被强制的字符串。

## 64. `apply_chat_template` 与原 `list[str]` prompt 的区别

### 64.1 两者不是同一维度的替代项

最核心的区别是：

- `apply_chat_template` 决定 **一条对话在送入模型前如何序列化**，即加入模型训练时约定的角色、分隔符、
  BOS/EOS 和 assistant 起始标记。
- 传给 `Engine.generate()` 的外层 `list` 决定 **一次提交多少条请求**。它只是 batch 容器，不会自动
  把普通文本变成 chat prompt。

因此，原代码中的：

```python
prompts = ["The capital of France is"]
generating_prompts = [prefix + prompt for prompt in prompts]
outputs = llm.generate(generating_prompts, sampling_params)
```

其类型和含义是：

```text
prompts / generating_prompts: list[str]  -> 一批普通文本续写请求
```

具体位置是 `tests/sglang_simo/e2e_test/test_basic_generate_single.py:164-182`。虽然这里只有一个
字符串，外层仍表示 batch size 为 1。SGLang 的 `Engine.generate()` 明确接受单个 `str` 或
`list[str]`，也接受单个 `list[int]` 或 batch 形式的 `list[list[int]]`，见
`sglang_kernel_src/python/sglang/srt/entrypoints/engine.py:318-324`；请求归一化时，字符串被判定为单请求，
字符串列表被判定为 batch，见 `sglang_kernel_src/python/sglang/srt/managers/io_struct.py:371-389`。

而下面的 `messages`：

```python
messages = [
    {
        "role": "user",
        "content": "What is the capital of France? Answer with exactly one word.",
    }
]
```

是 **一条对话的消息列表**，不是一批 prompt。`apply_chat_template` 把它转换成一个格式化字符串或一组
token ids；转换完成后，仍需用外层 list 才组成 batch：

```text
list[dict]       -> 一条 conversation
str              -> 格式化后的一条文本 prompt
list[int]        -> 格式化并 tokenize 后的一条 token prompt
list[str]        -> Engine 的文本 batch
list[list[int]]  -> Engine 的 token-id batch
```

### 64.2 模型实际看到的内容不同

原代码直接拼接 `prefix + prompt`。除 tokenizer 可能自动添加的 BOS 外，模型看到的就是这些普通文本；
其中没有 user/assistant 边界，也没有“现在开始 assistant 回答”的控制 token。对 causal LM 而言，这更像
“继续写前文”，所以原教师面试前缀会强烈影响它是续写面试问题、填空，还是回答法国首都。

`apply_chat_template` 会读取当前模型 `tokenizer_config.json` 中的 `chat_template`。两个模型的模板不同：

- DeepSeek-V2-Lite-Chat 大致渲染为
  `BOS + "User: ...\n\nAssistant:"`。模板位于
  `/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights/tokenizer_config.json:34`。
- Llama-3.1-8B-Instruct 会加入 `<|begin_of_text|>`、system/user/assistant header 和
  `<|eot_id|>` 等控制 token。模板位于
  `/data/like/hf-models/Llama3.1-8B-Instruct/tokenizer_config.json:2053`。

例如同一句问题，逻辑上分别接近：

```text
# DeepSeek
<BOS>User: What is the capital of France? Answer with exactly one word.

Assistant:

# Llama
<BOS><system-header>...<EOT><user-header>
What is the capital of France? Answer with exactly one word.
<EOT><assistant-header>
```

`add_generation_prompt=True` 的作用就是在末尾加入 assistant 消息的起始标记；它不生成答案，而是告诉
chat/instruct 模型“下一段应当由 assistant 生成”。因此第 63 节中输出从临界或填空行为变为高 margin
的 `Paris`，主要是 **输入语义和格式变得明确**，不是 `list`、W4A16 或 `temperature=0` 的含义改变了。

### 64.3 `tokenize=False` 和 `tokenize=True`

`apply_chat_template` 有两种常用返回方式：

```python
# 返回已经带角色/控制标记的 str
rendered = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# 直接返回正确模板对应的 list[int]
prompt_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=False,
)
```

当前 conda 环境中的 Transformers 实现在
`/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/transformers/tokenization_utils_base.py:2991-3008`
定义返回类型，在
`/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/transformers/tokenization_utils_base.py:3030-3042`
说明 `add_generation_prompt` 和 `tokenize`，并在
`/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/transformers/tokenization_utils_base.py:3121-3128`
对渲染结果使用 `add_special_tokens=False` 编码。这是因为模板本身已经负责放置 BOS/EOS 等控制 token。

如果使用 `tokenize=False`，随后把字符串传给 `Engine.generate(prompt=...)`，SGLang 还会再次调用其内部
tokenizer，见 `sglang_kernel_src/python/sglang/srt/managers/tokenizer_manager.py:793-832`。当前普通 tokenizer
路径使用默认 `encode()`/`tokenizer()` 参数，见
`sglang_kernel_src/python/sglang/srt/managers/tokenizer_manager.py:775-786`，这可能再次添加 BOS。

本次对当前 editable SGLang 实际加载的 tokenizer 做 token 对照，结果是：

| 模型 | `apply_chat_template(tokenize=True)` | 模板字符串再走默认 `encode()` |
| --- | --- | --- |
| Llama-3.1-8B-Instruct | 48 tokens，以 `[128000, 128006, ...]` 开头 | 49 tokens，以 `[128000, 128000, 128006, ...]` 开头 |
| DeepSeek-V2-Lite-Chat | 20 tokens，以 `[100000, 5726, ...]` 开头 | 21 tokens，以 `[100000, 100000, 5726, ...]` 开头 |

也就是说，第 63.3 节中 `tokenize=False` 后再传 `prompt=[prompt]` 的写法在这个简单问题上实测仍能输出
`Paris`，但当前两种内部 tokenizer 都会多放一个 BOS。它适合说明 chat template 的语义效果，却不是做
token 级可复现测试时最干净的路径。SGLang 自己的 OpenAI chat 实现也明确在模板渲染后用
`add_special_tokens=False` 编码以避免 double BOS，见
`sglang_kernel_src/python/sglang/srt/entrypoints/openai/serving_chat.py:854-875`。

### 64.4 推荐写法：模板直接生成 ids

对当前 `sgl.Engine` 测试，推荐让外部 tokenizer 一次性完成模板和 tokenization，再通过 `input_ids=`
传入 Engine：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)
messages = [
    {
        "role": "user",
        "content": "What is the capital of France? Answer with exactly one word.",
    }
]
prompt_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=False,
)

outputs = llm.generate(
    input_ids=[prompt_ids],  # 外层 list 表示 batch size = 1
    sampling_params={
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "max_new_tokens": 4,
    },
)
generated_text = outputs[0]["text"]
assert generated_text.strip().rstrip(".") == "Paris"
```

`Engine.generate()` 会把 `prompt=` 放入请求的 `text` 字段，把 `input_ids=` 放入同名字段，见
`sglang_kernel_src/python/sglang/srt/entrypoints/engine.py:370-400`。TokenizerManager 检测到已有
`input_ids` 后会直接使用，不再编码文本，见
`sglang_kernel_src/python/sglang/srt/managers/tokenizer_manager.py:805-832`。SGLang 的 Ollama chat 入口也采用
`apply_chat_template(tokenize=True)` 后传 `input_ids` 的方式，见
`sglang_kernel_src/python/sglang/srt/entrypoints/ollama/serving.py:79-94`。

该 token-id 写法已在两个 W4A16 模型上重新实测：Llama 输出 `Paris.`，DeepSeek 输出 ` Paris`。它与
第 63 节的自然生成结论一致，同时去掉了额外 BOS。

如果有多条对话，分别格式化后再组成 token-id batch：

```python
conversations = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "What is the capital of Japan?"}],
]
batch_input_ids = [
    tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    for messages in conversations
]
outputs = llm.generate(input_ids=batch_input_ids, sampling_params=sampling_params)
```

不要同时传 `prompt=` 和 `input_ids=`，也不要对已经包含 chat 控制 token 的字符串再次调用
`apply_chat_template`。如果改用 SGLang 的 OpenAI `/v1/chat/completions` 接口，则直接发送结构化
`messages`，server 会负责应用 chat template；客户端不应再预先套一遍模板。

一句话总结：原来的 `list[str]` 是“把若干原始文本作为 completion batch”，而
`apply_chat_template` 是“按当前 instruct 模型训练时的对话协议构造每条输入”；正确组合是先对每条
conversation 应用模板，再把得到的字符串或 token ids 组成 batch。对当前 Engine 和这两个模型，优先
使用 `tokenize=True` + `input_ids=[prompt_ids]`，可以同时获得正确对话格式和无重复 BOS 的 token 序列。

## 65. 使用 SGLang 原生插件框架注册 `sglang_simo`

### 65.1 结论

SIMO 已使用 SGLang 的 general plugin 机制替代 `sitecustomize.py` 注册。需要注意
`SGLANG_PLUGINS` 的作用：它不是 Python 模块路径，也不会直接执行环境变量中写出的函数。SGLang 先通过
Python distribution metadata 查找 `sglang.srt.plugins` 组中的 entry point，再把
`SGLANG_PLUGINS` 当作 **entry point 名称白名单**。实现位于
`sglang_kernel_src/python/sglang/srt/plugins/__init__.py:35-86`；环境变量定义位于
`sglang_kernel_src/python/sglang/srt/environ.py:974-976`。

本次迁移已经完成两个部分：

1. SIMO 的 `pyproject.toml` 已声明 SGLang entry point，安装后的 distribution metadata 中可以发现该插件；
2. 启动 SGLang 前设置 `SGLANG_PLUGINS=sglang_simo_extensions`，由白名单选中它。

当前配置在原有 vLLM entry point 后使用独立的 SGLang 组：

```toml
[project.entry-points."vllm.general_plugins"]
vllm_simo_extensions = "simo.extensions.vllm_simo:register_simo_extensions"

[project.entry-points."sglang.srt.plugins"]
sglang_simo_extensions = "simo.extensions.sglang_simo:register_simo_extensions"
```

这里三个字段的含义分别是：

| 字段 | 值 | 含义 |
| --- | --- | --- |
| entry point group | `sglang.srt.plugins` | SGLang general plugin loader 固定查询的组名 |
| entry point name | `sglang_simo_extensions` | `SGLANG_PLUGINS` 中应填写的名字 |
| entry point value | `simo.extensions.sglang_simo:register_simo_extensions` | `ep.load()` 要导入的模块和解析的函数 |

不要写成下面任一种形式：

```bash
# 错误：环境变量匹配的是 entry point 左侧名称，不是模块路径或函数名
export SGLANG_PLUGINS=simo.extensions.sglang_simo
export SGLANG_PLUGINS=register_simo_extensions
```

正确写法是：

```bash
export SGLANG_PLUGINS=sglang_simo_extensions
```

如果要同时允许多个 general plugin，用逗号分隔其 entry point 名称，例如
`SGLANG_PLUGINS=sglang_simo_extensions,another_plugin`。解析代码见
`sglang_kernel_src/python/sglang/srt/plugins/__init__.py:51-70`。

### 65.2 当前迁移状态与安装

迁移后的状态如下：

| 检查项 | 当前状态 |
| --- | --- |
| `pyproject.toml` | 已声明 `sglang.srt.plugins` / `sglang_simo_extensions` |
| `setup.py` | 已删除 `py_modules=["sitecustomize"]`，未重复声明 entry point |
| 仓库根目录 `sitecustomize.py` | 已删除 |
| 安装元数据 | 已同时包含 SGLang 和 vLLM 两组 entry point |
| Python 模块发现 | `importlib.util.find_spec("sitecustomize") is None` |

entry point 属于安装元数据；即使 SIMO 使用 editable 安装，修改 `pyproject.toml` 后也必须重装。2026-07-24
实际使用的安装命令为：

```bash
cd /share/users/like/package/simo_conda_sglang
source /share/users/like/package/sglang_kernel_src/like-useful/env-build-pip.sh

/share_data/users/like/miniconda3/envs/simo_sglang/bin/python -m pip uninstall -y simo
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python \
  -m pip install -e ".[dev]" --no-build-isolation
```

当前仓库曾保留一份被 `.gitignore` 忽略的旧 `simo.egg-info`。新 editable wheel 的 `dist-info` 虽然正确，
但从仓库根目录启动 Python 时，旧 `egg-info` 会优先遮蔽它。本次通过以下 setuptools 命令重新生成该构建
元数据后，仓库内外查询结果一致：

```bash
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python setup.py egg_info
```

不要手工修改 `simo.egg-info/entry_points.txt`；它是构建工具根据 `pyproject.toml` 生成的文件。

### 65.3 `SGLANG_PLUGINS` 到 SIMO 函数的完整调用栈

以用户代码中的 `sgl.Engine(...)` 为例，主进程调用链如下：

```text
用户代码
  -> sgl.Engine(...)
  -> sglang.srt.entrypoints.engine.Engine.__init__()
       engine.py:204
  -> load_plugins()
       engine.py:210-212
  -> load_plugins_by_group("sglang.srt.plugins", ...)
       plugins/__init__.py:124-127
  -> envs.SGLANG_PLUGINS.get()
       plugins/__init__.py:51-55
  -> os.getenv("SGLANG_PLUGINS")
       environ.py:48-61
  -> importlib.metadata.entry_points(group="sglang.srt.plugins")
       plugins/__init__.py:57
  -> 比较 ep.name 与 {"sglang_simo_extensions"}
       plugins/__init__.py:67-70
  -> ep.load()
       plugins/__init__.py:79-82
       导入 simo.extensions.sglang_simo
       解析属性 register_simo_extensions
  -> func()
       plugins/__init__.py:129-134
  -> simo.extensions.sglang_simo.register_simo_extensions()
       simo/extensions/sglang_simo/__init__.py:11-32
  -> HookRegistry.apply_hooks()
       plugins/__init__.py:140-141
  -> Engine 构造 ServerArgs
       engine.py:214-223
```

其中 `ep.load()` 只负责导入模块并返回函数对象；真正调用
`register_simo_extensions()` 的位置是 `func()`，两步不能混为一谈。

`register_simo_extensions()` 不需要参数，返回值也不会被使用，因此它已经满足 general plugin 的调用
约定。该函数执行后通过导入或显式调用完成以下注册：

- `simo/extensions/sglang_simo/model_loader/loader.py:92-102` 注册 model loader 相关 patch；
- `simo/extensions/sglang_simo/quantization/quantization_registry.py:8-15` 把 `simo` 加入量化映射；
- `simo/extensions/sglang_simo/server_args.py:1-4` 加入 `simo` 和 `triton_simo` 参数选项；
- `simo/extensions/sglang_simo/layers/attention/attention_backend.py:5-18` 注册
  `triton_simo` attention backend；
- `simo/extensions/sglang_simo/__init__.py:19-27` 加载 DeepSeek-V2 patch 并应用 memory-pool patch。

SGLang 还会从其他入口调用同一个 loader：

| 启动路径 | 首次调用位置 | 调用时机 |
| --- | --- | --- |
| `sglang serve ...` | `sglang_kernel_src/python/sglang/cli/serve.py:89-93` | 解析模型和 server 参数之前 |
| `python -m sglang.launch_server ...` | `sglang_kernel_src/python/sglang/launch_server.py:64-68` | `prepare_server_args()` 之前 |
| `sgl.Engine(...)` | `sglang_kernel_src/python/sglang/srt/entrypoints/engine.py:204-223` | `ServerArgs` 构造之前 |
| Engine 主进程防御性调用 | `sglang_kernel_src/python/sglang/srt/entrypoints/engine.py:782-790` | 检查 server 参数之前 |
| scheduler 子进程 | `sglang_kernel_src/python/sglang/srt/managers/scheduler.py:4281-4295` | scheduler 配置和构造之前 |

`sglang_kernel_src/python/sglang/srt/plugins/__init__.py:31-32,119-122` 的 `_plugins_loaded` 保证同一进程中
只执行一次。scheduler 是新进程，有自己的 `_plugins_loaded=False`，所以会再次发现并执行插件。这正是
SIMO 所需要的：主进程要在参数校验前认识 `simo`/`triton_simo`，scheduler 进程也要在模型和 attention
backend 初始化前应用对应注册及 patch。`SGLANG_PLUGINS` 必须在启动 Python 进程前设置，以便子进程继承；
同一进程首次 `load_plugins()` 完成后再修改它不会重新加载插件。

### 65.4 `setup.py` 和 `pyproject.toml` 的职责

当前项目使用 `setuptools.build_meta`，见 `pyproject.toml:1-3`。SGLang entry point 只在
`pyproject.toml` 中声明；`setup.py` 不再重复提供 `entry_points={...}`，避免出现两份配置源。

旧自动导入路径已被彻底删除：`setup.py` 不再包含 `py_modules=["sitecustomize"]`，仓库根目录也不再有
`sitecustomize.py`。因此 `SIMO_SGLANG_REGISTER` 和 `SIMO_DISABLE_SGLANG_REGISTER` 不再参与 SIMO 的
SGLang 注册；实际入口统一为 SGLang loader 和 `SGLANG_PLUGINS`。

这两项删除必须同时完成。仅从 packaging 配置移除 `sitecustomize`，但保留仓库根目录文件，仍可能导致从
该目录启动 Python 时发生自动导入；仅删除源码文件，但保留旧安装或旧 metadata，也可能留下错误的模块
映射。重新安装后应同时检查 entry point 和 `find_spec("sitecustomize")`。

### 65.5 安装后如何验证

先验证安装元数据和旧模块清理状态，不启动模型、不占用 GPU：

```bash
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
import importlib.util
from importlib.metadata import entry_points

plugins = {
    ep.name: ep.value
    for ep in entry_points(group="sglang.srt.plugins")
}
print(plugins)
assert plugins["sglang_simo_extensions"] == (
    "simo.extensions.sglang_simo:register_simo_extensions"
)
assert importlib.util.find_spec("sitecustomize") is None
PY
```

预期至少包含：

```text
{'sglang_simo_extensions': 'simo.extensions.sglang_simo:register_simo_extensions'}
```

再验证 SGLang 确实执行了函数，而不是只有元数据：

```bash
SGLANG_PLUGINS=sglang_simo_extensions \
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
from sglang.srt.plugins import load_plugins

load_plugins()

from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES, QUANTIZATION_CHOICES
from sglang.srt.layers.quantization import BASE_QUANTIZATION_METHODS

assert "simo" in QUANTIZATION_CHOICES
assert "simo" in BASE_QUANTIZATION_METHODS
assert "triton_simo" in ATTENTION_BACKEND_CHOICES
print("sglang_simo plugin registration: OK")
PY
```

实际启动方式为：

```bash
CUDA_VISIBLE_DEVICES=7 \
SGLANG_PLUGINS=sglang_simo_extensions \
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python your_test.py
```

还需注意两个 loader 语义：

- `SGLANG_PLUGINS` 未设置或为空时，`allowed_set` 为 `None`，SGLang 会加载所有已经安装的
  `sglang.srt.plugins`，不是“一个也不加载”，见
  `sglang_kernel_src/python/sglang/srt/plugins/__init__.py:51-70`。因此 entry point 安装后，即使不设置该变量，
  SIMO 也会被自动发现和执行；显式设置它的价值是限定只加载指定插件。
- 如果设置了一个不存在或拼错的名称，discovered entry point 会被跳过，但这里不会因为“白名单名称没有
  命中”而主动抛错。插件导入或执行异常也会在
  `sglang_kernel_src/python/sglang/srt/plugins/__init__.py:79-84,129-138` 被记录为日志。因此应保留上面的
  metadata 和注册结果断言，不能只根据进程仍能启动就判定插件生效。

### 65.6 四项 tokenizer smoke 结果

`like-useful/test_basic_generate_tokenizer_prompt_all_in_one.py` 不再扫描配置目录，而是固定为两个模型和两个
代表性量化配置，共四项：

| 模型 | 量化配置 | `attention_backend` | 2026-07-24 结果 |
| --- | --- | --- | --- |
| DeepSeek-V2-Lite-Chat | `quant_config_kvquant_fp8_per_group.json` | `triton_simo` | PASS，`Paris` 断言通过 |
| DeepSeek-V2-Lite-Chat | `quant_config_w4a16_int4_per_group.json` | 未显式传入 | PASS，`Paris` 断言通过 |
| Llama-3.1-8B-Instruct | `quant_config_kvquant_fp8_per_group.json` | `triton_simo` | PASS，`Paris` 断言通过 |
| Llama-3.1-8B-Instruct | `quant_config_w4a16_int4_per_group.json` | 未显式传入 | PASS，`Paris` 断言通过 |

Llama 使用当前可访问的 `/data/like/hf-models/Llama3.1-8B-Instruct`；DeepSeek 使用
`/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights`。
四项都保留模型 chat template、`temperature=0.0` 和生成文本必须包含 `Paris` 的断言。

在一块空闲 H100（`CUDA_VISIBLE_DEVICES=7`）上串行执行：

```bash
source /share/users/like/package/sglang_kernel_src/like-useful/env-build-pip.sh
CUDA_VISIBLE_DEVICES=7 \
SGLANG_PLUGINS=sglang_simo_extensions \
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python -m pytest -q \
  like-useful/test_basic_generate_tokenizer_prompt_all_in_one.py
```

pytest 明确收集 4 项，最终结果为 `4 passed, 2 warnings in 276.70s (0:04:36)`。两条 warning 是 SWIG
类型的既有 `DeprecationWarning`；日志中没有旧 `sitecustomize` 注册错误。独立的 `load_plugins()` 验证还确认
了 `simo` 量化选择、`simo` quantization method、`triton_simo` attention backend、DeepSeek attention
替换和 SIMO memory-pool patch 均已生效。

## 66. 为什么 INT8 per-tensor 激活在 ONNX Q/DQ 中被写成 per-channel

### 66.1 直接结论

这里没有把量化的数学语义从 per-tensor 改成真正的 per-channel。它做的是一次 lowering：先把整个激活
张量展平成只有一行的二维张量，再用 `axis=0` 的 per-channel 内核处理这一行。因为 channel 维只有 1，
内核最终仍只计算一个 scale；其结果与对原张量做 per-tensor 量化等价。

因此需要区分两层含义：

| 层次 | 本例中的值 | 含义 |
| --- | --- | --- |
| 用户配置和 `spec` | `per_tensor` | 整个激活共享一个量化 scale |
| 写入 `com.simo::Quantize/Dequantize` 的属性 | `per_channel, axis=0` | 选择当前插件已有的 per-channel kernel ABI |
| 传给 Q/DQ 的实际布局 | `[1, B*S*18]` | 只有一个 channel，所以仍只有一个 scale |

换句话说，代码改的是自定义算子的内核选择属性，不是原始量化配置，也没有让每个 token 或每个 feature
获得独立 scale。

### 66.2 为什么这个 case 会进入 `single_row` 分支

`like-useful/test_dynamic_qdq_runtime_debug-debug.py:1594-1612` 中该参数为：

```python
shape_id = "symbolic"
dtype = "int8"
input_shape = ["B", "S", 18]
```

测试在 `:1215-1217` 传入的激活配置实际只有 `{"dtype": "int8"}`，没有 `axis` 和 `group_size`。
`get_quantize_granularity()` 在 `simo/quantization/config.py:229-233` 中把
`axis is None and group_size is None` 判定为 `PER_TENSOR`。同时，
`simo/onnx/onnx_quant.py:44` 的 `PER_TENSOR_CHANNEL_KERNEL_DTYPES` 包含 `int8`，所以
`onnx_quant.py:760-762` 等价于：

```python
single_row = "int8" in {"fp8_e4m3", "int8"} and granularity == PER_TENSOR
```

结果必然为 `True`。这个条件根本没有读取 `shape_id` 或 `input_shape`；`symbolic` 不是分支触发原因，末维
`18` 也不是。`shape_id` 在该测试中主要用于 case 名和 ONNX 快照文件名。

### 66.3 图变换后为什么仍然是 per-tensor

`single_row=True` 后，`onnx_quant.py:764-768` 设置：

```python
attribute_axis = 0
attribute_granularity = "per_channel"
flatten_axis = 0
alignment = 1
```

随后 `onnx_quant.py:903-915,963-999` 构造 `Shape -> Flatten -> Q -> DQ -> Reshape`。对本例而言：

```text
X: [B, S, 18]
  -> Shape(X) 保存运行时原始形状
  -> Flatten(axis=0): [1, B*S*18]
  -> Quantize(per_channel, axis=0)
  -> Dequantize(per_channel, axis=0)
  -> Reshape(Shape(X)): [B, S, 18]
```

per-channel `axis=0` 会保留第 0 维的每一行，并沿第 1 维求 absmax。展平后第 0 维恒为 1，因此：

```text
per_channel_scale[0]
  = reduce_absmax(flatten(X)[0, :]) / quant_divisor
  = reduce_absmax(X) / quant_divisor
  = per_tensor_scale
```

量化范围、rounding 和 scale mode 也保持不变，所以逐元素 Q/DQ 结果相同。测试本身在
`like-useful/test_dynamic_qdq_runtime_debug-debug.py:1223-1228` 也明确使用
`_reference_int8_per_channel_qdq(tensor.reshape(1, -1)).reshape(tensor.shape)` 作为 INT8 per-tensor 的参考值。

在指定的 `/share_data/users/like/miniconda3/envs/simo_sglang/` 环境中读取并执行该 symbolic ONNX 快照，
运行时形状实际为：

```text
matmul_input_simo_rank2:       (1, 108), float32
matmul_SimoQuantInput:         (1, 108), uint8
matmul_SimoScale:              (1, 4),   uint8
matmul_SimoDequantOutput:      (1, 108), float32
matmul_input_simo_restore:     (2, 3, 18), float32
```

`matmul_SimoScale` 的 `(1, 4) uint8` 不表示 4 个 scale。插件把 FP32 scale 作为原始字节输出；
`simo_qdq_ops.cc:176-190,204-210` 对一个 per-channel scale 分配 `sizeof(float) == 4` 个字节。因此这里在
语义上仍然只有一个 FP32 scale。

### 66.4 为什么不直接保留 `granularity="per_tensor"`

原因是当前 embedded SM90 Q/DQ runtime 没有单独编译 FP8/INT8 per-tensor cubin。
`simo/onnx/ort_plugin/build_qdq_cubins.py:294-303` 对这两种 dtype 只生成 `per_block` 和
`per_channel` 两类 kernel；C++ resolver 又按 `dtype + granularity + scale_mode + quant range` 精确匹配
runtime spec。若图中直接写 `granularity="per_tensor"`，当前 resolver 找不到对应 spec，custom op 会以
unsupported semantic QDQ config 失败，而不是自动退回某个通用实现。

复用 per-channel kernel 也与 SIMO 的 PyTorch CUDA 路径一致：
`simo/ops/flex_api.py:777-793` 中的 `per_tensor_downcast_to_fp8_or_int8_cuda_impl()` 同样先执行
`src_tensor.contiguous().view(1, -1)`，然后调用
`per_channel_downcast_to_fp8_or_int8_triton(..., axis=0)`，最后恢复原形状。ONNX lowering 只是把这套运行时
适配显式表达成图节点，使动态 shape 下也能工作，同时避免维护一份计算完全重复的 per-tensor cubin。

### 66.5 symbolic shape 和末维 18 在这里的作用

`B`、`S` 是 symbolic 不会影响等价性。`Shape(X)` 在运行时取得真实尺寸，`Flatten(axis=0)` 不需要预先知道
`B*S`，最后的 `Reshape` 也使用运行时 shape 恢复。因此同一个模型可以接受不同的 `B` 和 `S`。

末维 `18` 在 INT8 per-tensor 分支中也不会引发 padding。该分支把 `alignment` 强制设为 1；INT8
per-channel kernel 可以用 mask 处理任意行长度，所以运行时的量化长度是完整的 `B*S*18`。测试名中的
`with_padding` 同时覆盖 MXINT8 case；MXINT8 使用 block size 32，才需要对不对齐的末维 18 做 Pad/Slice。

综上，这段代码更准确的描述是“把 per-tensor canonicalize 为 single-row per-channel kernel 调用”，而不是
“把 per-tensor 强行改成多通道量化”。其正确性依赖三个条件必须成套保留：`Flatten(axis=0)`、
`per_channel(axis=0)` 和 Q/DQ 后按原始运行时 shape 恢复。如果将来增加原生 per-tensor custom kernel，可以
删除这层 lowering 并在节点属性中保留 `per_tensor`；在当前 kernel 支持矩阵下，现有实现是有意的内核复用。
