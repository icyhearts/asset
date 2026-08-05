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

## 67. `test_quant_onnx.sh` 的核心功能、Python 调用链、模型和数据集

### 67.1 直接结论

`/share_data/users/tangdehua/project/jdjv/silero_vad_clean/test_quant_onnx.sh` 的核心功能是：

1. 读取 Silero VAD v5 的浮点 ONNX 模型；
2. 按 W8A8 MXINT8 配置把模型改写成带 `com.simo::Quantize/Dequantize` 节点的 SIMO Q/DQ ONNX；
3. 将 AISHELL-4 test 划成 24 个 shard，在 8 张 GPU 上最多并发运行 8 个评测子进程；
4. 汇总每个 512-sample 音频 chunk 的 speech probability 和 RTTM 标签，计算 ROC-AUC 与
   `threshold=0.11` 下的 chunk accuracy；
5. 保存量化模型、分片日志/NPZ、运行元数据和最终 `summary.json`。

因此它是一个“动态 Q/DQ ONNX 量化加多 GPU 精度评测”脚本，不是训练脚本，也不是校准脚本。AISHELL-4
数据只用于量化后精度评测，不参与生成 scale 的离线校准；激活 scale 在模型运行时动态计算。

完整调用链为：

```text
test_quant_onnx.sh
  -> scripts/run_silero_onnx_float_sharded.py
       -> prepare_model()
            -> normalize_quant_config()
            -> simo.onnx.onnx_dynamic_qdq.rewrite_dynamic_qdq()  # 脚本编写时使用的旧 API
            -> onnx.save(quantized_model.onnx)
       -> 24 x scripts/evaluate_silero_vad_onnx_float.py
            -> evaluate_silero_vad_public_v5.SileroVadOnnx
            -> iter_aishell4()/load_audio()/labels_from_segments()
            -> ONNX Runtime CUDA 推理
            -> 每个 shard 写一个 NPZ 和日志
       -> aggregate_results()
       -> summary.json
```

### 67.2 Shell 脚本具体传了什么参数

`test_quant_onnx.sh:4-18` 的默认值如下：

| 项目 | 默认值 | 含义 |
| --- | --- | --- |
| Python | `python` | 可通过 `PY` 覆盖；子进程继续使用同一个 `sys.executable` |
| 主 Python | `scripts/run_silero_onnx_float_sharded.py` | 量化、分片调度和结果聚合 |
| 浮点模型 | `onnx_float_baseline/silero_vad.onnx` | Silero VAD v5 ONNX |
| 数据集 | `aishell4` | 只评测 AISHELL-4 test |
| GPU | `0,1,2,3,4,5,6,7` | 8 张卡 |
| shard | `24` | 文件按索引模 24 分片 |
| worker | `8` | 最多同时运行 8 个 shard 子进程 |
| 量化配置 | `quant_schema/w8a8_mx/w_mxint8_a_mxint8.json` | 权重和激活都使用 MXINT8 |
| ORT custom op | `/tmp/simo_ort_plugin_build/libSimoOnnxCustomOps.so` | 执行 `com.simo` Q/DQ 节点 |
| Triton manifest | `/tmp/simo_mxint8_cubins/simo_mxint8_sm90.manifest.tsv` | custom op 查找 SM90 kernel |
| 输出目录 | `logs/onnx_quant_aishell4_24shards` | 模型、日志、NPZ 和汇总结果 |

`test_quant_onnx.sh:20-24` 会先检查模型、配置、custom-op `.so` 和 manifest 是否存在，然后导出
`SIMO_ONNX_TRITON_MANIFEST`。默认 `FORCE=--no-skip-existing`，所以已有 shard 结果也会重新计算。
`MAX_FILES_PER_DATASET=0` 和 `LIMIT_SECONDS=0` 表示不限制文件数、不截短音频，即运行完整数据集。

模型和 Python 脚本使用相对路径，而 shell 本身没有 `cd`，所以按默认值运行时当前目录应当是
`/share_data/users/tangdehua/project/jdjv/silero_vad_clean`，否则最先执行的 `test -f "$MODEL"` 就可能失败。

### 67.3 第一层 Python：量化、分片调度和聚合

直接调用的是：

```text
/share_data/users/tangdehua/project/jdjv/silero_vad_clean/
  scripts/run_silero_onnx_float_sharded.py
```

虽然文件名带 `float_sharded`，但它同时支持传入 `--quant-config`。本 shell 一定会传该参数，因此会先进入
`prepare_model()`，而不是直接评测浮点模型。

#### 量化模型准备

`run_silero_onnx_float_sharded.py:134-161` 执行以下过程：

1. 校验浮点 ONNX、量化 JSON 和 custom-op library；
2. 将输出模型路径切换为 `logs/onnx_quant_aishell4_24shards/quantized_model.onnx`；
3. 调用 `normalize_quant_config()`；
4. 调用脚本编写时的 `rewrite_dynamic_qdq(source_model, config_path)`；
5. 用 `onnx.save()` 保存改写后的模型。

原始量化配置是：

```text
Conv2d/Conv3d:
  input  = mxint8, dynamic, axis=1
  weight = mxint8, dynamic, axis=1

Linear:
  input  = mxint8, dynamic
  weight = mxint8, dynamic
```

`normalize_quant_config()` 会把 PyTorch 模块名转换为 ONNX op type：

```text
Conv1d/Conv2d/Conv3d -> Conv
Linear               -> MatMul, Gemm
```

已有的 2026-07-03 量化产物显示，最终模型在 8 kHz 和 16 kHz 两个 `If` 分支中分别改写了 4 个
`Conv`：encoder 的第 1、2、3 个卷积和 decoder 的输出卷积。每个分支有 4 个 activation
`Quantize`、4 个 activation `Dequantize` 和 4 个 weight `Dequantize`，全模型合计：

```text
com.simo::Quantize   8
com.simo::Dequantize 16
```

STFT Conv、encoder 第 0 个 Conv 没有出现在该量化产物的 Q/DQ 集合中；模型内部使用 ONNX `LSTM`，没有
可匹配的 `MatMul/Gemm` 节点。因此“配置目标包含 Conv/MatMul/Gemm”不等于该具体模型中的所有计算都被量化。

#### 分片与并发

`build_tasks()` 创建 `shard_000` 到 `shard_023`，设备按 `index % 8` 分配。`run_task()` 为每个任务设置：

```text
CUDA_VISIBLE_DEVICES=<对应物理 GPU>
```

然后通过 `subprocess.run()` 调用 evaluator。`ThreadPoolExecutor(max_workers=8)` 只负责并发启动和等待子进程；
真正的 ONNX 推理在子进程中完成。AISHELL-4 这里只有 20 个文件，所以 shard 0-19 各处理 1 个文件，
shard 20-23 为空；使用 24 个 shard 并不会把单个长音频切成 24 份。

每完成一个 shard，主进程都会更新 `shard_summary.json`。全部成功后，`aggregate_results()` 按数据集拼接
各 NPZ 中的 `labels` 和 `scores`，重新计算全局指标并写入 `summary.json`。

### 67.4 第二层 Python：单个 shard 的 ONNX VAD 评测

每个 shard 实际调用：

```text
/share_data/users/tangdehua/project/jdjv/silero_vad_clean/
  scripts/evaluate_silero_vad_onnx_float.py
```

它又导入同目录的 `evaluate_silero_vad_public_v5.py`，复用数据集、音频、标签和 ONNX wrapper。

主要流程是：

1. 以 `CUDAExecutionProvider, CPUExecutionProvider` 创建 ONNX Runtime session；
2. 把 SIMO custom-op library 注册到 `SessionOptions`，从而加载量化模型中的 Q/DQ 节点；
3. 遍历 AISHELL-4 文件，只处理满足 `file_index % num_shards == shard_index` 的文件；
4. 加载音频、转单声道，并在需要时重采样到 16 kHz；
5. 对每个文件重置 LSTM state 和上下文，逐窗口运行 Silero VAD；
6. 从 RTTM speech interval 生成同长度的 chunk label；
7. 保存每个 chunk 的布尔标签和浮点 probability 到 shard NPZ；
8. 计算 ROC-AUC 和 `probability >= 0.11` 的 chunk accuracy。

16 kHz 路径每次读取 512 个新采样点，即 32 ms；wrapper 在前面拼接 64 个历史 context sample，因此送入
ONNX 的 `input` 长度是 576。显式 recurrent state 的形状初始化为 `(2, 1, 128)`，每次
`session.run()` 后更新 state 和 64-sample context。末尾不足 512 samples 的音频会补零。

注意：最终 JSON 的 `implementation` 固定写成 `onnxruntime_gpu_float_baseline_sharded`。这是复用浮点
baseline runner 遗留的标签；本 shell 传了 `--quant-config` 后，实际 session 加载的是生成的量化模型，
不能根据这个字段把本次运行误判成浮点评测。

### 67.5 使用的模型

输入模型是：

```text
/share_data/users/tangdehua/project/jdjv/silero_vad_clean/
  onnx_float_baseline/silero_vad.onnx
```

本地核对结果：

```text
模型：   Silero VAD v5
大小：   2,327,524 bytes
SHA256： 2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f
输入：   input(float), state(float), sr(int64)
输出：   output speech probability, stateN
采样率： 支持 8 kHz 和 16 kHz；本脚本默认使用 16 kHz
```

它与项目中的 `weights/silero_vad.onnx` 字节完全相同。配套 TorchScript archive 的根目录名是
`VADr_v5`，项目恢复代码也明确将其描述为 Silero VAD v5，因此这里可以确定是 v5，而不是仅凭 evaluator
文件名推测。模型结构包含 STFT convolution、四层 convolution encoder、128 hidden-size LSTM 和
Conv/Sigmoid speech-probability decoder。

真正参与评测的模型则是它经 W8A8 MXINT8 Q/DQ 改写后的：

```text
logs/onnx_quant_aishell4_24shards/quantized_model.onnx
```

已有产物的 SHA256 为 `611a851551aa7ad26ed3adc85cf5a5ce366856ed096f4d06887ded3a3011b477`。

### 67.6 使用的数据集

默认只使用 **AISHELL-4 test split**，不是 AISHELL-1，也不是多个数据集混跑。下载来源定义为 Hugging Face
仓库 `AISHELL/AISHELL-4`，对应 OpenSLR 111 的公开 test split。本地目录是：

```text
/share/mtang/work/JD/test/silero/data/public_official_v5/
  extracted/aishell4_hf/test/
    wav/*.flac
    TextGrid/*.rttm
    TextGrid/*.TextGrid
```

本地数据规模为：

```text
音频文件：20 个 FLAC
RTTM：    20 个
TextGrid：20 个
总时长：  12.7253036 小时
```

`iter_aishell4()` 优先读取同名 RTTM，把所有 speaker interval 合并成 VAD speech union；没有 RTTM 时才
退回 TextGrid 的所有 tier union。音频若为多声道会先按通道求平均，随后重采样到 16 kHz。一个 512-sample
chunk 只要与 speech interval 有重叠，就被标成 speech。该规则生成了 1,431,607 个 chunk 标签。

### 67.7 已有运行结果说明

项目中保留了一次 2026-07-03 的完整运行结果：

| 模型 | 文件 | 小时 | chunks | ROC-AUC | Acc@0.11 | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 浮点 ONNX | 20 | 12.7253 | 1,431,607 | 0.947480 | 0.835417 | 0 |
| W8A8 MXINT8 Q/DQ ONNX | 20 | 12.7253 | 1,431,607 | 0.947213 | 0.834620 | 0 |

这些是已有日志中的历史结果，不是本次重新执行全量 12.7 小时数据得到的结果。它们也说明该脚本关注的是
量化后 VAD 精度；量化模型相对浮点模型的 ROC-AUC 和 accuracy 只发生了小幅下降。

### 67.8 当前指定 conda 环境下的可运行性

截至 2026-07-27，在用户指定的：

```text
/share_data/users/like/miniconda3/envs/simo_sglang/
```

中，`simo` 实际解析到：

```text
/share/users/like/package/simo_conda_sglang/simo/__init__.py
```

按当前状态直接运行这个 shell 会遇到两个阻塞：

1. 默认的 `/tmp/simo_ort_plugin_build/libSimoOnnxCustomOps.so` 和
   `/tmp/simo_mxint8_cubins/simo_mxint8_sm90.manifest.tsv` 当前均不存在，shell 会先在
   `test_quant_onnx.sh:22-23` 退出；
2. 当前 SIMO 已没有 `simo.onnx.onnx_dynamic_qdq`，所以旧导入
   `from simo.onnx.onnx_dynamic_qdq import rewrite_dynamic_qdq` 无法成功。当前公开入口是
   `simo.onnx.quantize()`，底层为 `simo.onnx.onnx_quant.apply_qdq_quantization()`；而且当前 API 明确拒绝
   旧的 `targets_op_types` 字段，所以不能只改 import，还必须同步移除脚本的旧配置归一化方式。

已有 2026-07-03 日志使用的是
`/share_data/users/tangdehua/miniconda3/envs/vllm/bin/python`，且当时 `/tmp` 下的 custom-op 和 manifest
仍然存在。因此，历史日志证明这套旧调用链曾经跑通，但不代表它在当前指定 conda 环境中仍可原样运行。

综上，这个脚本的目标可以概括为：

```text
  Silero VAD v5 浮点 ONNX
  -> SIMO W8A8 MXINT8 动态 Q/DQ 图改写
  -> ONNX Runtime CUDA + SIMO custom ops
  -> AISHELL-4 test 全量、多 GPU、24-shard 评测
  -> ROC-AUC / Acc@0.11 汇总
```

## 68. 当前 editable SIMO 的 custom-op `.so` 与 `SIMO_ONNX_TRITON_MANIFEST`

本节针对用户指定的环境和文件核对：

```text
Python 环境：/share_data/users/like/miniconda3/envs/simo_sglang/
源码：      /share/users/like/package/simo_conda_sglang
目标库：    /share_data/users/like/package/simo_conda_sglang/
            simo/onnx/ort_plugin/libSimoOnnxCustomOps_sm90.so
```

### 68.1 直接结论

| 问题 | 结论 |
| --- | --- |
| 这个 `.so` 能否传给 `CUSTOM_OP_LIBRARY` | **可以**。它是当前 editable SIMO 生成的 sm90 ONNX Runtime custom-op library，路径可直接作为旧 shell 的 `CUSTOM_OP_LIBRARY` 值，也可传给 `SessionOptions.register_custom_ops_library()`。 |
| 当前 SIMO 是否需要 `SIMO_ONNX_TRITON_MANIFEST` | **不需要**。当前 v2 构建把 Triton cubin 和 kernel resolver 嵌入 `.so`，运行时不查找 manifest、外部 cubin 或 Triton cache。 |
| 当前版本怎样构建 manifest | **没有这个构建步骤**。`build_qdq_cubins.py` 生成的是临时的 `embedded_qdq_kernels_sm90.cc`，随后与 C++ custom-op 源码一起链接进 `.so`；它有意不生成 `*.manifest.tsv`。 |
| 旧 `test_quant_onnx.sh` 中的 manifest 检查是否代表 SIMO 的真实要求 | 不代表。那是旧脚本自己的 `test -f` 和 `export`，不是当前 plugin 的运行时 ABI 要求。 |

### 68.2 `.so` 的实际验证

目标文件已存在，大小为 `6,185,008` bytes，SHA256 为：

```text
1e9ceb9ae10a2babce7ea63124dfbb4de050547af1da9840e46a99259ae2eb79
```

它是 x86-64 ELF shared object，动态依赖均可解析，并导出 ONNX Runtime 要求的：

```text
RegisterCustomOps@@VERS_1.0.0
```

当前机器上的 GPU 是 H100，compute capability 为 `9.0`，所以库名中的 `sm90` 与硬件匹配。
`/share_data/users/like/package/...` 和 editable 源码显示的 `/share/users/like/package/...` 是同一设备、
同一 inode 的两个挂载路径；因此 Python 返回的另一种路径前缀不是另一份库。

在指定环境中实际执行下面的注册调用成功：

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.register_custom_ops_library(
    "/share_data/users/like/package/simo_conda_sglang/"
    "simo/onnx/ort_plugin/libSimoOnnxCustomOps_sm90.so"
)
```

因此旧 shell 的变量可以这样设置：

```bash
export CUSTOM_OP_LIBRARY=/share_data/users/like/package/simo_conda_sglang/simo/onnx/ort_plugin/libSimoOnnxCustomOps_sm90.so
```

需要区分两个变量的作用：`CUSTOM_OP_LIBRARY` 是 `test_quant_onnx.sh` 自己定义并通过
`--custom-op-library` 传给 Python 的变量；SIMO 公共 runtime 直接读取的覆盖变量名称是
`SIMO_ONNX_CUSTOM_OPS_LIBRARY`：

```bash
export SIMO_ONNX_CUSTOM_OPS_LIBRARY=/share_data/users/like/package/simo_conda_sglang/simo/onnx/ort_plugin/libSimoOnnxCustomOps_sm90.so
```

也可以显式使用当前 Python API：

```python
import onnxruntime as ort
import simo.onnx as sx

options = ort.SessionOptions()
sx.register_custom_ops(
    options,
    library_path="/share_data/users/like/package/simo_conda_sglang/"
    "simo/onnx/ort_plugin/libSimoOnnxCustomOps_sm90.so",
)
```

### 68.3 重要的版本兼容性：能注册不等于能加载任意旧模型

该库的注册入口可用，但它对应当前 SIMO ONNX QDQ **v2**。源码中的
`simo/onnx/onnx_quant.py` 将 domain opset 设为 `com.simo` version `2`，当前 custom-op C++ 注册的
`Quantize/Dequantize` 也只使用 version `2`。

已有的旧 Silero 量化模型（2026-07-03 产物）导入的是：

```text
('', 16), ('com.simo', 1)
```

用当前 `.so` 加载它时，ORT 报：

```text
TypeInferenceError: com.simo:Dequantize(-1) is not a registered function/op
```

这说明以下两件事要分开看：

```text
CUSTOM_OP_LIBRARY 路径正确       -> ORT 能找到并注册 .so
旧 v1 ONNX 图 + 当前 v2 .so      -> 版本不匹配，仍然不能运行
```

当前旧量化配置还存在两个迁移问题：

1. `/share_data/tangdehua/.../w_mxint8_a_mxint8.json` 中的 weight `is_dynamic` 为 `true`，而当前
   QDQ v2 要求 weight `is_dynamic` 必须为 `false`；
2. 当前 API 拒绝 `targets_op_types`，统一使用 `targets`。旧脚本的配置归一化代码会生成这个已废弃字段。

所以，仅把 `CUSTOM_OP_LIBRARY` 改成上述 `.so`，并不能使旧版
`from simo.onnx.onnx_dynamic_qdq import rewrite_dynamic_qdq` 调用链恢复。正确迁移方向是使用当前
`simo.onnx.quantize()` 重新生成 v2 ONNX 图。例如，配置迁移的最小原则是：

```python
for module in config["module_configs"]:
    module["weight"]["is_dynamic"] = False
    module.pop("targets_op_types", None)
# 目标类型放在 module["targets"] 中
```

用这个原则生成的图包含 `('com.simo', 2)`，再注册上述 `.so` 后，已在当前 H100 上成功创建
CUDA/CPU ORT session 并完成一次推理。也就是说，当前库和当前 v2 量化器是可配套工作的；旧 v1 图需要
重新量化，不能靠 manifest 或改环境变量修复。

### 68.4 当前版本的 Triton kernel 构建链

当前代码的实际链路是：

```text
build_sm90_runtime(output_path)
  -> 临时目录中调用 build_qdq_cubins(build_dir)
  -> Triton AOT 编译 sm90 kernel
  -> 生成 embedded_qdq_kernels_sm90.cc
  -> 与 custom_op_library.cc、simo_qdq_ops.cc、
     simo_qdq_cpu_ops.cc、triton_loader.cc 一起用 C++17 链接
  -> 输出 libSimoOnnxCustomOps_sm90.so
  -> 删除临时目录
```

生成的 C++ 文件中保存 cubin 字节数组、符号名、shared-memory/grid 元数据和精确的 runtime resolver。
`triton_loader.cc` 运行时对内存中的 cubin 数据调用 CUDA Driver API 的 `cuModuleLoadData`，而不是按路径
打开 `*.cubin` 或 `*.manifest.tsv`。

仓库测试也把这个契约写死了：`test_qdq_cubin_build.py` 断言输出为
`embedded_qdq_kernels_sm90.cc`，并断言临时目录中没有任何 `*.manifest.tsv` 和 `*.cubin`。README 的
v2 scope 也明确写着：runtime packaging 不包含 manifest 或 external cubin paths。

### 68.5 如果需要重建当前 `.so`

editable 安装的标准做法是在源码根目录使用指定解释器重新安装：

```bash
cd /share/users/like/package/simo_conda_sglang
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python -m pip install -e . --no-build-isolation
```

`setup.py` 的 editable/inplace 分支会把结果写回：

```text
simo/onnx/ort_plugin/libSimoOnnxCustomOps_sm90.so
```

只想调用 ONNX runtime builder 时也可以：

```bash
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
from simo.onnx.ort_plugin.build_runtime import build_sm90_runtime

print(build_sm90_runtime(
    "/share_data/users/like/package/simo_conda_sglang/"
    "simo/onnx/ort_plugin/libSimoOnnxCustomOps_sm90.so"
))
PY
```

构建机需要 C++17 编译器、CUDA headers、Triton，以及可链接的 CUDA driver library；目标架构固定为
sm90。这个过程仍然不会产生 manifest。当前目标文件已经存在且验证通过，没有必要为了设置
`SIMO_ONNX_TRITON_MANIFEST` 再重复构建。

### 68.6 旧 shell 应怎样修改

`test_quant_onnx.sh` 目前有：

```bash
test -f "$SIMO_ONNX_TRITON_MANIFEST"
export SIMO_ONNX_TRITON_MANIFEST
```

这两行以及变量默认值属于旧调用链。迁移到当前 SIMO 时应删除 manifest 变量、文件检查和 export，只保留
custom-op library 检查，并把库路径传给当前量化/评测 Python 代码。例如 shell 层可以保留：

```bash
CUSTOM_OP_LIBRARY=/share_data/users/like/package/simo_conda_sglang/simo/onnx/ort_plugin/libSimoOnnxCustomOps_sm90.so
test -f "$CUSTOM_OP_LIBRARY"
```

然后在 Python 中改用 `simo.onnx.quantize()` 生成 v2 图，并通过
`register_custom_ops(options, library_path=...)` 注册该库。不能只创建一个空的
`simo_mxint8_sm90.manifest.tsv` 来绕过 shell 检查，因为当前旧 Python import、旧 v1 图和 v2 plugin
仍然不兼容；空文件也不会提供任何 kernel 元数据。

如果必须继续运行某个真正依赖 manifest 的历史 plugin，则必须同时找回与该 plugin、ONNX
`com.simo` opset 版本和 Triton kernel ABI 匹配的旧源码及生成器。当前 checkout 中没有可调用的
manifest 生成器，不能用当前 `build_qdq_cubins.py` 生成一个可供旧 plugin 使用的 manifest。

### 68.7 最终判断

```text
当前 .so：       可以直接作为 CUSTOM_OP_LIBRARY
当前 SIMO API：  可用 SIMO_ONNX_CUSTOM_OPS_LIBRARY 覆盖路径
manifest：       当前 v2 不需要，也没有构建步骤
旧 shell：       需删除 manifest 硬检查，并迁移旧 quantize/import/config
旧 v1 ONNX：     不能直接配当前 v2 .so，必须重新生成
```

---
 
## 69. Silero VAD 浮点与量化 ONNX 的 AISHELL-4 评测对比
 
### 69.1 结论
 
有模型评价指标，而且两个 24-shard 运行都成功完成。结果文件分别是：
 
```text
浮点: /share/users/like/package/jdjv/silero_vad_clean/logs/onnx_float_baseline_aishell4_24shards/summary.json
量化: /share/users/like/package/jdjv/silero_vad_clean/logs/onnx_quant_aishell4_24shards/summary.json
```
 
两次运行使用相同的 AISHELL-4 数据、16 kHz、24 个 shard、阈值 0.11。逐 shard 合并后的标签数组完全一致，都是 20 个音频文件、12.725304 小时、1,431,607 个 chunk，且两次 errors 都为 0。因此下面的差异是模型输出差异，而不是数据或分片数量差异。
 
| 模型 | ROC-AUC | Accuracy@0.11 | 文件 | 小时 | chunks | errors |
|---|---:|---:|---:|---:|---:|---:|
| 浮点 ONNX | 0.947480 (94.7480%) | 0.835417 (83.5417%) | 20 | 12.725304 | 1,431,607 | 0 |
| 量化 ONNX | 0.946729 (94.6729%) | 0.828048 (82.8048%) | 20 | 12.725304 | 1,431,607 | 0 |
 
简要判断：量化后 ROC-AUC 只下降 0.000751，即 0.0751 个百分点，排序能力几乎保持；固定阈值 0.11 下 Accuracy 下降 0.007369，即 0.7369 个百分点，属于可见但不大的损失。量化模型仍然可以正常执行 CUDA 和 com.simo Q/DQ custom ops。
 
### 69.2 评测设置和产物核对
 
- 浮点模型 SHA256：2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f
- 量化模型 SHA256：ace6f9fdd081f014274d422d2e4c560d52219559a51029258a049aee37e7dbe2
- 两次日志都显示 CUDAExecutionProvider 已激活。
- 量化日志显式注册了 /share_data/users/like/package/simo_conda_sglang/simo/onnx/ort_plugin/libSimoOnnxCustomOps_sm90.so。
- 当前量化模型导入 com.simo opset 2，递归检查得到 12 个 com.simo::Quantize 和 24 个 com.simo::Dequantize 节点。
- 只有 20 个 AISHELL-4 文件，所以 shard 20 到 23 没有文件是正常的空 shard；它们状态仍为 done，不是失败。
 
量化目录里的 implementation 和 evaluator 输出中的 Backend: ONNXRuntime float baseline 是复用评测器留下的通用文案，不表示它加载了浮点模型。应以日志中的 ONNX model: 路径和 custom-op library 路径为准；量化运行实际加载的是 quantized_model.onnx。
 
### 69.3 指标含义
 
#### ROC-AUC
 
这是不依赖某一个分类阈值的排序指标。脚本把每个 512-sample 窗口（16 kHz 下约 32 ms）的模型输出分数与参考语音片段重叠生成的二值标签进行比较，再计算 ROC 曲线下面积。
 
直观地说，浮点模型 ROC-AUC=0.94748 表示随机抽取一个真实语音 chunk 和一个非语音 chunk 时，模型把语音 chunk 排在更高分的概率约为 94.75%（忽略并列分数的细节）。它不是“94.75% 的分类准确率”，也不对应某个固定阈值。
 
量化模型的 0.946729 与浮点模型非常接近，说明量化主要保留了语音/非语音的相对排序；如果应用会重新选择阈值，AUC 通常比单个固定阈值的 Accuracy 更能反映模型本身的区分能力。
 
#### Accuracy@0.11
 
这是阈值相关的 chunk-level 准确率，脚本的计算式是：
 
```text
prediction = (model_score >= 0.11)
Accuracy = 正确预测的 chunk 数 / 全部 chunk 数
```
 
浮点模型在该阈值下正确率为 83.5417%，量化模型为 82.8048%。这个值会受到阈值和语音/非语音类别比例影响，不能直接等同于 F1、precision、recall、DER，也不是按文件取平均的准确率；长音频包含的 chunk 更多，会对总体值贡献更多。
 
#### errors
 
errors=0 表示音频读取、ONNX 推理和结果写入过程中没有异常。它不是预测错误数，也不能替代 Accuracy 或 AUC。

summary.json 原生保存的预测指标是 roc_auc 和 accuracy；precision、recall、F1 不会直接写入 summary，本节 69.4 的混淆矩阵和派生指标是从 shard NPZ 的 labels/scores 按同一阈值计算得到的。

日志中的 official_roc_auc=0.94 和 official_accuracy=0.85 是代码内置的 AISHELL-4 参考值。尤其官方 Accuracy 使用了私有验证集选择的阈值，而本次脚本固定使用 0.11，所以这些数值只能作为方向性参考，不能当作完全同协议的严格复现基准。
 
### 69.4 量化损失的具体表现
 
在相同 1,431,607 个 chunk 上比较 NPZ：
 
| 指标 | 浮点 | 量化 | 量化 - 浮点 |
|---|---:|---:|---:|
| ROC-AUC | 0.947480001 | 0.946729208 | -0.000750793 (-0.0751 pp) |
| Accuracy@0.11 | 0.835417122 | 0.828047781 | -0.007369341 (-0.7369 pp) |
| score 平均值 | 0.621341 | 0.611784 | -0.009557 |
| 浮点/量化 score 相关系数 | - | 0.999094 | - |
| 阈值判定不同的 chunk | - | 11,966 / 1,431,607 (0.8358%) | - |
 
按阈值 0.11 展开混淆矩阵后：
 
```text
                 TP         TN       FP       FN       Recall      Precision
浮点         1,073,279   122,710    7,185   228,433    82.4513%     99.3350%
量化         1,062,251   123,188    6,707   239,461    81.6041%     99.3726%
```
 
量化后 FP 减少 478 个，但 FN 增加 11,028 个，所以固定阈值下整体 Accuracy 下降；这也说明量化分数整体略向下，模型变得稍微更保守。Precision 略升、Recall 下降，是这次 Accuracy 损失的主要形态，而不是所有类别都同等变差。
 
### 69.5 使用建议
 
如果目标是保持当前阈值和召回率，量化模型需要针对部署数据重新做 threshold sweep，并重点观察 Recall、F1 或业务真正关心的漏检率；不能只看 AUC。若目标是压缩/加速且允许不到 1 个百分点的固定阈值 Accuracy 损失，那么本次结果显示该量化模型与浮点基线相当接近，可以作为候选版本继续做更大范围数据和真实业务阈值验证。

---

## 70. `test_quant_onnx.sh` 在哪里保存和加载量化 ONNX

### 70.1 直接结论

`test_quant_onnx.sh` 本身既不调用 `onnx.save()`，也不调用 ONNX Runtime。它负责确定文件路径并启动 Python；真正的保存和加载分别发生在 SIMO 与评测器中：

| 动作 | 文件名 + 行号 + 函数名 | 关键代码 |
|---|---|---|
| 指定量化模型输出路径 | `test_quant_onnx.sh:37`，脚本顶层主命令（该 shell 没有定义函数） | `--quantized-model "$OUTPUT_DIR/quantized_model.onnx"` |
| 调用 SIMO 量化并传入输出路径 | `scripts/run_silero_onnx_float_sharded.py:160`，`prepare_model()` | `simo.onnx.quantize(..., output_path=args.quantized_model)` |
| **真正把量化模型写入磁盘** | `simo/onnx/api.py:24`，`quantize()` | `onnx.save(rewritten, output_path)` |
| 将量化模型交给分片 evaluator | `scripts/run_silero_onnx_float_sharded.py:96-97`，`command_for_task()` | `--model` 后面传入 `str(args.model)` |
| evaluator 创建模型封装 | `scripts/evaluate_silero_vad_onnx_float.py:212-219`，`main()` | `public_v5.SileroVadOnnx(args.model, ...)` |
| **真正由 ORT 加载量化 ONNX** | `scripts/evaluate_silero_vad_public_v5.py:105-109`，`SileroVadOnnx.__init__()` | `ort.InferenceSession(str(model_path), ...)` |

所以最短答案是：

```text
实际保存：simo/onnx/api.py:24，quantize()
实际加载：scripts/evaluate_silero_vad_public_v5.py:105-109，SileroVadOnnx.__init__()
```

### 70.2 保存调用链

#### 第一步：shell 决定输出文件名

`test_quant_onnx.sh:19`，脚本顶层主命令（无函数）定义默认输出目录：

```bash
OUTPUT_DIR=${OUTPUT_DIR:-$SCRIPT_DIR/logs/onnx_quant_${DATASETS}_${NUM_SHARDS}shards}
```

`test_quant_onnx.sh:33-44`，脚本顶层主命令启动分片 runner；其中 `test_quant_onnx.sh:37` 传入：

```bash
--quantized-model "$OUTPUT_DIR/quantized_model.onnx"
```

在默认 `DATASETS=aishell4`、`NUM_SHARDS=24` 时，目标文件是：

```text
/share/users/like/package/jdjv/silero_vad_clean/logs/onnx_quant_aishell4_24shards/quantized_model.onnx
```

这里仅仅是把路径放进命令行参数，还没有写 ONNX 文件。

#### 第二步：runner 准备并量化模型

`scripts/run_silero_onnx_float_sharded.py:460`，`main()` 是 runner 入口；`scripts/run_silero_onnx_float_sharded.py:465`，`main()` 调用：

```python
prepare_model(args)
```

`scripts/run_silero_onnx_float_sharded.py:135-160`，`prepare_model()` 完成量化准备。关键行是：

- `scripts/run_silero_onnx_float_sharded.py:151`，`prepare_model()`：若 shell 没传 `--quantized-model`，则回退到 `args.output_dir / "quantized_model.onnx"`。
- `scripts/run_silero_onnx_float_sharded.py:152`，`prepare_model()`：执行 `args.model = args.quantized_model`，把后续评测模型从原始浮点模型切换为量化模型。
- `scripts/run_silero_onnx_float_sharded.py:159`，`prepare_model()`：创建量化模型的父目录。
- `scripts/run_silero_onnx_float_sharded.py:160`，`prepare_model()`：调用 editable 安装的 SIMO 公共接口，并通过 `output_path` 传入目标路径。

```python
simo.onnx.quantize(source_model, config_path, output_path=args.quantized_model)
```

#### 第三步：SIMO 真正序列化并保存文件

当前 conda 环境中的 editable import 已解析到源码目录：

```text
/share/users/like/package/simo_conda_sglang/simo/onnx/api.py
```

`simo/onnx/api.py:15-25`，`quantize()` 的逻辑是：

```python
rewritten = apply_qdq_quantization(model, config, simplify=simplify)
if output_path is not None:
  onnx.save(rewritten, output_path)
return rewritten
```

其中 `simo/onnx/api.py:22`，`quantize()` 生成插入 Q/DQ 节点后的 `ModelProto`；`simo/onnx/api.py:24`，`quantize()` 的 `onnx.save(...)` 才是实际写出 `.onnx` 文件的代码。由于 runner 在第 160 行明确传了 `output_path`，保存分支一定会执行；保存失败也会直接抛出异常，在开始 shard 评测前终止。

### 70.3 加载调用链

#### 第一步：把量化路径变成评测模型路径

`scripts/run_silero_onnx_float_sharded.py:152`，`prepare_model()` 执行：

```python
args.model = args.quantized_model
```

这是保存链和加载链的连接点。没有这次赋值，后续 `command_for_task()` 仍会把原始浮点 `--model` 传给 evaluator。

#### 第二步：为每个 shard 传递量化模型

`scripts/run_silero_onnx_float_sharded.py:92-132`，`command_for_task()` 构造 evaluator 命令；其中第 96-97 行是：

```python
"--model",
str(args.model),
```

此时 `args.model` 已在 `prepare_model()` 中改为 `quantized_model.onnx`。

`scripts/run_silero_onnx_float_sharded.py:223-264`，`run_task()` 负责运行一个 shard；第 230 行调用 `command_for_task()`，第 244-251 行通过 `subprocess.run(...)` 启动 `evaluate_silero_vad_onnx_float.py` 子进程。

#### 第三步：evaluator 把路径传给 ONNX Runtime 封装

`scripts/evaluate_silero_vad_onnx_float.py:200-250`，`main()` 是单 shard evaluator 的入口：

- `scripts/evaluate_silero_vad_onnx_float.py:202`，`main()` 先确认 `args.model` 文件存在。
- `scripts/evaluate_silero_vad_onnx_float.py:212-219`，`main()` 用 `args.model` 创建 `public_v5.SileroVadOnnx`。
- `scripts/evaluate_silero_vad_onnx_float.py:216`，`main()` 同时传入 custom-op library，因为量化图含有 `com.simo` Q/DQ 节点。

#### 第四步：ORT 真正打开并加载 ONNX

`scripts/evaluate_silero_vad_public_v5.py:75-125`，`SileroVadOnnx.__init__()` 初始化 ONNX Runtime：

- `scripts/evaluate_silero_vad_public_v5.py:88`，`SileroVadOnnx.__init__()` 创建 `ort.SessionOptions()`。
- `scripts/evaluate_silero_vad_public_v5.py:96`，`SileroVadOnnx.__init__()` 先通过 `register_custom_ops_library(...)` 注册 SIMO custom-op 动态库。
- `scripts/evaluate_silero_vad_public_v5.py:105-109`，`SileroVadOnnx.__init__()` 调用 `ort.InferenceSession(str(model_path), ...)`；这一步才是真正读取、解析并加载量化 ONNX。

注册 custom-op library 必须发生在 `InferenceSession` 创建之前，否则 ORT 在解析量化图中的 `com.simo::Quantize` / `com.simo::Dequantize` 节点时无法找到对应实现。

### 70.4 完整时序和运行次数

```text
test_quant_onnx.sh:37（脚本顶层主命令）
  --quantized-model .../quantized_model.onnx
    -> run_silero_onnx_float_sharded.py:465，main()
       调用 prepare_model(args)
         -> run_silero_onnx_float_sharded.py:160，prepare_model()
            调用 simo.onnx.quantize(..., output_path=...)
              -> simo/onnx/api.py:24，quantize()
                 onnx.save(...)，量化模型保存一次
         -> run_silero_onnx_float_sharded.py:152，prepare_model()
            args.model 切换为量化模型路径
    -> run_silero_onnx_float_sharded.py:96-97，command_for_task()
       每个 shard 的 --model 都使用量化模型路径
      -> evaluate_silero_vad_onnx_float.py:212-219，main()
         创建 SileroVadOnnx(args.model, ...)
        -> evaluate_silero_vad_public_v5.py:105-109，SileroVadOnnx.__init__()
           ort.InferenceSession(...)，每个 shard 子进程各加载一次
```

量化发生在 `main()` 创建分片任务之前，因此整个 `test_quant_onnx.sh` 运行只生成并保存一次量化 ONNX。之后每个 shard 是独立 Python 子进程，每个子进程都会建立自己的 ORT Session，并各自加载同一个 `quantized_model.onnx`。默认 24 shards 时会创建 24 个独立 Session，但 `WORKERS=8` 表示最多同时运行 8 个 shard。

现有运行结果也验证了这条链路：量化文件确实位于上述默认路径，大小为 4,225,143 bytes；`logs/onnx_quant_aishell4_24shards/summary.json:3` 的 `model` 字段以及各个 `shards/shard_*.log:2,5` 都记录了同一个 `quantized_model.onnx` 路径。

---

## 71. `run_silero_onnx_float_sharded.py` 的 git diff 主要修改

### 71.1 修改范围和总体目的

对 `/share/users/like/package/jdjv/silero_vad_clean/scripts/run_silero_onnx_float_sharded.py` 执行 `git diff` 的结果是 **55 行新增、28 行删除**，修改集中在量化准备和量化配置迁移；数据分片、子进程调度和指标汇总没有被重写。

总体目的，是把旧版 SIMO ONNX 动态 QDQ 调用链迁移到当前 editable SIMO 提供的 ONNX QDQ v2 公共接口，同时保持原有命令行和评测流程：

```text
浮点 ONNX + 外部旧格式配置
  -> 生成独立的 QDQ v2 配置
  -> 当前 simo.onnx.quantize()
  -> quantized_model.onnx
  -> 原有 shard evaluator / CUDA / custom-op 流程
```

### 71.2 `prepare_model()`：替换旧量化 API

函数位置：`scripts/run_silero_onnx_float_sharded.py:135-160`，函数名 `prepare_model()`。

保留的前置检查仍在 `scripts/run_silero_onnx_float_sharded.py:139-149`，函数名 `prepare_model()`：检查浮点源模型、量化配置和 custom-op library，并允许通过 `SIMO_ONNX_CUSTOM_OPS_LIBRARY` 环境变量补充库路径。

关键变化如下：

- 旧代码 `scripts/run_silero_onnx_float_sharded.py:155-161`（旧版本，函数 `prepare_model()`）导入 `onnx` 和已不存在的 `simo.onnx.onnx_dynamic_qdq.rewrite_dynamic_qdq`，先改写出 `ModelProto`，再由 runner 自己调用 `onnx.save()`。
- 新代码 `scripts/run_silero_onnx_float_sharded.py:156`（函数 `prepare_model()`）改为延迟导入 `simo.onnx`。只有传入 `--quant-config` 时才需要 SIMO，纯浮点评测路径仍然在 `scripts/run_silero_onnx_float_sharded.py:136-137`（函数 `prepare_model()`）直接返回。
- `scripts/run_silero_onnx_float_sharded.py:151-152`（函数 `prepare_model()`）继续确定 `quantized_model.onnx` 路径，并把 `args.model` 切换到该路径，保证后续 shard evaluator 使用量化模型。
- `scripts/run_silero_onnx_float_sharded.py:158`（函数 `prepare_model()`）先调用新的配置迁移函数；`scripts/run_silero_onnx_float_sharded.py:159`（函数 `prepare_model()`）创建输出父目录。
- **真正的量化调用**现在是 `scripts/run_silero_onnx_float_sharded.py:160`（函数 `prepare_model()`）：

  ```python
  simo.onnx.quantize(source_model, config_path, output_path=args.quantized_model)
  ```

  这把量化实现交给当前 SIMO 公共 API，并明确让 API 将结果写到 `args.quantized_model`；不再依赖旧的 `rewrite_dynamic_qdq`。

### 71.3 `normalize_quant_config()`：迁移到 QDQ v2 配置

函数位置：`scripts/run_silero_onnx_float_sharded.py:163-220`，函数名 `normalize_quant_config()`。

这是本次 diff 中新增逻辑最多的部分。它不覆盖外部原始 JSON，而是生成独立文件 `$OUTPUT_DIR/onnx_quant_config.json`。

#### 输入结构校验和外层解包

- `scripts/run_silero_onnx_float_sharded.py:164-166`（函数 `normalize_quant_config()`）读取 JSON，并要求顶层是对象。
- `scripts/run_silero_onnx_float_sharded.py:167-169`（函数 `normalize_quant_config()`）兼容可选的 `quantization_config` 外层；迁移后的文件使用内部配置对象作为根，不再保留这个旧包装层。
- `scripts/run_silero_onnx_float_sharded.py:171-173`（函数 `normalize_quant_config()`）要求 `module_configs` 是非空列表，避免把空配置交给 SIMO 后才得到不明确的错误。

#### 构造只包含当前 API 支持内容的配置

- `scripts/run_silero_onnx_float_sharded.py:175-180`（函数 `normalize_quant_config()`）新建 `migrated` 配置，只保留 `module_configs`，并在格式合适时复制 `excludes` 和 `algorithm`。
- `scripts/run_silero_onnx_float_sharded.py:182-194`（函数 `normalize_quant_config()`）定义允许从 input/weight spec 复制的字段白名单，例如 `dtype`、`block_size`、`group_size`、`is_symmetric` 和 `scale_mode`；旧 API 不认识或当前 QDQ v2 不需要的字段不会继续传递。
- 新增的模块级 `import copy` 位于 `scripts/run_silero_onnx_float_sharded.py:7`，用于下面各处的深拷贝，避免迁移过程复用或改写输入对象。

#### 保留 `targets`，删除旧的 `targets_op_types` 路径

- `scripts/run_silero_onnx_float_sharded.py:195-202`（函数 `normalize_quant_config()`）逐个校验 module，并保留 `targets` 列表。
- 新配置不再生成 `targets_op_types`。旧版本的 `module_targets_to_onnx_ops()`（旧文件 `scripts/run_silero_onnx_float_sharded.py:181-193`，函数名 `module_targets_to_onnx_ops()`）曾把 `Conv1d`/`Linear` 等模块名手工映射为 ONNX `Conv`/`MatMul`/`Gemm`；这个 helper 已删除，因为当前 SIMO QDQ v2 入口直接接受 `targets`，并由 SIMO 自己处理目标别名。
- 因此，`targets` 的语义仍然保留，但旧的“同时传模块目标和 ONNX op 类型”双字段不再进入当前 API。

#### 将 activation/input 统一，并强制动态性符合 v2 约束

- `scripts/run_silero_onnx_float_sharded.py:203-204`（函数 `normalize_quant_config()`）优先读取 `input`，并兼容旧配置中的 `activation` 键，把它迁移为 v2 的 `input`。
- `scripts/run_silero_onnx_float_sharded.py:204-215`（函数 `normalize_quant_config()`）分别处理 input 和 weight spec，只复制白名单字段。
- `scripts/run_silero_onnx_float_sharded.py:214`（函数 `normalize_quant_config()`）强制 `input.is_dynamic = True`，因为运行时输入的量化参数需要动态计算。
- 同一行逻辑对 weight 使用 `field_name == "input"` 的结果为 `False`，因此强制 `weight.is_dynamic = False`，符合当前 SIMO ONNX QDQ v2 的静态权重量化约束；即使旧 JSON 错误地写成 `true`，迁移后的配置也会纠正它。
- 旧配置中的 `output` spec 不再复制到迁移结果；当前 ONNX QDQ v2 迁移只为 evaluator 所需的 input/weight QDQ 生成必要字段。

#### 独立输出配置，不修改原始配置

- `scripts/run_silero_onnx_float_sharded.py:218-220`（函数 `normalize_quant_config()`）固定把迁移结果写到 `output_dir / "onnx_quant_config.json"`，并返回这个新路径。
- 与旧版本 `scripts/run_silero_onnx_float_sharded.py:167-178`（旧函数 `normalize_quant_config()`）相比，新实现不再“没有变化就返回外部原配置”，也不在原始对象上追加 `targets_op_types`；每次量化都使用明确、可审计的副本配置。

### 71.4 哪些量化评测逻辑没有变化

本次 diff 没有改变以下职责：

- `scripts/run_silero_onnx_float_sharded.py:92-132`（函数 `command_for_task()`）仍只向每个 evaluator 传 `--model`、数据集/分片参数和 `--custom-op-library`；没有重新引入旧 manifest 参数。
- `scripts/run_silero_onnx_float_sharded.py:223-264`（函数 `run_task()`）仍按设备设置 `CUDA_VISIBLE_DEVICES`、启动 shard 子进程并读取 NPZ 结果。
- `scripts/run_silero_onnx_float_sharded.py:283-310`（函数 `split_npz_by_dataset()`）、`scripts/run_silero_onnx_float_sharded.py:313-341`（函数 `summarize_arrays()`）和 `scripts/run_silero_onnx_float_sharded.py:344-400`（函数 `aggregate_results()`）仍负责原来的分片合并、ROC-AUC 和阈值准确率计算。
- `scripts/run_silero_onnx_float_sharded.py:460-523`（函数 `main()`）仍先准备模型，再运行 shards，最后写 `run_metadata.json`、`shard_summary.json` 和 `summary.json`；量化迁移没有改变采样率、阈值、state 输入或指标定义。

### 71.5 修改前后对照

| 项目 | diff 之前 | diff 之后 |
|---|---|---|
| 量化入口 | `rewrite_dynamic_qdq`（旧内部 API） | `simo.onnx.quantize`（当前公开 API） |
| ONNX 写盘位置 | runner 中直接 `onnx.save` | SIMO `quantize(..., output_path=...)` 内部保存 |
| 配置目标字段 | 手工生成 `targets_op_types` | 保留 `targets`，由 SIMO 处理别名 |
| activation 字段 | 保留旧 `activation` 形态 | 迁移为 `input` |
| weight 动态性 | 可能沿用旧 JSON 的 `true` | 强制 `weight.is_dynamic=false` |
| 配置文件 | 可能直接返回原配置路径 | 始终输出 `$OUTPUT_DIR/onnx_quant_config.json` |
| 分片和指标 | 原有流程 | 保持不变 |

因此，这个 git diff 的本质不是更换评测算法，而是修复量化接口和配置 schema 的版本兼容性：让当前 editable SIMO 能够生成 `com.simo` QDQ v2 ONNX，同时尽量不改变原有 Silero AISHELL-4 分片评测行为。

---

## 72. main 分支新提交 `5b7571c` 与 `7c17fd9` 分别实现了什么

### 72.1 总览

这两个提交解决的是两类不同的问题：

| 提交 | 主要模块 | 核心功能 |
|---|---|---|
| `5b7571c1a03f1d4dcac2df65a7bead6fb5a69aa8` | `simo.accuracy_debug` / ONNX | 将 ONNX 精度调试从“只抓标准域 MatMul/Gemm/Conv 的输出”扩展为“可选择算子、domain、输入/输出及 initializer 的通用张量抓取和比较” |
| `7c17fd98217132583a97a0891526cacce3cdd767` | vLLM SIMO 量化线性层 / Gluon GEMM | 为 Hopper 增加 MXFP4×FP8、MXFP4×MXFP8、MXFP4×MXFP4 三种 Gluon GEMM 快速路径，并引入可扩展的量化 GEMM backend 注册、加载期准备和逐层 QDQ 回退机制 |

### 72.2 提交 `5b7571c`：扩展 ONNX accuracy debug 的张量抓取

提交标题是 `[feat] Extend ONNX accuracy debug tensor capture`，修改 4 个文件，净效果是把原来面向少数计算节点输出的专用工具泛化为 ONNX 节点张量调试工具。

#### 1. 新增通用张量抓取模型和 API

`simo/accuracy_debug/onnx_runner.py` 新增 `OnnxTensorCapture`，记录：

- 节点名、`op_type` 和 ONNX domain；
- 实际 tensor 名；
- tensor 是节点 `input` 还是 `output`；
- tensor 在节点输入或输出列表中的索引；
- 用于报告和模型间对齐的 capture 名。

同时新增两个主要 API：

- `find_onnx_tensor_captures()`：从图中筛选需要观察的节点输入/输出；
- `collect_onnx_tensors()`：临时把选中的 tensor 加到图输出，通过 ONNX Runtime 执行模型并收集 tensor 或统计摘要。

原有 `find_onnx_compute_outputs()` 和 `collect_onnx_compute_outputs()` 没有删除，而是保留为兼容接口；后者内部转到新的通用实现。因此，已有只比较 MatMul/Gemm/Conv 输出的调用方式仍可继续使用。

#### 2. 支持选择输入、输出和 initializer

抓取角色由 `AccuracyDebugConfig.capture` 控制，可选择：

- `("output",)`：只抓节点输出，仍是默认行为；
- `("input",)`：只抓节点输入；
- `("input", "output")`：同时抓输入和输出。

节点输入若来自 initializer，默认不抓取，避免把所有静态权重和 scale 都加入运行输出；设置 `capture_initializers=True` 后可一起抓取。实现还会根据 initializer 的 dtype 和 shape 构造 `ValueInfo`，使静态权重也能作为临时图输出交给 ORT。

命名规则保持首个输出与旧报告兼容：节点的 `output0` 仍直接使用节点名；其他张量使用 `node:input0`、`node:input1`、`node:output1` 等名称。

#### 3. 支持按算子类型和 domain 选择节点

旧实现只处理标准 ONNX domain 中的 `MatMul`、`Gemm` 和 `Conv`。新实现同时接受 `op_types` 和 `domains`，并提供四组 preset：

| preset | 选择内容 |
|---|---|
| `compute` | 标准域的 `MatMul`、`Gemm`、`Conv` |
| `simo_qdq` | `com.simo` 域的 `Quantize`、`Dequantize`、`DequantizeFloat16`、`DequantizeBFloat16` |
| `quantized_compute` | 标准计算节点、SIMO Q/DQ 节点，以及量化图常见的 `Transpose`、`Flatten`、`Pad`、`Slice`、`Reshape` |
| `all` | 标准域和 `com.simo` 域中的所有 op type |

`include` / `exclude` glob 也不再只匹配节点名，而会同时检查节点名、op type、`domain::op_type`、输入 tensor 名和输出 tensor 名。这使用户可以按诸如 `com.simo::Quantize` 或某个中间 tensor 名定位量化路径。

#### 4. 泛化模型比较和扫描

`compare_onnx_models()` 与 `scan_onnx_model()` 都切换到新的通用抓取逻辑：

- 参考模型和实际模型均可抓输入、输出或 initializer；
- 可分别为两边注册 SIMO ORT custom ops；
- 仍支持按名称或拓扑顺序对齐；
- 按顺序对齐时，现在会同时校验 domain、op type、input/output 角色和 tensor 索引，防止把语义不同的 tensor 错配；
- 原有误差计算和 JSON/Markdown 报告生成流程保持不变。

图插桩从 `_instrument_model_outputs()` 泛化为 `_instrument_tensor_outputs()`：先尝试 ONNX shape inference，再复用已有 `ValueInfo`；若仍没有类型信息，才退回未知 shape 的 float tensor 描述。

#### 5. 命令行和公共导出同步扩展

`examples/accuracy_debug/run_compare_onnx.py` 新增：

- `--domain`；
- `--op-preset`；
- `--capture input,output`；
- `--capture-initializers`；
- 任意 `--op-type`，不再限定为 MatMul/Gemm/Conv。

`simo/accuracy_debug/__init__.py` 公开导出了 `ONNX_OP_PRESETS`、`find_onnx_tensor_captures` 和 `collect_onnx_tensors`。

测试新增了四类覆盖：节点输入/输出抓取、initializer 抓取、`com.simo` QDQ preset，以及参考/实际模型的输入和输出联合比较。换言之，这个提交的直接用途是精确观察量化前后模型在 Q、DQ、布局变换和计算节点之间从哪里开始产生数值差异。

### 72.3 提交 `7c17fd9`：增加三种 Gluon 低比特 GEMM 及 vLLM backend 调度

提交标题是 `[feat] add w4a8_mxfp4_fp8/w4a8_mxfp4_mxfp8/w4a4_mxfp4_mxfp4 gluon GEMM impl`。它新增约 4,855 行，主要面向 `SIMOLinearMethod` 的量化 Linear；该提交没有改造 `SIMOFusedMoEMethod` 的 MoE GEMM 路径。

#### 1. 三种新增 GEMM 路径

三种路径的权重均为 packed MXFP4 E2M1：`uint8 [N, K/2]`，每个 byte 保存两个 FP4 值；权重 scale 是沿 K 每组 32 个元素一个 E8M0 scale。

| 路径 | 激活格式 | 主要执行策略 |
|---|---|---|
| W4A8 MXFP4×FP8 | FP8 E4M3，per-token/per-row scale | Gluon TMA + WGMMA；小 M 使用 split-K，大 M 使用 dense；满足形状条件时使用加载期准备的 PRMT 权重路径 |
| W4A8 MXFP4×MXFP8 | FP8 E4M3 payload，沿 K 每 32 个元素一个 E8M0 scale | `M <= 32` 直接从 packed W 走 split-K；较大 M 使用预展开的 FP8 权重，并按 activation group 对 FP32 partial accumulator 施加 scale |
| W4A4 MXFP4×MXFP4 | 激活也是 packed MXFP4，group size 32 | 每次调用先把动态激活展开为 E5M2；静态权重在加载期展开为 E4M3；随后执行 FP8×FP8 WGMMA，并在 epilogue 恢复行/列 scale |

三个 kernel 都使用 Gluon/Triton 的 Hopper TMA、shared-memory pipeline 和 WGMMA，生产输出主要支持 FP16/BF16，并以 FP32 accumulator 处理缩放或 split-K 汇总。入口会检查 CUDA device、rank、dtype、packed K 维、scale shape 和 group size 等约束。

#### 2. 静态权重的加载期准备和精确性保护

MXFP4×MXFP8 与 MXFP4×MXFP4 路径会在 `process_weights_after_loading()` 阶段尝试把静态 MXFP4 权重展开为 `[N, K]` 的 E4M3，并保存每个输出 row 的 FP32 base scale：

- 这样推理热路径不必在每次 GEMM 内重复解包和处理权重 group scale；
- 准备后的 tensor 注册为 layer 的非持久 buffer，能跟随 vLLM 的权重迁移并避开以 `data_ptr` 为 key 的 eager cache，适合 `torch.compile` 和 CUDA graph capture；
- 代价是快速路径额外占用约 `N*K` bytes 显存；
- 只有每个权重 row 的 E8M0 exponent span 不超过 14 时，才能无损折叠到 E4M3；不满足时该 layer 标记为 backend 未准备好，自动回到原 QDQ 路径。

MXFP4×FP8 路径的加载期优化不同：对较大的合适权重形状，预先生成 T2/PRMT 重排权重和 per-row E8M0 分解 scale；不满足 PRMT 条件时并不放弃 Gluon backend，而是继续走普通 Gluon dense/split-K 实现。

#### 3. 新增可扩展的 GEMM backend registry

新文件 `simo/extensions/vllm_simo/quantization/gemm_backends.py` 定义了 `QuantGemmBackend` 协议，backend 负责三件事：

- `matches(method)`：判断 weight/input quant spec、granularity、硬件和可选依赖是否匹配；
- `prepare_layer(method, layer)`：在权重加载完成后、compile/CUDA graph capture 前准备静态 buffer；
- `apply(...)`：执行对应的量化 GEMM。

三个 Gluon 实现都注册为名为 `gluon` 的 backend，由各自的 `matches()` 根据 spec 组合区分。当前匹配还要求 CUDA 可用且设备 compute capability 主版本为 9，即 Hopper 路径；导入 Gluon kernel 失败时不会阻断普通 QDQ 初始化。

`SIMOLinearMethod` 的变化是：

1. 初始化时通过 `SIMO_GEMM_BACKEND` 选择 backend；未设置或设为 `auto` 时按注册顺序选择第一个匹配实现，`gluon` 强制尝试 Gluon，`qdq` 显式禁用所有快速路径，未知名称会报错。
2. `process_weights_after_loading()` 调用 backend 的加载期准备，并把成功与否写到 `layer.simo_gemm_backend_ready`。
3. `apply()` 仍先用现有 downcast kernel 生成量化激活；backend 可用、该 layer 准备成功且 K 可按 group size 整除时调用快速 GEMM，否则执行原有的“激活反量化 + 权重反量化 + `torch.matmul`”路径。
4. GEMM 之后原有的 padded shard 裁剪、global scale、bias 和输出 shape/dtype 恢复逻辑保持不变。

代码通过 `torch.ops.simo` 注册了三个带 fake implementation 的 CUDA custom op，隔离 Gluon DSL，使调用可以出现在 `torch.compile` 图中并继续支持 CUDA graph capture：

- `simo::gluon_w4a8_gemm`；
- `simo::gluon_w4a8_mxfp8_gemm`；
- `simo::gluon_w4a4_mxfp4_gemm`。

需要注意的是，`gemm_backends.py` 顶部说明文字提到兼容旧的 `SIMO_W4A8_BACKEND`，但这个提交中 `SIMOLinearMethod.__init__()` 的实际代码只读取 `SIMO_GEMM_BACKEND`，没有读取旧环境变量。因此按实际实现，应使用 `SIMO_GEMM_BACKEND=auto|gluon|qdq`。

#### 4. 离线调优配置进入安装包

提交为三种 kernel 分别加入 sm90 离线调优 JSON：

- `gluon_mxfp4_fp8_configs.json`；
- `gluon_mxfp4_mxfp8_configs.json`；
- `gluon_mxfp4_mxfp4_configs.json`。

配置按 GPU 架构、输出 dtype、N、K 和 M bucket 选择 block 大小、stage、warp 数及 split-K 参数。生产路径不在首次请求或 CUDA graph warmup 时运行在线 autotune：没有精确 M bucket 时选同一 shape 最近的 bucket，shape 完全未覆盖时告警并使用保守固定配置。

`setup.py` 同时把 `simo.ops.kernels.gemm/configs/*.json` 加入 package data，确保安装 wheel 后仍能读取这些调优表。

### 72.4 两个提交的关系

它们没有直接调用关系，但都服务于量化模型的可用性：

- `7c17fd9` 负责让 vLLM 中三种 MXFP4 组合在 Hopper 上走真正的低比特 Gluon GEMM 快速路径，并在不满足精确性或硬件条件时保留 QDQ 回退；
- `5b7571c` 负责在 ONNX 侧更细粒度地抓取和比较原模型、SIMO Q/DQ 节点及其输入输出，定位量化误差来自量化、反量化、布局处理还是后续计算。

因此，前者是推理执行和性能能力，后者是模型数值诊断能力。

---

## 73. 当前 `simo.onnx.quantize()` 在插入 QDQ 前会做常量折叠吗

### 73.1 结论

**默认不会。** 当前 `simo/onnx/api.py` 中 `quantize()` 的 `simplify` 参数默认是 `False`，所以普通调用：

```python
simo.onnx.quantize(model, config)
```

不会在插入 QDQ 前运行常量折叠。

只有显式调用：

```python
simo.onnx.quantize(model, config, simplify=True)
```

才会在 QDQ 插入前尝试运行 ONNXSlim。ONNXSlim 的默认优化流程包含常量折叠，但这只是“尝试”：ONNXSlim 未安装、执行失败，或者某个常量子图不在它的可折叠范围内时，SIMO 会继续使用原模型插入 QDQ，而不会保证该子图已经折叠。

| 调用方式 | QDQ 插入前的行为 |
|---|---|
| `quantize(... )` | 不运行 ONNXSlim，不做图级常量折叠 |
| `quantize(..., simplify=False)` | 同上 |
| `quantize(..., simplify=True)` 且 ONNXSlim 成功 | 先执行包含常量折叠的模型简化，再插入 QDQ |
| `quantize(..., simplify=True)` 但 ONNXSlim 缺失或失败 | 记录 warning，退回原模型，然后继续插入 QDQ |

### 73.2 实际调用顺序

`simo/onnx/api.py:15-25` 中，公共 API 本身只做两件事：

```python
rewritten = apply_qdq_quantization(model, config, simplify=simplify)
if output_path is not None:
  onnx.save(rewritten, output_path)
```

`onnx.save()` 只负责保存，不会优化或折叠模型。是否简化完全由 `apply_qdq_quantization()` 的 `simplify` 参数控制。

`simo/onnx/onnx_quant.py:264-328` 中的顺序是：

```text
加载 ONNX / 复制 ModelProto
  -> 校验 GraphSurgeon 能否无损处理相关字段
  -> 加载并校验量化配置
  -> simplify=True 时调用 simplify_onnx_model()
  -> 将简化后或原始 graph 导入 GraphSurgeon
  -> 识别可量化节点和静态权重
  -> 插入 activation QDQ 与 weight DQ
```

因此，如果启用了简化，常量折叠明确发生在权重静态性判断和 QDQ 插入之前。

### 73.3 `simplify=False` 时哪些权重仍会被当作常量

默认不做图级常量折叠，并不表示 SIMO 只能读取 graph initializer。`simo/onnx/onnx_quant.py:781-796` 的 `_constant_array()` 原地识别两种已经显式存在的静态值：

1. ONNX initializer；导入 GraphSurgeon 后表现为 `gs.Constant`。
2. 直接由标准 ONNX `Constant` 节点产生的 tensor。

这是读取现成常量值，不是执行常量传播或常量折叠。比如：

```text
initializer ---------------------------> MatMul/LSTM W    可识别为静态权重
Constant ------------------------------> MatMul/LSTM W    可识别为静态权重
Constant -> Transpose -----------------> MatMul/LSTM W    默认不能识别
initializer -> Reshape ----------------> MatMul/LSTM W    默认不能识别
Constant + Constant -> Add/Concat -----> MatMul/LSTM W    默认不能识别
```

后三类虽然从数学上仍是常量子图，但 `_constant_array()` 不会递归执行 `Transpose`、`Reshape`、`Add`、`Concat` 等算子；在 `simplify=False` 下，它们会被视为动态算子输出。

### 73.4 ONNXSlim 路径及失败回退

`simo/onnx/onnx_quant.py:483-495` 的 `simplify_onnx_model()` 会先复制模型，然后调用：

```python
slim(candidate, skip_fusion_patterns=["FusionGemm"])
```

`FusionGemm` 被跳过，是为了避免简化过程改变 Gemm 目标结构；这不会关闭 ONNXSlim 的常量折叠。ONNXSlim 的官方优化实现默认会调用 GraphSurgeon 的 `fold_constants()`，然后再做 cleanup 和其他图优化，参见 [ONNXSlim 官方实现](https://github.com/inisis/OnnxSlim/blob/main/onnxslim/core/__init__.py)。

SIMO 对整个导入和简化过程使用了宽泛的 `except Exception`：

```python
except Exception as exc:
  logger.warning("ONNXSlim failed; continuing with the original model: %s", exc)
  return model
```

所以 `simplify=True` 不是“折叠失败就终止量化”，而是“尽力简化，失败则无简化继续量化”。测试 `test_apply_qdq_quantization_simplify_skips_gemm_and_warns_on_failure()` 覆盖了这个回退；`test_apply_qdq_quantization_does_not_simplify_by_default()` 则明确验证默认不会调用 ONNXSlim。

当前 `pyproject.toml` 的 `onnx` optional dependency 声明了 `onnxslim>=0.1.84,<0.2`，但本次检查的 `/share_data/users/like/miniconda3/envs/simo_sglang/bin/python` 环境中实际没有安装 `onnxslim`。因此在这个环境中，即使传入 `simplify=True`，当前结果也是记录“ONNXSlim failed” warning 并使用原图，不会完成常量折叠。

### 73.5 GraphSurgeon 的 cleanup 不是常量折叠

QDQ 插入完成后，`simo/onnx/onnx_quant.py:375-376` 调用：

```python
graph.cleanup(recurse_functions=False).toposort(recurse_functions=False, mode="nodes")
edited = gs.export_onnx(graph, do_type_check=False)
```

`cleanup()` 只删除不再连接到图输出的无用节点和 tensor，`toposort()` 只重新排列拓扑顺序；两者都不会调用 `graph.fold_constants()`。因此这一步可能删除已经被量化权重替换掉的旧 `Constant` 节点，但不能把一个常量计算子图求值成 initializer，也不存在 QDQ 插入后的第二次常量折叠。

### 73.6 对当前 LSTM QDQ 的直接影响

LSTM 的 W 和 R 都通过 `_constant_array()` 检查：

- W、R 都是 initializer 或直接 `Constant`：可以离线拆 gate 并量化。
- W 或 R 任一个仍是 `Transpose`、`Reshape`、`Concat` 等节点的输出：该 LSTM 记录 `skipped:dynamic_weight`，整节点不插入 QDQ。
- 使用 `simplify=True` 且 ONNXSlim 成功把这类常量子图折叠后，W/R 才可能转为可识别的静态值；是否能折叠取决于具体算子、shape 信息和 ONNXSlim 支持范围，SIMO 本身不提供额外的递归常量求值保证。

最终可简化为一句话：**当前 `quantize()` 默认不做常量折叠；`simplify=True` 时会在插入 QDQ 前交给 ONNXSlim 尝试折叠，但失败会记录 warning 后回退到“无折叠继续量化”的路径。**

---

## 74. `kws_simo_quant/test_quant_onnx.sh:59` 的 `case` 语法

### 74.1 所在函数和代码

这段语句位于 `/share/users/like/package/jdjv/kws_simo_quant/test_quant_onnx.sh:55-81`，函数名是 `run_one_config()`：

```bash
run_one_config() {
  local raw_config=$1
  local output_dir=$2
  local quant_config=$raw_config
  case "$quant_config" in
    /*) ;;
    *) quant_config="$CONFIG_DIR/$quant_config" ;;
  esac
  ...
}
```

### 74.2 `case` 的一般语法

`kws_simo_quant/test_quant_onnx.sh:59-62`，函数 `run_one_config()` 使用的是 Bash 的模式匹配分支：

```bash
case 要匹配的值 in
  模式1)
    模式1匹配时执行的命令
    ;;
  模式2)
    模式2匹配时执行的命令
    ;;
esac
```

含义是：先展开并读取 `case` 后面的值，按从上到下的顺序用 shell 通配模式匹配；第一个匹配成功的分支执行完后，由 `;;` 结束整个 `case`。`esac` 是 `case` 反写形式，表示语句结束。

这里的模式是 shell glob，不是正则表达式：

- `*` 表示任意长度的字符串，包括空字符串。
- `/*` 表示以 `/` 开头、后面跟任意字符的字符串。
- `*)` 是兜底模式，匹配前面模式都没有匹配的情况。

### 74.3 逐行解释

- `kws_simo_quant/test_quant_onnx.sh:56`，函数 `run_one_config()`：`local raw_config=$1` 保存函数第一个参数，也就是用户传入的配置路径。
- `kws_simo_quant/test_quant_onnx.sh:57`，函数 `run_one_config()`：`local output_dir=$2` 保存当前配置的结果目录。
- `kws_simo_quant/test_quant_onnx.sh:58`，函数 `run_one_config()`：复制一份 `raw_config` 到 `quant_config`，后面只修改这份待解析路径。
- `kws_simo_quant/test_quant_onnx.sh:59`，函数 `run_one_config()`：开始检查 `quant_config` 的路径形式。
- `kws_simo_quant/test_quant_onnx.sh:60`，函数 `run_one_config()`：`/*)` 匹配绝对路径。分支体为空，随后立即用 `;;` 结束，所以这是“什么都不做，保持原路径”的分支。
- `kws_simo_quant/test_quant_onnx.sh:61`，函数 `run_one_config()`：`*)` 匹配其他路径，通常就是相对路径；将其改成 `CONFIG_DIR/相对路径`。
- `kws_simo_quant/test_quant_onnx.sh:62`，函数 `run_one_config()`：`esac` 结束 `case`。

第 60 行的 `;;` 容易混淆：

```bash
/*) ;;
```

这里的两个分号合起来是 Bash `case` 的分支结束符；由于 `)` 和 `;;` 之间没有命令，所以该分支为空。它等价于写成带空命令的形式：

```bash
/*)
  :
  ;;
```

其中 `:` 是 Bash 的 no-op 命令。

### 74.4 两种输入的实际结果

`kws_simo_quant/test_quant_onnx.sh:16`，脚本顶层变量 `CONFIG_DIR` 默认是 `/share_data/mtang/simo_quant_config`。因此：

```text
调用：run_one_config /tmp/custom.json results/custom
匹配：/*)
结果：quant_config=/tmp/custom.json

调用：run_one_config w8a8/w_mxint8_a_mxint8.json results/mxint8
匹配：*)
结果：quant_config=/share_data/mtang/simo_quant_config/w8a8/w_mxint8_a_mxint8.json
```

也就是说，同一个脚本参数既可以是完整绝对路径，也可以是相对于 `CONFIG_DIR` 的配置文件名。该语句不会检查文件是否存在；真正的存在性检查在 `kws_simo_quant/test_quant_onnx.sh:64`，函数 `run_one_config()`：

```bash
test -f "$quant_config"
```

如果路径不存在，`set -e`（`kws_simo_quant/test_quant_onnx.sh:2`，脚本顶层执行环境）会使脚本在启动 Python runner 前退出。

### 74.5 为什么要这样写

`kws_simo_quant/test_quant_onnx.sh:69-80`，函数 `run_one_config()` 最终把解析后的 `quant_config` 传给 Python：

```bash
"$PY" "$SCRIPT" \
  --quant-config "$quant_config" \
  ...
```

如果不做 `case` 判断而无条件拼接：

```bash
quant_config="$CONFIG_DIR/$quant_config"
```

那么输入 `/tmp/custom.json` 会错误地变成：

```text
/share_data/mtang/simo_quant_config//tmp/custom.json
```

因此，这个 `case` 的核心功能是**规范化配置路径，同时兼容绝对路径和配置目录下的相对路径**；它不负责量化、不读取 JSON，也不改变配置内容。

---

## 75. `kws_simo_quant/test_quant_onnx.sh:83` 的 `if` 语句

### 75.1 所在位置和结构

代码位于 `/share/users/like/package/jdjv/kws_simo_quant/test_quant_onnx.sh:83-89`，属于脚本顶层执行逻辑：

```bash
if [[ -n "${QUANT_CONFIG:-}" ]]; then
  run_one_config "$QUANT_CONFIG" "${OUTPUT_DIR:-results/onnx_quant_${NUM_SHARDS}shards}"
else
  for entry in "${DEFAULT_CONFIGS[@]}"; do
    run_one_config "${entry%%|*}" "$OUTPUT_ROOT/onnx_quant_${entry##*|}_${NUM_SHARDS}shards"
  done
fi
```

Bash 的一般形式是 `if [[ 条件 ]]; then ... else ... fi`：`then` 后是条件为真时执行的分支，`else` 后是条件为假时执行的分支，`fi` 结束整个 `if`。两个分支只会执行一个。

### 75.2 第 83 行条件

`kws_simo_quant/test_quant_onnx.sh:83`，脚本顶层逻辑使用：

```bash
[[ -n "${QUANT_CONFIG:-}" ]]
```

逐部分理解：

- `[[ ... ]]` 是 Bash 条件表达式语法，这里不会启动外部命令。
- `-n` 判断字符串长度是否大于 0，即判断字符串是否非空。
- `${QUANT_CONFIG:-}` 表示变量未设置或为空时使用空字符串，否则使用 `QUANT_CONFIG` 的值。
- `kws_simo_quant/test_quant_onnx.sh:2`，脚本顶层执行环境启用了 `set -u`；`${QUANT_CONFIG:-}` 可以避免变量未设置时直接展开 `$QUANT_CONFIG` 导致脚本退出。

所以第 83 行的条件等价于：

- `QUANT_CONFIG` 非空：运行用户指定的单个配置。
- `QUANT_CONFIG` 未设置或为空：遍历 `DEFAULT_CONFIGS` 批量运行。

### 75.3 true 分支：单配置模式

`kws_simo_quant/test_quant_onnx.sh:84`，脚本顶层逻辑调用 `run_one_config()` 一次：

```bash
run_one_config "$QUANT_CONFIG" "${OUTPUT_DIR:-results/onnx_quant_${NUM_SHARDS}shards}"
```

`kws_simo_quant/test_quant_onnx.sh:55-81`，函数 `run_one_config()` 接收两个参数：第一个参数 `$1` 是配置路径，第二个参数 `$2` 是输出目录。

`${OUTPUT_DIR:-results/onnx_quant_${NUM_SHARDS}shards}` 表示优先使用非空 `OUTPUT_DIR`，否则使用默认输出目录。`NUM_SHARDS` 的默认值在 `kws_simo_quant/test_quant_onnx.sh:11`，是 `32`。

例如：

```text
QUANT_CONFIG=/tmp/custom.json OUTPUT_DIR=results/custom
  -> run_one_config /tmp/custom.json results/custom
  -> 只运行 custom.json
```

配置路径随后由 `kws_simo_quant/test_quant_onnx.sh:59-62`，函数 `run_one_config()` 解析；`kws_simo_quant/test_quant_onnx.sh:69-80`，函数 `run_one_config()` 再把它传给 Python runner。

### 75.4 false 分支：批量运行默认配置

`kws_simo_quant/test_quant_onnx.sh:85-89`，脚本顶层逻辑执行：

```bash
for entry in "${DEFAULT_CONFIGS[@]}"; do
  run_one_config "${entry%%|*}" "$OUTPUT_ROOT/onnx_quant_${entry##*|}_${NUM_SHARDS}shards"
done
```

`DEFAULT_CONFIGS` 在 `kws_simo_quant/test_quant_onnx.sh:20-32`，脚本顶层变量定义，是一个 Bash 数组。使用 `"${DEFAULT_CONFIGS[@]}"` 会让数组中的每个完整元素分别成为一次循环值。

每个元素用 `|` 分成配置路径和输出短名，例如：

```text
w8a8/w_mxint8_a_mxint8.json|w8a8_mxint8
```

#### `${entry%%|*}`：取左侧配置路径

`kws_simo_quant/test_quant_onnx.sh:87`，脚本顶层逻辑中的 `${entry%%|*}` 使用 Bash 后缀删除：`${变量%%模式}` 从末尾删除匹配模式的最长后缀。模式 `|*` 从分隔符开始匹配到字符串末尾，因此结果是：

```text
entry = w8a8/w_mxint8_a_mxint8.json|w8a8_mxint8
entry%%|* = w8a8/w_mxint8_a_mxint8.json
```

这个结果作为 `run_one_config()` 的第一个参数。

#### `${entry##*|}`：取右侧输出短名

同一行的 `${entry##*|}` 使用 Bash 前缀删除：`${变量##模式}` 从开头删除匹配模式的最长前缀。模式 `*|` 删除到最后一个分隔符，因此结果是：

```text
entry = w8a8/w_mxint8_a_mxint8.json|w8a8_mxint8
entry##*| = w8a8_mxint8
```

于是输出目录变成 `$OUTPUT_ROOT/onnx_quant_${entry##*|}_${NUM_SHARDS}shards`。`OUTPUT_ROOT` 的默认规则在 `kws_simo_quant/test_quant_onnx.sh:17`，脚本顶层变量定义：优先使用 `OUTPUT_ROOT`，否则使用 `OUTPUT_DIR`，再否则使用 `results_0708`。

默认配置的一个结果目录示例是：

```text
results_0708/onnx_quant_w8a8_mxint8_32shards
```

### 75.5 整体控制流

```text
test_quant_onnx.sh:83，脚本顶层逻辑
  |-- QUANT_CONFIG 非空
  |     -> test_quant_onnx.sh:84，脚本顶层逻辑
  |        -> run_one_config() 一次
  |           -> test_quant_onnx.sh:69-80，run_one_config()
  |              -> Python runner 一个配置
  |
  `-- QUANT_CONFIG 未设置或为空
        -> test_quant_onnx.sh:86-88，脚本顶层逻辑
           -> 遍历 DEFAULT_CONFIGS
              -> 拆出配置路径和输出短名
              -> run_one_config() 一次
              -> Python runner 一个配置
```

因此，第 83 行的 `if` 不是判断量化是否成功，而是在选择运行模式：有 `QUANT_CONFIG` 时运行一个配置；没有时批量运行 `DEFAULT_CONFIGS`。实际的路径检查、参数拼接和 Python 启动都集中在 `kws_simo_quant/test_quant_onnx.sh:55-81`，函数 `run_one_config()` 中。

## 76. Silero ONNX：LSTM 量化与无 LSTM 量化的精度对比

### 24.1 数据与可比性

比较的日志是：

- LSTM 量化：`silero_vad_clean/temp/test_quant_onnx.sh.log.siplify.lstm.loop.2026_07_29___11_19_17`
- 无 LSTM 量化：`silero_vad_clean/temp/test_quant_onnx.sh.log.siplify.no-lstm.loop.2026_07_29___11_28_31`

两组测试都使用 `aishell4`，20 个文件、1,431,607 个 chunks、24 个 shards，`threshold=0.11`，所有 shard 都完成且 `errors=0`。FP32 baseline 为：

```text
ROC-AUC  = 0.947480
Accuracy = 0.835417
```

两套 JSON 的实际差异已核对：`quant_schema_no_lstm` 只是删除了对应的 `LSTM` module 配置，Conv/Linear 配置没有变化。LSTM 日志中的量化插入统计为 `targets=14, inserted_by_op={"Conv": 12, "LSTM": 2}`；无 LSTM 日志为 `targets=12, inserted_by_op={"Conv": 12}`。

### 24.2 逐格式结果

下表的差值定义为 `LSTM - no-LSTM`，accuracy 差值单位是百分点：

| 格式 | LSTM ROC-AUC | 无 LSTM ROC-AUC | Delta AUC | LSTM accuracy | 无 LSTM accuracy | Delta accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `mxfp4_e2m1` | 0.930555 | 0.927574 | +0.002982 | 0.700299 | 0.633507 | **+6.679** |
| `mxfp6_e2m3` | 0.942664 | 0.941489 | +0.001175 | 0.767670 | 0.766768 | +0.090 |
| `mxfp6_e3m2` | 0.935991 | 0.936454 | -0.000463 | 0.711028 | 0.729514 | **-1.849** |
| `int8 per-channel/per-channel` | 0.946326 | 0.946673 | -0.000347 | 0.826529 | 0.828686 | -0.216 |
| `int8 per-channel/per-tensor` | 0.947174 | 0.947514 | -0.000340 | 0.852185 | 0.853845 | -0.166 |
| `int8 per-tensor/per-tensor` | 0.946199 | 0.945295 | +0.000904 | 0.833215 | 0.831788 | +0.143 |
| `fp8 2d-block/1d-block` | 0.946602 | 0.947480 | -0.000878 | 0.832431 | 0.835417 | -0.299 |
| `mxfp8_e4m3` | 0.942702 | 0.941704 | +0.000998 | 0.764326 | 0.765563 | -0.124 |
| `mxfp8_e5m2` | 0.939033 | 0.938949 | +0.000084 | 0.749316 | 0.767449 | **-1.813** |
| `mxint8` | 0.946483 | 0.946729 | -0.000246 | 0.825363 | 0.828048 | -0.268 |

结论分两层看：

1. 如果看 ROC-AUC，也就是整体排序能力，差异不大。最大差值是 `mxfp4_e2m1` 的约 0.003，其他格式都不超过约 0.0012。
2. 如果看固定阈值 `0.11` 下的 accuracy，差异确实存在。明显格式是 `mxfp4_e2m1`，其次是 `mxfp6_e3m2` 和 `mxfp8_e5m2`。其余格式的差异不大，均不超过 0.3 个百分点。

因此，最大的影响首先表现为 score calibration/阈值位置变化，而不是分类排序能力完全崩溃。不能只根据 accuracy 差异推断 AUC 同样程度地恶化。

### 24.3 `mxfp4_e2m1`：6.679 个百分点差异的具体表现

`mxfp4_e2m1` 是最极端的格式：E2M1 只有 1 个 mantissa bit，配置默认 block size 是 32，scale mode 是 `e8m0_sipu`。它同时量化 Conv 和 Linear，量化误差已经很大；LSTM 是否再经过 QDQ 会改变误差进入 recurrent state 的方式。

从全部 1,431,607 个 score 计算混淆矩阵：

| 版本 | TP | FN | TN | FP |
|---|---:|---:|---:|---:|
| LSTM 量化 | 875,652 | 426,060 | 126,901 | 2,994 |
| 无 LSTM 量化 | 779,080 | 522,632 | 127,853 | 2,042 |

LSTM 量化版本少了 96,572 个 FN，但多了 952 个 FP，净增加 95,620 个正确判断，正好对应约 6.679 个百分点的 accuracy 提升。这个差异不是由少量边界样本造成的：两版本 score 的平均绝对差为约 `0.0621`，95 分位绝对差约 `0.2386`；相对于 FP32 baseline，LSTM/no-LSTM 的平均绝对 score 误差分别约为 `0.2250/0.2783`。

正类 score 也显示出明显的整体偏移：

| 版本 | 正类平均 score | 正类 score 中位数 |
|---|---:|---:|
| LSTM 量化 | 0.437817 | 0.279197 |
| 无 LSTM 量化 | 0.377944 | 0.159844 |

所以在这个实验中，反直觉的结果是：量化 LSTM 反而把正类 score 整体推高，减少了大量 FN。不能将其解释为“LSTM 量化普遍更准确”；更准确的解释是，低精度 Conv 输出进入 recurrent 网络后，无 LSTM QDQ 路径的 score 被明显压低，而额外的 LSTM 动态 QDQ/权重 QDQ 改变了 scale、舍入和 gate 输入，使最终 score calibration 恰好更接近 baseline。这个方向是格式和模型相关的，后面的 `mxfp6_e3m2`、`mxfp8_e5m2` 正好表现为相反方向。

### 24.4 `mxfp6_e3m2` 与 `mxfp8_e5m2`：LSTM 量化造成向下偏移

这两个格式的 ROC-AUC 几乎没有变化，但 LSTM 量化后的 accuracy 分别低 1.849 和 1.813 个百分点。score 统计显示两者都有约 `-0.024` 的整体 LSTM-minus-no-LSTM 平均偏移：

| 格式 | LSTM/no-LSTM 平均 score 差 | 平均绝对差 | 95 分位绝对差 |
|---|---:|---:|---:|
| `mxfp6_e3m2` | -0.02422 | 0.02884 | 0.12333 |
| `mxfp8_e5m2` | -0.02404 | 0.02867 | 0.12133 |

对应的正类混淆矩阵为：

| 格式 | 版本 | TP | FN | TN | FP |
|---|---|---:|---:|---:|---:|
| `mxfp6_e3m2` | LSTM | 891,242 | 410,470 | 126,671 | 3,224 |
| `mxfp6_e3m2` | no-LSTM | 917,971 | 383,741 | 126,407 | 3,488 |
| `mxfp8_e5m2` | LSTM | 946,649 | 355,063 | 126,077 | 3,818 |
| `mxfp8_e5m2` | no-LSTM | 973,042 | 328,670 | 125,644 | 4,251 |

这里 LSTM 量化少了一些 FP，却多了约 26k FN，因此固定阈值下的总 accuracy 下降。AUC 仍接近，说明主要是 score 的幅度/偏置变化，而不是正负样本排序被完全打乱。两个格式都只有 2 个 mantissa bits，经过 block size 32 的 `e8m0_sipu` MX 量化后，LSTM 输入、W/R gate 权重和 recurrent state 的舍入误差会通过 sigmoid/tanh gate 以及时间递推放大；不同指数/尾数布局决定了最终偏移方向和大小。

### 24.5 为什么“无 LSTM 量化”不一定更准确

无 LSTM 配置只意味着 ONNX `LSTM` 节点及其 W/R 权重不插入 QDQ，并不意味着整个 LSTM 输入是 FP32 baseline。它前面的 Conv 仍然已经被 MX/INT8 量化，LSTM 接收到的是量化 Conv 的输出。

当前 SIMO ONNX 实现也不是把 ONNX LSTM 替换为一个独立的低精度 recurrent kernel：

- `simo/onnx/onnx_quant.py:296-326` 会遍历主图和 subgraph，为目标节点插入 activation QDQ，并为权重插入 dequant 节点。
- `simo/onnx/onnx_quant.py:808-945` 对 LSTM 的 W/R 按四个 gate 切分并分别量化。
- `simo/onnx/onnx_quant.py:1305-1345` 再把量化后的 gate 权重拼回 ONNX LSTM 的输入。

也就是说，LSTM 量化版本仍然执行 ONNX LSTM，但其输入、W 和 R 都经过量化/反量化；无 LSTM 版本保留原始 LSTM W/R，但仍承受上游低精度 Conv 的误差。对于 recurrent 模型，误差会影响 gate、hidden state 和 cell state 的连续递推，因此“保留 LSTM FP32”不是一个能保证最终 VAD score 更接近 baseline 的充分条件。

### 24.6 `fp8 2d-block/1d-block` 是一个有用的对照组

该配置有特殊的实现限制。日志显示：

```text
LSTM:    targets=14 inserted=2  skip_reasons={"conv_fp8_per_block": 12}
no-LSTM: targets=12 inserted=0  skip_reasons={"conv_fp8_per_block": 12}
```

无 LSTM 版本的 12 个 Conv 也因为 `conv_fp8_per_block` 被跳过，LSTM 又没有配置，因此实际上没有插入量化 QDQ，结果与 FP32 baseline 完全相同：ROC-AUC `0.947480`、accuracy `0.835417`。LSTM 版本只量化 2 个 LSTM，所以相对 baseline 只下降到 `0.946602/0.832431`。这个对照组直接证明，当前差异来自实际插入的 LSTM QDQ，而不是日志目录或评测数据不一致。

### 24.7 `w6a6` 的输出目录覆盖问题

默认列表中的以下两项使用了相同的输出短名：

```text
w6a6/w_mxfp6_e2m3_a_mxfp6_e2m3_scale_int_sipu.json|w_mxfp6_e3m2_a_mxfp6_e3m2_scale_int_sipu
w6a6/w_mxfp6_e3m2_a_mxfp6_e3m2_scale_int_sipu.json|w_mxfp6_e3m2_a_mxfp6_e3m2_scale_int_sipu
```

因此两次运行都写入 `onnx_quant_w_mxfp6_e3m2_a_mxfp6_e3m2_scale_int_sipu_aishell4_24shards`。当前该目录的 `summary.json` 是第二个 `e3m2` 配置的结果；第一个 `e2m3` 的结果只能从 loop 日志读取，不能再从独立 summary/npz 中复核。日志中的数值是：

- `e2m3`：LSTM `0.942664/0.767670`，no-LSTM `0.941489/0.766768`。
- `e3m2`：LSTM `0.935991/0.711028`，no-LSTM `0.936454/0.729514`。

后续复测必须给两个配置不同的短名或不同 `OUTPUT_ROOT`，否则 10 次循环只会留下 9 个独立结果目录。

### 24.8 跨机器归因限制

本机和 `bjh5` 已确认都是 8 张 H100 80GB，Python 环境版本也一致：NumPy `2.3.5`、ONNX `1.22.0`、ONNXRuntime `1.27.0`、Torch `2.11.0+cu130`；两端都从 `/share/users/like/package/simo_conda_sglang/simo` 加载 editable SIMO，custom-op `.so` 的 SHA256 也一致。

但两台机器的 NVIDIA driver 不同：

```text
本机   590.48.01
bjh5   595.71.05
```

因此上表严格来说是“本机 LSTM 量化”和“bjh5 无 LSTM 量化”的差异，不能把所有数值差异无条件归因于 LSTM 配置。CUDA driver、ONNXRuntime provider 或 custom kernel 的算法选择/舍入差异，尤其可能影响 MX QDQ 和 recurrent 路径。

最终归因建议：在同一台机器、同一个 driver 下，用同一份模型和同一份数据分别运行 `quant_schema` 与 `quant_schema_no_lstm`；同时修复两个 `w6a6` 输出短名冲突。比较时除了固定阈值 accuracy，还应报告 ROC-AUC，并在同一验证集重新选择阈值，否则 score calibration 偏移会把一个相对温和的排序差异放大成几个百分点的 accuracy 差异。

## 77. `DynamicQuantizeLSTM` 的两个矩阵乘法和 MLAS GEMM 数据类型

结论：对于每个门，`X_t W_g^T` 和 `H_{t-1} R_g^T` 这两个矩阵乘法都走了整数 QGEMM；不是只量化 W 或只量化其中一个 matmul。但是，整个 LSTM 并不是整数执行：QGEMM 的结果会立即反量化为 `float`，之后的 peephole、bias、sigmoid/tanh、cell state 和 hidden state 更新仍然在浮点路径中。

### 77.1 四个 gate 实际上是两个融合 GEMM

ONNX 文档中的 `i/f/c/o` 四组公式，在 ORT 中不是分别发起 8 个小矩阵乘法。`UniDirectionalLstm::ComputeImpl` 把四个 gate 沿输出列拼成一个宽矩阵，令 `N = 4 * hidden_size`：

- 第一次 GEMM 一次计算所有时间步的 `X * W[iofc]^T`，对应 `ComputeGemm(total_rows, 4*hidden_size, input_size, ...)`。
- 每个时间步再计算一次 `H_{t-1} * R[iofc]^T`，对应 `ComputeGemm(batch_rows, 4*hidden_size, hidden_size, ...)`。

因此，`it`、`ft`、`ct`、`ot` 的 X-W 项和 H-R 项都包含在这两个 4-gate fused GEMM 中，四个 gate 的对应切片都会经过同样的整数 GEMM 和 scale 处理。双向 LSTM 对两个 direction 分别使用各自的 W/R 和量化参数。

### 77.2 哪些部分量化，哪些部分没有量化

`DynamicQuantizeLSTM::Compute` 接收已经量化的 W/R（类型约束为 `uint8` 或 `int8`），并为它们设置 scale/zero point；权重还可能在 `PrePack` 中被 MLAS 打包。对激活侧，`ComputeGemm` 每次调用都会：

1. 用 `GetQuantizationParameter` 从当前 float A 计算动态 scale 和 zero point；
2. 用 `ParQuantizeLinearStd` 把 A 写成 `uint8`；
3. 用 A 的动态 scale 与 W/R 的权重 scale 相乘，交给 MLAS 输出处理器把 int32 累加结果转换回 float。

所以四个公式可按下表理解：

| 公式部分 | 是否经过整数 QGEMM | 说明 |
|---|---|---|
| `X_t*(Wi/Wf/Wc/Wo)^T` | 是 | X 动态量化为 uint8，W 为 uint8/int8；四个 gate 在一个 fused GEMM 中计算 |
| `H_{t-1}*(Ri/Rf/Rc/Ro)^T` | 是 | 每个时间步将当前 H 动态量化为 uint8，R 为 uint8/int8；结果累加到前一项 |
| `Pi(.)Ct-1`、`Po(.)Ct-1`、bias 项 | 否 | 由 LSTM 的 gate/state 浮点逻辑处理 |
| `Ct`、`Ht` 更新及 sigmoid/tanh | 否 | QGEMM 输出已经回到 float 后再执行 |

第一次 GEMM 用 `beta=0` 生成 `XW`；后续 `H R` GEMM 用 `beta=1`，在输出处理阶段将 `HR` 的反量化结果加到已有的 float `XW` 上。因此“两个 matmul 全部量化”准确地说是“两个矩阵乘法的乘法和 int32 累加都量化执行”，不是“门公式中的所有加法、乘法和激活都变成整数”。

### 77.3 MLAS integer GEMM 的输入、累加器和输出

在 `onnxruntime/core/providers/cpu/rnn/rnn_helpers.cc` 的量化 `ComputeGemm` 中，传给 `MlasGemm` 的类型是：

- A：`const uint8_t*`。`MLAS_GEMM_QUANT_SHAPE_PARAMS::AIsSigned` 保持默认 `false`，X/H 都由 `ParQuantizeLinearStd` 量化成无符号 8 bit。
- B：`const void*`，实际为 W 或 R 的 `uint8_t`/`int8_t`。`BIsSigned` 由 `weights.quant_para_->is_signed` 指示；因此支持 U8xU8 和 U8xS8 等对应的 MLAS kernel。
- C：`int32_t*`。这是 8-bit 乘积的整数累加器，`MlasGemm` 的量化接口明确以 `int32` 保存 C。
- 输出处理器：`MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR` 读取 int32 C，乘以 `a_scale * weight_scale`，并把结果写入 `float* output_iofc`。第一次调用覆盖输出，后续调用用 `AccumulateMode` 加到已有 float 输出上。

因此，如果“GEMM 输出”指 MLAS integer kernel 的原始 C，答案是 `int32`；如果指 `ComputeGemm` 返回给 LSTM gate 计算的结果，答案是 `float`。其数据流可写成：

```text
float X/H --dynamic quantize--> uint8 A
uint8/int8 W/R ----------------> B
                 MLAS QGEMM
             int32 accumulator C
                       |
       scale (a_scale * w_scale/r_scale)
                       v
                 float XW/HR
                       |
          float gate/state computation
```

这里的“动态”主要针对 X 和 H 的运行时量化参数；W/R 的量化数据和 scale/zero point 是 DynamicQuantizeLSTM 的输入，而不是在每个时间步重新从 float 权重计算。

### 77.4 按函数追踪完整调用链

下面的文件名和行号均相对于 `/softhome/like/package/onnxruntime/` 这个 ONNX Runtime codebase；行号对应当前 checkout。

#### (1) `DynamicQuantizeLSTM` 的输入契约

- `onnxruntime/core/graph/contrib_ops/quantization_defs.cc:657-758`，`ONNX_MS_OPERATOR_SET_SCHEMA(DynamicQuantizeLSTM, 1)`：
  - 输入 0 `X` 使用类型约束 `T`，而 `T` 被限制为 `tensor(float)`；
  - 输入 1 `W`、输入 2 `R` 使用 `T2`，而 `T2` 被限制为 `tensor(uint8)` 或 `tensor(int8)`；
  - 输入 3 `B`、输入 5 `initial_h`、输入 6 `initial_c`、输入 7 `P` 仍使用 `T=float`；
  - 输入 4 `sequence_lens` 是 `int32`；输入 8-11 是 W/R 的 scale 和 zero point。
- `onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc:238-248`，`ONNX_OPERATOR_TYPED_KERNEL_EX(... DynamicQuantizeLSTM ...)`：再次注册同样的类型约束。因此这个 Microsoft 域的 `DynamicQuantizeLSTM` 不是“接收 float W/R 后在算子内部首次量化”的接口，而是接收已经是 8 bit 的 W/R；它与标准 ONNX `LSTM` 的 float W/R 接口不同。

标准 LSTM 的公式和标准 W/R 形状仍可在 ONNX codebase 的 `docs/Operators.md:17332-17423`，`LSTM` 文档中看到。DynamicQuantizeLSTM 的自定义 schema 在 `quantization_defs.cc:692-700` 使用了内部 GEMM 友好的 W/R 布局 `[num_directions, input_size/hidden_size, 4*hidden_size]`，阅读实现时应以这个自定义 schema 和后面的 `ComputeGemm` 调用为准，不能只按标准 LSTM 文档的 float W/R 形状推断内存布局。

#### (2) 入口如何把 W/R 交给 LSTM 内核

- `onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc:94-118`，`DynamicQuantizeLSTM::PrePack`：当输入编号为 1 或 2 时，分别为 W 或 R 调用 `TryPackWeights`。
- `onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc:41-87`，`DynamicQuantizeLSTM::TryPackWeights`：
  - 在第 57 行读取 `weights.IsDataType<int8_t>()`，记录 B 是否有符号；
  - 第 58 行调用 `MlasGemmPackBSize(N, K, false /*AIsSigned*/, is_weight_signed, ...)`；
  - 第 80 行调用 `MlasGemmPackB(..., false /*AIsSigned*/, is_weight_signed, ...)`。
  这里明确了激活矩阵 A 固定按无符号处理，而 W/R 由 `is_weight_signed` 决定是 uint8 还是 int8。预打包只改变 B 的存储格式，不把 W/R 变成 float，也不在每个时间步重新量化 W/R。
- `onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc:166-236`，`onnxruntime::contrib::DynamicQuantizeLSTM::Compute`：
  - 第 168-178 行取得 W/R 和四个 scale/zero-point 输入；
  - 第 190-206 行确定 W/R 的 signedness，并构造 `QuantizationParameter`；
  - 第 215-216 行构造 `GemmWeights<uint8_t> W_1/R_1`。这里的模板参数 `uint8_t` 是内部“8 bit 权重”的封装类型，不意味着底层 B 一定是无符号；底层实际 signedness 仍由 `QuantizationParameter::is_signed` 和 MLAS 的 `BIsSigned` 字段表达；
  - 第 224-232 行在双向情况下建立 direction 1 的 W/R carrier，并把第二个 direction 的 scale/zero-point 指针向后移动；
  - 第 235 行调用 `LSTMBase::ComputeImpl<float, uint8_t>`。

因此，权重路径可以概括为：

```text
uint8/int8 W/R + W/R scale/zp
        -> PrePack/TryPackWeights（可选的 B 预打包）
        -> GemmWeights<uint8_t> + QuantizationParameter
        -> LSTMBase::ComputeImpl<float, uint8_t>
```

#### (3) `LSTMBase` 如何分别处理 direction 和可选输入

- `onnxruntime/core/providers/cpu/rnn/lstm_base.cc:23-27`，`LSTMBase::ComputeImpl<InputT, WeightT>`：DynamicQuantizeLSTM 实例化的是 `InputT=float, WeightT=uint8_t`。
- 同一函数第 32-39 行读取 X、B、`sequence_lens`、`initial_h`、`initial_c`、P；第 79-88 行把 B 和 P 转成 `gsl::span<const InputT>`，也就是 float span。
- 第 146-159 行创建 forward/reverse 两个 `UniDirectionalLstm<float>` 并分别调用 `fw.Compute(..., W_1, R_1, ...)` 和 `bw.Compute(..., W_2, R_2, ...)`；第 161-167 行是单向路径。也就是说，双向并不会把两个方向拼成一个有符号性不同的 GEMM，而是对两个 direction 重复同一套 X/W、H/R 量化 GEMM 流程。
- 第 50-58 行创建 Y、Y_h、Y_c 输出；这些输出的 `InputT` 也是 float。

#### (4) 第一次 GEMM：一次生成所有时间步的 `X·W`

- `onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc:626-634`，`UniDirectionalLstm<T>::Compute<WeightT>` 只是把参数转发给 `ComputeImpl`；CPU 动态量化实例在第 654-659 行显式实例化了 `UniDirectionalLstm<float>::Compute<uint8_t>`。
- `onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc:228-293`，`UniDirectionalLstm<T>::ComputeImpl<WeightT>`：
  - 第 281 行设置 `hidden_size_x4 = 4 * hidden_size_`；
  - 第 282 行设置 `total_rows = max_sequence_length * batch_size_`；
  - 第 287-293 行调用 `ComputeGemm(total_rows, hidden_size_x4, input_size_, ..., inputs, input_weights, beta=0, output_iofc, ...)`。

这一次不是为 i/f/c/o 各调用一次 GEMM，而是把四个 gate 的输出列合并成 `N=4H`。逻辑上它同时产生：

```text
XW_i, XW_f, XW_c, XW_o
```

具体的物理 gate 切片由 `GateComputations` 使用。`onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc:490-495` 将 4H 宽的行切成 `pi/po/pf/pc` 四个 float 指针，`595-598` 的调试输出也明确按 i/o/f/c 四段查看。这里的 gate 融合只影响内存布局，不改变“每个 gate 的 X-W 项都经过同一个整数 QGEMM”的结论。

#### (5) 第二次 GEMM：每个时间步生成 `H_{t-1}·R`

- `onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc:333-351` 位于 `UniDirectionalLstm<T>::ComputeImpl<WeightT>` 的时间步循环中。
- 第 342-349 行调用：

```text
ComputeGemm(num_seq_to_compute_adjusted,
            4 * hidden_size_,
            hidden_size_,
            ..., previous_state, recurrent_weights,
            beta=1, step_out_IOFC,
            ..., quantized_C_buffer_)
```

这里 `previous_state` 的模板类型 T 是 float，也就是 `H_{t-1}` 在进入量化 `ComputeGemm` 前仍是 float；第 297 行已经把 beta 改为 1，表示这次要把 `H R` 项加入已经保存的 `X W` 项。第 370-372 行随后调用 `GateComputations`。

因此，用户问题中的四个式子应逐项理解为：

| ONNX 公式中的项 | ORT 实际路径 |
|---|---|
| `Xt*(Wi^T)`、`Xt*(Wf^T)`、`Xt*(Wc^T)`、`Xt*(Wo^T)` | 第一次 `ComputeGemm` 的同一个 `N=4H` fused QGEMM |
| `Ht-1*(Ri^T)`、`Ht-1*(Rf^T)`、`Ht-1*(Rc^T)`、`Ht-1*(Ro^T)` | 时间步循环中的第二次 `ComputeGemm`，同一个 `N=4H` fused QGEMM |
| `Pi(.)Ct-1`、`Pf(.)Ct-1`、`Po(.)Ct` | `GateComputations` 中的 float elementwise product，第 505-508、519-521、560-563 行 |
| `Wb/Rb` bias | 构造函数 `UniDirectionalLstm<T>::UniDirectionalLstm` 第 48-90 行加载；`LoadBias` 第 180-193 行把 Wb 与 Rb 相加，之后由 `clip_with_bias_ptr_` 在 `GateComputations` 第 510-534、524-568 行处理 |
| sigmoid/tanh、`Ct`、`Ht` | `GateComputations` 第 510-585 行的 float activation、`merge_lstm_gates_to_memory` 和 H 输出 |

#### (6) `ComputeGemm` 中真正发生的动态量化

- `onnxruntime/core/providers/cpu/rnn/rnn_helpers.cc:247-317`，量化权重的 `ComputeGemm(..., const GemmWeights<uint8_t>& weights, ...)` 是关键函数：
  - 第 271-273 行调用 `GetQuantizationParameter(A, M*K, a_scale, a_zero_point, ...)`；
  - 第 275-276 行调用 `ParQuantizeLinearStd(A, quantized_A_buffer, M*K, a_scale, a_zero_point, ...)`；
  - 第 278-284 行读取 W/R 的 signedness 和 scale，并构造 `scale_multiplier[s] = a_scale * weights_scale[s]`；
  - 第 286-296 行准备 C 缓冲区和输出处理器；
  - 第 298-316 行填充 `MLAS_GEMM_QUANT_SHAPE_PARAMS`/`MLAS_GEMM_QUANT_DATA_PARAMS` 并调用 `MlasGemm`。
- `onnxruntime/core/util/qmath.h:50-110`，`GetQuantizationParameter<QType>` 通过当前 A 的 min/max 计算 scale 和 zero point；这里的 A 是整个本次 GEMM 的输入，不是单独为 i/f/c/o 四个 gate 各算一套 activation scale。
- `onnxruntime/core/util/qmath.h:122-135`，`ParQuantizeLinearStd<OutputType>` 把 float 输入逐元素写入量化输出。当前调用传入的 `quantized_A_buffer` 类型是 `uint8_t*`，所以这里的 `OutputType` 为 `uint8_t`。

动态量化的粒度因此是：

- 第一次 X-W GEMM：A 的元素数是 `M*K = total_rows * input_size`，通常覆盖当前 direction 的所有有效时间步和 batch 行；
- 每个 H-R GEMM：A 的元素数是 `M*K = num_seq_to_compute_adjusted * hidden_size`，每次时间步/批次分块重新根据 H 的 min/max 求一套 scale/zero point；
- W/R 的 scale 来自 DynamicQuantizeLSTM 输入 8/10，若是二维 scale，则 `scale_size=4H`，在 `ComputeGemm` 中作为 per-column scale multiplier；若是一维 scale，则是 per-matrix scale。

这里没有“先把 X 量化成一个图中的 uint8 输出再交给标准 LSTM”的过程。量化 buffer 是 ORT CPU kernel 的临时内存：`AllocateQuantizeBuffers` 位于 `uni_directional_lstm.cc:215-224`，X/H 共用 uint8 buffer，`quantized_C_buffer_` 是 H-R beta=1 路径使用的 int32 临时 buffer。

#### (7) MLAS QGEMM 的精确输入/输出类型

- `onnxruntime/core/mlas/inc/mlas.h:613-620`，`MLAS_GEMM_QUANT_SHAPE_PARAMS` 描述 M/N/K 和 A/B signedness；其中 `AIsSigned` 默认是 `false`，`BIsSigned` 由调用方设置。
- `onnxruntime/core/mlas/inc/mlas.h:622-634`，`MLAS_GEMM_QUANT_DATA_PARAMS` 的字段直接给出类型：
  - `A` 是 `const uint8_t*`；
  - `B` 是 `const void*`，实际由 `BIsSigned` 解释为 uint8 或 int8；
  - `C` 是 `int32_t*`；
  - `ZeroPointA/ZeroPointB` 是 uint8 zero point 指针/值；
  - `OutputProcessor` 是对 int32 C 的后处理器。
- `onnxruntime/core/mlas/inc/mlas.h:656-664`，`MlasGemm(const MLAS_GEMM_QUANT_SHAPE_PARAMS&, const MLAS_GEMM_QUANT_DATA_PARAMS&, ...)` 只是把单个 GEMM 转发给 `MlasGemmBatch`。
- `onnxruntime/core/mlas/lib/qgemm.cpp:134-202`，`MlasGemmBatch` 做线程切分；`onnxruntime/core/mlas/lib/qgemm.cpp:37-115`，`MlasGemmQuantThreaded` 根据 `Shape->AIsSigned` 和 `Shape->BIsSigned` 选择具体的整数 kernel，并根据 `Data->BIsPacked` 选择预打包或未打包 B。

所以 MLAS integer kernel 这一层看到的是：

```text
A = uint8 activation
B = uint8/int8 weight
C = int32 accumulator
```

但要注意 `ComputeGemm` 的两个 beta 分支：

- `onnxruntime/core/providers/cpu/rnn/rnn_helpers.cc:286-288`，量化重载 `ComputeGemm`：先把 `C` 视为 `int32_t*`；beta=0 时它指向 `output_iofc` 的同一块存储；
- `onnxruntime/core/providers/cpu/rnn/rnn_helpers.cc:288-291`，同一个量化重载 `ComputeGemm`：beta=1 时改用 `quantize_agg_C_buffer`，避免直接覆盖已经由第一次 GEMM 写入的 float `output_iofc`；
- `onnxruntime/core/providers/cpu/rnn/rnn_helpers.cc:293-296`，同一个量化重载 `ComputeGemm`：构造 `MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR`，beta=0 使用 `ZeroMode`，beta=1 使用 `AccumulateMode`；
- `onnxruntime/core/providers/cpu/rnn/rnn_helpers.cc:304-316`，同一个量化重载 `ComputeGemm`：把这两个 C 指针作为 `gemm_params.C` 传入 `MlasGemm`。

#### (8) 为什么 MLAS 的“输出”最终是 float

- `onnxruntime/core/mlas/lib/qpostprocessor.cpp:19-101`，`MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR::Process` 根据 ZeroMode/AccumulateMode 和 per-matrix/per-column 分派到 `ProcessImpl`。
- `onnxruntime/core/mlas/lib/qpostprocessor.cpp:106-190`，`ProcessImpl` 读取 `const int32_t* C`，把整数 C 转成 float，乘以 scale；
- `onnxruntime/core/mlas/lib/qpostprocessor.cpp:197-231` 的标量路径明确实现了：`result = float(c[offset]) * ScaleValue`，ZeroMode 写入 `c_out`，AccumulateMode 执行 `c_out += result`。

因此第二个 `H R` GEMM 不是把两个 int32 GEMM 结果直接相加。实际顺序是：

```text
第一次：QGEMM(uint8 X, uint8/int8 W) -> int32 C_XW
        -> scale(a_X * scale_W) -> float output_iofc = XW

第二次：QGEMM(uint8 H, uint8/int8 R) -> int32 C_HR
        -> scale(a_H * scale_R) -> float output_iofc += HR

最后：float output_iofc
      -> peephole/bias/activation/state update
      -> float Y/Y_h/Y_c
```

这就是“两个 matmul 都量化”与“整个 LSTM 都是整数”的边界：乘法和整数累加在 QGEMM 中完成，但每个矩阵乘法的结果在进入 gate 逻辑前已经被还原为 float；最终 `Y/Y_h/Y_c` 也由 `onnxruntime/core/providers/cpu/rnn/lstm_base.cc:50-58`，`LSTMBase::ComputeImpl` 以 `InputT=float` 分配。

## 78. Silero 日志中的 ROC-AUC、Acc@0.11、overall 与 aishell4

### 78.1 这些指标到底在评测什么

本次脚本不是按整段音频直接给一个标签，而是把每个音频切成连续的 512-sample chunk。当前采样率是 16 kHz，所以一个 chunk 约为 32 ms。`SileroVadOnnx.predict_proba()` 对每个 chunk 输出一个分数；AISHELL-4 的 RTTM（没有 RTTM 时才回退到 TextGrid）被合并为 speech 时间区间，再映射成同样长度的二值标签：与 speech 区间相交的 chunk 标为 `True`，其余标为 `False`。

因此日志中的 ROC-AUC 和 Accuracy 都是“chunk 级别”的指标，不是文件级别的指标，也不是直接测量最终语音切分段数的指标。模型输出虽然在代码变量中叫 `probs`，但下面的 ROC-AUC 只把它当作可排序的 speech score；它不保证这个分数已经是严格校准的概率。

### 78.2 ROC-AUC 的业务含义

ROC-AUC 是模型在所有可能阈值下区分 speech 和 non-speech 的整体排序能力。它可以直观理解为：随机取一个真实 speech chunk 和一个真实 non-speech chunk，模型把前者打分得更高的概率；例如 `ROC-AUC=0.94748` 大致表示这个概率为 94.748%（忽略分数相同的细节）。

它对业务的含义是：

- 衡量 VAD 的基础区分能力，而不绑定某个固定阈值；
- 当下游可以根据场景重新选择阈值，或需要在漏检和误检之间做不同权衡时，AUC 更适合比较模型/量化格式；
- AUC 越接近 1，通常说明 speech 分数和 non-speech 分数的排序重叠越少。

ROC-AUC 不是“94.748% 的分类准确率”，也不直接告诉我们在阈值 0.11 下漏检多少、误报多少。一个量化模型可能保持相近的 AUC，却因为所有 score 整体偏高或偏低，使某个固定阈值下的结果明显变化。

### 78.3 `Acc@0.11` 的业务含义

脚本中的定义是：

```text
prediction = (score >= 0.11)
Acc@0.11 = count(prediction == reference_label) / count(all_chunks)
```

所以 `Acc@0.11=0.835417` 的含义是：在当前 AISHELL-4 测试集的所有 1,431,607 个 32 ms chunk 上，固定使用 0.11 作为决策阈值时，约 83.5417% 的 chunk 被判对。它更接近“当前部署阈值下的总体命中率”，而不是模型在全部阈值上的能力。

这个指标有几个业务限制：

- 它强依赖阈值 0.11；阈值改变，Accuracy 也会改变；
- 它是所有 chunk 合并后的 micro accuracy，不是每个文件 Accuracy 的平均值，长文件的权重更大；
- 语音与非语音 chunk 的比例会影响 Accuracy，因此不能只看 Accuracy 判断漏检和误报；
- 它不等同于 precision、recall、F1、DER 或端点延迟。实际 VAD 业务还应同时看 confusion matrix、speech recall、non-speech false alarm，以及阈值变化下的曲线。

例如，若 ROC-AUC 基本不变但 `Acc@0.11` 下降，通常先检查 score calibration 是否发生整体偏移，并在同一验证集重新选择部署阈值；这不一定意味着模型的 speech/non-speech 排序能力同等幅度地恶化。

### 78.4 `aishell4` 是什么

`aishell4` 是本次评测明确请求的数据集名称，对应 AISHELL-4 test 数据。当前运行中它包含 20 个音频文件、约 12.7253 小时音频和 1,431,607 个 chunk。脚本对每个文件运行有状态的 Silero VAD，并使用该文件的 RTTM speaker 区间合并成 speech union 作为参考标签。

summary 中的 `per_dataset` 部分会列出一行：

```text
dataset = aishell4
```

这一行回答的是“模型在 AISHELL-4 上表现如何”，并且该数据集有代码内置的参考值 `official_roc_auc=0.94`、`official_accuracy=0.85`。这两个 official 数值是基准参考，不应与本次固定阈值 0.11 的结果当作完全相同的评测协议；尤其官方 Accuracy 的阈值/选择过程不一定就是本次的 0.11。

### 78.5 `overall` 是什么

`overall` 不是另一份数据，也不是官方指标。聚合脚本会先分别合并每个请求数据集的所有 shard，然后把所有数据集的 labels 和 scores 拼接起来，在拼接后的全部 chunk 上重新计算：

```text
overall ROC-AUC  = pooled labels/scores 的 ROC-AUC
overall Accuracy = 全部数据集正确 chunk 数 / 全部数据集 chunk 数
```

因此它是按 chunk 汇总的整体指标，而不是各数据集 AUC 的算术平均。多个数据集时，数据量更大的数据集会在 Accuracy 中贡献更多 chunk；AUC 也会在合并后的样本池上计算，可能与各数据集 AUC 的简单平均不同。`overall` 没有对应的 `official_*` 参考值，所以 summary 里通常为 `null`。

本次命令只传了 `DATASETS=aishell4`，所以 `overall` 和 `aishell4` 实际包含完全相同的 20 个文件、相同的 chunk、labels 和 scores；两行的 ROC-AUC、`Acc@0.11`、文件数、小时数和 errors 应当完全一致。只有把命令改成例如 `DATASETS=aishell4,voxconverse` 时，`overall` 才会代表两个数据集拼接后的整体结果，而 `aishell4` 仍只代表 AISHELL-4。

### 78.6 阅读当前日志的建议

1. 先用 `overall` 比较同一批数据、同一阈值下的整体结果；在本次单数据集运行中，它与 `aishell4` 可视为同一个数。
2. 用 ROC-AUC 判断量化是否破坏了整体排序能力。
3. 用 `Acc@0.11` 判断当前固定部署阈值下的实际 chunk 决策变化。
4. 当两者结论不一致时，进一步检查 score 分布、阈值下的 TP/FN/FP/TN，并重新调阈值，而不要把 AUC 直接当成 Accuracy。

## 79. 在本机从源码编译并安装 CUDA 13 版 ONNX Runtime

### 79.1 本机已核对的版本和路径

以下路径和版本已经在本机核对过：

| 项目 | 当前值 |
| --- | --- |
| ORT 源码 | `/softhome/like/package/onnxruntime/`（git 实际路径解析为 `/share_data/users/like/package/onnxruntime`） |
| ORT checkout | `branch-v1.27.0`，commit `8f0278c77b`，版本 `1.27.0` |
| Python 环境 | `/share_data/users/like/miniconda3/envs/simo_sglang/`，Python 3.12.12 |
| CUDA Toolkit | `/share_data/users/like/opt/cuda-13.0`，nvcc `13.0.48` |
| cuDNN | conda 环境的 `site-packages/nvidia/cudnn`，cuDNN `9.19.0.56`，包含 `include/cudnn.h` 和 `lib/libcudnn.so.9` |
| CMake / Ninja | conda 环境中分别为 `4.2.0` / `1.13.0` |
| GPU / driver | H100 80GB（sm90），driver `590.48.01` |

这个 checkout 的 CMake 已包含 CUDA 13 的处理：最低 CUDA 版本为 12.0，并对 CUDA 13 设置编译前端、runtime library 和 FP4/FP8 编译选项。因此应使用这份本地源码，而不是换成未带这些改动的普通 v1.27.0 tag。

`/softhome/like/package/onnx` 这份独立 ONNX 源码不是本次构建的输入。ONNX Runtime 会使用 `cmake/external/onnx` 中由 git submodule 固定的版本；只有该 submodule 缺失时才需要初始化 submodule，而不是把独立 ONNX checkout 手工塞进构建目录。

当前环境安装的是 `onnxruntime-gpu==1.27.0`，它同时公开了 `TensorrtExecutionProvider`、`CUDAExecutionProvider` 和 `CPUExecutionProvider`。下面的命令只启用 `--use_cuda`，因此新 wheel 只包含 CUDA/CPU EP；若直接安装它，当前 TensorRT EP 将消失。若业务依赖 TensorRT，先在独立环境验证，或另行准备匹配 CUDA 13 的 TensorRT 后加 `--use_tensorrt --tensorrt_home <TensorRT 根目录>` 构建。

### 79.2 推荐方式：构建 wheel 后覆盖安装

不建议把 ONNX Runtime 作为 editable package 安装。标准且可复现的路径是：使用 ORT 自带 `tools/ci_build/build.py` 生成 `onnxruntime_gpu` wheel，再将 wheel 安装到指定 conda 环境。SIMO 可以继续保持 editable 安装，两者互不要求改动 SIMO 源码。

首次构建前，先确认 ORT submodule 已就绪：

```bash
ORT_SRC=/softhome/like/package/onnxruntime
git -C "$ORT_SRC" submodule status --recursive
```

如果输出中有以 `-` 开头的条目，表示 submodule 尚未检出；此时在确认允许同步源码依赖后执行：

```bash
git -C "$ORT_SRC" submodule update --init --recursive
```

不要使用已有的通用 `build/` 目录来切换 CUDA 版本。CMake 会缓存 CUDA、cuDNN、编译器和 generator；下列流程使用独立目录，避免复用 CUDA 12 或旧 generator 的 cache。

### 79.3 准备干净的 CUDA 13 构建环境

在同一个 shell 中执行：

```bash
set -euo pipefail

ENV_ROOT=/share_data/users/like/miniconda3/envs/simo_sglang
PY="$ENV_ROOT/bin/python"
ORT_SRC=/softhome/like/package/onnxruntime
CUDA_HOME=/share_data/users/like/opt/cuda-13.0
CUDNN_HOME="$ENV_ROOT/lib/python3.12/site-packages/nvidia/cudnn"
BUILD_DIR=/share_data/users/like/build/onnxruntime-v1.27.0-cuda13

export CUDA_HOME CUDNN_HOME
export CUDNN_PATH="$CUDNN_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$ENV_ROOT/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDNN_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -x "$PY"
test -x "$ENV_ROOT/bin/cmake"
test -x "$ENV_ROOT/bin/ninja"
test -x "$CUDA_HOME/bin/nvcc"
test -f "$CUDA_HOME/include/cuda.h"
test -f "$CUDNN_HOME/include/cudnn.h"
test -f "$CUDNN_HOME/lib/libcudnn.so.9"
```

`CUDNN_HOME` 必须指向包含 `include/` 和 `lib/` 的目录，而不能只传 `.../nvidia/cudnn/lib`。ORT 的 build driver 在 Linux 上会显式检查 `--cuda_home` 和 `--cudnn_home` 是否存在；CMake 随后从该根目录查找头文件和 `libcudnn.so.9`。

构建和运行时都应让 CUDA 13 的 `$CUDA_HOME/lib64` 排在库搜索路径前面。不要混用 `/usr/local/cuda-12.8` 的库与该 CUDA 13 编译产物，否则同一进程可能加载不同版本的 `cudart`、`cublas`、`curand` 或 cuDNN，导致加载失败或不稳定结果。

### 79.4 构建 CUDA EP wheel

```bash
cd "$ORT_SRC"

"$PY" tools/ci_build/build.py \
  --config Release \
  --update \
  --build \
  --build_dir "$BUILD_DIR" \
  --cmake_path "$ENV_ROOT/bin/cmake" \
  --cmake_generator Ninja \
  --parallel 16 \
  --nvcc_threads 1 \
  --use_cuda \
  --cuda_version 13.0 \
  --cuda_home "$CUDA_HOME" \
  --cudnn_home "$CUDNN_HOME" \
  --build_wheel \
  --skip_tests
```

参数含义如下：

- `--update --build`：生成/更新 CMake 构建树后编译；该脚本在 native build 下默认也会这样做，这里显式写出以避免歧义。
- `--build_wheel`：生成名为 `onnxruntime_gpu-1.27.0-...whl` 的 Python wheel；`--use_cuda` 会使 wheel 包名为 `onnxruntime-gpu`。
- `--parallel 16 --nvcc_threads 1`：限制并发和每个 CUDA 编译任务的内部并发，降低 H100 主机上 CUDA 模板编译的峰值内存。资源充足时可以逐步提高 `--parallel`，不要一开始用 `--parallel 0`（所有 CPU 核）。
- `--skip_tests`：先完成构建和最小运行验证。需要完整 C++/Python test 时删除该参数，或者后续使用相同 `BUILD_DIR` 重新执行 `--test`。
- 未传 `--use_tensorrt`：明确构建 CUDA-only EP。这是最小且最适合先验证 SIMO 的配置。

如果以前曾在同一个 `$BUILD_DIR` 使用过其他 CUDA、cuDNN 或 generator，先删除这个专用构建目录再重新执行上面的命令：

```bash
rm -rf "$BUILD_DIR"
```

该命令只应作用于上文新建的 `onnxruntime-v1.27.0-cuda13` 目录，不能对 ORT 源码目录或不明的共享 build 目录执行。

### 79.5 安装生成的 wheel

构建成功后，不依赖固定的目录层级，直接查找产物并强制覆盖当前同版本的 `onnxruntime-gpu`：

```bash
WHEEL=$(find "$BUILD_DIR" -type f -path '*/dist/onnxruntime_gpu-*.whl' -print -quit)
test -n "$WHEEL"
printf 'wheel=%s\n' "$WHEEL"

"$PY" -m pip install --force-reinstall --no-deps "$WHEEL"
```

`--no-deps` 的目的不是跳过 CUDA 本体，而是避免 pip 为这个本地 wheel 重装或降级当前 conda 环境中的 NumPy、protobuf、flatbuffers 等已验证依赖。CUDA Toolkit 和 cuDNN 仍是动态链接依赖，必须保留第 79.3 节的运行时库路径或使用 `onnxruntime.preload_dlls()` 加载它们。

安装会替换当前的 `onnxruntime-gpu==1.27.0` 文件。开始前可记录当前 wheel 信息以便回退：

```bash
"$PY" -m pip show onnxruntime-gpu
```

若需要回退，重新安装此前保存的官方 wheel；不要在同一环境同时安装 `onnxruntime`（CPU 包）和 `onnxruntime-gpu`，两者会向同一 `onnxruntime` Python package 写文件。

### 79.6 最小验证：确认实际创建 CUDA session

下面的验证不只看 `get_available_providers()` 的编译期列表，还会对已有的 Silero 浮点 ONNX 创建 session，从而触发 CUDA EP 和动态库加载：

```bash
export MODEL=/share/users/like/package/jdjv/silero_vad_clean/onnx_float_baseline/silero_vad.onnx

"$PY" - <<'PY'
import os
import onnxruntime as ort
from onnxruntime.capi import build_and_package_info

ort.preload_dlls(cuda=True, cudnn=True)
print("onnxruntime:", ort.__version__)
print("wheel CUDA:", getattr(build_and_package_info, "cuda_version", None))
print("available:", ort.get_available_providers())

session = ort.InferenceSession(
    os.environ["MODEL"],
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print("active:", session.get_providers())
assert "CUDAExecutionProvider" in session.get_providers()
PY
```

成功标准是 `active:` 中包含 `CUDAExecutionProvider`。CUDA-only 重编译后不应再期待 `TensorrtExecutionProvider` 出现在列表中。可额外检查最终链接的库版本：

```bash
ORT_CAPI=$(
  "$PY" -c 'import onnxruntime.capi.onnxruntime_pybind11_state as s; from pathlib import Path; print(Path(s.__file__).resolve().parent)'
)
ldd "$ORT_CAPI/libonnxruntime_providers_cuda.so" | rg 'cudart|cublas|cudnn|curand|cufft'
```

其中 CUDA 13 库应解析到 `$CUDA_HOME`，cuDNN 应解析到 `$CUDNN_HOME/lib`。如果显示 `libcudnn.so.9 => not found`，优先检查当前 shell 的 `LD_LIBRARY_PATH` 和 `ort.preload_dlls(cuda=True, cudnn=True)`，而不是重新编译。

### 79.7 与 SIMO editable 安装的关系

SIMO 的 ONNX custom-op library 是 editable 安装时在 `simo/onnx/ort_plugin/` 下编译的；其构建代码使用随 SIMO 打包的 ONNX Runtime public headers，并只链接 CUDA driver library。因为本次 ORT 源码和当前安装版本同为 `1.27.0`，可以先直接运行现有 SIMO/Silero 测试验证，不需要预先重装 SIMO。

如果 `SessionOptions.register_custom_ops_library()` 报 ABI、符号或加载错误，或者同时修改了 ORT custom-op headers，再在相同 CUDA 13 环境下重建 editable SIMO：

```bash
cd /share/users/like/package/simo_conda_sglang
CUDA_HOME=/share_data/users/like/opt/cuda-13.0 \
  /share_data/users/like/miniconda3/envs/simo_sglang/bin/python -m pip install -e . --no-build-isolation
```

然后用已有的 `test_quant_onnx.sh` 做小规模 smoke test，再运行完整 24-shard 精度测试。这样可以将“ORT CUDA EP 能否加载”和“SIMO custom op 能否加载”分两步定位，避免一次完整评测后才发现动态库路径或 ABI 问题。

## 80. CUDA 13 ORT wheel 运行失败：cuDNN 8/9 链接错配

### 80.1 直接结论

本次失败的第一根因不是量化模型，也不是 SIMO 的 `Dequantize` 实现，而是新编译的 CUDA provider 动态库链接到了 cuDNN 8：

```text
libonnxruntime_providers_cuda.so: undefined symbol: cudnnGetLastErrorString
```

随后 ORT 打印：

```text
Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 13.*.
```

因为 CUDA EP 没有创建成功，ORT 只剩 CPU provider；量化 ONNX 中的 `Dequantize(2)` 节点又没有可用的 CPU 实现，于是最后出现：

```text
NOT_IMPLEMENTED: Could not find an implementation for Dequantize(2)
```

这个 `Dequantize(2)` 是 CUDA provider 加载失败后的连带错误，不是本次最先需要修复的问题。应先修复 `libonnxruntime_providers_cuda.so` 的 cuDNN 依赖。

### 80.2 证据链

失败 shard 日志 `/share/users/like/package/jdjv/silero_vad_clean/logs/out_no_lstm/onnx_quant_w_mxfp4_e2m1_a_mxfp4_e2m1_scale_int_sipu_aishell4_24shards/shards/shard_023.log` 中的顺序很明确：

1. `TryGetProviderInfo_CUDA` 加载 `libonnxruntime_providers_cuda.so` 失败；
2. 未定义符号是 `cudnnGetLastErrorString`；
3. CUDAExecutionProvider 创建失败；
4. session 初始化阶段才报告 `Dequantize(2)` 无实现。

安装后的 `readelf` 也直接显示该 provider 的 NEEDED 项是：

```text
libcublasLt.so.13
libcublas.so.13
libcufft.so.12
libcudart.so.13
libcudnn.so.8
```

而当前 conda 环境中的 cuDNN 9 库是：

```text
/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9
```

`ldd` 显示的实际加载对象是系统库：

```text
libcudnn.so.8 => /lib/x86_64-linux-gnu/libcudnn.so.8
```

该系统库是 cuDNN `8.9.7`，而 `cudnnGetLastErrorString` 是 cuDNN 9 中使用的符号；当前系统 cuDNN 8 只有旧的 `cudnnGetErrorString`。因此，动态 loader 找到了名为 `libcudnn.so.8` 的文件，却无法从其中解析编译代码所需要的 cuDNN 9 符号。

### 80.3 为什么编译时会得到 `libcudnn.so.8`

问题来自 CMake cache 中的两个不一致变量。当前构建目录 `/share_data/users/like/build/onnxruntime-v1.27.0-cuda13/Release/CMakeCache.txt` 显示：

```text
CUDNN_INCLUDE_DIR:PATH=/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/nvidia/cudnn/include
cudnn_LIBRARY:FILEPATH=/usr/lib/x86_64-linux-gnu/libcudnn.so
onnxruntime_CUDNN_HOME:UNINITIALIZED=/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/nvidia/cudnn
```

也就是说：

- 编译器看到的是 conda cuDNN 9 的头文件，所以生成了对 `cudnnGetLastErrorString` 的引用；
- linker 使用的是 CMake 之前缓存的系统 `/usr/lib/.../libcudnn.so`，该 symlink 指向 cuDNN 8；
- 最终生成了“cuDNN 9 headers + cuDNN 8 NEEDED”的错误混合物。

仅仅在构建脚本中设置 `CUDNN_HOME`，不能覆盖已经存在的 `cudnn_LIBRARY` CMake cache。`--update` 会复用这个 cache，不会自动清除已经选中的系统库。

此外，当前 `build-cuda.sh` 还有一个独立的 shell 参数错误：

```bash
--cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=86;90;120
```

未加引号的分号会被 Bash 当作命令分隔符，所以日志中出现：

```text
like-useful/build-cuda.sh: line 42: 90: command not found
```

构建本身实际只收到 `CMAKE_CUDA_ARCHITECTURES=86`，之后 shell 才尝试执行命令 `90` 并以返回码 127 退出。这个问题不是 cuDNN 错配的根因，但说明这次 build script 最终是失败的，即使 wheel 已经生成。

### 80.4 正确修复：新 build directory + 强制 cuDNN 9 library

不要给 `libcudnn.so.8` 建指向 `libcudnn.so.9` 的软链接，也不要用 `patchelf` 直接改 NEEDED。cuDNN 8 和 cuDNN 9 不是可以这样替换的 ABI，正确做法是让 CMake 从干净 cache 开始，并显式指定同一套 cuDNN 9 头文件和库。

先准备变量：

```bash
set -euo pipefail

ENV_ROOT=/share_data/users/like/miniconda3/envs/simo_sglang
PY="$ENV_ROOT/bin/python"
ORT_SRC=/share/users/like/package/onnxruntime
CUDA_HOME=/share_data/users/like/opt/cuda-13.0
CUDNN_HOME="$ENV_ROOT/lib/python3.12/site-packages/nvidia/cudnn"
BUILD_DIR=/share_data/users/like/build/onnxruntime-v1.27.0-cuda13-cuDNN9

export CUDA_HOME CUDNN_HOME
export CUDNN_PATH="$CUDNN_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$ENV_ROOT/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDNN_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -f "$CUDNN_HOME/include/cudnn.h"
test -f "$CUDNN_HOME/include/cudnn_version.h"
test -f "$CUDNN_HOME/lib/libcudnn.so.9"
```

然后使用独立目录编译。H100 是 `sm90`，本机所有 GPU 相同，因此只编译 `90` 即可；如果确实需要多个架构，必须把整个值放在引号中：

```bash
rm -rf "$BUILD_DIR"
cd "$ORT_SRC"

"$PY" tools/ci_build/build.py \
  --config Release \
  --update \
  --build \
  --build_dir "$BUILD_DIR" \
  --cmake_path "$ENV_ROOT/bin/cmake" \
  --cmake_generator Ninja \
  --parallel 16 \
  --nvcc_threads 1 \
  --use_cuda \
  --cuda_version 13.0 \
  --cuda_home "$CUDA_HOME" \
  --cudnn_home "$CUDNN_HOME" \
  --build_wheel \
  --skip_tests \
  --cmake_extra_defines \
    "CMAKE_CUDA_ARCHITECTURES=90" \
    "CUDNN_INCLUDE_DIR=$CUDNN_HOME/include" \
    "cudnn_LIBRARY=$CUDNN_HOME/lib/libcudnn.so.9"
```

这里使用小写的 `cudnn_LIBRARY` 是因为这份 ORT 源码的 `cmake/external/cuDNN.cmake` 使用的 cache 变量名就是 `cudnn_LIBRARY`。`CUDNN_INCLUDE_DIR` 和 `cudnn_LIBRARY` 必须成对指定，避免 headers 和 library 再次来自不同版本。

如果想保留多个架构，写法必须是：

```bash
--cmake_extra_defines \
  'CMAKE_CUDA_ARCHITECTURES=86;90;120' \
  "CUDNN_INCLUDE_DIR=$CUDNN_HOME/include" \
  "cudnn_LIBRARY=$CUDNN_HOME/lib/libcudnn.so.9"
```

但对于当前全是 H100 的机器，`90` 更快、更明确。`parallel=100` 也没有必要，建议先用 16；它不是当前 cuDNN 错误的原因，但会明显提高编译内存压力。

### 80.5 安装前检查 wheel，避免再次安装错误产物

编译完成后，先检查 CMake cache 和构建产物，确认 NEEDED 已经变成 cuDNN 9：

```bash
rg -n 'CUDNN_INCLUDE_DIR|cudnn_LIBRARY|CMAKE_CUDA_ARCHITECTURES' \
  "$BUILD_DIR/Release/CMakeCache.txt"

CUDA_PROVIDER=$(find "$BUILD_DIR/Release" -name libonnxruntime_providers_cuda.so -type f -print -quit)
test -n "$CUDA_PROVIDER"
readelf -d "$CUDA_PROVIDER" | rg 'NEEDED.*cudnn|RPATH|RUNPATH'
```

预期结果应类似：

```text
CUDNN_INCLUDE_DIR=.../site-packages/nvidia/cudnn/include
cudnn_LIBRARY=.../site-packages/nvidia/cudnn/lib/libcudnn.so.9
CMAKE_CUDA_ARCHITECTURES=90
Shared library: [libcudnn.so.9]
```

如果这里仍然显示 `/usr/lib/x86_64-linux-gnu/libcudnn.so` 或 `libcudnn.so.8`，不要安装 wheel，说明 cache 或 `--cmake_extra_defines` 仍未生效。

确认无误后再安装：

```bash
WHEEL=$(find "$BUILD_DIR" -type f -path '*/dist/onnxruntime_gpu-*.whl' -print -quit)
test -n "$WHEEL"

"$PY" -m pip install --force-reinstall --no-deps "$WHEEL"
```

安装后检查的预期是：

```bash
ORT_CAPI=$(
  "$PY" -c 'import onnxruntime.capi.onnxruntime_pybind11_state as s; from pathlib import Path; print(Path(s.__file__).resolve().parent)'
)
ldd "$ORT_CAPI/libonnxruntime_providers_cuda.so" | rg 'cudart|cublas|cudnn|cufft'
```

其中必须看到：

```text
libcudart.so.13 => .../cuda-13.0/...
libcublas.so.13 => .../cuda-13.0/...
libcudnn.so.9 => .../site-packages/nvidia/cudnn/lib/...
```

`ldd` 如果仍解析到 `libcudnn.so.8`，说明安装的还是旧 wheel 或新 wheel 仍然错误链接；不要继续运行完整量化测试。

### 80.6 重新运行前的最小 provider 验证

先在包含 CUDA 13 和 cuDNN 9 的同一 shell 中创建一个真正的 CUDA session：

```bash
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDNN_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MODEL=/share/users/like/package/jdjv/silero_vad_clean/onnx_float_baseline/silero_vad.onnx

"$PY" - <<'PY'
import os
import onnxruntime as ort

ort.preload_dlls(cuda=True, cudnn=True)
print("version:", ort.__version__)
print("available:", ort.get_available_providers())
session = ort.InferenceSession(
    os.environ["MODEL"],
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print("active:", session.get_providers())
assert "CUDAExecutionProvider" in session.get_providers()
PY
```

只有当 `active:` 中包含 `CUDAExecutionProvider` 后，才重新运行 `test_quant_onnx.sh`。否则量化模型会再次因为 CPU provider 不支持 SIMO `Dequantize` 而报告误导性的 `Dequantize(2)` 错误。

### 80.7 本次构建问题总结

| 现象 | 实际原因 | 修复 |
| --- | --- | --- |
| `undefined symbol: cudnnGetLastErrorString` | cuDNN 9 头文件与 cuDNN 8 动态库混用 | 清理 CMake cache，显式指定 `cudnn_LIBRARY=.../libcudnn.so.9` |
| `ldd` 显示 `libcudnn.so.8 => /lib/...` | provider 的 DT_NEEDED 已经写死为 `.so.8`，且系统库被加载 | 重新链接；只改 `LD_LIBRARY_PATH` 不足以把 `.8` 变成 `.9` |
| `Dequantize(2) NOT_IMPLEMENTED` | CUDA EP 先加载失败，CPU EP 无该量化节点实现 | 修复 CUDA EP 后再判断模型/custom op 问题 |
| `line 42: 90: command not found` | `CMAKE_CUDA_ARCHITECTURES=86;90;120` 的分号未引用 | 写成 `"CMAKE_CUDA_ARCHITECTURES=90"` 或整体单引号包裹 |
| build log 显示 `cmake_extra_defines=['CMAKE_CUDA_ARCHITECTURES=86']` | shell 只把 `86` 传给 build.py | 使用 H100 的 `90`，并检查 `CMakeCache.txt` |

## 81. `build-cuda-verbose.sh` 审查与本地 `_deps` 复用

### 81.1 结论

`/share/users/like/package/onnxruntime/like-useful/build-cuda-verbose.sh` 的 CUDA/cuDNN 核心修复是正确的：

- `CUDNN_INCLUDE_DIR` 和小写 `cudnn_LIBRARY` 都显式指向 conda 环境的 cuDNN 9；
- `CMAKE_CUDA_ARCHITECTURES=86;90;120` 已整体加引号，不会再被 Bash 拆成 `90`、`120` 两个命令；
- `bash -n build-cuda-verbose.sh` 已通过；
- `like_debug_verbose=1` 会触发本地对 `tools/ci_build/build.py` 的修改，使 `cmake --build` 带 `--verbose`，这与脚本名称一致。

但是，不建议把旧目录的整个 `_deps` 直接复制到新 build tree。应复用其中的 `*-src` 下载源码，不能复用 `*-build`、`*-subbuild` 和 `*-populate-prefix` 等生成目录。最稳妥的方式甚至不需要复制：通过 CMake 的 `FETCHCONTENT_SOURCE_DIR_<NAME>` 直接让新 build tree 使用旧 `_deps` 中已经下载好的源码。

### 81.2 脚本检查结果

脚本当前关键部分等价于：

```bash
--cmake_extra_defines \
  "CMAKE_CUDA_ARCHITECTURES=86;90;120" \
  "CUDNN_INCLUDE_DIR=$CUDNN_HOME/include" \
  "cudnn_LIBRARY=$CUDNN_HOME/lib/libcudnn.so.9"
```

这三项会作为三个独立的 `-D` CMake cache 定义传给 `build.py`，语法正确。相对于上次脚本，cuDNN 8/9 混用和未引用分号这两个问题都已修正。

仍有以下注意点：

| 级别 | 位置 | 问题和建议 |
| --- | --- | --- |
| 中 | 文件首行 | 没有 `#!/usr/bin/env bash`。用 `bash build-cuda-verbose.sh` 可以运行；若直接执行 `./build-cuda-verbose.sh` 会得到 exec format error。建议添加 shebang。 |
| 中 | 第 36 行 | `--parallel 100` 会让 100 个编译任务并发，CUDA 模板编译的内存压力很高。首次验证建议 `--parallel 16 --nvcc_threads 1`，确认内存余量后再增加。 |
| 低 | 第 45 行 | 本机 GPU 全部是 H100（sm90）；`86;90;120` 会显著增加 CUDA 编译时间和 wheel 大小。只服务本机时使用 `"CMAKE_CUDA_ARCHITECTURES=90"`。确有 Ada/Blackwell 部署需求时才保留多架构列表。 |
| 低 | 第 2、28 行 | `set -x` 加上 `like_debug_verbose=1` 会输出 shell 展开结果和所有编译命令，日志会很大，但不影响正确性。`like_debug_verbose` 依赖当前对 `tools/ci_build/build.py` 的本地修改；没有该修改时它只是无效环境变量。 |
| 中 | 构建完成后 | 脚本未自动检查最终 provider 是否 `NEEDED libcudnn.so.9`。在安装 wheel 前仍应执行第 80.5 节的 `readelf` 与 `CMakeCache.txt` 检查。 |

源码仓库的 git submodule 已全部检出。若确认不希望 `build.py --update` 额外执行 git submodule 同步，可增加 `--skip_submodule_sync`；不要在 submodule 缺失时添加该参数。

### 81.3 为什么不能整目录复制 `_deps`

旧目录是：

```text
/share/users/like/build/onnxruntime-v1.27.0-cuda13/Release/_deps
```

新目录是：

```text
/share/users/like/package/onnxruntime/build/onnxruntime-v1.27.0-cuda13/Release/_deps
```

两者虽然都位于同一个 NFS 文件系统上（`/share/users/like/build` 和 `/share_data/users/like/build` 经 `stat` 验证为同一个目录），但它们是不同的 CMake build directory。

旧 `_deps` 约 586 MiB，除下载源码外还包含各依赖的 build output、populate subbuild 和 CMake cache。例如旧 `cutlass-subbuild/CMakeCache.txt` 中记录了：

```text
CMAKE_CACHEFILE_DIR=/share_data/users/like/build/onnxruntime-v1.27.0-cuda13/Release/_deps/cutlass-subbuild
CMAKE_HOME_DIRECTORY=/share_data/users/like/build/onnxruntime-v1.27.0-cuda13/Release/_deps/cutlass-subbuild
```

这些绝对路径、generator、编译器和先前的 CMake 选择不能移植到新目录。更关键的是旧 build 的顶层 cache 正是 cuDNN 错配的来源：它缓存了 `/usr/lib/x86_64-linux-gnu/libcudnn.so`（cuDNN 8）。整树复制会把该错误状态和旧构建产物一同带入新 build。

可复用的是纯源码目录：

```text
*-src
```

不能复制或复用的是：

```text
*-build
*-subbuild
*-populate-prefix
顶层 CMakeCache.txt、CMakeFiles、wheel、已编译 .so/.a
```

当前目标 `_deps` 中已经有 17 个同名 `*-src` 目录，但并非全部完整。最明显的是：

```text
旧 cutlass-src: 165 MiB
新 cutlass-src:   4 KiB
```

其余常见依赖源码（abseil、onnx、protobuf、flatbuffers、Eigen 等）大小已基本一致。`_deps_copy` 当前是空目录，不能作为可用缓存。

### 81.4 推荐方案：不复制，直接指定本地 FetchContent 源码

CMake 的官方 `FETCHCONTENT_SOURCE_DIR_<UPPERCASE_NAME>` 变量会让 FetchContent 使用指定的本地目录，并且不进行 download 或 update；每个依赖的 binary directory 仍会在新的 `$BUILD_DIR` 内生成。这正好符合“复用已下载源码、但不继承旧 cache”的需求。

先在脚本中定义旧源码缓存和自动生成的 CMake 定义：

```bash
DEPS_SOURCE=/share/users/like/build/onnxruntime-v1.27.0-cuda13/Release/_deps

FETCHCONTENT_DEFINES=()
for source_dir in "$DEPS_SOURCE"/*-src; do
  test -f "$source_dir/CMakeLists.txt"
  dependency_name=${source_dir##*/}
  dependency_name=${dependency_name%-src}
  dependency_name=${dependency_name^^}
  FETCHCONTENT_DEFINES+=(
    "FETCHCONTENT_SOURCE_DIR_${dependency_name}=$source_dir"
  )
done
```

这会生成例如：

```text
FETCHCONTENT_SOURCE_DIR_CUTLASS=.../_deps/cutlass-src
FETCHCONTENT_SOURCE_DIR_PROTOBUF=.../_deps/protobuf-src
FETCHCONTENT_SOURCE_DIR_EIGEN3=.../_deps/eigen3-src
```

然后把原来的 `--cmake_extra_defines` 替换为下面这一段。H100-only 的 `90` 也一并采用：

```bash
"$PY" tools/ci_build/build.py \
  --config Release \
  --update \
  --build \
  --build_dir "$BUILD_DIR" \
  --cmake_path "$ENV_ROOT/bin/cmake" \
  --cmake_generator Ninja \
  --parallel 16 \
  --nvcc_threads 1 \
  --use_cuda \
  --cuda_version 13.0 \
  --cuda_home "$CUDA_HOME" \
  --cudnn_home "$CUDNN_HOME" \
  --build_wheel \
  --skip_tests \
  --skip_submodule_sync \
  --cmake_extra_defines \
    "CMAKE_CUDA_ARCHITECTURES=90" \
    "CUDNN_INCLUDE_DIR=$CUDNN_HOME/include" \
    "cudnn_LIBRARY=$CUDNN_HOME/lib/libcudnn.so.9" \
    "${FETCHCONTENT_DEFINES[@]}"
```

使用这个方案前，新 build directory 应没有顶层 CMake cache。当前 target directory 没有顶层 `Release/CMakeCache.txt`，但有不完整的 `_deps` 生成状态；为完全隔离旧状态，可删除仅属于新 build 的 `Release` 目录后再运行：

```bash
rm -rf "$BUILD_DIR/Release"
```

这不会删除 `DEPS_SOURCE` 中的旧下载源码，因为两者是不同目录。CMake 会从 `DEPS_SOURCE/*-src` 读取源码，并在新的 `Release` 目录重新生成所有 `*-build`/`*-subbuild`。

不要在首次新配置时依赖 `FETCHCONTENT_FULLY_DISCONNECTED=ON`。当前 CMake 的 FetchContent 文档明确说明，该开关不适合作为 first configure 的“禁止网络”手段；`FETCHCONTENT_SOURCE_DIR_*` 是官方提供的本地源码覆盖机制，更可靠。

### 81.5 备选方案：只同步 `*-src`

如果要求新 build tree 完全自包含，可以复制源码目录，但只复制 `*-src`。不要执行 `cp -a "$DEPS_SOURCE" "$BUILD_DIR/Release/"` 或直接复制整个 `_deps`。

```bash
DEPS_SOURCE=/share/users/like/build/onnxruntime-v1.27.0-cuda13/Release/_deps
DEPS_TARGET=/share/users/like/package/onnxruntime/build/onnxruntime-v1.27.0-cuda13/Release/_deps

mkdir -p "$DEPS_TARGET"
for source_dir in "$DEPS_SOURCE"/*-src; do
  target_dir="$DEPS_TARGET/${source_dir##*/}"
  rsync -a --delete "$source_dir/" "$target_dir/"
done
```

上述命令会修复当前不完整的 `cutlass-src`，并保持每个依赖源码与旧缓存一致。复制完成后，仍要使用第 81.4 节的 `FETCHCONTENT_SOURCE_DIR_*` 定义，只需把 `DEPS_SOURCE` 改成 `$DEPS_TARGET`。这样 CMake 不会尝试重新下载或更新它们。

如果只想补齐目前确认缺失的 CUTLASS，以下命令足够：

```bash
rsync -a --delete \
  /share/users/like/build/onnxruntime-v1.27.0-cuda13/Release/_deps/cutlass-src/ \
  /share/users/like/package/onnxruntime/build/onnxruntime-v1.27.0-cuda13/Release/_deps/cutlass-src/
```

但“直接引用旧 `DEPS_SOURCE`”更省空间，也更不容易把半完成目录误当作有效依赖。

### 81.6 运行前检查

在安装 wheel 前，至少确认以下四项：

```bash
rg -n 'CUDNN_INCLUDE_DIR|cudnn_LIBRARY|CMAKE_CUDA_ARCHITECTURES' \
  "$BUILD_DIR/Release/CMakeCache.txt"

CUDA_PROVIDER=$(find "$BUILD_DIR/Release" -name libonnxruntime_providers_cuda.so -type f -print -quit)
readelf -d "$CUDA_PROVIDER" | rg 'NEEDED.*cudnn'
```

预期为：

```text
CUDNN_INCLUDE_DIR=.../nvidia/cudnn/include
cudnn_LIBRARY=.../nvidia/cudnn/lib/libcudnn.so.9
CMAKE_CUDA_ARCHITECTURES=90
Shared library: [libcudnn.so.9]
```

只有这四项成立后，才安装新 wheel 并用第 80.6 节的真实 CUDA session 验证。这样可以在耗时的 Silero 24-shard 运行之前，排除 cuDNN 链接、下载缓存和 GPU 架构配置问题。

## 82. 在 ONNX Runtime C++ 代码中打日志

### 82.1 先区分两种代码位置

当前工程有两套不同的日志接口，不能混用：

| 修改位置 | 推荐接口 | 原因 |
| --- | --- | --- |
| `/softhome/like/package/onnxruntime/onnxruntime/...` 内部 ORT C++/CUDA host 代码 | `LOGS`、`LOGF`、`LOGS_IF` | 使用 ORT 私有的 `onnxruntime::logging::Logger`，能进入 session/default logger。 |
| `simo/onnx/ort_plugin/*.cc` 外部 custom-op 动态库 | `Ort::Logger`、`ORT_CXX_LOG*_NOEXCEPT` | 使用稳定的 ORT C/C++ public API，不依赖 ORT 内部 logging ABI。 |

内部宏定义在 `include/onnxruntime/core/common/logging/logging.h` 和 `macros.h`；外部 custom-op 的公开 logger 定义在 `include/onnxruntime/core/session/onnxruntime_cxx_api.h`。即使 SIMO wheel 中带有部分 ORT internal header，也不应让外部 `.so` 依赖内部 `LOGS` 实现。

### 82.2 ORT 核心源码：使用 `LOGS`

在 ORT 内部 `.cc` 或 `.cu` 的 host 代码中包含：

```cpp
#include "core/common/logging/logging.h"
```

优先传递已有的 `const logging::Logger&`，而不是自行取默认 logger：

```cpp
common::Status CheckSomething(const Node& node,
                              const logging::Logger& logger) {
  LOGS(logger, INFO) << "SIMO debug: node=" << node.Name()
                     << ", op=" << node.OpType();

  LOGS_IF(node.InputDefs().empty(), logger, WARNING)
      << "SIMO debug: node has no inputs: " << node.Name();

  if (node.OpType() == "Conv") {
    LOGF(logger, INFO, "SIMO debug: Conv node index=%d",
         static_cast<int>(node.Index()));
  }
  return common::Status::OK();
}
```

常用宏如下：

| 宏 | 用途 |
| --- | --- |
| `LOGS(logger, INFO) << ...` | 流式日志，最常用。 |
| `LOGF(logger, INFO, "x=%d", x)` | printf 风格；内部实现的格式化日志消息上限约为 2 KiB。 |
| `LOGS_IF(condition, logger, WARNING) << ...` | 条件日志，避免 stream-style 宏与未加花括号的 `if/else` 结合时的语法歧义。 |
| `LOGS_CATEGORY(logger, INFO, "simo.qdq") << ...` | 指定自定义 category；默认 category 是 `onnxruntime`。 |
| `LOGS_USER(...)` / `LOGF_USER(...)` | 可能含用户数据或 PII 时使用，sink 可按 user-data policy 过滤。 |

如果当前类没有 logger 参数，但已处于正常 ORT 生命周期中，可以使用：

```cpp
LOGS_DEFAULT(INFO) << "SIMO debug: CUDA provider initialized";
```

`LOGS_DEFAULT` 依赖一个有效的 `LoggingManager::DefaultLogger()`。它适合 provider 初始化等已有默认 ORT environment 的位置；在可获得 session、graph 或 execution-provider logger 时，仍应优先使用显式 logger。当前 CUDA EP 本身也采用这个模式，例如 `cuda_execution_provider.cc` 在启动时用 `LOGS_DEFAULT(INFO)` 输出 cuDNN 版本，在 capability/graph 路径中使用传入的 `logger` 或 `*GetLogger()`。

### 82.3 等级、过滤与 `VLOGS` 陷阱

内部等级从低到高为：

```text
VERBOSE = 0, INFO = 1, WARNING = 2, ERROR = 3, FATAL = 4
```

logger 的 severity 是“最低输出等级”。默认通常是 `WARNING`，因此新增的 `INFO` 和 `VERBOSE` 日志默认看不到，而 `WARNING`/`ERROR` 可见。

不要把希望在当前 Release wheel 中看到的日志写成：

```cpp
VLOGS(logger, 1) << "...";
```

`VLOGS`/`VLOGF` 只在非 `NDEBUG` 的 Debug build 中生效；Release build 的宏会在编译时丢弃日志表达式。对于需要在 Release 中按日志等级打开的调试信息，使用：

```cpp
LOGS(logger, VERBOSE) << "SIMO debug: qdq shape=" << outer_dim << "x" << quant_dim;
```

然后在调用方将 severity 设为 `VERBOSE`。只有在专门编译 Debug ORT 并需要分级 VLOG 时，再同时设置 verbosity level，例如 `VLOGS(logger, 1)` 需要 logger 的 max vlog level 至少为 1，并且 severity 为 `VERBOSE`。

不要在每个 kernel、每个 audio chunk 或每次 QDQ 调用无条件打印 INFO/VERBOSE。当前 Silero 全量评测有 1,431,607 个 chunk；这种日志会严重拖慢推理并淹没关键错误。应通过环境变量、首 N 次计数或特定 node name 进行 gate，并只记录 shape、dtype、spec、stream/device 等必要元数据，不记录完整 tensor 内容。

### 82.4 让内部日志在 C++ 调用方可见

一个独立 C++ 调用方可以在创建 Env 和 SessionOptions 时设置日志级别：

```cpp
#include <onnxruntime_cxx_api.h>

Ort::Env env{ORT_LOGGING_LEVEL_VERBOSE, "simo-debug"};
Ort::SessionOptions options;
options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);

// 此版本 C++ wrapper 没有 SetLogVerbosityLevel 包装；直接调用 C API。
Ort::ThrowOnError(
    Ort::GetApi().SetSessionLogVerbosityLevel(options, 1));
```

`Ort::SessionOptions` 可隐式转换为 `OrtSessionOptions*`，所以上述 C API 调用可以直接传入 `options`。若只用 `LOGS(..., INFO/VERBOSE)` 而没有 `VLOGS`，verbosity level 无需设置，关键是 severity 必须为 0。

如需把日志写入自己的 sink，可在创建 Env 时传入 `OrtLoggingFunction`：

```cpp
static void ORT_API_CALL MyOrtLog(
    void* /*param*/, OrtLoggingLevel severity, const char* category,
    const char* logid, const char* code_location, const char* message) {
  std::fprintf(stderr, "[ORT %d][%s][%s] %s: %s\n",
               static_cast<int>(severity), category, logid, code_location, message);
}

Ort::Env env{ORT_LOGGING_LEVEL_VERBOSE, "simo-debug", MyOrtLog, nullptr};
```

日志回调不能抛出异常，也不应在回调中再次调用可能触发 ORT 日志的 API，以免递归。 

### 82.5 当前 Python Silero 评测器如何显示 C++ 日志

当前 `evaluate_silero_vad_public_v5.py` 的 `SileroVadOnnx.__init__()` 会创建：

```python
opts = ort.SessionOptions()
```

实际检查得到这个对象默认 `log_severity_level == -1`、`log_verbosity_level == 0`。`-1` 表示继承默认 logger，而默认环境等级通常是 `WARNING`。为了让当前脚本中 ONNX Runtime C++ 的 `INFO`/`VERBOSE` 日志进入每个 shard 日志，应在创建 `opts` 后、`InferenceSession` 前加入：

```python
ort.set_default_logger_severity(0)
ort.set_default_logger_verbosity(1)  # 仅 Debug ORT 的 VLOGS 需要

opts.log_severity_level = 0
opts.log_verbosity_level = 1         # 仅 Debug ORT 的 VLOGS 需要
```

只关心普通 `LOGS(..., INFO)` 时，`opts.log_severity_level = 0` 已足够；`log_verbosity_level` 不会让 Release 中的 `VLOGS` 重新出现。修改该 Python 脚本后，24 个 shard 子进程会各自把 ORT C++ 日志写入对应的 `shards/shard_*.log`，因为调度脚本已将子进程 stdout/stderr 重定向到这些文件。

也可以仅在试验入口最早处执行：

```python
import onnxruntime as ort
ort.set_default_logger_severity(0)
```

由于当前 SessionOptions 默认 severity 为 `-1`，它会继承这个默认等级；但显式设置 `opts.log_severity_level = 0` 更清晰，也不会依赖环境默认值。

### 82.6 SIMO 外部 custom op：使用公开 `Ort::Logger`

`simo/onnx/ort_plugin/simo_qdq_ops.cc` 不是 ORT 内部目标，而是通过 `RegisterCustomOps` 加载的外部 shared library。应使用公开 C++ API：

```cpp
Ort::KernelContext kernel_context{&context};
const Ort::Logger logger = kernel_context.GetLogger();

ORT_CXX_LOGF_NOEXCEPT(
    logger, ORT_LOGGING_LEVEL_VERBOSE,
    "com.simo::Quantize: outer=%lld K=%lld packed=%lld block=%lld",
    static_cast<long long>(outer_dim),
    static_cast<long long>(quant_dim),
    static_cast<long long>(packed_dim),
    static_cast<long long>(spec_->block_size));
```

这里的 `context` 是当前 `QuantizeCustomOp::Compute(OrtKernelContext& context, ...)` 或 `DequantizeCustomOp::Compute(...)` 参数，所以 `&context` 可以直接传给 `Ort::KernelContext`。`KernelContext::GetLogger()` 最终调用公开的 `OrtApi::KernelContext_GetLogger`，得到与本次 session/run 相同的 logger。

在 custom-op `Compute` 内优先用 `ORT_CXX_LOG_NOEXCEPT` 或 `ORT_CXX_LOGF_NOEXCEPT`。普通 `ORT_CXX_LOG*` 在底层日志 API 返回错误或格式化失败时会抛出 `Ort::Exception`；`NOEXCEPT` 版本会忽略日志自身的失败，不会让调试输出改变推理控制流。

若需要在 kernel 构造阶段记录静态属性，可从现有的：

```cpp
Ort::ConstKernelInfo kernel_info{info};
```

取得：

```cpp
const Ort::Logger logger = kernel_info.GetLogger();
ORT_CXX_LOG_NOEXCEPT(
    logger, ORT_LOGGING_LEVEL_INFO,
    "com.simo QDQ kernel created");
```

不要把这个 logger 保存到超过 kernel/session 生命周期的全局对象中。构造阶段日志适合记录 semantic QDQ config；运行阶段日志必须限制频率。更稳妥的做法是在 `Compute` 里按当前 context 取得 logger，并用一个明确的环境变量（例如 `SIMO_ONNX_LOG_QDQ=1`）控制是否打印。

### 82.7 CUDA device 代码的边界

`LOGS` 和 `Ort::Logger` 都只能从 host C++ 路径调用，不能在 `__global__`/`__device__` CUDA kernel 内调用。排查 device kernel 时：

- 优先在 launch 前后用 host 日志输出 grid、block、shape、stream、runtime spec 和 `CUresult`；
- 临时使用 CUDA device `printf` 只适合极小输入和短期调试，必须在 host 侧同步 stream 后才能稳定看到输出，且会明显影响性能；
- 不要把 device `printf` 留在量化或 Silero 全量评测代码中；
- CUDA API 失败应返回/传播 `Ort::Status`，并在 host 侧记录错误码和 `cuGetErrorString`，当前 `SyncAfterLaunchIfRequested()` 已是这种模式。

### 82.8 重新编译边界

- 修改 ORT 内部 `onnxruntime/...` C++/CUDA 代码后：重新构建并安装 ONNX Runtime wheel；只有安装后的 `libonnxruntime*.so` 才会包含日志改动。
- 修改 `simo/onnx/ort_plugin/*.cc` 后：重新执行 editable SIMO 安装或其 wheel 构建，使 `libSimoOnnxCustomOps_sm90.so` 重编译；不需要为了这类插件日志重编译 ORT。
- 运行日志先用一个文件、一个 shard 的小规模测试验证，再启动完整 24-shard 作业，避免每个子进程同时输出大量调试信息。

## 83. 2026-07-31：仅修改 `dynamic_quantize_lstm.cc` 后，为什么 Debug 构建仍编译大量其他文件

### 83.1 结论

这不是正常的“修改一个 `.cc`，因此它的所有 C++ 源文件都必须重编译”，也不是 CUDA 架构、Debug/Release 或编译参数变化导致的。根因是 **Ninja build tree 放在 `/share` 的 NFS4 挂载上后，构建期间的输出时间戳记录不一致**。Ninja 因而无法可靠判断输出是否新于输入，出现了大面积误判 dirty、重复编译、以及构建结束后仍需重链接的现象。

`--update` 确实会让 `build.py` 每次重新执行 CMake 和 submodule sync，增加了触发面和时间成本；但它不是本次 859 个 C/C++ 编译的直接依赖原因。日志中同一普通 CPU 源文件在两次构建的完整编译命令完全一致，Ninja 的 command hash 也相同。若是正常 CMake 配置或 flags 改变，二者不会相同。

因此，当前 NFS 上的 Debug build tree 不应再作为可复现的增量构建基础。应在本机本地磁盘 `/data` 重建 Debug build tree；`/share_data` 不是替代方案，因为它和 `/share` 实际是同一个 NFS export。

### 83.2 两次日志的事实

两次命令都使用相同脚本、相同 `BUILD_CONFIG=Debug`、相同 CUDA/cuDNN/FetchContent 参数：

```bash
BUILD_CONFIG=Debug bash like-useful/build-cuda-verbose.sh > temp/build.verbose.debug.log.`nowstr.sh` 2>&1 &
```

| 构建 | 日志 | Ninja 总步骤 | 实际 C/C++ 编译命令数 | 结果 |
|---|---|---:|---:|---|
| 第一次 | `build.verbose.debug.log.2026_07_31___14_38_22` | 2,352 | 2,222 | 初次完整 Debug 构建，14:54:02 显示 `Build complete` |
| 第二次 | `build.verbose.debug.log.2026_07_31___15_20_12` | 982 | 859 | 不是全量，但远大于只应重编译的 `dynamic_quantize_lstm.cc` |

第二次构建确实编译了目标文件本身，日志第 445 个 Ninja task 是：

```text
... -o CMakeFiles/onnxruntime_providers.dir/.../dynamic_quantize_lstm.cc.o \
    -c .../onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc
```

同时它还重新编译了大部分 CPU/provider/graph/optimizer/session/test 目标，例如：

| target | 第二次重新编译的对象数 |
|---|---:|
| `onnxruntime_providers` | 222 |
| `onnxruntime_provider_test` | 227 |
| `onnxruntime_optimizer` | 96 |
| `onnxruntime_test_all` | 83 |
| `onnxruntime_framework` | 58 |
| `onnxruntime_graph` | 37 |
| `onnxruntime_session` | 33 |

反过来，第一次构建中的 CUDA 编译对象没有在第二次重编译：第一次有 `onnxruntime_providers_cuda` 528 个、`onnxruntime_providers_cuda_flash_attention` 48 个；第二次没有这些 CUDA source compile 命令，只发生了后续 provider `.so` 链接。这说明第二次不是 clean build，也说明“加一个 CPU 日志导致 CUDA 全部重编译”的解释不成立。

对未修改的 `onnxruntime/core/providers/cpu/activation/activations.cc`，两次日志中完整 `/usr/bin/c++ ... -c ...` 命令的 SHA-256 都是：

```text
423126ab022cc6ab3ec3deb3f268f22b029a1f2e0db75469f4e63aef3cb644f6
```

`dynamic_quantize_lstm.cc` 的改动也仅为包含 `core/common/logging/logging.h` 及一条 `LOGS_DEFAULT(INFO)`，Git worktree 没有显示其他受影响的 `.cc`/`.h` 改动。因此常见的“公共头文件被修改”原因可以排除。

### 83.3 直接证据：Ninja 的输出时间戳已经不可信

实际 build tree 是：

```text
/share/users/like/package/onnxruntime/build/onnxruntime-v1.27.0-cuda13/Debug
```

`findmnt` 显示它位于：

```text
10.97.128.245:/share  nfs4  rw,...,local_lock=none,...
```

`/share_data` 也解析到同一 `10.97.128.245:/share` NFS4 export。因此把 `BUILD_DIR` 从 `/share/...` 改成 `/share_data/...` 不会消除这个问题。

第二次构建的 `.ninja_log` 有 **981 个不同输出** 被记录成完全同一个 mtime：

```text
1785482423594883374  # 2026-07-31 15:20:23.594883374 +0800
```

其中既有早期的 `libonnxruntime_providers_shared.so`、Abseil 静态库，也有最终的 `onnxruntime_pybind11_state.so`、`onnxruntime_provider_test`。这些命令显然不可能在数分钟构建中全部得到同一个纳秒级文件 mtime。

进一步检查时，Ninja 自己给出了典型的错误状态：静态库的“recorded mtime”早于它所依赖对象的 mtime，例如：

```text
recorded mtime of libonnxruntime_common.a older than most recent input
CMakeFiles/onnxruntime_common.dir/.../logging.cc.o
(1785482423594883374 vs 1785482468212143635)
```

这会让下一次 `ninja` 即使不再编辑源代码，仍计划额外的链接或生成任务。它也解释了为什么第二次构建里能看到大量 archive/link 命令：Ninja 用不一致的 metadata 安排了图中的 dirty edges。这个现象不能通过“CMake 没有增量编译”来概括；CMake 生成的是正确的依赖图，而 Ninja 获取到的 NFS 元数据使该图的时间戳判断失真。

NFS 的 attribute cache/close-to-open 一致性不适合存放大量、高并发、频繁替换输出文件的 Ninja build tree。此脚本还设置了 `--parallel 100`，会同时创建、删除和 `stat` 大量 `.o`、`.a`、`.so`，显著放大这一问题。

### 83.4 `--update` 与 `--skip_tests` 的实际含义

当前 `like-useful/build-cuda-verbose.sh` 固定传入：

```bash
--update \
--build \
--build_wheel \
--skip_tests
```

在 `tools/ci_build/build.py` 中：

- `--update` 会执行 `git submodule sync --recursive`、`git submodule update --init --recursive`，随后执行 `generate_build_tree()`，即每次都重新调用 CMake；
- `--build` 才调用 `cmake --build <BUILD_DIR>/Debug`；
- `--build_wheel` 在 `--build` 成功后打包 wheel；
- `--skip_tests` 只是不运行测试，**不等于不构建默认 `all` target 中的测试可执行文件**。

所以即使没有本次 NFS 元数据问题，该“更新、构建全部目标、打 wheel”的命令也不适合每加一行日志就执行一次。它适合首次配置和最终发布 wheel；编辑-编译-观察日志的循环应复用已配置的本地 build tree，并避免 `--update`。

### 83.5 推荐修复：把 build output 放到本机 `/data`

本机检查结果：

```text
/data  ext4  约 12T 可用
/share 和 /share_data  均为 10.97.128.245:/share nfs4
当前 Debug build tree 大小约 19G
```

因此 `/data` 有足够容量，且是本地 ext4。保留源码和下载缓存仍在 NFS 没有问题，关键是 CMake/Ninja 的 build output、`.ninja_log`、`.ninja_deps`、对象文件和临时 archive 必须在本地磁盘。

建议把脚本中的硬编码：

```bash
BUILD_DIR=$ORT_SRC/build/onnxruntime-v1.27.0-cuda13
```

改为可由环境覆盖的本地默认路径：

```bash
BUILD_DIR=${BUILD_DIR:-/data/like/build/onnxruntime-v1.27.0-cuda13}
```

并先从干净的本地 Debug tree 开始。旧的 NFS Debug tree 时间戳已经不自洽，不应复制其中的 `Debug/_deps/*-build`、`CMakeCache.txt`、`.ninja_*` 或对象文件到新目录。可继续复用只读依赖源码：

```bash
DEPS_SOURCE=/share/users/like/build/onnxruntime-v1.27.0-cuda13/Release/_deps
```

即当前脚本生成的 `FETCHCONTENT_SOURCE_DIR_<NAME>=.../*-src` 参数可以保留；这些只是 source override，新的 `_deps/*-build` 仍会在本机 `$BUILD_DIR/Debug` 中生成。

第一次本地配置/构建使用：

```bash
BUILD_CONFIG=Debug \
BUILD_DIR=/data/like/build/onnxruntime-v1.27.0-cuda13 \
bash like-useful/build-cuda-verbose.sh
```

建议同时把 `--parallel 100` 改为先用 `--parallel 16`。这不是 NFS 问题的根本修复，但能明显减少 Debug/CUDA 编译的内存、文件系统和调度压力。确认稳定后再逐步提高。

若还要消除 NFS 源文件属性缓存带来的干扰，进一步把工作源码也放到 `/data/like/src/onnxruntime`，再定期从 NFS 工作树同步；但优先移动 build tree 已能消除本次最关键的 `.ninja_log`/输出 mtime 问题。

### 83.6 日常改一行 C++ 的构建方式

首次成功配置后，普通 `.cc` 日志改动不需要重新运行 FetchContent、submodule sync 或完整 CMake configure。先对本地 build tree 只构建 Python runtime 所需目标：

```bash
BUILD_DIR=/data/like/build/onnxruntime-v1.27.0-cuda13
cmake --build "$BUILD_DIR/Debug" \
  --target onnxruntime_pybind11_state \
  -- -j16
```

这个 target 会沿依赖关系重编译 `dynamic_quantize_lstm.cc`、重新归档 `libonnxruntime_providers.a`，并重链接 Python binding/核心 runtime；不会因为默认 `all` target 去重新构建无关的测试程序。

需要产出 wheel 时，再调用 `build.py` 的构建和打包阶段，但去掉 `--update`。需要保留同一 CUDA 参数以便 wheel 正确包含 CUDA provider，例如：

```bash
"$ENV_ROOT/bin/python" tools/ci_build/build.py \
  --config Debug \
  --build \
  --build_dir "$BUILD_DIR" \
  --cmake_path "$ENV_ROOT/bin/cmake" \
  --cmake_generator Ninja \
  --parallel 16 \
  --use_cuda \
  --cuda_version 13.0 \
  --cuda_home "$CUDA_HOME" \
  --cudnn_home "$CUDNN_HOME" \
  --build_wheel \
  --skip_tests
```

这一步仍会构建默认 target 后再打 wheel，但在健康的本地 Ninja tree 中，它只应重建被修改源文件及必要的归档/链接链，而不会重复数百个无关 CPU 源。若这一步仍出现大量不相关 `.cc` 编译，先执行：

```bash
ninja -C "$BUILD_DIR/Debug" -d explain -n
```

检查每个 dirty reason；不要先假定是 CMake 的正常行为。对 NFS tree 不建议用这个命令去“修复”，因为它本身可能继续推进不可靠的构建图。

### 83.7 无法立即迁移到本地磁盘时

临时方案是删除该 **专用 Debug build directory** 后做一次完整单线程或低并发重建，例如 `-j8`，并且每次改动后用 `ninja -d explain -n` 核实原因。它能降低复现概率，但不提供可靠保证；NFS build output 仍可能再次产生时间戳错序。

不要：

- 通过 `touch` 全部对象、静态库或 `build.ninja` 来“校正”时间戳；这会掩盖依赖关系并可能得到包含旧对象的 wheel；
- 从当前 NFS Debug 目录复制 `.ninja_log`、`.ninja_deps`、`CMakeCache.txt` 或 `*-build` 到本地；这些正是被污染的状态；
- 仅把 build tree 从 `/share` 改到 `/share_data`；两者在本机是同一个 NFS4 挂载；
- 把 `--parallel 100` 当作增量构建问题的唯一原因。降低并发有帮助，但主因仍是 NFS 输出元数据不可靠。

## 84. `seq_len=5, batch_size=3` 时 `DynamicQuantizeLSTM::Compute` 的执行过程

### 84.1 本节分析的确切 case

`like-useful/test-onnx-dynamic-quant-lstm.py` 的第 189 行：

```python
cpu_outputs = run_cases("CPU", cpu_session, cases)
```

会依次运行 `VERIFY_SHAPES` 中的三个 case。第一个正是 `(seq_len, batch_size) = (5, 3)`。`make_input_cases()` 用固定 seed `20260720` 生成：

```text
X / input : float32 [5, 3, 10]
h0        : float32 [1, 3, 20]
c0        : float32 [1, 3, 20]
```

这里的 `5` 是时间步数，不是一个 batch 里只有 5 个标量 token。更精确地说：

- 有 5 个时间片 `X[0]` 到 `X[4]`；每个时间片都是 `[3, 10]`；
- batch 中有 3 条互不串状态的序列，记为 `b=0,1,2`；
- 因而本次调用实际消费 15 个输入向量 `X[t,b,:]`，每个向量长度为 10；
- 每条 batch 序列各自沿 `t=0 -> 4` 递归，不能把同一条序列的五个时间步并行化。

下面的源码路径以 `/share/users/like/package/onnxruntime` 表示；当前它与提问中的 `/softhome/like/package/onnxruntime` 指向同一个文件（inode 与 SHA-256 都相同），所以代码结论对应同一份工作树。

### 84.2 实际量化 ONNX 图，而不是泛化的 LSTM 假设

用当前 `temp/test-lstm-quant.onnx` 检查得到，该图只有一个真正执行 LSTM 的节点：

```text
com.microsoft::DynamicQuantizeLSTM
  inputs:
    0  input                         [seq_len, batch, 10] float32
    1  onnx::LSTM_89_quantized       [1, 10, 80] int8
    2  onnx::LSTM_90_quantized       [1, 20, 80] int8
    3  onnx::LSTM_91                 [1, 160] float32    # B
    4  ""                                             # sequence_lens 不提供
    5  h0                            [1, batch, 20]
    6  c0                            [1, batch, 20]
    7  ""                                             # P / peephole 不提供
    8  W_scale                       [1, 80] float32
    9  W_zero_point                  [1, 80] int8
    10 R_scale                       [1, 80] float32
    11 R_zero_point                  [1, 80] int8
  outputs: internal_Y, hn, cn
  attribute: hidden_size = 20

ai.onnx::Squeeze(internal_Y, axis=1) -> output
```

所以 `80 = 4 * hidden_size`。该节点本身内部产生的 `Y` 是 ONNX LSTM 规范形状 `[5, 1, 3, 20]`；随后 `Squeeze` 去掉单向 LSTM 的 direction 轴，才成为 Python 看到的 `output [5, 3, 20]`。

图中没有 `direction` 属性，故为默认单向 forward；没有 `input_forget`、`activations`、`clip` 属性，故使用标准的输入门/遗忘门/输出门 `sigmoid`、候选门和输出状态 `tanh`，且 clip 等价于不限制。`sequence_lens` 和 peephole `P` 均为空，因此三条序列的有效长度全是 5，没有跳步、掩码或尾部置零的分支。

量化脚本使用：

```python
quantize_dynamic(..., weight_type=QuantType.QInt8, per_channel=True)
```

实际保存的 `W_scale` 与 `R_scale` 都是 `[1,80]`，80 个 scale 均不相同；对应的 int8 zero point 全为 0。换言之，权重是按输出列（也就是 80 个 I/O/F/C gate channel）做对称 per-channel INT8 量化。虽然 C++ 中包装类型写成 `GemmWeights<uint8_t>`，那只是原始字节的承载类型；`is_W_signed_`/`is_R_signed_` 和 MLAS 的 `BIsSigned` 会把本模型正确作为 signed INT8 权重处理。

### 84.3 从 Python 到 C++ 的一次调用路径

对这个 first case，一次 `session.run(["output", "hn", "cn"], feeds)` 只调用 **一次** `DynamicQuantizeLSTM::Compute`，并不是五个 ONNX 节点或五次外层 `Compute`。调用链为：

```text
run_cases()
  -> InferenceSession.run()
     -> CPUExecutionProvider 上的 com.microsoft::DynamicQuantizeLSTM
        -> DynamicQuantizeLSTM::Compute()
           -> LSTMBase::ComputeImpl<float, uint8_t>()
              -> UniDirectionalLstm<float>::Compute<uint8_t>()
                 -> UniDirectionalLstm<float>::ComputeImpl()
                    -> for (step = 0; step < 5; ++step) { ... }
```

`DynamicQuantizeLSTM::Compute` 位于 `onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc:174-250`。它自身不写时间步循环，职责是：

1. 取得或使用已 prepack 的 `W`/`R`，再读取 scale 和 zero point；
2. 校验 scale/zero-point 的形状；此 case 均为 `[1,80]`；
3. 创建 direction 0 的 `GemmWeights<uint8_t> W_1` 和 `R_1`；
4. 调用 `LSTMBase::ComputeImpl<float, uint8_t>`，把真正的序列计算交给公共 LSTM 实现。

由于 `W` 与 `R` 都是 graph initializer，session 初始化时会尝试调用该 kernel 的 `PrePack()`：`TryPackWeights()` 按 `K=10,N=80` 和 `K=20,N=80` 调用 `MlasGemmPackB`，使后续 GEMM 可以直接使用 packed B。是否成功 prepack 不改变数学结果，只改变运行时读取的是 initializer 还是 packed buffer。当前在 `Compute` 入口添加的 `packed_W_.buffer_` / `packed_R_.buffer_` 日志正可验证此点：非空表示本次运行走 packed-weight 路径。

### 84.4 进入 `LSTMBase` 后的状态、偏置与输出缓冲区

`LSTMBase::ComputeImpl` 在 `onnxruntime/core/providers/cpu/rnn/lstm_base.cc:23-177` 中解出：

```text
seq_length = 5
batch_size = 3
input_size = 10
hidden_size = 20
num_directions = 1
```

并分配：

```text
Y   : [5, 1, 3, 20]
Y_h : [1, 3, 20]
Y_c : [1, 3, 20]
```

随后构造一个 `UniDirectionalLstm<float>`。其构造函数会把输入 `h0[0,:,:]` 复制到内部的 `batched_hidden0_ [3,20]`，把 `c0[0,:,:]` 复制到 `batched_internal_memory_prev_ [3,20]`（`uni_directional_lstm.cc:131-144`）。因此本例并非默认全零初始状态，而是每条 batch 序列各自使用测试脚本生成的随机 `h0`/`c0`。

`B [1,160]` 含有 8H 个 float 偏置：前 4H 是 W 的 bias，后 4H 是 R 的 bias。构造 `UniDirectionalLstm` 时，`LoadBias()` 已把同一 gate 的两半相加，得到四个 `[20]` 的 float bias：`bias_WRi_`、`bias_WRo_`、`bias_WRf_`、`bias_WRc_`。偏置和 cell/hidden state 保持 float32，并没有作为持久张量再量化。

### 84.5 动态量化实际发生在哪里

这是最容易被“每个 token 动态量化一次”的直觉误导的部分。这里有两类 QGEMM，动态量化粒度不同。

#### 输入权重乘法：先对整个 5x3 序列做一次 QGEMM

`UniDirectionalLstm::ComputeImpl` 先设定：

```text
hidden_size_x4 = 80
total_rows = max_sequence_length * batch_size = 5 * 3 = 15
```

然后在时间循环之前执行一次：

```text
X_flat [15,10]  x  W_q [10,80]  ->  output_iofc [15,80]
```

也就是说，`X[0]` 到 `X[4]` 的 15 个输入向量在同一次 `ComputeGemm(M=15,N=80,K=10)` 中完成 `X * W`。`rnn_helpers.cc:271-316` 先对这 150 个 float 值整体计算一组 `a_scale`、`a_zero_point`，量化为临时 uint8 A，再由 MLAS 的 QGEMM 直接输出 float `output_iofc`。它不是每个时间步各算一次 `X_t * W`，也不是每个 batch 行单独算一次该输入 GEMM。

对第 `j` 个输出列，其数值含义可近似写为：

```text
sum_k (qA[m,k] - zpA) * (qW[k,j] - zpW[j]) * (scaleA * scaleW[j])
```

本模型 `zpW[j] = 0`，`scaleW[j]` 随列 `j=0..79` 变化。MLAS 的 output processor 在量化整数累加后应用该 scale，故 `output_iofc` 仍是 float32，可以继续与 R 路径相加。

#### 循环权重乘法：每个时间步都要做，并动态量化当前 `H`

在时间循环内，每次使用上一时刻的 hidden state：

```text
H_prev [M,20]  x  R_q [20,80]  ->  [M,80]
```

该调用的 `beta=1`，所以它把 `H_prev * R` 加到已经存在的 `X * W` 对应行上。因为 `H_prev` 是上一时刻刚算出的 float 值，它必须在每次调用 QGEMM 前重新计算动态 `a_scale/a_zero_point` 并量化为临时 uint8。因此：

- 输入路径：本 case 的 `X * W` 是 1 次 `M=15,N=80,K=10` QGEMM；
- 循环路径：逻辑上有 5 个时间步的 `H * R`，总覆盖 `5 * 3` 行、每行 `K=20,N=80`；
- 若 batch 被线程池切成多个工作组，某个时间步的循环 QGEMM 会按工作组拆为多次调用，因此动态 `H` 的 scale 是“每个实际 GEMM 工作组”而非必然“每个单独 batch 行”一组；数学上仍是相同的三条独立 LSTM 轨迹。

这里的“动态”指运行时激活量化，不表示权重每一步重做量化。`W_q/R_q`、80 个 per-channel scale 和 zero point 是模型内固定 initializer；临时 `qX/qH`、`scaleA/zpA` 随本次输入或当前 hidden state 变化。

### 84.6 5 个时间步如何循环

把输入的 `h0`/`c0` 记成 `H[-1]`/`C[-1]`，这样第 `t` 次循环的状态关系最清楚：

| `step` | 本时间片 | 读入状态 | 写出状态 | 输出位置 |
| --- | --- | --- | --- | --- |
| 0 | `X[0,:,:] [3,10]` | `H[-1]=h0[0]`、`C[-1]=c0[0]` | `H[0]`、`C[0]` | `Y[0,0,:,:]=H[0]` |
| 1 | `X[1,:,:] [3,10]` | `H[0]`、`C[0]` | `H[1]`、`C[1]` | `Y[1,0,:,:]=H[1]` |
| 2 | `X[2,:,:] [3,10]` | `H[1]`、`C[1]` | `H[2]`、`C[2]` | `Y[2,0,:,:]=H[2]` |
| 3 | `X[3,:,:] [3,10]` | `H[2]`、`C[2]` | `H[3]`、`C[3]` | `Y[3,0,:,:]=H[3]` |
| 4 | `X[4,:,:] [3,10]` | `H[3]`、`C[3]` | `H[4]`、`C[4]` | `Y[4,0,:,:]=H[4]` |

实际循环就是 `uni_directional_lstm.cc:332-404` 的：

```cpp
for (int step = 0; step < max_sequence_length; step++) {
  // 1. 从 output_iofc 的 step 对应 [batch,80] 行取得已经算好的 X*W
  // 2. 计算并累加 H_prev*R
  // 3. 对每个 batch 行调用 GateComputations，原地更新 C，并写出 H
  // 4. 把本步 H 作为下一 step 的 previous_state
}
```

因为本例没有 `sequence_lens`，内部补出的长度数组是 `[5,5,5]`，所以 `max_sequence_length=5`、`min_sequence_length=5`；循环恰好是 `step=0,1,2,3,4`，不会进入“已超过某行 sequence length，输出置零”的分支。

在一个 step 中，`output_iofc[(step * 3 + b), :]` 的 80 个 float 先是对应 `X[step,b,:] * W_q` 的结果；加上 `H[step-1,b,:] * R_q` 后，就是四个 gate 的未激活输入。每个 batch 行 `b` 都使用自己的上一状态，没有 `b=0` 到 `b=1` 的状态传递。

### 84.7 每个 step 的四个 gate 计算

ONNX LSTM 的融合 gate 列顺序是 **I/O/F/C**。代码也明确把一个 `[80]` 行切为：

```text
pi = [0 : 20]     # input gate I
po = [20 : 40]    # output gate O
pf = [40 : 60]    # forget gate F
pc = [60 : 80]    # cell candidate C（下式记为 G）
```

这与 `test-lstm.py` 中 PyTorch 参数的 I/F/G/O 存储顺序不同；ONNX 导出器已经完成了对应重排。因此不能拿 PyTorch 原始 tensor 的连续分块顺序直接解释量化 ONNX 中的 80 个列。对任意 `t,b`，令四段矩阵/偏置均已取出正确的 ONNX I/O/F/C 列，则数学过程为：

```text
z_i, z_o, z_f, z_g
  = X[t,b,:] * W_{i,o,f,g} + H[t-1,b,:] * R_{i,o,f,g}
    + (Wb_{i,o,f,g} + Rb_{i,o,f,g})

i = sigmoid(z_i)
f = sigmoid(z_f)
g = tanh(z_g)
C[t,b,:] = f * C[t-1,b,:] + i * g
o = sigmoid(z_o)
H[t,b,:] = o * tanh(C[t,b,:])
```

`GateComputations()` 位于 `uni_directional_lstm.cc:463-601`：

- `clip_with_bias_ptr_` 对 I/F/G/O 的线性结果加融合 bias 后再做 clip；本图没有有效 clip 限制；
- `merge_lstm_gates_to_memory()` 按 `C = C_prev * f + i * g` 直接原地覆盖内部的 `C_prev` buffer；
- `activation_h_` 使用刚更新的 C 和 output gate，写出当前 H；
- 没有 `P`，故所有 peephole 分支不执行；`input_forget=false`，故 forget gate 不是 `1-i` 的耦合模式。

因此，误差传播的时间顺序也很明确：本步的 INT8 GEMM 近似会影响 `i/f/g/o`，继而影响 `C[t]` 与 `H[t]`；`H[t]` 又是下一步循环量化和 `H*R` 的输入，故量化误差可以沿 5 个 step 递归累积。cell/hidden 不是 INT8 常驻状态，仍以 float 保存；每一步只是为了矩阵乘法临时量化当前 activation。

### 84.8 batch 并行不会打破时间依赖

本例 `batch_size=3`、`hidden_size=20`，满足 `SetNumThreads()` 的 batch-parallel 条件：`num_rows >= 2 && num_columns <= 256`。实际工作组大小还取决于 session thread pool 的线程数：可能是一个组处理三个 batch 行，也可能拆成若干组并发处理。

不过每个工作组的 lambda 都是“先完整执行 `step=0..4`，再结束”。所以允许的并行是：

```text
batch 0 的五步  ||  batch 1 的五步  ||  batch 2 的五步
```

而不是：

```text
同一 batch 的 step 0 || step 1 || ... || step 4
```

后者不成立，因为 `step+1` 需要前一 step 计算出的 `H` 和 `C`。不同的线程调度只会改变各 batch 行的执行先后，不应改变输出数值语义。

### 84.9 最终三个 Python 输出如何得到

循环结束后，`UniDirectionalLstm` 从每条序列的最后有效时间步拷贝最终 hidden state：

```text
output : Squeeze(Y, direction axis) = [5, 3, 20]
hn     : [1, 3, 20]，且 hn[0,b,:] = H[4,b,:]
cn     : [1, 3, 20]，且 cn[0,b,:] = C[4,b,:]
```

其中 `output[4,b,:]` 与 `hn[0,b,:]` 是同一最终 hidden state；`cn` 是最终 cell state，不等同于 `output`。第 189 行的 `run_cases()` 仅检查这些输出的形状和有限性；后续 `compare_outputs()` 才用于 CPU 与 CUDA 路径的数值比较。

若要在 C++ 调试器或日志中观察这个 case，最有价值的断点/日志位置依次是：

```text
dynamic_quantize_lstm.cc:174   # 一次外层 Compute，检查 W/R 是否 prepack
uni_directional_lstm.cc:286    # 一次完整 X[0:5] * W 的 QGEMM
uni_directional_lstm.cc:333    # 五次 step 循环入口
uni_directional_lstm.cc:342    # 每个 step 的 H_prev * R QGEMM
uni_directional_lstm.cc:370    # I/O/F/C 激活、C/H 更新
```

这样观察到的次数应是：外层 `Compute` 1 次；输入路径 QGEMM 1 次；逻辑时间步 5 次；每条 batch 轨迹的状态更新 5 次。若 batch 并行把 batch 拆成多个工作组，`H*R` 的底层 QGEMM 调用次数会比 5 多，但每条轨迹仍只有这 5 次严格有序的状态更新。

## 85. `context.Output()` 如何决定 Tensor 在 CPU 还是 GPU 上分配

### 85.1 先给结论

`lstm_base.cc` 中的：

```cpp
Tensor* Y = context.Output(/*index*/ 0, Y_dims);
```

并不根据 `Y_dims`、Tensor 的数据类型，或者当前机器是否有 GPU，临时判断“这次放 CPU 还是 GPU”。`Y_dims` 只描述形状；真正的内存位置在 session 建立执行计划时已经决定，`context.Output()` 执行时只是按照该计划找到/创建对应的 `OrtValue`。

对当前代码，决定链可以概括为：

```text
图节点被分配给哪个 Execution Provider
  -> 该 EP 选择的 KernelDef
     -> KernelDef 的 output memory type
        -> SequentialPlanner 为该输出记录 OrtDevice
           -> ExecutionFrame 按 allocation plan 获取该 device 的 allocator
              -> Tensor::InitOrtValue() 绑定这块 CPU/GPU 内存
```

因此，判断输出位置的核心问题不是“`context.Output()` 看到了什么”，而是“这个节点最后绑定了哪个 EP，以及该 kernel 声明的 output memory type 是什么”。

### 85.2 `context.Output()` 的直接调用链

`OpKernelContext::Output(int, const TensorShape&)` 的实现位于 `onnxruntime/core/framework/op_kernel.cc:44-47`：

```cpp
Tensor* OpKernelContext::Output(int index, const TensorShape& shape) {
  auto p_ml_value = OutputMLValue(index, shape);
  return p_ml_value ? p_ml_value->GetMutable<Tensor>() : nullptr;
}
```

随后 `OutputMLValue()` 在同一个文件的 `72-84` 行调用：

```cpp
execution_frame_->GetOrCreateNodeOutputMLValue(
    index, GetOutputArgIndex(index), &shape, p_ml_value, kernel_->Node());
```

这里发生了两个索引转换：

1. `index=0` 是当前算子的第 0 个输出，即 LSTM 的 `Y`；
2. `GetOutputArgIndex(0)` 把它转换成 `ExecutionFrame` 中的 graph value index；
3. `NodeIndexInfo` 再把 graph value index 映射到 `all_values_` 中的 `OrtValue`。

`GetOrCreateNodeOutputMLValue()` 位于 `execution_frame.cc:144-219`。它先处理特殊情况：

- 如果该输出是未使用的 optional output，返回 `nullptr`；
- 如果调用者在 `Run()` 前已经提供了预分配的输出 `OrtValue`，并且 shape 匹配，则复用调用者的 buffer；
- 否则调用 `CreateNodeOutputMLValueImpl()`，按照 session 的 allocation plan 分配。

所以，严格来说，预分配输出是一个例外：调用者提供的 buffer 及其 `OrtMemoryInfo` 已经决定了位置，ORT 只验证 shape 是否匹配，不会因为 `context.Output()` 再把它从 CPU 搬到 GPU 或反过来。普通 Python `session.run()` 不提供这种预分配输出时，走的是 ORT 的 allocation plan。

### 85.3 allocation plan 中的 device 从哪里来

session finalize 阶段由 `SequentialPlanner::ComputeValueLocation()` 为每个 graph value 计算位置。`allocation_planner.cc:766-778` 明确写着，节点的输出位置由“绑定到该节点的 OpKernel”决定：

```cpp
const KernelCreateInfo& kernel_create_info = ...;
const auto* p_kernel_def = kernel_create_info.kernel_def.get();
auto exec_provider = execution_providers_.Get(*pnode);
```

对每个输出，核心代码是 `allocation_planner.cc:907-937`：

```cpp
OrtDevice output_device = exec_provider->GetOrtDeviceByMemType(
    p_kernel_def->OutputMemoryType(i));
plan_.SetLocation(static_cast<size_t>(index), output_device);
```

也就是：

```text
output_device = 当前节点 EP.GetOrtDeviceByMemType(KernelDef.OutputMemoryType(i))
```

如果输出设备是 CPU-accessible memory，planner 还可能查看下游 consumer 对 CPU input 的建议，在 CPU、host-accessible 等位置之间选择更少拷贝的设备；这部分在 `allocation_planner.cc:916-935`。但对于当前 CPU LSTM，初始 `output_device` 已经是 CPU，通常不会改变结论。

`plan_.SetLocation()` 写入的是该 graph value 的 `AllocPlanPerValue.location`。之后 `ExecutionFrame::AllocateAsPerAllocationPlan()` 在 `execution_frame.cc:752-834` 取出：

```cpp
const auto& per_alloc_plan = alloc_plan[ort_value_index];
const auto& alloc_info = per_alloc_plan.location;
```

再根据 `alloc_kind` 选择自有分配、复用已有 buffer 或共享 `OrtValue`。自有分配最终进入 `AllocateMLValueTensorSelfOwnBufferHelper()`，在 `execution_frame.cc:615-634` 执行：

```cpp
alloc = GetAllocator(location);
Tensor::InitOrtValue(element_type, shape, std::move(alloc), ort_value);
```

因此，`context.Output()` 传入的 `shape` 只影响需要分配多少 bytes；`location` 来自预先计算的 allocation plan，`allocator` 再由该 location 查找。

### 85.4 `OrtMemTypeDefault` 到底代表什么

这里容易产生一个误解：`OrtMemTypeDefault` 并不永远等于 CPU。它的含义是“当前 Execution Provider 的默认设备内存”。

`KernelDef::OutputMemoryType()` 在 `include/onnxruntime/core/framework/kernel_def_builder.h:91-95` 中，如果没有针对某个 output 单独设置 memory type，就返回 `default_outputs_mem_type_`；默认值在同一文件 `159-161` 是 `OrtMemTypeDefault`。

Execution Provider 再把这个 memory type 转换成自己的 `OrtDevice`。公共实现位于 `include/onnxruntime/core/framework/execution_provider.h:439-444`：CPU input/output 类型返回 CPU device，其他情况返回该 EP 的 `default_device_`。

因此通常可以这样理解：

| 节点绑定的 EP | KernelDef output memory type | `GetOrtDeviceByMemType()` 的典型结果 | 输出位置 |
| --- | --- | --- | --- |
| CPU EP | `OrtMemTypeDefault` | `OrtDevice::CPU` | CPU 内存 |
| CUDA EP | `OrtMemTypeDefault` | CUDA device，例如 GPU 0 | GPU 显存 |
| CUDA EP | `OrtMemTypeCPUOutput` | host-accessible CPU device | CPU 可访问内存 |
| CPU EP | `OrtMemTypeCPUOutput` | CPU device | CPU 内存 |

“Default”是相对于 EP，而不是相对于整台 ORT session。一个 session 同时有 CPU EP 和 CUDA EP 时，必须先知道节点属于哪个 EP，单独看到 `OrtMemTypeDefault` 仍不足以判断 CPU/GPU。

### 85.5 当前 `DynamicQuantizeLSTM` 的实际结果

当前 `dynamic_quantize_lstm.cc:253-263` 的 kernel 注册是：

```cpp
ONNX_OPERATOR_TYPED_KERNEL_EX(
    DynamicQuantizeLSTM,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        ...,
    DynamicQuantizeLSTM);
```

这段注册有两个直接结论：

1. 当前这份 `DynamicQuantizeLSTM` kernel 属于 CPU EP；
2. `KernelDefBuilder` 没有调用 `OutputMemoryType(...)` 或 `SetDefaultOutputMemoryType(...)`，所以三个输出使用 `OrtMemTypeDefault`。

CPU EP 在 `cpu_execution_provider.cc:61-66` 创建的 preferred allocator 是 `CPUAllocator`。它的默认 `OrtDevice` 是 CPU，且 CPU EP 没有把 `OrtMemTypeDefault` 改成 GPU device。因此当前 `LSTMBase::ComputeImpl` 中：

```cpp
Tensor* Y   = context.Output(0, Y_dims);
Tensor* Y_h = context.Output(1, Y_h_dims);
Tensor* Y_c = context.Output(2, Y_c_dims);
```

在这个 CPU kernel 中得到的三个 Tensor 都是 CPU Tensor。`Y` 的 shape 是 `[seq, num_directions, batch, hidden]`，但它的 shape 与 CPU/GPU 无关；决定位置的是该 value 对应的 `AllocPlanPerValue.location`。

### 85.6 为什么 CUDA session 中它仍可能在 CPU

如果 Python 创建：

```python
ort.InferenceSession(
    model,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

并不意味着所有节点都在 CUDA 上。ORT 会先尝试把节点分区/绑定给 CUDA EP；如果 CUDA EP 没有 `com.microsoft::DynamicQuantizeLSTM` 的 kernel，就会把该节点回退给 CPU EP（前提是允许 CPU fallback）。此时：

```text
DynamicQuantizeLSTM -> CPU EP -> Y/Y_h/Y_c 在 CPU
```

CUDA session 中的其他 CUDA 节点如果把输出放在 GPU，那么跨 EP 的边上会插入或执行数据拷贝，使 CPU LSTM 能读到 CPU 输入；LSTM 输出再根据下游节点需要复制到 GPU。这个复制发生在节点之间，不是 `context.Output()` 把本节点的输出“自动选成 GPU”。

因此，“session 使用 CUDA EP”和“某一个 kernel 的输出在 GPU”是两个不同问题。必须查看实际节点 provider/profile，或者在 kernel 中直接打印 `Y->Location()`。

### 85.7 如何在源码中验证实际位置

`Tensor` 在 `include/onnxruntime/core/framework/tensor.h:199-202` 提供：

```cpp
const OrtMemoryInfo& Tensor::Location() const;
```

在 `Y` 分配完成后可以临时加入类似日志：

```cpp
Tensor* Y = context.Output(/*index*/ 0, Y_dims);
LOGS_DEFAULT(WARNING)
    << "Y location=" << Y->Location().ToString()
    << ", shape=" << Y->Shape()
    << ", data=" << Y->DataRaw();
```

预期结果大致是：

```text
CPU EP:   Y location=Cpu;0;... 或对应 CPU OrtMemoryInfo
CUDA EP:  Y location=Cuda;0;... 或对应 CUDA OrtMemoryInfo
```

具体字符串格式取决于 `OrtMemoryInfo::ToString()`，不要只根据 `Y->DataRaw()` 的指针地址判断位置；CPU 虚拟地址和 CUDA 映射地址都只是地址，可靠信息是 `Y->Location()`。

如果要打印 kernel provider，需要在持有 `OpKernel`/`OpKernelContext` 的外层代码中访问 kernel info；`LSTMBase::ComputeImpl` 本身不能直接访问 `OpKernelContext` 的私有 `kernel_` 成员。但在 `LSTMBase::ComputeImpl` 中通常没有必要通过 kernel provider 再推断，因为 `Tensor::Location()` 已经是最终实际绑定的 memory info。

### 85.8 与临时 workspace allocator 的区别

不要把 `context.Output()` 和 `context.GetTempSpaceAllocator()` 混为一谈。

`GetTempSpaceAllocator()` 在 `op_kernel.cc:95-100` 中通过：

```cpp
GetAllocator(kernel_->GetDevice(OrtMemTypeDefault))
```

直接取得当前 kernel device 的临时 allocator。LSTM 使用它来申请内部的 hidden/cell 临时 buffer、packed weight 等 workspace；这些也会跟随当前 kernel 的 EP。

而 `context.Output()` 走的是 `ExecutionFrame -> allocation plan -> per-value location`，因为输出 tensor 还要考虑 graph value 的生命周期、内存复用、memory pattern、下游 consumer、用户预分配输出等因素。

两条路径在当前 CPU `DynamicQuantizeLSTM` 上都会得到 CPU memory，但原因和控制层次不同：

```text
GetTempSpaceAllocator()
  -> 当前 kernel 的 device

context.Output()
  -> 当前 graph value 的 allocation plan location
  -> 对应 device 的 allocator
```

最终可以用一句话概括：`context.Output(0, Y_dims)` 不负责选择 CPU/GPU；图分区和 kernel 的 memory type 先决定 `Y` 的 `OrtDevice`，`ExecutionFrame` 再用该 device 的 allocator 创建 Tensor。对当前只注册 CPU kernel 的 `DynamicQuantizeLSTM`，`Y`、`Y_h`、`Y_c` 的实际分配位置是 CPU；只有真正执行 CUDA kernel 且其 output memory type 映射到 CUDA default device 时，类似代码才会得到 GPU Tensor。

## 86. 三种 `Compute` / 注册方式：应选择哪一种自定义算子实现

结论先说：这不是三种等价的“自定义算子”接口，而是两个**外部 ORT Custom Op API 层次**，加上一个**ORT 源码内部 OpKernel 层次**。

对 `simo` 这种由 `RegisterCustomOps` 导出的独立共享库，新的 `com.simo` 算子默认应使用第 1 种 Lite Custom Op：`Ort::Custom::CreateLiteCustomOp`。第 2 种 `Ort::CustomOpBase` 仍是受支持的低层 C++ 包装，适合需要手工控制 `OrtCustomOp` 回调的场景或保持已有实现；并非已经失效。第 3 种 `ONNX_OPERATOR_TYPED_KERNEL_EX` 不是外部插件 API，只有把实现合入并编译 ONNX Runtime 本体（或其内部 Execution Provider）时才应使用。

本节中的 `simo/...` 路径相对于 `/share/users/like/package/simo_conda_sglang`，`onnxruntime/...` 和 `include/...` 路径相对于 `/softhome/like/package/onnxruntime`；行号对应当前 checkout。

### 86.1 三个 `Compute` 签名的真正区别

| 实现 | 源码中的 `Compute` | API 层次 | 谁将输入/输出交给实现 | 错误返回 |
| --- | --- | --- | --- | --- |
| CUDA QDQ | `DequantizeCustomOp<T>::Compute(OrtKernelContext&, const Ort::Custom::Tensor<uint8_t>&, const Ort::Custom::Tensor<uint8_t>&, Ort::Custom::Tensor<T>&)` | 外部插件的 Lite C++ API | Lite 框架按函数签名构造类型安全的 `Tensor<T>` wrapper | `Ort::Status` |
| CPU QDQ | `DequantizeCpuDirectKernel<T>::Compute(OrtKernelContext*)` | 外部插件的经典 `OrtCustomOp` C ABI 的 C++ 包装 | 实现自行调用 `OrtApi::KernelContext_GetInput/GetOutput` 等函数取值、分配输出 | `void`；当前代码以 C++ exception 报错 |
| `DynamicQuantizeLSTM` | `DynamicQuantizeLSTM::Compute(OpKernelContext*) const` | ORT 内部 C++ `OpKernel` | ORT 内核执行器传入私有的 `OpKernelContext` | 内部 `onnxruntime::Status` |

第一项的实现在 `simo/onnx/ort_plugin/simo_qdq_ops.cc:507-569`，函数为 `DequantizeCustomOp<T>::Compute`，实际签名见 `:518-522`。`quantized`、`scale` 被声明为 const typed tensor，`output` 是可分配的 typed tensor；因此 `output.Allocate({outer_dim, quant_dim})` 在 `:548` 分配输出。它在 `:523-564` 中直接用 wrapper 读 shape、data，失败则返回 `Ort::Status`。这正是 Lite API 的核心价值：算子的输入数、输出数和 ONNX element type 不需要在另一个 `GetInputType*`/`GetOutputType*` 实现中再写一遍。

这个多参数签名只是作者面对的 C++ 层接口，不是 ORT 二进制 ABI 中直接调用的函数类型。`include/onnxruntime/core/session/onnxruntime_lite_custom_op.h:1028-1047` 的 `Ort::Custom::OrtLiteCustomStruct<CustomOp>::SetCompute` 会把成员函数签名解析为 input/output 类型，然后生成统一的 `OrtCustomOp::KernelCompute` 或 `KernelComputeV2` 回调；其中 `CreateTuple(...)` 将一个 `OrtKernelContext*` 及其输入/输出转换为 `Tensor<T>` 参数，最后再调用 `custom_op_->Compute(...)`。返回 `Ort::Status` 的版本走 `KernelComputeV2`，见该函数 `:1039-1047`。也就是说，底层仍然有一个 context 指针，只是 Lite adapter 替 `simo` 做了取输入、分配 output 和 status 传递。

第二项的实现位于 `simo/onnx/ort_plugin/simo_qdq_cpu_ops.cc:1072-1135`，函数 `DequantizeCpuDirectKernel<T>::Compute` 的签名在 `:1081`。它在 `:1083` 调用本文件函数 `KernelInputShape`，在 `:1108-1109` 调用 `KernelInputData<uint8_t>`，在 `:1111` 调用 `KernelOutputData<T>`。这些 helper 的定义在同一文件 `:53-85`，内部逐个调用 C API 的 `KernelContext_GetInput`、`KernelContext_GetOutput` 和 `GetTensorMutableData`。所以参数个数少，不表示它没有两个 input 和一个 output；恰好相反，input/output 的编号、类型和输出 shape 全部由函数体手工管理。

其 `void` 返回值也不等价于“不可能报告错误”。当前 `DequantizeCpuDirectKernel<T>::Compute` 在 shape 或 runtime 检查失败时 `throw std::runtime_error`，例如 `simo_qdq_cpu_ops.cc:1085-1107`、`:1126-1128`。但错误不能从 `Compute` 的返回值显式传回。`Ort::CustomOpBase<..., false>` 的 `false` 正是 void callback 模式：`include/onnxruntime/core/session/onnxruntime_cxx_api.h:3214-3263` 中的 `CustomOpBase` 在 `:3255-3262` 将其桥接为 `OrtCustomOp::KernelCompute(void*, OrtKernelContext*)`，并调用 `TKernel::Compute(context)`。若选用 `CustomOpBase<..., true>`，该模板要求的是另一套 `CreateKernelV2` / `ComputeV2`，其回调返回 `OrtStatusPtr`，见同一文件 `:3244-3254`；不是把现有 `void Compute` 简单改为 `Ort::Status Compute` 即可。

第三项位于 `onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc:19-47`。类 `DynamicQuantizeLSTM` 直接继承 ORT 内部的 `OpKernel`，函数声明在 `:32`，定义为 `DynamicQuantizeLSTM::Compute(OpKernelContext*) const` 在 `:174-251`。`OpKernelContext`、`Tensor`、`Status`、`LSTMBase` 都是 ORT 内部 C++ 实现类型，不是面向独立 custom-op `.so` 的稳定扩展 ABI。因此它的参数、`const` 限定和返回类型都不应与前两种作一一对应比较。

可以把三条实际调用路径概括为：

```text
Lite Custom Op
ORT 的 OrtCustomOp::KernelComputeV2(void*, OrtKernelContext*)
  -> Lite adapter 将 context 展开为 Typed Tensor 参数
  -> DequantizeCustomOp<T>::Compute(context, quantized, scale, output)

CustomOpBase（当前 CPU 实现）
ORT 的 OrtCustomOp::KernelCompute(void*, OrtKernelContext*)
  -> DequantizeCpuDirectKernel<T>::Compute(context)
  -> 实现自行按 index 0/1/0 取 input、input、output

ORT 内部 kernel
KernelRegistry 选中 OpKernel factory
  -> DynamicQuantizeLSTM::Compute(OpKernelContext*) const
```

因此，`Compute` 的参数类型和个数反映的是**封装层是否替你解析 tensor 参数**，并不是性能等级或算子能力的排序。前两者进入 ORT 时都以公开 `OrtCustomOp` 回调为边界；第三者则从始至终留在 ORT 内部。

### 86.2 三种“注册”实际注册了什么

第 1 种中，`Ort::Custom::CreateLiteCustomOp` 不是把 kernel 直接注册到 ORT 全局 registry；它是构造一个 `OrtLiteCustomOp`，而该类型继承公开 C struct `OrtCustomOp`。实现可见 `include/onnxruntime/core/session/onnxruntime_lite_custom_op.h:402-411` 与 `:1028-1096`：前者说明其作为 `OrtCustomOp` bridge，后者的 `CreateLiteCustomOp<CustomOp>` 返回该对象。`OrtLiteCustomOp` 会设置 `GetName`、EP type、input/output type、callback 等 function pointer；构造过程可见 `:764-865`。

当前 CUDA QDQ 正是这一模式。`simo/onnx/ort_plugin/simo_qdq_ops.cc` 的函数 `RegisterQdqOps` 在 `:573-605` 中，分别在 `:576-594` 调用 `CreateLiteCustomOp<...>`，然后在 `:597-604` 以 `domain.Add(op.get())` 加入 `com.simo` domain。`static const std::array<std::unique_ptr<...>>` 很重要：`CustomOpDomain::Add` 只登记指针，不接管 op 对象的生命周期；这可由公开 C++ API 声明 `include/onnxruntime/core/session/onnxruntime_cxx_api.h:1457-1467` 验证。

第 2 种中，`DequantizeCpuDirectOp<T>` 的实际对象类型不是“一个指向 `CustomOpBase` 的指针”，而是它自身的对象，**继承自** `Ort::CustomOpBase<DequantizeCpuDirectOp<T>, DequantizeCpuDirectKernel<T>, false>`；定义在 `simo/onnx/ort_plugin/simo_qdq_cpu_ops.cc:1172-1205`。`CustomOpBase` 又继承公开的 `OrtCustomOp`，并在构造函数中填写 C function table，见 `include/onnxruntime/core/session/onnxruntime_cxx_api.h:3214-3265`。所以 `domain.Add(...)` 最终接收的仍是 `const OrtCustomOp*`，只是这次 function table 由 `CustomOpBase` 填充，而不是 Lite API 填充。

`RegisterCpuQdqOps` 在 `simo/onnx/ort_plugin/simo_qdq_cpu_ops.cc:1209-1222` 中创建 static 的 op 对象并加入同一个 domain。现有的 `const_cast<DequantizeCpuDirectOp<T>*>(&dequantize_...)` 不是必要的注册机制，也不改变语义：`CustomOpDomain::Add` 的参数本来就是 `const OrtCustomOp*`。它只是在把 `static const` 对象转换成非 const 指针后又隐式转回 const；对象不会也不应在注册后被修改。

第 1、2 种最后共享完全相同的插件装载入口。`simo/onnx/ort_plugin/custom_op_library.cc` 的函数 `QdqDomain` 在 `:13-25` 根据 `SIMO_ONNX_QDQ_PROVIDER` 选择 `RegisterCpuQdqOps` 或 `RegisterQdqOps`；导出的 `RegisterCustomOps` 在 `:29-44` 获取 API v17（`simo/onnx/ort_plugin/simo_qdq_ops.h:11`）、调用 `session_options.Add(QdqDomain())`。公开 C API 对这条流程的定义是“创建 `OrtCustomOpDomain` -> 加每个 `OrtCustomOp` -> 将 domain 加入 session options”，见 `include/onnxruntime/core/session/onnxruntime_c_api.h:7492-7496`。应用侧从共享库加载时推荐 `RegisterCustomOpsLibrary_V2`；旧的 `RegisterCustomOpsLibrary` 已标记 deprecated，见同一头文件 `:1658-1674` 和 `:4472-4488`。

第 3 种的 `ONNX_OPERATOR_TYPED_KERNEL_EX` 则根本不操作 `OrtCustomOpDomain`。当前调用在 `onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc:253-263`：它描述 `DynamicQuantizeLSTM`、`kMSDomain`、opset 1、`kCpuExecutionProvider` 以及输入类型约束。宏定义在 `include/onnxruntime/core/framework/op_kernel.h:273-285`，其效果是生成 `BuildKernelCreateInfo<...>()` 的显式特化，创建一个含 `KernelDef` 和 `std::make_unique<DynamicQuantizeLSTM>(info)` factory 的 `KernelCreateInfo`。之后 ORT 的内部 `KernelRegistry` 才以 op/domain/provider/type constraints 查找它；registry 的职责和查找 API 见 `include/onnxruntime/core/framework/kernel_registry.h:20-50`。

另外，宏也**不注册算子 schema**。`com.microsoft::DynamicQuantizeLSTM` 的 schema、输入输出和类型约束是独立定义的，位于 `onnxruntime/core/graph/contrib_ops/quantization_defs.cc:657-758`，宏中的 kernel 注册仅为这个 schema 提供 CPU 实现。若要在 ORT 源码中新增同类内部 op，通常至少需要 schema、一个继承 `OpKernel` 的实现、kernel 注册和相应构建目标改动；这与制作可由 `SessionOptions::RegisterCustomOpsLibrary` 加载的外部 `.so` 是两条不同的集成路径。

### 86.3 对 `simo` 的推荐选择

| 目标 | 推荐实现 | 推荐注册 |
| --- | --- | --- |
| 新增或重写独立 `com.simo` CPU/CUDA 算子 | Lite Custom Op，typed `Tensor<T>` 参数，必要时首参保留 `OrtKernelContext&` | `CreateLiteCustomOp` -> `Ort::CustomOpDomain::Add` -> 导出的 `RegisterCustomOps`；应用侧 `RegisterCustomOpsLibrary_V2` |
| 已有低层实现，或需要手写 function table、input memory type / input-output characteristic 等 Lite 默认策略之外的精细控制 | `CustomOpBase`（或直接 C `OrtCustomOp`） | 仍是 `CustomOpDomain::Add` 和 `RegisterCustomOps` |
| 修改 ORT 内置或 contrib CPU/CUDA Execution Provider | `OpKernel`，`Status Compute(OpKernelContext*) const` | schema + `ONNX_OPERATOR_*_KERNEL*` 宏 + ORT 内部构建/registry |

所以，对当前代码最实用的建议是：继续把 `simo_qdq_ops.cc` 的 Lite 方式作为新实现的默认范式；CPU 的 `DequantizeCpuDirectKernel` 不是“错误写法”，但它承担了 Lite 已经封装的参数解析、类型声明和错误通道工作。只要目标运行时支持当前随插件发布的 Lite Custom Op header/API，CPU QDQ 可以逐步统一到 Lite 风格，行为上仍可通过首个 `OrtKernelContext&` 访问低层 API。不要为了让 `DynamicQuantizeLSTM` 的签名看起来一致而将 `simo` QDQ 改成 `OpKernel` 宏注册：那会把独立插件变成必须重编 ONNX Runtime 的内部 fork，破坏外部插件的版本隔离和部署方式。

## 87. 为什么 `ldd` 显示 Torch 库 `not found`，CUDA pytest 却能通过

结论先说：pytest 运行时并没有在“缺少 `libtorch_cuda.so`”的情况下执行。`libtorch_cuda.so` 等文件真实存在于当前 Python 环境的 `torch/lib` 中，并且在 ORT 加载 custom-op 之前已经由 `import torch` 装入同一个进程。`ldd` 则在一个没有启动 Python、没有导入 Torch的新加载器环境中检查依赖，因此两者看到的动态链接器状态不同。

当前五个依赖实际位于：

```text
/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/torch/lib/libtorch.so
/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so
/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so
/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/torch/lib/libc10.so
/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/torch/lib/libc10_cuda.so
```

### 87.1 当前 `RUNPATH` 为什么没有帮助 `ldd`

插件的 ELF 动态段包含：

```text
NEEDED  libtorch.so
NEEDED  libtorch_cpu.so
NEEDED  libtorch_cuda.so
NEEDED  libc10.so
NEEDED  libc10_cuda.so
RUNPATH $ORIGIN/../../../torch/lib
```

`$ORIGIN` 是被加载的 `.so` 所在目录。当前使用的是 editable install，插件仍位于源码树：

```text
/share/users/like/package/simo_conda_sglang/simo/onnx/ort_plugin/
```

所以该 RUNPATH 在当前文件位置实际展开为：

```text
/share/users/like/package/simo_conda_sglang/torch/lib
```

这个目录不存在，`ldd` 又没有从当前 shell 的 `LD_LIBRARY_PATH` 得到环境内的 `site-packages/torch/lib`，因此报告 `not found`。

这个相对 RUNPATH 是为普通 wheel 安装布局设计的。wheel 安装后插件通常位于：

```text
.../site-packages/simo/onnx/ort_plugin/libSimoOnnxCustomOps_sm90.so
```

此时 `$ORIGIN/../../../torch/lib` 正好归一化为 `.../site-packages/torch/lib`。editable install 将 Python 包映射到源码目录，打破了这个相对位置，因此代码还需要运行时预加载作为补充。

### 87.2 pytest 中的实际加载顺序

`simo/onnx/tests/test_dynamic_qdq_runtime_debug.py` 在模块初始化阶段就会 `import torch`。此外，`simo/onnx/runtime.py` 的 `register_custom_ops()` 在调用 ORT 注册函数前也显式执行一次 `import torch`：

```python
def register_custom_ops(sess_options, library_path=None):
  import torch

  path = get_custom_ops_library_path() if library_path is None else Path(library_path)
  sess_options.register_custom_ops_library(str(path))
  return sess_options
```

因此实际顺序是：

```text
pytest Python 进程
  -> import torch
     -> 动态链接器从 site-packages/torch/lib 装入
        libtorch.so、libtorch_cpu.so、libtorch_cuda.so、libc10.so、libc10_cuda.so
  -> register_custom_ops(...)
  -> ONNX Runtime dlopen(libSimoOnnxCustomOps_sm90.so)
  -> 动态链接器处理插件的 DT_NEEDED
  -> 发现相同 SONAME 的 Torch/C10 对象已经在当前加载器 namespace 中
  -> 复用已加载对象并解析符号
  -> custom-op 注册及 CUDA 测试成功
```

换言之，加载插件时不再需要通过其 RUNPATH 打开一份新的 `libtorch_cuda.so`；动态链接器直接复用当前进程已经加载的对象。`ldd` 只展示“从一个新进程按当前搜索路径开始加载会发生什么”，不会模拟 Python 先执行 `import torch` 后的进程状态。

在当前环境中的实测结果是：

```text
不 import torch，直接 ctypes.CDLL(plugin):
  FAIL: libtorch.so: cannot open shared object file

先 import torch，再 ctypes.CDLL(plugin):
  OK
```

而 `import torch` 后读取 `/proc/self/maps`，可以看到上述五个库都已从 `site-packages/torch/lib` 映射进进程。这直接解释了 pytest 为什么可以通过。

### 87.3 如何让 `ldd` 也显示正确路径

可以只为诊断命令补上 Torch library 目录：

```bash
PY=/share_data/users/like/miniconda3/envs/simo_sglang/bin/python
TORCH_LIB="$($PY -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')"
LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ldd simo/onnx/ort_plugin/libSimoOnnxCustomOps_sm90.so
```

此时五个依赖都会解析到 `simo_sglang/lib/python3.12/site-packages/torch/lib`。

需要注意的是，若调用方绕过 `simo.onnx.runtime.register_custom_ops()`，直接执行 `SessionOptions.register_custom_ops_library(...)`，并且此前没有导入 Torch，那么当前 editable 布局确实可能加载失败。可靠的使用方式是走 `register_custom_ops()`；或者由部署环境把 Torch library 目录加入 `LD_LIBRARY_PATH`。普通 wheel 安装则应由现有相对 RUNPATH 直接覆盖这一问题。

## 88. Silero VAD 两批量化测试的完整性与精度对比

### 88.1 直接结论

两次测试都完整生成了预期的 10 组量化模型和精度数据。根据主日志、每个 shard 的日志及汇总 JSON，没有发现 core dump、进程崩溃或 Python/ONNX Runtime 异常：

| 测试 | 配置完成数 | 量化模型 | `summary.json` | shard 结果 | 错误数 |
|---|---:|---:|---:|---:|---:|
| +LSTM 配置 | 10/10 | 10/10 | 10/10 | 240/240 `done` | 0 |
| no-LSTM 配置 | 10/10 | 10/10 | 10/10 | 240/240 `done` | 0 |

每批输出都同时具有 10 个 `onnx_quant_config.json`、10 个 `run_metadata.json`、10 个 `shard_summary.json`、240 个 shard NPZ 和 240 个 shard 日志。

每个配置都处理了相同的 20 个 AISHELL-4 文件、12.7253 小时音频和 1,431,607 个 chunk。每个配置启动 24 个 shard，因为数据集只有 20 个文件，所以 shard 20 到 23 正常地处理 0 个文件；它们仍写出 NPZ，状态为 `done`、`errors=0`、`returncode=0`。这不是缺失数据。

更严格的只读校验结果如下：

- 20 个 `quantized_model.onnx` 全部通过 `onnx.load` 和 `onnx.checker.check_model`。
- 480 个 shard NPZ 全部通过 ZIP 完整性检查，没有损坏或零字节文件。
- 20 个 summary 都包含数值型 `roc_auc` 和 `accuracy`、非空 `finished_at_utc`、24 个 `status=done` shard；`overall.errors`、所有 shard 的 `errors` 之和均为 0。
- 两份主日志均运行到第 518 行并写出最后一个 `mxint8` summary：[LSTM 主日志](/share/users/like/package/jdjv/silero_vad_clean/temp/simo_sglang_pip.test_quant_onnx.sh.log.siplify.lstm.gpu007.rd.sio-software.com.SimoQuantizeLSTM.2026_08_04___18_31_08:518)、[no-LSTM 主日志](/share/users/like/package/jdjv/silero_vad_clean/temp/simo_sglang_pip.test_quant_onnx.sh.log.siplify.lstm.gpu007.rd.sio-software.com.no-lstm.2026_08_04___19_38_18:518)。
- 扫描两份主日志和 480 个 shard 日志，没有命中 `core dumped`、`segfault`、`Traceback`、`Exception`、`RuntimeError`、`ONNXRuntimeError`、`FATAL`、非零 `returncode`，工作目录下也没有 core 文件。

shard 日志中存在 ONNX Runtime 的 `transformer_memcpy` 性能 warning，例如[这个空 shard 日志](/share/users/like/package/jdjv/silero_vad_clean/logs/simo_sglang_pip.gpu007.rd.sio-software.com.SimoQuantizeLSTM.2026_08_04___18_31_08/onnx_quant_w_mxint8_a_mxint8_aishell4_24shards/shards/shard_020.log:4)。它提示图中加入了 Memcpy 节点，不是异常，也没有令 shard 失败。

因此，对问题 1 的回答是：**两次测试都完整结束；没有发现 core dump；没有抛出导致任务失败的异常；精度数据完整。**

### 88.2 对比口径

下面的差值统一定义为：

```text
delta = 启用 LSTM 量化后的指标 - no-LSTM 指标
```

`pp` 表示百分点，例如 `-1.605 pp` 表示指标从 0.927574 变为 0.911524，而不是相对下降 1.605%。两次运行使用相同模型、数据、1,431,607 个 chunk 和固定阈值 0.11。对这 10 对配置做规范化 JSON 比较后，移除新增的 LSTM module config，其他配置完全相同，因此可以把成对差异归因于启用 LSTM replacement 路径。

这里的“LSTM 量化影响”是端到端影响：标准 ONNX LSTM 被 `com.simo::SimoQuantizeLSTM` 替换，同时加入权重 DQ 和逐时间步状态/输入 QDQ。因此差值既包含量化误差，也可能包含 custom-op/ATen 与标准 ORT LSTM 的数值路径差异，不能用这两组 aggregate summary 把二者拆开。summary 也没有逐 utterance 配对指标或置信区间，所以表格是这批固定数据的确定性实测差值，不是统计显著性结论。

需要修正“Linear + Conv”这个表述：量化配置确实包含 Linear target，但这个 Silero VAD 简化后的实际命中统计中没有 Linear。九组常规配置是 `12 Conv` 对比 `12 Conv + 2 LSTM`；也就是对这个模型而言，实际比较是 **Conv 与 Conv + LSTM**。LSTM 主日志对九组配置均记录 `inserted_by_op={"Conv": 12, "LSTM": 2}`，no-LSTM 日志记录 `{"Conv": 12}`。

### 88.3 完整精度变化

| 量化配置 | no-LSTM ROC-AUC | +LSTM ROC-AUC | delta AUC | no-LSTM Acc@0.11 | +LSTM Acc@0.11 | delta Acc |
|---|---:|---:|---:|---:|---:|---:|
| `w_mxfp4_e2m1_a_mxfp4_e2m1_scale_int_sipu` | 0.927574 | 0.911524 | -1.605 pp | 0.633507 | 0.784626 | +15.112 pp |
| `w_mxfp6_e2m3_a_mxfp6_e2m3_scale_int_sipu` | 0.941489 | 0.922599 | -1.889 pp | 0.766768 | 0.851374 | +8.461 pp |
| `w_mxfp6_e3m2_a_mxfp6_e3m2_scale_int_sipu` | 0.936454 | 0.933828 | -0.263 pp | 0.729514 | 0.819539 | +9.002 pp |
| `w_int8_per_channel_a_int8_per_channel` | 0.946673 | 0.538793 | -40.788 pp | 0.828686 | 0.690298 | -13.839 pp |
| `w_int8_per_channel_a_int8_per_tensor` | 0.947514 | 0.558937 | -38.858 pp | 0.853845 | 0.746381 | -10.746 pp |
| `w_int8_per_tensor_a_int8_per_tensor` | 0.945295 | 0.547201 | -39.809 pp | 0.831788 | 0.698989 | -13.280 pp |
| `w_fp8_2d_block_a_fp8_1d_block` | 0.947480 | 0.913916 | -3.356 pp | 0.835417 | 0.562707 | -27.271 pp |
| `w_mxfp8_e4m3_a_mxfp8_e4m3_scale_int_sipu` | 0.941704 | 0.941186 | -0.052 pp | 0.765563 | 0.818387 | +5.282 pp |
| `w_mxfp8_e5m2_a_mxfp8_e5m2_scale_int_sipu` | 0.938949 | 0.935727 | -0.322 pp | 0.767449 | 0.839683 | +7.223 pp |
| `w_mxint8_a_mxint8` | 0.946729 | 0.933877 | -1.285 pp | 0.828048 | 0.884793 | +5.675 pp |

原始 float baseline 是 ROC-AUC `0.947480`、Acc@0.11 `0.835417`，见 [float summary](/share/users/like/package/jdjv/silero_vad_clean/logs/onnx_float_baseline_aishell4_24shards/summary.json)。

结果可以概括为：

- **ROC-AUC 在 10 种配置中全部下降。** 因为 ROC-AUC 不依赖固定阈值，这说明加入 LSTM 量化后，没有一种配置改善整体样本排序质量。
- **MXFP8 E4M3 最稳。** 它的增量 AUC 损失只有 `0.000518`，即 `0.052 pp`；启用 LSTM 后的绝对 AUC `0.941186` 也是 10 组中最高。
- MXFP6 E3M2 和 MXFP8 E5M2 的增量 AUC 损失也较小，分别为 `0.263 pp` 和 `0.322 pp`。MXINT8 损失 `1.285 pp`，MXFP4/MXFP6 E2M3 损失约 `1.6` 到 `1.9 pp`。
- **三种普通 INT8 LSTM 配置发生严重退化。** AUC 从约 `0.945` 到 `0.948` 降到 `0.539` 到 `0.559`，损失 `38.858` 到 `40.788 pp`，已经接近随机排序水平；固定阈值 accuracy 也下降 `10.746` 到 `13.839 pp`。这三组不应作为可接受的部署候选。
- `w_fp8_2d_block_a_fp8_1d_block` 也明显退化：AUC 损失 `3.356 pp`，Acc@0.11 损失 `27.271 pp`。这一组尤其有诊断价值，因为当前 Conv per-block FP8 不受支持：LSTM 运行只插入 2 个 LSTM、跳过 12 个 Conv，而 no-LSTM 运行插入 0 个节点、跳过 12 个 Conv，证据见 [LSTM 日志第 338 行](/share/users/like/package/jdjv/silero_vad_clean/temp/simo_sglang_pip.test_quant_onnx.sh.log.siplify.lstm.gpu007.rd.sio-software.com.SimoQuantizeLSTM.2026_08_04___18_31_08:338) 和 [no-LSTM 日志第 338 行](/share/users/like/package/jdjv/silero_vad_clean/temp/simo_sglang_pip.test_quant_onnx.sh.log.siplify.lstm.gpu007.rd.sio-software.com.no-lstm.2026_08_04___19_38_18:338)。因此该行基本隔离了 FP8 block LSTM 本身的影响；它的数据完整，但不是“Conv + LSTM”结果。
- MX 系列的 Acc@0.11 反而上升 `5.282` 到 `15.112 pp`，但这与 AUC 全部下降并不矛盾。`Acc@0.11` 使用固定阈值，LSTM 量化改变输出分数的尺度或校准后，单点 accuracy 可能上升，而全阈值范围内的排序能力仍下降。不能据此认定 MXFP4 比 no-LSTM 更好；部署前应重新扫描阈值，并优先结合 ROC-AUC 判断。

如果优先考虑阈值无关的质量，当前结果中首选是 MXFP8 E4M3，其次是 MXFP8 E5M2 或 MXFP6 E3M2。若只看当前固定阈值 0.11，MXINT8 的 `0.884793` 最高，但它的 AUC 比 no-LSTM 低 `1.285 pp`，这更像校准变化，而不是无条件的模型质量提升。

### 88.4 关于“正式 wheel 安装测试”的重要限定

这些结果确认评估进程使用了 wheel 环境。每个 shard 的命令行都是：

```text
/share_data/users/like/miniconda3/envs/simo_sglang_pip/bin/python ...
```

但是，这两次命令**没有加载 wheel 内打包的 custom-op 动态库**。`test_quant_onnx.sh` 的默认值令日志中的每个评估命令都显式传入：

```text
--custom-op-library /share_data/users/like/package/simo_conda_sglang/simo/onnx/ort_plugin/libSimoOnnxCustomOps_sm90.so
```

wheel 实际安装的动态库位于：

```text
/share_data/users/like/miniconda3/envs/simo_sglang_pip/lib/python3.12/site-packages/simo/onnx/ort_plugin/libSimoOnnxCustomOps_sm90.so
```

当前核对时，这两个 `.so` 的 SHA-256 也不同：

```text
外部显式加载的 .so:
84162622636f2043f95d609aa072451d717880b205d2ef16b781cf5a22fef936

wheel site-packages 中的 .so:
57a5214dbe881005440c35a7f29abf0374793f17973b1340ab55a5341c767b78
```

所以严谨的表述是：**wheel 安装的 Python `simo` 代码与显式指定的外部 custom-op `.so` 组合，完整通过了这两批精度测试。** 现有日志不能证明 wheel 自带的 `.so` 被加载或其 RUNPATH 已在这次业务测试中得到验证。

要做纯 wheel 闭环复测，应显式改用环境内插件路径，例如：

```bash
PY=/share_data/users/like/miniconda3/envs/simo_sglang_pip/bin/python
CUSTOM_OP_LIBRARY="$($PY -c 'from simo.onnx.runtime import get_custom_ops_library_path; print(get_custom_ops_library_path())')" \
PY="$PY" \
CONFIG_DIR=/share/users/like/package/jdjv/quant_schema/ \
MODEL=onnx_float_baseline/silero_vad.onnx \
OUTPUT_ROOT=logs/simo_sglang_pip.wheel-only.SimoQuantizeLSTM \
SIMPLIFY=true \
bash test_quant_onnx.sh
```

另外，shard 日志中的 provider 列表是 `['CUDAExecutionProvider', 'CPUExecutionProvider']`，所以这也是允许其他节点 CPU fallback 的精度测试，不是严格的全图 CUDA placement 测试；这不影响上面的精度对比和完整性结论。
