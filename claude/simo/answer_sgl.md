# SGLang vs vLLM SIMO量化性能差异分析

## 性能数据对比

| 框架 | 不量化 | 量化后 | 耗时倍数 |
|------|--------|--------|----------|
| SGLang | 1.91s | 17.26s | 9.04x |
| vLLM | 2.02s | 6.73s | 3.33x |

**结论**: SGLang使用SIMO量化后的性能下降(15.35s增量)是vLLM(4.71s增量)的约3.26倍。

## 根本原因分析

### 1. FusedMoE层实现的关键差异

#### vLLM的实现路径 (更高效)
```
SIMOFusedMoEMethod.apply()
  → fused_experts_impl() [simo/extensions/vllm_simo/layers/fused_moe/fused_moe.py]
    → invoke_fused_moe_kernel() [vLLM原生优化kernel]
```

vLLM的`fused_experts_impl`直接调用vLLM原生的`invoke_fused_moe_kernel`,该kernel对int4 w4a16有专门优化:
- 位于 `vllm.model_executor.layers.fused_moe.fused_moe`
- 针对(bit=4, group_size=32)有专门的Triton路径优化(见代码line 26-68的patch)
- 所有import都在模块顶层,无运行时开销

#### SGLang的实现路径 (有额外开销)
```
SIMOFusedMoEMethod.apply()
  → MoeRunner.run()
    → fused_experts_none_to_triton() [simo/.../fused_moe.py line 469-502]
      → fused_experts()
        → inplace_fused_experts()
          → fused_experts_impl()
            → invoke_fused_moe_kernel() [monkey-patched版本]
```

SGLang需要经过**更多的抽象层**,增加了函数调用开销。

### 2. 运行时import开销 (关键问题)

SGLang的代码在**热路径**中执行import语句,每次前向传播都会触发:

**fused_moe.py** (每次前向调用):
```python
# line 73-76
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
    get_config_dtype_str,
    try_get_optimal_moe_config,
)

# line 98-107 (在循环外但每次forward调用)
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    _down_moe_use_tma,
    swiglu_with_alpha_and_limit,
    ...
)

# line 176 (在chunk循环内!)
from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import moe_align_block_size

# line 181-183 (在chunk循环内!)
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
    invoke_fused_moe_kernel,
)
```

**fused_moe_triton_kernels.py invoke_fused_moe_kernel()内部**:
```python
# line 77-79 (每次kernel调用)
from simo.ops.mx_api import upcast_from_mxfmt, downcast_to_mxfmt
from simo.quantization import dtypes

# line 159-162 (每次kernel调用)
from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
    fused_moe_kernel_gptq_awq,
    fused_moe_kernel,
)
```

而vLLM的所有import都在模块顶层完成,无运行时开销。

### 3. 内存分配差异

**SGLang** - 在chunk循环**内部**分配intermediate_cache2 (line 167-171):
```python
for chunk in range((num_tokens // CHUNK_SIZE) + 1):
    ...
    intermediate_cache2 = torch.empty(  # 每个chunk都重新分配!
        (total_tokens, N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
```

**vLLM** - 在循环**外部**分配一次 (line 227-230):
```python
intermediate_cache2 = torch.empty(
    (M * top_k_num, N // 2), device=hidden_states.device, dtype=cache_dtype
)
for chunk in range(...):  # 复用同一块内存
    ...
```

### 4. Linear层实现 (两者相同 - 都是伪量化)

SGLang `SIMOLinearMethod.apply()` (line 910-930):
```python
def apply(self, layer, x, bias=None):
    x_q, scale_a = self.input_downcast_kernel(x)
    qdq_x = self.input_upcast_kernel(x_q, scale_a, x.dtype)  # 反量化
    dq_w = self.weight_upcast_kernel(layer.weight, layer.weight_scale, x.dtype)  # 反量化
    output = F.linear(qdq_x, dq_w, bias)  # 用float计算
    return output
```

vLLM `SIMOLinearMethod.apply()` (line 369-399):
```python
def apply(self, layer, input, bias=None):
    input_q, input_scale = self.input_downcast_kernel(input_2d)
    qdq_x = self.input_upcast_kernel(input_q, input_scale, output_dtype)  # 反量化
    dq_w = self.weight_upcast_kernel(layer.weight, layer.weight_scale, output_dtype)  # 反量化
    output = torch.matmul(qdq_x, dq_w.T)  # 用float计算
    return output
```

**两者都使用伪量化**(Quantize-Dequantize),权重和激活都先反量化回float再做矩阵乘。这解释了为什么两个框架在量化后都有显著性能下降。

## 总结

SGLang性能更差的主要原因:

| 问题 | 影响程度 |
|------|----------|
| 运行时import开销(函数内部import) | **高** |
| 更多的抽象层(MoeRunner等) | 中 |
| 循环内内存分配 | 中 |
| 函数调用链更长 | 低 |

## 优化建议

1. **将所有import移到模块顶层**,避免运行时import开销
2. **减少抽象层**,让SIMOFusedMoEMethod直接调用fused_experts_impl
3. **将内存分配移到循环外**,复用intermediate_cache2
4. **考虑使用真正的量化kernel**而非伪量化,避免反量化开销

## 代码位置参考

- SGLang FusedMoE实现: `/share_data/users/like/package/h100/package/simo_conda_sglang/simo/extensions/sglang_simo/layers/fused_moe/fused_moe.py`
- SGLang kernel patch: `/share_data/users/like/package/h100/package/simo_conda_sglang/simo/extensions/sglang_simo/layers/fused_moe/fused_moe_triton_kernels.py`
- vLLM FusedMoE实现: `/share_data/users/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/layers/fused_moe/fused_moe.py`

---

## 深入分析（2024-01更新）

### 5. torch op注册差异分析（待验证）

#### SGLang原生代码使用`direct_register_custom_op`

SGLang的`fused_moe.py`中将核心函数注册为torch原生op：

```python
# sglang/srt/layers/moe/fused_moe_triton/fused_moe.py
direct_register_custom_op(
    op_name="inplace_fused_experts",
    op_func=inplace_fused_experts,
    mutates_args=["hidden_states"],
    fake_impl=inplace_fused_experts_fake,
)

direct_register_custom_op(
    op_name="outplace_fused_experts",
    op_func=outplace_fused_experts,
    mutates_args=[],
    fake_impl=outplace_fused_experts_fake,
)
```

SGLang原生的unquant/FP8量化通过以下方式调用：
```python
# 直接调用torch op
torch.ops.sglang.inplace_fused_experts(...)
# 或
torch.ops.sglang.outplace_fused_experts(...)
```

#### `direct_register_custom_op`的性能优势

查看`sglang/srt/utils/common.py:2002-2068`，这个函数的作用是：

```python
def direct_register_custom_op(...):
    """
    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    """
    my_lib.define(op_name + schema_str)
    my_lib.impl(op_name, op_func, "CUDA")  # 直接注册到CUDA后端
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)
```

注册为torch op的优势：
1. **跳过Python调度开销**：直接分派到CUDA后端，无需Python解释器介入
2. **torch.compile兼容**：可以被torch.compile识别和优化
3. **CUDA图支持**：支持CUDA Graph捕获，减少kernel launch开销
4. **减少GIL竞争**：在C++层面执行，释放Python GIL

#### SIMO SGLang扩展的调用路径

SIMO的SGLang扩展**没有**使用torch op路径，而是普通的Python函数调用：

```
SIMOFusedMoEMethod.apply()
  → MoeRunner.run()                    [Python函数]
    → TritonRunnerCore.run()           [Python函数]
      → invoke_fused_moe_kernel()      [Python函数]
        → fused_moe_kernel_gptq_awq[]  [Triton kernel]
```

每一层都是Python函数调用，无法享受torch op的优化。

#### 对比分析

| 特性 | SGLang原生 (torch op) | SIMO扩展 (Python函数) |
|------|----------------------|----------------------|
| Python调度开销 | 无 | 有（多层函数调用） |
| torch.compile优化 | 支持 | 不支持 |
| CUDA Graph捕获 | 支持 | 受限 |
| GIL竞争 | 最小 | 每次调用都需要GIL |

#### 验证方法

可以通过以下方式验证torch op注册是否是主要原因：

1. **Profiling**：使用`torch.profiler`对比两种路径的CPU时间
2. **Mock测试**：将SIMO的核心函数也注册为torch op，对比性能差异
3. **torch.compile测试**：在SIMO路径上启用torch.compile，观察是否有改善

#### 初步结论

虽然torch op注册可能带来性能优势，但这**可能不是主要原因**，因为：
- SGLang原生FP8量化也是通过`MoeRunner.run() → TritonRunnerCore.run()`路径调用（见`fp8.py:1165`）
- FP8性能很好(1.69s)，说明MoeRunner路径本身不是瓶颈

### 6. 实际调用路径深入分析

经过代码追踪，发现SGLang的FP8和SIMO的int4都使用相同的MoeRunner路径：

#### FP8路径（快速：~1.69s）
```python
# fp8.py Fp8MoEMethod.apply()
quant_info = TritonMoeQuantInfo(
    use_fp8_w8a8=True,
    ...
)
return self.runner.run(dispatch_output, quant_info)
```

#### SIMO int4路径（慢：~17s）
```python
# quantization.py SIMOFusedMoEMethod.apply()
quant_info = TritonMoeQuantInfo(
    use_int4_w4a16=True,
    block_shape=[0, group_size],
    ...
)
return self.runner.run(dispatch_output, quant_info)
```

两者都走`MoeRunner → TritonRunnerCore.run() → invoke_fused_moe_kernel()`路径。

### 7. 关键差异：不同的Triton Kernel

真正的差异在于最底层调用的Triton kernel不同：

#### FP8使用`fused_moe_kernel`
```python
# invoke_fused_moe_kernel中的条件判断
if use_fp8_w8a8:
    # 激活量化
    A, A_scale = scaled_fp8_quant(A, A_scale, ...)
    # 调用通用kernel
    fused_moe_kernel[grid](...)
```

#### int4使用`fused_moe_kernel_gptq_awq`
```python
if (use_int8_w8a16 or use_int4_w4a16) and block_shape is not None and block_shape[1] > 0:
    # 调用GPTQ/AWQ专用kernel
    fused_moe_kernel_gptq_awq[grid](...)
```

#### Kernel复杂度对比

**fused_moe_kernel**：
- 直接加载FP8数据
- 简单的scale乘法
- 可利用FP8 Tensor Core

**fused_moe_kernel_gptq_awq**：
- 需要解包int4（位操作）
- 需要应用zero point
- 需要应用scale
- 更多的内存访问
- 更多的算术运算

```python
# fused_moe_kernel_gptq_awq中的解包和反量化
b = tl.load(b_ptrs)
if use_int4_w4a16:
    b = (b >> b_shifter) & 0xF  # 解包
b_scale = tl.load(b_scale_ptrs)
if has_zp:
    b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
else:
    b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)  # 反量化
```

### 8. 为什么vLLM SIMO更快？

vLLM SIMO和SGLang SIMO都使用`fused_moe_kernel_gptq_awq`，但vLLM更快的可能原因：

1. **调用路径更短**：
   - vLLM: `apply() → fused_experts_impl() → invoke_fused_moe_kernel()`
   - SGLang: `apply() → MoeRunner.run() → pre_permute → TritonRunnerCore.run() → invoke_fused_moe_kernel()`

2. **Chunk处理优化**：
   - vLLM的`fused_experts_impl`有CHUNK循环，可以更好地管理内存和cache
   - SGLang的`TritonRunnerCore.run`没有明显的chunk机制

3. **pre_permute/post_permute开销**：
   - SGLang的MoeRunner在调用kernel前后有额外的permute操作

### 9. 总结与建议

#### 性能差异的多重原因

| 原因 | 影响程度 | 可优化性 |
|------|----------|----------|
| 不同的Triton kernel (GPTQ vs 通用) | **高** | 需要优化kernel |
| 更长的Python调用链 | 中 | 可以精简 |
| MoeRunner的permute开销 | 中 | 需要评估 |
| 未注册为torch op | 中-低 | 可以注册 |

#### 优化建议

1. **短期优化**：
   - 精简调用链，直接从`SIMOFusedMoEMethod.apply()`调用`invoke_fused_moe_kernel`
   - 将关键函数注册为torch op

2. **长期优化**：
   - 优化`fused_moe_kernel_gptq_awq`的性能
   - 考虑使用真正的int4 Tensor Core计算（如Hopper架构）
   - 评估是否可以使用FP8计算路径替代int4

3. **验证步骤**：
   - 使用`torch.profiler`定位真正的瓶颈
   - 对比SGLang原生GPTQ量化和SIMO量化的性能
   - 测试将SIMO函数注册为torch op后的性能变化

---

## 新增分析（MoeWNA16Method vs SIMOFusedMoEMethod 性能差异）

### 10. 新的性能数据

| 测试场景 | 模型 | 时间 |
|---------|------|------|
| SGLang原生AWQ (moe_wna16) | DeepSeek-V2-Lite-Chat-AWQ | 5.81s |
| SIMO注册 + 原生AWQ (moe_wna16) | DeepSeek-V2-Lite-Chat-AWQ | 4.43s |
| SIMO自身量化 (simo) | DeepSeek-V2-Lite-Chat (FP) | 17.06s |

### 11. 关键发现：SIMO注册后AWQ更快

`SIMO_SGLANG_REGISTER=1`启用后，SIMO的`fused_experts_none_to_triton`会接管SGLang原生实现。
AWQ走SIMO路径反而更快（4.43s vs 5.81s），说明SIMO的fused_moe实现本身效率不差。

### 12. MoeWNA16Method vs SIMOFusedMoEMethod：同一MoE路径不同性能

两者的MoE层都走完全相同的SIMO路径：
```
self.runner.run()
  → fused_experts_none_to_triton (SIMO版)
    → fused_experts (SIMO版)
      → inplace_fused_experts (SIMO版)
        → fused_experts_impl (SIMO版)
          → invoke_fused_moe_kernel()
            → fused_moe_kernel_gptq_awq[] (Triton kernel)
```

但总时间差异巨大：4.43s vs 17.06s。差异来源有两个层面：

#### 原因一（主要）：非MoE线性层的伪量化开销

**SIMO `--quantization simo`**：所有非MoE线性层使用`SIMOLinearMethod.apply()`，这是伪量化(QDQ)：

```python
# simo/extensions/sglang_simo/quantization/quantization.py:916-936
def apply(self, layer, x, bias=None):
    x_q, scale_a = self.input_downcast_kernel(x)       # ① 量化激活
    qdq_x = self.input_upcast_kernel(x_q, scale_a, x.dtype)  # ② 反量化激活
    dq_w = self.weight_upcast_kernel(layer.weight, layer.weight_scale, x.dtype)  # ③ 反量化权重
    output = F.linear(qdq_x, dq_w, bias)               # ④ float matmul
    return output
```

每次forward需要4步：量化激活 → 反量化激活 → 反量化权重 → float矩阵乘。
**所有attention层的q_proj、kv_a_proj、kv_b_proj、o_proj都经历这个开销。**

**AWQ `--quantization moe_wna16`**：非MoE线性层使用AWQ/GPTQ的优化kernel：

```python
# sglang/srt/layers/quantization/moe_wna16.py:198-212
elif isinstance(layer, LinearBase):
    if self.linear_quant_method == "gptq":
        return GPTQMarlinConfig.from_config(...).get_quant_method(layer, prefix)
    elif self.linear_quant_method == "awq":
        return AWQConfig.from_config(...).get_quant_method(layer, prefix)
```

AWQ/GPTQ使用Marlin或AWQ优化kernel，直接在量化格式上做矩阵乘，**不需要反量化到float**。

#### 性能差异估算

- 不量化基线：1.91s
- SIMO总时间：17.06s → 额外开销：15.15s
- AWQ (SIMO fused_moe)：4.43s → 额外开销：2.52s
- MoE层额外开销（假设两者相似）：约2-3s
- **非MoE层伪量化开销**：约12-13s（= 15.15s - 2.52s）

DeepSeek-V2-Lite有27个decoder层，每层有4个attention线性层（q_proj、kv_a_proj、kv_b_proj、o_proj），
共108次线性层调用 × 每次伪量化约0.12s ≈ 12.6s，完全可以解释差异。

#### 原因二（次要）：MoE层scale的dtype差异

**SIMO的scale是float32**：

```python
# simo/ops/formats/flexpoint/quant.py:156
amax = x_.to(torch.float32).abs().max(dim=-1, keepdim=True)[0]
x_s = amax / max_value  # float32 / float32 → float32
```

**AWQ的scale是params_dtype（bf16/fp16）**：

```python
# moe_wna16.py:294-300
w13_scales = torch.nn.Parameter(
    torch.zeros(num_experts, 2 * intermediate_size_per_partition,
                hidden_size // group_size,
                dtype=params_dtype),  # bf16或fp16
    requires_grad=False,
)
```

在`fused_moe_kernel_gptq_awq`的内循环中，每个K block都要加载scale：
```python
b_scale = tl.load(b_scale_ptrs, ...)  # fp32: 4 bytes/element vs fp16: 2 bytes/element
```

float32的scale比fp16的内存带宽占用多一倍，这在kernel内循环中会造成额外开销。
SGLang SIMO扩展的`process_weights_after_loading`中**缺少scale dtype转换**（vLLM扩展有此转换）。

### 13. 结论

| 因素 | 影响 | 时间估算 |
|------|------|----------|
| **非MoE线性层伪量化（QDQ）** | **主要** | ~12-13s |
| MoE层scale dtype (fp32 vs fp16) | 次要 | ~0.5-1s |
| MoE层权重打包格式差异 | 待验证 | 未知 |

### 14. 优化建议

1. **高优先级：消除非MoE线性层的伪量化开销**
   - 使用真正的量化kernel（如Marlin、Cutlass INT4）替代QDQ
   - 或者仅对MoE层做量化，非MoE层保持FP16/BF16

2. **中优先级：修复scale dtype**
   - 在`process_weights_after_loading`中添加scale dtype转换（参考vLLM扩展）：
   ```python
   if self.weight_spec.dtype in ["int4", "int8"] and self.input_spec.dtype in ["float16", "bfloat16"]:
       target_dtype = torch.bfloat16 if self.input_spec.dtype == "bfloat16" else torch.float16
       layer.w13_weight_scale = torch.nn.Parameter(
           layer.w13_weight_scale.data.to(target_dtype).contiguous(), requires_grad=False
       )
   ```

3. **低优先级：验证权重打包格式**
   - 确认SIMO的int4打包格式与`fused_moe_kernel_gptq_awq`的期望格式完全一致

---

## 新增分析：lm_eval离线测试 vs 在线测试的`promote_trition_scale_precision_gptq_awq`异常现象

### 15. 问题现象

| 测试方式 | promote=false | promote=true | 差异 | 符合预期? |
|---------|--------------|--------------|------|----------|
| **在线测试** | 0.5515 | 0.5531 | +0.0016 | ✓ 是 |
| **离线测试** | 0.5518 | 0.5509 | -0.0009 | ✗ 否 |

- **在线测试**：fp32 scale得分更高，符合理论预期（更高精度应该带来更好的准确率）
- **离线测试**：fp32 scale得分反而更低，与理论预期相反

### 16. 在线测试 vs 离线测试的差异

| 特性 | 在线测试 (`--model local-completions`) | 离线测试 (`--model sglang`) |
|------|----------------------------------------|----------------------------|
| 执行方式 | 独立sglang server进程 + HTTP API | lm_eval进程内直接加载模型 |
| batch_size | 固定16 | auto（动态调整） |
| 模型加载 | server启动时加载一次 | 每次测试都加载 |
| CUDA context | server专用 | 与lm_eval共享 |
| Triton JIT缓存 | server进程内持续 | 可能每次重新编译 |

### 17. 可能的原因分析

#### 原因一：Triton JIT编译差异（高可能性）

`fused_moe_kernel_gptq_awq_fp32_scale`是一个复杂的Triton kernel，通过uint16视图加载fp32 scale：

```python
# 加载两个uint16并组合成fp32
b_scale_low = tl.load(b_scale_base)
b_scale_high = tl.load(b_scale_base + 1)
low_u32 = b_scale_low.to(tl.uint32)
high_u32 = b_scale_high.to(tl.uint32)
combined_u32 = (high_u32 << 16) | low_u32
b_scale = combined_u32.to(tl.float32, bitcast=True)
```

不同的运行环境可能导致：
- Triton生成不同的PTX代码
- 内存访问模式不同
- 寄存器分配差异

#### 原因二：batch_size差异（中等可能性）

- 在线测试：`batch_size=16`（固定）
- 离线测试：`batch_size=auto`（可能更大或更小）

不同的batch size可能导致：
- 不同的CUDA kernel配置（grid size, block size）
- 不同的内存访问模式
- 不同的数值精度行为（累加顺序不同）

#### 原因三：Triton缓存问题（中等可能性）

离线测试可能使用了旧的Triton编译缓存，导致：
- 使用了错误版本的kernel
- 缓存的kernel与当前代码不匹配

#### 原因四：数值稳定性边界情况（低可能性）

fp32 scale kernel通过位操作重建fp32值：
```python
combined_u32 = (high_u32 << 16) | low_u32
b_scale = combined_u32.to(tl.float32, bitcast=True)
```

某些特殊的scale值（如subnormal数、非常小的值）可能在这个过程中出现问题。

### 18. 验证建议

1. **清理Triton缓存后重新测试**
   ```bash
   rm -rf ~/.triton/cache
   # 然后重新运行离线测试
   ```

2. **添加调试日志确认kernel调用**
   在`invoke_fused_moe_kernel`中添加：
   ```python
   if promote_trition_scale_precision_gptq_awq and B_scale.dtype == torch.float32:
       print(f"[DEBUG] Using fp32 scale kernel, B_scale.shape={B_scale.shape}, dtype={B_scale.dtype}")
   ```

3. **统一batch_size进行对比测试**
   ```bash
   # 离线测试也使用固定batch_size=16
   lm_eval --model sglang ... --batch_size 16
   ```

4. **使用相同代码版本同时进行在线和离线测试**
   确保两种测试方式使用完全相同的代码commit。

5. **比较kernel输出**
   对同一输入，分别用`fused_moe_kernel_gptq_awq`和`fused_moe_kernel_gptq_awq_fp32_scale`计算，比较输出差异：
   ```python
   # 用bf16 scale的标准kernel
   output1 = fused_moe_kernel_gptq_awq(A, B, B_scale_bf16, ...)

   # 用fp32 scale的优化kernel
   output2 = fused_moe_kernel_gptq_awq_fp32_scale(A, B, B_scale_fp32_as_u16, ...)

   # 比较差异
   diff = (output1 - output2).abs().max()
   print(f"Max diff: {diff}")
   ```

### 19. 临时解决方案

如果离线测试必须使用，可以：

1. **禁用fp32 scale优化**（回退到标准kernel）
   ```json
   {
     "quantization_config": {
       ...
       "promote_trition_scale_precision_gptq_awq": false
     }
   }
   ```

2. **或者改用在线测试方式**进行准确率评估

### 20. 根本原因推测

最可能的根本原因是：**离线测试时`fused_moe_kernel_gptq_awq_fp32_scale`的Triton编译结果与在线测试时不同**。

这可能是由于：
- 不同的CUDA driver版本
- 不同的Triton版本或配置
- 不同的GPU状态（显存碎片、温度等）
- 运行时的随机因素

### 21. 长期建议

1. **添加kernel单元测试**
   为`fused_moe_kernel_gptq_awq_fp32_scale`添加正确性单元测试，确保其输出与标准kernel（使用bf16 scale）在精度上可接受的范围内一致。

2. **考虑替代实现方案**
   如果uint16视图加载fp32的方案不稳定，可以考虑：
   - 直接使用bf16/fp16 scale（放弃fp32精度）
   - 使用PyTorch原生的fp32 scale加载（放弃Triton优化）
   - 使用CUDA原生kernel而非Triton

3. **完善CI测试流程**
   在CI中同时测试在线和离线方式，确保两种方式的结果一致

---

## 22. pip install 没有安装 sglang_simo/layers/ 目录的原因及修复

### 问题描述

执行 `pip install . --no-build-isolation` 后，`simo/extensions/sglang_simo/layers/` 目录没有被安装到 site-packages 目录下：

```
/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/simo/extensions/sglang_simo/
├── __init__.py
├── model_loader/     ✓ 已安装
├── quantization/     ✓ 已安装
└── layers/           ✗ 缺失！
```

### 根因分析

#### 1. `setup.py` 中的包发现机制

`setup.py` 第92行使用 `find_packages()` 来自动发现所有 Python 包：

```python
setup(
  ...
  packages=find_packages(exclude=["examples", "simo.csrc", "simo.csrc.*"]),
  ...
)
```

`find_packages()` 函数的工作原理是：**递归扫描目录树，只有包含 `__init__.py` 文件的目录才会被识别为 Python 包**。

#### 2. 缺少 `__init__.py` 文件

`simo/extensions/sglang_simo/layers/` 目录缺少 `__init__.py` 文件：

```
simo/extensions/sglang_simo/
├── __init__.py                          ✓
├── model_loader/
│   └── __init__.py                      ✓
├── quantization/
│   └── __init__.py                      ✓
└── layers/
    ├── __init__.py                      ✗ 缺失！
    └── fused_moe/
        └── __init__.py                  ✓
```

由于 `layers/` 目录没有 `__init__.py`，`find_packages()` 不会将其识别为 Python 包。这同时导致其子包 `layers/fused_moe/` 也无法被发现，因为**父目录不是合法的包，子目录即使有 `__init__.py` 也不会被递归扫描到**。

#### 3. 验证

修复前，`find_packages()` 只发现了以下 sglang 相关包：

```
simo.extensions.sglang_simo
simo.extensions.sglang_simo.model_loader
simo.extensions.sglang_simo.quantization
```

注意 `simo.extensions.sglang_simo.layers` 和 `simo.extensions.sglang_simo.layers.fused_moe` **完全缺失**。

### 修复方法

创建空的 `__init__.py` 文件：

```bash
touch simo/extensions/sglang_simo/layers/__init__.py
```

修复后，`find_packages()` 正确发现所有包：

```
simo.extensions.sglang_simo
simo.extensions.sglang_simo.layers              ← 新增
simo.extensions.sglang_simo.layers.fused_moe    ← 新增
simo.extensions.sglang_simo.model_loader
simo.extensions.sglang_simo.quantization
```

### 修复后重新安装

```bash
# 重新安装以使修复生效
pip install . --no-build-isolation

# 或者开发模式安装
pip install -e . --no-build-isolation
```

### 验证安装结果

```bash
# 检查 layers 目录是否已安装
python3 -c "import simo.extensions.sglang_simo.layers.fused_moe; print('layers/fused_moe 安装成功')"

# 或者检查 site-packages 目录
ls -la $(python3 -c "import simo.extensions.sglang_simo as m; import os; print(os.path.dirname(m.__file__))")/layers/
```

### 经验教训

在添加新的 Python 包目录时，必须确保：
1. **每个包目录都包含 `__init__.py` 文件**（即使是空文件）
2. **整个包层级链上的每一级都有 `__init__.py`** — 如果中间某一级缺少，其下所有子包都不会被 `find_packages()` 发现
3. 可以通过 `python3 -c "from setuptools import find_packages; print(find_packages())"` 快速验证包发现是否正确

---

## 23. nvfp4 (w4a4_nvfp) 量化：SGLang MMLU=0.4434 vs vLLM MMLU=0.5298 差异分析

### 测试环境

| 项目 | vLLM | SGLang |
|------|------|--------|
| conda环境 | simo_vllm | simo_sglang |
| 代码路径 | simo/extensions/vllm_simo/ | simo/extensions/sglang_simo/ |
| 模型 | DeepSeek-V2-Lite-Chat | DeepSeek-V2-Lite-Chat |
| 量化配置 | quant_config_w4a4_nvfp.json | quant_config_w4a4_nvfp.json |
| MMLU得分 | **0.5298** | **0.4434** |
| 差距 | — | **-0.0864 (16.3%相对下降)** |

### 23.1 根因一（最可能）：量化配置文件不同

两个框架使用的量化配置文件**名称相同但内容不同**：

**vLLM配置** (`simo/extensions/vllm_simo/example/.../quant_config_w4a4_nvfp.json`):
```json
{
    "excludes": [
        "lm_head",
        "*visual*"
    ],
    "flash_comm": "fast_all2all"
}
```

**SGLang配置** (`simo/extensions/sglang_simo/example/.../quant_config_w4a4_nvfp.json`):
```json
{
    "excludes": [
        "lm_head",
        "*visual*",
        "re:.*kv_b_proj"
    ],
    "flash_comm": null
}
```

**关键差异**：

| 配置项 | vLLM | SGLang | 影响 |
|--------|------|--------|------|
| `excludes` | 2项 | 3项（多了`re:.*kv_b_proj`） | SGLang不量化kv_b_proj |
| `flash_comm` | `"fast_all2all"` | `null` | 通信优化差异 |

**`re:.*kv_b_proj` 排除的影响**：

DeepSeek-V2-Lite使用MLA（Multi-head Latent Attention）架构，`kv_b_proj`是将潜在空间投影回KV空间的关键层。在SGLang配置中排除`kv_b_proj`的量化，意味着该层保持原始精度。

但这里的矛盾是：**排除量化通常应该提升精度而非降低精度**。因此配置差异不是导致SGLang得分更低的直接原因，但它改变了量化的总体行为模式。

**但需要注意**：如果用户运行时使用的是不同路径的config文件，则需要确认实际使用的是哪个config。从日志中确认：
- vLLM日志显示excludes为`["lm_head", "*visual*"]`
- SGLang日志显示excludes为`["lm_head", "*visual*", "re:.*kv_b_proj"]`

### 23.2 根因二：quantization代码实现差异（已修复但可能残留问题）

在本次会话中修复了以下SGLang实现问题：

#### 修复1：`get_downcast_kernel` 不支持nvfp4的e4m3 scale mode

**修复前**：只处理`e8m0_*`系列scale mode，nvfp4的`e4m3`返回`None`导致运行时crash。

**修复后**：使用`ObserverMode.from_string()`和`ScaleModeEnum.from_string()`通用处理所有MX格式。

#### 修复2：`get_upcast_kernel` 带有不必要的nvfp4特殊参数

**修复前**：nvfp4路径传入`global_amax`和`shard_sizes`参数。

**修复后**：简化为与vLLM一致的简单lambda。

#### 修复3：`apply`方法output shape/dtype处理

**修复前**：直接用`output.view(*x.shape[:-1], output.shape[-1])`，缺少dtype转换。

**修复后**：使用`output.to(dtype=x.dtype).view(*output_shape)`与vLLM一致。

#### 修复4：添加nvfp4 global scale支持

添加了`global_scale_factor`计算、`weight_global_scale`参数注册、`input_global_scale`计算等完整的nvfp4两级scale机制。

**这些修复在测试时是否已经全部生效需要确认**。如果MMLU测试是在所有修复完成之前运行的，则得分低是因为代码bug而非框架差异。

### 23.3 根因三：框架级差异

即使量化代码完全一致，SGLang和vLLM在以下方面仍有差异：

1. **Attention实现**：SGLang和vLLM使用不同的attention backend（FlashAttention/FlashInfer等），数值精度可能略有不同
2. **MLA实现细节**：DeepSeek-V2的MLA在两个框架中的实现可能有差异
3. **Token采样**：logprob计算和采样策略的差异
4. **Tokenizer/Chat template处理**：可能影响输入格式

### 23.4 验证建议

1. **使用完全相同的config文件**：将vLLM的config（不排除kv_b_proj）用于SGLang测试，消除配置差异
   ```bash
   # 复制vllm config到sglang
   cp simo/extensions/vllm_simo/example/.../quant_config_w4a4_nvfp.json \
      simo/extensions/sglang_simo/example/.../quant_config_w4a4_nvfp.json
   ```

2. **确认所有代码修复已生效**：重新安装simo包后运行测试
   ```bash
   conda activate simo_sglang
   pip install -e . --no-build-isolation
   ```

3. **对比不量化的基线**：在两个框架上运行不量化的MMLU测试，确认基线是否一致
   ```bash
   # 不带 --quantization simo 参数运行
   ```

4. **逐步排除法**：
   - 只量化weight不量化activation（w4a16），对比两个框架
   - 排除MoE层只量化attention层，对比两个框架
   - 使用相同config排除kv_b_proj，对比两个框架

### 23.5 结论

| 可能原因 | 可能性 | 影响程度 |
|----------|--------|----------|
| **代码bug未完全修复（修复前运行测试）** | **高** | **高** |
| 配置文件差异（excludes/flash_comm不同） | 中 | 中 |
| 框架级attention/MLA实现差异 | 中 | 中-低 |
| 数值精度累积误差 | 低 | 低 |

**最可能的原因**是MMLU测试在nvfp4相关代码修复完成之前运行。建议在所有修复完成并重新安装后，使用完全相同的config文件重新测试。

---

## 24. 深度分析：修复后SGLang nvfp4 MMLU仍为0.4378（2026-03-03）

### 24.1 问题背景

所有之前的代码修复（get_downcast_kernel、get_upcast_kernel、apply方法、global_scale_factor支持）已应用并重新安装（`pip install -e . --no-build-isolation`）。但SGLang nvfp4 MMLU得分仍为**0.4378**，与vLLM的**0.5298**相差0.092。

kv_b_proj在SGLang中被正确排除，因为DeepSeek-V2的SGLang实现中，kv_b_proj不通过linear.forward()前向计算，而是直接使用torch.bmm对权重进行矩阵乘法（见`deepseek_v2.py`的`q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)`），量化后的权重会导致dtype不匹配。

### 24.2 关键证据：只有nvfp4出现问题

对比所有量化格式的MMLU得分：

| 格式 | vLLM | SGLang | 差异 |
|------|------|--------|------|
| w8a8_fp8_per_block | 0.5683 | 0.5687 | +0.0004 |
| w8a8_fp8_per_channel | 0.5604 | 0.5607 | +0.0003 |
| w6a6_mxfp | 0.5624 | 0.5652 | +0.0028 |
| w4a16_int4 | 0.5444 | 0.5531 | +0.0087 |
| w4a4_mxfp | 0.4908 | 0.4915 | +0.0007 |
| **w4a4_nvfp** | **0.5298** | **0.4378** | **-0.0920** |

**所有其他格式SGLang和vLLM得分几乎完全一致（差异<0.01），唯独nvfp4相差0.092。** 这证明问题出在nvfp4特有的代码路径中，而非框架级别的差异。

### 24.3 穷举代码对比结果

对以下nvfp4特有的代码路径进行了逐行对比，均发现功能等价：

| 代码路径 | SGLang文件:行号 | vLLM文件:行号 | 结论 |
|----------|----------------|---------------|------|
| `get_downcast_kernel` | quantization.py:296-340 | quantization_method.py:85-128 | **完全相同** |
| `get_upcast_kernel` | quantization.py:342-372 | quantization_method.py:131-161 | **完全相同** |
| `online_moe_weight_loader` | quantization.py:1032-1117 | quantization_method.py:605-694 | **完全相同** |
| `SIMOLinearMethod.apply()` | quantization.py:935-968 | quantization_method.py:386-440 | **功能等价** |
| `SIMOFusedMoEMethod.apply()` | quantization.py:1323-1358 | quantization_method.py:967-1006 | **调用相同函数** |
| `fused_experts_impl` | fused_moe.py:1-367 | 共享代码 | **完全相同** |
| `create_weights`（MoE） | quantization.py:1120-1242 | quantization_method.py:696-851 | **功能等价** |
| `process_weights_after_loading` | quantization.py:1244-1288 | quantization_method.py:853-895 | **功能等价** |
| `global_scale_factor计算` | quantization.py:1017-1023 | quantization_method.py:587-593 | **完全相同** |
| `_load_per_tensor_weight_scale` | layer.py:325-344 | vLLM layer等价 | **功能等价** |

### 24.4 发现的关键架构差异：FusedMoEQuantConfig创建时机

**这是唯一发现的结构性差异。**

#### vLLM的延迟初始化（正确方式）

```
vLLM FusedMoE.__init__:
  1. create_weights() → 创建空参数
  2. 权重加载 → 填充参数数据
  3. process_weights_after_loading() → 重新包装参数
  4. 首次推理时 ensure_moe_quant_config_init() → 延迟创建FusedMoEQuantConfig
```

vLLM的`FusedMoE`层实现了延迟初始化模式（`layer.py:1486-1492`）：
```python
def ensure_moe_quant_config_init(self):
    if self.quant_method.moe_quant_config is None:
        # Note: the moe_quant_config can't be constructed until after
        # weight loading post processing.
        self.quant_method.moe_quant_config = (
            self.quant_method.get_fused_moe_quant_config(self)
        )
```

注释明确说明：**"moe_quant_config不能在权重加载后处理完成之前构建"**。

#### SGLang的提前初始化（可能有问题）

```
SGLang FusedMoE.__init__:
  1. create_weights() → 创建空参数
  2. create_moe_runner() → 立即创建FusedMoEQuantConfig ← ⚠️ 权重还未加载！
  3. 权重加载 → 填充参数数据
  4. process_weights_after_loading() → 重新包装参数
```

SGLang的`FusedMoE.__init__`（`layer.py:292-306`）在权重加载之前就调用了`create_moe_runner()`：
```python
self.quant_method.create_weights(layer=self, ...)  # 步骤1
self.quant_method.create_moe_runner(self, self.moe_runner_config)  # 步骤2 - 此时权重为空！
```

### 24.5 引用有效性分析

`FusedMoEQuantConfig`存储的是参数对象的**引用**（而非副本）：

```python
# create_moe_runner中：
self.moe_quant_config = FusedMoEQuantConfig.make(
    w1_scale=layer.w13_weight_scale,         # 引用到Parameter对象
    g1_alphas=layer.w13_weight_global_scale,  # 引用到Parameter对象
    ...
)
```

理论上引用应该保持有效：
- **global scale**: `w13_weight_global_scale`在`process_weights_after_loading`中**不被替换**，权重加载通过in-place修改`param.data[expert_id][idx]`填充数据，引用有效
- **weight scale**: `w13_weight_scale`在`process_weights_after_loading`中**被替换**为新Parameter，但如果`.contiguous()`是no-op（已经连续），新旧Parameter共享同一底层存储

**但是，如果底层tensor在process_weights_after_loading中因`.contiguous()`创建了新拷贝（原tensor不连续），则旧引用指向的数据可能是正确的但不是最终的。**

### 24.6 为什么其他格式不受影响

其他量化格式（mxfp4、fp8等）不需要`global_scale`，也不使用`FusedMoEQuantConfig`中的`alpha_or_gscale`字段。它们仍然使用`_w1.scale`引用，但由于scale数据通过in-place操作加载到已连续的tensor中，即使引用指向旧Parameter，数据仍然正确。

**nvfp4的独特之处**在于它使用了`alpha_or_gscale`（全局缩放因子），这是nvfp4特有的功能。如果这个引用因为任何原因（如PyTorch内部的tensor别名管理）变得无效或读到未初始化的值，就会导致精度问题。

### 24.7 推荐的运行时调试方案

由于静态代码分析无法确定最终根因（代码路径功能等价），需要通过运行时插桩对比中间值：

#### 方案1：验证global scale引用有效性

在SGLang的`SIMOFusedMoEMethod.apply()`开头添加一次性检查：
```python
# 在apply()方法开头添加（仅第一次调用时打印）
if not hasattr(self, '_debug_checked'):
    self._debug_checked = True
    gs = self.moe_quant_config._w1.alpha_or_gscale
    print(f"[DEBUG] g1_alphas is layer ref: {gs is layer.w13_weight_global_scale}")
    print(f"[DEBUG] g1_alphas data_ptr match: {gs.data_ptr() == layer.w13_weight_global_scale.data_ptr()}")
    print(f"[DEBUG] g1_alphas values: {gs}")
    print(f"[DEBUG] layer.w13_weight_global_scale values: {layer.w13_weight_global_scale}")

    ws = self.moe_quant_config._w1.scale
    print(f"[DEBUG] w1_scale is layer ref: {ws is layer.w13_weight_scale}")
    print(f"[DEBUG] w1_scale data_ptr match: {ws.data_ptr() == layer.w13_weight_scale.data_ptr()}")
```

#### 方案2：对比权重加载后的global scale值

在`process_weights_after_loading`结尾添加：
```python
if hasattr(layer, 'w13_weight_global_scale'):
    print(f"[DEBUG] w13_weight_global_scale: {layer.w13_weight_global_scale}")
    print(f"[DEBUG] w2_weight_global_scale: {layer.w2_weight_global_scale}")
```

#### 方案3：对比fused_experts_impl的中间值

在`fused_experts_impl`的全局scale计算处添加：
```python
if use_nvfp4_global_scale:
    print(f"[DEBUG] g1_alphas: {g1_alphas}")
    print(f"[DEBUG] g2_alphas: {g2_alphas}")
    print(f"[DEBUG] g1_alphas is None: {g1_alphas is None}")
    print(f"[DEBUG] g1_alphas.shape: {g1_alphas.shape if g1_alphas is not None else 'N/A'}")
    print(f"[DEBUG] global_scale_factor: {global_scale_factor}")
```

#### 方案4（最直接）：修复timing问题

将SGLang的`create_moe_runner`改为延迟初始化模式，与vLLM一致：

```python
# 在SIMOFusedMoEMethod.apply()中延迟创建moe_quant_config：
def apply(self, layer, dispatch_output):
    if self.moe_quant_config is None:
        self.moe_quant_config = self._create_moe_quant_config(layer)
    # ... 原有逻辑
```

**这是最可能解决问题的方案**，因为它直接复制了vLLM已验证工作的延迟初始化模式。

### 24.8 总结

| 发现 | 详情 |
|------|------|
| **问题范围** | 仅nvfp4受影响，其他5种格式正常 |
| **代码对比** | 所有nvfp4相关代码路径功能等价 |
| **架构差异** | SGLang提前创建FusedMoEQuantConfig，vLLM延迟创建 |
| **根因推测** | FusedMoEQuantConfig中对weight_scale/global_scale的引用可能因提前创建而指向未加载或被替换的Parameter |
| **建议修复** | 将create_moe_runner改为延迟初始化（方案4），或先运行方案1验证引用有效性 |

**优先级排序**：
1. 先运行方案1（验证引用），确认是否是timing问题
2. 如果引用无效 → 实施方案4（延迟初始化）
3. 如果引用有效 → 运行方案3，逐步对比fused_experts_impl中间值

---

## 25. 方案1调试结果分析（2026-03-04）

### 25.1 调试输出总结

日志文件：`logs_2026_03_04___11_45_22/DeepSeek-V2-Lite-Chat-16B_A2.4B_tp1_quant-simo_w4a4_nvfp.log`

共26个MoE层输出调试信息，**每个层的结论完全一致**：

| 检查项 | 结果 | 含义 |
|--------|------|------|
| `g1_alphas is layer ref` | **全部 True** | config引用和layer参数是同一对象 |
| `g1_alphas data_ptr match` | **全部 True** | 底层数据指针完全一致 |
| `g2_alphas is layer ref` | **全部 True** | w2全局scale引用有效 |
| `g2_alphas data_ptr match` | **全部 True** | 底层数据指针完全一致 |
| `w1_scale is layer ref` | **全部 False** | process_weights_after_loading替换了Parameter对象 |
| `w1_scale data_ptr match` | **全部 True** | 但底层存储相同（.contiguous()是no-op） |

### 25.2 全局scale值分析

**g1_alphas（w13_weight_global_scale）**：每个层所有64个expert值相同（unified），不同层有不同值：

```
Layer 1:  5.6312e-05  (weight_amax ≈ 0.151)
Layer 2:  2.0000e-04  (weight_amax ≈ 0.538)
Layer 3:  6.3942e-05  (weight_amax ≈ 0.172)
Layer 4:  8.1380e-05  (weight_amax ≈ 0.219)
...
```

值在 `5.6e-05` ~ `2.0e-04` 范围内，对应原始权重 amax 在 `0.15` ~ `0.54` 范围内。对于 `global_scale_factor = 6.0 * 448.0 = 2688.0`，这些值合理且正常。

**g2_alphas（w2_weight_global_scale）**：每个expert有独立值（per-expert），第一层示例：
```
expert 0: 1.2788e-04   expert 1: 1.1408e-04   expert 2: 1.2062e-04 ...
```
范围 `8.0e-05` ~ `2.0e-04`，合理。

**g1_alphas 和 layer.w13_weight_global_scale 值完全一致**，g2_alphas 和 layer.w2_weight_global_scale 值也完全一致。

### 25.3 结论：timing假设被排除

**FusedMoEQuantConfig提前创建（timing）假设被彻底排除。** 尽管SGLang在权重加载前就创建了config：

1. 全局scale参数未被`process_weights_after_loading`替换，引用始终有效
2. 权重scale虽然Parameter对象被替换（`is layer ref: False`），但底层存储未变（`data_ptr match: True`），因为`.contiguous()`对已连续tensor是no-op
3. 所有26个MoE层的全部引用均通过验证

### 25.4 MMLU得分变化

| 运行 | 得分 | 与vLLM差距 |
|------|------|-----------|
| 修复前（0303） | 0.4378 | -0.0920 |
| 本次运行（0304） | **0.4630** | **-0.0668** |
| vLLM参考 | 0.5298 | 0 |

得分从0.4378提升到0.4630（+0.025），但未做代码修改（仅添加调试打印）。这可能来自：
- CUDA非确定性（不同GPU状态、调度顺序）
- 或0.4378本身就在随机波动范围内

**无论如何，0.4630仍低于vLLM的0.5298，存在0.067的差距。**

### 25.5 排除总结与下一步分析方向

**已排除的假设：**
- ~~FusedMoEQuantConfig提前创建导致引用无效~~ → 引用全部有效
- ~~全局scale值未正确加载~~ → 值正确且一致
- ~~w1_scale底层数据不一致~~ → data_ptr匹配

**MoE路径已验证正确，问题可能在以下未验证的路径：**

1. **LINEAR层的nvfp4全局scale**：q_proj、kv_a_proj_with_mqa、o_proj、gate_up_proj、down_proj等attention和shared_expert linear层也使用nvfp4全局scale。需要验证：
   - `layer.weight_global_scale`在`process_weights_after_loading`后是否正确（PerTensorScaleParameter → .max() → 标量）
   - 对于merged projection（如gate_up_proj有2个shard），unified_global_scale是否正确应用

2. **权重加载顺序差异**：`unified_w13_global_scale`由第一个加载的w1/w3权重的amax决定。如果SGLang和vLLM的权重迭代顺序不同，可能计算出不同的全局scale，导致不同的量化噪声。
   - 建议：在`online_moe_weight_loader`中添加打印，记录第一次计算unified_w13_global_scale时的expert_id和shard_id
   - 在vLLM侧添加相同打印进行对比

3. **对比vLLM的全局scale值**：在vLLM的`SIMOFusedMoEMethod.apply`中添加相同的调试打印，对比每一层的g1_alphas和g2_alphas数值是否与SGLang一致

4. **fused_experts_impl中间值对比**：在`fused_experts_impl`中添加打印，对比以下中间值：
   ```python
   # 在fused_experts_impl的chunk循环中添加（仅第一次打印）
   if chunk == 0 and not hasattr(fused_experts_impl, '_debug_checked'):
       fused_experts_impl._debug_checked = True
       print(f"[DEBUG-FE] w1 upcast sample: {w1[0, 0, :8]}")
       print(f"[DEBUG-FE] input sample: {curr_hidden_states[0, :8]}")
       print(f"[DEBUG-FE] w1_output_scale: {w1_output_scale}")
       print(f"[DEBUG-FE] cache1 sample: {intermediate_cache1[0, 0, :8]}")
   ```

**建议优先级：**
1. **方案3**（对比vLLM全局scale值）：最快定位差异来源
2. **方案2**（检查权重加载顺序）：如果scale值不同，确认是否因加载顺序导致
3. **方案4**（fused_experts_impl中间值对比）：如果scale值相同，深入对比推理计算

---

## 26. 穷举代码对比分析（2026-03-04 session 3）

### 26.1 分析范围

在MoE timing假设被排除后，对SGLang和vLLM的SIMO nvfp4代码路径进行了**完全穷举对比**，覆盖以下所有代码路径：

| 代码路径 | SGLang文件 | vLLM文件 | 对比结果 |
|---------|-----------|---------|---------|
| get_downcast_kernel | sglang quantization.py:296 | vllm quantization_method.py:85 | ✅ 完全相同 |
| get_upcast_kernel | sglang quantization.py:342 | vllm quantization_method.py:131 | ✅ 完全相同 |
| LINEAR create_weights | sglang quantization.py:849-933 | vllm quantization_method.py:303-384 | ✅ 功能等价 |
| LINEAR process_weights_after_loading | sglang quantization.py:786-810 | vllm quantization_method.py:210-261 | ✅ nvfp4路径等价 |
| LINEAR apply | sglang quantization.py:935-968 | vllm quantization_method.py:386-440 | ✅ 功能等价 |
| LINEAR online_weight_loader | sglang quantization.py:813-847 | vllm quantization_method.py:263-301 | ✅ 完全相同 |
| MoE create_weights | sglang quantization.py:1120-1242 | vllm quantization_method.py:696-851 | ✅ 功能等价 |
| MoE process_weights_after_loading | sglang quantization.py:1244-1288 | vllm quantization_method.py:853-895 | ✅ 完全相同 |
| MoE online_moe_weight_loader | sglang quantization.py:1032-1117 | vllm quantization_method.py:605-694 | ✅ 完全相同 |
| MoE create_moe_runner / get_fused_moe_quant_config | sglang quantization.py:1290-1321 | vllm quantization_method.py:897-921 | ✅ 参数相同 |
| MoE apply → fused_experts_impl | sglang quantization.py:1323-1377 | vllm quantization_method.py:967-1006 | ✅ 同一函数 |
| fused_experts_impl | simo/.../fused_moe.py:15-366 | 同一文件 | N/A |
| dispatch_fused_moe_kernel | simo/.../fused_moe_triton_kernels.py | 同一文件 | N/A |
| FusedMoEQuantConfig.make | simo/.../config.py:264-364 | 同一文件 | N/A |
| QuantizeSpecMX | simo/quantization/config.py:411-474 | 同一文件 | N/A |
| parse_quantize_spec | 两者都调用同一函数 | 同一文件 | N/A |

### 26.2 确认的关键参数一致性

对于nvfp4_e2m1量化：

```
nvfp4 spec解析结果（两个框架完全相同）:
  ├─ 类型: QuantizeSpecMX (不是QuantizeSpecFP)
  ├─ dtype: "nvfp4_e2m1"
  ├─ scale_mode: "e4m3" → ScaleModeEnum.E4M3
  ├─ observer_mode: "abs_max" → ObserverMode.ABS_MAX
  ├─ block_size: 16 (从group_size=16同步)
  ├─ group_size: 16
  └─ axis: -1

weight_granularity: PER_GROUP (QuantizeSpecMX → always PER_GROUP)
input_granularity: PER_GROUP
global_scale_factor: 6.0 * 448.0 = 2688.0

downcast_kernel: downcast_to_mxfmt(..., dtype=nvfp4_e2m1, block_size=16, scale_mode=e4m3)
upcast_kernel: upcast_from_mxfmt(..., dtype=nvfp4_e2m1)

FusedMoEQuantConfig:
  ├─ _a1.dtype = "nvfp4_e2m1" → use_nvfp4_w4a4 = True
  ├─ _w1.dtype = "nvfp4_e2m1"
  ├─ block_shape = [0, 16] (PER_GROUP)
  ├─ _w1.alpha_or_gscale = layer.w13_weight_global_scale
  ├─ _w2.alpha_or_gscale = layer.w2_weight_global_scale
  └─ fast_all2all = False (TP=1, dp_size=1)
```

### 26.3 fast_all2all分析

**SGLang**: `self.fast_all2all = False` (硬编码, quantization.py:1025)
**vLLM**: `self.fast_all2all = (dp_size > 1 and use_ep and flash_comm == "fast_all2all" and ...)` (quantization_method.py:595-600)

在TP=1测试中，vLLM的dp_size=1，所以`fast_all2all = False`。两个框架在fused_experts_impl中走**完全相同的路径**（非fast_all2all路径）。

### 26.4 config差异分析

| 配置项 | SGLang | vLLM | 影响 |
|-------|--------|------|------|
| excludes | `["lm_head", "*visual*", "re:.*kv_b_proj"]` | `["lm_head", "*visual*"]` | SGLang排除kv_b_proj → **应该更有利于SGLang** |
| flash_comm | null | "fast_all2all" | TP=1时无影响 |
| 其他参数 | 相同 | 相同 | 无影响 |

**kv_b_proj排除影响**: SGLang中kv_b_proj使用bf16未量化权重（因为torch.bmm直接使用权重，不经过linear.forward），理论上应该**提高**精度。vLLM中kv_b_proj被nvfp4量化。因此config差异不能解释SGLang精度更低的现象。

### 26.5 SGLang FusedMoE层的_load_per_tensor_weight_scale差异

发现SGLang和vLLM的`_load_per_tensor_weight_scale`有一个细微差异：

**SGLang** (`sglang/.../fused_moe_triton/layer.py:325-344`):
```python
if shard_id in ("w1", "w3"):
    idx = 0 if shard_id == "w1" else 1
    if self.moe_runner_config.is_gated:        # ← 额外的is_gated检查
        param_data[expert_id][idx] = loaded_weight
    else:
        param_data[expert_id] = loaded_weight   # ← 非gated时覆盖整行
```

**vLLM** (`vllm/.../fused_moe/layer.py:907-911`):
```python
if shard_id in ("w1", "w3"):
    idx = 0 if shard_id == "w1" else 1
    param_data[expert_id][idx] = loaded_weight  # ← 总是按索引存储
```

**影响评估**: DeepSeek-V2-Lite使用SiLU gated MoE，`is_gated=True`，所以两者走相同路径。**无影响**。

### 26.6 结论

**穷举对比结果：SIMO量化代码在SGLang和vLLM中对nvfp4路径功能完全等价。**

代码层面找不到差异。0.067的MMLU gap必须从以下方向继续排查：

### 26.7 下一步调试方案

#### 方案A：对比权重加载产生的全局scale数值（最高优先级）

在SGLang和vLLM两侧添加相同的调试打印，直接对比unified_w13_global_scale和unified_w2_global_scale的数值。

**在SGLang的`online_moe_weight_loader`中添加**：
```python
# 在计算unified_w13_global_scale后添加（约line 1060）：
if shard_id in ("w1", "w3", "gate_proj", "up_proj"):
    if not hasattr(layer, "_debug_w13_scale_printed"):
        layer._debug_w13_scale_printed = True
        print(f"[DEBUG-WEIGHT] layer={layer._name if hasattr(layer, '_name') else '?'} "
              f"unified_w13_global_scale={layer.unified_w13_global_scale:.10e} "
              f"from expert_id={expert_id} shard_id={shard_id} "
              f"weight_amax={loaded_weight.abs().to(torch.float32).amax():.10e}")

# 在计算unified_w2_global_scale后添加（约line 1066）：
if shard_id in ("w2", "down_proj"):
    if not hasattr(layer, "_debug_w2_count"):
        layer._debug_w2_count = 0
    if layer._debug_w2_count < 3:
        layer._debug_w2_count += 1
        print(f"[DEBUG-WEIGHT] layer=? expert_id={expert_id} "
              f"unified_w2_global_scale={layer.unified_w2_global_scale:.10e} "
              f"weight_amax={loaded_weight.abs().to(torch.float32).amax():.10e}")
```

在vLLM的`online_moe_weight_loader`中添加相同打印。对比两侧的数值。

#### 方案B：对比LINEAR层全局scale数值

在SGLang和vLLM的`SIMOLinearMethod.apply`中添加一次性打印：
```python
if not hasattr(self, '_debug_linear_checked'):
    self._debug_linear_checked = True
    if hasattr(layer, 'weight_global_scale') and layer.weight_global_scale is not None:
        print(f"[DEBUG-LINEAR] prefix={layer.prefix if hasattr(layer, 'prefix') else '?'} "
              f"weight_global_scale={layer.weight_global_scale} "
              f"shape={layer.weight_global_scale.shape}")
```

#### 方案C：逐层输出对比

在SGLang和vLLM中给模型每一层的输出添加hook，记录每层的hidden_states统计信息（mean, std, abs_max）。然后对比两个框架中每一层的输出是否一致。这可以精确定位是哪一层开始出现分歧。

```python
# 在模型加载后添加
def add_debug_hooks(model):
    for name, module in model.named_modules():
        if hasattr(module, 'forward'):
            original_forward = module.forward
            def make_hook(n, orig_fwd):
                def hooked_forward(*args, **kwargs):
                    output = orig_fwd(*args, **kwargs)
                    if isinstance(output, torch.Tensor) and output.is_floating_point():
                        if not hasattr(module, '_debug_fwd_count'):
                            module._debug_fwd_count = 0
                        if module._debug_fwd_count < 1:
                            module._debug_fwd_count += 1
                            print(f"[LAYER-OUT] {n}: "
                                  f"mean={output.float().mean():.6e} "
                                  f"std={output.float().std():.6e} "
                                  f"absmax={output.float().abs().max():.6e}")
                    return output
                return hooked_forward
            module.forward = make_hook(name, original_forward)
```

#### 方案D：隔离MoE vs LINEAR（终极排查）

创建两个特殊的量化配置：
1. **仅量化MoE**：将excludes设为排除所有attention linear layers
2. **仅量化LINEAR**：不量化FusedMoE层

这可以精确确定精度gap是来自MoE路径还是LINEAR路径。

**建议执行顺序**: A → B → C → D

方案A和B只需添加几行打印代码，对比数值即可快速定位。如果全局scale数值在两个框架间不一致，则问题在权重加载阶段。如果一致，则问题在推理计算阶段，需要用方案C逐层对比。

---

## 27. 调试打印日志分析 — 根因定位 (2026-03-04)

### 日志来源
- SGLang: `temp/logs_sglang_simo_2026_03_04___17_17_46/DeepSeek-V2-Lite-Chat-16B_A2.4B_tp1_quant-simo_w4a4_nvfp.log`
- vLLM: `temp/logs_vllm_simo_2026_03_04___18_27_02/DeepSeek-V2-Lite-Chat-16B_A2.4B_tp1_quant-simo_w4a4_nvfp.log`

### MMLU分数
| 框架 | MMLU | humanities | other | social_sci | stem |
|------|------|-----------|-------|------------|------|
| vLLM | **0.5298** | 0.4714 | 0.6035 | 0.6146 | 0.4618 |
| SGLang | **0.4355** | 0.3983 | 0.4989 | 0.4859 | 0.3793 |
| 差异 | **-0.0943** | -0.0731 | -0.1046 | -0.1287 | -0.0825 |

SGLang精度显著低于vLLM，且本次运行(0.4355)比上次(0.4630)还低，说明问题具有**不确定性**。

### 调试打印数量统计
| 调试标签 | SGLang | vLLM |
|----------|--------|------|
| DEBUG-LINEAR-WEIGHT (Plan A) | 157 | 162 |
| DEBUG-LINEAR-APPLY (Plan B+C) | 6750 | 3140 |
| DEBUG-LINEAR-OUTPUT (Plan C) | 70464 | 28932 |
| DEBUG-MOE-WEIGHT (Plan A) | 210 | 104 |
| DEBUG-MOE-APPLY (Plan C) | 780 | 780 |
| DEBUG-MOE-OUTPUT (Plan C) | 780 | 780 |

### 方案A结果：权重加载时全局scale对比

#### LINEAR权重全局scale
以 `(weight_amax, logical_widths, loaded_shard_id)` 为key匹配：
- **匹配: 133, 不匹配: 0**
- 结论：所有匹配到的LINEAR权重全局scale值完全一致。

#### MoE权重全局scale
以 `weight_amax` 为key匹配：
- **W13 匹配: 24, 不匹配: 0**
- **W2 匹配: 56, 不匹配: 0**
- 结论：对于相同amax的权重，全局scale计算公式 `scale = amax / global_scale_factor` 结果完全一致。

**但这个结论有误导性！** 方案A只证明了 `scale = amax / factor` 的计算是正确的，并没有验证**同一层的同一参数是否使用了相同的scale**。

### 方案B+C结果：推理时全局scale对比 — 发现根因！

#### MoE层 w13_gscale 逐层对比（推理时的实际值）

| 层 | vLLM w13_gscale | SGLang w13_gscale | 比率 | 匹配 |
|----|-----------------|-------------------|------|------|
| 1 | 6.5395e-05 | 5.8855e-05 | 1.111 | NO |
| 2 | **2.0000e-04** | 6.5032e-05 | **3.075** | NO |
| 3 | 7.7384e-05 | 1.0000e-04 | 0.774 | NO |
| 4 | 8.1380e-05 | 8.0654e-05 | 1.009 | NO |
| 5 | 6.6485e-05 | 1.0000e-04 | 0.665 | NO |
| 6 | 8.7920e-05 | 8.7920e-05 | 1.000 | YES |
| 7 | 8.1017e-05 | 8.1017e-05 | 1.000 | YES |
| 8 | 8.4287e-05 | 8.4287e-05 | 1.000 | YES |
| 9 | **2.0000e-04** | 9.8819e-05 | **2.024** | NO |
| 10 | 1.0000e-04 | 1.0000e-04 | 1.000 | YES |
| 11 | 1.0000e-04 | 7.0844e-05 | 1.412 | NO |
| 12 | 7.4477e-05 | 8.1380e-05 | 0.915 | NO |
| 13 | 1.0000e-04 | 1.0000e-04 | 1.000 | YES |
| 14 | 1.0000e-04 | 8.5377e-05 | 1.171 | NO |
| 15 | 8.7556e-05 | 9.1916e-05 | 0.953 | NO |
| 16 | 1.0000e-04 | 9.8092e-05 | 1.019 | NO |
| 17 | 7.9564e-05 | 6.2125e-05 | 1.281 | NO |
| 18 | 8.0290e-05 | 9.4459e-05 | 0.850 | NO |
| 19 | 5.8855e-05 | 5.8855e-05 | 1.000 | YES |
| 20 | 8.7193e-05 | 6.2488e-05 | 1.395 | NO |
| 21 | **2.0000e-04** | 5.3769e-05 | **3.720** | NO |
| 22 | 6.3578e-05 | 8.1380e-05 | 0.781 | NO |
| 23 | 1.0000e-04 | 6.2125e-05 | 1.610 | NO |
| 24 | 1.0000e-04 | 7.1208e-05 | 1.404 | NO |
| 25 | 1.0000e-04 | 1.0000e-04 | 1.000 | YES |
| 26 | 6.0672e-05 | 6.8301e-05 | 0.888 | NO |

**w13 匹配: 7/26 (27%)**
**w2 匹配: 1/26 (4%)**

#### LINEAR层 gate_up_proj 全局scale对比

可通过层名匹配的gate_up_proj层：
- 匹配: 18/27
- 不匹配: 9/27 (layers 3, 9, 12, 16, 17, 18, 22, 25, 26)

### 根因分析

#### 直接原因

`unified_w13_global_scale` 是从**第一个被加载的expert的权重amax**计算得来的。SGLang和vLLM加载safetensor分片的顺序不同，导致不同的expert被用作"参考expert"：

| 框架 | 首个加载的expert | 首个shard |
|------|-----------------|-----------|
| vLLM | 几乎总是 expert 0 | 总是 w1 |
| SGLang | 随机 (13, 11, 18, 15, 16...) | w1 或 w3 随机 |

#### 关键代码路径

```python
# simo/extensions/{sglang_simo,vllm_simo}/quantization/quantization{,_method}.py
# online_moe_weight_loader 中:
if shard_id in ("w1", "w3", "gate_proj", "up_proj"):
    if hasattr(layer, "unified_w13_global_scale"):
        unified_global_scale = layer.unified_w13_global_scale  # 复用已有scale
    else:
        weight_amax = loaded_weight.abs().to(torch.float32).amax()
        unified_global_scale = weight_amax / self.global_scale_factor
        layer.unified_w13_global_scale = unified_global_scale  # 首次设置
```

**问题**: `unified_w13_global_scale` 取决于第一个加载的expert的amax。不同expert的amax差异巨大（例如layer 21: expert 0 amax=0.447 → scale=1.66e-4, 而SGLang的首个expert amax=0.145 → scale=5.38e-5, 相差3.7倍）。

#### 为什么只影响nvfp4？

其他量化格式不使用两级(global+local)缩放机制：
- **FP8, MXFP4/6/8**: 使用MX格式的e8m0 scale mode，不需要global scale → 不受影响
- **INT4 per_group**: 使用local per-group scale，不需要global scale → 不受影响
- **nvfp4_e2m1**: 唯一使用 `global_scale_factor = 6.0 * 448.0 = 2688.0` 两级缩放的格式 → **受影响**

#### 为什么SGLang MMLU分数有波动？

SGLang每次运行加载expert的顺序可能不同（取决于safetensor文件的读取顺序），因此每次运行的 `unified_w13_global_scale` 都可能不同：
- Run 1: 0.4378
- Run 2: 0.4630
- Run 3: 0.4355

而vLLM几乎总是先加载expert 0 (确定性加载顺序)，因此分数稳定在0.5298。

### 修复建议

**核心修复**: `unified_w13_global_scale` 应该使用所有expert中权重amax的**最大值**，而不是第一个加载的expert的amax。

```python
# 修复方案: 两阶段加载
# 阶段1: 预扫描所有expert的amax
# 阶段2: 使用max(amax)计算unified_global_scale

# 但这需要两次遍历权重文件，不够实用。

# 更实用的方案: 在process_weights_after_loading中重新计算scale
# 1. 在weight loading阶段，记录每个expert分别使用的global_scale
# 2. 在process_weights_after_loading中：
#    - 找到所有expert中amax的最大值
#    - 重新计算统一的global_scale
#    - 对每个expert的量化权重进行scale调整（rescale）
```

**最简修复**: 让SGLang和vLLM使用相同的expert加载顺序（确保expert 0总是第一个被加载）。但这是治标不治本的方案，因为第一个expert的amax可能不是最优的参考值。

**最佳修复**: 使用所有experts的权重amax最大值作为unified_global_scale。可以在 `process_weights_after_loading` 中实现：
1. 在加载阶段，每个expert用自己的amax/factor作为临时scale
2. 在 `process_weights_after_loading` 中，计算所有expert中最大的amax
3. 用这个最大amax重新计算统一scale
4. 对每个expert的已量化权重进行rescale补偿

---

## 28. Safetensor遍历顺序分析 — SGLang vs vLLM (2026-03-04)

### 问题

SGLang调用 `SIMOFusedMoEMethod.online_moe_weight_loader` 时，是按什么顺序遍历 safetensor 里面的 tensor，然后把 tensor 填充到 layer 的 parameter 里面的？决定遍历顺序的代码在哪里？vLLM 同理。

### SGLang 调用链

```
DefaultModelLoader.load_model()                                [loader.py:653]
  → self._get_all_weights(model_config, model)                 [获取 weights_iterator]
    → self._get_weights_iterator()                             [loader.py:501-521]
      → safetensors_weights_iterator()                         [weight_utils.py:713-736]
  → self.load_weights_and_postprocess(model, weights, ...)     [loader.py:685]
    → model.load_weights(weights)                              [loader.py:686]
      → DeepseekV2ForCausalLM.do_load_weights()                [deepseek_weight_loader.py:96-256]
        → 对每个 (name, loaded_weight) 匹配 expert_params_mapping
          → FusedMoE.make_expert_params_mapping()              [layer.py:1023-1048]
        → param.weight_loader(param, loaded_weight, shard_id, expert_id)
          → SIMOFusedMoEMethod.online_moe_weight_loader()      [quantization.py]
```

#### 1. 选择迭代器: `loader.py:501-521`

```python
# loader.py line 501-521
elif use_safetensors:
    weight_loader_disable_mmap = (
        get_global_server_args().weight_loader_disable_mmap
    )
    if self.load_config.load_format == LoadFormat.FASTSAFETENSORS:
        weights_iterator = fastsafetensors_weights_iterator(hf_weights_files)
    elif use_multithread:
        weights_iterator = buffered_multi_thread_safetensors_weights_iterator(
            hf_weights_files, ...)
    else:
        weights_iterator = safetensors_weights_iterator(
            hf_weights_files, disable_mmap=weight_loader_disable_mmap)
```

默认走最后一个分支: `safetensors_weights_iterator()`。

#### 2. 迭代器实现: `weight_utils.py:713-736` (**决定遍历顺序的关键代码**)

```python
# weight_utils.py line 713-736
def safetensors_weights_iterator(
    hf_weights_files: List[str],
    disable_mmap: bool = False,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    for st_file in tqdm(hf_weights_files, ...):
        if disable_mmap:
            with open(st_file, "rb") as f:
                result = safetensors.torch.load(f.read())
                for name in sorted(result.keys()):    # ← 字母序
                    yield name, result[name]
        else:
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
                for name in f.keys():                 # ← 文件内部顺序
                    yield name, f.get_tensor(name)
```

**关键结论:**
- **mmap模式 (默认)**: `f.keys()` — 返回 safetensor 文件**内部存储顺序**，不是字母序。这个顺序由模型保存时的序列化顺序决定，**不保证 expert_id 有序**。
- **disable_mmap模式**: `sorted(result.keys())` — 返回**字母序**。

#### 3. 多线程迭代器: `weight_utils.py:830-885` (备选路径)

```python
# weight_utils.py line 882
for name in sorted(state_dict.keys()):    # ← 始终字母序
    yield name, state_dict[name]
```

多线程迭代器始终使用 `sorted()` (字母序)。

#### 4. 权重匹配: `deepseek_weight_loader.py:230-256`

```python
# deepseek_weight_loader.py line 230-256
for mapping in expert_params_mapping:
    param_name, weight_name, expert_id, shard_id = mapping
    if weight_name not in name:
        continue
    name = name.replace(weight_name, param_name)
    param = params_dict[name]
    weight_loader = param.weight_loader
    weight_loader(param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id)
    break
```

`expert_params_mapping` 由 `FusedMoE.make_expert_params_mapping()` (layer.py:1023-1048) 生成，遍历顺序是 `expert_id in range(num_experts)` × `(w1, w2, w3)`。但这只是**匹配映射**的顺序，实际加载顺序取决于 `weights_iterator` yield 的顺序。

#### SGLang 结论

**默认配置下 (mmap=True, 单线程)**: tensor 的遍历顺序 = safetensor 文件内部存储顺序 (`f.keys()`)。这个顺序**不是字母序，不保证 expert_id 有序**。实际观察到的顺序是 expert 13 → 11 → 18 → ... 这样的非确定性顺序。

---

### vLLM 调用链

```
DefaultModelLoader.load_weights()
  → DefaultModelLoader._get_weights_iterator()                [default_loader.py:184-229]
    → safetensors_weights_iterator()                          [weight_utils.py:674-732]
  → model.load_weights(weights_iterator)
    → DeepseekV2ForCausalLM.load_weights()                    [deepseek_v2.py]
      → 对每个 (name, loaded_weight) 匹配 expert_params_mapping
        → FusedMoE.make_expert_params_mapping()
      → param.weight_loader(param, loaded_weight, shard_id, expert_id)
        → SIMOFusedMoEMethod.online_moe_weight_loader()       [quantization_method.py]
```

#### 1. 选择迭代器: `default_loader.py:209-229`

```python
# default_loader.py line 209-229
elif use_safetensors:
    if self.load_config.load_format == "fastsafetensors":
        weights_iterator = fastsafetensors_weights_iterator(hf_weights_files, ...)
    else:
        if extra_config.get("enable_multithread_load"):
            weights_iterator = multi_thread_safetensors_weights_iterator(...)
        else:
            weights_iterator = safetensors_weights_iterator(
                hf_weights_files,
                self.load_config.use_tqdm_on_load,
                self.load_config.safetensors_load_strategy,    # ← 默认 "lazy"
            )
```

默认走: `safetensors_weights_iterator()`, strategy = `"lazy"`。

#### 2. 迭代器实现: `weight_utils.py:674-732` (**决定遍历顺序的关键代码**)

```python
# weight_utils.py line 674-732
def safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    safetensors_load_strategy: str = "lazy",
) -> Generator[tuple[str, torch.Tensor], None, None]:
    for st_file in tqdm(hf_weights_files, ...):
        if safetensors_load_strategy == "eager":
            with open(st_file, "rb") as f:
                state_dict = load(f.read())
            yield from state_dict.items()             # ← dict插入序 (Python 3.7+)
        elif safetensors_load_strategy == "torchao":
            ...
        else:  # "lazy" (默认)
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():                 # ← 文件内部顺序
                    param = f.get_tensor(name)
                    yield name, param
```

**关键结论:**
- **lazy模式 (默认)**: `f.keys()` — 返回 safetensor 文件**内部存储顺序**，与 SGLang 的 mmap 模式一样。
- **eager模式**: `state_dict.items()` — dict 插入顺序 (Python 3.7+ 保证有序，顺序取决于 `safetensors.torch.load` 的实现)。

#### vLLM 结论

**默认配置下 (lazy策略, 单线程)**: tensor 的遍历顺序 = safetensor 文件内部存储顺序 (`f.keys()`)。这与 SGLang 的默认路径**完全一致**。

---

### 核心对比

| 特性 | SGLang (默认) | vLLM (默认) |
|------|--------------|-------------|
| 迭代器 | `safetensors_weights_iterator()` | `safetensors_weights_iterator()` |
| 关键代码 | `weight_utils.py:734` `f.keys()` | `weight_utils.py:730` `f.keys()` |
| 遍历顺序 | safetensor 文件内部顺序 | safetensor 文件内部顺序 |
| 多线程路径 | `sorted()` 字母序 | 无 `sorted()` |
| disable_mmap | `sorted()` 字母序 | N/A |

**理论上**两者默认路径的遍历顺序应该相同 (都用 `f.keys()`)。但实际 debug log 显示 SGLang 先加载 expert 13, 11, 18... 而 vLLM 先加载 expert 0。可能的原因：

1. **safetensor 文件不同**: SGLang 和 vLLM 可能读取了不同的 `.safetensors` shard 文件 (模型路径不同，或者文件分片方式不同)
2. **`hf_weights_files` 排序不同**: 两个框架获取文件列表的排序方式可能不同，如果 expert 分布在不同 shard 文件中
3. **`f.keys()` 行为差异**: `safetensors` 库版本不同可能导致 `f.keys()` 返回顺序不同
4. **TP 分片逻辑差异**: 如果使用了 tensor parallelism，两个框架可能过滤了不同的 shard 文件

**推荐验证**: 在两个框架中分别打印 `hf_weights_files` 列表和每个文件的 `f.keys()` 前几个 key，确认实际遍历顺序。
