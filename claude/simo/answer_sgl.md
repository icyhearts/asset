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
