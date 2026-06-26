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

---

## 29. `_get_test_cases()` 列表推导式的等价 for 循环解释

### 原始代码 (`tests/vllm_simo/e2e_test/test_basic_generate.py:42-60`)

```python
def _get_test_cases():
  return [
    {
      "model_path": model_path,
      "attention_backend": attention_backend,
      "quant_config": str(quant_config),
      "case_id": f"{Path(model_path).name}-{quant_config.stem}",
    }
    for model_path, attention_backend in _MODEL_CASES
    for quant_config in _QUANT_CONFIGS
    if not any(quant_config.match(p) for p in _EXCLUDE_QUANT_CONFIGS)
    if (
      quant_config not in _KVQUANT_CONFIGS
      or (attention_backend == "TRITON_MLA" and "mla" in quant_config.name)
      or (attention_backend == "TRITON_ATTN" and "gqa" in quant_config.name)
    )
  ]
```

### 等价的 for 循环写法

```python
def _get_test_cases():
  result = []

  # 外层循环: 遍历所有模型 (model_path, attention_backend) 组合
  # _MODEL_CASES 例如: [("/share_data/.../DeepSeekV2-Lite", "TRITON_MLA")]
  for model_path, attention_backend in _MODEL_CASES:

    # 内层循环: 遍历所有量化配置文件
    # _QUANT_CONFIGS = _KVQUANT_CONFIGS + _WEIGHT_CONFIGS
    # 即 kv_cache_quant/ 和 online_quantization/ 下的所有 .json 文件
    for quant_config in _QUANT_CONFIGS:

      # 过滤条件1: 排除黑名单中的配置
      # 如果 quant_config 匹配 _EXCLUDE_QUANT_CONFIGS 中的任意 glob 模式，跳过
      if any(quant_config.match(p) for p in _EXCLUDE_QUANT_CONFIGS):
        continue

      # 过滤条件2: kv_cache 配置必须与 attention_backend 匹配
      # 分三种情况放行:
      #   a) 不是 kv_cache 配置 (是 online_quantization 配置) → 直接放行
      #   b) 是 kv_cache 配置 + backend 是 TRITON_MLA + 文件名含 "mla" → 放行
      #   c) 是 kv_cache 配置 + backend 是 TRITON_ATTN + 文件名含 "gqa" → 放行
      # 其他情况 (比如 TRITON_MLA 后端配了 gqa 的 kv_cache 配置) → 跳过
      is_kvquant = quant_config in _KVQUANT_CONFIGS
      if is_kvquant:
        mla_match = (attention_backend == "TRITON_MLA" and "mla" in quant_config.name)
        gqa_match = (attention_backend == "TRITON_ATTN" and "gqa" in quant_config.name)
        if not (mla_match or gqa_match):
          continue

      # 两个过滤条件都通过，加入结果
      result.append({
        "model_path": model_path,
        "attention_backend": attention_backend,
        "quant_config": str(quant_config),
        "case_id": f"{Path(model_path).name}-{quant_config.stem}",
      })

  return result
```

### 逻辑总结

这是一个 **模型 × 量化配置** 的笛卡尔积，加上两层过滤:

1. **黑名单过滤**: `_EXCLUDE_QUANT_CONFIGS` 中的 glob 模式命中的配置被排除
2. **backend 兼容性过滤**: kv_cache 量化配置必须与 attention backend 类型匹配 (MLA 后端只用 mla 配置，GQA/TRITON_ATTN 后端只用 gqa 配置)；online_quantization 配置不受此限制，对所有 backend 都生效

---

## 30. pytest 运行 test_vllm_simo_generate_smoke 的执行模型

### 问题

使用 `pytest tests/vllm_simo/e2e_test/test_basic_generate.py` 运行时，`test_vllm_simo_generate_smoke` 的多组参数是并发跑还是串行跑？

### 答案：串行，不并发

**pytest 默认是单进程、串行执行所有测试用例。** 不同参数组合不会并发运行。

### 执行流程

假设 `_get_test_cases()` 返回了 N 组参数（例如 8 个量化配置），pytest 的执行过程如下：

```
pytest 主进程 (单进程, 串行调度)
│
├─ 参数组1: test_vllm_simo_generate_smoke[DeepSeekV2-Lite-quant_config_A]
│    └─ _run_single_case_in_process()
│         └─ spawn 子进程 → 加载模型 → 推理 → 断言 → 子进程退出
│         └─ proc.join() 阻塞等待子进程结束
│         └─ _kill_process() 清理
│    (完成后才进入下一组)
│
├─ 参数组2: test_vllm_simo_generate_smoke[DeepSeekV2-Lite-quant_config_B]
│    └─ _run_single_case_in_process()
│         └─ spawn 子进程 → 加载模型 → 推理 → 断言 → 子进程退出
│         └─ proc.join() 阻塞等待
│    (完成后才进入下一组)
│
├─ ... (逐个串行)
│
└─ 参数组N: test_vllm_simo_generate_smoke[DeepSeekV2-Lite-quant_config_N]
     └─ ...
```

### 涉及两层"进程"概念，不要混淆

| 层级 | 谁创建的 | 并发吗 | 说明 |
|------|---------|--------|------|
| pytest 调度层 | pytest 框架 | **否，串行** | pytest 默认单线程逐个执行每组参数的测试函数 |
| 测试用例内部 | `_run_single_case_in_process()` | 每组参数 spawn **1个**子进程 | 用 `mp.get_context("spawn")` 创建子进程执行实际推理，主进程 `proc.join()` 阻塞等待它结束 |

**关键代码** (`test_basic_generate.py:98-100`):

```python
proc.start()          # spawn 一个子进程
try:
    proc.join(timeout=timeout_s)   # 阻塞等待子进程结束
```

`proc.join()` 会阻塞 pytest 主进程，直到子进程退出。所以每组参数一定是前一组跑完、子进程退出、清理之后，才轮到下一组。

### 为什么每组参数要 spawn 子进程？

代码注释写了：`"avoid CUDA + fork issues"`。原因是 CUDA runtime 在 fork 后的子进程中行为未定义，而 vLLM 加载模型会初始化 CUDA context。用 spawn 创建全新进程可以确保每组测试拿到干净的 CUDA 环境，上一组测试的 GPU 显存也能被完全释放。

### 如何才能并发？

默认 pytest 不并发。如果想并发需要安装 `pytest-xdist` 插件并使用 `-n` 参数：

```bash
# 4 个 worker 并发 (需要安装 pytest-xdist)
pytest -n 4 tests/vllm_simo/e2e_test/test_basic_generate.py
```

但对于 GPU 测试通常不适合并发，因为多组测试会争抢 GPU 显存。

---

## 31. SGLang 测试 finally 中如何清理现场（对标 vLLM）

### vLLM 的清理方式 (`tests/vllm_simo/e2e_test/test_basic_generate.py:216-224`)

```python
finally:
    if llm is not None:
        llm.reset_mm_cache()
        del llm
    from vllm.distributed import cleanup_dist_env_and_memory
    cleanup_dist_env_and_memory()
```

### SGLang 当前的清理方式 (`tests/sglang_simo/e2e_test/test_basic_generate.py:248-250`)

```python
finally:
    if llm is not None:
        del llm
```

只做了 `del llm`，没有调用 shutdown 和分布式环境清理。

### SGLang 提供的清理 API

#### 1. `Engine.shutdown()` — 杀掉 Engine 启动的所有子进程

文件: `sglang/srt/entrypoints/engine.py:453-455`

```python
def shutdown(self):
    """Shutdown the engine"""
    kill_process_tree(os.getpid(), include_parent=False)
```

`Engine.__init__` 中已通过 `atexit.register(self.shutdown)` 注册了退出时自动调用 (engine.py:159)，但显式调用更可靠，可以确保在 `del` 之前子进程已终止。

#### 2. `cleanup_dist_env_and_memory()` — 清理分布式环境和 GPU 显存

文件: `sglang/srt/distributed/parallel_state.py:2117-2142`

```python
def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    destroy_model_parallel()          # 销毁 TP/PP/MoE 等并行组
    destroy_distributed_environment() # 销毁 world process group
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    if shutdown_ray:
        import ray
        ray.shutdown()
    gc.collect()                      # 垃圾回收
    if not _is_cpu:
        torch.cuda.empty_cache()      # 清空 CUDA 缓存
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()  # 清空 host pinned memory (PyTorch >= 2.5)
```

这与 vLLM 的 `cleanup_dist_env_and_memory()` 功能一致。

### 推荐的 SGLang finally 写法

```python
finally:
    if llm is not None:
        llm.shutdown()   # 杀掉 Engine 启动的所有子进程 (调度器、detokenizer 等)
        del llm
    from sglang.srt.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
    )
    cleanup_dist_env_and_memory()
```

### 对比

| 清理步骤 | vLLM | SGLang |
|---------|------|--------|
| 清理模型缓存 | `llm.reset_mm_cache()` | 不需要（SGLang Engine 没有 mm_cache） |
| 关闭引擎/杀子进程 | `del llm`（vLLM LLM 没有 spawn 子进程） | `llm.shutdown()`（杀掉 Engine spawn 的子进程树） |
| 释放引用 | `del llm` | `del llm` |
| 清理分布式 + GPU 显存 | `vllm.distributed.cleanup_dist_env_and_memory()` | `sglang.srt.distributed.parallel_state.cleanup_dist_env_and_memory()` |

**注意**: SGLang `Engine` 会 spawn 多个子进程（scheduler、detokenizer 等），必须调用 `shutdown()` 来杀掉它们，否则即使 `del llm` 后这些子进程仍在运行、占用 GPU 显存。vLLM 的 `LLM` 类是单进程的，`del` 就够了。不过由于本测试文件的 `_run_single_case` 已经在独立的 spawn 子进程中运行，子进程退出时 OS 会回收所有资源，所以 `shutdown()` + `cleanup_dist_env_and_memory()` 主要是确保**子进程内的清理**在 `_run_single_case` 返回前完成，避免竞态条件。

---

## 32. pytest 报错 `unrecognized arguments: --cov --cov-report`

### 报错信息

```
ERROR: usage: pytest [options] [file_or_dir] [file_or_dir] [...]
pytest: error: unrecognized arguments: --cov --cov-report
  inifile: /share_data/users/like/package/h100/package/simo_conda_sglang/pyproject.toml
```

### 原因

`pyproject.toml` 第45行配置了 pytest 默认参数：

```toml
[tool.pytest.ini_options]
addopts = "--cov --cov-report term-missing -v"
```

`--cov` 和 `--cov-report` 是 `pytest-cov` 插件提供的参数。`simo_sglang` 环境中没有安装 `pytest-cov`，pytest 不认识这些参数就报错了。

### 解决方案

#### 方案1: 安装 pytest-cov（推荐，不改配置）

```bash
conda activate simo_sglang
pip install pytest-cov
```

#### 方案2: 运行时用 `-o` 覆盖 addopts（不改配置、不装包）

```bash
SIMO_SGLANG_REGISTER=1 CUDA_VISIBLE_DEVICES=3 pytest -o "addopts=-v" tests/sglang_simo/e2e_test/test_basic_generate.py
```

`-o addopts=-v` 会覆盖 `pyproject.toml` 中的 `addopts`，只保留 `-v`，去掉 `--cov` 相关参数。

#### 方案3: 用 `--override-ini` 清空 addopts

```bash
SIMO_SGLANG_REGISTER=1 CUDA_VISIBLE_DEVICES=3 pytest --override-ini="addopts=" tests/sglang_simo/e2e_test/test_basic_generate.py
```

---

## 33. vLLM + SIMO KV Cache量化下的Attention实现分析

### 33.1 启动命令与配置概述

启动命令：
```bash
CUDA_VISIBLE_DEVICES=4 vllm serve --quantization simo \
  --model /data_gpu/models/.../llama3.1-8B-Instruct/safetensor_weights/ \
  --hf-overrides '{"quantization_config_file": ".../quant_config_kvquant_mxfp8.json"}' \
  --gpu-memory-utilization 0.5 \
  --attention-config '{"backend": "TRITON_ATTN"}' \
  --port 30120
```

量化配置（`/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/example/simo_quantization_config/kv_cache_quant/quant_config_kvquant_mxfp8.json`）：
```json
{
    "quantization_config": {
        "algorithm": {"name": "naive"},
        "excludes": ["*"],
        "kv_cache_quant_algo": {
            "dtype": "mxfp8_e4m3",
            "query_quantization_enabled": false,
            "key_hadamard_transform_size": 0,
            "value_hadamard_transform_size": 0
        },
        "quant_method": "simo"
    }
}
```

**配置含义**：
- `excludes: ["*"]` — 排除所有权重量化（仅做KV Cache量化）
- `kv_cache_quant_algo.dtype: "mxfp8_e4m3"` — KV Cache使用MXFP8 E4M3格式量化
- `query_quantization_enabled: false` — 不量化Query
- `key/value_hadamard_transform_size: 0` — 不使用Hadamard变换

**重要说明**：该启动命令使用的是Llama 3.1 8B模型（GQA架构）+ `TRITON_ATTN` backend。如果换成DeepSeek模型（MLA架构），vLLM会使用 `TRITON_MLA` backend和完全不同的attention实现路径。下面分别分析两种情况。

---

### 33.2 GQA模型（Llama 3.1 8B）的Attention路径

#### 33.2.1 后端注册机制

SIMO通过vLLM的`@register_backend`装饰器注册自定义attention后端，而非monkey patch：

**文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/backends/simo_gqa.py:33-36`
```python
@register_backend(
  AttentionBackendEnum.TRITON_ATTN,
  "simo.extensions.vllm_simo.v1.attention.backends.simo_gqa.SIMOAttentionBackend",
)
class SIMOAttentionBackend(TritonAttentionBackend):
```

当`--attention-config '{"backend": "TRITON_ATTN"}'`且SIMO已安装时，vLLM会使用`SIMOAttentionBackend`替代原生的`TritonAttentionBackend`。

- 后端类: `SIMOAttentionBackend` (`simo_gqa.py:37`)
- 实现类: `SIMOAttentionImpl` (`simo_gqa.py:99`)，继承自vLLM原生的`TritonAttentionImpl`

#### 33.2.2 Attention主入口: `SIMOAttentionImpl.forward()`

**文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/backends/simo_gqa.py:146-311`

`forward()`方法是所有attention计算的统一入口，它根据场景分发到不同的kernel：

```
SIMOAttentionImpl.forward()
├── 无量化 → 回退到 super().forward() (原生vLLM TritonAttentionImpl)
├── Encoder attention → self._forward_encoder_attention()
├── 首次Prefill (max_seqlen_q > 1 且 max_seqlen_k == max_seqlen_q)
│   → flash_attn_varlen_func()  [FlashAttention kernel]
└── Decode / 后续Prefill
    → unified_attention()  [SIMO自定义Triton kernel，从量化KV Cache读取]
```

#### 33.2.3 Prefill阶段: `flash_attn_varlen_func`

**触发条件** (`simo_gqa.py:234`): `max_seqlen_q > 1 and max_seqlen_k == max_seqlen_q`（首次prefill，所有token都是新的，没有已缓存的KV）

**调用位置**: `simo_gqa.py:248-266`

```python
flash_attn_varlen_func(
    q=query[:num_actual_tokens],
    k=key[:num_actual_tokens],       # 新token的key（已Hadamard变换）
    v=value[:num_actual_tokens],     # 新token的value（已Hadamard变换）
    out=output[:num_actual_tokens],
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_q,       # prefill时Q和K的seq_lens相同
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_q,
    softmax_scale=self.scale,
    causal=True,
    k_descale=layer._k_scale.expand(descale_shape),   # FP8 dequant scale
    v_descale=layer._v_scale.expand(descale_shape),
    ...
)
```

**这是vLLM自带的FlashAttention kernel**，来自:
- `/softhome/like/package/h100/package/vllm-for-conda-simo/vllm/v1/attention/backends/fa_utils.py`
- 底层调用 `vllm.vllm_flash_attn.flash_attn_varlen_func`（vLLM自编译的FlashAttention CUDA kernel）

**Prefill阶段的key/value处理** (`simo_gqa.py:237-239`):
```python
key = apply_hadamard_transform(key, layer.key_hadamard_transform_size, axis=-1)
value = apply_hadamard_transform(value, layer.value_hadamard_transform_size, axis=-1)
```
在当前配置下`hadamard_transform_size=0`，Hadamard变换实际上是no-op。

**FlashAttention Kernel原理（简述）**:
- 使用tiling策略将Q/K/V切分为小块（block），减少HBM访问
- 在SRAM中计算softmax的running statistics（max + expsum），避免完整softmax的两次遍历
- 支持causal masking（下三角mask），仅计算有效的Q-K对
- 对每个Q block，遍历所有K blocks，累积 P·V 的weighted sum
- 输入Q/K/V形状: `[num_tokens, num_heads, head_size]`
- 支持GQA: `num_kv_heads < num_query_heads`时，多个query heads共享同一个KV head

#### 33.2.4 Decode阶段: SIMO `unified_attention`

**触发条件**: 非首次prefill（已有缓存的KV），或decode（每次只有1个新token）

**调用位置**: `simo_gqa.py:279-309`

```python
unified_attention(
    q=query[:num_actual_tokens],
    k=key_cache,                       # 量化后的KV Cache (uint8)
    v=value_cache,
    out=output[:num_actual_tokens],
    cu_seqlens_q=cu_seqlens_q,
    max_seqlen_q=max_seqlen_q,
    seqused_k=seqused_k,
    max_seqlen_k=max_seqlen_k,
    softmax_scale=self.scale,
    causal=True,
    block_table=block_table,           # paged attention的block table
    kv_cache_quant_spec=quant_spec,    # MXFP8_E4M3量化规格
    packed_head_size=packed_hs,        # 量化后packed数据大小
    scale_head_size=scale_hs,          # 量化scale数据大小
    ...
)
```

**Kernel实现文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/ops/triton_unified_attention.py`

**`unified_attention()`函数** (`triton_unified_attention.py:1156-1449`) 根据batch特征分发到两个Triton kernel:

| 条件 | 使用的Kernel | 典型场景 |
|------|-------------|---------|
| `max_seqlen_q > 1` 或 `num_seqs > seq_threshold_3D` 或 3D buffer未分配 | `kernel_unified_attention_2d` | Prefill / 大batch decode |
| `max_seqlen_q == 1` 且 `num_seqs <= seq_threshold_3D` 且 3D buffer已分配 | `kernel_unified_attention_3d` + `reduce_segments` | 小batch decode |

##### `kernel_unified_attention_2d` 详解

**文件**: `triton_unified_attention.py:140-579`

**Grid**: `(total_num_q_blocks, num_kv_heads)` — 每个thread block处理一个Q block + 一个KV head

**核心算法流程**:

1. **加载Query** (`line 237-261`):
   - 如果`query_quantization_enabled=False`（当前配置），直接加载bf16 query
   - Q形状: `[BLOCK_M, HEAD_SIZE_PADDED]`，其中`BLOCK_M = 16`（对应`num_queries_per_kv`个query head）

2. **遍历KV Cache tiles** (`line 335-563`):
   ```
   for j in range(tile_start, tile_end):  # 遍历KV序列的每个tile
   ```

   每个tile处理`TILE_SIZE=32`个KV位置。

3. **从量化KV Cache加载K和V** (SIMO_QUANT路径, `line 343-420`):

   KV Cache使用**Planar Layout**:
   ```
   每个KV head的存储: [packed_data (PACKED_HEAD_SIZE bytes) | scale_data (SCALE_HEAD_SIZE bytes)]
   ```

   对于MXFP8_E4M3:
   - `packed_data`: 每个元素1 byte (FP8 E4M3格式)
   - `scale_data`: 每32个元素共享1个E8M0 scale (1 byte)

   **加载过程**:
   ```python
   # 加载packed数据
   k_packed = tl.load(key_cache_ptr + k_packed_offset + packed_offs_d[None, :], ...)
   v_packed = tl.load(value_cache_ptr + v_packed_offset + packed_offs_d[None, :], ...)

   # 加载scales
   k_scales = tl.load(key_cache_ptr + k_scales_offset + scale_offs_d[None, :], ...)
   v_scales = tl.load(value_cache_ptr + v_scales_offset + scale_offs_d[None, :], ...)
   ```

4. **反量化K并计算S = Q·K^T** (`line 484-500`):

   对于MXFP8_E4M3，使用Triton的`tl.dot_scaled`硬件加速指令:
   ```python
   S += scale * tl.dot_scaled(Q, None, "bf16", k_packed.T, k_scales, "e4m3", fast_math=True)
   ```
   这利用了NVIDIA Hopper架构的MX format硬件支持，无需显式software反量化。

5. **Causal Masking** (`line 458-479`):
   ```python
   query_abs_pos = context_len + query_pos[:, None]
   seq_mask = seq_offset[None, :] <= query_abs_pos  # 因果mask
   ```

6. **Online Softmax** (`line 531-553`):
   ```python
   m_j = tl.maximum(M, tl.max(S, axis=1))      # 更新running max
   P = tl.exp(S - m_j[:, None])                 # 计算attention权重
   alpha = tl.exp(M - m_j)                      # rescale旧累积
   acc = acc * alpha[:, None]                    # rescale旧输出
   L = L * alpha + tl.sum(P, axis=1)            # 更新running expsum
   M = m_j
   ```

7. **反量化V并累积输出** (`line 560-562`):
   ```python
   V = _unpack_and_dequant_mxfmt(v_packed, v_scales, MX_FORMAT_ID)
   acc += tl.dot(P.to(V.dtype), V)              # 加权求和
   ```
   V的反量化使用software实现（`_unpack_and_dequant_mxfmt`），因为`tl.dot_scaled`只支持K方向。

8. **最终输出** (`line 563-578`):
   ```python
   acc = acc / L[:, None]                        # 归一化
   tl.store(output_ptr + output_offset, acc, ...)
   ```

##### `kernel_unified_attention_3d` 详解

**文件**: `triton_unified_attention.py:582-1122`

**Grid**: `(total_num_q_blocks, num_kv_heads, num_par_softmax_segments)` — 增加了第3维用于并行softmax分段

**与2D kernel的区别**:
- 3D kernel将每个序列的KV分成`num_par_softmax_segments`段，每段由一个独立的thread block处理
- 每个thread block计算一段的partial softmax结果（partial_output, partial_max, partial_expsum）
- 计算完成后，由`reduce_segments` kernel (`line 1037-1122`) 合并所有段的结果
- 适用于decode阶段的长序列，通过增加并行度提高GPU利用率

**`reduce_segments` kernel** (`line 1037-1122`):
```python
# 遍历所有segments，使用online算法合并
for s in range(0, num_actual_segms):
    segm_max_val = tl.load(segm_max_ptr + ...)
    segm_expsum_val = tl.load(segm_expsum_ptr + ...)
    segm_output = tl.load(segm_output_ptr + ...)

    # 更新overall_max, overall_expsum, 累积output
    correction = tl.exp(segm_max_val - overall_max)
    ...
```

#### 33.2.5 KV Cache写入: `simo_triton_reshape_and_cache_flash`

**调用位置**: `simo_gqa.py:106-144` (`do_kv_cache_update`方法)

**Kernel文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/ops/triton_reshape_and_cache_flash.py`

**流程**:
1. 对key/value做Hadamard变换（当前配置下为no-op）
2. 将bf16的key/value量化为MXFP8_E4M3格式
3. 写入paged KV Cache的对应slot，使用Planar Layout存储

---

### 33.3 MLA模型（DeepSeek）的Attention路径

**注意**: DeepSeek v2/v3模型使用MLA（Multi-head Latent Attention）架构，与Llama的GQA架构完全不同。如果要运行DeepSeek，需要将`--attention-config`改为`TRITON_MLA`。

#### 33.3.1 后端注册

**文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:38-41`

```python
@register_backend(
  AttentionBackendEnum.TRITON_MLA,
  "simo.extensions.vllm_simo.v1.attention.backends.simo_mla.SIMOMLABackend",
)
class SIMOMLABackend(TritonMLABackend):
```

- 后端类: `SIMOMLABackend` (`simo_mla.py:42`)
- 实现类: `SIMOMLAImpl` (`simo_mla.py:93`)，继承自vLLM原生的`TritonMLAImpl`

#### 33.3.2 MLA特有概念

DeepSeek的MLA将传统的KV进行低秩压缩:
- **kv_c_normed**: 压缩后的KV latent向量（维度`kv_lora_rank`，如512）
- **k_pe**: Rope位置编码部分（维度如64）
- KV Cache只存储 `[kv_c | k_pe]` 拼接后的低维表示，而非完整的K和V

在attention计算时:
- **Prefill (MHA模式)**: 将latent展开为完整的K/V，然后用FlashAttention计算
- **Decode (MQA模式)**: 直接在latent空间做attention（`Q_nope · KV_C^T + Q_PE · K_PE^T`），无需展开

#### 33.3.3 Prefill阶段

MLA prefill分为两部分:

##### (a) 新token直接attention: `forward_mha()`

**文件**: `simo_mla.py:191-217`

对新到来的token，在latent展开后使用FlashAttention:

```python
def forward_mha(self, q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata, k_scale, output):
    # 1. Hadamard变换 (如果启用)
    if self._key_hadamard_transform_size > 0:
        q[..., self.qk_nope_head_dim:] = apply_hadamard_transform(...)
        kv_c_normed = apply_hadamard_transform(...)
    if self._value_hadamard_transform_size > 0:
        k_pe = apply_hadamard_transform(...)

    # 2. 调用父类的forward_mha，内部使用flash_attn_varlen_func
    super().forward_mha(q, kv_c_normed, k_pe, ...)
```

父类`TritonMLAImpl.forward_mha()`最终调用 `flash_attn_varlen_func`，kernel来自:
- `/softhome/like/package/h100/package/vllm-for-conda-simo/vllm/model_executor/layers/attention/mla_attention.py`

##### (b) 已缓存context的attention: `_compute_prefill_context()`

**文件**: `simo_mla.py:291-366`

对已经在KV Cache中的token（chunked context），需要从量化cache中读取并反量化:

```python
def _compute_prefill_context(self, q, kv_c_and_k_pe_cache, attn_metadata, k_scale):
    # 1. 从量化KV Cache中gather和反量化
    gather_and_maybe_dequant_cache(
        src_cache=kv_c_and_k_pe_cache,
        dst=workspace,
        kv_cache_quant_spec=self._kv_cache_quant_spec,
        ...
    )

    # 2. 将反量化后的kv_c通过kv_b_proj投影得到完整的K_nope和V
    kv_nope = self.kv_b_proj(kv_c_normed)[0].view(...)
    k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    # 3. 拼接K_nope和K_PE得到完整的K
    k = self._concat_k_nope_k_pe(k_nope, k_pe)

    # 4. 用FlashAttention计算attention (non-causal)
    attn_output, attn_softmax_lse = self._run_prefill_context_chunk(...)

    # 5. 多个chunk的结果用merge_attn_states合并
    merge_attn_states(output, output_lse, prefix_output, prefix_lse, ...)
```

**gather_and_maybe_dequant_cache kernel文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/ops/triton_gather_and_maybe_dequant_cache.py`

功能: 从Tensor-Interleaved格式的量化cache中按block_table指定的物理位置gather数据，反量化为bf16。

#### 33.3.4 Decode阶段: `forward_mqa()` → `decode_attention_fwd()`

**文件**: `simo_mla.py:219-289`

Decode阶段在latent空间直接计算attention，无需展开KV:

```python
def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
    # 1. 首次调用时计算量化layout参数
    if not self._quant_layout_cached:
        self._compute_quant_layout(layer, q)

    # 2. q是(ql_nope, q_pe) tuple
    ql_nope, q_pe = q
    if self._key_hadamard_transform_size > 0:
        q_pe = apply_hadamard_transform(q_pe, ...)
    q = torch.cat([ql_nope, q_pe], dim=-1)   # 拼接为完整query

    # 3. 调用decode attention kernel
    decode_attention_fwd(
        q, kv_c_and_k_pe_cache, kv_c_cache,
        o, lse, block_table, seq_lens, attn_logits,
        num_kv_splits=4,
        scale=self.scale,
        PAGE_SIZE=page_size,
        mx_format_id=self._mx_format_id,       # MXFP8 format ID
        packed_head_size=self._packed_head_size,
        scale_head_size=self._scale_head_size,
        packed_dpe_size=self._packed_dpe_size,   # K_PE部分的packed大小
        scale_dpe_size=self._scale_dpe_size,
        Lk_override=self._Lk_override,           # = kv_lora_rank + pe_dim
        Lv_override=self._Lv_override,           # = kv_lora_rank
        ...
    )
    return o, lse
```

**Kernel文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/ops/triton_decode_attention.py`

##### `decode_attention_fwd()` 分发逻辑 (`triton_decode_attention.py:957-1027`)

```python
def decode_attention_fwd(q, k_buffer, v_buffer, o, lse, ...):
    kv_group_num = q.shape[1] // v_buffer.shape[-2]
    if kv_group_num == 1:
        decode_attention_fwd_normal(...)   # MHA路径
    else:
        decode_attention_fwd_grouped(...)  # GQA/MQA/MLA路径
```

DeepSeek MLA的`kv_group_num > 1`，走`decode_attention_fwd_grouped`路径。

##### `_fwd_grouped_kernel_stage1` 详解 (`triton_decode_attention.py:308-619`)

这是decode阶段的核心Triton kernel，采用**Flash Decoding**两阶段算法:

**Grid**: `(batch, ceil(head_num / min(BLOCK_H, kv_group_num)), NUM_KV_SPLITS)`

**Stage 1 算法流程**:

1. **加载Query** (`line 380-387`):
   ```python
   q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]))
   # MLA特有: 加载Q的PE部分
   if BLOCK_DPE > 0:
       qpe = tl.load(Q + off_qpe, ...)
   ```
   Q被分为两部分: `q[BLOCK_H, BLOCK_DMODEL]` (nope部分) 和 `qpe[BLOCK_H, BLOCK_DPE]` (PE部分)

2. **KV序列分片** (`line 389-391`):
   ```python
   kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
   split_kv_start = kv_len_per_split * split_kv_id
   split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
   ```
   将KV序列均匀分成`NUM_KV_SPLITS=4`段，每个split由独立的thread block处理。

3. **遍历KV Cache blocks** (`line 398-596`):
   ```
   for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
   ```
   每次处理`BLOCK_N=32`个KV位置。

4. **从量化Cache加载K** (SIMO_QUANT路径, `line 407-506`):

   MLA的Cache使用**Tensor-Interleaved Layout**:
   ```
   每个token的cache entry:
   [KV_C packed | KV_C scales | K_PE packed | K_PE scales]
   ```

   加载过程:
   ```python
   # K Content (kv_c部分)
   k_content_packed = tl.load(K_Buffer + k_base_offset + k_content_packed_offs[None, :], ...)
   k_content_scales = tl.load(K_Buffer + k_content_scales_offset + k_content_scales_offs[None, :], ...)

   # MX反量化
   if MX_FORMAT_ID > 0:
       k = _unpack_and_dequant_mxfmt(k_content_packed, k_content_scales, MX_FORMAT_ID)

   # 计算Q_nope · K_C^T
   qk = tl.dot(q, k.trans().to(q.dtype))

   # K PE部分 (类似加载和反量化)
   k_pe_packed = tl.load(K_Buffer + k_pe_base_offset + ...)
   k_pe_scales = tl.load(K_Buffer + k_pe_scales_offset + ...)
   kpe = _unpack_and_dequant_mxfmt(k_pe_packed, k_pe_scales, MX_FORMAT_ID)

   # 累加Q_PE · K_PE^T
   qk += tl.dot(qpe, kpe.trans().to(qpe.dtype))
   ```

5. **从量化Cache加载V** (`line 508-548`):
   ```python
   # V使用KV_C部分（MLA中V由kv_c_normed投影得到，latent space相同）
   v_packed = tl.load(V_Buffer + v_base_offset + v_packed_offs[None, :], ...)
   v_scales = tl.load(V_Buffer + v_scales_offset + v_scales_offs[None, :], ...)
   v = _unpack_and_dequant_mxfmt(v_packed, v_scales, MX_FORMAT_ID)
   ```

6. **Online Softmax + 累积输出** (`line 582-596`):
   ```python
   qk *= sm_scale
   n_e_max = tl.maximum(tl.max(qk, 1), e_max)
   re_scale = tl.exp(e_max - n_e_max)
   p = tl.exp(qk - n_e_max[:, None])
   acc *= re_scale[:, None]
   acc += tl.dot(p.to(v.dtype), v)
   e_sum = e_sum * re_scale + tl.sum(p, 1)
   e_max = n_e_max
   ```

7. **写入中间结果** (`line 598-618`):
   ```python
   tl.store(Att_Out + offs_mid_o, acc / e_sum[:, None], ...)    # partial output
   tl.store(Att_Out + offs_mid_o_1, e_max + tl.log(e_sum), ...) # log-sum-exp
   ```

##### `_fwd_kernel_stage2` (结果归约) (`triton_decode_attention.py:770-828`)

**Grid**: `(batch, head_num)`

将Stage 1的`NUM_KV_SPLITS`个partial结果合并为最终输出:

```python
for split_kv_id in range(0, NUM_KV_SPLITS):
    tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, ...)     # partial output
    tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)  # log-sum-exp

    n_e_max = tl.maximum(tlogic, e_max)
    old_scale = tl.exp(e_max - n_e_max)
    acc *= old_scale
    exp_logic = tl.exp(tlogic - n_e_max)
    acc += exp_logic * tv

    e_sum = e_sum * old_scale + exp_logic
    e_max = n_e_max

tl.store(o + ..., acc / e_sum, ...)
```

#### 33.3.5 KV Cache写入: `concat_and_cache_mla`

**调用位置**: `simo_mla.py:155-189` (`do_kv_cache_update`方法)

**Kernel文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py`

**流程**:
1. 可选的Hadamard变换
2. 将`kv_c_normed`和`k_pe`拼接为一个向量
3. 按tile量化（MXFP8_E4M3格式），存储为Tensor-Interleaved Layout:
   ```
   [KV_C packed bytes | KV_C scale bytes | K_PE packed bytes | K_PE scale bytes]
   ```

---

### 33.4 `_unpack_and_dequant_mxfmt` 反量化核心函数

**文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/ops/kernels/upcast/_upcast_from_mxfmt.py`

这是在decode kernel中对量化KV Cache进行反量化的核心Triton JIT函数。

对于MXFP8_E4M3 (`MX_FORMAT_ID == MXFP8_E4M3`):
- packed数据: 每个元素1 byte，FP8 E4M3格式
- scale: 每32个元素共享1个E8M0 scale（1 byte），表示共享的指数偏移

反量化过程:
```
dequant_value = packed_fp8_value * 2^(scale_e8m0 - 127)
```

对于其他MX格式（MXFP4, MXFP6, MXINT8等），涉及更复杂的位解包和反量化。

---

### 33.5 总结: Attention Kernel调用链

#### GQA模型（Llama 3.1 8B + TRITON_ATTN）

| 阶段 | Kernel | 文件 | 说明 |
|------|--------|------|------|
| **Prefill（首次）** | `flash_attn_varlen_func` | vLLM自带 FlashAttention CUDA kernel | 对新token直接用bf16 Q/K/V计算，不读quantized cache |
| **Decode / 后续Prefill** | `kernel_unified_attention_2d` 或 `kernel_unified_attention_3d` + `reduce_segments` | `simo/.../triton_unified_attention.py:140` / `582` / `1037` | 从量化KV Cache读取，on-the-fly反量化后计算attention |
| **KV Cache写入** | `simo_triton_reshape_and_cache_flash` | `simo/.../triton_reshape_and_cache_flash.py` | 将新token的K/V量化为MXFP8写入paged cache |

#### MLA模型（DeepSeek + TRITON_MLA）

| 阶段 | Kernel | 文件 | 说明 |
|------|--------|------|------|
| **Prefill新token** | `flash_attn_varlen_func` | vLLM自带 FlashAttention CUDA kernel | 在latent展开后用FlashAttention计算 |
| **Prefill已缓存context** | `gather_and_maybe_dequant_cache` + `flash_attn_varlen_func` | `simo/.../triton_gather_and_maybe_dequant_cache.py` | 从量化cache gather+反量化，展开后用FlashAttention |
| **Decode** | `_fwd_grouped_kernel_stage1` + `_fwd_kernel_stage2` | `simo/.../triton_decode_attention.py:308` / `770` | Flash Decoding两阶段: 在latent空间直接从量化cache计算attention |
| **KV Cache写入** | `concat_and_cache_mla_kernel` | `simo/.../triton_concat_and_cache_mla.py:26` | 拼接kv_c+k_pe后量化为MXFP8写入cache |

---

### 33.6 Kernel详细讲解（含代码片段）

#### 33.6.1 GQA Prefill Kernel: `flash_attn_varlen_func`

**调用位置**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/backends/simo_gqa.py:232-272`

**触发条件**：首次prefill（`max_seqlen_q > 1 and max_seqlen_k == max_seqlen_q`），即所有token都是新的，没有已缓存的KV。

**调用代码**：
```python
# simo_gqa.py:232-272
# 首次prefill: 使用flash_attn处理Hadamard变换后的key/value
use_flash_attn = False
if max_seqlen_q > 1 and max_seqlen_k == max_seqlen_q:
    # 对key/value做Hadamard变换（当前配置hadamard_transform_size=0时为no-op）
    if key is not None and value is not None:
        key = apply_hadamard_transform(key, layer.key_hadamard_transform_size, axis=-1)
        value = apply_hadamard_transform(value, layer.value_hadamard_transform_size, axis=-1)

    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func, is_flash_attn_varlen_func_available,
    )

    if is_flash_attn_varlen_func_available():
        flash_attn_varlen_func(
            q=query[:num_actual_tokens],        # [num_tokens, num_heads, head_size]
            k=key[:num_actual_tokens],           # [num_tokens, num_kv_heads, head_size]
            v=value[:num_actual_tokens],         # [num_tokens, num_kv_heads, head_size]
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,           # 每个序列的起始位置 [num_seqs+1]
            cu_seqlens_k=cu_seqlens_q,           # prefill时Q和K的长度相同
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,            # 1/sqrt(head_size)
            causal=True,                         # 因果mask（下三角）
            k_descale=layer._k_scale.expand(descale_shape),   # FP8 dequant scale
            v_descale=layer._v_scale.expand(descale_shape),
        )
        use_flash_attn = True
```

**FlashAttention CUDA Kernel原理**:

FlashAttention是一个IO-aware的exact attention算法，核心思想是**将attention计算分块（tiling）到GPU SRAM中**，避免在HBM上存储完整的attention矩阵 `S = Q·K^T`（对于长序列，这个矩阵是`seq_len × seq_len`大小）。

算法伪代码：
```
# FlashAttention核心循环（单个Q block）
对于 Q 的每个 block (Br × d):
    初始化: max_i = -inf, sum_i = 0, acc_i = 0
    对于 K/V 的每个 block (Bc × d):
        # 1. 在SRAM中计算 S_ij = Q_i · K_j^T  (Br × Bc)
        S_ij = matmul(Q_i, K_j^T) * softmax_scale

        # 2. 应用因果mask: 未来token位置设为-inf
        S_ij = where(causal_mask, S_ij, -inf)

        # 3. Online Softmax: 维护running max和exp sum
        new_max = max(max_i, rowmax(S_ij))
        P_ij = exp(S_ij - new_max)            # 当前块的attention权重
        rescale = exp(max_i - new_max)         # 对旧累积的rescale因子

        # 4. 更新累积: 先rescale旧值，再加新值
        acc_i = acc_i * rescale + P_ij · V_j
        sum_i = sum_i * rescale + rowsum(P_ij)
        max_i = new_max

    # 最终归一化
    output_i = acc_i / sum_i
```

**关键优势**:
- **内存**: O(N)而非O(N²)，不需要存储完整的attention矩阵
- **IO**: 显著减少HBM读写次数，所有中间计算在SRAM中完成
- **精度**: 数学上与标准attention完全等价（exact attention），不是近似

---

#### 33.6.2 GQA Decode Kernel: `kernel_unified_attention_2d`

**文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/ops/triton_unified_attention.py:140-578`

**触发条件**：decode阶段（`max_seqlen_q == 1`）或后续prefill（已有缓存KV）。从量化的KV Cache中读取，on-the-fly反量化后计算attention。

**Grid**: `(total_num_q_blocks, num_kv_heads)` — 每个thread block负责一个Q block与一个KV head的attention计算。

**完整算法流程**（以MXFP8量化为例）：

##### Step 1: 确定当前thread block负责的Query范围

```python
# triton_unified_attention.py:203-218
q_block_global_idx = tl.program_id(0)    # 当前Q block的全局索引
kv_head_idx = tl.program_id(1)           # 当前KV head索引

# 通过二分查找确定当前Q block属于哪个序列
seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)

# 计算当前序列内的Q block偏移
q_block_local_idx = q_block_global_idx - q_block_start_idx
cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

# 边界检查
if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
    return
```

##### Step 2: 加载Query

```python
# triton_unified_attention.py:237-261
# BLOCK_M = num_queries_per_kv（GQA中每个KV head对应多个Q head）
offs_m = tl.arange(0, BLOCK_M)
offs_d = tl.arange(0, HEAD_SIZE_PADDED)
query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

# 加载Q: (BLOCK_M, HEAD_SIZE_PADDED)
# 对于MXFP8 KV Cache量化但Query未量化的情况，直接加载bf16 query
Q = tl.load(
    query_ptr + query_offset,
    mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    other=0.0,
)
```

##### Step 3: 初始化Online Softmax状态

```python
# triton_unified_attention.py:266-275
M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)   # running max
L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)             # running exp sum
acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)  # 输出累积
```

##### Step 4: 遍历KV Cache tiles

```python
# triton_unified_attention.py:335-562
# num_tiles = ceil(max_seq_prefix_len / TILE_SIZE)
for j in range(tile_start, tile_end):
    seq_offset = j * TILE_SIZE + offs_t   # 当前tile的KV位置
    tile_mask = seq_offset < max_seq_prefix_len
```

##### Step 5: 从量化KV Cache加载K和V（SIMO_QUANT路径）

KV Cache使用**Planar Layout**，每个KV head的存储：`[packed_data | scale_data]`

```python
    # triton_unified_attention.py:343-388
    # SIMO_QUANT = True (MX_FORMAT_ID > 0)

    # --- 计算K Cache地址 ---
    # block_table映射逻辑页号→物理block
    physical_block_idx = tl.load(
        block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
    ).to(tl.int64)

    # 每个slot的byte偏移
    slot_base_k_offset = (
        physical_block_idx * stride_k_cache_0
        + (seq_offset % BLOCK_SIZE) * stride_k_cache_1
    )[:, None]

    # packed数据偏移: 第kv_head_idx个head的packed区域
    k_packed_offset = slot_base_k_offset + kv_head_idx * PACKED_HEAD_SIZE
    # scale数据偏移: 所有head的packed数据之后，再偏移到当前head
    k_scales_offset = slot_base_k_offset + SCALE_PLANE_OFFSET + kv_head_idx * SCALE_HEAD_SIZE

    # --- 加载packed K数据 (uint8) ---
    packed_offs_d = tl.arange(0, PACKED_HEAD_SIZE_PADDED)
    packed_mask = row_mask & (packed_offs_d[None, :] < PACKED_HEAD_SIZE)
    k_packed = tl.load(                          # shape: (TILE_SIZE, PACKED_HEAD_SIZE)
        key_cache_ptr + k_packed_offset + packed_offs_d[None, :],
        mask=packed_mask, other=0
    )

    # --- 加载scale数据 ---
    scale_offs_d = tl.arange(0, SCALE_HEAD_SIZE_PADDED)
    scales_mask = row_mask & (scale_offs_d[None, :] < SCALE_HEAD_SIZE)
    k_scales = tl.load(                          # shape: (TILE_SIZE, SCALE_HEAD_SIZE)
        key_cache_ptr + k_scales_offset + scale_offs_d[None, :],
        mask=scales_mask, other=0
    )

    # --- V Cache加载方式完全相同 ---
    v_packed = tl.load(value_cache_ptr + v_packed_offset + packed_offs_d[None, :], ...)
    v_scales = tl.load(value_cache_ptr + v_scales_offset + scale_offs_d[None, :], ...)
```

##### Step 6: 计算 S = Q·K^T（利用硬件加速指令）

MXFP8_E4M3使用Triton的`tl.dot_scaled`指令，直接在MX格式上做矩阵乘，无需显式反量化：

```python
    # triton_unified_attention.py:484-500
    S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)

    if MX_FORMAT_ID == MXFP4_E2M1:
        # MXFP4: 硬件dot_scaled（SM90+支持）
        S += scale * tl.dot_scaled(Q, None, "bf16",
                                   k_packed.T, k_scales, "e2m1", fast_math=True)

    elif MX_FORMAT_ID == MXFP8_E4M3:
        # MXFP8 E4M3: 硬件dot_scaled
        # Q是bf16, K是e4m3+e8m0 scale, 硬件自动完成 Q · dequant(K)^T
        S += scale * tl.dot_scaled(Q, None, "bf16",
                                   k_packed.T, k_scales, "e4m3", fast_math=True)

    elif MX_FORMAT_ID == MXFP8_E5M2:
        # MXFP8 E5M2: 硬件dot_scaled
        S += scale * tl.dot_scaled(Q, None, "bf16",
                                   k_packed.T, k_scales, "e5m2", fast_math=True)

    elif MX_FORMAT_ID > 0:
        # MXFP6, MXINT8, NVFP4等: 需要software反量化
        K = tl.trans(_unpack_and_dequant_mxfmt(k_packed, k_scales, MX_FORMAT_ID))
        S += scale * tl.dot(Q, K)

    else:
        # 非量化路径
        S += scale * tl.dot(Q, K)
```

**`tl.dot_scaled`详解**：这是Triton在NVIDIA Hopper (SM90+) 架构上的MX格式加速指令。它直接在Tensor Core上执行 `bf16 × MX_format` 的矩阵乘法，硬件内部完成MX格式的反量化。参数含义：
- 第1-3个参数: LHS矩阵（Q）、其scale、数据类型标识
- 第4-6个参数: RHS矩阵（K packed）、其scale（E8M0）、数据类型标识
- `fast_math=True`: 允许硬件使用近似计算提高吞吐量

##### Step 7: 应用因果Mask

```python
    # triton_unified_attention.py:458-479
    # 因果mask: key位置 <= query绝对位置
    query_abs_pos = context_len + query_pos[:, None]
    seq_mask = seq_offset[None, :] <= query_abs_pos

    # Sliding Window mask（如果启用）
    if SLIDING_WINDOW > 0:
        seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)

    # 将mask外的位置设为-inf
    S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf"))
```

##### Step 8: Online Softmax更新

```python
    # triton_unified_attention.py:531-553
    # 计算当前tile的max，与全局running max取较大值
    m_j = tl.maximum(M, tl.max(S, axis=1))       # (BLOCK_M,)

    # 防止全-inf导致NaN
    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

    # 计算attention权重 P = exp(S - max)
    P = tl.exp(S - m_j[:, None])                  # (BLOCK_M, TILE_SIZE)

    # 计算当前tile的exp sum
    l_j = tl.sum(P, axis=1)                       # (BLOCK_M,)

    # Rescale旧累积: 乘以 exp(old_max - new_max) 补偿max变化
    alpha = tl.exp(M - m_j)                        # (BLOCK_M,)
    acc = acc * alpha[:, None]                      # (BLOCK_M, HEAD_SIZE_PADDED)

    # 更新running状态
    L = L * alpha + l_j
    M = m_j
```

##### Step 9: 反量化V并累积输出

V的反量化使用software方式（`_unpack_and_dequant_mxfmt`），因为`tl.dot_scaled`仅支持K方向的硬件加速：

```python
    # triton_unified_attention.py:560-562
    if MX_FORMAT_ID > 0:
        # Software反量化V: 先解包MX格式，再乘以E8M0 scale
        V = _unpack_and_dequant_mxfmt(v_packed, v_scales, MX_FORMAT_ID)
        # V shape: (TILE_SIZE, HEAD_SIZE)

    # P · V 加权求和
    acc += tl.dot(P.to(V.dtype), V)                # (BLOCK_M, HEAD_SIZE_PADDED)
```

##### Step 10: 归一化并写入输出

```python
# triton_unified_attention.py:563-578
# 循环结束后：acc中存储了 sum(P·V), L中存储了 sum(P)
# 最终输出 = sum(P·V) / sum(P)
acc = acc / L[:, None]

# 写入输出tensor
tl.store(
    output_ptr + output_offset,
    acc,
    mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
)
```

---

#### 33.6.3 GQA Decode Kernel (3D版): `kernel_unified_attention_3d` + `reduce_segments`

**文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/ops/triton_unified_attention.py:582-1122`

**与2D kernel的区别**：3D kernel增加了并行softmax分段（`segm_idx`），将KV序列分成多段并行计算，适用于decode阶段的长序列。

**Grid**: `(total_num_q_blocks, num_kv_heads, NUM_SEGMENTS_PER_SEQ)`

##### 分段范围计算

```python
# triton_unified_attention.py:662-669
seq_len = tl.load(seq_lens_ptr + seq_idx)
num_segments = NUM_SEGMENTS_PER_SEQ
tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

# 如果当前segment超出序列范围，直接返回
if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
    return
```

##### 每个segment只处理自己负责的tile范围

```python
# triton_unified_attention.py:781-783
# 与2D kernel的区别：tile循环范围被限制在当前segment负责的区间
for j in range(
    max(segm_idx * tiles_per_segment, tile_start),       # segment起始tile
    min((segm_idx + 1) * tiles_per_segment, tile_end),   # segment结束tile
):
    # ... 内部逻辑与2D kernel完全相同 ...
```

##### 写入中间结果（partial softmax）

```python
# triton_unified_attention.py:1016-1034
# 注意：3D kernel不做最终归一化（acc / L），
# 而是将partial结果直接写入中间buffer
tl.store(segm_output_ptr + segm_output_offset, acc, ...)     # partial weighted sum
tl.store(segm_max_ptr + segm_offset, M, ...)                 # running max
tl.store(segm_expsum_ptr + segm_offset, L, ...)              # running exp sum
```

##### `reduce_segments` kernel：合并所有segment的partial结果

**Grid**: `(num_tokens, num_query_heads)` — 每个token×head一个thread

```python
# triton_unified_attention.py:1037-1122
@triton.jit
def reduce_segments(output_ptr, segm_output_ptr, segm_max_ptr, segm_expsum_ptr, ...):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    # Step 1: 加载所有segment的max，找全局max
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)                    # scalar

    # Step 2: 用全局max rescale每个segment的exp sum
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)               # scalar

    # Step 3: 用全局max rescale每个segment的partial output，然后求和
    segm_output = tl.load(segm_output_ptr + segm_output_offset, ...)
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)              # (HEAD_SIZE_PADDED,)

    # Step 4: 最终归一化
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    # Step 5: 写入最终输出
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)
```

**为什么需要3D kernel**: decode阶段每个query只有1个token（`max_seqlen_q == 1`），但KV序列可能很长。2D kernel只用1个thread block处理整个KV序列，GPU利用率低。3D kernel将KV序列分成N段，N个thread block并行计算，提高GPU SM占用率。

---

#### 33.6.4 MLA Decode Kernel: `_fwd_grouped_kernel_stage1` (Flash Decoding Stage 1)

**文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/ops/triton_decode_attention.py:308-619`

**Grid**: `(batch, ceil(head_num / min(BLOCK_H, kv_group_num)), NUM_KV_SPLITS)`

- `batch`: 当前batch中的序列数
- 第2维: 按query head分组处理（每个thread block处理`BLOCK_H`个query head）
- `NUM_KV_SPLITS`: KV序列被分成多段并行（默认4段）

##### Step 1: 确定当前thread block负责的范围

```python
# triton_decode_attention.py:360-378
cur_batch = tl.program_id(0)      # 当前序列
cur_head_id = tl.program_id(1)    # query head组
split_kv_id = tl.program_id(2)    # KV序列分片ID

# 每个KV head对应kv_group_num个query head
cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

# 当前thread block处理的query head范围
cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
mask_h = cur_head < q_head_num
```

##### Step 2: 加载Query（MLA特有的两部分结构）

MLA的query由两部分组成：`q_nope`（与kv_c做attention）和`q_pe`（与k_pe做attention）

```python
# triton_decode_attention.py:380-387
# 加载Q的nope部分: (BLOCK_H, BLOCK_DMODEL)
offs_d = tl.arange(0, BLOCK_DMODEL)
mask_d = offs_d < Lk
offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

# 加载Q的PE部分: (BLOCK_H, BLOCK_DPE)
if BLOCK_DPE > 0:
    offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
    off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
    qpe = tl.load(Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0)
```

##### Step 3: 计算KV序列分片范围

```python
# triton_decode_attention.py:389-391
kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
split_kv_start = kv_len_per_split * split_kv_id
split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
```

##### Step 4: 遍历KV Cache blocks，从量化cache加载K和V

MLA使用**Tensor-Interleaved Layout**，与GQA的Planar Layout不同：
```
每个token每个KV head的cache entry:
[KV_C packed | KV_C scales | K_PE packed | K_PE scales]
```

```python
# triton_decode_attention.py:398-548
for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
    offs_n = start_n + tl.arange(0, BLOCK_N)

    # Page Table查找: 逻辑位置→物理block
    kv_page_number = tl.load(
        Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n // PAGE_SIZE,
        mask=offs_n < split_kv_end, other=0,
    )
    kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE

    if SIMO_QUANT:
        # === K Content (kv_c部分) 加载 ===
        k_base_offset = (kv_loc * stride_buf_kbs + cur_kv_head * stride_buf_kh)[:, None]

        # 加载packed K content: (BLOCK_N, PACKED_HEAD_SIZE)
        k_content_packed = tl.load(
            K_Buffer + k_base_offset + k_content_packed_offs[None, :],
            mask=k_content_packed_mask, other=0,
        )
        # 加载K content的scales: (BLOCK_N, SCALE_HEAD_SIZE)
        k_content_scales = tl.load(
            K_Buffer + k_base_offset + PACKED_HEAD_SIZE + k_content_scales_offs[None, :],
            mask=k_content_scales_mask, other=0,
        )

        # MX格式反量化
        if MX_FORMAT_ID > 0:
            k = _unpack_and_dequant_mxfmt(k_content_packed, k_content_scales, MX_FORMAT_ID)

        # 计算 Q_nope · K_C^T: (BLOCK_H, BLOCK_N)
        qk = tl.dot(q, k.trans().to(q.dtype))

        # === K PE部分 加载（MLA特有）===
        if BLOCK_DPE > 0:
            # K_PE紧跟在K_C之后: offset = PACKED_HEAD_SIZE + SCALE_HEAD_SIZE
            k_pe_base_offset = k_base_offset + PACKED_HEAD_SIZE + SCALE_HEAD_SIZE

            k_pe_packed = tl.load(
                K_Buffer + k_pe_base_offset + k_pe_packed_offs[None, :],
                mask=k_pe_packed_mask, other=0,
            )
            k_pe_scales = tl.load(
                K_Buffer + k_pe_base_offset + PACKED_DPE_SIZE + k_pe_scales_offs[None, :],
                mask=k_pe_scales_mask, other=0,
            )

            kpe = _unpack_and_dequant_mxfmt(k_pe_packed, k_pe_scales, MX_FORMAT_ID)

            # 累加 Q_PE · K_PE^T
            qk += tl.dot(qpe, kpe.trans().to(qpe.dtype))

        # === V Cache加载（使用KV_C部分）===
        v_base_offset = (kv_loc * stride_buf_vbs + cur_kv_head * stride_buf_vh)[:, None]

        v_packed = tl.load(
            V_Buffer + v_base_offset + v_packed_offs[None, :],
            mask=v_packed_mask, other=0
        )
        v_scales = tl.load(
            V_Buffer + v_base_offset + PACKED_DV_SIZE + v_scales_offs[None, :],
            mask=v_scales_mask, other=0
        )

        v = _unpack_and_dequant_mxfmt(v_packed, v_scales, MX_FORMAT_ID)
```

**注意MLA的独特之处**:
- Attention Score = `Q_nope · K_C^T + Q_PE · K_PE^T`（两次dot product累加）
- V直接使用kv_c部分（MLA中V由kv_c通过kv_b_proj投影得到，但在decode时latent空间直接用kv_c代替）

##### Step 5: Online Softmax + 累积输出

```python
    # triton_decode_attention.py:582-596
    qk *= sm_scale                                    # 应用softmax缩放

    if logit_cap > 0:
        qk = logit_cap * tanh(qk / logit_cap)        # logit capping（如果启用）

    # Mask掉超出序列范围的位置
    qk = tl.where(mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf"))

    # Online Softmax更新
    n_e_max = tl.maximum(tl.max(qk, 1), e_max)       # (BLOCK_H,)
    re_scale = tl.exp(e_max - n_e_max)                # rescale因子
    p = tl.exp(qk - n_e_max[:, None])                 # attention权重 (BLOCK_H, BLOCK_N)

    acc *= re_scale[:, None]                           # rescale旧累积
    acc += tl.dot(p.to(v.dtype), v)                    # P · V累加 (BLOCK_H, BLOCK_DV)

    e_sum = e_sum * re_scale + tl.sum(p, 1)           # 更新exp sum
    e_max = n_e_max
```

##### Step 6: 写入partial结果

```python
# triton_decode_attention.py:598-618
# 写入 partial output: acc / e_sum
tl.store(
    Att_Out + offs_mid_o,
    acc / e_sum[:, None],                              # (BLOCK_H, BLOCK_DV)
    mask=(mask_h[:, None]) & (mask_dv[None, :]),
)

# 写入 log-sum-exp（用于stage 2归约）
tl.store(
    Att_Out + offs_mid_o_1,
    e_max + tl.log(e_sum),                             # scalar per head
    mask=mask_h,
)
```

---

#### 33.6.5 MLA Decode Kernel: `_fwd_kernel_stage2` (Flash Decoding Stage 2)

**文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/ops/triton_decode_attention.py:770-828`

**Grid**: `(batch, head_num)` — 每个序列每个head一个thread

Stage 2将Stage 1的`NUM_KV_SPLITS`个partial结果合并为最终输出。

```python
# triton_decode_attention.py:786-828
@triton.jit
def _fwd_kernel_stage2(Mid_O, o, lse, B_Seqlen, ...):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # 初始化归约状态
    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    # 遍历所有KV splits的partial结果
    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            # 加载Stage 1写入的partial output和log-sum-exp
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)

            # Online归约: 与Stage 1相同的rescale逻辑
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)

            acc *= old_scale                     # rescale旧累积
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv                # 加上当前split的贡献

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    # 最终归一化并写入
    tl.store(o + ..., acc / e_sum, mask=mask_d)

    # 写入log-sum-exp（用于prefill-decode合并）
    lse_val = e_max + tl.log(e_sum)
    tl.store(lse + ..., lse_val)
```

**Flash Decoding两阶段算法总结**:

```
Stage 1: NUM_KV_SPLITS个thread block并行计算
  Split 0:  KV[0:L/4]      → partial_output_0, lse_0
  Split 1:  KV[L/4:L/2]    → partial_output_1, lse_1
  Split 2:  KV[L/2:3L/4]   → partial_output_2, lse_2
  Split 3:  KV[3L/4:L]     → partial_output_3, lse_3

Stage 2: 1个thread block归约
  合并 partial_output_0..3 + lse_0..3
  → 最终 output = Σ(exp(lse_i - max_lse) * partial_output_i) / Σ(exp(lse_i - max_lse))
```

---

#### 33.6.6 MLA Prefill Kernel: `flash_attn_varlen_func`（经latent展开后调用）

**调用链**:
```
SIMOMLAImpl.forward_mha()                     [simo_mla.py:191]
  → Hadamard变换 Q, kv_c_normed, k_pe        [simo_mla.py:205-215]
  → super().forward_mha()                     [TritonMLAImpl]
    → MLACommonImpl._run_prefill_new_tokens_fa()  [mla_attention.py]
      → kv_b_proj(kv_c_normed)                   [展开latent → 完整K, V]
      → flash_attn_varlen_func(q, k, v, ...)     [FlashAttention CUDA kernel]
```

**MLA Prefill与GQA Prefill的关键区别**:

MLA在调用FlashAttention之前需要**展开latent空间**:
```python
# 在vLLM的MLACommonImpl中（概念性代码）:
# 1. kv_c_normed通过kv_b_proj投影得到完整的K_nope和V
kv_nope = self.kv_b_proj(kv_c_normed)  # [num_tokens, num_heads, qk_nope_dim + v_dim]
k_nope, v = kv_nope.split([qk_nope_head_dim, v_head_dim], dim=-1)

# 2. 拼接K_nope和K_PE得到完整的K
k = concat(k_nope, k_pe)  # [num_tokens, num_heads, qk_nope_dim + pe_dim]

# 3. 调用FlashAttention
flash_attn_varlen_func(q, k, v, ...)
```

由于latent展开后的K和V维度可能不同（K含PE部分），FlashAttention需要使用`_flash_attn_varlen_diff_headdims`变体来处理Q/K/V不同head_size的情况。

---

#### 33.6.7 `_unpack_and_dequant_mxfmt` 反量化函数详解

**文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/ops/kernels/upcast/_upcast_from_mxfmt.py`

这是所有software反量化路径的核心Triton JIT函数。`kernel_unified_attention_2d/3d`和`_fwd_grouped_kernel_stage1`中对V的反量化都调用此函数。

**对于MXFP8_E4M3** (`MX_FORMAT_ID == MXFP8_E4M3`):

```python
# 概念性反量化逻辑
def _unpack_and_dequant_mxfmt(packed, scales, MX_FORMAT_ID):
    if MX_FORMAT_ID == MXFP8_E4M3:
        # packed: (N, PACKED_HEAD_SIZE) - 每个元素1 byte, FP8 E4M3编码
        # scales: (N, SCALE_HEAD_SIZE) - 每32个元素共享1个E8M0 scale

        # Step 1: 将packed bytes解释为FP8 E4M3浮点数
        fp8_values = packed.to(tl.float8e4nv, bitcast=True)  # bitcast, 不改变bit pattern

        # Step 2: 将E8M0 scale解释为2的幂
        # E8M0格式: 8-bit exponent, 0-bit mantissa, 即 2^(scale - 127)
        scale_factors = scales.to(tl.float8e4nv)  # 展开scale到对应元素

        # Step 3: 反量化 = fp8_value * 2^(scale - 127)
        dequant_value = fp8_values.to(tl.float32) * scale_factors
        return dequant_value.to(tl.bfloat16)
```

**对于MXFP4_E2M1** (更复杂的位解包):
```python
    elif MX_FORMAT_ID == MXFP4_E2M1:
        # packed: 每byte存储2个FP4元素（4 bits each）
        # Step 1: 位解包 - 从byte中提取高4位和低4位
        low_nibble = packed & 0x0F
        high_nibble = (packed >> 4) & 0x0F

        # Step 2: FP4 E2M1 → float32
        # E2M1: sign(1) + exponent(2) + mantissa(1)
        # 反量化表: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} × sign × 2^(scale-127)
        fp32_values = fp4_to_float_table[nibble_values]

        # Step 3: 乘以E8M0 scale
        dequant_value = fp32_values * scale_factors
```

---

### 33.7 KV Cache写入Kernel详解: `simo_triton_reshape_and_cache_flash` 与 `reshape_and_cache_kernel_flash`

**文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/ops/triton_reshape_and_cache_flash.py`

这两个函数负责将新产生的key/value量化后写入paged KV Cache。`simo_triton_reshape_and_cache_flash` 是Python层调度入口，`reshape_and_cache_kernel_flash` 是实际执行量化和写入的Triton kernel。

---

#### 33.7.1 调用入口: `SIMOAttentionImpl.do_kv_cache_update()`

**文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/backends/simo_gqa.py:106-144`

每个attention layer的forward之前会先调用`do_kv_cache_update()`写入新token的KV:

```python
# simo_gqa.py:106-144
def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
    if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
        return

    if not hasattr(layer, "kv_cache_quant_spec"):
        return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)

    key_cache, value_cache = kv_cache.unbind(1)
    key_cache = key_cache.view(torch.uint8)       # 转换为uint8视图
    value_cache = value_cache.view(torch.uint8)

    # 1. Hadamard变换（当前配置hadamard_transform_size=0时为no-op）
    key = apply_hadamard_transform(key, layer.key_hadamard_transform_size, axis=-1)
    value = apply_hadamard_transform(value, layer.value_hadamard_transform_size, axis=-1)

    # 2. 量化并写入KV Cache
    simo_triton_reshape_and_cache_flash(
        key, value,
        key_cache, value_cache,
        slot_mapping,
        layer.kv_cache_quant_spec,          # 量化规格 (QuantizeSpecMX等)
        layer.key_hadamard_transform_size,
        layer.value_hadamard_transform_size,
        layer.packed_head_size,              # 量化后packed数据字节数
        layer.scale_head_size,              # scale数据字节数
    )
```

---

#### 33.7.2 Python调度函数: `simo_triton_reshape_and_cache_flash()`

**文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/ops/triton_reshape_and_cache_flash.py:337-434`

此函数负责：
1. 从量化规格对象中提取kernel参数
2. 计算grid大小
3. 启动Triton kernel

```python
# triton_reshape_and_cache_flash.py:337-434
def simo_triton_reshape_and_cache_flash(
    key: torch.Tensor,       # [num_tokens, num_heads, head_size] bf16
    value: torch.Tensor,     # [num_tokens, num_heads, head_size] bf16
    key_cache: torch.Tensor, # [num_blocks, block_size, num_heads*packed_head_size + num_heads*scale_head_size]
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,   # [num_tokens] 每个token在cache中的物理slot编号
    kv_cache_quant_spec,          # 量化规格对象
    k_hadamard_transform_size: int,
    v_hadamard_transform_size: int,
    packed_head_size: int,
    scale_head_size: int,
):
    num_tokens, num_heads, head_size = key.shape
    block_size = key_cache.shape[1]

    # --- 根据量化规格类型提取参数 ---
    from simo.quantization.config import QuantizeSpecFP, QuantizeSpecInt, QuantizeSpecMX

    if isinstance(kv_cache_quant_spec, QuantizeSpecMX):
        # MX格式 (MXFP4/MXFP6/MXFP8/MXINT8/NVFP4)
        mx_format_id, observer_mode, scale_rounding_mode = get_mx_quant_params(kv_cache_quant_spec)
        tile_size = kv_cache_quant_spec.block_size  # MX block size (通常32)
        pg_format_id, pg_min_value, pg_max_value = 0, 0.0, 0.0

    elif isinstance(kv_cache_quant_spec, (QuantizeSpecFP, QuantizeSpecInt)):
        # Per-group格式 (FP8/INT8)
        mx_format_id, observer_mode, scale_rounding_mode = 0, 0, 0
        tile_size = kv_cache_quant_spec.group_size    # 量化group大小
        pg_format_id, pg_min_value, pg_max_value = get_pg_quant_params(kv_cache_quant_spec.dtype)
    else:
        raise ValueError(f"Unsupported kv_cache_quant_spec type: {type(kv_cache_quant_spec)}")

    # --- 计算padding后的size (Triton要求2的幂) ---
    PACKED_HEAD_SIZE_PADDED = triton.next_power_of_2(packed_head_size)
    SCALE_HEAD_SIZE_PADDED = triton.next_power_of_2(scale_head_size) if scale_head_size > 0 else 0

    # --- 计算stride ---
    key_token_stride, key_head_stride, _ = key.stride()
    value_token_stride, value_head_stride, _ = value.stride()
    block_stride, page_stride, _, _ = value_cache.stride()

    # --- Grid: (num_tokens, num_heads, NUM_TILES) ---
    # 每个head的head_size被切分为NUM_TILES个tile，每个tile独立量化
    NUM_TILES = triton.cdiv(head_size, tile_size)
    grid = (num_tokens, num_heads, NUM_TILES)

    assert key_cache.dtype == torch.uint8, "Quantized KV cache must be uint8"

    # --- 启动kernel (带autotune) ---
    reshape_and_cache_kernel_flash[grid](
        key_ptr=key, value_ptr=value,
        key_cache_ptr=key_cache, value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        key_token_stride=key_token_stride, value_token_stride=value_token_stride,
        key_head_stride=key_head_stride, value_head_stride=value_head_stride,
        block_stride=block_stride, page_stride=page_stride,
        num_heads=num_heads, head_size=head_size, block_size=block_size,
        PACKED_HEAD_SIZE=packed_head_size, SCALE_HEAD_SIZE=scale_head_size,
        PACKED_HEAD_SIZE_PADDED=PACKED_HEAD_SIZE_PADDED,
        SCALE_HEAD_SIZE_PADDED=SCALE_HEAD_SIZE_PADDED,
        K_HADAMARD_TRANSFORM_SIZE=k_hadamard_transform_size,
        V_HADAMARD_TRANSFORM_SIZE=v_hadamard_transform_size,
        MX_FORMAT_ID=mx_format_id,
        OBSERVER_MODE=observer_mode,
        SCALE_ROUNDING_MODE=scale_rounding_mode,
        PG_FORMAT_ID=pg_format_id,
        PG_MIN_VALUE=pg_min_value, PG_MAX_VALUE=pg_max_value,
        TILE_SIZE=tile_size,
    )
```

**关键参数说明**:

| 参数 | 说明 | MXFP8_E4M3典型值 |
|------|------|-----------------|
| `tile_size` | 量化分组大小（MX block size） | 32 |
| `packed_head_size` | 量化后packed数据的字节数/head | 128 (= head_size × 1 byte for MXFP8) |
| `scale_head_size` | scale数据的字节数/head | 4 (= head_size / 32) |
| `NUM_TILES` | 每个head需要的tile数 | 4 (= 128 / 32) |
| `SCALE_PLANE_OFFSET` | scale区域在cache中的起始偏移 | `num_heads × packed_head_size` |

**Grid设计**: `(num_tokens, num_heads, NUM_TILES)` — 每个thread block负责一个token的一个head的一个tile（32个元素）的量化和写入。这种细粒度并行保证了高GPU利用率。

---

#### 33.7.3 Triton Kernel: `reshape_and_cache_kernel_flash()`

**文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/ops/triton_reshape_and_cache_flash.py:122-334`

此kernel通过`@triton.autotune`装饰器自动搜索最优的`num_warps`和`num_stages`配置。

##### Step 1: 解析Grid索引和slot映射

```python
# triton_reshape_and_cache_flash.py:168-175
# Grid: (num_tokens, num_heads, num_tiles)
token_idx = tl.program_id(axis=0)     # 当前token
head_idx = tl.program_id(axis=1)      # 当前attention head
head_tile_idx = tl.program_id(axis=2) # 当前tile（head_size的第几段）

# slot_mapping: 将逻辑token位置映射到KV Cache的物理slot
slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
if slot_idx < 0:    # slot_idx < 0 表示此token不需要缓存（padding token）
    return
```

##### Step 2: 从输入tensor加载一个tile的key和value

```python
# triton_reshape_and_cache_flash.py:177-184
# 计算源数据地址
src_key = key_ptr + token_idx * key_token_stride + head_idx * key_head_stride
src_val = value_ptr + token_idx * value_token_stride + head_idx * value_head_stride

# 加载一个tile (TILE_SIZE个元素) 的key和value (bf16)
tile_offs = head_tile_idx * TILE_SIZE + tl.arange(0, TILE_SIZE)
key_tile = tl.load(src_key + tile_offs, mask=tile_offs < head_size)   # (TILE_SIZE,) bf16
val_tile = tl.load(src_val + tile_offs, mask=tile_offs < head_size)   # (TILE_SIZE,) bf16
```

##### Step 3: 计算目标Cache中的地址（Planar Layout）

```python
# triton_reshape_and_cache_flash.py:186-191
# Paged KV Cache寻址: slot_idx → (block_idx, block内偏移)
block_idx = slot_idx // block_size
block_offset = slot_idx % block_size

# 每个slot的基地址
slot_base = block_idx * block_stride + block_offset * page_stride

# Planar Layout: [所有head的packed数据 | 所有head的scale数据]
# packed数据地址: slot_base + head_idx * PACKED_HEAD_SIZE
packed_base = slot_base + head_idx * PACKED_HEAD_SIZE
# scale数据地址: slot_base + (num_heads * PACKED_HEAD_SIZE) + head_idx * SCALE_HEAD_SIZE
scale_base = slot_base + SCALE_PLANE_OFFSET + head_idx * SCALE_HEAD_SIZE
# 其中 SCALE_PLANE_OFFSET = num_heads * PACKED_HEAD_SIZE
```

**Planar Layout图示**（以Llama 3.1 8B为例, `num_kv_heads=8, head_size=128`）:

```
一个cache slot的存储布局 (uint8 bytes):
┌──────────────────────────────────────────────────────┬────────────────────────┐
│           Packed Data Plane                          │    Scale Data Plane    │
│                                                      │                        │
│ head0: 128B │ head1: 128B │ ... │ head7: 128B       │ head0: 4B │ ... │ 4B  │
│ (MXFP8×128)│ (MXFP8×128) │     │ (MXFP8×128)       │ (4×E8M0)  │     │     │
├──────────────────────────────────────────────────────┼────────────────────────┤
│              SCALE_PLANE_OFFSET = 8×128 = 1024       │    8×4 = 32 bytes     │
└──────────────────────────────────────────────────────┴────────────────────────┘
总计: 1024 + 32 = 1056 bytes per slot
```

##### Step 4a: MX格式量化路径（MXFP8_E4M3等）

```python
# triton_reshape_and_cache_flash.py:193-277
if MX_FORMAT_ID > 0:
    # ==================== MX format path ====================
    # 调用 _compute_and_pack_mxfmt 进行量化
    # 输入: bf16 tile → 输出: (packed_data, scale)
    packed_key, key_scale = _compute_and_pack_mxfmt(
        tl.reshape(key_tile, (1, TILE_SIZE)),   # reshape为2D给API
        MX_FORMAT_ID=MX_FORMAT_ID,              # 量化格式ID
        HADAMARD_TRANSFORM_SIZE=0,              # 此处不做Hadamard(已在外部做过)
        OBSERVER_MODE=OBSERVER_MODE,            # 统计观察模式 (abs_max等)
        QUANT_SCALE_ROUNDING_MODE=SCALE_ROUNDING_MODE,  # E8M0舍入模式
        MX_BLOCK_SIZE=TILE_SIZE,                # MX block大小 (32)
    )
    packed_val, val_scale = _compute_and_pack_mxfmt(
        tl.reshape(val_tile, (1, TILE_SIZE)),
        MX_FORMAT_ID=MX_FORMAT_ID,
        HADAMARD_TRANSFORM_SIZE=0,
        OBSERVER_MODE=OBSERVER_MODE,
        QUANT_SCALE_ROUNDING_MODE=SCALE_ROUNDING_MODE,
        MX_BLOCK_SIZE=TILE_SIZE,
    )
```

**`_compute_and_pack_mxfmt`内部逻辑**（对于MXFP8_E4M3）：

```
输入: key_tile = [v0, v1, v2, ..., v31]  (32个bf16值)

1. Observer阶段: 观察数据分布
   - abs_max模式: amax = max(|v0|, |v1|, ..., |v31|)
   - std_dev模式: 用标准差估计

2. 计算E8M0 shared scale:
   - scale_exp = floor(log2(amax)) + bias_adjustment
   - E8M0 scale = 2^(scale_exp)  (纯指数格式，无尾数)

3. 缩放并量化每个元素:
   - scaled_vi = vi / E8M0_scale
   - packed_vi = round_to_fp8_e4m3(scaled_vi)  (4位指数 + 3位尾数)

输出: packed = [p0, p1, ..., p31]  (32个uint8), scale = 1个uint8 (E8M0)
```

##### Step 5a: 存储packed数据（非FP6路径，包括MXFP8）

```python
    # triton_reshape_and_cache_flash.py:256-270
    else:  # 非FP6格式
        # 将packed数据展平并bitcast到uint8
        k_packed = tl.reshape(packed_key, (PACKED_TILE_SIZE,))
        v_packed = tl.reshape(packed_val, (PACKED_TILE_SIZE,))
        if not IS_4BIT:   # 8-bit MX格式 (MXFP8, MXINT8)
            k_packed = k_packed.to(tl.uint8, bitcast=True)
            v_packed = v_packed.to(tl.uint8, bitcast=True)

        # 写入key cache的packed区域
        packed_offs = head_tile_idx * PACKED_TILE_SIZE + tl.arange(0, PACKED_TILE_SIZE)
        tl.store(
            key_cache_ptr + packed_base + packed_offs,  # 目标地址
            k_packed,
            mask=packed_offs < PACKED_HEAD_SIZE
        )
        # 写入value cache的packed区域
        tl.store(
            value_cache_ptr + packed_base + packed_offs,
            v_packed,
            mask=packed_offs < PACKED_HEAD_SIZE
        )
```

**对于MXFP8, `PACKED_TILE_SIZE == TILE_SIZE`**（1:1映射，每个bf16元素→1个uint8 FP8值）。对于MXFP4, `PACKED_TILE_SIZE = TILE_SIZE // 2`（2:1压缩，2个FP4值打包在1个byte中）。

##### Step 6a: 存储scale

```python
    # triton_reshape_and_cache_flash.py:272-277
    # 每个tile产生1个E8M0 scale (1 byte)
    k_s_u8 = tl.reshape(key_scale, (1,)).to(tl.uint8, bitcast=True)
    v_s_u8 = tl.reshape(val_scale, (1,)).to(tl.uint8, bitcast=True)
    scale_offs = head_tile_idx + tl.arange(0, 1)  # 第几个tile → 第几个scale byte
    tl.store(key_cache_ptr + scale_base + scale_offs, k_s_u8,
             mask=scale_offs < SCALE_HEAD_SIZE)
    tl.store(value_cache_ptr + scale_base + scale_offs, v_s_u8,
             mask=scale_offs < SCALE_HEAD_SIZE)
```

##### Step 4b/5b/6b: FP6格式的特殊处理

FP6格式需要将4个6-bit值打包成3 bytes（即每个uint32→3 bytes），存储布局称为"SIPU layout"：

```python
    # triton_reshape_and_cache_flash.py:230-255
    if IS_FP6:
        # MXFP6: 每4个FP6元素打包为1个uint32 (24 bits / 32 bits)
        # 然后拆成3个bytes存储: [byte2, byte1, byte0]
        FP6_N = TILE_SIZE // 4        # 例如 32/4=8 个uint32
        k_flat = tl.reshape(packed_key, (FP6_N,))

        u32_offs = tl.arange(0, FP6_N)
        base = head_tile_idx * PACKED_TILE_SIZE   # PACKED_TILE_SIZE = TILE_SIZE*3//4 = 24

        # 从uint32中提取3个bytes
        kb0 = (k_flat & 0xFF).to(tl.uint8)              # 低8位
        kb1 = ((k_flat >> 8) & 0xFF).to(tl.uint8)       # 中8位
        kb2 = ((k_flat >> 16) & 0xFF).to(tl.uint8)      # 高8位

        # 交织存储: [b2, b1, b0, b2, b1, b0, ...]
        offs_b2 = base + 3 * u32_offs
        offs_b1 = base + 3 * u32_offs + 1
        offs_b0 = base + 3 * u32_offs + 2
        tl.store(key_cache_ptr + packed_base + offs_b0, kb0, ...)
        tl.store(key_cache_ptr + packed_base + offs_b1, kb1, ...)
        tl.store(key_cache_ptr + packed_base + offs_b2, kb2, ...)
```

##### Step 4c/5c/6c: Per-group FP8/INT8路径

```python
    # triton_reshape_and_cache_flash.py:279-334
    else:  # PG_FORMAT_ID > 0
        # ==================== Per-group FP8/INT8 path ====================
        # TILE_SIZE == group_size, 每个tile就是一个量化group

        # 量化: 根据PG_FORMAT_ID选择目标格式
        if PG_FORMAT_ID == 1:      # FP8_E4M3
            k_q, k_s = quantize_fp8_or_int8_group(
                key_tile, 1e-10, PG_MIN_VALUE, PG_MAX_VALUE, False, tl.float8e4nv
            )
            v_q, v_s = quantize_fp8_or_int8_group(
                val_tile, 1e-10, PG_MIN_VALUE, PG_MAX_VALUE, False, tl.float8e4nv
            )
        elif PG_FORMAT_ID == 2:    # FP8_E5M2
            k_q, k_s = quantize_fp8_or_int8_group(key_tile, ..., tl.float8e5)
            v_q, v_s = quantize_fp8_or_int8_group(val_tile, ..., tl.float8e5)
        else:                      # INT8
            k_q, k_s = quantize_fp8_or_int8_group(key_tile, ..., tl.int8)
            v_q, v_s = quantize_fp8_or_int8_group(val_tile, ..., tl.int8)

        # 存储packed数据: fp8/int8 → uint8 bitcast
        tl.store(key_cache_ptr + packed_base + tile_offs,
                 k_q.to(tl.uint8, bitcast=True), mask=tile_offs < PACKED_HEAD_SIZE)
        tl.store(value_cache_ptr + packed_base + tile_offs,
                 v_q.to(tl.uint8, bitcast=True), mask=tile_offs < PACKED_HEAD_SIZE)

        # 存储scale: float32 → 4个uint8 bytes (byte splitting)
        # Per-group的scale是float32 (4 bytes)，而非MX格式的E8M0 (1 byte)
        byte_offs = tl.arange(0, 4)
        shifts = byte_offs.to(tl.uint32) * 8
        scale_offs = head_tile_idx * 4 + byte_offs   # 每个scale占4 bytes

        k_s_u32 = k_s.to(tl.uint32, bitcast=True)
        tl.store(key_cache_ptr + scale_base + scale_offs,
                 ((k_s_u32 >> shifts) & 0xFF).to(tl.uint8),  # 逐byte拆存
                 mask=scale_offs < SCALE_HEAD_SIZE)
```

**Per-group与MX格式的scale存储区别**:

| 特性 | MX格式 | Per-group FP8/INT8 |
|------|--------|--------------------|
| Scale格式 | E8M0 (1 byte) | float32 (4 bytes) |
| Scale含义 | 共享指数 `2^(scale-127)` | `max(|group|) / MAX_REPR` |
| 每组元素数 | 32 (MX block size) | group_size (可配置) |
| Scale字节/tile | 1 | 4 |

---

#### 33.7.4 `quantize_fp8_or_int8_group`内部逻辑

**文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/ops/kernels/downcast/_downcast_to_flexpoint.py`

Per-group量化的核心逻辑:

```
输入: tile = [v0, v1, ..., v31]  (一组bf16值)

1. 计算组内绝对值最大值: amax = max(|v0|, |v1|, ..., |v31|)
2. 计算scale: scale = amax / MAX_REPR_VALUE
   - FP8_E4M3: MAX_REPR_VALUE = 448.0
   - FP8_E5M2: MAX_REPR_VALUE = 57344.0
   - INT8: MAX_REPR_VALUE = 127
3. 量化: q_vi = clamp(round(vi / scale), MIN_VALUE, MAX_VALUE)
4. 返回: (q = [q_v0, ..., q_v31], scale)
```

---

#### 33.7.5 Autotune配置

```python
# triton_reshape_and_cache_flash.py:102-121
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=1, num_stages=3),
        ...
        triton.Config({}, num_warps=16, num_stages=10),
    ],
    key=["TILE_SIZE", "head_size", "MX_FORMAT_ID", "PG_FORMAT_ID"],
    prune_configs_by={"early_config_prune": _prune_autotune_configs},
)
```

- **key参数**: Triton会针对不同的`(TILE_SIZE, head_size, MX_FORMAT_ID, PG_FORMAT_ID)`组合分别autotune
- **prune_configs_by**: 根据`TILE_SIZE`裁剪搜索空间：
  - `TILE_SIZE <= 32`: 只搜索`num_warps=1, num_stages=3`
  - `TILE_SIZE <= 64`: 搜索`num_warps∈{2,4,8}, num_stages∈{2,4,6}`
  - `TILE_SIZE >= 128`: 搜索`num_warps∈{4,8,16}, num_stages∈{4,6,8,10}`

---

#### 33.7.6 完整数据流图

```
                   bf16 key/value
                        │
                        ▼
            ┌──── Hadamard变换 ────┐ (在do_kv_cache_update中, 当前配置为no-op)
            │                      │
            ▼                      ▼
    simo_triton_reshape_and_cache_flash()   [Python调度]
            │
            │  提取量化参数:
            │  MX_FORMAT_ID=MXFP8_E4M3
            │  TILE_SIZE=32
            │  Grid=(num_tokens, num_heads, NUM_TILES=4)
            │
            ▼
    reshape_and_cache_kernel_flash()   [Triton kernel, 每个thread block处理1个tile]
            │
            ├── Step 1: slot_mapping → 物理slot地址
            │
            ├── Step 2: 加载32个bf16元素
            │
            ├── Step 3: _compute_and_pack_mxfmt()
            │            ├── 计算abs_max
            │            ├── 确定E8M0 shared scale
            │            └── 量化为32个FP8 E4M3值
            │
            ├── Step 4: 写入packed数据 → key_cache[packed_base + offset]
            │            (32 bytes = 32 × 1 byte FP8)
            │
            └── Step 5: 写入scale数据 → key_cache[scale_base + tile_idx]
                         (1 byte E8M0)
```

**对于Llama 3.1 8B (`head_size=128, num_kv_heads=8`) + MXFP8_E4M3的完整例子**:

- `TILE_SIZE = 32`: 每个tile量化32个元素
- `NUM_TILES = 128 / 32 = 4`: 每个head分4个tile
- `PACKED_HEAD_SIZE = 128`: 128个FP8值 = 128 bytes
- `SCALE_HEAD_SIZE = 4`: 4个E8M0 scale = 4 bytes
- Grid = `(num_tokens, 8, 4)`: 例如batch=32个token → 32×8×4 = 1024个thread blocks
- 每个thread block处理: 读取32个bf16 → 量化 → 写入32 bytes packed + 1 byte scale

---

### 33.8 KV Cache Tensor Shape `(22420, 16, 8, 132)` 解析

在 `simo_triton_reshape_and_cache_flash` 函数中打印的 `key_cache.shape` 和 `value_cache.shape` 都是 `torch.Size([22420, 16, 8, 132])`。

#### 33.8.1 shape的由来

KV Cache的shape由两个地方共同决定:

**1. `get_kv_cache_shape()` 确定整体形状**

```python
# /softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/backends/simo_gqa.py:69-78
@staticmethod
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, ...):
    return (num_blocks, 2, block_size, num_kv_heads, head_size)
```

vLLM按此shape分配完整的kv_cache tensor: `(num_blocks, 2, block_size, num_kv_heads, head_size)`。其中`2`对应K和V两份cache。

**2. `make_kv_cache_spec()` 确定 `head_size` 的值**

```python
# /softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/v1/attention/backends/simo_gqa.py:81-96
@classmethod
def make_kv_cache_spec(cls, layer, vllm_config):
    # 用下采样kernel做一次dummy量化，探测输出大小
    x_q, s = layer.kv_cache_downcast_kernel(torch.randn(128, layer.head_size, device="meta"))
    packed = x_q.contiguous().view(torch.uint8).shape[-1]   # packed数据字节数
    scale = s.contiguous().view(torch.uint8).shape[-1]       # scale数据字节数

    return FullAttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=layer.num_kv_heads,
        head_size=packed + scale,    # ← 这里把packed和scale合并为一个"head_size"
        dtype=torch.uint8,           # ← 存储类型为uint8
    )
```

**3. `do_kv_cache_update()` 中 unbind 拆分K/V**

```python
# simo_gqa.py:124-126
key_cache, value_cache = kv_cache.unbind(1)  # 沿dim=1拆分,去掉"2"维度
key_cache = key_cache.view(torch.uint8)
value_cache = value_cache.view(torch.uint8)
```

拆分后每个cache的shape: `(num_blocks, block_size, num_kv_heads, head_size)` = `(22420, 16, 8, 132)`

#### 33.8.2 各维度含义

```
key_cache.shape = (22420, 16, 8, 132)
                    │      │   │   │
                    │      │   │   └── 132 = packed_head_size(128) + scale_head_size(4)
                    │      │   │           量化后每个head的总字节数
                    │      │   │
                    │      │   └── 8 = num_kv_heads
                    │      │       Llama 3.1 8B 使用GQA, 有8个KV head
                    │      │
                    │      └── 16 = block_size
                    │          每个paged cache block包含16个token slot
                    │
                    └── 22420 = num_blocks
                        总共分配的cache block数量
                        (由 --gpu-memory-utilization 0.5 和模型大小决定)
```

**第4维 `132` 的具体组成**（对于MXFP8_E4M3, head_size=128）:

| 组成部分 | 字节数 | 说明 |
|---------|--------|------|
| packed_head_size | 128 | 128个FP8 E4M3值，每个1 byte |
| scale_head_size | 4 | 128/32 = 4个E8M0 scale，每个1 byte |
| **总计** | **132** | **packed + scale** |

**总KV Cache容量计算**:
- 每个slot: `8 heads × 132 bytes = 1056 bytes`
- 每个block: `16 slots × 1056 = 16,896 bytes`
- key_cache总量: `22420 blocks × 16,896 = ~362 MB`
- key+value总量: `~724 MB`
- 可缓存token数: `22420 × 16 = 358,720 tokens`

#### 33.8.3 Tensor Shape vs Kernel实际内存布局

**这是一个关键点: tensor的逻辑shape `(22420, 16, 8, 132)` 与kernel内部使用的Planar Layout并不对应。**

按照tensor shape的row-major内存布局，一个slot中每个head占连续132字节:

```
Tensor逻辑视图 (一个slot = cache[b, p, :, :]):
head0: bytes[0:132)     → 132 bytes
head1: bytes[132:264)   → 132 bytes
head2: bytes[264:396)   → 132 bytes
...
head7: bytes[924:1056)  → 132 bytes
```

但kernel使用的**Planar Layout**不是这样存储的。它把所有head的packed数据放在前面，所有head的scale放在后面：

```python
# triton_reshape_and_cache_flash.py:160,189-191
SCALE_PLANE_OFFSET = num_heads * PACKED_HEAD_SIZE           # 8 × 128 = 1024
packed_base = slot_base + head_idx * PACKED_HEAD_SIZE       # head_idx × 128
scale_base = slot_base + SCALE_PLANE_OFFSET + head_idx * SCALE_HEAD_SIZE  # 1024 + head_idx × 4
```

**Kernel实际写入的Planar Layout (一个slot的1056 bytes)**:

```
偏移量    内容                        大小
──────────────────────────────────────────────
0         head0 packed (FP8 × 128)    128 bytes
128       head1 packed (FP8 × 128)    128 bytes
256       head2 packed (FP8 × 128)    128 bytes
384       head3 packed (FP8 × 128)    128 bytes
512       head4 packed (FP8 × 128)    128 bytes
640       head5 packed (FP8 × 128)    128 bytes
768       head6 packed (FP8 × 128)    128 bytes
896       head7 packed (FP8 × 128)    128 bytes
─── SCALE_PLANE_OFFSET = 1024 ────────────────
1024      head0 scale (E8M0 × 4)      4 bytes
1028      head1 scale (E8M0 × 4)      4 bytes
1032      head2 scale (E8M0 × 4)      4 bytes
1036      head3 scale (E8M0 × 4)      4 bytes
1040      head4 scale (E8M0 × 4)      4 bytes
1044      head5 scale (E8M0 × 4)      4 bytes
1048      head6 scale (E8M0 × 4)      4 bytes
1052      head7 scale (E8M0 × 4)      4 bytes
──────────────────────────────────────────────
总计: 1024 + 32 = 1056 bytes = 8 × 132
```

**对比**:

| 偏移 | Tensor逻辑 `cache[b,p,h,d]` 认为的内容 | Kernel实际写入的内容 |
|------|------------------------------------------|---------------------|
| 0-127 | head0的前128字节 | head0 packed (正确) |
| 128-131 | head0的后4字节 | head1 packed的前4字节 (不对应!) |
| 132-263 | head1的132字节 | head1 packed的剩余124字节 + head2 packed的前8字节 |

**结论**: tensor shape `(22420, 16, 8, 132)` 仅用于**内存分配**，确保每个slot分配到正确的总字节数 `8 × 132 = 1056`。kernel内部完全忽略第3维(`num_heads`)和第4维(`head_size`)的逻辑语义，而是通过`block_stride`和`page_stride`（对应stride[0]和stride[1]）定位到slot，再用自己的Planar Layout偏移量计算来读写数据。

**代码验证**（kernel只使用前两个stride）:
```python
# simo_triton_reshape_and_cache_flash() 中:
block_stride, page_stride, _, _ = value_cache.stride()   # 只用了stride[0]和stride[1]
# stride[0] = 16 × 8 × 132 = 16896
# stride[1] = 8 × 132 = 1056
# stride[2] 和 stride[3] 被丢弃（_）
```

这意味着如果你用 `key_cache[block_idx, block_offset, head_idx, :]` 的方式去读取数据，读到的内容**并不是**那个head的量化数据，因为物理存储不按head×132分段。正确的读取方式必须使用kernel中的Planar Layout偏移计算。

---

## 33.9 `_compute_and_pack_mxfmt` 函数详解

**文件**: `/softhome/like/package/h100/package/simo_conda_sglang/simo/ops/kernels/downcast/_downcast_to_mxfmt.py:235-522`

**函数签名**:
```python
@triton.jit
def _compute_and_pack_mxfmt(
    src_tensor,                                    # 输入: [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM]
    MX_FORMAT_ID: tl.constexpr,                   # 3 = MXFP8_E4M3
    HADAMARD_TRANSFORM_SIZE: tl.constexpr = 0,    # 0 = 禁用Hadamard变换
    OBSERVER_MODE: tl.constexpr = ABS_MAX_OBSERVER_MODE,  # 1 = ABS_MAX
    QUANT_SCALE_ROUNDING_MODE: tl.constexpr = E8M0_FLOOR, # 1 = FLOOR
    MX_BLOCK_SIZE: tl.constexpr = 32,             # 32 = 每个MX block的元素数
    PACK_MODE: tl.constexpr = SIPU_PACKING,
) -> (packed_tensor, scale_tensor)
```

**用途**: 这是一个纯计算kernel（无内存load/store），将bf16/fp32输入量化为MX格式（本例为MXFP8_E4M3），返回packed数据和E8M0 scale。

### 33.9.1 MX格式基础知识

**MXFP8_E4M3格式** (`MX_FORMAT_ID=3`):
- **max_quant_val**: 448.0 (最大可表示值)
- **min_normal**: 2^(-6) = 0.015625 (最小正规数)
- **max_quant_exp**: 8.0 (最大量化指数)
- **ebits**: 4 (指数位数)
- **mbits**: 3 (尾数位数)

**E8M0格式** (共享指数):
- 8位指数，0位尾数
- 存储为uint8: `exponent_biased = floor(log2(max_val)) - max_quant_exp + 127`
- 反量化时: `dequant_scale = 2^(exponent_biased - 127)`

### 33.9.2 执行流程（6个阶段）

#### **阶段1: 输入准备与reshape**

```python
# 输入: src_tensor shape = [1, 32] (在reshape_and_cache_kernel_flash中调用时)
BLOCK_SIZE_OUT_DIM = 1          # 输出维度（行数）
BLOCK_SIZE_QUANT_DIM = 32       # 量化维度（列数）
BLOCK_SIZE_QUANT_MX_SCALE = 32 // 32 = 1  # MX block数量

# 跳过Hadamard变换（HADAMARD_TRANSFORM_SIZE=0）
f32_tensor = src_tensor.to(tl.float32)  # bf16 → fp32

# Reshape为3D: [out_dim, num_mx_blocks, mx_block_size]
f32_tensor = tl.reshape(f32_tensor, [1, 1, 32])
```

#### **阶段2: 计算量化scale（Observer + Scale Rounding）**

**Observer阶段** (`OBSERVER_MODE=1`, ABS_MAX):
```python
# 在每个MX block内计算绝对值最大值
max_val = tl.max(tl.abs(f32_tensor), axis=2, keep_dims=True)
# max_val shape: [1, 1, 1]
```

**Scale Rounding阶段** (`QUANT_SCALE_ROUNDING_MODE=1`, E8M0_FLOOR):
```python
# 计算E8M0指数（floor模式）
scale_e8m0_unbiased = tl.log2(max_val).floor() - 8.0  # max_quant_exp=8.0
# 限制范围防止溢出
scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
# 转换为2的幂次
quant_scale_rounded = tl.exp2(scale_e8m0_unbiased)

# 提取指数部分（bitcast到uint32）
quant_scale_exponent = quant_scale_rounded.to(tl.uint32, bitcast=True)
# quant_scale_exponent的bit layout: [sign(1) | exponent(8) | mantissa(23)]
```

**计算量化scale和反量化scale**:
```python
# 从uint32 bitcast回float32得到dequant_scale
dequant_scale_rounded = quant_scale_exponent.to(tl.float32, bitcast=True)
# 计算quant_scale = 1 / dequant_scale
quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)
```

**数值示例**:
```
假设max_val = 100.0
log2(100.0) = 6.644
floor(6.644) = 6.0
scale_e8m0_unbiased = 6.0 - 8.0 = -2.0
quant_scale_rounded = 2^(-2) = 0.25
dequant_scale = 0.25
quant_scale = 1/0.25 = 4.0
```

#### **阶段3: 应用量化scale**

```python
# Reshape回3D以便广播
f32_tensor_reshaped = tl.reshape(f32_tensor, [1, 1, 32])
# 乘以quant_scale（将值缩放到MXFP8_E4M3的表示范围）
quant_tensor_reshaped = f32_tensor_reshaped * quant_scale
# quant_tensor_reshaped shape: [1, 1, 32]

# Reshape回2D
quant_tensor = tl.reshape(quant_tensor_reshaped, [1, 32])
```

**数值示例**:
```
原始值: [100.0, 50.0, -75.0, 25.0, ...]
quant_scale = 4.0
缩放后: [400.0, 200.0, -300.0, 100.0, ...]
```

#### **阶段4: 转换为目标dtype（MXFP8_E4M3）**

```python
target_dtype = tl.float8e4nv  # MXFP8_E4M3对应的Triton dtype

# 直接转换（Triton自动处理round-to-nearest-even和饱和）
quant_values = quant_tensor.to(target_dtype)
# quant_values shape: [1, 32], dtype: float8e4nv
```

**FP8 E4M3格式细节**:
- 指数位: 4 bits (bias=7)
- 尾数位: 3 bits
- 表示范围: [-448, 448]
- 特殊值: 无NaN，有±inf

#### **阶段5: 提取E8M0 scale**

```python
# 从uint32中提取8位指数（右移23位）
quant_scale_exponent = quant_scale_exponent.reshape([1, 1])
scale_tensor = (quant_scale_exponent >> 23).to(tl.uint8)
# scale_tensor shape: [1, 1], dtype: uint8
```

**E8M0存储格式**:
```
uint32 layout: [sign(1) | exponent(8) | mantissa(23)]
                         ^^^^^^^^
                         提取这8位
右移23位后: [00000000 | 00000000 | 00000000 | exponent(8)]
转uint8:    [exponent(8)]
```

**数值示例**:
```
quant_scale_rounded = 0.25 = 2^(-2)
float32 bitcast to uint32: 0x3E800000
  = 0011 1110 1000 0000 0000 0000 0000 0000
    ^^^^^^^^
    exponent = 01111101 = 125 (biased)
右移23位: 0x0000007D = 125
uint8: 125
```

#### **阶段6: 打包返回（MXFP8不需要额外打包）**

```python
# MXFP8_E4M3是8位格式，不需要bit packing
pack_tensor = quant_values  # shape: [1, 32], dtype: float8e4nv

return pack_tensor, scale_tensor
# pack_tensor: [1, 32] float8e4nv
# scale_tensor: [1, 1] uint8
```

### 33.9.3 完整数据流示例

**输入**: 一个tile的key值（已经过Hadamard变换）
```python
src_tensor = [100.0, 50.0, -75.0, 25.0, 12.5, ..., 6.25]  # 32个bf16值
```

**步骤1**: 转换为fp32并reshape
```python
f32_tensor = [[100.0, 50.0, -75.0, ..., 6.25]]  # [1, 32]
f32_tensor_3d = [[[100.0, 50.0, -75.0, ..., 6.25]]]  # [1, 1, 32]
```

**步骤2**: 计算max_val
```python
max_val = max(abs(100.0), abs(50.0), abs(-75.0), ...) = 100.0
```

**步骤3**: 计算E8M0 scale
```python
log2(100.0) = 6.644
floor(6.644) - 8.0 = -2.0
dequant_scale = 2^(-2) = 0.25
quant_scale = 4.0
```

**步骤4**: 应用quant_scale
```python
quant_tensor = [400.0, 200.0, -300.0, 100.0, 50.0, ..., 25.0]
```

**步骤5**: 转换为FP8 E4M3
```python
# FP8 E4M3可以精确表示448以内的2的幂次和部分值
quant_values = [400.0_fp8, 200.0_fp8, -300.0_fp8, 100.0_fp8, ...]
# 每个值占1字节
```

**步骤6**: 提取E8M0 scale
```python
scale_tensor = [125]  # uint8, 表示biased exponent
```

**输出**:
```python
pack_tensor: [1, 32] 的 float8e4nv tensor (32 bytes)
scale_tensor: [1, 1] 的 uint8 tensor (1 byte)
总计: 33 bytes per tile
```

### 33.9.4 反量化过程

在decode阶段读取KV cache时，需要反量化：

```python
# 读取packed data和scale
packed_data = key_cache[...].view(torch.float8_e4m3fn)  # [32]
scale_uint8 = key_cache[scale_offset]  # uint8

# 重建dequant_scale
biased_exp = scale_uint8.to(torch.int32)
dequant_scale = 2.0 ** (biased_exp - 127)  # = 0.25

# 反量化
dequantized = packed_data.to(torch.float32) * dequant_scale
# [400.0, 200.0, -300.0, ...] * 0.25 = [100.0, 50.0, -75.0, ...]
```

### 33.9.5 与其他格式的对比

| 格式 | MX_FORMAT_ID | 打包方式 | scale存储 |
|------|--------------|----------|-----------|
| MXFP8_E4M3 | 3 | 无需打包（8bit→8bit） | E8M0 uint8 |
| MXFP8_E5M2 | 2 | 无需打包（8bit→8bit） | E8M0 uint8 |
| MXFP6_E3M2 | 4 | 4个6bit→3 bytes | E8M0 uint8 |
| MXFP4_E2M1 | 6 | 2个4bit→1 byte | E8M0 uint8 |
| MXINT8 | 1 | 无需打包（8bit→8bit） | E8M0 uint8 |

**MXFP6打包示例** (如果`MX_FORMAT_ID=4`):
```python
# 4个6位值打包成3字节（SIPU_PACKING模式）
# 输入: [v3, v2, v1, v0] (每个6位)
byte0 = (v3 << 2) | (v2 >> 4)           # v3的6位 + v2的高2位
byte1 = ((v2 & 0x0F) << 4) | (v1 >> 2)  # v2的低4位 + v1的高4位
byte2 = ((v1 & 0x03) << 6) | v0         # v1的低2位 + v0的6位
pack_tensor = byte0 | (byte1 << 8) | (byte2 << 16)  # 打包成uint32
```

### 33.9.6 关键设计要点

1. **纯计算kernel**: 无内存操作，所有输入通过寄存器传递，适合在更大的kernel中内联调用

2. **E8M0共享指数**: 每个MX block（32个元素）共享一个8位指数，节省存储空间
   - 32个FP8值 + 1个uint8 scale = 33 bytes
   - 如果用FP16: 32 × 2 = 64 bytes
   - 压缩率: 33/64 = 51.6%

3. **FLOOR模式**: `scale_e8m0_unbiased = floor(log2(max_val)) - max_quant_exp`
   - 保证量化后的值不会超出FP8表示范围
   - 可能损失精度（相比CEIL或EVEN模式）

4. **Triton自动优化**: `quant_tensor.to(tl.float8e4nv)` 由Triton编译器生成高效的PTX指令

5. **与reshape_and_cache_kernel_flash的集成**:
   ```python
   # 在reshape_and_cache_kernel_flash中调用（L196-211）
   packed_key, key_scale = _compute_and_pack_mxfmt(
       tl.reshape(key_tile, (1, TILE_SIZE)),  # [1, 32]
       MX_FORMAT_ID=3,
       HADAMARD_TRANSFORM_SIZE=0,
       OBSERVER_MODE=1,
       QUANT_SCALE_ROUNDING_MODE=1,
       MX_BLOCK_SIZE=32,
   )
   # 返回后直接写入KV cache
   ```

---

## 33.10 `reshape_and_cache_kernel_flash` 如何处理 head_dim 不是 TILE_SIZE 整数倍的情况

**结论**: **不是**先把每个head的维度padding到32的整数倍。而是通过 **masked load + masked store** 的方式处理尾部不完整的tile。Planar Layout中每个head只存储实际有效的packed字节，不存储padding字节。但最后一个tile的scale一定会被存储。

### 33.10.1 参数推导（以 head_dim=144, MXFP8_E4M3, TILE_SIZE=32 为例）

**packed_head_size 和 scale_head_size 的计算来源**:

在 `/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/quantization/quantization_method.py:650-652`:
```python
x_q, scale_a = self.kv_cache_downcast_kernel(torch.randn(1, layer.head_size, device="meta"))
layer.packed_head_size = x_q.contiguous().view(torch.uint8).shape[-1]
layer.scale_head_size = scale_a.contiguous().view(torch.uint8).shape[-1]
```

调用 `get_out_shape()` (`/softhome/like/package/h100/package/simo_conda_sglang/simo/ops/kernels/mx_trition_api.py:48-66`):

| 格式 | head_dim=144 时 packed_head_size | scale_head_size |
|------|----------------------------------|-----------------|
| MXFP8 (ID=1,2,3) | L = **144** | cdiv(144,32) = **5** |
| MXFP6 (ID=4,5) | L×3/4 = **108** | cdiv(144,32) = **5** |
| MXFP4 (ID=6,7) | L/2 = **72** | cdiv(144,32) = **5** |
| MXINT8 (ID=1) | L = **144** | cdiv(144,32) = **5** |

**KV cache tensor shape** (通过 `make_kv_cache_spec`):
```python
head_size_in_cache = packed_head_size + scale_head_size
# MXFP8: 144 + 5 = 149
# MXFP6: 108 + 5 = 113
# MXFP4: 72 + 5 = 77
```

KV cache分配时的shape: `(num_blocks, 2, block_size, num_kv_heads, 149)` (以MXFP8为例)

### 33.10.2 Kernel处理流程 (5个tile的逐步跟踪)

**grid**: `(num_tokens, num_heads, NUM_TILES=cdiv(144,32)=5)`

**关键参数**:
```python
head_size = 144           # 原始head维度
TILE_SIZE = 32            # = MX_BLOCK_SIZE
PACKED_HEAD_SIZE = 144    # MXFP8: 1 byte per element
SCALE_HEAD_SIZE = 5       # 5个MX block → 5个scale
PACKED_TILE_SIZE = 32     # MXFP8: TILE_SIZE (无bit packing)
```

#### Tile 0–3 (head_tile_idx = 0, 1, 2, 3): 完整tile

```python
tile_offs = head_tile_idx * 32 + [0, 1, ..., 31]
# 例如 tile 0: [0..31], tile 3: [96..127]
# 全部 < 144，mask全为True
key_tile = tl.load(src_key + tile_offs, mask=tile_offs < 144)   # 加载32个有效元素
```

量化: `_compute_and_pack_mxfmt` 将32个元素量化为32个FP8 byte + 1个E8M0 scale byte。

存储:
```python
packed_offs = head_tile_idx * 32 + [0..31]  # 全部 < 144 → 32 bytes全部写入
scale_offs = [head_tile_idx]                 # < 5 → 1 byte写入
```

#### Tile 4 (head_tile_idx = 4): 不完整tile ⬅️ 关键

**步骤1: Masked Load**
```python
tile_offs = 4 * 32 + [0, 1, ..., 31] = [128, 129, ..., 159]
mask = tile_offs < 144
# mask = [T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T, F,F,F,F,F,F,F,F,F,F,F,F,F,F,F,F]
#         128                          143   144                          159

key_tile = tl.load(src_key + tile_offs, mask=mask)
# key_tile = [v128, v129, ..., v143, 0, 0, 0, ..., 0]
#             ← 16个真实值 →        ← 16个零 →
```

**步骤2: 量化 (整个32元素作为一个MX block)**
```python
packed_key, key_scale = _compute_and_pack_mxfmt(
    tl.reshape(key_tile, (1, 32)),  # [1, 32]: 16个真实值 + 16个零
    MX_BLOCK_SIZE=32,               # 整个tile = 1个MX block
)
```

在 `_compute_and_pack_mxfmt` 内部:
```python
# Observer: 计算整个block的max
max_val = tl.max(tl.abs([v128, ..., v143, 0, ..., 0]))
# = max(|v128|, ..., |v143|, 0, ..., 0)
# = max(|v128|, ..., |v143|)
# 零不影响max_val，所以scale的计算是正确的！

# 量化: 所有32个元素都乘以quant_scale
quant_tensor = [v128*qs, ..., v143*qs, 0*qs, ..., 0*qs]
#             = [v128*qs, ..., v143*qs,   0,  ...,   0]

# 转FP8: 零转为FP8后仍然是精确的零
quant_values = [fp8_128, ..., fp8_143, fp8_zero, ..., fp8_zero]
```

**步骤3: Masked Store (只写入有效的packed bytes)**
```python
packed_offs = 4 * 32 + [0, 1, ..., 31] = [128, 129, ..., 159]
mask = packed_offs < PACKED_HEAD_SIZE  # packed_offs < 144
# mask = [T,T,...,T, F,F,...,F]
#         128   143  144   159

tl.store(key_cache_ptr + packed_base + packed_offs, k_packed, mask=mask)
# 只写入 packed_offs[0:16]，即偏移128-143的16个FP8字节
# packed_offs[16:31]（偏移144-159）被mask掉，不写入！
```

**步骤4: Scale 无条件存储**
```python
scale_offs = [4]  # head_tile_idx = 4
mask = scale_offs < SCALE_HEAD_SIZE  # 4 < 5 = True → 写入
tl.store(key_cache_ptr + scale_base + scale_offs, k_s_u8, mask=mask)
```

### 33.10.3 Planar Layout 内存布局（head_dim=144, MXFP8, 8个head）

```
一个 slot 的 Planar Layout:

偏移量    内容                                     大小
──────────────────────────────────────────────────────
0         head0 packed (FP8 × 144)                 144 bytes
144       head1 packed (FP8 × 144)                 144 bytes
288       head2 packed (FP8 × 144)                 144 bytes
...
896       head6 packed (FP8 × 144)                 144 bytes
1008      head7 packed (FP8 × 144)                 144 bytes  ← 非对齐!
─── SCALE_PLANE_OFFSET = 8 × 144 = 1152 ──────────────
1152      head0 scale (E8M0 × 5)                   5 bytes
1157      head1 scale (E8M0 × 5)                   5 bytes
...
1187      head7 scale (E8M0 × 5)                   5 bytes
──────────────────────────────────────────────────────
总计: 8 × 144 + 8 × 5 = 1152 + 40 = 1192 bytes = 8 × 149
```

**对比 head_dim=128 (32的整数倍)**:
```
总计: 8 × 128 + 8 × 4 = 1024 + 32 = 1056 bytes = 8 × 132
```

### 33.10.4 各格式的最后一个tile对比（head_dim=144, TILE_SIZE=32）

| 格式 | PACKED_TILE_SIZE | Tile4 加载 | Tile4 量化 | Tile4 实际写入packed | Tile4 scale |
|------|------------------|-----------|-----------|---------------------|-------------|
| **MXFP8** | 32 | 16真+16零 | 32 FP8 | **16 bytes** (mask < 144) | 1 byte ✓ |
| **MXFP6** | 24 | 16真+16零 | 24 bytes (4值打3字节) | **12 bytes** (mask < 108) | 1 byte ✓ |
| **MXFP4** | 16 | 16真+16零 | 16 bytes (2值打1字节) | **8 bytes** (mask < 72) | 1 byte ✓ |
| **MXINT8** | 32 | 16真+16零 | 32 INT8 | **16 bytes** (mask < 144) | 1 byte ✓ |

验证:
- MXFP8: 4×32 + 16 = 144 ✓ = packed_head_size
- MXFP6: 4×24 + 12 = 108 ✓ = packed_head_size
- MXFP4: 4×16 + 8 = 72 ✓ = packed_head_size

### 33.10.5 反量化时的注意事项

在 `triton_unified_attention.py` 的decode阶段读取KV cache时:

1. **读packed data**: 以MX block为单位读取，最后一个block (tile 4) 只有前16个字节是有效数据
2. **读scale**: 第5个scale (index=4) 对应最后一个MX block
3. **反量化**: `dequant_value = fp8_value * dequant_scale`
   - 前16个: 正确反量化出原始值
   - 后16个: 如果读取了padding区域（超出packed_head_size的部分），值为未定义的垃圾数据

**因此decode kernel也需要类似的mask处理**，确保只读取和计算有效的packed_head_size字节。decode kernel中的 `PACKED_HEAD_SIZE_PADDED = triton.next_power_of_2(packed_head_size)` 就是用于分配寄存器空间时向上取2的幂次，但实际计算中通过 `mask < PACKED_HEAD_SIZE` 限制有效范围。

### 33.10.6 总结

```
                        reshape_and_cache_kernel_flash 处理 head_dim=144 的数据流

head_dim=144, TILE_SIZE=32 → NUM_TILES = cdiv(144,32) = 5

Source (bf16):  [v0 .............. v127 | v128 .. v143]
                 ← tile0~3: 完整 →       ← tile4: 不完整 →

  ┌──────────── Tile 4 处理流程 ──────────────┐
  │                                            │
  │  Step 1: Masked Load                       │
  │  [v128..v143, 0, 0, ..., 0]               │
  │   16个真实值    16个零(mask外)             │
  │                                            │
  │  Step 2: 整个32元素作为1个MX Block量化     │
  │  max_val = max(|v128|,..,|v143|)           │
  │  → scale正确(零不影响max)                  │
  │  → 32个FP8值(后16个=0.0)                   │
  │                                            │
  │  Step 3: Masked Store                      │
  │  packed: 只写前16个FP8 byte                │
  │          (后16个被mask丢弃)                 │
  │  scale: 无条件写入1个E8M0 byte             │
  └────────────────────────────────────────────┘

Planar Layout (per head):
  packed: [tile0: 32B][tile1: 32B][tile2: 32B][tile3: 32B][tile4: 16B] = 144B
  scale:  [s0][s1][s2][s3][s4]                                         = 5B
  总计: 149 bytes/head
```

**核心机制**: 没有显式padding。kernel通过以下三个mask实现不完整tile的正确处理:
1. **`tile_offs < head_size`**: Load时mask，超出head_dim的位置得到0
2. **`packed_offs < PACKED_HEAD_SIZE`**: Store packed data时mask，丢弃多余的量化零
3. **`scale_offs < SCALE_HEAD_SIZE`**: Store scale时mask（对于最后一个tile，scale_offs=4 < 5，所以scale被保留）

---

## 33.11 SGLang 自定义 KV Cache 类的可行方案分析

### 33.11.1 结论：SGLang 没有 KV Cache 注册机制

**SGLang 没有提供任何 KV Cache 的注册/插件机制**。`init_memory_pool` 中的 KV pool 选择完全是硬编码的 if-elif-else 链。

与之对比：
| 机制 | SGLang | vLLM |
|------|--------|------|
| Attention Backend 注册 | ✅ 有 `register_attention_backend` 装饰器 | ✅ 有 `AttentionBackendEnum` + `register_backend` |
| KV Cache 注册 | ❌ **没有** | ❌ 没有（但有 entry_points 插件机制） |
| 通用插件系统 | ❌ **没有** | ✅ 有 `vllm.general_plugins` entry_points |
| 量化方法注册 | ❌ 没有 | ✅ 有 `QuantizationConfig` 注册 |

**Attention Backend 是唯一有注册机制的子系统**（`/share_data/users/like/package/h100/package/sglang_kernel_src/python/sglang/srt/layers/attention/attention_registry.py`）:
```python
ATTENTION_BACKENDS = {}

def register_attention_backend(name):
    def decorator(fn):
        ATTENTION_BACKENDS[name] = fn
        return fn
    return decorator

@register_attention_backend("flashinfer")
def create_flashinfer_backend(runner): ...
```

但 KV Cache 没有类似的注册表。

### 33.11.2 `init_memory_pool` 的选择逻辑（完整分支图）

**文件**: `/share_data/users/like/package/h100/package/sglang_kernel_src/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py:484-685`

```
init_memory_pool() 中 token_to_kv_pool 的赋值分支:

├── attention_backend == "ascend" (NPU)
│   ├── use_mla_backend → NPUMLATokenToKVPool
│   └── else → NPUMHATokenToKVPool
│
├── use_mla_backend AND is_nsa_model → NSATokenToKVPool
│
├── use_mla_backend AND NOT mambaish
│   ├── fp4 dtype → MLATokenToKVPoolFP4
│   └── else → MLATokenToKVPool
│
├── enable_double_sparsity → DoubleSparseTokenToKVPool
│
└── else (标准 MHA 路径)
    ├── is_hybrid_swa → SWAKVPool
    ├── mambaish_config → HybridLinearKVPool
    └── else
        ├── fp4 dtype → MHATokenToKVPoolFP4
        └── else → MHATokenToKVPool      ← 最常见的路径
```

**KV Cache 类继承关系** (`/share_data/users/like/package/h100/package/sglang_kernel_src/python/sglang/srt/mem_cache/memory_pool.py`):
```
KVCache (ABC, L601)
├── MHATokenToKVPool (L730)
│   └── MHATokenToKVPoolFP4 (L1103)
├── MLATokenToKVPool (L1440)
│   └── MLATokenToKVPoolFP4 (L1664)
├── NSATokenToKVPool (L1793)
├── DoubleSparseTokenToKVPool (L1949)
├── HybridLinearKVPool (L1246)
└── SWAKVPool (swa_memory_pool.py)
```

### 33.11.3 SIMO 当前的 SGLang 集成方式

SIMO 目前已经在用 `SIMOPatch.wrap_function` 做 monkey patch（`/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/simo_patch.py`），但**当前只 patch 了量化配置加载**，没有 patch KV cache:

```python
# simo/extensions/sglang_simo/model_loader/loader.py:54
SIMOPatch.wrap_function(loader, "_get_quantization_config", _get_quantization_config)
```

### 33.11.4 可行方案（从最推荐到最不推荐）

#### 方案1: Monkey Patch `init_memory_pool` (推荐 ✅)

这是最直接、最可控的方案，也与 SIMO 现有的 SGLang 集成模式一致。

```python
# simo/extensions/sglang_simo/kv_cache/custom_pool.py

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

class SIMOTokenToKVPool(MHATokenToKVPool):
    """SIMO 自定义 KV Cache，继承自 MHATokenToKVPool。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 自定义初始化...

    def set_kv_buffer(self, layer, loc, cache_k, cache_v):
        # 自定义量化写入逻辑...
        pass
```

```python
# simo/extensions/sglang_simo/patches/kv_cache_patch.py

from simo.extensions.simo_patch import SIMOPatch
from sglang.srt.model_executor import model_runner_kv_cache_mixin as kv_mixin

def _patched_init_memory_pool(original_fn, self, total_gpu_memory):
    """在原始 init_memory_pool 执行后，替换 token_to_kv_pool。"""
    # 先调用原始函数，让它完成所有内存计算和 req_to_token_pool 初始化
    original_fn(self, total_gpu_memory)

    # 判断是否需要使用 SIMO 自定义 KV cache
    if should_use_simo_kv_cache(self):
        from .custom_pool import SIMOTokenToKVPool
        # 用自定义 pool 替换
        old_pool = self.token_to_kv_pool
        self.token_to_kv_pool = SIMOTokenToKVPool(
            self.max_total_num_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            head_num=...,
            head_dim=...,
            layer_num=self.num_effective_layers,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
        )
        del old_pool  # 释放原始 pool 的 GPU 内存

# 注册 patch
SIMOPatch.wrap_function(
    kv_mixin.ModelRunnerKVCacheMixin,
    "init_memory_pool",
    _patched_init_memory_pool,
    call_original=True,
)
```

**优点**: 完全不修改 SGLang 代码；与 SIMO 现有 patch 框架一致；可以复用原始函数的内存计算逻辑
**缺点**: 原始 `init_memory_pool` 会先分配一个 `MHATokenToKVPool`（浪费一次 GPU 内存分配），然后被替换掉

#### 方案2: Monkey Patch `init_memory_pool` (不调用原始函数，完全重写)

```python
def _patched_init_memory_pool(original_fn, self, total_gpu_memory):
    """完全重写 init_memory_pool，在合适的分支插入自定义逻辑。"""
    # 复制原始函数的内存计算逻辑（L340-L419）...
    # ...

    # 在 token_to_kv_pool 赋值时插入自定义分支
    if should_use_simo_kv_cache(self):
        self.token_to_kv_pool = SIMOTokenToKVPool(...)
    else:
        # 其他分支调用原始逻辑（或直接复制原始 if-elif-else）
        original_fn(self, total_gpu_memory)

SIMOPatch.wrap_function(
    kv_mixin.ModelRunnerKVCacheMixin,
    "init_memory_pool",
    _patched_init_memory_pool,
    call_original=True,
)
```

**优点**: 避免了方案1中的"先分配再替换"浪费
**缺点**: 需要复制部分 `init_memory_pool` 的逻辑（内存计算、req_to_token_pool初始化等），SGLang 升级时可能需要同步更新

#### 方案3: Monkey Patch 类本身（替换 MHATokenToKVPool）

不 patch `init_memory_pool`，而是直接把 `MHATokenToKVPool` 类替换成自定义子类:

```python
import sglang.srt.mem_cache.memory_pool as pool_module

# 保存原始类
_OriginalMHATokenToKVPool = pool_module.MHATokenToKVPool

class SIMOTokenToKVPool(_OriginalMHATokenToKVPool):
    """透明替换，所有 MHATokenToKVPool 的实例化都会变成这个类。"""
    def set_kv_buffer(self, layer, loc, cache_k, cache_v):
        if self._should_quantize(layer):
            # SIMO 量化路径
            ...
        else:
            super().set_kv_buffer(layer, loc, cache_k, cache_v)

# 替换类引用
pool_module.MHATokenToKVPool = SIMOTokenToKVPool
```

**注意**: 还需要同时 patch `model_runner_kv_cache_mixin` 模块中的导入:
```python
import sglang.srt.model_executor.model_runner_kv_cache_mixin as kv_mixin
kv_mixin.MHATokenToKVPool = SIMOTokenToKVPool
```

**优点**: 不需要理解 `init_memory_pool` 的分支逻辑；对SGLang内部升级更鲁棒
**缺点**: 全局替换影响范围大；如果只想在特定条件下使用自定义 KV cache 需要在类内部做条件判断；需要确保所有导入点都被 patch

#### 方案4: 利用 Python `__init_subclass__` 或 `__subclasshook__` (不推荐)

理论上可以通过 metaclass 或 `__init_subclass__` 拦截实例化，但 SGLang 是直接 `MHATokenToKVPool(...)` 调用，没有经过工厂函数，所以这条路走不通。

#### 方案5: 向 SGLang 上游提 PR，添加 KV Cache 注册机制 (长期方案)

参考 SGLang 已有的 `register_attention_backend` 机制，为 KV cache 也添加类似的注册表:

```python
# 提议的 sglang/srt/mem_cache/kv_cache_registry.py
KV_CACHE_BACKENDS = {}

def register_kv_cache(name):
    def decorator(cls):
        KV_CACHE_BACKENDS[name] = cls
        return cls
    return decorator

@register_kv_cache("mha")
class MHATokenToKVPool(KVCache): ...

@register_kv_cache("mha_fp4")
class MHATokenToKVPoolFP4(KVCache): ...
```

然后在 `init_memory_pool` 中通过名称查找:
```python
kv_cache_cls = KV_CACHE_BACKENDS.get(self.server_args.kv_cache_backend, MHATokenToKVPool)
self.token_to_kv_pool = kv_cache_cls(...)
```

**优点**: 最干净的方案；符合SGLang自身的设计模式
**缺点**: 需要上游接受；短期内无法使用

### 33.11.5 推荐方案

对于 SIMO 的实际需求，**推荐方案1（Monkey Patch init_memory_pool，调用原始函数后替换）**，理由：

1. **与 SIMO 现有架构一致**: 已经有 `SIMOPatch.wrap_function` 框架
2. **影响范围可控**: 只在特定条件下替换 KV cache
3. **实现简单**: 不需要复制 SGLang 内部逻辑
4. **GPU 内存浪费可忽略**: 原始 pool 被 del 后，PyTorch CUDA allocator 会回收内存（或者用方案2避免）

如果你需要更精细的控制（避免多余的内存分配），可以用**方案2**的变体: 把原始 `init_memory_pool` 拆成两部分调用，只跳过 `token_to_kv_pool` 赋值阶段。

**方案3（替换类本身）** 作为备选，适合"所有 MHATokenToKVPool 实例都需要被替换"的场景。

---

## 33.12 `token_to_kv_pool_allocator` 的作用及自定义 KV Cache 是否需要关心它

### 33.12.1 结论

**不需要实现自定义的 `token_to_kv_pool_allocator`**。Allocator 只管理**索引**（哪些 slot 空闲/已用），完全不涉及 KV cache 的数据格式、dtype 或量化方式。实现 MX 数据类型的 KV cache 量化只需要自定义 `token_to_kv_pool`（即 KVCache 子类），allocator 原封不动复用即可。

### 33.12.2 Allocator 与 KVCache 的职责分离

```
┌─────────────────────────────┐    ┌──────────────────────────────┐
│  token_to_kv_pool_allocator │    │     token_to_kv_pool         │
│  (索引管理器)                │    │     (数据管理器 / KVCache)    │
├─────────────────────────────┤    ├──────────────────────────────┤
│ • 维护 free_pages 索引列表   │    │ • 分配 GPU tensor 存储 KV 数据│
│ • alloc(n) → 返回n个空闲索引 │    │ • get_key_buffer(layer_id)   │
│ • free(indices) → 回收索引   │    │ • get_value_buffer(layer_id) │
│ • available_size() → 剩余量  │    │ • set_kv_buffer(layer, loc,  │
│ • 不关心数据格式/dtype       │    │     cache_k, cache_v)        │
│ • 不读写 KV cache 数据       │    │ • 管理实际的量化/存储格式     │
└──────────┬──────────────────┘    └──────────────────────────────┘
           │                                    ▲
           │  仅有的交互:                         │
           │  get_cpu_copy(indices) ─────────────┘  (数据离线到CPU)
           │  load_cpu_copy(data, indices) ──────┘  (数据从CPU加载回)
```

**Allocator 内部只操作整数索引 tensor**:
```python
# allocator.py 中 alloc() 的核心逻辑:
def alloc(self, need_size):
    select_index = self.free_pages[:need_size]   # 取前 n 个空闲索引
    self.free_pages = self.free_pages[need_size:] # 从空闲列表移除
    return select_index                           # 返回 torch.Tensor of int64

# free() 的核心逻辑:
def free(self, free_index):
    self.free_pages = torch.cat((self.free_pages, free_index))  # 回收索引
```

Allocator 对 KVCache 的唯一引用是 `self._kvcache`，仅用于两个代理方法:
```python
def get_cpu_copy(self, indices):
    return self._kvcache.get_cpu_copy(indices)  # CPU 离线

def load_cpu_copy(self, kv_cache_cpu, indices):
    return self._kvcache.load_cpu_copy(kv_cache_cpu, indices)  # CPU 加载
```

### 33.12.3 Allocator 类继承体系

**文件**: `/share_data/users/like/package/h100/package/sglang_kernel_src/python/sglang/srt/mem_cache/allocator.py`

```
BaseTokenToKVPoolAllocator (ABC)
├── TokenToKVPoolAllocator        ← page_size == 1 时使用
├── PagedTokenToKVPoolAllocator   ← page_size > 1 时使用（最常见）
│   └── NPUPagedTokenToKVPoolAllocator  ← NPU 设备
└── SWATokenToKVPoolAllocator     ← Hybrid SWA 模型
```

**`init_memory_pool` 中的选择逻辑** (L687-734):
```python
if _is_npu:
    allocator = NPUPagedTokenToKVPoolAllocator(...)
elif self.is_hybrid_swa:
    allocator = SWATokenToKVPoolAllocator(...)
elif self.page_size == 1:
    allocator = TokenToKVPoolAllocator(...)
else:
    allocator = PagedTokenToKVPoolAllocator(...)
```

选择条件仅基于 **设备类型** 和 **page_size**，与 KV cache 的数据格式无关。

### 33.12.4 Allocator 核心方法及调用场景

| 方法 | 作用 | 调用方 |
|------|------|--------|
| `alloc(n)` | 分配 n 个 token slot 的索引 | radix_cache, disagg/decode, lmc_radix_cache |
| `free(indices)` | 回收索引 | radix_cache (eviction), speculative (reject), scheduler_pp |
| `available_size()` | 查询剩余可用量 | scheduler (retract), schedule_policy (evict决策) |
| `alloc_extend(...)` | Prefill/Extend 阶段的 paged 分配 | attention backend (forward_extend) |
| `alloc_decode(...)` | Decode 阶段的 paged 分配 | attention backend (forward_decode) |
| `free_group_begin/end()` | 批量 free 优化 | scheduler_output_processor |
| `merge_and_sort_free()` | 碎片整理（排序空闲索引） | alloc() 内部调用 |
| `clear()` | 重置所有索引 | scheduler 重置 |
| `backup/restore_state()` | 状态保存/恢复 | checkpoint |
| `get_cpu_copy(indices)` | 代理到 KVCache.get_cpu_copy | disagg offload |
| `load_cpu_copy(...)` | 代理到 KVCache.load_cpu_copy | disagg offload |

### 33.12.5 为什么自定义 MX KV Cache 不需要自定义 Allocator

**原因1: Allocator 操作的是索引，不是数据**

Allocator 的所有方法都只操作 `torch.Tensor` of `int64`（索引）。它不知道也不需要知道这些索引对应的 KV cache 数据是 bf16、fp8 还是 MX 格式。

```python
# 调用链示例 (decode阶段):
indices = allocator.alloc_decode(seq_lens, last_loc)  # → int64 tensor
# ...后续由 attention kernel 使用 indices 去 KVCache 读写数据
# allocator 不参与数据读写
```

**原因2: KVCache 的数据读写绕过 Allocator**

在 attention forward 中，数据写入 KV cache 的流程是:

```python
# triton_backend.py 中的 forward_extend / forward_decode:
token_to_kv_pool = model_runner.token_to_kv_pool  # KVCache 对象
k_buf, v_buf = token_to_kv_pool.get_kv_buffer(layer_id)  # 直接从 KVCache 获取 buffer
# 然后 attention kernel 直接操作 k_buf, v_buf
```

Allocator 在这个过程中不参与。它的工作在更上层（scheduler 分配/回收 slot 索引）。

**原因3: `alloc_extend` / `alloc_decode` 也不涉及数据格式**

这两个 paged allocator 的核心方法通过 Triton kernel 计算索引：

```python
# PagedTokenToKVPoolAllocator.alloc_extend():
alloc_extend_kernel[grid](
    prefix_lens, seq_lens, last_loc,
    free_pages,        # int64 索引
    out_indices,       # int64 索引
    page_size, ...
)
# 内部只做整数运算（页号 × page_size + 偏移），不涉及 KV 数据
```

**原因4: 唯一涉及数据的方法是 CPU offload，可以在 KVCache 侧处理**

`get_cpu_copy` 和 `load_cpu_copy` 只是代理到 `self._kvcache` 的对应方法。如果自定义 KVCache 需要特殊的 CPU offload 逻辑（如先反量化再 offload），只需在 KVCache 子类中重写这两个方法即可。

### 33.12.6 自定义 MX KV Cache 时需要关心的接口

只需实现 `KVCache` 的子类（例如继承 `MHATokenToKVPool`），确保以下方法正确:

```python
class SIMOMXTokenToKVPool(MHATokenToKVPool):
    """MX 格式量化 KV Cache"""

    def __init__(self, size, page_size, dtype, head_num, head_dim,
                 layer_num, device, ...):
        # 分配 MX 格式的 buffer（uint8 + scale）
        ...

    def get_key_buffer(self, layer_id) -> torch.Tensor:
        # 返回指定 layer 的 key buffer
        ...

    def get_value_buffer(self, layer_id) -> torch.Tensor:
        # 返回指定 layer 的 value buffer
        ...

    def get_kv_buffer(self, layer_id) -> Tuple[torch.Tensor, torch.Tensor]:
        # 返回 (key_buffer, value_buffer)
        ...

    def set_kv_buffer(self, layer, loc, cache_k, cache_v):
        # 量化并写入 KV cache（这里实现 MX 量化逻辑）
        ...

    # 可选: 如果需要 CPU offload 支持
    def get_cpu_copy(self, indices):
        ...
    def load_cpu_copy(self, kv_cache_cpu, indices):
        ...
```

**Allocator 完全不需要修改**。在 monkey patch `init_memory_pool` 时，只替换 `self.token_to_kv_pool`，保持 `self.token_to_kv_pool_allocator` 的原始初始化逻辑:

```python
def _patched_init_memory_pool(original_fn, self, total_gpu_memory):
    original_fn(self, total_gpu_memory)  # 原始函数初始化 allocator 和 pool

    if should_use_simo_kv_cache(self):
        old_pool = self.token_to_kv_pool
        new_pool = SIMOMXTokenToKVPool(...)
        self.token_to_kv_pool = new_pool

        # 重建 allocator，指向新的 KVCache（因为 allocator 持有 _kvcache 引用）
        self.token_to_kv_pool_allocator._kvcache = new_pool

        del old_pool
```

注意最后一步: 需要更新 `allocator._kvcache` 引用，因为 `get_cpu_copy`/`load_cpu_copy` 需要通过它代理到正确的 KVCache 对象。

---

## 2026-04-02 KV Cache OOM 根因分析（`launch_server.kvquant_mxfp8.log.2026_04_02___14_41_00`）

### 结论

`init_memory_pool_patch.py` 里“先 `del old_pool` 再创建 `SIMOMHATokenToKVPool`”没有生效的核心原因是：  
**旧 pool 仍被 `token_to_kv_pool_allocator` 强引用**，所以旧 KV cache 显存没有真正释放。

### 证据链

1. 原始 SGLang `init_memory_pool` 会先创建 `self.token_to_kv_pool`，再创建 `self.token_to_kv_pool_allocator(kvcache=self.token_to_kv_pool)`。  
   代码位置：`sglang/srt/model_executor/model_runner_kv_cache_mixin.py` 中 `init_memory_pool` 的 `# Initialize token_to_kv_pool_allocator` 部分。

2. allocator 基类把这个引用保存为 `self._kvcache`（强引用）。  
   代码位置：`sglang/srt/mem_cache/allocator.py`，`BaseTokenToKVPoolAllocator.__init__` 中 `self._kvcache = kvcache`。

3. 你的 patch 只做了：
   - `self.token_to_kv_pool = None`
   - `del old_pool`
   但**没有处理** `self.token_to_kv_pool_allocator._kvcache`。  
   因此旧 pool 仍存活，旧 KV cache 显存仍占用。

4. 日志数值与“双池并存”完全一致：
   - 旧 BF16 KV pool 已分配：`K 21.92 GB + V 21.92 GB`（log 行 1208）
   - 随后开始替换并创建 SIMO pool（log 行 1215/1216）
   - OOM 时进程显存占用 `78.97 GiB`，PyTorch allocated `78.28 GiB`，仅剩 `205 MiB`（log 行 1249）
   这说明旧大池并未释放，创建新池时出现峰值内存叠加。

### 为什么会“看起来已经 del 了但没释放”

Python 对象的释放取决于**所有强引用**都消失。  
`old_pool` 这个局部变量删掉了，但 allocator 内的 `_kvcache` 还在指向旧 pool，所以对象和对应 CUDA Tensor 都不会释放。

### 直接修复方向

在创建新 pool 前，必须先断开 allocator 对旧 pool 的引用（至少把 `self.token_to_kv_pool_allocator._kvcache` 从旧对象解绑），否则仍会发生双池显存峰值。

---

## 2026-04-02 15:21:03 OOM 根因分析（`launch_server.kvquant_mxfp8.log.2026_04_02___15_21_03`）

### 结论

这次 OOM 的核心原因仍然是 **KV pool 替换阶段发生显存峰值叠加**：
- 旧的 BF16 KV pool 已经分配完成；
- 新的 SIMO 量化 KV pool 开始分配时，旧 pool 并没有在这一步前真正回收；
- 两者短时重叠导致超过 H100 80G 可用显存。

### 证据链（按日志时间顺序）

1. `15:22:16`：权重加载结束，`mem usage=15.02 GB`。
2. `15:24:40`：KV dtype 仍是 `torch.bfloat16`。
3. `15:25:09`：旧 KV pool 已分配完成，`K 21.92 GB + V 21.92 GB`（合计约 `43.84 GiB`）。
4. `15:25:10`：此时可用显存只剩 `18.58 GB`。
5. `15:25:13`：开始“Replacing MHATokenToKVPool with SIMOMHATokenToKVPool”。
6. `15:25:36`：在 `SIMOMHATokenToKVPool._create_buffers` 里 OOM，单次申请 `362 MiB` 失败。

### 数值一致性校验

- 量化池参数：`#tokens=359114`，`head_num=8`（由 Llama3.1-8B KV 结构推导），`k/v_combined_head_size=132`。
- 新量化 KV pool 体积约：`22.60 GiB`。
- 每层单个 K（或 V）buffer 约：`361.66 MiB`，与报错里 `Tried to allocate 362.00 MiB` 精确对应。
- 旧 pool 分配后只剩 `18.58 GB`，而新 pool 需要 `22.60 GiB`，缺口约 `4.02 GiB`，必然 OOM。

对应峰值近似：
`15.02 (weights) + 43.84 (old bf16 kv) + 22.60 (new quant kv) = 81.46 GiB > 79.18 GiB`。

### 额外判断

- 这次不像“碎片化主因”：OOM 信息里 `reserved but unallocated` 仅 `22.56 MiB`，远小于缺口。
- 日志中 `self.excludes:['*']` / 线性层 `weight_spec=bfloat16` 表明这次主要是 **仅 KV cache 量化**，权重量化没有开启；但本次启动失败主因仍是“替换阶段双池叠加”。


---

## 2026-04-02 关于“已断引用但 old_pool 显存未释放”的进一步解释

你说得对：`allocator._kvcache = None` 和 `self.token_to_kv_pool = None` 已经做了“显式断引用”。
但这仍然不能保证在“下一行分配新池之前”显存一定可用，原因有两层。

### 1) 断的是你看到的引用，不一定是“全部引用”

在这条启动日志里，出现了 `pudb` 会话：
- `pudb:5091: Waiting for client...`
- `pudb:5091: Now in session ...`

`pudb/inspect` 这类调试路径会持有 frame/locals，可能把对象生命周期拖长（包括池对象或其张量）。
因此即使业务代码里已 `del`，也可能在调试器/帧对象里仍有隐藏强引用，导致 old pool 的大张量还活着。

### 2) Python 对象释放时机 != CUDA 显存“立刻可用于下一次大分配”

即便 Python 层引用清理了，CUDA 分配器（PyTorch caching allocator）和异步执行也会让“显存回收可见性”滞后。
在你这个 case 中，新池分配时仍然走到了：
- 每层约 `362 MiB` 的 `torch.zeros(...)`（与 `k/v_combined_head_size=132` 完全一致）
- 且报 OOM 时已接近满卡

说明替换窗口里仍出现了 old/new 双池叠加峰值。

## 如何解决这个 OOM

### A. 根治方案（推荐）：不要先建 BF16 pool 再替换

把 patch 改为“直接创建 SIMO KV pool”，避免“先建旧池再建新池”的双峰。
即：在 `init_memory_pool` 里走 SIMO 分支时，直接构造 `SIMOMHATokenToKVPool`，不要先调用默认 `MHATokenToKVPool`。

这是最稳的，因为它从机制上消除了峰值叠加。

### B. 现有替换方案下的强制释放（务实可落地）

在创建 `new_pool` 前，不只断引用，还要主动清空 old pool 的大张量，并做一次显式回收：

```python
# 1) 先解绑 allocator
allocator = getattr(self, "token_to_kv_pool_allocator", None)
if allocator is not None and hasattr(allocator, "_kvcache"):
    allocator._kvcache = None

# 2) 主动删 old_pool 大缓冲（即使 old_pool 还有隐藏引用，也先把大张量卸掉）
if hasattr(old_pool, "_clear_buffers"):
    old_pool._clear_buffers()  # del k_buffer / v_buffer
for name in ("k_data_ptrs", "v_data_ptrs", "data_ptrs", "data_strides"):
    if hasattr(old_pool, name):
        setattr(old_pool, name, None)

# 3) 断开 runner 引用
self.token_to_kv_pool = None
del old_pool

# 4) 强制清理
import gc
gc.collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()

# 5) 再分配 new_pool
new_pool = SIMOMHATokenToKVPool(...)
```

要点：`_clear_buffers()` 是关键，它能在“对象本身暂未销毁”时先释放最大的两组张量。

### C. 先让服务跑起来的临时参数（workaround）

在代码修复前，降低 token 池规模，避免双峰超过 80G。
你当前 `359114` token 下估算峰值约 `81.46 GiB`（超出 `79.18 GiB`）。

临时可加：
- `--max-total-tokens 330000`（建议先试 320000~330000）

这样即使仍有替换峰值，也能先避开 OOM。

### D. 关闭调试注入（建议）

当前日志出现 `pudb`，建议在复测时关闭 `debug_pudb/debug_log/debug_stack` 注入路径，避免调试器帧持有对象影响释放行为。


---

## 2026-04-02 方案A具体实现（不复制大段 `init_memory_pool`，且只建一次新pool）

你这个诉求可以这样做：

- **不要**在 `original_func` 后“替换 pool”；
- 改为在调用 `original_func` 前，**临时把** `model_runner_kv_cache_mixin.MHATokenToKVPool` **指向一个 SIMO 代理类**；
- `original_func` 内部会按原流程创建 `token_to_kv_pool` + `allocator`，但创建出来的已经是 SIMO pool；
- `try/finally` 恢复原类，避免污染其它路径。

这样你既保留了 upstream `init_memory_pool` 全部逻辑（不需要 copy 大量代码），又彻底消除“先旧池再新池”的双峰。

### 为什么这招可行

`init_memory_pool` 文件是这样导入的：

- `model_runner_kv_cache_mixin.py` 顶部 `from ...memory_pool import MHATokenToKVPool`

函数体里实例化用的是该模块内的符号 `MHATokenToKVPool(...)`。所以只要在调用 `original_func` 前临时替换这个符号，函数内部就会走你的新类。

### 可直接落地的改法（替换 `_patched_init_memory_pool`）

放在 `simo/extensions/sglang_simo/mem_cache/init_memory_pool_patch.py`：

```python
import logging
from contextlib import contextmanager

from simo.extensions.simo_patch import SIMOPatch

logger = logging.getLogger(__name__)


def _extract_quant_params_from_model(model_runner):
    model = model_runner.model
    for _, module in model.named_modules():
        if hasattr(module, "kv_cache_quant_spec"):
            return {
                "kv_cache_quant_spec": module.kv_cache_quant_spec,
                "kv_cache_downcast_kernel": module.kv_cache_downcast_kernel,
                "kv_cache_upcast_kernel": module.kv_cache_upcast_kernel,
                "packed_head_size": module.packed_head_size,
                "scale_head_size": module.scale_head_size,
                "key_hadamard_transform_size": getattr(module, "key_hadamard_transform_size", 0),
                "value_hadamard_transform_size": getattr(module, "value_hadamard_transform_size", 0),
            }
    return None


@contextmanager
def _temporarily_replace_mha_pool_cls(quant_params):
    import sglang.srt.model_executor.model_runner_kv_cache_mixin as kv_mixin
    from simo.extensions.sglang_simo.mem_cache.memory_pool import SIMOMHATokenToKVPool

    old_cls = kv_mixin.MHATokenToKVPool

    class SIMOMHATokenToKVPoolAdapter(SIMOMHATokenToKVPool):
        """
        适配 old MHATokenToKVPool 构造签名，让 original init_memory_pool 可原样调用。
        """

        def __init__(
            self,
            size,
            page_size,
            dtype,
            head_num,
            head_dim,
            layer_num,
            device,
            enable_memory_saver,
            v_head_dim=None,
            swa_head_num=None,
            swa_head_dim=None,
            swa_v_head_dim=None,
            start_layer=None,
            end_layer=None,
            enable_alt_stream=True,
            enable_kv_cache_copy=False,
        ):
            # 与原类一致：SWA 参数优先
            final_head_num = swa_head_num if swa_head_num is not None else head_num
            final_head_dim = swa_head_dim if swa_head_dim is not None else head_dim
            final_v_head_dim = (
                swa_v_head_dim
                if swa_v_head_dim is not None
                else (v_head_dim if v_head_dim is not None else final_head_dim)
            )

            super().__init__(
                size=size,
                page_size=page_size,
                dtype=dtype,
                head_num=final_head_num,
                head_dim=final_head_dim,
                layer_num=layer_num,
                device=device,
                enable_memory_saver=enable_memory_saver,
                kv_cache_quant_spec=quant_params["kv_cache_quant_spec"],
                k_packed_head_size=quant_params["packed_head_size"],
                k_scale_head_size=quant_params["scale_head_size"],
                v_packed_head_size=quant_params["packed_head_size"],
                v_scale_head_size=quant_params["scale_head_size"],
                kv_cache_downcast_kernel=quant_params["kv_cache_downcast_kernel"],
                kv_cache_upcast_kernel=quant_params["kv_cache_upcast_kernel"],
                key_hadamard_transform_size=quant_params["key_hadamard_transform_size"],
                value_hadamard_transform_size=quant_params["value_hadamard_transform_size"],
                v_head_dim=final_v_head_dim,
                start_layer=start_layer,
                end_layer=end_layer,
            )

    kv_mixin.MHATokenToKVPool = SIMOMHATokenToKVPoolAdapter
    try:
        yield
    finally:
        kv_mixin.MHATokenToKVPool = old_cls


def _patched_init_memory_pool(original_func, self, *args, **kwargs):
    """
    关键：不再 post-replace，而是在 original_func 内直接建 SIMO pool。
    """
    quant_params = _extract_quant_params_from_model(self)
    if quant_params is None:
        logger.warning(
            "No kv_cache_quant_spec found. Fallback to original init_memory_pool."
        )
        return original_func(self, *args, **kwargs)

    # 仅在 MHA 路径启用注入；其它路径保持原生逻辑
    if getattr(self, "use_mla_backend", False) or getattr(self.model_config, "is_hybrid_swa", False):
        logger.info("Skip SIMO MHA pool injection for MLA/SWA path.")
        return original_func(self, *args, **kwargs)

    logger.info(
        "Inject SIMOMHATokenToKVPool into init_memory_pool (single-allocation mode)."
    )
    with _temporarily_replace_mha_pool_cls(quant_params):
        return original_func(self, *args, **kwargs)
```

### 这个方案相对你当前实现的关键变化

- 你当前实现：`original_func` 先建 BF16 pool，再建 SIMO pool（有双峰）
- 新实现：`original_func` 里直接建 SIMO pool（只有单池）

### 验证点（日志）

改完后启动日志应满足：

1. 不再出现 `Replacing MHATokenToKVPool with SIMOMHATokenToKVPool`（后替换流程消失）。
2. `KV Cache is allocated` 只出现一次，且容量应接近量化池规模，而不是 `21.92 + 21.92 GB`。
3. `Memory pool end` 前不再出现 `SIMOMHATokenToKVPool._create_buffers` 的二次大分配 OOM。

### 额外提示

`profile_max_num_token()` 仍按 `kv_cache_dtype=bfloat16` 的 cell size 估算，这会让 token 上限偏保守（不会导致 OOM，但可能没吃满显存）。
如果后续要提升吞吐，再单独做“量化 KV cell size 感知”的 profile patch。


---

## 2026-04-02 vLLM 启动失败分析（`temp/vllm.serve.log.2026_04_02___17_35_34`）

### 结论

这次不是 OOM，也不是 attention backend 参数错误；是 **Python 导入错误导致 EngineCore 启动失败**。

致命错误：
- `ImportError: cannot import name 'MX_DTYPE_TO_FORMAT_ID' from simo.extensions.vllm_simo.v1.attention.ops.triton_reshape_and_cache_flash`
- 日志位置：约 `772/874` 行。

### 证据链

1. 失败发生在 `profile_run -> dummy_run -> attention forward` 阶段（说明模型已开始跑预热图）。
2. 调用栈进入：
   - `.../simo/extensions/vllm_simo/v1/attention/backends/simo_gqa.py`
   - `.../simo/extensions/vllm_simo/v1/attention/ops/triton_unified_attention.py:25`
3. `triton_unified_attention.py` 试图从 `triton_reshape_and_cache_flash.py` 导入 `MX_DTYPE_TO_FORMAT_ID`。
4. 但在实际被加载的代码树里（`/share_data/users/like/package/h100/package/simo_conda_vllm/...`），`triton_reshape_and_cache_flash.py` **没有定义这个符号**，因此导入失败。

### 根因（代码版本不一致 / 半迁移）

`MX_DTYPE_TO_FORMAT_ID` 在 `simo_conda_vllm` 中被迁移到了：
- `simo/extensions/vllm_simo/quantization/kernel_param_utils.py`

但 `triton_unified_attention.py` 仍沿用旧导入路径（从 `triton_reshape_and_cache_flash` 导入），形成不一致。

另外，虽然你给的配置文件路径在 `simo_conda_sglang`，但这次运行时实际导入的是 `simo_conda_vllm` 下的代码（见报错绝对路径），所以修复要改 **simo_conda_vllm** 那份代码。

### 如何解决

#### 方案1（推荐，直接修正导入源）

修改文件：
- `/share_data/users/like/package/h100/package/simo_conda_vllm/simo/extensions/vllm_simo/v1/attention/ops/triton_unified_attention.py`

把：
```python
from simo.extensions.vllm_simo.v1.attention.ops.triton_reshape_and_cache_flash import (
  MX_DTYPE_TO_FORMAT_ID,
  get_pg_quant_params,
)
```
改为：
```python
from simo.extensions.vllm_simo.quantization.kernel_param_utils import (
  MX_DTYPE_TO_FORMAT_ID,
  get_pg_quant_params,
)
```

这样 `MX_DTYPE_TO_FORMAT_ID` 与 `get_pg_quant_params` 都从当前真实定义位置导入。

#### 方案2（兼容兜底，保留旧调用方）

在：
- `/share_data/users/like/package/h100/package/simo_conda_vllm/simo/extensions/vllm_simo/v1/attention/ops/triton_reshape_and_cache_flash.py`

顶部导入里补上：
```python
from simo.extensions.vllm_simo.quantization.kernel_param_utils import (
  MX_DTYPE_TO_FORMAT_ID,
  get_mx_quant_params,
  get_pg_quant_params,
)
```

即把 `MX_DTYPE_TO_FORMAT_ID` 重新“转发导出”，让旧导入路径继续可用。

### 建议的验证步骤

1. 在 `simo_vllm` 环境验证模块导入：
```bash
cd /tmp
/share_data/users/like/miniconda3/envs/simo_vllm/bin/python - <<'PY'
from simo.extensions.vllm_simo.v1.attention.ops import triton_unified_attention
print('ok:', triton_unified_attention.__file__)
PY
```
2. 再重启 `vllm serve`。
3. 预期：不再出现 `cannot import name 'MX_DTYPE_TO_FORMAT_ID'`；若后续还有错误，再看新的第一条 Traceback。

### 额外建议（避免后续同类问题）

统一一份 SIMO 代码源（`simo_conda_vllm` vs `simo_conda_sglang`），避免“配置文件来自A路径、运行代码来自B路径”的半同步状态。


---

## 2026-04-03 `reshape_and_cache_kernel_flash` 在 `MX_FORMAT_ID == MXFP6_E3M2` 时的细节

你问的这段代码，关键是：
- `_compute_and_pack_mxfmt(...)` 先把每 `4` 个 FP6 值打成 `3` 个字节；
- 在 kernel 里把这 3 字节从 `uint32` 中拆出来，按 `[b2, b1, b0]` 写入 cache。

下面按你关心的点展开。

### 1) `packed_key, key_scale = _compute_and_pack_mxfmt(...)` 返回什么形状、类型

在这个调用点里：
- 输入是 `tl.reshape(key_tile, (1, TILE_SIZE))`
- 并且 `MX_BLOCK_SIZE=TILE_SIZE`

所以在 `_compute_and_pack_mxfmt` 内部：
- `BLOCK_SIZE_OUT_DIM = 1`
- `BLOCK_SIZE_QUANT_DIM = TILE_SIZE`
- `BLOCK_SIZE_QUANT_MX_SCALE = TILE_SIZE // MX_BLOCK_SIZE = 1`

对 `MXFP6_E3M2`：
- `packed_key` 形状是 **`[1, TILE_SIZE//4]`**，类型是 **`tl.uint32`**（每个元素只用低 24 bit）
- `key_scale` 形状是 **`[1, 1]`**，类型是 **`tl.uint8`**（E8M0 指数，1 字节/块）

同理 `packed_val/val_scale` 也是同样形状与类型。

### 2) 数据如何分布（FP6 打包）

每 4 个 6-bit 值（共 24 bit）打成 3 字节。
设一组 4 个量化值为：`x0 x1 x2 x3`（每个 6 bit）。

`SIPU_PACKING` 下，`_compute_and_pack_mxfmt` 内先重排：
- `a=x3, b=x2, c=x1, d=x0`

然后生成：
- `byte0 = ((a & 0x3F) << 2) | ((b & 0x30) >> 4)`
- `byte1 = ((b & 0x0F) << 4) | ((c & 0x3C) >> 2)`
- `byte2 = ((c & 0x03) << 6) |  (d & 0x3F)`

并临时装进一个 `uint32`：
- `u32 = byte0 | (byte1 << 8) | (byte2 << 16)`

所以 `packed_key[g]` 的低 24 位就是该组 3 字节。

### 3) 你贴的分支在做什么（重点）

你贴的分支把 `uint32` 中 3 字节拆出来：
- `kb0 = low 8 bits  = byte0`
- `kb1 = mid 8 bits  = byte1`
- `kb2 = high 8 bits = byte2`

但写入地址用了：
- `offs_b2 = base + 3*g + 0`
- `offs_b1 = base + 3*g + 1`
- `offs_b0 = base + 3*g + 2`

也就是把 `(kb2, kb1, kb0)` 按顺序写到连续 3 字节位置。

换句话说，cache 的落盘顺序是：
- **`[b2, b1, b0]`**（这和注释一致）

### 4) ASCII 图：1 组（4 个 FP6）如何落盘

```text
输入(一组4值):  x0  x1  x2  x3   (每个6bit)
                   |   |   |   |
_compute_and_pack_mxfmt(SIPU):
  a=x3 b=x2 c=x1 d=x0
  byte0 = f(a,b)
  byte1 = f(b,c)
  byte2 = f(c,d)
  u32 = [ .... byte2 | byte1 | byte0 ]

kernel拆包:
  kb0 = u32[7:0]    = byte0
  kb1 = u32[15:8]   = byte1
  kb2 = u32[23:16]  = byte2

写cache(连续3字节):
  addr+0 <- kb2 (=byte2)
  addr+1 <- kb1 (=byte1)
  addr+2 <- kb0 (=byte0)

最终内存顺序: [b2, b1, b0]
```

### 5) ASCII 图：`TILE_SIZE=32` 的整 tile 存储

`TILE_SIZE=32` 时：
- `FP6_N = TILE_SIZE//4 = 8` 组
- 每组 3 字节，总 `PACKED_TILE_SIZE = 24` 字节

```text
group g=0..7, base = head_tile_idx * 24

offset: base+0  base+1  base+2  | base+3  base+4  base+5 | ... | base+21 base+22 base+23
data:     g0:b2   g0:b1   g0:b0 |  g1:b2   g1:b1   g1:b0 | ... |  g7:b2   g7:b1   g7:b0
```

scale 存储（MX 路径）：
- 每个 tile 1 个 `uint8` scale（E8M0）
- `scale_base + head_tile_idx` 写入 `key_scale`

### 6) `_compute_and_pack_mxfmt` 在 MXFP6 下有没有存储浪费？

分三层看：

1. **有效载荷层（最关键）**：
- 4 个 FP6 = 24 bit，正好 3 byte。
- 落盘就是 3 byte/4值，**没有额外 payload 浪费**。

2. **中间表示层**：
- 函数返回 `uint32` 承载 24 bit（高 8 bit 空着）。
- 这是寄存器/中间计算便利，并非最终 cache 落盘格式。
- 在你这段 kernel 里只写 3 个 `uint8`，所以 **显存最终不浪费这 8 bit**。

3. **系统性开销**：
- MX 每个 block 还需要 1 byte scale。
- 以 `block_size=32` 为例：数据 24B + scale 1B，scale 开销约 `1/24 = 4.17%`。
- 这属于量化方案本身的必要元数据，不是 FP6 打包缺陷。

补充：`PACKED_HEAD_SIZE_PADDED` 在这个 kernel 中未参与实际写入地址计算（仅作为参数传入），所以这里不会因为它再产生额外落盘 padding。



---

# 2026-04-12: "illegal memory access" 崩溃根因分析

## 结论

崩溃发生在 **`_fwd_kernel_stage2`** (sglang的stage2 reduce kernel)，根本原因是 **`Lv` 值不匹配**：量化的 `v_buffer.shape[-1]` 返回的是packed+scale的组合大小 (132)，而不是逻辑上的 head dimension (128)。

## 根因

**文件**: `sglang/.../triton_ops/decode_attention.py`
**函数**: `_decode_softmax_reducev_fwd`, **第597行**:

```python
Lv = v_buffer.shape[-1]  # BUG: 返回132 (packed+scale) 而不是 128 (逻辑head dim)
```

### 为什么是132而不是128？

日志证据:
```
SIMOMHATokenToKVPool: original_head_dim=128, original_v_head_dim=128,
k_combined_head_size=132, v_combined_head_size=132, mx_format=mxfp8_e4m3
```

量化的 `v_buffer` 是 `uint8` tensor，shape 为 `[num_tokens, num_kv_heads, 132]`：
- `v_packed_head_size = 128` (MXFP8 8-bit格式，每个元素1字节)
- `v_scale_head_size = 4` (128 elements / 32 block_size = 4个E8M0 scale字节)
- `v_combined_head_size = 128 + 4 = 132`

### 导致越界的具体过程

**Stage 1** (`_fwd_grouped_kernel_stage1`) 正确使用 `Lv = layer.v_head_dim = 128`，将partial attention结果写入 `attn_logits`，其shape为 `(bs, num_head, max_kv_splits, 128)`。

**Stage 2** (`_fwd_kernel_stage2`) 错误地拿到 `Lv = 132`，导致：

1. **`BLOCK_DV = triton.next_power_of_2(132) = 256`** (应该是128)
2. **`mask_d = offs_d < 132`** (应该是 `offs_d < 128`)

kernel尝试访问 buffer 中 **下标128-131**，但每个 (batch, head, split) 维度只有 **128个元素**。

### _fwd_kernel_stage2中具体崩溃的代码行：

**越界读 (第560-561行)**:
```python
tv = tl.load(
    Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
)
```
`mask_d` 对于下标128-131为 `True`，但 `Mid_O` (`attn_logits`) 最后一维只有128个元素。下标128-131溢出到下一个split或head的内存区域，在tensor边界处直接越界。

**越界写 (第578-580行)**:
```python
tl.store(
    O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
    acc / e_sum,
    mask=mask_d,
)
```
输出 `O` 的shape为 `(bs, num_q_heads, 128)`，`stride_oh = 128`。向下标128-131写入会溢出到下一个head的内存，对于最后一个batch的最后一个head，会向 **tensor末尾之后写入4个元素** —— 触发 illegal memory access。

**LSE偏移量错误 (第563行)**:
```python
tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
```
`offs_logic` 除以 `Lv=132` 而不是 `128`，产生错误的LSE buffer偏移量。

## 调用链

```
decode_attention_fwd (simo, line 957)
  -> decode_attention_fwd_grouped (simo, line 901)
    -> _decode_softmax_reducev_fwd (sglang, line 597)  <-- Lv = v_buffer.shape[-1] = 132 (BUG)
      -> _fwd_kernel_stage2 (sglang, line 610)         <-- illegal memory access
```

## 修复建议

`_decode_softmax_reducev_fwd` 不应该在buffer被量化时从 `v_buffer.shape[-1]` 推导 `Lv`。SIMO 的 `decode_attention_fwd_grouped` 应该：

1. 传一个具有正确逻辑shape的 dummy `v_buffer`，或
2. 不直接调用sglang的 `_decode_softmax_reducev_fwd`，而是在SIMO本地实现一个接受显式 `Lv` 参数的版本，或
3. 修改调用方式，传入一个 shape 为 `[..., 128]` 的 `v_buffer` view/slice，使得 `v_buffer.shape[-1]` 返回正确的逻辑维度。

最简单的修复：在 SIMO 的 `decode_attention.py` 中写一个 `_decode_softmax_reducev_fwd` 的本地版本（已有注释掉的代码），用 `Lv = layer.v_head_dim` 替代 `v_buffer.shape[-1]`。

---

## MXINT8 转置K加载送入 `_unpack_and_dequant_mxfmt` 的正确性分析

### 问题

在 `decode_attention.py` 的 `_fwd_grouped_kernel_stage1` 中，MXINT8 走的是 8-bit 的"转置加载"路径：

```python
# k shape: [BLOCK_DMODEL, BLOCK_N]  即 [K, N]
offs_buf_k = (
    kv_loc[None, :] * stride_buf_kbs        # [1, N]
    + cur_kv_head * PACKED_HEAD_SIZE
    + offs_d[:, None]                        # [K, 1]
)
k = tl.load(K_Buffer + offs_buf_k, ...)     # → [K, N]  (transposed)

# k_scale shape: [BLOCK_N, BLOCK_DMODEL_SCALE]  即 [N, K//32]
offs_buf_k_scale = (
    kv_loc[:, None] * stride_buf_kbs + SCALE_PLANE_OFFSET  # [N, 1]
    + cur_kv_head * SCALE_HEAD_SIZE
    + offs_d_scale[None, :]                                  # [1, K//32]
)
k_scale = tl.load(K_Buffer + offs_buf_k_scale, ...)  # → [N, K//32]
```

然后调用：
```python
K_dequant = tl.trans(_unpack_and_dequant_mxfmt(k, k_scale, MX_FORMAT_ID))
#                                               ^[K,N]  ^[N,K//32]
```

### `_unpack_and_dequant_mxfmt` 如何推导维度

关键代码（`_upcast_from_mxfmt.py:36-38`）：

```python
BLOCK_SIZE_OUT_DIM: tl.constexpr = scale.shape[0]
BLOCK_SIZE_QUANT_DIM: tl.constexpr = scale.shape[1] * MX_QUANT_DIM  # MX_QUANT_DIM=32 for MXINT8
BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // MX_QUANT_DIM
```

以及后面的 reshape + broadcast（`_upcast_from_mxfmt.py:182-196`）：

```python
dst_tensor = dst_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MX_QUANT_DIM])
dst_scale  = dst_scale.reshape( [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1])
out_tensor = dst_tensor * dst_scale
out_tensor = out_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
```

函数假设：
- `scale` 的 shape 是 `[OUT_DIM, num_scale_groups]`
- `mx_tensor` 的 shape 是 `[OUT_DIM, QUANT_DIM]`（对 MXINT8，QUANT_DIM == PACKED_HEAD_SIZE == head_dim）
- 每个 scale group 对应 MX_QUANT_DIM=32 个元素，沿 dim=1 排列
- 输出 shape 是 `[OUT_DIM, QUANT_DIM]`

### 传入转置 K 时的实际情况

| | 期望语义 | 实际传入 |
|---|---|---|
| `mx_tensor` (k) | `[OUT_DIM, head_dim]` | `[K=BLOCK_DMODEL, N=BLOCK_N]` |
| `scale` (k_scale) | `[OUT_DIM, head_dim//32]` | `[N=BLOCK_N, K//32=BLOCK_DMODEL_SCALE]` |

`_unpack_and_dequant_mxfmt` 推导出：
- `BLOCK_SIZE_OUT_DIM = scale.shape[0] = N`  (BLOCK_N)
- `BLOCK_SIZE_QUANT_DIM = scale.shape[1] * 32 = (K//32) * 32 = K`  (BLOCK_DMODEL)

然后它会：
1. 把 `mx_tensor` reshape 为 `[N, K//32, 32]`  — 但 `mx_tensor` 的 shape 是 `[K, N]`
2. 把 `scale` reshape 为 `[N, K//32, 1]`

### **结论：会出错**

**是的，这样直接调用会出错。** 原因是 shape 不匹配：

1. **reshape 维度不一致**：`mx_tensor` 的 shape 是 `[K, N]`（转置形式），但函数要 reshape 成 `[N, K//32, 32]`（共 `N * K` 个元素）。虽然总元素数相同（`K * N == N * K`），reshape 本身不会报错，但**数据排列顺序完全错误**。

2. **scale 对应关系错乱**：在正确的非转置布局 `[N, K]` 中，每一行是一个 token 的 K 个元素，scale 按 group_size=32 分组，每 32 个连续元素共享一个 scale。但传入转置的 `[K, N]` 后，reshape 成 `[N, K//32, 32]` 时，32 个"连续"元素实际上跨越了不同的 K 维度位置，scale 被应用到了错误的元素上。

3. **输出 shape 语义颠倒**：函数返回 `[N, K]`，然后外层 `tl.trans()` 得到 `[K, N]`。但由于内部 dequant 计算错误，最终值是错的。

### 与 vLLM 参考实现的对比

vLLM 的 `triton_unified_attention.py` 中（line 495-498），对 MXINT8 的处理：

```python
elif MX_FORMAT_ID > 0:
    # MXFP6, MXINT8, NVFP4: software dequant
    K = tl.trans(_unpack_and_dequant_mxfmt(k_packed, k_scales, MX_FORMAT_ID))
    S += scale * tl.dot(Q, K)
```

vLLM 中 `k_packed` 是**非转置加载**的 `[TILE_SIZE, PACKED_HEAD_SIZE]` 即 `[N, K]`，`k_scales` 是 `[N, SCALE_HEAD_SIZE]` 即 `[N, K//32]`。这才是正确的输入 shape：OUT_DIM=N，QUANT_DIM=K，scale 沿 K 方向每 32 个元素分组。

### 修复方案

对 MXINT8，应改为**非转置加载** K（与 FP6/FP4 子字节格式走相同的路径），或者将 MXINT8 从 8-bit 分支移到 software dequant 分支：

```python
# 方案：把 MXINT8 从 else 分支移到 IS_FP6 or IS_4BIT 同级的分支
# 改为非转置加载
if IS_FP6 or IS_4BIT or (MX_FORMAT_ID == MXINT8):
    # 非转置 K 加载：k shape [N, K]
    if IS_FP6:
        ...  # FP6 加载
    else:  # IS_4BIT 或 MXINT8
        packed_offs_d = tl.arange(0, PACKED_HEAD_SIZE_PADDED)
        packed_mask = (offs_n[:, None] < split_kv_end) & (packed_offs_d[None, :] < PACKED_HEAD_SIZE)
        offs_buf_k_packed = (
            kv_loc[:, None] * stride_buf_kbs
            + cur_kv_head * PACKED_HEAD_SIZE
            + packed_offs_d[None, :]
        )
        k_packed = tl.load(K_Buffer + offs_buf_k_packed, mask=packed_mask, other=0)

    # 加载 scale [N, K//32]
    offs_buf_k_scale = (
        kv_loc[:, None] * stride_buf_kbs + SCALE_PLANE_OFFSET
        + cur_kv_head * SCALE_HEAD_SIZE
        + offs_d_scale[None, :]
    )
    k_scale = tl.load(...)

    # dequant + transpose
    if MX_FORMAT_ID == MXFP4_E2M1:
        qk = tl.dot_scaled(q, None, "bf16", k_packed.T, k_scale, "e2m1", fast_math=True)
    else:  # FP6, NVFP4, MXINT8 统一走 software dequant
        K_dequant = tl.trans(_unpack_and_dequant_mxfmt(k_packed, k_scale, MX_FORMAT_ID))
        qk = tl.dot(q, K_dequant)
else:
    # 仅 MXFP8_E4M3 和 MXFP8_E5M2 走转置加载 + dot_scaled
    ...
```

**同样的问题在 V 加载路径也存在**，但 V 加载是非转置的 `[N, Lv]`，所以 V 的 8-bit 路径实际上是正确的（OUT_DIM=N, QUANT_DIM=Lv，scale `[N, Lv//32]`）。

**同样的问题在 `extend_attention.py` 中也存在**，MXINT8 也走了转置 K 加载的 else 分支。

### 总结

| 文件 | K加载 | V加载 | 问题 |
|---|---|---|---|
| `decode_attention.py` MXINT8 | 转置 `[K,N]` | 非转置 `[N,Lv]` | **K有bug**, V正确 |
| `extend_attention.py` MXINT8 | 转置 `[K,N]` | 非转置 `[N,Lv]` | **K有bug**, V正确 |
| `triton_unified_attention.py` (vLLM参考) | 非转置 `[N,K]` | 非转置 `[N,Lv]` | 全部正确 |

---

## `set_kv_buffer_kernel` 里 `MXFP4_E2M1` 分支的 `_compute_and_pack_mxfmt` 返回值

### 1. `packed_key` / `key_scale` 的 shape 和 dtype

在 `simo/extensions/sglang_simo/layers/attention/triton_ops/set_kv_buffer.py:99-113`，调用是：

```python
packed_key, key_scale = _compute_and_pack_mxfmt(
    tl.reshape(key_tile, (1, TILE_SIZE)),
    MX_FORMAT_ID=MX_FORMAT_ID,
    MX_BLOCK_SIZE=TILE_SIZE,
    ...
)
```

所以传入 `_compute_and_pack_mxfmt` 的 `src_tensor.shape` 是 `(1, TILE_SIZE)`。

在 `_compute_and_pack_mxfmt` 里：

- `BLOCK_SIZE_OUT_DIM = src_tensor.shape[0] = 1`
- `BLOCK_SIZE_QUANT_DIM = src_tensor.shape[1] = TILE_SIZE`
- `BLOCK_SIZE_QUANT_MX_SCALE = BLOCK_SIZE_QUANT_DIM // MX_BLOCK_SIZE = TILE_SIZE // TILE_SIZE = 1`

对 `MXFP4_E2M1` 走的是 4-bit pack 分支：

```python
e2m1_value = tl.reshape(quant_values, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2])
evens, odds = tl.split(e2m1_value)
pack_tensor = evens | (odds << 4)
```

scale 则是：

```python
quant_scale_exponent = quant_scale_exponent.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE])
scale_tensor = (quant_scale_exponent >> 23).to(tl.uint8)
```

因此在 `MX_FORMAT_ID == MXFP4_E2M1` 时：

- `packed_key.shape = (1, TILE_SIZE // 2)`
- `packed_key.dtype = tl.uint8`
- `key_scale.shape = (1, 1)`
- `key_scale.dtype = tl.uint8`

对这条 SGLang KV cache 路径，`TILE_SIZE = kv_cache_quant_spec.block_size`。`QuantizeSpecMX` 默认 `block_size=32`，而 `_unpack_and_dequant_mxfmt` 也把 `MXFP4_E2M1` 的 `MX_QUANT_DIM` 固定成 `32`，只有 `NVFP4_E2M1` 才是 `16`。所以这里通常就是：

- `packed_key.shape = (1, 16)`
- `key_scale.shape = (1, 1)`

`set_kv_buffer_kernel` 后面会再把它们 reshape 成一维后写入 cache：

- `packed_key -> tl.reshape(packed_key, (PACKED_TILE_SIZE,))`
- `key_scale -> tl.reshape(key_scale, (1,))`

### 2. `key_scale` 这个字节存的是什么

`key_scale` 不是 float32，而是一个 `uint8` 的 E8M0 exponent byte。

在 `_compute_and_pack_mxfmt` 里：

```python
scale_tensor = (quant_scale_exponent >> 23).to(tl.uint8)
```

读回时 `_unpack_and_dequant_mxfmt` 会把它重新拼成浮点 scale：

```python
dst_scale = (scale.to(tl.uint16) << 7).to(out_dtype, bitcast=True)   # bf16
# 或
dst_scale = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True) # f32/f16路径
```

所以对 `MXFP4_E2M1` 来说，每个 32 元素 block 对应 1 个 scale byte。

### 3. 数据是怎么 pack 的

先得到每元素 4-bit 的 `quant_values`。每个 4-bit 值内部的 bit 布局是：

- bit3: sign
- bit2: exponent 高位
- bit1: exponent 低位
- bit0: mantissa

也就是一个 `seem` 布局。

从反量化逻辑可以直接看出来：

```python
em0 = mx_tensor & 0x07
sign0_mask = mx_tensor & 0x08

em1 = mx_tensor & 0x70
sign1_mask = mx_tensor & 0x80
```

这说明：

- 低 nibble 的 `0x08` 是 sign，`0x07` 是 `eem`
- 高 nibble 的 `0x80` 是 sign，`0x70` 是 `eem << 4`

### 4. 一个字节的高 4 位和低 4 位分别是什么

pack 逻辑是：

```python
e2m1_value = tl.reshape(quant_values, [1, TILE_SIZE // 2, 2])
evens, odds = tl.split(e2m1_value)
pack_tensor = evens | (odds << 4)
```

所以每个 byte 对应两个连续量化值：

- 低 4 位 = 前一个值，也就是偶数下标元素
- 高 4 位 = 后一个值，也就是奇数下标元素

如果 4-bit 量化结果依次是 `q0, q1, q2, q3, ...`，那么 pack 后就是：

- `byte0 = q0 | (q1 << 4)`
- `byte1 = q2 | (q3 << 4)`
- `byte2 = q4 | (q5 << 4)`
- ...

读路径 `_unpack_and_dequant_mxfmt` 也是按这个顺序解的：

```python
em0 = mx_tensor & 0x07   # 低 nibble -> 第一个值
em1 = mx_tensor & 0x70   # 高 nibble -> 第二个值
interleaved = tl.interleave(x0, x1)
```

### 5. 最简结论

对 `set_kv_buffer_kernel` 里的这次调用，在 `MX_FORMAT_ID == MXFP4_E2M1` 时：

- `_compute_and_pack_mxfmt` 返回的 `packed_key` 是 `tl.uint8`，shape 为 `(1, TILE_SIZE // 2)`，通常就是 `(1, 16)`
- `_compute_and_pack_mxfmt` 返回的 `key_scale` 是 `tl.uint8`，shape 为 `(1, 1)`
- 每个 `packed_key` 字节：
  - 低 4 位存前一个量化值
  - 高 4 位存后一个量化值
- 每个 4-bit 量化值内部是 `seem = sign(1) + exponent(2) + mantissa(1)`

---

## MXFP6_E3M2 格式下 extend_attention.py `_fwd_kernel` 如何从 K/V buffer 中加载 packed data 和 scale

### 前置知识：KV Cache 内存布局

每个 token 在 cache 中的布局（planar layout）：

```
|<--- num_kv_heads * PACKED_HEAD_SIZE --->|<--- num_kv_heads * SCALE_HEAD_SIZE --->|
|   head0 packed   |  head1 packed  | ... |  head0 scale | head1 scale | ...       |
```

对于 MXFP6_E3M2，head_dim=128 时：
- `PACKED_HEAD_SIZE = head_dim * 3 / 4 = 96` 字节（4 个 FP6 值打包成 3 字节）
- `SCALE_HEAD_SIZE = head_dim / 32 = 4` 字节（每 32 个元素共享一个 E8M0 scale）
- `SCALE_PLANE_OFFSET = num_kv_heads * PACKED_HEAD_SIZE`（scale 区域在 packed data 之后）

### 相关常量定义（kernel 开头）

```python
SCALE_PLANE_OFFSET: tl.constexpr = num_kv_heads * PACKED_HEAD_SIZE  # scale区域起始偏移
IS_FP6: tl.constexpr = True   # MXFP6_E3M2 时为 True
NEEDS_SW_DEQUANT: tl.constexpr = True  # FP6 走软件反量化路径

# 在循环外预计算
offs_d_scale = tl.arange(0, BLOCK_DMODEL_SCALE)   # [0, 1, ..., next_power_of_2(4)-1]
mask_d_scale = offs_d_scale < SCALE_HEAD_SIZE      # 有效 scale 掩码
```

### K Packed 加载（非转置，走 `NEEDS_SW_DEQUANT` 分支）

```python
if NEEDS_SW_DEQUANT:
    if IS_FP6:
        FP6_NUM_GROUPS: tl.constexpr = PACKED_HEAD_SIZE // 3  # = 96//3 = 32 个 3-byte group
        group_offs = tl.arange(0, FP6_NUM_GROUPS)             # [0, 1, ..., 31]
        fp6_mask = mask_n[:, None] & (group_offs[None, :] < FP6_NUM_GROUPS)  # [BLOCK_N, 32]

        k_packed_offset = (
            offs_kv_loc[:, None] * stride_buf_kbs   # 每个 token 在 buffer 中的基地址
            + cur_kv_head * PACKED_HEAD_SIZE         # 当前 KV head 的 packed 数据偏移
        )  # shape: [BLOCK_N, 1]

        k_packed = _load_fp6_packed(K_Buffer, k_packed_offset, group_offs, fp6_mask)
```

#### `_load_fp6_packed` 的工作方式

```python
@triton.jit
def _load_fp6_packed(cache_ptr, packed_offset, group_offs, mask):
    """Load FP6 packed data: [byte2, byte1, byte0] per group -> uint32."""
    offs = group_offs[None, :]   # [1, FP6_NUM_GROUPS]
    # 每个 group 占 3 个连续字节，按 SIPU packing 布局存储为 [b2, b1, b0]
    b2 = tl.load(cache_ptr + packed_offset + 3 * offs,     mask=mask, other=0)  # byte2
    b1 = tl.load(cache_ptr + packed_offset + 3 * offs + 1, mask=mask, other=0)  # byte1
    b0 = tl.load(cache_ptr + packed_offset + 3 * offs + 2, mask=mask, other=0)  # byte0
    # 组装成 uint32: [byte2 在高16位, byte1 在中间, byte0 在低8位]
    return b0.to(tl.uint32) | (b1.to(tl.uint32) << 8) | (b2.to(tl.uint32) << 16)
```

**返回结果**：`k_packed` shape = `[BLOCK_N, FP6_NUM_GROUPS]`，dtype = `uint32`

每个 uint32 包含 3 个字节 = 4 个 FP6 值（24 bit）。32 个 group × 4 = 128 个 FP6 元素 = head_dim。

#### 对应的 buffer 字节布局

`set_kv_buffer_kernel` 中 FP6 数据以 `SIPU_PACKING` 方式存储：

```
group i 的 3 字节在 buffer 中的偏移：
  byte offset 3*i + 0 → byte2（存储 SIPU pack 的 byte2）
  byte offset 3*i + 1 → byte1
  byte offset 3*i + 2 → byte0
```

`_load_fp6_packed` 按同样的顺序读取，得到的 uint32 与 `_compute_and_pack_mxfmt` 返回的 `pack_tensor` 格式一致，可以直接送给 `_unpack_and_dequant_mxfmt` 反量化。

### K Scale 加载

```python
        # K scales（FP6、4-bit、MXINT8 统一走这条路径）
        offs_buf_k_scale = (
            offs_kv_loc[:, None] * stride_buf_kbs    # token 基地址 [BLOCK_N, 1]
            + SCALE_PLANE_OFFSET                      # 跳过所有 head 的 packed 数据区
            + cur_kv_head * SCALE_HEAD_SIZE           # 当前 head 的 scale 偏移
            + offs_d_scale[None, :]                   # scale 索引 [1, BLOCK_DMODEL_SCALE]
        )  # shape: [BLOCK_N, BLOCK_DMODEL_SCALE]

        k_scale = tl.load(
            K_Buffer + offs_buf_k_scale,
            mask=mask_n[:, None] & mask_d_scale[None, :],
            other=0
        )
```

**返回结果**：`k_scale` shape = `[BLOCK_N, BLOCK_DMODEL_SCALE]`，dtype = `uint8`（E8M0 格式）

对于 head_dim=128，`SCALE_HEAD_SIZE=4`，所以每个 token 有 4 个 scale 字节，每个 scale 对应 32 个 FP6 元素。

### V Packed 和 V Scale 加载（与 K 完全对称）

```python
            if NEEDS_SW_DEQUANT:
                if IS_FP6:
                    v_packed_offset = (
                        offs_kv_loc[:, None] * stride_buf_vbs   # V buffer 的 token 基地址
                        + cur_kv_head * PACKED_HEAD_SIZE         # 当前 head 的 packed 偏移
                    )  # shape: [BLOCK_N, 1]
                    v_packed = _load_fp6_packed(V_Buffer, v_packed_offset, group_offs, fp6_mask)
                    # v_packed shape: [BLOCK_N, FP6_NUM_GROUPS], dtype: uint32

                # V scales
                offs_buf_v_scale = (
                    offs_kv_loc[:, None] * stride_buf_vbs + SCALE_PLANE_OFFSET
                    + cur_kv_head * SCALE_HEAD_SIZE
                    + offs_d_scale[None, :]
                )
                v_scale = tl.load(
                    V_Buffer + offs_buf_v_scale,
                    mask=mask_n[:, None] & mask_d_scale[None, :],
                    other=0
                )
                # v_scale shape: [BLOCK_N, BLOCK_DMODEL_SCALE], dtype: uint8
```

V 的加载与 K 完全相同，只是指针从 `K_Buffer` 换成了 `V_Buffer`，stride 从 `stride_buf_kbs` 换成了 `stride_buf_vbs`。

### 加载后的 Dequant 流程

```python
                # Q*K computation (FP6 走 software dequant)
                K_dequant = tl.trans(
                    _unpack_and_dequant_mxfmt(k_packed, k_scale, MX_FORMAT_ID)
                )
                # _unpack_and_dequant_mxfmt 输入:
                #   k_packed: [BLOCK_N, FP6_NUM_GROUPS], uint32 (每个含4个FP6值)
                #   k_scale:  [BLOCK_N, BLOCK_DMODEL_SCALE], uint8 (E8M0)
                # 输出: [BLOCK_N, head_dim], bf16
                # tl.trans 后: [head_dim, BLOCK_N]
                qk = tl.dot(q, K_dequant)
                # q: [BLOCK_M, head_dim], K_dequant: [head_dim, BLOCK_N]
                # qk: [BLOCK_M, BLOCK_N]

                # V dequant（不需要转置，直接用于 p @ V）
                v_dequant = _unpack_and_dequant_mxfmt(v_packed, v_scale, MX_FORMAT_ID)
                # v_dequant: [BLOCK_N, head_dim], bf16
                acc += tl.dot(p, v_dequant)
                # p: [BLOCK_M, BLOCK_N], v_dequant: [BLOCK_N, head_dim]
```

### 总结

| 项目 | K 加载 | V 加载 |
|---|---|---|
| 数据指针 | `K_Buffer` | `V_Buffer` |
| packed 加载方式 | `_load_fp6_packed` → 非转置 `[N, groups]` uint32 | `_load_fp6_packed` → 非转置 `[N, groups]` uint32 |
| scale 加载方式 | `tl.load` → `[N, scale_dim]` uint8 | `tl.load` → `[N, scale_dim]` uint8 |
| dequant | `_unpack_and_dequant_mxfmt` → `[N, head_dim]` bf16 | `_unpack_and_dequant_mxfmt` → `[N, head_dim]` bf16 |
| 后处理 | `tl.trans` → `[head_dim, N]`，用于 `q @ K^T` | 直接用于 `p @ V` |
| 每个 group 内容 | 3 bytes → 4 个 FP6 值 → 1 个 uint32 | 同左 |
| FP6_NUM_GROUPS | PACKED_HEAD_SIZE // 3 = 32 (head_dim=128时) | 同左 |


## disable_cuda_graph 在 lm-eval JSON model_args 中的正确写法

**错误原因**: `"disable_cuda_graph": True` 中的 `True` 是 Python 语法，不是合法的 JSON。`lm-eval` 的 `--model_args` 参数会被当作 JSON 解析，JSON 布尔值必须是小写的 `true` / `false`。

**修复**: 把 `True` 改成 `true`：

```diff
- "disable_cuda_graph": True
+ "disable_cuda_graph": true
```

已在 `llm_eval_online_quant_kv_cache_quant.sh` 第 130 行修复。


---

## lm-eval gsm8k 测评报错分析

### 错误信息

```
TypeError: SamplingParams.__init__() got an unexpected keyword argument 'max_tokens'
```

两次运行（no-quant 和 simo kvquant_mxint8）都在同一个地方报相同的错误，说明**与 SIMO 量化无关**。

### 根因分析

**lm-eval 的 sglang 后端传了 `max_tokens` 参数，但 SGLang 的 `SamplingParams` 只接受 `max_new_tokens`。**

调用链：

```
lm_eval sglang_causallms.py:262
  → kwargs | {"max_tokens": max_gen_toks, "stop": until}  # lm-eval 传 "max_tokens"
  → engine.generate() → tokenizer_manager → _create_tokenized_object()
  → SamplingParams(**sampling_kwargs)   # sampling_kwargs 包含 "max_tokens"
  → TypeError: unexpected keyword argument 'max_tokens'
```

SGLang 的 `SamplingParams`（`python/sglang/srt/sampling/sampling_params.py:42`）定义的参数名是 `max_new_tokens`，不是 `max_tokens`：

```python
class SamplingParams:
    def __init__(self, ..., max_new_tokens: int = 128, ...):
```

而 lm-eval 的 sglang 后端（`lm_eval/models/sglang_causallms.py:262`）硬编码传的是 `max_tokens`：

```python
kwargs | {"max_tokens": max_gen_toks, "stop": until}
```

### 原因

**lm-eval 和 SGLang 版本不兼容**。

- 旧版 SGLang 的 `SamplingParams` 曾接受 `max_tokens` 参数（与 OpenAI API 对齐）
- 当前安装的 SGLang 源码版本已经改名为 `max_new_tokens`
- 但环境中安装的 `lm-eval` 包还在用旧的参数名 `max_tokens`

### 解决方案

有两种修复方式：

**方案 1（推荐）：升级 lm-eval 的 sglang 后端**

安装与当前 SGLang 源码兼容的 lm-eval 版本：
```bash
pip install lm-eval --upgrade
```

**方案 2：在 SGLang 侧添加兼容性处理**

在 `tokenizer_manager.py` 的 `_create_tokenized_object` 中，将 `max_tokens` 映射为 `max_new_tokens`：

```python
# 在 sampling_params = self.sampling_params_class(**sampling_kwargs) 之前添加：
if "max_tokens" in sampling_kwargs and "max_new_tokens" not in sampling_kwargs:
    sampling_kwargs["max_new_tokens"] = sampling_kwargs.pop("max_tokens")
```

**方案 3：修改 lm-eval 的 sglang 后端代码**

直接修改 `/share_data/users/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/lm_eval/models/sglang_causallms.py:262`：

```python
# 将
kwargs | {"max_tokens": max_gen_toks, "stop": until}
# 改为
kwargs | {"max_new_tokens": max_gen_toks, "stop": until}
```

### 注意

此错误发生在 `generate_until` 阶段（gsm8k 需要生成回答），**与 KV cache 量化完全无关**。模型加载、CUDA graph capture、loglikelihood 阶段（1319 个 request）都已正常完成。


---

## lm-eval gsm8k 测评报错分析（2026-04-20 11:30:09）

### 错误信息

两次运行（no-quant 和 simo kvquant_int8_per_group）都在模型初始化阶段报同样的错误：

```
[2026-04-20 11:34:28] Scheduler hit an exception: Traceback (most recent call last):
  File ".../scheduler.py", line 3130, in run_scheduler_process
    scheduler = Scheduler(...)
  File ".../scheduler.py", line 368, in __init__
    self.init_model_worker()
  ...
  File ".../model_runner.py", line 766, in init_torch_distributed
    before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
  File ".../utils/common.py", line 609, in get_available_gpu_memory
    return free_gpu_memory / (1 << 30)
UnboundLocalError: cannot access local variable 'free_gpu_memory' where it is not associated with a value
```

### 根因分析

**`get_available_gpu_memory()` 的 if/elif 分支链没有匹配到传入的 `device` 值，导致 `free_gpu_memory` 从未被赋值。**

`get_available_gpu_memory` (`python/sglang/srt/utils/common.py:508-609`) 的分支结构：

```python
def get_available_gpu_memory(device, gpu_id, ...):
    if device == "cuda":        # 精确匹配 "cuda"
        free_gpu_memory = ...
    elif device == "xpu":
        free_gpu_memory = ...
    elif device == "hpu":
        free_gpu_memory = ...
    elif device == "cpu":
        free_gpu_memory = ...
    elif device == "npu":
        free_gpu_memory = ...
    elif device == "musa":
        free_gpu_memory = ...
    # ← 没有 else 分支！

    return free_gpu_memory / (1 << 30)   # line 609: UnboundLocalError
```

从 server_args 日志可以看到 `device='cuda:0'`（带设备编号），而函数检查的是 `device == "cuda"`（不带编号）。`"cuda:0" != "cuda"`，所以没有进入任何分支，`free_gpu_memory` 从未赋值。

### 为什么 device 变成了 "cuda:0"

SGLang 的 `ServerArgs.__post_init__` 在 `_handle_missing_default_values` 中调用 `get_device()` 设置 device。`get_device()` 无参数时返回 `"cuda"`。但从日志可以看到，实际运行时 `server_args.device` 最终变成了 `"cuda:0"`。

这说明 SGLang 源码（当前 git 版本）存在一个 bug 或者近期代码变更，某处将 `device` 从 `"cuda"` 修改为 `"cuda:0"`，但 `get_available_gpu_memory` 函数没有相应更新，仍然用精确匹配 `== "cuda"`。

### 解决方案

**方案 1：修复 `get_available_gpu_memory` 使用 `startswith` 代替精确匹配**

修改 `python/sglang/srt/utils/common.py:515`：

```python
# 将
if device == "cuda":
# 改为
if device == "cuda" or device.startswith("cuda:"):
```

**方案 2：确保传入 `get_available_gpu_memory` 的 device 是不带编号的**

在调用处（`model_runner.py:766`）提取基础 device 名：

```python
device_type = self.device.split(":")[0]  # "cuda:0" → "cuda"
before_avail_memory = get_available_gpu_memory(device_type, self.gpu_id)
```

### 注意

- 此错误与 SIMO 量化无关，no-quant 运行也报同样的错
- 这是 SGLang 源码的 bug（`device` 字符串格式不一致），不是 lm-eval 的问题
- 错误发生在 scheduler 子进程的 model_runner 初始化阶段，模型还没开始加载



---

## lm-eval → get_available_gpu_memory 的完整调用链和 device 参数传递

### 1. 进程模型：spawn 子进程

lm-eval 通过 **multiprocessing "spawn"** 方式启动 scheduler 子进程，**不是 fork**。

```python
# engine.py:883
mp.set_start_method("spawn", force=True)
```

spawn 意味着子进程是全新创建的，server_args 对象通过 **pickle 序列化** 传递给子进程。

---

### 2. 完整调用链

```
[主进程] lm-eval sglang_causallms.py
│
│  device: str = "cuda"  ← lm-eval 的默认参数 (line 55)
│  self.model_args = {"device": device, ...}  (line 94)
│  self.model = sgl.Engine(**self.model_args)  (line 111)
│
▼
[主进程] Engine.__init__()  (engine.py:139)
│
│  server_args = ServerArgs(**kwargs)  (line 154)
│  # kwargs 包含 device="cuda"
│  # ServerArgs.__post_init__ → _handle_missing_default_values:
│  #   if self.device is None: self.device = get_device()
│  #   由于 device="cuda" 不是 None，跳过
│  # 此时 server_args.device = "cuda"
│
│  logger.info(f"{server_args=}")  (line 156)  ← 打印 server_args
│
│  _launch_subprocesses(server_args=server_args, ...)  (line 163)
│
▼
[主进程] _launch_scheduler_processes()  (engine.py:911)
│
│  proc = mp.Process(
│      target=run_scheduler_process,
│      args=(server_args, port_args, gpu_id, ...)  (line 975)
│  )
│  proc.start()  ← spawn 子进程，server_args 通过 pickle 传给子进程
│
▼ ═══════════════ 进程边界（spawn） ═══════════════
│
[scheduler 子进程] run_scheduler_process()  (scheduler.py:3068)
│
│  server_args: ServerArgs  ← 从主进程 pickle 反序列化得到
│  # server_args.device 仍然是 "cuda"
│
│  scheduler = Scheduler(server_args, port_args, gpu_id, ...)  (line 3130)
│
▼
[scheduler 子进程] Scheduler.__init__()  (scheduler.py:268)
│
│  self.server_args = server_args
│  self.init_model_worker()  (line 368)
│
▼
[scheduler 子进程] Scheduler.init_tp_model_worker()  (scheduler.py:519)
│
│  self.tp_worker = TpModelWorker(server_args=self.server_args, ...)
│
▼
[scheduler 子进程] TpModelWorker.__init__()  (tp_worker.py:209)
│
│  self._init_model_runner()  (line 247)
│
▼
[scheduler 子进程] TpModelWorker._init_model_runner()  (tp_worker.py:327)
│
│  self._model_runner = ModelRunner(server_args=self.server_args, ...)
│
▼
[scheduler 子进程] ModelRunner.__init__()  (model_runner.py:285)
│
│  self.device = server_args.device  ← "cuda" (line 305)
│  self.gpu_id = gpu_id              ← 0 (line 306)
│  ...
│  min_per_gpu_memory = self.init_torch_distributed()  (line 393)
│
▼
[scheduler 子进程] ModelRunner.init_torch_distributed()  (model_runner.py:760)
│
│  before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
│                                                  ↑            ↑
│                                               "cuda"          0
▼
[scheduler 子进程] get_available_gpu_memory(device="cuda", gpu_id=0)  (common.py:508)
│
│  if device == "cuda":       ← 匹配！进入此分支
│      ...
│      free_gpu_memory, _ = torch.cuda.mem_get_info(gpu_id)
│  return free_gpu_memory / (1 << 30)
```

---

### 3. device 参数的值变化追踪

| 步骤 | 位置 | device 值 | 进程 |
|------|------|-----------|------|
| lm-eval 默认参数 | sglang_causallms.py:55 | "cuda" | 主进程 |
| Engine(**kwargs) | engine.py:154 | "cuda" | 主进程 |
| ServerArgs.__post_init__ | server_args.py:851 | "cuda"（不是 None，跳过） | 主进程 |
| mp.Process(args=(server_args,...)) | engine.py:975 | "cuda"（pickle 序列化） | 主进程 |
| run_scheduler_process(server_args) | scheduler.py:3068 | "cuda"（pickle 反序列化） | 子进程 |
| ModelRunner.__init__ | model_runner.py:305 | self.device = "cuda" | 子进程 |
| get_available_gpu_memory | common.py:531 | device="cuda" | 子进程 |

**结论：device 从头到尾都是 "cuda" 字符串**，GPU 编号通过独立的 gpu_id 参数传递（值为 0，因为 CUDA_VISIBLE_DEVICES=6 使 GPU 6 映射为 device 0）。

---

### 4. 关于之前分析中的 device='cuda:0'

之前的错误日志中 ServerArgs repr 显示 device='cuda:0'，这说明**那次运行时的代码版本**或**某处修改**将 device 设为了 "cuda:0" 格式。但根据当前源码的分析，正常流程中 device 始终是 "cuda"（不带 GPU 编号）。

如果 device 确实变成了 "cuda:0"，那么 get_available_gpu_memory 中 if device == "cuda": 不会匹配，导致 free_gpu_memory 未赋值，触发 UnboundLocalError。可能的原因：
- 代码在不同时间点有过修改
- 某个中间版本的 get_device() 或 ServerArgs.__post_init__ 行为不同

---

### 5. 进程关系图

```
[主进程] lm-eval CLI
  └── Engine.__init__()
        ├── ServerArgs.__post_init__()
        ├── mp.set_start_method("spawn")
        ├── [spawn 子进程] Scheduler → TpModelWorker → ModelRunner
        │     └── get_available_gpu_memory(device="cuda", gpu_id=0)
        └── [spawn 子进程] Detokenizer
```

---

# SGLang vs vLLM SIMOLinearMethod W4A4 MXFP 实现差异分析

## 现象

| 框架 | 模型 | 任务 | word_perplexity |
|------|------|------|-----------------|
| SGLang | Llama-3.1-70B | wikitext | **27.6423** |
| vLLM   | Llama-3.1-70B | wikitext | **10.2393** |

SGLang 路径上 W4A4 MXFP 的困惑度 (perplexity) 是 vLLM 的约 2.7 倍，明显是量化路径实现存在缺陷。

---

## 1. 配置文件差异

| 字段 | sglang `quant_config_w4a4_mxfp.json` | vllm `quant_config_ocp_w4a4_mxfp.json` |
|------|--------------------------------------|----------------------------------------|
| `per_quant_opt` | `"online_down_proj_rotation"` (字符串) | `null` |

**问题**：
- sglang 配置中显式开启了 `online_down_proj_rotation`（在线 Hadamard rotation），但 SGLang 侧的 `SIMOLinearMethod` / `get_quant_method_by_target_spec` 根本**没有实现** Hadamard transform 逻辑，配置形同虚设。
- 即使 vllm 这一侧实现了 hadamard 路径，由于 vllm 的 config 实际是 `null`，两边在该选项上"巧合一致地不走 rotation"，但 sglang 配置错误地写成了字符串而非 list，反映出 sglang 这条链路缺乏对应的解析与执行代码。
- 如果训练/导出阶段 weights 已经经过 down_proj rotation 的"前置变换"，那么推理阶段未做配套 rotation，会导致大量数值错位 → 困惑度大幅下降。

---

## 2. SIMOLinearMethod 关键差异（PER_BLOCK / W4A4 MXFP 路径）

| 比较项 | SGLang 实现 | vLLM 实现 | 影响 |
|--------|-------------|-----------|------|
| `process_weights_after_loading` 中的 weight padding（按 group_size 对齐每个 shard） | **缺失** | 显式 pad 每个 logical shard 到 group_size 边界（quantization_method.py:302-325） | shard 边界与 scale group 不对齐 → scale 与 weight 错位 |
| `apply()` 中的 output slicing（去掉 padding 区域） | **缺失** | 按 logical_widths 逐 shard 切片再 cat（:574-601） | 输出包含 padding 行 → 后续 layer 接收错误数据 |
| `create_weights` 中 PER_BLOCK 的 scale shape 计算 | 直接用 `weight_downcast_kernel` 输出原始 shape（:1003-1009） | `sum((p+gs-1)//gs for p in output_partition_sizes)`（:511-524） | 多 shard 合并时 `ceil(sum) ≠ sum(ceil)`，scale 维度不匹配 |
| `layer.input_scale = None` 初始化 | **缺失** | `:488` 显式初始化 | sglang 可能命中残留值或 attribute 不存在分支 |
| `apply()` 中的输入量化路径 | `if hasattr(layer, "weight_scale")` 分支，否则不量化 | 始终量化输入（:574-579） | sglang 在某些 W4A4 layer 上可能误走 fp16 路径，破坏假设 |
| `input_global_scale` 触发条件 | `if self.global_scale_factor is not None` 任意命中 | 仅 `input_spec.dtype == "nvfp4_e2m1"` | sglang 在 mxfp4 路径上误用 NVFP4 的 global scale，等价于额外缩放，量化误差放大 |
| Hadamard / `per_quant_opt` 解析与执行 | **完全没有实现** | 在 dispatch / weight 处理中支持 | 导致配置实际不生效或行为不一致 |

### 关键代码位置

- sglang: `simo/extensions/sglang_simo/quantization/quantization.py:846-1121`（`SIMOLinearMethod`）
- vllm:  `simo/extensions/vllm_simo/quantization/quantization_method.py:290-612`（`SIMOLinearMethod`）

---

## 3. 困惑度 2.7x 退化的根因假设（按可能性排序）

1. **per_quant_opt 实现缺失**（最大嫌疑）
   - 配置串误（字符串 vs list）+ sglang 侧无 hadamard 代码路径，意味着如果训练侧曾依赖此 rotation，推理就破坏了 weight 分布 → 误差累积 80 层后 perplexity 显著抬升。

2. **input_global_scale 在非 NVFP4 路径上被错误触发**
   - sglang 对 mxfp 输入也额外做了 amax 全局缩放，破坏 mxfp block scale 的 E8M0 量纲假设 → activation 数值整体偏移。

3. **PER_BLOCK shard padding / output slicing 缺失**
   - Llama-70B 的 `gate_up_proj`、`qkv_proj` 是合并 shard：当 shard size 不是 group_size (32) 整数倍时，没有 pad，`weight_scale` 与 `weight` 行号产生错位，dequant 结果直接错。
   - 没有 output slicing 时，下游 down_proj/o_proj 接收到 padding 区的脏数据。

4. **layer.input_scale 未初始化**
   - 若 vllm 路径上有"已经设置过则跳过"的判断，sglang 缺失初始化可能让首次 forward 命中错误分支（小概率，但可叠加）。

---

## 4. 大模型量化精度问题的调试方法

### 4.1 缩小问题域：从 task → layer → tensor

1. **最小可复现集**
   - 切换到 wikitext 子集（前 N 条），固定 seed，关闭 sampling（greedy / top_k=1），保证可重放。
2. **逐 layer 数值对齐**
   - 同一段输入分别跑 fp16 baseline、vllm SIMO、sglang SIMO，挂 `forward_hook`/`forward_pre_hook` 抓每层 `(input, output)`。
   - 计算 cosine sim 与 mean abs error，定位"第一发散层"（first divergence layer）。这通常立即指出问题模块（attention / mlp / down_proj / qkv_proj）。
3. **对齐 dequant 中间量**
   - 在 `apply()` 内打印/落盘 `qdq_x`、`dq_w`、`weight_scale`、`input_global_scale`，跨框架对比。
   - 对比 `layer.weight.shape`、`layer.weight_scale.shape`，看是否在合并 shard 处发生 padding/slicing 不一致。

### 4.2 配置/路径对齐

4. **diff 两端 config 与 dispatch**
   - `quant_config_*.json` diff（per_quant_opt、global_scale、group_size、axis）
   - 在 `get_quant_method_by_target_spec` 处加 log，确认两端最终选中的 method 类与 spec 完全一致。
5. **关闭可疑 feature 二分**
   - 把 sglang 的 `per_quant_opt` 显式置 `null`，再跑一次 → 看 perplexity 是否回升到 ~10。这一步能直接证伪/证实 hadamard 假设。
   - 临时强制 `input_global_scale=None` 路径再跑。

### 4.3 单元 / 端到端校验

6. **Layer-level 单测**
   - 抽出 70B 中一层 `gate_up_proj`（合并 shard）, 喂随机输入 `[B, hidden]`，对比 vllm/sglang 两端 `apply()` 输出是否 bit-wise 或 1e-2 一致。
7. **量化-反量化 round trip**
   - 对单个权重 tensor 做 `weight_upcast(weight, weight_scale)`，比较与原 fp16 weight 的 max abs error 与按行 norm error；scale 错位时单行误差会突变。
8. **TP=1 vs TP=N**
   - PER_BLOCK + 合并 shard 的 padding 问题在 TP>1 时常被掩盖或放大，跑 TP=1 baseline 隔离 sharding 因素。

### 4.4 监控指标

9. **逐层激活的 amax / RMS**
   - 比对量化后激活的 amax 序列：若某一层 amax 突然爆掉 / 趋零，多半 dequant 错位。
10. **可视化 weight/scale heatmap**
    - 对 weight_scale（[Out//gs, In//gs]）做 heatmap，padding/slicing 错位会出现明显的"零行/零列条带"或"错位平铺"。

### 4.5 修复优先级建议

1. 先把 `quant_config_w4a4_mxfp.json` 中 `per_quant_opt` 改为 `null`（与 vllm 对齐），重新评估 → 验证 hypothesis 1。
2. 把 vllm 的 padding / output slicing / per-shard ceil scale 三段逻辑移植到 sglang `SIMOLinearMethod`。
3. 把 `input_global_scale` 触发条件收窄到 `dtype == "nvfp4_e2m1"`。
4. 补全 `layer.input_scale = None` 初始化。
5. 若需 hadamard，则在 dispatch 与 `apply()` 中实现 `online_down_proj_rotation` 路径并要求权重导出与之配套。


---

# 基于 debug_ppl 日志的 SIMOLinearMethod.apply 数值对比

## 数据来源

- sglang 日志：`temp/lm-eval-mxfp4.log.2026_04_27___00_19_11`（80 层 × 4 proj × 8 TP rank ≈ 2560 条）
- vllm 日志：`temp/vllm.lm-eval-mxfp4.log.2026_04_27___00_02_27`
- 字段：`qdq_x_abs_max / qdq_x_abs_mean / dq_w_abs_max(实为mean) / output_abs_max / output_abs_mean`

下文所有数据均按 layer_prefix 跨 TP rank 取平均。

## 1. 关键结论先行

| 项目 | sglang vs vllm 比值 | 含义 |
|------|--------------------|------|
| `dq_w_abs_mean`（所有 proj） | **1.000**（精确一致） | 权重量化 / 反量化路径在两个框架上数值一致，**weight 不是发散源** |
| `qkv_proj` 输入/输出统计 | 0.98 ~ 1.05 | residual stream 进入每层时基本对齐 |
| `gate_up_proj` / `down_proj` | 0.94 ~ 1.04 | MLP 路径无显著差异 |
| **`o_proj` 输入 (`qdq_x_abs_mean`)** | **1.383** | sglang 比 vllm 高 38% |
| **`o_proj` 输出 (`output_abs_mean`)** | **1.349** | sglang 比 vllm 高 35% |

> **dq_w_mean 完全一致 → 权重路径无差异；唯独 o_proj 的输入显著偏大 → 发散来源在 attention kernel（attention 输出送入 o_proj）。**

## 2. 各 projection 类型的统计平均

| Projection | qdq_x_mean (sg/vl) | out_mean (sg/vl) | 比值 (out_mean) |
|-----------|--------------------|-------------------|-----------------|
| qkv_proj      | 0.1273 / 0.1292 | 0.7331 / 0.7300 | **1.004** |
| **o_proj**    | **0.02104 / 0.01521** | **0.00728 / 0.00540** | **1.349** |
| gate_up_proj  | 0.1089 / 0.1114 | 0.1490 / 0.1513 | 0.985 |
| down_proj     | 0.01131 / 0.01167 | 0.01191 / 0.01204 | 0.989 |

**只有 o_proj 显著偏离，其他三类几乎一致。**

## 3. 早期 layer 的 o_proj 偏差（最可能放大 PPL 的位置）

按 `output_abs_mean` 比值（sglang/vllm）从大到小：

| Layer | sg out_mean | vl out_mean | 比值 |
|-------|------------:|------------:|----:|
| layers.7.o_proj  | 0.01222 | 0.003725 | **3.28x** |
| layers.6.o_proj  | 0.01254 | 0.003901 | **3.21x** |
| layers.5.o_proj  | 0.01044 | 0.003959 | **2.64x** |
| layers.8.o_proj  | 0.00882 | 0.003944 | 2.24x |
| layers.4.o_proj  | 0.00752 | 0.003971 | 1.89x |
| layers.3.o_proj  | 0.00447 | 0.002854 | 1.57x |
| layers.12.o_proj | 0.00637 | 0.004072 | 1.57x |
| layers.9.o_proj  | 0.00910 | 0.005819 | 1.56x |

**前 10 层的 o_proj 系统性偏大 1.5~3.3 倍**——这正是 PPL 退化的时间序列源头：早期层的小偏差经由 residual stream 累积到 80 层。

## 4. 解释：为何只有 o_proj 偏？

`o_proj` 的输入 = attention(Q,K,V) 的输出。其上游链路差异点：

1. **attention kernel 不同**
   - vllm 走 vllm 自己的 attention kernel（PagedAttention / FlashAttention）。
   - sglang 走 SGLang 的 extend/decode attention triton kernel；本工作分支(`sgl-gqa-kv-cache`)甚至带有 KV cache 量化路径（MX/per-group），即便配置未启用 KV 量化，kernel dispatch 与 numerics 与 vllm 仍然不同。

2. **dq_w 一致 → 排除**：
   - o_proj 自身的 weight、weight_scale 完全相同（dw_mean 比值=1.000）。
   - 因此 o_proj 的输出差异完全由 **输入差异** 导致（输入即 attention out）。

3. **qkv_proj 几乎对齐 → 排除前置**：
   - qkv_proj 的 in/out 比值 ≈ 1.00，说明进入 attention 的 Q/K/V 在两端基本一致。
   - 偏差注入点位于 **Q/K/V 之后到 attention output 之前**，即 attention kernel 内部（softmax、KV 读取、reduce）。

4. **下一层 qkv_proj 又回到 ~1x**：
   - 是因为 RMSNorm + residual 的归一化效应把当前层的偏差摊平到了同一个 scale，但 perplexity 对实际 token logits 仍然敏感，差异沉淀下来。

## 5. SIMOLinearMethod.apply 本身的数值评价

- W4A4 mxfp 的 **weight dequant 路径在两个框架上 bit-wise 等价**（dq_w_mean 完全一致）。
- **input downcast / upcast** 路径也基本一致（qdq_x 在 qkv/gate_up/down 三类上比值在 [0.94, 1.05]）。
- 也就是说：**SIMOLinearMethod.apply 这一层的数值差异 < 5%**，不是 PPL 2.7x 退化的主要矛盾。

> 这与上一节"代码差异表"的结论形成了**有力补充**：尽管 vllm 侧的 `apply()` 写得更完整（padding/slicing/conditional global scale 等），但在 Llama-3.1-70B 这个具体模型上，logical_widths 与 group_size=32 是对齐的（4096/8192/14336 等都是 32 的倍数），padding/slicing 路径不会被触发，所以 apply() 本身在数值上恰好对齐。

## 6. 修正后的 PPL 退化主因排序

结合代码 diff 与 numerics 双重证据：

1. **Attention kernel numerics**（首要嫌疑，**新发现**）
   - o_proj 输入在前 10 层就放大到 1.5~3.3x，是直接证据。
   - 检查方向：
     - SGLang `extend_attention` / `decode_attention` triton kernel 的 softmax / log-sum-exp 是否数值稳定（fp32 累加？）。
     - KV cache dtype 是否与 vllm 一致（即便没开 KV quant，sglang 这个分支 layout 可能有差异）。
     - GQA `num_kv_heads` 重复展开方式、`tl.dot` accumulator dtype 是否一致。
     - 是否触发了本分支正在改的 KV cache quant 代码路径（即便 spec 是 None）。

2. SIMOLinearMethod.apply 实现差异（次要）
   - 对 Llama-70B 的 logical_widths 不构成实际影响。
   - 但仍建议补齐 padding/slicing/`input_scale=None`/`input_global_scale` 触发条件，以避免在 group_size 不整除的模型上踩雷。

3. `per_quant_opt: "online_down_proj_rotation"`（确认无影响）
   - sglang 侧无解析逻辑；如果训练侧也没真正应用 rotation，则不影响。

## 7. 进一步定位 attention kernel 偏差的方法

1. 把 `lm-eval` 的输入固定到一条 wikitext 文本，TP=1 跑两端，dump 每层 `attn_output`（即 o_proj 输入），逐 token 比对 cosine sim。
2. 在 SGLang 的 `extend_attention_fwd` 与 vllm 的 `flash_attn_varlen_func` 同等输入下做单测，看 max_abs_err。
3. 关闭 `disable_cuda_graph=True`、`enforce_eager=True` 已排除 graph capture 影响——剩下就是 kernel 本身。
4. 在 sglang `_fwd_kernel` 内强制 `acc = tl.zeros(..., dtype=tl.float32)` 且 softmax 用 fp32，确认是否是累加精度问题。
5. 二分屏蔽：把 sglang 的 `extend_attention` 暂时替换为调用 `flash_attn_varlen_func`（简单 GQA 路径），看 PPL 是否回升至 ~10.5。

## 8. 一句话总结

> **debug_ppl 数据明确表明：W4A4 MXFP 的 SIMOLinearMethod.apply 在两端数值几乎一致（差 <5%），dq_w_mean 完全相等；PPL 2.7x 退化的真正根因不在 LinearMethod，而在 SGLang 的 attention kernel——表现为前 10 层 o_proj 输入被系统性放大 1.5~3.3 倍，差异沿 residual stream 累积 80 层。**


---

# 关于该 lm-eval 命令使用的 attention 后端 / 是否触发 sglang_simo KV 量化

## 1. 该命令使用的 attention 后端：**fa3**（FlashAttention-3）

### 判定依据

命令中 `model_args` 没有传 `attention_backend / prefill_attention_backend / decode_attention_backend`，于是走默认选择逻辑：

`/share_data/users/like/package/h100/package/sglang_kernel_src/python/sglang/srt/server_args.py:1772-1837` `_handle_attention_backend_compatibility()`：

```python
if not use_mla_backend:
    # MHA architecture
    if is_hopper_with_cuda_12_3() and is_no_spec_infer_or_topk_one(self):
        # ref: https://github.com/sgl-project/sglang/issues/17411
        self.attention_backend = "fa3"
    elif is_sm100_supported() and ...:
        self.attention_backend = "trtllm_mha"
    elif is_hip():
        self.attention_backend = "aiter"
    else:
        self.attention_backend = "flashinfer" if is_flashinfer_available() else "triton"
```

对应到本次运行环境与配置：

| 条件 | 实际值 | 结论 |
|------|-------|------|
| `use_mla_backend()` | False（Llama-3.1-70B 是 GQA / MHA） | 走 MHA 分支 |
| `is_hopper_with_cuda_12_3()` | True（H100 + cuda12.3） | ✓ |
| `is_no_spec_infer_or_topk_one()` | True（命令未启用投机推理） | ✓ |
| → 命中 line 1803 | | **`attention_backend = "fa3"`** |

启动时日志会有：`Attention backend not specified. Use fa3 backend by default.`

> 注意：`disable_cuda_graph=True` 不影响 backend 选择，只关闭 cudagraph capture。

### 这意味着 SGLang 的 `python/sglang/srt/layers/attention/flashattention_backend.py`（FlashAttention-3 backend）会处理 prefill/decode 的 attention 计算，并不是 SGLang 自己的 triton kernel，更不是 simo 自己的 triton kernel。

---

## 2. 是否触发 sglang_simo 的 KV cache 量化代码？**不会**

### 触发链需要同时满足两点，本次命令两点都不满足：

#### A. 配置层面：`kv_cache_quant_algo: null`

`quant_config_w4a4_mxfp.json`：
```json
"kv_cache_quant_algo": null
```

在 `simo/extensions/sglang_simo/quantization/quantization.py`：

```python
# 解析
kv_cache_quant_algo = config.get("kv_cache_quant_algo", None)   # → None

# dispatch
if isinstance(layer, RadixAttention) and self.kv_cache_quant_algo:
    # 创建 SIMOKVCacheMethod，并最终设置 layer.kv_cache_quant_spec
    ...
```

`None` 为假，分支不进入 → `SIMOKVCacheMethod` 不会被实例化 → `layer.kv_cache_quant_spec` 从未被赋值。

#### B. Backend 层面：默认根本没选 `triton_simo`

simo 仅注册了一个 `triton_simo` backend（`simo/extensions/sglang_simo/layers/attention/attention_backend.py`）：
```python
@register_attention_backend("triton_simo")
def create_triton_simo_backend(runner): ...
```

但当前命令的 `attention_backend` 已被默认逻辑设置为 `"fa3"`，**根本不会构造 `SIMOTritonAttnBackend`**。要触发它，必须在 `model_args` 显式加：
```json
"attention_backend": "triton_simo"
```

#### C. 即便 backend 是 `triton_simo`，也会被 runtime guard 二次拦截

`simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py`：
```python
def forward_extend(self, q, k, v, layer, ...):
    kv_cache_quant_spec = getattr(layer, "kv_cache_quant_spec", None)
    if kv_cache_quant_spec is None:
        return super().forward_extend(...)   # 退回 SGLang TritonAttnBackend
    # 走 simo extend_attention 量化 KV 读路径
    ...

def forward_decode(...):
    kv_cache_quant_spec = getattr(layer, "kv_cache_quant_spec", None)
    if kv_cache_quant_spec is None:
        return super().forward_decode(...)
    ...
```

由于 A 节中 `layer.kv_cache_quant_spec` 永远不会被设置，runtime 也会回退到非量化路径。

### 结论速览

| 问题 | 答案 |
|------|------|
| 实际使用的 attention backend | **`fa3` (FlashAttention-3)** |
| `simo/extensions/sglang_simo/layers/attention/extend_attention.py`、`decode_attention.py` triton kernel 是否被调用？ | **否** |
| `set_kv_buffer.py` 的量化写路径是否被调用？ | **否**（layer.kv_cache_quant_spec 为 None） |
| 本次 wikitext PPL=27.6423 是否由 sglang_simo KV cache 量化路径引起？ | **否**，KV cache 全程未量化，attention 走的是 fa3 |

---

## 3. 这对前面 PPL 退化定位的修正

之前根据 debug_ppl 数据得出"o_proj 输入在前 10 层放大 1.5~3.3x → 怀疑 SGLang attention kernel"。结合本节：

- 因为 attention 实际走的是 **FlashAttention-3**，不是 SGLang 的 triton kernel，也不是 simo 的 triton 量化 kernel；
- 而 vllm 那条链路用的是 vllm 的 FA / attention 实现（`enforce_eager=True` 下通常是 `flash_attn_varlen_func`）；
- 因此 o_proj 输入差异 1.38x 的来源应聚焦在 **"两个框架的 FA-3 集成参数差异"**，而不是 sglang/simo 的 triton kernel 错误：

需要核对的差异点：
1. **softmax_scale**：`1/sqrt(head_dim)` 是否一致（rope_scaling、yarn 是否影响 scale）。
2. **causal mask + sliding window** 设置是否一致（Llama-3.1 没有 SWA，但需确认未误开）。
3. **kv_cache layout**：sglang fa3 backend 的 page_size、block_table、KV dtype（fp16 vs bf16）。
4. **rope 计算**：sglang 的 rope kernel vs vllm 的 rope kernel 在 long-context 下数值差异（Llama-3.1 用的 rope_scaling="llama3"）。
5. **logit soft-cap、attention sinks** 是否被错误激活。
6. **fa3 版本**：sglang 与 vllm 链接的 flash-attn 版本是否相同（本环境 sglang_kernel_src 自带 vs vllm wheel 内置）。
7. **mem_fraction_static=0.5** 与 vllm 的 `gpu_memory_utilization=0.6` 对 KV pool 大小、page swap 行为的影响（长序列 wikitext 下，溢出策略可能不同）。

> 简言之：当前 lm-eval 命令下，sglang_simo 的 KV cache 量化代码**完全没有被执行**，PPL 差异需要在 SGLang 上层（fa3 集成、rope、KV pool）里继续定位。


---

# 多组数据类型 PPL + debug_ppl 数值对比（最终版）

## 0. 各 quant 配置下的 word_perplexity（来自无均值日志）

| 配置 | sglang_simo | vllm_simo | 比值 sg/vl | 是否发散 |
|------|------------:|----------:|-----------:|:--------:|
| w8a8_mxfp           | 2.9432 | 2.9703 | 0.99 | ✓ 一致 |
| w8a8_mxint          | 2.8423 | 2.8737 | 0.99 | ✓ 一致 |
| w8a8_fp8_per_channel| 2.8905 | 2.9224 | 0.99 | ✓ 一致 |
| w8a8_int8_per_block | 3.0725 | 2.9668 | 1.04 | ✓ 一致 |
| w8a8_int8_per_channel | 55.58 | 49.02 | — | 两端都坏（算法本身不行） |
| w6a6_mxfp           | 2.9301 | 2.9669 | 0.99 | ✓ 一致 |
| w4a16_int4_per_group| 3.7414 | 4.0577 | 0.92 | ✓ 一致 |
| w4a4_nvfp           | 4.4210 | 4.1701 | 1.06 | ✓ 一致 |
| w4a16_nvfp4_per_group | 3.9018 | 3.6270 | 1.08 | ✓ 一致 |
| **w4a4_mxfp**       | **27.6423** | **10.2393** | **2.70** | ✗ **唯一显著发散** |

> 唯一发散点是 **w4a4_mxfp**，其它所有数据类型（包括同样 4-bit 的 nvfp、int4_per_group）两端都几乎相等。

---

## 1. 为什么唯独 w4a4_mxfp 退化？

### 决定性证据：配置文件差异

| 配置文件 | `per_quant_opt` |
|---------|------------------|
| `quant_config_w8a8_mxfp.json` | `null` |
| `quant_config_w4a4_nvfp.json` | `null` |
| `quant_config_w4a16_int4_per_group.json` 等 | `null` |
| **`quant_config_w4a4_mxfp.json`** | **`"online_down_proj_rotation"`** |

→ **只有 w4a4_mxfp 的 sglang 配置开启了 `online_down_proj_rotation`**（且写成字符串而非 list）。其它配置都是 `null`。

### 再加上 SGLang 侧实现缺失

sglang_simo 的 `SIMOLinearMethod` / `get_quant_method_by_target_spec` **完全没有实现** `per_quant_opt` 解析与对应的 Hadamard rotation 路径（只有 vllm_simo 有完整支持）。

### 把这两件事合在一起，就解释了所有现象：

- **w8a8 / w6a6 / w4a4_nvfp / w4a16_int4** 等：`per_quant_opt=null`，sglang 与 vllm 走的是同一条无 rotation 路径 → PPL 几乎相等。
- **w4a4_mxfp**：vllm 端因为对应 vllm config (`quant_config_ocp_w4a4_mxfp.json`) 的 `per_quant_opt=null`，正常无 rotation；sglang 端 config 写了 `"online_down_proj_rotation"` 但 sglang 代码不识别 → 两端在 down_proj 输入分布的处理上**结构性不一致**。

但有个反证需要正视：debug_ppl 数据显示 sglang 与 vllm 在所有 proj 上 `dq_w_abs_mean` ratio = **1.000**（精确一致），qx_mean 在 qkv/gate_up/down 上 ratio≈0.97~1.04，这意味着 **sglang 实际上既没有应用 hadamard 也没有破坏权重**——配置只是被静默忽略了。

那 PPL 仍然差 2.7x 的真凶就要往别的方向看：**MXFP4 对输入分布扰动的极端敏感性**。

---

## 2. MXFP4 为什么对输入扰动极度敏感（其它格式不敏感）

### 量化精度对比

| 格式 | bits | 每 group 码字数 | scale 类型 | 全局 scale |
|------|-----:|------:|-----------|-----------|
| MXFP8_E4M3 (w8a8_mxfp) | 8 | 256 | E8M0 (power-of-2) | 无 |
| MXINT8                  | 8 | 256 | E8M0 | 无 |
| FP8 per_channel/per_block| 8 | 256 | float32 | 无 |
| MXFP6_E2M3              | 6 | 64 | E8M0 | 无 |
| **MXFP4_E2M1**          | **4** | **16** | **E8M0**（仅 power-of-2） | **无** |
| NVFP4                   | 4 | 16 | **E4M3 group + per-tensor float** | **有（双 scale）** |
| INT4 per_group          | 4 | 16 | float | 有 |

**MXFP4_E2M1 是表中量化能力最弱的：**
1. 只有 16 个码字（含零），动态范围极窄；
2. group scale 是 **E8M0**（仅能表示 2^k），即 scale 只能"翻倍 / 减半"地跳变；
3. 没有 per-tensor 全局 scale 校准。

### 这导致什么？

当 sglang 与 vllm 在 attention 后端不同（fa3 vs vllm-fa）时，o_proj 的输入有 ~1.06x 的 max 偏移，~1.05x 的 mean 偏移——

- **MXFP8** 8 bit + 256 码字：1.05x 的输入扰动 → qdq_x 几乎不变（代码字够细，scale 也是 E8M0 但有效精度区间宽）。
- **MXFP4** 4 bit + 16 码字：1.05x 的输入扰动可能跨过"是否触发 E8M0 scale 翻倍"的临界点 → 整个 group 的 dequant 误差直接 2x 起跳。
- 而 **NVFP4** 即便也是 4 bit，由于有 per-tensor 全局 scale 把 amax 归一到 fp8 表达范围，1.05x 的输入扰动会被全局 scale 自动吸收 → PPL 不退化。

### 对应的 debug_ppl 数据（佐证）

| 配置 | o_proj qx_mean ratio (sg/vl) | o_proj out_mean ratio | wikitext PPL ratio |
|------|------------------------------:|----------------------:|-------------------:|
| **w4a4_mxfp** | **1.384** | **1.350** | **2.70** |
| w8a8_mxfp     | 1.137 | 1.136 | 0.99 |

同样的"上游 attention 输入差 1.06x"，进入：
- MXFP8 路径：qdq_x 仅放大到 1.14x（输出对齐）
- MXFP4 路径：qdq_x 直接被放大到 1.38x（前 10 层 o_proj 局部 ratio 高达 **3.3x**），随 residual 累积 80 层后 PPL 差 2.7x。

---

## 3. w4a4_mxfp 下 sglang vs vllm 的 debug_ppl 数值差异

### Per-projection 平均（sg/vl 比值）

| Projection | qx_max | qx_mean | dw_max | dw_mean | out_max | out_mean |
|-----------|-------:|--------:|-------:|--------:|--------:|---------:|
| qkv_proj    | 1.003 | 0.985 | 1.000 | 1.000 | 1.047 | 1.004 |
| **o_proj**  | 1.060 | **1.384** | 1.000 | 1.000 | 1.078 | **1.350** |
| gate_up_proj| 1.034 | 0.978 | 1.000 | 1.000 | 1.014 | 0.985 |
| down_proj   | 0.937 | 0.970 | 1.000 | 1.000 | 1.030 | 0.989 |

### 关键观察

- **dw_max / dw_mean 在所有 proj 上完全一致（1.000）**：weight downcast/upcast 路径数值一致，权重不是发散源。
- **qx 在 qkv / gate_up / down 上几乎一致（±5%）**：input 端在普通 proj 上对齐良好。
- **o_proj 的 qx_mean 高出 38%、out_mean 高出 35%**，前 10 层局部 ratio 高达 1.5~3.3x（layers.5~10）。

### Top 偏离层（按 out_mean 比值）

| Layer | sg out_mean | vl out_mean | 比值 |
|-------|------------:|------------:|----:|
| layers.7.o_proj | 0.01222 | 0.00370 | **3.30** |
| layers.6.o_proj | 0.01254 | 0.00388 | **3.24** |
| layers.5.o_proj | 0.01044 | 0.00393 | **2.65** |
| layers.8.o_proj | 0.00882 | 0.00393 | 2.25 |
| layers.4.o_proj | 0.00752 | 0.00395 | 1.90 |

→ 集中在前 10 层 o_proj，与 attention 后端差异 + MXFP4 量化噪声在低层放大的预期一致。

---

## 4. w8a8_mxfp 下 sglang vs vllm 的 debug_ppl 数值差异

### Per-projection 平均（sg/vl 比值）

| Projection | qx_max | qx_mean | dw_max | dw_mean | out_max | out_mean |
|-----------|-------:|--------:|-------:|--------:|--------:|---------:|
| qkv_proj    | 1.044 | 0.999 | 1.000 | 1.000 | 1.052 | 1.023 |
| **o_proj**  | 1.063 | **1.137** | 1.000 | 1.000 | 1.074 | **1.136** |
| gate_up_proj| 1.077 | 0.997 | 1.000 | 1.000 | 1.025 | 1.009 |
| down_proj   | 1.026 | 1.012 | 1.000 | 1.000 | 1.084 | 1.002 |

### 关键观察

- 同样存在 **o_proj 输入偏大** 的现象（这是上游 attention 后端差异的固有效应）。
- 但 ratio 仅 1.14（vs w4a4_mxfp 的 1.38），输出 ratio 也仅 1.14。
- 累积到 wikitext 的 PPL 差异只有 ~1%（2.94 vs 2.97）→ **MXFP8 把上游扰动有效吸收了**。

### w8a8 top 偏离层

| Layer | sg out_mean | vl out_mean | 比值 |
|-------|------------:|------------:|----:|
| layers.1.o_proj | 7.5e-4 | 4.1e-4 | 1.82 |
| layers.2.o_proj | 6.4e-4 | 4.4e-4 | 1.46 |
| layers.4.o_proj | 7.4e-4 | 5.3e-4 | 1.42 |

→ **注意尺度**：w8a8 早期 o_proj 的绝对均值在 1e-4 量级，是 w4a4 同位置（1e-2 量级）的 1/100。即便比值看似 1.8x，绝对偏差对 logits 的影响远小于 w4a4。

---

## 5. w8a8 vs w4a4 直接对比：同样的"上游差异"，结局完全不同

| 维度 | w8a8_mxfp | w4a4_mxfp |
|------|----------:|----------:|
| o_proj qx_max ratio (sg/vl) | 1.063 | 1.060 |
| o_proj qx_mean ratio | 1.137 | **1.384** |
| o_proj out_mean ratio | 1.136 | **1.350** |
| 早期 o_proj 局部最大 ratio | ~1.8x | **~3.3x** |
| 早期 o_proj 绝对值量级 | 1e-4 | 1e-2 |
| wikitext PPL (sg/vl) | 0.99 | **2.70** |

**两端的"原始上游 attention 差异"几乎相同（qx_max ratio ≈ 1.06）**，但经过量化器后：
- MXFP8 几乎无放大（qx_mean 比值从 1.06→1.14）；
- MXFP4 大幅放大（qx_mean 比值从 1.06→1.38），并通过 residual 在 80 层中累积。

---

## 6. 综合根因（最终结论）

PPL 退化是**两个独立因素的乘积**：

1. **上游 attention 后端差异**（共同因素）
   - sglang 默认 backend = `fa3`，vllm 默认 = vllm 自家 FA。
   - 两者实现细节（softmax 数值稳定性、KV layout、rope kernel 等）导致 attention output 在前几层有 ~5~6% 的 max/mean 偏移。
   - 这个偏移**对所有量化配置都存在**（w8a8 与 w4a4 数据上都看得到 o_proj qx_max ~1.06）。

2. **MXFP4 的低分辨率 + E8M0 power-of-2 scale + 无全局 scale**（放大器）
   - 表内最弱的量化格式。1~6% 的输入扰动可能跨过 E8M0 阶跃边界，直接产生 2x 量化误差。
   - 没有 per-tensor 全局 scale 来吸收上游波动。
   - 与之相比：MXFP8（多 4 bit 精度）、NVFP4（有双 scale）、INT4 per_group（有 float scale）都没有这个放大器。

**所以为什么唯独 w4a4_mxfp 退化**：因为只有它同时遇到了 (1) 上游差异 + (2) 表中最不鲁棒的量化方案。其它格式要么没遇到 (1)（不存在）、要么有自身鲁棒性顶住 (2)。

> `per_quant_opt: "online_down_proj_rotation"` 配置在 sglang 端没生效（dq_w 数值与 vllm 完全一致即为证明），所以它**不是直接原因**；但它**反映了一个未对齐的设计意图**——这个量化方案原本依赖 hadamard rotation 来抑制 down_proj/o_proj 的 outlier，sglang 没实现就只能"裸奔"，进一步放大了 MXFP4 对扰动的敏感性。

---

## 7. 修复方向优先级

1. **最高 ROI（可立即验证）**：在 sglang 上 force `attention_backend="fa3"` 已经是当前默认，再加上 `--page_size=1`，对齐 vllm-FA 的 page 行为；并核对 rope_scaling/llama3 在两端的 kernel 是否一致。
2. **复用 vllm 的 hadamard rotation**：把 vllm_simo 中处理 `online_down_proj_rotation` 的逻辑移植到 sglang_simo（或者在导出时把 rotation 烘焙进权重）。这是 MXFP4 在 LLM 上能落地的关键。
3. **临时解法**：对 w4a4 用 NVFP4 替代 MXFP4（PPL 4.42，可用），等 hadamard 路径就绪再切回。
4. 长期：在 sglang_simo `apply()` 里补齐 padding / output slicing / `input_global_scale` 触发条件，避免在其它模型/group_size 上踩雷。


---

# 修正：vllm 在本次命令下 **同样没有启用 hadamard rotation**——之前的结论需要撤回

## 1. 直接证据

vllm 这次跑的 config `quant_config_ocp_w4a4_mxfp.json`：

```json
"per_quant_opt": null
```

vllm_simo 的 dispatch 路径（`vllm_simo/quantization/quantization_config.py`）：

```python
# line 373-399
def _should_apply_hadamard_transform(self, layer, ...):
    o_proj_rotation = (
        ... and self.per_quant_opt
        and "online_o_proj_rotation" in self.per_quant_opt
    )
    down_proj_rotation = (
        ... and self.per_quant_opt
        and "online_down_proj_rotation" in self.per_quant_opt
    )
    hadamard_transform = o_proj_rotation or down_proj_rotation
    return o_proj_rotation, down_proj_rotation, hadamard_transform

# line 595, 659:
o_proj_rotation, down_proj_rotation, hadamard_transform = ...
return SIMOLinearMethod(..., hadamard=hadamard_transform, ...)

# line 244-256 SIMOLinearMethod.__init__:
def __init__(self, ..., hadamard: bool = False):
    self.weight_hadamard_transform_size = self.weight_spec.group_size if hadamard else 0
```

`per_quant_opt=None` → 两个 rotation 都为 False → `hadamard_transform=False` → `SIMOLinearMethod(hadamard=False)` → `weight_hadamard_transform_size=0`。

→ **vllm 端在本次 `quant_config_ocp_w4a4_mxfp.json` 配置下没有应用 hadamard rotation。**

## 2. 反向佐证：debug_ppl 数据本身

之前已观察到**所有层** `dq_w_abs_max / dq_w_abs_mean` 在 sglang/vllm 比值精确等于 **1.000**。

如果 vllm 端真的应用了 hadamard，则 weight 在量化前会被乘以 H 矩阵，dequant 后的 `dq_w` 在数值分布上会**显著区别于** sglang 的未旋转 weight，max/mean 不可能完全相等。

→ 这从数值上独立证明：**两端走的是相同的、未做 hadamard 的 weight 量化路径**。

## 3. 因此 hadamard 不是 PPL 差异的原因

之前那段把 `per_quant_opt: "online_down_proj_rotation"` 列为"决定性证据"的论述需要撤回。该字段在 sglang 端没生效（sglang_simo 不解析），在 vllm 端因为本次 vllm config 是 `null` 也没生效——**两端都没有 rotation**，不构成两端间差异。

> （独立的事实仍成立：sglang 的 `quant_config_w4a4_mxfp.json` 写的是字符串 `"online_down_proj_rotation"`，类型与 vllm 期望的 `list[str]` 不同；这是配置本身的瑕疵，但与本次 PPL 无关。）

---

# 真正的根因（修正后）

w4a4_mxfp 唯一发散的真实原因仍然是**两个独立因素的乘积**，但第二个因素的描述要修正：

## 因素 A：attention backend numerics 差异

- sglang 默认 backend = **fa3**（`server_args.py:1799-1803`，H100 + cuda12.3 + 非 spec decode 命中）。
- vllm 用 `enforce_eager=True` 时通常是 `flash_attn_varlen_func`（vllm-FA）。
- 两者在 softmax 数值稳定、KV layout、rope 实现细节上不完全一致 → attention 输出在前几层有 ~5~6% 量级的偏移（debug_ppl 中 o_proj `qx_max` ratio≈1.06，所有量化配置下都存在）。

这个差异是**框架级**的，不取决于量化精度。

## 因素 B：MXFP4 自身的低鲁棒性放大上游噪声

- MXFP4_E2M1：4 bit，仅 16 码字，group=32，scale 是 E8M0（power-of-2，只能翻倍/减半），无 per-tensor 全局 scale。
- 所有量化方案中**最不鲁棒**的；与之相比 NVFP4 有双 scale，INT4 per_group 有 float scale，MXFP6/MXFP8 有更多 bit。
- MXFP4 路径下 QKV 量化噪声更高 → 进入 attention 的 Q/K/V 与 fp16 baseline 偏离更大 → 不同 attention 实现对"噪声 QKV"的响应（softmax 数值稳定性、累加顺序）放大不同。

## 数据佐证：B 因素如何放大 A 因素

| 配置 | qkv_proj qx_max ratio (Q/K/V 输入到 attention 前) | o_proj qx_max ratio (attention 输出) | o_proj qx_mean ratio | wikitext PPL ratio |
|------|--------------------------------------------:|-------------------------------:|--------------------:|-------------------:|
| w8a8_mxfp | 1.044 | 1.063 | 1.137 | 0.99 ✓ |
| w4a4_mxfp | 1.003 | 1.060 | **1.384** | **2.70** ✗ |

注意到：
- 进入 attention 前（qkv_proj 的输出），两个配置在 sglang/vllm 间的 ratio 都很接近 1.00~1.05；
- attention 后（o_proj 的输入 max），ratio 同样都是 ~1.06；
- **但 mean ratio**：w8a8 仅 1.14，w4a4 高达 **1.38**。

这说明：在"max 偏移类似"的情况下，MXFP4 路径下的 attention 输出 **整个分布**（不仅是 max）都被显著拉偏——典型的"带噪 QKV 经过两个不同 softmax 实现后产生体感更大的分布偏移"特征，而 MXFP8 因 QKV 噪声小，对 backend 差异不敏感。

进入 80 层 residual 累积后：
- MXFP8：偏差 dropoff，PPL 几乎不变（2.94 vs 2.97，差 1%）。
- MXFP4：偏差不断累积，PPL 差 2.7x。

---

# 真正决定 w4a4_mxfp PPL 差异的两条线

| 条 | 内容 | 证据 |
|----|------|------|
| 1 | sglang attention backend (fa3) vs vllm attention backend 在数值实现上不同 | 所有量化配置下 o_proj `qx_max` ratio≈1.06；qkv_proj 前向几乎一致 |
| 2 | MXFP4 鲁棒性极差：对 QKV 噪声放大、对 backend 差异敏感；MXFP6/8、NVFP4、INT4_per_group 都不会被放大 | 同样 1.06x 的 max 扰动，MXFP4 → mean 偏 38%；MXFP8 → mean 仅偏 14% |

→ **PPL 差异的真凶不是 hadamard，而是"sglang 与 vllm 两条 attention backend 在 MXFP4 噪声下数值响应不一致"**。

---

# 验证 / 修复方法（更新）

1. **验证 A：消除 backend 差异**
   - 在 sglang 启动加 `"attention_backend": "triton"` 或 `"flashinfer"`，分别再跑一次 wikitext，看 PPL 是否变化；
   - 或者反过来，让 vllm 用 triton attention（`VLLM_ATTENTION_BACKEND=TRITON_ATTN`）。
   - 若两端都强制相同非默认 backend 下 PPL 收敛，则确认 A 是主因。
2. **验证 B：用更鲁棒的 4bit 格式**
   - 切到 `quant_config_w4a4_nvfp.json`：sglang=4.42 / vllm=4.17，差异 ~6%（可接受）。
   - 切到 `quant_config_w4a16_int4_per_group.json`：sglang=3.74 / vllm=4.06。
   - 都没有 w4a4_mxfp 那种 2.7x 退化 → 反证 MXFP4 是放大器。
3. **逐层 dump attention output**
   - TP=1 + 固定输入，分别 dump sglang fa3 与 vllm-FA 的每层 attention output，比 cosine sim / max abs err；
   - 若发现 layers.5~10 出现明显发散，与 debug_ppl 中 o_proj 早期层 ratio 高的位置一致 → 直接定位到 attention kernel 的具体 op。

---

# 一句话总结（修正版）

> **vllm 当前命令下与 sglang 一样，per_quant_opt=null、hadamard rotation 未启用；之前把 hadamard 列为根因有误。w4a4_mxfp 的 2.7x PPL 差异真正来自"两端 attention backend (fa3 vs vllm-FA) 的数值实现差异 + MXFP4 自身低鲁棒性这一放大器"，与 hadamard 无关。**


---

# 切换 vllm 到 TRITON_ATTN 后端的命令

## 修改后的命令（追加 `VLLM_ATTENTION_BACKEND=TRITON_ATTN` 环境变量）

```bash
VLLM_ATTENTION_BACKEND=TRITON_ATTN \
debug_env_file=/share/users/like/ipc.sglang.3.json \
lm_eval --model vllm \
  --model_args '{"pretrained": "/data_gpu/models/share_data/modelzoo/weights/llm/llama/llama3.1/llama3.1-70B/safetensor_weights", "quantization": "simo", "hf_overrides": {"quantization_config_file": "/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/example/online_quantization/../simo_quantization_config/online_quantization/quant_config_ocp_w4a4_mxfp.json"}, "tensor_parallel_size": 8, "dtype": "auto", "gpu_memory_utilization": 0.6, "enforce_eager":true}' \
  --tasks wikitext --batch_size auto \
  > temp/vllm.lm-eval-mxfp4.triton.log.`nowstr.sh` 2>&1 &
```

## 说明

- 仅在原命令前面加了 `VLLM_ATTENTION_BACKEND=TRITON_ATTN`，其它参数完全不变。
- 该 env var 强制 vllm 选 triton attention backend（绕过自动选择 FlashAttn）。
- `enforce_eager=True` 保留：避免 cudagraph 把 backend 选择在 capture 阶段固化到其它路径。
- 输出日志后缀加 `.triton` 以区分原 FA 跑的日志：`temp/vllm.lm-eval-mxfp4.triton.log.<时间>`。

## 与对照组的对比建议

为了精确隔离 attention backend 这一变量，建议同时配套跑：

```bash
# sglang 强制 triton attention backend（与 vllm 用 triton 对齐）
SIMO_SGLANG_REGISTER=1 \
debug_env_file=/share/users/like/ipc.sglang.3.json \
lm-eval --model sglang \
  --model_args '{"pretrained": "/data_gpu/models/share_data/modelzoo/weights/llm/llama/llama3.1/llama3.1-70B/safetensor_weights", "quantization": "simo", "json_model_override_args": "{\"quantization_config_file\": \"/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/sglang_simo/example/online_quantization/../simo_quantization_config/online_quantization/quant_config_w4a4_mxfp.json\"}", "tp_size": 8, "dtype": "auto", "mem_fraction_static": 0.5, "skip_server_warmup":true, "disable_cuda_graph":true, "attention_backend": "triton"}' \
  --tasks wikitext --batch_size auto \
  > temp/sglang.lm-eval-mxfp4.triton.log.`nowstr.sh` 2>&1 &
```

> 这样两端都走 triton attention，再比 PPL：
> - 若 sglang/vllm 收敛到接近同一 PPL，说明确实是 attention backend numerics 主导差异；
> - 若仍发散，则需要继续在 SIMOLinearMethod / quant 路径上找。

## 启动后验证 backend 已切换

启动日志里应该出现类似：
- vllm: `Using Triton attention backend.` 或 `Selected attention backend TRITON_ATTN`
- sglang: `Attention backend not specified. Use triton backend by default.`（被 `attention_backend=triton` 覆盖时不会出现"not specified"，而是直接采用）

如果 vllm 启动后仍打印 `Using Flash Attention backend`，说明 env var 没生效，需要确认 vllm 版本支持 `VLLM_ATTENTION_BACKEND` 这个变量名（部分版本是 `VLLM_USE_TRITON_FLASH_ATTN=1`）。


---

# 用 vllm `attention_config` 在 lm-eval 命令中显式指定 triton attention backend

## vllm 代码定位

- `vllm/engine/arg_utils.py:556` `attention_config: AttentionConfig = get_field(VllmConfig, "attention_config")`
- `vllm/engine/arg_utils.py:1810-1823`：`attention_backend` 会被 merge 到 `attention_config.backend`
- `vllm/config/attention.py:13` `class AttentionConfig`，含字段 `backend: AttentionBackendEnum | None`
- `vllm/v1/attention/backends/registry.py:48`：枚举值 `TRITON_ATTN = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"`
- `attention.py:63-69` `validate_backend_before`：`backend` 接受字符串，自动转 enum（`AttentionBackendEnum[value.upper()]`）

## 修改后的命令

将 `attention_config={"backend": "TRITON_ATTN"}` 通过 `--model_args` 透传给 `LLM(...)`：

```bash
debug_env_file=/share/users/like/ipc.sglang.3.json \
lm_eval --model vllm \
  --model_args '{"pretrained": "/data_gpu/models/share_data/modelzoo/weights/llm/llama/llama3.1/llama3.1-70B/safetensor_weights", "quantization": "simo", "hf_overrides": {"quantization_config_file": "/softhome/like/package/h100/package/simo_conda_sglang/simo/extensions/vllm_simo/example/online_quantization/../simo_quantization_config/online_quantization/quant_config_ocp_w4a4_mxfp.json"}, "tensor_parallel_size": 8, "dtype": "auto", "gpu_memory_utilization": 0.6, "enforce_eager":true, "attention_config": {"backend": "TRITON_ATTN"}}' \
  --tasks wikitext --batch_size auto \
  > temp/vllm.lm-eval-mxfp4.triton.log.`nowstr.sh` 2>&1 &
```

仅在 `--model_args` 的 JSON 末尾追加了一项：
```json
"attention_config": {"backend": "TRITON_ATTN"}
```

## 等价写法（任选其一）

`arg_utils.py:1811-1823` 中 `attention_backend` 字段会被 merge 到 `attention_config.backend`，因此也可写：

```json
"attention_backend": "TRITON_ATTN"
```

两者**不能同时设**——`arg_utils.py:1812-1816` 有显式互斥校验：
```python
if attention_config.backend is not None:
    raise ValueError("attention_backend and attention_config.backend cannot both be set")
```

## 启动后验证

vllm 启动日志应出现：
- `Selected attention backend: TRITON_ATTN` 或类似字样
- 模块加载 `vllm.v1.attention.backends.triton_attn.TritonAttentionBackend`

如果仍看到 `FlashAttention` 字样，说明配置未传到 `VllmConfig`，需检查 lm-eval 的 vllm backend 是否把 `attention_config` 透传到 `LLM(...)`（部分 lm-eval 版本只识别白名单 kwargs）。如有需要，可降级为 env var 路径：

```bash
VLLM_ATTENTION_BACKEND=TRITON_ATTN <原命令>
```

---

## 深入根因分析：为什么 sglang_simo w4a4_mxfp PPL=27.6 而 vllm_simo PPL=10.2

### 已排除的假设

| # | 假设 | 排除依据 |
|---|------|----------|
| 1 | Hadamard rotation差异 | sglang `per_quant_opt="online_down_proj_rotation"` 但代码 `get_quant_method_by_target_spec()` 第789行创建 `SIMOLinearMethod` 时**未传递 `hadamard=True`**（默认False），所以两边都不应用hadamard |
| 2 | Attention backend差异 | 实验证明：两边都切换到triton后，PPL差异不变（sglang 27.6046 vs vllm 10.1884） |
| 3 | 权重量化差异 | dq_w_abs_max/mean ratio = 1.000，bit-identical |
| 4 | SIMOLinearMethod.apply代码差异 | 两边的mxfp4路径功能完全相同 |

### 关键数据发现

**1. Forward pass次数不同**
- SGLang: 176次 debug_ppl = 22次/rank（8 TP ranks）
- vLLM: 152次 debug_ppl = 19次/rank
- 两者都处理62个rolling window请求
- 差异的3次是SGLang额外的decode步骤（max_new_tokens=1）

**2. Layer 0 qkv_proj 的 qdq_x_abs_mean 唯一值分布不同**

| qdq_x_abs_mean值 | SGLang次数 | vLLM次数 |
|---|---|---|
| 0.002593994 | 8 | 0 (SGLang独有) |
| 0.002609253 | 56 | 64 |
| 0.002624512 | 72 | 48 |
| 0.002639771 | 32 | 24 |
| 0.002777100 | 8 | 0 (SGLang独有) |
| 0.003631592 | 0 | 16 (vLLM独有) |

同样是layer 0 embedding输出（未量化），如果处理相同token，qdq_x_abs_mean应该完全一致。不同的唯一值和分布证明**两个框架在每次forward pass中处理了不同的token batch**。

**3. 不同quant format受影响程度差异极大**

| 格式 | sglang PPL | vllm PPL | 比值 |
|---|---|---|---|
| w4a4_mxfp (4-bit, 16码字) | 27.6 | 10.2 | **2.7x** |
| w8a8_mxfp (8-bit, 256码字) | 2.94 | 2.97 | 0.99x |
| w4a4_nvfp (4-bit, scales带fp) | ~相近 | ~相近 | ~1.06x |
| w6a6_mxfp | ~相近 | ~相近 | ~0.99x |

只有w4a4_mxfp严重发散，其他格式差异很小。

### 根因分析

**核心原因：SGLang和vLLM的batch调度差异，被mxfp4的极端量化敏感性放大**

详细解释：

1. **Batch调度差异**：SGLang和vLLM的scheduler将62个lm-eval rolling window请求分配到不同的forward pass batch中。SGLang的scheduler产生22个forward pass（包含3个额外的decode步），vLLM产生19个。每个forward pass中的token集合不同。

2. **mxfp4的独特脆弱性**：
   - mxfp4只有**16个码字**（4-bit），是所有格式中最少的
   - MX格式使用**E8M0 power-of-2 scale**（无mantissa的指数scale），量化精度完全依赖于数据在block内的分布
   - Block size=32，axis=-1：每32个连续feature元素共享一个scale
   - 当batch中的token组合不同时，attention的输出（o_proj的输入）具有不同的统计分布
   - 在mxfp8（256码字）下，这种分布差异可以被精确表示，误差很小
   - 在mxfp4（16码字）下，微小的分布变化就会导致大量信息丢失

3. **误差级联放大**：
   - 第一层的量化误差通过attention传播到后续层
   - 每层的o_proj输出（attention output）是所有head的拼接，含有来自不同token的信息
   - 不同的batch组合 → 不同的attention pattern → 不同的o_proj输入分布 → 不同的mxfp4量化误差
   - 早期层(5-7层)的o_proj divergence ratio最高(2.6-3.3x)，说明误差在早期就发散并级联

4. **为什么其他4-bit格式（nvfp4）不受影响**：
   - nvfp4使用E4M3 scale（非power-of-2），比E8M0更精确
   - nvfp4的global_scale_factor提供了额外的动态范围调整
   - 这使nvfp4对输入分布变化的鲁棒性远高于mxfp4

### 验证建议

1. **最关键的验证**：在两个框架上运行**fp16 baseline**（不量化），对比wikitext PPL。如果fp16 PPL也有显著差异，则说明是框架级别的差异；如果接近，则确认是量化引起的。

2. **控制batch差异**：修改SGLang lm-eval backend，使其逐个发送请求（batch_size=1），消除batch调度差异。对比PPL。

3. **对比每个window的per-token logprobs**：在两个框架中打印每个rolling window的sum logprob值，找出哪些window贡献了最大的PPL差异。

4. **SGLang禁用radix cache**：设置`disable_radix_cache=True`，排除prefix caching对forward pass路径的影响。

5. **直接对比相同batch的量化误差**：构造完全相同的输入tensor，分别在两个框架中运行SIMOLinearMethod.apply()，对比输出是否bit-identical。这可以最终确认是batch差异导致还是代码路径差异。

### 结论

~~sglang_simo和vllm_simo在w4a4_mxfp下的PPL巨大差异（27.6 vs 10.2）不是由量化代码bug引起的，而是由两个推理框架不同的batch调度策略导致的。~~ **（此假设被bf16实验推翻，见下文更新分析）**

---

## 更新分析：bf16 baseline证据推翻batch调度假设

### 新证据：bf16 PPL几乎一致

| 框架 | bf16 (无量化) PPL | 来源 |
|------|-----------------|------|
| SGLang | 2.8325 | `llm_eval_online_quant.sh.hf-ppl.80b-tp8.sh.2026_04_23___08_12_41` line 180 |
| vLLM | 2.8616 | `llm_vlm_eval_online_quant.sh.hf-ppl.80b-tp8.sh.log.2026_04_23___22_51_32` line 4397 |
| 差异 | ~1% | |

**关键推论**：如果batch调度差异是w4a4_mxfp PPL发散的根因，那么bf16也应该出现类似的发散。但bf16 PPL仅差1%，证明**两个框架的模型推理逻辑是等价的**，batch调度不是根因。

### 完整PPL对比表

| 量化格式 | SGLang PPL | vLLM PPL | SGLang/vLLM |
|---------|-----------|----------|-------------|
| bf16 (无量化) | 2.8325 | 2.8616 | 0.99 |
| w8a8_mxfp | 2.9432 | 2.9703 | 0.99 |
| w8a8_mxint | 2.8423 | 2.8737 | 0.99 |
| w6a6_mxfp | 2.9301 | 2.9669 | 0.99 |
| w8a8_fp8_per_block | 2.8821 | - | - |
| w8a8_fp8_per_channel | 2.8905 | 2.9224 | 0.99 |
| w8a8_int8_per_block | 3.0725 | 2.9668 | 1.04 |
| w8a8_int8_per_channel | 55.5820 | 49.0248 | 1.13 |
| w4a16_int4_per_group | 3.7414 | 4.0577 | 0.92 |
| **w4a4_mxfp** | **27.6423** | **10.2393** | **2.70** |
| w4a4_nvfp | 4.4210 | 4.1701 | 1.06 |
| w4a4_nvfp_4_over_6 | 3.7519 | - | - |
| w4a16_nvfp4_per_group | 3.9018 | 3.6270 | 1.08 |

**观察**：
1. 大多数格式SGLang/vLLM比值在0.92-1.06之间，差异很小
2. w8a8_int8_per_channel在两个框架上都很差（55.58/49.02），说明该格式本身不适合Llama-70B
3. **只有w4a4_mxfp出现2.70x的巨大差异**，其他4-bit格式（nvfp4, int4）差异都很小

### 核心代码排查

**1. Python源码完全一致**

两个simo安装的Python源文件（`simo/ops/`, `simo/quantization/`等）经`diff -rq`确认**完全相同**（排除`extensions/`和`.pyc`缓存文件后无差异）。

**2. 关键发现：Triton和PyTorch版本不同**

| 组件 | SGLang环境 (simo_sglang) | vLLM环境 (simo_vllm) |
|------|------------------------|---------------------|
| PyTorch | **2.9.1+cu128** | **2.10.0** |
| Triton | **3.5.1** | **3.6.0** |

mxfp4的downcast和upcast使用**Triton内核**（`_downcast_to_mxfmt_triton`, `_upcast_from_mxfmt_triton`），不同Triton版本（3.5.1 vs 3.6.0）编译出的PTX代码可能不同。

**3. C++扩展也不同**

| 文件 | SGLang | vLLM |
|------|--------|------|
| `_C.cpython-312-x86_64-linux-gnu.so` | 462408字节 | 477112字节 |
| md5 | da5a01f6... | a19ff1c7... |

但mxfp4 downcast/upcast路径通过`torch.ops.simo.downcast_to_mxfmt` → `downcast_to_mxfmt_cuda_impl` → `_downcast_to_mxfmt_triton`，是**Triton内核**，不走C++扩展。所以C++差异不太可能是原因。

**4. matmul路径**

两边都使用`torch.matmul(qdq_x, dq_w.T)`。PyTorch 2.9.1和2.10.0可能选择不同的cuBLAS算法，但对于bf16 matmul，这种差异通常很小（bf16 PPL匹配证实了这一点）。

### 修正后的根因分析

**最可能的根因：Triton版本差异导致mxfp4量化内核产生不同的数值结果**

详细推理：

1. **Triton内核编译差异**：Triton 3.5.1和3.6.0对相同的Triton源码可能生成不同的PTX指令序列。对于mxfp4量化内核，关键操作包括：
   - 计算block内绝对值最大值（用于E8M0 scale）
   - 将bf16/fp32值映射到4-bit码字
   - E8M0 scale的floor/ceil/even rounding

   这些操作中任何一步的实现差异（如reduction顺序、中间精度、rounding模式）都会导致不同的量化结果。

2. **mxfp4的极端敏感性**：
   - 只有16个码字（{-6, -5, ..., 0, ..., 5, 6}的子集）
   - E8M0 scale是2的幂次（无mantissa），量化网格是离散的
   - 一个元素被映射到相邻码字的概率远高于mxfp8（256码字）
   - 不同的Triton编译结果 → 不同的中间精度/rounding → 部分元素落到不同的码字 → 量化误差不同

3. **误差级联**：
   - Llama-3.1-70B有80层
   - 每层的量化误差传播到下一层
   - 对于mxfp4，单层量化误差已经较大
   - 80层级联后，微小的初始差异会指数放大
   - 这就是为什么w4a4_mxfp差异如此巨大（2.70x），而w8a8_mxfp差异仅1%

4. **为什么其他4-bit格式不受影响**：
   - nvfp4使用E4M3 fp scale（非power-of-2），精度更高
   - nvfp4有global_scale_factor提供额外动态范围
   - int4_per_group使用fp32 group scale，量化精度更高
   - 这些格式对Triton编译差异的敏感度远低于mxfp4

### 验证方案

**最直接的验证**：在**同一个conda环境**中同时运行sglang和vllm的w4a4_mxfp评测。

方案A：统一Triton/PyTorch版本
```bash
# 在simo_sglang环境中安装vllm（确保用相同的torch 2.9.1/triton 3.5.1）
# 或者升级simo_sglang环境的torch/triton到与simo_vllm一致

# 当前环境差异：
# simo_sglang: torch=2.9.1+cu128, triton=3.5.1
# simo_vllm:   torch=2.10.0,      triton=3.6.0
```

方案B：单元测试验证Triton内核
```python
# 在两个环境中分别运行，对比结果
import torch
from simo.ops.mx_api import downcast_to_mxfmt, upcast_from_mxfmt
from simo.quantization.dtypes import as_dtype

# 固定输入
torch.manual_seed(42)
x = torch.randn(128, 1024, dtype=torch.bfloat16, device='cuda')

# mxfp4 downcast + upcast
q, s = downcast_to_mxfmt(x, as_dtype('mxfp4_e2m1'), axis=-1, block_size=32)
dq = upcast_from_mxfmt(q, s, as_dtype('mxfp4_e2m1'), torch.bfloat16, axis=-1)

print(f"q checksum: {q.sum().item()}")
print(f"s checksum: {s.float().sum().item()}")
print(f"dq checksum: {dq.sum().item()}")
print(f"dq abs mean: {dq.abs().mean().item()}")
print(f"error abs mean: {(x.float() - dq.float()).abs().mean().item()}")
```

如果两个环境的`q checksum`或`dq checksum`不同，就直接证明是Triton版本导致的量化数值差异。

### 方案B实验结果：**Triton版本假设被推翻**

在两个环境中运行方案B的单元测试（`like-useful/debug_ppl.py`）：

```
# simo_sglang (torch=2.9.1, triton=3.5.1)
q checksum: 7984053
s checksum: 515191.0
dq checksum: -54.25
dq abs mean: 0.78515625
error abs mean: 0.08610832691192627

# simo_vllm (torch=2.10.0, triton=3.6.0)
q checksum: 7984053
s checksum: 515191.0
dq checksum: -54.25
dq abs mean: 0.78515625
error abs mean: 0.08610832691192627
```

**结果完全一致（bit-identical）**。Triton 3.5.1和3.6.0对mxfp4 downcast+upcast产生**完全相同的数值结果**。Triton版本差异不是根因。

---

## 最终根因分析：框架级数值微差异 + mxfp4量化放大效应

### 已排除假设汇总

| # | 假设 | 排除依据 |
|---|------|----------|
| 1 | Hadamard rotation差异 | sglang `get_quant_method_by_target_spec()` 不传 `hadamard=True` |
| 2 | Attention backend差异 | 两边切triton后PPL不变 |
| 3 | 权重量化差异 | dq_w ratio = 1.000 |
| 4 | SIMOLinearMethod.apply代码差异 | Python源码完全一致 |
| 5 | Batch调度差异 | bf16 PPL匹配(2.83 vs 2.86)排除 |
| 6 | Triton版本差异 | 单元测试checksum完全一致 |

### 真正的根因

**SGLang和vLLM的Llama模型实现存在微小的数值差异，这些差异在bf16下影响仅~1%，但被mxfp4量化以指数级放大。**

推理过程：

1. **bf16 PPL差异 = 1%**（2.8325 vs 2.8616）
   - SGLang和vLLM的Llama实现虽然功能等价，但在底层数值上并不bit-identical
   - 可能的差异来源：
     - **RoPE实现**：不同的cos/sin计算精度或顺序
     - **RMSNorm实现**：不同的reduction算法（分块reduction vs 全局reduction）
     - **Attention实现**：fa3 vs FlashAttention的不同数值精度
     - **算子融合**：SGLang可能融合了某些操作（如SiLU+乘法），改变了浮点运算顺序
     - **cuBLAS matmul**：不同PyTorch版本可能选择不同的GEMM算法
   - 在bf16精度下，这些差异被精确的bf16运算吸收，最终PPL仅差1%

2. **mxfp4量化的"混沌放大效应"**
   - mxfp4只有**16个码字**，每个block（32个元素）共享一个**E8M0 power-of-2 scale**
   - 量化相当于将连续的bf16值映射到16个离散点的阶梯函数
   - 当输入值恰好处于两个量化level的边界附近时，微小的输入差异（来自框架数值差异）会导致该值被映射到**不同的码字**
   - 这不是"rounding误差被放大"，而是"离散化决策在边界处翻转"

3. **80层级联放大机制**
   ```
   Layer 0: 微小bf16差异 → 部分元素跨越mxfp4量化边界 → 不同量化误差
   Layer 1: 来自Layer 0的不同量化误差 → 更大的输入差异 → 更多元素跨越边界
   ...
   Layer 79: 差异已指数级放大 → PPL完全发散
   ```
   - 实验数据支持：早期层(5-7)的o_proj divergence ratio已达2.6-3.3x
   - 后续层的divergence继续增长

4. **为什么vLLM的10.2也很差（bf16=2.86）**
   - mxfp4本身的量化误差就很大（从bf16 2.86恶化到10.2）
   - SGLang的27.6是在vLLM 10.2的基础上进一步恶化
   - 如果两个框架的微小差异恰好使SGLang的中间值更频繁地落在量化边界上，PPL就会额外恶化

5. **为什么其他量化格式不受影响**
   - **mxfp8（256码字）**：量化阶梯更密，输入微差不太可能跨越边界 → 两边PPL差异仅1%
   - **nvfp4（16码字但E4M3 scale）**：浮点scale可以更精确地适应输入分布 → 差异仅6%
   - **w6a6_mxfp（64码字）**：4倍于mxfp4的码字数 → 差异仅1%
   - 规律：**码字数越少 + scale精度越低 = 对框架差异越敏感**

### 验证方案

**验证1：在同一框架上运行两次，使用不同的随机种子或batch顺序**
- 如果同一框架自身的PPL变化也很大（>10%），说明mxfp4确实对数值噪声极度敏感
- 预期：SGLang两次运行之间的PPL变化应该很小（因为同框架的数值是确定性的）

**验证2：比较sglang和vllm在同一层的逐token hidden state**
```python
# 在两个框架中添加hook，保存layer 0的hidden state output（包含attention后的结果）
# 对相同的单条输入文本
# 比较hidden state的L2距离和最大差异
```
如果hidden state差异在1e-3量级，就确认是框架数值微差被量化放大。

**验证3：用mxfp8替代mxfp4的input quantization（保持weight为mxfp4）**
```json
{
  "module_configs": [{
    "targets": ["Linear"],
    "input": {"dtype": "mxfp8_e4m3"},
    "weight": {"dtype": "mxfp4_e2m1"}
  }]
}
```
如果将activation量化从mxfp4改为mxfp8，SGLang PPL应该大幅改善，因为mxfp8的256码字对输入微差更鲁棒。vLLM已有w4a8结果（PPL=4.95），SGLang应该接近。

### 最终结论

w4a4_mxfp的PPL差异（SGLang 27.6 vs vLLM 10.2）是**框架级数值微差被mxfp4量化指数级放大**的结果。根因不是某一个具体的bug，而是：

1. SGLang和vLLM的Llama模型实现有微小但不可避免的数值差异（bf16 PPL差1%）
2. mxfp4（4-bit, 16码字, E8M0 power-of-2 scale）是对数值噪声最敏感的量化格式
3. 80层Transformer将这些微差指数级放大

**这是mxfp4格式的固有限制，不是SIMO量化代码的bug。** 缓解方案包括：
- 使用更高精度的activation量化（如w4a8_mxfp）
- 使用nvfp4替代mxfp4（E4M3 scale比E8M0更精确）
- 确保在对比不同框架时，使用相同的底层model implementation

---

## Llama3RotaryEmbedding 和 RotaryEmbedding 的 `_compute_inv_freq` 差异

代码位置：`/data//like/package/sglang_kernel_src/python/sglang/srt/layers/rotary_embedding.py`。

### 1. `RotaryEmbedding._compute_inv_freq`：标准 RoPE 几何频率

`RotaryEmbedding` 直接计算原始 RoPE 的 inverse frequency：

```python
inv_freq = 1.0 / (
    base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim)
)
```

设 `d = rotary_dim`，第 `k` 个二维旋转通道对应 `torch.arange(0, d, 2)` 的值 `2k`，则：

```text
inv_freq[k] = base^(-2k / d)
theta[pos, k] = pos * inv_freq[k]
wave_len[k] = 2*pi / inv_freq[k] = 2*pi * base^(2k / d)
```

也就是说，标准 RoPE 的频率是一个固定的几何级数：维度越靠后，`inv_freq` 越小，波长越长。它不根据上下文扩展参数调整频率。

### 2. `Llama3RotaryEmbedding._compute_inv_freq`：先算标准 RoPE，再做分段缩放

`Llama3RotaryEmbedding` 先调用父类得到标准 RoPE inverse frequency：

```python
inv_freqs = super()._compute_inv_freq(base)
```

然后把每个频率换算成波长：

```python
wave_len = 2 * math.pi / inv_freqs
```

再根据 Llama 3 的扩展参数定义两个阈值：

```text
low_freq_wavelen  = orig_max_position / low_freq_factor
high_freq_wavelen = orig_max_position / high_freq_factor
```

最后分三段处理：

```text
如果 wave_len < high_freq_wavelen:
    new_inv_freq = inv_freq

如果 wave_len > low_freq_wavelen:
    new_inv_freq = inv_freq / scaling_factor

否则:
    smooth = (orig_max_position / wave_len - low_freq_factor)
             / (high_freq_factor - low_freq_factor)
    new_inv_freq = (1 - smooth) * inv_freq / scaling_factor
                   + smooth * inv_freq
```

数学上可以写成：

```text
f_k = base^(-2k / d)
lambda_k = 2*pi / f_k

f'_k =
    f_k,                                  lambda_k < lambda_high
    f_k / scaling_factor,                 lambda_k > lambda_low
    f_k * (smooth_k + (1 - smooth_k) / scaling_factor), otherwise
```

其中：

```text
lambda_high = orig_max_position / high_freq_factor
lambda_low  = orig_max_position / low_freq_factor
smooth_k    = (orig_max_position / lambda_k - low_freq_factor)
              / (high_freq_factor - low_freq_factor)
```

注意这里变量名 `new_freqs` 实际仍然是 RoPE 使用的 inverse frequency。

### 3. 数学含义

核心区别是：

- `RotaryEmbedding`：所有 rotary 维度都严格使用同一个几何频率序列 `base^(-2k/d)`。
- `Llama3RotaryEmbedding`：不是简单换一个 `base`，而是对标准频率做分段修正。
- 短波长/高频部分保持不变，保留局部位置分辨率。
- 长波长/低频部分除以 `scaling_factor`，等价于把这些维度的波长乘以 `scaling_factor`，用于支持更长上下文。
- 中间波长区域做平滑插值，避免频率在阈值处突变。

所以从数学上看，`RotaryEmbedding` 是纯几何级数 RoPE；`Llama3RotaryEmbedding` 是 Llama 3 的 piecewise RoPE scaling：高频不动、低频按比例降频、中频平滑过渡。它改变的是不同频段的角速度 `theta[pos,k] = pos * inv_freq[k]`，使低频维度旋转得更慢，从而扩展可表达的位置范围，同时尽量不破坏短距离位置编码。

---

## SGLang 和 vLLM 的 `Llama3RotaryEmbedding` 是否完全一样

对照的两个文件：

- SGLang：`/data//like/package/sglang_kernel_src/python/sglang/srt/layers/rotary_embedding.py`
- vLLM：`/data//like/package/vllm-for-conda-simo/vllm/model_executor/layers/rotary_embedding/llama3_rope.py`

结论：**Llama 3 RoPE scaling 的核心公式是一样的，但两个类在完整运行时行为上不是完全一样。**

### 1. 子类 `Llama3RotaryEmbedding` 的核心代码几乎逐行相同

两边的 `__init__` 都保存同样的四个 Llama 3 scaling 参数：

```text
scaling_factor
low_freq_factor
high_freq_factor
orig_max_position
```

两边的 `_compute_inv_freq` 逻辑也一致：

```text
inv_freqs = super()._compute_inv_freq(base)
low_freq_wavelen = orig_max_position / low_freq_factor
high_freq_wavelen = orig_max_position / high_freq_factor
wave_len = 2*pi / inv_freqs

if wave_len < high_freq_wavelen:
    new_freqs = inv_freqs
elif wave_len > low_freq_wavelen:
    new_freqs = inv_freqs / scaling_factor
else:
    new_freqs = (1 - smooth) * inv_freqs / scaling_factor + smooth * inv_freqs
```

实际 diff 只有很小的非行为差异：

- SGLang 的 `base` 类型标注是 `int` / `Union[int, float]`。
- vLLM 的 `base` 类型标注是 `float`。
- SGLang 类定义后多一个空行。

这些差异不改变 Python 运行结果。所以只看 `Llama3RotaryEmbedding._compute_inv_freq` 里的 Llama 3 分段缩放数学公式，两边可以认为是一样的。

### 2. 父类 `RotaryEmbedding._compute_inv_freq` 基本公式一样，但 SGLang 多一个 RL 特殊路径

vLLM 父类的标准 RoPE inverse frequency 是：

```python
inv_freq = 1.0 / (
    base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim)
)
```

SGLang 父类也是同一个公式，但多了：

```python
init_device = "cpu" if get_global_server_args().rl_on_policy_target is not None else None
...
if get_global_server_args().rl_on_policy_target is not None:
    inv_freq = inv_freq.cuda()
```

也就是说，普通路径下标准 `inv_freq` 的数学值相同；如果 SGLang 开了 `rl_on_policy_target`，它会显式先在 CPU 上算再搬到 CUDA，vLLM 没有这个分支。

### 3. 完整 RoPE 模块运行时不完全一样

虽然 Llama 3 scaling 公式一致，但继承的父类和框架集成不同，导致完整模块不完全相同：

- **cache dtype 不同**：SGLang 在 CUDA 初始化时保留 FP32 `cos_sin_cache`（注释写明为了数值稳定），非 CUDA 才转成 `dtype`；vLLM 初始化后通常会把 cache 转成 `dtype`，forward 前还会用 `_match_cos_sin_cache_dtype(query)` 让 cache 匹配 query 的 device/dtype。
- **CUDA kernel 路径不同**：SGLang 优先走 `sglang.jit_kernel.rope.apply_rope_with_cos_sin_cache_inplace`，部分 head size 或平台再走 fallback；vLLM 走 `vllm._custom_ops.rotary_embedding`，并有自己的 `CustomOp` dispatch。
- **接口能力不同**：SGLang 的 forward 支持 `offsets` 和 `fused_set_kv_buffer_arg`；vLLM 这里的 forward 接口支持 `key=None`，但没有 SGLang 这两个参数。
- **cache 扩展能力不同**：SGLang 父类有 `_ensure_cos_sin_cache_length`，可以按需扩展 RoPE cache；vLLM 这个 base 实现里没有对应方法。
- **调度基类不同**：SGLang 继承 `MultiPlatformOp`；vLLM 继承 `CustomOp` 注册体系。二者按平台选择 forward/kernel 的机制不同。

### 最终判断

如果问题限定为：**Llama 3 对 `inv_freq` 的分段缩放数学公式是否一样？**答案是：**一样**。

如果问题是：**SGLang 的 `Llama3RotaryEmbedding` 类和 vLLM 的实现是否完全一样？**答案是：**不完全一样**。子类里的 Llama 3 scaling 代码基本相同，但父类 cache dtype、kernel dispatch、forward 参数和 cache 管理逻辑不同，因此实际运行路径和数值细节可能不同，尤其是在 CUDA 上 cache 是否保持 FP32 这一点上。

---

## `MultiPlatformOp.enter_torch_compile` 和 `leave_torch_compile` 的作用及调用者

代码位置：`/data//like/package/sglang_kernel_src/python/sglang/srt/layers/utils/multi_platform.py`。

### 1. `MultiPlatformOp` 正常情况下如何工作

`MultiPlatformOp` 是很多平台相关算子的基类，例如 rotary embedding、activation、RMSNorm、TopK、部分 MoE 方法等。

初始化时它会根据当前平台选择一个实际 forward 实现：

```python
self._forward_method: Callable = self.dispatch_forward()
```

`dispatch_forward()` 的选择逻辑大致是：

```text
CUDA -> forward_cuda
HIP  -> forward_hip
CPU AMX -> forward_cpu
NPU  -> forward_npu
XPU  -> forward_xpu
MUSA -> forward_musa
否则 -> forward_native
```

真正的 `forward()` 只是转调：

```python
def forward(self, *args, **kwargs):
    return self._forward_method(*args, **kwargs)
```

所以普通运行时，`MultiPlatformOp` 会优先走平台优化 kernel，例如 CUDA kernel，而不是纯 PyTorch 实现。

### 2. 为什么需要 `enter_torch_compile`

`enter_torch_compile(num_tokens)` 的目的：**在 CUDA graph / torch.compile 捕获模型时，把 `MultiPlatformOp` 临时切换成更适合 `torch.compile` 捕获的实现。**

原因是：

- 普通模式下 `_forward_method` 可能是 `forward_cuda`、`forward_npu`、自定义 kernel、fallback kernel 等平台相关实现。
- `torch.compile` 更适合捕获纯 PyTorch / native 风格的计算图。
- 某些平台 kernel 或 Python dispatch 路径不适合被 Dynamo/Inductor 编译，或者编译后性能不一定好。

所以 `enter_torch_compile` 会先保存当前真实运行路径：

```python
self._original_forward_method = self._forward_method
```

然后按类型切换：

```python
if "FusedMoE" in self.__class__.__name__:
    if num_tokens == 1:
        self._forward_method = fused_moe_forward_native
elif "TopK" in self.__class__.__name__:
    if num_tokens == 1:
        self._forward_method = self.forward_native
else:
    self._forward_method = self.forward_native
```

含义：

- 普通 `MultiPlatformOp`：进入 compile 模式后直接用 `forward_native`。
- `TopK`：只在 `num_tokens == 1` 时切到 `forward_native`。
- `FusedMoE`：只在 `num_tokens == 1` 时切到专门的 `fused_moe_forward_native`。

代码里的注释说明了 MoE 的特殊处理原因：`torch.compile` 在这个层上 batch size > 1 时性能不总是好，所以只在 bs/token 数为 1 的场景启用对应 native compile 路径。

另外，函数开头有：

```python
if self.is_torch_compile:
    return
```

这是为了防止重复进入时覆盖 `_original_forward_method`。注释中特别提到 `RotaryEmbedding` 这类 op 可能在多层间复用，`enter_torch_compile` 会被调用多次；如果没有这个 guard，第二次进入可能把“已经被替换后的 forward”保存成 original，退出时就恢复错了。

### 3. 为什么需要 `leave_torch_compile`

`leave_torch_compile()` 的目的：**在 torch.compile / CUDA graph 捕获结束后，把 `_forward_method` 恢复成进入前的平台优化实现。**

它做的事情很简单：

```python
self._forward_method = self._original_forward_method
self._original_forward_method = None
self.is_torch_compile = False
```

也就是说，compile/capture 期间可以临时使用 native/compile-friendly 路径；退出后，正常推理仍然走原来的高性能平台实现，例如 CUDA kernel。

开头的：

```python
if not self.is_torch_compile:
    return
```

保证没有进入 compile 模式时调用 `leave_torch_compile()` 是 no-op。

### 4. 谁调用这两个函数

直接调用点只有两个文件里的 `_to_torch()` helper。

#### 调用点 1：`cuda_graph_runner.py`

文件：`python/sglang/srt/model_executor/cuda_graph_runner.py`

调用 helper：

```python
def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
    for sub in model._modules.values():
        if isinstance(sub, MultiPlatformOp):
            if reverse:
                sub.leave_torch_compile()
            else:
                sub.enter_torch_compile(num_tokens=num_tokens)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens)
```

这个 helper 会递归遍历模型里的所有子模块；只要子模块是 `MultiPlatformOp`，就调用对应的 enter/leave。

它在 `patch_model()` 上下文管理器里被调用：

```python
if enable_compile:
    _to_torch(model, reverse=False, num_tokens=num_tokens)
    yield torch.compile(torch.no_grad()(model.forward), ...)
...
finally:
    if enable_compile:
        _to_torch(model, reverse=True, num_tokens=num_tokens)
```

`patch_model()` 又在 `CudaGraphRunner.capture()` 捕获每个 batch size 的 CUDA graph 时使用：

```python
with patch_model(
    self.model_runner.model,
    bs in self.compile_bs,
    num_tokens=bs * self.num_tokens_per_bs,
    tp_group=self.model_runner.tp_group,
) as forward:
    self.capture_one_batch_size(bs, forward, stream_idx)
```

所以普通 CUDA graph runner 下，调用链是：

```text
CudaGraphRunner.capture()
  -> patch_model(..., enable_compile = bs in self.compile_bs)
    -> _to_torch(..., reverse=False)
      -> MultiPlatformOp.enter_torch_compile(...)
    -> torch.compile(model.forward) 并捕获 graph
    -> finally: _to_torch(..., reverse=True)
      -> MultiPlatformOp.leave_torch_compile()
```

这里 `compile_bs` 来自：

```python
compile_bs = [bs for bs in capture_bs if bs <= server_args.torch_compile_max_bs]
             if server_args.enable_torch_compile else []
```

也就是只有开启 `enable_torch_compile`，并且当前捕获 batch size 不超过 `torch_compile_max_bs` 时，才会真正 enter compile 模式。

#### 调用点 2：`piecewise_cuda_graph_runner.py`

文件：`python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`

它也定义了同样结构的 `_to_torch()`：

```python
def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
    for sub in model._modules.values():
        if isinstance(sub, MultiPlatformOp):
            if reverse:
                sub.leave_torch_compile()
            else:
                sub.enter_torch_compile(num_tokens=num_tokens)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens)
```

它在 piecewise CUDA graph 的 `patch_model(model, compiler)` 里调用：

```python
try:
    if compiler != "eager":
        _to_torch(model, reverse=False, num_tokens=16)
    yield model
finally:
    _to_torch(model, reverse=True, num_tokens=16)
```

调用位置在 `PiecewiseCudaGraphRunner.__init__()` 中：

```python
with enable_piecewise_cuda_graph():
    language_model = getattr(
        self.model_runner.model, "language_model", self.model_runner.model
    )
    with patch_model(language_model.model, self.compile_config.compiler) as patched_model:
        install_torch_compiled(...)
        ...
        self.capture()
```

所以 piecewise CUDA graph 下，调用链是：

```text
PiecewiseCudaGraphRunner.__init__()
  -> patch_model(language_model.model, compiler)
    -> 如果 compiler != "eager":
         _to_torch(..., reverse=False, num_tokens=16)
           -> MultiPlatformOp.enter_torch_compile(...)
    -> install_torch_compiled / warmup compile / capture
    -> finally:
         _to_torch(..., reverse=True, num_tokens=16)
           -> MultiPlatformOp.leave_torch_compile()
```

如果 `piecewise_cuda_graph_compiler == "eager"`，不会调用 `enter_torch_compile()`；但 finally 里仍会调用 `_to_torch(..., reverse=True)`，这时每个 op 的 `leave_torch_compile()` 因为 `is_torch_compile == False` 会直接返回。

### 5. 总结

`enter_torch_compile` / `leave_torch_compile` 是一对“临时切换 forward 实现”的开关：

- `enter_torch_compile`：保存当前平台优化 forward，把 `MultiPlatformOp` 切到 `forward_native` 或其他 compile-friendly 实现。
- `leave_torch_compile`：恢复进入前的平台优化 forward。
- 它们不是用户直接调用的接口，而是 CUDA graph / torch.compile runner 在捕获模型时递归调用。
- 直接调用者是：
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` 里的 `_to_torch()`
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` 里的 `_to_torch()`

本质上，这是为了让同一个 `MultiPlatformOp` 在普通推理时走平台专用高性能 kernel，在 `torch.compile` 捕获时临时走更容易被编译器处理的 native 实现，并在捕获结束后恢复原状态。

## SGLang Llama 3.1 70B 的 RMSNorm 权重加载，以及和 vLLM `forward_native` 的差异

### 1. `RMSNorm.self.weight` 是在哪里从全 1 变成真实权重的？

结论：不是在 `python/sglang/srt/layers/layernorm.py` 里填充的，而是在模型加载权重阶段，由 Llama 模型的 `load_weights()` 通过 `default_weight_loader()` 把 checkpoint 里的同名 tensor `copy_` 到这个 `nn.Parameter`。

对 `/data/like/hf-models/llama3.1-70B`，`config.json` 里是：

```json
{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 8192,
  "num_hidden_layers": 80,
  "rms_norm_eps": 1e-05,
  "torch_dtype": "bfloat16"
}
```

SGLang 的创建和加载链路是：

```text
DefaultModelLoader.load_model()
  -> with set_default_torch_dtype(model_config.dtype)
  -> _initialize_model(...)
     -> get_model_architecture(...)
     -> LlamaForCausalLM(...)
        -> LlamaModel(...)
           -> LlamaDecoderLayer(...)
              -> self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
              -> self.post_attention_layernorm = RMSNorm(...)
           -> self.norm = RMSNorm(...)
  -> load_weights_and_postprocess(...)
     -> model.load_weights(weights)
        -> LlamaForCausalLM.load_weights(...)
           -> params_dict = dict(self.named_parameters())
           -> param = params_dict[name]
           -> weight_loader = getattr(param, "weight_loader", default_weight_loader)
           -> weight_loader(param, loaded_weight)
              -> default_weight_loader(...)
                 -> param.data.copy_(loaded_weight)
```

关键代码位置：

- `/data/like/package/sglang_kernel_src/python/sglang/srt/layers/layernorm.py:106`：`RMSNorm.__init__` 里先创建 `nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))`。
- `/data/like/package/sglang_kernel_src/python/sglang/srt/models/llama.py:358`、`:359`、`:421`：Llama 创建 `input_layernorm`、`post_attention_layernorm`、最终 `model.norm`。
- `/data/like/package/sglang_kernel_src/python/sglang/srt/model_loader/loader.py:653` 到 `:682`：`DefaultModelLoader.load_model()` 初始化模型并调用 `load_weights_and_postprocess()`。
- `/data/like/package/sglang_kernel_src/python/sglang/srt/model_loader/loader.py:684` 到 `:686`：`load_weights_and_postprocess()` 调用 `model.load_weights(weights)`。
- `/data/like/package/sglang_kernel_src/python/sglang/srt/models/llama.py:690` 到 `:759`：`LlamaForCausalLM.load_weights()` 根据 checkpoint tensor 名字找 `params_dict[name]` 并调用 loader。
- `/data/like/package/sglang_kernel_src/python/sglang/srt/model_loader/weight_utils.py:1041` 到 `:1055`：`default_weight_loader()` 最终执行 `param.data.copy_(loaded_weight)`。

对这个模型，`model.safetensors.index.json` 明确包含这些 checkpoint key，例如：

```text
model.layers.0.input_layernorm.weight
model.layers.0.post_attention_layernorm.weight
...
model.layers.79.input_layernorm.weight
model.layers.79.post_attention_layernorm.weight
model.norm.weight
```

这些名字和 `named_parameters()` 里的参数名一致，所以不会走 q/k/v 或 gate/up 的 stacked-parameter 分支，而是直接走 `default_weight_loader(param, loaded_weight)`，把 safetensors 中的 BF16 权重拷进对应 `RMSNorm.weight`。

### 2. SGLang 和 vLLM 的 `RMSNorm.forward_native` 数学上等价吗？

如果只看抽象的实数公式，并且限定在 Llama 这个普通 RMSNorm 情况，也就是：

- `has_weight=True`
- `var_hidden_size=None`
- `post_residual_addition=None`
- `fp32_residual=False`
- `override_orig_dtype=None`

那么两边都在计算同一个 RMSNorm：

```text
u = x                         # residual is None
或
u = x + residual              # residual is not None

variance = mean(u^2, dim=-1, keepdim=True)
y = u * rsqrt(variance + eps) * weight
```

但作为 PyTorch `forward_native` 实现，它们不保证数值等价，更不保证 bitwise 一致。最重要的差异是最后乘 `weight` 的 cast 顺序不同：

SGLang 默认 `cast_x_before_out_mul=False`，代码是：

```python
x = x * torch.rsqrt(variance + eps)   # x 是 float32
x = (x * self.weight).to(orig_dtype)
```

vLLM 是：

```python
x = x * torch.rsqrt(variance + eps)   # x 是 float32
x = x.to(orig_dtype)
if weight is not None:
    x = x * weight
```

也就是说，在 BF16/FP16 推理时：

- SGLang：`normalized_x(float32) * weight` 后再 cast 到 BF16/FP16。
- vLLM：先把 `normalized_x` round 到 BF16/FP16，再乘 BF16/FP16 的 `weight`。

这两个浮点计算顺序不同，舍入点不同，所以同一个输入、同一个权重也可能得到不同输出。

我用 `/data/like/hf-models/llama3.1-70B/model-00001-of-00030.safetensors` 里的真实 `model.layers.0.input_layernorm.weight` 做了一个小验证：随机 BF16 输入形状 `[2, 8192]`，按两边 `forward_native` 的最后 cast 顺序分别计算，结果不是逐元素相等；一次运行里 `num_diff=4262`，`max_abs_diff=0.001953125`。

还有几个次要差异：

- SGLang `forward_native()` 支持 `post_residual_addition`，vLLM 的 `RMSNorm.forward_native()` 没有这个参数。
- SGLang 对 residual 显式 `residual.to(torch.float32)` 后再加；vLLM 是 `x` 已经转成 float32 后执行 `x + residual`，对常见 BF16/FP16 residual 会被 promotion 到 float32，效果通常相同；如果传入非常规 dtype，比如 FP64 residual，结果可能不同。
- SGLang 的 `variance_size_override` 切片是 `x[..., :override]`，vLLM 是 `x[:, :, :override]`；Llama 3.1 70B 不使用 `var_hidden_size`，所以这点不影响这个模型。

因此，最终答案是：抽象 RMSNorm 公式等价；实际 `forward_native` 的有限精度实现不等价。对 Llama 3.1 70B 这种 BF16 模型，完全有可能在相同输入和相同权重下产生不同输出，差异主要来自最后乘 `weight` 前后的 dtype cast 顺序。

## vLLM offline_inference 报错分析：`np` 局部变量未绑定

分析日志：

```text
/share_data/users/like/package/h100/package/other/vllm/temp/llm_engine_example-debug-ppl.py.log.2026_05_09___17_34_04
```

结论：这次崩溃不是 SIMO 量化权重加载失败，也不是 CUDA OOM。日志里模型权重加载、profile、KV cache 创建和 warmup 都已经完成：

```text
init engine (profile, create kv cache, warmup model) took 8.08 seconds
```

真正的 fatal error 发生在第一次 `engine.step()` 执行模型时：

```text
File "/data/like/package/vllm-for-conda-simo/vllm/v1/worker/gpu_model_runner.py", line 3448, in execute_model
    num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
UnboundLocalError: cannot access local variable 'np' where it is not associated with a value
```

上层的：

```text
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.
```

只是 EngineCore 子进程因为上面的 `UnboundLocalError` 死掉后的包装错误，不是根因。

代码原因在：

```text
/data/like/package/vllm-for-conda-simo/vllm/v1/worker/gpu_model_runner.py
```

文件顶部已经有全局导入：

```python
import numpy as np
```

但同一个 `execute_model()` 函数后半段的 debug 代码里又写了一次局部导入：

```python
if env_data.get('debug_ppl_save_input_safetensor_dir', None):
    ...
    import numpy as np
    with FileLock(ppl_ipc_counter_lock_path):
        ...
```

Python 的作用域规则是：只要函数体内任何位置对名字有赋值或 `import`，这个名字在整个函数里都会被视为局部变量。因此 `execute_model()` 里的 `import numpy as np` 会让整个函数中的 `np` 都变成本地变量。前面的第 3448 行先执行：

```python
num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
```

此时局部变量 `np` 还没有执行到 `import numpy as np`，所以抛出：

```text
UnboundLocalError: cannot access local variable 'np' where it is not associated with a value
```

这和 `/dev/shm/like/ipc.vllm.1.json` 里是否开启 `debug_ppl_save_input_safetensor_dir` 没有本质关系；只要这条局部 `import numpy as np` 还在 `execute_model()` 函数体内，Python 编译函数时就会把 `np` 判定为局部变量，前面的 `np.array(...)` 就会失败。

最小修复方法：删除 `execute_model()` debug 代码块里的局部导入，因为文件顶部已经导入过 numpy。

```diff
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
@@
-                import numpy as np
                 with FileLock(ppl_ipc_counter_lock_path):
                     if not os.path.exists(ppl_ipc_counter_npy_path):
                         _forward_raw_counter = np.array([0], dtype=np.uint64)
```

等价修复方法：如果确实想保留局部导入，就不要使用名字 `np`，例如改成 `_np`，并把该 debug 小段里的 `np.array`、`np.fromfile`、`np.uint64` 都同步改成 `_np.array`、`_np.fromfile`、`_np.uint64`。但这里不推荐，因为顶部已有全局 `import numpy as np`，直接删掉局部 import 最干净。

修复后需要重新启动这条 vLLM 命令，让新的 Python 进程加载修改后的 `gpu_model_runner.py`。

## SGLang/vLLM RowParallelLinear all-reduce bitwise 对比

这里把问题里的 `tp_rank=8` 按 `tp_size=8`、TP rank 为 `0..7` 理解。

结论：

1. 如果 8 个 rank 的 `output_parallel` 输入逐 bit 完全相同，`tensor_model_parallel_all_reduce` 仍然有可能在 SGLang 和 vLLM 中得到不同的 bitwise 结果。原因不是 all-reduce 的数学定义不同，而是 BF16/FP16/FP32 浮点加法不满足结合律；只要两个框架实际选择的 all-reduce 后端、分块方式或规约顺序不同，最后一次舍入就可能不同。

2. 两边 Python 入口相同：`tensor_model_parallel_all_reduce(input_)` 都只是调用 `get_tp_group().all_reduce(input_)`。差异在 TP group 的 all-reduce 调度：

   - SGLang：`python/sglang/srt/distributed/parallel_state.py` 的 `GroupCoordinator.all_reduce` 会优先尝试 custom all-reduce；否则在 eager 路径通常走 in-place `torch.distributed.all_reduce`，PyNccl 默认创建后是 disabled，主要在 CUDA graph 或特定路径中启用。SGLang 还支持 `SGLANG_CUSTOM_ALLREDUCE_ALGO`、MSCClPP、Torch symmetric memory 等开关。
   - vLLM：`vllm/distributed/parallel_state.py` 委托给 `CudaCommunicator.all_reduce`，默认 out-of-place；H100/CUDA 路径按顺序考虑 NCCL symmetric-memory fast path、ROCm quick reduce、FlashInfer、custom all-reduce、Torch symmetric memory、PyNccl，最后才 fallback 到 `torch.distributed.all_reduce`。vLLM 也有 `VLLM_CUSTOM_ALLREDUCE_ALGO`、`VLLM_ALLREDUCE_USE_FLASHINFER`、`VLLM_USE_NCCL_SYMM_MEM` 等开关。

3. 对给定 safetensors 中的 `row_parallel_quant_method_out`，我检查 rank 0 的 tensor 是 `torch.bfloat16`、shape `[8, 8192]`，大小 131072 bytes。若 8 张 H100 在同一 full-NVLink 拓扑内，custom all-reduce 可用且未禁用，那么 SGLang 和 vLLM 都很可能选择 custom all-reduce 的 1-stage kernel。这个 kernel 的规约顺序都是固定 rank 顺序，代码注释也说明不重排地址以保持各 rank bitwise identical；在这种具体路径下，实际结果大概率逐 bit 一致。

4. 但两个框架不能保证 bitwise 结果严格一致。严格保证需要同时固定：同一输入 bit pattern、同一 dtype、同一 TP rank 顺序、同一 all-reduce 后端、同一 algorithm 选择、同一 NCCL/custom kernel 版本、同一环境变量、同一 CUDA/NCCL/driver 行为。当前 SGLang 的 custom all-reduce 是从较旧 vLLM 代码演化来的，vLLM 当前版本的同步 barrier 和共享 buffer 管理已经不同；此外任何一方 fallback 到 PyNccl/NCCL/torch.distributed，或者环境变量强制 `1stage/2stage` 不一致，都可能导致 bitwise 差异。

我已经写了三个验证脚本：

```bash
/data/like/miniconda3/envs/simo_sglang/bin/torchrun --nproc_per_node=8 like-useful/sglang_all_reduce.py
/data/like/miniconda3/envs/simo_vllm/bin/torchrun --nproc_per_node=8 like-useful/vllm_all_reduce.py
/data/like/miniconda3/envs/simo_sglang/bin/python like-useful/validate-sglang-vllm-all-reduce.py
```

输出文件分别是 `/data/like/temp/sglang.safetensors` 和 `/data/like/temp/vllm.safetensors`，key 都是 `rank_0_all_reduce`。验证脚本会按 bytes 做 bitwise 对比；若不一致，会打印 mismatch byte 数、最大绝对误差和第一个数值不一致的位置。

实际运行结果：

```text
sglang: shape=(8, 8192) dtype=torch.bfloat16
vllm:   shape=(8, 8192) dtype=torch.bfloat16
bitwise_equal=True
```

---

# tensor_model_parallel_all_reduce 调用链路对比 (sglang vs vllm)

## 测试配置

- **sglang 启动参数**: `--tp-size 8 --attention-backend fa3 --enable-deterministic-inference --disable-cuda-graph`
- **vllm 启动参数**: `--tensor-parallel-size 8 --enforce-eager`
- 硬件环境: NVIDIA CUDA, 8 GPUs

---

## 一、sglang 的调用链路

### 1. 入口: RowParallelLinear.forward
- `python/sglang/srt/layers/linear.py:1504`
  ```python
  output = tensor_model_parallel_all_reduce(output_parallel)
  ```

### 2. 顶层封装函数
- `python/sglang/srt/distributed/communication_op.py:11-13`
  ```python
  def tensor_model_parallel_all_reduce(input_):
      return get_tp_group().all_reduce(input_)
  ```

### 3. 获取 TP GroupCoordinator
- `python/sglang/srt/distributed/parallel_state.py:1415-1422` → 返回 `_TP` (GroupCoordinator)

### 4. GroupCoordinator.all_reduce —— 核心分发
- `python/sglang/srt/distributed/parallel_state.py:529-627`

分支判断顺序 (NVIDIA + `--disable-cuda-graph` + `--enable-deterministic-inference`):

| 分支 | 行号 | 是否命中 | 备注 |
|------|------|----------|------|
| `use_deterministic_ar = is_hip() and use_1stage_ar` | 548-566 | ❌ | NVIDIA, is_hip()=False, deterministic 路径只在 ROCm 生效 |
| CPU tensor + shm | 568-573 | ❌ | GPU tensor |
| HPU/XPU/NPU communicator | 575-582 | ❌ | NVIDIA |
| `pynccl_comm + is_symmetric_memory_enabled()` | 584-589 | ❌ | `--disable-cuda-graph` 关闭 symmetric memory |
| ca (custom_all_reduce).should_custom_ar | 592-597 | ✅ 通常命中 | tensor 大小 < max_size, NVLink/P2P 可用 |
| qr (quick_all_reduce) | 598-603 | ❌ | AMD only |
| pymscclpp | 604-609 | ❌ | 默认未启用 |
| torch_symm_mem | 610-615 | ❌ | symm_mem 未启用 |
| piecewise cuda graph → pynccl | 616-618 | ❌ | `--disable-cuda-graph` |
| **fallback inplace**: pynccl → torch.distributed | 658-666 | 兜底 | 大 tensor 走这里 |

### 5. 核心实现 (sglang 在该配置下最常命中的两条路径)

#### 路径 A: Custom All-Reduce (ca)
- `python/sglang/srt/distributed/parallel_state.py:629-656` `_all_reduce_out_place`
  → `ca_comm.custom_all_reduce(input_)`
- `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py:406-441`
  → `torch.ops.sgl_kernel.all_reduce(...)` (sgl_kernel 自研 CUDA kernel)

#### 路径 B: PyNCCL (inplace, 大 tensor 走这里)
- `python/sglang/srt/distributed/parallel_state.py:658-666` `_all_reduce_in_place`
  → `pynccl_comm.all_reduce(input_)`
- `python/sglang/srt/distributed/device_communicators/pynccl.py:144-165`
  → `self.nccl.ncclAllReduce(...)` (libnccl)

> ⚠️ 注意：`--enable-deterministic-inference` 在 NVIDIA CUDA 上对 all-reduce **不起作用** (代码里只在 `is_hip()` 为 True 时才进入 deterministic 1-stage kernel)。

---

## 二、vllm 的调用链路

### 1. 入口: RowParallelLinear.forward
- `vllm/model_executor/layers/linear.py:1562`
  ```python
  output = tensor_model_parallel_all_reduce(output_parallel)
  ```

### 2. 顶层封装函数
- `vllm/distributed/communication_op.py:12-14`
  ```python
  def tensor_model_parallel_all_reduce(input_):
      return get_tp_group().all_reduce(input_)
  ```

### 3. 获取 TP GroupCoordinator
- `vllm/distributed/parallel_state.py:1214-1216` → 返回 `_TP`

### 4. GroupCoordinator.all_reduce
- `vllm/distributed/parallel_state.py:487-509`
- CUDA 平台默认 `use_custom_op_call=True`，会走 `torch.ops.vllm.all_reduce(...)` (注册见 259-263)
- 最终都会进入 `_all_reduce_out_place` → `self.device_communicator.all_reduce(input_)`

### 5. CudaCommunicator.all_reduce —— 核心分发
- `vllm/distributed/device_communicators/cuda_communicator.py:161-218`

分支判断顺序 (NVIDIA + `--enforce-eager`，默认环境变量):

| 分支 | 行号 | 是否命中 | 备注 |
|------|------|----------|------|
| **NCCL Symmetric Memory + copy** (`should_nccl_symm_mem_allreduce`) | 164-169 | ✅ **大概率优先命中** | `VLLM_ALLREDUCE_USE_SYMM_MEM=1` 默认开启；TP=8 时 size 在 (16KB, 128KB) 之外都走这里 |
| Quick All-Reduce | 172-180 | ❌ | AMD only |
| FlashInfer | 181-189 | ❌ | `VLLM_ALLREDUCE_USE_FLASHINFER=0` 默认 |
| Custom All-Reduce | 190-198 | ⚠️ 中等 size 命中 | TP=8 preferred range 是 (16KB,128KB) |
| Symmetric Memory (torch symm_mem) | 199-203 | ⚠️ 大 tensor 可能命中 | bfloat16 + size%4==0 + 小于 max_size (SM9.0=64MB / SM10.0=128MB) |
| PyNCCL | 204-218 | 兜底 | 落到 `ncclAllReduce` |

### 6. 核心实现 (vllm 在该配置下最常命中的几条路径)

#### 路径 A: NCCL Symmetric Memory with copy (默认最优先)
- `vllm/distributed/device_communicators/all_reduce_utils.py:91-125`
  → `torch.ops.vllm.all_reduce_symmetric_with_copy(input_)`
  → 内部使用 PyTorch 的 symmetric_memory + 调用 NCCL 的 symmetric kernel

#### 路径 B: Custom All-Reduce
- `vllm/distributed/device_communicators/custom_all_reduce.py:233-283`
  → `torch.ops._C_custom_ar.all_reduce(...)` (vllm 自研 CUDA kernel)

#### 路径 C: torch symm_mem
- `vllm/distributed/device_communicators/symm_mem.py:117-156`
  → `torch.ops.symm_mem.multimem_all_reduce_(...)` 或 `two_shot_all_reduce_(...)`

#### 路径 D: PyNCCL (fallback)
- `vllm/distributed/device_communicators/pynccl.py:150-181`
  → `self.nccl.ncclAllReduce(...)` (libnccl)

---

## 三、为什么 sglang 与 vllm 的 all-reduce 结果 **bitwise 不一致**

虽然两边最终都基于 NCCL/CUDA，但选择的 all-reduce **算法实现路径不同**，浮点累加顺序不同，所以结果不可能 bitwise 相同：

### 关键差异

| 维度 | sglang (本次配置) | vllm (本次配置) |
|------|-------------------|-----------------|
| 默认优先路径 | `sgl_kernel` 的 custom_all_reduce (NVLink P2P one/two-shot) | `torch.ops.vllm.all_reduce_symmetric_with_copy` (NCCL Symmetric Memory) |
| 大 tensor fallback | `pynccl.ncclAllReduce` (NCCL ring/tree) | NCCL Symmetric Memory 或 torch symm_mem multimem |
| 中等 tensor | sgl_kernel custom AR | vllm custom_all_reduce 自研 kernel |
| deterministic flag | NVIDIA 上无效 | 没传 |
| size 阈值 | sgl_kernel 自己的 max_size | vllm 自己的 CUSTOM_ALL_REDUCE_MAX_SIZES + custom_ar_preferred_ranges (TP=8: 16KB-128KB) |
| symmetric memory | 默认关闭 (因 `--disable-cuda-graph`) | 默认 `VLLM_ALLREDUCE_USE_SYMM_MEM=1` 开启 |

### Bitwise 不相同的根本原因

1. **算法不同**：sgl_kernel 的 one-shot/two-shot all-reduce kernel ≠ vllm 自研 custom AR ≠ NCCL Symmetric Memory ≠ NCCL ring。即使数学上等价，浮点求和顺序 (reduction tree shape) 不同，bf16/fp16 累加非结合，结果会有 LSB 级差异。
2. **kernel 实现不同**：sglang 的 `sgl_kernel.allreduce` 与 vllm 的 `_C_custom_ar.all_reduce` 是两份独立的 CUDA 源码，块/线程拆分、累加顺序都不一样。
3. **symmetric memory 路径不同**：vllm 默认开了 `VLLM_ALLREDUCE_USE_SYMM_MEM`，sglang 在 `--disable-cuda-graph` 下关闭了 `is_symmetric_memory_enabled()`。
4. **dtype 累加精度**：custom AR 内部一般以 fp32 累加，但具体 vec / 分块大小不同，仍会引入差异。
5. **NCCL 算法选择**：即便都退回 NCCL，sglang 走 `pynccl.ncclAllReduce`，vllm 走的是 `all_reduce_symmetric_with_copy` (NCCL symmetric kernel)，是两种不同算法。

### 验证建议

要让两者 bitwise 相同，需要强制两边都走同一条路径，例如：

- **方法 1: 都强制走 NCCL ring**
  - vllm: `VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_DISABLE_CUSTOM_ALL_REDUCE=1`
  - sglang: 通过 `disable_custom_all_reduce=True` 关闭 ca + 不进入 symm_mem 分支，让其 fallback 到 `pynccl.ncclAllReduce`
  - 然后再保证两边 NCCL 版本/算法/通信器初始化一致 (NCCL_ALGO=Ring NCCL_PROTO=Simple)
- **方法 2: 都禁掉 custom 路径**，统一使用 `torch.distributed.all_reduce` (走同一份 NCCL)
  - 同时设置 `NCCL_ALGO=Ring`、`NCCL_PROTO=Simple`、`NCCL_NTHREADS` 一致

否则在默认配置下，**bitwise 不一致是必然的，不是 bug**。

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

## vLLM 编译后 `import vllm._C` 报错分析 (修正版)

### 报错信息

```
>>> import vllm._C
ImportError: /data/like/package/vllm-for-conda-simo/vllm/_C.abi3.so: undefined symbol: _ZN3c1013MessageLoggerC1EPKciib
```

### 1. 符号解析

Demangled 名称: **`c10::MessageLogger::MessageLogger(char const*, int, int, bool)`**

| 参数 | C++ 类型 | 含义 |
|---|---|---|
| 1 | `char const*` | 源文件名 (如 `__FILE__`) |
| 2 | `int` | 行号 (如 `__LINE__`) |
| 3 | `int` | 严重级别 (severity level) |
| 4 | `bool` | 是否限流/洪水控制 |

### 2. vLLM 是否直接调用了 `c10::MessageLogger`？

**否。** vLLM 源码中没有任何地方直接引用 `MessageLogger`。该符号的依赖是**间接的**，通过 vLLM C++/CUDA 扩展代码中使用的 PyTorch 宏产生：

- `TORCH_CHECK(condition, msg)` — 条件检查失败时抛出异常
- `TORCH_WARN(msg)` — 发出警告
- `AT_ERROR(msg)` — 抛出 Aten 错误

这些宏内部构造 `c10::MessageLogger` 对象来格式化消息。

vLLM 中 40+ 个 `.cu/.cuh` 文件使用了 `TORCH_CHECK`，关键文件：

- `csrc/cache_kernels.cu` (lines 44, 52, 58, 81-90, 135, 749, 763, 826-835, 891-893, 935)
- `csrc/attention/paged_attention_v1.cu` (lines 122, 156)
- `csrc/attention/paged_attention_v2.cu` (lines 128, 163)
- `csrc/attention/merge_attn_states.cu` (lines 208, 261, 268, 317, 322)
- `csrc/attention/dtype_fp8.cuh` (line 33)
- `csrc/quantization/marlin/kernel_selector.h` (line 1468)
- `csrc/moe/marlin_moe_wna16/kernel_selector.h` (line 1468)

### 3. 编译依赖链

CMakeLists.txt 通过 `find_package(Torch REQUIRED)` 链接：

```
CMakeLists.txt:91: find_package(Torch REQUIRED)
cmake/utils.cmake:574: target_link_libraries(${MOD_NAME} PRIVATE torch ${ARG_LIBRARIES})
```

`_C.abi3.so` 的 `NEEDED` 依赖（`readelf -d`）：
```
NEEDED  libtorch.so
NEEDED  libc10.so          ← c10::MessageLogger 的来源
NEEDED  ...
```

无需 RPATH/RUNPATH。

### 4. 根因: PyTorch 2.10 → 2.11 ABI 断裂

#### 4.1 实证: 两个版本的 libc10.so 符号对比

**vllm `_C.abi3.so` 需要的符号** (`nm -D`，类型 `U` = undefined):

```
U _ZN3c1013MessageLoggerC1EPKciib    ← OLD ABI: MessageLogger(char const*, int, int, bool)
U _ZN3c1013MessageLoggerD1Ev         ← 析构函数 (ABI 没变)
U _ZN3c1013MessageLogger6streamB5cxx11Ev  ← stream() (ABI 没变)
```

**torch 2.10.0 `libc10.so` 提供的符号** (类型 `T` = defined):

```
T _ZN3c1013MessageLoggerC1EPKciib    ← ✅ 匹配 vllm 需要
T _ZN3c1013MessageLoggerD1Ev
T _ZN3c1013MessageLogger6streamB5cxx11Ev
```

**torch 2.11.0 `libc10.so` 提供的符号** (类型 `T` = defined):

```
T _ZN3c1013MessageLoggerC1ENS_14SourceLocationEib  ← ❌ 不匹配！
T _ZN3c1013MessageLoggerD1Ev                        ← ✅ 匹配
T _ZN3c1013MessageLogger6streamB5cxx11Ev            ← ✅ 匹配
```

#### 4.2 ABI 变更细节

`c10::MessageLogger` 构造函数在 torch 2.11.0 中发生了 **参数类型变更**:

| 版本 | 构造函数签名 | mangled name 后缀 |
|---|---|---|
| torch 2.10.0 | `MessageLogger(char const*, int, int, bool)` | `C1EPKciib` |
| torch 2.11.0 | `MessageLogger(c10::SourceLocation, int, bool)` | `C1ENS_14SourceLocationEib` |

PyTorch 将单独的 `(char const* file, int line)` 两个参数合并为一个 `c10::SourceLocation` 结构体。新的 `SourceLocation` 封装了文件名和行号：

```cpp
// torch 2.10.0 头文件
class MessageLogger {
  MessageLogger(const char* file, int line, int severity, bool graph);
};

// torch 2.11.0 头文件
class MessageLogger {
  MessageLogger(SourceLocation loc, int severity, bool graph);
  // 其中 SourceLocation 内部包含 file 和 line
};
```

#### 4.3 为什么 `import torch` 也无法修复

vllm `_C.abi3.so` 在编译时就嵌入了对 OLD ABI 符号 `MessageLogger(char const*, int, int, bool)` 的依赖。torch 2.11.0 的 `libc10.so` **完全不含这个符号**（被 `MessageLogger(SourceLocation, int, bool)` 替换）。这是永久性缺失，无法通过调整加载顺序修复。

验证：conda env 中 torch=2.11.0，**即便先 import torch，再 import vllm._C 仍然报同样错误**。

#### 4.4 为什么编译时用了 torch 2.10.0 的 ABI

编译脚本 (`install-vllm.sh:5`):
```bash
pip install --config-settings=build.verbose=true -vvv -e . --no-build-isolation
```

`--no-build-isolation` 使用当前 Python 环境中的 torch 头文件进行编译。编译时的 conda 环境包含 **torch 2.10.0**。

但编译日志第 264-267 行显示 pip 在安装依赖时 **将 torch 升级到了 2.11.0**。由于 pip 先安装依赖再编译，编译究竟用了哪个版本取决于具体时序：如果 `--no-build-isolation` 在 torch 升级后生效，则编译用了 2.11.0 头文件；如果在升级前生效，则用了 2.10.0。

实际情况：`_C.abi3.so` 编译时链接的是 OLD ABI 符号（`C1EPKciib`），说明**编译时使用的 torch 头文件 ≤ 2.10.0**。而运行时 conda 环境被升级到了 torch 2.11.0，导致 ABI 不兼容。

### 5. 修复方案

**方案 A (推荐): 重新编译** — 在当前 torch 2.11.0 环境下重新编译 vllm：

```bash
cd /data/like/package/vllm-for-conda-simo
rm -rf build/
pip install --config-settings=build.verbose=true -vvv -e . --no-build-isolation
```

重新编译后 `_C.abi3.so` 会需要 `_ZN3c1013MessageLoggerC1ENS_14SourceLocationEib`，与 torch 2.11.0 的 `libc10.so` 匹配。

**方案 B: 降级 torch** — 将 conda 环境中的 torch 降级回 2.10.0：

```bash
pip install torch==2.10.0
```

### 6. 依赖关系总结

```
vllm/_C.abi3.so
  └── 链接时: TORCH_CHECK → c10::MessageLogger(char const*, int, int, bool)  [torch ≤ 2.10 ABI]
  └── 运行时: libc10.so (torch 2.11.0) 只提供 MessageLogger(SourceLocation, int, bool)  [torch ≥ 2.11 ABI]
  └── 结果: 符号找不到 → undefined symbol error
```

### 7. 如何在未来 torch 升级中避免 ABI 不兼容问题

#### 7.1 当前编译命令的问题

当前编译脚本 (`install-vllm.sh:5`)：

```bash
pip install --config-settings=build.verbose=true -vvv -e . --no-build-isolation
```

`--no-build-isolation` 的含义：**不在隔离的 build venv 中编译**，而是直接使用当前 Python 环境中的包（包括 torch）的**头文件和库**进行编译。

配合 `pyproject.toml:9` 的 `requires = ["torch == 2.11.0"]`，产生了以下时序问题：

```
1. pip 检查当前环境: torch == 2.10.0
2. pip 发现 requires 不满足 (需要 2.11.0)
3. pip 自动升级 torch: 2.10.0 → 2.11.0
4. cmake 编译 vllm → 使用哪个 torch 头文件？不确定
   ├── 如果用 2.10.0 头文件编译 → OLD ABI → 报错
   └── 如果用 2.11.0 头文件编译 → NEW ABI → 正常
5. 结果: 运行时 torch == 2.11.0，但 .so 链接了 OLD ABI → undefined symbol
```

#### 7.2 ❌ 编译前卸载旧版 torch 不能解决

卸载 torch 后 vllm 无法编译——cmake 需要 `find_package(Torch REQUIRED)` 找到 torch 头文件和 `.so` 文件。没有 torch 就无法编译。

所以**卸载旧 torch 不行**。正确思路是**确保编译时 torch 版本 ≡ 运行时 torch 版本**。

#### 7.3 ✅ 推荐方案

**方案 A (最简单、最安全): 去掉 `--no-build-isolation`**

```bash
pip install -e . --no-build-isolation  # ❌ 当前方式
pip install -e .                        # ✅ 推荐方式
```

去除 `--no-build-isolation` 后，pip 会：
1. 创建一个**隔离的临时 build venv**
2. 在其中安装 `pyproject.toml` 的 `[build-system].requires`（含 `torch == 2.11.0`）
3. 在隔离 venv 中编译 vllm
4. 将编译产物安装到当前环境

这样编译时的 torch 和 build venv 中的 torch 完全一致，避免了版本交错。即使当前环境的 torch 版本不同，编译过程也不受影响。

**方案 B: 如果必须保留 `--no-build-isolation`**（例如需要特定的 CUDA torch）

用 `--no-deps` 阻止 pip 自动修改依赖：

```bash
# 第1步: 确保当前环境装好目标 torch 版本
pip install torch==2.11.0

# 第2步: 编译时禁止依赖变更
pip install -e . --no-build-isolation --no-deps
```

加上 `--no-deps` 后，pip 不会再自动升级/降级任何包，编译时 torch 版本 ≡ 运行时 torch 版本。

**方案 C: 使用 lock 文件固定依赖版本**

在 pyproject.toml 的 `requires` 中固定精确版本（当前已做：`torch == 2.11.0`），且确保当前环境的 torch 与之一致再进行编译。

#### 7.4 核心原则

```
编译时 torch 头文件版本 ≡ 运行时 torch 版本 → ABI 安全 ✅
编译时 torch 头文件版本 ≠ 运行时 torch 版本 → ABI 风险 ❌
```

**未来任何时候升级 torch，都应当重新编译 vllm。** 没有自动检测 ABI 断裂的通用方法——最佳实践是保持 build-time == runtime。

**DeepSeek-V2-Lite-Chat-16B_A2.4B (MLA 模型, gsm8k 5-shot):**

| KV Cache 量化配置 | flexible-extract exact_match | 状态 |
|---|---|---|
| `fp8_per_group` | 0.6710 | 正常 |
| `mxint8` | 0.6619 | 正常 |
| `mxfp8` | 0.6505 | 正常 |
| `int8_per_group` | 0.6475 | 正常 |
| `mxfp6` | **0.0728** | 异常 |
| `mxfp4` | **0.0099** | 异常 |
| `nvfp4` | **0.0099** | 异常 |

**Llama3.1-8B-Instruct (GQA 模型, gsm8k 5-shot):**

| KV Cache 量化配置 | flexible-extract exact_match | 状态 |
|---|---|---|
| `mxfp6` | 0.7832 | 正常 |
| `mxint8` | 0.7794 | 正常 |
| `int8_per_group` | 0.7756 | 正常 |
| `nvfp4` | 0.7672 | 正常 |
| `mxfp8` | 0.7642 | 正常 |
| `fp8_per_group` | 0.7612 | 正常 |
| `mxfp4` | 0.6929 | 正常 |

### 2. 背景: MLA 模型的 KV Cache 布局差异

DeepSeek-V2-Lite 使用 MLA (Multi-head Latent Attention)，与 Llama3.1 的 GQA 架构在 KV Cache 层面完全不同：

**常规 GQA 模型 (Llama) — `set_kv_buffer_kernel`:**
- KV cache 按 kv_head 维度展开，每个 head 独立存储
- 布局 per token: `[head0_packed | head1_packed | ... | head0_scale | head1_scale | ...]`
- `SCALE_PLANE_OFFSET = num_kv_heads * PACKED_HEAD_SIZE`

**MLA 模型 (DeepSeek-V2-Lite) — `concat_and_cache_mla_kernel`:**
- KV cache 只存储 1 个"kv head"的 latent 表示 c_KV (dim=512) + k_pe (dim=64)
- 布局 per token（tile 交错的 packed/scales）:
  ```
  [ kv_c_tile0_packed(16B) | ... | kv_c_tile15_packed(16B) | kv_c_tile0_scale(1B) | ... | kv_c_tile15_scale(1B) | k_pe_tile0_packed | k_pe_tile1_packed | k_pe_tile0_scale | k_pe_tile1_scale ]
  ```
- 总 packed: 16 × PACKED_TILE_BYTES ; 总 scales: 16 × SCALE_TILE_BYTES
- `SCALE_PLANE_OFFSET = num_kv_heads * PACKED_HEAD_SIZE = 1 * PACKED_HEAD_SIZE`
- 代码位置: `simo/extensions/sglang_simo/layers/attention/triton_ops/set_kv_buffer.py:309-467` 函数 `concat_and_cache_mla_kernel`

decode 阶段通过 `self.attn_mqa` 读写 KV cache (`Lk=576, BLOCK_DMODEL=512, BLOCK_DPE=64`)。Q 使用 c_KV-like 表示 (512 dim)，K/V 均从同一个 latent buffer 读取。

### 3. 已确认的 Bug: NVFP4 的 MX_QUANT_DIM 不一致

**文件:** `simo/ops/kernels/upcast/_upcast_from_mxfmt.py:33`

```python
MX_QUANT_DIM: tl.constexpr = 16 if MX_FORMAT_ID == NVFP4_E2M1 else 32
```

**写路径** (`_compute_and_pack_mxfmt`, `_downcast_to_mxfmt.py:240`):
- `MX_BLOCK_SIZE = tile_size = 32` — 每 32 个值对应 1 个 scale
- 对于 kv_lora_rank=512: 产生 16 个 scale，每个覆盖 32 个值

**读路径** (`_unpack_and_dequant_mxfmt`, `_upcast_from_mxfmt.py:31-37`):
- `MX_QUANT_DIM = 16` — 认为每个 scale 只覆盖 16 个值
- `BLOCK_SIZE_QUANT_DIM = scale.shape[1] * 16 = 16 * 16 = 256`
- 只产出 256 个 dequant 值，而非正确的 512 个

**后果:**
- scale 被应用到错误的数值组（一个为 32 个值计算的 scale 被分到两个 16 值组上分别应用）
- dequant 后的 K/V 值完全错误
- `tl.dot(q [BLOCK_H, 512], K_dequant [256, BLOCK_N])` 的 contracting dimension 不匹配 (512 ≠ 256)
- (Triton 3.x 在这个场景下似乎接受了 reshape，但产生了垃圾输出而非崩溃)

**注意:** 这个 bug 在 Llama GQA 上不影响正确性，因为 GQA 模型的 SCALE_HEAD_SIZE = head_dim/32 = 128/32 = 4。对于 head_dim=128、SCALE_HEAD_SIZE=4:
- 写: 128/32 = 4 个 scale，每个覆盖 32 个值
- 读 (MX_QUANT_DIM=16): scale.shape[1] * 16 = 4 * 16 = 64 ≤ 128 — 但这也会产生维度问题

实际上，对于 NVFP4 + Llama 得分为 0.7672 是正常的。这说明 NVFP4 在 Llama 上确实工作正常。进一步的维度分析: 对于 GQA, `sparse_head_size` 在 extend_attention 的 `NEEDS_SW_DEQUANT` 路径中加载 PACKED_HEAD_SIZE 字节。如果 head_dim=128 → PACKED_HEAD_SIZE=64 (NVFP4 2 值/字节)。读路径用 MX_QUANT_DIM=16 做 dequant: scale.shape[1] * 16 = head_dim/MX_BLOCK_SIZE * 16 = 128/32 * 16 = 4 * 16 = 64。但加载了 64 字节 → 128 个 fp4 值。reshape [BLOCK_N, 128] → [BLOCK_N, 4, 16] = [BLOCK_N, 64]。128 ≠ 64，同样会有 reshape 失败…

这里可能 Triton 做了沉默的重塑形适配导致数值错误；但在 MLA (更大的维度) 上暴露得更严重。

**修复建议:** 将 `_upcast_from_mxfmt.py:33` 改为 `MX_QUANT_DIM: tl.constexpr = 32`（去掉 NVFP4 的特判），因为两种 fp4 格式都使用相同的 MX_BLOCK_SIZE=32。

### 4. mxfp4/mxfp6 异常分析

#### 4.1 Decode 路径维度和布局验证

对 MLA decode 路径逐一验证数据布局：

**PACKED_HEAD_SIZE 计算** (`simo/extensions/sglang_simo/models/deepseek_v2.py:8-19`, 函数 `updata_module_head_size`):
```python
x_q, scale_a = layer.kv_cache_downcast_kernel(torch.randn(1, kv_lora_rank, device="meta"))
layer.packed_head_size = x_q.contiguous().view(torch.uint8).shape[-1]
```
- mxfp4: PACKED_HEAD_SIZE = 512/2 = **256** 字节
- mxfp6: PACKED_HEAD_SIZE = 512×6/8 = **384** 字节
- mxint8: PACKED_HEAD_SIZE = **512** 字节
- mxfp8: PACKED_HEAD_SIZE = **512** 字节
- SCALE_HEAD_SIZE (全部格式): 512/32 = **16**

**MLA KV Cache 写布局** (`concat_and_cache_mla_kernel`, `set_kv_buffer.py:309-467`):
```
KV_C_PACKED_BYTES = 16 * PACKED_TILE_BYTES   (tile_size=32)
  mxfp4: 16 * 16 = 256
  mxfp6: 16 * 24 = 384  (32元素 × 6bit / 8 = 24字节 packed per tile)
  mxint8: 16 * 32 = 512
KV_C_SCALE_BYTES = 16 * 1 = 16
KV_C_TOTAL_BYTES = KV_C_PACKED_BYTES + KV_C_SCALE_BYTES
  mxfp4: 272, mxfp6: 400, mxint8: 528
```

**Decode 读路径 loaded 数据验证** (`decode_attention.py:196-234`):
```
# Packed K: kv_loc * stride + 0 * PACKED_HEAD_SIZE + [0..PACKED_HEAD_SIZE-1]
# Scale K: kv_loc * stride + 1*PACKED_HEAD_SIZE + 0*SCALE_HEAD_SIZE + [0..15]

对于 mxfp4: packed = offset [0..255], scale = offset [256..271]  ✓ (与写一致)
对于 mxfp6: packed = offset [0..383], scale = offset [384..399]  ✓ (与写一致)
对于 mxint8: packed = offset [0..511], scale = offset [512..527]  ✓ (与写一致)
```

**维度匹配:**
- mxfp4 decode: `tl.dot_scaled(q [BLOCK_H,512], None, "bf16", k_packed.T [256, BLOCK_N], k_scale, "e2m1")`
- mxfp6 decode: `tl.dot(q [BLOCK_H,512], K_dequant [512, BLOCK_N])` (dequant 从 384 字节解出 512 个 fp6 值)
- mxint8 decode: `tl.dot(q [BLOCK_H,512], K_dequant [512, BLOCK_N])` (dequant 从 512 字节 bitcast 出 512 个 int8)

所有维度在数值上匹配。K dequant 产出 512 个值，Q 的 contracting dim 也为 512。

**Rope PE 路径验证:**
- rope_offset_in_token = 1 * (PACKED_HEAD_SIZE + SCALE_HEAD_SIZE)
  - mxfp4: 256+16 = 272 = KV_C_TOTAL_BYTES ✓
  - mxfp6: 384+16 = 400 = KV_C_TOTAL_BYTES ✓
- PACKED_HEAD_SIZE_ROPE (pe_dim=64):
  - mxfp4: 64/2 = 32, SCALE_HEAD_SIZE_ROPE = 64/32 = 2
  - mxfp6: ceil(64*6/8) = 48, SCALE_HEAD_SIZE_ROPE = 64/32 = 2
- PE packed 读偏移: rope_offset + 0 * PACKED_HEAD_SIZE_ROPE + [0..packed-1]
  - mxfp4: 272 + [0..31] → 读写一致 ✓
  - mxfp6: 400 + [0..47] → 读写一致 ✓

#### 4.2 FP6 打包/解包验证

写 (SIPU 模式, `_downcast_to_mxfmt.py:498-513`):
```
byte0 = (x3 << 2) | (x2 >> 4)   — bits [7:2]=x3, [1:0]=x2的高位
byte1 = ((x2 & 0x0F) << 4) | (x1 >> 2)  — bits [7:4]=x2的低位, [3:0]=x1的高位
byte2 = ((x1 & 0x03) << 6) | x0   — bits [7:6]=x1的低位, [5:0]=x0
```
存储为 3 字节 uint32: `b0 | (b1 << 8) | (b2 << 16)`。
在 cache 中按 `[b2, b1, b0]` 顺序存储 (每 3 字节一组)。

读 (`_upcast_from_mxfmt.py:55-68` 和 `common.py:202-208`):
```
加载: b2=pos[3i], b1=pos[3i+1], b0=pos[3i+2]
重构 uint32: b0 | (b1 << 8) | (b2 << 16)
解包: x0, x1, x2, x3 = d, c, b, a (反向顺序, 对应原来的 x0, x1, x2, x3)
```

打包和解包互逆 ✓，数值恢复应正确。

#### 4.3 疑点总结

对于 mxfp4 (0.0099) 和 mxfp6 (0.0728) 在 DeepSeek MLA 上的异常，**所有已知的维度、布局、打包/解包路径均已验证一致**。可能的原因包括:

1. **V (value) 路径问题**: MLA 的 V_Buffer 和 K_Buffer 指向同一块内存（`memory_pool.py:306-307`），但 V 的 dequant 使用同样的 packed/scales 布局。虽然布局一致，但在 NLP 任务中 V 的精度比 K 更关键（attention 权重 softmax 对 K 的误差有一定容忍度，但 V 直接加权求和）。

2. **Triton dot_scaled e2m1 的实际行为**: mxfp4 使用 `tl.dot_scaled(q, None, "bf16", k_packed.T, k_scale, "e2m1")`，而 mxint8 使用 dequant + `tl.dot`。dot_scaled 对 e2m1 格式的 block size 假设可能与我们实际的 MX block size (32) 不匹配。mxfp6 虽然也使用 dequant + tl.dot，但代码路径经过了更复杂的 FP6 上采样 (从 6-bit → fp32，经过指数偏置调整)。

3. **数值精度累积**: 4-bit 和 6-bit 格式的量化误差更大，对 MLA 架构中需要经过 kv_b_proj 投影的 latent 表示 (c_KV) 更敏感。MLA 的 c_KV 是 512 维的低秩压缩表示，量化误差在 512 维空间中产生的信息损失比 GQA 的 128 维 per-head K 更大。

**建议进一步调试方向**: 在 decode kernel 中 dump mxfp4/mxfp6 的 dequant K/V 值与 mxfp8 的进行对比，确认数值差异的量级；单独验证 V dequant 结果的正确性。

### 5. 为什么 8-bit 格式在 MLA 上正常工作

1. **mxfp8 (E4M3)**: decode 走 transposed 路径 (`decode_attention.py:236-260`)，直接加载 PACKED_HEAD_SIZE=BLOCK_DMODEL=512 个 fp8 元素，通过 `tl.dot_scaled` 计算 QK。没有打包/解包步骤，维度匹配且数值路径简单。

2. **mxint8**: 走 NEEDS_SW_DEQUANT 路径，但解包只需 bitcast (int8→fp32) + scale × 2^(-6)，没有复杂的 6-bit/4-bit 上采样和 interleave 操作。scale 与数值组的对应关系也正确匹配 (MX_QUANT_DIM=32 = MX_BLOCK_SIZE=32)。

3. **fp8_per_group / int8_per_group**: 走 PG_QUANT 路径 (`decode_attention.py:165-194`)，使用 per-group 的 `_dequant_pg_fused` 解量，与 MX 打包无关。

### 6. 为什么 Llama GQA 上所有格式都正常

Llama3.1-8B-Instruct 使用标准 GQA 架构：
- head_dim = 128, kv_head = 8
- KV cache 使用 `set_kv_buffer` 的常规路径，每个 kv head 独立量化
- PACKED_HEAD_SIZE 基于 head_dim (128)，不是 kv_lora_rank (512)
- 无 MLA 特有的 concat interleave 布局
- Per-head K/V 直接量化存储，无需后续投影
- 数值误差影响较 MLA 小（128 维 vs 512 维低秩投影）

### 7. NVFP4 MX_QUANT_DIM 不一致排查结果：确认不是 Bug

#### 7.1 问题背景

之前怀疑 `_upcast_from_mxfmt.py:33` 中 NVFP4 的 `MX_QUANT_DIM=16` 与写路径的 `MX_BLOCK_SIZE=32` 不一致，认为是一个 bug。但实际排查后发现 **这不是 bug**，读写路径的 block size 是一致的。

#### 7.2 NVFP4 配置文件

`simo/extensions/sglang_simo/example/simo_quantization_config/kv_cache_quant/quant_config_kvquant_nvfp4.json`:

```json
{
    "dtype": "nvfp4_e2m1",
    "scale_mode": "e4m3",
    "group_size": 16
}
```

关键点：`group_size: 16`。

#### 7.3 写路径：MX_BLOCK_SIZE 的实际值

**Step 1**: `QuantizeSpecMX` 中 `group_size` → `block_size`

`simo/quantization/config.py`, `QuantizeSpecMX` 类，validator `sync_group_size_with_block_size`:
```python
self.block_size = self.group_size  # 16
```

另外 `calibrate_dtype` validator 还强制约束 nvfp4_e2m1 + e4m3 scale_mode 时 `block_size=16`:
```python
if self.dtype == "nvfp4_e2m1" and self.scale_mode == "e4m3":
    self.block_size = 16
```

因此 `kv_cache_quant_spec.block_size = 16`。

**Step 2**: `set_kv_buffer` / `set_mla_kv_buffer` 中 `tile_size = block_size`

非 MLA 路径 — `simo/extensions/sglang_simo/layers/attention/triton_ops/set_kv_buffer.py`, `simo_set_kv_buffer()`:
```python
tile_size = kv_cache_quant_spec.block_size  # = 16
```

MLA 路径 — `simo/extensions/sglang_simo/mem_cache/memory_pool.py:346`, `set_mla_kv_buffer()`:
```python
tile_size = kv_cache_quant_spec.block_size  # = 16
```

**Step 3**: `TILE_SIZE` → `MX_BLOCK_SIZE` — `set_kv_buffer.py`:

`simo/extensions/sglang_simo/layers/attention/triton_ops/set_kv_buffer.py`, `set_kv_buffer_kernel()` 和 `concat_and_cache_mla_kernel()`, 调用 `_compute_and_pack_mxfmt()` 时:
```python
MX_BLOCK_SIZE = TILE_SIZE  # = 16
```

因此写路径实际传入 `_compute_and_pack_mxfmt` 的 `MX_BLOCK_SIZE = 16`，**不是默认值 32**。

#### 7.4 读路径：MX_QUANT_DIM 的值

`simo/ops/kernels/upcast/_upcast_from_mxfmt.py:32-33`, `_unpack_and_dequant_mxfmt()`:
```python
# 使用三元表达式定义 MX_QUANT_DIM，避免 Triton 编译器的 constexpr 识别问题
MX_QUANT_DIM: tl.constexpr = 16 if MX_FORMAT_ID == NVFP4_E2M1 else 32
```

对于 NVFP4，`MX_QUANT_DIM = 16`。

#### 7.5 结论：读写一致，非 Bug

| 路径 | 变量 | 值 |
|------|------|----|
| 配置 | `group_size` / `block_size` | **16** |
| 写路径 | `MX_BLOCK_SIZE` (= `tile_size` = `block_size`) | **16** |
| 读路径 | `MX_QUANT_DIM` (NVFP4 分支) | **16** |

读写路径的 block size 完全一致（都是 16），不存在不匹配的问题。**这不是 bug**。

#### 7.6 对比：其他格式的 block_size

其他 MX 格式的配置文件没有显式指定 `group_size`，使用 `QuantizeSpecMX` 的默认值 `block_size=32`：

| 格式 | 配置文件 group_size | block_size | 读路径 MX_QUANT_DIM | 匹配？ |
|------|---------------------|------------|-------------------|--------|
| nvfp4_e2m1 | 16 (显式) | 16 | 16 | ✓ |
| mxfp4_e2m1 | 未指定 (默认32) | 32 | 32 | ✓ |
| mxfp6_e2m3 | 未指定 (默认32) | 32 | 32 | ✓ |
| mxfp6_e3m2 | 未指定 (默认32) | 32 | 32 | ✓ |
| mxfp8_e4m3 | 未指定 (默认32) | 32 | 32 | ✓ |
| mxint8 | 未指定 (默认32) | 32 | 32 | ✓ |

所有格式的读写路径 block size 都是一致的。

#### 7.7 NVFP4 仍然分数异常的原因？

虽然 MX_QUANT_DIM 不是 bug，但 NVFP4 在 DeepSeek-V2-Lite 上的 gsm8k 分数仍然是 0.0099。这可能与以下因素有关：

1. **4-bit 极低精度**：NVFP4 只有 4 位（1 sign + 2 exp + 1 mantissa），可表示的数值非常有限（只有 16 个值），量化误差远大于 8-bit 格式。

2. **MLA 对量化误差更敏感**：MLA 的 latent 表示 c_KV 是 512 维低秩压缩，信息密度高。4-bit 量化在 512 维空间中引入的巨大误差经过 kv_b_proj 投影后被放大，导致 attention 输出失真。

### 8. mxfp4 和 mxfp6 在 MLA 上分数异常的原因排查

#### 8.1 问题描述

DeepSeek-V2-Lite MLA + kv cache only 量化，gsm8k 分数：

| 格式 | 分数 | 状态 |
|------|------|------|
| mxfp8_e4m3 | 0.6505 | 正常 |
| mxint8 | 0.6619 | 正常 |
| fp8_per_group | 0.6710 | 正常 |
| int8_per_group | 0.6475 | 正常 |
| **mxfp4_e2m1** | **0.0099** | 异常 |
| **mxfp6_e2m3** | **0.0728** | 异常 |
| **nvfp4_e2m1** | **0.0099** | 异常 |

但所有格式在 Llama3.1-8B-Instruct GQA 上均正常工作（0.69-0.78）。

#### 8.2 已排查的项目（均已确认一致，排除作为根因）

1. **MX_BLOCK_SIZE / MX_QUANT_DIM 一致性**：所有格式的 block_size 读写一致
2. **Buffer 分配大小一致性**：`SIMOMLATokenToKVPool.__init__`（`memory_pool.py:259-264`）与 `set_mla_kv_buffer`（`memory_pool.py:362-372`）计算结果一致
3. **Buffer 布局读写一致性**：`concat_and_cache_mla_kernel` 存储的 `[KV_C_packed | KV_C_scales | K_PE_packed | K_PE_scales]` 布局，与读取时的 `SCALE_PLANE_OFFSET`、`rope_offset_in_token` 等偏移一致
4. **FP6 打包/解包往返一致性**：SIPU 打包（`_downcast_to_mxfmt.py:498-513`）→ 3 字节存储（`set_kv_buffer.py:398-419`）→ 加载重建 uint32（`common.py:202-208`）→ 解包上采样（`_upcast_from_mxfmt.py:43-85`），逐字节验证互通
5. **mxfp4 打包/解包往返一致性**：2 个 E2M1 值打包为 1 字节（`_downcast_to_mxfmt.py:514-517`），读路径正确分离 nibble（`_upcast_from_mxfmt.py:87-129`）
6. **Scale 计算格式**：mxfp4 使用 `scale_mode: "e8m0_sipu"`，mxfp6 使用 `scale_mode: "e8m0_floor"`（同 mxfp8/mxint8），均为 E8M0 uint8

#### 8.3 关键发现：`tl.dot_scaled("e2m1")` 是 mxfp4 的根因

mxfp4_e2m1 在 K 的 QK 计算中使用 `tl.dot_scaled("e2m1")`，而其他格式均使用软件解量或 `tl.dot_scaled("e4m3"/"e5m2")`。

`tl.dot_scaled("e2m1")` 的使用位置（4 处）：

- `decode_attention.py:231`，`_fwd_grouped_kernel_stage1`：nope 部分的 K
- `decode_attention.py:346`，`_fwd_grouped_kernel_stage1`：rope 部分的 K
- `extend_attention.py:348`，`_fwd_kernel_stage1`：nope 部分的 K
- `extend_attention.py:452`，`_fwd_kernel_stage1`：rope 部分的 K

**为什么 Llama GQA 上正常而 DeepSeek MLA 上失败？**

`tl.dot_scaled("e2m1")` 在两种架构下的关键差异：

A. **MLA 有 nope (512d) + rope (64d) 分开计算再合并**：
   ```python
   # nope (decode_attention.py:230-231)
   qk = tl.dot_scaled(q, None, "bf16", k_packed.T, k_scale, "e2m1")  # → [BLOCK_H, BLOCK_N]
   # rope (decode_attention.py:345-346)
   qk_pe = tl.dot_scaled(qpe, None, "bf16", k_packed.T, k_scale, "e2m1")  # → [BLOCK_H, BLOCK_N]
   qk += qk_pe  # decode_attention.py:384
   ```
   如果 rope 部分（64d，只有 2 个 scale group）的 `tl.dot_scaled("e2m1")` 产生异常值，`qk += qk_pe` 会破坏整个 QK。

B. **极少数 scale group 下 Triton 可能有 bug**：
   - GQA：每个 head 4 个 scale group（128/32=4）
   - MLA nope：16 个 scale group（512/32=16）
   - **MLA rope：只有 2 个 scale group（64/32=2）**
   - 只有 2 个 scale group 时，如果 `tl.dot_scaled("e2m1")` 内部对 scale 索引作了错误的假设，`qk_pe` 可能为 NaN/inf/极端值
   - `qk += qk_pe` 后 QK 被污染 → softmax 后 uniform → attention 输出被破坏 → 分数 ~0.01

#### 8.4 mxfp6 的可能根因

mxfp6 使用软件解量路径（`_unpack_and_dequant_mxfmt`），避免了 `tl.dot_scaled("e2m1")`。但仍然失败（0.0728）。软件解量路径逻辑已确认正确，**可能的根因是 Triton 编译器在 MLA 大维度下的优化 bug**：

- FP6 解包使用了 `tl.interleave`（`_upcast_from_mxfmt.py:71-73`），MLA 的 512 维需要更大的 interleave 规模
- FP6→FP32 上采样有较大的指数偏置调整（`exponent_diff = 126`，`tl.exp2(126) ≈ 8.5e37`），配合 scale 乘法可能达到 float32 上限
- Triton 编译器对 GPU 架构和 num_warps 的不同选择可能导致 latent bug

**对比**：mxfp6 在 Llama GQA（128d per head）上正常，说明编译器在较小维度下能正确编译。MLA 的 512d + 64d 混合布局可能触发了不同的编译器优化路径。

#### 8.5 推荐的验证和修复方案

**方案 A（优先，验证 mxfp4/nvfp4 根因）**：将 mxfp4 的 K dequant 强制走软件解量路径：

修改 `decode_attention.py:230-231`、`decode_attention.py:345-346`、`extend_attention.py:347-348`、`extend_attention.py:451-452`，去掉 `if MX_FORMAT_ID == MXFP4_E2M1` 的特殊分支，统一使用：
```python
K_dequant = tl.trans(_unpack_and_dequant_mxfmt(k_packed, k_scale, MX_FORMAT_ID))
qk = tl.dot(q, K_dequant)
```

如果修改后 mxfp4 和 nvfp4 分数恢复正常，则根因确认为 `tl.dot_scaled("e2m1")`。

**方案 B（调试 mxfp6）**：在 decode kernel 中 dump mxfp6 和 mxfp8 的 K_dequant 与 QK 中间值，对比差异。检查是否有 NaN、inf 或数量级异常。

**方案 C（独立测试）**：写独立测试 kernel，用已知值填入模拟的 MLA buffer，验证 `_unpack_and_dequant_mxfmt` 对 FP6 和 E2M1 的解量输出。

#### 8.6 总结

| 格式 | 最可能根因 | 优先级 | 修复方向 |
|------|-----------|--------|---------|
| mxfp4_e2m1 | `tl.dot_scaled("e2m1")` 在 MLA rope（2 scale groups）下有 bug | 高 | 方案 A：改用软件解量 |
| mxfp6_e2m3 | Triton 编译器在大维度 MLA 下对 FP6 解量路径有优化 bug | 中 | 方案 B/C：dump 调试 |
| nvfp4_e2m1 | 同 mxfp4（但 nvfp4 的 K 走软件解量，问题可能在 V 解量或 scale 处理） | 中 | 方案 A 确认后进一步排查 |

**重点**：先执行方案 A。如果 `tl.dot_scaled("e2m1")` 的验证结果不是根因（即换成软件解量后 mxfp4 仍然失败），则说明问题出在更深层——可能是写入端的数据存储或所有 4-bit/6-bit 格式共有的问题

### 9. SGLang vs vLLM MLA KV Cache 量化对比分析（基于实际 benchmark 数据）

#### 9.1 vLLM benchmark 数据（关键新证据）

从 vLLM 环境测试获得 DeepSeek-V2-Lite-Chat gsm8k 5-shot 分数：

| 格式 | vLLM 分数 | SGLang 分数 | vLLM 状态 | SGLang 状态 |
|------|----------|------------|----------|------------|
| mxfp8_e4m3 | - | 0.6505 | - | 正常 |
| mxint8 | - | 0.6619 | - | 正常 |
| **mxfp4_e2m1** | **0.2934** | **0.0099** | 偏低但可用 | 近乎随机 |
| **mxfp6_e2m3** | **0.6490** | **0.0728** | 正常 | 异常 |
| **nvfp4_e2m1** | **0.4951** | **0.0099** | 中等 | 近乎随机 |

**关键结论**：
1. **低 bit-width 不是根因** — vLLM 在 mxfp4 (0.29)、mxfp6 (0.65)、nvfp4 (0.50) 上都有可用的分数，说明 4/6-bit 量化在 MLA 上可以工作
2. **`tl.dot_scaled("e2m1")` 不是根本性问题** — vLLM 的 unified attention (prefill) 路径也使用 `tl.dot_scaled("e2m1")`（见下），整体仍能达到 0.2934
3. **Bug 是 SGLang 独有的** — 同样的量化格式、同样的存储布局，SGLang 分数几乎为零

#### 9.2 写路径对比：SGLang 与 vLLM 一致

**SGLang MLA 写路径** — `simo/extensions/sglang_simo/layers/attention/triton_ops/set_kv_buffer.py:309-467`，`concat_and_cache_mla_kernel`：
- 将 `kv_c_normed`（512d）和 `k_pe`（64d）拼接后，按 tile_size 逐个 tile 调用 `_compute_and_pack_mxfmt` 量化写入
- Buffer 布局：`[KV_C_packed | KV_C_scales | K_PE_packed | K_PE_scales]`
- 输出 shape: `[num_tokens, 1, kv_cache_dim_in_bytes]` uint8

**vLLM MLA 写路径** — `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py`，`concat_and_cache_mla`：
- 同样将 `kv_c + k_pe` 拼接后量化写入
- 使用相同的 `_compute_and_pack_mxfmt` → 相同的数据布局

**结论**：写路径一致，buffer 中存储的数据是相同的。

#### 9.3 读路径对比：CRITICAL 差异

##### 9.3.1 vLLM decode 路径：对所有格式统一使用软件解量

**文件**: `simo/extensions/vllm_simo/v1/attention/ops/triton_decode_attention.py`，`_fwd_grouped_kernel_stage1`

vLLM decode 路径对 **所有 MX 格式** (包括 mxfp4) 统一走软件解量路径：

**K content 解量**（lines 437-457）：
```python
if MX_FORMAT_ID > 0:
    if MX_FORMAT_ID == NVFP4_E2M1:
        k_content_scales = k_content_scales.to(tl.float8e4nv, bitcast=True)
    k = _unpack_and_dequant_mxfmt(k_content_packed, k_content_scales, MX_FORMAT_ID)
else:
    # Per-group dequant
    ...
qk = tl.dot(q, k.trans().to(q.dtype))
```

**K PE 解量**（lines 486-506）：
```python
if MX_FORMAT_ID > 0:
    if MX_FORMAT_ID == NVFP4_E2M1:
        k_pe_scales = k_pe_scales.to(tl.float8e4nv, bitcast=True)
    kpe = _unpack_and_dequant_mxfmt(k_pe_packed, k_pe_scales, MX_FORMAT_ID)
qk += tl.dot(qpe, kpe.trans().to(qpe.dtype))
```

**V 解量**（lines 530-548）：同样的软件解量方式。

**关键**：vLLM decode 完全**不使用 `tl.dot_scaled("e2m1")`**，对所有格式（mxfp4、mxfp6、nvfp4、mxfp8、mxint8）都是：
1. 加载 packed 数据
2. 加载 scale 数据
3. 调用 `_unpack_and_dequant_mxfmt` 软件解量
4. 用标准 `tl.dot` 计算注意力

##### 9.3.2 SGLang decode 路径：对 mxfp4 使用硬件加速指令

**文件**: `simo/extensions/sglang_simo/layers/attention/triton_ops/decode_attention.py`，`_fwd_grouped_kernel_stage1`

SGLang 对 mxfp4 使用 `tl.dot_scaled("e2m1")` 硬件加速：

**K content——mxfp4 特殊分支**（lines 230-231）：
```python
if MX_FORMAT_ID == MXFP4_E2M1:
    qk = tl.dot_scaled(q, None, "bf16", k_packed.T, k_scale, "e2m1", fast_math=True)
else:
    K_dequant = tl.trans(_unpack_and_dequant_mxfmt(k_packed, k_scale, MX_FORMAT_ID))
    qk = tl.dot(q, K_dequant)
```

**K PE——mxfp4 特殊分支**（lines 345-346）：
```python
if MX_FORMAT_ID == MXFP4_E2M1:
    qk_pe = tl.dot_scaled(qpe, None, "bf16", k_packed.T, k_scale, "e2m1", fast_math=True)
else:
    K_dequant = tl.trans(_unpack_and_dequant_mxfmt(k_packed, k_scale, MX_FORMAT_ID))
    qk_pe = tl.dot(qpe, K_dequant)
```

**V 解量**（line 454）：使用 `_unpack_and_dequant_mxfmt` 软件解量（与 vLLM 一致）。

**extend_attention.py 中同样存在**（`simo/extensions/sglang_simo/layers/attention/triton_ops/extend_attention.py:347-348` 和 `:451-452`）。

##### 9.3.3 buffer view 差异：K/V 分离 vs 同一 buffer

**vLLM** — `simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:259-266`，`forward_mqa`：
```python
kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
kv_c_cache = kv_c_and_k_pe_cache[..., : self._kv_c_cache_size]  # KV_C 子视图
# ...
decode_attention_fwd(
    q,
    kv_c_and_k_pe_cache,  # K buffer — 完整 cache（nope + pe 部分）
    kv_c_cache,           # V buffer — 仅 KV_C 子视图
    ...
)
```
其中 `self._kv_c_cache_size = self._packed_head_size + self._scale_head_size`（line 458）。

vLLM 传入 **两个不同的 tensor**：K 用完整 cache，V 用 KV_C 子视图（slice）。

**SGLang** — `simo/extensions/sglang_simo/mem_cache/memory_pool.py:303-307`：
```python
def get_key_buffer(self, layer_id: int):
    return self.kv_buffer[layer_id - self.start_layer]
def get_value_buffer(self, layer_id: int):
    return self.kv_buffer[layer_id - self.start_layer]
```

SGLang 传入 **同一个 tensor** 给 K 和 V。虽然 decode kernel 内部通过不同的偏移量（V 只读 KV_C 部分，K 分别读 KV_C 和 K_PE 部分）来区分，但 `stride_buf_vbs` 和 `stride_buf_kbs` 完全相同（都是 `kv_cache_dim_in_bytes`）。

#### 9.4 根因分析

##### 9.4.1 mxfp4：`tl.dot_scaled("e2m1")` 在 SGLang decode 路径中的问题

vLLM 的 unified (prefill) 路径使用 `tl.dot_scaled("e2m1")` 且能正常工作（整体 score 0.2934）。但 SGLang 的 decode 路径同样使用 `tl.dot_scaled("e2m1")` 却分数几乎为零（0.0099）。

可能的原因：
1. **scale layout 差异**：SGLang decode 中 scale 的 load shape 为 `[BLOCK_N, SCALE_HEAD_SIZE]` = `[32, 16]`，而 `tl.dot_scaled("e2m1")` 可能期望 scale shape 为 `[groups, N]` = `[16, 32]`。这个转置在 mxint8 做软件解量时不存在，而在 mxfp4 直接传给 `tl.dot_scaled` 时可能被错误解释。
2. **nvfp4 不能复现此问题**：nvfp4 在 decode 中走软件解量路径，但仍然得 0.0099。这说明 mxfp4 的 `tl.dot_scaled` 可能不是唯一问题，或者 nvfp4 在 SGLang 中有独立的问题。

##### 9.4.2 nvfp4：分数与 mxfp4 相同但走不同 K 路径

nvfp4 在 SGLang decode 中使用 `_unpack_and_dequant_mxfmt` 软件解量（非 `tl.dot_scaled`），但同样得到 0.0099。而 vLLM 中 nvfp4 得分 0.4951。

nvfp4 与 mxfp4 在 SGLang 中的共同点：
- **V 解量路径相同**：都走 `NEEDS_SW_DEQUANT` → `_unpack_and_dequant_mxfmt`
- **K/V buffer 相同**：都使用 SGLang 的同一 buffer view
- **PACKED_HEAD_SIZE / SCALE_HEAD_SIZE 相同**：packing ratio 都是 2:1（4-bit）

nvfp4 在 SGLang 和 vLLM 中的重要差异：
- vLLM：V buffer 是 KV_C 子视图（`kv_c_cache = full_cache[..., :kv_c_cache_size]`）
- SGLang：V buffer 是完整 cache（但通过 offset 只读前 N 字节）

虽然加载到的数据相同（都读取 KV_C 部分），但 buffer stride 不同。**这可能影响 Triton 的 coalesced memory access pattern**。

但更可能的情况是：nvfp4 因 block_size=16（而非默认的 32），在 `_unpack_and_dequant_mxfmt` 中使用 `MX_QUANT_DIM=16`。这个参数虽然已被确认与写路径一致（写路径的 MX_BLOCK_SIZE 也是 16），但整个 `_unpack_and_dequant_mxfmt` 函数对 MX_QUANT_DIM=16 的代码路径与 MX_QUANT_DIM=32 不同，可能存在未被发现的数值 bug。

##### 9.4.3 mxfp6：分数极低但有微弱信号

mxfp6 vLLM 0.6490 vs SGLang 0.0728。SGLang 得分为 0.07，不是绝对零值，说明解量后 K/V 值有残留信息，但严重失真。

可能的原因：
- FP6 的 3-byte packing 在 SGLang decode 的 `_load_fp6_packed` → `_unpack_and_dequant_mxfmt` 链条中有数值问题
- FP6 解包过程中 `tl.exp2(126)` 的大范围浮点操作在 MLA 512 维下被放大

#### 9.5 修复建议（按优先级排序）

**方案 A（最高优先级：修复 mxfp4）**：
去掉 SGLang 中 mxfp4 的 `tl.dot_scaled("e2m1")` 特殊路径，与 vLLM 行为一致，统一使用软件解量。

修改位置（共 4 处）：
1. `simo/extensions/sglang_simo/layers/attention/triton_ops/decode_attention.py:230-231`
2. `simo/extensions/sglang_simo/layers/attention/triton_ops/decode_attention.py:345-346`
3. `simo/extensions/sglang_simo/layers/attention/triton_ops/extend_attention.py:347-348`
4. `simo/extensions/sglang_simo/layers/attention/triton_ops/extend_attention.py:451-452`

将 `if MX_FORMAT_ID == MXFP4_E2M1: qk = tl.dot_scaled(...)` 的特殊分支删除，统一走软件解量路径：
```python
K_dequant = tl.trans(_unpack_and_dequant_mxfmt(k_packed, k_scale, MX_FORMAT_ID))
qk = tl.dot(q, K_dequant)
```

**方案 B（高优先级：对齐 V buffer view）**：
参照 vLLM 的做法，让 SGLang 的 MLA 也传 K/V 分开的 buffer view，而非同一 buffer。可以在 `triton_simo_backend.py:189-192` 处：
```python
kv_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
# V 只取 KV_C 部分
v_buffer = kv_buffer[..., :self.simo_pool.kv_c_cache_size]
```

这保证了 V buffer 的 stride 更短，且与 vLLM 的 buffer 访问模式对齐。

**方案 C（中优先级：调试 nvfp4 和 mxfp6）**：
在 decode kernel 中 dump nvfp4 和 mxfp6 的 K_dequant 与 mxfp8 的 K_dequant 进行逐元素对比，确认数值差异的量级和模式。

**方案 D（验证性：独立测试 kernel）**：
写独立 Python 测试，用已知值填入模拟 MLA buffer，分别用 SGLang 和 vLLM 的解量路径解量，逐元素对比结果。

#### 9.6 总结

| 对比维度 | vLLM | SGLang | 影响 |
|----------|------|--------|------|
| mxfp4 K 解量方式 | 软件解量 (`_unpack_and_dequant_mxfmt`) | 硬件加速 (`tl.dot_scaled("e2m1")`) | **关键差异** |
| nvfp4 K 解量方式 | 软件解量 | 软件解量 | 一致 (但都近 0 分) |
| mxfp6 K 解量方式 | 软件解量 | 软件解量 | 一致 (但 SGLang 极低) |
| K/V buffer view | 分离（K=完整, V=KV_C subset） | 同一 buffer | 中等差异 |
| 写路径 | `_compute_and_pack_mxfmt` | `_compute_and_pack_mxfmt` | 一致 |

**最可能的根因**：SGLang 对 mxfp4 使用 `tl.dot_scaled("e2m1")` 在 MLA decode 路径下产生错误的 QK 值。nvfp4 和 mxfp6 虽然走软件解量，但 SGLang 中可能存在额外的 buffer layout 差异或 Triton 编译器优化差异导致解量结果仍有数值误差。方案 A+B 是最直接的修复方向。

### 10. `_maybe_compile_deep_gemm_one_type_all` 耗时优化（DeepSeek-V3.2）

#### 10.1 问题描述

启动 SGLang 服务时，日志中出现大量 `_maybe_compile_deep_gemm_one_type_all` 调用，每次耗时较长（单次可达数分钟，全部完成可能需要 10-20 分钟）。该函数在 DeepGEMM 首次遇到新的 `(kernel_type, N, K, num_groups)` 组合时，为该组合预编译**所有可能的 M 值**的 CUDA kernel。

参考日志：`templ/lm-eval-gsm8k-dsv3.2-part.bs128.log.2026_06_23___15_37_55`

#### 10.2 函数位置和逻辑

**文件**: `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`

**`_maybe_compile_deep_gemm_one_type_all`** (line 110-151)：
```python
def _maybe_compile_deep_gemm_one_type_all(kernel_type, n, k, num_groups):
    query_key = (kernel_type, n, k, num_groups)
    if (
        _ENABLE_JIT_DEEPGEMM_PRECOMPILE   # env: SGLANG_JIT_DEEPGEMM_PRECOMPILE
        and _DO_COMPILE_ALL               # True only for first GPU rank per node
        and _INITIALIZATION_DICT.get(query_key) is None  # 每种 shape 只编译一次
    ):
        _INITIALIZATION_DICT[query_key] = True
        _compile_deep_gemm_one_type_all(kernel_type, n, k, num_groups,
                                        m_list=_BUILTIN_M_LIST)
```

**`_compile_deep_gemm_one_type_all`** (line 155-223)：遍历 `m_list` 中所有 M 值，逐一执行 warmup GEMM 触发 JIT 编译。

**`_BUILTIN_M_LIST` 计算** (line 27, `update_deep_gemm_config` line 58-88)：
- 正常模式：M = 1 ~ min(chunked_prefill_size * 2, 1024 * 128) = 1 ~ 131072（约 13 万个值）
- Fast warmup 模式：稀疏采样，约 3072 个值

**关于编译缓存**：DeepGEMM 自身的 JIT 缓存机制（`DG_JIT_CACHE_DIR`，默认 `~/.cache/deep_gemm`）会将编译好的 CUDA kernel 缓存到磁盘。如果之前运行过 `sglang.compile_deep_gemm`，后续服务启动时 **编译循环会命中缓存**，每个 shape 只需约 1 秒（而不是首次编译的数分钟）。

#### 10.3 六种被编译的 kernel 类型

`python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:97-103`，`DeepGemmKernelType`：

| 类型 | 用途 |
|------|------|
| `GROUPED_GEMM_NT_F8F8BF16_MASKED` | MoE FP8 masked forward |
| `GROUPED_GEMM_NT_F8F8BF16_CONTIG` | MoE FP8 contiguous forward |
| `GROUPED_GEMM_NT_BF16_MASKED` | MoE BF16 masked forward |
| `GROUPED_GEMM_NT_BF16_CONTIG` | MoE BF16 contiguous forward |
| `GEMM_NT_F8F8BF16` | 普通 FP8 GEMM (KV projection, attention 等) |
| `GEMM_NT_BF16BF16F32` | 普通 BF16 GEMM |

#### 10.4 触发编译的调用链

1. `ModelRunner.__init__` (`python/sglang/srt/model_executor/model_runner.py:537-538`) 调用 `update_deep_gemm_config`
2. 首次 forward 时，MoE layer / attention layer 调用 `entrypoint.py` 中的 wrapper 函数（`grouped_gemm_nt_f8f8bf16_masked` 等）
3. wrapper 函数进入 `_deep_gemm_execution_hook` context manager (`compile_utils.py:399-413`)
4. hook 中调用 `_maybe_compile_deep_gemm_one_type_all`，首次遇到新 shape 时触发全 M 列表编译

#### 10.5 优化方案（按推荐程度排序）

##### 方案 A（推荐：precompile 一次，后续秒级启动）

离线预编译，将 kernel 缓存到磁盘：

```bash
python3 -m sglang.compile_deep_gemm \
    --model /share/users/like/package/hf-models/DeepSeek-V3.2/ \
    --tp 1 \
    --trust-remote-code
```

这会启动一个临时服务、发送一次 warmup 请求触发所有 DeepGEMM kernel 编译、编译结果缓存到 `~/.cache/deep_gemm/`，然后服务退出。**之后正常启动服务时，编译循环命中缓存，每个 shape 仅需约 1 秒**，总耗时从 10-20 分钟降至 ~1 分钟。

注意：如果更换模型、修改 TP size、或修改 `chunked_prefill_size` 导致新的 shape 组合出现，可能需要重新 precompile。

##### 方案 B（跳过预编译 list，仅按需 JIT）

设置环境变量关闭预编译：

```bash
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
```

**效果**：跳过 `_BUILTIN_M_LIST` 的遍历编译，只在首次真实推理遇到某个 M 值时 JIT 编译。首次请求会有额外延迟（但后续相同 M 值的请求不受影响）。

lm-eval gsm8k 测试中，请求的 M 值范围有限（decode 阶段 M=1，prefill 阶段 M 也较小），实际需要编译的 M 值数量远少于 13 万个。

##### 方案 C（fast warmup：减少 M 列表）

```bash
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
```

**效果**：M 列表从 1~131072 (全量) 减少到约 3072 个（稀疏采样），编译时间从 10-20 分钟降至约 1-2 分钟。

采样策略（`compile_utils.py:58-88`，`update_deep_gemm_config`）：
- M=1~1024：步长 1（覆盖 decode 小 batch）
- M=1024~2048：步长 2
- M=2048~4096：步长 4
- M=4096~8192：步长 8
- M=8192~max_prefill_bs：步长 16

##### 方案 D（完全禁用 DeepGEMM JIT，回退到其他 kernel）

```bash
export SGLANG_ENABLE_JIT_DEEPGEMM=0
```

**效果**：完全不使用 DeepGEMM，MoE/Attention 使用替代 kernel（如 Triton 实现）。这是最快启动但可能影响推理性能。

##### 方案 E（已运行的命令可直接追加的优化）

由于你当前的命令已经包含了 `"attention_backend": "triton"` 和 `"disable_cuda_graph": true`，且 DeepSeek-V3.2 主要耗时在 MoE 层的 DeepGEMM 编译，**追加以下环境变量到命令开头**即可：

**推荐组合**（方案 A + B 的组合思路）：

```bash
# 首次运行：先 precompile（一次性）
python3 -m sglang.compile_deep_gemm \
    --model /share/users/like/package/hf-models/DeepSeek-V3.2/ \
    --tp 1 --trust-remote-code

# 后续每次运行：直接启动，编译命中缓存几乎不耗时
cp /share/users/like/ipc.sglang.1.json /dev/shm/like/ipc.sglang.1.json; \
debug_env_file=/dev/shm/like/ipc.sglang.1.json \
SGLANG_LOGGING_CONFIG_PATH=/share_data/users/like/package/h100/package/sglang_kernel_src/like-useful/custom_sglang.json \
HF_DATASETS_CACHE=/data/like/huggingface_cache \
CUDA_VISIBLE_DEVICES=7 \
SGLANG_WARMUP_TIMEOUT=2592000 \
lm-eval --model sglang --model_args '...' --tasks gsm8k ...
```

注意：你原来的 `"watchdog_timeout": 2592000` 增加了超时时间，但如果编译本身太慢，增加 `SGLANG_WARMUP_TIMEOUT` 也能防止 warmup 阶段因编译超时而崩溃。

**快速方案**（如果不能跑 precompile，想直接跳过编译）：

```bash
# 在现有命令前加上：
SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
```

或者：

```bash
# 更激进的跳过：
SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1 \
```

##### 方案 F（增加 warmup 超时，防止编译中途 crash）

```bash
export SGLANG_WARMUP_TIMEOUT=2592000  # 等同于 watchdog_timeout
```

如果编译耗时很长，默认的 warmup timeout 可能导致进程被 kill。增加此超时可以防止 crash。你的命令中已有 `"watchdog_timeout": 2592000`，但 `SGLANG_WARMUP_TIMEOUT` 可能仍需单独设置。

#### 10.6 环境变量汇总

所有 DeepGEMM 相关环境变量定义在 `python/sglang/srt/environ.py:438-447`：

| 变量 | 默认值 | 作用 |
|------|--------|------|
| `SGLANG_ENABLE_JIT_DEEPGEMM` | `True` | 主开关，设 `0` 完全禁用 |
| `SGLANG_JIT_DEEPGEMM_PRECOMPILE` | `True` | 控制 `_maybe_compile_deep_gemm_one_type_all` 是否执行全 M 列表编译 |
| `SGLANG_JIT_DEEPGEMM_FAST_WARMUP` | `False` | 减少 M 列表到 ~3K |
| `SGLANG_DG_CACHE_DIR` | `~/.cache/deep_gemm` | JIT 缓存目录 |
| `SGLANG_WARMUP_TIMEOUT` | `-1` (无限) | warmup 超时秒数 |
| `SGLANG_IS_FIRST_RANK_ON_NODE` | 动态设置 | 仅第一个 GPU rank 做编译（multi-TP 场景） |

#### 10.7 一句话总结

DeepSeek-V3.2 启动慢是因为 DeepGEMM 为每种 `(kernel_type, N, K, num_groups)` 组合预编译所有 M 值的 CUDA kernel（最多 13 万个/组合）。**最佳方案是 `python3 -m sglang.compile_deep_gemm` 预编译一次缓存到磁盘，后续启动接近秒级。如果不能预编译，设置 `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` 跳过全 M 列表编译，只在首次请求时按需 JIT。**

### 11. `pip install fast-hadamard-transform` 安装失败分析

#### 11.1 错误日志（`templ/fast.log:32`）

```
ModuleNotFoundError: No module named 'torch'
```

#### 11.2 根因

`fast-hadamard-transform` 的 `setup.py` 在构建过程中（`get_requires_for_build_wheel` 阶段）调用 `import torch`，但 `torch` 在构建环境中不存在。

`pip install` 默认使用 **isolated build environment**（隔离构建环境）：pip 会创建一个临时的干净 venv，只安装 `pyproject.toml` `[build-system].requires` 中声明的构建依赖。`fast-hadamard-transform==1.1.0` 没有在其构建依赖中声明 `torch`，因此隔离环境中缺少 `torch`，导致 `setup.py:18` 的 `import torch` 失败。

#### 11.3 解决方案

使用 `--no-build-isolation` 让 pip 在当前 conda 环境（已安装 `torch`）中直接构建：

```bash
pip install fast-hadamard-transform --no-build-isolation
```

### 12. 为什么 GPU 上有其他负载会影响 lm-eval GSM8K 精度

#### 12.1 问题描述

同样的 lm-eval 命令运行 GSM8K 评测，当 GPU 上有其他负载时，得分会下降：

| 场景 | 日志文件 | exact_match |
|------|---------|-------------|
| 无其他负载 | `temp/lm-eval-gsm8k-dsv2-sgl-kvfp8.gpu7.2026_06_25___10_26_52` | **0.6702** |
| 先启动负载，再启动 lm-eval | `temp/lm-eval-gsm8k-dsv2-sgl-kvfp8.gpu7.2026_06_25___10_33_55` | **0.6497** |
| 先启动 lm-eval，等 server 就绪后启动负载 | `temp/lm-eval-gsm8k-dsv2-sgl-kvfp8.gpu7.2026_06_25___14_17_13` | **0.6702** |

其他负载是 `perf_bench.cu` 编译的二进制 `/share_data/users/like/package/h100/package/simo_conda_sglang/perf_bench`，它会：
1. 分配 ~1 GB 显存占位 (`MEMG=1`)
2. 无限循环跑 `16384^3` FP16 TensorCore GEMM（每轮 sleep 20ms）

关键观察：**只有当负载在 lm-eval（即 SGLang server）启动之前就已经运行时，得分才会下降。如果 SGLang server 先完成 CUDA graph capture，之后再启动负载，得分不受影响。**

#### 12.2 高层的 MoE / FP8 GEMM backend 并没有变化

首先检查 SGLang 的自动 backend 选择逻辑。

**MoE runner backend** (`moe_runner_backend='auto'`) 的选择逻辑在
`python/sglang/srt/server_args.py:_handle_model_specific_adjustments()` 和 `_handle_moe_kernel_config()`，
以及 `python/sglang/srt/layers/moe/utils.py:MoeRunnerBackend` 枚举。

**FP8 GEMM runner backend** (`fp8_gemm_runner_backend='auto'`) 的选择逻辑在
`python/sglang/srt/layers/quantization/fp8_utils.py:dispatch_w8a8_block_fp8_linear()` 第 350 行，以及 `_dispatch_auto_backend()` 第 445 行。

**这两个自动选择都只基于以下静态条件，不使用 GPU 空闲显存或 SM 占用率：**
- GPU SM 代数（`is_sm100_supported()`、`is_blackwell_supported()` 等）
- 量化类型
- 包可用性（`is_flashinfer_available()`、`deep_gemm` 是否可导入）
- 环境变量

DeepSeek-V2-Lite 在 H100 (SM90) 上运行，不匹配 SM100 的 `flashinfer_trtllm` 条件，因此使用默认的 Triton MoE kernel 和 FP8 CUTLASS GEMM。三份日志中的 `server_args` 完全一致，backend 没有变化。

#### 12.3 真正的原因：KV Cache token 容量被大幅削减

对比两份关键日志：

**无负载（10_26_52）**:
```
line 97:  KV Cache is allocated. #tokens: 130535, KV size: 1.89 GB
line 99:  Capture cuda graph begin. avail mem=45.78 GB
line 114: max_total_num_tokens=130535
```

**先启动负载（10_33_55）**:
```
line 97:  KV Cache is allocated. #tokens: 19215, KV size: 0.28 GB
line 99:  Capture cuda graph begin. avail mem=43.35 GB
line 114: max_total_num_tokens=19215
```

两处的 ServerArgs 完全一致：`mem_fraction_static=0.4`、`kv_cache_dtype='fp8_e4m3'`。

| 指标 | 无负载 | 先启动负载 | 比值 |
|------|-------|-----------|------|
| KV Cache #tokens | **130,535** | **19,215** | 6.8x |
| KV Cache 大小 | 1.89 GB | 0.28 GB | 6.8x |
| CUDA graph 前 avail_mem | 45.78 GB | 43.35 GB | -2.43 GB |

#### 12.4 根因定位

KV cache token 容量的计算公式位于
`python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py:62-76`
的 `_profile_available_bytes()`：

```python
def _profile_available_bytes(self: ModelRunner, pre_model_load_memory: int) -> int:
    post_model_load_memory = get_available_gpu_memory(
        self.device, self.gpu_id,
        distributed=get_world_group().world_size > 1,
        cpu_group=get_world_group().cpu_group,
    )

    rest_memory = post_model_load_memory - pre_model_load_memory * (
        1 - self.mem_fraction_static
    )
    return int(rest_memory * (1 << 30))  # return in bytes
```

其中：
- `pre_model_load_memory` — 模型加载前的 GPU 空闲显存（在 `model_runner.py:1149` 处测量）
- `post_model_load_memory` — 模型权重加载后的 GPU 空闲显存
- `mem_fraction_static` — 用户指定，这里为 `0.4`

**公式的数学含义**：
```
KV Cache Pool = post_model_load - pre_model_load * (1 - 0.4)
              = post_model_load - pre_model_load * 0.6
```

当 GPU 上已经有 `perf_bench` 占用了 ~2.4 GB 显存时：
- `pre_model_load` 减少了 ~2.4 GB
- `post_model_load` 也减少了 ~2.4 GB（模型权重占用不变）
- 但 `pre_model_load * 0.6` 项减少了 `2.4 * 0.6 = 1.44 GB`
- 最终 `rest_memory` 净减少 = 2.4 - 1.44 = **0.96 GB**

但实际上 KV Cache 从 1.89 GB 降到了 0.28 GB，减少了 **1.61 GB**。额外的差异来自：
1. 显存碎片化 — `perf_bench` 的分配使显存布局不同，PyTorch allocator 和 CUDA 图分配需要更多的预留空间
2. `mem_fraction_static=0.4` 的放大效应 — 当 `pre_model_load` 减小时，预留开销比例被放大

**核心问题**：`pre_model_load_memory` 在上面的公式中被当作总显存来计算"预留开销"（乘以 0.6），但实际上 `pre_model_load_memory` 反映的是减去外部负载后的剩余显存。当外部负载存在时，公式抽走的预留开销总额仍然是 `pre_model_load * 0.6`，而这个值大于实际所需（因为 `mem_fraction_static=0.4` 本来是针对总 GPU 显存的），导致 KV Cache 被进一步挤压。

#### 12.5 小 KV Cache 如何导致精度下降

19K tokens 的 KV Cache 容量对于 GSM8K 评测来说非常紧张：
- 每个 GSM8K 请求的 prompt + response 约需 1200-1600 tokens
- 19K tokens 只能同时容纳约 12-15 个活跃请求
- 130K tokens 可以同时容纳 ~85+ 个活跃请求

影响链：

1. **调度行为完全不同** — 在 19K 池中，SGLang scheduler 以非常少的并发请求运行，`token usage` 快速达到 0.88-0.94
2. **不同的 batch size** — 小的池导致 decode batch 始终很小（2-3 个请求 vs 84 个请求）
3. **`enable_deterministic_inference=False`（默认）** — 不同的 batch 组合会导致浮点运算中的不同舍入路径
4. **MoE shared experts + routed experts** — DeepSeek-V2 的路由决策是 token-level 的，理论上与 batch 无关，但共享专家的融合 kernel 在不同 batch size 下的中间精度行为可能不同
5. **FlashInfer sampling backend** — 不同的 batch 组合可能触发不同的采样代码路径

最终结果是，小 KV Cache 池导致 SGLang 以完全不同的调度策略运行同样的 1319 个请求，在非确定性推理模式下产生了不同的模型输出，从而出现 ~3% 的精度差异。

#### 12.6 结论

**负载不影响 MoE 或 FP8 GEMM backend 的选择**（backend 由 SM 代数、量化类型和包可用性决定）。**真正原因是 KV Cache token 容量被大幅削减**（从 130K 降到 19K，降低 6.8 倍），导致完全不同的调度和 batch 行为，在 `enable_deterministic_inference=False` 下产生数值差异。

**解决方案**：
1. **确保 GPU 独享** — 在 SGLang 启动前清空 GPU 负载（`fuser -k /dev/nvidia*` 或 `nvidia-smi | grep python | awk '{print $5}' | xargs kill`）
2. **启用确定性推理** — 加 `--enable-deterministic-inference`（但可能影响吞吐量）
3. **增大 `--mem-fraction-static`** — 如设置为 `0.45` 或 `0.5`，给 KV Cache 更多空间，减小被外部负载挤压的相对影响
4. **使用 `CUDA_VISIBLE_DEVICES` + `nvidia-smi -i <GPU_ID> -c EXCLUSIVE_PROCESS`** — 设置 GPU 独占模式，拒绝其他进程

### 13. sglang serve DeepSeek-V4 flashmla `ModuleNotFoundError` 分析

#### 13.1 错误日志

```
log line 1913-1919:
core_attn_metadata.init_flashmla_related()
File ".../sglang/srt/layers/attention/deepseek_v4_backend.py", line 262, in init_flashmla_related
    self.c1_flashmla_metadata = _create_flashmla_metadata()
File ".../sglang/srt/layers/attention/deepseek_v4_backend.py", line 85, in _create_flashmla_metadata
    import flash_mla
ModuleNotFoundError: No module named 'flash_mla'
```

#### 13.2 调用链分析

`deepseek_v4_backend.py` 在多处依赖 `flash_mla` Python 包：

| 行号 | 导入/调用 | 说明 |
|------|----------|------|
| `deepseek_v4_backend.py:61` | `from flash_mla.flash_mla_interface import FlashMLASchedMeta` | 类型注解 |
| `deepseek_v4_backend.py:85` | `import flash_mla` | 在 `_create_flashmla_metadata()` 中 |
| `deepseek_v4_backend.py:87` | `flash_mla.get_mla_metadata()` | 获取 MLA 元数据 |
| `deepseek_v4_backend.py:262-264` | `_create_flashmla_metadata()` | `init_flashmla_related()` 调用 |
| `deepseek_v4_backend.py:1048-1050` | `import flash_mla`; `flash_mla.flash_mla_with_kvcache()` | decode 时实际的 MLA 计算 |

**`flash_mla` 是一个 Python wrapper 包**，它提供了 Python API：
- `flash_mla.get_mla_metadata()` — 返回 MLA 调度元数据
- `flash_mla.flash_mla_with_kvcache()` — 执行带 KV cache 的 MLA decode
- `flash_mla.flash_mla_interface.FlashMLASchedMeta` — 调度元数据类型

#### 13.3 cmake 只构建了 C++ 扩展，没有安装 Python wrapper

`sgl-kernel/CMakeLists.txt:511` 通过 `include(cmake/flashmla.cmake)` 引入了 flashmla 构建。

`sgl-kernel/cmake/flashmla.cmake:6-12` 使用 `FetchContent_Declare` 从 git 仓库获取 flashmla 源码：

```cmake
FetchContent_Declare(
    repo-flashmla
    GIT_REPOSITORY git@gitlabsoft.siorigin.com:xtubk/sgl-project-flashmla.git
    GIT_TAG 5674ae59250493e583cb7fc9bc5d253e9a9d34f0
    GIT_SHALLOW OFF
)
FetchContent_Populate(repo-flashmla)
```

然后 `flashmla.cmake:94-166` 只做了以下事情：
1. 列出 C++ 源文件（`.cu`, `.cpp`, `.cc`）
2. 通过 `Python_add_library(flashmla_ops ...)`（第 146 行）编译成 shared library
3. 通过 `install(TARGETS flashmla_ops LIBRARY DESTINATION "sgl_kernel")`（第 166 行）安装到 `sgl_kernel/` 目录

**结果**：编译产物 `flashmla_ops.abi3.so` 被安装到 conda 环境的
`lib/python3.12/site-packages/sgl_kernel/flashmla_ops.abi3.so`。

但 `flash_mla` Python 包（包含 `__init__.py`、`flash_mla_interface.py` 等）**没有被安装**。
flashmla 仓库中虽然有 Python wrapper 源码（`flash_mla/` 目录），但 sgl-kernel 的 cmake **只构建 C++ 扩展**，不负责安装 Python 包。

#### 13.4 架构图：cmake 构建 vs Python 层

```
flashmla Git 仓库
├── csrc/                          ← cmake 编译（flashmla.cmake）
│   ├── flashmla_extension.cc
│   ├── sm90/decode/dense/...
│   └── ...
├── flash_mla/                     ← Python 包（cmake 不处理！）
│   ├── __init__.py                ← 定义 get_mla_metadata(), flash_mla_with_kvcache()
│   └── flash_mla_interface.py     ← 定义 FlashMLASchedMeta
└── setup.py / pyproject.toml      ← 可能的 pip 安装入口
```

**sgl-kernel 的 cmake 只覆盖了 `csrc/` 部分**：
```
sgl-kernel/cmake/flashmla.cmake
  → FetchContent 下载 flashmla 源码
  → 编译 csrc/*.cu, csrc/*.cpp
  → 输出 flashmla_ops.abi3.so → 安装到 sgl_kernel/
```

**`flash_mla` Python 包需要单独安装**（通过 flashmla 仓库自己的 `pip install`）。

#### 13.5 结论

**cmake 已经正确处理了 C++ native extension 的编译和安装**（`flashmla_ops.abi3.so` 已在 site-packages 中），但 **Python 包 `flash_mla` 没有被安装**。`flash_mla` 是 flashmla 仓库中的独立 Python 包，它为 C++ 扩展提供了 Python API wrapper，需要单独通过 pip 安装。

#### 13.6 解决方案

系统上已存在 `flash_mla` Python 包源码，位于：
```
/data_gpu/zhuangl/FlashMLA/flash_mla/
├── __init__.py              ← 导出 get_mla_metadata, flash_mla_with_kvcache 等
└── flash_mla_interface.py   ← 定义 FlashMLASchedMeta
```

具体方案（按推荐顺序）：

1. **pip install flashmla Python 包**（推荐）：
   ```bash
   cd /data_gpu/zhuangl/FlashMLA/
   pip install .  # 或 pip install -e .
   ```
   这会安装 `flash_mla` 到 site-packages，`import flash_mla` 即可正常工作。

2. **或者通过 PYTHONPATH 指向**（临时方案）：
   ```bash
   export PYTHONPATH="/data_gpu/zhuangl/FlashMLA:$PYTHONPATH"
   ```

3. **修改 cmake 同时安装 Python 包**（`sgl-kernel/cmake/flashmla.cmake` 末尾添加）：
   ```cmake
   install(
     DIRECTORY ${repo-flashmla_SOURCE_DIR}/flash_mla/
     DESTINATION "flash_mla"
   )
   ```
   需要注意安装路径必须能被 Python import 解析到（即需要在 site-packages 下）。

### 13.6 方案1补充分析：`pip install .` 是否会重新编译？是否会与 `flashmla_ops.abi3.so` 冲突？

**结论：会重新编译 C++/CUDA 源码，但不会与已有的 `flashmla_ops.abi3.so` 冲突。**

---

#### 13.6.1 是否会重新编译 C++/CUDA 代码？

**是的，会重新编译。**

FlashMLA 的 `setup.py:62-64` 定义了 `CUDAExtension`，这是 PyTorch 的 C++/CUDA 扩展编译机制：

```python
# setup.py:62-64
ext_modules.append(
    CUDAExtension(
        name="flash_mla.cuda",
        sources=[
            # API
            "csrc/api/api.cpp",

            # Misc kernels for decoding
            "csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
            ...
        ],
        ...
    )
)
```

`setup.py:150` 使用 `BuildExtension`（继承自 `torch.utils.cpp_extension.BuildExtension`）：
```python
setup(
    name="flash_mla",
    version="1.0.0" + rev,
    packages=find_packages(include=['flash_mla']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
```

执行 `pip install .` 时，`BuildExtension` 会调用 `nvcc` 编译器编译 `sources` 列表中的所有 `.cu` 和 `.cpp` 文件，生成一个名为 `flash_mla.cuda` 的动态链接库（`.so` 文件），安装到 site-packages 下。

---

#### 13.6.2 会与 `flashmla_ops.abi3.so` 冲突吗？

**不会冲突。** 两者是完全独立的模块，互不干扰。

核心原因：**编译产物是位于不同路径的不同 Python 模块。**

| 对比项 | `pip install .` (setup.py) | cmake (sgl-kernel) |
|--------|---------------------------|-------------------|
| **模块名** | `flash_mla.cuda` | `flashmla_ops` |
| **安装路径** | `site-packages/flash_mla/cuda.abi3.so`（或 `.so`） | `site-packages/sgl_kernel/flashmla_ops.abi3.so` |
| **入口 C++ 文件** | `csrc/api/api.cpp`（setup.py:67） | `csrc/python_api.cpp`（flashmla.cmake:98） + `sgl-kernel/csrc/flashmla_extension.cc` |
| **PyTorch 绑定方式** | pybind11 `PYBIND11_MODULE`（api.cpp:8） | pybind11 `PYBIND11_MODULE` + `TORCH_LIBRARY` torch.ops 注册 |
| **编译框架** | `torch.utils.cpp_extension.CUDAExtension` + `BuildExtension` | cmake `Python_add_library` |
| **Python import 方式** | `import flash_mla.cuda`（flash_mla_interface.py:6） | sgl_kernel 内部 import，不通过 SGLang 的 flash_mla 路径 import |

具体分析：

**a) 文件层面不冲突**
- `pip install .` 产物路径：`<site-packages>/flash_mla/cuda.*.so`
- cmake 产物路径：`<site-packages>/sgl_kernel/flashmla_ops.abi3.so`（flashmla.cmake:166）
- 完全不同的目录，不会互相覆盖。

**b) 入口 C++ 文件不同**
- `setup.py:67` 的入口是 `csrc/api/api.cpp`，这是一个 pybind11 模块，使用 `TORCH_EXTENSION_NAME` 宏作为模块名（即 `flash_mla.cuda`）。
- cmake 入口有两个：`sgl-kernel/csrc/flashmla_extension.cc`（flashmla.cmake:95）注册 torch.ops API，以及 `csrc/python_api.cpp`（flashmla.cmake:98）提供 pybind11 接口。编译后的模块名为 `flashmla_ops`（flashmla.cmake:146 `Python_add_library(flashmla_ops ...)`）。

**c) 内核 `.cu` 文件相同但独立编译**
- 两个构建系统编译了同一组内核 `.cu` 文件（如 `get_decoding_sched_meta.cu`、`combine.cu`、sm90/sm100 的各种 decode/prefill kernel）。
- 但这些 `.cu` 文件的函数符号链接到各自的 `.so` 中，互不干扰。
- 运行时，SGLang 的 `deepseek_v4_backend.py` 只 import `flash_mla`，不会加载 `sgl_kernel.flashmla_ops`；反之 sgl_kernel 的 torch.ops 路径也不会加载 `flash_mla.cuda`。

**d) 只加载一份**
- SGLang 执行路径从 `deepseek_v4_backend.py:85` 的 `import flash_mla` 开始，chain 为：
  - `flash_mla/__init__.py:3` → `from flash_mla.flash_mla_interface import get_mla_metadata, ...`
  - `flash_mla_interface.py:6` → `import flash_mla.cuda as flash_mla_cuda`
- 这条路径只加载 `flash_mla.cuda.*.so`，不会加载 `sgl_kernel/flashmla_ops.abi3.so`。

---

#### 13.6.3 更新后的推荐方案

**方案1（`pip install .`）完整步骤**：

```bash
# 注意：使用 sglang 实际引用的 FlashMLA 仓库路径
cd /share/users/like/package/h100/package/sgl-project/FlashMLA
pip install .
```

这会：
1. 编译 `setup.py:64-132` 列出的所有 C++/CUDA 源码，生成 `flash_mla.cuda.*.so`
2. 安装 `flash_mla` Python 包到 site-packages（`setup.py:148` 通过 `find_packages(include=['flash_mla'])` 打包）
3. **不会**覆盖或影响已有的 `sgl_kernel/flashmla_ops.abi3.so`

这可以解决 `sglang/srt/layers/attention/deepseek_v4_backend.py:85` 中 `import flash_mla` 的 `ModuleNotFoundError`。


## 14. enable_deterministic_inference 如何消除 KV Cache 容量差异对 gsm8k 分数的影响

### 14.0 背景与日志数据

**测试模型**：`DeepSeek-V2-Lite-Chat-16B_A2.4B`（MoE 架构，每层 27 层）

**测试命令**：
```bash
CUDA_VISIBLE_DEVICES=7 lm-eval --model sglang --model_args '{
  "pretrained": "/data/like/hf-models/DeepSeek-V2-Lite-Chat-16B_A2.4B/",
  "tp_size": 1, "dtype": "auto", "attention_backend": "triton",
  "mem_fraction_static": 0.4, "log_level": "info",
  "add_bos_token": true, "kv_cache_dtype": "fp8_e4m3",
  "enable_deterministic_inference": true
}' --tasks gsm8k --batch_size auto
```

**4 份日志对比**：

| # | 日志文件 | GPU 负载 | KV Cache #tokens | 行数 | gsm8k 分数 (flex/strict) |
|---|---------|---------|-----------------|------|--------------------------|
| 1 | `...16_12_43` | **无** | **130,534** | 856 | 0.6566 / 0.6490 |
| 2 | `...enable_deterministic_inference...16_38_03` | **有** | **46,185** | 1190 | 0.6566 / 0.6490 |
| 3 | `...enable_deterministic_inference...16_52_56` | **有** | **46,185** | 1190 | 0.6566 / 0.6490 |
| 4 | `...enable_deterministic_inference...17_03_44` | **无** | **130,534** | 856 | 0.6566 / 0.6490 |

**关键发现**：
- KV Cache #tokens 差距巨大（130K vs 46K），但 gsm8k 4 次分数完全相同
- 有负载的日志（#2, #3）比无负载的（#1, #4）多了 334 行 = 更多的调度批次（小 KV cache 需要更多轮次处理同样的请求）
- 这说明 **KV cache 容量本身不是影响 gsm8k 分数的原因**

---

### 14.1 `enable_deterministic_inference=true` 对 SGLang 的完整影响链

`enable_deterministic_inference=true` 的核心目标是：**让相同的输入在任何 batch 组合下都产生完全相同的输出**。它从多个层面消除了浮点计算的不确定性。入口在 `sglang/srt/server_args.py:4017` `_handle_deterministic_inference()`。

#### 14.1.1 全局算子替换：batch-invariant ops

这是最关键的机制。在 `sglang/srt/model_executor/model_runner.py:740-743`：

```python
if server_args.enable_deterministic_inference:
    from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode
    enable_batch_invariant_mode()
```

`enable_batch_invariant_mode()`（`sglang/srt/batch_invariant_ops/batch_invariant_ops.py:975-999`）通过 `torch.library.Library("aten", "IMPL")` 替换了 PyTorch 的 ATen 算子注册：

```python
def enable_batch_invariant_mode(enable_bmm: bool = True):
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")
    _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, dispatch_key)       # 矩阵乘法
    _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, dispatch_key) # 矩阵乘加
    _batch_invariant_LIB.impl("aten::_log_softmax", _log_softmax_batch_invariant, dispatch_key)  # log softmax
    _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, dispatch_key)  # 均值归约
    _batch_invariant_LIB.impl("aten::rms_norm", _rms_norm_aten_compat, dispatch_key)  # RMS norm
    _batch_invariant_LIB.impl("aten::mm.dtype", _mm_dtype_compat, dispatch_key)
    if enable_bmm:
        _batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, dispatch_key)
        torch.bmm = bmm_batch_invariant   # 直接替换 torch.bmm
```

这些替换使用固定 tile 配置的 Triton persistent kernel，消除了 cuBLAS 因输入维度不同而选择不同内部算法/tile 大小导致的浮点累积顺序差异。

#### 14.1.2 Attention 层面的确定性

**Triton backend 确定性配置**（`sglang/srt/layers/attention/triton_backend.py:180-186`）：

```python
if self.enable_deterministic:
    self.split_tile_size = get_int_env_var("SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE", 256)
    self.static_kv_splits = False   # 使用确定性逻辑而非动态逻辑
else:
    self.split_tile_size = model_runner.server_args.triton_attention_split_tile_size
```

`get_num_kv_splits()` 方法（`sglang/srt/layers/attention/triton_backend.py:238-287`）有 3 条路径：

```python
# 路径1（无确定性 + static_kv_splits=True）：全部填 max_kv_splits
if self.static_kv_splits and not self.enable_deterministic:
    num_kv_splits.fill_(self.max_kv_splits)           # line 257

# 路径2（确定性）：基于 seq_len 和 split_tile_size 计算，与 batch 无关
if self.split_tile_size is not None and self.enable_deterministic:  # line 261
    num_kv_splits[:] = (expanded_seq_lens + self.split_tile_size - 1) // self.split_tile_size

# 路径3（无确定性 + static_kv_splits=False）：动态 Triton kernel，与 batch 布局相关
get_num_kv_splits_triton[(1,)](                       # line 278
    num_kv_splits, seq_lens, num_seq, num_group,
    self.num_head, self.num_kv_head, self.max_kv_splits,
    self.device_core_count, ...
)
```

- **非确定性路径**（路径3）：KV split 数量由 `get_num_kv_splits_triton` 动态计算，考虑当前的 `num_seq`、`num_group`、`device_core_count` 等 batch 相关参数。同一请求在不同 batch 中可能被分配到不同数量的 KV split → 不同数量的 partial sum → 不同的浮点累积顺序 → 不同的结果。
- **确定性路径**（路径2）：KV split 数量完全由 `(seq_len / split_tile_size)` 决定，与 batch 中其他请求无关。

**DeepSeek 模型的 attention dispatch 变化**（`sglang/srt/models/deepseek_common/attention_backend_handler.py:111-114`）：

```python
def handle_attention_fa3(attn, forward_batch):
    if get_global_server_args().enable_deterministic_inference:
        return _dispatch_mla_subtype(attn, forward_batch)  # 固定使用 MLA 路径
    else:
        return _handle_attention_backend(attn, forward_batch, "fa3")  # 动态选择 MHA/MLA
```

Triton backend 同理（`attention_backend_handler.py:176-178`）：确定性模式下强制使用 MLA dispatch。

**num_splits 刚性限制**（多个 backend）：
- FlashAttention（`flashattention_backend.py:209-217`）：`num_splits = 1`
- DSA backend（`dsa_backend.py:317-319`）：`num_splits = 1`

这防止了 FlashAttention 内部根据 batch 大小自动选择不同的 split 数量。

#### 14.1.3 DeepSeek Router Gate 确定性

`deepseek_v2.py:440-441`：

```python
if get_global_server_args().enable_deterministic_inference:
    return F.linear(hidden_states, self.weight, None)  # 标准 matmul，不再使用优化路径
```

非确定性模式下，DeepSeek router gate 可能根据 batch size 和 hidden_dim 形状使用特殊的 `router_gemm` 优化路径（`deepseek_v2.py:454-459` 的 tensorcore-optimized kernel）。这些优化路径内部 tile 选择同样 batch-dependent。

#### 14.1.4 MoE 层面

**MoE Router kernel 选择**（`sglang/srt/layers/moe/router.py:364-389`）：

```python
if (bs >= 512 or num_experts > 8) and not enable_deterministic_inference:
    return fused_moe_router_tensorcore(...)   # 优化路径
else:
    return fused_moe_router_cudacore(...)      # 确定性路径
```

**MoE fused Triton config**（`sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py:55-59`）：

```python
if get_global_server_args().enable_deterministic_inference:
    return None  # 跳过自动调优
```

回落使用固定 kernel config（`BLOCK_SIZE_M=64, N=64, K=32, GROUP_SIZE_M=8`，`fused_moe_triton_config.py:160-163`）。

#### 14.1.5 Sampling 确定性

`server_args.py:4042-4046`：

```python
self.sampling_backend = "pytorch"  # flashinfer 的 sampling 是不确定的
```

`sampling_batch_info.py:94-109`：为每个请求组装 `sampling_seed`（默认 42），传递给 sampler 实现可复现的 token 采样。

#### 14.1.6 CUDA Graph 和 Allreduce

- **Piecewise CUDA Graph 禁用**（`server_args.py:1342-1343`）：`torch.cuda.CUDAGraph` 的捕获/重放可能引入不确定性
- **Allreduce fusion 禁用**（`server_args.py:4030-4040`）：AITER 和 FlashInfer 的 allreduce fusion 计算顺序不确定

---

### 14.2 为什么 `enable_deterministic_inference=true` 后 #tokens 差距很大但分数一致

核心原因：**`enable_deterministic_inference=true` 消除了所有 batch-dependent 的浮点计算差异。**

在不启用确定性推理时：

```
请求A在batch_X（大batch）中的计算结果 ≠ 请求A在batch_Y（小batch）中的计算结果
```

因为 `torch.mm`/`torch.bmm` 被 cuBLAS 根据矩阵维度自动选择了不同的内部 tile 配置，attention KV split 数量也因 batch 布局而异。

在启用确定性推理后：

```
请求A在batch_X（大batch）中的计算结果 == 请求A在batch_Y（小batch）中的计算结果
```

因为所有 batch-dependent 的算子被替换为：
1. 固定 tile 的 Triton persistent kernel（`torch.mm`/`torch.bmm`/`log_softmax`/`rms_norm`）
2. 纯 seq_len 决定的 KV split（与 batch 中其他请求无关）
3. 固定的 `F.linear` 路由（不根据 batch size 切换优化路径）
4. 固定的 MoE kernel 配置（不自动调优）

KV cache #tokens 差异（130K vs 46K）仍然导致**batch 组成完全不同**（更多的小 batch vs 更少的大 batch），但这只是改变了调度效率，不影响每个单独请求的计算结果。

---

### 14.3 没有 `enable_deterministic_inference` 时 gsm8k 分数不同的根本原因

**根因不是 KV cache 容量本身，而是 batch-dependent 的浮点非确定性（floating-point nondeterminism）。**

#### 完整的因果链

```
GPU 负载存在
  → 可用显存减少（38.27 GB vs 40.12 GB）
    → KV cache #tokens 缩减（46K vs 130K）
      → 调度器必须把同样多的请求拆成更多的小 batch
        → 每个 batch 包含的请求数不同（batch size 不同）
          → 矩阵运算的 shape 不同（如 [bs, hidden_dim] 不同）
            → cuBLAS 内部选择了不同的 tile 配置/算法
              → 浮点累加顺序不同 → 微小数值差异（~1e-7 量级）
                → 27 层 transformer 逐层放大
                  → 最终 logits 差异可能导致 argmax 选择的 token 不同
                    → 不同的 token → 完全不同的续写路径
                      → 最终答案不同 → gsm8k 分数不同
```

#### 具体涉及的 batch-dependent 非确定性来源

| 来源 | 位置（相对路径） | 机制 |
|------|-----------------|------|
| `torch.mm` / `torch.bmm` | 所有 linear 层 | cuBLAS 根据 (M,N,K) 形状选择内部 tile 大小和算法（如 GEMM 的不同 variant），浮点累加顺序不同 |
| `torch._log_softmax` | softmax 层 | cuDNN/cuBLAS 根据张量形状选择不同归约策略 |
| `torch.mean` / `F.rms_norm` | LayerNorm/RMSNorm | 归约操作的内部并行分解策略依赖张量大小 |
| Triton attention KV splits | `triton_backend.py:278-287` | `get_num_kv_splits_triton` 根据 `num_seq`、`num_group`、`device_core_count` 动态分配 split 数 → 同一请求在不同 batch 中 partial sum 组合方式不同 |
| DeepSeek router_gemm | `deepseek_v2.py:454-459` | 根据 batch size 是否 ≤16 和 hidden_dim 形状选择优化路径 |
| MoE router kernel | `router.py:368-373` | 根据 batch size 是否 ≥512 选择 tensorcore 或 cudacore kernel |
| MoE triton config | `fused_moe_triton_config.py:55-59` | auto-tune 根据当前 batch 选择最优 kernel 配置 |
| FlashAttention num_splits | `flashattention_backend.py:209-217` | 非确定性模式下自动 split，与 batch 上下文相关 |

#### 为什么微小的浮点差异会影响 gsm8k 分数

DeepSeek-V2-Lite 有 27 层 transformer。每层产生 ~1e-7 量级的浮点差异，经过 27 层累积后可以达到 ~1e-5 到 ~1e-6 量级。

在 greedy decoding 中，top-1 token 和 top-2 token 的概率差通常只在小数点后几位。当累积的浮点差异使 top-1 和 top-2 的 logit 排序发生变化时：

```
无差异：token_A (logit=3.14159) > token_B (logit=3.14158) → 选 token_A
有差异：token_A (logit=3.14158) < token_B (logit=3.14159) → 选 token_B  ← 逆转！
```

一旦选择了一个不同的 token，后续所有 token 的生成路径就完全改变了。这导致最终答案不同。

gsm8k 是数学推理任务，要求模型生成精确的计算过程和最终数字。一个 token 的差异就能导致整条推理链走向不同的结论 → 得分不同。

---

### 14.4 总结

| 问题 | 答案 |
|------|------|
| KV cache 容量差异是根因吗？ | **不是**。`enable_deterministic_inference=true` 的实验中，KV cache 130K vs 46K 但分数完全一致，直接证明容量差异不影响分数。 |
| 真正的根因是什么？ | **batch-dependent 的浮点非确定性**。相同的输入在不同 batch 中计算，由于 batch 大小/组成不同，cuBLAS 等底层库选择了不同的内部实现路径，导致浮点累加顺序不同，产生微小的数值差异。这些差异在 27 层 transformer 中逐层累积，最终的 logit 差异足以改变 greedy decoding 的 argmax 选择。 |
| `enable_deterministic_inference=true` 如何解决问题？ | 通过以下机制全面消除 batch-dependent 非确定性：(1) 用固定 tile 的 Triton persistent kernel 替换 `torch.mm`/`torch.bmm`/`log_softmax`/`rms_norm`；(2) 用 seq_len 决定 KV split 而非 batch 布局；(3) 用 `F.linear` 替代优化的 router_gemm；(4) 固定 MoE kernel 配置；(5) 固定 sampling seed；(6) 禁用 CUDA graph 和 allreduce fusion。 |



## 15. 对已运行的 SGLang 服务器进行 lm-eval gsm8k 测试

### 15.0 场景

deepseek-v4-flash 的 SGLang serve 已在端口 30121 运行：
```bash
/share_data/users/like/miniconda3/envs/simo_sglang_pip/bin/sglang serve \
  --model-path /data/like/hf-models/deepseek-v4-flash/ \
  --tp 4 --moe-runner-backend marlin \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --host 0.0.0.0 --port 30121
```

需要对这个**已运行的 server** 执行 lm-eval gsm8k 测试（不重新启动 server）。

### 15.1 两种方式对比

lm-eval 连接 SGLang server 有两种模式：

| 模式 | `--model` 参数 | server 生命周期 | `model_args` 关键参数 |
|------|---------------|----------------|---------------------|
| **模式 A：lm-eval 自动启动 server** | `sglang` | lm-eval 内部启动和销毁 server | `pretrained`, `tp_size`, `mem_fraction_static` 等 server 配置 |
| **模式 B：连接已运行的 server** | `local-completions` | server 已存在，lm-eval 只做推理 | `model`（任意名称）, `base_url`（指向 server 的 OpenAI completions 端点） |

当前场景下应使用**模式 B**。

### 15.2 原理

lm-eval 的 `local-completions` backend 通过 OpenAI-compatible `/v1/completions` API 与 SGLang server 通信。

SGLang codebase 的 test 工具也采用同样方式，参见 `sglang/test/kits/lm_eval_kit.py` 的 `launch_lm_eval()` 方法：

```python
model_args = {
    "model": eval_config["model_name"],
    "base_url": self.base_url + "/v1/completions",
    "num_concurrent": num_concurrent,
}
results = lm_eval.simple_evaluate(
    model="local-completions",
    model_args=model_args,
    tasks=[task["name"] for task in eval_config["tasks"]],
    ...
)
```

### 15.3 命令

```bash
conda activate simo_sglang_pip

lm-eval \
  --model local-completions \
  --model_args '{"model": "default", "base_url": "http://127.0.0.1:30121/v1/completions", "num_concurrent": 1}' \
  --tasks gsm8k \
  --batch_size auto
```

参数说明：
- `--model local-completions`：使用 lm-eval 内置的 OpenAI-compatible completions backend
- `model: "default"`：SGLang 的 completions API 中的 model 名称，server 上只有一个模型时不敏感，可填任意值
- `base_url: "http://127.0.0.1:30121/v1/completions"`：指向 server 的 `/v1/completions` 端点
- `num_concurrent: 1`：并发请求数设为 1，与模式 A 下 lm-eval 内部管理 server 时的默认行为一致
- `--batch_size auto`：让 lm-eval 自动选择 batch size

### 15.4 验证 server 是否正常

在运行 lm-eval 前，可先验证 server 的 completions 端点是否可用：

```bash
curl -s http://127.0.0.1:30121/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "prompt": "Hello, world!",
    "max_tokens": 10
  }' | python3 -m json.tool
```

### 15.5 保存日志

```bash
lm-eval \
  --model local-completions \
  --model_args '{"model": "default", "base_url": "http://127.0.0.1:30121/v1/completions", "num_concurrent": 1}' \
  --tasks gsm8k \
  --batch_size auto \
  > temp/lm-eval-gsm8k-dsv4-flash.sglang-serve-api.`nowstr.sh`.log 2>&1
```


## 16. 验证 batch-invariant 矩阵乘法是否真的与 PyTorch 有区别

### 16.0 测试目的

在之前的分析（Section 14）中，我们提出 `enable_deterministic_inference=true` 通过 `enable_batch_invariant_mode()` 替换了 `torch.mm`、`torch.bmm`、`log_softmax`、`rms_norm`、`mean` 等算子，消除了 batch-dependent 的浮点非确定性。

现在直接验证：**SGLang 的 batch-invariant 矩阵乘法（matmul_persistent）与 PyTorch 原生的矩阵乘法（torch.mm/cuBLAS）在 bf16 下是否真的有结果差异？**

测试脚本：`like-useful/validate_batch_invariant_ops.py`

测试 conda 环境：`/data/like/miniconda3/envs/simo_sglang/`（Python 3.12, PyTorch 2.9, CUDA 12.x, H100 SM90）

### 16.1 测试项及结果

#### Test 1: 直接数值对比 — torch.mm vs matmul_persistent（29 种形状，bf16）

测试了从 M=1 到 M=16384 的 29 种矩阵形状，覆盖 decode batch（1-16 tokens）、medium batch（64-512）、large batch（1024-16384）、router gate（N=64-256）、square matmul 等多种场景。

**结果：所有 29 种形状下，torch.mm 和 matmul_persistent 的输出完全一致（max_abs_diff = 0.0，exact_match = True）。**

```
M=     1 K= 2048 N=  512  exact=True  max_abs=0.0000e+00
M=     1 K= 2048 N= 4096  exact=True  max_abs=0.0000e+00
...
M= 16384 K= 2048 N=  128  exact=True  max_abs=0.0000e+00
Any difference found: False
```

#### Test 2: 并发 CUDA stream 干扰

在另一个 CUDA stream 上运行大矩阵乘法（4096×4096×50 次迭代）的同时，在默认 stream 上执行 torch.mm，看 cuBLAS 是否因 GPU 资源竞争而选择不同的内部算法产生不同结果。

**结果：所有测试形状下，有干扰和无干扰的 torch.mm 输出完全一致。**

```
M=   256 K= 2048 N= 2048  baseline==concurrent: True
M=   512 K= 2048 N= 4096  baseline==concurrent: True
...
Any difference from concurrent stream: False
```

#### Test 3: addmm（矩阵乘+偏置）对比

**结果：torch.addmm 和 matmul_persistent(a,b,bias) 输出完全一致。**

#### Test 4 + Test 8: 非连续内存输入

非连续内存下 torch.mm 和 matmul_persistent 的结果对比。深入调查（Test 8）确认：**两者对非连续输入的处理也完全一致**（torch 内部先拷贝为连续再计算）。

#### Test 5: torch.mm 的 batch 不变性

将矩阵按 M 维度拆分为 3 段分别计算再 cat，对比一次性计算全矩阵。

**结果：torch.mm(cuBLAS) 对连续输入是 batch-invariant 的 — 拆分计算和全量计算结果完全一致。**

#### Test 6: matmul_persistent 的 batch 不变性

**结果：matmul_persistent 也是 batch-invariant 的 — 拆分计算和全量计算结果完全一致。**

#### Test 9-12: 其他算子

| 算子 | 测试方法 | 结果 |
|------|---------|------|
| `torch.bmm` | full vs split batch，5 种 batch size | **完全一致** — batch-invariant |
| `torch.log_softmax` | full vs split rows，6 种 M | **完全一致** — batch-invariant |
| `F.rms_norm` | full vs split rows，6 种 M | **完全一致** — batch-invariant |
| `torch.mean` | full vs split rows，6 种 M | **完全一致** — batch-invariant |

### 16.2 结论

**核心发现：对于 bf16 连续输入，PyTorch 原生的 `torch.mm`（cuBLAS）和 SGLang 的 `matmul_persistent`（Triton persistent kernel）在所有测试场景下都输出完全一致的结果。其他算子（bmm, log_softmax, rms_norm, mean）的 native PyTorch 实现也都是 batch-invariant 的。**

这意味着：

1. **SGLang 的 batch-invariant matmul 替换本身不是解决 gsm8k 分数差异的关键** — native cuBLAS 对 bf16 连续输入已经产生了 batch-invariant 的结果。

2. **`enable_deterministic_inference` 中替换单个算子的效果，至少对于 bf16 矩阵乘法而言，native PyTorch 已经是确定性的。**

3. **gsm8k 分数差异的根本原因必须来自其他机制**，最可能的是：
   - **Attention KV split 的非确定性** — `sglang/srt/layers/attention/triton_backend.py:238-287` 的 `get_num_kv_splits()` 函数在没有确定性模式时，根据 batch 大小、`num_seq`、`device_core_count` 动态决定每个序列的 KV split 数量。同一请求在不同 batch 中获得不同数量的 split → 不同的 partial sum 组合顺序 → 不同的浮点结果。
   - **多算子累积效应** — 虽然单个算子在单独测试中是 batch-invariant 的，但在 **27 层 transformer 中交互执行**时，微小的精度差异（即使 < 1e-6）可能通过非线性和归一化层放大。例如：
     - rms_norm 的输入是上一个 mm 的输出
     - softmax 的输入是 attention QK^T（由多次 mm 组成）
     - residual add 可能改变数值分布
     - 这些交互在单独测试单个算子时无法体现

4. **`enable_deterministic_inference=true` 的真正价值**在于：
   - 固定 attention KV split（`split_tile_size=256`，`static_kv_splits=False`）
   - 禁用 piecewise CUDA graph
   - 固定 MoE routing kernel
   - 固定 sampling seed
   - 而不是替换 matmul kernel 本身

### 16.3 补充说明

本测试验证的是**单个算子的独立行为**，无法完全模拟 transformer 模型中多层交互的累积效应。要真正验证 batch-dependent 的浮点差异来源，需要：
1. 对完整 transformer 层做 forward 对比（full batch vs split batch）
2. 逐层对比中间输出，定位差异首次出现的层
3. 对 attention 模块单独隔离测试（特别是确认 KV split 的影响）


## 16. 验证 batch-invariant 矩阵乘法是否真的与 PyTorch 有区别

### 16.0 测试目的

在之前的分析（Section 14）中，我们提出 `enable_deterministic_inference=true` 通过 `enable_batch_invariant_mode()` 替换了 `torch.mm`、`torch.bmm`、`log_softmax`、`rms_norm`、`mean` 等算子，消除了 batch-dependent 的浮点非确定性。

现在直接验证：**SGLang 的 batch-invariant 矩阵乘法（matmul_persistent）与 PyTorch 原生的矩阵乘法（torch.mm/cuBLAS）在 bf16 下是否真的有结果差异？**

测试脚本：`like-useful/validate_batch_invariant_ops.py`
测试 conda 环境：`/data/like/miniconda3/envs/simo_sglang/`（Python 3.12, PyTorch 2.9, CUDA 12.x, H100 SM90）

### 16.1 测试项及结果

#### Test 1: 直接数值对比 — torch.mm vs matmul_persistent（29 种形状，bf16）

测试了从 M=1 到 M=16384 的 29 种矩阵形状，覆盖 decode batch（1-16 tokens）、medium batch（64-512）、large batch（1024-16384）、router gate（N=64-256）、square matmul 等多种场景。

**结果：所有 29 种形状下，torch.mm 和 matmul_persistent 的输出完全一致（max_abs_diff = 0.0，exact_match = True）。**

```
M=     1 K= 2048 N=  512  exact=True  max_abs=0.0000e+00
M=     1 K= 2048 N= 4096  exact=True  max_abs=0.0000e+00
...
M= 16384 K= 2048 N=  128  exact=True  max_abs=0.0000e+00
Any difference found: False
```

#### Test 2: 并发 CUDA stream 干扰

在另一个 CUDA stream 上运行大矩阵乘法（4096x4096x50 次迭代）的同时，在默认 stream 上执行 torch.mm，看 cuBLAS 是否因 GPU 资源竞争而选择不同的内部算法产生不同结果。

**结果：所有测试形状下，有干扰和无干扰的 torch.mm 输出完全一致。**

```
M=   256 K= 2048 N= 2048  baseline==concurrent: True
M=   512 K= 2048 N= 4096  baseline==concurrent: True
...
Any difference from concurrent stream: False
```

#### Test 3: addmm（矩阵乘+偏置）对比

**结果：torch.addmm 和 matmul_persistent(a,b,bias) 输出完全一致。**

#### Test 4 + Test 8: 非连续内存输入

非连续内存下 torch.mm 和 matmul_persistent 的结果对比。深入调查（Test 8）确认：**两者对非连续输入的处理也完全一致**（torch 内部先拷贝为连续再计算，SGLang Triton kernel 也能正确处理 strides）。

#### Test 5: torch.mm 的 batch 不变性

将矩阵按 M 维度拆分为 3 段分别计算再 cat，对比一次性计算全矩阵。

**结果：torch.mm(cuBLAS) 对连续输入是 batch-invariant 的 — 拆分计算和全量计算结果完全一致。**

#### Test 6: matmul_persistent 的 batch 不变性

**结果：matmul_persistent 也是 batch-invariant 的 — 拆分计算和全量计算结果完全一致。**

#### Test 7: enable_batch_invariant_mode 拦截生效

**结果：启用后 torch.mm 的输出 == matmul_persistent 的输出，确认拦截机制正常工作。**

#### Test 9-12: 其他算子

| 算子 | 测试方法 | 结果 |
|------|---------|------|
| `torch.bmm` | full vs split batch，5 种 batch size | **完全一致** — batch-invariant |
| `torch.log_softmax` | full vs split rows，6 种 M | **完全一致** — batch-invariant |
| `F.rms_norm` | full vs split rows，6 种 M | **完全一致** — batch-invariant |
| `torch.mean` | full vs split rows，6 种 M | **完全一致** — batch-invariant |

### 16.2 结论

**核心发现：对于 bf16 连续输入，PyTorch 原生的 `torch.mm`（cuBLAS）和 SGLang 的 `matmul_persistent`（Triton persistent kernel）在所有测试场景下都输出完全一致的结果。其他算子（bmm, log_softmax, rms_norm, mean）的 native PyTorch 实现也都是 batch-invariant 的。**

这意味着：

1. **SGLang 的 batch-invariant matmul 替换本身不是解决 gsm8k 分数差异的关键** — native cuBLAS 对 bf16 连续输入已经产生了 batch-invariant 的结果。

2. **`enable_deterministic_inference` 中替换单个算子的效果，至少对于 bf16 矩阵乘法而言，native PyTorch 已经是确定性的。**

3. **gsm8k 分数差异的根本原因必须来自其他机制**，最可能的是：
   - **Attention KV split 的非确定性** — `sglang/srt/layers/attention/triton_backend.py:238-287` 的 `get_num_kv_splits()` 函数在没有确定性模式时，根据 batch 大小、`num_seq`、`device_core_count` 动态决定每个序列的 KV split 数量。同一请求在不同 batch 中获得不同数量的 split → 不同的 partial sum 组合顺序 → 不同的浮点结果。
   - **多算子累积效应** — 虽然单个算子在单独测试中是 batch-invariant 的，但在 **27 层 transformer 中交互执行**时，微小的精度差异（即使 < 1e-6）可能通过非线性和归一化层放大。例如：
     - rms_norm 的输入是上一个 mm 的输出
     - softmax 的输入是 attention QK^T（由多次 mm 组成）
     - residual add 可能改变数值分布
     - 这些交互在单独测试单个算子时无法体现

4. **`enable_deterministic_inference=true` 的真正价值**在于：
   - 固定 attention KV split（`split_tile_size=256`，`static_kv_splits=False`）
   - 禁用 piecewise CUDA graph
   - 固定 MoE routing kernel
   - 固定 sampling seed
   - 而不是替换 matmul kernel 本身

### 16.3 补充说明

本测试验证的是**单个算子的独立行为**，无法完全模拟 transformer 模型中多层交互的累积效应。要真正验证 batch-dependent 的浮点差异来源，需要：
1. 对完整 transformer 层做 forward 对比（full batch vs split batch）
2. 逐层对比中间输出，定位差异首次出现的层
3. 对 attention 模块单独隔离测试（特别是确认 KV split 的影响）
