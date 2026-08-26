# 1. `hp_to_mx` 如何知道内部 tile 尺寸 + 模板参数解释

## 1.1 问题回顾

以 `test_quantize_to_mxint8` 中 `x=[7,96]` 的 BF16 tensor 为例，Python 端经过 padding → `linear_to_tileformat` 后得到 `[32, 128]` 的二维 tensor。`linear_to_tileformat` 做的 `permute(0,2,1,3)` 把 `[tile_m, row_in_tile, tile_n, col_in_tile]` 转成 `[tile_m, tile_n, row_in_tile, col_in_tile]`，再 reshape 回 `[32, 128]`。这个二维 shape 里已经没有 `row_in_tile`、`col_in_tile` 的显式信息了。

而 C++ 端 `launch_quantize_to_mxint8` 只传了：

```cpp
::hp_to_mx<sifmt::mxint8, HPType, 0>(
    input.data_ptr(), output.data_ptr(), 1,
    input.size(0),   // 32
    input.size(1),   // 128
    stream);
```

`hp_to_mx` 的 host wrapper 内部需要知道**每个 input tile 有多少行**和**每行有多少字节**才能构造 tensor map 并 launch kernel。这些信息从哪来？

## 1.2 Rows（tile 行数）：运行时从 dim0 推导

`hp_to_mx` 的 `Rows` **不是从 Python 传入的模板参数**，而是 host wrapper 内部根据 `dim0`（即 `input.size(0)`）在**运行时选择**的。

在 `hp_to_mx_kernel.su:84-86`：

```cpp
auto launch = [&]<int Rows>() {
    // ... 构造 tensor map、launch kernel ...
};

if (dim0 > 16)      launch.template operator()<32>();
else if (dim0 > 8)  launch.template operator()<16>();
else                 launch.template operator()<8>();
```

以 `dim0=32` 为例：`32 > 16` → 选择 `Rows=32`。

这个选择随后通过 C++ 模板实例化传播到所有编译期常量（input tile 列数、output tile 列数、转换指令选择等），所以**不需要作为显式模板参数从 Python 传下来**。

## 1.3 每行字节数和列数：编译期常量

所有 input 类型的 **input staging tile 大小固定为 1024 bytes**：

```cpp
// hp_to_output_traits.hpp:24
struct HpToOutputRowGeometry<Rows> {
    static constexpr int tile_bytes = 1024;   // 固定
};
```

在此基础上，每行字节数和列数是纯编译期推导：

```cpp
// hp_to_mx_tensormap.hpp:73-74
input_tile_dim0 = 1024 / (Rows * element_bytes)
```

对于 BF16 + Rows=32：

```
element_bytes  = 2  (BF16)
input_tile_dim0 = 1024 / (32 * 2) = 16 列
```

所以每个 input staging tile 是 `32行 × 16列 × 2bytes = 1024 bytes`。这些值在模板实例化时已全部确定，不需要额外传递。

## 1.4 Tensor Map：硬件 tile 解释的关键

`hp_to_mx` 的 DTE (Data Transfer Engine) 路径使用 **tensor map** 来解释 flat buffer 中的 tile 结构。在 `hp_to_mx_tensormap.hpp:168-198` 中：

```cpp
encode_hp_to_mx_input_map(..., &input_map) {
    const std::array<uint32_t, 3> input_global_elements{
        input_map_cols,     // dim1 = 128
        input_map_rows,     // dim0 = 32
        shape.batch         // 1
    };
    const std::array<uint32_t, 3> input_box_elements{
        G::output_tile_dim0,   // BF16+Rows=32 → 32 列
        static_cast<uint32_t>(Rows),  // 32 行
        1
    };
    // TMAP_FORMAT_TILED: 按硬件 tiled layout 解释
    siTensorMapEncodeTiled(input_map, ..., input_global_elements, input_box_elements, 0);
}
```

`input_box_elements` 就是告诉硬件 DTE "每个 tile box 有多大"：`[32列, 32行, 1]`。

**这就是 Python 端 `linear_to_tileformat` 和硬件端的约定：**

- Python 把数据从 `[tile_m, row_in_tile, tile_n, col_in_tile]` 重排成 tile-major 的 flat buffer
- 硬件通过 tensor map 中的 `input_box_elements` 知道"每读 32 行、32 列就完成一个 tile"
- 两边对 tile 尺寸的约定一致，DTE 就能从正确的偏移读取数据

## 1.5 模板参数只有 3 个的完整解释

`hp_to_mx` 的完整模板签名是：

```cpp
template <typename OUT_T,          // ← 显式: sifmt::mxint8
          typename HP_T,           // ← 显式: sifmt::bfloat16
          int TNOCP,               // ← 显式: 0
          TmapFormat INPUT_LAYOUT = TMAP_FORMAT_TILED,  // ← 默认值
          bool ZERO_OUTPUT = false>                      // ← 默认值
void hp_to_mx(void* hp_ptr, void* out_ptr,
              int batch_size, int dim0, int dim1,
              sipuStream_t stream);
```

调用处 `hp_to_mx<sifmt::mxint8, HPType, 0>(...)` 只写了 3 个，后 2 个用默认值。`Rows` 甚至不是这个模板的参数——它是 host wrapper lambda 的内部模板参数，根据 `dim0` 运行时选择。

`sikernel.h:2282-2313` 的文档注释明确说明了每个模板参数的含义和默认值，`TNOCP=0` 对 mxint8 表示标准 OCP 量化（1 是 `sipu` 优化变体，仅对 mxfp6 有意义）。

## 1.6 完整数据流：tile 尺寸在哪里决定

```
Python: x=[7,96] BF16
  → pad → [32,128]                    ← 补到 Rows×32 对齐 + 128 列对齐
  → linear_to_tileformat              ← 按 tile=32行×16列 重排
  → flat [32,128] buffer              ← shape 丢失了 tile 结构，但字节顺序保留了

C++: input.size(0)=32, input.size(1)=128
  → host wrapper: dim0=32 > 16 → Rows=32          ← 运行时选择
  → HpToMxTensorMapGeometry:
      input_tile_dim0 = 1024/(32*2) = 16            ← 编译期推导
      output_tile_dim0 = 2*16 = 32                  ← 编译期推导
  → siTensorMapEncodeTiled:
      global=[128, 32, 1], box=[32, 32, 1]          ← 编码 tile 语义
  → hp_to_mx_kernel launch:
      DTE 按 tensor map 从 flat buffer 读 tile
      tcvt_mxi8 转换每个 32×32 的 MXInt8 output tile
      按物理 supertile 顺序写入 packed output
```

关键洞察：**Python 的 `linear_to_tileformat` 和 C++ 的 tensor map 约定相同 tile 尺寸**。shape metadata 在 reshape 回 2D 时确实丢失了，但 buffer 中元素的物理顺序已经按 tile-major 排好了。硬件的 DTE 通过 tensor map 的 `input_box_elements` 知道如何切分，不需要从 shape metadata 恢复。

## 1.7 为什么 Python 端的 tile 列数是 16 而不是 32

这是两个不同阶段的"tile"，尺寸不同但不矛盾：

| 阶段 | tile 含义 | BF16+Rows=32 的尺寸 |
| --- | --- | --- |
| Python `linear_to_tileformat` | **BF16 input staging tile** | 32行 × 16列 × 2B = 1024B |
| SiKernel MX output tile | **MXInt8 output tile** | 32行 × 32列 × 1B = 1024B data + 64B header |

一个 MXInt8 output tile（32×32 elements）消耗**两个**相邻的 BF16 input staging tile（每个 32×16）。所以 Python 按 16 列切，SiKernel 按 32 列切，是同一个物理数据在不同表示阶段的不同分块粒度。

`input_lmul=2`（BF16→MXInt8 的 traits）正体现了这个 2:1 关系：host wrapper 的 `input_box_dim0 = output_traits::input_lmul = 2`，表示每个 output tile 沿 dim0 方向消耗 2 个 input staging tile。

# 2. `fp8.py` 与 `fp8_linear.py` 代码讲解

## 2.1 两个文件各自的职责

**`fp8.py`** 是 `SIPU` 平台的 FP8 量化适配层，向上游 vLLM 注册 `fp8` 量化配置并覆写 linear/MoE 的量化方法。它的核心类有 4 个：

| 类 | 继承自 | 职责 |
| --- | --- | --- |
| `SIPUFp8Config` | `Fp8Config` | 注册名为 `fp8` 的量化配置，覆写 `get_quant_method` 分派规则 |
| `SIPUFp8LinearMethod` | `Fp8LinearMethod` | dense Linear 的 W8A8 block-quant 路径（offline checkpoint + dynamic activation） |
| `SIPUFp8MoEMethod` | `Fp8MoEMethod` | RoutedExperts MoE 的 offline FP8 checkpoint 路径 |
| `SIPUFp8PerTensorOnlineMoEMethod` | `Fp8PerTensorOnlineMoEMethod` | RoutedExperts MoE 的 online（动态）per-tensor FP8 路径 |

**`fp8_linear.py`** 向上游 vLLM 的 FP8 linear kernel dispatch 表注册 `SIPU` 原生 kernel，供非 `SIPUFp8LinearMethod` 路径（如 compressed-tensors）使用：

| 类 | 继承自 | 用途 |
| --- | --- | --- |
| `SIPUNonBlockFP8Kernel` | `CutlassFP8ScaledMMLinearKernel` | per-tensor/per-channel 非 block FP8，dispatch 到 SiInfer 的 `cutlass_scaled_mm` |
| `SIPUBlockFP8Kernel` | `Fp8BlockScaledMMLinearKernel` | 128×128 block FP8，使用 sideepgemm native kernel |

两者在模块加载时注册到上游 dispatch 表：
```python
_POSSIBLE_FP8_KERNELS[PlatformEnum.OOT] = [SIPUNonBlockFP8Kernel]
_POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.OOT] = [SIPUBlockFP8Kernel]
```

## 2.2 `SIPUFp8Config.get_quant_method` 分派规则

```python
def get_quant_method(layer, prefix):
    if is_layer_skipped(...):
        return super().get_quant_method(...)     # 走上游默认
    if isinstance(layer, LinearBase):
        return SIPUFp8LinearMethod(self)  # dense Linear
    if isinstance(layer, RoutedExperts):
        if not self.is_checkpoint_fp8_serialized:
            return SIPUFp8PerTensorOnlineMoEMethod(layer=layer)  # online MoE
        return SIPUFp8MoEMethod(self, layer)                     # offline MoE
    return super().get_quant_method(...)         # 其他走上游
```

分派逻辑一图流：

```text
LinearBase
  → SIPUFp8LinearMethod（always，无论 checkpoint 是否已量化）

RoutedExperts
  ├─ is_checkpoint_fp8_serialized = False → SIPUFp8PerTensorOnlineMoEMethod（online）
  └─ is_checkpoint_fp8_serialized = True  → SIPUFp8MoEMethod（offline）
```

## 2.3 `SIPUFp8LinearMethod`：dense Linear 的数据流

**权重是离线量化的（offline）：** checkpoint 已经是 FP8 格式，`create_weights` 创建 `torch.float8_e4m3fn` 权重参数和 `BlockQuantScaleParameter`（128×128 block scale）。`process_weights_after_loading` 调用 `process_fp8_weight_block_strategy` 对权重做布局调整（如 packing）。

**激活是在线量化的（dynamic）：** 每次 `apply()` 调用时：

```python
def apply(self, layer, x, bias=None):
    # 1. 动态量化 activation → FP8 per-token-group（group_size=128）
    x_fp8, x_scale = per_token_group_quant_fp8.siinfer(
        x, self.weight_block_size[1], dtype=torch.float8_e4m3fn)

    # 2. 转换权重为 grouped-MM layout（首次调用时缓存）
    _prepare_grouped_mm_layout(layer)

    # 3. 调用 torch._scaled_grouped_mm（torch_sipu 提供）
    output = torch._scaled_grouped_mm(x_fp8, w, x_scale, w_scale, out_dtype=torch.bfloat16)

    if bias is not None:
        output += bias
    return output
```

关键细节：
- `per_token_group_quant_fp8.siinfer` 是 SiInfer 提供的动态量化 kernel，group_size = block_size[1] = 128
- `_prepare_grouped_mm_layout` 把权重从 `[N, K]` 转成 `[K, N]`（contiguous），并在 `sipu` 上通过 CPU 中转完成 transpose
- `torch._scaled_grouped_mm` 由 `torch_sipu` 提供，是 block-quantized 的 scaled grouped matrix multiply
- **Per-tensor FP8 linear 未实现**：只支持 block-quant（128×128），per-tensor 直接 `raise NotImplementedError`

## 2.4 `SIPUFp8MoEMethod`：offline MoE

用于 `is_checkpoint_fp8_serialized=True` 的场景，即 checkpoint 已经是 FP8 格式（权重为 `torch.float8_e4m3fn`，带有 scale）。

`__init__` 选择 `DeepGemmExperts` 作为默认 backend，`_setup_kernel` 在权重加载后选择 `SIPU` 原生 backend：

```python
def _setup_kernel(self, layer, w13, w2, w13_scale, w2_scale, ...):
    sipu_backend = select_sipu_fp8_moe_backend(self.moe_quant_config, self.moe)
    if sipu_backend == SIPUFp8MoEBackend.TORCH:
        # TORCH backend 需要 BF16 权重，所以先反量化
        w13 = _dequant_fp8_moe_weight(w13, w13_scale, ...)
        w2 = _dequant_fp8_moe_weight(w2, w2_scale, ...)
    self.moe_kernel = make_sipu_fp8_moe_kernel(...)
```

`apply` 的分支：
- **非 TORCH backend**：调用 `super().apply()`（上游 `Fp8MoEMethod.apply`），使用 DeepGEMM 等 native kernel，权重保持 FP8
- **TORCH backend**：调用 `self.moe_kernel.apply()`（`SIPUTorchMoEKernel`），权重已反量化为 BF16，逐 expert 做 BF16 GEMM + 动态 FP8 量化

反量化函数 `_dequant_fp8_block_weight` 在 CPU 上逐 expert 执行，是一个纯 Python 循环，主要用于小模型或 CModel smoke test。

## 2.5 `SIPUFp8PerTensorOnlineMoEMethod`：online MoE

用于 `is_checkpoint_fp8_serialized=False` 的场景，即 checkpoint 是 BF16/FP16 原始精度。

它继承上游 `Fp8PerTensorOnlineMoEMethod`，但 `__init__` 覆写了 `moe_kernel` 为 `SIPUTorchMoEKernel()`，`process_weights_after_loading` 保留原始 dtype（不做任何量化），`get_fused_moe_quant_config` 返回 `None`。

`apply` 直接使用 `SIPUTorchMoEKernel`，在运行时对 BF16 权重和 activation 做动态量化。这是一个纯 torch 的 fallback 路径，不使用 DeepGEMM 或其他高效 kernel。

## 2.6 `fp8_linear.py`：向 upstream dispatch 表注册 native kernel

`SIPUNonBlockFP8Kernel` 继承 `CutlassFP8ScaledMMLinearKernel`，用于 per-tensor/per-channel FP8。它没有覆写 `apply_weights`，复用上游 cutlass 路径——在 `SIPU` 平台上 `cutlass_scaled_mm` 会被 dispatch 到 SiInfer。

`SIPUBlockFP8Kernel` 继承 `Fp8BlockScaledMMLinearKernel`，用于 128×128 block FP8。它：

1. **`can_implement`**：额外检查 activation group_shape 必须是 `(1, 128)`、weight group_shape 必须是 `(128, 128)`、dtype 必须是 BF16
2. **`process_weights_after_loading`**：调用 `pack_sideepgemm_block_fp8_weight` 做 native packing；如果 scale 是 `float8_e8m0fnu`（MX 规范格式），先转成 `float32`
3. **`apply_weights`**：调用 `apply_w8a8_block_fp8_linear.torch`（SiInfer native kernel），传入 block_size、weight、weight_scale，由 `SIPU` 原生 GEMM 完成计算

这个 kernel 被上游 compressed-tensors 路径使用（当压缩格式为 block FP8 时），而不是被 `SIPUFp8LinearMethod` 使用（后者走 `torch._scaled_grouped_mm`）。

## 2.7 `fp8.py` 中的其他工具

### `scaled_dequantize` monkey patch

`_install_scaled_dequantize_patch()` 在模块加载时用 `_scaled_dequantize_sipu` 替换上游的 `quant_utils.scaled_dequantize`。新函数在 `sipu` tensor 上先把数据搬到 CPU 做反量化，再搬回 `sipu`。这是一个兼容性 workaround：上游的 `scaled_dequantize` 可能不支持 `sipu` tensor，所以用 CPU fallback 保证不报错。

### `_dequant_fp8_moe_weight` / `_dequant_fp8_block_weight`

MoE 权重反量化工具。`_dequant_fp8_block_weight` 在 CPU 上逐 expert、逐 block 循环反量化，用于 TORCH backend 需要 BF16 权重的场景。`_dequant_fp8_tensor_weight` 处理 per-tensor scale 的情况，支持 0D/1D/2D scale tensor 的各种 broadcasting。

### `_prepare_grouped_mm_layout`

首次 `apply()` 调用时把权重从 `[N, K]` transpose 为 `[K, N]`（contiguous），在 `sipu` 上通过 CPU 中转完成。结果缓存在 layer 上（通过 `_GROUPED_MM_LAYOUT_MARKER` 标记），后续调用直接跳过。这是 `torch._scaled_grouped_mm` 要求的输入布局。

## 2.8 Linear/MoE 的量化是在线还是离线？

答案是**两者都有，取决于 layer 类型和 checkpoint 状态**：

| 场景 | 权重 | 激活 | 方法类 |
| --- | --- | --- | --- |
| Dense Linear（所有情况） | **离线**：checkpoint 已是 FP8，加载后做 block layout 处理 | **在线**：每次 forward 动态 `per_token_group_quant_fp8` | `SIPUFp8LinearMethod` |
| MoE + FP8 checkpoint | **离线**：checkpoint 已是 FP8，加载后选择 backend（DeepGEMM 保持 FP8，TORCH 反量化为 BF16） | **在线**：kernel 内部动态量化 activation | `SIPUFp8MoEMethod` |
| MoE + BF16/FP16 checkpoint | **在线**：权重保持原始 dtype，运行时由 kernel 动态量化 | **在线**：运行时动态量化 | `SIPUFp8PerTensorOnlineMoEMethod` |

总结：
- **Linear 总是 W8A8**：权重 FP8 离线，activation FP8 在线
- **MoE 看 checkpoint**：FP8 checkpoint → 离线权重 + 在线 activation；BF16 checkpoint → 全在线
- **没有纯离线路径**：即使权重来自 FP8 checkpoint，activation 也始终在 forward 时动态量化

## 2.9 单元测试覆盖

### `SIPUFp8LinearMethod`（dense Linear）

有测试，位于 `tests/kernels/quantization/test_fp8_linear.py::test_fp8_linear_method`（第 216-312 行）。

该测试：
1. 创建 `Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[128, 128])`
2. 手动构造 FP8 权重（`torch.float8_e4m3fn`）和 128×128 block scale
3. 调用 `create_weights` → `process_weights_after_loading` → `apply`
4. 与 CPU 参考实现（反量化 + BF16 GEMM）做 `atol=1e-1, rtol=1e-1` 的对比
5. 验证 `per_token_group_quant_fp8.siinfer` 被调用且 group_size=128

### `SIPUBlockFP8Kernel`（fp8_linear.py 的 block kernel）

有测试，位于 `tests/kernels/quantization/test_fp8_linear.py::test_sipu_block_fp8_linear_operator_stack`（第 89-174 行）。

该测试：
1. 创建 `SIPUBlockFP8Kernel`，验证 `is_supported` 和 `can_implement`
2. 测试 `process_weights_after_loading`（包括 `float8_e8m0fnu` → `float32` scale 转换）
3. 调用 `apply_weights` 与 CPU 参考对比（`atol=1e-1, rtol=1e-1`）

### `SIPUFp8MoEMethod`（offline FP8 MoE，`Fp8Config` 路径）

**没有专门的单元测试。** 搜索整个 `tests/` 目录未找到对 `SIPUFp8MoEMethod` 的直接测试。MoE 测试文件 `tests/kernels/moe/test_sipu_w8a8_fp8_moe.py` 测试的是 compressed-tensors 路径（`SIPUCompressedTensorsW8A8Fp8MoEMethod`），不是 `Fp8Config` 路径。

两者虽然最终可能调用类似的底层 kernel，但分派路径、权重创建方式和 scale 处理不同，compressed-tensors 测试不能替代 `SIPUFp8MoEMethod` 的测试。

### `SIPUFp8PerTensorOnlineMoEMethod`（online MoE）

**没有专门的单元测试。** 该类是纯 torch fallback 路径，没有被任何现有测试覆盖。

### `SIPUNonBlockFP8Kernel`（fp8_linear.py 的非 block kernel）

**没有专门的单元测试。** 它继承上游 `CutlassFP8ScaledMMLinearKernel`，在 `SIPU` 上 dispatch 到 SiInfer 的 `cutlass_scaled_mm`，但没有针对 `SIPU` 平台的单独测试。

### 测试覆盖总结

| 方法类 | 有单元测试 | 测试文件 |
| --- | --- | --- |
| `SIPUFp8LinearMethod` | ✅ | `tests/kernels/quantization/test_fp8_linear.py::test_fp8_linear_method` |
| `SIPUBlockFP8Kernel` | ✅ | `tests/kernels/quantization/test_fp8_linear.py::test_sipu_block_fp8_linear_operator_stack` |
| `SIPUFp8MoEMethod` | ❌ | — |
| `SIPUFp8PerTensorOnlineMoEMethod` | ❌ | — |
| `SIPUNonBlockFP8Kernel` | ❌ | — |

此外，MoE 测试文件 `tests/kernels/moe/test_sipu_w8a8_fp8_moe.py` 测试的是 compressed-tensors 路径（`SIPUCompressedTensorsW8A8Fp8MoEMethod`），不是 `Fp8Config` 路径的 `SIPUFp8MoEMethod`。两者虽然底层 kernel 可能重叠，但上层逻辑不同，测试覆盖不能互相替代。
