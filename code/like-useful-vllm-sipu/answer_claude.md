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
