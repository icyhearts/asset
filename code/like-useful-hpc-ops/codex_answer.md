# 1. torch::kBFloat16 — 类型、定义位置与命名空间解析链

## 1.1 类型

`torch::kBFloat16` 的类型是 `c10::ScalarType` 的 **constexpr 常量**，但 `ScalarType` 本身是一个 C++ **enum class**。

这不是 C 风格的裸 enum，而是 `enum class ScalarType : int8_t`，底层类型为 `int8_t`。

## 1.2 定义位置

### 1.2.1 enum class ScalarType 定义

`{conda_env}/lib/python3.12/site-packages/torch/include/torch/headeronly/core/ScalarType.h:258-264`

```cpp
enum class ScalarType : int8_t {
#define DEFINE_ST_ENUM_VAL_(_1, n) n,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ST_ENUM_VAL_)
#undef DEFINE_ENUM_ST_ENUM_VAL_
      Undefined,
  NumOptions
};
```

枚举值由宏 `AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS` 展开生成（同文件 lines 103-149）。`BFloat16` 位于第 15 个位置：

```cpp
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_) \
  _(uint8_t, Byte) /* 0 */                               \
  // ...                                                 \
  _(c10::BFloat16, BFloat16) /* 15 */                    \
  // ... (共约 45 个枚举值)
```

所以 `ScalarType::BFloat16` 的数值为 `static_cast<int8_t>(15)`。

### 1.2.2 constexpr 常量 kBFloat16 定义

`{conda_env}/lib/python3.12/site-packages/torch/include/c10/core/ScalarType.h:38-42`

```cpp
#define DEFINE_CONSTANT(_, name) \
  constexpr ScalarType k##name = ScalarType::name;

// NOLINTNEXTLINE(clang-diagnostic-unused-const-variable)
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT
```

这个宏在 `namespace c10` 内为每个枚举值生成一个 `constexpr ScalarType` 常量。对 `BFloat16` 生成的代码相当于：

```cpp
constexpr ScalarType kBFloat16 = ScalarType::BFloat16;
```

### 1.2.3 辅助包装头文件

以下两个头文件是薄包装，仅做 include 转发：

- `{conda_env}/lib/python3.12/site-packages/torch/include/ATen/core/ScalarType.h:2` — `#include <c10/core/ScalarType.h>`
- `{conda_env}/lib/python3.12/site-packages/torch/include/ATen/ScalarType.h:5` — `#include <c10/core/ScalarType.h>`

它们仅为向后兼容而存在。

## 1.3 命名空间解析链：c10 → at → torch

`torch::kBFloat16` 并非在 `torch` 命名空间中直接定义，而是通过两层 `using namespace` 逐级传播：

### 1.3.1 第 1 层：c10 → at

`{conda_env}/lib/python3.12/site-packages/torch/include/torch/headeronly/macros/Macros.h:156-165`

```cpp
// Since C10 is the core library for caffe2 (and aten), we will simply reroute
// all abstractions defined in c10 to be available in caffe2 as well.
// This is only for backwards compatibility. Please use the symbols from the
// c10 namespace where possible.
namespace caffe2 {
using namespace c10;
}
namespace at {
using namespace c10;
}
```

`namespace at { using namespace c10; }` 使得 `c10::kBFloat16` 同时可用作 `at::kBFloat16`。

### 1.3.2 第 2 层：at → torch

`{conda_env}/lib/python3.12/site-packages/torch/include/torch/csrc/api/include/torch/types.h:13-37`

```cpp
namespace torch {
// NOTE [ Exposing declarations in `at::` to `torch::` ]
//
// The following line `using namespace at;` is responsible for exposing all
// declarations in `at::` namespace to `torch::` namespace.
using namespace at; // NOLINT

// Fixed width dtypes.
constexpr auto kFloat16 = at::kHalf;
constexpr auto kFloat32 = at::kFloat;
constexpr auto kFloat64 = at::kDouble;
} // namespace torch
```

`using namespace at;` 将 `at` 命名空间中所有符号引入 `torch` 命名空间，因此 `at::kBFloat16` 自动变为 `torch::kBFloat16`。

### 1.3.3 解析链图示

```
c10::kBFloat16                    constexpr常量，定义于 c10/core/ScalarType.h:39
    │
    │  namespace at { using namespace c10; }
    │  torch/headeronly/macros/Macros.h:163-165
    ▼
at::kBFloat16                     自动可用（using-directive 引入）
    │
    │  namespace torch { using namespace at; }
    │  torch/csrc/api/include/torch/types.h:37
    ▼
torch::kBFloat16                  自动可用（using-directive 引入）
```

所以 `src/group_gemm/entry.cc:43` 中使用 `torch::kBFloat16` 时（通过 `#include <torch/all.h>` 引入），实际解析路径为：

```
torch::kBFloat16 → at::kBFloat16 → c10::kBFloat16 = constexpr ScalarType(15)
```

## 1.4 总结

| 问题 | 答案 |
|------|------|
| 是 enum class 吗？ | 是 `constexpr ScalarType` 常量，`ScalarType` 本身是 `enum class ScalarType : int8_t` |
| 真正定义在哪里？ | `c10/core/ScalarType.h:39`（常量），`torch/headeronly/core/ScalarType.h:258`（enum class） |
| 如何在 `torch::` 中可用？ | 通过 `at` → `c10` 两层 `using namespace` 链式传播 |
| 数值是多少？ | `static_cast<int8_t>(15)` |

---

# 2. `hpc.group_gemm_pertensor_fp8`：从 Python 到 Hopper CUDA kernel

本文分析的代码版本是 hpc-ops commit `a32acf4ca3798896acc939bfc955c3a7d374fdb8`。conda 环境中已安装的 wheel 报告版本为 `0.0.1.dev0+ga32acf4`，与当前源码 commit 一致，因此下面的源码调用链与 `/share/users/like/package/hpc-ops/temp/run.log` 对应。

## 2.1 直接回答：CUDA kernel 在哪里

`hpc.group_gemm_pertensor_fp8` 并不调用 cuBLAS，也不是直接启动 CUTLASS 提供的完整 GEMM kernel。hpc-ops 自己定义了持久化、warp-specialized CUDA kernel，并在 kernel 内使用 CuTe/CUTLASS 的 TMA、WGMMA、layout 和 barrier 原语。

主 GEMM kernel 的定义是：

- `src/group_gemm/kernels.cuh:215-530`，函数 `hpc::group_gemm::kernels::group_gemm_fp8_kernel`。

主 kernel 的 host 侧选择和 launch 位于：

- `src/group_gemm/group_gemm_pertensor_fp8.cu:25-327`，函数模板 `hpc::group_gemm::launch_group_gemm_fp8`；
- 其中真正选取 `group_gemm_fp8_kernel` specialization 并启动的代码位于同一函数 `:217-325`。

一次默认调用实际会启动两个 CUDA kernel，而不是只有一个：

1. `src/group_gemm/kernels.cuh:63-213`，CUDA kernel `hpc::group_gemm::kernels::update_grouped_tma`：为每个 group 更新 X/Y TMA descriptor，并计算 tile 前缀和。
2. `src/group_gemm/kernels.cuh:215-530`，CUDA kernel `hpc::group_gemm::kernels::group_gemm_fp8_kernel`：执行 FP8 group GEMM、per-tensor scaling 和 BF16 写回。

`src/group_gemm/group_gemm_pertensor_fp8.cu` 虽然文件名带 `pertensor`，但最终 `__global__` 函数放在公共头文件 `src/group_gemm/kernels.cuh` 中，由不同 config 实例化。

## 2.2 本次测试计算的数学含义

测试输入来自 `tests/test_group_gemm_pertensor_like.py:46-73`（函数 `test_group_gemm_pertensor_fp8`）：

```python
x.shape      == [576, 7168]       # FP8 E4M3
weight.shape == [8, 4096, 7168]   # FP8 E4M3
seqlens      == [16, 32, 48, 64, 80, 96, 112, 128]
cu_seqlens   == [0, 16, 48, 96, 160, 240, 336, 448, 576]
y_scale      == [0.25] * 8
output.shape == [576, 4096]       # BF16
```

对第 `g` 个 group，kernel 实际计算：

```text
start = cu_seqlens[g]
end   = cu_seqlens[g + 1]

Y[start:end, :]
  = BF16(y_scale[g] * FP32_accumulate(X[start:end, :] @ weight[g, :, :].T))
```

测试参考实现位于 `tests/test_group_gemm_pertensor_like.py:20-43`（函数 `naive_group_gemm_pertensor_fp8`），使用 `torch._scaled_mm`，其中 `scale_a=0.5`、`scale_b=0.5`；hpc-ops 接收预先相乘的 `scale_hpc=0.25`，见同一文件 `:57-66`（函数 `test_group_gemm_pertensor_fp8`）。

传入的 `actual_m=256` 在当前测试函数体中没有参与 shape 或 dispatch；真正的总 M 是 `sum(seqlens)=576`，平均 group 长度为：

```text
num_seq_per_group_avg = int(576 / 8) = 72
```

其计算位置是 `tests/test_group_gemm_pertensor_like.py:49-66`（函数 `test_group_gemm_pertensor_fp8`）。

## 2.3 Python 到 CUDA kernel 的完整调用栈

### 2.3.1 Python package 加载和函数导出

`hpc/__init__.py:43-52`（模块初始化代码）找到 wheel 内的 `_C.abi3.so`，通过 `torch.ops.load_library` 装载共享库。随后 `hpc/__init__.py:12-40`（函数 `_discover_modules`、`_export_functions`）导入 `hpc/group_gemm.py`，并把其中的公开函数放进 `hpc` 顶层命名空间。

因此测试中的：

```python
hpc.group_gemm_pertensor_fp8(...)
```

实际指向 `hpc/group_gemm.py:51-107`（函数 `group_gemm_pertensor_fp8`）。该包装函数不做计算，只把参数转发给：

```python
torch.ops.hpc.group_gemm_pertensor_fp8(...)
```

### 2.3.2 PyTorch dispatcher 注册

算子 schema 与 CUDA implementation 的绑定位于 `src/group_gemm/entry.cc:226-250`（注册块 `TORCH_LIBRARY_FRAGMENT(hpc, m)`）。关键代码是：

```cpp
m.def("group_gemm_pertensor_fp8(...) -> (Tensor)");
m.impl("group_gemm_pertensor_fp8", torch::kCUDA,
       &hpc::group_gemm::group_gemm_fp8_entry);
```

这里有一个重要细节：源码中没有单独的 `group_gemm_pertensor_fp8_entry`。以下两个 Python op 都绑定到同一个 C++ entry：

```text
hpc::group_gemm_fp8             ┐
                                ├-> hpc::group_gemm::group_gemm_fp8_entry
hpc::group_gemm_pertensor_fp8   ┘
```

对应代码是 `src/group_gemm/entry.cc:226-237`（注册块 `TORCH_LIBRARY_FRAGMENT(hpc, m)`）。

### 2.3.3 C++ entry

`src/group_gemm/entry.cc:16-89`（函数 `hpc::group_gemm::group_gemm_fp8_entry`）完成以下工作：

1. 取得 `x` 所在设备的当前 CUDA stream，见 `:23`；
2. 检查 CUDA device、FP8 E4M3 dtype、INT32 sequence metadata、FP32 scale、contiguous 和 shape，见 `:24-37`；
3. 从 tensor shape 得到 `m=576`、`n=4096`、`k=7168`、`num_group=8`，见 `:39-42`；
4. `output=None` 时分配 `[m,n]` BF16 输出，见 `:44-50`；
5. `tma_desc=None` 时分配 `num_group*2` 个 128-byte descriptor 的 backing tensor，并设置 `update_tma=true`，见 `:52-59`；
6. 分配 `tiles[num_group]` 和 `cu_tiles[num_group+1]`，见 `:70-82`；
7. 调用 `group_gemm_fp8_async`，见 `:84-86`；
8. 立即返回 output tensor，见 `:88`。kernel 仍按 PyTorch 当前 stream 异步执行，后续同 stream 算子会维持依赖顺序。

### 2.3.4 Host 侧 config 分发

`src/group_gemm/group_gemm_pertensor_fp8.cu:329-420`（函数 `hpc::group_gemm::group_gemm_fp8_async`）根据平均 group 长度选择编译期模板参数，然后调用 `launch_group_gemm_fp8`。

本次 `num_seq_per_group_avg=72`，满足 `64 < 72 <= 96`，因此命中 `:391-397`：

```cpp
constexpr int kTileM = 48;
constexpr int kStage = 8;
launch_group_gemm_fp8<48, 128, 128, 8, 2, 1, 128, 128, 64>(...);
```

其余固定参数定义于同一函数 `:335-343`：

```text
TileM=48, TileN=128, TileK=128
Stages=8
WarpgroupM=2, WarpgroupN=1
SwizzleX=128, SwizzleW=128, SwizzleY=64
use_pdl=true
```

### 2.3.5 两次 CUDA launch

`src/group_gemm/group_gemm_pertensor_fp8.cu:145-215`（函数模板 `launch_group_gemm_fp8`）在 `update_tma=true` 时先启动：

```text
update_grouped_tma<..., TileM=48,
                   kAssignTask=false, kUsePDL=true>
grid  = num_group + 1 = 9
block = 32
```

然后 `src/group_gemm/group_gemm_pertensor_fp8.cu:217-282`（函数模板 `launch_group_gemm_fp8`）选择并启动主 kernel。由于：

```text
task_map_ptr == nullptr
k=7168 > 1024
n=4096 > 1024
```

命中 `kTaskLoopPolicy=2` 分支 `:269-281`：

```text
group_gemm_fp8_kernel<Config48x128x128x8, ..., 2, true>
grid  = num_sm
block = 384
```

完整运行时调用链可写成：

```text
tests/test_group_gemm_pertensor_like.py:65
  test_group_gemm_pertensor_fp8
    |
    v
hpc/group_gemm.py:51-107
  group_gemm_pertensor_fp8
    |
    v
torch.ops.hpc.group_gemm_pertensor_fp8
    |
    v
src/group_gemm/entry.cc:226-237
  TORCH_LIBRARY_FRAGMENT -> CUDA dispatcher
    |
    v
src/group_gemm/entry.cc:16-89
  hpc::group_gemm::group_gemm_fp8_entry
    |
    v
src/group_gemm/group_gemm_pertensor_fp8.cu:329-420
  hpc::group_gemm::group_gemm_fp8_async
    |
    v
src/group_gemm/group_gemm_pertensor_fp8.cu:25-327
  hpc::group_gemm::launch_group_gemm_fp8<48,128,128,8,...>
    |
    +--> src/group_gemm/kernels.cuh:63-213
    |      update_grouped_tma<...,48,...,false,true>
    |
    `--> src/group_gemm/kernels.cuh:215-530
           group_gemm_fp8_kernel<Config48x128x128x8,...,policy2,true>
             |
             v
           CuTe cute::gemm
             |
             v
           wgmma.mma_async.sync.aligned.m64n48k32.f32.e4m3.e4m3
```

## 2.4 为什么日志中的 `tma_x` 是 `(_48,_128)`

`src/group_gemm/group_gemm_pertensor_fp8.cu:37-45`（函数模板 `launch_group_gemm_fp8`）建立 X/W/Y 的 CuTe global tensor，并构造 `GroupGEMMFp8Config`。`src/group_gemm/config.h:63-106`（类型 `GroupGEMMFp8Config`，成员函数 `get_tma`）据 `TileM/TileN/TileK/Stages` 生成 shared layout 和 TMA copy：

```cpp
SLayoutX: [TileM=48,  TileK=128, Stage=8]
SLayoutW: [TileN=128, TileK=128, Stage=8]

tma_x = make_tma_copy(SM90_TMA_LOAD{}, x, X shared tile);
tma_w = make_tma_copy(SM90_TMA_LOAD{}, w, W shared tile);
tma_y = make_tma_copy(SM90_TMA_STORE{}, y, output copy box);
```

`src/group_gemm/group_gemm_pertensor_fp8.cu:123-142`（函数模板 `launch_group_gemm_fp8`）在 `LIKE_DEBUG` 下打印 `tma_x`。日志 `temp/run.log:13-21` 显示：

```text
Tiler_MN:       (_48,_128)
ThrID:          _1:_0
ValLayoutSrc:   (_1,_6144):(_0,_1)
ValueType:      8b
```

解释如下：

- `48 x 128` 正是本次 X 的一个 `[TileM,TileK]` tile；
- `6144 = 48*128`，即一次 X TMA load 覆盖 6144 个 FP8 元素；
- `ValueType: 8b` 对应 E4M3 每元素 1 byte；
- `ThrID: _1:_0` 表示 TMA copy atom 由一个 elected thread 发出，不代表只有一个线程参与后续 WGMMA。

因此该日志是 `TileM=48` 分发已经生效的直接证据。

## 2.5 辅助 kernel：`update_grouped_tma`

### 2.5.1 为什么 group GEMM 需要动态 descriptor

W 的每个 group 都是固定 `[N,K]`，可以用第三维 group coordinate 从统一 W descriptor 访问。X/Y 则是 ragged 分段：每个 group 的 base address 和 M 长度不同，所以需要为每个 group 生成独立 X/Y descriptor。

Host 侧初始 tensor layout 位于 `src/group_gemm/group_gemm_pertensor_fp8.cu:37-42`（函数模板 `launch_group_gemm_fp8`）：

```cpp
X: shape(m,k),           stride(k,1)
W: shape(n,k,num_group), stride(k,1,n*k)
Y: shape(n,m),           stride(1,n)
```

Y 在 CuTe 中逻辑表示为 `[N,M]`，但 `stride(1,n)` 对应 PyTorch `[M,N]` row-major 的同一片物理内存。

### 2.5.2 每个 group 的 descriptor 更新

`src/group_gemm/kernels.cuh:172-207`（CUDA kernel `update_grouped_tma`）对 `blockIdx.x=g<num_group` 执行：

```cpp
x_group_base = x_ptr + cu_seqlens[g] * k;
y_group_base = y_ptr + cu_seqlens[g] * n;

gX: shape(seqlens[g], k), stride(k,1)
gY: shape(n, seqlens[g]), stride(1,n)
```

线程 0 更新 X descriptor，线程 1 更新 Y descriptor。`src/utils/tma.cuh:36-57`（函数模板 `hpc::update_tma_gtensor`）调用 CuTe helper 计算 TMA shape/stride，并使用 tensormap replace 指令修改 descriptor 的 base address 和 shape。

修改后的 shared descriptor 通过 `tma_descriptor_cp_fence_release` 写入 global `td_xy[g*2 + {0,1}]`。其底层 release copy 位于 `3rd/cutlass/include/cute/arch/copy_sm90_desc.hpp:419-436`（函数 `cute::tma_descriptor_cp_fence_release`）。

### 2.5.3 `tiles` 和 `cu_tiles`

额外的 `blockIdx.x == num_group` block 在 `src/group_gemm/kernels.cuh:143-170`（CUDA kernel `update_grouped_tma`）计算：

```text
tiles[g]    = ceil(seqlens[g] / TileM)
cu_tiles[g] = exclusive_prefix_sum(tiles)
```

本次 `TileM=48`：

| group | seqlen | `tiles[g]=ceil(seqlen/48)` | `cu_tiles[g]` |
|---:|---:|---:|---:|
| 0 | 16  | 1 | 0  |
| 1 | 32  | 1 | 1  |
| 2 | 48  | 1 | 2  |
| 3 | 64  | 2 | 3  |
| 4 | 80  | 2 | 5  |
| 5 | 96  | 2 | 7  |
| 6 | 112 | 3 | 9  |
| 7 | 128 | 3 | 12 |

最终 `cu_tiles[8]=15`，所以所有 group 合计有 15 个 M tile。

### 2.5.4 与主 kernel 的 PDL 依赖

`group_gemm_fp8_async` 在 `src/group_gemm/group_gemm_pertensor_fp8.cu:343`（函数 `group_gemm_fp8_async`）强制设置 `use_pdl=true`。两个 launch 都带 `cudaLaunchAttributeProgrammaticStreamSerialization`，见 `src/group_gemm/group_gemm_pertensor_fp8.cu:157-194` 和 `:226-281`（函数模板 `launch_group_gemm_fp8`）。

辅助 kernel 在 `src/group_gemm/kernels.cuh:75-77,210-212`（CUDA kernel `update_grouped_tma`）调用 `cudaGridDependencySynchronize` / `cudaTriggerProgrammaticLaunchCompletion`；主 kernel 在 `src/group_gemm/kernels.cuh:286-288,527-529`（CUDA kernel `group_gemm_fp8_kernel`）使用相同 PDL 协议。作用是主 kernel 在消费动态 descriptor/tile metadata 前等待 producer 发出 programmatic completion，而不要求 host 在两次 launch 之间同步整个 device。

## 2.6 主 kernel 的 CTA 结构与资源分工

主定义位于 `src/group_gemm/kernels.cuh:215-530`（CUDA kernel `group_gemm_fp8_kernel`），声明为：

```cpp
__global__ void __launch_bounds__(384, 1) group_gemm_fp8_kernel(...)
```

### 2.6.1 384 个线程如何分工

`src/group_gemm/kernels.cuh:324-400`（CUDA kernel `group_gemm_fp8_kernel`）按照 `idx < size(TiledMma{})` 分支：

```text
thread 0..255:   两个 math warpgroups，共 256 threads
thread 256..383: 一个 TMA load warpgroup，共 128 threads
```

依据如下：

- `src/group_gemm/config.h:98-100`（类型 `GroupGEMMFp8Config`）使用 `WarpgroupLayout=(2,1,1)`，把一个 128-thread WGMMA atom 沿 atom-M 方向复制两次；
- `3rd/cutlass/include/cute/atom/mma_traits_sm90_gmma_ext.hpp:9837-9855`（类型特化 `MMA_Traits<SM90_64x48x32_F32E4M3E4M3_SS_TN>`）定义 atom `ThrID=Layout<_128>`；
- 所以 `size(TiledMma{})=2*128=256`。

load warpgroup 调用 `warpgroup_reg_dealloc<24>()`，math warpgroups 调用 `warpgroup_reg_alloc<168>()`，见 `src/group_gemm/kernels.cuh:325-328,396-400`（CUDA kernel `group_gemm_fp8_kernel`）。这是 Hopper warpgroup register reconfiguration：降低 loader 的寄存器配额，把更多寄存器留给 FP32 accumulator。

### 2.6.2 shared memory 布局

`src/group_gemm/kernels.cuh:243-273`（CUDA kernel `group_gemm_fp8_kernel`）定义：

```text
readable[8]     8 个 TMA-complete mbarrier
writable[8]     8 个 consumer-release mbarrier
shm_a           [48,128,8] FP8
shm_b           [128,128,8] FP8
shm_c           [128,48] BF16
shm_tiles        policy-2 的 cu_tiles[9]
```

按源码 config 计算，host 传入的动态 shared memory 是：

```text
shm_a: 48*128*8*1       =  49,152 bytes
shm_b: 128*128*8*1      = 131,072 bytes
shm_c: 128*48*2         =  12,288 bytes
cu_tiles: (8+1)*4       =      36 bytes
----------------------------------------
dynamic shared total    = 192,548 bytes
```

Host 端计算位置是 `src/group_gemm/group_gemm_pertensor_fp8.cu:269-281`（函数模板 `launch_group_gemm_fp8`），基础 shared size 来自 `src/group_gemm/config.h:102-106`（类型 `GroupGEMMFp8Config`，成员函数 `get_shm_size`）。此数字不含源码中静态声明的 `readable/writable` barrier 数组。

### 2.6.3 persistent grid

`src/group_gemm/group_gemm_pertensor_fp8.cu:217-225`（函数模板 `launch_group_gemm_fp8`）设置：

```cpp
dim3 block(384);
dim3 grid(num_sm);
```

`num_sm` 由 `src/utils/utils.cc:15-27`（函数 `hpc::get_sm_count`）读取 device 0 的 `multiProcessorCount`。本次 H100 80GB HBM3 的值是 132，所以主 launch 是 132 个 persistent CTA；每个 CTA 完成一个 tile 后继续领取下一个，而不是每个输出 tile 启动一个 CTA。

## 2.7 `kTaskLoopPolicy=2` 如何把 group tiles 分给 CTA

主 kernel 在 `src/group_gemm/kernels.cuh:310-319`（CUDA kernel `group_gemm_fp8_kernel`）把 `cu_tiles[0..8]` 搬到 shared memory，并得到：

```text
total_m = cu_tiles[8] = 15
num_tile_n = ceil(4096 / 128) = 32
总输出 tasks = 15 * 32 = 480
```

每个 task 表示一个 group-local `[最多48行, 128列]` 输出 tile。policy 2 使用 `src/group_gemm/kernels.cuh:42-61`（函数 `get_next_tile_vert`）：

```cpp
itile_m_total = iblock % total_m;
itile_n       = iblock / total_m;
```

然后在 `cu_tiles` 中二分查找，得到 `igroup` 和该 group 内的 `itile_m`。每个 CTA 从 `iblock=blockIdx.x` 开始，并在每轮执行 `iblock += gridDim.x`，见 `src/group_gemm/kernels.cuh:416-445`（CUDA kernel `group_gemm_fp8_kernel` 的 math 分支）。

对 480 个 tasks 和 132 个 CTA：

```text
前 84 个 CTA 各处理 4 个 tasks
后 48 个 CTA 各处理 3 个 tasks
```

loader 和 math warpgroups 各自执行相同的 task 映射逻辑，因此对同一个 CTA，它们以同样顺序遍历 `(igroup,itile_m,itile_n)`，再通过 stage barrier 交换数据所有权。

## 2.8 TMA load warpgroup：global -> 8-stage shared pipeline

### 2.8.1 barrier 初始化

`src/group_gemm/kernels.cuh:277-281`（CUDA kernel `group_gemm_fp8_kernel`）初始化每个 stage：

```cpp
initialize_barrier(readable[stage], 1);
initialize_barrier(writable[stage], size(TiledMma{})); // 256
```

底层分别对应 PTX `mbarrier.init`，实现见 `3rd/cutlass/include/cute/arch/copy_sm90_desc.hpp:61-73`（函数 `cute::initialize_barrier`）。

- `readable[stage]`：一个 elected TMA thread 设置 transaction bytes；TMA A+B 完成后允许 math warpgroup 读取。
- `writable[stage]`：256 个 math threads 各 arrive 一次；全部完成当前 stage 的 WGMMA 读取后，允许 loader 覆盖。

### 2.8.2 只有一个线程发出 TMA

load warpgroup 内 `is_leader_in_load` 只选择其 warp 0 的一个 elected lane，见 `src/group_gemm/kernels.cuh:332-347`（CUDA kernel `group_gemm_fp8_kernel`）。其他 127 个 loader threads 主要用于形成独立 warpgroup 和寄存器资源分区，不重复发出 copy。

### 2.8.3 每个 K tile 的 copy

`src/group_gemm/kernels.cuh:372-392`（CUDA kernel `group_gemm_fp8_kernel`）对每个 `itile_k` 执行：

```cpp
wait_barrier(writable[stage], phase);

TMA X[group-local M tile, K tile] -> shm_a[stage]
TMA W[group, N tile, K tile]      -> shm_b[stage]

set_barrier_transaction_bytes(readable[stage], transaction_bytes);
```

本次每 stage 的 transaction bytes 为：

```text
(TileM + TileN) * TileK * sizeof(FP8)
=(48 + 128) * 128 * 1
=22,528 bytes
```

X 使用 `td_xy[igroup*2]` 中动态更新的 descriptor；W 使用 host 创建、以 `__grid_constant__` 参数传入的统一 descriptor，并通过 `igroup` 坐标选择 weight group。

`wait_barrier`、`set_barrier_transaction_bytes` 和 `arrive_barrier` 的 PTX wrapper 位于 `3rd/cutlass/include/cute/arch/copy_sm90_desc.hpp:75-126`（函数 `cute::set_barrier_transaction_bytes`、`cute::wait_barrier`、`cute::arrive_barrier`）。stage index 到 8 后回到 0，并翻转 phase bit，形成 8 级环形流水线。

本次 `K=7168`、`TileK=128`，因此每个输出 task 有：

```text
ntile_k = 7168 / 128 = 56
```

loader 会为该 task 连续生产 56 个 K tile。

## 2.9 Math warpgroups：shared -> WGMMA -> FP32 accumulator

### 2.9.1 为什么代码把 W partition 成 A、把 X partition 成 B

`src/group_gemm/kernels.cuh:402-411`（CUDA kernel `group_gemm_fp8_kernel`）执行：

```cpp
auto tBs4r = thr_mma.partition_A(sB); // W: [N,K]
auto tAs4r = thr_mma.partition_B(sA); // X: [M,K]
```

kernel 内部计算的是：

```text
C_internal[N,M] = W[N,K] * X[M,K]^T
```

这与用户看到的：

```text
Y[M,N] = X[M,K] * W[N,K]^T
```

是同一组标量结果，只是内部把输出看成转置的 `[N,M]` 逻辑 tensor。由于 Y 使用 `shape(n,m), stride(1,n)`，内部 `C_internal(n_idx,m_idx)` 的物理地址正好是 PyTorch `Y[m_idx,n_idx]`。

### 2.9.2 本次使用的 WGMMA atom

`src/group_gemm/config.h:42-60`（函数模板 `hpc::group_gemm::mma_selector`）对 `TileM=48` 选择：

```cpp
cute::SM90_64x48x32_F32E4M3E4M3_SS_TN<>
```

其 traits 位于 `3rd/cutlass/include/cute/atom/mma_traits_sm90_gmma_ext.hpp:9831-9855`（类型特化 `MMA_Traits<SM90_64x48x32_F32E4M3E4M3_SS_TN>`）：

```text
atom shape = 64 x 48 x 32
input A/B  = FP8 E4M3, shared-memory descriptors
accumulator = FP32
threads per atom = 128
```

两个 math warpgroups 沿 atom 的 64 维并排，因此一个 CTA 的 WGMMA tile 是内部 `[N,M]=[128,48]`。

最终 PTX 位于 `3rd/cutlass/include/cute/arch/mma_sm90_gmma_ext.hpp:28810-28854`（函数 `SM90::GMMA::MMA_64x48x32_F32E4M3E4M3_SS_TN::fma`）：

```ptx
wgmma.mma_async.sync.aligned.m64n48k32.f32.e4m3.e4m3
```

### 2.9.3 一个 `TileK=128` 如何执行

math 分支位于 `src/group_gemm/kernels.cuh:447-483`（CUDA kernel `group_gemm_fp8_kernel`）：

1. `wait_barrier(readable[stage], phase)` 等待 X/W TMA 完成；
2. 将 `tiled_mma.accumulate_` 设为 `Zero`，使该 128-wide K tile 的第一个 WGMMA 覆盖 `tCr`；
3. `TileK=128` 被分成 `128/32=4` 个 WGMMA K atom；第一个之后把 accumulate mode 改为 `One`；
4. `warpgroup_commit_batch()` 提交 WGMMA group；
5. `warpgroup_wait<0>()` 等所有 WGMMA 完成；
6. 256 个 math threads 对 `writable[stage]` arrive，允许 loader 复用 shared stage；
7. 把本 K tile 的 FP32 partial result 乘 group scale，再累加到 `tDr`。

核心代码是：

```cpp
for (int ik = 0; ik < size<2>(tAr); ++ik) {
  cute::gemm(tiled_mma, tBr(...), tAr(...), tCr(...));
}
warpgroup_commit_batch();
warpgroup_wait<0>();

tDr = tCr * yscale[igroup] + tDr;
```

所以每个输出 task、每个 math warpgroup 发出：

```text
56 K tiles * 4 WGMMA instructions = 224 WGMMA instructions
```

两个 warpgroup 合起来覆盖完整的 `[128,48]` output tile。

### 2.9.4 `cute::gemm` 到 PTX 的内层调用链

`src/group_gemm/kernels.cuh:457-469`（CUDA kernel `group_gemm_fp8_kernel`）中的 `cute::gemm` 继续经过：

1. `3rd/cutlass/include/cute/algorithm/gemm.hpp:77-89`（函数模板 `cute::gemm`）：把三参数形式扩展成 `D=A*B+C` 四参数形式；
2. `3rd/cutlass/include/cute/algorithm/gemm.hpp:175-386`（函数模板 `cute::gemm` 的 register-tensor dispatch）：按 fragment layout 展开 atom 调用；
3. `3rd/cutlass/include/cute/atom/mma_atom.hpp:87-105`（成员函数 `cute::MMA_Atom::call`）：执行 `mma_unpack`；
4. `3rd/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:382-430`（函数模板 `cute::SM90::GMMA::mma_unpack`）：把 descriptor/accumulator tensor 解包成 PTX operand；
5. `3rd/cutlass/include/cute/arch/mma_sm90_gmma_ext.hpp:28817-28854`（函数 `MMA_64x48x32_F32E4M3E4M3_SS_TN::fma`）：发出 `wgmma.mma_async`。

## 2.10 Epilogue：FP32 -> BF16 -> TMA store

完成全部 56 个 K tile 后，`src/group_gemm/kernels.cuh:485-523`（CUDA kernel `group_gemm_fp8_kernel`）执行：

1. 将 FP32 `tDr` 转换为 BF16 `tCrh`；
2. 使用 `SM90_U16x8_STSM_T` 将每个 math warpgroup 的 register fragment 写入 shared `shm_c`；
3. `syncwarpgroup(iwarpgroup)` 保证该 warpgroup 的 shared writes 完成。其 wrapper 位于 `src/utils/utils.cuh:601-607`（函数 `hpc::syncwarpgroup`）；
4. `tma_store_fence()` 建立 generic/shared 到 async TMA proxy 的可见性；
5. 每个 math warpgroup 的一个 elected leader 发出一个 TMA store。

`GroupGEMMFp8Config::CopyBoxY` 在 `src/group_gemm/config.h:85-95`（类型 `GroupGEMMFp8Config`，成员函数 `get_tma`）定义为：

```text
[TileN / WarpgroupM, TileM] = [64,48]
```

因此两个 warpgroup 分别写 output N 方向的前 64 列和后 64 列。store 坐标位于 `src/group_gemm/kernels.cuh:512-522`（CUDA kernel `group_gemm_fp8_kernel`）：

```cpp
tDg(_, itile_n * 2 + iwarpgroup, itile_m)
```

其中 `td_y=td_xy+igroup*2+1` 是该 group 的动态 Y descriptor，所以 store 自动使用该 group 的 output base 和实际 sequence length。

`tma_store_wait<0>`、`tma_store_fence`、`tma_store_arrive` 的底层实现位于 `3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:1212-1259`（函数 `cute::tma_store_fence`、`cute::tma_store_arrive`、`cute::tma_store_wait<Count>`）。

## 2.11 整体数据流

```text
Python tensors
  X[576,7168] FP8
  W[8,4096,7168] FP8
  seqlens/cu_seqlens
  scale[8] FP32
       |
       | torch.ops dispatcher
       v
C++ group_gemm_fp8_entry
  - validate
  - allocate Y/tma descriptors/tile metadata
       |
       +-----------------------------------+
       | CUDA kernel 1                     |
       | update_grouped_tma                |
       | - X/Y descriptor per group        |
       | - tiles/cu_tiles                  |
       +------------------+----------------+
                          | PDL dependency
                          v
       +-----------------------------------+
       | CUDA kernel 2                     |
       | group_gemm_fp8_kernel             |
       |                                   |
       | persistent CTA scheduler          |
       |     |                             |
       |     v                             |
       | TMA load WG (128 threads)         |
       | global X/W -> shared stage[0..7]  |
       |     | readable mbarrier           |
       |     v                             |
       | 2 WGMMA WGs (256 threads)         |
       | FP8 x FP8 -> FP32 partial         |
       | scale + accumulate                |
       |     | writable mbarrier           |
       |     v                             |
       | FP32 -> BF16 -> shared -> TMA     |
       +------------------+----------------+
                          |
                          v
                  Y[576,4096] BF16
```

## 2.12 最终结论

1. Python `hpc.group_gemm_pertensor_fp8` 的主 CUDA 实现在 `src/group_gemm/kernels.cuh:215-530`（函数 `hpc::group_gemm::kernels::group_gemm_fp8_kernel`）。
2. `pertensor` Python op 与 `group_gemm_fp8` 共用 `src/group_gemm/entry.cc:16-89`（函数 `group_gemm_fp8_entry`）和 `src/group_gemm/group_gemm_pertensor_fp8.cu:329-420`（函数 `group_gemm_fp8_async`）。
3. 默认调用先执行 `update_grouped_tma`，再执行主 GEMM kernel；前者解决 ragged groups 的动态 descriptor 和 tile prefix，后者完成实际 GEMM。
4. 本次参数实际实例为 `TileM=48, TileN=128, TileK=128, Stages=8, TaskLoopPolicy=2, UsePDL=true`；`temp/run.log:13-21` 的 `Tiler_MN=(_48,_128)` 与该分发完全一致。
5. 主 kernel 每 CTA 使用一个 128-thread TMA loader warpgroup 和两个 128-thread WGMMA warpgroup；WGMMA 最终发出 `m64n48k32.f32.e4m3.e4m3`，FP32 累加并乘 group scale，最后转换成 BF16，由 TMA store 写回。

---

## 3. `src/group_gemm/entry.cc` 完善 tmas 调试打印 + dtype / device 是否为 enum 的说明

### 3.1 修改内容（函数 `hpc::group_gemm::group_gemm_fp8_entry`，`src/group_gemm/entry.cc:17`）

原 `LIKE_DEBUG` 块（`src/group_gemm/entry.cc:61-86`）有两处错误且没有真正打印，本次完善如下：

```cpp
#ifdef LIKE_DEBUG
  auto tmas_shape = tmas.sizes();        // 原为 tmas.shape()
  auto tmas_dtype = tmas.dtype();
  auto tmas_stride = tmas.strides();     // 原为 tmas.stride()
  auto tmas_device = tmas.device();

  printf("[LIKE_DEBUG] tmas info:\n");
  printf("  shape = [");
  for (size_t i = 0; i < tmas_shape.size(); ++i) {
    printf("%s%lld", i ? ", " : "", static_cast<long long>(tmas_shape[i]));
  }
  printf("]\n");
  printf("  stride = [");
  for (size_t i = 0; i < tmas_stride.size(); ++i) {
    printf("%s%lld", i ? ", " : "", static_cast<long long>(tmas_stride[i]));
  }
  printf("]\n");
  printf("  dtype  = %s\n", c10::toString(tmas_dtype.toScalarType()));
  printf("  device = %s\n", tmas_device.str().c_str());
#endif
```

修正点说明：

1. `tmas.shape()` → `tmas.sizes()`：本 torch 版本的 `at::Tensor` 没有无参 `shape()` 方法；shape 用 `TensorBase::sizes()` 获取（`torch_include/ATen/core/TensorBase.h:255`，返回 `IntArrayRef`）。
2. `tmas.stride()` → `tmas.strides()`：`Tensor::stride(int64_t dim)` 必须带维度参数、只返回单个维度的 stride；要拿完整 stride 数组应调用 `strides()`（`torch_include/ATen/core/TensorBase.h:264`，返回 `IntArrayRef`）。
3. `tmas.device()` 本身是对的：返回 `c10::Device`（`torch_include/ATen/core/TensorBase.h:449`）。
4. 打印：dtype 先经 `TypeMeta::toScalarType()` 取回 enum 值，再用 `c10::toString` 转字符串；device 用 `Device::str()`。

编译开关：`LIKE_DEBUG` 由 `setup.py:41-43` 读取 `ENABLE_LIKE_DEBUG` 环境变量，传给 CMake `-DENABLE_LIKE_DEBUG=ON`（`CMakeLists.txt:22-24` 定义 `-DLIKE_DEBUG=1`）。构建命令：

```bash
CUDA_HOME=/share_data/users/like/opt/cuda-13.0 ENABLE_LIKE_DEBUG=1 make wheel > temp/make.log
```

### 3.2 dtype 是 enum 类型吗？

**dtype 的“值”本质上是 enum，但 `Tensor::dtype()` 返回的不是 enum 本身，而是一个封装结构体。**

- `at::Tensor::dtype()` 的返回类型是 `caffe2::TypeMeta`（`torch_include/ATen/core/TensorBase.h:444`，函数 `TensorBase::dtype`），它是一个 `class C10_API TypeMeta final`（`torch_include/c10/util/typeid.h:319`）。
- `TypeMeta` 内部只保存一个 `uint16_t index_`，这个 index 就是 `c10::ScalarType` enum 的整数值；`toScalarType()` 把它转回 enum（`torch_include/c10/util/typeid.h:480`）。
- 真正的 dtype enum 是 `c10::ScalarType`：`enum class ScalarType : int8_t`（`torch_include/torch/headeronly/core/ScalarType.h:258`），所有枚举值由宏 `AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS` 展开生成（`torch_include/torch/headeronly/core/ScalarType.h:103`、`:260`），包括 `Float8_e4m3fn`、`Int32`、`Float`、`BFloat16` 等。
- 把 enum 转成名字打印用 `c10::toString(ScalarType)`（`torch_include/torch/headeronly/core/ScalarType.h:320`，函数 `c10::toString`）。
- `src/group_gemm/entry.cc:29` 的 `x.dtype() == torch::kFloat8_e4m3fn` 能直接比较，是因为 `torch_include/c10/core/ScalarTypeToTypeMeta.h:42-56` 提供了 `operator==(TypeMeta, ScalarType)`（以及反向）的自由函数。

> 小结：tensor dtype 的“语义类型”是 `c10::ScalarType` —— 它是 `enum class`；`Tensor::dtype()` 通过 `caffe2::TypeMeta` 这个 wrapper 返回，需 `.toScalarType()` 才拿到 enum。

### 3.3 device 是 enum 类型吗？

**device 本身不是 enum，它是一个结构体；结构体内部保存的“设备类型”才是 enum。**

- `at::Tensor::device()` 返回 `c10::Device`（`torch_include/ATen/core/TensorBase.h:449`，函数 `TensorBase::device`）。
- `c10::Device` 是 `struct C10_API Device final`（`torch_include/c10/core/Device.h:32`），它包含两个成员：`DeviceType type_`（`torch_include/c10/core/Device.h:171`）和 `DeviceIndex index_`（即设备序号，如 0）。
- 设备类型 enum 是 `c10::DeviceType`：`enum class DeviceType : int8_t`（`torch_include/torch/headeronly/core/DeviceType.h:35`），取值 `CPU=0`、`CUDA=1`、`HIP=6`、`MPS=13` 等。
- 打印设备用 `Device::str()`（`torch_include/c10/core/Device.h:168`），返回形如 `cuda:0` 的字符串。

> 小结：device 的“语义类型”里，`DeviceType` 是 enum class；但 `Tensor::device()` 返回的是 `c10::Device` 结构体（enum 成员 + 设备序号）。

### 3.4 tensor dtype / device enum 定义位置汇总

| 类型 | 定义位置 | 说明 |
| --- | --- | --- |
| `c10::ScalarType`（dtype enum） | `torch_include/torch/headeronly/core/ScalarType.h:258`（`enum class ScalarType : int8_t`） | 枚举值由宏 `AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS` 在 `:103`、`:260` 展开 |
| `c10::kFloat8_e4m3fn` 等常量 | `torch_include/c10/core/ScalarType.h:38-42`（宏 `DEFINE_CONSTANT`） | `constexpr ScalarType`；`torch::` 通过 `using namespace at;` 可见（`torch_include/torch/csrc/api/include/torch/types.h:37`） |
| `caffe2::TypeMeta`（`Tensor::dtype()` 返回类型） | `torch_include/c10/util/typeid.h:319`；`toScalarType()` 在 `:480` | wrapper，内含 ScalarType index |
| `c10::DeviceType`（device 类型 enum） | `torch_include/torch/headeronly/core/DeviceType.h:35` | `CPU=0, CUDA=1, ...`；由 `torch_include/c10/core/DeviceType.h:8` 引入 |
| `c10::Device`（`Tensor::device()` 返回类型） | `torch_include/c10/core/Device.h:32`；`DeviceType type_` 在 `:171`，`str()` 在 `:168` | struct：enum 成员 + 设备序号 |
| TypeMeta<->ScalarType 比较桥接 | `torch_include/c10/core/ScalarTypeToTypeMeta.h:42-56` | 使 `entry.cc:29` 的 `x.dtype() == torch::kFloat8_e4m3fn` 可编译 |
| dtype 转字符串 | `torch_include/torch/headeronly/core/ScalarType.h:320`（`c10::toString`） | 打印用 |

### 3.5 运行效果

启用 `ENABLE_LIKE_DEBUG=1` 后，调用 `group_gemm_fp8` / `group_gemm_pertensor_fp8` 会输出类似：

```text
[LIKE_DEBUG] tmas info:
  shape = [16, 128]
  stride = [128, 1]
  dtype  = Float8_e4m3fn
  device = cuda:0
```

（`tmas` 在 `src/group_gemm/entry.cc:53-60` 分配：传入 `tma_desc` 时复用外部 tensor，否则 `torch::empty({num_group * 2, 128}, options)`，dtype 继承 `x` 的 `Float8_e4m3fn`，device 为当前 CUDA 设备。）

---

## 5. 以本次运行为例详解 update_grouped_tma

本节以 temp/run.log 记录的 tests/test_group_gemm_pertensor_like.py 一次运行为例，追踪 update_grouped_tma 从 Python 到 CUDA kernel 的完整调用链。代码引用均使用相对代码库根目录的路径 + 行号 + 函数名。

### 5.1 本次运行的数据与形状

测试入口 tests/test_group_gemm_pertensor_like.py:46（函数 test_group_gemm_pertensor_fp8），实际参数在 tests/test_group_gemm_pertensor_like.py:75-80：

    num_group=8; actual_m=256; n=4096; k=7168

- seqlens = (1+torch.arange(8))*16，得 [16,32,48,64,80,96,112,128]（tests/test_group_gemm_pertensor_like.py:49）
- cu_seqlens = [0,16,48,96,160,240,336,448,576]（tests/test_group_gemm_pertensor_like.py:61）
- total_seq=576，所以 mean_seq=576/8=72（tests/test_group_gemm_pertensor_like.py:51-53）作为 num_seq_per_group_avg 传入
- x.shape=[576,7168]、w.shape=[8,4096,7168]、结果 y=[576,4096]（BF16）

以上与 temp/run.log:1-4 一致；temp/run.log:13-17 是本仓库上一轮加的 [LIKE_DEBUG] 打印，说明 tma buffer dtype=Float8_e4m3fn、shape=[16,128]，与 entry.cc 的 torch::empty({num_group*2,128}, options) 对应。

### 5.2 分派：num_seq_per_group_avg=72 选中哪一组模板

group_gemm_fp8_async（src/group_gemm/group_gemm_pertensor_fp8.cu:360，函数 group_gemm_fp8_async）按平均 seq 长度分派模板。72 落入 else if (num_seq_per_group_avg <= 96) 分支：

- src/group_gemm/group_gemm_pertensor_fp8.cu:422 —— group_gemm_fp8_async 设 kTileM=48, kStage=8
- 固定 kTileN=128, kTileK=128, kWarpgroupM=2, kWarpgroupN=1, kSwizzleX=128, kSwizzleW=128, kSwizzleY=64（src/group_gemm/group_gemm_pertensor_fp8.cu:366-372）

与 temp/run.log:18 打印的 "int kTileM:48, ... kStage:8, ..." 完全一致；temp/run.log:38 的 Tiler_MN=(_48,_128) 印证 X 的 TMA tile 是 48(M) x 128(K)。

随后进入 launch_group_gemm_fp8（src/group_gemm/group_gemm_pertensor_fp8.cu:27，函数 launch_group_gemm_fp8）：

    auto X = make_tensor(make_gmem_ptr(...), make_shape(m,k), make_stride(k, Int<1>{}));   // :37
    auto W = make_tensor(make_gmem_ptr(...), make_shape(n,k,num_group), ...);              // :39
    auto Y = make_tensor(make_gmem_ptr(...), make_shape(n,m), make_stride(Int<1>{},n));    // :41
    ...
    auto [tma_x, tma_w, tma_y] = config.get_tma(X, W, Y);                                 // :124

config.get_tma 在 src/group_gemm/config.h（GroupGEMMFp8Config::get_tma）用 CUTLASS/CuTe 的 make_tma_copy 生成三个 TMA 描述符，tile 形状对应 temp/run.log:37-65：
- tma_x：X 的 fetch，tile (_48,_128)（M x K）
- tma_w：W 的 fetch，tile (_128,_128)（N x K），因 W 是 [num_group,n,k]，额外带第 3 维 num_group
- tma_y：Y 的 store，tile (_64,_48)（N x M），用于结果写回

### 5.3 td_xy 与 tma_xy 的来历

src/group_gemm/group_gemm_pertensor_fp8.cu:176（launch_group_gemm_fp8）把 tmas_ptr 位转成 cute::TmaDescriptor 数组（每个 group 2 个：1 个 X + 1 个 Y）：

    auto *tma_xy = static_cast<cute::TmaDescriptor*>(tmas_ptr);                             // :176
    vec_t<cute::TmaDescriptor,2> td_xy{ *tma_x.get_tma_descriptor(), *tma_y.get_tma_descriptor() };  // :180-183

td_xy 是"干净模板"：X 的基址还指向 x_ptr 起点、Y 指向 y_ptr 起点，shape 是整张大 [m,k]/[n,m]。tma_xy 则是 entry.cc:59 分配的字节buffer（torch::empty({num_*2,128})，每个 TmaDescriptor 占 128 字节），update_grouped_tma 会往里面逐 group 写入改写后的描述符。

注意：vec_t 定义在 src/utils/utils.cuh:30（struct vec_t），是固定长度小数组包装。

### 5.4 启动网格：cudaLaunchKernelEx 与 PDL

launch_group_gemm_fp8 用 cudaLaunchKernelEx 且启用程序化流串行化：

1. cudaLaunchAttributeProgrammaticStreamSerialization（src/group_gemm/group_gemm_pertensor_fp8.cu:190-193）声明这个 kernel 允许后续 kernel 用 PDL 直接依赖，省去流级同步。
2. 本测试未传 task_map workspace，task_map_ptr 为 null，走 kAssignTask=false 分支：gridDim = num_group+1 = 9（src/group_gemm/group_gemm_pertensor_fp8.cu:213-214）；blockDim = kThreadPerBlock = 32（src/group_gemm/group_gemm_pertensor_fp8.cu:186）。
3. 实例化并启动 kernels::update_grouped_tma<Tin,Tout,TmapX,TmaY,kTileM=48,kGroupPerThread=8,kThreadPerBlock=32,kAssignTask=false,kUsePDL=true>（src/group_gemm/group_gemm_pertensor_fp8.cu:205-211）。

说明：num_group+1=9 个 block，block 0..7 每个负责 1 个 group 的 2 个描述符（X、Y），block 8（=num_group）专门算 tiles/cu_tiles。若传 task_map，会走 kAssignTask=true 分支，gridDim=2*num_group+1（src/group_gemm/group_gemm_pertensor_fp8.cu:201），多出的 num_group 个 block 用来生成 task_map；本测试不走。

### 5.5 kernel 内部分工

进入 update_grouped_tma（src/group_gemm/kernels.cuh:65，函数 update_grouped_tma）：

- int igroup = blockIdx.x（src/group_gemm/kernels.cuh:73）
- 若 kUsePDL 先 cudaGridDependencySynchronize()（src/group_gemm/kernels.cuh:75-77），确保上一个 producer 完成
- 聚合 block（igroup==num_group，即 block 8）走 if 分支（见 5.6 的 cu_tiles）；其余 8 个 block 走 else 分支（src/group_gemm/kernels.cuh:172）更新描述符

#### 5.5.1 每 group 的 X/Y 描述符改写（src/group_gemm/kernels.cuh:172-207）

    int num_seq = seqlens_ptr[igroup];            // 该 group 实际 seq 长度
    uint64_t cu_seqlen = cu_seqlens_ptr[igroup];  // 该 group 起始 seq 偏移
    auto *x_ibatch_ptr = x_ptr + cu_seqlen * k;   // 该 group 在 X 中的基址
    auto *y_ibatch_ptr = y_ptr + cu_seqlen * n;   // 该 group 在 Y 中的基址

    if (idx < 2) smem_tma_desc[idx] = td_xy[idx];  // :180-182  把模板拷进共享内存
    __syncwarp();
    if (idx == 0) {   // 改 X 描述符
      auto gX = make_tensor(make_gmem_ptr(x_ibatch_ptr), make_shape(num_seq, k),
                            make_stride(k, Int<1>{}));
      update_tma_gtensor<TmaX>(smem_tma_desc[0], gX);   // :187-190
    }
    if (idx == 1) {   // 改 Y（结果写回）描述符
      auto gY = make_tensor(make_gmem_ptr(y_ibatch_ptr), make_shape(n, num_seq),
                            make_stride(Int<1>{}, n));
      update_tma_gtensor<TmaY>(smem_tma_desc[1], gY);   // :194-197
    }

update_tma_gtensor（src/utils/tma.cuh:37，函数 update_tma_gtensor）：
- cute::detail::fill_tma_gmem_shape_stride(...) 求得该 group 的 shape/stride（src/utils/tma.cuh:42）
- cute::tma_descriptor_replace_addr_in_shared_mem 改写全局内存基址为该 group 的基址（src/utils/tma.cuh:45）
- kUpdateShape=true 时 tma_descriptor_replace_shapes_in_shared_mem 改写 shape 为 [num_seq, k] 之对应（src/utils/tma.cuh:47-48）

于是每个 group 的 X 描述符从"整张 [576,7168]"变成"只指向该 group 自己 [num_seq,7168] 的开头"，ragged 分组的偏移与长度被编码进描述符。

改写后，逐描述符提交到全局 tma_xy：

    for (int i = 0; i < 2; i++) {
      __syncwarp();
      if (cute::elect_one_sync()) { cute::tma_desc_commit_group(); cute::tma_desc_wait_group(); }
      tma_descriptor_cp_fence_release(tma_xy + igroup*2 + i, smem_tma_desc[i]);  // :199-207
    }

这里 tma_descriptor_cp_fence_release 是切散、带 release 语义的 cp 描述符到全局的操作，实现在 3rd/cutlass/include/cute/arch/copy_sm90_desc.hpp:425（函数 tma_descriptor_cp_fence_release）。

#### 5.5.2 聚合 block 计算 cu_tiles（src/group_gemm/kernels.cuh:143-170）

block 8（igroup==num_group）不做描述符更新，转而计算每个 group 的 M tile 数与前缀和：

    tiles[i] = (seqlens_ptr[igroup] + kTileM - 1) / kTileM;   // 该 group 的 M tile 数（向上取整）
    BlockScan::ExclusiveSum(...)   // 得到前缀和 cu_tiles（src/group_gemm/kernels.cuh:156-159）
    cu_tiles_ptr[igroup] = tiles[i];                          // :165
    cu_tiles_ptr[num_group] = block_aggregate;                 // :169  总 M tile 数

本例 kTileM=48：各 group M tile 数 = ceil(16/48)...ceil(128/48) = {1,1,1,2,2,2,3,3}；前缀和 cu_tiles = {0,1,2,3,5,7,9,12}，cu_tiles[8]=15。主 GEMM kernel 用这些 cu_tiles 做 per-block 工作量分配。
```
# simulate.py
import numpy as np
import torch
num_group=8
seqlens = (1+torch.arange(num_group, dtype=torch.int32, device='cuda'))*16
cu_seqlens = torch.cumsum( torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens]), dim=0).to(torch.int32)
seqlens_ptr = seqlens
kTileM = 48
kGroupPerThread = 8
kThreadPerBlock = 32

# out
tiles_ptr = torch.zeros(num_group, dtype=torch.int32, device=seqlens.device)
cu_tiles_ptr =  torch.zeros(num_group+1, dtype=torch.int32, device=seqlens.device)

tiles_thread_block = torch.zeros(kThreadPerBlock, kGroupPerThread, dtype=torch.int32, device=seqlens.device)
block_aggregate = 0
for idx in range(kThreadPerBlock):
    tiles = torch.zeros(kGroupPerThread, dtype=torch.int32, device=seqlens.device)
    for i in range(kGroupPerThread):
        igroup = idx * kGroupPerThread + i
        if igroup < num_group:
            tiles[i] = (seqlens_ptr[igroup] + kTileM - 1) // kTileM
            tiles_ptr[igroup] = tiles[i]
        else:
            tiles[i] = 0
        block_aggregate += tiles[i]
    print(f"idx:{idx},tiles:{tiles}")
    tiles_thread_block[idx] = tiles # This is cpu
# simulate ExclusiveSum
zero=torch.zeros(1,dtype=tiles_thread_block.dtype, device=tiles_thread_block.device)
inclusive=torch.cumsum(tiles_thread_block.reshape(-1), dim=0)
ExclusiveSum_flatten = torch.cat([zero, inclusive])
for idx in range(kThreadPerBlock):
    tiles_thread_block[idx] = ExclusiveSum_flatten[idx*kGroupPerThread: (idx+1)*kGroupPerThread]

#block_aggregate = tiles_thread_block.sum().item()
for idx in range(kThreadPerBlock):
    for i in range( kGroupPerThread):
      igroup = idx * kGroupPerThread + i
      tiles = tiles_thread_block[idx]
      if igroup < num_group:
        cu_tiles_ptr[igroup] = tiles[i]
    if idx == 0:
      cu_tiles_ptr[num_group] = block_aggregate
print(f"cu_tiles_ptr:{cu_tiles_ptr}")
```
### 5.6 主 GEMM 如何消费这些描述符

描述符就绪：update_grouped_tma 结尾用 PDL 的 cudaTriggerProgrammaticLaunchCompletion()（src/group_gemm/kernels.cuh:210-212）发出完成信号；随后启动的 group_gemm_fp8_kernel（src/group_gemm/kernels.cuh:215，函数 group_gemm_fp8_kernel）在开头 cudaGridDependencySynchronize()（src/group_gemm/kernels.cuh:286-287）等描述符就绪。

kernel 内按 task loop 策略取每个 tile 的 (igroup, itile_m, itile_n)（本测试 kTaskLoopPolicy=2，见 src/group_gemm/kernels.cuh:314）：
- load WG 用 td_x = td_xy + igroup*2 做 tma_a.with(td_x, ...)（src/group_gemm/kernels.cuh:372-380）
- store 用 td_y = td_xy + igroup*2+1 写回（src/group_gemm/kernels.cuh:519 处 cute::copy(tma_d.with(td_y),...)）

这些 td_xy（也就是 update_grouped_tma 写入的 tma_xy buffer）正是描述符数组。消费前用 tma_descriptor_fence_acquire 做 acquire（src/group_gemm/kernels.cuh:302-303），其底层实现在 3rd/cutlass/include/cute/arch/copy_sm90_desc.hpp:459（函数 tma_descriptor_fence_acquire）。

### 5.7 小结

对本次 temp/run.log 的调用，update_grouped_tma（src/group_gemm/kernels.cuh:65）：
1. 以 num_group+1=9 个 block 启动（PDL 程序化单响应器）；block 0..7 每个负责 1 个 group 的 X、Y 描述符，block 8 算 tiles/cu_tiles。
2. thread 0 / 1 分别用 update_tma_descriptor_gtensor 把 X / Y 描述符的全局基址改成本 group 的 x/y+cu_seqlen*stride、形状改成本 group 的 [num_seq,...]。
3. 用 tma_descriptor_cp_fence_release 把改好的描述符 cp 到 tma_xy + igroup*2 [ +1 ] 并 release。
4. block num_group 计算 tiles/cu_tiles（M tile 前缀和）。
5. 末尾 cudaTriggerProgrammaticLaunchCompletion；主 GEMM 凭 cudaGridDependencySynchronize 拿到更新后的描述符开始计算。

---

## 6. 详解 elect_one_sync（cutlass 集群/warp 选举）

本节讲解 3rd/cutlass/include/cute/arch/cluster_sm90.hpp 中的 elect_one_sync 函数。该函数在 hpc-ops 的 group_gemm 实现里被多处使用（例如更新 TMA 描述符后让"单个线程"提交 commit/wait，见 src/group_gemm/kernels.cuh:202），其作用是"在一个 warp 内选出一个代表线程"。

### 6.1 函数签名与语义

函数定义在 3rd/cutlass/include/cute/arch/cluster_sm90.hpp:180（函数 elect_one_sync）：

    CUTE_HOST_DEVICE uint32_t elect_one_sync()
    {
    #if defined(CUTE_ARCH_ELECT_ONE_SM90_ENABLED)
      uint32_t pred = 0;
      uint32_t laneid = 0;
      asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "     elect.sync %%rx|%%px, %2;\n"
        "@%%px mov.s32 %1, 1;\n"
        "     mov.s32 %0, %%rx;\n"
        "}\n"
        : "+r"(laneid), "+r"(pred)
        : "r"(0xFFFFFFFF));
      return pred;
    #elif defined(__CUDA_ARCH__)
      return (threadIdx.x % 32) == 0;
    #else
      return true;
    #endif
    }

- 返回类型 uint32_t，返回值作为谓词（predicate）：被选中的那条线程得到 1，其余线程得到 0。
- 因此调用方式通常是 if (cute::elect_one_sync()) { ... }，让 warp 内只有一条线程进入分支执行"一次即可"的工作。

### 6.2 分支依赖：CUTE_ARCH_ELECT_ONE_SM90_ENABLED

该宏在 3rd/cutlass/include/cute/arch/cluster_sm90.hpp:42-44 定义，要求：

- __CUDA_ARCH__ >= 900（即 sm_90 / Hopper 及以上，使用该宏说明需要现代 TMA/集群指令）
- __CUDACC_VER_MAJOR__ >= 12（CUDA 12.x 编译器，因为 elect.sync 及 elect semantics 需要新 ptxas）

### 6.3 SM90 路径下的 PTX 内联汇编

真实路径走 elect.sync 指令：

    .reg .b32 %%rx;    // 32 位寄存器 rx:用来拿"被选中线程的 lane id"
    .reg .pred %%px;   // 谓词寄存器 px:1 表示当前线程被选中
    elect.sync rx|px, 0xFFFFFFFF;
    // 如果自己是被选中者(px=1)，把 pred 置 1
    // 最后把 lane id 保存到 laneid(但本函数没用到，只返回 pred)

关键点：

1. elect.sync 需要"成员掩码"参数，这里传 0xFFFFFFFF，表示参与 mask 的 32 条 lane（整个 warp 全部参与）。
2. 辅助汇编把"谓词结果"写进 pred，而把"被选的 lane id"写进 laneid；elect_one_sync 只返回 pred，elect_one_leader_sync（cluster_sm90.hpp:203-210）则把二者都返回。
3. 兼容路径：当 CUTE_ARCH_ELECT_ONE_SM90_ENABLED 未定义但仍是 __CUDA_ARCH__（例如较旧编译器被禁用）时，退化为 return (threadIdx.x % 32) == 0，即固定选 lane 0。非设备环境返回 true（主机路径基本只在 CUTE_HOST_DEVICE 里做模拟验证）。

### 6.4 在 hpc-ops 中的用法示例

以本项目的 group_gemm 主 kernel 为例（src/group_gemm/kernels.cuh:240-241）：

    int elected = cute::elect_one_sync();
    bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

- iwarp 来自 idx/32（每个 warp 的 warp id）。iwarp%4==0 说明是 warpgroup 内第一个 warp（load warpgroup 或 math warpgroup 的第一个 warp）。
- 对四个 warp 中，"第一个 warp"（%4==0）+ elect_one_sync() 选出的 1 条 lane 共同决定 leader：这个 leader 才有资格做 arriver（如 tma_desc_commit/wait、tma store 的 arrive、barrier 管理）。

同样地，update_grouped_tma 里提交描述符也用到了 elect_one_sync（src/group_gemm/kernels.cuh:199-206）：

    for (int i = 0; i < 2; i++) {
      __syncwarp();
      if (cute::elect_one_sync()) {       // <= 每条 warp 里只有一条 lane 进入
        cute::tma_desc_commit_group();
        cute::tma_desc_wait_group();
      }
      tma_descriptor_cp_fence_release(tma_xy + igroup*2 + i, smem_tma_desc[i]);  // release 描述符到全局
    }

这里所有 warp 都要负责 release（store 到全局），但 elect_one_sync 则只让一个 lane 做 TMA commit/wait，避免对"全局描述符队列"的并发。

### 6.5 与"单纯看 threadIdx.x==0"的区别

- elect.sync 是硬件指令，支持对任意"由并发执行到该点的 threads"做选举；而 (threadIdx.x%32)==0 只是"固定 lane 0 为 leader"。
- 在带有分支发散、或某条 lane 提前退出（mask 中有 inactive lane）时，elect.sync 仍保证在参与 mask 线程里恰好选出一个。
- 但 elect_one_sync 每次调用都重新选举，**不保证**每次都选同一条 lane；它只保证"恰恰选出一个代表"，适合用它的地方都是"只需一个线程来执行一次副作用"（arrive/commit/wait）。

### 6.6 相关兄弟函数（cluster_sm90.hpp）

- cluster_arrive_relaxed / cluster_arrive / cluster_wait / cluster_sync（cluster_sm90.hpp:48-83）：基于 barrier.cluster 的集群同步。
- cluster_grid_dims / cluster_id_in_grid / block_id_in_cluster / cluster_shape / block_rank_in_cluster（cluster_sm90.hpp:86-163）：读取 nclusterid/clusterid/cluster_ctaid/cluster_nctaid/cluster_ctarank 的内建寄存器，用于确定 cluster 的几何形状与当前 block 在 cluster 中的身份。
- set_block_rank（cluster_sm90.hpp:166-177）：用 mapa.shared::cluster 把"本地共享地址+目标 block rank"映射成集群内可访问的远程共享地址，是集群间 SMEM 通信的基础。
- store_shared_remote（cluster_sm90.hpp:234-244）：组合 set_block_rank + st.async.shared::cluster 做远程共享内存写（带 mbarrier 完成计数）。

### 6.7 参考规范

elect.sync 指令在 PTX ISA（sm_90 引入 elect 这一谓词选举指令）有明确定义：

    elect.sync d | p{, membermask};

- p 为真只对"满足了该并发执行束且被选中的那条 lane"为真；
- d 输出被选中 lane 在 membermask 中的索引（0 .. membermask 内被选中的序号）。

CUTLASS/CuTe 的 elect_one_sync 即 (elect.sync 的 pred + laneid) 中只保留了 pred，elect_one_leader_sync 额外保留 laneid。

---

## 7. `src/group_gemm/kernels.cuh:203-204` 的 `commit + wait`

### 7.1 结论

在当前的 `update_grouped_tma` kernel 中，203--204 行之前**没有**发起 X/Y tensor 的 TMA copy。这里的

```cpp
cute::tma_desc_commit_group();
cute::tma_desc_wait_group();
```

是从 CUTLASS 的“动态重写一个可能已被 TMA 使用的 descriptor”流程带来的通用保护。在真正有此前发起的、由同一线程管理的 bulk async 请求时，它保证 TMA engine 已经读完 tensormap，才允许用 206 行覆盖该全局 descriptor；但对这个独立的 updater kernel 而言，它会提交一个空 group 后立刻返回，功能上是冗余的防御性代码，不是在等待 189/196 行的 descriptor 修改完成。

更精确地说，它也不能等待别的线程、别的 warp 或别的 kernel 的 TMA 请求：PTX 将 bulk async-group 定义为 *per-thread*，`commit_group` 只收集“执行该指令的线程”此前发起、尚未提交且采用 `bulk_group` completion 的请求。

### 7.2 203--204 行实际发出的指令

- `cute::tma_desc_commit_group()` 在 `3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:1235` 中直接发出 `cp.async.bulk.commit_group`。
- `cute::tma_desc_wait_group()` 在同文件 `:1262` 中发出 `cp.async.bulk.wait_group.read 0`。
- `read 0` 的含义是等待该线程此前所有已提交 bulk group 至少完成对 tensormap 和源地址的读取；这正是“之后可替换 descriptor”的关键，而不是普通的 CTA/warp barrier。
- 若此前没有可提交请求，`commit_group` 按 PTX 规范创建空 group；本处随后的 `wait_group.read 0` 因而没有可等的 TMA 工作。循环运行两次也不会增加当前路径的正确性。

这也是为什么一般动态 descriptor 重用场景需要二者连用：先 `commit` 把未提交请求纳入可等待的 group，之后 `wait_group.read 0` 让 descriptor 的异步读取阶段结束。CUTLASS 在动态 tensormap 更新路径中使用了同一惯用法，并明确说明“没有未提交指令时会得到 empty bulk async-group”（例如 `3rd/cutlass/include/cutlass/epilogue/collective/sm100_epilogue_array_tma_warpspecialized.hpp:1564`）。

### 7.3 本函数在 203 行之前实际做了什么

执行顺序如下：

```text
180--182  把模板 TMA descriptor 复制到 shared memory（不是 tensor TMA copy）
187--196  用 tensormap.replace 修改 shared descriptor 的地址和 shape（不是数据搬运）
201       __syncwarp()，使 warp 对更新后的 shared descriptor 对齐
203--204  commit + wait；当前 kernel 中是空 bulk group
206       tensormap.cp_fenceproxy：将 128B descriptor 从 shared 发布到 global，并做 release
后续主 kernel
379--380  TMA load X: global -> shared
520--522  TMA store Y: shared -> global，并提交 store group
```

具体证据：

- `update_tma_gtensor` 只调用 `tma_descriptor_replace_addr_in_shared_mem` 和 shape replacement（`src/utils/tma.cuh:54-58`）；底层是 `tensormap.replace.*.shared`，只改 128B tensor-map 对象的字段。
- 206 行的 `tma_descriptor_cp_fence_release` 实现为 `tensormap.cp_fenceproxy.global.shared...release.gpu.sync.aligned`（`3rd/cutlass/include/cute/arch/copy_sm90_desc.hpp:425-432`）。它复制的是 **descriptor 本身**，并建立 generic -> tensormap proxy 的 release 顺序；它不是 X/Y tensor 的 TMA data copy。
- 真正以 `td_x`/`td_y` 作为 descriptor 发起数据搬运的是后续 `group_gemm_fp8_kernel` 中的 `cute::copy`：load 在 `src/group_gemm/kernels.cuh:379-380`，store 在 `src/group_gemm/kernels.cuh:520-522`。该主 kernel 在 host 侧由 updater 之后启动（`src/group_gemm/group_gemm_pertensor_fp8.cu:178-246`、`:248` 之后）。

因此，当前路径真正必须保留的发布机制是 201 行的 warp 对齐及 206 行的 `tensormap.cp_fenceproxy`（消费端在需要时以 `tma_descriptor_fence_acquire` 获取）；203--204 不能替代这些 fence，也不负责把 189/196 的 `tensormap.replace` “提交”。若未来把 descriptor 更新和实际 TMA 请求放进同一个、会复用 descriptor 的 producer kernel，才需要依据该线程此前是否有 in-flight bulk group 来保留这对 `commit + wait`。

---

## 8. 为什么 host 看起来是 policy 2，而 kernel 打印 policy 1

### 8.1 直接原因：`kTaskLoopPolicy` 被声明成了 `bool`

`group_gemm_fp8_kernel` 的模板参数明确是 `int`：`src/group_gemm/kernels.cuh:215-221`（函数 `group_gemm_fp8_kernel`）。但 host 端 `launch_group_gemm_fp8` 的六个 dispatch 分支都写成了 `constexpr bool kTaskLoopPolicy`：

- PDL 分支：`src/group_gemm/group_gemm_pertensor_fp8.cu:137`、`:152`、`:165`（函数 `launch_group_gemm_fp8`）；
- 非 PDL 分支：同文件 `:182`、`:196`、`:208`（函数 `launch_group_gemm_fp8`）。

因此第三个分支中的

```cpp
constexpr bool kTaskLoopPolicy = 2;
```

在声明时就发生 C++ 布尔转换：`2 != 0`，所以变量的实际值是 `true`，数值表现为 `1`。随后它在 `src/group_gemm/group_gemm_pertensor_fp8.cu:169-171`（函数 `launch_group_gemm_fp8`）作为非类型模板实参传给 `group_gemm_fp8_kernel`；`true` 再转换为该模板要求的 `int` 后仍然是 `1`。所以 device printf 打印 `kTaskLoopPolicy=1` 是编译期实例化结果，并不是运行时把 `2` 改成了 `1`。

这不是 `printf` 的格式问题，也不是 host/device 变量不同步，而是 `constexpr bool` 的类型错误。策略 0 和策略 1 恰好不会暴露问题；只有策略 2 被压成了策略 1。

### 8.2 为什么 `shm_seq:36` 仍然能打印

host 的 `shm_seq` 打印位于第三个 `else` 分支的局部代码中。该分支的选择条件是运行时条件：`task_map_ptr == nullptr` 且 `k > 1024`、`n > 1024`，见 `src/group_gemm/group_gemm_pertensor_fp8.cu:136-176`（函数 `launch_group_gemm_fp8`）。进入这个分支只说明选择了“原本意图对应 policy 2 的代码块”，并不说明其中 `constexpr bool kTaskLoopPolicy` 的值是整数 2。

本次日志也逐项吻合：

- `temp/run.log:10-14` 显示 `n=4096`、`k=7168`；因此 `k <= 1024 || n <= 1024` 为假；
- `temp/run.log:73` 显示 `task_map_ptr(nil)`，所以没有进入 task-map 的 policy 0 分支；
- `temp/run.log:75` 的 `shm_seq:36` 等于 `sizeof(int) * (num_group + 1)`，这里 `num_group=8`，即 `4 * 9 = 36`；这只是该分支的 shared-memory 计算结果。

另外，`src/group_gemm/entry.cc:61-68`（函数 `group_gemm_fp8_entry`）只有在 `num_seq_per_group_avg <= 8` 且提供 task-map workspace 时才设置 `task_map_ptr`。本次平均 sequence length 为 72（日志中的 `16,32,...,128`），因此 `task_map_ptr` 为 null 是预期行为。

### 8.3 kernel 实际编译和执行的是 policy 1

在 `group_gemm_fp8_kernel` 中，policy 1 和 policy 2 是 `if constexpr` 的不同编译路径：

- policy 1 在 `src/group_gemm/kernels.cuh:310-313`（函数 `group_gemm_fp8_kernel`）读取 `tiles_ptr`；随后在 `:357-362` 调用 `get_next_tile_horizon`；
- policy 2 在 `src/group_gemm/kernels.cuh:314-318`（函数 `group_gemm_fp8_kernel`）读取 `cu_tiles_ptr` 并设置 `total_m`；随后在 `:363-367` 调用 `get_next_tile_vert`。

由于实际模板值是 1，policy 2 分支在编译期被丢弃，kernel 走的是 horizon 路径。日志中 device printf 的 `total_m:0` 也与此一致：只有 policy 2 的 `src/group_gemm/kernels.cuh:315`（函数 `group_gemm_fp8_kernel`）会给 `total_m` 赋值；policy 1 不会赋值，因此保留初始化值 0。

### 8.4 还要注意 `use_pdl` 的覆盖

`group_gemm_fp8_entry` 在 `src/group_gemm/entry.cc:84-86`（函数 `group_gemm_fp8_entry`）把 `use_pdl=false` 传给 `group_gemm_fp8_async`，但 `src/group_gemm/group_gemm_pertensor_fp8.cu:224-246`（函数 `group_gemm_fp8_async`）第 238 行又无条件执行 `use_pdl = true`。所以本次运行实际走 `launch_group_gemm_fp8` 的 PDL 分支，具体是 `src/group_gemm/group_gemm_pertensor_fp8.cu:121-176`（函数 `launch_group_gemm_fp8`）中的第三个 `else`。

### 8.5 修复建议

将 pertensor host dispatch 中的六处声明改成 `constexpr int kTaskLoopPolicy`：

```text
src/group_gemm/group_gemm_pertensor_fp8.cu:137,152,165,182,196,208
```

至少本次命中的 `:165` 必须改为 `constexpr int kTaskLoopPolicy = 2;`。重新编译后，传给 `group_gemm_fp8_kernel` 的模板实参才会是整数 2，device printf 才会打印 2，并且 `if constexpr (kTaskLoopPolicy == 2)` 会真正选择 `get_next_tile_vert`。同样的 `constexpr bool` 三态策略问题也存在 blockwise host dispatch，但不影响本次 pertensor 日志的根因。

注：当前工作树的 tracked 源码中未保留问题描述里的两条临时 `printf`，但 `temp/run.log` 显然来自加入这些插桩后的构建；这不会改变上述模板类型转换结论。

---

## 9. `get_next_tile_vert`：policy 2 如何把线性任务映射回 group tile

### 9.1 本次日志确实执行了 policy 2

当前 `temp/run.log:94-570` 大量打印 `kTaskLoopPolicy:2`，`temp/run.log:648-651` 也再次打印 policy 2；因此这次运行确实进入了 `get_next_tile_vert`，不是只有 host 端进入了 policy-2 的 dispatch 分支。当前源码在 `src/group_gemm/group_gemm_pertensor_fp8.cu:309`（函数 `launch_group_gemm_fp8`）使用 `constexpr int kTaskLoopPolicy = 2`，并在 `src/group_gemm/group_gemm_pertensor_fp8.cu:320-327`（函数 `launch_group_gemm_fp8`）将它作为模板实参传给 `group_gemm_fp8_kernel`。此前第 8 节记录的是类型修复前的历史状态；本节以当前源码和当前日志为准。

### 9.2 函数签名和业务对象

函数定义在 `src/group_gemm/kernels.cuh:42`（函数 `get_next_tile_vert`）：

```cpp
__device__ __forceinline__ void get_next_tile_vert(
    const int *cu_tiles_ptr, int iblock, int num_group,
    int &igroup, int &itile_m, int &itile_n, int total_m)
```

它不是做 GEMM 或内存拷贝的函数，而是一个纯粹的“任务坐标解码器”：把 persistent CTA 当前拿到的一维任务号 `iblock`，解码成 grouped GEMM 需要的三元坐标：

```text
(igroup, itile_m, itile_n)
```

三者的业务含义是：

- `igroup`：第几个 ragged group。后续用它选择该 group 的 X descriptor，例如 `src/group_gemm/kernels.cuh:419`（函数 `group_gemm_fp8_kernel`）计算 `td_x = td_xy + igroup * 2`，并在 `src/group_gemm/kernels.cuh:429`（函数 `group_gemm_fp8_kernel`）选择 W 的第三维 group 坐标。
- `itile_m`：该 group 内沿 M/sequence 方向的 tile 编号，从 0 开始；对应的 X tile 在 `src/group_gemm/kernels.cuh:426-427`（函数 `group_gemm_fp8_kernel`）使用 `tAg(_, itile_m, itile_k)`。它是 tile index，不是元素行号；实际行起点约为 `itile_m * kTileM`，最后一个 tile 可能是部分有效。
- `itile_n`：沿 N/output-channel 方向的 tile 编号，从 0 开始；对应 W tile 在 `src/group_gemm/kernels.cuh:429-430`（函数 `group_gemm_fp8_kernel`）使用 `tBg(_, itile_n, itile_k, igroup)`。本例 `N=4096`、`kTileN=128`，共有 32 个 N tiles。

### 9.3 输入参数和输出参数

`cu_tiles_ptr`、`iblock`、`num_group`、`total_m` 的含义如下：

1. `cu_tiles_ptr` 是长度为 `num_group + 1` 的 M-tile 排布表。它不是每组的 tile 数，而是 exclusive prefix sum：`cu_tiles_ptr[g]` 是 group `g` 在“所有 group 的 M tiles 拼接数组”中的起始 offset，`cu_tiles_ptr[g+1]` 是结束 offset。该数组由 `src/group_gemm/kernels.cuh:143-170`（函数 `update_grouped_tma`）生成：每组 tile 数在 `src/group_gemm/kernels.cuh:149-150`（函数 `update_grouped_tma`）计算并写入 `tiles_ptr`，BlockScan exclusive sum 在 `src/group_gemm/kernels.cuh:156-159`（函数 `update_grouped_tma`）执行，前缀结果在 `src/group_gemm/kernels.cuh:161-169`（函数 `update_grouped_tma`）写入 `cu_tiles_ptr`，其中 `cu_tiles_ptr[num_group]` 是总 M-tile 数。

2. `iblock` 是当前 CTA 的线性任务号，不是 group id。load warpgroup 在 `src/group_gemm/kernels.cuh:381`（函数 `group_gemm_fp8_kernel`）初始化为 `blockIdx.x`，math warpgroup 在 `src/group_gemm/kernels.cuh:463`（函数 `group_gemm_fp8_kernel`）也初始化为 `blockIdx.x`；每完成一次任务后分别在 `src/group_gemm/kernels.cuh:417` 和 `src/group_gemm/kernels.cuh:492`（函数 `group_gemm_fp8_kernel`）加上 `gridDim.x`，让一个 persistent CTA 继续领取自己的后续任务。本次 host 侧在 `src/group_gemm/group_gemm_pertensor_fp8.cu:258-263`（函数 `launch_group_gemm_fp8`）将 `gridDim` 设为 `num_sm=132`，所以一个 CTA 的任务序列形如 `blockIdx.x, blockIdx.x+132, blockIdx.x+264, ...`。

3. `num_group` 是 group 数，同时也是 `cu_tiles_ptr` 的最后一个合法下标。二分查找在 `src/group_gemm/kernels.cuh:48-57`（函数 `get_next_tile_vert`）的 `[0, num_group]` 范围内查找 prefix boundary。

4. `total_m` 是所有 group 的 M tiles 总数，即 `cu_tiles_ptr[num_group]`。policy 2 在 `src/group_gemm/kernels.cuh:357-361`（函数 `group_gemm_fp8_kernel`）把该值读入 `total_m` 并把 prefix table 搬到 shared memory。它必须大于 0，因为 `get_next_tile_vert` 在 `src/group_gemm/kernels.cuh:45-46` 对它做取模和整除。

5. `igroup`、`itile_m`、`itile_n` 是引用输出参数；函数不会依赖调用前的 `igroup` 值，而是在 `src/group_gemm/kernels.cuh:59-60`（函数 `get_next_tile_vert`）覆盖写入。返回后，`igroup` 是 group 下标，`itile_m` 是 group-local M tile 下标，`itile_n` 是 N tile 下标。

### 9.4 第一阶段：把 `iblock` 拆成 N tile 和全局 M tile

函数前两行（`src/group_gemm/kernels.cuh:45-46`，函数 `get_next_tile_vert`）是：

```cpp
int itile_m_total = iblock % total_m;
itile_n = iblock / total_m;
```

令 `r = itile_m_total`，则线性任务编号满足：

```text
iblock = itile_n * total_m + r
```

这里的 `r` 不是某个 group 内的 M tile，而是把所有 group 的 M tiles 首尾拼接后的“全局 M tile rank”。因此 policy 2 的任务顺序是：固定一个 N tile，沿所有 group 的 M tiles（纵向）遍历；遍历完该 N tile 后再进入下一个 N tile。这就是 `vert` 的含义。

对比之下，policy 1 的 `get_next_tile_horizon` 在 `src/group_gemm/kernels.cuh:28`（函数 `get_next_tile_horizon`）使用另一种除法/取模顺序，先按全局 M rank 再按 N tile 展开；两种 policy 访问的是同一批输出 tiles，只是线性调度顺序不同。

### 9.5 第二阶段：用 prefix sum 二分反查 group 和局部 M tile

假设 `r = itile_m_total`。group `g` 覆盖的全局 M rank 区间是：

```text
[cu_tiles_ptr[g], cu_tiles_ptr[g + 1])
```

所以需要找到满足下面条件的最大 `g`：

```text
cu_tiles_ptr[g] <= r < cu_tiles_ptr[g + 1]
```

代码在 `src/group_gemm/kernels.cuh:48-57`（函数 `get_next_tile_vert`）做的是 upper-bound 风格的二分查找：

- 若 `cu_tiles_ptr[mid] > r`，`mid` 及其右侧不可能是所属 group，收缩 `right`；
- 若 `cu_tiles_ptr[mid] <= r`，`mid` 仍可能是答案，向右推进 `left`，寻找更大的合法 boundary；
- 循环结束后，`right` 是最后一个不超过 `r` 的 prefix 下标。

最后两行（`src/group_gemm/kernels.cuh:59-60`，函数 `get_next_tile_vert`）完成反查：

```cpp
itile_m = r - cu_tiles_ptr[right];
igroup = right;
```

也就是说，`cu_tiles_ptr[right]` 把全局 M rank 转回该 group 的起点，做差后得到 group-local `itile_m`。如果某些 group 的 sequence length 为 0，prefix 中可能有重复值；取最后一个不超过 `r` 的 prefix 会跳过空 group，仍落到真正包含该 tile 的 group。

### 9.6 用本次数据走一遍

`temp/run.log:77-84` 给出的 sequence lengths 是 `[16,32,48,64,80,96,112,128]`；本次 `kTileM=48`，见 `temp/run.log:6`。因此 `update_grouped_tma` 在 `src/group_gemm/kernels.cuh:149`（函数 `update_grouped_tma`）计算出：

```text
每组 M tile 数 tiles = [1, 1, 1, 2, 2, 2, 3, 3]
cu_tiles              = [0, 1, 2, 3, 5, 7, 9, 12, 15]
total_m               = 15
```

`temp/run.log:10-14` 显示 `N=4096`，源码在 `src/group_gemm/kernels.cuh:275`（函数 `group_gemm_fp8_kernel`）得到 `num_tile_n=32`。有效任务号范围是 `0..479`，任务总数为 `15 * 32 = 480`。代表性解码如下：

| `iblock` | `r = iblock % 15` | `itile_n = iblock / 15` | prefix 反查 | 输出 `(igroup,itile_m,itile_n)` |
|---:|---:|---:|---|---|
| 0 | 0 | 0 | `cu[0]=0 <= 0 < cu[1]=1` | `(0,0,0)` |
| 3 | 3 | 0 | `cu[3]=3 <= 3 < cu[4]=5` | `(3,0,0)` |
| 4 | 4 | 0 | `cu[3]=3 <= 4 < cu[4]=5` | `(3,1,0)` |
| 9 | 9 | 0 | `cu[6]=9 <= 9 < cu[7]=12` | `(6,0,0)` |
| 14 | 14 | 0 | `cu[7]=12 <= 14 < cu[8]=15` | `(7,2,0)` |
| 15 | 0 | 1 | `cu[0]=0 <= 0 < cu[1]=1` | `(0,0,1)` |
| 479 | 14 | 31 | `cu[7]=12 <= 14 < cu[8]=15` | `(7,2,31)` |

这些结果与日志一致：`temp/run.log:94` 是 `iblock=0 -> (0,0,0)`，`temp/run.log:95` 是 `iblock=2 -> (2,0,0)`，`temp/run.log:99` 是 `iblock=4 -> (3,1,0)`，`temp/run.log:97` 是 `iblock=130 -> (6,1,8)`。例如 `iblock=130` 时，`130 % 15 = 10`、`130 / 15 = 8`，全局 M rank 10 落在 group 6 的区间 `[9,12)`，所以局部 M tile 是 1。

### 9.7 它如何服务于 TMA load 和 WGMMA

policy 2 下，load warpgroup 和 math warpgroup 都执行相同的映射：load 侧在 `src/group_gemm/kernels.cuh:390-413`（函数 `group_gemm_fp8_kernel`），math 侧在 `src/group_gemm/kernels.cuh:469-490`（函数 `group_gemm_fp8_kernel`）。这样同一个 CTA 的 producer/consumer 对同一个 `iblock` 得到相同的 `(igroup,itile_m,itile_n)`，随后：

- load 侧用 `itile_m` 选择 X 的 group-local M tile、用 `itile_n` 和 `igroup` 选择 W tile，见 `src/group_gemm/kernels.cuh:426-430`（函数 `group_gemm_fp8_kernel`）；
- math 侧等待对应 stage barrier 后消费 shared A/B 做 WGMMA，见 `src/group_gemm/kernels.cuh:501-518`（函数 `group_gemm_fp8_kernel`）；
- epilogue 的 TMA store 也用同一组坐标：`src/group_gemm/kernels.cuh:566-569`（函数 `group_gemm_fp8_kernel`）把 `itile_m` 和 `itile_n` 映射到输出的 M/N tile，并写回该 `igroup` 的 Y 区域；
- 当 `itile_n >= num_tile_n` 时，调用者在 `src/group_gemm/kernels.cuh:410-412` 和 `src/group_gemm/kernels.cuh:487-489`（函数 `group_gemm_fp8_kernel`）退出循环。因为 `iblock` 按 `gridDim.x` 递增，最后一轮可能超出有效任务数，正是这个边界检查负责终止。

因此，`get_next_tile_vert` 的业务本质是：在 ragged group GEMM 中，把 persistent grid 的线性 work index 映射成正确的 group、该 group 内的 M tile、以及 N tile；它利用 prefix sum 处理不同 group 的 sequence length，并用二分查找避免逐 group 线性扫描。
