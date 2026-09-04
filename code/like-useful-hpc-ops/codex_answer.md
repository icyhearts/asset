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

# 3. `src/group_gemm/entry.cc` 完善 tmas 调试打印 + dtype / device 是否为 enum 的说明

## 3.1 修改内容（函数 `hpc::group_gemm::group_gemm_fp8_entry`，`src/group_gemm/entry.cc:17`）

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

## 3.2 dtype 是 enum 类型吗？

**dtype 的“值”本质上是 enum，但 `Tensor::dtype()` 返回的不是 enum 本身，而是一个封装结构体。**

- `at::Tensor::dtype()` 的返回类型是 `caffe2::TypeMeta`（`torch_include/ATen/core/TensorBase.h:444`，函数 `TensorBase::dtype`），它是一个 `class C10_API TypeMeta final`（`torch_include/c10/util/typeid.h:319`）。
- `TypeMeta` 内部只保存一个 `uint16_t index_`，这个 index 就是 `c10::ScalarType` enum 的整数值；`toScalarType()` 把它转回 enum（`torch_include/c10/util/typeid.h:480`）。
- 真正的 dtype enum 是 `c10::ScalarType`：`enum class ScalarType : int8_t`（`torch_include/torch/headeronly/core/ScalarType.h:258`），所有枚举值由宏 `AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS` 展开生成（`torch_include/torch/headeronly/core/ScalarType.h:103`、`:260`），包括 `Float8_e4m3fn`、`Int32`、`Float`、`BFloat16` 等。
- 把 enum 转成名字打印用 `c10::toString(ScalarType)`（`torch_include/torch/headeronly/core/ScalarType.h:320`，函数 `c10::toString`）。
- `src/group_gemm/entry.cc:29` 的 `x.dtype() == torch::kFloat8_e4m3fn` 能直接比较，是因为 `torch_include/c10/core/ScalarTypeToTypeMeta.h:42-56` 提供了 `operator==(TypeMeta, ScalarType)`（以及反向）的自由函数。

> 小结：tensor dtype 的“语义类型”是 `c10::ScalarType` —— 它是 `enum class`；`Tensor::dtype()` 通过 `caffe2::TypeMeta` 这个 wrapper 返回，需 `.toScalarType()` 才拿到 enum。

## 3.3 device 是 enum 类型吗？

**device 本身不是 enum，它是一个结构体；结构体内部保存的“设备类型”才是 enum。**

- `at::Tensor::device()` 返回 `c10::Device`（`torch_include/ATen/core/TensorBase.h:449`，函数 `TensorBase::device`）。
- `c10::Device` 是 `struct C10_API Device final`（`torch_include/c10/core/Device.h:32`），它包含两个成员：`DeviceType type_`（`torch_include/c10/core/Device.h:171`）和 `DeviceIndex index_`（即设备序号，如 0）。
- 设备类型 enum 是 `c10::DeviceType`：`enum class DeviceType : int8_t`（`torch_include/torch/headeronly/core/DeviceType.h:35`），取值 `CPU=0`、`CUDA=1`、`HIP=6`、`MPS=13` 等。
- 打印设备用 `Device::str()`（`torch_include/c10/core/Device.h:168`），返回形如 `cuda:0` 的字符串。

> 小结：device 的“语义类型”里，`DeviceType` 是 enum class；但 `Tensor::device()` 返回的是 `c10::Device` 结构体（enum 成员 + 设备序号）。

## 3.4 tensor dtype / device enum 定义位置汇总

| 类型 | 定义位置 | 说明 |
| --- | --- | --- |
| `c10::ScalarType`（dtype enum） | `torch_include/torch/headeronly/core/ScalarType.h:258`（`enum class ScalarType : int8_t`） | 枚举值由宏 `AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS` 在 `:103`、`:260` 展开 |
| `c10::kFloat8_e4m3fn` 等常量 | `torch_include/c10/core/ScalarType.h:38-42`（宏 `DEFINE_CONSTANT`） | `constexpr ScalarType`；`torch::` 通过 `using namespace at;` 可见（`torch_include/torch/csrc/api/include/torch/types.h:37`） |
| `caffe2::TypeMeta`（`Tensor::dtype()` 返回类型） | `torch_include/c10/util/typeid.h:319`；`toScalarType()` 在 `:480` | wrapper，内含 ScalarType index |
| `c10::DeviceType`（device 类型 enum） | `torch_include/torch/headeronly/core/DeviceType.h:35` | `CPU=0, CUDA=1, ...`；由 `torch_include/c10/core/DeviceType.h:8` 引入 |
| `c10::Device`（`Tensor::device()` 返回类型） | `torch_include/c10/core/Device.h:32`；`DeviceType type_` 在 `:171`，`str()` 在 `:168` | struct：enum 成员 + 设备序号 |
| TypeMeta<->ScalarType 比较桥接 | `torch_include/c10/core/ScalarTypeToTypeMeta.h:42-56` | 使 `entry.cc:29` 的 `x.dtype() == torch::kFloat8_e4m3fn` 可编译 |
| dtype 转字符串 | `torch_include/torch/headeronly/core/ScalarType.h:320`（`c10::toString`） | 打印用 |

## 3.5 运行效果

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

# 4. 以本次运行为例详解 update_grouped_tma

本节以 temp/run.log 记录的 tests/test_group_gemm_pertensor_like.py 一次运行为例，追踪 update_grouped_tma 从 Python 到 CUDA kernel 的完整调用链。代码引用均使用相对代码库根目录的路径 + 行号 + 函数名。

## 4.1 本次运行的数据与形状

测试入口 tests/test_group_gemm_pertensor_like.py:46（函数 test_group_gemm_pertensor_fp8），实际参数在 tests/test_group_gemm_pertensor_like.py:75-80：

    num_group=8; actual_m=256; n=4096; k=7168

- seqlens = (1+torch.arange(8))*16，得 [16,32,48,64,80,96,112,128]（tests/test_group_gemm_pertensor_like.py:49）
- cu_seqlens = [0,16,48,96,160,240,336,448,576]（tests/test_group_gemm_pertensor_like.py:61）
- total_seq=576，所以 mean_seq=576/8=72（tests/test_group_gemm_pertensor_like.py:51-53）作为 num_seq_per_group_avg 传入
- x.shape=[576,7168]、w.shape=[8,4096,7168]、结果 y=[576,4096]（BF16）

以上与 temp/run.log:1-4 一致；temp/run.log:13-17 是本仓库上一轮加的 [LIKE_DEBUG] 打印，说明 tma buffer dtype=Float8_e4m3fn、shape=[16,128]，与 entry.cc 的 torch::empty({num_group*2,128}, options) 对应。

## 4.2 分派：num_seq_per_group_avg=72 选中哪一组模板

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

## 4.3 td_xy 与 tma_xy 的来历

src/group_gemm/group_gemm_pertensor_fp8.cu:176（launch_group_gemm_fp8）把 tmas_ptr 位转成 cute::TmaDescriptor 数组（每个 group 2 个：1 个 X + 1 个 Y）：

    auto *tma_xy = static_cast<cute::TmaDescriptor*>(tmas_ptr);                             // :176
    vec_t<cute::TmaDescriptor,2> td_xy{ *tma_x.get_tma_descriptor(), *tma_y.get_tma_descriptor() };  // :180-183

td_xy 是"干净模板"：X 的基址还指向 x_ptr 起点、Y 指向 y_ptr 起点，shape 是整张大 [m,k]/[n,m]。tma_xy 则是 entry.cc:59 分配的字节buffer（torch::empty({num_*2,128})，每个 TmaDescriptor 占 128 字节），update_grouped_tma 会往里面逐 group 写入改写后的描述符。

注意：vec_t 定义在 src/utils/utils.cuh:30（struct vec_t），是固定长度小数组包装。

## 4.4 启动网格：cudaLaunchKernelEx 与 PDL

launch_group_gemm_fp8 用 cudaLaunchKernelEx 且启用程序化流串行化：

1. cudaLaunchAttributeProgrammaticStreamSerialization（src/group_gemm/group_gemm_pertensor_fp8.cu:190-193）声明这个 kernel 允许后续 kernel 用 PDL 直接依赖，省去流级同步。
2. 本测试未传 task_map workspace，task_map_ptr 为 null，走 kAssignTask=false 分支：gridDim = num_group+1 = 9（src/group_gemm/group_gemm_pertensor_fp8.cu:213-214）；blockDim = kThreadPerBlock = 32（src/group_gemm/group_gemm_pertensor_fp8.cu:186）。
3. 实例化并启动 kernels::update_grouped_tma<Tin,Tout,TmapX,TmaY,kTileM=48,kGroupPerThread=8,kThreadPerBlock=32,kAssignTask=false,kUsePDL=true>（src/group_gemm/group_gemm_pertensor_fp8.cu:205-211）。

说明：num_group+1=9 个 block，block 0..7 每个负责 1 个 group 的 2 个描述符（X、Y），block 8（=num_group）专门算 tiles/cu_tiles。若传 task_map，会走 kAssignTask=true 分支，gridDim=2*num_group+1（src/group_gemm/group_gemm_pertensor_fp8.cu:201），多出的 num_group 个 block 用来生成 task_map；本测试不走。

## 4.5 kernel 内部分工

进入 update_grouped_tma（src/group_gemm/kernels.cuh:65，函数 update_grouped_tma）：

- int igroup = blockIdx.x（src/group_gemm/kernels.cuh:73）
- 若 kUsePDL 先 cudaGridDependencySynchronize()（src/group_gemm/kernels.cuh:75-77），确保上一个 producer 完成
- 聚合 block（igroup==num_group，即 block 8）走 if 分支（见 4.6 的 cu_tiles）；其余 8 个 block 走 else 分支（src/group_gemm/kernels.cuh:172，函数 `update_grouped_tma`）更新描述符

### 4.5.1 每 group 的 X/Y 描述符改写（src/group_gemm/kernels.cuh:172-207）

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

### 4.5.2 聚合 block 计算 cu_tiles（src/group_gemm/kernels.cuh:143-170）

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
## 4.6 主 GEMM 如何消费这些描述符

描述符就绪：`update_grouped_tma` 结尾用 PDL 的 `cudaTriggerProgrammaticLaunchCompletion()`（`src/group_gemm/kernels.cuh:215-217`，函数 `update_grouped_tma`）发出完成信号；随后启动的 `group_gemm_fp8_kernel`（`src/group_gemm/kernels.cuh:220-226`，函数 `group_gemm_fp8_kernel`）在开头 `cudaGridDependencySynchronize()`（`src/group_gemm/kernels.cuh:302-304`，函数 `group_gemm_fp8_kernel`）等描述符更新 kernel 到达 PDL 依赖点。

kernel 内按 task loop 策略取每个 tile 的 (igroup, itile_m, itile_n)（本测试 kTaskLoopPolicy=2，见 src/group_gemm/kernels.cuh:314）：
- load WG 用 td_x = td_xy + igroup*2 做 tma_a.with(td_x, ...)（src/group_gemm/kernels.cuh:372-380）
- store 用 td_y = td_xy + igroup*2+1 写回（src/group_gemm/kernels.cuh:519 处 cute::copy(tma_d.with(td_y),...)）

这些 `td_xy`（也就是 `update_grouped_tma` 写入的 `tma_xy` buffer）正是描述符数组。当前只有 policy 0 在 `src/group_gemm/kernels.cuh:306-319`（函数 `group_gemm_fp8_kernel`）显式调用 `tma_descriptor_fence_acquire`；本次 policy 2 路径没有该调用，具体风险和建议见第 14.8 节。其底层实现位于 `3rd/cutlass/include/cute/arch/copy_sm90_desc.hpp:457-470`（函数 `tma_descriptor_fence_acquire`）。

## 4.7 小结

对本次 temp/run.log 的调用，update_grouped_tma（src/group_gemm/kernels.cuh:65）：
1. 以 num_group+1=9 个 block 启动（PDL 程序化单响应器）；block 0..7 每个负责 1 个 group 的 X、Y 描述符，block 8 算 tiles/cu_tiles。
2. thread 0 / 1 分别用 update_tma_descriptor_gtensor 把 X / Y 描述符的全局基址改成本 group 的 x/y+cu_seqlen*stride、形状改成本 group 的 [num_seq,...]。
3. 用 tma_descriptor_cp_fence_release 把改好的描述符 cp 到 tma_xy + igroup*2 [ +1 ] 并 release。
4. block num_group 计算 tiles/cu_tiles（M tile 前缀和）。
5. 末尾 cudaTriggerProgrammaticLaunchCompletion；主 GEMM 凭 cudaGridDependencySynchronize 拿到更新后的描述符开始计算。

---

# 5. 详解 elect_one_sync（cutlass 集群/warp 选举）

本节讲解 3rd/cutlass/include/cute/arch/cluster_sm90.hpp 中的 elect_one_sync 函数。该函数在 hpc-ops 的 group_gemm 实现里被多处使用（例如更新 TMA 描述符后让"单个线程"提交 commit/wait，见 src/group_gemm/kernels.cuh:202），其作用是"在一个 warp 内选出一个代表线程"。

## 5.1 函数签名与语义

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

## 5.2 分支依赖：CUTE_ARCH_ELECT_ONE_SM90_ENABLED

该宏在 3rd/cutlass/include/cute/arch/cluster_sm90.hpp:42-44 定义，要求：

- __CUDA_ARCH__ >= 900（即 sm_90 / Hopper 及以上，使用该宏说明需要现代 TMA/集群指令）
- __CUDACC_VER_MAJOR__ >= 12（CUDA 12.x 编译器，因为 elect.sync 及 elect semantics 需要新 ptxas）

## 5.3 SM90 路径下的 PTX 内联汇编

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

## 5.4 在 hpc-ops 中的用法示例

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

## 5.5 与"单纯看 threadIdx.x==0"的区别

- elect.sync 是硬件指令，支持对任意"由并发执行到该点的 threads"做选举；而 (threadIdx.x%32)==0 只是"固定 lane 0 为 leader"。
- 在带有分支发散、或某条 lane 提前退出（mask 中有 inactive lane）时，elect.sync 仍保证在参与 mask 线程里恰好选出一个。
- 但 elect_one_sync 每次调用都重新选举，**不保证**每次都选同一条 lane；它只保证"恰恰选出一个代表"，适合用它的地方都是"只需一个线程来执行一次副作用"（arrive/commit/wait）。

## 5.6 相关兄弟函数（cluster_sm90.hpp）

- cluster_arrive_relaxed / cluster_arrive / cluster_wait / cluster_sync（cluster_sm90.hpp:48-83）：基于 barrier.cluster 的集群同步。
- cluster_grid_dims / cluster_id_in_grid / block_id_in_cluster / cluster_shape / block_rank_in_cluster（cluster_sm90.hpp:86-163）：读取 nclusterid/clusterid/cluster_ctaid/cluster_nctaid/cluster_ctarank 的内建寄存器，用于确定 cluster 的几何形状与当前 block 在 cluster 中的身份。
- set_block_rank（cluster_sm90.hpp:166-177）：用 mapa.shared::cluster 把"本地共享地址+目标 block rank"映射成集群内可访问的远程共享地址，是集群间 SMEM 通信的基础。
- store_shared_remote（cluster_sm90.hpp:234-244）：组合 set_block_rank + st.async.shared::cluster 做远程共享内存写（带 mbarrier 完成计数）。

## 5.7 参考规范

elect.sync 指令在 PTX ISA（sm_90 引入 elect 这一谓词选举指令）有明确定义：

    elect.sync d | p{, membermask};

- p 为真只对"满足了该并发执行束且被选中的那条 lane"为真；
- d 输出被选中 lane 在 membermask 中的索引（0 .. membermask 内被选中的序号）。

CUTLASS/CuTe 的 elect_one_sync 即 (elect.sync 的 pred + laneid) 中只保留了 pred，elect_one_leader_sync 额外保留 laneid。

---

# 6. `src/group_gemm/kernels.cuh:203-204` 的 `commit + wait`

## 6.1 结论

在当前的 `update_grouped_tma` kernel 中，203--204 行之前**没有**发起 X/Y tensor 的 TMA copy。这里的

```cpp
cute::tma_desc_commit_group();
cute::tma_desc_wait_group();
```

是从 CUTLASS 的“动态重写一个可能已被 TMA 使用的 descriptor”流程带来的通用保护。在真正有此前发起的、由同一线程管理的 bulk async 请求时，它保证 TMA engine 已经读完 tensormap，才允许用 206 行覆盖该全局 descriptor；但对这个独立的 updater kernel 而言，它会提交一个空 group 后立刻返回，功能上是冗余的防御性代码，不是在等待 189/196 行的 descriptor 修改完成。

更精确地说，它也不能等待别的线程、别的 warp 或别的 kernel 的 TMA 请求：PTX 将 bulk async-group 定义为 *per-thread*，`commit_group` 只收集“执行该指令的线程”此前发起、尚未提交且采用 `bulk_group` completion 的请求。

## 6.2 203--204 行实际发出的指令

- `cute::tma_desc_commit_group()` 在 `3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:1235` 中直接发出 `cp.async.bulk.commit_group`。
- `cute::tma_desc_wait_group()` 在同文件 `:1262` 中发出 `cp.async.bulk.wait_group.read 0`。
- `read 0` 的含义是等待该线程此前所有已提交 bulk group 至少完成对 tensormap 和源地址的读取；这正是“之后可替换 descriptor”的关键，而不是普通的 CTA/warp barrier。
- 若此前没有可提交请求，`commit_group` 按 PTX 规范创建空 group；本处随后的 `wait_group.read 0` 因而没有可等的 TMA 工作。循环运行两次也不会增加当前路径的正确性。

这也是为什么一般动态 descriptor 重用场景需要二者连用：先 `commit` 把未提交请求纳入可等待的 group，之后 `wait_group.read 0` 让 descriptor 的异步读取阶段结束。CUTLASS 在动态 tensormap 更新路径中使用了同一惯用法，并明确说明“没有未提交指令时会得到 empty bulk async-group”（例如 `3rd/cutlass/include/cutlass/epilogue/collective/sm100_epilogue_array_tma_warpspecialized.hpp:1564`）。

## 6.3 本函数在 203 行之前实际做了什么

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

# 7. 为什么 host 看起来是 policy 2，而 kernel 打印 policy 1

## 7.1 直接原因：`kTaskLoopPolicy` 被声明成了 `bool`

`group_gemm_fp8_kernel` 的模板参数明确是 `int`：`src/group_gemm/kernels.cuh:215-221`（函数 `group_gemm_fp8_kernel`）。但 host 端 `launch_group_gemm_fp8` 的六个 dispatch 分支都写成了 `constexpr bool kTaskLoopPolicy`：

- PDL 分支：`src/group_gemm/group_gemm_pertensor_fp8.cu:137`、`:152`、`:165`（函数 `launch_group_gemm_fp8`）；
- 非 PDL 分支：同文件 `:182`、`:196`、`:208`（函数 `launch_group_gemm_fp8`）。

因此第三个分支中的

```cpp
constexpr bool kTaskLoopPolicy = 2;
```

在声明时就发生 C++ 布尔转换：`2 != 0`，所以变量的实际值是 `true`，数值表现为 `1`。随后它在 `src/group_gemm/group_gemm_pertensor_fp8.cu:169-171`（函数 `launch_group_gemm_fp8`）作为非类型模板实参传给 `group_gemm_fp8_kernel`；`true` 再转换为该模板要求的 `int` 后仍然是 `1`。所以 device printf 打印 `kTaskLoopPolicy=1` 是编译期实例化结果，并不是运行时把 `2` 改成了 `1`。

这不是 `printf` 的格式问题，也不是 host/device 变量不同步，而是 `constexpr bool` 的类型错误。策略 0 和策略 1 恰好不会暴露问题；只有策略 2 被压成了策略 1。

## 7.2 为什么 `shm_seq:36` 仍然能打印

host 的 `shm_seq` 打印位于第三个 `else` 分支的局部代码中。该分支的选择条件是运行时条件：`task_map_ptr == nullptr` 且 `k > 1024`、`n > 1024`，见 `src/group_gemm/group_gemm_pertensor_fp8.cu:136-176`（函数 `launch_group_gemm_fp8`）。进入这个分支只说明选择了“原本意图对应 policy 2 的代码块”，并不说明其中 `constexpr bool kTaskLoopPolicy` 的值是整数 2。

本次日志也逐项吻合：

- `temp/run.log:10-14` 显示 `n=4096`、`k=7168`；因此 `k <= 1024 || n <= 1024` 为假；
- `temp/run.log:73` 显示 `task_map_ptr(nil)`，所以没有进入 task-map 的 policy 0 分支；
- `temp/run.log:75` 的 `shm_seq:36` 等于 `sizeof(int) * (num_group + 1)`，这里 `num_group=8`，即 `4 * 9 = 36`；这只是该分支的 shared-memory 计算结果。

另外，`src/group_gemm/entry.cc:61-68`（函数 `group_gemm_fp8_entry`）只有在 `num_seq_per_group_avg <= 8` 且提供 task-map workspace 时才设置 `task_map_ptr`。本次平均 sequence length 为 72（日志中的 `16,32,...,128`），因此 `task_map_ptr` 为 null 是预期行为。

## 7.3 kernel 实际编译和执行的是 policy 1

在 `group_gemm_fp8_kernel` 中，policy 1 和 policy 2 是 `if constexpr` 的不同编译路径：

- policy 1 在 `src/group_gemm/kernels.cuh:310-313`（函数 `group_gemm_fp8_kernel`）读取 `tiles_ptr`；随后在 `:357-362` 调用 `get_next_tile_horizon`；
- policy 2 在 `src/group_gemm/kernels.cuh:314-318`（函数 `group_gemm_fp8_kernel`）读取 `cu_tiles_ptr` 并设置 `total_m`；随后在 `:363-367` 调用 `get_next_tile_vert`。

由于实际模板值是 1，policy 2 分支在编译期被丢弃，kernel 走的是 horizon 路径。日志中 device printf 的 `total_m:0` 也与此一致：只有 policy 2 的 `src/group_gemm/kernels.cuh:315`（函数 `group_gemm_fp8_kernel`）会给 `total_m` 赋值；policy 1 不会赋值，因此保留初始化值 0。

## 7.4 还要注意 `use_pdl` 的覆盖

`group_gemm_fp8_entry` 在 `src/group_gemm/entry.cc:84-86`（函数 `group_gemm_fp8_entry`）把 `use_pdl=false` 传给 `group_gemm_fp8_async`，但 `src/group_gemm/group_gemm_pertensor_fp8.cu:224-246`（函数 `group_gemm_fp8_async`）第 238 行又无条件执行 `use_pdl = true`。所以本次运行实际走 `launch_group_gemm_fp8` 的 PDL 分支，具体是 `src/group_gemm/group_gemm_pertensor_fp8.cu:121-176`（函数 `launch_group_gemm_fp8`）中的第三个 `else`。

## 7.5 修复建议

将 pertensor host dispatch 中的六处声明改成 `constexpr int kTaskLoopPolicy`：

```text
src/group_gemm/group_gemm_pertensor_fp8.cu:137,152,165,182,196,208
```

至少本次命中的 `:165` 必须改为 `constexpr int kTaskLoopPolicy = 2;`。重新编译后，传给 `group_gemm_fp8_kernel` 的模板实参才会是整数 2，device printf 才会打印 2，并且 `if constexpr (kTaskLoopPolicy == 2)` 会真正选择 `get_next_tile_vert`。同样的 `constexpr bool` 三态策略问题也存在 blockwise host dispatch，但不影响本次 pertensor 日志的根因。

注：当前工作树的 tracked 源码中未保留问题描述里的两条临时 `printf`，但 `temp/run.log` 显然来自加入这些插桩后的构建；这不会改变上述模板类型转换结论。

---

# 8. `get_next_tile_vert`：policy 2 如何把线性任务映射回 group tile

## 8.1 本次日志确实执行了 policy 2

当前 `temp/run.log:94-570` 大量打印 `kTaskLoopPolicy:2`，`temp/run.log:648-651` 也再次打印 policy 2；因此这次运行确实进入了 `get_next_tile_vert`，不是只有 host 端进入了 policy-2 的 dispatch 分支。当前源码在 `src/group_gemm/group_gemm_pertensor_fp8.cu:309`（函数 `launch_group_gemm_fp8`）使用 `constexpr int kTaskLoopPolicy = 2`，并在 `src/group_gemm/group_gemm_pertensor_fp8.cu:320-327`（函数 `launch_group_gemm_fp8`）将它作为模板实参传给 `group_gemm_fp8_kernel`。此前第 8 节记录的是类型修复前的历史状态；本节以当前源码和当前日志为准。

## 8.2 函数签名和业务对象

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

## 8.3 输入参数和输出参数

`cu_tiles_ptr`、`iblock`、`num_group`、`total_m` 的含义如下：

1. `cu_tiles_ptr` 是长度为 `num_group + 1` 的 M-tile 排布表。它不是每组的 tile 数，而是 exclusive prefix sum：`cu_tiles_ptr[g]` 是 group `g` 在“所有 group 的 M tiles 拼接数组”中的起始 offset，`cu_tiles_ptr[g+1]` 是结束 offset。该数组由 `src/group_gemm/kernels.cuh:143-170`（函数 `update_grouped_tma`）生成：每组 tile 数在 `src/group_gemm/kernels.cuh:149-150`（函数 `update_grouped_tma`）计算并写入 `tiles_ptr`，BlockScan exclusive sum 在 `src/group_gemm/kernels.cuh:156-159`（函数 `update_grouped_tma`）执行，前缀结果在 `src/group_gemm/kernels.cuh:161-169`（函数 `update_grouped_tma`）写入 `cu_tiles_ptr`，其中 `cu_tiles_ptr[num_group]` 是总 M-tile 数。

2. `iblock` 是当前 CTA 的线性任务号，不是 group id。load warpgroup 在 `src/group_gemm/kernels.cuh:381`（函数 `group_gemm_fp8_kernel`）初始化为 `blockIdx.x`，math warpgroup 在 `src/group_gemm/kernels.cuh:463`（函数 `group_gemm_fp8_kernel`）也初始化为 `blockIdx.x`；每完成一次任务后分别在 `src/group_gemm/kernels.cuh:417` 和 `src/group_gemm/kernels.cuh:492`（函数 `group_gemm_fp8_kernel`）加上 `gridDim.x`，让一个 persistent CTA 继续领取自己的后续任务。本次 host 侧在 `src/group_gemm/group_gemm_pertensor_fp8.cu:258-263`（函数 `launch_group_gemm_fp8`）将 `gridDim` 设为 `num_sm=132`，所以一个 CTA 的任务序列形如 `blockIdx.x, blockIdx.x+132, blockIdx.x+264, ...`。

3. `num_group` 是 group 数，同时也是 `cu_tiles_ptr` 的最后一个合法下标。二分查找在 `src/group_gemm/kernels.cuh:48-57`（函数 `get_next_tile_vert`）的 `[0, num_group]` 范围内查找 prefix boundary。

4. `total_m` 是所有 group 的 M tiles 总数，即 `cu_tiles_ptr[num_group]`。policy 2 在 `src/group_gemm/kernels.cuh:357-361`（函数 `group_gemm_fp8_kernel`）把该值读入 `total_m` 并把 prefix table 搬到 shared memory。它必须大于 0，因为 `get_next_tile_vert` 在 `src/group_gemm/kernels.cuh:45-46` 对它做取模和整除。

5. `igroup`、`itile_m`、`itile_n` 是引用输出参数；函数不会依赖调用前的 `igroup` 值，而是在 `src/group_gemm/kernels.cuh:59-60`（函数 `get_next_tile_vert`）覆盖写入。返回后，`igroup` 是 group 下标，`itile_m` 是 group-local M tile 下标，`itile_n` 是 N tile 下标。

## 8.4 第一阶段：把 `iblock` 拆成 N tile 和全局 M tile

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

## 8.5 第二阶段：用 prefix sum 二分反查 group 和局部 M tile

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

## 8.6 用本次数据走一遍

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

## 8.7 它如何服务于 TMA load 和 WGMMA

policy 2 下，load warpgroup 和 math warpgroup 都执行相同的映射：load 侧在 `src/group_gemm/kernels.cuh:390-413`（函数 `group_gemm_fp8_kernel`），math 侧在 `src/group_gemm/kernels.cuh:469-490`（函数 `group_gemm_fp8_kernel`）。这样同一个 CTA 的 producer/consumer 对同一个 `iblock` 得到相同的 `(igroup,itile_m,itile_n)`，随后：

- load 侧用 `itile_m` 选择 X 的 group-local M tile、用 `itile_n` 和 `igroup` 选择 W tile，见 `src/group_gemm/kernels.cuh:426-430`（函数 `group_gemm_fp8_kernel`）；
- math 侧等待对应 stage barrier 后消费 shared A/B 做 WGMMA，见 `src/group_gemm/kernels.cuh:501-518`（函数 `group_gemm_fp8_kernel`）；
- epilogue 的 TMA store 也用同一组坐标：`src/group_gemm/kernels.cuh:566-569`（函数 `group_gemm_fp8_kernel`）把 `itile_m` 和 `itile_n` 映射到输出的 M/N tile，并写回该 `igroup` 的 Y 区域；
- 当 `itile_n >= num_tile_n` 时，调用者在 `src/group_gemm/kernels.cuh:410-412` 和 `src/group_gemm/kernels.cuh:487-489`（函数 `group_gemm_fp8_kernel`）退出循环。因为 `iblock` 按 `gridDim.x` 递增，最后一轮可能超出有效任务数，正是这个边界检查负责终止。

因此，`get_next_tile_vert` 的业务本质是：在 ragged group GEMM 中，把 persistent grid 的线性 work index 映射成正确的 group、该 group 内的 M tile、以及 N tile；它利用 prefix sum 处理不同 group 的 sequence length，并用二分查找避免逐 group 线性扫描。

# 9. `tma_a.with`/`tma_b.with` 重载与 barrier 顺序

## 9.1 两处调用分别命中了什么重载

先区分两层 `with`：源码中看到的成员函数是 `cute::Copy_Atom::with`，它只是一个可变参数转发器；真正决定参数含义的是 `Copy_Traits<SM90_TMA_LOAD, ...>::with`。

1. `src/group_gemm/kernels.cuh:437-438`（函数 `group_gemm_fp8_kernel`）中的：

   ```cpp
   tma_a.with(td_x, readable[ismem_write])
   ```

   先调用 `3rd/cutlass/include/cute/atom/copy_atom.hpp:77-83`（函数 `cute::Copy_Atom::with`），再转发到 `3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:135-145`（函数 `Copy_Traits<SM90_TMA_LOAD>::with`）这个重载：

   ```cpp
   with(TmaDescriptor const* new_tma_desc,
        uint64_t& tma_mbar,
        uint16_t const& multicast_mask = 0,
        CacheHintSm90 const& cache_hint = EVICT_NORMAL)
   ```

   所以两个显式参数的含义是：用 `td_x` 替换本次 TMA load 使用的 tensor-map/descriptor，并把 `readable[ismem_write]` 作为完成通知 barrier。`td_x` 的 `TmaDescriptor*` 可以转换为该重载需要的 `TmaDescriptor const*`。

2. `src/group_gemm/kernels.cuh:440-441`（函数 `group_gemm_fp8_kernel`）中的：

   ```cpp
   tma_b.with(readable[ismem_write])
   ```

   同样先经过 `3rd/cutlass/include/cute/atom/copy_atom.hpp:77-83`（函数 `cute::Copy_Atom::with`），但实际匹配的是 `3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:124-133`（函数 `Copy_Traits<SM90_TMA_LOAD>::with`）这个重载：

   ```cpp
   with(uint64_t& tma_mbar,
        uint16_t const& multicast_mask = 0,
        CacheHintSm90 const& cache_hint = EVICT_NORMAL)
   ```

   这里显式传入的唯一参数就是 barrier；descriptor 继续使用 `tma_b` 对象内部保存的 descriptor。两个重载后面的 multicast mask 和 cache hint 都有默认值，因此“2 个参数”和“1 个参数”说的是必需参数数量，不是该 API 能接受的总参数数量。

需要排除另一个同名函数：`3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:164-169`（函数 `Copy_Traits<SM90_TMA_LOAD>::with`）的单参数版本要求的是 `TmaDescriptor const*`，只返回一个仍未绑定 barrier 的 descriptor 版本；`readable[ismem_write]` 是 `uint64_t&`，因此 `tma_b.with(readable[ismem_write])` 不会匹配这个重载。

`with` 本身不会搬运数据，只是返回一个已经绑定了 descriptor/barrier 的可执行 `Copy_Atom`。真正发出 TMA 指令的是后面的 `cute::copy`；`3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:103-131`（函数 `SM90_TMA_LOAD_2D::copy`）最终生成 `cp.async.bulk.tensor.2d...mbarrier::complete_tx::bytes`。

## 9.2 为什么 A 要替换 descriptor，而 B 不需要

`src/group_gemm/config.h:90-95`（函数 `GroupGEMMFp8Config::get_tma`）为 X、W、Y 各创建一个 TMA atom；但 group GEMM kernel 的参数只显式传入了 W TMA，见 `src/group_gemm/group_gemm_pertensor_fp8.cu:286-327`（函数 `launch_group_gemm_fp8`）。X/Y 的每个 group descriptor 放在 `td_xy` 中，由更新 kernel 动态生成并发布。

- X 是 ragged 输入。`src/group_gemm/kernels.cuh:180-195`（函数 `update_grouped_tma`）按 `igroup` 计算 `cu_seqlens_ptr[igroup]` 对应的基地址和 `num_seq`，再修改 X descriptor；`src/group_gemm/kernels.cuh:204-212`（函数 `update_grouped_tma`）通过 `tma_descriptor_cp_fence_release` 发布修改后的 descriptor。因此 load warp 在 `src/group_gemm/kernels.cuh:430-438`（函数 `group_gemm_fp8_kernel`）必须把 `td_x = td_xy + igroup * 2` 传给 `.with`，否则会继续使用模板 descriptor，而不是当前 group 的地址/shape。
- W 的布局是一个固定的三维 tensor `(n, k, num_group)`，定义见 `src/group_gemm/group_gemm_pertensor_fp8.cu:37-40`（函数 `launch_group_gemm_fp8`）。`src/group_gemm/kernels.cuh:440-441`（函数 `group_gemm_fp8_kernel`）中的 `tBg(_, itile_n, itile_k, igroup)` 已经把 group 作为 TMA 坐标传入；因此不需要为每个 group 替换 W descriptor，只需给这次异步 load 绑定 barrier。

换句话说，A 的两个参数分别是“本 group 的动态 descriptor + 完成 barrier”，B 的一个参数是“完成 barrier”；不是 A/B 的 TMA 指令种类不同。

## 9.3 `set_barrier_transaction_bytes` 做了什么

在 `src/group_gemm/kernels.cuh:325-329`（函数 `group_gemm_fp8_kernel`）中，每个 `readable` barrier 以 arrival count 1 初始化；消费者在 `src/group_gemm/kernels.cuh:510-514`（函数 `group_gemm_fp8_kernel`）调用 `wait_barrier` 等待该阶段完成。

`set_barrier_transaction_bytes` 不是单纯写一个 C++ 变量，也不是等待 TMA。它在 `3rd/cutlass/include/cute/arch/copy_sm90_desc.hpp:75-87`（函数 `cute::set_barrier_transaction_bytes`）直接发出：

```text
mbarrier.arrive.expect_tx.shared::cta.b64 _, [barrier], bytes
```

该指令同时做两件事：增加当前 phase 的 expected transaction bytes，并消耗一个 pending arrival。TMA load 完成时，`cp.async.bulk.tensor...complete_tx::bytes` 再按实际完成字节数扣减 tx-count；只有 arrival count 和 tx-count 都归零，`wait_barrier` 才会通过。

本例中的 `src/group_gemm/kernels.cuh:377-378`（函数 `group_gemm_fp8_kernel`）把 `kTransactionBytes` 计算为 A tile 和 B tile 的总字节数，正好对应 `src/group_gemm/kernels.cuh:437-443`（函数 `group_gemm_fp8_kernel`）提交的两次 TMA load。也就是说，一个 barrier 跟踪两次 copy 的合计，而不是每次 copy 各自跟踪一个 barrier。

## 9.4 为什么当前代码可以先 copy、再设置 expected bytes

当前代码的实际顺序是：

```text
wait writable
提交 TMA A（异步）
提交 TMA B（异步）
mbarrier.arrive.expect_tx(total_bytes)
```

这里“提交”非常重要：`cute::copy` 只把 `cp.async.bulk.tensor` 指令发给 TMA，调用返回并不表示数据已经搬完。因此 `src/group_gemm/kernels.cuh:437-443`（函数 `group_gemm_fp8_kernel`）的两次 `copy` 与后面的 `set_barrier_transaction_bytes` 之间没有一个“等待 copy 完成”的隐含同步。

按 Hopper PTX 的 mbarrier 计数模型，异步操作可以先启动，再设置用于跟踪它们的 barrier；PTX 对 tx-count 也允许负值，所以极快的 `complete_tx` 先于后面的 `expect_tx` 到达时，完成字节可以先记成负债，随后 `expect_tx` 再把预期字节加回来。只要在 barrier phase 被复用或消费者真正通过之前完成这次 `arrive.expect_tx`，这种“issue-then-arm”顺序是可行的。可参见官方 [PTX ISA 的 mbarrier tracking](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier) 和 [CUDA asynchronous copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)。

当前实现满足这些前提：

- `src/group_gemm/kernels.cuh:381-384`（函数 `group_gemm_fp8_kernel`）只让 load warp 的一个 elected thread 发起这些操作；
- `src/group_gemm/kernels.cuh:435-443`（函数 `group_gemm_fp8_kernel`）在同一个连续代码段中完成 writable wait、两次 TMA submit 和一次 `arrive.expect_tx`；
- 消费者只在 `src/group_gemm/kernels.cuh:513`（函数 `group_gemm_fp8_kernel`）等待 `readable`，不会在 `arrive.expect_tx` 之前把该 phase 当作完成；
- `bytes` 必须等于同一 barrier 上所有 TMA load 的实际传输字节数，且每个 stage 每个 phase 只能执行一次对应的 arrive。

因此，当前顺序不能解释为“先把数据拷完再设置 barrier”；它是“先提交异步操作，再立即登记本 phase 的完成条件”。

## 9.5 tutorial 的顺序与推荐选择

外部 CUTLASS tutorial `../cutlass/examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:215-224`（函数 `gemm_device`）采用：

```text
ProducerBarType::arrive_and_expect_tx(...)
copy(tma_a.with(...), ...)
copy(tma_b.with(...), ...)
```

`ProducerBarType::arrive_and_expect_tx` 在 `../cutlass/include/cutlass/arch/barrier.h:586-602`（函数 `ClusterTransactionBarrier::arrive_and_expect_tx`）发出的核心 PTX 与本项目的 `set_barrier_transaction_bytes` 相同。tutorial 的 steady-state refill 也保持这一顺序，见 `../cutlass/examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:317-330`（函数 `gemm_device`）。

两种顺序的结论是：

1. **语义上都可以成立**：当前 hpc-ops 的 copy-then-arm 顺序不是因为 `cute::copy` 同步完成，而是利用异步 TMA 和 mbarrier tx-count 的记账模型；它必须满足上一节列出的单线程、字节数、phase 生命周期条件。
2. **新代码更推荐 tutorial 的 arm-then-copy 顺序**：先执行 `set_barrier_transaction_bytes`/`arrive_and_expect_tx`，再提交 TMA，屏障的“本 phase 等待什么”在发起异步操作前就明确建立，代码审查和跨版本维护更直观，也与 CUTLASS Hopper tutorial 及较新的 CUTLASS 示例保持一致。
3. **不要为了形式一致而单独改动并假设性能必然变好**：两次 TMA load 已经是异步提交，顺序通常不是主要性能因素；若保留当前写法，应加注释说明这是 issue-then-arm，并用目标 GPU 的正确性测试/benchmark 验证。若改成 tutorial 顺序，等价改写为：

   ```cpp
   set_barrier_transaction_bytes(readable[ismem_write], kTransactionBytes);
   cute::copy(tma_a.with(td_x, readable[ismem_write]), tAg(_, itile_m, itile_k),
              tAs(_, 0, 0, ismem_write));
   cute::copy(tma_b.with(readable[ismem_write]), tBg(_, itile_n, itile_k, igroup),
              tBs(_, 0, 0, ismem_write));
   ```

最后再强调：`tma_a.with` 的第二个参数和 `tma_b.with` 的唯一参数都是同一个 `readable` barrier；A 多出来的第一个参数只是在 group GEMM 的 ragged X 场景下替换 per-group descriptor。`with` 不负责发起搬运，`cute::copy` 才发起 TMA，`set_barrier_transaction_bytes`/`arrive_and_expect_tx` 负责把这些异步搬运纳入 barrier 的完成条件。

# 10. line 440：`tBg` 与 TMA descriptor 的分工

## 10.1 直接结论

`tBg(_, itile_n, itile_k, igroup)` **不包含 W 的 global memory base address**。它是 TMA 专用的 coordinate tensor，主要负责产生本次 TMA load 的起始坐标；它的 Tensor 对象仍然有 shape/layout（因此有坐标步长和 tile 分区信息），但其 `data()` 是算术坐标迭代器，不是 `w_ptr` 对应的 `gmem_ptr`。

W 的物理地址在 host/device launch 侧先进入原始 GMEM Tensor：`src/group_gemm/group_gemm_pertensor_fp8.cu:37-42`（函数 `launch_group_gemm_fp8`）用 `w_ptr`、`shape(n,k,num_group)` 和 `stride(k,1,n*k)` 构造 W。随后 `src/group_gemm/config.h:90-95`（函数 `GroupGEMMFp8Config::get_tma`）把 W 传给 `make_tma_copy`。CuTe 的 `detail::make_tma_copy_desc` 在 `3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:928-952`（函数 `make_tma_copy_desc`）从 `gtensor.data()` 取得 `gmem_address`，并读取全局 shape/stride；在 `3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:1029-1058`（函数 `make_tma_copy_desc`）把 global address、global shape/stride、shared-memory box shape/stride、数据格式和 swizzle 编入 `TmaDescriptor`。所以，**global base address 在 descriptor 中，而不在 `tBg` 中**。

需要更精确地说：原始 `tma_b` 的 atom 保存 `tma_desc_`；`tma_b.with(readable[ismem_write])` 并不是再复制一份 descriptor，而是 `3rd/cutlass/include/cute/atom/copy_atom.hpp:76-83`（函数 `Copy_Atom::with`）转调 `Copy_Traits::with`。对于本例，命中 `3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:124-133`（函数 `Copy_Traits<SM90_TMA_LOAD>::with`），返回的 executable atom 的运行时参数是 `&tma_desc_`、barrier 地址和 cache hint。

## 10.2 `tBg` 是怎样生成坐标的

`src/group_gemm/kernels.cuh:265-278`（函数 `group_gemm_fp8_kernel`）先执行：

```cpp
auto gB  = tma_b.get_tma_tensor(make_shape(n, k, num_group));
auto tBg = btma_b.partition_S(gB);
```

`get_tma_tensor` 的实现位于 `3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:147-154`（函数 `Copy_Traits<SM90_TMA_LOAD>::get_tma_tensor`）：它用全局 shape 和 TMA coordinate stride 构造 `make_coord_tensor`，并没有读取或复制 descriptor 的 base pointer。`make_coord_tensor` 位于 `3rd/cutlass/include/cute/tensor_impl.hpp:477-487`（函数 `make_coord_tensor`），明确把 layout 绑定到 counting/`ArithmeticTupleIterator`。

因此，`btma_b.partition_S(gB)`（`3rd/cutlass/include/cute/atom/copy_atom.hpp:365-373`，函数 `ThrCopy::partition_S`）只是保留这个坐标 engine 并重新组织 layout。`tBg(_, itile_n, itile_k, igroup)` 的含义是：保留 TMA box/atom 模式，用 `itile_n`、`itile_k` 和 `igroup` 选择当前 N/K/group tile，最终得到类似 `{coord_n, coord_k, coord_group}` 的逻辑坐标 tuple。坐标是 TMA 指令使用的坐标单位，不是已经计算好的字节地址；descriptor 的 base/stride 负责把这些坐标解释成 global memory 地址。

运行日志也印证了这一点：`temp/run.log:130-133`（函数 `group_gemm_fp8_kernel` 的 debug 输出）中 `gB` 打印为 `ArithTuple`，而不是 `gmem_ptr`；`temp/run.log:166-167`（同一函数）中 `tBg` 仍打印为 `ArithTuple`，并显示 `(TMA, TMA_N, TMA_K, num_group)` 的坐标分区。

## 10.3 `cute::copy` 两个参数分别给什么

在 line 440，`tma_b` 实际是 `make_tma_copy` 返回的 `TiledCopy`；`3rd/cutlass/include/cute/atom/copy_atom.hpp:185-189`（函数/类型 `TiledCopy`）说明它继承 `Copy_Atom`。`cute::copy` 在 `3rd/cutlass/include/cute/algorithm/copy.hpp:184-196`（函数 `copy` 的 `Copy_Atom` overload）把这个 atom 调到 `Copy_Atom::call`，再进入 `3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:64-90`（函数 `TMA_LOAD_Unpack::copy_unpack`）。该函数的关键代码是：

```cpp
auto src_coord = src(Int<0>{});       // 从第二个参数取坐标
void* dst_ptr = raw_pointer_cast(dst.data()); // 从第三个参数取 SMEM 地址
```

对应关系如下：

| `cute::copy` 参数 | 提供给 TMA 的信息 |
|---|---|
| 第 1 个：`tma_b.with(readable[ismem_write])` | TMA load 操作类型/位宽和编译期 source-destination 映射；descriptor 指针（global base、global shape/stride、TMA box、格式/swizzle 等）；`with` 注入的 completion barrier 指针和 cache hint。 |
| 第 2 个：`tBg(_, itile_n, itile_k, igroup)` | 本次调用的动态起始坐标 tuple，以及坐标 tensor 的 rank/layout/分区关系；**不提供 W 的物理 base pointer**。 |
| 第 3 个：`tBs(_, 0, 0, ismem_write)` | 目标 shared-memory 地址和 SMEM layout（这里只是补充完整指令所需的信息）。 |

最终，`TMA_LOAD_Unpack::copy_unpack` 把上述运行时参数展开给 `3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:327-349`（函数 `SM90_TMA_LOAD::copy`）。W 是三维 `(n,k,num_group)` tensor，因此对应三坐标 overload，概念上形成：

```text
cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes
    [smem_ptr], [tma_desc_ptr, {coord_n, coord_k, coord_group}], [mbar_ptr], cache_hint;
```

三维指令的具体参数和汇编模板见 `3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:159-186`（函数 `SM90_TMA_LOAD_3D::copy`）：descriptor 提供“这块 global tensor 是谁、怎样解释”，坐标提供“这次从哪里开始”，SMEM 指针提供“写到哪里”，barrier 提供“何时报告完成”。这四类信息是互补的，不能用 `tBg` 取代 descriptor。

在本 kernel 中，`igroup` 只改变 B descriptor 下的第三维坐标；因此同一个 W descriptor 可以覆盖所有 group。相对地，A 的 per-group descriptor 在 `src/group_gemm/kernels.cuh:430-441`（函数 `group_gemm_fp8_kernel`）通过 `td_x` 显式替换，正是因为 A 的每个 group 可能有不同的起始地址/有效形状。

# 11. line 437/440：一次 `cute::copy` 的 TMA 指令数

## 11.1 当前运行的直接答案

以 `temp/run.log` 中这次运行的实例为准（`temp/run.log:6`，函数 `launch_group_gemm_fp8` 的 debug 输出）：

| 代码行 | TMA box | 每次 `cute::copy` 的逻辑元素数 | 产生的 tensor TMA 指令 |
|---|---:|---:|---:|
| `src/group_gemm/kernels.cuh:437`（函数 `group_gemm_fp8_kernel`） | `kTileM × kTileK = 48 × 128` | `6144` 个 FP8 元素（6144 bytes） | **1 条** `cp.async.bulk.tensor.2d` |
| `src/group_gemm/kernels.cuh:440`（函数 `group_gemm_fp8_kernel`） | `kTileN × kTileK × 1 = 128 × 128 × 1` | `16384` 个 FP8 元素（16384 bytes） | **1 条** `cp.async.bulk.tensor.3d` |

这里的“1 条”是指一次已经固定了 `itile_m/itile_n` 和 `itile_k` 的 `cute::copy` 调用，不是说整个 K 维只传一次。代码在 `src/group_gemm/kernels.cuh:432-450`（函数 `group_gemm_fp8_kernel`）中对 `itile_k` 循环；日志里的 `tAg` 形状为 `(..., 12, 56)`（`temp/run.log:162-163`），所以对一个固定的 M/N tile 和 group，完整 K 扫描会发出 **56 条 A 指令 + 56 条 B 指令**。两条 copy 合计的每次迭代传输字节数为 `6144 + 16384 = 22528`，与 `src/group_gemm/kernels.cuh:377-378`（函数 `group_gemm_fp8_kernel`）计算的 `kTransactionBytes` 一致。该 TMA submit 只由 load warp 的 elected leader 发起，见 `src/group_gemm/kernels.cuh:380-383`（函数 `group_gemm_fp8_kernel`），因此不是每个 warp lane 各发一条。

这里统计的是 `cp.async.bulk.tensor` 数据搬运指令；`src/group_gemm/kernels.cuh:443`（函数 `group_gemm_fp8_kernel`）的 `set_barrier_transaction_bytes` 是另一个 mbarrier 记账指令，不应算作额外的 `cp.async.bulk.tensor`。

## 11.2 TiledCopy 和 CopyAtom 的实际关系

用户问题中“`tma_a.with(...)` 本身是一个 TiledCopy”需要做一个类型上的修正：

- `tma_a`/`tma_b` 本身是 `make_tma_copy` 返回的 `TiledCopy`。`src/group_gemm/config.h:90-95`（函数 `GroupGEMMFp8Config::get_tma`）创建了它们；`3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:1193-1233`（函数 `make_tma_copy_tiled`）先构造一个 `Copy_Atom atom`，再返回 `TiledCopy<decltype(atom), ...>{atom}`。
- `TiledCopy` 的模板参数和计数成员位于 `3rd/cutlass/include/cute/atom/copy_atom.hpp:185-203`（类型 `TiledCopy`）。因此对象中有一个 base `Copy_Atom`，另外保存 `Tiler_MN` 和 `TiledLayout_TV`，用来决定如何把一个 source/destination tensor 分区。
- `.with(...)` 来自 `3rd/cutlass/include/cute/atom/copy_atom.hpp:76-83`（函数 `Copy_Atom::with`），返回的是绑定了运行时 descriptor/barrier 的 **executable `Copy_Atom`**，不再是 `TiledCopy`。`src/group_gemm/kernels.cuh:437` 和 `src/group_gemm/kernels.cuh:440` 的第一个实参实际是这个 executable `Copy_Atom`。

即使直接把未调用 `.with` 的 `TiledCopy` 传给 `cute::copy`，`3rd/cutlass/include/cute/algorithm/copy.hpp:434-444`（函数 `copy` 的 `TiledCopy` overload）也会把它静态转换为 base `Copy_Atom` 再执行；本代码只是先通过 `.with` 绑定了 descriptor/barrier。

所以，“TiledCopy 内部包含几个 CopyAtom”要分两层理解：类型/对象层面是一个 base `Copy_Atom`；在本例的 tile 布局中，它也只需要调用这个 atom **一次**，不是 6144 次或 16384 次。

从日志布局还可以直接看到这个复制倍率：A 的 `TiledLayout_TV` 是 1 个逻辑线程、6144 个 value，而 atom 的 `ValLayout` 是 1 个线程、6144 个 value；B 对应为 1 个线程、16384 个 value（`temp/run.log:43-61`、`temp/run.log:93-113`）。因此 `TiledNumThr/AtomNumThr = 1` 且 `TiledNumVal/AtomNumVal = 1`，逻辑上的 CopyAtom invocation 数就是 1。

## 11.3 为什么一次调用只有一条 `cp.async.bulk`

当前日志给出的静态布局是：

- A：`Tiler_MN=(_48,_128)`，`ValLayoutSrc/Dst/Ref=(_1,_6144)`，见 `temp/run.log:43-51` 和 `temp/run.log:93-102`（函数 `group_gemm_fp8_kernel` 的 debug 输出）。`tAg` 为 `((( _128,_48),_1),12,56)`，见 `temp/run.log:162-165`。
- B：`Tiler_MN=(_128,_128)`，`ValLayoutSrc/Dst/Ref=(_1,_16384)`，见 `temp/run.log:53-61` 和 `temp/run.log:104-113`。`tBg` 为 `((( _128,_128),_1),32,56,8)`，见 `temp/run.log:166-169`。

在 `src/group_gemm/kernels.cuh:437` 中，`tAg(_, itile_m, itile_k)` 固定了 M/K tile 的外层索引，只留下一个大小为 6144 的 TMA value mode；`src/group_gemm/kernels.cuh:440` 的 `tBg(_, itile_n, itile_k, igroup)` 同理只留下一个大小为 16384 的 TMA value mode。于是调用链没有额外的 rest-mode 循环：

1. `cute::copy` 的 `Copy_Atom` overload 位于 `3rd/cutlass/include/cute/algorithm/copy.hpp:184-196`（函数 `copy`）。索引后 source/destination 是 rank-1 atom-sized tensor，直接执行 `copy_atom.call(src,dst)`；只有 rank 大于 1 时，`3rd/cutlass/include/cute/algorithm/copy.hpp:197-235`（同一函数）才会遍历剩余 mode。
2. `Copy_Atom::call` 位于 `3rd/cutlass/include/cute/atom/copy_atom.hpp:89-114`（函数 `Copy_Atom::call`），直接进入 `copy_unpack`。
3. TMA 的 `copy_unpack` 位于 `3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:64-90`（函数 `TMA_LOAD_Unpack::copy_unpack`）。它只取一次 `src_coord = src(Int<0>{})`、一次 SMEM `dst_ptr`，再用一次 `CallCOPY` 展开调用；这里没有按元素循环。
4. A 的坐标是二维，`3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:327-342`（函数 `SM90_TMA_LOAD::copy`）选择 2D overload，而 `3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:103-131`（函数 `SM90_TMA_LOAD_2D::copy`）在预处理后只选择一条 `cp.async.bulk.tensor.2d...` 汇编指令。
5. B 的 W 是三维 `(n,k,num_group)`，对应 `3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:344-349`（函数 `SM90_TMA_LOAD::copy`）的 3D overload；`3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:159-186`（函数 `SM90_TMA_LOAD_3D::copy`）在预处理后只选择一条 `cp.async.bulk.tensor.3d...` 汇编指令。第三维 box extent 是 1，`igroup` 是起始坐标，不会把一次 tile 拆成 8 条指令。

也就是说，一条 tensor TMA 指令的地址操作数是“descriptor + 起始坐标”，box shape 则在 descriptor/atom 中描述；它可以一次搬完整的 2D/3D box，不需要每个 FP8 元素一条指令。

## 11.4 元素数、CopyAtom 数和外层循环数的区别

对当前 `GroupGEMMFp8Config`，SMEM tile 的定义见 `src/group_gemm/config.h:81-84`（类型 `GroupGEMMFp8Config`），TMA 创建见 `src/group_gemm/config.h:90-95`（函数 `GroupGEMMFp8Config::get_tma`）：

```text
A：每次 atom copy 的元素数 = kTileM * kTileK
B：每次 atom copy 的元素数 = kTileN * kTileK * 1
```

当前运行参数为 `kTileM=48, kTileN=128, kTileK=128`，所以分别是 6144 和 16384。`kStage=8` 只是流水线的多个 SMEM buffer：`src/group_gemm/kernels.cuh:437-441`（函数 `group_gemm_fp8_kernel`）通过 `ismem_write` 选择其中一个 stage，并没有把元素数乘以 8。其他 `kTileM` specialization 仍是每次一条 TMA 指令，但 A 的元素数随 `kTileM*kTileK` 变化；若走 `src/group_gemm/group_gemm_pertensor_fp8.cu:420-425`（函数 `group_gemm_fp8_async`）中 `kTileK=64` 的分支，公式中的 `kTileK` 就是 64。边界 tile 仍由 TMA descriptor 的 OOB 规则处理，不会因为边界而自动拆成多条指令。

最后要区分三个数量：

- **元素数**：一条 A 指令当前搬 6144 个、一条 B 指令当前搬 16384 个 FP8 元素；
- **CopyAtom 数**：每个 `TiledCopy` 有一个 base `Copy_Atom`，这个已索引的 tile view 只调用它一次；
- **循环次数**：`itile_k` 每增加一次，就再次发出一条 A 和一条 B 指令。当前 K=7168、`kTileK=128` 时是 56 次，见 `src/group_gemm/kernels.cuh:387-388`（函数 `group_gemm_fp8_kernel`）及 `temp/run.log:162-167`。

# 12. math warpgroup 分支逐行说明

下面的行号以当前 code base 工作树为准。用户问题中所说的 `if (idx >= kNumThreads)` 的 `else`，当前位于 `src/group_gemm/kernels.cuh:340-421`（函数 `group_gemm_fp8_kernel`）；调试 `printf`/`print` 代码不在说明范围内。

## 12.1 本次运行的具体配置

本次日志给出的输入是 `m=576, n=4096, k=7168, num_group=8`，每个 group 的 sequence length 为 `16,32,48,64,80,96,112,128`，见 `temp/run.log:261-272`（函数 `group_gemm_fp8_kernel` 的运行时输出）。平均 sequence length 是 72，因此 host dispatch 选择 `kTileM=48` 分支；该分支及其 `kTileN=128,kTileK=128,kStage=8` 配置见 `src/group_gemm/group_gemm_pertensor_fp8.cu:437-443`（函数 `group_gemm_fp8_async`）。由于 `n>1024` 且 `k>1024`，任务策略选择 policy 2，见 `src/group_gemm/group_gemm_pertensor_fp8.cu:308-327`（函数 `launch_group_gemm_fp8`）。

`TiledMma` 的运行时打印见 `temp/run.log:172-181`：

```text
Shape_MNK      = (64, 48, 32)
ThrID          = 128
ThrLayoutVMNK  = (128, 2, 1, 1)
```

因此：

- 一个 GMMA atom 的逻辑形状是 `M=64,N=48,K=32`，由 128 个线程协作；
- M 方向复制因子是 2，所以整个 `TiledMma` 需要 `128*2=256` 个 math 线程；
- kernel block 是 384 线程，`src/group_gemm/kernels.cuh:340-342`（函数 `group_gemm_fp8_kernel`）把 `idx<256` 的线程分给 math warpgroup，把 `idx>=256` 的 128 个线程分给 load warpgroup；
- 两个 math warpgroup 各计算一个 `64x48` 子块，合起来计算本次输出 tile 的 `128x48` 元素。

本项目的矩阵逻辑是 `Y[n,m] = W[n,k] * X[m,k]^T`。为了匹配 CuTe 的 GEMM 约定 `A(M,K) x B(N,K) -> C(M,N)`，代码故意把 W 当作 GMMA 的 A、把 X 当作 GMMA 的 B，最后得到逻辑 `C(128,48)`，正好对应输出 Y 的 `(n,m)` tile。W/X 的原始 shape 见 `src/group_gemm/group_gemm_pertensor_fp8.cu:37-42`（函数 `launch_group_gemm_fp8`）。

## 12.2 进入 math warpgroup 和配置寄存器

```cpp
} else {
  // math warpgroup
  cutlass::arch::warpgroup_reg_alloc<168>();
```

位置：`src/group_gemm/kernels.cuh:421-423`（函数 `group_gemm_fp8_kernel`）。load warpgroup 在前一个分支中调用 `warpgroup_reg_dealloc<24>`，让出寄存器资源；math warpgroup 这里申请 168 个寄存器配额，给 WGMMA 的异步累加器和 descriptor iterator 使用。实现见 `3rd/cutlass/include/cutlass/arch/reg_reconfig.h:72-78`（函数 `warpgroup_reg_alloc`）。

```cpp
int iwarpgroup = idx / 128;
```

位置：`src/group_gemm/kernels.cuh:425`（函数 `group_gemm_fp8_kernel`）。math 分支中的 `idx` 仍是原始 block thread index，因此 `idx=0..127` 时 `iwarpgroup=0`，`idx=128..255` 时 `iwarpgroup=1`。后面的 TMA store 会用这个值选择两个 `64x48` shared-memory/global-memory 子块。

## 12.3 建立线程私有的 MMA 视图

```cpp
TiledMma tiled_mma;
auto thr_mma = tiled_mma.get_slice(idx);
```

位置：`src/group_gemm/kernels.cuh:427-429`（函数 `group_gemm_fp8_kernel`）。`TiledMma` 的类型由 `src/group_gemm/config.h:98-100`（类型 `GroupGEMMFp8Config`）定义；`kTileM=48` 时，`src/group_gemm/config.h:42-60`（函数 `mma_selector`）选择 `SM90_64x48x32_F32E4M3E4M3_SS_TN`。该 operation 的 trait 位于 `3rd/cutlass/include/cute/atom/mma_traits_sm90_gmma_ext.hpp:9831-9855`（类型 `MMA_Traits<SM90_64x48x32_F32E4M3E4M3_SS_TN>`），说明 A/B 是 FP8 E4M3，C/D 是 FP32，atom 的 `Shape_MNK` 是 `(64,48,32)`。

`get_slice(idx)` 根据线程号计算当前线程在 `(V,M,N,K)` tiled layout 中的坐标，CuTe 实现见 `3rd/cutlass/include/cute/atom/mma_atom.hpp:355-363`（函数 `TiledMMA::get_slice`）。它返回的是线程视图，不会执行任何数据搬运。

```cpp
auto tBs4r = thr_mma.partition_A(sB);
auto tAs4r = thr_mma.partition_B(sA);
```

位置：`src/group_gemm/kernels.cuh:430-431`（函数 `group_gemm_fp8_kernel`）。`partition_A/B` 的实现见 `3rd/cutlass/include/cute/atom/mma_atom.hpp:475-495`（函数 `ThrMMA::partition_A`/ `partition_B`）。这里的变量命名容易误导：`sB` 是 W 的 `(128,128)` shared-memory tile，但被映射成 GMMA 的逻辑 A `(M,K)`；`sA` 是 X 的 `(48,128)` tile，但被映射成 GMMA 的逻辑 B `(N,K)`。这样 GMMA 的逻辑输出 `(M,N)=(128,48)` 与 `gC` 布局一致。

```cpp
auto tBr = thr_mma.make_fragment_A(tBs4r);
auto tAr = thr_mma.make_fragment_B(tAs4r);
```

位置：`src/group_gemm/kernels.cuh:433-434`（函数 `group_gemm_fp8_kernel`）。对于 `SS_TN` GMMA，这两个 fragment 不是把 FP8 元素逐个读入普通寄存器，而是生成 GMMA shared-memory descriptor iterator；相关 `MakeTensor<GMMA::smem_desc>` 实现见 `3rd/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:361-371`（函数 `MakeTensor::operator()`）。日志 `temp/run.log:196-203` 中它们打印为 `GMMA::DescriptorIterator`。其中 `_4` K mode 表示一个 `kTileK=128` 被 atom 的 `K=32` 拆成 `4` 个子块。

```cpp
auto tCr = thr_mma.partition_fragment_C(gC);
```

位置：`src/group_gemm/kernels.cuh:436`（函数 `group_gemm_fp8_kernel`）。`gC` 在 `src/group_gemm/kernels.cuh:267-269`（函数 `group_gemm_fp8_kernel`）只是用 `nullptr` 指针和 `(kTileN,kTileM)` shape 构造的布局模板，并不是实际 global-memory 输出。实现见 `3rd/cutlass/include/cute/atom/mma_atom.hpp:497-503`（函数 `ThrMMA::partition_fragment_C`）。本次日志 `temp/run.log:204-205` 显示每个线程持有 24 个 FP32 累加值。
## 12.4 任务、stage 和 K tile 循环

```cpp
int ismem_read = 0;
int phase = 0;
```

位置：`src/group_gemm/kernels.cuh:438-439`（函数 `group_gemm_fp8_kernel`）。`ismem_read` 是 8-stage shared-memory ring buffer 的读索引；`phase` 是 mbarrier parity。load warpgroup 在 `src/group_gemm/kernels.cuh:351-355`（函数 `group_gemm_fp8_kernel`）从另一侧以 phase 1 开始，math warpgroup 以 phase 0 等待，二者配合 mbarrier 的奇偶 phase 协议。

```cpp
int iblock = blockIdx.x;
int igroup = 0;
int sum_tile_m = 0;
int itile_m, itile_n;
int4 task;
int iwave = 0;
```

位置：`src/group_gemm/kernels.cuh:441-446`（函数 `group_gemm_fp8_kernel`）。这些变量分别表示 CTA 当前处理的线性任务编号、group、horizon policy 使用的累计 M tile 数、当前 M/N tile 坐标、policy 0 的 task-map 临时值以及 task-map wave 下标。

```cpp
while (true) {
  if constexpr (kTaskLoopPolicy == 0) { ... }
  else if constexpr (kTaskLoopPolicy == 1) { ... }
  else { ... }
}
```

位置：`src/group_gemm/kernels.cuh:447-468`（函数 `group_gemm_fp8_kernel`）。三个分支在编译期裁剪，当前运行实际走 policy 2：

- policy 0（`src/group_gemm/kernels.cuh:448-456`，函数 `group_gemm_fp8_kernel`）：从 `shm_tiles[iwave]` 读取预生成的 `int4` task；`task.x/y/z` 分别是 M tile、N tile、group，`igroup<0` 是结束哨兵；
- policy 1（`src/group_gemm/kernels.cuh:457-462`，函数 `group_gemm_fp8_kernel`）：调用 `get_next_tile_horizon`，按 horizon 顺序从 `tiles_ptr` 推导 group 和局部 M tile；
- policy 2（`src/group_gemm/kernels.cuh:463-467`，函数 `group_gemm_fp8_kernel`）：调用 `get_next_tile_vert`，再以 `itile_n>=num_tile_n` 判断结束。

当前 policy 2 的 `get_next_tile_vert` 实现位于 `src/group_gemm/kernels.cuh:42-66`（函数 `get_next_tile_vert`）。本次每组 M tile 数为：

```text
ceil([16,32,48,64,80,96,112,128] / 48) = [1,1,1,2,2,2,3,3]
```

所以 `total_m=15`，`num_tile_n=ceil(4096/128)=32`；线性 `iblock` 通过 `% total_m` 得到跨 group 的 M 位置，通过 `/ total_m` 得到 N tile。二分查找 prefix-sum `cu_tiles_ptr` 后得到 `igroup` 和组内 `itile_m`。日志中的 `tAg` 有 12 个 M tile 槽位和 56 个 K tile 槽位，`tBg` 有 32 个 N tile 槽位、56 个 K tile 槽位和 8 个 group，见 `temp/run.log:162-171`。

本次 `cu_tiles_ptr` 的前缀和是 `[0,1,2,3,5,7,9,12,15]`，可由 `temp/run.log:261-272` 的 sequence length 和 `kTileM=48` 推出。例如：`iblock=0` 映射到 `itile_n=0, igroup=0, itile_m=0`；`iblock=3` 映射到 `itile_n=0, igroup=3, itile_m=0`；`iblock=14` 映射到 `itile_n=0, igroup=7, itile_m=2`；`iblock=15` 开始下一列，映射到 `itile_n=1, igroup=0, itile_m=0`。当前 `gridDim.x=num_sm=132`，所以 `iblock += gridDim.x`（`src/group_gemm/kernels.cuh:470`，函数 `group_gemm_fp8_kernel`）让每个 CTA 处理线性 tile 序列 `blockIdx.x, blockIdx.x+132,...`。

```cpp
iblock += gridDim.x;
```

位置：`src/group_gemm/kernels.cuh:470`（函数 `group_gemm_fp8_kernel`）。当前 CTA 处理完一个任务后，以 grid-stride 方式跳到下一个任务；这不是数学运算，而是让多个 CTA 分摊有效的 `(group,M,N)` tile。

## 12.5 软件跨 K 累加器和 per-group scale

```cpp
auto tDr = make_tensor_like(tCr);
clear(tDr);

float scale = yscale_ptr[igroup];
```

位置：`src/group_gemm/kernels.cuh:472-475`（函数 `group_gemm_fp8_kernel`）。`tDr` 是独立的 FP32 软件累加器，在一个输出 tile 开始时清零一次；`tCr` 只保存当前 128-wide K tile 的局部 GMMA 结果。scale 是当前 group 的输出缩放因子，来自 `yscale_ptr[igroup]`。

```cpp
int ntile_k = size<2>(tAg);
#pragma unroll 1
for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
```

位置：`src/group_gemm/kernels.cuh:477-479`（函数 `group_gemm_fp8_kernel`）。本次 `ntile_k=7168/128=56`。`#pragma unroll 1` 保留外层 K 循环，便于运行时流水线逐 stage 推进；每一次循环处理一个 A/B shared-memory stage 对应的 128 个 K 元素。

```cpp
wait_barrier(readable[ismem_read], phase);
```

位置：`src/group_gemm/kernels.cuh:480`（函数 `group_gemm_fp8_kernel`）。load warpgroup 先发起 A/B 的异步 TMA load，并在 `readable[stage]` 上设置 transaction bytes；math warpgroup 必须等该 mbarrier parity 完成后才能让 WGMMA 读取 shared memory。实现见 `3rd/cutlass/include/cute/arch/copy_sm90_desc.hpp:89-110`（函数 `wait_barrier`），内部循环 `mbarrier.try_wait.parity`。

本次每个 stage 的 transaction bytes 是 `sizeof(Tin) * (48+128) * 128 = 22528` 字节（`Tin` 为 1-byte FP8）：A TMA copy 搬 `48*128=6144` 字节，B TMA copy 搬 `128*128=16384` 字节，两者之和由 load leader 在 `src/group_gemm/kernels.cuh:345` 和 `src/group_gemm/kernels.cuh:410`（函数 `group_gemm_fp8_kernel`）设置到同一个 `readable` barrier。因而 math 侧等待的是两次 load 合计完成，而不是只等待其中一条 copy。

## 12.6 `accumulate_ = Zero/One` 的含义

核心代码是：

```cpp
tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
warpgroup_fence_operand(tCr);
warpgroup_arrive();
#pragma unroll
for (int ik = 0; ik < size<2>(tAr); ++ik) {
  cute::gemm(tiled_mma,
             tBr(_, _, ik, ismem_read),
             tAr(_, _, ik, ismem_read),
             tCr(_, _, _));
  tiled_mma.accumulate_ = GMMA::ScaleOut::One;
}
```

位置：`src/group_gemm/kernels.cuh:482-490`（函数 `group_gemm_fp8_kernel`）。

CuTe 对 GMMA 的定义在 `3rd/cutlass/include/cute/arch/mma_sm90_gmma.hpp:105-130`（命名空间 `SM90::GMMA`）：

```text
ScaleOut::Zero = 0
ScaleOut::One  = 1
C = (scaleA * A) * (scaleB * B) + (scaleD * C)
```

这里的 `C` 是 WGMMA 目的寄存器中原有的 accumulator 值；`ScaleOut` 只控制旧 C 是否参与，不是 A/B 输入的量化缩放，也不是 `yscale_ptr` 的替代品。

对本次 FP8 operation，具体 PTX wrapper 是 `3rd/cutlass/include/cute/arch/mma_sm90_gmma_ext.hpp:28805-28854`（函数 `MMA_64x48x32_F32E4M3E4M3_SS_TN::fma`）：

- `fma` 接收 `GMMA::ScaleOut scale_D`，见 `3rd/cutlass/include/cute/arch/mma_sm90_gmma_ext.hpp:28817-28827`；
- `setp.ne.b32 p, scale_D, 0` 生成是否使用旧 C 的谓词，见 `3rd/cutlass/include/cute/arch/mma_sm90_gmma_ext.hpp:28830-28834`；
- 随后发出 `wgmma.mma_async.sync.aligned.m64n48k32.f32.e4m3.e4m3`，见 `3rd/cutlass/include/cute/arch/mma_sm90_gmma_ext.hpp:28834-28850`。

CuTe 的 `mma_unpack` 把 `traits.accumulate_` 的地址传入这个 `fma`，见 `3rd/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:392-430`（函数 `SM90::GMMA::mma_unpack`）。因此，`tiled_mma.accumulate_` 会直接影响每条 WGMMA 的 D/C 累加控制位。

为什么先 Zero、随后 One：

1. 当前 `kTileK=128`，而 atom K 是 32；`size<2>(tAr)=4`，所以一个外层 `itile_k` 要发 4 条 WGMMA；
2. 第一个 `ik=0` 必须执行 `tCr = A0*B0`，即 `scale_D=Zero`，忽略进入循环前 `tCr` 中的旧值；
3. `ik=1,2,3` 要执行 `tCr += Ai*Bi`，所以在第一次 `cute::gemm` 返回后设置 `One`；
4. 这 4 条指令合起来才是当前 128-wide K tile 的完整局部乘加。

用公式表示就是：

```text
tCr = A0*B0
tCr = A1*B1 + tCr
tCr = A2*B2 + tCr
tCr = A3*B3 + tCr
```

必须在每个外层 `itile_k` 重新设 Zero，因为 `tCr` 是当前 K tile 的临时累加器，不是 56 个 K tile 的最终累加器；如果只在输出 tile 开头设一次，下一次 `itile_k` 的第一条 WGMMA 会错误地把上一个 K tile 的 `tCr` 再加一遍。

跨 56 个外层 K tile 的真正累加发生在后面：

```cpp
tDr(i) = tCr(i) * scale + tDr(i);
```

位置：`src/group_gemm/kernels.cuh:498-501`（函数 `group_gemm_fp8_kernel`）。因此 `accumulate_` 的作用范围是“当前 128-K tile 内的 4 条 WGMMA”，而 `tDr` 的作用范围是“整个 7168-K GEMM 的 56 个 tile”。`yscale_ptr` 的 group scale 也在这里显式乘入，不能由 `ScaleOut` 完成。

## 12.7 提交并等待异步 WGMMA，然后释放 shared-memory stage

```cpp
warpgroup_commit_batch();
warpgroup_wait<0>();
warpgroup_fence_operand(tCr);

arrive_barrier(writable[ismem_read]);
```

位置：`src/group_gemm/kernels.cuh:492-496`（函数 `group_gemm_fp8_kernel`）。

- `cute::gemm` 发出的 WGMMA 是异步指令；调用返回不代表 FP32 结果已经写回 `tCr`。
- `warpgroup_commit_batch()` 发出 `wgmma.commit_group.sync.aligned`，把此前由这个 warpgroup 发出的 WGMMA 放入一个可等待的 batch。实现见 `3rd/cutlass/include/cute/arch/mma_sm90_gmma.hpp:73-83`（函数 `warpgroup_commit_batch`）。
- `warpgroup_wait<0>()` 发出 `wgmma.wait_group.sync.aligned 0`。模板参数 0 表示等待到已提交的 WGMMA pending group 数为 0，也就是当前 `tCr` 可以被普通 CUDA 指令读取。实现见 `3rd/cutlass/include/cute/arch/mma_sm90_gmma.hpp:59-71`（函数 `warpgroup_wait`）。
- 第二个 `warpgroup_fence_operand(tCr)` 更准确地说，是 CUTLASS 用“空的 `volatile` inline asm + 每个 accumulator 的 read-write operand + `memory` clobber”实现的**编译器代码移动约束**。CUTLASS 的相关说明通常把它称为 **NVVM code-motion fence**，但它不是独立的 PTX/GPU 硬件原语，也不是等待 WGMMA 完成的指令。它只标记列出的 accumulator live range：约束编译器不要把会定义、覆盖、reload 或错误使用这些 accumulator 的普通指令放进 WGMMA in-flight 区间；完全不依赖这些 operand 的独立指令仍可能被调度到前后。实现见 `3rd/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:45-66`（函数 `warpgroup_fence_operand(Tensor<Engine, Layout>&)`）。

### 12.7.1 `warpgroup_fence_operand(tCr)` 在本次实例中的展开

`tCr` 是 `src/group_gemm/kernels.cuh:436`（函数 `group_gemm_fp8_kernel`）由 `partition_fragment_C` 创建的寄存器 tensor。运行日志显示每个线程的形状为 `((_2,_2,_6),_1,_1)`，即 `2*2*6=24` 个 FP32 accumulator，见 `temp/run.log:204-205`（函数 `group_gemm_fp8_kernel` 的调试输出）。这也与本次 atom 的输出规模吻合：一条 `m64n48` WGMMA 产生 `64*48=3072` 个 FP32 输出，由 128 个线程共同持有，所以每线程正好持有 `3072/128=24` 个逻辑 accumulator。

调用点的重载解析也值得明确：`tCr` 是 `cute::Tensor<Engine, Layout>` 左值，所以 `src/group_gemm/kernels.cuh:484` 和 `src/group_gemm/kernels.cuh:494`（函数 `group_gemm_fp8_kernel`）匹配的是 `warpgroup_fence_operand(Tensor<Engine, Layout>&)` 模板，而不是 `float&` 标量 overload。模板内部对 `f32_frg(i)` 逐元素取出 `float&` 后，才继续分派到 `warpgroup_fence_operand(float&)`。两个层次的实现分别见 `3rd/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:45-66`（函数 `warpgroup_fence_operand(Tensor<Engine, Layout>&)`）和 `3rd/cutlass/include/cute/arch/mma_sm90_gmma.hpp:97-103`（函数 `warpgroup_fence_operand(float&)）。
tensor overload 的执行路径是：

1. `CUTE_STATIC_ASSERT(is_static<Layout>::value)` 要求 fragment layout 在编译期已知。这里不是运行时检查，也不会访问 shared/global memory；目的是让下面的遍历可以静态展开。
2. 当前 `Engine::value_type=float`，所以进入 `if constexpr` 的 float 分支，而不是整数分支。
3. `recast<float>(frg)` 只改变 tensor view 的类型视图。因为原来的元素类型已经是 float，`recast` 的同类型路径直接复用原 data/layout，不做数值转换或内存 copy，见 `3rd/cutlass/include/cute/tensor_impl.hpp:756-780`（函数 `recast`）。
4. `CUTE_UNROLL` 把 24 次循环展开；概念上等价于对本线程的每一个 `tCr(i)` 分别调用一次标量 overload。因此它不是对一个“抽象 tensor 对象”发一条 fence，而是给该线程持有的每个 accumulator register 建立约束。

概念展开形式如下（这是语义等价的伪代码，不是额外的运行时循环）：

```cpp
for (int i = 0; i < 24; ++i) {
  asm volatile("" : "+f"(tCr(i)) :: "memory");
}
```

标量 overload 位于 `3rd/cutlass/include/cute/arch/mma_sm90_gmma.hpp:86-103`（函数 `warpgroup_fence_operand(uint32_t&)` 和 `warpgroup_fence_operand(float&)`）。如果 accumulator 是整数类型，tensor overload 会走另一分支：先 `recast<uint32_t>`，再使用 `"+r"`；本次 FP8->FP32 kernel 走的是 `"+f"` 浮点分支。标量实现还受 `#if defined(__CUDA_ARCH__)` 保护：host 编译 pass 中函数体为空，只有 device 编译 pass 才建立这些 inline-asm 约束。每个线程只约束自己持有的 24 个 fragment 元素；它没有 arrival count、phase 或跨线程通信，因此本身不是 128-thread warpgroup barrier。相对地，`src/group_gemm/kernels.cuh:485`（函数 `group_gemm_fp8_kernel`）调用的 `warpgroup_arrive` 展开为带 `.sync.aligned` 的硬件指令，要求 warpgroup 按一致路径执行；这正是“operand fence 是 per-thread 编译器标记、`wgmma.fence` 是 warpgroup 硬件同步”的区别。

### 12.7.2 空 inline asm 的每个部分分别表示什么

对本次 float overload：

```cpp
asm volatile("" : "+f"(reg) :: "memory");
```

- `""`：asm 模板为空，因此该 asm 本身不要求生成 `mov`、`membar`、`wgmma` 等显式硬件指令。编译器仍可能因为寄存器分配在附近生成普通 `mov`，但那不是这个 helper 发出的指令；真正有意义的是 operand/clobber 约束。
- `volatile`：要求编译器保留这条 asm，不把它当成可删除的纯空语句。`volatile` 单独并不是“禁止所有代码移动”的完整屏障；本例的寄存器依赖来自 `+f`/`+r`，内存访问约束来自 `memory` clobber。这里的 volatile 修饰的是 asm 语句，不是对某个 shared/global 地址执行 volatile load/store。
- `+`：operand 是 **read-write**。编译器必须把 `reg` 的旧定义作为 asm 输入，并把 asm 之后的值作为输出传给后续使用，于是形成 `旧值 -> fence -> 新值` 的数据流边界。它不表示 helper 真的改变了数值，也不表示硬件已经完成异步写回。
- `f`：要求该 operand 使用浮点寄存器约束，正好对应 WGMMA wrapper 中的 `float d00...d23`。
- `"memory"` clobber：告诉编译器这条 asm 可能观察或影响任意内存，从而约束编译器可见的普通内存访问不要跨过它任意重排。它不是 CUDA 的 `__threadfence()`，不刷新 cache，不建立 CTA/warpgroup 间可见性，也不等待异步单元完成；shared-memory 的 TMA readiness 由 `readable` mbarrier 路径另行保证。

因此，`warpgroup_fence_operand` 的本质是“告诉编译器这些寄存器在这里形成一个异步 WGMMA 的边界”，而不是“在这里让 GPU 停下来”。可以把它理解成给特定寄存器 live range 加标签，而不是给整个线程或整个 warpgroup 加全局栅栏。

### 12.7.3 为什么这个 operand 正好是 `tCr`

本次选择的 GMMA operation 是 `MMA_64x48x32_F32E4M3E4M3_SS_TN`，其 `CRegisters=float[24]`，见 `3rd/cutlass/include/cute/arch/mma_sm90_gmma_ext.hpp:28810-28815`（类型 `MMA_64x48x32_F32E4M3E4M3_SS_TN`）。其 `fma` 的 `d00...d23` 都以 `"+f"` 传给 `wgmma.mma_async`，见同文件 `3rd/cutlass/include/cute/arch/mma_sm90_gmma_ext.hpp:28817-28850`（函数 `MMA_64x48x32_F32E4M3E4M3_SS_TN::fma`）。

CuTe 的 `mma_unpack` 将 `D` 和 `C` 视为同一组寄存器：它把可写的 D tensor 重解释为 `rC`，再把每个寄存器传给 `fma`，见 `3rd/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:392-430`（函数 `SM90::GMMA::mma_unpack`）。三参数 `MMA_Atom::call(A,B,C)` 进一步明确执行 `call(C,A,B,C)`，即 D=C 原地更新，见 `3rd/cutlass/include/cute/atom/mma_atom.hpp:107-118`（函数 `MMA_Atom::call`）。所以需要被标记的正是 `tCr`，而不是 `tDr`、`tAr` 或 `tBr`。

### 12.7.4 第一条 WGMMA 使用 `ScaleOut::Zero`，为什么 fence 仍是 `+f`

`src/group_gemm/kernels.cuh:482-490`（函数 `group_gemm_fp8_kernel`）中第一条 WGMMA 的 `ScaleOut::Zero` 只控制硬件公式里的旧 C 项是否参与：wrapper 通过谓词令第一条指令计算 `D=A*B`，后续三条才计算 `D=A*B+D`。它不把 `tCr` 清零，也不改变 C++/inline-asm 层面对这 24 个 destination operands 的描述。

具体 wrapper 始终把 `d00...d23` 写成 `"+f"`，见 `3rd/cutlass/include/cute/arch/mma_sm90_gmma_ext.hpp:28817-28850`（函数 `MMA_64x48x32_F32E4M3E4M3_SS_TN::fma`）；是否使用旧值由同一函数中的 `scale_D` 谓词决定。于是 compiler helper 也统一使用 `+f` 来维持同一组 in-place accumulator 的 live range。这里的“read”是编译器数据流含义，不意味着 `ScaleOut::Zero` 时硬件把旧 `tCr` 加入结果，也不意味着 pre-fence 初始化了 `tCr`。

### 12.7.5 在当前代码中的两个边界

当前 WGMMA 的顺序是 `src/group_gemm/kernels.cuh:482-496`（函数 `group_gemm_fp8_kernel`）：

```text
ScaleOut::Zero
warpgroup_fence_operand(tCr)  // 编译器边界：开始
warpgroup_arrive()            // 硬件 wgmma.fence.sync.aligned
4 x cute::gemm(...)           // 异步 WGMMA，写同一组 tCr
warpgroup_commit_batch()
warpgroup_wait<0>()           // 硬件等待 batch 完成
warpgroup_fence_operand(tCr)  // 编译器边界：结束
读取 tCr，更新 tDr
```

- **前一个（`src/group_gemm/kernels.cuh:484`，函数 `group_gemm_fp8_kernel`）**：它标记在第一条 WGMMA 之前，`tCr` 可能存在的普通寄存器访问已经结束；随后 `src/group_gemm/kernels.cuh:485`（函数 `group_gemm_fp8_kernel`）的 `warpgroup_arrive()` 才是真正的 `wgmma.fence.sync.aligned`。这个硬件 fence 排序的是此前 accumulator（以及采用寄存器 A fragment 时的相关 A 寄存器）访问与后续 WGMMA；本例是 `SS_TN`，A/B 通过 shared-memory descriptor 提供，因此实际重点是 accumulator 寄存器。它不负责让 shared memory 变得可读；A/B stage 的 TMA readiness 已由 `src/group_gemm/kernels.cuh:480`（函数 `group_gemm_fp8_kernel`）的 `wait_barrier(readable[ismem_read], phase)` 保证。也就是说，operand fence、硬件 `wgmma.fence`、TMA mbarrier 分别服务于编译器寄存器数据流、WGMMA 寄存器顺序和 shared-memory 数据可用性，三者不能互相替代。
- **后一个（`src/group_gemm/kernels.cuh:494`，即用户指出的文档 line 1827 所解释的调用）**：`src/group_gemm/kernels.cuh:493`（函数 `group_gemm_fp8_kernel`）的 `warpgroup_wait<0>()` 确认已提交的 4 条 WGMMA 完成后，这个 fence 把 accumulator 的“异步定义”与后续普通 C++ 使用连接起来。当前紧接着的依赖使用是 `src/group_gemm/kernels.cuh:499-500`（函数 `group_gemm_fp8_kernel`）的 `tCr(i)`，随后还会经过 BF16 转换和 STSM epilogue（`src/group_gemm/kernels.cuh:510-533`，函数 `group_gemm_fp8_kernel`）。特别是 `warpgroup_wait<0>()` 自身的 inline asm 没有 `tCr` operand；仅靠它的 memory clobber，编译器不一定能把“这 24 个异步产生的寄存器”与后续 C++ tensor 访问建立完整的寄存器级边界，因此 CUTLASS 再显式调用 operand fence。

这里的“开始/结束”是编译器调度边界，不表示这条空 asm 在硬件上开启或关闭 WGMMA pipeline。真正发起运算的是 `src/group_gemm/kernels.cuh:488`（函数 `group_gemm_fp8_kernel`）的 `cute::gemm`，它最终展开为 `wgmma.mma_async`；`warpgroup_arrive` 只发出运算前的硬件寄存器排序 fence，`warpgroup_commit_batch` 定义异步 group 的提交边界，`warpgroup_wait` 等待已提交 group。三个 wrapper 的实现见 `3rd/cutlass/include/cute/arch/mma_sm90_gmma.hpp:47-84`（函数 `warpgroup_arrive`、`warpgroup_commit_batch`、`warpgroup_wait`）。

四个内层 `cute::gemm` 使用同一个形状 `m64n48k32` 和同一组 `tCr` accumulator。对这种连续、同形状的 accumulator WGMMA，硬件允许连续更新同一 accumulator，不需要在每两条 WGMMA 之间再执行 `wgmma.fence`；同时，两条 WGMMA 之间没有普通 C++ 代码读写 `tCr`，所以也没有必要重复加入 compiler operand marker。这里必须区分两种边界：空的 `warpgroup_fence_operand` 不会直接切分硬件 pipeline，硬件 async batch 由 `warpgroup_commit_batch` 定义。当前 helper 放在 batch 开始和 `wait_group` 之后，覆盖“普通寄存器访问 -> 异步 WGMMA batch -> 普通寄存器访问”三个阶段。当前 operation 是 `SS_TN`，A/B 数据来自 shared-memory descriptor，`tCr` 是需要重点标记的寄存器 accumulator；这也是代码只对 `tCr` 调用该 helper 的原因。


### 12.7.6 用“寄存器使用权交接”理解两个调用

这是一个帮助阅读代码的抽象模型，不是额外的 CUDA barrier：

1. 在 `src/group_gemm/kernels.cuh:484`（函数 `group_gemm_fp8_kernel`）之前，普通 C++/编译器代码仍可能拥有 `tCr` 的 live range；前置 operand marker 把“最后一次普通访问”标在 WGMMA batch 之前。紧接的 `src/group_gemm/kernels.cuh:485`（函数 `group_gemm_fp8_kernel`）才发出硬件 `wgmma.fence.sync.aligned`，满足 WGMMA 对相关寄存器顺序的要求。
2. 在 `src/group_gemm/kernels.cuh:488-490`（函数 `group_gemm_fp8_kernel`）期间，4 条 `wgmma.mma_async` 连续把同一组 24 个 accumulator 当作目的寄存器更新；这段区间没有普通 C++ 读取 `tCr`。`tiled_mma.accumulate_` 的赋值只改变下一条 WGMMA 的 `scale_D` 控制，不是对 `tCr` 做普通寄存器访问。
3. `src/group_gemm/kernels.cuh:492-493`（函数 `group_gemm_fp8_kernel`）先提交 batch，再用 `warpgroup_wait<0>()` 等待异步结果完成。**硬件“结果可读”由 wait 提供，不能由 operand marker 提供。**
4. `src/group_gemm/kernels.cuh:494`（函数 `group_gemm_fp8_kernel`）的 post marker 位于 wait 之后、`src/group_gemm/kernels.cuh:499-500`（函数 `group_gemm_fp8_kernel`）第一次普通读取 `tCr(i)` 之前；它把 compiler-visible 的 accumulator 定义与后续 C++ use 接起来。把这个调用提前到 wait 之前，不能替代当前顺序，因为 marker 就失去了“完成后再允许普通 use”的位置含义。
5. `src/group_gemm/kernels.cuh:496`（函数 `group_gemm_fp8_kernel`）的 `arrive_barrier(writable[ismem_read])` 是 shared-memory stage 的生产者/消费者协议，和 `tCr` 寄存器 marker 是两条独立的同步链路；前者不读取或刷新 `tCr`。

### 12.7.7 如果删除 `src/group_gemm/kernels.cuh:494`（函数 `group_gemm_fp8_kernel`），会发生什么

删除 operand fence **通常没有硬件同步或数学正确性含义**：`src/group_gemm/kernels.cuh:493`（函数 `group_gemm_fp8_kernel`）的硬件 `warpgroup_wait<0>()` 仍然等待全部已提交 group 完成，所以某些编译器版本下测试仍会通过。风险主要在编译器生成代码和性能：NVVM/ptxas 对“哪些寄存器属于 WGMMA in-flight batch、何时允许普通指令触碰这些 live range”的信息变得不完整，可能出现：

- 为了满足隐含的寄存器依赖而插入额外等待，多个 WGMMA 被串行化；
- ptxas 报告类似“non-WGMMA instructions defining accumulator registers between the start and end of the pipeline stage”的性能警告；
- 寄存器压力较高时发生额外 register move/reload，削弱 4 条 WGMMA 的异步重叠。

因此它主要是性能和编译器调度方面的标记，不是清零 `tCr`、不是 `ScaleOut::Zero`，也不是线程同步原语。对本次 kernel，最准确的职责分工是：

| 原语 | 是否生成硬件指令 | 主要职责 |
|---|---:|---|
| `warpgroup_fence_operand(tCr)` | 否（空 asm） | 给 NVVM 标记 24 个 accumulator register 的代码移动边界 |
| `warpgroup_arrive()` | 是，`wgmma.fence.sync.aligned` | 排序此前 accumulator（必要时的寄存器 A fragment）访问与后续 WGMMA；不负责 shared-memory 可见性 |
| `warpgroup_commit_batch()` | 是，`wgmma.commit_group.sync.aligned` | 把已发出的 WGMMA 组成可等待 batch |
| `warpgroup_wait<0>()` | 是，`wgmma.wait_group.sync.aligned 0` | 等待 batch 完成，之后才可读取 accumulator |



```cpp
arrive_barrier(writable[ismem_read]);
```

这一步发生在 WGMMA 完成之后，含义是“当前 math warpgroup 已经不再读取这个 A/B shared-memory stage”。`writable[]` 在 `src/group_gemm/kernels.cuh:293-297`（函数 `group_gemm_fp8_kernel`）初始化时的 arrival count 是 `size(TiledMma{})`；本次是 `256`，两个 128-thread math warpgroup 的所有线程都要 arrive。load warpgroup 在 `src/group_gemm/kernels.cuh:400-410`（函数 `group_gemm_fp8_kernel`）下一次复用该 stage 前会执行 `wait_barrier(writable[ismem_write], phase)`，因此不会覆盖仍可能被 WGMMA 读取的 shared memory。

这也解释了两类 barrier 的分工：

- `readable[stage]`：TMA global-to-shared transaction 完成，消费者可以开始 WGMMA；
- `writable[stage]`：消费者 WGMMA 完成，生产者可以再次写入该 stage。

## 12.8 跨 K tile 累加、stage 轮转

```cpp
#pragma unroll
for (int i = 0; i < size(tCr); ++i) {
  tDr(i) = tCr(i) * scale + tDr(i);
}
```

位置：`src/group_gemm/kernels.cuh:498-501`（函数 `group_gemm_fp8_kernel`）。这里每个线程只更新自己拥有的 `tCr/tDr` fragment 元素，不需要线程间归约。对本次运行：

- `k=7168`、`kTileK=128`，外层循环共有 `56` 次；
- 每次外层循环的 `tCr` 是该 128-K tile 的局部 FP32 结果；
- `tDr` 保存从第一个 K tile 到当前 K tile 的完整 FP32 结果；
- `scale = yscale_ptr[igroup]` 是当前 group 的输出 scale，先乘局部结果，再加到 `tDr`。

所以完整计算是：

```text
tDr = sum_{itile_k=0..55} (tCr[itile_k] * yscale_ptr[igroup])
```

这里的软件累加不能由 `accumulate_=One` 代替：`accumulate_` 只控制同一个外层 K tile 内的 4 条 `K=32` WGMMA。

```cpp
++ismem_read;
if (ismem_read == kStage) {
  phase ^= 1;
  ismem_read = 0;
}
```

位置：`src/group_gemm/kernels.cuh:503-507`（函数 `group_gemm_fp8_kernel`）。本次 `kStage=8`，所以 `readable[0..7]` 和 `writable[0..7]` 构成环形缓冲。索引绕回 0 时翻转 `phase`，用同一个 barrier 对象区分第 0、1、2... 轮 transaction；否则第 9 次使用 stage 0 时会误把第 1 次的完成状态当成第 9 次。

## 12.9 FP32 结果转换为 BF16

```cpp
auto tCrh = make_tensor_like<cute::bfloat16_t>(tCr);

#pragma unroll
for (int i = 0; i < size(tCr); ++i) {
  tCrh(i) = (Tout)(tDr(i));
}
```

位置：`src/group_gemm/kernels.cuh:510-515`（函数 `group_gemm_fp8_kernel`）。`tCrh` 与 `tCr` 具有相同的线程分片和布局，但元素类型从 FP32 变为 `cute::bfloat16_t`；`Tout` 在本次 pertensor kernel 中就是 BF16。转换放在所有 56 个 K tile 累加完成之后，因此不会在每个 K tile 都提前舍入，减少中间精度损失。此时 `tCrh` 仍然是寄存器 tensor，还没有写入 shared memory。

## 12.10 寄存器到 shared memory 的 epilogue

```cpp
auto sCT =
    make_tensor(make_smem_ptr(reinterpret_cast<Tout *>(shm_c)), SLayoutCT{});
```

位置：`src/group_gemm/kernels.cuh:518-520`（函数 `group_gemm_fp8_kernel`）。`shm_c` 是动态 shared memory 中的输出缓冲区，`sCT` 只是给它附加 BF16 元素类型和 `SLayoutY` swizzle 布局；本次 tile 的逻辑容量是 `kTileN*kTileM=128*48=6144` 个 BF16。这里没有分配新内存，也没有发生 copy。

```cpp
using STSM_ATOM =
    std::conditional_t<kTileM == 8, cute::SM90_U16x4_STSM_T, cute::SM90_U16x8_STSM_T>;
using R2SCopyAtomC = Copy_Atom<STSM_ATOM, Tout>;
auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
auto thr_copy_c = tiled_copy_c.get_slice(idx);
```

位置：`src/group_gemm/kernels.cuh:521-525`（函数 `group_gemm_fp8_kernel`）。

- `STSM_ATOM` 是寄存器到 shared-memory 的 STSM copy atom，不是 TMA atom。当前 `kTileM=48`，所以实例化 `SM90_U16x8_STSM_T`；只有 `kTileM=8` 的窄 tile 使用 `U16x4`。
- `R2SCopyAtomC` 把该硬件 atom 的元素类型固定为 `Tout=BF16`。
- `make_tiled_copy_C` 按 `tiled_mma` 的 C 布局把 atom 扩展成覆盖整个 `128x48` 输出 tile 的 `TiledCopy`。日志中该对象见 `temp/run.log:214-220`（调试输出）。
- `get_slice(idx)` 只生成当前线程的 copy 视图；它不执行 STSM。

```cpp
auto tCr4s = thr_copy_c.retile_S(tCrh);
auto tCs4r = thr_copy_c.partition_D(sCT);
```

位置：`src/group_gemm/kernels.cuh:527-528`（函数 `group_gemm_fp8_kernel`）。

- `retile_S` 把每个线程的 BF16 寄存器 fragment `tCrh` 重解释成 STSM atom 所需的 source 布局；
- `partition_D` 把 `sCT` 分割成当前线程应写的 shared-memory destination；
- 两个操作都只是 tensor view/layout 变换，真正的寄存器到 shared 写入发生在后面的 `cute::copy(tiled_copy_c,...)`。

## 12.11 TMA store 的流水线顺序

```cpp
tma_store_wait<0>();
syncwarpgroup(iwarpgroup);

cute::copy(tiled_copy_c, tCr4s, tCs4r);
syncwarpgroup(iwarpgroup);
cute::tma_store_fence();
```

位置：`src/group_gemm/kernels.cuh:530-535`（函数 `group_gemm_fp8_kernel`）。

### 12.11.1 `tma_store_wait<0>()`：等待上一轮释放 shared source

`tma_store_wait<Count>` 的实现见 `3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:1245-1259`（函数 `tma_store_wait`），生成：

```text
cp.async.bulk.wait_group.read 0
```

它等待本线程此前已经 `commit_group` 的 S2G TMA store，直到 pending store 数降到 0；`read` 语义重点是 TMA engine 已经完成对 shared-memory source 的读取。全局写入的尾部可能仍在进行，所以该等待点可以尽早复用 `shm_c`，形成计算/写回重叠。

本行位于本次输出 tile 的 epilogue 开头，等待的是“上一个输出 tile”在当前 leader 线程上提交的 store，而不是当前尚未发出的 store。因为本轮马上要用 `cute::copy(tiled_copy_c,...)` 覆盖同一个 `shm_c`；若不先等，上一轮 TMA 仍可能从 `shm_c` 取数，就会和本轮 STSM 写入发生 source/read 与 write 竞争。

`wait_group` 是每个发起 TMA 的线程自己的 bulk-group 状态，不是 `readable[]` 那种 shared-memory mbarrier。代码让所有 math 线程都执行 wait，再用下一行的 128-thread barrier 把它们汇合；只有 leader 实际有此前提交的 store，其他线程通常没有待等待的 S2G group。

### 12.11.2 第一个 `syncwarpgroup`：让整个 math warpgroup 一起复用 `shm_c`

`syncwarpgroup(iwarpgroup)` 的实现见 `src/utils/utils.cuh:605-607`（函数 `syncwarpgroup`），发出 `barrier.cta.sync <barrier_id>, 128`。它把 `iwarpgroup` 作为 0/1 两个独立 barrier id：

- warpgroup 0 的 128 个线程在 barrier 0 汇合；
- warpgroup 1 的 128 个线程在 barrier 1 汇合。

由于上一行的 `wait_group` 不是 CTA-wide mbarrier，这个汇合点保证 leader 已经完成旧 TMA source-read wait 后，其他线程才开始写 `shm_c`。

### 12.11.3 STSM：register -> shared

```cpp
cute::copy(tiled_copy_c, tCr4s, tCs4r);
```

位置：`src/group_gemm/kernels.cuh:533`（函数 `group_gemm_fp8_kernel`）。这不是 TMA copy，而是由 `STSM_ATOM` 产生的同步寄存器到 shared-memory 写入：`tCr4s` 是 BF16 寄存器 source，`tCs4r` 是 `shm_c` destination。每个线程只写自己的 fragment，所有 math 线程合起来填满当前 `128x48` tile。

```cpp
syncwarpgroup(iwarpgroup);
```

位置：`src/group_gemm/kernels.cuh:534`。第二个 warpgroup barrier 确保所有 STSM 写入都已完成；只有这样 leader 才能让 TMA engine 读取完整、稳定的 `shm_c` tile。

```cpp
cute::tma_store_fence();
```

位置：`src/group_gemm/kernels.cuh:535`（函数 `group_gemm_fp8_kernel`）。该 helper 在 `3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:1212-1221`（函数 `tma_store_fence`）生成 `fence.proxy.async.shared::cta`。STSM 使用普通 shared-memory proxy，而 TMA store 使用 async shared proxy；此 fence 建立两种 proxy 之间的可见性/顺序关系。它不是 `mbarrier.arrive.expect_tx`，也不负责等待 global write 完成。

## 12.12 形成 TMA destination view，并由 leader 发起 store

```cpp
if (is_leader_in_warpgroup) {
  auto gD = tma_d.get_tma_tensor(make_shape(n, m));
  auto btma_d = tma_d.get_slice(0);

  auto tDs = btma_d.partition_S(sCT);
  auto tDg = btma_d.partition_D(gD);

  auto *td_y = td_xy + igroup * 2 + 1;
  cute::copy(tma_d.with(td_y), tDs(_, iwarpgroup, Int<0>{}),
             tDg(_, itile_n * 2 + iwarpgroup, itile_m));
  tma_store_arrive();
}
```

位置：`src/group_gemm/kernels.cuh:537-547`（函数 `group_gemm_fp8_kernel`）。

### 12.12.1 为什么只让 leader 发一条 TMA 指令

`is_leader_in_warpgroup` 在 `src/group_gemm/kernels.cuh:243-246`（函数 `group_gemm_fp8_kernel`）中由 `elect_one_sync()` 选出，并限制 `iwarp % 4 == 0`。因此每个 128-thread math warpgroup 只有一个线程进入本段；本次两个 math warpgroup 各发一条 TMA store，而不是 256 个线程各发一条。

### 12.12.2 `gD`、`btma_d`、`tDs`、`tDg` 各自是什么

```cpp
auto gD = tma_d.get_tma_tensor(make_shape(n, m));
```

位置：`src/group_gemm/kernels.cuh:538`（函数 `group_gemm_fp8_kernel`）。这是根据 `tma_d` 中保存的 stride 和 TMA box 形状生成的 coordinate tensor。它提供全局 Y 的坐标空间 `(n,m)=(4096,576)`，但在 grouped GEMM 中，当前 group 的真实 base address 由后面替换进去的 `td_y` descriptor 决定；因此 `gD` 本身不重新读取或复制 Y。

```cpp
auto btma_d = tma_d.get_slice(0);
```

位置：`src/group_gemm/kernels.cuh:539`（函数 `group_gemm_fp8_kernel`）。TMA store 的 atom 只有一个 issuing thread，所以取 slice 0 生成单一 TMA source/destination 映射。

```cpp
auto tDs = btma_d.partition_S(sCT);
auto tDg = btma_d.partition_D(gD);
```

位置：`src/group_gemm/kernels.cuh:541-542`（函数 `group_gemm_fp8_kernel`）。

- `tDs` 是 shared-memory source view。日志 `temp/run.log:257-258` 显示它包含 TMA atom、warpgroup 子块和单 stage 维度；`tDs(_, iwarpgroup, Int<0>{})` 选出当前 math warpgroup 已写好的 `64x48` 子块。
- `tDg` 是 destination coordinate view，不是一个会被普通 CUDA load/store 解引用的 global pointer array。日志 `temp/run.log:259-260` 显示它的有效坐标维度为 `64`、`12`；调用时的 `itile_n*2+iwarpgroup` 和 `itile_m` 选择 N/M tile 坐标。

本次 `CopyBoxY` 的定义见 `src/group_gemm/config.h:85-95`（类型 `GroupGEMMFp8Config`、函数 `get_tma`）：`kTileN/kWarpgroupM=128/2=64`，`kTileM=48`，所以每个 TMA store 搬运一个 `64x48` BF16 box。两个 warpgroup 的 `iwarpgroup=0,1` 合计覆盖一个 `128x48` 输出 tile；日志中 TMA Y tiler 也为 `(_64,_48)`，见 `temp/run.log:63-71`。

`td_xy` 每个 group 占两个 descriptor 槽位：`update_grouped_tma` 在 `src/group_gemm/kernels.cuh:180-211`（函数 `update_grouped_tma`）先把 X descriptor 放在 `td_xy + igroup*2`，再把 Y descriptor 放在 `td_xy + igroup*2 + 1`。所以 line 544 的 `+1` 不是 tile 坐标，而是在 descriptor 数组中选择当前 group 的 Y base address/shape/stride。


## 12.13 为什么 TMA store 不需要传 shared-memory barrier

load 侧的代码在 `src/group_gemm/kernels.cuh:404-410`（函数 `group_gemm_fp8_kernel`）是：

```cpp
cute::copy(tma_a.with(td_x, readable[ismem_write]), ...);
cute::copy(tma_b.with(readable[ismem_write]), ...);
set_barrier_transaction_bytes(readable[ismem_write], kTransactionBytes);
```

这是 G2S TMA load。其底层指令带有 `mbarrier::complete_tx::bytes`，TMA engine 完成 global-to-shared transaction 后要更新 `readable[]`；因此 `.with` 必须把 descriptor 和 shared-memory mbarrier 一起提供，随后用 `mbarrier.arrive.expect_tx.shared::cta` 设置期望字节数。相关实现见：

- `3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:103-131`（G2S TMA copy）；
- `3rd/cutlass/include/cute/arch/copy_sm90_desc.hpp:75-85`（函数 `set_barrier_transaction_bytes`）。

而本行：

```cpp
cute::copy(tma_d.with(td_y), tDs(_, iwarpgroup, Int<0>{}),
           tDg(_, itile_n * 2 + iwarpgroup, itile_m));
```

是 S2G TMA store。当前 CuTe 路径生成的二维 PTX 是：

```text
cp.async.bulk.tensor.2d.global.shared::cta.bulk_group
    [descriptor, {coord0, coord1}], [smem_ptr]
```

实现见 `3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:980-1000`（函数 `SM90_TMA_STORE_2D::copy`）。该指令没有 `mbarrier::complete_tx::bytes` 操作数，因此不存在让 `tma_d.with` 接收 `readable[]` 的必要；store 的 `.with` 只需替换 descriptor pointer。

这里的 `.with(td_y)` 对应的重载是 `Copy_Traits<SM90_TMA_STORE, ...>::with(TmaDescriptor const*)`，见 `3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:393-398`（函数 `Copy_Traits::with`）。它返回 `SM90_TMA_STORE_PTR`，后续 `copy_unpack` 只提取 descriptor pointer、shared source pointer 和 destination coordinate，见同文件 `3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:441-464`（函数 `copy_unpack`）。

因此准确说法不是“TMA store 不需要 barrier”，而是它使用另一套同步协议：

| 目的 | G2S load | S2G store |
|---|---|---|
| TMA 完成通知 | `readable[]` transaction mbarrier | per-thread bulk async group |
| 发起后的提交 | `mbarrier.arrive.expect_tx` | `tma_store_arrive()` = `cp.async.bulk.commit_group` |
| 等待/允许 source 复用 | math 侧 `wait_barrier` | 下一轮 `tma_store_wait<0>()` |
| 普通 shared 写入对 TMA 可见 | load 侧由 TMA transaction 本身处理 | STSM 后显式 `tma_store_fence()` |
| warpgroup 内生产者协作 | `writable/readable` mbarrier | `syncwarpgroup` + leader 发起 |

也就是说，S2G store 仍然需要同步，只是没有使用 load 的 transaction-completion mbarrier。当前代码用 `tma_store_wait` 保护 `shm_c` 的复用，用 `syncwarpgroup` 保证 128 个线程完成 STSM，用 `tma_store_fence` 建立普通 shared proxy 到 TMA async proxy 的顺序，再用 `tma_store_arrive` 提交当前 store。

## 12.14 为什么先 `wait`，后 `arrive`：按“上一轮/当前轮”理解

把一个输出 tile 的 epilogue 和下一 tile 的开头串起来，实际时间线是：

```text
tile i:
  wait(tile i-1 的已提交 store)
  registers -> shm_c
  fence
  issue tile i 的 TMA store
  commit tile i 的 store

tile i+1:
  wait(tile i 的已提交 store)
  覆盖 shm_c
  ...
```

代码对应：

- `src/group_gemm/kernels.cuh:530`：本轮开头 `tma_store_wait<0>()`（函数 `group_gemm_fp8_kernel`）；
- `src/group_gemm/kernels.cuh:545-546`：本轮 `cute::copy(tma_d.with(...))`（函数 `group_gemm_fp8_kernel`）；
- `src/group_gemm/kernels.cuh:547`：本轮 `tma_store_arrive()`（函数 `group_gemm_fp8_kernel`）。

`tma_store_arrive()` 的实现见 `3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:1223-1232`（函数 `tma_store_arrive`），实际发出：

```text
cp.async.bulk.commit_group
```

`commit_group` 只提交“当前线程此前已经发出、但尚未提交”的 bulk async copies。因此它必须位于当前 `cute::copy` 之后，才能把当前 tile 的 TMA store 放进下一轮 `wait_group.read 0` 所跟踪的队列。

### 12.14.1 如果把 `arrive` 放到当前 `copy` 前

在当前 tile 尚未发出 TMA copy 时先执行 `commit_group`，提交的 group 不包含当前 copy；随后发出的 copy 会留在未提交状态。下一轮的 `tma_store_wait<0>()` 只等待已提交 group，不能可靠地保护 `shm_c` 不被下一轮 STSM 覆盖，形成 source-read/write 竞争。除非再额外补一个正确位置的 `commit_group`，否则语义就是错误的。

### 12.14.2 如果在当前 copy 后立即交换成“arrive 再 wait”

若把当前 copy 后的 `arrive` 和下一轮开头的 `wait` 紧挨着执行，通常可以得到正确但更保守的代码：当前 store 会在同一轮立刻被等待，TMA global write 的延迟不能与下一轮计算重叠。当前实现把 `arrive` 留在本轮末、把 `wait` 放到下一轮开头，正好实现“一轮延迟”的流水线。

所以这两个调用不能按文本位置简单互换：

- `wait` 负责释放上一轮占用的 shared source；
- `arrive` 负责提交本轮刚发出的 TMA store；
- 二者分别服务于不同的异步传输，时间上必须是“上一轮 wait，当前轮 issue+arrive”。

仓库中相同的推荐顺序也出现在 `src/gemm/sm90/gemm_bf16xfp32.cu:335-363`（函数 `gemm_bf16xfp32_kernel`）：先 `tma_store_wait<0>()`，再 STSM、`tma_store_fence()`、TMA copy，最后 `tma_store_arrive()`。

## 12.15 本次 shape 的一次完整迭代

以一个 `(igroup,itile_m,itile_n)` 为例：

1. `wait_barrier(readable[stage], phase)` 等待 A/W 的 `128x128` 和 `48x128` TMA load 完成；
2. `ScaleOut::Zero` 清除当前 `tCr` 的旧 C 参与；
3. 4 次 WGMMA 分别处理 4 个 `K=32` 子块，第一次 Zero、后三次 One；
4. `commit_batch + wait<0>` 等待 4 条异步 WGMMA 真的完成；
5. `arrive_barrier(writable[stage])` 允许 load warpgroup 以后复用该 A/W stage；
6. `tDr += tCr * yscale_ptr[igroup]` 累加到完整 7168-K 结果；
7. 56 次外层 K 循环结束后，FP32 `tDr` 转为 BF16 `tCrh`；
8. 本轮开始的 `tma_store_wait<0>()` 释放上一个输出 tile 的 `shm_c` source；
9. 两个 math warpgroup 通过 STSM 各写一个 `64x48` 子块；
10. 两个 leader 各发一条 S2G TMA store，`tma_store_arrive()` 提交它们，合起来写完整 `128x48` 输出 tile。

math 分支结束后，循环回到下一项任务；本次 pertensor kernel 的 PDL 收尾位于 `src/group_gemm/kernels.cuh:633-635`（函数 `group_gemm_fp8_kernel`）。`cudaTriggerProgrammaticLaunchCompletion()` 是 programmatic dependency 的调度触发点，不是 `readable[]` transaction barrier，也不能替代需要时的显式 TMA wait；循环末尾不再复用 `shm_c`，所以代码没有为下一次 shared-buffer 写入再增加一个 wait。

## 12.16 直接结论

- `tiled_mma.accumulate_` 不改变 FP8 A/B 输入；它对应 GMMA 公式中的 `scale_D*C`，控制本条 WGMMA 是否把旧的 C accumulator 加入结果。对本次 `kTileK=128`，4 条 `K=32` WGMMA 的模式必须是 `Zero, One, One, One`。
- `tma_d.with(td_y)` 不是“没有同步”，而是没有 load 所用的 transaction mbarrier 参数。S2G 指令用 `bulk_group`，由 `tma_store_arrive` 提交、下一轮 `tma_store_wait<0>` 等待；STSM 前后还需要两个 `syncwarpgroup` 和一个 `tma_store_fence`。
- `tma_store_wait<0>` 保护上一轮 `shm_c` source 的复用；`tma_store_arrive` 提交当前刚发出的 store。正确的流水线顺序是“上一轮 wait -> 当前轮写 shared/fence/copy/arrive”，不能把两者当成同一轮的可交换调用。

## 12.17 能否把 `tDr(i) * scale` 移到 `itile_k` 循环之后

先给直接结论：**可以提取这个公共 scale，但不能只删除循环内的整条更新语句。** 如果循环内不再把 `tCr` 加到 `tDr`，而循环结束后只执行

```cpp
for (int i = 0; i < size(tDr); ++i) {
  tDr(i) = tDr(i) * scale;
}
```

那么结果不正确。`tDr` 在每个 output task 开始时由 `make_tensor_like` 创建并在 `clear(tDr)` 清零，见 `src/group_gemm/kernels.cuh:472-475`（函数 `group_gemm_fp8_kernel`）；如果中间没有任何写入，最后乘法仍然只是 `0 * scale`。

反过来，如果保留循环内原来的 `tCr(i) * scale`，又在循环结束后乘一次 `scale`，就会把同一个因子应用两遍，结果变成 `scale^2 * Σ_j p_j`，同样不正确。

### 12.17.1 必须采用的改写

要提取 scale，`itile_k` 循环内仍须进行**未缩放的累加**，循环结束后再统一缩放：

```cpp
#pragma unroll 1
for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
  // wait + WGMMA + wait_group，与原代码相同
#pragma unroll
  for (int i = 0; i < size(tDr); ++i) {
    tDr(i) = tDr(i) + tCr(i);
  }
}

#pragma unroll
for (int i = 0; i < size(tDr); ++i) {
  tDr(i) = tDr(i) * scale;
}
```

原始的 K 循环、WGMMA 完成等待和 `tDr` 更新位于 `src/group_gemm/kernels.cuh:477-501`（函数 `group_gemm_fp8_kernel`）。实际改代码时，`tDr(i) + tCr(i)` 仍应放在每次 `warpgroup_wait<0>()` 和 `warpgroup_fence_operand(tCr)` 之后；统一乘法应放在外层循环结束、`tCrh` 的 BF16 转换之前，即 `src/group_gemm/kernels.cuh:508-515`（函数 `group_gemm_fp8_kernel`）之间。不能在 `warpgroup_wait<0>()` 之前读取 `tCr`，也不能等转换成 BF16 后才乘 scale。

### 12.17.2 为什么当前 per-tensor 路径可以提出公因子

令第 `j` 个 K tile 的 `tCr(i)` 为 `p_j(i)`。当前代码的逻辑是：

```text
D_0(i) = 0
D_{j+1}(i) = D_j(i) + p_j(i) * scale
```

`scale` 在进入 K 循环前只读取一次：`float scale = yscale_ptr[igroup]`，见 `src/group_gemm/kernels.cuh:472-479`（函数 `group_gemm_fp8_kernel`）。在一个 task 的整个 `itile_k` 循环中，`igroup` 不变，因此这个 scale 不依赖 `itile_k` 或元素下标 `i`。host 入口也把 `y_scale` 检查为 FP32，并把它作为 `yscale_ptr` 传入 kernel，见 `src/group_gemm/entry.cc:17-20`、`:33`、`:99-112`（函数 `group_gemm_fp8_entry`）。

所以在**实数算术**下：

```text
Σ_j (p_j(i) * scale) = (Σ_j p_j(i)) * scale
```

这里还有一个必要条件：`tDr` 初值必须是零。若初值是 `D0 != 0`，原式是 `D0 + scale * Σp_j`，改写式却是 `scale * (D0 + Σp_j)`，除非 `scale == 1`，两者并不相等。当前 `clear(tDr)` 正好满足这个条件，而且 `tDr` 在 while 的每个 task 内重新创建，不会把上一个 task 的结果带进来，见 `src/group_gemm/kernels.cuh:447-475`（函数 `group_gemm_fp8_kernel`）。

本次运行的日志显示 `tAg` 的 K 维 tile 数为 `56`，每线程的 `tCr` 和 `tDr` 都有 `24` 个 FP32 元素，见 `temp/run.log:162-207`（函数 `group_gemm_fp8_kernel` 的调试输出）。因此当前每线程执行 `56 * 24 = 1344` 次“缩放并累加”，改写后是 `1344` 次“只累加”加 `24` 次最终缩放。`size(tDr)` 与 `size(tCr)` 相同是 `make_tensor_like(tCr)` 的直接结果。

测试也明确使用了 per-group 公共 scale：参考路径传 `scale_a=0.5`、`scale_b=0.5`，HPC 路径传每 group 的 `scale_hpc=0.5*0.5=0.25`，见 `tests/test_group_gemm_pertensor_like.py:39-40`、`:57-66`（函数 `naive_group_gemm_pertensor_fp8`、`test_group_gemm_pertensor_fp8`）。对这一数据接口，提出公因子的数学前提成立。

### 12.17.3 “数学正确”不等于“逐位相同”

两种写法的 FP32 舍入点不同。原写法每个 K tile 都先做乘法再累加：

```text
old_j = round(old_{j-1} + round(p_j * scale))
```

而改写后先累加未缩放值，最后只舍入一次乘法：

```text
sum_j = round(sum_{j-1} + p_j)
new   = round(sum_last * scale)
```

在 CUDA 编译器允许乘加融合时，原表达式还可能生成一条 FP32 FMA（一次舍入）；改写后通常是多次 FP32 add 加一次 FP32 multiply。因而不能承诺 `torch.equal` 或 bitwise identical，即使 `scale=0.25` 是 2 的负幂也不能对所有 subnormal、边界值和舍入情形作此承诺。

极端情况下两者还可能在以下方面不同：

- **舍入和消去误差**：先缩放再相加与先相加再缩放，对正负 partial 的消去顺序不同；
- **范围**：改写后的未缩放 `tDr` 可能先溢出，而最终乘以小 scale 后本来可以落回有限范围；反过来，原写法也可能把很小的 partial 先下溢为零；
- **特殊值**：NaN、Inf 和 signed zero 的传播也不保证一致。

结果在最后才从 FP32 `tDr` 转成 BF16，见 `src/group_gemm/kernels.cuh:510-515`（函数 `group_gemm_fp8_kernel`）。BF16 的较低精度通常会掩盖一部分 ULP 差异，但不能把两种 FP32 算法变成严格等价。

### 12.17.4 性能上不一定更快

“scale 乘法只剩 `size(tDr)` 次”这个观察是对的，但它没有消除 K 循环中的累加。对于本次 `ntile_k=56`、`size(tCr)=24`：

```text
原写法：1344 次 multiply-add（常见情况下可融合为 1344 次 FFMA）
改写后：1344 次 add + 24 次 multiply
```

所以在原表达式已经被融合为 FMA 时，改写反而可能多出最终的 24 条标量乘法，不能仅凭循环次数断定加速；`#pragma unroll` 也意味着 24 元素循环的控制开销通常已经被编译期展开。若编译选项或别名分析阻止 FMA，指令数才可能呈现“少很多 multiply”的另一种情况。最终应检查目标 `sm_90a` 的 SASS（`FFMA`、`FADD`、`FMUL`）并用 CUDA event/Nsight 做 A/B benchmark，而不是只看 C++ 循环文本。当前构建使用 Release/O3，编译选项见 `CMakeLists.txt:5`、`:105-115`（目标 `_C`）；这仍不保证每个版本的最终指令选择完全相同。

此外，WGMMA/TMA 的等待和 shared-memory 流水通常比这 24 个标量运算更昂贵；改写后的收益若存在，必须用完整 kernel 的端到端测量确认。若必须保持现有数值行为，应保留 `src/group_gemm/kernels.cuh:499-501`（函数 `group_gemm_fp8_kernel`）的原有乘加形式（最终是否融合为 FMA 由编译器决定）。

### 12.17.5 哪些同步位置不能随改写移动

- `src/group_gemm/kernels.cuh:480-494`（函数 `group_gemm_fp8_kernel`）中的 `wait_barrier`、`warpgroup_wait<0>` 和后置 `warpgroup_fence_operand(tCr)` 必须保持；统一缩放只操作已经完成的 `tDr`，不能用它替代 WGMMA 完成等待。
- `src/group_gemm/kernels.cuh:496`（函数 `group_gemm_fp8_kernel`）的 `arrive_barrier(writable[ismem_read])` 仍在每个 K tile 内，负责释放 A/B shared-memory stage；把累加改成未缩放不会改变这条生产者/消费者协议。
- 最终 `tDr *= scale` 必须在外层 K 循环结束后、`src/group_gemm/kernels.cuh:511-515`（函数 `group_gemm_fp8_kernel`）的 FP32->BF16 转换前；把它移到下一个 output task 或 TMA store 之后都会改变数据归属或精度。
- `scale` 只能在当前 task 的 K 循环范围内视为常量；不能把一次读取提升到整个 while 循环之外，因为下一个 task 的 `igroup` 可能不同，见 `src/group_gemm/kernels.cuh:447-475`（函数 `group_gemm_fp8_kernel`）。

### 12.17.6 不要套用到 blockwise 分支

这个结论仅针对 `group_gemm_fp8_kernel` 的 per-group scalar `yscale`。blockwise kernel 在每个 K tile 读取 `wscale = sBS(ismem_read, itile_k % 4)`，并在输出列循环中使用随列/当前 tile 变化的 `yscale`，见 `src/group_gemm/kernels.cuh:909-941`（函数 `group_gemm_blockwise_fp8_kernel`）。那里通常不存在一个可以提出到整个 K 循环外的单一 scale；照搬本节改写会把不同 K tile 的 scale 错误地合并。

### 12.17.7 最终判断

| 改法 | 结果判断 |
|---|---|
| 删除循环内更新，只在末尾 `tDr *= scale` | **错误**：`tCr` partial 没有累加进 `tDr` |
| 循环内 `tDr += tCr`，末尾 `tDr *= scale` | **实数算术正确**，但 FP32/BF16 输出不保证逐位相同；性能需实测 |
| 保留 `tDr = tCr * scale + tDr` | 保持当前数值舍入和已验证行为 |

对本次测试的 `scale=0.25` 和 `allclose(rtol=0.08, atol=0.01)`，第二种写法大概率仍能通过容差，测试断言见 `tests/test_group_gemm_pertensor_like.py:69-73`（函数 `test_group_gemm_pertensor_fp8`）；但如果验收要求 bitwise 一致或希望无风险地提升性能，不能只凭代数恒等式改动，应该先做两版本的正确性与性能 A/B 测试。

## 12.18 让 WGMMA 直接把结果写进 `tDr`

这个改法**原则上可以正确**，而且比“只把 scale 移到循环外”更有可能真正省掉指令；但它不是只把第 488 行的 `tCr` 改成 `tDr` 就结束了。`tDr` 一旦作为 `cute::gemm` 的第三个参数，就变成 WGMMA 的异步 C/D accumulator，必须同时调整 `ScaleOut`、`warpgroup_fence_operand` 和 accumulator 的生命周期。

### 12.18.1 先区分当前两个 tensor 的职责

当前实现中，`tCr` 是**一个 K tile 内部**的 WGMMA accumulator：每次 `itile_k` 先设置 `ScaleOut::Zero`，4 条 `K=32` 的 WGMMA 在同一个 `tCr` 上累加；`tDr` 则是**跨所有 K tile**的 software accumulator，WGMMA 完成后再由普通 CUDA 指令更新。完整代码见 `src/group_gemm/kernels.cuh:472-501`（函数 `group_gemm_fp8_kernel`）。数据流可以写成：

```text
当前：每个 itile_k
  tCr = WGMMA(partial_K_tile)
  tDr = tDr + tCr

改写：整个 output tile
  tDr = WGMMA(partial_0)
      + WGMMA(partial_1)
      + ...
      + WGMMA(partial_{ntile_k-1})
```

因此，直接累加时应删除 software `tDr += tCr` 循环，而不是删除“累加”这个动作本身。`tDr` 仍然必须由每条后续 WGMMA 作为 C 输入继续累加。

### 12.18.2 为什么 `tDr` 可以作为 WGMMA 的 C/D

`make_tensor_like(tCr)` 会创建与 `tCr` 相同布局和元素类型的 owning register tensor，见 `3rd/cutlass/include/cute/tensor_impl.hpp:417-443`（函数 `make_tensor_like`）。在本次实例中，日志显示每个线程的 `tCr`/`tDr` 都是 24 个 FP32 元素，见 `temp/run.log:204-207`（函数 `group_gemm_fp8_kernel` 的调试输出）。

CuTe 的 GMMA unpack 路径明确要求 D、C 都是 register tensor，并要求两者 value type 和 layout 相同；它把可写的 D 重解释为传给硬件的 `rC`，见 `3rd/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:398-430`（函数 `SM90::GMMA::mma_unpack`）。`cute::gemm(mma, A, B, C)` 的三参数重载明确转发为 `gemm(mma, C, A, B, C)`，见 `3rd/cutlass/include/cute/algorithm/gemm.hpp:77-89`（函数 `gemm`）；底层 `MMA_Atom::call` 也把三参数形式转发为 `call(C, A, B, C)`，见 `3rd/cutlass/include/cute/atom/mma_atom.hpp:107-118`（函数 `MMA_Atom::call`）。因此第三个参数同时充当 D 和 C，`tDr` 只要保持同一 C layout，下面的形式在类型/布局上就是合法的：

```cpp
cute::gemm(tiled_mma,
           tBr(_, _, ik, ismem_read),
           tAr(_, _, ik, ismem_read),
           tDr(_, _, _));
```

当前选择的 atom 的 `CRegisters` 正好是 `float[24]`，且 24 个目的寄存器都以 `+f` 传给 `wgmma.mma_async`，见 `3rd/cutlass/include/cute/arch/mma_sm90_gmma_ext.hpp:28810-28850`（类型 `MMA_64x48x32_F32E4M3E4M3_SS_TN`、函数 `MMA_64x48x32_F32E4M3E4M3_SS_TN::fma`）。这说明 `tDr` 不是“写回 global 的 tensor”，而是每个线程持有的那组 FP32 accumulator registers。

若追求最小寄存器占用，可以进一步让 `tDr` 直接由 `partition_fragment_C` 创建，并用它的 layout 生成 BF16 fragment：

```cpp
auto tDr = thr_mma.partition_fragment_C(gC);
auto tCrh = make_tensor_like<cute::bfloat16_t>(tDr);
```

这样可以避免同时保留一个只用于提供 layout 的 `tCr`。是否真的减少物理寄存器，要以编译后的寄存器/ spill 报告为准。

### 12.18.3 `ScaleOut` 必须是“整个 output tile 只初始化一次”

用户提出的方案是：

```cpp
clear(tDr);
tiled_mma.accumulate_ = GMMA::ScaleOut::One;
```

然后所有 `itile_k`、所有 `ik` 都使用 `tDr`。这在 `clear` 确实发生且 `tDr` 是零的前提下是正确的：第一条 WGMMA 计算 `A*B + 0`，后续 WGMMA 计算 `A*B + tDr`。`clear` 对 register tensor 的实现是把元素填成 `T{}`，见 `3rd/cutlass/include/cute/algorithm/clear.hpp:54-62`（函数 `clear`）；`ScaleOut` 的硬件谓词和 C/D 操作数见 `3rd/cutlass/include/cute/arch/mma_sm90_gmma_ext.hpp:28826-28850`（函数 `MMA_64x48x32_F32E4M3E4M3_SS_TN::fma`）。

更符合 CUTLASS 常见写法的方案是：不清零，**第一条** WGMMA 使用 `ScaleOut::Zero`，从第二条起永久使用 `ScaleOut::One`。这样第一条指令会忽略旧 C，且可以省掉 `clear(tDr)`；但必须保证 Zero 只出现在整个 output tile 的第一条 WGMMA 上。现有代码把 Zero 放在 `itile_k` 循环内的 `src/group_gemm/kernels.cuh:482`（函数 `group_gemm_fp8_kernel`），直接照搬会导致每个 K tile 都覆盖 `tDr`，最后只剩最后一个 K tile，结果错误。

两种初始化方式的条件如下：

| 初始化方式 | 第一条 WGMMA | 是否需要 `clear(tDr)` | 注意事项 |
|---|---|---:|---|
| 用户方案 | `ScaleOut::One`，C 已由 `clear` 置零 | 是 | `clear` 必须在第一条 WGMMA 前完成 |
| 推荐 accumulator 方案 | `ScaleOut::Zero` | 否（`ntile_k>0` 时） | Zero 只能使用一次，随后一直 One |

如果 `ntile_k==0` 也要返回确定的零结果，第二种方案仍需保留 clear 或单独处理空 K；用户方案因为显式 clear 自然覆盖这个边界。

### 12.18.4 一个可行的代码骨架

下面是“保留现有 task/barrier 结构、采用 `clear + One`”的骨架。它展示必须改变的 operand；`wait_barrier`、stage 索引和 TMA load 逻辑保持不变：

```cpp
auto tDr = make_tensor_like(tCr);
clear(tDr);
float scale = yscale_ptr[igroup];
tiled_mma.accumulate_ = GMMA::ScaleOut::One;

#pragma unroll 1
for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
  wait_barrier(readable[ismem_read], phase);

  // tDr 是异步 WGMMA 的目的/累加寄存器
  warpgroup_fence_operand(tDr);
  warpgroup_arrive();
#pragma unroll
  for (int ik = 0; ik < size<2>(tAr); ++ik) {
    cute::gemm(tiled_mma, tBr(_, _, ik, ismem_read),
               tAr(_, _, ik, ismem_read), tDr(_, _, _));
  }
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(tDr);

  arrive_barrier(writable[ismem_read]);
  // advance ismem_read/phase
}

#pragma unroll
for (int i = 0; i < size(tDr); ++i) {
  tDr(i) = tDr(i) * scale;
}
```

对应原代码位置是 `src/group_gemm/kernels.cuh:472-508`（函数 `group_gemm_fp8_kernel`）。要点是：

1. 删除每个 `itile_k` 的 `tiled_mma.accumulate_ = ScaleOut::Zero`；
2. 删除 `ik` 循环中每次都执行的 `ScaleOut::One` 赋值；
3. 把两处 `warpgroup_fence_operand(tCr)` 改成 `warpgroup_fence_operand(tDr)`，因为异步 destination 已经从 tCr 变为 tDr；
4. 保留每个 K tile 的 `warpgroup_commit_batch()`/`warpgroup_wait<0>()`，以及之后的 `arrive_barrier(writable[ismem_read])`；
5. 最后的 scale 必须在最后一次 wait 之后、`src/group_gemm/kernels.cuh:511-515`（函数 `group_gemm_fp8_kernel`）的 FP32->BF16 转换之前执行。

如果采用“Zero 一次、One 永久”的方案，则第一个 `cute::gemm` 前设置 Zero，第一条发出后切换到 One；切换点必须跨越 `itile_k` 循环保留，不能在下一轮外层循环重新置 Zero。

### 12.18.5 为什么 fence 也必须从 `tCr` 改成 `tDr`

`warpgroup_fence_operand` 不是数值清零，也不是等待 WGMMA 完成；它通过 tensor 重载逐元素调用 float/uint32 标量重载，而标量重载的空 inline asm `+f`/`+r` 约束和 `memory` clobber 只建立编译器的数据依赖。实现见 `3rd/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:45-66`（函数 `warpgroup_fence_operand(Tensor&)`）和 `3rd/cutlass/include/cute/arch/mma_sm90_gmma.hpp:86-103`（函数 `warpgroup_fence_operand(float&)`、`warpgroup_fence_operand(uint32_t&)`）。当前代码在 `src/group_gemm/kernels.cuh:484-494`（函数 `group_gemm_fp8_kernel`）标记的是 tCr；直接把 WGMMA destination 换成 tDr 后，继续标记 tCr 会让 compiler 看不到 tDr 的异步 live range。

- 前置 `warpgroup_fence_operand(tDr)`：把 `clear(tDr)` 及其它普通 tDr 定义放在第一条 WGMMA 之前；
- `warpgroup_arrive()`：发出 `wgmma.fence.sync.aligned`，排序相关寄存器访问与后续 WGMMA；
- 中间四条（本例）WGMMA：连续更新同一组 tDr registers；
- `warpgroup_wait<0>()`：等待 batch 完成，硬件结果才可读；
- 后置 `warpgroup_fence_operand(tDr)`：把异步写回后的 tDr 与后面的普通 scale、BF16 转换连接起来。

这些 wrapper 的底层实现见 `3rd/cutlass/include/cute/arch/mma_sm90_gmma.hpp:47-84`（函数 `warpgroup_arrive`、`warpgroup_commit_batch`、`warpgroup_wait`）。即使中间没有普通代码读取 tDr，保留现有“每批前后各一个 operand fence”的写法最稳妥；想进一步只保留整个序列首尾的 fence，需要单独检查 ptxas 诊断和生成代码，不能仅凭 C++ 数据流删掉。

### 12.18.6 wait 仍然不能删除

把 software 累加改成 WGMMA 累加，并不会改变 shared-memory stage 的所有权协议。每个 `itile_k` 的 WGMMA 仍然读取 `sA/sB` 当前 stage；在 `warpgroup_wait<0>()` 完成前，不能让 load warpgroup 通过 `arrive_barrier(writable[ismem_read])` 复用/覆盖该 stage。相关顺序位于 `src/group_gemm/kernels.cuh:480-496`（函数 `group_gemm_fp8_kernel`）。

因此不能因为 tDr 已经是跨 K 的 accumulator，就把每批的 `commit/wait` 改成只在最外层结束时调用。那会使后续 TMA load 可能覆盖仍被未完成 WGMMA 读取的 shared memory。若想减少 wait 次数，需要重新设计更深的 WGMMA/TMA pipeline 和 stage 数量，不是这次寄存器重定向的直接结果。

### 12.18.7 数值结果：实数上等价，浮点上不保证逐位一致

对当前 per-tensor 路径，`scale` 在 task 开始时由 `yscale_ptr[igroup]` 读取一次，见 `src/group_gemm/kernels.cuh:472-479`（函数 `group_gemm_fp8_kernel`），所以直接 accumulator 的实数结果是：

```text
Σ_j partial_j * scale = (Σ_j partial_j) * scale
```

但它与现有实现仍有两个舍入差异：

1. 现有实现每个 K tile 先得到独立的 tCr，再由普通 FP32 指令执行 `tDr = tCr * scale + tDr`；
2. 直接方案让 WGMMA 在 tDr 中跨 K tile 累加，最后才做一次 scale。

WGMMA 内部累加、software FP32 加法和最终乘法的舍入位置不同，因此不能承诺 bitwise identical。结果随后才转为 BF16，见 `src/group_gemm/kernels.cuh:510-515`（函数 `group_gemm_fp8_kernel`）；通常测试容差可以吸收小的 ULP 差异，但仍应实际比较 `max_abs_diff`、`max_rel_diff` 和 BF16 输出。

此外，直接方案让未缩放的总和一直留在 FP32 accumulator 中：它可能先溢出，而最终乘以小 scale 后理论上本可有限；scale 很小时也可能避免“每个 partial 先下溢”。这属于算法数值范围变化，不是 barrier 问题。

### 12.18.8 性能与寄存器压力

对本次日志的 `ntile_k=56`、每线程 24 个 accumulator，当前 software 累加大约执行 `56*24=1344` 次普通更新；直接 WGMMA 后，这 1344 次更新循环可以去掉，累加由 tensor core 的 C/D path 完成。相比之前仅移动 scale 的方案，这才是可能产生明显收益的地方。

但要同时检查以下反效果：

- `tDr` 必须作为 WGMMA 的 register operand；CuTe 在 `mma_unpack` 中对此有 `is_rmem` 静态要求，见 `3rd/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:398-401`（函数 `SM90::GMMA::mma_unpack`）；
- 如果仍保留无用的 `tCr`，会让两个 24-float fragment 同时占用寄存器，抵消收益；
- tDr 的 WGMMA live range 延长到整个 56-tile K 循环，可能触发 register move/reload 或 spill；
- `clear(tDr)` 是额外的普通寄存器写入，采用“Zero 首条 WGMMA”才可能把它也去掉；
- 生成指令和占用率必须看目标 `sm_90a` 的 ptxas 报告/SASS，仓库当前的 CUDA 编译选项（包括 `-Xptxas ... -v`）见 `CMakeLists.txt:105-115`（目标 `_C`）。

所以建议用三个版本做 A/B：原实现、`tDr` 直接 WGMMA 且 `clear+One`、`Zero 一次+One` 且无 clear；同时检查数值和 kernel 时间，不能只按 C++ 循环次数判断。

### 12.18.9 适用范围

这个直接 accumulator 改法适用于本节的 `group_gemm_fp8_kernel` per-group scalar scale，因为同一个 `igroup` 的所有 K tile 共用一个 `scale`，见 `src/group_gemm/kernels.cuh:475-479`（函数 `group_gemm_fp8_kernel`）。blockwise 分支在每个 K tile/输出列使用变化的 scale，见 `src/group_gemm/kernels.cuh:909-941`（函数 `group_gemm_blockwise_fp8_kernel`）；若仍需逐 K tile 乘不同 scale，就不能把所有结果无条件直接累加到同一个 tDr 后再统一乘一个 scale。

### 12.18.10 最终判断

| 改法 | 数学结果 | 工程判断 |
|---|---|---|
| 只把 gemm 的输出从 tCr 改成 tDr，但每个 `itile_k` 仍置 `ScaleOut::Zero` | 错误，前面 K tile 会被覆盖 | 不能这样改 |
| `clear(tDr)` + `ScaleOut::One` 一次 + 所有 WGMMA 写 tDr | 实数算术正确 | 可行；需把 fence 改为 tDr，并验证浮点差异/寄存器压力 |
| 首条 WGMMA `ScaleOut::Zero`，之后永久 `One`，所有 WGMMA 写 tDr | 实数算术正确 | 更接近 CUTLASS accumulator 用法，可能省掉 clear；需处理 `ntile_k==0` |
| 直接 WGMMA 写 tDr 后仍保留旧的 `tDr += tCr` 循环 | scale/partial 被重复累加 | 错误 |

因此，对用户给出的具体方案，答案是：**在 `clear(tDr)`、`ScaleOut::One` 只设置一次、所有 WGMMA 都以 tDr 为 D/C、并把 operand fence 改为 tDr 的前提下，结果在实数意义上正确；但不是逐位等价，也不能保证一定加速。**

## 12.19 上一次直接 accumulator 改动的验证结果

- 在 `src/group_gemm/kernels.cuh:472-507`（函数 `group_gemm_fp8_kernel`）让 WGMMA 直接累加到 `tDr`，并将 scale 移到 K 循环之后执行。
- 已提交：`fd6960e8a9ed821ee82bf10813df4b97d3e18753`。
- `bash temp//rebuild.sh` 成功，安装了 `dist/hpc_ops-0.0.1.dev0+gfd6960e-cp39-abi3-linux_x86_64.whl`。

### 12.19.1 测试结果

- `tests/test_group_gemm_pertensor_like.py`（函数 `test_group_gemm_pertensor_fp8`）执行失败，退出码 1，失败位置为 `tests/test_group_gemm_pertensor_like.py:73`。
- `temp/run.modifiy.log`：Mean Absolute Error 0.025967、Mean Relative Error 0.004456，最大绝对误差 1.0。
- 原因是累加顺序改变：旧实现每个 K tile 先缩放再软件累加；新实现先由 WGMMA 累加未缩放结果，最后统一缩放，浮点舍入结果不再满足当前 allclose。
- PTX 中每个 K tile 的软件 FMA 被移除，替换为循环末尾乘法；但相关 specialization 仍使用约 168 个寄存器，尚未证明有性能提升。

---

# 13. 最后一个 TMA store 与 LIKE_DEBUG 审计（2026-08-28）

## 13.1 本次 shape 和 policy=2 的任务空间

本次日志 `temp/run.log:261` 到 `:264` 给出：`m=576` 、`n=4096` 、`k=7168` 、`num_group=8`，每组长度为 `[16,32,48,64,80,96,112,128]`，`cu_seqlens=[0,16,48,96,160,240,336,448,576]`。日志 `temp/run.log:6` 显示这次实例的 `kTileM=48`，`kTileN=128`，`kTileK=128`，`temp/run.log:73` 显示 `gridDim.x=132` 。

`src/group_gemm/group_gemm_pertensor_fp8.cu:437` 到 `:443`（函数 `group_gemm_fp8_async`）对平均长度 72 选择 `kTileM=48`。因此：

- `ntile_k = 7168 / 128 = 56`；
- `num_tile_n = ceil(4096 / 128) = 32`；
- 各组 M tile 数为 `[1,1,1,2,2,2,3,3]`，累计 `cu_tiles=[0,1,2,3,5,7,9,12,15]`，所以 `total_m=15`；
- `src/group_gemm/kernels.cuh:45` 到 `:46`（函数 `get_next_tile_vert`）把一维 task index `iblock` 解码为 `itile_m_total=iblock%15` 和 `itile_n=iblock/15`，有效 task 范围是 `0..479`。当 `itile_n=32` 时，`src/group_gemm/kernels.cuh:465`（函数 `group_gemm_fp8_kernel`）中的条件成立并退出。

所以，对一个 warpgroup 来说，最后一个有效 task 会完成一次 store，然后下一次调用 `get_next_tile_vert` 只是得到结束条件，不会再进入 epilogue。

## 13.2 每个 tile 的 store 同步序

`src/group_gemm/kernels.cuh:530` 到 `:547`（函数 `group_gemm_fp8_kernel`）的序列不是“等待当前 store 再写”，而是一个针对复用 `sCT` 的生产者流程：

| 代码位置 | 操作 | 业务/硬件含义 |
|---|---|---|
| `src/group_gemm/kernels.cuh:530` (`group_gemm_fp8_kernel`) | `tma_store_wait<0>()` | 等待本 issuing thread 之前提交的 bulk group 至少已经读完 `sCT`，以允许下一个 task 的 STSM 覆盖该共享缓冲区。 |
| `src/group_gemm/kernels.cuh:531` (`group_gemm_fp8_kernel`) | `syncwarpgroup(iwarpgroup)` | 让该 128-thread warpgroup 的所有 lane 在重新写 `sCT` 前会合。 |
| `src/group_gemm/kernels.cuh:533` (`group_gemm_fp8_kernel`) | `cute::copy(tiled_copy_c, ...)` | STSM 把寄存器中的 BF16 结果写入 `sCT`，这是 generic proxy 的 shared-memory 写入。 |
| `src/group_gemm/kernels.cuh:534` (`group_gemm_fp8_kernel`) | 第二次 `syncwarpgroup` | 确保所有 lane 已写完自己那部分 `sCT`，领导线程才可发起 TMA store。 |
| `src/group_gemm/kernels.cuh:535` (`group_gemm_fp8_kernel`) | `cute::tma_store_fence()` | 在 `3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:1214` 的 `tma_store_fence` 中发出 `fence.proxy.async.shared::cta`，把 generic proxy 的 shared 写入排序到 async proxy 的 TMA 读取之前。 |
| `src/group_gemm/kernels.cuh:545` (`group_gemm_fp8_kernel`) | `cute::copy(tma_d.with(...), ...)` | 只由 `is_leader_in_warpgroup` 的线程发起非阻塞 `cp.async.bulk.tensor...global.shared::cta.bulk_group`。 |
| `src/group_gemm/kernels.cuh:547` (`group_gemm_fp8_kernel`) | `tma_store_arrive()` | `3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:1225` 的 `tma_store_arrive` 发出 `cp.async.bulk.commit_group`，只是提交本线程的待处理 store group，不是完成通知。 |

`3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:1248` 的 `tma_store_wait` 实际发出 `cp.async.bulk.wait_group.read 0`（`:1251`）。PTX 规定 `.read` 只等到源地址读取完成；默认的无 `.read` 形式才会额外等待目的地写入并对执行线程可见。参见 [PTX `cp.async.bulk.wait_group`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)。只有 leader 发起 TMA 时，它的 per-thread bulk group 才非空；其他 lane 执行同一个 wait 时通常只观察到空 group。

## 13.3 最后 tile 没有下一次 wait 是否会错

**对普通 CUDA stream 和本次测试，不会因为缺少这一次 tail wait 而自动丢失最后一个 tile。** 原因是：

1. `tma_store_wait<0>()` 在每轮开头的主要任务是保护下一次对 `sCT` 的覆盖；最后一个 task 后没有再次写 `sCT`，因而不需要为“重用源缓冲”而等。
2. `tma_store_arrive` 已将当前操作提交给 TMA bulk-group；线程达到函数末尾不会取消该异步操作。普通 stream 的后续操作或 `cudaStreamSynchronize` 在需要输出时会遵守 kernel 的完成顺序。
3. Colfax 的 TMA store 示例在 kernel 发起 store 后直接退出，明确说明如果没有同 kernel 内的源缓冲重用，`arrive/wait` 不是必须的。参见 [Colfax TMA store 说明](https://research.colfax-intl.com/tutorial-hopper-tma/)。
CUDA Programming Guide 的 bulk-asynchronous 示例也采用“发起 shared→global TMA、commit、`wait_group.read 0`”后结束 kernel 的序列，见 [CUDA asynchronous copies 示例](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)。
仓库内的 `src/gemm/sm90/gemm_bf16xfp32.cu:367` 到 `:372`（函数 `gemm_bf16xfp32_kernel`）也只在 `kSplitK>1` 、后续逻辑会读取/标记 store 结果时补尾部 `tma_store_wait<0>()`；不需要后续依赖的路径不会因为“最后一个 tile”自动多出一次 wait。

**但这不等于当前代码已经在 PDL 解释下保证了 global write 完成。** `src/group_gemm/group_gemm_pertensor_fp8.cu:389` 的 `group_gemm_fp8_async` 无条件将 `use_pdl` 设为 `true`，`src/group_gemm/group_gemm_pertensor_fp8.cu:268` 到 `:269`（函数 `launch_group_gemm_fp8`）为后续 kernel 开启 `cudaLaunchAttributeProgrammaticStreamSerialization`，而 `src/group_gemm/kernels.cuh:634`（函数 `group_gemm_fp8_kernel`）调用 `cudaTriggerProgrammaticLaunchCompletion()`。这个 trigger 是“依赖 kernel 可以开始启动”的释放点，不是 TMA destination 完成点。CUDA 指南明确要求 PDL 的 secondary kernel 在读取主 kernel 结果前调用 `cudaGridDependencySynchronize()`；当前 kernel 自身在 `src/group_gemm/kernels.cuh:303`（函数 `group_gemm_fp8_kernel`）有这个等待点。参见 [CUDA PDL 指南](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)。

因此需要区分两种情况：

- 同一 stream 中的普通后续操作，或者 host 等待 stream 完成：最后一个 TMA store 可以没有下一次循环内 wait，这不是当前测试中的数值错误来源。
- 后续 kernel 也使用 PDL 并在 `cudaGridDependencySynchronize` 之前就读 `y`：这是违反 PDL 合同的用法，可以产生读到旧数据的 race。

如果需要在 `cudaTriggerProgrammaticLaunchCompletion` 之前明确保证 global destination 已完成，不应只加一个现有的 `tma_store_wait<0>()`，因为它发出的是 `.read` 版本；应使用无 `.read` 的 `cp.async.bulk.wait_group 0` 封装，或保持正确的 secondary-side `cudaGridDependencySynchronize()`。为了只保护同 kernel 内的 `sCT` 重用，在 task loop 退出后补一个 `.read` tail wait 即可，但它不是 global write completion wait。
如果真要补这个 tail wait，必须让每个发起 TMA store 的 warpgroup leader 等待自己的 per-thread bulk group，并在 PDL trigger 前用 CTA/warpgroup 同步确保所有 issuing leader 都到达了尾部；单个 leader 的 wait 不代表整个 CTA 的 store 都已 drain。

## 13.4 LIKE_DEBUG 新增 print/printf 审计

以 `git diff fix_kTaskLoopPolicy_type..like` 为范围，只统计这个差异中新增的输出调用，不把仓库原有的 vendor/test 输出算进来。

### 13.4.1 已被 `#ifdef LIKE_DEBUG` 保护的 C++/CUDA 输出

- `src/group_gemm/entry.cc:63` 到 `:83`（函数 `group_gemm_fp8_entry`）：TMA descriptor 的 shape/stride/dtype/device 输出。
- `src/group_gemm/group_gemm_pertensor_fp8.cu:130` 到 `:183`（函数 `launch_group_gemm_fp8`），以及 `:315` 到 `:317`（同函数）：host-side layout/TMA/shm 调试输出。
- `src/group_gemm/kernels.cuh:386` 到 `:392`（函数 `group_gemm_fp8_kernel`）：任务映射调试 `printf`；`src/group_gemm/kernels.cuh:549` 到 `:627`（同函数）：大量 CuTe `print/printf`。
- `src/utils/tma.cuh:45` 到 `:50`（函数 `update_tma_gtensor`）：设置 TMA shape/stride 后的 device `printf`。

LIKE_DEBUG 未定义时，以上内容在预处理阶段被完全删除；剩下的空 `{}`、题外的 `<cstdio>` 和编译期类型声明不会产生 kernel 运行时的 `printf/print` 指令。

### 13.4.2 不在 `#ifdef LIKE_DEBUG` 之内的新增输出

- `setup.py:44` 和 `:47`（函数 `CMakeBuild.build_extension`）的 Python `print` 无条件执行，只增加构建日志，不进入 GPU kernel。
- `tests/test_group_gemm_pertensor_like.py:32`、`:56`、`:62`（函数 `naive_group_gemm_pertensor_fp8` 和 `test_group_gemm_pertensor_fp8`）的 Python `print` 也无条件执行，只影响测试主机输出。
- `CMakeLists.txt:20`（CMake 构建配置，无 CMake 函数）的 `message` 无条件输出；`src/group_gemm/group_gemm_pertensor_fp8.cu:13`（文件作用域）还无条件定义 `T_LIKE_DEBUG`。后者当前只包住 `TD/ITD` 前向声明和 `#if 0` 死代码，没有 runtime print，但它并未与 `LIKE_DEBUG` 绑定。

所以，如果问题中的“全部 print/printf”包括 setup 和 test 脚本，答案是 **不是全部**；如果只指 C++/CUDA 运行时调试输出，则是 **全部已被 LIKE_DEBUG 保护**。

## 13.5 构建 cache 的额外风险

`CMakeLists.txt:19` 的 `ENABLE_LIKE_DEBUG` 默认值是 `OFF`；`setup.py:41` 到 `:43`（函数 `CMakeBuild.build_extension`）只在 `ENABLE_LIKE_DEBUG>0` 时传入 `-DENABLE_LIKE_DEBUG=ON`，并没有在环境变量为 0 或未设置时主动传入 `OFF`。因此旧 CMake cache 可能继续保留开启状态，`SKIP_CMAKE_GENERATE` 时更会直接跳过重配置。

本工作区当前实际验证到 `build/temp.linux-x86_64-cpython-312/hpc/CMakeCache.txt:454` 是 `ENABLE_LIKE_DEBUG:BOOL=ON`，`build/temp.linux-x86_64-cpython-312/hpc/CMakeFiles/_C.dir/flags.make:6` 中也包含 `-DLIKE_DEBUG=1`。这意味着当前构建的 wheel 不能用“环境没设 `LIKE_DEBUG`”来推断调试代码已被去除。要确保关闭，应清理该 build 目录后重新配置，或直接使 CMake cache 变为 `-DENABLE_LIKE_DEBUG=OFF`；仅给现有 `setup.py` 设置 `ENABLE_LIKE_DEBUG=0` 并不一定能覆盖旧 cache。

## 13.6 总结

- 最后一个 tile 没有下一次循环内 `tma_store_wait<0>()`，对当前普通 stream 用法不会自动导致结果错误；这个 wait 的核心目的是保护 `sCT` 的下一次重用。
- `tma_store_arrive()` 是 commit，不是完成通知；PDL 下必须由 secondary 在读结果前执行 `cudaGridDependencySynchronize()`，或在 producer 端使用真正包含 destination completion 的 wait。
- 新增的 C++/CUDA debug 输出全部有 `#ifdef LIKE_DEBUG`；setup.py 和 test 中的 Python 输出不是，且旧 CMake cache 可能让 `LIKE_DEBUG` 实际仍然开启。

---

# 14. TMA/MMA 边界与 group 隔离（本次 shape）

## 14.1 先校正 box、矩阵轴和本次输入

这里的 `[64,48]` **不是 X 的 TMA box**，而是 Y store 每个 warpgroup 发出的 box。`GroupGEMMFp8Config::get_tma` 在 `src/group_gemm/config.h:81-95`（类型 `GroupGEMMFp8Config`、函数 `get_tma`）中定义了三种 tile：X load 的逻辑形状是 `(kTileM,kTileK)=(48,128)`，W load 是 `(kTileN,kTileK)=(128,128)`，Y store 的完整 output tile 是 `(128,48)`；由于 `kWarpgroupM=2`，Y 又被拆成两个 `[64,48]` store box。TMA 的 `boxDim` 是“本次请求遍历多少元素”，不要求小于 `globalDim`；tiled-mode 允许 bounding box 跨过 tensor 边界，并按 OOB 模式处理。

本次测试在 `tests/test_group_gemm_pertensor_like.py:46-80`（函数 `test_group_gemm_pertensor_fp8`）中使用 `num_group=8`、`n=4096`、`k=7168`，每组 `seqlens=[16,32,48,64,80,96,112,128]`。运行日志 `temp/run.log:261-264`（测试输出）确认 `m=576`、`n=4096`、`k=7168` 和相同的 `cu_seqlens=[0,16,48,96,160,240,336,448,576]`；`temp/run.log:6`（测试输出）确认实例参数为 `kTileM=48,kTileN=128,kTileK=128`。因此本次运行中 N、K 都整除 tile，真正需要边界处理的是每个 group 的 M 尾 tile。

数学上的 Y 是每组 `M x N`，但代码按 `[N,M]` 存储：`src/group_gemm/group_gemm_pertensor_fp8.cu:37-42`（函数 `launch_group_gemm_fp8`）给 X 使用 stride `(k,1)`，给 Y 使用 shape `(n,m)` 和 stride `(1,n)`。所以日志里 group 0 的“16 x 4096”在 Y descriptor 中显示为 `[4096,16]`，不是维度弄反了。

## 14.2 每个 group 的 descriptor 把物理连续内存切成逻辑局部 tensor

`update_grouped_tma` 对 group `g` 做两件关键的地址/边界设置，代码在 `src/group_gemm/kernels.cuh:180-202`（函数 `update_grouped_tma`）：

1. `x_ibatch_ptr = x_ptr + cu_seqlen * k`，并以 `gX=(num_seq,k)`、stride `(k,1)` 更新 X descriptor；因此 descriptor 的 base 是本 group 的第一行，`globalDim` 的 M 维是本 group 的真实长度 `num_seq`。
2. `y_ibatch_ptr = y_ptr + cu_seqlen * n`，并以 `gY=(n,num_seq)`、stride `(1,n)` 更新 Y descriptor；这里第二个逻辑维是 M，同样只允许本 group 的 `num_seq` 行。

`update_tma_gtensor` 在 `src/utils/tma.cuh:36-66`（函数 `update_tma_gtensor`）先调用 `fill_tma_gmem_shape_stride`，再用 `tma_descriptor_replace_addr_in_shared_mem` 替换 base，用 `tma_descriptor_replace_shapes_in_shared_mem`（`src/utils/tma.cuh:10-31`，函数 `tma_descriptor_replace_shapes_in_shared_mem`）替换 global dimensions。TMA descriptor 的轴序会按硬件约定排列，因此 group 0 的日志 `temp/run.log:77-92`（测试输出）显示 X 为 `shape=[7168,16], stride=[1,7168]`，Y 为 `shape=[4096,16], stride=[1,4096]`；这正是 `(K,M)` 和 `(N,M)` 的 descriptor 表示。

这意味着 `tAg`、`tBg`、`tDg` 本身不负责携带另一个 group 的 base pointer：

- `src/group_gemm/kernels.cuh:271-278`（函数 `group_gemm_fp8_kernel`）创建的 `tAg/tBg` 是按照静态 TMA tensor/layout 分区出来的坐标视图。
- 真正用于当前 copy 的 descriptor 指针分别由 `src/group_gemm/kernels.cuh:397-408`（函数 `group_gemm_fp8_kernel`）的 `td_x=td_xy+igroup*2` 和 `src/group_gemm/kernels.cuh:544-546`（函数 `group_gemm_fp8_kernel`）的 `td_y=td_xy+igroup*2+1` 提供。
- 因而 `tAg(_,itile_m,itile_k)`、`tBg(_,itile_n,itile_k,igroup)`、`tDg(_,itile_n*2+iwarpgroup,itile_m)`提供的是坐标/布局；group-local base、global shape/stride 和 OOB 模式在 descriptor 中。

特别是，代码没有把 `cu_seqlen` 再加到 `tAg` 或 `tDg` 的坐标里：偏移只写入 descriptor 的 global base，坐标在每个 group 的局部原点重新从 0 开始。这样不会出现“descriptor 已经偏移一次、coordinate 又偏移一次”的双重偏移。

## 14.3 policy=2 不会生成完整 tile 之外的 M task

边界安全的第一层不是依赖“碰巧遇到零内存”，而是 task 映射本身只为每个 group 生成必要的 tile 数。`src/group_gemm/kernels.cuh:148-175`（函数 `update_grouped_tma`）计算

`tiles[g] = ceil(seqlens[g] / kTileM)`，随后用 block exclusive scan 写出 `cu_tiles_ptr`。本次输入得到：

| group | 有效 M | `ceil(M/48)` | `cu_tiles` 区间 |
|---:|---:|---:|---:|
| 0 | 16 | 1 | `[0,1)` |
| 1 | 32 | 1 | `[1,2)` |
| 2 | 48 | 1 | `[2,3)` |
| 3 | 64 | 2 | `[3,5)` |
| 4 | 80 | 2 | `[5,7)` |
| 5 | 96 | 2 | `[7,9)` |
| 6 | 112 | 3 | `[9,12)` |
| 7 | 128 | 3 | `[12,15)` |

所以 `total_m=15`。`get_next_tile_vert` 在 `src/group_gemm/kernels.cuh:42-66`（函数 `get_next_tile_vert`）先把 `iblock` 解成 `itile_m_total=iblock%total_m`、`itile_n=iblock/total_m`，再用 `cu_tiles_ptr` 反查 group 和 group-local `itile_m`。主 loader 和 math warpgroup 都在 `src/group_gemm/kernels.cuh:363-384`、`src/group_gemm/kernels.cuh:447-468`（函数 `group_gemm_fp8_kernel`）使用同一映射；当 `itile_n>=num_tile_n` 时先退出。当前 `num_tile_n=4096/128=32`，因此有效 task 正好是 `15*32=480` 个，`iblock=480` 不会发起 copy 或 MMA。

这一步保证了每一个有效 M tile 的起点都在本 group 内，最多只有 tile 的尾部元素超出本 group；不会生成“整块起点已经落到下一个 group”的任务。

在主 kernel 中，`src/group_gemm/kernels.cuh:340-417`（函数 `group_gemm_fp8_kernel`）的 `idx>=kNumThreads` 分支是 TMA loader，`src/group_gemm/kernels.cuh:421-547`（函数 `group_gemm_fp8_kernel`）的 `else` 分支是 math warpgroup/epilogue；两边都按同一个 `(igroup,itile_m,itile_n)` 取 tile。也就是说，边界判断不是只存在于 loader：math 侧消费的正是 loader 为同一 local tile 写入的 zero-filled shared tile，store 侧再用同一个 group 的 Y descriptor 裁剪目的坐标。

## 14.4 X load 的 `[48,128]` 为什么不会读到下一个 group

以用户点名的 group 0 为例。X descriptor 的 base 是 `x_ptr`，global M=16、K=7168；`tAg(_,itile_m,itile_k)`在 `src/group_gemm/kernels.cuh:399-405`（函数 `group_gemm_fp8_kernel`）请求一个 48 x 128 tile。唯一的有效任务是 `itile_m=0`，所以请求的 local row 范围是 `[0,48)`：

- local row `[0,16)` 映射到 group 0 的真实 X 行 `[0,16)`；
- local row `[16,48)` 超出该 descriptor 的 global M=16。虽然物理数组中紧接着的地址确实是 group 1 的 X 行，但这些坐标被 TMA 判定为 OOB，不会把 group 1 的值送进 shared memory。

一般地，group `g` 的有效 X 地址集合是

`[x_ptr + cu_seqlens[g]*k, x_ptr + (cu_seqlens[g]+seqlens[g])*k)`。

令 `L=seqlens[g]`、`T=kTileM`。因为 task 生成规则给出 `0 <= itile_m < ceil(L/T)`，tile 起点满足 `itile_m*T < L`；对 tile 内偏移 `u`，当 `itile_m*T+u < L` 时才落入上述有效区间，否则就是 descriptor OOB。由于 `cu_seqlens[g+1]=cu_seqlens[g]+L`，任何被判定为有效的地址都严格小于下一个 group 的起点。

TMA 对每个 box 坐标先相对于 descriptor 的 `globalDim` 做边界判断；只有 local row `< seqlens[g]` 且 local K `< k` 的元素才按 stride 形成 global address。OOB load 使用零填充，而不是继续沿 stride 访问相邻内存。CUDA PTX 对 tensor memory OOB 的定义见 [PTX Tensor Memory Access 的 OOB 规则](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)，CUDA Driver API 对 `globalDim`、`boxDim` 和 `oobFill` 的定义见 [CUDA Tensor Memory API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html)。

这里“不会读到”指架构可观察的 copy 结果：越界元素不会以 group 1 的值写入 shared memory，也不会参与 MMA。底层实现是否为对齐 cache line 做了内部取数，不是该 TMA API 对应用提供的隔离承诺；若要求物理内存事务也绝不触碰相邻 allocation，需要额外的分配/保护页设计。

当前 CUTLASS descriptor 的默认 `DescriptorAuxParams::oobfill_` 是 `OOBFill::ZERO`，见 `3rd/cutlass/include/cute/arch/copy_sm90_desc.hpp:150-183`（类型 `OOBFill`、`DescriptorAuxParams`）；`to_CUtensorMapFloatOOBfill` 在 `3rd/cutlass/include/cute/arch/copy_sm90_desc.hpp:265-271`（函数 `to_CUtensorMapFloatOOBfill`）把它编码成 `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`。这里的 `NONE` 不是“关闭边界检查”，而是“不请求特殊 NaN 模式”，默认 OOB 值为零；特殊的 `NAN_REQUEST_ZERO_FMA` 才是另一种模式。

X 的 shared 目标由 `src/group_gemm/kernels.cuh:251-275`（函数 `group_gemm_fp8_kernel`）按完整 `(48,128)` tile 分配，因此 32 个零填充行落在已经分配的 shared tile 内，不会造成 shared OOB。随后 `cute::gemm` 读取这块完整 tile，硬件只看到合法的 shared 地址。

## 14.5 W load 和 MMA atom 维度较大时为什么仍然安全

W descriptor 是固定的三维 tensor，不随 M group 改写。host 端在 `src/group_gemm/group_gemm_pertensor_fp8.cu:39-40`（函数 `launch_group_gemm_fp8`）定义 W shape `(n,k,num_group)`、stride `(k,1,n*k)`；`tBg(_,itile_n,itile_k,igroup)`在 `src/group_gemm/kernels.cuh:277-278` 和 `src/group_gemm/kernels.cuh:407-408`（函数 `group_gemm_fp8_kernel`）用第三维 `igroup` 选择对应的 W slice。当前 n=4096、k=7168 都整除 box，所以 W 本次没有尾部 OOB；若某个配置的 N/K tile 跨出 descriptor 的 globalDim，同样由 TMA load 的 OOB-zero 规则处理，而不会沿第三维跳到别的 group。

日志 `temp/run.log:31-40`（测试输出）显示本次 `TiledMma` 的 MMA atom 为 `Shape_MNK=(64,48,32)`。在 `SS_TN` 映射下，`src/group_gemm/kernels.cuh:430-436`（函数 `group_gemm_fp8_kernel`）把 W 分到 A、X 分到 B，因此 atom 的 64 轴对应输出 N、48 轴对应输出 M；两个 warpgroup 的 64 轴合起来正好是 `kTileN=128`。这个 atom 是固定的硬件计算形状，不能因为 group 0 只有 16 行就缩成 16 行；代码让它照常执行完整 atom：

1. X 的有效 16 行正常装入，其余 M 位置是 FP8 zero；
2. 对这些位置，WGMMA 的每个 K 乘加都是 `0 * W`，所以对应输出 accumulator 为零（假设有效输入是有限值）；
3. `src/group_gemm/kernels.cuh:472-501`（函数 `group_gemm_fp8_kernel`）每个 output tile 都先建立/清零 `tDr`，并用 `ScaleOut::Zero` 初始化该 K tile 的首条 WGMMA，因此不会把上一个 task 的寄存器值带入尾部位置。

这就是“用零填充支持固定 MMA atom”的数值依据：矩阵乘法对缺失的行/列补零后，合法输出区域与未补零的原始矩阵乘法完全相同。零填充不是任意垃圾值；如果把 OOB load 配成非零值，确实会污染结果。

TMA load 的 transaction bytes 也按完整 box 计算。`src/group_gemm/kernels.cuh:340-346`（函数 `group_gemm_fp8_kernel`）定义 `kTransactionBytes=sizeof(Tin)*(kTileM+kTileN)*kTileK`，并在 `src/group_gemm/kernels.cuh:399-410`（函数 `group_gemm_fp8_kernel`）两条 copy 后调用 `set_barrier_transaction_bytes`。本次每个 K tile 的 expected bytes 是 `(48+128)*128*1=22528`；其中包含写入 shared 的零填充部分，所以 consumer 等待的是完整 tile，而不是只等待 16 行。

## 14.6 Y store 的 `[64,48]` 为什么不会写到下一个 group

Y descriptor 的 group 0 base 是 `y_ptr`，global dimensions 是 N=4096、M=16；group 1 的 base 则是 `y_ptr + 16*4096`。epilogue 在 `src/group_gemm/kernels.cuh:538-546`（函数 `group_gemm_fp8_kernel`）取 `td_y=td_xy+igroup*2+1`，并以两个 `[64,48]` box 覆盖一个 N=128、M=48 output tile。group 0 的第一个 tile 具体是：

- `iwarpgroup=0`：N `[0,64)`、local M `[0,48)`；
- `iwarpgroup=1`：N `[64,128)`、local M `[0,48)`；
- descriptor 只允许 local M `[0,16)` 写回；local M `[16,48)` 的 store 元素被 TMA 丢弃。

因此 group 0 的 store 绝不会写到物理上紧邻的 group 1 输出区。一般 group `g` 的 Y 有效地址集合是

`[y_ptr + cu_seqlens[g]*n, y_ptr + (cu_seqlens[g]+seqlens[g])*n)`，

store 的 local M 坐标仍是 `itile_m*kTileM+u`；只有它小于 `seqlens[g]` 时才映射到上式。因为 Y 的 M stride 是 `n`，最后一个有效 local row 对应的最后一列地址仍小于 `y_ptr+(cu_seqlens[g]+seqlens[g])*n`，不会跨过 group 边界。

而所有超出 descriptor global M 的 store 坐标被抑制。CUDA 编程指南对 TMA/异步 tile store 的边界行为说明见 [CUDA Tile Kernels：边界 tile store](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html)。这里没有“先写到下一个 group 再回滚”的过程；TMA 在形成目的地址前就按 descriptor 边界筛掉 OOB 元素。

同样，完整的 `sCT` 由 `src/group_gemm/kernels.cuh:518-528`（函数 `group_gemm_fp8_kernel`）分配和分区，写满 `[128,48]` 的 shared tile 不会越过 shared allocation；只有最终 global store 的 OOB 部分被丢弃。即使寄存器 fragment 中保留了无效 M 位置，它们也没有对应的可写 global destination。

## 14.7 “padding”到底是哪一种，以及它为什么不改变结果

当前实现**没有给每个 group 额外分配物理 padding 行**。准确说是两层逻辑 padding：

- load 侧：descriptor 的 box 大于 group 的 `globalDim` 时，TMA 把越界 operand 元素填成 zero；
- store 侧：descriptor 的 box 大于 group 的 `globalDim` 时，TMA 直接丢弃越界 destination 元素。

所以在 `group_gemm_fp8_kernel` 中看不到逐元素的 `if (row < seqlen[g])`：边界谓词由 TMA tensor-map 的 `globalDim` 在异步 copy 引擎内执行，而不是由每个 CUDA lane 手工分支。

对 X 的 group 0，补出的 32 行在每个 K 位置都是零，所以对任一合法输出行 `r<16`，WGMMA 计算仍是

`C[r,n] = Σ_k X[r,k] * W[n,k]`；补出的 `r>=16` 只产生零 accumulator，并且不会被写到 Y。若是 K 尾部，缺失的 K 项同样贡献 `0*W=0`，但 N/M 仍然有效的输出照常写回；只有 N 或 M 尾部的无效输出坐标会在 store 侧被丢弃。

如果改成真正的**物理 padding**，必须同时满足：每组的 padded stride/分配互不重叠、padding 值初始化为 zero、descriptor 的 globalDim 仍然是逻辑长度而不是把下一个 group 当成 padding。仅仅在一块连续数组后面“预留一些字节”不能替代 descriptor 边界；否则尾部访问仍可能落入下一个 group。当前动态 descriptor 方案不依赖这种物理 padding。

## 14.8 重要的实现前提：policy=2 的 descriptor acquire

上面的地址证明以“consumer 看到的是已经更新好的 group descriptor”为前提。producer 在 `src/group_gemm/kernels.cuh:204-212`（函数 `update_grouped_tma`）把 shared 中改好的 descriptor 通过 `tma_descriptor_cp_fence_release` 发布到 `tma_xy`；其底层 release 指令在 `3rd/cutlass/include/cute/arch/copy_sm90_desc.hpp:423-435`（函数 `tma_descriptor_cp_fence_release`）。

但当前 `group_gemm_fp8_kernel` 只有 policy 0 在 `src/group_gemm/kernels.cuh:306-319`（函数 `group_gemm_fp8_kernel`）显式调用 `tma_descriptor_fence_acquire`；policy 2 的分支 `src/group_gemm/kernels.cuh:330-338`（函数 `group_gemm_fp8_kernel`）只复制 `cu_tiles` 并做 CTA `__syncthreads()`，没有对 `td_x/td_y` 调 acquire。acquire 的实现位于 `3rd/cutlass/include/cute/arch/copy_sm90_desc.hpp:457-470`（函数 `tma_descriptor_fence_acquire`）。CUDA 文档明确指出，普通 stream/grid 同步不能替代 tensor-map proxy 的 release/acquire；参见 [CUDA Programming Guide：异步拷贝与 tensor map proxy](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)。

所以应区分两个结论：

1. **边界算法本身**是正确的：动态 base + per-group globalDim + TMA OOB zero/discard + 只生成合法 local tile，足以阻止逻辑上的跨 group 读写；本次日志中的 group 0 `[16,7168]` / `[16,4096]` 尾 tile 正是这样处理的。
2. **当前 policy=2 的形式化内存可见性**仍有缺口：在 descriptor 被 device 动态改写、尤其是 descriptor cache 被复用时，不能仅凭 `cudaGridDependencySynchronize` 和 `__syncthreads` 宣称 acquire 已完成。工程上应在 policy=2 第一次使用每个 `td_x/td_y` 前补相应的 `tma_descriptor_fence_acquire`，再按需要做 CTA 同步，并用实际构建/运行验证。

这项 acquire 缺口不等于“必然会读到别的 group”；它说明的是 descriptor 更新的发布/消费顺序还没有按 CUDA tensor-map proxy 的严格规则闭合。当前测试能稳定通过只能说明该环境下 descriptor 可见性恰好满足，不能替代上述同步审计。

## 14.9 结论

对本次 `group 0: M=16,N=4096,K=7168`：

- X 的真实 box 是 `[48,128]`，group 0 的第 16--47 行由 TMA zero-fill；不会读取 group 1 的 X；
- Y 的每次 store box 是 `[64,48]`，group 0 的第 16--47 行由 TMA OOB store discard；不会写入 group 1 的 Y；
- `[64,48]` 大于有效 M、`(64,48,32)` MMA atom 大于有效 M 都不是错误，因为 shared tile/寄存器 fragment 采用固定形状，OOB operand 为零，OOB destination 不写；
- 本次 N、K 整除 tile，因此没有实际 N/K 尾部；若出现尾部，仍需保持相同 OOB 模式和正确的 tile 计数；
- 这不是依靠物理 padding 保护连续数组，而是依靠每 group descriptor 的 base 和 `globalDim` 做隔离；同时应补审 policy=2 的 tensor-map acquire。

以上还依赖输入元数据契约成立：`cu_seqlens` 必须是非降的合法前缀和、`cu_seqlens[num_group] == m`，且每个 group 的实际内存区间确实落在已分配的 X/Y tensor 内；各 descriptor 的 base/stride 还必须满足 TMA 的对齐约束。本次 `k=7168`、`n=4096` 使所有 group base 偏移满足 16-byte 对齐；当前 kernel 没有为损坏的 `seqlens/cu_seqlens` 或未对齐指针另加校验。

# 15. `tiled_copy_c` 与 STSM 输出搬运（本次 shape）

> **路径说明**：当前 `like` 分支经过架构重构，原问题中的 `src/group_gemm/kernels.cuh` 已移动为 `src/group_gemm/sm90/kernels.cuh`。下面的源码引用和行号均以当前 code base 的实际路径为准；讨论的函数仍是 `group_gemm_fp8_kernel`。

## 15.1 本次实例的编译期参数

测试函数在 `tests/test_group_gemm_pertensor_like.py:46-66`（`test_group_gemm_pertensor_fp8`）生成 8 个 group，`seqlens=[16,32,48,64,80,96,112,128]`，所以日志中的总 M 是 576，N=4096，K=7168（`temp/run.log:261-264`，由 `test_group_gemm_pertensor_fp8` 打印）。平均 sequence length 是 72。

`src/group_gemm/sm90/group_gemm_pertensor_fp8.cu:363-375`（`group_gemm_fp8_async`）固定 `kTileN=128`、`kTileK=128`、`kWarpgroupM=2`、`kWarpgroupN=1`。平均长度 72 命中 `src/group_gemm/sm90/group_gemm_pertensor_fp8.cu:425-431`（`group_gemm_fp8_async`）的 `kTileM=48` 分支；`src/group_gemm/sm90/config.h:43-52`（`mma_selector`）因此选择 `SM90_64x48x32_F32E4M3E4M3_SS_TN`。

本次日志给出的关键对象是（`temp/run.log:31-40,214-223`，由 `group_gemm_fp8_kernel` 的 debug 输出产生）：

| 对象 | 本次值 | 直接含义 |
|---|---:|---|
| MMA atom `Shape_MNK` | `(64,48,32)` | 一个 WGMMA atom 的逻辑 M/N/K 形状 |
| `ThrLayoutVMNK` 的线程总数 | `128 x 2 = 256` | 两个 128-thread math warpgroup |
| `Tiler_MN` | `(128,48)` | 一个 C 输出 tile 的坐标域 |
| `TiledLayout_TV` 的线程数 | `4 x 8 x 8 = 256` | tiled copy 的逻辑线程数 |
| `TiledLayout_TV` 的每线程值数 | `2 x 2 x 6 x 1 x 1 = 24` | 每个 math lane 的 BF16 输出片段大小 |

因此本次一个 output tile 的元素总数是 `128 x 48 = 6144` 个 BF16，也等于 `256 x 24`。

## 15.2 `make_tiled_copy_C` 实际构造了什么

在 `src/group_gemm/sm90/kernels.cuh:500-506`（`group_gemm_fp8_kernel`）中，WGMMA 的 FP32 accumulator `tCr` 先转换为 BF16 tensor `tCrh`。随后 `src/group_gemm/sm90/kernels.cuh:511-518`（`group_gemm_fp8_kernel`）执行：

```cpp
using STSM_ATOM = std::conditional_t<kTileM == 8,
                                    cute::SM90_U16x4_STSM_T,
                                    cute::SM90_U16x8_STSM_T>;
using R2SCopyAtomC = Copy_Atom<STSM_ATOM, Tout>;
auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
```

本次 `kTileM=48`，所以实际类型是 `Copy_Atom<SM90_U16x8_STSM_T, cute::bfloat16_t>`。`Tout` 的定义见 `src/group_gemm/sm90/group_gemm_pertensor_fp8.cu:35-46`（`launch_group_gemm_fp8`）。

`3rd/cutlass/include/cute/atom/copy_atom.hpp:439-446`（`make_tiled_copy_C`）做两件事：

1. 取 `tiled_mma.get_layoutC_TV()` 作为 tiled-copy 的 `(thread,value) -> C-tile-coordinate` 映射；
2. 取 `make_shape(tile_size<0>(tiled_mma), tile_size<1>(tiled_mma))` 作为 tile 的坐标范围。

接着 `3rd/cutlass/include/cute/atom/copy_atom.hpp:405-415`（`make_tiled_copy_impl`）把这三个部分包装成一个 `TiledCopy<Copy_Atom, LayoutCopy_TV, ShapeTiler_MN>`。因此 `tiled_copy_c` 不是“一个能自动搬完整 tile 的单条指令”，而是：

- 一个 STSM `Copy_Atom`；
- 一份覆盖整个 C tile 的 TV layout；
- 一个 `(128,48)` 的 tile shape。

`3rd/cutlass/include/cute/atom/copy_atom.hpp:185-206`（`TiledCopy`）还明确显示 `TiledCopy` **继承一个** `Copy_Atom`，并用 `TiledLayout_TV` 复制它的线程和值模式；它内部没有存放 24 个独立的 `Copy_Atom` 对象。`TiledNumThr` 和 `TiledNumVal` 必须分别能被 atom 的线程数和值数整除（同文件 196-206 行的静态断言）。

`src/group_gemm/sm90/kernels.cuh:515-518`（`group_gemm_fp8_kernel`）中的 `get_slice(idx)`、`retile_S(tCrh)` 和 `partition_D(sCT)`，把完整 tiled copy 变成当前 lane 要读的寄存器片段 `tCr4s` 和要写的 shared-memory 片段 `tCs4r`。对应实现见 `3rd/cutlass/include/cute/atom/copy_atom.hpp:338-392`（`TiledCopy::get_slice`、`ThrCopy::partition_D`、`ThrCopy::retile_S`）。

## 15.3 `SM90_U16x8_STSM_T::copy` 一次搬多少 BF16

### 15.3.1 每个 lane 的输入寄存器

`3rd/cutlass/include/cute/arch/copy_sm90.hpp:140-157`（`SM90_U16x8_STSM_T::copy`）定义：

```cpp
using SRegisters = uint32_t[4];
using DRegisters = uint128_t[1];
```

`ValueType` 是 BF16，1 个 BF16=16 bit。于是每个参与 lane 的源寄存器是 `4 x 32 bit = 128 bit = 8 BF16`；目的寄存器 `uint128_t` 也正好容纳 8 个 BF16。换句话说，**一次 atom 操作中，每个线程提供 8 个 BF16**。这里的“线程”是一个 STSM atom 的一个 lane，不是整个 CTA。

### 15.3.2 一条 warp-level STSM 的搬运量

同一个函数在 `3rd/cutlass/include/cute/arch/copy_sm90.hpp:151-153`（`SM90_U16x8_STSM_T::copy`）发出：

```ptx
stmatrix.sync.aligned.x4.trans.m8n8.shared.b16
```

这是 warp-collective 指令。32 个 lane 协同执行时，总搬运量是 `32 x 8 = 256 BF16`。`.x4` 表示四个 `8x8` 的 BF16 matrix（`4 x 8 x 8 = 256` 个元素），不是“每线程 4 个 BF16”；每线程的 4 是 4 个 `uint32` 寄存器，每个寄存器装 2 个 BF16。`.trans` 只改变寄存器到 shared 的矩阵排列，不改变元素总数。

从 C++ 视角，每个 lane 都会执行内联的 `copy` 函数；从硬件计数视角，应把 32 个 lane 合并计为 **1 个 warp-level STSM operation**。因此不要把“函数被 32 个线程调用”误读成 32 条独立的矩阵 store。

### 15.3.3 两个 math warpgroup 的数量

`src/group_gemm/sm90/kernels.cuh:330-333`（`group_gemm_fp8_kernel`）用 `kNumThreads=size(TiledMma{})` 区分 256 个 math 线程和 128 个 loader 线程；`src/group_gemm/sm90/kernels.cuh:411-426`（`group_gemm_fp8_kernel`）的 math 分支用 `iwarpgroup=idx/128`，所以一个 math warpgroup 有 128 个线程，即 4 个 warp。

下面把“执行一次”分成三个层次，避免歧义：

| 范围 | 参与 warp/lane | atom 次数 | 搬运 BF16 |
|---|---:|---:|---:|
| 一个 lane 的一次 `SM90_U16x8_STSM_T::copy` | 1 lane | 1 | 提供 8（由 warp 合作写入） |
| 一个 warp 的一次 atom operation | 1 warp=32 lanes | 1 | 256 |
| 一个 128-thread math warpgroup 的一轮 atom operation | 4 warps | 4 | 1024 |
| 两个 math warpgroup 的一轮 atom operation | 8 warps | 8 | 2048 |

最后一行是“每个 warp 各执行一次 atom”的一轮；它还不是当前完整 C tile 的最终答案。完整次数见下一节。

在本次完整 `cute::copy` 中，每个 warp 需要 3 个 value group，所以一个 128-thread math warpgroup 实际执行 `4 x 3 = 12` 个 atom，写 `12 x 256 = 3072 BF16`；两个 math warpgroup 合计 `24 x 256 = 6144 BF16`。如果把问题中的“一次 `SM90_U16x8_STSM_T::copy`”严格理解为一轮 atom，那么答案是上表的 2048 BF16；如果理解为这一行 `cute::copy` 完整覆盖一个 tile，则答案是 6144 BF16。

## 15.4 `cute::copy(tiled_copy_c, tCr4s, tCs4r)` 的展开次数

### 15.4.1 当前线程片段为什么是 24 个值

日志中 `tCr4s` 和 `tCs4r` 的布局分别是 `((_8,_3),_1,_1)`（`temp/run.log:238-241`，由 `group_gemm_fp8_kernel` 打印）。其第一个模式有 `8 x 3 = 24` 个 BF16：每个 lane 总共要搬 24 个值，并按“每次 atom 8 个值”分成 3 组。

完整 tiled layout 也给出同一结果：`TiledLayout_TV` 的线程模式是 `4 x 8 x 8 = 256`，值模式是 `2 x 2 x 6 x 1 x 1 = 24`。因此：

```text
TiledNumThr / AtomNumThr = 256 / 32 = 8  个 warp-level atom 位置
TiledNumVal / AtomNumVal = 24  /  8 = 3  个值分组
总 atom operation = 8 x 3 = 24
```

这也与 `128 x 48 = 6144` 个 tile 元素相符：`24 x 256 = 6144 BF16`。

### 15.4.2 CUTE 的“循环”是怎样发生的

`3rd/cutlass/include/cute/algorithm/copy.hpp:434-444`（`cute::copy(TiledCopy,...)`）先把 `TiledCopy` 向下转换为它继承的 `Copy_Atom`，所以第一参数不会触发一个神秘的单条“大拷贝”指令。随后 `3rd/cutlass/include/cute/algorithm/copy.hpp:189-235`（`cute::copy(Copy_Atom,...)`）对除 value 主模式之外的 rest modes 执行 `CUTE_UNROLL` 循环，并在每个分组调用 `copy_atom.call`。

`3rd/cutlass/include/cute/atom/copy_atom.hpp:89-113`（`Copy_Atom::call`）在片段尺寸匹配时进入 `copy_unpack`；`3rd/cutlass/include/cute/atom/copy_traits.hpp:108-136`（`copy_unpack`）把 8 个 BF16 重解释为 4 个 `uint32_t` 源寄存器和 1 个 `uint128_t` 目的寄存器，然后通过 `3rd/cutlass/include/cute/arch/util.hpp:153-159`（`detail::CallCOPY::operator()`）调用 `SM90_U16x8_STSM_T::copy`。

所以答案是：

- 源码只有一行 `cute::copy`，但它**逻辑上会重复调用 atom**；
- 对当前 tile，每个 lane 有 3 次 atom 调用，每个 warp 发出 3 条 warp-level `stmatrix.x4.trans` 语义操作；
- 两个 math warpgroup 共 8 个 warp，因此一次完整 tiled copy 是 `8 x 3 = 24` 个 warp-level STSM operation，覆盖 `6144 BF16`；
- 若按 C++ lane 的概念计数，是 `256 lanes x 3 = 768` 次内联 `SM90_U16x8_STSM_T::copy` 路径；这 768 次由硬件按 32 lane 一组体现为 24 个 warp-level 指令实例。

这里的 `CUTE_UNROLL`（宏定义见 `3rd/cutlass/include/cute/config.hpp:49-59`，`CUTE_UNROLL`）是静态布局已知时的编译期展开提示，最终 SASS 中不一定保留一个可见的运行时 `for`。因此“会循环调用”在语义上是肯定的，但不应据此期待看到一个动态循环计数器。`stmatrix.sync` 的具体 SASS mnemonic 可能因 `ptxas` 版本而变化；上面的 24 是该 tile 的逻辑 STSM 操作数。

### 15.4.3 和外层 task 循环的关系

`src/group_gemm/sm90/kernels.cuh:438-498`（`group_gemm_fp8_kernel`）先在 `while` 中取得一个 `(igroup,itile_m,itile_n)` task，再遍历所有 K tile；`cute::copy` 位于 K 循环结束后的 epilogue（`src/group_gemm/sm90/kernels.cuh:469-523`，`group_gemm_fp8_kernel`）。因此：

- 一个 task/output tile 执行一次完整的 `cute::copy`，即 24 个 warp-level STSM；
- 它不是对每个 `itile_k` 都做一次 STSM；
- 本次 N=4096、`kTileN=128`，每 group 有 32 个 N tiles；各 group 的 M tile 数为 `1,1,1,2,2,2,3,3`，逻辑上共 480 个 output tasks。若只按语义总量统计，全 kernel 约为 `480 x 24 = 11520` 个 warp-level STSM operation，由不同 CTA 分摊执行。

## 15.5 `Tiler_MN: (_128,_48)` 是谁的 layout

严格说它**不是 layout，而是 shape**。`TiledCopy` 在 `3rd/cutlass/include/cute/atom/copy_atom.hpp:199-203`（`TiledCopy`）把第三个模板参数命名为 `ShapeTiler_MN`，并定义 `using Tiler_MN = ShapeTiler_MN`；`make_tiled_copy_C` 在 `3rd/cutlass/include/cute/atom/copy_atom.hpp:442-445`（`make_tiled_copy_C`）传入 `make_shape(tile_size<0>(mma), tile_size<1>(mma))`。因此日志里没有冒号的 `(_128,_48)` 是 `Shape<_128,_48>`，不是 `Layout<Shape,Stride>`。

它描述的是“本次 tiled copy 要覆盖的 C tile 坐标域”，元素数是 `128 x 48`。kernel 自己在 `src/group_gemm/sm90/kernels.cuh:268-270`（`group_gemm_fp8_kernel`）把 `gC` 建成 `(kTileN,kTileM)=(128,48)`，所以在这个 `SS_TN` 和输出张量约定下，第一坐标语义上是 N=128，第二坐标是 M=48。CuTe 的成员名仍沿用通用的 `Tiler_MN`，不要因为名字中的 MN 就把它当成独立的内存 stride layout。

这个 shape 会被 `TiledCopy::tidfrg_S` 和 `tidfrg_D` 在 `3rd/cutlass/include/cute/atom/copy_atom.hpp:218-247`（`TiledCopy::tidfrg_S`、`TiledCopy::tidfrg_D`）用于 `zipped_divide(..., Tiler_MN{})`，也就是把 `tCrh`/`sCT` 按一个 128×48 tile 切分。

## 15.6 `TiledLayout_TV` 的来源和逐层含义

### 15.6.1 来源

`3rd/cutlass/include/cute/atom/mma_atom.hpp:397-410`（`TiledMMA::get_layoutC_TV`）根据 TiledMMA 的 C fragment ownership，构造 `(thr_idx,val_idx) -> (M,N)` 的 C layout。`make_tiled_copy_C` 把这份 layout 原样传给 `make_tiled_copy_impl`，所以日志中的 `TiledLayout_TV` 是 **TiledMMA 的 C ownership layout 被用作 copy 的 TV layout**，不是 `SLayoutY`、`sCT` 或 `tCr4s` 的 layout。

### 15.6.2 形状部分

日志打印的是：

```text
((_4,_8,_8), ((_2,_2,_6), (_1,_1)))
```

可以按外层 `(T,V)` 读：

- `T=(_4,_8,_8)`：逻辑线程因子化为 4×8×8，乘积 256；这是两个 math warpgroup 的全部 256 个 math lane，而不是 256 个独立 CopyAtom；
- `V=((_2,_2,_6),(_1,_1))`：值模式因子化为 2×2×6×1×1，乘积 24；每个逻辑线程有 24 个 BF16 值；
- 两个 `_1` 模式是退化的 size-one 模式，不增加元素数。

### 15.6.3 stride 部分

冒号右侧：

```text
((_256,_1,_16), ((_128,_8,_1024), (_0,_0)))
```

是与上述各层 shape 一一对应的 stride 因子。它描述 CuTe 如何把 `(T,V)` 坐标组合成 C tile 坐标；这些 stride 是**逻辑 layout 坐标的 stride**，不是 global-memory 字节 stride，也不是 `stmatrix` 的寄存器编号表。`_0` 对应 size-one 模式，因其只有一个坐标值，对最终覆盖范围没有影响。

因此这份 layout 的关键不在于把右侧数字直接当作线性地址，而在于它同时编码了：哪个 math lane 负责哪个 C 坐标、一个 lane 的 24 个值如何分组、以及不同 warp/warpgroup 之间如何覆盖 128×48 tile。`TiledCopy` 的 `tile2thrfrg` 在 `3rd/cutlass/include/cute/atom/copy_atom.hpp:254-280`（`TiledCopy::tile2thrfrg`）利用这份映射产生每线程的 source/destination fragment。

## 15.7 `Copy_Atom` 打印字段逐项解释

`3rd/cutlass/include/cute/atom/copy_atom.hpp:620-642`（`cute::print(Copy_Atom)`、`cute::print(TiledCopy)`）正是日志字段的打印实现。下面区分 atom-local layout 和 full-tile layout。

### 15.7.1 `ThrID: _32:_1`

`3rd/cutlass/include/cute/atom/copy_traits_sm90.hpp:117-130`（`Copy_Traits<SM90_U16x8_STSM_T>`）定义 `using ThrID = Layout<_32>`。`Layout<_32>` 的 shape 是 32，stride 是 1，所以打印为 `_32:_1`：逻辑 thread id 为 0..31，连续对应一个硬件 warp 的 32 个 lane。

这是 STSM 的硬件约束：`stmatrix.sync` 是 warp-collective，不能用一个 128-thread warpgroup 作为单个 atom 的 thread domain。TiledCopy 再把这个 32-thread atom 沿 `TiledLayout_TV` 复制到 `256/32=8` 个 warp。因而 `_32:_1` 不表示整个 kernel 只有 32 个线程，也不表示两 math warpgroup 只执行一次 atom。

### 15.7.2 `ValLayoutSrc: ((_4,_8),(_1,_2,_4)):((_16,_1),(_1,_8,_64))`

`3rd/cutlass/include/cute/atom/copy_atom.hpp:57-74`（`Copy_Atom`）先从 `Copy_Traits` 取得 bit-level `SrcLayout`，再用 `recast_layout<uint1_t, ValType>` 生成 `ValLayoutSrc`。STSM traits 在 `3rd/cutlass/include/cute/atom/copy_traits_sm90.hpp:123-126`（`Copy_Traits<SM90_U16x8_STSM_T>`）把 source layout 取为 `SM75_U16x8_LDSM_T::DstLayout`；其原始层次定义见 `3rd/cutlass/include/cute/atom/copy_traits_sm75.hpp:127-140`（`Copy_Traits<SM75_U16x8_LDSM_T>`）。

读法如下：

- 左半部 `((_4,_8),(_1,_2,_4))` 是 `(thread,value)` 的层次 shape；thread 因子 4×8=32，value 因子 1×2×4=8；
- 右半部 `((_16,_1),(_1,_8,_64))` 是对应的层次 stride；由于已经 recast 到 16-bit `ValType`，这里应理解为 BF16/逻辑值单位的布局偏移，而不是直接的 byte address；
- 它描述一个 STSM atom 从每个 lane 的寄存器片段中取哪些值，以及如何把这些值排列成该 atom 的输入矩阵；它不是完整 `tCrh` 的 layout，也不是 shared-memory 的 `SLayoutY`。

形状的元素数是 `32 x 8 = 256 BF16`，与一条 warp-level `stmatrix.x4` 的搬运量一致。

### 15.7.3 `ValLayoutDst: (_32,_8):(_8,_1)`

STSM traits 在 `3rd/cutlass/include/cute/atom/copy_traits_sm90.hpp:123-129`（`Copy_Traits<SM90_U16x8_STSM_T>`）把 destination bit layout 取为 `SM75_U16x8_LDSM_T::SrcLayout`；原始 layout 是 `(_32,_128):(_128,_1)` bit mapping，经过 `Copy_Atom` 的 16-bit recast 后打印成 `(_32,_8):(_8,_1)`。

- shape `(_32,_8)`：32 个 lane，每 lane 8 个 BF16；
- stride `(_8,_1)`：描述 atom-local destination 坐标中 lane/value 的排列；
- 它是单个 STSM atom 的 shared-side 逻辑 fragment，不是最终整个 shared tile 的 swizzled 地址表。真正的 shared tensor 是 `sCT`（`src/group_gemm/sm90/kernels.cuh:509-510`，`group_gemm_fp8_kernel`），`partition_D` 再把两者组合起来。

### 15.7.4 `ValLayoutRef: ((_4,_8),(_1,_2,_4)):((_16,_1),(_1,_8,_64))`

`3rd/cutlass/include/cute/atom/copy_traits_sm90.hpp:128-129`（`Copy_Traits<SM90_U16x8_STSM_T>`）明确设置 `RefLayout = SrcLayout`，所以 recast 后 `ValLayoutRef` 与 `ValLayoutSrc` 完全相同。

`RefLayout` 是 source 和 destination 共用的 canonical `(thread,value)` 坐标系，不是第三个实际 buffer，也不会额外执行一次 copy。`TiledCopy::tidfrg_S/D` 在 `3rd/cutlass/include/cute/atom/copy_atom.hpp:225-247`（`TiledCopy::tidfrg_S`、`TiledCopy::tidfrg_D`）分别计算：

```text
reference -> source layout
reference -> destination layout
```

这样即使 STSM 是 transpose store，CuTe 仍能保证 `tCr4s` 中的第 i 个逻辑值对应 `tCs4r` 中正确的 shared 坐标。由于本例选择 source 作为 reference，打印结果看起来就和 `ValLayoutSrc` 一样。

### 15.7.5 `ValueType: 16b`

`cute::print(Copy_Atom)` 在 `3rd/cutlass/include/cute/atom/copy_atom.hpp:621-629`（`cute::print(Copy_Atom)`）打印 `sizeof_bits<typename Atom::ValType>::value`。本例的 `ValType` 是 `Tout=cute::bfloat16_t`，所以 `16b` 表示：**每个逻辑 copy value 是 16 bit，也就是 2 byte 的 BF16**。

它不表示：

- 每线程 16 个 BF16；
- 每条指令 16 个 byte；
- `uint128_t` 只有 16 bit。

本例的宽度关系是：`ValueType=16 bit`，每个源寄存器 32 bit 装 2 个 BF16，4 个源寄存器装 8 个 BF16，目的寄存器 128 bit 装 8 个 BF16。

## 15.8 从寄存器到 shared 的一次完整路径

对一个 `(igroup,itile_m,itile_n)` task，代码执行顺序可以压缩成下面的映射链：

1. WGMMA 在 `src/group_gemm/sm90/kernels.cuh:477-485`（`group_gemm_fp8_kernel`）把结果写入每 lane 的 24 个 FP32 `tCr` 值；
2. `src/group_gemm/sm90/kernels.cuh:500-506`（`group_gemm_fp8_kernel`）把这 24 个值转换成 24 个 BF16 `tCrh` 值；
3. `make_tiled_copy_C` 用 TiledMMA 的 C TV ownership 和 `(128,48)` tile shape 构造 `tiled_copy_c`；
4. `retile_S` 把每 lane 的 24 值重排成 `8 x 3` 的 atom-local source fragment，`partition_D` 得到对应的 shared destinations；
5. `cute::copy` 对 8 个 warp atom 位置和每个位置的 3 个 value group 做静态展开；每个 atom 调用发出一条 warp-level `stmatrix.x4.trans.m8n8.shared.b16` 语义操作；
6. 两个 math warpgroup 最终共同把整个 `128 x 48 = 6144 BF16` C tile 写入 `sCT`，之后才由后续 TMA store 从 shared 写回 global（`src/group_gemm/sm90/kernels.cuh:520-537`，`group_gemm_fp8_kernel`）。

## 15.9 直接结论

- `SM90_U16x8_STSM_T::copy`：每 lane 一次提供 8 个 BF16；一个 32-lane warp 一次完成 256 个 BF16。
- 两个 128-thread math warpgroup 若各 warp 做一轮 atom：8 个 warp、2048 个 BF16；对本次完整 `tiled_copy_c`，每 warp 有 3 组值，所以是 24 个 warp-level STSM、6144 个 BF16。
- `cute::copy(tiled_copy_c,...)` 不是一条覆盖整个 tile 的 PTX；它通过静态 `CUTE_UNROLL`/布局递归重复调用 atom。每 lane 3 次、每 warp 3 次、两个 math warpgroup 合计 24 次 warp-level STSM。
- `Tiler_MN` 是 `(128,48)` 的 tile shape；`TiledLayout_TV` 是从 `TiledMMA::get_layoutC_TV()` 得到的 full-tile `(thread,value)->tile-coordinate` layout；`ValLayoutSrc/Dst/Ref` 则是单个 CopyAtom 内部的三份 32-thread/8-value 映射。
- `ThrID=_32:_1` 表示一个 atom 使用一个硬件 warp；`ValueType=16b` 表示逻辑元素是 16-bit BF16。

# 16. Blockwise `total_seq_pad` 与不等长 group 测试

## 16.1 先给结论

对 `hpc.group_gemm_blockwise_fp8`，`total_seq_pad` 不是把所有 group 的真实长度相加后再统一向上取整。给定 kernel 选出的统一 `kTileM`，每个 group 单独计算：

```text
pad_i = ceil(seqlens[i] / kTileM) * kTileM
total_seq_pad = sum_i(pad_i)
```

这是 `x_scale` 第二维的**最小所需长度**。允许调用方提供更大的末尾容量，但每个 group 的有效 scale 必须放在上述紧凑 padded segment 的起点，不能把所有真实 token 直接拼成 `total_seq` 列。

`total_seq` 是 X/Y 的真实 token 行数；`total_seq_pad` 还包括每个 group 为固定 M 方向 TMA box 保留的尾部空槽。两者只有在特殊情况下才相等，例如所有 group 长度已经是 `kTileM` 的整数倍。

## 16.2 `m_pad` 是怎样从 API 传进 kernel 的

### 16.2.1 C++ 入口不计算 padding

`src/group_gemm/entry.cc:145-150`（`group_gemm_blockwise_fp8_entry`）直接执行：

```cpp
int m = x.size(0);
int m_pad = x_scale.size(1);
```

随后 `src/group_gemm/entry.cc:193-198`（`group_gemm_blockwise_fp8_entry`）把 `m_pad` 原样传给 `group_gemm_blockwise_fp8_async`。因此 `hpc.group_gemm_blockwise_fp8` 的调用者负责分配正确的 `x_scale` 第二维；入口目前没有根据 `seqlens` 重新计算或检查它。

### 16.2.2 XS TMA tensor 需要 padded 列空间

`src/group_gemm/sm90/group_gemm_blockwise_fp8.cu:106-125`（`launch_group_gemm_blockwise_fp8`）把 XS 构造成：

```cpp
auto XS = make_tensor(make_gmem_ptr(reinterpret_cast<const TS *>(xscale_ptr)),
                      make_shape(num_block_k, m_pad),
                      make_stride(m_pad, Int<1>{}));
```

这里 `num_block_k=(k+kTileK-1)/kTileK`，而 `m_pad` 是 x-scale 的 token-slot 维度。`src/group_gemm/sm90/config.h:140-153`（`GroupGEMMBlockWiseFp8Config::get_tma`）为 XS 使用 `CopyBoxXS=(1,kTileM)`，即一次按一个 M tile 读取一列 K-block 的 `kTileM` 个 token scale。即使最后一个 group 只有少量真实 token，TMA 仍会请求完整 box，剩余 slot 必须存在于 `x_scale` 中。

### 16.2.3 每组 tile 前缀决定 scale segment 起点

`src/group_gemm/sm90/kernels.cuh:149-176`（`update_grouped_tma`）先计算每组的 tile 数：

```cpp
tiles[i] = (seqlens_ptr[igroup] + kTileM - 1) / kTileM;
```

再用 `BlockScan(...).ExclusiveSum` 生成 `cu_tiles_ptr`。这个前缀保存的是 tile 数，不是 token 数。blockwise kernel 在 `src/group_gemm/sm90/kernels.cuh:703-704` 和 `src/group_gemm/sm90/kernels.cuh:820-823`（`group_gemm_blockwise_fp8_kernel`）使用：

```cpp
tASg(_, itile_k, cu_tiles_ptr[igroup] + itile_m)
```

`CopyBoxXS` 的 M 方向一个坐标对应 `kTileM` 个连续 scale 列，所以 group `i` 的 scale 起点实际是：

```text
offset_i = kTileM * sum_{j < i} ceil(seqlens[j] / kTileM)
         = sum_{j < i} pad_j
```

这正是“每组独立 padding 后再拼接”的布局。若把 `x_scale` 只分配成 `ceil(total_seq/kTileM)*kTileM`，即使总长度恰好整除 tile，也可能没有空间容纳各 group 之间的 padding 间隔。

## 16.3 用本次日志和具体数字说明

### 16.3.1 原始等长用例

原始 `temp/block.log:17-20` 显示 8 个 group、每组 `seqlen=30`，`total_seq=240`，xscale 形状为 `[32,256]`。平均长度 30 选择 `kTileM=32`，所以：

```text
每组 pad_i = ceil(30/32)*32 = 32
total_seq_pad = 8*32 = 256
```

这里 `m_pad=256` 恰好等于旧测试中的 `m*num_group`，但它不等于 `total_seq=240`。旧代码看起来没有问题，是因为所有 group 恰好等长。

### 16.3.2 改造后的 8-group smoke

修改后的 `tests/test_group_gemm_blockwise_like.py:77-116`（`test_group_gemm1`）使用：

```python
seqlens = (1 + torch.arange(num_group, dtype=torch.int32, device="cuda")) * 16
```

脚本入口 `tests/test_group_gemm_blockwise_like.py:259-260`（模块主程序）用 8 个 group 时，长度为 `[16,32,48,64,80,96,112,128]`。平均值为 72，当前 dispatch 选择 `kTileM=48`，因此：

```text
pad_i = [48,48,48,96,96,96,144,144]
total_seq = 576
total_seq_pad = 720
```

如果错误地对总长度统一取整，会得到 `ceil(576/48)*48=576`，少了 144 个必须保留的 segment slot。实际运行日志 `temp/block.modify.log:17-22` 记录了 `tile_m=48`、`total_seq_pad=720`、`x_scale=[32,720]`。

### 16.3.3 pytest 的 128-group 参数

参数化测试 `tests/test_group_gemm_blockwise_like.py:77-80`（`test_group_gemm1`）使用 128 个 group，长度为 `[16,32,...,2048]`。平均值是 1032，dispatch 选择 `kTileM=64`。此时：

```text
total_seq = 16*(1+2+...+128) = 132096
total_seq_pad = 64*(4*(1+2+...+32)) = 135168
```

注意这里 `total_seq` 本身恰好是 64 的整数倍，但仍不能用它作为 `m_pad`，因为每个 group 都要单独补齐到 64 的倍数。`temp/block.pytest.log:257-275` 输出了 `x_scale=[32,135168]` 及对应 `total_seq`。

### 16.3.4 一个更小的反例

若统一 `kTileM=48`，`seqlens=[17,33,65]`，则：

```text
pad_i = [48,48,96]
total_seq = 115
total_seq_pad = 192
```

整体取整只有 `ceil(115/48)*48=144`，不足以表示第二、第三 group 之间的 tile-aligned 起点。这说明问题不是本次测试数据的偶然现象。

## 16.4 为什么 `x_scale` 不能用 `total_seq`

### 16.4.1 X 的行是 compact，scale 的列是 padded

入口在 `src/group_gemm/entry.cc:145-148`（`group_gemm_blockwise_fp8_entry`）把 `m=x.size(0)` 当作 X 的真实 M；X tensor 在 `src/group_gemm/sm90/group_gemm_blockwise_fp8.cu:109-110`（`launch_group_gemm_blockwise_fp8`）按 `(m,k)` 构造，所以 X 只存实际 token，行索引由 `cu_seqlens` 的 compact prefix 给出。

XS tensor 却在同一 launch 函数的 `src/group_gemm/sm90/group_gemm_blockwise_fp8.cu:115-116`（`launch_group_gemm_blockwise_fp8`）按 `(num_block_k,m_pad)` 构造。它的第二维不是 X 的物理行号，而是按 group tile 前缀排列的 scale slot。group 的有效 token 只占其 segment 的前 `seqlens[i]` 列，后面的 `pad_i-seqlens[i]` 列是固定 TMA box 的空槽。

### 16.4.2 错用 `total_seq` 会造成什么

假设 8-group 不等长例子错误地把 xscale 分配成 `[32,576]`：

1. group 0 的 `[0,16)` 可能还能读到正确 scale；
2. group 1 的正确起点应是 48，而不是紧接在 group 0 的 16 后面；
3. 后续 group 的 `cu_tiles_ptr[igroup] + itile_m` 会继续按 48/96/144 等 padded 起点寻址；
4. 最终会出现 scale buffer 越界，或者把别的 group/别的 token 的 scale 当成当前 group 的 scale。

因此 `total_seq` 不能替代 `total_seq_pad`，即使 X 本身是 compact 且总 token 数很小。

## 16.5 `x_scale` 的形状、排列和 padding 值

### 16.5.1 直接 blockwise API 的格式

Python API 文档在 `hpc/group_gemm.py:164-169`（`group_gemm_blockwise_fp8`）规定：

```text
x_scale.shape = [hidden_size // 128, total_seq_pad]
```

第一维对应 K 方向的 128-element scale block；第二维对应按 group padded segment 拼接的 token-slot。对 group `i`：

```text
segment_start = sum_{j<i} pad_j
valid columns  = [segment_start, segment_start + seqlens[i])
padding columns = [segment_start + seqlens[i], segment_start + pad_i)
```

padding columns 不参与有效 token 的结果，通常应初始化为 0 以便调试和后续复用；kernel 的正确性依赖的是它们存在且不越过下一个 segment，而不是依赖 padding 列的随机值。

### 16.5.2 Deepep 输入需要先 reformat

如果输入 scale 原本是 `[total_seq_pad, hidden_size//128]` 的 per-token 格式，`hpc.reformat_x_scale` 会转置并按每组 padded 长度放入 compact output。其 CUDA 实现 `src/group_gemm/sm90/group_gemm_blockwise_fp8.cu:20-23`（`reformat_x_scale_kernel`）用 `cu_seqlens` 定位源 group；`src/group_gemm/sm90/group_gemm_blockwise_fp8.cu:42-47`（`reformat_x_scale_kernel`）累加此前各 group 的 `ceil(seqlen/tilem)*tilem`，得到目标 segment 起点；有效范围判断位于同函数 `:59-82`。

这个 reformat 路径与直接调用 `group_gemm_blockwise_fp8` 要求的最终 `[k//128,total_seq_pad]` 格式不同。直接 API 不会自动替调用者插入 padding 或转置。

## 16.6 测试改造内容

### 16.6.1 `naive_group_gemm` 支持不等长

原实现 `tests/test_group_gemm_blockwise_like.py:20-34`（`naive_group_gemm`）用 `m // num_group` reshape scale，因此只适用于等长 group。现在 `tests/test_group_gemm_blockwise_like.py:40-74`（`naive_group_gemm`）改为：

1. 根据同一个 `tile_m` 为每个 group 计算 `padded_lengths`；
2. 用 `scale_offset` 累加每个 padded segment 的起点；
3. 只取当前 group 的 `xscale[:, scale_offset:scale_offset+seq_len]`，转置并扩展每个 K block 的 128 个值；
4. 从 compact X 的 `cu_seqlens[i]` 行开始做当前 group 的矩阵乘；
5. 即使有 zero-length group，也推进对应 padding offset，避免后续 group 错位。

### 16.6.2 测试端 padding 计算

`tests/test_group_gemm_blockwise_like.py:77-116`（`test_group_gemm1`）现在使用实际 `seqlens` 的总和作为 `total_seq`，用平均长度选择 `tile_m`，再按每组向上取整求 `total_seq_pad`，最后创建精确形状 `xscale=(k//128,total_seq_pad)`。`tests/test_group_gemm_blockwise_like.py:20-37`（`get_tile_m`）镜像 `src/group_gemm/sm90/group_gemm_blockwise_fp8.cu:369-457`（`group_gemm_blockwise_fp8_async`）的 dispatch 表，包含 `64->48`、`96->32`、`144->48` 这些不能用“超过 32 一律 64”替代的分支。

## 16.7 验证结果

### 16.7.1 8-group 脚本入口

在指定环境并使用空闲 GPU 7 执行：

```bash
CUDA_VISIBLE_DEVICES=7 /share_data/users/like/miniconda3/envs/simo_sglang/bin/python \
  tests/test_group_gemm_blockwise_like.py > temp/block.modify.log 2>&1
```

结果 `temp/block.modify.log:17-22`：

```text
mean_seq:72, tile_m:48, total_seq:576, total_seq_pad:720
x.shape=[576,4096], xscale.shape=[32,720]
max_abs_diff:0.03125, mean_abs_diff:0.0011507084
```

脚本正常退出，naive 与 `hpc.group_gemm_blockwise_fp8` 的断言通过。

### 16.7.2 128-group pytest

同一环境执行：

```bash
CUDA_VISIBLE_DEVICES=7 /share_data/users/like/miniconda3/envs/simo_sglang/bin/python \
  -m pytest tests/test_group_gemm_blockwise_like.py::test_group_gemm1 -q -s \
  > temp/block.pytest.log 2>&1
```

结果 `temp/block.pytest.log:273-276` 为 `max_abs_diff=0.03125`、平均绝对误差约 `0.00115693`，并显示 `1 passed in 1.44s`。这次比较覆盖 128 个不等长 group，而不是只验证等长的旧布局。

## 16.8 使用时的边界条件

### 16.8.1 `num_seq_per_group_avg` 必须与 padding 计算一致

`kTileM` 是由调用方传入的 `num_seq_per_group_avg` 选择的，不是 kernel 根据每个 group 动态选择。dispatch 表在 `src/group_gemm/sm90/group_gemm_blockwise_fp8.cu:385-457`（`group_gemm_blockwise_fp8_async`）；如果调用者传入的平均值与测试端不同，就必须按实际选出的 `kTileM` 重新计算全部 `pad_i`。

### 16.8.2 `m_pad` 是最小容量，不是 X 的真实 M

`m_pad` 可以大于 `sum_i(pad_i)` 作为额外容量，但不能小于它；额外尾部不会被当前 group 前缀使用。最重要的是，group segment 的起点必须保持 `sum_{j<i} pad_j`，不能按 `cu_seqlens[i]` 或 `sum_{j<i} seqlens[j]` 直接定位。

### 16.8.3 输入元数据仍需合法

上述证明依赖 `cu_seqlens[0]=0`、`cu_seqlens[i+1]=cu_seqlens[i]+seqlens[i]`、各 group 区间不重叠，且 X/Y 与 scale buffer 已按这些尺寸分配。kernel 当前入口没有为损坏的 prefix、负长度或不足的 `m_pad` 增加完整运行时校验。

## 16.9 最终回答

- `total_seq_pad` 是每个 group 按统一 `kTileM` 独立向上取整后的长度之和，不是总 `total_seq` 一次向上取整。
- `x_scale` 的直接 API 形状是 `[hidden_size//128, total_seq_pad]`；`total_seq` 只描述 compact X 的真实 token 行。
- blockwise kernel 通过 `cu_tiles_ptr` 的 tile 前缀定位每组 scale segment，再由 `CopyBoxXS=(1,kTileM)` 读取固定 box，所以每组尾部 padding 是地址布局的一部分。
- 测试已改为 `[16,32,...]` 不等长 seqlen，naive 基准按 padded offset 取 scale；8-group 脚本和 128-group 参数化测试均通过，结果误差在原断言阈值内。

# 17. `HPC_ARCH_DISPATCH` 的宏展开与设计原因

## 17.1 这次调用的上下文

问题中的调用位于 `src/group_gemm/entry.cc:122-200`（函数 `group_gemm_blockwise_fp8_entry`），实际 dispatch 语句是 `src/group_gemm/entry.cc:193-198`（函数 `group_gemm_blockwise_fp8_entry`）：

```cpp
HPC_ARCH_DISPATCH(
    "group_gemm_blockwise_fp8", 90,
    group_gemm_blockwise_fp8_async(
        y_ptr, x_ptr, weight_ptr, seqlens_ptr, cu_seqlens_ptr, xscale_ptr, wscale_ptr, tmas_ptr,
        tiles_ptr, cu_tiles_ptr, task_map_ptr, num_waves, num_group, m, n, k, m_pad,
        num_block_k_pad4, num_seq_per_group_avg, update_tma, false, stream));
```

这里的参数格式不是 `(op, arch, function)`，而是：

```text
HPC_ARCH_DISPATCH(op, arch_0, expression_0, arch_1, expression_1, ...)
```

本例只有一个 `(90, expression)` pair。`group_gemm_blockwise_fp8_async` 的共享声明在 `src/group_gemm/group_gemm.h:21-27`（函数声明 `group_gemm_blockwise_fp8_async`），而实际实现位于 sm90 专用源文件；因此入口文件需要在不同架构模块中都能编译，但不能让每个模块都链接所有架构实现。

构建系统在 `CMakeLists.txt:53-82`（架构源文件选择逻辑）按 `HPC_TARGET_ARCH` 只加入当前架构的源文件，并在 `CMakeLists.txt:162-171`（目标 `target_compile_definitions`）把 `HPC_TARGET_ARCH` 作为预处理宏传给编译器。`setup.py:34-69`（`CMakeBuild::build_extension`）会为每个目标架构分别生成 `_C_sm<arch>.abi3.so`。

## 17.2 宏链逐层做了什么

定义集中在 `src/utils/utils.h:34-97`（宏 `HPC_ARCH_UNEVAL`、`HPC_ARCH_EVAL_*`、`HPC_ARCH_DISPATCH`）：

| 宏 | 位置 | 作用 |
|---|---|---|
| `HPC_ARCH_NARGS` | `src/utils/utils.h:65-66`（宏 `HPC_ARCH_NARGS`） | 统计 variadic 参数个数；本例 `90, expression` 共 2 个 |
| `HPC_ARCH_CAT` | `src/utils/utils.h:62-63`（宏 `HPC_ARCH_CAT`） | 把 `HPC_ARCH_EVALS_` 和数字 2 拼成 `HPC_ARCH_EVALS_2` |
| `HPC_ARCH_EVALS_2/4/6/8/10` | `src/utils/utils.h:68-80`（宏族） | 每次消费一个 `(arch, expression)` pair，并递归处理剩余 pair |
| `HPC_ARCH_EVAL_90/100/103` | `src/utils/utils.h:44-60`（宏族） | 由编译时 `HPC_TARGET_ARCH` 决定该架构 pair 是 live 还是 unevaluated |
| `HPC_ARCH_IMPL_STR_*` | `src/utils/utils.h:82-86`（宏族） | 只提取架构号，生成错误信息中的 `"90"` 或 `"90, 100"` |
| `HPC_ARCH_DISPATCH__` | `src/utils/utils.h:90-97`（宏 `HPC_ARCH_DISPATCH__`） | 建立 dispatched 标志、执行匹配 pair、未命中时抛错 |

所以它看起来啰嗦，主要是因为 C 预处理器没有“遍历任意数量 pair”的原生语法；这里用固定的 `2/4/6/8/10` 宏递归模拟了最多五个架构 pair。

## 17.3 `HPC_TARGET_ARCH=90` 时的实际展开

### 17.3.1 先展开参数个数和 token 拼接

`HPC_ARCH_DISPATCH` 在 `src/utils/utils.h:88-89`（宏 `HPC_ARCH_DISPATCH`、`HPC_ARCH_DISPATCH_`）先变成：

```cpp
HPC_ARCH_DISPATCH__(
    "group_gemm_blockwise_fp8", 2,
    90,
    group_gemm_blockwise_fp8_async(/* 22 个实参 */));
```

这里 `HPC_ARCH_NARGS` 数的是 variadic 参数，而不是函数实参：外层的 `group_gemm_blockwise_fp8_async(...)` 有很多逗号，但它们位于括号内，整体仍是一个宏参数。因此 `n=2`。

随后 `HPC_ARCH_CAT(HPC_ARCH_EVALS_, 2)` 展开成 `HPC_ARCH_EVALS_2`，`HPC_ARCH_EVALS_2` 在 `src/utils/utils.h:68`（宏 `HPC_ARCH_EVALS_2`）变成 `HPC_ARCH_EVAL_90(expression)`。

### 17.3.2 `HPC_TARGET_ARCH=90` 选择 live 分支

因为 `HPC_TARGET_ARCH == 90`，`src/utils/utils.h:44-45`（宏 `HPC_ARCH_EVAL_90`）定义为 `HPC_ARCH_EVAL_LIVE(90, expression)`；`src/utils/utils.h:36-42`（宏 `HPC_ARCH_EVAL_LIVE`）再展开为运行时架构检查。

把本例的完整表达式代回后，预处理结果等价于：

```cpp
do {
  bool hpc_arch_dispatched_ = false;
  do {
    if (::hpc::get_sm_arch() == (90)) {
      group_gemm_blockwise_fp8_async(
          y_ptr, x_ptr, weight_ptr, seqlens_ptr, cu_seqlens_ptr, xscale_ptr, wscale_ptr, tmas_ptr,
          tiles_ptr, cu_tiles_ptr, task_map_ptr, num_waves, num_group, m, n, k, m_pad,
          num_block_k_pad4, num_seq_per_group_avg, update_tma, false, stream);
      hpc_arch_dispatched_ = true;
    }
  } while (0);
  if (!hpc_arch_dispatched_) {
    ::hpc::throw_arch_not_supported("group_gemm_blockwise_fp8", "90");
  }
} while (0);
```

这里有两个 `do { ... } while (0)`：内层来自 `HPC_ARCH_EVAL_LIVE`，外层来自 `HPC_ARCH_DISPATCH__`。它们都只执行一次；用途是让整个宏在 `if (...) HPC_ARCH_DISPATCH(...); else ...` 等语境中表现为一个安全的单语句。

## 17.4 `HPC_TARGET_ARCH=100/103` 时的展开

如果同一份 `entry.cc` 被编译为 sm100 模块，`src/utils/utils.h:46-48`（宏 `HPC_ARCH_EVAL_90`）不会选择 live 分支，而是选择：

```cpp
#define HPC_ARCH_EVAL_90(expr) HPC_ARCH_UNEVAL(expr)
```

于是本例等价于：

```cpp
do {
  bool hpc_arch_dispatched_ = false;
  static_cast<void>(sizeof((
      group_gemm_blockwise_fp8_async(
          y_ptr, x_ptr, weight_ptr, seqlens_ptr, cu_seqlens_ptr, xscale_ptr, wscale_ptr, tmas_ptr,
          tiles_ptr, cu_tiles_ptr, task_map_ptr, num_waves, num_group, m, n, k, m_pad,
          num_block_k_pad4, num_seq_per_group_avg, update_tma, false, stream),
      0)));
  if (!hpc_arch_dispatched_) {
    ::hpc::throw_arch_not_supported("group_gemm_blockwise_fp8", "90");
  }
} while (0);
```

`HPC_ARCH_UNEVAL` 在 `src/utils/utils.h:34`（宏 `HPC_ARCH_UNEVAL`）使用 `sizeof` 的未求值操作数：

- `group_gemm_blockwise_fp8_async(...)` 会被解析和类型检查，因此声明/参数类型错误仍能在编译时发现；
- 函数不会被执行，也不会生成这次调用的运行时代码；
- 因而 sm100/103 模块不需要链接只存在于 sm90 源目录中的实现。

这不是把调用“注释掉”：它保留了编译期接口检查，但去除了运行时求值。CMake 的未定义符号保护在 `CMakeLists.txt:200-204`（目标 `target_link_options`）启用 `-Wl,--no-undefined`；如果非目标架构分支是普通函数调用而不是 `sizeof`，链接阶段就可能直接因为缺少 sm90 实现失败。

## 17.5 如果同时写多个架构 pair

宏的通用形式可以写成：

```cpp
HPC_ARCH_DISPATCH("op", 90, expr90, 100, expr100);
```

`HPC_ARCH_NARGS` 得到 4，`src/utils/utils.h:69-71`（宏 `HPC_ARCH_EVALS_4`）展开为概念上的：

```cpp
HPC_ARCH_EVAL_90(expr90);
HPC_ARCH_EVAL_100(expr100);
```

在 sm90 模块中，`expr90` 是 live、`expr100` 是 `sizeof`；在 sm100 模块中相反。错误字符串由 `src/utils/utils.h:83-86`（宏 `HPC_ARCH_IMPL_STR_4`）生成 `"90, 100"`。因此架构之间不仅可以调用不同的 `*_async` 函数，也可以传入不同的参数准备或 launch 表达式；这比强行要求所有架构共享一个函数指针更灵活。

## 17.6 运行时到底检查哪一个架构

### 17.6.1 编译期和运行时是两道不同的筛选

这套宏有两个层次：

1. **编译期筛选**：`HPC_TARGET_ARCH` 决定哪个 `HPC_ARCH_EVAL_<arch>` 是 live，其他 pair 进入 `sizeof`；
2. **运行时筛选**：live pair 内部调用 `::hpc::get_sm_arch()`，只有实际 GPU 架构等于 pair 中的数字才执行表达式。

`src/utils/utils.cc:47-58`（函数 `get_sm_arch`）通过 `cudaGetDeviceProperties` 计算 `major * 10 + minor`，并缓存结果。`src/utils/utils.cc:61-67`（函数 `throw_arch_not_supported`）在没有任何 live pair 成功时给出实际设备、支持列表和已加载模块架构。

`hpc_arch_dispatched_` 只表示“某个架构条件命中并执行了 expression”，不表示 expression 自己返回成功。当前 `group_gemm_blockwise_fp8_async` 在 `src/group_gemm/group_gemm.h:21-27`（函数声明）返回 `void`，所以这个布尔值足以表示本次分支已被选中；若某个其它 operator 的 async 函数返回 `bool`，调用者仍需像 `src/gemm/entry.cc:142-148`（函数 `gemm_bf16xfp32_entry`）那样另行检查返回值。

### 17.6.2 本例的几种情况

| 编译模块 | 实际设备 | `group_gemm_blockwise_fp8_async` 是否执行 | 结果 |
|---|---:|---:|---|
| sm90 | sm90 | 是 | 设置 `hpc_arch_dispatched_=true`，正常返回 |
| sm90 | sm100 | 否 | bool 保持 false，抛出不支持错误 |
| sm100 | sm90 | 否（90 pair 在 `sizeof` 中） | 抛出不支持错误 |
| sm100 | sm100 | 本调用没有 sm100 pair | 抛出不支持错误 |

Python 层通常在 `hpc/__init__.py:58-95`（函数 `_load_module`）按 NVML/设备能力选择匹配的 `_C_sm<arch>.abi3.so`，但 C++ 宏的运行时检查仍是防御性保障：它覆盖 `HPC_SM_ARCH` 覆盖、手工加载错误模块或多设备环境不一致等情况。

## 17.7 为什么要设计得这么“啰嗦”

### 17.7.1 避免每个模块链接不属于自己的 kernel

blockwise 实现当前在 `src/group_gemm/sm90/group_gemm_blockwise_fp8.cu:369-458`（函数 `group_gemm_blockwise_fp8_async`）的 sm90 源目录中。CMake 为每个目标架构筛选源文件；如果入口直接写普通调用，sm100/103 编译单元仍可能产生对 sm90 符号的引用，最终与 `-Wl,--no-undefined` 冲突。`sizeof` 分支正是为解决这个链接边界而存在。

### 17.7.2 允许各架构使用不同表达式

宏参数是任意 C++ expression，而不是固定函数名。未来可以写成 `90, launch_sm90(...)`、`100, launch_sm100(...)`，甚至让不同架构使用不同模板实例、不同配置或不同返回值处理。`HPC_ARCH_EVALS_*` 负责遍历 pair，调用点仍保持一个统一入口。

### 17.7.3 同时提供运行时安全检查和清晰错误

仅靠编译期选择无法阻止用户加载错误的 `.so`；仅靠运行时 `switch` 又会让所有实现都进入链接依赖。当前组合让错误在调用点立即变成：`hpc::<op> does not run on sm<actual>: implemented for <list>, loaded module built for sm<target>`，而不是更晚的非法 kernel launch 或难以解释的 unresolved symbol。

### 17.7.4 让所有架构专用入口遵循同一模式

`src/utils/utils.h:5-7`（文件级架构 dispatch 说明）明确规定：架构专用 operator 使用 `HPC_ARCH_DISPATCH`，真正对所有架构都有共享实现的 operator 才直接调用 `*_async`。因此当前 blockwise 只有一个 `90` pair 时看起来有些过度封装，但它与整个仓库的多架构模块模型一致。

## 17.8 这层宏带来的实际代价和限制

### 17.8.1 只有 host launch 路径有少量开销

sm90 模块的最终代码多出一次 `get_sm_arch()` 判断、一个 bool 赋值和一次失败检查；`get_sm_arch()` 首次调用才查询 CUDA 属性，之后走缓存原子变量（`src/utils/utils.cc:47-58`，函数 `get_sm_arch`）。这些操作发生在 host 的 Python/C++ API 调用路径，不会加入 GPU kernel 内部，也不会影响 WGMMA/TMA 的每元素性能。

### 17.8.2 只支持固定数量的 pair

`src/utils/utils.h:68-80`（宏 `HPC_ARCH_EVALS_*`）目前只定义了 2、4、6、8、10 个 variadic 参数，即最多五个 `(arch, expression)` pair；参数必须成对出现。表达式内的逗号必须处在括号、函数调用或其它宏参数保护结构中，否则 `HPC_ARCH_NARGS` 会误计数。

### 17.8.3 单架构、共享实现时可以更简单

如果某个 `*_async` 实现在所有已知架构都存在，仓库说明建议直接调用，而不是为了形式统一套一层 dispatch。对当前只有 sm90 实现的 blockwise operator，直接写：

```cpp
group_gemm_blockwise_fp8_async(
    y_ptr, x_ptr, weight_ptr, seqlens_ptr, cu_seqlens_ptr, xscale_ptr, wscale_ptr, tmas_ptr,
    tiles_ptr, cu_tiles_ptr, task_map_ptr, num_waves, num_group, m, n, k, m_pad,
    num_block_k_pad4, num_seq_per_group_avg, update_tma, false, stream);
```

在“永远只构建并加载 sm90、且不需要错误架构诊断”的前提下可以工作；但它会放弃上述源文件隔离、运行时设备校验和统一扩展点，所以当前仓库选择保留宏。

## 17.9 最终结论

- 本次一个 pair 的 `HPC_ARCH_DISPATCH` 在 `HPC_TARGET_ARCH=90` 下最终就是“`get_sm_arch()==90` 时调用 `group_gemm_blockwise_fp8_async`，否则抛错”的嵌套 `do/while` 代码。
- 在非 sm90 模块中，同一表达式变成 `sizeof((expression,0))`：只做编译期类型检查，不执行、不产生该函数调用的链接依赖。
- 宏的冗长部分来自可变参数 pair 计数、token 拼接、递归展开、错误字符串生成和语句安全包装；它服务的是一个 `.cc` 源文件对应多个架构 `.so` 的构建模型。
- 对当前 API 调用，宏开销只在 host launch 路径上，主要是一次缓存架构查询和比较；不会增加 blockwise GPU kernel 的运行时指令。

# 18. `btma_as.partition_D(sAS)` 中第三个 mode 的来源

## 18.1 先给本次实例的结论

本次 `temp/block.log:1-3` 的编译期参数是 `kTileM=48`、`kTileS=64`、`kStage=8`。因此：

本仓库当前 checkout 将原先的 `src/group_gemm/kernels.cuh` 放在 `src/group_gemm/sm90/kernels.cuh`；下文按当前 code base 的实际相对路径引用。

```text
sAS 的逻辑 shape                 = (8, 64)
TMA x-scale box 的 Tiler_MN      = (1, 48)
tASs 的实际 shape                 = ((48, 1), 8, 2)
```

第三个顶层 mode 的 `_2` 在这个实例中确实来自：

```text
ceil_div(kTileS, kTileM) = ceil_div(64, 48) = 2
```

但要准确理解它，必须把它称为 **TMA tile 之外的 rest/tile-count mode**。它不是 TMA descriptor 的第三个硬件维度，不是 `kWarpgroupM=2`，也不是 double buffering；它也不会让一次 `cute::copy` 自动执行两次 TMA copy。当前调用把这个 mode 显式固定为 `0`。

## 18.2 `sAS` 的 shape 和 stride 从哪里来

### 18.2.1 配置类型定义的两个维度

`src/group_gemm/sm90/config.h:136-137`（类型 `GroupGEMMBlockWiseFp8Config`）定义：

```cpp
using SLayoutXS = decltype(make_layout(
    make_shape(Int<kStage>{}, Int<kTileS>{}),
    make_stride(Int<kTileS>{}, Int<1>{})));
```

在本次实例中 `kStage=8`、`kTileS=64`，所以它是：

```text
SLayoutXS = Layout<Shape<_8,_64>, Stride<_64,_1>>
```

`src/group_gemm/sm90/kernels.cuh:679-682`（函数 `group_gemm_blockwise_fp8_kernel`）用该 layout 和 shared-memory 指针构造 `sAS`。日志 `temp/block.log:90-93` 打印为：

```text
smem_ptr[32b](...) o (_8,_64):(_64,_1)
```

这里：

- 第 0 维 `_8` 是 8 个 pipeline stage；`ismem_read/ismem_write` 选择这一维。
- 第 1 维 `_64` 是每个 stage 预留的 x-scale slot 容量（`kTileS`）。
- `(64,1)` 是 row-major 的 shared-memory 地址映射，所以 `sAS(stage, slot)` 的线性 offset 是 `stage * 64 + slot`。

`kTileS` 是 shared buffer 的容量参数，不等于本次 TMA 实际传输的第二维；本次传输宽度由 `kTileM=48` 决定。

### 18.2.2 本次 `kTileM` 的来源

`src/group_gemm/sm90/group_gemm_blockwise_fp8.cu:403-417`（函数 `group_gemm_blockwise_fp8_async`）固定 `kTileS=64`；本次日志的平均长度为 72，因此命中 `src/group_gemm/sm90/group_gemm_blockwise_fp8.cu:459-466`（函数 `group_gemm_blockwise_fp8_async` 的 `kTileM=48` 分支）调用模板 kernel。日志中的 `kTileM:48` 与此一致。

## 18.3 TMA 的 `Tiler_MN` 为什么是 `(1,48)`

### 18.3.1 `CopyBoxXS` 定义的是一个 TMA box

`src/group_gemm/sm90/config.h:142-145`（类型 `GroupGEMMBlockWiseFp8Config`）定义：

```cpp
using CopyBoxXS = decltype(make_layout(
    make_shape(Int<1>{}, Int<kTileM>{}),
    make_stride(Int<kTileM>{}, Int<1>{})));
```

本次即 `shape=(1,48)`、`stride=(48,1)`。`src/group_gemm/sm90/config.h:147-154`（成员函数 `GroupGEMMBlockWiseFp8Config::get_tma`）把它传给：

```cpp
auto tma_xs = make_tma_copy(SM90_TMA_LOAD{}, xs, CopyBoxXS{});
```

### 18.3.2 三参数 `make_tma_copy` 的默认行为

`3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:1341-1352`（函数 `make_tma_copy`）把 `product_each(shape(slayout))` 作为 CTA tiler。因此 `CopyBoxXS` 的 shape 被用作 TMA 的 tile shape，得到：

```text
Tiler_MN = (1, 48)
```

日志 `temp/block.log:47-56` 的 `tma_xs` 也直接显示：

```text
Tiler_MN:       (_1,_48)
ThrID:          _1:_0
ValLayout...    (_1,_48):(_0,_1)
ValueType:      32b
```

这表示一个非 multicast TMA atom 负责一个逻辑线程、48 个 `float` 值（1536 bit），而不是负责整个 `(8,64)` 的 shared tensor。

## 18.4 `get_slice(0)` 和 `partition_D` 的调用链

### 18.4.1 `get_slice(0)` 只选择逻辑 TMA thread

`src/group_gemm/sm90/kernels.cuh:692-695`（函数 `group_gemm_blockwise_fp8_kernel`）执行：

```cpp
auto btma_as = tma_as.get_slice(0);
```

`3rd/cutlass/include/cute/atom/copy_atom.hpp:338-345`（函数 `TiledCopy::get_slice`）返回一个 `ThrCopy`，保存 `thr_idx_=0`。TMA load atom 的 `ThrID=Layout<_1>` 定义在 `3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:101-110`，所以这里的 `0` 只是在唯一逻辑 TMA thread 中选 thread 0；它没有选择 stage，也没有把 rest mode 变成 1。

### 18.4.2 `partition_D` 的实际展开

`btma_as.partition_D(sAS)`（`src/group_gemm/sm90/kernels.cuh:703-704`，函数 `group_gemm_blockwise_fp8_kernel`）调用 `3rd/cutlass/include/cute/atom/copy_atom.hpp:375-383`（函数 `ThrCopy::partition_D`）。该函数先构造 `tidfrg_D(sAS.layout())`，最后再以 `thr_idx_` 切出一个线程 slice。

`tidfrg_D` 的核心在 `3rd/cutlass/include/cute/atom/copy_atom.hpp:239-248`（函数 `TiledCopy::tidfrg_D`）：

```cpp
tile2thrfrg(
    zipped_divide(dtensor, Tiler_MN{}),
    right_inverse(AtomLayoutRef{}).compose(AtomLayoutDst{}));
```

这不是读写 shared memory 的运行时操作，而是构造一个带新 layout 的 Tensor view。

## 18.5 `zipped_divide` 如何产生 `_8` 和 `_2`

### 18.5.1 先看不做 atom 映射的中间结果

对本次 `sAS` 和 `Tiler_MN`，可以把 `zipped_divide` 的中间结果写成：

```text
zipped_divide(sAS, (1,48))
  shape  = ((1,48), (8,2))
  stride = ((0, 1), (64,48))
```

第一组 `((1,48))` 是一个 TMA tile，第二组 `((8,2))` 是 tile 外的 rest modes。其含义是：

```text
stage rest = 8 / 1 = 8
slot  rest = ceil_div(64, 48) = 2
```

`3rd/cutlass/include/cute/tensor_impl.hpp:930-942`（函数 `zipped_divide` 的 Tensor overload）先保留原 Tensor 的 data pointer，再把 layout 运算应用到它；`3rd/cutlass/include/cute/layout.hpp:1606-1614`（函数 `zipped_divide`）保留 tile/rest 的分组；`3rd/cutlass/include/cute/layout.hpp:1555-1578`（函数 `logical_divide`）建立逻辑除法；非整除时，`3rd/cutlass/include/cute/layout.hpp:1216-1223`（函数 `detail::complement`）通过 `ceil_div` 计算 rest shape。因此 `_2` 是静态 layout 运算的结果，不是 kernel 中执行的一条 `ceil` 指令。

### 18.5.2 atom 的 TV 映射为什么让第一个 mode 变成 `(_48,_1)`

TMA `TiledCopy` 的日志 layout 是：

```text
TiledLayout_TV: (_1,((_48,_1))):(_0,((_1,_0)))
```

`3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:1193-1232`（函数 `detail::make_tma_copy_tiled`）构造这个 TV layout；`3rd/cutlass/include/cute/atom/copy_atom.hpp:229-237`（函数 `TiledCopy::tidfrg_D` 的布局约定）把 value 部分解释为 `(FrgV,FrgX)`；其中：

- 最外层 `_1` 是唯一逻辑 TMA thread；
- `((_48,_1))` 是 `(FrgV,FrgX)` 的嵌套 mode，乘积仍为 48；本例一个 TMA atom 有 `FrgV=48`、`FrgX=1`，所以内层 `_1` 是退化的 atom-replica mode，不是额外的值；
- 对应 stride `((_1,_0))` 表示 value 方向每步加 1，退化 mode 的 stride 为 0。

`ThrCopy::partition_D` 选择 thread 0 后，最外层 thread mode 被切掉，于是最终打印为：

```text
((_48,_1),_8,_2):((_1,_0),_64,_48)
```

换句话说，`tidfrg_D` 在切出逻辑 thread 之前可概念性地写成：

```text
(_1, (_48,_1), (_8,_2)) : (_0, (_1,_0), (_64,_48))
```

`ThrCopy::partition_D` 代入 `thr_idx_=0` 后去掉最外层 `_1`，正好得到日志中的 `((_48,_1),_8,_2)`。因此用户所说的“在 TMA 之外先得到 `(8,2)`，再加上 TMA mode”在这个实例中是正确的；只是 TMA mode 本身保留了 `(_48,_1)` 的嵌套结构。

所以日志中的“外层三个 mode”应这样读：

1. `(_48,_1)`：单次 TMA atom 的 48-value fragment（带一个退化子 mode）；
2. `_8`：shared 的 stage rest mode；
3. `_2`：沿每个 64-slot stage 按 48-value tile 分割得到的 rest tile 数。

## 18.6 最终 `tASs` 的地址公式

对最终 view，令第一个 nested mode 的有效 value 坐标为 `v=0..47`，stage 为 `s=0..7`，rest 为 `r=0..1`，由日志中的 shape/stride 可写成：

```text
tASs(v, s, r) -> shm_as[s * 64 + r * 48 + v]
```

即：

```text
tASs(_, 0, 0) -> offsets 0..47
tASs(_, 0, 1) -> offsets 48..95
tASs(_, 1, 0) -> offsets 64..111
```

这也说明了一个 CuTe layout 的重要性质：**view 的 shape 不等于每个坐标组合都已经经过边界检查**。当 tile 宽度 48 不能整除行宽 64 时，`rest=1` 的完整 48-value 视图会超出当前 64-slot 行，并在物理地址上进入后续 stage。`partition_D` 本身不会替调用者插入 shared-memory predicate。

`src/group_gemm/sm90/kernels.cuh:670-672`（函数 `group_gemm_blockwise_fp8_kernel`）把 `shm_bs` 放在 `shm_as + cosize(SLayoutAS{})` 之后；本例 `SLayoutAS` 需要 `8*64=512` 个 float。因此对最后一个 stage 使用 `rest=1` 时，地址还会越过 `sAS` 的 512-float 区域，进入后面的 `sBS` 区域。这正是当前代码必须显式选择 `rest=0` 的原因之一。

## 18.7 为什么实际 copy 只用第三索引 `0`

### 18.7.1 当前调用显式固定 rest tile

`src/group_gemm/sm90/kernels.cuh:880-883`（函数 `group_gemm_blockwise_fp8_kernel`）的 x-scale copy 是：

```cpp
cute::copy(tma_as.with(readable[ismem_write]),
           tASg(_, itile_k, cu_tiles_ptr[igroup] + itile_m),
           tASs(_, ismem_write, 0));
```

这里最后的 `0` 就是第三个顶层 mode `r`。因此每个 `(itile_k, stage)` 只选择第一块 48 个 slot；`r=1` 没有被传给 `cute::copy`。

### 18.7.2 一次切片后的 `cute::copy` 不会因为 `_2` 自动循环

切片后 source 和 destination 都只剩 TMA fragment mode。`3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:64-90`（函数 `TMA_LOAD_Unpack::copy_unpack`）取选定 source coordinate 和 destination shared pointer，调用一个 TMA copy atom。`_2` 只是原始 view 中可供调用者选择的 rest 坐标；只有调用者显式遍历/保留该 mode，才会有多次 copy 调用。

### 18.7.3 `tASg` 的两个 rest mode 是另一回事

日志 `temp/block.log:164-167` 中：

```text
tASg ... shape (((_48,_1),56,15))
```

这里 `56` 是 `num_block_k`，`15=ceil(720/48)` 是 global `m_pad` 被 48-wide TMA box 切成的 tile 数。调用中的 `itile_k` 和 `cu_tiles_ptr[igroup]+itile_m` 分别选择这两个 global rest 坐标；它们与 destination `sAS` 的 `_8,_2` 是不同 Tensor 上的不同坐标域。

## 18.8 为什么 `_2` 不影响本次计算结果

### 18.8.1 MMA tile 只需要 48 个 x-scale

`src/group_gemm/sm90/kernels.cuh:686-688`（函数 `group_gemm_blockwise_fp8_kernel`）构造的 `gC` shape 是 `(kTileN,kTileM)=(128,48)`。随后 `src/group_gemm/sm90/kernels.cuh:916-918`（函数 `group_gemm_blockwise_fp8_kernel`）从它生成 identity/fragment 坐标；在 `src/group_gemm/sm90/kernels.cuh:962-968`（函数 `group_gemm_blockwise_fp8_kernel`）读取 x-scale 时：

```cpp
sAS(ismem_read, get<1>(tI_mn(0, in)))
```

第二坐标只覆盖 `0..kTileM-1 = 0..47`，所以不会读取 shared stage 中的 `slot=48..63`。这 16 个 slot 是 `kTileS=64` 为其它 `kTileM` 配置预留的容量。

### 18.8.2 transaction bytes 也只计算 48 个 x-scale

`src/group_gemm/sm90/kernels.cuh:813-814`（函数 `group_gemm_blockwise_fp8_kernel`）定义：

```cpp
(kTileM + 4) * sizeof(float)
```

其中 `kTileM=48`，即本次 barrier transaction 计入 48 个 x-scale float 和 4 个 w-scale float，而不是 64 个 x-scale float。这与 `tASs(_, ismem_write, 0)` 的实际使用完全一致。

## 18.9 `_1` 注释为何与日志 `_2` 不一致

`src/group_gemm/sm90/kernels.cuh:703-704`（函数 `group_gemm_blockwise_fp8_kernel`）的注释写的是：

```cpp
// (TMA, kStage, _1)
```

这应理解为“预期使用的主要 tile/rest 形态”或旧的简化注释，不是对所有模板实例的精确静态类型声明。当前真实类型由日志打印为：

```text
((_48,_1),_8,_2):((_1,_0),_64,_48)
```

在本实现支持的 `kTileM<=64` 范围内，第三 mode 的通式是：

```text
ceil_div(kTileS, kTileM) = ceil_div(64, kTileM)
```

例如：

```text
kTileM=8   -> 8
kTileM=16  -> 4
kTileM=32  -> 2
kTileM=48  -> 2   (本次实例)
kTileM=64  -> 1
```

因此只有当前 `kTileM=64` 的实例会打印 `_1`；`kTileM=48` 打印 `_2` 是正常且可由 layout algebra 直接推导的结果。

## 18.10 对问题的直接回答

1. 是的，在本次 `kTileS=64`、`Tiler_MN` 第二维为 `48` 的实例中，第三个 mode 的大小就是 `ceil(64/48)=2`；更准确地说，它是 `zipped_divide` 为 destination shared layout 生成的 rest tile 数。
2. `sAS` 的 `_8` 不是由 TMA box 计算出来的，而是 `kStage=8` 除以 Tiler 第一维 `_1` 后保留的 stage rest mode。
3. 最外层 `(_48,_1)` 是 TMA atom 的 48-value fragment 的嵌套表示；它不是第三个 rest mode。
4. `tASs` 的完整静态 view 可以读成 `((TMA-value), stage, rest)`，即 `((_48,_1),_8,_2)`；但实际 x-scale copy 使用 `tASs(_, ismem_write, 0)`，每次只传第一块 48 个 float，不会自动传第二块。
5. 第三个 mode 若被错误地选为 `1`，`partition_D` 不会替你做边界保护；当前代码固定为 `0`，并且 MMA/transaction bytes 也都只依赖这 48 个有效 x-scale。

# 19. `group_gemm_blockwise_fp8_kernel` 第 881 行为何仍读取 `cu_tiles_ptr`

## 19.1 结论

`src/group_gemm/sm90/kernels.cuh:881`（函数 `group_gemm_blockwise_fp8_kernel`）读取的 `cu_tiles_ptr[igroup]` 是一个 **用于计算 TMA 坐标的整数前缀值**，不是 TMA 要搬运的 x-scale 数据本身。对于 `kTaskLoopPolicy == 2`，现有代码已经把同一份前缀数组复制到 `shm_tiles`，并在使用前执行了 CTA 级同步；因此从正确性上说，可以把这一策略的表达式改成 `shm_tiles[igroup] + itile_m`。

但是不能把所有策略都无条件改成 `shm_tiles[igroup]`：

1. policy 0 的 `shm_tiles` 元素是 `int4` task record；
2. policy 1 的 `shm_tiles` 元素是每个 group 的 tile 数量，不是 exclusive prefix；
3. 只有 policy 2 的 `shm_tiles[i]` 才与 `cu_tiles_ptr[i]` 逐项相等。

所以当前写法主要是一个适用于三个模板实例的公共路径写法，并不是 Hopper TMA 要求坐标必须从 global memory 读取。仓库中也没有注释说明“此处必须使用 global”；从代码历史和数据流看，更像是早期公共实现保留下来的保守写法。

## 19.2 两个数组分别是什么

### 19.2.1 `cu_tiles_ptr` 是 device global 的前缀数组

`src/group_gemm/entry.cc:178-191`（函数 `group_gemm_blockwise_fp8_entry`）用 `torch::empty` 创建长度为 `num_group + 1` 的 CUDA `int32` tensor，并把它的 device pointer 传入异步 launch。它不是 kernel 内的 shared-memory 地址。

`src/group_gemm/sm90/kernels.cuh:149-175`（函数 `update_grouped_tma`）先计算

```text
tiles_ptr[g] = ceil(seqlens[g] / kTileM)
```

随后对每个 block 的 `tiles` 做 `cub::BlockScan::ExclusiveSum`，再写入 `cu_tiles_ptr[g]`；`cu_tiles_ptr[num_group]` 是所有 group 的 tile-M 总数。因此：

```text
cu_tiles_ptr[g] = sum(tiles_ptr[0 .. g-1])
```

它给出了 group `g` 在拼接后的 padded-M tile 空间中的起始 tile。

`src/group_gemm/sm90/group_gemm_blockwise_fp8.cu:165-245`（函数 `launch_group_gemm_blockwise_fp8`）先在同一 CUDA stream 提交 `update_grouped_tma`，再提交 GEMM kernel；所以 GEMM 使用时这个 global 前缀数组已经被生产出来。PDL 路径还在 `src/group_gemm/sm90/kernels.cuh:770-772`（函数 `group_gemm_blockwise_fp8_kernel`）执行 grid dependency 同步。

### 19.2.2 `shm_tiles` 是 policy 相关的动态 shared 工作区

`src/group_gemm/sm90/kernels.cuh:666-674`（函数 `group_gemm_blockwise_fp8_kernel`）把 `extern __shared__` 区域尾部解释成 `Ttask *shm_tiles`，其中

```cpp
using Ttask = std::conditional_t<kTaskLoopPolicy == 0, int4, int>;
```

它是每个 CTA 私有的临时副本，不是跨 CTA 或跨 kernel 的持久数组。policy 2 的 launch 额外预留 `(num_group + 1) * sizeof(int)`，见 `src/group_gemm/sm90/group_gemm_blockwise_fp8.cu:383-397`（函数 `launch_group_gemm_blockwise_fp8`）。

三种 policy 的写入内容不同：

| policy | `shm_tiles` 内容 | 用途 |
|---|---|---|
| 0 | `int4` task records，见 `src/group_gemm/sm90/kernels.cuh:774-793`（函数 `group_gemm_blockwise_fp8_kernel`） | 直接保存 `(itile_m, itile_n, igroup)` 等任务信息 |
| 1 | `tiles_ptr[i]`，见 `src/group_gemm/sm90/kernels.cuh:794-797`（函数 `group_gemm_blockwise_fp8_kernel`） | 让 horizon scheduler 累加每组 tile 数 |
| 2 | `cu_tiles_ptr[0..num_group]`，见 `src/group_gemm/sm90/kernels.cuh:798-803`（函数 `group_gemm_blockwise_fp8_kernel`） | 让 vertical scheduler 二分查找 exclusive prefix |

## 19.3 第 881 行到底在计算什么

### 19.3.1 TMA 数据源和坐标元数据要分开看

`src/group_gemm/sm90/kernels.cuh:689-704`（函数 `group_gemm_blockwise_fp8_kernel`）建立 `gAS` 和 `tASg`：`gAS` 是全局 x-scale tensor，`tASg` 是它按 TMA tile 切分后的 view。第 881 行的核心表达式是：

```cpp
tASg(_, itile_k, cu_tiles_ptr[igroup] + itile_m)
```

其中：

- `itile_k` 选择 x-scale 的 K-block；
- `itile_m` 是当前 group 内的局部 M tile；
- `cu_tiles_ptr[igroup]` 把局部 M tile 转成拼接后全局 padded-M tile 的起点；
- 两者相加后才是 `gAS` 第二维的 TMA tile 坐标。

随后 `cute::copy` 使用 `tma_as` 将该 global x-scale tile 搬到 `tASs` 指向的 shared stage。也就是说，`cu_tiles_ptr[igroup]` 只参与 **地址/坐标计算**；把它换成从 shared 读取的整数，不会把 x-scale 的 TMA 数据源改成 shared，也不会改变 TMA descriptor。

### 19.3.2 policy 2 下二者确实相等

在 policy 2 中，`src/group_gemm/sm90/kernels.cuh:798-806`（函数 `group_gemm_blockwise_fp8_kernel`）执行：

```cpp
total_m = cu_tiles_ptr[num_group];
for (int i = idx; i < num_group + 1; i += blockDim.x) {
  shm_tiles[i] = cu_tiles_ptr[i];
}
__syncthreads();
```

因此对任意合法任务 `igroup < num_group`：

```text
shm_tiles[igroup] == cu_tiles_ptr[igroup]
```

`src/group_gemm/sm90/kernels.cuh:43-67`（函数 `get_next_tile_vert`）也把参数名写成 `cu_tiles_ptr`，但调用点 `src/group_gemm/sm90/kernels.cuh:848-853`（函数 `group_gemm_blockwise_fp8_kernel`）传入的实际对象是 `shm_tiles`。这说明该参数只是“prefix array”语义，变量名并不强制它必须位于 global memory。

`__syncthreads()` 位于所有 policy 的 shared 初始化之后；而 `shm_tiles` 在后续 scheduler 和 TMA load 阶段没有再写入。因此在当前 kernel 生命周期内，用 `shm_tiles[igroup]` 没有数据竞争或可见性问题。

## 19.4 为什么不能直接全局替换

### 19.4.1 policy 0：shared 中根本不是 prefix

policy 0 在 `src/group_gemm/sm90/kernels.cuh:774-793`（函数 `group_gemm_blockwise_fp8_kernel`）把 task record 写入 `shm_tiles[iwave * warp_count + iwarp]`。后续 `src/group_gemm/sm90/kernels.cuh:833-841`（函数 `group_gemm_blockwise_fp8_kernel`）按 `iwave` 取出 `task.x/task.y/task.z`。此时 `shm_tiles[igroup]` 既不是按 group 编排的数组，元素类型也不是单个整数前缀；直接替换会造成类型或语义错误。这个策略仍需使用 `cu_tiles_ptr[igroup]` 来把 task 的局部 `itile_m` 映射到全局 x-scale tile。

### 19.4.2 policy 1：数量不等于前缀

policy 1 的 `shm_tiles[i] = tiles_ptr[i]` 是本组 tile 数量。比如四个 group 的数量是 `[1, 1, 1, 2]`，exclusive prefix 是 `[0, 1, 2, 3, 5]`；对 group 3，`shm_tiles[3]` 为 `2`，但正确的起始偏移是 `cu_tiles_ptr[3] = 3`。因此 policy 1 直接改成 `shm_tiles[igroup] + itile_m` 会把 x-scale 坐标向前错移一个 tile。

### 19.4.3 `cu_tiles_ptr` 仍有其他必要用途

即使 policy 2 的第 881 行改用 shared，global 数组仍不能删除：policy 2 在 `src/group_gemm/sm90/kernels.cuh:798-801`（函数 `group_gemm_blockwise_fp8_kernel`）需要先从 `cu_tiles_ptr[num_group]` 得到 `total_m`，policy 0 在 `src/group_gemm/sm90/kernels.cuh:774-777`（函数 `group_gemm_blockwise_fp8_kernel`）也用它计算 `actual_tiles`。此外它还是 update kernel 与 GEMM kernel 之间的工作区。

## 19.5 当前写法的性能含义

### 19.5.1 这不是 384 个线程各读一次 global

第 881 行位于 `src/group_gemm/sm90/kernels.cuh:869-885`（函数 `group_gemm_blockwise_fp8_kernel`）的 `itile_k` 循环内，但 load 路径只有 `src/group_gemm/sm90/kernels.cuh:817-820`（函数 `group_gemm_blockwise_fp8_kernel`）选出的 load leader 执行 TMA 发起。因此它是每个 outer tile 的每个 K tile 至多一次 32-bit 元数据读取，而不是整个 CTA 的 384 次 coalesced global load。

本次 `temp/block.log:1-3` 的实例 `kTileM=48、kTileN=128、kTileK=128、num_block_k=56`，所以源码层面一个任务会在第 881 行重复使用同一个 group 前缀最多 56 次。实际 SASS 是否把不变加载提升到循环外，需要看编译器生成代码；不能仅凭 C++ 源码断言一定有 56 条 global load。

### 19.5.2 global 命中缓存时收益可能很小

本例 prefix 数组只有 `num_group + 1 = 9` 个 `int`，总共 36 字节，通常会留在 L1/L2。与此同时，`src/group_gemm/sm90/kernels.cuh:813-814`（函数 `group_gemm_blockwise_fp8_kernel`）按本例 FP8 输入计算每个 K tile 的 TMA transaction bytes 约为：

```text
(48 + 128) * 128 * 1 + (48 + 4) * 4 = 22,736 bytes
```

相比之下，第 881 行的前缀值只有 4 字节。因此换成 shared 可能降低元数据 load 的延迟，但通常不会显著改变总内存带宽；是否能改善端到端时间仍应通过 SASS 和 benchmark 验证。

### 19.5.3 shared 更快不等于当前表达式就最优

shared 读取通常比未命中的 device-global 读取延迟更低，但 policy 2 已经为 scheduler 把前缀数组复制到 shared，复制本身的代价由 `src/group_gemm/sm90/kernels.cuh:798-806`（函数 `group_gemm_blockwise_fp8_kernel`）承担。更重要的是，第 881 行的前缀对一个 outer tile 的所有 `itile_k` 都不变；如果编译器没有自动提升，反复计算才是可以直接消除的冗余。

## 19.6 推荐的低风险优化

不要在所有 specialization 中直接写 `shm_tiles[igroup]`。在 scheduler 成功得到 `igroup/itile_m` 后、`src/group_gemm/sm90/kernels.cuh:869-870`（函数 `group_gemm_blockwise_fp8_kernel`）进入 K 循环前，按编译期 policy 选择一次前缀，并把完整坐标留在寄存器中：

```cpp
int group_tile_begin;
if constexpr (kTaskLoopPolicy == 2) {
  group_tile_begin = shm_tiles[igroup];
} else {
  group_tile_begin = cu_tiles_ptr[igroup];
}
int xscale_tile_m = group_tile_begin + itile_m;

for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
  // A/B/scale TMA copies
  cute::copy(tma_as.with(readable[ismem_write]),
             tASg(_, itile_k, xscale_tile_m), tASs(_, ismem_write, 0));
}
```

`if constexpr` 会在 policy 0/1 的实例中编译掉 shared 分支，所以不会访问 `int4` task record；policy 2 则只做一次 shared 读取。这个改法保持三种策略的语义不变，并把潜在的每-K-tile global load 降为每任务一次。

对 policy 2 还可以做更激进的改法：`src/group_gemm/sm90/kernels.cuh:43-67`（函数 `get_next_tile_vert`）内部已经计算 `itile_m_total = iblock % total_m`，并返回 `itile_m = itile_m_total - shm_tiles[igroup]`；所以 `shm_tiles[igroup] + itile_m` 代数上就是 `itile_m_total`。若让 scheduler 同时返回这个 flattened tile index，就能完全省掉第 881 行的 prefix load，但这需要修改 helper 的接口，风险和验证成本高于前面的寄存器缓存方案。

## 19.7 对问题的直接回答

1. **为什么仍读 `cu_tiles_ptr`？** 当前第 881 行是三个 task-loop policy 共用的代码；只有 policy 2 的 shared 数组是 prefix，保留 global 读取可以让 policy 0/1 也使用同一表达式。代码没有证据表明这是 TMA 硬件的强制要求，更可能是公共路径/历史实现的选择。
2. **policy 2 能否改成 `shm_tiles`？** 能。第 798-806 行已复制相同的 `cu_tiles_ptr` 内容并同步，`shm_tiles[igroup] + itile_m` 与原表达式数值等价。
3. **shared 是否一定明显更快？** 单个 load leader 读取的 4 字节元数据很小，global 数组通常缓存命中；收益可能有限。把前缀和最终坐标在 K 循环外缓存到寄存器，比只替换地址空间更稳妥。
4. **能否删除 `cu_tiles_ptr`？** 不能。它仍用于构造 shared 快照、得到 `total_m`、计算 task 数量，并作为 update 与 GEMM 之间的 global workspace。
