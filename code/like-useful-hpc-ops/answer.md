# hpc.group_gemm_pertensor_fp8 完整调用链分析

## 1. CUDA 实现位置

CUDA kernel 实现在以下文件中：

| 文件 | 作用 |
|------|------|
| `src/group_gemm/kernels.cuh:143-424` | 主 CUDA kernel `group_gemm_pertensor_fp8_kernel` |
| `src/group_gemm/kernels.cuh:63-141` | TMA 描述符更新 kernel `update_grouped_tma` |
| `src/group_gemm/kernels.cuh:22-61` | Tile 调度辅助函数 `get_next_tile_horizon` / `get_next_tile_vert` |
| `src/group_gemm/group_gemm_pertensor_fp8.cu:16-91` | Kernel launch 封装 `launch_group_gemm_fp8` |
| `src/group_gemm/group_gemm_pertensor_fp8.cu:93-136` | 异步调度入口 `group_gemm_pertensor_fp8_async` |
| `src/group_gemm/entry.cc:15-74` | C++ PyTorch 算子入口 `group_gemm_pertensor_fp8_entry` |
| `src/group_gemm/config.h:63-107` | Tile / MMA / Swizzle 配置结构体 `GroupGEMMFp8Config` |
| `src/group_gemm/group_gemm.h:12-17` | `group_gemm_pertensor_fp8_async` 函数声明 |
| `src/utils/tma.cuh:36-57` | TMA 描述符更新工具函数 `update_tma_gtensor` |

## 2. 算子注册机制

算子通过 **PyTorch TorchScript `TORCH_LIBRARY_FRAGMENT`** 机制注册，分两步：

### 2.1 C++ 端注册 (TORCH_LIBRARY_FRAGMENT)

`src/group_gemm/entry.cc:192-198`

```cpp
TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "group_gemm_pertensor_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, Tensor "
      "y_scale, int num_seq_per_group_avg, Tensor? output, Tensor? tma_desc) -> (Tensor)");
  m.impl("group_gemm_pertensor_fp8", torch::kCUDA,
         &hpc::group_gemm::group_gemm_pertensor_fp8_entry);
}
```

- `m.def(...)` 声明算子签名（TorchScript 类型系统）
- `m.impl("group_gemm_pertensor_fp8", torch::kCUDA, ...)` 将 CUDA dispatch key 绑定到 `group_gemm_pertensor_fp8_entry` 函数
- 这段代码被编译进 `_C.*.so` 共享库

### 2.2 Python 端加载 .so

`hpc/__init__.py:43-45`

```python
so_files = list(Path(__file__).parent.glob("_C.*.so"))
assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"
torch.ops.load_library(so_files[0])
```

`torch.ops.load_library()` 加载共享库，触发其中所有 `TORCH_LIBRARY_FRAGMENT` 静态初始化，将算子注册到 `torch.ops.hpc` 命名空间下。

### 2.3 Python 端 fake kernel 注册 (torch.compile 支持)

`hpc/group_gemm.py:155-159`

```python
@torch.library.register_fake("hpc::group_gemm_pertensor_fp8")
def group_gemm_pertensor_fp8_fake(x, weight, seqlens, cu_seqlens, y_scale,
                                   num_seq_per_group_avg, output, tma_des):
    return torch.empty((x.shape[0], weight.shape[1]), dtype=torch.bfloat16)
```

这个 fake kernel 为 `torch.compile` 提供 shape/dtype 推断信息，在 tracing 阶段使用。

### 2.4 Python 端函数导出

`hpc/__init__.py:30-49`

```python
def _export_functions(modules):
    for module_name, module in modules.items():
        funcs = {
            name: obj for name, obj in vars(module).items()
            if callable(obj) and not name.startswith("_")
        }
        globals().update(funcs)
        __all__.extend(funcs.keys())

_export_functions(_discover_modules())
```

`_discover_modules()` 扫描 `hpc/` 下所有 `.py` 文件（排除以 `_` 开头的），import 后提取所有 callable，注入到 `hpc` 包的全局命名空间。

## 3. 完整调用链

```
Python: hpc.group_gemm_pertensor_fp8(x, weight, seqlens, cu_seqlens, y_scale, ...)
│
│   hpc/group_gemm.py:49-96 函数 group_gemm_pertensor_fp8
│   薄包装，直接转发到 torch.ops.hpc.group_gemm_pertensor_fp8
│
▼
torch.ops.hpc.group_gemm_pertensor_fp8(...)
│
│   由 TORCH_LIBRARY_FRAGMENT(hpc, m) 注册
│   src/group_gemm/entry.cc:192-198
│
▼
hpc::group_gemm::group_gemm_pertensor_fp8_entry(...)
│
│   src/group_gemm/entry.cc:15-74 函数 group_gemm_pertensor_fp8_entry
│   职责：
│     - 获取 CUDA stream (line 22)
│     - 校验设备/连续性/形状 (lines 23-31)
│     - 提取 m, k, n, num_group (lines 33-36)
│     - 分配/复用 output tensor (bfloat16) (lines 39-44)
│     - 分配/复用 TMA descriptor tensor (lines 46-53)
│     - 分配 tile / cu_tiles 临时 tensor (int32) (lines 55-56)
│     - 提取所有裸指针 (lines 58-67)
│     - 调用异步 dispatch (lines 69-71)
│
▼
hpc::group_gemm::group_gemm_pertensor_fp8_async(...)
│
│   src/group_gemm/group_gemm_pertensor_fp8.cu:93-136
│   函数 group_gemm_pertensor_fp8_async
│   职责：
│     - 固定编译期常量: kTileN=128, kTileK=128, kWarpgroupM=2,
│       kWarpgroupN=1, kSwizzleX=128, kSwizzleW=128, kSwizzleY=64 (lines 99-105)
│     - 根据 num_seq_per_group_avg 选择 kTileM 和 kStage (lines 107-135):
│         ● <=16  → kTileM=16, kStage=8
│         ● <=32  → kTileM=32, kStage=8
│         ● <=48  → kTileM=48, kStage=8
│         ● else  → kTileM=64, kStage=8
│     - 调用对应模板实例化的 launch_group_gemm_fp8
│
▼
hpc::group_gemm::launch_group_gemm_fp8<kTileM, kTileN, kTileK, ...>(...)
│
│   src/group_gemm/group_gemm_pertensor_fp8.cu:16-91
│   函数 launch_group_gemm_fp8
│   职责：
│     - 用 CuTe 创建 X(W, Y) 的 gmem tensor 视图 (lines 27-32)
│     - 通过 GroupGEMMFp8Config::get_tma() 创建 TMA copy 对象 (lines 34-37)
│     - Step 0: 若需要，启动 update_grouped_tma kernel
│       初始化每组的 TMA 描述符，计算 tile 数 (lines 42-55)
│     - Step 1: 启动 group_gemm_pertensor_fp8_kernel (lines 58-90)
│         ● block dim = 384 threads
│         ● grid dim = get_sm_count()
│         ● 动态共享内存 = config shm + sizeof(int) * (num_group + 1)
│         ● 根据 k <= 1024 || n <= 1024 选择 IsLoopH (水平循环)
│           或 IsLoopH = false (垂直循环)
│
▼
hpc::group_gemm::kernels::group_gemm_pertensor_fp8_kernel<Config, TmaA, TmaB, TmaD, IsLoopH>(...)
│
│   src/group_gemm/kernels.cuh:143-424
│   kernel group_gemm_pertensor_fp8_kernel
│   warp-specialized 架构（block=384 threads，其中 256 math + 128 load）:
│
│   初始化阶段 (lines 163-241):
│     - 布局共享内存: writable[kStage] / readable[kStage] barriers,
│       shm_a (X tile), shm_b (W tile), shm_c (accumulator), shm_tiles (tile metadata)
│     - 获取所有 group 的 TMA descriptor fence (lines 183-185)
│     - Leader thread 初始化 barriers (lines 207-213)
│     - 加载 tile count 到共享内存 (lines 228-236)
│     - 线程分叉: idx >= kNumThreads → load warpgroup, 否则 → math warpgroup
│
│   Load Warpgroup (lines 243-299):
│     - dealloc registers → 24
│     - Leader 线程循环调用 get_next_tile_horizon / get_next_tile_vert
│       找到当前 block 应处理的 (igroup, itile_m, itile_n)
│     - 对每个 k-tile: 用 TMA 异步拷贝 X/W tiles 到 shared memory
│     - 通过 barrier 通知 math warpgroup
│
│   Math Warpgroup (lines 301-423):
│     - alloc registers → 168
│     - 设置 CuTe TiledMma 及 A, B, C 片段
│     - 每 tile 循环:
│         1. 获取当前 group 的 yscale (line 347)
│         2. 等待 load warpgroup 完成 (barrier)
│         3. cute::gemm 在 warpgroup 内做 MMA (lines 359-361)
│         4. 应用 per-tensor scale: tDr(i) = tCr(i) * scale + tDr(i) (lines 373-375)
│         5. 通知 load warpgroup (barrier)
│     - FP32 → BF16 转换 (lines 385-390)
│     - SM90_U16x8_STSM_T 写回共享内存 (lines 396-406)
│     - Epilogue: leader warpgroup 用 TMA store 写 tile 到 global memory Y (lines 410-421)
│
▼
GPU 硬件执行: SM90 FP8 MMA (E4M3), TMA 异步拷贝, warpgroup barrier 同步
```

## 4. Tile 调度策略

`src/group_gemm/kernels.cuh:22-61`

两种调度模式根据问题规模自动选择（`src/group_gemm/group_gemm_pertensor_fp8.cu:69`）:

### 水平循环 (IsLoopH = true): k <= 1024 || n <= 1024
- `get_next_tile_horizon` (`src/group_gemm/kernels.cuh:22-40`)
- 每个 block 在 N 维度上依次取 tile，跨 group 工作
- 适合小规模问题

### 垂直循环 (IsLoopH = false): 其他情况
- `get_next_tile_vert` (`src/group_gemm/kernels.cuh:42-61`)
- 每个 block 固定 M tile，在 N 维度上迭代
- 通过 cu_tiles_ptr 二分查找对应的 group
- 适合大规模问题

## 6. TORCH_LIBRARY_FRAGMENT 宏详解

### 6.1 宏的功能

`TORCH_LIBRARY_FRAGMENT` 是 PyTorch 提供的自定义算子注册宏，用于在程序静态初始化阶段（`main()` 之前）向 PyTorch 的 dispatcher 注册自定义算子。

调用 `TORCH_LIBRARY_FRAGMENT(hpc, m) { ... }` 后，代码块 `{ ... }` 会在进程启动时自动执行，完成算子签名声明 (`m.def`) 和实现绑定 (`m.impl`)。整个过程无需 `main()` 手动调用任何初始化函数。

在 hpc-ops 项目中，`src/group_gemm/entry.cc:192-198` 使用该宏完成注册：

```cpp
TORCH_LIBRARY_FRAGMENT(hpc, m) {
  m.def(
      "group_gemm_pertensor_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, Tensor "
      "y_scale, int num_seq_per_group_avg, Tensor? output, Tensor? tma_desc) -> (Tensor)");
  m.impl("group_gemm_pertensor_fp8", torch::kCUDA,
         &hpc::group_gemm::group_gemm_pertensor_fp8_entry);
}
```

### 6.2 宏的实现位置

PyTorch 头文件：

| 文件 | 行号 | 内容 |
|------|------|------|
| `{conda_env}/lib/python3.12/site-packages/torch/include/torch/library.h` | 994-1002 | `TORCH_LIBRARY_FRAGMENT` 公开宏定义 |
| `{conda_env}/lib/python3.12/site-packages/torch/include/torch/library.h` | 1004-1023 | `_TORCH_LIBRARY_FRAGMENT` 内部宏展开 |
| `{conda_env}/lib/python3.12/site-packages/torch/include/torch/library.h` | 937-954 | `TorchLibraryInit` RAII 辅助类 |
| `{conda_env}/lib/python3.12/site-packages/torch/include/torch/library.h` | 546-555 | `torch::Library::Kind` 枚举 (DEF/IMPL/FRAGMENT) |
| `{conda_env}/lib/python3.12/site-packages/torch/include/c10/macros/Macros.h` | 100-118 | `C10_CONCATENATE`、`C10_UID`、`C10_STRINGIZE` 辅助宏 |

### 6.3 宏的完整展开

**第一步**：公开宏 (`torch/library.h:994-1002`)：

```cpp
#define TORCH_LIBRARY_FRAGMENT(ns, m) _TORCH_LIBRARY_FRAGMENT(ns, m, C10_UID)
```

其中 `C10_UID` (`c10/macros/Macros.h:108-111`)：

```cpp
#ifdef __COUNTER__
#define C10_UID __COUNTER__
#else
#define C10_UID __LINE__
#endif
```

**第二步**：内部宏 `_TORCH_LIBRARY_FRAGMENT` (`torch/library.h:1004-1023`)：

```cpp
#define _TORCH_LIBRARY_FRAGMENT(ns, m, uid)                           \
  static void C10_CONCATENATE(                                         \
      TORCH_LIBRARY_FRAGMENT_init_##ns##_, uid)(torch::Library&);      \
  static const torch::detail::TorchLibraryInit C10_CONCATENATE(        \
      TORCH_LIBRARY_FRAGMENT_static_init_##ns##_, uid)(                \
      torch::Library::FRAGMENT,                                        \
      &C10_CONCATENATE(TORCH_LIBRARY_FRAGMENT_init_##ns##_, uid),      \
      C10_STRINGIZE(ns),                                               \
      std::nullopt,                                                    \
      __FILE__,                                                        \
      __LINE__);                                                       \
  void C10_CONCATENATE(                                                \
      TORCH_LIBRARY_FRAGMENT_init_##ns##_, uid)(torch::Library & m)
```

**以 `src/group_gemm/entry.cc:192` 为例的实际展开**（假设 `__COUNTER__` = 42）：

```cpp
static void TORCH_LIBRARY_FRAGMENT_init_hpc_42(torch::Library&);
static const torch::detail::TorchLibraryInit TORCH_LIBRARY_FRAGMENT_static_init_hpc_42(
    torch::Library::FRAGMENT,                          // Kind: 片段模式
    &TORCH_LIBRARY_FRAGMENT_init_hpc_42,               // 初始化回调函数指针
    "hpc",                                              // 命名空间 (ns 字符串化)
    std::nullopt,                                       // 无 dispatch key (DEF 模式)
    "src/group_gemm/entry.cc",                         // __FILE__
    192);                                               // __LINE__
void TORCH_LIBRARY_FRAGMENT_init_hpc_42(torch::Library& m) {
    // 用户代码块
    m.def("group_gemm_pertensor_fp8(...) -> (Tensor)");
    m.impl("group_gemm_pertensor_fp8", torch::kCUDA, &...);
}
```

**执行流程**：

```
进程启动
  │
  ▼
全局静态对象构造 (main() 前)
  │
  ▼
TorchLibraryInit 构造函数
  │  torch/library.h:943-953
  │  : lib_(kind, ns, k, file, line) { fn(lib_); }
  │
  ├─► 1. 构造 torch::Library 对象 (Kind=FRAGMENT, ns="hpc")
  │      torch/library.h:561-566
  │
  └─► 2. 立即调用回调函数 fn(lib_)
         即 TORCH_LIBRARY_FRAGMENT_init_hpc_42(m)
         │
         ├─► m.def(...)  声明算子签名到 dispatcher
         └─► m.impl(...) 绑定 CUDA 实现到 dispatcher
```

### 6.4 `hpc` 和 `m` 参数的含义

| 参数 | 含义 | 传入值 | 展开后 |
|------|------|--------|--------|
| `ns` (第一个参数) | **算子命名空间**，必须是合法的 C++ 标识符 | `hpc` | 字符串 `"hpc"`，所有 `m.def("xxx")` 注册的算子都在 `hpc::xxx` 命名空间下 |
| `m` (第二个参数) | **`torch::Library` 引用变量名**，用于在代码块内调用注册方法 | `m` | 函数参数 `torch::Library& m`，通过它调用 `m.def()` / `m.impl()` |

因此在 Python 端，注册的算子通过 `torch.ops.hpc.group_gemm_pertensor_fp8` 访问：
- `hpc` → 对应宏的第一个参数（命名空间）
- `group_gemm_pertensor_fp8` → 对应 `m.def()` 中的算子名

### 6.5 为什么使用 `FRAGMENT` 而非 `TORCH_LIBRARY`

| 特性 | `TORCH_LIBRARY` | `TORCH_LIBRARY_FRAGMENT` |
|------|-----------------|--------------------------|
| Library::Kind | `DEF` | `FRAGMENT` |
| 同一 namespace 单文件可调用次数 | 1 次（变量名冲突） | 多次（C10_UID 防冲突） |
| 跨文件的同 namespace | 无法使用（违反单一定义） | 可以使用 |

`TORCH_LIBRARY` 限制每个 namespace 在整个程序中只能定义一个 Library 块。而 `TORCH_LIBRARY_FRAGMENT` 的 `C10_UID` 为每次宏调用生成唯一标识符，允许多个 `.cc` 文件各自使用 `TORCH_LIBRARY_FRAGMENT(hpc, m)` 向同一 `hpc` namespace 注册不同的算子。

在 hpc-ops 项目中，这正是必须使用 `FRAGMENT` 的原因——多个模块的文件各自注册算子到同一个 `hpc` namespace：

| 文件 | 注册的算子 |
|------|-----------|
| `src/group_gemm/entry.cc:192` | `group_gemm_pertensor_fp8`, `group_gemm_blockwise_fp8`, `reformat_x_scale` |
| `src/attention/entry.cc:443` | `attention_prefill_bf16`, `attention_with_kvcache_prefill_bf16` 等 |
| `src/rope/entry.cc` | rope 相关算子 |
| `src/activation/entry.cc` | 激活函数相关算子 |

### 6.6 `TorchLibraryInit` RAII 类

`torch/library.h:937-954`：

```cpp
class TorchLibraryInit final {
 private:
  using InitFn = void(Library&);
  Library lib_;

 public:
  TorchLibraryInit(
      Library::Kind kind,
      InitFn* fn,
      const char* ns,
      std::optional<c10::DispatchKey> k,
      const char* file,
      uint32_t line)
      : lib_(kind, ns, k, file, line) {
    fn(lib_);  // 构造后立即调用用户注册函数
  }
};
```

这是一个典型的 RAII 模式：`static const` 对象在进程启动时构造，构造函数中先创建 `Library`，再调用用户提供的回调函数完成注册。

## 7. `Tensor?` 问号语法：可选参数

### 7.1 含义

在 TorchScript 算子签名中，`Tensor?` 表示该参数是**可选的 (Optional)**，调用时可以传入 `None` 或不传。

`src/group_gemm/entry.cc:194-196`：

```cpp
m.def(
    "group_gemm_pertensor_fp8(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, Tensor "
    "y_scale, int num_seq_per_group_avg, Tensor? output, Tensor? tma_desc) -> (Tensor)");
```

其中 `Tensor? output` 和 `Tensor? tma_desc` 是可选参数。

### 7.2 C++ 端如何接收

Schema 中的 `Tensor?` 在 C++ 实现侧映射为 `std::optional<torch::Tensor>`。

`src/group_gemm/entry.cc:15-21`：

```cpp
torch::Tensor group_gemm_pertensor_fp8_entry(
    const torch::Tensor &x,
    const torch::Tensor &weight,
    const torch::Tensor &seqlens,
    const torch::Tensor &cu_seqlens,
    const torch::Tensor &y_scale,
    const int64_t num_seq_per_group_avg,
    std::optional<torch::Tensor> output,   // ← Tensor? 映射为此
    std::optional<torch::Tensor> tma_desc  // ← Tensor? 映射为此
)
```

### 7.3 类型系统实现

`Tensor?` 在 PyTorch 内部表示为 `OptionalType(TensorType)`：

`{conda_env}/lib/python3.12/site-packages/torch/include/ATen/core/jit_type.h:186-196`：

```cpp
// Optional[T] == Union[T, None] for all T
struct TORCH_API OptionalType : public UnionType {
  static OptionalTypePtr create(const TypePtr& contained);
  static const TypeKind Kind = TypeKind::OptionalType;
  // ...
};
```

schema 字符串中的 `?` 后缀被 `parseSchema()` 函数 (`torch/csrc/jit/frontend/function_schema_parser.h:16-22`) 解析为 `OptionalType`。

### 7.4 为什么设计成可选参数

`output` 和 `tma_desc` 设为可选参数是为了**内存复用**优化：

- **`output` (`Tensor?`)**: 如果调用者传入预先分配好的 output tensor，kernel 直接写入该 tensor，避免重复分配。如果传入 `None`，则 kernel 内部新分配。
- **`tma_desc` (`Tensor?`)**: 持有 per-group TMA 描述符的持久化 buffer。如果调用者缓存并重复传入同一个 `tma_desc` tensor，kernel 可以跳过 `update_tma` 阶段（因为 TMA 描述符未变），显著减少 kernel launch 开销。

在 `src/group_gemm/entry.cc:40-53` 可以看到对应的处理逻辑：

```cpp
// output 可选 → 有则复用，无则分配
if (output.has_value()) {
    y = output.value();
} else {
    y = torch::empty({m, n}, options.dtype(torch::kBFloat16));
}

// tma_desc 可选 → 有则复用（跳过 update_tma），无则分配
if (tma_desc.has_value()) {
    tmas = tma_desc.value();
    update_tma = false;  // 跳过 TMA 更新
} else {
    tmas = torch::empty({num_group * 2, 128}, options);
}
```

在 Python 调用侧 (`hpc/group_gemm.py:49-96`)，`output` 和 `tma_desc` 的默认值均为 `None`，对应 schema 中的 `Tensor?`：调用者不传这些参数时，它们等价于 `None`，在 C++ 端对应 `std::nullopt`。

## 8. `torch::empty` 的 device 推断、`options.dtype()` 定义与副作用分析

本节分析 `src/group_gemm/entry.cc:38-43` 这段代码：

```cpp
auto options = x.options();
torch::Tensor y;
if (output.has_value()) {
    y = output.value();
} else {
    y = torch::empty({m, n}, options.dtype(torch::kBFloat16));
}
```

### 8.1 y 的 device 如何确定

y 的 device **继承自输入 tensor `x` 的 device**，通过 `x.options()` 获得：

**第 1 步** — `x.options()` 返回携带 x 属性的 `TensorOptions`：

`{conda_env}/lib/python3.12/site-packages/torch/include/ATen/core/TensorBase.h:610-614`

```cpp
TensorOptions options() const {
    return TensorOptions().dtype(dtype())
                          .device(device())
                          .layout(layout());
}
```

`TensorBase::options()` 创建一个全新的 `TensorOptions`，并从当前 tensor 上取出 `dtype`、`device`、`layout` 三个属性设置上去。由于 `x` 是 CUDA tensor，`device()` 返回 GPU 设备，因此 `options` 携带的是 x 所在的 GPU device。

注意：`options()` 不保留 `requires_grad`、`pinned_memory`、`memory_format` 属性（参见 `TensorOptions.h:536-541` 处的注释警告）。

**第 2 步** — `options.dtype(torch::kBFloat16)` 返回**新副本**，只改了 dtype：

`{conda_env}/lib/python3.12/site-packages/torch/include/c10/core/TensorOptions.h:228-233`

```cpp
[[nodiscard]] TensorOptions dtype(
    std::optional<ScalarType> dtype) const noexcept {
  TensorOptions r = *this;
  r.set_dtype(dtype);
  return r;
}
```

关键点：
- 方法是 `const noexcept`（不修改 `*this`）
- 先拷贝 `*this`：`TensorOptions r = *this;`
- 在拷贝上调用 `r.set_dtype(dtype)` 修改 dtype
- **device 属性原封不动地保留了** — 来自第 1 步的 x.device()
- 返回值类型是 `TensorOptions`（值语义，新对象）
- `[[nodiscard]]` 表示返回值不应被丢弃

**第 3 步** — `torch::empty()` 用新 options 分配 tensor：

`{conda_env}/lib/python3.12/site-packages/torch/include/ATen/ops/empty.h:37-38`

```cpp
inline at::Tensor empty(at::IntArrayRef size, at::TensorOptions options={}, ...) {
    return at::_ops::empty_memory_format::call(
        c10::fromIntArrayRefSlow(size),
        c10::optTypeMetaToScalarType(options.dtype_opt()),
        options.layout_opt(),
        options.device_opt(),    // ← 这里提取 device，来自 x 的 device
        options.pinned_memory_opt(),
        ...);
}
```

`options.device_opt()` 返回的就是 x 所在 GPU 设备，因此分配的 `y` tensor 与 `x` 在同一 GPU 上。

**完整device流转链**：

```
x (CUDA tensor on device 0)
  → x.device()          → c10::Device(kCUDA, 0)
  → x.options()         → TensorOptions{ .device_ = c10::Device(kCUDA, 0), ... }
  → .dtype(kBFloat16)   → TensorOptions{ .device_ = c10::Device(kCUDA, 0),
                                          .dtype_ = ScalarType::BFloat16, ... }
  → torch::empty(...)   → 在 device 0 上分配 bfloat16 tensor y
```

### 8.2 `options` 本身是否被 `dtype()` 修改

**不会。** `options` 的原始值保持不变。

`options.dtype(torch::kBFloat16)` 调用的 `dtype(std::optional<ScalarType>)` 重载是 `const noexcept` 方法（`TensorOptions.h:228-233`），它：

1. 拷贝 `*this` 到局部变量 `r`
2. 修改 `r` 的 dtype
3. 返回 `r`（新对象）
4. `options` 自身没有任何变化

因此下面这段代码是安全的，`options` 在调用前后完全一致：

```cpp
auto options = x.options();                        // dtype=FP8, device=GPU0
y = torch::empty({m, n}, options.dtype(kBFloat16)); // 传入新 TensorOptions(dtype=BF16, device=GPU0)
                                                    // options 依然 = {dtype=FP8, device=GPU0}
tmas = torch::empty({num_group * 2, 128}, options); // 复用 options，分配 FP8 类型 tensor
```

这正是 `src/group_gemm/entry.cc:38-52` 中实际发生的模式 — `options`（FP8 dtype）在 line 43 被 `.dtype()` 临时改为 BF16 用于分配输出 tensor，随后 line 52 再次使用原始 `options`（FP8）分配 TMA descriptor tensor。

### 8.3 `TensorOptions` setter 方法的三类重载对比

`{conda_env}/lib/python3.12/site-packages/torch/include/c10/core/TensorOptions.h:220-241`

| 重载 | 签名 | 是否修改 `*this` | 返回值 | 对应调用方式 |
|------|------|:---:|------|------|
| TypeMeta setter | `[[nodiscard]] TensorOptions dtype(std::optional<TypeMeta>) const noexcept` | 否 | 副本（值） | 传入 `TypeMeta` 对象 |
| ScalarType setter | `[[nodiscard]] TensorOptions dtype(std::optional<ScalarType>) const noexcept` | 否 | 副本（值） | `options.dtype(torch::kBFloat16)` |
| Template setter | `TensorOptions& dtype<ScalarType>()` | **是** | 引用 | `options.dtype<float>()` |

在实际代码中，`options.dtype(torch::kBFloat16)` 匹配的是 ScalarType 重载（第 2 种），因此**不修改原对象**。

### 8.4 `torch::empty` 的完整调用路径

`{conda_env}/lib/python3.12/site-packages/torch/include/torch/csrc/autograd/generated/variable_factories.h:275-277`

```cpp
inline at::Tensor empty(at::IntArrayRef size, at::TensorOptions options = {}, ...) {
  at::AutoDispatchBelowADInplaceOrView guard;
  return autograd::make_variable(
      at::empty(size, at::TensorOptions(options).requires_grad(std::nullopt), memory_format),
      options.requires_grad());
}
```

`torch::empty()` 的流程：

1. **剥离 requires_grad**：`at::TensorOptions(options).requires_grad(std::nullopt)` — 拷贝 options 后清除 `requires_grad` 标志（ATen 层总返回不带 autograd 的 tensor）
2. **调用 ATen 的 `at::empty()`**：实际分配 tensor
3. **包装为 Variable**：`autograd::make_variable(..., options.requires_grad())` — 根据原始 options 的 `requires_grad` 设置来包装
4. 由于步骤 1 设置了 `requires_grad(std::nullopt)`，`at::empty()` 返回的是不追踪梯度的 tensor；步骤 3 再根据原始设置决定是否启用梯度追踪

## 9. y_scale 与 torch._scaled_mm 的 scale_a/scale_b 的数学关系

### 9.1 两种接口的差异

在测试文件 `tests/test_group_gemm_pertensor_like.py:39-41`，naive 实现使用 `torch._scaled_mm`：

```python
y_group = torch._scaled_mm(
    x_group, w_group.t(), scale_a=scale, scale_b=scale, bias=None, out_dtype=torch.bfloat16
)
```

而在 `hpc/group_gemm.py:94`，自定义算子只接收一个 `y_scale` 参数，传入 `src/group_gemm/entry.cc:15-21` 的 `y_scale`（per-group tensor），最终在 CUDA kernel 中应用。

### 9.2 torch._scaled_mm 的数学定义

`torch._scaled_mm(A, B, scale_a, scale_b, out_dtype=torch.bfloat16)` 的计算逻辑等价于：

```
C = (A * scale_a) @ B
```

由于 FP8 量化的惯例，`scale_a` 和 `scale_b` 是**逆量化因子**（inverse scale），即存储的 `A_fp8` 代表的真实浮点值是 `A_fp8 * scale_a`。因此：

```
C = (A_fp8 * scale_a) @ B_fp8  → 累积后再乘以 scale_b → C * scale_b
```

更准确的公式（对应 NVIDIA cuBLASLt 和 Hopper MMA 的语义）：

```
C = scale_a * scale_b * (A_fp8 @ B_fp8)
```

### 9.3 自定义 CUDA kernel 的数学定义

在 `src/group_gemm/kernels.cuh:347-374`，CUDA kernel 的计算为：

```cpp
float scale = yscale_ptr[igroup];   // line 347: 取当前 group 的 y_scale

// ... cute::gemm 计算 FP8 MMA，结果为 tCr = A_fp8 @ B_fp8

#pragma unroll
for (int i = 0; i < size(tCr); ++i) {
    tDr(i) = tCr(i) * scale + tDr(i);  // line 374: result = mma_result * y_scale
}
```

所以 kernel 的计算为：

```
C = y_scale * (A_fp8 @ B_fp8)
```

### 9.4 等价关系推导

令两种实现的结果相等：

```
scale_a * scale_b * (A_fp8 @ B_fp8) = y_scale * (A_fp8 @ B_fp8)
```

消去公共因子 `(A_fp8 @ B_fp8)`：

```
y_scale = scale_a * scale_b
```

### 9.5 测试代码中的验证

在 `tests/test_group_gemm_pertensor_like.py:57-58`：

```python
scale = torch.tensor(1.0, dtype=torch.float, device="cuda")
scale_hpc = torch.full((num_group,), 1.0, dtype=torch.float, device="cuda")
```

- `scale_a = 1.0`, `scale_b = 1.0` → `scale_a * scale_b = 1.0`
- `y_scale[i] = 1.0` for all groups

满足 `y_scale = scale_a * scale_b = 1.0`，所以两种实现的计算结果等价。

### 9.6 补充说明：per-tensor vs per-group

`torch._scaled_mm` 的 `scale_a` / `scale_b` 可以是标量（所有 token 共享），也可以是 1D tensor（per-token scale）。

`group_gemm_pertensor_fp8` 的 `y_scale` 是一个形状为 `[num_group]` 的 1D tensor（`tests/test_group_gemm_pertensor_like.py:58`），每个 group 可以有不同的 scale。尽管算子名称中包含 "pertensor"，但这里的 "tensor" 实际指的是对每个 group 内使用**单一 scale**（而非逐元素的 block-wise scale），与 `group_gemm_blockwise_fp8` 的 block-wise 量化形成对比。

## 10. tmas 形状 `{num_group * 2, 128}` 的设计原因

`src/group_gemm/entry.cc:52`：

```cpp
tmas = torch::empty({num_group * 2, 128}, options);
```

### 10.1 总体布局

`tmas` 是一个存储 TMA 描述符的 tensor，按 **每 group 2 个描述符** 布局：

| 索引 | 内容 | 用途 |
|------|------|------|
| `igroup * 2 + 0` | X 的 TMA descriptor | 当前 group 的输入激活子 tensor 的加载描述符 |
| `igroup * 2 + 1` | Y 的 TMA descriptor | 当前 group 的输出子 tensor 的存储描述符 |

### 10.2 逐层代码证据

**入口层** — `src/group_gemm/entry.cc:39`：

```cpp
auto *tma_xy = static_cast<cute::TmaDescriptor *>(tmas_ptr);
```

将 `tmas` 的裸指针强转为 `cute::TmaDescriptor *` 指针，后续按 group-offset 索引。

**TMA 更新 kernel** — `src/group_gemm/kernels.cuh:138`：

```cpp
tma_descriptor_cp_fence_release(tma_xy + igroup * 2 + i, smem_tma_desc[i]);
```

其中 `i ∈ {0, 1}`：`0` 代表 X descriptor，`1` 代表 Y descriptor。每个 group 写入两个相邻的 TMA descriptor。

**主 kernel 加载侧 (Load Warpgroup)** — `src/group_gemm/kernels.cuh:277`：

```cpp
auto *td_x = td_xy + igroup * 2;  // X descriptor for group igroup
```

使用 `igroup * 2 + 0` 处的 descriptor 进行 X 的 TMA 加载。

**主 kernel 写回侧 (Epilogue)** — `src/group_gemm/kernels.cuh:417`：

```cpp
auto *td_y = td_xy + igroup * 2 + 1;  // Y descriptor for group igroup
```

使用 `igroup * 2 + 1` 处的 descriptor 进行 Y 的 TMA 存储。

### 10.3 为什么需要 per-group TMA 描述符

每个 group 的 X 子 tensor 和 Y 子 tensor **起始地址和形状都不同**：

- X 子 tensor：`(cu_seqlens[igroup], k)` 处的 `seqlens[igroup] × k` 子矩阵
- Y 子 tensor：输出 `(cu_seqlens[igroup], n)` 处的 `seqlens[igroup] × n` 子矩阵

TMA 描述符包含硬件加速拷贝所需的目标地址和 tensor shape/stride 信息。因为每个 group 的这些参数不同，必须为每个 group 独立准备描述符。

在 `src/group_gemm/group_gemm_pertensor_fp8.cu:42-46`，先用 W 的全局 descriptor 和 Y 的全局 descriptor 作为**模板**：

```cpp
vec_t<cute::TmaDescriptor, 2> td_xy{
    *tma_x.get_tma_descriptor(),
    *tma_y.get_tma_descriptor(),
};
```

然后 `update_grouped_tma` kernel 对每个 group 拷贝模板，并调用 `update_tma_gtensor()` 替换描述符中的地址和形状字段（`src/utils/tma.cuh:37-57`），生成每 group 专属的 TMA descriptor。

### 10.4 为什么每个 descriptor 128 字节

每个 `cute::TmaDescriptor` 的大小是 **128 字节**。这是 NVIDIA Hopper (SM90) GPU 硬件定义的 TMA (Tensor Memory Access) 描述符固定大小。SM90 架构规格中，TMA descriptor 固定为 1024 bits = 128 bytes。

因此 tensor 的第二维度 `128` 正好容纳一个 `cute::TmaDescriptor`，加上第一维 `num_group * 2` 个条目，总字节数 = `num_group * 2 * 128`，即 `tmas.nbytes()`。

### 10.5 复用场景（跳过 TMA 更新）

当调用者传入预缓存的 `tma_desc` 时（`src/group_gemm/entry.cc:48-50`）：

```cpp
if (tma_desc.has_value()) {
    tmas = tma_desc.value();
    update_tma = false;  // 跳过 TMA 更新 kernel
}
```

`update_tma = false` 使得 `launch_group_gemm_fp8`（`src/group_gemm/group_gemm_pertensor_fp8.cu:42`）跳过 `update_grouped_tma` kernel launch，直接使用上一次写入的 per-group TMA descriptor。这在同一个 x/w 形状被反复调用时可以省去 TMA 更新开销。

## 11. 为什么 W 不需要 per-group TMA descriptor

### 11.1 三种 tensor 的数据布局差异

三种 tensor 在全局内存中的布局由 `src/group_gemm/group_gemm_pertensor_fp8.cu:27-32` 定义：

```cpp
auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)),
                     make_shape(m, k),                          // 2D: (total_seq, k)
                     make_stride(k, Int<1>{}));

auto W = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)),
                     make_shape(n, k, num_group),                // 3D: (n, k, num_group)
                     make_stride(k, Int<1>{}, n * k));

auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)),
                     make_shape(n, m),                           // 2D: (n, total_seq)
                     make_stride(Int<1>{}, n));
```

关键区别：

| Tensor | 维度 | Group 如何影响访问 | 是否需要 per-group descriptor |
|--------|------|--------------------|:--:|
| **X** | 2D `(m, k)`, 所有 groups 拼接在 M 维上 | 不同 group 的 sub-tensor 起始地址和 M 长度不同 | 是 |
| **W** | 3D `(n, k, num_group)`，group 是第 3 维 | 通过坐标索引，W 是一个整体连续的大 tensor | **否** |
| **Y** | 2D `(n, m)`，所有 groups 拼接在 M 维上 | 不同 group 的 sub-tensor 起始地址和 M 长度不同 | 是 |

### 11.2 W 如何通过单一 TMA 描述符访问不同 group

W 的 TMA 描述符 `tma_w` 在 `launch_group_gemm_fp8` 中创建（`src/group_gemm/group_gemm_pertensor_fp8.cu:34-37`），作为**模板参数**和**kernel 参数**传递，不存储在 `td_xy` 表中：

```cpp
auto [tma_x, tma_w, tma_y] = config.get_tma(X, W, Y);

// W 的 TMA descriptor 作为 __grid_constant__ 参数传递
kernel<<<...>>>(tma_w, tma_xy, ...);   // line 76-78: tma_w 是第一个 kernel 参数
```

在 kernel 内部（`src/group_gemm/kernels.cuh:145`）：

```cpp
__global__ void group_gemm_pertensor_fp8_kernel(
    const __grid_constant__ TmaB tma_b,   // ← W 的 TMA descriptor，__grid_constant__
    cute::TmaDescriptor *td_xy,           // ← X 和 Y 的 per-group descriptor 表
    ...
)
```

`__grid_constant__` 是 CUDA SM90 的特性：将 kernel 参数标记为对整个 grid 所有 block 都相同的常量，由硬件 load 一次后缓存在 constant memory 中，所有 block 共享。

TMA 加载 W 时（`src/group_gemm/kernels.cuh:287`）：

```cpp
cute::copy(tma_b.with(readable[ismem_write]),
           tBg(_, itile_n, itile_k, igroup),   // ← igroup 是坐标索引
           tBs(_, 0, 0, ismem_write));
```

`tBg` 的 partitioned source tensor 是 `btma_b.partition_S(gB)`，其中 `gB = tma_b.get_tma_tensor(make_shape(n, k, num_group))`（`kernels.cuh:191`）。这产生一个 4D view：

```
tBg = (TMA, TMA_N, TMA_K, num_group)     // kernels.cuh:202
```

**`igroup` 是 tBg 的第 4 个坐标索引**，不是 descriptor 索引。TMA 硬件根据（固定）descriptor 中的基地址 + 坐标 `(*, *, *, igroup)` 自动计算目标地址。由于 W 是 3D contiguous tensor（stride: `(k, 1, n*k)`），`igroup` 维度的步长为 `n*k`，硬件自动加上 `igroup * n * k * sizeof(element)` 的偏移。

### 11.3 为什么 X 和 Y 不能用同样的坐标方式

X 和 Y 的 group 划分方式与 W 根本不同：

**W 的 group 划分**（规则，3D tensor）：
```
W 是一个规则的 3D tensor，所有 groups 的矩阵有相同的 shape (n, k)
W[igroup] 位于内存 offset = igroup * n * k 处
```

**X 的 group 划分**（不规则，2D tensor 切分）：
```
X 是所有 groups 的激活拼接在一起的 2D tensor
X[igroup] 起始于 cu_seqlens[igroup]，长度为 seqlens[igroup]
每个 group 的 M 维长度不同（各组 seqlen 不同）
```

TMA descriptor **必须在创建时指定完整的边界形状（bounding box shape）**。由于 X 的 group 边界和非均匀大小无法表示为坐标索引，必须为每个 group 创建独立 descriptor，用 `update_tma_gtensor`（`src/utils/tma.cuh:36-57`）替换 descriptor 中的地址指针和边界形状。

Y 同理：output tensor 按 `cu_seqlens` 拼接，各 group 的子区域非均匀，需要独立 descriptor。

### 11.4 总结对比

```
                    ┌─────────┬──────────────┬─────────────────────┐
                    │    W    │      X       │          Y          │
├───────────────────┼─────────│──────────────│─────────────────────┤
│ 全局内存布局      │ 3D tensor│ 2D, 按seqlen│ 2D, 按seqlen 拼接    │
│                   │ (n,k,g) │    拼接       │                      │
├───────────────────┼─────────│──────────────│─────────────────────┤
│ group 如何访问    │ 坐标    │ per-group    │ per-group            │
│                   │ igroup  │ descriptor   │ descriptor           │
├───────────────────┼─────────│──────────────│─────────────────────┤
│ descriptor 数量   │ 1个     │ num_group个  │  num_group个          │
├───────────────────┼─────────│──────────────│─────────────────────┤
│ 传递方式          │__grid_  │ td_xy 表中   │  td_xy 表中           │
│                   │constant_│              │                      │
└───────────────────┴─────────│──────────────│─────────────────────┘
```
