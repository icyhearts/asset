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

`src/group_gemm/config.h:63-107` `struct GroupGEMMFp8Config`

| 配置项 | 说明 |
|--------|------|
| `SLayoutXAtom` | X 的 shared memory swizzle 布局 (kSwizzleX=128, K-major) |
| `SLayoutWAtom` | W 的 shared memory swizzle 布局 (kSwizzleW=128, K-major) |
| `SLayoutYAtom` | Y 的 shared memory swizzle 布局 (kSwizzleY=64, MN-major) |
| `TiledMma` | `SM90_64x<kTileM>x32_F32E4M3E4M3_SS_TN` + warpgroup layout |
| `get_tma()` | 创建 SM90_TMA_LOAD / SM90_TMA_STORE copy atoms |
| `get_shm_size()` | 计算动态共享内存总大小 |
