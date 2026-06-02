# torch::kBFloat16 — 类型、定义位置与命名空间解析链

## 1. 类型

`torch::kBFloat16` 的类型是 `c10::ScalarType` 的 **constexpr 常量**，但 `ScalarType` 本身是一个 C++ **enum class**。

这不是 C 风格的裸 enum，而是 `enum class ScalarType : int8_t`，底层类型为 `int8_t`。

## 2. 定义位置

### 2.1 enum class ScalarType 定义

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

### 2.2 constexpr 常量 kBFloat16 定义

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

### 2.3 辅助包装头文件

以下两个头文件是薄包装，仅做 include 转发：

- `{conda_env}/lib/python3.12/site-packages/torch/include/ATen/core/ScalarType.h:2` — `#include <c10/core/ScalarType.h>`
- `{conda_env}/lib/python3.12/site-packages/torch/include/ATen/ScalarType.h:5` — `#include <c10/core/ScalarType.h>`

它们仅为向后兼容而存在。

## 3. 命名空间解析链：c10 → at → torch

`torch::kBFloat16` 并非在 `torch` 命名空间中直接定义，而是通过两层 `using namespace` 逐级传播：

### 第 1 层：c10 → at

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

### 第 2 层：at → torch

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

### 解析链图示

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

## 4. 总结

| 问题 | 答案 |
|------|------|
| 是 enum class 吗？ | 是 `constexpr ScalarType` 常量，`ScalarType` 本身是 `enum class ScalarType : int8_t` |
| 真正定义在哪里？ | `c10/core/ScalarType.h:39`（常量），`torch/headeronly/core/ScalarType.h:258`（enum class） |
| 如何在 `torch::` 中可用？ | 通过 `at` → `c10` 两层 `using namespace` 链式传播 |
| 数值是多少？ | `static_cast<int8_t>(15)` |
