# TileLang `@tilelang.jit` + `compile()` 参数机制解析

以 `examples/gemm/example_gemm-like.py:5-27` 的 `matmul` 函数为例:

```python
@tilelang.jit
def matmul(A, B, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    ...
```

调用端 (`examples/gemm/example_gemm-like.py:31,38`):

```python
kernel = matmul.compile(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)
c = kernel(a, b)
```

---

## 1. M, N, K 不是函数参数，为什么可以传给 compile()?

### 1.1 `T.const("M, N, K")` — 创建符号变量

`tilelang/language/eager/builder.py:922-962`, `const()`:

```python
if builder.eager_jit == "phase1":
    if "," in name:
        names = re.split(r"\s*,\s*", name)     # "M, N, K" → ["M","N","K"]
        return tuple(builder.constexpr(n, dtype) for n in names)
```

`constexpr()` (`tilelang/language/eager/builder.py:754-758`):

```python
def constexpr(self, name, dtype="int32"):
    var = tirx.Var(name, dtype)       # 创建 TVM 符号变量
    self.constexpr_var.add(var)        # 记录到 constexpr 集合
    return var
```

`"M, N, K"` 被逗号分割 → 创建三个 `tirx.Var` 符号变量 `M`, `N`, `K`，记录到 `constexpr_var` 集合。

这些符号变量是**占位符**（symbolic variable），不具有具体值，只用于构造 IR 表达式。

### 1.2 `TirTemplate` — 建立符号变量到 buffer 维度的 matcher

`tilelang/language/eager/builder.py:1043-1067`, `TirTemplate.create()`:

```python
matcher = {}
for k, v in prim_func.buffer_map.items():
    for i, s in enumerate(v.shape):
        if s in constexpr and s not in matcher:
            matcher[s] = (k.name, "shape", i, s.name)
```

遍历所有 buffer 的 shape/stride，当某一维是之前 `T.const` 创建的符号变量时，建立映射关系:
```
符号变量 → (buffer_name, "shape"|"stride", 维度索引, 变量名)
```

例如 `C = T.empty((M, N), dtype)` 中:
- `matcher[M] = ("C", "shape", 0, "M")`
- `matcher[N] = ("C", "shape", 1, "N")`

### 1.3 `_bind_fast` — 接收 "额外" 关键字参数

这是关键机制。`tilelang/language/eager/builder.py:1274-1279`, `_bind_fast()`:

```python
for name, value in kwargs.items():
    index = self.param_index.get(name)  # 在函数正式参数中查找
    if index is None:
        # TileLang allows extra compile-time kwargs for explicit T.const bindings.
        extra_kwargs[name] = value      # ← M,N,K 不在签名中, 归入 extra_kwargs
        continue
```

`matmul` 的 Python 函数签名是 `(A, B, block_M, block_N, block_K, dtype, accum_dtype)`。`param_index` 只含这 7 个名字。

`M`, `N`, `K` 不在 `param_index` 中 → 落入 `extra_kwargs` → 随后被合并到 `compile_kwargs` (`tilelang/language/eager/builder.py:1298-1300`):

```python
if extra_kwargs:
    p1_values.append(self._extra_key(extra_kwargs))
    compile_kwargs.update(extra_kwargs)
```

### 1.4 Phase-2 替换 — 符号变量 → 具体值

`tilelang/language/eager/builder.py:1074-1087`, `_parse_phase2_key()`:

```python
def _parse_phase2_key(self, **kwargs):
    for k, ty, i, name in self.matcher.values():
        if name in kwargs:
            result.append(kwargs.get(name))   # 从 kwargs 取出 M=1024
```

`compile_kwargs = {M: 1024, N: 1024, K: 1024, ...}`。`_parse_phase2_key` 按 matcher 记录的顺序，从 kwargs 中提取 M,N,K 的具体值 → 用于 phase-2 的 TIR 实例化。

**总结**: `T.const("M, N, K")` 创建**无具体值**的符号变量 → `TirTemplate.create()` 从 buffer shape 建立这些变量到维度的映射（matcher）→ compile() 时用户传入的具体值 `M=1024` 通过 `_bind_fast` 的 `extra_kwargs` 机制接收 → phase-2 用这些值替换符号变量，实例化 TIR。

---

## 2. A, B 是函数参数，为什么没传给 compile()?

### 2.1 AST 阶段: `_parse_arg_annot` 识别 tensor 参数

`tilelang/language/eager/ast.py:510-541`, `_parse_arg_annot()`:

```python
def _parse_arg_annot(self, stmt, arg_names):
    if not isinstance(stmt, ast.AnnAssign):  return   # 必须是 A: Type 语法
    if not isinstance(stmt.target, ast.Name): return   # 必须是简单名称
    if stmt.value is not None:               return   # 不能有 = 号: "A = ..." 不是 annot
    name = stmt.target.id
    if name not in arg_names:                return   # 必须是函数参数名

    # 检测 annotation 是否是 T.Tensor(...)
    if inner is not None and inner.attr in ["Tensor", "StridedTensor", "ptr"]:
        eval_res = self._try_eval(inner)
        if isinstance(eval_res, (TensorProxy, StridedTensorProxy)) or eval_res is ptr:
            self.extra_type_hints[name] = ptr   # 记录为 tensor 参数
```

当 AST 解析遇到 `A: T.Tensor((M, K), dtype)`（无 `=` 号的 `AnnAssign`，目标名 `A` 是函数参数），将 `extra_type_hints["A"] = ptr`。

### 2.2 `prim_func` 阶段: 分离 tensor_args 和 compile_kwargs

`tilelang/language/eager/builder.py:1527`, `prim_func()`:

```python
tensor_args = {k: v for k, v in annot.items() if isinstance(v, (Buffer, Var))}
```

标注了 `ptr` 的参数被识别为 tensor 参数类型 → 归入 `tensor_args = {"A": ...}`。

### 2.3 arg-binding 阶段: tensor 参数不进入 compile_kwargs

`tilelang/language/eager/builder.py:1291-1297`, `_bind_fast()`:

```python
for name, value in zip(self.param_names, values):
    if name in self.tensor_arg_names:
        if value is not _MISSING_ARG:
            tensor_args[name] = value    # ← 归入 tensor_args
    else:
        p1_values.append(value)          # ← 归入 compile_kwargs
        compile_kwargs[name] = value
```

`tensor_arg_names = {"A", "B"}`（来自 `tensor_args.keys()`）。A,B 被归入 `tensor_args`，不进入 `compile_kwargs`。

**compile() 只接收 compile_kwargs** (`block_M, block_N, block_K, dtype, accum_dtype, M, N, K`)。tensor 的实际数据（`torch.Tensor`）只在运行时 `kernel(a, b)` 阶段传入。

---

## 3. 为什么 `A: T.Tensor((M, K), dtype)` 不引起变量重复定义?

`A` 已是函数参数（`def matmul(A, B, ...)`），函数体内又出现 `A: T.Tensor((M, K), dtype)`。

关键在 `_parse_arg_annot` (`tilelang/language/eager/ast.py:510-541`) 的五重守卫:

| 检查 | 代码 | `A: T.Tensor(...)` 结果 |
|------|------|------------------------|
| 必须是 AnnAssign | `isinstance(stmt, ast.AnnAssign)` | True — `A: expr` 语法 |
| 必须是简单名称 | `isinstance(stmt.target, ast.Name)` | True — `A` 是 Name |
| **不能有 = 号** | `stmt.value is not None` | **False** — 没有 `=expr`，跳过！ |
| 必须是函数参数 | `name in arg_names` | True — `A` 在参数列表中 |
| annotation 类型检查 | `inner.attr in ["Tensor",...]` | True — `T.Tensor` |

关键点: `stmt.value is not None` — 只有当 `A = T.Tensor(...)`（即 `stmt.value` 有值）时才会执行赋值语义。`A: T.Tensor(...)` 是**裸类型标注**（bare annotation），`.value` 为 `None`，不产生赋值。

因此 `_parse_arg_annot` 的工作是:

```
将已有参数 A 的类型意图记录到 extra_type_hints["A"] = ptr
然后 return（不执行任何赋值 AST 生成）
```

这是 **Python 3.6+ 变量标注的语法复用**（PEP 526），语义是**声明**而非**赋值**。

---

## 4. 三个函数签名关系

### 4.1 DSL 定义签名

`examples/gemm/example_gemm-like.py:6`, `matmul()`:

```python
def matmul(A, B, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
```

| 类别 | 参数 | 说明 |
|------|------|------|
| **Tensor 参数** | `A`, `B` | 运行时传入的 tensor 数据指针 |
| **Compile-time 参数** | `block_M`, `block_N`, `block_K` | 普通 Python 参数，有默认值则可选 |
| **Compile-time 默认值** | `dtype`, `accum_dtype` | 有默认值的 compile-time 参数 |

### 4.2 compile() 签名

`examples/gemm/example_gemm-like.py:31`:

```python
kernel = matmul.compile(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)
```

实际传入 compile() 的参数通过 `_bind_fast` 分离后形成 `compile_kwargs`:

| 来源 | 参数 | 机制 |
|------|------|------|
| 函数签名参数 | `block_M=128`, `block_N=128`, `block_K=32` | `param_index` 中匹配 |
| 默认值 | `dtype=T.float16`, `accum_dtype=T.float32` | `param_index` + 默认值自动填充 |
| `T.const` 符号变量 | `M=1024`, `N=1024`, `K=1024` | `extra_kwargs` 机制 |
| **不参与** | **A, B** | 归入 `tensor_args`，compile 阶段不需要 |

### 4.3 kernel 运行时签名

`examples/gemm/example_gemm-like.py:38`:

```python
c = kernel(a, b)
```

`JITKernel.__call__()` (`tilelang/jit/kernel.py:186-202`) 将调用转发到 `self.torch_function`（由 adapter 创建，底层为编译好的 CUDA kernel）。

**运行时签名** = **仅 tensor 参数**: 接收 `torch.Tensor`，返回 `torch.Tensor`。M, N, K 等形状信息已固化在编译好的 kernel 代码中。

### 4.4 三签名对照

```
┌─────────────────────────────────────────────────────────────────────┐
│ DSL 定义签名 (matmul)                                               │
│   matmul(A, B, block_M, block_N, block_K, dtype, accum_dtype)       │
│                                                                     │
│   _parse_arg_annot → tensor_args = {A, B}                           │
│   param_index      → other_params = {block_M, block_N, ...}         │
│   T.const(...)     → constexpr = {M, N, K}                          │
│                                                                     │
│   ┌───────── bind ─────────┐                                        │
│   │ tensor_args:   {A,B}   │  ── 不进 compile_kwargs                │
│   │ compile_kwargs: other_params + constexpr                         │
│   └────────────────────────┘                                        │
├─────────────────────────────────────────────────────────────────────┤
│ compile() 签名                                                      │
│   matmul.compile(M=1024, N=1024, K=1024,                           │
│                  block_M=128, block_N=128, block_K=32)               │
│                                                                     │
│   TirTemplate: 符号变量 M,N,K 替换为具体值 1024→ TIR 实例化          │
│   compile(): 将 TIR PrimFunc → CUDA kernel                          │
│   → kernel (JITKernel 对象, 已编译)                                  │
├─────────────────────────────────────────────────────────────────────┤
│ 运行时签名 (kernel)                                                  │
│   c = kernel(a, b)                                                  │
│                                                                     │
│   a, b: torch.Tensor                                                │
│   JITKernel.__call__ → adapter.func → CUDA kernel launch            │
│   → torch.Tensor 输出                                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.5 完整调用链

`tilelang/jit/__init__.py:495-540`, `JITImpl.__call__()`:

```python
def __call__(self, *args, **kwargs):
    key, kernel_args = self.func.parse_args(*args, **kwargs)  # Step 1: 参数解析&分类
    kernel = self._kernel_cache.get(key, None)                 # Step 2: 缓存命中检查
    if kernel is None:
        kernel = self.compile(*args, **kwargs)                 # Step 3: 编译 (仅用 compile_kwargs)
        self._kernel_cache[key] = kernel
    if self.mode == "eager":
        return kernel(*kernel_args.values())                  # Step 4: 运行时调用 (仅 tensor_args)
```

`parse_args()` (`tilelang/language/eager/builder.py:1437-1450`):

```python
def parse_args(self, *args, **kwargs):
    bound = self._argument_binder.bind(args, kwargs)
    tir_temp = self._build_tir_template(**bound.compile_kwargs)   # 仅 compile_kwargs
    p2_key = tir_temp._parse_phase2_key(**bound.tensor_args, **bound.compile_kwargs)
    return (bound.p1_key, p2_key), bound.tensor_args               # kernel_args = tensor_args
```

`kernel_args = {"A": tensor_a_ptr, "B": tensor_b_ptr}` — 仅含 tensor，运行时会传给 CUDA kernel。
