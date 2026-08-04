## 1. 提交 `82a8d2031ffc3323a31d76dd1862f6adef345412` 主要实现了什么

### 总体结论

这是一个小型兼容性修复提交，而不是新增量化算法或模型功能。它主要修复了 Flexpoint
FP8/INT8 downcast 的 Triton kernel 中 `eps` 参数编译期类型不一致的问题，同时提高了 SGLang
可选依赖的最低版本，并删除了一个仓库根目录下的临时调试脚本。

提交信息如下：

```text
commit:  82a8d2031ffc3323a31d76dd1862f6adef345412
parent:  d19622e34e8cd86d7d6e2194f9539a7906049856
subject: fix (model-opt/simo!191)
规模:    3 files changed, 4 insertions(+), 7 deletions(-)
```

### 1. 修复 Flexpoint Triton kernel 的 `eps` 常量传递

核心修改位于 `simo/ops/kernels/downcast/_downcast_to_flexpoint.py`。提交给以下三个 Triton
kernel 的 `eps` 参数增加了 `tl.constexpr` 标注：

```python
eps: tl.constexpr
```

涉及的执行路径是：

| Triton kernel | 对应量化路径 |
| --- | --- |
| `_per_block_quant_fp8_or_int8_kernel` | FP8/INT8 per-block downcast |
| `_per_token_group_quant_fp8_or_int8_kernel` | 普通行主序 per-group downcast |
| `_per_token_group_quant_fp8_or_int8_colmajor_kernel` | 列主序 scale 的 per-group downcast |

这些 kernel 会把 `eps` 继续传给 `quantize_fp8_or_int8_group()`，后者再传给
`calculate_flexpoint_scale()`；这两个辅助函数原本就要求 `eps` 是 `tl.constexpr`。提交前，外层
kernel 却把 `eps` 声明为普通运行时参数，导致调用链两端的 Triton 类型语义不一致。

调用端传入的值本身就是固定 Python 常量：

- per-block 路径使用 `1e-4`；
- per-group 路径使用 `1e-10`。

将其标记为 `tl.constexpr` 后，Triton 会在编译 kernel 时完成特化，并把该常数正确传入下层
constexpr 辅助函数，从而避免相关 kernel 的编译/类型处理问题。

`eps` 的用途是给 `absmax` 设置下限，避免全零或极小输入产生零 scale，随后发生除零或
`log2(0)`。这个提交没有改变 `eps` 的数值，也没有改变 scale 公式、量化范围、舍入规则、tensor
布局或 Python 公共 API，因此它属于 kernel 编译兼容性修复，而不是数值算法调整。

### 2. 提高 SGLang 最低版本

`pyproject.toml` 中 `sglang` 可选依赖由：

```text
sglang>=0.5.6
```

调整为：

```text
sglang>=0.5.15
```

这意味着安装 SIMO 的 `sglang` extra 时，不再允许解析到 0.5.6 至 0.5.14。该提交本身没有修改
`simo/extensions/sglang_simo` 的适配代码，因此能从 diff 直接确认的是依赖基线升级，不能把它
解释成新增了某项 SGLang 推理功能。

### 3. 删除临时调试文件

提交删除了仓库根目录的 `test_common.py`。该文件只有：

```python
from simo.quantization.dtypes import as_dtype

print(as_dtype("mxint8").value)
```

它没有 pytest 测试函数，只是在导入/收集时打印一个 dtype 值，属于临时验证脚本。删除它可以避免
被 pytest 当作测试模块收集时产生无意义的导入和输出。

### 最终概括

```text
主要修复：
  把三个 Flexpoint downcast Triton kernel 的 eps 参数设为编译期常量，统一与下层
  constexpr 量化辅助函数的类型要求，覆盖 per-block 和两种 per-group GPU 路径。

配套变化：
  将 SGLang 最低版本从 0.5.6 提升到 0.5.15；删除临时 test_common.py。

没有改变：
  FP8/INT8 的量化公式、eps 数值、量化范围、输出格式和对外 API。
```

## 2. 当前项目如何编译 SIMO ONNX custom-op 的 Debug 版本

### 2.1 先区分两个不同的 C++ 构建目标

当前仓库有两条容易混淆的 C++ 构建路径：

```text
simo/csrc/**/*.cpp, *.cu
  -> setup.py:get_extensions()
  -> PyTorch CppExtension/CUDAExtension
  -> simo/_C*.so

simo/onnx/ort_plugin/custom_op_library.cc
simo/onnx/ort_plugin/simo_qdq_ops.cc
simo/onnx/ort_plugin/simo_qdq_cpu_ops.cc
simo/onnx/ort_plugin/triton_loader.cc
generated embedded QDQ cubin source
  -> simo/onnx/ort_plugin/build_runtime.py
  -> libSimoOnnxCustomOps_sm90.so
```

提问中的 `simo/onnx/ort_plugin/simo_qdq_ops.cc` 属于第二条路径。它不是 `simo._C` 的 source，也没有独立的 CMake target；`build_runtime.py:70-75` 把它和其他 custom-op source 一起传给一次 `c++ -shared` 命令，最终生成 `libSimoOnnxCustomOps_sm90.so`。

尽管 `simo_qdq_ops.cc` 包含 `cuda.h` 并通过 CUDA Driver API 启动 kernel，它本身仍是 host C++，由 `CXX`（本机是 `/usr/bin/c++`）编译，不由 `nvcc` 编译。真正的 device code 由 `build_qdq_cubins.py` 调用 Triton 生成并嵌入 library，所以给 nvcc 增加 `-G` 不会让 `simo_qdq_ops.cc` 获得 host 调试信息。

这也意味着仓库 `CLAUDE.md:31-32` 中的：

```bash
DEBUG=1 pip install -e ".[dev]" --no-build-isolation
```

只能直接保证 `setup.py:get_extensions()` 为 `simo._C` 设置 `-O0 -g`。`setup.py:56-71` 的 Debug flags 确实会影响 `simo/csrc` 的 C++/CUDA extension，但 `build_runtime.py:83-98` 当前把 custom-op host C++ 的命令硬编码为：

```text
-std=c++17 -O3 -fPIC -shared ...
```

它没有读取 `DEBUG`。所以单独执行 `DEBUG=1 pip install -e ...`，不能可靠地把 `simo_qdq_ops.cc` 编译成 `-O0 -g` 版本；这份代码当前存在两个 Debug 开关不完全贯通的问题。

### 2.2 前置环境

在当前机器使用用户给出的环境：

```bash
export REPO=/share/users/like/package/simo_conda_sglang
export PYTHON=/share_data/users/like/miniconda3/envs/simo_sglang/bin/python
export CUDA_HOME=/share_data/users/like/opt/cuda-13.0

cd "$REPO"
"$PYTHON" -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.__version__, torch.version.cuda, CUDA_HOME, torch.cuda.is_available())'
```

当前实际检查结果是 PyTorch `2.11.0+cu130`、`torch.version.cuda=13.0`、`CUDA_HOME=/share_data/users/like/opt/cuda-13.0`，并且 GPU 可见。`build_runtime.py` 的 `_cuda_home()` 会检查 `$CUDA_HOME/include/cuda.h`，所以需要确保这个变量指向 CUDA 13.0，而不是本机 PATH 中可能存在的 `/usr/local/cuda-12.8`。

此外，插件目标是 `sm90`：

- `setup.py` 为 CUDA extension 增加 `-gencode=arch=compute_90,code=sm_90`；
- `build_qdq_cubins.py` 使用 Triton `GPUTarget("cuda", 90, ...)`；
- `build_runtime.py` 输出固定命名的 `libSimoOnnxCustomOps_sm90.so`。

用户给出的 `/softhome/like/package/onnxruntime` 源码目录不参与这个插件的编译。`build_runtime.py:77-81` 使用的是 SIMO 仓库内 vendored public headers：

```text
simo/onnx/ort_plugin/include/onnxruntime
simo/onnx/ort_plugin/include/onnxruntime/core/session
```

当前 production `.so` 的 dynamic dependency 也只有 `libcuda.so.1`、C/C++ runtime 等，没有直接链接 `libonnxruntime.so`。ORT 在 `RegisterCustomOps()` 时把 `OrtApi` function table 传入插件。因此只调试 `simo_qdq_ops.cc` 不需要重新编译 `/softhome/like/package/onnxruntime`；但若断点需要继续单步进入 ORT 内部实现，则还需要另行构建并让 Python 加载带 Debug symbols 的 ONNX Runtime。

### 2.3 推荐方案：只重编译 Debug custom-op library

如果目标只是调试 `simo_qdq_ops.cc`，不需要重新编译 `simo._C`。可以给 `build_runtime.py` 提供一个 C++ wrapper：它把该脚本硬编码的 `-O3` 替换成 `-O0`，再补上 `-g3` 和 frame pointer。wrapper 自身必须是一个可执行文件，因为 `build_runtime.py` 调用的是：

```python
shutil.which(os.environ.get("CXX", "c++"))
```

创建 wrapper：

```bash
cat > /tmp/simo-cxx-debug <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

args=()
for arg in "$@"; do
  if [[ "$arg" == "-O3" ]]; then
    args+=("-O0")
  else
    args+=("$arg")
  fi
done

exec /usr/bin/c++ "${args[@]}" -O0 -g3 -fno-omit-frame-pointer
EOF
chmod +x /tmp/simo-cxx-debug
```

然后只调用现有的 runtime builder，把 Debug library 放到一个独立路径，避免覆盖当前 editable 安装里的生产版 `.so`：

```bash
export CXX=/tmp/simo-cxx-debug
export DEBUG=1

mkdir -p "$REPO/temp/simo-debug"
"$PYTHON" -c \
  'from simo.onnx.ort_plugin.build_runtime import build_sm90_runtime; \
   build_sm90_runtime("/share/users/like/package/simo_conda_sglang/temp/simo-debug/libSimoOnnxCustomOps_sm90_debug.so")'
```

这个命令会：

1. 调用 Triton 生成当前配置对应的 embedded QDQ cubin source；
2. 用 `/tmp/simo-cxx-debug` 编译 `custom_op_library.cc`、`simo_qdq_ops.cc`、`simo_qdq_cpu_ops.cc`、`triton_loader.cc` 和生成的 source；
3. 链接 `libcuda`，生成 `temp/simo-debug/libSimoOnnxCustomOps_sm90_debug.so`。

这里 `DEBUG=1` 对 `build_runtime.py` 本身不是必要条件，真正改变 `simo_qdq_ops.cc` 编译参数的是 `CXX` wrapper。保留 `DEBUG=1` 是为了在同一 shell 中运行其他 SIMO Debug 构建时语义一致。

### 2.4 让运行中的 ORT 确实加载 Debug `.so`

`simo/onnx/runtime.py:7-26` 默认查找：

```text
simo/onnx/ort_plugin/libSimoOnnxCustomOps_sm90.so
```

但它支持 `SIMO_ONNX_CUSTOM_OPS_LIBRARY` 覆盖。因此构建完成后不要只看文件存在，要显式指定：

```bash
export SIMO_ONNX_CUSTOM_OPS_LIBRARY="$REPO/temp/simo-debug/libSimoOnnxCustomOps_sm90_debug.so"

"$PYTHON" -c \
  'from simo.onnx.runtime import get_custom_ops_library_path; \
   print(get_custom_ops_library_path())'
```

输出必须是 `temp/simo-debug/libSimoOnnxCustomOps_sm90_debug.so`。已有测试中的 `_simo_custom_ops_library()` 会优先读取该环境变量，因此可以直接运行目标 CUDA runtime 测试，例如：

```bash
SIMO_ONNX_CUSTOM_OPS_LIBRARY="$REPO/temp/simo-debug/libSimoOnnxCustomOps_sm90_debug.so" \
  "$PYTHON" -m pytest -s -vv \
  simo/onnx/tests/test_dynamic_qdq_runtime_debug.py \
  -k 'tiny_tensor or dynamic_sequence'
```

若只想先验证库能被加载，也可以运行：

```bash
SIMO_ONNX_CUSTOM_OPS_LIBRARY="$REPO/temp/simo-debug/libSimoOnnxCustomOps_sm90_debug.so" \
  "$PYTHON" -c 'import onnxruntime as ort; from simo.onnx.runtime import register_custom_ops; o=ort.SessionOptions(); register_custom_ops(o); print("custom-op options prepared")'
```

调试 `simo_qdq_ops.cc` 时不要设置：

```bash
SIMO_ONNX_QDQ_PROVIDER=CPU
```

因为 `custom_op_library.cc:16-21` 在该变量为 `CPU` 时注册 `RegisterCpuQdqOps(domain)`，会绕开 CUDA 版 `RegisterQdqOps(domain)`；此时即使加载了 Debug library，也不会进入目标 CUDA source 的执行路径。要调试 `simo_qdq_ops.cc`，应保持该变量未设置，或者明确清除：

```bash
unset SIMO_ONNX_QDQ_PROVIDER
```

### 2.5 用 GDB 在 `simo_qdq_ops.cc` 断点

先验证 Debug 信息确实存在：

```bash
DEBUG_SO="$REPO/temp/simo-debug/libSimoOnnxCustomOps_sm90_debug.so"

file "$DEBUG_SO"
readelf -S "$DEBUG_SO" | grep -E '\.debug_(info|line|abbrev|str)' || true
readelf -Ws "$DEBUG_SO" | c++filt | grep -E 'RegisterQdqOps|QuantizeCustomOp|DequantizeCustomOp'
```

`file` 应显示 shared object；`readelf -S` 应能看到 `.debug_info`、`.debug_line` 等 section。若只有 `.symtab` 而没有 `.debug_*`，说明编译器 wrapper 没有真正生效，或者查看的仍是原来的生产 `.so`。

由于 custom-op library 是 ORT 运行时通过 `register_custom_ops_library()` 动态加载的，GDB 需要允许 pending breakpoint：

```bash
export SIMO_ONNX_CUSTOM_OPS_LIBRARY="$REPO/temp/simo-debug/libSimoOnnxCustomOps_sm90_debug.so"
export SIMO_ONNX_SYNC_AFTER_LAUNCH=1

gdb --args "$PYTHON" -m pytest -s -vv \
  simo/onnx/tests/test_dynamic_qdq_runtime_debug.py \
  -k 'tiny_tensor'
```

在 GDB 中：

```gdb
set breakpoint pending on
break simo/onnx/ort_plugin/simo_qdq_ops.cc:458
break simo/onnx/ort_plugin/simo_qdq_ops.cc:518
break simo/onnx/ort_plugin/simo_qdq_ops.cc:573
run
```

当前源码中比较有用的位置是：

- `:458`：`QuantizeCustomOp::Compute()` 入口；
- `:518`：`DequantizeCustomOp::Compute()` 入口；
- `:573`：`RegisterQdqOps()` 注册 CUDA custom ops；
- `:263`：`LaunchQdqKernel()`，负责解析 CUDA stream、准备参数并调用 `cuLaunchKernel`。

如果文件行断点无法解析，可以在 GDB 里先运行到 library 加载后，再执行：

```gdb
info sharedlibrary
sharedlibrary libSimoOnnxCustomOps_sm90_debug.so
break simo/onnx/ort_plugin/simo_qdq_ops.cc:458
continue
```

`SIMO_ONNX_SYNC_AFTER_LAUNCH=1` 会让 host 代码在 kernel launch 后调用 `cuStreamSynchronize()`，更容易把异步 CUDA 错误定位到对应的 QDQ op。它不会给 Triton device kernel 自动增加 host C++ 的调试符号；本方案主要调试 `simo_qdq_ops.cc` 的 host 逻辑、属性解析、shape 检查、launch 参数和错误处理。

### 2.6 如果希望使用仓库标准 editable Debug 安装

如果还需要同时 Debug `simo._C`，可以复用同一个 wrapper：

```bash
cd "$REPO"
DEBUG=1 \
CXX=/tmp/simo-cxx-debug \
CUDA_HOME=/share_data/users/like/opt/cuda-13.0 \
  "$PYTHON" -m pip install -e . \
    --no-build-isolation \
    --no-deps \
    -v 2>&1 | tee temp/simo-editable-debug-build.log
```

这条命令的效果是：

- `DEBUG=1` 让 `simo._C` 的 `setup.py` 参数使用 `-O0 -g`；
- `CXX=/tmp/simo-cxx-debug` 让 `simo_qdq_ops.cc` 所属的 custom-op library 把硬编码的 `-O3` 替换为 `-O0 -g3`；
- editable 安装会把最终 library 写回 `simo/onnx/ort_plugin/libSimoOnnxCustomOps_sm90.so`。

最后一点很重要：这条命令会覆盖当前 source package 中已有的 production `.so`。如果需要同时保留 production 和 Debug 版本，优先使用 2.3 节的独立输出路径，并通过 `SIMO_ONNX_CUSTOM_OPS_LIBRARY` 选择 Debug library。

### 2.7 当前 builder 的增量编译特性与常见误区

`build_runtime.py` 当前使用 `tempfile.TemporaryDirectory()`，生成 source 后直接执行一次带所有 `.cc` 的 `c++ -shared` 命令；它没有 Ninja/CMake build tree，也没有 `.o` 级别的增量编译。因此每次调用 `build_sm90_runtime()` 都会重新：

- 生成 Triton embedded cubin source；
- 编译 `simo_qdq_ops.cc` 及其他 custom-op `.cc`；
- 链接完整的 shared library。

这次构建可能比只编译一个 `.cc` 慢，但它保证最终 Debug library 中所有 host source 与同一次生成的 cubin ABI 一致。不要只给 `simo_qdq_ops.cc` 做 `g++ -c` 后替换生产 `.so`，因为当前 builder 没有公开的 object-file/link manifest，容易漏掉 include、generated source、version script 或 `-lcuda` 参数。

常见错误包括：

- 只执行 `DEBUG=1 pip install -e .`，但忘记 `build_runtime.py` 仍固定使用 `-O3`；
- 编译出了 Debug `.so`，运行时却没有设置 `SIMO_ONNX_CUSTOM_OPS_LIBRARY`，实际加载的是 source package 里的旧库；
- 设置了 `SIMO_ONNX_QDQ_PROVIDER=CPU`，导致没有进入 `simo_qdq_ops.cc`；
- `CUDA_HOME` 指向 CUDA 12.x，但 PyTorch、ORT 或 Triton 使用 CUDA 13.0；
- 只检查 `.symtab`，没有检查 `.debug_info`/`.debug_line`，误以为已有完整 Debug 信息；
- 只在 host 代码中设置断点，却期待 GDB 进入 embedded Triton device kernel。后者需要单独的 CUDA device-side 调试方案，不由 `c++ -g3` 提供。

### 2.8 可复用的最短流程

针对本次 `simo_qdq_ops.cc` 调试，最短可靠流程是：

```bash
cd /share/users/like/package/simo_conda_sglang
export PYTHON=/share_data/users/like/miniconda3/envs/simo_sglang/bin/python
export CUDA_HOME=/share_data/users/like/opt/cuda-13.0
export CXX=/tmp/simo-cxx-debug

mkdir -p temp/simo-debug
"$PYTHON" -c \
  'from simo.onnx.ort_plugin.build_runtime import build_sm90_runtime; \
   build_sm90_runtime("temp/simo-debug/libSimoOnnxCustomOps_sm90_debug.so")'

export SIMO_ONNX_CUSTOM_OPS_LIBRARY="$PWD/temp/simo-debug/libSimoOnnxCustomOps_sm90_debug.so"
unset SIMO_ONNX_QDQ_PROVIDER
export SIMO_ONNX_SYNC_AFTER_LAUNCH=1

readelf -S "$SIMO_ONNX_CUSTOM_OPS_LIBRARY" | grep -E '\.debug_(info|line)' \
  || { echo "Debug sections are missing" >&2; exit 1; }

"$PYTHON" -m pytest -s -vv \
  simo/onnx/tests/test_dynamic_qdq_runtime_debug.py \
  -k 'tiny_tensor'
```

结论是：当前项目对 `simo_qdq_ops.cc` 的 Debug 构建对象是完整的 `libSimoOnnxCustomOps_sm90.so`，不是 `simo._C`。临时调试使用 `build_sm90_runtime()` + `CXX` wrapper 最直接；整个 editable 开发环境使用 `DEBUG=1` 时，也必须保留该 wrapper，才能覆盖 `build_runtime.py` 内部硬编码的 `-O3`。
