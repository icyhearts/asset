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

## 3. `sglang_sipu` 中 `2a. Run CUDA` 命令的功能

### 3.1 一句话结论

命令

```bash
bash test_utils/run_test.sh \
  --config-yaml configs/deepseek/ds_v32_2layer.yaml \
  --launch-config deepep_deepgemm_text \
  --test-case text-only \
  --device cuda
```

不是在当前机器直接启动一个本地 CUDA 服务，也不是执行 CUDA 与 SIPU 的比较。它是
SIPU 精度验证流程的 **CUDA 基线生成步骤**：读取 `ds_v32_2layer.yaml`，在远端
CUDA 主机的 SGLang CUDA 容器中运行一次 `text-only` 离线推理，并把中间层张量和
输出保存为 `.pt` dump，供后续 SIPU dump 与 `compare_cuda_sipu.py` 对比。

该结论对应的源码调用链是：

```text
run_test.sh
  -> run_cuda
  -> ssh <user>@10.96.11.15
  -> docker run lmsysorg/sglang:v0.5.18-cu130
  -> test_utils/run_test_job.py --device cuda
  -> sgl.Engine(**cuda_block).generate(...)
  -> forward hooks 保存 CUDA 张量
```

### 3.2 正确的执行目录

文档在执行命令前要求：

```bash
cd /share/users/like/package/sglang_sipu/test/srt/sipu
```

然后再执行上面的命令。用户当前的 shell 目录是
`/share/users/like/package/simo_conda_sglang`；在该目录直接写
`bash test_utils/run_test.sh` 会找不到脚本。也可以从 `sglang_sipu` 仓库根目录使用
绝对脚本路径，例如：

```bash
cd /share/users/like/package/sglang_sipu
bash test/srt/sipu/test_utils/run_test.sh \
  --config-yaml test/srt/sipu/configs/deepseek/ds_v32_2layer.yaml \
  --launch-config deepep_deepgemm_text \
  --test-case text-only \
  --device cuda
```

脚本会把相对 YAML 路径解析到
`<sglang_sipu>/test/srt/sipu/` 下，因此从文档指定目录运行时，
`configs/deepseek/ds_v32_2layer.yaml` 才是最直观的写法。

### 3.3 四个参数分别选择什么

| 参数 | 实际作用 |
| --- | --- |
| `--config-yaml configs/deepseek/ds_v32_2layer.yaml` | 读取模型路径、prompt、采样参数、各设备的 Engine 参数和允许的测试 case。 |
| `--launch-config deepep_deepgemm_text` | 选择 YAML 的 `launch_configs.deepep_deepgemm_text` 块；不会执行其他 launch 配置。 |
| `--test-case text-only` | 选择 `run_test_job.py` 的 `run_text_only`；只执行文本生成，不执行同一 YAML 中的 `prefix-caching`。 |
| `--device cuda` | 选择 `run_test.sh` 的 `run_cuda` 分支，并把 `--device cuda` 传给 `run_test_job.py`；不会运行 SIPU 分支。 |

`run_test_job.py` 会先校验：YAML 存在 `model_path` 和 `launch_configs`，所选 launch
同时有 `cuda`/`sipu` 两个 block，且 `text-only` 出现在 `tests` 列表中。校验失败时
不会创建 Engine。

### 3.4 这个具体 YAML 会运行什么

`configs/deepseek/ds_v32_2layer.yaml` 中与本命令相关的设置如下：

* 模型 ID 是 `ds_v32_2layer`，模型目录是
  `/share_data/inference-framework/tiny-models/DeepSeek-V3.2-2layer/safetensor_weights`。
* 顶层只有一个长文本 prompt；`text-only` 会把它作为一个元素传给
  `llm.generate`。
* 采样参数是 `temperature: 0`、`top_p: 0.95`、`max_new_tokens: 2`。
  这不是吞吐 benchmark，而是一个短的确定性 smoke/accuracy run。
* `enable_tensor_dump: true` 会注入 `forward_hooks`。
* CUDA Engine 参数包括：`attention_backend: dsa`、`kv_cache_dtype: fp8_e4m3`、
  `moe_runner_backend: deep_gemm`、`moe_a2a_backend: deepep`、
  `deepep_mode: low_latency`、`disable_shared_experts_fusion: true`、
  `disable_cuda_graph: true`、`context_length: 256`、`max_total_tokens: 512`、
  `page_size: 64`、`skip_server_warmup: true` 和 debug 日志。
* `test_env.text-only` 会设置
  `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0`，影响 DSA prefill 的 dense
  attention 选择；它不是一个额外的命令行参数。

YAML 中的 `base_gpu_id: 2` 会作为 SGLang Engine 参数传入，并参与 scheduler 的
GPU ordinal 计算。远端物理 GPU 可见性则由 `run_test.sh` 的
`CUDA_GPUS`/`NVIDIA_VISIBLE_DEVICES` 环境控制（默认值为 `0`）。因此当前示例存在
一个需要运行前确认的风险：如果容器实际上只看到 GPU `0`，SGLang 仍可能尝试选择
`cuda:2`，从而报设备不存在或使用错误的卡。应让 `base_gpu_id`、可见 GPU 列表和
并行度保持一致（单卡容器通常使用逻辑 ordinal `0`，或者显式暴露并使用对应的
多卡编号）。当前脚本的实际 Docker 参数是 `--gpus all` 加
`NVIDIA_VISIBLE_DEVICES`，不是注释中所说的字面 `--gpus device=N`，部署时应按脚本
实际行为检查 GPU 映射。

### 3.5 `--device cuda` 的实际执行位置和依赖

`run_test.sh` 的 `run_cuda` 会：

1. 默认通过 SSH 连接 `${USER}@10.96.11.15`；可用 `ACCURACY_CUDA_HOST` 覆盖。
2. 在远端先删除同名的旧 CUDA 容器，再启动
   `lmsysorg/sglang:v0.5.18-cu130`。
3. 将当前 `sglang_sipu` checkout 挂载到容器内
   `/sgl-workspace/sglang`，并以读写方式挂载 `/share_data/sglang_sipu`，以只读方式
   挂载 tiny-models 目录。
4. 设置 CUDA 运行所需的 `PYTHONPATH`、`PYTHONUNBUFFERED`、
   `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`、DeepGEMM/NCCL/NVSHMEM 相关环境，
   然后调用容器内的 `run_test_job.py`。
5. 通过本地 `tee` 把远端 stdout/stderr 同时写入日志文件。

因此 CUDA 主机必须满足：SSH 可达、Docker 可用、上述镜像可拉取或已存在、模型和
共享目录在相同路径可见，且挂载后的 checkout 能导入 SGLang。执行 CUDA 分支不要求
设置 `SIPU_TEST_CONTAINER`；该环境变量只在 `--device sipu` 或 `both` 时使用。

### 3.6 推理期间保存的内容

`run_test_job.py` 会执行近似如下逻辑：

```python
cfg = load_yaml(...)
kwargs = deepcopy(cfg["launch_configs"]["deepep_deepgemm_text"]["cuda"])
kwargs["model_path"] = cfg["model_path"]
kwargs["device"] = "cuda"
kwargs["forward_hooks"] = build_forward_hooks(dump_dir=...)
llm = sgl.Engine(**kwargs)
outputs = llm.generate(cfg["prompts"], cfg["sampling"])
llm.shutdown()
```

文本 hook 通常会保存以下类别的 CPU `.pt` 文件（每次 forward 位于一个
`pass_XXX/` 目录）：

* `embedding.pt`；
* 每个文本层的 `layer_<i>_attn.pt`、`layer_<i>_attn_plus_residual.pt`、
  `layer_<i>_mlp.pt` 和 `layer_<i>_mlp_plus_residual.pt`；
* `layer_last_mlp_plus_residual.pt` 和 `lm_head.pt`；
* `manifest.json` 以及记录 prompt/生成结果的 `run_meta.json`。

hook writer 在一次新运行开始时会清理目标目录中已有的 `pass_*`、`manifest.json`
和 `run_meta.json`，因此重复执行同一命令会覆盖该 tuple 的旧 CUDA dump。

### 3.7 输出位置

使用默认参数时，CUDA 张量 dump 是：

```text
/share_data/sglang_sipu/accuracy_verify/<当前用户名>/
  ds_v32_2layer/deepep_deepgemm_text/text-only/cuda/
```

日志默认写回当前 checkout：

```text
/share/users/like/package/sglang_sipu/test/srt/sipu/logs/deepseek/
  ds_v32_2layer_deepep_deepgemm_text_text-only_cuda.log
```

实际目录可分别用 `--dump-base` 和 `--log-base` 覆盖。命令结束时打印 dump base 和
log host 路径；生成文本也会打印到终端和日志中。

### 3.8 它不做什么，以及下一步如何使用

这个命令本身：

* 不启动 SIPU 推理；
* 不调用 `compare_cuda_sipu.py`；
* 不运行 YAML 中列出的 `prefix-caching`，因为显式选择了 `text-only`；
* 不代表多 rank、在线服务或性能回归已经通过。文档明确说明该 accuracy 测试是
  one-rank/archmodel 范围。

完整的 CUDA↔SIPU 验证顺序是：

```text
1. 本命令：生成 .../text-only/cuda/ 基线
2. 同参数改为 --device sipu：生成 .../text-only/sipu/
3. 在 SIPU 容器中运行 compare_cuda_sipu.py：比较两侧 pass_XXX/*.pt
```

比较脚本会按 checkpoint 计算 shape、余弦相似度、绝对误差等指标；当前文档的门槛
是最后一个 `lm_head.pt` 满足 `cos_sim >= 0.999` 且 `mean_atol <= 0.05`。因此，
`--device cuda` 的准确描述是“为 SIPU 精度测试准备可复现的 CUDA 参考结果”，而不是
“验证 SIPU 已经正确”。

### 3.9 最短可操作示例

```bash
cd /share/users/like/package/sglang_sipu/test/srt/sipu

# 可选：指定远端 CUDA 主机和可见 GPU
export ACCURACY_CUDA_HOST="${USER}@10.96.11.15"
export CUDA_GPUS=0

bash test_utils/run_test.sh \
  --config-yaml configs/deepseek/ds_v32_2layer.yaml \
  --launch-config deepep_deepgemm_text \
  --test-case text-only \
  --device cuda
```

如果共享目录中已经有同一模型/launch/case 的 CUDA baseline，可以跳过本步骤，直接
生成 SIPU dump，并把 `--dump-base` 指向包含该 baseline 的用户目录。

## 4. `base_gpu_id` 与 `run_test.sh` CUDA 容器可见 GPU 的关系

### 4.1 直接结论

`test/srt/sipu/configs/deepseek/ds_v32_2layer.yaml` 中的
`base_gpu_id: 2` **不会控制 Docker 容器暴露哪些 GPU**。它是在容器已经启动后、
SGLang 创建 scheduler 子进程的启动阶段使用的 GPU 起始编号，用来决定 SGLang 在
“当前进程可见的 CUDA 设备编号空间”中绑定哪张卡。

当前 `run_test.sh` 的 GPU 选择分成两层：

```text
远端物理 GPU 选择/注入：CUDA_GPUS -> NVIDIA_VISIBLE_DEVICES（Docker/NVIDIA runtime）
应用内逻辑 GPU 选择：base_gpu_id、gpu_id_step、tp_size、pp_size（SGLang）
进程内再次屏蔽/重排：CUDA_VISIBLE_DEVICES（如果被设置）
```

所以：

* 改 `base_gpu_id` 不会把 GPU 2 注入容器，也不会改变 `nvidia-smi` 能看到的设备。
* 选定“主机物理 GPU 2”应在 Docker 层设置 `CUDA_GPUS=2`，并使用明确的
  `--gpus '"device=2"'`（或等价 CDI 请求）；不能只把 `base_gpu_id` 写成 2。
* 如果运行时只注入主机 GPU 2，它在只有这一张卡的容器进程中通常会重新编号为
  逻辑 `cuda:0`，这时 SGLang 应使用 `base_gpu_id: 0`。
* `base_gpu_id: 2` 只有在 SGLang 进程最终确实看到了至少三个逻辑设备，并且确实
  想绑定第三个逻辑设备时才合理。

NVIDIA 官方文档把 `NVIDIA_VISIBLE_DEVICES` 定义为容器 GPU enumeration/access
控制项；Docker 的 `--gpus` 也可以指定 GPU。另一个层面的 CUDA
`CUDA_VISIBLE_DEVICES` 则控制应用看到的设备及其枚举顺序：[NVIDIA Container
Toolkit GPU Enumeration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)、
[CUDA `CUDA_VISIBLE_DEVICES` 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)。

### 4.2 当前脚本到底怎样设置可见 GPU

`run_test.sh` 的关键代码是：

```bash
CUDA_GPU="${CUDA_GPU:-0}"
CUDA_GPUS="${CUDA_GPUS:-${CUDA_GPU}}"
```

以及远端 Docker 命令：

```bash
docker run --rm \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES="${CUDA_GPUS}" \
  ...
```

因此默认值为：

```text
CUDA_GPU=0
CUDA_GPUS=0
NVIDIA_VISIBLE_DEVICES=0
```

这意味着脚本**想要**只让远端主机的物理 GPU 0 进入 CUDA 容器。实际选择设备的
变量是 `CUDA_GPUS`，不是始终是 `CUDA_GPU`：如果用户预先设置了
`CUDA_GPUS=2,3`，脚本会使用 `2,3`；此时 `CUDA_GPU` 主要还影响默认容器名中的
`gpu<编号>` 字样。

脚本第 41 行的注释写着“`--gpus device=N`”，但当前实际第 228 行写的是
`--gpus all`，第 230 行才通过 `NVIDIA_VISIBLE_DEVICES` 传入选择列表。这两套
机制同时出现会产生歧义：按标准 NVIDIA Container Toolkit 语义，
`NVIDIA_VISIBLE_DEVICES` 是设备 enumeration 选择器，而 `--gpus all` 是 Docker
GPU resource request；当两者冲突时，最终结果取决于 Docker/NVIDIA runtime 的
版本和 legacy/CDI 模式。不能只看 `--gpus all` 就断定容器一定只看到一张卡，也
不能只看环境变量就跳过实测。

若希望设备选择在 Docker 层完全明确，建议二选一：

```bash
# 方案 A：保留当前脚本的写法（只有在远端 runtime 已验证时才使用）
docker run --rm --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=2,3 \
  ...

# 方案 B：使用 Docker device request（推荐；去掉 NVIDIA_VISIBLE_DEVICES）
docker run --rm --gpus '"device=2,3"' \
  ...
```

不要把 `--gpus all` 和另一个互相矛盾的 `--gpus '"device=..."'` 同时传入。若
使用 UUID（例如 `GPU-...`）而不是主机 index，设备选择对主机 GPU 排序变化更稳定；
无论使用 index 还是 UUID，SGLang 的 `base_gpu_id` 仍应按容器内的逻辑 ordinal
设置。对于本仓库当前脚本，若需要可靠隔离，建议把 `run_cuda` 的 Docker 参数改为
`--gpus '"device=${CUDA_GPUS}"'`，并删除（或保证完全一致地设置）
`-e NVIDIA_VISIBLE_DEVICES=...`；这比同时使用 `--gpus all` 和环境变量更可预测。

在当前工作节点的 Docker 29.6.2 + NVIDIA Container Toolkit 1.19.1 环境中，用同系列
CUDA 13.0 SGLang runtime 镜像做过一次临时 probe：`--gpus all
-e NVIDIA_VISIBLE_DEVICES=0` 仍列出了全部 8 张卡，而
`--gpus '"device=0"'` 只列出 1 张卡。这个结果说明当前脚本的环境变量不能被当成
跨 runtime 的唯一隔离机制；远端 `10.96.11.15` 仍应在实际运行前自行 probe，不能把
本机 runtime 的结果直接假定为远端结果。

### 4.3 `base_gpu_id` 在 SGLang 内部的实际路径

这条命令中，`run_test_job.py` 会把所选 CUDA block 复制成 `sgl.Engine` 参数，
所以 YAML 的 `base_gpu_id` 会进入 `ServerArgs`，但 `run_test.sh` 本身不会读取或
改写它。

在当前 SGLang source 中，普通 scheduler 路径大致是：

```python
gpu_id = (
    server_args.base_gpu_id
    + pp_offset
    + tp_offset * server_args.gpu_id_step
)
with maybe_reindex_device_id(gpu_id) as gpu_id:
    start_scheduler_process(..., gpu_id=gpu_id)
```

对应源码位置是 `python/sglang/srt/entrypoints/engine.py:892-924`。随后
`ModelRunner` 保存这个 `gpu_id`，并在初始化时调用：

```python
torch.get_device_module(self.device).set_device(ps.gpu_id)
```

见 `python/sglang/srt/model_executor/model_runner.py:387-397`。因此
`base_gpu_id` 本质上是 SGLang 传给 `torch.set_device` 的起始 **逻辑 ordinal**，
不是 NVIDIA runtime 的 host-physical selector。

默认情况下 `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS` 在
`python/sglang/srt/environ.py:627` 是 `false`，`maybe_reindex_device_id` 不会
自动把这个编号转换成另一张物理卡；`ParallelState` 也会使用传入的 local rank
（`python/sglang/srt/distributed/parallel_state.py:319-323`）。只有显式打开
`SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=true` 时，SGLang 才会按已有的
`CUDA_VISIBLE_DEVICES` 列表为子进程做额外重映射（`python/sglang/srt/utils/common.py:1207-1230`）。
这不是当前 `run_test.sh` 的默认行为，不应拿它作为修复 `base_gpu_id` 配置的替代品。

### 4.4 物理编号与逻辑编号的例子

假设远端主机物理 GPU 编号为 `0,1,2,3`，且没有额外的应用级
`CUDA_VISIBLE_DEVICES`：

| `CUDA_GPUS` / `NVIDIA_VISIBLE_DEVICES` | 容器实际注入的主机卡 | 进程通常看到的逻辑序号 | 合适的 `base_gpu_id` |
| --- | --- | --- | --- |
| `0` | 物理卡 0 | `cuda:0` | `0` |
| `2` | 物理卡 2 | `cuda:0`（单卡被重新枚举） | `0` |
| `2,3` | 物理卡 2、3 | `cuda:0 -> 2`，`cuda:1 -> 3` | `0`；TP=2 时再设 `tp_size: 2` |
| `0,1,2` | 物理卡 0、1、2 | `cuda:0`、`cuda:1`、`cuda:2` | `2` 表示第三张可见逻辑卡 |
| `all` | 全部物理卡 | 通常为 `cuda:0 ... cuda:N-1` | 按实际并行布局决定 |

表中的“通常”是因为 MIG、UUID、容器 runtime mode 和应用级
`CUDA_VISIBLE_DEVICES` 都可能改变枚举细节；最终应以容器内
`torch.cuda.device_count()` 和 UUID/PCI bus 检查为准。关键点不变：
`NVIDIA_VISIBLE_DEVICES=2` 选择的是主机物理卡 2，而不等价于让应用拥有一个
名为 `cuda:2` 的逻辑设备。

### 4.5 当前 `ds_v32_2layer.yaml` 的风险与推荐配置

当前 YAML 的 CUDA block 是：

```yaml
base_gpu_id: 2
```

而脚本默认是 `CUDA_GPUS=0`。在标准单卡可见场景中，进程只有逻辑 `cuda:0`，
SGLang 可能执行 `set_device(2)` 并得到 `invalid device ordinal`；即使某个 runtime
配置让所有 GPU 意外可见，也可能把测试跑到并非预期的第三张卡上。

如果目标是只使用远端物理 GPU 2，推荐：

```text
CUDA_GPU=2
CUDA_GPUS=2
YAML base_gpu_id=0
```

例如先复制一份 CUDA 测试 YAML，将 CUDA block 的 `base_gpu_id` 改为 `0`，再运行：

```bash
cd /share/users/like/package/sglang_sipu/test/srt/sipu
export CUDA_GPU=2
export CUDA_GPUS=2
export ACCURACY_CUDA_HOST="${USER}@10.96.11.15"

bash test_utils/run_test.sh \
  --config-yaml configs/deepseek/ds_v32_2layer_cuda_gpu2.yaml \
  --launch-config deepep_deepgemm_text \
  --test-case text-only \
  --device cuda
```

如果目标是让 TP=2 使用远端物理 GPU 2、3，则应使用：

```text
CUDA_GPUS=2,3
YAML base_gpu_id=0
YAML tp_size=2
YAML gpu_id_step=1（默认值通常就是 1）
```

这里 `base_gpu_id=0` 表示第一张**容器逻辑卡**，不是主机物理卡 0。
仅设置 `CUDA_GPUS=2,3` 而不设置 `tp_size=2`，只会让两张卡可见，不保证 SGLang
真的启动两个 TP worker；“可见”与“被并行布局使用”是两个独立条件。

### 4.6 如何在远端确认最终可见 GPU

不要用 YAML 的 `base_gpu_id` 推测容器可见性，建议用与 `run_test.sh` 相同的镜像和
环境做一次轻量 probe。下面命令只打印设备信息，不加载模型：

```bash
export ACCURACY_CUDA_HOST="${USER}@10.96.11.15"
export CUDA_GPUS=2,3

ssh "${ACCURACY_CUDA_HOST}" \
  "docker run --rm --gpus all \
     -e NVIDIA_VISIBLE_DEVICES='${CUDA_GPUS}' \
     lmsysorg/sglang:v0.5.18-cu130 \
     nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv,noheader"

ssh "${ACCURACY_CUDA_HOST}" \
  "docker run --rm --gpus all \
     -e NVIDIA_VISIBLE_DEVICES='${CUDA_GPUS}' \
     lmsysorg/sglang:v0.5.18-cu130 \
     python3 -c 'import os,torch; \
print(\"NVIDIA_VISIBLE_DEVICES=\", os.getenv(\"NVIDIA_VISIBLE_DEVICES\")); \
print(\"CUDA_VISIBLE_DEVICES=\", os.getenv(\"CUDA_VISIBLE_DEVICES\")); \
print(\"torch.cuda.device_count=\", torch.cuda.device_count()); \
[print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]'"
```

应同时检查：

* `NVIDIA_VISIBLE_DEVICES` 的值是否是期望的 `CUDA_GPUS`；
* `nvidia-smi` 输出的 UUID/PCI bus 是否对应主机目标卡（不要只比较容器内的 index）；
* `torch.cuda.device_count()` 是否等于预期可见卡数；
* 对 `CUDA_GPUS=2` 的单卡测试，`torch.cuda.device_count()` 是否为 1，此时
  SGLang 的 `base_gpu_id` 应为 0。

实际 SGLang 作业的日志也应搜索 `gpu_id`、`CUDA_VISIBLE_DEVICES`、`tp_rank` 和
`tp_size`。如果 `device_count=1` 但配置仍是 `base_gpu_id=2`，应在启动前修正
YAML，而不是继续调大 Docker 的 `--gpus` 参数。

### 4.7 最终判定

对本问题可以归纳为：

```text
base_gpu_id       = SGLang 进程选择哪个逻辑 ordinal
CUDA_GPUS         = run_test.sh 传给 NVIDIA runtime 的选择列表
NVIDIA_VISIBLE_DEVICES = 容器层允许/注入哪些主机 GPU
CUDA_VISIBLE_DEVICES   = 应用层进一步屏蔽或重排逻辑 ordinal
```

因此，当前示例的 `base_gpu_id: 2` 不会影响“容器是否看得到 GPU”；它只会影响
SGLang 在容器已经可见的逻辑设备中尝试绑定哪一个。脚本默认选择 `CUDA_GPUS=0`，
所以最稳妥的单卡 CUDA baseline 配置是把 `base_gpu_id` 改成 `0`；若要选择主机
物理 GPU 2，则设置 `CUDA_GPU/ CUDA_GPUS=2`，同时仍保持 `base_gpu_id=0`。

## 5. `MXINT8_COLS_ALIGNMENT=128` 的硬件、格式和 sikernel 原因

### 5.1 分析范围与直接结论

本节使用以下路径作为相对路径基准：

* `vllm-sipu/`：`/share/users/like/package/vllm-sipu`
* `sikernel/`：`/share/users/like/package/vllm-sipu/.deps/sikernel-src`
* `sdk/`：`/share_data/sicx_sdk/release/2609020046`
* `compiler-toolchain/`：`/share/users/like/package/compiler-toolchain`
* `isa/`：`/softhome/like/asset/code/isa`

验证日期为 **2026-09-15**，SDK 为 `2609020046`，运行后端为日志中显示的
`swemusp` CModel。

先给出结论：

1. **对当前未修改的 vllm-sipu + 当前 `mma_bf16_mxi8_universal`，K 维必须继续
   padding 到 128 的整数倍。不能只把 `MXINT8_COLS_ALIGNMENT` 从 128 改成 64。**
2. **128 不是 r32 MXINT8 ISA 的最小 K 粒度。**硬件的一个 r32 MXINT8 tile 是
   `32 x 32` 个 MXINT8 element；`tmma` 有 m1、m2、m4 三种输入规模，对应
   K=32、64、128。
3. **单独看 `hp_to_mx` 量化器，r32 BF16 -> MXINT8 只要求 K 是 32 的整数倍。**
   当前 sikernel 自测明确覆盖并通过 K=32、64、288。
4. 128 同时解决了当前代码中的两个软件问题：
   * universal GEMM 的主路径按 K=128、`tmxint8m4` 计算；
   * vllm-sipu 的 packed buffer 大小公式只按“逻辑 tile 数 x 1088 B”计算，
     没有按四个 tile 的完整 supertile 计算。`rows % 32 == 0` 且
     `cols % 128 == 0` 时，1x4 supertile 总能被完整填满，因此这个简化公式成立。
5. **硬件和内层 K=64 kernel 是可用的，但当前 universal wrapper 的 K=64 分支有
   host/device 分块粒度不一致。**隔离实验修正该分块后，`64x64x64` GEMM 的平均
   相对误差为 `0.01253165`；只放宽 128 校验、不修分块时结果包含 NaN。

把不同层的要求分开看最清楚：

| 层次 | r32 下的 K/列要求 | 说明 |
| --- | --- | --- |
| BF16 普通 tile | 16 个 BF16 element/tile | 32 行 x 每行 32 B，即 `32 x 16` BF16 |
| BF16 -> MXINT8 `hp_to_mx` | K 是 32 的整数倍 | 两个 BF16 tile 转换成一个 MXINT8 tile |
| 单条 r32 MXINT8 `tmma` | K=32/64/128 | 分别是 m1/m2/m4 |
| MXINT8 物理存储 | 每个 supertile 固定 4 个 tile | supertile 可排成 4x1、2x2、1x4 |
| 当前 vllm-sipu API | K 是 128 的整数倍 | Python、JIT C++ 都显式校验 |
| 当前 universal GEMM 常用分支 | `M,N` 按 32，K 按 128 | `32x32x128` macro-kernel |

### 5.2 需要先纠正的 tile/supertile 计算

问题中的计算把 **BF16 tile 的列宽**用于解释 **MXINT8 supertile**，这两者不是
同一种 format 下的 tile。

`isa/index.html:569-575 - Tile Reg 排布方式` 给出 r32 tile 为 `32 x 32B`。
因此：

```text
BF16:   每个 element 2B -> 一个 r32 tile = 32 x 16 element = 1024B
MXINT8: 每个 element 1B -> 一个 r32 tile = 32 x 32 element = 1024B data
```

BF16 转成 MXINT8 是 2:1 压缩，所以一次 r32 转换是：

```text
2 个 BF16 tile = 32 x 32 BF16
              -> tcvt.tt.mxi8.bf16.r32
1 个 MXINT8 tile = 32 x 32 MXINT8 + 64B tile header
```

对应代码是：

* `sikernel/source/source_builtin/misc/hp_to_mx/kernel/hp_to_output_traits.hpp:43-46`
  定义 `tbfloat16m1/m2/...`。
* 同文件 `:162-164 - HpToOutputTraits<mxint8, HP_T>` 把 MXINT8 的
  `INPUT_LMUL_16BIT` 设为 2，并选择 `tcvt_mxi8`。
* `compiler-toolchain/llvm-project/clang/test/CodeGen/RISCV/xsotile150g/wrappers/tcvt_test.cpp:4238-4241 - __rvv_tcvt_tt_mxi8_bf16_r32`
  明确验证 `tbfloat16m2_t -> tmxint8m1_t` 生成
  `tcvt.tt.mxi8.bf16.r32`。

因此，若 supertile 是 MXINT8 的 1x4 tile：

```text
1 x 4 个 MXINT8 tile
= [1 * 32, 4 * 32] 个 MXINT8 element
= [32, 128]
```

它不是 `[32,64]`。`[32,64]` 是“4 个 BF16 tile”的元素范围，但一个 1x4
MXINT8 supertile 对应的是 8 个 BF16 input tile。

另外，1x4 也不是所有 r32 MX tensor 的唯一 supertile 形状：

* `sdk/include/SiTe/site/adapter/sipu150.hpp:36-45 - site::adapter::sipu150::Traits::tile_columns`
  对 8-bit、32-row format 算出每个 tile 为 32 列。
* 同文件 `:48-61 - site::adapter::sipu150::Traits::supertile_shape` 规定：
  * 总列数 `<= 32`：4x1；
  * 总列数 `<= 64`：2x2；
  * 总列数 `> 64`：1x4。
* `isa/index.html:4809-4814 - superTileDim` 也列出 MX 合法形状
  `[4,1]`、`[2,2]`、`[1,4]`，而 NonMX 是 `[1,1]`。

所以，对 r32 MXINT8：

```text
K <= 32:  4x1 supertile，物理覆盖 128x32
K <= 64:  2x2 supertile，物理覆盖 64x64
K > 64:   1x4 supertile，单个 supertile 物理覆盖 32x128
```

“物理覆盖”不等于逻辑 shape 必须等于该值；不足部分由 tiled layout/DTE 作为
padding，但分配仍以完整 supertile 为单位。

### 5.3 从 vllm-sipu 到 sikernel 的调用链

权重侧调用链：

```text
vllm_sipu/model_executor/layers/quantization/mxint8.py:156-174
MXInt8LinearMethod::process_weights_after_loading
  -> get_padded_mxint8_shape(output_size, input_size)
  -> quantize_to_mxint8(weight, padded_shape)
  -> 保存 packed_weight 和 mxint8_weight_shape
```

推理 activation 侧和 GEMM 调用链：

```text
vllm_sipu/model_executor/layers/quantization/mxint8.py:176-190
MXInt8LinearMethod::apply
  -> vllm_sipu/model_executor/layers/quantization/utils/mxint8_utils.py:202-224
     apply_mxint8_linear
       -> activation rows padding
       -> quantize_to_mxint8(input, [M_pad, K_pad])
       -> mxint8_bf16_matmul.sikernel(qinput, qweight, M_pad, N_pad, K_pad)
       -> crop 回原始 M、N
```

量化调用链：

```text
vllm_sipu/model_executor/layers/quantization/utils/mxint8_utils.py:163-199
quantize_to_mxint8
  -> :129-160 build_mxint8_tile_tensor_on_cpu
  -> :70-97 linear_to_tileformat
  -> vllm_sipu/ops/quantization.py:108-110 quantize_to_mxint8
  -> vllm_sipu/ops/op_list.yaml:971-978 provider dispatch
  -> vllm_sipu/ops/backends/sikernel/jit/mxint8.py:104-117 quantize_to_mxint8
  -> csrc/jit/quantization/mxint8.cpp:72-112 quantize_to_mxint8
  -> csrc/jit/quantization/mxint8.cpp:55-60 launch_quantize_to_mxint8
  -> sikernel/include/sikernel.h:2350-2354 hp_to_mx（默认 tiled input）
  -> sikernel/source/source_builtin/misc/hp_to_mx/kernel/hp_to_mx_kernel.su:28-87 hp_to_mx
  -> sikernel/source/source_builtin/misc/hp_to_mx/kernel/hp_to_mx_kernel.hpp:55-108 hp_to_mx_kernel
```

GEMM 调用链：

```text
vllm_sipu/ops/quantization.py:113-115 mxint8_bf16_matmul
  -> vllm_sipu/ops/op_list.yaml:979-985 provider dispatch
  -> vllm_sipu/ops/backends/sikernel/jit/mxint8.py:120-130 mxint8_bf16_matmul
  -> csrc/jit/quantization/mxint8.cpp:114-148 mxint8_bf16_matmul
  -> csrc/jit/quantization/mxint8.cpp:62-68 launch_mxint8_matmul
  -> sikernel/source/source_builtin/blas/L3/mma/
     tile_mma_universal_bf16_mxi8_mxi8_tiled_tensor/kernel/
     tile_mma_universal_bf16_mxi8_mxi8_tiled_tensor.su:33-96
     mma_bf16_mxi8_universal
  -> 同目录 tile_mma_universal_bf16_mxi8_mxi8_tiled_tensor.hpp:154-203
     universal::kernel_tile_mma_32x32_bf16_mxi8_mxi8_gmem_tiled_tensor
  -> 同文件 :65-150
     universal::tile_mma_32x32_bf16_mxi8_mxi8_tiled_tensor
  -> tld_blk_global_m1/m2/m4 + tmma + tcvt_bf16 + tst_*
```

### 5.4 vllm-sipu 为什么选择 128

#### 5.4.1 Python 和 JIT C++ 把 128 当成正式 API contract

`vllm-sipu/vllm_sipu/model_executor/layers/quantization/utils/mxint8_utils.py:26-31`
定义：

```python
MXINT8_ROWS_ALIGNMENT = 32
MXINT8_COLS_ALIGNMENT = 128
MXINT8_BLOCK_ELEMS = 1024
MXINT8_BLOCK_BYTES = 1088
```

随后：

* `:104-110 - get_padded_mxint8_shape` 把 K pad 到 128；
* `:113-126 - get_mxint8_storage_size` 拒绝非 128 倍数的列，并用
  `(rows * cols / 1024) * 1088` 算 packed bytes；
* `:202-220 - apply_mxint8_linear` 把同一个 padded K 同时传给 A 和 B。

JIT Python 在
`vllm-sipu/vllm_sipu/ops/backends/sikernel/jit/mxint8.py:57-68 - _expected_storage_size`
重复了相同的 128 校验和存储公式。

JIT C++ 又在
`vllm-sipu/csrc/jit/quantization/mxint8.cpp:35-52 - expected_mxint8_storage_size`
第三次定义 128，并在 `:128-132 - mxint8_bf16_matmul` 强制：

```text
N % 32 == 0
K % 128 == 0
qinput bytes == expected(M, K)
qweight bytes == expected(N, K)
```

因此只改 Python 常量会先被 JIT Python 或 C++ 拒绝；即使三处一起改，后面的
physical storage 和 GEMM 分块仍然需要修改。

#### 5.4.2 1088 B 公式隐含了“supertile 必须填满”

一个 MXINT8 tile 是：

```text
1024 B data + 64 B header = 1088 B
```

但是硬件内存不是按任意一个独立 tile 分配，而是按四个 tile 的 supertile：

```text
4 * 1024 B data + 4 * 64 B header
= 4096 B data + 256 B header
= 4352 B
```

代码证据：

* `sdk/include/SiTe/tensor/tensor.hpp:964-1005` 展示 MX8/4 的四个 64 B header，
  以及 4x1、2x2、1x4 三种摆放；
* 同文件 `:1006-1032 - SuperTile::storageSize` 对 MX format 固定使用 4 个 tile；
* 同文件 `:1417-1423 - LayoutEngine::storageSize` 返回
  `numSuperTiles * SuperTile::storageSize()`；
* `sdk/include/SiTe/sifmt/sifmt.hpp:42-68 - sifmt::tensor::superTileSize`
  对普通 MX8 返回 `4 * 1024 + 256 = 4352` B。

比较几个 shape，可以看到当前公式为何需要 128：

| MXINT8 logical shape | 逻辑 tile 数 | 选中的 supertile | 正确物理 bytes | 简化公式 bytes |
| --- | ---: | --- | ---: | ---: |
| 32x32 | 1 | 4x1 | 4352 | 1088 |
| 32x64 | 2 | 2x2 | 4352 | 2176 |
| 64x64 | 4 | 2x2 | 4352 | 4352 |
| 32x128 | 4 | 1x4 | 4352 | 4352 |
| 64x128 | 8 | 两个 1x4 | 8704 | 8704 |

表中“简化公式”是假设放宽列校验后继续使用当前公式的结果。特别是
`32x64`：如果只把列对齐改成 64，vllm-sipu 只分配 2176 B，而 `hp_to_mx`
按一个完整 2x2 supertile 需要 4352 B，会造成输出 buffer 不足。

当前 `rows % 32 == 0`、`cols % 128 == 0` 的组合会选择 1x4，并保证每个
32-row slab 都填满四个列 tile，所以“逻辑 tile 数 x 1088 B”恰好等于真实
supertile storage。这是 128 的重要原因，不只是 MMA 的计算粒度。

### 5.5 `hp_to_mx` 的真实 r32 列对齐要求是 32

`sikernel/source/source_builtin/misc/hp_to_mx/kernel/hp_to_mx_tensormap.hpp:67-82 - HpToMxTensorMapGeometry`
定义：

```cpp
input_tile_dim0  = 1024 / (Rows * input_element_bytes)
input_box_dim0   = OutputPolicy::input_lmul
output_tile_dim0 = input_box_dim0 * input_tile_dim0
```

代入 r32、BF16、MXINT8：

```text
Rows = 32
input_element_bytes = 2
input_lmul = 2

input_tile_dim0  = 1024 / (32 * 2) = 16 BF16 columns
output_tile_dim0 = 2 * 16 = 32 MXINT8 columns
```

同文件 `:84-90 - hp_to_mx_direct_shape_supported` 的直接条件就是：

```cpp
dim1 % output_tile_dim0 == 0
```

所以 r32 MXINT8 的 `dim1/K` 最小对齐是 32，不是 128。

接下来，`hp_to_mx_physical_extents` 在同文件 `:92-113` 根据逻辑列 tile 数选择
4x1、2x2 或 1x4；`hp_to_mx_checked_launch_extents` 在 `:126-165` 用
`physical supertiles * 4 tiles * (1024 + 64)` 算真实存储大小。

device code 的关键步骤位于
`sikernel/source/source_builtin/misc/hp_to_mx/kernel/hp_to_mx_kernel.hpp:55-108 - hp_to_mx_kernel`：

1. `:61-62`：每个线程分配 `input_lmul * 1024 = 2048 B` shared staging；
2. `:73-77`：一个循环迭代处理一个 logical MX output tile；
3. `:82-86`：列坐标乘 `input_lmul=2`，DTE 取两个 BF16 input tile；
4. `:88-89`：等待异步 DTE copy 完成；
5. `:90`：以 `tbfloat16m2` 从 shared memory 加载；
6. `:91`：执行 `tcvt_mxi8(..., t_r32)`，得到 `tmxint8m1`；
7. `:92-105`：把逻辑 row/col tile 映射成 4x1、2x2 或 1x4 内的物理 tile id；
8. `:106`：用 `tst_blk_global_m1` 写一个 MXINT8 tile。

`hp_to_mx` wrapper 位于
`sikernel/source/source_builtin/misc/hp_to_mx/kernel/hp_to_mx_kernel.su:28-87 - hp_to_mx`：

* `:46-48` 检查上述 K=32 粒度；
* `:49-57` 计算 physical extents、logical tile count 和真实 output bytes；
* `:60-62` 编码 tiled/linear input tensor map；
* `:79-80` 启动 device kernel；
* `:84-86` 对 `dim0 > 16` 选择 Rows=32。

注意：`hp_to_mx` API 接收的是调用者提供的裸 output pointer，没有接收 output
buffer length。它虽然在内部算出了正确 `output_bytes`，但不能替调用者扩容；
vllm-sipu 必须在调用之前使用相同的 physical-layout 规则分配 buffer。

### 5.6 ISA 和生成头文件也证明 K=32/64/128 都存在

`isa/index.html:6221 - tmma.ttt` 给出的助记符格式包含可选 `[m2/m4]`。
`isa/index.html:6266-6321 - 32 Rows Layout` 的 normal 模式明确列出：

| 模式 | A/B shape | A/B TileRegNum | 等效 K |
| --- | --- | --- | ---: |
| No-extend | 32x32B | 1 | 32 |
| K-extend, micro_num=1 | 32x64B | 2 | 64 |
| K-extend, micro_num=2 | 32x128B | 4 | 128 |

SDK 生成头文件
`sdk/bin/nds64le-elf-newlib-v5d/lib/clang/20/include/siorigin_tile_150g.h:8926-8946`
提供了三套 r32 MXINT8 x MXINT8 overload：

```text
tmxint8m1 x tmxint8m1 -> tmma ... r32       (K=32)
tmxint8m2 x tmxint8m2 -> tmma ... r32.m2    (K=64)
tmxint8m4 x tmxint8m4 -> tmma ... r32.m4    (K=128)
```

编译器测试
`compiler-toolchain/llvm-project/clang/test/CodeGen/RISCV/xsotile150g/autogenerated/xsotile_tmma_ttt_test.cpp:20243-20247 - __rvv_tmma_ttt_f32_mxi8_mxi8_r32_m1`
也检查了 m1 最终生成无 m 后缀的
`tmma.ttt.f32.mxi8.mxi8.r32`。

对当前 vllm-sipu JIT ELF 使用 SDK 的 `llvm-objdump -d --demangle`，实际看到：

```text
量化：tcvt.tt.mxi8.bf16.r32
K=128 分支：tmma.ttt.f32.mxi8.mxi8.r32.m4
K=64 分支： tmma.ttt.f32.mxi8.mxi8.r32.m2
K=32 分支： tmma.ttt.f32.mxi8.mxi8.r32
```

因此从 ISA、SDK declaration、compiler CodeGen test 和最终 ELF 四个层面，都不能
把 128 解释为硬件最小 K 粒度。

### 5.7 当前 universal GEMM 为什么仍然不能直接使用 K=64

host wrapper 位于
`sikernel/source/source_builtin/blas/L3/mma/tile_mma_universal_bf16_mxi8_mxi8_tiled_tensor/kernel/tile_mma_universal_bf16_mxi8_mxi8_tiled_tensor.su:33-96 - mma_bf16_mxi8_universal`。

对 `M >= 32`，`:42-51` 选择：

```text
K >= 128 -> Tile_M=32,  Tile_K=128
K >= 64  -> Tile_M=64,  Tile_K=64
K < 64   -> Tile_M=128, Tile_K=32
```

但 `:38` 把 host 的 `Tile_N` 永久写成 32，`:61-63` 用它计算：

```cpp
m_tile = M / Tile_M;
n_tile = N / 32;
total_tiles = m_tile * n_tile;
```

device kernel 在同目录 `.hpp:154-177 - universal::kernel_tile_mma_32x32_bf16_mxi8_mxi8_gmem_tiled_tensor`
却使用：

```text
K=128 template -> M_gran=32,  N_gran=32,  K_gran=128
K=64 template  -> M_gran=64,  N_gran=64,  K_gran=64
K=32 template  -> M_gran=128, N_gran=128, K_gran=32
```

也就是说 K=64 时 host 按 `N/32` 启动，device 按 `N/64` 解码 block id。
以 `M=N=K=64` 为例：

```text
host:   m_tile=64/64=1, n_tile=64/32=2, total_tiles=2
device: n_tile_num=64/64=1

block_id=0 -> block_m=0, block_n=0，合法
block_id=1 -> block_m=1, block_n=0，越过 M=64 的唯一 macro block
```

`.hpp:179-192` 随后会用这个多出来的 `block_m_id` 计算 A supertile 和 output
地址。因此当前 K=64 分支不是只缺少上层校验，而是 wrapper 的 launch count 本身
不匹配。K=32 分支同样存在固定 `Tile_N=32` 与 device `N_gran=128` 的差异。

K=64 内层计算本身是完整的：`.hpp:96-132 -
universal::tile_mma_32x32_bf16_mxi8_mxi8_tiled_tensor<64>` 加载两组
`tmxint8m2`，执行四个 r32.m2 MMA，得到一个 `64x64x64` macro block。
问题在其上层工作项数量和地址解码。

另外，K=64 和 K=32 分支没有 K-loop，分别只适用于精确 K=64 和 K=32；
K=96 不能通过简单地选择 K=64 分支处理。K>=128 分支才在 `.hpp:74-89`
以 4 个 tile 为步长遍历 K，因此现有主路径要求 K 是 128 的整数倍，否则
`K / 128` 会丢弃尾部。

当前该 GEMM 自带测试
`sikernel/source/source_builtin/blas/L3/mma/tile_mma_universal_bf16_mxi8_mxi8_tiled_tensor/test/test_host.cpp:30-32 - main`
只使用 `M=8,N=128,K=1024`，没有覆盖 r32 K=64/K=32 分支。

### 5.8 2026-09-15 的运行验证

#### 5.8.1 当前 vllm-sipu 基线

按用户给出的环境和命令，使用 SDK `2609020046` 重新运行：

```bash
cd /share/users/like/package/vllm-sipu
source /share_data/users/like/miniconda3/bin/activate \
  /share_data/users/like/miniconda3/envs/vllm_dev
source sipu_sdk_setup.sh
python tests/kernels/quantization/test_mxint8_unpack_pytest.py
```

结果：

```text
test_quantize_to_mxint8(m=7, k=63)               PASSED
test_mxint8_bf16_matmul(m=7, n=65, k=63)         PASSED
test_mxint8_linear_method(seq_len=7, n=65, k=63) PASSED
All 3 MXInt8 test cases passed.
```

这个测试实际把 M pad 到 32、N pad 到 96、K pad 到 128，所以证明的是当前
128 contract 正确，不证明 K=64 路径正确。

#### 5.8.2 `hp_to_mx` 的 K=32/64 量化验证

使用同一 SDK 构建并运行
`sikernel/source/source_builtin/misc/hp_to_mx/test/test_host.cpp:428-442 - main`。

测试 shape 来自同文件 `:241-247 - kMxint8Shapes`，其中包含：

```text
(1,32,32)
(1,32,64)
(1,32,288)
```

tiled input 和 linear input 两组全部通过：

```text
mxint8/bf16/tiled  (1,32,32)  ok
mxint8/bf16/tiled  (1,32,64)  ok
mxint8/bf16/tiled  (1,32,288) ok
mxint8/bf16/linear (1,32,32)  ok
mxint8/bf16/linear (1,32,64)  ok
mxint8/bf16/linear (1,32,288) ok
```

该测试在 `:173-218 - run_case` 使用 SiTe 的真实 `golden.storageSize()` 分配
supertile storage，并在 buffer 前后放 guard bytes，因此不仅验证结果，还验证
没有越过正确的 physical allocation。

#### 5.8.3 K=64 隔离 GEMM 对照实验

为了不修改主工作区，在 `/tmp` Git worktree 中做了两组 `M=N=K=64` CModel
实验。这个 shape 的 A、B 都是 `64x64`，正好完整填满一个 2x2 supertile，
每个 packed buffer 为 4352 B，不存在 `32x64` 的 buffer 少分配干扰。

实验 A 只做以下放宽：

```text
Python MXINT8_COLS_ALIGNMENT: 128 -> 64
JIT Python _expected_storage_size: 128 -> 64
JIT C++ kMxInt8ColsAlignment: 128 -> 64
```

保持 sikernel universal GEMM 不变，结果：

```text
a_shape=(64, 64), qa_bytes=4352
b_shape=(64, 64), qb_bytes=4352
finite_ratio=0.93750000
mean_relative_error=nan
```

实验 B 在此基础上，仅令 host launch 的 N tile 粒度和 K=64 device 分支一致，
即 K=64 时使用 `Tile_N=64`，结果：

```text
a_shape=(64, 64), qa_bytes=4352
b_shape=(64, 64), qb_bytes=4352
finite_ratio=1.00000000
mean_relative_error=0.01253165
```

这组实验说明：

* r32.m2 指令和当前内层 `64x64x64` kernel 能正确计算；
* “只把 128 改成 64”在当前代码上确实不正确；
* 该临时修改只是定位和可行性验证，不是完整 production patch。

### 5.9 最终回答：进入 quantize + MMA 前到底要 pad 到多少

对矩阵：

```text
A: [M, K]
B: [N, K]，当前 kernel 计算 A @ B^T
C: [M, N]
```

应区分三种答案。

#### 只调用当前 r32 `hp_to_mx`

```text
M/N 行方向：按所选 row family；本问题只讨论 r32 时按 32-row tile 处理
K 列方向：32 的整数倍
packed bytes：必须按完整 physical supertile grid 计算，不能只算逻辑 tile 数
```

所以量化器本身不要求 128。

#### 调用当前未修改的 vllm-sipu `quantize_to_mxint8 + mma_bf16_mxi8_universal`

```text
A: [round_up(M, 32), round_up(K, 128)]
B: [round_up(N, 32), round_up(K, 128)]
```

这就是当前正确且已验证的端到端 contract。这里 B 的“列”仍是 reduction K，
output N 是 B 的行方向，只要求按 32 padding。

#### 若专门启用当前内层 K=64 macro-kernel

在修复 wrapper、validation 和 storage 计算后，可使用：

```text
A: [round_up(M, 64), 64]
B: [round_up(N, 64), 64]
```

这是因为当前 K=64 分支一次计算 `64x64x64`，而不是任意 M/N 的
`32x32x64` tail kernel。对于原始 M 或 N 只有 32 行，若仍只 pad 到 32，
既不能完整填满 2x2 MX supertile，也不满足该 macro-kernel 的 M/N=64 粒度。

同理，当前 K=32 内层分支对应 `M,N` 以 128 为 macro granularity；它也不能由
一个全局 `COLS_ALIGNMENT=64` 自动、安全地启用。

### 5.10 `MXINT8_COLS_ALIGNMENT` 能否改小及建议改法

**不能直接把全局常量改成 64。**需要把“一个全局列对齐”改成“按 GEMM kernel
variant 选择的 layout contract”。至少要同时完成：

1. **按 variant 选择 shape。**例如：
   * 通用主路径：M/N=32 granularity，K 为 128 的倍数；
   * short-K64 路径：M/N=64 granularity，K 精确为 64；
   * short-K32 路径：M/N=128 granularity，K 精确为 32。
2. **重写 packed storage size。**复用 SiTe layout engine，或在 vllm-sipu 中按
   tile shape、supertile shape 和 supertile grid 计算
   `num_supertiles * 4352`，不能继续无条件使用
   `(rows * cols / 1024) * 1088`。
3. **修复 universal host launch。**`Tile_N`、`total_tiles` 必须与 device 的
   `N_gran` 一致，并对 M/N/K 做完整 divisibility validation。
4. **明确 K tail 策略。**K=96、160、192 等不能靠当前 K=64 分支解决；需要：
   * 继续 pad 到下一个 128；或
   * 新增 m4+m2/m1 混合 tail kernel；或
   * 新增真正循环 K=64 chunk 的 kernel。
5. **处理 partial supertile padding。**当前 `hp_to_mx` 默认 `ZERO_OUTPUT=false`；
   `sikernel/source/source_builtin/misc/hp_to_mx/kernel/hp_to_mx_kernel.hpp:110-118 - hp_to_mx_zero_output`
   说明未访问的 supertile padding 默认不保证为零。若后续 kernel 可能读取
   padding，应启用/实现确定性清零。
6. **补齐测试。**至少覆盖：
   * quantize：K=32、64、96、128、160、288，M/N=32、64、96；
   * GEMM K128：K=128、256、384；
   * GEMM K64：M/N 的 64 边界和 crop；
   * 非法 K=96 不得静默漏算；
   * packed buffer guard、NaN/Inf 检查及数值 reference。

因此，最终建议是：

```text
当前代码立即使用：保持 MXINT8_COLS_ALIGNMENT = 128。

若目标只是优化 K <= 64 的小矩阵：
新增一个明确的 K64 layout/kernel variant，而不是修改全局常量；
同时把 M/N pad 到 64、修复 host launch，并按 2x2 supertile 分配 4352B 的整数倍。

若目标是让所有 K 都按 64 对齐：
需要新增支持 K 循环和 K tail 的 GEMM 实现，属于 kernel/layout 接口改造，
不是 padding 常量调整。
```

换句话说，vllm-sipu 当前的 128 对齐确实比 `hp_to_mx` 和单条 SIPU ISA 的
最小要求更保守；但它并非纯粹无效 padding，而是在当前代码中同时保证
**完整 1x4 supertile storage、正确的 K=128 MMA 循环和一致的 M/N=32
工作项划分**。在这些配套逻辑改造前，不能把它安全地缩小为 64。

## 6. 独立 sikernel 的 mma_dte_tile_tensor 构建失败分析

分析日期：2026-09-16。分析对象是独立仓库
`/share/users/like/package/sikernel`，不是前一节 vllm-sipu 的 `.deps/sikernel-src`。
本次检查的 sikernel commit 为 `6d4797d4e49a`，
`mma_dte_tile_tensor` submodule commit 为 `8e368c6f1dd2`。

下文代码位置均相对独立 sikernel 根目录；CMake 的顶层语句标为“顶层配置”，
脚本/CMake 函数使用其实际函数名。

### 6.1 结论：缺少 host 测试依赖 GTest，不是 SIPU kernel 编译错误

**这份 `mma_dte_tile_tensor/cmake.log` 中，主库目标已经成功；
`--all` 随后构建测试项目时，因找不到 GoogleTest 的开发文件而失败。**
准确的失败阶段是测试子项目的 **CMake configure**，还没有进入这些失败项目的
C++ / SIPU device 源码编译阶段。

关键证据：

| 位置 | 日志内容 | 含义 |
| --- | --- | --- |
| `mma_dte_tile_tensor/cmake.log:5-8` | `Built target gen_dispatch_cases`、`mma_dte_build_manifest`、`mma_dte_umbrella`、`tile_mma_dte` | 本次增量构建的主库阶段成功 |
| `mma_dte_tile_tensor/cmake.log:9` | `Configured 74 testcase projects` | 顶层测试工程登记了 74 个独立测试项目，并非这 74 个项目都已配置/编译成功 |
| `mma_dte_tile_tensor/cmake.log:14` | `Performing configure step for ...` | 开始对子项目分别执行 CMake |
| `mma_dte_tile_tensor/cmake.log:74-82` | 第一个 `Could NOT find GTest` 及调用栈 | 首个有效错误 |
| `mma_dte_tile_tensor/cmake.log:307-308` | `mma_dte_all_testcases ... Error 2` | 下层配置失败向上层 make 传播的最终状态，不是新的根因 |

第一个错误的核心内容：

```text
Could NOT find GTest (missing: GTEST_LIBRARY GTEST_INCLUDE_DIR
GTEST_MAIN_LIBRARY)
```

整份日志共有 **16 次同类 GTest 错误**。这是多个测试子项目并行配置时重复暴露
同一个依赖缺失，不是 16 个不同的 kernel bug，也不能据此说全部 74 个项目都已尝试构建。
`mma_dte_tile_tensor/build.sh:13-20 - 顶层脚本` 将未指定时的并行数限制到最多 16。

日志里的 `SiCrossCompiler Compile Options ...` 只是 CMake 注册 SCC 编译规则时
打印的配置，不能当作对应 `.su` 已经编译完成的证据。
当前日志没有提供 intrinsic 不存在、LLVM 后端失败、链接未定义符号或 OOM 的证据。

### 6.2 从 build.sh 到 find_package(GTest) 的调用链

```text
bash mma_dte_tile_tensor/build.sh --all
  |
  +-- build.sh:40-41，BUILD_MODE="all"
  |
  +-- build.sh:118-120
       |
       +-- build_library()                      [build.sh:92]
       |    +-- cmake -S . -B build             [build.sh:96]
       |    +-- cmake --build build             [build.sh:97]
       |         -> 本次日志中成功
       |
       +-- build_testcases()                    [build.sh:100]
            +-- cmake -S testcase -B build/testcase
            |                                  [build.sh:104-105]
            +-- 构建 mma_dte_all_testcases      [build.sh:106-108]
                 |
                 +-- testcase/CMakeLists.txt:229
                      -> mma_dte_publish_testcase_binaries
                         -> 各 ExternalProject 子项目
```

继续展开一条实际失败的 SplitK 路径：

1. `mma_dte_tile_tensor/testcase/CMakeLists.txt:197-204 - 顶层配置`：
   扫描测试目录下的 `CMakeLists.txt`，对每个项目调用 `mma_dte_add_testcase_project`。
2. `mma_dte_tile_tensor/testcase/CMakeLists.txt:131 - mma_dte_add_testcase_project`：
   为每个测试建立独立 build/stamp 目录；第 150-152 行构造子项目
   `cmake -S ... -B ...` 命令，第 173-185 行交给 `ExternalProject_Add`。
3. `mma_dte_tile_tensor/testcase/func_test/splitK/test_bf16_mxfp6e3m2_r16_host/CMakeLists.txt:23-24 - 顶层配置`：
   引入 SplitK 公共配置，再调用 `add_mma_dte_splitk_test`。
4. `mma_dte_tile_tensor/testcase/func_test/splitK/common/SplitKTest.cmake:31 - add_mma_dte_splitk_test`：
   第 45-48 行登记 SCC object，第 50-53 行登记 host 可执行文件及其依赖，
   第 54 行调用 `mma_dte_test_enable_gtest`。
5. `mma_dte_tile_tensor/testcase/func_test/CMakeMmaDteCommon.cmake:352 - mma_dte_test_enable_gtest`：
   **第 353 行 `find_package(GTest REQUIRED)` 失败**，因而后面的
   `GTest::gtest`、`GTest::gtest_main` 链接设置不能完成。

另外两条日志中的路径直接调用 GTest 查找：

- `mma_dte_tile_tensor/testcase/func_test/test_validation_host/CMakeLists.txt:27 - 顶层配置`。
- `mma_dte_tile_tensor/testcase/autotune/CMakeLists.txt:23 - 顶层配置`。

因此，单独排除 autotune 不会解决问题，SplitK 和 host validation 也需要 GTest。
这里 GTest 用于 host 测试程序，不需要给 SIPU device 交叉编译一份 GoogleTest。

### 6.3 缺的是哪些文件，为什么 source SDK 没有解决

实际使用的 CMake 是：

```text
/share_data/users/like/miniconda3/envs/siinfer_dev/bin/cmake
version: 4.0.3
```

其自带模块位于该环境下的 `share/cmake-4.0/Modules/FindGTest.cmake`。
**有 `FindGTest.cmake` 只代表 CMake 知道如何查找 GTest，不代表已安装 GTest。**

该模块的查找顺序可以直接由本地源码核对：

| 相对 siinfer_dev 环境根目录的位置 | 行为 |
| --- | --- |
| `share/cmake-4.0/Modules/FindGTest.cmake:197 - 顶层配置` | 先执行 `find_package(GTest QUIET NO_MODULE)`，尝试读取安装好的 `GTestConfig.cmake` |
| `share/cmake-4.0/Modules/FindGTest.cmake:244 - 顶层配置` | config package 不存在时，查找 `gtest/gtest.h` |
| `share/cmake-4.0/Modules/FindGTest.cmake:263-265 - 顶层配置` | 查找 `gtest` 和 `gtest_main` 库 |
| `share/cmake-4.0/Modules/FindGTest.cmake:273 - 顶层配置` | 要求 `GTEST_LIBRARY`、`GTEST_INCLUDE_DIR`、`GTEST_MAIN_LIBRARY` 全部有效 |

本次检查中，`siinfer_dev` 下没有标准的 GTest 头文件、库及 config package；
现有测试子项目 cache 也记录了查找失败。例如：

```text
mma_dte_tile_tensor/build/testcase/_build/2609151958/sipu_150/default/
specialized/func_test/test_validation_host/CMakeCache.txt

第 211 行：GTEST_INCLUDE_DIR:PATH=GTEST_INCLUDE_DIR-NOTFOUND
第 214 行：GTEST_LIBRARY:FILEPATH=GTEST_LIBRARY-NOTFOUND
第 220 行：GTEST_MAIN_LIBRARY:FILEPATH=GTEST_MAIN_LIBRARY-NOTFOUND
第 226 行：GTest_DIR:PATH=GTest_DIR-NOTFOUND
```

使用 `--debug-find-pkg=GTest` 做最小复现时，确认 CMake **已经搜索了**
`siinfer_dev/include`、`siinfer_dev/lib` 等目录，所以不是简单的
“激活了 Conda，但 CMake 完全没搜索 Conda”问题。对应开发文件确实不在这些位置。

SDK 中虽然存在 `lib/libsiGtestMain.so`，但当前项目要求的是标准 GTest package：

```text
include/gtest/gtest.h
lib/libgtest.a 或 lib/libgtest.so
lib/libgtest_main.a 或 lib/libgtest_main.so
lib/cmake/GTest/GTestConfig.cmake   # config 模式的常见安装位置
```

`libsiGtestMain.so` 的名字不匹配上述模块的库搜索规则，单独存在这个文件也没有补齐
标准 GTest 头文件/config。不能把它当成已经满足 `find_package(GTest REQUIRED)`。
仅修改 `LD_LIBRARY_PATH` 也不会凭空产生这些缺失文件。

同样，仓库其他位置有 GoogleTest 源码不代表它会被自动使用。
`mma_dte_tile_tensor/testcase/CMakeLists.txt:178 - mma_dte_add_testcase_project`
设置的是 `DOWNLOAD_COMMAND ""`；当前调用链没有下载、构建或安装 GTest 的步骤。

### 6.4 独立的环境问题：日志使用的 SDK 不是 2609020046

本次观察到三个不同的版本标识，必须分开：

| 来源 | SDK |
| --- | --- |
| 问题描述中指定的目录 | `2609020046` |
| 本次失败日志实际使用的目录 | **`2609151958`** |
| 2026-09-16 本次检查时 `release/latest` 指向的目录 | `2609161039` |

日志证据为 `mma_dte_tile_tensor/cmake.log:33` 的 include 路径；另有：

- `mma_dte_tile_tensor/build/CMakeCache.txt:264`：`scc_DIR` 指向 `2609151958/share/cmake/scc`。
- `mma_dte_tile_tensor/build/mma_dte_build_manifest.txt:2`：`SI_SDK_ROOT=.../2609151958`。
- `mma_dte_tile_tensor/cmake.log:9`：testcase variant 也是 `2609151958/sipu_150/default/specialized`。

原因是用户执行的**根仓库** setup 脚本会加载 `latest`：

```text
setup.sh:63 - 顶层脚本
  -> source set_sdk.sh
     set_sdk.sh:23 - 顶层脚本
       _sikernel_sdk_setup=/share_data/sicx_sdk/release/latest/sipu_sdk_setup.sh
     set_sdk.sh:67 / 98 - _sikernel_source_sdk
       -> source 上述 SDK setup
```

SDK 自身的 `sipu_sdk_setup.sh:12-18 - sipu_sdk_get_loc` 使用 `readlink -f`
解析实际目录并设置 `SI_SDK_ROOT`。因此，即使之前设置过 `SI_SDK_ROOT=.../2609020046`，
再 source 根仓库的 `setup.sh` 也不能据此保证仍在用这个固定版本。

**这不是本次 GTest 缺失的直接原因，但会使复现时发生工具链漂移。**
本文最小验证显式固定在原日志的 `2609151958`，没有用更新的 SDK 来替代它。

注意两个 setup 的参数不一样：

- `setup.sh:34-54 - 顶层脚本`：根仓库版本只接受架构参数 `150|160|170`。
- `mma_dte_tile_tensor/setup.sh:63-102 - 顶层脚本`：子模块版本支持显式 SDK 路径和架构。

因此，在新的 shell 中复现原日志，应使用：

```bash
source /share_data/users/like/miniconda3/bin/activate \
  /share_data/users/like/miniconda3/envs/siinfer_dev
cd /share/users/like/package/sikernel/mma_dte_tile_tensor
source ./setup.sh /share_data/sicx_sdk/release/2609151958 150
```

若确实需要测试 `2609020046`，将子模块 setup 的参数改为那个版本，
并使用新的 `cmake -B` 构建目录核验，避免原有 `scc_DIR` cache 继续指向旧目录。
`build.sh:96 - build_library` 的主库目录始终是 `build`，不会自动按 SDK 隔离；
不要误以为切换了环境变量就一定清除了主库的历史配置。

### 6.5 最小实测：只补 GTest 即可越过当前阻塞

没有修改 Conda 环境、SDK 或业务源码，也没有覆盖原始 `cmake.log`。
全部诊断产物放在：

```text
/tmp/codex-mma-dte-gtest-20260916.qvy8IG
```

测试条件：SDK `2609151958`，SIPU150，Conda 环境 `siinfer_dev`，
host compiler GNU 13.3.0，CMake 4.0.3。

| 实验 | 结果 | 诊断日志，均相对上述临时目录 |
| --- | --- | --- |
| 不补 GTest，单独配置 `test_validation_host` | 退出码 1，同样缺少三个 GTest 项 | `baseline.log` |
| 用仓库已有 GoogleTest 1.16.0 源码离线构建并安装到临时 prefix | 成功生成 `libgtest.a`、`libgtest_main.a` 和 config package | `gtest-build.log`、`gtest-install.log` |
| 只向 `CMAKE_PREFIX_PATH` 添加该 prefix，重新配置同一个 validation build 目录 | `Found GTest ... version "1.16.0"`，配置成功 | `validation-configure.log` |
| 构建并运行 host validation | 构建成功，**11/11 测试通过** | `validation-build.log`、`validation-ctest.log` |
| 使用相同依赖路径，单独配置 `splitK/test_bf16_fp32_host` | 找到 SCC 和 GTest，configure/generate 成功 | `splitk-configure.log` |

离线源码来源为
`sideepgemm/third_party/googletest/CMakeLists.txt:7 - 顶层配置`，其中版本明确为 `1.16.0`。
构建它时只使用该目录作为 source，所有 build/install 输出都在 `/tmp`。

这个对照验证足以确认当前 GTest 阻塞的原因和解决方向。
**没有重跑完整的 74 个测试项目，也没有编译/执行 SplitK device kernel；
不能把上述结果扩展成“整个 --all 构建和所有设备测试已经通过”。**
补齐依赖后，后续阶段是否还有其他问题，应根据新的完整构建日志再判断。

### 6.6 可直接采用的处理方法

#### 方法 A：离线构建一份独立 GTest，避免改动 Conda 环境

下面使用本机已经存在且本次实测过的源码，无需下载。
安装目录改用持久目录，不依赖 `/tmp` 的保留时间。
请在前述固定 SDK/Conda 的环境中执行：

```bash
GTEST_SRC=/share/users/like/package/sikernel/sideepgemm/third_party/googletest
GTEST_PREFIX="$HOME/.local/mma-dte-deps/gtest-1.16.0-siinfer"
GTEST_BUILD="$HOME/.cache/mma-dte-deps/gtest-1.16.0-siinfer"

cmake -S "$GTEST_SRC" -B "$GTEST_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_GMOCK=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_INSTALL_PREFIX="$GTEST_PREFIX"
cmake --build "$GTEST_BUILD" -j2
cmake --install "$GTEST_BUILD"

export CMAKE_PREFIX_PATH="$GTEST_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

cd /share/users/like/package/sikernel/mma_dte_tile_tensor
BUILD_JOBS=4 bash ./build.sh --test > cmake.gtest-fixed.log 2>&1
```

这里选择 `--test` 是因为原日志主库阶段已经成功，先检查测试构建即可；
需要连主库一起构建时再使用 `--all`。`BUILD_JOBS=4` 是便于观察的并行设置，
**不是修复 GTest 缺失所必需的条件**。

只补 GTest、保持同一 SDK 时，通常不必删除 build 目录：本次实测就是在失败过的
validation build 目录内重新配置成功的。新的日志另存，保留原始错误证据。

#### 方法 B：把 GTest 安装到 siinfer_dev

也可以通过环境所使用的 Conda channel 安装 `gtest`，随后确认这些文件存在：

```bash
conda install -p /share_data/users/like/miniconda3/envs/siinfer_dev \
  -c conda-forge gtest

ls "$CONDA_PREFIX/include/gtest/gtest.h"
ls "$CONDA_PREFIX/lib/cmake/GTest/GTestConfig.cmake"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
```

这是可选的环境修改方案，**本次没有执行 Conda 安装**；实际安装前应检查 solver
给出的依赖变更，避免顺带改变当前 host 工具链。方法 A 已完成对应离线验证。

#### 不要只把 -DGTest_DIR 传给最外层测试工程

这里有一个与 `--all` 构建结构有关的陷阱：

`mma_dte_tile_tensor/testcase/CMakeLists.txt:90-112 - 顶层配置`
列出了转发到各个 ExternalProject 的 cache 变量，但其中没有
`GTest_DIR`、`GTEST_ROOT` 或 `CMAKE_PREFIX_PATH`。
第 150-152 行又显式构造了每个子项目的独立 CMake 命令。

因此：

- 对某个具体测试的 `cmake -S testcase/func_test/... -B ... -DGTest_DIR=...` 有效，
  不代表相同 `-D` 只传给外层 `testcase` 工程也会自动传给所有子项目。
- 推荐如方法 A 那样 **export 环境变量 `CMAKE_PREFIX_PATH`**，让 make 启动的
  子 CMake 进程共同继承依赖搜索路径，同时保留原 SDK prefix。
- `mma_dte_tile_tensor/build.sh:35-50 - 顶层脚本` 只解析 `--test/--all/--help`，
  不能写成 `bash build.sh --all -DGTest_DIR=...`。
- `BUILD_TESTING=OFF` 也不是本脚本的跳过测试接口；只构建主库应直接
  `bash build.sh`，见 `mma_dte_tile_tensor/build.sh:112-114 - 顶层脚本`。

长期改进可以在测试聚合工程开始时提前检查 GTest，并统一传递其 package 路径；
同时在 README/环境定义中注明测试构建依赖，避免先完成主库构建才发现测试依赖未安装。
这些属于建议，本次按“分析并追加答案”的要求没有修改构建脚本。

**最终结论：先补齐并让测试子项目找到 host GTest 开发包，同时固定 SDK 版本。
这份日志本身不支持将失败归因于 SIPU ISA、LLVM intrinsic 或 MMA device kernel。**
