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
