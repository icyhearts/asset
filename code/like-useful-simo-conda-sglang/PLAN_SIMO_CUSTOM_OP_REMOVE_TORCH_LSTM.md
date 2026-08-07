# SimoQuantizeLSTM Custom-Op 运行时去 Torch

## Summary

- 第一项操作是把本计划写入 `like-useful/PLAN_SIMO_CUSTOM_OP_REMOVE_TORCH_LSTM.md`，之后才修改代码。
- 保持 `SimoQuantizeLSTM` v2 的八输入、三输出、属性、QDQ Triton kernel 和模型转换接口不变。
- 仅解除 ONNX custom-op 动态库运行时对 Torch/c10 的依赖；整个 SIMO Python 包和 QDQ 构建链仍可依赖 Torch。
- 本轮不实现四门合并 GEMM，先完成逐门 ORT native-op 版本的正确性基线。

## Implementation Changes

- 在 `SimoQuantizeLstmCustomOp<T>` 构造阶段按 FP32/FP16/BF16 创建并缓存：
  - `Gemm-13`，设置 `transB=1`；
  - `Sigmoid-13`、`Tanh-13`；
  - `Add-14`、`Mul-14`。
- 使用父 custom-op 的 `OrtKernelInfo` 创建 native op，并在每个 time step 调用缓存的 `Ort::Op::Invoke`，不重复创建 kernel。
- 用 `KernelContext_GetAllocator`、CUDA `MemoryInfo` 和 RAII workspace 替换 ATen allocator；通过预分配 `OrtValue` 和 raw shape view 管理输入、输出及临时 buffer。
- 保持 W/R 四门独立反量化及每步八次 GEMM；bias 合并、gate 加法、sigmoid/tanh、浮点 C recurrence 和 H update 使用 ORT native op。
- 新增 FP32/FP16/BF16 CUDA layout kernel，处理二维转置、MX padding、unpadding、连续化和必要的数据拷贝；reshape 只更新 view shape。
- initial H/C、Y/Y_h/Y_c 的连续区域使用当前 ORT CUDA stream 做异步 copy/zero；C 状态始终保持浮点且不进行 QDQ。
- 从 LSTM C++ 实现删除 ATen/c10 include、tensor 操作、stream guard、InferenceMode、ATen 算子和 `c10::Error` catch。
- 修改 `build_sm90_runtime`：
  - 继续使用系统 C++ 编译器完成最终链接；
  - 新增 layout CUDA 源文件的 nvcc 编译；
  - CUDA 根目录唯一由 `os.environ.get("CUDA_HOME", "/usr/local/cuda")` 决定；
  - nvcc 路径严格使用 `<CUDA_HOME>/bin/nvcc`；
  - 如果 `CUDA_HOME` 缺失、头文件不存在或 `<CUDA_HOME>/bin/nvcc` 不可执行，立即报清晰错误；
  - 不读取 `PATH` 中的 nvcc，不添加任何固定 `/share_data/.../cuda-13.0` 路径，不引入固定 CUDA 版本。
- 修改 `runtime.py` 删除强制 `import torch`。
- 更新源码打包清单以包含新增 `.cu/.h` 文件。
- 更新构建测试，验证：
  - `CUDA_HOME=/tmp/cuda` 时只使用 `/tmp/cuda/bin/nvcc`；
  - `PATH` 中存在另一个 nvcc 时仍不会使用它；
  - 构建命令不存在 torch/c10 include、library、ABI define 和 rpath。

## Test Plan

- 安装前明确执行：

```bash
export CUDA_HOME=/share_data/users/like/opt/cuda-13.0
python -m pip uninstall -y simo
python -m pip install -e ".[dev]" --no-build-isolation
```

- 扩展 CUDA LSTM 测试，覆盖 forward、reverse、bidirectional、bias 可选、非零 initial H/C、固定三输出、全部 QDQ layout，以及 FP32/FP16/BF16。
- 保留浮点 C 回归：
  - FP32 单步 reference 最大绝对误差不超过 `1e-4`；
  - 新实现与旧 C-QDQ reference 保持大于 `1e-3` 的可观测差异；
  - FP16/BF16 容差分别为 `5e-3` 和 `5e-2`；
  - 所有输出 shape、dtype 正确且有限。
- 新增独立子进程测试：不导入 SIMO 或 Torch，仅通过 ONNX Runtime、NumPy、模型文件和 `.so` 注册并执行 custom op，并断言 `torch` 不在 `sys.modules`。
- 使用 `readelf -d` 和 `ldd` 验证新 `.so` 不包含 `libtorch*`、`libc10*` 或 Torch rpath。
- 运行静态转换测试、构建测试和全部 LSTM CUDA pytest；在可用的非零 CUDA device 上额外验证 standalone-op 的 stream、allocator 和 device 行为。
- 运行：

```bash
like-useful/compare_silero_vad_DynamicQuantizeLSTM_vs_SimoQuantizeLSTM.py --overwrite
```

保留每个实现独立回灌自身 `Y_h/Y_c` 的 256 步闭环，记录独立样本、三组 rollout、最终步误差、cosine、relative L2，以及 SIMO 和 DynamicQuantizeLSTM 谁更接近 float baseline。
- 最后执行 `py_compile`、Ruff、ONNX checker、动态库加载检查和 `git diff --check`。

## Results And Assumptions

- 将实现说明、实际构建环境、`CUDA_HOME`/nvcc 解析结果、Torch/c10 依赖检查、Silero 新结果及变化原因追加到 `like-useful/answer_codex_v2.md`，不覆盖已有内容；代码说明使用“文件名 + 行号 + 函数名”格式。
- `CUDA_HOME` 是本轮 custom-op layout kernel 和 SIMO CUDA 扩展的唯一 CUDA 工具链入口；执行 pip 安装前必须由调用环境显式设置。
- `CUDA_HOME` 未设置时允许回退到 `/usr/local/cuda`，但不会通过 `PATH` 自动选择 nvcc。
- ORT 1.27 完整构建及公开 standalone-op API 是最低运行条件；minimal ORT 不在本轮支持范围。
- 新增 `libcudart`/CUDA driver 依赖允许存在；禁止的是 Torch/c10 动态库和 Torch rpath。
