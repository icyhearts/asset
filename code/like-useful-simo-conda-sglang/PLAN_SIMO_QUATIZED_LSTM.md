# CUDA SimoQuantizeLSTM 实现计划

## Summary

- 实现 `com.simo::SimoQuantizeLSTM` v2，注册到 `CUDAExecutionProvider`，不修改 ONNX Runtime 源码。
- 实施第一步将本计划写入 `like-useful/PLAN_SIMO_QUATIZED_LSTM.md`，之后再修改代码。
- 优先使用 Lite Custom Op；当前接口可以设计成固定输入/输出，因此采用 `Ort::Custom::CreateLiteCustomOp`。只有 ORT 1.27 的实际注册或执行测试证明不可行时，才改用 status-enabled `CustomOpBase`。
- 仅当标准 LSTM 的 W、R 都是 initializer 或直接 `Constant` 时进行整节点替换，否则保持原节点不变。

## Operator Interface

- Custom Op 使用固定输入：
  `X, W_q, R_q, W_scale, R_scale, B_arg, initial_h_arg, initial_c_arg`。
- W/R 及 scale 为 `uint8` carrier；其余输入和三个输出使用模型计算类型 T。
- 原 LSTM 缺少 B、initial_h 或 initial_c 时，量化器插入一个 T 类型零占位 initializer，并设置 `has_bias`、`has_initial_h`、`has_initial_c` 属性；Compute 根据属性忽略占位并在 CUDA 上创建正确形状的零状态。
- Custom Op 始终声明并生成 `Y, Y_h, Y_c` 三个输出。原 LSTM 缺少某个输出时使用唯一的内部名称承接，使 Lite Custom Op 始终保持固定输出 arity，但不把额外结果暴露为模型输出。
- Lite Compute 签名使用 `OrtKernelContext&`、固定 typed tensor 参数和三个 typed output tensor；为 FP32、FP16、BF16 分别注册同名 kernel。
- 增加 shape inference，根据 X、direction、hidden_size 推导 layout=0 的三个输出形状。
- 沿用 `direction`、`hidden_size`、`layout=0`，并增加 input/weight QDQ 属性、占位存在标志及 W/R carrier 恢复元数据。

## LSTM Semantics

- 支持 `forward`、`reverse`、`bidirectional`、bias、initial_h、initial_c。
- 按已确认的基础默认语义，以下情况整节点跳过：非空 sequence_lens、peephole P、layout != 0、clip、input_forget != 0、非默认 activation。显式默认 `Sigmoid/Tanh/Tanh` 可接受。
- W/R 每个 direction 按 ONNX `iofc` 顺序拆成四个 gate；单向共四块，双向共八块。
- 每个 gate 在进入时间循环前独立调用现有 Triton DQ kernel，恢复为浮点 `[hidden_size, K]`，并在所有时间步复用。
- 每个时间步：
  - `Xt` 使用 module config 的 `input` quantize spec 完成 Triton QDQ，一次结果供四个 W gate 使用。
  - time 0：存在 initial_h/initial_c 时，分别使用同一个 `input` spec 完成 QDQ；不存在时创建零状态。
  - time >= 1：`Ht-1`、`Ct-1` 分别使用同一个 `input` spec 完成 QDQ。
  - 四个 gate 分别执行 `QDQ(Xt) @ DQ(W_gate)^T` 和 `QDQ(Ht-1) @ DQ(R_gate)^T`，两个浮点 MatMul 结果再与对应 bias 相加。
  - `Ct = ft * QDQ(Ct-1) + it * ct`；`Ht = ot * tanh(Ct)`。
- 新产生的 Ht、Ct 和输出保持浮点；它们只在下一时间步作为历史状态使用时进行 QDQ。
- Sigmoid、Tanh、MatMul、加法和逐元素乘法直接使用 Torch/ATen CUDA 算子，并绑定到 ORT 提供的 CUDA stream。

## Quantization Changes

- 将 `LSTM` 加入目标校验，但使用独立的 prepare/replace 路径，不复用标准 MatMul/Gemm/Conv 的前置 QDQ 插入流程。
- LSTM 配置必须同时提供 `input` 和 `weight` spec；W、R 共用 weight spec，X、initial_h、Ht、initial_c、Ct 共用 input spec。
- 支持当前 SIMO 已接受的全部量化规格：MX、per-tensor、per-channel、per-group、per-block，以及 int4 weight。
- 重构权重量化 helper，使逻辑二维 `[output_channel, K]` gate 可直接量化，同时保持现有 MatMul/Gemm/Conv 行为和名称兼容。
- 每个 gate 的 carrier 增加 leading gate 维后合并：
  - W/R：`[num_directions*4, q_rows, q_cols]`
  - W_scale/R_scale：`[num_directions*4, scale_rows, scale_cols]`
- 保存每类权重的 source shape、canonical DQ shape、padding、restore shape 和转置信息；运行时逐 gate 恢复。
- 动态 activation QDQ 在 C++ 中复现当前二维 canonicalization：move-axis、reshape、MX padding、Q/DQ、unpadding和布局恢复。
- 校验 direction、hidden_size、计算 dtype、W/R rank、方向数、`4*hidden_size`、W input_size 和 R hidden_size。非法常量权重给出明确错误，不生成部分量化模型。
- 保留 exclude、嵌套子图递归、直接 Constant、唯一命名、无插入时模型字节保持及 skip reason 统计。

## Plugin Build

- 在 Custom Op 动态库构建中加入 Torch/ATen include、library、C++ ABI 配置以及指向同一 Python 环境 `torch/lib` 的相对 RUNPATH。
- 使用 `KernelContext_GetGPUComputeStream` 获取 ORT stream，通过 `getStreamFromExternal` 和 `CUDAStreamGuard` 运行 Torch 与 Triton。
- 将 ORT CUDA 输入输出包装成无所有权 Torch tensor；临时 Q/DQ carrier、DQ weight 和循环状态由 Torch CUDA allocator 管理。
- 不做 CPU staging，不强制同步；所有工作在同一 ORT CUDA stream 上按序执行。

## Test Plan

- 在 `test_qdq_utils.py` 增加结构测试：单向、反向、双向拆分；4/8 gate carrier；W/R scale；initializer/直接 Constant；占位输入；固定三个内部输出；嵌套子图和唯一命名。
- 验证 W/R 使用同一 weight spec，X/H/C 使用同一 input spec，并检查所有 input/weight QDQ 属性和 carrier 恢复元数据。
- 覆盖跳过条件：W 或 R 动态、sequence_lens、P、layout=1、clip、input_forget、自定义 activation；确认原 LSTM 没有被部分修改。
- 覆盖错误条件：缺少 input spec、非法 direction/hidden_size、W/R rank、方向数、gate 维、R hidden K 和计算 dtype 不一致。
- 参数化覆盖当前每种有效 quantization layout，并为 MX、FP8 各粒度、INT8 各粒度和 INT4 weight 添加代表性 CUDA runtime smoke case。
- 在 `test_dynamic_qdq_runtime_debug.py` 增加 FP32 单向、反向和双向测试；核心单向/双向使用 `mxint8 axis=1 block_size=32`，包含 bias 和非零 initial_h/initial_c。
- 使用禁用 CPU fallback 的 ONNX Runtime CUDA LSTM 作为浮点基准；比较 `Y/Y_h/Y_c` 的 shape、dtype、有限值、余弦相似度和 relative L2 error。
- 增加逐时间步 Torch/SIMO QDQ reference，验证 gate 顺序、W/R 独立 DQ，以及 H/C 使用 input spec 的 QDQ 时机。
- 验证 Lite Custom Op 固定占位接口、固定三输出和动态 shape inference；只有这些测试证明 Lite 机制不可用时，才将注册层切换为 `CustomOpBase`，不改变模型接口和计算逻辑。
- 使用指定环境执行：
  `python -m pip uninstall -y simo`；
  `python -m pip install -e ".[dev]" --no-build-isolation`；
  相关静态及 CUDA pytest；
  ONNX checker、Ruff、`py_compile` 和动态库依赖/RUNPATH 检查。

## Assumptions

- 算子名称严格使用 `SimoQuantizeLSTM`，domain/opset 使用现有 `com.simo` v2。
- 不实现 sequence_lens masking、peephole、layout=1、custom activation、clip 或 coupled input-forget。
- 不引入整数 GEMM；Triton 仅负责 SIMO Q/DQ，LSTM 数学计算使用 Torch CUDA 浮点算子。
