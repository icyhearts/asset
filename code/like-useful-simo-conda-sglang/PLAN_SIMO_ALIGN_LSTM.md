# SimoQuantizeLSTM 与 DynamicQuantizeLSTM 语义对齐

## Summary

- 第一项操作是将本计划写入 `like-useful/PLAN_SIMO_ALIGN_LSTM.md`，之后才修改代码。
- 对齐当前已支持的 LSTM 子集；继续跳过 peephole P、sequence_lens、clip、input_forget、custom activation 和 layout=1。
- 对齐“哪些操作参与量化”：仅 W/R 以及 GEMM 的 X/H 操作数使用 SIMO QDQ；bias、sigmoid、tanh、逐元素计算、C 状态和输出保持浮点。
- 不要求与 ORT 数值逐位一致：ORT 使用动态 asymmetric UINT8 GEMM，SIMO 继续使用配置指定的 QDQ 类型和粒度。

## Implementation Changes

- 修改 CUDA kernel：
  - 保留 W/R 的离线 QDQ/DQ。
  - 保留每个时间步 `X_t` 和 `H_{t-1}` 的 input-spec QDQ，包括非零 `initial_h`。
  - 删除 `initial_c` 和后续 `C_{t-1}` 的 QDQ。
  - 使用浮点公式 `C_t = f_t * C_{t-1} + i_t * c_t`。
  - sigmoid、tanh、bias、gate 加法/乘法、`H_t = o_t * tanh(C_t)`、Y/Y_h/Y_c 均不增加 QDQ。
- 不改变 `com.simo::SimoQuantizeLSTM` v2 的八输入、三输出、属性或模型转换接口。
- 更新 CUDA 单元测试中的逐步参考实现，使 C 始终保持浮点，并验证 custom op 与新参考结果一致。
- 更新 Silero 对比脚本：
  - 删除旧 `simo_torch_full` C-QDQ 语义。
  - 将原 `simo_without_c_qdq` 变为唯一的 SIMO Torch 参考。
  - 保持 float、Dynamic、SIMO 各自独立回灌自身 `Y_h/Y_c` 的闭环 rollout。
  - 保留逐步误差、最终步误差、权重反量化和 custom-op/reference 一致性检查。
- 测试完成后将新结果、与旧结果的变化及原因分析追加到 `like-useful/answer_codex_v2.md`，不覆盖已有内容。

## Verification

- 使用指定环境依次执行：
  - `python -m pip uninstall -y simo`
  - `python -m pip install -e ".[dev]" --no-build-isolation`
- 运行 LSTM 相关静态和 CUDA pytest，至少覆盖 forward、reverse、bidirectional、非零 initial_h/initial_c、固定三输出、各量化 layout 和 FP32/FP16/BF16。
- 新增或强化回归断言：
  - custom op 与不量化 C 的逐步 Torch/SIMO 参考最大绝对误差不超过 `1e-6`。
  - 构造容易暴露 C-QDQ 差异的非零 cell state，确保 recurrence 使用原始浮点 C。
  - 所有输出 shape、dtype 和有限值保持正确。
- 重新运行：
  `like-useful/compare_silero_vad_DynamicQuantizeLSTM_vs_SimoQuantizeLSTM.py --overwrite`
- 对全部真实 Silero LSTM 执行独立样本和多组 256 步闭环 rollout；记录 cosine、relative L2、最终步误差以及 SIMO 与 DynamicQuantizeLSTM 谁更接近 float baseline。
- 最后执行 `py_compile`、Ruff、ONNX checker、`git diff --check`，并检查新构建 custom-op 动态库可被指定环境加载。

## Assumptions

- “语义对齐”指量化边界对齐，而非量化参数、整数 GEMM或舍入结果与 ORT 完全一致。
- SIMO 的 input/weight spec、carrier 格式、gate IOFC 顺序和现有 ABI 保持不变。
- 本次不修改 ONNX Runtime 或 ONNX 源码，不扩展当前明确跳过的 LSTM 功能。
- 闭环精度不设定必须与 ORT 完全相同的阈值；硬性验收是实现与新参考严格一致、测试通过，并且完整记录实际精度变化。
