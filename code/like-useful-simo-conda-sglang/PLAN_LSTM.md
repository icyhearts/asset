# ONNX LSTM SIMO QDQ 量化实现

## Summary

扩展 `apply_qdq_quantization()` 支持标准 `ai.onnx::LSTM`。只量化 X、W、R，保持其余可选输入不变。

## Key Changes

- 将 `LSTM` 加入支持的量化目标和配置校验，继续使用现有 `input` spec 处理 X，使用同一个 `weight` spec 处理 W、R。
- X 按 MatMul 的逻辑二维布局 `[tokens, input_size]` 处理：对 `[seq_length, batch_size, input_size]` 插入必要的 `Shape/Flatten/Pad/Quantize/Dequantize/Slice/Reshape`，最终恢复原始 LSTM 输入形状。
- 要求 W、R 都是 initializer 或直接 `Constant`；任一是动态算子输出时，整个 LSTM 按 `skipped:dynamic_weight` 跳过，保持原图不变。
- 校验 `direction`、`hidden_size` 以及 W/R 的三维形状。`forward`、`reverse` 按单向处理，`bidirectional` 按双向处理。
- 离线将 W、R 分别按 direction 和 ONNX `iofc` gate 顺序拆成二维 `[hidden_size, K]`：
  - 单向：W 4 块、R 4 块。
  - 双向：W 8 块、R 8 块。
- 每块独立调用现有 SIMO 权重量化 kernel。LSTM 权重保持 `[output_channel, K]` 逻辑布局，配置中的 `axis=1` 对应每个 gate 矩阵的 K 维。
- 每块在图中保存独立 uint8 weight/scale carrier，并插入 SIMO `Dequantize*`。按 `iofc` 在 axis 0 拼回每个 direction，再增加 direction 维并拼回 `[num_directions, 4*hidden_size, K]`，分别替换 LSTM 的 W、R。
- 不量化 B、sequence_lens、initial_h、initial_c、P；保留 LSTM 属性、输出和嵌套子图递归处理行为。
- 重构准备结果和权重插入 helper，使 MatMul/Gemm/Conv 现有行为与生成名称保持兼容。

## Test Plan

- 在 `simo/onnx/tests/test_qdq_utils.py` 增加结构单测：
  - 单向与双向 LSTM 的拆分数量、carrier 数量、DQ/Concat/Unsqueeze 结构和最终 W/R 形状。
  - W、R 使用同一 weight spec，X 使用 input spec，可选输入保持不变。
  - initializer 与 Constant 权重、嵌套子图、唯一命名。
  - W 或 R 任一动态时整节点跳过。
  - 非法 direction、W/R rank、方向数、gate/hidden_size 维度不一致时给出明确错误。
- 在 `simo/onnx/tests/test_dynamic_qdq_runtime_debug.py` 追加 CUDA runtime 测试：
  - 单向 LSTM：`mxint8` X + `mxint8 axis=1 block_size=32` W/R。
  - 双向 LSTM：相同配置，覆盖 8 个 W gate 和 8 个 R gate。
  - 使用动态 sequence/batch 维度，验证三个 LSTM 输出的形状、dtype、有限值，并与使用相同 SIMO QDQ 参考结果比较数值误差。
  - 检查 X 的补齐和恢复路径，以及所有 gate weight carrier 均被实际消费。
- 使用指定 conda Python 运行相关静态单测和两个 LSTM runtime case，并执行 ONNX checker、Ruff 和 `py_compile`。

## Assumptions

- ONNX 输入编号按文档的一基表述：X/W/R 分别对应 `node.input[0:3]`。
- 不引入 `DynamicQuantizeLSTM`，最终仍是标准 ONNX `LSTM`，前面连接显式 SIMO QDQ 子图。
- 不修改 ONNX 或 ONNX Runtime 源码，也不自动对动态 W/R 做常量折叠。
- W/R 部分动态时采用已确认的“整节点跳过”策略。
