## 49. `quantize_dynamic` 后的 LSTM 能否由 ONNX Runtime CUDA 执行

### 49.1 修改内容

已修改 `like-useful/test-onnx-dynamic-quant-lstm.py`，脚本现在按以下顺序执行：

1. `like-useful/test-onnx-dynamic-quant-lstm.py:22-42` 的 `quantize_model()` 调用
   `quantize_dynamic()`，再用 ONNX checker 检查量化模型；
2. `like-useful/test-onnx-dynamic-quant-lstm.py:181-189` 的 `main()` 显式创建
   `CPUExecutionProvider` session，先完成 CPU 推理；
3. `like-useful/test-onnx-dynamic-quant-lstm.py:116-178` 的 `try_cuda()` 再创建
   CUDA session，运行相同输入并采集 ONNX Runtime profiling；
4. 最后创建禁止 CPU fallback 的严格 CUDA session，判断核心量化算子是否真的能够在
   CUDA 上执行。

输入由 `like-useful/test-onnx-dynamic-quant-lstm.py:45-58` 的
`make_input_cases()` 生成，覆盖三组动态 shape：

```text
seq_len=5, batch=3
seq_len=7, batch=2
seq_len=3, batch=4
```

`like-useful/test-onnx-dynamic-quant-lstm.py:61-87` 的 `run_cases()` 对每组输入检查
`output/hn/cn` 的 shape，并检查结果中不存在 NaN/Inf。CPU 和 CUDA 混合 session 的结果还会由
`like-useful/test-onnx-dynamic-quant-lstm.py:90-101` 的 `compare_outputs()` 逐个比较。

### 49.2 量化后不再是标准 ONNX `LSTM`

实测量化模型 `temp/test-lstm-quant.onnx` 的节点为：

```text
ai.onnx::Constant
com.microsoft::DynamicQuantizeLSTM
ai.onnx::Squeeze
```

也就是说，`quantize_dynamic()` 把原来的 `ai.onnx::LSTM` 替换成了 ONNX Runtime contrib
算子 `com.microsoft::DynamicQuantizeLSTM`。`like-useful/test-onnx-dynamic-quant-lstm.py:37-40`
也显式检查了这个图结构，避免误以为测试的仍是浮点 `LSTM`。

### 49.3 CPU 运行结果

指定环境中的 ONNX Runtime 版本为 `onnxruntime-gpu 1.27.0`。显式使用
`CPUExecutionProvider` 后，三组动态 shape 均成功运行：

```text
CPU_session_providers=['CPUExecutionProvider']
CPU: seq_len=5 batch=3 output_shapes=[(5, 3, 20), (1, 3, 20), (1, 3, 20)]
CPU: seq_len=7 batch=2 output_shapes=[(7, 2, 20), (1, 2, 20), (1, 2, 20)]
CPU: seq_len=3 batch=4 output_shapes=[(3, 4, 20), (1, 4, 20), (1, 4, 20)]
PASS: CPU verified 3 dynamic LSTM input shapes
```

因此量化模型本身有效，动态 `sequence length` 和 `batch` 也不是本次 CUDA 失败的原因。

### 49.4 第一个 CUDA 环境问题：cuDNN 不在默认动态库搜索路径

如果直接创建 CUDA session，当前环境会报告：

```text
Failed to load library .../libonnxruntime_providers_cuda.so with error:
libcudnn.so.9: cannot open shared object file: No such file or directory
```

这并不表示 conda 环境没有安装 cuDNN。实际文件位于：

```text
<conda-env>/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9
```

只是该目录不在进程默认的动态库搜索路径中。脚本在 CPU 验证之后，于
`like-useful/test-onnx-dynamic-quant-lstm.py:123-128` 的 `try_cuda()` 中调用：

```python
ort.preload_dlls(directory="")
```

这是 ONNX Runtime 提供的预加载接口，会从 pip 安装的 `nvidia` site-packages 中加载匹配的
CUDA/cuDNN 动态库。调用后 CUDA EP 能够正常创建，session 的 provider 为：

```text
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

也可以在启动 Python 前设置对应的 `LD_LIBRARY_PATH`，但只解决动态库加载问题，不解决下一节的
算子支持问题。

### 49.5 CUDA session 能运行不等于量化 LSTM 在 CUDA 上运行

`providers=['CUDAExecutionProvider', 'CPUExecutionProvider']` 允许 CPU fallback。只检查
`session.run()` 成功会产生误判：CUDA EP 不支持的节点会自动交给 CPU EP。

`like-useful/test-onnx-dynamic-quant-lstm.py:130-148` 的 `try_cuda()` 开启 profiling，
`like-useful/test-onnx-dynamic-quant-lstm.py:104-113` 的 `read_profile_assignments()` 读取每个
kernel 的实际 provider。实测结果是：

```text
CUDA_profile_assignments=[
  ('DynamicQuantizeLSTM', 'CPUExecutionProvider'),
  ('MemcpyFromHost', 'CUDAExecutionProvider'),
  ('Squeeze', 'CUDAExecutionProvider')
]
```

因此：

- CUDA EP 已经成功加载；
- 图尾部的 `Squeeze` 确实在 CUDA 上运行；
- 核心 `com.microsoft::DynamicQuantizeLSTM` 仍然在 CPU 上运行；
- CPU 与这个混合 session 的结果完全一致，实测所有输出的 `max_abs_diff` 都是 0。

### 49.6 严格 CUDA 验证为什么失败

`like-useful/test-onnx-dynamic-quant-lstm.py:163-176` 的 `try_cuda()` 使用：

```python
strict_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
providers=["CUDAExecutionProvider"]
```

禁止 CPU fallback 后，session 初始化立即失败：

```text
This session contains graph nodes that are assigned to the default CPU EP,
but fallback to CPU EP has been explicitly disabled by the user.
```

这和 profiling 的结论相互印证：当前 `onnxruntime-gpu 1.27.0` 构建没有可用于该模型中
`com.microsoft::DynamicQuantizeLSTM` 的 CUDA EP kernel。它是 CPU 实现的 contrib 算子；仅仅把
`CUDAExecutionProvider` 放在 provider 列表首位不会把它转换成 CUDA 实现。

### 49.7 如何处理

当前模型不能作为“全 CUDA 动态量化 LSTM”运行。可行选择是：

1. 需要使用 `quantize_dynamic()` 生成的 int8 LSTM 时，明确使用 CPU EP；这是当前模型的实际
   支持路径。
2. 必须在 GPU 上执行时，使用未被改写成 `DynamicQuantizeLSTM` 的浮点 FP32/FP16 LSTM，并单独
   验证所有节点的 CUDA provider 分配。
3. 必须同时满足 int8 LSTM 和 CUDA 时，需要换用具有对应 GPU kernel 的推理后端/图表示，或者
   实现并注册支持相同语义的 CUDA custom op；安装 cuDNN 或调整 `LD_LIBRARY_PATH` 本身不够。

验证命令：

```bash
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python \
  like-useful/test-onnx-dynamic-quant-lstm.py \
  > temp/test-onnx-dynamic-quant-lstm.log 2>&1
```

脚本退出码为 0；CPU 与允许 fallback 的混合 session 均完成三组输入验证，日志最终明确报告
`CUDA-only inference is unavailable`。这里保留退出码 0 是因为脚本已经完成预期诊断，并把
当前 ORT 不支持严格 CUDA 执行作为检测结果输出，而不是把 CPU 成功路径误报为测试失败。

## 50. ONNX Runtime `DynamicQuantizeLSTM` 是真 int8 计算还是 QDQ 模拟

本节中 ONNX Runtime 路径均相对于 code base：

```text
/softhome/like/package/onnxruntime
```

SIMO 路径均相对于 code base：

```text
/share/users/like/package/simo_conda_sglang
```

### 50.1 直接结论

1. **是的，`DynamicQuantizeLSTM` 的 CPU kernel 入口实现在
   `onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc`。**
2. 但完整实现不只在这一个文件中：该文件负责 operator、量化权重 prepack、量化参数读取和
   kernel 注册；LSTM 循环在 `onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc`，真正的
   动态量化与量化 GEMM 在 `onnxruntime/core/providers/cpu/rnn/rnn_helpers.cc`，底层使用 MLAS。
3. 它不是 SIMO 当前 MatMul 那种显式 `Quantize -> Dequantize -> MatMul` 图结构，而是
   **融合型量化算子**：图中只有一个 `com.microsoft::DynamicQuantizeLSTM`，量化、int8 GEMM 和
   反量化都封装在它的 CPU kernel 内部。
4. 它的两个主要矩阵乘确实使用 8-bit 整数输入做 GEMM。对当前
   `weight_type=QInt8` 模型，准确类型是 **UINT8 activation x INT8 weight，INT32 accumulate**，
   随后乘 scale 写回 FP32。通常可以简称为 int8 GEMM，但并不是严格的 INT8 x INT8。
5. 它不是端到端全 int8 LSTM：bias、gate 激活、cell state、hidden state以及 operator 输入/输出
   都是 FP32。

### 50.2 为什么该算子不属于标准 ONNX

在 ONNX code base `/softhome/like/package/onnx` 中搜索不到 `DynamicQuantizeLSTM`。它的 domain 是
`com.microsoft`，属于 ONNX Runtime contrib op，而不是 ONNX 标准算子。

schema 位于
`onnxruntime/core/graph/contrib_ops/quantization_defs.cc:657-758` 的
`ONNX_MS_OPERATOR_SET_SCHEMA(DynamicQuantizeLSTM, 1, ...)`：

- `onnxruntime/core/graph/contrib_ops/quantization_defs.cc:688-691` 定义 `X` 为输入 sequence；
- `onnxruntime/core/graph/contrib_ops/quantization_defs.cc:692-701` 定义量化权重 `W`、`R`；
- `onnxruntime/core/graph/contrib_ops/quantization_defs.cc:727-742` 定义 `W/R` 的 scale 和
  zero-point；
- `onnxruntime/core/graph/contrib_ops/quantization_defs.cc:755-757` 明确规定 `X` 和输出为
  `tensor(float)`，`W/R` 为 `tensor(uint8)` 或 `tensor(int8)`。

所以从 operator ABI 就可以看出：

```text
外部 X、H、C、Y：FP32
静态 W、R：INT8/UINT8
activation 的动态量化：发生在 kernel 内部
```

### 50.3 `quantize_dynamic()` 如何生成该节点

`onnxruntime/python/tools/quantization/quantize.py:875-977` 的 `quantize_dynamic()` 做了两项关键
配置：

- `onnxruntime/python/tools/quantization/quantize.py:937-940` 选择
  `QuantizationMode.IntegerOps`，不是 `QDQQuantizer`；
- `onnxruntime/python/tools/quantization/quantize.py:961-974` 创建 `ONNXQuantizer`，其中第 968 行
  明确写着动态 activation 只支持 `QUInt8`。

`onnxruntime/python/tools/quantization/registry.py:35-38` 把标准 `LSTM` 映射到
`LSTMQuant`。随后
`onnxruntime/python/tools/quantization/operators/lstm.py:17-117` 的
`LSTMQuant.quantize()` 完成改图：

1. `onnxruntime/python/tools/quantization/operators/lstm.py:26-38` 要求 `W`、`R` 是可量化的
   initializer，且均为 rank 3；
2. `onnxruntime/python/tools/quantization/operators/lstm.py:43-58` 对 `W` 和 `R` 做离线
   per-channel INT8 量化；
3. `onnxruntime/python/tools/quantization/operators/lstm.py:63-77` 将量化权重从标准 LSTM 布局
   转成该 contrib kernel 需要的布局；
4. `onnxruntime/python/tools/quantization/operators/lstm.py:84-88` 把 per-channel scale 和
   zero-point 整理成 `[num_directions, 4 * hidden_size]`；
5. `onnxruntime/python/tools/quantization/operators/lstm.py:90-116` 组合 float 输入、int8 权重、
   scale、zero-point，创建 `com.microsoft::DynamicQuantizeLSTM`。

当前 `temp/test-lstm-quant.onnx` 的实际 initializer 也与此一致：

```text
W_quantized: INT8, shape=[1, 10, 80]
R_quantized: INT8, shape=[1, 20, 80]
W_scale:     FLOAT, shape=[1, 80]
R_scale:     FLOAT, shape=[1, 80]
W/R zero point: INT8，全为 0
bias: FLOAT
```

这里 `hidden_size=20`，所以 `4 * hidden_size=80`。`per_channel=True` 因而为 80 个 gate/output
channel 分别保存 weight scale。

量化后的 graph 没有为 LSTM 展开显式 `QuantizeLinear/DequantizeLinear` 节点：

```text
FP32 X, FP32 h0, FP32 c0
            |
            v
com.microsoft::DynamicQuantizeLSTM
            |
            v
FP32 Y, FP32 Y_h, FP32 Y_c
```

这是 operator-oriented/fused dynamic quantization，而不是 graph-level QDQ 表示。

### 50.4 CPU kernel 的完整调用链

#### 50.4.1 kernel 入口和注册

`onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc:11-39` 定义
`DynamicQuantizeLSTM`，继承 `OpKernel` 和 `LSTMBase`。

`onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc:238-248` 使用
`ONNX_OPERATOR_TYPED_KERNEL_EX` 注册 kernel，并明确指定：

```cpp
kCpuExecutionProvider
```

`onnxruntime/contrib_ops/cpu/cpu_contrib_kernels.cc:111` 声明该 CPU kernel，
`onnxruntime/contrib_ops/cpu/cpu_contrib_kernels.cc:279` 将它加入 CPU contrib kernel registry。
在 `onnxruntime/contrib_ops/cuda/` 和 `onnxruntime/core/providers/cuda/` 中没有对应
`DynamicQuantizeLSTM` 注册，这也解释了上一节 profiling 中该节点只能分配给 CPU EP。

#### 50.4.2 权重保持 8-bit，并进行 MLAS prepack

`onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc:41-87` 的
`DynamicQuantizeLSTM::TryPackWeights()`：

- 第 48-51 行读取 8-bit `W/R` 的 `[K, N]`；
- 第 57 行记录 weight 是 signed INT8 还是 UINT8；
- 第 58 行调用 `MlasGemmPackBSize()`；
- 第 78-83 行直接把 8-bit weight 传给 `MlasGemmPackB()`。

这里的 prepack 是为 MLAS integer GEMM 调整布局，不是先把权重反量化成 float。
`onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc:94-115` 的
`DynamicQuantizeLSTM::PrePack()` 分别处理 input weight `W` 和 recurrent weight `R`。

#### 50.4.3 读取量化参数并进入量化版 LSTM

`onnxruntime/contrib_ops/cpu/quantization/dynamic_quantize_lstm.cc:166-235` 的
`DynamicQuantizeLSTM::Compute()`：

- 第 166-178 行取得 int8 `W/R` 以及对应 scale/zero-point；
- 第 190-206 行构造 `QuantizationParameter`；
- 第 208-216 行构造 `GemmWeights<uint8_t>`。这里模板存储类型写成 `uint8_t`，实际 signedness
  由 `is_W_signed/is_R_signed` 单独传递，所以仍可表示 INT8 weight；
- 第 235 行调用 `LSTMBase::ComputeImpl<float, uint8_t>()`。

模板参数已经表达了其混合精度边界：LSTM state/input/output 是 float，GEMM weight 是 8 bit。

`onnxruntime/core/providers/cpu/rnn/lstm_base.cc:22-27` 的 `LSTMBase::ComputeImpl()` 接收
`GemmWeights<WeightT>`；第 146-167 行创建 `UniDirectionalLstm<float>` 并调用其 `Compute()`。

#### 50.4.4 两个矩阵乘都进入量化 GEMM

`onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc:228-457` 的
`UniDirectionalLstm<T>::ComputeImpl()` 执行 LSTM 主循环：

1. **输入投影 `X * W`**：
   `onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc:284-293` 对所有有效 timestep 的
   `X` 调用 `ComputeGemm()`，结果写入 float `output_iofc`。
2. **循环投影 `H[t-1] * R`**：
   `onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc:332-349` 在每个 timestep 再调用
   `ComputeGemm()`，并把结果累加到已有的 `X * W` 结果。
3. **门函数和状态更新**：
   `onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc:463-588` 的
   `UniDirectionalLstm<T>::GateComputations()` 对 FP32 gate buffer 执行 bias、clip、sigmoid、tanh、
   cell state 和 hidden state 更新。

因此被量化的是 LSTM 中计算量最大的两个 affine/GEMM 部分，不是 sigmoid、tanh 和 state 更新。

### 50.5 `ComputeGemm()` 为什么能证明是真正的整数 GEMM

量化 overload 位于 `onnxruntime/core/providers/cpu/rnn/rnn_helpers.cc:247-317` 的
`rnn::detail::ComputeGemm()`：

1. `onnxruntime/core/providers/cpu/rnn/rnn_helpers.cc:271-276` 调用
   `GetQuantizationParameter()` 和 `ParQuantizeLinearStd()`，把本次 float activation `A`
   **实际写入 UINT8 buffer**；
2. `onnxruntime/core/util/qmath.h:50-109` 的 `GetQuantizationParameter()` 根据本次输入的 min/max
   动态计算 `a_scale` 与 `a_zero_point`；
3. `onnxruntime/core/util/qmath.h:122-135` 的 `ParQuantizeLinearStd()` 调用
   `MlasQuantizeLinear()` 生成 8-bit activation；
4. `onnxruntime/core/providers/cpu/rnn/rnn_helpers.cc:281-296` 计算
   `a_scale * weight_scale`，并创建 `MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR`；
5. `onnxruntime/core/providers/cpu/rnn/rnn_helpers.cc:298-316` 把量化后的 A、8-bit B、双方
   zero-point 和 int32 C buffer 交给 `MlasGemm()`。

MLAS 的 ABI 进一步证明累加类型：

- `onnxruntime/core/mlas/inc/mlas.h:613-633` 的 `MLAS_GEMM_QUANT_*_PARAMS` 定义 A/B 的
  signedness，并把 C 定义为 `int32_t*`；
- `onnxruntime/core/mlas/inc/mlas.h:540-582` 的 output processor 接收 `const int32_t* C`；
- `onnxruntime/core/mlas/lib/qpostprocessor.cpp:103-118` 的
  `MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR::ProcessImpl()` 明确说明把 C 转换回 floating point；
- `onnxruntime/core/mlas/lib/qpostprocessor.cpp:161-183` 将 int32 accumulator 转成 float，并乘
  scale；recurrent GEMM 使用 accumulate mode 时，再加到已有的 float `X * W` 结果。

对应的数学过程可简化为：

```text
A_q = quantize_uint8(A_fp32, a_scale, a_zero_point)       # 每次运行动态计算
B_q = quantize_int8(B_fp32, b_scale, b_zero_point)        # 模型转换时已完成

C_int32[m,n] = sum_k(
  (A_q[m,k] - a_zero_point) * (B_q[k,n] - b_zero_point)
)

C_fp32[m,n] = C_int32[m,n] * a_scale * b_scale[n]
```

当前 QInt8 模型的 `b_zero_point=0`；per-channel 变化的是 `b_scale[n]`。输入 `X * W` 的
activation 参数针对整次输入 GEMM 动态计算，循环中的 `H[t-1] * R` 则在每次 recurrent GEMM
调用时根据当时的 hidden-state 数据重新计算。

对当前模型：

```text
X * W:       UINT8 x INT8 -> INT32 -> FP32
H[t-1] * R:  UINT8 x INT8 -> INT32 -> FP32，并累加到 X * W
gate/state:  FP32
```

所以答案是：**矩阵乘核心是真正的量化整数 GEMM，不是先把 W 反量化成 FP32 后再调用浮点
MatMul。** 但它也不是端到端全 int8，正确描述是“动态 activation 量化 + 静态 weight 量化的
混合精度 LSTM”。

### 50.6 与 SIMO MatMul QDQ 的区别

SIMO 当前路径是显式 QDQ graph：

`simo/onnx/onnx_quant.py:608-669` 的 `_insert_qdq_in_graph()` 在找到 MatMul target 后：

- 第 636-642 行创建 activation QDQ；
- 第 643 行创建 weight DQ；
- 第 665 行仍把原始标准 `MatMul` 节点追加回 graph。

activation 路径由 `simo/onnx/onnx_quant.py:949-974` 的 `_create_qdq_nodes()` 创建：

```text
X(float) -> com.simo::Quantize -> packed q/scale
         -> com.simo::Dequantize -> X_dequant(float)
```

weight 路径由 `simo/onnx/onnx_quant.py:977-1031` 的 `_create_weight_dq_nodes()` 创建：量化后的
weight 和 scale 作为 initializer 写入 graph，第 1018-1025 行插入
`com.simo::Dequantize`，然后标准 `MatMul` 使用反量化输出。

custom-op ABI 也明确了数据类型：

- `simo/onnx/ort_plugin/simo_qdq_ops.cc:301-305` 的 `SimoQuantizeOp::Compute()` 接收
  `Tensor<float>`，输出 packed `Tensor<uint8_t>`；
- `simo/onnx/ort_plugin/simo_qdq_ops.cc:404-408` 的 `SimoDequantizeOp::Compute()` 接收 packed
  `uint8`，输出 `Tensor<float>`；
- `simo/onnx/ort_plugin/simo_qdq_ops.cc:504-510` 的 `RegisterQdqOps()` 把 Quantize 和
  Dequantize 注册为 CUDA custom op。

因此 SIMO 当前 MatMul 的数据流是：

```text
X_fp32 -> Quantize -> X_q -> Dequantize -> X_dq_fp32 ---+
                                                        +-> ai.onnx::MatMul -> FP32 output
W_fp32 --offline quantize--> W_q -> Dequantize -> W_dq_fp32 ---+
```

在当前实现中，没有把这组 `com.simo::Quantize/Dequantize + ai.onnx::MatMul` 融合成一个直接消费
packed q/scale 的整数 MatMul kernel。由于标准 MatMul 接收到的两个输入均为 float，它执行的是
**浮点 MatMul**。SIMO QDQ 确实产生量化误差、保存量化 weight，并真实执行 Q/DQ CUDA kernel，
但它当前不是 int8 MatMul 加速路径。

需要强调：**QDQ 是一种图表示，不天然等于“假量化”。** 如果某个 execution provider 能识别
QDQ pattern 并把它融合成量化 kernel，底层同样可以是真正的整数计算。但 SIMO 这里使用
`com.simo` 自定义 Q/DQ，当前代码没有对应的 MatMul fusion，所以不能仅凭 graph 中出现 QDQ 就
声称 MatMul 已经使用 int8 arithmetic。

### 50.7 对照总结

| 项目 | ONNX Runtime `DynamicQuantizeLSTM` | SIMO 当前 MatMul QDQ |
|---|---|---|
| graph 表示 | 单个融合型 `com.microsoft` 算子 | 显式 `Quantize -> Dequantize -> MatMul` |
| weight | 转换期量化为 INT8/UINT8 | 转换期量化并保存 packed q/scale |
| activation | kernel 内运行时动态量化为 UINT8 | 自定义 Quantize 后立即 Dequantize 为 float |
| MatMul/GEMM 输入 | UINT8 activation、INT8 weight（当前模型） | 两个输入均为反量化后的 float |
| 累加 | INT32 | FP32 |
| GEMM 后 | 乘 scale 转回 FP32 | 已直接得到 FP32 |
| gate/state | FP32 | 不适用 |
| 当前 EP | CPU EP | Q/DQ 在 CUDA，标准 MatMul 处理 float |

一句话总结：

```text
DynamicQuantizeLSTM 不是 SIMO 当前的“QDQ 后再做浮点 MatMul”；它把动态量化和两个 MLAS
整数 GEMM 融合在 CPU kernel 内。当前 QInt8 模型的 GEMM 是 U8 x S8 -> S32，再缩放回 FP32；
因此 GEMM 是真量化计算，但整个 LSTM 不是全 int8。
```
