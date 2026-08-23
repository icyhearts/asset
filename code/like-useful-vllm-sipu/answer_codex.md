# 1. `pip install -e . --no-build-isolation -vvv` 失败分析

## 1.1 根因

根因不是 `pip` 或 `--no-build-isolation`，而是 **CMake 复用了旧的
`sikernel` 浅克隆缓存，但该缓存中没有当前版本固定的提交**
`ed042c571e949d86a403630f5257ee8ca205a93a`。

## 1.2 因果链

1. `sikernel.version` 把依赖固定为 `ed042c...`，
   `cmake/external_projects/sikernel.cmake` 将它作为 FetchContent 的
   `GIT_TAG`。
2. `.deps/sikernel-src` 是旧版构建在 2026-08-14 按 `GIT_TAG dev` 和
   `GIT_SHALLOW TRUE` 创建的浅仓库；当前代码虽已移除浅克隆并改用固定
   SHA，但旧缓存和 clone stamp 仍在。
3. 日志 1783-1784 行显示 CMake 认为 clone stamp 是最新的，因此跳过
   重新 clone；1785-1791 行只在旧仓库上执行普通 fetch，更新了
   `origin/dev`，却没有补齐固定 SHA 对应的对象。
4. 随后 checkout `ed042c...`，日志 1792 行报出首个实质错误：

   ```text
   fatal: reference is not a tree: ed042c571e949d86a403630f5257ee8ca205a93a
   ```

5. Git 返回 128，导致 `sikernel-populate-update`、Ninja、
   `FetchContent_MakeAvailable(sikernel)` 和 CMake 配置依次失败，最终才被
   pip 包装成 `InstallWheelBuildError`。因此日志末尾的
   `Failed to build vllm_sipu` 是结果，不是最初原因。

## 1.3 处理方法

排查时执行按 SHA 的显式 fetch 已成功，说明该提交在远端仍可访问，SSH
权限和网络也正常；问题是旧浅缓存配合 CMake 的普通更新方式没有取得该
对象。最稳妥的处理与仓库 `README.zh.md` 的建议一致：

```bash
rm -rf .deps
source sipu_sdk_setup.sh
pip install -e . --no-build-isolation -vvv
```

## 1.4 非根因警告

日志中的以下信息不是本次失败根因：

- `kineto_LIBRARY-NOTFOUND` 只是 CMake warning；
- `libtinfo.so.6: no version information available` 只是运行时 warning；
- `Build failed with PCH, retrying without PCH` 后续重试成功，`siinfer` 最终也
  成功构建。

# 2. DeepSeek-V3.2 Tiny 离线推理命令讲解

## 2.1 命令的整体作用

```bash
python3 examples/offline/offline_inference_recipe.py \
  --recipe examples/recipes/deepseek/deepseek_v32_tiny.json
```

这条命令使用统一 recipe runner 读取
`examples/recipes/deepseek/deepseek_v32_tiny.json`，在 vLLM-SIPU 上加载本地的
DeepSeek-V3.2 Tiny FP8 真实权重，对文本 `Hello, my name is` 做一次离线推理，
并按贪心策略最多生成 **1 个新 token**。

它主要是一个模型端到端 smoke test：检查 recipe 解析、SIPU 插件注册、模型与
tokenizer 加载、FP8 权重加载、引擎预热、prefill 和采样路径是否能够贯通。
它不会启动 HTTP/OpenAI API 服务，不监听端口，也不是交互式聊天程序；执行结束后
进程退出。

## 2.2 命令行各部分

| 部分 | 作用 |
| --- | --- |
| `python3` | 使用当前环境中的 Python 3 解释器运行脚本。必须是已经安装 vLLM 和 vLLM-SIPU 的开发环境。 |
| `examples/offline/offline_inference_recipe.py` | 通用离线推理入口，把 JSON recipe 转换成 `vllm.LLM`、`SamplingParams` 和输入。 |
| `--recipe` | 指定 recipe。该参数也支持 `deepseek/deepseek_v32_tiny` 这样的简写。 |
| `examples/recipes/deepseek/deepseek_v32_tiny.json` | 本次运行使用的完整 recipe 路径。由于这个相对路径实际存在，脚本会直接读取它。 |

两个参数之间出现多个空格只起 shell 分隔作用，与单个空格等价。因为脚本和 recipe
都使用相对路径，这条写法应在仓库根目录执行；从其他目录执行时应改用绝对路径。

## 2.3 Recipe 最终展开成什么调用

本命令没有提供任何 CLI 覆盖参数，recipe 也没有 `model_config`，所以不会创建模型
配置 overlay。忽略内部对象表示后，核心调用等价于：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/share_data/vllm-sipu/models/DeepSeek-V3.2-tiny",
    trust_remote_code=True,
    quantization="fp8",
    gpu_memory_utilization=0.01,
    max_model_len=64,
    max_num_batched_tokens=32,
    max_num_seqs=1,
    enforce_eager=True,
    skip_mm_profiling=True,
    block_size=64,
    skip_tokenizer_init=False,
)

sampling_params = SamplingParams(
    max_tokens=1,
    temperature=0,
)

outputs = llm.generate(["Hello, my name is"], sampling_params)
```

`skip_tokenizer_init=False` 虽未直接写在 JSON 中，但文本输入构造逻辑会补上它，
因此 vLLM 会加载模型目录中的 tokenizer，并把字符串 prompt 转换成 token IDs。

## 2.4 执行流程

1. 脚本模块加载时设置 `SIRT_THREAD_PRIVATE_MEMORY=4194304`，即每个 SIRT 线程
   4 MiB 私有内存。
2. 解析命令行，定位并用 `json.load()` 读取 recipe，同时验证 JSON 顶层是对象。
3. 以 recipe 名 `deepseek_v32_tiny` 创建日志文件，并把 stdout/stderr 同时输出到
   控制台和日志文件。
4. 在导入 vLLM 前应用 recipe 中的环境变量，确保 worker 使用 `spawn`，并扩大模型
   执行超时时间。
5. 合并模型路径和 `llm` 参数；验证绝对模型目录存在。本机该目录存在，约 4.0 GiB，
   包含 `config.json`、tokenizer 和一个 `model.safetensors` 权重文件。
6. 构造文本 prompt 和 `SamplingParams(max_tokens=1, temperature=0)`。
7. 导入 vLLM，激活 SIPU platform plugin，初始化引擎，加载 tokenizer、模型配置和
   FP8 权重，分配 KV cache，并执行内核编译或预热。
8. 初始化成功后调用 `llm.generate()`；vLLM 完成 prompt 的 prefill，并从末尾
   logits 采样一个 token，最后由 runner 打印结果。由于 `max_tokens=1`，请求通常
   随即结束，不会把这个新 token 再送入模型执行下一轮独立 decode forward。

## 2.5 Recipe 参数含义

| 配置 | 值 | 含义 |
| --- | --- | --- |
| `model` | `/share_data/vllm-sipu/models/DeepSeek-V3.2-tiny` | 使用本地模型目录，不从 Hugging Face 下载。它含真实权重，不是 `sipu_dummy` 随机权重。 |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | 使用新进程启动 worker，避免 SIPU 设备状态在 `fork` 后无法安全重新初始化。 |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `3000000` | 把模型执行超时设置得很大，避免仿真或首次内核编译耗时触发超时。 |
| `trust_remote_code` | `true` | 允许模型加载所需的自定义代码；当前日志实际解析到 `DeepseekV32ForCausalLM`。 |
| `quantization` | `fp8` | 按 FP8 量化模型路径初始化；模型配置还声明了动态激活量化、E4M3 和 128x128 权重块。 |
| `gpu_memory_utilization` | `0.01` | vLLM 用 1% 作为设备内存预算比例，主要影响可用于 KV cache 等运行时空间的计算；它不表示只加载 1% 的模型权重。 |
| `max_model_len` | `64` | 将单条序列支持的最大总长度限制为 64 tokens，以缩小 smoke test 资源需求。 |
| `max_num_batched_tokens` | `32` | 单次调度最多处理 32 tokens；vLLM 因此启用 chunked prefill。 |
| `max_num_seqs` | `1` | 同时调度的序列数最多为 1。 |
| `enforce_eager` | `true` | 禁用 vLLM 的 `torch.compile` 图优化和 CUDAGraph 路径，使用 eager 执行；SIPU 自定义内核仍可能进行自己的 JIT 编译。 |
| `skip_mm_profiling` | `true` | 跳过多模态 profiling；本 recipe 是纯文本输入，因此不需要图像/音频 profiling。 |
| `block_size` | `64` | KV cache 每个块按 64 tokens 组织。 |
| `max_tokens` | `1` | 最多只生成一个新 token，不会生成完整句子；通常只覆盖 prefill 后的首次采样，不覆盖下一轮独立 decode forward。 |
| `temperature` | `0` | 使用 greedy decoding，每一步选择分数最高的 token，不做随机采样。 |
| `input.type` | `text` | 走纯文本输入分支。 |
| `input.prompt` | `Hello, my name is` | 原始 prompt，不自动套用聊天模板。 |

命令行参数的优先级高于 recipe。例如下面的命令会替换 prompt，并把生成上限从 1
改为 4：

```bash
python3 examples/offline/offline_inference_recipe.py \
  --recipe deepseek/deepseek_v32_tiny \
  --prompt "The capital of China is" \
  --max-tokens 4
```

更通用的覆盖可使用 `--llm-arg KEY=JSON`、`--sampling-param KEY=JSON`、
`--model-config-arg KEY=JSON` 和 `--env KEY=VALUE`。

## 2.6 “Tiny” 模型的含义

这里的 Tiny 不是完整 DeepSeek-V3.2。当前本地 `config.json` 表明它保留
`hidden_size=7168` 等主要维度，但只保留 2 个 decoder layers、4 个 routed
experts、1 个 shared expert，并且每个 token 选择 2 个 routed experts；权重文件约
4.0 GiB。

因此它适合快速验证 DeepSeek-V3.2 的 MLA、稀疏 indexer、MoE 和 FP8 路径，也可作为
SIPU/CUDA 逐层 logits 对比的较小输入模型，但不能用来评价完整模型的生成质量。

## 2.7 成功时的输出与日志

默认日志目录是 `logs/model_smoke`。脚本在文件描述符层做 tee，所以 Python 日志、
C/C++ 扩展输出和子进程输出会同时出现在终端与下面的文件中：

```text
logs/model_smoke/deepseek_v32_tiny_YYYYMMDD_HHMMSS.log
```

若初始化和生成成功，末尾会打印：

```text
Prompt: 'Hello, my name is'
Prompt token IDs: [实际分词结果]
Output: '生成的一个 token 所对应的文本'
Generated IDs: [生成 token 的 ID]
```

具体 token 取决于模型权重和当前实现，不能仅从 recipe 静态推断。命令只打印普通
生成结果；尽管 recipe 的 `description` 提到 logit comparison，这条命令本身不会
导出逐层 logits，也不会比较 SIPU 与 CUDA。该用途需要另行使用
`examples/logit_compare/run_logits.py` 和 `compare.py`。

## 2.8 当前工作区的实际运行结果

当前已有日志：

```text
logs/model_smoke/deepseek_v32_tiny_20260817_110808.log
```

该次运行成功识别 SIPU plugin 和 `DeepseekV32ForCausalLM`，也加载了约 3.92 GiB 的
safetensors checkpoint；但它在 `LLM(...)` 初始化的模型 warm-up 阶段失败，尚未
执行 `llm.generate()`，因此日志中没有前述四行 Prompt/Output 结果。

首个实质异常是：

```text
TypeError: sparse_attn_indexer_fake() missing 1 required positional argument:
'dense_mha_metadata_layer_name'
```

原因是本地 SIPU 适配层
`vllm_sipu/model_executor/layers/sparse_attn_indexer.py` 仍按旧参数表调用上游 vLLM
的 `sparse_attn_indexer_fake()`，而当前上游函数签名已经增加 `use_pcp`、
`dense_mha_metadata_layer_name` 等参数。外层的
`RuntimeError: Engine core initialization failed` 只是该异常向上传播后的结果。
这与 prompt、1-token 采样配置或执行超时无关，需要同步 SIPU 适配层与当前 vLLM
接口后，命令才能进入实际生成阶段。

## 2.9 运行前提与适用边界

- 从仓库根目录运行，或把脚本和 recipe 改为绝对路径。
- 激活安装了兼容版本 vLLM、vLLM-SIPU、PyTorch-SIPU 的 Python 环境，并完成 SIPU
  SDK/runtime 环境初始化。
- 模型目录 `/share_data/vllm-sipu/models/DeepSeek-V3.2-tiny` 必须存在且可读；若路径
  不同，可用 `--model /path/to/model` 覆盖。
- `logs/model_smoke` 的父目录需要可写。
- 这是功能正确性和执行链路检查，不是吞吐/时延 benchmark；单 prompt、单序列、
  单 token 的配置不代表生产推理负载。

# 3. `mxint8.py` 量化适配层讲解

## 3.1 文件定位与核心结论

`vllm_sipu/model_executor/layers/quantization/mxint8.py` 是 **vLLM 量化框架与
SIPU MXInt8 kernel 之间的适配层**。它本身不实现量化公式或矩阵乘 kernel，主要负责：

1. 向 vLLM 注册用户可见的 `mxint8` 量化名称；
2. 决定哪些 vLLM layer 使用 MXInt8、保持 BF16，或明确报“不支持”；
3. 按 tensor-parallel 本地分片尺寸创建可加载的普通权重；
4. 权重加载完成后，把二维 BF16 权重一次性转换成 SIPU MXInt8 packed buffer；
5. 每次 forward 时动态量化输入 activation，调用 MXInt8 x MXInt8 GEMM，再裁剪、加
   bias 并恢复原输出形状。

因此当前 dense Linear 路径可概括为 **W8A8、BF16 输入/输出**：权重和 activation
都以 MXInt8 参与 GEMM，kernel 输出 BF16。权重只在加载后量化一次，activation 则在
每次前向中重新量化。

## 3.2 本次核对使用的环境

按给定路径执行了以下环境初始化：

```bash
source /share_data/users/like/miniconda3/etc/profile.d/conda.sh
conda activate /share_data/users/like/miniconda3/envs/vllm_dev
source ./sipu_sdk_setup.sh
```

仓库的 `sdk.version` 为 `2608121443`，因此脚本解析并设置：

```text
CONDA_PREFIX=/share_data/users/like/miniconda3/envs/vllm_dev
SIPU_SDK_PATH=/share_data/sicx_sdk/release/2608121443
SI_SDK_ROOT=/share_data/sicx_sdk/release/2608121443
SI_CMODEL_ROOT=/share_data/arch_cmodel_release/sipu1.5/2608120400
SI_CMODEL_HW_ARCH=1.5
```

实测软件版本为 Python 3.10.16、vLLM `0.27.1+sipu1`、torch-sipu
`0.7.0+sdk260801`；`torch.sipu.is_available()` 为 `True`，当前 CModel 环境可见 1 个
SIPU device。若不先 source SDK，`torch_sipu` 会因找不到 `libsipu.so` 而无法导入。

## 3.3 `mxint8` 如何注册到 vLLM

入口是类上的装饰器：

```python
@register_quantization_config("mxint8")
class MXInt8Config(QuantizationConfig):
    ...
```

导入该模块时，装饰器把字符串 `mxint8` 加入 vLLM 的量化方法表，并把它映射到
`MXInt8Config`。正常启动时的导入链是：

```text
SIPUPlatform.pre_register_and_update()
  -> import vllm_sipu.model_executor.layers.quantization
     -> quantization/__init__.py import mxint8
        -> @register_quantization_config("mxint8")
```

这个注册必须发生在 vLLM 构造 `VllmConfig` 之前，否则 `quantization="mxint8"` 会被
判为无效值。本次也验证了：单独导入 vLLM 时 `mxint8` 尚不在方法表中；导入 SIPU
quantization package 后，`get_quantization_config("mxint8")` 正确返回
`vllm_sipu...MXInt8Config`。

使用侧只需要显式配置：

```python
llm = LLM(model="/path/to/model", quantization="mxint8", dtype="bfloat16")
```

或在离线 recipe 的 `llm` 对象中写入：

```json
{
  "quantization": "mxint8"
}
```

## 3.4 MXInt8 数据格式

这里的 MXInt8 是 micro-scaling INT8，不等同于只有一个全 tensor scale 的普通
INT8。SDK 定义显示：

- 每个 micro block 包含 32 个 8-bit integer elements；
- 每个 block 共享一个 1-byte scale；物理 header 中该 scale 的存储 stride 为 2
  bytes，所以每 1024 个 element 对应 64 bytes header；
- element 数据和 scale/header 元数据按 SIPU tiled layout 打包；
- packed tensor 在 PyTorch 中表现为一维 `torch.uint8` buffer，但这些字节是 opaque
  storage，不能当作普通 unsigned INT8 tensor 直接运算。

当前实现每 1024 个对齐后的逻辑元素分配 1088 bytes：其中 1024 bytes 是 8-bit
element 数据，另外 64 bytes 是 tiled header/scale 存储。故对齐良好时：

```text
MXInt8 bytes = logical_elements / 1024 * 1088
MXInt8 / BF16 = 1088 / (1024 * 2) = 53.125%
```

即 packed storage 理论上约比 BF16 少 46.875%。但形状会先 padding，所以小 tensor
不一定节省空间。例如测试中的 `[65, 96]` 权重会补到 `[96, 128]`，packed 后为
13056 bytes，反而略大于原 BF16 的 12480 bytes；大型且自然对齐的 Linear 权重才
接近上述 53.125% 比例。

## 3.5 `MXInt8Config` 逐项说明

| 方法或字段 | 作用 |
| --- | --- |
| `ignored_layers` | 记录不转换为 MXInt8 的 Linear layer 完整前缀，默认为空。 |
| `get_name()` | 返回注册名称 `mxint8`。 |
| `get_supported_act_dtypes()` | 只声明支持 `torch.bfloat16` 模型/activation dtype。 |
| `get_min_capability()` | 返回 `0`，让 vLLM 的通用 capability 门槛不拦截；这不代表任意硬件都支持，实际仍依赖 SIPU plugin 和 kernel。 |
| `get_config_filenames()` | 返回空列表，表示不要求模型目录提供独立的 MXInt8 配置文件。 |
| `from_config()` | 从 `ignored_layers` 或兼容名称 `modules_to_not_convert` 中读取跳过列表；两个字段同时存在时前者优先。 |
| `get_quant_method()` | 根据 layer 类型选择实际量化策略。 |

当模型 `config.json` 没有量化配置、用户只显式传入 `quantization="mxint8"` 时，vLLM
会直接构造默认 `MXInt8Config()`，即默认尝试量化所有 `LinearBase`。若模型配置中
提供了 `ignored_layers`/`modules_to_not_convert`，匹配默认是 layer 的完整 prefix；
对 QKV、gate/up 等 fused layer，vLLM 还会借助 `packed_modules_mapping` 检查各逻辑
shard 是否采用一致精度，避免只跳过 fused layer 的一部分。

## 3.6 Layer 分派规则

`get_quant_method(layer, prefix)` 有三条路径：

| Layer 类型 | 返回值 | 行为 |
| --- | --- | --- |
| `LinearBase` 且命中 `ignored_layers` | `UnquantizedLinearMethod` | 保留普通 BF16 Linear。 |
| 其他 `LinearBase` | `MXInt8LinearMethod` | 使用本文件实现的 W8A8 Linear。 |
| `RoutedExperts` | `SIPUUnsupportedMoEMethod(MXINT8_W8A8)` | 创建 expert 权重时明确抛出 `NotImplementedError`。 |
| 其他 layer | `None` | 该配置不接管，沿用 layer 自身或 vLLM 默认实现。 |

因此当前 MXInt8 只接通了 dense Linear GEMM。RMSNorm、embedding、attention 等并不会
因为启用 `mxint8` 自动变成 MXInt8；更重要的是，RoutedExperts MoE 目前不能使用此
配置，报错原因会是：

```text
SIPU MoE quantization 'mxint8_w8a8' is not supported yet:
grouped MXInt8 expert GEMM is not wired up.
```

当前 `RoutedExperts` 分支没有检查 `ignored_layers`，所以仅把 MoE prefix 写入跳过列表
也不能绕过这个限制。

## 3.7 `create_weights()`：先创建可加载的普通权重

vLLM 构造 Linear layer 时先调用 `MXInt8LinearMethod.create_weights()`。设当前 TP rank
上的输入宽度为 `K_local`，各逻辑输出分片宽度之和为 `N_local`，本方法会：

1. 保存 `input_size_per_partition=K_local`、`output_size_per_partition=N_local`；
2. 保存 `logical_widths`，以保留 QKV 或 gate/up 等 merged Linear 的逻辑边界；
3. 保存 checkpoint 参数 dtype 到 `orig_dtype`；
4. 把 `mxint8_weight_shape` 初始化为 `None`，表示尚未 pack；
5. 创建形状 `[N_local, K_local]`、dtype 为 `params_dtype` 的
   `ModelWeightParameter`；
6. 设置 `input_dim=1`、`output_dim=0` 和 vLLM 的 `weight_loader`，使 tensor-parallel
   loader 能把 checkpoint 的正确本地 shard 加载进来。

这里故意先创建 BF16/原始 dtype 的二维权重，而不是直接创建 packed `uint8` 权重。
所以该实现消费的是普通 dense checkpoint，再在加载后转换；它不是直接读取已经
预打包好的 MXInt8 checkpoint。

## 3.8 `process_weights_after_loading()`：权重只量化一次

checkpoint 或 dummy 权重装载完成后，vLLM 遍历所有量化 layer 并调用该方法：

```text
原始 weight [N_local, K_local], BF16
  -> N 向上补齐到至少 32 且为 32 的倍数
  -> K 向上补齐到至少 128 且为 128 的倍数
  -> 在原设备生成 padded tensor
  -> 搬到 CPU 转换为 tiled layout
  -> 拷回 SIPU
  -> hp_to_mx kernel 量化并 pack
  -> 一维 uint8 packed_weight
```

若原始权重形状是 `[N, K]`，保存的逻辑物理形状为：

```text
N_pad = ceil(max(N, 32) / 32) * 32
K_pad = ceil(max(K, 128) / 128) * 128
layer.mxint8_weight_shape = (N_pad, K_pad)
```

实际 `layer.weight` 被替换成不可训练的一维 `torch.uint8` Parameter；原来的二维关系
不再编码在 tensor shape 中，而由 `mxint8_weight_shape` 单独保存。scale/header 也已
嵌入 packed buffer，没有独立的 `weight_scale` Parameter。

这个替换也意味着原 `ModelWeightParameter` 上的 `input_dim`、`output_dim` 和
`weight_loader` 元数据不再保留在新 Parameter 上。正常的首次 checkpoint load 不受
影响，因为替换发生在 load 之后；但直接再次调用外部权重加载、refit 或其他在线更新
路径不能假定仍支持，需要针对具体 vLLM 加载流程验证。

方法开头的条件：

```python
mxint8_weight_shape is not None and weight.dtype == torch.uint8
```

是幂等保护，避免 post-load hook 重复量化。若输入权重不是二维，会直接抛出
`ValueError`。

需要注意，转换阶段原始权重、padding tensor、CPU tiled staging tensor 和 packed
结果可能短暂共存，所以“最终权重约为 BF16 的 53%”不等于加载峰值内存也只有 53%。

## 3.9 `apply()`：每次前向的数据流

假设输入为 `[..., K]`，将前导维展平后有 `M` 行，本 rank 输出宽度为 `N`：

```text
BF16 input [..., K]
  -> reshape [M, K]
  -> 在原设备 zero-pad [M_pad, K_pad]
  -> 搬到 CPU 做 tile reorder，再传回 SIPU
  -> hp_to_mx: packed activation A (uint8 byte buffer)
                                      \
                                       -> MXInt8 GEMM -> tiled BF16 buffer
                                      /
预先缓存的 packed weight B (uint8) --
GEMM output -> tiled-to-linear 还原为 [M_pad, N_pad]
  -> 裁剪为 [M, N]
  -> 可选地加 BF16 bias
  -> reshape 回 [..., N]
```

这里 `M_pad`、`N_pad` 至少为 32 且为 32 的倍数，`K_pad` 至少为 128 且为 128 的
倍数。C++ FFI 入口会再次检查：两个输入必须是一维连续 `uint8` packed buffer，输入、
权重和输出的 device type/id 必须一致，`N` 必须 32 对齐、`K` 必须 128 对齐，输出
必须为二维 BF16。FFI 本身未显式断言 device type 是 SIPU，但后续 kernel 和正常调用
环境要求 SIPU。

底层调用链为：

```text
MXInt8LinearMethod.apply
  -> apply_mxint8_linear
     -> vllm_sipu.ops.quantization.quantize_to_mxint8
        -> JIT provider -> hp_to_mx<sifmt::mxint8, ...>
     -> vllm_sipu.ops.quantization.mxint8_bf16_matmul
        -> JIT provider -> mma_bf16_mxi8_universal
        -> _tileformat_to_linear
```

Python helper 负责 padding、shape bookkeeping、CPU tile reorder、输出裁剪和 bias；
C++/sikernel 负责实际 pack 与 GEMM；JIT wrapper 再把 kernel 的 tiled BF16 输出还原为
线性二维布局。当前高层始终令 `M_pad >= 32`，所以 detile 使用 `(32, 16)` tile。kernel
使用当前 SIPU stream；JIT module 在首次调用时按需加载/构建，随后在同一进程中由
`lru_cache` 复用。

## 3.10 Tensor Parallel 与 merged Linear

本文件始终针对 **当前 rank 的本地权重 shard** 工作，而不是对全局权重先 pack 再
切分：

- ColumnParallelLinear 对本地 `N_local` 做 padding、pack 和 GEMM，必要的 all-gather
  仍由外层 vLLM Linear 实现完成；
- RowParallelLinear 使用本地 `K_local`，GEMM 后的跨 rank reduce 仍由外层完成；
- QKV、gate/up 等 merged Linear 通过 `output_partition_sizes` 和 `logical_widths`
  保留多个逻辑输出宽度，但 MXInt8 kernel 看到的是合并后的 `N_local`；
- bias 不量化，在 BF16 GEMM 结果裁剪后相加；若 layer 使用 `skip_bias_add`，bias 由
  外层 vLLM 路径延后处理。

换言之，`mxint8.py` 替换的是每个 Linear 的本地数学核心，通信语义仍归 vLLM 的
parallel Linear layer 管理。

## 3.11 当前实现的限制与注意事项

1. **高层接口只支持 BF16。** `MXInt8Config` 仅声明 BF16。底层 pack C++ 虽接受
   BF16、FP16、FP32，但集成的 GEMM 固定输出 BF16，测试也只覆盖 BF16。
2. **MoE 未实现。** `RoutedExperts` 会通过占位 method 尽早抛错，因为 grouped
   MXInt8 expert GEMM 尚未接线。
3. **activation 每次动态量化。** 每个 Linear forward 都会生成临时 packed input，
   不是静态 activation quantization。
4. **当前布局转换经过 CPU。** `quantize_to_mxint8()` 先在原设备生成 padded tensor，
   `build_mxint8_tile_tensor_on_cpu()` 再把它搬到 CPU 做 tile reorder，随后传回 SIPU
   完成 pack；这会引入设备临时量、同步与双向传输开销，是理解当前性能时最重要的
   实现细节之一。
5. **小形状 padding 成本高。** helper 当前总是把 row 补到至少 32；虽然更底层
   kernel 支持 8/16/32 row family，但该 Linear 高层路径实际统一走 32-row padding。
   `for_weight` 参数目前也不会改变 row padding 结果。
6. **不是预量化 checkpoint loader。** 权重先以原 dtype 完整加载，再 post-load
   pack；不能把任意一维 `uint8` checkpoint 直接当作本实现的 packed weight。
7. **存在有损误差。** 单测采用 normalized MAE 小于 `0.15` 的判据，不应期待结果与
   FP32/BF16 reference 逐 bit 一致。
8. **其他 layer 不会自动量化。** 本文件只接管 `LinearBase`，不能把它理解成整个
   model graph 的全算子 INT8 化。
9. **输入宽度依赖上层保证。** 高层没有显式检查 `input.shape[-1]` 是否等于 layer 的
   `K_local`；标准 vLLM Linear 调用会满足该契约，但脱离 layer 直接调用 method 时应
   自行校验，否则较小宽度可能被补零后继续执行。

## 3.12 测试验证与覆盖边界

在指定 conda 与 SDK/CModel 环境中执行：

```bash
source /share_data/users/like/miniconda3/etc/profile.d/conda.sh
conda activate /share_data/users/like/miniconda3/envs/vllm_dev
source ./sipu_sdk_setup.sh
pytest -q tests/kernels/quantization/test_mxint8.py
```

结果为：

```text
3 passed in 20.52s
```

三个测试分别验证：

- `[7, 96]` BF16 tensor padding 到 `[32, 128]` 后能正确产出一维 `uint8` packed
  buffer，且 storage size 符合公式；
- `[7, 96] x [65, 96]^T` 的 MXInt8 GEMM 与 FP32 reference 的 normalized MAE，
  即 `mean(abs(output-reference)) / mean(abs(reference))`，小于 `0.15`；
- `MXInt8LinearMethod` 能完成权重 post-load 量化、forward、输出裁剪和 bias 相加。

当前测试范围仍较窄：单个 seed、单组小形状、BF16、单 SIPU CModel device；没有
覆盖真实模型端到端、tensor parallel、merged QKV、ignored layer、物理 SIPU 性能，
也没有覆盖明确不支持的 MoE 路径。因此“3 个单测通过”证明基础 dense Linear 数据
流在当前环境可用，但不能外推为所有模型和部署形态都已验证。

## 3.13 当前工作区的端到端运行证据

工作区中还有两次使用
`examples/recipes/qwen/qwen25_05b_instruct_dummy___mxint8.json` 的实际日志，它们可以
进一步说明“配置错误”和“MXInt8 执行结果”的区别。

第一次日志：

```text
temp/qwen25_05b_instruct_dummy___mxint8.json.log.2026_08_17___14_44_53
```

在 `LLM(**llm_kwargs)` 参数解析阶段就失败：

```text
TypeError: EngineArgs.__init__() got an unexpected keyword argument 'quantized'
```

这次尚未构造模型，更未调用 MXInt8 pack/GEMM。原因是当时 recipe 向当前 vLLM
`EngineArgs` 传了已经不支持的 `quantized` 参数；当前工作区中的 recipe 已不再包含
该字段，所以这不是 `mxint8.py` 或 kernel 的失败。

第二次修正后的日志：

```text
temp/qwen25_05b_instruct_dummy___mxint8.json.log.2026_08_17___14_55_46
```

该次端到端执行成功，日志明确记录了：

```text
quantization=mxint8
dtype=torch.bfloat16
load_format=sipu_dummy
Prompt: 'Hello, my name is'
Prompt token IDs: [9707, 11, 847, 829, 374]
Output: '₲'
Generated IDs: [147294]
```

运行经过 2 层裁剪 Qwen2 模型的 `QKVParallelLinear`、`RowParallelLinear`、
`MergedColumnParallelLinear` 和最终 logits/sample 路径，证明当前单 rank CModel 环境下
MXInt8 已能嵌入实际 vLLM 模型执行链。该请求处理耗时约 `410.84s`，这反映的是当前
CModel、debug hook、JIT/布局转换等组合下的 smoke-test 时间，不能当作物理 SIPU
性能指标。

同时，该 recipe 使用 `sipu_dummy` 随机权重并把模型缩小到 smoke-test 维度，所以
生成字符 `₲` 没有语言质量意义。此结果验证的是执行链路，而不是模型准确率，也仍
不能替代真实权重、物理设备、TP/多卡和性能测试。

# 4. `MXInt8LinearMethod.create_weights()` 的内存设备如何决定

## 4.1 直接结论

对于题目给出的这次运行：

```bash
python3 examples/offline/offline_inference_recipe.py \
  --recipe examples/recipes/qwen/qwen25_05b_instruct_dummy___mxint8.json
```

`create_weights()` 中下面这次 `torch.empty()` **最初在 CPU 上分配**：

```python
torch.empty(
    output_size_per_partition,
    input_size_per_partition,
    dtype=params_dtype,
)
```

但这只是未量化 dense weight 的初始位置，并不代表推理时 weight 仍在 CPU。该 recipe
使用 `load_format="sipu_dummy"`：模型和 dummy weight 先在 CPU 构造，之后 loader 调用
`model.to(sipu:0)`，最后再在 `sipu:0` 上把 dense BF16 weight 转成 MXInt8 packed
weight。因此本次运行中：

```text
create_weights 刚返回：CPU 上的二维 BF16 ModelWeightParameter
完成 model.to(...) 后：sipu:0 上的二维 BF16 Parameter
完成 post-load 量化后：sipu:0 上的一维 uint8 MXInt8 packed Parameter
```

## 4.2 `torch.empty()` 本身采用什么规则

这里没有传 `device=`，也没有可供继承 device 的输入 tensor，所以设备不是由 shape、
`params_dtype` 或 `ModelWeightParameter` 决定。PyTorch 使用调用发生时的 **默认设备**：

1. 若 factory 显式传入 `device=...`，使用显式设备；
2. 否则使用当前 `with torch.device(...)` 上下文或 `torch.set_default_device(...)` 设置的
   默认设备；
3. 如果外层没有修改默认设备，通常就是 CPU。

`params_dtype` 只决定 element dtype。本次为 `torch.bfloat16`，它不携带 device 信息。
`ModelWeightParameter(data=...)` 只是把已经创建好的 tensor 包装成 vLLM Parameter，并
附加 TP shard/weight loader 元数据；其构造函数不会把 `data` 搬到另一个设备。

还要区分两个容易混淆的概念：

- `torch.sipu.set_device(...)` 或 `torch.accelerator.set_device_index(...)` 选择当前 SIPU
  device index；
- PyTorch 默认 factory device 决定省略 `device=` 的 `torch.empty()` 是否使用 SIPU。

前者本身不会把一个仍处于 CPU 默认设备上下文中的无 `device=` factory 改成 SIPU
分配。source `sipu_sdk_setup.sh` 的作用是让 SIPU runtime/CModel 可用，也不会单独改变
这条 factory 的默认设备。

## 4.3 本次 `sipu_dummy` 的实际调用链

recipe 明确包含：

```json
{
  "load_format": "sipu_dummy",
  "quantization": "mxint8"
}
```

日志也记录了 `SipuDummyModelLoader` 的注册，以及
`load_format=sipu_dummy, device_config=sipu, quantization=mxint8`。对应 loader 在
`vllm_sipu/model_executor/model_loader/sipu_loader.py` 中执行：

```python
with set_default_torch_dtype(model_config.dtype):
    with _cpu_default_device_context():
        model = initialize_model(...)
```

`_cpu_default_device_context()` 的主要实现是：

```python
previous = torch.get_default_device()
torch.set_default_device("cpu")
try:
    yield
finally:
    torch.set_default_device(previous)
```

`initialize_model()` 会调用 Qwen2 模型构造函数；模型构造 Linear layer 时，vLLM 再调用
`MXInt8LinearMethod.create_weights()`。因此 `torch.empty()` 运行时仍位于上述 CPU
default-device scope 中，初始 weight 必然是 CPU tensor。完整关系是：

```text
SipuDummyModelLoader.load_model
  -> torch.set_default_device("cpu")
  -> initialize_model
     -> Qwen2....__init__
        -> QKV/Column/Row/Merged Linear.__init__
           -> MXInt8LinearMethod.create_weights
              -> torch.empty(..., device omitted)  # CPU
              -> ModelWeightParameter(data=...)    # 仍是 CPU
```

## 4.4 初始 CPU weight 如何变成 SIPU MXInt8 weight

模型构造后，`SipuDummyModelLoader` 依次执行：

| 阶段 | 主要操作 | weight 位置与形式 |
| --- | --- | --- |
| 1. 构造 | `create_weights()` | CPU，二维 BF16 dense weight；内容尚未初始化。 |
| 2. dummy 初始化 | `_initialize_sipu_dummy_weights()` | CPU，二维 BF16 random weight。 |
| 3. 搬运模型 | `model.to(target_device)` | `sipu:0`，二维 BF16 dense weight。 |
| 4. post-load | `process_weights_after_loading(...)` | 在 SIPU weight 上执行 padding、布局转换和 MXInt8 pack。 |
| 5. 完成 | `layer.weight = Parameter(packed_weight)` | `sipu:0`，一维 `uint8` packed weight。 |

`target_device` 由 `_target_device_for_model_load()` 决定：若
`vllm_config.load_config.device` 显式配置，则优先使用它；否则使用当前 platform 的
device type 和 `device_config.device` 中的 index，index 未给出时取 `0`。本次是单 rank、
`device_config=sipu`，所以结果是 `torch.device("sipu", 0)`。

在第 4 阶段，`process_weights_after_loading(model, ..., target_device)` 还用
`device_loading_context` 保证待处理参数位于目标设备。这里 weight 已由
`model.to(sipu:0)` 搬好，所以 `MXInt8LinearMethod.process_weights_after_loading()` 读到的
`layer.weight.device` 是 `sipu:0`；MXInt8 op 最终创建的 packed buffer 也跟随输入位于
`sipu:0`。中间的 tile reorder 会经过 CPU staging，但最终 packed weight 不留在 CPU。

## 4.5 为什么 `create_weights()` 不直接写 `device="sipu"`

这是 vLLM layer 的上下文驱动构造方式：量化 method 只描述参数形状、dtype 和加载
元数据，实际放置策略交给 model loader。这样同一份 layer 代码可以配合 CPU staging、
目标加速器直接构造、测试环境以及其他 loader。

例如 vLLM 通用 `BaseModelLoader` 的路径不是强制 CPU，而是：

```python
target_device = torch.device(load_device)
with set_default_torch_dtype(model_config.dtype):
    with target_device:
        model = initialize_model(...)
```

此时同一个没有 `device=` 的 `torch.empty()` 会直接分配到 `target_device`。如果脱离
loader 单独调用 `create_weights()`，它就使用调用者当时的 `torch.get_default_device()`；
默认通常为 CPU。因此不能脱离外层构造上下文，仅根据 `mxint8.py` 这一行判断它永远
在 CPU 或永远在 SIPU。

## 4.6 环境实测与日志印证

在题目给定 conda 和 SDK 环境中做了最小 factory 验证：

```text
initial default: cpu
initial empty: cpu
sipu context default: sipu:0
sipu context empty: sipu:0
after context default: cpu
forced cpu default: cpu
forced cpu empty: cpu
```

这验证了同一个 `torch.empty(1)` 在无上下文时分配到 CPU，在
`with torch.device("sipu:0")` 中则分配到 `sipu:0`。

提供的日志没有打印 `create_weights()` 刚返回时的 parameter device，但它明确显示本次
选择了 `sipu_dummy` loader；模型加载完成后的 QKV、RowParallel 和 MergedColumn
Linear 输入输出均为 `device=sipu:0`，且 MXInt8 推理成功完成。这与“CPU 构造 ->
`model.to(sipu:0)` -> SIPU 上 post-load pack”的源码路径一致。

## 4.7 一句话回答

这行 `torch.empty()` 通过 **调用时的 PyTorch 默认设备上下文** 决定分配位置；在本次
`sipu_dummy` recipe 中该上下文被 loader 显式设为 CPU，所以初始内存在 CPU，随后
注册到 model 的 weight 被 `model.to(sipu:0)` 搬到 SIPU，并在那里转换为最终 MXInt8
packed weight。

# 5. `hp_to_mx` 与 `_tileformat_to_linear` 的功能和定义位置

## 5.1 两者的区别

这两个函数处在 MXInt8 Linear 的不同阶段，功能并不相似：

| 函数 | 所处阶段 | 是否改变数值精度 | 是否改变内存排布 |
| --- | --- | --- | --- |
| `hp_to_mx` | GEMM 之前，量化 activation 或 weight | 是：BF16/FP16/FP32 转 MXInt8 | 是：输出 SIPU MX tiled packed layout |
| `_tileformat_to_linear` | GEMM 之后，还原 BF16 output | 否：输入输出都是 BF16 | 是：tile-major 转普通 row-major |

简化地说：

```text
hp_to_mx：高精度数值 -> 低精度、带 scale 的 MXInt8 packed bytes
_tileformat_to_linear：GEMM 已算出的 BF16 tiled bytes -> 正常二维 BF16 tensor
```

## 5.2 `hp_to_mx` 在当前调用中的功能

`hp` 表示 high precision，`mx` 表示 micro-scaling format。当前 C++ bridge 的调用是：

```cpp
::hp_to_mx<sifmt::mxint8, HPType, 0>(
    input.data_ptr(),
    output.data_ptr(),
    1,
    input.size(0),
    input.size(1),
    current_stream(input.device()));
```

模板和参数在这里分别表示：

- `sifmt::mxint8`：目标格式；
- `HPType`：源格式，C++ FFI 根据输入 dtype 选择 `sifmt::bfloat16`、
  `sifmt::float16` 或 `sifmt::float32`；本次 vLLM 集成实际使用 BF16；
- `0`：`TNOCP` 模板参数；SiKernel 文档说明它主要用于 MXFP6 的量化模式，本次
  MXInt8 路径使用默认值；
- `batch_size=1`、`dim0=rows`、`dim1=cols`；
- 最后一个参数是当前 SIPU stream，因此 kernel 与调用方的 stream 顺序保持一致；
- 未显式给出的模板参数使用默认值：输入按 `TMAP_FORMAT_TILED` 解释，且不预先清零
  整个输出 buffer。

在调用 `hp_to_mx` 之前，Python helper 已经完成 shape padding，并通过
`linear_to_tileformat()` 把普通 row-major 高精度矩阵改成输入 tiled layout。因此
`hp_to_mx` 不是接收任意未对齐的普通矩阵；当前高层路径保证 row 至少为 32 且 32
对齐、column 至少为 128 且 128 对齐。

## 5.3 `hp_to_mx` 内部做了什么

其 host wrapper 和 device kernel 的主要步骤是：

1. 校验 batch、`dim0`、`dim1`、指针及 tile 对齐条件；
2. 根据 `dim0` 选择 8、16 或 32 行的硬件 tile geometry；当前高层总会选择 32 行；
3. 编码输入 tensor map，并用 DTE 把一个高精度 tile 搬到每个线程私有的 shared/L2B
   staging 区；
4. 通过 `HpToOutputTraits<sifmt::mxint8, HPType>` 选择 MXInt8 转换 intrinsic
   `tcvt_mxi8`；
5. 对每个 32-element micro block 生成共享的 1-byte scale，并把各 element 量化成
   8-bit value；
6. 将 element data 和 scale/header 按 GEMM kernel 需要的 MX tile/supertile 物理顺序
   写入输出 buffer。

SDK 中 `sifmt::mxint8` 的默认 micro block size 是 32，scale type 是 `uint8`。SiKernel
的 MXInt8 output traits 规定每个物理 tile 包含 1024 bytes element data 和 64 bytes
header，所以当前代码每处理 1024 个对齐后的逻辑元素，输出 1088 bytes。这也是
Python `_expected_storage_size()` 中 `1024 -> 1088` 公式的来源。

最终输出在 PyTorch 侧表现为一维连续 `torch.uint8` tensor，但它同时包含量化 element、
scale/header 和硬件排布信息。它是 opaque packed storage，不能当作普通 row-major
`uint8` 矩阵使用；后续 `mma_bf16_mxi8_universal` 会按相同的 MX layout 读取它。

在 vLLM 生命周期中：weight 在 `process_weights_after_loading()` 中调用一次
`hp_to_mx`；activation 则在每次 `MXInt8LinearMethod.apply()` 时调用一次。

## 5.4 `hp_to_mx` 的定义位置

`hp_to_mx` 不定义在 `mxint8.py`，而是来自项目拉取的 SiKernel dependency：

| 层次 | 文件与位置 | 作用 |
| --- | --- | --- |
| vLLM-SIPU JIT source 清单 | `vllm_sipu/ops/backends/sikernel/jit/mxint8.py:26-47` | 把下面的 SiKernel 文件加入 `quantization_mxint8` JIT module。 |
| C++ FFI 调用点 | `csrc/jit/quantization/mxint8.cpp:55-60` | 根据 PyTorch dtype 实例化并调用 `hp_to_mx<sifmt::mxint8, HPType, 0>`。 |
| 公共声明 | `.deps/sikernel-src/include/sikernel.h:2282-2313` | 声明模板接口及参数语义。 |
| host wrapper 定义 | `.deps/sikernel-src/source/source_builtin/misc/hp_to_mx/kernel/hp_to_mx_kernel.su:26-87` | 校验 shape、构造 tensor map、选择 row geometry、launch kernel。 |
| device kernel 定义 | `.deps/sikernel-src/source/source_builtin/misc/hp_to_mx/kernel/hp_to_mx_kernel.hpp:53-108` | 搬运 tile、调用转换 intrinsic、写 packed MX tile。 |
| MXInt8 traits | `.deps/sikernel-src/source/source_builtin/misc/hp_to_mx/kernel/hp_to_output_traits.hpp:132-164` | 指定 `tcvt_mxi8`、8-bit data、64-byte header 等。 |
| tile/shape 规则 | `.deps/sikernel-src/source/source_builtin/misc/hp_to_mx/kernel/hp_to_mx_tensormap.hpp` | 定义 tile geometry、对齐、物理 supertile 索引及 tensor map。 |

其中 `.su` 是 SIPU kernel compilation unit。`.deps/sikernel-src` 是当前仓库的 SiKernel
checkout；JIT builder 按 `MXINT8_MODULE.sikernel_sources` 将这些源文件编入运行时 module。

更底层的类型/硬件定义来自给定 SDK：

- `/share_data/sicx_sdk/release/2608121443/include/SiTe/sifmt/sifmt_common.hpp:19`
  定义 MX block size 为 32；
- `/share_data/sicx_sdk/release/2608121443/include/SiTe/sifmt/sifmt_mx.hpp:24-107`
  定义 `SiMxData` 和 `sifmt::mxint8`；
- SDK toolchain 的 `siorigin_tile_150g.h` 声明 `tcvt_mxi8` 硬件 intrinsic。

## 5.5 `_tileformat_to_linear` 的功能

MXInt8 GEMM 的数学结果是 BF16，但 kernel 不是按普通 row-major 顺序写 output，而是按
硬件 tile 顺序连续写。Python 虽然预先分配了 shape 为 `[m, n]` 的 `out`，该 shape
metadata 并不能改变 kernel 实际写入的 byte order；若直接返回 `out`，从普通 PyTorch
二维索引观察时，element 会被打乱。

`_tileformat_to_linear()` 负责把这段 tile-major BF16 storage 重新排列为正常的
row-major `[m, n]`：

```python
num_tile_m = m // tile_rows
num_tile_n = n // tile_cols
return (
    x.reshape(num_tile_m, num_tile_n, tile_rows, tile_cols)
     .permute(0, 2, 1, 3)
     .contiguous()
     .reshape(m, n)
)
```

索引含义可写成：

```text
[tile_m, tile_n, row_in_tile, col_in_tile]
                    |
                    v
[tile_m, row_in_tile, tile_n, col_in_tile]
```

也就是先恢复 tile 内部的行，再把同一输出行上的各个 column tile 拼接起来。该函数：

- 不做 MXInt8 dequantization；GEMM kernel 已直接产生 BF16；
- 不重新计算 GEMM，也不改变任何数值；
- 不负责删除 padding，crop 在后续 `apply_mxint8_linear()` 中完成；
- `.contiguous()` 会在当前 device 上实际生成一份按线性顺序排列的 BF16 buffer，实际
  路径中不会为此往返 CPU，但会产生一次排布 copy 和临时内存。

它与量化前使用的 `linear_to_tileformat()` 在索引变换上互为逆操作。本次在指定环境中
用 `[32, 32]` BF16 tensor、`(32, 16)` tile 做了 round-trip 验证：tiled tensor 的顺序
确实不同，而 `_tileformat_to_linear(linear_to_tileformat(x))` 与原 tensor 逐元素完全
相同。

## 5.6 输出 tile shape 如何选择

`_mxint8_output_tile_shape()` 按 `m` 选择：

```text
m >= 32  -> (tile_rows, tile_cols) = (32, 16)
m >= 16  -> (16, 32)
其他     -> (8, 64)
```

三种 tile 都正好是 1024 bytes BF16 storage：

```text
32 * 16 * 2 bytes = 1024 bytes
16 * 32 * 2 bytes = 1024 bytes
 8 * 64 * 2 bytes = 1024 bytes
```

当前 `apply_mxint8_linear()` 总会把 `M` 补到至少 32，因此实际走 `(32, 16)`。这里的
`(32, 16)` 是 **BF16 output 的物理存储 tile**，不要与 GEMM 的 `(32, 32)` 计算 tile
混为一谈：一个 `(32, 32)` BF16 计算结果需要两个 `(32, 16)`、每个 1024 bytes 的
存储 tile。函数签名虽然接收 `k`，当前实现立即 `del k`，tile shape 实际只由 `m`
决定。

## 5.7 `_tileformat_to_linear` 的定义和调用位置

该函数是 vLLM-SIPU 仓库内的私有 Python helper：

```text
定义：vllm_sipu/ops/backends/sikernel/jit/mxint8.py:80-101
调用：vllm_sipu/ops/backends/sikernel/jit/mxint8.py:120-130
逆变换：vllm_sipu/model_executor/layers/quantization/utils/mxint8_utils.py:70-97
```

`mxint8_bf16_matmul()` 先分配 BF16 `out`，调用 JIT module 中的 C++ FFI；C++ 再 launch
`mma_bf16_mxi8_universal`。kernel 返回后，Python 才选择 tile shape 并调用
`_tileformat_to_linear(out, ...)`。所以第 3 章的调用链更严格地应读作：

```text
vllm_sipu.ops.quantization.mxint8_bf16_matmul
  -> Python JIT provider: mxint8_bf16_matmul
     -> 分配 BF16 out
     -> C++ FFI mxint8_bf16_matmul
        -> mma_bf16_mxi8_universal  # 写出 tiled BF16
     -> _mxint8_output_tile_shape
     -> _tileformat_to_linear       # Python 中还原 linear BF16
```

即 `_tileformat_to_linear` 不是 `mma_bf16_mxi8_universal` 内部函数，也不是 JIT C++
module 的 exported function；它是 provider 在 JIT kernel 返回后顺序执行的 Python
post-processing。

## 5.8 完整数据流中的位置

将两个函数放回整个 Linear 路径，可得到：

```text
BF16 activation / BF16 weight
  -> padding + linear_to_tileformat
  -> hp_to_mx
  -> MXInt8 tiled packed activation / weight
  -> mma_bf16_mxi8_universal
  -> tiled BF16 output
  -> _tileformat_to_linear
  -> linear BF16 padded output
  -> crop + optional bias
```

因此二者分别解决两个独立问题：`hp_to_mx` 负责“如何把高精度输入变成 GEMM 可消费的
MXInt8”，`_tileformat_to_linear` 负责“如何把 GEMM 的硬件排布输出恢复成 vLLM 后续
算子可正常索引的 BF16 tensor”。

# 6. `quantize_to_mxint8()` 从 Python 到 SiKernel 的完整过程

## 6.1 函数定位与返回值

本章讲解的入口是：

```text
vllm_sipu/model_executor/layers/quantization/utils/mxint8_utils.py:163-180
```

函数签名为：

```python
def quantize_to_mxint8(
    x: torch.Tensor,
    padded_shape: tuple[int, int] | None = None,
    *,
    for_weight: bool = False,
) -> tuple[torch.Tensor, tuple[int, int]]:
```

它接受一个二维 BF16/FP16/FP32 dense tensor，完成 shape padding、CPU tile reorder、
SIPU MXInt8 量化和物理打包，返回：

1. 一维、连续、`torch.uint8` 的 opaque packed buffer；
2. packed buffer 所代表的二维逻辑 padded shape。

第二个返回值非常重要。packed tensor 自身只有 `[packed_bytes]` 这一维，原来的 rows 和
cols 已不能从 PyTorch shape 直接恢复；后续 GEMM 必须另外接收 padded rows/cols。

这里还有三个同名函数，需要按层次区分：

| 层次 | 位置 | 返回值 |
| --- | --- | --- |
| 高层 helper | `.../quantization/utils/mxint8_utils.py:163` | `(packed, padded_shape)` |
| JIT provider | `vllm_sipu/ops/backends/sikernel/jit/mxint8.py:104` | `packed` |
| C++ FFI export | `csrc/jit/quantization/mxint8.cpp:72` | 原地写入预分配的 `output` |

## 6.2 Python 高层函数逐行说明

函数主体可拆成五步：

```python
if x.ndim != 2:
    raise ValueError(...)

if padded_shape is None:
    padded_shape = get_padded_mxint8_shape(...)

x_padded = pad_to_mxint8_shape(x, padded_shape)
x_tiled_cpu = build_mxint8_tile_tensor_on_cpu(x_padded, padded_shape)
x_tiled = x_tiled_cpu.to(device=x.device)
packed = quantize_mxint8_op(x_tiled)
return packed, padded_shape
```

具体含义如下：

1. **只接受二维 tensor。** Linear activation 在进入这里前已把前导维展平成 `[M,K]`，
   weight 原本就是 `[N,K]`。
2. **计算 padded shape。** rows 至少为 32 并向上对齐到 32；cols 至少为 128 并向上
   对齐到 128。
3. **在原 device 上补零。** `pad_to_mxint8_shape()` 创建 padded tensor，把原值复制到
   左上角，其余区域为零。
4. **在 CPU 上重排高精度数据。** tensor 被搬到 CPU，再由
   `linear_to_tileformat()` 从普通行优先顺序改成 SIPU tiled 顺序。
5. **搬回原 device 并量化。** tiled BF16 tensor 回到 `x.device`，通过 operator
   dispatch 进入 JIT SiKernel provider，输出同 device 上的 packed MXInt8 buffer。

当前 `get_padded_mxint8_rows()` 的 `for_weight` 参数并未参与计算，所以 activation 和
weight 的 row padding 规则实际相同。`for_weight=True` 目前只是保留在接口中，并不会
选择另一种 layout 或 tile family。

## 6.3 Padding 规则与 storage 公式

若输入 shape 为 `[rows, cols]`：

```text
padded_rows = ceil(max(rows, 32) / 32) * 32
padded_cols = ceil(max(cols, 128) / 128) * 128
```

packed storage 公式为：

```text
packed_bytes = padded_rows * padded_cols / 1024 * 1088
```

每 1024 个 padded 高精度逻辑元素会变成：

```text
1024 bytes  MXInt8 element data
  64 bytes  scale/header metadata
----------
1088 bytes  packed storage
```

MXInt8 的 micro block size 是 32 elements，每个 block 共享一个 `uint8` scale。scale
嵌在 tiled header 中，没有单独返回 `scale` tensor。

## 6.4 `linear_to_tileformat()` 到底改变了什么

测试使用 BF16，每个 element 为 2 bytes。`MXINT8_TILE_BYTES=32`，所以输入 tile 的列宽
为：

```text
32 bytes / 2 bytes per BF16 = 16 columns
```

`linear_to_tileformat()` 执行：

```python
x.reshape(num_tile_m, 32, num_tile_n, 16)
 .permute(0, 2, 1, 3)
 .contiguous()
 .reshape(m, n)
```

元素的物理遍历顺序由：

```text
[tile_m, row_in_tile, tile_n, col_in_tile]
```

变为：

```text
[tile_m, tile_n, row_in_tile, col_in_tile]
```

也就是先存完整的 `32 x 16` tile，再存下一个 column tile。

需要区分两种“layout”含义：

- 从 PyTorch metadata 看，重排前后都是 `torch.strided`、二维、contiguous，stride 也
  都是 `(cols, 1)`；
- 从 buffer 内元素的语义顺序看，重排前是普通 row-major，重排后是 tile-major。

因此不能仅根据 `is_contiguous() == True` 或 stride 判断这块 buffer 是否仍按普通矩阵
顺序存放。`x_tiled_cpu` 的 shape 仍为 `[padded_rows,padded_cols]`，只是相同坐标不再能
按普通二维语义直接解释。

## 6.5 从 public operator 到 JIT provider

高层 helper 导入的是：

```python
from vllm_sipu.ops.quantization import quantize_to_mxint8 as quantize_mxint8_op
```

`vllm_sipu/ops/quantization.py:107-109` 本身只调用 `_dispatch()`。provider 映射定义在：

```text
vllm_sipu/ops/op_list.yaml:668-674
```

其中 `quantize_to_mxint8` 唯一 provider 是：

```text
vllm_sipu.ops.backends.sikernel.jit.mxint8:quantize_to_mxint8
```

JIT provider 做三件事：

1. 确保输入是二维 contiguous tensor；
2. 根据 `(rows * cols / 1024) * 1088` 在 `input.device` 上分配一维 `uint8` output；
3. 调用运行时加载的 `quantization_mxint8` JIT module：

```python
get_mxint8_module().quantize_to_mxint8(input, packed)
```

`MXINT8_MODULE` 在 `vllm_sipu/ops/backends/sikernel/jit/mxint8.py:26-49` 注册。它把仓库内
的 C++ FFI 和 `.deps/sikernel-src` 中的 `hp_to_mx` 源文件一起编译/加载，并通过
`lru_cache(maxsize=1)` 在进程内复用已加载 module。

## 6.6 C++ FFI 层的检查与模板选择

C++ 入口位于：

```text
csrc/jit/quantization/mxint8.cpp:72-112
```

它检查：

- input 是二维 contiguous；output 是一维 contiguous；
- input/output 的 device type 和 device id 相同；
- rows 属于 8/16/32 alignment family，cols 为 128 的倍数；
- output 长度等于预期 packed bytes；
- output dtype 是 `uint8`；
- input dtype 是 BF16、FP16 或 FP32 之一。

本测试输入是 BF16，因此选择 `HPType=sifmt::bfloat16`，实际调用：

```cpp
hp_to_mx<sifmt::mxint8, sifmt::bfloat16, 0>(
    input_ptr,
    output_ptr,
    1,            // batch_size
    rows,         // dim0
    cols,         // dim1
    current_stream(input.device()));
```

省略的模板参数采用默认值：

```text
INPUT_LAYOUT = sipu::TMAP_FORMAT_TILED
ZERO_OUTPUT  = false
```

这与 Python 先做 `linear_to_tileformat()` 相匹配。若把普通 row-major buffer 直接传给该
模板实例，SiKernel 会按 tiled tensor map 错误解释数据。

## 6.7 SiKernel `hp_to_mx` host wrapper

公共声明位于：

```text
.deps/sikernel-src/include/sikernel.h:2282-2313
```

host wrapper 定义位于：

```text
.deps/sikernel-src/source/source_builtin/misc/hp_to_mx/kernel/
  hp_to_mx_kernel.su:26-87
```

它先校验 shape 和指针，然后根据 `dim0` 选择硬件 row geometry：

```text
dim0 > 16  -> Rows=32
dim0 > 8   -> Rows=16
其他       -> Rows=8
```

测试中的 padded rows 分别为 32 和 96，所以两者都选择 `Rows=32`。

对于 `HP_T=BF16`、`OUT_T=MXInt8`、`Rows=32`，traits 推导为：

```text
BF16 element bytes       = 2
input_tile_dim0          = 1024 / (32 * 2) = 16 columns
input_lmul               = 2
MX output_tile_dim0      = 2 * 16 = 32 columns
MX output tile data      = 32 * 32 * 1 byte = 1024 bytes
MX output tile header    = 64 bytes
MX physical tile         = 1088 bytes
```

因此 Python 输入侧用 `32 x 16` BF16 tiles 重排，而一次 MX 转换消费两个相邻的 BF16
input tiles，生成一个覆盖 `32 x 32` logical elements 的 MXInt8 output tile。两种 tile
尺寸描述的是不同阶段，不矛盾。

wrapper 随后：

1. 计算 row/output-column tile 数量及 supertile geometry；
2. 用 `siTensorMapEncodeTiled()` 编码输入 tensor map；
3. 根据 logical tile 数计算 grid/block；
4. 在当前 SIPU stream 上 launch `hp_to_mx_kernel`。

## 6.8 SiKernel device kernel

device kernel 定义位于：

```text
.deps/sikernel-src/source/source_builtin/misc/hp_to_mx/kernel/
  hp_to_mx_kernel.hpp:53-108
```

每个工作线程的主要数据流是：

```text
SIPU global tiled BF16 input
  -> DTE/tacp 搬到线程私有 shared/L2B staging
  -> tile register load
  -> HpToOutputTraits::convert<Rows>()
  -> tcvt_mxi8 hardware intrinsic
  -> MXInt8 register（element data + scale/header）
  -> 按 physical tile/supertile index 写入 global packed output
```

`HpToOutputTraits<sifmt::mxint8, HP_T>` 定义在
`hp_to_output_traits.hpp:132-164`，它指定 8-bit packed element、64-byte header 和
`tcvt_mxi8` intrinsic。kernel 将 logical `[batch,row_tile,col_tile]` 映射为硬件期望的
supertile-interleaved `physical_tile`，所以最终 output 不只是“数值转成 INT8”，还完成了
GEMM 所需的物理打包。

最终返回的 PyTorch tensor 是一维 `uint8`，其 bytes 同时包含：

- MXInt8 element data；
- 每 32 elements 的共享 scale；
- header padding；
- tile/supertile 物理排列。

它不是普通 `uint8` 数值数组，也不能用 `reshape(padded_rows,padded_cols)` 恢复矩阵。

## 6.9 `main()` 中实际触发的量化调用

`tests/kernels/quantization/test_mxint8_unpack_pytest.py` 当前参数为：

```text
M_VALUES   = [7]
N_VALUES   = [65]
K_VALUES   = [96]
SEQ_LENS   = [7]
dtype      = torch.bfloat16
device     = sipu:0
```

三个 test case 内部总共调用 `quantize_to_mxint8()` 5 次：

| test case | 被量化对象 | 原始 shape | 调用方式 |
| --- | --- | --- | --- |
| `test_quantize_to_mxint8` | `x` | `[7,96]` | 自动计算 shape |
| `test_mxint8_bf16_matmul` | `x` | `[7,96]` | 自动计算 shape |
| `test_mxint8_bf16_matmul` | `weight` | `[65,96]` | `for_weight=True` |
| `test_mxint8_linear_method` | `weight` | `[65,96]` | method 先算出并显式传入 `[96,128]` |
| `test_mxint8_linear_method` | activation `x` | `[7,96]` | method 显式传入 `[32,128]` |

因此只有两种唯一的 shape 变化：activation 的 `[7,96]` 路径和 weight 的 `[65,96]`
路径。bias `[65]` 不进入该函数，也不会被 MXInt8 量化。

## 6.10 Activation `[7,96]` 的完整变化表

当前环境实测结果如下：

| 阶段 | Python shape / stride | device | dtype | 语义 layout |
| --- | --- | --- | --- | --- |
| 原始 `x` | `[7,96]` / `(96,1)` | `sipu:0` | BF16 | linear row-major |
| `x_padded` | `[32,128]` / `(128,1)` | `sipu:0` | BF16 | linear row-major，右侧和下方补零 |
| `x_cpu` | `[32,128]` / `(128,1)` | CPU | BF16 | linear row-major |
| `x_tiled_cpu` | `[32,128]` / `(128,1)` | CPU | BF16 | tile-major，tile=`32x16` |
| `x_tiled` | `[32,128]` / `(128,1)` | `sipu:0` | BF16 | tile-major，tile=`32x16` |
| provider 预分配 `packed` | `[4352]` / `(1)` | `sipu:0` | `uint8` | 未初始化的一维 output storage |
| `hp_to_mx` 完成后 | `[4352]` / `(1)` | `sipu:0` | `uint8` | opaque MXInt8 tile/supertile packed |

大小计算为：

```text
原始 BF16       = 7 * 96 * 2       = 1344 bytes
padded BF16     = 32 * 128 * 2     = 8192 bytes
MXInt8 packed   = 32 * 128 / 1024 * 1088
                = 4 * 1088         = 4352 bytes
```

SiKernel geometry 为：

```text
row_tiles          = 32 / 32 = 1
output_col_tiles   = 128 / 32 = 4
logical MX tiles   = 1 * 4 = 4
physical MX tiles  = 4
output bytes       = 4 * 1088 = 4352
```

## 6.11 Weight `[65,96]` 的完整变化表

weight 路径的实测结果为：

| 阶段 | Python shape / stride | device | dtype | 语义 layout |
| --- | --- | --- | --- | --- |
| 原始 `weight` | `[65,96]` / `(96,1)` | `sipu:0` | BF16 | linear row-major |
| `x_padded` | `[96,128]` / `(128,1)` | `sipu:0` | BF16 | linear row-major，右侧和末尾 31 行补零 |
| `x_cpu` | `[96,128]` / `(128,1)` | CPU | BF16 | linear row-major |
| `x_tiled_cpu` | `[96,128]` / `(128,1)` | CPU | BF16 | tile-major，tile=`32x16` |
| `x_tiled` | `[96,128]` / `(128,1)` | `sipu:0` | BF16 | tile-major，tile=`32x16` |
| provider 预分配 `packed` | `[13056]` / `(1)` | `sipu:0` | `uint8` | 未初始化的一维 output storage |
| `hp_to_mx` 完成后 | `[13056]` / `(1)` | `sipu:0` | `uint8` | opaque MXInt8 tile/supertile packed |

大小计算为：

```text
原始 BF16       = 65 * 96 * 2      = 12480 bytes
padded BF16     = 96 * 128 * 2     = 24576 bytes
MXInt8 packed   = 96 * 128 / 1024 * 1088
                = 12 * 1088        = 13056 bytes
```

SiKernel geometry 为：

```text
row_tiles          = 96 / 32 = 3
output_col_tiles   = 128 / 32 = 4
logical MX tiles   = 3 * 4 = 12
physical MX tiles  = 12
output bytes       = 12 * 1088 = 13056
```

这里能看出小 shape 的 padding 成本：packed weight `13056 bytes` 甚至略大于原始 BF16
weight 的 `12480 bytes`。这不是 MXInt8 element 本身比 BF16 大，而是 `[65,96]` 必须先
扩展到 `[96,128]`。

## 6.12 临时 tensor 与 device 往返

以当前未对齐输入为例，量化期间可能同时存在：

```text
SIPU: 原始 x + x_padded + x_tiled + packed
CPU : x_cpu + x_tiled_cpu
```

`build_mxint8_tile_tensor_on_cpu()` 中再次调用 `pad_to_mxint8_shape()`，但传入 tensor 已经
是目标 shape 且 contiguous，所以该次通常直接返回 `x_cpu`，不会再创建第二个 padded
CPU tensor。真正的新 CPU layout buffer来自 `permute(...).contiguous()`。

函数返回后只保留 `packed` 和 Python tuple `padded_shape`；其余临时 tensor 可释放，但
内存可能仍留在 PyTorch allocator cache 中。因而：

- 最终 packed storage 大小不等于量化过程峰值内存；
- activation 每次 forward 都产生这套临时量和 SIPU/CPU 往返；
- weight 通常仅在 model post-load 阶段执行一次该过程。

## 6.13 调用链总结

完整调用链为：

```text
mxint8_utils.quantize_to_mxint8
  -> get_padded_mxint8_shape
  -> pad_to_mxint8_shape                    # SIPU linear BF16
  -> build_mxint8_tile_tensor_on_cpu
     -> SIPU -> CPU
     -> linear_to_tileformat                # CPU tiled BF16
  -> CPU -> SIPU                            # SIPU tiled BF16
  -> vllm_sipu.ops.quantization.quantize_to_mxint8
     -> operator _dispatch
     -> JIT SiKernel provider quantize_to_mxint8
        -> allocate 1D SIPU uint8 packed buffer
        -> quantization_mxint8 JIT module
           -> C++ FFI quantize_to_mxint8
              -> hp_to_mx<sifmt::mxint8, sifmt::bfloat16, 0>
                 -> encode tiled tensor map
                 -> hp_to_mx_kernel<..., Rows=32, ...>
                    -> DTE load
                    -> tcvt_mxi8
                    -> tiled/supertile packed store
```

在指定 conda 和 SDK/CModel 环境中重新运行：

```bash
source /share_data/users/like/miniconda3/etc/profile.d/conda.sh
conda activate /share_data/users/like/miniconda3/envs/vllm_dev
source ./sipu_sdk_setup.sh
python tests/kernels/quantization/test_mxint8_unpack_pytest.py
```

结果为 `All 3 MXInt8 test cases passed.`，退出码为 `0`。这验证了上述两种 shape 的
quantize、GEMM 和 Linear 集成路径在当前单设备 CModel 环境中可执行。

# 7. `build_mxint8_tile_tensor_on_cpu` 为什么要先 copy 到 CPU 再做 `linear_to_tileformat`

## 7.1 直接回答

`linear_to_tileformat()` 的核心操作是：

```python
x.reshape(num_tile_m, 32, num_tile_n, 16)
 .permute(0, 2, 1, 3)
 .contiguous()
 .reshape(m, n)
```

这些全是标准 PyTorch op（reshape、permute、contiguous），在 `sipu` tensor 上**能正常执行、结果正确**。实验验证了这一点：对 `[32, 32]` 和 `[96, 128]` 的 BF16 tensor，直接在 `sipu` 上调用 `linear_to_tileformat`，输出与 CPU 路径**逐 byte 完全一致**。后续经过 `hp_to_mx` 量化后，packed MXInt8 结果也**完全相同**。

所以搬 CPU **不是因为 `sipu` 上不能做**，而是出于性能和工程考虑。

## 7.2 性能实测：CPU 路径反而更快

在 CModel 环境中对 `[96, 128]` BF16 tensor benchmark（20 次取平均）：

| 路径 | 耗时 |
| --- | --- |
| 直接在 `sipu` 上做 tile reorder | 83.24 ms |
| copy 到 CPU + CPU tile reorder + copy 回 `sipu` | 1.70 ms |

CPU 路径快了约 **49 倍**。

## 7.3 为什么 CPU 更快

三个原因叠加：

1. **CModel 是纯软件仿真器。** 每一条 `sipu` kernel 都被 host 端逐指令模拟，设备端 `.contiguous()`（实质是一个 permuted read + linear write 的 element-wise copy kernel）在 CModel 上的开销远大于原生 CPU memcpy。这在物理硬件上差距会缩小，但下面两个因素在物理硬件上仍然成立。

2. **tile reorder 是纯内存搬运、不做计算。** `.permute(0, 2, 1, 3).contiguous()` 本质是按新顺序读、按线性顺序写——它是 memory-bandwidth-bound 的 copy 操作，不涉及任何浮点计算。加速器的优势在大规模并行计算，对这种 memory-bound copy 的加速效果有限，而 CPU 本地内存的延迟和带宽对小/中尺寸 tensor 已经足够。

3. **避免不必要的设备 kernel launch 和同步开销。** 在设备上做 reorder 意味着额外一次 kernel launch + 可能的 stream 同步。对 weight（只执行一次）影响不大，但对 activation（每次 forward 都执行）这些固定开销会累积。

## 7.4 工程层面的考量

除了性能，还有两个实际因素：

- **与 `hp_to_mx` 的 tensor map 对齐。** `hp_to_mx` 的 C++ FFI 使用 `TMAP_FORMAT_TILED` 模板参数，即假设输入已按 tiled layout 排列。在 CPU 上用确定性的 Python 代码完成 tile reorder，可以保证无论 `sipu` 的 `.contiguous()` 实现细节如何变化，传给 `hp_to_mx` 的字节顺序始终一致。如果在设备上做 reorder，就需要假设设备 `.contiguous()` 的行为与 CPU 完全一致——虽然目前确实一致，但多了一层隐式依赖。

- **CModel 与物理硬件的一致性。** 当前开发和测试大量依赖 CModel。在 CPU 上做 reorder 消除了 CModel vs 物理硬件在 `.contiguous()` kernel 实现上可能存在的行为差异，使 tile layout 转换在所有环境下表现相同。

## 7.5 如果改成直接在 `sipu` 上做会怎样

功能上完全可行。只需把 `build_mxint8_tile_tensor_on_cpu` 改为：

```python
def build_mxint8_tile_tensor_on_device(
    x: torch.Tensor,
    padded_shape: tuple[int, int],
) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got {x.shape}")
    x_padded = pad_to_mxint8_shape(x, padded_shape)
    return linear_to_tileformat(
        x_padded,
        m_alignment_elem=MXINT8_TILE_ROWS,
        n_alignment_bytes=MXINT8_TILE_BYTES,
    )
```

即可省去 `sipu` → CPU → `sipu` 的两次 copy。实测 packed 结果逐 byte 一致。

但在当前环境下这样做**会变慢**（CModel 上约慢 50×）。在物理硬件上需要重新 benchmark，才能量化省去 copy 后设备 kernel launch 开销与设备 reorder kernel 效率之间的净收益。

## 7.6 一句话结论

搬 CPU 不是因为 `sipu` 上做不了或会出错，而是因为 tile reorder 是纯内存搬运、不做计算，在 CPU 上做更快（CModel 实测快 49 倍），同时避免了设备 `.contiguous()` 实现的隐式依赖。

# 8. SIMO `SIMOLinearMethod.apply` 为什么会调用 CPU downcast 实现

## 8.1 直接结论

在 SIMO 的这条路径中，`input_2d` 确实先位于 `sipu:0`。但是
`self.input_downcast_kernel` 不是直接调用某个 Python CPU 函数，而是
`torch.ops.simo.downcast_to_mxfmt` 的 wrapper：

```text
SIMOLinearMethod.__init__                         # quantization_method.py:152-157
  -> get_downcast_kernel(input_spec, ...)         # kernels.py:48-88
     -> lambda src: torch.ops.simo.downcast_to_mxfmt(src, ...)
```

SIMO 给这个 custom op 注册了 `CUDA` 和 `CPU` 两个 dispatch implementation，但没有
注册 `PrivateUse1`/`SIPU` implementation：

```text
simo::downcast_to_mxfmt
  CPU       -> downcast_to_mxfmt_cpu_impl
  CUDA      -> downcast_to_mxfmt_cuda_impl
  Meta      -> fake implementation
  PrivateUse1/SIPU -> 没有专用 kernel
```

因此，`downcast_to_mxfmt_cpu_impl` 不是因为 Python 看到 `input_2d.device == cpu` 才被
选择，而是因为 SIPU 的 dispatch key 没有对应 kernel，触发了 SIPU 的 CPU fallback。

## 8.2 为什么 SIPU 对应 `PrivateUse1`

当前 conda 环境的 `torch_sipu/__init__.py:35-37` 执行：

```python
torch._register_device_module("sipu", torch_sipu.sipu)
torch.utils.rename_privateuse1_backend("sipu")
torch.utils.generate_methods_for_privateuse1_backend()
```

所以用户看到的 `torch.device("sipu:0")`，底层 dispatcher key 仍然是
`PrivateUse1`。而 `simo/ops/mx_api.py:364-378` 注册 downcast 时只有：

```python
direct_register_custom_op(..., dispatch_key="CUDA")
direct_register_custom_op(..., dispatch_key="CPU")
```

`direct_register_custom_op()` 的实现 `simo/ops/torch_utils.py:13-55` 会把函数直接挂到
指定 dispatch key；它不会把 `CPU` implementation 自动变成 SIPU implementation。

在当前运行环境打印 dispatcher 表，实际结果是：

```text
CPU:       registered
CUDA:      registered
Meta:      registered
PrivateUse1: False
```

## 8.3 `src_tensor` 为什么在进入 CPU 函数时已经是 CPU

torch-sipu 为缺少 SIPU kernel 的算子安装了 boxed fallback。其头文件
`torch_sipu/include/torch_sipu/csrc/aten/native/sipu/aten_fallback.h:270-276` 中，
`sipu_fallback_op()` 最终调用：

```cpp
at::native::cpu_fallback(op, stack);
```

PyTorch 的 `$CONDA_PREFIX/lib/python3.10/site-packages/torch/include/ATen/native/CPUFallback.h:13-16`
将它定义为“boxed fallback to CPU”。这个 fallback 在 custom-op 的边界处理 boxed
参数：

```text
SIPU tensor 参数
  -> copy 到 CPU tensor
  -> 调用该 op 的 CPU dispatch implementation
  -> 得到 CPU 输出
  -> 按原始设备把输出 copy 回 SIPU
```

因此，搬运发生在进入 Python 函数
`downcast_to_mxfmt_cpu_impl()` 之前，函数本身并没有显式写
`src_tensor = src_tensor.to("cpu")`。函数第 280 行的 debug 日志看到的已经是 fallback
生成的 CPU tensor。

## 8.4 当前日志中的完整证据

给定日志
`temp/simo.log.2026_08_21___16_29_43` 在同一次 `SIMOLinearMethod.apply` 中记录了：

```text
quantization_method.py:487  input_2d device:sipu:0
mx_api.py:279              downcast_to_mxfmt_cpu_impl ... src_tensor device:cpu
quantization_method.py:505  input_qdevice:sipu:0, input_scale.device:sipu:0
```

对应日志行是 `:180-184`，后续其他 Linear 层也重复同样模式，例如 `:218-221`。
这三行正好对应 fallback 的三个阶段：

```text
apply 中的原始输入       sipu:0
CPU implementation 内部  cpu
custom op 返回给 apply   sipu:0
```

## 8.5 为什么返回结果又自动回到 SIPU

`downcast_to_mxfmt_cpu_impl()` 的主体在 `simo/ops/mx_api.py:287-317` 做 transpose、
可选 Hadamard transform，并在 `:308-315` 调用 `_downcast_to_mxfmt_torch` 参考实现。
这些计算在 CPU fallback 阶段都使用 CPU tensor，函数返回的 `quantized` 和 `scale` 也
首先是 CPU tensor。

`cpu_fallback` 返回 boxed 结果时会根据原始 SIPU 调用的设备语义恢复 tensor 设备，因而
Python 调用者拿到的是 `sipu:0` 上的结果。随后 `SIMOLinearMethod.apply()` 才能继续：

```python
# quantization_method.py:489-505
input_q, input_scale = self.input_downcast_kernel(input_2d)
# input_q.device == input_scale.device == sipu:0
```

这并不表示 downcast 的量化计算在 SIPU 上完成；它表示“CPU fallback 计算结束后，结果
被复制回 SIPU”，以便后续的 SIMO GEMM backend 或 QDQ 路径继续使用设备 tensor。

## 8.6 最小实验：CPU implementation 内部看到 CPU，调用者仍拿到 SIPU

在相同 conda/SIPU SDK 环境中注册一个只提供 CPU implementation 的最小 custom op，
其 CPU 函数只打印输入设备并执行 `x + 1`。用 SIPU tensor 调用的输出为：

```text
before_device= sipu:0
[Fallback](Fallback Operator) function: simo::probe_cpu_fallback
inside_cpu_impl_device= cpu
after_device= sipu:0 value=tensor([2., 2.])
```

这个实验不依赖 SIMO 的量化算法，直接证明了当前 torch-sipu fallback 的设备行为：
缺少 `PrivateUse1` kernel 时，CPU 函数参数会自动变成 CPU，结果再自动返回原始 SIPU
设备。给定 SIMO 日志中的 `src_tensor device:cpu` 和返回后的两个 `sipu:0` 输出，属于
完全相同的机制。

## 8.7 重要区分

- `self.input_downcast_kernel` 选择的是 `torch.ops.simo.downcast_to_mxfmt`；不是
  `SIMOLinearMethod` 主动硬编码调用 `downcast_to_mxfmt_cpu_impl`。
- `downcast_to_mxfmt_cpu_impl` 被调用，是因为当前 custom op 没有 SIPU/PrivateUse1
  kernel，SIPU fallback 选择了 CPU implementation。
- CPU fallback 只保证兼容性和正确的设备语义，会引入每次 activation quantization 的
  SIPU -> CPU -> SIPU 数据往返；它不是 SIPU 原生 downcast kernel。
- 如果关闭该 fallback 而仍不注册 `PrivateUse1` kernel，dispatcher 将无法为 SIPU
  tensor 找到可执行的实现，通常会直接报缺少该 backend kernel，而不会自动调用 CPU
  Python 函数。

因此，本次现象可以准确表述为：**输入在 SIPU，custom op 因缺少 SIPU kernel 进入
torch-sipu 的 CPU fallback；CPU fallback 把参数复制到 CPU 后调用
`downcast_to_mxfmt_cpu_impl`，再把 CPU 结果复制回 SIPU。**

# 9. 可复现实验脚本与 SIPU custom-op dispatch key

## 9.1 实验脚本

第 8.6 节的最小实验已写入：

`like-useful/demo_fallback.py`

脚本只注册一个 `CPU` implementation，不修改 SIMO 源码。它创建一个 `sipu:0`
输入，调用 custom op，并在 CPU implementation 内部打印输入设备：

```python
def probe_cpu_fallback(src_tensor: torch.Tensor) -> torch.Tensor:
    print(f"inside_cpu_impl_device= {src_tensor.device}", flush=True)
    return src_tensor + 1

direct_register_custom_op(
    op_name="probe_cpu_fallback",
    op_func=probe_cpu_fallback,
    dispatch_key="CPU",
)
```

在仓库根目录使用题目指定的 conda 环境和 SDK 运行：

```bash
source /share_data/users/like/miniconda3/etc/profile.d/conda.sh
conda activate /share_data/users/like/miniconda3/envs/vllm_dev
source ./sipu_sdk_setup.sh
python like-useful/demo_fallback.py
```

本次实际输出为：

```text
before_device= sipu:0
[Fallback](Fallback Operator) function: simo::probe_cpu_fallback
inside_cpu_impl_device= cpu
after_device= sipu:0 value= tensor([2., 2.])
```

它复现了第 8 节的关键现象：CPU implementation 内看到的是 CPU，调用者收到的结果
又位于 SIPU。

## 9.2 支持 SIPU 的 implementation 应使用哪个 key

推荐使用 PyTorch 的 canonical dispatch key：

```python
direct_register_custom_op(
    op_name="my_sipu_op",
    op_func=my_sipu_impl,
    dispatch_key="PrivateUse1",
)
```

原因是 `torch_sipu/__init__.py:35-37` 使用
`torch.utils.rename_privateuse1_backend("sipu")` 将设备名称 `sipu` 映射到底层
`DispatchKey.PrivateUse1`。所以：

```text
torch.device("sipu:0")  ->  dispatcher key PrivateUse1
```

在相同环境中实测，注册 `dispatch_key="PrivateUse1"` 后，dispatcher 表显示：

```text
PrivateUse1: registered
```

调用函数时 implementation 内收到的 tensor 是 `sipu:0`，不会经过 CPU fallback，返回
结果也保持 `sipu:0`。

当前环境还接受大写别名：

```python
dispatch_key="SIPU"
```

`torch._C._dispatch_key_parse("SIPU")` 会解析为 `DispatchKey.PrivateUse1`，实测行为
与 `"PrivateUse1"` 相同。但小写：

```python
dispatch_key="sipu"
```

在当前 PyTorch 2.10.0+cpu/torch-sipu 环境中会报：

```text
RuntimeError: could not parse dispatch key: sipu
```

所以跨环境代码应使用 `"PrivateUse1"`；`"SIPU"` 只是当前 backend rename 后可用的
别名，不应依赖它。

## 9.3 注册 SIPU implementation 后的语义

如果为 `downcast_to_mxfmt` 额外注册 `PrivateUse1` implementation，dispatcher 会直接
选择该实现：

```text
sipu tensor
  -> PrivateUse1 kernel
  -> sipu result
```

不会再自动调用 `downcast_to_mxfmt_cpu_impl`，也不会自动进行 SIPU -> CPU -> SIPU
往返。因此这个实现必须真正接受和处理 SIPU tensor，并在需要创建输出时把输出放在
SIPU 上；注册 key 本身不会把一个 CPU-only 函数转换成 SIPU kernel。

如果算子需要支持 `torch.compile`/fake tensor，还应继续提供 `fake_impl`；这与运行时
设备 dispatch key 是两个独立问题：

```text
PrivateUse1 implementation  -> 真实 SIPU 执行
fake_impl                   -> Meta/fake tracing 与 shape 推导
```

最终答案：**在 `direct_register_custom_op` 中注册支持 SIPU 的真实实现，使用
`dispatch_key="PrivateUse1"`；不要使用小写 `"sipu"`。当前环境的大写 `"SIPU"` 可作为
别名，但不如 `"PrivateUse1"` 稳定、明确。**

# 10. SIPU custom op 示例与 `fake_impl` 的作用

## 10.1 可运行示例

已新增脚本 [like-useful/demo_register_sipu.py](/share/users/like/package/vllm-sipu/like-useful/demo_register_sipu.py:1)。它用 `direct_register_custom_op` 注册一个名为 `simo::demo_register_sipu` 的自定义算子：

```python
direct_register_custom_op(
    op_name="demo_register_sipu",
    op_func=sipu_add_one,
    fake_impl=fake_add_one,
    dispatch_key="PrivateUse1",
)
```

其中 `sipu_add_one` 是真实运行时实现，执行 `src_tensor + 1`；`fake_add_one` 只根据
输入返回同形状的空 tensor，用于只需要元数据的 fake/meta 执行。

使用题目指定的环境运行：

```bash
source /share_data/users/like/miniconda3/etc/profile.d/conda.sh
conda activate /share_data/users/like/miniconda3/envs/vllm_dev
source ./sipu_sdk_setup.sh
python -u like-useful/demo_register_sipu.py
```

实际输出（省略 SDK 启动信息）：

```text
inside_sipu_impl_device= sipu:0
real_input_device= sipu:0 real_output_device= sipu:0 value= tensor([2., 2.])
inside_fake_impl_device= meta
meta_input_device= meta fake_output_device= meta fake_output_shape= (2,)
inside_fake_impl_device= cpu
fake_input_type= FakeTensor fake_output_type= FakeTensor fake_output_device= cpu
```

第一段说明 SIPU tensor 由 `PrivateUse1` runtime implementation 执行，结果仍在
`sipu:0`。第二段显式使用 Meta tensor，第三段在 `FakeTensorMode` 中运行；这两段都
调用 `fake_impl`，不会启动真实 SIPU kernel。FakeTensorMode 中显示 `cpu` 是 fake
tensor 的代理 device 元数据，不代表发生了 CPU 实际计算。

## 10.2 为什么需要 `fake_impl`

`direct_register_custom_op` 做了两类注册：

1. `my_lib.impl(op_name, op_func, dispatch_key="PrivateUse1")` 注册真实设备实现。
2. 提供 `fake_impl` 时，调用 `my_lib._register_fake(op_name, fake_impl)` 注册 Meta/fake 实现。

PyTorch 的 `torch.compile`、`torch.export`、AOTAutograd、FakeTensorMode 等流程通常
先用没有真实存储的 Meta/Fake tensor 做 tracing 和 shape propagation。此时 dispatcher
不会调用 SIPU runtime kernel，而是寻找 Meta/fake kernel；因此自定义算子需要
`fake_impl` 来描述输出的结构、shape、dtype 和 device 等元数据。

在普通 eager 路径中传入真实 `sipu:0` tensor 时，只会调用 `sipu_add_one`，不会调用
`fake_add_one`。如果不注册 `fake_impl`，真实 SIPU eager 调用仍可能成功，但 Meta
或 fake tensor 调用会失败。本环境的实际错误是：

```text
NotImplementedError: simo::demo_no_fake: attempted to run this operator with Meta tensors,
but there was no fake impl or Meta kernel registered.
```

## 10.3 `fake_impl` 的编写要求

`fake_impl` 不应读取真实数据或发起设备 kernel，而应使用 fake-compatible 的 PyTorch
操作（示例中的 `torch.empty_like`），并满足以下约束：

- 返回值的数量和嵌套结构与真实实现一致；
- 输出 shape、dtype、device 与真实实现的规则一致；
- 只依赖输入的尺寸、dtype、device 等可追踪元数据；
- 若输出 shape 依赖输入数值，需改写为可被 symbolic/fake tracing 处理的逻辑。

因此，这里的 `fake_impl` 是编译/导出阶段的“形状与元数据实现”，不是 SIPU
runtime implementation 的替代品。注册 SIPU 算子时应同时提供真实的
`dispatch_key="PrivateUse1"` implementation 和与其语义匹配的 `fake_impl`。
