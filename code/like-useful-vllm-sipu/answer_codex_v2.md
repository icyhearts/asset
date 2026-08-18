# Qwen2.5 recipe 中 `quantization` 传给 vLLM Engine 的解析链路

## 1. 结论

当前 recipe 中的字段是 `llm.quantization`：

```json
"llm": {
  "load_format": "sipu_dummy",
  "quantization": "mxint8"
}
```

它不会写入生成的模型 `config.json`，而是作为 Python 关键字参数直接传给
`vllm.LLM`。在本次环境的 vLLM `0.27.1+sipu1` 中，完整链路是：

```text
recipe["llm"]["quantization"] == "mxint8"
  -> llm_kwargs["quantization"]
  -> LLM(quantization="mxint8")
  -> EngineArgs.quantization
  -> ModelConfig.quantization
  -> vLLM quantization registry 查询 "mxint8"
  -> VllmConfig.quant_config = MXInt8Config(ignored_layers=[])
  -> Qwen2 各 LinearBase.quant_method = MXInt8LinearMethod
  -> load 后 pack 权重，forward 时量化 activation 并执行 MXInt8 GEMM
```

本次最终的几个容易混淆的对象如下：

| 对象 | 本次值 | 含义 |
| --- | --- | --- |
| `EngineArgs.quantization` | `"mxint8"` | 量化方法选择字符串 |
| `ModelConfig.quantization` | `"mxint8"` | 经过模型配置校验后的方法名 |
| `ModelConfig.quantization_config` | `None` | vLLM 面向 online/per-layer 量化的用户配置，本次未使用 |
| `VllmConfig.quant_config` | `MXInt8Config(ignored_layers=[])` | registry 解析后真正传给模型和 layer 的配置对象 |

因此日志中的 `quantization=mxint8, quantization_config=None` 是正常结果，后者为
`None` 不代表 MXInt8 没有生效。

## 2. Recipe 如何变成 `LLM` 参数

### 2.1 JSON 加载和原样复制

recipe 的定义位于：

```text
examples/recipes/qwen/qwen25_05b_instruct_dummy___mxint8.json:13-24
```

其中第 22 行是：

```json
"quantization": "mxint8"
```

runner 在 `examples/offline/offline_inference_recipe.py:218-223` 使用
`json.load()` 读取 JSON。在 `build_llm_kwargs()` 中：

```python
# examples/offline/offline_inference_recipe.py:242-246
raw_llm = recipe.get("llm", {})
llm_kwargs = dict(raw_llm)
```

所以 `recipe["llm"]` 中的 `quantization` 被原样复制为：

```python
llm_kwargs["quantization"] == "mxint8"
```

### 2.2 CLI 覆盖优先级

`offline_inference_recipe.py:257-268` 对参数进行覆盖。对于 quantization，优先级从低
到高是：

```text
recipe 的 llm.quantization
  < --quantization VALUE
  < --llm-arg quantization=JSON_VALUE
```

本次命令没有传 `--quantization` 或相应的 `--llm-arg`，所以 recipe 中的
`"mxint8"` 保持不变。

### 2.3 `model_config` overlay 不承载该字段

recipe 的顶层 `model_config.overrides` 用于修改 Hugging Face 模型的
`config.json`。`offline_inference_recipe.py:328-382` 创建 overlay 后，只把：

```python
llm_kwargs["model"] = str(overlay_dir)
```

替换为 overlay 路径，不会修改 `llm_kwargs["quantization"]`。实际生成的：

```text
logs/model_smoke/qwen25_05b_instruct_dummy_20260817_145546_model/config.json
```

只有 Qwen2 模型结构和 BF16 dtype 等配置，没有 `quantization_config`。这说明本次
MXInt8 的入口是 `LLM` 参数，不是 checkpoint/HF config 自动识别。

runner 在 `offline_inference_recipe.py:584-597` 先应用环境变量，再导入 vLLM，最终
执行：

```python
llm = LLM(**llm_kwargs)
```

关键部分等价于：

```python
LLM(
    model="logs/model_smoke/qwen25_05b_instruct_dummy_20260817_145546_model",
    load_format="sipu_dummy",
    quantization="mxint8",
    # 其他 recipe 参数省略
)
```

先设置 `VLLM_PLUGINS=sipu,general`、后导入 vLLM 也很重要，它保证 SIPU platform
plugin 可以参与后续配置构造。

## 3. `LLM` 如何传到 Engine

本次 conda 环境中的上游源码目录为：

```text
/share_data/users/like/miniconda3/envs/vllm_dev/lib/python3.10/site-packages/vllm
```

下文将其简称为 `$VLLM_PKG`。

### 3.1 `LLM` 构造 `EngineArgs`

`$VLLM_PKG/entrypoints/llm.py:177-222` 的 `LLM.__init__()` 显式接收两个不同参数：

```python
quantization: QuantizationMethods | None = None
quantization_config: dict[str, Any] | QuantizationConfigArgs | None = None
```

随后在同文件 `:295-335` 构造 `EngineArgs`：

```python
engine_args = EngineArgs(
    ...,
    quantization=quantization,                 # :307
    ...,
    quantization_config=quantization_config,   # :329
)
```

因此此时：

```text
EngineArgs.quantization        = "mxint8"
EngineArgs.quantization_config = None
```

`LLM` 在 `$VLLM_PKG/entrypoints/llm.py:339-341` 调用
`LLMEngine.from_engine_args()`；V1 engine 在
`$VLLM_PKG/v1/engine/llm_engine.py:160-180` 继续调用
`engine_args.create_engine_config()`。

### 3.2 为什么 `quantization_config` 仍为 `None`

`EngineArgs.__post_init__()` 位于 `$VLLM_PKG/engine/arg_utils.py:755-795`，其中
`:786-790` 调用：

```python
self.quantization_config = resolve_quantization_config(
    self.quantization, self.quantization_config
)
```

`resolve_quantization_config()` 位于
`$VLLM_PKG/config/quantization.py:153-189`。它只会把 vLLM 定义的 online shorthand
（例如 `fp8_per_tensor`、`fp8_per_block`、`mxfp8`）展开成
`QuantizationConfigArgs`。

`mxint8` 是 SIPU 注册的自定义 quantization method，不属于 online
shorthand。因此函数在 `:165-172` 直接返回 `None`，同时保留独立的
`EngineArgs.quantization == "mxint8"`。

这也是为什么不能把日志中的两个字段理解成同一件事：

```text
quantization=mxint8          # 方法选择有效
quantization_config=None     # 没有使用 online/per-layer 用户配置
```

当前 vLLM 还会拒绝把非 online 方法和 `LLM(quantization_config=...)` 组合使用；对应
检查也在 `$VLLM_PKG/config/quantization.py:165-171`。

## 4. SIPU 如何注册 `mxint8`

### 4.1 Platform plugin 入口

repo 的 `pyproject.toml:47-51` 注册了：

```toml
[project.entry-points."vllm.platform_plugins"]
sipu = "vllm_sipu:register"

[project.entry-points."vllm.general_plugins"]
general = "vllm_sipu:register_general_plugin"
```

日志 `temp/qwen25_05b_instruct_dummy___mxint8.json.log.2026_08_17___14_55_46:5-15`
也显示 vLLM 发现并激活了 `sipu -> vllm_sipu:register`。

### 4.2 必须在 `ModelConfig` 校验前注册

`EngineArgs.create_engine_config()` 在
`$VLLM_PKG/engine/arg_utils.py:1896-1930` 中先执行：

```python
current_platform.pre_register_and_update()  # :1906
model_config = self.create_model_config()   # :1929
```

SIPU 的实现位于 `vllm_sipu/platform.py:184-195`。它在第 188 行导入：

```python
import vllm_sipu.model_executor.layers.quantization
```

随后发生以下导入和注册：

```text
vllm_sipu/model_executor/layers/quantization/__init__.py:17-21
  -> import mxint8
     -> vllm_sipu/model_executor/layers/quantization/mxint8.py:49-50
        -> @register_quantization_config("mxint8")
        -> class MXInt8Config(QuantizationConfig)
```

vLLM registry 的装饰器实现在
`$VLLM_PKG/model_executor/layers/quantization/__init__.py:59-106`。对于新的方法名，它会：

1. 把 `mxint8` 加入运行时 `QUANTIZATION_METHODS`；
2. 写入 `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG["mxint8"] = MXInt8Config`；
3. 必要时同步 platform 的 supported quantization 列表。

`get_quantization_config()` 在同文件 `:109-183` 查询配置类，并在 `:180-183` 合并
上述自定义映射。因此最终 registry 关系为：

```text
"mxint8" -> vllm_sipu...mxint8.MXInt8Config
```

如果这一步没有在 `ModelConfig` 创建前发生，后续会报
`Unknown quantization method: mxint8`。

## 5. Engine 如何解析成 `MXInt8Config`

### 5.1 `ModelConfig` 保存并校验方法名

`EngineArgs.create_model_config()` 在 `$VLLM_PKG/engine/arg_utils.py:1676-1708`
继续传递：

```python
ModelConfig(
    ...,
    quantization=self.quantization,                 # :1706
    quantization_config=self.quantization_config,   # :1707
)
```

`ModelConfig` post-init 在 `$VLLM_PKG/config/model.py:806-810` 调用
`_verify_quantization()`，实现位于同文件 `:1119-1206`。

在校验前，同文件 `:872-877` 的 field validator 会对字符串执行 `.lower()`；本例
本来就是小写 `mxint8`，所以值不变。

本次 overlay 的 HF `config.json` 不含 `quantization_config`，所以不会从 checkpoint
自动推导方法，也不存在 checkpoint 方法与显式 `mxint8` 不一致的问题。代码只会在
`:1200-1206` 检查：

1. `mxint8` 是否已经注册到 `QUANTIZATION_METHODS`；
2. 当前 SIPU platform 是否接受该方法。

检查完成后：

```text
ModelConfig.quantization        = "mxint8"
ModelConfig.quantization_config = None
```

若 checkpoint 自带 `quantization_config.quant_method`，则同文件 `:1124-1198` 还会做
自动识别和一致性校验；显式参数与 checkpoint 方法不一致会直接报错。

### 5.2 `VllmConfig` 实例化内部配置对象

`$VLLM_PKG/engine/arg_utils.py:2464-2493` 创建 `VllmConfig`。其 post-init 在
`$VLLM_PKG/config/vllm.py:1028-1031` 发现内部 `quant_config` 尚未设置，于是调用：

```python
VllmConfig._get_quantization_config(model_config, load_config)
```

该函数位于 `$VLLM_PKG/config/vllm.py:705-739`，并在 `:712-715` 进入
`weight_utils.get_quant_config()`。后者位于
`$VLLM_PKG/model_executor/model_loader/weight_utils.py:240-394`：

1. `:243-245` 使用 `model_config.quantization == "mxint8"` 查询 registry，得到
   `MXInt8Config` 类；
2. `:247-331` 尝试从 HF `quantization_config`、`compression_config`、
   `hf_overrides` 或 online config 读取额外配置，本次都没有；
3. `MXInt8Config.get_config_filenames()` 在 repo 的 `mxint8.py:67-69` 返回空列表；
4. 所以上游 `weight_utils.py:360-364` 直接执行 `return quant_cls()`。

最终结果是：

```python
vllm_config.quant_config = MXInt8Config(ignored_layers=[])
```

`VllmConfig._get_quantization_config()` 还会检查硬件 capability 和 activation dtype。
repo 的 `mxint8.py:59-65` 声明只支持 `torch.bfloat16`，最低 capability 为 0；本次
overlay 模型的 dtype 是 BF16，因此检查通过。

## 6. 配置对象如何落到 Qwen2 Linear

Qwen2 模型在 `$VLLM_PKG/model_executor/models/qwen2.py:343-381` 从
`vllm_config.quant_config` 取出 `MXInt8Config`，再传给各 decoder layer。MLP 的
`gate_up_proj`、`down_proj` 位于同文件 `:80-103`，attention 的 `qkv_proj`、
`o_proj` 位于 `:156-170`。

每个 `LinearBase` 在 `$VLLM_PKG/model_executor/layers/linear.py:272-281` 调用：

```python
quant_config.get_quant_method(self, prefix=prefix)
```

repo 的 `vllm_sipu/model_executor/layers/quantization/mxint8.py:78-95` 按 layer 类型返回：

| Layer | 结果 |
| --- | --- |
| 未命中 `ignored_layers` 的 `LinearBase` | `MXInt8LinearMethod` |
| 命中 `ignored_layers` 的 `LinearBase` | `UnquantizedLinearMethod` |
| `RoutedExperts` | `SIPUUnsupportedMoEMethod`，当前 MXInt8 MoE 路径未接通 |
| 其他 layer | `None`，MXInt8 config 不接管 |

因此，本例的 QKV、attention output、MLP gate/up 和 down projection 都绑定
`MXInt8LinearMethod`；这不是把整个模型的每种算子都改成 MXInt8。

## 7. `sipu_dummy` 加载与 MXInt8 的关系

`load_format="sipu_dummy"` 和 `quantization="mxint8"` 是两条独立配置：

- `sipu_dummy` 选择权重 loader；
- `mxint8` 选择 Linear 的 quantization method。

`SipuDummyModelLoader` 位于
`vllm_sipu/model_executor/model_loader/sipu_loader.py:191-248`。它先创建普通二维 BF16
权重并填入随机值，移动到 SIPU 后，在第 245 行调用通用
`process_weights_after_loading()`。

上游 `$VLLM_PKG/model_executor/model_loader/utils.py:100-119` 遍历所有模块，并调用每个
`quant_method.process_weights_after_loading()`。对 MXInt8 来说：

```text
mxint8.py:103-133  先创建可供 loader 填充的普通二维权重
mxint8.py:135-153  load 后 padding、量化、pack，并替换为 uint8 packed weight
mxint8.py:155-169  forward 时进入 apply_mxint8_linear()
```

`vllm_sipu/model_executor/layers/quantization/utils/mxint8_utils.py:183-205` 的 forward
数据流为：

```text
BF16 activation
  -> padding + 在线 MXInt8 pack
  -> packed activation x packed weight
  -> mxint8_bf16_matmul
  -> 裁剪 padding、加 bias
  -> BF16 output
```

也就是说，本例 dense Linear 是 W8A8，权重只在 load 后 pack 一次，activation 则在
每次 forward 中动态 pack。

## 8. 本次日志证据

日志文件：

```text
temp/qwen25_05b_instruct_dummy___mxint8.json.log.2026_08_17___14_55_46
```

关键证据如下：

1. `:5-15`：发现并激活 SIPU platform plugin；
2. `:19`：`non-default args` 明确包含 `'quantization': 'mxint8'`；
3. `:57`：V1 Engine 配置明确显示 vLLM `0.27.1+sipu1`、
   `quantization=mxint8`、`quantization_config=None`、BF16 和 SIPU device；
4. `:61-66`：模型加载和 post-load 阶段完成；
5. `:91-132`、`:165-208`：QKV、O、gate/up、down 等 Linear 路径完成执行；
6. `:224-228`：1 个 prompt 处理完成并生成 1 个 token；全日志没有 Traceback、
   ERROR 或 Exception。

`VllmConfig.__str__()` 位于 `$VLLM_PKG/config/vllm.py:2129-2160`。日志第 57 行中的：

```python
quantization=self.model_config.quantization
quantization_config=self.model_config.quantization_config
```

只打印了 `ModelConfig` 上的两个字段，没有打印已经实例化的
`VllmConfig.quant_config`，因此日志不会直接出现 `MXInt8Config` 类名。使用当前 SDK
和 conda 环境做配置构造探针，实际结果为：

```text
EngineArgs.quantization = mxint8
EngineArgs.quantization_config = None
ModelConfig.quantization = mxint8
ModelConfig.quantization_config = None
type(VllmConfig.quant_config) =
  vllm_sipu.model_executor.layers.quantization.mxint8.MXInt8Config
VllmConfig.quant_config.ignored_layers = []
```

另外，日志末尾的 `Selected Config: Config_128_128` 属于 Flash Attention 配置选择，
不是 MXInt8 config；KV cache 底层先分配为 `int8` 字节 buffer 再 view 成 BF16，也不是
MXInt8 权重量化的证据。

## 9. 最终判断

本次字段传递和解析均成功：recipe 的 `"mxint8"` 作为 `LLM` kwarg 进入 vLLM，
保存在 `EngineArgs.quantization` 和 `ModelConfig.quantization`，经 SIPU 提前注册的
registry 解析为内部 `MXInt8Config()`，再下传到 Qwen2 的 dense Linear 并绑定
`MXInt8LinearMethod`。dummy BF16 权重在 post-load 阶段转换为 packed MXInt8，forward
时 activation 也被动态量化后执行 MXInt8 GEMM。

成功生成的字符 `₲` 来自 dummy 随机权重，不具备模型质量意义；它能证明的是参数
解析、模型构造、权重量化、kernel 执行和采样链路已贯通。
