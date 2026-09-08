# Llama-3.1-8B-Instruct 4-layer 裁剪模型对比

## 1.1 路径校正和结论

用户给出的原始模型路径使用了单数目录名 `safetensor_weight`，该路径在当前文件系统中不存在。实际可访问的原始模型路径是：

```text
/data_gpu/modelzoo/weights/llm/llama/llama3.1/llama3.1-8B-Instruct/safetensor_weights
```

裁剪模型路径是：

```text
/share_data/inference-framework/tiny-models/Llama-3.1-8B-Instruct-4layer/safetensor_weights
```

结论：这个裁剪模型是一个保留原始模型前 4 个 Transformer 层的结构裁剪模型。它没有改变 hidden size、FFN 宽度、attention head 数、KV head 数、词表、tokenizer 或权重数值，也没有做量化或重新训练。主要修改如下：

1. `config.json` 中 `num_hidden_layers` 从 32 改成 4。
2. 只保留 `model.layers.0` 到 `model.layers.3` 的权重，删除 `model.layers.4` 到 `model.layers.31`。
3. 保留的权重重新写入两个 safetensors 分片，并重建 `model.safetensors.index.json`。
4. 增加描述裁剪信息的 `TRIM_INFO.md`。

## 1.2 裁剪脚本的直接证据

以下代码路径均相对于 `/share_data/inference-framework/tiny-models` code base。

`trim/trim_llama31_8b.py:5-12，模块级变量 SPEC` 定义了源目录、目标目录和裁剪规则：

```python
src=Path("/data_gpu/modelzoo/weights/llm/llama/llama3.1/llama3.1-8B-Instruct"),
dst=Path("/share_data/inference-framework/tiny-models/Llama-3.1-8B-Instruct-4layer/safetensor_weights"),
layer_map={0: 0, 1: 1, 2: 2, 3: 3},
scale_suffix='.weight_scale_inv',
```

`trim/smoke_trim.py:101-107，resolve_src` 会在 `src` 下优先查找 `safetensor_weights/config.json`，因此脚本实际读取的是上面的复数目录。

目标目录中的 `TRIM_INFO.md:5-14` 也记录了同样的事实：`num_hidden_layers: 32 -> 4`、layer map 为 `{0: 0, 1: 1, 2: 2, 3: 3}`，源目录大小约 15G，目标目录大小约 3.7G。

## 2.1 `config.json` 的差异

对两个 `config.json` 做 JSON 语义比较，只有一个字段变化：

| 字段 | 原始模型 | 裁剪模型 |
|---|---:|---:|
| `num_hidden_layers` | 32 | 4 |

下列字段保持不变：

| 字段 | 值 |
|---|---:|
| `hidden_size` | 4096 |
| `intermediate_size` | 14336 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 |
| `vocab_size` | 128256 |
| `max_position_embeddings` | 131072 |
| `rope_theta` | 500000.0 |
| `rope_scaling` | 完全相同 |
| `torch_dtype` | `bfloat16` |
| `bos_token_id` | 128000 |
| `eos_token_id` | `[128001, 128008, 128009]` |

证据位置：原始 `config.json:14-37`，裁剪 `config.json:14-37`。两份文件逐行 diff 只有 `num_hidden_layers` 所在行。

## 2.2 safetensors 张量集合的差异

原始模型的 `model.safetensors.index.json` 有 291 个 tensor key：

- 32 个 Transformer 层；每层 9 个张量，共 288 个。
- 3 个非层张量：`model.embed_tokens.weight`、`lm_head.weight`、`model.norm.weight`。

裁剪模型的索引有 39 个 tensor key：

- 4 个 Transformer 层；每层仍为同样的 9 个张量，共 36 个。
- 同样的 3 个非层张量。

对索引做集合比较得到：

```text
destination_keys = 39
expected_kept_keys = 39
exact_key_set = True
```

其中 `expected_kept_keys` 定义为原始索引中不属于层，或者层号小于 4 的 key。也就是说，裁剪模型没有选择后面的层，也没有重新排列层；保留下来的层号仍然是 0、1、2、3。

`trim/smoke_trim.py:697-724，should_keep` 实现了这一过滤。`:706-708` 取得 key 中的层号，并在层号不属于 `spec.layer_map.values()` 时返回 `False`。本模型的 values 是 `{0, 1, 2, 3}`，所以 `model.layers.4` 至 `model.layers.31` 全部被丢弃。

## 2.3 保留权重是否被修改

两边的保留张量都为 BF16，代表性形状完全相同：

| 张量 | 形状 |
|---|---|
| `model.embed_tokens.weight` | `(128256, 4096)` |
| `model.layers.0.self_attn.q_proj.weight` | `(4096, 4096)` |
| `model.layers.0.self_attn.k_proj.weight` | `(1024, 4096)` |
| `model.layers.0.self_attn.v_proj.weight` | `(1024, 4096)` |
| `model.layers.0.mlp.gate_proj.weight` | `(14336, 4096)` |
| `model.layers.0.mlp.down_proj.weight` | `(4096, 14336)` |
| `model.norm.weight` | `(4096,)` |
| `lm_head.weight` | `(128256, 4096)` |

我按 safetensors header 中的 `data_offsets` 对 39 个同名保留张量做了分块原始字节比较：

```text
destination_keys=39
checked=39
missing_source_keys=0
shape_mismatches=0
dtype_mismatches=0
value_mismatches=0
```

因此，保留下来的权重不是经过微调、平均、量化或数值重算，而是从原始模型对应张量原样复制后重新序列化。重新序列化会改变 safetensors 文件的分片布局和 header，但没有改变 tensor payload。

另外，两个索引都没有包含 `scale`、`packed` 或 `shape` 量化辅助张量。`trim/trim_llama31_8b.py:11，模块级变量 SPEC` 中的 `scale_suffix='.weight_scale_inv'` 只是通用裁剪器的接口配置；本次 BF16 原始模型没有这类 key，所以没有发生 scale 转换。

## 2.4 分片、参数量和辅助文件

原始模型：

- 4 个分片：`model-00001-of-00004.safetensors` 到 `model-00004-of-00004.safetensors`。
- 索引记录的总大小：`16060522496` bytes，约 15G。
- 按 tensor shape 统计约 `8030261248` 个参数。

裁剪模型：

- 2 个分片：`model-00001-of-00002.safetensors`、`model-00002-of-00002.safetensors`。
- 索引记录的总大小：`3846255024` bytes，约 3.7G。
- 按 tensor shape 统计约 `1923125248` 个参数。

参数量约从 8.03B 降到 1.92B，减少约 76.05%。固定保留的 embedding、`lm_head` 和 final norm 仍然占据约 1.05B 参数，因此存储量不会简单地变成原来的四分之一。

`trim/smoke_trim.py:1427-1559，_trim_with_index` 读取源索引，只处理通过 `should_keep` 的 key，按约 2GB 的目标分片大小写入临时 safetensors，再在 `:1540-1553` 重命名为最终分片并生成新的 `model.safetensors.index.json`。

原始和裁剪目录中的以下辅助文件逐字节相同：

- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `LICENSE`
- `README.md`
- `USE_POLICY.md`

裁剪目录额外增加 `TRIM_INFO.md`，它只是说明文件，不是模型输入权重。

## 3.1 配置裁剪流程

`trim/smoke_trim.py:1738-1824，run_smoke_trim` 是总入口，执行顺序为：

1. 解析源目录并创建目标目录。
2. 调用 `apply_config` 生成裁剪后的配置。
3. 调用 `copy_aux_files` 复制 tokenizer 和其他辅助文件。
4. 调用 `_trim_with_index` 过滤并重写权重。
5. 调用 `verify_basic` 检查层号、索引和张量形状。
6. 写入 `TRIM_INFO.md` 和裁剪摘要。

`trim/smoke_trim.py:281-567，apply_config` 先深拷贝源配置。`:350` 将 `num_hidden_layers` 设置为 `len(spec.layer_map)`，即 4；本 SPEC 的其他缩小参数（`hidden_size`、`intermediate_size`、head 数、expert 数等）都是 `None`，所以不会修改对应字段。

`trim/smoke_trim.py:968-1424，transform` 先通过 `remap_layer_key` 映射层名。由于本模型的 layer map 是恒等映射，保留层的 key 不变；随后在 `:977-1014` 进入 structure-only fast path，对未发生维度变化的 tensor 执行 `tensor.clone()`。

`trim/trim_common.py:190-248，copy_aux_files` 明确跳过 safetensors、配置和索引文件，只复制小于 64MiB 的辅助文件，因此不会把原始的四个大分片错误地带入目标目录。

`trim/smoke_trim.py:1562-1735，verify_basic` 检查 `num_hidden_layers == 4`、索引中不存在大于等于 4 的层号，并进行 tensor spot-check。实际生成日志 `/share_data/inference-framework/tiny-models/_trim_logs/batch43/trim_llama31_8b.log:11-18` 显示：保留 39 个 tensor、生成 2 个分片、配置和形状验证通过。

## 3.2 对运行时行为的影响

加载裁剪模型时，Transformers/SGLang 根据裁剪后的 `config.json` 创建 4 个 decoder layer，因此前向路径只执行原始模型的第 0、1、2、3 层，然后执行原样保留的 `model.norm` 和 `lm_head`。embedding、RoPE 配置、词表和 tokenizer 行为不变，但网络深度大幅降低。

因此它适合用于：

- SGLang/SIPU 的快速启动、接口和 kernel smoke test。
- 验证量化、attention、KV cache 等代码是否能在小模型上运行。

它不等价于原始 8B 模型的精度模型。由于删除了 28 个 Transformer 层，生成质量和基准精度预计会显著下降；这种下降来自结构裁剪，而不是量化误差或 tokenizer 改动。

## 4.1 最终判断

“裁剪模型在原始模型上做了哪些修改”的准确描述是：

> 将原始 Llama-3.1-8B-Instruct 的 32 层 decoder 截取为前 4 层，更新层数配置，删除其余层权重，保留所有宽度、head、词表、tokenizer 和保留张量数值不变，然后重新分片保存。

没有证据表明该过程做了以下操作：

- 改变 hidden size 或 intermediate size；
- 改变 attention/KV head 数；
- 改变 RoPE、最大上下文长度或 token id；
- 权重量化、反量化或 scale 重算；
- 蒸馏、微调、权重平均或其他训练。
