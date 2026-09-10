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

## 5.1 对 commit 11d03eaeef 的核查范围

本次核查在 Docker 容器 /sipu-dev 中进行。容器把宿主机的
/share/users/like/package/sglang_sipu 挂载到 /sgl-workspace/sglang，
容器内 sglang 的 editable 安装也指向 /sgl-workspace/sglang/python。
容器当前的 PyTorch 是 CPU 版且没有 CUDA/FlashInfer，因此下面的 kernel
数量是由源码和调用关系推导的，不能把本容器的运行结果当成真实 CUDA
profile。

## 5.2 先给结论

1. 11d03eaeef 有实际意义，但它做的不是“FP8 activation quantization +
   FP8 TensorCore MMA 融合”。它融合的是 RMSNorm（或 residual-add-RMSNorm）
   + 静态 per-tensor FP8 activation quantization。下游 FP8 GEMM/MMA 仍然
   是另一个 kernel。
2. 对满足条件的 CUDA 静态 FP8 路径，典型 kernel 数量从
   RMSNorm + quant + GEMM = 3 变成 fused RMSNorm+quant + GEMM = 2，
   因此少一个 launch，并省掉 normalized BF16 中间张量的一次写入和一次
   读取。
3. 该提交不适用于 dynamic per-token、block quant、MXFP8 或 Marlin
   路径；这些路径不会因为该提交自动少一个 launch。
4. “quant 产生的 scale 是否写 global memory”必须分静态和动态两种情况：
   - 静态 per-tensor：scale 不是 quant kernel 计算出来的，而是权重检查点中
     已存在的 layer.input_scale。普通 scalar 路径只从 device/global tensor
     读取它，不再写一个新 scale；为不支持 scalar-A-scale 的 CUTLASS 路径，
     旧代码会额外生成并写入每 token 一个 scale 的 global tensor。
   - 动态 per-token（或动态 per-tensor）：scale 是 quant kernel 计算出的
     输出，确实写入 device/global memory。因为旧 quant 和 GEMM 是两个
     kernel，不能只把 scale 留在前一个 kernel 的 register/shared memory 中
     交给后一个 kernel。

## 5.3 提交前的真实调用路径

在 11d03eaeef^：

- python/sglang/srt/models/llama.py:341-363
  (LlamaDecoderLayer::forward) 只调用
  self.input_layernorm(hidden_states) 和
  self.post_attention_layernorm(...)，没有把下游 linear 传给 RMSNorm。
- python/sglang/srt/layers/layernorm.py:405-489
  (RMSNorm::forward_cuda) 走 rmsnorm 或 fused_add_rmsnorm，输出
  BF16/FP16 normalized activation。
- python/sglang/srt/layers/quantization/fp8.py:960-1046
  (Fp8LinearMethod::apply) 的普通分支调用 apply_fp8_linear。
- python/sglang/srt/layers/quantization/fp8_utils.py:1713-1816
  (apply_fp8_linear) 先调用 quant：static_quant_fp8（静态 scale）或
  sglang_per_token_quant_fp8（动态 scale），随后在
  python/sglang/srt/layers/quantization/fp8_utils.py:1799-1815
  (apply_fp8_linear) 调用 triton_scaled_mm/fp8_scaled_mm。这两个
  Python 调用对应两个独立的 GPU operation，不是一个 quant+MMA kernel。

提交自带的 benchmark 也明确把
benchmark/kernels/bench_fused_rmsnorm_fp8_quant.py:1-13
(module docstring) 的 unfused 定义为 “RMSNorm followed by a separate
static FP8 quant”，把 fused 定义为 FlashInfer rmsnorm_quant；
benchmark/kernels/bench_fused_rmsnorm_fp8_quant.py:67-84
(run_unfused/_run_fused) 也分别写出了两个调用和一个调用。这个 benchmark
本身不包含下游 GEMM，所以不能解读成 quant+GEMM 已经融合。

## 5.4 提交后的变化，以及实际少了哪个 launch

提交在 python/sglang/srt/models/llama.py:341-369
(LlamaDecoderLayer::forward, 11d03eaeef) 把
quant_linear=self.self_attn.qkv_proj 和
quant_linear=self.mlp.gate_up_proj 传给 RMSNorm（Qwen2 也有同样修改）。

python/sglang/srt/layers/layernorm.py:370-418
(_fp8_static_input_scale/_is_static_per_tensor_fp8_linear, 11d03eaeef)
只接受 native Fp8LinearMethod 的非 block、非 MXFP8、非 Marlin
情况，并且要求 input_scale.numel() == 1。满足条件且 FlashInfer 可用时，
python/sglang/srt/layers/layernorm.py:469-517
(RMSNorm::forward_cuda, 11d03eaeef) 调用
forward_with_per_tensor_quant_fusion。

python/sglang/srt/layers/layernorm.py:858-915
(RMSNorm::forward_with_per_tensor_quant_fusion, 11d03eaeef)：

- 分配 FP8 输出 out；
- 调用 _flashinfer_rmsnorm_quant 或 _flashinfer_fused_add_rmsnorm_quant；
- 返回 (fp8_out, scale, orig_dtype)（有 residual 时再带 residual）。

这里的 scale 是调用者传入的既有 tensor，函数只是把同一个 scale 放进
返回 tuple，并没有分配或计算一个新的 scale 输出。

然后 python/sglang/srt/layers/quantization/fp8.py:1035-1061
(Fp8LinearMethod::apply, 11d03eaeef) 识别这个 tuple，
python/sglang/srt/layers/quantization/fp8_utils.py:1741-1772
(apply_fp8_linear, 11d03eaeef) 把它当作已经量化的 FP8 输入，跳过
再次 quantize；但 python/sglang/srt/layers/quantization/fp8_utils.py:1838-1854
(apply_fp8_linear, 11d03eaeef) 仍然单独调用 fp8_scaled_mm
（其它形状则调用 torch._scaled_mm）。因此：

| 情况 | 提交前 | 提交后 | 是否由该提交减少 |
| --- | --- | --- | --- |
| 静态 per-tensor、无 residual | RMSNorm + quant + GEMM（3） | fused RMSNorm+quant + GEMM（2） | 是，少 1 |
| 静态 per-tensor、有 residual | fused-add-RMSNorm + quant + GEMM（3） | fused-add-RMSNorm+quant + GEMM（2） | 是，少 1 |
| dynamic per-token | RMSNorm + dynamic quant + GEMM | 通常仍为这三步 | 否 |
| block/MXFP8/Marlin | 各自原有路径 | helper 不匹配，仍走原路径 | 否 |

这里的“3/2”是该线性层前向中对应 operation 的逻辑数量；CUDA Graph
捕获只会把这些 operation 作为 graph node，不能把两个 kernel 变成一个。
此外，旧静态 quant kernel 已经使用了 PDL（见
python/sglang/kernels/ops/quantization/fp8_kernel.py:885-899
(_static_quant_fp8)）；PDL 可以改善 producer/consumer 的依赖等待，
但仍不等价于 kernel fusion。

## 5.5 scale 是否写入 global memory

### 5.5.1 静态 per-tensor scale

python/sglang/srt/layers/quantization/fp8.py:613-629
(Fp8LinearMethod::create_fp8_weight_) 在静态 activation scheme 下注册
layer.input_scale；python/sglang/srt/layers/quantization/fp8.py:879-950
(Fp8LinearMethod::process_weights_after_loading) 将它保留为不可训练的
device parameter。因此这个 scale 在进入 quant kernel 之前就已经存在，
不是本次量化临时算出来的。

旧实现 python/sglang/kernels/ops/quantization/fp8_kernel.py:865-921
(static_quant_fp8, 11d03eaeef^) 的行为是：

- x_q = torch.empty_like(..., device=x.device)：FP8 activation 输出在
  device/global memory；
- repeat_scale=False 时，_static_quant_fp8 在
  python/sglang/kernels/ops/quantization/fp8_kernel.py:814-863
  (_static_quant_fp8, 11d03eaeef^) 的 line 852 只执行 tl.load(y_s_ptr)，
  line 860 写 FP8 输出，不写新 scale；返回的仍是原来的 x_s；
- repeat_scale=True 时，line 890-897 分配 (M,1) 的 x_s_repeat，line 862
  把 scale 写入这个 global tensor。提交前
  python/sglang/srt/layers/quantization/fp8_utils.py:1773-1779
  (apply_fp8_linear, 11d03eaeef^) 对 CUTLASS channelwise 情况会请求这种
  repeat。

提交后的 python/sglang/srt/layers/quantization/fp8_utils.py:1754-1763
(apply_fp8_linear, 11d03eaeef) 引入 native_scalar_a_scale，在
SM90/SM100/SM120 且 CUTLASS 支持 scalar A scale 时，
python/sglang/srt/layers/quantization/fp8_utils.py:1812-1818
(apply_fp8_linear, 11d03eaeef) 不再 repeat scale，直接把原 scalar
传给 GEMM。这减少了 repeated-scale 的 global allocation/store，但仍然
不能消除 FP8 activation qinput 的 global 写入，因为 GEMM 还是后续独立
kernel。

### 5.5.2 动态 per-token scale

提交前 python/sglang/kernels/ops/quantization/fp8_kernel.py:788-804
(sglang_per_token_quant_fp8, 11d03eaeef^) 明确分配
x_s = torch.empty(x.shape[0], 1, device=x.device, dtype=torch.float32)，
再调用 CUDA quant op。

在其 CUDA 实现
python/sglang/kernels/aot/csrc/gemm/per_token_quant_fp8.cu:17-121
(per_token_quant_fp8_kernel, 11d03eaeef^) 中，line 32 把 token_scale
指向 output_s + token_id，line 78 执行 token_scale[0] = scale；
小 batch kernel 也把 output_s.data_ptr() 作为输出（line 289-309）。
宿主入口
python/sglang/kernels/aot/csrc/gemm/per_token_quant_fp8.cu:242-314
(sgl_per_token_quant_fp8, 11d03eaeef^) 负责发射这些 kernel。
所以动态 scale 的确会写入 global/device memory。

即使量化 kernel 内部先把 scale 放在 register 或 shared memory 做归约，
旧架构中它必须在 kernel 边界前写到 x_s，下一个 GEMM 才能读取。
register/shared memory 的生命周期和可见范围都不能跨 kernel launch；只有把
quant 和 GEMM 真正写成同一个 kernel（或采用专门的 persistent producer/
consumer 设计）才能完全省掉这个跨 kernel global hand-off。

## 5.6 该提交是否把 weight 放进 shared/register

没有。提交的 fused FlashInfer kernel 只负责 RMSNorm 和 FP8 输出生成；
python/sglang/srt/layers/quantization/fp8_utils.py:1838-1854
(apply_fp8_linear, 11d03eaeef) 随后仍把 FP8 activation、weight 和 scale
交给独立的 fp8_scaled_mm。CUDA 实现
python/sglang/kernels/aot/csrc/gemm/fp8_gemm_kernel.cu:1125-1205
(fp8_scaled_mm) 自己检查输入、分配输出并 dispatch CUTLASS。前一个
RMSNorm/quant kernel 中的 register/shared 数据不会保留给它；FP8 activation
也必须作为 global tensor 交接。weight 是长期驻留的 device tensor，但是否
被 GEMM tile 缓存到该 GEMM 自身的 shared/register 由 GEMM kernel 决定，
不是 11d03eaeef 新增的跨 kernel 融合。

## 5.7 对 sglang_sipu 的实际影响

这个提交的目标架构是 CUDA SM90/SM100/SM120。当前 SIPU 路径
python/sglang/srt/layers/layernorm.py:575-614
(RMSNorm::forward_sipu) 在收到量化的 quant_linear 时会直接抛出
NotImplementedError("RMSNorm with quant_linear is not supported on SIPU")，
随后只调用普通 SIPU rmsnorm/fused_add_rmsnorm。因此在 sglang_sipu
上不能把 11d03eaeef 的 CUDA FlashInfer fusion 当作已经生效；SIPU 的
fp8_scaled_mm 和 sgl_per_token_quant_fp8 仍是独立接口
（sgl-kernel-sipu/sgl_kernel/gemm.py:56-64 (fp8_scaled_mm)、
sgl-kernel-sipu/sgl_kernel/gemm.py:150-155 (sgl_per_token_quant_fp8)）。

## 5.8 最终回答

对核心问题的直接回答是：

> 在 11d03eaeef 之前，若 Fp8LinearMethod 走 dynamic per-token
> activation quantization，BF16->FP8 计算出的 scale 会写入
> device/global memory（x_s[M,1]），因为后面的独立 GEMM 必须读取它。
> 若走 static per-tensor activation quantization，不存在“quant 新产生的
> scale”：layer.input_scale 早已是 device tensor，旧 kernel 只读取 scalar；
> 只有为旧 CUTLASS channelwise 接口做 repeat_scale 时，才会额外把重复
> scale 写到 global memory。

所以你的判断“如果 11d03eaeef 把 BF16->FP8 quant 和 TensorCore MMA
放进同一个 kernel，那它意义不大”是成立的；但实际提交并没有做这件事。
它把 RMSNorm + static quant 合并，留下 FP8 GEMM 为独立 kernel，因此在
适用的 CUDA 静态路径上确实少一次 launch 和一轮 BF16 中间结果的 global
memory traffic，而不是消除了 quant+MMA 之间的全部 global hand-off。

---

# sglang_sipu 代码库 `scripts/ci/sipu_ci_exec.sh` 逐行讲解

> 代码库根目录:`/share/users/like/package/sglang_sipu`(git HEAD `3fe7ff93b`)。
> 本文件内所有“相对 code base 路径 + 行号 + 函数名”均指该仓库内的脚本。
> 普通函数用函数名,类的成员函数用 `类::函数名` 形式(本套脚本全为 bash 函数)。
> 注释不做逐行讲解;只讲会执行的语句。

## 0. 一句话总结

`sipu_ci_exec.sh` 是 SIPU 精度 CI 的**宿主机侧入口**:先在宿主机上用
`sipu-ci-builder` 容器把 `sgl-kernel-sipu`(及 `deepep`、`siorigin_triton_kernels`、
`siorigin_tilelang_kernels`、`triton_ops`)打成 wheel,再把单个用例或整个 suite
分发到**本机直跑**(archmodel)或 **QEMU 虚拟机**(qemu)里执行 `sipu_ci_run.sh`,
最后由 `sipu_ci_summary.py` 汇总判题。

## 1. `scripts/ci/sipu_ci_exec.sh` 逐行讲解

### 1.1 初始化(第 13-30 行)

- `scripts/ci/sipu_ci_exec.sh:13` — `set -eo pipefail`:任何命令非 0 退出立即终止,
  管道中任一环节失败也视为失败。
- `scripts/ci/sipu_ci_exec.sh:15` — 取脚本自身所在目录(绝对路径),存 `SCRIPT_DIR`。
- `scripts/ci/sipu_ci_exec.sh:16` — `SGLANG="$(cd "$SCRIPT_DIR/../.." && pwd)"`:
  由 `scripts/ci` 上溯两级得到代码库根 `/share/users/like/package/sglang_sipu`。
- `scripts/ci/sipu_ci_exec.sh:17` — `KERNEL="$SGLANG/sgl-kernel-sipu"`:子模块路径。
- `scripts/ci/sipu_ci_exec.sh:18` — `SUITES_YAML`:suite 定义文件
  `scripts/ci/sipu_ci_suites.yaml`。
- `scripts/ci/sipu_ci_exec.sh:20` — `IMAGE`:CI 容器镜像,默认
  `harbor.siorigin.com/sglang-sipu/release:v0.5.18-sipu-dev-0.1.0`,可被
  `SIPU_CI_IMAGE` 覆盖。
- `scripts/ci/sipu_ci_exec.sh:21` — `CI_USER`:默认 `acndd`,决定 CUDA golden 目录。
- `scripts/ci/sipu_ci_exec.sh:22` — `CI_ROOT`:CI 日志根,默认
  `<repo>/ci_logs`。
- `scripts/ci/sipu_ci_exec.sh:23` — `HOST_UID="$(id -u)"`;第 24 行
  `HOST_GID="$(id -g)"`:用于把容器产物 chown 回宿主机用户。
- `scripts/ci/sipu_ci_exec.sh:25` — `CONTAINER_KERNEL=/sgl-workspace/sglang/sgl-kernel-sipu`:
  wheel 构建容器内挂载的 kernel 源码路径。
- `scripts/ci/sipu_ci_exec.sh:27` — `SIPU_CI_CUDA_DUMP`:CUDA golden 根目录,默认
  `/share_data/sglang_sipu/accuracy_verify/$CI_USER`。
- `scripts/ci/sipu_ci_exec.sh:28` — `SIPU_CI_DUMP`:本次 SIPU 精度 dump 根,默认
  `<cwd>/ci_dumps`。
- `scripts/ci/sipu_ci_exec.sh:29` — `SIPU_CI_WHEEL_DIR`:wheel 输出目录,默认
  `<cwd>/ci_wheels`。
- `scripts/ci/sipu_ci_exec.sh:30` — `SIPU_CI_QUEUE_DIR`:用例队列目录,默认
  `<cwd>/ci_queues`。

### 1.2 `usage` 与参数解析(第 32-77 行)

- `scripts/ci/sipu_ci_exec.sh:32` — `usage()`:打印两种用法并 `exit 1`。
- `scripts/ci/sipu_ci_exec.sh:42` — `CONFIG_YAML="" LAUNCH_CONFIG="" TEST_CASE=""
  TIMEOUT="30m"`:单用例参数初始值。
- `scripts/ci/sipu_ci_exec.sh:43` — `TEST_SUITES=""`。
- `scripts/ci/sipu_ci_exec.sh:44` — `PLATFORM=""`。
- `scripts/ci/sipu_ci_exec.sh:45` — `MODE="eager"`。
- `scripts/ci/sipu_ci_exec.sh:46` — `MODE_SET=0`:记录 `--mode` 是否被显式给出。
- `scripts/ci/sipu_ci_exec.sh:47` — `SIPU_CI_TMP_SGL_KERNEL_SIPU`:环境变量覆盖值,
  默认为空。
- `scripts/ci/sipu_ci_exec.sh:48` — `RUN_ARGS=()`:收集单用例模式要透传给
  `sipu_ci_run.sh` 的参数。
- `scripts/ci/sipu_ci_exec.sh:50` — `while [[ $# -gt 0 ]]` 开始参数循环。
- `scripts/ci/sipu_ci_exec.sh:51` — `case "$1" in` 按参数名分发。
- `scripts/ci/sipu_ci_exec.sh:52` — `--config-yaml`:保存路径并原样追加进
  `RUN_ARGS`。
- `scripts/ci/sipu_ci_exec.sh:53` — `--launch-config`:同上。
- `scripts/ci/sipu_ci_exec.sh:54` — `--test-case`:同上。
- `scripts/ci/sipu_ci_exec.sh:55` — `--timeout`:保存超时并透传。
- `scripts/ci/sipu_ci_exec.sh:56` — `--test-suites`:保存 suite 列表(逗号分隔)。
- `scripts/ci/sipu_ci_exec.sh:57` — `--mode`:保存 `MODE` 并置 `MODE_SET=1`;
  第 60-62 行校验只能是 `eager|compile|graph`。
- `scripts/ci/sipu_ci_exec.sh:64` — `--platform`:保存 `PLATFORM`;第 66-68 行校验
  只能是 `archmodel|qemu`。
- `scripts/ci/sipu_ci_exec.sh:70` — `--tmp-sgl-kernel-sipu`:记录跳过
  submodule init 与 wheel 构建的目录;第 71 行要求非空。
- `scripts/ci/sipu_ci_exec.sh:74` — `-h|--help` 调 `usage()`。
- `scripts/ci/sipu_ci_exec.sh:75` — 未知参数报错并调 `usage()`。

### 1.3 参数互斥校验(第 79-89 行)

- `scripts/ci/sipu_ci_exec.sh:79` — 若给了 `--test-suites`:
- `scripts/ci/sipu_ci_exec.sh:80` — 第 80-81 行:单用例参数与 suite 参数不能混用。
- `scripts/ci/sipu_ci_exec.sh:82` — 第 82-83 行:suite 模式不允许 `--platform`
  (platform 由 yaml 决定)。
- `scripts/ci/sipu_ci_exec.sh:84` — 否则单用例三参数缺一即调 `usage()`。
- `scripts/ci/sipu_ci_exec.sh:86` — 第 86-88 行:单用例模式不允许 `--mode`
  (单用例用 `--launch-config`,compile/graph 只属于 suite)。

### 1.4 加载公共函数与清理 trap(第 91-104 行)

- `scripts/ci/sipu_ci_exec.sh:92` — source `scripts/ci/sipu_ci_single_case.sh`
  (只取其中的 `detect_current_platform`、`setup_docker_args`、`run_container`、
  `reclaim_host_ownership` 等函数)。
- `scripts/ci/sipu_ci_exec.sh:93` — 调 `detect_current_platform()`:宿主机没有
  `/dev/sipu`,所以 `SIPU_CURRENT_PLATFORM=host`。
- `scripts/ci/sipu_ci_exec.sh:95` — 创建 `ci_dumps`、`ci_wheels`、`ci_queues`。
- `scripts/ci/sipu_ci_exec.sh:96` — `cleanup()`:退出时依次
  `sipu_ci_qemu.sh stop` 停 QEMU、`reclaim_host_ownership()` 归还文件属主、
  删除 `ci_queues`、按 `SIPU_CI_KEEP_WHEELS` 决定是否删 `ci_wheels`。
- `scripts/ci/sipu_ci_exec.sh:104` — `trap cleanup EXIT`:保证任何退出路径都清理。

### 1.5 `use_tmp_kernel` / `init_submodule` / `build_wheels`(第 106-140 行)

- `scripts/ci/sipu_ci_exec.sh:106` — `use_tmp_kernel()`:校验临时 kernel 目录存在
  且有 `Makefile`,然后导出 `SIPU_CI_TMP_SGL_KERNEL_SIPU`(第 108-112 行)。
- `scripts/ci/sipu_ci_exec.sh:115` — `init_submodule()`:在仓库根执行
  `git submodule update --init sgl-kernel-sipu`(第 117 行,仅当
  `$KERNEL/Makefile` 不存在时);第 118-119 行再检查一次,失败即退出。
- `scripts/ci/sipu_ci_exec.sh:122` — `build_wheels()`:宿主机侧 wheel 构建。
- `scripts/ci/sipu_ci_exec.sh:124` — 调 `run_container sipu-ci-builder ...`:起一个
  名为 `sipu-ci-builder` 的一次性容器(镜像即 `IMAGE`),在其中依次执行:
- `scripts/ci/sipu_ci_exec.sh:126` — `cd $CONTAINER_KERNEL`。
- `scripts/ci/sipu_ci_exec.sh:127` — `bash scripts/init_submodules.sh` 初始化
  kernel 子模块。
- `scripts/ci/sipu_ci_exec.sh:128` — `source setup.sh` 加载 SIPU SDK 环境。
- `scripts/ci/sipu_ci_exec.sh:129` — `unset TORCH_DEVICE_BACKEND_AUTOLOAD`。
- `scripts/ci/sipu_ci_exec.sh:130` — `make clean`。
- `scripts/ci/sipu_ci_exec.sh:131` — `make install-kernels` 编译安装 kernel。
- `scripts/ci/sipu_ci_exec.sh:132` — `pip wheel . --no-deps --no-build-isolation`:
  打 `sgl-kernel-sipu` wheel 到 `$SIPU_CI_WHEEL_DIR`。
- `scripts/ci/sipu_ci_exec.sh:133` — `pip wheel ./deepep ...`。
- `scripts/ci/sipu_ci_exec.sh:134` — `pip wheel ./siorigin_triton_kernels ...`。
- `scripts/ci/sipu_ci_exec.sh:135` — `pip wheel ./siorigin_tilelang_kernels ...`。
- `scripts/ci/sipu_ci_exec.sh:136` — `pip wheel ./triton_ops ...`。
- `scripts/ci/sipu_ci_exec.sh:137` — `ls -la $SIPU_CI_WHEEL_DIR` 打印产物。
- `scripts/ci/sipu_ci_exec.sh:139` — `reclaim_host_ownership()`:把容器内 root
  产生的文件 chown 回 `HOST_UID:HOST_GID`。

### 1.6 主流程前半段(第 142-150 行)

- `scripts/ci/sipu_ci_exec.sh:142` — 若设置了 `SIPU_CI_TMP_SGL_KERNEL_SIPU`,调
  `use_tmp_kernel()`;否则(第 145 行)调 `init_submodule()`。
- `scripts/ci/sipu_ci_exec.sh:147` — `setup_docker_args()`:按
  `SIPU_CURRENT_PLATFORM` 组装后续 `run_container` 要用的 `docker_args`(挂载与
  环境变量;qemu 内还会加 `--net=host` 与 `/dev/sipu` 设备)。
- `scripts/ci/sipu_ci_exec.sh:148` — 非临时 kernel 模式才调 `build_wheels()`。

### 1.7 `suite_raw` / `run_local` / `run_on_qemu` / `dispatch`(第 152-190 行)

- `scripts/ci/sipu_ci_exec.sh:152` — `suite_raw()`:用
  `scripts/ci/sipu_ci_cases.py --suites-yaml ... --suite <name>` 把 suite 展开成
  `#key=value` 注释行 + 每行一个 `family|model|config|launch|case|timeout` 用例。
- `scripts/ci/sipu_ci_exec.sh:156` — `run_local()`:直接在本机
  `bash scripts/ci/sipu_ci_run.sh "$@"`(archmodel 模式)。
- `scripts/ci/sipu_ci_exec.sh:161` — `run_on_qemu()`:qemu 模式。
- `scripts/ci/sipu_ci_exec.sh:162` — `qemu_run="${SIPU_CI_QEMU_RUN_DIR:-$PWD/qemu_run}"`,
  `rc=0`。
- `scripts/ci/sipu_ci_exec.sh:164` — `bash scripts/ci/sipu_ci_qemu.sh start`:准备
  qcow2、挑 12xxx 端口、启动 `qemu-system-x86_64`,等 guest SSH 就绪。
- `scripts/ci/sipu_ci_exec.sh:165` — 从 `qemu_run/ssh.port` 读 SSH 端口。
- `scripts/ci/sipu_ci_exec.sh:167` — `sipu_ci_qemu.sh ssh "..."`:在 guest 内导出
  `SIPU_CI_USER/CUDA_DUMP/DUMP/WHEEL_DIR/QUEUE_DIR/IMAGE/KEEP_DUMPS/CI_ROOT/TMP_SGL_KERNEL_SIPU`
  (第 168-176 行),然后 `cd $SGLANG && bash scripts/ci/sipu_ci_run.sh <原参数>`
  (第 177 行);失败码记入 `rc`。
- `scripts/ci/sipu_ci_exec.sh:178` — 无论成败都 `sipu_ci_qemu.sh stop` 停 VM。
- `scripts/ci/sipu_ci_exec.sh:179` — `return "$rc"`。
- `scripts/ci/sipu_ci_exec.sh:182` — `dispatch()`:导出 `SIPU_CI_PLATFORM`;按
  `qemu` 与否分别调 `run_on_qemu()` 或 `run_local()`(第 185-189 行)。

### 1.8 `dispatch_suite`(第 192-215 行)

- `scripts/ci/sipu_ci_exec.sh:192` — `dispatch_suite()`:展开并分发一个 suite。
- `scripts/ci/sipu_ci_exec.sh:194` — `suite_raw "$suite"` 拿原始展开。
- `scripts/ci/sipu_ci_exec.sh:195` — 用 `sed` 提取 `#include=...`。
- `scripts/ci/sipu_ci_exec.sh:196` — 有 include 时,第 197-203 行:逗号拆分后逐个
  递归 `dispatch_suite` 子 suite,任一失败置 `overall=1`(第 202 行),然后返回。
- `scripts/ci/sipu_ci_exec.sh:206` — 提取 `#platform=`。
- `scripts/ci/sipu_ci_exec.sh:207` — `run_args=(--test-suites "$suite")`。
- `scripts/ci/sipu_ci_exec.sh:210` — 仅当显式给了 `--mode` 才追加 `--mode`(第 211 行)。
- `scripts/ci/sipu_ci_exec.sh:213` — 打印分发信息。
- `scripts/ci/sipu_ci_exec.sh:214` — `dispatch "$platform" "${run_args[@]}"`。

### 1.9 主流程收尾(第 217-228 行)

- `scripts/ci/sipu_ci_exec.sh:217` — `overall=0`。
- `scripts/ci/sipu_ci_exec.sh:218` — suite 模式:逗号拆分,逐个
  `dispatch_suite`,失败置 `overall=1`(第 219-224 行)。
- `scripts/ci/sipu_ci_exec.sh:226` — 否则单用例模式:`dispatch
  "${PLATFORM:-archmodel}" "${RUN_ARGS[@]}"`(第 226 行)。
- `scripts/ci/sipu_ci_exec.sh:228` — `exit "$overall"`:0 为全过。

## 2. 依赖脚本讲解(执行链上会真正跑到的)

### 2.1 `scripts/ci/sipu_ci_single_case.sh`

- `scripts/ci/sipu_ci_single_case.sh:10` — `detect_current_platform()`:看
  `/dev/sipu` 是否存在,置 `SIPU_CURRENT_PLATFORM=qemu|host`。
- `scripts/ci/sipu_ci_single_case.sh:18` — `sipu_ci_init_paths()`:补全路径变量,
  再调 `detect_current_platform()`(第 31 行)。
- `scripts/ci/sipu_ci_single_case.sh:34` — `setup_docker_args()`:组装 docker 参数。
  - 第 36-53 行:传 `PYTHONUNBUFFERED` 与各 `SIPU_CI_*` 环境变量;挂载
    `$SGLANG:/sgl-workspace/sglang:rw`、`ci_wheels`、`ci_logs`、SDK、cmodel、
    `tiny-models`、`torch_sipu`、`triton`、`tilelang` 等。
  - 第 54-57 行:`SIPU_CI_DUMP` 单独以 rw 挂载。
  - 第 58 行:`$HOME/.ssh` 只读挂进 `/root/.ssh`。
  - 第 59-64 行:临时 kernel 模式下额外挂载
    `$SIPU_CI_TMP_SGL_KERNEL_SIPU:/sgl-workspace/sglang/sgl-kernel-sipu:rw`。
  - 第 65-86 行:qemu guest 内追加 `SIRT_SHIM_NAME=ksipu`、`--net=host`、
    `/opt/siorigin` 与 4 个 `/dev/sipu/usipu*` 设备(第 71-78 行,缺设备即报错
    退出);第 80-85 行把 `/dev/dri/*` 字符/块设备也挂进去。
- `scripts/ci/sipu_ci_single_case.sh:89` — `run_container()`:先 `docker rm -f`
  同名容器(第 96 行),`docker run --name <name> ${docker_args[@]} <IMAGE> bash -lc
  "<cmd>"`(第 97 行),退出后再次 `docker rm -f`(第 98 行),返回 rc。
- `scripts/ci/sipu_ci_single_case.sh:104` — `reclaim_host_ownership()`:qemu
  guest 内直接返回(第 107 行);宿主机上对 dump/wheels/ci_root 等目录起一个
  root 容器统一 `chown -R $HOST_UID:$HOST_GID`(第 114-115 行,失败不致命)。
- `scripts/ci/sipu_ci_single_case.sh:118` — `run_one_case()`:按
  `|` 拆分一行用例(第 122 行),`setup_docker_args()`(第 123 行),组装
  `cmd`(第 126-130 行)为
  `/bin/bash /sgl-workspace/sglang/scripts/ci/sipu_ci_test.sh --config-yaml ...`,
  容器名 `sipu-ci-<cid>-$$`(第 131 行),按 `SIPU_CI_LIVE_LOG` 决定是否显示日志
  (第 132-136 行),`reclaim_host_ownership()`(第 137 行),把 rc 写入
  `ci_queues/status/<cid>`(第 138-139 行);`SIPU_CI_FAIL_FAST=1` 时若失败调
  `sipu_ci_summary.py --check` 决定是否 `exit 255`(第 141-147 行)。

### 2.2 `scripts/ci/sipu_ci_run.sh`

- `scripts/ci/sipu_ci_run.sh:21` — `run_dir_for()`:nightly 用
  `/share_data/sglang_sipu/nightly/<date>`;否则 `ci_logs/<suite>[/<mode>]/<date>`。
- `scripts/ci/sipu_ci_run.sh:72` — `section_id()`:把任意字符串转成
  `[A-Za-z0-9_]` 片段。
- `scripts/ci/sipu_ci_run.sh:74` — `section_start()`:`section_id()` 生成 id 后
  输出 GitLab CI 风格的 `section_start:<ts>:<id>[collapsed=true]` 控制序列
  (第 78-82 行)。
- `scripts/ci/sipu_ci_run.sh:85` — `section_end()`:输出对应的 `section_end` 序列。
- `scripts/ci/sipu_ci_run.sh:89` — 参数校验:suite 与单用例互斥、缺参则
  `usage()`(第 90-97 行)。
- `scripts/ci/sipu_ci_run.sh:100` — `cleanup()`:`reclaim_host_ownership()`、删
  `ci_queues`、按 `SIPU_CI_KEEP_DUMPS` 删 `ci_dumps`(第 101-105 行);
  `trap cleanup EXIT`(第 107 行)。
- `scripts/ci/sipu_ci_run.sh:109` — `compgen -G "$SIPU_CI_WHEEL_DIR"/*.whl`:
  **找不到宿主机构建的 wheel 直接 exit 1**(第 110-111 行)。
- `scripts/ci/sipu_ci_run.sh:114` — `replay_case_logs()`:live-log 模式直接返回
  (第 116 行);否则按 `ci_queues/cases` 逐条回放 sipu/compare 日志,用
  `sipu_ci_summary.py --check` 决定折叠(第 117-133 行)。
- `scripts/ci/sipu_ci_run.sh:136` — `run_cases()`:按 suite 计算
  `SIPU_CI_RUN_DIR`(第 139 行),`parallel>1` 时关 live log(第 140-144 行),建
  compare/logs/status 目录(第 145 行),打印统计(第 147-150 行),第 152-153 行
  `xargs -P <parallel>` 并行调 `sipu_ci_single_case.sh`(失败不中断),最后
  `sipu_ci_summary.py` 汇总(第 157-161 行)并 `replay_case_logs()`。
- `scripts/ci/sipu_ci_run.sh:166` — `run_named_suite()`:调 `sipu_ci_cases.py` 展开
  suite(第 175 行),解析 `#parallel=` 与 `#continue_on_fail=`(第 176-177 行),
  清 status、把非注释行写入 `ci_queues/cases`(第 178-179 行),再 `run_cases()`。
- `scripts/ci/sipu_ci_run.sh:192` — 单用例模式:置 `SIPU_CI_FAIL_FAST=1`,把
  `family|model|config|launch|case|timeout` 一行写入 `ci_queues/cases`
  (第 193-195 行),然后 `run_cases single 1`。

### 2.3 `scripts/ci/sipu_ci_qemu.sh`

- `scripts/ci/sipu_ci_qemu.sh:24` — `port_in_use()`:`ss -ltn` 判断端口是否被占。
- `scripts/ci/sipu_ci_qemu.sh:30` — `pick_free_port()`:从 12000-12999 里挑空闲
  端口(跳过已占用的第一个参数端口)。
- `scripts/ci/sipu_ci_qemu.sh:44` — `resolve_cmodel()`:读
  `sgl-kernel-sipu/sipu_sdk_path.txt` 第一行得到 SDK setup 脚本(第 48 行),推导
  `SI_CMODEL_ROOT=.../sipu1.5_cmodel`(第 51 行),校验其中的
  `bin/qemu-system-x86_64` 与 `sipu_cmodel_setup.sh`(第 52-55 行)。
- `scripts/ci/sipu_ci_qemu.sh:58` — `qemu_ssh()`:要求 `sshpass`(第 59-62 行),
  用密码 `123456` 以 root ssh 到 `127.0.0.1:$SSH_PORT`(第 63-66 行)。
- `scripts/ci/sipu_ci_qemu.sh:69` — `guest_ssh_ok()`:`qemu_ssh 'true'` 探活。
- `scripts/ci/sipu_ci_qemu.sh:76` — `prepare_guest()`:guest 内 9p 挂载
  `/share_data` 与 `/local_data`(第 79-91 行),`modprobe sipu`(第 92 行),最多
  等 20 秒直到出现 `/dev/sipu`(第 93-98 行)。
- `scripts/ci/sipu_ci_qemu.sh:102` — `load_ports()`:从 `qemu_run/ssh.port` 与
  `gdb.port` 读端口。
- `scripts/ci/sipu_ci_qemu.sh:108` — `cmdline_of()`:读 `/proc/<pid>/cmdline`。
- `scripts/ci/sipu_ci_qemu.sh:113` — `is_our_qemu()`:进程 cmdline 必须同时含
  `qemu-system-x86_64`、本 job 的 `hostfwd=tcp::${SSH_PORT}-:22` 与
  `$QCOW2_DST`(第 117-119 行)——避免误杀别人家的 QEMU。
- `scripts/ci/sipu_ci_qemu.sh:123` — `find_our_qemu()`:扫 `/proc/[0-9]*` 找匹配
  qemu(第 125-128 行);0 个返回 1,多个报错返回 2(第 129-135 行)。
- `scripts/ci/sipu_ci_qemu.sh:139` — `record_qemu_pid()`:把唯一 pid 写入
  `qemu.pid`。
- `scripts/ci/sipu_ci_qemu.sh:150` — `status()`:端口与 ssh 都通输出 `up`。
- `scripts/ci/sipu_ci_qemu.sh:159` — `stage_qemu_dir()`:校验
  `run_qemu.sh`、`archmodel.toml`、qcow2 存在(第 160-165 行),拷贝到
  `qemu_run/`(第 167-168 行),把 29G 的 `torch_sipu.overlay.qcow2` 拷到本地
  (第 169-174 行,除非 `SIPU_CI_SKIP_QCOW2_COPY=1`)。
- `scripts/ci/sipu_ci_qemu.sh:177` — `start()`:已起则直接
  `prepare_guest` + `record_qemu_pid`(第 178-183 行);否则
  `resolve_cmodel`(第 185 行)、`stage_qemu_dir`(第 186 行)、挑 SSH/GDB 端口并
  写文件(第 187-190 行),后台 `sg kvm -c './run_qemu.sh'` 启动(第 193-200 行),
  循环最多 90 次、每次 5 秒等 guest 起来(第 202-211 行),超时打印日志返回 1
  (第 212-215 行)。
- `scripts/ci/sipu_ci_qemu.sh:218` — `wait_gone()`:轮询 `/proc/$pid` 消失。
- `scripts/ci/sipu_ci_qemu.sh:227` — `stop()`:找不到 qemu 则返回(第 229-232 行);
  先 `qemu_ssh 'poweroff'`(第 244 行,失败忽略),等 20 秒(第 245 行),再
  SIGTERM(第 255 行)、SIGKILL(第 266 行)逐级关闭,每一步都先
  `is_our_qemu()` 复核(第 250/261 行)。

### 2.4 容器内真正跑测试的 `scripts/ci/sipu_ci_test.sh`

- `scripts/ci/sipu_ci_test.sh:10` — `SGLANG=/sgl-workspace/sglang`(宿主机挂进来的
  代码树);第 13 行 `KERNEL=$SGLANG/sgl-kernel-sipu`;第 14 行
  `SIPU=$SGLANG/test/srt/sipu`。
- `scripts/ci/sipu_ci_test.sh:15` — `DUMP/CUDA_DUMP/WHEEL_DIR/RUN_DIR` 都必须已
  由环境传入(第 15-18 行)。
- `scripts/ci/sipu_ci_test.sh:20` — `PYTHONPATH=$SGLANG/python:$SIPU/test_utils`。
- `scripts/ci/sipu_ci_test.sh:28` — 解析 `--config-yaml/--launch-config/--test-case/
  --timeout`(第 30-37 行),缺参 `usage()`(第 38 行)。
- `scripts/ci/sipu_ci_test.sh:40` — `cd "$KERNEL"`;第 43-47 行 source
  `setup.sh`、`setup_triton.sh`、`setup_tilelang.sh`;第 49 行
  `unset TORCH_DEVICE_BACKEND_AUTOLOAD`;第 50 行追加 `/opt/siorigin/lib` 到
  `LD_LIBRARY_PATH`。
- `scripts/ci/sipu_ci_test.sh:52` — 相对路径的 config 前补 `$SIPU/`(第 53 行),
  即 `test/srt/sipu/<config-yaml>`。
- `scripts/ci/sipu_ci_test.sh:58` — `case_uses_custom_deepep()`:用内嵌 python
  读 yaml 的 `launch_configs.<launch>.test_env.<case>.SGLANG_SIPU_USE_CUSTOM_DEEPEP`
  (第 59-77 行),为真退出码 0、为假 1、读不了 2。
- `scripts/ci/sipu_ci_test.sh:80` — 开启 `nullglob`;第 81-94 行按退出码决定
  `install_deepep=1`(第 86-88 行)或跳过(第 89-90 行)。
- `scripts/ci/sipu_ci_test.sh:95` — 需要 custom deep_ep 但 wheel 目录没有
  `deep_ep*.whl` 时报错退出(第 95-98 行)。
- `scripts/ci/sipu_ci_test.sh:99` — 第 99-106 行:遍历 `$WHEEL_DIR/*.whl`,
  `deep_ep*` 且不需要时跳过(第 101-104 行),其余
  `pip install --force-reinstall --no-deps`(第 105 行)。
- `scripts/ci/sipu_ci_test.sh:107` — 打印 `sgl_kernel.__file__` 验证安装。
- `scripts/ci/sipu_ci_test.sh:109` — `model`/`family` 从 config 文件名推得
  (第 109-110 行);compare 日志路径 `RUN_DIR/compare/<model>_<launch>_<case>.log`
  (第 111 行);sipu 运行日志 `RUN_DIR/logs/<family>/..._sipu.log`(第 115 行),
  并建目录(第 116 行)。
- `scripts/ci/sipu_ci_test.sh:122` — `is_compile`:`launch_config` 以 `_compile`
  结尾置 1(第 122-123 行)。
- `scripts/ci/sipu_ci_test.sh:125` — compile 模式:开 3 个 tiled 环境变量
  (第 127-130 行),CUDA golden 用去掉 `_compile/_graph` 后缀的 launch 名
  (第 131-132 行)。
- `scripts/ci/sipu_ci_test.sh:138` — CUDA golden 在
  `$CUDA_DUMP/$model/$cuda_launch/$TEST_CASE/cuda`(第 138 行);第 140-143 行
  不存在直接失败;第 144-147 行在
  `$DUMP/$model/$LAUNCH_CONFIG/$TEST_CASE/cuda` 处做符号链接指向 golden。
- `scripts/ci/sipu_ci_test.sh:150` — 第 150-154 行:`timeout --foreground
  --kill-after=30s $TIMEOUT` 包裹
  `python3 -u $SIPU/test_utils/run_test_job.py --config-yaml ... --device sipu
  --dump-base $DUMP --log-base $RUN_DIR/logs`,输出 `tee` 到 sipu 日志。
- `scripts/ci/sipu_ci_test.sh:156` — compile 模式检查
  `compile_runner.json` 契约(第 157-185 行),非
  `sipu_inductor_compile_only/inductor/graph_capture=false` 判 FAIL。
- `scripts/ci/sipu_ci_test.sh:188` — 第 188-197 行:调
  `analyse_utils/compare_cuda_sipu.py --dump-base $DUMP --launch-config ...` 比对
  CUDA/SIPU 精度 dump,输出到 compare 日志。

### 2.5 辅助脚本

- `scripts/ci/sipu_ci_cases.py:34` — `main()`:读 yaml、按 suite 展开;qemu 平台
  强制 `parallel=1`(第 51 行);输出 `#parallel/#continue_on_fail/#platform`
  (第 52-54 行);有 include 只输出 `#include=`(第 55-57 行);否则按 mode 过滤
  case,逐行输出 `parent|stem|config_yaml|launch|case|timeout`(第 58-76 行)。
- `scripts/ci/sipu_ci_summary.py`(docstring):判题门限
  `COS_SIM_MIN=0.999`、`MEAN_ATOL_MAX=0.05`;eager 要求退出 0 且每 pass
  `cos_sim>=0.999`、`mean_atol<=0.05`;compile 还要满足
  `compile_runner.json` 契约。写 `summary.tsv`/`summary.log`,有失败返回 1。

## 3. 本命令 `... --config-yaml configs/deepseek/ds_v3_2layer.yaml --launch-config
deepep_deepgemm_dp_ep --test-case text-only --platform qemu` 的执行动作

参数解析结果:`CONFIG_YAML=configs/deepseek/ds_v3_2layer.yaml`、
`LAUNCH_CONFIG=deepep_deepgemm_dp_ep`、`TEST_CASE=text-only`、
`PLATFORM=qemu`、`TIMEOUT=30m`(默认)、`TEST_SUITES=` 空、`MODE=eager`。

### 3.1 前置阶段(宿主机)

1. `scripts/ci/sipu_ci_exec.sh:79-89` 校验通过:单用例三参数齐全、无 `--mode`。
2. `scripts/ci/sipu_ci_exec.sh:92-93` source 单用例脚本并
   `detect_current_platform()` → `host`。
3. `scripts/ci/sipu_ci_exec.sh:95` 建 `ci_dumps/ci_wheels/ci_queues`。
4. `scripts/ci/sipu_ci_exec.sh:142-146` 无临时 kernel → `init_submodule()`:
   当前 `sgl-kernel-sipu` 已检出,跳过 clone。
5. `scripts/ci/sipu_ci_exec.sh:147` `setup_docker_args()`:宿主机分支,组装
   `$SGLANG:/sgl-workspace/sglang:rw` 等挂载,不传 `/dev/sipu`。
6. `scripts/ci/sipu_ci_exec.sh:148-150` `build_wheels()`:起 `sipu-ci-builder`
   容器,依次 `init_submodules.sh` → `setup.sh` → `make clean` →
   `make install-kernels` → 打 `sgl-kernel-sipu`、`deepep`、
   `siorigin_triton_kernels`、`siorigin_tilelang_kernels`、`triton_ops` 共 5 个
   wheel 到 `ci_wheels`;完成后 `reclaim_host_ownership()`。

### 3.2 分发与 QEMU 启动

7. `scripts/ci/sipu_ci_exec.sh:217-226` 单用例分支 →
   `dispatch qemu --config-yaml ... --launch-config ... --test-case ...`(第 226 行)。
8. `scripts/ci/sipu_ci_exec.sh:185-186` `SIPU_CI_PLATFORM=qemu` →
   `run_on_qemu()`。
9. `scripts/ci/sipu_ci_exec.sh:164` `sipu_ci_qemu.sh start`:
   - `resolve_cmodel()`:读 `sgl-kernel-sipu/sipu_sdk_path.txt` → SDK
     `/share_data/sicx_sdk/release/260827`,得到
     `sipu1.5_cmodel`(含 `qemu-system-x86_64`)。
   - `stage_qemu_dir()`:校验并把
     `/share_data/sglang_sipu/qemu_ci/torch_sipu.overlay.qcow2`(约 29G)拷到
     `<repo>/torch_sipu.overlay.qcow2`,同时拷 `run_qemu.sh`、`archmodel.toml`。
   - 挑 12xxx 空闲 SSH/GDB 端口,`sg kvm -c './run_qemu.sh'` 后台起 4 个
     `sipu1.5_cmodel` 设备(见 `archmodel.toml` 的 `sipu_num=4`)。
   - 轮询等 guest SSH;起来后 `prepare_guest()`:guest 内 9p 挂载
     `/share_data`、`/local_data`,`modprobe sipu`,等 `/dev/sipu` 出现。
10. `scripts/ci/sipu_ci_exec.sh:167-177` `sipu_ci_qemu.sh ssh ...`:把
    `SIPU_CI_*`、`CI_ROOT`、`SIPU_CI_IMAGE` 等导出到 guest,然后在
    `/sgl-workspace/sglang`(9p 共享的同一代码树)里执行
    `bash scripts/ci/sipu_ci_run.sh --config-yaml ... --launch-config ... --test-case ...`。

### 3.3 guest 内 `sipu_ci_run.sh`(单用例路径)

11. `scripts/ci/sipu_ci_run.sh:109-111` 检查 `ci_wheels/*.whl` 存在(guest 内
    9p 可见)。
12. `scripts/ci/sipu_ci_run.sh:192-196` 写 `ci_queues/cases` 一行:
    `deepseek|ds_v3_2layer|configs/deepseek/ds_v3_2layer.yaml|deepep_deepgemm_dp_ep|text-only|30m`,
    然后 `run_cases single 1`。
13. `scripts/ci/sipu_ci_run.sh:139` `run_dir_for()`:单用例走 eager 分支 →
    `ci_logs/single/<yyyyMMdd>`。
14. `scripts/ci/sipu_ci_run.sh:152-153` `xargs -P 1` 调
    `sipu_ci_single_case.sh 'deepseek|ds_v3_2layer|...|30m'`。

### 3.4 用例容器内 `sipu_ci_single_case.sh` + `sipu_ci_test.sh`

15. `sipu_ci_single_case.sh:122` 拆分用例;
    `sipu_ci_single_case.sh:65-86` guest 分支给 docker 加
    `SIRT_SHIM_NAME=ksipu`、`--net=host`、4 个 `/dev/sipu/usipu*` 设备。
16. `sipu_ci_single_case.sh:126-130` 组装命令并
    `run_container "sipu-ci-ds_v3_2layer_deepep_deepgemm_dp_ep_text-only-<pid>"`:
    `docker run --name <name>` 起 `harbor.siorigin.com/sglang-sipu/release:v0.5.18-sipu-dev-0.1.0`
    执行 `/bin/bash /sgl-workspace/sglang/scripts/ci/sipu_ci_test.sh --config-yaml
    configs/deepseek/ds_v3_2layer.yaml --launch-config deepep_deepgemm_dp_ep
    --test-case text-only --timeout 30m`,跑完再 `docker rm -f` 删除
    (`run_container()` 第 96-98 行)。
17. `sipu_ci_test.sh` 内:
    - `sipu_ci_test.sh:43-47` 加载 SDK/triton/tilelang 环境;
    - `sipu_ci_test.sh:58-94` 检测到 `test_env.text-only.SGLANG_SIPU_USE_CUSTOM_DEEPEP=1`
      → 需要 custom deep_ep wheel;
    - `sipu_ci_test.sh:99-106` 把 `ci_wheels` 里所有 wheel
      `pip install --force-reinstall --no-deps`(deep_ep 只装 custom 版);
    - `sipu_ci_test.sh:138-147` 检查 CUDA golden
      `/share_data/sglang_sipu/accuracy_verify/acndd/ds_v3_2layer/deepep_deepgemm_dp_ep/text-only/cuda`
      存在,并在 `ci_dumps` 下做软链;
    - `sipu_ci_test.sh:150-154` `timeout 30m` 跑 `run_test_job.py --device sipu`:
      按 yaml 用 **2 个相同 prompt、TP=2/DP=2/EP=2、`attention_backend=sipu`、
      `moe_runner_backend=deep_gemm`、`moe_a2a_backend=deepep`、
      `enable_dp_attention=true`、`base_gpu_id=0 gpu_id_step=2`** 在 4 个
      `sipu1.5_cmodel` 上做 SIPU 前向,把 tensor dump 写到 `ci_dumps`;
    - `sipu_ci_test.sh:188-197` `compare_cuda_sipu.py` 比对 CUDA golden 与 SIPU
      dump,输出 `ci_logs/single/<date>/compare/ds_v3_2layer_deepep_deepgemm_dp_ep_text-only.log`。
18. `sipu_ci_single_case.sh:137-139` chown 回宿主机用户,把 rc 写
    `ci_queues/status/<cid>`。
19. `sipu_ci_run.sh:157-161` `sipu_ci_summary.py` 判题:要求退出 0 且每个
    pass `cos_sim>=0.999`、`mean_atol<=0.05`。
20. `sipu_ci_exec.sh:178` 退出前 `sipu_ci_qemu.sh stop` 关 VM;
    `sipu_ci_exec.sh:96-104` cleanup 收尾。脚本退出码:全过 0,任一失败 1。

## 4. 本机是否具备执行条件

### 4.1 已具备(实测)

- 代码与镜像:仓库在 `/share/users/like/package/sglang_sipu`;
  `harbor.siorigin.com/sglang-sipu/release:v0.5.18-sipu-dev-0.1.0` 已 pull;
  **`sipu-dev` 容器正在运行**(Up 8 days),且 sglang 以 editable 方式装在
  `/sgl-workspace/sglang/python/sglang`(包名 sglang 0.5.18),`sgl-kernel-sipu`
  子模块已检出并含 `Makefile`、`sipu_sdk_path.txt`。
- 挂载:代码目录 `<repo> → /sgl-workspace/sglang`(rw);`/share_data/sicx_sdk`(ro)、
  `/share_data/arch_cmodel_release`(ro)、`/share_data/torch_sipu`(ro)、
  `/share_data/sglang_sipu`(rw)、`/share2`、`/softhome`、`/data_gpu` 等都挂进
  `sipu-dev`。
- QEMU 资源:SDK 含
  `/share_data/sicx_sdk/release/260827/sipu1.5_cmodel/bin/qemu-system-x86_64` 与
  `sipu_cmodel_setup.sh`;`/share_data/sglang_sipu/qemu_ci/` 有 `run_qemu.sh`、
  `archmodel.toml`(sipu_num=4)和约 29G 的 `torch_sipu.overlay.qcow2`。
- 系统:当前用户 `like` 在 `kvm` 组、`docker` 组;`/dev/kvm` 存在;12xxx 端口
  空闲;工作区可写;磁盘 `/share` 剩约 11T。
- 权限:`id -un` 为 `like`,与容器绑定的宿主用户一致,`chown`/读写均可行。

### 4.2 缺失/风险点(实测缺,需补或规避)

- **宿主机与 guest 都缺 `sshpass`**:`sipu_ci_qemu.sh:58-62` 的 `qemu_ssh()`
  会直接失败。需 `apt install sshpass`(guest 内的 `prepare_guest` 流程不受影响,
  但 `start` 的探活、`ssh`、`stop` 全依赖它)。
- **qcow2 拷贝开销**:`stage_qemu_dir()` 默认把 29G 的 qcow2 拷到仓库根
  (`$PWD/torch_sipu.overlay.qcow2`)。耗时大;可设
  `SIPU_CI_SKIP_QCOW2_COPY=1` 复用已拷好的文件(前提路径匹配
  `is_our_qemu()` 的 `$QCOW2_DST` 检查)。
- 机器名/并发:多 job 并发需 12xxx 端口足够;本机当前空闲。
- 注意:本脚本自建的 `sipu-ci-builder`/`sipu-ci-*` 容器与常驻的 `sipu-dev`
  是**不同容器**;脚本不直接用 `sipu-dev`,只是利用同一镜像和同一批共享挂载。

### 4.3 结论

- 代码层面:该命令的参数组合**合法**,会走
  `qemu 平台 + 单用例 + eager(默认)` 的完整链路;
- 本机**基本具备**执行条件(镜像、SDK、cmodel、qcow2、kvm、挂载、磁盘全齐),
  唯一硬性缺口是 **`sshpass` 未安装**,装上后即可直接跑;
- 若只想先验证代码本身,建议先跑 archmodel 单用例
  (`--platform archmodel` + `--launch-config deepep_deepgemm_text`),可跳过
  QEMU;要跑 `deepep_deepgemm_dp_ep`(DP/EP 分布式用例)则必须 qemu 平台
  (见 `scripts/ci/sipu_ci_suites.yaml:320-335`,`distributed` suite 指定
  `platform: qemu`)。

## 6.1 结论

`/share/users/like/audit.log:2-5` 是 `auditctl -l` 的规则列表，不是已经发生
的 audit 事件。根据这四条规则，当前机器只为以下 syscall 建立了显式的
`always,exit` 规则：

- `execve`、`execveat`，key 为 `gpu-monitor-exec`；
- `rename`、`renameat`、`renameat2`、`unlink`、`unlinkat`、`rmdir`，key 为
  `gpu-monitor-file`。

`mydd` 构建产物是 x86-64 ELF，因此正常运行时匹配 `arch=b64` 规则；
`arch=b32` 规则不适用于这个二进制。

直接结论是：

1. **启动 `mydd` 会被发现。** 正常启动可执行文件需要 `execve` 或
   `execveat`，会产生 key=`gpu-monitor-exec` 的事件。
2. **`mydd` 清空了哪些文件，当前规则不会发现。** 真正清空文件的是
   `ftruncate(fd, 0)`，但规则中没有 `ftruncate` 或 `truncate`。
3. **遍历、检查、打开和关闭文件也不会被这组规则记录。** 对应的
   `openat`、`getdents64`、`newfstatat`、`close` 等 syscall 均不在规则中。
4. **`gpu-monitor-file` 不会被 `mydd` 命中。** `mydd` 不删除、不重命名文件，
   也不删除目录；将文件长度改为零不等于 `unlink`。

所以 audit 能看到“某个用户在某个工作目录以某个阈值运行了 `mydd`”，但
仅凭现有规则不能看到“哪些文件被它截断、截断前多大，以及每个截断是否
成功”。

## 6.2 逐项对照 mydd 的运行时操作

下面路径均相对于 `/softhome/like/asset/code/cpp_guard`。

| 源码位置和函数 | 操作 | 本机实际 syscall/行为 | 当前规则是否记录 |
| --- | --- | --- | --- |
| 程序进入 `mydd.cpp:192-215 (main)` 之前 | 启动 `mydd` 进程映像 | `execve`（也可能由启动方使用 `execveat`） | **是**，`gpu-monitor-exec` |
| `mydd.cpp:192-204 (main)`、`mydd.cpp:40-57 (parse_options)` | 检查参数、解析 MiB、检查整数溢出 | 用户态计算 | 否 |
| `mydd.cpp:154-185 (scan_current_directory)` | 打开目录并递归枚举目录项 | `openat(O_DIRECTORY)`、`getdents64`、`newfstatat`、`close` | 否 |
| `mydd.cpp:167-175 (scan_current_directory)` | `symlink_status` 判断目录项类型 | 本机 libstdc++/glibc 使用 `newfstatat(..., AT_SYMLINK_NOFOLLOW)` | 否 |
| `mydd.cpp:79-97 (truncate_file)` | `lstat`、判断普通文件并比较大小 | 本机 glibc 将 `lstat` 实现为 `newfstatat(..., AT_SYMLINK_NOFOLLOW)` | 否 |
| `mydd.cpp:99-109 (truncate_file)` | 以 `O_WRONLY\|O_CLOEXEC\|O_NOFOLLOW` 打开待处理文件 | 本机 glibc 使用 `openat` | 否 |
| `mydd.cpp:111-132 (truncate_file)` | 用已打开的 fd 再次确认类型和大小 | 本机 glibc 使用 `newfstatat(fd, "", ..., AT_EMPTY_PATH)` 实现 `fstat` | 否 |
| `mydd.cpp:134-140 (truncate_file)` | 把文件长度设为零 | `ftruncate(fd, 0)` | **否，这是最关键的审计缺口** |
| `mydd.cpp:115-146 (truncate_file)` | 关闭文件描述符 | `close` | 否 |
| `mydd.cpp:59-63 (report_errno)`、`mydd.cpp:148-150 (truncate_file)`、`mydd.cpp:208-213 (main)` | 输出 warning、被截断文件名和汇总 | `write` 到 stdout/stderr | 否 |
| `mydd.cpp:214-215 (main)` | 退出进程 | `exit_group` | 否 |

动态链接器在 `main` 之前还会用 `openat`、`read`、`mmap`、`mprotect`、
`newfstatat` 和 `close` 加载 libc/libstdc++ 等动态库；这些也不在当前规则中。

我在临时目录中用当前构建的 `mydd 0` 做了 syscall 验证。可见一次
`execve`；每个实际清空的文件依次出现 `newfstatat`、`openat`、
`newfstatat`、`ftruncate`、`close`；没有出现 `rename`、`renameat*`、
`unlink`、`unlinkat` 或 `rmdir`。测试只操作临时测试文件。

## 6.3 audit 实际会留下什么信息

命中 `gpu-monitor-exec` 后，Linux Audit 通常会把同一个事件拆成
`SYSCALL`、`EXECVE`、`CWD`、`PATH`、`PROCTITLE` 等记录。因此通常可以看到：

- executable 路径；
- `argv[0]` 和 `argv[1]`，也就是 `mydd` 路径和 MiB 阈值；
- cwd；
- pid/ppid、uid/euid、auid、session、terminal；
- `execve` 是否成功。

因为这两条 exec 规则没有 `exe`、`path`、`uid` 或 `dir` 过滤器，它们会
匹配系统中相应架构的所有 `execve`/`execveat`，不只匹配 `mydd`。可以按
key 查询，例如：

```bash
sudo ausearch -k gpu-monitor-exec -x mydd -i
```

但这个执行事件只证明程序被启动。即使 `argv[1]=0`，也不能由它证明目录中
每个普通文件均成功执行了 `ftruncate`；文件可能在遍历期间消失、变更类型，
也可能因权限等原因打开或截断失败。

## 6.4 两组容易混淆的操作

### 6.4.1 `ftruncate` 不等于 `unlink`

`mydd.cpp:135 (truncate_file)` 的 `ftruncate(file_descriptor, 0)` 保留目录项
和 inode，只把 `st_size` 变成零。当前 `gpu-monitor-file` 规则只监控删除目录项
或重命名相关 syscall，所以不会因为文件内容被清空而触发。

### 6.4.2 `open(O_WRONLY)` 不等于 `open(O_TRUNC)`

`mydd.cpp:99-104 (truncate_file)` 打开文件时没有传 `O_TRUNC`；真正改变文件
长度的是后面的独立 `ftruncate`。不过即使改为 `open(..., O_TRUNC)`，当前规则
也仍然没有监控 `open`/`openat`，同样看不到文件内容被清空。

符号链接也不会被误认为已审计的删除操作：
`mydd.cpp:169 (scan_current_directory)` 使用 `symlink_status` 跳过它，
`mydd.cpp:100-102 (truncate_file)` 又用 `O_NOFOLLOW` 保护打开阶段；无论成功
跳过还是因竞态返回 `ELOOP`，都不会触发当前的 rename/unlink 规则。

## 6.5 编译和启动命令的边界

如果把“所有操作”也扩展到 CMake 编译过程，那么 `cmake`、构建工具、
编译器、assembler 和 linker 等每一次新进程启动都会命中全局 exec 规则。
编译产生 `.o` 和 `mydd` 二进制时使用的普通 `openat`/`write` 本身不会命中；
如果某个构建工具实际调用了 `rename` 或 `unlink` 置换、清理文件，则那些调用
会命中 `gpu-monitor-file`。这是构建工具的行为，不是 `mydd.cpp` 的运行时行为。

同样，shell 的输出重定向（例如 `./mydd 10 > result.log`）通常通过
`openat(..., O_TRUNC)` 创建或清空日志文件；现有规则也不会记录这次日志截断。

## 6.6 若目标是审计 mydd 的截断行为

至少需要把 `ftruncate` 加入 64 位 syscall 规则；如果还要覆盖其它程序使用
路径形式的截断，以及 32 位兼容程序，还需要相应覆盖 `truncate` 和 32 位
变体。概念上最小的 64 位补充是：

```bash
sudo auditctl -a always,exit -F arch=b64 -S ftruncate,truncate \
  -F key=gpu-monitor-truncate
```

不过 `ftruncate` 的 syscall 参数只有 fd 和新长度，单靠 syscall 记录不一定
能稳定给出便于阅读的目标绝对路径。若要求可靠回答“哪个目录下的哪个文件
被改写”，还应按实际保护目录增加 Audit 文件系统 watch/`dir` 规则，并经过
本机 audit 版本验证 PATH 记录。还要审计 `open(..., O_TRUNC)`、普通写入或
内存映射写入时，则需分别覆盖 `open/openat/openat2`、`write/pwrite*`、
`mmap` 等行为；只加 `ftruncate` 不能构成完整的文件内容修改审计。

以上结论仅基于 `/share/users/like/audit.log` 中列出的规则。`auditctl -l` 不显示
audit daemon 的运行状态、backlog 丢失情况，也不包含 SELinux/AppArmor AVC
等可能独立产生的安全事件；因此“规则应当匹配”与“事件最终已经持久化到
audit 日志”是两个不同判断。

## 7.1 按当前 `O_TRUNC` 实现重新核对

本节针对当前版本的
`/softhome/like/asset/code/cpp_guard/mydd.cpp`（已经把 `ftruncate` 改为
单次 `open(..., O_WRONLY | O_TRUNC, ...)`），因此**优先于第 6 节中针对旧版
`ftruncate` 实现的描述**。`/share/users/like/audit.log:2-5` 仍然只列出
`execve/execveat` 和 `rename*`、`unlink*`、`rmdir` 规则，没有
`open/openat`、`truncate/ftruncate`、`stat`、`write` 或目录遍历规则。

当前构建产物是 x86-64 ELF，所以正常运行匹配 `arch=b64`；`arch=b32` 规则
只有在 32 位进程实际发出 32 位 syscall 时才会匹配。两组规则都没有
`path`、`dir`、`exe`、`uid` 或 `comm` 过滤，是按 syscall 的全局规则。

## 7.2 当前源码操作与规则的对应关系

以下路径均相对于 `/softhome/like/asset/code/cpp_guard`。

| 源码位置和函数 | 当前操作 | 常见 Linux syscall/行为 | 当前 audit 结果 |
| --- | --- | --- | --- |
| `mydd.cpp:164-186 (main)` 进入程序之前 | 启动 `mydd` | `execve` 或 `execveat` | **会记录**，key=`gpu-monitor-exec` |
| `mydd.cpp:164-175 (main)`、`mydd.cpp:40-57 (parse_options)` | 解析 MiB 参数、溢出检查 | 用户态计算 | 不记录 |
| `mydd.cpp:126-160 (scan_current_directory)` | 打开目录、枚举递归目录项 | `openat(O_DIRECTORY)`、`getdents64`、`close` | 不记录 |
| `mydd.cpp:139-147 (scan_current_directory)` | 检查目录项类型 | `newfstatat(..., AT_SYMLINK_NOFOLLOW)` | 不记录 |
| `mydd.cpp:79-97 (truncate_file)` | `lstat`、检查是否普通文件、比较大小 | glibc 通常转为 `newfstatat(..., AT_SYMLINK_NOFOLLOW)` | 不记录 |
| `mydd.cpp:99-107 (truncate_file)` | 打开候选文件并截断 | 源码是 `open(..., O_WRONLY|O_TRUNC|...)`；glibc 在本机表现为 `openat` | **不记录** |
| `mydd.cpp:114-118 (truncate_file)` | 关闭 fd | `close` | 不记录 |
| `mydd.cpp:59-63 (report_errno)`、`mydd.cpp:120-122 (truncate_file)`、`mydd.cpp:180-185 (main)` | 输出错误、文件名和统计 | `write` 到 stdout/stderr | 不记录 |
| `mydd.cpp:186 (main)` | 进程结束 | `exit_group` | 不记录 |

`O_TRUNC` 是传给 `open/openat` 的 flag，由内核在打开文件时完成长度置零；
它不是名为 `truncate` 或 `ftruncate` 的独立 syscall。因此当前代码已经没有
任何命中 `gpu-monitor-file` 的操作：它不执行 `rename`、`renameat`、
`renameat2`、`unlink`、`unlinkat` 或 `rmdir`。

## 7.3 哪些信息会被 audit 看到

正常执行类似：

```bash
/softhome/like/asset/code/cpp_guard/build/mydd 100
```

通常会产生 `gpu-monitor-exec` 事件。由于规则是 `always,exit` 且没有程序
过滤，审计记录一般可包含 executable、`argv[0]`、`argv[1]`、cwd、pid/ppid、
uid/euid/auid 以及 syscall 成功状态。用 `sudo ausearch -k gpu-monitor-exec -i`
可以按 key 查询。

但是该事件不能证明文件处理结果：

- audit 不会记录 `mydd` 遍历过的文件列表；
- audit 不会记录哪些 `openat(O_TRUNC)` 成功或失败；
- audit 不会记录截断前的文件大小；
- audit 不会把 `mydd` 写到 stdout 的 `truncated '...'` 文本自动变成审计事件；
- `max_file_size_mib=0` 只会改变用户态分支，不能让当前规则额外覆盖截断。

如果通过 shell、脚本或构建系统启动 `mydd`，这些外围程序自身的
`execve/execveat` 也会因为规则全局匹配而被记录。外围程序若另外执行了
`rename*`、`unlink*` 或 `rmdir`，则会产生 `gpu-monitor-file` 事件，但这不是
`mydd.cpp` 的运行时操作。

## 7.4 本机 syscall 验证

在临时目录中运行当前构建的 `mydd 0`，实际观察到：

`execve`、目录遍历用的 `openat/getdents64/newfstatat`、候选文件的
`openat(..., O_WRONLY|O_TRUNC|O_NOFOLLOW|O_CLOEXEC)`、`close` 和输出用的
`write`。没有观察到 `ftruncate`、`truncate`、`rename*`、`unlink*` 或
`rmdir`。所以按 `/share/users/like/audit.log` 的规则集合，唯一确定会命中
的 `mydd` 运行时 syscall 是启动时的 `execve`（64 位规则）。该 trace 是
对临时测试文件的验证，不是 audit 日志事件本身。

## 7.5 如果要审计新版的实际截断

需要额外审计 `open/openat/openat2`，并根据 open flags 识别 `O_TRUNC`；或者
对需要保护的目录增加带写权限/属性权限的 Audit `dir`/path watch。若还要覆盖
其它程序直接调用 `truncate/ftruncate` 的情况，再另外加入对应 syscall。仅保留
当前的 `execve` 与删除/重命名规则，无法审计 `mydd` 的文件内容截断。

以上判断只依据 `/share/users/like/audit.log` 的规则列表；规则被选中不等于
事件一定已写入持久化日志，后者还取决于 audit 状态、backlog 和日志服务运行情况。
