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

# sikernel RMSNorm 实现与调用链

本文从
`/softhome/like/package/sikernel/source/source_builtin/attention/rms_norm/test/test_host.cpp`
开始，结合当前源码、SIPU ISA 文档
`/softhome/like/asset/code/isa/index.html` 和实际生成的 device 汇编，说明
RMSNorm 的 host 入口、wrapper dispatch、SIPU device kernel 以及测试验证流程。

## 1. 先给出结论

对输入的每一行 `x`，当前实现计算的是：

```text
sum  = sum_i(x[i] * x[i])
mean = sum / original_normalized_size
if eps_opt != 0:
    mean = mean + eps
rstd = 1 / sqrt(mean)
out[i] = x[i] * rstd
if weight_opt != 0:
    out[i] = out[i] * weight[i]
```

这是真正的 RMSNorm，不做 LayerNorm 中的 `x - mean(x)`。实现有两个特点：

1. normalized 维度按 1024B tile 处理，没有 tail，因此 wrapper 要求
   `normalized_size * sizeof(T)` 是 1024B 对齐的。
2. `bf16`/`fp16` 输入先转成 `fp32`，平方、累加、归约和缩放都使用 `fp32`，
   最后再转回原始 dtype；`fp32` 路径直接在 `fp32` tile 上运算。

本次测试的 `DATATYPE` 是 `sifmt::bfloat16`，所以实际主路径是
`rms_norm_bf16_kernel`，实际的非连续输入路径是
`rms_norm_bf16_strided_input_kernel`。

## 2. 文件和调用链

### 2.1 文件职责

| 文件 | 职责 |
| --- | --- |
| `test/test_host.cpp` | 分配 host/device buffer、构造 tensor metadata、调用 API、计算 golden、比较结果 |
| `kernel/rms_norm_kernel.su` | host-side wrapper；检查 metadata、选择 kernel、配置 grid/block、发起 device launch |
| `kernel/rms_norm_kernel_bf16.hpp` | `bf16` 连续 kernel 和 `bf16` strided-input kernel |
| `kernel/rms_norm_kernel_f16.hpp` | `fp16` 连续 kernel |
| `kernel/rms_norm_kernel_f32.hpp` | `fp32` 连续 kernel |
| `kernel/legacy_api.su` | 兼容旧裸指针接口，将裸指针包装为 `sikernel::tensor<T>` |
| `CMakeLists.txt` | 用 `scc_add_library` 编译 `.su` device/library，编译并链接 `test_host` |
| `README.md` | API、对齐条件和非连续布局的约束说明 |

### 2.2 总调用链

默认执行 `./build/test_host` 时，调用关系如下：

```text
test_host.cpp::main
├── run_case(16, 7168, 7168, false, false)
│   ├── malloc host_A/host_B/host_D/host_G
│   ├── sipuMalloc bo_A/bo_B/bo_D
│   ├── sipuMemcpy host -> device
│   ├── 构造 input_tensor/output_tensor/weight_tensor
│   ├── rms_norm<bfloat16>(tensor API)
│   │   └── rms_norm_launch<T>
│   │       ├── check_rms_norm_tensor_metadata
│   │       ├── is_contiguous_layout(input/output/weight)
│   │       └── rms_norm_contiguous_launch<T>
│   │           └── rms_norm_bf16_kernel<<<grid=8, cluster=1, block=2>>>
│   ├── sipuMemcpy device -> host
│   ├── golden(...)
│   └── 逐元素误差比较
└── run_strided_case(33, 512)
    ├── 构造 input stride=(2176, 1)
    ├── rms_norm<bfloat16>(tensor API)
    │   └── rms_norm_launch<T>
    │       ├── 连续性判断失败
    │       ├── 命中 bf16 特殊 stride case
    │       └── rms_norm_bf16_strided_input_kernel
    ├── sipuMemcpy device -> host
    ├── golden_strided_input(...)
    └── 逐元素误差比较
```

如果外部用户调用公共头文件中的旧裸指针接口，则还有一条不被本测试直接走到
的兼容链：

```text
legacy_api.su::rms_norm(const void* ...)
└── 创建 output_tensor/input_tensor/weight_tensor
    └── 调用 tensor 版本 rms_norm<T>(...)
        └── rms_norm_launch<T>(...)
```

### 2.3 构建链

```text
cd /softhome/like/package/sikernel
source setup.sh
└── 设置 SIKERNEL_ROOT_DIR
└── 加载 SIPU SDK
└── 加载 SIPU CMODEL
└── 默认设置 SIPU_ARCH=150

cd source/source_builtin/attention/rms_norm
bash build.sh
└── cmake -B build ./
    ├── scc_add_library(rms_norm SHARED ...)
    │   ├── kernel/rms_norm_kernel.su
    │   └── kernel/legacy_api.su
    └── add_executable(test_host test/test_host.cpp)
        └── 链接 librms_norm.so、sipurt、sipu
```

这里的 `.su` 不是普通 host C++ 源文件，而是由 SiCrossCompiler 面向 SIPU
架构编译。实际构建输出显示目标架构为 `sipu_150`。

## 3. test_host.cpp 入口和测试逻辑

### 3.1 类型和形状

`test_host.cpp:44-50` 定义了：

```cpp
#define K  7168
#define VK 7168
#define M  16
#define DATATYPE sifmt::bfloat16
```

因此默认连续 case 是：

```text
batch_size              = 16
normalized_size         = 7168
original_normalized_size= 7168
dtype                   = bf16
```

每行大小为 `7168 * 2 = 14336B`，即 `14` 个 1024B tile。

`K` 和 `VK` 当前相同，所以测试没有实际 padding 差异；接口仍然保留
`original_normalized_size`，用于表示均值分母中的原始长度。

### 3.2 golden 函数

`golden` 在 `test_host.cpp:52-69` 中执行 host 参考计算：

1. 外层 `j` 遍历 batch 中的行。
2. 内层 `i` 遍历 `normalized_size` 个物理元素并计算平方和。
3. 用 `original_normalized_size` 作除数得到 `mean`。
4. `eps_opt` 非零时加入 `eps`。
5. 用 `sqrt` 和倒数得到 `rstd`。
6. 每个元素乘以 `rstd`，`weight_opt` 非零时再乘 `weight[i]`。

注意一个容易忽略的语义：当前 `golden` 的分子循环长度是
`normalized_size`，分母却是 `original_normalized_size`。因此当两者不同时，
物理 padding 区域仍会参与平方和，只是分母使用原始长度。device kernel 也保持
同样行为。若调用者希望 padding 不参与分子，需要在输入中把 padding 显式置零，
或者修改 kernel 的有效元素处理逻辑。

`golden_strided_input` 在 `test_host.cpp:71-90` 中按
`row = pin + j * input_stride0` 取得每行起点，然后对逻辑上的
`normalized_size` 个连续元素计算参考值，输出按紧凑布局写入。

### 3.3 连续 case 的 host 流程

`run_case` 在 `test_host.cpp:171-302`：

- `nIn0 = normalized_size * batch_size`：输入物理元素数。
- `nIn1 = normalized_size`：weight 元素数。
- `nOut = normalized_size * batch_size`：输出元素数。
- `host_A`、`host_B` 随机初始化；输入分布为 `[-10, 100]`，weight 分布为
  `[-1, 1]`。
- `sipuMalloc` 分配三个 device buffer，两个 `sipuMemcpy` 将输入和 weight
  拷贝到 device。
- `sikernel::tensor` 的 initializer-list 构造函数自动生成 contiguous stride：
  对 shape `(batch_size, normalized_size)`，stride 是
  `(normalized_size, 1)`；weight shape `(normalized_size)` 的 stride 是 `(1)`。
- `input_tensor.data`、`output_tensor.data`、`weight_tensor.data` 被设置成
  device buffer 地址。
- 非 emulation 模式调用 `rms_norm<DATATYPE>(...)`。
- 拷回 output 后释放 device buffer，再执行 golden 和误差比较。

默认 `weight_opt=1`、`eps_opt=1`、`eps=1e-6`，所以 device 侧会执行 weight
乘法和 epsilon 加法。

### 3.4 strided case 的 host 流程

`run_strided_case(33, 512)` 在 `test_host.cpp:92-169`：

```text
input shape        = (33, 512)
input stride       = (2176, 1)
output shape       = (33, 512)
output stride      = (512, 1)
weight shape       = (512)
weight stride      = (1)
```

输入分配的物理长度是 `33 * 2176` 个 bf16，而不是 `33 * 512`。这表示每行
有间隔，但行内 normalized 维度仍然连续。输出是紧凑的 `33 * 512`。

该 case 传入 `rms_norm` 后，wrapper 不能走 contiguous kernel，因而命中
`rms_norm_bf16_strided_input_kernel`。device kernel 需要同时维护：

```text
input_batch_offset  = bid * input_stride0 * sizeof(bfloat16_t)
output_batch_offset = bid * normalized_size * sizeof(bfloat16_t)
```

所以读输入按 2176 元素跳行，写输出按 512 元素紧密排列。

### 3.5 main 的两种模式

`test_host.cpp:304-340` 有两种入口：

- 普通模式要求 `argc == 1`，依次运行 `(16,7168)` 连续 case 和
  `(33,512)` strided case。
- `--emu-multi <case_size>:<hidden_size> ...` 模式支持批量 emulation case，
  可以关闭 compare 并重复测量；每个 case 的 kernel 名称为
  `rms_norm_<hidden_size>`。

emulation 测量模式中，`rms_norm_timed` 仍然进入同一个
`rms_norm_launch`，只是传入时间戳和 perf event 输出参数。公共测试工具在
`repeat_count > 1` 时把第一次作为 warmup，不计入平均值，然后将结果写入
`dsv3_op_result_rms_norm.log`。

## 4. rms_norm_kernel.su：wrapper 和 dispatch

### 4.1 contiguous 判断

`is_contiguous_layout` 位于 `rms_norm_kernel.su:36-49`：

1. 要求 `dim` 非零，`sizes` 和 `strides` 的长度等于 `dim`。
2. 从最后一维开始，期望 stride 初始为 1。
3. 每一维检查 size 为正且实际 stride 等于期望值。
4. 下一维的期望 stride 乘上当前维度 size。

对于 2D `(M,K)`，只有 `(K,1)` 被认为 contiguous；对于 1D weight，只有
`(1)` 被认为 contiguous。

### 4.2 metadata 检查

`check_rms_norm_tensor_metadata` 位于 `rms_norm_kernel.su:51-81`，检查：

- output/input data 非空；weight 开启时 weight data 非空。
- input 和 output 必须是 2D。
- `sizes` 和 `strides` 必须有两个元素。
- input 的两个 size 必须为正。
- output shape 必须等于 input shape。
- `normalized_size * sizeof(T) % 1024 == 0`。
- `0 < original_normalized_size <= normalized_size`。
- weight 开启时必须是 1D，长度等于 normalized size。

该检查是 kernel 没有 tail 处理的前置条件。对 bf16/fp16，它等价于
normalized size 是 512 元素对齐；对 fp32，它等价于 256 元素对齐。

### 4.3 grid、cluster、block

`rms_norm_contiguous_launch` 在 `rms_norm_kernel.su:83-160` 中设置：

```cpp
uint32_t block_dim = 1;
uint32_t cluster_dim = 1;
uint32_t grid_dim = 1;

if (batch_size >= 2)
    block_dim = 2;
if (batch_size >= 32)
    grid_dim = 16;
else
    grid_dim = (batch_size + 1) / 2;
```

因此：

- batch 为 1 时使用 1 个 thread。
- batch 为 2 到 31 时使用 2 threads/block，并用足够多的 block 覆盖 batch。
- batch 大于等于 32 时固定 16 blocks、每 block 2 threads，共 32 threads。
- cluster 维度固定为 1。

对于默认 `batch_size=16`，launch 维度是 `grid=8, cluster=1, block=2`，总共
16 个 thread，通常每个 thread 负责一行。对于 strided `batch_size=33`，维度
是 `grid=16, block=2`，总共 32 个 thread，最后一个 thread 通过 row loop
处理 `bid=32`。

SIPU 的 launch 形式是：

```cpp
kernel<<<grid, cluster, block, 0, stream>>>(...);
```

第四个参数是动态 shared memory 大小，本实现传 0，因为 shared memory 是由
device kernel 中的静态 `__shared__` 数组声明的；最后一个参数是 SIPU stream。

### 4.4 dtype dispatch

`rms_norm_contiguous_launch` 用 `if constexpr` 根据 `T` 选择：

- `sifmt::float32` -> `rms_norm_f32_kernel`
- `sifmt::float16` -> `rms_norm_f16_kernel`
- `sifmt::bfloat16` -> `rms_norm_bf16_kernel`

当 `kernel_elapsed_ms != nullptr` 时，wrapper 在 kernel launch 两侧包围
`SIKERNEL_PERF_EVENT_START/STOP`，同时用 `record_host_timestamp` 记录 API
launch 时间。普通模式两个指针都是空，不走 perf event。

### 4.5 non-contiguous dispatch

`rms_norm_launch` 在 `rms_norm_kernel.su:204-240` 中执行：

1. 先做 metadata 检查。
2. 取 `batch_size=input.sizes[0]`、`normalized_size=input.sizes[1]`。
3. 检查 input/output/weight 是否 contiguous。
4. 三者都连续时直接走 `rms_norm_contiguous_launch`。
5. 如果 `T=bf16`，进一步要求：
   - normalized size 只能是 512 或 1536；
   - `original_normalized_size == normalized_size`；
   - input stride 必须精确是 `(2176,1)`；
   - output stride 必须是 `(normalized_size,1)`；
   - weight 开启时 weight stride 必须是 `(1)`。
6. 命中后调用 `rms_norm_bf16_strided_input_launch`。
7. 其它非连续布局直接 `sipu::check(false, ...)` 报错。

这里的限制不是 ISA 的一般 strided load 能力，而是当前 RMSNorm wrapper 只
实现并验证了这两个固定输入布局。

## 5. SIPU ISA 中 tile 的关键概念

### 5.1 tile 大小和 element 数

SIPU ISA 文档的 unit-stride tile load/store 以 1024B 的完整 tile 为基本单位：

- `m1` = 1024B
- `m2` = 2048B
- `m4` = 4096B
- `m8` = 8192B
- `mf2` = 512B
- `mf4` = 256B
- `mf8` = 128B

因此一个 `m1` 中的元素数取决于 dtype：

| dtype | 一个 m1 的元素数 |
| --- | ---: |
| fp32 | 1024 / 4 = 256 |
| fp16 | 1024 / 2 = 512 |
| bf16 | 1024 / 2 = 512 |

这正好对应源码中的：

```cpp
// f32
norm_tile_num = norm_size / 256;

// f16/bf16
norm_tile_num = norm_size / 512;
```

### 5.2 offset 是 byte offset

ISA 文档中 unit-stride load/store 的地址由 base 加上 offset 构成；源码把
`batch_offset` 写成：

```cpp
bid * norm_size * sizeof(dtype)
```

并把每个 tile 的增量固定为 `1024`。生成的 device 汇编也显示动态 offset
放在寄存器中，例如：

```text
tld.trir.linear.u32.global T7, (a1), 0x0, s11
tst.trir.linear.u32.global T18, (a0), 0x0, s10
```

因此这里 `tld_linear_global_m1(ptr, offset)` 和
`tst_linear_global_m1(tile, ptr, offset)` 的 offset 应按 byte 理解；
`1024` 表示下一个 1024B tile，而不是下一个 1024 个 bf16 元素。

### 5.3 本实现用到的 intrinsic 对照

| C++ intrinsic | device 汇编/ISA 类别 | 作用 |
| --- | --- | --- |
| `tld_linear_global_m1` | `tld.trir.linear.u32.global` | 从 global memory 连续加载 1 个 1024B tile |
| `tst_linear_global_m1` | `tst.trir.linear.u32.global` | 把 1 个 tile 连续写回 global memory |
| `tst_linear_share_m1` | `tst.trir.linear.u32.share` | 把 1 个 tile 写入 shared memory |
| `tld_linear_share_m1` | `tld.trir.linear.u32.share` | 从 shared memory 连续读取 1 个 tile |
| `tmv_t_f32(scalar)` | `tmv.trr.broadcast` | 把 scalar 广播到完整 tile |
| `tcvt_f32(bf16_tile,t_r32)` | `tcvt.tt.f32.bf16.r32` | bf16 -> fp32，元素宽度扩大，m1 -> m2 |
| `tcvt_bf16(f32_tile,t_r32)` | `tcvt.tt.bf16.f32.r32` | fp32 -> bf16，元素宽度缩小，m2 -> m1 |
| `tmul(a,b)` | `tmul.ttt.f32` | tile 级逐元素乘法 |
| `tadd(a,b)` | `tadd.ttt.f32` | tile 级逐元素加法 |
| `tmv_v_f32(tile,index)` | `tmv.vtr.e32` | 取 tile 的一个 128B segment 到 RVV 向量寄存器 |
| `twait_store_share(0)` | `twait.i.store.share 0` | 等待 shared-memory store 请求全部完成 |

ISA 文档对 `tmv.vtr.e32` 的要求是 RVV 设置为 `SEW=32, vl=32, lmul=1`。
源码没有手写 `vsetvl`，但当前编译器在生成汇编中插入了：

```text
vsetivli zero, 0x1, e32, m1, ta, ma
vsetvli  a5, zero, e32, m1, ta, ma
```

随后才发出 `tmv.vtr.e32` 和 `vfredosum.vs`。所以源代码层面依赖 SiCrossCompiler
对 RVV intrinsic 的 VTYPE/VL 管理；如果改用其它编译器或手写汇编，应显式满足
ISA 的这个前置条件。

## 6. bf16 contiguous device kernel 逐行讲解

源码文件：
`kernel/rms_norm_kernel_bf16.hpp:25-112`。

### 6.1 函数签名和局部状态：25-47 行

```cpp
25  __global__ void rms_norm_bf16_kernel(
       sifmt::bfloat16* pout,
       sifmt::bfloat16* pin,
       sifmt::bfloat16* weight,
       float eps,
       int64_t batch_size,
       int64_t norm_size,
       int64_t original_norm_size,
       int weight_opt,
       int eps_opt) {
```

- `__global__` 表示这是 device kernel，host 通过 triple-chevron launch。
- `pout`、`pin`、`weight` 是 global memory 地址。
- `eps` 是 epsilon；`batch_size` 是行数；`norm_size` 是物理 normalized 长度。
- `original_norm_size` 只用于均值分母。
- `weight_opt`、`eps_opt` 是运行时开关，而不是编译期模板参数。

```cpp
29  const int tile_size = 1024;
```

一个 unit-stride `m1` tile 固定 1024B；后面的 offset 以 byte 为单位。

```cpp
30  __andescore_fp_mode(BF16);
```

告诉 Andes/SIPU 编译器和 device 浮点执行环境：当前输入/输出的低精度格式
是 BF16。它不是 RMSNorm 数学操作，而是为 BF16 tile 操作选择正确的 FP mode。

```cpp
31  tbfloat16m1_t bf16_din;
32  tbfloat16m1_t bf16_wgt;
```

- `bf16_din` 保存从输入加载的一个 BF16 `m1` tile。
- `bf16_wgt` 保存一个 weight tile。
- 每个 tile 是 1024B，即 512 个 bf16 元素。

```cpp
33  tfloat32m2 f32_din;
34  tfloat32m2 f32_wgt;
35  tfloat32m2 f32_sq;
36  tfloat32m2 f32_sq_sum;
```

BF16 转 FP32 后，每个元素从 2B 变成 4B，所以一个 BF16 m1 的 1024B
数据会变成 2048B，需要一个 FP32 `m2`。`m2` 包含两个 FP32 m1 部分，源码
通过 `.m1e0`、`.m1e1` 分别访问这两个部分：

- `f32_din`：转换后的输入。
- `f32_wgt`：转换后的 weight。
- `f32_sq`：输入平方。
- `f32_sq_sum`：跨 tile 累加的平方和。

```cpp
37  tfloat32m1_t tile_rsqrt;
38  float sum;
39  float mean;
40  float sqrt;
41  tfloat32m2 f32_out;
42  tbfloat16m1_t bf16_out;
```

- `tile_rsqrt` 是广播后的 `rstd`，每个 FP32 元素都相同。
- `sum`、`mean`、`sqrt` 是 scalar FP32，用于最终归约和开根。
- `f32_out` 是 FP32 中间输出，仍然是 m2。
- `bf16_out` 是转回 BF16 的 m1 输出。

```cpp
43  int64_t norm_tile_num = norm_size / 512;
```

一个 BF16 m1 包含 512 个元素；对齐检查已经保证这里没有余数，因此该值是
每一行必须处理的完整 tile 数。默认 `7168 / 512 = 14`。

```cpp
44  const int shared_buffer_size =
        512*1024/sizeof(bfloat16_t)/2;
```

计算每个 `shared_buff[threadIdx.x]` 可容纳的 BF16 元素数：

```text
512 KiB / 2B / 2 threads = 131072 bf16 elements
131072 * 2B = 256 KiB per thread
```

```cpp
45  __shared__ bfloat16_t shared_buff[2][shared_buffer_size];
```

为 block 的两个 thread 各分配一块 shared memory。当前 launch 逻辑保证
`blockDim.x` 最大为 2，所以第一维用 `threadIdx.x` 不会越界。整个数组占：

```text
2 * 131072 * 2B = 524288B = 512 KiB
```

生成的 `.llvm.resource` 也显示该 kernel 的 `.sram = 524288`。这里的第一维
是 thread 私有区域，不是两个 ping-pong buffer。

```cpp
46  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
47  unsigned int all_threads = gridDim.x * blockDim.x;
```

- `tid` 是整个 grid 中的线性 thread id。
- `all_threads` 是 grid 中的总 thread 数。
- kernel 采用 grid-stride loop，让每个 thread 以 `all_threads` 为步长领取多行。

### 6.2 行分工和第一遍：48-64 行

```cpp
48  for (long bid = tid; bid < batch_size;
        bid += all_threads) {
```

每次循环处理一整行 `bid`。当 batch 大于总 thread 数时，同一个 thread 会在
后续迭代处理 `bid + all_threads`。

```cpp
49  int64_t batch_offset =
        bid * norm_size * sizeof(bfloat16_t);
```

计算该行在 global memory 中的 byte 起点。连续布局下第 `bid` 行紧跟在前一行
之后，所以 stride 是 `norm_size * 2` bytes。

```cpp
50  f32_sq_sum.m1e0 = tmv_t_f32(0U);
51  f32_sq_sum.m1e1 = tmv_t_f32(0U);
```

将两个 FP32 m1 累加器都广播初始化为 0。`tmv_t_f32` 属于 scalar-to-tile
move/broadcast；这两行把整个 `f32_sq_sum` m2 的所有 lanes 清零。

```cpp
52  for (int64_t i = 0; i < norm_tile_num; ++i) {
53      int64_t offset = batch_offset + i * tile_size;
```

遍历当前行的每个 1024B tile：

```text
第 i 个 tile 的 global byte offset = 行起点 + i * 1024
```

```cpp
54  bf16_din = tld_linear_global_m1(
        (bfloat16_t*)pin, offset);
```

从 global memory 连续加载 1024B BF16 tile。ISA 类别是
`tld.trir.linear.u32.global`；`linear` 表示 unit-stride，`m1` 表示一个完整
1024B tile。

```cpp
55  if (i < shared_buffer_size / 512) {
56      tst_linear_share_m1(
            bf16_din,
            shared_buff[threadIdx.x],
            i * tile_size);
57  }
```

`shared_buffer_size / 512 = 256`，所以前 256 个 tile 会同时写入 shared memory。

- `shared_buff[threadIdx.x]` 选择当前 thread 的私有行缓存。
- `i * tile_size` 是 shared memory 中的 byte offset。
- `tst_linear_share_m1` 是 unit-stride shared-memory store。
- 当前默认行只有 14 个 tile，全部会被缓存；如果 normalized 维度超过
  `256 * 512 = 131072` 个 BF16 元素，超出部分不会缓存。

```cpp
59  f32_din.m2e0 = tcvt_f32(bf16_din, t_r32);
```

把 BF16 m1 转成 FP32 m2，转换布局参数是 `t_r32`。由于元素宽度扩大一倍，
1024B BF16 数据变成 2048B FP32 数据，因此返回值必须是 m2。生成汇编对应：

```text
tcvt.tt.f32.bf16.r32 T10, T6
```

源码字段名是 `.m2e0`，表示把转换得到的两个 FP32 m1 部分作为一个 m2 视图
存放。

```cpp
60  f32_sq.m1e0 = tmul(f32_din.m1e0, f32_din.m1e0);
61  f32_sq.m1e1 = tmul(f32_din.m1e1, f32_din.m1e1);
```

分别对 m2 的两个 FP32 m1 部分做逐元素平方。ISA 对应两个
`tmul.ttt.f32` tile ALU 操作。

```cpp
62  f32_sq_sum.m1e0 = tadd(
        f32_sq_sum.m1e0, f32_sq.m1e0);
63  f32_sq_sum.m1e1 = tadd(
        f32_sq_sum.m1e1, f32_sq.m1e1);
```

把当前 tile 的平方结果加到跨 tile 累加器中。每个 thread 独立处理一行，
不需要 thread 间共享归约，也没有 `tsync`。

```cpp
64  }
```

第一遍结束后，`f32_sq_sum` 中包含当前行所有 `norm_size` 个物理元素的平方和，
但它仍然是按 tile/segment 分布在寄存器中的形式。

### 6.3 FP32 tile 到 RVV 的归约：65-78 行

```cpp
65  f32_sq_sum.m1e0 = tadd(
        f32_sq_sum.m1e0,
        f32_sq_sum.m1e1);
```

把 m2 的两个 FP32 m1 部分先合并成一个 m1。此时一个 m1 有 256 个 FP32
元素，正好对应一整行的一个“归约输入 tile”中的 256 个平方值。

```cpp
66  vfloat32m1_t vsum_vec =
        tmv_v_f32(f32_sq_sum.m1e0, 0U);
```

SIPU tile 的一个 m1 是 1024B，但 `tmv.vtr.e32` 一次把其中一个 128B
segment 搬到 RVV vector，因此 `index=0` 取出前 32 个 FP32 元素。

ISA 文档中 `tmv.vtr.e32` 的含义是：取 tile 指定 index 的 128B 数据到 RVV
向量寄存器。对 FP32 来说，128B / 4B = 32 lanes。

```cpp
67  vfloat32m1_t zero_vec =
        __riscv_vfmv_v_f_f32m1(0.0f, 1U);
68  vfloat32m1_t temp_vec;
```

- `zero_vec` 的 lane 0 作为 reduction seed，值为 0。
- `vl=1` 足够，因为 `vfredosum.vs` 使用 seed vector 的标量首元素。
- `temp_vec` 保存后续 tile segment。

```cpp
69  for (long i = 1; i < 8; ++i) {
70      temp_vec = tmv_v_f32(
            f32_sq_sum.m1e0, i);
71      vsum_vec = __riscv_vfadd_vv_f32m1(
            temp_vec, vsum_vec, 32);
72  }
```

一个 FP32 m1 有 `1024 / 128 = 8` 个 128B segment：

- 第 66 行得到 segment 0。
- `i=1..7` 依次得到 segment 1 到 7。
- `vfadd.vv` 对 32 个 lanes 做逐 lane 相加。

执行完后，`vsum_vec[lane]` 等于 tile 中相隔 128B 的 8 个元素之和；32 个
lane 共同覆盖原 m1 的 256 个 FP32 元素。

生成的汇编正是 7 次 `tmv.vtr.e32` 加 `vfadd.vv`。

```cpp
73  #ifdef RVV_UNORDER_REDUCE
74      zero_vec = __riscv_vfredusum_vs_f32m1_f32m1(
            vsum_vec, zero_vec, 32);
75  #else
76      zero_vec = __riscv_vfredosum_vs_f32m1_f32m1(
            vsum_vec, zero_vec, 32);
77  #endif
```

把 32 个 RVV lanes 做标量求和：

- `vfredusum.vs` 是 unordered reduction，允许更自由的结合顺序。
- `vfredosum.vs` 是 ordered reduction，结果更接近固定顺序累加。
- 当前构建的汇编显示 `vfredosum.vs`，说明本次没有定义
  `RVV_UNORDER_REDUCE`。
- reduction 的初始值来自 `zero_vec[0] = 0`，结果也放在 `zero_vec[0]`。

```cpp
78  sum = __riscv_vfmv_f_s_f32m1_f32(zero_vec);
```

把 reduction 结果的 lane 0 取回 scalar `float`，得到当前行平方和。

### 6.4 计算 rstd：79-88 行

```cpp
79  mean = sum / original_norm_size;
80  if (eps_opt) mean += eps;
```

得到 RMSNorm 的分母项：

```text
mean = sum(x[i]^2) / original_normalized_size + eps (可选)
```

注意分子在第一遍按 `norm_size` 个物理元素累加，分母使用传入的
`original_norm_size`。

```cpp
81  asm volatile (
82      "fsqrt.s %0, %1"
83      : "=f"(sqrt)
84      : "f"(mean)
85  );
```

通过 inline assembly 发出 RISC-V `fsqrt.s`，对 scalar FP32 `mean` 开平方。
这里没有使用 tile SFU，而是先把归约结果变成 scalar，再用 scalar FP 指令。

```cpp
86  sqrt = 1.0 / sqrt;
```

把标准差形式的 `sqrt(mean)` 变成 RMSNorm 需要的倒数
`rstd = 1/sqrt(mean)`。

```cpp
87  tile_rsqrt = tmv_t_f32(sqrt);
```

把 scalar `rstd` 广播到完整 FP32 m1 tile，后面用同一个 tile 乘数对输入做
逐元素缩放。生成汇编对应 `tmv.trr.broadcast`。

```cpp
88  twait_store_share(0);
```

这是本 kernel 中最重要的同步点之一。第一遍中的
`tst_linear_share_m1` 是 tile LSU 的 shared-memory store；ISA 文档中
`twait.i.store.share cnt` 的 `cnt=0` 表示等待剩余 shared store 数量降到 0，
即保证前面写入 shared buffer 的 tile 已经可被第二遍读取。

它不是 thread block barrier：

- 本 kernel 每个 thread 只读写自己的 `shared_buff[threadIdx.x]`。
- 没有 thread 之间的数据交换。
- 因此不需要 `tsync` 或 `__syncthreads` 来等待另一个 thread 的数据。

当前 device 汇编确认生成了：

```text
twait.i.store.share 0x0
```

### 6.5 第二遍：归一化、weight、写回：89-112 行

```cpp
89  for (int64_t i = 0; i < norm_tile_num; ++i) {
90      int64_t offset = i * tile_size;
91      int64_t g_offset = batch_offset + offset;
```

第二遍再次遍历所有 tile：

- `offset` 是当前行内部的 byte offset。
- `g_offset` 是 global input/output 的绝对 byte offset。

```cpp
92  if (i < shared_buffer_size / 512) {
93      bf16_din = tld_linear_share_m1(
            shared_buff[threadIdx.x], offset);
94  } else {
95      bf16_din = tld_linear_global_m1(
            (bfloat16_t*)pin, g_offset);
96  }
```

前 256 个 tile 从 shared memory 读，超出 shared capacity 的 tile 从 global
memory 重读。默认 7168 维只有 14 个 tile，所以默认 case 的第二遍完全走
`tld_linear_share_m1`，避免再次访问输入 global memory。

```cpp
97  f32_din.m2e0 = tcvt_f32(bf16_din, t_r32);
```

把当前 BF16 tile 再次转成 FP32 m2，因为 `tile_rsqrt` 和后续 ALU 都是 FP32
tile。此处与第一遍第 59 行相同。

```cpp
98  f32_out.m1e0 = tmul(
        f32_din.m1e0, tile_rsqrt);
99  f32_out.m1e1 = tmul(
        f32_din.m1e1, tile_rsqrt);
```

对 m2 的两个 FP32 m1 部分执行逐元素乘法：

```text
f32_out = f32_din * rstd
```

```cpp
100 if (weight_opt) {
101     bf16_wgt = tld_linear_global_m1(
            (bfloat16_t*)weight, offset);
102     f32_wgt.m2e0 = tcvt_f32(
            bf16_wgt, t_r32);
103     f32_out.m1e0 = tmul(
            f32_out.m1e0, f32_wgt.m1e0);
104     f32_out.m1e1 = tmul(
            f32_out.m1e1, f32_wgt.m1e1);
105 }
```

weight 是一维、按 normalized 维度复用的 scale，不随 batch 变化，因此 offset
只使用 `i * 1024`，不加 `batch_offset`。

- 第 101 行从 weight global memory 读取与当前输入 tile 对齐的 BF16 weight tile。
- 第 102 行将 weight 转成 FP32 m2。
- 第 103-104 行分别完成两半的逐元素乘法。
- `weight_opt=0` 时整个分支跳过，输出只做 RMS 缩放。

这几行之后，数学结果是：

```text
f32_out[i] = input[i] * rstd * weight[i]
```

```cpp
106 bf16_out = tcvt_bf16(
        f32_out.m2e0, t_r32);
```

把 FP32 m2 按 `t_r32` 转回 BF16 m1。这里的 `m2e0` 是包含两个 FP32 m1
部分的 m2 视图，转换后数据量从 2048B 缩回 1024B。

```cpp
107 tst_linear_global_m1(
        bf16_out,
        (bfloat16_t*)pout,
        g_offset);
```

将 BF16 m1 以 unit-stride 方式写回输出的当前行位置。ISA 对应
`tst.trir.linear.u32.global`，写回 1024B。

```cpp
109 }
111 }
112 }
```

- 第 109 行结束第二遍 tile loop。
- 第 111 行结束 batch/grid-stride row loop。
- 第 112 行结束 device kernel。

因此一个 thread 对一行的完整执行顺序是：

```text
global load BF16 tile
-> optional shared store
-> BF16 to FP32
-> square
-> FP32 tile accumulation
-> RVV horizontal reduction
-> scalar sqrt and reciprocal
-> wait shared stores
-> shared/global reload
-> BF16 to FP32
-> multiply rstd
-> optional load/convert/multiply weight
-> FP32 to BF16
-> global store
```

## 7. bf16 strided-input device kernel 逐行讲解

源码文件：
`kernel/rms_norm_kernel_bf16.hpp:114-201`。

这个 kernel 的计算和连续 kernel 完全相同，唯一的核心差异是 input 和 output
的行地址分开计算。

### 7.1 签名和局部变量：114-134 行

```cpp
114 __global__ void rms_norm_bf16_strided_input_kernel(
        sifmt::bfloat16* pout,
        sifmt::bfloat16* pin,
        sifmt::bfloat16* weight,
        float eps,
        int64_t batch_size,
        int64_t norm_size,
        int64_t original_norm_size,
        int64_t input_stride0,
        int weight_opt,
        int eps_opt) {
```

新增的 `input_stride0` 是 input 第 0 维的 stride，单位是元素数。当前 wrapper
只允许它等于 2176。

```cpp
116 const int tile_size = 1024;
117 __andescore_fp_mode(BF16);
118-129 与 contiguous kernel 相同的 tile 类型声明
130 int64_t norm_tile_num = norm_size / 512;
131 const int shared_buffer_size =
        512*1024/sizeof(bfloat16_t)/2;
132 __shared__ bfloat16_t shared_buff[2][shared_buffer_size];
133 unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
134 unsigned int all_threads = gridDim.x * blockDim.x;
```

这些行与连续 kernel 的第 29-47 行含义完全一致：仍然是 BF16 m1、FP32 m2、
每 thread 256KiB shared cache、最大 2 threads/block 和 grid-stride row loop。

### 7.2 第一遍地址计算和缓存：135-153 行

```cpp
135 for (long bid = tid; bid < batch_size;
        bid += all_threads) {
136     int64_t input_batch_offset =
        bid * input_stride0 * sizeof(bfloat16_t);
137     int64_t output_batch_offset =
        bid * norm_size * sizeof(bfloat16_t);
```

这里是整个 strided kernel 的关键：

- input 行起点按 `input_stride0` 计算，因此第 `bid` 行与上一行相隔
  `2176 * 2 = 4352B`。
- output 是 contiguous，行起点按 `norm_size * 2` 计算；对 `norm_size=512`，
  输出行间距为 1024B。

```cpp
138 f32_sq_sum.m1e0 = tmv_t_f32(0U);
139 f32_sq_sum.m1e1 = tmv_t_f32(0U);
```

初始化 FP32 m2 平方和累加器为零，与连续 kernel 第 50-51 行相同。

```cpp
140 for (int64_t i = 0; i < norm_tile_num; ++i) {
141     int64_t offset = i * tile_size;
142     int64_t input_offset = input_batch_offset + offset;
143     bf16_din = tld_linear_global_m1(
        (bfloat16_t*)pin, input_offset);
```

`offset` 是当前行内部的 tile byte offset；`input_offset` 是带 stride 的真实
输入 byte 地址。因此虽然每行间有 padding，行内仍然可以用 unit-stride m1 load。

```cpp
144 if (i < shared_buffer_size / 512) {
145     tst_linear_share_m1(
        bf16_din, shared_buff[threadIdx.x], offset);
146 }
```

把逻辑输入 tile 紧凑地缓存到 shared memory。shared cache 不复制输入行之间的
padding，只保存 normalized 区域，所以第二遍使用的是逻辑 offset `offset`。

```cpp
148 f32_din.m2e0 = tcvt_f32(bf16_din, t_r32);
149 f32_sq.m1e0 = tmul(f32_din.m1e0, f32_din.m1e0);
150 f32_sq.m1e1 = tmul(f32_din.m1e1, f32_din.m1e1);
151 f32_sq_sum.m1e0 = tadd(
        f32_sq_sum.m1e0, f32_sq.m1e0);
152 f32_sq_sum.m1e1 = tadd(
        f32_sq_sum.m1e1, f32_sq.m1e1);
153 }
```

完成 BF16 -> FP32、平方和跨 tile 累加。因为 `input_stride0` 只影响 load 地址，
数学计算与连续 kernel 不变。

### 7.3 归约和 rstd：154-177 行

```cpp
154 f32_sq_sum.m1e0 = tadd(
        f32_sq_sum.m1e0, f32_sq_sum.m1e1);
155 vfloat32m1_t vsum_vec =
        tmv_v_f32(f32_sq_sum.m1e0, 0U);
156 vfloat32m1_t zero_vec =
        __riscv_vfmv_v_f_f32m1(0.0f, 1);
157 vfloat32m1_t temp_vec;
158 for (long i = 1; i < 8; ++i) {
159     temp_vec = tmv_v_f32(f32_sq_sum.m1e0, i);
160     vsum_vec = __riscv_vfadd_vv_f32m1(
        temp_vec, vsum_vec, 32);
161 }
162 #ifdef RVV_UNORDER_REDUCE
163 zero_vec = __riscv_vfredusum_vs_f32m1_f32m1(
        vsum_vec, zero_vec, 32);
164 #else
165 zero_vec = __riscv_vfredosum_vs_f32m1_f32m1(
        vsum_vec, zero_vec, 32);
166 #endif
167 sum = __riscv_vfmv_f_s_f32m1_f32(zero_vec);
168 mean = sum / original_norm_size;
169 if (eps_opt) mean += eps;
170 asm volatile (
171     "fsqrt.s %0, %1"
172     : "=f"(sqrt)
173     : "f"(mean)
174 );
175 sqrt = 1.0 / sqrt;
176 tile_rsqrt = tmv_t_f32(sqrt);
177 twait_store_share(0);
```

这段逐行含义与连续 kernel 第 65-88 行完全一致：

1. 合并两个 FP32 m1 累加器。
2. 每个 128B segment 通过 `tmv.vtr.e32` 送进 RVV。
3. 用 7 次 lane-wise `vfadd` 合并 8 个 segment。
4. 用 `vfredosum.vs` 汇总 32 个 lanes。
5. 除以 `original_norm_size`，可选加入 epsilon。
6. 用 `fsqrt.s` 开方，再取倒数并广播成 tile。
7. 等待 shared-memory store 完成。

strided case 的 wrapper 强制 `original_norm_size == norm_size`，因此测试中均值
分母就是 512；但 kernel 本身仍接收该参数，并按参数执行。

### 7.4 第二遍读写：178-201 行

```cpp
178 for (int64_t i = 0; i < norm_tile_num; ++i) {
179     int64_t offset = i * tile_size;
180     int64_t input_offset = input_batch_offset + offset;
181     int64_t output_offset = output_batch_offset + offset;
```

同时计算输入和输出地址：

- `input_offset` 保留原始输入 stride。
- `output_offset` 使用紧凑输出 stride。

```cpp
182 if (i < shared_buffer_size / 512) {
183     bf16_din = tld_linear_share_m1(
        shared_buff[threadIdx.x], offset);
184 } else {
185     bf16_din = tld_linear_global_m1(
        (bfloat16_t*)pin, input_offset);
186 }
```

前 256 个 tile 从按逻辑位置排列的 shared cache 读取，超出容量才从带 stride
的 global input 读取。对于 `normalized_size=512`，只有一个 tile，直接从 shared
memory 读取。

```cpp
187 f32_din.m2e0 = tcvt_f32(bf16_din, t_r32);
188 f32_out.m1e0 = tmul(
        f32_din.m1e0, tile_rsqrt);
189 f32_out.m1e1 = tmul(
        f32_din.m1e1, tile_rsqrt);
```

再次转为 FP32，并应用 `rstd`。

```cpp
190 if (weight_opt) {
191     bf16_wgt = tld_linear_global_m1(
        (bfloat16_t*)weight, offset);
192     f32_wgt.m2e0 = tcvt_f32(bf16_wgt, t_r32);
193     f32_out.m1e0 = tmul(
        f32_out.m1e0, f32_wgt.m1e0);
194     f32_out.m1e1 = tmul(
        f32_out.m1e1, f32_wgt.m1e1);
195 }
```

按逻辑 normalized offset 读取 contiguous weight，转 FP32 并逐元素相乘。
weight 不带 batch stride。

```cpp
196 bf16_out = tcvt_bf16(
        f32_out.m2e0, t_r32);
197 tst_linear_global_m1(
        bf16_out,
        (bfloat16_t*)pout,
        output_offset);
```

FP32 m2 转回 BF16 m1，然后按 compact output offset 写回。和连续 kernel 的
差别只有最后使用 `output_offset` 而非 `g_offset`。

```cpp
199 }
200 }
201 }
```

依次结束 tile loop、batch row loop 和 kernel。由此可见，strided kernel 并没有
实现通用二维 stride；它只实现“每行有固定间隔、行内连续、输出紧凑”的输入布局。

## 8. fp32 和 fp16 device kernel 的差异

### 8.1 fp32：`rms_norm_kernel_f32.hpp:25-98`

fp32 kernel 的整体结构与 BF16 相同，但不需要 dtype conversion：

- `tfloat32m1_t f32_din` 直接接收 global m1 load。
- `norm_tile_num = norm_size / 256`，因为 fp32 一个 m1 是 256 元素。
- `f32_sq_sum` 是单个 `tfloat32m1_t`，不需要 `.m1e0/.m1e1` 两半。
- 直接 `tmul(f32_din, f32_din)` 和 `tadd(f32_sq_sum, f32_sq)`。
- 第二遍直接乘 `tile_rsqrt`，weight 若开启则直接读取 fp32 m1 并相乘。
- 不需要最后的 `tcvt_bf16`，直接 global store fp32 m1。

但归约仍然把一个 1024B FP32 m1 拆成 8 个 128B segment，使用同样的
`tmv_v_f32`、`vfadd.vv` 和 `vfredosum.vs`。

### 8.2 fp16：`rms_norm_kernel_f16.hpp:25-112`

fp16 kernel 与 BF16 kernel 的 tile 宽度和 FP32 accumulator 组织基本相同：

- 第 26 行设置 `__andescore_fp_mode(FP16)`。
- 一个 fp16 m1 也是 512 元素，因此 `norm_tile_num = norm_size / 512`。
- fp16 m1 通过 `tcvt_f32(..., t_r32)` 转成 fp32 m2。
- 平方和、归约和 rstd 计算使用 fp32。
- 第二遍把 fp16 转 fp32 后缩放和乘 weight。
- 最后用 `tcvt_f16` 转回 fp16，再 global store。

所以三种 dtype 的主要设计是：

```text
fp32:  fp32 load -> fp32 arithmetic -> fp32 store
fp16:  fp16 load -> fp32 arithmetic -> fp16 store
bf16:  bf16 load -> fp32 arithmetic -> bf16 store
```

## 9. 本次运行日志和结果

按用户给出的方式执行：

```bash
cd /softhome/like/package/sikernel
source setup.sh
cd source/source_builtin/attention/rms_norm
bash build.sh
./build/test_host
```

在 2026-09-11 实际执行成功，进程返回码为 `0`。构建环境为：

```text
SIPU SDK  : /share_data/sicx_sdk/release/2609101917
CMODEL    : /share_data/arch_cmodel_release/sipu1.5/2609080400
TARGET    : sipu_150
```

提供的 `/share/users/like/package/sikernel/source/source_builtin/attention/rms_norm/run.log`
显示了以下顺序：

```text
argc:1
kernel to launch!
kernel return!
in golden function !!!
strided kernel to launch, batch_size=33, normalized_size=512
strided kernel return!
in strided golden function !!!
PASS banner
```

其中第一段 `kernel to launch!` 对应连续 `(16,7168)` case，第二段带有
`strided kernel` 文本，对应 `(33,512)` case。当前工作区的 `test_host.cpp`
还保留了用户已有的 debug 输出，因此重新执行时会额外打印 batch、shape、
`argc` 等字段；这些 debug 改动未被本次说明修改。

生成 device 汇编
`source/source_builtin/attention/rms_norm/build/_SipuWork.rms_norm/librms_norm_sipu_fatbin.llvm.asm`
验证了关键 lowering：

```text
tcvt.tt.f32.bf16.r32
tmul.ttt.f32
tadd.ttt.f32
tmv.vtr.e32
vfredosum.vs
twait.i.store.share 0x0
tcvt.tt.bf16.f32.r32
tst.trir.linear.u32.global
```

因此运行日志证明的是 host/device 调用和数值比较通过，汇编则进一步证明了
源码中 SIPU tile intrinsic 到 ISA 指令的对应关系。

## 10. 需要记住的实现边界

1. 当前 kernel 没有 tail 处理，normalized byte size 必须 1024B 对齐。
2. 非连续输入只支持 bf16 的 `(num_tokens,512)` 或 `(num_tokens,1536)`，
   stride 必须是 `(2176,1)`。
3. `original_normalized_size` 是分母参数，不会自动阻止 padding 元素参与分子。
4. `shared_buff[2][...]` 依赖 block 最大 2 threads；若修改 launch block size，
   必须同步修改 shared buffer 设计。
5. `twait_store_share(0)` 只等待 shared-memory store 完成，不是通用 block barrier。
6. RVV 归约依赖 `e32/m1/vl=32` 的向量设置；当前由编译器生成对应的
   `vsetvli/vsetivli`。
7. host 侧在 kernel API 返回后立即执行 D2H `sipuMemcpy`；测试依赖 runtime/cmodel
   对该 copy 的完成语义保证 output 已经可读。

# 从 intrinsic 名字反查功能、LLVM intrinsic 和 ISA

本节回答一个通用问题：当 SDK 的 `siorigin_tile_150g.h` 只有声明、没有注释时，
如何从 `tst_linear_share_m1` 这类名字找到功能、LLVM intrinsic、汇编助记符和
ISA 文档位置。

通用链路是：SDK generated header -> compiler-toolchain TableGen builtin family
-> Clang CodeGen 动态 LLVM intrinsic 名 -> LLVM CodeGen test 的 LLVM IR/汇编
对照 -> LLVM Target TableGen 指令族/encoding -> ISA 文档语义。

不要只按函数名猜。函数名用于生成搜索关键词，最终功能应由 compiler test、
Target TableGen 和 ISA 文档交叉确认。

## 1. 先看 sikernel 的真实调用点

`source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_bf16.hpp:56`
中的 `rms_norm_bf16_kernel` 调用：

`tst_linear_share_m1(bf16_din, shared_buff[threadIdx.x], i * tile_size)`。

调用上下文已经给出第一层语义：把从 global memory 加载的 BF16 tile 写入当前
thread 的 shared-memory scratch buffer。

`source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_f16.hpp:56`
的 `rms_norm_f16_kernel` 和
`source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_f32.hpp:51`
的 `rms_norm_f32_kernel` 也把 tile 写入同一个 shared-memory scratch buffer，
只是 tile dtype 不同。因此这个 intrinsic 的 memory operation 语义与 dtype 无关。

## 2. 先拆函数名

`tst_linear_share_m1` 可以先按命名约定拆成：

| 片段 | 第一层含义 |
| --- | --- |
| `tst` | tile store，把 tile register 写入 memory |
| `linear` | unit-stride/连续访问，区别于 stride/index/block |
| `share` | 目标 memory space 是 share memory |
| `m1` | 操作 1 个完整 tile，SIPU 中完整 tile 为 1024B |

所以第一层结论是：它表示 tile register 到 share memory 的 unit-stride store，
操作一个 m1 tile。

但 `trr` operand 形式、offset 是 immediate 还是 GPR、mask/remote 属性等，不能
只从公开函数名确定，必须继续查 compiler source。

## 3. SDK 生成头中的声明

当前 `latest` SDK 在 2026-09-12 解析为：

`/share_data/sicx_sdk/release/2609111951`。

声明位于：

`/share_data/sicx_sdk/release/latest/bin/nds64le-elf-newlib-v5d/lib/clang/20/include/siorigin_tile_150g.h:15526-15546`

BF16 overload 是：

`TILE_HEADER(tst_trr_linear_share_m1)`，其函数签名为
`void tst_linear_share_m1(tbfloat16m1_t src, bfloat16_t *baseAddr, unsigned long offset, uint32_t attr = 0)`。

这里最重要的是 `TILE_HEADER` 中的 compiler family name：
`tst_trr_linear_share_m1`。它是后续搜索 compiler-toolchain 的关键词。

## 4. compiler-toolchain 的 TableGen 定义

### 4.1 include 入口

`compiler-toolchain/llvm-project/clang/include/clang/Basic/riscv_siorigin_xsotile.td:73`
include 了 `riscv_siorigin_xsotile_memory.td`。

### 4.2 `tst_trr_linear` family

`compiler-toolchain/llvm-project/clang/include/clang/Basic/riscv_siorigin_xsotile_memory.td:258-285`
定义了 `TStoreLinearBuiltin_m`。

这个 multiclass 的关键配置是：

| 配置 | 作用 |
| --- | --- |
| `prototype = "0sPeu"` | store 无返回 tile result；第一个主要 operand 是 tile；有 pointer/offset/attr 类型 |
| `overloaded_name = "tst_linear"` | 公开函数名前缀 |
| `MemoryOpKind::LINEAR` | 进入 unit-stride memory lowering |
| `param_names = ["src", "baseAddr", "offset"]` | source tile、base pointer、offset 的参数名 |

### 4.3 为什么生成 `tst_linear_share_m1`

SIPU150 的本地配置在：

`compiler-toolchain/llvm-project/clang/include/clang/Basic/riscv_siorigin_xsotile_memory.td:326-358`。

其中 `tst_trr_linear` 使用 `MemBasicStoreNoTmConfigList`。每个 configuration
有一个 `IntrinsicStr`，例如 `linear_global_m1`、`linear_share_m1`、
`linear_global_m2`、`linear_share_m2`。

TableGen 通过 `overloaded_name # Conf.IntrinsicStr` 生成公开函数名，因此：

`tst_trr_linear + linear_share_m1 -> tst_linear_share_m1`。

这是一条可推广到其它 intrinsic 的规则：先把公开函数名拆成 family 和
configuration suffix，再回到 TableGen 的 multiclass。

## 5. Clang CodeGen 到 LLVM intrinsic

`TStoreLinearBuiltin_m` 的 ManualCodegen 在
`compiler-toolchain/llvm-project/clang/include/clang/Basic/riscv_siorigin_xsotile_memory.td:265-267`，调用 `GetTileMemoryBuiltinExpr(..., MemoryOpKind::LINEAR, false)`。
最后一个 `false` 表示 store，不是 load。

实现位于 `compiler-toolchain/llvm-project/clang/lib/CodeGen/CGBuiltin.cpp:21863-21882` 的 `GetTileMemoryBuiltinExpr`，并转到
`compiler-toolchain/llvm-project/clang/lib/CodeGen/CGBuiltin.cpp:21731-21775` 的 `GetTileLinearBuiltinExpr`。

SIPU150 的 family 重写逻辑位于
`compiler-toolchain/llvm-project/clang/lib/CodeGen/CGBuiltin.cpp:21406-21444` 的 `combineLinearIntrinsicName`：

- immediate + pass-through -> `ttrii_`；
- immediate + no pass-through -> `trii_`；
- GPR offset + pass-through -> `ttrir_`；
- GPR offset + no pass-through -> `trir_`。

当前 `tst_linear_share_m1` 是 store、使用 GPR offset、没有 load pass-through，
所以 `tst_trr_linear_share_m1` 被改写成 `tst_trir_linear_share_m1`。
`GetTileLinearBuiltinExpr:21741-21747` 还会在 SIPU150 下插入一个额外的 32B offset operand。

compiler-toolchain 的 LLVM CodeGen test 给出了准确的 BF16 intrinsic 名：
`compiler-toolchain/llvm-project/llvm/test/CodeGen/RISCV/xsotile150g/tst-linear-gpr.ll:4897-4904`。
该 intrinsic 是：

`llvm.riscv.tst.trir.linear.share.m1.tv512bf16.p0.i64.i64.i32`

各片段可读成：`tst` 是 tile store，`trir` 是 SIPU150 的 tile/GPR offset 形式，
`linear` 是 unit-stride，`share` 是 share memory，`m1` 是一个 tile，
`tv512bf16` 是 512 个 BF16 元素的 tile，后面的类型串是 LLVM 参数类型编码。

## 6. LLVM intrinsic 到汇编

同一个 LLVM test 的 FileCheck 结果在
`compiler-toolchain/llvm-project/llvm/test/CodeGen/RISCV/xsotile150g/tst-linear-gpr.ll:4898-4904`：

`tst.trir.linear.u32.share T0, (a0), 0, a1`

Clang 侧的 autogenerated wrapper test 位于
`compiler-toolchain/llvm-project/clang/test/CodeGen/RISCV/xsotile150g/autogenerated/xsotile_tst_linear_gpr_test.cpp:3078-3085`，其中 `__rvv_tst_trr_linear_share_m1` 的 CHECK 也是
`tst.trir.linear.u32.share`。

因此，查不到 backend 源码细节时，LLVM CodeGen test 是最可靠的现成索引：它同时
展示 LLVM intrinsic declaration、调用参数和最终汇编助记符。

## 7. LLVM Target TableGen 到 ISA 指令

SIPU150 的 memory instruction 定义在：

`compiler-toolchain/llvm-project/llvm/lib/Target/RISCV/RISCVInstrInfoXSOTile150GMemory.td:4-49`

其中 `TMemLinear150G_m:44-49` 明确选择：

- `_GLOBAL -> T_GblMemOp`；
- `_SHARE -> T_ShrMemOp`。

因此函数名里的 `share` 最终对应 instruction encoding 的 share-memory address-space operand。

store family 在：

`compiler-toolchain/llvm-project/llvm/lib/Target/RISCV/RISCVInstrInfoXSOTile150GMemory.td:196-198`

其中 `TST_TRII_LINEAR_U32` 是 immediate offset store，`TST_TRIR_LINEAR_U32` 是 GPR offset store；当前 intrinsic 选择后者。

M1/M2/M4/M8/MF2/MF4/MF8 的展开在：

`compiler-toolchain/llvm-project/llvm/lib/Target/RISCV/RISCVInstrInfoXSOTile150GMemory.td:22-31`

`_M1` 对应 `T_M1Op` 和 `TPRM1` tile register class。

汇编字符串由 `TMemLinear150G:18-21` 中的 `toAsmStr<NAME>` 生成。更底层的 bit
字段由 `compiler-toolchain/llvm-project/llvm/lib/Target/RISCV/RISCVInstrFormatsXSOTile.td:110-138` 的 `RVInstTMemLinear` 定义，包括 tile register、base GPR、offset register、`imm1`、`memuop` 和 memory-space 属性。

## 8. ISA 文档位置和功能

最终 ISA 小节是：

`/softhome/like/asset/code/isa/index.html:1324-1346`，标题为 `Tile Unit-stride Share Memory Store`。

其指令格式是 `tst.trir.linear.u32.share.[m2/m4/m8/mf2/mf4/mf8].[tm] Ts1, (rs1), imm1, rs3`。
M1 在该文档语法中不显式打印。

总的 unit-stride 规则在 `index.html:951-1051`：`tuop=001` 表示 share memory，
`lsuop=00` 表示 unit-stride，`memuop=001000` 表示 32B store，`rs1` 是 base，
`rs3` 是额外 offset。

## 9. 本例完整链路

从 `source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_bf16.hpp:56` 的 `rms_norm_bf16_kernel` 出发：

`tst_linear_share_m1` -> `TILE_HEADER(tst_trr_linear_share_m1)` ->
TableGen 的 `tst_trr_linear` + `linear_share_m1` ->
`tst_trir_linear_share_m1` ->
`llvm.riscv.tst.trir.linear.share.m1.tv512bf16.p0.i64.i64.i32` ->
`tst.trir.linear.u32.share` -> ISA 的 Tile Unit-stride Share Memory Store。

当前 RMSNorm 生成汇编也直接验证了这条链：
`source/source_builtin/attention/rms_norm/build/_SipuWork.rms_norm/librms_norm_sipu_fatbin.llvm.asm:144` 出现 `tst.trir.linear.u32.share`；同一 kernel 后面还出现 `twait.i.store.share`，对应源码
`source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_f32.hpp:79` 的 `rms_norm_f32_kernel::twait_store_share` 调用。

## 5. 从 Clang CodeGen 找 LLVM intrinsic 名

### 5.1 memory operation 进入 CodeGen

`TStoreLinearBuiltin_m` 的 `ManualCodegen` 位于：

`compiler-toolchain/llvm-project/clang/include/clang/Basic/riscv_siorigin_xsotile_memory.td:265-267`。

它调用 `GetTileMemoryBuiltinExpr(..., MemoryOpKind::LINEAR, false)`，最后一个
`false` 表示当前是 store，不是 load。

实现位于：

`compiler-toolchain/llvm-project/clang/lib/CodeGen/CGBuiltin.cpp:21863-21882`
的 `GetTileMemoryBuiltinExpr`。

`MemoryOpKind::LINEAR` 会转到：

`compiler-toolchain/llvm-project/clang/lib/CodeGen/CGBuiltin.cpp:21731-21775`
的 `GetTileLinearBuiltinExpr`。

### 5.2 SIPU150 的 `trr -> trir`

`compiler-toolchain/llvm-project/clang/lib/CodeGen/CGBuiltin.cpp:21406-21444`
的 `combineLinearIntrinsicName` 负责根据架构、offset 形式和 pass-through
形式重写 intrinsic family。

SIPU150 的规则是：

| 条件 | family 前缀 |
| --- | --- |
| immediate + pass-through | `ttrii_` |
| immediate + no pass-through | `trii_` |
| GPR offset + pass-through | `ttrir_` |
| GPR offset + no pass-through | `trir_` |

当前 `tst_linear_share_m1` 是 store、使用 GPR offset、没有 load pass-through，
所以：

`tst_trr_linear_share_m1 -> tst_trir_linear_share_m1`。

`GetTileLinearBuiltinExpr:21741-21747` 还会在 SIPU150 下插入一个 32B offset
operand。因此源码调用的 `src, baseAddr, offset` 会在 LLVM IR 中表现为 tile、
pointer、额外的 `0`、GPR offset、attr。

### 5.3 `tst_linear_share_m1` 的 LLVM intrinsic

compiler-toolchain 自带的 LLVM CodeGen test 已经给出完整映射：

`compiler-toolchain/llvm-project/llvm/test/CodeGen/RISCV/xsotile150g/tst-linear-gpr.ll:4897-4904`

BF16 case 的 LLVM intrinsic 是：

`llvm.riscv.tst.trir.linear.share.m1.tv512bf16.p0.i64.i64.i32`

关键部分含义：

| 片段 | 含义 |
| --- | --- |
| `tst` | tile store |
| `trir` | SIPU150 的 tile/GPR/offset operand 形式 |
| `linear` | unit-stride |
| `share` | share memory |
| `m1` | 一个 tile |
| `tv512bf16` | 512 个 BF16 元素的 tile，即 1024B |
| `p0` | pointer 类型/地址空间编码 |
| `i64.i64.i32` | offset/attribute 等 LLVM 参数类型编码 |

同一个 LLVM test 的 IR declaration 和调用可以直接作为 intrinsic 的准确名称
来源，不需要从 SDK 函数名人工拼接。

# SIPU C Intrinsic 的来源

## 1. 结论

`tld_linear_global_m1`、`tst_linear_global_m1` 不是 sikernel 自己实现的普通函数，
而是 SIPU SDK 配套的 SiOrigin Clang intrinsic wrapper。调用链是：

```text
tld_linear_global_m1(...)
  -> SDK header declaration
  -> TILE_HEADER(tld_trr_linear_global_m1)
  -> __builtin_rvv_tld_trr_linear_global_m1
  -> SiOrigin Clang/SCC backend
  -> tld.trir.linear.u32.global
```

`tst_linear_global_m1` 的链路相同，最终生成
`tst.trir.linear.u32.global`。SDK 头文件提供声明和 builtin alias；真正的
指令选择、operand lowering 和 encoding 在 SIPU 版 Clang/SCC compiler backend。

## 2. 当前构建使用的 SDK

执行：

```bash
cd /softhome/like/package/sikernel
source setup.sh
```

当前实际环境为：

```text
SI_SDK_BIN  = /share_data/sicx_sdk/release/2609101917/bin
SI_SDK_LIB  = /share_data/sicx_sdk/release/2609101917/lib
CMODEL      = /share_data/arch_cmodel_release/sipu1.5/2609080400
SIPU_ARCH   = 150
```

本次 CMake 输出为 `TARGET_SIPU_ARCH=150` 和 `-arch=sipu_150`。

## 3. include 链路

RMSNorm device 文件 `kernel/rms_norm_kernel.su:19-23` 直接包含：

```cpp
#include <riscv_vector.h>
#include <siorigin_tile.h>
#include <sipu_runtime.h>
#include "sipu.h"
```

SDK 的 `SiTe.hpp:27-30` 也包含：

```cpp
#include <siorigin_tile.h>
#include <siorigin_tile_inter.h>
```

对应文件位于：

```text
/share_data/sicx_sdk/release/2609101917/include/SiTe/SiTe.hpp
/share_data/sicx_sdk/release/2609101917/bin/nds64le-elf-newlib-v5d/lib/clang/20/include/siorigin_tile.h
/share_data/sicx_sdk/release/2609101917/bin/nds64le-elf-newlib-v5d/lib/clang/20/include/siorigin_tile_150g.h
/share_data/sicx_sdk/release/2609101917/bin/nds64le-elf-newlib-v5d/lib/clang/20/include/siorigin_tile_inter.h
```

`siorigin_tile.h:19-26` 根据 `__riscv_xsotile150g/160g/170g` 选择架构头；
本次 SIPU 1.5 选择 `siorigin_tile_150g.h`。

## 4. 函数声明位置

在 `siorigin_tile_150g.h` 中：

- `tld_linear_global_m1` 约在 `3882-3902` 行。
- `tst_linear_global_m1` 约在 `14756-14776` 行。

当前 BF16 overload 是：

```cpp
TILE_HEADER(tld_trr_linear_global_m1)
tbfloat16m1_t tld_linear_global_m1(
    const bfloat16_t* baseAddr,
    unsigned long offset,
    uint32_t attr = 0);

TILE_HEADER(tst_trr_linear_global_m1)
void tst_linear_global_m1(
    tbfloat16m1_t src,
    bfloat16_t* baseAddr,
    unsigned long offset,
    uint32_t attr = 0);
```

同一文件还提供 int32、int8、fp16、fp32、FP8 等 dtype overload。

`siorigin_tile_inter.h` 也有同名声明：

- `tld_trr_linear_global_m1` 约在 `2318-2358` 行。
- `tst_trr_linear_global_m1` 约在 `10390-10430` 行。

它们使用 `sifmt::bfloat16` 等 SiTe 类型，是 SDK 的兼容/交互 API 声明集合。

## 5. `TILE_HEADER` 如何绑定到 compiler builtin

SDK 的 `siorigin_tile.h:15-18` 定义：

```cpp
#define __rvv_generic \
    static inline __device__ \
    __attribute__((__always_inline__, __nodebug__))

#define TILE_HEADER(T) \
    __rvv_generic \
    __attribute__((clang_builtin_alias(__builtin_rvv_##T)))
```

所以：

```cpp
TILE_HEADER(tld_trr_linear_global_m1)
tbfloat16m1_t tld_linear_global_m1(...);
```

概念上会绑定到：

```cpp
__builtin_rvv_tld_trr_linear_global_m1(...)
```

这解释了为什么 SDK header 中只有声明没有函数体：调用在编译阶段被识别为
compiler builtin，不会生成对某个 runtime C 函数的普通调用。

`trr` 是 SDK/compiler 对 operand pattern 的内部命名；源码公开使用的名字仍然
是 `tld_linear_global_m1`。带 `passThru` 的另一组重载使用
`tld_ttrr_linear_global_m1`，不要与当前三参数版本混淆。

## 6. LLVM intrinsic 注册

SDK 还提供：

```text
/share_data/sicx_sdk/release/2609101917/bin/nds64le-elf-newlib-v5d/include/llvm/IR/IntrinsicsRISCV.h
```

其中登记了：

```text
riscv_tld_trr_linear_global_m1  // llvm.riscv.tld.trr.linear.global.m1
riscv_tst_trr_linear_global_m1  // llvm.riscv.tst.trr.linear.global.m1
riscv_tld_tri_linear_global_m1  // llvm.riscv.tld.tri.linear.global.m1
riscv_tst_tri_linear_global_m1  // llvm.riscv.tst.tri.linear.global.m1
```

这个文件主要是 LLVM intrinsic ID 注册，不是最终指令实现。真正的 lowering
位于随 SDK 发布的 SIPU 定制 Clang/SCC backend 中。

## 7. 为什么不是 runtime library 函数

动态库中没有普通的 `tld_`/`tst_` 符号：

```bash
nm -D source/source_builtin/attention/rms_norm/build/librms_norm.so \
  | rg 'tld_|tst_|__builtin'
```

相反，生成的 device 汇编直接出现：

```text
tld.trir.linear.u32.global T4, (a1), 0x0, s11
tst.trir.linear.u32.global T3, (a0), 0x0, s10
```

本次 RMSNorm 的实际对应关系是：

```text
tld_linear_global_m1(...)
  -> TILE_HEADER(tld_trr_linear_global_m1)
  -> __builtin_rvv_tld_trr_linear_global_m1
  -> llvm.riscv.tld.trr.linear.global.m1
  -> tld.trir.linear.u32.global

tst_linear_global_m1(...)
  -> TILE_HEADER(tst_trr_linear_global_m1)
  -> __builtin_rvv_tst_trr_linear_global_m1
  -> llvm.riscv.tst.trr.linear.global.m1
  -> tst.trir.linear.u32.global
```

## 8. 和 ISA 文档的关系

`/softhome/like/asset/code/isa/index.html` 描述最终 ISA 的语义、operand 和
encoding，例如：

```text
tld.trir.linear.u32.global.[m2/m4/m8/mf2/mf4/mf8] ...
tst.trir.linear.u32.global.[m2/m4/m8/mf2/mf4/mf8] ...
```

它不提供可 include 的 C 函数，也不实现 intrinsic。各层职责如下：

| 层 | 位置 | 职责 |
| --- | --- | --- |
| ISA 规范 | `asset/code/isa/index.html` | 指令语义、operand、encoding |
| C/C++ 声明 | `siorigin_tile_150g.h`、`siorigin_tile_inter.h` | overload API |
| builtin alias | `siorigin_tile.h` | `TILE_HEADER` -> `__builtin_rvv_*` |
| LLVM ID | `IntrinsicsRISCV.h` | `llvm.riscv.tld/tst...` 注册 |
| compiler backend | SIPU Clang/SCC | builtin/LLVM IR -> SIPU 指令 |
| runtime | `libsipu.so`、`libsipurt.so` | kernel 装载、launch、copy、执行 |

## 9. 本次运行验证

按指定流程重新执行：

```bash
cd /softhome/like/package/sikernel
source setup.sh
cd source/source_builtin/attention/rms_norm
bash build.sh
./build/test_host
```

结果：

- `TARGET_SIPU_ARCH=150`。
- build 成功。
- `./build/test_host` 返回码为 `0`。
- 连续 `(16,7168)` 和 strided `(33,512)` case 完成。
- 生成汇编包含 `tld.trir.linear.u32.global` 和
  `tst.trir.linear.u32.global`。

构建过程中的 `Clock skew detected` 是共享文件系统时间戳告警，不影响编译和
测试结果。

以后查这类 intrinsic，最直接的命令是：

```bash
rg -n 'tld_linear_global_m1|tst_linear_global_m1' \
  "$SI_SDK_BIN/nds64le-elf-newlib-v5d/lib/clang/20/include"

rg -n -C 3 'define TILE_HEADER|clang_builtin_alias|__builtin_rvv_' \
  "$SI_SDK_BIN/nds64le-elf-newlib-v5d/lib/clang/20/include"

rg -n 'tld_trr_linear_global_m1|tst_trr_linear_global_m1' \
  "$SI_SDK_BIN/nds64le-elf-newlib-v5d/include/llvm/IR/IntrinsicsRISCV.h"
```

一句话总结：**这些 C intrinsic 的声明在 SIPU SDK 的 Clang include 头文件中，
通过 `TILE_HEADER` 绑定到 SIPU Clang builtin；真正实现和编码在 SIPU 版
Clang/SCC backend，最终生成 `tld.trir`/`tst.trir` ISA 指令。**

## 10. 通用反查命令清单

第一步，找 SDK declaration 和 `TILE_HEADER` family：

`rg -n -C 3 '目标函数名|TILE_HEADER\\(' "$SI_SDK_BIN/nds64le-elf-newlib-v5d/lib/clang/20/include"`

第二步，按 family 查 compiler TableGen 和 CodeGen：

`rg -n '目标 family|TStoreLinearBuiltin_m|MemoryOpKind::LINEAR' /share/users/like/package/compiler-toolchain/llvm-project/clang/include/clang/Basic /share/users/like/package/compiler-toolchain/llvm-project/clang/lib/CodeGen`

第三步，查动态 LLVM intrinsic 名字拼接：

`rg -n 'combineLinearIntrinsicName|GetTileLinearBuiltinExpr|GetTileMemoryBuiltinExpr' /share/users/like/package/compiler-toolchain/llvm-project/clang/lib/CodeGen/CGBuiltin.cpp`

第四步，优先查 compiler 自带 CodeGen test：

`rg -n -C 3 '目标 family|CHECK:.*目标 mnemonic' /share/users/like/package/compiler-toolchain/llvm-project/llvm/test/CodeGen/RISCV/xsotile150g /share/users/like/package/compiler-toolchain/llvm-project/clang/test/CodeGen/RISCV/xsotile150g`

第五步，查 Target TableGen 的 instruction family、memory space 和 tile size：

`rg -n -C 5 'TST_TRIR_LINEAR_U32|TMemLinear150G|T_ShrMemOp|toAsmStr' /share/users/like/package/compiler-toolchain/llvm-project/llvm/lib/Target/RISCV`

第六步，最后用最终助记符查 ISA：

`rg -n -C 4 '最终助记符|ISA 小节标题' /softhome/like/asset/code/isa/index.html`

对本例，查找结果可以压缩成一条链：

`tst_linear_share_m1` -> `tst_trr_linear_share_m1` -> `tst_trir_linear_share_m1` -> `llvm.riscv.tst.trir.linear.share.m1.tv512bf16.p0.i64.i64.i32` -> `tst.trir.linear.u32.share` -> ISA `Tile Unit-stride Share Memory Store`。

这套流程也适用于 `tld_*`、`tst_*`、`tcvt_*`、`tmv_*`、`tmul_*` 和 `tadd_*`：
先找 TableGen family，再看 CodeGen 的 IR name 组合，最后用 LLVM test 和 ISA
小节确认功能。
