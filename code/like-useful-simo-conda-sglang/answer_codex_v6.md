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
