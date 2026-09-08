## 1. Llama-3.1-8B-Instruct 裁剪模型（4-layer tiny model）相比原始模型做了哪些修改

**路径**:
- 原始: `/data_gpu/modelzoo/weights/llm/llama/llama3.1/llama3.1-8B-Instruct/safetensor_weights`（15G, 4 个 safetensors 分片）
- 裁剪后: `/share_data/inference-framework/tiny-models/Llama-3.1-8B-Instruct-4layer/safetensor_weights`（3.7G, 2 个 safetensors 分片）

**裁剪脚本**: `/share_data/inference-framework/tiny-models/trim/trim_llama31_8b.py`（说明文档见裁剪目录下的 `TRIM_INFO.md`）

### 修改内容（唯一实质修改：层数 32 → 4）

1. **`config.json`**: 仅 `num_hidden_layers: 32 → 4`，其余字段（hidden_size=4096、intermediate_size=14336、num_attention_heads=32、num_key_value_heads=8、rope_scaling、vocab_size 等）全部不变。

2. **权重张量**: 291 个 key → 39 个 key。层映射为 `{0:0, 1:1, 2:2, 3:3}`，即**保留前 4 层（layer 0-3）原封不动**，丢弃 layer 4-31。保留下来的张量 dtype/shape 与原始完全一致（逐 key 对比验证通过，无任何 dtype/shape 差异）：
   - `model.embed_tokens.weight`、`model.norm.weight`、`lm_head.weight` 全部原样保留；
   - `model.layers.{0..3}.*`（self_attn q/k/v/o proj、mlp gate/up/down proj、input/post_attention layernorm）原样保留；
   - 权重中无量化 scale（`weight_scale_inv` 后缀的张量不存在，两个模型都是纯 bf16）。

3. **tokenizer / generation 配置**: `tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`、`generation_config.json` 未修改。

即：这是一个纯粹的**深度截断**（depth truncation）冒烟测试模型——取前 4 层 + 完整 embedding/lm_head/final_norm，参数量约 8B → 1.9B 左右，用于快速验证推理流程，输出质量不做保证。

---

## 2. `layer_map={0: 0, 1: 1, 2: 2, 3: 3}` 参数的功能

**一句话**：`layer_map` 是「**目标层号 → 源层号**」的映射（`dst_idx -> src_idx`，见 `smoke_trim.py` 的 `SmokeSpec.layer_map` 字段注释），它同时决定两件事：

1. **保留哪些源层**（筛选）：源层号只要没出现在 `layer_map.values()` 里，该层的所有权重张量都会被丢弃（`should_keep()`）。
2. **把保留的层放到裁剪后模型的第几层**（重编号）：源键名 `model.layers.{src}.` 会被改写成 `model.layers.{dst}.`（`remap_layer_key()`）。

### 对 Llama-3.1-8B 的实际效果（32 层 → 4 层，纯深度截断）

```
Src: llama3.1-8B (32层)                     Dst: Llama-3.1-8B-Instruct-4layer (4层)
─────────────────────────────────           ─────────────────────────────────
  layer 0  ──(keep, dst←0)─────────────►   layer 0
  layer 1  ──(keep, dst←1)─────────────►   layer 1
  layer 2  ──(keep, dst←2)─────────────►   layer 2
  layer 3  ──(keep, dst←3)─────────────►   layer 3
  layer 4  ──✗ drop ──────────────────┐
  layer 5  ──✗ drop                   │   (只有这 4 层被保留，
    ...                                 │    其余 28 层全部丢弃)
  layer 31 ──✗ drop ──────────────────┘
```

- 因 `layer_map.values() = {0,1,2,3}`，源层 4~31 的张量全部被 `should_keep()` 过滤掉。
- 因每个 `dst_i == src_i`（恒等映射），保留层的键名里 `layers.{src}.` 重编号后仍是 `layers.{dst}.`，名字不变。
- 结果就是「保留前 4 层 + 完整 embedding / final_norm / lm_head」，`config.json` 同步 `num_hidden_layers: 32 → 4`。

### 一般情形（非恒等映射）

`layer_map` 的**键**是目标层号，**值**是来源层号；所以它不只是「截前 N 层」，还可以**跳过 / 重排 / 跳转**。仓库里大量脚本正是这么用的：

```
layer_map = {dst: src}                       语义
─────────────────────────────────────        ───────────────────────────────
  {0:0, 1:1, 2:2, 3:3}   llama31_8b          顺序保留前4层（恒等）
  {0:0, 1:3}               deepseek_v3/glm47 保源层0和3 → 目标层0和1（跳1层）
  {0:0, 1:2, 2:3}         deepseek_v4_flash  保源层0,2,3 → 目标0,1,2（跳过源层1）
  {0:0, 1:1, 2:6, 3:7}    ling26_flash       保源层0,1,6,7 → 目标0,1,2,3
```

以 `{0:0, 1:3}`（DeepSeek-V3 / GLM-4.7 等）画图：

```
# layer_map = {0:0, 1:3}   →  只保留源层0和源层3
Src (深, 示意标出相关层)                Dst (2层)
layer 0 ──(dst←0)────────────────►  layer 0
layer 1 ──✗ drop
layer 2 ──✗ drop                    (注：dst←3 表示把源层3放到第1个位置)
layer 3 ──(dst←1)────────────────►  layer 1
```

### 为什么需要非恒等映射

- **跳过不需要的层**：例如把 dense 层与 MoE/混合层交错时，只挑「类型合适」的源层放到对应位置。`smoke_trim.py` 中 `layer_group_size` / `layer_types` 相关注释强调：混合模型里某目标槽位需要 linear/full-attn 的特定权重布局，`layer_map` 选取的源层类型必须与之匹配。
- **跨边界重编号**：像 DeepSeek 这类模型 `first_k_dense_replace` 之后是 MoE 层，取子集时层号会错位，用 `layer_map` 把稀疏的源层号对齐回连续的 `0..dst_layers-1`（`build_layer_map()` 就是为此生成此类映射）。
- 本质上 `layer_map` 是一个「**裁剪器 + 重排器**」：决定砍谁、留谁、留给它哪个槽位。
