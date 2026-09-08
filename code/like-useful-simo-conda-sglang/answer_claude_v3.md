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
