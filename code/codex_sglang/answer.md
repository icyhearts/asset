对于 `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` 里的 `extend_attention_fwd`，它的 launch grid 是：

```python
grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
```

对应到 kernel 内部就是：

```python
cur_seq = tl.program_id(0)
cur_head = tl.program_id(1)
cur_block_m = tl.program_id(2)
```

也就是说，一个 Triton program 的语义是：

- 固定一个 batch 内的序列 `cur_seq`
- 固定一个 query head `cur_head`
- 固定这个序列的 extend query 上的第 `cur_block_m` 个 M 方向 tile

所以总 program 数量是：

```text
batch_size * head_num * ceil(max_len_extend / BLOCK_M)
```

## 1. grid 的设计思路

这个 kernel 只在 **extend/prefill 新增出来的 query token** 上并行，不在 prefix 长度上开 grid。

- 第 0 维 `batch_size`：按请求/序列并行
- 第 1 维 `head_num`：按 query head 并行
- 第 2 维 `ceil(max_len_extend / BLOCK_M)`：按 extend query 的长度分块并行

注意这里没有按 K/V 长度开 grid。  
对于每个 `(seq, head, block_m)` program，kernel 会自己在内部用：

```python
for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
...
for start_n in range(0, cur_block_m_end, BLOCK_N):
```

去扫描 prefix 和 extend 侧的 K/V 块。所以并行切的是 **Q 的 M 维**，而 **K/V 的 N 维**是在单个 program 内循环完成的。

## 2. 每个 program 处理多少数据

一个 program 处理的是一个 attention tile，核心是：

- Query 方向：最多 `BLOCK_M` 个 query token
- Head 方向：1 个 query head
- Head dim 方向：`Lq` / `Lv`（内部会 pad 到 `BLOCK_DMODEL` / `BLOCK_DV`）
- K/V 方向：每次内部循环处理 `BLOCK_N` 个 key/value token

更具体地说，这个 program 会：

1. 读取当前 `(seq, head)` 下，extend 段里一块 query：

```text
q rows = [cur_block_m * BLOCK_M, (cur_block_m + 1) * BLOCK_M)
```

实际有效行数由：

```python
mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend
```

控制，所以最后一个 block 可能不足 `BLOCK_M` 行。

2. 对这块 query，先和 prefix KV 做 attention：

- 每次拿 `BLOCK_N` 个 prefix KV
- 形成一个 `[BLOCK_M, BLOCK_N]` 的 score tile
- 累积到 `acc`，其中

```python
acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
```

3. 再对 extend 段自身做 attention：

- 也是每次 `BLOCK_N` 个 KV
- causal 情况下只扫描到当前 block 的右边界：

```python
cur_block_m_end = min(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
```

- non-causal 则扫描整个 extend 段

4. 最后写回当前这 `BLOCK_M` 行 query 的输出：

```text
output tile shape ~= [BLOCK_M, Lv]
```

所以可以把一个 program 近似理解成：

- 负责 1 个 `(seq, q_head)`
- 负责最多 `BLOCK_M` 个 query token
- 对所有相关 KV 做归约
- 每次按 `BLOCK_N` 个 KV tile 迭代

## 3. `max_len_extend` 是什么

`max_len_extend` 不是总序列长度，也不是 prefix 长度，而是 **当前 batch 里所有样本的 extend 长度最大值**。

它在 `python/sglang/srt/layers/attention/triton_backend.py` 里来自：

```python
qo_indptr[1 : bs + 1] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
max_extend_len = max(forward_batch.extend_seq_lens_cpu)
```

这里的 `forward_batch.extend_seq_lens` 就是每个请求这次 extend/prefill 新增了多少 token。  
因此：

- `qo_indptr` 描述的是 extend query 的分段边界
- `max_len_extend` 只是用来决定 grid 第 3 维要开多少个 M-block

这也意味着：

- 如果 batch 内不同样本的 extend 长度不一样
- 那么较短样本对应的后面一些 `cur_block_m` program 会被 launch 出来
- 但它们会因为 `mask_m` 全 false 而基本不做有效计算

这是用 batch 内最大长度统一 launch 的典型做法。

## 4. `BLOCK_M` 是什么

`BLOCK_M` 表示 **一个 program 在 Q 的 M 维上一次处理多少个 query token**。  
这里的 M 可以理解成 attention score 矩阵的“行数”。

对应代码里：

```python
offs_m = tl.arange(0, BLOCK_M)
mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend
```

所以：

- `cur_block_m = 0` 处理第 `0 ~ BLOCK_M-1` 个 extend token
- `cur_block_m = 1` 处理第 `BLOCK_M ~ 2*BLOCK_M-1` 个 extend token
- 以此类推

## 5. 对 Llama 3.1 8B Instruct 这一路径的具体值

模型配置：

- `hidden_size = 4096`
- `num_attention_heads = 32`
- `num_key_value_heads = 8`

所以对 Llama 而言，`head_dim = hidden_size / num_attention_heads = 4096 / 32 = 128`。  
SGLang 的 `LlamaAttention` 也是这样取的：

```python
self.head_dim = getattr(config, "head_dim", self.hidden_size // self.total_num_heads)
```

因此这个模型在该 kernel 里：

- `Lq = 128`
- `Lv = 128`
- `BLOCK_DMODEL = 128`
- `BLOCK_DPE = 0`
- `BLOCK_DV = 128`

如果你当前环境是 H100，那么 `CUDA_CAPABILITY[0] >= 9`，而且 `Lq <= 256`，因此 `_get_block_sizes_for_extend_attention()` 会选：

```python
BLOCK_M = 128
BLOCK_N = 64
num_warps = 8
```

也就是说，对 `llama3.1-8B-Instruct + H100 + triton attention backend`：

- 一个 program 处理 1 个序列、1 个 query head、最多 128 个 extend query token
- 输出是这 128 个 query token 在该 head 上的 attention 输出
- K/V 侧不是一次全吃完，而是每轮按 64 个 token 的 tile 去扫

## 6. 一句话总结

这个 grid 的本质是：

- `batch_size` 维切序列
- `head_num` 维切 head
- `ceil(max_len_extend / BLOCK_M)` 维切 extend query token block

而每个 Triton program 本质上是在做：

```text
一个(seq, head, query_block) -> 处理最多 BLOCK_M 行Q，
对所有相关KV按 BLOCK_N 分块归约，最后产出这块Q的输出
```
