# `hc_split_sinkhorn_kernel` 详细讲解

源码位置：`/data/like/hf-models/deepseek-v4-flash-git-control-by-like/inference/kernel.py`

相关函数：

- `hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float)`：TileLang JIT kernel 生成器。
- `hc_split_sinkhorn(...)`：PyTorch 包装函数，分配输出张量并调用 kernel。
- `Block.hc_pre(...)` / `Block.hc_post(...)`：模型里实际使用这三个输出的地方。

## 1. 这个 kernel 解决什么问题

DeepSeek V4 Flash 这份推理代码里，Transformer block 使用了 Hyper-Connections，简称 HC。普通 residual 只有一份 hidden state，而 HC 维护 `hc_mult` 份 hidden state copy，例如配置里：

```json
"hc_mult": 4,
"hc_sinkhorn_iters": 20
```

所以每个 token 的隐藏状态形状大致是：

```text
x: [batch, seq, hc, dim]
```

在进入 attention 或 FFN 前，`hc_pre` 需要把 `hc` 份 hidden state 合成 1 份：

```python
y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
```

在 attention 或 FFN 计算后，`hc_post` 又需要把 1 份输出扩展回 `hc` 份，并和之前的 `hc` 份 residual 做混合：

```python
y = post.unsqueeze(-1) * x.unsqueeze(-2) \
  + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
```

`hc_split_sinkhorn_kernel` 的作用就是：对每个 token，根据一个线性层算出来的 `mixes`，一次性生成三组 HC 系数：

```text
pre : [batch, seq, hc]
post: [batch, seq, hc]
comb: [batch, seq, hc, hc]
```

其中：

- `pre`：进入子层前，把 `hc` 份 hidden state 加权合成 1 份。
- `post`：子层输出回写到每个 HC copy 时的权重。
- `comb`：旧 residual 的 `hc` 份之间如何互相混合，经过 Sinkhorn 归一化后接近双随机矩阵。

## 2. 外层包装函数和输入输出形状

包装函数：

```python
def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    b, s, _ = mixes.size()
    pre = mixes.new_empty(b, s, hc_mult)
    post = mixes.new_empty(b, s, hc_mult)
    comb = mixes.new_empty(b, s, hc_mult, hc_mult)
    kernel = hc_split_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
    kernel(
        mixes.view(-1, (2 + hc_mult) * hc_mult),
        hc_scale,
        hc_base,
        pre.view(-1, hc_mult),
        post.view(-1, hc_mult),
        comb.view(-1, hc_mult, hc_mult),
    )
    return pre, post, comb
```

这里把 `[b, s, ...]` flatten 成 `[n, ...]`，其中：

```text
n = b * s
hc = hc_mult
mix_hc = (2 + hc) * hc = 2 * hc + hc * hc
```

当 `hc = 4` 时：

```text
mix_hc = (2 + 4) * 4 = 24
```

也就是说，每个 token 的 `mixes` 有 24 个 FP32 值，被切成三段：

```text
mixes[0 : hc]              -> pre  的原始 logits，长度 hc
mixes[hc : 2 * hc]         -> post 的原始 logits，长度 hc
mixes[2 * hc : 2 * hc+hchc] -> comb 的原始 logits，形状 hc x hc
```

`hc_scale` 的形状是 `[3]`，三段分别使用不同 scale：

```text
hc_scale[0] 用于 pre
hc_scale[1] 用于 post
hc_scale[2] 用于 comb
```

`hc_base` 的形状是 `[mix_hc]`，给每个输出通道提供单独 bias。

## 3. TileLang kernel 结构

kernel 生成器：

```python
@tilelang.jit(pass_configs=pass_configs)
def hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
    n = T.symbolic("n")
    mix_hc = (2 + hc) * hc
    threads = 64
```

重点：

- `@tilelang.jit` 表示这个 Python 函数会生成 TileLang CUDA kernel。
- `hc`、`sinkhorn_iters`、`eps` 是编译期参数，会参与 kernel specialization。
- `n = T.symbolic("n")` 是运行期符号维度，对应 flatten 后的 token 数。
- `threads = 64` 表示每个 token 对应一个 TileLang kernel instance，由 64 个线程协作处理。

内部 prim func：

```python
@T.prim_func
def hc_split_sinkhorn_kernel_(
    mixes: T.Tensor[(n, mix_hc), FP32],
    hc_scale: T.Tensor[(3,), FP32],
    hc_base: T.Tensor[(mix_hc,), FP32],
    pre: T.Tensor[(n, hc), FP32],
    post: T.Tensor[(n, hc), FP32],
    comb: T.Tensor[(n, hc, hc), FP32],
):
```

输入输出全部是 FP32。这里没有直接处理 `batch` 和 `seq`，而是处理展平后的第 `i` 个 token。

```python
with T.Kernel(n, threads=threads) as i:
```

这表示 grid 维度是一维 `n`，每个 program 处理一个 token：

```text
i = 0, 1, ..., n-1
```

## 4. 临时存储：shared memory 和 fragment

```python
mixes_shared = T.alloc_shared(mix_hc, FP32)
comb_frag = T.alloc_fragment((hc, hc), FP32)
T.copy(mixes[i, :], mixes_shared)
```

含义：

- `mixes_shared`：把当前 token 的 `mix_hc` 个值搬到 shared memory。
- `comb_frag`：线程本地/register fragment，用来保存 `hc x hc` 的组合矩阵。

因为默认 `hc = 4`，所以：

```text
mixes_shared: 24 个 FP32
comb_frag: 4 x 4 = 16 个 FP32
```

数据量很小，适合让一个小 kernel instance 单独处理一个 token，避免启动多个 PyTorch op 做 sigmoid、softmax、sum、div 等造成额外开销。

## 5. 生成 `pre`

代码：

```python
for j in T.Parallel(hc):
    pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
```

数学等价：

```text
pre_j = sigmoid(mix_j * scale_pre + base_j) + eps
```

其中：

```text
j = 0 .. hc-1
scale_pre = hc_scale[0]
base_j = hc_base[j]
```

特点：

- `sigmoid(...)` 把值限制在 `(0, 1)`。
- `+ eps` 保证 `pre` 严格为正，避免后续权重完全为 0。
- `pre` 没有做归一化，所以它不是 softmax 权重；它是 learned gate。

在 `Block.hc_pre` 中使用：

```python
y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
```

也就是：

```text
y = sum_j pre_j * x_j
```

## 6. 生成 `post`

代码：

```python
for j in T.Parallel(hc):
    post[i, j] = 2 * T.sigmoid(
        mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc]
    )
```

数学等价：

```text
post_j = 2 * sigmoid(mix_{hc+j} * scale_post + base_{hc+j})
```

特点：

- `post` 范围是 `(0, 2)`。
- 和 `pre` 不同，`post` 没有 `+ eps`。
- 乘以 2 说明它允许子层输出对某些 HC copy 的注入强度超过 1。

在 `Block.hc_post` 中使用：

```python
y = post.unsqueeze(-1) * x.unsqueeze(-2) + ...
```

也就是每个 HC copy 都接收一份子层输出 `x`，但权重不同：

```text
y_j += post_j * sublayer_output
```

## 7. 生成 `comb` 原始矩阵

代码：

```python
for j, k in T.Parallel(hc, hc):
    comb_frag[j, k] = (
        mixes_shared[j * hc + k + hc * 2] * hc_scale[2]
        + hc_base[j * hc + k + hc * 2]
    )
```

数学等价：

```text
comb_logit[j, k] = mix_{2hc + jhc + k} * scale_comb
                 + base_{2hc + jhc + k}
```

这一步只是生成 `hc x hc` 的 logits，还不是最终权重矩阵。后面会对它做 softmax 和 Sinkhorn normalization。

## 8. 第一步：对 `comb` 做 row-wise softmax

源码注释：

```python
# comb = comb.softmax(-1) + eps
```

代码分三步。

先算每一行最大值：

```python
row_max = T.alloc_fragment(hc, FP32)
T.reduce_max(comb_frag, row_max, dim=1)
```

`dim=1` 表示沿着列方向 reduce，所以得到每一行的最大值：

```text
row_max[j] = max_k comb_logit[j, k]
```

再做指数，并减去行最大值：

```python
for j, k in T.Parallel(hc, hc):
    comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
```

这是标准 softmax 数值稳定技巧，避免 `exp(logit)` 溢出。

然后求每一行的指数和：

```python
T.reduce_sum(comb_frag, row_sum, dim=1)
```

最后归一化并加 `eps`：

```python
for j, k in T.Parallel(hc, hc):
    comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps
```

数学等价：

```text
comb[j, k] = softmax(comb_logit[j, :])[k] + eps
```

此时每一行的和大约是：

```text
sum_k comb[j, k] = 1 + hc * eps
```

加 `eps` 的目的不是让行和等于 1，而是确保每个元素严格大于 0。Sinkhorn 迭代要求矩阵元素非负，严格正数也更稳定。

## 9. 第二步：先做一次 column normalization

源码注释：

```python
# comb = comb / (comb.sum(-2) + eps)
```

代码：

```python
T.reduce_sum(comb_frag, col_sum, dim=0)
for j, k in T.Parallel(hc, hc):
    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)
```

`dim=0` 表示沿着行方向 reduce，所以得到每一列的和：

```text
col_sum[k] = sum_j comb[j, k]
```

然后每列除以自己的列和：

```text
comb[j, k] = comb[j, k] / (col_sum[k] + eps)
```

这一步之后，列和接近 1。

## 10. Sinkhorn 迭代主体

代码：

```python
for _ in T.serial(sinkhorn_iters - 1):
    # comb = comb / (comb.sum(-1) + eps)
    T.reduce_sum(comb_frag, row_sum, dim=1)
    for j, k in T.Parallel(hc, hc):
        comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)

    # comb = comb / (comb.sum(-2) + eps)
    T.reduce_sum(comb_frag, col_sum, dim=0)
    for j, k in T.Parallel(hc, hc):
        comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)
```

Sinkhorn normalization 的目标是把一个正矩阵变成近似双随机矩阵：

```text
每一行 sum_k comb[j, k] ≈ 1
每一列 sum_j comb[j, k] ≈ 1
comb[j, k] > 0
```

流程是反复交替做：

```text
1. row normalize
2. column normalize
```

因为前面已经做过：

```text
row softmax + column normalize
```

所以循环里只需要再做 `sinkhorn_iters - 1` 次。如果 `sinkhorn_iters = 20`，总共相当于：

```text
1 次 row softmax
20 次 column normalization
19 次 row normalization
```

最终矩阵的最后一步是 column normalization，因此列和通常更接近 1；行和也会随着迭代接近 1。`eps` 会让结果不是数学上严格的双随机矩阵，但能提升数值稳定性，避免除零或极小值放大。

## 11. 写回输出

代码：

```python
T.copy(comb_frag, comb[i, :, :])
```

最终把当前 token 的 `hc x hc` 矩阵写回：

```text
comb[i, :, :]
```

包装函数再把它 view 回：

```text
[batch, seq, hc, hc]
```

## 12. 和 `Block.hc_pre` / `Block.hc_post` 的完整关系

`Block.hc_pre` 里：

```python
x = x.flatten(2).float()
rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
mixes = F.linear(x, hc_fn) * rsqrt
pre, post, comb = hc_split_sinkhorn(...)
y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
return y.to(dtype), post, comb
```

这里 `x.flatten(2)` 把：

```text
[b, s, hc, d]
```

变成：

```text
[b, s, hc * d]
```

然后线性层 `hc_fn` 生成：

```text
mixes: [b, s, (2 + hc) * hc]
```

`hc_split_sinkhorn_kernel` 把 `mixes` 拆成：

```text
pre : [b, s, hc]
post: [b, s, hc]
comb: [b, s, hc, hc]
```

`pre` 用来把多份 HC state 合成单份 state，送入 attention 或 FFN。

`Block.hc_post` 里：

```python
y = post.unsqueeze(-1) * x.unsqueeze(-2) \
  + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
```

按索引展开，输出第 `j` 个 HC copy：

```text
y_j = post_j * sublayer_output + sum_k comb[k, j] * residual_k
```

注意 PyTorch 广播里 `residual.unsqueeze(-2)` 的形状是：

```text
[b, s, 1, hc, d]
```

`comb.unsqueeze(-1)` 的形状是：

```text
[b, s, hc, hc, 1]
```

然后 `dim=2` 求和，所以这里可以理解为把第一个 HC 维度作为来源维度聚合掉，得到目标 HC copy。由于 `comb` 经过 Sinkhorn 近似双随机归一化，它在不同 HC copy 之间搬运 residual 信息时不会让某一行或某一列的总量明显失衡。

## 13. 伪代码等价实现

如果不用 TileLang，单个 token 的逻辑可以近似写成：

```python
def one_token_hc_split_sinkhorn(mix, hc_scale, hc_base, hc=4, sinkhorn_iters=20, eps=1e-6):
    mix_hc = (2 + hc) * hc
    assert mix.shape == (mix_hc,)

    pre_logits = mix[:hc] * hc_scale[0] + hc_base[:hc]
    post_logits = mix[hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc]
    comb_logits = mix[2 * hc:].reshape(hc, hc) * hc_scale[2] + hc_base[2 * hc:].reshape(hc, hc)

    pre = torch.sigmoid(pre_logits) + eps
    post = 2 * torch.sigmoid(post_logits)

    comb = torch.softmax(comb_logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb
```

整个 kernel 就是把上面的逻辑融合到一个 GPU kernel 中，并行处理 `n = batch * seq` 个 token。

## 14. 为什么要融合成一个 kernel

这个计算如果用普通 PyTorch op 写，会涉及很多小算子：

- slicing
- sigmoid
- softmax
- exp
- reduce sum
- 多轮 row/column normalization
- reshape/view

对于 `hc = 4` 这种很小的矩阵，单个 token 只有 24 个输入值和 16 个 `comb` 元素。真正的开销不在 FLOPs，而在大量小 kernel launch 和中间张量读写。TileLang 融合后：

- 每个 token 在一个 kernel instance 内完成全部 `pre/post/comb` 生成。
- `comb` 的 Sinkhorn 迭代留在 fragment/register 中，减少 global memory 往返。
- 只需要最终写回 `pre`、`post`、`comb`。
- 对 inference 这种小 batch、逐 token 或短序列场景更友好。

## 15. 数值细节

几个值得注意的点：

1. `comb` 先做 row-wise softmax，而不是直接对原始 logits 做 Sinkhorn。

   这样能保证初始矩阵为正，并且每行已经有合理概率分布。

2. softmax 使用 `row_max` 做稳定化。

   ```text
   exp(x - max(x))
   ```

   这是避免指数溢出的标准写法。

3. `eps` 出现三类位置。

   ```text
   pre = sigmoid(...) + eps
   comb = softmax(...) + eps
   normalize denominator = sum + eps
   ```

   这些都是为了避免 0 权重、0 除和极小数导致的不稳定。

4. 最终 `comb` 是近似双随机，不是严格双随机。

   原因包括：

   - 迭代次数有限。
   - 分母加了 `eps`。
   - FP32 有舍入误差。
   - 最后一步是 column normalization，所以列和通常比行和更接近 1。

5. `pre` 和 `post` 不做 softmax。

   它们不是概率分布，而是门控系数。`pre` 在 `(eps, 1 + eps)`，`post` 在 `(0, 2)`。

## 16. 一句话总结

`hc_split_sinkhorn_kernel` 是 Hyper-Connections 的系数生成 kernel：它对每个 token 的 `mixes` 向量做拆分，生成 `pre`、`post` 两组 sigmoid gate，以及一个经过 softmax + Sinkhorn 迭代归一化的 `comb` 矩阵。`pre` 负责把多份 HC hidden state 合成单份输入，`post` 负责把子层输出注入回多份 HC state，`comb` 负责在多份 residual state 之间做近似守恒的混合。
