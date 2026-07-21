# DeepSeek-V4 论文公式与 `inference/` 代码对应解析

> 说明：本解析基于仓库根目录的 `DeepSeek_V4.pdf` 与 `inference/` 下的参考实现。`inference/` 是一个面向 DeepSeek-V4-Flash（以及可复用于 Pro）的简化推理实现，核心用 PyTorch + TileLang 写成。下面按论文结构把关键数学公式映射到具体代码位置。

---

## 1. `inference/` 目录结构与整体流程

| 文件 | 职责 |
|------|------|
| `inference/generate.py` | 入口：加载 tokenizer、模型，执行 prefill + decode 的自回归生成。 |
| `inference/model.py` | 模型定义：`Transformer`、`Block`、`Attention`、`MoE`、`Gate`、`Expert`、`MTPBlock`、`ParallelHead`、HC 相关参数。 |
| `inference/kernel.py` | TileLang 写的核心算子：FP8/FP4 量化、`fp8_gemm`、`fp4_gemm`、`sparse_attn`、`hc_split_sinkhorn`。 |
| `inference/convert.py` | 把 HuggingFace 格式 checkpoint 转换成本实现所需的 sharded safetensors，处理 FP4/FP8 权重与 scale。 |
| `inference/config.json` | 运行时的超参数（对应 `ModelArgs`），例如 `n_layers=43`、`n_hash_layers=3`、`compress_ratios=[0,0,4,128,...]`。 |

整体前向流程在 `model.py:809`：`Transformer.forward`：

```python
h = self.embed(input_ids)                      # Embedding
h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)  # 展开成 hc_mult 份
for layer in self.layers:
    h = layer(h, start_pos, input_ids)         # Block
logits = self.head(h, ...)                     # ParallelHead
```

对应论文图 2 的整体架构：Embedding → HC expand → \(L\) 个 Transformer Block → HC head → Prediction Head。

---

## 2. Manifold-Constrained Hyper-Connections（mHC）

### 论文公式

论文第 2.2 节给出标准 HC 与 mHC 的更新：

- 标准 HC（式 1）：
  \[
  X_{l+1} = B_l X_l + C_l F_l(A_l X_l)
  \]

- mHC 把 \(B_l\) 约束在双随机矩阵流形 \(\mathcal{M}\) 上（式 2）。
- 动态参数化（式 3–5）：
  \[
  \tilde A_l = \alpha_l^{pre} \cdot (\hat X_l W_l^{pre}) + S_l^{pre}
  \]
  \[
  \tilde B_l = \alpha_l^{res} \cdot \text{Mat}(\hat X_l W_l^{res}) + S_l^{res}
  \]
  \[
  \tilde C_l = \alpha_l^{post} \cdot (\hat X_l W_l^{post})^T + S_l^{post}
  \]
- 约束（式 6–7）：\(A_l = \sigma(\tilde A_l)\)，\(C_l = 2\sigma(\tilde C_l)\)。
- \(B_l\) 通过 Sinkhorn-Knopp 迭代得到（式 8）：
  \[
  M^{(t)} = T_r(T_c(M^{(t-1)}))
  \]

### 代码对应

`Block` 在 `model.py:659` 把 mHC 拆成 `hc_pre` 与 `hc_post` 两步：

```python
x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
x = self.attn_norm(x)
x = self.attn(x, start_pos)
x = self.hc_post(x, residual, post, comb)
```

- `hc_pre`（`model.py:680`）对应式 3 的生成过程：
  - 先把 `x` flatten 成 `[b,s,hc*d]`；
  - 做 RMSNorm（对应 \(\hat X_l = \text{RMSNorm}(\text{vec}(X_l))\)）；
  - `F.linear(x, hc_fn)` 对应 \(\hat X_l W_l^{pre/res/post}\)；
  - 调用 `hc_split_sinkhorn`（`kernel.py:371`）生成 `pre`、`post`、`comb`：
    - `pre = sigmoid(mixes * hc_scale[0] + hc_base[:hc]) + eps` 对应式 6 的 \(A_l\)；
    - `post = 2 * sigmoid(...)` 对应式 7 的 \(C_l\)；
    - `comb` 先 softmax 再用 Sinkhorn 迭代归一化，对应式 8 的双随机约束 \(B_l\)。

- `hc_post`（`model.py:690`）对应式 1 的更新：
  \[
  y = \text{post} \otimes x + \sum_j \text{comb}_{ij} \cdot \text{residual}_j
  \]
  其中 `post.unsqueeze(-1) * x.unsqueeze(-2)` 是 \(C_l F_l(A_l X_l)\) 部分，`comb` 与 `residual` 的 einsum 对应 \(B_l X_l\) 部分。

- `Transformer.forward`（`model.py:822`）把隐状态 `h` 复制成 `hc_mult` 份，对应论文中把残差流从 \(\mathbb{R}^d\) 扩展到 \(\mathbb{R}^{n_{hc}\times d}\)。

- `ParallelHead.hc_head`（`model.py:735`）在最后一层把 `hc_mult` 份聚合回一份，再进 RMSNorm 和 lm_head。

---

## 3. Hybrid Attention：CSA / HCA

论文第 2.3 节把注意层分为 **Compressed Sparse Attention（CSA，压缩比 \(m=4\)）** 和 **Heavily Compressed Attention（HCA，压缩比 \(m'=128\)）**，两者交替放置。代码里通过 `args.compress_ratios[layer_id]` 控制每层压缩比：列表中为 `0` 表示仅 sliding window，`4` 为 CSA，`128` 为 HCA。

### 3.1 低秩 Q 投影（Shared KV MQA）

论文 CSA 式 13–14、HCA 式 24–25：

\[
c_t^Q = h_t W^{DQ}, \qquad q_t = c_t^Q W^{UQ}
\]

代码在 `Attention.forward`（`model.py:491`）：

```python
qr = q = self.q_norm(self.wq_a(x))       # h_t -> c_t^Q
q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))  # c_t^Q -> q_t
q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)       # 额外的 per-head RMSNorm
```

- `wq_a` 对应 \(W^{DQ}\)；
- `q_norm` 是低秩 latent 上的 RMSNorm；
- `wq_b` 对应 \(W^{UQ}\)。

### 3.2 KV 压缩：Compressor

#### CSA（ratio=4，overlap）

论文式 9–12：

\[
C^a = H W_{KV}^a, \; C^b = H W_{KV}^b, \;
Z^a = H W_Z^a, \; Z^b = H W_Z^b
\]

\[
[S_{mi:m(i+1)-1}^a; S_{m(i-1):mi-1}^b] =
\text{Softmax}_{row}([Z_{mi:m(i+1)-1}^a + B^a; Z_{m(i-1):mi-1}^b + B^b])
\]

\[
C_i^{Comp} = \sum_{j=mi}^{m(i+1)-1} S_j^a \odot C_j^a +
             \sum_{j=m(i-1)}^{mi-1} S_j^b \odot C_j^b
\]

代码 `Compressor`（`model.py:286`）实现：

- `wkv(x)` 产生 \(C\)（相当于把 \(C^a, C^b\) 拼接在 `overlap` 维度上）；
- `wgate(x)` 产生 \(Z\)；
- `ape`（`model.py:301`）对应可学习位置偏置 \(B^a, B^b\)；
- `overlap_transform`（`model.py:314`）把当前块与前一块拼接，生成 2m 个元素；
- `score.softmax(dim=2)` 与 `kv * score` 的加权求和（`model.py:349`）对应式 11–12。

> 注意：代码只显式维护了一份 `kv_state` / `score_state`，通过 overlap_transform 把前一块的信息带进来，等价于 \(C^a/C^b\) 两路。

#### HCA（ratio=128，无 overlap）

论文式 20–23：

\[
C = H W_{KV}, \quad Z = H W_Z
\]

\[
S_{m'i:m'(i+1)-1} = \text{Softmax}_{row}(Z_{m'i:m'(i+1)-1} + B)
\]

\[
C_i^{Comp} = \sum_{j=m'i}^{m'(i+1)-1} S_j \odot C_j
\]

代码在 `compress_ratio != 4` 时走 `overlap=False` 分支（`model.py:297`），直接对 `ratio` 个 token 做 softmax 加权求和，没有 overlap_transform，对应式 22–23。

### 3.3 Lightning Indexer（稀疏选择）与 `sparse_attn`

论文 CSA 式 15–17：

\[
w_t^I = h_t W_w
\]

\[
I_{t,s} = \sum_{h=1}^{n_h^I} w_{t,h}^I \cdot \text{ReLU}(q_{t,h}^I \cdot K_s^{I,Comp})
\]

\[
C_t^{Sprs,Comp} = \{ C_s^{Comp} \mid I_{t,s} \in \text{Top-k}(I_{t,:}) \}
\]

代码 `Indexer`（`model.py:387`）：

```python
q = self.wq_b(qr)                                  # 复用 Attention 的 c_t^Q
q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
apply_rotary_emb(q[..., -rd:], freqs_cis)
q = rotate_activation(q)
fp4_act_quant(q, fp4_block_size, True)             # FP4 模拟

weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
```

- `weights_proj` 对应 \(W_w\)，产生 \(w_t^I\)；
- `index_score` 的 `relu_()` 与加权求和对应式 16；
- `topk(...)` 对应式 17 的 Top-k 选择。

`sparse_attn`（`kernel.py:276`）是核心 attention 算子：

- 通过 `topk_idxs` gather 出被选中的 KV；
- 使用 online softmax（running max / sum）做 FlashAttention 风格的计算；
- 对 `idxs == -1` 的位置把 score 设成 \(-\infty\)，实现 causal / padding 掩码。

### 3.4 Sliding Window + Attention Sink

论文 2.3.3 节：引入最近 \(n_{win}\) 个 token 的 sliding window KV，以及可学习的 attention sink。

- Sliding window：`get_window_topk_idxs`（`model.py:262`）构造最近 `window_size=128` 个 token 的索引；在 decode 阶段通过 `start_pos % win` 循环覆写 KV cache（`model.py:537`）。
- Attention sink：式 27
  \[
  s_{h,i,j} = \frac{\exp(z_{h,i,j})}{\sum_k \exp(z_{h,i,k}) + \exp(z'_h)}
  \]
  在 `sparse_attn_kernel`（`kernel.py:345`）中：
  ```python
  sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
  ```
  然后 `acc_o /= sum_exp`，即在分母额外加上 \(\exp(z'_h)\)。

### 3.5 Partial RoPE 与 Grouped Output Projection

论文 2.3.3：只对 Q/KV/O 的 **最后 64 维** 应用 RoPE；并在 attention 输出后做反向 RoPE。

代码：

- `rope_head_dim=64`，`head_dim=512`；
- `apply_rotary_emb(q[..., -rd:], freqs_cis)`（`model.py:506`）；
- `apply_rotary_emb(kv[..., -rd:], freqs_cis)`（`model.py:511`）；
- 输出后 `apply_rotary_emb(o[..., -rd:], freqs_cis, True)`（`model.py:541`），`inverse=True` 对应论文的 “position \(-i\)” 反向旋转。

Grouped Output Projection（论文图 3/4 描述）：

- `wo_a` 把 `n_heads * head_dim // n_groups` 映射到 `n_groups * o_lora_rank`；
- `wo_b` 把 `n_groups * o_lora_rank` 映射回 `dim`；
- 代码 `model.py:544`：
  ```python
  o = o.view(bsz, seqlen, self.n_local_groups, -1)
  wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
  o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
  x = self.wo_b(o.flatten(2))
  ```
  对应先按 group 投影到低秩空间，再投影回 hidden size。

### 3.6 KV 存储的混合精度

论文 2.3.4：RoPE 维度用 BF16，其余维度用 FP8。

代码 `Attention.forward`（`model.py:513`）：

```python
act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)   # 非 RoPE 部分量化 FP8
```

`kv[..., -rd:]` 保持 bf16，实现论文所说的混合精度 KV cache。

---

## 4. DeepSeekMoE

### 4.1 Gate / Hash Routing

论文 2.1：DeepSeek-V4 沿用 DeepSeekMoE，但把 score 函数从 Sigmoid 改为 \(\sqrt{\text{Softplus}(\cdot)}\)；前 `n_hash_layers` 层使用 Hash routing。

代码 `Gate`（`model.py:553`）：

```python
scores = linear(x.float(), self.weight.float())
if self.score_func == "softmax": ...
elif self.score_func == "sigmoid": ...
else:
    scores = F.softplus(scores).sqrt()          # 对应 sqrt(softplus)

if self.hash:
    indices = self.tid2eid[input_ids]            # Hash routing：按 token ID 直接查表
else:
    indices = scores.topk(self.topk, dim=-1)[1]

weights = original_scores.gather(1, indices)
if self.score_func != "softmax":
    weights /= weights.sum(dim=-1, keepdim=True)
weights *= self.route_scale
```

- `route_scale=1.5`（`config.json`）对应论文的负载均衡后缩放；
- `bias` 只在非 Hash 层使用，作为 auxiliary-loss-free 的 score correction。

### 4.2 Expert（SwiGLU）

论文中的 FFN 专家为 SwiGLU：

\[
\text{Expert}(x) = W_2(\text{SiLU}(W_1 x) \odot (W_3 x))
\]

代码 `Expert.forward`（`model.py:603`）：

```python
gate = self.w1(x).float()
up   = self.w3(x).float()
if self.swiglu_limit > 0:
    up   = torch.clamp(up,   min=-self.swiglu_limit, max=self.swiglu_limit)
    gate = torch.clamp(gate, max=self.swiglu_limit)
x = F.silu(gate) * up
if weights is not None:
    x = weights * x
return self.w2(x.to(dtype))
```

对应 SwiGLU 公式；`swiglu_limit` 是训练稳定性裁剪。

### 4.3 MoE 聚合

代码 `MoE.forward`（`model.py:636`）：

```python
weights, indices = self.gate(x, input_ids.flatten())
y = torch.zeros_like(x, dtype=torch.float32)
for i in range(self.experts_start_idx, self.experts_end_idx):
    if counts[i] == 0: continue
    idx, top = torch.where(indices == i)
    y[idx] += expert(x[idx], weights[idx, top, None])
if world_size > 1:
    dist.all_reduce(y)
y += self.shared_experts(x)
```

对应论文：每个 token 激活 top-k routed experts + 1 shared expert，结果求和。

---

## 5. FP8 / FP4 量化与 Kernel 映射

论文 2.3.4/5.2.1 提到：

- KV 与激活用 FP8；
- Lightning Indexer 的 Q/K 路径用 FP4；
- Routed expert 权重用 FP4。

### 5.1 激活量化 `act_quant` / `fp4_act_quant`

代码 `kernel.py:40` 的 `act_quant_kernel`：

- 对输入按 `block_size=128` 分块；
- 计算每块 absmax，得到 scale `s = amax / 448`（或 round 到 2 的幂实现 MXFP 的 `ue8m0`）；
- 量化值 `y = clamp(x / s, -448, 448)`。

`fp4_act_quant`（`kernel.py:128`）类似，但 `block_size=32`，max=6.0，scale 使用 `float8_e8m0fnu`。对应论文对 Indexer Q/K 以及 expert 权重的 FP4 量化。

### 5.2 FP8 GEMM `fp8_gemm`

论文对量化矩阵乘法的描述：两个 FP8 矩阵分别带 per-block scale，乘法时把 scale 乘回 accumulator。

代码 `kernel.py:203`：

```python
Scale_C_shared[i] = scales_a[by*block_M+i, k] * scales_b[bx*block_N//group_size, k]
...
C_local_accum[i,j] += C_local[i,j] * Scale_C_shared[i]
```

即 \(C = (A_{fp8} \cdot B_{fp8}) \cdot s_A \cdot s_B\)。

### 5.3 FP4 权重 GEMM `fp4_gemm`

代码 `kernel.py:441`：

- A 是 FP8 激活（per-128 scale）；
- B 是 FP4 权重，物理存储为 `[N, K//2]` 的 `float4_e2m1fn_x2`（每字节 2 个 FP4，沿 K 方向 packed），逻辑 `[N, K]`；
- B 的 scale 是 per-32 的 `float8_e8m0fnu`；
- kernel 内先把 FP4 cast 到 FP8，再做 FP8 GEMM，最后乘上各自的 scale。

对应 `linear` 函数（`model.py:115`）的分发：

```python
if weight.dtype == torch.float4_e2m1fn_x2:
    x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
    return fp4_gemm(x, s, weight, weight.scale, scale_dtype)
elif weight.dtype == torch.float8_e4m3fn:
    x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
    return fp8_gemm(x, s, weight, weight.scale, scale_dtype)
else:
    return F.linear(x, weight)
```

### 5.4 `convert.py` 中的权重量化

- `cast_e2m1fn_to_e4m3fn`（`convert.py:17`）把 HF 中 int8 形式的 FP4 权重无损地转回 `float8_e4m3fn`，并重新计算 per-128 scale，用于 `--expert-dtype=fp8` 模式；
- 若 `--expert-dtype=fp4`，则直接 `view(torch.float4_e2m1fn_x2)`（`convert.py:149`），配合 `weight.scale`。

---

## 6. Multi-Token Prediction（MTP）

论文 2.1 说明沿用 DeepSeek-V3 的 MTP。代码 `MTPBlock`（`model.py:745`）在 `Transformer` 末尾额外堆叠 `n_mtp_layers` 层：

```python
self.mtp = torch.nn.ModuleList()
for layer_id in range(args.n_mtp_layers):
    self.mtp.append(MTPBlock(args.n_layers + layer_id, args))
    self.mtp[-1].embed = self.embed
    self.mtp[-1].head = self.head
```

`MTPBlock.forward`（`model.py:764`）：

\[
x_{mtp} = e\_proj(\text{Norm}(E_{t})) + h\_proj(\text{Norm}(h_t))
\]

然后过一个完整的 `Block`，再用共享的 `head` 预测下一个 token。对应 MTP 的 “将当前层隐状态与真实 token embedding 相加后额外预测” 结构。

---

## 7. 关键超参数与配置对应

`inference/config.json` 中的值直接对应论文/代码中的符号：

| config 键 | 论文符号 | 含义 |
|-----------|----------|------|
| `n_layers=43` | \(L\) | Transformer 层数 |
| `n_hash_layers=3` | — | 前 3 层使用 Hash routing |
| `n_routed_experts=256` | — | routed expert 总数 |
| `n_activated_experts=6` | top-k | 每 token 激活的 routed expert 数 |
| `score_func=sqrtsoftplus` | \(\sqrt{\text{Softplus}}\) | gate score 函数 |
| `route_scale=1.5` | — | routing weight 缩放 |
| `q_lora_rank=1024` | \(d_c\) | Q 低秩压缩维度 |
| `head_dim=512` | \(c\) | attention head 维度 |
| `rope_head_dim=64` | — | 参与 RoPE 的维度 |
| `o_groups=8`, `o_lora_rank=1024` | \(g, d_g\) | grouped output projection 参数 |
| `window_size=128` | \(n_{win}\) | sliding window 长度 |
| `compress_ratios` | \(m=4\) / \(m'=128\) | 每层 KV 压缩比；`0` 表示不压缩 |
| `index_n_heads=64`, `index_head_dim=128`, `index_topk=512` | \(n_h^I, c_I, k\) | Lightning Indexer 参数 |
| `hc_mult=4` | \(n_{hc}\) | Hyper-Connections 展开倍数 |
| `hc_sinkhorn_iters=20` | \(t_{max}\) | Sinkhorn 迭代次数（论文式 8） |
| `dtype=fp8`, `scale_fmt=ue8m0`, `expert_dtype=fp4` | — | 量化配置 |

---

## 8. 小结

| 论文模块 | 主要公式/段落 | `inference/` 对应代码 |
|----------|----------------|------------------------|
| mHC | 式 1–8 | `model.py:659-707`, `kernel.py:371` |
| 低秩 Q 投影 | 式 13–14, 24–25 | `model.py:503-505` |
| KV 压缩（CSA/HCA） | 式 9–12, 20–23 | `model.py:286-384` |
| Lightning Indexer | 式 15–17 | `model.py:387-440` |
| Sparse Attention + Sink | 式 19, 27 | `kernel.py:276-368` |
| Sliding Window | 2.3.3 | `model.py:262-272`, `model.py:514`, `model.py:537` |
| Partial RoPE | 2.3.3 | `model.py:506`, `model.py:511`, `model.py:541` |
| Grouped Output Proj | 2.3.1/2.3.2 | `model.py:544-549` |
| DeepSeekMoE Gate | 2.1 | `model.py:553-591` |
| SwiGLU Expert | 2.1 | `model.py:594-613` |
| FP8/FP4 量化/GEMM | 2.3.4, 5.2.1 | `kernel.py:40-537`, `model.py:115-127` |
| MTP | 2.1 | `model.py:745-773` |

以上即为 `DeepSeek_V4.pdf` 中关键数学公式与 `inference/` 参考实现之间的主要对应关系。

---

## 9. `/share/users/like/temp/gtest.png` 图片内容

该图片是《龙珠》中 **少年孙悟空（Kid Goku）** 的卡通立绘：

- 身穿橙色龟仙流武道服，胸前有黄色圆形“龟”字标志；
- 黑色腰带、蓝色腕带与蓝色武道鞋；
- 标志性的黑色刺猬头、尾巴以及握拳的战斗姿势；
- 背景为透明（PNG 透明格子）。
