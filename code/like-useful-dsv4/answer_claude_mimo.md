# DeepSeek V4 Flash 推理代码与论文数学公式对应分析

## 目录
1. [RMSNorm 归一化](#1-rmsnorm-归一化)
2. [RoPE 旋转位置编码](#2-rope-旋转位置编码)
3. [MLA 多头潜在注意力](#3-mla-多头潜在注意力)
4. [KV Cache 压缩](#4-kv-cache-压缩)
5. [MoE 混合专家](#5-moe-混合专家)
6. [Hyper-Connections 超连接](#6-hyper-connections-超连接)
7. [量化 (FP8/FP4)](#7-量化-fp8fp4)
8. [采样策略](#8-采样策略)

---

## 1. RMSNorm 归一化

### 论文公式

RMSNorm (Root Mean Square Normalization) 公式:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}} \cdot \gamma$$

其中 $d$ 是隐藏维度，$\epsilon$ 是防止除零的小常数，$\gamma$ 是可学习的缩放参数。

### 代码实现

**文件**: `model.py:190-203`

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)  # 计算均方: (1/d) * sum(x^2)
        x = x * torch.rsqrt(var + self.eps)       # x / sqrt(var + eps)
        return (self.weight * x).to(dtype)         # 乘以gamma参数
```

### 对应关系

| 论文公式 | 代码实现 | 行号 |
|---------|---------|------|
| $\frac{1}{d}\sum x_i^2$ | `x.square().mean(-1, keepdim=True)` | 201 |
| $\sqrt{\cdot + \epsilon}$ | `torch.rsqrt(var + self.eps)` | 202 |
| $\cdot \gamma$ | `self.weight * x` | 203 |

---

## 2. RoPE 旋转位置编码

### 论文公式

旋转位置编码 (Rotary Position Embedding) 核心公式:

对于位置 $m$ 的向量 $x$，应用旋转:

$$f(x, m) = x \odot e^{im\theta}$$

其中 $\theta_j = 10000^{-2j/d}$ 是频率基数。

YaRN 扩展公式 (用于长序列外推):

$$\theta'_j = \theta_j \cdot \begin{cases}
\frac{1}{factor} & \text{if dim is high freq} \\
1 & \text{if dim is low freq} \\
\text{interpolate} & \text{otherwise}
\end{cases}$$

### 代码实现

**文件**: `model.py:206-251`

```python
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """预计算旋转位置编码的复数指数"""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # 基础频率: theta_j = base^(-2j/d)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    if original_seq_len > 0:  # YaRN 扩展
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        # 频率插值: 低频保持不变，高频缩放
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)           # [seq_len, dim/2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^{i*m*theta}
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False):
    """应用旋转位置编码"""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()  # 逆旋转
    # ... 维度处理 ...
    x = torch.view_as_real(x * freqs_cis).flatten(-2)  # 复数乘法 = 旋转
    y.copy_(x)
    return y
```

### 对应关系

| 论文公式 | 代码实现 | 行号 |
|---------|---------|------|
| $\theta_j = base^{-2j/d}$ | `1.0 / (base ** (torch.arange(0, dim, 2) / dim))` | 227 |
| YaRN 频率插值 | `freqs / factor * (1 - smooth) + freqs * smooth` | 231 |
| $e^{im\theta}$ | `torch.polar(torch.ones_like(freqs), freqs)` | 235 |
| $x \odot e^{im\theta}$ | `x * freqs_cis` (复数乘法) | 249 |

---

## 3. MLA 多头潜在注意力

### 论文公式

Multi-head Latent Attention (MLA) 核心思想是使用低秩投影压缩KV缓存:

**Query 投影** (低秩):
$$q = W_q^B \cdot \text{RMSNorm}(W_q^A \cdot h)$$

**KV 压缩** (低秩):
$$kv = W_{kv} \cdot h$$

**注意力计算**:
$$o = \text{Attention}(q, kv) = \text{softmax}\left(\frac{q \cdot kv^T}{\sqrt{d}}\right) \cdot kv$$

**输出投影** (分组低秩):
$$out = W_o^B \cdot \text{Group}(o) \cdot W_o^A$$

### 代码实现

**文件**: `model.py:443-550`

```python
class Attention(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        # Query 低秩投影
        self.wq_a = Linear(self.dim, self.q_lora_rank)        # W_q^A
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)  # W_q^B

        # KV 投影 (共享)
        self.wkv = Linear(self.dim, self.head_dim)  # W_{kv}
        self.kv_norm = RMSNorm(self.head_dim, self.eps)

        # 输出低秩投影
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * args.o_lora_rank
        )  # W_o^A
        self.wo_b = RowParallelLinear(
            self.n_groups * args.o_lora_rank, self.dim
        )  # W_o^B

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]

        # 1. Query 低秩投影
        qr = q = self.q_norm(self.wq_a(x))  # q = RMSNorm(W_q^A * h)
        q = self.wq_b(q)                     # q = W_q^B * q
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)  # Q归一化
        apply_rotary_emb(q[..., -rd:], freqs_cis)  # 应用RoPE

        # 2. KV 投影
        kv = self.wkv(x)                    # kv = W_{kv} * h
        kv = self.kv_norm(kv)               # RMSNorm
        apply_rotary_emb(kv[..., -rd:], freqs_cis)  # 应用RoPE

        # 3. 注意力计算 (稀疏)
        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)

        # 4. 输出投影
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)  # 分组投影
        x = self.wo_b(o.flatten(2))  # 最终投影
        return x
```

### 对应关系

| 论文公式 | 代码实现 | 行号 |
|---------|---------|------|
| $q = W_q^B \cdot \text{RMSNorm}(W_q^A \cdot h)$ | `q = self.wq_b(self.q_norm(self.wq_a(x)))` | 503-504 |
| $kv = W_{kv} \cdot h$ | `kv = self.wkv(x)` | 509 |
| $\text{softmax}(q \cdot kv^T / \sqrt{d})$ | `sparse_attn(q, kv, ...)` | 535 |
| $out = W_o^B \cdot \text{Group}(o) \cdot W_o^A$ | `wo_b(wo_a投影后的结果)` | 548-549 |

---

## 4. KV Cache 压缩

### 论文公式

KV Cache 压缩使用学习型门控池化:

对于压缩窗口大小 $r$ 的tokens:

$$kv_{compressed} = \sum_{i=1}^{r} \text{softmax}(score_i) \cdot kv_i$$

其中 $score_i = \text{gate}(x_i) + ape_i$，$ape_i$ 是可学习的绝对位置编码。

### 代码实现

**文件**: `model.py:286-384`

```python
class Compressor(nn.Module):
    def __init__(self, args, compress_ratio, head_dim, rotate=False):
        # 可学习的绝对位置编码
        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim))
        # 门控投影
        self.wkv = Linear(self.dim, coff * self.head_dim)
        self.wgate = Linear(self.dim, coff * self.head_dim)

    def forward(self, x, start_pos):
        # 1. 计算KV和门控分数
        kv = self.wkv(x)       # [batch, seq, dim]
        score = self.wgate(x)  # [batch, seq, dim]

        # 2. 添加位置编码
        score = score + self.ape[position_within_window]

        # 3. 门控池化压缩
        if should_compress:
            # score: [batch, ratio, dim]
            # softmax压缩
            kv_compressed = (kv * score.softmax(dim=2)).sum(dim=2)

        return kv_compressed
```

### 对应关系

| 论文公式 | 代码实现 | 行号 |
|---------|---------|------|
| $score_i = \text{gate}(x_i)$ | `score = self.wgate(x)` | 331 |
| $score_i + ape_i$ | `score + self.ape[...]` | 339, 345, 352 |
| $\sum \text{softmax}(score_i) \cdot kv_i$ | `(kv * score.softmax(dim=2)).sum(dim=2)` | 349, 359, 366 |

---

## 5. MoE 混合专家

### 论文公式

Mixture-of-Experts 路由机制:

**路由分数计算** (支持多种激活函数):

$$s_i = \begin{cases}
\text{softmax}(W_g \cdot h)_i & \text{if score\_func=softmax} \\
\sigma(W_g \cdot h)_i & \text{if score\_func=sigmoid} \\
\sqrt{\text{softplus}(W_g \cdot h)_i} & \text{if score\_func=sqrtsoftplus}
\end{cases}$$

**Top-K 路由**:

$$\text{indices} = \text{TopK}(s + bias, k)$$

$$\text{weights}_i = \frac{s_{\text{indices}_i}}{\sum_j s_{\text{indices}_j}} \cdot \text{route\_scale}$$

**专家计算** (SwiGLU FFN):

$$\text{Expert}(x) = W_2 \cdot (\text{SiLU}(W_1 \cdot x) \odot W_3 \cdot x)$$

**MoE 聚合**:

$$y = \sum_{i \in \text{TopK}} \text{weights}_i \cdot \text{Expert}_i(x) + \text{SharedExpert}(x)$$

### 代码实现

**文件**: `model.py:553-651`

```python
class Gate(nn.Module):
    def forward(self, x, input_ids):
        # 1. 计算路由分数
        scores = linear(x.float(), self.weight.float())

        # 2. 激活函数
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus
            scores = F.softplus(scores).sqrt()

        # 3. Top-K 选择 (带偏置)
        if self.bias is not None:
            scores = scores + self.bias
        indices = scores.topk(self.topk, dim=-1)[1]

        # 4. 提取权重并归一化
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale

        return weights, indices

class Expert(nn.Module):
    def forward(self, x, weights):
        # SwiGLU FFN
        gate = self.w1(x)   # W_1 * x
        up = self.w3(x)     # W_3 * x
        x = F.silu(gate) * up  # SiLU(W_1*x) * W_3*x

        if weights is not None:
            x = weights * x  # 路由权重

        return self.w2(x)   # W_2 * x

class MoE(nn.Module):
    def forward(self, x, input_ids):
        # 1. 路由
        weights, indices = self.gate(x, input_ids)

        # 2. 聚合专家输出
        y = torch.zeros_like(x, dtype=torch.float32)
        for i in range(self.experts_start_idx, self.experts_end_idx):
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx], weights[idx, top, None])

        # 3. 添加共享专家
        y += self.shared_experts(x)

        return y
```

### 对应关系

| 论文公式 | 代码实现 | 行号 |
|---------|---------|------|
| $\sqrt{\text{softplus}(W_g \cdot h)}$ | `F.softplus(scores).sqrt()` | 578 |
| $\text{TopK}(s + bias, k)$ | `scores.topk(self.topk, dim=-1)` | 586 |
| $\text{SiLU}(W_1 x) \odot W_3 x$ | `F.silu(gate) * up` | 610 |
| $\sum w_i \cdot \text{Expert}_i(x)$ | `y[idx] += expert(x[idx], weights[...])` | 647 |
| $+ \text{SharedExpert}(x)$ | `y += self.shared_experts(x)` | 650 |

---

## 6. Hyper-Connections 超连接

### 论文公式

Hyper-Connections (HC) 维护 $k$ 个隐藏状态副本 ( $k = \text{hc\_mult}$):

**预处理 (Pre-HC)** - 将 $k$ 个副本压缩为 1 个:

$$\text{mixes} = \text{Linear}(x) \cdot \text{rsqrt}(\cdot)$$

$$(pre, post, comb) = \text{SplitSinkhorn}(\text{mixes})$$

$$y = \sum_{i=1}^{k} pre_i \cdot x_i$$

**后处理 (Post-HC)** - 扩展回 $k$ 个副本:

$$y_i = post_i \cdot x + \sum_{j=1}^{k} comb_{ij} \cdot \text{residual}_j$$

**Sinkhorn 归一化** (用于 $comb$ 矩阵):

$$comb = \text{softmax}(comb) + \epsilon$$

重复行归一化和列归一化:

$$comb = \frac{comb}{\sum_{\text{row}} comb + \epsilon}, \quad comb = \frac{comb}{\sum_{\text{col}} comb + \epsilon}$$

### 代码实现

**文件**: `model.py:654-693` 和 `kernel.py:371-438`

```python
class Block(nn.Module):
    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        # x: [b,s,hc,d] -> y: [b,s,d]
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt  # 线性变换 + 归一化

        # SplitSinkhorn 操作
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, ...)

        # 加权聚合
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y, post, comb

    def hc_post(self, x, residual, post, comb):
        # x: [b,s,d] -> y: [b,s,hc,d]
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + \
            torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y

@tilelang.jit
def hc_split_sinkhorn_kernel(hc, sinkhorn_iters, eps):
    """Sinkhorn 归一化 GPU 内核"""

    @T.prim_func
    def kernel_(mixes, hc_scale, hc_base, pre, post, comb):
        # 1. 计算 pre (sigmoid激活)
        pre[i, j] = sigmoid(mixes[j] * hc_scale[0] + hc_base[j]) + eps

        # 2. 计算 post (sigmoid激活，2倍缩放)
        post[i, j] = 2 * sigmoid(mixes[j + hc] * hc_scale[1] + hc_base[j + hc])

        # 3. 计算 comb (带softmax初始化)
        comb[j, k] = mixes[...] * hc_scale[2] + hc_base[...]
        comb = softmax(comb) + eps  # 行归一化

        # 4. Sinkhorn 迭代 (行列交替归一化)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1) + eps)  # 行归一化
            comb = comb / (comb.sum(-2) + eps)  # 列归一化

        return pre, post, comb
```

### 对应关系

| 论文公式 | 代码实现 | 行号 |
|---------|---------|------|
| $\text{mixes} = \text{Linear}(x) \cdot \text{rsqrt}$ | `F.linear(x, hc_fn) * rsqrt` | 685 |
| $pre = \sigma(\cdot) + \epsilon$ | `sigmoid(...) + eps` | kernel.py:392 |
| $y = \sum pre_i \cdot x_i$ | `torch.sum(pre.unsqueeze(-1) * x, dim=2)` | 687 |
| $y_i = post_i \cdot x + \sum comb_{ij} \cdot r_j$ | kernel.py:394 和 model.py:692 |
| Sinkhorn 迭代 | kernel.py:415-423 |

---

## 7. 量化 (FP8/FP4)

### 论文公式

**块量化 (Block Quantization)**:

对于输入向量 $x$，块大小 $B$:

$$s = \frac{\max(|x_{\text{block}}|)}{\text{FP8\_MAX}}$$

$$x_{\text{quant}} = \text{clamp}\left(\frac{x}{s}, -448, 448\right)$$

**MXFP 格式** (Power-of-2 缩放):

$$s_{\text{MXFP}} = 2^{\lceil \log_2(s) \rceil}$$

**FP8 矩阵乘法**:

$$C = (A_{\text{fp8}} \cdot s_A) @ (B_{\text{fp8}} \cdot s_B)^T$$

### 代码实现

**文件**: `kernel.py:40-126`

```python
def act_quant(x, block_size=128, scale_fmt=None, ...):
    """块量化 FP8"""

    # 1. 计算块内最大绝对值
    amax = reduce_absmax(x, dim=last_dim)  # max(|x_block|)

    # 2. 计算缩放因子
    if round_scale:  # MXFP
        s = 2 ** ceil(log2(amax / 448))
    else:
        s = amax / 448

    # 3. 量化
    x_quant = clamp(x / s, -448, 448)

    return x_quant, s

@tilelang.jit
def fp8_gemm_kernel(N, K, ...):
    """FP8 矩阵乘法内核"""

    @T.prim_func
    def kernel_(A, B, C, scales_a, scales_b):
        # 分块加载 A 和 B
        # 累加器清零
        T.clear(C_local)

        for k in T.Pipelined(K_iters):
            # 加载 FP8 块
            T.copy(A[...], A_shared)
            T.copy(B[...], B_shared)

            # 矩阵乘法
            T.gemm(A_shared, B_shared, C_local)

            # 应用缩放因子
            Scale_C = scales_a[i, k] * scales_b[j, k]
            C_local_accum += C_local * Scale_C

        # 写回结果
        T.copy(C_local_accum, C)
```

### 对应关系

| 论文公式 | 代码实现 | 行号 |
|---------|---------|------|
| $s = \max(\|x\|) / \text{FP8\_MAX}$ | `amax * fp8_max_inv` | kernel.py:83 |
| $\text{clamp}(x/s, -448, 448)$ | `T.clamp(x / s, fp8_min, fp8_max)` | kernel.py:94 |
| $2^{\lceil \log_2(s) \rceil}$ | `fast_round_scale(amax, fp8_max_inv)` | kernel.py:37 |
| $C += (A \cdot s_A) @ (B \cdot s_B)^T$ | kernel.py:248-249 |

---

## 8. 采样策略

### 论文公式

**Gumbel-Max 采样** (等价于多项式采样，但更快):

$$\text{sample} = \argmax_i \left(\frac{p_i}{G_i}\right)$$

其中 $G_i \sim \text{Exp}(1)$ 是独立的指数分布随机变量。

**温度缩放**:

$$p_i = \text{softmax}\left(\frac{\text{logits}_i}{\text{temperature}}\right)$$

### 代码实现

**文件**: `generate.py:22-27`

```python
def sample(logits, temperature=1.0):
    """Gumbel-max trick"""
    logits = logits / max(temperature, 1e-5)  # 温度缩放
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)  # softmax
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)  # Gumbel-max
```

**贪心解码** (temperature=0):

```python
if temperature > 0:
    next_token = sample(logits, temperature)
else:
    next_token = logits.argmax(dim=-1)  # 贪心选择
```

### 对应关系

| 论文公式 | 代码实现 | 行号 |
|---------|---------|------|
| $\text{logits} / T$ | `logits / max(temperature, 1e-5)` | 25 |
| $p = \text{softmax}(\cdot)$ | `torch.softmax(logits, ...)` | 26 |
| $\argmax(p_i / G_i)$ | `probs.div_(...exponential_(1)).argmax(...)` | 27 |

---

## 9. 稀疏注意力 (Sparse Attention)

### 论文公式

稀疏注意力只计算 Top-K 位置的注意力:

$$\text{indices} = \text{TopK}(\text{score}, k)$$

$$o_h = \frac{\sum_{i \in \text{indices}} \exp(q_h \cdot k_i / \sqrt{d}) \cdot v_i + \exp(\text{sink}_h) \cdot 0}{\sum_{i \in \text{indices}} \exp(q_h \cdot k_i / \sqrt{d}) + \exp(\text{sink}_h)}$$

使用在线 Softmax (FlashAttention 风格) 避免显式存储注意力矩阵。

### 代码实现

**文件**: `kernel.py:277-368`

```python
@tilelang.jit
def sparse_attn_kernel(h, d, scale):
    """稀疏注意力内核"""

    @T.prim_func
    def kernel_(q, kv, o, attn_sink, topk_idxs):
        # 初始化
        T.clear(acc_o)
        T.clear(sum_exp)
        T.fill(scores_max, -infinity)

        for t in T.Pipelined(num_blocks):
            # 1. 根据索引收集 KV
            for i in T.Parallel(block):
                idxs[i] = topk_idxs[by, bx, t * block + i]
            kv_shared = gather(kv, idxs)

            # 2. 计算注意力分数
            T.gemm(q_shared, kv_shared, acc_s)  # q @ k^T
            acc_s *= scale  # / sqrt(d)

            # 3. 在线 Softmax (数值稳定)
            scores_max_prev = scores_max
            scores_max = max(acc_s, dim=1)
            scores_scale = exp(scores_max_prev - scores_max)
            acc_s = exp(acc_s - scores_max)
            scores_sum = sum(acc_s, dim=1)
            sum_exp = sum_exp * scores_scale + scores_sum

            # 4. 累加输出
            acc_o *= scores_scale
            acc_o += acc_s @ kv_shared  # 注意力加权

        # 5. 添加 sink 并归一化
        sum_exp += exp(attn_sink - scores_max)
        acc_o /= sum_exp
```

### 对应关系

| 论文公式 | 代码实现 | 行号 |
|---------|---------|------|
| $\text{gather}(kv, \text{indices})$ | `kv_shared[i, j] = kv[by, idxs[i], j]` | kernel.py:325 |
| $q \cdot k^T / \sqrt{d}$ | `T.gemm(q, k) * scale` | kernel.py:328-330 |
| 在线 softmax | kernel.py:331-339 |
| $+ \exp(\text{sink})$ | `sum_exp += exp(attn_sink - scores_max)` | kernel.py:346 |

---

## 总结

DeepSeek V4 Flash 的推理代码实现了论文中的所有核心组件:

1. **RMSNorm**: 简单高效的归一化，减少计算开销
2. **RoPE + YaRN**: 旋转位置编码，支持长序列外推
3. **MLA**: 低秩KV压缩，大幅减少KV Cache内存
4. **KV Cache 压缩**: 学习型门控池化，进一步压缩历史信息
5. **MoE**: 稀疏激活专家，扩展模型容量
6. **Hyper-Connections**: 多状态混合，改善梯度流
7. **FP8/FP4 量化**: 极致压缩，降低显存和计算需求
8. **Gumbel 采样**: 高效的随机采样策略

代码通过 TileLang 框架实现了高性能 GPU 内核，在保持数学等价性的同时，充分利用了现代 GPU 的并行计算能力。
