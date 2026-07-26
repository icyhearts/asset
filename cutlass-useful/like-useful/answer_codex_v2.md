# `gemm-multi-stage-like.cu` 中 `gemm_multi_stage` 的工作原理

本文只分析 `gemm_multi_stage` 的有效计算路径，忽略 kernel 内所有 `print/printf` 及其控制变量。源码行号以当前 `/share/users/like/package/cute-gemm/gemm-multi-stage-like.cu` 为准，运行数据来自 `/share/users/like/package/cute-gemm/run.gemm-multi-stage-like.log`。

## 1. 先看结论

这个 kernel 计算的不是“内存中存成 `K x N` 的 B”，而是：

```text
A: (M, K)，row-major
B: (N, K)，row-major
D: (M, N)，row-major

D(i, j) = sum_p A(i, p) * B(j, p)
        = A * B^T
```

每个 CTA 负责一个 `128 x 128` 的 D tile，沿 K 方向每次处理 32 个元素。A/B 的下一个 K tile 由 `cp.async` 从 global memory 异步送入三阶段 shared-memory 环形缓冲；当前 shared tile 再通过 `ldmatrix` 装入寄存器，最后用 Ampere 的 `mma.sync.m16n8k16` Tensor Core 指令累加。主循环结束后，累加器先写入复用的 shared-memory scratch，再以 128-bit 向量写回 D。

```text
global A/B
    |
    | cp.async，16 B/指令
    v
三阶段 sA/sB 环形缓冲
    |
    | ldmatrix.x4
    v
每线程 A/B 寄存器 fragment
    |
    | mma.sync.m16n8k16
    v
每线程 D 累加器 fragment
    |
    | 32-bit R2S
    v
复用一个 sA stage 作为 sC scratch
    |
    | 128-bit S2G
    v
global D
```

它之所以叫 multi-stage，核心是 global-to-shared 的多阶段软件流水，而不只是把矩阵分块。

## 2. 本次运行的实例参数

`main` 在源码第 497 行实例化：

```cpp
config::GemmConfig<T, 128, 128, 32, 3>
```

日志和配置合起来得到：

| 项目 | 本次取值 | 含义 |
|---|---:|---|
| `T` | `cute::half_t` | A、B、D 都是 fp16 |
| `M, N, K` | `81920, 256, 256` | 问题规模 |
| `kTileM, kTileN, kTileK` | `128, 128, 32` | 一个 CTA 的逻辑 GEMM tile |
| `kStage` | `3` | A/B shared-memory 环形缓冲级数 |
| `block` | `(128,1,1)` | 4 个 warp |
| `grid` | `(2,640,1)` | x 切 N，y 切 M，共 1280 个 CTA |
| `ntile` | `256 / 32 = 8` | 每个 CTA 的 K 主循环次数 |
| dynamic shared memory | `49152 B` | 三份 A tile 加三份 B tile |
| MMA atom | `16 x 8 x 16` | 单条 warp-level Tensor Core 指令 |

本次每个 CTA 最终写一个 `128 x 128` 输出块。`grid.x=2` 覆盖 N 的两个 tile，`grid.y=640` 覆盖 M 的 640 个 tile。

## 3. Config 决定了哪些硬件路径

虽然 kernel 本身写成 `template <typename Config>`，本次运行的关键类型都来自 `GemmConfig`（源码第 322--421 行）。

### 3.1 Tensor Core MMA

配置使用：

```cpp
using mma_op = SM80_16x8x16_F16F16F16F16_TN;
```

它封装的是：

```ptx
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
```

单条指令由一个完整 warp 协作，计算一个逻辑 `16 x 8 x 16` 的矩阵乘加。四个 `f16` 表明 D、A、B、C fragment 都是 fp16；因此这里不是常见的 fp16 输入、fp32 累加，而是 fp16 累加。

`MMA_EU_RepeatT` 为 `(2,2,1)`，所以 4 个 warp 在 M/N 两个方向按 `2 x 2` 排列。日志中的：

```text
ThrLayoutVMNK:  (_32,_2,_2,_1):(_1,_32,_64,_0)
PermutationMNK: (_32,_32,_16)
```

表示线程号可以分解为 32 个 lane、2 个 M warp 位置、2 个 N warp 位置和 1 个 K 位置；这个 `TiledMMA` 的基础逻辑 tile 是 `32 x 32 x 16`。kernel 再通过 fragment 的外层 M/N mode 重复它，覆盖 CTA 的 `128 x 128` 输出 tile。

### 3.2 Global-to-shared copy

配置使用：

```cpp
SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>
```

因此一次 copy atom 是一条 16-byte `cp.async.cg.shared.global`。对 fp16 来说，每条指令搬 8 个连续元素。日志中的 `G2SCopyA` 为：

```text
Tiler_MN:       (_32,_32)
TiledLayout_TV: ((_4,_32),_8):((_256,_1),_32)
```

其线程映射可以直观写成：

```text
tid = 4 * row + k_group
row     = tid / 4       // 0..31
k_group = tid % 4       // 0..3

该线程搬运 K 坐标 [8*k_group, 8*k_group+7]
```

所以 128 个线程的一份基础 copy 覆盖 `32 x 32`。CTA 的 A tile 是 `128 x 32`，CuTe 的 `partition_S/partition_D` 会在 M 外层再得到 4 份基础 copy；每线程对 A 发 4 条 16-byte copy，对 B 也发 4 条。一个 K stage 总共搬：

```text
A: 128 * 32 * 2 B = 8192 B
B: 128 * 32 * 2 B = 8192 B
合计                  = 16384 B
```

### 3.3 Shared-to-register copy

配置使用：

```cpp
SM75_U32x4_LDSM_N
```

它对应非转置的 `ldmatrix ... x4` 路径。`make_tiled_copy_A/B` 不是随意构造另一个线程布局，而是根据 `TiledMMA` 所要求的 A/B lane-value 布局生成 copy，使 shared-memory 数据装入寄存器后正好能直接作为 MMA fragment。

### 3.4 Epilogue copy

寄存器到 shared 使用 `UniversalCopy<int>`，即一次搬 32 bit，也就是 2 个 half；shared 到 global 使用 `UniversalCopy<uint128_t>`，即一次写 8 个连续 half。这样既能适配 Tensor Core 累加器的 lane 分布，又能让最终 global store 保持 16-byte 向量化。

## 4. 建立 global tensor 与 CTA tile

源码第 50--56 行把三个裸指针包装成 CuTe Tensor：

```cpp
A: shape (m,k), stride (k,1)
B: shape (n,k), stride (k,1)
D: shape (m,n), stride (n,1)
```

三者都是最后一维连续的 row-major tensor。注意 B 的逻辑 shape 是 `(N,K)`，所以最终公式是 A 的一行与 B 的一行做点积，也就是 `A * B^T`。

源码第 59--64 行再按 block 坐标切片：

```cpp
gA = A 在 M 方向的第 blockIdx.y 个 128-row tile，K 方向保留全部 tile
gB = B 在 N 方向的第 blockIdx.x 个 128-row tile，K 方向保留全部 tile
gD = D 的 (blockIdx.y, blockIdx.x) 号 128 x 128 tile
```

本次日志给出的实际 shape/stride 是：

```text
gA: (_128,_32,8):(256,_1,_32)
gB: (_128,_32,8):(256,_1,_32)
gD: (_128,_128):(256,_1)
```

`gA/gB` 的第三个 mode 大小为 8，对应 8 个 K tile。其第三维 stride 为 32，所以 tile `t` 的起始 K 坐标是 `32*t`。

这里的 `local_tile` 和后续各种 `partition_*` 主要创建 tensor view，不立即搬运数据。真正的数据移动只发生在后面的 `cute::copy`。

## 5. Shared memory 的三阶段布局

源码第 40--43、97--101 行把一块 dynamic shared memory 分成连续的 A、B 两部分：

```text
shm_data
  +-- sA: (128,32,3)
  +-- sB: (128,32,3)
```

日志中的 layout 是：

```text
sA/sB: Sw<3,3,3> o _0 o (_128,_32,_3):(_32,_1,_4096)
```

忽略 swizzle 时，一个 stage 是 row-major `128 x 32`，stage stride 为 4096 个 half。`Swizzle<3,3,3>` 只重排物理 shared-memory 地址，不改变 `(row,k,stage)` 的逻辑坐标。

对一个 stage 内的线性下标 `p = 32*row + col`，其核心地址变换可写为：

```text
p_swizzled = p ^ ((p & 0x1c0) >> 3)
```

低 3 bit 不变，因此一个 8-half/16-byte 向量内部仍然连续；较高地址 bit 被 XOR 到 bank-select 相关 bit，用来降低 `ldmatrix` 访问 shared memory 时的 bank conflict。

shared-memory 用量为：

```text
sA = 128 * 32 * 3 * 2 B = 24576 B
sB = 128 * 32 * 3 * 2 B = 24576 B
总计                         = 49152 B
```

这与日志的 `shm = 49152` 完全一致。

## 6. 创建每线程 MMA fragment

源码第 114--140 行先取得当前线程的 MMA slice：

```cpp
auto thr_mma = tiled_mma.get_slice(threadIdx.x);
```

然后创建 A、B、D 的每线程寄存器 fragment：

```cpp
tCrA = thr_mma.partition_fragment_A(gA(_,_,0));
tCrB = thr_mma.partition_fragment_B(gB(_,_,0));
tCrD = thr_mma.partition_fragment_C(gD);
clear(tCrD);
```

这里有两个容易混淆的点：

1. `partition_fragment_A/B/C` 根据传入 tensor 的 shape/layout 推导当前线程应持有的坐标，并创建寄存器存储；它不会在这里读取 gA、gB 或 gD。
2. `tCrD` 在寄存器中被清零，因此 kernel 实现的是覆盖式 `D=A*B^T`，不是把原 D 作为 C 再累加。

日志给出的 fragment shape 是：

```text
tCrA: ((_2,_2,_2),_4,_2)   // (MMA value, MMA_M, MMA_K)
tCrB: ((_2,_2),(_2,_4),_2) // (MMA value, MMA_N, MMA_K)
tCrD: ((_2,_2),_4,_8)      // (MMA value, MMA_M, MMA_N)
```

最重要的是：

```cpp
int nk = size<2>(tCrA); // 2
```

一个 CTA K tile 是 32，而一条 MMA atom 的 K 是 16，所以每个 shared stage 被分成两个寄存器/MMA K 子片：

```text
ik=0: 当前 K tile 的 [0,15]
ik=1: 当前 K tile 的 [16,31]
```

`tCrD` 每线程有 `4*4*8=128` 个 half 累加器元素；128 个线程合起来恰好覆盖 `128*128=16384` 个输出元素。

## 7. Copy partition 只是把同一数据换成适合 copy 的视图

源码第 142--188 行分别为 shared-to-register 和 global-to-shared 创建线程视图。

### 7.1 Shared-to-register 视图

以 A 为例：

```cpp
auto tAsA      = s2r_thr_copy_a.partition_S(sA);
auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);
```

`tAsA` 是当前线程在 shared tensor 中应读取的位置；`tCrA_view` 仍然引用 `tCrA` 的同一组寄存器，只是把它重新解释成与 `ldmatrix` copy 相匹配的 mode。日志中二者为：

```text
tAsA:      ((_8,_1),_4,_2,_3)
tCrA_view: ((_8,_1),_4,_2)
```

最后一个 `_3` 是 shared stage；倒数第二个 `_2` 是本例的两个 `K=16` 子片。B 的结构相同。

### 7.2 Global-to-shared 视图

以 A 为例：

```cpp
tAgA_copy = g2s_thr_copy_a.partition_S(gA);
tAsA_copy = g2s_thr_copy_a.partition_D(sA);
```

日志中：

```text
tAgA_copy: ((_8,_1),_4,_1,8)
tAsA_copy: ((_8,_1),_4,_1,_3)
```

可以把 mode 读成：

```text
8 个连续 half / 指令
x 4 个 32-row copy tile
x 1 个 K copy tile
x 8 个 global K tile（源）或 3 个 shared stage（目标）
```

这些 view 把复杂的线程和地址映射提前编码进 tensor layout，因此主循环里只需要用 mode 下标选择“哪个 K tile”和“哪个 shared stage”。

## 8. 三阶段流水的三个游标

源码第 212--214 行初始化：

| 变量 | 初值 | 含义 |
|---|---:|---|
| `itile_to_read` | 0 | 下一个尚未从 global 发起读取的 K tile |
| `ismem_read` | 0 | 当前/下一次 S2R 要读取的 shared stage |
| `ismem_write` | 0 | 下一次 G2S 要写入的 shared stage |

三者不是同一个概念：`itile_to_read` 在 0 到 `ntile` 间单调增长，而两个 shared 下标都按 `% kStage` 环形轮转。

## 9. Prologue：先预取 `kStage-1` 个 tile

源码第 216--228 行预取两个 K tile：

```text
tile 0 -> shared stage 0 -> commit group 0
tile 1 -> shared stage 1 -> commit group 1
```

之后：

```text
itile_to_read = 2
ismem_write   = 2
ismem_read    = 0
```

每次 A/B 的全部 `cp.async` 发出后，`cp_async_fence()` 对应 `cp.async.commit_group`，把这一批指令提交为一个 async group。

接着源码第 230--232 行执行：

```cpp
cp_async_wait<kStage - 2>(); // 本例是 wait_group 1
__syncthreads();
```

`wait_group 1` 的语义不是“等待 group 1”，也不是“全部等待完成”，而是等待到未完成的旧 group 最多剩 1 个。prologue 已提交两个 group，所以它保证最老的 tile 0 已可读取，同时允许 tile 1 仍在传输。

`cp_async_wait` 只约束当前线程发出的异步 copy；`__syncthreads()` 再保证整个 CTA 的线程都完成这一阶段并使跨线程 shared 数据安全可见。两者缺一不可。

等待后，源码第 234--237 行只把 tile 0 的第一个 `K=16` 子片从 stage 0 装入 `tCrA/B[...,ik=0]`，作为主循环的寄存器预热。

## 10. Mainloop：G2S、S2R、MMA 三条流水交错

主循环位于源码第 239--278 行。本次 `ntile=8`、`nk=2`。对每个外层 K tile，两个内层迭代的实际行为如下。

### 10.1 `ik=0`

执行顺序是：

1. 用 `ldmatrix` 把当前 shared stage 的 `ik_next=1` 装入另一组 A/B 寄存器。
2. 若还有未来 tile，则把它异步提交到 `ismem_write` 指向的空闲 stage。
3. `cp_async_fence()` 提交这一组异步 copy。
4. 用此前已经准备好的 `ik=0` 寄存器执行 MMA。

因此执行当前 tile 前半段 MMA 时，当前 tile 后半段已经在寄存器里，未来 tile 则在 global-to-shared 路上。

### 10.2 `ik=1`

执行顺序是：

1. `cp_async_wait<1>()`，确保流水中下一块需要读取的 shared stage 已完成。
2. `__syncthreads()`，等待 CTA 内全部 producer。
3. `ismem_read = (ismem_read + 1) % 3`，切到下一 stage。
4. 把下一 K tile 的 `ik_next=0` 预装到刚刚用完的寄存器槽 0。
5. 用仍保存在寄存器槽 1 中的当前 tile 后半段执行 MMA。

这里的关键是先装 `ik_next`，再计算 `ik`。二者引用不同寄存器槽，所以不会覆盖当前 MMA 的输入。这在 shared-memory 三缓冲之外，又形成了一个以 `ik` 为槽位的寄存器级循环预取。

概念伪代码如下：

```cpp
prefetch_g2s(tile0, stage0);
prefetch_g2s(tile1, stage1);
wait_until(stage0_ready);
reg[0] = load_s2r(stage0, k_half0);

for (tile = 0; tile < ntile; ++tile) {
  // ik = 0
  reg[1] = load_s2r(current_stage, k_half1);
  prefetch_g2s(tile + 2, next_free_stage);
  accum += mma(reg[0]);

  // ik = 1
  wait_until(next_stage_ready);
  advance_read_stage();
  reg[0] = load_s2r(next_stage, k_half0);
  accum += mma(reg[1]);
}
```

伪代码中的 `tile+2` 是本次三阶段、两个 prologue tile 下的直观距离；真实源码通过 `itile_to_read` 表达，因此不把这个距离硬编码进索引。

### 10.3 本次 8 个 K tile 的 stage 时间线

每个 tile 的 stage 是环形的：

| K tile | 全局 K 范围 | shared stage | 何时提交 |
|---:|---:|---:|---|
| 0 | 0--31 | 0 | prologue |
| 1 | 32--63 | 1 | prologue |
| 2 | 64--95 | 2 | 计算 tile 0 的 `ik=0` 时 |
| 3 | 96--127 | 0 | 计算 tile 1 的 `ik=0` 时 |
| 4 | 128--159 | 1 | 计算 tile 2 的 `ik=0` 时 |
| 5 | 160--191 | 2 | 计算 tile 3 的 `ik=0` 时 |
| 6 | 192--223 | 0 | 计算 tile 4 的 `ik=0` 时 |
| 7 | 224--255 | 1 | 计算 tile 5 的 `ik=0` 时 |

stage 0 在 tile 0 的两个子片都已装入寄存器后才会被 tile 3 覆盖；其余 stage 同理。这就是环形缓冲不发生读写冲突的原因。

源码把 `cp_async_fence()` 放在 `itile_to_read < ntile` 判断之外。PTX 在没有新 `cp.async` 时会提交一个空 group；这让尾部仍保持相同的 commit/wait 节拍，使之前最后一个真实 copy group 能在读取前被 `wait_group 1` 推进完成。

最后一个外层迭代的 `ik=1` 仍会按统一路径从“下一 stage”预装一个 `ik=0` fragment，但该 fragment 后续不再参与 MMA；它只是为了避免给主循环增加尾部分支。

## 11. `cute::gemm` 最终做了什么

源码第 276 行：

```cpp
cute::gemm(tiled_mma,
           tCrD,
           tCrA(_,_,ik),
           tCrB(_,_,ik),
           tCrD);
```

语义是寄存器内的：

```text
tCrD = tCrA(ik) * tCrB(ik) + tCrD
```

CuTe 根据 fragment 的层级 shape 展开 M/N repeat，最终发出一组 `mma.sync.m16n8k16`。外层遍历 8 个 `K=32` tile，内层遍历 2 个 `K=16` fragment，因此完整累加 K 的 256 个元素。

由于 `tCrD` 初始清零，最终得到：

```text
D_tile(m,n) = sum_{p=0}^{255} A_tile(m,p) * B_tile(n,p)
```

kernel 中没有 alpha、beta、bias 或 activation。`GemmConfig` 虽然声明了 `ComputeType` 模板参数，但该类型没有参与这里的 MMA 类型选择；实际累加类型由 `SM80_16x8x16_F16F16F16F16_TN` 决定，仍为 fp16。

## 12. Epilogue：为什么不直接从寄存器写 global

主循环结束时，D 的元素按 Tensor Core 要求分散在不同 warp/lane 的寄存器中。这个布局适合 MMA，却不适合让相邻线程直接做连续 16-byte global store。因此源码第 280--319 行使用 shared memory 做一次布局重排：

```text
tCrD registers -> sC -> gD
```

### 12.1 复用已经释放的 A stage

源码第 282 行：

```cpp
auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});
```

主循环已经不再需要 A/B tile，所以直接拿一个空闲 A stage 的地址建立 `sC`，不额外申请 shared memory。

`SmemLayoutC` 是：

```text
Sw<2,3,3> o (_32,_32,_2):(_32,_1,_1024)
```

也就是两个 `32 x 32` 的 fp16 scratch plane：

```text
32 * 32 * 2 * 2 B = 4096 B
```

它小于一个 A stage 的 8192 B，所以配置中的 `static_assert` 可以通过。整个 kernel 的 dynamic shared-memory 大小取 `max(A+B pipeline, C scratch)`，仍是 49152 B。

### 12.2 寄存器到 shared

```cpp
tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);
tCsC_r2s = r2s_thr_copy_c.partition_D(sC);
```

`retile_S` 把同一批 D 累加器寄存器改看成适合 R2S copy 的布局。循环内先构造：

```cpp
auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
```

再把 accumulator copy 到 `t`。这个临时 fragment 的意义是执行“累加类型到输出类型 T”的转换；本次两者都是 half，但代码结构允许两者不同时在这里转换。

随后用 32-bit copy 把每线程 fragment 写入 `sC` 的第 `j` 个 plane。

### 12.3 Shared 到 global

`group_modes<1,3>` 把输出的 M/N 外层 tile mode 展平成一个 mode，便于线性循环。`make_tiled_copy_C` 的基础 tile 是 `32 x 32`；一个 `128 x 128` CTA 输出共有 `4 x 4 = 16` 个这种子 tile。`step=size<3>(tCsC_r2s)=2`，所以 scratch 每轮容纳两个子 tile，共进行 8 轮。

每轮的同步顺序是：

```text
两个子 tile: registers -> sC
__syncthreads()
两个子 tile: sC -> global D（每次 16-byte vector store）
__syncthreads()
复用 sC 处理下一对
```

第一个 barrier 保证所有 shared 写完成后才有线程读取；第二个 barrier 保证所有 global-store 的 shared 读取完成后才覆盖 scratch。

## 13. 为什么这套流水能隐藏延迟

稳态时，一个 CTA 同时存在三类工作：

```text
1. Tensor Core 正在计算当前寄存器 fragment；
2. ldmatrix 已把相邻 K fragment 从 shared 装入另一寄存器槽；
3. cp.async 正把更远的 K tile 从 global 搬入另一个 shared stage。
```

三阶段 shared buffer 让 producer 不必等 consumer 完全结束才开始下一次 global load；两个 `K=16` 寄存器槽又允许 S2R 和相邻 MMA 交错。`wait_group 1` 只等待即将消费的最老 stage，而不是每次 `wait_all`，因此尽量保留尚不急用的异步传输。

代价是 shared-memory 占用达到 48 KiB，并且每线程持有较大的 A/B/D fragment。实际 occupancy 还取决于编译后的寄存器数和目标 GPU；日志没有给出寄存器占用或性能计时，不能仅凭正确性输出判断吞吐率。

## 14. 这份实现的尺寸与功能限制

这不是带完整边界处理的通用 GEMM kernel。当前代码有以下硬前提。

### 14.1 M、N 必须整除 CTA tile

launch 的 grid 使用向上取整，但 global load/store 没有 predicate，也没有 zero-fill copy：

```text
M % 128 == 0
N % 128 == 0
```

否则最后一行/列 CTA 会越界访问。仅仅把 grid 写成 ceil-div 并不等于 kernel 已支持尾块。

### 14.2 K 必须整除 32

```cpp
int ntile = k / kTileK;
```

使用整数除法，且 copy 没有 K-tail predicate。因此需要：

```text
K % 32 == 0
```

否则 K 尾部被忽略，或者先行 copy 与主循环范围不一致。

### 14.3 K tile 数至少为 `kStage-1`

prologue 无条件预取 `kStage-1` 个 tile。本例三阶段会无条件读取 tile 0 和 tile 1，所以还需要：

```text
ntile >= 2
K >= 64
```

否则 prologue 本身就会越界。

### 14.4 对齐要求

G2S 和 S2G 使用 16-byte copy atom，因而裸指针及每行/每 tile 起始地址必须满足相应对齐要求。本次 `cudaMalloc` 返回的基址以及 32/128 的 tile 粒度满足要求，但任意子指针或任意 leading dimension 不一定满足。

### 14.5 固定算子语义

当前 kernel 只实现：

```text
D = A * B^T
```

它不读取旧 D，不实现 `alpha*A*B + beta*C`，也没有 bias、激活、split-K 或 batch 维。

### 14.6 架构要求

`cp.async` 和这里的 MMA atom 是 Ampere SM80 路径；仓库 Makefile 使用 `-arch=sm_86`，因此本次构建目标与这些指令兼容。

## 15. 与运行日志的对应及结论

日志最终给出：

```text
block = (128, 1), gird = (2, 640), shm = 49152
err = 0, str = no error
check ok, max_error = 0.000000
check ok, max_error = 0.000000
M = 81920, N = 256, K = 256
```

两个 `check ok` 分别说明本实现与 cuBLAS、cuBLASLt 的输出比较通过，且这次输入上的最大误差均为 0。这个结果验证了当前 `81920 x 256 x 256`、整 tile、正确对齐场景下的功能正确性；它不消除上一节列出的尾块限制，也不提供性能结论。

## 16. 按源码区间快速定位

| 源码区间 | 作用 |
|---|---|
| 15--43 | 读取 Config，切分 dynamic shared memory |
| 50--64 | 建立 A/B/D global tensor 与 CTA tile |
| 97--101 | 建立三阶段 sA/sB tensor |
| 112--140 | 建立每线程 MMA fragment 并清零累加器 |
| 142--188 | 建立 S2R 和 G2S 的 per-thread copy view |
| 212--237 | 初始化游标、prologue 预取、首个寄存器预取 |
| 239--278 | 三阶段 K mainloop 与 Tensor Core MMA |
| 280--319 | shared-memory epilogue 与 D 写回 |
| 326--420 | 本 kernel 所用 tile、copy、MMA、shared layout 配置 |
| 497、705--732 | 本次 Config 实例、grid/block/shared-memory launch |

一句话概括整个 kernel：它让 4 个 warp 共同计算一个 `128 x 128` 输出 tile，用三份 swizzled shared-memory A/B tile 和两个寄存器 K 子片，把“未来 tile 的 `cp.async`、下一 fragment 的 `ldmatrix`、当前 fragment 的 `mma.sync`”交错起来，最后复用一份 A shared stage 完成适合 128-bit global store 的输出重排。

## 17. 手动展开本次 `gemm_multi_stage` 主循环

本节只代入本次运行的三个数值：

```text
kStage = 3
ntile  = K / kTileK = 256 / 32 = 8
nk     = size<2>(tCrA) = 2
```

因此外层 `itile=0..7`、内层 `ik=0..1`，主循环共有 `8 * 2 = 16` 次内层迭代。

### 17.1 分片符号与 prologue 后的状态

下表用 `tile t` 表示第 `t` 个 K tile。它覆盖 A/B 的同一段 K 下标：

```text
tile t / fragment 0: K[32t,      32t + 15]
tile t / fragment 1: K[32t + 16, 32t + 31]
tile t 整体:          K[32t,      32t + 31]
```

A 和 B 的 G2S、S2R 始终使用相同的 tile、stage 和 fragment 下标。为让 16 行时间线保持可读，表中使用下面两个缩写；它们同时代表 A、B 两条 copy：

```text
S2R(f, s -> r):
  tAsA(_,_,f,s) -> tCrA_view(_,_,r)
  tBsB(_,_,f,s) -> tCrB_view(_,_,r)

G2S(t -> s):
  tAgA_copy(_,_,_,t) -> tAsA_copy(_,_,_,s)
  tBgB_copy(_,_,_,t) -> tBsB_copy(_,_,_,s)
```

这里 `f=ik_next`，目标寄存器槽 `r=ik_next`。prologue 已经把 tile 0 提交到 shared stage 0、把 tile 1 提交到 stage 1，并执行 `cp_async_wait<1>()` 与 `__syncthreads()`，使 tile 0 可读。进入主循环前的精确状态是：

```text
(itile_to_read, ismem_read, ismem_write) = (2, 0, 2)
register slot 0 = tile 0 / fragment 0   // S2R(0, stage 0 -> slot 0)
```

### 17.2 十六次内层迭代的精确时间线

“写游标”一列记录 `(itile_to_read, ismem_write)`；`ismem_read` 则在“末片判断”列单独记录。新增的“最大已就绪 tile id”列取 S2R copy 发起前、由 `cp_async_wait<1>()` 明确保证已经就绪的 shared-memory tile 上界；硬件可能提前完成但尚未被 wait 保证的更新 group 不计入。每一行的操作顺序与源码一致：必要时先 wait/sync 并轮转 read stage，然后 S2R，再按条件 G2S 和 fence，最后执行 GEMM。

| # | `(itile,ik)`；`ik_next` | `ik==nk-1`；`ismem_read` 前后 | S2R 前最大已就绪 tile id | S2R 源 -> 目标；实际数据 | `ik==0`；`itile_to_read<ntile` | G2S / fence；写游标前后 | `cute::gemm` 实际消费 |
|---:|---|---|---:|---|---|---|---|
| 1 | `(0,0)`；1 | 假；`0 -> 0` | `0` | `S2R(1,0 -> 1)`；tile 0/f1 | 真；`2<8` 真 | `G2S(2 -> 2)`；有数据 group；`(2,2) -> (3,0)` | tile 0/f0 |
| 2 | `(0,1)`；0 | 真；wait/sync；`0 -> 1` | `1` | `S2R(0,1 -> 0)`；tile 1/f0 | 假；内层条件未求值 | 无 G2S，不 fence；`(3,0) -> (3,0)` | tile 0/f1 |
| 3 | `(1,0)`；1 | 假；`1 -> 1` | `1` | `S2R(1,1 -> 1)`；tile 1/f1 | 真；`3<8` 真 | `G2S(3 -> 0)`；有数据 group；`(3,0) -> (4,1)` | tile 1/f0 |
| 4 | `(1,1)`；0 | 真；wait/sync；`1 -> 2` | `2` | `S2R(0,2 -> 0)`；tile 2/f0 | 假；内层条件未求值 | 无 G2S，不 fence；`(4,1) -> (4,1)` | tile 1/f1 |
| 5 | `(2,0)`；1 | 假；`2 -> 2` | `2` | `S2R(1,2 -> 1)`；tile 2/f1 | 真；`4<8` 真 | `G2S(4 -> 1)`；有数据 group；`(4,1) -> (5,2)` | tile 2/f0 |
| 6 | `(2,1)`；0 | 真；wait/sync；`2 -> 0` | `3` | `S2R(0,0 -> 0)`；tile 3/f0 | 假；内层条件未求值 | 无 G2S，不 fence；`(5,2) -> (5,2)` | tile 2/f1 |
| 7 | `(3,0)`；1 | 假；`0 -> 0` | `3` | `S2R(1,0 -> 1)`；tile 3/f1 | 真；`5<8` 真 | `G2S(5 -> 2)`；有数据 group；`(5,2) -> (6,0)` | tile 3/f0 |
| 8 | `(3,1)`；0 | 真；wait/sync；`0 -> 1` | `4` | `S2R(0,1 -> 0)`；tile 4/f0 | 假；内层条件未求值 | 无 G2S，不 fence；`(6,0) -> (6,0)` | tile 3/f1 |
| 9 | `(4,0)`；1 | 假；`1 -> 1` | `4` | `S2R(1,1 -> 1)`；tile 4/f1 | 真；`6<8` 真 | `G2S(6 -> 0)`；有数据 group；`(6,0) -> (7,1)` | tile 4/f0 |
| 10 | `(4,1)`；0 | 真；wait/sync；`1 -> 2` | `5` | `S2R(0,2 -> 0)`；tile 5/f0 | 假；内层条件未求值 | 无 G2S，不 fence；`(7,1) -> (7,1)` | tile 4/f1 |
| 11 | `(5,0)`；1 | 假；`2 -> 2` | `5` | `S2R(1,2 -> 1)`；tile 5/f1 | 真；`7<8` 真 | `G2S(7 -> 1)`；有数据 group；`(7,1) -> (8,2)` | tile 5/f0 |
| 12 | `(5,1)`；0 | 真；wait/sync；`2 -> 0` | `6` | `S2R(0,0 -> 0)`；tile 6/f0 | 假；内层条件未求值 | 无 G2S，不 fence；`(8,2) -> (8,2)` | tile 5/f1 |
| 13 | `(6,0)`；1 | 假；`0 -> 0` | `6` | `S2R(1,0 -> 1)`；tile 6/f1 | 真；`8<8` 假 | 无 G2S；提交空 group；`(8,2) -> (8,2)` | tile 6/f0 |
| 14 | `(6,1)`；0 | 真；wait/sync；`0 -> 1` | `7` | `S2R(0,1 -> 0)`；tile 7/f0 | 假；内层条件未求值 | 无 G2S，不 fence；`(8,2) -> (8,2)` | tile 6/f1 |
| 15 | `(7,0)`；1 | 假；`1 -> 1` | `7` | `S2R(1,1 -> 1)`；tile 7/f1 | 真；`8<8` 假 | 无 G2S；提交空 group；`(8,2) -> (8,2)` | tile 7/f0 |
| 16 | `(7,1)`；0 | 真；wait/sync；`1 -> 2` | `7` | `S2R(0,2 -> 0)`；stage 2 中旧 tile 5/f0，结果不用 | 假；内层条件未求值 | 无 G2S，不 fence；`(8,2) -> (8,2)` | tile 7/f1 |

最后一行的 `S2R(0,2 -> 0)` 不是 global tile 8 的读取。stage 2 最后一次真实写入的是 tile 5，所以这次统一流水路径重新读到旧 tile 5/f0；主循环随即结束，slot 0 不再传给任何 GEMM。

### 17.3 G2S 写游标的独立核对

只看真实 G2S 提交，两个写游标按下列顺序变化：

| 时点 | `(itile_to_read, ismem_write)` |
|---|---|
| prologue 完成后 | `(2,2)` |
| tile 2 -> stage 2 | `(3,0)` |
| tile 3 -> stage 0 | `(4,1)` |
| tile 4 -> stage 1 | `(5,2)` |
| tile 5 -> stage 2 | `(6,0)` |
| tile 6 -> stage 0 | `(7,1)` |
| tile 7 -> stage 1 | `(8,2)` |
| 最后四次内层迭代 | 始终为 `(8,2)` |

prologue 的 G2S 只读取 tile 0、1，主循环的真实 G2S 只读取 tile 2--7。到 `itile_to_read=8` 后，`8<8` 为假，因而没有 tile 8 的 global-memory 越界读取。

### 17.4 为什么槽位覆盖与尾部空操作都正确

- `ik_next=(ik+1)%2`：`ik=0` 时为 1，`ik=1` 时为 0，两个寄存器 fragment 槽循环覆盖。每次覆盖的都是当前 GEMM 不读取的另一个槽。
- 每次 `ik=1` 都执行 `cp_async_wait<1>()`、`__syncthreads()`，再令 `ismem_read=(ismem_read+1)%3`。8 次轮转是 `0 -> 1 -> 2 -> 0 -> 1 -> 2 -> 0 -> 1 -> 2`，最终 `ismem_read=2`。
- 每次 `ik=0` 都执行 `cp_async_fence()`。`itile=0..5` 提交真实 G2S group；`itile=6,7` 没有新 copy，但仍提交空 group，以保持统一的 commit/wait 节拍。
- `cute::gemm` 的源码参数只出现 `ik`，没有 `itile`。外层 tile 身份来自该寄存器槽此前由哪个 shared tile/fragment 装入；因此表中的“实际消费”必须沿寄存器数据流判断，不能只看调用表达式。
- 第 16 次迭代的最后一次 S2R 是统一流水路径产生的无用预取，不参与任何后续 GEMM，也不会改变累加结果。

最终状态为：

```text
ismem_read    = 2
itile_to_read = 8
ismem_write   = 2
```

16 次 `cute::gemm` 恰好消费 `(tile 0/f0, tile 0/f1, ..., tile 7/f0, tile 7/f1)`：8 个 K tile 的两个 fragment 各参与一次，完整且无重复地覆盖 `K[0,255]`。

## 18. 交替调用 `cp.async` 和 `cp.async.wait_group 1` 时的剩余 group 数

以下按一个执行线程的 per-thread async-group 序列分析。题设中的每个“`cp.async` 后立即 `cp.async.commit_group`”会建立一个只含 1 条 `cp.async` 的 group：

```cpp
// 第一批
for (int i = 0; i < 10; ++i) {
  cp.async(...);
  cp.async.commit_group();       // G0, G1, ..., G9
}
cp.async.wait_group 1;

// 第二批
for (int i = 0; i < 10; ++i) {
  cp.async(...);
  cp.async.commit_group();       // G10, G11, ..., G19
}
cp.async.wait_group 1;
```

### 18.1 直接结论

第二次 `cp.async.wait_group 1` 返回时，未完成的 group 数量是：

```text
最多 1 组；实际可能是 0 组或 1 组，不能断言恰好是 1 组。
```

如果把每个 group 中的 `cp.async` 数量也限定为题设中的 1 条，那么“未完成的 `cp.async` 数量”同样是 0 或 1 条。

### 18.2 两次 wait 的状态变化

| 时点 | 已提交的 group | `wait_group 1` 的保证 | wait 返回后允许未完成的 group |
|---|---|---|---|
| 第一次 wait 前 | G0--G9，共 10 组 | 除最多 1 个最近 group 外，其余完成 | G9 最多 1 组 |
| 第一次 wait 后再提交第二批 | G0--G19，共 20 组 | 第二次 wait 不会把 group 计数器“重置”为第二批；它仍针对该线程此前提交的所有 group | 在第二次 wait 返回时，G0--G18 已完成，G19 最多 1 组 |

因此第二次 wait 结束后不是“第一批留 1 组、第二批再留 1 组”，也不是把第一批残留的 1 组加上第二批的 10 组后留下 11 组。完成顺序按提交顺序推进；若确实还有 1 组未完成，只可能是最新的 G19。若 G19 在等待期间也完成，则返回时为 0 组未完成。

`wait_group 1` 的 `1` 是“允许保留的最近 group 数上限”，不是 group 编号，也不是每次调用后固定留下 1 组。需要保证全部历史 copy 完成时，应使用 `cp.async.wait_group 0`（本仓库 `cute::cp_async_wait<0>()` 映射为 `cp.async.wait_all`）。此外，wait 的完成和可见性是 per-thread 的；若其他线程还要读取同一 shared-memory 数据，仍需按算法需要使用 `__syncthreads()` 等 CTA 级同步。

上述语义对应 NVIDIA PTX ISA 对 `cp.async.commit_group` 和 `cp.async.wait_group N` 的定义：`commit_group` 为每个线程建立 group，`wait_group N` 等待到最近的至多 N 个 group 仍可 pending，而更早的 group 全部完成。参见 [NVIDIA PTX ISA：`cp.async.wait_group` / `cp.async.wait_all`](https://docs.nvidia.com/cuda/parallel-thread-execution/#cp-async-wait-group-cp-async-wait-all)。
