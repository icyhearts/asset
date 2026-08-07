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

## 19. 第 13 步的空 group 会不会让第 14 步真的等待

这里对应 17.2 表中的第 13、14 行：

```text
第 13 步：(itile, ik) = (6, 0)，没有真实 G2S，cp_async_fence() 提交空 group
第 14 步：(itile, ik) = (6, 1)，cp_async_wait<1>()，随后 S2R tile 7/f0
```

把主循环中的 group 按提交顺序编号，相关尾部是：

| 步骤 | 新提交的 group | group 内容 | 该步结束时的含义 |
|---|---|---|---|
| 第 11 步 | G7 | 真实的 tile 7 G2S copy | tile 7 可能仍在异步传输 |
| 第 12 步 | 无 | `cp_async_wait<1>()` | 更早的 tile 6 group 已完成；G7 仍可能未完成 |
| 第 13 步 | G8 | 空 group，没有任何 `cp.async` | 空 group 不包含 tile，也不引入需要等待的 copy |
| 第 14 步 | 无 | `cp_async_wait<1>()` | G7 作为 G8 之前的真实 group 必须完成，然后才执行 S2R |

因此，第 14 步的 `cp.async.wait_group 1` **会执行，但不一定发生可观察的阻塞**：

- 如果第 11 步提交的 G7（tile 7）到达第 14 步时仍未完成，wait 会等待 G7 完成。
- 如果 G7 已经自然完成，wait 会立即返回。
- 第 13 步的空 G8 是 trivially complete；它只是形式上成为“最近的至多一个允许 pending 的 group”，并不是一个未完成的 group，也不会替 tile 7 提供就绪保证。

在第 14 步的 wait 返回时，G7 已完成，G8 又没有实际 copy，所以没有未完成的真实 `cp.async` group。随后 `__syncthreads()` 再把各线程对 tile 7 的完成状态汇合起来，S2R 才从 stage 1 读取 tile 7/f0。于是 17.2 表中第 14 行的“ S2R 前最大已就绪 tile id = 7”来源是 **第 11 步的 tile 7 group 被第 14 步 wait 推进完成**，不是第 13 步空 G8 仍处于未完成状态。

第 15 步同理提交另一个空 G9；到第 16 步时，真实的 G7 已在第 14 步处理完，G8/G9 都没有 copy，因此第 16 步的 wait 通常不需要再等待真实数据。空 group 的“trivially complete”性质和 `wait_group 1` 的“最多保留一组”语义并不矛盾：允许保留的是 G8（或 G9）这个无操作 group，但实际 pending 数量仍为 0。

## 20. 逐行解释 280--343：C epilogue 的寄存器、shared 和 global 数据流

本节对应当前 `gemm-multi-stage-like.cu` 的源码第 280--343 行，以及 `run.gemm-multi-stage-like.log` 中从 `auto sC` 到 `step=2` 的打印结果。第 278 行刚结束 K 主循环；此时 `tCrD` 已经包含当前 CTA 的 `128 x 128` 输出 tile 的累加结果，后面的代码不再做 MMA，而是把 Tensor Core 的寄存器布局重排成适合 global store 的布局。

### 20.1 280--282：复用一个 A shared stage 作为 C scratch

```cpp
// use less shared memory as a scratchpad tile to use large wide instuction
// Dreg -> shm -> reg -> global
auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});
```

- `sA` 是三阶段 A shared tensor。主循环已经结束，A/B tile 不再需要，因此 `sA(_,_,ismem_read).data()` 取出当前 read stage 的基地址，把这块已释放的空间借给 epilogue。
- `make_tensor` 没有分配或复制内存；它只是用 `SmemLayoutC{}` 重新解释同一个 shared 指针。当前配置中主循环结束时 `ismem_read=2`，所以复用的是 A 的 stage 2。
- 日志打印为：

  ```text
  Sw<2,3,3> o _0 o (_32,_32,_2):(_32,_1,_1024)
  ```

  其逻辑 shape 是两个 `32 x 32` plane，最后一个 stride `1024=32*32` 用来在两个 plane 间跳转；swizzle `Sw<2,3,3>` 改变 shared 地址排列以配合后续 copy。两个 plane 共 `32*32*2` 个 half，即 `4096 B`，小于一个 A stage 的 `128*32*2=8192 B`，所以复用不会越过该 stage 的边界。
- 注释中的 `Dreg -> shm -> reg -> global` 是硬件数据路径的概括：代码显式完成 `tCrD -> sC` 和 `sC -> gD`，S2G 的向量化 copy 内部会使用线程寄存器承接 load/store；这里没有另一个需要手写的普通 S2R 循环。

### 20.2 284--287：建立寄存器到 shared 的 C copy view

```cpp
auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);
auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);
```

`R2SCopyAtomC` 在 `GemmConfig` 中是 `Copy_Atom<UniversalCopy<int>, T>`：一个 atom 搬运 32 bit，也就是当前 half 输出中的两个元素。`make_tiled_copy_C` 再把这个 atom 和 `tiled_mma` 的 C/M/N 线程布局结合起来，使每个线程拿到与自身累加器对应的 R2S 位置。

`get_slice(idx)` 选择线程 `idx` 的 copy slice；它只生成视图，不移动数据。随后两个 partition 的方向不同：

| 视图 | 作用 | 日志中的形状含义 |
|---|---|---|
| `tCrC_r2s = retile_S(tCrD)` | 把每线程的 MMA accumulator 重新排成 R2S copy 的 source layout | `((_2,(_2,_2)),_4,_4)`：copy value 层级加上 4 个 M 子块和 4 个 N 子块 |
| `tCsC_r2s = partition_D(sC)` | 把 scratch 的目的地址按相同 copy atom 分片 | `((_2,(_2,_2)),_1,_1,_2)`：前几维是 copy value，最后的 `_2` 是两个 scratch pipe |

因此 `retile_S` 不会把 `tCrD` 的元素重排到新内存，而是改变 CuTe 对同一寄存器 fragment 的 mode 解释；`partition_D` 则把 `sC` 的地址和布局投影成每线程可写的目标视图。

### 20.3 289--295：建立 shared 到 global 的向量化 copy view

```cpp
S2GCopyC s2g_tiled_copy_c;
auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);
auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);

auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);
```

`S2GCopyC` 在配置中使用 `UniversalCopy<cute::uint128_t>`，所以一次 S2G atom 是 128 bit，即 16 B；对 half 来说是 8 个连续元素。日志中的 S2G `TiledCopy` 为：

```text
Tiler_MN:       (_32,_32)
TiledLayout_TV: ((_4,_32),_8):((_256,_1),_32)
```

这表示一个基础 copy tile 是 `32 x 32`，128 个线程共同覆盖它，每个线程的 value layout 有 8 个 half。对应的 partition 结果是：

```text
tCsC_s2g: ((_8,_1),_1,_1,_2)       // sC source，最后一维是 pipe
tCgC_s2g: ((_8,_1),_4,_4)           // gD destination，4 x 4 个 32x32 子块
```

`group_modes<1,3>` 把 rank-3 view 的第 1、2 两个 mode（半开区间 `[1,3)`）合并成一个逻辑 M/N mode：

```text
tCgC_s2g  -> tCgC_s2gx: ((_8,_1),(_4,_4))
tCrC_r2s  -> tCrC_r2sx: ((_2,(_2,_2)),(_4,_4))
```

这是 view 级别的展平，不会发生内存访问。合并后第二个 mode 有 `4*4=16` 个输出子 tile；两边使用同一个线性索引 `i+j`，因此寄存器 source 和 global destination 能一一对应。日志中的 global stride `(8192,32)` 也符合一个 `128 x 128` row-major D tile：沿 M 子块移动 32 行要跳 `32*256=8192` 个 half，沿 N 子块移动 32 列只跳 32 个 half。

### 20.4 297--321：确定每轮搬两个 pipe，并打印调试布局

```cpp
int step = size<3>(tCsC_r2s);  // pipe
```

`tCsC_r2s` 的第 3 号 mode 是 scratch pipe，日志给出 `(_2)`，所以：

```text
step = 2
size<1>(tCrC_r2sx) = 16
```

`step` 不是线程数、字节数或 MMA K 片数，而是一次写入 scratch 的 pipe 数。外层一共有 16 个 `32 x 32` 子 tile，故需要 `16/2=8` 个 outer round。

第 298--321 行的 `if (this_thread_print)` 只打印 `sC`、R2S/S2G 的 thread slice、partition 和 grouped view；它不参与数据搬运，也不改变同步或循环边界。运行日志中的 `step=2, size<1>(tCrC_r2sx)=16` 正是这段代码打印的结果。

### 20.5 322--343：两级循环完成 R2S、同步、S2G、同步

```cpp
#pragma unroll
for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
  // registers -> shared
  #pragma unroll
  for (int j = 0; j < step; ++j) {
    auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
    cute::copy(tCrC_r2sx(_, i + j), t);
    cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
  }
  __syncthreads();

  // shared -> global
  for (int j = 0; j < step; ++j) {
    cute::copy(s2g_tiled_copy_c,
               tCsC_s2g(_, 0, 0, j),
               tCgC_s2gx(_, i + j));
  }
  __syncthreads();
}
```

外层 `i` 依次为 `0,2,4,...,14`；内层 `j` 只有 0、1。令 `q=i+j`，每个 q 对应一个 `32 x 32` 输出子 tile（逻辑上是 4 个 M 子块乘 4 个 N 子块）：

```text
q = 0,1,2,3       -> 第一行的 4 个 N 子块
q = 4,5,6,7       -> 第二行的 4 个 N 子块
q = 8,9,10,11     -> 第三行的 4 个 N 子块
q = 12,13,14,15   -> 第四行的 4 个 N 子块
```

每个 outer round 的具体顺序如下：

1. **临时类型转换（329--330）**：`tCrC_r2sx(_,i+j)` 是当前线程的 accumulator fragment；`make_tensor_like<T>` 建立同 shape、元素类型为输出 `T` 的临时 tensor。`cute::copy` 把累加类型转换/复制到 `t`。本次 `SM80_16x8x16_F16F16F16F16_TN` 的 accumulator 也是 half，因此数值类型恰好相同，但这一步保留了通用的 accumulator/output 类型转换路径。
2. **R2S（332）**：`t` 写入 `sC` 的 pipe `j`。所有线程共同写这两个 `32 x 32` scratch plane；`tCsC_r2s(_,0,0,j)` 中的 `_` 是 copy atom 的 value mode，两个 `_1` mode 固定到当前 pipe 对应的单一 M/N scratch 位置。
3. **第一个 barrier（334）**：防止任何线程还没完成 R2S 写入时，其他线程就开始 S2G 读取 `sC`。它是跨线程的 shared-memory producer/consumer 边界。
4. **S2G（339）**：从同一个 pipe `j` 的 `tCsC_s2g` 读取，通过 128-bit atom 写到 `tCgC_s2gx(_,i+j)` 对应的 global D 子 tile。CuTe 的 copy view 已经把线程、value 和 M/N 坐标配好，循环本身只推进子 tile 线性索引。
5. **第二个 barrier（342）**：确保本轮所有线程都完成对当前 scratch pipe 的读取后，外层下一轮才会用 R2S 覆盖这两个 plane。它保护的是 scratch 的复用，不是替代 global memory 的错误检查或 kernel 完成同步。

第 343 行关闭 outer loop；紧随其后的第 344 行关闭 `gemm_multi_stage` kernel。对本次运行，8 轮各处理 2 个子 tile，正好覆盖 CTA 的 16 个 `32 x 32` 子块，也就是完整的 `128 x 128` D tile。运行日志最后给出 `err = 0`、两个 `check ok`，说明这条 epilogue 路径在 `M=81920,N=256,K=256` 的本次测试中写回了正确结果。

## 21. `R2SCopyAtomC` 如何把每线程 8 个 half 搬到 shared

323--343 行中最容易产生误解的是这一句：

```cpp
cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
```

当前线程的 source tensor `t` 的逻辑 size 是 8，元素类型是 half；但配置中的 atom 是：

```cpp
using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
```

`UniversalCopy<int>` 的单次底层 copy 宽度确实只有 32 bit。不过这里的 `cute::copy` 接收的是 `TiledCopy`，不是裸的 `Copy_Atom`。`TiledCopy` 会把一个 32-bit atom 沿 value mode 平铺多次。

### 21.1 atom 的 32 bit 对应两个 half

`UniversalCopy<S,D>` 在 `Copy_Traits` 中把 source/destination 的 bit layout 定义为 `sizeof_bits<S>` 和 `sizeof_bits<D>`。构造：

```cpp
Copy_Atom<UniversalCopy<int>, cute::half_t>
```

时，`Copy_Atom` 用 `recast_layout<uint1_t, half_t>` 把 32 个 bit 重新看成 half value layout。日志因此打印：

```text
ValLayoutSrc: (_1,_2):(_0,_1)
ValLayoutDst: (_1,_2):(_0,_1)
ValueType:    16b
```

这里：

```text
_1       = 1 个线程
_2       = 2 个 half value
2 * 16b  = 32b = 1 个 int copy
```

所以 atom 不是“只能复制一个 half”，而是用一个 32-bit `int` 寄存器搬运两个 half 的 bit pattern。`UniversalCopy<int>::copy` 的抽象动作等价于一次 32-bit 赋值；CuTe 负责把两个 half 组成这个 32-bit copy 的 source/destination view。

### 21.2 R2S `TiledCopy` 把 atom 复制成 8-value/thread

运行日志中 R2S 的完整布局是：

```text
Tiler_MN:       (_32,_32)
TiledLayout_TV: ((_4,_8,_2,_2),((_2,_2),(_1,_2))):
                ((_64,_1,_16,_256),((_32,_8),(_0,_512)))
Copy_Atom
  ValLayoutSrc: (_1,_2):(_0,_1)
  ValLayoutDst: (_1,_2):(_0,_1)
```

`TiledLayout_TV` 的第一个大 mode 是线程布局，第二个大 mode 是 value 布局。只看 value 部分：

```text
Tiled value shape = ((_2,_2),(_1,_2))
size(Tiled value)  = 2 * 2 * 1 * 2 = 8 half/thread

Atom value shape   = (_1,_2)
size(Atom value)    = 1 * 2 = 2 half/atom invocation
```

因此 TiledCopy 的 value mode 可以理解为：

```text
4 个 rest-value 位置 × 每个位置 2 个 atom-value half
= 4 × 2
= 8 个 half/thread
```

`TiledCopy` 的静态约束也正是这个关系：`TiledNumVal % AtomNumVal == 0`。本例为 `8 % 2 == 0`，所以一个线程的 8-value fragment 可以被完整地切成 4 个 atom-sized 子片。

### 21.3 一次高层 `cute::copy` 的实际展开

`cute::copy(r2s_tiled_copy_c, ...)` 通过 `TiledCopy` 继承的 `Copy_Atom` 接口进入 CuTe copy algorithm。对于 atom value 以外的 rest mode，copy algorithm 逐个展开；每次传给 `Copy_Atom::call` 的 rank-1 子 tensor 只有 2 个 half，随后 `copy_unpack` 将这 2 个 half recast 成一个 32-bit `int` source/destination register。

可以把每个线程的一次高层 copy 画成：

```text
t 的 8 个 half（逻辑 value mode）
        |
        +-- rest 0: half[0:2] -> 32-bit atom copy #0
        +-- rest 1: half[2:4] -> 32-bit atom copy #1
        +-- rest 2: half[4:6] -> 32-bit atom copy #2
        +-- rest 3: half[6:8] -> 32-bit atom copy #3
        |
tCsC_r2s 的 8 个 half
```

这里的 `half[0:2]` 是逻辑 value 分组，不表示寄存器物理地址必然连续。日志中的 `tCrC_r2sx` layout 为：

```text
((_2,(_2,_2)),(_4,_4)):((_1,(_2,_16)),(_4,_32))
```

固定 outer 子 tile `i+j` 后，source 的第一 mode `(_2,(_2,_2))` 仍有 8 个元素；其嵌套 stride `(_1,(_2,_16))` 由 `retile_S` 生成，用来把 MMA accumulator 的真实寄存器位置映射到这 4 个 atom 子片。也就是说，CuTe 展开的是“布局中的 4 组、每组 2 个 value”，而不是假设 8 个 half 在内存中连续排列。

### 21.4 字节数和 `32 x 32` tile 的一致性

对一个线程的一次 `j`：

```text
8 half/thread       = 8 * 2 B = 16 B
4 个 int32 atom     = 4 * 4 B = 16 B
```

R2S TiledCopy 使用 128 个线程覆盖一个 `32 x 32` half tile，因此整个高层 `cute::copy` 的数据量是：

```text
128 threads * 8 half/thread
= 1024 half
= 32 * 32 half
= 2048 B
```

这正好是一个 `sC` scratch plane 的大小。`step=2` 时，外层循环先对 `j=0`、`j=1` 分别执行两次这样的 R2S copy，把两个 `32 x 32` plane 填满；barrier 之后再由 S2G 的 128-bit atom 把它们写回 global D。

因此不存在“一个 32-bit atom 强行复制 8 个 half”的情况：真实层次是 **1 个 atom = 2 个 half，1 个线程的 TiledCopy = 4 个 atom = 8 个 half，128 个线程 = 1 个 `32 x 32` 输出子 tile**。`cute::copy` 这一行只是把这四次 atom 操作通过静态 value/thread layout 组合成一个高层调用。

## 22. 这 4 次 atom 是如何执行的：展开、串行与并行

对“是否循环 4 次、是否串行”要区分三个层次。

### 22.1 源码层面：有一个 rest-value 循环

`cute::copy(r2s_tiled_copy_c, src, dst)` 的 `TiledCopy` 继承自 `Copy_Atom`，会匹配 CuTe 的 `copy(Copy_Atom, ...)` overload；该 overload 再转到 `copy_if`。在 `3rd/cutlass/include/cute/algorithm/copy.hpp` 中，核心逻辑等价于：

```cpp
// src/dst 的第一 mode 是一个 atom 的 V=2 个 half
for (int rest = 0; rest < 4; ++rest) {
  copy_atom.call(src_v(_, rest), dst_v(_, rest));
}
```

真实源码用的是 `CUTE_UNROLL`，不是普通的未标注 runtime loop。因为本例的 `TiledCopy` value shape 和 `AtomNumVal` 都是编译期常量，`size<1>(src_v)=4` 会在模板实例化时确定；所以这里可以说“逻辑上循环 4 次”，但最终目标通常是展开为 4 个 atom call site，而不是运行时维护一个 `rest` 计数器。

`Copy_Atom::call` 收到的每个 `src_v(_,rest)` 只有 2 个 half，随后 `copy_unpack` 把这两个 half recast 成一个 32-bit `int` 寄存器值并执行一次 `UniversalCopy<int>`。因此层次关系是：

```text
一个 cute::copy(TiledCopy, ...)
    -> 4 个 rest-value atom call
        -> 每个 call 搬 2 个 half / 32 bit
```

### 22.2 单个线程层面：4 个 32-bit 操作按程序顺序发出

从一个线程的指令流看，4 个 atom 操作不是一条 128-bit 指令，也没有在四个 atom 之间插入 `__syncthreads()`。它们是该线程需要发出的 4 个 32-bit R2S store，逻辑顺序可以写成：

```text
atom #0: 2 half -> 32-bit shared store
atom #1: 2 half -> 32-bit shared store
atom #2: 2 half -> 32-bit shared store
atom #3: 2 half -> 32-bit shared store
```

这里“串行”指同一线程的程序指令需要逐条 issue；不应理解成 4 次高层 `cute::copy` 调用之间有同步，或整个 CTA 先完成 atom #0 才允许 atom #1。编译器可以对独立指令做调度，最终 PTX/SASS 的具体相邻关系应以编译产物为准，但语义上每个 atom 都只负责自己的 2 个 half。

### 22.3 warp/CTA 层面：不同线程的同一 atom 操作是 SIMT 并行的

R2S TiledCopy 的线程布局覆盖 128 个线程。每个线程都执行自己的 4 个 atom 操作；一个 warp 的 32 个线程会以 SIMT 方式共同执行当前 atom 指令，不同 warp 也由 SM 的 warp 调度器交错推进。因此：

```text
单线程：     4 个 32-bit atom 操作
一个 j：     128 线程 × 4 个 atom = 512 个线程级 atom 操作
一个 i：     j=0 和 j=1，各 4 个 atom，即每线程 8 个 atom
整个 C tile：16 个子 tile × 每线程 4 个 atom = 每线程 64 个 atom
```

`__syncthreads()` 只在 334 行、完成当前 `j=0/1` 的全部 R2S 写入后才出现；它保证所有线程都写完 scratch 后再进入 S2G，不参与 4 个 atom 之间的拆分，也不把 4 个 atom 变成 CTA 级串行阶段。

所以对问题的直接回答是：**是的，CuTe 在逻辑上会把一次高层 `cute::copy` 展开成 4 次 atom copy；对单个线程而言这 4 个 32-bit 操作按指令流依次执行，但 128 个线程/各个 warp 的对应操作以 SIMT 方式并行执行。最终不是一个 runtime `for` 循环，也不是 4 次带 barrier 的 R2S 阶段。**

## 23. `ProducerBarType::init(...,1)` 和 `ConsumerBarType::init(...,128)`

### 23.1 `init` 的第二个参数是什么

相关代码位于 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:182-195`（函数 `gemm_device`）：

```cpp
using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;
using ConsumerBarType = cutlass::arch::ClusterBarrier;

ProducerBarType::init(&producer_mbar[pipe],   1);
ConsumerBarType::init(&consumer_mbar[pipe], 128);
```

第二个参数是该 mbarrier 在每个 phase 中的 expected arrival count，即这个 phase 完成前必须收到多少次 arrive-on 操作。`include/cutlass/arch/barrier.h:390-405`（函数 `cutlass::arch::ClusterBarrier::init`）把它直接作为 `arrive_count` 传给：

```ptx
mbarrier.init.shared::cta.b64 [barrier], arrive_count;
```

它不是：

- 执行 `init` 的线程数；两个 barrier 实际都只由一个线程初始化；
- 调用 `wait` 的线程数；wait 不会消耗 arrival count；
- TMA copy 的条数；
- TMA transaction 的字节数。

两个 barrier 的协议可以概括为：

| Barrier | expected arrival count | 谁执行 arrive | 谁等待 | 额外完成条件 |
|---|---:|---|---|---|
| `producer_mbar` | 1 | 一个 elected producer lane | 128 个 consumer 线程 | TMA tx-count 也必须归零 |
| `consumer_mbar` | 128 | CTA 的 128 个 consumer 线程 | 一个 elected producer lane | 无额外 transaction count |

### 23.2 为什么 `ProducerBarType` 的参数是 1

`ProducerBarType` 是 `ClusterTransactionBarrier`，用于表示“A/B 的 TMA load 已经把当前 stage 填满”。

#### 23.2.1 每个 phase 只有一个线程执行软件 arrive

在 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:182-207`（函数 `gemm_device`）中：

```cpp
int warp_idx = cutlass::canonical_warp_idx_sync();
int lane_predicate = cute::elect_one_sync();

if ((warp_idx == 0) && lane_predicate) {
  ProducerBarType::arrive_and_expect_tx(...);
}
```

`elect_one_sync()` 在一个 warp 中只选出一个 lane，再加上 `warp_idx == 0`，最终每个 CTA 只有一个 producer thread 对该 pipe 执行一次 `arrive_and_expect_tx`。因此 expected arrival count 必须是 1。

同样的条件也用于后续 stage 重填，见 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:272-281`（函数 `gemm_device`），所以每次 barrier phase 始终只有一次软件 arrive。

#### 23.2.2 `arrive_and_expect_tx` 同时做一份 arrival 和 transaction-byte 登记

`include/cutlass/arch/barrier.h:544-602`（函数 `cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx`）最终执行：

```ptx
mbarrier.arrive.expect_tx.shared::cta.b64 _, [barrier], transaction_bytes;
```

它同时完成两件事：

1. 对 pending arrival count 执行一次 arrive，使 `1 -> 0`；
2. 将本 phase 需要等待的 TMA transaction bytes 加入 tx-count。

所以 producer barrier 的完成条件是：

```text
pending arrival count == 0
并且
pending transaction bytes == 0
```

软件 arrive 很快就让第一项归零，但 barrier 不会马上完成；它仍要等待 TMA engine 完成所有登记的字节。

#### 23.2.3 两次 `copy` 为什么不是 arrival count 2

`examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:200-209`（函数 `gemm_device`）在同一个 producer phase 中发起 A、B 两次高层 copy：

```cpp
ProducerBarType::arrive_and_expect_tx(
    &producer_mbar[pipe], tma_transaction_bytes);
copy(tma_a.with(producer_mbar[pipe]), ...);
copy(tma_b.with(producer_mbar[pipe]), ...);
```

这两次 copy 不会各自执行普通 `mbarrier.arrive`。它们使用带 `mbarrier::complete_tx::bytes` 的 TMA load；例如 `include/cute/arch/copy_sm90_tma.hpp:103-130`（函数 `cute::SM90_TMA_LOAD_2D::copy`）把同一个 mbarrier 地址传给 TMA 指令，由 TMA 完成时减少 tx-count。

`examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:158-160`（函数 `gemm_device`）已经把 A 和 B 的总字节数合并为 `tma_transaction_bytes`：

```text
A tile: 128 * 64 * 2 = 16384 bytes
B tile: 128 * 64 * 2 = 16384 bytes
total                    32768 bytes
```

因此这里是“一次软件 arrive + 32768 bytes transaction completion”，不是“两次 arrival”。这就是 `init(...,1)` 而不是 `init(...,2)` 的原因。

### 23.3 为什么 `ConsumerBarType` 的参数是 128

`ConsumerBarType` 是普通 `ClusterBarrier`，用于表示“所有使用当前 shared-memory stage 的线程都已经消费完毕，producer 可以覆盖该 stage”。

#### 23.3.1 当前 CTA 恰好有 128 个线程

`examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:329-347`（函数 `gemm_nt`）用：

```cpp
TiledMMA tiled_mma = make_tiled_mma(
    SM90_64x64x16_F16F16F16_SS<...>{});
dim3 dimBlock(size(tiled_mma));
```

当前 `size(tiled_mma) == 128`，所以一个 CTA 是一个 128-thread warpgroup，即 4 个 warp。运行日志中的 `ThrLayoutVMNK: (_128,...)` 和 `dimBlock.x:128` 也验证了这一点。

#### 23.3.2 每个线程都会执行一次 `ConsumerBarType::arrive`

在 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:253-270`（函数 `gemm_device`）中，以下调用不在 elected-lane 条件内部：

```cpp
warpgroup_wait<0>();
ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
```

所以 CTA 的 128 个线程在完成当前 stage 的 WGMMA 消费后，每个线程都对 consumer barrier arrive 一次。`include/cutlass/arch/barrier.h:507-523`（函数 `cutlass::arch::ClusterBarrier::arrive`）中的每次调用执行一个 `mbarrier.arrive`，因此 pending count 的变化是：

```text
128 -> 127 -> 126 -> ... -> 1 -> 0
```

只有第 128 个线程也报告消费完成后，这个 phase 才完成。随后 producer 才能通过 `ConsumerBarType::wait`，重新向该 pipe 发起 TMA load；等待位置见 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:272-281`（函数 `gemm_device`）。

如果把 consumer count 错设为 1，第一个到达的线程就可能使 barrier phase 提前完成，producer 会覆盖仍被其他线程读取的 shared-memory stage。如果设为大于 128，则永远收不到足够的 arrivals，producer 会一直等待。

### 23.4 waiter 数量与 arrival count 没有直接关系

producer barrier 上是 128 个线程执行 wait，但 arrive count 是 1；consumer barrier 上只有一个 elected producer thread 执行 wait，但 arrive count 是 128。原因是 wait 只观察 phase 是否完成，不会递减 pending arrival count。

`include/cutlass/arch/barrier.h:408-427`（函数 `cutlass::arch::ClusterBarrier::wait`）只循环执行 `mbarrier.try_wait.parity`；这里没有 `mbarrier.arrive`。所以必须根据“谁调用 arrive”设置 init count，而不能根据“谁调用 wait”设置。

### 23.5 为什么 cluster 有 2 个 CTA，但不是 256

虽然本例 launch 使用 `dimCluster(2,1,1)`，这里的两个 barrier 都位于每个 CTA 自己的 shared memory 中。当前 consumer 调用的是本地 overload：

```cpp
ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
```

`include/cutlass/arch/barrier.h:507-519`（函数 `cutlass::arch::ClusterBarrier::arrive`）对应 `mbarrier.arrive.shared::cta`，不是带 `cta_id` 的 remote cluster arrive。因此每个 CTA 独立统计自己的 128 个线程，不是把 cluster 内两个 CTA 合成 256。

producer barrier 同理：每个 CTA 有自己的 elected producer lane 和自己的 TMA barrier，所以 expected arrival count 仍是 1。

### 23.6 一个 pipe 的完整握手过程

```text
初始化:
  producer_mbar expected arrivals = 1
  consumer_mbar expected arrivals = 128

Producer 填充 stage:
  1 个 elected lane 执行 arrive_and_expect_tx
  -> producer arrival: 1 -> 0
  -> 登记 A+B 共 32768 transaction bytes
  -> TMA 完成全部 bytes
  -> producer barrier phase 完成

Consumer 使用 stage:
  128 个线程等待 producer barrier
  -> 执行 WGMMA
  -> 128 个线程分别 arrive consumer barrier
  -> consumer arrival: 128 -> 0
  -> consumer barrier phase 完成

Producer 复用 stage:
  1 个 elected lane等待 consumer barrier
  -> 安全地覆盖这个 shared-memory pipe
```

`examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:248-282`（函数 `gemm_device`）使用 `PipelineState` 管理 pipe index 和 phase。`include/cutlass/pipeline/sm90_pipeline.hpp:168-213`（函数 `cutlass::PipelineState::operator++`）在循环绕过最后一个 stage 时执行 `phase_ ^= 1`，使同一个物理 barrier 可以在后续 K tile 中重复使用。

所以两个数字的本质是：

```text
1   = 每个 producer phase 的软件 arrive 调用次数
128 = 每个 consumer phase 的线程 arrive 调用次数
```

如果以后修改成 warp-specialized kernel、改变 consumer 线程集合或让多个 producer 参与 arrive，这两个值也必须随实际协议修改；它们并不是所有 Hopper TMA kernel 都固定使用的常数。

## 24. `cluster_sync()` 中 `barrier.cluster.arrive` 与 `barrier.cluster.wait`

### 24.1 `cluster_sync()` 的实际展开

在 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:201-202`（函数 `gemm_device`）中：

```cpp
// Ensure barrier init is complete on all CTAs
cluster_sync();
```

`cluster_sync()` 的实现位于 `include/cute/arch/cluster_sm90.hpp:75-83`（函数 `cute::cluster_sync`）：

```cpp
CUTE_DEVICE void cluster_sync()
{
  cluster_arrive();
  cluster_wait();
}
```

其中两个包装函数分别在 `include/cute/arch/cluster_sm90.hpp:57-73`（函数 `cute::cluster_arrive` 和 `cute::cluster_wait`）展开为：

```cpp
asm volatile("barrier.cluster.arrive.aligned;\n" : : );
asm volatile("barrier.cluster.wait.aligned;\n" : : );
```

因此它是一个“分离式 cluster barrier”：先 arrive，后 wait，而不是一条既登记又等待的单独指令。

### 24.2 `barrier.cluster.arrive.aligned` 的功能

```ptx
barrier.cluster.arrive.aligned;
```

`barrier.cluster.arrive` 的作用是登记当前 warp 已经到达 cluster barrier。它不会等待其他 warp 或其他 CTA 到达，执行线程可以继续运行后面的代码。`.aligned` 表示同一个 warp 的非退出线程必须执行相同的这条 barrier 指令；不能让同一个 warp 中只有部分线程执行它。

在本例中，`arrive` 的默认语义是 release：在 arrive 之前发出的普通内存访问可以通过后续匹配的 cluster wait 与其他参与者建立可见性关系。它本身不提供“所有 CTA 已经到达”的等待效果，也不代表 barrier phase 已经完成。

可以把它理解为：

```text
每个 warp: 记录“我到达了”
当前线程: 不等待其他 warp/CTA
barrier 状态: 累积本 phase 的 arrivals
```

### 24.3 `barrier.cluster.wait.aligned` 的功能

```ptx
barrier.cluster.wait.aligned;
```

`barrier.cluster.wait` 会阻塞执行线程，直到 cluster 中所有非退出参与者都已经执行对应的 `barrier.cluster.arrive`。当条件满足时，barrier phase 完成并重新初始化，下一轮可以复用同一个隐式 cluster barrier。

`.aligned` 同样要求 warp 内线程收敛地执行同一条指令。wait 默认带 acquire 语义；wait 返回后，其他线程在 arrive 之前完成的普通内存访问对当前线程可见。也就是说，`arrive` 负责“发布到达和之前的写入”，`wait` 负责“等待全部到达并获取这些写入”。

在当前代码中，两个 CTA 通过 `dimCluster(2,1,1)` 组成 cluster，launch 参数见 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:345-354`（函数 `gemm_nt`）。因此这里同步的是 cluster 内的两个 CTA，而不是只同步单个 CTA；`__syncthreads()` 不能替代它。

### 24.4 为什么初始化 barrier 后必须同时使用两条指令

`gemm_device` 中的 producer/consumer mbarrier 初始化只由每个 CTA 的一个 elected lane 执行，代码见 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:182-200`（函数 `gemm_device`）：

```cpp
if ((warp_idx == 0) && lane_predicate) {
  ProducerBarType::init(...);
  ConsumerBarType::init(...);
}
```

`cluster_sync()` 的目的就是让每个 CTA 的 barrier 初始化完成并对 cluster 内其他 CTA 可见，然后才在 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:204-212`（函数 `gemm_device`）开始调用 `arrive_and_expect_tx` 和发起 TMA load。

### 24.5 只保留 `arrive` 会怎样

如果替换成：

```cpp
asm volatile("barrier.cluster.arrive.aligned;\n" : : );
```

而删除 `wait`，当前线程和 CTA 会在登记 arrival 后直接继续执行。结果是：

1. 不再等待 cluster 内其他 CTA 的 elected lane 完成 `mbarrier.init`；某个 CTA 可能已经开始 `arrive_and_expect_tx` 或 TMA load，而另一个 CTA 还在初始化自己的 barrier，形成初始化访问竞态。
2. 当前 cluster barrier phase 没有完成，因为完成条件要求参与者后续执行匹配的 wait；该 barrier 也不会按正常流程重置以供下一轮使用。
3. 这条路径没有建立 arrive-wait 的 acquire 侧同步，因此不能把它当作 cluster 级内存可见性屏障。

本例可能表现为 TMA/barrier 状态错误、数据竞态或后续同步挂起；即使某次运行碰巧没有报错，也不能认为初始化已经被正确同步。

### 24.6 只保留 `wait` 会怎样

如果替换成：

```cpp
asm volatile("barrier.cluster.wait.aligned;\n" : : );
```

则没有当前 phase 的 `barrier.cluster.arrive`。在本 kernel 第一次执行 `cluster_sync()` 时，cluster barrier 没有任何匹配的到达记录，所有执行 wait 的线程都会等待，通常表现为 kernel 在第 202 行永久卡住。

只有在某个更早的、同一 barrier phase 的代码已经执行过匹配 arrive 时，单独 wait 才可能继续；本例并不存在这样的前置 arrive，所以不能省略 arrive。

### 24.7 两条指令的配合关系

```text
所有 warp 执行 barrier.cluster.arrive.aligned
    -> 每个 warp 登记 arrival，释放 arrive 之前的内存访问
所有 warp 执行 barrier.cluster.wait.aligned
    -> 等待 cluster 内全部 arrival
    -> acquire arrive 之前的可见写入
    -> barrier phase 完成并可复用
```

因此，`cluster_sync()` 不是简单的函数拆分，而是一个完整的 release/acquire cluster synchronization 协议：`arrive` 不能替代 `wait`，`wait` 也不能在没有匹配 `arrive` 的情况下独立工作。

PTX 规范对这两个指令的描述见 [NVIDIA PTX ISA barrier.cluster 文档](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)。

## 25. NVIDIA PTX 中 release/acquire 语义的详细例子

### 25.1 release/acquire 先看成一条“发布-获取”链

对一段跨线程共享的数据，可以先用下面的抽象顺序理解：

```text
线程 A 写数据
    -> release 操作：发布此前的写入
    -> acquire 操作：等待并获取发布结果
线程 B 读数据
```

如果 release 操作和 acquire 操作通过同一个同步对象建立了同步关系，那么线程 A 在 release 之前的写入，才能保证在线程 B 的 acquire 之后可见。release 不会发布它之后才发生的写入；acquire 也不会替线程 B 执行一个普通的读取。

在本例中，
- `barrier.cluster.arrive.aligned` 没有显式写 `.release`，但默认使用 release 语义；
- `barrier.cluster.wait.aligned` 没有显式写 `.acquire`，但默认使用 acquire 语义。

因此下面两种写法在默认内存序上等价：

```ptx
barrier.cluster.arrive.aligned;
barrier.cluster.wait.aligned;

barrier.cluster.arrive.release.aligned;
barrier.cluster.wait.acquire.aligned;
```

相关封装分别位于 `include/cute/arch/cluster_sm90.hpp:57-83`（函数 `cute::cluster_arrive`、`cute::cluster_wait`、`cute::cluster_sync`）。

### 25.2 例子一：两个 CTA 交换 shared-memory 数据

假设 cluster 中有 CTA 0 和 CTA 1，各自写一个 shared-memory slot，写完后希望两个 CTA 都读取对方的 slot：

```ptx
// CTA 0: 写 slot0；CTA 1: 写 slot1
st.shared::cluster.u32 [payload_slot], value;

// 两个 CTA 的所有参与 warp 都执行
barrier.cluster.arrive.release.aligned;
barrier.cluster.wait.acquire.aligned;

// wait 返回后再读取对方写入的 slot
ld.shared::cluster.u32 result, [other_payload_slot];
```

执行顺序可以画成：

```text
CTA 0: st payload0 -> arrive.release -----\\
                                          +-> wait.acquire -> ld payload1
CTA 1: st payload1 -> arrive.release -----/
```

这里的关键不是 arrive 指令本身“把数据复制给了 CTA 1”，而是：

1. `st.shared::cluster` 把数据写入 cluster 可寻址的 shared memory；
2. `arrive.release` 把该线程在 arrive 之前的写入发布到 barrier 同步关系中；
3. `wait.acquire` 等待整个 cluster 的 arrivals，并在返回后建立读取侧的可见性；
4. `ld.shared::cluster` 放在 wait 之后，因此读取对方 slot 时才满足这条同步链。

本例不能把 `ld` 放在 wait 之前；acquire 只约束 acquire 之后的内存访问，不能追溯地修复已经发生的早读。

### 25.3 例子二：release 只负责它之前的写入

考虑下面的程序顺序：

```ptx
st.shared::cluster.u32 [data], 42;
barrier.cluster.arrive.release.aligned;
st.shared::cluster.u32 [data2], 99;
```

与之匹配的另一线程执行：

```ptx
barrier.cluster.wait.acquire.aligned;
ld.shared::cluster.u32 r0, [data];
ld.shared::cluster.u32 r1, [data2];
```

release 只保证第一条 `st`（位于 arrive 之前）参与 release/acquire 同步。第二条 `st data2` 位于 arrive 之后，不会因为前面的 release 自动被发布；它需要自己的同步协议，或者必须移动到 release 之前。

所以 release/acquire 不是“对整个线程未来所有内存操作加锁”，而是一个有程序顺序边界的内存序关系：

```text
release 之前的访问  --被发布-->  acquire 之后的访问
release 之后的访问  --不自动包含在本次发布中--
```

### 25.4 例子三：`.relaxed` 仍然同步到达，但不发布普通内存写入

PTX 还提供：

```ptx
barrier.cluster.arrive.relaxed.aligned;
```

`.relaxed` 版本仍然登记 arrival，`wait` 仍然可以等待这个 barrier；但 arrive 不再为 arrive 之前的普通内存访问提供 release 排序。下面的代码不能仅凭 barrier 保证 `data` 在另一 CTA 的 wait 后可见：

```ptx
st.shared::cluster.u32 [data], 42;
barrier.cluster.arrive.relaxed.aligned;
barrier.cluster.wait.acquire.aligned;
```

如果确实要使用 relaxed arrive，需要显式的 cluster fence 在 arrive 之前建立内存顺序。PTX 文档给出的模式是：

```ptx
st.shared::cluster.u32 [data], 42;
fence.cluster.acq_rel;
barrier.cluster.arrive.relaxed.aligned;
barrier.cluster.wait.acquire.aligned;
```

这说明“barrier 到达同步”和“内存访问顺序”是两个可分别控制的维度：`.relaxed` 只去掉 arrive 的内存序保证，不会把 barrier 变成普通计算指令。

### 25.5 例子四：global 原子操作中的 `.release` / `.acquire`

release/acquire 不只出现在 `barrier.cluster`。CUTLASS 的 CuTeDSL 协作启动代码中，`examples/python/CuTeDSL/ampere/cooperative_launch.py:300-350`（函数 `_read_barrier`）使用：

```ptx
ld.global.acquire.gpu.b32 value, [barrier_ptr];
```

这个 load 的含义是：当它读取到同步协议要求的 barrier 状态后，后续 global/shared 内存访问不能被放到 acquire 之前，并可以观察 release 侧已经发布的 GPU-scope 写入。这里的 `.gpu` 是作用域，表示 GPU 范围；`.acquire` 是内存序，两者不是同一个概念。

同一个文件的 `examples/python/CuTeDSL/ampere/cooperative_launch.py:352-407`（函数 `_increment_barrier`）使用：

```ptx
atom.add.release.gpu.u32 old, [barrier_ptr], increment;
```

它是一个 GPU-scope 的 release 原子加：原子性负责“多个线程的加法不会互相破坏”，release 负责“该原子操作之前的写入可以通过匹配的 acquire 读取被观察到”。因此：

```text
.gpu      = 同步作用域
.release  = 发布方向
.add      = 原子读改写操作
```

当前 `barrier.cluster` 的语法把作用域隐含在 `cluster` 中，而 `ld.global.acquire.gpu` / `atom.add.release.gpu` 把作用域显式写在指令后缀中。

### 25.6 例子五：`volatile` 不是 release/acquire

当前 wrapper 写成：

```cpp
asm volatile("barrier.cluster.arrive.aligned;\n" : : );
```

这里有两个不同层次：

- `volatile` 是 inline asm 对编译器的副作用声明，防止汇编被当作无用代码删除；
- `barrier.cluster.arrive.aligned` 的 release/acquire 语义来自 PTX 指令本身及其默认 `.release` 修饰，而不是来自 C++ 的 `volatile`。

把同一条指令写成普通的非 volatile asm，不能把它变成 relaxed；反过来，写了 `asm volatile` 也不能给一条没有 release/acquire 语义的普通指令凭空增加内存序。

### 25.7 例子六：当前 TMA pipeline 中的两套同步协议

当前 kernel 需要区分 cluster barrier 和 TMA transaction barrier：

```text
cluster_sync():
  barrier.cluster.arrive.release.aligned
  barrier.cluster.wait.acquire.aligned
  -> 保证每个 CTA 已完成 mbarrier.init，且初始化写入可见

TMA producer barrier:
  mbarrier.arrive.expect_tx
  cp.async.bulk.tensor...mbarrier::complete_tx::bytes
  mbarrier.try_wait.parity
  -> 等待异步 TMA transaction 的字节数完成
```

`cluster_sync()` 的调用位置和用途见 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:201-213`（函数 `gemm_device`）。TMA producer barrier 的登记和 copy 见 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:204-213`（函数 `gemm_device`），真正等待 TMA phase 的代码见 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:253-259`（函数 `gemm_device`）。

因此，不能把下面两件事混为一谈：

1. `barrier.cluster.wait.acquire` 保证 cluster barrier 之前的普通内存访问具有获取侧可见性；
2. `mbarrier.try_wait.parity` 保证与该 mbarrier 关联的异步 TMA transaction 已完成并可被使用。

前者是 cluster 范围的 release/acquire 内存同步，后者是 TMA/mbarrier 的异步完成协议；在本 kernel 中两者分别负责 barrier 初始化和 shared-memory tile 数据就绪。

### 25.8 一句话记忆规则

```text
release: 我在这里之前写的内容，可以被匹配的 acquire 之后读取
acquire: 我在这里之后读取时，可以依赖匹配 release 之前的写入
relaxed: 仍可参与同步协议，但不自动提供上述内存顺序
scope: 决定同步可传播到 CTA、cluster、GPU 还是 system
volatile: 约束编译器保留 asm，不定义 GPU 内存序
```

PTX 对 `barrier.cluster` 的默认 release/acquire 规则、`.relaxed` 差异和 `fence.cluster` 用法，见 [NVIDIA PTX ISA barrier.cluster 文档](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)。

## 26. `size(tAsA)`、`size<0>(tAsA)` 和 `K_PIPE_MAX` 为什么打印错误

### 26.1 日志中的 0 不是 Tensor 的真实 size

打印代码位于 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:176-182`（函数 `gemm_device`）：

```cpp
auto K_PIPE_MAX = size<1>(tAsA);
int k_tile_count = size<1>(tAgA);

printf(" size(tAsA):%d, size<0>(tAsA):%d, K_PIPE_MAX:%d\n",
       size(tAsA), size<0>(tAsA), K_PIPE_MAX);
printf("size(tAgA):%d, size<0>(tAgA):%d, k_tile_count:%d\n",
       size(tAgA), size<0>(tAgA), k_tile_count);
```

日志显示：

```text
size(tAsA):32768, size<0>(tAsA):0, K_PIPE_MAX:0
size(tAgA):262144, size<0>(tAgA):0, k_tile_count:32
pipe=0, K_PIPE_MAX:0
pipe=1, K_PIPE_MAX:0
pipe=2, K_PIPE_MAX:0
```

其中 `size<0>(tAsA)=0`、`K_PIPE_MAX=0`、`size<0>(tAgA)=0` 都是假象。根因是把 CuTe 的静态整数类型 `cute::C<N>` 直接传给了 C 风格可变参数 `printf("%d", ...)`，导致 format 与实际参数类型不匹配。该调用属于未定义行为，打印值不能用于判断 Tensor shape。

### 26.2 根据 Tensor layout 计算正确结果

`tAgA` 和 `tAsA` 在 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:153-168`（函数 `gemm_device`）由 `tma_partition` 生成。日志中的 layouts 是：

```text
tAsA = ((_512,_16),(_1,_3)):((_1,_512),(_0,_8192))
tAgA = (((_64,_8),(_2,_8)),32):
        (((_1@0,_1@1),(_64@0,_8@1)),_64@1)
```

#### 26.2.1 `tAsA` 的正确 size

`tAsA` 的 top-level mode 0 是 `(_512,_16)`：

```text
size<0>(tAsA) = 512 * 16 = 8192
```

top-level mode 1 是 `(_1,_3)`：

```text
size<1>(tAsA) = 1 * 3 = 3
K_PIPE_MAX    = 3
```

整个 Tensor 的元素数是：

```text
size(tAsA) = 8192 * 3 = 24576
```

所以正确结果不是 `32768,0,0`，而是：

```text
size(tAsA)=24576, size<0>(tAsA)=8192, K_PIPE_MAX=3
```

#### 26.2.2 `tAgA` 的正确 size

`tAgA` 的 top-level mode 0 是 `((_64,_8),(_2,_8))`：

```text
size<0>(tAgA) = 64 * 8 * 2 * 8 = 8192
```

top-level mode 1 是运行时 K tile count `32`：

```text
size<1>(tAgA) = 32
k_tile_count  = 32
size(tAgA)    = 8192 * 32 = 262144
```

因此第二行中 `size(tAgA)=262144` 和 `k_tile_count=32` 恰好打印正确，只有静态的 `size<0>(tAgA)` 被错误打印为 0。

### 26.3 `cute::size` 为什么有时返回普通 `int`，有时返回 `C<N>`

`include/cute/tensor_impl.hpp:548-555`（函数 `cute::size(Tensor)`）把请求转发给 Tensor layout；`include/cute/layout.hpp:603-610`（函数 `cute::size(Layout)`）再对选中的 shape 调用 `size`。最终，`include/cute/int_tuple.hpp:221-276`（函数 `cute::Product::operator()` 和 `cute::size(IntTuple)`）计算各 shape leaf 的乘积。

CuTe 会保留 shape 的静态/动态属性：

| 表达式 | 正确值 | 返回类型特征 | 直接传 `%d` |
|---|---:|---|---|
| `size(tAsA)` | 24576 | 全静态，`C<24576>` | 错误 |
| `size<0>(tAsA)` | 8192 | 全静态，`C<8192>` | 错误 |
| `size<1>(tAsA)` | 3 | 全静态，`C<3>` | 错误 |
| `size(tAgA)` | 262144 | 包含动态 K tile mode，运行时整数 | 正确 |
| `size<0>(tAgA)` | 8192 | mode 0 全静态，`C<8192>` | 错误 |
| `size<1>(tAgA)` | 32 | 动态 K tile mode，运行时整数 | 正确 |

日志中静态值带下划线，例如 `_512`、`_16`、`_3`；动态值没有下划线，例如最外层的 `32`。这也能直接看出哪些 size 计算会保留为 `C<N>`。

### 26.4 为什么 `C<N>` 有整数转换，`printf` 仍然打印错

`include/cute/numeric/integral_constant.hpp:40-47`（函数 `cute::C<v>::operator value_type`）确实为 `C<N>` 提供了 constexpr 整数转换：

```cpp
template <auto v>
struct C {
  static constexpr auto value = v;
  constexpr operator value_type() const noexcept { return value; }
};
```

这个转换在普通、类型受检查的 C++ 表达式中会生效，例如：

```cpp
int n = size<0>(tAsA);
if (pipe < K_PIPE_MAX) { ... }
```

但是 `printf` 的 `...` 是 C 风格可变参数。`%d` 只是告诉 `printf` 从参数区读取一个 `int`，不会让调用点对 class 类型执行类型安全的用户定义转换。直接传入空的静态常量 class `C<N>` 后，device varargs ABI 中没有与 `%d` 匹配的普通 `int`，于是 `printf` 从错误的寄存器或参数槽读取数据。

这解释了：

- `size<0>(tAsA)`、`K_PIPE_MAX`、`size<0>(tAgA)` 恰好显示为 0；
- 全静态的 `size(tAsA)` 显示为 32768；
- 这些错误值可能随编译器、优化等级或周围代码变化。

日志中的 `32768` 与之前计算的 `tma_transaction_bytes` 相同只是当前 ABI/寄存器状态下的偶然结果，不能解释为 `tAsA` 的真实元素数。

### 26.5 为什么循环仍然执行了 3 次

`examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:176-201`（函数 `gemm_device`）中，`K_PIPE_MAX` 的类型是 `C<3>`。在普通 C++ 比较中，它会通过 `C<3>::operator int()` 转换成 3：

```cpp
for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
  ...
}
```

所以循环实际执行 `pipe=0,1,2`。日志已经打印出这三次循环，这反过来证明 `K_PIPE_MAX` 的逻辑值是 3；每一行中的 `K_PIPE_MAX:0` 仍然只是同一个 `%d` 类型不匹配问题。

同理，`cutlass::PipelineState<K_PIPE_MAX>` 使用的是编译期值 3，不会因为调试打印显示 0 而变成零 stage pipeline。

### 26.6 正确的打印方法

最直接的修正是在进入 varargs 前显式转换为 `int`：

```cpp
printf("size(tAsA):%d, size<0>(tAsA):%d, K_PIPE_MAX:%d\n",
       int(size(tAsA)), int(size<0>(tAsA)), int(K_PIPE_MAX));
printf("size(tAgA):%d, size<0>(tAgA):%d, k_tile_count:%d\n",
       int(size(tAgA)), int(size<0>(tAgA)), int(k_tile_count));
```

预计输出为：

```text
size(tAsA):24576, size<0>(tAsA):8192, K_PIPE_MAX:3
size(tAgA):262144, size<0>(tAgA):8192, k_tile_count:32
```

如果希望同时观察一个值是否为 CuTe 静态常量，可以使用 CuTe 自己的打印函数。`include/cute/numeric/integral_constant.hpp:478-486`（函数 `cute::print(C<Value>)`）会把静态值打印成带下划线的形式：

```cpp
print(size<0>(tAsA));  // _8192
print(K_PIPE_MAX);     // _3
```

结论是：Tensor shape 和 pipeline 配置本身都是正确的；错误只发生在调试输出这一层。

## 27. 为什么先 `arrive_and_expect_tx` 再执行 TMA `copy`

### 27.1 正常顺序先登记 transaction bytes，再允许 TMA 完成

相关代码位于 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:208-217`（函数 `gemm_device`）：

```cpp
ProducerBarType::arrive_and_expect_tx(
    &producer_mbar[pipe], tma_transaction_bytes);
copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
```

同样的顺序也用于 pipeline stage 的后续复用，见 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:280-289`（函数 `gemm_device`）。

producer barrier 初始化时的状态是：

```text
pending arrival count = 1
tx-count              = 0
```

`include/cutlass/arch/barrier.h:586-602`（函数 `cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx`）执行：

```ptx
mbarrier.arrive.expect_tx.shared::cta.b64 _, [barrier], transaction_bytes;
```

这条 PTX 指令内部先执行 expect-tx，再执行 arrive-on。因此，当前例子的状态变化为：

```text
初始化:
  arrivals=1, tx=0

arrive_and_expect_tx(32768):
  expect_tx: tx 0 -> 32768
  arrive:    arrivals 1 -> 0
  当前 phase 仍未完成，因为 tx != 0

A/B TMA 完成:
  complete_tx: tx 32768 -> 0
  arrivals==0 且 tx==0
  -> producer barrier phase 完成
```

`tma_transaction_bytes` 在 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:159-161`（函数 `gemm_device`）中计算为 A、B 两个 128x64 half tile 的总字节数：

```text
A: 128 * 64 * 2 = 16384 bytes
B: 128 * 64 * 2 = 16384 bytes
合计                 32768 bytes
```

先登记 32768 bytes 的优点是，从 TMA 指令开始有机会完成的那一刻起，barrier 已经知道本 phase 必须等待多少工作。tx-count 只从正数下降到 0，phase 状态最直观，也最不容易被其他 arrive/complete 操作破坏。

### 27.2 为什么不能只做 arrive 而不 expect bytes

mbarrier phase 只有在下面两项都为 0 时才完成：

```text
pending arrival count == 0
tx-count              == 0
```

如果这里只执行普通 arrive，状态会立即从 `(1,0)` 变为 `(0,0)`，barrier 会在 TMA copy 发出前进入下一 phase。之后消费者可能把尚未填充的 shared-memory stage 当作 ready，TMA completion 也可能被记入错误的 phase。

`arrive_and_expect_tx` 的关键作用不是“启动 copy”，而是在消费唯一 arrival 的同时，用非零 tx-count 把当前 phase 保持在 incomplete 状态，直到 TMA engine 报告全部字节完成。

### 27.3 如果把顺序反过来，是否一定失败

假设改成：

```cpp
copy(tma_a.with(producer_mbar[pipe]), ...);
copy(tma_b.with(producer_mbar[pipe]), ...);
ProducerBarType::arrive_and_expect_tx(
    &producer_mbar[pipe], tma_transaction_bytes);
```

结论需要分两层：

- 对这个特定 kernel，反序不一定立即失败；
- 作为通用 barrier 协议，这种顺序更脆弱，不应依赖。

TMA `copy` 是异步 issue。即使 copy 写在前面，通常 CPU 风格的下一条 producer 指令 `arrive_and_expect_tx` 仍可能在 DMA 真正完成前执行，此时状态最终仍与正常顺序相同。

更重要的是，PTX 的 mbarrier tx-count 合法范围允许负数。如果部分 TMA 已在晚到的 expect-tx 之前完成，complete-tx 可以先把 tx-count 减成负值。以全部 32768 bytes 都提前完成为例：

```text
初始化:                    arrivals=1, tx=0
TMA complete_tx 先发生:    arrivals=1, tx=-32768
晚到的 expect_tx(+32768):  arrivals=1, tx=0
同一指令随后 arrive:       arrivals=0, tx=0
                           -> phase 完成
```

当前 pending arrival 在最后一条指令之前一直为 1，所以即使 tx-count 暂时回到 0，phase 也不会提前完成。这也是这个简单、单 producer 协议反序后仍可能工作的原因。

### 27.4 反序的实际风险和适用条件

反序要保持正确，至少依赖以下条件：

1. 在 late expect-tx 之前，必须还有非零 pending arrival 阻止 phase transition；
2. 提前发生的 complete-tx 不能让负 tx-count 超出 PTX 允许范围；
3. 所有 complete-tx 和 late expect-tx 必须仍作用于同一个 barrier phase；
4. 不能有其他 producer/线程先执行 arrive，使 pending arrival 提前归零；
5. copy 和 late expect 之间不能出现提前退出或控制流分歧。

只要协议改成多个 producer、分开的 `expect_transaction`/`arrive`、额外 arrive，或者 barrier 在中间发生 phase transition，晚到的 expect-tx 就可能登记到下一 phase，导致当前 phase 提前 ready、下一 phase tx-count 错误或消费者永久等待。

所以源码采用“先 expect/arrive，后 issue async copy”的保守顺序。它把所有将要发生的 completions 预先登记到当前 phase，不需要依赖负 tx-count 或额外的 phase 不变量。

PTX 对 tx-count 可为负、expect-tx 增加计数、complete-tx 减少计数，以及 phase 完成必须同时满足 arrivals/tx-count 为 0 的定义，见 [NVIDIA PTX ISA mbarrier 文档](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)。

### 27.5 `with(producer_mbar[pipe])` 先生成可执行的 TMA Copy Atom

`tma_a` 和 `tma_b` 本身只有 TMA descriptor，没有本次传输要通知的 mbarrier。`include/cute/atom/copy_traits_sm90_tma.hpp:99-162`（成员函数 `cute::Copy_Traits<SM90_TMA_LOAD, ...>::with`）明确把这种 traits 称为 non-executable；其不带 mbarrier 的 `copy_unpack` 甚至被声明为 deleted。

表达式：

```cpp
tma_a.with(producer_mbar[pipe])
```

先进入 `include/cute/atom/copy_atom.hpp:77-83`（成员函数 `cute::Copy_Atom::with`），再调用上面的 traits `with`。后者把三个运行时参数保存到新的 traits 中：

```text
TMA descriptor 指针
producer_mbar[pipe] 指针
cache hint
```

返回类型变为可执行的 `Copy_Atom<Copy_Traits<SM90_TMA_LOAD_OP, ...>, half_t>`。该 executable traits 的定义位于 `include/cute/atom/copy_traits_sm90_tma.hpp:172-202`（类型 `cute::Copy_Traits<SM90_TMA_LOAD_OP, ...>`），并继承负责最终参数展开的 `TMA_LOAD_Unpack`。

因此，`.with(...)` 不是执行 copy，也不是等待 barrier；它是把当前 pipe 的 mbarrier 地址绑定到临时 Copy Atom。之后由该 Atom 发出的每条 TMA 指令都会把 completion 记到同一个 `producer_mbar[pipe]`。

### 27.6 表面调用首先命中接收 mutable temporary 的转发重载

以 A 为例，`examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:215-217`（函数 `gemm_device`）中的调用是：

```cpp
copy(tma_a.with(producer_mbar[pipe]),
     tAgA(_,k_tile),
     tAsA(_,pipe));
```

`tAsA(_,pipe)` 是一个可写的临时 Tensor 视图，即 rvalue。它首先匹配 `include/cute/algorithm/copy.hpp:536-545`（函数 `cute::copy(CopyPolicy const&, Tensor const&, Tensor&&)`）：

```cpp
return copy(copy_policy, src, dst);
```

这个重载自身不搬运数据。它把函数体内有名字的 `dst` 作为 lvalue，再转发给真正的 copy 实现。这样既允许调用者传临时 Tensor 视图，又不会把 destination 当作只读对象。

转发后，第一参数的实际类型是 `Copy_Atom<...>`，所以比通用 CopyPolicy overload 更具体、真正被选择的是 `include/cute/algorithm/copy.hpp:184-235`（函数 `cute::copy(Copy_Atom const&, Tensor const&, Tensor&)`）。B 的调用经过完全相同的两个重载。

### 27.7 `Copy_Atom` 专用重载怎样处理当前 Tensor shape

日志中的 A partition 是：

```text
tAgA = (((_64,_8),(_2,_8)),32)
tAsA = ((_512,_16),(_1,_3))
```

选择一个 `k_tile` 和一个 `pipe` 后，source/destination 的 TMA mode 都有 8192 个 half 元素。顶层是一个带嵌套 shape 的 rank-1 Tensor，所以 `include/cute/algorithm/copy.hpp:195-196`（函数 `cute::copy(Copy_Atom const&, Tensor const&, Tensor&)`）先调用 `copy_atom.call(src, dst)`。

当前 Atom 日志中的 `ValLayoutSrc` 和 `ValLayoutDst` 都是 `(_1,_512)`，即一条底层 TMA copy 对应 512 个 half。`include/cute/atom/copy_atom.hpp:89-114`（成员函数 `cute::Copy_Atom::call`）发现 Tensor 的 8192 个元素不等于 Atom 的 512 个元素，但 shape 可以继续展开，于是执行：

```cpp
copy(*this, tensor<0>(src), tensor<0>(dst));
```

`include/cute/tensor_impl.hpp:506-518`（函数 `cute::tensor<Is...>`）取出嵌套 mode 的 layout，使这次调用看到：

```text
第 0 mode: 512 个 half，交给一条 TMA copy
rest mode: 16 次
```

随后再次进入 `include/cute/algorithm/copy.hpp:184-235`（函数 `cute::copy(Copy_Atom const&, Tensor const&, Tensor&)`）。这次 Tensor rank 大于 1，代码在 `include/cute/algorithm/copy.hpp:224-227` 的循环中，对 16 个 rest-mode slice 分别调用一次 `copy_atom.call`。每个 slice 恰好有 512 个 half，因此命中 `include/cute/atom/copy_atom.hpp:100-103`（成员函数 `cute::Copy_Atom::call`）的 `copy_unpack(...)` 分支。

所以一条源代码级的 A `copy(...)` 展开为 16 条底层 TMA 2D load；B 同样是 16 条。每条搬运 `512 * sizeof(half) = 1024` bytes，两者合计：

```text
A: 16 * 1024 = 16384 bytes
B: 16 * 1024 = 16384 bytes
总计             32768 bytes
```

这与 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:159-161`（函数 `gemm_device`）登记给 producer barrier 的 `tma_transaction_bytes` 一致。

### 27.8 从 `copy_unpack` 到最终 PTX 指令

每个 512-element slice 的底层调用链如下：

```text
cute::copy(Copy_Atom, src, dst)
  -> Copy_Atom::call
  -> copy_unpack(Copy_Traits<SM90_TMA_LOAD_OP>, src, dst)
  -> detail::CallCOPY<SM90_TMA_LOAD_OP>::operator()
  -> SM90_TMA_LOAD_OP::copy（继承自 SM90_TMA_LOAD）
  -> SM90_TMA_LOAD_2D::copy
  -> cp.async.bulk.tensor.2d...mbarrier::complete_tx::bytes
```

各层对应的代码是：

1. `include/cute/atom/copy_traits_sm90_tma.hpp:64-90`（函数 `cute::copy_unpack`）：从 source Tensor 取 2D TMA coordinate，从 destination Tensor 取 shared-memory 地址，并展开 descriptor、mbarrier、cache hint 和 coordinate 参数；
2. `include/cute/arch/util.hpp:149-160`（成员函数 `cute::detail::CallCOPY<CopyOp>::operator()`）：调用 `CopyOp::copy(...)`；
3. `include/cute/arch/copy_sm90_tma.hpp:327-342`（函数 `cute::SM90_TMA_LOAD::copy`）：当前 coordinate 有两个分量，因此选择 2D overload；
4. `include/cute/arch/copy_sm90_tma.hpp:103-135`（函数 `cute::SM90_TMA_LOAD_2D::copy`）：在 SM90 路径最终发出：

```ptx
cp.async.bulk.tensor.2d.shared::cluster.global
    .mbarrier::complete_tx::bytes.L2::cache_hint ...;
```

其中 `.mbarrier::complete_tx::bytes` 表示 TMA 数据完成时，硬件按实际完成的字节数更新 `.with(producer_mbar[pipe])` 所绑定的 transaction barrier。

### 27.9 “调用的是哪个 `copy`”的简要结论

这里不是只命中一个函数，而是一条分层重载/分派链：

```text
入口重载:
  cute::copy(CopyPolicy const&, Tensor const&, Tensor&&)
  // 接收 tAsA(_,pipe) / tBsB(_,pipe) 这样的可写临时视图

核心算法重载:
  cute::copy(Copy_Atom const&, Tensor const&, Tensor&)
  // 按 Copy Atom 的 512-element 粒度展开 8192-element TMA mode

最终硬件实现:
  cute::SM90_TMA_LOAD_2D::copy
  // 发出 cp.async.bulk.tensor.2d...mbarrier::complete_tx::bytes
```

因此，“先 `arrive_and_expect_tx`”负责在 barrier 当前 phase 中预登记全部 32768 bytes；后续两次高层 `copy` 经上述重载链展开为 A/B 的 TMA 指令，这些指令完成时再把同一个 barrier 的 tx-count 减回 0。

## 28. `tAgA` 的坐标 Tensor、`@` stride 与 TMA copy

### 28.1 本节对应的运行条件

运行命令为：

```bash
./build-bjh100/examples/cute/tutorial/hopper/cute_tutorial_wgmma_tma_sm90_like \
  512 1024 2048 N T
```

因此：

```text
M = 512
N = 1024
K = 2048
transA = N
transB = T
```

`examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:529-541`（函数 `gemm`）据此选择 `gemm_nt`。`examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:347-379`（函数 `gemm_nt`）为 A 建立 `(M,K):(1,ldA)` 布局，其中 `ldA=M=512`。GEMM 的 CTA tile 是 `128x128x64`，为 A 创建 TMA descriptor 时使用其中的 `(bM,bK)=(128,64)` copy tile。

日志只在 `blockIdx=(1,0)` 的 CTA 打印，条件见 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:104-118`（函数 `gemm_device`）。该 CTA 的 A tile 从：

```text
m = blockIdx.x * BLK_M = 1 * 128 = 128
k = 0
```

开始，因此后面会看到坐标起点 `(128,0)`。

### 28.2 `tAgA` 不是普通 global-memory pointer Tensor

需要区分源码中名字相同但用途不同的两个 `mA`：

1. `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:372-379`（函数 `gemm_nt`）中的 host-side `mA` 是真正以 A 的 global-memory pointer 为 engine 的数据 Tensor。`make_tma_atom` 用它建立 TMA descriptor；
2. `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:98-107`（函数 `gemm_device`）中的 device-side `mA` 是 `tma_a.get_tma_tensor(...)` 返回的 TMA **坐标 Tensor**，随后被切成 `gA`，再在 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:141-157`（函数 `gemm_device`）中由 `tma_partition` 变为 `tAgA`。

`include/cute/atom/copy_traits_sm90_tma.hpp:147-154`（成员函数 `cute::Copy_Traits<SM90_TMA_LOAD, ...>::get_tma_tensor`）的实现是：

```cpp
return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
```

`include/cute/tensor_impl.hpp:477-487`（函数 `cute::make_coord_tensor`）又把这个 layout 绑定到 `make_inttuple_iter(...)`，所以它的 engine 是 `ArithmeticTupleIterator`，不是内存指针。

因此：

```text
普通数据 Tensor 的 value = half/float 等数据
tAgA 坐标 Tensor 的 value = TMA global coordinate
```

### 28.3 完整打印结果怎样拆解

日志为：

```text
ArithTuple(128,_0) o
(((_64,_8),(_2,_8)),32):
(((_1@0,_1@1),(_64@0,_8@1)),_64@1)
```

`include/cute/tensor_impl.hpp:1117-1121`（函数 `cute::print(Tensor const&)`）的打印格式就是：

```cpp
print(tensor.data());
print(" o ");
print(tensor.layout());
```

所以三部分分别是：

```text
Engine 起点   = ArithTuple(128,_0)
Layout shape = (((64,8),(2,8)),32)
Layout stride= (((1@0,1@1),(64@0,8@1)),64@1)
```

这里的 `o` 表示“engine 按 layout 产生的 offset 取值”的 Tensor 打印形式；它不表示 `ArithTuple` 是 global-memory base pointer，也不表示这里一定存在一个 `ComposedLayout` 对象。

`128` 是运行时坐标值；带下划线的 `_0` 是 CuTe 编译期常量 `Int<0>`。数值上二者仍分别是 128 和 0。

### 28.4 stride 中的 `@` 是坐标基向量

普通 pointer Tensor 的 stride 通常是一个标量元素偏移，例如 `1`、`512`。`tAgA` 的 layout codomain 是二维 TMA 坐标 `(m,k)`，所以它的 stride 必须能表示“增加哪个坐标分量”。

`include/cute/numeric/arithmetic_tuple.hpp:216-258`（类型 `cute::ScaledBasis` 和别名 `cute::E`）把 `E<N>` 定义为 ArithmeticTuple 坐标空间第 N 维的基向量。`include/cute/numeric/arithmetic_tuple.hpp:463-475`（函数 `cute::print(ScaledBasis const&)`）用 `@N` 输出这个维度编号。

本例中的含义是：

| 打印形式 | 等价二维增量 | 含义 |
|---|---:|---|
| `_1@0` | `(1,0)` | M 坐标增加 1 |
| `_1@1` | `(0,1)` | K 坐标增加 1 |
| `_64@0` | `(64,0)` | M 坐标增加 64 |
| `_8@1` | `(0,8)` | K 坐标增加 8 |
| `_64@1` | `(0,64)` | 进入下一个 `BLK_K=64` tile |

`@` 因而不是地址空间标记，也不是字节单位，更不是某种 swizzle 符号；它表示 `ScaledBasis` 作用于 ArithmeticTuple 的哪一个坐标维度。`include/cute/numeric/arithmetic_tuple.hpp:394-430`（函数 `cute::operator*` 与 `cute::operator+`）负责缩放这些基向量并把它们相加成完整坐标增量。

### 28.5 shape 和 stride 的逐层含义

shape：

```text
(((_64,_8),(_2,_8)),32)
```

可读成：

```text
((TMA box, TMA iteration), K tile)

TMA box       = (64,8)  -> 512 个 half
TMA iteration = (2,8)   -> 一个 128x64 CTA tile 内重复 16 次
K tile        = 32      -> 2048 / 64 = 32
```

总逻辑大小为：

```text
(64 * 8) * (2 * 8) * 32 = 262144
```

这与日志中的 `size(tAgA)=262144` 一致。只选一个 `k_tile` 后，mode-0 大小是：

```text
512 * 16 = 8192 = 128 * 64
```

也就是当前 CTA 的完整 A tile。

该层次结构由 `include/cute/atom/copy_traits_sm90_tma.hpp:1395-1435`（函数 `cute::tma_partition`）产生：

- `1411-1418` 行按 shared-memory layout 重排 TMA mode，并使用 `Copy_Atom::NumValSrc` 分离一条 TMA 指令负责的部分；
- `1420-1425` 行把它组织成 `((TMA,TMA_Iter),Rest...)`；
- `1432-1435` 行应用当前 CTA/multicast offset 后返回 global-coordinate Tensor 和 shared-memory Tensor。

### 28.6 `tAgA.operator()` 仍然使用普通 Tensor 的实现

`tAgA` 没有专门重载另一套 `operator()`。`include/cute/tensor_impl.hpp:233-255`（成员函数 `cute::Tensor::operator()`）对所有 Tensor 都执行同一个核心表达式：

```cpp
return data()[layout()(coord)];
```

区别在 `data()` 的 engine 类型：

```text
普通 pointer Tensor:
  layout(coord) -> 标量物理元素偏移
  pointer[偏移] -> 解引用并返回内存元素

tAgA coordinate Tensor:
  layout(coord) -> ArithmeticTuple 坐标增量
  ArithmeticTupleIterator[增量] -> 返回“起点 + 增量”的坐标值
```

`include/cute/numeric/arithmetic_tuple.hpp:182-205`（成员函数 `cute::ArithmeticTupleIterator::operator[]`、`operator+` 和 `operator*`）表明，这个 iterator 的 `operator[]` 只是计算 `coord_ + offset` 并返回 ArithmeticTuple，不会执行 global-memory load。

所以 `tAgA(coord)` 的过程是：

```text
1. layout(coord) 产生二维坐标增量；
2. 增量加到 engine 起点 (128,0)；
3. 返回 ArithmeticTuple(m,k)。
```

返回类型显示成 `cute::ArithmeticTuple<int, unsigned int>`，只是因为动态 layout/index 运算最终为两个坐标分量推导出了 `int` 和 `unsigned int` 类型；它的逻辑语义仍是二维 `(m,k)` TMA coordinate。

### 28.7 用一个层次坐标直接计算

令符合 shape 的坐标为：

```text
coord = (((a,b),(c,d)),q)

0 <= a < 64
0 <= b < 8
0 <= c < 2
0 <= d < 8
0 <= q < 32
```

根据打印出的 stride：

```text
layout(coord)
  = a * (1,0)
  + b * (0,1)
  + c * (64,0)
  + d * (0,8)
  + q * (0,64)

  = (a + 64*c,
     b + 8*d + 64*q)
```

再加 engine 起点 `(128,0)`：

```text
tAgA(coord)
  = (128 + a + 64*c,
     b + 8*d + 64*q)
```

例如：

```cpp
auto coord = make_coord(
    make_coord(make_coord(5, 3), make_coord(1, 2)),
    4);
auto elem = tAgA(coord);
```

计算结果为：

```text
M coordinate = 128 + 5 + 64*1 = 197
K coordinate = 3 + 8*2 + 64*4 = 275

elem = (197,275)
```

这个 `(197,275)` 不是 `A[197,275]` 的 half value，而是交给 TMA descriptor 使用的全局逻辑坐标。

### 28.8 为什么 `tAgA(0..15)` 只打印 `(128..143,0)`

`include/cute/layout.hpp:161-172`（成员函数 `cute::Layout::operator()`）通过 `crd2idx` 处理传入坐标。传入标量 `i`、但 shape 是 tuple 时，`include/cute/stride.hpp:47-124`（函数 `cute::crd2idx`）按照各 mode 的 size 递归进行 div/mod，最左侧叶子 mode 变化最快。

把一个线性 `i` 分解到当前 shape：

```text
a =  i         % 64
b = (i /   64) % 8
c = (i /  512) % 2
d = (i / 1024) % 8
q = (i / 8192) % 32
```

于是：

```text
tAgA(i) = (128 + a + 64*c,
           b + 8*d + 64*q)
```

`examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:162-172`（函数 `gemm_device`）中的循环只打印 `i=0..15`。这些索引全部满足：

```text
a=i, b=0, c=0, d=0, q=0
```

所以日志自然是：

```text
tAgA(0)  = (128,0)
tAgA(1)  = (129,0)
...
tAgA(15) = (143,0)
```

这些输出证明最内层 `_64` mode 沿 M 坐标、以 `_1@0` 为 stride 变化。要观察其他 mode，应该查看跨越 mode 边界的索引：

| 线性索引 | 分解后的关键变化 | `tAgA(i)` |
|---:|---|---:|
| `0` | 起点 | `(128,0)` |
| `63` | 第一个 64-element M 段末尾 | `(191,0)` |
| `64` | `b:0 -> 1` | `(128,1)` |
| `511` | 第一个 64x8 TMA box 末尾 | `(191,7)` |
| `512` | `c:0 -> 1` | `(192,0)` |
| `1023` | 第二个 64x8 box 末尾 | `(255,7)` |
| `1024` | `d:0 -> 1` | `(128,8)` |
| `8191` | 当前 128x64 CTA tile 末尾 | `(255,63)` |
| `8192` | `q:0 -> 1`，下一个 K tile | `(128,64)` |

### 28.9 `copy` 实际怎样消费这些 ArithmeticTuple

copy 调用位于 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:215-224`（函数 `gemm_device`）：

```cpp
copy(tma_a.with(producer_mbar[pipe]),
     tAgA(_,k_tile),
     tAsA(_,pipe));
```

选择 `k_tile` 后，`tAgA(_,k_tile)` 包含当前 128x64 tile 的 8192 个逻辑坐标。它首先按 Atom 的 512-element 粒度拆为 16 个 source slice；分解过程见 `include/cute/atom/copy_atom.hpp:89-114`（成员函数 `cute::Copy_Atom::call`）和 `include/cute/algorithm/copy.hpp:184-235`（函数 `cute::copy(Copy_Atom const&, Tensor const&, Tensor&)`）。

这里有一个关键点：**底层并不会把 8192 个 ArithmeticTuple 全部作为参数传给一条 TMA 指令。**

对每个 512-element、shape 为 `(64,8)` 的 slice，`include/cute/atom/copy_traits_sm90_tma.hpp:64-90`（函数 `cute::copy_unpack`）只执行：

```cpp
auto src_coord = src(Int<0>{});
```

即只取这个 64x8 box 的第一个坐标，作为该 TMA box 的 global lower-corner coordinate。随后它调用 `explode_tuple`，把：

```text
traits.opargs_ = (TMA descriptor pointer, mbarrier pointer, cache hint)
destination    = shared-memory pointer
src_coord      = (m,k)
```

展开成底层 copy 参数。tuple 展开实现见 `include/cute/arch/util.hpp:282-315`（函数 `cute::detail::explode_tuple`），而 `include/cute/arch/util.hpp:149-160`（成员函数 `cute::detail::CallCOPY<CopyOp>::operator()`）再调用 `CopyOp::copy(...)`。

二维 coordinate 最终选择 `include/cute/arch/copy_sm90_tma.hpp:327-342`（函数 `cute::SM90_TMA_LOAD::copy`）的 2D overload，并进入 `include/cute/arch/copy_sm90_tma.hpp:103-135`（函数 `cute::SM90_TMA_LOAD_2D::copy`）：

```ptx
cp.async.bulk.tensor.2d.shared::cluster.global
    .mbarrier::complete_tx::bytes.L2::cache_hint
    [smem_ptr], [tensor_map, {crd0, crd1}], [mbarrier], cache_hint;
```

### 28.10 当前 tile 的 16 个 TMA 起始坐标

对第 `j` 个 512-element TMA slice：

```text
c = j % 2
d = j / 2
```

所以第 `q=k_tile` 个 K tile 中，16 个 box 的起始坐标依次为：

```text
j= 0: (128,  0 + 64*q)
j= 1: (192,  0 + 64*q)
j= 2: (128,  8 + 64*q)
j= 3: (192,  8 + 64*q)
...
j=14: (128, 56 + 64*q)
j=15: (192, 56 + 64*q)
```

每个起点配合 TMA descriptor 中的 64x8 box 定义，使硬件搬运 `64*8=512` 个 half，即 1024 bytes。16 个 box 合起来正好覆盖 CTA 的 128x64 A tile。

TMA descriptor 已在 `make_tma_atom` 阶段记录原始 A global-memory base address、global shape/stride、box shape、元素类型和 shared-memory swizzle。相关构造见 `include/cute/atom/copy_traits_sm90_tma.hpp:917-952`（函数 `cute::detail::make_tma_copy_desc`）以及 `include/cute/atom/copy_traits_sm90_tma.hpp:1025-1058`（函数 `cute::detail::make_tma_copy_desc`）。

因此底层 TMA source operand 的逻辑模型是：

```text
(opaque TensorMap descriptor, box 起始坐标)
```

而不是：

```text
普通 load 得到的一组 half value
```

数据仍然是原始 A Tensor 中的 half；ArithmeticTuple 只是告诉 TMA 硬件“在 descriptor 描述的全局 Tensor 中，从哪个 `(m,k)` 坐标开始异步搬运”。用 coordinate Tensor 表示 source，使 CuTe 可以继续使用统一的 layout、tile、partition 和 copy 代数，同时避免软件线程先把 global-memory 数据逐元素读进寄存器。

例如，对本例的 column-major A layout `(M,K):(1,512)`，TMA 收到坐标 `(197,275)` 后，概念上会从下面的地址开始取数：

```text
A element offset = 197 + 275 * 512
A byte address    = A_base + 2 * (197 + 275 * 512)
```

这个地址计算和后续 64x8 bulk load 由 TensorMap descriptor 与 TMA 硬件完成；`tAgA(…)` 本身只产生 `(197,275)`，既不计算/返回该物理指针，也不读取该地址处的 half value。

## 29. A 矩阵 TMA descriptor 的生成与初始化函数栈

### 29.1 先给结论：这个示例确实调用 `cuTensorMapEncodeTiled`

对于命令：

```text
./build-bjh100/examples/cute/tutorial/hopper/cute_tutorial_wgmma_tma_sm90_like 512 1024 2048 N T
```

A 的 TMA descriptor 在 host 端执行 `gemm_nt` 时生成，发生在 kernel launch 之前；它不是在 `gemm_device` 中逐 CTA、逐 pipe 或逐 K tile 生成的。

调用路径的核心部分是：

```text
main
  -> gemm
    -> gemm_nt
      -> make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(128,64))
        -> detail::make_tma_copy_atom
          -> detail::construct_tma_gbasis
          -> detail::make_tma_copy_desc
            -> CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)
              -> cutlass::call_cuTensorMapEncodeTiled
                -> CUDA Driver API cuTensorMapEncodeTiled
```

`include/cute/atom/copy_traits_sm90_tma.hpp:1035-1058`（函数 `cute::detail::make_tma_copy_desc`）是本仓库中实际出现 `cuTensorMapEncodeTiled` 的位置。普通 CUDA 12+、非 `__CUDACC_RTC__` 的构建会走这段 host 代码；如果只做 device runtime compilation 或使用不满足版本条件的编译配置，这个 API 分支会被预处理器排除。

### 29.2 A descriptor 生成的入口参数

`main` 解析 `512 1024 2048 N T`，见 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:560-597`（函数 `main`）；`N,T` 由 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:540-557`（函数 `gemm`）分派到 `gemm_nt`。

在 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:339-379`（函数 `gemm_nt`）中，A 的关键对象是：

```cpp
M = 512;
N = 1024;
K = 2048;
dA = make_stride(Int<1>{}, ldA);  // ldA = M = 512
mA = make_tensor(A, make_shape(M,K), dA);
sA = tile_to_shape(
       GMMA::Layout_MN_SW128_Atom<TA>{},
       make_shape(bM,bK,bP));       // bM=128,bK=64,bP=3
tmaA = make_tma_atom(
       SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
```

这里 `mA` 的逻辑 shape/stride 是：

```text
global shape  = (512, 2048)
global stride = (1, 512) half elements
```

`sA(_,_,0)` 选择 pipeline 的第 0 个 shared-memory stage；它的 TMA box 设计目标是一个 `128x64` A tile，而不是把 3 个 pipeline stage 一起编码进一个 descriptor。三个 stage 使用同一个 A descriptor，只改变 copy 的 shared-memory destination 和 barrier。

### 29.3 `make_tma_atom` 到 descriptor 编码的详细栈

#### 29.3.1 `gemm_nt -> make_tma_atom`

入口调用见 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:372-379`（函数 `gemm_nt`）：

```cpp
Copy_Atom tmaA = make_tma_atom(
    SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
```

`include/cute/atom/copy_traits_sm90_tma.hpp:1373-1393`（函数 `cute::make_tma_atom`）做两件事：

1. 用 `make_identity_layout(shape(gtensor)).compose(cta_tiler)` 生成 CTA value-to-global-mode 映射 `cta_v_tile`；
2. 将 `TmaInternalType` 推导为 `GEngine::value_type`。本例 A 是 `half_t`，然后调用 `detail::make_tma_copy_atom`。

#### 29.3.2 `make_tma_copy_atom` 拆出 shared-memory swizzle 和 global-mode basis

`include/cute/atom/copy_traits_sm90_tma.hpp:1131-1178`（函数 `cute::detail::make_tma_copy_atom`）的顺序是：

```cpp
auto smem_swizzle = get_swizzle_portion(slayout);
auto smem_layout  = get_nonswizzle_portion(slayout);
auto tma_gbasis = detail::construct_tma_gbasis(
    gtensor, smem_layout, cta_v_map);
auto [tma_desc, aux_params] = detail::make_tma_copy_desc(
    gtensor, tma_gbasis, smem_swizzle, num_multicast);
```

本例没有显式传 multicast 参数，`make_tma_atom` 的默认 `cluster_size` 是 1，因此 `num_multicast=size(cluster_size)=1`，descriptor 的 box 不会被 multicast 再切分。

#### 29.3.3 `construct_tma_gbasis` 如何决定 TMA box

`include/cute/atom/copy_traits_sm90_tma.hpp:728-849`（函数 `cute::detail::construct_tma_gbasis`）把 shared-memory layout 反向映射到 A 的 global modes：

```text
get_nonswizzle_portion(slayout)
  -> right_inverse                 // SMEM index -> SMEM coordinate
  -> composition(cta_v_map, ...)   // SMEM coordinate -> A 的 M/K mode
  -> recast<TmaInternalType>        // 按 half 元素单位观察
  -> coalesce_256                   // 合并到 TMA 单维最大 256 元素约束
  -> make_layout / group            // 形成 tma_gbasis
```

对当前 A layout，日志中 `tmaA` 的 `ValLayoutSrc=(_1,_512)`，并且 `tAgA` 的 mode-0 shape 是 `((64,8),(2,8))`。这说明 `tma_gbasis` 的一条指令 box 是 `64x8`，即 `512` 个 half；一个 `128x64` CTA tile 由 16 个这样的 box 组成。

这个阶段只构造静态的“box shape -> A 的 M/K mode”映射，不访问 A 的数据，也不调用 CUDA driver API。

### 29.4 `make_tma_copy_desc` 如何填充 descriptor 输入

`include/cute/atom/copy_traits_sm90_tma.hpp:922-1128`（函数 `cute::detail::make_tma_copy_desc`）先从原始 global-memory Tensor 收集地址、全局 shape/stride，再准备 shared-memory box 参数。

#### 29.4.1 global address、shape 和 stride

关键代码在 `include/cute/atom/copy_traits_sm90_tma.hpp:943-952`（函数 `cute::detail::make_tma_copy_desc`）：

```cpp
Tensor gtensor_T = recast<TmaInternalType>(gtensor);
void* gmem_address = (void*) raw_pointer_cast(gtensor_T.data());

gmem_prob_shape  = {1,1,1,1,1};
gmem_prob_stride = {0,0,0,0,0};
fill_tma_gmem_shape_stride(
    gtensor_T, stride(tma_gbasis),
    gmem_prob_shape, gmem_prob_stride);
```

`include/cute/atom/copy_traits_sm90_tma.hpp:852-900`（函数 `cute::fill_tma_gmem_shape_stride`）通过 `tma_gbasis` 把原始 A Tensor 的 shape/stride 映射到 TMA 维度。对本例可读成：

```text
TMA rank             = 2
globalDim            = (512, 2048)
global stride        = (1, 512) half elements
global stride bytes  = (2, 1024)
```

代码在 `include/cute/atom/copy_traits_sm90_tma.hpp:967-983`（函数 `cute::detail::make_tma_copy_desc`）说明 descriptor 使用 byte stride，并且第 0 维 stride 在 TensorMap API 中隐含为一个 `TmaInternalType` 元素。调用 API 时传的是 `gmem_prob_stride.data()+1`，所以本例实际显式传入的非主维 byte stride 是 `1024`。

#### 29.4.2 shared-memory box 和 swizzle

`include/cute/atom/copy_traits_sm90_tma.hpp:989-1023`（函数 `cute::detail::make_tma_copy_desc`）根据 `tma_gbasis` 设置：

```text
smem_box_shape  = (64,8)
smem_box_stride = (1,1)
num_multicast   = 1
```

`sA` 的 swizzle 是日志中显示的 `Sw<3,4,3>`。`include/cute/atom/copy_traits_sm90_tma_swizzle.hpp:45-85`（函数 `cute::detail::get_tma_swizzle_bits` 和 `cute::detail::get_tma_swizzle_base`）把它转换为：

```text
SmemSwizzleBits = B128
SmemSwizzleBase = SWIZZLE_BASE_16B
```

随后 `include/cute/arch/copy_sm90_desc.hpp:239-263`（函数 `cute::TMA::to_CUtensorMapSwizzle`）映射为 `CU_TENSOR_MAP_SWIZZLE_128B`。

### 29.5 实际调用 `cuTensorMapEncodeTiled` 的位置和参数

descriptor 的存储对象先在 `include/cute/atom/copy_traits_sm90_tma.hpp:1025-1029`（函数 `cute::detail::make_tma_copy_desc`）创建：

```cpp
TmaDescriptor tma_desc{};
```

在 CUDA 12+、非 `__CUDACC_RTC__` 配置下，`TmaDescriptor` 是 `CUtensorMap` 的别名，见 `include/cute/arch/copy_sm90_desc.hpp:291-297`（类型别名 `cute::TmaDescriptor`）。它是一个由 CUDA 定义的 opaque TensorMap 对象，不是 CuTe 自己逐字段填写的普通结构体。

随后 `include/cute/atom/copy_traits_sm90_tma.hpp:1035-1058`（函数 `cute::detail::make_tma_copy_desc`）准备并传入：

```cpp
CUtensorMapDataType   tma_format     = TMA::to_CUtensorMapDataType<TmaInternalType>();
CUtensorMapInterleave tma_interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
CUtensorMapL2promotion tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
CUtensorMapFloatOOBfill tma_oobFill   = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

CUtensorMapSwizzle smem_swizzle = TMA::to_CUtensorMapSwizzle(...);

cuTensorMapEncodeTiled(
    &tma_desc,
    tma_format,
    tma_dim,
    gmem_address,
    gmem_prob_shape.data(),
    gmem_prob_stride.data() + 1,
    smem_box_shape.data(),
    smem_box_stride.data(),
    tma_interleave,
    smem_swizzle,
    tma_l2Promotion,
    tma_oobFill);
```

对 A，本次调用的核心值是：

| 参数 | 本例含义 |
|---|---|
| `&tma_desc` | 输出 descriptor 的地址 |
| `tma_format` | `CU_TENSOR_MAP_DATA_TYPE_FLOAT16` |
| `tma_dim` | `2` |
| `gmem_address` | A 的 device-memory base address |
| `globalDim` | `(512,2048)` |
| `globalStrides` | 显式 byte stride `1024`；第 0 维隐含为 1 个 half |
| `boxDim` | `(64,8)` half |
| `elementStrides` | `(1,1)` |
| `interleave` | `NONE` |
| `swizzle` | `128B` |
| `L2 promotion` | `L2_128B` |
| `OOB fill` | `NONE` |

`cuTensorMapEncodeTiled` 的作用是把这些可读参数编码成硬件使用的 TensorMap descriptor 位域；它不搬运 A 数据。返回的 `CUresult` 在 `include/cute/atom/copy_traits_sm90_tma.hpp:1071-1086`（函数 `cute::detail::make_tma_copy_desc`）中检查，失败时打印 descriptor 参数并触发断言。

编码成功后，`include/cute/atom/copy_traits_sm90_tma.hpp:1060-1068`（函数 `cute::detail::make_tma_copy_desc`）还会查询 driver 版本，并针对旧 driver 的特定兼容性条件修改 descriptor 的一个 bit。这个 A Tensor 的 footprint 是 `512*2048*2 = 2 MiB`，大于该分支判断的 `131072` bytes，因此本次运行不会进入实际 bit 修改；它属于编码后的兼容性后处理，而不是重新生成 descriptor。

### 29.6 Driver wrapper 这一层的函数栈

源码表面写的是宏：

```cpp
CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(...);
```

`include/cutlass/cuda_host_adapter.hpp:149-156`（宏 `CUTLASS_CUDA_DRIVER_WRAPPER_DECL` 和 `CUTLASS_CUDA_DRIVER_WRAPPER_CALL`）把它展开为：

```text
cutlass::call_cuTensorMapEncodeTiled(args...)
```

具体 wrapper 取决于编译配置：

```text
CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL:
  call_cuTensorMapEncodeTiled -> cuTensorMapEncodeTiled(args...)

非 direct、CUDA <= 12:
  call_cuTensorMapEncodeTiled
    -> cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", ...)
    -> 取得函数指针并调用

非 direct、CUDA > 12:
  call_cuTensorMapEncodeTiled
    -> cudaGetDriverEntryPointByVersion(..., 12000, ...)
    -> 取得函数指针并调用
```

这些分支的宏实现位于 `include/cutlass/cuda_host_adapter.hpp:97-145`（宏 `CUTLASS_CUDA_DRIVER_WRAPPER_DECL`）。因此，虽然源文件没有直接写裸的 `cuTensorMapEncodeTiled(...)` 函数调用，运行时仍会通过 CUTLASS wrapper 到达 CUDA Driver API。

### 29.7 descriptor 编码完成后的 traits/Copy Atom 初始化

`cuTensorMapEncodeTiled` 返回后，`make_tma_copy_desc` 还会计算 TMA 坐标 Tensor 所需的 global basis stride，见 `include/cute/atom/copy_traits_sm90_tma.hpp:1092-1128`（函数 `cute::detail::make_tma_copy_desc`）。这部分生成的是 `AuxTmaParams::g_stride_` 等辅助元数据，不会重新编码 `tma_desc`。

然后回到 `include/cute/atom/copy_traits_sm90_tma.hpp:1162-1178`（函数 `cute::detail::make_tma_copy_atom`）：

```cpp
constexpr int num_bits_per_tma = size(tma_gbasis) * sizeof_bits_v<TmaInternalType>();
using Traits = Copy_Traits<CopyOp, C<num_bits_per_tma>, decltype(aux_params)>;
using Atom   = Copy_Atom<Traits, typename GEngine::value_type>;

Traits tma_traits{tma_desc, aux_params};
return Atom{tma_traits};
```

本例 `size(tma_gbasis)=64*8=512`、元素是 half，所以：

```text
num_bits_per_tma = 512 * 16 = 8192 bits
```

这里发生的是 C++ 对象初始化/拷贝：

```text
tma_desc (CUtensorMap bits)
  -> Copy_Traits<SM90_TMA_LOAD,...>::tma_desc_
  -> tma_traits
  -> Copy_Atom tmaA
```

`include/cute/atom/copy_traits_sm90_tma.hpp:99-122`（类型 `cute::Copy_Traits<SM90_TMA_LOAD,...>` 的成员函数 `get_tma_descriptor`）显示 non-executable traits 持有 descriptor 对象；它还持有 `AuxTmaParams`。这一步不是再次调用 `cuTensorMapEncodeTiled`。

### 29.8 kernel launch 时 descriptor 如何传递和初始化为 kernel 参数

`gemm_nt` 在 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:410-415`（函数 `gemm_nt`）用 `decltype(tmaA)` 实例化 `gemm_device` 的 kernel 指针；随后在 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:445-450`（函数 `gemm_nt`）把 `tmaA` 作为 kernel 参数传入：

```cpp
launch_kernel_on_cluster(params, kernel_ptr,
                         prob_shape, cta_tiler,
                         A, tmaA,
                         B, tmaB, ...);
```

`include/cutlass/cluster_launch.hpp:362-392`（函数 `cutlass::launch_kernel_on_cluster`）为每个 host 参数取得地址，组装 `kernel_params`；`include/cutlass/cluster_launch.hpp:215-249`（函数 `cutlass::ClusterLauncher::launch`）最终调用 `cudaLaunchKernelExC`。

kernel 端的参数声明在 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:66-78`（函数 `gemm_device`）：

```cpp
TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
```

`CUTLASS_GRID_CONSTANT` 在 `include/cutlass/device_kernel.h:41-56`（宏 `CUTLASS_GRID_CONSTANT`）通常展开为 CUDA 的 `__grid_constant__`（满足版本和架构条件时）。因此 descriptor 是随 `tmaA` 作为只读 kernel 参数传入的；每个 CTA 复用同一份编码后的 TensorMap，不会为 `(blockIdx.x,blockIdx.y)` 重新调用编码 API。

### 29.9 “初始化 descriptor”和“使用 descriptor”不是同一件事

这个示例中没有另一个名为 `init_tma_descriptor` 的函数。可以按下面三层理解“初始化”：

1. **存储初始化**：`TmaDescriptor tma_desc{}`，见 `include/cute/atom/copy_traits_sm90_tma.hpp:1025-1029`（函数 `cute::detail::make_tma_copy_desc`）；
2. **硬件格式初始化/编码**：`cuTensorMapEncodeTiled(&tma_desc, ...)`，见 `include/cute/atom/copy_traits_sm90_tma.hpp:1035-1058`（函数 `cute::detail::make_tma_copy_desc`）；
3. **对象和 kernel 参数初始化**：`Traits tma_traits{tma_desc, aux_params}`、`Atom{tma_traits}`，再由 `launch_kernel_on_cluster` 复制到 `gemm_device` 的 `tma_a` 参数，见上述 `make_tma_copy_atom`、`launch_kernel_on_cluster` 和 `gemm_device` 引用。

而下面几步不是 descriptor 初始化：

```text
tma_a.get_tma_tensor(...)      -> 只生成坐标 Tensor
tma_a.with(producer_mbar[pipe]) -> 只绑定 mbarrier/cache hint，生成 executable Copy Atom
copy(...)                        -> 使用已有 descriptor 发出 TMA load
```

`get_tma_tensor` 的实现见 `include/cute/atom/copy_traits_sm90_tma.hpp:147-154`（成员函数 `cute::Copy_Traits<SM90_TMA_LOAD,...>::get_tma_tensor`）；`.with` 的实现见 `include/cute/atom/copy_traits_sm90_tma.hpp:124-133`（成员函数 `cute::Copy_Traits<SM90_TMA_LOAD,...>::with`）。本示例也没有调用 `prefetch_tma_descriptor`；该函数是可选的 descriptor 预取工具，不负责生成或编码 descriptor，定义见 `include/cute/arch/copy_sm90_desc.hpp:299-315`（函数 `cute::prefetch_tma_descriptor`）。

### 29.10 A 矩阵完整函数栈汇总

#### 生成并编码 descriptor

```text
main
  -> gemm
    -> gemm_nt
      -> make_tma_atom
        -> detail::make_tma_copy_atom
          -> detail::construct_tma_gbasis
            -> get_nonswizzle_portion / right_inverse / composition
            -> recast / coalesce_256 / group
          -> detail::make_tma_copy_desc
            -> recast / raw_pointer_cast
            -> fill_tma_gmem_shape_stride
            -> get_tma_swizzle_bits / get_tma_swizzle_base
            -> to_CUtensorMapDataType
            -> to_CUtensorMapSwizzle
            -> cutlass::call_cuTensorMapEncodeTiled
              -> CUDA Driver API cuTensorMapEncodeTiled
```

#### 封装、传递并使用 descriptor

```text
detail::make_tma_copy_desc
  -> detail::make_tma_copy_atom
    -> Copy_Traits<SM90_TMA_LOAD>::tma_desc_
    -> Copy_Atom tmaA
  -> gemm_nt
    -> cutlass::launch_kernel_on_cluster
      -> ClusterLauncher::launch
        -> cudaLaunchKernelExC
          -> gemm_device(..., TmaA const tma_a, ...)
            -> tma_a.with(producer_mbar[pipe])
              -> copy_unpack
                -> SM90_TMA_LOAD_2D::copy
                  -> cp.async.bulk.tensor.2d...
```

最重要的时序是：descriptor 只在 host 端编码一次；kernel 内的 `tAgA` 只生成坐标，TMA 指令使用同一 descriptor 加不同坐标和 shared-memory destination 来加载各 CTA/K tile 的 A 数据。

## 30. `CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)` 如何展开

### 30.1 第一层展开：`##` 拼接出 wrapper 函数名

调用位置在 `include/cute/atom/copy_traits_sm90_tma.hpp:1046-1058`（函数 `cute::detail::make_tma_copy_desc`）：

```cpp
CUresult result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &tma_desc,
    tma_format,
    tma_dim,
    gmem_address,
    gmem_prob_shape.data(),
    gmem_prob_stride.data() + 1,
    smem_box_shape.data(),
    smem_box_stride.data(),
    tma_interleave,
    smem_swizzle,
    tma_l2Promotion,
    tma_oobFill);
```

宏定义位于 `include/cutlass/cuda_host_adapter.hpp:154-156`（宏 `CUTLASS_CUDA_DRIVER_WRAPPER_CALL`）：

```cpp
#define CUTLASS_CUDA_DRIVER_WRAPPER_CALL(func) cutlass::call_##func
```

预处理器先将形参 `func` 替换为实参 token `cuTensorMapEncodeTiled`：

```text
cutlass::call_##func
  -> cutlass::call_##cuTensorMapEncodeTiled
```

`##` 是 C/C++ 预处理器的 token-pasting 运算符，将左右两个 preprocessing token 拼成一个新 token：

```text
call_ ## cuTensorMapEncodeTiled
  -> call_cuTensorMapEncodeTiled
```

所以第一层的准确展开结果是：

```cpp
CUresult result = cutlass::call_cuTensorMapEncodeTiled(
    &tma_desc,
    tma_format,
    tma_dim,
    gmem_address,
    gmem_prob_shape.data(),
    gmem_prob_stride.data() + 1,
    smem_box_shape.data(),
    smem_box_stride.data(),
    tma_interleave,
    smem_swizzle,
    tma_l2Promotion,
    tma_oobFill);
```

注意，`CUTLASS_CUDA_DRIVER_WRAPPER_CALL` 的替换文本本身不包含函数调用括号。原始源码紧跟在宏调用后的 `(...)` 被保留下来，和宏生成的函数名共同组成正常的 C++ 函数调用。

### 30.2 `call_cuTensorMapEncodeTiled` 在哪里定义

它不是手写的普通函数，而是下面这行继续用另一个宏生成的，见 `include/cutlass/cuda_host_adapter.hpp:149-152`（宏调用 `CUTLASS_CUDA_DRIVER_WRAPPER_DECL`）：

```cpp
CUTLASS_CUDA_DRIVER_WRAPPER_DECL(cuTensorMapEncodeTiled, 12000);
```

这里：

```text
func = cuTensorMapEncodeTiled
ver  = 12000
```

`CUTLASS_CUDA_DRIVER_WRAPPER_DECL` 有三个条件编译版本：

```text
1. CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL 已定义
2. 非 direct，并且 CUDA major > 12
3. 非 direct，并且 CUDA major <= 12
```

因此，`WRAPPER_CALL` 只负责选择统一名字 `cutlass::call_cuTensorMapEncodeTiled`；这个名字背后的函数体是在 `WRAPPER_DECL` 展开时由编译配置决定的。

### 30.3 当前 CUDA 13 构建实际生成的函数

当前二进制导入：

```text
cudaGetDriverEntryPointByVersion@libcudart.so.13
```

而没有直接导入 `cuTensorMapEncodeTiled`，所以它使用的是：

```text
未定义 CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL
__CUDACC_VER_MAJOR__ > 12
```

对应宏体位于 `include/cutlass/cuda_host_adapter.hpp:107-124`（宏 `CUTLASS_CUDA_DRIVER_WRAPPER_DECL`，生成函数 `cutlass::call_cuTensorMapEncodeTiled`）：

```cpp
#define CUTLASS_CUDA_DRIVER_WRAPPER_DECL(func, ver)             \
  template <typename... Args>                                   \
  CUresult call_##func(Args... args) {                          \
    cudaDriverEntryPointQueryResult cuda_status;                \
    void* pfn = nullptr;                                        \
    cudaError_t cuda_err = cudaGetDriverEntryPointByVersion(    \
        CUTLASS_CUDA_DRIVER_STRINGIFY(func),                    \
        &pfn, ver,                                              \
        cudaEnableDefault,                                      \
        &cuda_status);                                          \
    if (cuda_status != cudaDriverEntryPointSuccess ||           \
        cuda_err != cudaSuccess) {                              \
      return CUDA_ERROR_UNKNOWN;                                \
    }                                                           \
    return reinterpret_cast<PFN_##func##_v##ver>(pfn)(args...); \
  }
```

代入 `func=cuTensorMapEncodeTiled`、`ver=12000` 后，等价代码是：

```cpp
template <typename... Args>
CUresult call_cuTensorMapEncodeTiled(Args... args) {
  cudaDriverEntryPointQueryResult cuda_status;
  void* pfn = nullptr;

  cudaError_t cuda_err = cudaGetDriverEntryPointByVersion(
      "cuTensorMapEncodeTiled",
      &pfn,
      12000,
      cudaEnableDefault,
      &cuda_status);

  if (cuda_status != cudaDriverEntryPointSuccess ||
      cuda_err != cudaSuccess) {
    return CUDA_ERROR_UNKNOWN;
  }

  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(pfn)(
      args...);
}
```

这里发生了三种预处理操作：

| 原宏表达式 | 展开结果 | 操作 |
|---|---|---|
| `call_##func` | `call_cuTensorMapEncodeTiled` | `##` 拼接函数名 |
| `CUTLASS_CUDA_DRIVER_STRINGIFY(func)` | `"cuTensorMapEncodeTiled"` | 内层 `#tok` 字符串化 |
| `PFN_##func##_v##ver` | `PFN_cuTensorMapEncodeTiled_v12000` | 多次 `##` 拼接函数指针类型名 |

字符串化宏定义在 `include/cutlass/cuda_host_adapter.hpp:93-95`（宏 `CUTLASS_CUDA_DRIVER_STRINGIFY`）：

```cpp
#define CUTLASS_CUDA_DRIVER_STRINGIFY(tok) #tok
```

`12000` 表示 wrapper 请求 CUDA 12.0 版本的 Driver API 入口，并不表示当前驱动版本必须恰好等于 12000。CUDA 13 runtime 通过 `cudaGetDriverEntryPointByVersion` 返回兼容的函数地址到 `pfn`。

`PFN_cuTensorMapEncodeTiled_v12000` 不是 CUTLASS 自己声明的类型。`include/cutlass/cuda_host_adapter.hpp:86-93`（生成函数 `cutlass::call_cuTensorMapEncodeTiled` 所在的条件编译区域）包含 CUDA 的 `cudaTypedefs.h`；该 CUDA header 提供相应 Driver API 函数指针 typedef，wrapper 用它把无类型的 `void* pfn` 转回正确签名。

### 30.4 运行时真正发生的调用链

经过预处理和模板实例化后，`make_tma_copy_desc` 中这一行的运行时调用链是：

```text
cute::detail::make_tma_copy_desc
  -> cutlass::call_cuTensorMapEncodeTiled(actual arguments...)
    -> cudaGetDriverEntryPointByVersion(
         "cuTensorMapEncodeTiled", &pfn, 12000, ...)
      -> 返回实际 Driver API 函数地址
    -> reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(pfn)
    -> 通过函数指针调用真正的 cuTensorMapEncodeTiled
      -> 将编码结果写入 &tma_desc
      -> 返回 CUresult
  -> 将 CUresult 赋给局部变量 result
```

`Args...` 是函数模板参数包，由 `include/cute/atom/copy_traits_sm90_tma.hpp:1046-1058`（函数 `cute::detail::make_tma_copy_desc`）传入的 12 个实参自动推导。wrapper 不修改参数含义，只把它们原样通过 `args...` 转发给取得的 Driver API 函数指针。

如果入口查询失败，wrapper 返回 `CUDA_ERROR_UNKNOWN`；如果查询成功，返回真正 `cuTensorMapEncodeTiled` 的 `CUresult`。调用方随后在 `include/cute/atom/copy_traits_sm90_tma.hpp:1071-1086`（函数 `cute::detail::make_tma_copy_desc`）检查 `result != CUDA_SUCCESS` 并报告错误。

### 30.5 另外两个编译分支怎样展开

#### 30.5.1 Direct Driver Call

如果定义了 `CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL`，会选择 `include/cutlass/cuda_host_adapter.hpp:97-103`（宏 `CUTLASS_CUDA_DRIVER_WRAPPER_DECL`，生成函数 `cutlass::call_cuTensorMapEncodeTiled`）：

```cpp
template <typename... Args>
CUresult call_cuTensorMapEncodeTiled(Args... args) {
  return cuTensorMapEncodeTiled(args...);
}
```

此时不通过 `cudaGetDriverEntryPoint*` 查询函数指针，而是直接链接并调用 Driver API。

#### 30.5.2 非 direct、CUDA 12

CUDA major 不大于 12 时，会选择 `include/cutlass/cuda_host_adapter.hpp:126-143`（宏 `CUTLASS_CUDA_DRIVER_WRAPPER_DECL`，生成函数 `cutlass::call_cuTensorMapEncodeTiled`）：

```cpp
template <typename... Args>
CUresult call_cuTensorMapEncodeTiled(Args... args) {
  cudaDriverEntryPointQueryResult cuda_status;
  void* pfn = nullptr;

  cudaError_t cuda_err = cudaGetDriverEntryPoint(
      "cuTensorMapEncodeTiled",
      &pfn,
      cudaEnableDefault,
      &cuda_status);

  if (cuda_status != cudaDriverEntryPointSuccess ||
      cuda_err != cudaSuccess) {
    return CUDA_ERROR_UNKNOWN;
  }

  return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(pfn)(args...);
}
```

三个分支生成的 wrapper 名字完全相同，因此调用处无需根据 CUDA 版本编写不同代码。

### 30.6 完整展开过程汇总

可以把编译过程压缩为下面四步：

```text
步骤 1：生成 wrapper 声明/定义

CUTLASS_CUDA_DRIVER_WRAPPER_DECL(cuTensorMapEncodeTiled, 12000)
  -> template <typename... Args>
     CUresult cutlass::call_cuTensorMapEncodeTiled(Args... args) { ... }

步骤 2：展开调用宏

CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)
  -> cutlass::call_##cuTensorMapEncodeTiled
  -> cutlass::call_cuTensorMapEncodeTiled

步骤 3：保留原调用括号和参数

cutlass::call_cuTensorMapEncodeTiled(&tma_desc, ..., tma_oobFill)

步骤 4：当前 CUDA 13 wrapper 在运行时查询并调用 Driver API

cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", ..., 12000, ...)
  -> PFN_cuTensorMapEncodeTiled_v12000
  -> cuTensorMapEncodeTiled(&tma_desc, ...)
```

因此，`CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)` 最直接的答案就是：它先展开成 `cutlass::call_cuTensorMapEncodeTiled`；当前构建中的这个 wrapper 再动态查询并调用真正的 `cuTensorMapEncodeTiled`。

## 31. `CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL` 在哪里启用或关闭

### 31.1 当前源码的直接结论

当前工作树中，这个名字只在 `include/cutlass/cuda_host_adapter.hpp:97-147`（宏 `CUTLASS_CUDA_DRIVER_WRAPPER_DECL` 的条件编译区域）出现；当前顶层 `CMakeLists.txt` 没有再声明同名的 CMake option。

当前判断规则是：

```cpp
#if defined(CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL)
  // 直接调用 cuTensorMapEncodeTiled
#else
  // 通过 cudaGetDriverEntryPoint* 查询函数指针
#endif
```

宏只检查 `defined(...)`，不检查宏值是否为 0：

```text
没有 -D                                  -> 关闭 direct call
-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL -> 启用 direct call
-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1 -> 启用 direct call
-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=0 -> 仍然启用，因为宏已定义
```

要关闭它，必须移除这个 `-D`，或者在包含 `cuda_host_adapter.hpp` 之前明确 `#undef`；仅仅改成 `=0` 不会关闭分支。

### 31.2 历史上的 CMake option

在较早版本中，顶层 `CMakeLists.txt:237`（顶层 CMake 配置语句，无 CMake 函数；历史提交 `4dbf5dbe`）曾有：

```cmake
set(CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL OFF CACHE BOOL
    "Enable CUTLASS to directly call driver API.")
```

旧版本可以使用：

```bash
cmake -S . -B build \
  -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=ON
```

但该 option 在后续 CUTLASS 3.6 更新中被移除；当前 `CMakeLists.txt` 不再把这个 cache 变量转换成编译器的 `-D`。所以在当前 checkout 中单独执行上面的 `-D...=ON`，不能保证预处理器看到该宏。

### 31.3 当前版本如何启用

#### 方法一：通过 `CMAKE_CUDA_FLAGS`

这是当前版本最直接的全局方式：

```bash
cmake -S . -B build-direct \
  -DCMAKE_CUDA_FLAGS=-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL
cmake --build build-direct --target cute_tutorial_wgmma_tma_sm90_like -j
```

`CMAKE_CUDA_FLAGS` 是 CMake 的 CUDA 编译器选项；当前 cache 中对应字段见 `build-bjh100/CMakeCache.txt:42-54`（CMake cache 配置），CUTLASS 会将这些选项传给 CUDA 编译。若已有 CUDA flags，应在原值后追加该 `-D`，不要覆盖原有选项。

#### 方法二：通过 CUTLASS 的 CUDA flags 列表

当前顶层 `CMakeLists.txt:697-737`（函数 `cutlass_apply_standard_compile_options`）先缓存 `CUTLASS_CUDA_FLAGS`，再把它作为 CUDA target 的 compile options。可在缓存快照之前加入：

```cmake
list(APPEND CUTLASS_CUDA_FLAGS
     -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL)
```

放置位置必须早于 `CMakeLists.txt:699-703`（缓存 `__CUTLASS_CUDA_FLAGS` 的语句）；否则已经创建的内部缓存列表不会自动更新。

#### 方法三：只对某个 example target 添加

Hopper tutorial 在 `examples/cute/tutorial/hopper/CMakeLists.txt:35-42`（函数 `cutlass_example_add_executable` 的调用处）注册目标。若只想改变这个 example，可以在目标创建后添加：

```cmake
target_compile_options(
  cute_tutorial_wgmma_tma_sm90_like PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL>)
```

`CUDA.cmake:326-357`（函数 `cutlass_add_executable`）创建 CUDA executable，并在 `CUDA.cmake:344-347` 调用 `cutlass_apply_standard_compile_options`；target-specific option 可在此基础上叠加。

### 31.4 当前 `build-bjh100` 是关闭状态

当前生成的目标 flags 位于 `build-bjh100/examples/cute/tutorial/hopper/CMakeFiles/cute_tutorial_wgmma_tma_sm90_like.dir/flags.make:4-9`（生成的 CUDA target 编译配置）：

```text
CUDA_DEFINES =
CUDA_FLAGS = ... -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 ...
```

其中没有 `-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL`。`build-bjh100/CMakeCache.txt:42` 的 `CMAKE_CUDA_FLAGS` 也是空值，`build-bjh100/CMakeCache.txt:1337-1344`（内部 CUDA flags cache）同样没有该宏。

所以当前命令构建出的程序走的是：

```text
CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL 未定义
  -> include/cutlass/cuda_host_adapter.hpp:107-124
     cudaGetDriverEntryPointByVersion（当前 CUDA 13）
```

这并不表示 TMA 被关闭；只表示 Driver API 入口通过 CUDA Runtime 动态查询，而不是直接链接调用。

### 31.5 当前版本如何关闭并确保生效

如果之前曾通过编译选项启用 direct call，关闭时应：

1. 从 `CMAKE_CUDA_FLAGS`、`CUTLASS_CUDA_FLAGS` 或 target compile options 中移除 `-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL`；
2. 重新运行 CMake 配置，让 `__CUTLASS_CUDA_FLAGS` 等内部 cache 刷新；
3. 必要时删除旧 build 目录或执行 clean build，避免旧对象文件继续沿用旧宏定义。

不能使用：

```bash
-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=0
```

因为 `include/cutlass/cuda_host_adapter.hpp:97`（条件编译 `#if defined(CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL)`）只判断“是否定义”。

### 31.6 启用/关闭路径汇总

```text
启用（当前版本）:
  编译命令
    -> -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL
      -> #if defined(...) 为真
        -> call_cuTensorMapEncodeTiled
          -> 直接 cuTensorMapEncodeTiled

关闭（当前 build-bjh100）:
  编译命令不含该 -D
    -> #if defined(...) 为假
      -> call_cuTensorMapEncodeTiled
        -> cudaGetDriverEntryPointByVersion / cudaGetDriverEntryPoint
          -> 通过函数指针调用 Driver API
```

最终要点是：当前源码没有一个仍然生效的同名 CMake 开关；真正的开关是传给预处理器的 `-D`，并且由于代码使用 `defined`，关闭必须移除定义而不是把值设为 0。

## 32. `explode_tuple` 的实现、TMA 实例与无名 `int_sequence` 形参

### 32.1 `explode_tuple` 的目的

`include/cute/arch/util.hpp:278-316`（函数 `cute::detail::explode_tuple`）提供三个 overload，分别接收 1、2、3 个 tuple-like 对象，并把其中指定下标的元素展开成一次函数调用的独立实参。

三个 tuple 的 overload 位于 `include/cute/arch/util.hpp:304-316`（函数 `cute::detail::explode_tuple`）：

```cpp
template <class Fn,
          class TupleA, int... Ia,
          class TupleB, int... Ib,
          class TupleC, int... Ic>
CUTE_HOST_DEVICE constexpr
void
explode_tuple(Fn fn,
              TupleA&& a, int_sequence<Ia...>,
              TupleB&& b, int_sequence<Ib...>,
              TupleC&& c, int_sequence<Ic...>)
{
  return fn(get<Ia>(a)..., get<Ib>(b)..., get<Ic>(c)...);
}
```

它可以概括为：

```text
输入:
  fn
  tuple A + A 的下标序列
  tuple B + B 的下标序列
  tuple C + C 的下标序列

输出动作:
  fn(A[指定下标]..., B[指定下标]..., C[指定下标]...)
```

这不是运行时循环。下标都位于模板参数包中，编译期直接生成一串 `get<N>(tuple)` 表达式。

### 32.2 模板参数怎样阅读

模板声明中的：

```cpp
class TupleA, int... Ia,
class TupleB, int... Ib,
class TupleC, int... Ic
```

包含三组类型和三组非类型模板参数包：

```text
TupleA = 第一个 tuple 的类型
Ia...  = 第一个 tuple 要取的编译期下标

TupleB = 第二个 tuple 的类型
Ib...  = 第二个 tuple 要取的编译期下标

TupleC = 第三个 tuple 的类型
Ic...  = 第三个 tuple 要取的编译期下标
```

例如模板推导出：

```text
Ia... = 0,1,2
Ib... = 0
Ic... = 0,1
```

函数体：

```cpp
fn(get<Ia>(a)..., get<Ib>(b)..., get<Ic>(c)...);
```

就等价于：

```cpp
fn(get<0>(a), get<1>(a), get<2>(a),
   get<0>(b),
   get<0>(c), get<1>(c));
```

每个 `...` 只展开自己对应的参数包，所以 `Ia`、`Ib`、`Ic` 的长度可以不同。

### 32.3 三个 `int_sequence` 没有参数名是否合法

合法。C++ 函数的形参可以只有类型而省略名称，即使这是函数定义而不仅是函数声明。例如：

```cpp
void f(int, float) {
  // 合法；函数体不需要使用两个形参对象
}
```

所以：

```cpp
int_sequence<Ia...>,
int_sequence<Ib...>,
int_sequence<Ic...>
```

是三个真实的函数形参，只是没有名字。若写出名字，语法上也可以：

```cpp
int_sequence<Ia...> indices_a,
int_sequence<Ib...> indices_b,
int_sequence<Ic...> indices_c
```

但 `explode_tuple` 的函数体不需要读取 `indices_a` 等对象；它只需要函数模板推导得到的 `Ia...`、`Ib...`、`Ic...`，因此省略名称更准确，也避免 unused-parameter 警告。

需要区分两件事：

```text
函数定义处:
  形参对象存在，但省略了形参名

函数调用处:
  仍然传入了 int_sequence<...>{} 临时对象
```

它不是“这个位置没有参数”，而是“参数的运行时对象没有被函数体引用”。

### 32.4 `int_sequence` 和 `tuple_seq` 是什么

`include/cute/numeric/integer_sequence.hpp:37-42`（命名空间 `cute` 中的类型导入）把标准库实现的 `integer_sequence` 引入 CuTe。`include/cute/numeric/integer_sequence.hpp:73-122`（别名 `cute::int_sequence`、`cute::seq`、`cute::make_seq` 和 `cute::tuple_seq`）定义：

```cpp
template <int... Ints>
using int_sequence = integer_sequence<int, Ints...>;

template <int... Ints>
using seq = int_sequence<Ints...>;

template <int N>
using make_seq = make_int_sequence<N>;

template <class Tuple>
using tuple_seq = make_seq<tuple_size<remove_cvref_t<Tuple>>::value>;
```

因此：

```text
seq<0>                    = int_sequence<0>
make_seq<3>               = int_sequence<0,1,2>
tuple_seq<三元素 tuple>   = int_sequence<0,1,2>
tuple_seq<二元素 tuple>   = int_sequence<0,1>
```

`integer_sequence<int,0,1,2>` 的 `0,1,2` 是类型的一部分，不是对象中的运行时数组。调用处的 `{}` 只是值初始化一个无状态的类型标签；经过内联和模板实例化后通常没有运行时开销。

### 32.5 TMA `copy_unpack` 中三个 tuple 分别是什么

调用位于 `include/cute/atom/copy_traits_sm90_tma.hpp:64-90`（函数 `cute::copy_unpack`，定义在类型 `cute::TMA_LOAD_Unpack` 中）：

```cpp
auto src_coord = src(Int<0>{});
void* dst_ptr = cute::raw_pointer_cast(dst.data());

return detail::explode_tuple(
    detail::CallCOPY<CopyOp>{},
    traits.opargs_, tuple_seq<decltype(traits.opargs_)>{},
    make_tuple(dst_ptr), seq<0>{},
    src_coord, tuple_seq<decltype(src_coord)>{});
```

三个 tuple group 是：

| group | tuple 对象 | index sequence | 用途 |
|---|---|---|---|
| A | `traits.opargs_` | `tuple_seq<decltype(...)>` | TMA descriptor、mbarrier、cache hint |
| B | `make_tuple(dst_ptr)` | `seq<0>` | shared-memory destination pointer |
| C | `src_coord` | `tuple_seq<decltype(...)>` | TMA global coordinate 分量 |

当前调用的是 executable TMA-load traits。`include/cute/atom/copy_traits_sm90_tma.hpp:172-194`（构造函数 `cute::Copy_Traits<SM90_TMA_LOAD_OP,...>::Copy_Traits`）定义：

```cpp
tuple<
  TmaDescriptor const*,
  uint64_t*,
  uint64_t
> const opargs_;
```

所以 `traits.opargs_` 有三个元素：

```text
get<0>(traits.opargs_) = TMA descriptor pointer
get<1>(traits.opargs_) = shared-memory mbarrier pointer
get<2>(traits.opargs_) = cache hint
```

其 sequence 为：

```cpp
tuple_seq<decltype(traits.opargs_)>{}
// -> make_seq<3>{}
// -> int_sequence<0,1,2>{}
```

### 32.6 destination 和 source coordinate 的 sequence

第二组主动把单个 `dst_ptr` 包成一元素 tuple：

```cpp
make_tuple(dst_ptr), seq<0>{}
```

所以：

```text
TupleB = tuple<void*>
Ib...  = 0
get<Ib>(b)... = get<0>(b) = dst_ptr
```

这里写 `seq<0>{}` 与为该一元素 tuple 构造 `tuple_seq<...>{}` 的效果相同，只是前者更直接。

第三组 `src_coord` 来自：

```cpp
auto src_coord = src(Int<0>{});
```

对当前 2D A TMA load，它是之前分析过的 `ArithmeticTuple<int,unsigned int>`，语义为 `(m,k)`。`include/cute/numeric/arithmetic_tuple.hpp:497-508`（类型 traits `tuple_size<cute::ArithmeticTuple<...>>` 和 `tuple_element`）把 `ArithmeticTuple` 声明为 tuple-like 类型，因此：

```cpp
tuple_seq<decltype(src_coord)>{}
// -> make_seq<2>{}
// -> int_sequence<0,1>{}
```

对应：

```text
get<0>(src_coord) = crd0，即 M coordinate
get<1>(src_coord) = crd1，即 K coordinate
```

如果 TMA descriptor 是 1D、3D、4D 或 5D，`tuple_seq<decltype(src_coord)>` 会自动生成相应长度的下标包；不需要为每个维度重写 `copy_unpack`。

### 32.7 当前调用的完整模板推导和展开结果

对当前 A 的 2D TMA load，调用可以概念化为：

```cpp
explode_tuple(
    CallCOPY<SM90_TMA_LOAD_OP>{},
    traits.opargs_, int_sequence<0,1,2>{},
    make_tuple(dst_ptr), int_sequence<0>{},
    src_coord, int_sequence<0,1>{});
```

第三个 `explode_tuple` overload 据此推导：

```text
Fn    = CallCOPY<SM90_TMA_LOAD_OP>
Ia... = 0,1,2
Ib... = 0
Ic... = 0,1
```

函数体完整展开成：

```cpp
return fn(
    get<0>(a),  // traits.opargs_: descriptor pointer
    get<1>(a),  // traits.opargs_: mbarrier pointer
    get<2>(a),  // traits.opargs_: cache hint
    get<0>(b),  // make_tuple(dst_ptr): shared-memory pointer
    get<0>(c),  // src_coord: crd0
    get<1>(c)); // src_coord: crd1
```

也就是六个顺序固定的参数：

```text
(descriptor, mbarrier, cache_hint, smem_ptr, crd0, crd1)
```

### 32.8 `fn` 怎样继续调用 TMA 指令

`fn` 的类型是 `CallCOPY<CopyOp>`。`include/cute/arch/util.hpp:149-160`（成员函数 `cute::detail::CallCOPY<CopyOp>::operator()`）继续执行：

```cpp
return CopyOp::copy(static_cast<Args&&>(args)...);
```

本例 `CopyOp=SM90_TMA_LOAD_OP`；其继承关系定义在 `include/cute/atom/copy_traits_sm90_tma.hpp:93-102`（类型 `cute::SM90_TMA_LOAD_OP`）。六个参数中的两个 coordinate 使 overload resolution 选择 `include/cute/arch/copy_sm90_tma.hpp:327-342`（函数 `cute::SM90_TMA_LOAD::copy`）的 2D 版本，再调用 `include/cute/arch/copy_sm90_tma.hpp:103-130`（函数 `cute::SM90_TMA_LOAD_2D::copy`）：

```text
explode_tuple
  -> CallCOPY<SM90_TMA_LOAD_OP>::operator()
    -> SM90_TMA_LOAD_OP::copy(desc,mbar,cache,smem,crd0,crd1)
      -> SM90_TMA_LOAD_2D::copy(...)
        -> cp.async.bulk.tensor.2d...
```

所以 `explode_tuple` 的功能只是编译期参数适配：把三个 tuple-like 参数包拼成底层 TMA copy 所需的平坦函数参数列表。它本身不执行 global-memory load，也不了解 TMA descriptor 的内容。

### 32.9 `return fn(...)` 出现在 `void` 函数中是否合法

`explode_tuple` 的返回类型是 `void`，但函数体写了：

```cpp
return fn(...);
```

这同样是合法 C++：当 `fn(...)` 的表达式类型为 `void` 时，`void` 函数允许 `return` 一个 `void` expression。这里 `CallCOPY::operator()` 和最终 TMA `copy` 都返回 `void`，所以该写法相当于调用 `fn(...)` 后结束当前函数，同时让模板代码也能保持统一的 forwarding 形式。

### 32.10 最简化示例

脱离 TMA 后，可以用下面的例子理解这套机制：

```cpp
auto a = make_tuple(10, 20, 30);
auto b = make_tuple(40);
auto c = make_tuple(50, 60);

explode_tuple(
    [](auto... x) { /* 收到 10,20,30,40,50,60 */ },
    a, int_sequence<0,1,2>{},
    b, int_sequence<0>{},
    c, int_sequence<0,1>{});
```

编译期等价于：

```cpp
fn(get<0>(a), get<1>(a), get<2>(a),
   get<0>(b),
   get<0>(c), get<1>(c));
```

三个无名 `int_sequence` 形参只是让编译器从实参类型中推导出三组下标；真正被函数体使用的是模板参数包 `Ia...`、`Ib...`、`Ic...`。

## 33. `tAgA -> tAsA -> tCsA -> tCrA -> WGMMA` 的数据流和 `tCrA` descriptor

本节针对以下运行条件：

```text
./build-bjh100/examples/cute/tutorial/hopper/cute_tutorial_wgmma_tma_sm90_like \
    512 1024 2048 N T
```

即：

```text
M = 512, N = 1024, K = 2048
bM = 128, bN = 128, bK = 64, PIPE = 3
MMA atom = 64 x 64 x 16
A/B operand = shared-memory descriptor（SS）
```

相关配置位于 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:343-386`（函数 `gemm_nt`）；kernel 中四个 Tensor 的创建和使用位于 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:153-154,240-251,287-315`（函数 `gemm_device`）。

先给出核心结论：

1. 真正的 A 矩阵 `half_t` 数据只发生一次显式搬运：TMA 将 global memory 数据写入 `sA` 对应的 shared memory。
2. `tAsA` 和 `tCsA` 是同一块 `sA` shared memory 的两种 Tensor view；两者之间没有 copy。
3. `tCrA` 不是保存 1024 个 A 元素的寄存器 fragment，而是一个逻辑上的 `GmmaDescriptor` Tensor。它用一个 base descriptor 加 layout stride，按需生成指向不同 `64x16` shared-memory A 子块的 64-bit descriptor。
4. WGMMA 接收 `tCrA` 生成的 descriptor 后，由硬件直接读取 shared memory 中的 A 数据。不存在 `copy(tCsA, tCrA)`。

### 33.1 `tCrA` 和普通 Tensor of value 的区别

CuTe Tensor 的通用索引规则没有变化。`include/cute/tensor_impl.hpp:218-255`（成员函数 `Tensor::operator[]` 和 `Tensor::operator()`）的核心仍然是：

```cpp
return data()[layout()(coord)];
```

区别在于 `data()` 返回的 iterator 类型以及 iterator 的解引用语义。

普通 Tensor of value 可以概念化为：

```text
layout(coord) -> 元素偏移
base_pointer[element_offset] -> half/float/int 等实际 value
```

例如普通 register fragment 的 owning Tensor 会使用 `ArrayEngine<T,N>`，真正分配 `N` 个 `T`；这一分支位于 `include/cute/tensor_impl.hpp:350-383`（函数 `MakeTensor<T>::operator()`）。

当前 WGMMA atom 的 traits 则明确指定：

```cpp
using FrgTypeA = GMMA::smem_desc<tnspA>;
using FrgTypeB = GMMA::smem_desc<tnspB>;
```

代码位置是 `include/cute/atom/mma_traits_sm90_gmma.hpp:648-670`（特化 `MMA_Traits<SM90_64x64x16_F16F16F16_SS<...>>`）。`FrgTypeA` 不是 `half_t`，而是继承自 `DescriptorIterator` 的标记类型。

因此，`make_fragment_A` 在 `include/cute/atom/mma_atom.hpp:145-165`（函数 `MMA_Atom::make_fragment_A`）走的是 `has_dereference<FrgTypeA>` 分支：

```cpp
return make_tensor<FrgTypeA>(atensor);
```

它不会进入 `make_fragment_like<FrgTypeA>` 的 value-array 分支，也不会为 A 分配 1024 个 `half_t` 寄存器。

`GMMA::smem_desc` 的专用构造发生在 `include/cute/atom/mma_traits_sm90_gmma.hpp:354-373`（函数 `MakeTensor<SM90::GMMA::smem_desc<MajorMode>>::operator()`）：

```cpp
return make_tensor(
    DescriptorIterator{make_gmma_desc<MajorMode>(tensor<0>(smem_tensor))},
    replace<0>(recast<uint128_t const>(smem_tensor).layout(),
               Layout<_1,_0>{}));
```

所以 `tCrA` 的实际结构是：

```text
一个 base GmmaDescriptor
+ 一个以 uint128_t（16 byte）为偏移单位的 layout
```

它逻辑上有 24 个 descriptor 元素，但物理上不是一个 `GmmaDescriptor[24]` 数组。`ViewEngine` 只保存 iterator；其定义位于 `include/cute/tensor_impl.hpp:106-117`（类型 `ViewEngine`）。每次索引时，再由 base descriptor 和 layout offset 合成目标 descriptor。

因此 `tCrA(i)` 可以概念化为：

```text
descriptor_offset = tCrA.layout()(i)
result = tCrA.data()[descriptor_offset]
       = base_descriptor，将 start-address 部分增加 descriptor_offset
```

`include/cute/atom/mma_traits_sm90_gmma.hpp:303-330`（类型 `SM90::GMMA::DescriptorIterator`，成员函数 `operator*`、`operator[]` 和 `operator+`）给出了确切实现：

```cpp
reference operator[](Index const& i) const { return *(*this + i); }

DescriptorIterator operator+(Index const& offset) const {
  GmmaDescriptor ret;
  ret.reg32_[0] = desc_.reg32_[0] + uint32_t(offset);
  ret.reg32_[1] = desc_.reg32_[1];
  return {ret};
}
```

这里返回的是 `GmmaDescriptor` value，而不是 shared memory 中某个 `half_t` value。这也解释了为什么：

```cpp
auto elem = tCrA(idx);
elem.foo();
```

会报错 `union "cute::GmmaDescriptor" has no member "foo"`：`elem` 的确切类型就是 `cute::GmmaDescriptor`。该 union 的字段定义位于 `include/cute/arch/mma_sm90_desc.hpp:80-130`（类型 `cute::GmmaDescriptor`）。

需要区分两种“在寄存器中”：

- A 的 1024 个 `half_t` 数据没有进入每线程寄存器；它们仍在 shared memory。
- WGMMA 发射时，64-bit descriptor 本身会作为指令操作数进入普通寄存器。`r` in `tCrA` 指的是 MMA fragment 接口这一侧，不表示 A value fragment 已经装入寄存器。

### 33.2 `tAsA` 和 `tCsA` 的关系

`sA` 是完整的三阶段 shared-memory Tensor。当前日志 `temp/run.wgmma_tma_sm90_like.log:78-87` 打印出：

```text
sA:
Sw<3,4,3>_smem_ptr[16b](...0400)
o ((_64,_2),(_8,_8),(_1,_3))
 : ((_1,_512),(_64,_1024),(_0,_8192))

tAsA:
Sw<3,4,3>_smem_ptr[16b](...0400)
o ((_512,_16),(_1,_3))
 : ((_1,_512),(_0,_8192))
```

`tAsA` 由 `tma_partition` 从 `sA` 构造。`include/cute/atom/copy_traits_sm90_tma.hpp:1403-1435`（函数 `tma_partition`）通过 `compose`、`coalesce` 和 `domain_offset` 重新组织输入 Tensor，最后返回 `gresult` 和 `sresult`。它没有为 `sresult` 分配新的 shared memory。

对 A 而言：

```text
size<0>(tAsA) = 512 * 16 = 8192 half
size<1>(tAsA) = 1 * 3 = 3 pipes
size(tAsA)    = 8192 * 3 = 24576 half
```

一个 pipe 的 8192 个 `half_t` 正好等于一个 `128x64` A CTA tile。

`tCsA` 则由：

```cpp
Tensor tCsA = thr_mma.partition_A(sA);
```

产生。`include/cute/atom/mma_atom.hpp:288-314`（函数 `TiledMMA::thrfrg_A`）先按 atom 的 `64x16` 形状 tile A Tensor，再用 `AtomLayoutA_TV` 将 atom 内坐标变换为 `(ThrV,FrgV)`；`include/cute/atom/mma_atom.hpp:475-484`（函数 `ThrMMA::partition_A`）继续固定当前 thread 的 `(ThrV,ThrM,ThrK)` 坐标。

日志 `temp/run.wgmma_tma_sm90_like.log:148-161` 给出：

```text
tCsA:
Sw<3,4,3>_smem_ptr[16b](...0400)
o ((_64,(_8,_2)),_2,_4,(_1,_3))
 : ((_1,(_64,_1024)),_512,_2048,(_0,_8192))
```

其各 mode 含义为：

```text
((_64,(_8,_2))) : 一个 WGMMA A operand，64 * (8*2) = 64x16 = 1024 half
_2               : MMA_M，128 / 64 = 2 个 M atom
_4               : MMA_K， 64 / 16 = 4 个 K atom
(_1,_3)          : 3 个 pipeline stage；前面的 _1 是退化 mode
```

总元素数仍然是：

```text
1024 * 2 * 4 * 3 = 24576 half
```

而且 `tAsA` 与 `tCsA` 的 base 都是日志中的同一个 `Sw<3,4,3>_smem_ptr(...0400)`。所以二者的准确关系是：

```text
tAsA：从 TMA copy 的角度索引 sA
tCsA：从 64x64x16 WGMMA atom 的角度索引同一个 sA
```

它们是 alias，不是前后两个 storage。TMA 经 `tAsA` 写入的数据，在 producer barrier 完成后立刻可以经 `tCsA` 的另一套坐标读取；不存在 `tAsA -> tCsA` 的数据搬运。

### 33.3 `tCrA` 和 `tCsA` 的关系

`tCrA` 是从 `tCsA` 的第一个 `64x16` A 子块生成 base descriptor，然后保留外层 `MMA_M`、`MMA_K` 和 `PIPE` 的步进信息。

专用 `MakeTensor` 在 `include/cute/atom/mma_traits_sm90_gmma.hpp:361-372`（函数 `MakeTensor<SM90::GMMA::smem_desc<MajorMode>>::operator()`）做两件事：

1. `make_gmma_desc<MajorMode>(tensor<0>(smem_tensor))` 检查第一个 `64x16` 子块的 shared-memory layout，并编码 base descriptor。
2. `recast<uint128_t const>(smem_tensor).layout()` 把偏移单位从 `half_t` 改成 128 bit，即每单位 16 byte；随后把 mode 0 替换成 `Layout<_1,_0>`，因为一个 descriptor 已经描述整个 atom operand。

因此：

```text
tCsA: ((_64,(_8,_2)), _2, _4, (_1,_3))
       整个 atom 内有 1024 个 half value

tCrA: (_1,             _2, _4, (_1,_3))
       整个 atom 内只需 1 个 descriptor
```

外层 stride 的换算也完全一致。一个 `uint128_t` 包含 8 个 `half_t`：

```text
tCsA MMA_M stride =  512 half =  512 / 8 =   64 uint128 units
tCsA MMA_K stride = 2048 half = 2048 / 8 =  256 uint128 units
tCsA PIPE  stride = 8192 half = 8192 / 8 = 1024 uint128 units
```

所以日志中的：

```text
tCrA:
GMMA::DescriptorIterator
o (_1,_2,_4,(_1,_3))
 : (_0,_64,_256,(_0,_1024))
```

不是巧合，而是 `tCsA` 的 shared-memory 子块起点以 16-byte 单位表示后的结果。

`tCrA` 与 `tCsA` 之间同样没有 A value copy。源码本身也在 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:230-251`（函数 `gemm_device`）明确说明 descriptor 是 SMEM view，因此 mainloop 不需要 `copy(tCsA,tCrA)`。

### 33.4 完整的数据流

准确的数据流应写成：

```text
global-memory A values
        ^
        |  TMA descriptor + tAgA coordinates 指定源地址
        |
        +---- cp.async.bulk.tensor / TMA ---->
                                             sA shared-memory bytes
                                                   ^
                                                   |
                                      tAsA：TMA destination view
                                                   |
                                      同一 storage，无 copy
                                                   |
                                      tCsA：WGMMA tiled view
                                                   |
                                      生成地址/布局 metadata
                                                   v
                                      tCrA：GmmaDescriptor view
                                                   |
                                                   | 64-bit desc_a
                                                   v
                                      wgmma.mma_async
                                      硬件读取 sA，累加到 tCrC
```

逐步解释如下。

第一步，`tAgA` 不是装着 A value 的普通 global-memory pointer Tensor。它是 TMA coordinate Tensor；当前 CTA `(blockIdx.x,blockIdx.y)=(1,0)` 的日志显示 base coordinate 为 `(128,0)`，见 `temp/run.wgmma_tma_sm90_like.log:65-120`。TMA 用已经编码好的 tensor-map descriptor 加上 `tAgA` 产生的 `(m,k)` coordinates，解析真正的 global address。

第二步，只有 elected producer lane 发射：

```cpp
copy(tma_a.with(producer_mbar[pipe]),
     tAgA(_,k_tile),
     tAsA(_,pipe));
```

代码位于 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:215-225`（函数 `gemm_device`）。这一步才把一个 `128x64` A tile 的实际 `half_t` 数据异步写入选定 pipe 的 `sA`。

第三步，consumer 在 `ProducerBarType::wait` 成功后使用该 pipe；等待与 GEMM 调用位于 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:287-307`（函数 `gemm_device`）。`tCsA` 只是把已经到达的 `sA` bytes 分解成 `2` 个 MMA_M 子块和 `4` 个 MMA_K 子块。

第四步，`tCrA` 从 `tCsA` 的地址和 layout 生成 descriptor。这个动作只生成 metadata，不读取 A 的 1024 个 value。

第五步，`gemm` 把 descriptor 解引用成一个 64-bit 操作数。`include/cute/atom/mma_traits_sm90_gmma.hpp:385-429`（函数 `SM90::GMMA::mma_unpack`）将 A/B fragment recast 为底层 operation 所需的 register 类型；当前 operation 的 `ARegisters` 和 `BRegisters` 都是 `uint64_t[1]`，见 `include/cute/arch/mma_sm90_gmma.hpp:409-430`（类型及函数 `MMA_64x64x16_F16F16F16_SS::fma`）。

最后，底层 PTX：

```text
wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16
```

直接接收 `desc_a` 和 `desc_b`，代码位于 `include/cute/arch/mma_sm90_gmma.hpp:423-451`（函数 `MMA_64x64x16_F16F16F16_SS::fma`）。真正的 A/B value 是 WGMMA 硬件根据 descriptor 从 shared memory 读取的。

### 33.5 CTA 内所有线程看到的 Tensor 是否一样

对当前这个具体的 SS WGMMA kernel，答案如下：

| Tensor | CTA 内 128 个线程的逻辑内容是否相同 | 说明 |
|---|---:|---|
| `tAgA` | 是 | 构造只依赖同一个 CTA 的 `gA`、TMA atom 和固定的 `Int<0>{}, Layout<_1>{}`，不依赖 `threadIdx.x`；每线程有自己的轻量 Tensor 对象，但 coordinate mapping 相同。只有 elected lane 实际使用它发射 TMA。 |
| `tAsA` | 是 | 所有线程的 view 都指向 CTA 的同一个 `sA` 和相同 pipe。实际 shared-memory bytes 是 CTA 共享的；只有 producer lane 发起写入。 |
| `tCsA` | 在当前 SS atom 中是 | API 名称是“thread partition”，但当前 A source layout 的 thread mode stride 为 0，所以 128 个线程得到相同的 A shared-memory view。这个结论不能泛化到所有 MMA atom。 |
| `tCrA` | 在当前 SS atom 中是 | 它由相同的 `tCsA` 地址和 layout 生成，所以 warpgroup 的 128 个线程得到相同的 A descriptors。各线程的 accumulator `tCrC` 则不同。 |

为什么当前 `tCsA/tCrA` 是 warpgroup-uniform，可以从 atom layout 直接看出。`include/cute/atom/mma_traits_sm90_gmma.hpp:462-465`（类型别名 `GMMA::ABLayout`）定义：

```cpp
using ABLayout = Layout<Shape<_128, Shape<Int<M>,Int<K>>>,
                        Stride<_0,   Stride<_1,Int<M>>>>;
```

第一维 `_128` 是 warpgroup 的 128 个线程，而对应 stride 是 `_0`。当前 traits 在 `include/cute/atom/mma_traits_sm90_gmma.hpp:657-670`（特化 `MMA_Traits<SM90_64x64x16_F16F16F16_SS<...>>`）选择 `ABLayout<64,16>`。日志也直接打印：

```text
LayoutA_TV: (_128,(_64,_16)):(_0,(_1,_64))
```

所以 thread id 从 0 变到 127 不会改变 A operand 的 `(m,k)` 地址。`ThrMMA::partition_A` 虽然在 `include/cute/atom/mma_atom.hpp:475-484`（函数 `ThrMMA::partition_A`）固定了当前 thread 坐标，但该坐标落在 stride-0 的 mode 上，结果仍相同。这正符合 SS WGMMA 的模型：一个 128-thread warpgroup 共同使用相同的 shared-memory A/B descriptors，而每个线程持有自己那部分 accumulator registers。

这里的“相同”有三个边界条件：

1. 指同一个 CTA 内；不同 CTA 的 `blockIdx` 不同，`tAgA` 指向的 global tile 也可能不同。
2. 指逻辑 view、地址和在同步完成后的 value 相同；每个 CUDA thread 仍有自己的局部 Tensor 对象和 descriptor 临时值。
3. 在 `ProducerBarType::wait` 完成前，TMA 可能尚未写完 shared memory，此时不能把“大家指向同一地址”误解为“数据已经可安全消费”。等待发生在 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:290-307`（函数 `gemm_device`）。

### 33.6 为什么 `tCrA` 打印成该 shape 和 stride

日志 `temp/run.wgmma_tma_sm90_like.log:160-164` 为：

```text
GMMA::DescriptorIterator
o (_1,_2,_4,(_1,_3))
 : (_0,_64,_256,(_0,_1024))
```

shape 的含义是：

```text
_1       : 一个 64x16 A atom operand 由一个 descriptor 表示
_2       : 两个 MMA_M atom，覆盖 CTA_M = 2 * 64 = 128
_4       : 四个 MMA_K atom，覆盖 CTA_K = 4 * 16 = 64
(_1,_3)  : 三个 pipe
```

所以：

```text
size(tCrA)          = 1 * 2 * 4 * 1 * 3 = 24 descriptors
size(tCrA per pipe) = 1 * 2 * 4         = 8 descriptors
```

stride 的单位不是 `half_t`，而是 `uint128_t`，即 16 byte。这由 `include/cute/atom/mma_traits_sm90_gmma.hpp:361-372`（函数 `MakeTensor<SM90::GMMA::smem_desc<MajorMode>>::operator()`）中的 `recast<uint128_t const>` 决定。

因此：

```text
MMA mode stride =    0：一个 atom 内只有一个 descriptor
MMA_M stride    =   64：下一个 64-row A atom 起点相差 64 * 16 = 1024 bytes
MMA_K stride    =  256：下一个 16-column K atom 起点相差 256 * 16 = 4096 bytes
PIPE stride     = 1024：下一个 pipe 起点相差 1024 * 16 = 16384 bytes
```

这些 byte offset 分别与原 `tCsA` 的 `512`、`2048`、`8192` 个 `half_t` stride 完全一致。

### 33.7 `tCrA(0..3)` 如何通过 layout 变成四个 descriptor

`Tensor::operator()` 在 `include/cute/tensor_impl.hpp:233-255`（成员函数 `Tensor::operator()`）先调用 `layout()(idx)`，再把结果传给 iterator 的 `operator[]`。

对 shape：

```text
(_1,_2,_4,(_1,_3))
```

flat index 的低位 mode 先变化。因此前四个 linear index 对应：

| `idx` | 逻辑坐标 `(MMA,MMA_M,MMA_K,(1,PIPE))` | layout offset，单位 16 B | shared-memory byte address | descriptor `start_addr` |
|---:|---|---:|---:|---:|
| 0 | `(0,0,0,(0,0))` | `0` | `0x400 + 0*16 = 0x400` | `0x040` |
| 1 | `(0,1,0,(0,0))` | `64` | `0x400 + 64*16 = 0x800` | `0x080` |
| 2 | `(0,0,1,(0,0))` | `256` | `0x400 + 256*16 = 0x1400` | `0x140` |
| 3 | `(0,1,1,(0,0))` | `64+256=320` | `0x400 + 320*16 = 0x1800` | `0x180` |

这里的 `0x400` 是 `sA` 在 CTA shared-address space 中的起始 byte address。日志展示的是完整 pointer 的低部 `...0400`；`make_gmma_desc` 会先通过 `cast_smem_ptr_to_uint` 得到 shared address，再去掉 4 个低位，见 `include/cute/atom/mma_traits_sm90_gmma.hpp:196-216`（函数 `SM90::GMMA::make_gmma_desc`）：

```cpp
uint32_t start_address = cast_smem_ptr_to_uint(...);
desc.bitfield.start_address_ = uint16_t(start_address >> 4);
```

因此 descriptor 中的 `start_addr` 本来就是 16-byte 单位。`DescriptorIterator::operator+` 增加 `0/64/256/320` 后，打印出的起始地址自然依次为：

```text
0x0040, 0x0080, 0x0140, 0x0180
```

前 8 个 flat element 都属于 pipe 0；`idx=8` 才会进入 pipe 1，并额外增加 pipe stride `1024`。

### 33.8 每个 `GmmaDescriptor` 字段为什么是日志中的值

以 `idx=0` 为例，日志 `temp/run.wgmma_tma_sm90_like.log:168-198` 打印：

```text
GmmaDescriptor: 0x4000008000000040
  start_addr :  0x0040
  leading_off:  0x0000 (0)
  stride_off :  0x0080 (128)
  base_offset:  0x0
  layout_type:  0x1 (B128)
```

各字段来源如下。

`start_addr=0x40`：`sA` shared byte address 为 `0x400`，descriptor 不保存低 4 bit，所以编码为 `0x400 >> 4 = 0x40`。地址字段的 bit 定义见 `include/cute/arch/mma_sm90_desc.hpp:107-125`（类型 `GmmaDescriptor::bitfield`）。

`layout_type=1 (B128)`：`sA` 的 pointer 带 `Swizzle<3,4,3>`。`include/cute/atom/mma_traits_sm90_gmma.hpp:128-150`（函数 `SM90::GMMA::layout_type`）把 `num_bits B=3` 映射成 `LayoutType::B128`；枚举值最终编码在 descriptor 的最高 2 bit。

`stride_off=0x80`：对于 `Major::MN` 的 B128 layout，`make_gmma_desc` 从 canonical layout 取 stride dimension offset，见 `include/cute/atom/mma_traits_sm90_gmma.hpp:227-260`（函数 `SM90::GMMA::make_gmma_desc`）。当前对应 `tCsA` atom 内 `_1024` 个 `half_t` 的 stride：

```text
1024 half * 2 bytes / 16 bytes = 128 = 0x80
```

`leading_off=0`：该字段对 swizzled layout 不使用；`GmmaDescriptor` 初始值为 0，当前构造没有为 B128 写入非零 leading offset。字段用途的说明位于 `include/cute/arch/mma_sm90_desc.hpp:111-119`（类型 `GmmaDescriptor::bitfield`）。

`base_offset=0`：当前实现显式设置 `constexpr uint8_t base_offset = 0`，见 `include/cute/atom/mma_traits_sm90_gmma.hpp:214-225`（函数 `SM90::GMMA::make_gmma_desc`）。

完整 64-bit word 可以拆成：

```text
layout_type B128 : 0x4000000000000000
stride_off 0x80 : 0x0000008000000000
start_addr 0x40 : 0x0000000000000040
------------------------------------------------
                    0x4000008000000040
```

`idx=1/2/3` 只通过 `DescriptorIterator::operator+` 改变低 32 bit 中的 start-address 部分，高位 layout metadata 不变，所以得到：

```text
idx 0: 0x4000008000000040
idx 1: 0x4000008000000080
idx 2: 0x4000008000000140
idx 3: 0x4000008000000180
```

### 33.9 一个 pipe 中 descriptor 如何覆盖 `128x128x64` GEMM

主循环选择一个 pipe 后传入：

```cpp
gemm(mma,
     tCrA(_,_,_,read_pipe),
     tCrB(_,_,_,read_pipe),
     tCrC);
```

代码位置是 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:287-311`（函数 `gemm_device`）。切掉 PIPE mode 后：

```text
A descriptor tensor shape = (1,2,4)  // V,M,K
B descriptor tensor shape = (1,2,4)  // V,N,K
C accumulator shape        = (V,2,2)  // V,M,N
```

`include/cute/algorithm/gemm.hpp:388-416`（rank-3 函数 `cute::gemm`，Dispatch [5]）遍历 4 个 MMA_K；每个 K step 再进入 `include/cute/algorithm/gemm.hpp:263-300`（rank-2 A/B 函数 `cute::gemm`，Dispatch [4]），遍历 `2` 个 MMA_M 和 `2` 个 MMA_N。

因此一个 `128x128x64` CTA tile 共发射：

```text
MMA_M * MMA_N * MMA_K = 2 * 2 * 4 = 16
```

条 `m64n64k16` WGMMA 指令。每条指令：

- 从 `tCrA` 取一个描述 `64x16` A 子块的 descriptor；
- 从 `tCrB` 取一个描述 `64x16` B 子块的 descriptor；
- 由硬件直接读取相应 shared-memory 数据；
- 将结果累加到每线程各自的 `tCrC` register fragment。

这也是整个链路最容易混淆、但最重要的区别：

```text
tAgA -> tAsA   ：发生实际 global-to-shared value movement
tAsA -> tCsA   ：同一 shared storage 的 view change，不移动数据
tCsA -> tCrA   ：生成 descriptor metadata，不移动 A values
tCrA -> WGMMA  ：传递 64-bit descriptor；硬件从 shared memory 读取 values 并计算
```

## 34. llama.cpp CUDA backend 的 GEMM/matmul 使用 CUTLASS、cuBLAS 还是自有 kernel

本节依据 llama.cpp 本地代码库提交 `c588c4f47`。本节中的源码路径均相对于：

```text
/share/users/like/package/llama.cpp
```

### 34.1 结论

当前 llama.cpp CUDA backend 的 `GGML_OP_MUL_MAT` 不是固定只使用一种实现，而是：

```text
cuBLAS + llama.cpp 自己实现的 CUDA kernels，运行时按数据类型、矩阵形状和 GPU 架构选择
```

更具体地说：

| 场景 | 常见执行路径 | 实现者 |
|---|---|---|
| F32/F16/BF16，形状适合自有窄矩阵 kernel | MMVF 或 MMF | llama.cpp 自有 CUDA kernel |
| 量化权重，batch 很小 | MMVQ | llama.cpp 自有 CUDA kernel |
| 量化权重，形状和架构适合量化矩阵乘 | MMQ | llama.cpp 自有 CUDA kernel |
| 上述自有 kernel 不适用，或输入/输出类型要求如此 | `cublasSgemm`、`cublasGemmEx` 或 batched 版本 | NVIDIA cuBLAS |
| 常规 `GGML_OP_MUL_MAT` | 不使用 CUTLASS | 当前源码没有接入 CUTLASS |

因此，“llama.cpp CUDA matmul 使用什么”的准确答案是：

- 会使用 cuBLAS。
- 也大量使用 llama.cpp 自己实现的 matmul/matvec CUDA kernels，尤其是量化模型。
- 当前这份代码的常规 matmul 分派不使用 CUTLASS。

这里的“自己实现”不等于只用普通 CUDA core。llama.cpp 的自有 kernel 会直接使用 `mma.sync`、`ldmatrix`、`dp4a` 等底层指令来利用 Tensor Core 或整数点积硬件，但这仍然不是 CUTLASS。

### 34.2 从模型中的 `ggml_mul_mat` 到 CUDA 分派入口

模型建图时，例如 `src/llama-graph.cpp:1382-1389`（函数 `llm_graph_context::build_lora_mm`）调用：

```cpp
ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);
```

`ggml/src/ggml.c:3270-3292`（函数 `ggml_can_mul_mat` 和 `ggml_mul_mat`）检查两个 Tensor 的 K 维是否匹配，创建 F32 输出 Tensor，并设置：

```cpp
result->op     = GGML_OP_MUL_MAT;
result->src[0] = a;
result->src[1] = b;
```

当 graph node 被放到 CUDA backend 执行时，`ggml/src/ggml-cuda/ggml-cuda.cu:2011-2012,2200-2205`（函数 `ggml_cuda_compute_forward`）把 `GGML_OP_MUL_MAT` 分派到：

```cpp
ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
```

这里才开始决定使用哪个实际 GEMM/matmul 实现。

### 34.3 `ggml_cuda_mul_mat` 的运行时决策树

核心分派完整地写在 `ggml/src/ggml-cuda/ggml-cuda.cu:1812-1852`（函数 `ggml_cuda_mul_mat`）：

```cpp
if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
    ggml_cuda_mul_mat_cublas(...);
    return;
}

if (ggml_cuda_should_use_mmvf(...)) {
    ggml_cuda_mul_mat_vec_f(...);
    return;
}
if (ggml_cuda_should_use_mmf(...)) {
    ggml_cuda_mul_mat_f(...);
    return;
}
if (ggml_cuda_should_use_mmvq(...)) {
    ggml_cuda_mul_mat_vec_q(...);
    return;
}
if (ggml_cuda_should_use_mmq(...)) {
    ggml_cuda_mul_mat_q(...);
    return;
}
ggml_cuda_mul_mat_cublas(...);
```

可整理为：

```text
GGML_OP_MUL_MAT
        |
        v
ggml_cuda_mul_mat
        |
        +-- 特殊 padding，或 src1/dst 不是 F32 --------> cuBLAS
        |
        +-- 浮点小 batch，MMVF 更合适 -----------------> 自有 MMVF kernel
        |
        +-- 浮点窄矩阵，MMF 更合适 ---------------------> 自有 MMF kernel
        |
        +-- 量化权重且 batch <= MMVQ 阈值 ------------> 自有 MMVQ kernel
        |
        +-- 量化权重且 MMQ heuristic 通过 -------------> 自有 MMQ kernel
        |
        +-- 以上都不满足 -------------------------------> cuBLAS fallback
```

这些判断不只看 `GGML_TYPE_*`，也会看：

- `src1` 的列数，即 token/batch 数量；
- GPU compute capability；
- Tensor 是否连续和对齐；
- 矩阵行数是否满足 kernel tile；
- 是否为 `MUL_MAT_ID`/MoE；
- 编译时是否强制 MMQ 或 cuBLAS。

所以不能仅凭“模型是 F16”或“模型是 Q4”就断言所有 matmul 都走同一条路径。

### 34.4 cuBLAS 路径具体调用什么

`ggml_cuda_mul_mat_cublas` 先选择 F32、F16 或 BF16 compute type。量化的 `src0` 会先被转换为适合 cuBLAS 的浮点格式。相关逻辑位于 `ggml/src/ggml-cuda/ggml-cuda.cu:1619-1660`（函数 `ggml_cuda_mul_mat_cublas`）。

真正的 cuBLAS 调用位于 `ggml/src/ggml-cuda/ggml-cuda.cu:1405-1617`（函数模板 `ggml_cuda_mul_mat_cublas_impl`）：

- 单个 F32 GEMM 使用 `cublasSgemm`，见 `ggml/src/ggml-cuda/ggml-cuda.cu:1537-1546`（函数模板 `ggml_cuda_mul_mat_cublas_impl`）。
- 单个 F16/BF16 GEMM 使用 `cublasGemmEx`，见 `ggml/src/ggml-cuda/ggml-cuda.cu:1547-1555`（同一函数）。
- 连续 batch 使用 `cublasGemmStridedBatchedEx`，见 `ggml/src/ggml-cuda/ggml-cuda.cu:1556-1571`（同一函数）。
- 需要独立 pointer 数组的 batch 使用 `cublasGemmBatchedEx`，见 `ggml/src/ggml-cuda/ggml-cuda.cu:1572-1609`（同一函数）。

CUDA backend 的构建也明确链接 cuBLAS。`ggml/src/ggml-cuda/CMakeLists.txt:158-176`（CUDA backend target 配置）对 static build 链接 `CUDA::cublas_static`，对 shared build 链接 `CUDA::cublas`。

因此，cuBLAS 不是可有可无的注释代码，而是 CUDA backend 的实际 fallback 和部分主执行路径。

### 34.5 llama.cpp 自有 matmul kernels

当前 CUDA backend 至少有四类自有乘法 kernel：

```text
MMVF = floating-point matrix-vector / small-batch kernel
MMF  = floating-point matrix-matrix kernel
MMVQ = quantized matrix-vector / small-batch kernel
MMQ  = quantized matrix-matrix kernel
```

例如 MMF 的 CUDA kernel `mul_mat_f` 定义在 `ggml/src/ggml-cuda/mmf.cuh:48-80`（函数模板 `mul_mat_f`），kernel launch 位于 `ggml/src/ggml-cuda/mmf.cuh:572-616`（函数模板 `mul_mat_f_switch_ids`）。`ggml/src/ggml-cuda/mmf.cu:133-190`（函数 `ggml_cuda_should_use_mmf`）根据类型、对齐、矩阵列数和 GPU MMA 能力决定是否走 MMF。

量化矩阵乘的主 kernel `mul_mat_q` 定义在 `ggml/src/ggml-cuda/mmq.cuh:918-990`（函数模板 `mul_mat_q`）。它不是对 cuBLAS 的简单 wrapper；kernel 自己完成 shared-memory tiling、量化 dot product、累加和结果写回。

`ggml/src/ggml-cuda/mmq.cuh:1361-1440`（函数模板 `launch_mul_mat_q`）直接使用 CUDA launch syntax：

```cpp
mul_mat_q<type, J, fallback><<<grid, block, shared_bytes, stream>>>(...);
```

这说明 MMQ 是 llama.cpp 编译进 `ggml-cuda` 的自有 `__global__` kernel。

### 34.6 为什么自有 kernel 使用 Tensor Core 仍不等于 CUTLASS

llama.cpp 自己实现了一个轻量的 MMA tile/fragment 层。`ggml/src/ggml-cuda/mma.cuh:1-19`（文件级说明及 `ggml_cuda_mma` 实现）明确说该文件暴露 CUDA Tensor Core PTX primitives，并定义自己的 tile 数据布局。

例如 `ggml/src/ggml-cuda/mma.cuh:920-963`（重载函数 `ggml_cuda_mma::mma`）直接内联：

```cpp
asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 ...");
```

MMQ 的量化 dot-product 再调用这套自有 `mma` primitive。以 Q4_0 的 MMA data layout 为例，`ggml/src/ggml-cuda/mmq.cuh:684-696`（函数模板 `ggml_cuda_mmq_get_util_funcs`）选择：

```cpp
ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma<...>
```

后者在 `ggml/src/ggml-cuda/mmq-vec-dot.cuh:142-202`（函数模板 `ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma`）加载自有 tile 并调用 `mma(C,A,B)`。

这条调用链是：

```text
llama.cpp mul_mat_q kernel
  -> llama.cpp ggml_cuda_mmq_vec_dot_*_mma
    -> llama.cpp ggml_cuda_mma::mma
      -> inline PTX mma.sync
```

而不是：

```text
llama.cpp -> cutlass::gemm::device::Gemm -> CUTLASS kernel
```

在当前提交中，对 `ggml/src/ggml-cuda` 的 `cutlass` 搜索只命中 `ggml/src/ggml-cuda/common.cuh:57-59` 的一条 NVIDIA 文档 URL 注释；没有 CUTLASS header include、CUTLASS GEMM type 或 CUTLASS CMake dependency。`ggml/src/ggml-cuda/CMakeLists.txt:102-130`（CUDA source target 配置）编译的是 llama.cpp 自己的 `.cu/.cuh` 和 template instances。

所以应该区分：

```text
使用 NVIDIA Tensor Core 指令 != 使用 NVIDIA CUTLASS 库
```

### 34.7 具体例子：Ampere 上的 Q4_0 线性层

考虑一个常见的量化 FFN 线性层：

```text
src0 = W, GGML_TYPE_Q4_0, shape [K=4096, M=11008]
src1 = X, GGML_TYPE_F32,  shape [K=4096, N=32]
dst  = Y, GGML_TYPE_F32,  shape [M=11008, N=32]
GPU  = NVIDIA Ampere
GGML_CUDA_FORCE_CUBLAS = OFF（默认）
GGML_CUDA_CUBLAS_COMPUTE_TYPE = auto/unset，且没有 GGML_PREC_F32 override
src0 是普通模型权重，不触发 bad_padding_clear
```

在 ggml 的 `ne[]` 表示中，对应：

```text
src0->ne[0] = 4096   // K
src0->ne[1] = 11008  // M
src1->ne[0] = 4096   // K
src1->ne[1] = 32     // N，也就是 ne11
dst->ne[0]  = 11008
dst->ne[1]  = 32
```

调用链如下。

第一步，`ggml_cuda_compute_forward` 将 `GGML_OP_MUL_MAT` 交给 `ggml_cuda_mul_mat`，见 `ggml/src/ggml-cuda/ggml-cuda.cu:2200-2205`（函数 `ggml_cuda_compute_forward`）。

第二步，`src1` 和 `dst` 都是 F32，而且普通模型权重不触发 `bad_padding_clear`，因此不会在 `ggml_cuda_mul_mat` 的第一个检查中直接进入 cuBLAS。

第三步，MMVF/MMF 是浮点权重路径；`src0` 是量化 Q4_0，所以不适用。特别是 `ggml/src/ggml-cuda/mmf.cu:133-137`（函数 `ggml_cuda_should_use_mmf`）对 quantized type 直接返回 `false`。

第四步，MMVQ 只用于很小的 batch。`ggml/src/ggml-cuda/mmvq.cuh:1-5`（常量 `MMVQ_MAX_BATCH_SIZE` 和函数声明 `ggml_cuda_should_use_mmvq`）把通用最大 batch 定为 8；`ggml/src/ggml-cuda/mmvq.cu:280-328`（函数 `ggml_cuda_should_use_mmvq`）最终对普通 NVIDIA 情况判断 `ne11 <= 8`。本例 `N=32`，所以 MMVQ 返回 `false`。

第五步，MMQ 判断为 `true`。`ggml/src/ggml-cuda/mmq.cu:253-307`（函数 `ggml_cuda_should_use_mmq`）列出 Q4_0 为支持类型，并且 NVIDIA Turing 及以上有 MMA 时直接返回 `true`；Ampere 满足该条件。

因此 `ggml_cuda_mul_mat` 在 `ggml/src/ggml-cuda/ggml-cuda.cu:1847-1849`（函数 `ggml_cuda_mul_mat`）调用：

```cpp
ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
```

第六步，`ggml/src/ggml-cuda/mmq.cu:79-170`（函数 `ggml_cuda_mul_mat_q`）先把 F32 activation `X` 量化成 MMQ 使用的 Q8_1 tile，然后通过 `ggml_cuda_mul_mat_q_switch_type` 选择 Q4_0 template specialization。Q4_0 的 type switch 位于 `ggml/src/ggml-cuda/mmq.cu:8-15`（函数 `ggml_cuda_mul_mat_q_switch_type`）。

第七步，本例 `11008 % 128 == 0`，所以 `ggml/src/ggml-cuda/mmq.cuh:1526-1535`（函数模板 `mul_mat_q_case`）选择非 fallback layout。Ampere 的 Q4_0/J=32 配置存在于 `ggml/src/ggml-cuda/mmq-config-ampere.cuh:19-34`（函数 `ggml_cuda_mmq_get_config_ampere`）。

最后，`launch_mul_mat_q` 发射 llama.cpp 自己的：

```text
mul_mat_q<GGML_TYPE_Q4_0, J, false><<<...>>>
```

并在 kernel 内通过 llama.cpp 自己的 MMA primitive 执行 `mma.sync`。所以该例默认路径的答案是：

```text
使用 llama.cpp 自己实现的 MMQ CUDA kernel，不使用 cuBLAS，也不使用 CUTLASS。
```

### 34.8 同一例子如何改为 cuBLAS

构建选项定义在 `ggml/CMakeLists.txt:199-203`（CUDA options 配置）：

```cmake
option(GGML_CUDA_FORCE_MMQ    "ggml: use mmq kernels instead of cuBLAS"        OFF)
option(GGML_CUDA_FORCE_CUBLAS "ggml: always use cuBLAS instead of mmq kernels" OFF)
```

`ggml/src/ggml-cuda/CMakeLists.txt:138-144`（CUDA backend compile-definition 配置）把启用的选项变成相应预处理宏。

若对上述 `N=32` 的 Q4_0 例子启用：

```text
-DGGML_CUDA_FORCE_CUBLAS=ON
```

则 `ggml/src/ggml-cuda/mmq.cu:253-256`（函数 `ggml_cuda_should_use_mmq`）会立即返回 `false`。由于 `N=32` 也不满足 MMVQ 的 `N<=8` 条件，`ggml_cuda_mul_mat` 最终走到 `ggml_cuda_mul_mat_cublas` fallback。

在上述默认 compute precision 前提下，Ampere 会把量化 Q4_0 权重转换为适合计算的 F16；对这个非 batched 例子，最终调用 `ggml/src/ggml-cuda/ggml-cuda.cu:1547-1555`（函数模板 `ggml_cuda_mul_mat_cublas_impl`）中的 `cublasGemmEx`。如果通过环境变量或 op precision 强制 F32，则 compute type 和具体 cuBLAS API 也会相应改变；该 override 逻辑位于 `ggml/src/ggml-cuda/ggml-cuda.cu:1619-1645`（函数 `ggml_cuda_mul_mat_cublas`）。

这进一步说明选择发生在 llama.cpp 的 dispatch 层：同一个 ggml `MUL_MAT` node 可以因为类型、batch、GPU 或编译选项不同，在自有 kernel 和 cuBLAS 之间切换。

### 34.9 最终总结

```text
CUTLASS：当前常规 CUDA GEMM/matmul 路径不使用。

cuBLAS：使用；负责通用浮点 GEMM、batched GEMM，以及自有 kernel 不适用时的 fallback。

llama.cpp 自有 CUDA kernel：使用；MMVF/MMF/MMVQ/MMQ 覆盖小 batch、窄矩阵和量化矩阵乘，
                         并可直接使用 dp4a、ldmatrix、mma.sync 等硬件指令。
```

对典型量化模型而言，很多主要线性层会走自有 MMQ/MMVQ kernel；不能因为 CUDA backend 链接了 cuBLAS，就认为所有 `ggml_mul_mat` 都由 cuBLAS 计算。反过来，也不能因为存在大量自有 kernel，就认为 cuBLAS 完全没有参与。

## 35. `tCrA` descriptor Tensor 的物理存储、单线程大小和 CTA 大小

本节仍针对：

```text
./build-bjh100/examples/cute/tutorial/hopper/cute_tutorial_wgmma_tma_sm90_like \
    512 1024 2048 N T
```

以及：

```cpp
Tensor tCrA = thr_mma.make_fragment_A(tCsA);
```

代码位置为 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:240-251`（函数 `gemm_device`）。

### 35.1 直接结论

`tCrA` 不在 shared memory 中。准确数字如下：

| 统计口径 | 一个 thread | 一个 128-thread CTA |
|---|---:|---:|
| `sizeof(tCrA)` 的 C++ 对象表示 | 8 bytes | `128 x 8 = 1024` bytes 的源级聚合状态 |
| 通常用于保存 base descriptor 的 GPR | 2 个 32-bit registers | 256 个 32-bit registers，即 1024 bytes 的 register-file payload |
| `tCrA` 新增的 shared memory | 0 bytes | 0 bytes |
| 如果把 24 个逻辑 descriptor 错当成数组 | 误算为 `24 x 8 = 192` bytes | 误算为 24576 bytes；这是错误算法 |

这里必须区分：

1. `sizeof(tCrA)=8` 是当前具体 Tensor 类型的 C++ 对象大小，可以从类型定义确定。
2. “通常占两个 32-bit GPR”是正常优化代码的物理映射；最终寄存器复用、临时值消除或 spill 由 `ptxas` 决定。
3. `1024 bytes/CTA` 不是一块可以用 pointer 寻址的 CTA buffer，只是 128 个线程各自 8-byte descriptor 状态的源级合计。
4. 对 shared-memory 容量的答案是确定的 0 bytes；`tCrA` 只描述 shared memory，不存放在 shared memory。

### 35.2 先区分“Tensor 对象在哪里”和“Tensor 指向的数据在哪里”

“`tCsA` 在 shared memory”是一种常用但不够严格的简写。

实际上：

```text
tCsA Tensor 对象：每个 thread 的局部 view/metadata
tCsA 指向的 half 数据：CTA shared memory 中的 sA

tCrA Tensor 对象：每个 thread 的局部 DescriptorIterator/metadata
tCrA 解引用得到的 value：64-bit GmmaDescriptor
descriptor 所描述的 A 数据：仍然是 CTA shared memory 中的 sA
```

`tCsA` 由 `ThrMMA::partition_A` 创建。`include/cute/atom/mma_atom.hpp:475-484`（函数 `ThrMMA::partition_A`）把原 `sA.data()` 作为新 Tensor 的 data iterator：

```cpp
auto thr_tensor = make_tensor(atensor.data(), this->thrfrg_A(atensor.layout()));
```

因为 `sA.data()` 是 `smem_ptr`，`tCsA` 的元素访问最终读写 shared memory。`include/cute/pointer.hpp:145-161`（类型 `smem_ptr` 和 trait `is_smem`）把这种 iterator 标记为 shared-memory iterator。

但是 `tCsA` 这个 C++ view 对象本身并没有被 placement-new 到 `extern __shared__` 中；它只是每线程局部变量。它所引用的 `half_t` elements 才位于 shared memory。

`tCrA` 更进一步：它连 `smem_ptr` 都不再直接保存，而是保存一个已经编码了 shared-memory 地址和 layout 信息的 base `GmmaDescriptor`。日志 `temp/run.wgmma_tma_sm90_like.log:148-164` 的区别很明确：

```text
tCsA data engine: Sw<3,4,3>_smem_ptr[16b](...0400)
tCrA data engine: GMMA::DescriptorIterator
```

所以正确关系是：

```text
tCrA descriptor -> 描述 tCsA/sA 中的数据
```

而不是：

```text
tCrA descriptor -> 存储在 tCsA/sA 中
```

### 35.3 `tCrA` 的确切 C++ 类型结构

当前 MMA traits 在 `include/cute/atom/mma_traits_sm90_gmma.hpp:648-670`（特化 `MMA_Traits<SM90_64x64x16_F16F16F16_SS<...>>`）指定：

```cpp
using FrgTypeA = GMMA::smem_desc<tnspA>;
```

`MMA_Atom::make_fragment_A` 检测到这个 fragment type 可以解引用，于是在 `include/cute/atom/mma_atom.hpp:145-165`（函数 `MMA_Atom::make_fragment_A`）调用专用 `make_tensor<FrgTypeA>(atensor)`，而不是分配 value array。

专用构造位于 `include/cute/atom/mma_traits_sm90_gmma.hpp:361-372`（函数 `MakeTensor<SM90::GMMA::smem_desc<MajorMode>>::operator()`）：

```cpp
return make_tensor(
    GMMA::DescriptorIterator{make_gmma_desc<MajorMode>(tensor<0>(smem_tensor))},
    replace<0>(recast<uint128_t const>(smem_tensor).layout(), Layout<_1,_0>{}));
```

注意它传给通用 `make_tensor` 的 iterator 是 `DescriptorIterator`。`include/cute/tensor_impl.hpp:350-367`（函数 `MakeTensor<T>::operator()`）检测到 iterator 可解引用后，生成：

```text
Tensor<
  ViewEngine<GMMA::DescriptorIterator>,
  static descriptor layout
>
```

`ViewEngine` 在 `include/cute/tensor_impl.hpp:106-117`（类型 `ViewEngine`）只有一个运行时数据成员：

```cpp
iterator storage_;
```

而 `DescriptorIterator` 在 `include/cute/atom/mma_traits_sm90_gmma.hpp:303-331`（类型 `GMMA::DescriptorIterator`）也只有一个数据成员：

```cpp
GmmaDescriptor desc_;
```

最后，`GmmaDescriptor` 是一个以 `uint64_t desc_` 为完整表示的 union，定义在 `include/cute/arch/mma_sm90_desc.hpp:80-131`（类型 `GmmaDescriptor`）：

```cpp
union GmmaDescriptor {
  uint64_t desc_;
  uint32_t reg32_[2];
  uint16_t reg16_[4];
  // bitfield view...
};
```

所以运行时 payload 的链路为：

```text
tCrA Tensor
  -> ViewEngine
    -> DescriptorIterator
      -> GmmaDescriptor
        -> uint64_t desc_                 8 bytes
```

### 35.4 为什么静态 layout 没有再增加对象大小

`Tensor` 在 `include/cute/tensor_impl.hpp:135-150,330-341`（类型 `Tensor`）用下面的 tuple 保存 layout 和 engine：

```cpp
cute::tuple<layout_type, engine_type> rep_;
```

当前 `tCrA` 的 shape/stride 全部由 `_1`、`_2`、`_4`、`_3`、`_64`、`_256`、`_1024` 这类 compile-time integral constants 构成，因此它的 `Layout` 是 empty type。`include/cute/layout.hpp:95-109`（类型 `Layout`）也明确使用 empty-base optimization 保存静态 shape/stride。

更关键的是，`cute::tuple` 不会为 empty element 留一个普通 C++ 对象字节。`include/cute/container/tuple.hpp:42-76`（类型 `cute::tuple` 和 empty-structure optimization 说明）说明 empty template arguments 不被实际存储；`include/cute/container/tuple.hpp:111-120`（类型 `eso::ESO<true,false,...>`）在第一个元素为空、后续元素非空时只保存 `rest_`。

所以当前类型的大小关系是：

```text
sizeof(GmmaDescriptor)                = 8
sizeof(DescriptorIterator)             = 8
sizeof(ViewEngine<DescriptorIterator>) = 8
sizeof(decltype(tCrA))                  = 8
```

如果 layout 含有 runtime shape/stride，这个结论可能变化；但日志中的当前 `tCrA` layout 完全是 static layout，因此这里就是 8 bytes。

### 35.5 为什么逻辑上 24 个 descriptor，物理上却只保存一个

日志 `temp/run.wgmma_tma_sm90_like.log:160-164` 打印：

```text
tCrA:
GMMA::DescriptorIterator
o (_1,_2,_4,(_1,_3))
 : (_0,_64,_256,(_0,_1024))
```

因此 `size(tCrA)` 在逻辑上是：

```text
1 x 2 x 4 x 1 x 3 = 24
```

但这个 `24` 是 Tensor 的 logical domain size，不是 `ArrayEngine` 的元素个数。`tCrA` 使用的是 `ViewEngine<DescriptorIterator>`，只保存一个 base descriptor。

每次索引时，`Tensor::operator()` 在 `include/cute/tensor_impl.hpp:233-255`（成员函数 `Tensor::operator()`）先用 layout 得到 offset，再执行：

```cpp
data()[offset]
```

`DescriptorIterator::operator[]` 和 `operator+` 在 `include/cute/atom/mma_traits_sm90_gmma.hpp:311-330`（成员函数 `DescriptorIterator::operator[]` 和 `DescriptorIterator::operator+`）按需生成结果：

```cpp
ret.reg32_[0] = desc_.reg32_[0] + uint32_t(offset);
ret.reg32_[1] = desc_.reg32_[1];
```

即：

```text
24 个 logical descriptor
= 1 个 8-byte base descriptor
+ compile-time layout offsets
+ 索引时生成的一个 8-byte temporary descriptor
```

不存在：

```text
GmmaDescriptor descriptors[24];
```

因此不能计算成 `24 x 8 = 192 bytes/thread`。

调试代码中的：

```cpp
auto elem = tCrA(idx);
```

位于 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:299-305`（函数 `gemm_device`）。`elem` 是一个 8-byte `GmmaDescriptor` temporary；循环的四次迭代也不意味着四个 temporary 必须同时存活，编译器通常会复用相同寄存器。

### 35.6 descriptor 最终在哪种物理存储中

从 CuTe 的 memory-space dispatch 看，`tCrA` 属于 rmem。`include/cute/pointer.hpp:226-232`（trait `is_rmem`）把既不是 gmem 也不是 smem 的 iterator 归为 rmem；`DescriptorIterator` 正属于这一类。

SM90 GMMA 的专用 unpack 在 `include/cute/atom/mma_traits_sm90_gmma.hpp:385-429`（函数 `SM90::GMMA::mma_unpack`）也要求 A/B descriptor Tensor 满足 `is_rmem`，随后把它们 recast 为底层 MMA operation 的 register type。

当前 operation 在 `include/cute/arch/mma_sm90_gmma.hpp:409-430`（类型及函数 `MMA_64x64x16_F16F16F16_SS::fma`）声明：

```cpp
using ARegisters = uint64_t[1];
using BRegisters = uint64_t[1];

fma(uint64_t const& desc_a, uint64_t const& desc_b, ...)
```

并在 `include/cute/arch/mma_sm90_gmma.hpp:434-451`（函数 `MMA_64x64x16_F16F16F16_SS::fma`）用 inline-PTX constraint：

```cpp
"l"(desc_a), "l"(desc_b)
```

把 descriptor 作为 64-bit register operand 传给 `wgmma.mma_async`。因此至少在 WGMMA 指令发射时，`desc_a` 必须物化为寄存器操作数。常规 SASS 映射会用两个 32-bit general-purpose registers 表示一个 64-bit descriptor。

但是，“rmem”是 CuTe 的 dispatch 分类，不是 C++ 对某个物理寄存器编号的承诺。CUDA compiler 可以：

- 把 base descriptor 长时间保存在两个 GPR 中；
- 在使用点重新计算一部分 descriptor；
- 复用其他已经死亡的 registers；
- 在 register pressure 很高时 spill 到每线程 local memory/stack。

即使发生 spill，也会进入 thread-private local memory，而不是 CTA shared memory。

所以最严谨的说法是：

```text
源级对象空间：thread-local/rmem view
正常物理实现：register file
极端编译结果：可能 spill 到 per-thread local memory
shared memory：不会
```

### 35.7 一个 thread 和一个 CTA 应该怎样计数

#### 一个 thread

当前 `tCrA` Tensor 对象的确定大小是：

```text
8 bytes/thread
```

这 8 bytes 是一个 base descriptor，不是 24 个 descriptor。访问某个 logical element 时会生成一个 8-byte temporary，temporary 的 registers 可以与其他临时状态复用，不能简单与 base 永久相加。

#### 一个 CTA

当前 launch 使用：

```cpp
dim3 dimBlock(size(tiled_mma));
```

代码位于 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:419-428`（函数 `gemm_nt`）。当前 MMA traits 的 `ThrID` 是 `_128`，见 `include/cute/atom/mma_traits_sm90_gmma.hpp:656-670`（特化 `MMA_Traits<SM90_64x64x16_F16F16F16_SS<...>>`）；日志 `temp/run.wgmma_tma_sm90_like.log:55` 也确认 `dimBlock.x=128`。

因此按源级对象表示求和：

```text
128 threads x 8 bytes/thread = 1024 bytes/CTA
```

换成通常的 GPR payload：

```text
2 x 32-bit registers/thread x 128 threads
= 256 x 32-bit registers
= 1024 bytes of register-file payload
```

但这不是一个“CTA 的 1024-byte descriptor allocation”。每个 thread 有自己的寄存器上下文，哪怕 128 个线程中的 descriptor bit pattern 相同，也不能通过一个普通 shared pointer 把它们视为一份 CTA 对象。寄存器实际分配还会按 GPU/warp 的 allocation granularity 向上取整，并与 kernel 的其他寄存器一起计算。

对当前已构建的 `gemm_nt` cubin 执行：

```text
cuobjdump --dump-resource-usage \
  build-bjh100/examples/cute/tutorial/hopper/cute_tutorial_wgmma_tma_sm90_like
```

得到整个 kernel 的 `REG:81, STACK:304, LOCAL:0`。这只能说明该调试版本每线程的整体资源用量；报告不会给出“其中哪两个 registers 属于 tCrA”。因此不能用 81 个 registers 反推出 `tCrA` 的独立物理占用，最终变量级映射需要查看编译后的 SASS 和 live range。

### 35.8 为什么 `tCrA` 的 shared-memory 占用严格为 0

kernel 的 shared-memory storage 类型只声明了 A、B 和两组 barrier。`examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:55-64`（类型 `SharedStorage`）为：

```cpp
struct SharedStorage {
  alignas(128) ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
  uint64_t tma_barrier[PIPE];
  uint64_t mma_barrier[PIPE];
};
```

`gemm_device` 在 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:110-115`（函数 `gemm_device`）只把这一个 `SharedStorage` 映射到 `extern __shared__`。host 端又在 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:419-428`（函数 `gemm_nt`）用 `sizeof(SharedStorage)` 设置动态 shared-memory 大小。

当前参数下：

```text
A: 128 x 64 x 3 x 2 bytes = 49152 bytes
B: 128 x 64 x 3 x 2 bytes = 49152 bytes
barriers: 2 x 3 x 8 bytes = 48 bytes
struct alignment/padding: 80 bytes
-------------------------------------------------
SharedStorage total: 98432 bytes
```

这与日志 `temp/run.wgmma_tma_sm90_like.log:55` 的 `smem_size:98432` 完全一致。`SharedStorage` 中没有 `tCrA` 或 descriptor array，98432 bytes 也已经完全由 A、B、barriers 和 padding 解释。

因此：

```text
tCrA 对每个 CTA 的额外 dynamic shared memory = 0 bytes
```

`tCrA` 中的 64-bit 数字包含 shared-memory start address、stride 和 swizzle type；“descriptor 指向 shared memory”和“descriptor 自己存储在 shared memory”是两件不同的事。

### 35.9 最终汇总

```text
tCsA:
  Tensor 对象是 thread-local view；实际 A half values 位于 CTA shared memory。

tCrA:
  Tensor 对象也是 thread-local；只保存一个 8-byte base GmmaDescriptor 和零运行时大小的 static layout。
  24 个 logical descriptors 由 base descriptor + layout offset 按需生成，不是 descriptor[24]。

一个 thread:
  sizeof(tCrA) = 8 bytes；正常情况下对应两个 32-bit GPR，可能被编译器重算、复用或 spill。

一个 128-thread CTA:
  源级聚合 descriptor payload = 128 x 8 = 1024 bytes；它分散在各线程寄存器上下文中。
  tCrA 新增 shared memory = 0 bytes。
```

## Section 36. `tCsA` 的 48-bit 泛型指针如何编码进 14-bit GMMA start address

### 36.1 结论

不会因为这里的转换丢失地址。问题中的前提“把 `0x7fdc01000400` 的低 14 bit 直接保留下来”并不成立，实际有两次不同的地址表示转换：

```text
CUDA C++ generic pointer
    0x7fdc01000400
          |
          | __cvta_generic_to_shared / cvta.to.shared
          v
当前 CTA 的 shared-memory byte address
    0x00000400
          |
          | matrix-descriptor-encode(x): 省略恒为 0 的 4 个低位
          v
GMMA descriptor.start_address_[13:0]
    0x0040
```

其中：

```text
0x0040 << 4 = 0x0400
```

因此 `GmmaDescriptor` 的 14-bit `start_address_` 并不是“14-bit byte address”，而是一个以 16 bytes 为单位的、当前 CTA shared-memory 地址编码。WGMMA 也不需要恢复 `0x7fdc01000400` 这个泛型指针；它直接访问当前 CTA 的 `.shared` 地址空间。

### 36.2 为什么 `tCsA` 会打印一个看似 48-bit 的地址

`gemm_device` 先从 dynamic shared memory 构造 `sA`，然后 `partition_A` 只产生这个 shared-memory Tensor 的线程视图：

```cpp
extern __shared__ char shared_memory[];
SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});

Tensor tCsA = thr_mma.partition_A(sA);
```

代码位于：

- `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:110-115`（函数 `gemm_device`，构造 `sA`）；
- `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:240-242`（函数 `gemm_device`，构造 `tCsA`）。

`smem_ptr<T>` 是一个带有“这是 shared-memory 数据”类型信息的指针包装器。不过它的打印函数仍然打印包装器中保存的 CUDA C++ 指针 `ptr.get()`：

```cpp
template <class T>
CUTE_HOST_DEVICE void print(smem_ptr<T> ptr)
{
  printf("smem_"); print(ptr.get());
}
```

代码位于 `include/cute/pointer.hpp:371-375`（函数 `cute::print(smem_ptr<T>)`）。Tensor 的打印函数先打印 `tensor.data()`，再打印 layout，见 `include/cute/tensor_impl.hpp:1117-1121`（函数 `cute::print(Tensor<...>)`）。

CUDA C++ 中普通的 `T*` 使用 generic address 表示，所以日志中的：

```text
Sw<3,4,3>_smem_ptr[16b](0x7fdc01000400)
```

表示“能够在 CUDA C++ generic address space 中指向这块 shared memory 的指针”。它不是 WGMMA 所要求的 shared-space offset，更不是 shared-memory 芯片上的 48-bit 物理地址。其高位参与 generic address-space 的表示；在转换成 `.shared` 地址以后不再需要。

因此，“至少需要 48 bit 存储”只是在讨论 CUDA generic pointer 的表示；对本 CTA 内的这块 shared allocation，真正用于寻址的 offset 很小（本例起点是 `0x400`）。指针对象通常仍按 64-bit CUDA pointer 传递/保存，但这不等于 descriptor 需要保存同样宽度的地址。

### 36.3 `tCrA` 构造 descriptor 的准确调用路径

本例调用：

```cpp
Tensor tCrA = thr_mma.make_fragment_A(tCsA);
```

代码位于 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:249-251`（函数 `gemm_device`）。对于这个 SS GMMA，traits 把 `FrgTypeA` 定义成 `GMMA::smem_desc<tnspA>`，见 `include/cute/atom/mma_traits_sm90_gmma.hpp:656-670`（类型 `MMA_Traits<SM90_64x64x16_F16F16F16_SS<...>>`）。因此它走的是 descriptor view 分支：

1. `include/cute/atom/mma_atom.hpp:145-165`（函数 `MMA_Atom::make_fragment_A`）检测到 `FrgTypeA` 可以解引用，并调用 `make_tensor<FrgTypeA>(atensor)`；
2. `include/cute/atom/mma_traits_sm90_gmma.hpp:361-373`（函数 `MakeTensor<GMMA::smem_desc<MajorMode>>::operator()`）对 `tensor<0>(tCsA)` 调用 `GMMA::make_gmma_desc`；
3. `include/cute/atom/mma_traits_sm90_gmma.hpp:196-219`（函数 `GMMA::make_gmma_desc`）把 shared pointer 和 layout 元数据编码进 `GmmaDescriptor`。

最关键的实现是：

```cpp
// Start address (4LSB not included)
uint32_t start_address =
    cast_smem_ptr_to_uint(raw_pointer_cast(u128_tensor.data()));
desc.bitfield.start_address_ =
    static_cast<uint16_t>(start_address >> 4);
```

代码位于 `include/cute/atom/mma_traits_sm90_gmma.hpp:214-219`（函数 `GMMA::make_gmma_desc`）。这里先得到 `uint32_t` shared-space address，之后才右移 4 bit；代码没有把原始 generic pointer 强制转换成 14 bit。

### 36.4 `cast_smem_ptr_to_uint` 不是简单截断指针

`cast_smem_ptr_to_uint` 的首选实现调用 CUDA 的地址空间转换 intrinsic：

```cpp
return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
```

如果 intrinsic 不可用，其 inline PTX fallback 做的是同一件事：

```cpp
asm(
"{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; "
"cvt.u32.u64 %0, smem_ptr; }\n"
  : "=r"(smem_ptr) : "l"(ptr));
```

代码位于 `include/cute/arch/util.hpp:90-122`（函数 `cute::cast_smem_ptr_to_uint`）。操作次序是：

```text
generic address --cvta.to.shared--> shared-space address --cvt.u32--> uint32_t
```

这和下面这种错误理解完全不同：

```text
错误理解：generic_address & 0x3fff
```

CUDA 官方 Programming Guide 的 Address Space Conversion Functions 也明确说明：向期望 `.shared` 地址的 PTX 指令传地址之前，需要使用 `__cvta_generic_to_shared`；shared、local、constant 地址空间的范围较小，因此其非泛型表示可以保存在 32-bit 整数中。

### 36.5 用本次日志逐位验证

日志 `temp/run.wgmma_tma_sm90_like.log:148-149` 给出的 `tCsA` generic pointer 为：

```text
G = 0x7fdc01000400
```

日志 `temp/run.wgmma_tma_sm90_like.log:168-174` 又打印出第一个 A descriptor：

```text
GmmaDescriptor: 0x4000008000000040
  start_addr : 0x0040
  leading_off: 0x0000
  stride_off : 0x0080
  base_offset: 0x0
  layout_type: 0x1 (B128)
```

其地址转换和编码为：

```text
G = 0x7fdc01000400                 CUDA generic pointer

S = cvta.to.shared(G)
  = 0x00000400                    当前 CTA 的 shared byte address

start_address_ = S >> 4
               = 0x400 >> 4
               = 0x0040

硬件解码出的 16-byte-aligned 起点：
start_address_ << 4 = 0x0040 << 4 = 0x0400
```

如果真的把日志里的 generic 数值直接取低 14 bit，结果会是 `0x0400`，而不是 descriptor 打印的 `0x0040`；这正好反证了“直接截取 generic pointer”的理解是错误的。正确的输入是 `cvta.to.shared` 的结果 `S`，然后才执行 `S >> 4`。

日志中 A、B 的 generic pointer 分别是 `...0400` 和 `...c400`，相差 `0xc000 = 49152` bytes，见 `temp/run.wgmma_tma_sm90_like.log:78-82`。这也正好等于 `SharedStorage::A` 的大小：

```text
128 * 64 * 3 * sizeof(half) = 49152 bytes
```

`SharedStorage` 中 A、B 的声明位于 `examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:53-64`（类型 `SharedStorage`）。这说明 `...0400` 和 `...c400` 的低位确实是在描述同一个 CTA shared-memory window 内的 byte offsets，而高位属于 generic representation。

本次运行 A 从 shared offset `0x400` 开始，也与 Hopper 上 CUDA 为每个 block 保留 1 KiB shared memory 的规则一致；这 1 KiB 不属于用户的 `SharedStorage`。

### 36.6 14 bit 为什么足够：它实际覆盖 18-bit byte offset

`GmmaDescriptor` 的字段定义为：

```cpp
// start_address, bit [0,14), 4LSB not included
uint16_t start_address_ : 14, : 2;
```

代码位于 `include/cute/arch/mma_sm90_desc.hpp:103-125`（类型 `cute::GmmaDescriptor`）。PTX 的 matrix descriptor 编码规则等价于：

```text
matrix-descriptor-encode(x) = (x & 0x3ffff) >> 4
```

所以 start field 保存的是 shared byte address 的 bit `[17:4]`：

```text
14 encoded bits + 4 implicit zero bits = 18-bit byte-address range
```

可表示的 16-byte-aligned 起始地址范围为：

```text
最小值：0x0000 << 4 = 0x00000
最大值：0x3fff << 4 = 0x3fff0
覆盖窗口：小于 0x40000 bytes，也就是小于 256 KiB
```

H100 每个 thread block 最多可使用 227 KiB shared memory，小于这个 256 KiB descriptor address window；本例实际只申请 98432 bytes，日志见 `temp/run.wgmma_tma_sm90_like.log:55`。因此有效的 Hopper CTA shared-memory 地址不会因为这个字段发生截断冲突。

4 个低位能够省略，是因为 GMMA descriptor 的 start address、leading offset 和 stride offset 都要求/使用 16-byte 粒度。CUTE 也先把 Tensor recast 为 `uint128_t const`，见 `include/cute/atom/mma_traits_sm90_gmma.hpp:203-216`（函数 `GMMA::make_gmma_desc`）；`uint128_t` 正好是 16 bytes。

这里仍有两个使用约束：

1. 起始地址必须满足 GMMA layout 所需的对齐；任意未对齐 generic pointer 不能合法地编码成 descriptor。
2. 地址及 layout 产生的访问必须落在当前 CTA 已分配的 shared-memory 区间内。超过范围不是靠“更多 generic pointer 高位”解决，而是无效的 descriptor/越界访问。

### 36.7 WGMMA 如何找到数据：CTA 上下文是隐式的

`GmmaDescriptor` 是一个 64-bit 寄存器值；除 start address 外，还包含 leading offset、stride offset、base offset 和 swizzle mode，完整 bitfield 见 `include/cute/arch/mma_sm90_desc.hpp:103-125`（类型 `cute::GmmaDescriptor`）。

本例最终的 wrapper 把 `desc_a` 和 `desc_b` 作为两个 64-bit register operands 传给 PTX：

```cpp
wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 ...
```

其 `fma` 参数和 inline assembly 位于 `include/cute/arch/mma_sm90_gmma.hpp:416-451`（函数 `GMMA::MMA_64x64x16_F16F16F16_SS::fma`）。其中约束：

```cpp
"l"(desc_a),
"l"(desc_b)
```

表示每个 descriptor 作为一个 64-bit register operand 进入指令。

对于本例的本 CTA shared memory，可把硬件定位过程概念化为：

```text
当前执行 CTA 的 shared-memory window
              +
(descriptor.start_address << 4)
              +
由 leading/stride、MMA core-matrix 坐标产生的 offset
              +
B128 swizzle 对 bank/地址位的映射
              =
此次 WGMMA 读取的 shared-memory location
```

关键是第一行：“当前执行 CTA 的 shared-memory window”由硬件执行上下文隐式确定。不同 CTA 即使都使用 descriptor start field `0x0040`，也会访问各自 CTA 的 shared offset `0x400`，不会访问同一块全局内存。WGMMA 从来不需要把 `0x0040` 反向恢复成某个 CUDA generic pointer。

### 36.8 `tCrA` 后续 logical descriptor 如何移动起始地址

`tCrA` 日志为：

```text
GMMA::DescriptorIterator o (_1,_2,_4,(_1,_3)):
                           (_0,_64,_256,(_0,_1024))
```

`DescriptorIterator::operator+` 只对 descriptor 的低 32 bit 加 layout 给出的 offset，高 32 bit 的 stride/swizzle 元数据保持不变：

```cpp
ret.reg32_[0] = desc_.reg32_[0] + uint32_t(offset);
ret.reg32_[1] = desc_.reg32_[1];
```

代码位于 `include/cute/atom/mma_traits_sm90_gmma.hpp:303-329`（函数 `GMMA::DescriptorIterator::operator+`）。这些 offset 也已经是 16-byte descriptor units。例如本次日志中的前几个值：

```text
tCrA(0): start field 0x0040 -> shared byte address 0x0400
tCrA(1): start field 0x0080 -> shared byte address 0x0800
tCrA(2): start field 0x0140 -> shared byte address 0x1400
tCrA(3): start field 0x0180 -> shared byte address 0x1800
```

对应打印位于 `temp/run.wgmma_tma_sm90_like.log:168-198`。这再次说明 iterator 操作的是“已经编码好的 CTA-relative shared offsets”，不是 48-bit generic pointers。

### 36.9 最终区分

```text
tCsA.data() 打印值：
  CUDA C++ generic pointer representation
  示例：0x7fdc01000400

cast_smem_ptr_to_uint 的结果：
  当前 CTA shared-memory address-space 中的 byte address
  本例：0x00000400

GmmaDescriptor.start_address_：
  shared byte address 的 bit [17:4]
  本例：0x0040

WGMMA 实际定位：
  当前 CTA shared-memory window + descriptor 编码的 offset/layout
  不会也不需要恢复 generic pointer 的高位
```

因此，丢弃的是在 GMMA shared-address 语境中不需要的 generic-address-space 表示，以及由 16-byte 对齐保证为 0 的 4 个低位；有效的 CTA-relative shared-memory 地址信息没有丢失。

外部规范依据：NVIDIA [PTX ISA 8.0 的 Matrix Descriptor Format](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html) 和 [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)。
