# `examples/cute/tutorial/hopper/wgmma_tma_sm90.cu` 讲解

## 1. 文件概览

这是 CUTLASS 3.x / CuTe 在 **Hopper (SM90)** 架构上的 GEMM tutorial 示例，展示了 TMA (Tensor Memory Accelerator)、GMMA (Group MMA)、warp group 同步、cluster launch 和 software pipeline 等 SM90 特有特性。

代码路径（相对 cutlass code base）：`examples/cute/tutorial/hopper/wgmma_tma_sm90.cu`

---

## 2. 代码结构

| 函数/结构体 | 行号 | 作用 |
|------------|------|------|
| `struct SharedStorage` | 52-63 | 定义 shared memory 存储布局（A/B tile + TMA/MMA barrier） |
| `gemm_device` kernel | 70-260 | 核心 GEMM kernel |
| `gemm_nt` | 262-344 | host 端 NT 布局（A 列主序，B 行主序转置）的配置与 launch |
| `gemm_tn` | 346-426 | host 端 TN 布局（A 行主序转置，B 列主序）的配置与 launch |
| `gemm` | 428-446 | 根据 transA/transB 分发到 gemm_nt 或 gemm_tn |
| `main` | 448-561 | 入口，参数解析与正确性/性能测试 |

---

## 3. 逐段详解

### 3.1 SharedStorage（行 52-63）

```cpp
template <class ElementA, class ElementB,
          class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};
```

- **A/B 数组**：`alignas(128)` 保证 shared memory 的 128 字节对齐（TMA 要求）。
- **cosize_v**：编译期计算 smem layout 所需元素总数。
- **tma_barrier / mma_barrier**：每种 barrier 有 `K_PIPE_MAX`（= pipeline 深度，本例为 3）个实例，分别用于 TMA producer 和 MMA consumer 的同步。

### 3.2 gemm_device kernel（行 70-260）

#### 3.2.1 模板参数

```cpp
template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
```

- `ProblemShape` / `CtaTiler`：问题规模和 CTA tile 大小（本例 M=128, N=128, K=64）
- `SmemLayoutA/B`：shared memory 的 swizzle layout
- `TmaA/B`：TMA copy atom（host 端通过 `make_tma_atom` 创建）
- `TiledMma`：`SM90_64x64x16_F16F16F16_SS` MMA 指令（warp-group 级别，128 线程）

#### 3.2.2 全局张量构造与分块（行 99-114）

```cpp
Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));  // 行 99
Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));  // 行 100
Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);  // 行 101

auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);  // 行 104
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // 行 105
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // 行 106
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // 行 107
```

- `get_tma_tensor`：创建 TMA 感知的全局内存张量（TMA 需要知道全局内存的 shape 和 stride）
- `local_tile` + `Step`：将问题按 CTA tile 分块。`Step` 控制三类 CTA 在 M/N/K 维度上的遍历方式——NT 布局中 M 方向由 blockIdx.y 遍历、N 方向由 blockIdx.x 遍历、K 方向由循环遍历。

#### 3.2.3 Shared Memory 张量（行 110-114）

```cpp
extern __shared__ char shared_memory[];
SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)
```

通过 `extern __shared__` 动态分配 shared memory（kernel launch 时指定大小 `smem_size`）。

#### 3.2.4 TMA 分区（行 128-136）

```cpp
auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                  group_modes<0,2>(sA), group_modes<0,2>(gA));
auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                  group_modes<0,2>(sB), group_modes<0,2>(gB));
```

- `tma_partition`：TMA 专用的分区函数，返回 `(源张量, 目标张量)` pair
- `Int<0>` + `Layout<_1>`：表示**不使用 TMA multicast**（单播模式）
- `group_modes<0,2>`：将按 K 分块的 tensor `(BLK_M, BLK_K, k_tiles) → ((BLK_M,BLK_K), k_tiles)` 合并前两维，因为 **TMA 负责 mode-0 的全部数据搬运**
- `tma_transaction_bytes`（行 135-136）：计算一次 TMA 搬运的总字节数（A+B 的 mode-0 大小之和）

#### 3.2.5 Barrier 初始化（行 128-165）

```cpp
using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;  // TMA 用
using ConsumerBarType = cutlass::arch::ClusterBarrier;             // MMA 用

// warp 0 的 elected lane 初始化所有 pipe 的 barrier
for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
        ProducerBarType::init(&producer_mbar[pipe], 1);     // 期望 1 个 transaction
        ConsumerBarType::init(&consumer_mbar[pipe], 128);   // 期望 128 个线程 arrive
    }
}
cluster_sync();  // 确保 barrier 初始化在 cluster 内所有 CTA 完成
```

两种 barrier 各有用途：
- **ClusterTransactionBarrier**：TMA 完成时自动 arrive（硬件机制），mmA consumer 等待它
- **ClusterBarrier**：MMA consumer 显式 `arrive`，TMA producer 等待它
- `cluster_sync()` 保证 cluster 内所有 CTA 的 barrier 初始化完成

#### 3.2.6 预填充 Pipeline（行 167-180）

```cpp
for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
        ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
        copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
        copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
    }
    --k_tile_count;
    ++k_tile;
}
```

- 在主循环前预先发出 `K_PIPE_MAX` 个 TMA 异步加载请求，填满 pipeline
- `tma_a.with(producer_mbar[pipe])`：将 TMA transaction 与 barrier 绑定，TMA 完成时自动 arrive barrier
- `arrive_and_expect_tx`：设置该 barrier 期望接收的字节数

#### 3.2.7 MMA Fragment 分区（行 183-203）

```cpp
ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
Tensor tCsA = thr_mma.partition_A(sA);       // (MMA, MMA_M, MMA_K, PIPE)
Tensor tCsB = thr_mma.partition_B(sB);       // (MMA, MMA_N, MMA_K, PIPE)
Tensor tCgC = thr_mma.partition_C(gC);       // (MMA, MMA_M, MMA_N)

Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // accumulator registers
clear(tCrC);

Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // MMA descriptor fragment
Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // MMA descriptor fragment
```

**SM90 的关键区别**：
- `tCrA/tCrB` **不是寄存器 fragment**，而是 **MMA Descriptor**——这是指向 shared memory 的描述符结构
- GMMA 指令直接从 shared memory 读取，不需要中间的 `s2r` copy（寄存器到寄存器的加载）
- 因此不需要 `copy(tCsA, tCrA)` 这样的显式加载指令

对比 SM80 代码中必须 `cute::copy(s2r_copy, tAsA, tCrA_view)` 先加载到寄存器，SM90 的 GMMA 可以直接从 smem 读。

#### 3.2.8 Pipeline 主循环（行 205-253）

```cpp
auto write_state = cutlass::PipelineState<K_PIPE_MAX>();  // 行 217: TMA write 状态
auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();  // 行 218: MMA  read  状态

CUTE_NO_UNROLL
while (k_tile_count > -K_PIPE_MAX)  // 行 221
{
    // --- Consumer: MMA ---
    int read_pipe = read_state.index();
    ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());  // 等待 TMA 完成
    warpgroup_arrive();                                                     // warp group 开始同步
    gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);       // GMMA 指令
    warpgroup_commit_batch();                                              // 提交异步批次
    warpgroup_wait<0>();                                                   // 等待 MMA 完成
    ConsumerBarType::arrive(&consumer_mbar[read_pipe]);                    // 通知 producer
    ++read_state;

    // --- Producer: TMA (仅 warp 0 的 elected lane) ---
    if ((warp_idx == 0) && lane_predicate && (k_tile_count > 0)) {
        int pipe = write_state.index();
        ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());  // 等待 MMA consumer 完成
        ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
        copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
        copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
        ++write_state;
    }
    --k_tile_count;
    ++k_tile;
}
```

**Pipeline 运转机制**：

```
        TMA (producer)                              MMA (consumer)
        warp 0 elected lane                         warp group (128 threads)
        ──────────────────                          ──────────────────────────
        TMA load → pipe[0]                          ...
        TMA load → pipe[1]                          ...
        TMA load → pipe[2]                          ...
                                                     wait TMA done → pipe[0]
                                                     warpgroup_arrive
                                                     gemm(A[pipe0], B[pipe0], C)
                                                     warpgroup_commit_batch
                                                     warpgroup_wait<0>
                                                     arrive consumer_bar[0]
        wait consumer done → pipe[0]
        TMA load → pipe[0] (reuse)
                                                     wait TMA done → pipe[1]
                                                     gemm(A[pipe1], B[pipe1], C)
                                                     ...
                                                     arrive consumer_bar[1]
        wait consumer done → pipe[1]
        TMA load → pipe[1] (reuse)
                                                     wait TMA done → pipe[2]
                                                     ...
```

`PipelineState` 管理循环索引和 phase 位翻转：`index()` 返回当前 pipe 编号，`phase()` 跟踪 pipeline 的奇偶轮次，避免 ABA 问题。

`warpgroup_arrive/commit_batch/wait<0>` 是 Hopper 特有的 warp group 同步机制：
- `warpgroup_arrive`：warp group 开始一个异步批量操作
- `gemm(...)`：发出异步 GMMA 指令（立即返回，不阻塞）
- `warpgroup_commit_batch`：提交异步批次
- `warpgroup_wait<0>`：等待批次 0 完成（阻塞直到所有 GMMA 指令完成）

#### 3.2.9 Epilogue（行 259）

```cpp
axpby(alpha, tCrC, beta, tCgC);  // C = alpha * accum + beta * C
```

直接使用 `axpby` 将累加器寄存器写回全局内存 C，无需如 SM80 那样经 shared memory 中转。

### 3.3 gemm_nt（行 262-344）

- **Tile 配置**：`bM=128, bN=128, bK=64`，pipeline depth `bP=3`
- **Shared Memory Layout**：`GMMA::Layout_MN_SW128_Atom<TA>{}` 产生 SM90 GMMA 兼容的 shared memory swizzle layout（按 128 字节粒度交错，消除 bank conflict）
- **MMA 指令**：`SM90_64x64x16_F16F16F16_SS` 表示：
  - M=64, N=64, K=16 的 warp-group 级 tile
  - F16 输入/F16 累加
  - `S`=Sparse=A/B 均为 smem 直接读取（GMMA 特性）
  - 一个 warp group = 4 warps = 128 threads
- **TMA Atom**：通过 `make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK))` 创建。这个调用在 **host 端** 检查 TMA 的合法性（global→smem copy 的对齐、tile 大小、数据类型等），TMA descriptor 编码进常量内存
- **Cluster Launch**：dimCluster=(2,1,1)，2 个 CTA 共享同一组 barrier（通过 `cluster_sync` 同步）

### 3.4 gemm_tn（行 346-426）

与 `gemm_nt` 类似，区别：
- 矩阵 A 的 stride 和 shared memory layout 改为 K-major：`GMMA::Layout_K_SW128_Atom`
- MMA 指令改用 K-major：`SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>`

---

## 4. 是否使用了 Warp Specialization 技术？

### 结论：**没有。**

### 分析

#### 什么是 Warp Specialization？

Warp specialization 是指将 thread block 内的不同 warp **持久地**分配不同角色——例如某些 warp 只做 producer（全局→共享内存搬运），另一些 warp 只做 consumer（MMA 计算），且 producer 和 consumer **同时并发执行**，通过显式同步协调。

伪代码示例：
```cpp
if (warp_id == PRODUCER_WARP) {
    while (...) {
        wait consumer done on pipe;
        TMA load → pipe;
        arrive TMA done on pipe;
    }
} else { // CONSUMER_WARPS
    while (...) {
        wait TMA done on pipe;
        warpgroup_arrive;
        gemm(pipe);
        warpgroup_wait<0>;
        arrive consumer done on pipe; // ⚠ 仅 consumer warp arrive (不是全 warp group)
    }
}
```

#### 为什么这个 tutorial 不是 Warp Specialization？

**证据 1：Producer 和 Consumer 串行执行（行 221-253）**

主循环结构是：
```
while (k_tile_count > -K_PIPE_MAX) {
    // ① 所有 warp 做 consumer (MMA)
    ProducerBarType::wait(...);
    warpgroup_arrive();
    gemm(mma, ...);              // ← warp group 全 128 线程参与
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    ConsumerBarType::arrive(...); // ← warp group 全 128 线程 arrive

    // ② warp 0 的 elected lane 做 producer (TMA)
    if (warp_idx == 0 && lane_predicate && k_tile_count > 0) {
        // 只发 TMA 指令
    }
}
```

关键点：
- **所有 128 线程**都执行 consumer 路径（`warpgroup_arrive/commit/wait` 要求 warp group 全参与）
- **然后**才检查 `warp_idx == 0` 执行 producer 路径
- producer 和 consumer **串行执行**，不存在并发

**证据 2：warpgroup_arrive 需要全 warp group（行 228）**

`warpgroup_arrive()` 是 Hopper 的 warp group 级别同步原语，以 `__syncwarp()` 语义要求 warp group 内所有 4 个 warp 都执行到此处才能继续。这说明所有线程都走到了 MMA 路径。

**证据 3：ConsumerBarrier 期望 128 个线程 arrive（行 161）**

```cpp
ConsumerBarType::init(&consumer_mbar[pipe], 128);  // 期望 128 个线程
```

barrier 的预期 arrive 数是 128（= 整个 warp group 的所有线程），说明全 CTA 都参与 consumer arrive。

如果是 warp specialization，应为：
```cpp
ConsumerBarType::init(&consumer_mbar[pipe], 96);  // 只有 3 个 consumer warp = 96 线程
```

**证据 4：TMA 由 warp 0 elected lane 发起是硬件约定，不是 specialization**

`warp_idx == 0 && lane_predicate` 只是表明 TMA 指令只需一个线程发出（TMA 是异步的，一个线程发出后即可继续执行），其他 warp 没有对应的 TMA 工作要做，所以跳过 `if` 继续执行。并不是它们被"专门化"为不同的角色。

**证据 5：tutorial 自己的注释（行 212）**

```cpp
//   More advanced pipeline and warp-specialization strategies are available in CUTLASS mainloops.
```

明确说明**本 tutorial 不包含 warp-specialization**，更高级的 pipeline 和 warp-specialization 策略存在于 CUTLASS 的其他 mainloop 实现中。

#### 那么它的架构应该叫什么？

这个 kernel 使用的是 **Warp-Group Synchronous Pipeline**（warp group 同步流水线），而非 warp specialization。特点是：
- 全 warp group（128 线程）统一行动：先 MMA 消费，再 TMA 生产
- 通过 `PipelineState` + barrier 实现多级软件流水（K_PIPE_MAX=3）
- TMA 和 MMA 通过 barrier 显式同步，但在同一个 warp group 内串行交替

在 CUTLASS 项目中，真正的 warp specialization GEMM 可以参考 `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized*.hpp` 等头文件。

---

## 5. 与 SM80 GEMM 的关键差异

| 特性 | SM80 (Ampere) | SM90 (Hopper) |
|------|-------------|--------------|
| 数据搬运 | `cp.async` (LDSM) | **TMA** (硬件异步搬运) |
| 从 smem 到 MMA | `ldmatrix` → 寄存器 fragment | **GMMA** 直接从 smem 读，tCrA/tCrB 是 descriptor 不是数据 |
| 写回全局 | reg → smem → global（需中转） | `axpby` 直接 reg → global |
| 同步 | `cp_async_wait` + `__syncthreads` | **barrier** + `warpgroup_wait` |
| 多 CTA 协作 | 不支持 | **cluster launch** + `cluster_sync` |
| 流水线控制 | 手动管理 `ismem_read/write` | `PipelineState` |

---

## 6. 总结

`wgmma_tma_sm90.cu` 是 CUTLASS 在 Hopper 上的入门级 GEMM tutorial，演示了 SM90 架构的核心特性：TMA（Tensor Memory Accelerator）、GMMA（Group MMA）、warp group 同步、barrier 软件流水线、cluster launch。它是一个 **warp-group synchronous pipeline** 实现，**没有使用 warp specialization**——所有 warp 统一先做 MMA 消费，再通过 warp 0 elected lane 发出 TMA 负载，两阶段串行交替。

真正的 warp specialization GEMM 在 CUTLASS 代码库中有专门的 `*_warpspecialized_*` 实现文件，它们使用持久化的 producer/consumer warp 分工和并发执行。

