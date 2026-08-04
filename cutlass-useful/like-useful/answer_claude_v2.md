
---

## `gemm-multi-stage-like.cu` 第 280-343 行讲解（Epilogue: 累加器写回全局内存）

### 总览

这段代码是 multi-stage pipelined GEMM kernel 的 **epilogue 阶段**。在完成所有 K 维度的主循环计算后，每个线程的寄存器中存放着累加结果 `tCrD`（形状 `(MMA=8, MMA_M=4, MMA_N=4)`，本用例中每线程持有 128 个元素），需要将这些结果写回到全局内存 `gD` 中。

关键设计思路：
- **复用 A 的 shared memory 做 scratchpad**：主循环结束后 `sA` 不再需要，直接用其底层指针 + `SmemLayoutC` layout 构造 `sC`，零额外显存开销。
- **寄存器 → shared memory → 全局内存 的 two-step 路径**：利用 shared memory 做中间缓冲区，使得写全局内存时可以使用 **128-bit wide store**（一次写 8 个 `half`），比逐元素写高效得多。
- **双缓冲流水线**：`kSmemLayoutCBatch=2`，两个 pipeline stage 交替使用，一个 stage 正在被寄存器写入的同时，另一个 stage 可以被读出发往全局内存。

```
寄存器 tCrD (per-thread, 128 elems)
      │
      ▼  ┌─────────────────────┐
         │  R2SCopyAtomC       │  UniversalCopy<int>  (每次复制 2 个 half)
         │  reg → shm          │
         ▼                     │
   sC (复用 sA 的 shm, 双缓冲)  │
         │                     │
         ▼  ┌─────────────────────┐
         │  S2GCopyC             │  UniversalCopy<uint128_t> (每次复制 8 个 half)
         │  shm → global         │
         ▼
全局内存 gD
```

---

### 逐行解析

#### 280-282: 复用 Shared Memory 构造 sC

```cpp
// use less shared memory as a scratchpad tile to use large wide instuction
// Dreg -> shm -> reg -> global
auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});
```

**作用**：复用 A 矩阵 shared memory 中当前已读完的 pipe（`ismem_read`）的底层指针，用 `SmemLayoutC` 重新解释这块内存。

**关键数据**（来自运行日志）：
- `sA(_, _, ismem_read).data()` 返回的原始指针：`0x7f7200004400`
- `SmemLayoutC` = `Sw<2,3,3> o (_32, _32, _2) : (_32, _1, _1024)`
  - 这是一个 32×32 的 scratchpad tile，带 2 个 pipeline stage
  - cosize = 32×32×2 = 2048 个 element = 4096 字节
  - 每个 stage 占据连续的 1024 个 element（2048 字节）
- 静态断言保证 `cosize(SmemLayoutA) >= cosize(SmemLayoutC)`，即 sA 的单个 pipe 足以容纳 sC

**为什么能这样复用？** 主循环结束后，sA 的 shared memory 数据已经全部消费完毕，不再需要。同时 sC 只需要约 4KB 空间，远小于 sA 的 128×32=4096 个 element（8KB）。

---

#### 284-287: 寄存器到 Shared Memory 的 Copy 计划 (R2S)

```cpp
auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);   // (CPY, CPY_M, CPY_N)
auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);  // (CPY, _1, _1, pipe)
```

**`make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma)`**：创建一个与 MMA 输出布局兼容的 tiled copy。

其中：
- `R2SCopyAtomC` = `Copy_Atom<UniversalCopy<int>, T>`：每个原子操作复制 `_2` 个元素（32-bit 传输），即一次复制 2 个 `half`

**运行日志中的实际值**：
```
r2s_tiled_copy_c:
  Tiler_MN:       (_32,_32)
  TiledLayout_TV: ((_4,_8,_2,_2),((_2,_2),(_1,_2))):((_64,_1,_16,_256),((_32,_8),(_0,_512)))
Copy_Atom:
  ValLayoutSrc: (_1,_2):(_0,_1)   // 每次复制 2 个 half
  ValLayoutDst: (_1,_2):(_0,_1)
```

**`get_slice(idx)`**：获取当前线程 1 的 slice。

**`retile_S(tCrD)`**：将累加器寄存器张量按 copy 的源布局重新分块：
```
tCrC_r2s:
  ptr o ((_2,(_2,_2)),_4,_4):((_1,(_2,_16)),_4,_32)
```
- Mode 0 = `CPY` = `(_2,(_2,_2))` → 共 8 条 copy 指令
- Mode 1 = `CPY_M` = `_4` → 每条 copy 覆盖 M 方向的 4 个元素
- Mode 2 = `CPY_N` = `_4` → 每条 copy 覆盖 N 方向的 4 个元素
- 总计 per-thread = 8×4×4 = **128 个元素**（与 tCrD 一致）

**`partition_D(sC)`**：将共享内存 sC 按 copy 的目标布局分块：
```
tCsC_r2s:
  smem_ptr o ((_2,(_2,_2)),_1,_1,_2):((_1,(_256,16)),_0,_0,_1024)
```
- Mode 0 = `CPY` = `(_2,(_2,_2))` → 与源对应
- Mode 1,2 = `_1` → 标量（每个 CPY entry 对应共享内存中唯一的一个位置）
- Mode 3 = `pipe` = `_2` → 双缓冲 pipeline stage
- 各 CPY entry 在 shm 中的偏移：`a×1 + b×256 + c×16 + p×1024`，均落在 sC 的合法范围内

---

#### 289-292: Shared Memory 到全局内存的 Copy 计划 (S2G)

```cpp
S2GCopyC s2g_tiled_copy_c;
auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);  // (CPY, _1, _1, pipe)
auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);  // (CPY, CPY_M, CPY_N)
```

**`S2GCopyC`** 使用更宽的 copy atom：
- `Copy_Atom<UniversalCopy<cute::uint128_t>, T>`：每次复制 `_8` 个元素（128-bit 传输）→ 一次写 8 个 `half`
- `Tiler_MN: (_32, _32)`，tile 粒度与 MMA 的 C 输出对齐

**运行日志中的实际值**：
```
s2g_tiled_copy_c:
  Tiler_MN:       (_32,_32)
  TiledLayout_TV: ((_4,_32),_8):((_256,_1),_32)
Copy_Atom:
  ValLayoutSrc: (_1,_8):(_0,_1)   // 每次复制 8 个 half
  ValLayoutDst: (_1,_8):(_0,_1)
```

**`partition_S(sC)`**：共享内存作为 S2G copy 的源：
```
tCsC_s2g:
  smem_ptr o ((_8,_1),_1,_1,_2):((_1,_0),_0,_0,_1024)
```
- 与 R2S 的分块不同：S2G 的 copy atom 更宽，因此 CPY_M/N 维度的分块粒度不同（这里 CPY_M=CPY_N=`_1`，CPY=`_8`）
- pipe 维度同样是 `_2`

**`partition_D(gD)`**：全局内存作为 S2G copy 的目标：
```
tCgC_s2g:
  gmem_ptr o ((_8,_1),_4,_4):((_1,_0),8192,_32)
```
- Mode 0 = `CPY` = `(_8,_1)` → 8 条 copy
- Mode 1 = `CPY_M` = `_4`
- Mode 2 = `CPY_N` = `_4`
- 这个布局直接映射到全局内存 gD 的 tile 区域

---

#### 294-295: group_modes 扁平化

```cpp
auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);  // (CPY_, CPY_MN)
auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);  // (CPY_, CPY_MN)
```

**作用**：将 Mode 1 (CPY_M) 和 Mode 3 (CPY_N) (注意这里 mode 3 是原来的最后一个 mode — 实际上是把共索引器可能返回的 4 个 mode 中的第 1 和第 3 合并) 从 `(CPY_M, CPY_N)` 展平为 `CPY_MN`，使得原本二维的 M×N 区域变成一维的"元素序列"，便于流水线式顺序处理。

**实际结果**：
```
tCgC_s2gx: gmem_ptr o ((_8,_1),(_4,_4)):((_1,_0),(8192,_32))  // 128 elements
tCrC_r2sx: ptr      o ((_2,(_2,_2)),(_4,_4)):((_1,(_2,_16)),(_4,_32))  // 128 elements
```

---

#### 297: step —— Pipeline 深度

```cpp
int step = size<3>(tCsC_r2s);  // pipe
```

**输出**：`step=2`

`tCsC_r2s` 的最内层 mode（索引 3）是 pipe 维度，大小为 2。这个值决定了流水线的级数，也是外层循环的步长。

---

#### 322-343: Pipelined 写回循环

```cpp
#pragma unroll
for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
    // reg -> shm
    #pragma unroll
    for (int j = 0; j < step; ++j) {
        auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
        cute::copy(tCrC_r2sx(_, i + j), t);
        cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
    }
    __syncthreads();

    // shm -> global
    #pragma unroll
    for (int j = 0; j < step; ++j) {
        cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }
    __syncthreads();
}
```

**执行逻辑**：

从日志可知 `size<1>(tCrC_r2sx) = 16`，`step = 2`，因此外层循环执行 **16/2 = 8 次**，i = 0, 2, 4, 6, 8, 10, 12, 14。

每次外层迭代执行一个完整的双缓冲流水线阶段：

```
iteration i=0:  │  iteration i=2:  │  ...  │  iteration i=14:
                │                  │       │
 R2S j=0:       │  R2S j=0:       │       │  R2S j=0:
  shm pipe[0]   │   shm pipe[0]   │       │   shm pipe[0]
  ← reg[0]      │   ← reg[2]      │       │   ← reg[14]
                │                  │       │
 R2S j=1:       │  R2S j=1:       │       │  R2S j=1:
  shm pipe[1]   │   shm pipe[1]   │       │   shm pipe[1]
  ← reg[1]      │   ← reg[3]      │       │   ← reg[15]
                │                  │       │
 __syncthreads  │  __syncthreads  │       │  __syncthreads
                │                  │       │
 S2G j=0:       │  S2G j=0:       │       │  S2G j=0:
  gmem[0]       │   gmem[2]       │       │   gmem[14]
  ← shm pipe[0] │   ← shm pipe[0] │       │   ← shm pipe[0]
                │                  │       │
 S2G j=1:       │  S2G j=1:       │       │  S2G j=1:
  gmem[1]       │   gmem[3]       │       │   gmem[15]
  ← shm pipe[1] │   ← shm pipe[1] │       │   ← shm pipe[1]
                │                  │       │
 __syncthreads  │  __syncthreads  │       │  __syncthreads
```

**第 329-330 行的拷贝和类型转换**：
```cpp
auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
cute::copy(tCrC_r2sx(_, i + j), t);
```
- `make_tensor_like<T>` 创建一个与源张量形状相同、但元素类型为 `T` 的临时寄存器张量
- `cute::copy` 做逐元素赋值，处理累加器类型（可能是 `float`）和输出类型（`half`）之间的转换
- 然后将这个中间张量 `t` 传给 R2S copy，写入 shared memory

**两个 `__syncthreads()` 的作用**：
1. 第一个 sync：确保所有线程的 R2S 写操作完成后，共享内存对全 warp/block 可见
2. 第二个 sync：确保 S2G 读操作完成后，下一轮迭代可以安全覆写共享内存

---

### 与 Config 的关联

| 配置项 | 值 | 用途 |
|--------|-----|------|
| `R2SCopyAtomC` | `Copy_Atom<UniversalCopy<int>, T>` | reg→shm: 每次复制 2 个 half (32-bit) |
| `S2GCopyAtomC` | `Copy_Atom<UniversalCopy<cute::uint128_t>, T>` | shm→global: 每次复制 8 个 half (128-bit) |
| `SmemLayoutC` | `tile_to_shape(SmemLayoutAtomC{}, (32,32,2))` | 32×32 scratchpad tile, 2-stage pipeline |
| `kSmemLayoutCBatch` | 2 | 双缓冲 pipeline stage 数 |
| `kMmaPM`, `kMmaPN` | 32, 32 | MMA 单次处理 C 的 M×N 大小 |

### 为什么走 shm 中转？

如果直接从寄存器写全局内存（`reg → global`），每次只能写 2 个 `half`（32-bit），因为 `R2SCopyAtomC` 的原子操作粒度只有 32-bit。通过 shared memory 中转后，可以用 `S2GCopyC` 的 128-bit wide store 写入全局内存，大幅提升**内存带宽利用率**。Shared memory 的带宽远高于全局内存，因此 `reg → shm` 的开销可被 `shm → global` 的收益抵消。

### 正确性验证

运行日志最终验证：`check ok, max_error = 0.000000`，三个实现（our-impl / cublas / cublaslt）输出结果完全一致。


---

## 追问：R2S copy 如何用 32-bit atom 完成 8 个 half 的复制？

### 问题重述

第 330-332 行：

```cpp
auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
cute::copy(tCrC_r2sx(_, i + j), t);
cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
```

其中 `tCrC_r2sx(_, i+j)` 从 8 个 CPY 条目中各取 1 个元素，得到 **8 个 half**。`R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>` 每次只能复制 2 个 half（即 1 个 32-bit int）。那么 `cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j))` 是如何完成 8 个 half 的复制的？

### 核心答案

**`cute::copy` 配合 TiledCopy 会在内部自动迭代，将 8 个元素拆成 4 组、每组 2 个元素，每组调用一次 copy atom，共 4 次 32-bit store。**

### 详细推导

#### Step 1: 源张量 `t` 的形状

从日志：

```
auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);   // (CPY, CPY_M, CPY_N)
ptr[16b] o ((_2,(_2,_2)),_4,_4):((_1,(_2,_16)),_4,_32)
```

`tCrC_r2s` 把每线程的 128 个累加器元素组织为 `CPY=8, CPY_M=4, CPY_N=4`（8×4×4=128）。

```
auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);  // (CPY_, CPY_MN)
ptr[16b] o ((_2,(_2,_2)),(_4,_4)):((_1,(_2,_16)),(_4,_32))
```

`group_modes<1,3>` 将 CPY_M 和 CPY_N 合并为 `CPY_MN=(_4,_4)=16`。结果：`CPY_=8, CPY_MN=16`。

`tCrC_r2sx(_, i+j)` 取 mode 1 位置 `i+j`，保留全部 mode 0（8 个 CPY 条目），得到一个 **8 元素的 1D tensor**。

#### Step 2: 目标张量 `tCsC_r2s(_, 0, 0, j)` 的地址分布

从日志：

```
auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);  // (CPY, _1, _1, pipe)
smem_ptr[16b](0x7f7200004404) o ((_2,(_2,_2)),_1,_1,_2):((_1,(_256,16)),_0,_0,_1024)
```

`tCsC_r2s(_, 0, 0, j)` 给出 8 个 shared memory 地址。根据 strides `(_1, _256, _16, _1024)` 计算：

| CPY 条目 | shared memory 地址偏移 |
|----------|----------------------|
| CPY=0: (0,(0,0)) | base + 0    |
| CPY=1: (1,(0,0)) | base + 1    |
| CPY=2: (0,(1,0)) | base + 256  |
| CPY=3: (1,(1,0)) | base + 257  |
| CPY=4: (0,(0,1)) | base + 16   |
| CPY=5: (1,(0,1)) | base + 17   |
| CPY=6: (0,(1,1)) | base + 272  |
| CPY=7: (1,(1,1)) | base + 273  |

**关键发现：8 个目标地址天然形成 4 对连续地址：**
- (CPY=0, CPY=1): base+0, base+1
- (CPY=2, CPY=3): base+256, base+257
- (CPY=4, CPY=5): base+16, base+17
- (CPY=6, CPY=7): base+272, base+273

这正是为什么每个 atom 能写入 2 个 consecutive half —— 因为 `retile_S` 在分区时已经将连续的对地址配对好了。

#### Step 3: TiledCopy 内部迭代

`r2s_tiled_copy_c` 的定义（日志）：

```
TiledCopy
  Tiler_MN:       (_32,_32)
  TiledLayout_TV: ((_4,_8,_2,_2),((_2,_2),(_1,_2))):(...)
Copy_Atom
  ValLayoutSrc: (_1,_2):(_0,_1)   // 每次读取 2 个连续元素
  ValLayoutDst: (_1,_2):(_0,_1)   // 每次写入 2 个连续元素
```

当执行 `cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j))` 时，TiledCopy 根据其内部迭代空间（由 `Tiler_MN` 和 `TiledLayout_TV` 决定）对源/目标进行分区。每个 atom 的 `ValLayoutSrc/Dst` 为 `(_1,_2)`，表示每次调用处理 2 个连续元素。

对于本线程的 8 元素源 + 8 地址目标：

```
源 t (8 half, 连续):        目标 shm (8 addr, 4组连续对):

  t[0] ————┐               shm[base+0]   ─┐
  t[1] ————┤  atom #1      shm[base+1]   ─┘  (连续)
            │  UniversalCopy<int>
  t[2] ————┤               shm[base+256] ─┐
  t[3] ————┘  atom #2      shm[base+257] ─┘  (连续)

  t[4] ————┐               shm[base+16]  ─┐
  t[5] ————┤  atom #3      shm[base+17]  ─┘  (连续)
            │  UniversalCopy<int>
  t[6] ————┤               shm[base+272] ─┐
  t[7] ————┘  atom #4      shm[base+273] ─┘  (连续)
                     UniversalCopy<int>
```

**8 个 half = 4 次 atom 调用，每次 1 个 32-bit store（2 half）。**

### 为什么中间需要一个临时 tensor `t`？

```cpp
auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
cute::copy(tCrC_r2sx(_, i + j), t);
cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
```

代码注释已经说明了原因：
> we add a temp tensor to cope with accumulator and output data type difference

累加器可能是 `float` 类型（即使输入是 `half`，MMA 累加通常用 `float`），而输出是 `half`。`make_tensor_like<T>` 创建一个与源形状相同、但元素类型为 `T` 的临时寄存器张量。`cute::copy(tCrC_r2sx(_, i+j), t)` 逐元素复制时会自动做 float→half 的类型转换。然后把类型正确的 `t` 传给 R2S copy。

### 总结

| 问题 | 答案 |
|------|------|
| 源有几个元素？ | 8 个 half（来自 8 个 CPY 条目 × 1 个 CPY_MN 位置） |
| atom 每次能复制几个？ | 2 个 half（1 个 `int` = 32-bit） |
| 怎么复制 8 个？ | TiledCopy 内部迭代 4 次，每次取 2 个元素调用 atom |
| 为什么目标能接受 2-连续写入？ | `retile_S` 已确保 CPY 条目按连续对分组 |

