# 为什么 shm_seq 打印显示走 kTaskLoopPolicy=2 分支，但 kernel 内打印 kTaskLoopPolicy=1

## 结论

这不是矛盾，而是一个 C++ **bool 窄化 (narrowing)** 的 bug：主机端把 `kTaskLoopPolicy` 声明成了 `bool` 而不是 `int`。

`constexpr bool kTaskLoopPolicy = 2;` 中，整数 `2` 赋给 `bool` 会被隐式转换为 `true`（即整数值 `1`）。当这个 `bool` 值作为模板实参传给形如 `int kTaskLoopPolicy` 的 kernel 模板参数时，`true` 再转换回 `int`，得到 `1`。

于是：

- 主机端 `if / else if / else` 分支判断用的是那个 `bool` 变量。`task_map_ptr == nullptr` 且 `k > 1024 && n > 1024`，于是命中 `else` 分支，这个分支里 `constexpr bool kTaskLoopPolicy = 2`，所以 `shm_seq = sizeof(int) * (num_group + 1) = 4 * 9 = 36` 被打印出来 —— 这让你以为走了 policy == 2。
- 但真正实例化 kernel 时，模板实参 `kTaskLoopPolicy` 已经是 `bool = true`，传给 `int` 模板形参后变成 `1`，所以 kernel 内部 `if constexpr (kTaskLoopPolicy == 1)` 分支被编译/执行，printf 输出 `kTaskLoopPolicy:1`。

## 证据（代码定位）

1. 主机端声明为 `bool`：
   - `src/group_gemm/group_gemm_pertensor_fp8.cu:281` `constexpr bool kTaskLoopPolicy = 0;`（`launch_group_gemm_fp8` 函数内，PDL 分支）
   - `src/group_gemm/group_gemm_pertensor_fp8.cu:296` `constexpr bool kTaskLoopPolicy = 1;`
   - `src/group_gemm/group_gemm_pertensor_fp8.cu:309` `constexpr bool kTaskLoopPolicy = 2;` ← `2` 被窄化成 `true`
   - 非 PDL 分支同样重复了三个 `bool` 声明：`:333` / `:347` / `:359`

2. kernel 模板形参声明为 `int`：
   - `src/group_gemm/kernels.cuh:215` `template <... int kTaskLoopPolicy, bool kUsePDL = false>`（`group_gemm_fp8_kernel`）

3. 实例化时 bool 传给 int：
   - `src/group_gemm/group_gemm_pertensor_fp8.cu:321-322` `kernels::group_gemm_fp8_kernel<..., kTaskLoopPolicy, kUsePDL>`（`launch_group_gemm_fp8`）

4. kernel 内按 int 值分派：
   - `src/group_gemm/kernels.cuh:353` `else if constexpr (kTaskLoopPolicy == 1)`（`group_gemm_fp8_kernel`）—— 实际命中的分支
   - `src/group_gemm/kernels.cuh:357` `else if constexpr (kTaskLoopPolicy == 2)` —— 从未命中
   - `src/group_gemm/kernels.cuh:414-415` printf 打印的正是模板形参 `kTaskLoopPolicy`，其值此时为 `1`

## 日志佐证（temp/run.log）

- `run.log:75` `shm_seq:36, config.get_shm_size():192512, shm_size:192548`
  `36 = sizeof(int) * (num_group + 1) = 4 * (8 + 1)`，正是 policy==2 分支的 `shm_seq` 公式（`group_gemm_pertensor_fp8.cu:310`），说明主机端确实进了 `else`（policy==2）分支。
- `run.log:73` `task_map_ptr(nil)`，且 k=7168、n=4096 均大于 1024，满足进入 `else` 分支的条件（`group_gemm_pertensor_fp8.cu:295` 的 `k <= 1024 || n <= 1024` 为假）。
- `run.log` 中 kernel 打印全是 `kTaskLoopPolicy:1`，且 `total_m:0`。
  如果真是 policy==2，`total_m = cu_tiles_ptr[num_group]`（`kernels.cuh:358`）应为 576（`run.log:654` 的 `cu_seqlens` 末值），而日志中 `total_m` 恒为 0，进一步证明实际执行的是 policy==1 分支（该分支从不给 `total_m` 赋值，保持初值 0，见 `kernels.cuh:326`）。

## 修复建议

把主机端这 6 处 `constexpr bool kTaskLoopPolicy` 改成 `constexpr int kTaskLoopPolicy`（即 `group_gemm_pertensor_fp8.cu` 的 `:281`、`:296`、`:309`、`:333`、`:347`、`:359`），与 kernel 模板的 `int` 形参保持一致；或者反过来把 `kernels.cuh:215` 的模板形参改成 `bool`。二选一即可，前者改动最小、语义最清晰。

---

# get_next_tile_vert 的功能与参数业务含义（policy == 2 的分块策略）

## 结论

`get_next_tile_vert`（`src/group_gemm/kernels.cuh:42`）是 group GEMM 在 `kTaskLoopPolicy == 2` 时使用的**"垂直（列优先）分块"**函数：它把一个线性递增的 block 编号 `iblock`（即 `blockIdx.x`）映射到一个具体要计算的 GEMM 瓦片 `(igroup, itile_m, itile_n)` —— 即"哪个 group（expert）、这个 group 内的第几个 M-tile、第几个 N-tile"。

之所以叫 "vert"（垂直），是相对 policy == 1 用的 `get_next_tile_horizon`（`src/group_gemm/kernels.cuh:22`，"水平/行优先"）而言的：两者都把全部 `(M-tile, N-tile)` 二维瓦片摊平成一条一维的 block 序列，只是遍历方向相反。

## 函数签名与调用关系

```cuda
__device__ __forceinline__ void get_next_tile_vert(const int *cu_tiles_ptr, int iblock,
                                                   int num_group, int &igroup, int &itile_m,
                                                   int &itile_n, int total_m);
```

调用点有两处，均在 `group_gemm_fp8_kernel`（`src/group_gemm/kernels.cuh:218`）内：

- load warpgroup（生产者）：`src/group_gemm/kernels.cuh:407`
- math warpgroup（消费者）：`src/group_gemm/kernels.cuh:486`

调用前，policy == 2 分支先把前缀和数组搬进 shared memory（`src/group_gemm/kernels.cuh:357-362`）：

```cuda
total_m = cu_tiles_ptr[num_group];
for (int i = idx; i < (num_group + 1); i += blockDim.x) {
  shm_tiles[i] = cu_tiles_ptr[i];
}
```

即 `shm_tiles`（作为 `cu_tiles_ptr` 传入）存放的是每个 group 的 M-tile 数**前缀和**，`total_m` 是全体 group 的 M-tile 总数。

## 输入参数业务含义

| 形参 | 业务含义 |
|------|----------|
| `cu_tiles_ptr` | 指向 shared memory 中的**前缀和**数组，长度 `num_group+1`。`cu_tiles_ptr[i]` = 前 i 个 group（expert）的 M-tile 数之和（不含第 i 个），即第 i 个 group 在"展平后的全局 M 维"里的起始行-tile 偏移。由 `update_grouped_tma`（`src/group_gemm/kernels.cuh:65`）在 `src/group_gemm/kernels.cuh:165` 写入（对 `tiles[]` 做 `ExclusiveSum` 后的结果），`cu_tiles_ptr[num_group]` 写入的是全体 M-tile 总数（`src/group_gemm/kernels.cuh:169`） |
| `iblock` | `blockIdx.x`（线性 block 编号），业务上可看作一个"全局线性 tile id"，在整个 grid 的一维任务序列里的序号 |
| `num_group` | group（expert）个数 |
| `total_m` | 所有 group 的 M-tile 总数，等于 `cu_tiles_ptr[num_group]` |

补充：M-tile 数来自序列长度，`tiles[i] = (seqlens_ptr[igroup] + kTileM - 1) / kTileM`（`src/group_gemm/kernels.cuh:149`），即每个 group 的 token 序列按 `kTileM` 向上取整切成多少个 M 瓦片。`seqlens` / `cu_seqlens` 见测试 `tests/test_group_gemm_pertensor_like.py:49`、`:61`。

## 输出参数业务含义（均为引用，就地写回）

| 形参 | 业务含义 |
|------|----------|
| `igroup` | 本 block 应处理的 group（expert）编号 |
| `itile_m` | 该 group 内的第几个 M-tile（行瓦片，0 起，局部偏移） |
| `itile_n` | 该 group 内的第几个 N-tile（列瓦片，0 起，范围 `[0, num_tile_n)`） |

三者在 `group_gemm_fp8_kernel` 里直接决定加载/计算的地址：`itile_m` 索引 A（激活）的 M 瓦片、`itile_n` 索引 B（权重）的 N 瓦片、`igroup` 索引权重 W 的第 3 维 `n*k` 分片（`src/group_gemm/kernels.cuh:426-430`），以及 `y_scale` 的 per-group 缩放 `yscale_ptr[igroup]`（`src/group_gemm/kernels.cuh:497`）。

## 具体算法（逐行语义）

```cuda
int itile_m_total = iblock % total_m;   // 全局展平的 M-tile 位置（跨 group 的行-tile 序号）
itile_n = iblock / total_m;             // N-tile 列号：每 total_m 个 block 覆盖一整列 M

// 在 cu_tiles_ptr 前缀和里二分，找 itile_m_total 落在哪个 group 的区间
int left = 0;
int right = num_group;
while (left <= right) {
  int mid = left + (right - left) / 2;
  if (cu_tiles_ptr[mid] > itile_m_total) right = mid - 1;
  else left = mid + 1;
}

itile_m = itile_m_total - cu_tiles_ptr[right];  // 减去该 group 的起始偏移，得到局部 M-tile 号
igroup = right;
```

核心是 `itile_m_total = iblock % total_m`、`itile_n = iblock / total_m` 这两行：**N 维（列）是外层循环、M 维（行）是内层循环**。block 0..`total_m-1` 依次扫过所有 group 的第 0 个 N 列的全部 M-tile，block `total_m`..`2*total_m-1` 扫第 1 个 N 列，依此类推。二分查找把"全局行号" `itile_m_total` 还原成 `(igroup, 局部 itile_m)`。

终止条件由调用方判断 `if (itile_n >= num_tile_n) break;`（`src/group_gemm/kernels.cuh:410-412`、`:487-489`）：当 `iblock` 超过 `total_m * num_tile_n` 时 `itile_n` 越界，循环退出。

## 与 get_next_tile_horizon（policy == 1）的对比

| | `get_next_tile_horizon`（horizon，水平） | `get_next_tile_vert`（vert，垂直） |
|---|---|---|
| 位置 | `src/group_gemm/kernels.cuh:22` | `src/group_gemm/kernels.cuh:42` |
| 输入数组 | `tiles_ptr`（**每 group 的 M-tile 个数**，非前缀和） | `cu_tiles_ptr`（**前缀和**） |
| 摊平方式 | `flat_divider(itile_m_total, itile_n, iblock)`：`itile_n = iblock % num_tile_n`，N 是**内层**（行优先，横着扫） | `itile_n = iblock / total_m`，N 是**外层**（列优先，竖着扫） |
| 查找 group 方式 | 从上次 `igroup` 起线性累加 `tiles_ptr[i]` 直到覆盖 | 对前缀和数组二分查找 |

一句话记忆：**horizon = N 快变（横向扫），vert = M 快变（纵向扫）**。

## 选择 vert 策略的条件

在 `launch_group_gemm_fp8`（`src/group_gemm/group_gemm_pertensor_fp8.cu:27`）里，policy == 2 分支进入条件是 `task_map_ptr == nullptr` 且 `!(k <= 1024 || n <= 1024)`（`src/group_gemm/group_gemm_pertensor_fp8.cu:295`），即 k、n 都较大（>1024）时才用垂直分块。结合 run.log 的 `k=7168, n=4096`（`temp/run.log:8,11`）与 `num_tile_n = 32`（`temp/run.log:647`）可印证。具体为何大 k/n 场景倾向列优先，代码本身未注释，此处不做过度推断。

---

# group_gemm_fp8_kernel 里两处 `tma.with(...)` 分别命中哪个重载，以及 `expect_tx` 与 `tma copy` 的先后顺序

## 结论

1. `tma_a.with(td_x, readable[ismem_write])`（`src/group_gemm/kernels.cuh:437`）命中的是 **2 参重载**：`Copy_Traits<SM90_TMA_LOAD>::with(TmaDescriptor const* new_tma_desc, uint64_t& tma_mbar, uint16_t multicast_mask = 0, CacheHint ...)`，位于 `3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:138`（注释 `temp. overloaded for grouped gemm/ptr array gemm`）。

2. `tma_b.with(readable[ismem_write])`（`src/group_gemm/kernels.cuh:440`）命中的是 **1 参重载**：`Copy_Traits<SM90_TMA_LOAD>::with(uint64_t& tma_mbar, uint16_t multicast_mask = 0, CacheHint ...)`，位于 `3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:127`。

3. 两者都要多一个"为什么一个传 2 参、一个传 1 参"的根因：**A 矩阵每个 expert 的 gmem 基地址/形状随 seqlens 变化，需要在运行期换 descriptor；B 矩阵把 expert 编号编码成 gmem 张量的第 3 维，一个 descriptor 覆盖全部 expert，无需换**。所以 `tma_a.with` 多传一个 `TmaDescriptor const*`（指向 `update_grouped_tma` 预生成的 per-group descriptor），`tma_b.with` 只用给 barrier。

4. `expect_tx` 与 `tma copy` 的先后：**两种顺序都正确**，因为 `set_barrier_transaction_bytes`（cute）与 `arrive_and_expect_tx`（cutlass）本质都是同一条 `mbarrier.arrive.expect_tx.shared::cta.b64` 指令，它同时完成"arrive"和"expect_tx"两步，且是在 TMA 的 `complete_tx` 之前就已被发出。但 **tutorial（`arrive_and_expect_tx` 在前、copy 在后）是更推荐/更符合惯例的顺序**，CUTLASS 生产级 warp-specialized mainloop 也是这么做的。

## `with` 是怎么被解析到的：`TiledCopy` 继承自 `Copy_Atom`

`tma_a` / `tma_b` 的类型是 `TiledCopy<...>`（由 `make_tma_copy` 生成），它继承自 `Copy_Atom`（`3rd/cutlass/include/cute/atom/copy_atom.hpp:188`），自身没有定义 `with`。`.with(...)` 实际调用的是 `Copy_Atom::with`（`3rd/cutlass/include/cute/atom/copy_atom.hpp:80-82`）：

```cuda
template <class... TraitsArgs>
CUTE_HOST_DEVICE
auto with(TraitsArgs&&... args) const {
  auto traits = Traits::with(static_cast<TraitsArgs&&>(args)...);
  return Copy_Atom<decltype(traits), CopyInternalType>{traits};
}
```

它把参数原样转发给底层 `Copy_Traits<SM90_TMA_LOAD, ...>::with(...)` 去匹配重载。所以真正参与重载决议的是 `Copy_Traits` 里的几个 `with`。

`make_tma_copy(SM90_TMA_LOAD{}, x, ...)` 生成的是"不可执行"的 `Copy_Traits<SM90_TMA_LOAD, ...>`（只有 `tma_desc_` 和 `aux_params_`，没有 barrier，`copy_unpack` 被 `= delete`，见 `3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:156-162`）。必须调用 `with` 补上 `tma_mbar`（以及可选的新 descriptor）才得到可执行的 `Copy_Traits<SM90_TMA_LOAD_OP, ...>`。

该结构体里的三个 `with` 重载（`3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp`）：

| 行号 | 签名 | 作用 |
|------|------|------|
| `:127` | `with(uint64_t& tma_mbar, uint16_t multicast_mask = 0, CacheHint = EVICT_NORMAL)` | 只补 barrier（+默认 multicast mask / cache hint），descriptor 用编译期 baked-in 的那个 |
| `:138` | `with(TmaDescriptor const* new_tma_desc, uint64_t& tma_mbar, uint16_t multicast_mask = 0, CacheHint = EVICT_NORMAL)` | 换 descriptor + 补 barrier（注释明确"temp. overloaded for grouped gemm/ptr array gemm"） |
| `:167` | `with(TmaDescriptor const* new_tma_desc) const` | 只换 descriptor，不补 barrier，返回的仍是不可执行的 `SM90_TMA_LOAD` |

## 逐处匹配

### `tma_b.with(readable[ismem_write])` → 命中 `:127`

- `readable` 是 `__shared__ uint64_t readable[kStage]`（`src/group_gemm/kernels.cuh:249`），所以实参 `readable[ismem_write]` 是 `uint64_t&`。
- `:127` 第一形参 `uint64_t& tma_mbar` 精确匹配（左值引用，无需转换），后两个有默认值，1 个显式实参即可。
- `:138` 至少要 2 个实参，不匹配；`:167` 的形参是 `TmaDescriptor const*`，`uint64_t&` 不能隐式转成指针，不匹配。
- 唯一候选 = `:127`。

### `tma_a.with(td_x, readable[ismem_write])` → 命中 `:138`

- `td_x = td_xy + igroup * 2`（`src/group_gemm/kernels.cuh:430`），`td_xy` 是 `cute::TmaDescriptor*`（kernel 形参，`src/group_gemm/kernels.cuh:223`），所以 `td_x` 是 `TmaDescriptor*`，可隐式转成 `TmaDescriptor const*`。
- 第二个实参 `readable[ismem_write]` 是 `uint64_t&`。
- `:138` 的前两个形参 `(TmaDescriptor const*, uint64_t&)` 精确匹配，后两个有默认值。
- `:127` 第一形参是 `uint64_t&`，`TmaDescriptor*` 转不成 `uint64_t&`，不匹配；`:167` 只有 1 个形参，实参有 2 个，不匹配。
- 唯一候选 = `:138`。

## 为什么 `tma_a` 需要换 descriptor，`tma_b` 不需要

两者的 TMA 源张量不同，见 `src/group_gemm/config.h:91-95` 的 `get_tma`：

```cuda
auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, x, take<0, 2>(SLayoutX{}));   // A：X 是 (m, k)
auto tma_w = make_tma_copy(SM90_TMA_LOAD{}, w, take<0, 2>(SLayoutW{}));   // B：W 是 (n, k, num_group)
```

- **A（X 激活）**：源张量是 `(m, k)`，`make_tma_copy` 把 descriptor 的 gmem 基地址烤死在整块 X 的首地址上。但 group GEMM 里每个 expert 的 token 行是"变长拼接"的（seqlens 不同），第 `igroup` 个 expert 的行在 X 里的起始偏移 = `cu_seqlens[igroup]`。所以每个 group 需要一份"基地址 + 形状都改过"的 descriptor。这些 descriptor 由前一个 kernel `update_grouped_tma`（`src/group_gemm/kernels.cuh:70`）预生成，写到 `td_xy` 数组里；`td_x = td_xy + igroup*2` 取第 `igroup` 个 group 的 A-descriptor（`+1` 是 `td_y`，见 `src/group_gemm/kernels.cuh:577`）。因此 `tma_a` 必须用 `:138` 这个"换 descriptor + 补 barrier"的重载，把运行期算出来的 `td_x` 传进去。
- **B（W 权重）**：源张量是 `(n, k, num_group)`，expert 编号就是第 3 维。`tBg(_, itile_n, itile_k, igroup)`（`src/group_gemm/kernels.cuh:440`）通过 TMA 坐标直接选中第 `igroup` 个 expert 的 `n×k` 分片，一份 descriptor 覆盖全部 group，无需运行期换。所以 `tma_b` 只需 `:127` 补 barrier 即可。

（对照：`tma_y` 是 `SM90_TMA_STORE`，`make_tma_copy(SM90_TMA_STORE{}, y, CopyBoxY{})`，见 `config.h:94`，它的 descriptor 同样在 `src/group_gemm/kernels.cuh:577` 用 `td_y = td_xy + igroup*2 + 1` 换，与 `tma_a` 同理。）

## `expect_tx` 与 `tma copy` 谁先谁后

两边发出的其实是**同一条指令**：

- `set_barrier_transaction_bytes`（cute，`3rd/cutlass/include/cute/arch/copy_sm90_desc.hpp:78-87`）内联汇编是 `mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;`。
- `ProducerBarType::arrive_and_expect_tx`（cutlass，`3rd/cutlass/include/cutlass/arch/barrier.h:588-602`）内联汇编也是 `mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;`（寄存器顺序不同，语义相同）。

即两者都是"**arrive + expect_tx**"合并的一条指令：它既让该 barrier 完成一次 arrive（`readable` 用 `initialize_barrier(readable[idx], 1)` 初始化，`src/group_gemm/kernels.cuh:327`，只需 1 次 arrive），又把本阶段两个 TMA 事务的总字节数 `kTransactionBytes` 加到 `expected-tx` 计数上。之后两个 `cute::copy` 各 issue 一条 `cp.async.bulk.tensor`，TMA 引擎搬完数据时用 `complete_tx` 把 tx 计数扣回去；当 `arrive-count==0` 且 `tx-count==0` 时相位翻转，消费者 `wait_barrier(readable[...])` 才能通过。

关键点在于顺序约束：`expect_tx` 必须在 TMA 的 `complete_tx` 生效**之前**写入，否则 tx 计数会被先扣成负、barrier 提前翻转。由于：

- TMA 拷贝是异步的，`cute::copy` 只是把 `cp.async.bulk.tensor` 指令 issue 出去，真正的 `complete_tx` 发生在 DMA 搬完（远晚于 issue 时刻）；
- `set_barrier_transaction_bytes` / `arrive_and_expect_tx` 是普通指令，发出后立刻执行。

所以 **hpc-ops 的"先 copy、后 set_barrier_transaction_bytes"（`kernels.cuh:437-443`）和 tutorial 的"先 arrive_and_expect_tx、后 copy"（`/share/users/like/package/cutlass/examples/cute/tutorial/hopper/wgmma_tma_sm90_like.cu:222-224`）在功能上都正确**——两种写法里 `expect_tx` 都先于 `complete_tx` 生效。

## 谁更推荐：tutorial 的顺序

推荐 **tutorial / CUTLASS 生产代码的顺序：先 `arrive_and_expect_tx`，再发 copy**。理由：

1. `expect_tx`（arrive）与 TMA issue 之间不留任何"异步事务可能先完成"的窗口，语义上最稳。
2. 这是 CUTLASS 生产级 warp-specialized mainloop 的写法：`sm90_mma_tma_gmma_ss_warpspecialized.hpp:374` 先 `pipeline.producer_acquire(...)`（其内部在 `:512-517` 调 `full_barrier_ptr_[stage].arrive_and_expect_tx(...)`），然后 `:384-385` 才 `copy(tma_load_a.with(...))`、`copy(tma_load_b.with(...))`。与 tutorial 完全一致。
3. hpc-ops 的"先 copy 后 set"虽然正确，但把 arrive/expect_tx 放到了两个 `cp.async.bulk.tensor` 之后，读起来不如"先声明期望字节、再发事务"直观，且是少数派写法（hpc-ops 仓库里 `src/attention/prefill/kernels.cuh`、`src/gemm/sm90/gemm_bf16xfp32.cu` 等也都是 copy 后 `set_barrier_transaction_bytes`，属同一风格）。

一句话：**这条指令叫 `mbarrier.arrive.expect_tx`，官方惯例是"先 arrive+expect_tx，后发 TMA copy"（tutorial 那样）；hpc-ops 的顺序也能跑，但 tutorial 的顺序更推荐**。


---

# TMA 版 `cute::copy`：`tBg` 里到底有没有 base address，Copy_Atom 与 gmem tensor 各自提供什么

## 结论

1. **`tBg(_, itile_n, itile_k, igroup)` 里没有 global base address，只有坐标。** `tBg` 来自 `gB = tma_b.get_tma_tensor(make_shape(n, k, num_group))`（`src/group_gemm/kernels.cuh:266`），而 `get_tma_tensor` 内部用 `make_coord_tensor` 生成（`3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:151-153`）：`make_coord_tensor` 把一个 **counting iterator（坐标发生器，0 偏移）**绑定到 layout 上（`3rd/cutlass/include/cute/tensor_impl.hpp:484-486`），根本不含数据指针。所以 `tBg` 的 `data()` 是坐标迭代器，取出的 `src(Int<0>{})` 是一组坐标 `(itile_n, itile_k, igroup)` 经 `g_stride_` 映射后的 TMA 坐标 `crd0..crd4`。

2. **`tBg` 的作用就是只生成 TMA coordinate，且这个 coordinate 甚至被忽略了它自己的 data 指针。** 看 `TMA_LOAD_Unpack::copy_unpack`（`3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:71-89`）：`auto src_coord = src(Int<0>{});` 只读坐标，`void* dst_ptr = raw_pointer_cast(dst.data());` 读的是目标 smem 的地址——**源 gmem tensor 的 `data()`（指针）从头到尾没被使用**。

3. 三个参数分别给 TMA 硬件提供的信息：

| 参数 | 提供的硬件信息 |
|------|----------------|
| 第 1 个参数 `Copy_Atom`（`tma_b.with(...)` 的返回值） | ① TMA descriptor：gmem **base address + global shape + byte stride + smem box shape + swizzle**（这些在 `make_tma_copy_desc` 里经 `cuTensorMapEncodeTiled` 烤进描述符，`3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:946/1050`）；② smem **mbarrier 指针**（`uint64_t*`，来自 `readable[ismem_write]`）；③ **cache hint**（L2 cache_hint，默认 `EVICT_NORMAL`） |
| 第 2 个参数 gmem tensor `tBg(...)` | 只有 **TMA 坐标** `crd0..crd4`（对应汇编里的 `[desc, {crd0,crd1,...}]`） |
| 第 3 个参数 smem tensor `tBs(...)` | 只有 **smem 目标地址** `smem_ptr`（汇编里的 `[%0]`，即 `dst_ptr`） |

## 逐层证据

### 1. `get_tma_tensor` 生成的是"坐标张量"，不是"数据张量"

`tma_b` 是 `make_tma_copy(SM90_TMA_LOAD{}, w, take<0,2>(SLayoutW{}))` 的返回值（`src/group_gemm/config.h:93`），类型是 `TiledCopy<Copy_Atom<Copy_Traits<SM90_TMA_LOAD, NumBitsPerTMA, AuxParams>, ...>, ...>`。它的 `get_tma_tensor`（`3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:149-154`）：

```cuda
template <class GShape>
CUTE_HOST_DEVICE constexpr
auto get_tma_tensor(GShape const& g_shape) const {
  static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
  return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
}
```

`make_coord_tensor` 定义（`3rd/cutlass/include/cute/tensor_impl.hpp:481-487`）：

```cuda
template <class Layout, __CUTE_REQUIRES(is_layout<Layout>::value)>
CUTE_HOST_DEVICE constexpr
auto make_coord_tensor(Layout const& layout) {
  return make_tensor(make_inttuple_iter(coprofile(layout)), layout);
}
```

`make_inttuple_iter` 是坐标迭代器，`coprofile(layout)` 是 0 偏移 profile——即 **engine 是"坐标发生器"，layout 是 TMA 坐标空间的映射（`aux_params_.g_stride_`）**。所以 `gB` 的元素值就是坐标，不是地址。

### 2. `copy_unpack` 只取源张量的坐标、丢弃其指针

`3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:71-89`：

```cuda
copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
            Tensor<TS,SLayout>           const& src,   // <- tBg(...) 切片
            Tensor<TD,DLayout>                & dst)   // <- tBs(...) 切片
{
  auto src_coord = src(Int<0>{});                        // 只取坐标
  void* dst_ptr  = cute::raw_pointer_cast(dst.data());  // 只取 smem 地址
  return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                               traits.opargs_,          // (desc_ptr, mbar_ptr, cache_hint)
                               ...,
                               make_tuple(dst_ptr),     // smem_ptr
                               ...,
                               src_coord, ...);         // crd0..crd4
}
```

`CallCOPY<SM90_TMA_LOAD>` 最终调用 `SM90_TMA_LOAD::copy(desc_ptr, mbar_ptr, cache_hint, smem_ptr, crd0, ...)`（`3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:327-363`），它再 dispatch 到 `SM90_TMA_LOAD_Nd::copy`，例如 3D 版本（`3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:159` 起的 `SM90_TMA_LOAD_3D`）里汇编是：

```
cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint
  [smem_ptr], [desc_ptr, {crd0, crd1, crd2}], [mbar_ptr], cache_hint;
```

可以清楚看到：`smem_ptr`（第 3 参数）与 `desc_ptr`/`mbar_ptr`/`cache_hint`（第 1 参数）和 `crd0..crd2`（第 2 参数）各就各位，源 gmem 张量的 `data()` 指针没有出现。

### 3. base address / shape / stride 在 descriptor 构造时就烤死了

`make_tma_copy_desc`（`3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:946`）里：

```cuda
void* gmem_address = (void*) raw_pointer_cast(gtensor_T.data());   // :946
cute::array<uint64_t, 5> gmem_prob_shape  = {1,1,1,1,1};          // :949
cute::array<uint64_t, 5> gmem_prob_stride = {0,0,0,0,0};          // :950
fill_tma_gmem_shape_stride(gtensor_T, stride(tma_gbasis), gmem_prob_shape, gmem_prob_stride); // :952
...
CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
    &tma_desc, tma_format, tma_dim,
    gmem_address,                         // :1050  ← 基地址进 descriptor
    gmem_prob_shape.data(),               //         ← global shape
    gmem_prob_stride.data() + 1,          //         ← byte stride
    smem_box_shape.data(),                //         ← smem box
    smem_box_stride.data(),
    tma_interleave, smem_swizzle, tma_l2Promotion, tma_oobFill);  // :1058
```

即 `tma_b` 对应的 descriptor 在 host 端（`make_tma_copy` 时）就已经把 W 的 base address、`(n,k,num_group)` 的 shape 与 stride、smem box 全部编码进去了。之后 `tma_b.with(readable[ismem_write])` 只是把这个 descriptor 指针 + barrier 指针 + cache hint 塞进可执行的 `Copy_Traits<SM90_TMA_LOAD_OP,...>`（`3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:127-133`）。

## 一句话总结

- **第 1 参数（Copy_Atom）**：TMA 需要知道"从哪块 gmem 搬、搬到 smem 的 box 长什么样、用哪个 mbarrier 同步、L2 怎么 cache"——这些全部来自 descriptor + mbarrier + cache_hint，其中 **base address / global shape / stride / smem box / swizzle 都在 descriptor 里**。
- **第 2 参数（gmem tensor `tBg`）**：TMA 只需要知道"这一次要搬的是这个 tensor 里的第几个 tile"——即 **coordinate**，不含 base address；它的 `data()` 指针在 TMA 路径下被忽略。
- **第 3 参数（smem tensor `tBs`）**：TMA 需要知道"搬到 smem 的哪个地址"——即 **smem_ptr**。

这也是为什么 `tma_a` 需要 `.with(td_x, ...)` 换 descriptor（换 base address + shape），而 `tma_b` 只需 `.with(barrier)`：B 的所有 expert 共用一个 descriptor，第 `igroup` 维只是坐标 `crd` 的一部分，由 `tBg(_, itile_n, itile_k, igroup)` 在坐标里表达。


---

# `cute::copy` 一次 TMA copy 需要几条 `cp.async.bulk`？TiledCopy 内部有几个 CopyAtom？

## 结论（以 run.log 实际实例化的 tile 配置为准：kTileM=48, kTileN=128, kTileK=128）

| | `tma_a`（`kernels.cuh:437`） | `tma_b`（`kernels.cuh:440`） |
|---|---|---|
| Tiler_MN（一个 tile 的形状） | `(_48,_128)` | `(_128,_128)` |
| 一次 copy 搬运的元素数 | **48×128 = 6144**（fp8，6144 bytes） | **128×128 = 16384**（fp8，16384 bytes） |
| TiledCopy 内部 CopyAtom 个数 | **1 个** | **1 个** |
| 需要的 `cp.async.bulk` 指令条数 | **1 条**（`.tensor.2d`，box 48×128） | **1 条**（`.tensor.3d`，box 128×128×1） |

**一句话：两个 `cute::copy` 都只发 1 条 `cp.async.bulk.tensor` 指令，TiledCopy 内部都只含 1 个 CopyAtom。** 因为 TMA 的"一个 CopyAtom 的 value"就是整块 box，一条指令搬完整块，CuTe 不会对 TMA 做逐元素展开。

## 为什么是 1 个 CopyAtom、1 条指令

TiledCopy 里 CopyAtom 的个数 = `(TiledNumThr / AtomNumThr) × (TiledNumVal / AtomNumVal)`，这三个量都能在 `run.log` 里直接读到：

- `tma_a`（`run.log:93-102`）：`TiledLayout_TV = (_1,(((_128,_48),_1)))` → TiledNumThr=1、TiledNumVal=128×48=6144；`Copy_Atom` 的 `ThrID = _1`（1 个线程）、`ValLayoutRef = (_1,_6144)` → AtomNumThr=1、AtomNumVal=6144。于是 `(1/1)×(6144/6144) = 1` 个 CopyAtom。
- `tma_b`（`run.log:104-113`）：`TiledLayout_TV = (_1,(((_128,_128),_1)))` → TiledNumVal=128×128=16384；`ValLayoutRef = (_1,_16384)` → AtomNumVal=16384。同样 `(1/1)×(16384/16384) = 1` 个 CopyAtom。

关键点：TMA Copy_Atom 的 `ThrID = _1`，即"**单线程 issue**"，且 `ValLayoutRef` 里的 6144/16384 个 value 不是"6144 次拷贝"，而是 CuTe 把整块 TMA box 的位布局抽象成"1 个线程 × N 个 value"；真正落到硬件是**一条** `cp.async.bulk.tensor` 指令搬完整块 box。

执行链（`3rd/cutlass/include/cute/atom/copy_traits_sm90_tma.hpp:71-89` 的 `TMA_LOAD_Unpack::copy_unpack`）：

```cuda
auto src_coord = src(Int<0>{});                        // 只取坐标
void* dst_ptr  = cute::raw_pointer_cast(dst.data());  // 只取 smem 地址
return detail::explode_tuple(detail::CallCOPY<CopyOp>{},
                             traits.opargs_, ...,      // (desc_ptr, mbar_ptr, cache_hint)
                             make_tuple(dst_ptr), ..., // smem_ptr
                             src_coord, ...);          // crd0..crd4
```

`copy_unpack` 被调用一次，`CallCOPY<SM90_TMA_LOAD>` 就 dispatch 一次到 `SM90_TMA_LOAD::copy`（`3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:327-363`），按坐标个数选择 `.1d/.2d/.3d/...`，各自只内联一条 `cp.async.bulk.tensor.Nd`（例如 2D 见 `3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:103` 起的 `SM90_TMA_LOAD_2D`，3D 见 `:159` 起的 `SM90_TMA_LOAD_3D`）。

## A / B 的指令维度为何不同

- **A（`tma_a`）是 2D**：源 gmem 张量 `gA = tma_a.get_tma_tensor(make_shape(m, k))`（`src/group_gemm/kernels.cuh:265`）只有 `(m=576, k=7168)` 两维，坐标 `tAg(_, itile_m, itile_k)` 提供 2 个坐标 → `cp.async.bulk.tensor.2d`，box `(48, 128)`。
- **B（`tma_b`）是 3D**：源 gmem 张量 `gB = tma_b.get_tma_tensor(make_shape(n, k, num_group))`（`src/group_gemm/kernels.cuh:266`）是 `(n=4096, k=7168, num_group=8)` 三维，坐标 `tBg(_, itile_n, itile_k, igroup)` 提供 3 个坐标 → `cp.async.bulk.tensor.3d`，box 是 `(128, 128, 1)`（第 3 维 num_group 是"纯坐标维"，box 大小 1，只有真实 stride）。

这与 `run.log` 打印的坐标张量一致：`gA = (576,7168):(_1@1,_1@0)`（2 维），`gB = (4096,7168,8):(_1@1,_1@0,_1@2)`（3 维）；`tBg = (((_128,_128),_1),32,56,8)` 里 32/56/8 正是 N-tile/K-tile/group 三个坐标维。

## 补充：一个 CopyAtom 不等于"一次 element copy"

不要把 CuTe 里 `ValLayoutSrc: (_1,_6144)` 的 6144 理解成"6144 次拷贝操作"。对 TMA 而言，整块 box（6144 或 16384 个元素）由**一条** `cp.async.bulk.tensor` 指令完成；"6144 个 value"只是 CuTe 对"这个 box 的位布局"的抽象表示。所以：

- `cute::copy(tma_a.with(...), tAg(_, itile_m, itile_k), tAs(...))` → **1 条 `cp.async.bulk.tensor.2d`，搬 6144 个 fp8（48×128）**。
- `cute::copy(tma_b.with(...), tBg(_, itile_n, itile_k, igroup), tBs(...))` → **1 条 `cp.async.bulk.tensor.3d`，搬 16384 个 fp8（128×128）**。

（注：`tAg(_, itile_m, itile_k)` 里那个 `_` 已经就是"整块 box"——`tAg` 的第 0 维是 TMA 分区 `(TMA_M,TMA_K)`，`run.log:162-163` 显示 `tAg = (((_128,_48),_1),12,56)`，切片后剩下的正是 box 本身，所以一次 `cute::copy` 就搬完一整块，无需循环。）


---

# math warpgroup 逐行讲解 + 三个问题（accumulate_、TMA store 无 barrier、wait/arrive 顺序）

本次运行 shape：`m=576, n=4096, k=7168, num_group=8`，tile 配置 `kTileM=48, kTileN=128, kTileK=128, kStage=8, kWarpgroupM=2, kWarpgroupN=1`（`temp/run.log:6`）。

## 0. 线程分工（进入 else 分支的前提）

`kNumThreads = size(TiledMma{})`（`src/group_gemm/kernels.cuh:340`）。`run.log:30` 打印 `ThrLayoutVMNK: (_128,_2,_1,_1)`，故 `kNumThreads = 128×2 = 256`。整个 kernel `__launch_bounds__(384, 1)`（`src/group_gemm/kernels.cuh:222`）：

- `idx < 256`（= math warpgroup）：256 个线程 = **2 个 warpgroup**（`iwarpgroup = idx/128` = 0 或 1），每个 warpgroup 128 线程算 M 方向的一半（`kWarpgroupM=2`）。
- `idx >= 256`（= load warpgroup）：128 个线程 = 1 个 warpgroup，只负责 TMA 搬 A/B。

## 1. math warpgroup 逐行讲解（`src/group_gemm/kernels.cuh:421-629`，跳过 printf/print）

```cuda
} else {
  // math warpgroup
  cutlass::arch::warpgroup_reg_alloc<168>();   // :423
```
请求 168 个寄存器/线程（`warpgroup_reg_alloc` 触发 `setmaxnreg` 动态寄存器重配置），math 侧需要大量寄存器存累加器/描述符，把多出的寄存器从 load 侧（`:343` 的 `warpgroup_reg_dealloc<24>`）拿过来。

```cuda
  int iwarpgroup = idx / 128;                  // :425
  TiledMma tiled_mma;                          // :427
  auto thr_mma = tiled_mma.get_slice(idx);     // :429
  auto tBs4r = thr_mma.partition_A(sB);        // :430
  auto tAs4r = thr_mma.partition_B(sA);        // :431
```
`TiledMma` 是 `make_tiled_mma(mma_selector<48>(), WarpgroupLayout{})`（`src/group_gemm/config.h:100`），`mma_selector<48>` 选的是 `SM90_64x48x32_F32E4M3E4M3_SS_TN`（`config.h:51`）。`get_slice(idx)` 取本线程在 MMA 里的切片；`partition_A/B` 把 smem 里的 B/A 张量按 GMMA 需要的 smem 描述符布局分片。注意 A/B 角色互换：B 进 `partition_A`、A 进 `partition_B`，因为这是 `_TN`（A 转置 N 方向）。

```cuda
  auto tBr = thr_mma.make_fragment_A(tBs4r);   // :433  (MMA, MMA_N, MMA_K, kStage)
  auto tAr = thr_mma.make_fragment_B(tAs4r);   // :434  (MMA, MMA_M, MMA_K, kStage)
```
把 smem 分片变成 GMMA 描述符迭代器（`run.log:200-203`：`GMMA::DescriptorIterator o (_1,_1,_4,(_1,_8))`，第 3 维 `_4` = 每个 itile_k 内 4 个 K-atom，第 4 维 `(_1,_8)` = 8 级流水 stage）。

```cuda
  auto tCr = thr_mma.partition_fragment_C(gC); // :436
```
`tCr` 是本线程的 f32 累加器寄存器片段（`run.log:205`：`ptr[32b] o ((_2,_2,_6))` = 每线程 24 个 f32；128 线程 × 24 = 3072 = 64×48，即一个 warpgroup 的 M64×N48 累加器）。

```cuda
  int ismem_read = 0;                          // :438
  int phase = 0;                               // :439
  int iblock = blockIdx.x;                     // :441
  int igroup = 0, sum_tile_m = 0, itile_m, itile_n, task, iwave = 0;  // :442-446
  while (true) {                               // :447
```
`ismem_read/phase` 是读侧流水指针；`while(true)` 按 `kTaskLoopPolicy` 三种策略之一取下一个任务 `(igroup, itile_m, itile_n)`（`:448-468`），与 load warpgroup 用同一套 `get_next_tile_*` 逻辑，保证两边顺序一致。policy==2 走 `get_next_tile_vert`（`:464`），`itile_n >= num_tile_n` 时 break（`:465-467`）。

```cuda
  iblock += gridDim.x;                         // :470
  auto tDr = make_tensor_like(tCr);            // :472
  clear(tDr);                                  // :473
```
`tDr` 是"按 group scale 缩放后的最终累加器"，**每个 tile 只清一次**；`tCr` 是 raw MMA 累加器（每个 itile_k 用 ScaleOut::Zero 重置，见下）。

```cuda
  float scale = yscale_ptr[igroup];            // :475
  int ntile_k = size<2>(tAg);                  // :477
  for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {   // :479
    wait_barrier(readable[ismem_read], phase);  // :480
```
`scale` 是 per-group 输出缩放（per-tensor fp8 反量化系数）。`ntile_k = 56`（k=7168/kTileK=128）。`wait_barrier` 等 load warpgroup 把这一 stage 的 A/B 通过 TMA 搬进 smem（mbarrier 相位翻转）。

```cuda
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;   // :482
    warpgroup_fence_operand(tCr);                   // :484
    warpgroup_arrive();                             // :485
    for (int ik = 0; ik < size<2>(tAr); ++ik) {     // :487  （size<2>(tAr)=4）
      cute::gemm(tiled_mma, tBr(_,_,ik,ismem_read), tAr(_,_,ik,ismem_read), tCr);
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;  // :489
    }
    warpgroup_commit_batch();                       // :492
    warpgroup_wait<0>();                            // :493
    warpgroup_fence_operand(tCr);                   // :494
    arrive_barrier(writable[ismem_read]);           // :496
```
- 先 `accumulate_ = Zero`，再在循环体里 `= One`：第一个 K-atom（ik=0）**覆盖写**累加器，后续 3 个 atom **累加**（详见第 2 节）。
- `warpgroup_fence_operand(tCr)`（`:484`）＝对累加器寄存器做 fence，保证 wgmma 异步写 `tCr` 之前对它的旧读写已可见。
- `warpgroup_arrive()`（`:485`）＝`wgmma.fence.sync.aligned`，标记 wgmma 组的开始。
- 内层 `for ik`（`:487-490`）发 4 次 `cute::gemm`，每次一个 64×48×32 的 GMMA atom，4×32=128 覆盖整个 kTileK。`tBr/tAr` 第 3 维 `ik` 选 K-atom，第 4 维 `ismem_read` 选 stage。
- `warpgroup_commit_batch()`（`:492`）＝`wgmma.commit_group` 标记这 4 条 wgmma 为一批；`warpgroup_wait<0>()`（`:493`）＝`wgmma.wait_group 0` 等这批全部算完、`tCr` 可读；`warpgroup_fence_operand(tCr)`（`:494`）再 fence 使结果对后续读可见。
- `arrive_barrier(writable[ismem_read])`（`:496`）告诉 load warpgroup"这块 smem 我已读完，可复用"。

```cuda
    for (int i = 0; i < size(tCr); ++i)          // :499
      tDr(i) = tCr(i) * scale + tDr(i);          // :500
    ++ismem_read;                                // :503
    if (ismem_read == kStage) { phase ^= 1; ismem_read = 0; }  // :504-507
```
把本 itile_k 的 raw 累加 `tCr` 乘 scale 后累进 `tDr`（跨 56 个 itile_k 求和），并推进读流水。

```cuda
  auto tCrh = make_tensor_like<cute::bfloat16_t>(tCr);   // :511
  for (int i = 0; i < size(tCr); ++i)
    tCrh(i) = (Tout)(tDr(i));                            // :514-515
```
把 f32 累加器转成 bf16（`Tout`）。

```cuda
  auto sCT = make_tensor(make_smem_ptr(reinterpret_cast<Tout*>(shm_c)), SLayoutCT{});  // :519-520
  using STSM_ATOM = std::conditional_t<kTileM==8, SM90_U16x4_STSM_T, SM90_U16x8_STSM_T>; // :521-522
  using R2SCopyAtomC = Copy_Atom<STSM_ATOM, Tout>;        // :523
  auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);  // :524
  auto thr_copy_c = tiled_copy_c.get_slice(idx);          // :525
  auto tCr4s = thr_copy_c.retile_S(tCrh);                 // :527
  auto tCs4r = thr_copy_c.partition_D(sCT);               // :528
```
epilogue：`tiled_copy_c` 是 `SM90_U16x8_STSM_T`（smem store，把 bf16 寄存器写进 smem C）构成的 TiledCopy。`retile_S` 把累加器片段按 STSM 的源布局重新排列（`tCr4s`），`partition_D` 给出 smem C 目标分片（`tCs4r`）。

```cuda
  tma_store_wait<0>();                          // :530
  syncwarpgroup(iwarpgroup);                    // :531
  cute::copy(tiled_copy_c, tCr4s, tCs4r);       // :533
  syncwarpgroup(iwarpgroup);                    // :534
  cute::tma_store_fence();                      // :535
```
见第 4 节：`tma_store_wait<0>()` 先等上一轮的 TMA store 读 smem 完毕（保护 smem C 不被覆盖）；两个 `syncwarpgroup` 保证 warpgroup 内 128 线程的 STSM 完成；`tma_store_fence()`＝`fence.proxy.async.shared::cta`，让 smem 写对后续 TMA store 可见。

```cuda
  if (is_leader_in_warpgroup) {                 // :537
    auto gD = tma_d.get_tma_tensor(make_shape(n, m));   // :538
    auto btma_d = tma_d.get_slice(0);                   // :539
    auto tDs = btma_d.partition_S(sCT);                 // :541  (TMA, _2, _1)
    auto tDg = btma_d.partition_D(gD);                  // :542  (TMA, TMA_M, TMA_N)
    auto *td_y = td_xy + igroup * 2 + 1;                // :544
    cute::copy(tma_d.with(td_y), tDs(_, iwarpgroup, Int<0>{}),
               tDg(_, itile_n * 2 + iwarpgroup, itile_m));  // :545-546
    tma_store_arrive();                                 // :547
  }
```
只有每个 warpgroup 的 leader 线程发 TMA store。`tDs` 的 `_2` 维是 warpgroup 的 M 方向分半（`run.log:257` 的 `((_32,_8),(_2,_6)),_2,_1`），`tDs(_, iwarpgroup, Int<0>{})` 选本 warpgroup 的那一半；`tDg` 的 TMA_N 网格是 `64`（n=4096/64），因为每个 warpgroup 只存 64 列 = tile 的一半，所以全局 N-tile 号是 `itile_n*2 + iwarpgroup`（`run.log:259-260`）。`td_y = td_xy + igroup*2 + 1` 取本 group 的 store descriptor（`update_grouped_tma` 预生成，`+1` 是 Y/store 描述符，`+0` 是 X/load 描述符）。

## 2. 为什么先 `accumulate_ = ScaleOut::Zero` 再 `= ScaleOut::One`

GMMA 指令的语义是 `D = A*B + scale_D * C`（`3rd/cutlass/include/cute/arch/mma_sm90_gmma.hpp:129` 注释 `C = (scaleA*A)*(scaleB*B) + (scaleD*C)`）。`accumulate_` 就是这个 `scale_D`，枚举值 `ScaleOut { Zero=0, One=1 }`（`mma_sm90_gmma.hpp:112-115`）。

它**只影响输出侧 D 的累加行为，不影响 A/B 输入**：

- `ScaleOut::Zero`（scale_D=0）→ `D = A*B`，旧的 C 被忽略/覆盖。
- `ScaleOut::One`（scale_D=1）→ `D = A*B + C`，累加到旧 C 上。

在 `cute::gemm` 里，`MMA_Traits` 的成员 `accumulate_`（默认 `One`，`3rd/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:496`）通过 `&(traits.accumulate_)` 传给 `MMA_Op::fma`（`mma_traits_sm90_gmma.hpp:429`），最终变成 wgmma 指令里那条"是否累加"的谓词 `p`（例如 F16 版本的 `setp.ne.b32 p, %7, 0`，`mma_sm90_gmma.hpp:204-209`；fp8 版本同构）。

**为什么这样写**：一个 `itile_k`（K=128）被拆成 4 个 K-atom（每个 K=32，`size<2>(tAr)=4`，`run.log:202`）。这 4 个 atom 共享同一个累加器 `tCr`：

- 第 1 个 atom（ik=0）必须**覆盖**累加器，否则会把上一轮 `itile_k` 留下的旧值再加一遍 → 所以进内层循环前设 `Zero`。
- 第 2~4 个 atom（ik=1,2,3）必须在同一 K 上**累加** → 循环体里设 `One`。

这就是标准的"K 链上第一个 MMA 初始化累加器、其余 MMA 累加"惯用法，用硬件 scale_D 谓词实现，省掉一次显式清 `tCr` 寄存器。注意 `:489` 每次循环都赋 `One`，对 ik=1,2,3 冗余但无害。

## 3. 为什么 `tma_d.with(td_y)` 不需要 barrier，而 `tma_a/tma_b` 需要

因为 **TMA load 和 TMA store 的同步机制根本不同**：

- **TMA load**（`SM90_TMA_LOAD`）的指令是 `cp.async.bulk.tensor.Nd.shared::cluster.global.mbarrier::complete_tx::bytes`（`3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:69-74`），**必须带 mbarrier**。因为搬完数据后要靠 mbarrier 的 tx-count/arrive-count 相位翻转来通知**另一个 warpgroup**（consumer math）"数据到了"。所以 `tma_a/tma_b` 的 `.with` 里要传 `uint64_t& tma_mbar`（`readable[ismem_write]`），对应 `Copy_Traits<SM90_TMA_LOAD>::with` 的 barrier 形参（`copy_traits_sm90_tma.hpp:127/138`）。

- **TMA store**（`SM90_TMA_STORE`）的指令是 `cp.async.bulk.tensor.Nd.global.shared::cta.bulk_group`（`3rd/cutlass/include/cute/arch/copy_sm90_tma.hpp:969/992/1015`），**没有 mbarrier 操作数**，而是用 **bulk async group**（`bulk_group`）同步：`tma_store_arrive()`＝`cp.async.bulk.commit_group`，`tma_store_wait<N>()`＝`cp.async.bulk.wait_group.read N`（`copy_sm90_tma.hpp:1225-1258`）。

所以 store 版的 `.with` 只有 1 个描述符指针参数，没有 barrier 参数：`Copy_Traits<SM90_TMA_STORE>::with(TmaDescriptor const* new_tma_desc)`（`copy_traits_sm90_tma.hpp:396`），返回 `SM90_TMA_STORE_PTR`，其 `opargs` 只含描述符指针（`copy_traits_sm90_tma.hpp:439`）。

**为什么这样设计**：load 是"生产者→消费者"跨 warpgroup 的数据就绪通知，需要 mbarrier 做细粒度 phase 同步；store 是"写回 gmem"这一侧的 fire-and-forget，唯一需要保证的是"smem 源缓冲在 TMA 读完之前别被覆盖"，而这是**同一个 warpgroup 内部**的复用问题，用 bulk_group 的 commit/wait 计数就够了，不需要 mbarrier。

## 4. 为什么先 `tma_store_wait<0>()` 后 `tma_store_arrive()`，不能交换

两者是不同职责：

- `tma_store_wait<0>()` = `cp.async.bulk.wait_group.read 0`：**等之前已 commit 的所有 TMA store 全部读完 smem 源**（0 个 pending）。
- `tma_store_arrive()` = `cp.async.bulk.commit_group`：**把当前这条 TMA store 提交进一个新的 bulk_group**，标记边界，供下一轮的 wait 使用。

正确顺序（`src/group_gemm/kernels.cuh:530-547`）构成一条"等上一轮 → 写 smem → 发本轮 → 提交本轮"的流水：

```
tma_store_wait<0>();                 // 等上一轮 store 读完 smem C
syncwarpgroup(); cute::copy(STSM);   // 把本轮结果写进 smem C（此时安全）
cute::tma_store_fence();             // smem 写对 TMA 可见
cute::copy(tma_d.with(td_y), ...);   // 发本轮的 TMA store
tma_store_arrive();                  // commit 本轮 store 为一个新 group
```

**为什么不能交换**：

1. `wait<0>` 的目的是**保护 smem C 缓冲不被覆盖**——它必须发生在"写 smem C"（`cute::copy(tiled_copy_c,...)` STSM）**之前**。如果上一条 store 还在异步读这块 smem，就先写新结果，会读到被覆盖的数据。交换后 wait 就跑到写 smem 之后了，保护失效。

2. `arrive`（commit_group）必须发生在**发完 store 之后**，才能把这条 store 纳入 group。若先 arrive，此刻还没有 store 可 commit（空提交/无意义），真正发 store 在 arrive 之后反而没被 commit，下一轮的 `wait<0>` 就管不住它。

3. 更深一层：`wait<0>` 语义是"等 pending group 数归零"，若把它放到 store 发出之后（交换位置等效于"发完 store 立刻 wait<0>"），会把**刚发的这条 store 也一起等掉**，直接串行化、毁掉 store 与下一轮 MMA 的异步重叠。所以必须"wait 在前（清旧账）、arrive 在后（记新账）"。

一句话：**wait 是"收账"（清上一轮），arrive 是"记账"（记本轮），先收旧账再记新账**，顺序不能反。

---

## scale 乘法放在 itile_k 循环内 vs 提出的"挪到循环外只乘一次"是否等价

**问题**：`src/group_gemm/kernels.cuh` group_gemm_fp8_kernel 的 math warpgroup 分支里，499~501 行的 `tDr(i) = tCr(i) * scale + tDr(i);` 目前放在 `for (int itile_k = 0; itile_k < ntile_k; ++itile_k)` 循环内部，对整个 tile 总共执行 `size(tCr) * ntile_k` 次。若把 scale 乘法挪到循环结束后，改成 `for (int i = 0; i < size(tDr); ++i) { tDr(i) = tDr(i) * scale; }`（只循环 `size(tDr)` 次），结果还正确吗？

**结论：正确，可以安全地把 scale 乘法挪出循环。** 数学上完全等价，且略有精度/性能好处。唯一差别是浮点舍入顺序，量级在几个 ULP 以内。

### 1. 为什么数学上等价

关键是 `scale` 是循环不变量：

- `src/group_gemm/kernels.cuh:475` `float scale = yscale_ptr[igroup];` — scale 只依赖 `igroup`。
- `igroup` 在每个 tile task 开始时由任务调度（`get_next_tile_vert` / task map，见 458~467 行区域）一次性确定，在同一个 tile 的整个 K 循环内固定不变，不随 `itile_k` 变化。因此 `scale` 在 `for (int itile_k ...)` 循环里对每次迭代都是同一个值。

原始写法按 k 展开（tDr 初值为 0，见 :473 `clear(tDr)`）：

```
k=0 : tDr = tCr_0 * scale + 0       = tCr_0 * scale
k=1 : tDr = tCr_1 * scale + tCr_0 * scale
k=2 : tDr = tCr_2 * scale + tCr_1 * scale + tCr_0 * scale
...
k=ntile_k-1 : tDr = scale * Σ_k tCr_k
```

即 `tDr = scale * Σ_{k} tCr_k`。

提出写法：先裸累加（不乘 scale），循环结束后再乘一次：

```
循环后 : tDr = Σ_k tCr_k
再乘   : tDr = (Σ_k tCr_k) * scale
```

两者都是 `scale * Σ tCr_k`，数学恒等。整数/重数下结果一致。

### 2. 关键正确性前提都成立

- **`scale` 真正循环不变**（:475 只在 tile 开始时读一次，:479 的循环内不重读）。
- **`tCr` 每个 K-tile 被重置为全新部分和**：`src/group_gemm/kernels.cuh:482` `tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;`，所以 `tCr` 只存放"当前 K-tile"这 32 个 K 值的部分积，而非跨 K 的累积；跨 K 的累积正是靠 `tDr`。这点很关键——挪出后 `Σ tCr_k` 由 `tDr` 完整保存，循环内不再有 scale 参与累积。
- **`tDr` 只在循环结束后被消费**：`src/group_gemm/kernels.cuh:511-516` `tCrh(i) = (Tout)(tDr(i));` 在循环外面。所以把 scale 延后到循环后乘，不会影响任何读 tDr 的时序。

### 3. 唯一的差别：浮点舍入（可忽略）

- 原始：每个 K 步做一次 `scale * tCr_k`（乘法）再加进 `tDr`，共 `ntile_k` 次乘法 + `ntile_k` 次加法（编译器通常会融合为 FMA，但每个 FMA 仍是一次舍入）。
- 提出：循环内只做 `ntile_k` 次加法，最后做 1 次乘法。

两者最终结果的 fp32 低几位会略有不同（舍入顺序不同），但误差都是"标准累积求和 + 1 次缩放"量级，属于几个 ULP 的差异，不会改变数值正确性。实际上提出版本只做 1 次乘法而非 `ntile_k` 次，乘法舍入更少，通常**略微更精确或至少不更差**。

### 4. 性能

挪出后每个元素少做 `ntile_k - 1` 次乘法（fp32 寄存器运算），且能让 `#pragma unroll`（:486/:498）的 K 内层循环更轻。不过该循环大概率受 TMA / wgmma 等待（`warpgroup_commit_batch` + `warpgroup_wait<0>`，:492-493）约束而非受这几条 fp32 运算约束，所以实际收益可能很小，但无任何副作用。

### 建议的等价改写

```cuda
// K 循环内改为纯累加
for (int i = 0; i < size(tCr); ++i) {
  tDr(i) += tCr(i);
}
// 循环结束后一次缩放
for (int i = 0; i < size(tDr); ++i) {
  tDr(i) = tDr(i) * scale;
}
```


---

## 进一步合并：直接让 gemm 累加到 tDr，去掉 tCr，并把 accumulate_ 提到循环外

**问题**：在上一问（把 scale 乘法挪出 `for (int itile_k ...)`）的基础上，进一步：让 `cute::gemm` 直接输出到 `tDr` 而非 `tCr`：
- 循环前 `clear(tDr)`（让 tDr 为 0），并只设置一次 `tiled_mma.accumulate_ = GMMA::ScaleOut::One;`
- 内层 `for (int ik ...)` 不再反复设 `accumulate_`
- 循环结束后 `for (int i = 0; i < size(tDr); ++i) tDr(i) = tDr(i) * scale;`

结果还正确吗？

**结论：正确。这是标准 CUTLASS Hopper mainloop（fp32 直接累加、最后统一缩放）的写法，且能省掉 `tCr` 这一整份片段寄存器。** 但有一个关键点必须跟着改：**`warpgroup_fence_operand` 必须从 `tCr` 改为 `tDr`**，否则编译器屏障落在已经不再被 wgmma 改写的 `tCr` 上，`tDr` 这个真正的累加器没有被强制物化到寄存器，是潜在的正确性隐患。

### 1. 为什么数学上正确

`accumulate_` 控制 GMMA 的 `scale_D`（`src/group_gemm/kernels.cuh:482` 附近），即 `D = (scaleA*A)*(scaleB*B) + (scaleD*D)`。原始写法每个 `itile_k` 先把 `tCr` 用 `ScaleOut::Zero` 重置为"本 K-tile 的部分和"，再用 `ScaleOut::One` 在 `ik` 内累加，最后 `tDr = tCr*scale + tDr`。逐项展开即：

```cuda
tDr = scale * ( Σ_{itile_k} Σ_{ik} A*B )
// 括号里就是整个 K（7168）维的完整点积，因为循环一次把 tAg 的 K 维（size<2>=56，56*128=7168）全部走完
```

新写法 `clear(tDr)` 后全程 `ScaleOut::One`：

- 第 1 次 gemm：`tDr = A_0*B_0 + 0 = A_0*B_0`（等价于 `Zero` 的起始效果）
- 后续全部：`tDr += A*B`
- 整个 K 循环后：`tDr = Σ_{全部 K} A*B`
- 最后乘一次：`tDr = (Σ A*B) * scale`

两者都等于 `scale * Σ_K (A*B)`，数学恒等。

### 2. 为什么可以全程用 One / 只 clear 一次

- `src/group_gemm/kernels.cuh:473` `clear(tDr);` 已经存在，且它位于每个 tile task 的 while 循环体内（`auto tDr = make_tensor_like(tCr);` 在 :472）。所以每个输出 tile 开始时 `tDr` 保证为 0，第一次 gemm 的 `A*B + 0` 就是干净的起始值，等价于只在"第一次"用一次 `Zero`。全程 `One` 是安全的。
- `scale` 是循环不变量（`src/group_gemm/kernels.cuh:475` `float scale = yscale_ptr[igroup];`，只依赖 `igroup`，在单个 tile 的整个 K 循环内恒定）。一个 tile 的 `igroup` 固定，整个 K 维都属于同该 group，所以只乘一次即可。
- `tDr` 目前只在 K 循环**结束后**被消费（`src/group_gemm/kernels.cuh:513-516` `tCrh(i) = (Tout)(tDr(i));`），循环内除了 `tCr` 没有任何地方读 `tDr` 的中间值，因此把缩放和消费都推迟到循环后不影响时序。
- 每次 `itile_k` 结尾的 `warpgroup_commit_batch(); warpgroup_wait<0>();`（`src/group_gemm/kernels.cuh:492-493`）已保证每次批次的 wgmma 在进入下一次 `itile_k` 前完成，因此循环退出时最后一次 wgmma 也已 wait 完，`tDr` 是稳定可读的。

### 3. 必须改的点（否则有隐患）

1. **`warpgroup_fence_operand(tCr)` 改成 `warpgroup_fence_operand(tDr)`**：`src/group_gemm/kernels.cuh:484` 和 `:494`。`warpgroup_fence_operand` 的实现（`3rd/cutlass/include/cute/arch/mma_sm90_gmma.hpp:88-103`）是 `asm volatile("" : "+f"(reg) :: "memory")`——一个以累加器寄存器作为读写约束的空汇编，作用是**强制编译器把该寄存器张量在 wgmma 前物化到寄存器、不跨这条屏障乱序**。它必须作用于实际被 wgmma 读写的那个累加器张量。现在累加器是 `tDr`，若屏障仍挂在 `tCr` 上，`tDr` 不被强制物化，是正确性风险。

2. **只设一次 `tiled_mma.accumulate_ = GMMA::ScaleOut::One;`** 放在 `for (int itile_k ...)` 之前（`clear(tDr)` 之后）。其实 `GMMA::ScaleOut accumulate_` 的默认值就是 `One`（`3rd/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:496`），所以不设也默认是 One；显式设一次更清晰。

3. **`tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;`（`src/group_gemm/kernels.cuh:482`）删掉**，因为不再需要每个 `itile_k` 重置部分和。

4. **内层 `ik` 循环里的 `tiled_mma.accumulate_ = GMMA::ScaleOut::One;`（`src/group_gemm/kernels.cuh:489`）删掉**，gemm 直接写 `tDr(_, _, _)`。

5. 循环后的乘 scale 循环用 `size(tDr)`（而非 `size(tCr)`）作边界；顺带修正 `:513-516` epilogue 里 `for (int i = 0; i < size(tCr); ++i)` 的边界为 `size(tDr)`（二者当前相等，只是语义上更严谨）。

### 4. 精度：不更差，通常略更好

原始写法每个 `itile_k` 都要做一次 `scale * tCr`（有舍入）再 FMA 进 `tDr`；新写法把 7168 个 fp32 累加全部在 `tDr` 里做，最后只做**一次**乘法。两者总累积精度相当（都是 ~7168 元素 fp32 求和），新写法乘法舍入由 `ntile_k` 次降为 1 次，通常略更精确。不做 per-K-tile 的 scale 缩放，也避免了 scale 带来的逐段舍入。

### 5. 性能 / 寄存器收益

`tCr` 和 `tDr` 是两份独立的寄存器片段（`make_tensor_like(tCr)` 复制布局但分配新存储，见 run.log 中 `tCr`/`tDr`/`tCrh` 三个不同的 `ptr[...]`）。合并后 `tCr` 彻底不再需要，主循环内少一份累加器片段的寄存器压力（配合 `warpgroup_reg_alloc<168>()` 的动态寄存器分配更稳），内层循环也不再每条 gemm 后插入一次分支/赋值。这是实打实的收益。

### 6. 一句话总结

只要**把 `warpgroup_fence_operand` 一起切到 `tDr`**、删掉 per-`itile_k` 的 `Zero`/`One` 切换、保留 `clear(tDr)`，这套"直接累加 + 循环后一次缩放"的写法在数学和时序上都正确，且省寄存器、少舍入。合并后的结构大致为：

```cuda
auto tDr = make_tensor_like(tCr);
clear(tDr);
float scale = yscale_ptr[igroup];
tiled_mma.accumulate_ = GMMA::ScaleOut::One;
for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
  wait_barrier(readable[ismem_read], phase);
  warpgroup_fence_operand(tDr);
  warpgroup_arrive();
  for (int ik = 0; ik < size<2>(tAr); ++ik) {
    cute::gemm(tiled_mma, tBr(_, _, ik, ismem_read), tAr(_, _, ik, ismem_read), tDr(_, _, _));
  }
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  arrive_barrier(writable[ismem_read]);
  ++ismem_read;
  if (ismem_read == kStage) { phase ^= 1; ismem_read = 0; }
}
for (int i = 0; i < size(tDr); ++i) {
  tDr(i) = tDr(i) * scale;
}
```

