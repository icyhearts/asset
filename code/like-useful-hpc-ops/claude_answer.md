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

