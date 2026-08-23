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
