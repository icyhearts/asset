# 1. SGLang v0.5.18-local-dep 启动耗时分析

分析对象：

- 旧版本：main-local-dep（日志中还包含一次 main-2026_07_08___17_53_37）
- 新版本：release/v0.5.18-local-dep
- 启动参数：like-useful/dsv4-flash-run.sh，TP=4，Marlin，EAGLE，speculative-num-steps=3，speculative-num-draft-tokens=4，cuda-graph-max-bs=16
- 环境：/share_data/users/like/miniconda3/envs/simo_sglang/；源码以 editable 方式安装在 /share/users/like/package/sglang_kernel_src

## 1.1 结论

启动时间增加的主因已经定位到：

1. 新版本的 full CUDA Graph 目标验证（target_verify）阶段从旧版本约 6.1～7.7 分钟增加到约 18.4～19.3 分钟。
2. 这段时间内会触发 moe_wna16_marlin 的 C++/CUDA JIT。新版本的 content-addressed JIT cache 在当前 /softhome -> /share_data symlink 布局下把构建目录中的临时 cuda.cu 记录成了绝对依赖；构建结束后临时目录被删除，下一次查找缓存时必然判定依赖失效。
3. 4 个 TP rank 因锁机制依次重新编译同一个 Marlin JIT 模块，每次约 3 分 33 秒～3 分 44 秒，单次启动出现 4 次，合计约 14.5～15 分钟。这与 target_verify 首个 bs=16 桶耗时约 17.8～18.3 分钟高度吻合，是本次升级后每次启动变慢的主要根因。

因此，不能把问题简单归结为“CUDA Graph 桶数量变多”：新旧版本都捕获了 12 个桶 [1,2,3,4,5,6,7,8,10,12,14,16]，且最慢的都是首个 bs=16。旧版本日志把该路径统称为 generic CUDA graph；在 EAGLE 模式下它实际上也是 target-verify 语义。新版本只是把它拆分并显式命名为 target_verify，同时 JIT/cache 路径发生了变化。

MHC/DeepGEMM 的懒编译、分布式初始化变慢以及环境混用会增加几十秒到约 1 分钟，或造成少量波动，但不能解释 10～13 分钟的主要差值。

## 1.2 启动总时间和阶段对比

以日志文件名时间作为开始时间，以 “The server is fired up and ready to roll!” 作为结束时间；阶段时间取各 rank 中的最长值。

| 日志 | 分支/环境 | 总时间 | 分布式初始化 | 主模型权重 | 主要 CUDA Graph | draft Graph |
|---|---|---:|---:|---:|---:|---:|
| 2026-06-25 17:33:23 | 旧 | 599 s（9:59） | 13.1～13.4 s | 约 42 s；draft 18.0～18.2 s | generic 463.5 s（7:44） | 4.37 s |
| 2026-07-02 10:07:39 | 旧 | 484 s（8:04） | 10.6～10.7 s | 36.0～36.3 s；draft 15.5～15.8 s | generic 367.4～367.5 s（6:07） | 3.62 s |
| 2026-09-01 17:32:27 | 新，simo_sglang | 1507 s（25:07） | 65.97～68.12 s | 约 122～141 s；MHC prewarm 53.2～53.6 s | target_verify 1156.27～1156.29 s（19:16） | decode 26.35 s，extend 8.16 s |
| 2026-09-01 18:04:02 | 新，simo_sglang_pip | 1716 s（28:36） | 63.68～63.70 s | 约 87～106 s；MHC prewarm 52.7～54.4 s | target_verify 1149.43～1149.44 s（19:09） | decode 25.39 s，extend 7.71 s |
| 2026-09-03 16:21:20 | 新，simo_sglang | 1374 s（22:54） | 62.72～62.77 s | 约 55～73 s；MHC prewarm 3.2～3.7 s | target_verify 1105.97～1106.02 s（18:26） | decode 24.94 s，extend 1.30 s |

关键日志位置：

- 旧版本 generic graph：2026-06-25 日志 146～165 行、2026-07-02 日志 146～164 行。
- 新版本 target_verify：2026-09-01 日志 2292～2300、约 2300 行以后以及 480 行；2026-09-03 日志 2243～2259、2270 行以后。
- 新版本启动汇总：2026-09-01 日志约 2324 行和 273 行；2026-09-03 日志约 2275 行。
- ready 行：旧日志均为 256 行；新日志分别为 2402、286、2290 行附近。

### 1.2.1 主要差值

旧版本 generic graph 为 367.4～463.5 s，新版本对应的 target_verify 为 1106.0～1156.3 s：

- 增加 642.5～788.9 s，即增加约 10.7～13.2 分钟；
- 新/旧比值约 2.4～3.1 倍；
- 仅这一项就解释了总启动时间从 8～10 分钟变成 23～29 分钟的大部分差异。

新版本进度条显示首个 bs=16：

- 2026-09-01：约 1093.98 s（18:14）；
- 2026-09-03：约 1069.06 s（17:49）。

首个 bs=16 完成后，bs=14、12、10 等剩余桶大多在秒级或几十秒内完成。因此主耗时是首个桶中的 kernel/JIT 初始化，而不是 12 个桶平均变慢。

## 1.3 主要根因：新 JIT cache 没有命中，Marlin 被 4 个 rank 串行重编译

### 1.3.1 新旧 loader/cache 行为不同

旧分支的 python/sglang/jit_kernel/utils.py:276-291：

- 使用 TVM_FFI_CACHE_DIR；
- 对模块生成稳定的直接 .so 路径；
- 已存在的预编译产物可以直接加载。
- 当前旧缓存中仍能看到例如：
  /data/like/cache/tvm_ffi_cache_dir/sgl_kernel_jit_moe_wna16_marlin_bf16_t_36ecfb0dd421562b__arch_9.0__tvmffi_0.1.11/sgl_kernel_jit_moe_wna16_marlin_bf16_t_36ecfb0dd421562b.so

新分支的 python/sglang/kernels/jit/utils/compile/loader.py:108-169：

- 先生成 build key，再使用 build_key_dir；
- 默认 JIT 根目录是 ~/.cache/sglang/jit；也可以由 SGLANG_JIT_CACHE_DIR 指定；
- 先查 prebuilt，未命中后在 .staging-UUID 目录编译，再 rename 发布；
- loader.py:128-133 的注释明确希望 TP rank 形成“一次编译、其余 rank 命中缓存”。

当前 dsv4-flash-run.sh 只设置了：

- DG_JIT_CACHE_DIR=/data/like/cache/deep_gemm_cache_dir
- TVM_FFI_CACHE_DIR=/data/like/cache/tvm_ffi_cache_dir
- TRITON_CACHE_DIR=/data/like/cache/triton_cache_like

没有设置 SGLANG_JIT_CACHE_DIR。因此 release/v0.5.18-local-dep 不会使用旧版 TVM FFI 的直接 .so 作为新 loader 的缓存；实际使用的是 /softhome/like/.cache/sglang/jit（该路径最终指向 /share_data/users/like/.cache/sglang/jit）。

### 1.3.2 缓存 manifest 中记录了已删除的 staging 文件

新版本 python/sglang/kernels/jit/utils/compile/cache.py:463-486 在扫描依赖时会调用 candidate.resolve()，然后用 path.is_relative_to(build_dir) 判断是否应排除构建目录内部文件。

本机存在：

- /softhome/like/.cache -> /share_data/users/like/.cache

所以实际发生了以下路径不一致：

- build_dir 仍是 /softhome/like/.cache/sglang/jit/.../.staging-UUID；
- candidate.resolve() 后是 /share_data/users/like/.cache/sglang/jit/.../.staging-UUID/cuda.cu；
- 未对 build_dir 同时 resolve，is_relative_to(build_dir) 判断失败；
- 生成的 staging/cuda.cu 被写进 sgl_deps.json 的 abs 依赖；
- loader.py:170-171 随后删除 staging 目录；
- 下一次 find_prebuilt 时，abs 依赖已经不存在，于是缓存被判定为 changed，重新编译。

实测证据：

- 在 Marlin 叶子缓存上，419 个依赖条目中包含 abs staging 依赖；
- find_prebuilt 返回 None，并打印：
  Rebuilding JIT module ...: abs:/share_data/users/like/.cache/sglang/jit/.../.staging-.../cuda.cu changed
- 扫描 /softhome/like/.cache/sglang/jit/sm90a 后，316 个 manifest 中 316 个包含缺失的 abs staging 路径；
- 这不是“第一次启动缓存为空”，而是“每次发布后缓存 manifest 都自带一个下一次必失效的依赖”。

修复方向是让比较双方使用同一 canonical path，例如在 _to_entries 进入 is_relative_to 前先执行 build_dir = build_dir.resolve()。现有坏 manifest 需要重新生成；为避免误删共享缓存，优先使用新的、canonical 的 SGLANG_JIT_CACHE_DIR 做验证。

### 1.3.3 每次启动出现四次连续 Marlin 编译

当前缓存目录中，模块 sgl_kernel_jit_moe_wna16_marlin_bf16_t_false_false 的 .so 时间戳如下：

| 启动 | 4 个 .so 完成时间 | 单次编译耗时 |
|---|---|---:|
| 2026-09-01 17:32 | 17:43:03、17:46:48、17:50:24、17:54:02 | 约 3:33～3:45 |
| 2026-09-01 18:04 | 18:17:55、18:21:32、18:25:36、18:29:22 | 约 3:37～3:45 |
| 2026-09-03 16:21 | 16:30:21、16:33:59、16:37:43、16:41:21 | 约 3:37～3:39 |

四次编译在同一个模块、同一个 build key 下按时间串行出现，与 4 个 TP rank 争用 loader lock、但每个 rank 的二次 cache check 都因坏 manifest 失败的行为一致。四次合计约 14.5～15.0 分钟，正是 target_verify 首个 bs=16 长时间段的主要组成部分。

python/sglang/kernels/ops/moe/moe_wna16_marlin.py:18-33 会调用新 load_jit；在当前参数（bf16、is_ep=false、has_bias=false）下对应上述模块名。旧版本使用的 loader 和模块命名不同，所以旧版已有的 TVM FFI .so 不能直接证明新 loader 能命中。

## 1.4 次要增量和其它差异

### 1.4.1 MHC fused post/pre 与 TileLang 懒编译

当前 environ.py:1264-1268：

- SGLANG_OPT_USE_TILELANG_MHC_PRE=true
- SGLANG_OPT_USE_TILELANG_MHC_POST=true
- SGLANG_OPT_FUSE_MHC_POST_PRE=true
- SGLANG_OPT_USE_FLASHINFER_MHC=false

提交历史中 5b60b7c651 将 SGLANG_OPT_FUSE_MHC_POST_PRE 从 false 改为 true；当前 deepseek_v4.py:222-230、约 1900～1994 行会走 fused MHC 路径。14bef7cd11 的 lazy-load 改动也可能把部分 TileLang 编译推迟到第一次实际调用。

日志证据：

- 2026-09-01 两次冷启动的 MHC prewarm compile 为 52.7～54.4 s；
- 2026-09-03 已命中缓存，MHC prewarm 仅 3.2～3.7 s；
- 2026-09-01 target_verify 过程中 17:55:43 和 17:56:29 仍创建了 mhc_fused_post_pre_fma_tilelang_kernel 产物。

所以 MHC 是真实的次要冷启动成本，并可能在 target_verify 中触发形状特化 kernel；但 2026-09-03 MHC 已基本 warm，而 target_verify 仍需 18.4 分钟，说明 MHC 不是本次 10～13 分钟增长的主因。

### 1.4.2 DeepGEMM 预编译被显式关闭，缓存根目录也迁移

脚本设置 SGLANG_JIT_DEEPGEMM_PRECOMPILE=0。新版本 compile_utils.py:31-40 会把 DG_JIT_CACHE_DIR 重定向到 SGLANG_DG_CACHE_DIR；同时 release 分支把默认 DeepGEMM cache 从旧的 ~/.cache/deep_gemm 迁移到 ~/.cache/sglang/deep_gemm。日志中可见 MHC prewarm 和 target_verify 期间产生 DeepGEMM 产物。

这会造成额外 JIT 和缓存冷启动，但从时间量级看属于次要项。开启 precompile 可能只是把成本前移，必须用相同 cache 和相同环境做 A/B，不能直接当作根因修复。

### 1.4.3 分布式初始化变慢

旧版分布式初始化约 10.6～13.4 s，新版约 62.7～68.1 s，增加约 50～57 s。新版日志使用 NCCL 2.29.7，旧版为 NCCL 2.28.9；两者存在时间相关性，但仅凭现有日志不能证明 NCCL 版本是唯一原因。它明显小于 target_verify 的十几分钟差值。

### 1.4.4 环境不完全可比

2026-09-01 18:04:02 这次日志引用了 simo_sglang_pip 的 site-packages；2026-09-01 17:32:27 和 2026-09-03 使用的是 editable source。当前 Marlin 的 cuda.cu 依赖中同时出现：

- /share/users/like/package/sglang_kernel_src/python/sglang/...
- /share_data/users/like/miniconda3/envs/simo_sglang_pip/lib/python3.12/site-packages/sglang/...

这会造成 build key 和缓存碎片化，因此 18:04:02 不应作为严格的同环境 A/B 样本。后续应固定使用 simo_sglang 的 python 和 sglang 可执行文件。

### 1.4.5 不能归因的告警/阶段

- 新日志明确提示 prefill CUDA graph disabled；因此启动慢不是 prefill graph 捕获。
- FlashInfer 的 libcudart_stub.so: undefined symbol: cudaDeviceReset 会禁用 allreduce fusion。这是 CUDA/TileLang 动态库不匹配，应修复后复测，但现有证据不能把它归因到 18 分钟的 target_verify。
- SIMO plugin 报告找不到 model_runner_kv_cache_mixin，服务仍继续启动；目前没有分钟级耗时证据。
- 2026-09-01 18:04:02 在权重/内存池结束到 target_verify 开始之间还有约 202 s 未被细分，startup timing 的 scheduler_e2e 包含该间隔，需要增加更细日志；这属于额外待查项，不改变 Marlin JIT 是主要热点的判断。

## 1.5 建议的验证和修复顺序

### 1.5.1 先固定环境和新的 canonical cache

在启动进程导入 sglang 之前设置，并确认所有 rank 继承：

~~~bash
export SGLANG_CACHE_DIR=/share_data/users/like/.cache/sglang
export SGLANG_JIT_CACHE_DIR=/data/like/cache/sglang_jit
export SGLANG_DG_CACHE_DIR=/data/like/cache/deep_gemm_cache_dir
export SGLANG_JIT_CACHE_DEBUG=1
which python
which sglang
python -c 'import sglang; print(sglang.__file__)'
~~~

SGLANG_JIT_CACHE_DIR 也可以使用 /share_data/users/like/.cache/sglang/jit，但不要再通过 /softhome 的 symlink 路径作为一侧路径。旧 TVM FFI 目录中的直接 .so 不要直接复制成新 loader 的 manifest；新 loader 的模块名、build key 和依赖布局不同，应在新根目录重新编译一次。

### 1.5.2 修复 cache.py 的 canonical path 比较

在 python/sglang/kernels/jit/utils/compile/cache.py 的 _to_entries 中，让 build_dir 与 candidate 使用同一个 resolve 结果，再执行 is_relative_to。重新生成缓存后，检查 sgl_deps.json 不再包含已删除的 .staging-*/cuda.cu。

不要仅删除共享缓存后就宣布修复；需要用 SGLANG_JIT_CACHE_DEBUG=1 连续启动两次，确认：

- 第一次最多为每个模块编译一次；
- 后续 TP rank 显示 cache hit，而不是四次连续的 moe_wna16_marlin 编译；
- 第二次启动不再出现新的 .staging-UUID 依赖；
- target_verify 的首个 bs=16 时间显著下降。

### 1.5.3 再做隔离 A/B

固定同一个 editable 环境、同一个 canonical cache 后，分别比较：

~~~bash
SGLANG_OPT_FUSE_MHC_POST_PRE=0
~~~

以及当前值；必要时临时降低 cuda-graph-max-bs 或使用 disable-cuda-graph 仅作诊断。若关闭 fused MHC 只减少几十秒，而修复 JIT cache 能减少十几分钟，就能进一步确认归因。

### 1.5.4 最后处理环境告警

修正 FlashInfer/TileLang 与 CUDA 13.0 的动态库匹配，清理或更新 stale SIMO plugin import；同时记录 NCCL 版本和启动阶段的更细时间点。它们是稳定性和次要启动时间问题，不应替代 JIT cache 修复。

## 1.6 最终判断

按当前证据对启动时间增长做排序：

1. **主要根因（高置信度）**：release/v0.5.18-local-dep 新 JIT cache 的 staging 依赖记录受 symlink canonicalization 影响而失效，导致 target_verify 首个 bs=16 触发 4 次串行 Marlin JIT，贡献约 14.5～15 分钟。
2. **阶段级表现（确定）**：因此日志中最明显的新增耗时出现在 target_verify full CUDA Graph capture，约比旧 generic/verify 路径多 10.7～13.2 分钟。
3. **次要因素（中等置信度）**：MHC fused post/pre 的 TileLang 懒编译、DeepGEMM precompile=0 及新 cache 根目录迁移，冷启动约几十秒到 1 分钟。
4. **独立波动项**：分布式初始化多约 50～57 秒；Sep-01 18:04 的环境混用和约 202 s 未细分间隔需要单独复测。
5. **待修复但未证明为主因**：FlashInfer 动态库符号错误、SIMO plugin 导入错误。

因此，优先修复/绕过 SGLANG JIT cache 的 manifest 问题并固定 editable 环境；在此之前对 CUDA Graph、MHC 或 NCCL 做性能结论都容易被反复 JIT 编译噪声掩盖。

---

# 2. 为什么 SIMO 要关闭 chunked prefix cache

本节针对 `sglang` 的 `release/v0.5.18-local-dep`（当前源码提交 `982d8495b7`）和本仓库的 SIMO 量化路径。下面的行号以本次检查的工作树为准，后续提交若插入代码，行号可能变化。所有路径均为各自 code base 的相对路径：SGLang 的根目录是 `/share/users/like/package/sglang_kernel_src`，SIMO 的根目录是 `/share/users/like/package/simo_conda_sglang`。

代码引用采用“相对路径:行号，类::函数”的形式；对于字段、常量和模块级注册，则直接写出对应符号名。

## 2.1 结论

`chunked prefix cache` 有两个容易混淆的“默认值”:

1. **参数定义层面默认允许开启**：`ServerArgs.disable_chunked_prefix_cache` 的默认值是 `False`。这是一个否定式选项，未传 `--disable-chunked-prefix-cache` 就表示“不禁用”。见 `python/sglang/srt/server_args.py:958-962，ServerArgs.disable_chunked_prefix_cache`。
2. **运行时不一定实际开启**：模型加载时，SGLang 会检查模型是否使用 MLA，以及 prefill attention backend 是否在支持列表中；不满足条件时会把生效值改成 `True`（禁用）。见 `python/sglang/srt/model_executor/model_runner.py:369-372，ModelRunner.__init__` 和 `python/sglang/srt/model_executor/model_runner_components/misc_utils.py:25-48，maybe_disable_chunked_prefix_cache`。

所以更准确的回答是：**SGLang 的开关默认是“开启倾向”（disable=False），但功能有运行时能力门控；在当前 SIMO 评测中实际生效值应为 `True`，即关闭。**

字段帮助文本还说明，关闭它可以为短序列节省额外调度/路径开销，见 `python/sglang/srt/server_args.py:958-962，ServerArgs.disable_chunked_prefix_cache`。这只是原生 SGLang 的通用性能取舍；对 SIMO 来说，首要原因是量化 pool 和 kernel 尚未实现该路径所需的接口，不能把本次关闭理解成单纯的性能调参。

## 2.2 “chunked prefix cache”不是普通 prefix cache

普通的 Radix/prefix cache 由 `disable_radix_cache` 控制，该选项在 `python/sglang/srt/server_args.py:937-939，ServerArgs.disable_radix_cache` 中默认也是 `False`。`disable_chunked_prefix_cache` 只控制 DeepSeek MLA 在长前缀场景下采用的“分块 MHA 前缀读取”路径，不能把两个开关等同起来。关闭 chunked prefix cache **不会自动关闭所有 Radix prefix cache**。

## 2.3 SGLang 默认值如何变成实际生效值

### 2.3.1 参数本身

`disable_chunked_prefix_cache=False` 位于 `schedule` 配置组，含义是“不要禁用”。因此命令行不写该选项时，配置初值是允许功能的，而不是明确关闭功能。见 `python/sglang/srt/server_args.py:958-962，ServerArgs.disable_chunked_prefix_cache`。

### 2.3.2 加载时能力门控

`ModelRunner.__init__` 在初始化过程中调用 `maybe_disable_chunked_prefix_cache`，见 `python/sglang/srt/model_executor/model_runner.py:369-372，ModelRunner.__init__`。该函数的判断逻辑见 `python/sglang/srt/model_executor/model_runner_components/misc_utils.py:25-48，maybe_disable_chunked_prefix_cache`：

- draft worker 直接跳过这项修改；
- 读取当前 resolved 的 prefill backend；
- 如果 `use_mla_backend` 为假，或者 prefill backend 不在支持列表中，就通过 `get_context().override` 把生效配置改为 `disable_chunked_prefix_cache=True`；
- 只有条件满足且最终仍为 `False` 时，才记录 “Chunked prefix cache is turned on.”。

支持列表定义在 `python/sglang/srt/server_args.py:211-224，CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS`，当前包括 `flashinfer`、`fa3`、`fa4`、`flashmla`、`cutedsl_mla`、`cutlass_mla`、`trtllm_mla`、`tokenspeed_mla`，不包括 `triton` 或 `triton_simo`。第三方后端可以调用 `python/sglang/srt/server_args.py:399-400，add_chunked_prefix_cache_attention_backend` 主动加入，但“加入白名单”只解决门控，不代表其 kernel 已实现该功能。

对本次两个模型和 backend，实际结果可以概括为：

| 模型/运行路径 | MLA 条件 | backend 是否在白名单 | 未显式传参时的运行时结果 |
|---|---|---|---|
| Llama 3.1 + `triton` | 非 MLA | 否 | 自动禁用 |
| DeepSeek-V2-Lite + `triton` | MLA | 否 | 自动禁用 |
| DeepSeek-V2-Lite + `triton_simo` | MLA | 否 | 自动禁用 |
| DeepSeek MLA + `fa3`/`flashinfer` 等 | MLA | 是 | 可以保持开启 |

这里的 `triton_simo` 只是被 SIMO 注册为 attention backend，见 `simo/extensions/sglang_simo/server_args.py:1-4，SIMO attention backend registration` 和 `simo/extensions/sglang_simo/layers/attention/attention_backend.py:10-18，create_triton_simo_backend`；它没有加入 SGLang 的 chunked-prefix 支持列表。

如果用户没有指定 backend，默认 backend 本身还会随模型架构和 GPU 变化，由 `python/sglang/srt/server_args.py:5870-5942，ServerArgs._get_default_attn_backend` 选择，并由 `python/sglang/srt/arg_groups/overrides.py:2111-2125，_attention_backend_default` 写入 resolved 配置。因此不能只根据“未传 disable 参数”断言每次运行都实际启用。

另外，DeepSeek 的 forward-method registry 对未知 backend 会回退到 `triton` handler，见 `python/sglang/srt/models/deepseek_common/attention_backend_handler.py:37-46，AttentionBackendRegistry.get_handler`。`handle_attention_triton` 对有前缀的 extend 直接走 MLA，只有前缀长度为零时才走 MHA，见 `python/sglang/srt/models/deepseek_common/attention_backend_handler.py:212-226，handle_attention_triton`。这解释了为什么当前 SIMO eager 路径通常不会主动选择 chunked MHA，但不能据此把 SIMO 标成“支持”该特性。

## 2.4 SGLang 的 chunked 路径具体做了什么

DeepSeek MLA 初始化时把 schedule 中的开关和阈值复制到 attention 对象，见 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:132-139，DeepseekMHAForwardMixin.init_mha_forward`。阈值注释说明，前缀总长度默认达到 8192 才考虑该路径；较短的非空前缀继续使用吸收式 MLA，见 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:90-100，DeepseekMHAForwardMixin 的 chunk 配置说明`；默认值实际定义在 `python/sglang/srt/environ.py:586，RuntimeEnvs.SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD`。

通用 DeepSeek backend dispatch 在 `python/sglang/srt/models/deepseek_common/attention_backend_handler.py:99-132，_handle_attention_backend` 中要求同时满足：extend 模式、前缀长度达到阈值、且 `not attn.disable_chunked_prefix_cache`；满足后选择 `MHA_ONE_SHOT` 或 `MHA_CHUNKED_KV`，否则选择 MLA 子路径。

真正的分块执行见 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:286-320，DeepseekMHAForwardMixin.forward_normal_chunked_kv_core`：

1. 先对当前 extend 部分做一次 MHA；
2. 当存在缓存前缀时，把 `forward_batch.mha_return_lse` 设为 `True`；
3. 对每个前缀 chunk 调用 `_chunked_prefix_attn_mha`，再用 LSE 合并各块结果。

逐块读取和合并的实现位于 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:352-418，DeepseekMHAForwardMixin._chunked_prefix_attn_mha`。该函数通过 `_get_mla_kv_buffer` 读取 latent/rope 缓存，经过 `kv_b_proj` 重建 MHA 的 K/V，然后调用 attention 并执行 `merge_state_v2`。上游的 raw reader 入口是 `python/sglang/srt/mem_cache/memory_pool.py:4236-4265，MLATokenToKVPool.get_mla_kv_buffer`，它返回未量化的 MLA buffer。

`RadixAttention.forward` 会根据 `mha_return_lse` 选择带 LSE 的 unified attention op，并在该标志为真时返回 `(output, lse)`，见 `python/sglang/srt/layers/radix_attention.py:150-159,244-277，RadixAttention.forward`。因此这条路径不仅需要“能读 KV”，还需要 backend 正确产出 LSE。

此外，完整 prefill CUDA graph 也读取这个开关：`python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:410-425，PrefillCudaGraphRunner.__init__` 用 `not get_schedule().disable_chunked_prefix_cache` 决定是否建立 chunked-prefix graph 拓扑。它不是单纯的“是否启用 CUDA Graph”开关；`disable_cuda_graph` 与它是两个独立选项。

## 2.5 SIMO 为什么不能沿用这条路径

### 2.5.1 SIMO pool 不是上游 raw MLA buffer

SIMO 的 MLA pool 用 `uint8` 保存打包后的量化 payload 和 scale bytes，见 `simo/extensions/sglang_simo/mem_cache/memory_pool.py:329-349，SIMOMLATokenToKVPool._create_buffers`。这与上游 `MLATokenToKVPool.get_mla_kv_buffer` 所要求的未量化 latent/rope dtype 不同。

因此 SIMO 明确拒绝上游 chunked helper 调用 raw reader，见 `simo/extensions/sglang_simo/mem_cache/memory_pool.py:360-380，SIMOMLATokenToKVPool.get_mla_kv_buffer`。如果把打包字节直接当 BF16 latent 读取，结果不是精度下降这么简单，而是会得到错误的 K/V；要支持它，必须新增按 chunk 索引读取、反量化并恢复布局的实现。

### 2.5.2 SIMO kernel 没有 LSE 返回值

量化路径的 `SIMOTritonAttnBackend.forward_extend` 最终调用 SIMO 自定义 dequant attention kernel。该函数在检测到 `forward_batch.mha_return_lse` 时显式抛出异常，见 `simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py:124-164，SIMOTritonAttnBackend.forward_extend`。原因是当前 kernel 只返回 attention output，不返回 chunk 合并所需的 LSE；而上游 caller 会在 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:302-315，DeepseekMHAForwardMixin.forward_normal_chunked_kv_core` 解包 `(attn_output, lse)`。

所以当前支持 chunked prefix cache 需要同时补齐：

- 量化 MLA pool 的 chunked raw-reader/dequant 接口；
- 与 `merge_state_v2` 对齐的 LSE 输出和数值语义；
- SIMO attention backend 的 prefix-chunk metadata、布局和 CUDA graph 支持。

仅把 `triton_simo` 加入白名单，或仅删除脚本中的 disable 参数，都会绕过自动保护并触发上述未实现接口，不是正确修复。

## 2.6 为什么评测脚本要显式写 `true`

当前脚本在指定 attention backend 时统一传入该选项，见 `simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:103-114，run_simo_config_list`：

```json
"disable_chunked_prefix_cache": true
```

## 16. simo AOT MMA DTE 实现落地、验证结果与问题总结（2026-09-20）

### 16.1 本次实际落地内容

本次实现落在当前可见的 simo code base：

```text
/share/users/like/package/simo_conda_sglang
```

用户指定的 `/share_data/users/like/package/simo_conda_vllm_sipu` 在本次执行环境中不存在，因此没有向该路径写入代码。

新增 `third_party/mma_dte_tile_tensor` submodule，URL 为：

```text
git@gitlabsoft.siorigin.com:oplib/mma_dte_tile_tensor.git
```

构建和安装入口位于 `setup.py:98-144 / _require_mma_dte_artifacts`、`setup.py:162-251 / get_extensions` 和 `setup.py:254-262 / SimoBuildExtension::run`：

1. 只在 `torch.sipu.is_available()` 时构建 MMA DTE AOT library。
2. CMake 只选择本次需要的 7 个 component：MXINT8、MXFP8 BF16/FP16、MXFP6 E2M3/E3M2、MXFP4、MXINT4。
3. `libtile_mma_dte.so`、`libtile_mma_dte.so.1` 和 component DSO 安装到 `simo/mma_dte/`。
4. `_C` 显式链接 `libtile_mma_dte.so` 和 `libtorch_sipu.so`，并使用 `$ORIGIN` RPATH。
5. SIPU SDK 头文件使用 C++20 concepts，因此 SIPU host extension 需要 `-std=c++20`。

`simo/csrc/torch_bindings.cpp:18-32 / TORCH_LIBRARY_FRAGMENT` 注册的最终 schema 是：

```text
mma_dte(Tensor a, Tensor b, int m, int n, int k,
        str mx_dtype, ScalarType out_dtype) -> Tensor
```

A/B layout 参数没有暴露给 Python，C++ wrapper 在 `simo/csrc/sipu/mma_dte.cpp:12-16` 中固定为用户要求的 R32 standard 4x1：

```text
MXINT8/MXFP8: (32, 32, 4, 1, 1), K multiple of 128
MXFP6:         (128, 32, 4, 1, 1), K multiple of 512
MXFP4/MXINT4:  (64, 32, 4, 1, 1), K multiple of 256
C: linear layout
```

Python 薄封装位于 `simo/ops/mma_dte.py:12-27 / mma_dte`，直接调用 `torch.ops.simo.mma_dte`。

### 16.2 构建和安装中发现并修复的问题

正式安装命令：

```bash
pip install . --no-build-isolation
```

首次构建遇到以下问题，均已修复：

| 阶段 | 问题 | 修复 |
|---|---|---|
| host 编译 | SDK `deprecated.h` 使用 C++20 `concept`，PyTorch `CppExtension` 默认用 C++17 | SIPU backend 加 `-std=c++20` |
| `_C` import | `c10::sipu::SIPUStream::stream()` unresolved symbol | 显式链接 `libtorch_sipu.so` |
| wheel runtime | `_C` 依赖 `libtile_mma_dte.so.1`，只复制 `libtile_mma_dte.so` 会缺少 SONAME 文件 | staging 时保留 umbrella 的版本文件和 symlink |
| 测试输入 | `siinfer.hp_to_mx(..., input_layout="tiled")` 要求输入已经 tile reorder；直接传 linear tensor 会得到错误结果 | 用户 linear A/B 使用 `input_layout="linear"` |

安装后 smoke test 已验证：

```text
import simo._C                         成功
torch.ops.simo.mma_dte                 存在
PrivateUse1 dispatch kernel            已注册
```

### 16.3 正确 linear 输入路径的通过结果

测试文件：`tests/sipu_mma_dte/test_mma_dte.py:112-174 / test_mma_dte_matches_cpu_gold`。

测试流程是：CPU 任意形状 A/B -> SIPU -> padding -> `siinfer.hp_to_mx(input_layout="linear", zero_output=True)` -> `torch.ops.simo.mma_dte` -> 裁剪 C -> CPU gold 对比。

输入原始形状为 `A[37,129]`、`B[45,129]`，padding 后按 MX 类型得到合法的 `M=64`、`N=64` 和对应 K。

通过的组合和实际指标如下：

| MX dtype / output | cosine | relative L2 | absolute error | mean relative error |
|---|---:|---:|---:|---:|
| MXINT8 / BF16 | 0.999926 | 0.012152 | 0.513222 | 0.062958 |
| MXINT8 / FP16 | 0.999928 | 0.011977 | 0.480101 | 0.062842 |
| MXINT8 / FP32 | 0.999928 | 0.011977 | 0.480040 | 0.062843 |
| MXFP6 E2M3 / BF16 | 0.999190 | 0.040245 | 1.594976 | 0.180213 |
| MXFP6 E2M3 / FP16 | 0.999194 | 0.040141 | 1.587164 | 0.180155 |
| MXFP6 E2M3 / FP32 | 0.999194 | 0.040146 | 1.587164 | 0.180159 |
| MXFP6 E3M2 / BF16 | 0.997216 | 0.074779 | 3.585276 | 0.435489 |
| MXFP6 E3M2 / FP16 | 0.997225 | 0.074653 | 3.593088 | 0.435287 |
| MXFP6 E3M2 / FP32 | 0.997225 | 0.074653 | 3.592600 | 0.435282 |
| MXFP4 E2M1 / BF16 | 0.987775 | 0.156019 | 6.666809 | 0.793068 |
| MXINT4 / BF16 | 0.983928 | 0.181130 | 7.227488 | 1.170379 |

测试命令使用 `pytest -o addopts=''`，因为当前环境没有 `pytest-cov`，项目 `pyproject.toml:39` 的全局 `addopts=--cov --cov-report` 会导致 pytest 在收集前退出；这不是 MMA DTE 测试失败。

最终结果：

```text
5 passed, 2 xfailed
```

### 16.4 MXFP8 NaN 问题

MXFP8 E4M3/E5M2 的 BF16/FP16 组合都实际调用了 simo MMA DTE，但当前 SDK stack 产生非有限值，因此测试标记为 `xfail`，不是静默跳过。

复现现象：

```text
MXFP8 E4M3 -> BF16: NaN
MXFP8 E4M3 -> FP16: NaN
MXFP8 E5M2 -> BF16: NaN/Inf
MXFP8 E5M2 -> FP16: NaN/Inf
```

这个问题不是 simo wrapper 独有：

1. `siinfer.hp_to_mx(..., "mxfloat8e4m3/e5m2", input_layout="linear")` 生成的 MX buffer，经 `torch_sipu` `MXTileTensor.to_mx(...).to_linear()` 也得到 NaN。
2. 使用 `torch_sipu` 自己的 `MXTileTensor` 和 `_mx_mm`，同样的 MXFP8 输入也得到 NaN。
3. MXINT8、MXFP6、MXFP4、MXINT4 使用同样的 AOT wrapper、stream、C layout 和测试流程时结果正常。

因此当前证据指向 `SIPU SDK v0.4.2 / siinfer hp_to_mx / torch_sipu MXFP8 conversion or hardware model` 这一层，而不是 `simo::mma_dte` 的 dispatcher、DSO 链接或 C++ output dispatch。需要 SDK/torch_sipu 维护者进一步用原始 MMA DTE testcase 和 MXFP8 packed bytes 定位是 quantization header、scale 编码、TensorMap 或 device kernel 问题。

### 16.5 错误输入 layout 的结果

早期测试把普通 row-major linear A/B 直接传给：

```python
siinfer.hp_to_mx(..., input_layout="tiled")
```

此路径的 MXINT8 结果为：

```text
cosine similarity: 0.003040
relative L2:       1.164662
absolute error:    53.346035
```

改为：

```python
siinfer.hp_to_mx(..., input_layout="linear")
```

后 MXINT8 cosine 变为 `0.999926`。这说明 `input_layout="tiled"` 并不是“输出 tile format”的开关，而是声明输入地址已经按 tile 输入布局排列；普通 linear 输入必须使用 `input_layout="linear"`，或者调用方先显式做 tile reorder。

### 16.6 当前限制和后续注意事项

1. 第一版只支持 SIPU AOT、固定 R32 standard 4x1，不支持 R8/R16、2x2、1x4、运行时 layout 参数或 mixed A/B dtype。
2. raw `uint8` MX buffer 不携带逻辑 shape、MX dtype 或 layout metadata，因此 `M/N/K`、`mx_dtype` 必须显式传入。
3. 输出固定为 linear layout，C 的 `tensor_format=0`；返回 shape 是 padded `[M,N]`，由 Python 调用方裁剪真实输出。
4. 当前 output dtype 支持矩阵由 MMA release 的显式实例化决定：MXINT8 和 MXFP6 支持 BF16/FP16/FP32；MXFP8 支持 BF16/FP16；MXFP4/MXINT4 当前只支持 BF16。
5. `check_physical_storage` 位于 `simo/csrc/sipu/mma_dte.cpp:71-99 / check_physical_storage`，按 MX family 检查 physical buffer 容量和 256-byte pointer alignment；不再只检查 `Tensor.numel()>0`。
6. wheel 不携带完整 SIPU SDK，也不修改 Conda 环境；运行前仍需要加载匹配的 SIPU SDK 和 `torch_sipu` runtime。
7. 当前 `mma_dte_tile_tensor` 的 MXFP8 独立 testcase 命名与 layout 元组存在需要核对之处：部分 `test_mxfp8_r32_shape_4x1.cpp` 使用 `(32,32,1,4,1)`，而本次用户要求及 production instantiation 使用 `(32,32,4,1,1)`。扩展 MXFP8 支持前应统一该命名和 `tensor_layout` 维度语义。

### 16.7 本次代码提交范围

本次 commit 只包含：

```text
.gitmodules
setup.py
simo/csrc/torch_bindings.cpp
simo/csrc/sipu/mma_dte.cpp
simo/mma_dte/__init__.py
simo/ops/mma_dte.py
tests/sipu_mma_dte/test_mma_dte.py
third_party/mma_dte_tile_tensor (submodule pointer)
```

工作区中已有的模型、日志、临时目录、编辑器文件和其他未跟踪脚本不会进入本次 commit。

这是一个正确性和防御性设置，原因有三点：

1. 它明确记录了 SIMO 当前不支持该上游路径，不依赖模型识别或 backend 白名单的隐式结果；
2. 即使后续注册流程把 `triton_simo` 加入支持列表，或 SGLang 改变 handler fallback，也不会误进入需要 raw reader/LSE 的路径；
3. 它与 `disable_cuda_graph=true` 的作用不同：前者关闭 DeepSeek chunked-prefix dispatch，后者关闭 CUDA graph，本次显式关闭前者是针对 SIMO KV cache 量化接口的限制。

因此本次评测中应把“实际生效值”理解为 `True`。普通 Radix prefix cache 仍由 `disable_radix_cache=False` 独立控制，短前缀也仍可走普通的吸收式 MLA；关闭的只是长前缀 chunked MHA 方案。

## 2.7 如何判断一次启动的真实状态

不要只看命令行参数名的直觉含义。先看 `maybe_disable_chunked_prefix_cache` 是否打印 `Chunked prefix cache is turned on.`，该日志分支见 `python/sglang/srt/model_executor/model_runner_components/misc_utils.py:42-48，maybe_disable_chunked_prefix_cache`；再结合 resolved schedule config。由于自动门控写入的是 runtime config bag，`python/sglang/srt/runtime_context.py:1119-1120，get_schedule` 返回的是生效配置视图，而不是未经覆盖的原始 `ServerArgs` 对象。

最终判断：**SGLang 的 `ServerArgs` 默认值是 `disable=False`，即功能默认允许；实际是否开启由 MLA/backend 能力门控决定。对当前 Llama3.1 + DeepSeek-V2-Lite 的 SIMO 量化测试，显式 `disable_chunked_prefix_cache=true` 是必要且正确的，当前有效值为关闭。**

## 2.8 四个环境变量的作用，以及与 Marlin/DeepGEMM 编译的关系

本节按当前 checkout（release/v0.5.18-local-dep）的实际代码说明。四个变量都是“缓存/诊断”变量，不是选择 Marlin 或 DeepGEMM 算法的开关；是否走 Marlin 由启动参数 `--moe-runner-backend marlin` 决定，DeepGEMM 是否启用/预编译由另外的 `SGLANG_ENABLE_JIT_DEEPGEMM`、`SGLANG_JIT_DEEPGEMM_PRECOMPILE` 等变量决定。

### 2.8.1 一览

| 环境变量 | 当前代码默认值 | 直接控制的内容 | 对 Marlin | 对 DeepGEMM |
|---|---|---|---|---|
| `SGLANG_CACHE_DIR` | `~/.cache/sglang` | SGLang 通用缓存根，以及 Triton/Inductor/FlashInfer/CUDA driver 等第三方缓存的默认根 | 不直接决定新 `load_jit` 的目录 | 通过默认值间接决定 `<root>/deep_gemm` |
| `SGLANG_JIT_CACHE_DIR` | 未设置时回退到 `~/.cache/sglang/jit` | SGLang 自己的 content-addressed C++/CUDA JIT（新 `load_jit`）根目录 | **直接控制 Marlin .so 的缓存和复用** | 不控制 DeepGEMM |
| `SGLANG_DG_CACHE_DIR` | `<SGLANG_CACHE_DIR>/deep_gemm` | DeepGEMM 编译产物缓存目录 | 不控制 Marlin | **直接控制 DeepGEMM 的 kernel.cu/cubin 缓存和复用** |
| `SGLANG_JIT_CACHE_DEBUG` | `false` | JIT cache miss/rebuild 原因的日志级别 | 只帮助诊断 Marlin cache miss | 不改变 DeepGEMM 编译或其缓存 |

布尔变量接受 `true/1/yes/y` 和 `false/0/no/n`（见 `environ.py:133-140`）。

### 2.8.2 `SGLANG_CACHE_DIR`：通用缓存根

定义在 `python/sglang/srt/environ.py:991`，默认是 `~/.cache/sglang`。导入 `sglang` 时，`sglang/__init__.py:6-14` 很早调用 `redirect_third_party_caches()`；该函数在 `environ.py:1633-1657` 用这个根，通过 `setdefault` 派生：

~~~text
<SGLANG_CACHE_DIR>/triton       -> TRITON_CACHE_DIR
<SGLANG_CACHE_DIR>/inductor     -> TORCHINDUCTOR_CACHE_DIR
<SGLANG_CACHE_DIR>/nv           -> CUDA_CACHE_PATH
<SGLANG_CACHE_DIR>              -> FLASHINFER_WORKSPACE_BASE
~~~

SGLang 的 Rust extension、Torch compile cache、FlashInfer autotune 等也使用这个根。因此它适合把一台机器上的运行时缓存统一放到一个持久卷；它不等同于 Hugging Face 的 `HF_HUB_CACHE`（具体权重下载目录仍由模型加载配置决定），也不是新 Marlin JIT 目录的唯一开关。

一个容易混淆的细节是：当前 `cache.py:301-304` 对未设置的 `SGLANG_JIT_CACHE_DIR` 使用字面量回退 `~/.cache/sglang/jit`，而不是 `$SGLANG_CACHE_DIR/jit`。所以只设置 `SGLANG_CACHE_DIR=/data/like/cache/sglang`，**不会**自动把新 Marlin JIT 移到 `/data/like/cache/sglang/jit`；需要同时显式设置 `SGLANG_JIT_CACHE_DIR`。

由于第三方变量是 `setdefault`，如果 shell 里已经设置了 `TRITON_CACHE_DIR` 等变量，`SGLANG_CACHE_DIR` 不会覆盖它们。当前脚本显式设置了 `TRITON_CACHE_DIR=/data/like/cache/triton_cache_like`，因此实际 Triton 路径不是 `/data/like/cache/sglang/triton`。

### 2.8.3 `SGLANG_JIT_CACHE_DIR`：Marlin 使用的 SGLang JIT 根

新版本的 `load_jit` 使用 `python/sglang/kernels/jit/utils/compile/cache.py:301-304` 选择根目录，缓存布局为：

~~~text
<SGLANG_JIT_CACHE_DIR>/
  <target>/
    <module_name>/
      build-<build_key>/
        deps-<deps_key>/
          <module_name>.so
          sgl_deps.json
~~~

`loader.py:108-115` 先根据源码、编译参数和环境计算 key 并查找有效 leaf；`loader.py:128-169` 在 miss 时加锁、编译、发布；`loader.py:170-171` 删除临时 `.staging-UUID` 目录。有效命中时不再运行 nvcc；TP rank 之间应当是“一次编译，其余 rank 命中”。

当前 Marlin 调用链是：

~~~text
moe_wna16_marlin.py:18-33
    -> load_jit("moe_wna16_marlin", ...)
    -> <SGLANG_JIT_CACHE_DIR>/.../sgl_kernel_jit_moe_wna16_marlin_*.so
~~~

因此：

- Marlin 首次被 CUDA Graph/warmup 触及时，若 leaf 不存在或 manifest 依赖失效，会发生 C++/CUDA 编译；
- 使用同一个有效 cache root，后续进程可以复用 `.so`；
- `SGLANG_JIT_CACHE_DIR` 只改变新 loader 的存放位置和跨进程复用边界，不改变 Marlin kernel 本身；
- 旧 `main-local-dep` 的 loader 使用 `TVM_FFI_CACHE_DIR` 下的直接 `.so`；新 release 的 loader 不读取该旧变量，所以旧目录不会自动成为新 Marlin cache。

这也是本次启动变慢的关键关联：历史日志没有显式设置该变量时，新 loader 使用了 `~/.cache/sglang/jit`；在本机 `/softhome` 是指向 `/share_data` 的 symlink，manifest 中的 staging 路径失效，导致 Marlin 每次启动反复编译。将它设为真实的 canonical 路径（例如 `/data/like/cache/sglang_jit`）可以绕开这类路径混用，但第一次启动仍可能需要编译，必须用第二次启动验证命中。

### 2.8.4 `SGLANG_DG_CACHE_DIR`：DeepGEMM 的缓存目录

定义在 `environ.py:980-982`。默认值是一个懒解析的 `<SGLANG_CACHE_DIR>/deep_gemm`；如果显式设置 `SGLANG_DG_CACHE_DIR`，就使用显式值。

DeepGEMM 原生识别的变量名是 `DG_JIT_CACHE_DIR`。在 `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:31-40` 导入时，SGLang 无条件执行：

~~~python
os.environ["DG_JIT_CACHE_DIR"] = envs.SGLANG_DG_CACHE_DIR.get()
~~~

所以 `SGLANG_DG_CACHE_DIR` 是 SGLang 侧真正的控制入口；只设置 shell 中的 `DG_JIT_CACHE_DIR` 可能在 `compile_utils` 导入后被覆盖。必须在启动 Python、尤其是首次导入 DeepGEMM wrapper 之前设置。

DeepGEMM 的调用/编译过程是：

~~~text
DeepGEMM wrapper execution_hook
    -> 首次遇到 kernel type / shape 时按需预编译
    -> DG_JIT_CACHE_DIR/cache/kernel....../kernel.cu
                                      /kernel.cubin
~~~

相关代码为 `compile_utils.py:115-156,160-220,406-421`。目录变量只决定产物放在哪里以及下次能否复用；是否预编译由：

- `SGLANG_ENABLE_JIT_DEEPGEMM`
- `SGLANG_JIT_DEEPGEMM_PRECOMPILE`
- `SGLANG_JIT_DEEPGEMM_FAST_WARMUP`

等变量决定。当前启动脚本设置 `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`，因此不会执行“所有 M 值的显式预编译”，但实际首次使用的 DeepGEMM shape 仍可能发生按需 JIT。DeepGEMM 的 cache 与 Marlin 的 SGLang JIT cache 是两套独立目录和校验逻辑；清理/迁移其中一套不会让另一套命中。

### 2.8.5 `SGLANG_JIT_CACHE_DEBUG`：只增加 cache miss 原因可见性

定义在 `environ.py:994-996`，默认 `false`。它唯一的行为在 `cache.py:394-397`：

- `false`：把 cache leaf 的依赖变化/缺失原因以 DEBUG 级别记录；
- `true`：提升为 INFO，例如：
  `Rebuilding JIT module ...: abs:/path/to/staging/cuda.cu changed`。

它不会：

- 强制重新编译；
- 清空或修复已有 cache；
- 改变 build key、缓存路径、锁行为或编译并行度；
- 打开/关闭 Marlin 或 DeepGEMM；
- 改变 DeepGEMM 的日志。

另外，若整个 cache scope 尚不存在，当前实现可能没有“changed dependency”可打印；所以没有看到这条 debug 日志，不能单独证明没有发生编译。要判断是否命中，应同时观察 `.so` 时间戳、编译器输出和后续启动是否再次出现 staging/build。

### 2.8.6 当前 `dsv4-flash-run.sh` 的实际配置

当前 `sglang_kernel_src/like-useful/env-build-pip.sh:17-37` 已显式设置：

~~~bash
export SGLANG_CACHE_DIR=/data/like/cache/sglang
export SGLANG_JIT_CACHE_DIR=/data/like/cache/sglang_jit
export SGLANG_DG_CACHE_DIR=/data/like/cache/deep_gemm_cache_dir
export SGLANG_JIT_CACHE_DEBUG=1
~~~

同时还设置了旧/底层变量：

~~~bash
export DG_JIT_CACHE_DIR=/data/like/cache/deep_gemm_cache_dir
export TVM_FFI_CACHE_DIR=/data/like/cache/tvm_ffi_cache_dir
export TRITON_CACHE_DIR=/data/like/cache/triton_cache_like
~~~

当前配置的实际关系是：

| 编译对象 | 实际缓存根 | 说明 |
|---|---|---|
| 新 SGLang JIT / Marlin | `/data/like/cache/sglang_jit` | 由 `SGLANG_JIT_CACHE_DIR` 直接选择 |
| DeepGEMM | `/data/like/cache/deep_gemm_cache_dir` | `compile_utils` 将 `DG_JIT_CACHE_DIR` 强制设为 `SGLANG_DG_CACHE_DIR`；两者当前恰好同值 |
| Triton 等第三方 | `/data/like/cache/triton_cache_like` | 因脚本已显式设置，优先于 `SGLANG_CACHE_DIR` 派生值 |
| 其它 SGLang/第三方缓存 | 以 `/data/like/cache/sglang` 为根 | 例如 FlashInfer、Inductor、Rust extension 等 |

建议在所有 rank 启动前确认：

~~~bash
source /share/users/like/package/sglang_kernel_src/like-useful/env-build-pip.sh
which python
which sglang
python -c 'import os, sglang; from sglang.srt.environ import envs; print({k: (os.environ.get(k), getattr(envs, k).get()) for k in ("SGLANG_CACHE_DIR", "SGLANG_JIT_CACHE_DIR", "SGLANG_DG_CACHE_DIR", "SGLANG_JIT_CACHE_DEBUG")})'
~~~

### 2.8.7 与本次启动耗时问题的对应关系

可以用下面的因果链理解四个变量：

~~~text
SGLANG_JIT_CACHE_DIR
    -> Marlin load_jit 是否能找到有效 .so/manifest
    -> 是否在 target_verify 的 bs=16 首次 warmup 中运行 nvcc
    -> target_verify 是否出现每个 TP rank 约 3.5 分钟的串行编译

SGLANG_DG_CACHE_DIR
    -> DeepGEMM kernel.cubin 是否可复用
    -> 影响 DeepGEMM 按需编译/预编译的冷启动时间
    -> 不会修复或触发 Marlin 的 load_jit cache

SGLANG_CACHE_DIR
    -> 通用第三方/运行时缓存的根
    -> 可能影响 Triton、FlashInfer、Inductor 等其它 kernel 的冷启动
    -> 当前代码不会单独改变 Marlin 的 JIT 根

SGLANG_JIT_CACHE_DEBUG=1
    -> 把 Marlin JIT cache 的失效原因显示在 INFO 日志
    -> 便于确认“缺文件/依赖变化/manifest 失效”
    -> 不改变上述任何编译动作
~~~

所以，针对本次问题最重要的配置是 `SGLANG_JIT_CACHE_DIR`；DeepGEMM 对应看 `SGLANG_DG_CACHE_DIR`；`SGLANG_CACHE_DIR` 不能替代前两者；`SGLANG_JIT_CACHE_DEBUG` 只是诊断开关。四项都应在启动前统一 export，并对同一 editable 环境连续启动两次：第一次允许产生编译，第二次应看到 Marlin/DeepGEMM cache 命中而不再重复产生相同的编译序列。

---

## 2.9 本次追加结论：chunked prefix cache

按 `release/v0.5.18-local-dep` 当前源码，`ServerArgs.disable_chunked_prefix_cache` 的字段默认值是 `False`，因此**配置语义是默认允许 chunked prefix cache**，见 `python/sglang/srt/server_args.py:958-962，ServerArgs.disable_chunked_prefix_cache`。这不是“每个模型都实际开启”：`python/sglang/srt/model_executor/model_runner.py:369-372，ModelRunner.__init__` 会调用 `python/sglang/srt/model_executor/model_runner_components/misc_utils.py:25-48，maybe_disable_chunked_prefix_cache`，对非 MLA 模型或不在 `python/sglang/srt/server_args.py:211-224，CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS` 中的 prefill backend 自动覆写为 `True`（禁用）。

此前版本的评测曾把 Llama3.1 权重量化也指定为 `triton`，而 DeepSeek KV 量化使用 `triton_simo`。本次修正后，权重量化不再指定 attention backend，只有 KV 量化保留 `triton_simo`；KV 分支在 `simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:107-114，run_simo_config_list` 显式传入 `"disable_chunked_prefix_cache": true`，是为了把 packed-KV 的兼容性限制固定下来，而不是依赖隐式门控。

关闭的直接原因是 SIMO 尚未实现上游 chunked 路径的两个契约：`simo/extensions/sglang_simo/mem_cache/memory_pool.py:360-380，SIMOMLATokenToKVPool.get_mla_kv_buffer` 不能把 packed `uint8` + scale 当作 raw MLA BF16 buffer 读取；`simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py:124-164，SIMOTritonAttnBackend.forward_extend` 不产生上游分块合并所需的 LSE。上游分块 caller 在 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:286-320，DeepseekMHAForwardMixin.forward_normal_chunked_kv_core` 会读取每个 prefix chunk 并解包/合并 `(output, lse)`。

因此：**字段默认值为“允许”（False），当前 SIMO KV 分支的有效值为“关闭”（True）**。这不会关闭 `disable_radix_cache=False` 所控制的普通 Radix prefix cache，也不会关闭普通 chunked prefill；仅禁止 DeepSeek 长前缀的 chunked-MHA + LSE 拓扑。若删除 KV 分支的保护参数，或把 `triton_simo` 加入 chunked-prefix 支持白名单而不实现上述 reader/LSE 接口，都会重新暴露错误路径。

---

# 3. `sglang serve` 如何把命令行参数变成 `ServerArgs` 成员

本节针对 SGLang `release/v0.5.18-local-dep`，源码提交 `982d8495b7`。代码引用统一采用“相对 code base 路径:行号，类::函数”的形式；字段、常量和模块级配置使用对应符号名。

## 3.1 结论先行

这不是在某处手写一条 `parser.add_argument("--disable-chunked-prefix-cache", ...)`，而是 dataclass 字段自动生成 CLI 参数，再由同名 `dest` 传回 dataclass：

```text
--disable-chunked-prefix-cache
    -> sglang.cli.main::main 的 extra_argv
    -> sglang.cli.serve::serve 的 request.argv
    -> server_args::prepare_server_args 的 argparse Namespace
    -> Namespace.disable_chunked_prefix_cache
    -> ServerArgs::from_cli_args 的 kwargs
    -> ServerArgs(...).disable_chunked_prefix_cache
```

## 3.2 `sglang` 可执行文件先进入哪里

安装元数据在 `python/pyproject.toml:201-203，[project.scripts]` 声明：

```toml
sglang = "sglang.cli.main:main"
```

因此执行 `sglang serve ...` 时，生成的 console-script wrapper 只负责导入并调用 `python/sglang/cli/main.py:12-40，main`。

## 3.3 顶层 parser 为什么不直接解析这个选项

`python/sglang/cli/main.py:12-21，main` 创建顶层 `argparse` 和 `serve` 子命令。`serve` 子命令没有注册全部 server 选项，并设置了 `add_help=False`。

`python/sglang/cli/main.py:35-40，main` 调用 `parser.parse_known_args()`。对于：

```bash
sglang serve --model-path /path/to/model --disable-chunked-prefix-cache
```

顶层 parser 只识别 `subcommand="serve"`，其余 server 选项留在 `extra_argv`，再传给 `python/sglang/cli/serve.py:166-205，serve`。

`serve` 在 `python/sglang/cli/serve.py:169-177，serve` 只处理 `--model-type` 和位置形式的 model path；`--disable-chunked-prefix-cache` 原样保留在 `dispatch_argv` 和 `ServeRequest.argv`。LLM backend 的 `run=_run_llm` 注册于 `python/sglang/cli/serve.py:129-135，_create_backend_registry`。

## 3.4 LLM backend 把 argv 交给真正的 server parser

`python/sglang/cli/serve.py:90-99，_run_llm` 执行：

```python
server_args = prepare_server_args(list(request.argv))
run_server(server_args)
```

也就是说，`sglang serve` 的参数解析分成两层：顶层 CLI 负责识别子命令，`prepare_server_args` 才负责识别 `--disable-chunked-prefix-cache`。

## 3.5 `prepare_server_args` 创建 parser、解析 argv

`python/sglang/srt/server_args.py:9770-9804，prepare_server_args` 的关键顺序是：

1. `:9781` 创建 `ArgumentParser(prog="sglang serve")`；
2. `:9782` 调用 `ServerArgs.add_cli_args(parser)`；
3. 有 `--config` 时，`:9784-9791` 通过 `ConfigArgumentMerger.merge_config_with_args` 合并 YAML 参数；
4. `:9793` 执行 `raw_args = parser.parse_args(argv)`；
5. `:9804` 调用 `ServerArgs.from_cli_args(raw_args)`。

所以 `raw_args` 是一个 `argparse.Namespace`，在这一步已经有 `raw_args.disable_chunked_prefix_cache` 属性。

## 3.6 `ServerArgs.add_cli_args` 使用 dataclass 反射

`python/sglang/srt/server_args.py:8658-8662，ServerArgs::add_cli_args` 没有为该选项单独写 `add_argument`，而是调用 `add_cli_args_from_dataclass(parser, ServerArgs)`。自动注册函数位于 `python/sglang/srt/arg_groups/arg_utils.py:218-337，add_cli_args_from_dataclass`，它读取类型注解和 dataclass 字段，逐个生成 argparse action。

## 3.7 字段声明决定 CLI 名称和默认值

字段定义在 `python/sglang/srt/server_args.py:958-962，ServerArgs::disable_chunked_prefix_cache`：

```python
disable_chunked_prefix_cache: A[
    bool,
    "Disable chunked prefix cache feature for deepseek, ...",
    NS("schedule"),
] = False
```

`ServerArgs` 的说明在 `python/sglang/srt/server_args.py:447-470，ServerArgs` 指出：`A` 是 `typing.Annotated` 的别名，字段名会自动转换为 CLI 名称。这里的 `NS("schedule")` 只是运行时配置分组标记，不是 CLI 名称的一部分；它不会生成 `--schedule-disable...`。

## 3.8 下划线如何变成连字符

`python/sglang/srt/arg_groups/arg_utils.py:208-210，_field_to_cli_name` 的实现是：

```python
return "--" + name.replace("_", "-")
```

所以 `disable_chunked_prefix_cache` 会变成 `--disable-chunked-prefix-cache`。在 `python/sglang/srt/arg_groups/arg_utils.py:231-249，add_cli_args_from_dataclass` 中，函数先取得 `Annotated` 元数据，再调用 `_field_to_cli_name(field.name)` 生成 `cli_name`，并把该名称交给 `parser.add_argument`。

`python/sglang/srt/arg_groups/arg_utils.py:147-163，_unwrap_annotated` 负责取出 `Annotated` 的内部类型和 metadata；字段中的裸帮助字符串会被转换成 `Arg(help=...)`，所以该字段具备 CLI 注册所需的 metadata。

## 3.9 为什么 argparse 的 `dest` 恰好是下划线字段名

`python/sglang/srt/arg_groups/arg_utils.py:247-254，add_cli_args_from_dataclass` 还计算 `auto_dest`，并在字段名与自动生成的 dest 不同时才显式传入 `dest`：

```python
auto_dest = cli_name.lstrip("-").replace("-", "_")
dest_kwarg = {"dest": field.name} if field.name != auto_dest else {}
```

对本字段，`cli_name` 是 `--disable-chunked-prefix-cache`，`auto_dest` 和 `field.name` 都是 `disable_chunked_prefix_cache`。两者相等，所以不需要显式传 `dest`；argparse 的默认规则也会得到同名 dest。这就是连字符 CLI 名与下划线 Python 成员之间的直接连接。

## 3.10 `bool` 字段如何生成 `store_true`

`python/sglang/srt/arg_groups/arg_utils.py:313-319，add_cli_args_from_dataclass` 对 `bool` 类型走专门分支：

```python
kwargs = dict(action="store_true", help=arg_meta.help, **dest_kwarg)
kwargs["default"] = default
parser.add_argument(*names, **kwargs)
```

因此该参数实际等价于：

```python
parser.add_argument(
    "--disable-chunked-prefix-cache",
    action="store_true",
    default=False,
)
```

行为是：

| 命令行 | `raw_args.disable_chunked_prefix_cache` |
|---|---:|
| 不写该选项 | `False` |
| 写一次 `--disable-chunked-prefix-cache` | `True` |
| 写 `--disable-chunked-prefix-cache=true` | 解析错误，因为 `store_true` 不接收值 |

这里不是 `BooleanOptionalAction`，所以不能写成 `--no-disable-chunked-prefix-cache` 来显式恢复 `False`；恢复默认值的方式是不传该 flag，或由配置/代码设置字段。

## 3.11 `Namespace` 如何变成 `ServerArgs` 成员

`python/sglang/srt/server_args.py:8920-8928，ServerArgs::from_cli_args` 遍历 `dataclasses.fields(cls)`，只保留 Namespace 中存在的同名属性，然后执行：

```python
return cls(**{
    attr: getattr(args, attr)
    for attr in attrs
})
```

当命令行带 flag 时，效果等价于：

```python
ServerArgs(
    ...,
    disable_chunked_prefix_cache=True,
)
```

这里的 `cls` 就是 `ServerArgs`，所以 dataclass 生成的构造函数把该关键字写入实例成员 `server_args.disable_chunked_prefix_cache`。`ServerArgs::__post_init__` 在 `python/sglang/srt/server_args.py:3585-3586，ServerArgs::__post_init__` 中继续调用 `_run_resolution_pipeline`；这发生在成员已经由构造函数接收之后。

## 3.12 之后谁读取这个成员

`python/sglang/cli/serve.py:90-99，_run_llm` 将构造好的对象传给 `python/sglang/launch_server.py:16-53，run_server`，再进入 HTTP/gRPC launcher。后续 ModelRunner 初始化时，`python/sglang/srt/model_executor/model_runner.py:369-372，ModelRunner::__init__` 会读取/处理该配置，并调用 `python/sglang/srt/model_executor/model_runner_components/misc_utils.py:25-48，maybe_disable_chunked_prefix_cache` 做 MLA/backend 能力门控。

这一步可能通过 runtime context override 将生效配置视图改为禁用，但它与前面的 CLI 映射是两个阶段：

```text
argv -> Namespace -> ServerArgs 成员       （本节说明的映射）
                         |
                         -> ModelRunner/runtime gate 可能覆写生效值
```

## 3.13 可复现的最小检查

在目标 editable 环境中，可以不启动模型而直接观察两层值：

```bash
/share_data/users/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
import argparse
from sglang.srt.server_args import ServerArgs

parser = argparse.ArgumentParser(prog="sglang serve")
ServerArgs.add_cli_args(parser)
ns = parser.parse_args([
    "--model-path", "dummy",
    "--disable-chunked-prefix-cache",
])
print(vars(ns)["disable_chunked_prefix_cache"])  # True
args = ServerArgs.from_cli_args(ns)
print(args.disable_chunked_prefix_cache)           # True
PY
```

这个结果对应源码中的两步：`python/sglang/srt/server_args.py:9793，prepare_server_args` 产生 Namespace，`python/sglang/srt/server_args.py:8921-8928，ServerArgs::from_cli_args` 再把同名属性传入构造函数。

兼容的旧入口 `python -m sglang.launch_server` 不经过 `sglang.cli.main::main`，但会在 `python/sglang/launch_server.py:65-70，__main__` 调用同一个 `prepare_server_args(sys.argv[1:])`，所以字段映射结果相同。

最终可用一句话概括：**`--disable-chunked-prefix-cache` 先由 `_field_to_cli_name` 从 `disable_chunked_prefix_cache` 自动生成，再由 argparse 的默认 `dest` 存回同名 Namespace 属性，最后由 `ServerArgs::from_cli_args` 以同名关键字构造出 `ServerArgs.disable_chunked_prefix_cache`。**

# 4. 评测脚本中 CUDA graph、attention backend 与 EP 的修正

本节针对 SGLang `release/v0.5.18-local-dep`（源码提交 `982d8495b7`）和当前 SIMO 工作树。所有代码说明均使用相对 code base 路径、行号和函数名；SGLang 的 code base 根目录是 `/share/users/like/package/sglang_kernel_src`，SIMO 的 code base 根目录是 `/share/users/like/package/simo_conda_sglang`。

## 4.1 为什么之前加了 `disable_cuda_graph`

此前工作树中的
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:103-114，run_simo_config_list`
把下面三个选项放在同一个分支中：

```json
"skip_server_warmup": true,
"disable_cuda_graph": true,
"disable_chunked_prefix_cache": true
```

这是一种适配期间的保守 workaround，用来先避开启动阶段的 graph capture、warmup 和 DeepSeek chunked-prefix 路径；它不是 release 分支要求 SIMO 必须关闭 CUDA graph 的接口变更。当前提交历史和源码中没有针对 `quantization="simo"` 的全局 graph 禁止规则。

`python/sglang/srt/server_args.py:1914，ServerArgs::disable_cuda_graph` 的默认值是 `False`，并且该字段标注为 `Arg(no_cli=True)`，说明它是兼容字段而不是新的公开 CLI 开关。真正的副作用在
`python/sglang/srt/server_args.py:4485-4517，ServerArgs::_parse_cuda_graph_config`：当该字段为 `True` 时，函数同时执行：

```python
_set(Phase.DECODE, "backend", Backend.DISABLED)
_set(Phase.PREFILL, "backend", Backend.DISABLED)
```

因此它关闭的是 decode 和 prefill 两个阶段，而不是只绕过 SIMO KV cache 的某一个不兼容分支。

release 的普通 CUDA 默认仍然允许 graph：

- `python/sglang/srt/model_executor/cuda_graph_config.py:110-119，default_prefill_backend` 在 CUDA 上返回 `Backend.BREAKABLE`；
- `python/sglang/srt/model_executor/cuda_graph_config.py:122-131，CudaGraphConfig::__init__` 将 decode 默认设为 `Backend.FULL`，prefill 默认设为上述 backend；
- `python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py:89-180，capture_cuda_graphs` 按配置捕获 prefill 和 decode；
- `python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py:401-490，capture_decode_graph` 只有在 decode backend 解析为 `disabled` 等条件下才返回空 runner。

SIMO 量化代码也没有拒绝普通 graph。比如：

- `simo/extensions/sglang_simo/layers/attention/triton_ops/extend_attention.py:896-904，extend_attention_fwd` 在 CUDA capture 中只跳过 debug 文件读写；
- `simo/extensions/sglang_simo/layers/attention/triton_ops/decode_attention.py:603-609，decode_attention_fwd` 做同样的 capture 判断；
- `simo/extensions/sglang_simo/mem_cache/memory_pool.py:332-351，SIMOMLATokenToKVPool::_create_buffers` 使用固定形状的 `uint8` buffer，适合 graph replay；
- `tests/simo_quant/test_mx_ops.py:391-423，test_upcast_from_mxfmt_compile_cuda_graph_replay_matches_eager` 和 `tests/simo_quant/test_flexpoint_ops.py:524-563，test_per_group_downcast_compile_cuda_graph_replay_matches_eager` 直接用 `torch.compile`、`torch.cuda.CUDAGraph` 和 replay 验证 SIMO kernel。

所以结论是：**适配 release 后并不是“基础 CUDA graph 不支持了”；之前的开关过于宽泛。** 当前修改删除了 `skip_server_warmup`/`disable_cuda_graph` 这组临时参数，让 SGLang 恢复正常 warmup 和 graph 决策。需要区分的是，release 仍可能因为模型、prefill backend、DCP、LoRA 等独立规则自动关闭某一阶段的 graph；这不等于 SIMO 普遍不支持 decode graph。

## 4.2 恢复 graph 参数时为什么使用新字段

旧脚本注释中的 `cuda_graph_max_bs` 不能原样复制到 lm-eval 的 JSON model args。lm-eval 的 SGLang adapter 最终调用 Engine；
`python/sglang/srt/entrypoints/engine.py:232-252，Engine::__init__` 在没有现成 `server_args` 时直接执行：

```python
server_args = self.server_args_class(**kwargs)
```

release 的 dataclass 字段是
`python/sglang/srt/server_args.py:1879-1883，ServerArgs::cuda_graph_max_bs_decode`，而旧名字只在
`python/sglang/srt/server_args.py:8774-8780，ServerArgs::add_cli_args` 中作为命令行 `--cuda-graph-max-bs` 的 deprecated alias，映射到 `cuda_graph_max_bs_decode`。直接把旧 key 放进 Engine kwargs 会因为 `ServerArgs` 构造函数没有该成员而报 `TypeError`。

因此当前脚本使用：

```json
"cuda_graph_max_bs_decode": 128
```

具体位置是
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:103-106，run_simo_config_list`
和
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:123-137，run_no_quant_eval`。

这个参数只设置 decode graph 的最大捕获 batch size，不会关闭 graph。没有显式传 `cuda_graph_backend_decode` 时，
`python/sglang/srt/server_args.py:4519-4527，ServerArgs::_parse_cuda_graph_config`
会把它写入 decode phase 的 `max_bs`，backend 仍来自默认配置，通常是 `full`。如果是实际的 `sglang serve` 命令行，旧的 `--cuda-graph-max-bs 128` 仍可用但会给出 deprecated 语义；在本评测脚本的 Engine JSON 中应使用新字段。

评测日志若出现 `Disable prefill CUDA graph because the capture size is not set`，不应误认为 decode graph 被禁用。
`python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py:294-297，capture_prefill_graph`
在 prefill capture bucket 为空时只关闭 prefill runner；本脚本通过 lm-eval 使用 `chunked_prefill_size=-1` 的默认值时，出现这条日志是预期行为，decode phase 仍按 `cuda_graph_max_bs_decode` 捕获。

## 4.3 attention backend 只在 KV 量化时切换

`SIMOLinearMethod` 和 `SIMOFusedMoEMethod` 只替换权重加载/专家计算，不读取 SIMO 打包 KV cache。把 `triton` 强行传给所有权重量化评测，会改变原脚本的 backend 选择范围，也会让不涉及 KV 的测试承担额外的 Triton attention 约束。

当前
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:141-154，run_model_evaluations`
传入空字符串：

```bash
run_simo_config_list ... "online_quantization" "QUANT_CONFIGS" "" ""
```

`run_simo_config_list` 在
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:107-108，run_simo_config_list`
只有在 backend 非空时才把 `attention_backend` 加进 model args，因此权重量化测试完全交给 SGLang 默认解析。默认选择逻辑在
`python/sglang/srt/server_args.py:5870-5942，ServerArgs::_get_default_attn_backend`：MHA/MLA 会根据 GPU、模型结构和可用实现选择 `fa3`、`flashinfer` 或 `triton`，而不是由 SIMO 评测脚本硬编码。

KV 量化则必须走 SIMO backend。当前
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:157-168，run_model_evaluations_kv_cache_quant`
仍传入 `triton_simo`；对应的条件分支在
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:107-114，run_simo_config_list`：

```text
attention_backend == "triton_simo"
    -> attention_backend="triton_simo"
    -> disable_chunked_prefix_cache=true
```

这是有意的两类分流：

- `simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py:108-122，SIMOTritonAttnBackend::_layer_uses_simo_kv_cache` 检查 layer 的 `kv_cache_quant_spec` 是否和 SIMO pool/backend 一致；普通权重量化不需要这个 backend；
- `simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py:124-164，SIMOTritonAttnBackend::forward_extend` 对 `mha_return_lse=True` 明确抛出异常，因为当前自定义 kernel 只返回 attention output；
- `simo/extensions/sglang_simo/mem_cache/memory_pool.py:360-380，SIMOMLATokenToKVPool::get_mla_kv_buffer` 拒绝把 packed `uint8` + scale bytes 当成上游 raw MLA BF16 buffer。

最后两点只影响 DeepSeek 的 chunked-prefix MHA/LSE 拓扑，因此 `disable_chunked_prefix_cache=true` 仍保留在 KV 分支，但不会污染权重量化分支。它也不会关闭普通 Radix prefix cache 或普通 chunked prefill。

## 4.4 为什么不再无条件限制 `SIMOFusedMoEMethod` 为 EP=1

此前在
`simo/extensions/sglang_simo/quantization/quantization.py:1413-1417，SIMOFusedMoEMethod::get_moe_weight_loader`
临时加入了：

```python
if layer.moe_ep_size > 1:
    raise NotImplementedError(...)
```

这个限制不是 release API 的必然要求，也不是原有 SIMO EP 设计的一部分。它更像是当时尚未确认 release 的 global/local expert loader 语义时的防御性假设；如果 loader 语义不匹配，正确修复应是调整映射，而不是在方法构造阶段阻断所有 EP 配置。本次已删除该无条件 raise，保留现有 EP 分支。

release 的 EP 数据流和当前 SIMO 代码是相互对应的：

1. `python/sglang/srt/server_args.py:2338-2346，ServerArgs::ep_size` 默认值为 `1`，所以本次脚本的 TP=1 测试自然仍运行在 EP=1；这只是测试范围，不应升级成方法级硬限制。
2. `python/sglang/srt/layers/moe/fused_moe_triton/layer.py:280-306，FusedMoE::__init__` 根据 `moe_ep_size` 计算 `_num_local_routed` 和 `num_local_experts`；
   `python/sglang/srt/layers/moe/fused_moe_triton/layer.py:414-429，FusedMoE::__init__` 把 `num_experts=self.num_local_experts` 传给 quant method，权重 tensor 本来就是本 rank 的 local expert 形状。
3. `python/sglang/srt/layers/moe/token_dispatcher/standard.py:186-238，StandardDispatcher::dispatch` 在 Triton 标准路径把 global top-k expert id 映射为 local id，不属于本 rank 的专家用 `-1` 标记。
4. `simo/extensions/sglang_simo/quantization/quantization.py:1413-1425，SIMOFusedMoEMethod::get_moe_weight_loader` 对 checkpoint 的 global expert id 调用 `_map_global_expert_id_to_local_expert_id`，非本 rank 专家跳过，本 rank 专家继续交给 release 的原始 loader；
   `simo/extensions/sglang_simo/quantization/quantization.py:1715-1757，SIMOFusedMoEMethod::apply` 根据 `num_experts != num_local_experts` 设置 `is_ep`，传入 local expert 数并启用 `filter_expert`。
5. `simo/extensions/vllm_simo/model_executor/layers/fused_moe/fused_moe.py:539-632，moe_align_block_size` 在 `filter_expert=True` 时把 `-1` 转成 invalid slot，再把对应 block 标成 `-1`；这正是 StandardDispatcher 输出格式所需的过滤。

因此对当前两个目标模型和本次最小改动目标，结论是：**不要在 `SIMOFusedMoEMethod` 构造时无条件拒绝 EP>1；保持 release/SIMO 已有的 global-to-local 和 invalid-expert 处理。** 本次回归范围仍是脚本默认的 EP=1，尚未把多进程 EP>1 端到端精度作为验收项；若以后要宣称 EP>1 的生产支持，应另做多 rank loader、dispatch、all-reduce 和精度测试，而不是重新加一个未经验证的全局禁用。

## 4.5 本次修改与检查结果

已完成的代码修改：

- `simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:103-114，run_simo_config_list` 删除 `skip_server_warmup` 和 `disable_cuda_graph`，加入 `cuda_graph_max_bs_decode`，并只对 `triton_simo` 加 `disable_chunked_prefix_cache`；
- `simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:141-168，run_model_evaluations/run_model_evaluations_kv_cache_quant` 让权重量化使用默认 attention，KV 量化继续使用 `triton_simo`；
- `simo/extensions/sglang_simo/quantization/quantization.py:1413，SIMOFusedMoEMethod::get_moe_weight_loader` 删除 EP=1 硬限制。

检查结果：

- `bash -n simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh` 通过；
- 生成的 weight/KV model args 均通过 JSON 解析检查，weight args 不含 `attention_backend`，KV args 含 `triton_simo` 和 `disable_chunked_prefix_cache=true`，两者都不含 `disable_cuda_graph`；
- `python -m py_compile` 通过（修改的 SIMO quantization、attention、memory pool 文件）；
- `CUDA_VISIBLE_DEVICES=4 pytest -q tests/simo_quant/test_mx_ops.py::test_upcast_from_mxfmt_compile_cuda_graph_replay_matches_eager` 通过；
- `CUDA_VISIBLE_DEVICES=4 pytest -q tests/simo_quant/test_flexpoint_ops.py::test_per_group_downcast_compile_cuda_graph_replay_matches_eager` 通过。

这两项 graph replay 测试验证的是 SIMO 量化 kernel 的 CUDA graph 兼容性；它们不替代两个完整模型的 lm-eval 精度测试。完整评测重新开启 graph 后，日志中应重点检查 `server_args.cuda_graph_config.decode.backend` 是否为 `full`，以及 decode 请求是否显示 `cuda graph: True`；prefill 是否采用 graph 则仍由 release 的模型/backend 能力门控决定。

补充做了一次真实 Engine smoke：使用 Llama3.1-8B、`w8a8_mxfp`、单 GPU、`cuda_graph_max_bs_decode=2` 和 `gsm8k --limit 1` 启动。日志实际显示 `attention_backend='fa3'`、`disable_cuda_graph=False`、`cuda_graph_config.decode.backend='full'`，随后出现 `Capture target decode CUDA graph begin/end`，请求阶段显示 `cuda graph: True`，因此恢复后的参数链路已在真实模型上验证。第一次尝试因 shell `PATH` 未包含 conda 环境中的 `ninja` 而失败，补充 `PATH=/share_data/users/like/miniconda3/envs/simo_sglang/bin:$PATH` 后成功；这属于构建工具环境问题，不是 graph 或 SIMO kernel 错误。

随后用 Llama3.1-8B 的 `kvquant_mxfp8` 配置、`attention_backend=triton_simo`、`disable_chunked_prefix_cache=true` 做了同样的单请求 smoke。`simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py:47-106，SIMOTritonAttnBackend::__init__` 成功创建量化 KV pool/backend，日志完成 `Capture target decode CUDA graph`，请求阶段同样显示 `cuda graph: True`。这验证的是当前 release 下 SIMO 自定义 KV read/write 路径可以参与普通 decode graph；它不表示 chunked-prefix 的 LSE 路径已实现，后者仍由前述显式开关关闭。

## 4.6 恢复完整评测矩阵

应用户要求，当前
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:25-39，QUANT_CONFIGS`
和
`:42-50，KV_CACHE_QUANT_CONFIGS`
已恢复为完整配置数组，不再只筛选 `w8a8_mxfp` 与 `kvquant_mxfp8`。同时
`:141-154，run_model_evaluations`
重新调用
`:123-137，run_no_quant_eval`
先执行每个模型的无量化基线，再执行全部权重量化配置；KV 量化仍由
`:157-168，run_model_evaluations_kv_cache_quant`
单独执行，并继续只在该分支使用 `triton_simo` 和 `disable_chunked_prefix_cache=true`。因此脚本现在会运行两个模型的无量化、完整权重量化和完整 KV cache 量化测试。

# 5. v0.5.18 适配 commit 逐处说明

## 5.1 commit 范围与结果

本次代码 commit 为 `4da2709396207c241003db49085e516093a62df1`，提交信息是
`fix(sglang-simo): adapt quantization hooks to v0.5.18`。提交基于 SIMO 当前分支
`like-debug-log`，目标 SGLang 是 `/share/users/like/package/sglang_kernel_src` 的
`release/v0.5.18-local-dep`（源码提交 `982d8495b7`）。commit 只包含 7 个已经跟踪的
适配文件，共 `236 insertions(+), 60 deletions(-)`：

1. `simo/extensions/sglang_simo/__init__.py`
2. `simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh`
3. `simo/extensions/sglang_simo/layers/attention/triton_ops/decode_attention.py`
4. `simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py`
5. `simo/extensions/sglang_simo/mem_cache/init_memory_pool_patch.py`
6. `simo/extensions/sglang_simo/mem_cache/memory_pool.py`
7. `simo/extensions/sglang_simo/quantization/quantization.py`

工作树中原有的编辑器文件、`temp/`、`results/`、实验脚本和 `like-useful` 符号链接
没有被加入 commit。`QUANT_CONFIGS` 与 `KV_CACHE_QUANT_CONFIGS` 在 commit 的父版本中
已经恢复，因此它们不是本 commit 的新增 diff；本 commit 只继续修正这些数组调用时的
Engine 参数和 backend 分流。`SIMOFusedMoEMethod` 的主体也没有在本 commit 中改写；
此前已经完成的 EP 映射修改保留在父版本中，本次 commit 只修改线性权重 loader 的
release 签名兼容层。

## 5.2 插件注册入口

### 5.2.1 `register_simo_extensions`

文件：`simo/extensions/sglang_simo/__init__.py:22-27，register_simo_extensions`

旧代码从已经删除的 `ModelRunnerKVCacheMixin` 路径导入并调用
`apply_init_memory_pool_patch()`。v0.5.18 把 KV pool 的构造职责移到
`KVCacheConfigurator._build_token_to_kv_pool`，旧模块导入会在插件注册阶段直接触发
`ModuleNotFoundError`。因此这里改为导入并调用
`apply__build_token_to_kv_pool_patch()`，让 SIMO 的 KV 量化 patch 在新 configurator
入口执行。其余 loader、quantization registry、attention backend 和 DeepSeek 模型
注册顺序不变，非 KV 量化模型不会额外创建 SIMO pool。

## 5.3 在线评测脚本

### 5.3.1 管道失败状态

文件：`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:3，脚本顶层`

`run_eval`（`:57-75，run_eval`）把 `lm-eval` 输出通过 `tee` 写入日志。只有
`set -e` 时，某些 shell 配置可能只观察到 `tee` 的成功退出码，掩盖上游评测失败。
加入 `set -o pipefail` 后，管道任一环节失败都会使该测试返回失败，便于完整矩阵脚本
在第一处错误停止并保留正确的失败状态。

### 5.3.2 `run_simo_config_list` 的通用 Engine 参数

文件：`.../llm_eval_online_quant.sh:103-114，run_simo_config_list`

这里把原先分成两套、且包含临时 `disable_cuda_graph` 的字符串拼接统一为一套
`model_args`。v0.5.18 的 `lm-eval` SGLang adapter 最终以 Python kwargs 构造
`ServerArgs`；`cuda_graph_max_bs` 只是 CLI 兼容别名，不是可直接传给 dataclass
构造函数的字段，所以改用当前字段 `cuda_graph_max_bs_decode`。这样设置 decode
graph 的最大 batch size，同时不把 graph 本身关闭，也避免 Engine 因未知 key 报错。

### 5.3.3 只对需要的分支设置 attention backend

文件：`.../llm_eval_online_quant.sh:107-108，run_simo_config_list`

当 `attention_backend` 参数非空时才写入 JSON。调用方
`run_model_evaluations`（`:141-154，run_model_evaluations`）为普通权重量化传空值，
所以 `SIMOLinearMethod`/`SIMOFusedMoEMethod` 测试继续使用 SGLang 的默认 attention
选择；调用方 `run_model_evaluations_kv_cache_quant`（`:157-167，
run_model_evaluations_kv_cache_quant`）才传 `triton_simo`，从而让打包的 SIMO KV
buffer 由自定义读 kernel 消费。这个条件分流避免权重-only 测试无故承担自定义 KV
backend 的限制。

### 5.3.4 仅 KV 量化关闭 chunked-prefix 路径

文件：`.../llm_eval_online_quant.sh:109-113，run_simo_config_list`

只有 `attention_backend == "triton_simo"` 时才追加
`disable_chunked_prefix_cache=true`。SIMO 的量化 attention kernel 返回 attention
output，但没有 SGLang chunked-prefix 合并所需的 LSE；MLA pool 也不是上游 raw
latent/rope buffer。将开关限制在 KV 量化分支可以绕过这条不兼容路径，同时保留普通
prefix cache 和权重量化的默认行为。

### 5.3.5 无量化基线的 graph 字段

文件：`.../llm_eval_online_quant.sh:123-137，run_no_quant_eval`

无量化 baseline 的 `model_args` 同样把旧的 `cuda_graph_max_bs` 替换为
`cuda_graph_max_bs_decode`（`:131`）。这样 baseline 与量化测试使用同一 release
字段，比较结果不会因为参数名过时而在启动阶段失败；此处没有加入
`disable_cuda_graph`，基础 CUDA graph 仍由 SGLang 自己决定。

### 5.3.6 完整矩阵注释

文件：`.../llm_eval_online_quant.sh:147-149，run_model_evaluations`

这处 diff 更新了注释并明确 `run_no_quant_eval` 先运行 baseline、再运行整个
`QUANT_CONFIGS`。函数调用本身在父版本已经存在，所以该 hunk 的运行时行为没有另行
改变；它是对当前完整测试矩阵意图的准确标注。数组内容和 KV 调用仍位于父版本的
`:25-50` 与 `:157-167`，不应把它们误记为本 commit 新增。

## 5.4 decode kernel 导入路径

### 5.4.1 `_fwd_kernel_stage2` 导入

文件：`simo/extensions/sglang_simo/layers/attention/triton_ops/decode_attention.py:6-8，模块导入`

v0.5.18 删除/移动了旧的
`sglang.srt.layers.attention.triton_ops.decode_attention` 模块，公共 Triton decode
stage-2 kernel 现在位于 `sglang.kernels.ops.attention.decode_attention`。只改导入路径，
不改 SIMO 自己的量化解包和 attention 计算逻辑，因此仍复用同一个上游 stage-2 kernel
实现而不会在插件导入时失败。

## 5.5 SIMO Triton attention backend

### 5.5.1 `SIMOTritonAttnBackend::forward_extend` 新参数

文件：`simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py:124-135，
SIMOTritonAttnBackend::forward_extend`

release 基类 `TritonAttnBackend::forward_extend` 新增 `score_mod` 和 `aux_tensors`。
子类若仍使用旧签名，RadixAttention 以关键字调用时会得到 `TypeError`。因此在
`:133-134` 补齐两个参数；普通（非 SIMO KV）层在 `:136-147` 将它们原样转发给
`super()`，不改变 SGLang 原有功能。

### 5.5.2 量化 extend 的能力边界

文件：`.../triton_simo_backend.py:149-164，SIMOTritonAttnBackend::forward_extend`

SIMO 自定义 extend kernel 没有实现 `score_mod`/`aux_tensors`，所以在量化层收到这类
输入时显式抛出 `NotImplementedError`，避免静默忽略相对位置 bias 等张量而得到错误
结果。`:154-164` 对 `forward_batch.mha_return_lse` 也做显式拒绝：chunked-prefix
路径会期待 `(output, LSE)`，而当前 SIMO kernel 只产生 output；明确报错比把单个 tensor
误当成二元返回值更安全，调用方应设置 `disable_chunked_prefix_cache=True`。

### 5.5.3 `SIMOTritonAttnBackend::forward_decode` 签名与转发

文件：`.../triton_simo_backend.py:246-274，SIMOTritonAttnBackend::forward_decode`

decode 入口同样补上 `score_mod`/`aux_tensors`（`:255-256`）。非量化层通过
`:258-269` 转发到 release 基类；量化层在 `:271-274` 显式拒绝尚未实现的扩展参数。
这样同一个 `triton_simo` backend 在普通层和量化层之间都遵守 release 的调用协议，
且不伪装支持未实现的 score modification。

### 5.5.4 MLA decode 的位置包装

文件：`.../triton_simo_backend.py:288-297，SIMOTritonAttnBackend::forward_decode`

旧代码在 SIMO MLA 分支直接传 `forward_batch.out_cache_loc`。release 的写入接口接受
`KVWriteLoc`，其中可能同时携带 SWA location 和统一内存下的 physical `full_loc`。
`:291` 使用已有的 `_make_kv_write_loc`，`:292-296` 把包装对象交给 pool，使统一内存
的 physical 地址在写入前保持正确。`:288` 的嵌套 `getattr` 也让旧/新 backend 对象在
缺少 `is_simo_mla_quantized` 属性时安全落到 `False`，避免默认参数表达式提前访问
不存在的属性。

## 5.6 KV pool configurator patch

### 5.6.1 模块说明和量化参数提取

文件：`simo/extensions/sglang_simo/mem_cache/init_memory_pool_patch.py:1-6，模块文档`

文档字符串从旧的 `init_memory_pool` 改为 `_build_token_to_kv_pool`，准确描述 patch
所在的 release 生命周期；否则维护者会误以为仍在 patch 已删除的 ModelRunner mixin。

文件：`.../init_memory_pool_patch.py:13，模块导入`。

新增 `get_parallel`，供新的 configurator 计算 attention TP size。

文件：`.../init_memory_pool_patch.py:18-40，_extract_quant_params_from_model`

新入口的 `self` 是 `KVCacheConfigurator`，在某些初始化阶段可能尚未挂载 `model`。
`:20-22` 用 `getattr(..., None)` 并在缺失时返回 `None`，让 wrapper 回退到 SGLang
标准 pool，而不是在探测阶段因 `AttributeError` 阻断整个服务。找到带有
`kv_cache_quant_spec` 的 attention layer 后，`:26-38` 仍提取 SIMO kernel、packed/scale
尺寸及 MLA rope 尺寸，保持原有量化布局计算。

### 5.6.2 临时替换目标模块

文件：`.../init_memory_pool_patch.py:43-53，_temporarily_replace_simo_kv_pool_cls`

旧目标 `sglang.srt.model_executor.model_runner_kv_cache_mixin` 在 v0.5.18 已不存在。
`:46` 改为导入 `sglang.srt.mem_cache.kv_cache_configurator`，并在 `:52-53` 保存新模块
中的 `MHATokenToKVPool`/`MLATokenToKVPool`。`:134` 与 `:171` 只在调用原始
configurator 期间替换类，`:173-177` 的 `finally` 恢复原类，避免 patch 泄漏到后续
非量化 pool 或其它线程。

### 5.6.3 MHA adapter 的新构造参数

文件：`.../init_memory_pool_patch.py:55-131，SIMOMHATokenToKVPoolAdapter::__init__`

release 的 MHA 构造函数新增 `quant_method` 和 `allocation_label` 参数，分别位于
`:77`、`:79`。SIMO 的 packed byte pool 不兼容 SGLang 的 FP4 `quant_method` 或其它
带标签的独立 allocation，因此 `:82-90` 对非空值显式报错，而不是忽略参数后分配错误
布局。`:91-97` 继续限制 NHD layout 和 post-capture；`:99-106` 保留上游 SWA 参数的
维度优先级；`:108-131` 将 release 传入的尺寸和 SIMO 量化元数据组合后调用父类，
使新入口仍能创建正确的 MHA pool。

### 5.6.4 MLA adapter

文件：`.../init_memory_pool_patch.py:135-169，SIMOMLATokenToKVPoolAdapter::__init__`

MLA pool 的符号也从新 configurator 模块替换（`:171`）。adapter 保持 v0.5.18 的
`start_layer`、`end_layer`、`use_dsa`、`override_kv_cache_dim` 参数（`:138-152`），
并在 `:153-169` 追加 SIMO 的量化 spec/downcast kernel。这样 DeepSeek-V2 Lite 的
新 `_build_mla_kv_pool` 调用可以匹配，同时仍由 SIMO 类计算 byte buffer 大小。

### 5.6.5 新的 `_build_token_to_kv_pool` wrapper 和 fallback

文件：`.../init_memory_pool_patch.py:180-188，_patched__build_token_to_kv_pool`

函数名和 docstring 改为 release 实际调用的
`KVCacheConfigurator::_build_token_to_kv_pool`。若 `:182-183` 找不到量化 attention
层，`:188` 将所有参数原样传给 original function，保证普通模型继续使用标准 pool。

### 5.6.6 不支持的 pool/layout 保护

文件：`.../init_memory_pool_patch.py:190-217，_patched__build_token_to_kv_pool`

这些检查把本次明确不支持的 release feature 变成启动时的可读错误：

- `:193-196` 拒绝 DSA 和 DeepSeek-V4 的专用 pool layout；
- `:198-205` 拒绝 hybrid SWA 与 Mamba/linear pool，避免 SIMO flat NHD reader 读取非
  SIMO buffer；
- `:207-213` 拒绝 DCP size 大于 1，因为当前写 kernel 没有 decode context parallel
  的 location/mask 处理；
- `:214-217` 拒绝 page-major 和 unified memory，这两者会改变物理地址或 page 组织，
  而 SIMO kernel 只按普通 NHD byte rows 索引；
- `:240-241` 拒绝 post-capture KV sizing，SIMO pool 的固定 packed buffer 不能在
  capture 后再动态 backing。

显式 `NotImplementedError`/`ValueError` 是有意的：目标只覆盖 Llama3.1 GQA 与
DeepSeek-V2 Lite MLA，不能让其它 layout 静默落入错误的量化解释。

### 5.6.7 attention backend 约束

文件：`.../init_memory_pool_patch.py:219-239，_patched__build_token_to_kv_pool`

`:222-229` 收集普通、prefill、decode 三个 backend 字段，并尝试读取 release 已解析
的 backend pair；`:230-239` 只允许 `triton` 和 `triton_simo`。FA/FlashInfer 等
backend 不认识 SIMO 的 uint8+scale buffer，若继续创建 pool 会在后续 kernel 中把字节
当作普通 dtype。提前拒绝能把配置错误定位在 pool 构造阶段，而不是产生难以诊断的
CUDA 结果。`:243-245` 在所有检查通过后才临时替换类并调用原始构造函数。

### 5.6.8 `DefaultPoolConfigurator::_compute_cell_size`

文件：`.../init_memory_pool_patch.py:248-279，_compute_cell_size`

v0.5.18 的基类签名是 `DefaultPoolConfigurator::_compute_cell_size(self, kvc,
num_layers)`，其中第二个对象是 `KVCacheConfigurator` 而不是旧版 ModelRunner。
因此 `:248-256` 全部改用 `kvc`，无量化或无法提取量化参数时也把同一个 `kvc` 透传
给 original。`:260` 用 `get_parallel().attn_tp_size` 替代旧的
`get_attention_tp_size()`，适配 release 的 runtime context API。`:261-279` 对 MLA/MHA
分别按照 packed data、scale bytes、KV head 数和 layer 数计算每 token 成本；FP4 或
DSA 仍回退到上游计算，避免覆盖 release 已有的专用公式。

### 5.6.9 安装函数与兼容别名

文件：`.../init_memory_pool_patch.py:282-302，apply__build_token_to_kv_pool_patch`

`:284-291` 将 patch 安装在 `KVCacheConfigurator::_build_token_to_kv_pool`，
`:295-301` 将 cell-size patch 安装在 `DefaultPoolConfigurator::_compute_cell_size`，
并保留 `call_original=True`，使 wrapper 只负责 SIMO 情况、其它情况继续走 SGLang。

文件：`.../init_memory_pool_patch.py:305-307，模块级兼容别名`

`_patched_init_memory_pool` 和 `apply_init_memory_pool_patch` 指向新实现，兼容仍引用
旧函数名的外部脚本；真正注册入口已在 `sglang_simo/__init__.py:22-27，
register_simo_extensions` 切换到新名称。

## 5.7 SIMO memory pool

### 5.7.1 MHA `set_kv_buffer` 的 `KVWriteLoc`

文件：`simo/extensions/sglang_simo/mem_cache/memory_pool.py:249-279，
SIMOMHATokenToKVPool::set_kv_buffer`

release 调用方现在可能传入 `KVWriteLoc` 而非裸 location。`:265` 使用
`unwrap_write_loc` 解出 generic、SWA 和 unified `full_loc`；`:266-267` 对只有 SWA
子池地址的情况显式拒绝，因为 SIMO 没有 SWA 专用 buffer；`:268-269` 在 unified
memory 情况使用预先解析好的 physical `full_loc`。其余量化写入仍在 `:274-278`
调用 `simo_set_kv_buffer`，因此只改变地址协议，不改变 mxfp/mxint/per-group 的量化
格式。

### 5.7.2 MLA packed buffer 的 storage dtype

文件：`.../memory_pool.py:327-350，SIMOMLATokenToKVPool::_create_buffers`

父类 `MLATokenToKVPool::__init__` 会动态调用子类 `_create_buffers`，并通常按计算
dtype（例如 BF16）设置 `store_dtype`。SIMO buffer 的真实布局是 packed payload 加
scale bytes，必须逐字节存储；`:329-331` 在分配前强制 `self.store_dtype = torch.uint8`，
`:343-349` 随后分配 uint8 buffer，避免父类的 BF16 view 造成容量和 stride 错误。

### 5.7.3 禁止 raw MLA prefix reader

文件：`.../memory_pool.py:358-378，SIMOMLATokenToKVPool::get_mla_kv_buffer`

release 的 chunked-prefix helper 会调用这个方法并期待未量化的 latent/rope 张量。
SIMO 保存的是 packed uint8 与 tile scales，直接复用父类读取会把字节误解释成 BF16。
因此新方法在 `:374-378` 直接抛出 `NotImplementedError`，并提示关闭
`disable_chunked_prefix_cache`；正常 `triton_simo` attention 仍直接从 packed buffer
解量化，不会经过这个 raw reader。

### 5.7.4 MLA `set_kv_buffer` 的位置解包

文件：`.../memory_pool.py:380-403，SIMOMLATokenToKVPool::set_kv_buffer`

与 MHA 相同，`:393-397` 支持 release `KVWriteLoc` 和 unified physical `full_loc`，
拒绝 SIMO 不实现的 SWA-only location；`:400-402` 再把 combined K 拆成 nope/rope，
调用 SIMO 自己的 `set_mla_kv_buffer`。这使 v0.5.18 的普通 MLA 写入调用和旧的裸
tensor 调用都能进入同一量化 kernel。

### 5.7.5 MLA 专用写入位置解包

文件：`.../memory_pool.py:405-419，SIMOMLATokenToKVPool::set_mla_kv_buffer`

release 也可能直接以 `KVWriteLoc` 调用 MLA 专用入口。`:415-419` 再次解包并检查
SWA/full location，保证该入口无论由 `set_kv_buffer` 还是 attention backend 直接调用，
都不会把 dataclass 当作 tensor 索引；随后原有 `concat_and_cache_mla_kernel` 从 `:421`
之后继续使用解析后的 `loc` 写入 packed buffer。

## 5.8 线性权重 loader 兼容层

### 5.8.1 `_call_weight_loader`

文件：`simo/extensions/sglang_simo/quantization/quantization.py:3，模块导入`

新增 `inspect`，用于在运行时判断 release loader 是否声明
`loaded_shard_id`，而不依赖类名猜测签名。

文件：`.../quantization.py:143-171，_call_weight_loader`

v0.5.18 普通 Column/Row 的 `weight_loader_v2` 通常是两参数
`(param, loaded_weight)`，而 merged/QKV loader 仍可能接受 `loaded_shard_id`。SIMO
包装器同时服务这两类原始 loader，因此：

- `:156-157` 没有 shard id 时保持最短的两参数调用；
- `:159-162` 对无法反射的扩展 callable 安全回退；
- `:164-167` 只有签名明确声明该参数或接受 `**kwargs` 时才转发 shard id；
- `:169-171` 对普通 v2 loader 不传多余关键字，让其自行完成 TP slicing。

这样既保留 merged/QKV 的分片语义，又避免普通 release loader 因意外收到
`loaded_shard_id` 而报 `TypeError`。

### 5.8.2 `SIMOLinearMethod::get_weight_loader`

文件：`.../quantization.py:973-1043，SIMOLinearMethod::get_weight_loader` 内部的
`online_weight_loader`

`:974-979` 保持同时接受显式 `loaded_shard_id` 和旧调用方的 `**kwargs`；`:980-981`
从 `kwargs` 兼容读取 `loaded_shard_id`/`shard_id`。随后三类原始参数写入全部改走
`:991`、`:1019`、`:1030-1033`、`:1035-1041` 的 `_call_weight_loader`：

1. prequantized packed 权重直接加载时，按 loader 签名决定是否传 shard id；
2. float-source 权重经过 `weight_downcast_kernel` 后，packed weight 使用同一规则；
3. `weight_scale` 和 NVFP4 `weight_global_scale` 也使用同一规则，避免只修权重而在
   scale 参数上再次触发签名错误。

这处修改不改变 SIMO 的量化数值、global scale 或 checkpoint format 判定，只修正
release v0.5.18 weight-loader-v2 接口变化。

## 5.9 验证与边界

本次 commit 前后执行了以下检查：

- pre-commit 的 Python AST、冲突、ruff、ruff-format、末尾换行检查全部通过；
- `git diff --cached --check` 通过；
- 修改的 SIMO Python 文件 `python -m compileall` 通过；
- `bash -n simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh`
  通过；
- 在真实 v0.5.18 模块环境设置 `SIMO_TEST_USE_REAL_SGLANG=1` 后，
  `tests/sglang_simo/test_prequantized_checkpoint_loading.py` 的 8 个测试全部通过；
- 此前的真实 Llama3.1-8B weight-only 与 `kvquant_mxfp8` 单请求 smoke 均完成 decode
  CUDA graph capture/replay，日志显示 `cuda graph: True`。

默认 fake-SGLang 测试收集器仍引用旧的模拟模块布局，因此不带
`SIMO_TEST_USE_REAL_SGLANG=1` 的该测试文件会在收集阶段找不到
`sglang.srt.layers.quantization.kv_cache`；这属于测试夹具与当前 release 布局不一致，
不是本 commit 的运行时失败。此次没有运行完整 42 项 lm-eval 矩阵，完整精度矩阵仍需
在目标 GPU 和模型权重环境中执行。

## 5.10 `_call_weight_loader` 是否真的有必要

### 5.10.1 结论

这个函数不是 SIMO 量化算法本身的一部分，而是权重加载接口的兼容层。结论分两层：

1. **对 v0.5.18 的通用 SIMO 适配，建议保留。** 原来的
   `original_weight_loader(param, packed_weight, **kwargs)` 只在底层 loader 接受
   `loaded_shard_id` 时成立；v0.5.18 同时存在两参数和三参数 loader。只要一个普通
   `Column/Row/Replicated` 层被带着非空 shard id 调用，原写法就会直接抛
   `TypeError: got an unexpected keyword argument 'loaded_shard_id'`。
2. **对本次限定的 Llama3.1-8B-Instruct + DeepSeek-V2-Lite-Chat、TP=1、默认权重加载路径，
   它不是每次都必然被用到。** 这两个模型当前的带 shard id 映射主要落在 fused
   `MergedColumn`/`QKV` 层，普通层通常以两个参数调用，所以删除它可能仍能完成这组测试。
   但这属于当前调用图的偶然保证，不是旧写法满足 v0.5.18 合同的证明；打开新的权重加载器、
   改变模型映射或引入另一种 fused/非 fused 组合后，旧写法会重新暴露问题。

### 5.10.2 原写法为什么不总是够用

文件：`simo/extensions/sglang_simo/quantization/quantization.py:973-1043，SIMOLinearMethod::get_weight_loader`

`:974-979` 的 `online_weight_loader` 同时声明 `loaded_shard_id` 和 `**kwargs`，因为上游
调用方既可能用第三个位置参数，也可能使用 `shard_id=`/`loaded_shard_id=` 关键字。commit
`4da2709^` 的同一文件 `:954` 先把非空 id 组装成 `{"loaded_shard_id": id}`，再在权重、scale、global scale 三处
直接执行：

```python
original_weight_loader(param, packed_weight, **kwargs)
```

当 `kwargs` 为空时这等价于两参数调用，没有问题；当 `kwargs` 非空时，Python 会在进入
loader 函数体之前按关键字匹配形参。若底层函数只有
`(param, loaded_weight)`，调用立即失败，loader 根本没有机会自行完成 TP slicing。
因此不能简单地“始终传 id”，也不能简单地“始终不传 id”：后者会破坏 fused QKV/gate-up
权重的分片定位。

### 5.10.3 v0.5.18 中确实存在两套签名

以下均为 `sglang_kernel_src` code base 的相对路径：

- `python/sglang/srt/layers/linear.py:272-294，ReplicatedLinear::weight_loader`：只有
  `(param, loaded_weight)`。
- `python/sglang/srt/layers/linear.py:465-490，ColumnParallelLinear::weight_loader_v2`：
  只有两个业务参数；TP rank 和 slicing 由函数内部传给 parameter object。
- `python/sglang/srt/layers/linear.py:1574-1602，RowParallelLinear::weight_loader_v2`：
  同样只有两个业务参数。
- `python/sglang/srt/layers/linear.py:863-941，MergedColumnParallelLinear::weight_loader_v2`：
  额外接受 `loaded_shard_id`，并在 `:915-941` 用它计算 fused 分片偏移。
- `python/sglang/srt/layers/linear.py:1139-1191，QKVParallelLinear::weight_loader_v2`：
  额外接受 `loaded_shard_id`；`:1171` 还要求其为 `q/k/v`，随后在 `:1183-1191` 依据它
  选择对应 Q/K/V 的偏移和 TP rank。

这也解释了为什么把 id 丢掉并不是安全的“简化”：对于 Merged/QKV，id 是数据应该落入
哪一个逻辑 shard 的必要信息；对于普通 Column/Row，id 是不被声明的多余信息。

### 5.10.4 shard id 是怎样进入 SIMO 包装器的

- `python/sglang/srt/model_loader/auto_loader.py:86-108，StackedParamsDispatch::try_load`：
  `:98-107` 从 checkpoint 名称得到 `shard_id`，再调用
  `param.weight_loader(param, tensor, shard_id)`。
- `python/sglang/srt/models/utils.py:187-207，AutoWeightsLoader::_load_param`：
  `:205-206` 只以 `(param, weight_data)` 两个参数调用。也就是说，打开 v2 loader
  本身不会自动产生 shard id；真正需要兼容第三参数的是仍使用 stacked dispatch 的模型路径。
- `python/sglang/srt/models/llama.py:829-900，LlamaForCausalLM::_legacy_load_weights`：
  `:830-836` 把 `q_proj/k_proj/v_proj` 映射到 `qkv_proj`，把 `gate_proj/up_proj` 映射到
  `gate_up_proj`；`:873-884` 将对应 id 作为第三个参数传入。
- `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:289-318，
  DeepseekV2WeightLoaderMixin::do_load_weights`：`:289-318` 对 `gate_up_proj` 也把
  `shard_id` 传给参数 loader；而普通 MLA 投影在 `:447-457` 以两参数调用。

SIMO 在 `simo/extensions/sglang_simo/quantization/quantization.py:903-940，
SIMOLinearMethod::__init__` 将 `SIMOLinearMethod` 加入 `WEIGHT_LOADER_V2_SUPPORTED`，所以
上面这些 SGLang 线性类会把其 v2 loader 交给
`simo/extensions/sglang_simo/quantization/quantization.py:973-1043，
SIMOLinearMethod::get_weight_loader` 包装。包装器因此必须同时处理两种签名。

### 5.10.5 `_call_weight_loader` 做了什么

文件：`simo/extensions/sglang_simo/quantization/quantization.py:143-171，_call_weight_loader`

- `:156-157` 在 id 为空时保持两参数调用，兼容普通参数和 fused-in-checkpoint 的整块权重。
- `:159-167` 反射底层 callable 的签名；只有声明了 `loaded_shard_id`，或明确接受
  `**kwargs` 时，才把 id 作为关键字转发。
- `:169-171` 对普通两参数 loader 不转发 id，让其按自身的 TP 逻辑处理输入。
- `:991`、`:1019`、`:1031-1041` 让 packed weight、`weight_scale`、NVFP4 global scale
  使用同一规则，避免只修主权重而在 scale 加载时再次失败。

用“捕获 `TypeError` 后重试两参数”也能实现类似效果，但会把 loader 内部真正的
`TypeError` 误判为签名不兼容并重复加载；当前的签名判断避免了这种副作用。反射只发生在
模型启动加载阶段，不在推理热路径上，性能影响可以忽略。

### 5.10.6 对本次两个模型的实际判断与建议

`python/sglang/srt/environ.py:293，环境变量 SGLANG_ENABLE_WEIGHT_LOADER_V2` 的默认值是
`False`。本次 `llm_eval_online_quant.sh` 没有打开它；在默认 legacy 路径中，Llama 的
第三参数映射落到 QKV/Merged 层，DeepSeek 的第三参数映射落到 `gate_up_proj`，普通层走
两参数路径。因此在严格限定的两模型测试中，旧写法并非严格必需。此前的
`temp/llm_eval_online_quant.sh...TASKS_mmlu...2026_09_04___15_43_28` 和
`...TASKS_gsm8k...2026_09_04___15_40_59` 日志确实完成了评测且没有出现
`loaded_shard_id` 的 `TypeError`，但不能据此证明测试使用的是 commit 前的 direct-call
实现：日志中已经出现由工作树未提交改动产生的 SIMO patch 日志，而 commit `4da2709` 是之后
才创建的。因此这些日志只能证明整体调用流程成功，不能作为删除 helper 的严格回归证明。

但 `SIMOLinearMethod` 是通用 quant method，不能把这个调用图假设写成全局接口保证。建议
保留 `_call_weight_loader`，因为它改动小、只影响启动加载、不会改变量化数值，而且覆盖了：

- 与 `SGLANG_ENABLE_WEIGHT_LOADER_V2=true` 引入的模型新 loader 路径共存，并覆盖其中可能
  继续使用 stacked shard dispatch 的模型；
- 某个模型把 shard id 传给普通 Column/Row 参数的路径；
- 后续 release 对 loader 分层或参数类型的扩展。

`_call_weight_loader` 只包住 `SIMOLinearMethod::get_weight_loader`；
`simo/extensions/sglang_simo/quantization/quantization.py:1411-1508，
SIMOFusedMoEMethod::get_moe_weight_loader` 仍使用独立的五参数
`original_weight_loader` 协议。因此不能把这个 helper 当成 SIMO MoE loader 的修复。

如果目标是进一步收紧代码，优先为 `_call_weight_loader` 增加两个单元测试：一个用两参数
loader 验证非空 id 被丢弃，一个用 Merged/QKV 风格三参数 loader 验证 id 被保留；不建议直接
删除该函数。当前实现唯一需要留意的边界是 `inspect.signature` 无法反射的 opaque callable
会静默走两参数 fallback；在本次两个模型的 Python bound-method loader 中不会触发，但若以后
接入 C/C++ callable，最好改为显式报错而不是静默丢弃 shard id。

### 5.10.7 对“原来的 `kwargs` 已经兼容 2/3 参数”的澄清

这个判断有一半是对的：**对本次两个模型实际会遇到的 loader，父提交中的写法确实能够工作**；
但它不是 Python 意义上的“自动按参数个数兼容”。关键在于 `**kwargs` 展开后传的是一个
**关键字参数**，而不是无条件的第三个位置参数。

文件：`simo/extensions/sglang_simo/quantization/quantization.py:973-1043，SIMOLinearMethod::get_weight_loader`
（父提交 `4da2709^` 的对应代码为 `:941-1003`）。父提交的逻辑是：

```python
kwargs = {"loaded_shard_id": loaded_shard_id} \
    if loaded_shard_id is not None else {}
original_weight_loader(param, packed_weight, **kwargs)
```

它在 Python 中分别等价于：

```python
# loaded_shard_id is None
original_weight_loader(param, packed_weight)

# loaded_shard_id == "q"
original_weight_loader(param, packed_weight, loaded_shard_id="q")
```

第二种调用只有在底层函数声明了名字为 `loaded_shard_id` 的形参，或声明了 `**kwargs` 时才会
成功。它**不**等价于下面这个位置参数调用：

```python
original_weight_loader(param, packed_weight, "q")
```

因此，底层签名的行为是：

| 底层签名 | `kwargs={}` | `kwargs={"loaded_shard_id": "q"}` |
|---|---:|---:|
| `f(param, weight)` | 成功 | `TypeError: unexpected keyword argument` |
| `f(param, weight, loaded_shard_id=None)` | 成功 | 成功 |
| `f(param, weight, shard_id=None)` | 成功 | `TypeError: unexpected keyword argument` |
| `f(param, weight, **kwargs)` | 成功 | 成功 |

也就是说，父提交的代码根据“id 是否为空”选择两参数或带关键字的调用，却没有根据
`original_weight_loader` 的真实签名选择调用方式。若某一条路径把非空 id 送到只有两个形参的
普通 loader，即使 Python 代码写了 `**kwargs`，也不会被忽略，函数体甚至不会开始执行。把 id
改成第三个位置参数同样不能修复两参数 loader，只会变成 `too many positional arguments`。

### 5.10.8 结合 v0.5.18 签名后的准确结论

文件（相对于 `sglang_kernel_src` code base）：`python/sglang/srt/layers/linear.py`：

- `:465-490，ColumnParallelLinear::weight_loader_v2` 和 `:1574-1602，RowParallelLinear::weight_loader_v2`
  只有 `(param, loaded_weight)`；非空 id 传入父提交的 `**kwargs` 会失败。
- `:863-941，MergedColumnParallelLinear::weight_loader_v2` 和 `:1139-1191，QKVParallelLinear::weight_loader_v2`
  显式声明 `loaded_shard_id`；父提交的关键字传递完全正确，而且该 id 会参与 fused shard
  的偏移计算。

文件（相对于 `sglang_kernel_src` code base）：`python/sglang/srt/models/llama.py:829-900，LlamaForCausalLM::_legacy_load_weights`。
`:830-836` 将 q/k/v、gate/up 映射到 fused 参数，`:873-884` 只对这些 fused 参数传第三个
`shard_id`；普通 Column/Row 参数在 `:898` 以两个参数调用。

文件（相对于 `sglang_kernel_src` code base）：`python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:289-318，DeepseekV2WeightLoaderMixin::do_load_weights`。
该路径只对 gate/up 的 fused 参数携带 shard id，普通 MLA 投影仍是两参数调用。因此对于用户
限定的 Llama3.1-8B-Instruct、DeepSeek-V2-Lite、当前 legacy loader 和 TP=1，父提交的
`kwargs` 写法的“id 与 loader 类型恰好匹配”，所以**原始写法足够，`_call_weight_loader` 不是
这组测试严格必需的修复**。

文件：`simo/extensions/sglang_simo/quantization/quantization.py:143-171，_call_weight_loader`。
该函数的额外价值仅在于处理“非空 id 意外到达两参数 loader”的情况：`:159-171` 先检查
签名，只有发现 `loaded_shard_id` 或 `**kwargs` 才转发，否则主动省略 id。故两种写法在当前
目标调用图上结果相同，但在更换模型映射、启用另一套 loader 或未来 release 改变调用图时，
`_call_weight_loader` 更稳妥；若目标始终严格限定这两个模型，也可以删除它并恢复父提交写法，
不会改变量化算法本身。

还要注意：`_call_weight_loader` 并非“所有三参数函数的通用适配器”。它和父提交一样只识别
名为 `loaded_shard_id` 的关键字（另加 `**kwargs`）；如果未来出现第三参数名为 `shard_id` 或
仅允许位置传参的 callable，两者都需要专门适配。当前 v0.5.18 的 Merged/QKV 签名正好使用
`loaded_shard_id`，所以不存在这个边界问题。

## 5.11 pip wheel 环境删除 `_call_weight_loader` 后的复测

### 5.11.1 输入、版本和提取规则

本次复测工作树是 `simo_conda_sglang_pip` 的 `sglv0518` 分支，SGLang 通过 wheel 安装，
Simo 也通过 wheel 安装。使用的两个日志（路径相对于该工作树）是：

```text
temp/llm_eval_online_quant.sh.MAX_RUNNING_REQUESTS_128_CUDA_GRAPH_MAX_BS_128_ADD_BOS_TOKEN_true__TASKS_mmlu__CUDA_VISIBLE_DEVICES_6.log.2026_09_07___18_15_43
temp/llm_eval_online_quant.sh.MAX_RUNNING_REQUESTS_128_CUDA_GRAPH_MAX_BS_128_ADD_BOS_TOKEN_true__TASKS_gsm8k__CUDA_VISIBLE_DEVICES_7.log.2026_09_07___18_16_02
```

脚本中的 `QUANT_CONFIGS` 有 13 个权重量化 JSON，`KV_CACHE_QUANT_CONFIGS` 有 7 个 KV
cache 量化 JSON；每个模型还运行一次 no-quant baseline。因此每个任务应有
`2 * (1 + 13 + 7) = 42` 个评测实例。

MMLU 取每个实例结果表中任务名为 `mmlu` 的总 `acc` 行。lm-eval 会在详细结果表和
`Groups` 表各打印一次，所以日志中有 84 行总行，但两行属于同一个实例，按实例去重后为
42 个分数。例如首个 baseline 在 MMLU 日志 `:1247`，最后一个 KV 配置在 `:59999`。

GSM8K 只取 `flexible-extract` 的 `exact_match`，忽略同一表中的 `strict-match`。首个
baseline 位于 GSM8K 日志 `:732`；DeepSeek 的两个有差异的结果位于 `:26055` 和 `:27059`，
最后一个 KV 配置位于 `:37591`。

pip 工作树原本没有 `tests/sglang_simo/references_accuracy/mmlu-v0.5.18.yaml` 和
`gsm8k-v0.5.18.yaml`；本次以同目录现有的 `mmlu.yaml`/`gsm8k.yaml` 作为字段顺序，并与
同级 `simo_conda_sglang` 工作树中的 v0.5.18 基准逐字节核对，二者内容一致。恢复结果写入：

- `tests/sglang_simo/references_accuracy/mmlu-v0.5.18.recover.yaml`
- `tests/sglang_simo/references_accuracy/gsm8k-v0.5.18.recover.yaml`

两个文件均保留了基准的模型、量化类型和字段顺序；每个模型各有 21 条记录。

### 5.11.2 运行完整性和日志异常

两个任务都是 42/42 个实例出现了对应的总分行；也就是说，两个模型的 13 个权重量化和
7 个 KV cache JSON 都实际执行并产生了结果，没有在中途因 loader 或 CUDA 错误停止。

日志中确实有 traceback，但它们是每个实例结束后的进程清理噪声，不是模型推理异常：

```text
resource_tracker: process died unexpectedly, relaunching
Traceback (most recent call last):
  .../multiprocessing/resource_tracker.py:264, in main
    cache[rtype].remove(name)
KeyError: '/loky-...'
```

该片段总是在 `kill_process_tree called` 之后出现；MMLU 和 GSM8K 日志各出现 42 次，之后
下一个配置继续运行且最终 42 个结果齐全。因此应记录为 cleanup 阶段的已知警告/异常，不能
把它等同于 core dump。日志中没有发现 `core dumped`、`SIGSEGV`、`SIGABRT`、
`Segmentation fault`、OOM、`Killed` 或 `fatal error`。另外，每个实例都有一次可选模型
`sarashina2_vision` 的 `Ignore import error`（缺少 `MultimodalDataItem`）；该模块与本次
Llama/DeepSeek 目标模型无关，SGLang 明确将其忽略。

### 5.11.3 MMLU 总分提取结果和对比

下面的权重量化顺序严格对应 recover YAML：

```text
w8a8_fp8_per_block, w4a16_int4_per_group, w8a8_int8_per_block,
w8a8_fp8_per_channel, w8a8_int8_per_channel, w8a8_mxint, w8a8_mxfp,
w6a6_mxfp, w4a4_mxfp, w4a16_nvfp4_per_group,
w4a16_nvfp4_per_group_4_over_6, w4a4_nvfp, w4a4_nvfp_4_over_6
```

```text
Llama-3.1-8B-Instruct
baseline: 68.47
weight:   68.02, 66.21, 67.98, 68.02, 67.48, 68.20, 67.67,
          68.07, 57.99, 66.22, 66.49, 64.13, 64.45
KV (mxfp8, mxfp4, mxfp6, mxint8, fp8_per_group_64, int8_per_group_64, nvfp4):
          68.27, 68.27, 68.27, 68.27, 68.27, 68.27, 68.27

DeepSeek-V2-Lite-Chat-16B_A2.4B
baseline: 56.65
weight:   56.38, 54.88, 56.47, 55.97, 55.88, 56.51, 55.83,
          55.86, 48.43, 54.04, 54.55, 52.83, 52.94
KV (mxfp8, mxfp4, mxfp6, mxint8, fp8_per_group_64, int8_per_group_64, nvfp4):
          56.72, 56.72, 56.72, 56.72, 56.72, 56.72, 56.72
```

将 `mmlu-v0.5.18.recover.yaml` 与 v0.5.18 基准逐模型、逐条目比较，42/42 条在两位小数
精度下完全相同。因此删除 `_call_weight_loader` 后，本次两个模型的 MMLU 总体精度没有
可见变化。

### 5.11.4 GSM8K flexible-extract 总分提取结果和对比

GSM8K 的权重量化顺序与上一节相同，下面只列 `flexible-extract` 总分（百分数）：

```text
Llama-3.1-8B-Instruct
baseline: 77.63
weight:   77.26, 73.39, 77.94, 76.88, 77.03, 77.48, 77.03,
          76.35, 47.61, 73.24, 75.51, 69.07, 70.13
KV (mxfp8, mxfp4, mxfp6, mxint8, fp8_per_group_64, int8_per_group_64, nvfp4):
          76.72, 69.90, 77.79, 77.94, 78.47, 77.33, 76.57

DeepSeek-V2-Lite-Chat-16B_A2.4B
baseline: 66.03
weight:   65.66, 56.86, 65.88, 65.88, 63.31, 65.28, 64.90,
          64.37, 38.51, 63.08, 63.91, 56.79, 57.77
KV (mxfp8, mxfp4, mxfp6, mxint8, fp8_per_group_64, int8_per_group_64, nvfp4):
          66.03, 31.39, 64.37, 66.03, 66.34, 66.34, 47.08
```

与 `gsm8k-v0.5.18.yaml` 比较，Llama 的 21/21 条全部相同；DeepSeek 的 21 条中 19 条相同，
只有两项发生变化：

| 模型/量化配置 | v0.5.18 基准 | 本次 recover | 变化 |
|---|---:|---:|---:|
| DeepSeek / `w8a8_fp8_per_block` | 65.96 | 65.66 | -0.30 个百分点 |
| DeepSeek / `w8a8_fp8_per_channel` | 65.58 | 65.88 | +0.30 个百分点 |

这两项的变化方向相反、绝对值均为 0.30 个百分点，而日志中的 GSM8K flexible-extract 标准
误差约为 1.3 个百分点，不能据此判断为系统性精度回归。低比特配置相对 baseline 的较大
差异（例如 Llama `w4a4_mxfp` 为 47.61、DeepSeek `mxfp4` KV cache 为 31.39）在基准中
也已经存在，本次不是由删除 helper 新引入的变化。

### 5.11.5 结论

本次 wheel 环境复测证明：所有目标 JSON 配置均完成 MMLU/GSM8K 评测并写入 recover YAML；
没有 core dump 或 CUDA 崩溃。删除 `_call_weight_loader` 对当前 Llama3.1-8B-Instruct 和
DeepSeek-V2-Lite、TP=1、当前加载路径的总体精度没有明显影响；MMLU 完全一致，GSM8K 仅有
两项相反方向的 0.30 个百分点波动。`resource_tracker` 的 `KeyError` 应另行清理，但不
影响本次分数提取结论。

## 6. 用新版 mma_dte 替换 vLLM-SIPU MXINT8 GEMM 的分析（2026-09-16）

### 6.1 结论与引用范围

**建议使用 `include/simma.h:180 / mma_dte`：显式指定 inputT、outputT、scalarT、layoutA、layoutB、layoutC，不带 workspace 的重载。** 对当前 vLLM-SIPU 的 MXINT8 路径，选择：

```cpp
inputT  = sifmt::mxint8
outputT = sifmt::bfloat16
scalarT = sifmt::bfloat16
layoutA = mma_dte_api::tensor_layout(32, 32, 4, 1, 1)
layoutB = mma_dte_api::tensor_layout(32, 32, 4, 1, 1)
layoutC = mma_dte_api::tensor_layout(16, 32, 1, 1, 0)
```

最后一个 `0` 使 C 直接采用 linear format。对应的生产实例化已经存在于 `kernel/instantiations/mxint8/inst_mxint8_r32_4x1.su:29 / mma_dte`，不是需要重新开发的算子能力。

本节引用约定：

- **DTE** 根目录：`/share/users/like/package/sikernel/mma_dte_tile_tensor`。未注明 VLLM 的代码引用均相对此目录；绝不拿 vLLM 内旧 submodule 的实现解释新接口。核对时 HEAD 为 `8e368c6f1dd2aa41770583571e852b87449223c1`。
- **VLLM** 根目录：`/share/users/like/package/vllm-sipu`，仅用于解释现有调用链、打包方式和将来的接入位置。
- 引用格式为 `相对路径:行号 / 函数名`；类成员使用 `类名::函数名`。GTest 的 `TEST/TEST_F/TEST_P` 是注册宏，表中单独标明测试名，不把测试名误写成普通成员函数。
- 这是源码与已有 ELF 符号的静态分析。本次没有修改算子、更新依赖、重新编译、执行 SIPU 测试或测量性能；下文“有测试”不等于“本次测试通过”。

**四类 MX 输入都能用这一组显式 layout 重载，但必须选择正确的模板实例化和 packed layout。不是 MXFP4 使用一个重载、MXINT8 使用另一个重载。** 更具体地说，支持 MXFP4 E2M1、MXFP6 E2M3/E3M2、MXFP8 E4M3/E5M2 和 MXINT8；“支持”不能扩展解释成任意低比特编码、任意输入布局或 A/B 混合类型。

### 6.2 三组重载应当如何选择

| 声明位置 / 函数 | 实际实现 | 本次选择 |
|---|---|---|
| `include/simma.h:147 / mma_dte` | `kernel/mma_dte_tiled_tensor.hpp:179 / mma_dte`；显式 layout，额外传 workspace 和字节数 | 可用，但当前 workspace 参数已不参与 TensorMap 构造，不必为了性能选它 |
| `include/simma.h:180 / mma_dte` | `kernel/mma_dte_tiled_tensor.hpp:193 / mma_dte`；显式 layout，不带 workspace | **推荐作为 vLLM 接入入口** |
| `include/simma.h:202 / mma_dte` | `template<class inputT, class outputT, int tensor_format_in=0, int tensor_format_out=1>` 的简化声明 | **当前不要选**：在本次核对的 `include/src/kernel` 中未找到定义，生产实例化也不是这一签名 |

简化版看起来最像旧的 `mma_bf16_mxi8_universal(A,B,C,M,N,K,stream)`，但不能仅根据头文件就写成 `mma_dte<sifmt::mxint8,sifmt::bfloat16,1,0>(...)` 并认为可以链接。当前找到的已有主库也没有这种 `(void*,void*,void*,int,int,int,stream)` 签名的 `mma_dte` 导出。它的默认值还是 linear 输入、tiled 输出，恰好不是本次所需的 tiled MX 输入、linear 输出。

不带 workspace 的调用链为：

```text
kernel/mma_dte_tiled_tensor.hpp:193 / mma_dte
  -> kernel/mma_dte_tiled_tensor.hpp:165 / mma_dte_implement
  -> kernel/mma_dte_tiled_tensor.hpp:151 / mma_dte_implement
  -> kernel/mma_dte_tiled_tensor.hpp:54 / mma_dte_implement_with_control
```

`kernel/detail/mma_dte_tiled_tensor_tensor_map.hpp:71 / create_tensor_maps` 在第 75 行起明确忽略 `workspace/workspaceSizeBytes/stream`，构造 host-side TensorMap，后续按值传给 kernel。`kernel/detail/mma_dte_tiled_tensor_tensor_map.hpp:38 / release_tensor_maps` 也是空操作。因此不能把两个显式重载的区别解释成“每次分配 workspace”和“复用 workspace”的性能区别。

另外，**这是外形类似 BLAS 的接口，不是已实现全部 BLAS 语义的接口**：`kernel/mma_dte_tiled_tensor.hpp:54 / mma_dte_implement_with_control` 的第 69 行仍写着 TODO，`transa/transb/alpha/beta/lda/ldb/ldc` 当前尚未用于计算。替换时按已有测试传 `OP_N, OP_N`，A 逻辑形状为 `[M,K]`，B 逻辑形状为 `[N,K]`，计算 `A * B^T`。不要因数学上有 `B^T` 就把 B 的 packed buffer 再转置，也不要期望 `OP_T` 会执行转置、`beta` 会累加原 C、`ldc` 能控制任意输出 stride。

这点有测试侧的交叉证据：`testcase/func_test/test_mxfp6e2m3_host/test_bf16_mxf6e2m3_host.cpp:152 / run_loop_k_one_identity_decode_case` 在第 172 行明确说明 B 按 `[N,K]` 存储，golden 使用 `mm_mnk` 计算 `A * B^T`。

### 6.3 当前 vLLM 输入能否保持不变

#### 6.3.1 旧调用链和输出重排的位置

| VLLM 相对路径:行号 / 函数 | 当前行为 |
|---|---|
| `tests/kernels/quantization/test_mxint8_unpack_pytest.py:77 / test_mxint8_bf16_matmul` | 量化 activation 和 weight，调用 `mxint8_bf16_matmul.sikernel`，最后裁剪至原始 m、n |
| `vllm_sipu/ops/backends/sikernel/jit/mxint8.py:120 / mxint8_bf16_matmul` | 分配 BF16 `[m,n]`，调用 C++，然后把 tiled 输出转换成 linear |
| `csrc/jit/quantization/mxint8.cpp:114 / mxint8_bf16_matmul` | 检查 packed buffer、尺寸、设备和输出 dtype |
| `csrc/jit/quantization/mxint8.cpp:62 / launch_mxint8_matmul` | 第 64 行实际调用旧 `::mma_bf16_mxi8_universal` |
| `.deps/sikernel-src/include/sikernel.h:277 / mma_bf16_mxi8_universal` | 旧接口声明，C 为 BF16，A/B 为 MXINT8 packed 数据 |
| `.deps/sikernel-src/source/source_builtin/blas/L3/mma/tile_mma_universal_bf16_mxi8_mxi8_tiled_tensor/kernel/tile_mma_universal_bf16_mxi8_mxi8_tiled_tensor.su:33 / mma_bf16_mxi8_universal` | 按 M/K 选择旧 kernel；当前 padded M>=32、K>=128 的路径在第 83 行 |

精确地说，重排不在测试文件本身，而在 **VLLM** `vllm_sipu/ops/backends/sikernel/jit/mxint8.py:80 / _tileformat_to_linear`：

```python
x.reshape(num_tile_m, num_tile_n, tile_rows, tile_cols) \
 .permute(0, 2, 1, 3) \
 .contiguous() \
 .reshape(m, n)
```

主要数据搬运来自 `.contiguous()`；前面的 reshape/permute 是构造视图，不能把整个开销简单归因于 reshape。某些退化尺寸下视图本来连续，但一般多 tile 的情形会发生实际拷贝。

#### 6.3.2 当前测试走 R32，不是 R8

**VLLM** `vllm_sipu/model_executor/layers/quantization/utils/mxint8_utils.py:100 / get_padded_mxint8_rows` 把行数至少补到 32；`:104 / get_padded_mxint8_shape` 把 K 补到 128 的倍数。当前测试模块配置为 `m=7,n=65,k=63`，因此传到 GEMM 的是：

```text
M = 32, N = 96, K = 128
A = packed MXINT8 [32,128]
B = packed MXINT8 [96,128]
C = BF16 [32,96]，随后由上层裁剪为 [7,65]
```

对应入口为 **VLLM** `tests/kernels/quantization/test_mxint8_unpack_pytest.py:77 / test_mxint8_bf16_matmul`；尺寸常量在该文件第 39 行起。不能根据原始 `m=7` 就选 DTE 的 R8 模板，因为量化时已经按 R32 的 padded shape 打包。

#### 6.3.3 packed layout 的匹配依据

**VLLM** `csrc/jit/quantization/mxint8.cpp:56 / launch_quantize_to_mxint8` 调用 `hp_to_mx<sifmt::mxint8, HPType, 0>`。其依赖的两个函数说明了输出布局：

- `.deps/sikernel-src/source/source_builtin/misc/hp_to_mx/kernel/hp_to_mx_kernel.su:28 / hp_to_mx`：第 84 行起，`dim0>16` 选择 Rows=32。
- `.deps/sikernel-src/source/source_builtin/misc/hp_to_mx/kernel/hp_to_mx_tensormap.hpp:93 / hp_to_mx_physical_extents`：Rows=32 时，当输出 K tile 数大于 2，采用 4 个列向 tile 的 supertile。当前 K>=128，MXINT8 的一个 R32 数据 tile 为 32x32，符合这一路径。

这与 DTE `kernel/instantiations/mxint8/inst_mxint8_r32_4x1.su:29 / mma_dte` 的 A/B `(32,32,4,1,1)` 一致。**第一阶段可以保留现有 MXINT8 量化、padding 和 packed buffer 契约，只替换 GEMM 与输出后处理。** 这是布局层面的源码匹配，接入后仍需用同一份 packed buffer 做新旧算子的交叉精度验证。

存储大小也吻合：**VLLM** `csrc/jit/quantization/mxint8.cpp:40 / expected_mxint8_storage_size` 采用每 1024 个 MXINT8 元素占 1088 字节；DTE `kernel/detail/mma_dte_tiled_tensor_dtype_meta.hpp:149 / get_input_dtype_info` 的 MXINT8 分支返回 1024 字节 payload 加 64 字节 header。因此输入不是普通 INT8 数组，也不是一块数据加单独的 scale tensor，而是包含 MX metadata 的 SDK packed 存储。

### 6.4 推荐调用示例及尺寸约束

以下仅是方案示例，**没有写入 vLLM 源码，也没有执行**：

```cpp
#include "simma.h"

constexpr mma_dte_api::tensor_layout a_layout{32, 32, 4, 1, 1};
constexpr mma_dte_api::tensor_layout b_layout{32, 32, 4, 1, 1};
constexpr mma_dte_api::tensor_layout c_layout{16, 32, 1, 1, 0};

// M/N/K are padded dimensions; A/B are already packed MXINT8.
::mma_dte<sifmt::mxint8, sifmt::bfloat16, sifmt::bfloat16,
          a_layout, b_layout, c_layout>(
    mma_dte_api::Operation::OP_N,
    mma_dte_api::Operation::OP_N,
    M, N, K,
    sifmt::bfloat16(1.0f), A, K,
    B, K,
    sifmt::bfloat16(0.0f), C, N,
    stream);
```

这里刻意传 `alpha=1,beta=0` 表达纯 GEMM 的意图；不要照搬部分测试里的 `0.1/0.2` 并以为当前实现真的应用了这两个系数。C 必须是符合当前实现的连续 padded `[M,N]` 存储；stream 应沿用 **VLLM** `csrc/jit/quantization/mxint8.cpp:62 / launch_mxint8_matmul` 取得的当前 SIPU stream，而不是擅自改成默认 stream。

`tensor_layout` 字段顺序见 `include/simma.h:47 / tensor_layout::tensor_layout`：

```text
(tile_dim0, tile_dim1, supertile_shape0, supertile_shape1, tensor_format)
```

其中 dim0 是连续的列/K 方向，dim1 是行方向；不是日常写矩阵 shape 时的 `(rows,cols)`。`testcase/func_test/test_mxfp8_host/common/test_mxfp8_common.hpp:48 / run_case` 的第 62 行起专门说明：SiTe 的 tiling 配置使用 `(row,column)`，而这里的 `tensor_layout` 使用 `(column,row)`。所以 `(32,32,4,1,1)` 表示沿 K 放 4 个 tile；C 的 `(16,32,1,1,0)` 表示 32 行、16 列的 BF16 输出 tile 几何以及 **linear 全局输出**。

尺寸对齐的实际检查在 `kernel/mma_dte_tiled_tensor.hpp:54 / mma_dte_implement_with_control` 第 77 行起：

```text
M % (A.tile_dim1 * A.supertile_shape1) == 0
K % (A.tile_dim0 * A.supertile_shape0) == 0
N % (B.tile_dim1 * B.supertile_shape1) == 0
K % (B.tile_dim0 * B.supertile_shape0) == 0
```

对上面的 R32 MXINT8 组合，即 `M%32==0,N%32==0,K%128==0`。直接输出 linear **不等于** 自动支持未补齐的任意 m/n/k，也不等于直接写入原始 `[7,65]` 小 buffer。

如果将来另外优化小 M，才考虑下面这些 A/C 实例，且必须同步调整量化打包与 K padding：

| MXINT8 家族 | A layout | B layout | BF16 linear C layout | K 对齐 | 实例化位置 / 函数 |
|---|---|---|---|---|---|
| R32，当前方案 | `(32,32,4,1,1)` | `(32,32,4,1,1)` | `(16,32,1,1,0)` | 128 | `kernel/instantiations/mxint8/inst_mxint8_r32_4x1.su:29 / mma_dte` |
| R16 | `(64,16,4,1,1)` | `(32,32,4,1,1)` | `(32,16,1,1,0)` | 256 | `kernel/instantiations/mxint8/inst_mxint8_r16.su:29 / mma_dte` |
| R8 | `(128,8,4,1,1)` | `(32,32,4,1,1)` | `(64,8,1,1,0)` | 512 | `kernel/instantiations/mxint8/inst_mxint8_r8.su:29 / mma_dte` |

不要把 R32 packed buffer 直接换模板当作 R8/R16 输入。也不要只因为原函数名含 `universal`，就假定一个固定 layout 的新模板能覆盖旧函数的全部 shape 分支。

### 6.5 四类 MX 输入的具体模板选择

下表均指 `include/simma.h:180 / mma_dte` 的 **同一个不带 workspace 的显式 layout 重载**，并以最适合先接入的 R32、K 方向 4-tile supertile 为例。A/B 均为 tiled packed 输入，C 为 BF16 linear 输出 `(16,32,1,1,0)`。

| 输入类别 | inputT | A/B layout | M/N 对齐；K 对齐 | BF16 linear 生产实例化位置 / 函数 |
|---|---|---|---|---|
| MXINT8 | `sifmt::mxint8` | `(32,32,4,1,1)` | 32；128 | `kernel/instantiations/mxint8/inst_mxint8_r32_4x1.su:29 / mma_dte` |
| MXFP4 E2M1 | `sifmt::mxfloat4e2m1` | `(64,32,4,1,1)` | 32；256 | `kernel/instantiations/mxfp4/inst_mxfp4.su:34 / mma_dte` |
| MXFP6 E2M3 | `sifmt::mxfloat6e2m3` | `(128,32,4,1,1)` | 32；512 | `kernel/instantiations/mxfloat6e2m3/inst_mxfloat6e2m3_r32_4x1.su:28 / mma_dte` |
| MXFP6 E3M2 | `sifmt::mxfloat6e3m2` | `(128,32,4,1,1)` | 32；512 | `kernel/instantiations/mxfloat6e3m2/inst_mxfloat6e3m2_r32_4x1.su:15 / mma_dte` |
| MXFP8 E4M3 | `sifmt::mxfloat8e4m3` | `(32,32,4,1,1)` | 32；128 | `kernel/instantiations/mxfp8/inst_mxfp8_r32_4x1_bf16.su:9 / mma_dte` |
| MXFP8 E5M2 | `sifmt::mxfloat8e5m2` | `(32,32,4,1,1)` | 32；128 | `kernel/instantiations/mxfp8/inst_mxfp8_r32_4x1_bf16.su:29 / mma_dte` |

表中的 `outputT`、`scalarT` 都是 `sifmt::bfloat16`。如果需要改输出精度，必须选择已有的完整实例化，而不是任意替换一个模板参数：

| 输入类别 | 本次检查到的生产实例化输出 | 依据 |
|---|---|---|
| MXINT8 | BF16、FP16、FP32 | `kernel/instantiations/mxint8/inst_mxint8_r32_4x1.su:29 / mma_dte`，同文件第 33、37 行分别是 FP16/FP32 linear |
| MXFP6 两种编码 | BF16、FP16、FP32 | `kernel/instantiations/mxfloat6e2m3/inst_mxfloat6e2m3_r32_4x1.su:28 / mma_dte`；E3M2 对应文件第 15 行起 |
| MXFP8 两种编码 | BF16、FP16 | `kernel/instantiations/mxfp8/inst_mxfp8_r32_4x1_bf16.su:9 / mma_dte` 和 `kernel/instantiations/mxfp8/inst_mxfp8_r32_4x1_f16.su:9 / mma_dte` |
| MXFP4 E2M1 | BF16 | `kernel/instantiations/mxfp4/inst_mxfp4.su:25 / mma_dte` 起的生产实例化清单 |

FP32 输出的 R32 C tile 应为 `(8,32,1,1,0)`，不是 BF16/FP16 的 `(16,32,1,1,0)`；对应 scalarT 也随已有实例化使用 FP32。公共枚举允许列出某个 dtype，不代表每个输入/输出/标量/layout 的笛卡尔积都已经生成了可链接符号。

还需要明确四个边界：

1. **A/B 必须是同一种 inputT。** 接口只有一个 `inputT`，不是分别声明 `inputA/inputB`；当前这些入口不是 W4A8、BF16 activation + MXFP4 weight 等混合输入接口。旧函数中的 `bf16` 也是输出类型，不是说 A 输入可以不量化。
2. **MX 类型不等于普通低比特类型。** 这里的 MXFP4 具体是 E2M1，不包含 E1M2；MXFP8 不能用普通 `sifmt::float8e4m3` 或不含 metadata 的 FP8 数据冒充。`kernel/detail/mma_dte_tiled_tensor_dtype_meta.hpp:149 / get_input_dtype_info` 区分了普通 FP8 与 MXFP8 的 TensorMap dtype/header。
3. **“支持 linear 输入/输出”是泛型接口能力，不代表本表 MX 生产实例化接受 linear MX 输入。** 上表全部使用 `layoutA/B.tensor_format=1`。现有 vLLM packed 输入就应保持 1，只把 C 的 format 设为 0。
4. **MXFP6 需区分公开逻辑 layout 与芯片物理 tile。** `kernel/detail/mma_dte_tiled_tensor_dtype_meta.hpp:121 / mma_dte_physical_tile_k` 在 SIPU>=160 时把 R32/R16/R8 的物理 K tile 映射为 32/64/128；`kernel/detail/mma_dte_tiled_tensor_tensor_map.hpp:71 / create_tensor_maps` 的第 86 行起据此调整 TensorMap。不能看到物理 R32 是 32 就把公开 `(128,32,4,1,1)` 随意改成 `(32,32,4,1,1)`，也不能在不同架构之间沿用未经核对的 packed buffer。

此外确实还有 `(2,2)`、`(1,4)` supertile 实例，但它们不是仅影响性能的任选开关：输入物理排布、M/N/K 对齐和 shape guard 都要匹配。`kernel/detail/mma_dte_tiled_tensor_layout_contracts.hpp:139 / mma_dte_mx_2x2_shape_supported` 对 2x2 额外要求 `K == 2 * physical_tile_k` 且行数大于 16；`kernel/mma_dte_tiled_tensor.hpp:54 / mma_dte_implement_with_control` 还会做前述逻辑对齐检查。当前 vLLM MXINT8 路径不需要引入这些分支。

### 6.6 linear 输出究竟如何实现，能省掉什么

**能直接输出 linear；不是先生成完整 tiled C，再在另一个 kernel 中把 C 转为 linear。** 有三层源码证据：

| 层次 | 位置 / 函数 | 行为 |
|---|---|---|
| 描述符 | `kernel/detail/mma_dte_tiled_tensor_tensor_map.hpp:71 / create_tensor_maps` | 第 199 行起把 `layoutC.tensor_format==0` 映射为 `TMAP_FORMAT_LINEAR`；第 249 行起编码输出 TensorMap |
| GEMM 写回分支 | `kernel/mma_dte_tiled_tensor_kernel.hpp:1546 / kernel_tile_mma_procon_v2_tiled_tensor_map` | 第 1726 行起在 `store_tile` 与 `store_linear` 之间编译期分支；其他 schedule 也有同样的分支，例如同文件第 106 行的 `kernel_tile_mma_smem_dte_tiled_tensor_map`，写回分支在第 318 行 |
| MXINT8/MXFP6 R32 具体写回 | `kernel/util/mma_dte_tiled_tensor_kernel_util_mxint8_mxfp6_r32.hpp:6366 / loadL2B_mma_r32_1m_1n_impl::store_linear` | BF16 路径先 `tcvt.tt.bf16.f32.r32`，再用 `tst.trrr.stride.u32.global` 按 N 计算行步长和地址，直接写入 linear C |

这里要区分两个概念：`tcvt` 做 FP32 累加结果到 BF16 的数值转换；随后带 stride 的 store 完成 row-major 写回。**不能因为函数名叫 mma_dte，就把这条 R32 输出路径说成一定由 DTE 执行独立 tile-to-linear copy。** 上面这条具体路径使用的是 kernel 内的 tile strided store。

因此未来接入时，**VLLM** `vllm_sipu/ops/backends/sikernel/jit/mxint8.py:120 / mxint8_bf16_matmul` 应在 C++ 调用后直接返回 `out`，不再执行第 129 行起的 `_mxint8_output_tile_shape` 和 `_tileformat_to_linear`。保留旧重排会把已经是 linear 的结果再次错排，这不是单纯的性能浪费。

性能上可以确定的是：一般需要物化重排的形状将不再需要这次额外输出分配、完整 C 的读取和再次写入。对于 BF16 padded `[M,N]`，这次 full-tensor reorder 的读写量约为 `4*M*N` 字节。**这不是整体提速比例的承诺**：新 kernel 的调度成本、shape、带宽和量化开销都需要单独测量。

当前还有一项不能与本次收益混淆：**VLLM** `vllm_sipu/model_executor/layers/quantization/utils/mxint8_utils.py:145 / build_mxint8_tile_tensor_on_cpu` 会在 CPU 做 tile 重排，`:163 / quantize_to_mxint8` 又将其送回原设备。只替换 GEMM 不会消除这些输入侧传输。第一阶段可以不改它，但端到端收益不能只看 GEMM 时间。

### 6.7 MXFP4、MXFP6、MXFP8、MXINT8 功能测试清单

以下列出 DTE 当前源码中的 **29 个类型专用功能测试 `.cpp` 文件**：MXFP4 6 个、MXFP6 8 个、MXFP8 11 个、MXINT8 4 个。包括普通功能测试和 SplitK 功能测试；参数化文件用 shape 集合描述，不逐行展开笛卡尔积。后面另外列出辅助测试与 JSON 配置，避免把配置文件或模板实例化文件误算成一个执行测试。

所有路径均相对 `/share/users/like/package/sikernel/mma_dte_tile_tensor`。除特别标注外，输入都是相应 MX 类型的 tiled packed A/B。

#### 6.7.1 MXFP4 E2M1：普通功能测试

| 相对路径:行号 / 执行函数 | GTest 名称 | 输出与 shape 覆盖 |
|---|---|---|
| `testcase/func_test/test_bf16_mxfp4_r32_host/test_bf16_mxfp4_r32_host.cpp:202 / run_param_case`；`:182 / launch_mma` | `MmaDteBf16Mxfp4R32Test.Computes`，注册宏在第 1024 行 | BF16 tiled/linear；标准布局 M/N=`32,64,...,256`，K=`256,512,768,1024,1280,1536`；2x2 布局 M/N=`64,128,192,256`，K=128；另一个 1x4 API 布局 M/N=`128,256,384,512`，K=64 |
| `testcase/func_test/test_bf16_mxfp4_r16_host/test_bf16_mxfp4_r16_host.cpp:123 / run_test_case` | `MmaDteBf16Mxfp4R16Test.Computes`，第 259 行 | BF16 tiled/linear；M=16，N=`32,64,...,256`，K=`512,1024,1536,2048` |
| `testcase/func_test/test_bf16_mxfp4_r8_host/test_bf16_mxfp4_r8_host.cpp:129 / run_test_case` | `MmaDteBf16Mxfp4R8Test.Computes`，第 272 行 | BF16 tiled/linear；M=8，K=`1024,2048,3072,4096`；tiled 的 N=`32,64,...,256`，linear 的 N=`64,128,192,256` |

R32 参数生成函数为 `testcase/func_test/test_bf16_mxfp4_r32_host/test_bf16_mxfp4_r32_host.cpp:102 / make_shape_params`。当前 GTest 调用的是 `:1013 / MmaDteBf16Mxfp4R32Test::run_current_shape`，再进入表中的 `run_param_case`；不要把同文件保留的老 `run_tests*` 大循环重复统计为额外 GTest。

注意 R32 测试里的 `SubTile4x1` 是按 SiTe 行列方向命名，其 `launch_mma` 第 196/198 行实际传的是 `tensor_layout(64,32,1,4,1)`。这与“API 的 supertile_shape0=4、shape1=1”不是同一个方向。

#### 6.7.2 MXFP6：普通功能测试

| 相对路径:行号 / 执行函数 | GTest 名称 | 输出与 shape |
|---|---|---|
| `testcase/func_test/test_mxfp6e2m3_host/test_bf16_mxf6e2m3_host.cpp:51 / run_default_case` | `MmaDteBf16Mxfp6e2m3Test.ComputesDefaultShape`，第 249 行 | E2M3 -> BF16，R32，**tiled 输出**，`(M,N,K)=(32,1024,1024)` |
| 同文件 `:152 / run_loop_k_one_identity_decode_case` | `MmaDteBf16Mxfp6e2m3Test.DecodesSingleKPanelIdentity`，第 253 行 | E2M3 -> BF16，**tiled 输出**，`(32,512,512)`；先运行上一个多 K 用例，再用 identity B 验证单 K panel 的累加器初始化 |
| `testcase/func_test/test_bf16_mxfp6e3m2_host/test_bf16_mxf6e3m2_host.cpp:50 / run_default_case` | `MmaDteBf16Mxfp6e3m2Test.ComputesDefaultShape`，第 155 行 | E3M2 -> BF16，R32，**tiled 输出**，`(1024,256,1024)` |

上表是两个源文件、三个注册用例。它们不能单独作为 MXFP6 linear 输出已验证的证据；linear 的现成例子在下一组 SplitK 测试。

#### 6.7.3 MXFP8：普通功能测试

| 相对路径:行号 / 执行函数或注册宏 | GTest 名称 | 输出与 shape |
|---|---|---|
| `testcase/func_test/test_mxfp8_host/r32/test_mxfp8_r32.cpp:35 / run_param` | `MmaDteMxfp8R32Test.Computes`，第 51 行 | E5M2 -> FP16 tiled `(96,96,256)`；E4M3 -> FP16 **linear** `(96,96,256)`；daily 增加 E4M3 linear `(128,128,256)` 和 E5M2 tiled `(128,128,1024)`、`(512,1024,2048)` |
| 同文件 `:63 / TEST`，执行 `run_case` | `MmaDteMxfp8R32Regression.SingleFullChunkReinitializesAccumulator` | E4M3 -> FP16 **linear**，重复运行 `(96,96,256)`，验证每次 launch 的累加器初始化 |
| `testcase/func_test/test_mxfp8_host/r16/test_mxfp8_r16.cpp:30 / TEST_P`，执行 `run_case` | `MmaDteMxfp8R16Test.Computes` | E4M3 -> FP16 **tiled**；默认 `(16,128,512)`；daily 为 `(16,128,1024)`、`(16,256,2048)`、`(16,6144,2048)` |
| `testcase/func_test/test_mxfp8_host/r8/test_mxfp8_r8.cpp:30 / TEST_P`，执行 `run_case` | `MmaDteMxfp8R8Test.Computes` | E5M2 -> FP16 **tiled**；默认 `(8,96,512)`；daily 为 `(8,32,512)`、`(8,160,1024)`、`(8,2560,2048)` |
| 同文件 `:41 / TEST`，执行 `run_case` | `MmaDteMxfp8R8Regression.SingleFullChunkReinitializesAccumulator` | E5M2 -> FP16 **tiled**，重复运行 `(8,32,512)` |
| `testcase/func_test/test_mxfp8_host/r32/test_mxfp8_r32_shape_2x2.cpp:48 / TEST_P`，执行 `run_case` | `MmaDteMxfp8R32Shape2x2Test.Computes` | E4M3 -> FP16 **linear**，API supertile `(2,2)`；默认 `(64,64,64)`，daily `(128,128,64)` |
| 同文件 `:28 / invoke_2x2`；注册宏在第 36、40、44 行 | `MmaDteMxfp8R32LayoutDeathTest.RejectsMultipleKSupertiles`、`.RejectsSmallM`、`.RejectsSmallN` | 错误参数拒绝测试：分别传 `(128,128,256)`、`(16,64,64)`、`(64,16,64)`，不是正常 GEMM 精度用例 |
| `testcase/func_test/test_mxfp8_host/r32/test_mxfp8_r32_shape_4x1.cpp:26 / TEST_P`，执行 `run_case` | `MmaDteMxfp8R32Shape4x1Test.Computes` | E4M3 -> FP16 **linear**，`(128,128,32)`；这里 API layout 实际是 `(32,32,1,4,1)` |

上表对应五个源文件。共享执行函数为 `testcase/func_test/test_mxfp8_host/common/test_mxfp8_common.hpp:48 / run_case`：第 99 行调用显式 layout、不带 workspace 的 `mma_dte`；第 113 行起按 LayoutC 决定比较 tiled memory order 还是 linear golden。daily 开关在同文件 `:41 / is_daily_suite`，读取 `SITEST_CASE_LEVEL=daily`。

不要把文件名、旧 `.su` 实例化清单或 BF16 相关常量当作当前 GTest 全部覆盖 BF16 输出的证据；上表这些 MXFP8 普通执行用例实际使用 FP16 输出。**BF16 linear 生产实例化存在，但若接入 MXFP8->BF16，仍应补相应 vLLM contract 回归。**

#### 6.7.4 MXINT8：普通功能测试

| 相对路径:行号 / 执行函数 | GTest 名称 | 输出与 shape |
|---|---|---|
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:126 / run_bf16_default_case` | `MmaDteBf16Mxint8Test.ComputesDefaultShape`，第 138 行 | R8 MXINT8 -> BF16 **linear**，`(8,576,1024)`；源码说明这是 linear-store 回归 |
| 同文件 `:131 / run_default_case` | `MmaDteFloat16Mxint8Test.ComputesDefaultShape`，第 144 行 | R32 MXINT8 -> FP16 **tiled**，`(32,576,1024)` |

实际执行在同文件 `:43 / run_case`，第 90 行调用 `mma_dte`。虽然文件位于 `test_bf16_mxint8_host`，其中第二个用例已经是 FP16 输出，不能只按目录名判断覆盖范围。

#### 6.7.5 SplitK 功能测试：四类 MX 均有 linear 输出例子

下面 18 个文件全部以 `run_default_cases` 为执行入口，全部使用显式 layout、不带 workspace 的 `mma_dte`，全部配置 **linear 输出**。这些文件的 GTest case 名都是 `ComputesDefaultCases`；fixture 名包含相应目录/文件名。MXINT8 的注册宏位于各文件第 32 行，其余位于第 33 行。

| 输入 -> 输出 | 家族 | 相对路径:行号 / 函数 |
|---|---|---|
| MXFP4 E2M1 -> BF16 | R32 | `testcase/func_test/splitK/test_bf16_mxfp4_host/test_bf16_mxfp4_splitK_host.cpp:21 / run_default_cases` |
| MXFP4 E2M1 -> BF16 | R16 | `testcase/func_test/splitK/test_bf16_mxfp4_r16_host/test_bf16_mxfp4_splitK_host_r16.cpp:21 / run_default_cases` |
| MXFP4 E2M1 -> BF16 | R8 | `testcase/func_test/splitK/test_bf16_mxfp4_r8_host/test_bf16_mxfp4_splitK_host_r8.cpp:21 / run_default_cases` |
| MXFP6 E2M3 -> BF16 | R32 | `testcase/func_test/splitK/test_bf16_mxfp6e2m3_host/test_bf16_mxfp6e2m3_splitK_host.cpp:21 / run_default_cases` |
| MXFP6 E2M3 -> BF16 | R16 | `testcase/func_test/splitK/test_bf16_mxfp6e2m3_r16_host/test_bf16_mxfp6e2m3_splitK_host_r16.cpp:21 / run_default_cases` |
| MXFP6 E2M3 -> BF16 | R8 | `testcase/func_test/splitK/test_bf16_mxfp6e2m3_r8_host/test_bf16_mxfp6e2m3_splitK_host_r8.cpp:21 / run_default_cases` |
| MXFP6 E3M2 -> BF16 | R32 | `testcase/func_test/splitK/test_bf16_mxfp6e3m2_host/test_bf16_mxfp6e3m2_splitK_host.cpp:21 / run_default_cases` |
| MXFP6 E3M2 -> BF16 | R16 | `testcase/func_test/splitK/test_bf16_mxfp6e3m2_r16_host/test_bf16_mxfp6e3m2_splitK_host_r16.cpp:21 / run_default_cases` |
| MXFP6 E3M2 -> BF16 | R8 | `testcase/func_test/splitK/test_bf16_mxfp6e3m2_r8_host/test_bf16_mxfp6e3m2_splitK_host_r8.cpp:21 / run_default_cases` |
| MXFP8 E4M3 -> FP16 | R32 | `testcase/func_test/splitK/test_fp16_mxfp8e4m3_host/test_fp16_mxfp8e4m3_splitK_host.cpp:21 / run_default_cases` |
| MXFP8 E4M3 -> FP16 | R16 | `testcase/func_test/splitK/test_fp16_mxfp8e4m3_r16_host/test_fp16_mxfp8e4m3_splitK_host_r16.cpp:21 / run_default_cases` |
| MXFP8 E4M3 -> FP16 | R8 | `testcase/func_test/splitK/test_fp16_mxfp8e4m3_r8_host/test_fp16_mxfp8e4m3_splitK_host_r8.cpp:21 / run_default_cases` |
| MXFP8 E5M2 -> FP16 | R32 | `testcase/func_test/splitK/test_fp16_mxfp8e5m2_host/test_fp16_mxfp8e5m2_splitK_host.cpp:21 / run_default_cases` |
| MXFP8 E5M2 -> FP16 | R16 | `testcase/func_test/splitK/test_fp16_mxfp8e5m2_r16_host/test_fp16_mxfp8e5m2_splitK_host_r16.cpp:21 / run_default_cases` |
| MXFP8 E5M2 -> FP16 | R8 | `testcase/func_test/splitK/test_fp16_mxfp8e5m2_r8_host/test_fp16_mxfp8e5m2_splitK_host_r8.cpp:21 / run_default_cases` |
| MXINT8 -> BF16 | R32 | `testcase/func_test/splitK/test_bf16_mxint8_host/test_bf16_mxint8_splitK_host.cpp:21 / run_default_cases` |
| MXINT8 -> BF16 | R16 | `testcase/func_test/splitK/test_bf16_mxint8_r16_host/test_bf16_mxint8_splitK_host_r16.cpp:21 / run_default_cases` |
| MXINT8 -> BF16 | R8 | `testcase/func_test/splitK/test_bf16_mxint8_r8_host/test_bf16_mxint8_splitK_host_r8.cpp:21 / run_default_cases` |

共享执行函数为 `testcase/func_test/splitK/common/test_splitk_mx_common.hpp:95 / run_mx_case`，第 134 行调用 `mma_dte`，第 154 行起选择 linear golden。shape 遍历分别在同文件 `:214 / run_mx_cases`、`:221 / run_r8_mx_cases`、`:228 / run_r16_mx_cases`，再调用 `:182 / run_mx_cases_for_shapes`。

核对时真正启用的 shape 常量位于 `testcase/func_test/splitK/common/test_splitk_shapes.hpp`：第 28 行的 R32 集合只有 `(64,64,2048)`，第 37 行的 R8 集合只有 `(8,64,16384)`，第 43 行的 R16 集合只有 `(16,64,16384)`。注释掉的 `(32,32,2048)` 等不能算成当前实际测试覆盖。

**最适合参考本次 MXINT8 替换调用的用例**是 `testcase/func_test/splitK/test_bf16_mxint8_host/test_bf16_mxint8_splitK_host.cpp:21 / run_default_cases`：A/B `(32,32,4,1,1)`，C `(16,32,1,1,0)`，正好与建议一致。

但 SplitK 测试还有专用编译控制，例如 `testcase/func_test/splitK/test_bf16_mxint8_host/kernel/inst_bf16_mxint8_splitK.su:18` 定义 `MMA_DTE_SPLITK_TEST_FORCE_SPLITK`。所以这些用例是调用与 linear 输出契约的参考，不代表普通生产 planner 在所有相同 shape 上也必定选择 SplitK，更不代表主库的全部调度路径已经被本次验证。

#### 6.7.6 辅助测试与配置文件

下列辅助测试不能替代上面的设备 GEMM 数值测试：

| 相对路径:行号 / 注册宏 | 测试名与用途 |
|---|---|
| `testcase/autotune/tests/test_autotune_shape_guard_mxint8.cpp:5 / TEST` | `AutotuneShapeGuardMxint8Test.AcceptsValidMxint8Shape`，MXINT8 shape guard |
| `testcase/autotune/tests/test_autotune_override_candidates_mxfp8.cpp:5 / TEST` | `AutotuneOverrideCandidatesMxfp8Test.FiltersKnownBadMxfloat8Overrides`，MXFP8 候选过滤 |
| `testcase/autotune/tests/test_autotune_layout_contracts.cpp:5 / TEST` | `AutotuneLayoutContractsTest.ValidatesSupportedLayouts`，包含 MXINT8/MXFP6 layout 静态契约 |
| `testcase/autotune/tests/test_autotune_case_buffers_layout.cpp:35 / TEST` | `AutotuneCaseBuffersLayoutTest.SupportsMxSupertileLayouts`，MXINT8 buffer 布局 |
| `testcase/autotune/tests/test_autotune_shape_guard_manifest_coverage.cpp:9 / TEST` | `AutotuneShapeGuardManifestCoverageTest.CoversManifestInputDtypes`，包括六个目标 MX 编码的 manifest 覆盖 |

此外还有 `testcase/func_test/test_validation_host/test_validation_host.cpp:40 / TEST` 起的参数校验测试，其目标函数是 `src/simma_validation.cpp:251 / mma_dte_validate_inputs`，不是运行 GEMM。校验 API 与实际 kernel 入口不是完全同一组规则：例如 `src/simma_validation.cpp:177 / validate` 第 229 行仍保留 R8 非 MXFP8 的 BF16/FP16 linear 奇数 N tile 拒绝条件，而 `kernel/mma_dte_tiled_tensor.hpp:54 / mma_dte_implement_with_control` 第 86 行已说明 kernel 写回可以处理此情形。第一阶段采用 R32 不受此差异影响；以后扩展 R8 时应专门核对，不能把 helper 的结果当成所有 kernel 能力的完整定义。

仓库还保留以下 MXINT8/MXFP6 JSON 数据集，都是配置文件，无函数名；不能从文件名直接推断上述 GTest 会读取它们：

```text
testcase/func_test/test_bf16_mxint8_host/config_mxint8_full.json
testcase/func_test/test_mxfp6e2m3_host/config_mxfp6e2m3_full.json
testcase/func_test/test_mxfp6e2m3_host/config_mxfp6e2m3_exact_full.json
testcase/func_test/test_mxfp6e2m3_host/config_mxfp6e2m3_structural_full.json
testcase/func_test/test_mxfp6e2m3_host/config_mxfp6e2m3_supertile.json
testcase/func_test/test_mxfp6e2m3_host/config_mxfp6e2m3_all.json
testcase/func_test/test_bf16_mxfp6e3m2_host/config_mxfp6e3m2_full.json
testcase/func_test/test_bf16_mxfp6e3m2_host/config_mxfp6e3m2_exact_full.json
testcase/func_test/test_bf16_mxfp6e3m2_host/config_mxfp6e3m2_structural_full.json
testcase/func_test/test_bf16_mxfp6e3m2_host/config_mxfp6e3m2_supertile.json
testcase/func_test/test_bf16_mxfp6e3m2_host/config_mxfp6e3m2_all.json
```

这里将“可执行功能测试”“辅助规则测试”“配置数据”分开列出，避免把存在大量 JSON shape 等同于当前测试全部已经编译、执行和通过。

### 6.8 将来接入时需要改动的边界（本次不执行）

1. **C++ GEMM 入口**：在 **VLLM** `csrc/jit/quantization/mxint8.cpp:62 / launch_mxint8_matmul` 改用第 6.4 节的 `mma_dte`，保留当前 stream、packed buffer 检查和 BF16 输出契约。不要同时改变 MX 量化语义。
2. **Python 输出契约**：在 **VLLM** `vllm_sipu/ops/backends/sikernel/jit/mxint8.py:120 / mxint8_bf16_matmul` 直接返回 linear `out`；上层 **VLLM** `vllm_sipu/model_executor/layers/quantization/utils/mxint8_utils.py:202 / apply_mxint8_linear` 的裁剪、bias、最终形状恢复继续保留。当前 `beta` 未实现，不能借它融合 bias。
3. **构建与依赖**：不能只换一个 include。**VLLM** `vllm_sipu/ops/backends/sikernel/jit/mxint8.py:26` 的 `MXINT8_MODULE` 仍声明旧 GEMM `.su/.hpp`；`:53 / get_mxint8_module` 通过 JIT builder 构建。**VLLM** `vllm_sipu/ops/backends/sikernel/jit/_runtime/builder.py:165 / build_module` 当前第 222 行调用 `generate_ninja_build` 时没有传外部 GEMM 库链接参数；`vllm_sipu/ops/backends/sikernel/jit/_runtime/cpp_ext.py:201 / generate_ninja_build` 已使用 C++20，但默认只链接 TVM FFI 和 SIPU runtime。接入需明确固定 DTE revision、头文件、库和运行时搜索路径，不能假定头文件模板声明会自动产生实现。
4. **共享库部署**：当前主库是 umbrella DSO，不能只复制一个 `libtile_mma_dte.so`。DTE `CMakeLists.txt:318` 起的构建逻辑生成各实例化 component DSO；`tools/generate_mma_dte_umbrella.py:47 / write_cpp` 生成的 `__mma_dte_resolve_component_symbol` 会从主库所在目录 `dlopen` 对应 component。至少要携带用到的 component 及其匹配的 SDK runtime；这是打包契约，不是要求把整个新 DTE 再放进 vLLM JIT 编译一遍。
5. **分层验证**：先用同一批 packed A/B 比较旧 GEMM+tiled-to-linear 与新 GEMM linear，再比较模型输出，最后分别测 GEMM、reorder 和端到端时间。至少覆盖当前 `(32,96,128)` padded 案例以及更大的 M/N/K，验证非默认 stream 和错误尺寸。**VLLM** `tests/kernels/quantization/test_mxint8_unpack_pytest.py:148 / main` 有手动参数遍历入口，而且该文件第 34 行设置 `__test__=False`；不要只运行文件名对应的 pytest 命令就以为真正执行了这些用例。

### 6.9 本次只读核对的构建快照与最终判断

2026-09-16 核对时，指定目录下的 `build/` 只有测试构建子目录，未找到 `build/libtile_mma_dte.so`；已有的主库及 component 位于 `build-old/`。这只是本次文件系统快照，不推断是谁移动了目录，也没有触碰这些构建产物。

只读检查 `build-old/libtile_mma_dte.so` 的动态符号表得到 504 个 `mma_dte` 模板导出，未发现简化重载的 `(A,B,C,M,N,K,stream)` 签名。更重要的是 `build-old/mma_dte_build_manifest.txt:2` 记录：

```text
SI_SDK_ROOT=/share_data/sicx_sdk/release/2609151958
TARGET_SIPU_ARCH=150
```

这与本题提供的 SDK `2609020046` 不一致。因此这里只把 ELF 检查作为“已有构建包含显式重载、不包含简化重载”的辅助证据，**没有宣称这份库已能在指定的 2609020046/vllm_dev 环境中直接链接运行**。实施前需固定并验证双方 SDK、SiTe 类型 ABI 和目标架构；本次不处理测试构建失败，也不调整库位置。

**最终建议：当前先选显式 layout、不带 workspace 的 R32 MXINT8->BF16 实例，A/B 保持 `(32,32,4,1,1)`，C 改为 `(16,32,1,1,0)`，同步取消 Python 输出重排。四类 MX 都能通过同一重载家族接入，区别在具体 inputT、layout、输出实例化与打包契约；无需为了支持不同 MX 类型选择不同函数重载。linear 写回能力已经在源码中实现，实际收益和跨 SDK 可用性留待正式接入时验证。**

### 6.10 scalarT 表示哪个矩阵的类型？（2026-09-16 补充）

**`scalarT` 不表示 A、B、C 中任何一个矩阵的类型。它表示两个标量参数 `alpha`、`beta` 的类型。** 本节重新核对的代码根目录是 `/share/users/like/package/sikernel/mma_dte_tile_tensor`，以下全部使用该目录的相对路径，不引用 vLLM 内的旧实现。

#### 6.10.1 三个模板类型的对应关系

`include/simma.h:180 / mma_dte` 的相关参数声明是：

```cpp
template <class inputT, class outputT, class scalarT,
          mma_dte_api::tensor_layout layoutA,
          mma_dte_api::tensor_layout layoutB,
          mma_dte_api::tensor_layout layoutC>
void mma_dte(mma_dte_api::Operation transa,
             mma_dte_api::Operation transb,
             int M, int N, int K,
             scalarT alpha,
             void* A, int lda,
             void* B, int ldb,
             scalarT beta,
             void* C, int ldc,
             sipuStream_t stream = nullptr);
```

同一函数的第 161 行注释明确把 `scalarT` 描述为 alpha 和 beta 的标量类型；第 182、185 行分别声明 `scalarT alpha`、`scalarT beta`。所以之前推荐的三个类型应当这样读：

| 模板参数 | 推荐值 | 表示什么 |
|---|---|---|
| `inputT` | `sifmt::mxint8` | A 和 B 的输入元素格式，二者都为 packed MXINT8；不是只控制 A |
| `outputT` | `sifmt::bfloat16` | 输出矩阵 C 的元素类型 |
| `scalarT` | `sifmt::bfloat16` | `alpha`、`beta` 这两个数值参数的类型；不是另一个矩阵类型 |

这里的 A/B/C 参数虽然写成 `void*`，其数据格式仍必须与模板类型及 layout 匹配，不能因为是无类型指针就任意混用数据。

#### 6.10.2 alpha、beta 原本分别作用于什么

按照 `include/simma.h:180 / mma_dte` 的接口设计，第 170 行将 alpha 描述为矩阵乘积的系数，第 175 行将 beta 描述为 C 的系数。沿用本题 A 为 `[M,K]`、B 为 `[N,K]` 的表示法，带这两个系数的数学形式应写为：

```text
C_new = alpha * (A @ B^T) + beta * C_old
```

alpha 是整个乘积的统一系数，不是 A 的 dtype；beta 是原有 C 的统一系数，也不是 C 的 dtype。它们各是一个按值传入的标量，不是矩阵、每行/每列的 scale 数组，也不是 MXINT8 packed 数据中每个 block 的共享 scale。

对于你需要的纯 GEMM `C = A @ B^T`，表达调用意图时使用：

```cpp
sifmt::bfloat16 alpha(1.0f);
sifmt::bfloat16 beta(0.0f);
```

**但这个带系数的公式是接口设计含义，不是当前实现已经完整支持的行为。**

#### 6.10.3 当前代码实际上没有使用 alpha、beta

`kernel/mma_dte_tiled_tensor.hpp:193 / mma_dte` 会把两个参数继续传给包装层，最终进入 `kernel/mma_dte_tiled_tensor.hpp:54 / mma_dte_implement_with_control`。该函数第 69 行仍明确写着：

```cpp
// TODO: Use transa, transb, alpha, beta, lda, ldb, ldc in implementation
```

这不仅是注释：该函数第 104 行调用 `mma_r32_dte<inputT, outputT, layoutA, layoutB, layoutC>` 时，既没有传入 `scalarT`，也没有传入 alpha、beta 的值。因此在本次讨论的 MXINT8 R32 路径上：

- 改 alpha 为 2，当前不会得到两倍的矩阵乘积。
- 改 beta 为 1，当前不会把调用前 C 的内容加到结果中。
- 推荐仍传 `alpha=1,beta=0` 表达纯 GEMM 意图，但不要据此假定该接口已经实现一般的缩放与累加功能。

#### 6.10.4 为什么 scalarT 推荐 BF16，而不是 FP32

**原因是匹配现成的模板实例化，不是因为矩阵乘法必须用 BF16 标量或 BF16 累加。**

`kernel/instantiations/mxint8/inst_mxint8_r32_4x1.su:29 / mma_dte` 已经显式实例化了：

```cpp
mma_dte<sifmt::mxint8, sifmt::bfloat16, sifmt::bfloat16,
        MXINT8_LAYOUT_A_R32, MXINT8_LAYOUT_B,
        MXINT8_LAYOUT_C_R32_R16BIT_LINEAR>(...);
```

这里第二个 BF16 是 C 的类型，第三个 BF16 才是 alpha/beta 的类型。这个文件还实例化了 FP16 输出搭配 FP16 标量、FP32 输出搭配 FP32 标量的组合，但没有为上述 BF16 输出 layout 实例化 `scalarT=sifmt::float32` 的组合。即使 alpha/beta 当前未用于计算，擅自更换 scalarT 仍会选择不同的 C++ 模板符号，不能假定已有库能链接它。

`scalarT` 与 `outputT` 在模板声明中是两个独立参数；当前推荐组合取相同类型，是这个生产实例化的选择，不是“scalarT 就等于输出矩阵类型”的定义。

#### 6.10.5 scalarT 也不是累加器类型

本次 MXINT8 R32 路径的内部累加是 FP32，源码证据为：

- `kernel/util/mma_dte_tiled_tensor_kernel_util_mxint8_mxfp6_r32.hpp:6086 / loadL2B_mma_r32_1m_1n_impl::load_mma_1st_stp`：第 6091 行起使用 `tmma.ttt.f32.mxi8.mxi8.r32...` 指令，两个输入是 MXINT8，累加结果为 FP32。
- `kernel/util/mma_dte_tiled_tensor_kernel_util_mxint8_mxfp6_r32.hpp:6366 / loadL2B_mma_r32_1m_1n_impl::store_linear`：根据 `outputT` 选择写回类型；BF16 分支第 6369 行使用 `tcvt.tt.bf16.f32.r32`，将 FP32 结果转换为 BF16 后写出。

因此正确的数据类型关系是：

```text
A/B: packed MXINT8 -> GEMM 内部 FP32 累加 -> C: BF16
alpha/beta: BF16 标量参数，当前实现忽略它们
```

**一句话回答：`inputT` 管 A/B，`outputT` 管 C，`scalarT` 管 alpha/beta；`scalarT` 不表示任何矩阵的类型，也不决定这里的累加精度。** 本次仅追加说明，没有执行构建命令、修改算子或运行设备测试。

## 7. MXINT8 Host 测试逐行讲解、SiTe 定义与设备代码链接（2026-09-16）

### 7.1 先回答链接方式

**本例不是链接包含所有类型的主库，也不是把设备 `.o` 直接链接进测试 executable，而是第三种方式：测试专用 `.su` -> 测试专用 `.o` -> 测试专用 `.so` -> executable 动态链接该 `.so`。**

```text
testcase/func_test/test_bf16_mxint8_host/kernel/inst_bf16_mxi8.su
    | SDK scc -c
    v
_SipuObj/tile_mma_dte_mxint8/kernel/inst_bf16_mxi8.su.o
    | /usr/bin/c++ -shared
    v
libtile_mma_dte_mxint8.so
    ^
    | 动态链接
test_bf16_mxi8_host.cpp.o + GTest + SIPU runtime
    |
    v
test_bf16_mxi8_host
```

直接依据是 `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:43 / scc_add_library`、`:52 / add_custom_command`、`:74 / target_link_libraries`。这份 `.so` 不是主库 `libtile_mma_dte.so`，名称和源码集合都不同。

还必须区分：**两个 `mma_dte` host 模板入口，不等于只有两个 device kernel。** 测试专用实例化文件只定义 BF16/R8/linear 和 FP16/R32/tiled 两个公开组合；各组合下面还会编译运行时 dispatch 所需的多个 chunk、stage、tail 等模板变体。本次检查已有局部 `.so`，发现恰好两个 `mma_dte` 定义，以及 301 个 `__device_stub__kernel_tile_mma_procon_v2_tiled_tensor_map<...>` 导出定义。该数量是当前二进制快照，不是接口固定保证。

### 7.2 根目录与核对范围

本节代码引用采用 `相对 code base 路径:行号 / 函数名`；成员函数使用 `类::函数`。枚举、类型别名、文件作用域声明、CMake 命令、日志和构建产物会注明类别，不为它们虚构 C++ 函数名。

| 标记 | 根目录 | 使用范围 |
|---|---|---|
| DTE，未标注时的默认根目录 | `/share/users/like/package/sikernel/mma_dte_tile_tensor` | 当前测试、kernel、构建脚本和 CMake；不使用 vLLM 的旧 DTE 副本 |
| SDK | `/share_data/sicx_sdk/release/2609020046` | 题目指定 SDK 的 SiTe 与 runtime 头文件定义 |
| SDK-LOG | `/share_data/sicx_sdk/release/2609161442` | 旧 `temp/cxx.log` 和本次找到的已有构建实际使用的 SDK |
| COMPILER | `/share/users/like/package/compiler-toolchain` | 解释 SIPU host stub、fatbin 嵌入和注册的编译器源码 |

SDK 与 SDK-LOG 的 `include/SiTe/SiTe.hpp`、`include/SiTe/tensor/tensor.hpp` 本次做过 SHA-256 对比，分别逐字节相同，因此下文 SiTe 定义行号也适用于这份旧日志。**这不意味着两套 SDK 的所有文件或 ABI 相同**；例如两套 `sccConfig.cmake` 的行号就不同。

本次仅阅读源码、日志，并执行 `nm/readelf` 等只读检查；没有运行构建脚本、SIPU GEMM 或 GTest。旧日志中的 `Built target` 是构建记录，不是本次测试通过记录。

### 7.3 这个测试实际测什么

测试文件为 `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp`，当前共 146 行，包含两个用例：

| 执行入口 | A、B 的逻辑形状与 dtype | C 的形状、dtype、format |
|---|---|---|
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:126 / run_bf16_default_case` | A=`[8,1024]`，B=`[576,1024]`，都为 MXINT8 | `[8,576]`，BF16，**linear** |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:131 / run_default_case` | A=`[32,1024]`，B=`[576,1024]`，都为 MXINT8 | `[32,576]`，FP16，**tiled** |

虽然目录和文件名包含 `bf16`，第二个用例已经是 FP16 输出。两个用例都计算 `C = A @ B^T`。

数据流程如下：

```text
host 随机 float A/B
  -> SiTe 在 host 上量化为 MXINT8，并按 tile/supertile 布局打包
  -> SiTe 从这份 packed 数据解量化，用 CPU float 三重循环计算 golden
  -> packed A/B 拷到 SIPU
  -> mma_dte 在 SIPU 执行 GEMM
  -> C 拷回 host
  -> 按 C 的 linear/tiled format 比较
```

因此它主要检查 **MXINT8 GEMM 相对于同一份量化后输入的数值正确性及布局正确性**，不是直接拿未量化的随机 float A/B 计算结果来测量量化误差。

### 7.4 测试文件逐行说明

以下行号全部属于 `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp`。每个小节给出完整路径和所属函数，表内继续使用该文件的原始行号。许可证、连续空行和单纯括号合并说明；计算、内存、调用和校验语句逐行解释。

#### 7.4.1 第 1 到 42 行：依赖、布局与模板

定位：`testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:1`，文件作用域；`:43 / run_case` 是随后定义的模板函数。

| 行号 | 代码作用 |
|---|---|
| 1 | 引入 GTest 的 `testing::Test`、`TEST_F`、`EXPECT_EQ` 等定义。 |
| 2 到 18 | 版权与 Apache-2.0 许可注释，不产生计算行为。 |
| 19 | 引入 SIPU runtime，包括设备内存分配、拷贝和释放接口。 |
| 20 | 引入 SDK 的 `sipu.h`。注意它不是 `make_shape/make_tensor` 的定义文件。 |
| 21 | 引入 `SiTe.hpp`，本例 `sipu::make_shape/make_layout/make_tensor` 及 `sifmt` 类型主要由它间接提供。 |
| 22 | 引入 C 标准 I/O 头；本文件没有直接使用 `printf` 等接口。 |
| 23 | 引入 `malloc/free` 等 C 标准库内存接口。 |
| 24 | 引入 filesystem；当前文件没有直接使用其接口。 |
| 25 | 引入 `std::cerr`、`std::endl` 等流输出。 |
| 26 | 引入随机数引擎和分布。 |
| 27 | 引入 `std::fabs`。 |
| 28 | 引入 `std::vector`。 |
| 29 | 引入本仓库 `include/simma.h`，取得 layout 类型、Operation 枚举和 `mma_dte` 声明；不是 device kernel 实现头。 |
| 30 | 空行。 |
| 31、32 | 注释说明保留 BF16/R8 linear-store 回归，同时保留 FP16/R32 覆盖。 |
| 33 | B layout=`(32,32,4,1,1)`：MXINT8 tile 为 32 行 x 32 列，沿 K 组合 4 个 tile，输入为 tiled。 |
| 34 | R8 的 A layout=`(128,8,4,1,1)`：8 行 x 128 列，沿 K 组合 4 个 tile。 |
| 35 | R8 的 C layout=`(64,8,1,1,0)`：输出 tile 几何为 8 行 x 64 列；最后的 0 表示 **linear 全局输出**。 |
| 36 | R32 的 A layout=`(32,32,4,1,1)`。 |
| 37 | R32 的 C layout=`(16,32,1,1,1)`：32 行 x 16 列；最后的 1 表示 **tiled 全局输出**。 |
| 38 | 空行。 |
| 39 | 声明 `run_case` 的输出类型模板参数和三个编译期 shape 参数 `CaseM/CaseN/CaseK`。 |
| 40 | A 的 layout 是编译期模板参数。 |
| 41 | B 的 layout 是编译期模板参数。 |
| 42 | C 的 layout 是编译期模板参数；结束模板参数列表。 |

布局构造函数定义于 `include/simma.h:47 / tensor_layout::tensor_layout`，字段依次是 `(tile_dim0,tile_dim1,supertile_shape0,supertile_shape1,tensor_format)`。在这里 dim0 是列/K/N 方向、dim1 是行/M 方向，不要按普通 `(rows,cols)` 反着理解。

#### 7.4.2 第 43 到 61 行：host 生成与量化 A/B

定位：`testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:43 / run_case`。

| 行号 | 代码作用 |
|---|---|
| 43 | 定义文件内部可见的 `static` 函数模板，返回整型状态码。 |
| 44 | 将输入格式固定为 `sifmt::mxint8`；A/B 共用这一个类型。 |
| 45 | 默认构造 host 随机引擎；没有使用设备随机数或 `random_device` 随机播种。 |
| 46 | 创建 `[-100,100)` 的 float 均匀分布。 |
| 47 | 分配包含 `CaseM*CaseK` 个 float 的 host vector，并初始化为 0；这里还不是 MXINT8 packed 数据。 |
| 48 | 遍历 A 的每个 float 元素，使用引用以便修改 vector 内容。 |
| 49 | 用随机分布生成一个值并写入 A。 |
| 50 | 结束 A 的随机填充循环。 |
| 51 | `make_shape(CaseM,CaseK)` 创建逻辑 `[M,K]` 的 `sipu::Shape<2>`。 |
| 52 | 创建带 `LayoutTag::Tiled` 标签的逻辑 layout 描述；此时只描述 shape/tag，不进行设备内存分配。 |
| 53 | `make_tensor<InputT>` 构造 host 侧 MXINT8 Tensor，按输入数据量化、生成 scale、分配并填充 tiled packed 存储。 |
| 54 | 空行。 |
| 55 | 分配 B 的 host float vector，共 `CaseN*CaseK` 个元素，所以 B 的逻辑形状是 `[N,K]`，不是 `[K,N]`。 |
| 56 | 遍历 B 的元素。 |
| 57 | 沿用同一个随机引擎继续产生 B 的值，没有重置引擎。 |
| 58 | 结束 B 的随机填充循环。 |
| 59 | 创建 B 的 `[N,K]` shape。 |
| 60 | 创建 B 的 tiled layout 描述。 |
| 61 | 构造 B 的 MXINT8 packed Tensor，与 A 一样在 host 完成。 |

一个容易忽略的细节：第 52/60 行没有传入 `LayoutA/LayoutB`，只传了 `LayoutTag::Tiled`。**host SiTe 的物理布局由 dtype 和 shape 自动推导，GEMM 的布局由前面显式模板参数指定；二者必须恰好一致，API 不会自动比较两份描述。** 本例的 `[8,1024]`、`[32,1024]`、`[576,1024]` 满足这种一致性，推导依据见第 7.5 节。

#### 7.4.3 第 63 到 81 行：golden 和存储大小

定位仍为 `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:43 / run_case`；第 62 行为空行。

| 行号 | 代码作用 |
|---|---|
| 63 | 开始声明 `gold_tiled_tensor`，由下一行函数的返回类型推导。 |
| 64 | `sipu::tensor::mm_mnk` 将量化后的 A/B 解量化为 float，在 host 计算 `A @ B^T`，返回 FP32、tiled 的 SiTe Tensor；不是启动另一个 SIPU GEMM。 |
| 65 | `Tensor::toVector` 按逻辑顺序导出 `[M,N]` 的 `std::vector<float>`，得到 linear golden。 |
| 66 | 创建 C 的逻辑 shape `[M,N]`。 |
| 67 | 创建 C 的 tiled layout 描述，供下面制作 tiled golden 使用；即使测试目标 C 是 linear，这一步仍会执行。 |
| 68 | 构造元素类型为 `OutputT` 的 tiled Tensor，将 FP32 golden 舍入成 BF16 或 FP16，并按对应输出格式打包。 |
| 69 | 按物理内存顺序导出 tiled golden。`true` 表示包含该接口暴露的 padding 项；返回值仍是 `vector<float>`，不是原始字节 buffer。 |
| 70 | 开始声明输出元素数。 |
| 71 | tiled 输出使用 `gold_tiled.size()`；linear 输出使用 `gold_linear.size()`，避免把逻辑元素数与含 padding 的物理元素数混为一谈。 |
| 72 | 将元素数乘以 `sizeof(OutputT)` 得到 C 的字节数；两种 OutputT 都是 16 bit。 |
| 73 | 空行。 |
| 74 | 用 `malloc` 分配 host 接收 C 的内存，再转成 `OutputT*`；这里没有检查分配失败。 |
| 75 | 空行。 |
| 76 | 声明设备 A 指针，尚未分配内存。 |
| 77 | 声明设备 B 指针。 |
| 78 | 声明设备 C 指针。 |
| 79 | 空行。 |
| 80 | 从 `mx_tensor_a.storageSize()` 获取 A 的实际 packed 存储字节数，包含 MX header/scale 和布局 padding。 |
| 81 | 同样获取 B 的实际 packed 字节数。 |

这里 `InputT* d_A/d_B` 只是保存设备地址的 C++ 指针类型，**不表示 packed buffer 就是连续的 `InputT` C++ 对象数组**。本例正确地使用 `storageSize()` 分配并整块拷贝，而不是用 `M*K*sizeof(InputT)` 估算存储。

#### 7.4.4 第 83 到 99 行：设备内存与 GEMM

定位仍为 `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:43 / run_case`；第 82 行为空行。

| 行号 | 代码作用 |
|---|---|
| 83 | `sipuMalloc` 按字节数分配设备 A buffer，并写入 `d_A`。 |
| 84 | 分配设备 B buffer。 |
| 85 | 分配设备 C buffer；没有初始化 C，当前纯 GEMM 不读取原 C。 |
| 86 | 空行。 |
| 87 | 将 A 的 host packed 原始字节拷到 SIPU，方向为 `sipuMemcpyHostToDevice`。 |
| 88 | 将 B 的 host packed 原始字节拷到 SIPU。 |
| 89 | 空行。 |
| 90 | 选择显式 layout、不带 workspace 的 `mma_dte<InputT,OutputT,OutputT,LayoutA,LayoutB,LayoutC>`。第二个 OutputT 是输出 dtype，第三个是 alpha/beta 的 scalarT。 |
| 91 | 传入 `OP_N,OP_N`；当前实现不使用这两个标志，实际存储契约是 A=`[M,K]`、B=`[N,K]`，计算乘以 B 的转置。 |
| 92 | 传入 M/N/K、`alpha=OutputT(0.1f)`、A 地址、`lda=K`。 |
| 93 | 传入 B 地址、`ldb=K`、`beta=OutputT(0.2f)`。 |
| 94 | 传入 C 地址、`ldc=N`，结束调用。没有传 stream，因此采用 `include/simma.h:180 / mma_dte` 声明中第 187 行的默认 `nullptr`。 |
| 95 | 空行。 |
| 96 | 将设备 C 同步拷回 `host_C`。SDK 对 device-to-host 的同步 memcpy 约定是拷贝完成后返回，所以后续 CPU 才能比较结果。 |
| 97 | 释放设备 A buffer。 |
| 98 | 释放设备 B buffer。 |
| 99 | 释放设备 C buffer；host_C 中的结果副本仍保留。 |

alpha/beta 的含义必须结合当前实现解释：`kernel/mma_dte_tiled_tensor.hpp:54 / mma_dte_implement_with_control` 第 69 行仍是未使用它们的 TODO，后续 dispatch 也不传这两个系数。因此此测试的 golden 是 `A @ B^T`，**不是 `0.1*(A @ B^T)+0.2*C_old`**。未来实现一般 alpha/beta 语义时，测试也要同步调整，不能仍保留这些非 1/0 系数却比较纯乘积。

这里所有 `sipuMalloc/sipuMemcpy/sipuFree` 的返回状态均未检查。这是现有测试源码的行为，不应当解读成分配或设备执行失败一定会在精度比较阶段被正确诊断。

#### 7.4.5 第 101 到 124 行：结果校验与清理

定位仍为 `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:43 / run_case`；第 100 行为空行。

| 行号 | 代码作用 |
|---|---|
| 101 | 错误计数置 0。 |
| 102 | 遍历 C 的所有待比较元素；本例不是抽样检查。 |
| 103 | 声明当前元素的期望值，类型为 OutputT。 |
| 104 | 编译期选择 tiled 或 linear golden 分支。 |
| 105 | tiled 输出从物理顺序的 `gold_tiled[i]` 取值并转换成 OutputT。 |
| 106 | linear 分支开始。 |
| 107 | linear 输出从逻辑顺序的 `gold_linear[i]` 取值并舍入为 OutputT。 |
| 108 | 结束分支。 |
| 109 | 将期望值转换成 float，供统一误差计算使用。 |
| 110 | 将设备输出的 OutputT 元素转换成 float。 |
| 111 | 开始计算误差。 |
| 112 | 当 golden 为 0 时使用绝对误差 `abs(result)`，避免除以 0。 |
| 113 | 其他情况下使用相对误差 `abs((gold-result)/gold)`。 |
| 114 | 阈值是 **0.0f**，不是 0.0076 等宽松容差；对有限数值，只要存在非零误差就计错。 |
| 115 | 错误计数加一。 |
| 116 | 打印错误下标和实际结果。 |
| 117 | 继续打印期望值和误差。 |
| 118 | 输出换行并刷新错误流。 |
| 119 | 结束当前错误分支。 |
| 120 | 结束比较循环。 |
| 121 | 空行。 |
| 122 | 释放 `malloc` 分配的 host_C。 |
| 123 | 错误数为 0 返回 0，否则返回 -1。 |
| 124 | 结束函数；局部 vector 和 SiTe Tensor 析构，后者释放各自拥有的 host packed 内存。 |

第 114 行是严格的**有限数值误差检查**，不等于完整的 bitwise 校验。特别是当计算得到 NaN 时，`NaN > 0.0f` 为 false，当前写法不会将它计错；FP16 输出溢出为 infinity 后的相减也可能产生 NaN。源码没有单独检查 `isfinite/isnan`，因此不能把返回 0 解读成所有异常浮点值都被正确检查。这里仅说明现状，不修改测试。

#### 7.4.6 第 126 到 146 行：两个 GTest 用例

| 相对路径:行号 / 函数或注册宏 | 逐行含义 |
|---|---|
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:126 / run_bf16_default_case` | 第 126 行定义 wrapper；第 127 行选择 BF16、M=8、N=576、K=1024；第 128 行传 R8 A/B/C layouts；第 129 行结束。 |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:131 / run_default_case` | 第 131 行定义 wrapper；第 132 行选择 FP16、M=32、N=576、K=1024；第 133 行传 R32 layouts；第 134 行结束。 |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:136`，类声明 | 定义空 fixture `MmaDteBf16Mxint8Test`，继承 `testing::Test`，没有自定义 Setup/TearDown。 |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:138 / TEST_F` | 第 138 行注册 `MmaDteBf16Mxint8Test.ComputesDefaultShape`；第 139 行断言 BF16 wrapper 返回 0；第 140 行结束宏生成的 test body。 |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:142`，类声明 | 定义第二个空 fixture `MmaDteFloat16Mxint8Test`。 |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:144 / TEST_F` | 第 144 行注册 `MmaDteFloat16Mxint8Test.ComputesDefaultShape`；第 145 行断言 FP16 wrapper 返回 0；第 146 行结束。 |

第 125、130、135、137、141、143 行都是空行。文件没有定义 `main`；`testcase/func_test/CMakeMmaDteCommon.cmake:352 / mma_dte_test_enable_gtest` 链接 `GTest::gtest_main` 来提供它。

### 7.5 sipu:: 类型与函数的定义在哪里

#### 7.5.1 头文件入口及完整直接调用清单

本例的 SiTe 头文件包含关系为：

```text
SDK/include/SiTe/SiTe.hpp:24  -> sifmt/sifmt.hpp
SDK/include/SiTe/SiTe.hpp:25  -> tensor/tensor.hpp
```

`SiTe.hpp` 第 21 行明确说明它是 V0 compatibility entry point；新的 `site.hpp/site::` 是另一套入口。不能搜索到 `site::make_shape` 就将它作为本例 `sipu::make_shape` 的实现。

下面路径均相对 **SDK 根目录 `/share_data/sicx_sdk/release/2609020046`**。表中前五行覆盖源码显式写出的全部 `sipu::` 名称，后面列出 `auto` 隐藏的返回类型和直接调用的成员方法：

| 源码中使用的名称 | SDK 相对路径:行号 / 函数或类型 | 实际含义 |
|---|---|---|
| `sipu::make_shape` | `include/SiTe/tensor/tensor.hpp:343 / make_shape` | 按参数个数构造 `Shape<Rank>`；本例两个参数，返回 `Shape<2>`，维度转换为 uint64_t |
| `sipu::LayoutTag::Tiled` | `include/SiTe/tensor/tensor.hpp:473`，枚举 `LayoutTag` | `Manual/Linear/Tiled` 标签；不是物理格式转换函数 |
| `sipu::make_layout(shape,tag)` | `include/SiTe/tensor/tensor.hpp:597 / make_layout` | 返回 `Layout(shape,tag)`，调用 `:498 / Layout::Layout` |
| `sipu::make_tensor<T>(layout,data)` | `include/SiTe/tensor/tensor.hpp:2483 / make_tensor` 起的重载组，另见 `:2489 / make_tensor`、`:2504 / make_tensor` | 包装 `Tensor<T,Layout>` 构造；第 2504 行版本还提供 QuantMode 模板参数。此处数据最终进入接收 float span 的 Tensor 构造路径 |
| `sipu::tensor::mm_mnk` | `include/SiTe/tensor/tensor.hpp:2679 / mm_mnk` | `sipu::tensor` 命名空间内的自由函数，不是 `Tensor` 成员；计算 A `[M,K]` 与 B `[N,K]` 的乘积，返回 FP32 Tensor |
| `shape_a/shape_b/shape_d` 的类型 | `include/SiTe/tensor/tensor.hpp:201`，类 `Shape` | 本例是 `Shape<2>`，继承 `MultiDimObj<2,uint64_t>` |
| `layout_a/layout_b/layout_d` 的类型 | `include/SiTe/tensor/tensor.hpp:481`，类 `Layout` | 本例为二维 Layout，默认没有显式 TilingConfig |
| `mx_tensor_a/mx_tensor_b/out_tensor_d/gold_tiled_tensor` 的类型 | `include/SiTe/tensor/tensor.hpp:1755`，类 `Tensor` | 持有 host 内存、逻辑 layout 和 `LayoutEngine<DataType,Layout>` |
| Tensor 的 host 数据构造 | `include/SiTe/tensor/tensor.hpp:1772 / Tensor::Tensor` | 根据 LayoutEngine 计算字节数、malloc，再调用 `fillWithContainer` |
| `toVector()` | `include/SiTe/tensor/tensor.hpp:2114 / Tensor::toVector` | 按逻辑元素顺序返回 `vector<float>`；MX 类型会解量化 |
| `toVectorAsMemoryOrder(true)` | `include/SiTe/tensor/tensor.hpp:2141 / Tensor::toVectorAsMemoryOrder` | 按物理数据存储顺序返回值，并按参数处理 padding；不是返回 raw header/payload 字节 |
| `storageSize()` | `include/SiTe/tensor/tensor.hpp:2238 / Tensor::storageSize` | 返回 LayoutEngine 的物理字节数 |
| `data()` | `include/SiTe/tensor/tensor.hpp:2243 / Tensor::data` | 返回 `mHostData`，即 host packed buffer 的地址，不是 device pointer |
| 局部 Tensor 析构 | `include/SiTe/tensor/tensor.hpp:1796 / Tensor::~Tensor` | 释放其拥有的 host 内存 |

同文件 `:2534` 开始 `namespace tensor`，所以应写 `sipu::tensor::mm_mnk`；类成员则是 `sipu::Tensor<...>::toVector` 等。这两种 `tensor/Tensor` 不是同一标识符。

#### 7.5.2 自动 layout 与 MXINT8 打包的真实实现

SDK `include/SiTe/tensor/tensor.hpp:1235 / LayoutEngine::LayoutEngine` 根据 dtype、shape、tag 构造物理 layout；第 1276 行起在没有显式 TilingConfig 时调用：

| SDK 相对路径:行号 / 成员函数 | 本例中的结果 |
|---|---|
| `include/SiTe/tensor/tensor.hpp:1097 / LayoutEngine::getTileShape` | MXINT8 `[8,1024]` -> tile `(8,128)`；`[32,1024]` 和 `[576,1024]` -> tile `(32,32)`。这是 SiTe 的 `(row,column)` 次序 |
| `include/SiTe/tensor/tensor.hpp:1166 / LayoutEngine::getSuperTileShape` | 本例 MXINT8 都采用 SiTe supertile `(1,4)`，对应 DTE API 的 `(supertile_shape0,supertile_shape1)=(4,1)` |
| `include/SiTe/tensor/tensor.hpp:1417 / LayoutEngine::storageSize` | tiled 情况按 supertile 数量乘以每个 supertile 的物理存储大小 |
| `include/SiTe/tensor/tensor.hpp:1029 / SuperTile::storageSize` | 将 header 存储与数据存储相加；MXINT8 一个 supertile 是四个 1024-byte payload，加 256-byte header |

SDK `include/SiTe/tensor/tensor.hpp:1950 / Tensor::fillWithContainer` 第 2015 行起逐行、逐量化 block 遍历 float 输入；第 2040 行构造 MX block，第 2041/2042 行获取 scale/data，第 2050 行调用 `fillBlock` 写入对应的 packed 位置。这是在 **host 上量化与排列**，并没有调用 vLLM 的 `hp_to_mx` 设备量化算子。

进一步的 MX block 类型定义位于 SDK `include/SiTe/sifmt/sifmt_mx.hpp:115`，类型别名 `sifmt::mxint8`；其构造函数 `include/SiTe/sifmt/sifmt_mx.hpp:46 / SiMxData::SiMxData` 调用量化 policy，将数据与 scale 分开产生。`sifmt` 是另一个命名空间，不是 `sipu`。

对本例无额外 shape padding 的输入，由这些存储规则可算出：

| buffer | 元素数量 | 物理字节数 |
|---|---:|---:|
| R8 A `[8,1024]` MXINT8 | 8192 | 8704 |
| R32 A `[32,1024]` MXINT8 | 32768 | 34816 |
| 两个用例的 B `[576,1024]` MXINT8 | 589824 | 626688 |
| R8 C `[8,576]` BF16 linear | 4608 | 9216 |
| R32 C `[32,576]` FP16 tiled | 18432 | 36864 |

这些值解释了为什么测试必须调用 `storageSize()`，以及为什么逻辑元素数不等于 MX buffer 字节数。

#### 7.5.3 mm_mnk 是 CPU golden，不是库内设备 GEMM

SDK `include/SiTe/tensor/tensor.hpp:2679 / mm_mnk` 的核心步骤：

1. 第 2698/2699 行调用 A/B 的 `Tensor::toVector`，先获得量化后数据的解量化 float 值。
2. 第 2702/2703 行构造 C=`[M,N]` 和调用者指定的 `layoutTag`；本例传 Tiled。
3. 第 2707 行起执行三重 CPU 循环，`idA=i*K+k`、`idB=j*K+k`，第 2716 行做 float 乘加，因此是 `A @ B^T`。
4. 第 2722 行构造 `Tensor<sifmt::float32,...>` 返回。

函数前的旧注释写着“always ... linear”，但第 2703 行实际使用参数 `layoutTag`。本例返回 **FP32 tiled Tensor** 才是按实现得到的结论，不能只照抄该注释。

#### 7.5.4 其他容易混淆的 SDK 符号

以下不是 `sipu::` 命名空间成员，但测试确实调用或使用了它们，因此一并列出：

| 名称 | SDK 相对路径:行号 / 函数或类型 | 说明 |
|---|---|---|
| `sifmt::float16`、`sifmt::bfloat16` | `include/SiTe/sifmt/sifmt_fp.hpp:277`、`:278`，类型别名 | 分别使用 `SiFpBase<5,10,...>` 和 `SiFpBase<8,7,...>` |
| `OutputT(float)` | `include/SiTe/sifmt/sifmt_fp.hpp:85 / SiFpBase::SiFpBase` | 将 float 转换到相应 16-bit 浮点存储 |
| `static_cast<float>(OutputT)` | `include/SiTe/sifmt/sifmt_fp.hpp:113 / SiFpBase::operator InterFpType` | 将存储值转换回内部 float 表示 |
| `sipuMalloc(T**,bytes)` | `include/sipurt/sipu_runtime.h:560 / sipuMalloc` | 全局 C++ 模板 wrapper，转调 `void**` 的 C runtime API |
| `sipuMalloc(void**,bytes)` | `include/sipurt/sipu_runtime_api.h:3540 / sipuMalloc` | runtime 外部函数声明，实际链接到 SIPU runtime；这里不是头文件内的设备分配实现 |
| `sipuMemcpy(...)` | `include/sipurt/sipu_runtime_api.h:4661 / sipuMemcpy` | 全局 runtime API 声明；该头第 45 行说明同步 D2H 拷贝完成后返回 |
| `sipuFree(...)` | `include/sipurt/sipu_runtime_api.h:3752 / sipuFree` | 全局 runtime API 声明 |
| `sipuMemcpyHostToDevice/DeviceToHost` | `include/sipurt/driver_types.h:1090`，枚举 `sipuMemcpyKind` | 第 1093/1094 行分别为两个方向值，不是函数 |

这些 runtime 头给出公开声明，不能把声明位置冒称为 `.so` 内部实现源码的位置；本例的实际动态依赖包含 `libsipurt.so.0`。

### 7.6 本例 mma_dte 的实现和实例化在哪里

#### 7.6.1 声明、模板定义、实例化是三个不同位置

| 层次 | DTE 相对路径:行号 / 函数 | 作用 |
|---|---|---|
| 公共声明 | `include/simma.h:180 / mma_dte` | 让 host `.cpp` 知道函数签名；该文件没有此函数体 |
| 模板定义 | `kernel/mma_dte_tiled_tensor.hpp:193 / mma_dte` | 不带 workspace 的实际 host API 包装层，转调内部实现 |
| 本测试的 BF16/R8 实例化 | `testcase/func_test/test_bf16_mxint8_host/kernel/inst_bf16_mxi8.su:24 / mma_dte` | 产生 `mxint8,bfloat16,bfloat16,A(128,8,4,1,1),B(32,32,4,1,1),C(64,8,1,1,0)` 的实现 |
| 本测试的 FP16/R32 实例化 | `testcase/func_test/test_bf16_mxint8_host/kernel/inst_bf16_mxi8.su:31 / mma_dte` | 产生 `mxint8,float16,float16,A(32,32,4,1,1),B(32,32,4,1,1),C(16,32,1,1,1)` 的实现 |

`.su` 第 18 到 20 行包含 R32/R16/R8 计算 helper 头，第 21 行包含定义 `mma_dte` 的 `.hpp`，然后用 `template void ...` 显式实例化。**包含 R16 helper 头不代表这个测试就实例化了 R16 的公开 `mma_dte` 入口**；入口清单是第 24/31 行的两个组合。

本例不是通过 `kernel/instantiations/mxint8/*.su` 那套生产实例化文件获取定义。它使用 `testcase/.../kernel/inst_bf16_mxi8.su` 单独编译；源级模板定义可以相同，但最终承载定义的目标文件不同。

#### 7.6.2 从 host API 到 device kernel

```text
kernel/mma_dte_tiled_tensor.hpp:193 / mma_dte
  -> kernel/mma_dte_tiled_tensor.hpp:165 / mma_dte_implement
  -> kernel/mma_dte_tiled_tensor.hpp:151 / mma_dte_implement
  -> kernel/mma_dte_tiled_tensor.hpp:54 / mma_dte_implement_with_control
       -> kernel/detail/mma_dte_tiled_tensor_planner.hpp:2615 / get_best_solution
       -> kernel/detail/mma_dte_tiled_tensor_tensor_map.hpp:71 / create_tensor_maps
       -> R8:  kernel/detail/mma_dte_tiled_tensor_dispatch_entrypoints.hpp:1211 / mma_r8_dte
       -> R32: kernel/detail/mma_dte_tiled_tensor_dispatch_entrypoints.hpp:1436 / mma_r32_dte
       -> dispatch 宏展开后的 <<<grid,cluster,block,0,stream>>> 启动
```

`mma_dte_implement_with_control` 第 102 行根据 `layoutA.tile_dim1` 做编译期分流，BF16 用例选择 R8、FP16 用例选择 R32。`get_best_solution` 使用 M/N/K 为本次调用选 runtime schedule/chunk/stage 等参数。

设备启动宏定义见 `kernel/detail/mma_dte_tiled_tensor_dispatch_entrypoints.hpp:198 / CALL_PROCON_V2_KERNEL`（宏），第 206 到 211 行启动的模板函数体位于 `kernel/mma_dte_tiled_tensor_kernel.hpp:1546 / kernel_tile_mma_procon_v2_tiled_tensor_map`。

这一 kernel 的第 1726 行起按 `tensor_format_out` 选择 `store_tile` 或 `store_linear`。MXINT8 真正的 tile load/MMA/store 实现位于 helper 成员函数中，例如：

- `kernel/util/mma_dte_tiled_tensor_kernel_util_mxint8_mxfp6_r8.hpp:79 / loadL2B_mma_r8_1m_1n_impl::load_mma_1st_stp`：R8 的 MXINT8 MMA 指令；`:307 / loadL2B_mma_r8_1m_1n_impl::store_linear`：对应 linear 写回。
- `kernel/util/mma_dte_tiled_tensor_kernel_util_mxint8_mxfp6_r32.hpp:6086 / loadL2B_mma_r32_1m_1n_impl::load_mma_1st_stp`：R32 的 MXINT8 MMA；`:6329 / loadL2B_mma_r32_1m_1n_impl::store_tile`：对应 tiled 写回。

这里列的是 helper 变体实例，**没有声称本次未执行的两个 shape 在 runtime 必然选择 1m_1n**。具体选择需看 planner 结果或运行时 dispatch 日志。已有 `.so` 的启动 stub 则可以确认这份构建收录的 kernel 主体家族是 `kernel_tile_mma_procon_v2_tiled_tensor_map`。

### 7.7 CMake 如何构建设备代码并链接测试

#### 7.7.1 当前 --test 入口不依赖先构建整个主库

`build-modify.sh:105 / build_testcases` 用 `cmake -S testcase -B build/testcase` 配置聚合测试工程，再构建 `mma_dte_all_testcases`。当前脚本第 120/121 行的 `test` 分支只调用 `build_testcases`；只有 `all` 分支才先调用 `build_library`。

`testcase/CMakeLists.txt:131 / mma_dte_add_testcase_project` 第 173 行使用 `ExternalProject_Add`，为每个 testcase CMake 工程创建独立子构建目录，公共 runtime 输出目录按 SDK/架构/profile/lane 隔离。它不是把所有测试与所有设备代码揉成一个大 executable。

`BUILD_CXX=/usr/bin/c++` 经 `build-modify.sh:67` 附近脚本逻辑传为 `-DCMAKE_CXX_COMPILER=/usr/bin/c++`，影响 host `.cpp` 编译和 host `.so`/executable 链接；**不会把 `.su` 的 SIPU device 编译替换为普通 g++**，`.su` 仍由 SDK 的 `scc` 处理。

#### 7.7.2 本测试 CMake 的局部依赖图

以下均为 `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt` 的命令：

| 相对路径:行号 / CMake 命令 | 实际效果 |
|---|---|
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:12 / find_package` | 找到 SDK 的 scc CMake 支持函数 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:24 / mma_dte_test_setup_compile_options` | 设置本测试设备编译选项，其中包含 `-fPIC`、`-DDTE_DEV_MODE` 等 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:41 / set` | 默认 `TEST_INST_SRC` 指向测试专用 `kernel/inst_bf16_mxi8.su`，也允许 cache override |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:43 / scc_add_library` | 为这一个 `.su` 建立 OBJECT 编译目标 `tile_mma_dte_mxint8` |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:50 / scc_target_get_output` | 取出 OBJECT 目标的实际 `.o` 文件列表，而不是取主库路径 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:52 / add_custom_command` | 用 `${CMAKE_CXX_COMPILER} -shared` 把该 `.o` 链成子构建目录内的 `libtile_mma_dte_mxint8.so` |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:60 / add_custom_target` | 把局部 `.so` 纳入构建依赖 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:66 / set` | `TEST_HOST_SRC` 默认就是本题的 `.cpp` |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:67 / add_executable` | 由普通 C++ 编译器编译 host 测试源码并建立 executable 目标 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:74 / target_link_libraries` | 链接局部 `.so`、`sipurt`、`sipu`，没有列出主库 `libtile_mma_dte.so` |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:79 / add_dependencies` | 保证 executable 构建依赖局部 `.so` 目标 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:81 / mma_dte_test_enable_gtest` | 链接 GTest 和 GTest main，注册测试发现流程 |

公共 GTest helper 的定义是 `testcase/func_test/CMakeMmaDteCommon.cmake:352 / mma_dte_test_enable_gtest`，它使用 `DISCOVERY_MODE PRE_TEST`。公共 SDK include 收集函数为 `cmake/MmaDteConfig.cmake:19 / mma_dte_collect_sdk_include_directories`，包含 `$SI_SDK_ROOT/include/SiTe` 等目录，解释了为什么 `#include "SiTe.hpp"` 能找到 SDK 文件。

公共文件虽然定义了 `MMA_DTE_REUSE_MAIN_OBJECTS` 和 `testcase/func_test/CMakeMmaDteCommon.cmake:209 / mma_dte_get_main_object`，但**本测试的上述 CMake 没有调用该复用函数**。不能因为 cache 中显示 `AUTO`，就说此测试复用了生产 object 或链接了主库。实际设置仍是第 43 行直接编译测试专用 `.su`。

另外，`DTE_DEV_MODE` 会跳过 `kernel/mma_dte_tiled_tensor.hpp:54 / mma_dte_implement_with_control` 第 64 行起的一部分生产 layout 静态断言；它不意味着跳过所有 runtime shape 检查。因此这套测试构建与生产实例化的编译选项也不能自动等同。

#### 7.7.3 scc_add_library(OBJECT) 的 SDK 实现

本次日志实际使用 **SDK-LOG** `share/cmake/scc/sccConfig.cmake:311 / scc_add_library`：

1. 第 347 行起识别 OBJECT 类型。
2. 第 426 行构造 `_SipuObj/<target>/<relative-source>.o` 输出路径。
3. 第 431 行起建立 `${SCC_EXECUTABLE} ... -c -o ...` 的逐源文件编译命令。
4. 第 447 行起为 OBJECT 模式创建依赖 `.o` 的 custom target，不在这个函数里生成 `.so`。

**SDK-LOG** `share/cmake/scc/sccConfig.cmake:876 / scc_target_get_output` 第 890 行起返回 `SIPU_OBJECT_FILES`。之后生成局部 `.so` 是 DTE 测试 CMake 自己的第 52 行 custom command 完成的。

在题目指定 **SDK** `2609020046` 中，对应函数分别位于 `share/cmake/scc/sccConfig.cmake:315 / scc_add_library`、`:884 / scc_target_get_output`。这里明确区分两个版本的行号。

#### 7.7.4 为什么普通 c++ 能链接含 SIPU device code 的 .o

`scc -c` 产生的 `.su.o` 不是单纯的 SIPU 指令流。现有 object 的 ELF header 是 **x86-64 relocatable object**，里面同时有 host launch/dispatch 代码、注册代码、设备 kernel metadata，以及嵌入的设备 fatbin。普通 `/usr/bin/c++` 可以按 host ELF 规则把它链接成 `.so`，设备程序作为 ELF section 被保留，而不是由 x86 linker 把 SIPU 指令变成 x86 指令。

编译器仓库中的机制对应如下，路径相对 **COMPILER** 根目录：

| 相对路径:行号 / 成员函数 | 作用 |
|---|---|
| `llvm-project/clang/lib/CodeGen/CGCUDANV.cpp:602 / CGNVCUDARuntime::emitDeviceStubBodySIPU` | 生成 host 侧 kernel 启动 stub，打包参数；第 923 行起生成 `sipuLaunchKernelExC` 调用 |
| `llvm-project/clang/lib/CodeGen/CGCUDANV.cpp:1195 / CGNVCUDARuntime::makeModuleCtorFunction` | SIPU 分支第 1290 行起设置 `__sipu_fatbin`、`.sipuFatBinSegment` 等内容；第 1449 行起生成 fatbin 注册，之后注册 kernels |
| `llvm-project/clang/lib/CodeGen/CGCUDANV.cpp:1040 / CGNVCUDARuntime::makeRegisterGlobalsFn` | 为编译产生的各个 kernel 生成 runtime 注册调用，把 host 入口与设备 kernel 名称联系起来 |

因此程序执行时的大体过程是：动态加载局部 `.so`，执行其模块初始化/注册逻辑；host 调用 `mma_dte` 做调度；相应 host stub 经 SIPU runtime 启动 fatbin 中的设备 kernel。**不是 executable 的 x86 CPU 直接执行 fatbin 内的 SIPU 指令。** 这里用当前编译器源码解释机制，不宣称该工作树 revision 与日志里 SDK 编译器二进制完全一致；已有 ELF 的 section 和外部 runtime 符号是额外的独立证据。

### 7.8 temp/cxx.log 与已有产物的交叉证据

#### 7.8.1 旧日志确实记录了局部编译和动态链接

以下是 DTE `temp/cxx.log` 的原始行号，日志不是函数定义；为便于阅读省略了长绝对路径和无关 flags：

| 日志位置 | 记录的事实 |
|---|---|
| `temp/cxx.log:11`、`:15`、`:18` | 旧构建目录是 `build-cxx`，模式为 `all`，SDK 为 `2609161442` |
| `temp/cxx.log:44` | 旧构建先调用 `build_library`；这解释了前面为什么出现大量主库编译输出，不代表测试链接了这些全部产物 |
| `temp/cxx.log:6117` | `scc -arch=sipu_150 -c .../inst_bf16_mxi8.su -o .../inst_bf16_mxi8.su.o`，仅编译这个测试的实例化源文件 |
| `temp/cxx.log:6127` | `/usr/bin/c++ -shared -o .../libtile_mma_dte_mxint8.so .../inst_bf16_mxi8.su.o` |
| `temp/cxx.log:6140` | `/usr/bin/c++ ... test_bf16_mxi8_host.cpp.o ... -ltile_mma_dte_mxint8 -lsipurt -lsipu ... libgtest.a libgtest_main.a` |
| `temp/cxx.log:6142` | `Built target test_bf16_mxi8_host`，仅表示该次构建完成 |

旧日志模式是 `all`，当前 `build-modify.sh --test` 的源码语义却是 test-only，应以第 7.7.1 节的当前脚本为准。本节不推测旧脚本当时如何解析具体命令行，只陈述日志的 `BUILD_MODE=all`。

#### 7.8.2 当前找到的 executable 与局部共享库

本次找到的产物位置相对 DTE 根目录为：

```text
executable:
build/testcase/2609161442/sipu_150/default/specialized/test_bf16_mxi8_host

局部 .so:
build/testcase/_build/2609161442/sipu_150/default/specialized/func_test/test_bf16_mxint8_host/libtile_mma_dte_mxint8.so

设备编译产生的 host object:
build/testcase/_build/2609161442/sipu_150/default/specialized/func_test/test_bf16_mxint8_host/_SipuObj/tile_mma_dte_mxint8/kernel/inst_bf16_mxi8.su.o
```

当前产物子构建目录在 `build/testcase/_build/...`，旧日志中的子构建目录在 `build-cxx/testcase/_build/...`；不是把两份路径当作同一次构建。现有子目录的 `CMakeFiles/test_bf16_mxi8_host.dir/link.txt:1` 再次确认相同的局部 `.so` 链接方式，`CMakeCache.txt:255` 的 `TEST_INST_SRC` 仍为上述测试专用 `.su`。

只读检查得到：

| 检查对象与工具 | 关键结果 | 可以推出什么 |
|---|---|---|
| executable，`readelf -dW` | `DT_NEEDED` 包含 `libtile_mma_dte_mxint8.so` 和 `libsipurt.so.0`；没有主库 `libtile_mma_dte.so`；RPATH 含对应局部 `.so` 目录 | 测试动态依赖局部库，不是把主库整体静态嵌进自身 |
| executable，`nm -D -C` | 两个 `mma_dte<...>` 都显示 `U` | 这是由共享库满足的外部引用，不是 executable 自身定义的实现；`U` 不等于链接失败 |
| executable，`readelf -SW` | 没有发现 `__sipu_fatbin` section | 这份 executable 自身没有承载本例设备镜像 |
| 局部 `.so`，`nm -D -C --defined-only` | 两个公开 `mma_dte<...>` 定义，显示 `W` | 显式模板实例化的实现位于该 `.so`；weak 定义不意味着没有实现 |
| 局部 `.so`，`readelf -SW` | 有 `.sipu_kernel_meta`、`__sipu_fatbin`、`.sipuFatBinSegment` | 设备镜像实际随局部 `.so` 部署 |
| `.su.o`，`readelf -hW/-SW` | x86-64、`REL`；已经有同样的 SIPU metadata/fatbin section | 设备镜像在 SCC 编译阶段已进入 host object，后面的 c++ shared link 是封装/链接 |
| 局部 `.so`，`nm -D` | 外部引用包含 `__sipuRegisterFatBinary`、`__sipuRegisterFunction`、`__sipuUnregisterFatBinary`、`sipuLaunchKernelExC` | 与编译器生成注册和启动逻辑的解释相互印证 |

CMake 命令里有 `-lsipu`，但 executable 的直接 `DT_NEEDED` 列表中未必保留它；当前链接命令带 `--as-needed`。不要把“命令行写了一个库”和“ELF 一定直接依赖这个库”当成完全相同的概念。

#### 7.8.3 为什么是 2 个公开入口，却有 301 个启动 stub

本次 `nm -D -C --defined-only` 的统计结果是：

```text
void mma_dte<...> 定义数：2
void __device_stub__kernel_tile_mma_procon_v2_tiled_tensor_map<...> 定义数：301
```

原因不是把其他类型的主库全打包进来了，而是一个公开 `mma_dte` 实例本来就带运行时选择逻辑：M/N/K 是它的普通 `int` 参数，不是其模板参数。即使 `run_case` 的 CaseM/CaseN/CaseK 是编译期常量，也没有让另一个翻译单元中的显式 `mma_dte` 自动退化成“只保留当前 shape 恰好运行的一个 kernel”。

`kernel/detail/mma_dte_tiled_tensor_dispatch_entrypoints.hpp:198 / CALL_PROCON_V2_KERNEL` 会按不同 NUM_STAGE、CHUNK_M/N/K、REMAINDER_M/N 组合实例化底层 kernel；`kernel/detail/mma_dte_tiled_tensor_dispatch_entrypoints.hpp:1211 / mma_r8_dte`、`:1436 / mma_r32_dte` 在运行时选择其中适合本次 shape 的路径。因此需要区分四层：

| 层次 | 本例结论 |
|---|---|
| 整个 DTE 仓库的所有输入类型、输出类型和布局 | **没有全部链接进这个测试局部库** |
| 本测试 `.su` 指定的公开 API 组合 | 两个：MXINT8->BF16/R8/linear，MXINT8->FP16/R32/tiled |
| 这两个组合可调度的 device kernel 模板变体 | 多个；现有局部库可见 301 个 kernel launch stub，而不是只两个 |
| 单次运行真正选中并启动的变体 | 由 planner/dispatch 决定，不能将“编进库里”与“这次执行了”混为一谈 |

统计的是 host 侧 device launch stub 定义数，**不是宣称做过运行时 301 次 kernel 启动追踪**。编译 profile、SDK、架构、SplitK 或 generic-tail 等设置变化后，数量也可能变化。

### 7.9 最终回答与使用边界

**`make_shape/make_layout/make_tensor` 等定义主要在 SDK 的 `include/SiTe/tensor/tensor.hpp`；本例用 SiTe 在 host 上准备 MXINT8 packed 数据和 CPU golden。`mma_dte` 的函数模板定义在 `kernel/mma_dte_tiled_tensor.hpp:193 / mma_dte`，具体两个实现由测试专用 `kernel/inst_bf16_mxi8.su` 显式实例化。CMake 将它编成局部 `.o`，再链接成 `libtile_mma_dte_mxint8.so`，测试 executable 动态链接该库。设备代码位于该 `.so` 中，是这两个 API 组合的 dispatch 变体集合，不是整个 DTE 主库全部 device code，也不只是两个叶子 kernel。**

因此，只拷贝 `test_bf16_mxi8_host` 文件并不能保证换目录后独立运行，还需要这份局部 `.so` 和兼容的 SIPU runtime，并确保动态库搜索路径有效。

本次旧日志和已存在产物使用 `2609161442`，编译 flags 还出现 `siinfer_dev` 路径；不能据此宣称题目指定的 `2609020046/vllm_dev` 组合已完成构建或设备测试。这里的源码解释已按指定 SDK 定位，动态链接结论则由旧日志和当前 ELF 分别验证，二者的范围没有混用。

## 8. MXINT8 Host 测试 CMake 逐行说明与三个关键命令（2026-09-17）

### 8.1 范围与结论

本节依据本次读取的文件，不沿用旧版本的行号。根目录约定如下：

| 标记 | 相对路径所对应的根目录 |
|---|---|
| **DTE** | `/share/users/like/package/sikernel/mma_dte_tile_tensor`，以下未加标记的项目源码路径均相对该目录 |
| **SDK** | `/share_data/sicx_sdk/release/2609151958`，题目指定版本 |
| **CMAKE** | `/share_data/users/like/package/h100/package/cmake/github/cmake-4.2.0-rc2-linux-x86_64/share/cmake-4.2`，日志使用的 CMake 安装所带模块和文档 |
| **GTEST** | `/usr/src/googletest`，本机 GoogleTest 源码 |

**版本区别：**`temp/cxx.log.nodir:15` 记录的是 `BUILD_MODE=test`；`:18`、`:26` 和 `:5781` 显示该次实际 SDK 是 **2609161442**，不是题目指定的 **2609151958**。本节以 2609151958 定位 SCC 函数；只读比较确认这两个 SDK 的 `share/cmake/scc/sccConfig.cmake` 内容完全相同，所以这里的 SCC CMake 行号和行为也一致。不能由此推断两个 SDK 的编译器、头文件或运行库全部相同。

三个重点命令的直接答案：

| 调用位置 / 函数名或 CMake 命令名 | 本例作用 |
|---|---|
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:50 / scc_target_get_output` | 读取 SCC target 保存的输出路径，把本例 `.su.o` 的绝对路径写入变量 `TILE_MMA_DTE_MXINT8_O`；**不执行编译或链接** |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:60 / add_custom_target` | 创建名为 `tile_mma_dte_mxint8_so` 的构建入口，使默认构建检查并按需生成局部 `.so`；**这个 target 本身不是共享库文件，也不是原生 SHARED library target** |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:81 / mma_dte_test_enable_gtest` | 找到并链接 GoogleTest 与默认 `main()`，通过 `gtest_discover_tests(... DISCOVERY_MODE PRE_TEST)` 接入 CTest；**不在这一行立即运行 GEMM 测试** |

完整构建结果仍是：**测试专用 `.su` -> SCC 编译 `.o` -> 普通 C++ 编译器链接测试专用 `.so` -> host executable 动态链接该 `.so`。**不是链接整个主库 `libtile_mma_dte.so`。

### 8.2 scc_target_get_output 到底取出什么

调用点是 `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:50 / scc_target_get_output`：

```cmake
scc_target_get_output(tile_mma_dte_mxint8 TILE_MMA_DTE_MXINT8_O)
```

这里两个参数的身份不同：`tile_mma_dte_mxint8` 是已经定义的 target 名；`TILE_MMA_DTE_MXINT8_O` 是接收结果的**变量名**，不是文件名，也不是另一个 target。传入变量名而不是 `${TILE_MMA_DTE_MXINT8_O}`，是为了让函数给调用者设置这个变量。

**SDK** `share/cmake/scc/sccConfig.cmake:876 / scc_target_get_output` 的实际步骤：

| 实现位置 / 函数名 | 行为 |
|---|---|
| `share/cmake/scc/sccConfig.cmake:878 / scc_target_get_output` | 检查 target 参数非空 |
| `share/cmake/scc/sccConfig.cmake:883 / scc_target_get_output` | 检查输出变量名非空 |
| `share/cmake/scc/sccConfig.cmake:888 / scc_target_get_output` | 用 `get_target_property` 读取 target 的 `SIPU_LIBRARY_TYPE` 属性 |
| `share/cmake/scc/sccConfig.cmake:891 / scc_target_get_output` | 如果类型是 `OBJECT`，进入本例使用的分支 |
| `share/cmake/scc/sccConfig.cmake:892 / scc_target_get_output` | 从 `SIPU_OBJECT_FILES` 属性读取 object 路径列表 |
| `share/cmake/scc/sccConfig.cmake:893 / scc_target_get_output` | 属性为空时报错；这里不是逐个检查 `.o` 文件是否已经存在 |
| `share/cmake/scc/sccConfig.cmake:899 / scc_target_get_output` | `set(${OUTPUT_VARIABLE} ${OBJ_FILES} PARENT_SCOPE)`：把列表写回调用者作用域，不是写入 Cache |
| `share/cmake/scc/sccConfig.cmake:905 / scc_target_get_output` | 非 OBJECT 分支读取 `SIPU_OUTPUT_FILE`，再在第 912 行写回；可用于取共享库、静态库或 executable 的单个输出路径 |

**这些路径为什么在编译之前就已知？**

**SDK** `share/cmake/scc/sccConfig.cmake:311 / scc_add_library` 在 CMake 配置阶段就构造了输出路径，并创建将来执行的编译规则：

| 实现位置 / 函数名 | 本例结果 |
|---|---|
| `share/cmake/scc/sccConfig.cmake:347 / scc_add_library` | 识别调用参数 `OBJECT`，设置 `LIBRARY_TYPE=OBJECT` |
| `share/cmake/scc/sccConfig.cmake:426 / scc_add_library` | 根据 source 相对路径，构造 `${CMAKE_CURRENT_BINARY_DIR}/_SipuObj/${TARGET}/${REL_SOURCE_SAFE}.o` |
| `share/cmake/scc/sccConfig.cmake:431 / scc_add_library` | 建立 `scc ... -c ...` 的 `add_custom_command(OUTPUT ...)`；第 439、440 行同时记录源文件依赖和编译器生成的 `.d` 依赖文件 |
| `share/cmake/scc/sccConfig.cmake:449 / scc_add_library` | OBJECT 模式实际创建的是 `add_custom_target(${TARGET} ALL DEPENDS ${OBJECT_FILES})` |
| `share/cmake/scc/sccConfig.cmake:492 / scc_add_library` | 保存 `SIPU_SOURCE_FILES`、`SIPU_OBJECT_FILES`、`SIPU_COMPILER_OPTIONS`、`SIPU_LIBRARY_TYPE` 等自定义 target 属性 |

所以，本例默认 source 只有一个时，返回的列表只有一个元素：

```text
TILE_MMA_DTE_MXINT8_O =
  <当前测试子项目的 build 目录>/_SipuObj/tile_mma_dte_mxint8/kernel/inst_bf16_mxi8.su.o
```

它返回的是**约定的输出位置**，不是通过扫描磁盘找到已经编好的文件。未来 target 含多个源文件时，这个变量可以是多个 `.o` 路径组成的 CMake 列表，后面的 `${TILE_MMA_DTE_MXINT8_O}` 会把它们作为多个参数展开。

还要注意：本 SDK 的 `scc_add_library(... OBJECT ...)` **不是**对 CMake 原生 `add_library(... OBJECT ...)` 的简单转发。因此应使用 SDK 的 `scc_target_get_output` 取它的 object 路径，不能把它直接当成原生 OBJECT library，照搬 `$<TARGET_OBJECTS:tile_mma_dte_mxint8>`。

### 8.3 add_custom_command、add_custom_target 与 add_dependencies 的分工

本例相关代码分别在：

- `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:52 / add_custom_command`：定义“如何生成 `.so`”的文件生成规则。
- `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:60 / add_custom_target`：创建可被默认构建或其他 target 请求的入口。
- `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:79 / add_dependencies`：让 host executable 的构建依赖上述入口。

#### 8.3.1 add_custom_command 负责实际的链接命令

```cmake
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libtile_mma_dte_mxint8.so
    COMMAND ${CMAKE_CXX_COMPILER} -shared -o ${CMAKE_CURRENT_BINARY_DIR}/libtile_mma_dte_mxint8.so
        ${TILE_MMA_DTE_MXINT8_O}
    DEPENDS tile_mma_dte_mxint8 ${TILE_MMA_DTE_MXINT8_O}
    COMMENT "Linking libtile_mma_dte_mxint8.so"
)
```

这段是在配置时**登记规则**。真正执行 `${CMAKE_CXX_COMPILER}` 是之后的 build 阶段。本例由脚本选中 `/usr/bin/c++` 后，核心命令就是：

```text
/usr/bin/c++ -shared -o <build>/libtile_mma_dte_mxint8.so <build>/_SipuObj/.../inst_bf16_mxi8.su.o
```

`testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:56 / add_custom_command` 同时写 target 和 object 文件，目的不同：

| DEPENDS 项 | 作用 |
|---|---|
| `tile_mma_dte_mxint8` | target 级依赖，保证 SCC object 构建目标先完成 |
| `${TILE_MMA_DTE_MXINT8_O}` | 文件级依赖，object 更新后使 `.so` 需要重新链接 |

这不是把同一件事重复写两次。尤其本例的 SCC OBJECT target 实际是 custom target，没有一个原生 library 输出可供 CMake 自动当作文件依赖，所以显式列出 `.o` 很重要。CMake 通用规则对应 **CMAKE** `Help/command/add_custom_command.rst:163 / add_custom_command` 的 `DEPENDS` 定义。

#### 8.3.2 add_custom_target 提供构建入口，不等于每次重链

```cmake
add_custom_target(tile_mma_dte_mxint8_so ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtile_mma_dte_mxint8.so
)
```

`testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:60 / add_custom_target` 的含义是：

1. 创建 target `tile_mma_dte_mxint8_so`。它是构建图中的节点，不会产生同名 executable，也不会自动按名字创建 `.so`。
2. `ALL` 把它加入**当前子项目的默认构建目标**，不是“编译主库的全部 kernels”。即使不显式指定 `--target tile_mma_dte_mxint8_so`，默认 build 也会检查它。
3. `DEPENDS` 要求 `.so` 存在且满足依赖；如果缺失或 object 更新，由第 52 行的规则重新生成。
4. 这里没有写 `COMMAND`。custom target 本身总是被视为需要处理，但它依赖的 `.so` 是有时间戳的文件，**并不因为 target 带 `ALL` 就每次执行链接命令**。

CMake 的对应说明位于 **CMAKE** `Help/command/add_custom_target.rst:20 / add_custom_target`、`:30 / add_custom_target`。现有生成物也能验证：在当前测试子构建目录的 `CMakeFiles/tile_mma_dte_mxint8_so.dir/build.make:69`，入口依赖 `.so`；`:71` 是 `.so` 对 `.o` 的文件依赖；`:73` 才是链接命令；`:81` 把入口标为 `.PHONY`。这些是生成的 Make 规则，不是 CMake 函数定义。

#### 8.3.3 链接库和构建顺序不能混为一谈

`testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:74 / target_link_libraries` 告诉 linker：host executable 要链接哪个 `.so` 和哪些 runtime 库。

`testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:79 / add_dependencies` 则明确告诉 build system：先完成 `tile_mma_dte_mxint8_so` 这个构建目标，再构建依赖它的 executable。它本身不添加 `-l` 参数，也不把 `.so` 内容复制到 executable。

这里没有 `add_library(tile_mma_dte_mxint8_so SHARED ...)`。因此不能把 `tile_mma_dte_mxint8_so` 当成一个原生库 target 去调用 `target_link_libraries`；当前代码选择把实际 `.so` 路径列入链接输入。

### 8.4 mma_dte_test_enable_gtest 的完整作用

`testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:81 / mma_dte_test_enable_gtest` 调用的不是 CMake 内建函数，而是项目自定义函数。其定义在 `testcase/func_test/CMakeMmaDteCommon.cmake:352 / mma_dte_test_enable_gtest`：

```cmake
function(mma_dte_test_enable_gtest target_name)
    find_package(GTest REQUIRED)
    include(GoogleTest)
    target_link_libraries(${target_name} PRIVATE
        GTest::gtest
        GTest::gtest_main
    )
    gtest_discover_tests(${target_name}
        DISCOVERY_MODE PRE_TEST
    )
endfunction()
```

| 实现位置 / 函数名 | 本例作用 |
|---|---|
| `testcase/func_test/CMakeMmaDteCommon.cmake:352 / mma_dte_test_enable_gtest` | 定义函数，接收已有 target 的名称 `test_bf16_mxi8_host` |
| `testcase/func_test/CMakeMmaDteCommon.cmake:353 / mma_dte_test_enable_gtest` | `find_package(GTest REQUIRED)` 查找 GoogleTest，找不到则配置失败；这里没有下载或源码构建 GoogleTest 的逻辑 |
| `testcase/func_test/CMakeMmaDteCommon.cmake:354 / mma_dte_test_enable_gtest` | 加载 CMake 的 `GoogleTest` 模块，以获得 `gtest_discover_tests` 等 CMake 函数；不是 C++ 的 `#include <gtest/gtest.h>` |
| `testcase/func_test/CMakeMmaDteCommon.cmake:355 / mma_dte_test_enable_gtest` | 为 host target 增加 PRIVATE 链接依赖，不覆盖前面已经添加的局部 `.so`、`sipurt`、`sipu` |
| `testcase/func_test/CMakeMmaDteCommon.cmake:356 / mma_dte_test_enable_gtest` | `GTest::gtest` 是 GoogleTest 核心 imported target，提供断言、测试注册和运行框架及对应使用要求 |
| `testcase/func_test/CMakeMmaDteCommon.cmake:357 / mma_dte_test_enable_gtest` | `GTest::gtest_main` 提供默认 `main()`，所以测试 `.cpp` 不必自己写主函数 |
| `testcase/func_test/CMakeMmaDteCommon.cmake:358 / mma_dte_test_enable_gtest` | 结束本次 `target_link_libraries` 参数列表 |
| `testcase/func_test/CMakeMmaDteCommon.cmake:359 / mma_dte_test_enable_gtest` | 注册针对该 executable 的 GoogleTest 测试发现机制，使每个测试用例可以作为独立 CTest 条目 |
| `testcase/func_test/CMakeMmaDteCommon.cmake:360 / mma_dte_test_enable_gtest` | `DISCOVERY_MODE PRE_TEST`：把测试发现延迟到 CTest 准备测试时，不在 executable 刚链接完后立即发现 |
| `testcase/func_test/CMakeMmaDteCommon.cmake:361 / mma_dte_test_enable_gtest` | 结束 `gtest_discover_tests` 参数列表 |
| `testcase/func_test/CMakeMmaDteCommon.cmake:362 / mma_dte_test_enable_gtest` | 结束函数定义 |

默认主函数可直接核对 **GTEST** `googletest/src/gtest_main.cc:49 / main`：调用 `testing::InitGoogleTest(&argc, argv)`，然后返回 `RUN_ALL_TESTS()` 的结果。

#### 8.4.1 PRE_TEST 是“延迟发现”，不是“提前执行 GEMM”

**CMAKE** `Modules/GoogleTest.cmake:552 / gtest_discover_tests` 是实际模块函数；`:718 / gtest_discover_tests` 是 PRE_TEST 分支。它生成供 CTest 读取的脚本，在需要更新测试列表时调用 executable，核心参数是：

```text
test_bf16_mxi8_host --gtest_list_tests
```

实际执行列表查询的位置是 **CMAKE** `Modules/GoogleTestAddTests.cmake:360 / gtest_discover_tests_impl`。列表查询不执行 `TEST_F` 测试体，但 executable 必须能够启动，因此它需要的动态库和初始化环境仍须可用。不能把 PRE_TEST 理解成“不再需要 SIPU runtime”。

本例当前测试体位于：

- `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:146 / TEST_F(MmaDteBf16Mxint8Test, ComputesDefaultShape)`。
- `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:152 / TEST_F(MmaDteFloat16Mxint8Test, ComputesDefaultShape)`。

上面的 `TEST_F` 是 GoogleTest 宏，不是手写类成员函数名。发现后的 CTest 名称分别是 `MmaDteBf16Mxint8Test.ComputesDefaultShape` 和 `MmaDteFloat16Mxint8Test.ComputesDefaultShape`；正式执行时才按对应 `--gtest_filter` 运行测试体。

`enable_testing()` 已在 `testcase/func_test/CMakeMmaDteCommon.cmake:74 / enable_testing` 调用。还需区分：`build-modify.sh:105 / build_testcases` 执行的是 CMake configure/build，并没有运行 `ctest`。所以 `bash build-modify.sh --test` 中的 `--test` 表示**构建测试项目**，不能单凭构建成功认定这些 GEMM 用例已运行通过。

#### 8.4.2 为什么日志出现 discovery mode 变量未使用的警告

`temp/cxx.log.nodir:5740` 给子项目传了 `-DCMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE:STRING=PRE_TEST`，但 `:5756` 至 `:5759` 警告这个变量未使用。

原因可直接从实现解释：**CMAKE** `Modules/GoogleTest.cmake:585 / gtest_discover_tests` 只有在调用者**没有显式指定** `DISCOVERY_MODE` 时才读取这个默认变量。项目 helper 在 `testcase/func_test/CMakeMmaDteCommon.cmake:360 / mma_dte_test_enable_gtest` 已明确传入 `PRE_TEST`，所以本调用不需要读取 Cache 中的默认值。

因此，该警告不表示 PRE_TEST 失效，也不是链接失败。当前子构建目录的 `test_bf16_mxi8_host[1]_include.cmake:1` 起确实生成了“检查 executable 与测试列表时间戳，再发现测试”的 PRE_TEST 脚本；这里是生成脚本行号，不是源函数定义。

### 8.5 CMakeLists.txt 逐行说明，跳过注释和空行

以下覆盖 `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt` 的全部非注释、非空行。CMake 顶层不是 C++ 类成员函数，因此使用“相对路径:行号 / CMake 命令或函数名”；跨行参数和结束括号按其所属命令标注。第 57 行 `COMMENT` 是有效参数，**不是 `#` 注释**，仍需解释。

#### 8.5.1 第 2 至 16 行：项目和路径

| 相对路径:行号 / 命令名 | 代码及作用 |
|---|---|
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:2 / cmake_minimum_required` | `cmake_minimum_required(VERSION 3.27)`：要求 CMake 至少为 3.27，并设置相应 policy 基线；不是要求 C++ 编译器版本为 3.27 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:3 / set` | `set(CMAKE_CXX_STANDARD 20)`：为后续原生 C++ targets 初始化 C++20 标准设置。日志中的 host 编译因此有 `-std=gnu++20`；不要据此认为自定义 SCC 命令自动继承这项设置 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:6 / project` | `project(mma_dte_mxint8_test`：开始定义 CMake 项目，项目名为 `mma_dte_mxint8_test`；它不是 executable target 的名字 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:7 / project` | `VERSION 1.0.0`：设置项目版本信息，不表示 SDK、GEMM 算法或 GoogleTest 的版本 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:8 / project` | `DESCRIPTION "MMA DTE Tiled Tensor - MXINT8 Test"`：设置项目描述元数据 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:9 / project` | `LANGUAGES CXX`：启用 C++ 语言及 host 编译器检测；没有在这里把 SIPU 注册成 CMake 原生语言，`.su` 后面由 SCC helper 的自定义命令处理 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:10 / project` | `)`：结束 `project` 参数列表 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:12 / find_package` | `find_package(scc REQUIRED)`：加载 SCC CMake 包，提供 `scc_add_library`、`scc_target_*` 等函数；找不到包则停止配置。这一行不是开始编译 kernel |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:13 / include` | 加载并执行同级上一层的 `CMakeMmaDteCommon.cmake`；它既定义 helper，也立即执行查找 DTE 根目录、选择架构/dispatch family、设置 SDK 链接目录、启用 CTest 等配置 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:16 / set` | `set(FUNC_TEST_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernel")`：把测试源码目录下的 `kernel` 子目录保存为普通变量；这是源码路径，不是输出目录 |

第 12 行使用的 **SDK** `share/cmake/scc/sccConfig.cmake:19 / set` 从环境变量 `SI_SDK_ROOT` 设置 `SCC_ROOT`；`:33 / find_program` 在未预设 `SCC_EXECUTABLE` 时查找 SDK 的 `bin/scc`。`scc_DIR`、`SI_SDK_ROOT` 和缓存的 `SCC_EXECUTABLE` 应保持一致；只在文字中指定 SDK 路径不会改变已有 CMake Cache。

#### 8.5.2 第 19 至 36 行：编译配置

| 相对路径:行号 / 命令名 | 代码及作用 |
|---|---|
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:19 / option` | `option(MMA_DTE_GENERIC_TAIL "..." OFF)`：定义默认关闭的布尔配置项；开启后由第 34 行加入宏，使 dispatch 选择 generic tail 路径 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:20 / set` | `set(TEST_GRID_DIM "16" CACHE STRING`：定义可缓存的字符串配置，默认值 16；它随后被拼成编译宏，不是 `-j16` 编译并行度，也不是 GPU 设备编号 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:21 / set` | `"Test-only grid dim ...")`：给上述 Cache 项设置说明并结束 `set`；这段字符串不是传给 SCC 的参数 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:22 / set` | `set(TEST_ENABLE_DEBUG_PRINT OFF CACHE BOOL`：定义默认关闭的调试打印布尔配置 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:23 / set` | 给 `TEST_ENABLE_DEBUG_PRINT` 设置 Cache 说明并结束 `set` |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:24 / mma_dte_test_setup_compile_options` | 调用公共 helper，将后续参数与公共宏合并，写回普通列表变量 `COMMON_COMPILE_OPTIONS`；这里仍未编译 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:25 / mma_dte_test_setup_compile_options` | `-DSIRT_DEVICE_USE_PRINTF`：定义 SDK 设备 printf 开关；它与第 31 行的 `KERNEL_DBG` 是两个不同层次的开关 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:26 / mma_dte_test_setup_compile_options` | `-fPIC`：要求位置无关代码，为后续将 object 链接成共享库准备。这里 SCC target 是 OBJECT 模式，不能依赖 SDK 仅对 SHARED 模式自动补的 PIC 选项 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:27 / mma_dte_test_setup_compile_options` | `-DDTE_DEV_MODE`：启用代码中的开发分支；例如绕过 production layout contract 的特定静态断言，并允许下面的测试 grid 配置生效，不等于关闭所有参数校验 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:28 / mma_dte_test_setup_compile_options` | `-DMMA_DTE_TEST_GRID_DIM=${TEST_GRID_DIM}`：默认展开为 `-DMMA_DTE_TEST_GRID_DIM=16`，传给 planner 的测试 grid 配置 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:29 / mma_dte_test_setup_compile_options` | `)`：结束 helper 调用 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:30 / if` | `if(TEST_ENABLE_DEBUG_PRINT)`：在 CMake 配置时判断布尔选项，而不是 executable 运行时判断 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:31 / list` | `list(APPEND COMMON_COMPILE_OPTIONS -DKERNEL_DBG)`：选项为真时追加调试打印宏 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:32 / endif` | 结束调试打印条件块 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:33 / if` | `if(MMA_DTE_GENERIC_TAIL)`：判断是否启用 generic tail 编译路径 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:34 / list` | `list(APPEND COMMON_COMPILE_OPTIONS -DMMA_DTE_GENERIC_TAIL)`：为真时追加宏；单纯有一个 CMake Cache 项不会自动产生同名 C++ 宏，需要这里的 `-D` |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:35 / endif` | 结束 generic tail 条件块 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:36 / mma_dte_test_setup_include_dirs` | 将公共 include 目录去重后写到 `COMMON_INCLUDE_DIRECTORIES`，留给 SCC 和 host 两侧分别使用 |

这些配置的源码落点：

| 相对路径:行号 / 函数名 | 补充解释 |
|---|---|
| `testcase/func_test/CMakeMmaDteCommon.cmake:173 / mma_dte_test_setup_compile_options` | 合并调用者的 flags、架构宏、`MMA_DTE_SINGLE_FAMILY_DISPATCH=1` 和当前 family 的 dispatch header；还会按配置添加 SplitK、TMAP LMUL 宏；第 193 行用 `PARENT_SCOPE` 回传 |
| `testcase/func_test/CMakeMmaDteCommon.cmake:47 / elseif` | 当前 source 目录命中 `mxint8/mxi8`，选择 `MMA_DTE_TEST_DISPATCH_FAMILY=mxint8`，第 65 行形成 `dispatch_cases_mxint8_generated.hpp` 路径 |
| `testcase/func_test/CMakeMmaDteCommon.cmake:164 / mma_dte_test_setup_include_dirs` | 从公共目录列表出发，追加可选额外目录、去重，第 170 行回传；本例没有传额外目录 |
| `cmake/MmaDteConfig.cmake:41 / mma_dte_common_include_directories` | 公共 include 包括 SDK include 目录和 DTE 的 `include`、`kernel`、`kernel/util` |
| `cmake/MmaDteConfig.cmake:19 / mma_dte_collect_sdk_include_directories` | 在多个 SDK 目录结构候选中，只收集实际存在的 include 目录 |
| `cmake/MmaDteConfig.cmake:96 / mma_dte_configure_sipu_arch` | 为 SCC 设置规范化架构参数，例如本次日志的 `-arch=sipu_150` |
| `kernel/mma_dte_tiled_tensor.hpp:54 / mma_dte_implement_with_control` | 第 64 行的 `#ifndef DTE_DEV_MODE` 只保护 production layout contract 静态断言；之后仍有形状等检查 |
| `kernel/detail/mma_dte_tiled_tensor_planner.hpp:2471 / get_best_solution_heuristic` | 第 2474 行同时检查 `DTE_DEV_MODE` 和 `MMA_DTE_TEST_GRID_DIM`，然后取测试 grid 值参与规划 |
| `kernel/sipu_kernel_debug.h:9 / DEBUG_PRINT（宏）` | 未定义 `KERNEL_DBG` 时打印宏为空操作；定义后按 host/device 分支调用打印函数 |
| **SDK** `include/sipu_dev_apis/utils/libc.h:21 / sipu::printf` | 第 27 行用 `SIRT_DEVICE_USE_PRINTF` 控制设备打印逻辑；全局 `printf` 的对应实现位于第 72 行 |

这里的 `CACHE` 值是可配置的默认值，已有 Cache 或调用者的 `-DNAME=...` 可决定实际取值；当前这些 `set(... CACHE ...)` 没有 `FORCE`，不是每次重新配置都强制恢复默认值。

#### 8.5.3 第 41 至 62 行：SCC object 与局部共享库

| 相对路径:行号 / 命令名 | 代码及作用 |
|---|---|
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:41 / set` | `set(TEST_INST_SRC "${FUNC_TEST_KERNEL_DIR}/inst_bf16_mxi8.su" CACHE STRING "...")`：设置测试设备代码实例化源文件默认路径；可覆盖，不是递归收集整个主库源码 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:43 / scc_add_library` | `scc_add_library(tile_mma_dte_mxint8 OBJECT`：定义 SCC OBJECT 构建目标；只安排源文件编译为 `.o`，这里不生成 `.so` |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:44 / scc_add_library` | `${TEST_INST_SRC}`：把选中的 `.su` 作为该目标的源输入 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:45 / scc_add_library` | `)`：结束 SCC target 定义调用 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:46 / scc_target_compile_options` | 把 `COMMON_COMPILE_OPTIONS` 加到 SCC target 属性；SDK 编译规则在生成阶段读取这些属性，最终转为 `--clangopt=...` 参数 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:47 / scc_target_include_directories` | 把公共 include 路径加到 SCC target，最终在 SCC 命令中形成 `-I...` |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:50 / scc_target_get_output` | 把该 OBJECT target 的输出 `.o` 路径列表写到 `TILE_MMA_DTE_MXINT8_O`；详见 8.2 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:52 / add_custom_command` | 开始登记生成 `.so` 的自定义规则，不在 CMake 配置过程中直接执行 linker |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:53 / add_custom_command` | `OUTPUT .../libtile_mma_dte_mxint8.so`：声明该规则的输出文件；路径在当前测试子构建目录 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:54 / add_custom_command` | `COMMAND ${CMAKE_CXX_COMPILER} -shared -o ...`：使用 host C++ compiler driver 生成共享库，不是再次用 SCC 编译 `.su` |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:55 / add_custom_command` | `${TILE_MMA_DTE_MXINT8_O}`：是上一行链接命令的续行参数，即输入 object 列表，不是第二条命令 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:56 / add_custom_command` | `DEPENDS tile_mma_dte_mxint8 ${TILE_MMA_DTE_MXINT8_O}`：同时添加构建顺序依赖和 object 文件更新依赖，详见 8.3.1 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:57 / add_custom_command` | `COMMENT "Linking libtile_mma_dte_mxint8.so"`：该规则执行时显示的进度提示，不是源码注释，也不是额外 shell 命令 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:58 / add_custom_command` | `)`：结束自定义命令定义 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:60 / add_custom_target` | `add_custom_target(tile_mma_dte_mxint8_so ALL`：创建默认构建会处理的共享库构建入口，入口本身不是库文件 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:61 / add_custom_target` | `DEPENDS .../libtile_mma_dte_mxint8.so`：让入口依赖前面规则的输出，因此能按需触发生成规则 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:62 / add_custom_target` | `)`：结束 custom target 定义 |

第 46、47 行为何可以写在第 43 行后面：**SDK** `share/cmake/scc/sccConfig.cmake:705 / scc_target_compile_options` 在第 783 行保存 `SIPU_ALL_COMPILE_OPTIONS`；`:512 / scc_target_include_directories` 在第 610 行保存 `SIPU_ALL_INCLUDE_DIRECTORIES`。`:431 / scc_add_library` 创建的命令使用 `$<TARGET_PROPERTY:...>` generator expressions，在生成时取得这些后续设置的属性。它没有在第 43 行当场执行一条缺少 flags/includes 的 SCC 命令。

默认 `.su` 中有两个公开 API 显式实例化，见 `testcase/func_test/test_bf16_mxint8_host/kernel/inst_bf16_mxi8.su:24 / mma_dte` 和 `:31 / mma_dte`，分别对应 MXINT8 输入的 BF16/R8 与 FP16/R32 测试路径。因此“一个实例化源文件”不等于“只实例化一个函数”，更不等于“只有一个底层 device kernel”。

#### 8.5.4 第 66 至 81 行：host executable、runtime 和测试框架

| 相对路径:行号 / 命令名 | 代码及作用 |
|---|---|
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:66 / set` | `set(TEST_HOST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/test_bf16_mxi8_host.cpp" CACHE STRING "...")`：设置 host 测试源文件默认路径，可覆盖；与 device 实例化源 `TEST_INST_SRC` 是不同输入 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:67 / add_executable` | 创建原生 C++ executable target `test_bf16_mxi8_host`，由 `${CMAKE_CXX_COMPILER}` 编译和链接 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:68 / add_executable` | `${TEST_HOST_SRC}`：把当前选中的 host `.cpp` 加入 executable 源文件列表 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:69 / add_executable` | `)`：结束 executable 定义 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:70 / target_include_directories` | 为 host target 添加 PRIVATE include 目录，服务于该 target 的源码编译，不作为公共使用要求向外传播 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:71 / target_include_directories` | `${COMMON_INCLUDE_DIRECTORIES}`：使用 SDK 与 DTE 的公共 include 路径；这一次是 host 编译器的 include 设置，不是第 47 行的 SCC target 设置 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:72 / target_include_directories` | `${MMA_DTE_ROOT}`：额外加入 DTE 根目录，允许从该根目录起解析相对 include 路径 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:73 / target_include_directories` | `)`：结束 include 设置 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:74 / target_link_libraries` | 开始设置 host executable 的 PRIVATE 链接依赖；这是 CMake 内建命令，不是 SDK 的 `scc_target_link_libraries` |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:75 / target_link_libraries` | 链接当前测试生成的 `libtile_mma_dte_mxint8.so` 文件，提供 host 调用的 `mma_dte` 实现及其设备程序；不是主库 `libtile_mma_dte.so` |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:76 / target_link_libraries` | `sipurt`：加入 SIPU runtime 库链接项，日志中对应 `-lsipurt` |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:77 / target_link_libraries` | `sipu`：加入另一项 SDK 库依赖，日志中对应 `-lsipu`；这里写的是库名，实际由链接器按搜索路径定位文件 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:78 / target_link_libraries` | `)`：结束这组链接项设置 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:79 / add_dependencies` | 显式建立 `test_bf16_mxi8_host` 对 `tile_mma_dte_mxint8_so` 的 target 依赖，保证所需共享库构建先完成；它不替代第 74 行的链接项 |
| `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:81 / mma_dte_test_enable_gtest` | 为这个 executable 链接 GoogleTest/default main，并设置 PRE_TEST 测试发现，详见 8.4 |

第 76、77 行的 SDK 库搜索路径来自 `testcase/func_test/CMakeMmaDteCommon.cmake:70 / if` 和 `:71 / link_directories`：当 `$ENV{SI_SDK_ROOT}/lib` 存在时加入该目录。**include 路径解决编译时找头文件，library 路径解决链接时找库，target dependency 解决构建顺序，这三者不能互相替代。**

### 8.6 用实际构建日志把各阶段串起来

先约定本次日志里的两个输出目录，均相对 **DTE** 根目录：

```text
B = build/testcase/_build/2609161442/sipu_150/default/specialized/func_test/test_bf16_mxint8_host
R = build/testcase/2609161442/sipu_150/default/specialized
```

这里 `B` 是当前测试子项目的 `CMAKE_CURRENT_BINARY_DIR`；`R` 是外层传入的 `CMAKE_RUNTIME_OUTPUT_DIRECTORY`。因此 `.so` 和 executable **不一定在同一个目录**。该外层设置位于 `testcase/CMakeLists.txt:85 / set`，子构建目录在 `testcase/CMakeLists.txt:131 / mma_dte_add_testcase_project` 中构造，并由第 173 行的 `ExternalProject_Add` 使用。

```text
配置/生成阶段：
  读取 CMakeLists、设置 Cache、加载 SDK/helper
  定义 SCC 编译规则、SO 链接规则、host target、CTest 发现脚本
  scc_target_get_output 在此阶段取回预定 object 路径

构建阶段的产物流程：
  kernel/inst_bf16_mxi8.su
      -> scc -c
      -> B/_SipuObj/tile_mma_dte_mxint8/kernel/inst_bf16_mxi8.su.o
      -> /usr/bin/c++ -shared
      -> B/libtile_mma_dte_mxint8.so

  test_bf16_mxi8_host.cpp
      -> /usr/bin/c++ -c
      -> B/CMakeFiles/test_bf16_mxi8_host.dir/test_bf16_mxi8_host.cpp.o
      + B/libtile_mma_dte_mxint8.so
      + sipurt / sipu / gtest / gtest_main
      -> /usr/bin/c++ 链接
      -> R/test_bf16_mxi8_host

之后的 CTest 阶段：
  按需用 --gtest_list_tests 发现测试
      -> 注册各条 CTest 测试
      -> 正式执行选中的测试体
```

日志行号与源码对应如下。日志不是函数定义，表格同时给出对应的源码入口：

| 日志位置 | 实际记录 | 对应源码位置 / 命令名 |
|---|---|---|
| `temp/cxx.log.nodir:15`、`:44` | test 模式，只进入 `build_testcases` | `build-modify.sh:105 / build_testcases` |
| `temp/cxx.log.nodir:5740` | 配置当前测试子项目，host compiler 为 `/usr/bin/c++` | `testcase/CMakeLists.txt:131 / mma_dte_add_testcase_project` |
| `temp/cxx.log.nodir:5750` | SCC target 类型 OBJECT，初始选项是 `-arch=sipu_150;-c` | `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:43 / scc_add_library` |
| `temp/cxx.log.nodir:5751`、`:5752` | 追加 printf/PIC/dev/grid/arch/dispatch flags 与 include 目录 | `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:46 / scc_target_compile_options`、`:47 / scc_target_include_directories` |
| `temp/cxx.log.nodir:5753` | 找到系统 GoogleTest 1.11.0 的 CMake package | `testcase/func_test/CMakeMmaDteCommon.cmake:353 / mma_dte_test_enable_gtest` |
| `temp/cxx.log.nodir:5781` | SDK 2609161442 的 SCC 将测试 `.su` 编成 `.su.o` | **SDK** `share/cmake/scc/sccConfig.cmake:431 / scc_add_library` |
| `temp/cxx.log.nodir:5852` | `/usr/bin/c++ -shared -o B/libtile_mma_dte_mxint8.so B/_SipuObj/.../inst_bf16_mxi8.su.o` | `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:52 / add_custom_command` |
| `temp/cxx.log.nodir:5854` | `Built target tile_mma_dte_mxint8_so` | `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:60 / add_custom_target` |
| `temp/cxx.log.nodir:5862` | `/usr/bin/c++ ... -std=gnu++20 ... -c test_bf16_mxi8_host.cpp` | `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:67 / add_executable` |
| `temp/cxx.log.nodir:5945` | 链接 host `.o`、`-ltile_mma_dte_mxint8 -lsipurt -lsipu` 及系统 `libgtest.a`、`libgtest_main.a`，输出到 `R` | `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:74 / target_link_libraries`、`:81 / mma_dte_test_enable_gtest` |
| `temp/cxx.log.nodir:5947` | `Built target test_bf16_mxi8_host` | 表示该 executable 构建成功，不能当作 GEMM 测试通过的记录 |

虽然第 75 行写的是 `.so` 的绝对路径，现有生成的链接命令采用了 `-L<该目录> -ltile_mma_dte_mxint8`，并带该目录的运行时搜索路径；这是同一个库，不是换成了主库。

最后补充两个边界：

- `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:54 / add_custom_command` 是手工拼接的链接命令，并不会像原生 `add_library(... SHARED ...)` 那样自动注入全部 `CMAKE_CXX_FLAGS`、`CMAKE_SHARED_LINKER_FLAGS` 或 target link options。应以该行写出的参数和实际日志为准。
- `testcase/func_test/CMakeMmaDteCommon.cmake:209 / mma_dte_get_main_object` 虽然定义了复用主库 object 的 helper，**本文件没有调用它**。不能因 include 了公共文件，就认为这个测试正在复用整个主库或主库 object。

本次只读检查了当前源码、指定 SDK、CMake 模块、现有生成规则和日志，并追加本文；没有修改 CMake/算子代码，没有重新编译，也没有执行设备测试。

## 9. MXINT8 GEMM golden 的逻辑顺序、物理顺序与精度转换（2026-09-17）

### 9.1 先回答三个问题

本节项目源码路径以 **DTE** `/share/users/like/package/sikernel/mma_dte_tile_tensor` 为根；**SDK** 路径以 `/share_data/sicx_sdk/release/2609151958` 为根。行号按本次含新增打印的源码重新核对。

1. **逻辑顺序按矩阵坐标 `[m,n]` 展开，物理顺序按实际存储的 tile/subtile/block 顺序展开。**两者都返回 `std::vector<float>`，不是一种返回 FP32、另一种返回 BF16/FP16。
2. **`gold_tiled_tensor.toVector()` 没有发生 FP32 -> BF16/FP16 转换。**降精度发生在后面的 `make_tensor<OutputT>`。因此 `gold_linear` 与 `gold_tiled` 同时存在顺序差异和精度差异，不能保证逐位相同；即使先恢复相同逻辑顺序，也不能保证与原 FP32 相同。
3. **`mm_mnk(A, B, LayoutTag::Tiled)` 的第三个参数指定返回结果 Tensor 的 layout。**它不是声明或覆盖 A/B 的输入 layout，也不控制后面设备 `mma_dte` 的输出格式。严格说，它在 C++ 中仍是一个按值传入的配置参数，不是输出引用参数；配置对象是返回值。

还需区分运行环境：`temp/run.log:1` 显示加载的 runtime 位于 SDK **2609161442**；`build/testcase/test_bf16_mxi8_host` 当前也是指向该版本子目录 executable 的符号链接。`temp/run.log:121` 显示本次运行采用 `swemusp` shim。日志与题目指定的 SDK 2609151958 不完全同源；只读比较确认两版的 `tensor/tensor.hpp`、`sifmt_fp.hpp`、`quantize/quantize.hpp`、`quantize/nonmx.hpp` 这四份相关 SiTe 文件相同。下文源码语义以指定 SDK 为准，日志事实另行标注。

### 9.2 四个变量的真实类型与数据流

相关代码集中在 `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:67 / run_case` 至第 73 行：

| 相对路径:行号 / 函数名 | 变量或操作 | 类型、顺序与精度 |
|---|---|---|
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:68 / run_case` | `gold_tiled_tensor = mm_mnk(..., Tiled)` | `Tensor<sifmt::float32, ...>`；保存 CPU FP32 计算结果，物理 layout 为 tiled |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:69 / run_case` | `gold_linear = gold_tiled_tensor.toVector()` | `std::vector<float>`；逻辑 row-major 顺序；仍为 FP32，没有降低到 OutputT 精度 |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:72 / run_case` | `out_tensor_d = make_tensor<OutputT>(layout_d, gold_linear)` | `Tensor<OutputT, ...>`；每个 FP32 值转换为 BF16/FP16，再存入该 dtype 对应的 tiled 存储 |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:73 / run_case` | `gold_tiled = out_tensor_d.toVectorAsMemoryOrder(true)` | `std::vector<float>`；按 **OutputT tensor 的物理数据顺序**导出；BF16/FP16 数值重新扩展为 FP32 |

```text
MXINT8 A/B, each with its own layout
        |
        | toVector(): decode layout + dequantize MXINT8 to float
        v
CPU float dot products: X[m,n]
        |
        | mm_mnk(..., Tiled)
        v
gold_tiled_tensor
  dtype = FP32, storage = FP32 tiled
        |
        | toVector(): read by logical coordinates, no FP16/BF16 rounding
        v
gold_linear
  dtype = float, order = row-major, values = X[m,n]
        |
        | make_tensor<OutputT>(layout_d, gold_linear)
        | round to BF16/FP16 + pack into OutputT tiled layout
        v
out_tensor_d
  dtype = BF16/FP16, storage = OutputT tiled
        |                                  |
        | toVectorAsMemoryOrder(true)      | toVector() [not in original code]
        v                                  v
gold_tiled                             rounded_linear
  dtype = float                          dtype = float
  order = physical data order            order = row-major
  values = widened rounded values        values = widened rounded values
```

两个 exporter 都是在 host 上构造新的 vector，既不修改原 Tensor 的 layout，也不启动 SIPU kernel。`gold_tiled_tensor`、`gold_linear`、`out_tensor_d`、`gold_tiled` 是不同对象，不是同一块内存的不同 view。

这里的“原 FP32 golden”指**已经量化为 MXINT8 的 A/B 解量化后，再用 float 计算的结果**，不是未量化的 `arr_A @ arr_B^T`，也不是任意精度实数结果。依据是 **SDK** `include/SiTe/tensor/tensor.hpp:2698 / mm_mnk` 对输入先调用 `toVector()`，以及 `:2711 / mm_mnk` 使用 `float sum` 累加。

### 9.3 两种导出的实现区别

#### 9.3.1 Tensor::toVector：先确定逻辑坐标，再找物理地址

**SDK** `include/SiTe/tensor/tensor.hpp:2114 / Tensor::toVector` 的核心行为：

```cpp
std::vector<float> buffer(shape.size(), 0.0f);
for (uint64_t i = 0; i < shape.size(); i++) {
    auto data = a.at(i);
    // MX type: dequantize(0); ordinary FP type: cast to float.
    ...
}
```

关键不是 `i` 自增，而是 `at(i)` 的含义：

| SDK 相对路径:行号 / 成员函数 | 行为 |
|---|---|
| `include/SiTe/tensor/tensor.hpp:1839 / Tensor::at` | 把一维逻辑 index 交给 `Shape::indexToCoord`，不是直接读 `data()[i]` |
| `include/SiTe/tensor/tensor.hpp:218 / Shape::indexToCoord` | 从最后一维开始做除余；对于 `[M,N]`，得到 `m=i/N`、`n=i%N` |
| `include/SiTe/tensor/tensor.hpp:1851 / Tensor::at` | 根据坐标通过 layout engine 找到 block 地址和 block 内元素位置，读取该元素 |
| `include/SiTe/tensor/tensor.hpp:1524 / LayoutEngine::tensorCoord2StoragePtrOffset` | 根据 Tensor dtype/layout 计算实际存储地址；会处理 tile、subtile、block 等层次 |
| `include/SiTe/tensor/tensor.hpp:2129 / Tensor::toVector` | 非 MX 类型转为 `float`，放入 `buffer[i]` |

因此，二维矩阵的导出顺序一定是这里的逻辑顺序：

```text
Matrix coordinates:
          n=0       n=1       ...       n=N-1
m=0       X[0,0]    X[0,1]    ...       X[0,N-1]
m=1       X[1,0]    X[1,1]    ...       X[1,N-1]
 ...
m=M-1     X[M-1,0]  X[M-1,1]  ...       X[M-1,N-1]

toVector():
  [ X[0,0], X[0,1], ..., X[0,N-1],
    X[1,0], X[1,1], ..., X[1,N-1], ... ]

logical_index(m,n) = m*N + n
```

源 Tensor 即使是 tiled，导出结果仍会按这个坐标顺序排列；`toVector` 没有把源 Tensor 本身改为 Linear。

#### 9.3.2 Tensor::toVectorAsMemoryOrder：按数据 block 的存储位置遍历

**SDK** `include/SiTe/tensor/tensor.hpp:2141 / Tensor::toVectorAsMemoryOrder` 的流程不同：

| SDK 相对路径:行号 / 成员函数 | 行为 |
|---|---|
| `include/SiTe/tensor/tensor.hpp:2143 / Tensor::toVectorAsMemoryOrder` | 如果源 Tensor 已是 `Linear`，直接返回 `toVector()` |
| `include/SiTe/tensor/tensor.hpp:2157 / Tensor::toVectorAsMemoryOrder` | 遍历有效逻辑坐标，用 layout engine 建立“物理 block 地址 -> 该 block 的有效坐标”映射 |
| `include/SiTe/tensor/tensor.hpp:2184 / Tensor::toVectorAsMemoryOrder` | 按 plane、supertile、数据 block 的存储顺序遍历 |
| `include/SiTe/tensor/tensor.hpp:2189 / Tensor::toVectorAsMemoryOrder` | 跳过 supertile header，定位数据区；输出的是数值序列，不是包含 header 的完整字节镜像 |
| `include/SiTe/tensor/tensor.hpp:2203 / Tensor::toVectorAsMemoryOrder` | 对当前物理位置对应的有效坐标调用 `at` |
| `include/SiTe/tensor/tensor.hpp:2210 / Tensor::toVectorAsMemoryOrder` | 普通 FP 类型扩展成 `float`，按访问顺序追加到 vector |
| `include/SiTe/tensor/tensor.hpp:2215 / Tensor::toVectorAsMemoryOrder` | `true` 控制是否追加 block 尾部的 padding 标记，标记值为 `quiet_NaN()` |

**`true` 的参数名是 `hasPaddingData`。它不表示“转成 FP32”，也不表示“启用 tiled”；是否 tiled 由源 Tensor 已有的 layout 决定。**

本例输出是普通 BF16/FP16，supertile 没有 MX scale header，且 M/N 整齐覆盖完整 tiles。于是它可以理解成：依物理地址顺序读出各个 16-bit 浮点元素，再把每个元素的数值扩展为一个 32-bit `float`。

注意，`gold_tiled` 自身的内存仍是连续的 32-bit floats；并不是可直接 `memcpy` 到 `OutputT*` 的 16-bit Tensor 原始字节。Tensor 原始存储入口是 **SDK** `include/SiTe/tensor/tensor.hpp:2243 / Tensor::data`。

### 9.4 结合实际日志画出 tile/subtile 顺序

#### 9.4.1 先核对两种 dtype 的 tile 大小

`testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:140 / run_bf16_default_case` 选择 BF16、`M=8,N=576,K=1024`；`:145 / run_default_case` 选择 FP16、`M=32,N=576,K=1024`。

两种输出都是 16-bit，但 CPU 中间 golden 是 32-bit，所以同一个 `[M,N]`、同一个 `LayoutTag::Tiled`，不代表相同 tile 形状：

| 用例与对象 | dtype | 逻辑 shape | tile shape `[rows,cols]` | subtile shape | tile 数 | Tensor 存储字节 |
|---|---|---|---|---|---|---|
| BF16/R8：`gold_tiled_tensor` | FP32 | `[8,576]` | `[8,32]` | `[8,8]` | 18 | 18432 |
| BF16/R8：`out_tensor_d` | BF16 | `[8,576]` | `[8,64]` | `[8,16]` | 9 | 9216 |
| FP16/R32：`gold_tiled_tensor` | FP32 | `[32,576]` | `[32,8]` | `[8,8]` | 72 | 73728 |
| FP16/R32：`out_tensor_d` | FP16 | `[32,576]` | `[32,16]` | `[8,16]` | 36 | 36864 |

日志依据：`temp/run.log:68`、`:78`、`:96`、`:106`、`:188`、`:198`、`:216`、`:226`。这些是日志位置，不是函数定义。

源码依据是 **SDK** `include/SiTe/tensor/tensor.hpp:1097 / LayoutEngine::getTileShape`：第 1126 行处理 R8，第 1154 行处理 R32，列数都依赖 `DataType::kBitWidth`。`:1235 / LayoutEngine::LayoutEngine` 的第 1278 行选 tile shape；普通非 MX 类型在第 1287 行设置 supertile shape 为 `[1,1]`。本表每个 tile 都是 1024 字节，16-bit tile 能容纳的元素数是 32-bit tile 的两倍。

下面统一用 `Y[m,n]` 表示**已经舍入为 OutputT 后的数值**，暂时只讨论顺序。

#### 9.4.2 BF16/R8：8 行，tile 内的 4 个 subtile 横向排列

一张 `[8,576]` 输出有 9 个 `[8,64]` tiles。每个 tile 包含 4 个 `[8,16]` subtiles；每个 subtile 是 8 个 block，每个 block 是一行的 16 个 BF16 元素。

```text
Output matrix [8,576]:

cols    0........63 64......127                 512......575
       +----------+----------+----- ... -------+----------+
rows   |  tile 0  |  tile 1  |                 |  tile 8  |
0..7   |  [8,64]  |  [8,64]  |                 |  [8,64]  |
       +----------+----------+----- ... -------+----------+

Inside tile 0:

cols     0..15      16..31      32..47      48..63
       +----------+----------+----------+----------+
rows   | subtile0 | subtile1 | subtile2 | subtile3 |
0..7   |  [8,16]  |  [8,16]  |  [8,16]  |  [8,16]  |
       +----------+----------+----------+----------+

Physical element order:
  subtile0: Y[0,0..15], Y[1,0..15], ..., Y[7,0..15]
  subtile1: Y[0,16..31], Y[1,16..31], ..., Y[7,16..31]
  subtile2: Y[0,32..47], ...
  subtile3: Y[0,48..63], ...
  then tile1, tile2, ..., tile8

Logical row-major order:
  row0: Y[0,0..15], Y[0,16..31], ..., Y[0,560..575]
  row1: Y[1,0..15], Y[1,16..31], ..., Y[1,560..575]
  ...
```

**不要把这个 R8 tile 想成“先连续存满一整行 64 个元素”。**实际会先连续存完 subtile0 的 8 行，再存 subtile1。SDK 对应位置是 `include/SiTe/tensor/tensor.hpp:1650 / LayoutEngine::tensorCoord2StoragePtrOffset` 的 R8 分支；subtile 和 block 偏移分别在第 1615、1628 行计算。

只对本例 `[8,576]`、BF16、无 padding 的输出，物理元素索引可化简为：

```text
L(m,n) = m*576 + n
P8(m,n) = (n/16)*128 + m*16 + (n%16)   // integer division
byte_offset = 2 * P8(m,n)
```

例如：

| 逻辑坐标 | `L` | `P8` | 说明 |
|---|---:|---:|---|
| `[0,0]` | 0 | 0 | 起点相同 |
| `[1,0]` | 576 | 16 | 物理序列第 16 项已经切到下一行，不是 `[0,16]` |
| `[0,16]` | 16 | 128 | 下一个 subtile 起点 |
| `[0,64]` | 64 | 512 | 下一个 tile 起点 |

所以 `gold_linear[16]` 对应原 FP32 的 `[0,16]`；`gold_tiled[16]` 对应经过 BF16 舍入的 `[1,0]`。直接用相同下标比较，首先就不是同一个矩阵元素。

#### 9.4.3 FP16/R32：32 行，tile 内的 4 个 subtile 纵向排列

一张 `[32,576]` 输出有 36 个 `[32,16]` tiles，每个 tile 内部的 4 个 `[8,16]` subtiles 沿行方向排布。

```text
Output matrix [32,576]:

cols       0..15       16..31                560..575
         +----------+----------+--- ... ---+----------+
rows     |  tile 0  |  tile 1  |           | tile 35  |
0..31    | [32,16]  | [32,16]  |           | [32,16]  |
         +----------+----------+--- ... ---+----------+

Inside tile 0:                    Physical element order:
             +----------+
rows  0.. 7  | subtile0 |         Y[0,0..15], ..., Y[7,0..15]
             +----------+
rows  8..15  | subtile1 |         Y[8,0..15], ..., Y[15,0..15]
             +----------+
rows 16..23  | subtile2 |         Y[16,0..15], ..., Y[23,0..15]
             +----------+
rows 24..31  | subtile3 |         Y[24,0..15], ..., Y[31,0..15]
             +----------+
                                 then tile1: Y[0,16..31], ...
```

R32 分支位于 **SDK** `include/SiTe/tensor/tensor.hpp:1657 / LayoutEngine::tensorCoord2StoragePtrOffset`。本例每个 tile 内部碰巧等价于按 32 行依次存储，但整个矩阵仍是先存完 16 列宽的 tile，再到下一个 tile，并非先存完整的 576 列长行。

只对本例 `[32,576]`、FP16、无 padding 的输出：

```text
L(m,n) = m*576 + n
P32(m,n) = (n/16)*512 + m*16 + (n%16)
byte_offset = 2 * P32(m,n)

P32(1,0)  = 16
P32(8,0)  = 128
P32(0,16) = 512
```

同样，`gold_tiled[16]` 不是逻辑第 16 项，而是 `[1,0]`；只是这次 `gold_tiled[128]` 对应 `[8,0]`，不是 R8 情形下的 `[0,16]`。

这两条物理索引公式针对当前完整 shape 推导，**不能把其中的 `8/32` 直接替换成任意 M，就当成通用 SiTe 布局公式**。通用定位仍应调用 layout engine。

#### 9.4.4 为什么两个 vector 的 size 相同，顺序却不同

`temp/run.log:63` 给出 `4608/4608`，`:183` 给出 `18432/18432`，因为本例完整覆盖所有输出 tiles：

```text
BF16: 8*576 = 9 * (8*64) = 4608 elements
FP16: 32*576 = 36 * (32*16) = 18432 elements
```

没有 padding 时，重排不会改变元素数量。`gold_tiled` 的 vector payload 按 4 字节/项计算，而 `out_tensor_d` 按 2 字节/项存储；元素数量相同不代表字节数相同，更不代表值序列逐位相同。

此外，日志 `stride=[1,1]` 不能按普通矩阵 stride 理解成地址公式 `m+n`。**SDK** `include/SiTe/tensor/tensor.hpp:498 / Layout::Layout` 只保存 shape/tag；tiled 地址由 `LayoutEngine` 另外计算。日志中的 `Shape(SubTile): (NULL)` 也是 `include/SiTe/tensor/tensor.hpp:2442 / Tensor::print` 固定打印的字符串，并不是没有 subtile。

### 9.5 精度变化：两个 vector 能否 bitwise 相同

#### 9.5.1 先用公式把两类变化分开

定义：

```text
X[m,n]    = CPU FP32 golden
Q_T(x)    = convert float x to OutputT (16-bit encoding)
U_T(q)    = widen an OutputT value back to float
Y[m,n]    = U_T(Q_T(X[m,n]))
L(m,n)    = logical row-major index
P_T(m,n)  = physical element index in the OutputT tensor
```

则对于本例有效元素：

```text
gold_linear[L(m,n)]                 = X[m,n]
out_tensor_d.toVector()[L(m,n)]     = Y[m,n]
gold_tiled[P_T(m,n)]                = Y[m,n]
```

因此要区分三个问题：

| 比较方式 | 结论 |
|---|---|
| `gold_linear[i]` 与 `gold_tiled[i]` | 通常连坐标都不同，不能保证相同 |
| 把 `gold_tiled` 重排回 row-major 后，与 `gold_linear` 比 | 坐标相同，但还隔着 FP32 -> OutputT -> FP32 的舍入，不能保证逐位相同 |
| 对**同一个 `out_tensor_d`**分别调用两个 exporter，再按坐标对齐 | 没有额外降精度差异；当前无 padding 的 BF16/FP16 数据，是同一批已经舍入的数值，仅顺序不同 |

#### 9.5.2 真正的降精度发生在哪里

**SDK** `include/SiTe/tensor/tensor.hpp:1772 / Tensor::Tensor` 接收 float 数据并调用 `fillWithContainer`。`:1950 / Tensor::fillWithContainer` 的普通浮点分支在第 2063 行执行：

```cpp
nonMxBlock[i] = DataType::FromFloat(inputBlock[i]);
```

本例 `DataType` 就是 `OutputT`。这一步产生 BF16/FP16 编码，然后由 `include/SiTe/tensor/tensor.hpp:2402 / Tensor::fillBlock` 按布局写入存储。

转换定义可继续定位：

| SDK 相对路径:行号 / 函数名或类型 | 含义 |
|---|---|
| `include/SiTe/sifmt/sifmt_fp.hpp:277 / float16（类型别名）` | FP16 的指数位 5、尾数字段 10 |
| `include/SiTe/sifmt/sifmt_fp.hpp:278 / bfloat16（类型别名）` | BF16 的指数位 8、尾数字段 7 |
| `include/SiTe/sifmt/sifmt_fp.hpp:279 / float32（类型别名）` | FP32 的指数位 8、尾数字段 23 |
| `include/SiTe/sifmt/sifmt_fp.hpp:85 / SiFpBase::SiFpBase` | `OutputT(float)` 使用同一个 `FromFloat` 转换 |
| `include/SiTe/sifmt/sifmt_fp.hpp:113 / SiFpBase::operator InterFpType` | 转回 float 时调用 `ToFloat` |
| `include/SiTe/sifmt/quantize/quantize.hpp:307 / sifmt::f16::fromFloat` | 调用 FP32 -> FP16 软件转换 |
| `include/SiTe/sifmt/quantize/quantize.hpp:322 / sifmt::bf16::fromFloat` | 调用 FP32 -> BF16 软件转换 |
| `include/SiTe/sifmt/quantize/nonmx.hpp:364 / f32_to_f16` | 选择 round-to-nearest-even 舍入模式 |
| `include/SiTe/sifmt/quantize/nonmx.hpp:515 / f32_to_bf16` | 同样选择 round-to-nearest-even |
| `include/SiTe/sifmt/quantize/nonmx.hpp:95 / f16_to_f32` | 将已有 FP16 数值扩展为 FP32；有限 FP16 数值可以精确表示于 FP32 |
| `include/SiTe/sifmt/quantize/nonmx.hpp:162 / bf16_to_f32` | 将已有 BF16 数值扩展为 FP32；有限值的尾数字段在第 198 行左移 16 位 |
| `include/SiTe/sifmt/quantize/quantize.hpp:331 / sifmt::f32::toFloat` | FP32 wrapper 到 float 只做位表示转换，不会执行 BF16/FP16 舍入 |

**扩展回 FP32 并不能恢复第一次降精度时丢弃的信息。**有限 BF16/FP16 值转成 FP32 本身可精确，但“已舍入值的精确扩展”不等于“找回舍入前的 FP32”。

两个精确可复现的小例子，分别标明 FP32 的 32-bit 编码与中间 OutputT 的 16-bit 编码：

```text
BF16 path:
  FP32 1.00390625  [0x3f808000]
      -> BF16 1.0 [16-bit encoding 0x3f80]
      -> FP32 1.0 [0x3f800000]

FP16 path:
  FP32 1.00048828125 [0x3f801000]
      -> FP16 1.0   [16-bit encoding 0x3c00]
      -> FP32 1.0   [0x3f800000]
```

它们正好位于两个目标精度值的中点，由 ties-to-even 舍入到 1.0。若原 FP32 值恰好可由 OutputT 精确表示，则对齐坐标后可以相同；但这是对数据的条件，不是所有 GEMM 结果都有的保证。NaN 的 payload/canonicalization 另有规则，也不能承诺任意 NaN 编码逐位保真。

同样，不能为了省步骤而把 `gold_tiled_tensor.toVectorAsMemoryOrder(true)` 的每项直接转成 OutputT，就认为一定得到需要的 OutputT 物理 golden：**FP32 tensor 和 OutputT tensor 的 tile/subtile 形状不同**，9.4.1 的表格已经显示这种差异。当前代码先导出逻辑顺序再按 OutputT 布局重新构造，正是为了同时处理 dtype 和 layout。

### 9.6 现有比较为什么还要再次转为 OutputT

`testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:118 / run_case` 的分支是：

```cpp
if constexpr (LayoutC.tensor_format == 1) {
    gold = static_cast<OutputT>(gold_tiled[i]);
} else {
    gold = OutputT(gold_linear[i]);
}
```

这不是在验证 `gold_linear` 和 `gold_tiled` 两个 FP32 vectors 相同，而是在选择与设备输出一致的**存储顺序和输出精度**：

| 当前用例 | 设备 C 格式 | 比较所用 golden |
|---|---|---|
| BF16/R8，`testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:140 / run_bf16_default_case` | `kLayoutC_R8.tensor_format=0`，linear | 第 121 行：对 `gold_linear` 按需执行一次 BF16 转换 |
| FP16/R32，`testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:145 / run_default_case` | `kLayoutC_R32.tensor_format=1`，tiled | 第 119 行：把 `gold_tiled` 中已舍入、又扩展为 float 的值转回 FP16 |

对于有限、已经可由 OutputT 表示的值，同一转换下有：

```text
Q_T(U_T(Q_T(x))) = Q_T(x)
```

所以 tiled 分支最后这次回到 OutputT 一般不是再损失一轮有效信息，而是恢复之前已经得到的 16-bit 输出编码。正确的对应关系是：

```text
OutputT(gold_linear[L(m,n)])
      == OutputT(gold_tiled[P_T(m,n)])
```

这里两边要比较同一个逻辑坐标；它**不等价于** `gold_linear[L]` 与 `gold_tiled[P]` 的 FP32 编码相同。

还有一点容易从打印误判：BF16 用例虽然也构造并打印了 tiled 的 `out_tensor_d`，但它的设备 C 输出仍是 linear。这个 host reference tensor 的 layout，不会修改 `mma_dte` 的 `LayoutC`。相应常量在 `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:35 / kLayoutC_R8（常量）` 和 `:37 / kLayoutC_R32（常量）`。

### 9.7 mm_mnk 的 LayoutTag 明确作用于返回值

**SDK** `include/SiTe/tensor/tensor.hpp:2679 / mm_mnk` 定义为：

```cpp
mm_mnk(const TensorT1& t1,
       const TensorT2& t2,
       sipu::LayoutTag layoutTag = sipu::LayoutTag::Linear)
```

可以沿数据使用位置直接判断参数语义：

| SDK 相对路径:行号 / 函数名 | 对应行为 |
|---|---|
| `include/SiTe/tensor/tensor.hpp:2685 / mm_mnk` | 从 A/B 自身的 `layout().shape()` 取 shape |
| `include/SiTe/tensor/tensor.hpp:2698 / mm_mnk` | 调用 A/B 的 `toVector()`；它们各自已有的 layout 决定如何取数据，第三个参数不参与这一步 |
| `include/SiTe/tensor/tensor.hpp:2702 / mm_mnk` | 建立结果 shape `[A.rows, B.rows]`，即 `[M,N]` |
| `include/SiTe/tensor/tensor.hpp:2703 / mm_mnk` | `make_layout(shapeC, layoutTag)`：第三个参数首次用于创建 C 的 layout |
| `include/SiTe/tensor/tensor.hpp:2707 / mm_mnk` | 按 `i,j,k` 遍历，计算 `sum += A[i,k] * B[j,k]`，即 `C=A@B^T` |
| `include/SiTe/tensor/tensor.hpp:2722 / mm_mnk` | `make_tensor<sifmt::float32>(layoutC, buffer)`：返回 **FP32、所选 layout** 的 Tensor |

所以：

```text
mm_mnk(A, B)          -> FP32 Tensor, Linear
mm_mnk(A, B, Linear)  -> FP32 Tensor, Linear
mm_mnk(A, B, Tiled)   -> FP32 Tensor, Tiled

For the same A/B:
  logical matrix result: unchanged
  result dtype:          FP32 in all three calls
  result storage layout: selected by the third argument
  input layouts:         remain properties of A/B
```

第 2687 行拒绝 `LayoutTag::Manual`；不应把 Manual 当成这里可任意指定的结果布局。源码第 2668 行注释写着 “always return float32 tensor with linear layout”，但它没有反映当前带 `layoutTag` 参数的实现，**应以第 2703、2722 行执行代码为准**。

本例在 `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:52 / run_case`、`:62 / run_case` 已分别为 A/B 选择 tiled；第 68 行的 `Tiled` 选择 CPU golden 的布局；第 71 行的 `Tiled` 又独立选择 `out_tensor_d` 的布局；设备 `mma_dte` 输出则由第 104 行模板实参 `LayoutC` 选择。这是四个不同位置的配置，不能混用。

### 9.8 纯 CPU 复现结果与两个重要边界

#### 9.8.1 用指定 SDK 验证索引与精度，不重跑设备 kernel

本次在临时目录 `/tmp/sipu-layout-check-elGl9E` 创建了独立 host 检查程序。临时代码相对该目录的 `check.cpp:14 / run_case` 复用了当前测试的随机生成顺序、MXINT8 Tensor 构造、两组 M/N/K 以及上述 golden 链路；`check.cpp:65 / padding_case` 检查 padding；`check.cpp:77 / main` 检查舍入反例。没有改动 DTE 测试文件。

编译器为 `/usr/bin/c++` GCC 11.4.0，使用指定 SDK 2609151958 的 SiTe 头文件和 `SITE_TARGET_SIPU_ARCH=150`。执行命令是：

```bash
/usr/bin/c++ -std=c++20 -O2 -DSITE_TARGET_SIPU_ARCH=150 -DTARGET_SIPU_ARCH=150 \
  -I/share_data/sicx_sdk/release/2609151958/include/SiTe \
  /tmp/sipu-layout-check-elGl9E/check.cpp -o /tmp/sipu-layout-check-elGl9E/check
/tmp/sipu-layout-check-elGl9E/check
```

复现结果如下，统计的是**本次 CPU 检查**，不是 `run.log` 中未打印的设备数据：

| 检查项目 | BF16/R8 | FP16/R32 |
|---|---:|---:|
| 元素数 | 4608 | 18432 |
| 同逻辑坐标的原 FP32 与舍入后 FP32，编码不同的元素数 | 4561 | 17495 |
| 舍入后值为正/负 Inf 的元素数 | 0 | 10063 |
| 9.4 的索引公式与 layout engine 地址结果不一致数 | 0 | 0 |
| 对齐坐标后 `out_tensor_d.toVector()` 与 `gold_tiled` 的 float 编码不一致数 | 0 | 0 |
| `gold_tiled` 转回 OutputT 后与 tensor 内原始 16-bit 编码不一致数 | 0 | 0 |
| `OutputT(gold_linear[L])` 与相应物理位置 16-bit 编码不一致数 | 0 | 0 |
| `mm_mnk(...,Linear).toVector()` 与 `mm_mnk(...,Tiled).toVector()` 编码不一致数 | 0 | 0 |

这也说明：**逻辑矩阵可以相同，存储顺序可以不同；最终 16-bit 编码可以相同，而中途的原 FP32 与扩展后的 FP32 可以不同。**

本次 CPU 检查的具体元素反例：

```text
BF16/R8, [0,0], logical index = physical index = 0:
  gold_linear[0] = 22715   (FP32 0x46b17600)
  gold_tiled[0]  = 22656   (FP32 0x46b10000)
  stored BF16 encoding = 0x46b1

FP16/R32, [0,0], logical index = physical index = 0:
  gold_linear[0] = 87832   (FP32 0x47ab8c00)
  gold_tiled[0]  = +Inf    (FP32 0x7f800000)
  stored FP16 encoding = 0x7c00

FP16/R32, [0,2], logical index = physical index = 2:
  gold_linear[2] = -34408 (FP32 0xc7066800)
  gold_tiled[2]  = -34400 (FP32 0xc7066000)
```

这些例子故意选在逻辑/物理索引相同的位置，排除了“只是顺序不同”的解释，直接展示舍入或溢出造成的差别。随机分布实现和编译环境变化可能改变具体统计数值，不影响前面的源码结论。

#### 9.8.2 true 不保证返回完整 padding 字节布局

虽然参数注释称包含 padding，当前 **SDK** `include/SiTe/tensor/tensor.hpp:2194 / Tensor::toVectorAsMemoryOrder` 对完全没有有效坐标的 block 会直接 `continue`；第 2217 行仅补已有有效元素 block 的剩余位置，而且补的是新构造的 NaN，**不是读取物理 padding 字节**。

因此不能泛化成 `toVectorAsMemoryOrder(true).size() == storageSize()/sizeof(OutputT)` 对所有 shape 都成立。本次 `check.cpp:65 / padding_case` 的 BF16 `[3,17]` 反例是：

```text
logical elements                    = 3*17 = 51
allocated tiled storage elements    = 8*64 = 512
toVector().size()                   = 51
toVectorAsMemoryOrder(false).size()  = 51
toVectorAsMemoryOrder(true).size()   = 96
NaN placeholders in the last vector = 45
```

当前两组 `[8,576]`、`[32,576]` 无此 padding 问题，9.4 的物理索引对应关系成立。但将 `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:83 / run_case` 的输出大小计算直接推广到任意未对齐 shape 时，需要重新核对 padding 约定。

#### 9.8.3 日志 PASS 不能解释成 bitwise 一致

`temp/run.log:122`、`:242`、`:247` 确实记录两项测试通过；但当前 `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:125 / run_case` 至第 128 行计算浮点相对误差，再判断 `err_rate > 0.0f`，不是比较原始编码。

至少有两个区别：

- `+0.0` 和 `-0.0` 数值比较相等，但编码不同。
- NaN 不满足 `> 0`。例如 golden 是 `+Inf`、结果却是某个有限值时，`(Inf-finite)/Inf` 得到 NaN，这个条件不会报告失败；结果自身是 NaN 时同样可能漏报。

FP16 范围溢出的实现依据是 **SDK** `include/SiTe/sifmt/quantize/nonmx.hpp:205 / softfloat_roundPackToF16_sub`，第 219 行采用 overflow 模式，第 257 行可生成无穷大。上面的 CPU 复现证明这不是与当前数据无关的理论边界，而是本例 FP16 golden 中确实会发生的情况。

若今后要验证真正 bitwise 相同，应先让两边对应相同物理/逻辑坐标，再比较 OutputT 的 16-bit 编码，并明确 NaN、Inf、signed zero 的测试政策；若验证数值误差，则应单独处理非有限数。这里指出的是测试判据的边界，**没有据此断言本次设备输出错误，也没有修改测试代码**。

### 9.9 最终结论

`gold_linear` 是 FP32 CPU golden 的逻辑顺序导出；`gold_tiled` 是同一逻辑结果先舍入成 OutputT、按 OutputT 布局存储，再按物理数据顺序扩展回 float 的导出。它们不能保证原 FP32 编码相同。当前测试真正需要的是“与设备输出格式一致的、舍入到 OutputT 后的 golden”，因此两个比较分支分别使用不同顺序，并最终都转到 OutputT。

`mm_mnk` 的第三个 `LayoutTag::Tiled` 仅选择返回的 **FP32 CPU Tensor** 的存储 layout。它既不重设输入 A/B 的 layout，也不替代设备 `mma_dte` 的 `LayoutC`。

本次已完成源码和日志核对、独立 CPU SiTe 验证，并追加本文；没有修改算子或测试源码，没有重新构建主库或运行设备 GEMM。

## 10. SiTe uint16 Tiled Layout 与 CUTLASS CuTe Layout 对比验证（2026-09-17）

### 10.1 验证文件和测试范围

本次从 `temp/sifmt/test_sifmt_uint16_layout.cpp` 复制出：

```text
temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp
```

原来的逻辑保持不变：

- `test_sifmt_uint16_layout_vs_cute_layout.cpp:144 / run_test` 创建 `std::vector<uint16_t>`。
- `test_sifmt_uint16_layout_vs_cute_layout.cpp:145 / run_test` 遍历完整逻辑范围。
- `test_sifmt_uint16_layout_vs_cute_layout.cpp:146 / run_test` 填充 `arr_A[i] = i`。
- `test_sifmt_uint16_layout_vs_cute_layout.cpp:149 / run_test` 构造 SiTe `sifmt::uint16` tiled Tensor。
- `test_sifmt_uint16_layout_vs_cute_layout.cpp:150 / run_test` 导出 logical vector。
- `test_sifmt_uint16_layout_vs_cute_layout.cpp:151 / run_test` 导出 physical memory-order vector。
- `test_sifmt_uint16_layout_vs_cute_layout.cpp:159 / run_test` 仍保留原有 vector 打印。

新增比较覆盖两个 shape：

```text
8 x 576:  4608 logical indices
32 x 576: 18432 logical indices
```

### 10.2 CuTe Layout API 的使用方式

CUTLASS 本地实现位于 `include/cute/layout.hpp`，相关接口为：

| 相对 CUTLASS 路径:行号 / 函数或类型 | 作用 |
|---|---|
| `include/cute/layout.hpp:47 / cute::Shape` | `cute::tuple` 的 shape 别名 |
| `include/cute/layout.hpp:56 / cute::Coord` | `cute::tuple` 的 coordinate 别名 |
| `include/cute/layout.hpp:64 / cute::make_shape` | 构造可能嵌套的 shape |
| `include/cute/layout.hpp:70 / cute::make_stride` | 构造与 shape 对应的 stride |
| `include/cute/layout.hpp:98 / cute::Layout` | 保存 shape 与 stride 的 layout 类型 |
| `include/cute/layout.hpp:161 / cute::Layout::operator()` | 接收 coordinate，并在第 171 行调用 `crd2idx` 得到 linear physical index |
| `include/cute/layout.hpp:332 / cute::make_layout` | 用 shape 与 stride 创建 `cute::Layout` |
| `include/cute/stride.hpp:47 / cute::crd2idx` | 对整数或嵌套 tuple coordinate 执行 coordinate-to-index 映射 |
| `include/cute/stride.hpp:52 / cute::crd2idx` | 当 coordinate 是整数、shape/stride 是 tuple 时，对 logical mode 做 div/mod 分解 |
| `include/cute/stride.hpp:54 / cute::crd2idx` | 当 coordinate、shape、stride 都是 tuple 时，分别计算每个 mode 并求和 |

因此本验证没有假设 CuTe `Layout` 的 `operator()` 能直接接收 SiTe 的扁平 `(row,col)` 并用一组二维常量 stride 完成映射。SiTe 物理 index 含有：

```text
tile_n      = col / tile_width
subtile_n   = (col % tile_width) / block_width
element_n   = col % block_width
```

所以用 CuTe 的嵌套 shape/stride 表示分解后的 coordinate，再让 `cute::Layout::operator()` 计算 physical index。这仍是标准 CuTe Layout 映射，只是 logical coordinate 使用层次形式。

### 10.3 8x576 的 CuTe Layout

新增代码位于 `temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:37 / make_cute_layout` 至第 41 行：

```cpp
shape  = (8, (9, 4, 16))
stride = (16, (512, 128, 1))
```

逻辑坐标在 `temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:55 / make_cute_coord` 至第 59 行构造成：

```cpp
coord = (row, (col / 64, (col % 64) / 16, col % 16))
```

CuTe `operator()` 的结果为：

```text
P8(row,col)
  = row * 16
  + (col / 64) * 512
  + ((col % 64) / 16) * 128
  + (col % 16)
```

这等价于 SiTe 当前 16-bit、M=8、tile `[8,64]` 的 physical 顺序：

```text
tile0/subtile0: [0,0..15], [1,0..15], ..., [7,0..15]
tile0/subtile1: [0,16..31], ..., [7,16..31]
tile0/subtile2: [0,32..47], ..., [7,32..47]
tile0/subtile3: [0,48..63], ..., [7,48..63]
tile1:         columns 64..127
...
```

### 10.4 32x576 的 CuTe Layout

新增代码位于 `temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:42 / make_cute_layout` 至第 47 行：

```cpp
shape  = ((4, 8), (36, 16))
stride = ((128, 16), (512, 1))
```

逻辑坐标在 `temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:60 / make_cute_coord` 至第 62 行构造成：

```cpp
coord = ((row / 8, row % 8), (col / 16, col % 16))
```

CuTe `operator()` 的结果为：

```text
P32(row,col)
  = (row / 8) * 128
  + (row % 8) * 16
  + (col / 16) * 512
  + (col % 16)
```

这等价于 SiTe 当前 16-bit、M=32、tile `[32,16]` 的 physical 顺序：

```text
tile0: [0,0..15], [1,0..15], ..., [31,0..15]
tile1: [0,16..31], [1,16..31], ..., [31,16..31]
tile2: columns 32..47
...
```

### 10.5 逐 logical index 的实际比较

比较函数位于 `temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:66 / compare_with_cute`：

| 相对路径:行号 / 函数名 | 行为 |
|---|---|
| `temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:69 / compare_with_cute` | 创建当前 shape 对应的 CuTe layout |
| `temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:73 / compare_with_cute` | 从 `logical_index=0` 遍历到 logical vector 末尾，覆盖全部逻辑元素 |
| `temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:75 / compare_with_cute` | 计算 `row = logical_index / 576` |
| `temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:76 / compare_with_cute` | 计算 `col = logical_index % 576` |
| `temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:77 / compare_with_cute` | 将 flat `(row,col)` 拆成 CuTe 层次 coordinate |
| `temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:78 / compare_with_cute` | 调用 `cute_layout(cute_coord)`，得到 physical index |
| `temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:81 / compare_with_cute` | 检查 CuTe physical index 是否超出 SiTe physical vector |
| `temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:86 / compare_with_cute` | 从 SiTe logical vector 取当前 logical value，并转换为 `uint16_t` |
| `temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:88 / compare_with_cute` | 用 CuTe physical index 索引 SiTe physical vector，并转换为 `uint16_t` |
| `temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:90 / compare_with_cute` | 比较两个 `uint16_t` value；不相等时记录 mismatch |
| `temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:102 / compare_with_cute` | 输出当前 shape 的元素数、mismatch 数、越界数和最终结果 |

比较逻辑对应用户要求的关系：

```text
site_logical_index_value
    = tile_tensor_to_vector[logical_index]

cute_physical_index
    = cute_layout(cute_coord(row, col))

cute_index_value
    = tile_tensor_to_vector_physical[cute_physical_index]

compare(cute_index_value, site_logical_index_value)
```

当前两个 SiTe exporter 返回类型实际是 `std::vector<float>`；由于 Tensor 中存的是 `sifmt::uint16`，这些 float 是精确的整数值。代码在 `temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:86 / compare_with_cute` 和第 88 行显式转换成 `uint16_t` 后比较，避免把本次验证误解为 BF16/FP16 浮点误差比较。

### 10.6 编译命令

第一次没有 CUDA include 路径的编译命令会在 CUTLASS `include/cute/util/debug.hpp:38 / include` 因找不到 `cuda_runtime_api.h` 失败。最终成功编译命令为：

```bash
cd /share/users/like/temp/sifmt
env SI_SDK_ROOT=/share_data/sicx_sdk/release/2609151958 \
  /usr/bin/c++ \
  -std=c++20 -O2 \
  -DSITE_TARGET_SIPU_ARCH=150 \
  -DTARGET_SIPU_ARCH=150 \
  -I/share_data/sicx_sdk/release/2609151958/include/SiTe \
  -I/share/users/like/package/cutlass/include \
  -I/usr/local/cuda-12.8/include \
  test_sifmt_uint16_layout_vs_cute_layout.cpp \
  -o test_sifmt_uint16_layout_vs_cute_layout
```

运行命令为：

```bash
cd /share/users/like/temp/sifmt
./test_sifmt_uint16_layout_vs_cute_layout > run_vs_cute.log 2>&1
```

这里使用的是 CUTLASS 本地头文件，不需要编译或链接 CUDA kernel；CuTe layout 的这部分代码可以由 host C++ 编译器实例化。CUDA include 目录只是满足 CUTLASS 的公共 debug/config 头文件依赖。

### 10.7 实际运行结果

运行日志位于：

```text
temp/sifmt/run_vs_cute.log
```

关键结果在日志第 21 行和第 93 行：

```text
cute_layout_compare shape=8x576 \
  logical_elements=4608 physical_elements=4608 \
  mismatches=0 out_of_range=0 result=PASS

cute_layout_compare shape=32x576 \
  logical_elements=18432 physical_elements=18432 \
  mismatches=0 out_of_range=0 result=PASS
```

本次比较覆盖：

| shape | logical index 范围 | physical vector 长度 | mismatch | 越界 | 结论 |
|---|---:|---:|---:|---:|---|
| `8x576` | `0..4607` | 4608 | 0 | 0 | PASS |
| `32x576` | `0..18431` | 18432 | 0 | 0 | PASS |

因此，在本次验证的两个完整 shape、`sifmt::uint16`、SIPU 1.5/`SITE_TARGET_SIPU_ARCH=150` 布局条件下，SiTe 的 tiled physical layout **可以用 CUTLASS CuTe Layout 表示**。更准确地说，是用嵌套的 shape/stride 和对应的层次 coordinate 表示；不是用一个扁平二维 `(M,N):(stride_m,stride_n)` affine layout 直接表示。

`temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp:84 / main` 运行 `run_test<8>` 和 `run_test<32>`；原始 vector 的打印也仍然保留，physical vector 现在每个 tile 打印一行，每个 element 宽度为 5 个字符。

本次追加只修改了 `/share/users/like/temp/sifmt/test_sifmt_uint16_layout_vs_cute_layout.cpp` 和本答案文档，没有修改 CUTLASS、SiTe SDK 或 mma_dte_tile_tensor 源码。

## 11. 直接从 gold_tiled_tensor 导出失败的原因（2026-09-18）

### 11.1 结论先行

这次 mismatch 的**主要原因是比较思路不等价，不是 `Tensor::toVectorAsMemoryOrder` 的直接实现错误**。

当前两条链路实际处理的是两个不同的 Tensor：

```text
链路 A：gold_tiled_tensor
  Tensor<sifmt::float32, FP32 tiled layout>
      -> toVectorAsMemoryOrder(true)
      -> gold_tiled_like_debug

链路 B：out_tensor_d
  gold_tiled_tensor.toVector()
      -> gold_linear: std::vector<float>，逻辑顺序
      -> make_tensor<OutputT>(layout_d, gold_linear)
      -> Tensor<OutputT, OutputT tiled layout>
      -> toVectorAsMemoryOrder(true)
      -> gold_tiled
```

因此 `gold_tiled_like_debug[i]` 与 `gold_tiled[i]` 同时存在两类不等价：

1. **dtype 不同**：前者是 FP32 原值，后者已经转换成 BF16/FP16 并重新扩展为 float。
2. **physical layout 不同**：FP32 和 16-bit dtype 的 SiTe tiled tile shape 不同，physical vector 的同一个下标 `i` 不代表同一个逻辑坐标。

所以当前代码的直接比较：

```cpp
gold_tiled_like_debug[i] == gold_tiled[i]
```

不是合法的同坐标比较。`gold_tiled_tensor.toVectorAsMemoryOrder(true)` 本身仍然是对**源 FP32 Tensor 自己的 layout**进行正确导出；它不能自动知道目标设备 C 是 FP16/BF16，也不会自动转换到 `LayoutC`。

### 11.2 当前源码中的实际链路

当前测试源码相对 DTE 根目录的路径是：

```text
testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp
```

| 相对路径:行号 / 函数名 | 实际行为 |
|---|---|
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:67 / run_case` | 调用 `sipu::tensor::mm_mnk(..., LayoutTag::Tiled)`，生成 `gold_tiled_tensor` |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:68 / run_case` | `gold_tiled_tensor` 是 host CPU 计算得到的 FP32 tiled Tensor |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:69 / run_case` | `gold_linear = gold_tiled_tensor.toVector()`，得到逻辑顺序 FP32 vector |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:72 / run_case` | 调用 `make_tensor<OutputT>(layout_d, gold_linear)`，完成 FP32 -> BF16/FP16 和目标 tiled layout 打包 |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:73 / run_case` | `gold_tiled = out_tensor_d.toVectorAsMemoryOrder(true)`，导出目标 OutputT Tensor 的 physical order |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:74 / run_case` | 新增的 `gold_tiled_like_debug = gold_tiled_tensor.toVectorAsMemoryOrder(true)`，导出源 FP32 Tensor 的 physical order |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:120 / run_case` | 只有 `LayoutC.tensor_format == 1` 的 tiled C 分支才比较 debug vector；当前 BF16/R8 的 C 是 linear，不进入该 debug 比较 |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:121 / run_case` | 正式 golden 使用 `gold_tiled[i]`，不是 `gold_tiled_like_debug[i]` |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:122 / run_case` | 计算两个 debug vector 的差值，但没有把 mismatch 加入 `err` |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:123 / run_case` | 只更新 `like_debug_err`，没有 `EXPECT`、`ASSERT` 或失败返回 |
| `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:128 / run_case` | 正式设备结果仍与 `gold` 比较 |

因此日志中的 GTest PASS 不能说明两个 debug vector 相等。当前 debug 仅打印诊断信息，不参与测试判定。

### 11.3 两个 Tensor 的 dtype 和 tile shape 确实不同

#### 11.3.1 gold_tiled_tensor 是 FP32 Tensor

**SDK** `include/SiTe/tensor/tensor.hpp:2679 / mm_mnk` 的返回路径如下：

| SDK 相对路径:行号 / 函数名 | 行为 |
|---|---|
| `include/SiTe/tensor/tensor.hpp:2698 / mm_mnk` | 把 A 解码成 FP32 vector |
| `include/SiTe/tensor/tensor.hpp:2699 / mm_mnk` | 把 B 解码成 FP32 vector |
| `include/SiTe/tensor/tensor.hpp:2711 / mm_mnk` | 用 `float sum` 做累加 |
| `include/SiTe/tensor/tensor.hpp:2703 / mm_mnk` | 用 `layoutTag` 创建结果 C layout |
| `include/SiTe/tensor/tensor.hpp:2722 / mm_mnk` | 明确返回 `make_tensor<sifmt::float32>(layoutC, buffer)` |

所以：

```text
gold_tiled_tensor.dtype = sifmt::float32
```

#### 11.3.2 out_tensor_d 是 BF16/FP16 Tensor

**SDK** `include/SiTe/tensor/tensor.hpp:1772 / Tensor::Tensor` 接收 float span 并分配目标 dtype 的存储；`include/SiTe/tensor/tensor.hpp:1950 / Tensor::fillWithContainer` 的普通非 MX 分支在第 2063 行调用 `DataType::FromFloat`。

所以：

```text
out_tensor_d.dtype = OutputT
OutputT = sifmt::bfloat16  // R8 test
OutputT = sifmt::float16   // R32 test
```

目标 dtype 定义在：

| SDK 相对路径:行号 / 类型 | 说明 |
|---|---|
| `include/SiTe/sifmt/sifmt_fp.hpp:277 / sifmt::float16` | 16-bit FP16 |
| `include/SiTe/sifmt/sifmt_fp.hpp:278 / sifmt::bfloat16` | 16-bit BF16 |
| `include/SiTe/sifmt/sifmt_fp.hpp:279 / sifmt::float32` | 32-bit FP32 |

#### 11.3.3 Tile shape 依赖 dtype bit width

**SDK** `include/SiTe/tensor/tensor.hpp:1097 / LayoutEngine::getTileShape` 根据 `DataType::kBitWidth` 计算 tile shape；`include/SiTe/tensor/tensor.hpp:1235 / LayoutEngine::LayoutEngine` 在第 1278 行选中该 tile shape。

当前日志中的实际 shape 是：

| 用例 | `gold_tiled_tensor`：FP32 tiled | `out_tensor_d`：OutputT tiled |
|---|---|---|
| BF16/R8，M=8 | tile `[8,32]`，18 个 tile | BF16 tile `[8,64]`，9 个 tile |
| FP16/R32，M=32 | tile `[32,8]`，72 个 tile | FP16 tile `[32,16]`，36 个 tile |

证据来自 `temp/run.log`：

- `temp/run.log:68` 至 `:90`：BF16/R8 的 FP32 `gold_tiled_tensor` 显示 `Shape(tile): [8,32]`、`BitWidth: 32`。
- `temp/run.log:96` 至 `:118`：BF16/R8 的 BF16 `out_tensor_d` 显示 `Shape(tile): [8,64]`、`BitWidth: 16`。
- `temp/run.log:188` 至 `:210`：FP16/R32 的 FP32 `gold_tiled_tensor` 显示 `Shape(tile): [32,8]`、`BitWidth: 32`。
- `temp/run.log:216` 至 `:238`：FP16/R32 的 FP16 `out_tensor_d` 显示 `Shape(tile): [32,16]`、`BitWidth: 16`。

这已经说明：两个 Tensor 的 physical vector 不能按同一个 `i` 直接比较。

### 11.4 首个 layout 错位点可以精确算出来

#### 11.4.1 FP16/R32 当前实际 debug 路径

FP32 源 Tensor 的 tile 是 `[32,8]`，所以其物理索引为：

```text
P_fp32_R32(m,n)
  = (n/8) * 256
  + (m/8) * 64
  + (m%8) * 8
  + (n%8)
```

FP16 目标 Tensor 的 tile 是 `[32,16]`，所以其物理索引为：

```text
P_fp16_R32(m,n)
  = (n/16) * 512
  + (m/8) * 128
  + (m%8) * 16
  + (n%16)
```

因此相同 physical index `i` 的逻辑坐标从 `i=8` 开始就不一样：

| physical index | FP32 source `gold_tiled_like_debug` | FP16 target `gold_tiled` |
|---:|---|---|
| `0..7` | `[0,0..7]` | `[0,0..7]` |
| `8..15` | `[1,0..7]` | `[0,8..15]` |
| `16..23` | `[2,0..7]` | `[1,0..7]` |
| `128..135` | `[16,0..7]` | `[8,0..7]` |
| `256..263` | `[0,8..15]` | `[16,0..7]` |

日志也显示了这个模式叠加精度变化后的结果：

```text
temp/run.log:242
i=0: gold_tiled_like_debug=87832, gold_tiled=inf

temp/run.log:244
i=2: gold_tiled_like_debug=-34408, gold_tiled=-34400

temp/run.log:252
i=10: gold_tiled_like_debug=55688, gold_tiled=-57280

temp/run.log:258
i=16: gold_tiled_like_debug=-130823, gold_tiled=-7316
```

其中：

- `i=0` 仍是同一个逻辑坐标，但 FP32 的 `87832` 转成 FP16 已经溢出为 `inf`。
- `i=2` 仍是同一个逻辑坐标，但 FP16 发生舍入：`-34408 -> -34400`。
- `i=10` 已经不是同一个逻辑坐标：FP32 physical 序列对应 `[1,2]`，FP16 physical 序列对应 `[0,10]`。
- `i=16` 更明显：FP32 physical 序列对应 `[2,0]`，FP16 physical 序列对应 `[1,0]`。

所以日志中的大差值不是 `toVectorAsMemoryOrder` 把一个数组随机打乱，而是两个不同 layout 的 physical index 被错误地拿来对齐。

#### 11.4.2 BF16/R8 也存在同样问题

虽然当前代码对 BF16/R8 的 `LayoutC_R8.tensor_format=0` 走 linear 分支，不打印 debug 比较，但如果比较：

```cpp
gold_tiled_tensor.toVectorAsMemoryOrder(true)
```

与 BF16 tiled `out_tensor_d.toVectorAsMemoryOrder(true)`，也会遇到：

```text
FP32 source tile:  [8,32]
BF16 target tile:  [8,64]
```

对应物理索引：

```text
P_fp32_R8(m,n)
  = (n/32) * 256
  + ((n%32)/8) * 64
  + m * 8
  + (n%8)

P_bf16_R8(m,n)
  = (n/64) * 512
  + ((n%64)/16) * 128
  + m * 16
  + (n%16)
```

例如 `physical index=8`：

```text
FP32 source: [1,0]
BF16 target: [0,8]
```

### 11.5 toVectorAsMemoryOrder 的实现是否有 bug

从当前现象判断，**不是本次 mismatch 的主要原因**。

**SDK** `include/SiTe/tensor/tensor.hpp:2141 / Tensor::toVectorAsMemoryOrder` 使用的是当前对象自己的 `mLayoutEngine`：

| SDK 相对路径:行号 / 成员函数 | 作用 |
|---|---|
| `include/SiTe/tensor/tensor.hpp:2148 / Tensor::toVectorAsMemoryOrder` | 读取当前 Tensor 自己的 tiled shape |
| `include/SiTe/tensor/tensor.hpp:2184 / Tensor::toVectorAsMemoryOrder` | 遍历当前 Tensor 自己的 plane/supertile |
| `include/SiTe/tensor/tensor.hpp:2189 / Tensor::toVectorAsMemoryOrder` | 按当前 dtype 的 supertile data storage 遍历 |
| `include/SiTe/tensor/tensor.hpp:2203 / Tensor::toVectorAsMemoryOrder` | 调用当前 Tensor 的 `at(coord)` 取值 |
| `include/SiTe/tensor/tensor.hpp:2210 / Tensor::toVectorAsMemoryOrder` | 将当前 Tensor 的元素转换为 float 放入 vector |

它没有目标 dtype 或目标 layout 参数，因此：

```text
gold_tiled_tensor.toVectorAsMemoryOrder(true)
```

必然按 FP32 source 的 layout 导出；它不可能自动按 `out_tensor_d` 的 FP16/BF16 layout 导出。

另外，SDK 实现中有一处独立的代码可疑点，值得后续单独修复/核对：

- `include/SiTe/tensor/tensor.hpp:2159 / Tensor::toVectorAsMemoryOrder` 将第三个返回值命名为 `blockIndex`，但该返回参数实际对应 `tensorCoord2StoragePtrOffset` 的 `indexInBlock`。
- `include/SiTe/tensor/tensor.hpp:2161 / Tensor::toVectorAsMemoryOrder` 用 `blockIndex` 查 map。
- `include/SiTe/tensor/tensor.hpp:2165 / Tensor::toVectorAsMemoryOrder` 却用 `dataPtr` 作为 map key 插入。

这会造成 map 查找 key 不一致，可能带来额外插入和 padding 路径风险。但对于本次两个完整对齐 shape，日志中的差异已经被 dtype/layout 公式完全解释，不能把这个独立问题当成当前 mismatch 的根因。

### 11.6 当前 GTest 为什么仍然 PASS

`temp/run.log:18674` 记录：

```text
[       OK ] MmaDteFloat16Mxint8Test.ComputesDefaultShape
```

`temp/run.log:18679` 记录：

```text
[  PASSED  ] 2 tests.
```

原因是 `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:122 / run_case` 至 `:124 / run_case` 只打印 debug 差异；没有修改 `err`。正式测试错误计数从 `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:116 / run_case` 开始，仍然只在 `:128 / run_case` 以后根据设备输出和已经正确转换/重排的 `gold` 判定。

此外，FP16 golden 中有 `inf`。当前 `run_case` 的相对误差逻辑对非有限值存在盲区：`inf` 与 `inf` 的减法/除法可能得到 NaN，而 `err_rate > 0.0f` 不会成立。因此 debug mismatch 与 GTest PASS 可以同时出现；PASS 不能证明 `gold_tiled_like_debug` 和 `gold_tiled` 相等。

### 11.7 正确的直接优化方向

#### 11.7.1 不能直接替换成这一行

下面这行只能得到 FP32 source Tensor 的 physical vector：

```cpp
auto gold_tiled_like_debug = gold_tiled_tensor.toVectorAsMemoryOrder(true);
```

它不能替换当前目标输出 golden：

```cpp
auto out_tensor_d = sipu::make_tensor<OutputT>(layout_d, gold_linear);
auto gold_tiled = out_tensor_d.toVectorAsMemoryOrder(true);
```

#### 11.7.2 可以设计一个“目标 dtype/layout 感知”的直接转换 API

真正可以优化的是去掉中间 `gold_linear` vector，但仍必须同时完成两件事：

1. 按 source Tensor 的逻辑坐标读取 FP32 值。
2. 按目标 `OutputT + layout_d` 将值转换、打包并按目标 physical 顺序输出。

抽象上应类似：

```text
source FP32 tiled Tensor
  -- logical coordinate read --> float value
  -- OutputT conversion ------> BF16/FP16 value
  -- target layout write -----> target physical vector
```

可以新增一个明确带目标类型和目标 layout 的 helper，例如概念上的：

```cpp
exportAsMemoryOrder<OutputT>(source_tensor, target_layout, true);
```

这个 helper 不能只调用 source 的 `toVectorAsMemoryOrder`；它必须使用 target Tensor 的 `LayoutEngine` 或等价的 target layout mapping。现有 `out_tensor_d` 路径虽然多了一个 logical `std::vector<float>`，但语义是正确的。

#### 11.7.3 什么时候 direct export 才能直接复用

只有以下条件同时成立时，source direct export 才能直接与目标 physical vector 比较：

```text
source dtype == target dtype
source tiled layout == target tiled layout
source padding policy == target padding policy
```

例如 source 和 target 都是同一个 FP32 tiled layout，直接 export 才能按同一 physical index 对齐。当前测试是 FP32 source 对 BF16/FP16 target，不满足前两个条件。

### 11.8 构建版本范围

题目指定 SDK 是 `2609151958`，但本次提供的实际日志使用的是 SDK **2609161442**：

- `temp/inc.log:18` 的 `SI_SDK_ROOT` 是 `/share_data/sicx_sdk/release/2609161442`。
- `temp/inc.log:26` 的 testcase SDK tag 是 `2609161442`。
- `temp/inc.log:5918` 显示 `Built target test_bf16_mxi8_host`。
- `temp/run.log:1` 加载的 runtime 是 `/share_data/sicx_sdk/release/2609161442/lib/libsipu.so.0`。

本次检查确认指定 SDK `2609151958` 与日志 SDK `2609161442` 的相关 `SiTe/tensor/tensor.hpp`、`sifmt/sifmt_fp.hpp`、`sifmt/quantize/nonmx.hpp` 内容相同；但运行时结论严格属于日志中的 2609161442 环境，不能把它描述成已经在 2609151958 runtime 上执行。

### 11.9 最终结论

```text
gold_tiled_tensor.toVectorAsMemoryOrder(true)
```

这个调用思路作为“导出 FP32 source Tensor 的 physical layout”没有问题；问题在于把它当成了“导出 OutputT 目标 Tensor 的 physical golden”。

本次失败的根因按优先级是：

1. **dtype 不同**：FP32 vs FP16/BF16，导致舍入和 FP16 overflow。
2. **tile layout 不同**：FP32 的 tile shape 与 OutputT 的 tile shape 不同，导致同一 vector 下标对应不同逻辑坐标。
3. **debug 比较没有做 logical coordinate 对齐，也没有做 OutputT 转换**。

因此，应判断为**替换思路的语义不成立**，不是当前观察到的主要 SiTe exporter bug。若目标是优化性能，应实现一个显式的“source logical read + target dtype conversion + target layout pack”接口，而不是直接调用 source Tensor 的 `toVectorAsMemoryOrder`。

## 12. 五类 MX 输入的 R8/R16/R32 mma_dte 最小形状与输出格式（2026-09-18）

### 12.1 结论口径

本节覆盖当前 `mma_dte_tile_tensor` 仓库中的：

```text
MXINT8: sifmt::mxint8
MXFP8 : sifmt::mxfloat8e4m3 / sifmt::mxfloat8e5m2
MXFP6 : sifmt::mxfloat6e2m3 / sifmt::mxfloat6e3m2
MXFP4 : sifmt::mxfloat4e2m1
MXINT4: sifmt::mxint4
```

约定矩阵逻辑形状为：

```text
A: [M,K]
B: [N,K]
C: [M,N] = A @ B^T
```

下表回答的是当前模板 `mma_dte<...>` 主执行路径在 **A/B 都是 tiled MX input** 时的 layout/API 最小形状。它不是旧接口 `mma_bf16_mxi8_universal` 的 `K >= 128` 规则，也不是把三个 R32 layout 的最小分量强行合并成一个不存在的形状。

如果 A/B 使用 `tensor_format=0` 的 linear input，不能直接套用本节的 layout 元组；linear input 还要满足 `kernel/mma_dte_tiled_tensor.hpp:83-85 / mma_dte_implement_with_control` 的 R32 偶数 K-tile stride 约束，并使用另一组 A/B TensorMap 语义。本节沿用当前 MX testcase 的 tiled A/B 输入场景，只比较 C 的 linear/tiled 输出。

### 12.2 形状约束来源和计算公式

当前主入口在 `kernel/mma_dte_tiled_tensor.hpp:54-119 / mma_dte_implement_with_control` 中检查：

```text
M % (layoutA.tile_dim1 * layoutA.supertile_shape1) == 0
N % (layoutB.tile_dim1 * layoutB.supertile_shape1) == 0
K % (layoutA.tile_dim0 * layoutA.supertile_shape0) == 0
K % (layoutB.tile_dim0 * layoutB.supertile_shape0) == 0
```

因此最小对齐值为：

```text
M_min = layoutA.tile_dim1 * layoutA.supertile_shape1
N_min = layoutB.tile_dim1 * layoutB.supertile_shape1
K_min = lcm(layoutA.tile_dim0 * layoutA.supertile_shape0,
            layoutB.tile_dim0 * layoutB.supertile_shape0)
```

对应的 layout contract 在 `kernel/detail/mma_dte_tiled_tensor_layout_contracts.hpp:147-182 / mma_dte_layout_contract`。MX 输入必须占满规定的输入 supertile；当前 `kernel/detail/mma_dte_tiled_tensor_layout_contracts.hpp:117-145 / mma_dte_expected_input_supertile_tiles` 和 `mma_dte_mx_2x2_shape_supported` 分别约束 4-tile MX supertile 以及 R32 2x2 的额外条件。

对于 R32 2x2，除了对齐公式外还要求 `M > 16`、`N > 16`、`K == 2 * tile_k`，见 `kernel/mma_dte_tiled_tensor.hpp:71-76 / mma_dte_implement_with_control`。下表中的 2x2 最小值正好满足这些条件。

注意 MXFP6 在 `kernel/detail/mma_dte_tiled_tensor_dtype_meta.hpp:117-136 / mma_dte_physical_tile_k` 中有 SIPU160 的内部物理 K tile 特殊处理；但公开 `tensor_layout` 的 M/N/K API 对齐仍由 `mma_dte_implement_with_control` 上述检查决定，所以本节的外部最小 K 按显式 layout 元组计算。

### 12.3 输出 C 的 layout 以及 linear/tiled 的含义

`layoutC.tensor_format` 的含义是输出格式：`0` 是 linear，`1` 是 tiled。当前输出 payload contract 要求 C 的物理 tile 为 1024 bytes，且 `layoutC.supertile_shape0/1 == 1`、`layoutC.tile_dim1 == layoutA.tile_dim1`，见 `kernel/detail/mma_dte_tiled_tensor_layout_contracts.hpp:159-181 / mma_dte_layout_contract`。

典型 C layout 如下：

| row family | BF16/FP16 C layout | FP32 C layout |
|---|---|---|
| R8 | `tensor_layout(64,8,1,1,fmt)` | `tensor_layout(32,8,1,1,fmt)` |
| R16 | `tensor_layout(32,16,1,1,fmt)` | `tensor_layout(16,16,1,1,fmt)` |
| R32 | `tensor_layout(16,32,1,1,fmt)` | `tensor_layout(8,32,1,1,fmt)` |

其中 `fmt=0` 是 linear，`fmt=1` 是 tiled。当前显式实例化的输出类型范围是：

| MX input | 当前仓库显式实例化的 outputT |
|---|---|
| MXINT8 | BF16、FP16、FP32 |
| MXFP8 | BF16、FP16 |
| MXFP6 | BF16、FP16、FP32 |
| MXFP4 | BF16 |
| MXINT4 | BF16 |

linear 和 tiled 的**逻辑矩阵结果相同**，但 C 的物理内存排列不同，不能直接按同一个 `std::vector` 下标做 bitwise 比较。R8 的当前主 dispatch 在 `kernel/detail/mma_dte_tiled_tensor_dispatch_entrypoints.hpp:285-324 / DEFINE_FUSED_IMPL_FIXED_CK_R8_DUAL` 和 `:440-482 / DEFINE_FUSED_IMPL_FIXED_CK_R8_DUAL_BY_FAMILY` 中根据 `layoutC.tensor_format` 选择 linear store 或 tiled store。

### 12.4 MXINT8

当前 layout 来源包括 `kernel/instantiations/mxint8/inst_mxint8_r8.su:21-27 / mma_dte`、`inst_mxint8_r16.su:21-27 / mma_dte` 以及 `inst_mxint8_r32_4x1.su:21-27 / mma_dte`、`inst_mxint8_r32_2x2.su:21-27 / mma_dte`、`inst_mxint8_r32_1x4.su:21-27 / mma_dte`。

本节三元组统一写成 `[min_value,max_value,step]`，区间为左闭右闭。`inf` 表示当前 dtype/layout contract 没有有限上界；仍受正 `int` 参数、内存、地址和 `uint32_t total_Chunks` 等工程限制。唯一允许值写成 `[x,x,None]`。用户示例中的 `[8,0,None]` 与左闭右闭语义矛盾，这里采用 `[8,8,None]`。

| row family | layoutA | layoutB | M | N | K |
|---|---|---|---:|---:|---:|
| R8 | `(128,8,4,1,1)` | `(32,32,4,1,1)` | `[8,8,None]` | `[32,inf,32]` | `[512,inf,512]` |
| R16 | `(64,16,4,1,1)` | `(32,32,4,1,1)` | `[16,16,None]` | `[32,inf,32]` | `[256,inf,256]` |
| R32 standard 4x1 | `(32,32,4,1,1)` | `(32,32,4,1,1)` | `[32,inf,32]` | `[32,inf,32]` | `[128,inf,128]` |
| R32 2x2 | `(32,32,2,2,1)` | `(32,32,2,2,1)` | `[64,inf,64]` | `[64,inf,64]` | `[64,64,None]` |
| R32 1x4 | `(32,32,1,4,1)` | `(32,32,1,4,1)` | `[128,inf,128]` | `[128,inf,128]` | `[32,32,None]` |

实际含义：R8/R16 的 M 被当前 SiTe short-M MX packer 固定为单 tile；N/K 可以按步长增加。R32 4x1 的 M/N/K 都可以增加。R32 2x2 的 M/N 可以增加，但 K 严格固定为单个 2x2 supertile 的 K。R32 1x4 当前实现只有单 K tile 的正确结果，K 增加到 64 时实际结果错误。

R32 的最小 Pareto 形状仍是：

```text
4x1: (M,N,K) = (32, 32,128)
2x2: (M,N,K) = (64, 64, 64)
1x4: (M,N,K) = (128,128, 32)
```

R32 2x2 的 K 单点上界来自 `kernel/detail/mma_dte_tiled_tensor_layout_contracts.hpp:139-145 / mma_dte_mx_2x2_shape_supported` 的 `k == 2 * mma_dte_physical_tile_k()`；新增 death test 入口见 `testcase/func_test/test_mx_shape_bounds_host/test_mx_shape_bounds_host.cpp:288 / MxInt8R32TwoByTwo.KAboveSingleSupertileIsRejected`。R32 1x4 的 K 单点是当前实现的实测边界，MXINT8 负例见 `testcase/func_test/test_mx_shape_bounds_host/test_mx_shape_bounds_host.cpp:305 / MxInt8R32OneByFour.KAboveSingleTileProducesInvalidResult`。

MXINT8 的 batch testcase 用 `testcase/func_test/test_bf16_mxint8_host/batch_test.py:64-68 / SUPERTILE_LAYOUT_B`、`:110-140 / resolve_layout_c、resolve_layout_a` 生成这三类 input layout 和 C layout。

### 12.5 MXFP8

MXFP8 的 `e4m3` 与 `e5m2` 使用相同的 layout 几何，因此最小 M/N/K 相同。当前 layout 定义可见 `kernel/instantiations/mxfp8/inst_mxfp8_r8_bf16.su:4-7 / mma_dte`、`inst_mxfp8_r16_bf16.su:4-7 / mma_dte` 以及 R32 的 `inst_mxfp8_r32_4x1_bf16.su:4-7 / mma_dte`、`inst_mxfp8_r32_2x2_bf16.su:4-7 / mma_dte`、`inst_mxfp8_r32_1x4_bf16.su:4-7 / mma_dte`。

| row family | layoutA | layoutB | M | N | K |
|---|---|---|---:|---:|---:|
| R8 | `(128,8,4,1,1)` | `(32,32,4,1,1)` | `[8,8,None]` | `[32,inf,32]` | `[512,inf,512]` |
| R16 | `(64,16,4,1,1)` | `(32,32,4,1,1)` | `[16,16,None]` | `[32,inf,32]` | `[256,inf,256]` |
| R32 standard 4x1 | `(32,32,4,1,1)` | `(32,32,4,1,1)` | `[32,inf,32]` | `[32,inf,32]` | `[128,inf,128]` |
| R32 2x2 | `(32,32,2,2,1)` | `(32,32,2,2,1)` | `[64,inf,64]` | `[64,inf,64]` | `[64,64,None]` |
| R32 1x4 | `(32,32,1,4,1)` | `(32,32,1,4,1)` | `[128,inf,128]` | `[128,inf,128]` | `[32,32,None]` |

linear/tiled C 的最小 M/N/K 在当前主入口相同。MXFP8 R32 1x4 的单点 K 上界由 `kernel/detail/mma_dte_tiled_tensor_planner.hpp:546-559 / validate_layout_k_contract` 明确限制为一个 K tile，即 `K <= layoutA.tile_dim0`，因此是 `[32,32,None]`。R32 2x2 的 K 也由 `mma_dte_mx_2x2_shape_supported` 固定为单点；R8/R16 的 M 单点上界由 SiTe short-M input packing 实测确认。

对应 testcase 包括：

- `testcase/func_test/test_mxfp8_host/r8/test_mxfp8_r8.cpp:14-37 / case_params、MmaDteMxfp8R8Test::Computes`。
- `testcase/func_test/test_mxfp8_host/r16/test_mxfp8_r16.cpp:14-38 / case_params、MmaDteMxfp8R16Test::Computes`。
- `testcase/func_test/test_mxfp8_host/r32/test_mxfp8_r32.cpp:20-44 / case_params、run_param`。
- `testcase/func_test/test_mxfp8_host/r32/test_mxfp8_r32_shape_2x2.cpp:14-56 / case_params、invoke_2x2、MmaDteMxfp8R32Shape2x2Test::Computes`。
- `testcase/func_test/test_mxfp8_host/r32/test_mxfp8_r32_shape_4x1.cpp:13-31 / case_params、MmaDteMxfp8R32Shape4x1Test::Computes`。

### 12.6 MXFP6

MXFP6 的 `mxfloat6e2m3` 与 `mxfloat6e3m2` 使用相同的 tile geometry。当前 batch 配置明确说明 B 的 tile K 为 128，而不是 MXINT8 的 32：`testcase/func_test/test_mxfp6e2m3_host/batch_test.py:65-69 / SUPERTILE_LAYOUT_B`，A 的 layout 选择在 `:133-143 / resolve_layout_a`。

| row family | layoutA | layoutB | M | N | K |
|---|---|---|---:|---:|---:|
| R8 | `(512,8,4,1,1)` | `(128,32,4,1,1)` | `[8,8,None]` | `[32,inf,32]` | `[2048,inf,2048]` |
| R16 | `(256,16,4,1,1)` | `(128,32,4,1,1)` | `[16,16,None]` | `[32,inf,32]` | `[1024,inf,1024]` |
| R32 standard 4x1 | `(128,32,4,1,1)` | `(128,32,4,1,1)` | `[32,inf,32]` | `[32,inf,32]` | `[512,inf,512]` |
| R32 2x2 | `(128,32,2,2,1)` | `(128,32,2,2,1)` | `[64,inf,64]` | `[64,inf,64]` | `[256,256,None]` |
| R32 1x4 | `(128,32,1,4,1)` | `(128,32,1,4,1)` | `[128,inf,128]` | `[128,inf,128]` | `[128,128,None]` |

因此 MXFP6 的三个 R32 Pareto 形状的最小值是：

```text
4x1: (32, 32, 512)
2x2: (64, 64, 256)
1x4: (128,128, 128)
```

本次边界测试验证了 R32 4x1 的 K 可以从 512 增加到 1024/4096；R32 2x2 的 K=512 被拒绝；R32 1x4 的 K=256 虽可进入当前 dispatch，但输出与 SiTe gold 不一致，所以当前可用上界仍是 K=128。R8/R16 的 M 是 SiTe short-M 单点，N/K 按表中步长增加。

当前配置文件同时覆盖 linear/tiled 两种 C format、BF16/FP16/FP32 三种 outputT，并覆盖 2x2/1x4 的小 K：`testcase/func_test/test_mxfp6e2m3_host/test_config_mxfp6e2m3.md:103-143 / 配置说明`。所以对 MXFP6，linear/tiled 的逻辑结果相同，主入口的最小 M/N/K 也相同。

### 12.7 MXFP4

这里的 MXFP4 是 `sifmt::mxfloat4e2m1`，不是 MXINT4。当前显式实例化位于 `kernel/instantiations/mxfp4/inst_mxfp4.su:23-57 / mma_dte`，包含 R8、R16、R32 standard/2x2/1x4，并分别生成 tiled C 与 linear C 的 BF16 输出。

| row family | layoutA | layoutB | M | N | K |
|---|---|---|---:|---:|---:|
| R8 | `(256,8,4,1,1)` | `(64,32,4,1,1)` | `[8,8,None]` | `[32,inf,32]` | `[1024,inf,1024]` |
| R16 | `(128,16,4,1,1)` | `(64,32,4,1,1)` | `[16,16,None]` | `[32,inf,32]` | `[512,inf,512]` |
| R32 standard 4x1 | `(64,32,4,1,1)` | `(64,32,4,1,1)` | `[32,inf,32]` | `[32,inf,32]` | `[256,inf,256]` |
| R32 2x2 | `(64,32,2,2,1)` | `(64,32,2,2,1)` | `[64,inf,64]` | `[64,inf,64]` | `[128,128,None]` |
| R32 1x4 | `(64,32,1,4,1)` | `(64,32,1,4,1)` | `[128,inf,128]` | `[128,inf,128]` | `[64,64,None]` |

当前 MXFP4 testcase 的最小 shape 枚举在：

- `testcase/func_test/test_bf16_mxfp4_r8_host/test_bf16_mxfp4_r8_host.cpp:34-40 / M_VALUES、N_VALUES_TILED_OUT、N_VALUES_LINEAR_OUT、K_VALUES`。
- `testcase/func_test/test_bf16_mxfp4_r16_host/test_bf16_mxfp4_r16_host.cpp:34-38 / M_VALUES、N_VALUES、K_VALUES`。
- `testcase/func_test/test_bf16_mxfp4_r32_host/test_bf16_mxfp4_r32_host.cpp:34-48 / M_VALUES、N_VALUES、K_VALUES、2x2/1x4 参数`。

这些 testcase 仍把 R8 16-bit linear 的 N 从 64 开始列举；本次边界测试对 R8 使用 tiled C，避免把旧 linear-store validation 差异混入 shape contract。R8/R16 的 M 单点上界、R32 2x2 的 K 单点上界和 R32 1x4 的 K 单点上界均由新增边界测试实测。

### 12.8 MXINT4

MXINT4 是 `sifmt::mxint4`。当前显式实例化位于 `kernel/instantiations/mxint4/inst_mxint4.su:27-61 / mma_dte`。

| row family | layoutA | layoutB | M | N | K |
|---|---|---|---:|---:|---:|
| R8 | `(256,8,4,1,1)` | `(64,32,4,1,1)` | `[8,8,None]` | `[32,inf,32]` | `[1024,inf,1024]` |
| R16 | `(128,16,4,1,1)` | `(64,32,4,1,1)` | `[16,16,None]` | `[32,inf,32]` | `[512,inf,512]` |
| R32 standard 4x1 | `(64,32,4,1,1)` | `(64,32,4,1,1)` | `[32,inf,32]` | `[32,inf,32]` | `[256,inf,256]` |
| R32 2x2 | `(64,32,2,2,1)` | `(64,32,2,2,1)` | `[64,inf,64]` | `[64,inf,64]` | `[128,128,None]` |
| R32 1x4 | `(64,32,1,4,1)` | `(64,32,1,4,1)` | `[128,inf,128]` | `[128,inf,128]` | `[64,64,None]` |

MXINT4 与 MXFP4 的 tile geometry 和当前边界范围相同，所以 M/N/K 区间也相同；差别是输入数值编码和 TensorMap dtype，不是 shape contract。MXINT4 testcase 的枚举在：

- `testcase/func_test/test_bf16_mxi4_r8_host/test_bf16_mxi4_r8_host.cpp:42-48 / M_VALUES、N_VALUES_TILED_OUT、N_VALUES_LINEAR_OUT、K_VALUES`。
- `testcase/func_test/test_bf16_mxi4_r16_host/test_bf16_mxi4_r16_host.cpp:43-47 / M_VALUES、N_VALUES、K_VALUES`。
- `testcase/func_test/test_bf16_mxi4_r32_host/test_bf16_mxi4_r32_host.cpp:42-56 / M_VALUES、N_VALUES、K_VALUES、2x2/1x4 参数`。

### 12.9 五类 MX 输入的总表

下面只列 direct `mma_dte` 主入口的 layout/API 最小值；R32 的每一行是独立 input layout。

| MX input | row/layout | M | N | K |
|---|---|---:|---:|---:|
| MXINT8 | R8 | `[8,8,None]` | `[32,inf,32]` | `[512,inf,512]` |
| MXINT8 | R16 | `[16,16,None]` | `[32,inf,32]` | `[256,inf,256]` |
| MXINT8 | R32 4x1 | `[32,inf,32]` | `[32,inf,32]` | `[128,inf,128]` |
| MXINT8 | R32 2x2 | `[64,inf,64]` | `[64,inf,64]` | `[64,64,None]` |
| MXINT8 | R32 1x4 | `[128,inf,128]` | `[128,inf,128]` | `[32,32,None]` |
| MXFP8 | R8 | `[8,8,None]` | `[32,inf,32]` | `[512,inf,512]` |
| MXFP8 | R16 | `[16,16,None]` | `[32,inf,32]` | `[256,inf,256]` |
| MXFP8 | R32 4x1 | `[32,inf,32]` | `[32,inf,32]` | `[128,inf,128]` |
| MXFP8 | R32 2x2 | `[64,inf,64]` | `[64,inf,64]` | `[64,64,None]` |
| MXFP8 | R32 1x4 | `[128,inf,128]` | `[128,inf,128]` | `[32,32,None]` |
| MXFP6 | R8 | `[8,8,None]` | `[32,inf,32]` | `[2048,inf,2048]` |
| MXFP6 | R16 | `[16,16,None]` | `[32,inf,32]` | `[1024,inf,1024]` |
| MXFP6 | R32 4x1 | `[32,inf,32]` | `[32,inf,32]` | `[512,inf,512]` |
| MXFP6 | R32 2x2 | `[64,inf,64]` | `[64,inf,64]` | `[256,256,None]` |
| MXFP6 | R32 1x4 | `[128,inf,128]` | `[128,inf,128]` | `[128,128,None]` |
| MXFP4 | R8 | `[8,8,None]` | `[32,inf,32]` | `[1024,inf,1024]` |
| MXFP4 | R16 | `[16,16,None]` | `[32,inf,32]` | `[512,inf,512]` |
| MXFP4 | R32 4x1 | `[32,inf,32]` | `[32,inf,32]` | `[256,inf,256]` |
| MXFP4 | R32 2x2 | `[64,inf,64]` | `[64,inf,64]` | `[128,128,None]` |
| MXFP4 | R32 1x4 | `[128,inf,128]` | `[128,inf,128]` | `[64,64,None]` |
| MXINT4 | R8 | `[8,8,None]` | `[32,inf,32]` | `[1024,inf,1024]` |
| MXINT4 | R16 | `[16,16,None]` | `[32,inf,32]` | `[512,inf,512]` |
| MXINT4 | R32 4x1 | `[32,inf,32]` | `[32,inf,32]` | `[256,inf,256]` |
| MXINT4 | R32 2x2 | `[64,inf,64]` | `[64,inf,64]` | `[128,128,None]` |
| MXINT4 | R32 1x4 | `[128,inf,128]` | `[128,inf,128]` | `[64,64,None]` |

### 12.10 Linear output 和 tiled output 是否相同

要分成三个层次回答：

1. **主 `mma_dte` 入口的 M/N/K 范围是否因 C format 改变？**

   不改变。对上述 MX input，`layoutC.tensor_format=0` 和 `1` 不参与 A/B 的 M/N/K 对齐计算；因此本节表中的 M/N/K 范围对 linear/tiled C 相同。R8 主 dispatch 会显式计算 `remainder_n_tile`，见 `kernel/detail/mma_dte_tiled_tensor_dispatch_entrypoints.hpp:1211-1226 / mma_r8_dte`，并按 linear/tiled 选择不同 store case，见 `:299-323 / DEFINE_FUSED_IMPL_FIXED_CK_R8_DUAL`。本次新增边界测试为避免旧 R8 linear-store validation 差异，R8 范围测试使用 tiled C；这不改变 API shape contract 的结论。

2. **逻辑数值是否相同？**

   是。对相同 A/B、alpha、beta，linear C 和 tiled C 表示同一个逻辑矩阵 `C[M,N]`。区别是：

   ```text
   linear C: C[m * N + n]
   tiled C : 按 layoutC 对 tile/subtile 重新排列
   ```

   因此不能把 linear buffer 和 tiled buffer 直接按物理下标逐元素比较；必须先将 tiled buffer 按 layoutC 解码成 logical `[M,N]`，或者把 linear buffer pack 成同一 tiled layout。

3. **独立的 `mma_dte_validate_inputs()` 是否也认为两者相同？**

   对 R16/R32 是相同的；对 R8 的 16-bit linear output，当前 validation API 仍有一条额外限制：`src/simma_validation.cpp:222-233 / validate` 对非 MXFP8 input 要求 `N/32` 为偶数。也就是说：

   | R8、C 类型 | direct `mma_dte` 主路径 | `mma_dte_validate_inputs` |
   |---|---|---|
   | tiled BF16/FP16 | `N=32` 可进入 shape/dispatch 路径 | `N=32` 通过 |
   | linear BF16/FP16、MXINT8/MXFP6/MXFP4/MXINT4 | 主入口没有这条拒绝，源码 dispatch 有 linear remainder case | `N=32` 返回 `unsupported_output_shape`，`N` 至少 64 |
   | linear BF16/FP16、MXFP8 | `N=32` 可进入 | validation 也放行，因为 MXFP8 被排除在旧限制之外 |
   | linear FP32 | `N=32` 可进入 | 32-bit output 不触发该限制 |

   这说明当前仓库存在**主执行入口和独立 validation API 的行为不一致**。若上层先调用 `mma_dte_validate_inputs()`，再调用 `mma_dte()`，非 MXFP8 的 R8 16-bit linear `N=32` 会被辅助验证 API 拦住；若直接调用主模板，则入口本身没有该 shape reject。旧 MXFP4/MXINT4 testcase 仍在 `test_bf16_mxfp4_r8_host/test_bf16_mxfp4_r8_host.cpp:34-40 / M_VALUES、N_VALUES_LINEAR_OUT、K_VALUES` 和 `test_bf16_mxi4_r8_host/test_bf16_mxi4_r8_host.cpp:42-48 / M_VALUES、N_VALUES_LINEAR_OUT、K_VALUES` 中从 `N=64` 开始，这反映的是旧限制和测试覆盖策略，不是 A/B layout 的基本 N alignment。另一个独立事实是：R8/R16 的 M 单点上界来自 SiTe input packing；R32 2x2 和当前 R32 1x4 的 K 单点上界来自 layout/dispatch 实际能力，已在新增边界测试中验证。

### 12.11 最终工程结论

```text
MXINT8:
  R8  M=[8,8,None],     N=[32,inf,32],   K=[512,inf,512]
  R16 M=[16,16,None],   N=[32,inf,32],   K=[256,inf,256]
  R32 4x1 M=[32,inf,32],   N=[32,inf,32],   K=[128,inf,128]
  R32 2x2 M=[64,inf,64],   N=[64,inf,64],   K=[64,64,None]
  R32 1x4 M=[128,inf,128], N=[128,inf,128], K=[32,32,None]

MXFP8: 与 MXINT8 的范围相同；R32 1x4 的 K=32 是明确 planner 上界。

MXFP6:
  R8  M=[8,8,None],     N=[32,inf,32],   K=[2048,inf,2048]
  R16 M=[16,16,None],   N=[32,inf,32],   K=[1024,inf,1024]
  R32 4x1 M=[32,inf,32],   N=[32,inf,32],   K=[512,inf,512]
  R32 2x2 M=[64,inf,64],   N=[64,inf,64],   K=[256,256,None]
  R32 1x4 M=[128,inf,128], N=[128,inf,128], K=[128,128,None]

MXFP4:
  R8  M=[8,8,None],     N=[32,inf,32],   K=[1024,inf,1024]
  R16 M=[16,16,None],   N=[32,inf,32],   K=[512,inf,512]
  R32 4x1 M=[32,inf,32],   N=[32,inf,32],   K=[256,inf,256]
  R32 2x2 M=[64,inf,64],   N=[64,inf,64],   K=[128,128,None]
  R32 1x4 M=[128,inf,128], N=[128,inf,128], K=[64,64,None]

MXINT4: tile geometry and current range behavior match MXFP4.
```

因此，对“Linear output 和 tiled output 是否相同”的简短答案是：

```text
主 mma_dte：最小 M/N/K 相同，逻辑 C 数值相同，物理内存排列不同。
validation API：R8 + 非 MXFP8 + BF16/FP16 linear output 仍额外要求 N >= 64，
                 这是当前辅助验证逻辑与主 dispatch 不一致，不能解释成硬件 A/B shape 对齐要求。
```

### 12.12 已有 SIPU150 构建的最小 shape smoke check

以下检查使用当前仓库已有的 `build/testcase/2609161442/sipu_150/default/specialized` 可执行文件，目标架构是 SIPU150；运行时 SDK 是日志中携带的 `2609161442`，不是题目指定的 `2609151958`。

已通过的最小组合：

| testcase | 形状 | 输出 |
|---|---|---|
| `test_bf16_mxi4_r32_host` | standard `(M,N,K)=(32,32,256)` | tiled、linear |
| `test_bf16_mxi4_r32_host` | 2x2 `(64,64,128)` | tiled、linear |
| `test_bf16_mxi4_r32_host` | 1x4 `(128,128,64)` | tiled、linear |
| `test_mxfp8_r8` | `(8,32,512)` | tiled FP16 |
| `test_mxfp8_r32_shape_2x2` | `(64,64,64)` | linear FP16 |
| `test_mxfp8_r32_shape_4x1` | `(128,128,32)` | linear FP16 |

例如 MXINT4 R32 的过滤运行命令为：

```bash
./build/testcase/2609161442/sipu_150/default/specialized/test_bf16_mxi4_r32_host \
  '--gtest_filter=*Standard*Out_M32_N32_K256:*SubTile2x2*Out_M64_N64_K128:*SubTile4x1*Out_M128_N128_K64'
```

该次运行结果为 6 个测试全部通过，覆盖三种 R32 input supertile 和两种 C output format。MXFP8 三个最小 shape 也分别通过对应的 R8、R32 2x2、R32 1x4 testcase。

### 12.13 M/N/K 范围边界的实际 GTest 验证（2026-09-19）

新增测试工程：

```text
testcase/func_test/test_mx_shape_bounds_host/CMakeLists.txt
testcase/func_test/test_mx_shape_bounds_host/test_mx_shape_bounds_host.cpp
```

测试源直接链接当前仓库的 MXINT8、MXFP8、MXFP6、MXFP4 `.su` 实例，覆盖 R8、R16、R32 4x1、R32 2x2、R32 1x4。核心执行函数为 `test_mx_shape_bounds_host/test_mx_shape_bounds_host.cpp:66-166 / run_shape`，每个 shape 都经过 SiTe input pack、`sipu::tensor::mm_mnk` gold、H2D、`mma_dte`、D2H 和逐元素比较。

按题目要求执行了：

```bash
. /share/users/like/miniconda3/bin/activate vllm_dev
cd /share/users/like/package/vllm-sipu
source sipu_sdk_setup.sh
cd /share/users/like/package/oplib/mma_dte_tile_tensor
BUILD_JOBS=64 BUILD_CXX=/usr/bin/c++ bash build-modify.sh --test
```

最终增量构建日志为：

```text
temp/build.test.shape_bounds.final.log
```

聚合 testcase 构建成功，并发布了 82 个 testcase binary。边界 executable 为：

```text
build/testcase/v0.4.2_2609111541/sipu_150/default/specialized/test_mx_shape_bounds_host
```

运行日志为：

```text
temp/run_mx_shape_bounds3.log
```

结果：

```text
[==========] 36 tests from 20 test suites ran.
[  PASSED  ] 36 tests.
```

验证覆盖：

| 维度/路径 | 实际验证 |
|---|---|
| R8 | M 超过单 input tile 时 SiTe packer 拒绝；N 增加一个及多个 tile；K 增加多个 tile |
| R16 | M 超过单 input tile 时 SiTe packer 拒绝；N/K 多 tile 通过 |
| R32 4x1 | M、N、K 各增加一步及多个 tile 均通过 |
| R32 2x2 | M/N 增加通过；K 增加到下一个值由 death test 拒绝 |
| R32 1x4 | M/N 增加通过；K 增加后 MXINT8/MXFP6/MXFP4 输出错误，MXFP8 planner 拒绝 |

因此表中的 `inf` 不是声称已经穷举到无限大，而是表示当前代码没有有限的 shape 上界；本次测试至少覆盖了多个 M/N tile 和多个 K tile。R32 1x4 的 K 被写成单点，是因为当前实现对多 K tile 不可用，不能仅依据 `M/N/K` 模运算检查得出“可继续增长”。

## 13. release DSO 的区别、应用链接方式和无 GTest 调用示例（2026-09-18）

本节以独立 debug 目录
`/share/users/like/package/oplib/mma_dte_tile_tensor`
为准。release 包目录为：

```text
/share/users/like/package/oplib/mma_dte_tile_tensor/release/mma_dte_tiled_tensor-1.0.0
```

### 13.1 先给结论：应用应该链接哪个 `.so`

普通应用应该链接：

```text
libtile_mma_dte.so
libsipurt.so
libsipu.so
```

也就是：

```cmake
target_link_libraries(app PRIVATE
    /path/to/mma_dte_tiled_tensor-1.0.0/lib/libtile_mma_dte.so
    /path/to/sipu-sdk/lib/libsipurt.so
    /path/to/sipu-sdk/lib/libsipu.so
    ${CMAKE_DL_LIBS}
)
```

`libtile_mma_dte_bf16_r16.so`、`libtile_mma_dte_bf16_r32_bf16out.so` 等是 device kernel 的 component DSO，不是普通应用的首选公共链接入口。它们必须和 `libtile_mma_dte.so` 放在同一个 `lib` 目录，因为 umbrella DSO 在第一次调用具体模板实例时，会按完整 C++ symbol 名称动态打开对应的 component DSO。

release 包自己的 pkg-config 文件也只导出 umbrella 库，见
`release/mma_dte_tiled_tensor-1.0.0/lib/pkgconfig/mma_dte_tiled_tensor.pc:1-13`：

```text
Libs: -L${libdir} -ltile_mma_dte
Cflags: -I${includedir}
```

### 13.2 `libtile_mma_dte.so` 和 component DSO 的关系

根 CMake 的构建关系如下。

1. `CMakeLists.txt:192-230 / foreach(INST_FILE IN LISTS INST_FILES)` 为每个
   `kernel/instantiations/*/inst_*.su` 创建一个 SCC 编译动作。每个 `.su` 被
   `scc` 编译成一个包含 SIPU device fatbin 的 x86-64 object，而不是把所有实例
   直接编译进应用的 host object。

2. `CMakeLists.txt:318-341 / foreach(TYPE IN LISTS OBJECT_LIB_TYPES)` 把每个
   instantiation object 单独链接成一个 component DSO，并链接 `sipurt`、`sipu`。
   这就是 `libtile_mma_dte_bf16_r16.so` 等文件的来源。

3. `CMakeLists.txt:343-390 / add_custom_command` 生成 umbrella 的 C++/汇编
   trampoline，再构建 `mma_dte_umbrella`，输出名设置为
   `libtile_mma_dte.so`。这里的 umbrella 只承载 host 侧的符号转发、校验和
   workspace 辅助代码；device implementation 仍在 component DSO 中。

4. `build_release.sh:108-123 / package_release` 把 umbrella 复制到 release 包，
   并把所有 `build/libtile_mma_dte_*.so` component DSO 复制到同一个 `lib/` 目录。
   `build_release.sh:125-137 / package_release` 生成的 pkg-config 只写
   `-ltile_mma_dte`，这进一步说明公共入口是 umbrella。

umbrella 的动态分发实现位于：

- `tools/generate_mma_dte_umbrella.py:20-44 / collect_symbols`：扫描每个
  component DSO 的动态符号，把每个 `_Z7mma_dte...` symbol 映射到组件文件名。
- `tools/generate_mma_dte_umbrella.py:47-101 / write_cpp`：生成
  `__mma_dte_resolve_component_symbol()`，用 `dladdr()` 找到 umbrella 的目录，
  再用 `dlopen(base_dir + component_name)` 和 `dlsym()` 找到真正实现。
- `tools/generate_mma_dte_umbrella.py:104-181 / write_asm`：为每个模板符号
  生成 trampoline。第一次调用保存参数、解析并缓存函数地址；后续调用直接跳转
  到缓存地址。

因此它不是下面这种结构：

```text
应用 -> 一个包含所有 device code 的单体 libtile_mma_dte.so
```

而是：

```text
应用
  |
  +-- link -> libtile_mma_dte.so       host umbrella/trampoline
                    |
                    +-- first call: dlopen 同目录 component DSO
                    |       |
                    |       +-- libtile_mma_dte_bf16_r8.so
                    |       +-- libtile_mma_dte_mxint8_r32_2x2.so
                    |       +-- ...
                    |
                    +-- component DSO -> SIPU device fatbin + sipurt + sipu
```

当前 release 中 `libtile_mma_dte.so.1.0.0` 大约 0.85 MB，而各 component
DSO 通常是数 MB 到数十 MB；这也与“umbrella 是转发层、component 携带具体
device code”的结构相符。`readelf -d` 还显示 umbrella 本身没有把
`libsipurt.so.0`、`libsipu.so.0` 作为静态的 `DT_NEEDED` component 依赖，
因为 component 是通过 `dlopen()` 延迟加载的。

### 13.3 题目中几个 component DSO 的精确区别

文件名中的 `bf16`/`fp16` 是输入数据类型，`r8`/`r16`/`r32` 是 kernel
row family；`r32_bf16out` 和 `r32_fp32out` 才明确区分 R32 的输出类型。
每个 component 内仍可能包含多个 A/B/C layout 和 linear/tiled format 的显式
模板实例。

| DSO | 输入类型 | row family | 输出实例 | 主要来源 |
|---|---|---:|---|---|
| `libtile_mma_dte_bf16_r16.so` | BF16 | R16 | BF16 output、FP32 output | `kernel/instantiations/bf16/inst_bf16_r16.su:8-26 / mma_dte` |
| `libtile_mma_dte_bf16_r32_bf16out.so` | BF16 | R32 | BF16 output | `kernel/instantiations/bf16/inst_bf16_r32_bf16out.su:8-16 / mma_dte` |
| `libtile_mma_dte_bf16_r32_fp32out.so` | BF16 | R32 | FP32 output | `kernel/instantiations/bf16/inst_bf16_r32_fp32out.su:8-16 / mma_dte` |
| `libtile_mma_dte_bf16_r8.so` | BF16 | R8 | BF16 output、FP32 output | `kernel/instantiations/bf16/inst_bf16_r8.su:8-26 / mma_dte` |
| `libtile_mma_dte_fp16_r16.so` | FP16 | R16 | FP16 output、FP32 output | `kernel/instantiations/fp16/inst_fp16_r16.su:21-39 / mma_dte` |

#### `libtile_mma_dte_bf16_r16.so`

`kernel/instantiations/bf16/inst_bf16_r16.su:8-16 / mma_dte` 是 BF16 input
到 BF16 output 的 R16 实例；`kernel/instantiations/bf16/inst_bf16_r16.su:18-26 / mma_dte`
是 BF16 input 到 FP32 output 的 R16 实例。也就是说，文件名没有写
`bf16out`/`fp32out`，不是因为只支持一种输出，而是这两个 R16 输出族被放在同一个
component DSO 中。

其中典型布局是：

```text
R16 BF16 output: A=(32,16), B=(16,32), C=(32,16)
R16 FP32 output: A=(32,16), B=(16,32), C=(16,16)
```

每个布局还分别有 `tensor_format=0` 的 linear 和 `tensor_format=1` 的 tiled
组合；完整组合以 `.su` 中的显式 template line 为准。

#### `libtile_mma_dte_bf16_r32_bf16out.so`

`kernel/instantiations/bf16/inst_bf16_r32_bf16out.su:8-16 / mma_dte` 只显式
实例化 BF16 input、BF16 output、FP32 scalar 的 R32 kernel。典型布局是：

```text
A=(16,32), B=(16,32), C=(16,32)
```

其中 `C` 可以是 tiled 或 linear，A/B 也包含相应 format 组合。

#### `libtile_mma_dte_bf16_r32_fp32out.so`

`kernel/instantiations/bf16/inst_bf16_r32_fp32out.su:8-16 / mma_dte` 只显式
实例化 BF16 input、FP32 output、FP32 scalar 的 R32 kernel。典型布局是：

```text
A=(16,32), B=(16,32), C=(8,32)
```

FP32 element 是 32 bit，因此同样的 R32 行组织下，C 的 tile_dim0 与 BF16 C
不同，这是输出类型导致的物理 tile 容量差异，不是另一个 GEMM 算法。

#### `libtile_mma_dte_bf16_r8.so`

`kernel/instantiations/bf16/inst_bf16_r8.su:8-16 / mma_dte` 是 BF16 input
到 BF16 output 的 R8 实例；`kernel/instantiations/bf16/inst_bf16_r8.su:18-26 / mma_dte`
是 BF16 input 到 FP32 output 的 R8 实例。

典型布局为：

```text
BF16 output: A=(64,8), B=(16,32), C=(64,8)
FP32 output: A=(64,8), B=(16,32), C=(32,8)
```

这也是为什么 R8 的 output 不能只按 `M*N` 物理元素数分配，必须按选定的
`layoutC` 和 output element size 计算 buffer。

#### `libtile_mma_dte_fp16_r16.so`

`kernel/instantiations/fp16/inst_fp16_r16.su:21-29 / mma_dte` 是 FP16 input
到 FP16 output 的 R16 实例；`kernel/instantiations/fp16/inst_fp16_r16.su:31-39 / mma_dte`
是 FP16 input 到 FP32 output 的 R16 实例。典型布局分别是：

```text
FP16 output: A=(32,16), B=(16,32), C=(32,16)
FP32 output: A=(32,16), B=(16,32), C=(16,16)
```

它与 `libtile_mma_dte_bf16_r16.so` 的结构相同，差异是 inputT 从
`sifmt::bfloat16` 换成 `sifmt::float16`。

### 13.4 为什么不能只链接某个 component DSO

如果应用显式调用的模板实例正好只存在于某一个 component DSO，理论上可以直接
链接该 DSO。但这不是 release API 的稳定使用方式，原因有三点：

1. 模板符号包含 `inputT`、`outputT`、`scalarT`、`layoutA`、`layoutB`、`layoutC`
   的完整编码。只要 C layout 或 linear/tiled format 改变，就是另一个符号。
2. 一个看似相同的 `R8`/`R16` component 可能包含多个 output type 和多个 format
   组合，但不会保证未来所有新增实例仍放在同一个文件里。
3. release 的 `build_release.sh:113-123 / package_release` 明确把 component
   作为 umbrella 的 sibling private implementation 打包；应用只通过 umbrella
   获得统一的符号入口。

因此推荐：

```text
应用链接 libtile_mma_dte.so
release/lib 中保留所有 libtile_mma_dte_*.so
应用同时链接 SDK 的 libsipurt.so 和 libsipu.so
```

### 13.5 无 GTest 应用的目录和源码

已经创建：

```text
/share/users/like/shareVR/mma_dte_call/main.cpp
/share/users/like/shareVR/mma_dte_call/CMakeLists.txt
```

`/share/users/like/shareVR/mma_dte_call/main.cpp` 的关键部分：

- `main.cpp:15-19` 定义与原 testcase 相同的 `kLayoutB`、R8/R32 A/C layout。
- `main.cpp:21-129 / run_case` 保留原测试的完整流程：构造随机 FP32 输入、用
  `sipu::make_shape`、`sipu::make_layout`、`sipu::make_tensor` 转成 tiled MXINT8，
  用 `sipu::tensor::mm_mnk` 生成 gold，再分配 SIPU device buffer，调用
  `mma_dte`，D2H 拷贝并逐元素比较。
- `main.cpp:131-134 / run_bf16_default_case` 调用原 BF16/R8：
  `M=8,N=576,K=1024`，C 为 linear BF16。
- `main.cpp:136-139 / run_fp16_default_case` 调用原 FP16/R32：
  `M=32,N=576,K=1024`，C 为 tiled FP16。
- `main.cpp:143-155 / main` 直接顺序调用两个 `run_case`，失败返回
  `EXIT_FAILURE`，不依赖 GTest。

原 GTest testcase 的对应位置是：

- `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:39-124 / run_case`
- `testcase/func_test/test_bf16_mxint8_host/test_bf16_mxi8_host.cpp:126-134 /
  run_bf16_default_case、run_default_case`
- 原来的 GTest wrapper 在 `:136-146`，新应用删除了这些 class/`TEST_F`，改成
  `main.cpp:143-155 / main`。

### 13.6 应用 CMake 的作用

`/share/users/like/shareVR/mma_dte_call/CMakeLists.txt`：

- `CMakeLists.txt:9-18` 设置 release 包和 SIPU SDK 根目录；默认 SDK 是
  `/share_data/sicx_sdk/release/v0.4.2`。
- `CMakeLists.txt:20-34 / find_library` 查找 `tile_mma_dte`、`sipurt`、`sipu`。
- `CMakeLists.txt:38-43 / target_include_directories` 加入 release public header
  和 SDK 的 `include`、`include/SiTe`、`include/sipurt`。
- `CMakeLists.txt:45-50 / target_link_libraries` 链接 umbrella、SIPU runtime、
  SIPU host library 和 `dl`。
- `CMakeLists.txt:52-55 / set_target_properties` 把 release `lib` 和 SDK `lib`
  写入 build/install RPATH，保证 executable 能找到 umbrella、component 以及
  `libsipurt.so.0`、`libsipu.so.0`。

和原 testcase CMake 的差异是有意的：

- 原 testcase `testcase/func_test/test_bf16_mxint8_host/CMakeLists.txt:43-61`
  先用 SCC 编译测试专用 `inst_bf16_mxi8.su` object，再在
  `:52-61 / add_custom_command、tile_mma_dte_mxint8_so` 中生成测试专用 DSO。
- 原 testcase `:74-79 / target_link_libraries、add_dependencies` 链接的是这个
  测试专用 DSO，因此适合源码测试，不适合一个已经使用 release package 的普通
  应用。
- 新应用不重新编译 device `.su`，直接消费 release 包的 public header、umbrella
  和 sibling component DSO。

### 13.7 实际编译命令

使用的命令是：

```bash
source /share_data/users/like/miniconda3/bin/activate vllm_dev
cd /share/users/like/package/vllm-sipu
source sipu_sdk_setup.sh

cmake \
  -S /share/users/like/shareVR/mma_dte_call \
  -B /share/users/like/shareVR/mma_dte_call/build \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DSIPU_SDK_ROOT=/share_data/sicx_sdk/release/v0.4.2

cmake --build /share/users/like/shareVR/mma_dte_call/build --parallel 8
```

这里必须显式传 `-DCMAKE_CXX_COMPILER=/usr/bin/c++`。本次配置输出确认：

```text
MMA_DTE_LIBRARY=/share/users/like/package/oplib/mma_dte_tile_tensor/release/mma_dte_tiled_tensor-1.0.0/lib/libtile_mma_dte.so
SIPURT_LIBRARY=/share_data/sicx_sdk/release/v0.4.2/lib/libsipurt.so
SIPU_LIBRARY=/share_data/sicx_sdk/release/v0.4.2/lib/libsipu.so
```

注意：`vllm-sipu/sipu_sdk_setup.sh` 在其内部又把 SDK 解析为
`/share_data/sicx_sdk/release/v0.4.2_2609111541` 并打印了 override warning；
本次 CMake 通过 `-DSIPU_SDK_ROOT=/share_data/sicx_sdk/release/v0.4.2` 固定了
应用实际使用的 include/library 路径。运行时 `ldd` 也确认 executable 解析到：

```text
libtile_mma_dte.so.1 -> .../release/mma_dte_tiled_tensor-1.0.0/lib/libtile_mma_dte.so.1
libsipurt.so.0      -> /share_data/sicx_sdk/release/v0.4.2/lib/libsipurt.so.0
libsipu.so.0        -> /share_data/sicx_sdk/release/v0.4.2/lib/libsipu.so.0
```

### 13.8 实际运行和正确性结果

运行命令：

```bash
source /share_data/users/like/miniconda3/bin/activate vllm_dev
cd /share/users/like/package/vllm-sipu
source sipu_sdk_setup.sh
/share/users/like/shareVR/mma_dte_call/build/mma_dte_call
```

输出摘要：

```text
case M=8, N=576, K=1024 passed
[MMA_DTE_DISPATCH] R32 schedule=ONE_PRO_ONE_CON_V2 ...
case M=32, N=576, K=1024 passed
all mma_dte cases passed
```

这次运行验证了两件事：

1. 应用只链接 `libtile_mma_dte.so`，仍能通过 umbrella 的 `dlopen/dlsym`
   找到需要的 component DSO，并成功执行 SIPU kernel。
2. 原 testcase 的两个 `run_case` 在不使用 GTest 的普通 `main()` 中均通过：
   BF16/R8 linear output 和 FP16/R32 tiled output 的 device 结果都与 SiTe gold
   逐元素一致。

## 14. MXFP8 R32 测试中的 `TEST`、`run_case` 和 `INSTANTIATE_TEST_SUITE_P`（2026-09-18）

本节以 `/share/users/like/package/oplib/mma_dte_tile_tensor` 为准。

### 14.1 `SingleFullChunkReinitializesAccumulator` 是否调用 `run_case`

是。`testcase/func_test/test_mxfp8_host/r32/test_mxfp8_r32.cpp:63-74 / MmaDteMxfp8R32Regression::SingleFullChunkReinitializesAccumulator` 中调用了两次：

```text
run_case<mxfloat8e4m3, float16, R32, linear C>(
    "r32_e4m3_f16_linear_single_full_chunk_first", 96, 96, 256)

run_case<mxfloat8e4m3, float16, R32, linear C>(
    "r32_e4m3_f16_linear_single_full_chunk_repeat", 96, 96, 256)
```

之后 `:72-73 / SingleFullChunkReinitializesAccumulator` 用 `EXPECT_TRUE` 检查两次返回值。

`temp/run_mxfp8.log:5-12` 明确显示该测试实际运行，并且两次调用都通过：

```text
[ RUN      ] MmaDteMxfp8R32Regression.SingleFullChunkReinitializesAccumulator
[PASS] r32_e4m3_f16_linear_single_full_chunk_first (M=96, N=96, K=256)
[PASS] r32_e4m3_f16_linear_single_full_chunk_repeat (M=96, N=96, K=256)
[       OK ] MmaDteMxfp8R32Regression.SingleFullChunkReinitializesAccumulator
```

因此它不是“只定义未调用”。这个回归用例让相同的 `M=96,N=96,K=256` 连续执行两次，用于检查单个完整 K chunk 的第一次 launch 会重新初始化 accumulator，而不会错误地复用上一次 launch 的累加状态。

### 14.2 `run_case` 的定义和功能

定义位于 `testcase/func_test/test_mxfp8_host/common/test_mxfp8_common.hpp:46-48 / run_case`：

```cpp
template <class MX_T, class OUT_T,
          mma_dte_api::tensor_layout LayoutA,
          mma_dte_api::tensor_layout LayoutB,
          mma_dte_api::tensor_layout LayoutC>
bool run_case(const char* name, int M, int N, int K);
```

模板参数含义：`MX_T` 是 A/B 的 MX 输入类型，`OUT_T` 是 C 和 alpha/beta 类型，三个 `Layout*` 是 A/B/C 的 `mma_dte_api::tensor_layout`。`LayoutC.tensor_format=0` 表示 linear C，`1` 表示 tiled C。

函数流程：

1. `testcase/func_test/test_mxfp8_host/common/test_mxfp8_common.hpp:49-59 / run_case` 使用固定 seed `7` 生成 A `[M,K]` 和 B `[N,K]` 的随机 FP32 输入。
2. `:61-74 / run_case` 用 `sipu::make_shape`、`sipu::make_tiling_config`、`sipu::make_layout` 和 `sipu::make_tensor` 把 A/B 转成 MX tiled tensor；`:62` 特别说明 SiTe 的 `(row,column)` 与 `tensor_layout` 的 `(column,row)` 需要交换。
3. `:76-82 / run_case` 用 `sipu::tensor::mm_mnk` 生成 gold，再导出 `gold_linear` 和 `gold_tiled`，分别对应 logical/linear 与 physical/tiled 校验。
4. `:84-95 / run_case` 按 `LayoutC` 计算输出物理元素数，并分配对齐的 device A/B/C buffer。
5. `:96-103 / run_case` H2D 拷贝 A/B，然后调用：

   ```cpp
   mma_dte<MX_T, OUT_T, OUT_T, LayoutA, LayoutB, LayoutC>(
       OP_N, OP_N, M, N, K,
       OUT_T(0.1f), d_a, K, d_b, K,
       OUT_T(0.2f), d_c, N);
   ```

6. `:105-108 / run_case` 将 C D2H 并释放 device buffer。
7. `:110-136 / run_case` 逐元素比较结果；普通有限值使用 `kErrThreshold=0.0076f`，阈值定义在 `:21 / kErrThreshold`，同时处理 NaN/Inf。
8. `:138-143 / run_case` 释放 host 输出；成功打印 `[PASS]` 并返回 `true`，失败返回 `false`。

因此 `run_case` 是一个完整的端到端单 case 测试函数：

```text
随机输入 -> SiTe MX tensor -> SiTe gold GEMM
-> SIPU device allocation/H2D -> mma_dte -> D2H
-> linear/tiled gold comparison -> bool
```

### 14.3 参数化测试的调用链

`testcase/func_test/test_mxfp8_host/r32/test_mxfp8_r32.cpp:49 / MmaDteMxfp8R32Test` 继承 `::testing::TestWithParam<CaseParam>`；`:51-58 / MmaDteMxfp8R32Test::Computes` 是 `TEST_P` body：

```text
Computes -> GetParam() -> daily_only 判断 -> run_param(param) -> run_case(...)
```

`run_param` 位于 `:35-47 / run_param`，根据 `CaseKind` 选择 E5M2 tiled 或 E4M3 linear 的 `run_case` 模板实例。

如果 `param.daily_only` 且 `is_daily_suite()` 为 false，`:52-55 / MmaDteMxfp8R32Test::Computes` 执行 `GTEST_SKIP()`，此时不会调用 `run_param`，也不会调用 `run_case`。`is_daily_suite` 的环境变量判断位于 `testcase/func_test/test_mxfp8_host/common/test_mxfp8_common.hpp:41-44 / is_daily_suite`。

### 14.4 `INSTANTIATE_TEST_SUITE_P` 的功能

调用位于 `testcase/func_test/test_mxfp8_host/r32/test_mxfp8_r32.cpp:76-78 / INSTANTIATE_TEST_SUITE_P`：

```cpp
INSTANTIATE_TEST_SUITE_P(DefaultCases, MmaDteMxfp8R32Test,
                         ::testing::ValuesIn(case_params()), case_param_name);
```

它不会直接调用 `run_case`，而是把 `TEST_P` 的一个测试 body 按参数集合展开成多个独立 GTest test instance：

- `DefaultCases` 是测试 prefix。
- `MmaDteMxfp8R32Test` 是待实例化的 parameterized fixture。
- `::testing::ValuesIn(case_params())` 使用 `test_mxfp8_r32.cpp:20-29 / case_params` 返回的 5 个 `CaseParam`。
- `case_param_name` 位于 `:31-33 / case_param_name`，把 `CaseParam::name` 作为测试名后缀。

生成的测试名类似：

```text
DefaultCases/MmaDteMxfp8R32Test.Computes/r32_e5m2_f16_tiled_m96n96k256
DefaultCases/MmaDteMxfp8R32Test.Computes/r32_e4m3_f16_linear_m96n96k256
```

概念上相当于为每个 `CaseParam` 注册一个带参数的 `Computes`，运行时由 GTest 调用该实例的 `Computes`；真正调用 `run_case` 的位置仍是 `run_param`，不是 `INSTANTIATE_TEST_SUITE_P` 本身。

### 14.5 本次日志的实际调用次数

`temp/run_mxfp8.log` 显示总共运行 6 个测试：1 个 regression test 加 5 个参数化 test。当前非 daily 环境下：

| 测试 | 状态 | `run_case` 调用次数 |
|---|---|---:|
| `MmaDteMxfp8R32Regression.SingleFullChunkReinitializesAccumulator` | PASS | 2 |
| `.../r32_e5m2_f16_tiled_m96n96k256` | PASS | 1 |
| `.../r32_e4m3_f16_linear_m96n96k256` | PASS | 1 |
| 3 个 `daily_only` 参数 | SKIPPED | 0 |

所以这次运行中：

```text
run_case 实际调用次数 = 2 + 1 + 1 = 4
```

日志末尾 `temp/run_mxfp8.log:40-44` 也确认了 `3 tests passed`、`3 tests skipped`。若运行前设置 `SITEST_CASE_LEVEL=daily`，3 个 daily-only 参数才会继续进入 `run_param` 和 `run_case`。

## 15. simo 通过 Python 调用 MMA DTE 的 AOT 方案设计（2026-09-20）

### 15.1 结论先行

本次目标可以实现为：

```text
Python
  -> torch.ops.simo.mma_dte
  -> simo::_C 中的 TORCH_LIBRARY_IMPL(simo, PrivateUse1, ...)
  -> simo 的 C++ host wrapper
  -> mma_dte<MX_T, OUT_T, OUT_T, LayoutA, LayoutB, LayoutCLinear>(...)
  -> libtile_mma_dte.so umbrella
  -> 同目录的 libtile_mma_dte_*.so component DSO
  -> SIPU runtime / SIPU device kernel
  -> linear layout 的 BF16/FP16/FP32 output Tensor
```

推荐方案如下：

1. 保留 `simo` 当前的 `setuptools + torch.utils.cpp_extension`，继续构建现有的 CPU/CUDA/SIPU host C++ extension。
2. 新增 `third_party/mma_dte_tile_tensor` git submodule。
3. 新增一个很小的 MMA DTE AOT CMake 子工程，或者在现有 `setup.py` 的 `build_ext` 中调用 mma 子模块自己的 CMake。CMake 负责生成 MMA DTE 的 umbrella 和 component DSO；它不是因为 `torch.ops` 强制要求，而是因为 `.su` device code 的 AOT 构建需要 SIPU 编译器、device object、device fatbin 和 component DSO 管理。
4. 新增一个普通 C++ host wrapper，并将其注册为 `torch.ops.simo.mma_dte`。这个 wrapper 只调用公开的 `include/simma.h` API，不复制 `mma_dte_tile_tensor` 的 device kernel 实现。
5. 不引入 `tvm-ffi`；不新增 pybind11 绑定。PyTorch dispatcher 的 `TORCH_LIBRARY_FRAGMENT`/`TORCH_LIBRARY_IMPL` 已足够。
6. 第一个版本必须显式接收 `M/N/K`、MX dtype 和 MMA layout。`siinfer.hp_to_mx` 返回的是一维物理 `uint8` buffer，buffer 本身不携带这些元数据，不能只传两个 Tensor 让 C++ 自动猜测。

严格来说，`simo` **不因为 Python custom op 而必须新增 CMakeLists.txt**：如果已经有预编译的 `libtile_mma_dte.so`，普通 `CppExtension` 加 `-L/-l` 也能完成 host wrapper 的链接。但是本题要求把 `mma_dte_tile_tensor` 以 submodule 源码方式纳入，并且只考虑 AOT，因此建议增加最小的 CMake 构建入口。否则 `setup.py` 必须自己复制 `scc` 的 host/device 双 pass、device ELF link、component DSO staging 等逻辑，维护成本更高。

### 15.2 simo 当前代码适合如何接入

#### 15.2.1 现有 native extension 入口

`setup.py:36-42 / get_build_backend` 根据 `torch.sipu.is_available()`、`torch.cuda.is_available()` 和 CPU fallback 选择 `sipu`、`cuda` 或 `cpu`。`setup.py:45-100 / get_extensions` 当前用 `CppExtension`/`CUDAExtension` 收集 `simo/csrc/**/*.cpp` 和 CUDA 源文件；`setup.py:103-119 / SimoBuildExtension::run` 在普通 extension 构建后才处理 CUDA ONNX runtime。

因此 MMA DTE 不能作为无条件的普通 `.cpp` 加入所有 backend：

- CPU build 不应 include `simma.h`，也不应链接 `libtile_mma_dte.so`。
- CUDA build 不应尝试 SIPU `.su` 或 SIPU DSO。
- SIPU build 才执行 MMA DTE 的 CMake/AOT 子构建，并把 host wrapper 加入 `_C`。

建议的源文件边界：

```text
third_party/mma_dte_tile_tensor/       # git submodule
cmake/mma_dte.cmake                    # simo 的最小构建 glue
simo/csrc/sipu/mma_dte.cpp             # 仅 SIPU host wrapper
simo/csrc/sipu/mma_dte_layout.h        # dtype/layout 到模板的映射
simo/ops/mma_dte.py                    # Python 薄封装
simo/csrc/torch_bindings.cpp           # schema 注册
```

当前 `simo/csrc/torch_bindings.cpp:18-27 / TORCH_LIBRARY_FRAGMENT` 已经为 `simo` namespace 定义 schema；`simo/csrc/quantization/cpu/quantize_fp.cpp:43-45 / TORCH_LIBRARY_IMPL(simo, CPU, m)` 已经展示了按 dispatch key 注册实现的模式。因此不需要另建 Python C API。

#### 15.2.2 推荐的 build 顺序

建议把 `SimoBuildExtension` 扩展为以下顺序：

```text
SimoBuildExtension::run/build_extensions
  1. 如果 backend != sipu，跳过 MMA DTE
  2. 检查 third_party/mma_dte_tile_tensor/include/simma.h
  3. 调用 mma 子工程 CMake configure/build
  4. 得到：
       build/mma_dte/lib/libtile_mma_dte.so
       build/mma_dte/lib/libtile_mma_dte_*.so
  5. 编译 simo._C host wrapper，并 link -ltile_mma_dte
  6. 将 umbrella 与所有所需 component DSO 复制到：
       editable: source tree 下的 simo/mma_dte/
       wheel:    build_lib/simo/mma_dte/
```

`libtile_mma_dte.so` 和所有 `libtile_mma_dte_*.so` 必须放在同一个运行时目录。umbrella DSO 的实现会根据自身位置寻找 component DSO；只复制主库、不复制 component DSO 会导致第一次调用某些模板实例时 `dlopen` 失败。

推荐给 `_C` 设置相对 RPATH：

```text
$ORIGIN/mma_dte
```

不要把构建机的绝对路径写入 wheel。SIPU SDK 的 `libsipu.so`、`libsipurt.so` 等仍属于运行环境依赖，不应把 SDK 全部复制进 simo wheel；安装/运行环境应通过 SIPU SDK setup 或 torch_sipu 的运行环境提供它们。

### 15.3 `tvm-ffi` 和 pybind11 是否需要

#### 15.3.1 tvm-ffi：不需要

`torch.ops.simo.mma_dte` 的调用协议由 PyTorch dispatcher 管理。当前 `simo/__init__.py:7-13 / _get_native_ops` 只需要 import `simo._C`，然后返回 `torch.ops.simo`；这正是本次新 op 可以复用的机制。

引入 tvm-ffi 会增加：

- 一个额外的运行时/编译依赖；
- 一套与 PyTorch dispatcher 并行的 schema/registration 机制；
- wheel、editable 和 ABI 的额外兼容面。

本题没有需要 tvm-ffi 才能解决的问题，所以不建议引入。

#### 15.3.2 pybind11：不需要新增

`simo/csrc/torch_bindings.cpp:1-14 / PYBIND11_MODULE` 当前为了暴露 `RoundingMode` enum 已经使用 pybind11；但 `mma_dte` 本身不需要写新的 `py::def`。

推荐由 dispatcher 完成：

```cpp
TORCH_LIBRARY_FRAGMENT(simo, m) {
  m.def("mma_dte(Tensor a, Tensor b, int m, int n, int k, "
        "str mx_dtype, int a_layout, int b_layout, ScalarType out_dtype) -> Tensor");
}

TORCH_LIBRARY_IMPL(simo, PrivateUse1, m) {
  m.impl("mma_dte", &mma_dte_entry);
}
```

这里的 `PrivateUse1` 是 torch_sipu 的 SIPU device dispatch key。`vllm-sipu` 的 AOT binding generator 也在 `tools/generate_aot_bindings.py:120-149 / generate` 中使用 `torch::kPrivateUse1` 注册 SIPU operator，这与 simo 应采用的方向一致。

### 15.4 建议的 Python/C++ API

#### 15.4.1 第一版 schema

建议第一版先使用显式、可验证的 schema：

```text
torch.ops.simo.mma_dte(
    a: Tensor,                 # MX tiled physical uint8 buffer
    b: Tensor,                 # MX tiled physical uint8 buffer
    m: int,
    n: int,
    k: int,
    mx_dtype: str,             # mxint8/mxfloat8e4m3/.../mxint4
    a_layout: int,             # MMA DTE layout family
    b_layout: int,             # must equal layout contract's required RHS layout
    out_dtype: torch.dtype,    # bfloat16 / float16 / float32
) -> Tensor                  # [m, n], contiguous linear layout
```

`mx_dtype` 应使用具体格式名，而不是模糊的 `mxfp8`/`mxfp6`：

```text
mxint8
mxfloat8e4m3
mxfloat8e5m2
mxfloat6e2m3
mxfloat6e3m2
mxfloat4e2m1
mxint4
```

`a_layout`/`b_layout` 可以先用整数枚举，后续 Python 层再提供字符串常量：

```text
S4x1M32
S2x2M32
S1x4M32
S1x4M16
S1x4M8
```

这些 layout 的几何和 required RHS layout 应直接复用 `mma_dte_tile_tensor` 的 layout contract，不要在 simo 中重新维护一套 M/N/K 规则。`torch_sipu/torch_sipu/csrc/contrib/native/sipu/mma_dte/BlasMmaDte.suh:227-323 / mx_mma_dte_*_config_spec` 已经展示了按 MX6、8 bit、4 bit 家族选择几何的现成模式；如果 simo 直接链接独立 MMA release，则应在 simo wrapper 中做一个最小、同步的枚举映射。

Python 薄封装可以是：

```python
def mma_dte(a, b, m, n, k, mx_dtype, a_layout, b_layout,
            out_dtype=torch.bfloat16):
    return torch.ops.simo.mma_dte(
        a, b, m, n, k, mx_dtype, a_layout, b_layout, out_dtype
    )
```

#### 15.4.2 C++ wrapper 的职责

建议 `simo/csrc/sipu/mma_dte.cpp:mma_dte_entry` 只做以下事情：

1. 检查 `a`、`b` 在 `PrivateUse1`，且在同一个 SIPU device。
2. 检查 `a.scalar_type() == b.scalar_type() == torch::kUInt8`、contiguous、非空指针和 64-byte 对齐。
3. 检查 `M/N/K` 为正数并能转换为 `int`。
4. 根据 `mx_dtype + a_layout + b_layout + out_dtype` 选择一个**已显式实例化**的 C++ 模板。
5. 检查 input 的 physical byte capacity 足够；不能只检查 `Tensor.numel()` 是否等于 `M*K`，因为它是 tile physical storage。
6. 创建 `at::empty({M, N}, a.options().dtype(out_dtype))`。
7. 使用 `c10::sipu::getCurrentSIPUStream(a.device().index()).stream()` 获取当前 stream。
8. 调用 `mma_dte`，其中 `layoutC.tensor_format` 固定为 `0`，`alpha=1`，`beta=0`，`ldc=N`。
9. 调用 SIPU launch check，然后返回 output Tensor。

`torch_sipu/torch_sipu/csrc/contrib/native/sipu/mma_dte/BlasMmaDteMxKernel.suh:57-81 / run_mx_mma_dte_tiled_tensor` 是 stream、`M/N/K`、`lda/ldb/ldc` 传递方式的直接参考；`mma_dte_tile_tensor/kernel/mma_dte_tiled_tensor.hpp:51-119 / mma_dte_implement_with_control` 是底层 shape/layout 校验和 R8/R16/R32 dispatch 入口。

这里必须明确：

```text
layoutC.tensor_format = 0  -> linear output
layoutC.tensor_format = 1  -> tiled output
```

不要调用 torch_sipu 现有只固定 tiled C 的旧 wrapper；应使用独立 MMA 库已经显式实例化的 linear-C 模板。

### 15.5 `hp_to_mx` 输出与接口元数据问题

`si-infer/siinfer/quantization.py:10-42 / hp_to_mx` 的 Python docstring 虽然称返回 tiled MX physical buffer，但返回值没有附带 dtype/layout/shape 对象。

native 层 `si-infer/csrc/native/bindings.cpp:8974-9023 / hp_to_mx_native` 明确：

- 输入要求是 `[batch, dim0, dim1]`；
- 输出默认是 `torch.uint8`；
- 输出是 contiguous 1-D buffer；
- `required_bytes` 由 `hp_to_mx_storage_size` 计算。

device wrapper `si-infer/csrc/quantization/kernels/sipu150/hp_to_mx/hp_to_mx.su:25-85 / hp_to_mx` 会根据 `dim0` 选择 Rows=8/16/32；`hp_to_mx_tensormap.hpp:84-113 / hp_to_mx_direct_shape_supported` 和 `hp_to_mx_physical_extents` 决定物理 tile/supertile 尺寸。

所以 `mma_dte` 不能从 `a.shape`/`b.shape` 自动恢复完整契约。推荐调用方保留如下 metadata：

```text
MxTensor {
    storage: uint8 Tensor
    logical_shape: (rows, k)
    mx_dtype: string
    mma_layout: enum
    input_layout: tiled
}
```

如果当前不想引入 `MxTensor` Python class，至少把 `(M, N, K, mx_dtype, a_layout, b_layout)` 显式传给 `torch.ops.simo.mma_dte`。

特别注意：MMA DTE 对 B 的 layout 有约束。`torch_sipu/torch_sipu/csrc/contrib/native/sipu/mx.cpp:108-127 / check_mx_rhs_layout` 要求 B 使用 M32 row tile，并且必须等于 A layout 对应的 `required_rhs_layout`。因此不能简单地对任意 `[N, K]` 调用 `hp_to_mx(..., input_layout="tiled")` 后直接当作 MMA B：当 N 很小时，`hp_to_mx.su:81-85 / hp_to_mx` 可能选择 R8/R16，而当前 MMA layout contract 需要 R32 B。调用层应按 MMA layout 先 padding/选择正确的 quantization shape，并保存 layout metadata。

### 15.6 当前 MMA release 的真实 output dtype 支持矩阵

用户期望的“所有 MX dtype 都支持 BF16/FP16/FP32”不能仅靠 simo wrapper 达成。当前 `/share/users/like/package/oplib/mma_dte_tile_tensor` 源码的显式实例化情况是：

| MX input | 当前独立 MMA 源码的 linear-C output | 证据 |
|---|---|---|
| MXINT8 | BF16、FP16、FP32 | `kernel/instantiations/mxint8/inst_mxint8_r32_4x1.su:21-40`，R8/R16 结构相同 |
| MXFP8 E4M3/E5M2 | BF16、FP16；当前没有 FP32 实例 | `kernel/instantiations/mxfp8/inst_mxfp8_r32_4x1_bf16.su:6-48` 及对应 `f16` 文件 |
| MXFP6 E2M3/E3M2 | BF16、FP16、FP32 | `kernel/instantiations/mxfloat6e2m3/inst_mxfloat6e2m3_r32_4x1.su:21-39`，E3M2 结构相同 |
| MXFP4 E2M1 | BF16；当前没有 FP16/FP32 实例 | `kernel/instantiations/mxfp4/inst_mxfp4.su:23-61` |
| MXINT4 | BF16；当前没有 FP16/FP32 实例 | `kernel/instantiations/mxint4/inst_mxint4.su:27-61` |

因此建议分两阶段：

```text
第一阶段：simo 暴露当前 release 已存在的组合，unsupported 组合明确报错。
第二阶段：如果业务必须要求 MXFP8/MXFP4/MXINT4 的 FP32/FP16 output，
          先在 mma_dte_tile_tensor 中新增对应 .su 显式实例化和必要 device path，
          再更新 simo 的 dispatch manifest；不要在 simo wrapper 中伪造支持。
```

### 15.7 `torch_sipu` 的 `.so` 是否依赖 `libtile_mma_dte.so`

#### 15.7.1 当前安装包的 ELF 事实

检查目录：

```text
/share_data/users/like/miniconda3/envs/vllm_dev/lib/python3.10/site-packages/torch_sipu/
```

当前 `lib/150/` 中没有 `libtile_mma_dte.so` umbrella，只有多个 `libtile_mma_dte_*.so` component DSO。

`readelf -d` 得到的关键依赖是：

| 文件 | 直接 `DT_NEEDED` 中的 MMA 依赖 |
|---|---|
| `libtorch_sipu.so` | 没有 `libtile_mma_dte.so`，也没有 `libtile_mma_dte_*.so`；它直接包含 torch_sipu 自己编译出的 `mma_dte` 符号 |
| `_C.cpython-310-*.so` | 主要依赖 `libtorch_sipu.so` |
| `libsiblas.so` | 依赖 `libsiblas_kernel_150.so`，不直接依赖 tile MMA DSO |
| `libsiblas_gemm_dte_external_adapter_150.so` | 直接依赖 `libtile_mma_dte_bf16_r16.so`、`libtile_mma_dte_bf16_r32_bf16out.so`、`libtile_mma_dte_bf16_r8.so`、对应 FP16 component |

因此对问题“torch_sipu 生成的 `.so` 是否显式依赖 `libtile_mma_dte.so`”的回答是：

```text
libtorch_sipu.so：否。
torch_sipu 安装包中的 siBLAS external adapter：不依赖 umbrella，直接依赖若干 component DSO。
当前安装目录：没有 libtile_mma_dte.so umbrella。
```

这是安装包级事实，不应把之前独立 mma release 中存在的 umbrella 和 torch_sipu 当前 wheel 混为一谈。

#### 15.7.2 `torch_sipu` 源码实际采用 a 还是 b

对用户给出的二选一，必须分两层回答：

**torch_sipu 主库的 MMA wrapper 是 a。**

- `torch_sipu/cmake/public/mma_dte.cmake:90-106` 明确把 MMA DTE 标成 `header-only`，这里只提供 include dirs，不设置 `libtile_mma_dte.so` link target。
- `torch_sipu/torch_sipu/CMakeLists.txt:90-98` 从 torch_sipu 自己的 `csrc/.../*.su` 收集 SIPU source。
- `torch_sipu/torch_sipu/CMakeLists.txt:344-443` 对每个 `.su` 单独执行 host-only 和 device-only 编译。
- `torch_sipu/torch_sipu/CMakeLists.txt:462-480` 把 device objects link 成 `torch_sipu-sipu-sipu150.elf`。
- `torch_sipu/torch_sipu/CMakeLists.txt:507-520` 将 host objects 和 device ELF 通过 `--override-image=sipu=...` link 进 `libtorch_sipu.so`。

因此主库不是先通过 mma 子模块 CMake 生成 umbrella，再让 torch_sipu wrapper link umbrella；而是把 `mma_dte_tile_tensor` 的 header/device template 纳入 torch_sipu 自己的 AOT 构建图。

**同一个 torch_sipu 发布包中的 siBLAS external DTE adapter 是另一条路径。**

- `torch_sipu/third_party/siblas/CMakeLists.txt:535-618` 会把 MMA 子模块作为子构建，生成 component DSO。
- `:548-551` 明确说明 siBLAS runtime path 不使用 umbrella。
- `:877-915` 的 `siblas_gemm_dte_external_adapter_<arch>` 直接 link 选中的 component library。
- `torch_sipu/torch_sipu/CMakeLists.txt:936-952` 安装 adapter 和 `libtile_mma_dte_*.so` component。

所以更准确的结论是：

```text
torch_sipu 自己的 MMA operator：方案 a。
torch_sipu 同步构建的 siBLAS external adapter：子模块 CMake + component DSO，
                    但不是 link umbrella 的方案 b。
```

### 15.8 simo 应该链接 umbrella 还是 component DSO

对于 simo，不建议复制 siBLAS 的内部策略，也不建议直接依赖 component DSO 名称。建议：

```text
simo host wrapper -> link libtile_mma_dte.so umbrella
libtile_mma_dte.so -> 运行时在同目录 dlopen 对应 component DSO
```

原因：

1. `mma_dte_tile_tensor/CMakeLists.txt:318-392` 的正式发布结构就是 component DSO 加 umbrella trampoline。
2. `include/simma.h:121-187 / mma_dte` 是对外 host API；pkg-config 文件的 `Libs` 也只写 `-ltile_mma_dte`。
3. component 文件名和实例拆分是实现细节；直接 link component 会把 simo 绑定到当前 delivery profile 的内部拆分。
4. umbrella 能通过完整 C++ symbol 把调用转给正确 component，simo wrapper 不需要自己维护 component 选择表。

只有在确认组件 DSO ABI 永远稳定、并且需要极限缩小 wheel 体积时，才考虑直接 link selected component；那应作为后续优化，不作为第一版架构。

### 15.9 与 hpc-ops 的构建和注册模式对比

`hpc-ops` 的可复用部分是 operator registration，不是 CUDA 构建细节：

- `hpc-ops/src/C/C.cc:3-5` 用空的 `TORCH_LIBRARY(hpc, m)` 创建 namespace。
- `hpc-ops/src/group_gemm/entry.cc:397-421 / TORCH_LIBRARY_FRAGMENT` 定义 schema 并用 `m.impl(..., torch::kCUDA, ...)` 绑定 C++ entry。
- `hpc-ops/hpc/group_gemm.py:110-131 / group_gemm_fp8` 只是调用 `torch.ops.hpc.group_gemm_fp8`。
- `hpc-ops/hpc/group_gemm.py:200-225` 通过 `register_fake` 提供 fake/meta shape。
- `hpc-ops/setup.py:17-70 / CMakeExtension、CMakeBuild::build_extension` 用 setuptools 作为外层入口、CMake 作为真正的 native build driver。
- `hpc-ops/CMakeLists.txt:117-215` 构建 Python MODULE 并 link PyTorch/CUDA 库。

simo 应采用同样的分层：

```text
setup.py       -> Python packaging/build orchestration
CMake          -> SIPU AOT MMA DTE build and DSO staging
torch_bindings -> schema/dispatch registration
mma_dte.cpp    -> device-aware host wrapper
mma_dte.py     -> optional Python convenience API
```

但不应照搬 hpc-ops 的 `CUDAExtension`、SM architecture loop 或 `py_limited_api`；SIPU 的 `.su`/device ELF/RPATH 规则不同。

### 15.10 推荐的 simo 调用链

```text
pip install -e .
  -> setup.py:get_build_backend
  -> backend == sipu
  -> setup.py:SimoBuildExtension
  -> CMake configure third_party/mma_dte_tile_tensor
  -> CMake build tile_mma_dte
       -> scc compile .su
       -> link libtile_mma_dte_<type>.so
       -> generate umbrella trampoline
       -> link libtile_mma_dte.so
  -> CppExtension builds simo._C
       -> torch_bindings.cpp defines simo::mma_dte
       -> mma_dte.cpp registers PrivateUse1 implementation
       -> link libtile_mma_dte.so
  -> copy simo/mma_dte/*.so beside simo._C
  -> import simo
  -> simo._get_native_ops(): import simo._C
  -> torch.ops.simo.mma_dte(a, b, M, N, K, ...)
  -> mma_dte_entry
       -> validate PrivateUse1/uint8/layout/shape/storage
       -> allocate linear C [M,N]
       -> current SIPU stream
       -> mma_dte<... layoutC.tensor_format=0>
       -> return C
```

### 15.11 AOT 构建中需要避免的方案

#### 方案一：运行时用 `torch.utils.cpp_extension.load` 编译

不采用。这是 JIT，和本题只考虑 AOT 的目标冲突，也会在 vLLM worker 启动时引入编译、缓存目录和并发问题。

#### 方案二：在 simo 里复制 torch_sipu 的全部 `.su` wrapper

不作为第一版。它会同时复制：

- MX dtype/layout dispatch 表；
- R8/R16/R32 wrapper；
- host/device 双 pass CMake custom command；
- device ELF `--override-image` link；
- stream/error handling；
- 未来每次 mma 子模块更新的实例同步。

这条路能避免 component DSO，但代码重复和维护面明显大于“link umbrella + 同目录 component”。

#### 方案三：直接调用 torch_sipu 的私有 `SIPUExtFunctions::_mx_mm`

不采用。它不是 simo 的稳定公共 ABI；而且 `torch_sipu/torch_sipu/csrc/contrib/native/sipu/mx.cpp:798-819 / SIPUExtFunctions::_mx_mm` 当前只允许 BF16/FP16 output，并依赖 torch_sipu 内部 MX Tensor view/layout 约定。用户要求的是 simo 自己的 `torch.ops.simo.mma_dte` 和 linear-C 契约，直接耦合 torch_sipu 私有实现会把两个项目绑定在一起。

### 15.12 验证计划

第一阶段至少应有以下验证：

1. 构建验证：`readelf -d simo._C*.so` 能看到 `libtile_mma_dte.so`，且 RPATH 指向 `$ORIGIN/mma_dte`。
2. 依赖验证：`libtile_mma_dte.so` 和所需 `libtile_mma_dte_*.so` 均存在；`ldd -r` 不出现未解析的 host symbol。
3. dispatcher 验证：导入 `simo` 后，`torch.ops.simo.mma_dte.default` 存在，并且 `PrivateUse1` dispatch kernel 已注册。
4. layout 验证：A/B 使用 `siinfer.hp_to_mx(..., input_layout="tiled")` 生成的 physical buffer；A layout 和 B required RHS layout 正确匹配。
5. output 验证：返回 Tensor 为 SIPU device、shape `[M,N]`、contiguous、stride `(N,1)`、`layoutC.tensor_format=0`。
6. 数值验证：每个已支持的 MX dtype、R8/R16/R32 layout、BF16/FP16/FP32 output 组合，使用独立 dequant/reference GEMM 对比。
7. 异步语义验证：wrapper 不调用 device synchronize；只使用当前 SIPU stream，并在 launch 后做轻量 error check。
8. wheel/editable 验证：editable 结果从 source tree 加载 DSO；wheel 解压到干净目录后，`simo._C` 能通过相对 RPATH 找到 umbrella 和 component。

当前环境在做 Python import smoke test 时还遇到既有运行环境问题：`libtorch_sipu.so` 要求 `GCC_13.0.0`，而当前 `/lib/x86_64-linux-gnu/libgcc_s.so.1` 不提供该 symbol。这个错误发生在 `torch_sipu._C` 加载阶段，不是 simo MMA DTE 设计本身的错误；在正式运行验证前需要先加载与 torch_sipu wheel 匹配的 GCC runtime。

### 15.13 最终决策

```text
是否需要 tvm-ffi：不需要。
是否需要新增 pybind11 绑定：不需要；沿用现有 torch_bindings.cpp。
是否建议新增 CMakeLists.txt：建议。不是 torch.ops 的要求，而是 AOT SIPU .su/DSO 构建的要求。
torch_sipu 主库采用 a 还是 b：主库是 a；siBLAS external adapter 是 component DSO 路径，不是 umbrella b。
simo 应链接什么：优先 link libtile_mma_dte.so umbrella，并把 component DSO 放在同目录。
输入 API 是否只接收两个 Tensor：不可以；必须显式传 M/N/K、具体 MX dtype 和 layout metadata。
output 是否固定 linear：是；模板 layoutC 的 tensor_format 固定为 0。
是否当前五类 MX 都能输出三种 hp dtype：不能直接宣称；当前源码的支持矩阵必须如 15.6 所列，缺失组合先报 unsupported 或先扩展 mma 子模块。
```
