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
