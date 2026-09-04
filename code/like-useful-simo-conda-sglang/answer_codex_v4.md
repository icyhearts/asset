# SGLang v0.5.18-local-dep 启动耗时分析

分析对象：

- 旧版本：main-local-dep（日志中还包含一次 main-2026_07_08___17_53_37）
- 新版本：release/v0.5.18-local-dep
- 启动参数：like-useful/dsv4-flash-run.sh，TP=4，Marlin，EAGLE，speculative-num-steps=3，speculative-num-draft-tokens=4，cuda-graph-max-bs=16
- 环境：/share_data/users/like/miniconda3/envs/simo_sglang/；源码以 editable 方式安装在 /share/users/like/package/sglang_kernel_src

## 结论

启动时间增加的主因已经定位到：

1. 新版本的 full CUDA Graph 目标验证（target_verify）阶段从旧版本约 6.1～7.7 分钟增加到约 18.4～19.3 分钟。
2. 这段时间内会触发 moe_wna16_marlin 的 C++/CUDA JIT。新版本的 content-addressed JIT cache 在当前 /softhome -> /share_data symlink 布局下把构建目录中的临时 cuda.cu 记录成了绝对依赖；构建结束后临时目录被删除，下一次查找缓存时必然判定依赖失效。
3. 4 个 TP rank 因锁机制依次重新编译同一个 Marlin JIT 模块，每次约 3 分 33 秒～3 分 44 秒，单次启动出现 4 次，合计约 14.5～15 分钟。这与 target_verify 首个 bs=16 桶耗时约 17.8～18.3 分钟高度吻合，是本次升级后每次启动变慢的主要根因。

因此，不能把问题简单归结为“CUDA Graph 桶数量变多”：新旧版本都捕获了 12 个桶 [1,2,3,4,5,6,7,8,10,12,14,16]，且最慢的都是首个 bs=16。旧版本日志把该路径统称为 generic CUDA graph；在 EAGLE 模式下它实际上也是 target-verify 语义。新版本只是把它拆分并显式命名为 target_verify，同时 JIT/cache 路径发生了变化。

MHC/DeepGEMM 的懒编译、分布式初始化变慢以及环境混用会增加几十秒到约 1 分钟，或造成少量波动，但不能解释 10～13 分钟的主要差值。

## 1. 启动总时间和阶段对比

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

### 1.1 主要差值

旧版本 generic graph 为 367.4～463.5 s，新版本对应的 target_verify 为 1106.0～1156.3 s：

- 增加 642.5～788.9 s，即增加约 10.7～13.2 分钟；
- 新/旧比值约 2.4～3.1 倍；
- 仅这一项就解释了总启动时间从 8～10 分钟变成 23～29 分钟的大部分差异。

新版本进度条显示首个 bs=16：

- 2026-09-01：约 1093.98 s（18:14）；
- 2026-09-03：约 1069.06 s（17:49）。

首个 bs=16 完成后，bs=14、12、10 等剩余桶大多在秒级或几十秒内完成。因此主耗时是首个桶中的 kernel/JIT 初始化，而不是 12 个桶平均变慢。

## 2. 主要根因：新 JIT cache 没有命中，Marlin 被 4 个 rank 串行重编译

### 2.1 新旧 loader/cache 行为不同

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

### 2.2 缓存 manifest 中记录了已删除的 staging 文件

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

### 2.3 每次启动出现四次连续 Marlin 编译

当前缓存目录中，模块 sgl_kernel_jit_moe_wna16_marlin_bf16_t_false_false 的 .so 时间戳如下：

| 启动 | 4 个 .so 完成时间 | 单次编译耗时 |
|---|---|---:|
| 2026-09-01 17:32 | 17:43:03、17:46:48、17:50:24、17:54:02 | 约 3:33～3:45 |
| 2026-09-01 18:04 | 18:17:55、18:21:32、18:25:36、18:29:22 | 约 3:37～3:45 |
| 2026-09-03 16:21 | 16:30:21、16:33:59、16:37:43、16:41:21 | 约 3:37～3:39 |

四次编译在同一个模块、同一个 build key 下按时间串行出现，与 4 个 TP rank 争用 loader lock、但每个 rank 的二次 cache check 都因坏 manifest 失败的行为一致。四次合计约 14.5～15.0 分钟，正是 target_verify 首个 bs=16 长时间段的主要组成部分。

python/sglang/kernels/ops/moe/moe_wna16_marlin.py:18-33 会调用新 load_jit；在当前参数（bf16、is_ep=false、has_bias=false）下对应上述模块名。旧版本使用的 loader 和模块命名不同，所以旧版已有的 TVM FFI .so 不能直接证明新 loader 能命中。

## 3. 次要增量和其它差异

### 3.1 MHC fused post/pre 与 TileLang 懒编译

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

### 3.2 DeepGEMM 预编译被显式关闭，缓存根目录也迁移

脚本设置 SGLANG_JIT_DEEPGEMM_PRECOMPILE=0。新版本 compile_utils.py:31-40 会把 DG_JIT_CACHE_DIR 重定向到 SGLANG_DG_CACHE_DIR；同时 release 分支把默认 DeepGEMM cache 从旧的 ~/.cache/deep_gemm 迁移到 ~/.cache/sglang/deep_gemm。日志中可见 MHC prewarm 和 target_verify 期间产生 DeepGEMM 产物。

这会造成额外 JIT 和缓存冷启动，但从时间量级看属于次要项。开启 precompile 可能只是把成本前移，必须用相同 cache 和相同环境做 A/B，不能直接当作根因修复。

### 3.3 分布式初始化变慢

旧版分布式初始化约 10.6～13.4 s，新版约 62.7～68.1 s，增加约 50～57 s。新版日志使用 NCCL 2.29.7，旧版为 NCCL 2.28.9；两者存在时间相关性，但仅凭现有日志不能证明 NCCL 版本是唯一原因。它明显小于 target_verify 的十几分钟差值。

### 3.4 环境不完全可比

2026-09-01 18:04:02 这次日志引用了 simo_sglang_pip 的 site-packages；2026-09-01 17:32:27 和 2026-09-03 使用的是 editable source。当前 Marlin 的 cuda.cu 依赖中同时出现：

- /share/users/like/package/sglang_kernel_src/python/sglang/...
- /share_data/users/like/miniconda3/envs/simo_sglang_pip/lib/python3.12/site-packages/sglang/...

这会造成 build key 和缓存碎片化，因此 18:04:02 不应作为严格的同环境 A/B 样本。后续应固定使用 simo_sglang 的 python 和 sglang 可执行文件。

### 3.5 不能归因的告警/阶段

- 新日志明确提示 prefill CUDA graph disabled；因此启动慢不是 prefill graph 捕获。
- FlashInfer 的 libcudart_stub.so: undefined symbol: cudaDeviceReset 会禁用 allreduce fusion。这是 CUDA/TileLang 动态库不匹配，应修复后复测，但现有证据不能把它归因到 18 分钟的 target_verify。
- SIMO plugin 报告找不到 model_runner_kv_cache_mixin，服务仍继续启动；目前没有分钟级耗时证据。
- 2026-09-01 18:04:02 在权重/内存池结束到 target_verify 开始之间还有约 202 s 未被细分，startup timing 的 scheduler_e2e 包含该间隔，需要增加更细日志；这属于额外待查项，不改变 Marlin JIT 是主要热点的判断。

## 4. 建议的验证和修复顺序

### 4.1 先固定环境和新的 canonical cache

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

### 4.2 修复 cache.py 的 canonical path 比较

在 python/sglang/kernels/jit/utils/compile/cache.py 的 _to_entries 中，让 build_dir 与 candidate 使用同一个 resolve 结果，再执行 is_relative_to。重新生成缓存后，检查 sgl_deps.json 不再包含已删除的 .staging-*/cuda.cu。

不要仅删除共享缓存后就宣布修复；需要用 SGLANG_JIT_CACHE_DEBUG=1 连续启动两次，确认：

- 第一次最多为每个模块编译一次；
- 后续 TP rank 显示 cache hit，而不是四次连续的 moe_wna16_marlin 编译；
- 第二次启动不再出现新的 .staging-UUID 依赖；
- target_verify 的首个 bs=16 时间显著下降。

### 4.3 再做隔离 A/B

固定同一个 editable 环境、同一个 canonical cache 后，分别比较：

~~~bash
SGLANG_OPT_FUSE_MHC_POST_PRE=0
~~~

以及当前值；必要时临时降低 cuda-graph-max-bs 或使用 disable-cuda-graph 仅作诊断。若关闭 fused MHC 只减少几十秒，而修复 JIT cache 能减少十几分钟，就能进一步确认归因。

### 4.4 最后处理环境告警

修正 FlashInfer/TileLang 与 CUDA 13.0 的动态库匹配，清理或更新 stale SIMO plugin import；同时记录 NCCL 版本和启动阶段的更细时间点。它们是稳定性和次要启动时间问题，不应替代 JIT cache 修复。

## 5. 最终判断

按当前证据对启动时间增长做排序：

1. **主要根因（高置信度）**：release/v0.5.18-local-dep 新 JIT cache 的 staging 依赖记录受 symlink canonicalization 影响而失效，导致 target_verify 首个 bs=16 触发 4 次串行 Marlin JIT，贡献约 14.5～15 分钟。
2. **阶段级表现（确定）**：因此日志中最明显的新增耗时出现在 target_verify full CUDA Graph capture，约比旧 generic/verify 路径多 10.7～13.2 分钟。
3. **次要因素（中等置信度）**：MHC fused post/pre 的 TileLang 懒编译、DeepGEMM precompile=0 及新 cache 根目录迁移，冷启动约几十秒到 1 分钟。
4. **独立波动项**：分布式初始化多约 50～57 秒；Sep-01 18:04 的环境混用和约 202 s 未细分间隔需要单独复测。
5. **待修复但未证明为主因**：FlashInfer 动态库符号错误、SIMO plugin 导入错误。

因此，优先修复/绕过 SGLANG JIT cache 的 manifest 问题并固定 editable 环境；在此之前对 CUDA Graph、MHC 或 NCCL 做性能结论都容易被反复 JIT 编译噪声掩盖。

---

# 为什么 SIMO 要关闭 chunked prefix cache

本节针对 `sglang` 的 `release/v0.5.18-local-dep`（当前源码提交 `982d8495b7`）和本仓库的 SIMO 量化路径。下面的行号以本次检查的工作树为准，后续提交若插入代码，行号可能变化。所有路径均为各自 code base 的相对路径：SGLang 的根目录是 `/share/users/like/package/sglang_kernel_src`，SIMO 的根目录是 `/share/users/like/package/simo_conda_sglang`。

代码引用采用“相对路径:行号，类::函数”的形式；对于字段、常量和模块级注册，则直接写出对应符号名。

## 结论

`chunked prefix cache` 有两个容易混淆的“默认值”:

1. **参数定义层面默认允许开启**：`ServerArgs.disable_chunked_prefix_cache` 的默认值是 `False`。这是一个否定式选项，未传 `--disable-chunked-prefix-cache` 就表示“不禁用”。见 `python/sglang/srt/server_args.py:958-962，ServerArgs.disable_chunked_prefix_cache`。
2. **运行时不一定实际开启**：模型加载时，SGLang 会检查模型是否使用 MLA，以及 prefill attention backend 是否在支持列表中；不满足条件时会把生效值改成 `True`（禁用）。见 `python/sglang/srt/model_executor/model_runner.py:369-372，ModelRunner.__init__` 和 `python/sglang/srt/model_executor/model_runner_components/misc_utils.py:25-48，maybe_disable_chunked_prefix_cache`。

所以更准确的回答是：**SGLang 的开关默认是“开启倾向”（disable=False），但功能有运行时能力门控；在当前 SIMO 评测中实际生效值应为 `True`，即关闭。**

字段帮助文本还说明，关闭它可以为短序列节省额外调度/路径开销，见 `python/sglang/srt/server_args.py:958-962，ServerArgs.disable_chunked_prefix_cache`。这只是原生 SGLang 的通用性能取舍；对 SIMO 来说，首要原因是量化 pool 和 kernel 尚未实现该路径所需的接口，不能把本次关闭理解成单纯的性能调参。

## 1. “chunked prefix cache”不是普通 prefix cache

普通的 Radix/prefix cache 由 `disable_radix_cache` 控制，该选项在 `python/sglang/srt/server_args.py:937-939，ServerArgs.disable_radix_cache` 中默认也是 `False`。`disable_chunked_prefix_cache` 只控制 DeepSeek MLA 在长前缀场景下采用的“分块 MHA 前缀读取”路径，不能把两个开关等同起来。关闭 chunked prefix cache **不会自动关闭所有 Radix prefix cache**。

## 2. SGLang 默认值如何变成实际生效值

### 2.1 参数本身

`disable_chunked_prefix_cache=False` 位于 `schedule` 配置组，含义是“不要禁用”。因此命令行不写该选项时，配置初值是允许功能的，而不是明确关闭功能。见 `python/sglang/srt/server_args.py:958-962，ServerArgs.disable_chunked_prefix_cache`。

### 2.2 加载时能力门控

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

## 3. SGLang 的 chunked 路径具体做了什么

DeepSeek MLA 初始化时把 schedule 中的开关和阈值复制到 attention 对象，见 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:132-139，DeepseekMHAForwardMixin.init_mha_forward`。阈值注释说明，前缀总长度默认达到 8192 才考虑该路径；较短的非空前缀继续使用吸收式 MLA，见 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:90-100，DeepseekMHAForwardMixin 的 chunk 配置说明`；默认值实际定义在 `python/sglang/srt/environ.py:586，RuntimeEnvs.SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD`。

通用 DeepSeek backend dispatch 在 `python/sglang/srt/models/deepseek_common/attention_backend_handler.py:99-132，_handle_attention_backend` 中要求同时满足：extend 模式、前缀长度达到阈值、且 `not attn.disable_chunked_prefix_cache`；满足后选择 `MHA_ONE_SHOT` 或 `MHA_CHUNKED_KV`，否则选择 MLA 子路径。

真正的分块执行见 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:286-320，DeepseekMHAForwardMixin.forward_normal_chunked_kv_core`：

1. 先对当前 extend 部分做一次 MHA；
2. 当存在缓存前缀时，把 `forward_batch.mha_return_lse` 设为 `True`；
3. 对每个前缀 chunk 调用 `_chunked_prefix_attn_mha`，再用 LSE 合并各块结果。

逐块读取和合并的实现位于 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:352-418，DeepseekMHAForwardMixin._chunked_prefix_attn_mha`。该函数通过 `_get_mla_kv_buffer` 读取 latent/rope 缓存，经过 `kv_b_proj` 重建 MHA 的 K/V，然后调用 attention 并执行 `merge_state_v2`。上游的 raw reader 入口是 `python/sglang/srt/mem_cache/memory_pool.py:4236-4265，MLATokenToKVPool.get_mla_kv_buffer`，它返回未量化的 MLA buffer。

`RadixAttention.forward` 会根据 `mha_return_lse` 选择带 LSE 的 unified attention op，并在该标志为真时返回 `(output, lse)`，见 `python/sglang/srt/layers/radix_attention.py:150-159,244-277，RadixAttention.forward`。因此这条路径不仅需要“能读 KV”，还需要 backend 正确产出 LSE。

此外，完整 prefill CUDA graph 也读取这个开关：`python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:410-425，PrefillCudaGraphRunner.__init__` 用 `not get_schedule().disable_chunked_prefix_cache` 决定是否建立 chunked-prefix graph 拓扑。它不是单纯的“是否启用 CUDA Graph”开关；`disable_cuda_graph` 与它是两个独立选项。

## 4. SIMO 为什么不能沿用这条路径

### 4.1 SIMO pool 不是上游 raw MLA buffer

SIMO 的 MLA pool 用 `uint8` 保存打包后的量化 payload 和 scale bytes，见 `simo/extensions/sglang_simo/mem_cache/memory_pool.py:329-349，SIMOMLATokenToKVPool._create_buffers`。这与上游 `MLATokenToKVPool.get_mla_kv_buffer` 所要求的未量化 latent/rope dtype 不同。

因此 SIMO 明确拒绝上游 chunked helper 调用 raw reader，见 `simo/extensions/sglang_simo/mem_cache/memory_pool.py:360-380，SIMOMLATokenToKVPool.get_mla_kv_buffer`。如果把打包字节直接当 BF16 latent 读取，结果不是精度下降这么简单，而是会得到错误的 K/V；要支持它，必须新增按 chunk 索引读取、反量化并恢复布局的实现。

### 4.2 SIMO kernel 没有 LSE 返回值

量化路径的 `SIMOTritonAttnBackend.forward_extend` 最终调用 SIMO 自定义 dequant attention kernel。该函数在检测到 `forward_batch.mha_return_lse` 时显式抛出异常，见 `simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py:124-164，SIMOTritonAttnBackend.forward_extend`。原因是当前 kernel 只返回 attention output，不返回 chunk 合并所需的 LSE；而上游 caller 会在 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:302-315，DeepseekMHAForwardMixin.forward_normal_chunked_kv_core` 解包 `(attn_output, lse)`。

所以当前支持 chunked prefix cache 需要同时补齐：

- 量化 MLA pool 的 chunked raw-reader/dequant 接口；
- 与 `merge_state_v2` 对齐的 LSE 输出和数值语义；
- SIMO attention backend 的 prefix-chunk metadata、布局和 CUDA graph 支持。

仅把 `triton_simo` 加入白名单，或仅删除脚本中的 disable 参数，都会绕过自动保护并触发上述未实现接口，不是正确修复。

## 5. 为什么评测脚本要显式写 `true`

当前脚本在指定 attention backend 时统一传入该选项，见 `simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:85-90，run_simo_config_list`：

```json
"disable_chunked_prefix_cache": true
```

这是一个正确性和防御性设置，原因有三点：

1. 它明确记录了 SIMO 当前不支持该上游路径，不依赖模型识别或 backend 白名单的隐式结果；
2. 即使后续注册流程把 `triton_simo` 加入支持列表，或 SGLang 改变 handler fallback，也不会误进入需要 raw reader/LSE 的路径；
3. 它与 `disable_cuda_graph=true` 的作用不同：前者关闭 DeepSeek chunked-prefix dispatch，后者关闭 CUDA graph，本次显式关闭前者是针对 SIMO KV cache 量化接口的限制。

因此本次评测中应把“实际生效值”理解为 `True`。普通 Radix prefix cache 仍由 `disable_radix_cache=False` 独立控制，短前缀也仍可走普通的吸收式 MLA；关闭的只是长前缀 chunked MHA 方案。

## 6. 如何判断一次启动的真实状态

不要只看命令行参数名的直觉含义。先看 `maybe_disable_chunked_prefix_cache` 是否打印 `Chunked prefix cache is turned on.`，该日志分支见 `python/sglang/srt/model_executor/model_runner_components/misc_utils.py:42-48，maybe_disable_chunked_prefix_cache`；再结合 resolved schedule config。由于自动门控写入的是 runtime config bag，`python/sglang/srt/runtime_context.py:1119-1120，get_schedule` 返回的是生效配置视图，而不是未经覆盖的原始 `ServerArgs` 对象。

最终判断：**SGLang 的 `ServerArgs` 默认值是 `disable=False`，即功能默认允许；实际是否开启由 MLA/backend 能力门控决定。对当前 Llama3.1 + DeepSeek-V2-Lite 的 SIMO 量化测试，显式 `disable_chunked_prefix_cache=true` 是必要且正确的，当前有效值为关闭。**

## 7. 四个环境变量的作用，以及与 Marlin/DeepGEMM 编译的关系

本节按当前 checkout（release/v0.5.18-local-dep）的实际代码说明。四个变量都是“缓存/诊断”变量，不是选择 Marlin 或 DeepGEMM 算法的开关；是否走 Marlin 由启动参数 `--moe-runner-backend marlin` 决定，DeepGEMM 是否启用/预编译由另外的 `SGLANG_ENABLE_JIT_DEEPGEMM`、`SGLANG_JIT_DEEPGEMM_PRECOMPILE` 等变量决定。

### 7.1 一览

| 环境变量 | 当前代码默认值 | 直接控制的内容 | 对 Marlin | 对 DeepGEMM |
|---|---|---|---|---|
| `SGLANG_CACHE_DIR` | `~/.cache/sglang` | SGLang 通用缓存根，以及 Triton/Inductor/FlashInfer/CUDA driver 等第三方缓存的默认根 | 不直接决定新 `load_jit` 的目录 | 通过默认值间接决定 `<root>/deep_gemm` |
| `SGLANG_JIT_CACHE_DIR` | 未设置时回退到 `~/.cache/sglang/jit` | SGLang 自己的 content-addressed C++/CUDA JIT（新 `load_jit`）根目录 | **直接控制 Marlin .so 的缓存和复用** | 不控制 DeepGEMM |
| `SGLANG_DG_CACHE_DIR` | `<SGLANG_CACHE_DIR>/deep_gemm` | DeepGEMM 编译产物缓存目录 | 不控制 Marlin | **直接控制 DeepGEMM 的 kernel.cu/cubin 缓存和复用** |
| `SGLANG_JIT_CACHE_DEBUG` | `false` | JIT cache miss/rebuild 原因的日志级别 | 只帮助诊断 Marlin cache miss | 不改变 DeepGEMM 编译或其缓存 |

布尔变量接受 `true/1/yes/y` 和 `false/0/no/n`（见 `environ.py:133-140`）。

### 7.2 `SGLANG_CACHE_DIR`：通用缓存根

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

### 7.3 `SGLANG_JIT_CACHE_DIR`：Marlin 使用的 SGLang JIT 根

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

### 7.4 `SGLANG_DG_CACHE_DIR`：DeepGEMM 的缓存目录

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

### 7.5 `SGLANG_JIT_CACHE_DEBUG`：只增加 cache miss 原因可见性

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

### 7.6 当前 `dsv4-flash-run.sh` 的实际配置

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

### 7.7 与本次启动耗时问题的对应关系

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

## 本次追加结论：chunked prefix cache

按 `release/v0.5.18-local-dep` 当前源码，`ServerArgs.disable_chunked_prefix_cache` 的字段默认值是 `False`，因此**配置语义是默认允许 chunked prefix cache**，见 `python/sglang/srt/server_args.py:958-962，ServerArgs.disable_chunked_prefix_cache`。这不是“每个模型都实际开启”：`python/sglang/srt/model_executor/model_runner.py:369-372，ModelRunner.__init__` 会调用 `python/sglang/srt/model_executor/model_runner_components/misc_utils.py:25-48，maybe_disable_chunked_prefix_cache`，对非 MLA 模型或不在 `python/sglang/srt/server_args.py:211-224，CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS` 中的 prefill backend 自动覆写为 `True`（禁用）。

此前版本的评测曾把 Llama3.1 权重量化也指定为 `triton`，而 DeepSeek KV 量化使用 `triton_simo`。本次修正后，权重量化不再指定 attention backend，只有 KV 量化保留 `triton_simo`；KV 分支在 `simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:89-95，run_simo_config_list` 显式传入 `"disable_chunked_prefix_cache": true`，是为了把 packed-KV 的兼容性限制固定下来，而不是依赖隐式门控。

关闭的直接原因是 SIMO 尚未实现上游 chunked 路径的两个契约：`simo/extensions/sglang_simo/mem_cache/memory_pool.py:360-380，SIMOMLATokenToKVPool.get_mla_kv_buffer` 不能把 packed `uint8` + scale 当作 raw MLA BF16 buffer 读取；`simo/extensions/sglang_simo/layers/attention/triton_simo_backend.py:124-164，SIMOTritonAttnBackend.forward_extend` 不产生上游分块合并所需的 LSE。上游分块 caller 在 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:286-320，DeepseekMHAForwardMixin.forward_normal_chunked_kv_core` 会读取每个 prefix chunk 并解包/合并 `(output, lse)`。

因此：**字段默认值为“允许”（False），当前 SIMO KV 分支的有效值为“关闭”（True）**。这不会关闭 `disable_radix_cache=False` 所控制的普通 Radix prefix cache，也不会关闭普通 chunked prefill；仅禁止 DeepSeek 长前缀的 chunked-MHA + LSE 拓扑。若删除 KV 分支的保护参数，或把 `triton_simo` 加入 chunked-prefix 支持白名单而不实现上述 reader/LSE 接口，都会重新暴露错误路径。

---

## `sglang serve` 如何把命令行参数变成 `ServerArgs` 成员

本节针对 SGLang `release/v0.5.18-local-dep`，源码提交 `982d8495b7`。代码引用统一采用“相对 code base 路径:行号，类::函数”的形式；字段、常量和模块级配置使用对应符号名。

### 结论先行

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

### 1. `sglang` 可执行文件先进入哪里

安装元数据在 `python/pyproject.toml:201-203，[project.scripts]` 声明：

```toml
sglang = "sglang.cli.main:main"
```

因此执行 `sglang serve ...` 时，生成的 console-script wrapper 只负责导入并调用 `python/sglang/cli/main.py:12-40，main`。

### 2. 顶层 parser 为什么不直接解析这个选项

`python/sglang/cli/main.py:12-21，main` 创建顶层 `argparse` 和 `serve` 子命令。`serve` 子命令没有注册全部 server 选项，并设置了 `add_help=False`。

`python/sglang/cli/main.py:35-40，main` 调用 `parser.parse_known_args()`。对于：

```bash
sglang serve --model-path /path/to/model --disable-chunked-prefix-cache
```

顶层 parser 只识别 `subcommand="serve"`，其余 server 选项留在 `extra_argv`，再传给 `python/sglang/cli/serve.py:166-205，serve`。

`serve` 在 `python/sglang/cli/serve.py:169-177，serve` 只处理 `--model-type` 和位置形式的 model path；`--disable-chunked-prefix-cache` 原样保留在 `dispatch_argv` 和 `ServeRequest.argv`。LLM backend 的 `run=_run_llm` 注册于 `python/sglang/cli/serve.py:129-135，_create_backend_registry`。

### 3. LLM backend 把 argv 交给真正的 server parser

`python/sglang/cli/serve.py:90-99，_run_llm` 执行：

```python
server_args = prepare_server_args(list(request.argv))
run_server(server_args)
```

也就是说，`sglang serve` 的参数解析分成两层：顶层 CLI 负责识别子命令，`prepare_server_args` 才负责识别 `--disable-chunked-prefix-cache`。

### 4. `prepare_server_args` 创建 parser、解析 argv

`python/sglang/srt/server_args.py:9770-9804，prepare_server_args` 的关键顺序是：

1. `:9781` 创建 `ArgumentParser(prog="sglang serve")`；
2. `:9782` 调用 `ServerArgs.add_cli_args(parser)`；
3. 有 `--config` 时，`:9784-9791` 通过 `ConfigArgumentMerger.merge_config_with_args` 合并 YAML 参数；
4. `:9793` 执行 `raw_args = parser.parse_args(argv)`；
5. `:9804` 调用 `ServerArgs.from_cli_args(raw_args)`。

所以 `raw_args` 是一个 `argparse.Namespace`，在这一步已经有 `raw_args.disable_chunked_prefix_cache` 属性。

### 5. `ServerArgs.add_cli_args` 使用 dataclass 反射

`python/sglang/srt/server_args.py:8658-8662，ServerArgs::add_cli_args` 没有为该选项单独写 `add_argument`，而是调用 `add_cli_args_from_dataclass(parser, ServerArgs)`。自动注册函数位于 `python/sglang/srt/arg_groups/arg_utils.py:218-337，add_cli_args_from_dataclass`，它读取类型注解和 dataclass 字段，逐个生成 argparse action。

### 6. 字段声明决定 CLI 名称和默认值

字段定义在 `python/sglang/srt/server_args.py:958-962，ServerArgs::disable_chunked_prefix_cache`：

```python
disable_chunked_prefix_cache: A[
    bool,
    "Disable chunked prefix cache feature for deepseek, ...",
    NS("schedule"),
] = False
```

`ServerArgs` 的说明在 `python/sglang/srt/server_args.py:447-470，ServerArgs` 指出：`A` 是 `typing.Annotated` 的别名，字段名会自动转换为 CLI 名称。这里的 `NS("schedule")` 只是运行时配置分组标记，不是 CLI 名称的一部分；它不会生成 `--schedule-disable...`。

### 7. 下划线如何变成连字符

`python/sglang/srt/arg_groups/arg_utils.py:208-210，_field_to_cli_name` 的实现是：

```python
return "--" + name.replace("_", "-")
```

所以 `disable_chunked_prefix_cache` 会变成 `--disable-chunked-prefix-cache`。在 `python/sglang/srt/arg_groups/arg_utils.py:231-249，add_cli_args_from_dataclass` 中，函数先取得 `Annotated` 元数据，再调用 `_field_to_cli_name(field.name)` 生成 `cli_name`，并把该名称交给 `parser.add_argument`。

`python/sglang/srt/arg_groups/arg_utils.py:147-163，_unwrap_annotated` 负责取出 `Annotated` 的内部类型和 metadata；字段中的裸帮助字符串会被转换成 `Arg(help=...)`，所以该字段具备 CLI 注册所需的 metadata。

### 8. 为什么 argparse 的 `dest` 恰好是下划线字段名

`python/sglang/srt/arg_groups/arg_utils.py:247-254，add_cli_args_from_dataclass` 还计算 `auto_dest`，并在字段名与自动生成的 dest 不同时才显式传入 `dest`：

```python
auto_dest = cli_name.lstrip("-").replace("-", "_")
dest_kwarg = {"dest": field.name} if field.name != auto_dest else {}
```

对本字段，`cli_name` 是 `--disable-chunked-prefix-cache`，`auto_dest` 和 `field.name` 都是 `disable_chunked_prefix_cache`。两者相等，所以不需要显式传 `dest`；argparse 的默认规则也会得到同名 dest。这就是连字符 CLI 名与下划线 Python 成员之间的直接连接。

### 9. `bool` 字段如何生成 `store_true`

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

### 10. `Namespace` 如何变成 `ServerArgs` 成员

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

### 11. 之后谁读取这个成员

`python/sglang/cli/serve.py:90-99，_run_llm` 将构造好的对象传给 `python/sglang/launch_server.py:16-53，run_server`，再进入 HTTP/gRPC launcher。后续 ModelRunner 初始化时，`python/sglang/srt/model_executor/model_runner.py:369-372，ModelRunner::__init__` 会读取/处理该配置，并调用 `python/sglang/srt/model_executor/model_runner_components/misc_utils.py:25-48，maybe_disable_chunked_prefix_cache` 做 MLA/backend 能力门控。

这一步可能通过 runtime context override 将生效配置视图改为禁用，但它与前面的 CLI 映射是两个阶段：

```text
argv -> Namespace -> ServerArgs 成员       （本节说明的映射）
                         |
                         -> ModelRunner/runtime gate 可能覆写生效值
```

### 12. 可复现的最小检查

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

## 13. 评测脚本中 CUDA graph、attention backend 与 EP 的修正

本节针对 SGLang `release/v0.5.18-local-dep`（源码提交 `982d8495b7`）和当前 SIMO 工作树。所有代码说明均使用相对 code base 路径、行号和函数名；SGLang 的 code base 根目录是 `/share/users/like/package/sglang_kernel_src`，SIMO 的 code base 根目录是 `/share/users/like/package/simo_conda_sglang`。

### 13.1 为什么之前加了 `disable_cuda_graph`

此前工作树中的
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:85-90，run_simo_config_list`
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

### 13.2 恢复 graph 参数时为什么使用新字段

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
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:85-88，run_simo_config_list`
和
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:105-114，run_no_quant_eval`。

这个参数只设置 decode graph 的最大捕获 batch size，不会关闭 graph。没有显式传 `cuda_graph_backend_decode` 时，
`python/sglang/srt/server_args.py:4519-4527，ServerArgs::_parse_cuda_graph_config`
会把它写入 decode phase 的 `max_bs`，backend 仍来自默认配置，通常是 `full`。如果是实际的 `sglang serve` 命令行，旧的 `--cuda-graph-max-bs 128` 仍可用但会给出 deprecated 语义；在本评测脚本的 Engine JSON 中应使用新字段。

评测日志若出现 `Disable prefill CUDA graph because the capture size is not set`，不应误认为 decode graph 被禁用。
`python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py:294-297，capture_prefill_graph`
在 prefill capture bucket 为空时只关闭 prefill runner；本脚本通过 lm-eval 使用 `chunked_prefill_size=-1` 的默认值时，出现这条日志是预期行为，decode phase 仍按 `cuda_graph_max_bs_decode` 捕获。

### 13.3 attention backend 只在 KV 量化时切换

`SIMOLinearMethod` 和 `SIMOFusedMoEMethod` 只替换权重加载/专家计算，不读取 SIMO 打包 KV cache。把 `triton` 强行传给所有权重量化评测，会改变原脚本的 backend 选择范围，也会让不涉及 KV 的测试承担额外的 Triton attention 约束。

当前
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:123-135，run_model_evaluations`
传入空字符串：

```bash
run_simo_config_list ... "online_quantization" "QUANT_CONFIGS" "" ""
```

`run_simo_config_list` 在
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:89-90，run_simo_config_list`
只有在 backend 非空时才把 `attention_backend` 加进 model args，因此权重量化测试完全交给 SGLang 默认解析。默认选择逻辑在
`python/sglang/srt/server_args.py:5870-5942，ServerArgs::_get_default_attn_backend`：MHA/MLA 会根据 GPU、模型结构和可用实现选择 `fa3`、`flashinfer` 或 `triton`，而不是由 SIMO 评测脚本硬编码。

KV 量化则必须走 SIMO backend。当前
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:138-147，run_model_evaluations_kv_cache_quant`
仍传入 `triton_simo`；对应的条件分支在
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:89-95，run_simo_config_list`：

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

### 13.4 为什么不再无条件限制 `SIMOFusedMoEMethod` 为 EP=1

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

### 13.5 本次修改与检查结果

已完成的代码修改：

- `simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:85-95，run_simo_config_list` 删除 `skip_server_warmup` 和 `disable_cuda_graph`，加入 `cuda_graph_max_bs_decode`，并只对 `triton_simo` 加 `disable_chunked_prefix_cache`；
- `simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:123-147，run_model_evaluations/run_model_evaluations_kv_cache_quant` 让权重量化使用默认 attention，KV 量化继续使用 `triton_simo`；
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
