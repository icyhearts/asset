## 1. Llama-3.1-8B-Instruct 裁剪模型（4-layer tiny model）相比原始模型做了哪些修改

**路径**:
- 原始: `/data_gpu/modelzoo/weights/llm/llama/llama3.1/llama3.1-8B-Instruct/safetensor_weights`（15G, 4 个 safetensors 分片）
- 裁剪后: `/share_data/inference-framework/tiny-models/Llama-3.1-8B-Instruct-4layer/safetensor_weights`（3.7G, 2 个 safetensors 分片）

**裁剪脚本**: `/share_data/inference-framework/tiny-models/trim/trim_llama31_8b.py`（说明文档见裁剪目录下的 `TRIM_INFO.md`）

### 修改内容（唯一实质修改：层数 32 → 4）

1. **`config.json`**: 仅 `num_hidden_layers: 32 → 4`，其余字段（hidden_size=4096、intermediate_size=14336、num_attention_heads=32、num_key_value_heads=8、rope_scaling、vocab_size 等）全部不变。

2. **权重张量**: 291 个 key → 39 个 key。层映射为 `{0:0, 1:1, 2:2, 3:3}`，即**保留前 4 层（layer 0-3）原封不动**，丢弃 layer 4-31。保留下来的张量 dtype/shape 与原始完全一致（逐 key 对比验证通过，无任何 dtype/shape 差异）：
   - `model.embed_tokens.weight`、`model.norm.weight`、`lm_head.weight` 全部原样保留；
   - `model.layers.{0..3}.*`（self_attn q/k/v/o proj、mlp gate/up/down proj、input/post_attention layernorm）原样保留；
   - 权重中无量化 scale（`weight_scale_inv` 后缀的张量不存在，两个模型都是纯 bf16）。

3. **tokenizer / generation 配置**: `tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`、`generation_config.json` 未修改。

即：这是一个纯粹的**深度截断**（depth truncation）冒烟测试模型——取前 4 层 + 完整 embedding/lm_head/final_norm，参数量约 8B → 1.9B 左右，用于快速验证推理流程，输出质量不做保证。

---

## 2. `layer_map={0: 0, 1: 1, 2: 2, 3: 3}` 参数的功能

**一句话**：`layer_map` 是「**目标层号 → 源层号**」的映射（`dst_idx -> src_idx`，见 `smoke_trim.py` 的 `SmokeSpec.layer_map` 字段注释），它同时决定两件事：

1. **保留哪些源层**（筛选）：源层号只要没出现在 `layer_map.values()` 里，该层的所有权重张量都会被丢弃（`should_keep()`）。
2. **把保留的层放到裁剪后模型的第几层**（重编号）：源键名 `model.layers.{src}.` 会被改写成 `model.layers.{dst}.`（`remap_layer_key()`）。

### 对 Llama-3.1-8B 的实际效果（32 层 → 4 层，纯深度截断）

```
Src: llama3.1-8B (32层)                     Dst: Llama-3.1-8B-Instruct-4layer (4层)
─────────────────────────────────           ─────────────────────────────────
  layer 0  ──(keep, dst←0)─────────────►   layer 0
  layer 1  ──(keep, dst←1)─────────────►   layer 1
  layer 2  ──(keep, dst←2)─────────────►   layer 2
  layer 3  ──(keep, dst←3)─────────────►   layer 3
  layer 4  ──✗ drop ──────────────────┐
  layer 5  ──✗ drop                   │   (只有这 4 层被保留，
    ...                                 │    其余 28 层全部丢弃)
  layer 31 ──✗ drop ──────────────────┘
```

- 因 `layer_map.values() = {0,1,2,3}`，源层 4~31 的张量全部被 `should_keep()` 过滤掉。
- 因每个 `dst_i == src_i`（恒等映射），保留层的键名里 `layers.{src}.` 重编号后仍是 `layers.{dst}.`，名字不变。
- 结果就是「保留前 4 层 + 完整 embedding / final_norm / lm_head」，`config.json` 同步 `num_hidden_layers: 32 → 4`。

### 一般情形（非恒等映射）

`layer_map` 的**键**是目标层号，**值**是来源层号；所以它不只是「截前 N 层」，还可以**跳过 / 重排 / 跳转**。仓库里大量脚本正是这么用的：

```
layer_map = {dst: src}                       语义
─────────────────────────────────────        ───────────────────────────────
  {0:0, 1:1, 2:2, 3:3}   llama31_8b          顺序保留前4层（恒等）
  {0:0, 1:3}               deepseek_v3/glm47 保源层0和3 → 目标层0和1（跳1层）
  {0:0, 1:2, 2:3}         deepseek_v4_flash  保源层0,2,3 → 目标0,1,2（跳过源层1）
  {0:0, 1:1, 2:6, 3:7}    ling26_flash       保源层0,1,6,7 → 目标0,1,2,3
```

以 `{0:0, 1:3}`（DeepSeek-V3 / GLM-4.7 等）画图：

```
# layer_map = {0:0, 1:3}   →  只保留源层0和源层3
Src (深, 示意标出相关层)                Dst (2层)
layer 0 ──(dst←0)────────────────►  layer 0
layer 1 ──✗ drop
layer 2 ──✗ drop                    (注：dst←3 表示把源层3放到第1个位置)
layer 3 ──(dst←1)────────────────►  layer 1
```

### 为什么需要非恒等映射

- **跳过不需要的层**：例如把 dense 层与 MoE/混合层交错时，只挑「类型合适」的源层放到对应位置。`smoke_trim.py` 中 `layer_group_size` / `layer_types` 相关注释强调：混合模型里某目标槽位需要 linear/full-attn 的特定权重布局，`layer_map` 选取的源层类型必须与之匹配。
- **跨边界重编号**：像 DeepSeek 这类模型 `first_k_dense_replace` 之后是 MoE 层，取子集时层号会错位，用 `layer_map` 把稀疏的源层号对齐回连续的 `0..dst_layers-1`（`build_layer_map()` 就是为此生成此类映射）。
- 本质上 `layer_map` 是一个「**裁剪器 + 重排器**」：决定砍谁、留谁、留给它哪个槽位。

---

## 3. `/share/users/like` 空间占用排查（2026-09-10）

**背景**：`/share`（NFS `10.97.128.245:/share`）78T 已用 77T，仅剩 **1.1T（99%）**。本次排查对象 `/share/users/like`，`du -x -sh`（同文件系统，不计子挂载点）总计 **565G**。

**一句话结论**：占空间的大头不是数据集或模型权重（这两类加起来不到 10G），而是**重复的 conda 环境 / 构建产物 / 缓存 / 虚拟机磁盘镜像**四类。其中 `.cache`(39G)、`qemu_demo` 的 qcow2 镜像(96G)、`bench-io/miniconda3`(26G，与主 miniconda3 完全重复)、`/share/users/like/package` 下 6 份历史 `build-src-*`(38.8G) 和 ONNX Runtime 的 Debug 构建目录(22G) 属于**几乎零风险**的可回收项，按 3.9 的清单可安全回收 **约 260G**。

### 3.1 顶层目录占用总览

| 目录 | 大小 | 性质 |
|---|---|---|
| `package/` | 252G | 源码仓库 + 构建产物 + 打包 tar（最大头） |
| `miniconda3/` | 99G | conda 环境（envs 96G） |
| `qemu_demo/` | 96G | qcow2 虚拟机磁盘镜像 |
| `.cache/` | 39G | 各类工具缓存（pip 24G） |
| `bench-io/` | 26G | 另一套完整 miniconda3（重复） |
| `opt/` | 23G | 3 套 CUDA 安装（12.8 / 13.0 / 13.1） |
| `docker-image/` | 5.5G | 单个镜像 tar |
| `sipu_sdk_debug/` | 4.1G | sdk_rel.tar.bz2 |
| `huggingface_cache/` | 3.5G | **数据集** |
| `temp/` | 2.6G | 临时文件 |
| `build/` | 2.2G | onnxruntime-v1.27.0-cuda13 构建目录 |
| `go-install/` | 1.2G | Go 工具链 |
| `vim-docker-compile-22.04/` | 1.1G |  |
| `wheel/` | 1018M |  |
| `shareVR/` | 530M |  |
| `vim-go/` | 467M |  |
| `arch/`、`bash-bin/`、`.bash_history_sessions/`、`private/`、`.ac_ssh_config/`、`.conda/`、`cron-jobs/`、`.pip/`、`from-h100/` | 均 < 100M |  |

顶层散落的大文件（不在任何子目录里，容易漏掉）：

| 文件 | 大小 |
|---|---|
| `.cache.tar` | 9.6G |
| `vim-go.tar` | 740M |
| `other.tar.bz2` | 429M |
| `ac.tar` | 140K |

### 3.2 缓存类目录（cache）

**`/share/users/like/.cache` — 39G**

| 子目录 | 大小 | 说明 |
|---|---|---|
| `.cache/pip` | 24G | pip HTTP 缓存（`http-v2` 24G + `wheels` 50M），可 `pip cache purge` |
| `.cache/vllm` | 12G | `torch_compile_cache` 11G（其中 `torch_aot_compile` 5.7G） |
| `.cache/sglang` | 1008M | |
| `.cache/huggingface` | 756M | `hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered` 645M、`datasets--cais--mmlu` 101M |
| `.cache/flashinfer` | 554M | |
| `.cache/jedi` | 373M | vim/YouCompleteMe 的 Python 补全缓存 |
| `.cache/pre-commit` | 224M | |
| `.cache/go-build` | 177M | |
| 其余（tvm-ffi 63M、deep_gemm 32M、clangd 23M、virtualenv 22M、tracker3 17M、sgl_eval 15M 等） | < 100M | |

**另有一份 2025-12-09 的整目录备份 `.cache.tar` = 9.6G**，内容与 `.cache` 高度重叠，属于纯冗余。

### 3.3 临时 / 中间产物目录（temp、tmp、build）

| 目录 | 大小 | 说明 |
|---|---|---|
| `package/sglang_kernel_src/temp` | 8.0G | 内核开发时的 `.safetensors` 测试数据：`prepare_data_sgl_decode_attention_fwd.*.safetensors` 1.8G × 2、`extend_forward_triton_input.safetensors` 1.2G、`decode_forward_stage1_triton_input.safetensors` 1.2G | done
| `temp/` | 2.6G | `temp/sgl-unzip` 1.6G（解压出来的 sgl_kernel/sglang）+ `temp/mydd` 23M 等 | done
| `package/onnxruntime/build` | 22G | `build/onnxruntime-v1.27.0-cuda13`（**Debug 构建**，含 `libonnxruntime_providers.a` 1.3G） | done
| `build/` | 2.2G | 同为 onnxruntime-v1.27.0-cuda13 | done
| `package/h100/package/old` | 4.0G | 4 个历史 build：`build-rtx-5060ti` 1.1G、`build-5060ti-v2` 1.1G、`build-5060ti` 1.1G、`build-rtx-5060ti-v2` 935M | done
| `package/onnxruntime/temp` | 34M | | done
| `package/siorigin-container-toolkit/temp` | 160K | |

### 3.4 数据集目录

全部数据集加起来只有约 7G，**不是空间问题的主因**。

| 目录 / 文件 | 大小 |
|---|---|
| `huggingface_cache/other.tar` | 1.5G |
| `huggingface_cache/mit-han-lab___pile-val-backup` | 1.3G |
| `huggingface_cache/cais___mmlu` | 326M |
| `huggingface_cache/NeelNanda___pile-10k` | 212M |
| `huggingface_cache/wikitext` | 66M |
| `huggingface_cache/EleutherAI___wikitext_document_level` | 13M |
| `huggingface_cache/TIGER-Lab___mmlu-pro` | 8.7M |
| `huggingface_cache/{openai___gsm8k,gsm8k}` | 4.6M × 2 |
| `.cache/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered` | 645M |
| `.cache/huggingface/hub/datasets--cais--mmlu` | 101M |
| `package/jdjv/kokoro_clean{,-old}/data/seedtts_testset.tar` | 1.2G × 2（两份内容相同） |
| **合计** | **≈ 7G** |

### 3.5 模型权重

**该目录下没有大模型权重**（llama3.1 之类都在 `/data_gpu/modelzoo`，不在 `/share/users/like`）。唯一与权重沾边的是内核开发留下的测试用 safetensors（见 3.3 的 `package/sglang_kernel_src/temp`，约 6G），可随 temp 一起清。

### 3.6 conda 环境 —— 最大的单一可回收项（≈ 145G）

| 位置 | 大小 | 明细 |
|---|---|---|
| `miniconda3/envs` | **96G** | `simo_sglang_pip` 17G、`simo_sglang` 17G、`simo_vllm` 13G、`simo_vllm_pip` 11G、`vllm_new` 9.8G、`vllm_src_5060` 9.0G、`vllm_src` 9.0G、`vllm_dev` 6.2G、`vllm_dev_2` 4.0G、`simo_sglang_5060` 232M | done: 5060
| `miniconda3/pkgs` | 2.3G | conda 包缓存 |
| `bench-io/miniconda3` | **26G** | 一整套独立的 miniconda：`envs/simo_sglang` 14G + `envs/simo_vllm` 11G + `pkgs` 1.1G，与上面**高度重复** | done
| `package/jdjv/kws_simo_quant/.venv-*` | 22.4G | 三个并列 venv：`.venv-simo-fixed` 7.7G、`.venv-simo-wheel` 7.5G、`.venv-simo` 7.2G | done

注意 `miniconda3` 与 `bench-io/miniconda3` 是两套完全独立的安装，同名环境各一份；`_pip` 后缀的环境与对应非 `_pip` 环境也都是重复的。

### 3.7 构建产物与打包文件

| 目录 / 文件 | 大小 | 说明 |
|---|---|---|
| `package/vllm-for-conda-simo/build-src-bjh-*`（6 个） | **38.8G** | v0.11.2 7.0G、v0.13.0 7.2G、v0.15.0 7.2G、v0.17.0 6.3G、v0.20.0 5.8G、v0.27.1 5.3G；每个都含一份 1.2~1.3G 的 `_vllm_fa3_C.abi3.so` |
| `package/vllm-for-conda-simo/.deps` + `.deps-back` | 4.4G | 依赖预编译产物 |
| `package/h100/package/other/vllm` | 31G | 同上，含 2.3G / 2.2G / 1.2G 三个 `_vllm_fa3_C.abi3.so` |
| `package/dev-ubuntu-24.04.tar` | **15.6G** | 基础镜像 tar |
| `package/siorigin-container-toolkit/sipu_sdk_rel.tar` | **13.4G** | |
| `package/cuda/*.run`（5 个） | **22G** | cuda 12.8.0 5.0G、12.8.1 5.0G、13.0.2 4.0G、13.1.2 4.0G、13.0.0 4.0G —— 安装包，装完即可删 |
| `opt/cuda-{12.8,13.0,13.1}` | 23G | 已安装的 3 套 CUDA：8.8G + 7.2G + 7.1G，若只用一套可删另外两套 |
| `package/h100/package.h100.tar.bz2` | 3.8G | |
| `sipu_sdk_debug/sipu_sdk_rel.tar.bz2` | 4.0G | 与 13.4G 的 `.tar` 内容重复 |
| `package/sipu_sw.tar` | 2.8G | |
| `package/rocm-terminal.tar` | 2.8G | | done
| `package/h100/package/{cutlass 5.4G, TensorRT-LLM 1.8G, ARM-software 960M, cutlass-v3 662M, triton-lang 631M, nccl 577M, cmake 437M}` | ~10G | 第三方源码/依赖 |
| `.git/objects/pack` 大包 | 3.1G | TensorRT-LLM 1.6G + onnxruntime 1.5G |

### 3.8 虚拟机 / 容器镜像（qcow2、docker tar）

| 文件 | 大小 | 说明 |
|---|---|---|
| `qemu_demo/yubo/ctk.copy2.qcow2` | 29.6G | 2025-11-24 |
| `qemu_demo/yubo/ctk.copy.qcow2` | 16.1G | 2025-10-23 |
| `qemu_demo/yubo/ctk.qcow2` | 12.8G | 2025-10-20 |
| `qemu_demo/ctk.copy2.qcow2` | 14.4G | 2025-10-20 |
| `qemu_demo/yubo_scp/ctk.qcow2` | 12.8G | |
| `qemu_demo/ci/CI.qcow2` | 9.2G | 2025-09-08 |
| `docker-image/dcsmbuild-x86_64-like-v1.tar` | 5.5G | 2025-10-14 |
| **`qemu_demo/` 合计** | **96G** | 其中 `ctk*` 系列是同源镜像的多个副本，`yubo/ctk.copy2.qcow2`(29.6G) 与 `yubo/ctk.copy.qcow2`(16.1G)、`yubo/ctk.qcow2`(12.8G) 极可能是同一份镜像的连续快照 |

### 3.9 建议清理优先级

**第一梯队 —— 纯缓存，删了会自动重建（≈ 87G）** # done

| 项目 | 可回收 |
|---|---|
| `rm -rf ~/.cache/pip` 或 `pip cache purge` | 24G |
| `rm -f ~/.cache.tar` | 9.6G |
| `rm -rf ~/.cache/vllm/torch_compile_cache` | 11G |
| `rm -rf ~/bench-io/miniconda3`（整份重复的 conda） | 26G |
| `rm -rf ~/package/sglang_kernel_src/temp/*` | 8.0G |
| `rm -rf ~/.cache/{sglang,flashinfer,jedi,pre-commit,go-build}` | ~2.3G |
| `rm -rf ~/build ~/temp` | 4.8G |
| `rm -f ~/vim-go.tar ~/other.tar.bz2` | 1.2G |
| **小计** | **≈ 87G** |

**第二梯队 —— 历史构建产物 / 重复打包文件，确认无回滚需求即可删（≈ 172G）**

| 项目 | 可回收 |
|---|---|
| `package/vllm-for-conda-simo/build-src-bjh-*`（保留当前在用的 v0.27.1，删其余 5 个） | ≈ 33G | # d  one
| `package/onnxruntime/build`（Debug 构建） | 22G | # done
| `qemu_demo/` 中的旧快照：`yubo/ctk.copy.qcow2` 16.1G + `yubo/ctk.qcow2` 12.8G + `yubo_scp/ctk.qcow2` 12.8G + 根目录 `ctk.copy2.qcow2` 14.4G（保留 `yubo/ctk.copy2.qcow2` 与 `ci/CI.qcow2`） | ≈ 56G | # done
| `package/cuda/*.run`（5 个安装包，安装已完成） | 22G | # done
| `package/dev-ubuntu-24.04.tar` | 15.6G | done
| `sipu_sdk_debug/*.tar.bz2` 4.0G + `package/.../sipu_sdk_rel.tar` 13.4G（与已解压的 `sipu_sdk_rel/` 目录 14G 重复，tar 与目录只需留一份） | 17.4G | # done
| `package/h100/package/old`（4 个历史 build） | 4.0G | done
| **小计** | **≈ 172G** |

**第三梯队 —— 需要人工确认的（conda/venv 冗余，≈ 80G）**

- `miniconda3/envs` 中的 `vllm_new`(9.8G)、`vllm_src_5060`(9.0G)、`vllm_src`(9.0G)、`vllm_dev`(6.2G)、`vllm_dev_2`(4.0G) —— 5 个 `vllm_*` 环境疑似同一用途的不同时间点副本，保留最新一个即可回收约 38G。
- `simo_sglang_pip`(17G) / `simo_vllm_pip`(11G) 与 `simo_sglang`(17G) / `simo_vllm`(13G) 成对，`_pip` 版疑似安装方式不同的副本，可回收约 28G。
- `package/jdjv/kws_simo_quant/.venv-simo{,-fixed,-wheel}` 三个 venv 共 22.4G，保留一个即可回收约 15G。 done
- `package/jdjv/kokoro_clean-old/`（1.6G，另一份 `kokoro_clean` 已存在）。 done

**附带发现**：`/share` 整体 99% 占用，但 `/share/users/like` 只占 565G —— 说明绝大部分空间是**其他用户**占用的，本目录的清理能腾出约 200~250G，对整盘缓解有限，建议同步推动其他用户排查。

---

## 4. 用 `mydd.cpp` truncate `/share/liwang/VLM_projects/.snapshot` 能真正释放空间吗？

**结论：不能。** 有两层独立原因，任一层都足以否定这个做法。

### 4.1 先明确 `mydd.cpp` 做什么

`/softhome/like/asset/code/cpp_guard/mydd.cpp`（187 行）是一个「递归截断」工具：

- `main`（`mydd.cpp:164`）接收一个参数 `max_file_size_mib`；`parse_options`（`:40`）把 `0` 解释为**无条件截断**（`truncate_unconditionally = true`，`:56`）。
- `scan_current_directory`（`:126`）用 `fs::recursive_directory_iterator` 递归遍历**当前目录**（`. `，`:129`），`skip_permission_denied`。
- 对每个普通文件调 `truncate_file`（`:79`）：先 `lstat`（`:85`），非普通文件跳过（`:90`）；满足阈值时用 `open(path, O_WRONLY | O_TRUNC | O_CLOEXEC | O_NOFOLLOW)`（`:102-107`）把文件**截断为 0 字节**。

即：它的作用范围就是**当前工作目录**，把其中的普通文件清零。

### 4.2 原因一：`.snapshot` 是存储端只读视图，`O_TRUNC` 会直接失败

只读检查（未做任何写入）：

```
W_OK=False   /share/liwang/VLM_projects/.snapshot
W_OK=False   /share/liwang/VLM_projects/.snapshot/2-hourly.2026-09-10_1015
W_OK=True    /share/liwang/VLM_projects
```

`.snapshot` 目录的权限位是 `drwxrwxrwx`（777），但对它做**写权限判定返回 False** —— 说明拒绝来自 **NAS 服务端**，而不是本地权限位。NetApp 的 `.snapshot` 命名空间按设计就是只读的。

因此在 `.snapshot` 里跑 mydd，每个文件都会走到 `open(O_TRUNC)` 失败分支（`:108-112`），刷出一堆

```
[WARNING] open(O_TRUNC) '...' failed: Read-only file system
```

最终输出 `truncated 0 file(s)` —— **一个字节都不会变**。

### 4.3 原因二：即使能写，也释放不了空间（概念性错误）

真正"钉住"那 ~5T 的是**存储端的快照对象**（ONTAP snapshot），它持有被删数据块的引用。

- 快照是**只读的 COW 时点镜像**，不是一块你可以进去"清理"的磁盘区域。
- 要释放空间，必须让**快照本身消失**（删除快照对象，或等保留策略轮转过期），而不是透过 `.snapshot` 去改文件。
- 退一步，即便某实现允许"写入"快照视图，写入也只会走 COW **新分配块**，而快照仍引用旧块 —— 空间不降反升。

一句话：**要删的是「快照」这个对象，不是「快照里的文件」。**

### 4.4 正确做法

1. 找**存储管理员删除相关快照**（最老的是 `hourly.2026-07-30_1605`、`hourly.2026-07-30_1705`、`weekly.2026-09-06_0015`）。
2. 或**等待保留策略**（2-hourly / daily / weekly）自然轮转。
3. 快照释放后 `df` 才会降；在此之前重复 `rm` 或 truncate **都无效**。

### 4.5 重要风险提示（比"无效"更严重）

- `mydd` 传 `0` 是**无条件截断**；而 `recursive_directory_iterator`（`:128`）默认**会进入 `.snapshot`**，同时也会递归进当前目录下所有子目录。
- 所以：
  - 在 `.snapshot` 里跑：只会满屏 EROFS 警告，**无害但完全无效**。
  - **在 live 目录（例如 `/share/liwang/VLM_projects` 或任何真实数据目录）里跑：会把该目录下所有普通文件全部截成 0 字节**，这是**不可逆的数据破坏**。切勿在生产数据目录执行。
- 本次**未执行** `mydd`（按你的要求：只写答案，未对其他任何文件做写操作）。

### 4.6 小结

| 做法 | 能释放空间吗 | 原因 |
|---|---|---|
| 用 mydd truncate `.snapshot` | ❌ 不能 | 快照只读，`open(O_WRONLY\|O_TRUNC)` 被服务端拒绝（EROFS） |
| 假设"能写"快照 | ❌ 不能 | 快照是只读 COW 镜像，空间由**快照对象**持有；写只会新分配块 |
| 请管理员删除快照本身 | ✅ 能 | 解除对已删数据块的引用 |
| 等保留策略轮转过期 | ✅ 能 | 同上 |
| 在 live 数据目录跑 mydd | ❌ **灾难** | 会把真实数据全部清零，不可逆 |

**核心结论：释放空间要靠「删快照」，不是「truncate 快照里的文件」；后者既写不进去，写进去也没用。**
