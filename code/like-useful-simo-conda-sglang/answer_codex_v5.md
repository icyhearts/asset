# `sikernel/setup.sh` 无命令行参数时的 `SI_CMODEL_ROOT` 设置过程

本文把 `/share/users/like/package/sikernel` 作为 code base。代码引用统一写成“相对 code base 路径:行号（函数名）”；没有函数的语句标为 `<top-level>`。`sipu_cmodel_setup.sh` 属于外部 SDK，不在该 code base 内，因此用其绝对路径标注。

## 1. 直接结论

在以下前提下：

- 调用方式是 `source /share/users/like/package/sikernel/setup.sh`；
- 没有传任何位置参数；
- 调用前没有设置 `SIPU_ARCH`；
- 没有设置 `SIKERNEL_SIPU_CMODEL_SETUP` 覆盖 CModel 配置脚本；

最终结果是：

```text
SIPU_ARCH=150
SI_CMODEL_HW_ARCH=1.5
SI_CMODEL_ROOT=$(readlink -f /share_data/sicx_sdk/release/latest/sipu1.5_cmodel)
```

当前机器上，符号链接解析后的实际值为：

```text
SI_CMODEL_ROOT=/share_data/arch_cmodel_release/sipu1.5/2609040400
```

这是一次实测结果，不应把最后的时间戳目录当成永久常量：当前链接为
`/share_data/sicx_sdk/release/latest/sipu1.5_cmodel -> /share_data/arch_cmodel_release/sipu1.5/2609040400`。

关键点是：`setup.sh` 自身不直接给 `SI_CMODEL_ROOT` 赋值；它通过 `set_sdk.sh` source 外部的 CModel setup 脚本，后者才完成真正的赋值。

## 2. 无参数时的调用链

### 2.1 定位 `setup.sh` 所在目录

`setup.sh:18-26（_sikernel_setup_source_path）` 根据 shell 类型返回当前被 source 的脚本路径。随后 `setup.sh:28-32（<top-level>）` 对该路径执行 `dirname`、`cd` 和 `pwd -P`，得到物理路径形式的 `_sikernel_setup_dir`。这一步只决定后续从哪里加载 `set_src_dir.sh` 和 `set_sdk.sh`，尚未设置 `SI_CMODEL_ROOT`。

### 2.2 处理空的第一个参数

`setup.sh:34-55（<top-level>）` 先执行 `_sikernel_setup_sipu_arch="${1:-}"`。没有命令行参数时它是空字符串，随后 `case` 的空分支 `setup.sh:39-41（<top-level>）` 什么也不做：

- 不会在这里设置 `SIPU_ARCH`；
- 也不会清除调用环境中原来已有的 `SIPU_ARCH`。

因此，“没有 CLI 参数”与“`SIPU_ARCH` 一定为空”不是同一个条件。默认 1.5 还要看调用前环境中是否已有 `SIPU_ARCH`。

### 2.3 加载两个辅助脚本

`setup.sh:57-61（<top-level>）` source `set_src_dir.sh`。该脚本的 `set_src_dir.sh:18-28（<top-level>）` 通过 `BASH_SOURCE[0]` 和 `pwd -P` 导出 `SIKERNEL_ROOT_DIR`；它不设置 `SI_CMODEL_ROOT`。

接着 `setup.sh:63-67（<top-level>）` source `set_sdk.sh`。`SI_CMODEL_ROOT` 的选择和赋值都发生在这条分支后面。

## 3. `set_sdk.sh` 如何选择 CModel setup 脚本

### 3.1 固定 SDK 根目录

`set_sdk.sh:23-24（<top-level>）` 把 SDK setup 脚本固定为：

```text
_sikernel_sdk_setup=/share_data/sicx_sdk/release/latest/sipu_sdk_setup.sh
_sikernel_sdk_root=/share_data/sicx_sdk/release/latest
```

`set_sdk.sh:67-78（_sikernel_source_sdk）`，以及其调用和错误检查 `set_sdk.sh:98-102（<top-level>）`，会先 source 这个 SDK setup；它主要设置 `SI_SDK_ROOT`、`PATH`、`LD_LIBRARY_PATH` 和 CMake 路径，不直接设置 `SI_CMODEL_ROOT`。

### 3.2 从 `SIPU_ARCH` 得到 CModel 版本

`set_sdk.sh:52-58（<top-level>）` 使用：

```bash
_sikernel_requested_cmodel_version="${SIPU_ARCH:-1.5}"
```

所以在“无参数且 `SIPU_ARCH` 未预设”时，requested version 为 `1.5`。随后 `set_sdk.sh:26-50（_sikernel_normalize_cmodel_version）` 将以下写法归一化：

| 输入 | 归一化版本 |
|---|---|
| 空、`1.5`、`150` | `1.5` |
| `1.6`、`160` | `1.6` |
| `1.7`、`170` | `1.7` |

`set_sdk.sh:60-64（<top-level>）` 再把归一化版本转换成导出的整数形式；默认情况下执行 `export SIPU_ARCH="150"`。

### 3.3 计算并加载 CModel setup 路径

`set_sdk.sh:65（<top-level>）` 选择：

```bash
_sikernel_cmodel_setup="${SIKERNEL_SIPU_CMODEL_SETUP:-${_sikernel_sdk_root}/sipu${_sikernel_cmodel_version}_cmodel/sipu_cmodel_setup.sh}"
```

默认版本 1.5 时，逻辑路径是：

```text
/share_data/sicx_sdk/release/latest/sipu1.5_cmodel/sipu_cmodel_setup.sh
```

`SIKERNEL_SIPU_CMODEL_SETUP` 非空时会优先使用它，默认路径不再生效。`set_sdk.sh:80-96（_sikernel_source_cmodel）` 先检查该文件存在，再在 `set_sdk.sh:104-108（<top-level>）` 调用该函数并传播错误。

## 4. `SI_CMODEL_ROOT` 的真正赋值位置

默认路径对应的外部文件是：

```text
/share_data/sicx_sdk/release/latest/sipu1.5_cmodel/sipu_cmodel_setup.sh
```

该文件不属于 `/share/users/like/package/sikernel` code base。其关键代码如下：

### 4.1 计算物理 CModel 目录

`/share_data/sicx_sdk/release/latest/sipu1.5_cmodel/sipu_cmodel_setup.sh:14-21（sipu_cmodel_get_loc）` 执行：

1. 从 `BASH_SOURCE[0]` 取被 source 的 setup 脚本路径；
2. 用 `dirname` 取其目录；
3. 用 `readlink -f` 解析所有符号链接；
4. 在第 19 行执行 `SI_CMODEL_ROOT="${cmodel_loc}"`；
5. 在第 20 行把 `${cmodel_loc}/lib` 放到 `LD_LIBRARY_PATH` 前面。

因此最终变量不是 setup 脚本的逻辑路径，而是 CModel 目录的 canonical（物理）路径。

### 4.2 导出变量

`/share_data/sicx_sdk/release/latest/sipu1.5_cmodel/sipu_cmodel_setup.sh:29-38（sipu_cmodel_setup_env）` 在第 30 行执行 `export SI_CMODEL_ROOT="${SI_CMODEL_ROOT}"`，在第 31 行执行 `export SI_CMODEL_HW_ARCH="1.5"`（该文件对 1.5 版本写死）。该版本脚本的顶层代码 `:40-42（<top-level>）` 依次调用 `sipu_cmodel_get_loc`、`sipu_cmodel_banner` 和 `sipu_cmodel_setup_env`。

所以赋值时序可以压缩为：

```text
setup.sh
  -> source set_sdk.sh
     -> 选择 sipu1.5_cmodel/sipu_cmodel_setup.sh
     -> source 外部 CModel setup
        -> sipu_cmodel_get_loc: SI_CMODEL_ROOT=$(readlink -f(dirname(BASH_SOURCE[0])))
        -> sipu_cmodel_setup_env: export SI_CMODEL_ROOT
```

## 5. 不同调用环境下的结果

| 调用前条件 | 选择的 CModel | 最终 `SI_CMODEL_ROOT` |
|---|---|---|
| 无参数，`SIPU_ARCH` 未设置 | `sipu1.5_cmodel` | `readlink -f` 后的 1.5 CModel 目录 |
| 无参数，`SIPU_ARCH=160` | `sipu1.6_cmodel` | `readlink -f` 后的 1.6 CModel 目录 |
| 无参数，`SIPU_ARCH=1.7` | `sipu1.7_cmodel` | `readlink -f` 后的 1.7 CModel 目录 |
| 任意参数状态，`SIKERNEL_SIPU_CMODEL_SETUP=/path/custom.sh` | 自定义脚本 | 由自定义脚本决定；若遵循 SDK 约定，则是该脚本所在目录的物理路径 |

还有两个容易误判的行为：

1. 如果调用前已经有 `SI_CMODEL_ROOT`，外部脚本的 `sipu_cmodel_get_loc` 仍会在第 19 行无条件覆盖它，不会把旧值作为优先级更高的配置。
2. `source setup.sh` 才会把 `export` 的结果留在当前 shell；若写成 `bash setup.sh` 或直接执行 `./setup.sh`，变量只存在脚本子进程，脚本结束后不会回写父 shell。

## 6. 验证方式与当前结果

可以在不污染当前终端的子 shell 中验证默认路径：

```bash
env -u SIPU_ARCH -u SIKERNEL_SIPU_CMODEL_SETUP -u SI_CMODEL_ROOT \
  bash -c '
    source /share/users/like/package/sikernel/setup.sh >/dev/null &&
    printf "SIPU_ARCH=%s\\nSI_CMODEL_HW_ARCH=%s\\nSI_CMODEL_ROOT=%s\\n" \
      "$SIPU_ARCH" "$SI_CMODEL_HW_ARCH" "$SI_CMODEL_ROOT"
  '
```

在当前环境得到：

```text
SIPU_ARCH=150
SI_CMODEL_HW_ARCH=1.5
SI_CMODEL_ROOT=/share_data/arch_cmodel_release/sipu1.5/2609040400
```

结论因此是：**无命令行参数时，`setup.sh` 先保留或默认决定 `SIPU_ARCH`；在干净环境中默认选择 1.5，然后 source 对应的 CModel setup，由 `sipu_cmodel_get_loc` 将 `SI_CMODEL_ROOT` 设置为 `readlink -f` 后的 `sipu1.5_cmodel` 目录，并由 `sipu_cmodel_setup_env` 导出。**

## 7. 让 Vim 高亮 SiPU 的 `clusterDim` 和 `clusterIdx`

### 7.1 当前配置和现象

本节的配置 code base 是 `/data/like/vim-port-all/config`，Vim runtime code base 是
`/data/like/vim-port-all/binary/vim-install-ubuntu22.04/share/vim/vim91`，SiRT
code base 是 `/share/users/like/package/sirt`。

当前 Vim 是 Vim 9.1（patch 1-1357）。配置文件有两个重要行为：

- `.vimrc:5-10（<top-level>）` 计算配置目录，并把
  `/data/like/vim-port-all/config/.vim` 放到 `runtimepath` 首位，把
  `/data/like/vim-port-all/config/.vim/after` 放到末位；
- `.vimrc:90-91（<top-level>）` 开启 syntax 和 filetype plugin/indent。

当前用户 filetype 规则 `.vim/filetype.vim:1（BufNewFile/BufRead 顶层 autocmd）` 只
处理 `*.cl`。Vim 内置规则 `filetype.vim:560-561（CUDA 顶层 autocmd）` 只把
`*.cu,*.cuh` 识别为 `cuda`，没有 `*.su` 规则。因此打开
`test/cuda/rt/04_execution/kernels.su:25-35（<top-level>）` 时，当前实测：

```text
:set filetype?  -> filetype=
:set syntax?    -> syntax=
```

目标词出现在 `kernels.su:27（<top-level>）`。所以需要先解决文件类型识别，再增加
语法词表。

### 7.2 Vim 的加载链为什么需要两步

`syntax/syntax.vim:19-35（<top-level>）` 加载 syntax 支持，并建立
`FileType` 到 `syntax` 的自动设置；`syntax/syntax.vim:42-44（<top-level>）` 还会
对已有 buffer 重新触发 filetype 检测。

当文件类型变成 `cuda` 后，`syntax/synload.vim:34-63（s:SynSet）` 在第 59 行执行
`runtime! syntax/cuda.vim`。CUDA 语法文件
`syntax/cuda.vim:7-13（<top-level>）` 先加载 C++ 语法，
`syntax/cuda.vim:39（<top-level>）` 定义：

```vim
syn keyword cudaVariable gridDim blockIdx blockDim threadIdx warpSize
```

`syntax/cuda.vim:46-51（<top-level>）` 把 `cudaVariable` 链接到通用的
`Identifier` 高亮组。因此 `clusterDim` 和 `clusterIdx` 最适合加入同一个
`cudaVariable` 组，视觉效果会和 `blockDim`、`threadIdx` 一致。

### 7.3 推荐的最小持久化方案：复用 `cuda` filetype

如果项目约定所有相关的 `.su` 文件都是 SiPU CUDA-like C++，建议增加两个用户 runtime
文件（本次没有创建）：

**文件类型检测：** `/data/like/vim-port-all/config/.vim/ftdetect/sipu.vim`

```vim
" SiPU kernels use CUDA-like C++ syntax.
au BufRead,BufNewFile *.su setfiletype cuda
```

Vim 内置 `filetype.vim:3357-3359（<top-level>）` 会在默认检测规则之后执行
`runtime! ftdetect/*.vim`，所以该文件会被自动加载。`setfiletype` 的含义是：只有在
尚未确定 filetype 时才设置，避免无意覆盖别的检测器。按照 Vim 的 ftdetect 约定，文件
中不需要再包一层 `augroup`；该文件是在 `filetypedetect` 组中加载的。

**增加两个 CUDA 变量：** `/data/like/vim-port-all/config/.vim/after/syntax/cuda.vim`

```vim
" SiPU-specific CUDA-like built-in variables.
syntax keyword cudaVariable clusterDim clusterIdx
```

`.vimrc:10（<top-level>）` 已经把 `.vim/after` 放进 `runtimepath`，因此这个文件会
在发行版的 `syntax/cuda.vim` 之后执行。这样：

```text
打开 *.su
  -> ftdetect/sipu.vim 设置 filetype=cuda
  -> syntax/syntax.vim 设置 syntax=cuda
  -> syntax/cuda.vim 加载 C++/CUDA 规则
  -> after/syntax/cuda.vim 增加 clusterDim、clusterIdx
```

该方案的优点是改动最少、保留 `cuda` filetype 对 C++/CUDA 插件（例如 YCM）的兼容性。
代价是两个词也会对所有 `*.cu`/`*.cuh` buffer 成为合法的 `cudaVariable`。如果希望
只对 `.su` 生效，可以把 after 文件写成：

```vim
if &filetype ==# 'cuda' && expand('%:e') ==# 'su'
  syntax keyword cudaVariable clusterDim clusterIdx
endif
```

如果不想新建 `ftdetect` 文件，也可以把同一条检测规则追加到现有
`.vim/filetype.vim:1（<top-level autocmd）` 的下一行；但独立的 `ftdetect/sipu.vim`
更容易维护，也不会把不同用途的规则混在一起。

### 7.4 不污染 CUDA 的专用 filetype 方案

如果不希望普通 CUDA 文件认识 SiPU 关键字，可以使用专用 `sipu` filetype。需要选择
这一方案时，不再使用上一节的 `after/syntax/cuda.vim`，而是新建：

`/data/like/vim-port-all/config/.vim/ftdetect/sipu.vim`：

```vim
au BufRead,BufNewFile *.su setfiletype sipu
```

`/data/like/vim-port-all/config/.vim/syntax/sipu.vim`：

```vim
if exists('b:current_syntax')
  finish
endif

" Reuse all C++/CUDA rules first.
runtime! syntax/cuda.vim
unlet! b:current_syntax

syntax keyword sipuVariable clusterDim clusterIdx
highlight default link sipuVariable Identifier

let b:current_syntax = 'sipu'
```

这里先 `runtime! syntax/cuda.vim`，再清掉它留下的 `b:current_syntax`，是为了让
`sipu.vim` 继续追加自己的规则；最后把当前 syntax 标记为 `sipu`，避免重复加载。
`sipuVariable` 使用 `Identifier`，所以默认颜色仍与普通变量相同，也可以单独改成
`Special` 或自定义颜色。

这个方案的代价是 `&filetype` 变成 `sipu`。只支持 `c/cpp/cuda` 的插件可能不会自动
把它当作 C++ 处理；若需要 YCM、clangd 或 ftplugin 的 CUDA 行为，上一节的“复用
`cuda`”方案更合适。若确实要用专用 filetype，也可以在插件配置中把 `sipu` 加入对应
的 filetype 白名单，并自行设置编译器参数。

如果同一棵目录树中还存在 GCC `-fstack-usage` 生成的其他 `.su` 文件，不能只靠
`sipu` 这个名字解决误判：上面的 `*.su` 规则仍会匹配它们。此时应把两个方案中的检测
模式收窄到 SiPU 目录，例如：

```vim
au BufRead,BufNewFile */sirt/test/cuda/rt/*.su setfiletype cuda
" 专用 filetype 方案把上面的 cuda 改为 sipu。
```

也可以按文件内容在用户 `scripts.vim` 中判断 `__global__`、`#include <sipu.h>` 等
特征后再 `setfiletype`。路径模式应按实际工程布局调整；关键是让“文件类型检测”本身
区分 SiPU `.su` 与其他 `.su`，专用 syntax 只负责隔离高亮规则。

### 7.5 为什么使用 `syntax keyword`

`syntax keyword cudaVariable clusterDim clusterIdx`（或专用组的
`syntax keyword sipuVariable ...`）按完整单词匹配，不会把
`my_clusterDim_value` 的中间片段误判为关键字；这正符合它们像 `blockDim`、`threadIdx`
一样作为内建变量使用的场景。它只改变 Vim 的语法组和颜色，不会：

- 修改 SiPU/C++ 编译器的关键字集合；
- 改变语义检查、补全或 clangd 的解析结果；
- 自动为编辑器提供声明、类型或跳转信息。

如果还需要补全，应另行在 YCM/clangd 的编译参数或头文件中声明这些符号；语法高亮和
语义补全是两条独立链路。

### 7.6 立即试用而不修改任何文件

在持久化配置前，可以只对当前 buffer 执行：

```vim
:setfiletype cuda
:syntax keyword cudaVariable clusterDim clusterIdx
```

随后把光标放在 `clusterDim` 或 `clusterIdx` 上执行：

```vim
:set filetype? syntax?
:syntax list cudaVariable
:echo synIDattr(synID(line('.'), col('.'), 1), 'name')
```

预期分别看到 `filetype=cuda`、`syntax=cuda`，以及最后一条返回
`cudaVariable`。当前未扩展时，实测 `blockDim` 的 syntax ID 是 `cudaVariable`，而
`clusterDim`/`clusterIdx` 的 syntax ID 为 0；这可以直接验证规则是否生效。

如果选择专用方案，则把第一条改为 `:setfiletype sipu`，并使用
`:syntax list sipuVariable` 检查。

### 7.7 持久化后如何确认加载了正确文件

重新启动 Vim（文件类型检测文件是在启动时加载的），打开目标文件后检查：

```vim
:set runtimepath?
:set filetype? syntax?
:scriptnames
:syntax list cudaVariable
```

`:scriptnames` 应能看到用户 runtime 下的 `ftdetect/sipu.vim` 和
`after/syntax/cuda.vim`（专用方案则应看到 `syntax/sipu.vim`）。若 `filetype` 仍为空，
先检查启动时是否真正使用了 `/data/like/vim-port-all/config/.vimrc`；当前配置的
`.vimrc:5-10（<top-level>）` 只有在该 vimrc 被加载时才会把这些目录加入
`runtimepath`。

不要直接修改
`/data/like/vim-port-all/binary/vim-install-ubuntu22.04/share/vim/vim91/syntax/cuda.vim`
或 `.../vim91/filetype.vim`：它们属于 Vim 安装目录，升级 Vim 时容易被覆盖。Vim 官方
文档 `doc/filetype.txt:195-263（<documentation>）` 推荐用用户 `ftdetect`/`filetype.vim`
扩展文件类型，`doc/syntax.txt:174-193（<documentation>）` 推荐用
`after/syntax/<name>.vim` 扩展现有语法。

## 8. sglang v0.5.18 GSM8K 评测日志审计（2026-09-04）

### 8.1 评测范围和完整性

本节分析的日志是
`temp/llm_eval_online_quant.sh.MAX_RUNNING_REQUESTS_128_CUDA_GRAPH_MAX_BS_128_ADD_BOS_TOKEN_true__TASKS_gsm8k__CUDA_VISIBLE_DEVICES_7.log.2026_09_04___15_44_03`。
本次命令实际使用 `CUDA_VISIBLE_DEVICES=7`、`MAX_RUNNING_REQUESTS=128`、
`CUDA_GRAPH_MAX_BS=128`、`ADD_BOS_TOKEN=true` 和 `TASKS=gsm8k`；日志中的
`ServerArgs` 也显示 `cuda_graph_max_bs_decode=128`。权重量化调用没有额外指定
attention backend，KV-cache 调用则由
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:106-113（run_simo_config_list）`
加入 `attention_backend=triton_simo` 和 `disable_chunked_prefix_cache=true`。
脚本在 `simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:25-39（<top-level>）`
定义了 13 个权重量化 JSON，在
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:42-50（<top-level>）` 定义了 7 个 KV-cache 量化
JSON。`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:122-136（run_no_quant_eval）`
运行未量化基线，
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:140-154（run_model_evaluations）`
运行基线和 13 个权重量化，
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:157-167（run_model_evaluations_kv_cache_quant）`
运行 7 个 KV-cache 配置；Llama 和 DeepSeek 的调用位于
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:176-204（<top-level>）`。

日志中的计数如下：

| 项目 | 数量 |
| --- | ---: |
| `Running evaluation for` | 42 |
| `Running generate_until requests: ... 100%` | 42 |
| `gsm8k` 结果表 | 42 |
| `flexible-extract` 结果行 | 42 |
| `strict-match` 结果行 | 42 |

因此实际完成的是 `2 个模型 x (1 个基线 + 13 个权重量化 + 7 个 KV-cache 量化) = 42`
项。20 个 JSON 配置各被两个模型加载一次（`Loaded config from` 共 40 次；基线不加载
JSON），每个 KV 配置的日志都出现了 `Applying KV cache quantization`。没有发现某个 JSON
启动后静默退回未量化或缺少得分的情况。

### 8.2 core dump 和异常判断

对该日志检索 `core dump`、`core dumped`、`SIGSEGV`、`SIGABRT`、`Segmentation fault`、
`CUDA error`、`OutOfMemoryError`、`illegal memory access`、`FATAL` 和
`RuntimeError`，均为 0 次。因此日志没有记录 GPU 推理 core dump、CUDA 致命错误或因
OOM 中止。

日志确实打印了异常堆栈，但都是评测进程结束后的 loky/multiprocessing 清理异常，共
42 次同样的 traceback。第一次位于
`temp/llm_eval_online_quant.sh.MAX_RUNNING_REQUESTS_128_CUDA_GRAPH_MAX_BS_128_ADD_BOS_TOKEN_true__TASKS_gsm8k__CUDA_VISIBLE_DEVICES_7.log.2026_09_04___15_44_03:739-745`，
最后一次位于
`temp/llm_eval_online_quant.sh.MAX_RUNNING_REQUESTS_128_CUDA_GRAPH_MAX_BS_128_ADD_BOS_TOKEN_true__TASKS_gsm8k__CUDA_VISIBLE_DEVICES_7.log.2026_09_04___15_44_03:38873-38879`，模式是：

```text
kill_process_tree called
resource_tracker: process died unexpectedly, relaunching
Traceback (most recent call last):
  .../multiprocessing/resource_tracker.py:264, in main
    cache[rtype].remove(name)
KeyError: '/loky-...'
```

`resource_tracker` 的 warning 文本在日志中每次占两行（包含 `warnings.warn` 的续行），
所以文本行数为 84，但 traceback 和 `KeyError` 事件各为 42。每次 traceback 都出现在
该配置的结果表之后，随后马上进入下一个 `run_eval`，最后一个配置也正常输出分数；这说明
它是 teardown 阶段的资源清理竞态/泄漏风险，不是模型加载、请求生成或 GSM8K 计分失败。
严格说日志中“有 Python 异常堆栈”，但没有导致本次 42 项评测失败的异常。建议后续单独
修复 `kill_process_tree` 与 loky resource tracker 的退出顺序，避免清理阶段泄漏。

另外，`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:78（<top-level>）`
对应日志有被显式捕获的 `sarashina2_vision` 可选模型导入 warning，及多处 torch
`register_constant()` 弃用 warning、instruct/chat template 提示；它们与本次两个目标
模型和量化结果无关。

### 8.3 提取规则和结果文件

每项分数取日志中形如
`|gsm8k| ... |flexible-extract| ... |0.xxxx|` 的 `exact_match` 值，再乘以 100
转换为百分比并保留两位小数（例如 `0.699` -> `69.90`）。字段顺序逐项复制
`tests/sglang_simo/references_accuracy/gsm8k.yaml:1-84`；日志名称
`kvquant_fp8_per_group` 和 `kvquant_int8_per_group` 对应参考字段
`fp8_per_group_64` 和 `int8_per_group_64`。完整结果已写入
`tests/sglang_simo/references_accuracy/gsm8k-v0.5.18.yaml`。
已使用 simo conda 环境中的 YAML 解析器，并按模型和配置名将日志中的 42 行分数与该文件
逐项校验，42/42 全部匹配。

### 8.4 Llama-3.1-8B-Instruct 结果和对比

下表的“旧基准”来自 `tests/sglang_simo/references_accuracy/gsm8k.yaml`，差值为
`v0.5.18 - 旧基准`，单位是百分点（pp）；日志行是对应的 flexible-extract 行。

| 配置 | 日志行 | v0.5.18 (%) | 旧基准 (%) | 差值 (pp) |
| --- | ---: | ---: | ---: | ---: |
| no-quant | 736 | 77.63 | 77.63 | +0.00 |
| w8a8_fp8_per_block | 7661 | 77.26 | 76.95 | +0.31 |
| w4a16_int4_per_group | 1601 | 73.39 | 72.93 | +0.46 |
| w8a8_int8_per_block | 9395 | 77.94 | 77.26 | +0.68 |
| w8a8_fp8_per_channel | 8524 | 76.88 | 77.71 | -0.83 |
| w8a8_int8_per_channel | 10247 | 77.03 | 75.59 | +1.44 |
| w8a8_mxint | 11974 | 77.48 | 77.48 | +0.00 |
| w8a8_mxfp | 11122 | 77.03 | 77.03 | +0.00 |
| w6a6_mxfp | 6792 | 76.35 | 76.35 | +0.00 |
| w4a4_mxfp | 4213 | 47.61 | 47.61 | +0.00 |
| w4a16_nvfp4_per_group | 3327 | 73.24 | 73.46 | -0.22 |
| w4a16_nvfp4_per_group_4_over_6 | 2461 | 75.51 | 74.00 | +1.51 |
| w4a4_nvfp | 5923 | 69.07 | 69.07 | +0.00 |
| w4a4_nvfp_4_over_6 | 5077 | 70.13 | 70.13 | +0.00 |
| mxfp8 | 12802 | 76.72 | 76.72 | +0.00 |
| mxfp4 | 13627 | 69.90 | 69.90 | +0.00 |
| mxfp6 | 14443 | 77.79 | 77.79 | +0.00 |
| mxint8 | 15251 | 77.94 | 77.94 | +0.00 |
| fp8_per_group_64 | 16048 | 78.47 | 76.95 | +1.52 |
| int8_per_group_64 | 16860 | 77.33 | 77.10 | +0.23 |
| nvfp4 | 17667 | 76.57 | 76.57 | +0.00 |

### 8.5 DeepSeek-V2-Lite-Chat-16B_A2.4B 结果和对比

| 配置 | 日志行 | v0.5.18 (%) | 旧基准 (%) | 差值 (pp) |
| --- | ---: | ---: | ---: | ---: |
| no-quant | 18549 | 66.03 | 66.03 | +0.00 |
| w8a8_fp8_per_block | 26652 | 65.96 | 64.97 | +0.99 |
| w4a16_int4_per_group | 19570 | 56.86 | 58.68 | -1.82 |
| w8a8_int8_per_block | 28761 | 65.88 | 66.64 | -0.76 |
| w8a8_fp8_per_channel | 27724 | 65.58 | 65.50 | +0.08 |
| w8a8_int8_per_channel | 29784 | 63.31 | 64.06 | -0.75 |
| w8a8_mxint | 31848 | 65.28 | 65.28 | +0.00 |
| w8a8_mxfp | 30822 | 64.90 | 64.90 | +0.00 |
| w6a6_mxfp | 25598 | 64.37 | 64.37 | +0.00 |
| w4a4_mxfp | 22583 | 38.51 | 38.51 | +0.00 |
| w4a16_nvfp4_per_group | 21568 | 63.08 | 63.84 | -0.76 |
| w4a16_nvfp4_per_group_4_over_6 | 20570 | 63.91 | 61.71 | +2.20 |
| w4a4_nvfp | 24565 | 56.79 | 56.18 | +0.61 |
| w4a4_nvfp_4_over_6 | 23573 | 57.77 | 60.20 | -2.43 |
| mxfp8 | 32874 | 66.03 | 66.03 | +0.00 |
| mxfp4 | 33859 | 31.39 | 31.39 | +0.00 |
| mxfp6 | 34863 | 64.37 | 64.37 | +0.00 |
| mxint8 | 35878 | 66.03 | 66.03 | +0.00 |
| fp8_per_group_64 | 36886 | 66.34 | 66.03 | +0.31 |
| int8_per_group_64 | 37870 | 66.34 | 66.26 | +0.08 |
| nvfp4 | 38870 | 47.08 | 47.08 | +0.00 |

### 8.6 精度变化结论

1. 两个模型的未量化基线分别为 77.63% 和 66.03%，与旧基准完全一致，说明这次
   对比的任务、数据和基本推理路径没有出现整体偏移。
2. KV-cache 量化没有明显升级回归：两模型的 `mxfp8`、`mxfp4`、`mxfp6`、`mxint8`
   和 `nvfp4` 都与旧基准（四舍五入到 0.01 pp）一致；per-group 配置的最大差值是
   Llama `fp8_per_group_64` 的 +1.52 pp，DeepSeek 为 +0.31 pp（`int8_per_group_64`
   为 +0.23/+0.08 pp）。这更像单次评测波动，不能据此判定 KV kernel 发生回归。
3. 权重量化多数变化不超过约 1 pp。Llama 中较大的变化是
   `w4a16_nvfp4_per_group_4_over_6` +1.51 pp、`w8a8_int8_per_channel` +1.44 pp；
   `w8a8_fp8_per_channel` 为 -0.83 pp。DeepSeek 中最值得复测的是
   `w4a4_nvfp_4_over_6` -2.43 pp、`w4a16_nvfp4_per_group_4_over_6` +2.20 pp 和
   `w4a16_int4_per_group` -1.82 pp。
4. 当前日志每项 stderr 约为 1.1--1.4 个百分点（1319 个 GSM8K 样本）；因此上述
   0.5--1.5 pp 的变化不能单凭一次运行解释为版本回归，DeepSeek 的约 2 pp 变化应使用
   相同随机种子重复运行或比较逐样本预测后再归因。
5. `Llama w4a4_mxfp=47.61%`、`DeepSeek w4a4_mxfp=38.51%`、`DeepSeek KV mxfp4=31.39%`
   和 `DeepSeek KV nvfp4=47.08%` 的绝对精度确实较低，但它们与旧基准完全相同，属于
   量化配置本身的精度特征，而不是 v0.5.18 适配新引入的退化。
