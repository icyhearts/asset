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

## 9. sglang v0.5.18 MMLU 评测日志审计（2026-09-04）

### 9.1 评测范围和结果完整性

本节分析的日志是
`temp/llm_eval_online_quant.sh.MAX_RUNNING_REQUESTS_128_CUDA_GRAPH_MAX_BS_128_ADD_BOS_TOKEN_true__TASKS_mmlu__CUDA_VISIBLE_DEVICES_6.log.2026_09_04___15_43_28`。
脚本配置定义位于
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:25-39（<top-level>）`
（13 个权重量化 JSON）和
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:42-50（<top-level>）`
（7 个 KV-cache JSON）。基线、权重量化和 KV-cache 评测分别由
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:122-136（run_no_quant_eval）`、
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:140-154（run_model_evaluations）` 和
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:157-167（run_model_evaluations_kv_cache_quant）` 调用；两个模型的顶层调用位于
`simo/extensions/sglang_simo/example/online_quantization/llm_eval_online_quant.sh:176-204（<top-level>）`。

日志共启动 42 次评测，即每个模型各有 `1 个 no-quant + 13 个权重量化 + 7 个 KV-cache
量化`。每次 MMLU 评测的 56,168 个 loglikelihood 请求都达到 `100%`，并输出结果表；
因此 42 个配置（包含基线）均有总分。每次结果会打印一张详细 Tasks 表和一张 Groups
表，两张表中的 `mmlu` 总行完全相同，所以本节每个配置只取第一张表的总行，不展开或
写入学科分数。20 个 JSON 配置各出现两次 `Loaded config from`（两个模型各一次），KV
配置均有 `Applying KV cache quantization` 日志，未发现缺失配置或静默回退。

### 9.2 core dump 和异常判断

在完整日志中检索 `core dump`、`core dumped`、`SIGSEGV`、`SIGABRT`、`SIGBUS`、
`CUDA error`、`OutOfMemoryError`、`illegal memory access`、`FATAL` 和 `RuntimeError`，
均没有匹配。因此没有记录 GPU core dump、CUDA 致命错误或 OOM 中止。

日志中确实有 42 次 Python traceback 和 42 次 `KeyError`，但全部发生在评测结果输出
之后的进程清理阶段：

```text
kill_process_tree called
resource_tracker: process died unexpectedly, relaunching
Traceback (most recent call last):
  .../multiprocessing/resource_tracker.py:264, in main
    cache[rtype].remove(name)
KeyError: '/loky-...'
```

第一次清理位于
`temp/llm_eval_online_quant.sh.MAX_RUNNING_REQUESTS_128_CUDA_GRAPH_MAX_BS_128_ADD_BOS_TOKEN_true__TASKS_mmlu__CUDA_VISIBLE_DEVICES_6.log.2026_09_04___15_43_28:1322-1328`，
最后一次位于同一日志的 `:61323-61329`。每次 traceback 后脚本都进入下一个配置，且
最后一个配置也输出了总分；这是 loky/multiprocessing resource tracker 的 teardown
清理竞态（仍然属于日志中的异常堆栈），没有导致本次评测任务失败。另有被显式忽略的
`sarashina2_vision` 可选模型导入 warning 和 torch 弃用 warning，与两个目标模型的
MMLU 结果无关。

### 9.3 总分提取和结果文件

总分取日志中形如
`|mmlu| ... |acc| ... |0.xxxx|` 的聚合行，乘以 100 后保留两位小数；同一配置的
第二张 Groups 表只用于交叉校验。结果字段顺序严格复制
`tests/sglang_simo/references_accuracy/mmlu.yaml:1-85`，并保留该文件中的 no-quant
基线条目。完整总分已写入
`tests/sglang_simo/references_accuracy/mmlu-v0.5.18.yaml`；没有把学科（humanities、
stem 等）明细写入 YAML。

### 9.4 Llama-3.1-8B-Instruct 总分对比

旧基准来自 `tests/sglang_simo/references_accuracy/mmlu.yaml`；差值为
`v0.5.18 - 旧基准`，单位为百分点（pp）。日志行是第一张 Tasks 表的总 `mmlu` 行。

| 配置 | 日志行 | v0.5.18 (%) | 旧基准 (%) | 差值 (pp) |
| --- | ---: | ---: | ---: | ---: |
| no-quant | 1251 | 68.47 | 68.47 | +0.00 |
| w8a8_fp8_per_block | 12790 | 68.02 | 68.18 | -0.16 |
| w4a16_int4_per_group | 2703 | 66.21 | 66.22 | -0.01 |
| w8a8_int8_per_block | 15672 | 67.98 | 68.17 | -0.19 |
| w8a8_fp8_per_channel | 14231 | 68.02 | 67.89 | +0.13 |
| w8a8_int8_per_channel | 17113 | 67.48 | 67.73 | -0.25 |
| w8a8_mxint | 19995 | 68.20 | 68.20 | +0.00 |
| w8a8_mxfp | 18554 | 67.67 | 67.67 | +0.00 |
| w6a6_mxfp | 11349 | 68.07 | 68.07 | +0.00 |
| w4a4_mxfp | 7026 | 57.99 | 57.99 | +0.00 |
| w4a16_nvfp4_per_group | 5585 | 66.22 | 66.10 | +0.12 |
| w4a16_nvfp4_per_group_4_over_6 | 4144 | 66.49 | 66.53 | -0.04 |
| w4a4_nvfp | 9908 | 64.13 | 64.13 | +0.00 |
| w4a4_nvfp_4_over_6 | 8467 | 64.45 | 64.45 | +0.00 |
| mxfp8 | 21395 | 68.27 | 68.27 | +0.00 |
| mxfp4 | 22776 | 68.27 | 68.27 | +0.00 |
| mxfp6 | 24157 | 68.27 | 68.27 | +0.00 |
| mxint8 | 25538 | 68.27 | 68.27 | +0.00 |
| fp8_per_group_64 | 26919 | 68.27 | 68.27 | +0.00 |
| int8_per_group_64 | 28300 | 68.27 | 68.27 | +0.00 |
| nvfp4 | 29681 | 68.27 | 68.27 | +0.00 |

### 9.5 DeepSeek-V2-Lite-Chat-16B_A2.4B 总分对比

| 配置 | 日志行 | v0.5.18 (%) | 旧基准 (%) | 差值 (pp) |
| --- | ---: | ---: | ---: | ---: |
| no-quant | 31055 | 56.65 | 56.72 | -0.07 |
| w8a8_fp8_per_block | 43197 | 56.38 | 56.52 | -0.14 |
| w4a16_int4_per_group | 32579 | 54.88 | 54.62 | +0.26 |
| w8a8_int8_per_block | 46250 | 56.47 | 56.55 | -0.08 |
| w8a8_fp8_per_channel | 44737 | 55.97 | 55.94 | +0.03 |
| w8a8_int8_per_channel | 47763 | 55.88 | 55.88 | +0.00 |
| w8a8_mxint | 50789 | 56.51 | 56.56 | -0.05 |
| w8a8_mxfp | 49276 | 55.83 | 55.86 | -0.03 |
| w6a6_mxfp | 41657 | 55.86 | 55.93 | -0.07 |
| w4a4_mxfp | 37118 | 48.43 | 48.66 | -0.23 |
| w4a16_nvfp4_per_group | 35605 | 54.04 | 54.03 | +0.01 |
| w4a16_nvfp4_per_group_4_over_6 | 34092 | 54.55 | 54.69 | -0.14 |
| w4a4_nvfp | 40144 | 52.83 | 53.06 | -0.23 |
| w4a4_nvfp_4_over_6 | 38631 | 52.94 | 52.98 | -0.04 |
| mxfp8 | 52300 | 56.72 | 56.76 | -0.04 |
| mxfp4 | 53792 | 56.72 | 56.76 | -0.04 |
| mxfp6 | 55284 | 56.72 | 56.76 | -0.04 |
| mxint8 | 56776 | 56.72 | 56.76 | -0.04 |
| fp8_per_group_64 | 58268 | 56.72 | 56.76 | -0.04 |
| int8_per_group_64 | 59760 | 56.72 | 56.76 | -0.04 |
| nvfp4 | 61252 | 56.72 | 56.76 | -0.04 |

### 9.6 精度变化结论

1. Llama 基线与旧参考完全一致（68.47%）；DeepSeek 基线低 0.07 pp（56.65% 对
   56.72%）。
2. Llama 权重量化的最大绝对变化为 0.25 pp（`w8a8_int8_per_channel`），所有
   KV-cache 配置均为 68.27%，与旧参考完全一致。
3. DeepSeek 权重量化的最大变化为 +0.26 pp（`w4a16_int4_per_group`），下降最大为
   -0.23 pp（`w4a4_mxfp` 和 `w4a4_nvfp`）；所有 KV-cache 配置均为 56.72%，比旧参考
   的 56.76% 低 0.04 pp。
4. 当前 MMLU 总分的报告 stderr 约为 0.37--0.41 pp，所有配置与旧参考的差值绝对值
   不超过 0.26 pp，低于该次运行的不确定性范围。因此没有观察到 sglang v0.5.18
   适配导致的明显精度回归或提升；结论仅基于这一次运行和已四舍五入的总分，若要判断
   细微差异，应使用相同种子重复评测或比较逐样本预测。

## 10. release/v0.5.18 -> release/v0.5.19 重要修改

### 10.1 比较范围

本节以 /share/users/like/package/sglang_kernel_src 为 code base，比较两个 branch
的 tip；当前工作区未提交改动不在比较中。代码引用统一为“相对 code base 路径:行号
（函数名）”，类成员函数写成 类名::函数名。

- release/v0.5.18: 9d18277c2720f3a1d6e64f90259e1d1b9ba9a26a（2026-08-26）。
- release/v0.5.19: 0bcd822377da7b5718e674eaf9c870d349424dd1（2026-09-03）。
- 版本间 794 个提交，2786 个文件变化，新增 299538 行、删除 60541 行。
- 变化最多的目录是 python、test、docs、rust；因此 v0.5.19 是功能和架构扩展版，
  不是只修几个 CUDA kernel 的补丁版。

题目指定 branch，不能用 tag 结果替代上面的结论；release 分支上的 cherry-pick
可能使 branch-to-branch 与 tag-to-tag 的文件集合不同。

### 10.2 重要修改总览

| 子系统 | 主要提交/内容 | 直接影响 |
| --- | --- | --- |
| DSV4/DSA | Q8KV8 sparse prefill、fused top-k、top-k v2、FP4 runner 自动选择 | 新的 attention/MoE 分支和 JIT 模块 |
| MoE | DeepGEMM 0.1.7、MegaMoE MXFP4、DeepEPv2、可插拔 runner | 通信布局、scale、SM 预算和编译缓存契约改变 |
| Spec/graph | mixed chunk、DFlash2 selector、TP sync、PP FullCG | target_verify graph 的 shape 和 warmup 组合增加 |
| KV/HiCache | ReqKvInfo、统一字节池、统一 radix/tree/linker | KV 所有权、容量、evict/retract 和外部存储语义改变 |
| 配置/前端 | resolution pipeline、namespace bags、嵌入式 Rust server | 运行时配置读取和 IPC 兼容性改变 |
| 平台/依赖 | FlashInfer 0.6.18、ROCm10/gfx1250、CUDA13.4/Rubin、XPU/NPU | native wheel、驱动和 JIT cache 需要重新对齐 |
| 模型/扩散 | Qwen3.8、GLM-5.3-Flash、Ling-3.0-flash、EPD 等 | 模型覆盖面大幅扩大，diffusion 变更与文本路径相对独立 |

### 10.3 DeepSeek V4、DSA 和 FP4

1. **Q8KV8 sparse MLA prefill（9db4ba8da1）。**

   入口是 python/sglang/srt/server_args.py:1777-1788
   （ServerArgs::dsv4_prefill_backend），新增 auto、flashmla_sparse、
   flashmla_sparse_q8。真正的 Q8 开关由
   python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py:60-75
   （use_dsv4_q8kv8_sparse_prefill）决定：显式选 flashmla_sparse_q8，或设置
   SGLANG_DSV4_Q8KV8_PREFILL=true。

   python/sglang/srt/layers/attention/deepseek_v4_backend.py:532-587
   （DeepseekV4AttnBackend::__init__）检查 SM90、d_v=512；
   :1668-1788（DeepseekV4AttnBackend::forward）在大查询的 extend path 路由；
   :1933-1983（DeepseekV4AttnBackend::_prepare_q8kv8_q_and_sink）把 head 补齐到
   64-head CTA；
   :1985-2122（DeepseekV4AttnBackend::_forward_prefill_sparse_q8kv8）把压缩 KV
   gather、dequant/requant 到 FP8 workspace 后调用新 kernel。

   新模块在 python/sglang/kernels/ops/attention/sparse_mla_q8kv8_prefill_sm90.py:
   60-74（_jit_sparse_mla_q8kv8_prefill_module），公开入口为 :268-528
   （sparse_mla_q8kv8_prefill_fwd）。它要求 CUDA、FP8 E4M3、h_q 为 64 的倍数、
   topk 为 128 的倍数、d_v=512；第一次命中可能编译，错误 shape 会快速报错。
   backend=auto 本身不会打开 q8。

2. **DSA top-k 后端和 v2 kernel。**

   5edcd0a445 增加 FlashInfer 0.6.18 fused top-k。枚举和 target/draft 分流在
   python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py:27-50
   （DSATopKBackend::resolve）；实现位于 :55-91
   （DSATopKBackend::topk_func）和 :93-210（DSATopKBackend::topk_transform），
   支持 sgl-kernel、torch、flashinfer。target/draft 参数分别是
   python/sglang/srt/server_args.py:1816-1823
   （ServerArgs::dsa_topk_backend）和 :2186-2193
   （ServerArgs::speculative_dsa_topk_backend）。默认仍为 sgl-kernel。

   DSV4 的 FlashInfer transform 在
   python/sglang/srt/layers/attention/dsv4/indexer.py:407-432
   （topk_transform_512_flashinfer_fused），路由在 :855-890
   （C4IndexerBackendMixin::forward_c4_indexer）。top-k v2 的单一 JIT 模块在
   python/sglang/kernels/ops/attention/dsv4/topk.py:35-47
   （_jit_topk_v2_module），paged/ragged 入口在 :78-92、:95-129、:132-178；
   它减少按 k 编译的模块数量，但首次加载仍可能触发 JIT。

3. **DSV4 FP4 的 auto runner 改变。**

   60ff1e33a5 在
   python/sglang/srt/arg_groups/model_overrides/deepseek_v4.py:20-76
   （_deepseek_v4_overrides）中规定：CUDA、非 HIP、SM90/100/120、FP4 experts、
   无 A2A、未强制 dequant 且 runner=auto 时使用 flashinfer_mxfp4；带
   nvfp4_moe_meta 的混合 checkpoint 使用 flashinfer_trtllm_routed。因此同一个
   DSV4 FP4 checkpoint 在两个版本的 auto 选择可能不同，不能假定仍走 Marlin。

### 10.4 MoE、DeepGEMM、DeepEPv2 和 Marlin

1. **DeepGEMM 0.1.5.post3 -> 0.1.7（a7ee399904、07c8f7294d）。**

   MegaMoE 现在在 python/sglang/srt/layers/moe/mega_moe.py:48-50
   （_mega_moe_mma_type）区分 fp8xfp4 与 mxf4xmxf4；
   :52-86（_mega_moe_max_num_sms、
   _configure_mega_moe_deep_gemm_num_sms）在 Blackwell 为 whole-grid barrier
   预留 SM，默认值在 python/sglang/srt/environ.py:1132-1135
   （SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS）为 2；
   :89-122（_get_mega_moe_symm_buffer）把 mma_type 纳入 buffer key；
   :257-304（forward_mega_moe、_run_mega_routed）使用新的 pre-dispatch；
   :312-337（_interleave_mega_moe_gate_up 等）和 :351-408
   （build_mega_moe_experts_weights）改写 MXFP4/UE8M0 权重布局。

2. **本地 rank 的重复 JIT 编译被去重（7769ff8f1e）。**

   python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:120-140
   （_local_rank_compile_lock）在 SGLANG_DG_CACHE_DIR/locks 下按
   kernel、N、K、num_groups 加 fcntl 锁；:144-186
   （_maybe_compile_deep_gemm_one_type_all）让锁持有者执行 :190-260
   （_compile_deep_gemm_one_type_all）。锁只避免同节点多份 nvcc，不会消除首个
   rank 的完整 M sweep；:161-170 明确提示无预编译时通常需要 10--20 分钟。
   非 fast warmup 的 M 列表 :83-91 仍可覆盖到 128K。

   预编译入口在 python/sglang/compile_deep_gemm.py:63-96
   （warm_up_compile）；:183-204（compile_server_args）会关闭 CUDA graph 和
   torch.compile 后发起 warmup。必须使用和 launch_server 相同的模型、TP、dtype、
   MoE/A2A 参数，否则 cache 可能仍不命中。

3. **DeepEPv2 ElasticBuffer A2A（a3ae667d67）。**

   新文件 python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py 中，
   :136-210（DeepEPv2Buffer::get_buffer）按通信组、hidden、topk、容量、FP8、
   hybrid mode 复用进程级 ElasticBuffer；
   :279-396（_DeepEPv2Impl::dispatch）支持 decode expanded/masked 和 extend
   contiguous layout，使用 FP8 group-128 scale；
   :398-419（_DeepEPv2Impl::combine）回收单次 handle；
   :421-461（DeepEPv2Dispatcher::dispatch/combine）接入通用 dispatcher。

   配置在 python/sglang/srt/server_args.py:2366-2414
   （ServerArgs::moe_a2a_backend、ServerArgs::deepep_v2_mode），新增
   deepep_v2 及 direct/hybrid；环境默认容量和 SM 数在
   python/sglang/srt/environ.py:1107-1109
   （SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK、
   SGLANG_DEEPEP_V2_NUM_SMS）为 128、0。它要求 DeepGEMM runner，适配代码在
   python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:1566-1634
   （pre_permute_deepep_v2_to_deep_gemm）和 :1701-1722
   （post_permute_deep_gemm_to_deepep_v2）。

4. **MoE runner 扩展和 Marlin。**

   python/sglang/srt/layers/moe/moe_runner/runner.py:33-50
   （register_moe_runner_core）及 :53-150（MoeRunner::__init__）增加自定义
   runner core 注册接口；直接接收 dispatch 表示的接口基类在
   python/sglang/srt/layers/moe/moe_runner/base.py:117-134
   （DispatchMoeRunnerCore）。旧的私有 runner/monkey-patch 需要审计。

   Marlin 核心算法在本范围内没有同等级重写。其变化主要是
   python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py:130-167
   （Mxfp4MarlinMoEMethod::process_weights_after_loading）的 SM90/SM120 和
   block-32 检查，以及 python/sglang/srt/layers/quantization/mxfp4.py:617-644、
   :1458-1478（Mxfp4MoEMethod::process_weights_after_loading、
   Mxfp4MoEMethod::_apply_marlin）。因此只有显式选择 Marlin 或其它模型
   选中该 runner 时才会走 Marlin；DSV4 FP4 auto 通常优先 FlashInfer MXFP4。

5. **FlashInfer FP4 后端增加。**

   9a85473a89 增加 Blackwell NVFP4 W4A16 CuTe DSL，开关是
   python/sglang/srt/environ.py:965-969
   （SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16），权重准备在
   python/sglang/srt/layers/quantization/modelopt_quant.py:1692-1701、
   :1793-1809、:2053-2068。5b04408784 增加 SM90 MXFP4 W4A8
   CUTLASS/Humming。它们是 Marlin 的并列 backend，不应把所有 FP4 转换都叫
   Marlin 编译。

6. **JIT 和 expert-pack 目录重构（db6f0a9d53）。**

   elementwise/speculative csrc、MiniCPM-SALA ops 和 tools/expert_pack 已移到
   package 内；工具入口在 python/sglang/srt/model_loader/expert_pack/build.py:230-274
   （build），运行时准备在 python/sglang/srt/model_loader/expert_pack_runtime.py:
   87-100（prepare_kimi_model_metadata）。外部脚本若仍引用 tools.expert_pack 或
   旧 JIT source path，会在 editable 升级后失败。

### 10.5 Speculative、CUDA graph 和调度

1. 07d84ebd6d 让 mixed chunk prefill 与 spec 联动：
   python/sglang/srt/speculative/spec_info.py:132-143
   （SpeculativeAlgorithm::supports_mixed_chunk）允许 EAGLE/EAGLE3/DFlash/
   DSpark；python/sglang/srt/managers/schedule_batch.py:2886-2957
   （ScheduleBatch::mix_with_running）合并新 prefill 和 running decode；
   python/sglang/srt/managers/overlap_utils.py:471-513
   （FutureMap::resolve_mixed_spec_tails）在 publish fence 后重建 spec tail。
   吞吐更好，但 forward mode、graph bucket 和跨 stream 状态组合更多。

2. adaptive draft 容量与运行时宽度解耦：
   python/sglang/srt/speculative/spec_info.py:242-263
   （SpeculativeAlgorithm::resolve_max_speculative_num_draft_tokens）从
   candidate-step 算最大 slot；python/sglang/srt/runtime_context.py:959-970
   （RuntimeContext::set_server_args）把它写入 spec bag。它会改变显存预算和
   target_verify/draft graph capture bucket。

3. f60bc73c58 增加 DSpark/DFlash 的 TP 状态同步：
   python/sglang/srt/speculative/spec_tp_sync.py:14-42
   （SpecTpSyncSite）和 :99-129（SpecTpSync::sync）按
   SGLANG_SPEC_TP_SYNC 对 memory、selector、sample、accept、target 等站点
   broadcast；这修复 rank divergence，但增加同步边界。

4. Full prefill CUDA graph 支持 PP proxy/full graph，并修复 padding/EAGLE capture
   （b77cac06a9、26fd7fdaa2、9a9e167179）。最终逻辑在
   python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py:
   284-491（capture_prefill_graph），按 backend、max request、context length
   筛选 buckets并记录耗时；python/sglang/srt/model_executor/model_runner.py:
   1424-1456（ModelRunner::init_decode_cuda_graph、
   ModelRunner::init_prefill_cuda_graph）把 target_verify/draft_decode 资源纳入
   统计，:1774-1802（ModelRunner::_forward_raw）实际执行 prefill graph。

5. 新增 prefill_decode_interval（python/sglang/srt/server_args.py:718-722，
   ServerArgs::prefill_decode_interval）和 gated launch
   （python/sglang/srt/distributed/gated_launch.py:22-37、
   :40-65，maybe_wait_for_gated_launch、_wait_until_activated）。两者默认不会
   改变普通单机路径；97ba99067d 的 per-scheduler load socket 主要给 load-aware
   router。

6. a3c4936438 让 FlashInfer autotune tactic 在 TP ranks 一致；路径在
   python/sglang/srt/model_executor/runner/flashinfer_autotune.py:177-285。额外 EXTEND dummy
   autotune 由 python/sglang/srt/environ.py:993-997
   （SGLANG_FLASHINFER_AUTOTUNE_EXTEND）控制，v0.5.19 默认 false。

7. **DFlash2 和新的 candidate selector。**

   c14312a664、99c12218c3 为 DFlash 增加本地 grouped convolution 和 selector。
   python/sglang/srt/models/dflash.py:944-1063
   （CandidateSelector::build_lattice、CandidateSelector::sample_path）在候选
   token 之间建立 K x K lattice；:1066-1140
   （DFlash2DraftModel::__init__、DFlash2DraftModel::compute_candidates）只在
   TP ranks 间 gather 每个 shard 的 top-k，而不是 gather 全词表。FlashInfer 不可用
   时会退回 torch.topk，首次 selector/conv graph 也可能引入额外 warmup。

8. **Beam search 成为受限的正式 API。**

   ec4bdbfa4a 增加 beam group/coordinator。请求字段在
   python/sglang/srt/sampling/sampling_params.py:74-76
   （SamplingParams::beam_width），校验在 :157-159
   （SamplingParams::verify）；搜索状态在
   python/sglang/srt/beam_search/beam_group.py:60-160
   （BeamGroup::__init__、BeamGroup::advance_frontier）和
   python/sglang/srt/beam_search/coordinator.py:101-180
   （BeamCoordinator::validate_and_init）。一个 beam_width=k 请求使用一个 leader
   和 k-1 个 req-to-token member rows，因此会直接增加 KV row/容量压力。
   当前明确不支持 speculative、PD、page_size>1、DP/PP attention、HiCache、
   LoRA、约束解码和 hybrid SWA/Mamba；不能把 beam_width 当成普通 sampling n。

### 10.6 KV cache、HiCache 和统一内存

1. ReqKvInfo 收拢 KV 所有权。字段在
   python/sglang/srt/managers/schedule_batch.py:848-900（ReqKvInfo），有效
   committed length 在 :1330-1335（Req::effective_kv_committed_len）；按行
   释放在 python/sglang/srt/mem_cache/base_prefix_cache.py:372-383
   （BasePrefixCache::free_kv_row）。自定义 attention、PD、hybrid wrapper
   应传递真实 ReqKvInfo 和 KV index translator。

2. 统一字节池在
   python/sglang/srt/mem_cache/unified_memory_pool.py:293-345
   （UnifiedKVPool::__init__）构造多种 view；:1810-1970
   （init_unified_mamba_swa_pools）形成 [mamba up | swa float | full down]，
   :1916-1936（_check_bs1_feasibility_floor）在启动时检查最低字节预算。
   ef9e58fd6d、98cb3535b7、961beee9e5 改变了 hybrid cache 的 sizing、relocation、
   eviction 和 retraction 失败行为。

3. 44806dc507 让 unified radix tree 成为默认；旧开关在
   python/sglang/srt/environ.py:638-643（SGLANG_ENABLE_UNIFIED_RADIX_TREE）
   已 deprecated。9cf157c252 增加可选 Rust TreeCore，注册在
   python/sglang/srt/mem_cache/unified_cache/tree_core_registry.py:57-75
   （_rust_tree_core_factory、create_tree_core）。c9eb475a88 到 b21000aef1
   定义统一 external linker，接口在
   python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py:51-112
   （UnifiedCacheLinker），运行时 attach/detach 在
   python/sglang/srt/mem_cache/unified_cache/storage_attachment.py:37-165
   （StorageAttachment::attach、StorageAttachment::detach）。

4. 977412ae61 增加 HiCache buffer_only host mode，把 host memory 作为
   GPU<->storage staging 而非持久 L2 cache；启用时要重新验证 sidecar、unified-SWA、
   prefetch/write-back 的组合。

### 10.7 配置架构、Rust 前端和 EPD

1. c2928e86d7 到 e51a3ae65e 把 ServerArgs 改为 raw input + resolution result：
   python/sglang/srt/server_args.py:3663-3671（ServerArgs::__post_init__）不再
   立即改写字段，:3673-3706（ServerArgs::resolve_once）只运行一次 pipeline，
   :3708-3719（ServerArgs::resolved_dict）返回解析值；pipeline 在
   python/sglang/srt/arg_groups/pipeline.py:26-110、:340-370
   （run_resolution_pipeline）；bags 和 runtime override 在
   python/sglang/srt/runtime_context.py:798-846（_build_config_bags）、
   :939-989（RuntimeContext::set_server_args）、:1020-1060
   （RuntimeContext::override）、:1465-1512（publish）。

   out-of-tree code 应在启动阶段读取 resolving_view/resolved_view，运行阶段读取
   get_exec/get_spec 等 bags；直接写 server_args.foo 不再保证同步到 runtime。

2. SGLANG_RUST_SERVER 默认 false。打开后
   python/sglang/srt/rust_server/server.py:47-166（RustServer::launch）在
   scheduler 进程内启动 Rust threads，:168-229（RustServer::drain）用
   columnar input-id buffer 传输请求。Rust 边界在
   rust/sglang-server/src/lib.rs:75-130（Server::start）和 :132-163
   （Server::recv_requests、Server::wait_request）；sampling ABI 在
   rust/sglang-server/src/message/sampling.rs:94-115、:247-275、:321-340。
   这属于 opt-in，默认配置不会进入 Rust 前端路径。

3. 8a123cbd0e 重构 EPD encoder disaggregation。入口包括
   python/sglang/srt/disaggregation/encoder/http_server.py:264
   （handle_encode_request）、
   python/sglang/srt/disaggregation/encoder/runtime.py:101（EncoderScheduler）、
   :353（EncoderRuntime）和 python/sglang/srt/disaggregation/encoder/server.py:2012
   （run_encoder）。它主要影响 multimodal/encoder-only PD。

### 10.8 硬件和依赖

| 平台 | 重要内容 |
| --- | --- |
| NVIDIA | CUDA 13.4/Rubin 容器、SM90 FP8 decode 修复、Blackwell NVFP4 CuTe DSL、SM90 MXFP4 W4A8 |
| AMD | ROCm10/gfx1250、Lean persistent-CTA decode、12-head MLA Gluon、MoRI MXFP8 dispatch |
| XPU | dense AWQ/GPTQ INT4 native int4pack、FP8、SYCL DSV4 top-k、fused GDN |
| NPU | DSV4 compressor/sparse-attn、ModelSlim W4A4 MXFP4、NPU kernel wheel |
| CPU | bidirectional attention 的 is_causal 修复、GPTQ INT4、ERNIE |

代表性代码位置：AMD Lean 在
python/sglang/kernels/ops/attention/decode_attention.py:1156-1234
（decode_attention_fwd）和 :1737-1872（lean_capture_policy、
lean_decode_seqlen_gate）；XPU INT4 在
python/sglang/srt/hardware_backend/xpu/quantization/int4pack_utils.py:17-80
（pack_int4_to_uint8、xpu_int4pack_mm）；CPU 修复在
python/sglang/kernels/aot/csrc/cpu/extend.cpp:26-69、:239-310
（extend_attention_kernel_impl 和 stage-2 mask）。

Qwen3.8 rebase（5f216fc33f）还加入 Blackwell/NVLink 专用的 MNNVL CuTe DSL
AllReduce + residual/RMSNorm/MoE finalize fusion。总开关位于
python/sglang/srt/environ.py:1587-1595
（SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION，默认 false），workspace 入口在
python/sglang/kernels/ops/communication/mnnvl_cutedsl_ar.py:96-150
（MNNVLCuteDSLAllReduceFusionWorkspace::__init__）。它会构建大量 CuTe kernel，
但只在 Qwen3.5/3.8 等匹配模型并显式打开时才进入该路径；默认 DSV4 配置不受其影响。

python/pyproject.toml 的硬依赖变化为：

- :31 compressed-tensors==0.18.0；
- :42 flashinfer_python[cu13]==0.6.18；
- :44 humming-kernels[cu13]==0.1.12；
- :77 sgl-deep-ep==0.1.2；
- :78 sgl-deep-gemm==0.1.7；
- :83 tilelang==0.1.12；
- :85 tokenizers==0.22.2；
- build-system :1-8 新增 setuptools-rust>=1.11、torch==2.13.0。

python/setup.py:1-27（<top-level>）、:56-109（_cargo_metadata/
_cargo_workspace_metadata）、:131-162（_discovered_rust_extensions）现在自动
发现并构建 Cargo/PyO3 扩展。editable install 仍指向源码，但 Rust extension、
PyTorch ABI、FlashInfer/DeepGEMM wheel 必须一起重建或核对；只切换 branch 不会
自动同步这些二进制依赖。

### 10.9 升级重点、兼容性与迁移建议

本节只总结 `release/v0.5.18` 到 `release/v0.5.19` 的代码、依赖和接口差异。

#### 10.9.1 影响优先级

| 优先级 | v0.5.19 的变化 | 影响范围 |
| --- | --- | --- |
| P0 | DSV4/DSA 新 attention 路径、DeepGEMM 0.1.7、DeepEPv2、FP4 runner 选择 | DSV4、FP4 MoE、跨 GPU 专家并行的默认/可选执行路径 |
| P0 | `ServerArgs` resolution pipeline、runtime config bags、统一 KV memory/tree | 自定义 scheduler、attention、PD/HiCache 和 out-of-tree 集成 |
| P1 | speculative mixed chunk、adaptive draft、TP sync、FullCG/PP、graph-pool borrowing | EAGLE/DFlash/DSpark 的 shape、同步和 CUDA graph 资源 |
| P1 | FlashInfer/CuTe DSL、AMD/XPU/NPU/CPU 后端扩展 | 硬件条件分支、native extension 和 JIT 编译输入 |
| P2 | Rust server、EPD encoder、beam search、DFlash2 selector、新模型/音频接口 | opt-in 服务端、API 和模型覆盖面 |

#### 10.9.2 需要特别审计的行为变化

1. **DSV4 FP4 的默认 runner 不再固定等同于 Marlin。**
   `python/sglang/srt/arg_groups/model_overrides/deepseek_v4.py:20-76（_deepseek_v4_overrides）`
   在 CUDA、非 HIP、SM90/100/120、FP4 experts、无 A2A、runner=auto 等条件满足时
   选择 `flashinfer_mxfp4`；显式 `--moe-runner-backend marlin` 才保持 Marlin 路径。
   同一 checkpoint 在升级前后的实际 kernel 集合可能不同，应把 runner 选择写入回归矩阵。

2. **DSV4 sparse prefill 与 DSA top-k 从单一路径扩展为可配置后端。**
   `python/sglang/srt/server_args.py:1777-1788（ServerArgs::dsv4_prefill_backend）`
   增加 `flashmla_sparse_q8`；`python/sglang/srt/layers/attention/deepseek_v4_backend.py:1933-2122`
   （`DeepseekV4AttnBackend::_prepare_q8kv8_q_and_sink`、
   `DeepseekV4AttnBackend::_forward_prefill_sparse_q8kv8`）接入 Q8KV8 kernel。
   `python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py:27-210`
   （`DSATopKBackend::resolve`、`DSATopKBackend::topk_func`、
   `DSATopKBackend::topk_transform`）增加 sgl-kernel/torch/FlashInfer 分流；默认仍是
   sgl-kernel，只有显式参数或模型 override 才改变后端。

3. **MoE 通信和量化布局发生实质扩展。**
   `python/sglang/srt/layers/moe/mega_moe.py:48-122、257-408`
   （`_mega_moe_mma_type`、`_get_mega_moe_symm_buffer`、`forward_mega_moe`、
   `build_mega_moe_experts_weights`）增加 MXFP4 MMA 类型、SM 预留和新的 expert layout。
   `python/sglang/srt/layers/moe/token_dispatcher/deepep_v2.py:136-461`
   （`DeepEPv2Buffer::get_buffer`、`_DeepEPv2Impl::dispatch`、
   `_DeepEPv2Impl::combine`、`DeepEPv2Dispatcher::dispatch/combine`）引入 ElasticBuffer
   A2A；对应 `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:1566-1722`
   增加 permute/unpermute。DeepEPv2 要求 DeepGEMM runner，不能直接和任意旧 runner 组合。
   `python/sglang/srt/layers/moe/moe_runner/runner.py:33-150`
   （`register_moe_runner_core`、`MoeRunner::__init__`）开放 runner core 注册，私有
   monkey-patch/自定义 runner 需要重新验证接口。

4. **DeepGEMM 编译契约和依赖版本升级。**
   `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py:120-260`
   （`_local_rank_compile_lock`、`_maybe_compile_deep_gemm_one_type_all`、
   `_compile_deep_gemm_one_type_all`）新增本地 rank 锁和新的 M/SM 编译组合；
   `python/sglang/compile_deep_gemm.py:63-204`
   （`warm_up_compile`、`compile_server_args`）改为先解析一次 ServerArgs 再预编译。
   `python/pyproject.toml:42-44、77-85` 把 FlashInfer、Humming、DeepEP、DeepGEMM、
   TileLang、tokenizers 等 pin 到 v0.5.19 对应版本；旧 wheel 或旧 native extension
   不能视为等价运行时。

5. **Speculative 和 CUDA graph 的资源/状态模型扩大。**
   `python/sglang/srt/speculative/spec_info.py:132-143、242-263`
   （`SpeculativeAlgorithm::supports_mixed_chunk`、
   `SpeculativeAlgorithm::resolve_max_speculative_num_draft_tokens`）支持 mixed chunk
   并把候选步数转换为运行时容量；`python/sglang/srt/speculative/spec_tp_sync.py:14-129`
   （`SpecTpSyncSite`、`SpecTpSync::sync`）增加 target/draft 等同步站点。
   `python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py:284-491`
   （`capture_prefill_graph`）和 `python/sglang/srt/model_executor/runner_utils/pool.py:45-230`
   （`graph_pool_capture_scope`、`borrow_graph_pool`）扩展 FullCG、PP proxy 和 graph-pool
   生命周期；自定义 graph shape、memory pool 或 speculative runner 必须重新检查。

6. **KV/HiCache 从分散配置转向统一所有权和字节容量模型。**
   `python/sglang/srt/managers/schedule_batch.py:848-900（ReqKvInfo）`、
   `python/sglang/srt/managers/schedule_batch.py:1330-1335（Req::effective_kv_committed_len）`
   统一请求侧 KV 元数据；`python/sglang/srt/mem_cache/unified_memory_pool.py:293-345、1810-1970`
   （`UnifiedKVPool::__init__`、`init_unified_mamba_swa_pools`）按字节预算管理多种 view。
   unified radix tree 成为默认，Rust TreeCore 注册于
   `python/sglang/srt/mem_cache/unified_cache/tree_core_registry.py:57-75`
   （`create_tree_core`）；external linker/attachment 位于
   `python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py:51-112`
   （`UnifiedCacheLinker`）和 `storage_attachment.py:37-165`。

7. **配置读取语义和可选 Rust 前端改变。**
   `python/sglang/srt/server_args.py:3663-3719`
   （`ServerArgs::__post_init__`、`ServerArgs::resolve_once`、`ServerArgs::resolved_dict`）
   将原始参数和解析结果分开；`python/sglang/srt/arg_groups/pipeline.py:26-110、340-370`
   （`run_resolution_pipeline`）及 `python/sglang/srt/runtime_context.py:798-1060`
   （`_build_config_bags`、`RuntimeContext::set_server_args`、`RuntimeContext::override`）
   负责运行时 bags。扩展代码不应继续假设直接写 `server_args.foo` 就能同步全部运行时状态。
   `SGLANG_RUST_SERVER` 默认仍为 false；打开后才使用
   `python/sglang/srt/rust_server/server.py:47-229（RustServer::launch、RustServer::drain）`。

8. **新增 API/模型能力有明确约束。**
   Beam search 通过 `python/sglang/srt/sampling/sampling_params.py:74-76、157-159`
   （`SamplingParams::beam_width`、`SamplingParams::verify`）和
   `python/sglang/srt/beam_search/beam_group.py:60-160（BeamGroup::__init__、BeamGroup::advance_frontier）`
  接入，但不支持 speculative、PD、page_size>1、DP/PP attention、HiCache、LoRA 等组合。
   请求 fan-out 和 `n<=beam_width` 语义由 `python/sglang/srt/managers/io_struct.py:473-503（GenerateReqInput::_sampling_params_beam_width、GenerateReqInput::_handle_beam_search_parallel_sampling）` 处理；
   它不是普通的 `sampling n`，每个 beam 会占用独立的 req-to-token row。
   DFlash2 candidate selector 位于 `python/sglang/srt/models/dflash.py:944-1140`
   （`CandidateSelector::build_lattice`、`DFlash2DraftModel::compute_candidates`）；
   expert-pack、EPD encoder 和新的音频/模型适配器则改变了工具入口与模型注册表，旧的
   `tools.expert_pack` 等 out-of-tree import 需要迁移到 package 内路径。

9. **OpenAI/多模态协议和服务接口扩展。**
   `python/sglang/srt/entrypoints/openai/protocol.py:355-449`
   （`CompletionRequest::return_spec_tokens_details`、`ChatCompletionRequest::return_spec_tokens_details`、`SpecTokensDetails`、`SglExt`）
   配合 `python/sglang/srt/entrypoints/openai/utils.py:158-192`
   （`spec_tokens_details_from_meta_info`、`process_spec_tokens_details_from_ret`）
   支持把 speculative 接受率/长度等统计返回到 OpenAI chat/completions。
   同文件 :586-628（`ChatCompletionMessageContentInputAudio`、
   `ChatCompletionMessageContentAudioInlinePart`、`_to_audio_url_part`）接受 inline
   base64 音频并统一为 data URI；`python/sglang/srt/entrypoints/openai/transcription_adapters/base.py:13-186`
   （`TranscriptionAdapter`、`register_transcription_adapter`、`resolve_adapter`）和
   `python/sglang/srt/entrypoints/openai/serving_transcription.py:65-129`（`OpenAIServingTranscription::create_transcription`）
   引入可注册 ASR adapter。
   `python/sglang/srt/utils/msgpack_utils.py:22-221`（`_pack_ext`、`enc_hook`、`dec_hook`、
   `ext_hook`）及 `python/sglang/srt/managers/io_struct.py:2460-2476`
   （`msgpack_encode`、`msgpack_decode`、`sock_send`、`sock_recv`）增加 tensor/SHM/CUDA-IPC
   的 msgpack 传输；这会影响多模态 worker 和自定义 IPC 结构。
   另外，`python/sglang/srt/sampling/sampling_params.py:38、220-260（SamplingParams::normalize）` 增加 stop/regex
   数量和长度上限，`python/sglang/srt/entrypoints/openai/serving_tokenize.py:39（OpenAIServingTokenize::_handle_non_streaming_request）`
   修正 unlimited tokenizer context；旧客户端/扩展应检查错误码和返回字段。
   HTTP/2 并发窗口也变为可配置：commit `3904b309db` 在
   `python/sglang/srt/server_args.py:1273-1288（ServerArgs::enable_http2、ServerArgs::http2_max_concurrent_streams、ServerArgs::http2_initial_connection_window_size）` 声明参数，
   `python/sglang/srt/entrypoints/http_server.py:2449-2489`（HTTP server startup config）将其传给 Granian。

10. **调度观测接口。**
    `python/sglang/srt/disaggregation/kv_events.py:139（resolve_load_pub_range）`、
    `python/sglang/srt/managers/scheduler_components/load_publisher.py:114-193`
    （`SchedulerLoadPublisher`、`SchedulerLoadPublisher::publish_load_stat`）新增
    `--load-publish-endpoint`，由每个 scheduler 发布 running/waiting/token load；
    这是 router/load-aware 部署的新 opt-in 协议，不改变默认调度路径。

11. **KV-aware Router 和外部 KV indexer。**
    commit `360d10d6bc` 在 `experimental/sgl-router/sgl-kv-indexer` 增加进程内内存
    indexer，以及 ZMQ KV-event -> gRPC 的 bridge；服务端抽象在
    `experimental/sgl-router/sgl-kv-indexer/src/service.rs:45-178`
    （`KvIndexerBackend`、`KvIndexerService::new`、`KvIndexerService::into_server`），
    bridge 配置在 `experimental/sgl-router/sgl-kv-indexer/src/bridge.rs:36-78（BridgeConfig::from_env）`。
    `experimental/sgl-router/src/policies/cache_aware_zmq.rs:175-298`
    （`CacheAwareZmqPolicy::select`）可用 `--kv-indexer-endpoint` 查询最长 KV 前缀，
    再按 active load 选 worker；indexer 是 soft-state，重启不提供持久化/复制，
    因此部署和故障恢复语义与 v0.5.18 的 router-local radix tree 不同。

12. **PD/DCP 和 SSD expert-pack 路径。**
    `python/sglang/srt/layers/attention/trtllm_mla_backend.py:189-370`
    （`TRTLLMMLABackend::__init__`、`TRTLLMMLABackend::_get_dcp_local_seq_lens`、
    `TRTLLMMLABackend::_get_dcp_local_max_seq_len`）增加 TRT-LLM MLA decode context parallel；
    `python/sglang/srt/disaggregation/common/dcp_pack.py:17-120`
    （`dcp_pack_buffer_bytes`、`try_pack_dcp_src`、`init_dcp_pack_buffers`）把
    DCP1->DCP-N 的 PD KV 行打包为目标连续 RDMA block，Mooncake/NIXL 的传输接口
    随之改变。`python/sglang/srt/model_loader/expert_pack_loader.py:142-204`
    （`ExpertPackModelLoader::__init__`、`ExpertPackModelLoader::load_model`）和
    `python/sglang/srt/arg_groups/expert_pack_hook.py:32-204（handle_expert_pack）`
    新增 DeepSeek-V4/Kimi-K3 的 GGUF + manifest + SSD expert-pack；它有专用的
    并行度、CUDA graph 和 shared-expert-fusion 约束，不能当作普通 checkpoint loader。

13. **模型和 Diffusion 覆盖面扩大。**
    v0.5.19 新增/完整接入多个模型（commit `20621aa14b`、`8a1e6e4e46`、
    `dfc40e0efe`、`092d85eb87`、`af39ad9349`、`0c42a44cd7`、`046454404a`、
    `41e7612dee`、`70983bd7db`）。代表性实现及入口为：
    `python/sglang/srt/models/bailing_moe_v3.py:1299-1719`
    （`BailingMoeV3ForCausalLM`、`BailingMoeV3ForCausalLM::forward`、
    `BailingMoeV3ForCausalLM::load_weights`）；
   `python/sglang/srt/models/minicpm.py:517-563`
   （`MiniCPMSALAForCausalLM`、`MiniCPMSALAForCausalLM::forward`、
   `MiniCPMSALAForCausalLM::load_weights`）；
    sparse attention backend 在 `python/sglang/srt/layers/attention/minicpm/backend.py:122（MiniCPMSparseBackend）`；
    `python/sglang/srt/models/dots3_common/modeling.py:2719-2823`
    （`DotsNoteOmniForConditionalGeneration`、`Dots3NoteForCausalLM`）和
    `python/sglang/srt/multimodal/processors/dots_note_omni.py:179-373`
    （`DotsNoteOmniProcessor`、`DotsNoteOmniProcessor::process_mm_data_async`）；
    另有 Qwen3.8、GLM-5.3-Flash、Spark、LFM2-DSpark、Nemotron 3.5 Lightning、
    Granite SWA 等模型及 speculative 适配。
    Diffusion 侧同时增加按 component 的权重来源/精度 override、Comfy/NVFP4/INT8
    混合 checkpoint、cache-dit/offload、FLUX.2/Qwen-Image/MiniMax-H3/Cosmos3 等路径。
    这些主要是新模型和新 pipeline 能力，不应与核心 autoregressive scheduler 的行为
    变化混为一谈；但会扩大模型注册、权重格式和 native kernel 的兼容矩阵。


#### 10.9.3 升级时的最小核对清单

1. 以 `release/v0.5.19` 的 `python/pyproject.toml`、Dockerfile 和 `setup.py` 同步
   Python wheel、CUDA/ROCm 扩展、Rust/PyO3 扩展；editable install 只解决源码指向，
   不会自动替换已安装的第三方二进制包。
2. 固定并记录 `dsv4_prefill_backend`、DSA target/draft top-k backend、
   `moe_runner_backend`、`moe_a2a_backend`、speculative 算法和 HiCache/radix 设置，
   再比较功能与性能；不要用 `auto` 结果推断两个分支一定走同一 kernel。
3. 对自定义 attention/PD/LoRA/runner 适配代码，按 `ReqKvInfo`、resolution views/bags、
   `DispatchMoeRunnerCore` 和 unified linker 的新接口做编译与单元测试。
4. 按目标 GPU 检查对应 native wheel 和开关：NVIDIA 的 FlashInfer/CuTe DSL，AMD 的
   ROCm10/gfx1250，XPU/NPU/CPU 的专用量化和 attention 路径；不匹配的平台应验证
   fallback，而不是复用另一平台的 JIT/native cache。
5. 将 v0.5.18 共有的提交与真正的 v0.5.18→v0.5.19 新提交分开记录。例如
   content-addressed JIT loader `b784726863` 已在两个 release branch 中存在，
   不能把它列为 v0.5.19 独有改动；真正的新依赖和上述新后端则应在升级记录中单独 pin。

综上，v0.5.19 的核心变化不是某一个 kernel 的小修补，而是 DSV4/DSA attention、
DeepGEMM/MegaMoE、DeepEPv2、speculative graph、统一 KV/cache、配置解析和多平台
构建链的同步扩展。对现有部署最可能产生行为变化的两个开关是 DSV4 FP4 的 auto
runner 选择和 DeepEPv2/MoE runner 组合；对现有扩展最需要改代码的是 ServerArgs
resolution/bags、ReqKvInfo 和 runner core 接口。
