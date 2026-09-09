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

python/pyproject.toml 的 v0.5.18 -> v0.5.19 依赖差异为：

- `python/pyproject.toml:25 -> :31`，`compressed-tensors` 从不固定改为 `==0.18.0`；
- `python/pyproject.toml:36 -> :42`，`flashinfer_python[cu13]` `0.6.17 -> 0.6.18`；
- `python/pyproject.toml:38 -> :44`，`humming-kernels[cu13]` `0.1.10 -> 0.1.12`；
- `python/pyproject.toml:71 -> :77`，`sgl-deep-ep` `0.1.0 -> 0.1.2`；
- `python/pyproject.toml:72 -> :78`，`sgl-deep-gemm` `0.1.5.post3 -> 0.1.7`；
- `python/pyproject.toml:77 -> :83`，`tilelang` `0.1.11 -> 0.1.12`；
- `python/pyproject.toml:85`（v0.5.19）新增 `tokenizers==0.22.2`；
- `python/pyproject.toml:2 -> :2-8`（build-system），`setuptools-rust` `>=1.10 -> >=1.11`，并新增 build-time `torch==2.13.0`。

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
   Beam search 通过 `python/sglang/srt/sampling/sampling_params.py:74-76、157-159`
   （`SamplingParams::beam_width`、`SamplingParams::verify`）和
   `python/sglang/srt/beam_search/beam_group.py:60-160`
   （`BeamGroup::__init__`、`BeamGroup::advance_frontier`）
  接入，但不支持 speculative、PD、page_size>1、DP/PP attention、HiCache、LoRA 等组合。
   请求 fan-out 和 `n<=beam_width` 语义由 `python/sglang/srt/managers/io_struct.py:473-503`（
   `GenerateReqInput::_sampling_params_beam_width`、`GenerateReqInput::_handle_beam_search_parallel_sampling`）处理；
   它不是普通的 `sampling n`，每个 beam 会占用独立的 req-to-token row。
   DFlash2 candidate selector 位于 `python/sglang/srt/models/dflash.py:944-1140`
   （`CandidateSelector::build_lattice`、`DFlash2DraftModel::compute_candidates`）；
   expert-pack、EPD encoder 和新的音频/模型适配器则改变了工具入口与模型注册表，旧的
   `tools.expert_pack` 等 out-of-tree import 需要迁移到 package 内路径。

9. **OpenAI/多模态协议和服务接口扩展。**
   `python/sglang/srt/entrypoints/openai/protocol.py:355-356、418-449、863-864`
   （`CompletionRequest::return_spec_tokens_details`、`ChatCompletionRequest::return_spec_tokens_details`、`SpecTokensDetails`、`SglExt`）
   配合 `python/sglang/srt/entrypoints/openai/utils.py:158-192`
   （`spec_tokens_details_from_meta_info`、`process_spec_tokens_details_from_ret`）
   支持把 speculative 接受率/长度等统计返回到 OpenAI chat/completions。
   `python/sglang/srt/entrypoints/openai/protocol.py:586-634`（`ChatCompletionMessageContentInputAudio`、
   `ChatCompletionMessageContentAudioInlinePart`、`_to_audio_url_part`）接受 inline
   base64 音频并统一为 data URI；`python/sglang/srt/entrypoints/openai/transcription_adapters/base.py:13-186`
   （`TranscriptionAdapter`、`register_transcription_adapter`、`resolve_adapter`）和
   `python/sglang/srt/entrypoints/openai/serving_transcription.py:65-129`（`OpenAIServingTranscription::create_transcription`）
   引入可注册 ASR adapter。
   `python/sglang/srt/utils/msgpack_utils.py:22-247`（`_pack_ext`、`enc_hook`、`dec_hook`、
   `ext_hook`）及 `python/sglang/srt/managers/io_struct.py:2417-2497`
   （`_msgpack_encoder`、`_msgpack_decoder`、`msgpack_encode`、`msgpack_decode`、`sock_send`、`sock_recv`）增加 tensor/SHM/CUDA-IPC
   的 msgpack 传输；这会影响多模态 worker 和自定义 IPC 结构。
   另外，`python/sglang/srt/sampling/sampling_params.py:38、220-260（SamplingParams::normalize）` 增加 stop/regex
   数量和长度上限，`python/sglang/srt/entrypoints/openai/serving_tokenize.py:39（OpenAIServingTokenize::_handle_non_streaming_request）`
   修正 unlimited tokenizer context；旧客户端/扩展应检查错误码和返回字段。
   HTTP/2 并发窗口也变为可配置：commit `3904b309db` 在
   `python/sglang/srt/server_args.py:1273-1288（ServerArgs::enable_http2、ServerArgs::http2_max_concurrent_streams、ServerArgs::http2_initial_connection_window_size）` 声明参数，
   `python/sglang/srt/entrypoints/http_server.py:2449-2489`（HTTP server startup config）将其传给 Granian。

10. **调度观测接口。**
    commit `97ba99067d`： `python/sglang/srt/disaggregation/kv_events.py:139-220（resolve_load_pub_range）`、
    `python/sglang/srt/managers/scheduler_components/load_publisher.py:114-264`
    （`SchedulerLoadPublisher`、`SchedulerLoadPublisher::publish_load_stat`、`SchedulerLoadPublisher::close`）新增
    `--load-publish-endpoint`，并在 `python/sglang/srt/managers/scheduler.py:2178、4151` 接入；每个 scheduler 发布 running/waiting/token load；
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
5. 将 merge-base 之前的共有基础改动与真正的 v0.5.18→v0.5.19 新提交分开记录，
   以 `git diff release/v0.5.18..release/v0.5.19` 及 v0.5.19 的依赖清单为准；
   新依赖、新后端和接口迁移应在升级记录中单独 pin/标注。

综上，v0.5.19 的核心变化不是某一个 kernel 的小修补，而是 DSV4/DSA attention、
DeepGEMM/MegaMoE、DeepEPv2、speculative graph、统一 KV/cache、配置解析和多平台
构建链的同步扩展。对现有部署最可能产生行为变化的两个开关是 DSV4 FP4 的 auto
runner 选择和 DeepEPv2/MoE runner 组合；对现有扩展最需要改代码的是 ServerArgs
resolution/bags、ReqKvInfo 和 runner core 接口。

## 11. `sglang_sipu` 相对 `v0.5.18` 的 SIPU 适配

### 11.1 比较范围和总览

本节只比较 `/share/users/like/package/sglang_sipu` 中的开源 tag `v0.5.18`
和当前 HEAD，不把上一个问题中的慢启动日志作为差异来源。`v0.5.18` 是
annotated tag，实际代码提交为 `v0.5.18^{commit}=71de97b264b04dcd514cf904003028aefe9775c8`；
当前 HEAD 为 `7a3d9d7137d0dd8da8334b0d12759d721d38fa33`（分支
`v0.5.18-sipu-dev`）。执行 `git diff v0.5.18..HEAD` 的结果是 **254 个文件，
新增 20,777 行，删除 736 行**，因此 HEAD 是一个在 v0.5.18 上叠加的 SIPU
适配 fork，并不是上游 `release/v0.5.19` 的普通升级。

| 提交 | 主要内容 | 变更规模 |
|---|---|---:|
| `91462e29` | 首次加入 SIPU device、attention、DSA、MoE、通信、加载器和大量算子分支 | 246 files, +19,767/-747 |
| `e9adcef3` | SIPU 使用文档、支持模型和测试说明 | 8 files, +684/-448 |
| `770f80bc` | DeepSeek-V4 清理、e2e/精度对齐；删除早期临时 workaround | 17 files, +54/-339 |
| `a8eb80fc` | SIPU CI、Docker 和 `sgl-kernel-sipu` 子模块指针 | 10 files, +1,118/-63 |
| `7a3d9d71` | CI 的 `ci_logs` 挂载修复 | 1 file, +21/-6 |

最重要的判断是：这批代码没有只增加一个 `device` 字符串，而是把
SGLang 的设备探测、算子分派、KV cache、DeepSeek attention、MoE、通信、权重加载
和测试/镜像链路全部接上 SIPU。另一方面，它也没有实现一个完整的
`SipuSRTPlatform` 插件；核心方式是使用 `torch_sipu` 注册的 PyTorch
`PrivateUse1` 后端，再在 SGLang 的关键热路径中显式走 `is_sipu()` 分支。

以下 `python/...` 路径均相对于 `sglang_sipu` 根目录；文中出现的
`sgl-kernel-sipu/...` 路径相对于容器内 `/sgl-workspace/sgl-kernel-sipu`，它是
独立的 kernel 工程/二进制，不是 SGLang Python 包自身编译出来的代码。

### 11.2 `device="sipu"` 是怎样进入执行链的

实际链路可以概括为：

```text
temp/env-offlie-infer.sh
  -> sgl-kernel-sipu/setup.sh + SDK/CModel/torch_sipu 环境
  -> torch_sipu 将 PrivateUse1 命名为 sipu，提供 torch.sipu
  -> sgl.Engine(device="sipu")
  -> DeviceConfig / ServerArgs 解析和后端校正
  -> is_sipu() + fused-op platform dispatch
  -> attention/MoE/communicator 等 SIPU 实现
  -> sgl_kernel、torch_sipu，必要时 CPU fallback
```

1. **容器和 editable 源码。** `docker inspect sipu-dev` 显示该容器使用镜像
   `harbor.siorigin.com/sglang-sipu/release:v0.5.18-sipu-dev-0.1.0`，并将宿主机
   `/share/users/like/package/sglang_sipu` 以读写方式挂载到容器
   `/sgl-workspace/sglang`，所以容器内 editable 安装实际指向
   `/sgl-workspace/sglang/python`；import 名和发行包名仍为 `sglang`，修改宿主机源码
   会直接影响容器运行。关键挂载如下：

   | 宿主机路径 | 容器路径 | 模式 |
   |---|---|---|
   | `/share/users/like/package/sglang_sipu` | `/sgl-workspace/sglang` | rw |
   | `/share_data/inference-framework/tiny-models` | `/share_data/inference-framework/tiny-models` | ro |
   | `/share_data/sicx_sdk` | `/share_data/sicx_sdk` | ro |
   | `/share_data/sicx_sdk/release` | `/share_data/sicx_sdk/release` | ro |
   | `/share_data/arch_cmodel_release` | `/share_data/arch_cmodel_release` | ro |
   | `/share_data/torch_sipu` | `/share_data/torch_sipu` | ro |
   | `/share_data/sglang_sipu` | `/share_data/sglang_sipu` | rw |
   | `/data_gpu` | `/data_gpu` | rw |
   | `/share` | `/share` | rw |
   | `/share2` | `/share2` | rw |
   | `/softhome` | `/softhome` | rw |

   SICX SDK、CModel、`torch_sipu` 和 tiny model 是只读挂载；
   `/sgl-workspace/sgl-kernel-sipu` 是镜像内预置目录，不在 mount 列表中，也不会随
   SGLang git checkout 自动重编译。

2. **环境初始化。** `temp/env-offlie-infer.sh:1-4（顶层脚本）` source
   `sgl-kernel-sipu/setup.sh`，设置 SIPU 动态库路径、SDK/CModel 和 `SGL_KERNEL_LOG`，
   并把 SGLang `python` 放到 `PYTHONPATH`。容器默认
   `TORCH_DEVICE_BACKEND_AUTOLOAD=0`；脚本将其 unset，允许
   `torch_sipu:_autoload` 被 PyTorch 调用。外部
   `sgl-kernel-sipu/setup.sh:3-38（顶层脚本）` 负责库路径和依赖检查，
   `sgl-kernel-sipu/scripts/source_sipu_env.sh:4-29（顶层脚本）` 默认选择
   `SIPU_ARCH=150` 并装载 SDK/CModel。实测环境为
   `SI_SDK_ROOT=/share_data/sicx_sdk/release/2608282232`、
   `SI_CMODEL_ROOT=/share_data/arch_cmodel_release/sipu1.5/2608270400`。
   这一步必须发生在首次 import SGLang/相关 backend 之前，因为
   `is_sipu()` 使用 `lru_cache`，多个模块还会在 import 时保存 `_is_sipu`。

3. **PyTorch 设备名。** 这里的 `torch.sipu` 来自已安装的 `torch_sipu`，不是
   SGLang 自己实现的 `torch.Device`。`python/sglang/srt/utils/common.py:193-203（is_sipu）`
   缓存检查 `torch.sipu` 是否存在且可用；
   `python/sglang/srt/configs/device_config.py:10-23（DeviceConfig::__init__）`
   把 `sipu` 加入支持列表并构造 `torch.device("sipu")`；
   `python/sglang/srt/utils/common.py:495-509（get_available_gpu_memory）`、
   `python/sglang/srt/utils/common.py:764-770（get_sipu_memory_capacity）`、
   `python/sglang/srt/utils/common.py:854-870（get_device_memory_capacity）`
   改用 `torch.sipu.device_count/current_device/mem_get_info`；
   `python/sglang/srt/utils/common.py:916-945（get_device）` 在显式 SIPU 可用时返回
   `sipu` 或 `sipu:<id>`。SGLang 自定义算子还由
   `python/sglang/srt/utils/common.py:2848-2858（direct_register_custom_op）`
   注册到 PyTorch `PrivateUse1` dispatch key；因此算子可以在不改写每个
   `torch.ops` 调用点的情况下落到 `torch_sipu`。

4. **通用算子分派。** `python/sglang/srt/platforms/device_mixin.py:38-55（PlatformEnum）`
   增加 `SIPU` 枚举，
   `python/sglang/srt/platforms/device_mixin.py:85-93（_DEVICE_TO_DISTRIBUTED_BACKEND）` 根据
   `SGLANG_SIPU_USE_SICCL` 选择 `siccl` 或 `gloo`，
   `python/sglang/srt/platforms/device_mixin.py:131-132（DeviceMixin::is_sipu）`
   提供平台谓词。`python/sglang/kernels/fused_op.py:147-155（_PLATFORM_METHODS）`
   为 SIPU 定义 `forward_sipu`，并在 `python/sglang/kernels/fused_op.py:163-198（_platform_key）`
   中**先于 CUDA**检查
   SIPU，因为某些 `torch_sipu` 构建会让 `torch.cuda.is_available()` 得到误导性的结果；
   `python/sglang/kernels/fused_op.py:523-567（BaseFusedOp::_platform_method、BaseFusedOp::_resolve_forward_method）`
   先找 `forward_sipu`，再退回 `forward_cuda` 或 native 实现。

5. **ServerArgs 和 graph 约束。**
   `python/sglang/srt/server_args.py:3664-3675（ServerArgs::_run_resolution_pipeline）`
   将 SIPU 校正纳入参数解析；`python/sglang/srt/server_args.py:4390-4402（ServerArgs::_handle_sipu_backends）`
   调用 `python/sglang/srt/hardware_backend/sipu/utils.py:28-33（set_default_server_args）`，
   HEAD 最终只在未指定时设置 `page_size=32`，并把 SIPU 的 prefill compiler 强制为
   `eager`。`python/sglang/srt/server_args.py:4614-4635（ServerArgs::_disable_tc_piecewise_cudagraph_if_incompatible）`
   将 SIPU 列为不支持 tc-piecewise CUDA graph 的设备。不要把首个提交中曾出现、
   后来由 `770f80bc` 删除的内存比例、特殊 graph 或 custom-allreduce 默认值当作当前
   默认行为。

6. **为什么看不到 SIPU platform 插件也能运行。** 此 fork 没有新增
   `sglang.srt.platforms` entry point 或 `SipuSRTPlatform` 类，容器中
   `current_platform` 仍可能是基础 `SRTPlatform(device=unknown)`。这是因为真正的
   kernel 选择发生在上面的 `is_sipu()`、`DeviceConfig`、`BaseFusedOp` 和各模块的
   SIPU 分支，而不是依赖 out-of-tree platform resolver。运行时应显式传
   `device="sipu"`（以及需要时 `attention_backend="sipu"`），不要依赖自动猜测。

### 11.3 主要新增功能

#### 11.3.1 Attention、KV cache 和 DeepSeek/DSA

- **SIPU attention backend 注册。**
  `python/sglang/srt/layers/attention/attention_registry.py:133-150（create_sipu_backend）`
  扫描模型是否有 `indexer`：普通模型使用
  `SIPUAttnBackend`，DSA 模型使用 `SIPUDSAAttnBackend`；
  `python/sglang/srt/layers/attention/attention_registry.py:153-163（create_dsa_backend）`
  在 `is_sipu()` 时把通用 DSA 请求改到 SIPU 实现。
  `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py:4-47（AttnForwardMethod）`
  增加 `MHA_SIPU`、`MLA_SIPU`；
  `python/sglang/srt/models/deepseek_common/attention_backend_handler.py:83-98（handle_attention_sipu）`
  按 dense MHA、DSA、extend/verify 阶段选择方法，
  `python/sglang/srt/models/deepseek_common/attention_backend_handler.py:251-266（AttentionBackendRegistry::register）`
  完成注册。

- **DeepSeek MHA/MLA。**
  `python/sglang/srt/models/deepseek_v2.py:2014-2041（DeepseekV2AttentionMLA::dispatch_attn_forward_method）`、
  `python/sglang/srt/models/deepseek_v2.py:2082-2211（DeepseekV2AttentionMLA::forward_prepare）` 和
  `python/sglang/srt/models/deepseek_v2.py:2213-2250（DeepseekV2AttentionMLA::forward_core）`
  增加 SIPU prepare/core 分支。
  具体实现位于
  `python/sglang/srt/hardware_backend/sipu/modules/deepseek_v2_attention_sipu.py:28-95（forward_mha_prepare_sipu）`、
  `python/sglang/srt/hardware_backend/sipu/modules/deepseek_v2_attention_sipu.py:98-112（forward_mha_core_sipu）`、
  `python/sglang/srt/hardware_backend/sipu/modules/deepseek_v2_attention_sipu.py:115-193（forward_mla_prepare_sipu）`、
  `python/sglang/srt/hardware_backend/sipu/modules/deepseek_v2_attention_sipu.py:196-289（forward_mla_core_sipu）`。
  其中包含 QKV/latent 准备、可选 DSA indexer、
  absorbed MLA 的 `torch.bmm` 路径和 SIPU 输出投影；不支持的 DeepGEMM BMM、部分 FP8
  `w_kc/w_vc` 路径会明确抛出 `NotImplementedError`，避免静默走错误 kernel。

- **Paged FlashAttention 和 speculative decode。**
  `python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:42-93（SIPUAttentionMetadata）`
  定义序列长度、page table、累计长度、窗口和 encoder metadata；
  `python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:95-246（SIPUAttnBackend::__init__）`
  初始化 MHA/MLA、speculative、SWA、cascade 状态；
  `python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:384-796（SIPUAttnBackend::init_forward_metadata）`
  构造 paged metadata；
  `python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:798-1295（SIPUAttnBackend::forward_extend）`
  处理 KV 写入和 prefill，
  `python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:1297-1607（SIPUAttnBackend::forward_decode）`
  处理 paged MHA/MLA decode。
  底层调用 `sgl_kernel.flash_attn_varlen_func`、`flash_attn_with_kvcache` 和
  `merge_state_v2`，并支持部分 cascade、context-parallel、SWA 和 speculative 状态。
  该 backend 的注释明确说明 extend/draft-extend 不做 graph，graph 主要限于 decode/target
  verify；page table 还额外保留对齐空间，因为 SIPU kernel 会读取对齐 page。
  具体的 `python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:248-257（SIPUAttnBackend::_page_table_width）`
  将宽度临时扩展 `128`，避免 paged kernel 按 `Chunk_Bc/page_size` 读取时越界。

- **DSA/FlashMLA/indexer。**
  `python/sglang/srt/hardware_backend/sipu/attention/sipu_dsa_backend.py:244-305（DeepseekSparseAttnBackend::__init__）`
  和 `python/sglang/srt/hardware_backend/sipu/attention/sipu_dsa_backend.py:362-624（DeepseekSparseAttnBackend::init_forward_metadata）`
  构造 DSA metadata、
  real/page table 和 FlashMLA schedule；
  `python/sglang/srt/hardware_backend/sipu/attention/sipu_dsa_backend.py:1151（DeepseekSparseAttnBackend::forward_extend）`
  与 `python/sglang/srt/hardware_backend/sipu/attention/sipu_dsa_backend.py:1250（DeepseekSparseAttnBackend::forward_decode）`
  分别覆盖 prefill/decode；
  `python/sglang/srt/hardware_backend/sipu/attention/sipu_dsa_backend.py:1336-1396（DeepseekSparseAttnBackend::_forward_flashmla_kv）` 调用
  `sgl_kernel.flash_mla_with_kvcache`。当前 SIPU FlashMLA-KV 要求 FP8 DSA KV cache、
  real page size 64，并把 query head 补齐到 kernel 支持的 64/128 变体；短 prefill
  由 `python/sglang/srt/hardware_backend/sipu/attention/sipu_dsa_backend.py:1458-1482（DeepseekSparseAttnBackend::set_dsa_prefill_impl）` 选择 dense MHA
  以降低开销。文件末尾
  `python/sglang/srt/hardware_backend/sipu/attention/sipu_dsa_backend.py:1618-1622（SIPUDSAAttnBackend 导出别名）`
  将该实现导出为 `SIPUDSAAttnBackend`。

- **DSA indexer 和 top-k。**
  `python/sglang/srt/hardware_backend/sipu/attention/dsa_sipu_indexer.py:409-474（DSASIPUIndexerMixin::forward_sipu）`
  使用 `sgl_kernel.act_quant_triton` 做 FP8 query/index-K cache，按 paged/ragged
  MQA 计算 logits 和 top-k；
  `python/sglang/srt/hardware_backend/sipu/attention/dsa_sipu_indexer.py:108-135（_store_index_k_cache）`
  负责量化存 cache，
  `python/sglang/srt/hardware_backend/sipu/attention/dsa_sipu_indexer.py:248-406（_get_topk_paged、_get_topk_ragged）`
  负责两种布局。相应的
  `python/sglang/srt/layers/attention/dsa/dsa_indexer.py:203（Indexer::__init__）`
  把 SIPU mixin 放入 indexer MRO。`python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py:146-151（DSATopKBackend::should_use_topk_v2）`
  禁用 CUDA-only topk-v2；
  `python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py:206-237（DSATopKBackend::topk_transform）`
  对不满足 SIPU
  fused kernel 形状的 top-k 转到 CPU Torch，支持的形状才调用
  `fast_topk_transform_fused`。

- **DSV4 低层 wrapper。**
  DeepSeek-V4 的公共 Python 逻辑保留在 SGLang，设备相关操作在以下函数转发到
  `sgl_kernel`：`python/sglang/kernels/ops/attention/dsv4/compress.py:417（compress_forward）`、
  `python/sglang/kernels/ops/attention/dsv4/elementwise.py:119（fused_rope_inplace）`、
  `python/sglang/kernels/ops/attention/dsv4/gemm.py:145（linear_bf16_fp32）`、
  `python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py:196（init_compression_metadata）`、
  `python/sglang/kernels/ops/attention/dsv4/moe.py:113（hash_topk）` 和
  `python/sglang/kernels/ops/attention/dsv4/topk.py:49-68（topk_transform_512）`。
  `python/sglang/srt/arg_groups/overrides.py:1158-1163（_deepseek_v4_overrides）` 对
  `device="sipu"` 保留 prefill/decode 的 `dsv4` backend，不把 DSV4 错误地降成普通
  `sipu` DSA/MHA backend，并沿用该函数默认的 DSV4 `page_size=256`；普通 SIPU
  模型的 helper 默认值仍是 `page_size=32`。
  DSV4 自己的频率表由 `python/sglang/srt/models/deepseek_v4.py:643-652（MqaAttentionBase::__init__）`
  按 `config.max_position_embeddings` 计算后移动到 SIPU；这与通用 RotaryEmbedding
  仍在 CPU 初始化的 workaround 是两条不同路径。
  当启用 TileLang MHC 时，`python/sglang/srt/models/deepseek_v4.py:1715-1810（DeepseekV4DecoderLayer::hc_pre）`
  和 `python/sglang/srt/models/deepseek_v4.py:1850-1876（DeepseekV4DecoderLayer::hc_post）`
  会把 SIPU 的 MHC pre/post 改调用 `sgl_kernel.mhc_pre_tilelang`/
  `mhc_post_tilelang`。

#### 11.3.2 MoE、DeepGEMM 和 DeepEP

- **SIPU DeepGEMM masked 路径。**
  `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:249-299（DeepGemmRunnerCore::run）`
  将 SIPU 限定为 DeepEP low-latency 的 masked GEMM；normal/contiguous 模式和 BF16
  masked 模式会明确拒绝。
  `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:777-787（DeepGemmRunnerCore::_run_masked_gemm_sipu）`
  转到 `python/sglang/srt/hardware_backend/sipu/moe/deep_gemm_sipu.py`；
  `python/sglang/srt/hardware_backend/sipu/moe/deep_gemm_sipu.py:32-53（_prepare_fp8_hidden）`
  做 group-128 per-token FP8 quant，
  `python/sglang/srt/hardware_backend/sipu/moe/deep_gemm_sipu.py:56-85（_sipu_grouped_gemm_nt_f8f8bf16_masked）`
  调用 `torch._scaled_grouped_mm(..., gemm_type="grouped_masked")`，
  `python/sglang/srt/hardware_backend/sipu/moe/deep_gemm_sipu.py:88-184（run_masked_gemm_sipu）`
  串起 gate/up GEMM、SILU/post-quant 和 down GEMM。
  因此这里不是复用 NVIDIA `deep_gemm` 的 CUDA kernel，而是使用 SIPU 的 grouped-mm
  能力；能否执行仍取决于权重布局和 external kernel/torch_sipu 版本。

- **普通 fused MoE。**
  `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py:387-391（_moe_support_tma）`
  对 SIPU 关闭 TMA；
  `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py:394-477（_prepare_fused_moe_run）`
  使用固定的 SIPU 配置，
  `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py:480-653（_fused_moe_kernel_sequence）`
  调用 `sgl_kernel.invoke_fused_moe_kernel`。配置由
  `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py:180-187（get_sipu_default_moe_config）` 固定为
  `BLOCK_SIZE_M=16、BLOCK_SIZE_N=32、BLOCK_SIZE_K=32`，大模型 expert 数量时
  `GROUP_SIZE_M=4`。`python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py:35-201（moe_align_block_size）`
  也改为 SIPU kernel 的对齐实现，避免直接调用会挂起的 Triton kernel。

- **DeepEP 通信/dispatch。**
  `python/sglang/srt/layers/moe/token_dispatcher/deepep.py:166-172（_use_sipu_custom_deepep）`
  只有 `device=sipu` 且 `SGLANG_SIPU_USE_CUSTOM_DEEPEP=1` 时才启用自定义
  `sipu_deep_ep.Buffer`；`python/sglang/srt/layers/moe/token_dispatcher/deepep.py:173-240（DeepEPBuffer::get_deepep_buffer）`
  默认仍使用 bundled `sgl-kernel-sipu/deepep` 的 CPU/Gloo fallback。
  `python/sglang/srt/layers/moe/token_dispatcher/deepep.py:818-866（_DeepEPDispatcherImplLowLatency::_dispatch_core）`
  和 `python/sglang/srt/layers/moe/token_dispatcher/deepep.py:894-956（_DeepEPDispatcherImplLowLatency::_combine_core）` 对自定义 SIPU
  实现删减 CUDA/NPU 专用 overlap 参数并处理 hidden padding。因此代码具备 DeepEP
  接口适配，但默认配置不是设备内高速 A2A。

- **路由 top-k 的临时准确率 workaround。**
  `python/sglang/srt/layers/moe/topk.py:132-210（_fake_fixed_expert_ids、fake_topk_softmax、
  fake_topk_sigmoid、fake_unified_grouped_topk）` 固定 token 到 expert 的模式，注释说明
  CModel 与 CUDA 的 router logits 在近似并列时会漂移。
  `python/sglang/srt/layers/moe/topk.py:941-1003（fused_topk）` 和
  `python/sglang/srt/layers/moe/topk.py:2167-2179（select_experts）` 的 fallback 会调用这些函数。
  它的目的是让 SIPU/CUDA
  小模型 dump 比较 GEMM，而不是提供真实生产路由；部署真实 MoE 前必须确认是否已替换
  该 workaround。

#### 11.3.3 多卡通信

- `python/sglang/srt/distributed/parallel_state.py:280-334（GroupCoordinator::__init__）`
  为 SIPU 选择 `torch.device("sipu:<local_rank>")`；
  `python/sglang/srt/distributed/parallel_state.py:418-543（GroupCoordinator::__init__，communicator setup）`
  创建 `SipuCommunicator`。
  `python/sglang/srt/distributed/parallel_state.py:661-719（GroupCoordinator::all_reduce）`
  的顺序是 custom all-reduce、SIPU communicator、通用 fallback。
- `python/sglang/srt/distributed/device_communicators/sipu_communicator.py:9-41（SipuCommunicator::__init__、
  SipuCommunicator::all_reduce、SipuCommunicator::all_gather）` 在默认情况下把 tensor
  拷到 CPU 后走 Gloo，再拷回 SIPU；`SGLANG_SIPU_USE_SICCL=1` 才直接在设备上走 SiCCL。
  `python/sglang/srt/distributed/parallel_state.py:1094-1124（GroupCoordinator::reduce_scatter_tensor）` 和
  `python/sglang/srt/distributed/parallel_state.py:1238-1289（GroupCoordinator::_all_gather_into_tensor）`
  对 reduce-scatter/all-gather 采用相同策略；
  `python/sglang/srt/distributed/parallel_state.py:1932-1973（init_model_parallel_group）`
  禁用 PyNccl 并启用 SIPU communicator。
- `python/sglang/srt/distributed/device_communicators/sipu_custom_all_reduce.py:14-123（SipuCustomAllreduce::__init__、
  SipuCustomAllreduce::should_custom_ar、SipuCustomAllreduce::custom_all_reduce）`
  增加可选的设备内 custom all-reduce，支持 world size 2/4/6/8、连续的 fp32/fp16/bf16
  和大小/对齐限制；只有 `SGLANG_SIPU_USE_CUSTOM_ALLREDUCE=1` 才初始化。对应环境
  开关定义在 `python/sglang/srt/environ.py:847-858（Envs）`，三个 SIPU 通信开关默认
  都是 false。

#### 11.3.4 FP8/量化、加载和内存管理

- **FP8 linear。** `python/sglang/srt/layers/quantization/fp8_utils.py:748-751（_dispatch_auto_backend）`
  在 auto backend 中优先选择 SIPU；
  通过 `sglang_per_token_group_quant_fp8` 做输入量化，再调用
  `torch._scaled_grouped_mm`，输出 bf16。
  `python/sglang/srt/layers/quantization/fp8_utils.py:1140-1200（sipu_w8a8_block_fp8_linear）`、
  `python/sglang/kernels/ops/quantization/fp8_kernel.py:527-572（_run_per_token_group_quant_8bit_kernel）`
  和 `python/sglang/kernels/ops/quantization/fp8_kernel.py:825-847（sglang_per_token_quant_fp8）`
  提供对应 `sgl_kernel` 量化入口。
  CModel 不支持或效率不佳的 block/channel scale 转换在
  `python/sglang/srt/layers/quantization/fp8_utils.py:1421-1471（block_quant_to_tensor_quant）`、
  `python/sglang/srt/layers/quantization/fp8_utils.py:1771-1787（channel_quant_to_tensor_quant）`
  使用 CPU workaround。
- **未量化层和权重加载。**
  `python/sglang/srt/layers/quantization/unquant.py:217-229（UnquantizedLinearMethod::apply）`
  对 SIPU 用 CPU `F.linear` 后拷回，绕过慢或不稳定的 CModel bf16 GEMM；
  `python/sglang/srt/layers/quantization/unquant.py:849-856（UnquantizedFusedMoEMethod::forward_sipu）`
  要求使用 Triton/fused MoE
  runner。`python/sglang/srt/model_loader/loader.py:151-180（device_loading_context）`
  在 SIPU 跳过 shard load 后的 bulk `p.data.to(sipu)`，因为该操作可能使 CModel hang；
  `python/sglang/srt/model_loader/utils.py:290-305（should_async_load）` 关闭 threaded async H2D。
- **Embedding/lm_head 和 KV allocator。**
  `python/sglang/srt/models/cpu_embedding_lm_head.py:21-30（use_cpu_embedding_lm_head）`
  默认让 SIPU 的大 embedding/lm_head 留在 CPU，
  `python/sglang/srt/models/cpu_embedding_lm_head.py:38-56（embed_input_ids）`、
  `python/sglang/srt/models/cpu_embedding_lm_head.py:59-103（logits_processor_with_cpu_lm_head）`
  负责 CPU/SIPU 间搬运。`python/sglang/srt/mem_cache/allocator/paged.py:215-232（PagedTokenToKVPoolAllocator::alloc_extend）`
  和 `python/sglang/srt/mem_cache/allocator/paged.py:274-284（PagedTokenToKVPoolAllocator::alloc_decode）` 因 Triton allocator 在 SIPU
  上可能挂起，改用 CPU naive 分配后复制索引；这保证可用性，但会引入同步和拷贝开销。
- **其它算子覆盖。** `python/sglang/srt/layers/activation.py`、`python/sglang/srt/layers/layernorm.py`、旋转
  embedding、conv、logits processor、linear attention/GDN 等文件增加 SIPU forward 或
  fallback。新增的 `python/sglang/srt/layers/attention/linear/kernels/gdn_custom.py:25-175（CustomGDNKernel::packed_decode、CustomGDNKernel::decode、CustomGDNKernel::extend、CustomGDNKernel::target_verify）`
  通过 `sgl_kernel` 提供 SIPU GDN 的 decode/extend/verify；选择入口是
  `python/sglang/srt/layers/attention/linear/gdn_backend.py:66-68（模块设备分支）` 和
  `python/sglang/srt/layers/attention/linear/gdn_backend.py:147-155（GDNKernelDispatcher::__init__ 的 custom 分支）`。
  `python/sglang/srt/managers/scheduler.py:1681-1694（Scheduler::run_event_loop）`
  将 SIPU 的 schedule stream 绑定到 forward stream，
  `python/sglang/srt/model_executor/model_runner.py:397-415（ModelRunner::__init__）`
  也把 forward stream 绑定到 default stream，当前实现优先保证 CModel 正确性而不是多流并行。
  例如 `python/sglang/srt/layers/layernorm.py:575-614（RMSNorm::forward_sipu）` 和
  `python/sglang/srt/layers/activation.py:153-154（SiluAndMul::forward_sipu）` 分别接入
  fused RMSNorm、SiLU/mul；`python/sglang/srt/layers/rotary_embedding/base.py:194-199（RotaryEmbedding::_rope_cache_init_device）`
  则把 RoPE cache 暂时在 CPU 初始化。KV 写入侧的
  `python/sglang/srt/mem_cache/memory_pool.py:4078-4120（MLATokenToKVPool::_write_mla_kv_buffer）`
  复用 `quantize_k_cache_separate` 和 `set_mla_kv_buffer_triton`，以适配 SIPU 的
  MLA/DSA KV layout。

#### 11.3.5 模型、镜像、文档和测试

- 多个模型文件（DeepSeek、Llama、Qwen、GLM、Kimi、MiMo、StepFun、VL 模型等）接入
  SIPU 的 CPU embedding/lm_head、attention 或量化分支；
  `python/sglang/srt/speculative/draft_utils.py:278-295（DraftBackendFactory::_create_sipu_decode_backend、
  DraftBackendFactory::_create_sipu_prefill_backend）` 增加 SIPU speculative backend。
  `README.md:12-16（News）` 和 `docs-sipu/supported-models.md:1-17（支持模型清单）`
  将覆盖面记为 81 个模型；这是模型/测试 inventory，不等价于所有模型都已通过
  多卡或真实芯片验证。
- `python/sglang/check_env.py:417-475（SIPUEnv::__init__、SIPUEnv::get_info、
  SIPUEnv::get_device_info、SIPUEnv::_get_sipu_version_info）` 增加环境诊断，能输出
  `torch_sipu`、`sgl-kernel`、SDK/CModel 和设备能力。
- `.gitmodules:1-3（子模块声明）` 增加 `sgl-kernel-sipu` 子模块（当前 gitlink 为
  `42ed0661061e621b468a21d7a14f871887731065`）；`docker/sipu-0.5.18-base.Dockerfile`
  和 `docker/sipu-0.5.18.Dockerfile:77-102（Docker 构建步骤）` 固定 Ubuntu/toolchain、
  CPU 版 torch 2.10、`torch_sipu`、SiOrigin Triton、`siinfer` 和预编译 kernel，并执行
  `pip install --no-cache-dir -e python --no-deps`。也就是说，SGLang editable
  安装只改变 Python 源码指向，不会自动重编译或替换 `sgl-kernel-sipu`、CModel、
  `torch_sipu` 等二进制依赖。
- `scripts/ci/sipu_ci_cases.py`、`sipu_ci_exec.sh`、`sipu_ci_suites.yaml`、
  `sipu_ci_summary.py`、`sipu_ci_test.sh`，以及 `test/srt/sipu/` 下的模型 yaml、
  accuracy compare、通信单测和 shell suite，构成 SIPU 专用验证链；`docs-sipu/`
  补充安装、支持模型、精度和已知问题说明。

### 11.4 容器内离线推理的交叉验证

本次只做只读验证，没有修改容器或源码。执行用户给出的流程后：

- `temp/sipu_offline_infer.py:4-26（main）` 创建
  `sgl.Engine(model_path=..., device="sipu", attention_backend="sipu", disable_cuda_graph=True,
  skip_server_warmup=True)`，生成 8 个 token 后 shutdown。
- `temp/off.log.3:1-7（日志记录）` 显示 SIPU SDK、CModel shim 和 `SGL_KERNEL_LOG` 已加载；
  `temp/off.log.3:41-44（日志记录）` 显示 worker 使用 SIPU stream，两个 shard 完成加载；
  `temp/off.log.3:47-110（日志记录）` 的首个
  prefill 张量均为 `device=sipu:0`，并调用 SIPU `rmsnorm` 和 paged
  `flash_attn_with_kvcache`。
- `temp/off.log.3:2636` 记录请求完成：`prompt_tokens=13`、`completion_tokens=8`、
  `e2e_latency=133.1097s`，随后正常 shutdown，没有 traceback。这证明设备注册、
  scheduler、KV cache、attention kernel 和请求生命周期已经连通，但不能单凭这次
  4-layer tiny model 的输出证明精度或生产性能。
- 同一日志中的 RoPE 警告表明当前 external kernel 对
  `rotary_dim=128,is_neox=1` 走 fallback；相关实现为
  `sgl-kernel-sipu/sgl_kernel/element_wise.py:130-192（apply_rope_with_cos_sin_cache_inplace）`
  和 `sgl-kernel-sipu/sgl_kernel/_torch_impl.py:1240-1264（apply_rope_pos_ids_cos_sin_cache_torch）`。
  该 fallback 会产生额外拷贝，属于当前性能/覆盖限制。

### 11.5 当前边界和升级结论

1. **它是 SIPU 垂直适配，不是完整的新平台插件。** 设备名由
   `torch_sipu/PrivateUse1` 提供，SGLang 用显式分支接管热点路径；因此必须确认
   `torch_sipu`、SDK/CModel 和 `sgl-kernel-sipu` 版本彼此匹配。
2. **多卡默认不是设备内高速通信。** `SGLANG_SIPU_USE_SICCL`、
   `SGLANG_SIPU_USE_CUSTOM_ALLREDUCE`、`SGLANG_SIPU_USE_CUSTOM_DEEPEP` 默认关闭，
   默认路径会 CPU/Gloo 往返；单卡离线 smoke test 不会覆盖这些问题。
3. **存在明确 fallback 和性能折衷。** stream 是单流别名，权重加载、未量化 linear、
   embedding/lm_head、KV allocator、部分 DSA top-k/RoPE 会走 CPU 或同步拷贝；不能把
   “能启动”解释为所有 kernel 已经原生高性能。当前还没有经过验证的
   `MHA_CHUNKED_KV_SIPU` 和 dual-stream attention 路径。
4. **DSV4/MoE 仍不完整。** `python/sglang/srt/models/deepseek_v4.py:220-224（模块常量）`
   设置 `_SKIP_DSV4_ROUTED_MOE=True`；
   `python/sglang/srt/models/deepseek_v4.py:2038-2048（DeepseekV4DecoderLayer::_run_moe_ffn_dp_sync）`
   直接保留 residual，
   `python/sglang/srt/models/deepseek_v4.py:3406-3420（DeepseekV4ForCausalLM::load_weights）` 跳过
   `.experts.`/`tid2eid`，
   `python/sglang/srt/models/deepseek_v4.py:3653-3660（DeepseekV4ForCausalLM::load_weights 的参数检查）`
   也跳过对应未加载参数。这个 workaround 是为了让 attention/MHC 对齐测试继续，
   不能宣称已经支持完整 routed MXFP4 expert。
5. **fake top-k 影响精度解释。** `moe/topk.py` 的固定 expert 路由只适合当前
   CModel/CUDA 对齐实验；若测试目标是实际模型质量，必须先移除或替换它，并重新做
   端到端 accuracy test。
6. **HEAD 后的清理改变了早期实验行为。** `770f80bc` 删除了临时 SIPU 同步日志、
   1M RoPE 上限、特殊 MHC 分支以及若干默认设置；结论应以 HEAD 的
   `python/sglang/srt/hardware_backend/sipu/utils.py:28-33（set_default_server_args）` 为准，而不是
   首个 `91462e29` 的中间状态。
7. **验证范围有限。** `README.md:26-34（SIPU scope 说明）` 把当前测试限定为单 rank/archmodel，
   online serving 以及 TP/EP/DP 仍是 TODO；因此本次单卡离线日志不能代表多卡生产部署。

综上，`v0.5.18` 到 `sglang_sipu` HEAD 的核心新增是一个贯穿式 SIPU runtime：
PyTorch PrivateUse1 设备接入、SIPU attention/DSA/FlashMLA、masked DeepGEMM 和
DeepEP 接口、SiCCL/custom collective、FP8/量化、CPU fallback 加载与 KV 管理，以及
配套 Docker/CI/模型测试。它让 `device="sipu"` 真正可运行的关键，不是某个单独的
注册表条目，而是“环境注册设备 + ServerArgs 解析 + `is_sipu` 分派 + SIPU kernel/
torch_sipu 实现 + 明确 fallback”这一整条链；当前仍应把它视为面向 SIPU CModel/仿真
环境的适配版本，并按上述限制评估生产可用性。

## 12. SIMO MXFP8 离线推理失败分析：RMSNorm 的 quant_method 检查

### 12.1 环境、挂载和日志

本节针对容器 sipu-dev、工作目录 /sgl-workspace/sglang 和命令：

```bash
source /sgl-workspace/sglang/temp/env-offlie-infer.sh
debug_env_file=/dev/shm/like/ipc.sglang.2.json \
  python3 like-useful/sipu_offline_infer_simo_quant.py \
  > temp/simo.log.2026_09_08___16_25_46 2>&1
```

docker inspect sipu-dev 显示：宿主 /share/users/like/package/sglang_sipu 以 rw 挂载到
/sgl-workspace/sglang；/share 以 rw 挂载到 /share；SIPU SDK、CModel 和 torch_sipu
目录以 ro 挂载。因而宿主日志 temp/simo.log.2026_09_08___16_25_46 与容器内
/sgl-workspace/sglang/temp/simo.log.2026_09_08___16_25_46 是同一份文件。
SGLang 实际从 /sgl-workspace/sglang/python 的 editable 源码导入，SIMO 从
/share/users/like/package/simo_conda_sglang_sipu 的 editable 源码导入。

脚本 like-useful/sipu_offline_infer_simo_quant.py:6-10（main）使用的配置路径仍是
/share/users/like/package/simo_conda_sglang/simo/...，不是 _sipu checkout。两份 JSON
本次内容相同且成功加载，故不是本次根因；建议统一路径以避免版本漂移。

### 12.2 直接根因

真正的首个致命异常在 temp/simo.log.2026_09_08___16_25_46:150-199：

```text
NotImplementedError: RMSNorm with quant_linear is not supported on SIPU,
quant_method=<simo.extensions.sglang_simo.quantization.quantization.SIMOLinearMethod ...>
```

准确位置是 python/sglang/srt/layers/layernorm.py:587-589（RMSNorm::forward_sipu）。
此前 SDK/CModel、插件、两个权重 shard 都已成功初始化/加载，没有 OOM。
:201-202 的 SIGQUIT 是子进程异常后的清理动作；环境、AWQ/GGUF、floor_divide 和
KV-cache fallback 均为 warning。

| 日志位置 | 事件 |
| --- | --- |
| :3-23 | SIPU 运行时和 SIMO 插件初始化 |
| :69-80 | 读取 SIMO 配置，Linear input/weight 为 mxfp8_e4m3 |
| :81-143 | 创建量化 Linear 并完成 shard 加载 |
| :144-149 | 非致命 fallback |
| :150-199 | 首次 extend/prefill 在 RMSNorm 检查处失败 |

### 12.3 调用链

日志 traceback 对应下列相对 code base 路径和函数：

```text
python/sglang/srt/managers/scheduler.py:5121（run_scheduler_process）
 -> python/sglang/srt/managers/scheduler.py:1713（Scheduler::run_event_loop）
 -> python/sglang/srt/managers/scheduler.py:4970（dispatch_event_loop）
 -> python/sglang/srt/managers/scheduler.py:1811（Scheduler::event_loop_overlap）
 -> python/sglang/srt/managers/scheduler.py:3751（Scheduler::run_batch）
python/sglang/srt/managers/tp_worker.py:614（TpModelWorker::forward_batch_generation）
 -> python/sglang/srt/model_executor/model_runner.py:1559（ModelRunner::forward）
 -> python/sglang/srt/model_executor/model_runner.py:1746（ModelRunner::_forward_raw）
python/sglang/srt/model_executor/runner/eager_runner.py:210（EagerRunner::execute）
 -> python/sglang/srt/model_executor/runner/eager_runner.py:345（EagerRunner::_execute_extend）
python/sglang/srt/models/llama.py:612（LlamaForCausalLM::forward）
 -> python/sglang/srt/models/llama.py:466（LlamaModel::forward）
 -> python/sglang/srt/models/llama.py:360（LlamaDecoderLayer::forward）
python/sglang/kernels/fused_op.py:659（BaseFusedOp::forward）
 -> python/sglang/srt/layers/layernorm.py:587（RMSNorm::forward_sipu）
```

python/sglang/srt/models/llama.py:350-378（LlamaDecoderLayer::forward）有两处传入下游 Linear：
行 360-366 的 self_attn.qkv_proj 传给 input_layernorm，行 374-375 的 mlp.gate_up_proj
传给 post_attention_layernorm。即使第一处绕过，第二处仍会触发同类检查。
python/sglang/kernels/fused_op.py:629-662（BaseFusedOp::forward）在 SIPU 平台选择
forward_sipu，并转发 quant_linear 参数。

### 12.4 SIMOLinearMethod 的来源

脚本 like-useful/sipu_offline_infer_simo_quant.py:9-23（main）设置 quantization=simo、
device=sipu 和 override 配置。配置文件
simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w8a8_mxfp.json:6-26
对目标 Linear 设置 input/weight=mxfp8_e4m3，只排除 lm_head 和 re:.*kv_b_proj，所以
qkv_proj、o_proj、gate_up_proj、down_proj 都被量化。

具体分派链：

- simo/extensions/sglang_simo/quantization/quantization_registry.py:8-15（register_simo_quantization）
  把 simo 注册为 SIMOConfig；
- simo/extensions/sglang_simo/model_loader/loader.py:46-85（_get_quantization_config）
  读取 override 配置文件，并调用 simo/extensions/sglang_simo/quantization/quantization.py:682-726（SIMOConfig::from_config_file）；
- simo/extensions/sglang_simo/quantization/quantization.py:793-846（SIMOConfig::get_quant_method）
  对未排除的 LinearBase 选择目标规格；
- simo/extensions/sglang_simo/quantization/quantization.py:848-865（SIMOConfig::get_quant_method_by_target_spec）
  构造 SIMOLinearMethod；
- simo/extensions/sglang_simo/quantization/quantization.py:871-901（SIMOLinearMethod::__init__）
  说明它继承 LinearMethodBase，而非 UnquantizedLinearMethod；
- python/sglang/srt/layers/linear.py:167-199（LinearBase::__init__）
  将返回对象保存到 linear.quant_method。

运行时关系为：

```text
self.self_attn.qkv_proj.quant_method -> SIMOLinearMethod
self.mlp.gate_up_proj.quant_method   -> SIMOLinearMethod
```

容器内 issubclass(SIMOLinearMethod, UnquantizedLinearMethod) 为 False，条件必然成立。

### 12.5 条件表达式逐项含义

代码位于 python/sglang/srt/layers/layernorm.py:575-614（RMSNorm::forward_sipu）：

```python
if quant_linear is not None:
    quant_method = getattr(quant_linear, "quant_method", None)
    if quant_method is not None and not isinstance(
        quant_method, UnquantizedLinearMethod
    ):
        raise NotImplementedError(...)
```

它检查的是下游 Linear 的输入契约，而不是只看 dtype：

1. quant_linear=None：没有下游量化提示，执行普通 SIPU RMSNorm。
2. quant_method=None：对象没有声明量化策略，不在 norm 层猜测特殊格式。
3. UnquantizedLinearMethod：下游消费普通 tensor，RMSNorm 可返回普通输出。
4. 其它方法（包括 SIMOLinearMethod）：下游可能要求 packed tensor、scale、特殊 dtype
   或量化 GEMM；SIPU 当前没有对应的已验证实现，立即 fail-fast。

允许 UnquantizedLinearMethod 是必要的，因为 Llama 的统一调用点在未量化模型中也传入
Linear 对象。python/sglang/srt/layers/quantization/unquant.py:188-229（UnquantizedLinearMethod::apply）
在 SIPU 上把普通 Linear 输入/权重拷到 CPU 做 F.linear，再拷回设备；速度慢但契约兼容。

### 12.6 SIPU 与 CUDA 的能力差异

python/sglang/srt/layers/layernorm.py:606-614（RMSNorm::forward_sipu）检查后只调用
fused_add_rmsnorm 或 rmsnorm，不生成量化激活和 scale。

CUDA 的对应实现是：

- python/sglang/srt/layers/layernorm.py:509-522（RMSNorm::forward_cuda）在满足条件时融合；
- python/sglang/srt/layers/layernorm.py:375-423（_fp8_static_input_scale、_is_static_per_tensor_fp8_linear）
  只接受静态 per-tensor FP8，并排除 mxfp8、block quant、Marlin；
- python/sglang/srt/layers/layernorm.py:905-957（RMSNorm::forward_with_per_tensor_quant_fusion）
  返回带 scale 的特殊输出；
- python/sglang/srt/layers/quantization/fp8.py:986-1057（Fp8LinearMethod::apply）
  消费该特殊输出。

本次是 SIMO MXFP8，不满足 CUDA helper 的类型条件；同时
simo/extensions/sglang_simo/quantization/quantization.py:1116-1247（SIMOLinearMethod::apply）
开头直接对输入做 view、downcast/upcast 和 matmul，也没有消费上述特殊 tuple 的实现。

这不表示 SIMO GEMM 本身必然不能运行。若只删除检查，SIPU RMSNorm 可能返回普通 BF16
tensor，再由 SIMOLinearMethod 自行量化；但 residual、scale、layout、精度和性能均未验证。
该检查禁止的是未经实现和测试的 RMSNorm 加 SIMO activation quant 组合，不能把删除 raise
当作正确修复。

### 12.7 检查来源和处理建议

当前 `sglang_sipu` HEAD 中，git blame 表明
python/sglang/srt/layers/layernorm.py:575-589（RMSNorm::forward_sipu）的参数和检查
来自 commit `91462e291`（`feat(sipu): add sipu support on v0.5.18`）。该提交把
SIPU 的 `forward_sipu` 增加到统一分派，并将 `quant_linear` 纳入签名；随后用显式
能力检查替代潜在的参数签名错误。`d6dd88127`（`fix randn so slow and rmsnorm error`）
是另一条并行开发分支上的同名修复，不是当前 HEAD 的 blame 依据。

还要注意一个被上述异常遮蔽的后续问题：当前 SIMO HEAD 的
`simo/extensions/sglang_simo/quantization/quantization.py:1116-1247`
（`SIMOLinearMethod::apply`）在 debug 代码的 :1137 和 :1210 直接调用
`torch.cuda.is_current_stream_capturing()`。SIPU 的 `torch.cuda` 是 dummy backend，
绕过 RMSNorm 检查后可能先得到
`RuntimeError: Tried to instantiate dummy base class _cuda_isCurrentStreamCapturing`。
应改用设备无关的 capture helper（或 `torch.sipu.is_current_stream_capturing()`）后，
才能继续验证 SIMO 的 SIPU GEMM；这属于第二个独立兼容性问题，不是本次日志中首先
抛出的 RMSNorm 异常。

不改源码时：

1. 暂时去掉脚本的 quantization=simo 和 json_model_override_args，先验证纯 SIPU 基线；
   此时 Linear 使用 UnquantizedLinearMethod，但普通 Linear 会走 CPU fallback。
2. 局部量化实验可在配置副本 excludes 增加 re:.*qkv_proj 和 re:.*gate_up_proj，让传给
   RMSNorm 的两个 Linear 回到 unquantized。匹配逻辑是
   simo/extensions/sglang_simo/quantization/quantization.py:207-254（should_ignore_layer）；
   o_proj/down_proj 仍需独立做精度和性能测试。
3. 正式支持需要实现 SIPU norm+SIMO activation quant kernel（scale、layout、residual
   变体），并让 SIMOLinearMethod::apply 接受其输出，再补 prefill/decode shape 测试。

### 12.8 结论

```text
quantization=simo
 -> JSON 将 Linear 设为 MXFP8
 -> SIMOConfig::get_quant_method 返回 SIMOLinearMethod
 -> qkv_proj/gate_up_proj.quant_method 是真实量化方法
 -> LlamaDecoderLayer::forward 将它们传给 RMSNorm
 -> BaseFusedOp 选择 RMSNorm::forward_sipu
 -> SIPU 尚无 RMSNorm + SIMO activation-quant 融合契约
 -> layernorm.py:584-589 按设计 fail-fast
```

所以该 if 的目的，是只放行输出普通 tensor 的下游线性层；对需要量化输入语义的线性层
明确报告未实现，避免静默走未经验证的数值或布局路径。

## 13. `torch.ops.sgl_kernel.rmsnorm.default` 的真实实现与调用链

直接结论：`sgl_kernel/element_wise.py:91-97（rmsnorm）` 只是调用 PyTorch
custom-op overload，本身不做逐元素计算；SIPU 上的 dispatch 目标是
`csrc/attention/rmsnorm.cpp:34-72（rmsnorm）`，真正的平方归约、RMS 缩放和 weight
乘法在 `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_f32.hpp:25-98（rms_norm_f32_kernel）`、
`sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_f16.hpp:25-112（rms_norm_f16_kernel）`
和 `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_bf16.hpp:25-112（rms_norm_bf16_kernel）`
中完成。对本次对齐的 BF16 输入，当前进程执行的是这些 `.su` 源码预编译后形成的
`sikernel/release/lib/libsglang_sipu_kernels.so`；只有未通过
`sgl_kernel/element_wise.py:49-67（rmsnorm）` 对齐条件时，才会走 Python/CPU fallback。

### 13.1 先确定本次运行使用的 code base

本节的 `sgl-kernel-sipu` code base 根目录是容器内的
`/sgl-workspace/sgl-kernel-sipu`，因此下面所有 `sgl_kernel/...`、`csrc/...` 和
`sikernel/...` 路径都相对于这个根目录；SGLang 的相对路径相对于
`/sgl-workspace/sglang`。

`docker inspect sipu-dev` 的关键挂载是：

```text
/share/users/like/package/sglang_sipu -> /sgl-workspace/sglang       (rw)
/share                         -> /share                            (rw)
/share_data/torch_sipu         -> /share_data/torch_sipu             (ro)
/share_data/sicx_sdk           -> /share_data/sicx_sdk               (ro)
/share_data/arch_cmodel_release -> /share_data/arch_cmodel_release   (ro)
```

`/sgl-workspace/sgl-kernel-sipu` 本身不是上述 bind mount，而是镜像内的目录。执行
`source /sgl-workspace/sglang/temp/env-offlie-infer.sh` 时，
`temp/env-offlie-infer.sh:1-4（顶层脚本）` 会 source 该目录的 `setup.sh`；容器中
`sgl_kernel` 的 editable finder 也明确把包映射到
`/sgl-workspace/sgl-kernel-sipu/sgl_kernel`。因此，本题给出的
`/sgl-workspace/sgl-kernel-sipu/sgl_kernel/element_wise.py:91` 是实际导入的文件。
宿主机挂载树中另有 `/sgl-workspace/sglang/sgl-kernel-sipu`，但它不是这次 Python
import 命中的那一份，不能混用两份文件的行号。

容器中实际检查到的对象位置和 dispatcher 结果为：

```text
sgl_kernel.__file__
  /sgl-workspace/sgl-kernel-sipu/sgl_kernel/__init__.py
sgl_kernel.common_ops
  /sgl-workspace/sgl-kernel-sipu/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so
schema
  sgl_kernel::rmsnorm(Tensor($0! -> ) output, Tensor input,
                      Tensor weight, float eps, bool enable_pdl) -> ()
dispatch
  PrivateUse1: registered at
  /sgl-workspace/sgl-kernel-sipu/csrc/common_extension.cc:266 [kernel]
```

所以 `torch.ops.sgl_kernel.rmsnorm.default` 中的 `.default` 只是 PyTorch
`OpOverloadPacket` 选择默认 overload 的写法，不是一个 Python 实现文件名；真正执行
哪个函数由 dispatcher 按输入 tensor 的 device 选择。

### 13.2 Python 入口：`rmsnorm`

`sgl_kernel/element_wise.py:42-102（rmsnorm）` 是用户态 facade。它的职责是准备
输入和输出，随后把工作交给 C++ custom op，而不是在 Python 中逐元素实现 native
SIPU 路径。

1. `sgl_kernel/element_wise.py:49-67（rmsnorm）` 先检查最后一维的总字节数是否为
   1024 的倍数。若不满足，则把 `input`、`weight` 拷到 CPU 的 `float32`，计算
   `input * rsqrt(mean(input**2) + eps) * weight`，再按 `out` 的 device/dtype 拷回；
   这条分支不会调用 `torch.ops.sgl_kernel.rmsnorm.default`，只是为 trim/debug
   shape 准备的低性能 fallback。
2. `sgl_kernel/element_wise.py:69-89（rmsnorm）` 检查 `input`、`weight` 和 `out`
   的 contiguous 状态。非连续时创建连续临时 tensor；`out is None` 时先创建与
   `input` 同 shape/dtype/device 的输出。
3. `sgl_kernel/element_wise.py:91-97（rmsnorm）` 调用：

   ```python
   torch.ops.sgl_kernel.rmsnorm.default(
       out_work,
       input_work,
       weight_work,
       eps,
       bool(enable_pdl) if enable_pdl is not None else False,
   )
   ```

   custom op 的 schema 返回 `()`，但第一个参数是可变的 `Tensor! output`；因此
   `sgl_kernel/element_wise.py:99-102（rmsnorm）` 必要时把临时结果复制回原 `out`，
   最后由 Python facade 返回 `out`。
4. `sgl_kernel/element_wise.py:91-97（rmsnorm）` 传递了 `enable_pdl`，但下游
  `csrc/attention/rmsnorm.cpp:34-72（rmsnorm）` 在
  `csrc/attention/rmsnorm.cpp:40（rmsnorm）` 明确忽略它，当前实现没有 PDL 分支。

### 13.3 `torch.ops` 是怎样绑定到 C++ 的

导入顺序决定注册时机：

- `sgl_kernel/__init__.py:3-6（模块导入）` 先尝试导入 `_torch_impl`，注册 CPU
  reference custom ops；
- `sgl_kernel/__init__.py:8-11（模块导入）` 导入 `common_ops` 这个 C++ 扩展，触发
  `TORCH_LIBRARY_FRAGMENT` 静态初始化；
- `sgl_kernel/__init__.py:38-41（模块导入）` 再把 `element_wise.py` 的 Python
  facade 导出到包根。

`csrc/common_extension.cc:8（kSIPU）` 把 `kSIPU` 定义为
`torch::kPrivateUse1`。注册本身位于
`csrc/common_extension.cc:266-280（TORCH_LIBRARY_FRAGMENT(sgl_kernel)）`：

- `csrc/common_extension.cc:279（TORCH_LIBRARY_FRAGMENT(sgl_kernel)）` 定义
  `rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()`；
- `csrc/common_extension.cc:280（TORCH_LIBRARY_FRAGMENT(sgl_kernel)）` 把算子在
  `PrivateUse1` dispatch key 上绑定到 C++ 函数 `&rmsnorm`。

因此当 `input_work.device` 是 `sipu:0` 时，torch_sipu 把该设备映射到
`PrivateUse1`，dispatcher 命中 `csrc/attention/rmsnorm.cpp:34-72（rmsnorm）`；它
不会走 CUDA kernel，也不会走 `_torch_impl` 的 CPU reference。

### 13.4 C++ bridge 做了什么

`csrc/attention/rmsnorm.cpp:34-72（rmsnorm）` 是 PyTorch tensor 到 SiKernel API
之间的 bridge。

- `csrc/attention/rmsnorm.cpp:41-46（rmsnorm）` 要求 output/input 为二维、两者
  shape 相同、weight 为一维且长度等于 hidden size，并要求 input contiguous。Python
  facade 已经在 `sgl_kernel/element_wise.py:69-89（rmsnorm）` 处理了常见的非连续
  输入。
- `csrc/attention/rmsnorm.cpp:48-50（rmsnorm）` 取出
  `batch_size=input.size(0)` 和 `original_normalized_size=input.size(1)`，然后计算
  `normalized_size = ceil(original_normalized_size / 512) * 512`。普通 Llama
  BF16 hidden size（例如 4096）本来就是 512 的倍数，两个值相同。
- `csrc/attention/rmsnorm.cpp:52-71（rmsnorm）` 使用 `AT_DISPATCH_V2` 选择
  `float32`、`float16` 或 `bfloat16` 实例，在
  `csrc/attention/rmsnorm.cpp:57-67（rmsnorm）` 调用 SiKernel 的裸指针入口：

  ```cpp
  ::rms_norm<sifmt_type>(
      output.data_ptr(), input.data_ptr(), weight.data_ptr(),
      static_cast<float>(eps), batch_size, normalized_size,
      original_normalized_size, 1, 1, get_current_sipu_stream());
  ```

  两个 `1` 分别表示 `weight_opt=1` 和 `eps_opt=1`，所以这条 SGLang 路径始终乘
  RMSNorm weight，并始终加 epsilon。
- `include/op_sipu_type.h:39-65（OpSIPUType/opsipu_type）` 将 PyTorch 类型转换为
  SiKernel 类型：`at::Half -> sifmt::float16`、`at::BFloat16 -> sifmt::bfloat16`、
  `float -> sifmt::float32`。
- `include/sikernel_utils.h:11-13（get_current_sipu_stream）` 返回
  `at::sipu::getCurrentSIPUStream().stream()`，所以 native kernel 在当前 PyTorch
  SIPU stream 上异步 launch。

这里的 `::rms_norm` 不是 `torch.ops` 直接调用的 Python 函数，而是链接到
`libsglang_sipu_kernels.so` 的 C++/SCC 符号。`setup.py:34-46（sources）`
把 `csrc/attention/rmsnorm.cpp` 编进 `sgl_kernel.common_ops`，
`setup.py:77-93（CppExtension）` 则把它链接到
`libsglang_sipu_kernels.so`、`libtorch_sipu.so` 等库。

### 13.5 SiKernel 的裸指针兼容层

SiKernel 的公开声明在
`sikernel/include/sikernel.h:1262-1289（rms_norm）`：

- `sikernel/include/sikernel.h:1277-1280（rms_norm）` 是带 tensor metadata 的接口；
- `sikernel/include/sikernel.h:1285-1289（rms_norm）` 是 bridge 当前使用的裸指针
  兼容接口，额外接收 `batch_size`、`normalized_size` 和
  `original_normalized_size`。

裸指针接口的源码在
`sikernel/source/source_builtin/attention/rms_norm/kernel/legacy_api.su:10-23（rms_norm<T>）`：

1. `sikernel/source/source_builtin/attention/rms_norm/kernel/legacy_api.su:15-17（rms_norm<T>）` 用 `(batch_size, normalized_size)` 构造
   output/input tensor 和 `(normalized_size)` 构造 weight tensor；构造函数会生成默认
   contiguous strides。
2. `sikernel/source/source_builtin/attention/rms_norm/kernel/legacy_api.su:18-20（rms_norm<T>）` 把三个 PyTorch data pointer 写入这些 tensor
   的 `data` 字段；这里没有重新分配或复制数据，内存仍由 PyTorch tensor 管理。
3. `sikernel/source/source_builtin/attention/rms_norm/kernel/legacy_api.su:21-22（rms_norm<T>）` 转调 metadata 版本的
   `rms_norm<T>(output_tensor, input_tensor, weight_tensor, ...)`。
4. `sikernel/source/source_builtin/attention/rms_norm/kernel/legacy_api.su:25-36（rms_norm<T>）` 显式实例化 `float32`、`float16`、`bfloat16`
   三种类型，确保它们出现在 release shared library 中。

### 13.6 真正的 host launcher 和 device kernel

#### Host launcher

`sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:204-240（rms_norm_launch<T>）`
是 tensor API 的主要分派逻辑。

- `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:34-49（is_contiguous_layout）` 按 sizes/strides 逐维检查真实
  contiguous layout；这比只看 PyTorch 的某些 size-1 stride 更严格。
- `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:51-79（check_rms_norm_tensor_metadata）` 检查 data pointer
  非空、输入/输出为二维且 shape 相同、normalized dimension 的字节数为 1024 的
  倍数、`original_normalized_size` 在合法范围内，以及启用 weight 时 weight 的
  rank/长度正确。
- `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:83-160（rms_norm_contiguous_launch<T>）` 为连续 tensor 设置
  launch 配置：batch 小于 2 时每个 block 使用 1 个线程，batch 至少 2 时每个
  block 使用 2 个线程；batch 至少 32 时使用 16 个 grid block，否则使用
  `(batch_size+1)/2` 个 grid block。随后按模板类型在
  `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:102-120（rms_norm_contiguous_launch<T>）`、
  `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:121-139（rms_norm_contiguous_launch<T>）`、
  `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:140-159（rms_norm_contiguous_launch<T>）` 分别启动 F32、F16、
  BF16 kernel。
- `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:162-202（rms_norm_bf16_strided_input_launch）` 是 BF16 特殊
  非连续输入的 launch wrapper。
- `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:204-240（rms_norm_launch<T>）` 先在
  `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:213-217（rms_norm_launch<T>）` 走连续路径；
  如果不连续，只在 BF16、normalized size 为 512 或 1536、输入 stride 为
  `(2176, 1)`、输出和 weight 连续时，于
  `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:220-231（rms_norm_launch<T>）` 走 strided kernel；其它布局在
  `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:235-239（rms_norm_launch<T>）` 通过 `sipu::check(false, ...)` 报错。
- `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:242-248（rms_norm_timed<T>）` 是带性能时间戳的入口；
  `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:250-256（rms_norm<T>）` 是公开 tensor API，转调
  `rms_norm_launch<T>`；`sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:263-265（rms_norm<T>）` 显式实例化
  三种 dtype。

#### F32/F16/BF16 device kernel 的共同算法

三个 `.hpp` 文件中的 `__global__` 函数才是逐元素计算发生的地方；`.su` 是 SCC 的
SIPU kernel 源码，使用 SIPU 的 `<<<grid, cluster, block, 0, stream>>>` launch 语法，
不是 CUDA 编译出来的普通 GPU kernel。

对每一行输入，数学结果为：

```text
sum_kernel = sum(x[j] * x[j] for j in [0, normalized_size))
mean = sum_kernel / original_normalized_size
r = 1 / sqrt(mean + eps)
y[j] = x[j] * r * weight[j]       (weight_opt = 1)
```

实际代码会按 `normalized_size` 的 1024-byte tile 处理，并用
`original_normalized_size` 作分母。两者相等时就是通常的 RMSNorm；如果
`normalized_size` 是向上补齐的值，补齐区必须符合调用者约定（通常是有效的 padding），
否则额外 tile 也会参与平方和。因此调用者必须遵守 API 的对齐/填充契约。

- `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_f32.hpp:25-98（rms_norm_f32_kernel）`
  以 1024 字节（256 个 FP32）为 tile，从 global memory 读取数据，在
  `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_f32.hpp:47-70（rms_norm_f32_kernel）`
  做平方累加和 RVV reduction，在
  `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_f32.hpp:70-78（rms_norm_f32_kernel）`
  计算 reciprocal square root，再于
  `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_f32.hpp:80-94（rms_norm_f32_kernel）`
  乘以 weight 并写回 output。
- `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_f16.hpp:25-112（rms_norm_f16_kernel）`
  使用 FP16 输入/输出，但在
  `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_f16.hpp:59-78（rms_norm_f16_kernel）`
  先转换到 FP32 做平方、向量归约和 `1/sqrt(...)`，然后在
  `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_f16.hpp:97-107（rms_norm_f16_kernel）`
  转回 FP16 写回。
- `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_bf16.hpp:25-112（rms_norm_bf16_kernel）`
  对 BF16 做同样的 FP32 累加；`sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_bf16.hpp:52-63（rms_norm_bf16_kernel）`
  读取 tile 并累加平方，`sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_bf16.hpp:65-87（rms_norm_bf16_kernel）`
  完成 RVV reduction 和 reciprocal square root，`sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_bf16.hpp:89-107（rms_norm_bf16_kernel）`
  第二次读取数据、乘归一化因子和 weight、转回 BF16 并写回。
- `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_bf16.hpp:114-201（rms_norm_bf16_strided_input_kernel）`
  是 BF16 特殊 stride 版本：第一遍按 `input_stride0` 读取并归约，第二遍按连续
  output stride 写出；它只服务
  `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:220-231（rms_norm_launch<T>）`
  接受的两个布局。

连续 kernel 的共同结构是“两遍扫描”：第一遍把每个 tile 转为 FP32、平方并累加；
第二遍重用 shared memory 中的 tile（超出 shared buffer 时重新从 global memory 读），
乘同一行的 reciprocal RMS，再乘可选 weight。这样每一行只计算一次归一化因子，
而不是每个元素重复计算。

#### 编译/链接到运行时库

- `sikernel/source/source_builtin/attention/rms_norm/CMakeLists.txt:31-34（scc_add_library）`
  把 `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su` 和
  `sikernel/source/source_builtin/attention/rms_norm/kernel/legacy_api.su` 编译成
  rms_norm native library。
- `sgl-sikernel-integration/release.list:1-3（release module list）` 中的第 2 行包含
  `source/source_builtin/attention/rms_norm`。
- `sikernel/release/build/CMakeLists.txt:33-59（release source collection）` 收集
  release list 中各目录的 `.su` 文件，
  `sikernel/release/build/CMakeLists.txt:70-79（scc_add_library）` 将它们合并为
  `sipu_kernels`；`sikernel/release/build/CMakeLists.txt:93-95（scc_install）` 安装到
  `sikernel/release/lib`。
- `sgl-sikernel-integration/build.sh:31-56（构建脚本）` 最后把
  `libsipu_kernels.so` 重命名为 `libsglang_sipu_kernels.so`，供
  `setup.py:77-93（CppExtension）` 链接。

因此“真实实现”有两个可见层次：可读的 host launcher/device kernel 源码在
`sikernel/source/source_builtin/attention/rms_norm/kernel/`，当前进程实际执行的是
这些源码预编译后、并由 `sgl_kernel/common_ops...so` 间接链接的
`sikernel/release/lib/libsglang_sipu_kernels.so`。只修改 `.su` 文件而不重新构建该
shared library，不会改变正在运行的 kernel。

### 13.7 从离线推理脚本到 `rmsnorm.default` 的调用链

本次离线 smoke test 的上层链路如下（每一项都按相对 code base 路径标注）：

```text
temp/sipu_offline_infer.py:4-22（main）
  -> python/sglang/srt/entrypoints/engine.py:232-284（Engine::__init__）
  -> python/sglang/srt/entrypoints/engine.py:1060-1130（Engine::_launch_subprocesses）
  -> python/sglang/srt/managers/scheduler.py:5054-5142（run_scheduler_process）
  -> python/sglang/srt/model_executor/model_runner.py:632-649（ModelRunner::initialize）
  -> python/sglang/srt/model_executor/model_runner.py:1061-1122（ModelRunner::load_model）
  -> python/sglang/srt/models/llama.py:612（LlamaForCausalLM::forward）
  -> python/sglang/srt/models/llama.py:466（LlamaModel::forward）
  -> python/sglang/srt/models/llama.py:350-378（LlamaDecoderLayer::forward）
  -> python/sglang/kernels/fused_op.py:629-662（BaseFusedOp::forward）
  -> python/sglang/srt/layers/layernorm.py:575-614（RMSNorm::forward_sipu）
  -> sgl_kernel/element_wise.py:42-102（rmsnorm）
  -> csrc/common_extension.cc:266-280（TORCH_LIBRARY_FRAGMENT(sgl_kernel)）
  -> csrc/attention/rmsnorm.cpp:34-72（rmsnorm）
  -> sikernel/source/source_builtin/attention/rms_norm/kernel/legacy_api.su:10-23（rms_norm<T>）
  -> sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:250-256（rms_norm<T>）
  -> sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:204-240（rms_norm_launch<T>）
  -> sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_f32.hpp:25-98（rms_norm_f32_kernel）
  -> sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_f16.hpp:25-112（rms_norm_f16_kernel）
  -> sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_bf16.hpp:25-112（rms_norm_bf16_kernel）
  -> SIPU stream/runtime
```

关键的模型层细节如下：

- `temp/sipu_offline_infer.py:6-16（main）` 用 `device="sipu"`、
  `attention_backend="sipu"` 创建 Engine，并关闭 CUDA graph；这让模型执行走 SIPU
  eager 路径。
- `python/sglang/srt/models/llama.py:360-366（LlamaDecoderLayer::forward）` 在
  attention 前把 `hidden_states` 送进 `input_layernorm`；第一次进入 layer 时
  `residual is None`，因此最终调用的是普通 `rmsnorm`。
- `python/sglang/srt/models/llama.py:374-375（LlamaDecoderLayer::forward）` 在
  attention 后把 `hidden_states` 和 residual 送进
  `post_attention_layernorm`；这条分支调用的是
  `python/sglang/srt/layers/layernorm.py:606-610（RMSNorm::forward_sipu）` 中的
  `fused_add_rmsnorm`，它是另一个 custom op，不应与本题的 `rmsnorm.default` 混淆。
- `python/sglang/kernels/fused_op.py:182-185（_platform_key）` 在 SIPU 环境优先
  返回 `"sipu"`；`python/sglang/kernels/fused_op.py:538-567（BaseFusedOp::_resolve_forward_method）`
  因而选择 `forward_sipu`；最终
  `python/sglang/kernels/fused_op.py:629-662（BaseFusedOp::forward）` 调用绑定方法。
- `python/sglang/srt/layers/layernorm.py:575-599（RMSNorm::forward_sipu）` 处理空
  tensor、residual 和 reshape；无 residual 时在
  `python/sglang/srt/layers/layernorm.py:611-614（RMSNorm::forward_sipu）` 调用
  `rmsnorm(x, self.weight.data, self.variance_epsilon)`，这就是进入
  `sgl_kernel/element_wise.py:42-102（rmsnorm）` 的直接调用点。

`temp/env-offlie-infer.sh:3-4（顶层脚本）` 设置 kernel 日志开关和 SGLang 的
`PYTHONPATH`；`sgl_kernel/_kernel_log.py:188-204（kernel_log）` 包装公开 Python
  facade，`sgl_kernel/_kernel_log.py:312-320（install_kernel_log_hooks）`
  安装包装。因此 `temp/off.log.3` 中的
  `[SGL_KERNEL] -> rmsnorm` / `<- rmsnorm` 记录的是 Python wrapper 的进入/返回，
  不是另一个名为 `rmsnorm` 的 Python 实现；wrapper 内部紧接着调用了上面的 C++/SIPU
  链。该日志中的第一次普通 norm（例如 `temp/off.log.3:43-55`）显示输入
  `(13, 4096)`、BF16、`sipu:0`，与 native kernel 的二维 BF16 contract 相符。

### 13.8 运行时验证与限制

在 `sipu-dev` 中执行 `source /sgl-workspace/sglang/temp/env-offlie-infer.sh` 后，用
`sipu:0` 上的 BF16 `(2, 512)` tensor 做了最小调用：

```text
[SGL_KERNEL] -> rmsnorm
[SGL_KERNEL] <- rmsnorm (约 1.4ms)
output: shape=(2, 512), dtype=torch.bfloat16, device=sipu:0
finite=True
```

这验证了 Python facade、PrivateUse1 registration、C++ bridge、动态库和 SIPU
kernel 至少能完成一次端到端调用；它不等价于完整模型的数值精度验证。

需要特别注意：

1. `sgl_kernel/element_wise.py:49-67（rmsnorm）` 的 1024-byte 对齐检查只保证
   wrapper 的 native/fallback 分界；`csrc/attention/rmsnorm.cpp:48-67（rmsnorm）`
   还会把 normalized size 向上补到 512 元素。对直接调用 API 的非 512 hidden size，
   应确认调用者确实分配了 padded buffer，因为 device kernel 会按
   `normalized_size` tile 读取/写出；尤其 FP32 的 Python 1024-byte 检查允许 256
   元素，但 C++ bridge 仍会向上取到 512，不能仅凭 Python 检查断言这种 shape 安全。
2. `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:69-70（check_rms_norm_tensor_metadata）`
   要求 normalized dimension 的字节数为 1024 的倍数；F32、F16、BF16 支持范围由
   `csrc/attention/rmsnorm.cpp:68-71（rmsnorm）` 的 dtype dispatch 共同决定。
3. Python wrapper 会在 `sgl_kernel/element_wise.py:69-100（rmsnorm）` 对非连续
   tensor 做 copy，所以底层 `sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel_bf16.hpp:114-201（rms_norm_bf16_strided_input_kernel）`
   主要面向直接使用 SiKernel tensor API 的调用者；不能据此推断任意 PyTorch
   non-contiguous layout 都被支持。
4. `csrc/attention/rmsnorm.cpp:40（rmsnorm）` 把 `enable_pdl` 丢弃，故目前不能用
   该参数开启 PDL 优化。

最终可以把本题的“真实实现”压缩成下面这条链：

```text
Python facade（准备 contiguous output）
  -> PyTorch schema/PrivateUse1 dispatcher
  -> csrc/attention/rmsnorm.cpp:34-72（rmsnorm）
  -> sikernel/source/source_builtin/attention/rms_norm/kernel/legacy_api.su:10-23（rms_norm<T>）
  -> sikernel/source/source_builtin/attention/rms_norm/kernel/rms_norm_kernel.su:204-240（rms_norm_launch<T>）
  -> dtype-specific SIPU __global__ kernel
  -> SIPU runtime 在当前 stream 上写回 output
```

## 14. 提交 `11d03eaeef` 的核心功能与测试覆盖

### 14.1 提交范围和结论

提交信息为：

```text
commit:  11d03eaeefd87c867f3fcbdf63f58cc0ae04de39
parent:  7120f3ee13de565cc737e0598110e7f7603c4e9f
subject: runtime: Add flashinfer rmsnorm + quant fusion support SM90, SM100, SM120
```

提交修改 11 个文件（`+851/-33`）。`11d03eaeef` 是当前
`v0.5.18-sipu-dev` 分支的祖先，因此可以直接用该提交树的源码和行号复核。下面的
行号均以 `git show 11d03eaeef:<path>` 得到的提交快照为准；后续 SIPU 提交可能使
工作树中的行号发生变化。

容器中 `/share/users/like/package/sglang_sipu` 以读写方式挂载为
`/sgl-workspace/sglang`，SGLang 的 editable 安装指向
`/sgl-workspace/sglang/python`。但本提交的功能不是 SIPU kernel：它依赖 CUDA
FlashInfer，并针对 CUDA SM90、SM100、SM120 的 CUTLASS FP8 scale 约定做了适配。
在 SIPU 上，`torch.cuda.is_available()` 为 `False`，且容器没有 `flashinfer`，所以
不会进入这条融合路径。

核心优化可以概括为把原来的：

```text
RMSNorm -> 单独的 static per-tensor FP8 activation quant -> FP8 Linear GEMM
```

改成：

```text
FlashInfer rmsnorm_quant（或 fused_add_rmsnorm_quant）
    -> (fp8_activation, input_scale, original_dtype)
    -> FP8 Linear GEMM，跳过重复 quant
```

有 residual 时，`fused_add_rmsnorm_quant` 同时完成 residual add、RMSNorm 和量化。
因此通常少一次独立的 activation-quant kernel launch；同时通过
`original_dtype` 保留 FP16/BF16 模型的输出 dtype，避免把 FP8 输入误当成 BF16。

### 14.2 RMSNorm 融合的实现

#### FlashInfer 可用性和下游线性层识别

- `python/sglang/srt/layers/layernorm.py:57-58（模块级初始化）` 增加
  `_flashinfer_rmsnorm_quant_available`，默认值为 `False`。
- `python/sglang/srt/layers/layernorm.py:88-99（模块级初始化）` 尝试导入
  `flashinfer.norm.rmsnorm_quant` 和
  `flashinfer.norm.fused_add_rmsnorm_quant`；任意导入失败就保持关闭，运行时会
  回到原有 RMSNorm/quant 路径。
- `python/sglang/srt/layers/layernorm.py:370-390（_fp8_static_input_scale）`
  从下游 `linear` 取出 `input_scale`，只接受单元素（per-tensor）scale，并把它
  作为融合 kernel 的量化 scale 返回。
- `python/sglang/srt/layers/layernorm.py:393-418（_is_static_per_tensor_fp8_linear）`
  识别两类下游实现：
  1. 原生 `Fp8LinearMethod`，但排除 `block_quant`、`use_mxfp8` 和
     `use_marlin`；
  2. `CompressedTensorsLinearMethod` 加
     `CompressedTensorsW8A8Fp8`，且该 scheme 的
     `is_static_input_scheme` 为真。

这层过滤很重要：FlashInfer 这里实现的是静态 per-tensor activation quant，不能
把 block/MXFP8/Marlin 或 per-token scale 当成相同的接口。

#### `RMSNorm::forward_cuda` 的选择逻辑

`python/sglang/srt/layers/layernorm.py:469-568（RMSNorm::forward_cuda）` 的顺序是：

1. `:476-486` 处理空输入和高维输入 reshape。
2. `:487-503` 对 `variance_size_override`、batch-invariant 等不兼容情形直接走
   原有路径。
3. `:504-517` 在下列条件同时满足时启用融合：

   ```python
   quant_linear is not None
   and not self.cast_x_before_out_mul
   and _flashinfer_rmsnorm_quant_available
   and _fp8_static_input_scale(quant_linear) is not None
   ```

   得到 scale 后调用 `RMSNorm::forward_with_per_tensor_quant_fusion`。
4. 其它情况继续执行原有 HF-semantics、residual RMSNorm 或普通 `rmsnorm`。

`quant_linear` 是新增的可选参数。为使统一调用在其它设备不报 unexpected keyword，
提交也给 `RMSNorm::forward_npu`（`:570-584`）、`RMSNorm::forward_aiter`
（`:586-661`）、`RMSNorm::forward_hip`（`:663-701`）、`RMSNorm::forward_musa`
（`:703-725`）、`RMSNorm::forward_native`（`:727-776`）、
`RMSNorm::forward_cpu`（`:778-797`）和 `RMSNorm::forward_xpu`（`:799-825`）增加
了同名可选参数；这些后端本提交并不消费该参数。

#### 融合函数的数据契约

`python/sglang/srt/layers/layernorm.py:858-915（RMSNorm::forward_with_per_tensor_quant_fusion）`
实现具体的 FlashInfer 调用：

- `:885-891` 保存 `orig_dtype = x.dtype`，并把高维/非 contiguous 输入整理成二维
  contiguous tensor。
- `:893` 分配 `torch.float8_e4m3fn` 输出。
- `:894-908` 有 residual 时先合并 `post_residual_addition`，然后调用
  `_flashinfer_fused_add_rmsnorm_quant(out, x, residual, weight, scale, eps)`；该
  API 原地更新 residual，并把量化后的 RMSNorm 输出写入 `out`。
- `:910-915` 无 residual 时调用
  `_flashinfer_rmsnorm_quant(out, x, weight, scale, eps)`。

返回值约定为：

```text
无 residual:  (fp8_out, scale, orig_dtype)
有 residual: ((fp8_out, scale, orig_dtype), residual_out)
```

scale 的语义与 `static_quant_fp8` 一致：`q = normed / scale`，下游 GEMM 使用同一
scale 还原量化激活。`orig_dtype` 是量化前的 FP16/BF16（而不是 `fp8_out.dtype`）。

### 14.3 下游 FP8 Linear 如何消费融合结果

#### 原生和 compressed-tensors 线性层

- `python/sglang/srt/layers/quantization/fp8.py:957-1061（Fp8LinearMethod::apply）`
  在普通、非 block/MXFP8 分支的 `:1035-1051` 识别三元组，拆出
  `qx=x[0]`、`x_scale=x[1]`、`out_dtype=x[2]`，再调用
  `apply_fp8_linear(..., pre_quant_output_dtype=out_dtype)`。
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py:228-280（CompressedTensorsW8A8Fp8::apply_weights）`
  在 `:234-250` 做同样的 tuple 转发，并设置 `compressed_tensor_quant=True`。
- `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py:222-226（CompressedTensorsW8A8Fp8::process_weights_after_loading）`
  对静态输入 scheme 把加载的 input scale 归约为单元素 scale，正好满足上游
  `_fp8_static_input_scale` 的契约。

#### `apply_fp8_linear` 的预量化和输出 dtype

`python/sglang/srt/layers/quantization/fp8_utils.py:1713-1944（apply_fp8_linear）`
 进行了三组配套修改：

1. `:1724` 新增 `pre_quant_output_dtype` 参数。
2. `:1741-1752` 根据输入 dtype 判断是否已经是 FP8。若是 FP8，跳过再次 quant，
   要求 `input_scale` 非空且为单元素（`:1765-1767`），并把输出 dtype 设为
   `pre_quant_output_dtype`；没有传该参数时才兼容性回退到 BF16（`:1749-1751`）。
3. `:1754-1763` 将 channel-wise weight scale、矩阵 16 对齐和硬件能力组合成
   `use_cutlass_channelwise_gemm`，并在 SM90/SM100/SM120 上打开
   `native_scalar_a_scale`。

   - 这些架构的 CUTLASS `fp8_scaled_mm` 可以直接接收一个 scalar A（activation）
     scale，因此融合输出的单元素 scale 无需复制。
   - 其它仍要求每行 A scale 的 channel-wise 路径在 `:1768-1771` 将 scale 复制为
     `(M, 1)`；普通静态 quant 路径也在 `:1812-1818` 使用同一条件控制
     `repeat_scale`。

最后，`:1838-1944` 把 `output_dtype` 贯穿 Triton、CUTLASS、`torch._scaled_mm`
和 fallback GEMM。这样 FP16 模型的预量化输入不会因为输入 tensor 是 FP8 而被
错误地产生 BF16 输出。

### 14.4 模型调用链和兼容性改动

- `python/sglang/srt/models/llama.py:341-369（LlamaDecoderLayer::forward）`：
  `:351-357` 将 `self.self_attn.qkv_proj` 传给 input RMSNorm，`:365-367` 将
  `self.mlp.gate_up_proj` 传给 post-attention RMSNorm。这样 RMSNorm 可以看到
  紧接着的 Linear 是否有静态 FP8 input scale。
- `python/sglang/srt/models/qwen2.py:284-312（Qwen2DecoderLayer::forward）`：
  `:294-300` 和 `:308-310` 做同样的 qkv/gate-up 传递。
- `python/sglang/srt/models/llama_eagle.py:49-54（LlamaDecoderLayer::__init__）`
  和 `python/sglang/srt/models/qwen2_eagle.py:50-55（Qwen2DecoderLayer::__init__）`
  将跳过 layernorm 的 lambda 改为接受 `quant_linear=None`，避免 EAGLE 第 0 层
  因统一调用新增关键字而报错。

运行时的关键链路是：

```text
LlamaDecoderLayer::forward / Qwen2DecoderLayer::forward
  -> BaseFusedOp::forward
       python/sglang/kernels/fused_op.py:623-656（BaseFusedOp::forward）
  -> RMSNorm::forward_cuda
       python/sglang/srt/layers/layernorm.py:469-568（RMSNorm::forward_cuda）
  -> _fp8_static_input_scale
  -> RMSNorm::forward_with_per_tensor_quant_fusion
       python/sglang/srt/layers/layernorm.py:858-915（RMSNorm::forward_with_per_tensor_quant_fusion）
  -> flashinfer.norm.rmsnorm_quant / fused_add_rmsnorm_quant
  -> (fp8, scale, orig_dtype)
  -> Fp8LinearMethod::apply 或 CompressedTensorsW8A8Fp8::apply_weights
  -> apply_fp8_linear
       python/sglang/srt/layers/quantization/fp8_utils.py:1713-1944（apply_fp8_linear）
  -> fp8_scaled_mm / Triton / torch._scaled_mm
```

### 14.5 是否有测试用例

有，而且提交新增了一个专门的 RMSNorm 融合测试文件，并扩展了 FP8 utility 测试。
不过它们都是 CUDA CI 测试，不是 SIPU 测试。

#### RMSNorm 融合单元测试

`test/registered/layers/test_layernorm_fusion.py:13-142（TestRMSNormFp8QuantFusion）`
在 `:10` 注册为 `base-b`、`1-gpu-large` CUDA CI。

- `TestRMSNormFp8QuantFusion::setUpClass` `:21-29`：没有 CUDA 或没有
  `flashinfer.norm.rmsnorm_quant` 时直接 `SkipTest`。
- `TestRMSNormFp8QuantFusion::_run_fusion_test` `:31-78`：以
  `RMSNorm::forward_native` 为 reference，调用
  `RMSNorm::forward_with_per_tensor_quant_fusion`，检查：
  - 输出 dtype 是 `torch.float8_e4m3fn`、shape 正确；
  - 返回的是同一个 scale 对象，且 `orig_dtype` 保持 FP16/BF16；
  - residual 输出 dtype 和数值；
  - 反量化结果 cosine similarity `> 0.99`，平均相对误差 `< 0.1`。
- `TestRMSNormFp8QuantFusion::test_rms_norm_fp8_quant_fusion` `:80-93` 用
  `itertools.product` 覆盖 24 个组合：
  `num_tokens=[7,83,512]`、`hidden_size=[512,4096]`、residual 有/无、
  dtype 为 BF16/FP16。
- `TestRMSNormFp8QuantFusion::test_forward_cuda_quant_linear_dispatch` `:95-138`
  monkeypatch 静态 scale，验证普通 RMSNorm 会返回融合 tuple；同时验证
  `var_hidden_size` 和 `cast_x_before_out_mul=True` 两个不兼容配置仍返回普通
  dtype，而不会走该融合契约。

#### FP8 Linear scale/dtype 测试

`test/registered/quant/test_fp8_utils.py` 将 CUDA CI 预计时间从 9 秒调整为 12 秒
（`:15`），并增加：

- `TestApplyFp8LinearScaleDispatch::test_native_scalar_a_static_prequant_and_dynamic_scale_shapes`
  `:65-140`：分别伪造 SM90、SM100、SM120，mock `fp8_scaled_mm`，验证静态、预量化
  和动态输入在支持 native scalar-A scale 时的 scale 形状。
- `TestApplyFp8LinearScaleDispatch::test_without_native_scalar_a_static_scale_is_repeated`
  `:142-177`：验证不支持 scalar-A scale 的路径会把 scalar 复制成 `(M,1)`。
- `TestApplyFp8LinearScaleDispatch::test_linear_methods_forward_fused_scalar_tuple`
  `:179-227`：mock 原生 `Fp8LinearMethod::apply` 和
  `CompressedTensorsW8A8Fp8::apply_weights`，确认 `(fp8, scale, dtype)` 的 scale
  和 `pre_quant_output_dtype` 被正确转发。
- `TestApplyFp8LinearPrequantOutputDtype::_run` `:245-304` 及
  `::test_prequant_output_dtype` `:306-309`：对 FP16/BF16 验证预量化 FP8 输入的
  GEMM 输出遵循调用者传入 dtype，数值与未预量化 reference 接近；不传 dtype 时
  明确验证兼容性默认值为 BF16。
- 原有 `TestInverseTransformScaleUe8m0::test_round_trip` `:18-45` 仍保留，主要
  验证 UE8M0 scale transform 的 round-trip，并非本次 RMSNorm 融合测试。

#### Benchmark 中的 correctness 检查

`benchmark/kernels/bench_fused_rmsnorm_fp8_quant.py:1-185` 不是 unittest，但提供了
可运行的微基准：

- `run_unfused` `:67-74` 对比 RMSNorm 加独立 `static_quant_fp8`；
- `run_fused_default` `:87-89` 使用 FlashInfer 默认 fused kernel；
- `run_fused_cute` `:91-94` 使用可选 CuTe-DSL kernel；
- `_check_correctness` `:128-155` 在 hidden size 4096/8192、residual 有/无下比较
  反量化结果，要求 cosine `> 0.99`；
- `benchmark` `:175-179` 测量 token 数 512 到 16384 的延迟。

### 14.6 在 `sipu-dev` 中的实际测试结果和边界

按容器流程 source 环境后，对当前 editable checkout 的新增测试执行了：

```text
python3 test/registered/layers/test_layernorm_fusion.py -q
  Ran 0 tests ... OK (skipped=1)

python3 test/registered/quant/test_fp8_utils.py -q
  新增的两个测试类 skipped=2；原有 test_round_trip 直接申请 cuda tensor，
  因 Torch 未编译 CUDA 报 AssertionError: Torch not compiled with CUDA enabled
```

原因是容器使用 `torch==2.10.0+cpu` 加 `torch_sipu`，不是 CUDA Torch；同时
`flashinfer` 在容器中不可导入。第一个文件的 class-level skip 是预期行为，第二个
文件的失败来自旧测试没有 CUDA guard，不是新融合逻辑抛出的失败。

因此测试结论应准确表述为：

1. **有**针对提交核心功能的测试：覆盖 FlashInfer RMSNorm+FP8 的输出契约、数值、
   residual、dispatch guard，以及 SM90/100/120 的 scale dispatch 和 FP16/BF16
   输出 dtype。
2. 测试只在 CUDA + FlashInfer 环境运行；没有 SIPU 设备测试、SIMO/MXFP8/Marlin
   测试，也没有真实 Llama/Qwen 端到端请求把 RMSNorm tuple 一直跑到 FP8 GEMM。
3. 提交本身没有新增 `RMSNorm::forward_sipu` 实现；当前 SIPU 分支后来增加的
   SIPU 路径仍是独立实现。因此不能把这些 CUDA CI 的通过（或源码中存在测试）
   解读为 `sipu-dev` 上已经支持该融合。
4. 覆盖仍有空白：没有独立测试
   `python/sglang/srt/layers/layernorm.py:370-418（_fp8_static_input_scale / _is_static_per_tensor_fp8_linear）`
   对原生、compressed-tensors、block/MXFP8/Marlin 等识别和排除条件；也没有专门
   覆盖 `post_residual_addition`、高维/non-contiguous reshape、FlashInfer 不可用时
   的完整回退链。

最终答案：`11d03eaeef` 的核心是 **FlashInfer RMSNorm（含 residual add）与静态
per-tensor FP8 activation quant 的融合，以及预量化激活到 FP8 Linear 的 tuple/dtype
传递**；它有明确的 CUDA 单元测试和 benchmark correctness 检查，但目前没有 SIPU
或 SIMO 的对应测试覆盖。

## 15. `Engine(attention_backend="sipu", device="sipu")` 的最终实现与设备传播

本节中，`python/...` 路径都相对于 SGLang code base
`/share/users/like/package/sglang_sipu`。`sipu-dev` 的 mount 检查结果是
宿主机 `/share/users/like/package/sglang_sipu` 以读写方式挂载到容器
`/sgl-workspace/sglang`；`/sgl-workspace/sgl-kernel-sipu` 没有作为该 mount 的子目录
挂载，而是容器内单独的 editable kernel code base。下面带有
`sgl-kernel-sipu/...` 的路径，均相对于容器内 `/sgl-workspace/sgl-kernel-sipu`。

### 15.1 先给结论

对本次 `temp/sipu_offline_infer.py` 中的 4-layer Llama：

1. `attention_backend="sipu"` 最终构造的 Python backend 是
   `SIPUAttnBackend`，实现文件为
   `python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py`。
2. 该 Python 类不是计算公式的最底层实现。它在 prefill/decode 中调用
   `sgl_kernel.flash_attn_with_kvcache`，再通过 `torch.ops.sgl_kernel.fwd.default`
   进入 `csrc/attention/flash_attn.cpp::mha_fwd`，最后选择 SiKernel 的 paged
   decode、paged prefill 或 varlen prefill kernel。
3. `device="sipu"` 会成为 ModelRunner、模型加载器、KV pool 和大多数运行时
   metadata 的目标设备。因此普通 Transformer block 的权重、主 KV cache 数据和
   `req_to_token` 主表通常在 `sipu:0`。
4. 这不是对整个仓库所有 `torch.*` 调用的全局重写。当前代码在 SIPU 上默认把
   `embed_tokens` 和 `lm_head` 保留在 CPU；CPU staging/mirror、请求 bookkeeping、
   allocator 的临时计算，以及显式 offload/host cache 也可能在 CPU。因此“模型权重
   全部受 `device` 控制”的答案是**否**；“主 KV cache 是否受它控制”的答案是**是，
   但 host 副本和 bookkeeping 例外**。

### 15.2 `attention_backend="sipu"` 如何被解析

入口脚本在 `temp/sipu_offline_infer.py:4-17（main）` 将两个参数传给
`sgl.Engine`。`python/sglang/srt/entrypoints/engine.py:232-284（Engine::__init__）`
先把 kwargs 构造成 `ServerArgs`，然后启动 scheduler 子进程。

命令中的 `SGLANG_PLUGINS="mywhite"` 由
`python/sglang/srt/plugins/__init__.py:35-115（load_plugins_by_group）` 作为 entry-point
白名单处理；它决定哪些通用/平台插件被加载，不是 `attention_backend` 注册表本身。
`sipu` 的内置注册仍由 `python/sglang/srt/layers/attention/attention_registry.py:133-150（create_sipu_backend）`
提供。

`python/sglang/srt/server_args.py:1701-1709（ServerArgs::attention_backend）` 的
choices 包含 `sipu`；完整列表位于
`python/sglang/srt/server_args.py:182-211（ATTENTION_BACKEND_CHOICES）`。
`python/sglang/srt/server_args.py:3588-3590（ServerArgs::__post_init__）` 调用
`ServerArgs::_run_resolution_pipeline`（`python/sglang/srt/server_args.py:3591-3692`）。
其中：

- `python/sglang/srt/server_args.py:4260-4268（ServerArgs::_handle_missing_default_values）`
  在未指定设备时自动探测设备，并把类似 `sipu:0` 的字符串规范化为逻辑类型
  `sipu`；实际编号由 `gpu_id` 处理。
- `python/sglang/srt/server_args.py:4390-4402（ServerArgs::_handle_sipu_backends）`
  对 SIPU 设置默认 `page_size=32`（当用户没有显式设置时），并把 prefill
  compiler 限制为 eager；它不会把用户显式写的 `attention_backend="sipu"` 改成
  另一个普通 backend。
- `python/sglang/srt/arg_groups/overrides.py:277-290（attention_backends_of）`
  定义 prefill/decode split 的优先级：`prefill_attention_backend` 或
  `decode_attention_backend` 若存在，就优先于基础的 `attention_backend`；否则两者
  都回退到基础值。
- `python/sglang/srt/arg_groups/overrides.py:2118-2132（_attention_backend_default）`
  只在基础 backend 为空时填默认值，所以本脚本的显式 `sipu` 会继续传下去。

初始化顺序可写成：

```text
temp/sipu_offline_infer.py:4-17（main）
  -> Engine::__init__
       python/sglang/srt/entrypoints/engine.py:232-284
  -> ServerArgs::__post_init__ / ServerArgs::_run_resolution_pipeline
       python/sglang/srt/server_args.py:3588-3692
  -> Scheduler::init_model_worker / Scheduler::init_all_attention_backends
       python/sglang/srt/managers/scheduler.py:993-1006（Scheduler::init_model_worker）
       -> python/sglang/srt/managers/scheduler.py:981-985（Scheduler::init_all_attention_backends）
  -> ModelRunner::init_attention_backends
       python/sglang/srt/model_executor/model_runner.py:931-951
  -> resolve_attention_backend_strs -> build_attention_backends
       python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:69-178
  -> _build_full_attention_backend_from_str
       python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py:251-258
  -> ATTENTION_BACKENDS["sipu"] -> create_sipu_backend
       python/sglang/srt/layers/attention/attention_registry.py:133-150
```

`python/sglang/srt/layers/attention/attention_registry.py:34-39（register_attention_backend）`
用装饰器把字符串注册到 `ATTENTION_BACKENDS`。真正的 `sipu` 选择函数是
`python/sglang/srt/layers/attention/attention_registry.py:133-150（create_sipu_backend）`：

- 如果 `runner.model.modules()` 中有非空的 `indexer`，返回
  `SIPUDSAAttnBackend`。在当前仓库中它是
  `python/sglang/srt/hardware_backend/sipu/attention/sipu_dsa_backend.py:1618（SIPUDSAAttnBackend）`
  对 `DeepseekSparseAttnBackend` 的别名，面向 DSA/稀疏 MLA 模型。
- 否则返回
  `python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:95-121（SIPUAttnBackend::__init__）`。

本次模型是普通 dense Llama，没有 `indexer`，所以最终是
`SIPUAttnBackend`。因此，字符串 `"sipu"` 并不意味着所有模型都使用同一个类；
模型结构会参与最后一步选择。类似地，DeepSeek V4 等模型的
model-specific override 也可能把最终的 prefill/decode backend 解析成专用的
`dsv4`，不能只看构造函数参数就跳过解析流程。

### 15.3 从 Llama 层到 SIPU attention kernel 的调用链

在本次脚本的 `disable_cuda_graph=True` eager 路径，调用关系如下：

```text
ModelRunner::_forward_raw
  python/sglang/srt/model_executor/model_runner.py:1658-1669
  建立 ForwardContext(attn_backend=self.attn_backend)
      |
      v
get_attn_backend
  python/sglang/srt/model_executor/forward_context.py:46-67
      |
      v
LlamaDecoderLayer::forward
  python/sglang/srt/models/llama.py:350-378
  -> LlamaAttention::forward
       python/sglang/srt/models/llama.py:249-273
      |
      v
RadixAttention::forward
  python/sglang/srt/layers/radix_attention.py:150-287
      |
      v
AttentionBackend::forward
  python/sglang/srt/layers/attention/base_attn_backend.py:215-258
  -> SIPUAttnBackend::forward_extend（prefill）
     或 SIPUAttnBackend::forward_decode（decode）
```

`python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:384-796（SIPUAttnBackend::init_forward_metadata）`
根据当前 batch 的 `seq_lens.device` 创建 `cache_seqlens`、`cu_seqlens`、page table
等 metadata。普通 MHA 的两个执行函数是：

- `python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:798-1295（SIPUAttnBackend::forward_extend）`：先在
  `python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:820-865（SIPUAttnBackend::forward_extend）` 将新 K/V 写入 `token_to_kv_pool`，普通 paged MHA 在 `python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:1038-1058（SIPUAttnBackend::forward_extend）`
  调用 `flash_attn_with_kvcache`。
- `python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:1297-1607（SIPUAttnBackend::forward_decode）`：先在
  `python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:1310-1337（SIPUAttnBackend::forward_decode）` 写入新 K/V，随后在 `python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:1391-1397（SIPUAttnBackend::forward_decode）` 取出 cache；普通 self-attention
  在 `python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:1468-1488（SIPUAttnBackend::forward_decode）` 调用同一个 `flash_attn_with_kvcache`。

MLA 模型会走同一 Python 类中的 MLA 分支（`python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:1095-1295（SIPUAttnBackend::forward_extend）`、
`python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:1521-1607（SIPUAttnBackend::forward_decode）`），参数中会额外出现 `qv`/`k_rope`；有 `indexer` 的
DSA 模型则走 `sipu_dsa_backend.py`，不能把它和当前 dense Llama 路径混为一谈。

#### Python wrapper、C++ bridge 和 SiKernel

容器中真正被 import 的 `sgl_kernel` 文件是
`/sgl-workspace/sgl-kernel-sipu/sgl_kernel/flash_attn.py`，不是宿主机 SGLang
目录下的同名路径。相对 kernel code base 的实现链为：

1. `sgl-kernel-sipu/sgl_kernel/flash_attn.py:217-339（flash_attn_with_kvcache）`
   检查 SIPU 支持的参数、把 `q` 连续化，并在
   `sgl-kernel-sipu/sgl_kernel/flash_attn.py:300-336（flash_attn_with_kvcache）` 调用
   `torch.ops.sgl_kernel.fwd.default(...)`。`flash_attn_varlen_func` 位于
   `sgl-kernel-sipu/sgl_kernel/flash_attn.py:342-498（flash_attn_varlen_func）`，也通过同一个 `fwd` op 进入 native bridge。
2. `sgl-kernel-sipu/csrc/common_extension.cc:8（kSIPU）` 把 SIPU 映射为
   `torch::kPrivateUse1`；`sgl-kernel-sipu/csrc/common_extension.cc:266-327（TORCH_LIBRARY_FRAGMENT(sgl_kernel)）`
   定义 `fwd` schema，`sgl-kernel-sipu/csrc/common_extension.cc:328（TORCH_LIBRARY_FRAGMENT(sgl_kernel)）` 将
   `fwd` 的 PrivateUse1 dispatch 绑定到 `make_pytorch_shim(&mha_fwd)`。
3. `sgl-kernel-sipu/csrc/attention/flash_attn.cpp:51-529（mha_fwd）` 是 C++
   bridge。`sgl-kernel-sipu/csrc/attention/flash_attn.cpp:88-103（mha_fwd）` 检查 q/k/v 的 BF16、同 dtype 和 SIPU device；`sgl-kernel-sipu/csrc/attention/flash_attn.cpp:301-341（mha_fwd）`
   根据 page table、`max_seqlen_q`、head dimension 等条件判断 decode/paged
   prefill/varlen；不满足支持条件时，`sgl-kernel-sipu/csrc/attention/flash_attn.cpp:363-423（mha_fwd）` 调用
   `sgl_impl::mha_fwd_paged_torch` 或 `sgl_impl::mha_varlen_fwd_torch` fallback。
   满足条件时：
   - `sgl-kernel-sipu/csrc/attention/flash_attn.cpp:464-480（mha_fwd）` 调用 `mha_decode_with_kvcache`；
   - `sgl-kernel-sipu/csrc/attention/flash_attn.cpp:483-506（mha_fwd）` 调用 `mha_fwd_paged`；
   - `sgl-kernel-sipu/csrc/attention/flash_attn.cpp:507-524（mha_fwd）` 调用 `mha_varlen_fwd_bf16_any_len`。
4. 最底层 SiKernel `.su` 实现分别是：
   - `sgl-kernel-sipu/sikernel/source/source_builtin/attention/decode_attn/kernel/linearkv_mha_decode.su:24-130（mha_decode_with_kvcache）`，`sgl-kernel-sipu/sikernel/source/source_builtin/attention/decode_attn/kernel/linearkv_mha_decode.su:47-92（mha_decode_with_kvcache）` 校验 rank、page size、head dim，`sgl-kernel-sipu/sikernel/source/source_builtin/attention/decode_attn/kernel/linearkv_mha_decode.su:97-127（mha_decode_with_kvcache）` 按 head dim 64/128/256 选择异步 decode launcher；
   - `sgl-kernel-sipu/sikernel/source/source_builtin/attention/flash_attn/chunked_prefill_mla/kernel/kernel_mha_fwd_paged.su:233-551（mha_fwd_paged）`，`sgl-kernel-sipu/sikernel/source/source_builtin/attention/flash_attn/chunked_prefill_mla/kernel/kernel_mha_fwd_paged.su:471-550（mha_fwd_paged）` 按 `(head_size_qk, head_size_v)` 选择 `Config_64_64`、`Config_128_128`、`Config_192_128` 或 `Config_256_256`；
   - `sgl-kernel-sipu/sikernel/source/source_builtin/attention/flash_attn/flash_attn_prefill_bf16_1thread_1head/kernel/kernel_mha_varlen_fwd_bf16.su:243-360（mha_varlen_fwd_bf16_any_len）`，`sgl-kernel-sipu/sikernel/source/source_builtin/attention/flash_attn/flash_attn_prefill_bf16_1thread_1head/kernel/kernel_mha_varlen_fwd_bf16.su:301-344（mha_varlen_fwd_bf16_any_len）` 选择非 paged varlen 配置。

所以“最终实现在哪里”要分三层回答：

```text
SGLang backend object:
  python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py
native dispatch bridge:
  sgl-kernel-sipu/csrc/attention/flash_attn.cpp::mha_fwd
actual device kernel:
  sgl-kernel-sipu/sikernel/source/source_builtin/attention/.../*.su
```

`SIPUAttnBackend` 是 backend 的最终 Python 选择结果；`flash_attn_with_kvcache`
只是 Python API wrapper，真正执行哪一个 SiKernel kernel 还由 C++ bridge 根据 shape、
dtype、page layout 决定。即使选择了 `sipu` backend，也不能据此保证每一个边界 shape
都走 native kernel；不支持的组合会进入 `sgl_impl::*_torch` fallback。

### 15.4 `device="sipu"` 的传播路径

`device` 是一个逻辑设备类型和默认目标，不是把全仓库每个 tensor 的 `device=` 参数
进行字符串替换。它的主传播链是：

```text
ServerArgs.device = "sipu"
  -> ModelRunner.device
  -> DeviceConfig(torch.device("sipu"), gpu_id)
  -> model loader / KVCacheConfigurator / ForwardBatch
```

关键代码如下：

1. `python/sglang/srt/model_executor/model_runner.py:285-355（ModelRunner::__init__）`
   保存 `self.device = server_args.device`。`python/sglang/srt/model_executor/model_runner.py:383-393（ModelRunner::__init__）` 调用
   `torch.get_device_module(self.device).set_device(ps.gpu_id)` 设置当前 SIPU；
   `python/sglang/srt/model_executor/model_runner.py:404-415（ModelRunner::__init__）` 在当前实现中把 SIPU `forward_stream` 别名到 default stream。
2. `python/sglang/srt/configs/device_config.py:10-23（DeviceConfig::__init__）`
   接受 `sipu`，保存 `torch.device("sipu")` 和 `gpu_id`。
3. `python/sglang/srt/model_executor/model_runner.py:1061-1119（ModelRunner::load_model）`
   将 `self.device` 传给
   `python/sglang/srt/model_executor/model_runner_components/load_model_utils.py:268-325（load_model_with_memory_saver）`；后者在 `python/sglang/srt/model_executor/model_runner_components/load_model_utils.py:301-324（load_model_with_memory_saver）`
   创建 `DeviceConfig` 并调用 loader。
4. `python/sglang/srt/model_loader/loader.py:978-1017（DefaultModelLoader::load_model）`
   在 `python/sglang/srt/model_loader/loader.py:992-1001（DefaultModelLoader::load_model）` 建立 `target_device=torch.device(device_config.device)`，并在
   `with target_device:` 中调用 `_initialize_model`；随后
   `python/sglang/srt/model_loader/loader.py:1007-1009（DefaultModelLoader::load_model）` 调用
   `load_weights_and_postprocess(..., target_device)`。因此未被模型代码特别标记的
   普通参数和 buffer 会在 SIPU device context 中构造，权重也加载到这些目标参数。
5. `python/sglang/srt/model_loader/loader.py:151-160（device_loading_context）`
   对 SIPU 特意直接 `yield module`，没有执行通用的
   `p.data.to(target_device)`；注释说明 bulk move 会让 cmodel hang。这正是为什么
   “默认目标是 SIPU”和“所有参数无条件都在 SIPU”不能画等号：显式在 CPU 构造的
   模块会保持 CPU。

### 15.5 哪些 tensor 会落到 SIPU，哪些不会

| tensor / 对象 | 代码位置（相对 SGLang code base） | `device="sipu"` 下的实际结论 |
|---|---|---|
| 普通 Transformer block 参数、线性层权重、norm 权重 | `python/sglang/srt/model_loader/loader.py:978-1017（DefaultModelLoader::load_model）` | 通常在 `sipu:0`；模型在 SIPU default-device context 中创建并加载。 |
| `embed_tokens`、未绑定的 `lm_head` | `python/sglang/srt/models/cpu_embedding_lm_head.py:21-30（use_cpu_embedding_lm_head）`；`python/sglang/srt/models/cpu_embedding_lm_head.py:52-56（create_on_cpu）`；`python/sglang/srt/models/llama.py:395-411（LlamaModel::__init__）`、`python/sglang/srt/models/llama.py:559-595（LlamaForCausalLM::__init__）` | SIPU 默认 `SGLANG_CPU_EMBEDDING_LM_HEAD=true`，因此这些权重通常留在 CPU；`python/sglang/srt/models/cpu_embedding_lm_head.py:38-49（embed_input_ids）` 和 `python/sglang/srt/models/cpu_embedding_lm_head.py:59-113（logits_processor_with_cpu_lm_head）` 负责 CPU↔SIPU 拷贝。设置 `SGLANG_CPU_EMBEDDING_LM_HEAD=false` 才会选择普通 SIPU 构造路径。 |
| 主 MHA K/V cache | `python/sglang/srt/model_executor/model_runner.py:811-825（ModelRunner::alloc_memory_pool）` -> `python/sglang/srt/mem_cache/kv_cache_configurator.py:276-309（KVCacheConfigurator::configure）`；`python/sglang/srt/mem_cache/kv_cache_configurator.py:1570-1604（KVCacheConfigurator::_build_mha_kv_pool）` | pool 构造函数收到 `device=self.device`，标准主 K/V cache 在 SIPU。MLA、DSA、DSV4 的专用 pool builder 也把同一 device 向下传递。 |
| 标准 MHA K/V buffer | `python/sglang/srt/mem_cache/memory_pool.py:1761-1862（MHATokenToKVPool::__init__）`；`python/sglang/srt/mem_cache/memory_pool.py:1937-1954（MHATokenToKVPool::_create_buffers）`；`python/sglang/srt/mem_cache/memory_pool.py:2064-2115（MHATokenToKVPool::_create_buffers_normal）` | `torch.zeros(..., device=self.device)`，K/V 的物理 backing buffer 在 SIPU；`python/sglang/srt/mem_cache/memory_pool.py:2026-2049（MHATokenToKVPool::_init_data_ptrs_and_strides）` 的 pointer/stride metadata 也在 SIPU。KV dtype/layout 仍由 `kv_cache_dtype` 和 layout 配置决定，不由 `device` 决定。 |
| request-to-token page table | `python/sglang/srt/mem_cache/kv_cache_configurator.py:761-789（KVCacheConfigurator::_build_req_to_token_pool）`、`python/sglang/srt/mem_cache/kv_cache_configurator.py:924-946（KVCacheConfigurator::_build_default_req_pool）`；`python/sglang/srt/mem_cache/memory_pool.py:262-290（ReqToTokenPool::__init__）` | `req_to_token` 主表用传入 device 分配，通常为 `sipu:0`；但同一构造函数中的 `req_generation`（`python/sglang/srt/mem_cache/memory_pool.py:289（ReqToTokenPool::__init__）`）没有 device 参数，属于 CPU bookkeeping。 |
| 请求长度、page table、cu-seqlens 等 forward metadata | `python/sglang/srt/hardware_backend/sipu/attention/sipu_flashattention_backend.py:384-493（SIPUAttnBackend::init_forward_metadata）`；`python/sglang/srt/model_executor/forward_batch_info.py:809-848（ForwardBatch::init_new）` | 由 `seq_lens.device` 或 `model_runner.device` 创建，正常 eager SIPU forward 中在 `sipu:0`。 |
| `input_ids`、`seq_lens_cpu`、CPU 镜像和 pinned staging | `python/sglang/srt/managers/schedule_batch.py:2401-2411（ScheduleBatch::prepare_for_extend）`；`python/sglang/srt/mem_cache/allocation.py:303-315（alloc_for_extend）` | 明确保留 CPU/pinned CPU 版本，再异步或显式拷贝到 SIPU；它们不是 `device` 失效，而是调度设计的一部分。 |
| paged allocator 的临时索引计算 | `python/sglang/srt/mem_cache/allocator/paged.py:190-257（PagedTokenToKVPoolAllocator::alloc_extend）`、`python/sglang/srt/mem_cache/allocator/paged.py:259-307（PagedTokenToKVPoolAllocator::alloc_decode）` | 输出 `out_indices` 先在 SIPU 分配，但 `python/sglang/srt/mem_cache/allocator/paged.py:215-232（PagedTokenToKVPoolAllocator::alloc_extend）`、`python/sglang/srt/mem_cache/allocator/paged.py:275-284（PagedTokenToKVPoolAllocator::alloc_decode）` 因 Triton allocator 会 hang，先在 CPU 用 naive allocator 计算，再 copy 回 SIPU。 |
| host/offload/层级 cache 副本 | 由 `cpu_offload_gb`、weights CPU backup、HiCache/disaggregation 等独立配置控制 | 可能在 CPU 或远端 host；这是副本/迁移策略，不改变主 SIPU pool 的目标 device。 |

### 15.6 “模型权重、KV cache 都受它控制吗？”的直接回答

#### 模型权重

**大部分受控制，但不是无条件全部受控制。**

普通模型的初始化和权重加载目标是 `torch.device("sipu")`，所以 transformer block
的 linear、RMSNorm、RoPE cache 所依赖的参数通常驻留在 SIPU。当前 SIPU 代码又专门
提供了 CPU embedding/lm-head workaround：
`python/sglang/srt/models/cpu_embedding_lm_head.py:21-30（use_cpu_embedding_lm_head）`
在 SIPU 上默认返回 true；Llama 在
`python/sglang/srt/models/llama.py:395-411（LlamaModel::__init__）` 和
`python/sglang/srt/models/llama.py:569-595（LlamaForCausalLM::__init__）` 通过 `create_on_cpu` 创建 embedding 与
untied lm head。对于本脚本的 Llama 3.1 配置（`tie_word_embeddings=false`），这意味
着 embedding 和 lm head 默认是 CPU 权重，hidden state 在进入/离开它们时做搬运。

另外，CPU offload、weight backup、IPC/remote weight loader 或某个模型自己的
`load_weights` hook 也可以产生 CPU 保留或 host 副本；`device="sipu"` 不会覆盖显式
的 `device="cpu"`、`_keep_on_cpu` 标志或硬编码的其他 device。

#### KV cache

**主 KV cache 受控制。**

`ModelRunner::alloc_memory_pool` 在模型加载后调用
`KVCacheConfigurator::configure`，普通 Llama 最终走
`KVCacheConfigurator::_build_mha_kv_pool`，把 `self.device` 传给
`MHATokenToKVPool`。该 pool 的 K/V backing buffers、page pointer metadata 和
`req_to_token` 主表都以 SIPU 为目标，因此 attention kernel 看到的是
`sipu:0` 的 K/V cache。CPU 侧长度数组、allocator 临时结果、`req_generation` 和可选
host/offload 副本不属于这条“主 KV cache”结论。

还要区分三个独立参数：

- `device="sipu"`：决定设备类型和主 buffer 所在设备；
- `kv_cache_dtype`：决定 K/V 存储 dtype（例如 BF16、FP8/uint8 backing）；
- `page_size`、`mem_fraction_static`、`max_total_tokens`：决定页形状和容量。

改变后两类参数不会把 cache 搬到别的 device；改变 `device` 也不会自动改变 dtype
或容量。

### 15.7 2026-09-09 日志与源码的对应关系

`temp/off.log.2026_09_09___14_42_55` 提供了实际运行证据：

- `temp/off.log.2026_09_09___14_42_55:74-115` 是首次 prefill 的
  `flash_attn_with_kvcache`。`q=(13,32,128)`，
  `k_cache/v_cache=(9,32,8,128)`，page table 为 `(1,5)`，q/k/v 和 metadata 都是
  BF16 `device=sipu:0`；`temp/off.log.2026_09_09___14_42_55:109-112` 打印
  `Selected Config: Config_128_128`，与
  `kernel_mha_fwd_paged.su` 的 paged prefill 配置选择一致。
- `temp/off.log.2026_09_09___14_42_55:389-424` 是首次 decode，`q=(1,32,128)`、`max_seqlen_q=1`、page table 为
  `(1,1)`，同样由 `flash_attn_with_kvcache` 处理；这些条件满足 C++ bridge 的
  paged decode 判断，随后进入 `mha_decode_with_kvcache`。
- `temp/off.log.2026_09_09___14_42_55:52-59` 的 RMSNorm weight、
  `temp/off.log.2026_09_09___14_42_55:117-123` 的 fused RMSNorm input/residual/weight
  也都是 `sipu:0`，说明 transformer block 的运行张量确实在 SIPU。

日志只记录了 kernel API 名称，没有直接打印 Python backend 类名；
`SIPUAttnBackend` 的类名是由
`python/sglang/srt/layers/attention/attention_registry.py:133-150（create_sipu_backend）`
的模型结构判断推出的，而 `sipu:0` 参数和 `Config_128_128` 则由日志直接验证。

综上，本次运行可以概括为：

```text
attention_backend="sipu"
  选择 SIPUAttnBackend
  -> sgl_kernel.flash_attn_with_kvcache
  -> torch.ops.sgl_kernel.fwd (PrivateUse1/SIPU)
  -> mha_fwd
  -> SiKernel paged prefill/decode

device="sipu"
  设置当前 SIPU device
  -> 普通模型参数/激活/KV 主池/主 page table 使用 SIPU
  -> 显式 CPU embedding、CPU mirror、staging、bookkeeping、host cache 仍可在 CPU
```
