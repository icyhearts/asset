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
