# simo × mma_dte 方案探针

`reuse_probe.cpp` — 验证「只链接 libtorch_sipu.so，不编译任何设备代码，
即可调用 torch 已编好的 MX mma_dte kernel」的可行性探针。

## 正确的 include 顺序（关键）

SiPU 的 `c10/macros/Macros.h` 用 `#include_next`，所以 **SiPU include 目录必须排在
upstream torch include 之前**，否则 `C10_SIPU_API` 未定义：

    c++ -std=c++20 \
        -I$SP/include \                    # torch_sipu wheel 的 include（必须在前）
        $TORCH_INC \                       # torch 自身的 include
        -I$SDK/include -I$SDK/include/SiTe -I$SDK/include/sipurt \
        ...

## 链接

需要同时提供 torch 主库与 SDK 库：

    -L$SP/lib -ltorch_sipu -ltorch -ltorch_cpu -lc10
    -L$SDK/lib
    -Wl,-rpath,$SP/lib:$SDK/lib

`libtorch_sipu.so` 导出全部 MX 叶子符号（`mx_mma_dte_s1x4_m32_mxint8_fp32` 等），
所以 **不需要** `-ltile_mma_dte`，也不需要 component DSO。
