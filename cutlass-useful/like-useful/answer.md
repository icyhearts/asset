## cutlass_test_unit_cute_core 中 swizzle_layout_like.cpp 的 printf 为什么打印不出来

结论：不是 cmake 没有重新编译，也不是 gtest 把 stdout 吃掉了；根因是 `test_swizzle_2d` 这个 namespace-scope 函数模板在两个 `.cpp` 文件中同名、同签名、同模板实参实例化，但函数体不同，造成 ODR 违规/weak COMDAT 符号碰撞。最终链接出的 `cutlass_test_unit_cute_core` 里，`CuTe_core.SwizzleLayout_like` 调用到的是另一个没有 debug `printf` 的 `test_swizzle_2d` 实例，所以 `swizzle_layout_like.cpp` 里 helper 内部的 `printf("<<<<<<<\n")` 没有执行。

相关代码位置：

- `test/unit/cute/core/swizzle_layout_like.cpp:42 test_swizzle_2d` 定义了一个全局命名空间的函数模板 `test_swizzle_2d(SwLayout const&)`。
- `test/unit/cute/core/swizzle_layout_like.cpp:45 test_swizzle_2d` 里面有你加的 `printf("<<<<<<<\n")`。
- `test/unit/cute/core/swizzle_layout_like.cpp:48 test_swizzle_2d` 和 `test/unit/cute/core/swizzle_layout_like.cpp:49 test_swizzle_2d` 里面还有 `sw_layout`、`sw_tensor` 相关打印。
- `test/unit/cute/core/swizzle_layout_like.cpp:97 TEST(CuTe_core, SwizzleLayout_like)` 定义了实际运行的 gtest case。
- `test/unit/cute/core/swizzle_layout_like.cpp:102 TEST(CuTe_core, SwizzleLayout_like)` 到 `test/unit/cute/core/swizzle_layout_like.cpp:105 TEST(CuTe_core, SwizzleLayout_like)` 的 `printf` 能在 `run.log` 中打印出来。
- `test/unit/cute/core/swizzle_layout_like.cpp:110 TEST(CuTe_core, SwizzleLayout_like)`、`test/unit/cute/core/swizzle_layout_like.cpp:117 TEST(CuTe_core, SwizzleLayout_like)`、`test/unit/cute/core/swizzle_layout_like.cpp:124 TEST(CuTe_core, SwizzleLayout_like)` 都调用了 `test_swizzle_2d(sw_layout)`。
- `test/unit/cute/core/swizzle_layout.cpp:41 test_swizzle_2d` 也定义了同名、同签名的全局命名空间函数模板 `test_swizzle_2d(SwLayout const&)`，但它的函数体没有你在 `_like.cpp` 中加入的 `printf("<<<<<<<\n")`、`printf("sw_layout:\n")` 等 debug 输出。
- `test/unit/cute/core/swizzle_layout.cpp:92 TEST(CuTe_core, SwizzleLayout)` 是另一个测试 case，也会实例化同名 `test_swizzle_2d`。

日志现象能说明 gtest 没有屏蔽 stdout。`run.log` 中能看到 `test/unit/cute/core/swizzle_layout_like.cpp:102 TEST(CuTe_core, SwizzleLayout_like)` 到 `test/unit/cute/core/swizzle_layout_like.cpp:105 TEST(CuTe_core, SwizzleLayout_like)` 里的 header 输出：

```text
auto sw_layout = composition(Swizzle<3,0,3>{},
                   Layout<Shape <_8,_8>,
                          Stride<_8,_1>>{})
====================---
```

但是 `test/unit/cute/core/swizzle_layout_like.cpp:45 test_swizzle_2d` 的 `<<<<<<<` 没有出现在 `run.log`，说明执行已经进入了 `_like` 的 gtest `TestBody()`，但后续 helper 调用没有进入 `_like.cpp` 中那个带 debug 输出的 helper 实现。

`make.log` 也能排除“没重新编译”的可能：日志里显示 `swizzle_layout_like.cpp.o` 被重新编译，并且最终链接进 `cutlass_test_unit_cute_core`。链接命令里同时出现了 `swizzle_layout.cpp.o` 和 `swizzle_layout_like.cpp.o`，而且 `swizzle_layout.cpp.o` 排在 `swizzle_layout_like.cpp.o` 前面。两个目标文件都提供相同名字的模板实例时，最终链接器只保留/选择其中一个 weak 实现；当前现象对应的是选择了 `swizzle_layout.cpp` 里的无打印版本。

可以用下面的方式验证这个判断：

```bash
nm -C build-bjh100/test/unit/cute/core/CMakeFiles/cutlass_test_unit_cute_core.dir/swizzle_layout.cpp.o | rg "test_swizzle_2d"
nm -C build-bjh100/test/unit/cute/core/CMakeFiles/cutlass_test_unit_cute_core.dir/swizzle_layout_like.cpp.o | rg "test_swizzle_2d"
nm -C build-bjh100/test/unit/cute/core/cutlass_test_unit_cute_core | rg "test_swizzle_2d|SwizzleLayout_like_Test|SwizzleLayout_Test"
```

两个 `.o` 里都会看到 `W void test_swizzle_2d<...>(...)` 这样的 weak 模板实例符号；最终可执行文件中也能看到 `CuTe_core_SwizzleLayout_like_Test::TestBody()`，同时只存在最终被选中的一组 `test_swizzle_2d<...>` weak 实例。反汇编还能看到 `CuTe_core_SwizzleLayout_like_Test::TestBody()` 先调用自己的 header `printf`，然后调用最终二进制中的 `test_swizzle_2d<...>` 地址；这个地址不是 `_like.cpp` 私有的唯一实现，而是链接后被合并/选择出来的全局 weak 实例。

所以即使最终二进制里用 `strings` 能搜到 `<<<<<<<` 或 `sw_layout:`，也不能说明这段代码实际会执行。`swizzle_layout_like.cpp.o` 被链接进来了，相关字符串可能还在 `.rodata` 里；但 call 解析到的 `test_swizzle_2d<...>` 实现不是含有这些打印语句的那份。

建议修法：

1. 最直接：把 `test/unit/cute/core/swizzle_layout_like.cpp:42 test_swizzle_2d` 改名成唯一名字，例如 `test_swizzle_2d_like`，并同步修改 `test/unit/cute/core/swizzle_layout_like.cpp:110 TEST(CuTe_core, SwizzleLayout_like)`、`test/unit/cute/core/swizzle_layout_like.cpp:117 TEST(CuTe_core, SwizzleLayout_like)`、`test/unit/cute/core/swizzle_layout_like.cpp:124 TEST(CuTe_core, SwizzleLayout_like)` 的调用。
2. 或者把 helper 放进匿名 namespace，让它具有 translation-unit internal linkage。例如在 `swizzle_layout_like.cpp` 中写 `namespace { template <class SwLayout> void test_swizzle_2d(...) { ... } }`。如果 `swizzle_layout.cpp` 中也可能和别的文件同名，那里也建议同样处理。
3. 也可以把 helper 声明成 `static` 模板函数，但测试 `.cpp` 里更常见、更干净的写法是匿名 namespace。

不要依赖链接顺序解决这个问题。当前链接顺序碰巧让 `swizzle_layout.cpp` 的无打印版本赢了；换编译器、链接器、优化选项或目标文件顺序后行为可能变化，但本质上两个不同函数体共享同一个外部链接模板名字已经是不可靠的。

## cutlass_test_unit_cute_core 编译错误：print_tensor 未声明

`make.log` 里的直接错误是：

```text
test/unit/cute/core/swizzle_layout_like.cpp:52:15: error: 'print_tensor' was not declared in this scope
```

报错发生在 `test/unit/cute/core/swizzle_layout_like.cpp:42 test_like_swizzle_2d` 这个函数模板实例化时。具体代码是 `test/unit/cute/core/swizzle_layout_like.cpp:52 test_like_swizzle_2d` 调用了 `print_tensor(sw_tensor)`，但是当前文件没有把 `print_tensor` 的声明包含进来。

相关代码位置：

- `test/unit/cute/core/swizzle_layout_like.cpp:36 test_like_swizzle_2d` 包含了 `<cute/tensor_impl.hpp>`。
- `test/unit/cute/core/swizzle_layout_like.cpp:37 test_like_swizzle_2d` 包含了 `<cute/swizzle_layout.hpp>`。
- `test/unit/cute/core/swizzle_layout_like.cpp:38 test_like_swizzle_2d` 把 `#include <cute/util/print_tensor.hpp>` 注释掉了。
- `test/unit/cute/core/swizzle_layout_like.cpp:52 test_like_swizzle_2d` 调用了 `print_tensor(sw_tensor)`。
- `include/cute/util/print_tensor.hpp:104 print_tensor` 才是 `print_tensor(Tensor<Engine,Layout> const&, bool)` 的定义位置。
- `include/cute/tensor.hpp:63 cute/tensor.hpp` 也会间接包含 `<cute/util/print_tensor.hpp>`。
- `test/unit/cute/core/swizzle_layout.cpp:47 test_swizzle_2d` 里的 `print_tensor(sw_tensor)` 是注释状态，所以原始 `swizzle_layout.cpp` 不会触发这个错误。

最小修复：如果确实要在 `test_like_swizzle_2d` 里打印完整 tensor，就取消 `test/unit/cute/core/swizzle_layout_like.cpp:38 test_like_swizzle_2d` 的注释：

```cpp
#include <cute/util/print_tensor.hpp>
```

并建议把调用写成带 namespace 的形式，避免以后读代码时误判这个函数来自哪里：

```cpp
cute::print_tensor(sw_tensor);
```

也就是说，修复后的关键片段是：

```cpp
#include <cute/tensor_impl.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/util/print_tensor.hpp>

template <class SwLayout>
void
test_like_swizzle_2d(SwLayout const& sw_layout)
{
  using namespace cute;
  auto sw_tensor = make_tensor(counting_iterator<int>{0}, sw_layout);
  cute::print_tensor(sw_tensor);
}
```

如果只是想让 `cutlass_test_unit_cute_core` 编译通过，而不需要 `print_tensor` 的二维 pretty print，另一个更小的修法是把 `test/unit/cute/core/swizzle_layout_like.cpp:52 test_like_swizzle_2d` 再注释掉或删掉。因为 `test/unit/cute/core/swizzle_layout_like.cpp:50 test_like_swizzle_2d` 已经有 `print(sw_tensor); printf("\n");`，它不依赖 `print_tensor.hpp`。

这个编译错误和前面 `printf` 打不出来的问题是两个独立问题。上一个问题是两个 `.cpp` 中同名 `test_swizzle_2d` 模板的 weak 符号碰撞；现在 `test/unit/cute/core/swizzle_layout_like.cpp:42 test_like_swizzle_2d` 已经改成了不同名字，避开了那个链接期问题。当前失败发生在编译期，原因只是 `test/unit/cute/core/swizzle_layout_like.cpp:52 test_like_swizzle_2d` 使用了未包含声明的 `print_tensor`。

推荐最终做法：

1. 保留 `test_like_swizzle_2d` 这个唯一 helper 名字，避免再次和 `test/unit/cute/core/swizzle_layout.cpp:41 test_swizzle_2d` 冲突。
2. 如果需要完整打印 tensor，打开 `#include <cute/util/print_tensor.hpp>`，并使用 `cute::print_tensor(sw_tensor);`。
3. 如果只是做单元测试，不需要额外输出，删除或注释 `print_tensor(sw_tensor)`，减少测试日志噪声。

## cutlass_test_unit_cute_core 编译错误：print_tensor.hpp 依赖 pointer_flagged.hpp

当前 `make.log` 的错误已经不是 `print_tensor` 未声明，而是包含了 `<cute/util/print_tensor.hpp>` 之后，`print_tensor.hpp` 自己内部用到的类型/函数没有提前声明：

```text
include/cute/util/print_tensor.hpp:92:39: error: 'smem_ptr_flag_bits' was not declared in this scope
include/cute/util/print_tensor.hpp:94:16: error: there are no arguments to 'as_position_independent_swizzle_layout' ...
```

触发路径是：

- `test/unit/cute/core/swizzle_layout_like.cpp:38 file-scope include` 现在直接包含了 `<cute/util/print_tensor.hpp>`。
- `test/unit/cute/core/swizzle_layout_like.cpp:52 test_like_swizzle_2d` 调用了 `print_tensor(sw_tensor)`。
- `include/cute/util/print_tensor.hpp:92 print_layout` 定义了面向 `ComposedLayout<SwizzleFn, smem_ptr_flag_bits<B>, Layout>` 的 `print_layout` 重载。
- `include/cute/util/print_tensor.hpp:94 print_layout` 调用了 `as_position_independent_swizzle_layout(layout)`。
- `include/cute/pointer_flagged.hpp:51 smem_ptr_flag_bits` 才定义了 `smem_ptr_flag_bits`。
- `include/cute/pointer_flagged.hpp:93 as_position_independent_swizzle_layout` 才定义了 `as_position_independent_swizzle_layout`。

所以只包含 `<cute/util/print_tensor.hpp>` 不够。`print_tensor.hpp` 中的 `print_layout` 重载依赖 `pointer_flagged.hpp`，但当前 `swizzle_layout_like.cpp` 的 include 顺序没有先把 `pointer_flagged.hpp` 拉进来。

推荐的最小修复是在 `print_tensor.hpp` 前面显式包含 `pointer_flagged.hpp`：

```cpp
#include <cute/tensor_impl.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/util/print_tensor.hpp>
```

然后在 `test/unit/cute/core/swizzle_layout_like.cpp:52 test_like_swizzle_2d` 处最好写成：

```cpp
cute::print_tensor(sw_tensor);
```

这个修复比直接把 `test/unit/cute/core/swizzle_layout_like.cpp:36 file-scope include` 的 `<cute/tensor_impl.hpp>` 换成 `<cute/tensor.hpp>` 更贴近 CUTLASS 的本地风格。原因是 `include/cute/tensor_impl.hpp:38 tensor_impl.hpp` 的文件说明建议 CUTLASS 内部尽量使用 `tensor_impl.hpp` 加具体所需头文件，避免直接包含大入口 `tensor.hpp`。当然，`include/cute/tensor.hpp:41 tensor.hpp` 会先包含 `<cute/pointer_flagged.hpp>`，`include/cute/tensor.hpp:63 tensor.hpp` 再包含 `<cute/util/print_tensor.hpp>`，所以直接包含 `<cute/tensor.hpp>` 也能绕过这个错误，只是依赖面更大。

我用当前 `make.log` 中同一条 `swizzle_layout_like.cpp` 编译命令做了临时验证：不改源码，只额外加 `-include cute/pointer_flagged.hpp` 后，对 `test/unit/cute/core/swizzle_layout_like.cpp` 的单文件编译可以通过。因此当前这轮编译错误的直接修复就是让 `pointer_flagged.hpp` 在 `print_tensor.hpp` 之前可见。

最终建议：

1. 保留 `test_like_swizzle_2d`，继续避免和 `test/unit/cute/core/swizzle_layout.cpp:41 test_swizzle_2d` 的同名模板冲突。
2. 在 `test/unit/cute/core/swizzle_layout_like.cpp:38 file-scope include` 附近加入 `#include <cute/pointer_flagged.hpp>`，位置放在 `<cute/util/print_tensor.hpp>` 前。
3. 如果不需要 `print_tensor` 的 pretty-print 输出，最干净的测试修法仍然是删除或注释 `test/unit/cute/core/swizzle_layout_like.cpp:52 test_like_swizzle_2d`，这样也不需要新增 `print_tensor.hpp` 和 `pointer_flagged.hpp` 依赖。

## SwizzleLayout_like 第一个例子中 print_tensor 如何调用到 Swizzle::apply

本节只看 `TEST(CuTe_core_like, SwizzleLayout_like)` 的第一个例子：

```cpp
auto sw_layout = composition(Swizzle<3,0,3>{},
                             Layout<Shape <_8,_8>,
                                    Stride<_8,_1>>{});
```

当前 `run.log` 中对应的 layout 打印为：

```text
Sw<3,0,3> o _0 o (_8,_8):(_8,_1)
```

这表示一个 `ComposedLayout<Swizzle<3,0,3>, _0, Layout<Shape<_8,_8>, Stride<_8,_1>>>`，即先用普通 8x8 row-major layout 把 `(m,n)` 映射成线性下标，再对这个线性下标应用 `Swizzle<3,0,3>`。

调用链路如下：

1. `test/unit/cute/core/swizzle_layout_like.cpp:110 TEST(CuTe_core_like, SwizzleLayout_like)` 到 `test/unit/cute/core/swizzle_layout_like.cpp:112 TEST(CuTe_core_like, SwizzleLayout_like)` 构造 `sw_layout`。
2. `include/cute/swizzle_layout.hpp:321 composition` 接收 `Swizzle<B,M,S>` 和普通 `Layout`，并在 `include/cute/swizzle_layout.hpp:324 composition` 转成 `composition(sxor, Int<0>{}, layout)`。
3. `include/cute/layout_composed.hpp:363 composition` 到 `include/cute/layout_composed.hpp:367 composition` 构造 `ComposedLayout<LayoutA, Offset, LayoutB>`，所以这里得到 `Swizzle<3,0,3> o _0 o Layout<Shape<_8,_8>,Stride<_8,_1>>`。
4. `test/unit/cute/core/swizzle_layout_like.cpp:122 TEST(CuTe_core_like, SwizzleLayout_like)` 调用 `test_like_swizzle_2d(sw_layout)`。
5. `test/unit/cute/core/swizzle_layout_like.cpp:48 test_like_swizzle_2d` 用 `make_tensor(counting_iterator<int>{0}, sw_layout)` 构造 tensor；`test/unit/cute/core/swizzle_layout_like.cpp:54 test_like_swizzle_2d` 调用 `print_tensor(sw_tensor)`。
6. `include/cute/tensor_impl.hpp:409 make_tensor` 接收 iterator 和 layout，`include/cute/tensor_impl.hpp:413 make_tensor` 返回对应的 `Tensor`。
7. `include/cute/util/print_tensor.hpp:104 print_tensor` 进入打印函数。因为这个 tensor 的 layout rank 是 2，走 `include/cute/util/print_tensor.hpp:117 print_tensor` 的 rank-2 分支。
8. `include/cute/util/print_tensor.hpp:119 print_tensor` 和 `include/cute/util/print_tensor.hpp:120 print_tensor` 双层遍历 `m,n`；`include/cute/util/print_tensor.hpp:121 print_tensor` 调用 `pretty_print(tensor(m,n))`。
9. `include/cute/tensor_impl.hpp:272 Tensor::operator()` 把 `tensor(m,n)` 转成 `operator()(make_coord(m,n))`；`include/cute/tensor_impl.hpp:255 Tensor::operator()` 返回 `data()[layout()(coord)]`。
10. `include/cute/layout_composed.hpp:114 ComposedLayout::operator()` 进入 composed layout 的坐标映射；`include/cute/layout_composed.hpp:118 ComposedLayout::operator()` 执行 `layout_a()(offset() + layout_b()(coord))`。
11. 这里的 `layout_b` 是 `Layout<Shape<_8,_8>,Stride<_8,_1>>`。`include/cute/layout.hpp:167 Layout::operator()` 处理普通 layout 坐标；`include/cute/layout.hpp:171 Layout::operator()` 调用 `crd2idx(coord, shape(), stride())`，所以 `(m,n)` 变成 `8*m + n`。
12. 这里的 `layout_a` 是 `Swizzle<3,0,3>`。`include/cute/swizzle.hpp:84 Swizzle::operator()` 调用 `include/cute/swizzle.hpp:86 Swizzle::operator()` 的 `apply(offset)`。
13. `include/cute/swizzle.hpp:76 Swizzle::apply` 是真正的 swizzle 映射；`include/cute/swizzle.hpp:78 Swizzle::apply` 的核心公式是 `offset ^ shiftr(offset & yyy_msk{}, msk_sft{})`。
14. 最后，因为 data 是 `counting_iterator<int>{0}`，`include/cute/pointer_base.hpp:208 counting_iterator::operator[]` 返回 `n_ + i`。这里 `n_` 是 0，所以 `data()[layout()(coord)]` 的值就是 swizzle 后的线性下标本身。

所以 `print_tensor(sw_tensor)` 打印的不是内存中某个真实矩阵的数据，而是每个逻辑坐标 `(m,n)` 经过 `sw_layout` 映射后的线性 index。

对第一个例子，普通 layout 是 `(_8,_8):(_8,_1)`，因此：

```text
layout_b(m,n) = 8*m + n
```

`Swizzle<3,0,3>` 的模板参数含义来自 `include/cute/swizzle.hpp:54 Swizzle`：

- `BBits = 3`
- `MBase = 0`
- `SShift = 3`

根据 `include/cute/swizzle.hpp:66 Swizzle` 到 `include/cute/swizzle.hpp:69 Swizzle`：

```text
bit_msk = (1 << 3) - 1 = 0b111
yyy_msk = 0b111 << (0 + max(0,3)) = 0b111000
zzz_msk = 0b111 << (0 - min(0,3)) = 0b000111
msk_sft = 3
```

再代入 `include/cute/swizzle.hpp:78 Swizzle::apply`：

```text
Swizzle<3,0,3>::apply(offset)
  = offset ^ ((offset & 0b111000) >> 3)
```

对于 8x8 row-major layout，`offset = 8*m + n`，二进制可以写成：

```text
offset = 0bmmmnnn
```

其中 `mmm` 是行号 `m`，`nnn` 是列号 `n`。因此：

```text
(offset & 0b111000) >> 3 = m
apply(offset) = 0bmmmnnn ^ 0b000mmm
              = 0bmmm(nnn xor mmm)
              = 8*m + (n xor m)
```

这就是 `run.log` 中第一组 `print_tensor` 输出的规律：每一行的高 3 bit，也就是行块 `8*m`，保持不变；低 3 bit，也就是列号 `n`，被行号 `m` 做了一次 XOR。

逐行展开如下：

```text
m=0: 8*0 + (n xor 0) =  0  1  2  3  4  5  6  7
m=1: 8*1 + (n xor 1) =  9  8 11 10 13 12 15 14
m=2: 8*2 + (n xor 2) = 18 19 16 17 22 23 20 21
m=3: 8*3 + (n xor 3) = 27 26 25 24 31 30 29 28
m=4: 8*4 + (n xor 4) = 36 37 38 39 32 33 34 35
m=5: 8*5 + (n xor 5) = 45 44 47 46 41 40 43 42
m=6: 8*6 + (n xor 6) = 54 55 52 53 50 51 48 49
m=7: 8*7 + (n xor 7) = 63 62 61 60 59 58 57 56
```

这正好对应 `run.log` 第一组 `print_tensor` 的结果：

```text
    0    1    2    3    4    5    6    7
    9    8   11   10   13   12   15   14
   18   19   16   17   22   23   20   21
   27   26   25   24   31   30   29   28
   36   37   38   39   32   33   34   35
   45   44   47   46   41   40   43   42
   54   55   52   53   50   51   48   49
   63   62   61   60   59   58   57   56
```

因此它不是无规律，而是 `Swizzle<3,0,3>` 把 row-major index 的 bit[5:3] 复制到低 3 bit 上做 XOR。直观说：第 `m` 行内部，列号按 `n xor m` 重新排列；不跨行搬移，因为高 3 bit `m` 保持不变。
