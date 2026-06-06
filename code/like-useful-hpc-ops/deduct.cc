# <cutlass::float_e4m3_t, cutlass::bfloat16_t,  64,          128,         128,         8,
template <typename Tin_, typename Tout_,       int kTileM_, int kTileN_, int kTileK_, int kStage_,
#           2,                   1,                     128,                     128,
          int kWarpgroupM_ = 2, int kWarpgroupN_ = 1, int kSwizzleX = 128, int kSwizzleW = 128,
#          64>
          int kSwizzleY = 128>
struct GroupGEMMFp8Config {

# using SLayoutXAtom = decltype(slayout_selector<128, float_e4m3_t>());
  using SLayoutXAtom = decltype(slayout_selector<kSwizzleX, Tin>());
    |{ // slayout_selector 定义如下:
      #template <128,          float_e4m3_t, bool kKmajor = true>
      template <int kSwizzle, typename T, bool kKmajor = true>
      static constexpr auto slayout_selector()
      // hit
      if constexpr (kSwizzle == 128) {
        if constexpr (kKmajor) {
          #return cute::GMMA::Layout_K_SW128_Atom<float_e4m3_t>{};// hit
          return cute::GMMA::Layout_K_SW128_Atom<T>{};// hit
        }
      }
    }
}
//step2:
using SLayoutXAtom = cute::GMMA::Layout_K_SW128_Atom<float_e4m3_t>

`3rd/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:103-104`:

#template <class float_e4m3_t>
#using Layout_K_SW128_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW128_Atom_Bits{}));
template <class Type>
using Layout_K_SW128_Atom = decltype(upcast<sizeof_bits<Type>::value>(Layout_K_SW128_Atom_Bits{}));


其中 `Layout_K_SW128_Atom_Bits` 同文件 line 84：

```cpp
using Layout_K_SW128_Atom_Bits = ComposedLayout<
    Swizzle<3,4,3>,                        // 3-bit XOR swizzle, 4 LS bits unchanged
    smem_ptr_flag,                         // = smem_ptr_flag_bits<1>: 未设置的指针占位符
    Layout<Shape<_8, _1024>,               // 8 rows × 1024 columns (in BITS)
           Stride<_1024, _1>>
>;
```

`sizeof_bits<float_e4m3_t>` = 8。所以
`Layout_K_SW128_Atom<float_e4m3_t>` = `decltype(upcast<8>(Layout_K_SW128_Atom_Bits{}))`。

upcast<8>(Int<8>, Int<1024>)

```
upcast<8>(Layout<Shape<Int<8>, Int<1024>>, Stride<Int<1024>, Int<1>>>)
  │
  │  upcast<8>(layout.shape(), layout.stride())
  │
  ├─► Branch 1: is_tuple<Shape<Int<8>,Int<1024>>> = true  ← 唯一命中
  │     │
  │     │  transform_layout: 逐对调用 upcast<8>
  │     │
  │     ├─► upcast<8>(Int<8>, Int<1024>)
  │     │     │
  │     │     ├─► Branch 1: is_tuple<Int<8>> = false  ← 跳过
  │     │     ├─► Branch 2: is_constant<0, Int<1024>> = false (1024≠0)  ← 跳过
  │     │     └─► Branch 3: is_static<Int<1024>> = true  ← 选中
  │     │           shape= =ceil_div(8, ceil_div(8, 1024)) =ceil_div(8, 1) =8
  │     │           stride= signum(1024) * ceil_div(1024,8) =1*128
  │     │           结果: (shape=Int<8>, stride=Int<128>)
  │     │
  │     └─► upcast<8>(Int<1024>, Int<1>)
  │           │shape=ceil_div(1024,ceil_div(8, 1))=ceil_div(1024,8)=128
  │           │stride=signum(1) * ceil_div(*1,8)=1*1=1
  │           ├─► Branch 1: is_tuple<Int<1024>> = false  ← 跳过
  │           ├─► Branch 2: is_constant<0, Int<1>> = false (1≠0)  ← 跳过
  │           └─► Branch 3: is_static<Int<1>> = true  ← 选中
  │                 结果: (shape=Int<128>, stride=Int<1>)
  │
  └─► 最终: Layout<Shape<Int<8>,Int<128>>, Stride<Int<128>,Int<1>>>
```
