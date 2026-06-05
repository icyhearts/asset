# <cutlass::float_e4m3_t, cutlass::bfloat16_t,  64,          128,         128,         8,
template <typename Tin_, typename Tout_,       int kTileM_, int kTileN_, int kTileK_, int kStage_,
#           2,                   1,                     128,                     128,
          int kWarpgroupM_ = 2, int kWarpgroupN_ = 1, int kSwizzleX = 128, int kSwizzleW = 128,
#          64>
          int kSwizzleY = 128>
struct GroupGEMMFp8Config {

# using SLayoutXAtom = decltype(slayout_selector<128, float_e4m3_t>());
  using SLayoutXAtom = decltype(slayout_selector<kSwizzleX, Tin>());
    |{
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

`sizeof_bits<float_e4m3_t>` = 8。所以 `Layout_K_SW128_Atom<float_e4m3_t>` = `decltype(upcast<8>(Layout_K_SW128_Atom_Bits{}))`。
