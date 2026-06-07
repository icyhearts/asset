  TiledMMA mma2 = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                                Layout<Shape<_1,_1,_1>>{},   // Layout of Atoms
                                Tile<_8,_8,_4>{});           // Tiler
step1:
| call
template <class MMA_Op,
          class MMAThrLayout = Layout<Shape<_1,_1,_1>>,
          class Permutations = Tile<Underscore,Underscore,Underscore>>
CUTE_HOST_DEVICE constexpr
auto
make_tiled_mma(MMA_Op       const&,
               MMAThrLayout const& thr_layout   = {},
               Permutations const& permutations = {})
{
  // Attempt to wrap in an MMA_Atom<> and forward
  return make_tiled_mma(MMA_Atom<MMA_Op>{}, thr_layout, permutations);
}
step2

template <class MMA_Op,
          class MMAThrLayout = Layout<Shape<_1,_1,_1>>,
          class Permutations = Tile<Underscore,Underscore,Underscore>>
CUTE_HOST_DEVICE constexpr
auto
make_tiled_mma(MMA_Atom<MMA_Op> const& mma_atom, //actual=MMA_Atom<MMA_Op>{}=MMA_Atom<SM70_8x8x4_F32F16F16F32_NT>{}
               MMAThrLayout     const& thr_layout   = {},//=actual=Layout<Shape<_1,_1,_1>>
               Permutations     const& permutations = {})//actual=Tile<_8,_8,_4>{}
{
  auto thr_layout_mnk  = append<3>(thr_layout, Layout<_1,_0>{});
  auto permutation_mnk = append<3>(permutations, _);

  return TiledMMA<MMA_Atom<MMA_Op>,
                  decltype(thr_layout_mnk),
                  decltype(permutation_mnk)>{mma_atom, thr_layout_mnk};
}
