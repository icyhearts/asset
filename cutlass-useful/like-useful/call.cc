caller:  TiledMMA mma2 = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
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
  return make_tiled_mma(MMA_Atom<MMA_Op>{}, thr_layout/*  Layout<Shape<_1,_1,_1>>{} */, permutations/* Tile<_8,_8,_4>{}*/);
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
  auto thr_layout_mnk  = append<3>(thr_layout, Layout<_1,_0>{});//=Layout<Shape<_1,_1,_1>>
  auto permutation_mnk = append<3>(permutations, _); //Tile<_8,_8,_4>{}

  return TiledMMA<MMA_Atom<MMA_Op>,//MMA_Atom<MMA_Op>{}=MMA_Atom<SM70_8x8x4_F32F16F16F32_NT>
                  decltype(thr_layout_mnk), // Layout<Shape<_1,_1,_1>>
                  decltype(permutation_mnk)/*Tile<_8,_8,_4> */>{mma_atom, thr_layout_mnk/*Layout<Shape<_1,_1,_1>> */};
}
step3: ctor of the following classs:

template <class MMA_Atom,//= MMA_Atom<SM70_8x8x4_F32F16F16F32_NT>
          class AtomLayoutMNK,// Layout<Shape<_1,_1,_1>>
          class PermutationMNK = Tile<Underscore,Underscore,Underscore>>// Tile<_8,_8,_4>
struct TiledMMA : MMA_Atom {
  using AtomThrID      = typename MMA_Atom::ThrID;//= typename MMA_Atom<SM70_8x8x4_F32F16F16F32_NT>::ThrID= SM70_QuadPair=Layout<Shape <_4, _2>, Stride<_1,_16>>
  CUTE_HOST_DEVICE constexpr
  TiledMMA(MMA_Atom const& mma_atom = {}, AtomLayoutMNK const& thr_layout_mnk = {})
    : MMA_Atom(mma_atom),
      thr_layout_vmnk_(tiled_product(AtomThrID{}, thr_layout_mnk)) {}// AtomThrID: Layout<Shape <_4, _2>, Stride<_1,_16>>, thr_layout_mnk==Layout<Shape<_1,_1,_1>>
}
