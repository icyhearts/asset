hpc::group_gemm::GroupGEMMFp8Config<
  cutlass::float_e4m3_t, cutlass::bfloat16_t, 64,          128,         128,         8,           2, 1, 128, 128, 64>
template <typename Tin_, typename Tout_,      int kTileM_, int kTileN_, int kTileK_, int kStage_, int kWarpgroupM_ = 2, int kWarpgroupN_ = 1, int kSwizzleX = 128, int kSwizzleW = 128, int kSwizzleY = 128>

template <typename Tin_, typename Tout_, int kTileM_, int kTileN_, int kTileK_, int kStage_,
          int kWarpgroupM_ = 2, int kWarpgroupN_ = 1, int kSwizzleX = 128, int kSwizzleW = 128,
          int kSwizzleY = 128>
struct GroupGEMMFp8Config {

  using SLayoutXAtom = decltype(slayout_selector<kSwizzleX, Tin>());
}
