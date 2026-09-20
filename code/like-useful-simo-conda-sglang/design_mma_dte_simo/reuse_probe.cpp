// PROBE: call torch's already-compiled MX mma_dte directly, host-side only.
#include <c10/macros/Macros.h>
#include <cstdio>
#include <c10/util/Exception.h>
#include <c10/macros/Export.h>
#include <c10/sipu/SIPUStream.h>
#include <ATen/ATen.h>
#include <torch_sipu/csrc/contrib/native/sipu/mma_dte/BlasMmaDte.suh>

int main() {
  using at::native::MxMmaDteLayout;
  using at::native::MxMmaDteConfig;
  using at::native::mx_mma_dte_config_spec;
  using at::native::mx_mma_dte_output_layout_spec;

  // Can we see the enum + specs from the shipped header alone?
  auto cfg = MxMmaDteConfig{MxMmaDteLayout::kS1x4M32, at::MXDType::MXInt8};
  const auto spec = mx_mma_dte_config_spec(cfg);
  const auto ospec16 = mx_mma_dte_output_layout_spec(cfg, at::kBFloat16);
  const auto ospec32 = mx_mma_dte_output_layout_spec(cfg, at::kFloat);
  printf("MXINT8 S1x4M32: tile_k=%d tile_row=%d rhs_layout=%d\n",
         spec.dte_operand.tile_k_elem, spec.dte_operand.tile_row_elem,
         (int)spec.required_rhs_layout);
  printf("out bf16: tile_n=%d tile_m=%d\n", ospec16.tile_n_elem, ospec16.tile_m_elem);
  printf("out fp32: tile_n=%d tile_m=%d  (nonzero => fp32 supported)\n",
         ospec32.tile_n_elem, ospec32.tile_m_elem);
  printf("USEMMA  = %s\n", spec.dte_operand.tile_row_elem ? "ok" : "bad");
  return 0;
}
