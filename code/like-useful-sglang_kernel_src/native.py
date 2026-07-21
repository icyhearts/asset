import torch
from typing import Optional, Tuple, Union
from safetensors.torch import load_file

from sglang.srt.batch_invariant_ops.batch_invariant_ops import mean_batch_invariant
from sgl_kernel import fused_add_rmsnorm
# def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None = None):
layer2weight_dict = load_file('/share/users/like/package/simo_conda_sglang/temp/layer2weight.safetensors', device="cuda:0")

sgl_directory = "/data/like/temp/sgl_safe_tensor_batch_invariant_triton"
input_safe_tensor_path = sgl_directory + '/' + 'rank-0-prefix-model.layers.2-forwardcount-0-0.safetensors'
data_dict_key_input = 'in_decoder_0_input'
data_dict_key_residual = 'in_decoder_0_residual'

input_dict = load_file(input_safe_tensor_path, device='cuda:0')




self = torch.nn.Linear(2,3)
self.variance_size_override = None
self.weight.data = layer2weight_dict['layer2weight']
self.weight.requires_grad=False
self.cast_x_before_out_mul = False
self.variance_epsilon = 1e-6
self.fp32_residual = None

def forward_native(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    mean_batch_invariant_enable = False,
    post_residual_addition: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if not x.is_contiguous():
        x = x.contiguous()
    orig_dtype =  x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        if post_residual_addition is not None:
            x = x + post_residual_addition.to(torch.float32)
        if self.fp32_residual:
            residual = x.clone()
        else:
            residual = x.to(orig_dtype)

    hidden_size = x.shape[-1]
#        if hidden_size != self.hidden_size:
#            raise ValueError(
#                "Expected hidden_size to be "
#                f"{self.hidden_size}, but found: {hidden_size}"
#            )

    if self.variance_size_override is None:
        x_var = x
    else:
        if hidden_size < self.variance_size_override:
            raise ValueError(
                "Expected hidden_size to be at least "
                f"{self.variance_size_override}, but found: {hidden_size}"
            )

        x_var = x[..., : self.variance_size_override]

    print(f" mean_batch_invariant_enable:{mean_batch_invariant_enable}")
    if mean_batch_invariant_enable:
        variance =  mean_batch_invariant(x_var.pow(2), dim=[-1], keepdim=True)
    else:
        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + self.variance_epsilon)

    if self.cast_x_before_out_mul:
        x = self.weight * x.to(orig_dtype)
    else:
        x = (x * self.weight).to(orig_dtype)

    if residual is None:
        return x
    else:
        return x, residual
def pad_to_need_row(small_tensor, need_row):
    nrow = small_tensor.shape[0]
    new_row = need_row - nrow
    pad_tensor = torch.randn([new_row, small_tensor.shape[1]], device=small_tensor.device,
                             dtype=small_tensor.dtype)
    large_tensor = torch.cat([small_tensor, pad_tensor], dim=0)
    return large_tensor


input_b_i = input_dict[data_dict_key_input].clone()
residual_b_i = input_dict[data_dict_key_residual].clone()
need_row = 256
input_b_i_large = pad_to_need_row(input_b_i , need_row)
residual_b_i_large = pad_to_need_row(residual_b_i , need_row)

out_x, out_residual = forward_native(self,
               input_b_i,
               residual=residual_b_i,
               mean_batch_invariant_enable=True)

out_x_large, out_residual_large = forward_native(self,
               input_b_i_large,
               residual=residual_b_i_large,
               mean_batch_invariant_enable=True)
nrow = input_b_i.shape[0]

out_x_large_slice = out_x_large[:nrow]
out_residual_large_slice = out_residual_large[:nrow]
print(f"torch.equal(out_x, out_x_large_slice):{torch.equal(out_x, out_x_large_slice)}")
print(f"torch.equal(out_residual, out_residual_large_slice):{torch.equal(out_residual, out_residual_large_slice)}")



#input_b_i = input_dict[data_dict_key_input].clone()
#residual_b_i = input_dict[data_dict_key_residual].clone()
#out_x_torch_mean, out_residual_false_torch_mean = forward_native(self,
#               input_b_i,
#               residual=residual_b_i,
#               mean_batch_invariant_enable=False)



input_fast = input_dict[data_dict_key_input].clone()
residual_fast = input_dict[data_dict_key_residual].clone()
fused_add_rmsnorm(input_fast, residual_fast, self.weight, self.variance_epsilon)

input_fast_large = pad_to_need_row(input_fast , need_row)
residual_fast_large = pad_to_need_row(residual_fast , need_row)
fused_add_rmsnorm(input_fast_large, residual_fast_large, self.weight, self.variance_epsilon)

input_fast_large_slice = input_fast_large[:nrow]
residual_fast_large_slice = residual_fast_large[:nrow]

print(f"torch.equal(input_fast, input_fast_large_slice):{torch.equal(input_fast, input_fast_large_slice)}")
print(f"torch.equal(residual_fast, residual_fast_large_slice):{torch.equal(residual_fast, residual_fast_large_slice)}")
