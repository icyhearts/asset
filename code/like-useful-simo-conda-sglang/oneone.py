
from safetensors.torch import load_file
sgl_directory = "/data/like/temp/sgl_safe_tensor_batch_invariant_triton/"
ref_dir = "/data/like/temp/sgl_safe_tensor_batch_invariant/"

row_parallel_quant_method_out__safetensor = 'rank-0-prefix-model.layers.0.self_attn.o_proj-forwardcount-0-1.safetensors'
row_parallel_quant_method_out__safetensor_sgl = load_file(sgl_directory + row_parallel_quant_method_out__safetensor, device="cuda:0")['row_parallel_quant_method_out']
row_parallel_quant_method_out__safetensor_ref = load_file(ref_dir + row_parallel_quant_method_out__safetensor, device="cuda:0")['row_parallel_quant_method_out']
