
from safetensors.torch import load_file
sgl_directory = "/data/like/temp/sgl_safe_tensor_batch_invariant_triton/"
ref_dir = "/data/like/temp/sgl_safe_tensor_batch_invariant/"

row_parallel_quant_method_out__safetensor = 'rank-0-prefix-model.layers.0.self_attn.o_proj-forwardcount-0-1.safetensors'
row_parallel_quant_method_out__safetensor_sgl = load_file(sgl_directory + row_parallel_quant_method_out__safetensor, device="cuda:0")['row_parallel_quant_method_out']
row_parallel_quant_method_out__safetensor_ref = load_file(ref_dir + row_parallel_quant_method_out__safetensor, device="cuda:0")['row_parallel_quant_method_out']


act_out_safe_tensor_name = "rank-0-prefix-model.layers.0.mlp-forwardcount-0-0.safetensors"

act_out_tensor = load_file(sgl_directory + act_out_safe_tensor_name, device="cuda:0")['in_v2mlp_2_act_fn_out']

W_dict = load_file("/data/like/hf-models/DeepSeek-V2-Lite-Chat-16B_A2.4B/model-00001-of-000004.safetensors", device="cpu")
for k in W_dict.keys():
  if "layers.0.mlp.down_proj" in k:
    print(f"found k:{k}")
    layers_0_mlp_down_proj_w = W_dict.get(k)

