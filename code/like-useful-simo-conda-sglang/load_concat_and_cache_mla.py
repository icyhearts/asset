import json
import torch
import triton
from safetensors.torch import load_file
from simo.extensions.vllm_simo.v1.attention.ops.triton_concat_and_cache_mla import concat_and_cache_mla

save_dir = "temp/debug_kv_cache__concat_and_cache_mla/"
time_prefix="1779076544.2267976" # mxfp8
time_prefix="1779159409.4112957" # mxfp8
time_prefix="1779159753.7633486" # fp8 per group, group size=64

print(f"time_prefix:{time_prefix}")
safe_tensor_path = f"{save_dir}/concat_and_cache_mla.{time_prefix}.safetensors"
data_dict = load_file(safe_tensor_path, device="cuda:0")

args_json_path = f"{save_dir}/kv_cache_quant_spec.{time_prefix}.json"
from simo.extensions.vllm_simo.quantization.quantization_config import parse_quantize_spec
with open(args_json_path) as fp:
  kv_cache_quant_spec_dict = json.load( fp)
  kv_cache_quant_spec = parse_quantize_spec(kv_cache_quant_spec_dict)

kv_c=data_dict["kv_c"]
k_pe=data_dict["k_pe"]
kv_cache=data_dict["kv_cache"]
slot_mapping=data_dict["slot_mapping"]
concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_quant_spec)
print("end")
