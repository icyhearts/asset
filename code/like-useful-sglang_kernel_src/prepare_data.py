
from safetensors.torch import save_file
####### prefille
data_dict = { "q_extend": q_extend.contiguous(), "k_extend": k_extend, "v_extend": v_extend, "o_extend": o_extend, "k_buffer": k_buffer, "v_buffer": v_buffer, "qo_indptr": qo_indptr, "kv_indptr": kv_indptr, "kv_indices": kv_indices, }
save_file(data_dict, "temp/extend_forward_triton_input.safetensors")

# decode stage1
data_dict = { "q":q, "k_buffer":k_buffer, "v_buffer":v_buffer, "kv_indptr":kv_indptr, "kv_indices":kv_indices, "att_out":att_out, "att_lse":att_lse, "num_kv_splits":num_kv_splits, }
save_file(data_dict, "temp/decode_forward_stage1_triton_input.safetensors")

# decode stage2
data_dict = { "logits":logits, "lse":lse, "o":o, "kv_indptr":kv_indptr, "num_kv_splits":num_kv_splits, }
save_file(data_dict, "temp/decode_forward_stage2_triton_input.safetensors")
