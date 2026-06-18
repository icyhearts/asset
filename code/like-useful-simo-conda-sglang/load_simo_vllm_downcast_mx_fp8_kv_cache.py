from safetensors.torch import load_file
import triton

input_file_path = "temp/simo_vllm_downcast_mx_fp8_kv_cache.safetensors"
data_dict = load_file(input_file_path, device="cuda:0")
###
key=data_dict['key']
value=data_dict['value']
key_cache=data_dict['key_cache']
value_cache=data_dict['value_cache']
slot_mapping=data_dict['slot_mapping']


num_tokens, num_heads, head_size = key.shape

key_token_stride=1024
value_token_stride=1024
key_head_stride=128
value_head_stride=128
block_stride=33792
page_stride=1056
num_heads=8
head_size=128
block_size=16
packed_head_size=128
scale_head_size=4
PACKED_HEAD_SIZE_PADDED=128
SCALE_HEAD_SIZE_PADDED=4
k_hadamard_transform_size=0
v_hadamard_transform_size=0
mx_format_id=3
observer_mode=1
scale_rounding_mode=1
pg_format_id=0
pg_min_value=0.0
pg_max_value=0.0
tile_size=32
NUM_TILES = triton.cdiv(head_size, tile_size)



kernel_args = dict(
    key_ptr=key,
    value_ptr=value,
    key_cache_ptr=key_cache,
    value_cache_ptr=value_cache,
    slot_mapping_ptr=slot_mapping,
    key_token_stride=key_token_stride,
    value_token_stride=value_token_stride,
    key_head_stride=key_head_stride,
    value_head_stride=value_head_stride,
    block_stride=block_stride,
    page_stride=page_stride,
    num_heads=num_heads,
    head_size=head_size,
    block_size=block_size,
    PACKED_HEAD_SIZE=packed_head_size,
    SCALE_HEAD_SIZE=scale_head_size,
    PACKED_HEAD_SIZE_PADDED=PACKED_HEAD_SIZE_PADDED,
    SCALE_HEAD_SIZE_PADDED=SCALE_HEAD_SIZE_PADDED,
    K_HADAMARD_TRANSFORM_SIZE=k_hadamard_transform_size,
    V_HADAMARD_TRANSFORM_SIZE=v_hadamard_transform_size,
    MX_FORMAT_ID=mx_format_id,
    OBSERVER_MODE=observer_mode,
    SCALE_ROUNDING_MODE=scale_rounding_mode,
    PG_FORMAT_ID=pg_format_id,
    PG_MIN_VALUE=pg_min_value,
    PG_MAX_VALUE=pg_max_value,
    TILE_SIZE=tile_size,
  )

from simo.extensions.vllm_simo.v1.attention.ops.triton_reshape_and_cache_flash import reshape_and_cache_kernel_flash
grid = (num_tokens, num_heads, NUM_TILES)
reshape_and_cache_kernel_flash[grid](**kernel_args)
