from safetensors.torch import load_file
import triton

input_file_path = "temp/simo_vllm_unify_attention_3d.safetensors"
data_dict = load_file(input_file_path, device="cuda:0")
k_descale=data_dict['k_descale']
softmax_segm_expsum=data_dict['softmax_segm_expsum']
softmax_segm_max=data_dict['softmax_segm_max']
softmax_segm_output=data_dict['softmax_segm_output']
v_descale=data_dict['v_descale']
block_table=data_dict['block_table']
cu_seqlens_q=data_dict['cu_seqlens_q']
seqused_k=data_dict['seqused_k']
q=data_dict['q']
k=data_dict['k']
v=data_dict['v']


###
(total_num_q_blocks, num_kv_heads, num_par_softmax_segments) = (1, 8, 16) 
use_qq_bias=False
window_size = (-1, -1)
sinks=None
alibi_slopes=None
qq_bias=None
softmax_scale=0.08838834764831845
softcap=0
num_query_heads=32
num_queries_per_kv=4
block_size=16
TILE_SIZE_DECODE=32
head_size=128
use_alibi_slopes=False
use_alibi_sqrt=False
use_qq_bias=False
use_mm_prefix=False
max_mm_ranges=0
mm_prefix_range=None
BLOCK_Q=4
num_seqs=1
BLOCK_M=16
num_par_softmax_segments=16
MX_FORMAT_ID=3
MXFP4_Q=False
PG_FORMAT_ID=0
QUANT_BLOCK_SIZE=32
PACKED_HEAD_SIZE=128
SCALE_HEAD_SIZE=4
PACKED_HEAD_SIZE_PADDED=128
SCALE_HEAD_SIZE_PADDED=4
SCALE_PLANE_OFFSET=1024


###
from simo.extensions.vllm_simo.v1.attention.ops.triton_unified_attention import kernel_unified_attention_3d
kernel_unified_attention_3d[(total_num_q_blocks, num_kv_heads, num_par_softmax_segments)](
      segm_output_ptr=softmax_segm_output,
      segm_max_ptr=softmax_segm_max,
      segm_expsum_ptr=softmax_segm_expsum,
      query_ptr=q,
      key_cache_ptr=k,
      value_cache_ptr=v,
      sink_ptr=sinks,
      block_tables_ptr=block_table,
      seq_lens_ptr=seqused_k,
      alibi_slopes_ptr=alibi_slopes,
      qq_bias_ptr=qq_bias,
      scale=softmax_scale,
      k_scale=k_descale,
      v_scale=v_descale,
      softcap=softcap,
      num_query_heads=num_query_heads,
      num_queries_per_kv=num_queries_per_kv,
      block_table_stride=block_table.stride(0),
      query_stride_0=q.stride(0),
      query_stride_1=q.stride(1),
      qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
      BLOCK_SIZE=block_size,
      TILE_SIZE=TILE_SIZE_DECODE,
      HEAD_SIZE=head_size,
      HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
      USE_ALIBI_SLOPES=use_alibi_slopes,
      USE_ALIBI_SQRT=use_alibi_sqrt,
      USE_QQ_BIAS=use_qq_bias,
      USE_SOFTCAP=(softcap > 0),
      USE_SINKS=(sinks is not None),
      SLIDING_WINDOW=(1 + window_size[0]),
      USE_MM_PREFIX=use_mm_prefix,
      MAX_MM_RANGES=max_mm_ranges,
      mm_prefix_range_ptr=mm_prefix_range,
      stride_k_cache_0=k.stride(0),
      stride_k_cache_1=k.stride(1),
      stride_k_cache_2=k.stride(2),
      stride_k_cache_3=k.stride(3),
      stride_v_cache_0=v.stride(0),
      stride_v_cache_1=v.stride(1),
      stride_v_cache_2=v.stride(2),
      stride_v_cache_3=v.stride(3),
      query_start_len_ptr=cu_seqlens_q,
      BLOCK_Q=BLOCK_Q,
      num_seqs=num_seqs,
      BLOCK_M=BLOCK_M,
      NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
      MX_FORMAT_ID=MX_FORMAT_ID,
      MXFP4_Q=MXFP4_Q,
      PG_FORMAT_ID=PG_FORMAT_ID,
      QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
      PACKED_HEAD_SIZE=PACKED_HEAD_SIZE,
      SCALE_HEAD_SIZE=SCALE_HEAD_SIZE,
      PACKED_HEAD_SIZE_PADDED=PACKED_HEAD_SIZE_PADDED,
      SCALE_HEAD_SIZE_PADDED=SCALE_HEAD_SIZE_PADDED,
      SCALE_PLANE_OFFSET=SCALE_PLANE_OFFSET,
    )
