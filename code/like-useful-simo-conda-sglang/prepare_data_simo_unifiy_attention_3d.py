from safetensors.torch import save_file
####### prefille
log_line = ""

data_dict={}
if isinstance(softmax_segm_output, torch.Tensor):
  data_dict['softmax_segm_output'] = softmax_segm_output.contiguous()
else:
  log_line += f"softmax_segm_output={softmax_segm_output}\n"
if isinstance(softmax_segm_max, torch.Tensor):
  data_dict['softmax_segm_max'] = softmax_segm_max.contiguous()
else:
  log_line += f"softmax_segm_max={softmax_segm_max}\n"
if isinstance(softmax_segm_expsum, torch.Tensor):
  data_dict['softmax_segm_expsum'] = softmax_segm_expsum.contiguous()
else:
  log_line += f"softmax_segm_expsum={softmax_segm_expsum}\n"
if isinstance(q, torch.Tensor):
  data_dict['q'] = q.contiguous()
else:
  log_line += f"q={q}\n"
if isinstance(k, torch.Tensor):
  data_dict['k'] = k.contiguous()
else:
  log_line += f"k={k}\n"
if isinstance(v, torch.Tensor):
  data_dict['v'] = v.contiguous()
else:
  log_line += f"v={v}\n"
if isinstance(sinks, torch.Tensor):
  data_dict['sinks'] = sinks.contiguous()
else:
  log_line += f"sinks={sinks}\n"
if isinstance(block_table, torch.Tensor):
  data_dict['block_table'] = block_table.contiguous()
else:
  log_line += f"block_table={block_table}\n"
if isinstance(seqused_k, torch.Tensor):
  data_dict['seqused_k'] = seqused_k.contiguous()
else:
  log_line += f"seqused_k={seqused_k}\n"
if isinstance(alibi_slopes, torch.Tensor):
  data_dict['alibi_slopes'] = alibi_slopes.contiguous()
else:
  log_line += f"alibi_slopes={alibi_slopes}\n"
if isinstance(qq_bias, torch.Tensor):
  data_dict['qq_bias'] = qq_bias.contiguous()
else:
  log_line += f"qq_bias={qq_bias}\n"
if isinstance(softmax_scale, torch.Tensor):
  data_dict['softmax_scale'] = softmax_scale.contiguous()
else:
  log_line += f"softmax_scale={softmax_scale}\n"
if isinstance(k_descale, torch.Tensor):
  data_dict['k_descale'] = k_descale.contiguous()
else:
  log_line += f"k_descale={k_descale}\n"
if isinstance(v_descale, torch.Tensor):
  data_dict['v_descale'] = v_descale.contiguous()
else:
  log_line += f"v_descale={v_descale}\n"
if isinstance(softcap, torch.Tensor):
  data_dict['softcap'] = softcap.contiguous()
else:
  log_line += f"softcap={softcap}\n"
if isinstance(num_query_heads, torch.Tensor):
  data_dict['num_query_heads'] = num_query_heads.contiguous()
else:
  log_line += f"num_query_heads={num_query_heads}\n"
if isinstance(num_queries_per_kv, torch.Tensor):
  data_dict['num_queries_per_kv'] = num_queries_per_kv.contiguous()
else:
  log_line += f"num_queries_per_kv={num_queries_per_kv}\n"
if isinstance(block_size, torch.Tensor):
  data_dict['block_size'] = block_size.contiguous()
else:
  log_line += f"block_size={block_size}\n"
if isinstance(TILE_SIZE_DECODE, torch.Tensor):
  data_dict['TILE_SIZE_DECODE'] = TILE_SIZE_DECODE.contiguous()
else:
  log_line += f"TILE_SIZE_DECODE={TILE_SIZE_DECODE}\n"
if isinstance(head_size, torch.Tensor):
  data_dict['head_size'] = head_size.contiguous()
else:
  log_line += f"head_size={head_size}\n"
if isinstance(use_alibi_slopes, torch.Tensor):
  data_dict['use_alibi_slopes'] = use_alibi_slopes.contiguous()
else:
  log_line += f"use_alibi_slopes={use_alibi_slopes}\n"
if isinstance(use_alibi_sqrt, torch.Tensor):
  data_dict['use_alibi_sqrt'] = use_alibi_sqrt.contiguous()
else:
  log_line += f"use_alibi_sqrt={use_alibi_sqrt}\n"
if isinstance(use_qq_bias, torch.Tensor):
  data_dict['use_qq_bias'] = use_qq_bias.contiguous()
else:
  log_line += f"use_qq_bias={use_qq_bias}\n"
if isinstance(use_mm_prefix, torch.Tensor):
  data_dict['use_mm_prefix'] = use_mm_prefix.contiguous()
else:
  log_line += f"use_mm_prefix={use_mm_prefix}\n"
if isinstance(max_mm_ranges, torch.Tensor):
  data_dict['max_mm_ranges'] = max_mm_ranges.contiguous()
else:
  log_line += f"max_mm_ranges={max_mm_ranges}\n"
if isinstance(mm_prefix_range, torch.Tensor):
  data_dict['mm_prefix_range'] = mm_prefix_range.contiguous()
else:
  log_line += f"mm_prefix_range={mm_prefix_range}\n"
if isinstance(cu_seqlens_q, torch.Tensor):
  data_dict['cu_seqlens_q'] = cu_seqlens_q.contiguous()
else:
  log_line += f"cu_seqlens_q={cu_seqlens_q}\n"
if isinstance(BLOCK_Q, torch.Tensor):
  data_dict['BLOCK_Q'] = BLOCK_Q.contiguous()
else:
  log_line += f"BLOCK_Q={BLOCK_Q}\n"
if isinstance(num_seqs, torch.Tensor):
  data_dict['num_seqs'] = num_seqs.contiguous()
else:
  log_line += f"num_seqs={num_seqs}\n"
if isinstance(BLOCK_M, torch.Tensor):
  data_dict['BLOCK_M'] = BLOCK_M.contiguous()
else:
  log_line += f"BLOCK_M={BLOCK_M}\n"
if isinstance(num_par_softmax_segments, torch.Tensor):
  data_dict['num_par_softmax_segments'] = num_par_softmax_segments.contiguous()
else:
  log_line += f"num_par_softmax_segments={num_par_softmax_segments}\n"
if isinstance(MX_FORMAT_ID, torch.Tensor):
  data_dict['MX_FORMAT_ID'] = MX_FORMAT_ID.contiguous()
else:
  log_line += f"MX_FORMAT_ID={MX_FORMAT_ID}\n"
if isinstance(MXFP4_Q, torch.Tensor):
  data_dict['MXFP4_Q'] = MXFP4_Q.contiguous()
else:
  log_line += f"MXFP4_Q={MXFP4_Q}\n"
if isinstance(PG_FORMAT_ID, torch.Tensor):
  data_dict['PG_FORMAT_ID'] = PG_FORMAT_ID.contiguous()
else:
  log_line += f"PG_FORMAT_ID={PG_FORMAT_ID}\n"
if isinstance(QUANT_BLOCK_SIZE, torch.Tensor):
  data_dict['QUANT_BLOCK_SIZE'] = QUANT_BLOCK_SIZE.contiguous()
else:
  log_line += f"QUANT_BLOCK_SIZE={QUANT_BLOCK_SIZE}\n"
if isinstance(PACKED_HEAD_SIZE, torch.Tensor):
  data_dict['PACKED_HEAD_SIZE'] = PACKED_HEAD_SIZE.contiguous()
else:
  log_line += f"PACKED_HEAD_SIZE={PACKED_HEAD_SIZE}\n"
if isinstance(SCALE_HEAD_SIZE, torch.Tensor):
  data_dict['SCALE_HEAD_SIZE'] = SCALE_HEAD_SIZE.contiguous()
else:
  log_line += f"SCALE_HEAD_SIZE={SCALE_HEAD_SIZE}\n"
if isinstance(PACKED_HEAD_SIZE_PADDED, torch.Tensor):
  data_dict['PACKED_HEAD_SIZE_PADDED'] = PACKED_HEAD_SIZE_PADDED.contiguous()
else:
  log_line += f"PACKED_HEAD_SIZE_PADDED={PACKED_HEAD_SIZE_PADDED}\n"
if isinstance(SCALE_HEAD_SIZE_PADDED, torch.Tensor):
  data_dict['SCALE_HEAD_SIZE_PADDED'] = SCALE_HEAD_SIZE_PADDED.contiguous()
else:
  log_line += f"SCALE_HEAD_SIZE_PADDED={SCALE_HEAD_SIZE_PADDED}\n"
if isinstance(SCALE_PLANE_OFFSET, torch.Tensor):
  data_dict['SCALE_PLANE_OFFSET'] = SCALE_PLANE_OFFSET.contiguous()
else:
  log_line += f"SCALE_PLANE_OFFSET={SCALE_PLANE_OFFSET}\n"
    
save_file(data_dict, "temp/simo_vllm_unify_attention_3d.safetensors")
with open("temp/log_line_simo_vllm_unify_attention_3d.txt", 'w') as ofp:
  ofp.write(log_line)
