from safetensors.torch import save_file
####### prefille
log_line = ""

data_dict={}

if isinstance(key, torch.Tensor):
  data_dict['key'] = key.contiguous()
else:
  log_line += f"key={key}\n"
if isinstance(value, torch.Tensor):
  data_dict['value'] = value.contiguous()
else:
  log_line += f"value={value}\n"
if isinstance(key_cache, torch.Tensor):
  data_dict['key_cache'] = key_cache.contiguous()
else:
  log_line += f"key_cache={key_cache}\n"
if isinstance(value_cache, torch.Tensor):
  data_dict['value_cache'] = value_cache.contiguous()
else:
  log_line += f"value_cache={value_cache}\n"
if isinstance(slot_mapping, torch.Tensor):
  data_dict['slot_mapping'] = slot_mapping.contiguous()
else:
  log_line += f"slot_mapping={slot_mapping}\n"
if isinstance(key_token_stride, torch.Tensor):
  data_dict['key_token_stride'] = key_token_stride.contiguous()
else:
  log_line += f"key_token_stride={key_token_stride}\n"
if isinstance(value_token_stride, torch.Tensor):
  data_dict['value_token_stride'] = value_token_stride.contiguous()
else:
  log_line += f"value_token_stride={value_token_stride}\n"
if isinstance(key_head_stride, torch.Tensor):
  data_dict['key_head_stride'] = key_head_stride.contiguous()
else:
  log_line += f"key_head_stride={key_head_stride}\n"
if isinstance(value_head_stride, torch.Tensor):
  data_dict['value_head_stride'] = value_head_stride.contiguous()
else:
  log_line += f"value_head_stride={value_head_stride}\n"
if isinstance(block_stride, torch.Tensor):
  data_dict['block_stride'] = block_stride.contiguous()
else:
  log_line += f"block_stride={block_stride}\n"
if isinstance(page_stride, torch.Tensor):
  data_dict['page_stride'] = page_stride.contiguous()
else:
  log_line += f"page_stride={page_stride}\n"
if isinstance(num_heads, torch.Tensor):
  data_dict['num_heads'] = num_heads.contiguous()
else:
  log_line += f"num_heads={num_heads}\n"
if isinstance(head_size, torch.Tensor):
  data_dict['head_size'] = head_size.contiguous()
else:
  log_line += f"head_size={head_size}\n"
if isinstance(block_size, torch.Tensor):
  data_dict['block_size'] = block_size.contiguous()
else:
  log_line += f"block_size={block_size}\n"
if isinstance(packed_head_size, torch.Tensor):
  data_dict['packed_head_size'] = packed_head_size.contiguous()
else:
  log_line += f"packed_head_size={packed_head_size}\n"
if isinstance(scale_head_size, torch.Tensor):
  data_dict['scale_head_size'] = scale_head_size.contiguous()
else:
  log_line += f"scale_head_size={scale_head_size}\n"
if isinstance(PACKED_HEAD_SIZE_PADDED, torch.Tensor):
  data_dict['PACKED_HEAD_SIZE_PADDED'] = PACKED_HEAD_SIZE_PADDED.contiguous()
else:
  log_line += f"PACKED_HEAD_SIZE_PADDED={PACKED_HEAD_SIZE_PADDED}\n"
if isinstance(SCALE_HEAD_SIZE_PADDED, torch.Tensor):
  data_dict['SCALE_HEAD_SIZE_PADDED'] = SCALE_HEAD_SIZE_PADDED.contiguous()
else:
  log_line += f"SCALE_HEAD_SIZE_PADDED={SCALE_HEAD_SIZE_PADDED}\n"
if isinstance(k_hadamard_transform_size, torch.Tensor):
  data_dict['k_hadamard_transform_size'] = k_hadamard_transform_size.contiguous()
else:
  log_line += f"k_hadamard_transform_size={k_hadamard_transform_size}\n"
if isinstance(v_hadamard_transform_size, torch.Tensor):
  data_dict['v_hadamard_transform_size'] = v_hadamard_transform_size.contiguous()
else:
  log_line += f"v_hadamard_transform_size={v_hadamard_transform_size}\n"
if isinstance(mx_format_id, torch.Tensor):
  data_dict['mx_format_id'] = mx_format_id.contiguous()
else:
  log_line += f"mx_format_id={mx_format_id}\n"
if isinstance(observer_mode, torch.Tensor):
  data_dict['observer_mode'] = observer_mode.contiguous()
else:
  log_line += f"observer_mode={observer_mode}\n"
if isinstance(scale_rounding_mode, torch.Tensor):
  data_dict['scale_rounding_mode'] = scale_rounding_mode.contiguous()
else:
  log_line += f"scale_rounding_mode={scale_rounding_mode}\n"
if isinstance(pg_format_id, torch.Tensor):
  data_dict['pg_format_id'] = pg_format_id.contiguous()
else:
  log_line += f"pg_format_id={pg_format_id}\n"
if isinstance(pg_min_value, torch.Tensor):
  data_dict['pg_min_value'] = pg_min_value.contiguous()
else:
  log_line += f"pg_min_value={pg_min_value}\n"
if isinstance(pg_max_value, torch.Tensor):
  data_dict['pg_max_value'] = pg_max_value.contiguous()
else:
  log_line += f"pg_max_value={pg_max_value}\n"
if isinstance(tile_size, torch.Tensor):
  data_dict['tile_size'] = tile_size.contiguous()
else:
  log_line += f"tile_size={tile_size}\n"

save_file(data_dict, "temp/simo_vllm_downcast_mx_fp8_kv_cache.safetensors")
