import json

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from simo.extensions.sglang_simo.mem_cache.memory_pool import SIMOMLATokenToKVPool
from simo.extensions.sglang_simo.quantization.quantization import (
    get_downcast_kernel,
    get_upcast_kernel,
    parse_quantize_spec,
)
from simo.quantization.config import (
    QuantizeSpecFP,
    QuantizeSpecInt,
    QuantizeSpecMX,
    QuantizeSpecType,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

def cosine_similarity(a, b, dim=0):
    return F.cosine_similarity(a.float(), b.float(), dim=dim)

def relative_l2_error(x: torch.Tensor, x_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # 转为 float32 以保证计算精度
    x_f32 = x.float()
    x_hat_f32 = x_hat.float()

    # torch.norm 会计算整个张量的 L2 范数
    diff_norm = torch.norm(x_f32 - x_hat_f32)
    orig_norm = torch.norm(x_f32)

    return diff_norm / (orig_norm + eps)


def _flatten_mla_part(x: torch.Tensor, logical_dim: int) -> torch.Tensor:
  return x.reshape(x.shape[0], -1)[:, :logical_dim]


def native_sglang_fp8_cast_kv_cache(
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    num_cache_rows: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
  # This mirrors the non-DSA sglang native fp8 KV path:
  # cache_k_nope/cache_k_rope are cast to torch.float8_e4m3fn with scale=1.0,
  # then read back as bf16 for error comparison.
  if not hasattr(torch, "float8_e4m3fn"):
    raise RuntimeError("Current torch does not expose torch.float8_e4m3fn")

  result = torch.zeros(
      (num_cache_rows, 1, kv_lora_rank + qk_rope_head_dim),
      dtype=torch.bfloat16,
      device=cache_k_nope.device,
  )
  valid_token_mask = loc >= 0
  if not valid_token_mask.any():
    return result

  valid_loc = loc[valid_token_mask].long()
  cache_k_nope_2d = _flatten_mla_part(cache_k_nope, kv_lora_rank)
  cache_k_rope_2d = _flatten_mla_part(cache_k_rope, qk_rope_head_dim)
  result[valid_loc, 0, :kv_lora_rank] = (
      cache_k_nope_2d[valid_token_mask]
      .to(torch.float8_e4m3fn)
      .to(torch.bfloat16)
  )
  result[valid_loc, 0, kv_lora_rank:] = (
      cache_k_rope_2d[valid_token_mask]
      .to(torch.float8_e4m3fn)
      .to(torch.bfloat16)
  )
  return result

def _get_quant_tile_size(kv_cache_quant_spec: QuantizeSpecType) -> int:
  if isinstance(kv_cache_quant_spec, QuantizeSpecMX):
    return kv_cache_quant_spec.block_size
  if isinstance(kv_cache_quant_spec, (QuantizeSpecFP, QuantizeSpecInt)):
    return kv_cache_quant_spec.group_size
  raise ValueError(f"Unsupported kv_cache_quant_spec type: {type(kv_cache_quant_spec)}")


def _view_u8_as(row_bytes: torch.Tensor, dtype: torch.dtype, shape: torch.Size) -> torch.Tensor:
  return row_bytes.contiguous().view(dtype).reshape(tuple(shape))


def _dequant_region(
    row_bytes: torch.Tensor,
    packed_start: int,
    scale_start: int,
    logical_dim: int,
    downcast_kernel,
    upcast_kernel,
) -> torch.Tensor:
  meta_src = torch.empty(row_bytes.shape[0], logical_dim, device="meta")
  x_q_meta, scale_meta = downcast_kernel(meta_src)
  packed_bytes = x_q_meta.contiguous().view(torch.uint8).shape[-1]
  scale_bytes = scale_meta.contiguous().view(torch.uint8).shape[-1]

  packed = _view_u8_as(
      row_bytes[:, packed_start : packed_start + packed_bytes],
      x_q_meta.dtype,
      x_q_meta.shape,
  )
  scale = _view_u8_as(
      row_bytes[:, scale_start : scale_start + scale_bytes],
      scale_meta.dtype,
      scale_meta.shape,
  )
  return upcast_kernel(packed, scale, torch.bfloat16)


def dequant_simo_kv_cache(
    simo_one_cache: torch.Tensor,
    loc: torch.Tensor,
    kv_cache_quant_spec: QuantizeSpecType,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
  # 把 kv_cache_quant_spec 量化的 kv cache 中 loc 所在的行反量化成 bf16。
  # 支持 mxfp8/mxfp4/mxfp6/mxint8, fp8 per group, int8 per group, nvfp4。
  result = torch.zeros(
      (simo_one_cache.shape[0], 1, kv_lora_rank + qk_rope_head_dim),
      dtype=torch.bfloat16,
      device=simo_one_cache.device,
  )
  valid_loc = loc[loc >= 0].long()
  if valid_loc.numel() == 0:
    return result

  tile_size = _get_quant_tile_size(kv_cache_quant_spec)
  assert kv_lora_rank % tile_size == 0, (
      f"kv_lora_rank ({kv_lora_rank}) must be a multiple of tile_size ({tile_size})"
  )
  assert qk_rope_head_dim % tile_size == 0, (
      f"qk_rope_head_dim ({qk_rope_head_dim}) must be a multiple of tile_size ({tile_size})"
  )

  downcast_kernel = get_downcast_kernel(kv_cache_quant_spec, 0)
  upcast_kernel = get_upcast_kernel(kv_cache_quant_spec)
  x_q_tile_meta, scale_tile_meta = downcast_kernel(
      torch.empty(1, tile_size, device="meta")
  )
  packed_tile_bytes = x_q_tile_meta.contiguous().view(torch.uint8).shape[-1]
  scale_tile_bytes = scale_tile_meta.contiguous().view(torch.uint8).shape[-1]

  kv_c_num_tiles = kv_lora_rank // tile_size
  k_pe_num_tiles = qk_rope_head_dim // tile_size
  kv_c_packed_bytes = kv_c_num_tiles * packed_tile_bytes
  kv_c_scale_bytes = kv_c_num_tiles * scale_tile_bytes
  kv_c_total_bytes = kv_c_packed_bytes + kv_c_scale_bytes
  k_pe_packed_bytes = k_pe_num_tiles * packed_tile_bytes

  row_bytes = simo_one_cache[valid_loc, 0, :]
  kv_c = _dequant_region(
      row_bytes,
      packed_start=0,
      scale_start=kv_c_packed_bytes,
      logical_dim=kv_lora_rank,
      downcast_kernel=downcast_kernel,
      upcast_kernel=upcast_kernel,
  )
  k_pe = _dequant_region(
      row_bytes,
      packed_start=kv_c_total_bytes,
      scale_start=kv_c_total_bytes + k_pe_packed_bytes,
      logical_dim=qk_rope_head_dim,
      downcast_kernel=downcast_kernel,
      upcast_kernel=upcast_kernel,
  )

  result[valid_loc, 0, :kv_lora_rank] = kv_c.reshape(valid_loc.numel(), kv_lora_rank)
  result[valid_loc, 0, kv_lora_rank:] = k_pe.reshape(valid_loc.numel(), qk_rope_head_dim)
  return result


enable_memory_saver = False
pool_device='cuda'
kv_lora_rank, qk_rope_head_dim = 512, 64
pool_page_size, pool_size = 1, 232060
kv_cache_quant_dir='/data/like/package/simo_conda_sglang/simo/extensions/sglang_simo/example/simo_quantization_config/kv_cache_quant/'
kv_cache_quant_files= ['quant_config_kvquant_fp8_per_group.json', 'quant_config_kvquant_int8_per_group.json', 'quant_config_kvquant_mxfp4.json', 'quant_config_kvquant_mxfp6.json', 'quant_config_kvquant_mxfp8.json', 'quant_config_kvquant_mxint8.json', 'quant_config_kvquant_nvfp4.json']

sgl_pool = MLATokenToKVPool (
        size=pool_size,
        page_size=pool_page_size,
        dtype=torch.bfloat16,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        layer_num=1,
        device=pool_device,
        enable_memory_saver=enable_memory_saver,
        start_layer=None,
        end_layer=None,
        use_dsa=False,
        override_kv_cache_dim=None,
      )

for kv_cache_quant_file in kv_cache_quant_files:
  # 1. get quant config
  with open(kv_cache_quant_dir + kv_cache_quant_file) as ifp:
    config=json.load(ifp)
  quantization_config = config.get("quantization_config", config)
  kv_cache_quant_algo = quantization_config.get("kv_cache_quant_algo", None)
  assert kv_cache_quant_algo is not None
  _simo_kv_keys = {
          "query_quantization_enabled",
          "key_hadamard_transform_size",
          "value_hadamard_transform_size",
        }
  kv_spec_dict = {k: v for k, v in kv_cache_quant_algo.items() if k not in _simo_kv_keys}
  kv_cache_quant_spec = parse_quantize_spec(kv_spec_dict)
  key_hadamard_transform_size = 0
  kv_cache_downcast_kernel = get_downcast_kernel(
        kv_cache_quant_spec, key_hadamard_transform_size
      )

  # layer 到底是什么类型不重要，只要  set_mla_kv_buffer, set_kv_buffer  需要的属性， layer 可以提供，就行
  layer = torch.nn.Linear(2,3)
  layer.kv_cache_quant_spec = kv_cache_quant_spec
  layer.kv_cache_downcast_kernel = kv_cache_downcast_kernel
  layer.layer_id = 0

  # 2. get simo pool and sglang pool
  simo_pool=SIMOMLATokenToKVPool(
      size=pool_size,
      page_size=pool_page_size,
      dtype=torch.bfloat16,
      kv_lora_rank=kv_lora_rank,
      qk_rope_head_dim=qk_rope_head_dim,
      layer_num=1,
      device=pool_device,
      enable_memory_saver=enable_memory_saver,
      start_layer= None,
      end_layer= None,
      use_dsa= False,
      override_kv_cache_dim= None,
      ### simo parameters
      kv_cache_quant_spec=kv_cache_quant_spec,
      kv_cache_downcast_kernel=kv_cache_downcast_kernel,
      )
  # 3. load safe tensor
  data_dict = load_file('templ/set_mla_kv_buffer_args.safetensors', device=pool_device)
  assert 'loc' in data_dict
  assert 'cache_k_nope' in data_dict
  assert 'cache_k_rope' in data_dict
  loc = data_dict['loc']
  cache_k_nope = data_dict['cache_k_nope']
  cache_k_rope = data_dict['cache_k_rope']

  # 4. call simo and sglang set_mla_kv_buffer
  sgl_pool.kv_buffer[layer.layer_id].zero_()
  simo_pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)
  sgl_pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

  simo_one_cache = simo_pool.kv_buffer[layer.layer_id].clone()

  dequant_simo_one_cache = dequant_simo_kv_cache(
      simo_one_cache, loc, kv_cache_quant_spec, kv_lora_rank, qk_rope_head_dim
  )
  sgl_one_cache = sgl_pool.kv_buffer[layer.layer_id].clone()

  valid_loc = loc[loc >= 0].long()
  simo_flat = dequant_simo_one_cache[valid_loc].reshape(-1)
  sgl_bf16_flat = sgl_one_cache[valid_loc].reshape(-1)
  cos_sim = cosine_similarity(simo_flat, sgl_bf16_flat)
  l2_error = relative_l2_error(sgl_bf16_flat, simo_flat)
  print(f"kv_cache_quant_file:{kv_cache_quant_file}, cos_sim:{cos_sim}, l2_error:{l2_error}")

  if kv_cache_quant_file == "quant_config_kvquant_fp8_per_group.json":
    native_fp8_cast_cache = native_sglang_fp8_cast_kv_cache(
        loc,
        cache_k_nope,
        cache_k_rope,
        sgl_one_cache.shape[0],
        kv_lora_rank,
        qk_rope_head_dim,
    )
    native_fp8_flat = native_fp8_cast_cache[valid_loc].reshape(-1)
    native_fp8_vs_bf16_cos = cosine_similarity(native_fp8_flat, sgl_bf16_flat)
    native_fp8_vs_bf16_l2 = relative_l2_error(sgl_bf16_flat, native_fp8_flat)
    simo_vs_native_fp8_cos = cosine_similarity(simo_flat, native_fp8_flat)
    simo_vs_native_fp8_l2 = relative_l2_error(native_fp8_flat, simo_flat)
    print(
        "fp8_per_group_extra_compare:"
        f" native_fp8_cast_vs_bf16_cos:{native_fp8_vs_bf16_cos},"
        f" native_fp8_cast_vs_bf16_l2:{native_fp8_vs_bf16_l2},"
        f" simo_fp8_per_group_vs_native_fp8_cast_cos:{simo_vs_native_fp8_cos},"
        f" simo_fp8_per_group_vs_native_fp8_cast_l2:{simo_vs_native_fp8_l2}"
    )
