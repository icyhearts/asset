import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from simo.extensions.sglang_simo.mem_cache.memory_pool import SIMOMLATokenToKVPool
from simo.extensions.sglang_simo.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd_grouped as simo_decode_attention_fwd_grouped,
)
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
from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd_grouped as sglang_decode_attention_fwd_grouped,
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


def _metric_pair(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
  ref_flat = reference.reshape(-1)
  cand_flat = candidate.reshape(-1)
  return (
      float(cosine_similarity(cand_flat, ref_flat).detach().cpu()),
      float(relative_l2_error(ref_flat, cand_flat).detach().cpu()),
  )


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


def native_sglang_fp8_cast_decode_cache(
    k_buffer: torch.Tensor,
    v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  if not hasattr(torch, "float8_e4m3fn"):
    raise RuntimeError("Current torch does not expose torch.float8_e4m3fn")

  k_fp8 = k_buffer.to(torch.float8_e4m3fn).contiguous()
  v_fp8 = k_fp8[..., :v_head_dim].contiguous()
  return k_fp8, v_fp8


def load_kv_cache_quant_spec(kv_cache_quant_file: str) -> tuple[QuantizeSpecType, object]:
  with open(kv_cache_quant_dir + kv_cache_quant_file) as ifp:
    config = json.load(ifp)
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
  return kv_cache_quant_spec, kv_cache_downcast_kernel


def build_simo_decode_layer(
    kv_cache_quant_spec: QuantizeSpecType,
    kv_cache_downcast_kernel,
    qk_head_dim: int,
    v_head_dim: int,
) -> SimpleNamespace:
  rope_dim = qk_head_dim - v_head_dim
  assert rope_dim >= 0

  x_q, scale = kv_cache_downcast_kernel(torch.empty(1, v_head_dim, device="meta"))
  x_q_rope, scale_rope = kv_cache_downcast_kernel(torch.empty(1, rope_dim, device="meta"))

  return SimpleNamespace(
      layer_id=0,
      qk_head_dim=qk_head_dim,
      v_head_dim=v_head_dim,
      kv_cache_quant_spec=kv_cache_quant_spec,
      kv_cache_downcast_kernel=kv_cache_downcast_kernel,
      packed_head_size=x_q.contiguous().view(torch.uint8).shape[-1],
      scale_head_size=scale.contiguous().view(torch.uint8).shape[-1],
      packed_head_size_rope=x_q_rope.contiguous().view(torch.uint8).shape[-1],
      scale_head_size_rope=scale_rope.contiguous().view(torch.uint8).shape[-1],
  )


def build_simo_fp8_per_group_decode_cache(
    k_buffer: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    q_batch_size: int,
    layer: SimpleNamespace,
) -> torch.Tensor:
  active_token_count = int(kv_indptr[q_batch_size].detach().cpu())
  active_locs = kv_indices[:active_token_count].long()
  active_locs = torch.unique(active_locs[(active_locs >= 0) & (active_locs < k_buffer.shape[0])])

  simo_pool = SIMOMLATokenToKVPool(
      size=k_buffer.shape[0] - 1,
      page_size=1,
      dtype=torch.bfloat16,
      kv_lora_rank=layer.v_head_dim,
      qk_rope_head_dim=layer.qk_head_dim - layer.v_head_dim,
      layer_num=1,
      device=str(k_buffer.device),
      enable_memory_saver=False,
      start_layer=None,
      end_layer=None,
      use_dsa=False,
      override_kv_cache_dim=None,
      kv_cache_quant_spec=layer.kv_cache_quant_spec,
      kv_cache_downcast_kernel=layer.kv_cache_downcast_kernel,
  )

  if active_locs.numel() > 0:
    cache_k_nope = k_buffer[active_locs, :, : layer.v_head_dim].contiguous()
    cache_k_rope = k_buffer[active_locs, :, layer.v_head_dim : layer.qk_head_dim].contiguous()
    simo_pool.set_mla_kv_buffer(layer, active_locs.contiguous(), cache_k_nope, cache_k_rope)

  return simo_pool.kv_buffer[layer.layer_id].contiguous()

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

def _get_dump_timestamp(path: Path, prefix: str, suffix: str) -> str:
  name = path.name
  assert name.startswith(prefix) and name.endswith(suffix)
  return name[len(prefix) : -len(suffix)]


def iter_decode_dump_groups(dump_dir: str):
  dump_path = Path(dump_dir)
  for safe_path in sorted(dump_path.glob("decode_attention_fwd_grouped.*.safetensors")):
    ts = _get_dump_timestamp(
        safe_path, "decode_attention_fwd_grouped.", ".safetensors"
    )
    args_path = dump_path / f"non_tensor_args.{ts}.json"
    if args_path.exists():
      yield ts, safe_path, args_path


def _get_arg(name: str, tensors: dict[str, torch.Tensor], non_tensor_args: dict, default=None):
  if name in tensors:
    return tensors[name]
  return non_tensor_args.get(name, default)


def _empty_decode_outputs(tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  return (
      torch.empty_like(tensors["o"]),
      torch.empty_like(tensors["attn_logits"]),
      torch.empty_like(tensors["attn_lse"]),
  )


def replay_one_decode_dump(
    safe_path: Path,
    args_path: Path,
    kv_cache_quant_spec: QuantizeSpecType,
    kv_cache_downcast_kernel,
    device: str,
) -> dict[str, float]:
  tensors = load_file(str(safe_path), device=device)
  with open(args_path) as ifp:
    non_tensor_args = json.load(ifp)

  q = tensors["q"].contiguous()
  k_buffer = tensors["k_buffer"].contiguous()
  v_buffer = tensors["v_buffer"].contiguous()
  kv_indptr = tensors["kv_indptr"].contiguous()
  kv_indices = tensors["kv_indices"].contiguous()
  num_kv_splits = tensors["num_kv_splits"].contiguous()
  max_kv_splits = non_tensor_args["max_kv_splits"]
  sm_scale_withk = non_tensor_args["sm_scale_withk"]
  v_scale = _get_arg("v_scale", tensors, non_tensor_args, 1.0)
  logit_cap = non_tensor_args.get("logit_cap", 0.0)
  sinks = _get_arg("sinks", tensors, non_tensor_args, None)
  xai_temperature_len = non_tensor_args.get("xai_temperature_len", -1)
  has_mla = non_tensor_args.get("has_mla", True)
  use_pdl = non_tensor_args.get("use_pdl", False)

  if float(v_scale) != 1.0:
    raise ValueError(f"Unexpected v_scale={v_scale} in {args_path}")

  qk_head_dim = q.shape[-1]
  v_head_dim = v_buffer.shape[-1]
  assert k_buffer.shape[-1] == qk_head_dim

  o_bf16, attn_logits_bf16, attn_lse_bf16 = _empty_decode_outputs(tensors)
  sglang_decode_attention_fwd_grouped(
      q,
      k_buffer,
      v_buffer,
      o_bf16,
      kv_indptr,
      kv_indices,
      attn_logits_bf16,
      attn_lse_bf16,
      num_kv_splits,
      max_kv_splits,
      sm_scale_withk,
      v_scale,
      logit_cap=logit_cap,
      sinks=sinks,
      xai_temperature_len=xai_temperature_len,
      has_mla=has_mla,
      use_pdl=use_pdl,
  )

  k_fp8, v_fp8 = native_sglang_fp8_cast_decode_cache(k_buffer, v_head_dim)
  o_native_fp8, attn_logits_native_fp8, attn_lse_native_fp8 = _empty_decode_outputs(tensors)
  sglang_decode_attention_fwd_grouped(
      q,
      k_fp8,
      v_fp8,
      o_native_fp8,
      kv_indptr,
      kv_indices,
      attn_logits_native_fp8,
      attn_lse_native_fp8,
      num_kv_splits,
      max_kv_splits,
      sm_scale_withk,
      v_scale,
      logit_cap=logit_cap,
      sinks=sinks,
      xai_temperature_len=xai_temperature_len,
      has_mla=has_mla,
      use_pdl=use_pdl,
  )

  layer = build_simo_decode_layer(
      kv_cache_quant_spec,
      kv_cache_downcast_kernel,
      qk_head_dim=qk_head_dim,
      v_head_dim=v_head_dim,
  )
  simo_kv_cache = build_simo_fp8_per_group_decode_cache(
      k_buffer, kv_indices, kv_indptr, q.shape[0], layer
  )
  o_simo, attn_logits_simo, attn_lse_simo = _empty_decode_outputs(tensors)
  simo_decode_attention_fwd_grouped(
      q,
      simo_kv_cache,
      simo_kv_cache,
      o_simo,
      kv_indptr,
      kv_indices,
      attn_logits_simo,
      attn_lse_simo,
      num_kv_splits,
      max_kv_splits,
      sm_scale_withk,
      logit_cap=logit_cap,
      sinks=sinks,
      xai_temperature_len=xai_temperature_len,
      layer=layer,
  )
  torch.cuda.synchronize()

  native_cos, native_l2 = _metric_pair(o_bf16, o_native_fp8)
  simo_cos, simo_l2 = _metric_pair(o_bf16, o_simo)
  return {
      "native_fp8_cast_vs_bf16_cos": native_cos,
      "native_fp8_cast_vs_bf16_l2": native_l2,
      "simo_fp8_per_group_vs_bf16_cos": simo_cos,
      "simo_fp8_per_group_vs_bf16_l2": simo_l2,
      "batch": q.shape[0],
      "active_tokens": int(kv_indptr[q.shape[0]].detach().cpu()),
      "use_pdl": int(bool(use_pdl)),
  }


def run_decode_attention_dump_replay():
  dump_dir = os.environ.get(
      "DEBUG_DECODE_REPLAY_DIR",
      "/data/like/temp/sgl_safe_tensor_sgl_decode_attention_fwd_grouped_dir",
  )
  limit = int(os.environ.get("DEBUG_DECODE_REPLAY_LIMIT", "0"))
  device = os.environ.get("DEBUG_DECODE_REPLAY_DEVICE", pool_device)
  kv_cache_quant_spec, kv_cache_downcast_kernel = load_kv_cache_quant_spec(
      "quant_config_kvquant_fp8_per_group.json"
  )

  results = []
  for idx, (ts, safe_path, args_path) in enumerate(iter_decode_dump_groups(dump_dir), start=1):
    if limit > 0 and idx > limit:
      break
    result = replay_one_decode_dump(
        safe_path, args_path, kv_cache_quant_spec, kv_cache_downcast_kernel, device
    )
    results.append(result)
    print(
        "decode_replay:"
        f" idx:{idx}, ts:{ts}, batch:{result['batch']}, active_tokens:{result['active_tokens']},"
        f" use_pdl:{result['use_pdl']},"
        f" native_fp8_cast_vs_bf16_cos:{result['native_fp8_cast_vs_bf16_cos']:.9f},"
        f" native_fp8_cast_vs_bf16_l2:{result['native_fp8_cast_vs_bf16_l2']:.9f},"
        f" simo_fp8_per_group_vs_bf16_cos:{result['simo_fp8_per_group_vs_bf16_cos']:.9f},"
        f" simo_fp8_per_group_vs_bf16_l2:{result['simo_fp8_per_group_vs_bf16_l2']:.9f}"
    )

  if not results:
    print(f"decode_replay: no matched dump groups found in {dump_dir}")
    return

  native_l2_avg = sum(x["native_fp8_cast_vs_bf16_l2"] for x in results) / len(results)
  simo_l2_avg = sum(x["simo_fp8_per_group_vs_bf16_l2"] for x in results) / len(results)
  native_cos_avg = sum(x["native_fp8_cast_vs_bf16_cos"] for x in results) / len(results)
  simo_cos_avg = sum(x["simo_fp8_per_group_vs_bf16_cos"] for x in results) / len(results)
  print(
      "decode_replay_summary:"
      f" count:{len(results)},"
      f" native_fp8_cast_vs_bf16_cos_avg:{native_cos_avg:.9f},"
      f" native_fp8_cast_vs_bf16_l2_avg:{native_l2_avg:.9f},"
      f" simo_fp8_per_group_vs_bf16_cos_avg:{simo_cos_avg:.9f},"
      f" simo_fp8_per_group_vs_bf16_l2_avg:{simo_l2_avg:.9f}"
  )


def run_set_mla_kv_cache_compare():
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
    kv_cache_quant_spec, kv_cache_downcast_kernel = load_kv_cache_quant_spec(
        kv_cache_quant_file
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


if __name__ == "__main__":
  if os.environ.get("DEBUG_RUN_SET_MLA_COMPARE", "0") != "0":
    run_set_mla_kv_cache_compare()
  if os.environ.get("DEBUG_RUN_DECODE_REPLAY", "1") != "0":
    run_decode_attention_dump_replay()
