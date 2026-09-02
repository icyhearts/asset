# Attention API findings for SGLang v0.5.18-local-dep

## Current run blocker observed in the supplied log

`temp/sgl.dsv4-flash.log.2026_09_01___15_40_59` fails before model execution because the environment has `flashinfer==0.6.17` but `flashinfer-cubin==0.6.12`:

```text
RuntimeError: flashinfer-cubin version (0.6.12) does not match flashinfer version (0.6.17)
```

The release `python/pyproject.toml` pins `flashinfer_python[cu13]==0.6.17`; the SGLang CI dependency script installs `flashinfer-cubin==0.6.17` from `https://flashinfer.ai/whl`.  In the target conda environment, repair with (using the same Python/pip):

```bash
conda activate /share_data/users/like/miniconda3/envs/simo_sglang
python -m pip install --force-reinstall 'flashinfer-python[cu13]==0.6.17'
python -m pip install --force-reinstall 'flashinfer-cubin==0.6.17' \
  --index-url https://flashinfer.ai/whl
python -m pip show flashinfer-python flashinfer-cubin
```

The first command can use the site's configured PyPI mirror; the second must use the FlashInfer wheel index.  Verify both `Version:` fields match before launching.  `FLASHINFER_DISABLE_VERSION_CHECK=1` is only a diagnostic workaround and can leave incompatible cubins loaded.  This is an environment dependency issue and is independent of the Simo KV-cache API changes; after fixing it, re-run to expose any subsequent import/signature errors.

## Imports and moved kernels

The release branch deletes `sglang.srt.layers.attention.triton_ops.decode_attention` and moves the native implementation to `sglang.kernels.ops.attention.decode_attention`.  Therefore the import at `simo/extensions/sglang_simo/layers/attention/triton_ops/decode_attention.py:6` must become:

```python
from sglang.kernels.ops.attention.decode_attention import _fwd_kernel_stage2
```

The native extend implementation also moved to `sglang.kernels.ops.attention.extend_attention`, but Simo's `extend_attention.py` is a private packed-KV kernel and does not import that module.  `set_kv_buffer.py` likewise has no old SGLang kernel import; it can stay independent of the native writer.

## New backend arguments

Release `TritonAttnBackend.forward_extend` and `forward_decode` add `score_mod=None` and `aux_tensors=None` (release `python/sglang/srt/layers/attention/triton_backend.py`, around lines 1325 and 1829).  `SIMOTritonAttnBackend` overrides both methods without these parameters (`triton_simo_backend.py` around lines 124 and 217).  New callers can consequently fail with an unexpected-keyword error.  Add the two optional parameters to the overrides; pass them to `super()` on unquantized paths.  The custom quantized kernels currently do not implement score modification/aux tensors, so either thread them through all custom stages or reject them explicitly with a clear `NotImplementedError`/feature check rather than silently dropping them.

The release base backend still forwards these through `**kwargs`, so this is an override compatibility issue, not a required change to the base class.

## KVWriteLoc/full_loc

Release parent `TritonAttnBackend` wraps cache writes in `KVWriteLoc(..., full_loc=...)` for unified/virtual allocators and handles DCP translation before `set_kv_buffer` (release `triton_backend.py` around lines 1292-1323 and 1865-1898).  Simo already has `_make_kv_write_loc`, and uses it for quantized MHA, but the quantized MLA branch directly calls:

```python
self.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
```

(`triton_simo_backend.py` around lines 242-250).  Change this to use `_make_kv_write_loc(forward_batch, self.forward_metadata)` (or an equivalent `KVWriteLoc`) so `full_loc` is preserved.  This matters when the release allocator returns a virtual loc separate from the dense loc.

## Native versus Simo kernel signatures

Release native `decode_attention_fwd*` and `extend_attention_fwd*` add `score_mod`, `aux_tensors`; release extend also adds `extend_seq_lens_cpu`, and native kernels now support 3D/4D/page-aware strides.  Simo's wrappers are separate APIs for packed uint8 custom formats:

* `simo/.../decode_attention.py` `decode_attention_fwd` (around lines 842-890) has no `k_scale/v_scale` and calls the imported `_fwd_kernel_stage2` positionally.  Do not insert the release native `k_scale/v_scale` arguments into this wrapper unless every Simo call site is changed; keep its private signature or make additions keyword-only.
* `simo/.../extend_attention.py` `extend_attention_fwd` (around lines 700-723) likewise has a private signature and custom dequantization.  Do not replace it with the release native function: the native function expects logical/native K/V, while Simo's cache is packed.
* Simo custom decode/extend kernels enforce 3D NHD buffers and use custom quantization constexprs.  Keep that restriction (or add a clear guard) unless implementing 4D/page-stride support.  `page_size` is only the flat-slot semantic in the current custom implementation.

The imported release `_fwd_kernel_stage2` remains positionally compatible for Simo's call because its new `FORCED_KV_SPLITS` argument is optional.

## Other parent API observations

Release `TritonAttnBackend` moved imports for native attention, metadata, KV indices, and verify kernels under `sglang.kernels.ops.*`; Simo only needs the decode import above unless it starts calling those native helpers.  The old `get_verify_buffers_to_fill_after_draft` API was replaced by the `verify_mask` property; Simo inherits the release parent and does not appear to override/use the removed getter.

Release MHA pool adds `quant_method` and `allocation_label` constructor parameters and a logical `get_v_head_dim()`.  Simo's constructor signature introspection handles the older optional layout/capture parameters but does not forward the new two (custom quantization intentionally remains separate).  More importantly, Simo's `get_v_head_dim()` currently returns `v_combined_head_size` (packed bytes), not the logical V head dimension; release backend code can call this method for mambaish/hybrid paths, so it should return the logical `v_head_dim` or be guarded to custom-only callers.
