import multiprocessing as mp
from transformers import AutoTokenizer
from pathlib import Path
from typing import Optional

import pytest
import simo

# SIMO registers the UINT8-KV dequantizing Triton backend under `triton_simo`.

LLAMA_3_1_8B_INSTRUCT = (
  "/data_gpu/models/share_data/modelzoo/weights/llm/llama/llama3.1/"
  "llama3.1-8B-Instruct/safetensor_weights"
)

DEEPSEEK_V2_LITE_CHAT_16B_A2_4B = (
  "/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/"
  "DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights/"
)


_MODEL_CASES = [LLAMA_3_1_8B_INSTRUCT, DEEPSEEK_V2_LITE_CHAT_16B_A2_4B]


_SIMO_PACKAGE_ROOT = Path(simo.__file__).resolve().parent
_SIMO_EXAMPLE_QUANT_CONFIG_DIR = (
  _SIMO_PACKAGE_ROOT / "extensions" / "sglang_simo" / "example" / "simo_quantization_config"
)

_KV_CACHE_QUANT_DIR = _SIMO_EXAMPLE_QUANT_CONFIG_DIR / "kv_cache_quant"
_ONLINE_QUANT_DIR = _SIMO_EXAMPLE_QUANT_CONFIG_DIR / "online_quantization"

# Find all config files
_KVQUANT_CONFIGS = sorted(_KV_CACHE_QUANT_DIR.glob("*.json"))
_WEIGHT_CONFIGS = sorted(_ONLINE_QUANT_DIR.glob("*.json"))
_QUANT_CONFIGS = [*_WEIGHT_CONFIGS, *_KVQUANT_CONFIGS]


_EXCLUDE_QUANT_CONFIGS = []


def _get_test_cases():
  return [
    {
      "model_path": model_path,
      "quant_config": str(quant_config),  # Convert Path to string for serialization
      "case_id": (
        f"{Path(model_path).parent.name}-{quant_config.parent.name}-{quant_config.stem}"
      ),
    }
    for model_path in _MODEL_CASES
    for quant_config in _QUANT_CONFIGS
    # Skip quant configs that match excluded glob patterns.
    if not any(quant_config.match(p) for p in _EXCLUDE_QUANT_CONFIGS)
  ]


def _kill_process(proc: mp.Process, *, join_timeout_s: float = 1.0) -> None:
  """Best-effort hard kill to avoid orphan workers (no-op if already dead)."""
  try:
    if proc.is_alive():
      proc.kill()
      proc.join(timeout=join_timeout_s)
  except Exception:
    # We intentionally swallow errors here since this is cleanup code.
    pass


def _run_single_case_in_process(
  *,
  model_path: str,
  quant_config: str,
  timeout_s: Optional[float] = None,
):
  """Run a single e2e case in a dedicated child process and force-kill after completion.

  Notes:
  - We use the "spawn" context to avoid CUDA + fork issues.
  """
  ctx = mp.get_context("spawn")

  proc = ctx.Process(
    target=_single_case_process_entrypoint,
    args=(model_path, quant_config),
    daemon=False,
  )

  proc.start()
  try:
    proc.join(timeout=timeout_s)
    if proc.is_alive():
      # Let `finally` do the single kill/cleanup path.
      raise RuntimeError(
        "Single-case process hung and was killed "
        f"(timeout_s={timeout_s}): model_path={model_path}, "
        f"quant_config={quant_config}"
      )
    if proc.exitcode != 0:
      raise RuntimeError(
        "Single-case process failed "
        f"(exitcode={proc.exitcode}): model_path={model_path}, "
        f"quant_config={quant_config}"
      )
  finally:
    _kill_process(proc, join_timeout_s=1.0)


def _single_case_process_entrypoint(model_path, quant_config):
  """Child process entrypoint for running a single case."""
  sglang = pytest.importorskip("sglang")
  _run_single_case(
    sgl=sglang,
    model_path=model_path,
    quant_config=quant_config,
  )


@pytest.mark.parametrize(
  "model_path,quant_config",
  [
    pytest.param(
      c["model_path"],
      c["quant_config"],
      id=c["case_id"],
    )
    for c in _get_test_cases()
  ],
)
def test_sglang_simo_generate_smoke(model_path, quant_config):
  """E2E smoke test for sglang + SIMO quantization (one case per parametrized test)."""
  torch = pytest.importorskip("torch")
  if not torch.cuda.is_available():
    pytest.skip("CUDA is required for this e2e test.")

  _run_single_case_in_process(
    model_path=model_path,
    quant_config=quant_config,
  )


def _run_single_case(
  *,
  sgl,
  model_path,
  quant_config,
):
  model_dir = Path(model_path)
  if not model_dir.exists():
    raise pytest.skip.Exception(f"Model path does not exist: {model_dir}")

  quant_config_file = Path(quant_config)
  if not quant_config_file.exists():
    raise pytest.skip.Exception(f"Quantization config file not found: {quant_config_file}")

  # hf_overrides = {"quantization_config_file": str(quant_config_file)}
  json_model_override_args = f'{{"quantization_config_file": "{str(quant_config_file)}"}}'

  llm = None
  try:
    engine_kwargs = {
      "model_path": model_path,
      "quantization": "simo",
      "json_model_override_args": json_model_override_args,
      "mem_fraction_static": 0.5,
    }
    if quant_config_file.parent.name == "kv_cache_quant":
      engine_kwargs["attention_backend"] = "triton_simo"
    llm = sgl.Engine(**engine_kwargs)

    sampling_params = {"temperature": 0.0, "top_p": 0.95, "max_new_tokens": 16}

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "What is the capital of France? "
                    "Answer with exactly one word."
                ),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    generating_prompts = [prompt]

    outputs = llm.generate(generating_prompts, sampling_params)

    print("-" * 50)
    for prompt, output in zip(generating_prompts, outputs, strict=True):
      generated_text = output["text"]
      print(f"Prompt: {prompt}\nGenerated text: {generated_text}")
      assert "Paris" in generated_text, f"Generated text does not contain 'Paris': {generated_text}"
      print("-" * 50)

  finally:
    if llm is not None:
      llm.shutdown()  # 杀掉 Engine 启动的所有子进程 (调度器、detokenizer 等)
      del llm
    from sglang.srt.distributed.parallel_state import (
      cleanup_dist_env_and_memory,
    )

    cleanup_dist_env_and_memory()


if __name__ == "__main__":
  sglang = pytest.importorskip("sglang")
  model_path="/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights/"
  #quant_config="/share_data/users/like/package/h100/package/simo_conda_sglang/simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w4a4_nvfp_4_over_6.json"
  #quant_config="/share_data/users/like/package/h100/package/simo_conda_sglang/simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w8a8_int8_per_block.json"
  quant_config="/share_data/users/like/package/h100/package/simo_conda_sglang/simo/extensions/sglang_simo/example/simo_quantization_config/kv_cache_quant/quant_config_kvquant_mxfp8.json"
  _run_single_case(sgl=sglang, model_path=model_path, quant_config=quant_config)
