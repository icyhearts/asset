import multiprocessing as mp
from transformers import AutoTokenizer
import os
from pathlib import Path
from typing import Optional

import pytest

DEEPSEEK_V2_LITE_CHAT_16B_A2_4B = (
  "/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights/",
  "triton",
)


_MODEL_CASES = [DEEPSEEK_V2_LITE_CHAT_16B_A2_4B]


_SIMO_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SIMO_PACKAGE_ROOT = _SIMO_PROJECT_ROOT / "simo"
_SIMO_EXAMPLE_QUANT_CONFIG_DIR = (
  _SIMO_PACKAGE_ROOT / "extensions" / "sglang_simo" / "example" / "simo_quantization_config"
)

# _KV_CACHE_QUANT_DIR = _SIMO_EXAMPLE_QUANT_CONFIG_DIR / "kv_cache_quant"
_ONLINE_QUANT_DIR = _SIMO_EXAMPLE_QUANT_CONFIG_DIR / "online_quantization"

# Find all config files
_WEIGHT_CONFIGS = sorted(_ONLINE_QUANT_DIR.glob("*.json"))
_QUANT_CONFIGS = [*_WEIGHT_CONFIGS]


_EXCLUDE_QUANT_CONFIGS = []


def _get_test_cases():
  return [
    {
      "model_path": model_path,
      "attention_backend": attention_backend,
      "quant_config": str(quant_config),  # Convert Path to string for serialization
      "case_id": f"{str(Path(model_path)).split('/')[-2]}-{quant_config.stem}",
    }
    for model_path, attention_backend in _MODEL_CASES
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
  attention_backend: str,
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
    args=(model_path, attention_backend, quant_config),
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
        f"attention_backend={attention_backend}, quant_config={quant_config}"
      )
    if proc.exitcode != 0:
      raise RuntimeError(
        "Single-case process failed "
        f"(exitcode={proc.exitcode}): model_path={model_path}, "
        f"attention_backend={attention_backend}, quant_config={quant_config}"
      )
  finally:
    _kill_process(proc, join_timeout_s=1.0)


def _single_case_process_entrypoint(model_path, attention_backend, quant_config):
  """Child process entrypoint for running a single case."""
  sglang = pytest.importorskip("sglang")
  _run_single_case(
    sgl=sglang,
    model_path=model_path,
    attention_backend=attention_backend,
    quant_config=quant_config,
  )


@pytest.mark.parametrize(
  "model_path,attention_backend,quant_config",
  [
    pytest.param(
      c["model_path"],
      c["attention_backend"],
      c["quant_config"],
      id=c["case_id"],
    )
    for c in _get_test_cases()
  ],
)
def test_sglang_simo_generate_smoke(model_path, attention_backend, quant_config):
  """E2E smoke test for sglang + SIMO quantization (one case per parametrized test)."""
  torch = pytest.importorskip("torch")
  if not torch.cuda.is_available():
    pytest.skip("CUDA is required for this e2e test.")

  _run_single_case_in_process(
    model_path=model_path,
    attention_backend=attention_backend,
    quant_config=quant_config,
  )


def _run_single_case(
  *,
  sgl,
  model_path,
  attention_backend,
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
    llm = sgl.Engine(
      model_path=model_path,
      quantization="simo",
      json_model_override_args=json_model_override_args,
      mem_fraction_static=0.5,
      log_level="info",
    )

    sampling_params = {"temperature": 0.0, "top_p": 0.95, "max_new_tokens": 16}
    # Common prefix.
    prefix = (
      "You are an expert school principal, skilled in effectively managing "
      "faculty and staff. Draft 10-15 questions for a potential first grade "
      "Head Teacher for my K-12, all-girls', independent school that emphasizes "
      "community, joyful discovery, and life-long learning. The candidate is "
      "coming in for a first-round panel interview for a 8th grade Math "
      "teaching role. They have 5 years of previous teaching experience "
      "as an assistant teacher at a co-ed, public school with experience "
      "in middle school math teaching. Based on these information, fulfill "
      "the following paragraph: "
    )
    prefix=os.getenv("SMOKE_PREFIX", "You are an expert school principal, skilled in effectively managing faculty and staff. Draft 10-15 questions for a potential first grade Head Teacher for my K-12, all-girls', independent school that emphasizes community, joyful discovery, and life-long learning. The candidate is coming in for a first-round panel interview for a 8th grade Math teaching role. They have 5 years of previous teaching experience as an assistant teacher at a co-ed, public school with experience in middle school math teaching. Based on these information, fulfill the following paragraph: ")
    print(f"prefix:|{prefix}|")

    # Sample prompts.
    prompts = ["The capital of France is"]

    generating_prompts = [prefix + prompt for prompt in prompts]
    ###
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
  import os
  model_path=os.getenv("SMOKE_MODEL", "/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights/")
  print(f"model_path:{model_path}")
  attention_backend="triton"
  #quant_config="/share_data/users/like/package/h100/package/simo_conda_sglang/simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w4a4_nvfp_4_over_6.json"
  #quant_config="/share_data/users/like/package/h100/package/simo_conda_sglang/simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w8a8_int8_per_block.json"
  quant_config="/share_data/users/like/package/h100/package/simo_conda_sglang/simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w4a16_int4_per_group.json"
  _run_single_case(sgl=sglang, model_path=model_path, attention_backend=attention_backend, quant_config=quant_config)
