import sglang as sgl


def main():
    # First smoke: dense Llama tiny. Copy the YAML `sipu:` block for other models.
    engine_kwargs = {
      "watchdog_timeout": 2592000,
      "skip_server_warmup": True,
    }
    llm = sgl.Engine(
        model_path="/share_data/inference-framework/tiny-models/Llama-3.1-8B-Instruct-4layer/safetensor_weights",
        device="sipu",
        trust_remote_code=True,
        disable_cuda_graph=True,
        mem_fraction_static=0.6,
        max_total_tokens=256,
        page_size=32,
        context_length=128,
        skip_server_warmup=True,
        attention_backend="sipu",
        **engine_kwargs,
    )
    print(llm.generate(
        ["Hello, this is a SIPU offline inference smoke test."],
        {"temperature": 0, "max_new_tokens": 8},
    ))
    llm.shutdown()


if __name__ == "__main__":
    main()

