"""
Usage:
python3 offline_batch_inference.py  --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import dataclasses

import sglang as sgl
from sglang.srt.server_args import ServerArgs


def main(
    server_args: ServerArgs,
):
    # Sample prompts.
    prompts = [
        "The president of the United States is",
    ]
    # Create a sampling params object.
    # temperature: 设置为 0。当此值小于 _SAMPLING_EPS (1e-6) 时，SGLang会自动切换到贪婪采样模式。 max_new_tokens: 限制生成长度，例如 128。
    # top_p/top_k: 无需设置，在 temperature=0 时会忽略这些参数。
    sampling_params = {"temperature": 0,  "max_new_tokens": 10}

    # Create an LLM.
    llm = sgl.Engine(**dataclasses.asdict(server_args))

    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    main(server_args)
