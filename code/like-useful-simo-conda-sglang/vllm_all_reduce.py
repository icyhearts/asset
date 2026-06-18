#!/usr/bin/env python3
"""Run vLLM tensor-parallel all-reduce on saved RowParallelLinear outputs.

Usage:
  /data/like/miniconda3/envs/simo_vllm/bin/torchrun --nproc_per_node=8 \
    like-useful/vllm_all_reduce.py
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file


VLLM_REPO = "/data/like/package/vllm-for-conda-simo"
INPUT_TEMPLATE = (
    "/data/like/temp/qdqx_2026_05_09___16_46_17_safetensors-online/"
    "rank-{rank}-prefix-model.layers.0.self_attn.o_proj-forwardcount-0-0.safetensors"
)
INPUT_KEY = "row_parallel_quant_method_out"
OUTPUT_PATH = "/data/like/temp/vllm.safetensors"
OUTPUT_KEY = "rank_0_all_reduce"


def _add_repo_to_path() -> None:
    if VLLM_REPO not in sys.path:
        sys.path.insert(0, VLLM_REPO)


def _env_int(name: str, default: int | None = None) -> int:
    value = os.environ.get(name)
    if value is None:
        if default is None:
            raise RuntimeError(f"{name} is not set; launch with torchrun.")
        return default
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--input-template", default=INPUT_TEMPLATE)
    parser.add_argument("--input-key", default=INPUT_KEY)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--output-key", default=OUTPUT_KEY)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _add_repo_to_path()

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )

    rank = _env_int("RANK")
    local_rank = _env_int("LOCAL_RANK", rank)
    world_size = _env_int("WORLD_SIZE")
    if world_size != args.tp_size:
        raise RuntimeError(f"WORLD_SIZE={world_size}, expected tp_size={args.tp_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    with set_current_vllm_config(VllmConfig()):
        set_custom_all_reduce(not args.disable_custom_all_reduce)
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            backend=args.backend,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=args.tp_size,
            pipeline_model_parallel_size=1,
            prefill_context_model_parallel_size=1,
            decode_context_model_parallel_size=1,
            backend=args.backend,
        )

        try:
            input_path = args.input_template.format(rank=rank)
            tensor = load_file(input_path)[args.input_key].contiguous().to(device)
            dist.barrier()
            reduced = tensor_model_parallel_all_reduce(tensor)
            torch.cuda.synchronize(device)

            if rank == 0:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(
                    {args.output_key: reduced.detach().contiguous().cpu()},
                    str(output_path),
                )
                print(f"saved {args.output_key} to {output_path}")
            dist.barrier()
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
