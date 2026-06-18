#!/usr/bin/env python3
"""Run torch.distributed.all_reduce on saved RowParallelLinear outputs.

Usage:
  /data/like/miniconda3/envs/simo_sglang/bin/torchrun --nproc_per_node=8 \
    like-useful/torch_all_reduce.py --suffix test
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file


INPUT_TEMPLATE = (
    "/data/like/temp/qdqx_2026_05_09___16_46_17_safetensors-online/"
    "rank-{rank}-prefix-model.layers.0.self_attn.o_proj-forwardcount-0-1.safetensors"
)
INPUT_KEY = "row_parallel_quant_method_out"
OUTPUT_TEMPLATE = "/data/like/temp/torch.distributed.all_reduce.{suffix}.safetensors"
OUTPUT_KEY = "rank_0_all_reduce"


def _env_int(name: str, default: int | None = None) -> int:
    value = os.environ.get(name)
    if value is None:
        if default is None:
            raise RuntimeError(f"{name} is not set; launch with torchrun.")
        return default
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", required=True)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--input-template", default=INPUT_TEMPLATE)
    parser.add_argument("--input-key", default=INPUT_KEY)
    parser.add_argument("--output-template", default=OUTPUT_TEMPLATE)
    parser.add_argument("--output-key", default=OUTPUT_KEY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rank = _env_int("RANK")
    local_rank = _env_int("LOCAL_RANK", rank)
    world_size = _env_int("WORLD_SIZE")
    if world_size != args.tp_size:
        raise RuntimeError(f"WORLD_SIZE={world_size}, expected tp_size={args.tp_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend=args.backend, rank=rank, world_size=world_size)
    try:
        input_path = args.input_template.format(rank=rank)
        print(f"input_path:{input_path}, key:{args.input_key}")
        tensor_dict = load_file(input_path, device=str(device))
        tensor =  tensor_dict[args.input_key].contiguous()

        dist.barrier()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)

        if rank == 0:
            output_path = Path(args.output_template.format(suffix=args.suffix))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_file({args.output_key: tensor.detach().contiguous().cpu()}, str(output_path))
            print(f"saved {args.output_key} to {output_path}")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
