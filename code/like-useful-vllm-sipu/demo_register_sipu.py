"""Demonstrate a SIPU custom op and its fake implementation.

Run from the repository root after sourcing the SIPU SDK setup script:

    python like-useful/demo_register_sipu.py
"""

import torch

from simo.ops.torch_utils import direct_register_custom_op


OP_NAME = "demo_register_sipu"


def sipu_add_one(src_tensor: torch.Tensor) -> torch.Tensor:
    """Runtime implementation selected by the PrivateUse1/SIPU key."""
    print(f"inside_sipu_impl_device= {src_tensor.device}", flush=True)
    return src_tensor + 1


def fake_add_one(src_tensor: torch.Tensor) -> torch.Tensor:
    """Shape-only implementation used for Meta/FakeTensor execution."""
    print(f"inside_fake_impl_device= {src_tensor.device}", flush=True)
    return torch.empty_like(src_tensor)


def main() -> None:
    direct_register_custom_op(
        op_name=OP_NAME,
        op_func=sipu_add_one,
        fake_impl=fake_add_one,
        dispatch_key="PrivateUse1",
    )

    real_input = torch.ones(2, device="sipu", dtype=torch.float32)
    real_output = getattr(torch.ops.simo, OP_NAME)(real_input)
    print(
        f"real_input_device= {real_input.device} "
        f"real_output_device= {real_output.device} "
        f"value= {real_output.cpu()}",
        flush=True,
    )

    # A Meta tensor has no storage. The registered fake implementation only
    # supplies metadata, so this path does not execute the SIPU kernel.
    meta_input = torch.empty(2, device="meta", dtype=torch.float32)
    meta_output = getattr(torch.ops.simo, OP_NAME)(meta_input)
    print(
        f"meta_input_device= {meta_input.device} "
        f"fake_output_device= {meta_output.device} "
        f"fake_output_shape= {tuple(meta_output.shape)}",
        flush=True,
    )

    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        fake_input = torch.ones(2)
        fake_output = getattr(torch.ops.simo, OP_NAME)(fake_input)
        print(
            f"fake_input_type= {type(fake_input).__name__} "
            f"fake_output_type= {type(fake_output).__name__} "
            f"fake_output_device= {fake_output.device}",
            flush=True,
        )


if __name__ == "__main__":
    main()
