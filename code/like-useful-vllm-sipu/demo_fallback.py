"""Demonstrate torch-sipu CPU fallback for a custom SIMO operator.

Run from the repository root after sourcing the SIPU SDK setup script:

    python like-useful/demo_fallback.py

The operator deliberately has only a CPU implementation.  A SIPU tensor
therefore enters the CPU implementation through torch-sipu's boxed fallback,
and the fallback copies the result back to SIPU for the caller.
"""

import torch

from simo.ops.torch_utils import direct_register_custom_op


def probe_cpu_fallback(src_tensor: torch.Tensor) -> torch.Tensor:
    """CPU implementation; torch-sipu has already moved src_tensor to CPU."""
    print(f"inside_cpu_impl_device= {src_tensor.device}", flush=True)
    return src_tensor + 1


def main() -> None:
    direct_register_custom_op(
        op_name="probe_cpu_fallback",
        op_func=probe_cpu_fallback,
        dispatch_key="CPU",
    )

    src = torch.ones(2, device="sipu", dtype=torch.float32)
    print(f"before_device= {src.device}", flush=True)
    result = torch.ops.simo.probe_cpu_fallback(src)
    print(
        f"after_device= {result.device} value= {result.cpu()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
