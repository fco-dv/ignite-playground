import argparse

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


def _mp_train(rank):
    # Specific xla
    print(
        xm.get_ordinal(),
        "- device",
        xm.xla_device(),
    )

    print(xm.get_ordinal(), " with seed ", torch.initial_seed())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Torch Native - XLA")
    parser.add_argument("--backend", type=str, default="xla-tpu")
    parser.add_argument("--nproc_per_node", type=int, default=8)
    args_parsed = parser.parse_args()

    assert args_parsed.backend == "xla-tpu"

    # Specific xla
    xmp.spawn(_mp_train, nprocs=args_parsed.nproc_per_node)
