import argparse

import ignite.distributed as idist
import torch


def _mp_train(rank):

    # Specific ignite.distributed
    print(
        idist.get_rank(),
        "- backend=",
        idist.backend(),
        "- world size",
        idist.get_world_size(),
        "- device",
        idist.device(),
    )
    print(idist.get_rank(), " with seed ", torch.initial_seed())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pytorch Ignite - idist")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--nproc_per_node", type=int)
    args_parsed = parser.parse_args()

    # idist from ignite handles multiple backend (gloo, nccl, horovod, xla)
    # and launcher (torch.distributed.launch, horovodrun, slurm)
    spawn_kwargs = dict()
    if args_parsed.nproc_per_node is not None:
        spawn_kwargs["nproc_per_node"] = args_parsed.nproc_per_node

    # Specific ignite.distributed
    with idist.Parallel(backend=args_parsed.backend, **spawn_kwargs) as parallel:
        parallel.run(_mp_train)
