import argparse

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import Dataset

class RndDataset(Dataset):
    def __init__(self, nb_samples=128):
        self._nb_samples = nb_samples

    def __len__(self):
        return self._nb_samples

    def __getitem__(self, index):
        x = torch.randn((3, 32, 32))
        y = torch.randint(0, 100, (1,)).item()
        return x, y

def _mp_train(rank, world_size, backend):
    device = xm.xla_device()
    print(
        xm.get_ordinal(),
        "- backend=",
        backend,
        "- world size",
        xm.xrt_world_size(),
        "- device",
        device,
        " with seed ", torch.initial_seed()
    )
    #...
    # dataset  = RndDataset()
    # # Specific xla
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(),
    # )
    #
    # train_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=int(16 / xm.xrt_world_size()),
    #     num_workers=1,
    #     sampler=train_sampler,
    # )
    #
    # # Specific xla
    # para_loader = pl.MpDeviceLoader(train_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Torch Native - XLA")
    parser.add_argument("--backend", type=str, default="xla-tpu")
    parser.add_argument("--nproc_per_node", type=int, default=8)
    args_parsed = parser.parse_args()

    assert args_parsed.backend == "xla-tpu"

    args = (args_parsed.nproc_per_node, args_parsed.backend)
    # Specific xla
    xmp.spawn(_mp_train, args=args, nprocs=args_parsed.nproc_per_node)
