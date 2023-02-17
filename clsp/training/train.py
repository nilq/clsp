"""Training functionality."""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch.utils.data import DataLoader
from clsp.training.data import TestDataset, prepare_distributed_dataloader
from clsp.models.clsp import CLSP
from clsp.training.scheduler import cosine_lr

import torch.multiprocessing as mp

from typing import Optional

import os
import einops

# Distributed training helpers.


def _setup_distributed(rank: int, world_size: int) -> None:
    """Setup distributed training.

    Args:
        rank (int): Rank of the current process.
        world_size (int): Number of total replicas.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def _clean_up() -> None:
    """Clean up distributed stuff."""
    dist.destroy_process_group()


# Actual training.


def train(
    model: CLSP,
    epochs: int,
    data_loader: DataLoader,
    learning_rate: float = 5e-4,
    adamw_betas: tuple[float, float] = (0.9, 0.999),
    adamw_eps: float = 1e-8,
    warmup_len: int = 10e4,
) -> None:
    """Train model."""
    # Defining only optimizer, CLSP handles own loss.
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=learning_rate, betas=adamw_betas, eps=adamw_eps
    )

    total_steps = len(data_loader) * epochs
    scheduler = cosine_lr(optimizer, learning_rate, warmup_len, total_steps)

    for epoch in range(epochs):
        print("Epoch:", epoch)
        data_loader.sampler.set_epoch(epoch)

        for i, *batch in enumerate(data_loader):
            scheduler(i)
            # Needed for finding unused params in DDP model.
            optimizer.zero_grad(set_to_none=True)

            # Training only supports single sample at this point.
            for sample in batch:
                text, speech_tokens = sample
                loss = model(
                    text.squeeze(0), speech_tokens.squeeze(0), return_loss=True
                )
                loss.backward()

            optimizer.step()


def main(rank: int, world_size: int) -> None:
    """Main entry point.

    Args:
        rank (int): Rank of the current process.
        world_size (int): Number of total replicas.
    """
    _setup_distributed(rank, world_size)

    dataset = TestDataset()
    loader = prepare_distributed_dataloader(dataset, rank=rank, world_size=world_size)

    model = CLSP(text_mask_percentage=0.2, speech_mask_percentage=0.2).to(rank)
    model = DistributedDataParallel(
        model, device_ids=[rank], output_device=rank, find_unused_parameters=True
    )

    # Training time.
    print("Training ...")
    train(model, 10, data_loader=loader)

    _clean_up()


if __name__ == "__main__":
    world_size = 1

    mp.spawn(main, args=(world_size,), nprocs=world_size)
