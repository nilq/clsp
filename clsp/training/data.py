"""Training data management things."""

import os
import torch
import torchaudio
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


from clsp.utils.storage import gcs_download_bytes

from typing import Optional, Any


# Helpers.


def gcs_audio_from_url(
    url: str, do_retry: bool = False, project: Optional[str] = None
) -> tuple[torch.Tensor, int]:
    """Load GCS audio from URL.

    Args:
        url (str): GCS URL to audio blob.
        do_retry (bool, optional): Whether to retry default amount of times.
            Defaults to False.
        project (Optional[str]): Project, otherwise will use default.
            Defaults to None.

    Returns:
        tuple[torch.Tensor, int]: Waveform data and sample-rate.
    """
    buffer = gcs_download_bytes(
        url, raw_download=True, do_retry=do_retry, project=project
    )
    return torchaudio.load(buffer)


# Custom DataLoader.


class TestDataset(Dataset):
    """Dataset for testing without real data."""

    def __init__(self, size: int = 128) -> None:
        """Initialise dataset by size.

        Args:
            size (int): Size of dataset.
        """
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get random data.

        Args:
            index (int): This is ignored. We don't care about indices.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Text and speech tokens.
        """
        return (torch.randint(0, 256, (2, 120)), torch.randint(0, 8192, (2, 250)))


# Distributed training.


def prepare_distributed_dataloader(
    dataset: Dataset,
    rank: int,
    world_size: int,
    batch_size: int = 1,
    pin_memory: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Prepare distributed dataloader.

    Args:
        dataset (Dataset): Dataset to create loader for.
        rank (int): Rank of the current process.
        world_size (int): Number of total replicas.
        pin_memory (bool, optional): Whether to pin memory.
            Defaults to False.
        num_workers (int, optional): Number of loader workers.
            Defaults to 0.

    Returns:
        DataLoader: DataLoader with prepared distributed sampler.
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        drop_last=False,
        shuffle=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        sampler=sampler,
    )

    return loader
