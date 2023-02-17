"""Training data management things."""

import os
from pathlib import Path
import polars as pl
import torch
import torchaudio
import torch.distributed as dist
import torchaudio.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from clsp.utils.storage import gcs_download_bytes, bigquery_query
from clsp.utils.speech import encode_audio
from clsp.utils.text import encode_to_tensor

from typing import Optional, Any

import whisper


# Helpers.


def gcs_audio_from_url(
    url: str, do_retry: bool = False, project: Optional[str] = None
) -> tuple[torch.Tensor, int]:
    """Load GCS audio from URL.

    Args:
        url (str): GCS URL to audio blob.
        do_retry (bool, optional): Whether to retry default amount of times.
            Defaults to False.
        project (Optional[str], optional): Project, otherwise will use default.
            Defaults to None.

    Returns:
        tuple[torch.Tensor, int]: Waveform data and sample-rate.
    """
    buffer = gcs_download_bytes(
        url, raw_download=True, do_retry=do_retry, project=project
    )
    return torchaudio.load(buffer)


def encode_speech_from_url(
    url: str,
    whisper_name: str = "tiny",
    do_retry: bool = False,
    project: Optional[str] = None,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Whisper-encode speech directly from URL.

    Args:
        url (str): GCS URL to audio blob.
        do_retry (bool, optional): Whether to retry default amount of times.
            Defaults to False.
        project (Optional[str], optional): Project, otherwise will use default.
            Defaults to None.
        device (Optional[torch.device], optional): Device for encoder and returned embedding.
            Defaults to None.
    """
    waveform, sr = gcs_audio_from_url(url, do_retry, project)
    if sr != 16000:  # Whisper only works at 16kHz.
        waveform = F.resample(waveform, sr, 16000)

    return encode_audio(waveform, name=whisper_name, device=device)


# Custom Datasets.


class CloudDataset(Dataset):
    """Dataset loading from BigQuery and Cloud Storage."""

    def __init__(
        self,
        uri: str,
        limit: Optional[int] = None,
        text_column: str = "text",
        speech_url_column: str = "speech_gcs_url",
    ) -> None:
        """Initialise cloud dataset.

        Notes:
            Needs URI to BigQuery view with `text` and `speech_gcs_url` columns by default.
            All data will be loaded, unless limit is passed.

        Args:
            uri (str): BigQuery URI to grab transcripts and GCS URIs from.
            limit (Optional[int], optional): Optional limit on amount of data loaded.
            text_column (str, optional): Name of column in table containing text.
                Defaults to "text".
            speech_url_column (str, optional): Name of column in table containing speech URL.
                Defaults to "speech_gcs_url".
        """
        self.text_column = text_column
        self.speech_url_column = speech_url_column

        self._data: pl.DataFrame = bigquery_query(
            f"SELECT {text_column}, {speech_url_column} FROM {uri}" + f"LIMIT {limit}"
            if limit
            else ""
        )

    def __len__(self) -> int:
        """Length of dataset.

        Returns:
            int: Length of dataset.
        """
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get text and speech token pairs from cloud.

        Args:
            index (int): Index of row to get data for.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Text and speech token pair.
        """
        row = self._data[index]

        text = row[self.text_column]
        speech_url = row[self.text_column]

        tokens = encode_to_tensor(text)
        whisper_embedding = encode_speech_from_url(speech_url)

        return tokens, whisper_embedding


class TestDataset(Dataset):
    """Dataset for testing without real data."""
    

    def __init__(self, size: int = 128) -> None:
        """Initialise dataset by size.

        Args:
            size (int): Size of dataset.
        """
        whisper = whisper.load_model("tiny")
        waveform, sr = torchaudio.load(
            Path(os.path.dirname(__file__)) / ".." / ".." / "data" / "joe.wav"
        )
    
        


    def __len__(self) -> int:
        """Length of dataset.

        Returns:
            int: Length of dataset.
        """
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
