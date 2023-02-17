"""Speech encoder toolbox."""

import torch
import torch.nn as nn
import whisper


from functools import cache
from typing import Optional, Union
from whisper.audio import log_mel_spectrogram


def encoder(
    name: str = "tiny", device: Optional[torch.device | str] = None
) -> tuple[nn.Module, int, torch.device]:
    """Get Whisper encoder by model name.

    Args:
        name (str): Whisper model name.
            Defaults to "tiny".
        device (Optional[torch.device], optional): Device for encoder.
            Defaults to None.

    Returns:
        tuple[nn.Module, int, torch.device]: Whisper encoder, its dimensions and its device.
    """
    model = whisper.load_model(name, device=device)
    return model.encoder, model.dims, model.device


def encode_audio(
    audio: Union[str, torch.Tensor],
    name: str = "tiny",
    device: Optional[torch.device | str] = None,
) -> list[torch.Tensor]:
    """Get Whisper encoding.

    Args:
        name (str): Whisper model name.
            Defaults to "tiny".
        device (Optional[torch.device], optional): Device for encoder and embedding.
            Defaults to None.

    Returns:
        torch.Tensor: Whisper encoding of given audio.
    """
    # Only 80 filters supported by Whisper.
    model_encoder, dims, device = encoder(name, device=device)
    mel = log_mel_spectrogram(audio).to(device)

    embeddings: list[torch.Tensor] = []
    number_of_frames: int = mel.shape[-1]
    input_stride = whisper.utils.exact_div(
        whisper.audio.N_FRAMES, 1
    )

    for seek in range(0, number_of_frames, input_stride):
        segment = whisper.audio.pad_or_trim(mel[:, seek:], whisper.audio.N_FRAMES).to(device)
        embeddings.append(
            model_encoder(segment.unsqueeze(0))
        )

    return embeddings
