"""Speech encoder toolbox."""

import torch
import torch.nn as nn
import whisper


from functools import cache
from typing import Optional, Union, Any
from whisper.audio import log_mel_spectrogram


def detect_language(
    audio: torch.Tensor,
    model: Optional[whisper.Whisper] = None,
    model_name: str = "tiny",
    device: Optional[torch.device | str] = None,
) -> str:
    """Detect language from audio.

    Args:
        audio (torch.Tensor): Audio to detect language of.
        model (Optional[whisper.Whisper], optional): Whisper model to use, otherwise get new.
            Defaults to None.
        model_name (str, optional): Name of model to use if no model is passed.
            Defaults to "tiny"
        device (Optional[torch.device], optional): Device for encoder.
            Defaults to None.

    Returns:
        str: Detected language.
    """
    model = model or whisper.load_model(model_name, device=device)

    mel = whisper.audio.log_mel_spectrogram(audio)
    segment = (
        whisper.audio.pad_or_trim(mel, whisper.audio.N_FRAMES)
        .to(model.device)
        .to(torch.float16)
    )
    _, probs = model.detect_language(segment)
    language = max(probs, key=probs.get)

    return language


def encoder(
    name: str = "tiny", device: Optional[torch.device | str] = None
) -> tuple[nn.Module, torch.device]:
    """Get Whisper encoder by model name.

    Args:
        name (str): Whisper model name.
            Defaults to "tiny".
        device (Optional[torch.device], optional): Device for encoder.
            Defaults to None.

    Returns:
        tuple[nn.Module, torch.device]: Whisper encoder and its device.
    """
    model = whisper.load_model(name, device=device)
    return model.encoder, model.device


def encode_audio(
    audio: Union[str, torch.Tensor],
    name: str = "tiny",
    device: Optional[torch.device | str] = None,
) -> list[torch.Tensor]:
    """Get Whisper encodings for consistent audio frames.

    Args:
        name (str): Whisper model name.
            Defaults to "tiny".
        device (Optional[torch.device], optional): Device for encoder and embedding.
            Defaults to None.

    Returns:
        torch.Tensor: Whisper encoding of given audio.
    """
    # Only 80 filters supported by Whisper.
    model_encoder, device = encoder(name, device=device)
    mel = log_mel_spectrogram(audio).to(device)

    embeddings: list[torch.Tensor] = []
    number_of_frames: int = mel.shape[-1]
    input_stride = whisper.audio.N_FRAMES

    for seek in range(0, number_of_frames, input_stride):
        segment = whisper.audio.pad_or_trim(mel[:, seek:], whisper.audio.N_FRAMES).to(
            device
        )
        embeddings.append(model_encoder(segment.unsqueeze(0)))

    return embeddings


def encode_with_alignment(
    audio: Union[str, torch.Tensor],
    text: str,
    name: str = "tiny",
    device: Optional[torch.device | str] = None,
) -> dict[str, torch.Tensor]:
    """Get Whisper encodings for consistent audio frames.

    Args:
        audio (Union[str, torch.Tensor]): Path to file, or audio tensor (16kHz).
        text (str): Transcript of audio passed.
        name (str): Whisper model name.
            Defaults to "tiny".
        device (Optional[torch.device], optional): Device for encoder and embedding.
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Map of text to Whisper encoding of given audio.
    """
