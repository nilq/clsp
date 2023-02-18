"""Forced alignment."""
import string
import torch
import torchaudio
import torch.nn as nn
import numpy as np
from dtw import dtw
from scipy.ndimage import median_filter

import whisper
from whisper.tokenizer import Tokenizer, get_tokenizer

from typing import Optional


AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE

medfilt_width = 7
qk_scale = 1.0


def whisper_model(
    name: str = "tiny",
    device: Optional[torch.device | str] = None,
) -> tuple[nn.Module, torch.device]:
    """Get Whisper encoder by model name.

    Args:
        name (str): Whisper model name.
            Defaults to "tiny".
        device (Optional[torch.device], optional): Device for model.
            Defaults to None.

    Returns:
        tuple[nn.Module, torch.device]: Whisper model and its device.
    """
    model = whisper.load_model(name, device=device)
    return model, model.device


def split_tokens_on_unicode(
    tokens: torch.Tensor, tokenizer: Tokenizer
) -> tuple[list[str], list[list[int]]]:
    """Split tokens by unicode.

    Args:
        tokens (torch.Tensor): Tokens to split.
        tokenizer (Tokenizer): Tokenizer to use for decoding in split.

    Returns:
        tuple[list[str], list[int]]: Split tokens.
    """
    words: list[str] = []
    word_tokens: list[int] = []
    current_tokens: list[int] = []

    for token in tokens.tolist():
        current_tokens.append(token)
        decoded = tokenizer.decode_with_timestamps(current_tokens)

        if "\ufffd" not in decoded:
            words.append(decoded)
            word_tokens.append(current_tokens)
            current_tokens = []

    return words, word_tokens


def split_tokens_on_spaces(
    tokens: torch.Tensor, tokenizer: Tokenizer
) -> tuple[list[str], list[list[int]]]:
    """Split tokens on spaces.

    Args:
        tokens (torch.Tensor): Tokens to split on space.
        tokenizer (Tokenizer): Tokenizer used in decoding of tokens in split.

    Returns:
        tuple[list[str], list[int]]: Split tokens.
    """
    subwords, subword_tokens_list = split_tokens_on_unicode(tokens, tokenizer)
    words = []
    word_tokens = []

    for subword, subword_tokens in zip(subwords, subword_tokens_list):
        special = subword_tokens[0] >= tokenizer.eot
        with_space = subword.startswith(" ")
        punctuation = subword.strip() in string.punctuation

        if special or with_space or punctuation:
            words.append(subword)
            word_tokens.append(subword_tokens)
        else:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(subword_tokens)

    return words, word_tokens


def audio_text_forced_alignment(
    audio: torch.Tensor,
    text: str,
    model_name: str = "tiny",
    min_audio_duration: int = 1600,
    model: Optional[whisper.Whisper] = None,
    device: Optional[torch.device | str] = None,
) -> dict[str, tuple[float, float]]:
    """Get alignment mapping between audio and text.

    Args:
        audio (torch.Tensor): 16kHz audio tensor.
        text (str): Text to align.
        model_name (str, optional): Name of Whisper model to use for this.
            Defaults to "tiny".
        model (Optional[whisper.Whisper], optional): Existing model, if you have one.
            Defaults to None.
        device (Optional[torch.device], optional): Device for model.
            Defaults to None.

    Returns:
        dict[str, tuple[float, float]]: Words mapped to their interval in audio.
    """
    model = model.to(device) if model else whisper.load_model(model_name, device=device)
    device = model.device

    # Install hooks on the cross attention layers to receive attention weights.
    QKs = [None] * model.dims.n_text_layer  # Query-key matrices.

    for i, block in enumerate(model.decoder.blocks):
        # Read inside the whisper brain (thought police #1984)
        block.cross_attn.register_forward_hook(
            lambda _, _ins, outs, index=i: QKs.__setitem__(index, outs[-1])
        )

    duration = len(audio[-1])

    mel = whisper.audio.log_mel_spectrogram(whisper.audio.pad_or_trim(audio)).to(device)

    # Language detected for tokenizer.
    _, probs = model.detect_language(mel)

    if isinstance(probs, list):
        probs = probs[0]  # Bruh moment.

    language = max(probs, key=probs.get)
    tokenizer = get_tokenizer(model.is_multilingual, language=language)

    tokens = torch.tensor(
        [
            *tokenizer.sot_sequence,
            tokenizer.timestamp_begin,
        ]
        + tokenizer.encode(text)
        + [
            tokenizer.timestamp_begin + duration // AUDIO_SAMPLES_PER_TOKEN,
            tokenizer.eot,
        ]
    ).to(device)

    with torch.no_grad():
        logits = model(mel, tokens.unsqueeze(0))

    weights = torch.cat(QKs)  # layers * heads * tokens * frames
    weights = weights[:, :, :, : duration // AUDIO_SAMPLES_PER_TOKEN].cpu()
    weights = median_filter(weights, (1, 1, 1, medfilt_width))
    weights = torch.tensor(weights * qk_scale).softmax(dim=-1)

    w = weights / weights.norm(dim=-2, keepdim=True)
    matrix = w[-6:].mean(axis=(0, 1))

    if matrix.nelement() == 0:
        # No alignments and DTW will break after this.
        return dict()

    alignment = dtw(-matrix.double().numpy())

    jumps = np.pad(np.diff(alignment.index1s), (1, 0), constant_values=1).astype(bool)
    jump_times = alignment.index2s[jumps] * AUDIO_TIME_PER_TOKEN
    words, word_tokens = split_tokens_on_spaces(tokens, tokenizer)

    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
    begin_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]

    alignments: dict[str, tuple[float, float]] = {
        word: (begin, end)
        for word, begin, end in zip(words[:-1], begin_times, end_times)
        if not word.startswith("<|") and word.strip() not in ".,!?、。"
    }

    return alignments


def trim_audio_to_text(
    audio: torch.Tensor,
    text: str,
    model_name: str = "tiny",
    model: Optional[whisper.Whisper] = None,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Trims audio to align with text.

    Notes:
        This is useful when splitting text tokens to fit model and you then want to get
        the corresponding audio embeddings for those tokens. Super nice if I do say so myself.

    Args:
        audio (torch.Tensor): 16kHz audio tensor.
        text (str): Text to align.
        model_name (str, optional): Name of Whisper model to use for this.
            Defaults to "tiny".
        model (Optional[whisper.Whisper], optional): Existing model, if you have one.
            Defaults to None.
        device (Optional[torch.device], optional): Device for model.
            Defaults to None.

    Raises:
        ValueError: When audio is not 2D.

    Returns:
        Optional[torch.Tensor]: 16kHz audio tensor, cut to fit text, None if empty audio.
    """
    if len(audio.shape) != 2:
        raise ValueError(
            f"I know this is lazy, but we only deal with 2D audio. Got {len(audio.shape)}"
        )

    intervals = list(
        audio_text_forced_alignment(
            audio, text, model_name, model=model, device=device
        ).values()
    )

    if intervals:
        start_sec: float = intervals[0][0]
        end_sec: float = intervals[-1][1]

        return audio[
            :,
            # Do some conservative rounding in conversion here.
            max(0, round(start_sec * 16000 - 0.5)) : min(
                audio.shape[1] - 1, round(end_sec * 16000 + 0.5)
            ),
        ]
    else:
        return None
