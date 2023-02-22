"""Module containing processing stuff."""

import torch
import torchaudio
import torchaudio.functional as F
import whisper

from clsp.utils.alignment import (
    trim_audio_to_text,
    split_tokens_on_spaces,
    audio_text_forced_alignment,
)

from clsp.utils.speech import encode_audio
from clsp.utils.text import encode_to_tensor

from typing import Optional, Any


# This was a sleepy design, hopefully it's what's needed.
def slice_audio_and_text_by_token_windows(
    audio: torch.Tensor,
    text: str,
    tokenizer: Any,
    token_window_size: int,
    words_per_second: int = 1,
    alignment_window_overlap: int = 16000 // 2,
    model_name: str = "tiny",
    model: Optional[whisper.Whisper] = None,
    device: Optional[torch.device] = None,
) -> dict[str, torch.Tensor]:
    """Slice potentially long audio and transcript into (aligned) bite-sized chunks.

    Notes:
        This is useful when you need a specific token count, and let's say; the Whisper enmbeddings
        corresponding to those exact tokens. Pretty neat if I do say so myself.

    Args:
        audio (torch.Tensor): 16kHz audio tensor.
        text (str): Transcript of audio.
        tokenizer (Any): Tokenizer (probably tiktoken) used for tokenisation.
        token_window_size (int): Length of token window to split into.
        words_per_second (int, optional): Assumed words per second, used to estimate alignment window.
            Alignment breaks if you overshoot way too much.
            Defaults to 1.
        alignment_window_overlap (int, optional): Aligment window overlap with last token window's end.
            Overlap makes it likely to pick up aligned pair.
            Defaults to `16000 // 2` i.e. half a second.
        model_name (str, optional): Name of Whisper model to use for this.
            Defaults to "tiny".
        model (Optional[Whisper], optional): Existing model, if you have one.
            Defaults to None.
        device (Optional[torch.device], optional): Device for model.
            Defaults to None.

    Raises:
        ValueError: When audio is not 2D.

    Returns:
        dict[str, torch.Tensor]: Chronologicial mapping between texts and their aligned audio snippets.
    """
    if len(audio.shape) != 2:
        raise ValueError(
            f"I know this is lazy, but we only deal with 2D audio. Got {len(audio.shape)}"
        )

    model = model or whisper.load_model(model_name, device=device)

    # This is what we will hand back. Dicts are ordered today. :)
    slice_map: dict[str, torch.Tensor] = {}

    # Let's see ...
    tokens = torch.tensor(tokenizer.encode(text))
    all_words, all_tokens = split_tokens_on_spaces(tokens, tokenizer)
    token_buffer: list[int] = []

    # Tracking where we last cut in text and audio respectively.
    word_cursor_last: int = 0  # Unit here is number of words.
    audio_cursor_last: int = 0  # Unit here is number of samples.

    # Literal rocket science.
    forward_view: int = words_per_second * token_window_size

    for where_we_are, token in enumerate(all_tokens):
        token_buffer += token

        if len(token_buffer) > token_window_size:
            # We remove the extra tokens, carrying them to next circuit.
            extra_token_length: int = len(token)
            overflow_token = token_buffer[-extra_token_length:]
            del token_buffer[-extra_token_length:]

            # `-1` because of that extra word.
            window_words: list[str] = all_words[word_cursor_last : where_we_are - 1]
            window_sentence: str = (str.join("", window_words) + ".").replace("..", ".")

            # We align with everything but what we already have track of.
            window_audio: Optional[torch.Tensor] = trim_audio_to_text(
                audio=audio[
                    :,
                    max(
                        0, audio_cursor_last - alignment_window_overlap
                    ) : audio_cursor_last
                    + forward_view
                    + alignment_window_overlap,
                ],
                text=window_sentence,
                model_name=model_name,
                device=device,
            )

            if window_audio is None:
                # We are done.
                # No more audio to be trimmed.
                break

            slice_map[window_sentence] = window_audio

            # Before we forget ...
            word_cursor_last += len(window_words) - 1  # `-1` for extra word. :)
            audio_cursor_last += window_audio.shape[1]

            # Remember that token we sliced off? This is him now.
            token_buffer = [*overflow_token]

    return slice_map


def text_audio_embedding_windows(
    audio: torch.Tensor,
    text: str,
    tokenizer: Any,
    token_window_size: int,
    words_per_second: int = 1,
    alignment_window_overlap: int = 16000 // 2,
    model_name: str = "tiny",
    model: Optional[whisper.Whisper] = None,
    device: Optional[torch.device] = None, 
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """From arbitrarily long audio and transcript, you get specified token-chunks with their aligned audio embedding.

    Args:
        audio (torch.Tensor): 16kHz audio tensor.
        text (str): Transcript of audio.
        tokenizer (Any): Tokenizer (probably tiktoken) used for tokenisation.
        token_window_size (int): Length of token window to split into.
        words_per_second (int, optional): Assumed words per second, used to estimate alignment window.
            Alignment breaks if you overshoot way too much.
            Defaults to 1.
        alignment_window_overlap (int, optional): Aligment window overlap with last token window's end.
            Overlap makes it likely to pick up aligned pair.
            Defaults to `16000 // 2` i.e. half a second.
        model_name (str, optional): Name of Whisper model to use for this.
            Defaults to "tiny".
        model (Optional[Whisper], optional): Existing model, if you have one.
            Defaults to None.
        device (Optional[torch.device], optional): Device for model.
            Defaults to None.

    Raises:
        ValueError: When audio is not 2D.

    Returns:
        list[tuple[torch.Tensor, torch.Tensor]]: Chronologicial mapping between
            tokens (see utils.text for text-encoder details) and their aligned audio embeddings.
    """
    # Encode audio returns a list of one element when we already have
    # audio in Whisper-appropriate sized slices. :)
    return [
        (encode_to_tensor(text, device=device), encode_audio(aligned_audio)[0])
        for text, aligned_audio in slice_audio_and_text_by_token_windows(
            audio,
            text,
            tokenizer,
            token_window_size,
            words_per_second,
            alignment_window_overlap,
            model_name,
            model,
            device
        ).items()
    ]
