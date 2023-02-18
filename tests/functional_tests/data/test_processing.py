"""Test processing."""

import pytest

import whisper
import torch

from clsp.data.processing import slice_audio_and_text_by_token_windows


@pytest.mark.functional_test
def test_processing_slice_aligned_text_and_audio() -> None:
    """Test that we can slice Joe rogan sample into specified token windows with aligned audio."""
    text = (
        " on 60 Minutes, they interviewed her."
        " She's some new woman who works in the White House."
        " And they asked her about obesity."
        " She said the number one cause of obesity is genetics."
        " And it doesn't matter what you do,"
        " like you could be a person who has a perfect diet and exercises and sleeps right and you're still obese."
        " And the health experts went fucking nuts. Like that's not what the data shows."
        " The data shows that most people who are obese have obese parents and they come from an obese family,"
        " but they're all doing the wrong thing. It's not, there's not like a person in that family that's"
        " eating grass fed steak and running marathons and lifting weights and getting up at six in the morning"
        " and getting a cold plunge and doing all these different things, but it's still fat as fuck."
    )

    # text = " hello hello okay so this is a test of whatever that tortoise guy wrote when he was super drunk"
    audio = torch.tensor(whisper.audio.load_audio("data/joe.wav")).unsqueeze(0)
    tokenizer = whisper.tokenizer.get_tokenizer(True)

    result = slice_audio_and_text_by_token_windows(
        audio, text, tokenizer, token_window_size=20, model_name="tiny"
    )

    assert list(result) == [
        " on 60 Minutes, they interviewed her. She's some new woman who works in.",
        " in the White House. And they asked her about obesity. She said the number one cause of.",
        " of obesity is genetics. And it doesn't matter what you do, like you could be a person who.",
        " who has a perfect diet and exercises and sleeps right and you're still obese. And the health experts.",
        " experts went fucking nuts. Like that's not what the data shows. The data shows that.",
        " that most people who are obese have obese parents and they come from an obese family, but.",
        " but they're all doing the wrong thing. It's not, there's not like a person.",
        " person in that family that's eating grass fed steak and running marathons and lifting weights and.",
        " and getting up at six in the morning and getting a cold plunge and doing all these.",
    ]

    for audio_snippet in result.values():
        assert audio_snippet.nelement() > 0, "Something is fishy."
