"""Text encoding stuff."""

import torch
import tiktoken

from functools import cache
from typing import Optional


encoding = tiktoken.get_encoding("cl100k_base")


def encode_to_tensor(
    text: str, device: Optional[torch.device | str] = None
) -> torch.IntTensor:
    """Encode text to int tensor.

    Args:
        text (str): Text to encode.
        device (Optional[torch.device | str], optional): Device for tensor.
            Defaults to None.

    Returns:
        torch.IntTensor: Encoded text.
    """
    return torch.IntTensor(encoding.encode(text), device=device).unsqueeze(0)
