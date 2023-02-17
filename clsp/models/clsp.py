"""Contrastive Language Speech Pre-training"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from clsp.models.transformer import Transformer


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def masked_mean(t, mask, dim: int = 1):
    t = t.masked_fill(~mask[:, :, None], 0.0)
    return t.sum(dim=1) / mask.sum(dim=1)[..., None]


class CLSP(nn.Module):
    """Basically CLIP, but for speech."""

    def __init__(
        self,
        *,
        # Dimension sizes.
        size_text: int = 512,
        size_speech: int = 512,
        size_latent: int = 512,
        # Text related.
        num_text_tokens: int = 256,
        text_encoding_depth: int = 6,
        text_sequence_len: int = 120,
        text_heads: int = 8,
        # Speech related.
        num_speech_tokens: int = 8192,  # Fun fact: this is 2^13.
        speech_encoding_depth: int = 6,
        speech_heads: int = 8,
        speech_sequence_len: int = 250,
        # Masking.
        text_mask_percentage: float = 0,
        speech_mask_percentage: float = 0,
        # Device.
        device: Optional[str] = None
    ) -> None:
        """Initialise CLSP model.

        Args:
            size_text (int): Size of text dimension.
        """
        super().__init__()

        # Text.
        self.text_positional_embedding = nn.Embedding(text_sequence_len, size_text)
        self.text_embedding = nn.Embedding(num_text_tokens, size_text)
        self.to_text_latent = nn.Linear(size_text, size_latent)
        self.text_transformer = Transformer(
            size=size_text,
            depth=text_encoding_depth,
            attention_heads=text_heads,
        )

        # Speech.
        self.speech_positional_embedding = nn.Embedding(
            speech_sequence_len, size_speech
        )
        self.speech_embedding = nn.Embedding(num_speech_tokens, size_speech)
        self.to_speech_latent = nn.Linear(size_speech, size_latent)
        self.speech_transformer = Transformer(
            size=size_speech,
            depth=speech_encoding_depth,
            attention_heads=speech_heads,
        )

        # Learnable temperature.
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Masking.
        self.text_mask_percentage = text_mask_percentage
        self.speech_mask_percentage = speech_mask_percentage

        self.device = device or DEFAULT_DEVICE

    def forward(
        self, text: torch.Tensor, speech_tokens: torch.Tensor, return_loss: bool = False
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            text (torch.Tensor): Text tensor.
            speech_tokens (torch.Tensor): Speech token tensor.
            return_loss (bool, optional): Whether to return loss, otherwise CLSP similarity.

        Returns:
            torch.Tensor: Either loss or CLSP similarity.
        """
        # Compute either uniform or boolian mask.
        if self.training:
            text_mask = torch.rand_like(text.float()) > self.text_mask_percentage
            speech_mask = (
                torch.rand_like(speech_tokens.float()) > self.speech_mask_percentage
            )
        else:
            text_mask = torch.ones_like(text.float()).bool()
            speech_mask = torch.ones_like(speech_tokens.float()).bool()

        # Compute text embeddings.
        text_embedding = self.text_embedding(text)
        text_embedding += self.text_positional_embedding(
            torch.arange(text.shape[1], device=self.device)
        )

        speech_embedding = self.speech_embedding(speech_tokens)
        speech_embedding += self.speech_positional_embedding(
            torch.arange(speech_embedding.shape[1], device=self.device)
        )

        # Use transformer to get encodings.
        encoded_text = self.text_transformer(text_embedding, mask=text_mask)
        encoded_speech = self.speech_transformer(speech_embedding, mask=speech_mask)

        # Compute latents.
        text_latents = self.to_text_latent(masked_mean(encoded_text, text_mask, dim=1))
        speech_latents = self.to_speech_latent(
            masked_mean(encoded_speech, speech_mask, dim=1)
        )

        # Normalise latents.
        text_latents = F.normalize(text_latents, dim=-1)
        speech_latents = F.normalize(speech_latents, dim=-1)

        temperature = self.temperature.exp()

        if not return_loss:
            similarity = (
                torch.einsum("n d, n d -> n", text_latents, speech_latents)
                * temperature
            )
            return similarity
        else:
            similarity = (
                torch.einsum("i d, j d -> i j", text_latents, speech_latents)
                * temperature
            )
            labels = torch.arange(text.shape[0], device=self.device)

            loss = (
                F.cross_entropy(similarity, labels)
                + F.cross_entropy(similarity.t(), labels)
            ) / 2
            return loss
