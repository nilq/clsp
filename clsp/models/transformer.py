"""Carefully assembled transformer architecture."""

from typing import Any, Optional

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Feed forward section.


class GeGLU(nn.Module):
    """GeGLU activation - GELU and GLU love child. (https://arxiv.org/abs/2002.05202v1)."""

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    """Feed-forward GeGLU black."""

    def __init__(self, size: int, dropout: float = 0, multiplier: float = 4) -> None:
        """Configure FF.

        Args:
            size (int): Input dimension size.
            dropout (float, optional): Dropout between two layers.
            multiplier (float, optional): Upscale of input size.
        """
        super().__init__()

        self.linear_geglu = nn.Sequential(
            # Doubling for chunk in GeGLU.
            nn.Linear(size, size * multiplier * 2),
            GeGLU(),
            nn.Dropout(dropout),
            nn.Linear(size * multiplier, size),
        )

    def forward(self, x: Any):
        return self.linear_geglu(x)


# Multi-head attention section.


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        *,
        size: int,
        num_heads: int,
        head_size: int,
        dropout: float = 0,
    ) -> None:
        """Configure non-causal multi-head attention layer.

        Args:
            size (int): Input dimension size.
            num_heads (int): Number of attention heads.
            head_size (int): Size of heads (all of them).
            dropout (float, optional): Dropout probability.
                Defaults to 0.
        """
        super().__init__()

        self.num_heads: int = num_heads

        inner_size = num_heads * head_size

        # Define weights in a single chained layer. Easily unrollable.
        self.weights_qkv = nn.Linear(size, inner_size * 3, bias=False)
        # Gaussian weight normalisation.
        nn.init.normal_(
            self.weights_qkv.weight, mean=0, std=np.sqrt(2.0 / (size + head_size))
        )

        self.linear = nn.Linear(inner_size, size)
        # https://www.deeplearning.ai/ai-notes/initialization/index.html#IV
        nn.init.xavier_normal_(self.linear.weight)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(size)

    def _extract_qkv(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts chained q, k, v tensors from single tensor."""
        qkv = self.weights_qkv(x).chunk(3, dim=-1)
        queries, keys, values = map(
            lambda chunk: einops.rearrange(
                chunk, "b l (head k) -> b head l k", head=self.num_heads
            ),
            qkv,
        )
        return queries, keys, values

    def forward(self, x, mask=None):
        # Extract queries, keys and values.
        q, k, v = self._extract_qkv(x)
        residual = q  #

        # Compute query "q·k" for attention.
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) / np.sqrt(q.shape[-1])

        if mask is not None:
            _mask = einops.rearrange(mask, "b j -> b () () j")
            dots.masked_fill_(~_mask, -np.inf)

        attn = torch.softmax(dots, dim=-1)

        output = torch.einsum("hblt, hbtv -> hblv", [attn, v])
        output = einops.rearrange(output, "b head l v -> b l (head v)")

        output = self.dropout(self.linear(output))
        output = self.layer_norm(output)

        return output


# Going deeper with layer-scale.


class LayerScale(nn.Module):
    """Layer scaling wrapper. (https://arxiv.org/abs/2103.17239)"""

    def __init__(self, size: int, depth: int, layer: nn.Module) -> None:
        """Configure layer scaling.

        Args:
            size (int): Input size of layer.
            depth (int): Depth at which layer is located.
            layer (nn.Module): Layer to scale.
        """
        super().__init__()

        if depth <= 18:
            eps = 1e-1  # Scale a little for depth <= 18.
        elif 18 < depth <= 24:
            eps = 1e-5  # Scale more for these.
        else:
            eps = 1e-6  # Scale most for depth > 24.

        self.layer = layer
        self.scale = nn.Parameter(torch.zeros(1, 1, size).fill_(eps))

    def forward(self, x, *args, **kwargs):
        """Trickle-down forward pass.

        Args:
            x (Any): Whatever input.
            *args: Positional arguments for layer.
            **kwargs: Key-word arguments for layer.

        Returns:
            Any: Layer output. Whatever.
        """
        return self.layer(x, *args, **kwargs) * self.scale


# This is just tedious plumbing.


class PreNorm(nn.Module):
    """Pre-normalising wrapper."""

    def __init__(self, size, layer) -> None:
        """Initialise pre-normalisation wrapper.

        Args:
            size (int): Input dimension size.
            layer (nn.Module): Layer to pre-normalise.
        """
        super().__init__()

        self.norm = nn.LayerNorm(size)
        self.layer = layer

    def forward(self, x: Any, *args, **kwargs) -> Any:
        """Trickle-down forward pass.

        Args:
            x (Any): Input.
            *args: Positional arguments for layer.
            **kwargs: Key-word arguments for layer.

        Returns:
            Any: Whatever the layer's forward pass returns.
        """
        x = self.norm(x)
        return self.layer(x, *args, **kwargs)


# The actual transformer.


class TransformerLayer(nn.Module):
    """Gated linear layer with multi-head attention."""

    def __init__(
        self,
        *,
        size: int,
        depth: int,
        ff_dropout: float = 0,
        ff_multiplier: float = 4,
        attention_heads: int = 8,
        attention_head_size: int = 64,
        attention_dropout: float = 0,
    ) -> None:
        """Transformer layer."""
        super().__init__()

        # Multi-head attention with pre-normalisation.
        attention = MultiHeadAttention(
            size=size,
            num_heads=attention_heads,
            head_size=attention_head_size,
            dropout=attention_dropout,
        )
        norm_attention = PreNorm(size, attention)
        # Finally. Attention.
        self.attention = LayerScale(size, depth, norm_attention)

        # GeGLU feed-forward block with pre-normalisation.
        ff = FeedForward(size=size, dropout=ff_dropout, multiplier=ff_multiplier)
        norm_ff = PreNorm(size, ff)
        # Finally. Again.
        self.ff = LayerScale(size, depth, norm_ff)

    def forward(self, x: torch.Tensor, mask=None):
        """Forward pass.

        Args:
            x (torch.Tensor): Whatever.
            mask (Optional[torch.Tensor]): Optional mask, this one is passed to MHA layer.
                Defaults to None.

        Returns:
            torch.Tensor: Magic number sauce.
        """
        x = self.attention(x, mask)
        x = self.ff(x)
        return x


class Transformer(nn.Module):
    """Just a transformer."""

    def __init__(
        self,
        *,
        size: int,
        depth: int,
        ff_dropout: float = 0,
        ff_multiplier: float = 4,
        attention_heads: int = 8,
        attention_head_size: int = 64,
        attention_dropout: float = 0,
    ) -> None:
        """Configure transformer.

        Args:
            size (int): Input dimension size.
            depth (int): Number of layers in transformer.
            ff_dropout (float, optional): Dropout on each feed forward layer.
                Defaults to 0.
            ff_multiplier (float, optional): Feed forward hidden size multiplier.
                Defaults to 4.
            attention_heads (int, optional): Number of attention heads per layer.
                Defaults to 8.
            attention_head_szie (int, optional): Size of attention heads.
                Defaults to 64.
            attention_dropout (float, optional): Dropout on attention layers.
                Defaults to 0.
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    size=size,
                    depth=index,
                    ff_dropout=ff_dropout,
                    ff_multiplier=ff_multiplier,
                    attention_heads=attention_heads,
                    attention_head_size=attention_head_size,
                    attention_dropout=attention_dropout,
                )
                for index in range(depth)
            ]
        )

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x
