"""Temporal CNN encoder for one acoustic feature family."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualTemporalBlock(nn.Module):
    """Residual 1-D convolution block with optional downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()

        padding = dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(
            num_groups=min(8, out_channels),
            num_channels=out_channels,
        )
        self.act = nn.GELU()

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=min(8, out_channels),
            num_channels=out_channels,
        )
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(
                    num_groups=min(8, out_channels),
                    num_channels=out_channels,
                ),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout(x)

        return self.act(x + residual)


class FeatureEncoder(nn.Module):
    """
    Encode one feature matrix into a fixed-size embedding.

    Input:
        (batch, feature_channels, time)

    Output:
        (batch, embedding_dim)

    Compared with the previous encoder, this keeps temporal structure longer,
    uses dilated convolutions, and learns an attention-weighted temporal pool
    instead of immediately collapsing the sequence with average pooling.
    GroupNorm is used instead of BatchNorm so small batches and batch-size-1
    inference are stable.
    """

    def __init__(
        self,
        input_channels: int,
        embedding_dim: int = 128,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        if input_channels <= 0:
            raise ValueError("input_channels must be > 0.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0.")

        self.input_channels = input_channels
        self.embedding_dim = embedding_dim

        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                64,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            ResidualTemporalBlock(
                64,
                96,
                stride=2,
                dilation=1,
                dropout=dropout,
            ),
            ResidualTemporalBlock(
                96,
                128,
                stride=2,
                dilation=2,
                dropout=dropout,
            ),
            ResidualTemporalBlock(
                128,
                160,
                stride=1,
                dilation=4,
                dropout=dropout,
            ),
            ResidualTemporalBlock(
                160,
                192,
                stride=1,
                dilation=8,
                dropout=dropout,
            ),
        )

        self.temporal_score = nn.Sequential(
            nn.Conv1d(192, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(64, 1, kernel_size=1),
        )

        self.projection = nn.Sequential(
            nn.Linear(192 * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape (batch, channels, time), "
                f"got {tuple(x.shape)}."
            )

        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, "
                f"got {x.shape[1]}."
            )

        if x.shape[2] < 8:
            raise ValueError(
                "Input must contain at least 8 time frames."
            )

        if not torch.isfinite(x).all():
            raise ValueError("Feature tensor contains NaN or Inf.")

        x = self.stem(x)
        x = self.blocks(x)

        # Learned temporal pooling.
        scores = self.temporal_score(x).squeeze(1)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.sum(
            x * weights.unsqueeze(1),
            dim=-1,
        )

        # Complementary statistics preserve global acoustic information.
        mean_pool = x.mean(dim=-1)
        max_pool = x.amax(dim=-1)

        pooled = torch.cat(
            [attended, mean_pool, max_pool],
            dim=1,
        )

        return self.projection(pooled)


__all__ = [
    "FeatureEncoder",
    "ResidualTemporalBlock",
]
