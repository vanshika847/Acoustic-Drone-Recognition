"""Complete acoustic drone detection network."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.attention import FeatureAttention
from models.classifier import DroneClassifier
from models.feature_encoder import FeatureEncoder
from models.fusion import FeatureFusion


class AcousticDroneNet(nn.Module):
    """
    End-to-end network for binary acoustic drone detection.

    Inputs
    ------
    Dictionary containing:

        mfcc
        mel
        spectral
        chroma
        zcr
        energy

    Outputs
    -------
    logits:
        (batch_size, 2)

    attention_weights:
        (batch_size, 6)
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        fused_dim: int = 256,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.mfcc_encoder = FeatureEncoder(
            input_channels=120,
            embedding_dim=embedding_dim,
        )

        self.mel_encoder = FeatureEncoder(
            input_channels=128,
            embedding_dim=embedding_dim,
        )

        self.spectral_encoder = FeatureEncoder(
            input_channels=12,
            embedding_dim=embedding_dim,
        )

        self.chroma_encoder = FeatureEncoder(
            input_channels=12,
            embedding_dim=embedding_dim,
        )

        self.zcr_encoder = FeatureEncoder(
            input_channels=1,
            embedding_dim=embedding_dim,
        )

        self.energy_encoder = FeatureEncoder(
            input_channels=1,
            embedding_dim=embedding_dim,
        )

        self.attention = FeatureAttention(
            embedding_dim=embedding_dim,
        )

        self.fusion = FeatureFusion(
            num_features=6,
            embedding_dim=embedding_dim,
            fused_dim=fused_dim,
        )

        self.classifier = DroneClassifier(
            fused_dim=fused_dim,
            num_classes=num_classes,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        embeddings = [
            self.mfcc_encoder(batch["mfcc"]),

            self.mel_encoder(batch["mel"]),

            self.spectral_encoder(batch["spectral"]),

            self.chroma_encoder(batch["chroma"]),

            self.zcr_encoder(batch["zcr"]),

            self.energy_encoder(batch["energy"]),
        ]

        features = torch.stack(
            embeddings,
            dim=1,
        )

        weighted_features, attention_weights = (
            self.attention(features)
        )

        fused = self.fusion(weighted_features)

        logits = self.classifier(fused)

        return logits, attention_weights