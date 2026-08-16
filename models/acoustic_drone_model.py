"""Complete acoustic drone recognition model."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.attention import FeatureAttention
from models.classifier import DroneClassifier
from models.feature_encoder import FeatureEncoder
from models.fusion import FeatureFusion


class AcousticDroneModel(nn.Module):
    """
    End-to-end acoustic drone recognition model.

    The model processes six acoustic feature types independently,
    converts each into a fixed-size embedding, applies feature-level
    attention, fuses the weighted embeddings, and performs classification.

    Inputs
    ------
    batch:
        Dictionary containing:

        mfcc:
            (batch, 120, time)

        mel:
            (batch, 128, time)

        spectral:
            (batch, 12, time)

        chroma:
            (batch, 12, time)

        zcr:
            (batch, 1, time)

        energy:
            (batch, 1, time)

    Outputs
    -------
    logits:
        (batch, num_classes)

        Raw classification logits.

    attention_weights:
        (batch, 6)

        Learned importance of each acoustic feature.

        Order:
            0 -> MFCC
            1 -> Mel Spectrogram
            2 -> Spectral
            3 -> Chroma
            4 -> ZCR
            5 -> Energy
    """

    FEATURE_NAMES = (
        "mfcc",
        "mel",
        "spectral",
        "chroma",
        "zcr",
        "energy",
    )

    def __init__(
        self,
        embedding_dim: int = 128,
        fused_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError(
                f"num_classes must be at least 2, got {num_classes}."
            )

        # ---------------------------------------------------------
        # Individual feature encoders
        # ---------------------------------------------------------

        self.mfcc_encoder = FeatureEncoder(
            input_channels=120,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        self.mel_encoder = FeatureEncoder(
            input_channels=128,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        self.spectral_encoder = FeatureEncoder(
            input_channels=12,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        self.chroma_encoder = FeatureEncoder(
            input_channels=12,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        self.zcr_encoder = FeatureEncoder(
            input_channels=1,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        self.energy_encoder = FeatureEncoder(
            input_channels=1,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        # ---------------------------------------------------------
        # Feature attention
        # ---------------------------------------------------------

        self.attention = FeatureAttention(
            embedding_dim=embedding_dim,
        )

        # ---------------------------------------------------------
        # Feature fusion
        # ---------------------------------------------------------

        self.fusion = FeatureFusion(
            num_features=len(self.FEATURE_NAMES),
            embedding_dim=embedding_dim,
            fused_dim=fused_dim,
            dropout=dropout,
        )

        # ---------------------------------------------------------
        # Classification head
        # ---------------------------------------------------------

        self.classifier = DroneClassifier(
            fused_dim=fused_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.fused_dim = fused_dim

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # ---------------------------------------------------------
        # Check that all required features are present
        # ---------------------------------------------------------

        missing_features = [
            name
            for name in self.FEATURE_NAMES
            if name not in batch
        ]

        if missing_features:
            raise KeyError(
                "Missing required acoustic features: "
                + ", ".join(missing_features)
            )

        # ---------------------------------------------------------
        # Encode each acoustic feature
        # ---------------------------------------------------------

        embeddings = [
            self.mfcc_encoder(batch["mfcc"]),
            self.mel_encoder(batch["mel"]),
            self.spectral_encoder(batch["spectral"]),
            self.chroma_encoder(batch["chroma"]),
            self.zcr_encoder(batch["zcr"]),
            self.energy_encoder(batch["energy"]),
        ]

        # (batch, 6, embedding_dim)
        feature_embeddings = torch.stack(
            embeddings,
            dim=1,
        )

        # ---------------------------------------------------------
        # Feature-level attention
        # ---------------------------------------------------------

        weighted_features, attention_weights = self.attention(
            feature_embeddings
        )

        # ---------------------------------------------------------
        # Feature fusion
        # ---------------------------------------------------------

        fused_features = self.fusion(
            weighted_features
        )

        # (batch, num_classes)
        logits = self.classifier(
            fused_features
        )

        return logits, attention_weights