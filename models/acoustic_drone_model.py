"""End-to-end acoustic drone detector."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.attention import FeatureAttention
from models.classifier import DroneClassifier
from models.feature_encoder import FeatureEncoder
from models.fusion import FeatureFusion


class AcousticDroneModel(nn.Module):
    """
    Multi-feature acoustic binary drone detector.

    Inputs:
        mfcc     -> (B, 120, T)
        mel      -> (B, 128, T)
        spectral -> (B, 12, T)
        chroma   -> (B, 12, T)
        zcr      -> (B, 1, T)
        energy   -> (B, 1, T)

    Outputs:
        logits            -> (B, 2)
        attention_weights -> (B, 6)

    Class index:
        0 = background
        1 = drone
    """

    FEATURE_NAMES = (
        "mfcc",
        "mel",
        "spectral",
        "chroma",
        "zcr",
        "energy",
    )

    INPUT_CHANNELS = {
        "mfcc": 120,
        "mel": 128,
        "spectral": 12,
        "chroma": 12,
        "zcr": 1,
        "energy": 1,
    }

    def __init__(
        self,
        embedding_dim: int = 128,
        fused_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        if num_classes != 2:
            raise ValueError(
                "AcousticDroneModel is a binary detector."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        if fused_dim <= 0:
            raise ValueError(
                "fused_dim must be greater than zero."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        # One encoder per acoustic feature type.
        self.mfcc_encoder = FeatureEncoder(
            self.INPUT_CHANNELS["mfcc"],
            embedding_dim,
            dropout,
        )

        self.mel_encoder = FeatureEncoder(
            self.INPUT_CHANNELS["mel"],
            embedding_dim,
            dropout,
        )

        self.spectral_encoder = FeatureEncoder(
            self.INPUT_CHANNELS["spectral"],
            embedding_dim,
            dropout,
        )

        self.chroma_encoder = FeatureEncoder(
            self.INPUT_CHANNELS["chroma"],
            embedding_dim,
            dropout,
        )

        self.zcr_encoder = FeatureEncoder(
            self.INPUT_CHANNELS["zcr"],
            embedding_dim,
            dropout,
        )

        self.energy_encoder = FeatureEncoder(
            self.INPUT_CHANNELS["energy"],
            embedding_dim,
            dropout,
        )

        # Learns which acoustic feature types are most useful
        # for the current sample.
        self.attention = FeatureAttention(
            embedding_dim=embedding_dim,
            hidden_dim=64,
        )

        # Combines the six attended feature representations.
        self.fusion = FeatureFusion(
            num_features=len(self.FEATURE_NAMES),
            embedding_dim=embedding_dim,
            hidden_dim=384,
            fused_dim=fused_dim,
            dropout=dropout,
        )

        # Final binary classifier.
        self.classifier = DroneClassifier(
            fused_dim=fused_dim,
            hidden_dim=128,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.embedding_dim = embedding_dim
        self.fused_dim = fused_dim
        self.num_classes = num_classes

    def _validate_inputs(
        self,
        batch: dict[str, torch.Tensor],
    ) -> None:
        """Validate feature presence and channel dimensions."""

        missing = [
            name
            for name in self.FEATURE_NAMES
            if name not in batch
        ]

        if missing:
            raise KeyError(
                "Missing acoustic features: "
                + ", ".join(missing)
            )

        for name in self.FEATURE_NAMES:
            value = batch[name]

            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Feature '{name}' must be a torch.Tensor."
                )

            if value.ndim != 3:
                raise ValueError(
                    f"Feature '{name}' must have shape "
                    f"(B, C, T), got {tuple(value.shape)}."
                )

            expected_channels = self.INPUT_CHANNELS[name]

            if value.shape[1] != expected_channels:
                raise ValueError(
                    f"Feature '{name}' expected "
                    f"{expected_channels} channels, "
                    f"got {value.shape[1]}."
                )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(batch)

        embeddings = [
            self.mfcc_encoder(batch["mfcc"]),
            self.mel_encoder(batch["mel"]),
            self.spectral_encoder(batch["spectral"]),
            self.chroma_encoder(batch["chroma"]),
            self.zcr_encoder(batch["zcr"]),
            self.energy_encoder(batch["energy"]),
        ]

        # (B, 6, embedding_dim)
        feature_embeddings = torch.stack(
            embeddings,
            dim=1,
        )

        # Attention weights:
        # (B, 6)
        weighted, attention_weights = self.attention(
            feature_embeddings
        )

        # Fused representation:
        # (B, fused_dim)
        fused = self.fusion(weighted)

        # Binary logits:
        # (B, 2)
        logits = self.classifier(fused)

        return logits, attention_weights


# Backward compatibility:
# Existing code importing AcousticDroneNet will continue to work.
AcousticDroneNet = AcousticDroneModel


__all__ = [
    "AcousticDroneModel",
    "AcousticDroneNet",
]