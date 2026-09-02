from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.data_loader import create_dataloader
from models.network import AcousticDroneNet


MANIFEST = PROJECT_ROOT / "outputs/features/train_shard_manifest.csv"

loader = create_dataloader(
    MANIFEST,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    validate_features=True,
)

batch = next(iter(loader))

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("ACOUSTIC DRONE ONE-BATCH TRAINING SMOKE TEST")
print("=" * 60)

print(f"Device: {device}")

model = AcousticDroneNet().to(device)
model.train()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)

criterion = torch.nn.CrossEntropyLoss()

features = {
    name: batch[name].to(device)
    for name in (
        "mfcc",
        "mel",
        "spectral",
        "chroma",
        "zcr",
        "energy",
    )
}

labels = batch["label"].to(device)

optimizer.zero_grad(set_to_none=True)

logits, attention_weights = model(features)

print(f"Output shape: {logits.shape}")
print(f"Attention shape: {attention_weights.shape}")
print(f"Labels shape: {labels.shape}")

loss = criterion(logits, labels)

print(f"Loss: {loss.item():.6f}")

loss.backward()

gradient_count = 0

for parameter in model.parameters():
    if parameter.grad is not None:
        gradient_count += 1

print(f"Parameters with gradients: {gradient_count}")

optimizer.step()

print()
print("TRAINING STEP PASSED")