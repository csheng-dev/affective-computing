#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a sequence classifier by re-using the encoder part of the pretrained
AutoencoderModel. The script keeps the existing codebase untouched and
consumes classification samples derived from the raw datasets located in
`data/7 - WESAD`, `data/8 - CASE_full`, `data/9 - K-EmoCon`, and
`data/10 - PhyMER`.

Expected workflow:
1. Create a manifest CSV (or TSV) that lists the segments you derived from the
   above datasets along with their class labels. Required columns:
      - `path`: relative or absolute path to a tensor/array file containing a
        segment shaped [seq_len, channels]. The path must lie inside one of the
        four dataset folders mentioned above.
      - `label`: the string label (e.g. emotion category).
2. Run this script with the manifest, the pretrained autoencoder checkpoint
   (from `run_pretrain_concatAll`), and the model hyper-parameters that were
   used to train the autoencoder.

Example:
    python classification/train_classifier.py \
        --manifest classification/manifests/case_wesad.csv \
        --autoencoder_ckpt pretrain/ckpts/pretrain/exp/fold_0/ckpt/best.pt \
        --tcn_channels 16 32 \
        --transformer_dim 64 \
        --transformer_heads 4 \
        --transformer_layers 2 \
        --lstm_hidden 32 \
        --lstm_layers 1 \
        --num_classes 4 \
        --freeze_encoder
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Allow importing project modules without modifying existing files.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.append(str(PROJECT_ROOT))

from models.full_model import AutoencoderModel  # noqa: E402

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

ALLOWED_DATASETS = {
    "7": PROJECT_ROOT / "data" / "7 - WESAD",
    "8": PROJECT_ROOT / "data" / "8 - CASE_full",
    "9": PROJECT_ROOT / "data" / "9 - K-EmoCon",
    "10": PROJECT_ROOT / "data" / "10 - PhyMER-20250109T224509Z-001",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_sample_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.exists():
        raise FileNotFoundError(f"Sample path does not exist: {candidate}")

    if not any(str(candidate).startswith(str(root.resolve())) for root in ALLOWED_DATASETS.values()):
        allowed = ", ".join(str(p) for p in ALLOWED_DATASETS.values())
        raise ValueError(
            f"Sample path {candidate} is outside of the allowed dataset roots:\n{allowed}\n"
            "Please make sure the manifest only references segments extracted from datasets 7-10."
        )
    return candidate


def load_sequence_tensor(file_path: Path) -> torch.Tensor:
    suffix = file_path.suffix.lower()
    if suffix == ".pt":
        data = torch.load(file_path, map_location="cpu")
        if isinstance(data, dict) and "tensor" in data:
            data = data["tensor"]
        return torch.as_tensor(data, dtype=torch.float32)
    if suffix in {".npy", ".npz"}:
        arr = np.load(file_path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr["arr_0"]
        return torch.tensor(arr, dtype=torch.float32)
    if suffix in {".csv", ".txt"}:
        df = pd.read_csv(file_path)
        return torch.tensor(df.values, dtype=torch.float32)
    if suffix in {".pkl"}:
        import pickle  # local import to avoid unnecessary dependency at runtime

        with open(file_path, "rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, dict):
            # heuristic: prefer keys commonly used in physiological datasets
            for key in ("sequence", "signal", "data"):
                if key in obj:
                    obj = obj[key]
                    break
        return torch.tensor(obj, dtype=torch.float32)

    raise ValueError(f"Unsupported file extension for {file_path}")


def standardize(sequence: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = sequence.mean(dim=0, keepdim=True)
    std = sequence.std(dim=0, keepdim=True)
    return (sequence - mean) / (std + eps)


def fit_to_length(sequence: torch.Tensor, seq_len: int) -> torch.Tensor:
    if sequence.dim() != 2:
        raise ValueError(f"Expected 2-D tensor, got shape {tuple(sequence.shape)}")
    length = sequence.size(0)
    if length == seq_len:
        return sequence
    if length > seq_len:
        start = (length - seq_len) // 2
        return sequence[start : start + seq_len]
    pad_len = seq_len - length
    pad = torch.zeros(pad_len, sequence.size(1), dtype=sequence.dtype)
    return torch.cat([sequence, pad], dim=0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class SampleRecord:
    path: Path
    label: str


class SequenceClassificationDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SampleRecord],
        seq_len: int,
        channels: int,
        standardize_input: bool = True,
    ) -> None:
        self.records = list(records)
        self.seq_len = seq_len
        self.channels = channels
        self.standardize = standardize_input
        labels = sorted({rec.label for rec in self.records})
        self.label_to_id = {label: i for i, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.records)

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 1:
            raise ValueError("Sequence tensor must be 2-D (time, channels)")
        if tensor.size(-1) != self.channels and tensor.size(0) == self.channels:
            tensor = tensor.transpose(0, 1)
        if tensor.size(-1) != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got tensor with shape {tuple(tensor.shape)}"
            )
        tensor = fit_to_length(tensor, self.seq_len)
        if self.standardize:
            tensor = standardize(tensor)
        return tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        record = self.records[idx]
        tensor = load_sequence_tensor(record.path)
        tensor = self._prepare_tensor(tensor)
        return tensor, self.label_to_id[record.label]


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class AutoencoderBackbone(nn.Module):
    """
    Reuses the encoder (3 TCN branches + transformer) from AutoencoderModel.
    """

    def __init__(self, autoencoder: AutoencoderModel, pooling: str = "mean") -> None:
        super().__init__()
        self.tcn1 = autoencoder.tcn1
        self.tcn2 = autoencoder.tcn2
        self.tcn3 = autoencoder.tcn3
        self.transformer = autoencoder.transformer
        self.pooling = pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, 3]
        x1 = x[..., 0].unsqueeze(-1).permute(0, 2, 1)  # [B, 1, L]
        x2 = x[..., 1].unsqueeze(-1).permute(0, 2, 1)
        x3 = x[..., 2].unsqueeze(-1).permute(0, 2, 1)

        z1 = self.tcn1(x1)
        z2 = self.tcn2(x2)
        z3 = self.tcn3(x3)

        cat = torch.cat([z1, z2, z3], dim=1).permute(0, 2, 1)  # [B, L, C]
        encoded = self.transformer(cat)  # [B, L, D]

        if self.pooling == "mean":
            return encoded.mean(dim=1)
        if self.pooling == "max":
            return encoded.max(dim=1).values
        if self.pooling == "last":
            return encoded[:, -1, :]
        raise ValueError(f"Unsupported pooling mode: {self.pooling}")


class EncoderClassifier(nn.Module):
    def __init__(
        self,
        backbone: AutoencoderBackbone,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.classifier(feats)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int,
    writer: SummaryWriter | None,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    for step, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

        if writer and step % log_interval == 0:
            global_step = epoch * len(loader) + step
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
    accuracy = correct / max(1, total)
    epoch_loss = running_loss / max(1, total)
    return epoch_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str,
    writer: SummaryWriter | None,
    epoch: int,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)

    avg_loss = running_loss / max(1, total)
    accuracy = correct / max(1, total)
    if writer:
        writer.add_scalar(f"{split_name}/loss", avg_loss, epoch)
        writer.add_scalar(f"{split_name}/accuracy", accuracy, epoch)
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Manifest utilities
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Path) -> List[SampleRecord]:
    df = pd.read_csv(manifest_path)
    if "path" not in df.columns or "label" not in df.columns:
        raise ValueError("Manifest must contain 'path' and 'label' columns.")
    records: List[SampleRecord] = []
    for _, row in df.iterrows():
        sample_path = resolve_sample_path(str(row["path"]))
        label = str(row["label"])
        records.append(SampleRecord(path=sample_path, label=label))
    return records


def split_records(
    records: List[SampleRecord],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_total = len(records)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    pick = lambda idxs: [records[i] for i in idxs]
    return pick(train_idx), pick(val_idx), pick(test_idx)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classifier on datasets 7-10 using the pretrained encoder.")
    parser.add_argument("--manifest", required=True, type=Path, help="CSV with columns path,label.")
    parser.add_argument("--autoencoder_ckpt", required=True, type=Path, help="Path to pretrained autoencoder checkpoint.")
    parser.add_argument("--num_classes", required=True, type=int, help="Number of target classes.")
    parser.add_argument("--tcn_channels", nargs="+", type=int, required=True, help="TCN channel sizes (must match the pretrained model).")
    parser.add_argument("--transformer_dim", type=int, required=True)
    parser.add_argument("--transformer_heads", type=int, required=True)
    parser.add_argument("--transformer_layers", type=int, required=True)
    parser.add_argument("--lstm_hidden", type=int, required=True)
    parser.add_argument("--lstm_layers", type=int, required=True)
    parser.add_argument("--seq_len", type=int, default=960)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used in classifier head.")
    parser.add_argument("--classifier_hidden", type=int, default=256)
    parser.add_argument("--pooling", choices=["mean", "max", "last"], default="mean")
    parser.add_argument("--freeze_encoder", action="store_true", help="If set, only train the classifier head.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--log_dir", type=Path, default=PROJECT_ROOT / "classification_runs")
    parser.add_argument("--ckpt_dir", type=Path, default=PROJECT_ROOT / "classification_ckpts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    return parser.parse_args()


def load_autoencoder_weights(model: AutoencoderModel, ckpt_path: Path) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys when loading checkpoint: {unexpected}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(args.manifest)
    if len(records) < args.num_classes:
        raise ValueError("Not enough samples compared to number of classes.")

    train_records, val_records, test_records = split_records(records, args.train_ratio, args.val_ratio, args.seed)
    if not train_records or not val_records or not test_records:
        raise ValueError("Split produced an empty subset. Adjust train/val ratios.")

    train_dataset = SequenceClassificationDataset(train_records, args.seq_len, channels=3)
    val_dataset = SequenceClassificationDataset(val_records, args.seq_len, channels=3)
    test_dataset = SequenceClassificationDataset(test_records, args.seq_len, channels=3, standardize_input=False)

    label_mapping = train_dataset.label_to_id
    print(f"Detected labels: {label_mapping}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    autoencoder = AutoencoderModel(
        tcn_channels=args.tcn_channels,
        transformer_dim=args.transformer_dim,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        output_dim=3,
        seq_len=args.seq_len,
        dropout=args.dropout,
    )
    load_autoencoder_weights(autoencoder, args.autoencoder_ckpt)

    backbone = AutoencoderBackbone(autoencoder, pooling=args.pooling)
    if args.freeze_encoder:
        for param in backbone.parameters():
            param.requires_grad = False

    model = EncoderClassifier(
        backbone=backbone,
        feature_dim=args.transformer_dim,
        num_classes=args.num_classes,
        hidden_dim=args.classifier_hidden,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=args.log_dir / f"run_{args.seed}")

    best_val = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs} — device: {device.type}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.log_interval, writer
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, "val", writer, epoch)
        scheduler.step(val_loss)
        print(
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.3f} | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.3f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "label_to_id": label_mapping,
            }
            torch.save(best_state, args.ckpt_dir / "best_classifier.pt")

    writer.close()
    if best_state is None:
        raise RuntimeError("Training did not complete properly.")
    model.load_state_dict(best_state["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, "test", writer=None, epoch=args.epochs)
    print(f"\nTest Loss {test_loss:.4f} | Test Acc {test_acc:.3f}")
    print(f"Best checkpoint saved to {args.ckpt_dir / 'best_classifier.pt'}")


if __name__ == "__main__":
    main()

