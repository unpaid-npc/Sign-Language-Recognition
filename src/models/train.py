"""
Reproducible Training and Validation Pipeline for SignBiLSTMClassifier on WLASL-100.
Saves best and last model checkpoints along with training history curves.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    EARLY_STOPPING_PATIENCE,
    LAST_MODEL_PATH,
    LEARNING_RATE,
    METADATA_CSV_PATH,
    NUM_EPOCHS,
    OUTPUTS_DIR,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from src.data.dataset import create_dataloaders
from src.models.lstm import SignBiLSTMClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_topk_correct(
    logits: torch.Tensor, targets: torch.Tensor, k: int = 5
) -> int:
    """Compute number of samples where ground-truth target is in top-k predictions."""
    max_k = min(k, logits.size(1))
    _, topk_pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    correct = topk_pred.eq(targets.view(-1, 1).expand_as(topk_pred))
    return int(correct.any(dim=1).sum().item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Run one training epoch and return (loss, top1_acc, top5_acc)."""
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        # Clip gradients to prevent exploding gradients in BiLSTM
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = y_batch.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct_top1 += int((preds == y_batch).sum().item())
        correct_top5 += compute_topk_correct(logits, y_batch, k=5)
        total += batch_size

    epoch_loss = running_loss / max(1, total)
    top1_acc = correct_top1 / max(1, total)
    top5_acc = correct_top5 / max(1, total)
    return epoch_loss, top1_acc, top5_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate model on validation/test loader and return (loss, top1_acc, top5_acc)."""
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        batch_size = y_batch.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct_top1 += int((preds == y_batch).sum().item())
        correct_top5 += compute_topk_correct(logits, y_batch, k=5)
        total += batch_size

    epoch_loss = running_loss / max(1, total)
    top1_acc = correct_top1 / max(1, total)
    top5_acc = correct_top5 / max(1, total)
    return epoch_loss, top1_acc, top5_acc


def save_checkpoint(
    model: SignBiLSTMClassifier,
    idx_to_gloss: Dict[int, str],
    epoch: int,
    val_acc: float,
    val_top5_acc: float,
    path: Path,
) -> None:
    """Save complete model checkpoint including architecture config and label mapping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "idx_to_gloss": {int(k): str(v) for k, v in idx_to_gloss.items()},
        "val_top1_acc": float(val_acc),
        "val_top5_acc": float(val_top5_acc),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    checkpoint_path: Path = BEST_MODEL_PATH,
    device: Optional[torch.device] = None,
) -> Tuple[SignBiLSTMClassifier, Dict[int, str], Dict[str, Any]]:
    """
    Load a trained SignBiLSTMClassifier and its label mapping from disk.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]

    model = SignBiLSTMClassifier(
        input_dim=int(cfg["input_dim"]),
        projection_dim=int(cfg["projection_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        num_classes=int(cfg["num_classes"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    idx_to_gloss = {int(k): str(v) for k, v in ckpt["idx_to_gloss"].items()}
    return model, idx_to_gloss, ckpt


def plot_training_history(history: Dict[str, List[float]], save_path: Path) -> None:
    """Generate and save Loss and Accuracy training curves."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1: Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title("WLASL-100 BiLSTM Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Accuracy
    axes[1].plot(
        epochs,
        [a * 100 for a in history["train_top1"]],
        label="Train Top-1 Acc (%)",
        linewidth=2,
    )
    axes[1].plot(
        epochs,
        [a * 100 for a in history["val_top1"]],
        label="Val Top-1 Acc (%)",
        linewidth=2,
    )
    axes[1].plot(
        epochs,
        [a * 100 for a in history["val_top5"]],
        label="Val Top-5 Acc (%)",
        linestyle="--",
        linewidth=2,
    )
    axes[1].set_title("WLASL-100 BiLSTM Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved training curves to {save_path}")


def run_training(
    metadata_csv_path: Path = METADATA_CSV_PATH,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = EARLY_STOPPING_PATIENCE,
    top_k_classes: Optional[int] = None,
    seed: int = RANDOM_SEED,
    best_model_path: Path = BEST_MODEL_PATH,
    last_model_path: Path = LAST_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Execute the complete training and validation loop.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training on device: {device} (seed={seed})")

    train_loader, val_loader, _, idx_to_gloss, class_weights = create_dataloaders(
        metadata_csv_path=metadata_csv_path,
        batch_size=batch_size,
        top_k_classes=top_k_classes,
        seed=seed,
    )

    num_classes = len(idx_to_gloss)
    model = SignBiLSTMClassifier(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Initialized SignBiLSTMClassifier with {total_params:,} trainable parameters "
        f"for {num_classes} classes."
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=0.05
    )
    val_criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.02
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_top1": [],
        "train_top5": [],
        "val_top1": [],
        "val_top5": [],
    }

    best_val_top1 = -1.0
    best_val_top5 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_top1, tr_top5 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_top1, val_top5 = validate(
            model, val_loader, val_criterion, device
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_top1"].append(tr_top1)
        history["train_top5"].append(tr_top5)
        history["val_top1"].append(val_top1)
        history["val_top5"].append(val_top5)

        is_best = (val_top1 > best_val_top1) or (
            abs(val_top1 - best_val_top1) < 1e-6 and val_top5 > best_val_top5
        )

        if is_best:
            best_val_top1 = val_top1
            best_val_top5 = val_top5
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                model, idx_to_gloss, epoch, val_top1, val_top5, best_model_path
            )
            marker = " * [BEST]"
        else:
            epochs_without_improvement += 1
            marker = ""

        if epoch == 1 or epoch % 5 == 0 or is_best or epoch == num_epochs:
            logger.info(
                f"Epoch {epoch:03d}/{num_epochs:03d} | "
                f"Train Loss: {tr_loss:.4f} Top1: {tr_top1*100:5.1f}% | "
                f"Val Loss: {val_loss:.4f} Top1: {val_top1*100:5.1f}% Top5: {val_top5*100:5.1f}%{marker}"
            )

        if epochs_without_improvement >= patience:
            logger.info(
                f"Early stopping triggered at epoch {epoch} "
                f"(no val Top-1 improvement for {patience} epochs)."
            )
            break

    save_checkpoint(
        model, idx_to_gloss, epoch, val_top1, val_top5, last_model_path
    )

    elapsed = time.time() - start_time
    plot_training_history(history, OUTPUTS_DIR / "training_curves.png")

    history_path = OUTPUTS_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info("=" * 65)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 65)
    logger.info(f"Total Training Time     : {elapsed:.1f}s")
    logger.info(f"Best Validation Epoch   : {best_epoch}")
    logger.info(f"Best Val Top-1 Accuracy : {best_val_top1 * 100:.2f}%")
    logger.info(f"Best Val Top-5 Accuracy : {best_val_top5 * 100:.2f}%")
    logger.info(f"Best Checkpoint Saved   : {best_model_path}")
    logger.info("=" * 65)

    return {
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "best_val_top5": best_val_top5,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SignBiLSTMClassifier on WLASL-100 landmark sequences."
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument(
        "--patience", type=int, default=22, help="Early stopping patience."
    )
    parser.add_argument(
        "--top-k-classes",
        type=int,
        default=None,
        help="Optional subset of classes (default: None = all 100 classes).",
    )
    args = parser.parse_args()

    run_training(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        top_k_classes=args.top_k_classes,
    )


if __name__ == "__main__":
    main()