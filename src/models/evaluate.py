"""
Comprehensive Evaluation Module for Trained WLASL-100 BiLSTM Model.
Computes actual Test Set Top-1 & Top-5 Accuracy, Macro & Weighted Precision,
Recall, F1-Score, Per-Class Performance CSV, and Confusion Matrix Visualization.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import BEST_MODEL_PATH, METADATA_CSV_PATH, OUTPUTS_DIR
from src.data.dataset import create_dataloaders
from src.models.train import compute_topk_correct, load_checkpoint
from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
    max_display_classes: int = 25,
) -> None:
    """
    Plot and save the confusion matrix. For 100 classes, plots both the full 100x100
    heatmap and a zoomed-in top-25 classes heatmap for readability.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Full 100x100 Matrix
    im0 = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[0].set_title(f"Full Confusion Matrix ({len(class_names)} Classes)")
    axes[0].set_xlabel("Predicted Class Index")
    axes[0].set_ylabel("True Class Index")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Right: Top N classes with highest test support or first 25 classes
    n_sub = min(max_display_classes, len(class_names))
    cm_sub = cm[:n_sub, :n_sub]
    sub_names = class_names[:n_sub]

    im1 = axes[1].imshow(cm_sub, interpolation="nearest", cmap="Blues")
    axes[1].set_title(f"First {n_sub} WLASL Glosses Detail")
    axes[1].set_xticks(np.arange(n_sub))
    axes[1].set_yticks(np.arange(n_sub))
    axes[1].set_xticklabels(sub_names, rotation=90, fontsize=8)
    axes[1].set_yticklabels(sub_names, fontsize=8)
    axes[1].set_xlabel("Predicted Gloss")
    axes[1].set_ylabel("True Gloss")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix visualization to {save_path}")


@torch.no_grad()
def evaluate_model(
    checkpoint_path: Path = BEST_MODEL_PATH,
    metadata_csv_path: Path = METADATA_CSV_PATH,
    outputs_dir: Path = OUTPUTS_DIR,
) -> Dict[str, Any]:
    """
    Evaluate the saved model checkpoint on both Validation and Test splits,
    saving full classification reports and confusion matrices.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_gloss, ckpt_meta = load_checkpoint(checkpoint_path, device=device)
    num_classes = len(idx_to_gloss)

    _, val_loader, test_loader, _, _ = create_dataloaders(
        metadata_csv_path=metadata_csv_path,
        top_k_classes=num_classes if num_classes < 100 else None,
    )

    all_preds: List[int] = []
    all_targets: List[int] = []
    top5_correct = 0
    total_samples = 0

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        preds = logits.argmax(dim=1)

        top5_correct += compute_topk_correct(logits, y_batch, k=5)
        total_samples += y_batch.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(y_batch.cpu().numpy().tolist())

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)

    test_top1_acc = float(accuracy_score(y_true, y_pred))
    test_top5_acc = float(top5_correct) / max(1, total_samples)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    labels_list = list(range(num_classes))
    class_names = [idx_to_gloss.get(i, f"class_{i}") for i in labels_list]

    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_list, average=None, zero_division=0
    )

    per_class_rows = []
    for i in labels_list:
        per_class_rows.append(
            {
                "class_id": i,
                "gloss": class_names[i],
                "precision": round(float(per_p[i]), 4),
                "recall": round(float(per_r[i]), 4),
                "f1_score": round(float(per_f1[i]), 4),
                "test_support": int(per_support[i]),
            }
        )

    per_class_df = pd.DataFrame(per_class_rows)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    per_class_csv_path = outputs_dir / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_csv_path, index=False)

    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    cm_plot_path = outputs_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_plot_path)

    top_performing = (
        per_class_df[per_class_df["test_support"] > 0]
        .sort_values(by=["f1_score", "test_support"], ascending=False)
        .head(15)
    )

    metrics_summary = {
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": int(ckpt_meta.get("epoch", -1)),
        "num_classes": num_classes,
        "test_samples": total_samples,
        "val_top1_accuracy": round(float(ckpt_meta.get("val_top1_acc", 0.0)), 4),
        "val_top5_accuracy": round(float(ckpt_meta.get("val_top5_acc", 0.0)), 4),
        "test_top1_accuracy": round(test_top1_acc, 4),
        "test_top5_accuracy": round(test_top5_acc, 4),
        "test_macro_precision": round(float(macro_p), 4),
        "test_macro_recall": round(float(macro_r), 4),
        "test_macro_f1": round(float(macro_f1), 4),
        "test_weighted_precision": round(float(weighted_p), 4),
        "test_weighted_recall": round(float(weighted_r), 4),
        "test_weighted_f1": round(float(weighted_f1), 4),
        "top_15_classes_by_f1": top_performing.to_dict(orient="records"),
    }

    metrics_json_path = outputs_dir / "evaluation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info("=" * 65)
    logger.info("TEST SET EVALUATION REPORT (ACTUAL METRICS)")
    logger.info("=" * 65)
    logger.info(f"Checkpoint Evaluated     : {checkpoint_path.name} (Epoch {ckpt_meta.get('epoch')})")
    logger.info(f"Total Test Sequences     : {total_samples} across {num_classes} classes")
    logger.info(f"Validation Top-1 / Top-5 : {metrics_summary['val_top1_accuracy']*100:.2f}% / {metrics_summary['val_top5_accuracy']*100:.2f}%")
    logger.info(f"Test Top-1 Accuracy      : {test_top1_acc * 100:.2f}%")
    logger.info(f"Test Top-5 Accuracy      : {test_top5_acc * 100:.2f}%")
    logger.info(f"Test Macro Precision     : {macro_p * 100:.2f}%")
    logger.info(f"Test Macro Recall        : {macro_r * 100:.2f}%")
    logger.info(f"Test Macro F1-Score      : {macro_f1 * 100:.2f}%")
    logger.info(f"Test Weighted F1-Score   : {weighted_f1 * 100:.2f}%")
    logger.info(f"Per-Class CSV Saved To   : {per_class_csv_path}")
    logger.info(f"Confusion Matrix Saved   : {cm_plot_path}")
    logger.info("=" * 65)

    return metrics_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained SignBiLSTMClassifier on test set.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BEST_MODEL_PATH,
        help="Path to model checkpoint (.pt).",
    )
    args = parser.parse_args()
    evaluate_model(checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()