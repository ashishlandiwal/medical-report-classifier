"""Metrics and diagnostic plots, oriented around the malignant (positive) class."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

POSITIVE_LABEL = 1


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    positive_label: int = POSITIVE_LABEL,
) -> dict[str, float]:
    """Headline metrics. Recall on the positive (malignant) class is the one that
    matters most clinically — a false negative is a missed cancer."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_pos": precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "recall_pos": recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "f1_pos": f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    return {k: float(v) for k, v in metrics.items()}


def plot_confusion_matrix(y_true, y_pred, class_names, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (best model)")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(y_true, y_prob, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Precision–Recall curve (best model)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
