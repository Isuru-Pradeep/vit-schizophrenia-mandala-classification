from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def compute_class_weights(targets, num_classes: int, device: str):
    targets = np.asarray(targets, dtype=int)
    class_counts = np.bincount(targets, minlength=num_classes)
    total = class_counts.sum()

    weights = np.zeros(num_classes, dtype=np.float32)
    non_zero_mask = class_counts > 0
    weights[non_zero_mask] = total / (num_classes * class_counts[non_zero_mask])

    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_per_class_metrics(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    return pd.DataFrame(
        {
            "class_index": labels,
            "class_name": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def build_confusion_matrix_df(y_true, y_pred, class_names):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )
    return pd.DataFrame(
        matrix,
        index=[f"true_{name}" for name in class_names],
        columns=[f"pred_{name}" for name in class_names],
    )


def save_confusion_matrix_plot(confusion_df: pd.DataFrame, output_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(confusion_df.values, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(confusion_df.columns)))
    ax.set_yticks(range(len(confusion_df.index)))
    ax.set_xticklabels(confusion_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(confusion_df.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for row_idx in range(confusion_df.shape[0]):
        for col_idx in range(confusion_df.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                str(confusion_df.iat[row_idx, col_idx]),
                ha="center",
                va="center",
                color="black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def save_training_history_plot(
    history_df: pd.DataFrame,
    output_path: Path,
    title: str = "Training History",
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    has_macro_f1 = "val_macro_f1" in history_df.columns

    fig, axes = plt.subplots(1, 3 if has_macro_f1 else 2, figsize=(16 if has_macro_f1 else 12, 4))

    # Loss curve
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Accuracy curve
    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], label="Train Accuracy")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    # Macro F1 curve — primary model selection metric per dissertation §3.5
    if has_macro_f1:
        if "train_f1" in history_df.columns:
            axes[2].plot(history_df["epoch"], history_df["train_f1"], label="Train F1 (weighted)")
        axes[2].plot(history_df["epoch"], history_df["val_macro_f1"], label="Val Macro F1", linewidth=2)
        axes[2].set_title("Macro F1 (model selection criterion)")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True
