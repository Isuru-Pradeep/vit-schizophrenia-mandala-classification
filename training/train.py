import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config.config import TrainConfig
from evaluation.metrics import (
    build_confusion_matrix_df,
    build_per_class_metrics,
    classification_metrics,
    compute_class_weights,
    save_confusion_matrix_plot,
    save_training_history_plot,
)
from models.vit_classifier import MandalaViTClassifier, freeze_backbone, unfreeze_last_blocks
from utils.dataset import MandalaClassificationDataset
from utils.transforms import get_eval_transforms, get_train_transforms


def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def setup_logger(log_path: Path):
    logger = logging.getLogger(f"mandala_classifier_{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def get_class_to_index(config: TrainConfig):
    if len(config.class_names) != config.num_classes:
        raise ValueError("class_names length must match num_classes.")
    active_class_names = config.get_active_class_names()
    if not active_class_names:
        raise ValueError("No active classes configured for training.")
    return {class_name: index for index, class_name in enumerate(active_class_names)}


def resolve_completion_time_csv(config: TrainConfig, data_dir: Path) -> Path:
    if not config.use_time_feature:
        return None

    if config.completion_time_csv:
        csv_path = Path(config.completion_time_csv)
        if not csv_path.is_absolute():
            csv_path = (PROJECT_ROOT / config.completion_time_csv).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Completion-time CSV not found: {csv_path}")
        return csv_path

    csv_candidates = sorted(data_dir.glob("*.csv"))
    if len(csv_candidates) == 1:
        return csv_candidates[0]
    if not csv_candidates:
        raise FileNotFoundError(
            f"No CSV file found in dataset directory: {data_dir}. "
            "Add a CSV with sample names and completion times."
        )
    raise ValueError(
        f"Multiple CSV files found in {data_dir}. Set completion_time_csv in config.py explicitly."
    )


def build_dataframe_from_class_directories(config: TrainConfig) -> pd.DataFrame:
    data_dir = (PROJECT_ROOT / config.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    class_to_index = get_class_to_index(config)
    rows = []
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    for class_name, class_index in class_to_index.items():
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in allowed_suffixes:
                continue

            rows.append(
                {
                    config.image_col: image_path.name,
                    config.image_path_col: str(image_path.resolve()),
                    config.sample_id_col: image_path.stem,
                    config.target_col: class_index,
                }
            )

    if not rows:
        raise ValueError(f"No image files found under class directories in: {data_dir}")

    image_df = pd.DataFrame(rows)

    if not config.use_time_feature:
        image_df[config.time_col] = 0.0
        return image_df

    csv_path = resolve_completion_time_csv(config, data_dir)
    time_df = pd.read_csv(csv_path)

    required_time_columns = [config.sample_id_col, config.time_col]
    missing_time_columns = [col for col in required_time_columns if col not in time_df.columns]
    if missing_time_columns:
        raise ValueError(
            f"Missing required columns in completion-time CSV {csv_path}: {missing_time_columns}"
        )

    time_df = time_df.dropna(subset=required_time_columns).copy()
    time_df[config.sample_id_col] = time_df[config.sample_id_col].astype(str).str.strip()
    time_df[config.time_col] = time_df[config.time_col].astype(float)
    time_df["sample_key"] = time_df[config.sample_id_col].str.lower()
    time_df["sample_key_stem"] = time_df["sample_key"].str.replace(
        r"\.[^.]+$",
        "",
        regex=True,
    )

    if time_df["sample_key"].duplicated().any() or time_df["sample_key_stem"].duplicated().any():
        duplicate_rows = time_df.loc[
            time_df["sample_key"].duplicated(keep=False)
            | time_df["sample_key_stem"].duplicated(keep=False),
            config.sample_id_col,
        ].tolist()
        raise ValueError(
            "Duplicate sample identifiers found in completion-time CSV: "
            + ", ".join(duplicate_rows[:10])
            + (" ..." if len(duplicate_rows) > 10 else "")
        )

    image_df["sample_key"] = image_df[config.sample_id_col].astype(str).str.strip().str.lower()
    image_df["sample_key_with_ext"] = image_df[config.image_col].astype(str).str.strip().str.lower()

    time_by_stem = time_df.set_index("sample_key_stem")[config.time_col]
    time_by_name = time_df.set_index("sample_key")[config.time_col]

    image_df[config.time_col] = image_df["sample_key"].map(time_by_stem)
    missing_mask = image_df[config.time_col].isna()
    image_df.loc[missing_mask, config.time_col] = image_df.loc[missing_mask, "sample_key_with_ext"].map(
        time_by_name
    )

    missing_mask = image_df[config.time_col].isna()
    if missing_mask.all():
        raise ValueError(
            "No images from the class directories matched the completion-time CSV."
        )

    if missing_mask.any():
        image_df = image_df.loc[~missing_mask].copy()

    df = image_df.drop(columns=["sample_key", "sample_key_with_ext"])
    return df


def load_metadata(config: TrainConfig) -> pd.DataFrame:
    df = build_dataframe_from_class_directories(config)

    required = [config.image_col, config.image_path_col, config.target_col, config.sample_id_col]
    if config.use_time_feature:
        required.append(config.time_col)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset dataframe: {missing}")

    df = df.dropna(subset=required).reset_index(drop=True)
    df[config.target_col] = df[config.target_col].astype(int)

    if df[config.target_col].min() < 0:
        raise ValueError("Class labels must start from 0.")

    unique_classes = sorted(df[config.target_col].unique().tolist())
    effective_num_classes = config.get_effective_num_classes()
    if max(unique_classes) >= effective_num_classes:
        raise ValueError(
            f"Found class label {max(unique_classes)} but num_classes={effective_num_classes}."
        )

    return df


def split_dataset(df: pd.DataFrame, config: TrainConfig, logger=None):
    """
    Stratified 50% / 25% / 25% split on the original samples.
    Dissertation Section 3.2.4: split is applied before augmentation.
    Augmented images are already in the training class directories.
    """
    y = df[config.target_col].astype(int)

    # First split: carve out 25% test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )

    # Second split: carve out 25% val from remaining 75% → val = 25/75 = 0.333
    y_train_val = train_val_df[config.target_col].astype(int)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=config.val_size,
        stratify=y_train_val,
        random_state=config.random_state,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    msg = (
        f"Dataset split → Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return train_df, val_df, test_df


def normalize_time_column(train_df, val_df, test_df, time_col: str):
    """
    Normalize completion time using mean/std computed from training set only.
    Dissertation Equation 3.5: t_norm = (t - mu) / sigma
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    mean = train_df[time_col].astype(float).mean()
    std = train_df[time_col].astype(float).std()

    if std == 0 or pd.isna(std):
        std = 1.0

    train_df[time_col] = ((train_df[time_col].astype(float) - mean) / std).astype(float)
    val_df[time_col] = ((val_df[time_col].astype(float) - mean) / std).astype(float)
    test_df[time_col] = ((test_df[time_col].astype(float) - mean) / std).astype(float)

    return train_df, val_df, test_df, mean, std


def create_dataloaders(train_df, val_df, test_df, config: TrainConfig):
    train_dataset = MandalaClassificationDataset(
        dataframe=train_df,
        image_col=config.image_col,
        image_path_col=config.image_path_col,
        target_col=config.target_col,
        time_col=config.time_col,
        transform=get_train_transforms(config.image_size, config),
        use_time_feature=config.use_time_feature,
    )

    val_dataset = MandalaClassificationDataset(
        dataframe=val_df,
        image_col=config.image_col,
        image_path_col=config.image_path_col,
        target_col=config.target_col,
        time_col=config.time_col,
        transform=get_eval_transforms(config.image_size),
        use_time_feature=config.use_time_feature,
    )

    test_dataset = MandalaClassificationDataset(
        dataframe=test_df,
        image_col=config.image_col,
        image_path_col=config.image_path_col,
        target_col=config.target_col,
        time_col=config.time_col,
        transform=get_eval_transforms(config.image_size),
        use_time_feature=config.use_time_feature,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def build_model(config: TrainConfig, device: str):
    model = MandalaViTClassifier(
        model_name=config.model_name,
        pretrained=config.pretrained,
        use_time_feature=config.use_time_feature,
        dropout=config.dropout,
        num_classes=config.get_effective_num_classes(),
    ).to(device)
    return model


def build_optimizer(model, config: TrainConfig, stage: str = "head_only"):
    if stage == "head_only":
        # Stage 1: train only the classification head
        return torch.optim.AdamW(
            model.classifier.parameters(),
            lr=config.head_lr,
            weight_decay=config.weight_decay,
        )

    # Stage 2: partial fine-tune - lower LR for unfrozen backbone blocks, head LR unchanged
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config.backbone_lr},
            {"params": head_params, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )


def build_criterion(train_df: pd.DataFrame, config: TrainConfig, device: str, logger=None):
    weight = None
    if config.use_class_weights:
        weight = compute_class_weights(
            train_df[config.target_col].astype(int).tolist(),
            num_classes=config.get_effective_num_classes(),
            device=device,
        )
        message = f"Class weights: {weight.detach().cpu().numpy().round(4).tolist()}"
        if logger is not None:
            logger.info(message)
        else:
            print(message)

    return nn.CrossEntropyLoss(
        weight=weight,
        label_smoothing=config.label_smoothing,
    )


def run_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: str,
    train: bool = True,
    return_predictions: bool = False,
):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_targets = []
    all_preds = []

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, times, targets, _ in loader:
            images = images.to(device)
            times = times.to(device)
            targets = targets.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images, times)
            loss = criterion(logits, targets)

            if train:
                loss.backward()
                optimizer.step()

            preds = torch.argmax(logits, dim=1)

            running_loss += loss.item()
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = running_loss / max(len(loader), 1)
    metrics = classification_metrics(all_targets, all_preds)
    metrics["loss"] = avg_loss
    if return_predictions:
        metrics["y_true"] = all_targets
        metrics["y_pred"] = all_preds
    return metrics


def evaluate_and_collect_predictions(model, loader, device: str, class_names):
    model.eval()
    names = []
    targets = []
    preds = []
    probabilities = []

    with torch.no_grad():
        for images, times, y, image_names in loader:
            images = images.to(device)
            times = times.to(device)

            logits = model(images, times)
            probs = torch.softmax(logits, dim=1)
            pred_classes = torch.argmax(probs, dim=1)

            preds.extend(pred_classes.cpu().numpy().tolist())
            targets.extend(y.numpy().tolist())
            names.extend(list(image_names))
            probabilities.extend(probs.cpu().numpy().tolist())

    data = {
        "image_name": names,
        "y_true": targets,
        "y_pred": preds,
    }
    for class_index, class_name in enumerate(class_names):
        data[f"prob_{class_name}"] = [row[class_index] for row in probabilities]

    return pd.DataFrame(data)


def main():
    config = TrainConfig()
    seed_everything(config.random_state)

    checkpoint_dir = PROJECT_ROOT / config.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_dir = checkpoint_dir / f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "training.log")

    device = choose_device()
    logger.info("Starting ViT-based Schizophrenia Severity Classification")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {config.model_name}")

    # Load all samples from class directories
    df = load_metadata(config)
    active_class_names = config.get_active_class_names()
    logger.info(f"Total samples loaded: {len(df)}")
    logger.info(f"Classes: {active_class_names}")

    # Stratified 50% / 25% / 25% split (Dissertation Section 3.2.4, Table 3.8)
    train_df, val_df, test_df = split_dataset(df, config, logger=logger)

    # Normalize completion time using training set statistics only (Dissertation Eq. 3.5)
    time_mean, time_std = 0.0, 1.0
    if config.use_time_feature:
        train_df, val_df, test_df, time_mean, time_std = normalize_time_column(
            train_df, val_df, test_df, config.time_col
        )
        logger.info(f"Completion time normalization → mean: {time_mean:.4f}, std: {time_std:.4f}")

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_loader, val_loader, test_loader = create_dataloaders(train_df, val_df, test_df, config)
    logger.info(
        f"Dataloaders → train batches: {len(train_loader)}, "
        f"val batches: {len(val_loader)}, test batches: {len(test_loader)}"
    )

    model = build_model(config, device)
    logger.info(f"ViT backbone feature dimension: {model.backbone.num_features}")

    criterion = build_criterion(train_df, config, device, logger=logger)

    # ── Stage 1: Head-only training (backbone frozen) ──────────────────────────
    # Dissertation Section 3.5: freeze backbone, train head for head_only_epochs
    freeze_backbone(model)
    logger.info("Stage 1: Backbone frozen — training classification head only")
    optimizer = build_optimizer(model, config, stage="head_only")

    best_state = None
    best_val_loss = float("inf")
    best_macro_f1 = -1.0
    best_epoch = -1
    no_improve_count = 0
    history_rows = []

    for epoch in range(config.max_epochs):

        # ── Stage 2: Partial fine-tuning (unfreeze last N blocks) ─────────────
        # Dissertation Section 3.5: unfreeze last 2 Transformer blocks at epoch head_only_epochs
        if epoch == config.head_only_epochs:
            unfreeze_last_blocks(model, num_blocks=config.unfreeze_last_n_blocks)
            optimizer = build_optimizer(model, config, stage="partial_finetune")
            logger.info(
                f"Stage 2: Unfroze last {config.unfreeze_last_n_blocks} Transformer block(s) "
                f"— backbone LR: {config.backbone_lr}, head LR: {config.head_lr}"
            )

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=False,
        )

        history_rows.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_macro_f1": val_metrics["macro_f1"],
        })

        logger.info(
            f"Epoch [{epoch + 1}/{config.max_epochs}] | "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
            f"Macro F1: {val_metrics['macro_f1']:.4f}"
        )

        # Best model selection: lowest val loss AND highest macro F1 (Dissertation Section 3.5)
        val_loss_improved = val_metrics["loss"] < best_val_loss
        macro_f1_improved = val_metrics["macro_f1"] > best_macro_f1

        if val_loss_improved or macro_f1_improved:
            if val_loss_improved:
                best_val_loss = val_metrics["loss"]
            if macro_f1_improved:
                best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve_count = 0
            logger.info(
                f"  → New best model at epoch {best_epoch} "
                f"(val loss: {best_val_loss:.4f}, macro F1: {best_macro_f1:.4f})"
            )
        else:
            no_improve_count += 1
            logger.info(f"  → No improvement for {no_improve_count} epoch(s)")

        if no_improve_count >= config.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is None:
        raise RuntimeError("No best model state was saved during training.")

    # ── Validation set evaluation with best checkpoint ──────────────────────────
    model.load_state_dict(best_state)

    val_final = run_one_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        train=False,
        return_predictions=True,
    )

    logger.info("=" * 70)
    logger.info("Best model — Validation set results")
    logger.info("=" * 70)
    logger.info(f"Best epoch:   {best_epoch}")
    logger.info(f"Val Loss:     {best_val_loss:.4f}")
    logger.info(f"Val Accuracy: {val_final['accuracy']:.4f}")
    logger.info(f"Val Macro F1: {val_final['macro_f1']:.4f}")

    val_per_class = build_per_class_metrics(val_final["y_true"], val_final["y_pred"], active_class_names)
    logger.info("Validation per-class metrics:\n" + val_per_class.to_string(index=False))

    val_confusion = build_confusion_matrix_df(val_final["y_true"], val_final["y_pred"], active_class_names)
    save_confusion_matrix_plot(val_confusion, run_dir / "val_confusion_matrix.png")

    history_df = pd.DataFrame(history_rows)
    save_training_history_plot(
        history_df,
        run_dir / "training_curves.png",
        title="ViT Schizophrenia Severity Classifier — Training History",
    )
    history_df.to_csv(run_dir / "training_history.csv", index=False)

    # ── Final evaluation on held-out test set ─────────────────────────────────
    # Dissertation Section 3.5: test set never used during training or model selection
    logger.info("=" * 70)
    logger.info("Final evaluation on held-out test set")
    logger.info("=" * 70)

    test_final = run_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        train=False,
        return_predictions=True,
    )

    logger.info(f"Test Accuracy:  {test_final['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_final['precision']:.4f}")
    logger.info(f"Test Recall:    {test_final['recall']:.4f}")
    logger.info(f"Test F1:        {test_final['f1']:.4f}")
    logger.info(f"Test Macro F1:  {test_final['macro_f1']:.4f}")

    test_per_class = build_per_class_metrics(test_final["y_true"], test_final["y_pred"], active_class_names)
    logger.info("Test per-class metrics:\n" + test_per_class.to_string(index=False))
    test_per_class.to_csv(run_dir / "test_per_class_metrics.csv", index=False)

    test_confusion = build_confusion_matrix_df(test_final["y_true"], test_final["y_pred"], active_class_names)
    test_confusion.to_csv(run_dir / "test_confusion_matrix.csv")
    save_confusion_matrix_plot(test_confusion, run_dir / "test_confusion_matrix.png")

    test_predictions = evaluate_and_collect_predictions(model, test_loader, device, active_class_names)
    test_predictions.to_csv(run_dir / "test_predictions.csv", index=False)

    # ── Save best model checkpoint ─────────────────────────────────────────────
    checkpoint_path = run_dir / "vit_mandala_classifier_best.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_macro_f1": best_macro_f1,
            "time_mean": time_mean,
            "time_std": time_std,
            "class_names": active_class_names,
            "test_accuracy": test_final["accuracy"],
            "test_macro_f1": test_final["macro_f1"],
        },
        checkpoint_path,
    )
    logger.info(f"Saved best model checkpoint to: {checkpoint_path}")

    # ── Summary report ─────────────────────────────────────────────────────────
    summary_path = run_dir / "model_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Model Configuration\n")
        f.write("=" * 50 + "\n")
        for key, value in config.__dict__.items():
            f.write(f"{key}: {value}\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write("Training Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Best epoch:          {best_epoch}\n")
        f.write(f"Best val loss:       {best_val_loss:.4f}\n")
        f.write(f"Best val macro F1:   {best_macro_f1:.4f}\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write("Test Set Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy:  {test_final['accuracy']:.4f}\n")
        f.write(f"Precision: {test_final['precision']:.4f}\n")
        f.write(f"Recall:    {test_final['recall']:.4f}\n")
        f.write(f"F1:        {test_final['f1']:.4f}\n")
        f.write(f"Macro F1:  {test_final['macro_f1']:.4f}\n")

    logger.info(f"Saved model summary to: {summary_path}")
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
