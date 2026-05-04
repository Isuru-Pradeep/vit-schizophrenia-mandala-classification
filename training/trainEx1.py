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
    logger = logging.getLogger(f"mandala_classifier_split_{log_path}")
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

    if len(config.class_names) != config.num_classes:
        raise ValueError("class_names length must match num_classes.")

    unique_classes = sorted(df[config.target_col].unique().tolist())
    effective_num_classes = config.get_effective_num_classes()
    if max(unique_classes) >= effective_num_classes:
        raise ValueError(
            f"Found class label {max(unique_classes)} but num_classes={effective_num_classes}."
        )

    return df


def create_splits(df: pd.DataFrame, config: TrainConfig):
    stratify_col = df[config.target_col].astype(int)
    class_counts = stratify_col.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        raise ValueError(
            "Stratified train/val/test split requires at least 2 samples in every class. "
            f"Observed class counts: {class_counts.to_dict()}"
        )

    train_val_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_col,
    )

    relative_val_size = config.val_size / (1.0 - config.test_size)
    train_val_stratify = train_val_df[config.target_col].astype(int)
    train_val_counts = train_val_stratify.value_counts()
    if len(train_val_counts) < 2 or train_val_counts.min() < 2:
        raise ValueError(
            "Validation split is too small to preserve class stratification after creating the test set. "
            f"Observed train/val class counts: {train_val_counts.to_dict()}"
        )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=config.random_state,
        stratify=train_val_stratify,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def normalize_time_column(train_df, val_df, test_df, time_col: str):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    mean = train_df[time_col].astype(float).mean()
    std = train_df[time_col].astype(float).std()

    if std == 0 or pd.isna(std):
        std = 1.0

    for dataframe in [train_df, val_df, test_df]:
        dataframe[time_col] = (
            (dataframe[time_col].astype(float) - mean) / std
        ).astype(float)

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
        return torch.optim.AdamW(
            model.classifier.parameters(),
            lr=config.head_lr,
            weight_decay=config.weight_decay,
        )

    backbone_params = []
    for _, param in model.backbone.named_parameters():
        if param.requires_grad:
            backbone_params.append(param)

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


def save_run_artifacts(output_dir: Path, history_rows, y_true, y_pred, class_names):
    history_df = pd.DataFrame(history_rows)
    history_path = output_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)

    save_training_history_plot(
        history_df,
        output_dir / "training_curves.png",
        title="Classification Training Curves",
    )

    per_class_df = build_per_class_metrics(y_true, y_pred, class_names)
    per_class_path = output_dir / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_path, index=False)

    confusion_df = build_confusion_matrix_df(y_true, y_pred, class_names)
    confusion_csv_path = output_dir / "confusion_matrix.csv"
    confusion_df.to_csv(confusion_csv_path)
    save_confusion_matrix_plot(
        confusion_df,
        output_dir / "confusion_matrix.png",
    )

    return per_class_df


def main():
    config = TrainConfig()
    seed_everything(config.random_state)

    checkpoint_dir = PROJECT_ROOT / config.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    time_stamp_dir = checkpoint_dir / f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    time_stamp_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(time_stamp_dir / "training.log")

    device = choose_device()
    logger.info("Starting classification train/val/test run")
    logger.info(f"Using device: {device}")

    df = load_metadata(config)
    logger.info(f"Loaded {len(df)} samples from class directories in '{config.data_dir}'")
    active_class_names = config.get_active_class_names()
    logger.info(f"Class names: {active_class_names}")
    train_df, val_df, test_df = create_splits(df, config)

    if config.use_time_feature:
        train_df, val_df, test_df, time_mean, time_std = normalize_time_column(
            train_df, val_df, test_df, config.time_col
        )
        logger.info(f"Time normalization -> mean: {time_mean:.4f}, std: {time_std:.4f}")

    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Val size:   {len(val_df)}")
    logger.info(f"Test size:  {len(test_df)}")
    logger.info(f"Model name: {config.model_name}")
    logger.info(f"Saving run outputs to: {time_stamp_dir}")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, config
    )
    logger.info(
        f"Created dataloaders with {len(train_loader)} train batches, "
        f"{len(val_loader)} val batches, and {len(test_loader)} test batches"
    )

    model = build_model(config, device)
    logger.info(f"Backbone feature dimension: {model.backbone.num_features}")

    freeze_backbone(model)
    logger.info("Backbone frozen for head-only training")
    optimizer = build_optimizer(model, config, stage="head_only")
    criterion = build_criterion(train_df, config, device, logger=logger)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    no_improve_count = 0
    history_rows = []

    for epoch in range(config.max_epochs):
        if epoch == config.head_only_epochs:
            logger.info("Switching to partial fine-tuning")
            unfreeze_last_blocks(model, num_blocks=config.unfreeze_last_n_blocks)
            optimizer = build_optimizer(model, config, stage="partial_finetune")
            logger.info(
                f"Unfroze last {config.unfreeze_last_n_blocks} backbone block(s) and rebuilt optimizer"
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
        })

        logger.info(
            f"Epoch [{epoch + 1}/{config.max_epochs}] | "
            f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve_count = 0
            logger.info(
                f"New best model at epoch {best_epoch} with val loss {best_val_loss:.4f}"
            )
        else:
            no_improve_count += 1
            logger.info(f"No validation improvement for {no_improve_count} epoch(s)")

        if no_improve_count >= config.early_stopping_patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    if best_state is None:
        raise RuntimeError("Training finished but no best model state was saved.")

    model.load_state_dict(best_state)

    test_metrics = run_one_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        train=False,
        return_predictions=True,
    )

    logger.info("Best model summary")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info("Test metrics:")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"  F1:        {test_metrics['f1']:.4f}")
    logger.info(f"  Macro F1:  {test_metrics['macro_f1']:.4f}")

    summary_path = time_stamp_dir / "model_summary.txt"
    per_class_df = save_run_artifacts(
        output_dir=time_stamp_dir,
        history_rows=history_rows,
        y_true=test_metrics["y_true"],
        y_pred=test_metrics["y_pred"],
        class_names=active_class_names,
    )
    with open(summary_path, "w") as f:
        f.write("Model Configuration\n")
        f.write("=" * 50 + "\n")
        for key, value in config.__dict__.items():
            f.write(f"{key}: {value}\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write("Best model summary\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best val loss: {best_val_loss:.4f}\n")
        f.write("Test metrics:\n")
        f.write(f"  Accuracy:  {test_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {test_metrics['recall']:.4f}\n")
        f.write(f"  F1:        {test_metrics['f1']:.4f}\n")
        f.write(f"  Macro F1:  {test_metrics['macro_f1']:.4f}\n")
    logger.info(f"Saved model summary to: {summary_path}")
    logger.info("Per-class test metrics:")
    logger.info("\n" + per_class_df.to_string(index=False))

    checkpoint_path = time_stamp_dir / "vit_mandala_classifier_best.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "time_mean": time_mean if config.use_time_feature else None,
            "time_std": time_std if config.use_time_feature else None,
        },
        checkpoint_path,
    )
    logger.info(f"Saved best model to: {checkpoint_path}")

    predictions_df = evaluate_and_collect_predictions(
        model,
        test_loader,
        device,
        active_class_names,
    )
    predictions_path = time_stamp_dir / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Saved test predictions to: {predictions_path}")
    logger.info("Training run completed successfully")


if __name__ == "__main__":
    main()
