from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from torchvision import transforms

from config.config import TrainConfig
from models.vit_classifier import MandalaViTClassifier


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_checkpoint_file(checkpoint_path: str, device: str):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)
    except Exception:
        # Trusted checkpoints created by this project may contain objects that are
        # not yet supported by weights-only loading. Fall back to the standard
        # loader so inference and explainability still work.
        return torch.load(checkpoint_path, map_location=device, weights_only=False)


def _normalize_model_name(model_name: str) -> str:
    deprecated_names = {
        "vit_base_patch16_224_in21k": "vit_base_patch16_224.augreg_in21k",
    }
    return deprecated_names.get(model_name, model_name)


def load_checkpoint_bundle(checkpoint_path: str, device: str):
    checkpoint = _load_checkpoint_file(checkpoint_path, device)
    checkpoint_config = checkpoint.get("config", {})
    config = TrainConfig()

    for key, value in checkpoint_config.items():
        if hasattr(config, key):
            setattr(config, key, value)

    model = MandalaViTClassifier(
        model_name=_normalize_model_name(config.model_name),
        pretrained=False,
        use_time_feature=config.use_time_feature,
        dropout=config.dropout,
        num_classes=config.get_effective_num_classes(),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config, checkpoint


def build_eval_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image_tensors(image_path: str, image_size: int, device: str) -> Tuple[torch.Tensor, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    resized = image.resize((image_size, image_size))
    image_tensor = build_eval_transform(image_size)(image).unsqueeze(0).to(device)
    return image_tensor, resized


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    image = image_tensor.detach().cpu().clone().squeeze(0)
    for channel_idx, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        image[channel_idx] = image[channel_idx] * std + mean
    return image.clamp(0.0, 1.0)


def prepare_time_tensor(
    completion_time: float,
    checkpoint: Dict,
    device: str,
    use_time_feature: bool,
    override_mean: float = None,
    override_std: float = None,
):
    if not use_time_feature:
        return torch.tensor([0.0], dtype=torch.float32, device=device)

    time_mean = checkpoint.get("time_mean", override_mean)
    time_std = checkpoint.get("time_std", override_std)

    if time_mean is None or time_std is None:
        raise ValueError(
            "Completion-time normalization stats were not found in the checkpoint. "
            "Provide --time-mean and --time-std."
        )

    if float(time_std) == 0.0:
        time_std = 1.0

    normalized = (float(completion_time) - float(time_mean)) / float(time_std)
    return torch.tensor([normalized], dtype=torch.float32, device=device)


def ensure_dir(path: str):
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
