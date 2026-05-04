import argparse
import json
import sys
import time
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from explainability.attribution import compute_vit_gradcam
from explainability.io_utils import (
    ensure_dir,
    load_checkpoint_bundle,
    load_image_tensors,
    prepare_time_tensor,
)
from explainability.visualize import save_grayscale_map, save_heatmap_overlay


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Grad-CAM on a trained Mandala ViT checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .pth checkpoint or checkpoint folder.")
    parser.add_argument(
        "--image-path",
        required=True,
        help="Path to a single input image or a directory containing images.",
    )
    parser.add_argument(
        "--completion-time",
        type=float,
        required=True,
        help="Raw completion time for the sample before normalization.",
    )
    parser.add_argument(
        "--output-dir",
        default="gradcam_outputs",
        help="Folder where Grad-CAM images and JSON summary will be saved.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Optional class index to explain. Defaults to the model prediction.",
    )
    parser.add_argument("--device", default=None, help="Force a device such as cpu or cuda.")
    parser.add_argument("--time-mean", type=float, default=None, help="Override checkpoint time mean.")
    parser.add_argument("--time-std", type=float, default=None, help="Override checkpoint time std.")
    parser.add_argument(
        "--upscale-factor",
        type=int,
        default=1,
        help="Upscale factor for saved Grad-CAM images. Use 1 to keep original image resolution.",
    )
    return parser.parse_args()


def choose_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_single_checkpoint(checkpoint_source: str) -> Path:
    source_path = Path(checkpoint_source).expanduser()
    if not source_path.is_absolute():
        source_path = (PROJECT_ROOT / source_path).resolve()

    if source_path.is_file():
        if source_path.suffix.lower() != ".pth":
            raise ValueError(f"Checkpoint file must end with .pth: {source_path}")
        return source_path

    if not source_path.exists():
        raise FileNotFoundError(f"Checkpoint source not found: {source_path}")

    fold_paths = sorted(source_path.glob("vit_mandala_classifier_fold_*.pth"))
    if fold_paths:
        return max(fold_paths, key=lambda path: path.stat().st_mtime)

    generic_paths = sorted(source_path.glob("*.pth"))
    if generic_paths:
        return max(generic_paths, key=lambda path: path.stat().st_mtime)

    raise FileNotFoundError(f"No .pth checkpoints found in: {source_path}")


def resolve_image_source(image_source: str) -> tuple[Path, list[Path]]:
    source_path = Path(image_source).expanduser()
    if not source_path.is_absolute():
        source_path = (PROJECT_ROOT / source_path).resolve()

    if source_path.is_file():
        if source_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file type: {source_path}")
        return source_path.parent, [source_path]

    if not source_path.exists():
        raise FileNotFoundError(f"Image source not found: {source_path}")

    image_paths = [
        path
        for path in sorted(source_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {source_path}")

    return source_path, image_paths


def build_image_output_dir(output_dir: Path, image_path: Path, image_root: Path) -> Path:
    try:
        relative_path = image_path.relative_to(image_root)
        return output_dir / relative_path.with_suffix("")
    except ValueError:
        return output_dir / image_path.stem


def process_single_image(
    image_path: Path,
    checkpoint_path: Path,
    model,
    config,
    checkpoint,
    output_dir: Path,
    image_root: Path,
    device: str,
    completion_time: float,
    target_class: Optional[int],
    time_mean: Optional[float],
    time_std: Optional[float],
    upscale_factor: int,
):
    inference_started = time.perf_counter()
    image_tensor, base_image = load_image_tensors(str(image_path), config.image_size, device)
    original_image = Image.open(image_path).convert("RGB")
    completion_time_tensor = prepare_time_tensor(
        completion_time=completion_time,
        checkpoint=checkpoint,
        device=device,
        use_time_feature=config.use_time_feature,
        override_mean=time_mean,
        override_std=time_std,
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor, completion_time_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    predicted_index = int(torch.argmax(probabilities).item())
    class_names = config.get_active_class_names()
    predicted_label = class_names[predicted_index]
    prediction_seconds = time.perf_counter() - inference_started

    gradcam_started = time.perf_counter()
    gradcam_result = compute_vit_gradcam(
        model=model,
        image_tensor=image_tensor,
        completion_time_tensor=completion_time_tensor,
        target_class=target_class,
    )
    gradcam_seconds = time.perf_counter() - gradcam_started

    resized_score_map = F.interpolate(
        gradcam_result["score_map"].unsqueeze(0).unsqueeze(0),
        size=base_image.size[::-1],
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()

    upscale_factor = max(1, int(upscale_factor))
    display_size = (original_image.width * upscale_factor, original_image.height * upscale_factor)
    display_base_image = original_image.resize(display_size, resample=Image.Resampling.LANCZOS)
    display_score_map = F.interpolate(
        torch.from_numpy(resized_score_map).unsqueeze(0).unsqueeze(0),
        size=display_size[::-1],
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()

    image_output_dir = build_image_output_dir(output_dir, image_path, image_root)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    mirrored_image_path = image_output_dir / image_path.name
    shutil.copy2(image_path, mirrored_image_path)

    target_class_index = int(gradcam_result["target_class"])
    target_class_name = (
        class_names[target_class_index] if target_class_index < len(class_names) else str(target_class_index)
    )

    overlay_path = image_output_dir / f"gradcam_overlay_class_{target_class_index}_{target_class_name}.png"
    grayscale_path = image_output_dir / f"gradcam_map_class_{target_class_index}_{target_class_name}.png"
    save_heatmap_overlay(display_base_image, display_score_map, overlay_path)
    save_grayscale_map(display_score_map, grayscale_path)

    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "image_path": str(image_path.resolve()),
        "completion_time": float(completion_time),
        "device": device,
        "model_name": config.model_name,
        "use_time_feature": bool(config.use_time_feature),
        "predicted_index": predicted_index,
        "predicted_label": predicted_label,
        "target_class": target_class_index,
        "target_class_name": target_class_name,
        "probabilities": {
            class_names[index] if index < len(class_names) else str(index): float(probability)
            for index, probability in enumerate(probabilities.tolist())
        },
        "overlay_path": str(overlay_path),
        "grayscale_path": str(grayscale_path),
        "mirrored_image_path": str(mirrored_image_path),
        "upscale_factor": upscale_factor,
        "output_image_size": [display_size[0], display_size[1]],
        "source_image_size": [original_image.width, original_image.height],
        "prediction_seconds": prediction_seconds,
        "gradcam_seconds": gradcam_seconds,
        "total_seconds": prediction_seconds + gradcam_seconds,
    }

    with open(image_output_dir / "gradcam_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


def main():
    args = parse_args()
    device = choose_device(args.device)
    checkpoint_path = resolve_single_checkpoint(args.checkpoint)
    image_root, image_paths = resolve_image_source(args.image_path)

    model, config, checkpoint = load_checkpoint_bundle(str(checkpoint_path), device)
    output_dir = ensure_dir(args.output_dir)
    all_summaries = []

    for image_path in image_paths:
        summary = process_single_image(
            image_path=image_path,
            checkpoint_path=checkpoint_path,
            model=model,
            config=config,
            checkpoint=checkpoint,
            output_dir=output_dir,
            image_root=image_root,
            device=device,
            completion_time=args.completion_time,
            target_class=args.target_class,
            time_mean=args.time_mean,
            time_std=args.time_std,
            upscale_factor=args.upscale_factor,
        )
        all_summaries.append(summary)
        print(json.dumps(summary, indent=2))

    with open(output_dir / "gradcam_batch_summary.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "image_source": str(Path(args.image_path).resolve()),
                "completion_time": float(args.completion_time),
                "device": device,
                "num_images": len(all_summaries),
                "results": all_summaries,
            },
            file,
            indent=2,
        )

    print(f"Saved Grad-CAM outputs for {len(all_summaries)} image(s) to: {output_dir}")


if __name__ == "__main__":
    main()