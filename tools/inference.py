import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from explainability.attribution import (
    compute_attention_visualization,
    compute_integrated_gradients,
    compute_occlusion_sensitivity,
    compute_saliency_map,
    compute_vit_gradcam,
)
from explainability.io_utils import load_checkpoint_bundle, load_image_tensors, prepare_time_tensor


def choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def log_message(message: str, progress_callback: Optional[Callable[[str], None]] = None):
    print(f"[explainability] {message}")
    if progress_callback is not None:
        progress_callback(message)


def raise_if_cancelled(cancel_callback: Optional[Callable[[], bool]] = None):
    if cancel_callback is not None and cancel_callback():
        raise InterruptedError("Task stopped by user.")


def choose_explainability_workers(num_tasks: int, device: str) -> int:
    if num_tasks <= 1:
        return 1
    if device == "cuda":
        # Keep GPU contention bounded when prediction/explainability overlap.
        return min(2, num_tasks)
    return min(4, num_tasks)


def find_latest_checkpoint_source(checkpoint_root: Path) -> Path:
    run_dirs = sorted(
        [path for path in checkpoint_root.glob("run_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for run_dir in run_dirs:
        if list(run_dir.glob("vit_mandala_classifier_fold_*.pth")):
            return run_dir

    raise FileNotFoundError(
        f"No checkpoint run directory with fold checkpoints was found in: {checkpoint_root}"
    )


def resolve_checkpoint_paths(checkpoint_source: Optional[str] = None) -> List[Path]:
    if checkpoint_source:
        source_path = Path(checkpoint_source).expanduser()
        if not source_path.is_absolute():
            source_path = (PROJECT_ROOT / source_path).resolve()
    else:
        source_path = find_latest_checkpoint_source(PROJECT_ROOT / "checkpoints")

    if source_path.is_file():
        if source_path.suffix.lower() != ".pth":
            raise ValueError(f"Checkpoint file must end with .pth: {source_path}")
        return [source_path]

    if not source_path.exists():
        raise FileNotFoundError(f"Checkpoint source not found: {source_path}")

    fold_paths = sorted(source_path.glob("vit_mandala_classifier_fold_*.pth"))
    if fold_paths:
        return fold_paths

    generic_paths = sorted(source_path.glob("*.pth"))
    if generic_paths:
        return generic_paths

    raise FileNotFoundError(f"No .pth checkpoints found in: {source_path}")


def predict_image(
    image_path: str,
    completion_time: float,
    checkpoint_source: Optional[str] = None,
    device: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> Dict:
    checkpoint_paths = resolve_checkpoint_paths(checkpoint_source)
    device = device or choose_device()

    probabilities = []
    predicted_indices = []
    active_class_names = None
    normalized_times = []
    image_size = None
    model_name = None
    checkpoint_dir = str(Path(checkpoint_paths[0]).parent.resolve()) if checkpoint_paths else None

    log_message(
        f"Found {len(checkpoint_paths)} checkpoint(s). Preparing inference on {device}.",
        progress_callback,
    )

    for checkpoint_index, checkpoint_path in enumerate(checkpoint_paths, start=1):
        raise_if_cancelled(cancel_callback)
        log_message(
            f"Loading checkpoint {checkpoint_index}/{len(checkpoint_paths)}: {checkpoint_path.name}",
            progress_callback,
        )

        model, config, checkpoint = load_checkpoint_bundle(str(checkpoint_path), device)
        model_name = config.model_name

        log_message(
            f"Running prediction with checkpoint {checkpoint_index}/{len(checkpoint_paths)}",
            progress_callback,
        )

        image_tensor, _ = load_image_tensors(image_path, config.image_size, device)
        time_tensor = prepare_time_tensor(
            completion_time=completion_time,
            checkpoint=checkpoint,
            device=device,
            use_time_feature=config.use_time_feature,
        ).unsqueeze(0)

        with torch.no_grad():
            logits = model(image_tensor, time_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        raise_if_cancelled(cancel_callback)

        probabilities.append(probs)
        predicted_indices.append(int(torch.argmax(probs).item()))
        active_class_names = config.get_active_class_names()
        normalized_times.append(float(time_tensor.squeeze(0).item()))
        image_size = config.image_size

    mean_probabilities = torch.stack(probabilities, dim=0).mean(dim=0)
    predicted_index = int(torch.argmax(mean_probabilities).item())

    log_message("Prediction complete.", progress_callback)

    return {
        "image_path": str(Path(image_path).resolve()),
        "completion_time": float(completion_time),
        "predicted_index": predicted_index,
        "predicted_label": active_class_names[predicted_index],
        "class_names": active_class_names,
        "probabilities": {
            class_name: float(mean_probabilities[idx].item())
            for idx, class_name in enumerate(active_class_names)
        },
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "num_checkpoints": len(checkpoint_paths),
        "checkpoint_dir": checkpoint_dir,
        "device": device,
        "model_name": model_name,
        "image_size": image_size,
        "normalized_time_values": normalized_times,
        "per_checkpoint_predictions": predicted_indices,
    }


def score_map_to_pil(score_map: torch.Tensor, image_size: int) -> Image.Image:
    resized = F.interpolate(
        score_map.unsqueeze(0).unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    heat = (resized.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype("uint8")
    return Image.fromarray(heat, mode="L")


def build_overlay_image(base_image: Image.Image, score_map: torch.Tensor, alpha: float = 0.45) -> Image.Image:
    base = base_image.convert("RGB")
    heat = score_map_to_pil(score_map, base.size[1]).resize(base.size).convert("L")

    heat_values = torch.tensor(list(heat.getdata()), dtype=torch.float32).reshape(base.size[1], base.size[0])
    heat_values = heat_values / 255.0

    red = (heat_values * 255.0).to(torch.uint8)
    green = torch.zeros_like(red, dtype=torch.uint8)
    blue = ((1.0 - heat_values) * 180.0).to(torch.uint8)
    overlay = torch.stack([red, green, blue], dim=-1).numpy()

    base_tensor = torch.tensor(list(base.getdata()), dtype=torch.float32).reshape(base.size[1], base.size[0], 3)
    overlay_tensor = torch.tensor(overlay, dtype=torch.float32)
    blended = ((1.0 - alpha) * base_tensor + alpha * overlay_tensor).clamp(0.0, 255.0).to(torch.uint8).numpy()
    return Image.fromarray(blended, mode="RGB")


def generate_explanations(
    image_path: str,
    completion_time: float,
    checkpoint_source: Optional[str] = None,
    device: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    result_callback: Optional[Callable[[str, Dict, Dict], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> Dict:
    checkpoint_paths = resolve_checkpoint_paths(checkpoint_source)
    device = device or choose_device()
    first_checkpoint_path = checkpoint_paths[0]

    log_message(
        f"Preparing ensemble explainability from {len(checkpoint_paths)} checkpoint(s).",
        progress_callback,
    )
    log_message(
        f"Explainability execution context: device={device}, checkpoint_source={first_checkpoint_path.parent.resolve()}",
        progress_callback,
    )

    first_model, first_config, first_checkpoint = load_checkpoint_bundle(str(first_checkpoint_path), device)
    image_tensor, base_image = load_image_tensors(image_path, first_config.image_size, device)
    first_time_tensor = prepare_time_tensor(
        completion_time=completion_time,
        checkpoint=first_checkpoint,
        device=device,
        use_time_feature=first_config.use_time_feature,
    ).unsqueeze(0)

    ensemble_logits = []
    checkpoint_bundles = [
        (first_checkpoint_path, first_model, first_config, first_checkpoint, image_tensor, first_time_tensor)
    ]

    with torch.no_grad():
        ensemble_logits.append(first_model(image_tensor, first_time_tensor).squeeze(0).detach().cpu())

    for checkpoint_path in checkpoint_paths[1:]:
        raise_if_cancelled(cancel_callback)
        log_message(f"Loading model for explainability: {checkpoint_path.name}", progress_callback)
        model, config, checkpoint = load_checkpoint_bundle(str(checkpoint_path), device)
        current_image_tensor, _ = load_image_tensors(image_path, config.image_size, device)
        time_tensor = prepare_time_tensor(
            completion_time=completion_time,
            checkpoint=checkpoint,
            device=device,
            use_time_feature=config.use_time_feature,
        ).unsqueeze(0)
        checkpoint_bundles.append(
            (checkpoint_path, model, config, checkpoint, current_image_tensor, time_tensor)
        )
        with torch.no_grad():
            ensemble_logits.append(model(current_image_tensor, time_tensor).squeeze(0).detach().cpu())

    mean_logits = torch.stack(ensemble_logits, dim=0).mean(dim=0)
    target_class = int(torch.argmax(mean_logits).item())
    run_metadata = {
        "model_name": first_config.model_name,
        "checkpoint_path": str(first_checkpoint_path.resolve()),
        "checkpoint_dir": str(first_checkpoint_path.parent.resolve()),
        "num_checkpoints": len(checkpoint_paths),
        "target_class": target_class,
        "target_class_name": first_config.get_active_class_names()[target_class],
        "image_size": first_config.image_size,
    }
    log_message(
        f"Ensemble target class for explainability: {target_class} ({run_metadata['target_class_name']})",
        progress_callback,
    )

    methods = [
        ("saliency", compute_saliency_map),
        ("gradcam", compute_vit_gradcam),
        ("attention", compute_attention_visualization),
        ("integrated_gradients", compute_integrated_gradients),
        ("occlusion", compute_occlusion_sensitivity),
    ]

    outputs = {}
    errors = {}
    explainability_workers = choose_explainability_workers(len(checkpoint_bundles), device)
    if explainability_workers > 1:
        log_message(
            f"Explainability checkpoint processing will use {explainability_workers} parallel worker(s).",
            progress_callback,
        )
    else:
        log_message("Explainability checkpoint processing will run sequentially.", progress_callback)

    for method_name, method_fn in methods:
        raise_if_cancelled(cancel_callback)
        log_message(f"Generating {method_name} explanation...", progress_callback)

        try:
            def run_checkpoint_method(bundle_payload):
                checkpoint_index, (_, model, _config, _, current_image_tensor, time_tensor) = bundle_payload
                raise_if_cancelled(cancel_callback)
                log_message(
                    f"{method_name}: running checkpoint {checkpoint_index}/{len(checkpoint_bundles)}",
                    progress_callback,
                )
                result = method_fn(
                    model=model,
                    image_tensor=current_image_tensor,
                    completion_time_tensor=time_tensor,
                    target_class=target_class,
                    cancel_callback=cancel_callback,
                )
                raise_if_cancelled(cancel_callback)
                return result["score_map"].detach().cpu(), result["probabilities"].detach().cpu()

            indexed_bundles = list(enumerate(checkpoint_bundles, start=1))
            if explainability_workers > 1:
                with ThreadPoolExecutor(max_workers=explainability_workers) as executor:
                    checkpoint_results = list(executor.map(run_checkpoint_method, indexed_bundles))
            else:
                checkpoint_results = [run_checkpoint_method(bundle_payload) for bundle_payload in indexed_bundles]

            method_score_maps = [score_map for score_map, _ in checkpoint_results]
            method_probabilities = [probabilities for _, probabilities in checkpoint_results]

            mean_score_map = torch.stack(method_score_maps, dim=0).mean(dim=0)
            mean_probabilities = torch.stack(method_probabilities, dim=0).mean(dim=0)
            outputs[method_name] = {
                "target_class": target_class,
                "probabilities": mean_probabilities,
                "grayscale": score_map_to_pil(mean_score_map, first_config.image_size),
                "overlay": build_overlay_image(base_image, mean_score_map),
            }
            if result_callback is not None:
                result_callback(method_name, outputs[method_name], run_metadata)
            log_message(f"{method_name} explanation generated successfully.", progress_callback)
        except Exception as exc:
            if isinstance(exc, InterruptedError):
                log_message(f"{method_name} explanation stopped by user.", progress_callback)
                raise
            errors[method_name] = str(exc)
            log_message(f"{method_name} explanation failed: {exc}", progress_callback)

    log_message("Explainability generation complete.", progress_callback)
    log_message(
        "Explainability summary: "
        f"succeeded={len(outputs)}, failed={len(errors)}, "
        f"successful_methods={sorted(outputs.keys())}, failed_methods={sorted(errors.keys())}",
        progress_callback,
    )

    return {
        **run_metadata,
        "explanations": outputs,
        "errors": errors,
    }
