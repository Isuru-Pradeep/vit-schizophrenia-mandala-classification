import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from explainability.attribution import (
    compute_attention_visualization,
    compute_integrated_gradients,
    compute_occlusion_sensitivity,
    compute_saliency_map,
    compute_vit_gradcam,
)
from explainability.io_utils import (
    ensure_dir,
    load_checkpoint_bundle,
    load_image_tensors,
    prepare_time_tensor,
)
from explainability.visualize import save_grayscale_map, save_heatmap_overlay


def parse_args():
    parser = argparse.ArgumentParser(description="Generate explainability artifacts for Mandala ViT.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .pth checkpoint.")
    parser.add_argument("--image-path", required=True, help="Path to the input image.")
    parser.add_argument(
        "--completion-time",
        type=float,
        required=True,
        help="Raw completion time for the sample before normalization.",
    )
    parser.add_argument(
        "--method",
        choices=["saliency", "gradcam", "attention", "integrated_gradients", "occlusion", "all"],
        default="all",
        help="Explainability method to run.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Class index to explain. Defaults to the model prediction.",
    )
    parser.add_argument("--output-dir", default="explainability_outputs", help="Where to save outputs.")
    parser.add_argument("--time-mean", type=float, default=None, help="Override checkpoint time mean.")
    parser.add_argument("--time-std", type=float, default=None, help="Override checkpoint time std.")
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=32,
        help="Number of interpolation steps for integrated gradients.",
    )
    parser.add_argument(
        "--occlusion-patch",
        type=int,
        default=32,
        help="Patch size for occlusion sensitivity.",
    )
    parser.add_argument(
        "--occlusion-stride",
        type=int,
        default=16,
        help="Stride for occlusion sensitivity.",
    )
    return parser.parse_args()


def score_map_to_image_size(score_map: torch.Tensor, image_size):
    resized = F.interpolate(
        score_map.unsqueeze(0).unsqueeze(0),
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze().cpu().numpy()


def save_method_outputs(result, base_image, output_dir: Path, class_names):
    class_idx = result["target_class"]
    class_name = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
    method_name = result["method"]

    map_array = score_map_to_image_size(result["score_map"], base_image.size[::-1])
    overlay_path = output_dir / f"{method_name}_overlay_class_{class_idx}_{class_name}.png"
    grayscale_path = output_dir / f"{method_name}_map_class_{class_idx}_{class_name}.png"
    save_heatmap_overlay(base_image, map_array, overlay_path)
    save_grayscale_map(map_array, grayscale_path)

    probability_payload = {
        class_names[index] if index < len(class_names) else str(index): float(prob)
        for index, prob in enumerate(result["probabilities"].tolist())
    }

    metadata = {
        "method": method_name,
        "target_class": class_idx,
        "target_class_name": class_name,
        "probabilities": probability_payload,
        "overlay_path": str(overlay_path),
        "grayscale_path": str(grayscale_path),
    }

    metadata_path = output_dir / f"{method_name}_summary.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, config, checkpoint = load_checkpoint_bundle(args.checkpoint, device)
    active_class_names = config.get_active_class_names()
    image_tensor, base_image = load_image_tensors(args.image_path, config.image_size, device)
    completion_time_tensor = prepare_time_tensor(
        completion_time=args.completion_time,
        checkpoint=checkpoint,
        device=device,
        use_time_feature=config.use_time_feature,
        override_mean=args.time_mean,
        override_std=args.time_std,
    ).unsqueeze(0)

    output_dir = ensure_dir(args.output_dir)
    methods = [args.method] if args.method != "all" else [
        "saliency",
        "gradcam",
        "attention",
        "integrated_gradients",
        "occlusion",
    ]

    summaries = []
    for method in methods:
        if method == "saliency":
            result = compute_saliency_map(
                model=model,
                image_tensor=image_tensor,
                completion_time_tensor=completion_time_tensor,
                target_class=args.target_class,
            )
        elif method == "gradcam":
            result = compute_vit_gradcam(
                model=model,
                image_tensor=image_tensor,
                completion_time_tensor=completion_time_tensor,
                target_class=args.target_class,
            )
        elif method == "attention":
            result = compute_attention_visualization(
                model=model,
                image_tensor=image_tensor,
                completion_time_tensor=completion_time_tensor,
                target_class=args.target_class,
            )
        elif method == "integrated_gradients":
            result = compute_integrated_gradients(
                model=model,
                image_tensor=image_tensor,
                completion_time_tensor=completion_time_tensor,
                target_class=args.target_class,
                steps=args.ig_steps,
            )
        else:
            result = compute_occlusion_sensitivity(
                model=model,
                image_tensor=image_tensor,
                completion_time_tensor=completion_time_tensor,
                target_class=args.target_class,
                patch_size=args.occlusion_patch,
                stride=args.occlusion_stride,
            )

        summaries.append(save_method_outputs(result, base_image, output_dir, active_class_names))

    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "image_path": args.image_path,
                "completion_time": args.completion_time,
                "device": device,
                "methods": summaries,
            },
            file,
            indent=2,
        )

    print(f"Saved explainability outputs to: {output_dir}")


if __name__ == "__main__":
    main()
