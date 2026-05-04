from pathlib import Path

import numpy as np
from PIL import Image


def save_heatmap_overlay(base_image: Image.Image, score_map, output_path: Path, alpha: float = 0.45):
    base = np.asarray(base_image.convert("RGB"), dtype=np.float32)
    heat = np.asarray(score_map, dtype=np.float32)

    heat = np.clip(heat, 0.0, 1.0)
    red = (heat * 255.0).astype(np.uint8)
    green = np.zeros_like(red, dtype=np.uint8)
    blue = ((1.0 - heat) * 180.0).astype(np.uint8)
    overlay = np.stack([red, green, blue], axis=-1).astype(np.float32)

    blended = (1.0 - alpha) * base + alpha * overlay
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    Image.fromarray(blended).save(output_path)


def save_grayscale_map(score_map, output_path: Path):
    heat = np.asarray(score_map, dtype=np.float32)
    heat = np.clip(heat, 0.0, 1.0)
    heat = (heat * 255.0).astype(np.uint8)
    Image.fromarray(heat, mode="L").save(output_path)

