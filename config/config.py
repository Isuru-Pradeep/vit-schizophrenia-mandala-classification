"""
Configuration for Vision Transformer (ViT)-based Schizophrenia Severity Classification
using Structured Mandala Coloring (SMC) Drawings.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainConfig:
    # -----------------------------
    # Paths
    # -----------------------------
    project_root: str = "."
    data_dir: str = "dataset"
    completion_time_csv: Optional[str] = "dataset.csv"
    checkpoint_dir: str = "checkpoints"

    # -----------------------------
    # Data
    # -----------------------------
    image_col: str = "image_name"  # Mandala drawing filename
    image_path_col: str = "image_path"  # Path to mandala drawing
    target_col: str = "severity_class"  # PANSS-aligned severity label
    sample_id_col: str = "sample_name"  # Participant ID
    time_col: str = "completion_time"  # Task completion time (behavioral indicator)
    image_size: int = 224
    num_workers: int = 0
    val_size: float = 0.333
    test_size: float = 0.25
    random_state: int = 7
    num_classes: int = 4
    class_names: List[str] = field(
        default_factory=lambda: ["healthy", "minor", "medium", "severe"]
    )  # PANSS-aligned severity categories
    include_healthy_samples: bool = True  # Include healthy controls in training

    # -----------------------------
    # Model - Vision Transformer for capturing global mandala structure
    # -----------------------------
    model_name: str = "vit_base_patch16_224_in21k"  # ViT for radial symmetry & global context
    pretrained: bool = True  # Use ImageNet-21K pre-training
    use_time_feature: bool = True  # Late-fusion: completion_time as behavioral indicator
    dropout: float = 0.30

    # -----------------------------
    # Augmentation
    # Dissertation §3.2.4, Table 3.7: symmetry-preserving geometric transforms only
    # (horizontal flip, vertical flip, 20° rotation)
    # -----------------------------
    rotation_degrees:     float = 20.0
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob:   float = 0.3

    # -----------------------------
    # Training
    # -----------------------------
    batch_size: int = 2
    max_epochs: int = 50
    head_only_epochs: int = 10
    early_stopping_patience: int = 5
    use_class_weights: bool = True
    label_smoothing: float = 0.05

    # Learning rates
    head_lr: float = 1e-4
    backbone_lr: float = 1e-7
    weight_decay: float = 0.05

    # Fine-tuning
    unfreeze_last_n_blocks: int = 2

    def get_active_class_names(self) -> List[str]:
        if self.include_healthy_samples:
            return list(self.class_names)
        return [class_name for class_name in self.class_names if class_name != "healthy"]

    def get_effective_num_classes(self) -> int:
        return len(self.get_active_class_names())
