import torch
import torch.nn as nn
import timm


class MandalaViTClassifier(nn.Module):
    """
    Vision Transformer classifier with optional completion-time fusion.

    Flow:
        image -> ViT backbone -> visual embedding
        visual embedding + completion time -> MLP classifier -> class logits
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224_in21k",
        pretrained: bool = True,
        use_time_feature: bool = True,
        dropout: float = 0.30,
        num_classes: int = 4,
    ):
        super().__init__()

        self.use_time_feature = use_time_feature

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        backbone_dim = self.backbone.num_features
        extra_dim = 1 if use_time_feature else 0

        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim + extra_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor, completion_time: torch.Tensor = None):
        features = self.backbone(images)

        if self.use_time_feature:
            if completion_time is None:
                raise ValueError("completion_time must be provided when use_time_feature=True")

            if completion_time.dim() == 1:
                completion_time = completion_time.unsqueeze(1)

            features = torch.cat([features, completion_time], dim=1)

        logits = self.classifier(features)
        return logits


def freeze_backbone(model: MandalaViTClassifier):
    for param in model.backbone.parameters():
        param.requires_grad = False


def unfreeze_last_blocks(model: MandalaViTClassifier, num_blocks: int = 1):
    for param in model.backbone.parameters():
        param.requires_grad = False

    if hasattr(model.backbone, "blocks"):
        for block in model.backbone.blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    if hasattr(model.backbone, "norm"):
        for param in model.backbone.norm.parameters():
            param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

