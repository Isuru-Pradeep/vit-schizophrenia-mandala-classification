from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int, config=None):
    """
    Training augmentation pipeline.

    Dissertation §3.2.4 (Table 3.7): only symmetry-preserving geometric
    transforms are applied — horizontal flip, vertical flip, and 20° rotation.
    These preserve mandala radial structure while increasing diversity.
    """
    rotation_degrees      = 20  if config is None else getattr(config, "rotation_degrees",      20)
    horizontal_flip_prob  = 0.5 if config is None else getattr(config, "horizontal_flip_prob",  0.5)
    vertical_flip_prob    = 0.3 if config is None else getattr(config, "vertical_flip_prob",    0.3)

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(rotation_degrees),
        transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
        transforms.RandomVerticalFlip(p=vertical_flip_prob),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int):
    """No augmentation for validation and test sets (Dissertation Table 3.8)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
