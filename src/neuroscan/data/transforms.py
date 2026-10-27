"""
NeuroScan AI — Image Preprocessing and Augmentation Transforms.
"""
from typing import Tuple

from torchvision import transforms

# ImageNet statistics for standard transfer learning models
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Construct modest, medically reasonable training transform pipeline:
    - Resize to (img_size, img_size)
    - Random horizontal flip (p=0.5) (brain axial symmetry)
    - Small random rotation (+/- 10 degrees) (slight head orientation shifts)
    - Convert to float tensor scaled to [0.0, 1.0]
    - Normalize with ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Construct deterministic evaluation transform pipeline (no random augmentations):
    - Resize to (img_size, img_size)
    - Convert to float tensor scaled to [0.0, 1.0]
    - Normalize with ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

