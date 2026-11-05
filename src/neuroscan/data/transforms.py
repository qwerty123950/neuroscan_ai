"""
NeuroScan AI — Image Preprocessing and Augmentation Transforms.

Implements preprocessing and augmentation pipelines conforming to Table 1 of:
Gómez-Guzmán et al., "Classifying Brain Tumors on Magnetic Resonance Imaging
by Using Convolutional Neural Networks", Electronics 2023, 12(4), 955.

Paper Preprocessing Specifications (Table 1):
- Rescale: 1.0 / 255 (scale pixel intensities to [0.0, 1.0])
- Rotation: 10 degrees
- Horizontal flip: True
- Shear: 0.1 (10% / 10 degrees)
- Zoom / Scaling: 0.1 (scale range [0.9, 1.1])
- Brightness range: 0.0 to 0.7 (or color jitter)
- Input dimensions: 256x256 for Generic CNN (Figure 2)
"""
import random
from typing import Optional, Tuple

from PIL import ImageEnhance
from torchvision import transforms

# ImageNet statistics for optional transfer learning models
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class PaperLiteralBrightness:
    """
    Literal brightness transform from Gómez-Guzmán et al. (2023) Table 1.
    Replicates Keras ImageDataGenerator(brightness_range=[0.0, 0.7]).
    Multiplies pixel intensities by a factor uniformly sampled from [min_factor, max_factor].
    """

    def __init__(self, min_factor: float = 0.0, max_factor: float = 0.7) -> None:
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, img):
        factor = random.uniform(self.min_factor, self.max_factor)
        return ImageEnhance.Brightness(img).enhance(factor)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_factor={self.min_factor}, max_factor={self.max_factor})"


def get_train_transforms(
    img_size: int = 256,
    normalize_mode: str = "rescale_only",
    brightness_mode: str = "paper_literal",
) -> transforms.Compose:
    """
    Construct training transform pipeline matching the paper's specification:
    - Resize to (img_size, img_size) (default 256 for Generic CNN)
    - Random horizontal flip (p=0.5)
    - Random rotation (+/- 10 degrees)
    - Random affine transform (shear +/- 10 degrees, zoom/scale [0.9, 1.1])
    - Brightness mode:
        'paper_literal': Table 1 literal range [0.0, 0.7] (pure darkening)
        'safe_jitter': Project-specific deviation using ColorJitter(0.3) -> [0.7, 1.3]
        'none': No brightness augmentation
    - Convert to float tensor scaled to [0.0, 1.0] (1.0 / 255 rescaling)
    - Optional ImageNet normalization (for future transfer learning backbones)
    """
    valid_brightness_modes = {"paper_literal", "safe_jitter", "none"}
    if brightness_mode not in valid_brightness_modes:
        raise ValueError(
            f"Invalid brightness_mode: '{brightness_mode}'. "
            f"Expected one of: {sorted(valid_brightness_modes)}."
        )

    transform_list = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1), shear=10),
    ]

    if brightness_mode == "paper_literal":
        transform_list.append(PaperLiteralBrightness(min_factor=0.0, max_factor=0.7))
    elif brightness_mode == "safe_jitter":
        transform_list.append(transforms.ColorJitter(brightness=0.3))
    elif brightness_mode == "none":
        pass

    transform_list.append(transforms.ToTensor())

    if normalize_mode == "imagenet":
        transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return transforms.Compose(transform_list)


def get_eval_transforms(
    img_size: int = 256,
    normalize_mode: str = "rescale_only",
) -> transforms.Compose:
    """
    Construct deterministic evaluation transform pipeline (no random augmentations):
    - Resize to (img_size, img_size) (default 256)
    - Convert to float tensor scaled to [0.0, 1.0] (1.0 / 255 rescaling)
    - Optional ImageNet normalization
    """
    transform_list = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]

    if normalize_mode == "imagenet":
        transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return transforms.Compose(transform_list)
