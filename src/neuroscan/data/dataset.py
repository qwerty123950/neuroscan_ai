"""
NeuroScan AI — Brain Tumor MRI Dataset Module.
"""
import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_TO_IDX: Dict[str, int] = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3,
}
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

VALID_SPLITS = ("Training", "Testing")
REQUIRED_MANIFEST_COLUMNS = {
    "rel_path",
    "split",
    "label",
    "file_sha256",
    "pixel_sha256",
    "is_duplicate",
    "is_cross_split_duplicate",
    "eval_status",
    "exclusion_reason",
}


def get_default_project_root() -> Path:
    """Return repository root (four levels up from src/neuroscan/data/dataset.py)."""
    return Path(__file__).resolve().parents[3]


class BrainTumorMRIDataset(Dataset):
    """
    PyTorch Dataset for brain tumor MRI scans driven by the verified dataset manifest.

    Guarantees:
    - Reads only manifest records matching the requested split and eval_status == 'include'.
    - Excludes cross-split leakage images deterministically.
    - Resolves all image paths relative to project root, independent of working directory.
    - Converts all image modes (L, Grayscale, RGBA, etc.) consistently to 3-channel RGB.
    - Performs lazy image loading to preserve memory.
    """

    def __init__(
        self,
        split: str,
        manifest_path: Optional[Union[str, Path]] = None,
        project_root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Expected one of {VALID_SPLITS}."
            )

        self.split = split
        self.project_root = Path(project_root).resolve() if project_root else get_default_project_root()
        self.transform = transform

        if manifest_path is None:
            self.manifest_path = self.project_root / "data" / "processed" / "dataset_manifest.csv"
        else:
            self.manifest_path = Path(manifest_path)
            if not self.manifest_path.is_absolute():
                self.manifest_path = self.project_root / self.manifest_path

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Dataset manifest not found at: {self.manifest_path}. "
                "Ensure scripts/prepare_dataset_manifest.py has been executed."
            )

        self.samples: List[Tuple[Path, int, str]] = []  # (abs_path, label_idx, rel_path)
        self._load_and_filter_manifest()

    def _load_and_filter_manifest(self) -> None:
        """Parse manifest, validate schema and labels, enforce path containment, and filter."""
        raw_root = (self.project_root / "data" / "raw").resolve()

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not REQUIRED_MANIFEST_COLUMNS.issubset(set(reader.fieldnames)):
                missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or [])
                raise ValueError(
                    f"Manifest at {self.manifest_path} is missing required columns: {sorted(list(missing))}"
                )

            for row_idx, row in enumerate(reader, start=2):
                row_split = row["split"]
                label = row["label"]
                eval_status = row["eval_status"]
                rel_path = row["rel_path"]

                if label not in CLASS_TO_IDX:
                    raise ValueError(
                        f"Row {row_idx}: Unknown class label '{label}'. "
                        f"Expected one of {list(CLASS_TO_IDX.keys())}."
                    )

                # Path containment validation: must resolve strictly inside project data/raw/
                path_obj = Path(rel_path)
                abs_path = path_obj if path_obj.is_absolute() else (self.project_root / path_obj)
                resolved_path = abs_path.resolve()

                try:
                    is_contained = resolved_path.is_relative_to(raw_root) and resolved_path != raw_root
                except (ValueError, AttributeError):
                    is_contained = False

                if not is_contained:
                    raise ValueError(
                        f"Row {row_idx}: Path traversal detected for '{rel_path}'. "
                        f"Resolved path '{resolved_path}' is outside dataset directory '{raw_root}'."
                    )

                # Filter: must match requested split and be approved for inclusion
                if row_split == self.split and eval_status == "include":
                    if not resolved_path.is_file():
                        raise FileNotFoundError(
                            f"Row {row_idx}: Image file not found at {resolved_path}"
                        )
                    label_idx = CLASS_TO_IDX[label]
                    self.samples.append((resolved_path, label_idx, rel_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.samples)}.")

        abs_path, label_idx, _ = self.samples[idx]

        try:
            with Image.open(abs_path) as img:
                # Convert all images (grayscale L, RGBA, Palette P) to standard 3-channel RGB
                img_rgb = img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load or decode image at index {idx} ({abs_path}): {exc}"
            ) from exc

        if self.transform is not None:
            image_tensor = self.transform(img_rgb)
        else:
            image_tensor = transforms.functional.to_tensor(img_rgb)

        return image_tensor, label_idx

