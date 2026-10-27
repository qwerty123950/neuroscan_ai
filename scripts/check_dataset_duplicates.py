"""
NeuroScan AI — Dataset Duplicate & Data Leakage Audit Script.

This script performs a read-only audit of the raw Brain Tumor MRI dataset:
1. Exact File Duplicates: SHA-256 of file binary contents.
2. Decoded Pixel Duplicates: SHA-256 of decoded raw pixel buffers (size, mode, bytes)
   to detect identical images saved with different headers or compression.
3. Cross-Split Duplicates (Data Leakage): Images appearing in both Training and Testing.
4. Cross-Class Duplicates (Label Noise): Identical images labeled with conflicting classes.
5. Intra-Split Duplicates: Redundant images solely within Training or Testing.

Memory efficient: Processes images iteratively in a streaming fashion.
Deterministic: All scans and reports follow sorted path ordering.
Safe: Does not alter, move, or delete any files.
"""

from collections import defaultdict
import hashlib
from pathlib import Path
import sys
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image

SPLITS = ["Training", "Testing"]
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 digest of a file in chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_pixel_sha256(file_path: Path) -> Tuple[Optional[str], Optional[Tuple[int, int]], Optional[str], Optional[str]]:
    """
    Open an image and compute SHA-256 of its uncompressed decoded pixel buffer.
    Returns: (pixel_hash, size, mode, error_message)
    """
    try:
        with Image.open(file_path) as img:
            size = img.size
            mode = img.mode
            hasher = hashlib.sha256()
            # Prefix with dimensions and mode to avoid shape collisions
            hasher.update(f"{size[0]}x{size[1]}_{mode}:".encode("ascii"))
            hasher.update(img.tobytes())
            return hasher.hexdigest(), size, mode, None
    except Exception as exc:
        return None, None, None, str(exc)


def analyze_duplicate_groups(
    hash_groups: Dict[str, List[dict]],
    hash_type_label: str
) -> dict:
    """
    Analyze duplicate groups for cross-split, cross-class, and intra-split occurrences.
    """
    dup_groups = {h: items for h, items in hash_groups.items() if len(items) > 1}
    total_dup_groups = len(dup_groups)
    total_dup_files = sum(len(items) for items in dup_groups.values())

    cross_split_groups: List[Tuple[str, List[dict]]] = []
    cross_class_groups: List[Tuple[str, List[dict]]] = []
    training_only_groups: List[Tuple[str, List[dict]]] = []
    testing_only_groups: List[Tuple[str, List[dict]]] = []

    for h, items in sorted(dup_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        splits = {item["split"] for item in items}
        labels = {item["label"] for item in items}

        if len(splits) > 1:
            cross_split_groups.append((h, items))
        elif "Training" in splits:
            training_only_groups.append((h, items))
        else:
            testing_only_groups.append((h, items))

        if len(labels) > 1:
            cross_class_groups.append((h, items))

    return {
        "label": hash_type_label,
        "total_groups": total_dup_groups,
        "total_files": total_dup_files,
        "cross_split_groups": cross_split_groups,
        "cross_class_groups": cross_class_groups,
        "training_only_groups": training_only_groups,
        "testing_only_groups": testing_only_groups,
        "groups": dup_groups,
    }


def print_group_samples(groups: List[Tuple[str, List[dict]]], title: str, max_groups: int = 5) -> None:
    """Print readable sample groups of duplicates."""
    print(f"\n--- {title} (Showing up to {max_groups} of {len(groups)} groups) ---")
    if not groups:
        print("  None detected.")
        return

    for idx, (h, items) in enumerate(groups[:max_groups], start=1):
        splits_str = ", ".join(sorted({item['split'] for item in items}))
        labels_str = ", ".join(sorted({item['label'] for item in items}))
        print(f"Group {idx} [Hash: {h[:12]}...] ({len(items)} copies | Splits: {splits_str} | Labels: {labels_str}):")
        for item in items:
            print(f"    - {item['rel_path']}")


def run_audit(data_dir: Path) -> None:
    """Main audit execution."""
    print("=" * 75)
    print("NeuroScan AI — Dataset Duplicate & Data-Leakage Audit")
    print("=" * 75)
    print(f"Target directory: {data_dir}")

    if not data_dir.exists():
        print(f"ERROR: Dataset directory '{data_dir}' does not exist.")
        sys.exit(1)

    file_hash_map: Dict[str, List[dict]] = defaultdict(list)
    pixel_hash_map: Dict[str, List[dict]] = defaultdict(list)
    unreadable: List[Tuple[str, str]] = []

    total_scanned = 0

    print("\nScanning images and calculating SHA-256 hashes (streaming)...")

    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            print(f"Warning: Split directory {split_dir} missing.")
            continue

        for cls_name in CLASSES:
            cls_dir = split_dir / cls_name
            if not cls_dir.is_dir():
                print(f"Warning: Class directory {cls_dir} missing.")
                continue

            file_list = sorted([p for p in cls_dir.iterdir() if p.is_file() and p.name != ".gitkeep"])

            for p in file_list:
                total_scanned += 1
                rel_path = p.relative_to(data_dir)

                # 1. Exact file hash
                f_hash = compute_file_sha256(p)

                # 2. Decoded pixel hash
                pix_hash, size, mode, err = compute_pixel_sha256(p)

                if err:
                    unreadable.append((str(rel_path), err))
                    continue

                record = {
                    "rel_path": str(rel_path),
                    "split": split,
                    "label": cls_name,
                    "file_hash": f_hash,
                    "pixel_hash": pix_hash,
                    "size": size,
                    "mode": mode,
                }

                file_hash_map[f_hash].append(record)
                pixel_hash_map[pix_hash].append(record)

    print(f"Completed scan of {total_scanned} images.")
    if unreadable:
        print(f"Encountered {len(unreadable)} unreadable files during scan.")

    # Analyze Exact File Duplicates
    file_results = analyze_duplicate_groups(file_hash_map, "Exact File Duplicates")

    # Analyze Decoded Pixel Duplicates
    pixel_results = analyze_duplicate_groups(pixel_hash_map, "Decoded Pixel Content Duplicates")

    # 1. Summary Overview
    print("\n" + "=" * 75)
    print("1. Executive Summary")
    print("=" * 75)
    print(f"{'Audit Metric':<40} {'Exact Files':<16} {'Decoded Pixels':<16}")
    print("-" * 72)
    print(f"{'Total duplicate groups':<40} {file_results['total_groups']:<16} {pixel_results['total_groups']:<16}")
    print(f"{'Total affected images':<40} {file_results['total_files']:<16} {pixel_results['total_files']:<16}")
    print(
        f"{'Cross-Split groups (Data Leakage)':<40} "
        f"{len(file_results['cross_split_groups']):<16} "
        f"{len(pixel_results['cross_split_groups']):<16}"
    )
    print(
        f"{'Cross-Class groups (Label Noise)':<40} "
        f"{len(file_results['cross_class_groups']):<16} "
        f"{len(pixel_results['cross_class_groups']):<16}"
    )
    print(
        f"{'Training-only duplicate groups':<40} "
        f"{len(file_results['training_only_groups']):<16} "
        f"{len(pixel_results['training_only_groups']):<16}"
    )
    print(
        f"{'Testing-only duplicate groups':<40} "
        f"{len(file_results['testing_only_groups']):<16} "
        f"{len(pixel_results['testing_only_groups']):<16}"
    )

    # 2. Detailed Cross-Split Leakage Report
    print("\n" + "=" * 75)
    print("2. Cross-Split Data Leakage Audit (Training <-> Testing)")
    print("=" * 75)
    if pixel_results["cross_split_groups"]:
        print(f"WARNING: Detected {len(pixel_results['cross_split_groups'])} duplicate groups spanning both Training and Testing!")
        # Count testing files that have identical twins in training
        leaked_test_count = 0
        for _, items in pixel_results["cross_split_groups"]:
            leaked_test_count += sum(1 for it in items if it["split"] == "Testing")
        print(f"Total testing images leaking information from training: {leaked_test_count} / 1600 ({leaked_test_count/1600*100:.2f}%)")
        print_group_samples(pixel_results["cross_split_groups"], "Representative Cross-Split Duplicates", max_groups=6)
    else:
        print("PASS: Zero cross-split duplicate groups detected. No data leakage between Training and Testing.")

    # 3. Detailed Cross-Class Label Conflict Report
    print("\n" + "=" * 75)
    print("3. Cross-Class Duplicate Audit (Label Conflict / Inconsistency)")
    print("=" * 75)
    if pixel_results["cross_class_groups"]:
        print(f"WARNING: Detected {len(pixel_results['cross_class_groups'])} duplicate groups assigned to conflicting classes!")
        print_group_samples(pixel_results["cross_class_groups"], "Representative Cross-Class Conflicting Duplicates", max_groups=6)
    else:
        print("PASS: Zero cross-class duplicate groups detected. No conflicting class assignments.")

    # 4. Intra-Split Redundancy Report
    print("\n" + "=" * 75)
    print("4. Intra-Split Redundancy Audit")
    print("=" * 75)
    print(f"Training-only duplicate groups : {len(pixel_results['training_only_groups'])}")
    if pixel_results["training_only_groups"]:
        print_group_samples(pixel_results["training_only_groups"], "Representative Training-Only Duplicates", max_groups=4)

    print(f"\nTesting-only duplicate groups  : {len(pixel_results['testing_only_groups'])}")
    if pixel_results["testing_only_groups"]:
        print_group_samples(pixel_results["testing_only_groups"], "Representative Testing-Only Duplicates", max_groups=4)

    # 5. Engineering Conclusion
    print("\n" + "=" * 75)
    print("5. Engineering Conclusion & Audit Recommendations")
    print("=" * 75)
    if pixel_results["cross_split_groups"]:
        print("CRITICAL FINDING: Train/Test leakage exists in the raw dataset.")
        print("Recommendation: Implement a dataset decontamination / deduplication split in preprocessing")
        print("so that test evaluations remain strictly unbiased and independent.")
    else:
        print("POSITIVE FINDING: Train/Test split is mathematically clean from leakage.")

    if pixel_results["cross_class_groups"]:
        print("NOTE: Cross-class duplicate images must be inspected or quarantined to remove label noise.")

    print("\nAudit complete. All dataset files remain strictly read-only and unaltered.")
    print("=" * 75)


if __name__ == "__main__":
    raw_data = get_project_root() / "data" / "raw"
    run_audit(raw_data)
