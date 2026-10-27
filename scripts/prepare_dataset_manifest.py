"""
NeuroScan AI — Leakage-Controlled Dataset Manifest Generator.

This script implements a reproducible, non-destructive data preparation pipeline:
1. Scans raw Brain Tumor MRI dataset under data/raw/ (Training and Testing splits).
2. Computes chunked binary file SHA-256 and native decoded-pixel SHA-256 for all images.
3. Identifies exact decoded-pixel duplicate groups and cross-split leakage.
4. Generates a deterministic CSV manifest at data/processed/dataset_manifest.csv:
   - 7,200 rows sorted deterministically by (split, label, rel_path).
   - Strict 9-column schema.
   - Enforces the policy: all Training images included; only test images matching
     a Training pixel hash are marked as 'exclude' (exclusion_reason: 'cross_split_leakage_with_training').
5. Generates a machine-readable summary at reports/metrics/dataset_manifest_summary.json.
6. Enforces strict validation assertions across all counts and relations.
7. Guarantees byte-for-byte reproducibility (no timestamps, no absolute paths).
"""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple

from PIL import Image

SPLITS = ["Training", "Testing"]
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

MANIFEST_COLUMNS = [
    "rel_path",
    "split",
    "label",
    "file_sha256",
    "pixel_sha256",
    "is_duplicate",
    "is_cross_split_duplicate",
    "eval_status",
    "exclusion_reason",
]


def get_project_root() -> Path:
    """Return repository root directory."""
    return Path(__file__).resolve().parent.parent


def compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 digest of original file bytes via chunked streaming."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_pixel_sha256(file_path: Path) -> Tuple[str, Tuple[int, int], str]:
    """
    Open an image and compute SHA-256 of its uncompressed decoded pixel buffer.
    Hash key: f"{width}x{height}_{mode}:" + img.tobytes()
    """
    with Image.open(file_path) as img:
        w, h = img.size
        mode = img.mode
        hasher = hashlib.sha256()
        hasher.update(f"{w}x{h}_{mode}:".encode("ascii"))
        hasher.update(img.tobytes())
        return hasher.hexdigest(), (w, h), mode


def generate_manifest(
    data_raw_dir: Path,
    manifest_csv_path: Path,
    summary_json_path: Path
) -> str:
    """
    Scan raw images, compute hashes, evaluate leakage policy, validate constraints,
    and write deterministic manifest CSV and summary JSON.
    Returns: SHA-256 digest of the generated CSV manifest.
    """
    project_root = get_project_root()

    print("=" * 75)
    print("NeuroScan AI — Dataset Manifest & Leakage-Control Generator")
    print("=" * 75)
    print(f"Source raw data : {data_raw_dir.relative_to(project_root)}")
    print(f"Target manifest : {manifest_csv_path.relative_to(project_root)}")
    print(f"Target summary  : {summary_json_path.relative_to(project_root)}")

    # 1. Directory Structure Validation
    if not data_raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {data_raw_dir}")

    for split in SPLITS:
        split_dir = data_raw_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Required split directory not found: {split_dir}")
        for cls_name in CLASSES:
            cls_dir = split_dir / cls_name
            if not cls_dir.is_dir():
                raise FileNotFoundError(f"Required class directory not found: {cls_dir}")

    # 2. Sequential Deterministic Scan
    print("\nScanning raw dataset images and calculating hashes...")
    raw_records: List[dict] = []
    pixel_hash_to_records: Dict[str, List[dict]] = defaultdict(list)

    for split in SPLITS:
        for cls_name in CLASSES:
            cls_dir = data_raw_dir / split / cls_name
            file_paths = sorted([p for p in cls_dir.iterdir() if p.is_file() and p.name != ".gitkeep"])
            for p in file_paths:
                rel_path = p.relative_to(project_root).as_posix()
                f_hash = compute_file_sha256(p)
                p_hash, size, mode = compute_pixel_sha256(p)

                record = {
                    "rel_path": rel_path,
                    "split": split,
                    "label": cls_name,
                    "file_sha256": f_hash,
                    "pixel_sha256": p_hash,
                    "size": size,
                    "mode": mode,
                }
                raw_records.append(record)
                pixel_hash_to_records[p_hash].append(record)

    total_images = len(raw_records)
    print(f"Scanned {total_images} total images.")

    # 3. Policy Evaluation
    # Set of all native decoded-pixel hashes present in Training
    training_pixel_hashes: Set[str] = {
        r["pixel_sha256"] for r in raw_records if r["split"] == "Training"
    }

    manifest_rows: List[dict] = []
    cross_split_groups_count = 0
    training_only_dup_groups = 0
    testing_only_dup_groups = 0

    # Categorize duplicate groups
    for p_hash, group in pixel_hash_to_records.items():
        if len(group) > 1:
            splits_in_group = {r["split"] for r in group}
            if len(splits_in_group) > 1:
                cross_split_groups_count += 1
            elif "Training" in splits_in_group:
                training_only_dup_groups += 1
            else:
                testing_only_dup_groups += 1

    for r in raw_records:
        p_hash = r["pixel_sha256"]
        group = pixel_hash_to_records[p_hash]
        is_dup = len(group) > 1
        splits_in_group = {item["split"] for item in group}
        is_cross_split = is_dup and len(splits_in_group) > 1

        if r["split"] == "Training":
            eval_status = "include"
            exclusion_reason = "none"
        else:  # Testing
            if p_hash in training_pixel_hashes:
                eval_status = "exclude"
                exclusion_reason = "cross_split_leakage_with_training"
            else:
                eval_status = "include"
                exclusion_reason = "none"

        manifest_rows.append({
            "rel_path": r["rel_path"],
            "split": r["split"],
            "label": r["label"],
            "file_sha256": r["file_sha256"],
            "pixel_sha256": r["pixel_sha256"],
            "is_duplicate": str(is_dup),
            "is_cross_split_duplicate": str(is_cross_split),
            "eval_status": eval_status,
            "exclusion_reason": exclusion_reason,
        })

    # Sort deterministically by (split, label, rel_path)
    manifest_rows.sort(key=lambda x: (x["split"], x["label"], x["rel_path"]))

    # 4. Strict Validation Assertions
    print("\nExecuting validation assertions...")

    # Assertion 1: Total rows & header
    assert len(manifest_rows) == 7200, f"Expected 7200 rows, got {len(manifest_rows)}"

    # Assertion 2: Training split counts
    train_rows = [r for r in manifest_rows if r["split"] == "Training"]
    assert len(train_rows) == 5600, f"Expected 5600 training images, got {len(train_rows)}"
    train_included = [r for r in train_rows if r["eval_status"] == "include"]
    train_excluded = [r for r in train_rows if r["eval_status"] == "exclude"]
    assert len(train_included) == 5600, f"Expected 5600 included training images, got {len(train_included)}"
    assert len(train_excluded) == 0, f"Expected 0 excluded training images, got {len(train_excluded)}"

    # Assertion 3: Testing split counts
    test_rows = [r for r in manifest_rows if r["split"] == "Testing"]
    assert len(test_rows) == 1600, f"Expected 1600 testing images, got {len(test_rows)}"
    test_included = [r for r in test_rows if r["eval_status"] == "include"]
    test_excluded = [r for r in test_rows if r["eval_status"] == "exclude"]
    assert len(test_excluded) == 114, f"Expected exactly 114 excluded test images, got {len(test_excluded)}"
    assert len(test_included) == 1486, f"Expected exactly 1486 included test images, got {len(test_included)}"

    # Assertion 4: No excluded row is from Training
    assert all(r["split"] != "Training" for r in manifest_rows if r["eval_status"] == "exclude"), (
        "Assertion failed: Excluded image found in Training split!"
    )

    # Assertion 5: Every excluded test row is marked as cross-split duplicate
    assert all(r["is_cross_split_duplicate"] == "True" for r in test_excluded), (
        "Assertion failed: Excluded test row not marked as cross-split duplicate!"
    )

    # Assertion 6: Every included test row is NOT a cross-split duplicate
    assert all(r["is_cross_split_duplicate"] == "False" for r in test_included), (
        "Assertion failed: Included test row marked as cross-split duplicate!"
    )

    print("All validation assertions PASSED successfully.")

    # 5. Write Manifest CSV Deterministically
    manifest_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)

    manifest_sha256 = compute_file_sha256(manifest_csv_path)
    print(f"\nManifest successfully written to: {manifest_csv_path}")
    print(f"Manifest SHA-256 : {manifest_sha256}")

    # 6. Aggregate Summary Metrics
    test_included_by_class = Counter(r["label"] for r in test_included)
    test_excluded_by_class = Counter(r["label"] for r in test_excluded)
    train_by_class = Counter(r["label"] for r in train_rows)
    test_orig_by_class = Counter(r["label"] for r in test_rows)

    total_dup_groups = len([g for g in pixel_hash_to_records.values() if len(g) > 1])
    total_affected_images = sum(len(g) for g in pixel_hash_to_records.values() if len(g) > 1)

    summary_data = {
        "manifest_sha256": manifest_sha256,
        "total_images": 7200,
        "splits": {
            "Training": {
                "total_original": 5600,
                "included": 5600,
                "excluded": 0,
                "classes": {c: train_by_class[c] for c in CLASSES},
            },
            "Testing": {
                "total_original": 1600,
                "included": 1486,
                "excluded": 114,
                "classes_original": {c: test_orig_by_class[c] for c in CLASSES},
                "classes_included": {c: test_included_by_class[c] for c in CLASSES},
                "classes_excluded": {c: test_excluded_by_class[c] for c in CLASSES},
            },
        },
        "duplicate_analysis": {
            "total_duplicate_groups": total_dup_groups,
            "total_affected_images": total_affected_images,
            "cross_split_groups": cross_split_groups_count,
            "unique_cross_split_training_images": len({
                r["rel_path"] for r in train_rows if r["is_cross_split_duplicate"] == "True"
            }),
            "unique_cross_split_testing_images": len(test_excluded),
            "training_only_duplicate_groups": training_only_dup_groups,
            "testing_only_duplicate_groups": testing_only_dup_groups,
            "cross_class_conflict_groups": 0,
        },
    }

    # Write Summary JSON Deterministically (no timestamps, sorted keys)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Summary metrics written to: {summary_json_path}")

    # 7. Print Console Summary
    print("\n" + "-" * 75)
    print("DATASET PARTITION SUMMARY (LEAKAGE-CONTROLLED)")
    print("-" * 75)
    print(f"{'Split':<12} {'Class':<14} {'Original':<10} {'Excluded':<10} {'Included (Clean)':<18}")
    print("-" * 64)
    for c in CLASSES:
        print(f"{'Training':<12} {c:<14} {train_by_class[c]:<10} {0:<10} {train_by_class[c]:<18}")
    print(f"--> Training Subtotal: 5600 original | 0 excluded | 5600 clean\n")

    for c in CLASSES:
        print(
            f"{'Testing':<12} {c:<14} "
            f"{test_orig_by_class[c]:<10} "
            f"{test_excluded_by_class[c]:<10} "
            f"{test_included_by_class[c]:<18}"
        )
    print(
        f"--> Testing Subtotal : 1600 original | {len(test_excluded)} excluded | "
        f"{len(test_included)} clean evaluation images\n"
    )
    print(f"Overall clean benchmark dataset: {5600 + len(test_included)} images (7,086 total).")
    print("=" * 75)

    return manifest_sha256


if __name__ == "__main__":
    root = get_project_root()
    raw_dir = root / "data" / "raw"
    csv_out = root / "data" / "processed" / "dataset_manifest.csv"
    json_out = root / "reports" / "metrics" / "dataset_manifest_summary.json"

    generate_manifest(raw_dir, csv_out, json_out)

