# NeuroScan AI — Dataset Duplicate Audit & Leakage-Control Documentation

---

## 1. Executive Summary

During exploratory data analysis (EDA) of the **Brain Tumor MRI Dataset** (curated by Masoud Nickparvar, comprising 7,200 images across four balanced classes), an automated data-leakage audit identified **111 duplicate groups spanning the Training and Testing partitions**.

This cross-split contamination directly affects **114 test images** (7.125% of the 1,600-image test set), whose exact decoded pixel content appears in the training set. Naive model evaluation against the uncleaned test set would introduce positive bias and artificially inflated performance metrics.

To ensure academic and experimental integrity without altering source data files, NeuroScan AI establishes a **reproducible, leakage-controlled virtual partitioning pipeline** managed via [`data/processed/dataset_manifest.csv`](file:///home/main/projects/neuroscan_ai/data/processed/dataset_manifest.csv).

---

## 2. Audit Methodology

### 2.1 Hashing Strategy
Duplicate detection was conducted using a two-tiered hashing protocol across all 7,200 images:

1. **Binary File Hashing (`file_sha256`):**
   * Computes the SHA-256 digest of raw file bytes using chunked streaming reads ($64\text{ KB}$).
   * **Result:** Zero cross-split duplicates were detected at the binary file level. Image files in `Training/` and `Testing/` possess distinct file headers and compression metadata.

2. **Decoded-Pixel Hashing (`pixel_sha256`):**
   * Decodes each JPEG into memory using Pillow (`PIL.Image.open`), retrieving image width $W$, height $H$, native color mode $M$ (e.g., `'RGB'`, `'L'`), and the uncompressed raster byte buffer (`img.tobytes()`).
   * The hash digest is computed over the prefixed byte sequence:
     $$\text{Key} = \text{SHA-256}\Big(\text{ASCII}(W \times H \_\, M :) \;\|\; \text{img.tobytes()}\Big)$$
   * **Result:** Decoded-pixel hashing uncovered **111 cross-split duplicate groups**, exposing images with identical underlying pixel matrices despite divergent file-level headers.

### 2.2 Independent Verification
The findings were verified independently via two separate scripts ([`scripts/check_dataset_duplicates.py`](file:///home/main/projects/neuroscan_ai/scripts/check_dataset_duplicates.py) and [`scripts/verify_duplicates_independent.py`](file:///home/main/projects/neuroscan_ai/scripts/verify_duplicates_independent.py)). Both native color mode hashing and standardized RGB conversion (`img.convert('RGB')`) produced identical duplicate sets, confirming zero cross-mode discrepancies.

---

## 3. Verified Audit Findings

| Metric | Exact File Hash | Decoded Raw Pixels | Status & Impact |
|---|---|---|---|
| **Total Images Analyzed** | 7,200 | 7,200 | 100% readable, zero corrupted files |
| **Total Duplicate Groups** | 153 | 277 | Significant internal duplication present |
| **Total Images in Duplicate Groups** | 340 | 599 | ~8.32% of total dataset |
| **Cross-Split Groups ($\text{Train} \leftrightarrow \text{Test}$)** | **0** | **111** | **Data Leakage Confirmed** |
| **Unique Leaked Test Images** | **0** | **114** | 114 / 1,600 test images (7.125%) |
| **Participating Training Images** | **0** | **115** | 115 / 5,600 training images (2.05%) |
| **Cross-Class Label Conflicts** | **0** | **0** | **Zero label noise** (100% consistent) |
| **Training-Only Redundant Groups** | 138 | 149 | Redundant samples within training |
| **Testing-Only Redundant Groups** | 15 | 17 | 17 groups (34 images) within testing |

### Key Observations:
* **Label Consistency:** Across all 277 duplicate groups, every duplicate copy shares the exact same class label. There is zero cross-class label conflict.
* **Intra-Split Duplication:** Redundancies exist solely within `Training` (149 groups) and solely within `Testing` (17 groups). Testing-only duplicates represent evaluation redundancy, not leakage from training.

---

## 4. Leakage-Control Exclusion Policy

NeuroScan AI enforces a non-destructive **virtual partitioning policy**:

1. **Non-Destructive Guarantee:**
   * No files in [`data/raw/`](file:///home/main/projects/neuroscan_ai/data/raw) are modified, moved, renamed, or deleted. All source files remain in their original states.
2. **Training Split Preservation:**
   * All **5,600 training images** are retained (`eval_status: include`, `exclusion_reason: none`).
3. **Primary Evaluation Set Decontamination:**
   * Exactly **114 test images** matching any training pixel hash are quarantined from primary benchmark evaluation (`eval_status: exclude`, `exclusion_reason: cross_split_leakage_with_training`).
4. **Retention of Testing-Only Duplicates:**
   * The remaining **1,486 test images**, including the 17 testing-only duplicate groups, are retained (`eval_status: include`, `exclusion_reason: none`). These do not receive information from training and represent independent test samples.

---

## 5. Resulting Dataset Partitions

### Class Distribution (Original vs. Decontaminated Evaluation Set)

| Diagnostic Class | Original Training | Original Testing | Excluded Test Images (Leakage) | Clean Evaluation Test Set | Clean Total (Train + Clean Test) |
|---|---|---|---|---|---|
| **Glioma** | 1,400 | 400 | **5** | **395** (26.58%) | 1,795 |
| **Meningioma** | 1,400 | 400 | **100** | **300** (20.19%) | 1,700 |
| **No Tumor** | 1,400 | 400 | **9** | **391** (26.31%) | 1,791 |
| **Pituitary** | 1,400 | 400 | **0** | **400** (26.92%) | 1,800 |
| **Total** | **5,600** | **1,600** | **114** | **1,486** (100.00%) | **7,086** |

*Note: Meningioma exhibits the overwhelming majority of cross-split duplicate leakage (100 images), followed by No Tumor (9) and Glioma (5). Pituitary had zero cross-split duplicates with training.*

---

## 6. Manifest Specification: `dataset_manifest.csv`

The manifest is generated deterministically by [`scripts/prepare_dataset_manifest.py`](file:///home/main/projects/neuroscan_ai/scripts/prepare_dataset_manifest.py) at [`data/processed/dataset_manifest.csv`](file:///home/main/projects/neuroscan_ai/data/processed/dataset_manifest.csv).

* **Header (9 columns):**
  `rel_path,split,label,file_sha256,pixel_sha256,is_duplicate,is_cross_split_duplicate,eval_status,exclusion_reason`
* **Row Count:** Exactly 7,200 rows + 1 header row.
* **Deterministic Ordering:** Sorted by `(split, label, rel_path)`.
* **Reproducibility:** Generates an identical SHA-256 digest across repeated runs.

Downstream PyTorch Dataset implementations must instantiate data using this manifest:
```python
# Primary evaluation test set:
clean_test_df = manifest_df[
    (manifest_df["split"] == "Testing") & (manifest_df["eval_status"] == "include")
]  # Exactly 1,486 images
```

---

## 7. Technical Limitations & Engineering Caveats

For academic reporting and thesis documentation, the following limitations must be noted:

1. **Bitwise Decoded-Pixel vs. Perceptual Equivalence:**
   * Decoded-pixel SHA-256 hashing requires exact raster byte equality.
   * **Limitation:** It does not capture near-duplicates that have undergone slight re-compression (JPEG quantization variations), spatial downsampling, border cropping, or intensity windowing adjustments. While it removes all 100% bit-identical twins, residual near-duplicates may persist.

2. **Absence of Patient & Acquisition Metadata:**
   * The publicly distributed Kaggle dataset provides only static JPEG files grouped into class directories.
   * The dataset lacks patient identifiers, hospital source tags, acquisition series IDs, and anatomical slice sequence numbers.
   * **Limitation:** In clinical practice, an MRI study consists of multi-slice 2D sequences from a single patient. Without patient identifiers, patient-level independence between Training and Testing cannot be mathematically established. It remains possible that different slices from the same patient's scan session exist across both splits.
