# NeuroScan AI — Brain Tumor Detection Using Deep Learning

**Project:** NeuroScan AI  
**Focus:** Case Study & Reproduction of *"Classifying Brain Tumors on Magnetic Resonance Imaging by Using Convolutional Neural Networks"* (Gómez-Guzmán et al., Electronics 2023)  
**Academic Context:** Deep Learning Case Study (23ADL401)  
**Institution:** Department of Artificial Intelligence & Data Science, Amal Jyothi College of Engineering (Autonomous)

---

## 1. Project Overview

NeuroScan AI is an end-to-end deep learning framework designed to detect and classify brain tumors from Magnetic Resonance Imaging (MRI) scans into four distinct categories:
- **Glioma**
- **Meningioma**
- **No Tumor** (Healthy tissue)
- **Pituitary**

This repository contains the verified implementation, trained model checkpoints, experiment metrics, test suite, and backend architecture reproducing the generic CNN architecture from Gómez-Guzmán et al. (2023).

---

## 2. Model Architecture & Experiment E2 Summary

- **Architecture:** `PaperGenericCNN` (3 convolutional blocks with 5×5 and 3×3 kernels, ReLU activations, 2×2 max-pooling, fully connected layer of 128 units, dropout rate of 0.5, and 4-way softmax output).
- **Trainable Parameters:** 651,044 parameters.
- **Optimizer:** Adam ($\alpha = 0.0001, \beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$).
- **Batch Size:** 16
- **Epochs:** 10
- **Input Resolution:** 256 × 256 pixels, RGB format.
- **Brightness Augmentation:** `paper_literal` (`torchvision.transforms.ColorJitter(brightness=(0.0, 0.7))`).
- **Best Validation Performance (Epoch 9):**
  - **Validation Accuracy:** **80.98%** (907 / 1,120 images)
  - **Validation Macro-F1:** **81.86%**
  - **Validation Loss:** **0.8291**
  - **Per-Class Metrics (Epoch 9):**
    - Glioma: Precision 92.37%, Recall 77.86%, F1 84.50%
    - Meningioma: Precision 59.23%, Recall 88.21%, F1 70.88%
    - No Tumor: Precision 97.69%, Recall 75.36%, F1 85.08%
    - Pituitary: Precision 92.03%, Recall 82.50%, F1 87.01%
- **Quarantined Clean Test Split:** 1,486 images strictly quarantined; never leaked, touched, or evaluated during training or hyperparameter tuning.

---

## 3. Repository & Project Structure

```text
neuroscan_ai/
├── README.md                                    # Project overview, E2 results, test instructions
├── requirements.txt                             # Python dependencies (torch, torchvision, pillow, pyyaml)
├── .gitignore                                   # Git ignore rules for virtualenv, weights, raw data
├── configs/
│   ├── data_config.yaml                         # Dataset splitting and path configs (256x256, rescale_only)
│   └── train_config.yaml                        # Experiment E2 hyperparameters (Adam, lr=1e-4, batch=16)
├── src/neuroscan/
│   ├── data/                                    # Dataset loader, transforms, splitting
│   ├── models/                                  # PaperGenericCNN and BaselineCNN architectures
│   ├── training/                                # Training loop, checkpointing, early validation
│   └── evaluation/                              # Multi-class metrics (Accuracy, Macro-F1, Precision, Recall)
├── tests/                                       # 37 comprehensive unit tests
├── models/
│   └── checkpoints/                             # Model checkpoint directory (weights ignored in Git)
├── reports/
│   └── metrics/
│       ├── paper_generic_cnn_training_summary.json  # Complete 10-epoch metrics and logs (Epoch 9 best)
│       └── dataset_manifest_summary.json        # Dataset distribution & 0% leakage audit summary
├── scripts/                                     # Data audit and verification utilities
├── docs/
│   ├── README.md                                # Documentation and deliverables index
│   ├── DL_Case_Study_Report.pdf                 # Final Academic Case Study Report (PDF)
│   ├── presentation.pdf                         # 19-slide Case Study Presentation (PDF)
│   ├── 23ADL401_Paper_Case_Study_Template.pptx  # Department slide presentation template
│   ├── 23ADT401_Case_Study_Report_Template.docx # Department Word report template
│   ├── electronics-12-00955-v2.pdf              # Base research paper (Gómez-Guzmán et al., 2023)
│   ├── dataset_leakage_audit.md                 # Cryptographic dataset audit report
│   ├── base_paper_reproduction_plan.md          # Technical reproduction roadmap
│   └── paper_reproduction_report.md             # Empirical reproduction analysis
└── data/
    ├── raw/                                     # Raw MRI datasets (quarantined and ignored in Git)
    └── processed/
        └── dataset_manifest.csv                 # Full audit manifest for all 7,086 images (split, hash, dimensions)
```

> **Data & Model Weight Management:**  
> - **Raw MRI Images:** Excluded from Git to preserve repository hygiene and enforce strict test-set quarantine.
> - **Model Checkpoints:** Model weight files (`*.pt`, `*.pth`) are ignored in Git. The complete submission bundle including trained checkpoint weights (`paper_generic_cnn_best.pt`, 7.84 MB) and sample MRI scans is packaged separately as `neuroscan_ai_backend_submission.zip` (17.79 MB) for Google Drive academic submission.
> - **Dataset Manifest:** The complete dataset audit metadata is tracked in `data/processed/dataset_manifest.csv` allowing instant verification and unit testing without downloading the raw dataset.

---

## 4. Setup & Verification Instructions

### 4.1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4.2. Running the Test Suite

All 37 unit tests can be executed without requiring the full raw image dataset:

```bash
python3 -m unittest discover tests
```

Expected output:
```text
Ran 37 tests in ~1.5s
OK
```

### 4.3. Inspecting Model Checkpoint & Summary

The best model weights and epoch-by-epoch training logs are pre-packaged:
- Weights: `models/checkpoints/paper_generic_cnn_best.pt`
- Epoch metrics: `reports/metrics/paper_generic_cnn_training_summary.json`

---

## 5. References

1. Gómez-Guzmán, F.J., et al. (2023). *Classifying Brain Tumors on Magnetic Resonance Imaging by Using Convolutional Neural Networks.* Electronics, 12(4), 955.
2. Brain Tumor MRI Dataset (7,086 images: 5,600 training, 1,486 testing).

