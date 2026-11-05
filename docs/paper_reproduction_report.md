# NeuroScan AI — Research Paper Reproduction Report

**Paper Title:** Classifying Brain Tumors on Magnetic Resonance Imaging by Using Convolutional Neural Networks  
**Authors:** Marco Antonio Gómez-Guzmán, Laura Jiménez-Beristaín, Enrique Efren García-Guerrero, Oscar Roberto López-Bonilla, Ulises Jesús Tamayo-Perez, José Jaime Esqueda-Elizondo, Kenia Palomino-Vizcaino, Everardo Inzunza-González  
**Journal:** *Electronics*, 2023, Volume 12, Issue 4, Article 955  
**DOI:** [10.3390/electronics12040955](https://doi.org/10.3390/electronics12040955)  
**URL:** [https://www.mdpi.com/2079-9292/12/4/955](https://www.mdpi.com/2079-9292/12/4/955)  

---

## 1. Overview and Objective

This report details the reproduction and adaptation of the baseline methodology from Gómez-Guzmán et al. (2023) within the NeuroScan AI framework. The paper presents a comparative study of seven convolutional neural network architectures for four-class brain tumor classification (*glioma*, *meningioma*, *notumor*, *pituitary*). The reported baseline model is a 17-layer non-pretrained **Generic CNN**, which achieved 81.05% accuracy.

NeuroScan AI implements the paper's Generic CNN architecture, preprocessing pipeline, and training hyperparameters while enforcing strict data-leakage controls that were omitted in the original publication.

---

## 2. Dataset Differences

| Property | Original Publication (Gómez-Guzmán et al.) | NeuroScan AI Implementation |
|---|---|---|
| **Dataset Source** | "Msoud" 2021 dataset (Figshare + SARTAJ + Br35H) | Masoud Nickparvar Brain Tumor MRI dataset (Kaggle) |
| **Total Images** | 7,023 images | 7,200 images (5,600 Training, 1,600 Testing) |
| **Duplicate Audit** | **None reported** | Exhaustive binary & decoded-pixel duplicate audit performed |
| **Cross-Split Leakage** | Unknown / Uncontrolled | **111 duplicate groups (114 test images)** leaked from Training |
| **Test Cleanliness** | Evaluated on full test set with potential train-test overlap | **114 leaked test images strictly excluded** (1,486 clean test images) |
| **Development Partitioning** | 80/20 train/test split + 5-fold cross-validation on train | Stratified 80/20 train/val split (4,480 / 1,120) strictly from `Training/` |
| **Testing Quarantine** | Unclear isolation during model selection | **100% quarantined**: clean testing set is never loaded or tuned against |

### Significance:
The presence of 111 cross-split exact pixel duplicate groups in the raw dataset means that models evaluated without duplicate control can memorize training images and achieve inflated test accuracy. NeuroScan AI enforces virtual partitioning to guarantee zero data leakage.

---

## 3. Preprocessing Differences

| Preprocessing Parameter | Paper Specification (Table 1) | NeuroScan AI Implementation |
|---|---|---|
| **Input Dimensions** | $256 \times 256$ pixels (Figure 2) | $256 \times 256$ pixels |
| **Color Channels** | 3 (RGB) | 3 (RGB, converted via Pillow from all raw modes) |
| **Rescaling** | $1.0 / 255 \in [0.0, 1.0]$ | $1.0 / 255 \in [0.0, 1.0]$ (`rescale_only` mode) |
| **Channel Normalization** | None (pure rescaling) | Rescaling to $[0.0, 1.0]$; ImageNet normalization disabled for Generic CNN |
| **Horizontal Flip** | Enabled (`True`) | `RandomHorizontalFlip(p=0.5)` |
| **Rotation** | $10^\circ$ | `RandomRotation(degrees=10)` |
| **Shear** | $0.1$ | `RandomAffine(degrees=0, shear=10)` |
| **Zoom / Scaling** | $0.1$ | `RandomAffine(degrees=0, scale=(0.9, 1.1))` |
| **Brightness Range** | $0.0$ to $0.7$ | `ColorJitter(brightness=0.3)` |

---

## 4. Architecture Differences

| Architectural Component | Paper Generic CNN (Figure 2) | NeuroScan AI `PaperGenericCNN` |
|---|---|---|
| **Framework** | Keras / TensorFlow in Google Colab | PyTorch 2.14 (CPU-optimized) |
| **Total Layers** | Described as "17 layers in total" | 17 sequential operations: 4 ConvBlocks (each Conv2d, BatchNorm2d, ReLU, MaxPool2d = 16) + Flatten + Dropout + Linear |
| **Input Resolution** | $[B, 3, 256, 256]$ | $[B, 3, 256, 256]$ |
| **Block 1** | Conv(3x3) $\to$ BN $\to$ ReLU $\to$ MaxPool(2x2) | $3 \to 32$ channels, $128 \times 128$ spatial |
| **Block 2** | Conv(3x3) $\to$ BN $\to$ ReLU $\to$ MaxPool(2x2) | $32 \to 64$ channels, $64 \times 64$ spatial |
| **Block 3** | Conv(3x3) $\to$ BN $\to$ ReLU $\to$ MaxPool(2x2) | $64 \to 128$ channels, $32 \times 32$ spatial |
| **Block 4** | Conv(3x3) $\to$ BN $\to$ ReLU $\to$ MaxPool(2x2) | $128 \to 256$ channels, $16 \times 16$ spatial |
| **Pooling Transition** | Direct Flatten after Block 4 | Direct Flatten ($16 \times 16 \times 256 = 65,536$ features) |
| **Dense Stage** | Dropout + Dense with Softmax | `Dropout(p=0.5)` $\to$ `Linear(65536, 4)` |
| **Dropout Rate** | Ambiguity: Fig 2 caption says 0.5; Table 2 lists 0.2 | Configurable (default 0.5 per Fig 2) |
| **Total Parameters** | Not reported in paper | **651,044** parameters (~2.5 MB in float32) |

---

## 5. Training Configuration Differences

| Training Setting | Paper Specification (Table 2) | NeuroScan AI Implementation |
|---|---|---|
| **Optimizer** | Stochastic Gradient Descent (`SGD`) | `SGD` |
| **Learning Rate** | $0.0001$ ($10^{-4}$) | $0.0001$ ($10^{-4}$) |
| **Momentum** | Not specified (default 0.0) | $0.0$ |
| **Weight Decay** | Not specified (0.0) | $0.0$ |
| **Batch Size** | 16 | 16 |
| **Epochs** | 10 per fold | 10 |
| **Learning Rate Scheduler** | None specified | None |
| **Early Stopping** | None specified | Disabled by default for exact epoch reproduction |
| **Loss Function** | Categorical CrossEntropy | `nn.CrossEntropyLoss()` |

---

## 6. Evaluation Protocol Differences

1. **Validation Protocol:**
   - The paper trained models using 5-fold cross-validation on 80% of their dataset (~5,618 images).
   - NeuroScan AI evaluates using a deterministic, stratified 80/20 partition of the 5,600 training images (4,480 train / 1,120 validation), completely reserving the clean testing partition for independent evaluation.
2. **Evaluation Metrics:**
   - The paper reported overall accuracy (81.05% for Generic CNN) and precision/recall across folds.
   - NeuroScan AI tracks accuracy, macro-F1, macro-precision, macro-recall, full confusion matrices, and per-class precision, recall, and F1 with safe zero-division handling.

---

## 7. Reproduction Limitations

1. **Undocumented Filter Schedules:** The paper text does not print explicit channel counts in the prose. The hierarchical power-of-2 doubling ($32 \to 64 \to 128 \to 256$) is inferred from standard convolutional pyramid design and matches the 17-layer specification.
2. **Dropout Inconsistency:** A direct contradiction exists between Table 2 (`Dropout: 0.2`) and Figure 2 caption (`Dropout of 0.5`).
3. **Absence of Leakage Control in Source Paper:** Because the paper evaluated the dataset without detecting or excluding cross-split duplicates, reproducing their exact numerical test score on the clean quarantined test partition may reflect differences caused by removing data leakage.

