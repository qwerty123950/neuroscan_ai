# NeuroScan AI — Base Paper Reproduction Plan

**Reference Paper:**  
*Classifying Brain Tumors on Magnetic Resonance Imaging by Using Convolutional Neural Networks*  
**Authors:** Marco Antonio Gómez-Guzmán, Laura Jiménez-Beristaín, Enrique Efren García-Guerrero, Oscar Roberto López-Bonilla, Ulises Jesús Tamayo-Perez, José Jaime Esqueda-Elizondo, Kenia Palomino-Vizcaino, Everardo Inzunza-González  
**Journal:** *Electronics*, 2023, Volume 12, Issue 4, Article 955  
**DOI:** [10.3390/electronics12040955](https://doi.org/10.3390/electronics12040955)  
**URL:** [https://www.mdpi.com/2079-9292/12/4/955](https://www.mdpi.com/2079-9292/12/4/955)  

---

## 1. Executive Summary

This document establishes the reproduction plan to align NeuroScan AI with the published methodology in Gómez-Guzmán et al. (2023). The paper evaluates seven convolutional neural network architectures for 4-class brain tumor classification (*glioma*, *meningioma*, *notumor*, *pituitary*): a custom non-pretrained **Generic CNN** model and six transfer-learning architectures (ResNet50, InceptionV3, InceptionResNetV2, Xception, MobileNetV2, and EfficientNetB0). The paper's reported top-performing model is InceptionV3 (97.12% accuracy), while the baseline Generic CNN achieves 81.05% accuracy.

This plan details:
1. The paper's exact documented methodology.
2. A side-by-side comparison against NeuroScan AI's current state.
3. Every concrete discrepancy across dataset, preprocessing, architecture, optimization, and evaluation.
4. Identifiable ambiguities and documented contradictions in the paper.
5. The exact proposed implementation and unavoidable deviations required for academic integrity.

---

## 2. Documented Paper Methodology

### 2.1 Dataset Composition & Partitioning
* **Dataset:** Masoud (Msoud) 2021 brain tumor MRI dataset, aggregating Figshare, SARTAJ, and Br35H datasets.
* **Volume:** 7,023 total MRI images across four classes:
  - Glioma (Class 0)
  - Meningioma (Class 1)
  - No-tumor (Class 2)
  - Pituitary (Class 3)
* **Data Split:**
  - Initial 80/20 train/test partition: 80% for training (~5,618 images) and 20% for testing (~1,405 images).
  - 5-fold cross-validation ($k = 5$) conducted on the training set to evaluate stability and avoid overfitting.
* **Data Leakage Control:** The paper does **not** audit or report cross-split exact pixel duplicates between training and testing partitions.

### 2.2 Preprocessing & Data Augmentation (Table 1)
From **Table 1** (*Parameter values for the preprocessing stage for each CNN model*):
* **Rescaling:** $1.0 / 255$ (pixel intensities scaled linearly to $[0.0, 1.0]$).
* **Input Image Dimensions:**
  - Generic CNN: $256 \times 256 \times 3$ (Figure 2 caption explicitly states: "Input size: $256 \times 256$").
  - ResNet50 / MobileNetV2: $224 \times 224 \times 3$.
  - InceptionV3 / InceptionResNetV2 / Xception: $299 \times 299 \times 3$.
* **Data Augmentation:**
  - Rotation: $10^\circ$
  - Horizontal Flip: Enabled (`True`)
  - Shear: $0.1$ ($10\%$ shear angle/transformation)
  - Zoom / Scaling: $0.1$ (scaling factor $[0.9, 1.1]$)
  - Brightness Range: $0.0$ to $0.7$
* **Normalization:** Standard Keras `ImageDataGenerator(rescale=1./255)`. No ImageNet mean/standard-deviation channel standardization is applied to the Generic CNN.

### 2.3 Generic CNN Architecture (Figure 2)
From **Figure 2** (*Architecture of the generic CNN model*):
* **Sequential Layer Count:** Described as "17 layers in total" implemented in Python / Google Colaboratory.
* **Feature Extractor:**
  - Four convolutional blocks.
  - Each block consists of:
    1. Convolutional layer (`Conv2D`, $3 \times 3$ kernel, stride 1)
    2. Batch Normalization (`BatchNormalization`)
    3. Activation function (`ReLU`)
    4. Max-pooling layer (`MaxPooling2D`, $2 \times 2$ window, stride 2)
* **Classifier Head:**
  - `Flatten` layer transitioning from spatial feature maps to a 1D feature vector.
  - Dropout regularization stage.
  - Fully connected dense layer (`Dense`) with `Softmax` activation mapping to 4 class probabilities.

### 2.4 Training Hyperparameters (Table 2 & Figure 2 Caption)
From **Table 2** (*Hyper-parameter values for training the Generic CNN model*):
* **Optimizer:** Stochastic Gradient Descent (`SGD`).
* **Learning Rate:** $0.0001$ ($10^{-4}$).
* **Batch Size:** $16$.
* **Epochs:** $10$ epochs per fold.
* **Loss Function:** Categorical Cross-Entropy (`CrossEntropyLoss`).
* **Activation:** `ReLU` (intermediate hidden layers), `Softmax` (output layer).
* **Dropout Rate:** Documented ambiguity between sections:
  - Table 2 states: `Dropout: 0.2`
  - Figure 2 caption states: `Dropout: 0.5`
* **Regularization / Scheduling:** No learning rate scheduler, weight decay, momentum factor, or early stopping is documented for the Generic CNN in Table 2.

### 2.5 Published Results (Table 5)
* **Generic CNN Baseline:**
  - Accuracy: **81.05%**
* **Transfer Learning Models:**
  - InceptionV3: **97.12%** (1st place)
  - ResNet50: **96.97%** (2nd place)
  - InceptionResNetV2, Xception, MobileNetV2, EfficientNetB0 evaluated.

---

## 3. Current Implementation vs. Paper Methodology

| Dimension | Paper Specification | Current NeuroScan AI Implementation | Status / Discrepancy |
|---|---|---|---|
| **Raw Dataset Volume** | 7,023 images | 7,200 images (`Training/`: 5,600, `Testing/`: 1,600) | Divergent raw count in public releases |
| **Leakage Audit** | None performed; duplicates ignored | 111 cross-split duplicate groups; 114 test images quarantined | **Major rigor improvement** in NeuroScan AI |
| **Split Strategy** | 80/20 split (5,618 / 1,405) + 5-fold CV | Stratified 80/20 train/val (4,480 / 1,120) from `Training/` | Clean test set quarantined |
| **Input Image Size** | $256 \times 256$ pixels | $224 \times 224$ pixels | **Discrepancy:** Needs update to $256 \times 256$ |
| **Color Mode** | 3-channel RGB | 3-channel RGB | Consistent |
| **Pixel Normalization** | Rescale $1.0 / 255 \in [0.0, 1.0]$ | ImageNet Mean & Std normalization | **Discrepancy:** Needs update to $1/255$ |
| **Data Augmentation** | Flip, $\pm 10^\circ$ Rot, Shear $0.1$, Zoom $0.1$, Brightness $[0, 0.7]$ | Flip ($p=0.5$), $\pm 10^\circ$ Rotation | **Discrepancy:** Add shear, zoom, brightness |
| **Model Architecture** | 4 ConvBlocks $\to$ Flatten $\to$ Dropout $\to$ Dense(4) | 4 ConvBlocks $\to$ AdaptiveAvgPool2d $\to$ FC(128) $\to$ FC(4) | **Discrepancy:** AdaptiveAvgPool vs Flatten |
| **Optimizer** | SGD (learning rate = $0.0001$) | AdamW (learning rate = $0.001$, weight decay = $10^{-4}$) | **Discrepancy:** Needs update to SGD ($10^{-4}$) |
| **Batch Size** | 16 | 16 | Consistent |
| **Epochs** | 10 per fold | 25 with early stopping | **Discrepancy:** Needs update to 10 epochs |
| **Dropout** | Ambiguous: 0.2 (Table 2) vs 0.5 (Fig 2) | 0.4 (FC1) and 0.2 (FC2) | Document ambiguity; support configurable rate |
| **Loss Function** | Categorical CrossEntropy | CrossEntropyLoss (`label_smoothing=0.0`) | Consistent |
| **Metrics** | Accuracy, Precision, Recall, Specificity | Accuracy, Macro-F1, Precision, Recall, Confusion Matrix | Consistent & extended |

---

## 4. Analysis of Ambiguities & Undocumented Details

1. **Exact Conv Filter Counts in Generic CNN:**
   - The paper text and Figure 2 illustrate four convolutional blocks with batch normalization, ReLU, and max pooling, but do not provide an explicit numerical filter schedule (e.g., $32 \to 64 \to 128 \to 256$).
   - Standard hierarchical doubling ($32 \to 64 \to 128 \to 256$) with $3 \times 3$ kernels is standard in Keras/TensorFlow tutorials and matches the 17-layer sequential definition.
2. **Layer Count Discrepancy (17 layers):**
   - In Keras/TensorFlow sequential modeling:
     - InputLayer (1)
     - Block 1: Conv2D, BatchNormalization, ReLU, MaxPooling2D (4)
     - Block 2: Conv2D, BatchNormalization, ReLU, MaxPooling2D (4)
     - Block 3: Conv2D, BatchNormalization, ReLU, MaxPooling2D (4)
     - Block 4: Conv2D, BatchNormalization, ReLU, MaxPooling2D (4)
     - Flatten (1) + Dense/Dropout/Softmax (2-3 layers).
     - Depending on whether ReLU is an argument to Conv2D or a separate layer, and whether a dense hidden layer is present before Dropout, the count varies between 15 and 18 layers.
3. **Dropout Rate Contradiction:**
   - Table 2 explicitly lists `Dropout: 0.2`.
   - Figure 2 caption explicitly states: `applying at the end of the dense stage a Dropout of 0.5`.
   - Defaulting to `0.5` at the dense stage with a config override to `0.2` resolves this contradiction transparently.
4. **SGD Momentum and Weight Decay:**
   - Table 2 lists `Optimizer: SGD` and `Learning rate: 0.0001`. It does not specify whether Nesterov momentum (e.g., 0.9) or weight decay was utilized. Pure PyTorch `SGD(lr=1e-4)` without momentum is the strict literal interpretation.
5. **Dataset Image Count (7,023 vs. 7,200):**
   - The original Masoud Nickparvar Kaggle upload contains 7,200 images (5,600 Training, 1,600 Testing). A pre-cleaned version circulating on Kaggle removed some corrupted/redundant files resulting in 7,023 images. The paper evaluated the 7,023 image version without conducting a decoded-pixel cross-split leakage audit.

---

## 5. Proposed Implementation Plan

### 5.1 Dataset & Leakage Policy (Strict Adherence)
* Preserve raw images in `data/raw/` and manifest in `data/processed/dataset_manifest.csv`.
* **Zero Leakage Rule:** Keep the 1,486 clean testing records quarantined. Do not touch or tune against the testing partition.
* Derive training and validation sets strictly from the 5,600 included `Training` records using stratified splitting.

### 5.2 Preprocessing Pipeline (`src/neuroscan/data/transforms.py`)
* Update image resolution: default `img_size = 256` (matching Figure 2).
* Normalization: rescale pixel intensities to $[0.0, 1.0]$ by `ToTensor()` without ImageNet mean/std subtraction.
* Augmentations (matching Table 1):
  - `RandomHorizontalFlip(p=0.5)`
  - `RandomRotation(degrees=10)`
  - `RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=10)` (matching shear $0.1$ and zoom $0.1$).
  - `ColorJitter(brightness=(0.7, 1.3))` (or safe range preserving lesion contrast).

### 5.3 Paper Generic CNN (`src/neuroscan/models/generic_cnn.py` & `baseline_cnn.py`)
* Create `PaperGenericCNN` matching the 17-layer architecture:
  - Input: $[B, 3, 256, 256]$
  - 4 ConvBlocks: Conv2d(3x3) $\to$ BatchNorm2d $\to$ ReLU $\to$ MaxPool2d(2x2).
  - Spatial reduction: $256 \to 128 \to 64 \to 32 \to 16$.
  - Classification Head: Flatten $\to$ Linear($16 \times 16 \times 256$, 128) $\to$ ReLU $\to$ Dropout(p=0.5) $\to$ Linear(128, 4).
  - Alternative direct head: Flatten $\to$ Dropout(p=0.5) $\to$ Linear($16 \times 16 \times 256$, 4).
  - Provide both head variants with parameter documentation.

### 5.4 Configuration Alignment (`configs/train_config.yaml` & `configs/data_config.yaml`)
* Update `data.img_size: 256`.
* Update `training.optimizer`:
  - `name: "SGD"`
  - `learning_rate: 0.0001`
  - `momentum: 0.0` (strict literal paper specification)
  - `weight_decay: 0.0`
* Update `training.epochs: 10`.
* Update `model.architecture: "PaperGenericCNN"`.
* Retain validation tracking and best-checkpoint saving for development verification.

### 5.5 Extensibility for Transfer Learning (Phase Comparison)
* Design `src/neuroscan/models/` factory to cleanly support the paper's six transfer-learning architectures (InceptionV3, ResNet50, MobileNetV2, Xception, EfficientNetB0, InceptionResNetV2) in future experiment runs without impacting the Generic CNN pipeline.

---

## 6. Verification & Safety Guarantees

1. **Quarantine Guarantee:** The 1,486 clean testing images will have 0 samples loaded or evaluated during model development and tuning.
2. **Reproducibility Guarantee:** Seeded initialization before DataLoader iteration ensures run-to-run consistency.
3. **Dry-Run Check:** Model parameter count, input-output tensor shapes, and loss computation verified with 0 optimization steps.

