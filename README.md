# Lip-Reading

End-to-end isolated-word lip reading on the **MIRACL-VC1** dataset using a **ResNet18 + Bidirectional GRU** architecture. The model is trained on RGB mouth-crop sequences and evaluated under a **speaker-independent** protocol, achieving **60.3% test accuracy** across 10 word classes.

---

## Repository Structure

```
Lip-Reading/
├── RGB_dataset_preprocessing__mouth_crop_.ipynb   # Mouth crop extraction pipeline
├── lip_reading_resnet18_gru.ipynb                 # Model training & evaluation
└── README.md
```

---

## Dataset

All experiments use the [MIRACL-VC1](https://sites.google.com/site/achrafbenhamadou/-datasets/miracl-vc1) dataset, an RGB-D audiovisual database featuring 15 subjects (5 male, 10 female) repeating a fixed vocabulary of **10 isolated words**:

| Code | Word       |
|------|------------|
| 01   | Begin      |
| 02   | Choose     |
| 03   | Connection |
| 04   | Navigation |
| 05   | Next       |
| 06   | Previous   |
| 07   | Start      |
| 08   | Stop       |
| 09   | Hello      |
| 10   | Well       |

Each subject repeats each word 10 times, resulting in **1,500 samples** for the word classification task.

### Speaker-Independent Split

To evaluate generalization to **unseen identities**, speakers are partitioned into disjoint sets — no speaker appears in more than one split:

| Split | Speakers | Count |
|-------|----------|-------|
| Train | F01, F02, F04, F05, F06, F07, F08, M01, M02, M04 | 10 |
| Val   | F09, M07 | 2 |
| Test  | F10, F11, M08 | 3 |

---

## Preprocessing

> Notebook: `RGB_dataset_preprocessing__mouth_crop_.ipynb`

Raw video frames are processed to isolate the mouth region, discarding irrelevant facial and background information.

**Pipeline:**
1. **Landmark Detection** — [MediaPipe FaceMesh](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) detects 20 lip-contour landmarks per frame
2. **Mouth Cropping** — A bounding box is computed around the lip landmarks and the Region of Interest (ROI) is cropped
3. **Resizing** — All crops are resized to a fixed resolution of **64 × 64 pixels**
4. **Temporal Padding** — Sequences are zero-padded to a fixed length of **T = 22 frames** to handle variable-duration samples

> 💡 *Place preprocessing figures here — e.g. the temporal padding figure and the RGB crop examples from the paper*

**Requirements:**
- Python 3.10.x
- MediaPipe 0.10.21 (newer versions are incompatible)

---

## Model Architecture

> Notebook: `lip_reading_resnet18_gru.ipynb`

The model follows a **CNN-RNN hybrid** design: a convolutional backbone extracts spatial features from each frame independently, and a recurrent module models temporal dependencies across the sequence.

```
Input: (B, T=22, 3, 64, 64)
        │
        ▼
  ResNet18 (per-frame)        ← ImageNet pretrained, FC layer removed
  Feature: (B, T, 512)
        │
        ▼
  Feature Dropout (p=0.5)
        │
        ▼
  Bidirectional GRU           ← hidden_size=128, 1 layer
  Output: (B, 256)            ← concat of forward + backward last hidden states
        │
        ▼
  Dropout (p=0.5)
        │
        ▼
  Linear Classifier → 10 classes
```

### Key Design Choices

- **Pack-padded sequences** — the GRU only processes real frames, ignoring zero-padding, via `pack_padded_sequence`
- **Frozen BatchNorm** — running statistics of frozen ResNet18 layers are kept in eval mode to avoid corruption from small batches
- **Two-phase fine-tuning** — ResNet18 is fully frozen for the first 10 epochs; `layer4` is unfrozen afterwards to adapt pretrained features to lip visemes without catastrophic forgetting

---

## Training

### Data Augmentation

The training set is expanded **4×** via augmentation applied consistently across all frames of each sequence:

| Type | Transformation |
|------|---------------|
| Spatial | Random horizontal flip, rotation ±8°, random shift/crop with reflect padding |
| Photometric | Additive uniform RGB noise (amplitude ±0.05), constant across frames |

Validation and test sets are **not augmented**.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 8 |
| Optimizer | AdamW |
| LR (GRU + classifier) | 1e-3 |
| LR (CNN backbone, phase 2) | 1e-4 |
| Weight decay | 1e-4 |
| Loss | CrossEntropyLoss (label smoothing 0.05) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Gradient clipping | 1.0 |
| Epochs | 40 |
| CNN frozen for | first 10 epochs |

---

## Results

### Test Accuracy: **60.3%** (speaker-independent)

> This result is competitive with the published baseline of 59% by Gutierrez & Robert (2017) on the same dataset and evaluation protocol.

### Confusion Matrix

> 💡 *Place your confusion matrix figure here*

**Per-class analysis:**

| Word | Recall | Notes |
|------|--------|-------|
| Hello | ~93% | Distinctive lip extension, easy for RGB |
| Start | ~80% | Strong visual signature |
| Well | ~77% | Clear tongue/lip movement |
| Connection | ~73% | Recognizable articulation |
| Navigation | ~33% | Visually ambiguous, short duration |
| Begin | ~33% | Confused with Next — viseme overlap |

The confusion pattern is not random — it concentrates on **visually similar word pairs** (e.g. Begin↔Next, Navigation↔Previous). This is a known challenge in lip reading caused by **viseme ambiguity**: different words that produce nearly identical mouth shapes. Words with distinctive 3D lip geometry (Hello, Start, Well) are recognized reliably, while words with subtle or similar articulation remain hard to distinguish from RGB alone.

### Generalization Gap

A significant gap exists between training (~99%) and test (~60%) accuracy, which is expected and characteristic of **speaker-independent evaluation on small datasets**. The model has sufficient capacity to memorize the 10 training speakers but must generalize to entirely unseen identities at test time. The close match between validation (61%) and test (60.3%) accuracy confirms the model generalizes consistently to new speakers — it is not overfitting to the validation set.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/Lip-Reading.git
cd Lip-Reading

# Install dependencies
pip install torch torchvision mediapipe==0.10.21 opencv-python tqdm matplotlib seaborn scikit-learn

# Run preprocessing first, then training
# Open notebooks in Jupyter or Kaggle
```

> ⚠️ Preprocessing requires **Python 3.10** and **MediaPipe 0.10.21** specifically.  
> Training was run on Kaggle with a GPU accelerator.
