# Lip-Reading

End-to-end isolated-word lip reading on the **MIRACL-VC1** dataset using a **ResNet18 + Bidirectional GRU** architecture. The model is trained on RGB mouth-crop sequences and evaluated under a **speaker-independent** protocol, achieving **60.0% test accuracy** across 10 word classes.

---

## Repository Structure

Lip-Reading/
├── RGB_dataset_preprocessing_(mouth_crop).ipynb   # Mouth crop extraction pipeline
├── lip-reading-resnet18-gru.ipynb                 # Model training & evaluation
├── requirements.txt                               # Project dependencies for reproducibility
└── README.md                                      # Documentation & analysis

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

> Notebook: `RGB_dataset_preprocessing_(mouth_crop).ipynb`

Raw video frames are processed to isolate the mouth region, discarding irrelevant facial and background information.

**Pipeline:**
1. **Landmark Detection** — [MediaPipe FaceMesh](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) detects 20 lip-contour landmarks per frame
2. **Mouth Cropping** — A bounding box is computed around the lip landmarks and the Region of Interest (ROI) is cropped
3. **Resizing** — All crops are resized to a fixed resolution of **64 × 64 pixels**
4. **Temporal Padding** — Sequences are zero-padded to a fixed length of **T = 22 frames** to handle variable-duration samples

*Example of a preprocessed sequence: 7 real mouth-crop frames (t=0–6) zero-padded with black frames to reach the fixed length of T=22.*

<img width="1300" height="870" alt="Temporal padding example" src="https://github.com/user-attachments/assets/4038bc0c-e8f0-4d46-b977-6adcab4ce4ec" />

**Requirements:**
- Python 3.10.x
- MediaPipe 0.10.21 (newer versions are incompatible)

---

## Model Architecture

> Notebook: `lip-reading-resnet18-gru.ipynb`

The model follows a **CNN-RNN hybrid** design: a convolutional backbone extracts spatial features from each frame independently, and a recurrent module models temporal dependencies across the sequence.

<img width="2466" height="1728" alt="Architecture diagram" src="https://github.com/user-attachments/assets/f2edcb47-1ee2-44da-b4ba-19cfb66d066b" />

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
| Epochs | 30 |
| CNN frozen for | first 10 epochs |

---

## Results

### Test Accuracy: **60.0%** (speaker-independent)

### Confusion Matrix

*Row-normalized confusion matrix on the test set. Diagonal values represent per-class recall.*

<img width="835" height="690" alt="conf_matrix" src="https://github.com/user-attachments/assets/3936ce66-30e2-4b49-8a5a-7b55c627eadc" />

### Per-class Performance

| Word       | Recall |
|:-----------|:-------|
| Hello      | 0.93   |
| Start      | 0.77   |
| Connection | 0.67   |
| Choose     | 0.63   |
| Next       | 0.63   |
| Well       | 0.63   |
| Navigation | 0.50   |
| Stop       | 0.50   |
| Previous   | 0.43   |
| Begin      | 0.30   |

The confusion pattern is not random — it concentrates on **visually similar word pairs** (Begin↔Next, Navigation↔Previous). This is a known challenge in lip reading caused by **viseme ambiguity**: different words that produce nearly identical mouth shapes. Words with distinctive lip geometry (Hello, Start, Connection) are recognized reliably, while words with subtle or similar articulation remain harder to distinguish from RGB alone.

### Conclusion

A significant gap exists between training accuracy (~99%) and test accuracy (60.0%). This is a known characteristic of **speaker-independent** evaluation, especially on low-resource datasets like MIRACL-VC1.

#### 1. The Challenge of Speaker Independence
In this setting, the model is tested on identities it has never seen during training. It must decouple **"what is being said"** (visemes) from **"who is saying it"** (lip geometry, skin tone, and individual articulation styles). While the model easily memorizes the 10-12 training speakers (99% acc), generalizing to the unique physiological traits of a new person is significantly harder.

#### 2. Performance in Context (Benchmark Alignment)
Although 60% might seem low in a vacuum, it is highly competitive for the MIRACL-VC1 dataset:

* **Data Scarcity:** MIRACL-VC1 contains only ~1,500 word samples. For a high-dimensional temporal task like video classification, this is a very "small-data" regime. 
* **The Complexity of the Task:** Unlike static image classification, lip reading requires the model to learn subtle spatio-temporal nuances. To bridge the gap to production-level accuracy (90%+), a massive increase in data is necessary.

#### 3. Strategic Takeaways
The fact that **Validation** and **Test** accuracies are nearly identical confirms that the model is not "overfitting" to a specific split, but rather that it has reached the **performance ceiling** allowed by the current dataset size. 

**Main Conclusion:** The model architecture (ResNet18 + GRU) is robust, but the primary bottleneck is the limited diversity and volume of the data. For a complex task like speaker-independent visual speech recognition, collecting a larger, more diverse dataset or utilizing extensive synthetic data augmentation is the only viable path to significantly higher generalization performance.

---

## 🛠️ Quick Start
Install dependencies to run the notebooks:
```bash
pip install -r requirements.txt
```

> ⚠️ Preprocessing requires **Python 3.10** and **MediaPipe 0.10.21** specifically.  
> Training was run on Kaggle with a GPU accelerator.

---
