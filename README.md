# Lip-Reading

End-to-end isolated-word lip reading on the **MIRACL-VC1** dataset using a **ResNet18 + Bidirectional GRU** architecture. The model is trained on RGB mouth-crop sequences and evaluated under a **speaker-independent** protocol, achieving **60.3% test accuracy** across 10 word classes.

---

## Repository Structure

```
Lip-Reading/
├── RGB_dataset_preprocessing__mouth_crop_.ipynb   # Mouth crop extraction pipeline
├── lip_reading_resnet18_gru.ipynb                 # Model training & evaluation
└── README.md
``
<svg width="100%" viewBox="0 0 680 490.88" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- Input box -->
  <g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
    <rect x="230" y="20" width="220" height="52" rx="8" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
    <text x="340" y="41" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Data preprocessed</text>
    <text x="340" y="59" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">B × T × 3 × 64 × 64</text>
  </g>

  <!-- Arrow 1 -->
  <line x1="340" y1="72" x2="340" y2="108" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

  <!-- CNN container -->
  <g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
    <rect x="80" y="108" width="520" height="148" rx="12" stroke-width="0.5" style="fill:rgb(12, 68, 124);stroke:rgb(133, 183, 235);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
    <text x="340" y="132" text-anchor="middle" dominant-baseline="central" style="fill:rgb(181, 212, 244);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Feature extractor per frame (CNN)</text>
  </g>

  <!-- ResNet18 inner box -->
  <g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
    <rect x="108" y="148" width="200" height="88" rx="8" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
    <text x="208" y="178" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">ResNet18</text>
    <text x="208" y="196" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">ImageNet pretrained</text>
    <text x="208" y="212" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">layer4 unfrozen after epoch 10</text>
  </g>

  <!-- Arrow inside CNN -->
  <line x1="308" y1="192" x2="352" y2="192" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

  <!-- Frame features box -->
  <g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
    <rect x="352" y="148" width="140" height="54" rx="8" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
    <text x="422" y="168" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Frame features</text>
    <text x="422" y="186" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">B × T × 512</text>
  </g>

  <!-- Arrow inside CNN -->
  <line x1="492" y1="192" x2="532" y2="192" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

  <!-- Dropout box -->
  <g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
    <rect x="532" y="162" width="50" height="58" rx="8" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
    <text x="557" y="184" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Drop</text>
    <text x="557" y="202" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">0.5</text>
  </g>

  <!-- Arrow 2 -->
  <line x1="340" y1="256" x2="340" y2="292" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

  <!-- GRU container -->
  <g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
    <rect x="80" y="292" width="520" height="144" rx="12" stroke-width="0.5" style="fill:rgb(114, 36, 62);stroke:rgb(237, 147, 177);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
    <text x="340" y="316" text-anchor="middle" dominant-baseline="central" style="fill:rgb(244, 192, 209);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Sequence modeling (RNN)</text>
  </g>

  <!-- Bi-GRU box -->
  <g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
    <rect x="108" y="332" width="180" height="84" rx="8" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
    <text x="198" y="362" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Bidirectional GRU</text>
    <text x="198" y="380" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">hidden = 128</text>
    <text x="198" y="396" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">concat fwd + bwd</text>
  </g>

  <!-- Arrow inside GRU -->
  <line x1="288" y1="374" x2="336" y2="374" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

  <!-- FC + Classifier box -->
  <g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
    <rect x="336" y="332" width="236" height="84" rx="8" stroke-width="0.5" style="fill:rgb(68, 68, 65);stroke:rgb(180, 178, 169);color:rgb(255, 255, 255);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
    <text x="454" y="358" text-anchor="middle" dominant-baseline="central" style="fill:rgb(211, 209, 199);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">FC (dropout 0.5)</text>
    <text x="454" y="376" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">B × 256</text>
    <text x="454" y="394" text-anchor="middle" dominant-baseline="central" style="fill:rgb(180, 178, 169);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Linear classifier → B × 10</text>
  </g>

  <!-- Arrow 3 -->
  <line x1="340" y1="436" x2="340" y2="458" marker-end="url(#arrow)" style="fill:none;stroke:rgb(156, 154, 146);color:rgb(255, 255, 255);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

  <!-- Output -->
  <text x="340" y="472" text-anchor="middle" dominant-baseline="central" style="fill:rgb(250, 249, 245);stroke:none;color:rgb(255, 255, 255);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">10-class prediction</text>
</svg>![resnet18_gru_architecture](https://github.com/user-attachments/assets/9154ce6f-98d1-48bf-b834-d82e942e6628)
`

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

*Example of a preprocessed sequence: 7 real mouth-crop frames (t=0–6) zero-padded with black frames to reach the fixed length of T=22.*

<img width="1300" height="870" alt="Temporal padding example" src="https://github.com/user-attachments/assets/4038bc0c-e8f0-4d46-b977-6adcab4ce4ec" />

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
