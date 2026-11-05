# Deep Learning Model Explanation (`SignBiLSTMClassifier`)

## 1. Feature Representation ($F = 144 \rightarrow 288$)

Each frame $t \in \{1, \dots, 30\}$ is represented by $48$ 3D landmarks extracted via MediaPipe Holistic:
- **6 Upper-Body Pose Landmarks ($6 \times 3 = 18$ features):** Nose (`0`), Mouth-Left (`9`), Left Shoulder (`11`), Right Shoulder (`12`), Left Elbow (`13`), Right Elbow (`14`).
- **21 Left-Hand Landmarks ($21 \times 3 = 63$ features).**
- **21 Right-Hand Landmarks ($21 \times 3 = 63$ features).**

### Body-Centric Spatial Normalization
Given Left Shoulder $\mathbf{P}_{11}$ and Right Shoulder $\mathbf{P}_{12}$:
$$\mathbf{C} = \frac{\mathbf{P}_{11} + \mathbf{P}_{12}}{2}, \quad d = \Vert{}\mathbf{P}_{11}^{(xy)} - \mathbf{P}_{12}^{(xy)}\Vert{}_2 + \epsilon$$
Every detected landmark $\mathbf{L}_i$ is normalized via $\hat{\mathbf{L}}_i = (\mathbf{L}_i - \mathbf{C}) / d$. Undetected hands remain strictly $\mathbf{0}$.

### Kinematic Velocity Augmentation
Inside `SignBiLSTMClassifier.compute_kinematic_features()`, first-order temporal velocity is computed on the fly:
$$\Delta \mathbf{x}_t = \mathbf{x}_t - \mathbf{x}_{t-1} \quad (\text{with } \Delta \mathbf{x}_1 = \mathbf{0})$$
Concatenating position and velocity yields a $288$-dimensional kinematic vector per frame.

---

## 2. Network Architecture

1. **Input Projection Stem:** `Linear(288 -> 128)` $\rightarrow$ `LayerNorm(128)` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(0.15)`.
2. **Recurrent Backbone:** 2-layer **Bidirectional LSTM** (`input_size=128`, `hidden_size=128`, `dropout=0.30`). Outputs $\mathbf{H} \in \mathbb{R}^{B \times 30 \times 256}$.
3. **Dual Temporal Pooling:**
   - **Temporal Attention Pooling (`TemporalAttentionPool`):** Learns frame importance weights $\alpha_t = \text{Softmax}(\mathbf{w}_2^\top \tanh(\mathbf{W}_1 \mathbf{h}_t))$ to emphasize the sign's core stroke frames ($\mathbf{c}_{\text{attn}} \in \mathbb{R}^{256}$).
   - **Global Temporal Max Pooling:** Captures peak activation across time ($\mathbf{c}_{\text{max}} \in \mathbb{R}^{256}$).
4. **Classification Head:** Concatenates $[\mathbf{c}_{\text{attn}}; \mathbf{c}_{\text{max}}] \in \mathbb{R}^{512}$ $\rightarrow$ `Linear(512 -> 128)` $\rightarrow$ `LayerNorm` $\rightarrow$ `GELU` $\rightarrow$ `Dropout(0.40)` $\rightarrow$ `Linear(128 -> 100)`.
5. **Total Trainable Parameters:** `800,548`.

---

## 3. Training Methodology & Actual Evaluation Metrics

- **Optimizer & Schedule:** `AdamW` ($\text{lr}=10^{-3}$, $\text{weight\_decay}=10^{-4}$) with `CosineAnnealingLR` over 100 epochs (early stopping patience = 25) and gradient norm clipping (`max_norm = 1.0`).
- **Loss Function:** Class-balanced `CrossEntropyLoss` with `label_smoothing = 0.05`.

### Actual Benchmark Results (`best_bilstm_wlasl100.pt`, Epoch 54)

| Metric | Validation Split (`176` sequences) | Unseen Test Split (`128` sequences) |
| :--- | :--- | :--- |
| **Top-1 Accuracy** | **41.48%** | **39.06%** |
| **Top-5 Accuracy** | **72.73%** | **67.19%** |
| **Macro Precision** | — | **30.83%** |
| **Macro Recall** | — | **39.17%** |
| **Macro F1-Score** | — | **32.72%** |
| **Weighted F1-Score** | — | **33.76%** |

*Technical Context:* On 100 classes, random guessing yields `1.00%` Top-1 accuracy and `5.00%` Top-5 accuracy. Achieving **39.06% Top-1** and **67.19% Top-5** on unseen signers with an average of only `7.09` training sequences per class demonstrates strong temporal feature learning while remaining lightweight enough for `<1.5 ms` CPU inference.