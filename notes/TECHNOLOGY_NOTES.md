# Technology Stack Notes

## 1. MediaPipe Holistic (`mediapipe==0.10.21`)
- **What it is:** Google's cross-platform ML framework that chains multiple lightweight TFLite models (BlazePose + BlazePalm + HandLandmark) using a region-of-interest (ROI) tracking pipeline.
- **How it works:** Instead of running heavy hand detection on every full frame, Holistic first estimates the body pose, crops around the predicted wrist locations, and runs the 21-point 3D hand landmark model on those crops using CPU XNNPACK acceleration.
- **Coordinates $(x, y, z)$:** $x$ and $y$ are normalized image coordinates in $[0.0, 1.0]$; $z$ represents depth relative to the wrist (for hands) or hips (for pose).

## 2. Sequence Modeling: RNN vs. LSTM vs. BiLSTM
- **Why Sequence Modeling?** A static image can recognize finger-spelling (letters A–Z), but word-level ASL signs are defined by **temporal trajectory** (where the hands start, how they move, and where they end).
- **LSTM (Long Short-Term Memory):** Solves the vanishing gradient problem of vanilla RNNs using a cell state $\mathbf{c}_t$ and three multiplicative gates:
  - **Forget Gate ($\mathbf{f}_t$):** Decides what past context to discard.
  - **Input Gate ($\mathbf{i}_t$):** Decides what new frame information to store.
  - **Output Gate ($\mathbf{o}_t$):** Controls what information flows into the hidden state $\mathbf{h}_t$.
- **BiLSTM (Bidirectional LSTM):** Processes the 30-frame sequence both forward ($t=1 \rightarrow 30$) and backward ($t=30 \rightarrow 1$), concatenating hidden states $[\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]$. This allows every frame's representation to be informed by both the preparation and completion of the sign.

## 3. Evaluation Metrics Explained
- **Accuracy (Top-1 & Top-5):** Top-1 measures how often the highest-probability class equals the true sign. Top-5 measures how often the true sign is among the model's top 5 guesses.
- **Precision:** $\frac{TP}{TP + FP}$ — When the model predicts sign $X$, how often is it actually sign $X$?
- **Recall:** $\frac{TP}{TP + FN}$ — Out of all true instances of sign $X$, what fraction did the model successfully detect?
- **F1-Score:** $2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ — The harmonic mean of Precision and Recall.