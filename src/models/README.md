# Trained Model Checkpoints (`models/`)

This directory stores trained PyTorch model checkpoints (`.pt`). Binary weights are excluded from Git via `.gitignore` to keep the repository lightweight.

## Saved Checkpoints

- `best_bilstm_wlasl100.pt`: Best validation checkpoint saved at **Epoch 54**.
- `last_bilstm_wlasl100.pt`: Final epoch checkpoint saved at **Epoch 79** (early stopping).

## Model Architecture (`SignBiLSTMClassifier`)

| Component | Specification |
| :--- | :--- |
| **Input Sequence** | `(Batch, 30, 144)` normalized MediaPipe Holistic landmarks |
| **Kinematic Augmentation** | Position $\mathbf{x}_t$ (144-D) + Velocity $\Delta\mathbf{x}_t$ (144-D) = `288-D` per frame |
| **Input Projection** | `Linear(288 -> 128)` + `LayerNorm` + `GELU` + `Dropout(0.15)` |
| **Recurrent Backbone** | 2-Layer Bidirectional LSTM (`hidden_dim=128`, `256-D` bidirectional output) |
| **Temporal Aggregation** | Concat(`TemporalAttentionPool(256)`, `GlobalMaxPool(256)`) = `512-D` |
| **Classification Head** | `Linear(512 -> 128)` + `LayerNorm` + `GELU` + `Dropout(0.40)` + `Linear(128 -> 100)` |
| **Trainable Parameters** | `800,548` |

## Actual WLASL-100 Benchmark Results

| Metric | Validation Set (`176` clips) | Test Set (`128` clips) |
| :--- | :--- | :--- |
| **Top-1 Accuracy** | **41.48%** | **39.06%** |
| **Top-5 Accuracy** | **72.73%** | **67.19%** |
| **Macro Precision** | — | **30.83%** |
| **Macro Recall** | — | **39.17%** |
| **Macro F1-Score** | — | **32.72%** |
| **Weighted F1-Score** | — | **33.76%** |

## Reproducing Training and Evaluation

```powershell
# Train BiLSTM model from scratch
python -m src.models.train --epochs 100 --patience 25

# Evaluate best checkpoint on unseen test split
python -m src.models.evaluate