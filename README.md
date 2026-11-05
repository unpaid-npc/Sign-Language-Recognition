# Real-Time Sign Language Recognition Using MediaPipe and WLASL-100

A modular, CPU-friendly real-time isolated American Sign Language (ASL) recognition system built with **Python**, **MediaPipe Holistic**, **PyTorch (BiLSTM + Temporal Attention)**, and **OpenCV** on the **WLASL-100** benchmark.

> **Scope Note:** WLASL-100 contains isolated word-level ASL signs (100 glosses). This system performs isolated word-level sign recognition from live webcam sequences, not continuous full-sentence sign language translation.

---

## Key Features

- **MediaPipe Holistic Skeletal Extraction:** Extracts 48 3D upper-body and dual-hand landmarks (`144` features/frame) normalized relative to the signer's shoulder center and width for translation and scale invariance.
- **Kinematic BiLSTM + Temporal Attention (`SignBiLSTMClassifier`):** Combines normalized 3D positions with frame-to-frame velocity ($\Delta \mathbf{x}_t$, `288-D`), a 2-layer Bidirectional LSTM, and learnable Temporal Attention Pooling (`800,548` parameters).
- **Real-Time OpenCV HUD & Reference Sign Player:** Features a 30-frame sliding buffer, hand-presence gating, 6-frame moving-average smoothing, confidence thresholding, Top-3 candidate leaderboard, and an interactive Picture-in-Picture (PiP) WLASL reference video player.
- **Reproducible & Fully Tested:** Includes automated dataset acquisition via KaggleHub, leakage-free split verification, and **19 unit tests** (`pytest`).

---

## Actual WLASL-100 Benchmark Results

Trained on CPU in `111.1s` across `1,013` local WLASL-100 video sequences (`709` Train / `176` Val / `128` Test across all `100` classes):

| Metric | Validation Split (`176` clips) | Unseen Test Split (`128` clips) | Random Baseline (100 Classes) |
| :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | **41.48%** | **39.06%** | 1.00% |
| **Top-5 Accuracy** | **72.73%** | **67.19%** | 5.00% |
| **Macro Precision** | — | **30.83%** | 1.00% |
| **Macro Recall** | — | **39.17%** | 1.00% |
| **Macro F1-Score** | — | **32.72%** | 1.00% |
| **Weighted F1-Score** | — | **33.76%** | 1.00% |

---

## Repository Structure

```text
sign-language-recognition/
├── app/
│   └── webcam_app.py              # Live OpenCV HUD + PiP reference sign player
├── src/
│   ├── config.py                  # Central hyperparameters, paths, and constants
│   ├── data/
│   │   ├── download_wlasl.py      # Automated WLASL-100 JSON & video acquisition
│   │   ├── preprocessing.py       # Video-to-landmark (.npy) extraction & split verification
│   │   └── dataset.py             # PyTorch Dataset, DataLoaders & 3D augmentations
│   ├── landmarks/
│   │   └── extraction.py          # MediaPipe Holistic wrapper & shoulder normalizer
│   ├── models/
│   │   ├── lstm.py                # BiLSTM + Temporal Attention architecture
│   │   ├── train.py               # Reproducible training & validation pipeline
│   │   └── evaluate.py            # Test metrics, per-class CSV & confusion matrix
│   ├── inference/
│   │   └── predictor.py           # Sliding buffer, hand-presence gate & EMA smoothing
│   └── utils/
│       └── logger.py              # Standardized console and file logging
├── data/                          # Raw videos & processed .npy sequences (Git-ignored)
├── models/                        # Saved PyTorch checkpoints (Git-ignored)
├── outputs/                       # Training curves, confusion matrix, metrics & screenshots
├── tests/                         # 19 Pytest unit tests covering all modules
├── docs/                          # Engineering & architectural documentation (6 guides)
├── notes/                         # Interview prep, theory & command reference (6 guides)
├── requirements.txt               # Pinned dependencies for Python 3.12
└── README.md