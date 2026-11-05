# Real-Time Sign Language Recognition Using MediaPipe and WLASL-100

A modular, CPU-friendly real-time isolated American Sign Language (ASL) recognition system built with **Python**, **MediaPipe Holistic**, **PyTorch (BiLSTM)**, **OpenCV**, and **Streamlit** on the **WLASL-100** benchmark.

> **Scope Note:** WLASL-100 contains isolated word-level ASL signs (100 glosses). This system performs isolated word-level sign recognition from live webcam sequences, not continuous full-sentence sign language translation.

---

## Project Architecture

```text
Live Webcam / WLASL Video (.mp4)
       │
       ▼
OpenCV Frame Capture ──► MediaPipe Holistic (6 Upper Pose + 21 Left Hand + 21 Right Hand)
                                │
                                ▼
             Body-Centric Spatial Normalization (144-D Vector / Frame)
                                │
                                ▼
             Sliding Sequence Buffer / Temporal Resampler (30 × 144)
                                │
                                ▼
               PyTorch Bidirectional LSTM + Temporal Attention
                                │
                                ▼
           Confidence Thresholding + EMA Smoothing ──► Real-Time HUD Overlay