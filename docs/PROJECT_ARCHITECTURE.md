# System Architecture — Real-Time Sign Language Recognition (WLASL-100)

## 1. High-Level Overview

This system implements **Isolated Sign Language Recognition (ISLR)** on the **WLASL-100** benchmark. Instead of classifying raw RGB video frames with computationally heavy 3D Convolutional Networks (I3D) or Video Vision Transformers, the architecture decouples **spatial skeletal estimation** from **temporal sequence classification**:

1. **Spatial Feature Extractor (MediaPipe Holistic):** Compresses each $640 \times 480 \times 3$ RGB video frame ($921,600$ values) into a normalized $144$-dimensional skeletal coordinate vector representing upper-body posture and both 3D hand skeletons.
2. **Temporal Sequence Classifier (PyTorch BiLSTM + Attention):** Processes a fixed $30$-frame temporal window ($30 \times 144$) enriched with first-order frame-to-frame velocity ($\Delta \mathbf{x}_t$, yielding $30 \times 288$) to classify the isolated sign across 100 ASL glosses.

---

## 2. End-to-End Data Flow

```text
+-------------------------------------------------------------------------+
|                        OFFLINE TRAINING PIPELINE                        |
+-------------------------------------------------------------------------+
  WLASL_v0.3.json + Raw .mp4 Videos (1,013 local clips across 100 classes)
         │
         ▼
  [src/landmarks/extraction.py] -> HolisticLandmarkExtractor
    • Extracts 6 Upper-Body Pose points + 21 Left-Hand + 21 Right-Hand points
    • Centers at Mid-Shoulder & scales by 2D Shoulder Width -> (144,) float32
    • Uniformly resamples temporal length to T = 30 frames -> (30, 144)
         │
         ▼
  [src/data/preprocessing.py]
    • Caches sequences to data/processed/sequences/<video_id>.npy
    • Verifies zero video_id leakage across splits (709 Train / 176 Val / 128 Test)
         │
         ▼
  [src/data/dataset.py] -> WLASLDataset & DataLoaders
    • Preloads tensors into RAM for fast CPU training
    • Applies stochastic 3D coordinate augmentations (rotation, scaling, shift, noise)
         │
         ▼
  [src/models/lstm.py & src/models/train.py] -> SignBiLSTMClassifier
    • Computes kinematic velocity features (144 pos + 144 vel = 288-D)
    • 2-Layer BiLSTM + Temporal Attention + Max Pooling -> 100-class logits
    • Saves best checkpoint: models/best_bilstm_wlasl100.pt

+-------------------------------------------------------------------------+
|                      ONLINE REAL-TIME INFERENCE                         |
+-------------------------------------------------------------------------+
  Live Webcam Frame (OpenCV BGR)
         │
         ▼
  Unmirrored MediaPipe Holistic Extraction -> (144,) Normalized Vector
         │
         ▼
  [src/inference/predictor.py] -> RealTimeSignPredictor
    • Gate 1: Hand Presence Check (>=40% recent frames must contain hands)
    • Gate 2: Sequence Completeness Check (buffer length must equal 30)
    • BiLSTM Forward Pass -> 100-class Softmax Probabilities
    • Temporal Smoothing (6-frame weighted moving average)
    • Gate 3: Confidence Threshold Check (>= threshold -> RECOGNIZED, else UNCERTAIN)
         │
         ▼
  [app/webcam_app.py] -> Mirrored Selfie View + OpenCV HUD Overlay