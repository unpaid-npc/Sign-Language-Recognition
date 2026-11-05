# Real-Time Webcam Inference Pipeline

## 1. End-to-End Execution Pipeline

```text
Webcam (OpenCV) ──► Unmirrored BGR Frame (640×480)
                         │
                         ▼
             MediaPipe Holistic (CPU XNNPACK)
                         │
                         ▼
         Body-Centric Spatial Normalization (144-D)
                         │
                         ▼
           Sliding Sequence Buffer (deque maxlen=30)
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
 [Hand Presence Gate]         [Completeness Gate]
 (<40% hands -> STANDBY)      (<30 frames -> BUFFERING)
          │                             │
          └──────────────┬──────────────┘
                         ▼
          PyTorch BiLSTM Forward Pass (1, 30, 144)
                         │
                         ▼
         6-Frame Weighted Moving Average Smoothing
                         │
                         ▼
       Confidence Thresholding (default τ = 0.35–0.45)
        ├── Conf >= τ  ──► STATUS_RECOGNIZED (Green HUD)
        └── Conf <  τ  ──► STATUS_UNCERTAIN ("Uncertain / Unknown")