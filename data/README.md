# Dataset Directory (`data/`)

This directory stores the raw **WLASL-100** dataset annotations/videos and the preprocessed **MediaPipe Holistic** landmark sequences. Large binary files (`.mp4`, `.npy`) are excluded from version control via `.gitignore`.

## Directory Structure

```text
data/
├── raw/
│   ├── WLASL_v0.3.json          # Official WLASL v0.3 annotation metadata
│   └── videos/                  # Pre-trimmed WLASL-100 .mp4 clips (1,013 local videos)
├── processed/
│   ├── label_map.json           # Bidirectional mapping (class index 0..99 <-> gloss)
│   ├── metadata.csv             # Leakage-free Train/Val/Test split manifest
│   └── sequences/               # Extracted (30, 144) float32 .npy landmark tensors
└── README.md