### File 2: `docs/DEVELOPMENT_LOG.md`
* **Exact Location:** `C:\Projects\sign-language-recognition\docs\DEVELOPMENT_LOG.md`
* **Action:** Create `DEVELOPMENT_LOG.md` inside `docs/` and paste the contents below.

```markdown
# Development Log — Real-Time Sign Language Recognition (WLASL-100)

## Day 1 — Environment Setup, WLASL-100 Acquisition & MediaPipe Preprocessing
**Date:** November 4–5, 2026  
**Status:** Completed & Verified

### Accomplishments
1. **Repository & Environment Initialization:**
   - Created modular project hierarchy (`app/`, `src/`, `data/`, `models/`, `tests/`, `docs/`, `notes/`, `outputs/`).
   - Configured isolated Python `3.12.10` virtual environment (`.venv`) to ensure native wheel compatibility with `mediapipe==0.10.21`, `opencv-python==4.10.0.84`, and `torch==2.11.0+cpu`.
2. **WLASL-100 Dataset Acquisition (`src/data/download_wlasl.py`):**
   - Downloaded official `WLASL_v0.3.json` metadata (100 glosses, 2,038 annotated instances across 97 unique signers).
   - Automated retrieval of pre-cropped `.mp4` videos from the `risangbaskoro/wlasl-processed` archive via `kagglehub`, filtering strictly to WLASL-100 (`1,013` available `.mp4` videos across all 100 glosses) and cleaning temporary cache.
3. **MediaPipe Holistic Landmark Extraction (`src/landmarks/extraction.py`):**
   - Designed a compact 144-dimensional per-frame feature vector combining 6 upper-body pose anchors (`[0, 9, 11, 12, 13, 14]` -> 18 features), 21 left-hand landmarks (63 features), and 21 right-hand landmarks (63 features).
   - Implemented body-centric spatial normalization (centered at the mid-shoulder point and scaled by 2D shoulder width) for translation and scale invariance.
   - Implemented uniform temporal resampling to convert variable-length clips into fixed `(30, 144)` `float32` tensors.
4. **Dataset Preprocessing & Leakage Prevention (`src/data/preprocessing.py`):**
   - Extracted and cached all `1,013` sequences to `data/processed/sequences/*.npy` (57m 05s on CPU via XNNPACK).
   - Verified zero `video_id` overlap across splits and balanced classes with missing validation/test clips, yielding **709 Train / 176 Val / 128 Test** sequences across **100 / 100 classes**.
5. **Unit Testing (`tests/test_landmarks_and_preprocessing.py`):**
   - Executed `pytest tests/test_landmarks_and_preprocessing.py -v`: **9 / 9 tests passed** in 6.15s.