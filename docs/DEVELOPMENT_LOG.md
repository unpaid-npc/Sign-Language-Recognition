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

---

## Day 2 — PyTorch Dataset, BiLSTM Architecture, Training & Evaluation
**Date:** November 5, 2026  
**Status:** Completed & Verified

### Accomplishments
1. **PyTorch Dataset & Spatial-Temporal Augmentations (`src/data/dataset.py`):**
   - Implemented `WLASLDataset` with RAM preloading (`1,013` tensors) and robust path resolution across `PROJECT_ROOT` and `DATA_DIR`.
   - Engineered physically-valid 3D landmark augmentations (random temporal speed perturbation, $\pm 12^\circ$ 2D rotation around shoulder center, $\pm 15\%$ anisotropic scaling, $\pm 0.04$ spatial translation, Gaussian sensor noise, and random hand-frame dropout) while preserving missing-hand zero masks.
   - Implemented smoothed inverse square-root class-frequency weighting to handle sample-count variance across the 100 glosses.
2. **Attention-Enhanced BiLSTM Model (`src/models/lstm.py`):**
   - Implemented `SignBiLSTMClassifier` (`800,548` trainable parameters) featuring automatic first-order velocity computation ($\Delta \mathbf{x}_t = \mathbf{x}_t - \mathbf{x}_{t-1}$), 288-D kinematic input projection, a 2-layer Bidirectional LSTM backbone, and dual temporal pooling (`TemporalAttentionPool` + `GlobalMaxPool`).
3. **Reproducible Model Training (`src/models/train.py`):**
   - Trained on CPU in `111.1s` (early stopping triggered at Epoch 79; best checkpoint saved at **Epoch 54**).
   - Achieved **41.48% Validation Top-1 Accuracy** and **72.73% Validation Top-5 Accuracy** (peaking at 76.1% Top-5 at Epoch 70).
   - Saved `models/best_bilstm_wlasl100.pt` and `outputs/training_curves.png`.
4. **Comprehensive Test Set Evaluation (`src/models/evaluate.py`):**
   - Evaluated `best_bilstm_wlasl100.pt` on the 128 unseen test sequences across 100 classes:
     - **Test Top-1 Accuracy:** `39.06%`
     - **Test Top-5 Accuracy:** `67.19%`
     - **Test Macro Precision / Recall / F1:** `30.83%` / `39.17%` / `32.72%`
     - **Test Weighted F1-Score:** `33.76%`
   - Generated `outputs/confusion_matrix.png`, `outputs/per_class_metrics.csv`, and `outputs/evaluation_metrics.json`.
5. **Unit Testing (`tests/test_model_and_dataset.py`):**
   - Executed `pytest -v`: **13 / 13 tests passed** in 19.19s.

---

   ## Day 3 — Real-Time Webcam Recognition & Inference Pipeline
**Date:** November 5, 2026  
**Status:** Completed & Verified

### Accomplishments
1. **Sliding-Window Inference Engine (`src/inference/predictor.py`):**
   - Implemented `RealTimeSignPredictor` managing a 30-frame rolling landmark deque (`maxlen=30`), a hand-presence tracking deque, and a 6-frame probability smoothing history.
   - Enforced strict completeness gating (blocking predictions when `len(buffer) < 30`) and hand-presence gating (`WAITING FOR HANDS` when hands are absent in `>60%` of recent frames).
   - Implemented linearly-weighted moving average probability smoothing across consecutive frames and configurable confidence thresholding (`Uncertain / Unknown` rejection below threshold) with Top-3 candidate extraction.
2. **Real-Time OpenCV Webcam Application (`app/webcam_app.py`):**
   - Integrated live webcam capture (`cv2.CAP_DSHOW` with standard fallback), unmirrored MediaPipe Holistic extraction (preserving true Left/Right handedness), and mirrored selfie HUD rendering.
   - Built interactive Heads-Up Display (HUD) displaying status badge, primary recognized sign, confidence bar with threshold marker, Top-3 candidates leaderboard, shoulder framing indicator, Picture-in-Picture (PiP) reference sign video player, and live FPS counter.
   - Added hotkeys for resetting buffers (`R`), toggling landmarks (`L`), toggling/cycling reference signs (`P`/`N`/`B`), adjusting confidence threshold (`+`/`-`), and saving screenshots (`S`).
3. **Unit Testing (`tests/test_inference_and_webcam.py`):**
   - Added 6 unit tests covering incomplete sequence blocking, hand-presence gating, unknown prediction handling, high-confidence recognition, HUD rendering, and webcam error handling.
   - Executed `pytest -v`: **19 / 19 tests passed**.

---

## Day 4 — Application Refinement, Documentation & Repository Finalization
**Date:** November 5, 2026  
**Status:** Completed & Verified

### Accomplishments
1. **Live Webcam Handedness Fix & PiP Reference Player (`app/webcam_app.py`):**
   - Fixed horizontal mirror-flip ordering so MediaPipe Holistic processes the unflipped frame first (preserving true Left/Right hand coordinates matching WLASL training clips) before flipping the display canvas for selfie viewing.
   - Added a real-time shoulder visibility check (`check_shoulders_in_frame`) and an interactive Picture-in-Picture (PiP) reference sign video player (`ReferenceSignPlayer`) allowing users to cycle through top WLASL signs (`[N]`/`[B]`) while signing.
2. **Complete Technical & Interview Documentation (`docs/` & `notes/`):**
   - Created all 6 engineering guides in `docs/` (`PROJECT_ARCHITECTURE.md`, `DATASET_GUIDE.md`, `MODEL_EXPLANATION.md`, `REALTIME_PIPELINE.md`, `TESTING.md`, `DEVELOPMENT_LOG.md`).
   - Created all 6 learning and interview reference guides in `notes/` (`PROJECT_OVERVIEW.md`, `TECHNOLOGY_NOTES.md`, `COMMAND_REFERENCE.md`, `CODE_EXPLANATION.md`, `LEARNING_NOTES.md`, `NEXT_STEPS.md`).
3. **End-to-End Verification:**
   - Executed full test suite (`pytest -v`): **19 / 19 tests passed**.
   - Captured live demonstration screenshots in `outputs/screenshots/` and finalized root `README.md`.