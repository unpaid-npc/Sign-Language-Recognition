# Codebase Walkthrough (`src/` and `app/`)

## 1. `src/landmarks/extraction.py`
- `extract_raw_landmarks_from_results(results)`: Reads `results.pose_landmarks`, `results.left_hand_landmarks`, and `results.right_hand_landmarks`, returning `(6, 3)`, `(21, 3)`, and `(21, 3)` `float32` coordinate arrays plus a boolean `hands_detected` flag.
- `normalize_frame_landmarks(pose, lh, rh)`: Computes the midpoint between Left Shoulder (`pose[2]`) and Right Shoulder (`pose[3]`) and divides all present coordinates by the 2D inter-shoulder distance. Keeps missing hands at `0.0`.
- `resample_or_pad_sequence(sequence, target_length=30)`: Uses `np.linspace(0, N - 1, 30)` to uniformly sample 30 frames from any variable-length video clip.

## 2. `src/data/dataset.py`
- `augment_landmark_sequence(sequence)`: Reshapes `(30, 144)` into `(30, 48, 3)`, records a boolean mask of non-zero landmarks, applies random speed cropping, 2D rotation ($\pm 12^\circ$), anisotropic scaling ($\pm 15\%$), translation jitter, and Gaussian noise, and restores the zero mask for missing hands.
- `WLASLDataset`: Preloads all `.npy` sequences into memory at initialization so 100 epochs of training complete in under 2 minutes on CPU.

## 3. `src/models/lstm.py`
- `TemporalAttentionPool`: A 2-layer MLP (`Linear -> Tanh -> Linear -> Softmax`) that computes a scalar importance weight $\alpha_t$ for each of the 30 time steps and outputs $\sum_t \alpha_t \mathbf{h}_t$.
- `SignBiLSTMClassifier`: Computes velocity $\Delta \mathbf{x}_t = \mathbf{x}_t - \mathbf{x}_{t-1}$, projects the 288-D vector to 128-D, runs a 2-layer BiLSTM, concatenates attention pooling and max pooling (`512-D`), and outputs 100 class logits.

## 4. `src/inference/predictor.py` & `app/webcam_app.py`
- `RealTimeSignPredictor.update()`: Appends each frame's 144-D vector to `landmark_buffer`, checks recent hand presence (`>=40%`) and buffer completeness (`30/30`), runs `model(seq_tensor)`, smooths probabilities across the last 6 frames, and compares the top probability against `confidence_threshold`.
- `run_webcam_recognition()`: Extracts landmarks on the unflipped camera frame (so physical right hand = WLASL right hand), flips the visual image for natural selfie viewing, overlays the Picture-in-Picture reference sign player, and renders the HUD.