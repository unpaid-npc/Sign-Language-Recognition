# Complete Windows PowerShell Command Reference

Every command below is executed from the project root directory:

```text
C:\Projects\sign-language-recognition
```

## 1. Environment Setup Commands

### Create Virtual Environment with Python 3.12

```powershell
py -3.12 -m venv .venv
```

- **Purpose:** Creates an isolated Python 3.12 virtual environment inside `.venv/`.
- **Arguments:** `-3.12` selects Python 3.12 via the Windows Python launcher; `-m venv .venv` runs the `venv` module targeting `.venv`.
- **Expected output:** Creates `.venv\Scripts\python.exe` and `.venv\Lib\site-packages`.

### Activate Virtual Environment

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\.venv\Scripts\Activate.ps1
```

- **Purpose:** Temporarily enables script execution in the current PowerShell window and activates `.venv`.
- **Expected output:** Prompt changes to `(.venv) PS C:\Projects\sign-language-recognition>`.

### Install Dependencies

```powershell
pip install -r requirements.txt
```

- **Purpose:** Installs pinned versions of OpenCV, MediaPipe, PyTorch, NumPy, Pandas, Scikit-Learn, Matplotlib, Pytest, and KaggleHub.

## 2. Dataset & Preprocessing Commands

### Download and Inspect WLASL-100

```powershell
python -m src.data.download_wlasl --download-videos --clean-cache
```

- **Purpose:** Downloads `WLASL_v0.3.json`, downloads the Kaggle WLASL video archive, copies the 1,013 WLASL-100 `.mp4` clips into `data/raw/videos/`, and removes the temporary cache.
- **Arguments:**
  - `--download-videos`: Triggers video archive download via KaggleHub.
  - `--clean-cache`: Removes the 4.8 GB temporary cache after copying WLASL-100 clips.

### Extract MediaPipe Holistic Landmarks

```powershell
python -m src.data.preprocessing
```

- **Purpose:** Extracts normalized `(30, 144)` landmark tensors from all local `.mp4` videos into `data/processed/sequences/*.npy` and generates `metadata.csv` and `label_map.json`.
- **Optional flags:** `--max-videos 10` (for quick smoke testing) and `--force` (to overwrite existing `.npy` files).

## 3. Model Training & Evaluation Commands

### Train the BiLSTM Classifier

```powershell
python -m src.models.train --epochs 100 --patience 25
```

- **Purpose:** Trains `SignBiLSTMClassifier` on the `train` split, validates on the `val` split, saves `models/best_bilstm_wlasl100.pt`, and plots `outputs/training_curves.png`.

### Evaluate on Unseen Test Set

```powershell
python -m src.models.evaluate
```

- **Purpose:** Evaluates `models/best_bilstm_wlasl100.pt` on the 128 test sequences, printing Top-1/Top-5 accuracy, Precision, Recall, F1, and saving `outputs/confusion_matrix.png` and `outputs/per_class_metrics.csv`.

## 4. Real-Time Webcam & Testing Commands

### Launch Live Webcam Recognition

```powershell
python app/webcam_app.py
```

- **Purpose:** Opens the live OpenCV webcam HUD with MediaPipe skeletal tracking, PiP reference sign player, and live BiLSTM recognition.
- **Arguments:** `--camera 0` (webcam index), `--threshold 0.35` (confidence cutoff), `--video <path>` (optional `.mp4` file input).

### Run Automated Unit Test Suite

```powershell
pytest -v
```

- **Purpose:** Executes all 19 unit tests across `tests/`.
- **Expected output:** `19 passed`.
