"""
Central configuration module for Real-Time Sign Language Recognition (WLASL-100).
Defines directory paths, landmark indices, sequence parameters, and hyperparameters.
"""

from pathlib import Path
from typing import List

# ==============================================================================
# 1. PROJECT DIRECTORY PATHS
# ==============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
RAW_VIDEOS_DIR: Path = RAW_DATA_DIR / "videos"
WLASL_JSON_PATH: Path = RAW_DATA_DIR / "WLASL_v0.3.json"

PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SEQUENCES_DIR: Path = PROCESSED_DATA_DIR / "sequences"
METADATA_CSV_PATH: Path = PROCESSED_DATA_DIR / "metadata.csv"
LABEL_MAP_PATH: Path = PROCESSED_DATA_DIR / "label_map.json"

MODELS_DIR: Path = PROJECT_ROOT / "models"
BEST_MODEL_PATH: Path = MODELS_DIR / "best_bilstm_wlasl100.pt"
LAST_MODEL_PATH: Path = MODELS_DIR / "last_bilstm_wlasl100.pt"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
LOG_FILE_PATH: Path = OUTPUTS_DIR / "project.log"

# Ensure critical directories exist
for directory in [
    RAW_DATA_DIR,
    RAW_VIDEOS_DIR,
    PROCESSED_DATA_DIR,
    SEQUENCES_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. DATASET & LANDMARK EXTRACTION PARAMETERS
# ==============================================================================
# WLASL subset size (100 for WLASL-100)
NUM_CLASSES: int = 100

# Fixed sequence length (frames per sign video)
SEQUENCE_LENGTH: int = 30

# Minimum valid frames required in a raw video to keep it
MIN_VIDEO_FRAMES: int = 5

# 6 Upper-body Pose Landmarks from MediaPipe Holistic (out of 33 pose landmarks):
# 0: Nose, 9: Mouth Left (Face anchor), 11: Left Shoulder, 12: Right Shoulder,
# 13: Left Elbow, 14: Right Elbow
SELECTED_POSE_INDICES: List[int] = [0, 9, 11, 12, 13, 14]

NUM_POSE_LANDMARKS: int = len(SELECTED_POSE_INDICES)  # 6
NUM_HAND_LANDMARKS: int = 21  # 21 per hand
COORDS_PER_LANDMARK: int = 3  # (x, y, z)

# Total features per frame: (6 * 3) + (21 * 3) + (21 * 3) = 18 + 63 + 63 = 144
POSE_FEATURE_DIM: int = NUM_POSE_LANDMARKS * COORDS_PER_LANDMARK      # 18
HAND_FEATURE_DIM: int = NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK      # 63
FEATURE_DIM: int = POSE_FEATURE_DIM + (2 * HAND_FEATURE_DIM)          # 144

# MediaPipe detection and tracking confidence thresholds
MP_MIN_DETECTION_CONFIDENCE: float = 0.5
MP_MIN_TRACKING_CONFIDENCE: float = 0.5
MP_MODEL_COMPLEXITY: int = 1

# Numerical stability epsilon for coordinate normalization
EPSILON: float = 1e-6

# ==============================================================================
# 3. DEEP LEARNING MODEL & TRAINING HYPERPARAMETERS
# ==============================================================================
RANDOM_SEED: int = 42

INPUT_PROJECTION_DIM: int = 128
LSTM_HIDDEN_DIM: int = 128
LSTM_NUM_LAYERS: int = 2
LSTM_DROPOUT: float = 0.3
FC_DROPOUT: float = 0.4

BATCH_SIZE: int = 32
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
NUM_EPOCHS: int = 80
EARLY_STOPPING_PATIENCE: int = 15

# ==============================================================================
# 4. REAL-TIME WEBCAM INFERENCE PARAMETERS
# ==============================================================================
CONFIDENCE_THRESHOLD: float = 0.55
SMOOTHING_WINDOW_SIZE: int = 6
MIN_HAND_PRESENCE_RATIO: float = 0.4
WEBCAM_INDEX: int = 0
WEBCAM_WIDTH: int = 640
WEBCAM_HEIGHT: int = 480
WEBCAM_FPS: int = 30