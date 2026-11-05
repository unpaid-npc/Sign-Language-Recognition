"""
MediaPipe Holistic Landmark Extraction and Spatial-Temporal Normalization Module.
Extracts 144 features per frame:
  - 6 Upper-Body Pose Landmarks x 3 (x, y, z) = 18 features
  - 21 Left-Hand Landmarks x 3 (x, y, z)      = 63 features
  - 21 Right-Hand Landmarks x 3 (x, y, z)     = 63 features
Total = 144 float32 features per frame, normalized relative to shoulder center & width.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src.config import (
    COORDS_PER_LANDMARK,
    EPSILON,
    FEATURE_DIM,
    HAND_FEATURE_DIM,
    MIN_VIDEO_FRAMES,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    MP_MODEL_COMPLEXITY,
    NUM_HAND_LANDMARKS,
    NUM_POSE_LANDMARKS,
    POSE_FEATURE_DIM,
    SELECTED_POSE_INDICES,
    SEQUENCE_LENGTH,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def extract_raw_landmarks_from_results(
    results: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Extract raw (unnormalized) pose, left-hand, and right-hand arrays from MediaPipe results.

    Args:
        results: Output object from `holistic.process(rgb_frame)`.

    Returns:
        Tuple of:
          - pose_coords: shape (6, 3) float32
          - left_hand_coords: shape (21, 3) float32
          - right_hand_coords: shape (21, 3) float32
          - hands_detected: bool (True if at least one hand is present in the frame)
    """
    # 1. Pose Landmarks (6 selected upper-body indices)
    pose_coords = np.zeros((NUM_POSE_LANDMARKS, COORDS_PER_LANDMARK), dtype=np.float32)
    if results.pose_landmarks and results.pose_landmarks.landmark:
        landmarks = results.pose_landmarks.landmark
        for i, idx in enumerate(SELECTED_POSE_INDICES):
            if idx < len(landmarks):
                lm = landmarks[idx]
                pose_coords[i] = [lm.x, lm.y, lm.z]

    # 2. Left Hand Landmarks (21 points)
    left_hand_coords = np.zeros((NUM_HAND_LANDMARKS, COORDS_PER_LANDMARK), dtype=np.float32)
    has_left = False
    if results.left_hand_landmarks and results.left_hand_landmarks.landmark:
        has_left = True
        for i, lm in enumerate(results.left_hand_landmarks.landmark[:NUM_HAND_LANDMARKS]):
            left_hand_coords[i] = [lm.x, lm.y, lm.z]

    # 3. Right Hand Landmarks (21 points)
    right_hand_coords = np.zeros((NUM_HAND_LANDMARKS, COORDS_PER_LANDMARK), dtype=np.float32)
    has_right = False
    if results.right_hand_landmarks and results.right_hand_landmarks.landmark:
        has_right = True
        for i, lm in enumerate(results.right_hand_landmarks.landmark[:NUM_HAND_LANDMARKS]):
            right_hand_coords[i] = [lm.x, lm.y, lm.z]

    hands_detected = has_left or has_right
    return pose_coords, left_hand_coords, right_hand_coords, hands_detected


def normalize_frame_landmarks(
    pose_coords: np.ndarray,
    left_hand_coords: np.ndarray,
    right_hand_coords: np.ndarray,
) -> np.ndarray:
    """
    Normalize 3D landmark coordinates relative to the signer's mid-shoulder center
    and scale by the 2D Euclidean distance between left and right shoulders.
    Missing hands (all zeros) remain Strictly Zero after normalization.

    Args:
        pose_coords: shape (6, 3) where index 2 is Left Shoulder (11)
                     and index 3 is Right Shoulder (12) per SELECTED_POSE_INDICES.
        left_hand_coords: shape (21, 3)
        right_hand_coords: shape (21, 3)

    Returns:
        Flattened 1D numpy array of shape (144,) with dtype float32.
    """
    pose_norm = pose_coords.copy().astype(np.float32)
    lh_norm = left_hand_coords.copy().astype(np.float32)
    rh_norm = right_hand_coords.copy().astype(np.float32)

    pose_present = not np.allclose(pose_norm, 0.0)
    lh_present = not np.allclose(lh_norm, 0.0)
    rh_present = not np.allclose(rh_norm, 0.0)

    # In SELECTED_POSE_INDICES = [0, 9, 11, 12, 13, 14]:
    # index 2 -> Landmark 11 (Left Shoulder)
    # index 3 -> Landmark 12 (Right Shoulder)
    left_shoulder = pose_norm[2]
    right_shoulder = pose_norm[3]

    if pose_present and not (
        np.allclose(left_shoulder, 0.0) or np.allclose(right_shoulder, 0.0)
    ):
        center = 0.5 * (left_shoulder + right_shoulder)
        shoulder_dist = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
        scale = shoulder_dist if shoulder_dist > 0.02 else 1.0
    else:
        # Fallback if shoulders are occluded: center at image midpoint (0.5, 0.5, 0.0)
        center = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        scale = 1.0

    inv_scale = 1.0 / (scale + EPSILON)

    if pose_present:
        pose_norm = (pose_norm - center) * inv_scale
    if lh_present:
        lh_norm = (lh_norm - center) * inv_scale
    if rh_present:
        rh_norm = (rh_norm - center) * inv_scale

    feature_vector = np.concatenate(
        [pose_norm.flatten(), lh_norm.flatten(), rh_norm.flatten()], axis=0
    ).astype(np.float32)

    assert feature_vector.shape == (FEATURE_DIM,), (
        f"Expected feature vector shape ({FEATURE_DIM},), got {feature_vector.shape}"
    )
    return feature_vector


def resample_or_pad_sequence(
    sequence: np.ndarray, target_length: int = SEQUENCE_LENGTH
) -> np.ndarray:
    """
    Convert a variable-length landmark sequence of shape (N, 144) into a fixed-length
    sequence of shape (target_length, 144).

    - If N == 0: returns zeros of shape (target_length, 144).
    - If N == target_length: returns sequence as-is.
    - If N > target_length: uniformly samples `target_length` frames across [0, N-1].
    - If N < target_length: linearly interpolates / resamples along the temporal axis
      so motion speed scales smoothly across the window.

    Args:
        sequence: Array of shape (N, FEATURE_DIM).
        target_length: Target sequence length T (default 30).

    Returns:
        Array of shape (target_length, FEATURE_DIM) with dtype float32.
    """
    if sequence is None or len(sequence) == 0:
        return np.zeros((target_length, FEATURE_DIM), dtype=np.float32)

    num_frames = sequence.shape[0]
    if num_frames == target_length:
        return sequence.astype(np.float32)

    # Uniform index sampling across [0, num_frames - 1]
    indices = np.linspace(0, num_frames - 1, num=target_length)
    rounded_indices = np.round(indices).astype(int)
    rounded_indices = np.clip(rounded_indices, 0, num_frames - 1)
    return sequence[rounded_indices].astype(np.float32)


class HolisticLandmarkExtractor:
    """
    Stateful/Context-managed wrapper around MediaPipe Holistic for both
    offline video preprocessing and live webcam frame extraction.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = MP_MODEL_COMPLEXITY,
        min_detection_confidence: float = MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MP_MIN_TRACKING_CONFIDENCE,
    ) -> None:
        self.holistic = mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=not static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_bgr_frame(
        self, bgr_frame: np.ndarray
    ) -> Tuple[np.ndarray, bool, Any]:
        """
        Process a single OpenCV BGR frame and return its normalized 144-D feature vector.

        Args:
            bgr_frame: HxWx3 uint8 BGR image from OpenCV.

        Returns:
            Tuple of:
              - feature_vector: (144,) float32 numpy array
              - hands_detected: bool indicating if at least one hand was detected
              - raw_results: MediaPipe Holistic results object (for optional visualization)
        """
        if bgr_frame is None or bgr_frame.size == 0:
            raise ValueError("Received empty or None BGR frame.")

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.holistic.process(rgb_frame)
        rgb_frame.flags.writeable = True

        pose_c, lh_c, rh_c, hands_detected = extract_raw_landmarks_from_results(results)
        feature_vec = normalize_frame_landmarks(pose_c, lh_c, rh_c)
        return feature_vec, hands_detected, results

    def extract_video_sequence(
        self,
        video_path: Path,
        frame_start: int = 1,
        frame_end: int = -1,
        target_length: int = SEQUENCE_LENGTH,
    ) -> Optional[Dict[str, Any]]:
        """
        Read a video file, extract frame-by-frame normalized landmarks within
        [frame_start, frame_end], and resample to (target_length, 144).

        Args:
            video_path: Path to the .mp4 video file.
            frame_start: 1-based start frame from WLASL annotation.
            frame_end: 1-based end frame from WLASL annotation (-1 means end of video).
            target_length: Fixed output sequence length (default 30).

        Returns:
            Dictionary containing:
              - 'sequence': np.ndarray of shape (target_length, 144)
              - 'num_raw_frames': int
              - 'hand_detection_rate': float in [0.0, 1.0]
            Or None if the video cannot be opened or has fewer than MIN_VIDEO_FRAMES.
        """
        if not video_path.exists():
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_idx = max(0, frame_start - 1) if frame_start > 1 else 0
        end_idx = (
            min(total_frames - 1, frame_end - 1)
            if (frame_end > 0 and total_frames > 0)
            else (total_frames - 1 if total_frames > 0 else 999999)
        )

        # If preprocessed video was already trimmed (common in Kaggle WLASL mirror),
        # start_idx might exceed total_frames. Fallback to full clip.
        if total_frames > 0 and (start_idx >= total_frames or (end_idx - start_idx) < MIN_VIDEO_FRAMES):
            start_idx = 0
            end_idx = total_frames - 1

        frame_features: List[np.ndarray] = []
        hands_present_count = 0
        current_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if current_idx >= start_idx and current_idx <= end_idx:
                feat_vec, has_hands, _ = self.process_bgr_frame(frame)
                frame_features.append(feat_vec)
                if has_hands:
                    hands_present_count += 1

            current_idx += 1
            if current_idx > end_idx:
                break

        cap.release()

        num_extracted = len(frame_features)
        if num_extracted < MIN_VIDEO_FRAMES:
            return None

        raw_sequence = np.stack(frame_features, axis=0)  # (N, 144)
        fixed_sequence = resample_or_pad_sequence(raw_sequence, target_length=target_length)
        hand_rate = float(hands_present_count) / float(num_extracted)

        return {
            "sequence": fixed_sequence,
            "num_raw_frames": num_extracted,
            "hand_detection_rate": round(hand_rate, 4),
        }

    @staticmethod
    def draw_landmarks_on_frame(bgr_frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw Pose, Left-Hand, and Right-Hand landmarks on a BGR frame for visualization.
        """
        annotated = bgr_frame.copy()
        if results is None:
            return annotated

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
        return annotated

    def close(self) -> None:
        if hasattr(self, "holistic") and self.holistic is not None:
            self.holistic.close()

    def __enter__(self) -> "HolisticLandmarkExtractor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()