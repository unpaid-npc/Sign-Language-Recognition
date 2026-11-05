"""
Unit tests for Day 1: MediaPipe landmark extraction, spatial normalization,
sequence resampling, and split leakage prevention.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    COORDS_PER_LANDMARK,
    FEATURE_DIM,
    NUM_HAND_LANDMARKS,
    NUM_POSE_LANDMARKS,
    SEQUENCE_LENGTH,
)
from src.data.preprocessing import verify_and_rebalance_splits
from src.landmarks.extraction import (
    HolisticLandmarkExtractor,
    normalize_frame_landmarks,
    resample_or_pad_sequence,
)


def test_normalize_frame_landmarks_dimensions_and_missing_hands() -> None:
    """Verify output dimension is (144,) and missing hands remain strictly 0.0."""
    pose = np.zeros((NUM_POSE_LANDMARKS, COORDS_PER_LANDMARK), dtype=np.float32)
    # Set Left Shoulder (index 2) and Right Shoulder (index 3)
    pose[2] = [0.6, 0.5, 0.0]
    pose[3] = [0.4, 0.5, 0.0]

    # Left hand present, Right hand missing (all zeros)
    left_hand = np.ones((NUM_HAND_LANDMARKS, COORDS_PER_LANDMARK), dtype=np.float32) * 0.5
    right_hand = np.zeros((NUM_HAND_LANDMARKS, COORDS_PER_LANDMARK), dtype=np.float32)

    feat = normalize_frame_landmarks(pose, left_hand, right_hand)
    assert feat.shape == (FEATURE_DIM,)
    assert feat.dtype == np.float32

    # Mid-shoulder is (0.5, 0.5, 0.0). Since left_hand is all (0.5, 0.5, 0.5),
    # x and y relative to mid-shoulder must be 0.0!
    lh_extracted = feat[18 : 18 + 63].reshape(21, 3)
    assert np.allclose(lh_extracted[:, 0], 0.0, atol=1e-5)
    assert np.allclose(lh_extracted[:, 1], 0.0, atol=1e-5)

    # Right hand was missing (all zeros) -> must remain strictly 0.0
    rh_extracted = feat[18 + 63 :]
    assert np.allclose(rh_extracted, 0.0)


def test_spatial_translation_and_scale_invariance() -> None:
    """
    Verify that shifting or scaling a signer in the frame produces identical
    normalized landmark coordinates.
    """
    pose_1 = np.array(
        [
            [0.50, 0.30, 0.0],  # Nose
            [0.50, 0.35, 0.0],  # Mouth
            [0.60, 0.50, 0.0],  # L Shoulder
            [0.40, 0.50, 0.0],  # R Shoulder
            [0.65, 0.70, 0.0],  # L Elbow
            [0.35, 0.70, 0.0],  # R Elbow
        ],
        dtype=np.float32,
    )
    lh_1 = np.full((21, 3), [0.62, 0.42, -0.05], dtype=np.float32)
    rh_1 = np.zeros((21, 3), dtype=np.float32)

    # Translate by (+0.1, +0.1, 0.0) and scale by 1.5x
    pose_2 = pose_1 * 1.5 + np.array([0.1, 0.1, 0.0], dtype=np.float32)
    lh_2 = lh_1 * 1.5 + np.array([0.1, 0.1, 0.0], dtype=np.float32)
    rh_2 = np.zeros((21, 3), dtype=np.float32)

    norm_1 = normalize_frame_landmarks(pose_1, lh_1, rh_1)
    norm_2 = normalize_frame_landmarks(pose_2, lh_2, rh_2)

    assert np.allclose(norm_1, norm_2, atol=1e-4)


@pytest.mark.parametrize("raw_len", [0, 7, 15, 30, 64])
def test_resample_or_pad_sequence_shapes(raw_len: int) -> None:
    """Verify temporal resampling always outputs (SEQUENCE_LENGTH, FEATURE_DIM)."""
    if raw_len == 0:
        raw_seq = np.zeros((0, FEATURE_DIM), dtype=np.float32)
    else:
        raw_seq = np.random.randn(raw_len, FEATURE_DIM).astype(np.float32)

    out_seq = resample_or_pad_sequence(raw_seq, target_length=SEQUENCE_LENGTH)
    assert out_seq.shape == (SEQUENCE_LENGTH, FEATURE_DIM)
    assert out_seq.dtype == np.float32


def test_holistic_extractor_on_synthetic_frame() -> None:
    """Verify HolisticLandmarkExtractor runs cleanly on a blank BGR frame."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with HolisticLandmarkExtractor(static_image_mode=True) as extractor:
        feat, has_hands, _ = extractor.process_bgr_frame(dummy_frame)
        assert feat.shape == (FEATURE_DIM,)
        assert feat.dtype == np.float32
        assert has_hands is False


def test_verify_and_rebalance_splits_no_leakage() -> None:
    """Verify split rebalancing creates missing val/test splits without leaking video_ids."""
    mock_df = pd.DataFrame(
        [
            {"video_id": "00001", "label": 0, "split": "train"},
            {"video_id": "00002", "label": 0, "split": "train"},
            {"video_id": "00003", "label": 0, "split": "train"},
            {"video_id": "00004", "label": 0, "split": "train"},
        ]
    )
    balanced_df = verify_and_rebalance_splits(mock_df)
    splits = set(balanced_df["split"].unique())
    assert splits == {"train", "val", "test"}
    assert len(balanced_df["video_id"].unique()) == 4