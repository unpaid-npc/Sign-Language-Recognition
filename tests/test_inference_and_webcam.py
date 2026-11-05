"""
Unit tests for Day 3: Real-time sequence buffering, incomplete sequence blocking,
hand-presence gating, unknown/uncertain prediction handling, HUD overlay rendering,
and graceful webcam error handling.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from app.webcam_app import open_video_source, render_hud_overlay
from src.config import FEATURE_DIM, SEQUENCE_LENGTH
from src.inference.predictor import (
    STATUS_BUFFERING,
    STATUS_RECOGNIZED,
    STATUS_UNCERTAIN,
    STATUS_WAITING_HANDS,
    RealTimeSignPredictor,
)
from src.models.lstm import SignBiLSTMClassifier


def _create_dummy_predictor(threshold: float = 0.50) -> RealTimeSignPredictor:
    model = SignBiLSTMClassifier(num_classes=5)
    idx_to_gloss = {0: "book", 1: "drink", 2: "computer", 3: "help", 4: "yes"}
    return RealTimeSignPredictor(
        model=model,
        idx_to_gloss=idx_to_gloss,
        device=torch.device("cpu"),
        sequence_length=SEQUENCE_LENGTH,
        confidence_threshold=threshold,
        smoothing_window=4,
        min_hand_presence_ratio=0.4,
    )


def test_incomplete_sequences_never_trigger_predictions() -> None:
    """Verify frames 1..29 return STATUS_BUFFERING and never invoke classification."""
    predictor = _create_dummy_predictor()
    dummy_feat = np.ones((FEATURE_DIM,), dtype=np.float32) * 0.1

    for step in range(1, SEQUENCE_LENGTH):
        res = predictor.update(dummy_feat, hands_detected=True)
        assert res.status == STATUS_BUFFERING
        assert res.buffer_length == step
        assert res.is_confident is False
        assert res.top_k_predictions == []


def test_waiting_for_hands_gate_when_no_hands_present() -> None:
    """Verify predictor outputs STATUS_WAITING_HANDS when hands are absent."""
    predictor = _create_dummy_predictor()
    zero_feat = np.zeros((FEATURE_DIM,), dtype=np.float32)

    for _ in range(SEQUENCE_LENGTH):
        res = predictor.update(zero_feat, hands_detected=False)

    assert res.status == STATUS_WAITING_HANDS
    assert res.is_confident is False
    assert res.confidence == 0.0


def test_unknown_prediction_handling_below_threshold() -> None:
    """
    Verify that when max softmax probability is below confidence_threshold,
    the predictor returns STATUS_UNCERTAIN ('Uncertain / Unknown').
    """
    # Set threshold to 0.99 so an untrained uniform 5-class model (~0.20 prob) is always below
    predictor = _create_dummy_predictor(threshold=0.99)
    dummy_feat = np.random.randn(FEATURE_DIM).astype(np.float32)

    res = None
    for _ in range(SEQUENCE_LENGTH):
        res = predictor.update(dummy_feat, hands_detected=True)

    assert res is not None
    assert res.status == STATUS_UNCERTAIN
    assert res.gloss == "Uncertain / Unknown"
    assert res.is_confident is False
    assert len(res.top_k_predictions) == 3


def test_confident_prediction_when_above_threshold() -> None:
    """
    Verify that when confidence exceeds threshold, STATUS_RECOGNIZED and a valid gloss are returned.
    """
    predictor = _create_dummy_predictor(threshold=0.05)
    dummy_feat = np.random.randn(FEATURE_DIM).astype(np.float32)

    res = None
    for _ in range(SEQUENCE_LENGTH):
        res = predictor.update(dummy_feat, hands_detected=True)

    assert res is not None
    assert res.status == STATUS_RECOGNIZED
    assert res.is_confident is True
    assert res.confidence >= 0.05
    assert res.gloss in {"BOOK", "DRINK", "COMPUTER", "HELP", "YES"}


def test_hud_overlay_renders_valid_bgr_image() -> None:
    """Verify render_hud_overlay preserves frame dimensions and dtype."""
    predictor = _create_dummy_predictor(threshold=0.10)
    dummy_feat = np.random.randn(FEATURE_DIM).astype(np.float32)
    for _ in range(SEQUENCE_LENGTH):
        res = predictor.update(dummy_feat, hands_detected=True)

    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hud = render_hud_overlay(
        blank_frame, result=res, fps=30.0, threshold=0.10, show_landmarks=True
    )
    assert hud.shape == (480, 640, 3)
    assert hud.dtype == np.uint8
    assert np.any(hud > 0)


def test_graceful_webcam_and_video_error_handling() -> None:
    """Verify open_video_source raises clear errors on missing files or invalid camera indices."""
    with pytest.raises(FileNotFoundError):
        open_video_source(video_path=Path("non_existent_demo_clip.mp4"))

    with pytest.raises(RuntimeError):
        open_video_source(camera_index=99)