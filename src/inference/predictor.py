"""
Real-Time Sign Language Inference Engine.
Manages the sliding sequence buffer, hand-presence gating, PyTorch BiLSTM forward pass,
probability smoothing (moving average + majority stability), and unknown sign rejection.
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    BEST_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    FEATURE_DIM,
    MIN_HAND_PRESENCE_RATIO,
    SEQUENCE_LENGTH,
    SMOOTHING_WINDOW_SIZE,
)
from src.models.lstm import SignBiLSTMClassifier
from src.models.train import load_checkpoint
from src.utils.logger import get_logger

logger = get_logger(__name__)

STATUS_WAITING_HANDS = "WAITING FOR HANDS"
STATUS_BUFFERING = "BUFFERING SEQUENCE"
STATUS_UNCERTAIN = "UNCERTAIN / UNKNOWN"
STATUS_RECOGNIZED = "RECOGNIZED"


@dataclass
class PredictionResult:
    """Structured output returned on every frame by RealTimeSignPredictor."""

    status: str
    gloss: str
    confidence: float
    is_confident: bool
    buffer_length: int
    sequence_length: int
    hand_presence_ratio: float
    top_k_predictions: List[Tuple[str, float]] = field(default_factory=list)


class RealTimeSignPredictor:
    """
    Stateful sliding-window predictor that converts a live stream of (144,) normalized
    landmark vectors into smoothed, thresholded sign predictions.
    """

    def __init__(
        self,
        model: SignBiLSTMClassifier,
        idx_to_gloss: Dict[int, str],
        device: Optional[torch.device] = None,
        sequence_length: int = SEQUENCE_LENGTH,
        confidence_threshold: float = 0.45,
        smoothing_window: int = SMOOTHING_WINDOW_SIZE,
        min_hand_presence_ratio: float = MIN_HAND_PRESENCE_RATIO,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.model.eval()

        self.idx_to_gloss = idx_to_gloss
        self.num_classes = len(idx_to_gloss)
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = max(1, smoothing_window)
        self.min_hand_presence_ratio = min_hand_presence_ratio

        # Sliding window buffers
        self.landmark_buffer: Deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self.hand_presence_buffer: Deque[bool] = deque(maxlen=self.sequence_length)
        self.prob_history: Deque[np.ndarray] = deque(maxlen=self.smoothing_window)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path = BEST_MODEL_PATH,
        confidence_threshold: float = 0.45,
        smoothing_window: int = SMOOTHING_WINDOW_SIZE,
        min_hand_presence_ratio: float = MIN_HAND_PRESENCE_RATIO,
    ) -> "RealTimeSignPredictor":
        """Load trained model checkpoint and initialize RealTimeSignPredictor."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, idx_to_gloss, ckpt = load_checkpoint(checkpoint_path, device=device)
        logger.info(
            f"Loaded RealTimeSignPredictor from {checkpoint_path.name} "
            f"(Epoch {ckpt.get('epoch')}, {len(idx_to_gloss)} classes) on {device}"
        )
        return cls(
            model=model,
            idx_to_gloss=idx_to_gloss,
            device=device,
            confidence_threshold=confidence_threshold,
            smoothing_window=smoothing_window,
            min_hand_presence_ratio=min_hand_presence_ratio,
        )

    def reset(self) -> None:
        """Clear all temporal buffers."""
        self.landmark_buffer.clear()
        self.hand_presence_buffer.clear()
        self.prob_history.clear()

    def update(
        self,
        feature_vector: np.ndarray,
        hands_detected: bool,
        top_k: int = 3,
    ) -> PredictionResult:
        """
        Append a new frame's normalized landmark vector to the sliding buffer and
        generate a smoothed prediction if and only if the sequence is complete and valid.

        Args:
            feature_vector: Normalized landmark vector of shape (144,).
            hands_detected: True if MediaPipe detected at least one hand in this frame.
            top_k: Number of top candidate glosses to return.

        Returns:
            PredictionResult instance.
        """
        if feature_vector is None or feature_vector.shape != (FEATURE_DIM,):
            raise ValueError(
                f"Expected feature_vector of shape ({FEATURE_DIM},), "
                f"got {None if feature_vector is None else feature_vector.shape}"
            )

        self.landmark_buffer.append(feature_vector.astype(np.float32))
        self.hand_presence_buffer.append(bool(hands_detected))

        current_len = len(self.landmark_buffer)
        hand_ratio = float(np.mean(self.hand_presence_buffer)) if current_len > 0 else 0.0

        # Check recent frames (last 12 frames) for active hand presence
        recent_hands = list(self.hand_presence_buffer)[-12:]
        recent_hand_ratio = float(np.mean(recent_hands)) if recent_hands else 0.0

        # Gate 1: If user's hands are down / out of frame, clear stale probabilities
        if recent_hand_ratio < self.min_hand_presence_ratio:
            self.prob_history.clear()
            return PredictionResult(
                status=STATUS_WAITING_HANDS,
                gloss="Waiting for hands...",
                confidence=0.0,
                is_confident=False,
                buffer_length=current_len,
                sequence_length=self.sequence_length,
                hand_presence_ratio=hand_ratio,
                top_k_predictions=[],
            )

        # Gate 2: Do NOT generate predictions from incomplete sequences (< 30 frames)
        if current_len < self.sequence_length:
            return PredictionResult(
                status=STATUS_BUFFERING,
                gloss=f"Buffering ({current_len}/{self.sequence_length})...",
                confidence=0.0,
                is_confident=False,
                buffer_length=current_len,
                sequence_length=self.sequence_length,
                hand_presence_ratio=hand_ratio,
                top_k_predictions=[],
            )

        # Gate 3: Check overall buffer hand presence ratio
        if hand_ratio < self.min_hand_presence_ratio:
            self.prob_history.clear()
            return PredictionResult(
                status=STATUS_WAITING_HANDS,
                gloss="Waiting for hands...",
                confidence=0.0,
                is_confident=False,
                buffer_length=current_len,
                sequence_length=self.sequence_length,
                hand_presence_ratio=hand_ratio,
                top_k_predictions=[],
            )

        # Model Forward Pass on complete (1, 30, 144) sequence tensor
        seq_np = np.stack(list(self.landmark_buffer), axis=0)  # (30, 144)
        seq_tensor = torch.from_numpy(seq_np).unsqueeze(0).to(self.device)  # (1, 30, 144)

        with torch.no_grad():
            logits = self.model(seq_tensor)  # (1, num_classes)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (num_classes,)

        # Temporal Probability Smoothing (Exponentially weighted moving average over window)
        self.prob_history.append(probs)
        weights = np.linspace(0.5, 1.0, num=len(self.prob_history), dtype=np.float32)
        weights /= np.sum(weights)
        smoothed_probs = np.average(
            np.stack(list(self.prob_history), axis=0), axis=0, weights=weights
        )

        # Extract Top-K predictions
        k = min(top_k, self.num_classes)
        top_indices = np.argsort(smoothed_probs)[::-1][:k]
        top_k_list: List[Tuple[str, float]] = [
            (self.idx_to_gloss.get(int(idx), f"class_{idx}"), float(smoothed_probs[idx]))
            for idx in top_indices
        ]

        best_gloss, best_conf = top_k_list[0]

        # Gate 4: Confidence Thresholding for Uncertain / Unknown Handling
        if best_conf >= self.confidence_threshold:
            return PredictionResult(
                status=STATUS_RECOGNIZED,
                gloss=best_gloss.upper(),
                confidence=best_conf,
                is_confident=True,
                buffer_length=current_len,
                sequence_length=self.sequence_length,
                hand_presence_ratio=hand_ratio,
                top_k_predictions=top_k_list,
            )
        else:
            return PredictionResult(
                status=STATUS_UNCERTAIN,
                gloss="Uncertain / Unknown",
                confidence=best_conf,
                is_confident=False,
                buffer_length=current_len,
                sequence_length=self.sequence_length,
                hand_presence_ratio=hand_ratio,
                top_k_predictions=top_k_list,
            )