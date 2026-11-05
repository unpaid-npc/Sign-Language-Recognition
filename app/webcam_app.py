"""
Real-Time Sign Language Recognition OpenCV Application.
Captures live webcam video (or a demonstration video file), extracts MediaPipe Holistic
landmarks on the unmirrored frame (preserving true Left/Right handedness), buffers 30-frame
sequences, runs the trained BiLSTM model, and renders a Heads-Up Display (HUD) with an
optional Picture-in-Picture (PiP) WLASL Reference Sign player.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# Ensure project root is on sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BEST_MODEL_PATH,
    METADATA_CSV_PATH,
    OUTPUTS_DIR,
    RAW_VIDEOS_DIR,
    WEBCAM_HEIGHT,
    WEBCAM_INDEX,
    WEBCAM_WIDTH,
)
from src.inference.predictor import (
    STATUS_BUFFERING,
    STATUS_RECOGNIZED,
    STATUS_UNCERTAIN,
    STATUS_WAITING_HANDS,
    PredictionResult,
    RealTimeSignPredictor,
)
from src.landmarks.extraction import HolisticLandmarkExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReferenceSignPlayer:
    """
    Loads one reference .mp4 clip per top-performing gloss from data/raw/videos/
    so the user can view a Picture-in-Picture (PiP) ASL demonstration while signing.
    """

    def __init__(
        self,
        metadata_csv: Path = METADATA_CSV_PATH,
        per_class_csv: Path = OUTPUTS_DIR / "per_class_metrics.csv",
        videos_dir: Path = RAW_VIDEOS_DIR,
    ) -> None:
        self.samples: List[Tuple[str, Path]] = []
        self.current_idx: int = 0
        self.cap: Optional[cv2.VideoCapture] = None

        if not metadata_csv.exists() or not videos_dir.exists():
            return

        meta_df = pd.read_csv(metadata_csv, dtype={"video_id": str})

        # Prioritize glosses with highest test F1 score so user practices well-learned signs
        preferred_glosses: List[str] = []
        if per_class_csv.exists():
            metrics_df = pd.read_csv(per_class_csv)
            top_df = metrics_df.sort_values(
                by=["f1_score", "test_support"], ascending=False
            )
            preferred_glosses = [str(g) for g in top_df["gloss"].tolist()[:25]]

        if not preferred_glosses:
            preferred_glosses = meta_df["gloss"].drop_duplicates().tolist()[:20]

        for gloss in preferred_glosses:
            sub = meta_df[meta_df["gloss"] == gloss]
            for _, row in sub.iterrows():
                vid = str(row["video_id"]).strip()
                if vid.isdigit() and len(vid) < 5:
                    vid = f"{int(vid):05d}"
                mp4_path = videos_dir / f"{vid}.mp4"
                if mp4_path.exists():
                    self.samples.append((gloss.upper(), mp4_path))
                    break

        if self.samples:
            self._open_current()

    def _open_current(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if not self.samples:
            self.cap = None
            return
        _, path = self.samples[self.current_idx]
        self.cap = cv2.VideoCapture(str(path))

    def next_sign(self) -> str:
        if not self.samples:
            return "N/A"
        self.current_idx = (self.current_idx + 1) % len(self.samples)
        self._open_current()
        return self.samples[self.current_idx][0]

    def prev_sign(self) -> str:
        if not self.samples:
            return "N/A"
        self.current_idx = (self.current_idx - 1) % len(self.samples)
        self._open_current()
        return self.samples[self.current_idx][0]

    def read_pip_frame(self, pip_w: int = 180, pip_h: int = 135) -> Tuple[Optional[np.ndarray], str]:
        if not self.samples or self.cap is None:
            return None, ""
        gloss, _ = self.samples[self.current_idx]
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, gloss
        resized = cv2.resize(frame, (pip_w, pip_h))
        return resized, gloss

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


def open_video_source(
    camera_index: int = WEBCAM_INDEX,
    video_path: Optional[Path] = None,
    width: int = WEBCAM_WIDTH,
    height: int = WEBCAM_HEIGHT,
) -> cv2.VideoCapture:
    """
    Initialize and validate a webcam or video file capture stream with graceful error handling.
    """
    if video_path is not None:
        if not video_path.exists():
            raise FileNotFoundError(f"Demo video file not found: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video file: {video_path}")
        return cap

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Unable to access webcam at index {camera_index}. "
            "Verify that a camera is connected and not locked by another application."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def render_hud_overlay(
    frame: np.ndarray,
    result: PredictionResult,
    fps: float,
    threshold: float,
    show_landmarks: bool,
    pip_frame: Optional[np.ndarray] = None,
    pip_gloss: str = "",
    shoulders_visible: bool = True,
) -> np.ndarray:
    """
    Render a clean, readable Heads-Up Display (HUD) on the video frame.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    if result.status == STATUS_RECOGNIZED:
        accent_color = (80, 210, 60)      # Bright Green
        status_label = "RECOGNIZED"
    elif result.status == STATUS_UNCERTAIN:
        accent_color = (0, 175, 255)      # Amber / Orange
        status_label = "LOW CONFIDENCE"
    elif result.status == STATUS_BUFFERING:
        accent_color = (230, 180, 50)     # Cyan-Blue
        status_label = "BUFFERING"
    else:
        accent_color = (150, 150, 150)    # Neutral Gray
        status_label = "STANDBY"

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 95), (22, 24, 28), -1)
    cv2.rectangle(overlay, (0, h - 38), (w, h), (22, 24, 28), -1)

    if result.top_k_predictions:
        cv2.rectangle(overlay, (w - 235, 105), (w - 10, 240), (22, 24, 28), -1)

    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

    # Status Badge
    cv2.rectangle(canvas, (15, 14), (175, 40), accent_color, -1)
    cv2.putText(
        canvas,
        status_label,
        (23, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (15, 15, 15),
        2,
        cv2.LINE_AA,
    )

    # Primary Recognized Sign Text
    cv2.putText(
        canvas,
        result.gloss,
        (190, 42),
        cv2.FONT_HERSHEY_DUPLEX,
        1.05,
        accent_color if result.is_confident else (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    # Confidence Bar
    bar_x, bar_y, bar_w, bar_h = 15, 58, 340, 20
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (65, 65, 70), -1)
    fill_w = int(bar_w * np.clip(result.confidence, 0.0, 1.0))
    if fill_w > 0:
        cv2.rectangle(
            canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), accent_color, -1
        )
    thresh_x = bar_x + int(bar_w * np.clip(threshold, 0.0, 1.0))
    cv2.line(
        canvas, (thresh_x, bar_y - 3), (thresh_x, bar_y + bar_h + 3), (255, 255, 255), 2
    )
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)

    conf_text = f"Confidence: {result.confidence * 100:5.1f}% (Thresh: {threshold * 100:.0f}%)"
    cv2.putText(
        canvas,
        conf_text,
        (bar_x + bar_w + 15, bar_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Top-3 Candidates Leaderboard
    if result.top_k_predictions:
        cv2.putText(
            canvas,
            "TOP-3 CANDIDATES",
            (w - 222, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 200, 255),
            1,
            cv2.LINE_AA,
        )
        for rank, (cand_gloss, cand_prob) in enumerate(result.top_k_predictions[:3], start=1):
            y_pos = 132 + rank * 30
            color = accent_color if (rank == 1 and result.is_confident) else (215, 215, 215)
            line_str = f"{rank}. {cand_gloss.upper():<12} {cand_prob * 100:4.1f}%"
            cv2.putText(
                canvas,
                line_str,
                (w - 222, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                1,
                cv2.LINE_AA,
            )

    # Framing tip if user is sitting too close and shoulders are cut off
    if not shoulders_visible:
        cv2.putText(
            canvas,
            "Framing Tip: Step back slightly so your shoulders are visible",
            (15, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )

    # Picture-in-Picture (PiP) Reference Sign Video (Bottom-Left above footer)
    if pip_frame is not None:
        ph, pw = pip_frame.shape[:2]
        px, py = 12, h - 48 - ph
        canvas[py : py + ph, px : px + pw] = pip_frame
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (255, 200, 60), 2)
        cv2.rectangle(canvas, (px, py - 24), (px + pw, py), (22, 24, 28), -1)
        cv2.putText(
            canvas,
            f"REF: {pip_gloss} ([N]ext)",
            (px + 6, py - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 215, 80),
            1,
            cv2.LINE_AA,
        )

    # Bottom Status Bar
    footer_text = (
        f"Buf:{result.buffer_length:02d}/{result.sequence_length} | "
        f"FPS:{fps:4.1f} | "
        f"[P]RefVideo [N]extSign [R]eset [L]andmarks [+/-]Thresh [S]hot [Q]uit"
    )
    cv2.putText(
        canvas,
        footer_text,
        (12, h - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    return canvas


def check_shoulders_in_frame(mp_results) -> bool:
    """Return True if both shoulders (landmarks 11 and 12) are comfortably inside the frame."""
    if mp_results is None or not mp_results.pose_landmarks:
        return False
    lms = mp_results.pose_landmarks.landmark
    if len(lms) <= 12:
        return False
    l_sh, r_sh = lms[11], lms[12]
    return (0.05 <= l_sh.y <= 0.93) and (0.05 <= r_sh.y <= 0.93)


def run_webcam_recognition(
    checkpoint_path: Path = BEST_MODEL_PATH,
    camera_index: int = WEBCAM_INDEX,
    video_path: Optional[Path] = None,
    confidence_threshold: float = 0.35,
    loop_video: bool = True,
    max_frames: Optional[int] = None,
) -> None:
    """
    Main real-time recognition loop for live webcam or demo video file.
    """
    predictor = RealTimeSignPredictor.from_checkpoint(
        checkpoint_path=checkpoint_path,
        confidence_threshold=confidence_threshold,
    )

    try:
        cap = open_video_source(camera_index=camera_index, video_path=video_path)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error(f"Video initialization error: {exc}")
        return

    screenshots_dir = OUTPUTS_DIR / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    ref_player = ReferenceSignPlayer() if video_path is None else None
    show_pip = True if (ref_player and ref_player.samples) else False

    show_landmarks = True
    prev_time = time.perf_counter()
    fps = 0.0
    frame_counter = 0

    window_title = "WLASL-100 Real-Time Sign Language Recognition (MediaPipe + BiLSTM)"
    logger.info(
        "Starting real-time recognition. Hotkeys: [P] Toggle Ref Sign | [N] Next Ref Sign | [R] Reset | [Q] Quit"
    )

    with HolisticLandmarkExtractor(static_image_mode=False) as extractor:
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    if video_path is not None and loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        predictor.reset()
                        continue
                    logger.info("End of video stream reached.")
                    break

                # Resize to consistent 640x480 canvas BEFORE MediaPipe processing
                if frame.shape[1] != WEBCAM_WIDTH or frame.shape[0] != WEBCAM_HEIGHT:
                    frame = cv2.resize(frame, (WEBCAM_WIDTH, WEBCAM_HEIGHT))

                # CRITICAL FIX: Extract landmarks on the UNFLIPPED camera frame
                # so the user's physical Right Hand matches WLASL's Right Hand!
                feat_vec, hands_detected, mp_results = extractor.process_bgr_frame(frame)
                shoulders_ok = check_shoulders_in_frame(mp_results)

                # Update sliding sequence buffer & run inference
                pred_result = predictor.update(feat_vec, hands_detected=hands_detected)

                # Draw MediaPipe landmarks on frame
                display_frame = (
                    extractor.draw_landmarks_on_frame(frame, mp_results)
                    if show_landmarks
                    else frame.copy()
                )

                # Now flip display_frame horizontally for live webcam so the user sees
                # a natural selfie mirror view WITHOUT corrupting Left/Right hand coordinates!
                if video_path is None:
                    display_frame = cv2.flip(display_frame, 1)

                # Read Picture-in-Picture reference sign frame if enabled
                pip_img, pip_gloss = (None, "")
                if show_pip and ref_player is not None:
                    pip_img, pip_gloss = ref_player.read_pip_frame()

                # Update FPS
                now = time.perf_counter()
                inst_fps = 1.0 / max(1e-6, (now - prev_time))
                fps = 0.85 * fps + 0.15 * inst_fps if fps > 0 else inst_fps
                prev_time = now

                # Render HUD overlay
                hud_frame = render_hud_overlay(
                    display_frame,
                    result=pred_result,
                    fps=fps,
                    threshold=predictor.confidence_threshold,
                    show_landmarks=show_landmarks,
                    pip_frame=pip_img,
                    pip_gloss=pip_gloss,
                    shoulders_visible=shoulders_ok if video_path is None else True,
                )

                cv2.imshow(window_title, hud_frame)
                frame_counter += 1

                if max_frames is not None and frame_counter >= max_frames:
                    logger.info(f"Reached max_frames={max_frames}; closing stream.")
                    break

                key = cv2.waitKey(1 if video_path is None else 25) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    logger.info("User requested exit.")
                    break
                elif key in (ord("r"), ord("R")):
                    predictor.reset()
                    logger.info("Sequence buffer reset by user.")
                elif key in (ord("l"), ord("L")):
                    show_landmarks = not show_landmarks
                elif key in (ord("p"), ord("P")):
                    show_pip = not show_pip
                elif key in (ord("n"), ord("N")) and ref_player is not None:
                    new_sign = ref_player.next_sign()
                    predictor.reset()
                    logger.info(f"Switched reference sign to: {new_sign}")
                elif key in (ord("b"), ord("B")) and ref_player is not None:
                    new_sign = ref_player.prev_sign()
                    predictor.reset()
                    logger.info(f"Switched reference sign to: {new_sign}")
                elif key in (ord("+"), ord("=")):
                    predictor.confidence_threshold = min(
                        0.95, round(predictor.confidence_threshold + 0.05, 2)
                    )
                elif key in (ord("-"), ord("_")):
                    predictor.confidence_threshold = max(
                        0.05, round(predictor.confidence_threshold - 0.05, 2)
                    )
                elif key in (ord("s"), ord("S")):
                    ts = int(time.time())
                    shot_path = screenshots_dir / f"sign_demo_{ts}.png"
                    cv2.imwrite(str(shot_path), hud_frame)
                    logger.info(f"Saved HUD screenshot to {shot_path}")

        finally:
            if ref_player is not None:
                ref_player.close()
            cap.release()
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Real-Time Sign Language Recognition on Webcam or Demo Video."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=WEBCAM_INDEX,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional path to a .mp4 video file to test the real-time HUD.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Confidence threshold for sign recognition (default: 0.35).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for automated testing.",
    )
    args = parser.parse_args()

    run_webcam_recognition(
        camera_index=args.camera,
        video_path=args.video,
        confidence_threshold=args.threshold,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()