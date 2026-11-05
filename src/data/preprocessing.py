"""
WLASL-100 End-to-End Video-to-Landmark Dataset Preprocessing Pipeline.
Extracts (30, 144) MediaPipe landmark tensors from local .mp4 clips, caches them as .npy
files, verifies zero data leakage across splits, and generates metadata.csv + label_map.json.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    FEATURE_DIM,
    LABEL_MAP_PATH,
    METADATA_CSV_PATH,
    NUM_CLASSES,
    RANDOM_SEED,
    RAW_VIDEOS_DIR,
    SEQUENCE_LENGTH,
    SEQUENCES_DIR,
    WLASL_JSON_PATH,
)
from src.landmarks.extraction import HolisticLandmarkExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_label_map(wlasl_subset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create bidirectional mapping between integer class indices [0..K-1] and gloss strings.
    """
    idx_to_gloss: Dict[int, str] = {}
    gloss_to_idx: Dict[str, int] = {}

    for idx, entry in enumerate(wlasl_subset):
        gloss = str(entry["gloss"]).strip().lower()
        idx_to_gloss[idx] = gloss
        gloss_to_idx[gloss] = idx

    return {
        "num_classes": len(wlasl_subset),
        "idx_to_gloss": {str(k): v for k, v in idx_to_gloss.items()},
        "gloss_to_idx": gloss_to_idx,
    }


def verify_and_rebalance_splits(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Ensure zero video_id leakage across splits and verify that every class with >=3 videos
    has at least 1 validation and 1 test sample (in case dead YouTube links left a class
    with 0 test videos in the raw JSON).
    """
    df = df.drop_duplicates(subset=["video_id"]).copy()
    rng = np.random.default_rng(seed)

    updated_splits = df["split"].tolist()
    df["split"] = updated_splits

    for label_id, group in df.groupby("label"):
        if len(group) < 3:
            continue

        train_indices = group[group["split"] == "train"].index.tolist()
        val_indices = group[group["split"] == "val"].index.tolist()
        test_indices = group[group["split"] == "test"].index.tolist()

        # If test is empty and train has >= 3 samples, move 1 sample from train -> test
        if len(test_indices) == 0 and len(train_indices) >= 3:
            chosen = int(rng.choice(train_indices))
            df.loc[chosen, "split"] = "test"
            train_indices.remove(chosen)

        # If val is empty and train has >= 2 samples, move 1 sample from train -> val
        if len(val_indices) == 0 and len(train_indices) >= 2:
            chosen = int(rng.choice(train_indices))
            df.loc[chosen, "split"] = "val"
            train_indices.remove(chosen)

    # Final leakage assertion: no video_id can appear in more than one split
    train_vids = set(df[df["split"] == "train"]["video_id"])
    val_vids = set(df[df["split"] == "val"]["video_id"])
    test_vids = set(df[df["split"] == "test"]["video_id"])

    assert len(train_vids.intersection(val_vids)) == 0, "Leakage between train and val!"
    assert len(train_vids.intersection(test_vids)) == 0, "Leakage between train and test!"
    assert len(val_vids.intersection(test_vids)) == 0, "Leakage between val and test!"

    return df


def run_preprocessing_pipeline(
    json_path: Path = WLASL_JSON_PATH,
    videos_dir: Path = RAW_VIDEOS_DIR,
    sequences_dir: Path = SEQUENCES_DIR,
    metadata_csv_path: Path = METADATA_CSV_PATH,
    label_map_path: Path = LABEL_MAP_PATH,
    num_classes: int = NUM_CLASSES,
    max_videos: Optional[int] = None,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """
    Process all available WLASL-100 videos through MediaPipe Holistic, save .npy sequences,
    and write metadata.csv and label_map.json.
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"WLASL JSON not found at {json_path}. Run `python -m src.data.download_wlasl` first."
        )

    sequences_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        wlasl_data = json.load(f)[:num_classes]

    # Save Label Map
    label_map = build_label_map(wlasl_data)
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    logger.info(f"Saved label map ({num_classes} classes) to {label_map_path}")

    # Collect candidate video instances that exist on disk
    candidates: List[Dict[str, Any]] = []
    missing_videos_count = 0

    for class_idx, entry in enumerate(wlasl_data):
        gloss = str(entry["gloss"]).strip().lower()
        for inst in entry.get("instances", []):
            video_id = str(inst.get("video_id", "")).strip()
            video_file = videos_dir / f"{video_id}.mp4"
            if not video_file.exists():
                missing_videos_count += 1
                continue

            candidates.append(
                {
                    "video_id": video_id,
                    "gloss": gloss,
                    "label": class_idx,
                    "split": inst.get("split", "train"),
                    "signer_id": int(inst.get("signer_id", -1)),
                    "frame_start": int(inst.get("frame_start", 1)),
                    "frame_end": int(inst.get("frame_end", -1)),
                    "video_path": video_file,
                }
            )

    if max_videos is not None and max_videos > 0:
        candidates = candidates[:max_videos]

    logger.info(
        f"Found {len(candidates)} local videos across {num_classes} classes "
        f"({missing_videos_count} unavailable annotations skipped)."
    )

    records: List[Dict[str, Any]] = []
    corrupted_or_empty_count = 0

    with HolisticLandmarkExtractor(static_image_mode=False) as extractor:
        for item in tqdm(candidates, desc="Extracting MediaPipe Landmarks"):
            video_id = item["video_id"]
            npy_path = sequences_dir / f"{video_id}.npy"

            # Fast resume if .npy already exists and has valid shape
            if skip_existing and npy_path.exists():
                try:
                    existing_arr = np.load(npy_path)
                    if existing_arr.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
                        # Compute hand presence from existing array (last 126 features are hands)
                        hand_feats = existing_arr[:, 18:]
                        non_zero_frames = np.any(np.abs(hand_feats) > 1e-6, axis=1)
                        hand_rate = float(np.mean(non_zero_frames))
                        records.append(
                            {
                                "video_id": video_id,
                                "gloss": item["gloss"],
                                "label": item["label"],
                                "split": item["split"],
                                "signer_id": item["signer_id"],
                                "num_raw_frames": SEQUENCE_LENGTH,
                                "hand_detection_rate": round(hand_rate, 4),
                                "npy_path": str(npy_path.relative_to(sequences_dir.parent.parent)),
                            }
                        )
                        continue
                except Exception:
                    pass

            result = extractor.extract_video_sequence(
                video_path=item["video_path"],
                frame_start=item["frame_start"],
                frame_end=item["frame_end"],
                target_length=SEQUENCE_LENGTH,
            )

            if result is None:
                corrupted_or_empty_count += 1
                continue

            sequence_arr: np.ndarray = result["sequence"]
            np.save(npy_path, sequence_arr)

            records.append(
                {
                    "video_id": video_id,
                    "gloss": item["gloss"],
                    "label": item["label"],
                    "split": item["split"],
                    "signer_id": item["signer_id"],
                    "num_raw_frames": result["num_raw_frames"],
                    "hand_detection_rate": result["hand_detection_rate"],
                    "npy_path": str(npy_path.relative_to(sequences_dir.parent.parent)),
                }
            )

    if not records:
        raise RuntimeError("No valid landmark sequences were extracted. Check raw videos directory.")

    df = pd.DataFrame(records)
    df = verify_and_rebalance_splits(df)
    df.to_csv(metadata_csv_path, index=False)

    split_counts = df["split"].value_counts().to_dict()
    avg_hand_rate = df["hand_detection_rate"].mean() * 100.0

    logger.info("=" * 65)
    logger.info("PREPROCESSING & LANDMARK EXTRACTION COMPLETE")
    logger.info("=" * 65)
    logger.info(f"Total Valid Sequences Saved : {len(df)}")
    logger.info(f"Corrupted / Too Short Videos: {corrupted_or_empty_count}")
    logger.info(f"Sequence Tensor Shape       : ({SEQUENCE_LENGTH}, {FEATURE_DIM}) float32")
    logger.info(
        f"Final Dataset Splits        : Train={split_counts.get('train', 0)} | "
        f"Val={split_counts.get('val', 0)} | Test={split_counts.get('test', 0)}"
    )
    logger.info(f"Unique Classes Represented  : {df['label'].nunique()} / {num_classes}")
    logger.info(f"Mean Hand Detection Rate    : {avg_hand_rate:.1f}%")
    logger.info(f"Metadata CSV Saved To       : {metadata_csv_path}")
    logger.info("=" * 65)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Holistic landmarks from WLASL-100 videos."
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help="Number of WLASL classes to preprocess (default: 100).",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional limit on number of videos for quick smoke testing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract landmarks even if .npy files already exist.",
    )
    args = parser.parse_args()

    run_preprocessing_pipeline(
        num_classes=args.num_classes,
        max_videos=args.max_videos,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()