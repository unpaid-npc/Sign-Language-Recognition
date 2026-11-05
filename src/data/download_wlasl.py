"""
WLASL-100 Dataset Acquisition and Inspection Script.
Downloads WLASL_v0.3.json and acquires preprocessed WLASL-100 videos via KaggleHub,
filtering strictly to the WLASL-100 subset to conserve disk space.
"""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Set, Tuple

from tqdm import tqdm

from src.config import NUM_CLASSES, RAW_DATA_DIR, RAW_VIDEOS_DIR, WLASL_JSON_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

OFFICIAL_WLASL_JSON_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
)
KAGGLE_DATASET_HANDLE = "risangbaskoro/wlasl-processed"


def download_wlasl_json(target_path: Path = WLASL_JSON_PATH) -> Path:
    """
    Download the official WLASL_v0.3.json metadata file if not already present.
    """
    if target_path.exists() and target_path.stat().st_size > 100_000:
        logger.info(f"WLASL JSON already exists at: {target_path}")
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading WLASL_v0.3.json from {OFFICIAL_WLASL_JSON_URL} ...")
    urllib.request.urlretrieve(OFFICIAL_WLASL_JSON_URL, target_path)
    logger.info(
        f"Successfully saved WLASL_v0.3.json ({target_path.stat().st_size / 1e6:.2f} MB)"
    )
    return target_path


def parse_wlasl_subset(
    json_path: Path = WLASL_JSON_PATH, num_classes: int = NUM_CLASSES
) -> Tuple[List[Dict], Set[str]]:
    """
    Parse WLASL_v0.3.json and return the top `num_classes` entries and their video IDs.

    Args:
        json_path: Path to WLASL_v0.3.json.
        num_classes: Number of glosses to include (100 for WLASL-100).

    Returns:
        Tuple of (subset_entries, set_of_required_video_ids).
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    subset_data = full_data[:num_classes]
    required_video_ids: Set[str] = set()

    for entry in subset_data:
        for inst in entry.get("instances", []):
            vid = str(inst.get("video_id", "")).strip()
            if vid:
                required_video_ids.add(vid)

    return subset_data, required_video_ids


def copy_wlasl100_videos_from_kaggle_cache(
    kaggle_cache_dir: Path,
    required_video_ids: Set[str],
    dest_videos_dir: Path = RAW_VIDEOS_DIR,
) -> int:
    """
    Locate all .mp4 files inside the KaggleHub cache directory and copy only
    the videos belonging to WLASL-100 into data/raw/videos/.
    """
    dest_videos_dir.mkdir(parents=True, exist_ok=True)

    # Also copy WLASL_v0.3.json from cache if not already present
    if not WLASL_JSON_PATH.exists():
        cached_jsons = list(kaggle_cache_dir.rglob("WLASL_v0.3.json"))
        if cached_jsons:
            shutil.copy2(cached_jsons[0], WLASL_JSON_PATH)
            logger.info(f"Copied WLASL_v0.3.json from Kaggle cache to {WLASL_JSON_PATH}")

    logger.info(f"Scanning Kaggle cache at {kaggle_cache_dir} for .mp4 files...")
    all_mp4_files = list(kaggle_cache_dir.rglob("*.mp4"))
    logger.info(f"Found {len(all_mp4_files)} total .mp4 files in Kaggle archive.")

    copied_count = 0
    for mp4_path in tqdm(all_mp4_files, desc="Copying WLASL-100 videos"):
        video_id = mp4_path.stem
        if video_id in required_video_ids:
            target_file = dest_videos_dir / f"{video_id}.mp4"
            if not target_file.exists():
                shutil.copy2(mp4_path, target_file)
            copied_count += 1

    logger.info(
        f"Copied {copied_count} / {len(required_video_ids)} WLASL-100 videos to {dest_videos_dir}"
    )
    return copied_count


def inspect_dataset_availability(
    json_path: Path = WLASL_JSON_PATH,
    videos_dir: Path = RAW_VIDEOS_DIR,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, int]:
    """
    Inspect the WLASL-100 metadata and compare against locally available .mp4 files.
    """
    subset_data, required_ids = parse_wlasl_subset(json_path, num_classes)

    local_mp4_ids = {p.stem for p in videos_dir.glob("*.mp4")} if videos_dir.exists() else set()

    split_totals = {"train": 0, "val": 0, "test": 0}
    split_available = {"train": 0, "val": 0, "test": 0}
    unique_signers: Set[int] = set()

    for entry in subset_data:
        for inst in entry.get("instances", []):
            vid = str(inst.get("video_id", "")).strip()
            split = inst.get("split", "train")
            signer_id = inst.get("signer_id", -1)
            unique_signers.add(signer_id)

            if split in split_totals:
                split_totals[split] += 1
                if vid in local_mp4_ids:
                    split_available[split] += 1

    sample_glosses = [entry["gloss"] for entry in subset_data[:15]]
    total_annotated = sum(split_totals.values())
    total_local = sum(split_available.values())

    logger.info("=" * 65)
    logger.info(f"WLASL-{num_classes} DATASET INSPECTION SUMMARY")
    logger.info("=" * 65)
    logger.info(f"Total Classes (Glosses)   : {len(subset_data)}")
    logger.info(f"First 15 Glosses          : {', '.join(sample_glosses)}")
    logger.info(f"Total Unique Signers      : {len(unique_signers)}")
    logger.info(f"Total Annotated Instances : {total_annotated}")
    logger.info(
        f"Annotated Splits          : Train={split_totals['train']} | "
        f"Val={split_totals['val']} | Test={split_totals['test']}"
    )
    logger.info(f"Local Videos Found (.mp4) : {total_local} / {total_annotated}")
    if total_local > 0:
        logger.info(
            f"Available Local Splits    : Train={split_available['train']} | "
            f"Val={split_available['val']} | Test={split_available['test']}"
        )
    logger.info("=" * 65)

    return {
        "num_classes": len(subset_data),
        "total_annotated": total_annotated,
        "total_available": total_local,
        "train_available": split_available["train"],
        "val_available": split_available["val"],
        "test_available": split_available["test"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and inspect the WLASL-100 dataset."
    )
    parser.add_argument(
        "--download-videos",
        action="store_true",
        help="Download the preprocessed WLASL video archive via KaggleHub (~4.8 GB).",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Delete the full 2000-class KaggleHub cache after copying WLASL-100 videos.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help="Number of WLASL classes to inspect/extract (default: 100).",
    )
    args = parser.parse_args()

    # Step 1: Ensure WLASL_v0.3.json is downloaded
    download_wlasl_json(WLASL_JSON_PATH)
    _, required_ids = parse_wlasl_subset(WLASL_JSON_PATH, args.num_classes)

    # Step 2: Optionally download videos from KaggleHub
    if args.download_videos:
        import kagglehub

        logger.info(
            f"Downloading '{KAGGLE_DATASET_HANDLE}' via KaggleHub. "
            "Note: This archive is ~4.8 GB and may take 5-15 minutes depending on internet speed..."
        )
        cache_path_str = kagglehub.dataset_download(KAGGLE_DATASET_HANDLE)
        cache_path = Path(cache_path_str)

        copy_wlasl100_videos_from_kaggle_cache(
            kaggle_cache_dir=cache_path,
            required_video_ids=required_ids,
            dest_videos_dir=RAW_VIDEOS_DIR,
        )

        if args.clean_cache and cache_path.exists():
            logger.info(f"Cleaning up KaggleHub cache at {cache_path} to save disk space...")
            shutil.rmtree(cache_path, ignore_errors=True)

    # Step 3: Run dataset availability inspection
    inspect_dataset_availability(WLASL_JSON_PATH, RAW_VIDEOS_DIR, args.num_classes)


if __name__ == "__main__":
    main()