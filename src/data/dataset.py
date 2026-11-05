"""
PyTorch Dataset and DataLoader module for WLASL-100 Landmark Sequences.
Includes physically-valid spatial-temporal coordinate augmentations for training
in low-sample regimes and kinematic velocity feature computation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    FEATURE_DIM,
    METADATA_CSV_PATH,
    PROJECT_ROOT,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    SEQUENCES_DIR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def augment_landmark_sequence(
    sequence: np.ndarray, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Apply stochastic, physically-valid spatial and temporal augmentations to a
    normalized landmark sequence of shape (T, 144).
    Note: Missing landmarks (strictly 0.0) are preserved as 0.0 so the model
    correctly learns missing-hand masks.

    Args:
        sequence: Normalized landmark array of shape (T, 144), dtype float32.
        rng: Optional NumPy random generator.

    Returns:
        Augmented landmark array of shape (T, 144), dtype float32.
    """
    if rng is None:
        rng = np.random.default_rng()

    seq = sequence.copy().astype(np.float32)
    T = seq.shape[0]

    # Reshape to (T, 48 landmarks, 3 coords [x, y, z])
    coords = seq.reshape(T, 48, 3)

    # Identify mask of valid (non-zero) landmarks: shape (T, 48)
    valid_mask = np.any(np.abs(coords) > 1e-6, axis=-1)

    # 1. Temporal Resampling / Speed Perturbation (50% probability)
    if rng.random() < 0.5 and T > 6:
        crop_ratio = rng.uniform(0.82, 1.0)
        crop_len = max(6, int(round(T * crop_ratio)))
        start_idx = int(rng.integers(0, T - crop_len + 1))
        sub_coords = coords[start_idx : start_idx + crop_len]
        resample_idx = np.round(np.linspace(0, crop_len - 1, T)).astype(int)
        coords = sub_coords[resample_idx]
        valid_mask = np.any(np.abs(coords) > 1e-6, axis=-1)

    # 2. Random 2D In-Plane Rotation around shoulder origin (70% probability, +/- 12 deg)
    if rng.random() < 0.7:
        angle_rad = float(rng.uniform(-np.deg2rad(12.0), np.deg2rad(12.0)))
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rot_matrix = np.array(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        coords = coords @ rot_matrix.T

    # 3. Random Anisotropic Scaling (70% probability, 0.85x to 1.15x)
    if rng.random() < 0.7:
        scale_xyz = rng.uniform(0.85, 1.15, size=(1, 1, 3)).astype(np.float32)
        coords = coords * scale_xyz

    # 4. Random Spatial Translation Jitter (50% probability, +/- 0.04)
    if rng.random() < 0.5:
        shift_xy = rng.uniform(-0.04, 0.04, size=(1, 1, 3)).astype(np.float32)
        shift_xy[0, 0, 2] = 0.0
        coords = coords + shift_xy

    # 5. Additive Gaussian Coordinate Noise (60% probability)
    if rng.random() < 0.6:
        noise = rng.normal(0.0, 0.006, size=coords.shape).astype(np.float32)
        coords = coords + noise

    # Restore zero mask for missing hands/landmarks
    coords[~valid_mask] = 0.0

    # 6. Random Frame Dropout (30% probability: zero out 1-2 random frames for occlusion robustness)
    if rng.random() < 0.3 and T > 10:
        num_drop = int(rng.integers(1, 3))
        drop_indices = rng.choice(T, size=num_drop, replace=False)
        # Drop only hand features (landmarks 6..47) in those frames
        coords[drop_indices, 6:, :] = 0.0

    return coords.reshape(T, FEATURE_DIM).astype(np.float32)


def resolve_sequence_file_path(
    npy_path_str: str, video_id: str, project_root: Path = PROJECT_ROOT
) -> Path:
    """
    Robustly resolve the path to an extracted .npy sequence file across relative
    roots (PROJECT_ROOT, DATA_DIR, or SEQUENCES_DIR).
    """
    rel_path = Path(npy_path_str)
    candidates = [
        rel_path if rel_path.is_absolute() else (project_root / rel_path),
        DATA_DIR / rel_path,
        SEQUENCES_DIR / f"{video_id}.npy",
        SEQUENCES_DIR / f"{int(video_id):05d}.npy" if video_id.isdigit() else SEQUENCES_DIR / f"{video_id}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Sequence file missing for video_id={video_id} (tried {[str(c) for c in candidates[:3]]})"
    )


class WLASLDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed WLASL-100 (30, 144) landmark sequences.
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        project_root: Path = PROJECT_ROOT,
        augment: bool = False,
        seed: int = RANDOM_SEED,
    ) -> None:
        super().__init__()
        self.df = metadata_df.reset_index(drop=True)
        self.project_root = project_root
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        self.sequences: List[np.ndarray] = []
        self.labels: List[int] = []
        self.video_ids: List[str] = []
        self.glosses: List[str] = []

        # Preload all compact (30, 144) tensors into RAM (~1,013 * 17KB = ~17 MB total)
        # for ultra-fast training epochs on CPU.
        for _, row in self.df.iterrows():
            vid = str(row["video_id"]).strip()
            # Ensure 5-digit zero-padded string if numeric (e.g., 7069 -> "07069")
            if vid.isdigit() and len(vid) < 5:
                vid = f"{int(vid):05d}"

            full_path = resolve_sequence_file_path(
                str(row["npy_path"]), vid, self.project_root
            )

            arr = np.load(full_path).astype(np.float32)
            if arr.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
                raise ValueError(
                    f"Invalid sequence shape {arr.shape} in {full_path}, "
                    f"expected ({SEQUENCE_LENGTH}, {FEATURE_DIM})"
                )

            self.sequences.append(arr)
            self.labels.append(int(row["label"]))
            self.video_ids.append(vid)
            self.glosses.append(str(row["gloss"]))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        if self.augment:
            seq = augment_landmark_sequence(seq, rng=self.rng)

        x_tensor = torch.from_numpy(seq).float()  # (30, 144)
        y_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return x_tensor, y_tensor


def create_dataloaders(
    metadata_csv_path: Path = METADATA_CSV_PATH,
    batch_size: int = BATCH_SIZE,
    top_k_classes: Optional[int] = None,
    seed: int = RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str], torch.Tensor]:
    """
    Create train, validation, and test PyTorch DataLoaders from metadata.csv,
    along with inverse class-frequency weights for balanced training.

    Args:
        metadata_csv_path: Path to data/processed/metadata.csv.
        batch_size: Mini-batch size.
        top_k_classes: Optional filter to train on the first K classes.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, idx_to_gloss, class_weights).
    """
    if not metadata_csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv_path}")

    df = pd.read_csv(metadata_csv_path, dtype={"video_id": str})

    if top_k_classes is not None and top_k_classes > 0:
        df = df[df["label"] < top_k_classes].copy()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    num_classes = int(df["label"].max()) + 1
    idx_to_gloss: Dict[int, str] = {}
    for _, row in df[["label", "gloss"]].drop_duplicates().iterrows():
        idx_to_gloss[int(row["label"])] = str(row["gloss"])

    # Compute smooth inverse class-frequency weights from training split
    class_counts = np.ones(num_classes, dtype=np.float32)
    for lbl, count in train_df["label"].value_counts().items():
        class_counts[int(lbl)] = float(count)

    # Smoothed inverse square-root weighting prevents extreme weight spikes
    weights = 1.0 / np.sqrt(class_counts)
    weights = weights / np.mean(weights)
    class_weights = torch.from_numpy(weights).float()

    train_dataset = WLASLDataset(train_df, augment=True, seed=seed)
    val_dataset = WLASLDataset(val_df, augment=False, seed=seed)
    test_dataset = WLASLDataset(test_df, augment=False, seed=seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    logger.info(
        f"Loaded DataLoaders ({num_classes} classes) -> "
        f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}"
    )
    return train_loader, val_loader, test_loader, idx_to_gloss, class_weights