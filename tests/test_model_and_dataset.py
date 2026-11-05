"""
Unit tests for Day 2: PyTorch Dataset loading, coordinate augmentations,
SignBiLSTMClassifier forward pass, and checkpoint serialization/deserialization.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import FEATURE_DIM, METADATA_CSV_PATH, SEQUENCE_LENGTH
from src.data.dataset import augment_landmark_sequence, create_dataloaders
from src.models.lstm import SignBiLSTMClassifier
from src.models.train import load_checkpoint, save_checkpoint


def test_augment_landmark_sequence_preserves_shape_and_zero_mask() -> None:
    """Verify coordinate augmentation preserves (30, 144) shape and keeps missing hands at 0.0."""
    raw = np.ones((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32) * 0.25
    # Simulate right hand missing (last 63 features = 0.0)
    raw[:, -63:] = 0.0

    rng = np.random.default_rng(42)
    aug = augment_landmark_sequence(raw, rng=rng)

    assert aug.shape == (SEQUENCE_LENGTH, FEATURE_DIM)
    assert aug.dtype == np.float32
    assert np.allclose(aug[:, -63:], 0.0)


def test_dataloaders_load_real_metadata() -> None:
    """Verify train, val, and test DataLoaders load preprocessed .npy tensors cleanly."""
    if not METADATA_CSV_PATH.exists():
        pytest.skip("metadata.csv not yet generated.")

    train_loader, val_loader, test_loader, idx_to_gloss, class_weights = create_dataloaders(
        batch_size=16
    )
    assert len(idx_to_gloss) == 100
    assert class_weights.shape == (100,)

    x_batch, y_batch = next(iter(train_loader))
    assert x_batch.ndim == 3
    assert x_batch.shape[1:] == (SEQUENCE_LENGTH, FEATURE_DIM)
    assert y_batch.ndim == 1
    assert x_batch.size(0) == y_batch.size(0)


def test_bilstm_forward_pass_and_invalid_input_rejection() -> None:
    """Verify SignBiLSTMClassifier forward pass output dimensions and input validation."""
    model = SignBiLSTMClassifier(num_classes=100)
    model.eval()

    dummy_input = torch.randn(8, SEQUENCE_LENGTH, FEATURE_DIM)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (8, 100)
    assert not torch.isnan(logits).any()

    # Verify wrong feature dimension raises ValueError
    bad_input = torch.randn(8, SEQUENCE_LENGTH, 63)
    with pytest.raises(ValueError):
        model(bad_input)


def test_checkpoint_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Verify model checkpoint saving and loading restores weights and label mapping."""
    model = SignBiLSTMClassifier(num_classes=10)
    idx_to_gloss = {i: f"sign_{i}" for i in range(10)}
    ckpt_file = tmp_path / "test_ckpt.pt"

    save_checkpoint(
        model=model,
        idx_to_gloss=idx_to_gloss,
        epoch=5,
        val_acc=0.45,
        val_top5_acc=0.80,
        path=ckpt_file,
    )
    assert ckpt_file.exists()

    loaded_model, loaded_map, ckpt_meta = load_checkpoint(
        ckpt_file, device=torch.device("cpu")
    )
    assert loaded_map == idx_to_gloss
    assert ckpt_meta["epoch"] == 5

    test_x = torch.randn(2, SEQUENCE_LENGTH, FEATURE_DIM)
    model.eval()
    with torch.no_grad():
        out_orig = model(test_x)
        out_loaded = loaded_model(test_x)
    assert torch.allclose(out_orig, out_loaded, atol=1e-6)