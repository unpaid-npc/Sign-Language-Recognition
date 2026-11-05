# WLASL-100 Dataset Guide

## 1. Overview of WLASL and WLASL-100

**WLASL (Word-Level American Sign Language)** is the largest public benchmark dataset for isolated American Sign Language recognition, introduced by Li et al. (WACV 2020).
- **Full Dataset:** 2,000 ASL glosses (words).
- **WLASL-100 Subset:** The top **100 glosses** ordered by instance frequency in `WLASL_v0.3.json` (indices `0` through `99`), including everyday vocabulary such as *book, drink, computer, before, chair, go, clothes, who, candy, cousin, deaf, fine, help, no, thin, walk, year, yes*, etc.

## 2. Dataset Availability & Acquisition Challenges

The original `WLASL_v0.3.json` distributes URLs pointing to external video sources (YouTube, ASLPro, SpreadTheSign). Over time, ~45–50% of those third-party URLs have experienced link rot. To ensure reproducibility without broken scrapers:
- `src/data/download_wlasl.py` fetches the official `WLASL_v0.3.json` annotation file and downloads the pre-cropped video archive (`risangbaskoro/wlasl-processed`) via `kagglehub`.
- It filters and copies strictly the available WLASL-100 `.mp4` files into `data/raw/videos/` and cleans the temporary cache.

## 3. Actual Preprocessed Dataset Telemetry

| Property | Count / Specification |
| :--- | :--- |
| **Annotated Classes** | 100 |
| **Represented Classes Locally** | 100 / 100 (100%) |
| **Total Annotated Instances** | 2,038 |
| **Available Local `.mp4` Videos** | 1,013 (`0` corrupted) |
| **Unique Signers** | 97 |
| **Training Split (`train`)** | 709 sequences |
| **Validation Split (`val`)** | 176 sequences |
| **Testing Split (`test`)** | 128 sequences |
| **Extracted Tensor Format** | `(30, 144)` `float32` `.npy` (`17.3 KB` per sequence) |

## 4. Data Leakage Prevention

`src/data/preprocessing.py` enforces strict disjointness across splits:
1. Every video has a unique 5-digit `video_id`.
2. Official `train`, `val`, and `test` assignments from `WLASL_v0.3.json` are preserved.
3. For any class where missing third-party URLs left `0` validation or test clips, `verify_and_rebalance_splits()` deterministically reallocates 1 clip from `train` so all 100 classes can be validated and tested.
4. Set-intersection assertions (`assert len(train_vids & val_vids) == 0`, etc.) run automatically before saving `metadata.csv`.