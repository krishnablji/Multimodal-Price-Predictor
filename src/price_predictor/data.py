"""Data validation, alignment, atomic storage, and submission generation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

REQUIRED_COLUMNS_BASE = ["sample_id", "catalog_content", "image_link"]
REQUIRED_COLUMNS_TRAIN = ["sample_id", "catalog_content", "image_link", "price"]


def ordered_id_hash(sample_ids: Sequence[Any]) -> str:
    """Generate deterministic SHA-256 hash for an ordered list of sample IDs."""
    hasher = hashlib.sha256()
    for sid in sample_ids:
        hasher.update(str(sid).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def validate_frame(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """Validate input DataFrame schema, missing values, duplicates, and target bounds."""
    required = REQUIRED_COLUMNS_TRAIN if training else REQUIRED_COLUMNS_BASE
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in DataFrame: {missing_cols}")

    if df["sample_id"].duplicated().any():
        dup_count = int(df["sample_id"].duplicated().sum())
        raise ValueError(f"Found {dup_count} duplicate sample_id entries.")

    if training:
        if df["price"].isnull().any():
            raise ValueError("Training DataFrame contains null values in 'price'.")
        if (df["price"] <= 0).any():
            raise ValueError("Training target 'price' must be strictly greater than zero.")
        if not np.issubdtype(df["price"].dtype, np.number):
            raise ValueError("Price column must be numeric.")

    return df


def align_by_sample_id(ordered_ids: Sequence[Any], df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame rows are aligned strictly to the given sample_id sequence."""
    indexed = df.set_index("sample_id")
    missing = [sid for sid in ordered_ids if sid not in indexed.index]
    if missing:
        raise ValueError(f"{len(missing)} sample IDs were not found in the DataFrame index.")
    return indexed.loc[list(ordered_ids)].reset_index()


def save_npy(path: str | Path, array: np.ndarray) -> None:
    """Atomically save a NumPy array using a temporary file to avoid partial writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dir_name = path.parent
    with tempfile.NamedTemporaryFile(dir=dir_name, delete=False, suffix=".npy") as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.save(tmp_path, array)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def load_npy(path: str | Path) -> np.ndarray:
    """Load a NumPy array from disk."""
    return np.load(path)


def validate_npy_artifact(
    path: str | Path,
    expected_shape: tuple[int, ...] | None = None,
    expected_dtype: str | None = None,
) -> np.ndarray:
    """Load and validate an array's dimensions, data type, and finite values."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Array artifact not found at: {path}")
    arr = np.load(path)
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(f"Shape mismatch for {path.name}: expected {expected_shape}, got {arr.shape}")
    if expected_dtype is not None and str(arr.dtype) != expected_dtype:
        raise ValueError(f"Dtype mismatch for {path.name}: expected {expected_dtype}, got {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values (NaN/Inf) detected in array: {path.name}")
    return arr


def build_submission(
    sample_ids: Sequence[Any],
    predictions: Sequence[float] | np.ndarray,
    min_price: float = 0.01,
) -> pd.DataFrame:
    """Create a validated submission DataFrame with positive, non-null prices."""
    if len(sample_ids) != len(predictions):
        raise ValueError(f"Length mismatch: {len(sample_ids)} IDs vs {len(predictions)} predictions.")
    
    preds = np.asarray(predictions, dtype=np.float64)
    if not np.all(np.isfinite(preds)):
        raise ValueError("Predictions contain NaN or Infinite values.")
    
    clipped_preds = np.clip(preds, a_min=min_price, a_max=None)
    
    return pd.DataFrame({
        "sample_id": list(sample_ids),
        "price": np.round(clipped_preds, 4)
    })


def load_pipeline_config(config_path: str | Path) -> tuple[dict[str, Any], str]:
    """Load canonical configuration JSON and compute its SHA-256 digest."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")
    
    raw_content = path.read_text(encoding="utf-8")
    config = json.loads(raw_content)
    digest = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
    
    # Contract checks
    expected_width = 2450
    if config.get("feature_width") != expected_width:
        raise ValueError(f"Invalid feature_width: expected {expected_width}, got {config.get('feature_width')}")
    
    return config, digest
