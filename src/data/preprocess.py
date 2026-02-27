from typing import Tuple

import numpy as np

from src.config import defaults


def normalize_dose_local(dose: np.ndarray, eps: float = defaults.DOSE_NORM_EPS) -> np.ndarray:
    max_val = float(np.max(dose))
    return dose / (max_val + eps)


def normalize_e0(e0_mev: float) -> float:
    e0_clamped = min(max(e0_mev, defaults.E0_MIN_MEV), defaults.E0_MAX_MEV)
    denom = defaults.E0_MAX_MEV - defaults.E0_MIN_MEV
    if denom <= 0:
        return 0.0
    return (e0_clamped - defaults.E0_MIN_MEV) / denom


def make_e0_channel(shape_3d: Tuple[int, int, int], e0_mev: float) -> np.ndarray:
    e0_norm = normalize_e0(e0_mev)
    return np.full(shape_3d, e0_norm, dtype=np.float32)


def _center_crop_indices(size: int, crop: int) -> Tuple[int, int]:
    if crop >= size:
        return 0, size
    start = (size - crop) // 2
    return start, start + crop


def center_crop_3d(volume: np.ndarray, crop_size: Tuple[int, int, int]) -> np.ndarray:
    d, h, w = volume.shape
    cd, ch, cw = crop_size
    ds, de = _center_crop_indices(d, cd)
    hs, he = _center_crop_indices(h, ch)
    ws, we = _center_crop_indices(w, cw)
    return volume[ds:de, hs:he, ws:we]


def maybe_crop_bev(
    d_low: np.ndarray,
    spr: np.ndarray,
    d_high: np.ndarray,
    crop_size: Tuple[int, int, int],
    enabled: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not enabled:
        return d_low, spr, d_high
    return (
        center_crop_3d(d_low, crop_size),
        center_crop_3d(spr, crop_size),
        center_crop_3d(d_high, crop_size),
    )
