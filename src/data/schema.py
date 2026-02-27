from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


REQUIRED_KEYS = {
    "d_low",
    "spr",
    "d_high",
    "e0_mev",
    "spacing_mm",
    "beam_axis",
    "case_id",
}


@dataclass
class CaseData:
    d_low: np.ndarray
    spr: np.ndarray
    d_high: np.ndarray
    e0_mev: float
    spacing_mm: np.ndarray
    beam_axis: int
    case_id: str
    beam_mask: np.ndarray | None = None


def validate_case_dict(data: Dict[str, Any]) -> None:
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise KeyError(f"Faltan llaves obligatorias: {sorted(missing)}")

    d_low = np.asarray(data["d_low"])
    spr = np.asarray(data["spr"])
    d_high = np.asarray(data["d_high"])

    if d_low.ndim != 3 or spr.ndim != 3 or d_high.ndim != 3:
        raise ValueError("d_low, spr y d_high deben ser volúmenes 3D")

    if d_low.shape != spr.shape or d_low.shape != d_high.shape:
        raise ValueError("d_low, spr y d_high deben tener la misma shape")

    for name, arr in [("d_low", d_low), ("spr", spr), ("d_high", d_high)]:
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contiene NaN o Inf")

    if "beam_mask" in data and data["beam_mask"] is not None:
        beam_mask = np.asarray(data["beam_mask"])
        if beam_mask.ndim != 3:
            raise ValueError("beam_mask debe ser un volumen 3D")
        if beam_mask.shape != d_low.shape:
            raise ValueError("beam_mask debe tener la misma shape que d_low")
        if not np.isfinite(beam_mask).all():
            raise ValueError("beam_mask contiene NaN o Inf")

    spacing_mm = np.asarray(data["spacing_mm"], dtype=np.float32)
    if spacing_mm.shape != (3,):
        raise ValueError("spacing_mm debe tener shape (3,)")
    if (spacing_mm <= 0).any():
        raise ValueError("spacing_mm debe ser positivo")

    beam_axis = int(data["beam_axis"])
    if beam_axis not in (0, 1, 2):
        raise ValueError("beam_axis debe ser 0, 1 o 2")


def to_case_data(data: Dict[str, Any]) -> CaseData:
    validate_case_dict(data)
    beam_mask = None
    if "beam_mask" in data and data["beam_mask"] is not None:
        beam_mask = np.asarray(data["beam_mask"], dtype=np.float32)

    return CaseData(
        d_low=np.asarray(data["d_low"], dtype=np.float32),
        spr=np.asarray(data["spr"], dtype=np.float32),
        d_high=np.asarray(data["d_high"], dtype=np.float32),
        e0_mev=float(np.asarray(data["e0_mev"]).item()),
        spacing_mm=np.asarray(data["spacing_mm"], dtype=np.float32),
        beam_axis=int(data["beam_axis"]),
        case_id=str(np.asarray(data["case_id"]).item()),
        beam_mask=beam_mask,
    )
