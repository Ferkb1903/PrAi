from pathlib import Path
from typing import Dict, List

import numpy as np

from src.data.schema import CaseData, to_case_data


def list_npz_files(directory: Path) -> List[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob("*.npz"))


def load_case_npz(path: Path) -> CaseData:
    path = Path(path)
    with np.load(path, allow_pickle=True) as npz_data:
        payload: Dict[str, np.ndarray] = {key: npz_data[key] for key in npz_data.files}
    return to_case_data(payload)


def save_case_npz(path: Path, case: CaseData) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "d_low": case.d_low.astype(np.float32),
        "spr": case.spr.astype(np.float32),
        "d_high": case.d_high.astype(np.float32),
        "e0_mev": np.asarray(case.e0_mev, dtype=np.float32),
        "spacing_mm": case.spacing_mm.astype(np.float32),
        "beam_axis": np.asarray(case.beam_axis, dtype=np.int64),
        "case_id": np.asarray(case.case_id),
    }
    if case.beam_mask is not None:
        payload["beam_mask"] = case.beam_mask.astype(np.float32)
    np.savez_compressed(path, **payload)
