from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import defaults
from src.data.schema import CaseData
from src.data.io_npz import save_case_npz


def make_synthetic_case(case_id: str, shape=(64, 64, 64), e0_mev: float = 150.0) -> CaseData:
    d, h, w = shape
    zz, yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, d),
        np.linspace(0.0, 1.0, h),
        np.linspace(0.0, 1.0, w),
        indexing="ij",
    )

    spr = (1.0 + 0.2 * np.sin(2 * np.pi * yy) * np.cos(2 * np.pi * xx)).astype(np.float32)

    bragg_center = 0.7
    bragg_width = 0.04
    depth_profile = np.exp(-((zz - bragg_center) ** 2) / (2 * bragg_width**2))

    lateral = np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (2 * 0.08**2))
    d_high = (depth_profile * lateral * spr).astype(np.float32)

    noise_sigma = 0.08
    noise = np.random.normal(0.0, noise_sigma, size=shape).astype(np.float32)
    d_low = np.clip(d_high + noise, a_min=0.0, a_max=None)

    return CaseData(
        d_low=d_low,
        spr=spr,
        d_high=d_high,
        e0_mev=e0_mev,
        spacing_mm=np.array([2.0, 2.0, 2.0], dtype=np.float32),
        beam_axis=0,
        case_id=case_id,
    )


def write_split(split_dir: Path, n_cases: int, split_name: str) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_cases):
        case_id = f"{split_name}_{i:03d}"
        e0_mev = float(np.random.choice([90, 120, 150, 180, 210, 230]))
        case = make_synthetic_case(case_id=case_id, e0_mev=e0_mev)
        save_case_npz(split_dir / f"{case_id}.npz", case)


def main() -> None:
    np.random.seed(defaults.SEED)

    write_split(defaults.TRAIN_DIR, n_cases=8, split_name="train")
    write_split(defaults.VAL_DIR, n_cases=2, split_name="val")
    write_split(defaults.TEST_DIR, n_cases=2, split_name="test")

    print(f"Dataset demo generado en: {defaults.DATA_ROOT}")


if __name__ == "__main__":
    main()
