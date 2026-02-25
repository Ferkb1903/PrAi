from typing import Tuple

import numpy as np


def gamma_pass_rate_simplified(
    d_ref: np.ndarray,
    d_eval: np.ndarray,
    dose_percent: float = 3.0,
) -> float:
    if d_ref.shape != d_eval.shape:
        raise ValueError("d_ref y d_eval deben tener misma shape")

    ref_max = float(np.max(np.abs(d_ref))) + 1e-8
    tol = dose_percent / 100.0 * ref_max
    passing = np.abs(d_ref - d_eval) <= tol
    return float(np.mean(passing) * 100.0)


def gamma_report(
    d_ref: np.ndarray,
    d_eval: np.ndarray,
    criteria: Tuple[Tuple[float, float], ...] = ((3.0, 3.0), (2.0, 2.0)),
) -> dict:
    report = {}
    for dose_pct, _dist_mm in criteria:
        key = f"gamma_{dose_pct:.0f}pct"
        report[key] = gamma_pass_rate_simplified(d_ref, d_eval, dose_percent=dose_pct)
    return report
