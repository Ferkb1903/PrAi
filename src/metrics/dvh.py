from typing import Dict

import numpy as np


def compute_dvh(dose: np.ndarray, mask: np.ndarray, bins: int = 100) -> Dict[str, np.ndarray]:
    vox = dose[mask > 0]
    if vox.size == 0:
        raise ValueError("Máscara vacía para DVH")

    d_min = float(np.min(vox))
    d_max = float(np.max(vox))
    edges = np.linspace(d_min, d_max, bins + 1)
    hist, _ = np.histogram(vox, bins=edges)

    cumsum_rev = np.cumsum(hist[::-1])[::-1].astype(np.float64)
    volume_frac = cumsum_rev / cumsum_rev[0]

    dose_axis = edges[:-1]
    return {
        "dose": dose_axis,
        "volume": volume_frac,
    }
