import numpy as np


def distal_range_mm(
    dose: np.ndarray,
    spacing_mm: np.ndarray,
    beam_axis: int,
    fraction_of_max: float = 0.8,
) -> float:
    profile_axes = tuple(i for i in range(3) if i != beam_axis)
    profile = dose.mean(axis=profile_axes)

    threshold = fraction_of_max * float(np.max(profile))
    above = np.where(profile >= threshold)[0]
    if above.size == 0:
        return 0.0

    distal_index = float(above.max())
    return distal_index * float(spacing_mm[beam_axis])


def distal_error_mm(
    d_ref: np.ndarray,
    d_eval: np.ndarray,
    spacing_mm: np.ndarray,
    beam_axis: int,
    fraction_of_max: float = 0.8,
) -> float:
    r_ref = distal_range_mm(d_ref, spacing_mm, beam_axis, fraction_of_max)
    r_eval = distal_range_mm(d_eval, spacing_mm, beam_axis, fraction_of_max)
    return abs(r_ref - r_eval)
