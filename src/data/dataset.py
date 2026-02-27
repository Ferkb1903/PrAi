from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import defaults
from src.data.io_npz import list_npz_files, load_case_npz
from src.data.preprocess import make_e0_channel, maybe_crop_bev, normalize_dose_local


class ProtonDoseDataset(Dataset):
    """Dataset de casos NPZ para corrección de dosis 3D.

    x tiene 3 o 4 canales: [d_low_norm, spr, e0_map, beam_mask?].
    y puede ser residual (d_high - d_low) o d_high directo.
    """

    def __init__(
        self,
        directory: Path,
        predict_residual: bool = True,
        normalize_local: bool = True,
        include_beam_mask: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.predict_residual = predict_residual
        self.normalize_local = normalize_local
        self.include_beam_mask = include_beam_mask
        self.files: List[Path] = list_npz_files(self.directory)
        if not self.files:
            raise ValueError(f"No se encontraron casos NPZ en {self.directory}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case = load_case_npz(self.files[idx])

        # Recorte opcional para limitar memoria y centrar región de interés.
        d_low, spr, d_high = maybe_crop_bev(
            case.d_low,
            case.spr,
            case.d_high,
            crop_size=defaults.BEV_CROP_SIZE,
            enabled=defaults.USE_BEV_CROP,
        )

        beam_mask = None
        if case.beam_mask is not None:
            beam_mask = case.beam_mask
            if defaults.USE_BEV_CROP:
                beam_mask = maybe_crop_bev(
                    case.beam_mask,
                    case.beam_mask,
                    case.beam_mask,
                    crop_size=defaults.BEV_CROP_SIZE,
                    enabled=True,
                )[0]

        # Normalización local de dosis para estabilizar escala de entrenamiento.
        if self.normalize_local:
            d_low = normalize_dose_local(d_low)
            d_high = normalize_dose_local(d_high)

        # E0 entra como canal constante (misma shape que el volumen).
        e0_map = make_e0_channel(d_low.shape, case.e0_mev)

        channels = [d_low, spr.astype(np.float32), e0_map]
        if self.include_beam_mask and beam_mask is not None:
            channels.append(beam_mask.astype(np.float32))
        x = np.stack(channels, axis=0).astype(np.float32)
        y = (d_high - d_low).astype(np.float32) if self.predict_residual else d_high.astype(np.float32)

        sample: Dict[str, torch.Tensor] = {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y[None, ...]),
            "d_low": torch.from_numpy(d_low[None, ...].astype(np.float32)),
            "d_high": torch.from_numpy(d_high[None, ...].astype(np.float32)),
            "spacing_mm": torch.from_numpy(case.spacing_mm.astype(np.float32)),
            "beam_axis": torch.tensor(case.beam_axis, dtype=torch.long),
        }
        return sample
