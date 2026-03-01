#!/usr/bin/env python3
"""
Procesa un ÚNICO par (case/energy/spot) y genera su NPZ.
Diseñado para ejecutarse en paralelo con Slurm.
Uso: python scripts/prepare_training_tensors_parallel.py \
    --pair-index-csv data/training_npz/pair_index_low5k.csv \
    --pair-idx 0 \
    --out-dir data/training_npz/spot_campaign_v2_low5k
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io_npz import save_case_npz
from src.data.schema import CaseData


def _to_jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


@dataclass
class PairRow:
    pair_idx: int
    case_id: str
    energy_mev: float
    spot_idx: str
    pre_mhd: Path
    low_out: Path
    high_out: Path


@dataclass
class QcResult:
    ok: bool
    reason: str
    low_unc_bragg: float
    high_unc_bragg: float


def _normalize_rel_path(value: str) -> str:
    return value.replace("\\", "/").strip().replace("\r", "")


def _resolve_existing_dir(path: Path) -> Path | None:
    if path.exists() and path.is_dir():
        return path
    alt = Path(str(path) + "\r")
    if alt.exists() and alt.is_dir():
        return alt
    return None


def _read_image(path: Path) -> tuple[sitk.Image, np.ndarray]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return img, arr


def _dose_paths(folder: Path, dose_stem: str) -> tuple[Path, Path]:
    return folder / f"{dose_stem}.mhd", folder / f"{dose_stem}-Uncertainty.mhd"


def read_pair_rows(pair_index_csv: Path) -> list[PairRow]:
    """Lee TODOS los pares del CSV"""
    rows = list(csv.DictReader(pair_index_csv.open(encoding="utf-8")))
    out: list[PairRow] = []
    for i, r in enumerate(rows):
        out.append(
            PairRow(
                pair_idx=i,
                case_id=str(r["case_id"]),
                energy_mev=float(r["energy_mev"]),
                spot_idx=str(r["spot_idx"]),
                pre_mhd=Path(_normalize_rel_path(r["pre_mhd"])),
                low_out=Path(_normalize_rel_path(r["low_out"])),
                high_out=Path(_normalize_rel_path(r["high_out"])),
            )
        )
    return out


def load_hu_spr_points(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carga tabla HU → SPR desde JSON"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if "points" in data:
        # Formato: {"points": [{"hu": ..., "spr": ...}, ...]}
        points = data["points"]
        hu_vals = np.array([p["hu"] for p in points], dtype=np.float32)
        spr_vals = np.array([p["spr"] for p in points], dtype=np.float32)
    else:
        # Formato legacy: {"hu": [...], "spr": [...]}
        hu_vals = np.array(data["hu"], dtype=np.float32)
        spr_vals = np.array(data["spr"], dtype=np.float32)
    return hu_vals, spr_vals


def hu_to_spr(hu: np.ndarray, hu_points: np.ndarray, spr_points: np.ndarray) -> np.ndarray:
    """Mapea HU → SPR usando interpolación linear"""
    return np.interp(hu, hu_points, spr_points).astype(np.float32)


def normalize_doses_global(
    low: np.ndarray, high: np.ndarray, dose_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Normaliza dosis con factor global"""
    if dose_scale <= 0:
        dose_scale = 1.0
    return low / dose_scale, high / dose_scale


def build_beam_mask(low_dose_norm: np.ndarray, rel_threshold: float) -> np.ndarray:
    thr = float(np.max(low_dose_norm)) * rel_threshold
    return (low_dose_norm >= thr).astype(np.float32)


def qc_pair(
    low: np.ndarray,
    high: np.ndarray,
    low_unc: np.ndarray,
    high_unc: np.ndarray,
    max_uncertainty: float,
) -> QcResult:
    """Valida un par low/high dose"""
    if low.shape != high.shape:
        return QcResult(False, "shape_mismatch", 0.0, 0.0)
    
    # Detecta bragg peak extremo
    if np.max(high) < 1e-6:
        return QcResult(False, "high_too_small", 0.0, float(np.max(high)))
    if np.max(low) < 1e-6:
        return QcResult(False, "low_too_small", float(np.max(low)), 0.0)
    
    low_unc_bragg = np.max(low_unc) / np.max(low)
    high_unc_bragg = np.max(high_unc) / np.max(high)
    
    if low_unc_bragg > max_uncertainty:
        return QcResult(False, "low_unc_too_high", low_unc_bragg, high_unc_bragg)
    if high_unc_bragg > max_uncertainty:
        return QcResult(False, "high_unc_too_high", low_unc_bragg, high_unc_bragg)
    
    return QcResult(True, "ok", low_unc_bragg, high_unc_bragg)


def process_single_pair(
    pair: PairRow,
    out_dir: Path,
    hu_points: np.ndarray,
    spr_points: np.ndarray,
    dose_scale: float,
    dose_stem: str = "dose_voxelized_ct_edep",
    spr_max: float = 2.0,
    max_uncertainty: float = 0.5,
    energy_norm_den: float = 250.0,
    beam_mask_rel_thr: float = 0.05,
) -> dict:
    """
    Procesa UN ÚNICO par y retorna dict con resultado.
    """
    result = {
        "pair_idx": pair.pair_idx,
        "case_id": pair.case_id,
        "energy_mev": pair.energy_mev,
        "spot_idx": pair.spot_idx,
        "qc_ok": False,
        "reason": "",
        "npz_path": "",
        "low_unc_bragg": 0.0,
        "high_unc_bragg": 0.0,
    }

    # Valida directorios
    low_dir = _resolve_existing_dir(pair.low_out)
    high_dir = _resolve_existing_dir(pair.high_out)
    if low_dir is None or high_dir is None:
        result["reason"] = "missing_low_or_high_dir"
        return result

    # Valida archivos MHD
    low_mhd, low_unc_mhd = _dose_paths(low_dir, dose_stem)
    high_mhd, high_unc_mhd = _dose_paths(high_dir, dose_stem)
    if not (low_mhd.exists() and high_mhd.exists()):
        result["reason"] = "missing_dose_files"
        return result

    if not pair.pre_mhd.exists():
        result["reason"] = "missing_pre_mhd"
        return result

    # Lee imágenes
    try:
        low_img, low = _read_image(low_mhd)
        _, high = _read_image(high_mhd)
        if low_unc_mhd.exists():
            _, low_unc = _read_image(low_unc_mhd)
        else:
            low_unc = np.full_like(low, 0.01, dtype=np.float32)
        if high_unc_mhd.exists():
            _, high_unc = _read_image(high_unc_mhd)
        else:
            high_unc = np.full_like(high, 0.01, dtype=np.float32)
    except Exception as e:
        result["reason"] = f"read_error: {str(e)}"
        return result

    # Valida shapes
    if low.shape != high.shape or low.shape != low_unc.shape or low.shape != high_unc.shape:
        result["reason"] = "shape_mismatch"
        return result

    # QC de dosis
    qc = qc_pair(low, high, low_unc, high_unc, max_uncertainty)
    if not qc.ok:
        result["reason"] = qc.reason
        result["low_unc_bragg"] = qc.low_unc_bragg
        result["high_unc_bragg"] = qc.high_unc_bragg
        return result

    # Lee CT
    try:
        ct_img, hu = _read_image(pair.pre_mhd)
    except Exception as e:
        result["reason"] = f"ct_read_error: {str(e)}"
        return result

    if hu.shape != low.shape:
        result["reason"] = "ct_dose_shape_mismatch"
        result["low_unc_bragg"] = qc.low_unc_bragg
        result["high_unc_bragg"] = qc.high_unc_bragg
        return result

    # Procesa HU→SPR
    spr = hu_to_spr(hu, hu_points, spr_points)
    spacing_mm = tuple(float(x) for x in low_img.GetSpacing()[::-1])

    # Normaliza dosis
    low_n, high_n = normalize_doses_global(low, high, dose_scale)
    spr_n = np.clip(spr, 0.0, spr_max) / max(spr_max, 1e-6)

    beam_mask = build_beam_mask(low_n, beam_mask_rel_thr)

    # Construye CaseData
    case_data = CaseData(
        d_low=low_n.astype(np.float32),
        spr=spr_n.astype(np.float32),
        d_high=high_n.astype(np.float32),
        e0_mev=float(pair.energy_mev / max(energy_norm_den, 1e-6)),
        spacing_mm=np.asarray(spacing_mm, dtype=np.float32),
        beam_axis=2,
        case_id=pair.case_id,
        beam_mask=beam_mask,
    )

    # Genera nombre archivo NPZ
    safe_case = pair.case_id.replace("/", "_").replace(" ", "_")
    npz_filename = f"{safe_case}_E{pair.energy_mev:.1f}_spot_{pair.spot_idx}.npz"
    npz_path = out_dir / npz_filename

    # Guarda NPZ
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_case_npz(npz_path, case_data)
        result["qc_ok"] = True
        result["npz_path"] = str(npz_path)
        result["reason"] = "ok"
        result["low_unc_bragg"] = qc.low_unc_bragg
        result["high_unc_bragg"] = qc.high_unc_bragg
    except Exception as e:
        result["reason"] = f"save_error: {str(e)}"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Procesa UN ÚNICO pair (para ejecución paralela en Slurm)"
    )
    parser.add_argument("--pair-index-csv", type=Path, required=True)
    parser.add_argument("--pair-idx", type=int, required=True, help="Índice del pair a procesar (0-based)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/training_npz/spot_campaign_v2_low5k"))
    parser.add_argument("--hu-spr-json", type=Path, default=Path("configs/hu_spr_schneider_v1.json"))
    parser.add_argument("--dose-stem", default="dose_voxelized_ct_edep")
    parser.add_argument("--dose-norm-const", type=float, default=1.0)
    parser.add_argument("--spr-max", type=float, default=2.0)
    parser.add_argument("--max-uncertainty", type=float, default=0.5)
    parser.add_argument("--energy-norm-den", type=float, default=250.0)
    parser.add_argument("--beam-mask-rel-thr", type=float, default=0.05)

    args = parser.parse_args()

    # Lee TODOS los pares
    pair_rows = read_pair_rows(args.pair_index_csv)
    if args.pair_idx < 0 or args.pair_idx >= len(pair_rows):
        print(f"ERROR: pair_idx={args.pair_idx} out of range [0, {len(pair_rows)-1}]", file=sys.stderr)
        sys.exit(1)

    # Selecciona el pair a procesar
    pair = pair_rows[args.pair_idx]

    # Carga tabla HU→SPR
    hu_points, spr_points = load_hu_spr_points(args.hu_spr_json)

    dose_scale = float(args.dose_norm_const) if args.dose_norm_const > 0 else 1.0

    # Procesa el pair
    result = process_single_pair(
        pair=pair,
        out_dir=args.out_dir,
        hu_points=hu_points,
        spr_points=spr_points,
        dose_scale=dose_scale,
        dose_stem=args.dose_stem,
        spr_max=args.spr_max,
        max_uncertainty=args.max_uncertainty,
        energy_norm_den=args.energy_norm_den,
        beam_mask_rel_thr=args.beam_mask_rel_thr,
    )

    # Escribe resultado a JSON (para agregador)
    result_json = args.out_dir / f"pair_{args.pair_idx:06d}.json"
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(
        json.dumps(_to_jsonable(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Log en stdout
    status = "✓" if result["qc_ok"] else "✗"
    print(f"[{args.pair_idx:06d}] {pair.case_id} E{pair.energy_mev} spot_{pair.spot_idx} {status}")
    if not result["qc_ok"]:
        print(f"  → {result['reason']}")


if __name__ == "__main__":
    main()
