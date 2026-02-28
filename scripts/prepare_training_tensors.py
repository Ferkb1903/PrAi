from __future__ import annotations

import argparse
import csv
import json
import random
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


def load_hu_spr_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = payload.get("points", [])
    if not points:
        raise ValueError(f"No se encontraron puntos HU->SPR en {path}")
    points_sorted = sorted(points, key=lambda p: float(p["hu"]))
    hu = np.asarray([float(p["hu"]) for p in points_sorted], dtype=np.float32)
    spr = np.asarray([float(p["spr"]) for p in points_sorted], dtype=np.float32)
    if np.any(np.diff(hu) <= 0):
        raise ValueError("La tabla HU->SPR debe tener HU estrictamente crecientes")
    return hu, spr


def hu_to_spr(hu_volume: np.ndarray, hu_points: np.ndarray, spr_points: np.ndarray) -> np.ndarray:
    hu_flat = hu_volume.reshape(-1)
    spr_flat = np.interp(hu_flat, hu_points, spr_points, left=float(spr_points[0]), right=float(spr_points[-1]))
    return spr_flat.reshape(hu_volume.shape).astype(np.float32)


def bragg_uncertainty(unc: np.ndarray, high_dose: np.ndarray) -> float:
    max_idx = np.unravel_index(int(np.argmax(high_dose)), high_dose.shape)
    value = float(unc[max_idx])
    if value > 1.0:
        value = value / 100.0
    return value


def qc_pair(
    low_dose: np.ndarray,
    high_dose: np.ndarray,
    low_unc: np.ndarray,
    high_unc: np.ndarray,
    max_uncertainty: float,
) -> QcResult:
    for name, arr in [("low", low_dose), ("high", high_dose), ("low_unc", low_unc), ("high_unc", high_unc)]:
        if not np.isfinite(arr).all():
            return QcResult(False, f"nan_or_inf_{name}", 1.0, 1.0)

    if np.min(low_dose) < 0 or np.min(high_dose) < 0:
        return QcResult(False, "negative_dose", 1.0, 1.0)

    if float(np.max(low_dose)) <= 0.0 or float(np.max(high_dose)) <= 0.0:
        return QcResult(False, "zero_or_empty_dose", 1.0, 1.0)

    low_u = bragg_uncertainty(low_unc, high_dose)
    high_u = bragg_uncertainty(high_unc, high_dose)
    if low_u > max_uncertainty or high_u > max_uncertainty:
        return QcResult(False, "high_uncertainty", low_u, high_u)

    return QcResult(True, "ok", low_u, high_u)


def normalize_doses_global(low: np.ndarray, high: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-8
    return (low / (scale + eps)).astype(np.float32), (high / (scale + eps)).astype(np.float32)


def build_beam_mask(low_dose_norm: np.ndarray, rel_threshold: float) -> np.ndarray:
    thr = float(np.max(low_dose_norm)) * rel_threshold
    mask = (low_dose_norm >= thr).astype(np.float32)
    return mask


def split_by_patient(
    case_ids: list[str],
    seed: int,
    train_patients: int,
    val_patients: int,
    test_patients: int,
) -> dict[str, set[str]]:
    unique = sorted(set(case_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)

    n = len(unique)
    if n == 0:
        return {"train": set(), "val": set(), "test": set()}

    if train_patients + val_patients + test_patients <= n:
        n_train = train_patients
        n_val = val_patients
        n_test = test_patients
    else:
        n_train = max(1, int(round(0.78 * n)))
        n_val = max(1, int(round(0.12 * n)))
        n_test = max(1, n - n_train - n_val)
        if n_train + n_val + n_test > n:
            n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_train > 1:
                n_train -= 1

    train = set(unique[:n_train])
    val = set(unique[n_train : n_train + n_val])
    test = set(unique[n_train + n_val : n_train + n_val + n_test])
    return {"train": train, "val": val, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser(description="QC + HU->SPR + normalización + empaquetado NPZ + split por paciente")
    parser.add_argument("--pair-index-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/training_npz/spot_campaign_v2"))
    parser.add_argument("--qc-report", type=Path, default=Path("data/training_npz/qc_spot_campaign.csv"))
    parser.add_argument("--manifest-all", type=Path, default=Path("data/training_npz/manifest_all.csv"))
    parser.add_argument("--manifest-train", type=Path, default=Path("data/training_npz/manifest_train.csv"))
    parser.add_argument("--manifest-val", type=Path, default=Path("data/training_npz/manifest_val.csv"))
    parser.add_argument("--manifest-test", type=Path, default=Path("data/training_npz/manifest_test.csv"))
    parser.add_argument("--split-summary", type=Path, default=Path("data/training_npz/split_summary.json"))
    parser.add_argument("--hu-spr-json", type=Path, default=Path("configs/hu_spr_schneider_v1.json"))
    parser.add_argument("--dose-stem", type=str, default="dose_voxelized_ct_edep")
    parser.add_argument("--max-uncertainty", type=float, default=0.10)
    parser.add_argument("--spr-max", type=float, default=2.0)
    parser.add_argument("--energy-norm-den", type=float, default=250.0)
    parser.add_argument("--beam-mask-rel-thr", type=float, default=0.05)
    parser.add_argument("--dose-norm-const", type=float, default=0.0, help="Si <=0 se estima global con p99 de máximos high")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-patients", type=int, default=40)
    parser.add_argument("--val-patients", type=int, default=6)
    parser.add_argument("--test-patients", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pair_rows = read_pair_rows(args.pair_index_csv)
    if args.limit > 0:
        pair_rows = pair_rows[: args.limit]
    if not pair_rows:
        raise RuntimeError("No hay filas para procesar en pair_index_csv")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.qc_report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_all.parent.mkdir(parents=True, exist_ok=True)

    hu_points, spr_points = load_hu_spr_points(args.hu_spr_json)

    qc_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []
    
    if args.dose_norm_const > 0:
        dose_scale = float(args.dose_norm_const)
    else:
        dose_scale = 1.0
    
    # DEBUG: Print dose_scale to verify it's being used
    print(f"\n{'='*70}")
    print(f"DOSE NORMALIZATION CONFIGURATION")
    print(f"{'='*70}")
    print(f"  dose_norm_const parameter: {args.dose_norm_const}")
    print(f"  dose_scale being used: {dose_scale:.6f}")
    print(f"  Output directory: {args.out_dir}")
    print(f"{'='*70}\n")

    for i, pair in enumerate(pair_rows, start=1):
        low_dir = _resolve_existing_dir(pair.low_out)
        high_dir = _resolve_existing_dir(pair.high_out)
        if low_dir is None or high_dir is None:
            qc_rows.append({
                "pair_idx": str(pair.pair_idx),
                "case_id": pair.case_id,
                "energy_mev": str(pair.energy_mev),
                "spot_idx": pair.spot_idx,
                "qc_ok": "0",
                "reason": "missing_low_or_high_dir",
                "low_unc_bragg": "",
                "high_unc_bragg": "",
                "npz_path": "",
            })
            continue

        low_mhd, low_unc_mhd = _dose_paths(low_dir, args.dose_stem)
        high_mhd, high_unc_mhd = _dose_paths(high_dir, args.dose_stem)
        if not (low_mhd.exists() and high_mhd.exists()):
            qc_rows.append({
                "pair_idx": str(pair.pair_idx),
                "case_id": pair.case_id,
                "energy_mev": str(pair.energy_mev),
                "spot_idx": pair.spot_idx,
                "qc_ok": "0",
                "reason": "missing_dose_files",
                "low_unc_bragg": "",
                "high_unc_bragg": "",
                "npz_path": "",
            })
            continue

        if not pair.pre_mhd.exists():
            qc_rows.append({
                "pair_idx": str(pair.pair_idx),
                "case_id": pair.case_id,
                "energy_mev": str(pair.energy_mev),
                "spot_idx": pair.spot_idx,
                "qc_ok": "0",
                "reason": "missing_pre_mhd",
                "low_unc_bragg": "",
                "high_unc_bragg": "",
                "npz_path": "",
            })
            continue

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

        if low.shape != high.shape or low.shape != low_unc.shape or low.shape != high_unc.shape:
            qc_rows.append({
                "pair_idx": str(pair.pair_idx),
                "case_id": pair.case_id,
                "energy_mev": str(pair.energy_mev),
                "spot_idx": pair.spot_idx,
                "qc_ok": "0",
                "reason": "shape_mismatch",
                "low_unc_bragg": "",
                "high_unc_bragg": "",
                "npz_path": "",
            })
            del low, high, low_unc, high_unc, low_img
            continue

        qc = qc_pair(low, high, low_unc, high_unc, args.max_uncertainty)
        if not qc.ok:
            qc_rows.append({
                "pair_idx": str(pair.pair_idx),
                "case_id": pair.case_id,
                "energy_mev": str(pair.energy_mev),
                "spot_idx": pair.spot_idx,
                "qc_ok": "0",
                "reason": qc.reason,
                "low_unc_bragg": f"{qc.low_unc_bragg:.6f}",
                "high_unc_bragg": f"{qc.high_unc_bragg:.6f}",
                "npz_path": "",
            })
            del low, high, low_unc, high_unc, low_img
            continue

        ct_img, hu = _read_image(pair.pre_mhd)
        if hu.shape != low.shape:
            qc_rows.append({
                "pair_idx": str(pair.pair_idx),
                "case_id": pair.case_id,
                "energy_mev": str(pair.energy_mev),
                "spot_idx": pair.spot_idx,
                "qc_ok": "0",
                "reason": "ct_dose_shape_mismatch",
                "low_unc_bragg": f"{qc.low_unc_bragg:.6f}",
                "high_unc_bragg": f"{qc.high_unc_bragg:.6f}",
                "npz_path": "",
            })
            del low, high, low_unc, high_unc, low_img, hu, ct_img
            continue

        spr = hu_to_spr(hu, hu_points, spr_points)
        spacing_mm = tuple(float(x) for x in low_img.GetSpacing()[::-1])

        low_n, high_n = normalize_doses_global(low, high, dose_scale)
        spr_n = np.clip(spr, 0.0, args.spr_max) / max(args.spr_max, 1e-6)
        beam_mask = build_beam_mask(low_n, args.beam_mask_rel_thr)

        case = CaseData(
            d_low=low_n,
            spr=spr_n.astype(np.float32),
            d_high=high_n,
            e0_mev=float(pair.energy_mev / max(args.energy_norm_den, 1e-6)),
            spacing_mm=np.asarray(spacing_mm, dtype=np.float32),
            beam_axis=2,
            case_id=pair.case_id,
            beam_mask=beam_mask,
        )

        npz_path = args.out_dir / f"{pair.case_id}_E{int(round(pair.energy_mev))}_spot_{pair.spot_idx}.npz"
        save_case_npz(npz_path, case)

        qc_rows.append({
            "pair_idx": str(pair.pair_idx),
            "case_id": pair.case_id,
            "energy_mev": str(pair.energy_mev),
            "spot_idx": pair.spot_idx,
            "qc_ok": "1",
            "reason": "ok",
            "low_unc_bragg": "",
            "high_unc_bragg": "",
            "npz_path": str(npz_path),
        })

        manifest_rows.append({
            "npz_path": str(npz_path),
            "case_id": pair.case_id,
            "energy_mev": f"{pair.energy_mev:.3f}",
            "spot_idx": pair.spot_idx,
        })
        
        del low, high, low_unc, high_unc, low_img, hu, ct_img, spr, spr_n, beam_mask, case

        if i % 200 == 0:
            print(f"Procesados {i}/{len(pair_rows)}, totales ok: {len(manifest_rows)}")

    splits = split_by_patient(
        case_ids=[r["case_id"] for r in manifest_rows],
        seed=args.seed,
        train_patients=args.train_patients,
        val_patients=args.val_patients,
        test_patients=args.test_patients,
    )

    split_rows: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for r in manifest_rows:
        cid = r["case_id"]
        if cid in splits["train"]:
            split_rows["train"].append(r)
        elif cid in splits["val"]:
            split_rows["val"].append(r)
        elif cid in splits["test"]:
            split_rows["test"].append(r)

    qc_fields = [
        "pair_idx",
        "case_id",
        "energy_mev",
        "spot_idx",
        "qc_ok",
        "reason",
        "low_unc_bragg",
        "high_unc_bragg",
        "npz_path",
    ]
    with args.qc_report.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=qc_fields)
        w.writeheader()
        w.writerows(qc_rows)

    manifest_fields = ["npz_path", "case_id", "energy_mev", "spot_idx"]
    for path, rows in [
        (args.manifest_all, manifest_rows),
        (args.manifest_train, split_rows["train"]),
        (args.manifest_val, split_rows["val"]),
        (args.manifest_test, split_rows["test"]),
    ]:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=manifest_fields)
            w.writeheader()
            w.writerows(rows)

    summary = {
        "pairs_input": len(pair_rows),
        "pairs_qc_ok": len(manifest_rows),
        "pairs_qc_bad": len(pair_rows) - len(manifest_rows),
        "dose_norm_const": dose_scale,
        "split_patients": {
            "train": sorted(splits["train"]),
            "val": sorted(splits["val"]),
            "test": sorted(splits["test"]),
        },
        "split_examples": {
            "train": len(split_rows["train"]),
            "val": len(split_rows["val"]),
            "test": len(split_rows["test"]),
        },
    }
    args.split_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Input pairs: {len(pair_rows)}")
    print(f"QC ok: {len(manifest_rows)} | QC bad: {len(pair_rows) - len(manifest_rows)}")
    print(f"Dose normalization constant: {dose_scale:.6f}")
    print(f"Train/Val/Test examples: {len(split_rows['train'])}/{len(split_rows['val'])}/{len(split_rows['test'])}")
    print(f"QC report: {args.qc_report}")
    print(f"Manifest all: {args.manifest_all}")
    print(f"Split summary: {args.split_summary}")


if __name__ == "__main__":
    main()
