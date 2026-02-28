#!/usr/bin/env python3
"""
Agrega resultados de procesamiento paralelo en Slurm.
Lee todos los JSONs generados por prepare_training_tensors_parallel.py
y construye manifests + split + QC report.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def split_by_patient(
    case_ids: list[str],
    seed: int = 42,
    train_patients: int = 25,
    val_patients: int = 6,
    test_patients: int = 5,
) -> dict[str, Set[str]]:
    """Divide case_ids en train/val/test por paciente (case_id único)"""
    unique = sorted(set(case_ids))
    random.Random(seed).shuffle(unique)

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
    parser = argparse.ArgumentParser(description="Agrega resultados de procesamiento paralelo")
    parser.add_argument("--out-dir", type=Path, default=Path("data/training_npz/spot_campaign_v2_low5k"))
    parser.add_argument("--qc-report", type=Path, default=Path("data/training_npz/qc_spot_campaign_low5k.csv"))
    parser.add_argument("--manifest-all", type=Path, default=Path("data/training_npz/manifest_all_low5k.csv"))
    parser.add_argument("--manifest-train", type=Path, default=Path("data/training_npz/manifest_train_low5k.csv"))
    parser.add_argument("--manifest-val", type=Path, default=Path("data/training_npz/manifest_val_low5k.csv"))
    parser.add_argument("--manifest-test", type=Path, default=Path("data/training_npz/manifest_test_low5k.csv"))
    parser.add_argument("--split-summary", type=Path, default=Path("data/training_npz/split_summary_low5k.json"))
    parser.add_argument("--train-patients", type=int, default=25)
    parser.add_argument("--val-patients", type=int, default=6)
    parser.add_argument("--test-patients", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Crea directorios
    args.qc_report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_all.parent.mkdir(parents=True, exist_ok=True)

    # Lee todos los JSONs de resultados
    json_files = sorted(args.out_dir.glob("pair_*.json"))
    if not json_files:
        print(f"ERROR: No se encontraron archivos pair_*.json en {args.out_dir}", file=sys.stderr)
        sys.exit(1)

    qc_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, str]] = []

    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"ERROR leyendo {json_file}: {e}", file=sys.stderr)
            continue

        qc_row = {
            "pair_idx": str(data["pair_idx"]),
            "case_id": data["case_id"],
            "energy_mev": str(data["energy_mev"]),
            "spot_idx": data["spot_idx"],
            "qc_ok": "1" if data["qc_ok"] else "0",
            "reason": data["reason"],
            "low_unc_bragg": f"{data.get('low_unc_bragg', 0.0):.6f}",
            "high_unc_bragg": f"{data.get('high_unc_bragg', 0.0):.6f}",
            "npz_path": data["npz_path"],
        }
        qc_rows.append(qc_row)

        if data["qc_ok"] and data["npz_path"]:
            manifest_rows.append({
                "npz_path": data["npz_path"],
                "case_id": data["case_id"],
                "energy_mev": str(data["energy_mev"]),
                "spot_idx": data["spot_idx"],
            })

    # Split por paciente
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

    # Escribe QC report
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

    # Escribe manifests
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

    # Resumen
    summary = {
        "pairs_input": len(qc_rows),
        "pairs_qc_ok": len(manifest_rows),
        "pairs_qc_bad": len(qc_rows) - len(manifest_rows),
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
    args.split_summary.parent.mkdir(parents=True, exist_ok=True)
    args.split_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Imprime resumen
    print(f"\n{'='*70}")
    print(f"AGREGACIÓN COMPLETADA")
    print(f"{'='*70}")
    print(f"Input pairs: {len(qc_rows)}")
    print(f"QC ok: {len(manifest_rows)} | QC bad: {len(qc_rows) - len(manifest_rows)}")
    print(f"Train/Val/Test: {len(split_rows['train'])}/{len(split_rows['val'])}/{len(split_rows['test'])}")
    print(f"QC report: {args.qc_report}")
    print(f"Manifests: {args.manifest_train}, {args.manifest_val}, {args.manifest_test}")
    print(f"Split summary: {args.split_summary}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
