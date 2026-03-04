from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Material:
    name: str
    spr: float


@dataclass
class StageConfig:
    name: str
    n_cases: int
    energy_min: float
    energy_max: float
    noise_rel: float
    num_slabs_min: int
    num_slabs_max: int
    materials: list[Material]


def water_range_mm_from_energy(e0_mev: float) -> float:
    return float(10.0 * 0.0022 * (max(e0_mev, 1.0) ** 1.77))


def build_spr_slab_volume(
    shape: tuple[int, int, int],
    spacing_mm: np.ndarray,
    num_slabs: int,
    materials: list[Material],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    z_dim, y_dim, x_dim = shape
    slab_edges = np.sort(rng.choice(np.arange(1, z_dim - 1), size=max(0, num_slabs - 1), replace=False))
    slab_edges = np.concatenate(([0], slab_edges, [z_dim]))

    spr_line = np.zeros((z_dim,), dtype=np.float32)
    slab_meta: list[dict[str, float]] = []

    for i in range(len(slab_edges) - 1):
        z0, z1 = int(slab_edges[i]), int(slab_edges[i + 1])
        m = materials[int(rng.integers(0, len(materials)))]
        spr_line[z0:z1] = np.float32(m.spr)
        slab_meta.append(
            {
                "slab_idx": i,
                "z_start": z0,
                "z_end": z1,
                "thickness_mm": float((z1 - z0) * spacing_mm[0]),
                "material": m.name,
                "spr": float(m.spr),
            }
        )

    spr = np.broadcast_to(spr_line[:, None, None], (z_dim, y_dim, x_dim)).astype(np.float32).copy()
    return spr, slab_meta


def generate_case(
    case_id: str,
    stage: StageConfig,
    shape: tuple[int, int, int],
    spacing_mm: np.ndarray,
    beam_axis: int,
    lateral_sigma_mm: float,
    beam_radius_mm: float,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict]:
    if beam_axis != 0:
        raise ValueError("This synthetic slab generator currently supports beam_axis=0 only")

    z_dim, y_dim, x_dim = shape

    n_slabs = int(rng.integers(stage.num_slabs_min, stage.num_slabs_max + 1))
    spr, slab_meta = build_spr_slab_volume(shape, spacing_mm, n_slabs, stage.materials, rng)

    e0_mev = float(rng.uniform(stage.energy_min, stage.energy_max))

    # Water-equivalent depth (WED)
    spr_line = spr[:, 0, 0]
    wed_mm = np.cumsum(spr_line * spacing_mm[0]).astype(np.float32)

    range_w_mm = water_range_mm_from_energy(e0_mev)
    sigma_peak_mm = float(max(4.0, 0.04 * range_w_mm + 2.0))

    peak = np.exp(-0.5 * ((wed_mm - range_w_mm) / sigma_peak_mm) ** 2)
    plateau = 0.18 * np.exp(-wed_mm / max(1e-3, 1.3 * range_w_mm))
    distal_arg = (wed_mm - (range_w_mm + 3.0)) / 2.2
    distal_arg = np.clip(distal_arg, -60.0, 60.0)
    distal_cut = 1.0 / (1.0 + np.exp(distal_arg))

    depth_profile = (peak + plateau) * distal_cut
    depth_profile = depth_profile / max(float(depth_profile.max()), 1e-8)

    yy = (np.arange(y_dim, dtype=np.float32) - (y_dim - 1) / 2.0) * spacing_mm[1]
    xx = (np.arange(x_dim, dtype=np.float32) - (x_dim - 1) / 2.0) * spacing_mm[2]
    yy_grid, xx_grid = np.meshgrid(yy, xx, indexing="ij")
    r2 = yy_grid**2 + xx_grid**2

    sigma_lat_mm = lateral_sigma_mm + 0.015 * wed_mm
    lateral = np.exp(-0.5 * r2[None, :, :] / np.maximum(sigma_lat_mm[:, None, None] ** 2, 1e-6))

    d_high = (depth_profile[:, None, None] * lateral).astype(np.float32)

    beam_mask_2d = (np.sqrt(r2) <= beam_radius_mm).astype(np.float32)
    beam_mask = np.broadcast_to(beam_mask_2d[None, :, :], d_high.shape).astype(np.float32).copy()

    floor = 0.03 * float(d_high.max())
    noise_std = stage.noise_rel * np.maximum(d_high, floor)
    noise = rng.normal(loc=0.0, scale=noise_std).astype(np.float32)
    d_low = np.clip(d_high + noise, a_min=0.0, a_max=None).astype(np.float32)

    payload = {
        "d_low": d_low,
        "spr": spr.astype(np.float32),
        "d_high": d_high,
        "e0_mev": np.asarray(e0_mev, dtype=np.float32),
        "spacing_mm": spacing_mm.astype(np.float32),
        "beam_axis": np.asarray(beam_axis, dtype=np.int64),
        "case_id": np.asarray(case_id),
        "beam_mask": beam_mask,
    }

    meta = {
        "case_id": case_id,
        "e0_mev": e0_mev,
        "noise_rel": stage.noise_rel,
        "n_slabs": n_slabs,
        "materials": slab_meta,
        "range_w_mm": range_w_mm,
    }

    return payload, meta


def write_manifest(rows: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["npz_path", "stage", "case_id"])
        writer.writeheader()
        writer.writerows(rows)


def split_rows(rows: list[dict[str, str]], seed: int, train_frac: float, val_frac: float) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    rows_shuffled = [rows[i] for i in idx]

    n = len(rows_shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(max(1, n_train), max(1, n - 2)) if n >= 3 else max(1, n)
    n_val = min(max(1, n_val), max(1, n - n_train - 1)) if n - n_train >= 2 else max(0, n - n_train)

    train_rows = rows_shuffled[:n_train]
    val_rows = rows_shuffled[n_train:n_train + n_val]
    test_rows = rows_shuffled[n_train + n_val:]

    if len(test_rows) == 0 and len(val_rows) > 1:
        test_rows.append(val_rows.pop())

    return train_rows, val_rows, test_rows


def load_config(path: Path) -> tuple[int, tuple[int, int, int], np.ndarray, int, float, float, list[StageConfig]]:
    cfg = json.loads(path.read_text(encoding="utf-8"))

    seed = int(cfg.get("seed", 42))
    shape = tuple(int(v) for v in cfg.get("shape", [96, 96, 96]))
    spacing_mm = np.asarray(cfg.get("spacing_mm", [2.0, 2.0, 2.0]), dtype=np.float32)
    beam_axis = int(cfg.get("beam_axis", 0))
    lateral_sigma_mm = float(cfg.get("lateral_sigma_mm", 18.0))
    beam_radius_mm = float(cfg.get("beam_radius_mm", 42.0))

    stages: list[StageConfig] = []
    for st in cfg.get("stages", []):
        mats = [Material(name=str(m["name"]), spr=float(m["spr"])) for m in st["materials"]]
        e_range = st.get("energy_range_mev", [70.0, 225.0])
        stages.append(
            StageConfig(
                name=str(st["name"]),
                n_cases=int(st["n_cases"]),
                energy_min=float(e_range[0]),
                energy_max=float(e_range[1]),
                noise_rel=float(st["noise_rel"]),
                num_slabs_min=int(st["num_slabs_min"]),
                num_slabs_max=int(st["num_slabs_max"]),
                materials=mats,
            )
        )

    if not stages:
        raise ValueError("No stages configured in curriculum JSON")

    return seed, shape, spacing_mm, beam_axis, lateral_sigma_mm, beam_radius_mm, stages


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate slab-only curriculum dataset (no CT)")
    parser.add_argument("--config", type=Path, default=Path("configs/curriculum/slabs_curriculum.json"))
    parser.add_argument("--out-root", type=Path, default=Path("data/curriculum/slabs"))
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    seed, shape, spacing_mm, beam_axis, lateral_sigma_mm, beam_radius_mm, stages = load_config(args.config)

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    global_train: list[dict[str, str]] = []
    global_val: list[dict[str, str]] = []
    global_test: list[dict[str, str]] = []
    summary: dict[str, dict] = {}

    for stage_idx, stage in enumerate(stages):
        stage_seed = seed + stage_idx * 10_000
        rng = np.random.default_rng(stage_seed)

        stage_dir = out_root / stage.name
        npz_dir = stage_dir / "npz"
        npz_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, str]] = []
        metas: list[dict] = []

        for i in range(stage.n_cases):
            case_id = f"{stage.name}_case_{i:05d}"
            payload, meta = generate_case(
                case_id=case_id,
                stage=stage,
                shape=shape,
                spacing_mm=spacing_mm,
                beam_axis=beam_axis,
                lateral_sigma_mm=lateral_sigma_mm,
                beam_radius_mm=beam_radius_mm,
                rng=rng,
            )
            npz_path = npz_dir / f"{case_id}.npz"
            np.savez_compressed(npz_path, **payload)

            row = {
                "npz_path": str(npz_path.resolve()),
                "stage": stage.name,
                "case_id": case_id,
            }
            rows.append(row)
            metas.append(meta)

        train_rows, val_rows, test_rows = split_rows(rows, seed=stage_seed + 1, train_frac=args.train_frac, val_frac=args.val_frac)
        write_manifest(rows, stage_dir / "manifest_all.csv")
        write_manifest(train_rows, stage_dir / "manifest_train.csv")
        write_manifest(val_rows, stage_dir / "manifest_val.csv")
        write_manifest(test_rows, stage_dir / "manifest_test.csv")

        global_train.extend(train_rows)
        global_val.extend(val_rows)
        global_test.extend(test_rows)

        summary[stage.name] = {
            "n_cases": len(rows),
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "n_test": len(test_rows),
            "noise_rel": stage.noise_rel,
            "energy_range_mev": [stage.energy_min, stage.energy_max],
            "num_slabs": [stage.num_slabs_min, stage.num_slabs_max],
            "materials": [{"name": m.name, "spr": m.spr} for m in stage.materials],
            "meta_preview": metas[:3],
        }

        print(f"[{stage.name}] cases={len(rows)} train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    # Combined manifests, useful for ablations
    write_manifest(global_train, out_root / "manifest_train_all_stages.csv")
    write_manifest(global_val, out_root / "manifest_val_all_stages.csv")
    write_manifest(global_test, out_root / "manifest_test_all_stages.csv")

    report = {
        "config": str(args.config.resolve()),
        "out_root": str(out_root.resolve()),
        "shape": list(shape),
        "spacing_mm": spacing_mm.tolist(),
        "beam_axis": beam_axis,
        "stages": summary,
    }
    (out_root / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nDone. Curriculum dataset generated at: {out_root}")
    print(f"Report: {out_root / 'dataset_report.json'}")


if __name__ == "__main__":
    main()
