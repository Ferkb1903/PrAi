from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path


def parse_energy_list(txt: str) -> list[int]:
    return [int(x.strip()) for x in txt.split(",") if x.strip()]


def make_spot_list(spots_per_energy: int, spot_radius_mm: float, seed_key: str) -> list[tuple[float, float]]:
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    spots: list[tuple[float, float]] = []
    for _ in range(spots_per_energy):
        r = spot_radius_mm * (rng.random() ** 0.5)
        theta = 2.0 * 3.141592653589793 * rng.random()
        x = r * __import__("math").cos(theta)
        y = r * __import__("math").sin(theta)
        spots.append((x, y))
    return spots


def job_header(scheduler: str, job_name: str, walltime: str, cpus: int, mem_gb: int) -> str:
    if scheduler == "slurm":
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                f"#SBATCH --job-name={job_name}",
                f"#SBATCH --time={walltime}",
                f"#SBATCH --cpus-per-task={cpus}",
                f"#SBATCH --mem={mem_gb}G",
                "#SBATCH --output=%x_%j.out",
                "set -euo pipefail",
            ]
        )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            f"#PBS -N {job_name}",
            f"#PBS -l walltime={walltime}",
            f"#PBS -l select=1:ncpus={cpus}:mem={mem_gb}gb",
            "#PBS -j oe",
            "set -euo pipefail",
            "cd ${PBS_O_WORKDIR:-$PWD}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate spot-campaign jobs (case x energy x random spots)")
    parser.add_argument("--manifest", type=Path, default=Path("data/ct_cases_by_case/manifest_sim_ready.csv"))
    parser.add_argument("--jobs-root", type=Path, default=Path("cluster_jobs/spot_campaign"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scheduler", choices=["slurm", "pbs"], default="slurm")

    parser.add_argument("--case-prefix", type=str, default="lung_")
    parser.add_argument("--cases-limit", type=int, default=10)

    parser.add_argument("--energies", type=str, default="70,110,150,190,230")
    parser.add_argument("--spots-per-energy", type=int, default=40)
    parser.add_argument("--launch-fraction", type=float, default=0.5)
    parser.add_argument("--spot-radius-mm", type=float, default=60.0)

    parser.add_argument("--low-events", type=int, default=50000)
    parser.add_argument("--high-events", type=int, default=1000000)
    parser.add_argument("--low-seed", type=int, default=101)
    parser.add_argument("--high-seed", type=int, default=202)

    parser.add_argument("--source-z-cm", type=float, default=-30.0)
    parser.add_argument("--resample-mm", type=float, default=2.0)
    parser.add_argument("--hu-map-json", type=Path, default=Path("configs/hu_material_map_v1.json"))

    parser.add_argument("--walltime", type=str, default="08:00:00")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--mem-gb", type=int, default=8)
    parser.add_argument("--array-concurrency", type=int, default=50)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    energies = parse_energy_list(args.energies)
    launch_spots = max(1, int(round(args.spots_per_energy * args.launch_fraction)))

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    selected = [
        r
        for r in rows
        if str(r.get("ready_for_sim", "0")) == "1" and r["case_id"].startswith(args.case_prefix)
    ]
    selected = selected[: args.cases_limit] if args.cases_limit > 0 else selected

    jobs_root = args.jobs_root
    jobs_root.mkdir(parents=True, exist_ok=True)
    project_root = args.project_root.resolve()

    submit_lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    pair_index_rows: list[dict[str, str]] = []

    count_jobs = 0
    for row in selected:
        case_id = row["case_id"]
        input_type = row["input_type"]
        input_path = row["input_path"]

        for energy in energies:
            case_energy_dir = jobs_root / case_id / f"E{energy}"
            case_energy_dir.mkdir(parents=True, exist_ok=True)
            run_sh = case_energy_dir / "run_case_energy.sh"

            header = job_header(
                scheduler=args.scheduler,
                job_name=f"prai_spot_{case_id[:10]}_{energy}",
                walltime=args.walltime,
                cpus=args.cpus,
                mem_gb=args.mem_gb,
            )

            pre_mhd = jobs_root / "_preprocessed" / case_id / "ct_preprocessed.mhd"
            pre_mhd.parent.mkdir(parents=True, exist_ok=True)

            lines = [
                header,
                f'cd "{project_root}"',
                'TMPDIR="${TMPDIR:-$PWD/.tmp_slurm}"',
                'mkdir -p "$TMPDIR"',
                'export TMPDIR',
                'PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"',
                'if [[ ! -x "$PYTHON_BIN" ]]; then',
                '  PYTHON_BIN="$(command -v python3 || command -v python || true)"',
                'fi',
                'if [[ -z "$PYTHON_BIN" ]]; then',
                '  echo "No se encontró Python ejecutable" >&2',
                '  exit 127',
                'fi',
                f'if [[ ! -f "{pre_mhd}" ]]; then',
                f'  "$PYTHON_BIN" scripts/preprocess_ct_for_gate.py --input "{input_path}" --input-type {input_type} --output-mhd "{pre_mhd}" --spacing-mm {args.resample_mm}',
                'fi',
            ]

            spots = make_spot_list(args.spots_per_energy, args.spot_radius_mm, f"{case_id}|{energy}")
            spots = spots[:launch_spots]

            for spot_idx, (sx, sy) in enumerate(spots):
                low_out = f"outputs/spot_campaign/{case_id}/E{energy}/spot_{spot_idx:03d}/low"
                high_out = f"outputs/spot_campaign/{case_id}/E{energy}/spot_{spot_idx:03d}/high"
                lines.append(f"mkdir -p {low_out} {high_out}")
                lines.append(
                    f"SOURCE_X_MM={sx:.3f} SOURCE_Y_MM={sy:.3f} PYTHON_BIN=\"$PYTHON_BIN\" bash scripts/run_gate_voxelized_shared_env.sh "
                    f"\"{pre_mhd}\" \"{low_out}\" {energy} {args.low_events} {args.low_seed} point 1 1 1.0 {args.source_z_cm} \"{args.hu_map_json}\""
                )
                lines.append(
                    f"SOURCE_X_MM={sx:.3f} SOURCE_Y_MM={sy:.3f} PYTHON_BIN=\"$PYTHON_BIN\" bash scripts/run_gate_voxelized_shared_env.sh "
                    f"\"{pre_mhd}\" \"{high_out}\" {energy} {args.high_events} {args.high_seed} point 1 1 1.0 {args.source_z_cm} \"{args.hu_map_json}\""
                )
                pair_index_rows.append(
                    {
                        "case_id": case_id,
                        "input_type": input_type,
                        "input_path": input_path,
                        "energy_mev": str(energy),
                        "spot_idx": str(spot_idx),
                        "spot_x_mm": f"{sx:.3f}",
                        "spot_y_mm": f"{sy:.3f}",
                        "pre_mhd": str(pre_mhd),
                        "low_out": low_out,
                        "high_out": high_out,
                    }
                )

            run_sh.write_text("\n".join(lines) + "\n", encoding="utf-8")
            run_sh.chmod(0o755)

            if args.scheduler == "slurm":
                submit_lines.append(f"sbatch \"{run_sh}\"")
            else:
                submit_lines.append(f"qsub \"{run_sh}\"")
            count_jobs += 1

    submit_script = jobs_root / "submit_all.sh"
    submit_script.write_text("\n".join(submit_lines) + "\n", encoding="utf-8")
    submit_script.chmod(0o755)

    pair_index_csv = jobs_root / "pair_index.csv"
    if pair_index_rows:
        with pair_index_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "case_id",
                    "input_type",
                    "input_path",
                    "energy_mev",
                    "spot_idx",
                    "spot_x_mm",
                    "spot_y_mm",
                    "pre_mhd",
                    "low_out",
                    "high_out",
                ],
            )
            w.writeheader()
            w.writerows(pair_index_rows)

    if args.scheduler == "slurm" and pair_index_rows:
        run_pair_array = jobs_root / "run_pair_array.sh"
        run_pair_array.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    f"#SBATCH --job-name=prai_pair",
                    f"#SBATCH --time={args.walltime}",
                    f"#SBATCH --cpus-per-task={args.cpus}",
                    f"#SBATCH --mem={args.mem_gb}G",
                    f"#SBATCH --output={jobs_root}/logs/%x_%A_%a.out",
                    "set -euo pipefail",
                    f'cd "{project_root}"',
                    'TMPDIR="${TMPDIR:-$PWD/.tmp_slurm}"',
                    'mkdir -p "$TMPDIR"',
                    'export TMPDIR',
                    'PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"',
                    'if [[ ! -x "$PYTHON_BIN" ]]; then',
                    '  PYTHON_BIN="$(command -v python3 || command -v python || true)"',
                    'fi',
                    'if [[ -z "$PYTHON_BIN" ]]; then',
                    '  echo "No se encontró Python ejecutable" >&2',
                    '  exit 127',
                    'fi',
                    f'mkdir -p "{jobs_root}/logs"',
                    f'PAIR_CSV="{pair_index_csv}"',
                    'ROW_NUM=$((SLURM_ARRAY_TASK_ID + 2))',
                    'ROW=$(sed -n "${ROW_NUM}p" "$PAIR_CSV")',
                    'if [[ -z "$ROW" ]]; then',
                    '  echo "No row for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2',
                    '  exit 2',
                    'fi',
                    'IFS="," read -r case_id input_type input_path energy_mev spot_idx spot_x_mm spot_y_mm pre_mhd low_out high_out <<< "$ROW"',
                    'if [[ ! -f "$pre_mhd" ]]; then',
                    '  mkdir -p "$(dirname "$pre_mhd")"',
                    '  "$PYTHON_BIN" scripts/preprocess_ct_for_gate.py --input "$input_path" --input-type "$input_type" --output-mhd "$pre_mhd" --spacing-mm 2.0',
                    'fi',
                    'mkdir -p "$low_out" "$high_out"',
                    f'SOURCE_X_MM="$spot_x_mm" SOURCE_Y_MM="$spot_y_mm" PYTHON_BIN="$PYTHON_BIN" bash scripts/run_gate_voxelized_shared_env.sh "$pre_mhd" "$low_out" "$energy_mev" {args.low_events} {args.low_seed} point 1 1 1.0 {args.source_z_cm} "{args.hu_map_json}"',
                    f'SOURCE_X_MM="$spot_x_mm" SOURCE_Y_MM="$spot_y_mm" PYTHON_BIN="$PYTHON_BIN" bash scripts/run_gate_voxelized_shared_env.sh "$pre_mhd" "$high_out" "$energy_mev" {args.high_events} {args.high_seed} point 1 1 1.0 {args.source_z_cm} "{args.hu_map_json}"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        run_pair_array.chmod(0o755)

        submit_array = jobs_root / "submit_array_50.sh"
        submit_array.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    f'N=$(( $(wc -l < "{pair_index_csv}") - 1 ))',
                    'if (( N <= 0 )); then echo "No pairs to submit"; exit 1; fi',
                    f'sbatch --array=0-$((N-1))%{args.array_concurrency} "{run_pair_array}"',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        submit_array.chmod(0o755)

    total_pairs = len(pair_index_rows)
    print(f"Selected cases: {len(selected)}")
    print(f"Energies: {len(energies)} | spots per energy: {args.spots_per_energy} | launch spots: {launch_spots}")
    print(f"Target pairs in this launch: {total_pairs}")
    print(f"Jobs generated (case x energy): {count_jobs}")
    print(f"Jobs root: {jobs_root}")
    print(f"Submit script: {submit_script}")
    print(f"Pair index: {pair_index_csv}")
    if args.scheduler == "slurm" and pair_index_rows:
        print(f"Array runner: {jobs_root / 'run_pair_array.sh'}")
        print(f"Array submit: {jobs_root / 'submit_array_50.sh'}")


if __name__ == "__main__":
    main()
