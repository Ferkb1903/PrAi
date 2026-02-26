from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_energy_list(txt: str) -> list[int]:
    return [int(x.strip()) for x in txt.split(",") if x.strip()]


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
    parser = argparse.ArgumentParser(description="Generate cluster jobs from manifest_sim_ready.csv")
    parser.add_argument("--manifest", type=Path, default=Path("data/ct_cases_by_case/manifest_sim_ready.csv"))
    parser.add_argument("--jobs-root", type=Path, default=Path("cluster_jobs"))
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root used as working directory inside generated jobs",
    )
    parser.add_argument("--scheduler", choices=["slurm", "pbs"], default="slurm")
    parser.add_argument("--cases-limit", type=int, default=0, help="0 means all cases")

    parser.add_argument("--energies", type=str, default="90,120,150,180,210")
    parser.add_argument("--low-events", type=int, default=50000)
    parser.add_argument("--high-events", type=int, default=10000000)
    parser.add_argument("--low-seed", type=int, default=101)
    parser.add_argument("--high-seed", type=int, default=202)

    parser.add_argument("--source-mode", choices=["point", "beamlet"], default="beamlet")
    parser.add_argument("--beamlet-nx", type=int, default=5)
    parser.add_argument("--beamlet-ny", type=int, default=5)
    parser.add_argument("--beamlet-pitch-mm", type=float, default=6.0)
    parser.add_argument("--source-z-cm", type=float, default=-30.0)

    parser.add_argument("--resample-mm", type=float, default=2.0)
    parser.add_argument("--flip-z", action="store_true")

    parser.add_argument("--walltime", type=str, default="24:00:00")
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--mem-gb", type=int, default=16)

    parser.add_argument("--hu-map-json", type=Path, default=Path("configs/hu_material_map_v1.json"))
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    energies = parse_energy_list(args.energies)
    project_root = args.project_root.expanduser().resolve()

    jobs_root = args.jobs_root
    jobs_root.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    selected = rows[: args.cases_limit] if args.cases_limit and args.cases_limit > 0 else rows

    submit_lines = ["#!/usr/bin/env bash", "set -euo pipefail"]

    count = 0
    for row in selected:
        if str(row.get("ready_for_sim", "0")) != "1":
            continue

        case_id = row["case_id"]
        input_type = row["input_type"]
        input_path = row["input_path"]

        case_dir = jobs_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        pre_mhd = case_dir / "ct_preprocessed.mhd"
        run_sh = case_dir / "run_case.sh"

        header = job_header(
            scheduler=args.scheduler,
            job_name=f"prai_{case_id[:18]}",
            walltime=args.walltime,
            cpus=args.cpus,
            mem_gb=args.mem_gb,
        )

        convert_cmd = (
            f"./.venv/bin/python scripts/preprocess_ct_for_gate.py --input \"{input_path}\" "
            f"--input-type {input_type} --output-mhd \"{pre_mhd}\" --spacing-mm {args.resample_mm}"
        )
        if args.flip_z:
            convert_cmd += " --flip-z"

        lines = [
            header,
            f"cd \"{project_root}\"",
            convert_cmd,
            f"mkdir -p outputs/cluster_runs/{case_id}/low outputs/cluster_runs/{case_id}/high",
        ]

        for e in energies:
            low_out = f"outputs/cluster_runs/{case_id}/low/E{e}"
            high_out = f"outputs/cluster_runs/{case_id}/high/E{e}"
            lines.append(
                "bash scripts/run_gate_voxelized_shared_env.sh "
                f"\"{pre_mhd}\" \"{low_out}\" {e} {args.low_events} {args.low_seed} "
                f"{args.source_mode} {args.beamlet_nx} {args.beamlet_ny} {args.beamlet_pitch_mm} {args.source_z_cm} \"{args.hu_map_json}\""
            )
            lines.append(
                "bash scripts/run_gate_voxelized_shared_env.sh "
                f"\"{pre_mhd}\" \"{high_out}\" {e} {args.high_events} {args.high_seed} "
                f"{args.source_mode} {args.beamlet_nx} {args.beamlet_ny} {args.beamlet_pitch_mm} {args.source_z_cm} \"{args.hu_map_json}\""
            )

        run_sh.write_text("\n".join(lines) + "\n", encoding="utf-8")
        run_sh.chmod(0o755)

        if args.scheduler == "slurm":
            submit_lines.append(f"sbatch \"{run_sh}\"")
        else:
            submit_lines.append(f"qsub \"{run_sh}\"")

        count += 1

    submit_script = jobs_root / "submit_all.sh"
    submit_script.write_text("\n".join(submit_lines) + "\n", encoding="utf-8")
    submit_script.chmod(0o755)

    print(f"Jobs generated: {count}")
    print(f"Jobs root: {jobs_root}")
    print(f"Submit script: {submit_script}")


if __name__ == "__main__":
    main()
