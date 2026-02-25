# PrAI — Dataset Generation (CT + OpenGATE)

Repositorio mínimo para la etapa actual: **generación de dataset**.

## Quickstart (6 comandos)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/organize_ct_cases.py
python scripts/build_sim_ready_manifest.py
python scripts/generate_cluster_jobs.py --manifest data/ct_cases_by_case/manifest_sim_ready.csv --jobs-root cluster_jobs/full --scheduler slurm
```

Luego envía manualmente en clúster:

```bash
bash cluster_jobs/full/submit_all.sh
```

## Qué hace

- Unifica CT por caso en `data/ct_cases_by_case`.
- Construye un manifest de entrada para simulación.
- Genera jobs SLURM/PBS para low/high por energía.

## Scripts clave

- `scripts/organize_ct_cases.py`
- `scripts/build_sim_ready_manifest.py`
- `scripts/preprocess_ct_for_gate.py`
- `scripts/gate_voxelized_ct_experiment.py`
- `scripts/generate_cluster_jobs.py`
- `scripts/run_gate_voxelized_shared_env.sh`

## Requisitos

- Python + `numpy` + `SimpleITK`
- OpenGATE/Geant4 disponible en nodo de ejecución
