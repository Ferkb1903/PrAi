# PrAI — Dataset Generation Pipeline (CT + OpenGATE)

Repositorio mínimo para la **etapa actual: generación de dataset**.

Incluye únicamente:
- ingesta de CT (DICOM/NRRD),
- organización por caso,
- preprocesamiento (2 mm isotrópico + convención BEV),
- simulación de beamlets en OpenGATE,
- generación de jobs de clúster.

## Estructura mínima

- `scripts/organize_ct_cases.py`: unifica CT por caso en `data/ct_cases_by_case`.
- `scripts/build_sim_ready_manifest.py`: genera `manifest_sim_ready.csv` por caso.
- `scripts/preprocess_ct_for_gate.py`: DICOM/MHD -> MHD preprocesado.
- `scripts/gate_voxelized_ct_experiment.py`: simulación OpenGATE voxelizada.
- `scripts/run_gate_voxelized_shared_env.sh`: wrapper de ejecución local/cluster.
- `scripts/generate_cluster_jobs.py`: crea jobs SLURM/PBS desde manifest.
- `scripts/ingest_tcia_nrrd_ct.py`: convierte NRRD -> MHD con manifest.
- `scripts/download_tcia_hn_ct_subset.py`: descarga subset de CT H&N sin clonar repo completo.
- `configs/hu_material_map_v1.json`: mapeo HU->material para GATE.

## Flujo recomendado

1. Organizar casos:

```bash
python scripts/organize_ct_cases.py
python scripts/build_sim_ready_manifest.py
```

2. Generar jobs para clúster (ejemplo SLURM):

```bash
python scripts/generate_cluster_jobs.py \
	--manifest data/ct_cases_by_case/manifest_sim_ready.csv \
	--jobs-root cluster_jobs/full \
	--scheduler slurm
```

3. Enviar (manual):

```bash
bash cluster_jobs/full/submit_all.sh
```

## Dependencias

- Python + `SimpleITK` + `numpy`.
- OpenGATE/Geant4 instalado en el entorno de ejecución.
