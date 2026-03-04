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

## Entrenamiento (baseline MVP)

1) Convertir pares `low/high` de `spot_campaign` a NPZ:

```bash
python scripts/build_spot_campaign_npz.py \
	--spot-root outputs/spot_campaign \
	--out-dir data/training_npz/spot_campaign
```

2) Entrenar modelo baseline 3D (predicción residual):

```bash
python scripts/train_npz_baseline.py \
	--npz-dir data/training_npz/spot_campaign \
	--epochs 30 \
	--batch-size 2 \
	--num-workers 4
```

Checkpoints y métricas quedan en `checkpoints/train_npz_baseline/<timestamp>/`.

## Pipeline recomendado antes de entrenar en MI210

1) Preparar tensores optimizados con QC + HU→SPR + split por paciente:

```bash
python scripts/prepare_training_tensors.py \
	--pair-index-csv cluster_jobs/spot_campaign/pair_index.csv \
	--out-dir data/training_npz/spot_campaign_v2 \
	--qc-report data/training_npz/qc_spot_campaign.csv \
	--manifest-all data/training_npz/manifest_all.csv \
	--manifest-train data/training_npz/manifest_train.csv \
	--manifest-val data/training_npz/manifest_val.csv \
	--manifest-test data/training_npz/manifest_test.csv
```

2) Entrenar Residual 3D U-Net:

```bash
python scripts/train_residual_unet3d.py \
	--manifest-train data/training_npz/manifest_train.csv \
	--manifest-val data/training_npz/manifest_val.csv \
	--manifest-test data/training_npz/manifest_test.csv \
	--epochs 80 \
	--batch-size 2 \
	--num-workers 8 \
	--base-channels 24
```

La red usa aprendizaje residual: `D_pred = D_low + Net(D_low, SPR, E0, BeamMask)`.

## Requisitos

- Python + `numpy` + `SimpleITK`
- OpenGATE/Geant4 disponible en nodo de ejecución

## Curriculum Learning (Slabs sintéticos, sin CT)

Primer paso de reestructuración: entrenar la red en un entorno controlado de slabs de materiales, antes de volver a CT real.

1) Generar dataset de curriculum por etapas (`easy -> medium -> hard`):

```bash
python scripts/generate_slab_curriculum_dataset.py \
	--config configs/curriculum/slabs_curriculum.json \
	--out-root data/curriculum/slabs
```

2) Entrenar por curriculum (continúa cada etapa desde `best.pt` de la etapa anterior):

```bash
bash scripts/train_curriculum_slabs.sh
```

Variables útiles para ajustar corrida:

```bash
DATA_ROOT=/ruta/a/data/curriculum/slabs \
EPOCHS_STAGE1=20 EPOCHS_STAGE2=25 EPOCHS_STAGE3=30 \
BATCH_SIZE=2 NUM_WORKERS=4 \
bash scripts/train_curriculum_slabs.sh
```
