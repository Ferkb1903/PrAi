# PrAI — Estado del proyecto (2026-02-26)

## 1) Objetivo actual
Este repositorio está enfocado en **generación de dataset de dosis** para protonterapia usando:
- CT clínicos
- Preprocesamiento a voxel grid uniforme
- Simulación OpenGATE/Geant4
- Producción de pares `low` / `high` por spot y energía

No se está priorizando entrenamiento en este punto; el foco es **completitud y calidad del dataset simulado**.

---

## 2) Pipeline implementado
1. Organización de CT por caso (`scripts/organize_ct_cases.py`).
2. Construcción de manifest de casos simulables (`scripts/build_sim_ready_manifest.py`).
3. Preprocesamiento CT a `.mhd/.raw` para GATE (`scripts/preprocess_ct_for_gate.py`).
4. Simulación voxelizada por energía y número de eventos (`scripts/gate_voxelized_ct_experiment.py`).
5. Wrapper de ejecución robusto en clúster (`scripts/run_gate_voxelized_shared_env.sh`).
6. Generación de jobs tradicionales y de campaña por spots:
   - `scripts/generate_cluster_jobs.py`
   - `scripts/generate_spot_campaign_jobs.py`

---

## 3) Campaña de spots (estado operativo)
- Diseño objetivo: múltiples casos de colorectal, múltiples energías clínicas, múltiples spots por energía.
- Producción esperada por par: carpeta `low` + carpeta `high`, cada una con `dose_voxelized_ct_edep.mhd/.raw`.
- Se opera con arrays SLURM y control de concurrencia (`%N`).

### Hallazgo clave en depuración
Se detectó que muchas carpetas de salida se crearon como:
- `high\r` (con carriage return oculto),
no como `high`.

Esto provocó falsos negativos en validaciones automáticas y comandos de búsqueda estándar.

---

## 4) Estado técnico confirmado en este punto
- El pipeline **sí genera datos válidos** en `low` y `high`.
- Se validó integridad de archivos de muestra (`.mhd` y `.raw` consistentes en dimensiones y tamaño).
- Se confirmó que parte de la discrepancia de conteo provenía de:
  1. rutas raíz distintas entre entornos,
  2. nombres de carpeta con `\r` en `high`.

### Estado de verificación actual
- Ya existe script de depuración para inspección de rutas y campañas:
  - `scripts/debug_spot_campaign_paths.py`
- El script permite:
  - resumen global (`strict_low/high`, `deep_low/high`),
  - foco por caso/energía/spot,
  - auditoría completa por un solo caso (`--single-case`),
  - cruce opcional con `pair_index.csv`.

---

## 5) Cambios recientes relevantes
1. **Robustez SLURM/TMPDIR**
   - Ajustes para exportar `TMPDIR` no vacío y reducir warnings de `slurmstepd`.
2. **Depuración de campaña**
   - Nuevo script `debug_spot_campaign_paths.py` para diagnosticar desajustes de ruta y nombres.
3. **Publicación continua**
   - Scripts actualizados se suben a GitHub para `git pull` inmediato en clúster.

---

## 6) Riesgos abiertos
1. **Nombres de carpeta con caracteres invisibles** (`\r`) pueden reaparecer si alguna etapa vuelve a introducirlos.
2. **Validación cruzada por índice** puede fallar si `pair_index.csv` y raíz de outputs no corresponden a la misma corrida.
3. **Re-ejecuciones** deben evitar duplicar trabajo (relanzar solo faltantes cuando aplique).

---

## 7) Próximas acciones recomendadas
1. Ejecutar renombre/normalización de rutas para eliminar `\r` en todo `outputs/spot_campaign`.
2. Re-auditar campaña por carpeta real (no solo por CSV) y generar resumen por caso.
3. Completar faltantes `high`/`low` con relanzamiento selectivo por índice.
4. Consolidar manifiesto final “training-ready” únicamente con pares completos (`low+high`).

---

## 8) Comandos de referencia
### Auditoría de un caso completo
```bash
python scripts/debug_spot_campaign_paths.py \
  --root /lustre/home/acastaneda/Fernando/PrAi \
  --single-case colorectal_StageII-Colorectal-CT-003
```

### Auditoría con cruce de CSV
```bash
python scripts/debug_spot_campaign_paths.py \
  --root /lustre/home/acastaneda/Fernando/PrAi \
  --pair-csv /lustre/home/acastaneda/Fernando/PrAi/cluster_jobs/spot_campaign_3060/pair_index.csv \
  --rows 40
```

### Conteos por estructura real de carpetas
```bash
ROOT=/lustre/home/acastaneda/Fernando/PrAi/outputs/spot_campaign
find "$ROOT" -type d -path '*/E*/spot_*/low'  | wc -l
find "$ROOT" -type d -path '*/E*/spot_*/high' | wc -l
find "$ROOT" -type f -path '*/E*/spot_*/low/dose_voxelized_ct_edep.raw'  | wc -l
find "$ROOT" -type f -path '*/E*/spot_*/high/dose_voxelized_ct_edep.raw' | wc -l
```
