# Requerimientos de Datos — Etapa Siguiente

Este documento define exactamente qué necesitas conseguir para arrancar generación masiva de pares `D_low / D_high` y entrenamiento.

## 1) Paquete mínimo por paciente (obligatorio)

- CT DICOM completo (serie usada para planificación).
- (Opcional pero muy recomendado) RTSTRUCT con:
  - CTV/PTV
  - OAR principales (médula/vejiga/recto/intestino según caso)
- Identificador anónimo del paciente (sin PHI).

## 2) Metadata obligatoria por caso

Cada caso debe tener estos campos (una fila por caso en CSV):

- `case_id`
- `site` (prostate, HN, thorax, pelvis, etc.)
- `ct_series_uid`
- `spacing_x_mm`, `spacing_y_mm`, `spacing_z_mm`
- `beam_axis` (0,1,2; para este proyecto usamos +z inicialmente)
- `hu_to_spr_curve_id`
- `notes`

## 3) Configuración de simulación (estándar inicial)

Para cada combinación caso + energía:

- Fuente: `beamlet` (pencil, monoenergética)
- Grid spots inicial: `5x5`
- Pitch inicial: `6.0 mm`
- `source_z_cm = -30.0`
- Energías iniciales: `90, 120, 150, 180, 210 MeV`

## 4) Estadística Monte Carlo por combinación

- `D_low`: `1e5` a `5e5` eventos
- `D_high`: `1e6` a `5e6` eventos
- Semillas:
  - al menos 2 para `D_low`
  - al menos 1 para `D_high`

## 5) Qué nos entregas en disco por combinación

- `dose_voxelized_ct_edep.mhd`
- `dose_voxelized_ct_edep.raw`
- archivo de metadatos de corrida (JSON o CSV) con:
  - energía
  - n_events
  - seed
  - source_mode
  - beamlet_nx/ny
  - beamlet_pitch_mm
  - source_z_cm

## 6) Criterio para empezar entrenamiento

Podemos empezar cuando tengamos como mínimo:

- 1 paciente completo
- 5 energías
- por energía: 1 `D_low` + 1 `D_high`

Ideal para baseline más robusto:

- 3–5 pacientes completos con el mismo protocolo.
