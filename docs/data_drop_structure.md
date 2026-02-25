# Estructura recomendada de entrega de datos

Usa esta estructura para evitar caos en ingestión:

- data/incoming_raw/
  - CASE_0001/
    - ct_dicom/
      - *.dcm
    - rtstruct/
      - rtstruct.dcm
    - simulations/
      - E090/
        - low_seed42/
          - dose_voxelized_ct_edep.mhd
          - dose_voxelized_ct_edep.raw
          - run_meta.json
        - high_seed42/
          - dose_voxelized_ct_edep.mhd
          - dose_voxelized_ct_edep.raw
          - run_meta.json
      - E120/
      - E150/
      - E180/
      - E210/
  - CASE_0002/

## Regla de naming

- Caso: `CASE_XXXX`
- Energía: `E090`, `E120`, etc.
- Tipo estadístico: `low_seedNN` / `high_seedNN`

## Notas

- No mezclar distintos protocolos de fuente dentro del mismo lote.
- Si cambias `beamlet_nx/ny` o `pitch`, crea un lote separado.
