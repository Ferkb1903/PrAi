# CT a Geometría Voxelizada en OpenGATE (Python)

Este flujo crea una geometría `Image` de OpenGATE a partir de tu serie DICOM CT.

## 1) Convertir DICOM CT a MHD

```bash
/home/fer/fer/ProtonAI/PrAI/.venv/bin/python scripts/convert_dicom_ct_to_mhd.py \
  --dicom-dir "manifest-1771465991435/NaF PROSTATE/NaF-PROSTATE-01-0002/10-13-2005-NA-PET F-18 NaF Bone Scan-66881/3.000000-LD CT 450FOV-16426" \
  --output-mhd data/raw/ct_prostate_0002/ct.mhd
```

Salida esperada:

- `data/raw/ct_prostate_0002/ct.mhd`
- `data/raw/ct_prostate_0002/ct.raw`

## 2) Construir/ejecutar simulación OpenGATE con volumen voxelizado

```bash
/home/fer/fer/ProtonAI/PrAI/.venv/bin/python scripts/gate_voxelized_ct_experiment.py \
  --ct-mhd data/raw/ct_prostate_0002/ct.mhd \
  --output-dir outputs/gate_ct_test \
  --energy-mev 150.0
```

Si quieres reutilizar el entorno compartido de `../ProtonAI` y los datos Geant4 ya instalados:

```bash
bash scripts/run_gate_voxelized_shared_env.sh \
  data/raw/ct_prostate_0002/ct.mhd \
  outputs/gate_ct_test \
  150.0
```

## Notas importantes

- El mapeo HU→material en el script es inicial y debe calibrarse para uso dosimétrico serio.
- Este paso es para validar geometría voxelizada y pipeline de simulación.
- El ground truth para entrenamiento seguirá siendo dosis MC alta estadística (`D_high`) con la misma geometría.
