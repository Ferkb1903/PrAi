# Contrato de Datos (NPZ por caso)

Cada archivo NPZ representa un caso 3D con malla isotrópica o conocida.

## Llaves obligatorias

- `d_low`: dosis baja estadística, `float32`, shape `(D, H, W)`.
- `spr`: mapa SPR, `float32`, shape `(D, H, W)`.
- `d_high`: dosis alta estadística (ground truth), `float32`, shape `(D, H, W)`.
- `e0_mev`: energía inicial del haz en MeV, escalar `float32`.
- `spacing_mm`: spacing voxel en mm, shape `(3,)`, `float32`.
- `beam_axis`: eje principal del haz, entero en `{0,1,2}`.
- `case_id`: identificador de caso, string.

## Convenciones

- Las tres matrices de volumen deben tener **idéntica shape**.
- Se recomienda orientación consistente por dataset.
- `e0_mev` se transforma a canal constante durante preprocesado.
- Si se usa aprendizaje residual: objetivo `delta_d = d_high - d_low`.

## División de datos

Se recomienda:

- `data/demo/train/*.npz`
- `data/demo/val/*.npz`
- `data/demo/test/*.npz`

## Validación mínima

1. Shapes compatibles.
2. Sin NaN/Inf.
3. `spacing_mm > 0`.
4. `beam_axis` válido.
