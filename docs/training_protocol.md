# Protocolo de Entrenamiento (MVP)

## Objetivo

Aprender `f_theta(D_low, SPR, E0)` para aproximar `D_high`.

## Modo recomendado

Entrenamiento residual:

- salida de red: `delta_d_pred`
- reconstrucción: `d_pred = d_low + delta_d_pred`

## Pérdida total

`L = L_MSE + λ1 * L_Bragg + λ2 * L_Gradient`

- `L_MSE`: error voxel-wise.
- `L_Bragg`: mayor peso en región distal sobre eje de haz.
- `L_Gradient`: preservación de gradientes espaciales (fall-off).

## Validación MVP

- Error medio absoluto y RMSE.
- Gamma simplificado (3%/3mm y 2%/2mm).
- DVH por máscara de estructura (si existe).
- Error distal de rango (`ΔR`).
