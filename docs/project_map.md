# Mapa del Proyecto

Este documento resume **qué hace cada carpeta**.

## `src/config`

Parámetros por defecto de entrenamiento, modelo, datos y pérdidas.

## `src/data`

- `schema.py`: contrato de llaves y validación estructural por caso NPZ.
- `io_npz.py`: lectura/escritura NPZ + metadatos.
- `preprocess.py`: transformaciones previas (normalización, BEV crop, canal energía).
- `dataset.py`: clase `Dataset` para PyTorch.

## `src/models`

- `unet3d.py`: arquitectura 3D U-Net mínima.
- `model_factory.py`: punto único para construir modelo según config.

## `src/losses`

Pérdidas combinadas físicas: MSE + peso distal (Bragg) + gradiente espacial.

## `src/metrics`

Métricas de evaluación MVP:

- gamma simplificado,
- DVH por máscara,
- error distal en rango.

## `src/train.py`

Entrenamiento y guardado de checkpoints.

## `src/eval.py`

Carga de checkpoint y cálculo de métricas en validación/test.
