# PrAI — Deep Learning para Reducción de Varianza en Monte Carlo de Protones

Este repositorio implementa un **MVP educativo** para aprender una corrección de dosis 3D:

- Entrada: `(D_low, SPR, E0)`
- Salida: `D_high` o residual `ΔD = D_high - D_low`

El objetivo es acercar la calidad de simulaciones Monte Carlo de alta estadística usando simulaciones de baja estadística + red neuronal 3D.

## Filosofía del repositorio

- Estructura mínima y legible.
- Cada archivo tiene una única responsabilidad.
- Configuración simple en Python (sin frameworks de configuración).
- PyTorch puro para entender el flujo completo.

## Estructura

- `docs/`: guías y contrato de datos.
- `src/config/`: constantes y parámetros por defecto.
- `src/data/`: esquema, I/O NPZ, preprocesado, dataset.
- `src/models/`: arquitectura 3D U-Net y factoría de modelos.
- `src/losses/`: pérdidas (MSE + Bragg + gradiente).
- `src/metrics/`: métricas MVP (gamma simplificado, DVH, error distal).
- `src/train.py`: entrenamiento.
- `src/eval.py`: evaluación.
- `scripts/`: utilidades demo para preparar datos y ejecutar entrenamiento/eval.

## Quickstart

1. Crear entorno e instalar dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Generar dataset demo sintético:

```bash
python scripts/prepare_demo_dataset.py
```

3. Entrenar (smoke test):

```bash
bash scripts/run_train_demo.sh
```

4. Evaluar:

```bash
bash scripts/run_eval_demo.sh
```

## Nota

Este MVP prioriza claridad y trazabilidad. Las métricas y el gamma están implementados en versión simple para validación inicial y se pueden reemplazar por implementaciones clínicas completas en una siguiente fase.
