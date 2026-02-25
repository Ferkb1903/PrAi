# Guía Archivo por Archivo

## README.md
- **Qué resuelve:** visión general del proyecto y ejecución rápida.
- **Entrada/salida:** comandos para generar datos demo, entrenar y evaluar.
- **Límites:** no describe implementación interna detallada.

## requirements.txt
- **Qué resuelve:** dependencias mínimas del MVP.
- **Entrada/salida:** lista de paquetes para instalar.
- **Límites:** no fija versiones exactas para producción clínica.

## src/config/defaults.py
- **Qué resuelve:** configuración centralizada del pipeline.
- **Entrada/salida:** constantes de rutas, entrenamiento, modelo y pérdida.
- **Límites:** no gestiona múltiples perfiles de experimento.

## src/data/schema.py
- **Qué resuelve:** contrato estricto para cada NPZ.
- **Entrada/salida:** valida llaves, tipos, shapes y retorna `CaseData`.
- **Límites:** no transforma datos; solo valida/estructura.

## src/data/io_npz.py
- **Qué resuelve:** lectura/escritura de casos NPZ.
- **Entrada/salida:** carga `CaseData` desde disco y guarda casos.
- **Límites:** no hace preprocesado ni augmentations.

## src/data/preprocess.py
- **Qué resuelve:** transformaciones previas al entrenamiento.
- **Entrada/salida:** normalización de dosis, mapa de energía y crop 3D.
- **Límites:** no aplica augmentación aleatoria avanzada.

## src/data/dataset.py
- **Qué resuelve:** dataset PyTorch para entrenamiento/evaluación.
- **Entrada/salida:** produce `x` de 3 canales y objetivo residual/directo.
- **Límites:** no balancea casos ni hace muestreo por anatomía.

## src/models/unet3d.py
- **Qué resuelve:** arquitectura base de denoising 3D.
- **Entrada/salida:** tensor `(B, 3, D, H, W)` a `(B, 1, D, H, W)`.
- **Límites:** no incluye atención ni variantes avanzadas.

## src/models/model_factory.py
- **Qué resuelve:** punto único de creación del modelo.
- **Entrada/salida:** instancia del modelo según `defaults.py`.
- **Límites:** de momento soporta una sola arquitectura.

## src/losses/losses.py
- **Qué resuelve:** pérdida total física compuesta.
- **Entrada/salida:** calcula MSE, Bragg, gradiente y total.
- **Límites:** peso distal simplificado por fracción del eje de haz.

## src/metrics/gamma.py
- **Qué resuelve:** gamma simplificado para control inicial.
- **Entrada/salida:** pass rate para 3% y 2% (sin búsqueda espacial completa).
- **Límites:** no reemplaza gamma clínico completo 3D.

## src/metrics/dvh.py
- **Qué resuelve:** cálculo básico de DVH desde máscara binaria.
- **Entrada/salida:** curva dosis-volumen acumulada.
- **Límites:** requiere máscaras ya preparadas.

## src/metrics/distal_range.py
- **Qué resuelve:** error distal en mm.
- **Entrada/salida:** estima rango distal por perfil medio y umbral relativo.
- **Límites:** aproximación simplificada al criterio clínico.

## src/train.py
- **Qué resuelve:** entrenamiento end-to-end y checkpoints.
- **Entrada/salida:** recorre train/val, calcula pérdidas y guarda pesos.
- **Límites:** logging básico; no incluye tracking externo.

## src/eval.py
- **Qué resuelve:** evaluación con checkpoint más reciente.
- **Entrada/salida:** reporta gamma simplificado, MAE y error distal.
- **Límites:** no exporta reportes extensos ni figuras.

## scripts/prepare_demo_dataset.py
- **Qué resuelve:** dataset sintético mínimo para probar pipeline.
- **Entrada/salida:** crea NPZ en `data/demo/{train,val,test}`.
- **Límites:** datos sintéticos, no clínicos.

## scripts/run_train_demo.sh
- **Qué resuelve:** comando corto para smoke test de entrenamiento.
- **Entrada/salida:** ejecuta `python -m src.train --dry-run`.
- **Límites:** no parametriza hiperparámetros.

## scripts/run_eval_demo.sh
- **Qué resuelve:** comando corto para evaluar checkpoint reciente.
- **Entrada/salida:** ejecuta `python -m src.eval`.
- **Límites:** asume que ya existe checkpoint.
