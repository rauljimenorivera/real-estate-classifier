# Real-Estate Image Classifier

Clasificacion automatica de imagenes inmobiliarias (15 clases) con transfer learning, tracking en W&B, API FastAPI y front en Streamlit.

## Que hace este repo

- Prepara datos reproducibles (`data/raw` -> `data/processed/{train,val,test}`).
- Entrena modelos de vision con `timm` y registra metricas en Weights & Biases.
- Ejecuta inferencia por API (`/predict`) y desde Streamlit.
- Mantiene un flujo de experimentacion con configs por GPU y sweep de hiperparametros.

## Estructura principal

- `src/prepare_data.py`: entrypoint para crear splits reproducibles.
- `src/train.py`: entrypoint de entrenamiento (W&B, AMP, guardado de checkpoints).
- `src/real_estate_ml/data`: dataset, transforms, dataloaders, split prep.
- `src/real_estate_ml/models`: construccion de backbones con `timm`.
- `src/real_estate_ml/training`: loop de train/eval y metricas.
- `src/real_estate_ml/inference`: predictor compartido por API/Streamlit.
- `api/main.py`: servicio FastAPI.
- `app/app.py`: app Streamlit.
- `configs/base_config.yaml`: baseline general.
- `configs/experiments/*.yaml`: configuraciones concretas (GTX/RTX/big_model).
- `configs/sweeps/big_sweep.yaml`: busqueda automatica en W&B.
- `notebooks/01..04`: flujo de EDA, training, seleccion y test de inferencia.

## Setup

```bash
uv venv --python 3.11
.venv\Scripts\activate
uv sync
```

## Flujo rapido (end-to-end)

1) Preparar datos:

```bash
uv run python src/prepare_data.py
```

2) Entrenar un modelo (ejemplo GTX):

```bash
uv run python -u src/train.py --config configs/experiments/gtx1650.yaml --wandb online
```

3) Levantar API:

```bash
uv run uvicorn api.main:app --reload --port 8000
```

4) Levantar Streamlit:

```bash
uv run streamlit run app/app.py --server.port 8501
```

5) Probar:
- Swagger: `http://127.0.0.1:8000/docs`
- App: `http://localhost:8501`

## Sweep (muchas combinaciones automaticas)

Crear sweep:

```bash
uv run wandb sweep configs/sweeps/big_sweep.yaml
```

Lanzar agente:

```bash
uv run wandb agent --count 20 <ENTITY>/<PROJECT>/<SWEEP_ID>
```

## Gestion de checkpoints (importante)

Cada run guarda su mejor modelo en:

- `artifacts/runs/<run_id>/best_model.pth`

Ademas, existe un "mejor global":

- `artifacts/best_model.pth`
- `artifacts/best_model.json`

`best_model.pth` solo se actualiza si el run actual supera el `best_val_macro_f1` global.

## Criterio de seleccion de modelo final

El modelo final debe elegirse con criterio explicito, no solo por intuicion. Recomendacion:

- Objetivo principal: maximizar `val/macro_f1`.
- Criterios de desempate: estabilidad entre runs, coste de entrenamiento y coste de inferencia.
- Trazabilidad: dejar referenciado en W&B el `run_id` ganador y su configuracion.

## API contract (FastAPI)

Endpoint principal:

- `POST /predict`
  - **Input:** `multipart/form-data` con campo `file` (imagen `jpg/jpeg/png`).
  - **Output 200:** JSON con `filename` y `predictions` (top clases con probabilidad).
  - **Errores:**
    - `400`: fichero no valido / payload no imagen.
    - `503`: modelo no cargado (checkpoint ausente o no inicializado).

Estado servicio:

- `GET /health` -> `status` y `model_loaded`.
- Documentacion OpenAPI: `/docs`.

## Notebooks (orden recomendado)

1. `notebooks/01_data_prep_eda.ipynb`
2. `notebooks/02_training_experiments.ipynb`
3. `notebooks/03_model_selection_eval.ipynb`
4. `notebooks/04_inference_api_test.ipynb`

## Clases

Bedroom, Coast, Forest, Highway, Industrial, Inside city, Kitchen, Living room, Mountain, Office, Open country, Store, Street, Suburb, Tall building

## Entrega y documentacion

- Plantilla informe: [`docs/report_outline.md`](docs/report_outline.md)
- Coordinacion de trabajo: [`tareas.md`](tareas.md)
- No commitear datos/modelos/runs locales (`data/`, `artifacts/`, `wandb/` en `.gitignore`)

## Final deliverable checklist

- [ ] Repositorio Git publico.
- [ ] URL del repositorio incluida en README/informe.
- [ ] URL del proyecto W&B incluida en README/informe.
- [ ] Invitaciones enviadas en W&B a `agascon@comillas.edu` y `rkramer@comillas.edu`.
- [ ] API con Swagger operativo y documentado.
- [ ] Streamlit conectada a la API.
- [ ] Historial de experimentacion trazable y modelo final justificado.
- [ ] Informe final (max 6 paginas, sin portada) completado.

