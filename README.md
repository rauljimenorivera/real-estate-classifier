# Real-Estate Image Classifier

Automatic classification of real-estate images using transfer learning.

Built for a simulated marketplace use case (Idealista/Zillow style).

## Project scope (phase 1 — MVP local)

- Reproducible data preparation (`training/validation` -> `train/val/test`)
- Transfer-learning training pipeline with Weights & Biases tracking
- FastAPI inference service with Swagger docs
- Streamlit front-end connected to the API

## Roadmap (resto del enunciado / entrega final)

La fase 1 cubre el stack funcionando en local. Para cumplir el proyecto completo falta:

**Fase 2 — Experimentación y modelo**

- Profundizar en W&B: diseño de experimentos, búsqueda de hiperparámetros, comparación de backbones (p. ej. EfficientNet vs ResNet).
- Criterio explícito de selección del modelo final (p. ej. macro-F1 en validación + coste/tiempo).
- Historial trazable: runs comparables, artifacts del mejor modelo, tags o convención de nombres de run.

**Fase 3 — Despliegue “production-ready”**

- API FastAPI **pública** con Swagger accesible (inputs, outputs, errores documentados).
- App Streamlit **pública** conectada a esa API (no solo `localhost`).
- Artefactos de despliegue reproducibles (p. ej. variables de entorno, URL del modelo, instrucciones de arranque).
- *Referencia habitual:* API en Render (u otro PaaS) + Streamlit Community Cloud; ajustad si el curso indica otro proveedor.

**Fase 4 — Informe y entrega formal**

- Informe técnico (máx. **6 páginas**, sin portada) con: contexto de negocio (marketplace), arquitectura, enfoque de modelado, proceso de experimentación W&B, **métricas por clase** (precision/recall/F1), interpretación de matriz de confusión, conclusiones y recomendaciones de negocio.
- Enlaces en documentación: **repositorio Git público** y **proyecto W&B**.
- W&B: invitar a `agascon@comillas.edu` y `rkramer@comillas.edu`.

Plantilla de secciones del informe: [`docs/report_outline.md`](docs/report_outline.md). Coordinación del equipo: [`tareas.md`](tareas.md).

## Classes

Bedroom, Coast, Forest, Highway, Industrial, Inside city, Kitchen, Living room, Mountain, Office, Open country, Store, Street, Suburb, Tall building

## Repo structure

- `src/real_estate_ml/data`: dataset and split preparation
- `src/real_estate_ml/models`: transfer-learning model builders
- `src/real_estate_ml/training`: training/evaluation engine
- `src/real_estate_ml/inference`: shared prediction utilities
- `src/train.py`: training entrypoint
- `api/main.py`: FastAPI inference API
- `app/app.py`: Streamlit UI
- `configs/base_config.yaml`: base config for training and data
- `configs/experiments/gtx1650.yaml`: fast config for Raul
- `configs/experiments/rtx3070.yaml`: full config for Natalia
- `notebooks/01_data_prep_eda.ipynb`: data prep and class distribution checks
- `notebooks/02_training_experiments.ipynb`: interactive training + W&B
- `notebooks/03_model_selection_eval.ipynb`: model comparison and test evaluation
- `notebooks/04_inference_api_test.ipynb`: local inference + API test
- `tareas.md`: coordinación del equipo, pendientes del enunciado y reparto de tareas

## Setup

```bash
uv venv --python 3.11
.venv\Scripts\activate
uv sync
```

## Quickstart

1) Prepare processed splits:

```bash
uv run python src/prepare_data.py
```

2) Train + track in W&B:

```bash
uv run python src/train.py --config configs/base_config.yaml
```

Para ajustar batch/epochs según GPU, usa por ejemplo `configs/experiments/gtx1650.yaml` o `configs/experiments/rtx3070.yaml` en lugar de `base_config.yaml`.

3) Run API:

```bash
uv run uvicorn api.main:app --reload
```

4) Run Streamlit:

```bash
uv run streamlit run app/app.py
```

## Notebook-first workflow

1) Run notebooks in order:
- `01_data_prep_eda.ipynb`
- `02_training_experiments.ipynb`
- `03_model_selection_eval.ipynb`
- `04_inference_api_test.ipynb`

2) Promote stable logic to `src/real_estate_ml/*`.

3) Keep API and Streamlit consuming `artifacts/best_model.pth`.

## Team split (4 people)

Detalle y checklist de entrega: ver [`tareas.md`](tareas.md).

| Persona | Rol principal |
|--------|----------------|
| Raúl (GTX) | Notebooks `01`, `04`; `gtx1650.yaml`; EDA e integración API/Streamlit |
| Natalia (RTX) | Notebooks `02`, `03`; `rtx3070.yaml`; experimentos largos y modelo final en W&B |
| Sofía | Diseño de experimentos W&B, análisis por clase, UX Streamlit y mensajes API |
| Marta | Informe (6 pág.), README operativo, capturas Swagger, enlaces y entrega formal |

## Team workflow

- Commits pequeños y frecuentes en `main`; avisar si dos personas editan el mismo notebook a la vez.
- Runs “oficiales” referenciados en W&B (nombre del run + config usada).
- Datos y `artifacts/` locales: no commitear (ver `.gitignore`).

## Delivery checklist

### Fase 1 (MVP local)

- [ ] Setup reproducible en máquina limpia (`uv sync`, README)
- [ ] Runs en W&B con config y artifacts trazables
- [ ] FastAPI con Swagger en `/docs`
- [ ] Streamlit conectada a la API y predicciones correctas en local

### Entrega final (enunciado)

- [ ] Codebase reproducible con instrucciones claras
- [ ] API pública con documentación OpenAPI/Swagger y manejo de errores
- [ ] Streamlit pública enlazada a la API (end-to-end)
- [ ] Historial de experimentación W&B trazable y modelo final justificado
- [ ] Informe (≤6 páginas): negocio, arquitectura, modelado, W&B, métricas por clase, matriz de confusión, API, conclusiones
- [ ] Repo Git **público** + enlace al proyecto W&B en README o informe
- [ ] Invitaciones W&B a `agascon@comillas.edu` y `rkramer@comillas.edu`

