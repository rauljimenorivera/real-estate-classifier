# Tareas y estado del proyecto

Mensaje de coordinación para el equipo (rama `main`).

## Resumen de lo que ya está hecho en el repo

- **Datos:** script/notebook para generar splits reproducibles `train` / `val` / `test` desde `data/raw/dataset/training` y `validation` (sin duplicar al repetir preparación).
- **Entrenamiento:** pipeline con transfer learning (`timm`), métricas por clase y registro en **Weights & Biases** (entidad de equipo `202529987-universidad-pontificia-comillas`, proyecto `real-estate-classifier`). Hay un run de referencia con buen resultado en test (macro-F1 ~0.93); el detalle está en W&B, no en Git.
- **Inferencia local:** FastAPI con `/health`, `/predict` y Swagger en `/docs`.
- **Front:** Streamlit en `http://localhost:8501` llamando a la API.
- **Trabajo en equipo:** notebooks numerados `01`–`04` y configs por GPU `configs/experiments/gtx1650.yaml` y `rtx3070.yaml` (README actualizado).

**Importante:** el modelo entrenado (`artifacts/best_model.pth`) y las carpetas de datos procesados **no van al repo** (están en `.gitignore`). Cada uno genera localmente con los mismos pasos.

## Cómo ponerse al día

1. `git pull` (desde `main`).
2. `uv venv --python 3.11` y `.\.venv\Scripts\activate` si hace falta.
3. `uv sync`.
4. Seguir el **README** (preparar datos → entrenar o notebooks → API → Streamlit).

## Qué queda por hacer (siguiente fase del enunciado)

- **Experimentación W&B en profundidad:** más runs, búsqueda de hiperparámetros, comparación de backbones, criterio explícito de modelo final (y trazabilidad en el proyecto W&B).
- **Métricas e informe:** tabla por clase (precision/recall/F1), matriz de confusión interpretada, conclusiones y recomendaciones de negocio (marketplace inmobiliario).
- **Despliegue “production-ready”:** API y app accesibles públicamente (cuando abráis la fase de despliegue del curso).
- **Entrega formal:** repo público, enlaces Git + W&B en documentación, invitaciones en W&B a `agascon@comillas.edu` y `rkramer@comillas.edu`.

## Reparto propuesto (podemos ajustarlo en grupo)

| Persona | Enfoque principal |
|--------|-------------------|
| **Raúl (GTX)** | Notebooks `01` y `04`, config `gtx1650.yaml`, EDA, pruebas API/Streamlit y estabilidad del flujo local. |
| **Natalia (RTX)** | Notebooks `02` y `03`, config `rtx3070.yaml`, entrenamientos largos, tuning y elección del modelo final en W&B. |
| **Sofía** | Diseño de experimentos en W&B, análisis de clases/desbalanceo, mejoras de producto (UX Streamlit, mensajes API, checklist de calidad). |
| **Marta (PC más justo)** | Documentación e informe (6 páginas), README operativo, capturas Swagger, enlaces y checklist de entrega; puede coordinar redacción con datos que aportéis desde W&B/notebooks. |
