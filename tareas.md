# Tareas y estado del proyecto

Estado actual: base tecnica lista (datos, entrenamiento, W&B, API y Streamlit).  
Pendiente principal: **experimentacion final + informe**.

## Hecho

- Setup reproducible con `uv`.
- Preparacion de datos reproducible (`train/val/test`).
- Entrenamiento con tracking en W&B (incluye sweep).
- Guardado robusto de checkpoints:
  - mejor por run en `artifacts/runs/<run_id>/best_model.pth`
  - mejor global en `artifacts/best_model.pth` + `artifacts/best_model.json`
- API FastAPI con `/health`, `/predict` y Swagger.
- App Streamlit conectada a la API.
- Notebooks consolidados: `01` a `04` (sin duplicados walkthrough).

## Pendiente (lo importante ahora)

1. **Experimentar mas modelos en W&B** (backbones e hiperparametros) para elegir modelo final con criterio.
2. **Ejecutar notebooks de analisis** para extraer evidencia:
   - metricas por clase
   - matriz de confusion
   - comparativa de runs
3. **Redactar informe final** (max 6 paginas) con resultados y conclusiones.

## Reparto propuesto final

| Persona | Tarea principal |
|--------|------------------|
| **Raul** | Mantener pipeline estable y seguir lanzando experimentos/sweeps desde GTX. |
| **Natalia** | Ejecutar notebooks y/o nuevos modelos (especialmente training/evaluacion). |
| **Ruben** | Ejecutar notebooks y/o nuevos modelos; apoyar comparativa de resultados en W&B. |
| **Marta** | Redaccion del informe final y coordinacion de la entrega. |

## Checklist de cierre

- [ ] Modelo final elegido y justificado (W&B + metrica objetivo).
- [ ] Evidencia lista para informe (tablas/graficas por clase + confusion matrix).
- [ ] Informe terminado y revisado en equipo.
- [ ] README y enlaces finales (repo + proyecto W&B) validados.
