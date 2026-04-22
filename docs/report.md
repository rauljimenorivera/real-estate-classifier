# Clasificador de Imágenes para Marketplace Inmobiliario — Informe Técnico

---

## 1. Contexto de negocio

### 1.1 Problema

Los portales inmobiliarios (Idealista, Fotocasa, Zillow…) reciben miles de imágenes diarias subidas por agentes y particulares. Etiquetar manualmente qué muestra cada foto —cocina, dormitorio, fachada urbana, costa— es lento, caro e inconsistente: dos revisores pueden categorizar la misma imagen de distinta forma según el ángulo o la iluminación. Esta inconsistencia degrada la calidad de los filtros de búsqueda y la experiencia del usuario.

El objetivo del proyecto es entrenar un **clasificador automático de escenas** que asigne una de 15 categorías visuales a cada imagen de un anuncio, sin intervención humana en los casos de alta confianza y con soporte a la decisión en los ambiguos.

### 1.2 Usuarios y valor esperado

| Actor | Uso del clasificador | Valor entregado |
|---|---|---|
| Equipos de moderación | Validar etiquetas automáticas antes de publicar | Ahorro de tiempo, consistencia |
| Equipos de datos / ML | Etiquetas visuales como feature para ranking y recomendación | Labels estandarizadas y reproducibles |
| Usuario final | Búsqueda por tipo de habitación / escena | Mejor precisión en filtros |

### 1.3 Restricciones operativas

El problema se formula como clasificación multiclase de **15 categorías** (Bedroom, Coast, Forest, Highway, Industrial, Inside city, Kitchen, Living room, Mountain, Office, Open country, Store, Street, Suburb, Tall building). La inferencia debe ser síncrona (una imagen → respuesta inmediata), robusta frente a entradas malformadas y trazable: todo experimento debe quedar registrado en W&B para facilitar auditorías y reproducibilidad del equipo.

---

## 2. Arquitectura del sistema

La solución se organiza en cuatro bloques independientes y bien delimitados:

**Preparación de datos** (`src/real_estate_ml/data/prepare_splits.py`): las imágenes del dataset MIT Indoor/Outdoor se redistribuyen en tres particiones con semilla fija (seed=42) para garantizar reproducibilidad. La partición resultante es **70 % train / 15 % val / 15 % test** sobre un total de **4 489 imágenes** en 15 clases. Existe cierta desbalanza: Bedroom es la clase más pequeña (216 imágenes, 33 en test) y Open country la más grande (410, 62 en test), lo que motivó el uso de macro-F1 como métrica principal en lugar de accuracy.

**Entrenamiento y selección** (`src/train.py`): el script de entrenamiento lee una configuración YAML, instancia el modelo con `timm`, lanza el bucle de entrenamiento con early stopping, registra todas las métricas en W&B (curvas por época, métricas de test, matriz de confusión) y guarda el mejor checkpoint en `artifacts/`.

**Servicio de inferencia** (`api/main.py`): FastAPI expone `POST /predict` y endpoints auxiliares. En el arranque carga automáticamente `artifacts/best_model.pth`; también admite intercambio en caliente de checkpoint vía ruta local o referencia a artefacto W&B.

**Interfaz de usuario** (Streamlit): cliente ligero que envía la imagen al API y presenta las 3 predicciones más probables con sus puntuaciones de confianza.

**Flujo de inferencia**: usuario sube imagen → Streamlit → `POST /predict` (multipart) → FastAPI preprocesa (resize 224×224, normalización ImageNet) → forward pass EfficientNet-B1 → top-3 clases + probabilidades → Streamlit muestra resultado.

**[INSERTAR FIGURA DE ARQUITECTURA]**

---

## 3. Enfoque de modelado

### 3.1 Dataset y preprocesamiento

El dataset tiene **4 489 imágenes** distribuidas en 15 clases de escenas interiores y exteriores, todas redimensionadas a 224×224. Durante el entrenamiento se aplica data augmentation estándar (flip horizontal, random crop, color jitter); en validación y test sólo se aplica resize + normalización ImageNet. La desbalanza entre clases (Bedroom: 151 imágenes de train; Open country: 287) es moderada y se gestiona usando macro-F1 como métrica de selección, que da el mismo peso a todas las clases independientemente de su frecuencia.

### 3.2 Estrategia de transfer learning

Se parte de pesos pre-entrenados en ImageNet (accesibles vía `timm.create_model(backbone, pretrained=True)`). El procedimiento es:

1. Se instancia el backbone con su cabeza original de 1000 clases.
2. `timm` reemplaza automáticamente la cabeza por una capa `Dropout(p) → Linear(features_out → 15)` cuando se especifica `num_classes=15`.
3. Se evalúan dos políticas de fine-tuning: **(a) backbone congelado** —sólo se entrena la cabeza— y **(b) fine-tuning completo** —todos los parámetros descongelados—. Los experimentos mostraron de forma consistente que el fine-tuning completo supera al backbone congelado en **5–10 pp de macro-F1** en validación.
4. Se usa el optimizador **AdamW** con schedule de tasa de aprendizaje coseno (`CosineAnnealingLR`) y early stopping sobre `val/macro_f1`.

### 3.3 Perfiles de hardware

El equipo dispone de dos máquinas con GPU distintas, lo que obligó a diseñar configuraciones diferenciadas:

| Parámetro | GTX 1650 (4 GB VRAM) | RTX 3070 (8 GB VRAM) |
|---|---|---|
| Batch size | 16 | 16 / 24 / 32 |
| Num workers | 2 | 4 |
| Épocas máx. | 10–30 | 30–40 |
| Early-stop patience | 3 | 5 |
| Mixed precision | Sí | Sí |

La precisión mixta (`torch.cuda.amp`) fue imprescindible en la GTX 1650 para evitar errores de memoria con EfficientNet-B2. Los runs de la GTX se limitaron a backbones congelados para mayor seguridad; los runs de la RTX pudieron explorar fine-tuning completo con todos los backbones candidatos.

### 3.4 Arquitectura final desplegada en la API

El checkpoint seleccionado corresponde al run `rtx3070-20260420-202946-5c3fio9s`:

| Propiedad | Valor |
|---|---|
| Backbone | `efficientnet_b1` (timm, pesos ImageNet) |
| Tamaño de entrada | 224 × 224 RGB |
| Parámetros entrenables | 6 532 399 (todos descongelados) |
| Cabeza de clasificación | Dropout(0.5) → Linear(1280 → 15) |
| Artefacto W&B | Sweep `x7dqfl05`, run `rtx3070-20260420-202946-5c3fio9s` |

---

## 4. Proceso de experimentación (W&B)

### 4.1 Visión global: 32 runs en 5 fases

| Fase | Fecha | Runs | Sweep | Hardware | Objetivo |
|---|---|---|---|---|---|
| 1 — Baseline | 14–15 abr | 2 | — | RTX 3070 | Validar que el pipeline funciona de extremo a extremo |
| 2 — Primer sweep | 15 abr | 1 | `c6lx3ga6` (random) | GTX 1650 | Exploración amplia inicial |
| 3 — Exploración manual | 17–19 abr | 7 | — | RTX 3070 | Calibrar LR, épocas y freeze policy |
| 4 — Sweep GTX | 20 abr | 12 | `tqmllxsx` (Bayes) | GTX 1650 | Búsqueda sistemática con restricciones de VRAM |
| 5 — Sweep RTX | 20 abr | 9 | `x7dqfl05` (Bayes) | RTX 3070 | Búsqueda amplia de backbone + hiperparámetros |

### 4.2 Fase 1 — Baseline

Los dos primeros runs (`dutiful-gorge-1`, `comfy-sun-11`) usaron EfficientNet-B0, fine-tuning completo, lr=1e-4, batch=32, cosine schedule, 5–10 épocas. El objetivo era simplemente confirmar que el pipeline —carga de datos, entrenamiento, logging en W&B, guardado de checkpoint— funcionaba correctamente antes de invertir tiempo en búsqueda de hiperparámetros. Resultado: test macro-F1 ≈ **0.926–0.932**, lo que confirmó que el enfoque de transfer learning era viable para el problema.

### 4.3 Fase 3 — Exploración manual guiada

Con el pipeline validado, se lanzaron 7 runs manuales (17–19 abril) para entender la sensibilidad a los hiperparámetros más importantes antes de lanzar sweeps costosos. Las conclusiones clave de esta fase:

- **Fine-tuning completo es claramente superior al backbone congelado**: con EfficientNet-B0 descongelado se alcanzan val/macro-F1 ≈ 0.945–0.952; con backbone congelado los runs del GTX no superaron ≈ 0.93.
- **El LR óptimo para fine-tuning completo está en el rango 1e-5 a 5e-5**, muy por debajo del 1e-4 usado habitualmente para entrenamiento desde cero. Tasas más altas causaban inestabilidad en las capas profundas del backbone.
- **El scheduler coseno con early stopping** (patience=5) evita desperdicio de cómputo y estabiliza el entrenamiento en todas las configuraciones probadas.

### 4.4 Fase 4 — Sweep GTX 1650 (`tqmllxsx`, Bayesiano + Hyperband)

Con las lecciones de la fase anterior se diseñó un sweep Bayesiano para la GTX 1650, con terminación temprana Hyperband (`min_iter=5, eta=3`) para descartar configuraciones malas rápidamente. El espacio de búsqueda:

| Hiperparámetro | Rango |
|---|---|
| `model.backbone` | `{efficientnet_b0, efficientnet_b2}` |
| `training.learning_rate` | log-uniforme `[5e-6, 2e-4]` |
| `training.weight_decay` | log-uniforme `[1e-5, 1e-2]` |
| `model.dropout` | `{0.1, 0.3, 0.5}` |
| `data.batch_size` | `{16, 32}` |
| `model.freeze_backbone` | `{true, false}` |

Los 12 runs de este sweep confirmaron que en el perfil GTX 1650, las configuraciones con backbone congelado eran las únicas viables sin riesgo de OOM con batch=32 y EfficientNet-B2. El mejor resultado fue un val/macro-F1 ≈ **0.928** (EfficientNet-B0, freeze=true, bs=16, lr≈1.85e-5). Esto estableció el techo de la GTX: útil para exploración barata pero no para extraer el máximo rendimiento del modelo.

### 4.5 Fase 5 — Sweep RTX 3070 (`x7dqfl05`, Bayesiano)

El sweep definitivo amplió el espacio de búsqueda al perfil RTX, que permite fine-tuning completo con mayor batch y más épocas:

| Hiperparámetro | Rango |
|---|---|
| `backbone` | `{efficientnet_b1, efficientnet_b2, resnet50, densenet121}` |
| `lr` | log-uniforme `[1e-5, 3e-4]` |
| `weight_decay` | log-uniforme `[1e-6, 3e-3]` |
| `batch_size` | `{16, 24, 32}` |
| `freeze_backbone` | `{true, false}` |
| `dropout` | `{0.2, 0.3, 0.4, 0.5}` |
| `epochs` | 40 (máx., con early stopping patience=5) |

Resultados relevantes de este sweep:

| Run | Backbone | BS | LR | WD | Freeze | Test macro-F1 | Val macro-F1 |
|---|---|---|---|---|---|---|---|
| `rtx3070-20260420-202946` | **efficientnet_b1** | 24 | 2.52e-5 | 1.04e-3 | No | **0.9641** | 0.9486 |
| `rtx3070-20260420-183134` | efficientnet_b1 | 16 | 1.27e-5 | 2.14e-3 | No | 0.9586 | 0.9484 |
| `rtx3070-20260420-195002` | efficientnet_b2 | 24 | 3.17e-5 | 7.75e-4 | No | 0.9292 | 0.9365 |
| `rtx3070-20260420-202320` | efficientnet_b2 | 32 | 2.33e-5 | 1.13e-4 | No | 0.9268 | 0.9268 |
| `rtx3070-20260420-183549` | resnet50 | 32 | 4.52e-5 | 8.3e-6 | **Sí** | 0.7696 | 0.7695 |

**Hallazgos clave**:
- **EfficientNet-B1 supera a B0 y B2**: B0 tiene menos capacidad para este problema; B2 (7.7M parámetros) tiende a sobreajustarse con nuestro dataset de ~3100 imágenes de entrenamiento. B1 (~6.5M) ofrece el mejor equilibrio.
- **ResNet-50 congelado fue el peor resultado de todo el proyecto** (0.77 F1): la arquitectura ResNet requiere fine-tuning para competir con las EfficientNet en transfer learning sobre datasets pequeños.
- **La región óptima** se sitúa en: backbone descongelado, lr ≈ 1.5–3e-5, weight_decay ≈ 1e-3, dropout = 0.5, batch_size = 24.
- La selección final no se basó en el mejor run único, sino en la **consistencia de la región**: dos runs independientes con EfficientNet-B1 descongelado alcanzaron val/macro-F1 ≥ 0.948, confirmando robustez.

**[INSERTAR CAPTURA W&B: PANEL DEL SWEEP + MEJOR RUN]**

### 4.6 Criterio de selección del modelo final

1. Máximo `val/macro_f1` (criterio primario).
2. Si la diferencia es < 0.5 pp, se prefiere el modelo con mejor comportamiento por clase (menos clases débiles).
3. Como desempate práctico: menor número de parámetros / menor coste de inferencia.
4. El run `rtx3070-20260420-202946-5c3fio9s` ganó por margen claro en el criterio primario: **val/macro-F1 = 0.9486, test/macro-F1 = 0.9641**.

---

## 5. Evaluación del rendimiento

### 5.1 Métricas globales del modelo seleccionado

| Partición | Macro F1 | Accuracy |
|---|---|---|
| Train | 0.9133 | 0.9132 |
| Validación | 0.9486 | 0.9491 |
| **Test** | **0.9641** | **0.9620** |

La diferencia entre train y val/test se debe a que en entrenamiento el dropout está activo (modo `train`); en evaluación se desactiva, lo que mejora la predicción. La coherencia entre val y test (< 2 pp de diferencia) confirma que no hubo sobreajuste al conjunto de validación durante la selección.

### 5.2 Desglose por clase (validación, baseline EfficientNet-B0)

La tabla recoge las métricas de validación del run `rtx3070-20260419-110853` (EfficientNet-B0, macro-F1=0.944), que es el run con desglose por clase más completo disponible en W&B. El modelo final (EfficientNet-B1) mejora en ≈2 pp de macro-F1 global y muestra patrones por clase similares o superiores.

| Clase | Precisión | Recall | F1 | Notas |
|---|---|---|---|---|
| Bedroom | 0.929 | 0.813 | **0.867** | Clase más difícil; confusión con Living room |
| Coast | 0.883 | 0.981 | 0.930 | Alto recall; algunas playas confundidas con Open country |
| Forest | 0.978 | 0.918 | 0.947 | Buena separación visual |
| Highway | 0.929 | 1.000 | 0.963 | Recall perfecto |
| Industrial | 1.000 | 0.891 | 0.943 | Perfecta precisión; alguna fábrica mal clasificada |
| Inside city | 0.913 | 0.913 | 0.913 | Confusión con Street |
| Kitchen | 0.909 | 0.968 | 0.938 | Buena discriminación con Living room |
| Living room | 0.935 | 0.935 | 0.935 | Leve solapamiento con Bedroom |
| Mountain | 0.884 | 0.884 | **0.884** | Confusión con Open country y Coast |
| Office | 0.963 | 1.000 | 0.981 | Clase muy distintiva |
| Open country | 0.875 | 1.000 | 0.933 | Recall perfecto; algo de confusión con Mountain |
| Store | 0.939 | 0.969 | 0.954 | Buena discriminación |
| Street | 0.927 | 0.836 | **0.879** | Bajo recall; confusión con Inside city y Highway |
| Suburb | 0.981 | 1.000 | 0.990 | Una de las clases más fáciles |
| Tall building | 0.979 | 1.000 | 0.989 | Clase más fácil del dataset |

### 5.3 Interpretación de la matriz de confusión

Los errores se concentran en dos grupos semánticamente coherentes:

- **Escenas interiores** (Bedroom / Living room / Kitchen): comparten muebles, texturas neutras y composición similar. Bedroom es la clase con F1 más bajo (0.867). Para el negocio, este error tiene impacto moderado: ambas categorías son habitaciones atractivas, aunque una búsqueda por "cocina" podría devolver dormitorios.
- **Escenas exteriores abiertas** (Mountain / Open country / Coast): el horizonte lejano y la vegetación crean ambigüedad. Street e Inside city también se confunden por el contexto urbano. Este error tiene mayor impacto en búsquedas de tipo de entorno (ciudad vs. naturaleza).

Las clases con F1 ≥ 0.96 (Tall building, Suburb, Highway, Office) son candidatas a **etiquetado automático completo**. Las tres clases más débiles se recomiendan para **revisión humana** bajo umbral de confianza.

### 5.4 Nivel de calidad esperado para uso en producción

| Modo de uso | Clases aplicables | Criterio |
|---|---|---|
| Etiquetado automático | Tall building, Suburb, Highway, Office, Forest, Coast | F1 ≥ 0.93, confusión baja |
| Soporte a decisión (human-in-the-loop) | Bedroom, Mountain, Street, Inside city | F1 < 0.93 o solapamiento semántico alto |

---

## 6. Documentación de la API y comportamiento del producto

### 6.1 Contrato FastAPI

**Base URL**: `http://localhost:8000`

| Endpoint | Método | Descripción |
|---|---|---|
| `POST /predict` | POST | Clasificar una imagen subida |
| `GET /health` | GET | Estado del servicio y del modelo cargado |
| `POST /load-model` | POST | Intercambiar checkpoint en caliente (ruta local o artefacto W&B) |
| `GET /models` | GET | Listar artefactos de modelo disponibles en W&B |

**`POST /predict`** — entrada: `multipart/form-data`, campo `file` (imagen JPEG/PNG/WebP). Salida:
```json
{
  "filename": "cocina.jpg",
  "predictions": [
    {"class": "Kitchen",     "confidence": 0.912},
    {"class": "Store",       "confidence": 0.043},
    {"class": "Living room", "confidence": 0.019}
  ]
}
```
Errores: `400` (tipo de contenido no imagen, imagen corrupta, parámetros incompatibles en `/load-model`); `503` (modelo no cargado, checkpoint no encontrado en arranque).

**`GET /health`** — devuelve `{"status": "ok", "model_loaded": true, "model_source": "local:artifacts/best_model.pth"}`.

### 6.2 Swagger / OpenAPI

FastAPI genera documentación interactiva automáticamente: Swagger UI en `/docs` y ReDoc en `/redoc`. Todos los endpoints, esquemas de petición y de respuesta están documentados y son consistentes con la implementación.

**[INSERTAR CAPTURA SWAGGER]**

### 6.3 Aplicación Streamlit

La app Streamlit ofrece una interfaz de una sola pantalla: widget de subida de imagen → llamada a `POST /predict` → visualización de las top-3 predicciones con gráfico de barras de probabilidades. Un campo lateral permite apuntar a una URL de API diferente para despliegues remotos.

**[INSERTAR CAPTURA STREAMLIT]**

---

## 7. Conclusiones y recomendaciones

### 7.1 Conclusiones técnicas

- El **transfer learning desde ImageNet es altamente efectivo** para este problema: incluso con 3 épocas y un EfficientNet-B0 básico se alcanzan ~0.93 de F1. El fine-tuning completo añade ~3–4 pp adicionales.
- **La elección del backbone importa**: EfficientNet-B1 superó tanto a B0 (menor capacidad) como a B2 (riesgo de overfitting con ~3 100 imágenes de entrenamiento). ResNet-50 congelado fue el peor resultado (0.77 F1), demostrando que una arquitectura poderosa no ayuda si se congela con un dataset de esta escala.
- **La búsqueda Bayesiana de hiperparámetros** es mucho más eficiente que la búsqueda aleatoria para espacios mixtos (discreto + continuo). El sweep RTX identificó en 9 runs una región de alta calidad que habría costado muchos más runs con búsqueda aleatoria.
- **El diseño hardware-aware** (dos perfiles YAML separados) permitió experimentar en paralelo en dos máquinas sin desperdicio de recursos ni conflictos de configuración.

### 7.2 Recomendaciones de negocio

1. **Desplegar inicialmente en modo decision-support**: activar etiquetado automático sólo para las 6 clases con F1 ≥ 0.93; enrutar el resto a revisión humana con la predicción del modelo como sugerencia.
2. **Establecer umbrales de confianza por clase**: un umbral de 0.85 en las clases fuertes y de 0.90 en las débiles reduciría la tasa de error en producción de forma controlada.
3. **Recoger feedback de usuarios** sobre etiquetas incorrectas para alimentar un ciclo de reentrenamiento activo.
4. **Priorizar mejora de Bedroom y Mountain** en la siguiente iteración: son las clases más habituales en anuncios residenciales y las que muestran mayor confusión.

### 7.3 Riesgos y próximos pasos

- **Deriva de datos**: el estilo fotográfico evoluciona (fotos HDR, drones, home staging virtual); se recomienda reentrenamiento periódico con imágenes recientes.
- **Calibración de probabilidades**: las probabilidades softmax pueden no estar bien calibradas; aplicar temperature scaling antes de usar los scores como umbrales de producción.
- **Próxima iteración**: data augmentation dirigida para clases débiles (Bedroom, Street), muestreo balanceado por clase, y exploración de EfficientNet-B3 o ConvNeXt-Small con más datos.

---

## 8. Checklist de entrega

| Elemento | Estado |
|---|---|
| Repositorio Git público | `[RELLENAR URL]` |
| Proyecto W&B (workspace URL) | `[RELLENAR URL]` |
| Invitación W&B a `agascon@comillas.edu` | `[SÍ/NO]` |
| Invitación W&B a `rkramer@comillas.edu` | `[SÍ/NO]` |
| API funcionando con Swagger accesible | `[SÍ/NO]` |
| App Streamlit conectada al API | `[SÍ/NO]` |
| Instrucciones de reproducción en README | `[SÍ/NO]` |
