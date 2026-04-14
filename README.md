# Real-Estate Image Classifier

Automatic classification of real-estate images using transfer learning.

Built for a simulated marketplace use case (Idealista/Zillow style).

## Project scope (phase 1)

- Reproducible data preparation (`training/validation` -> `train/val/test`)
- Transfer-learning training pipeline with Weights & Biases tracking
- FastAPI inference service with Swagger docs
- Streamlit front-end connected to the API

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

## Setup

```bash
uv venv --python 3.11
.venv\Scripts\activate
uv sync
```

## Quickstart

1) Prepare processed splits:

```bash
python src/prepare_data.py
```

2) Train + track in W&B:

```bash
python src/train.py --config configs/base_config.yaml
```

3) Run API:

```bash
uvicorn api.main:app --reload
```

4) Run Streamlit:

```bash
streamlit run app/app.py
```

## Team workflow (2-3 people)

- ML/Data: data pipeline + augmentations + quality checks
- Training/W&B: experiments + hyperparameter tuning + model selection
- Product/API/UI: FastAPI + Streamlit + end-to-end tests

All merges should be done through small PRs with reproducible run references in W&B.

## Delivery checklist (phase 1)

- [ ] Reproducible setup in a clean environment
- [ ] W&B runs with traceable configs and artifacts
- [ ] FastAPI docs available in `/docs`
- [ ] Streamlit app connected to API and returning predictions

