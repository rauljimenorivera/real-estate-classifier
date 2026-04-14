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
- `configs/experiments/gtx1650.yaml`: fast config for Raul
- `configs/experiments/rtx3070.yaml`: full config for Natalia
- `notebooks/01_data_prep_eda.ipynb`: data prep and class distribution checks
- `notebooks/02_training_experiments.ipynb`: interactive training + W&B
- `notebooks/03_model_selection_eval.ipynb`: model comparison and test evaluation
- `notebooks/04_inference_api_test.ipynb`: local inference + API test

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

## Team split (2 people)

- Raul (GTX 1650):
  - runs `configs/experiments/gtx1650.yaml`
  - owns notebooks `01` and `04`
  - validates data, API, and integration
- Natalia (RTX 3070):
  - runs `configs/experiments/rtx3070.yaml`
  - owns notebooks `02` and `03`
  - performs longer experiments and model selection

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

