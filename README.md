\# Real-Estate Image Classifier



Automatic classification of real-estate images using transfer learning.

Built for a simulated marketplace use case (Idealista / Zillow style).



\## Classes

Bedroom · Kitchen · Living room · Office · Store · Coast · Highway ·

Forest · Industrial · Inside city · Mountain · Open country · Street ·

Suburb · Tall building



\## Stack

\- \*\*Model:\*\* Transfer learning (EfficientNet / ResNet) via `timm`

\- \*\*Experiment tracking:\*\* Weights \& Biases

\- \*\*API:\*\* FastAPI + Swagger docs

\- \*\*Frontend:\*\* Streamlit



\## Setup



```bash

uv venv --python 3.11

.venv\\Scripts\\activate  # Windows

uv sync

```



\## Usage



```bash

\# Train

python src/train.py



\# API

uvicorn api.main:app --reload



\# App

streamlit run app/app.py

```



\## Team

\- W\&B project: \[link]

\- GitHub: \[link]

