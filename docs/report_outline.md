# Final Technical Report (almost final draft, max 6 pages)

## 0. Scope and deliverable statement
This report presents an end-to-end image-classification solution for a real-estate marketplace, including transfer learning, experiment tracking with Weights & Biases, and deployment through FastAPI (backend) + Streamlit (frontend).

The deliverable includes:
- reproducible code in a public repository;
- traceable experimentation in W&B;
- a working inference API with Swagger documentation;
- a working Streamlit interface connected to the API;
- final conclusions and business recommendations.

---

## 1. Customer context (real-estate marketplace)
### 1.1 Business problem
Real-estate platforms manage large volumes of listing images. Manual labeling is slow, costly, and inconsistent. The customer needs an automated classifier that predicts scene/property categories to improve listing quality, filtering, and downstream search/recommendation systems.

### 1.2 Target users
- Internal content/moderation teams that validate listing metadata.
- Data/ML teams that need standardized visual labels.
- End users indirectly, through better search and browsing quality.

### 1.3 Expected business value
- Faster listing processing and reduced manual effort.
- More consistent category tagging.
- Better discoverability of properties by visual type.
- Foundation for future ranking/personalization use cases.

### 1.4 Operational constraints
- Multi-class setting with 15 output classes.
- Production inference must be responsive and robust.
- Full experiment traceability required for team collaboration.
- Reproducibility and deployment simplicity are mandatory.

---

## 2. System architecture
### 2.1 High-level architecture
The solution is organized in four blocks:
1) data preparation and split generation;  
2) model training/selection with W&B tracking;  
3) FastAPI inference service exposing `/predict`;  
4) Streamlit UI consuming the API.

### 2.2 Inference flow
1. User uploads an image in Streamlit.  
2. Streamlit sends the file to FastAPI (`POST /predict`).  
3. FastAPI loads the selected checkpoint and returns class probabilities.  
4. Streamlit renders top predictions to the user.

### 2.3 Deployment view
- Backend: FastAPI service with OpenAPI/Swagger docs.
- Frontend: Streamlit app as a thin client over the API.
- Model artifact: best checkpoint exported from the training pipeline.

**[INSERT ARCHITECTURE FIGURE HERE]**

---

## 3. Modeling approach
### 3.1 Pre-trained model selection
The project evaluates transfer-learning candidates from `timm` and selects a final backbone based on validation macro F1, stability across runs, and practical training/inference cost.

### 3.2 Transfer-learning strategy
- Initialize from ImageNet pre-trained weights.
- Replace the final classification head with a 15-class output layer.
- Tune key settings (e.g., backbone family, learning rate, weight decay, dropout, freeze/unfreeze policy).
- Train with early stopping to reduce overfitting and wasted compute.

### 3.3 Final architecture used in API
The production API uses the checkpoint associated with the best validated run according to the model-selection criterion defined in Section 4.

**Final selected backbone:** `[FILL]`  
**Input size:** `[FILL]`  
**Trainable parameters:** `[FILL]`  
**Checkpoint path/artifact reference:** `[FILL]`

### 3.4 Hardware-aware experimentation setup (important for this project)
Modeling decisions were constrained by the available team hardware profiles (GTX-class and RTX-class GPUs). Instead of assuming unlimited compute, experiments were designed to be reproducible on the weakest available setup and scalable on stronger hardware.

Document explicitly:
- which configs were used for each GPU profile (e.g., `gtx1650.yaml` vs `rtx3070.yaml`);
- why those configs differ (batch size, workers, epochs, training time budget);
- which ranges were considered "safe" to avoid out-of-memory failures;
- how mixed precision and early stopping helped control resource usage.

Suggested wording:
"We adapted training profiles to available GPUs. GTX-oriented runs used more conservative memory settings, while RTX-oriented runs allowed larger batches/longer schedules. This ensured fair experimentation without sacrificing reproducibility across team machines."

---

## 4. Experimentation process (Weights & Biases)
### 4.1 Experimental design
The team follows a staged process:
- baseline runs to validate data/training pipeline;
- controlled experiments for architecture and optimization settings;
- sweep-based hyperparameter search for broader coverage.

### 4.2 Hyperparameter tuning strategy
Sweep configuration explores combinations of:
- backbone;
- learning rate;
- weight decay;
- batch size;
- freeze-backbone option.

Primary selection metric: **`val/macro_f1`** (maximize).  
Secondary monitoring metrics: `accuracy` (train/val/test) and confusion matrix on test.

Clarify the decision workflow:
1. Run broad sweeps to map which regions of hyperparameters consistently perform well.  
2. Identify stable patterns (not only single best run).  
3. Launch targeted follow-up runs with values close to those high-performing regions.  
4. Select final model using validation macro F1 + class-balance behavior + compute practicality.

Suggested wording:
"Hyperparameter selection was not based on a single lucky run. We first used W&B sweeps to detect robust high-performing regions, then refined with focused experiments around those values. Final settings were chosen from this combined evidence, considering both quality and GPU feasibility."

### 4.3 Tracking and reproducibility
Each run is logged in W&B with:
- config values and runtime metadata;
- train/val curves for macro F1 and accuracy;
- test metrics and confusion matrix;
- run naming convention with config + timestamp + run id.

### 4.4 Final model selection criteria
1. Highest `val/macro_f1`.  
2. If close, prefer the model with better class balance (confusion matrix / per-class behavior).  
3. Consider compute/inference cost as practical tie-breaker.
4. Prefer configurations that are reproducible across available team GPUs (not only on the strongest device).

**[INSERT W&B SCREENSHOT(S): SWEEP OVERVIEW + BEST RUN + METRICS PANEL]**

---

## 5. Performance evaluation (customer-facing quality)
### 5.1 Global summary
Report final test performance of the selected model:
- Test macro F1: `[FILL]`
- Test accuracy: `[FILL]`
- Optional: test loss: `[FILL]`

### 5.2 Per-class quality
Include per-class precision, recall, and F1-score to highlight strong/weak categories and error concentration.

### 5.3 Confusion matrix interpretation
Discuss major confusion patterns and their business impact (e.g., classes with similar visual context that may affect search relevance or moderation workflow).

### 5.4 Expected quality level for usage
State whether current quality is sufficient for:
- decision support (human-in-the-loop), or
- full automation for specific classes only.

---

## 6. API documentation and product behavior
### 6.1 FastAPI contract
Document:
- `POST /predict` input format (`multipart/form-data`, image file);
- output schema (predictions + probabilities);
- error handling (`400`, `503`, and other relevant responses).

### 6.2 Swagger/OpenAPI
Confirm that Swagger docs are available and consistent with implementation.

**[INSERT SWAGGER SCREENSHOT HERE]**

### 6.3 Streamlit end-to-end behavior
Show the end-user workflow and a sample prediction result.

**[INSERT STREAMLIT SCREENSHOT HERE]**

---

## 7. Conclusions and business recommendations
### 7.1 Technical conclusions
- Transfer learning is effective for this dataset and use case.
- W&B-based experimentation enables reproducible model selection.
- FastAPI + Streamlit provides a clean production-oriented inference path.

### 7.2 Business recommendations
- Deploy as decision-support first (human-in-the-loop).
- Prioritize improvement of most-confused classes.
- Add continuous monitoring and periodic retraining with fresh data.

### 7.3 Risks and next steps
- Dataset drift and class ambiguity.
- Potential calibration needs for confidence thresholds.
- Next iteration: targeted data augmentation, class-aware sampling, and threshold policy by class.

---

## 8. Mandatory links and access checklist (submission gate)
- Public Git repository URL: `[FILL]`
- W&B project/workspace URL: `[FILL]`
- Confirm invitations sent in W&B to:
  - `agascon@comillas.edu`
  - `rkramer@comillas.edu`
- API running with accessible Swagger docs: `[YES/NO]`
- Streamlit app connected to API: `[YES/NO]`
- Reproducible setup instructions in README: `[YES/NO]`

---

## 9. Suggested page budget (to stay within 6 pages)
- Section 1: 0.5 page
- Section 2: 0.75 page
- Section 3: 0.75 page
- Section 4: 1.25 pages (includes W&B figures)
- Section 5: 1.25 pages (includes confusion matrix and class table)
- Section 6: 0.75 page (Swagger + Streamlit figure)
- Section 7 + Section 8: 0.75 page

