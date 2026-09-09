# AI-Based PII Detection for Data Loss Prevention

**Hybrid machine-learning and rule-based pipeline for detecting sensitive information using XGBoost, TF-IDF character features, structured metadata, and SHAP explainability.**

This project explores how machine learning can complement deterministic pattern matching in a Data Loss Prevention (DLP) workflow for identifying personally identifiable information (PII).

## At a Glance

| Dimension | Project |
|---|---|
| Primary task | PII / sensitive-data classification |
| Final classifier | XGBoost |
| Text representation | TF-IDF character features |
| Additional signals | Regex / structured metadata |
| Explainability | SHAP |
| Baselines | Logistic Regression, Random Forest |
| Application layer | Python API / DLP inference pipeline |
| Best reported project accuracy | **94%** |

The reported accuracy reflects the project's evaluation setup and is not presented as a universal production-level DLP benchmark.

---

## Why This Project Exists

Traditional DLP systems often rely on deterministic pattern matching. Regex works well for highly structured identifiers, but sensitive-data detection becomes harder when formatting changes, text is incomplete, or contextual information matters.

This project therefore combines deterministic detection with machine-learning classification.

---

## Pipeline

```text
Input Text
    |
    v
Preprocessing
    |
    +----------------------+
    |                      |
    v                      v
Regex / Metadata       TF-IDF Features
    |                      |
    +----------+-----------+
               |
               v
            XGBoost
               |
               v
        PII Classification
               |
        +------+------+
        |             |
        v             v
   Policy Logic    SHAP Analysis
        |
        v
     DLP Output
```

---

## Model Development

### Logistic Regression
Used as a linear baseline.

### Random Forest
Used as a nonlinear ensemble baseline.

### XGBoost
Selected as the final classifier after experimentation on the project dataset.

Final serialized artifacts:

```text
artifacts_xgb/
â”œâ”€â”€ label_encoder.joblib
â”œâ”€â”€ tfidf_vectorizer_xgb.joblib
â””â”€â”€ xgboost_classifier.joblib
```

Earlier model-development experiments are retained separately under `experiments/`.

---

## Explainability

SHAP was used to inspect feature contributions across PII classes. Generated plots are stored under `artifacts/shap/`.

---

## Repository Structure

```text
DLP-Project/
â”œâ”€â”€ .github/workflows/        # Repository validation CI
â”œâ”€â”€ app/                      # API and inference layer
â”œâ”€â”€ artifacts/
â”‚   â”œâ”€â”€ policy.json
â”‚   â””â”€â”€ shap/                 # SHAP visualizations
â”œâ”€â”€ artifacts_xgb/            # Final serialized XGBoost pipeline
â”œâ”€â”€ data/                     # Project datasets
â”œâ”€â”€ demos/                    # Prototype web interfaces
â”œâ”€â”€ docs/                     # Project report and deliverables
â”œâ”€â”€ experiments/              # Earlier BERT / RF / XGBoost experiments
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data_preprocess.py
â”‚   â”œâ”€â”€ logistic_regression.py
â”‚   â”œâ”€â”€ metrics.py
â”‚   â””â”€â”€ train_xgb.py
â”œâ”€â”€ tests/
â”œâ”€â”€ README.md
â””â”€â”€ requirements.txt
```

---

## Setup

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

---

## Application Layer

The application/inference layer is under `app/`:

```text
app/
â”œâ”€â”€ api.py
â”œâ”€â”€ dlp_core.py
â””â”€â”€ server.py
```

It connects the trained model, deterministic rules, and policy logic to application-facing inference.

---

## Training

Canonical XGBoost training implementation:

```text
src/train_xgb.py
```

Supporting code:

```text
src/data_preprocess.py
src/logistic_regression.py
src/metrics.py
```

Previous experimental implementations are preserved under `experiments/`.

---

## Historical BERT Experiment

An earlier experiment used DistilBERT embeddings combined with regex-derived signals and XGBoost.

The source is retained as:

```text
experiments/bert_xgboost_optuna_shap.py
```

The generated embedding cache is intentionally not version-controlled because it is large and reproducible from the source experiment.

---

## Validation

Run repository checks with:

```bash
python -m pytest tests/test_repository.py
```

GitHub Actions also runs these checks automatically on pushes and pull requests.

---

## Limitations

This is an academic / portfolio DLP prototype rather than a production enterprise DLP platform.

Current limitations include:

- evaluation on a limited project dataset;
- no independent external benchmark;
- potential class imbalance and dataset-specific behaviour;
- regex rules may require localization;
- serialized models may not generalize to unseen enterprise data;
- no enterprise endpoint or document-management integration;
- no production-grade policy orchestration;
- no guarantee against adversarially crafted PII examples.

The reported **94% best project accuracy** should therefore be interpreted only within the documented project evaluation context.

---

## Future Work

Potential extensions include:

- larger and more diverse PII datasets;
- precision / recall / F1 reporting by class;
- independent held-out evaluation;
- multilingual sensitive-data detection;
- document and file-level DLP;
- active-learning feedback loops;
- transformer-based comparison models;
- enterprise policy integration;
- adversarial robustness testing.

---

## Responsible Use

This repository is intended for educational, defensive-security, and privacy-engineering work.

Real PII should not be committed to public repositories or processed without appropriate authorization and privacy controls.

