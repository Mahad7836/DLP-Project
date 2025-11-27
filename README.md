# DLP Clean Codebase


This repository contains a modular Data Loss Prevention (DLP) prototype.
It combines rule-based detectors (regex + strict phone parsing) with ML classifiers (TF-IDF + XGBoost) and interpretability (SHAP).


**Quick start**
1. Create and activate your virtualenv (Windows CMD):
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
