# Phase 2: Session Purchase Prediction – Modeling, Deployment & DevOps

## Overview
This phase builds a machine learning pipeline to predict whether a user session 
will result in a purchase. It extends the Phase 1 data pipeline with model 
training, validation, deployment, and CI automation using Azure ML and Azure DevOps.

The full workflow is:
Data → Feature Engineering → Model Training → Validation → Deployment → DevOps

---

## Problem Statement
Given early signals from a user session, predict whether the session will end 
in a purchase (binary classification: 1 = purchase, 0 = no purchase).

---

## Dataset
- Source: E-Commerce Behavior Dataset (Kaggle, Oct 2019)
- File: `data/session_dataset_safe.csv`
- Each row represents one user session with early session signals as features

### Features Used
| Feature         | Description                                        |
|-----------------|----------------------------------------------------|
|first_event_type | First action in the session (view, cart, purchase) |
|first_hour       | Hour of day when session started                   |
|first_dayofweek  | Day of week when session started                   |
|first_price      | Price of first product viewed                      |
|brand_missing    | Whether brand info was missing (1/0)               |
|category_missing | Whether category info was missing (1/0)            |

### Label
- `label = 1` if session resulted in a purchase
- `label = 0` if session did not result in a purchase

---

## Dataset Splits
The dataset was split into 4 partitions:

| Split      | Size              | Purpose                   |
|------------|-------------------|---------------------------|
| Train      | 60% (30,048 rows) | Model training            |
| Validation | 15% (6,439 rows)  | Hyperparameter tuning     |
| Test       | 15% (6,440 rows)  | Final offline evaluation  |
| Deploy     | 10% (4,770 rows)  | Simulates production data |

All splits are registered as Azure ML Data Assets:
- `session_train`
- `session_val`
- `session_test`
- `session_deploy`

---

## Model Development

### Baseline Model — Logistic Regression
A Logistic Regression model was used as the baseline to establish a performance 
lower bound.

### Main Model — Random Forest Classifier
Random Forest was selected because:
- It handles non-linear relationships between features
- It is robust to class imbalance with `class_weight="balanced"`
- It provides probability estimates for AUC computation
- It outperformed the baseline on all metrics

### Hyperparameter Tuning
An Azure ML Sweep Job was used to tune:
- `n_estimators`: [50, 100, 200]
- `max_depth`: [5, 10, 15]

Best configuration found: `n_estimators=100`, `max_depth=15`

---

## Results

### Model Comparison
| Metric        | Baseline (LR) | RF Default | RF Best (Sweep) |
|---------------|---------------|------------|-----------------|
| Val Accuracy  | 0.394         | 0.664      | 0.799           |
| Val AUC       | 0.668         | 0.798      | 0.804           |
| Val F1        | 0.059         | 0.091      | 0.112           |
| Runtime (s)   | 8.4           | 11.5       | 11.6            |

### Best Model — Final Metrics
| Split      | Accuracy | AUC   | F1    |
|------------|----------|-------|-------|
| Train      | 0.810    | 0.910 | 0.164 |
| Validation | 0.799    | 0.804 | 0.112 |
| Test       | 0.980    | 0.688 | 0.000 |

---

## Validation Strategy
- Train/val/test splits with fixed `random_state=42` for reproducibility
- Validation set used for hyperparameter tuning — test set never touched during training
- Metrics logged: Accuracy, AUC, Precision, Recall, F1
- No data leakage — splits created before any feature processing

### Known Limitation — Class Imbalance
The dataset is heavily imbalanced. Most sessions do not result in a purchase. 
This causes F1 and Precision to be 0 on the test set despite high accuracy. 
The model predicts the majority class on harder splits. 
`class_weight="balanced"` was used to partially address this.

---

## Model Versioning
Models are registered in the Azure ML Model Registry:

| Version | Model                  | Job                        | Notes                   |
|---------|------------------------|----------------------------|-------------------------|
| v1      | session-purchase-model | nice_chicken_d5rp2pp390    | Default hyperparameters |
| v2      | session-purchase-model | helpful_battery_857lhdkx20 | Best sweep config       |

---

## Repository Structure
├── azure-pipelines.yml       # Azure DevOps CI pipeline
├── data/                     # Dataset splits
│   ├── session_dataset_safe.csv
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── deploy.csv
├── env/
│   ├── conda.yml             # Training environment
│   └── inference_conda.yml   # Deployment environment
├── jobs/
│   ├── train_job.yml         # Main training job
│   ├── baseline_job.yml      # Baseline training job
│   ├── sweep_job.yml         # Hyperparameter sweep job
│   └── deployment.yml        # Deployment configuration
└── src/
    ├── train.py              # Main training script
    ├── train_baseline.py     # Baseline training script
    ├── score.py              # Scoring script for endpoint
    └── invoke_endpoint.py    # Deployment validation script

---

## Deployment
The best model (v2) was deployed as an Azure ML Managed Online Endpoint.

- Endpoint name: `session-endpoint`
- Scoring URL: `https://session-endpoint.qatarcentral.inference.ml.azure.com/score`
- Instance type: `Standard_F2s_v2`
- Auth mode: key

### Deployment Validation
The deploy dataset (4,770 rows) was sent to the endpoint using `invoke_endpoint.py`.

| Metric | Value |
|---|---|
| Deployment Accuracy | 0.9797 |
| Deployment F1 | 0.000 |
| Total Predictions | 4,770 |

Deployment accuracy matches test accuracy, confirming consistency between 
offline and online predictions. F1 of 0 reflects the class imbalance issue 
present in both test and deployment splits.

---

## DevOps Automation
Azure DevOps CI pipeline automatically submits the Azure ML training job 
on every push to the `phase2-modeling-deployment` branch.

Pipeline: `azure-pipelines.yml`
Trigger: push to `phase2-modeling-deployment`
Service Connection: `SC-UDST-CCIT-DSAI3202`

Workflow:
code push → Azure DevOps pipeline → Azure ML training job → MLflow metrics → model artifact

---

## Reproducibility
- Fixed `random_state=42` in all splits and models
- All dependencies pinned in `env/conda.yml`
- All datasets versioned as Azure ML Data Assets
- All models registered in Azure ML Model Registry with job lineage