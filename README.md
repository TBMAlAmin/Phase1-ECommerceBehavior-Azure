# Phase 2 — Modeling, Validation, and Deployment

## Overview
This phase focuses on developing, validating, and deploying a machine learning model to predict whether a user session results in a purchase. The system builds on the Phase 1 data pipeline and ensures consistency between training and production.

## Problem Definition
This is a binary classification task:
- Input: Early-session features from user activity
- Output: Purchase (1) or No Purchase (0)

## Data and Features
Dataset:
- Source: processed_2019_oct.csv (Phase 1 output)
- Development: processed_2019_oct_sample_200k.csv

Leakage-safe features (only first event used):
- first_event_type
- first_hour
- first_dayofweek
- first_price
- brand_missing
- category_missing

## Model Development
Baseline:
- Logistic Regression

Improved Model:
- Random Forest

Reproducibility:
- Fixed random_state=42
- Train/test split (80/20, stratified)

## Model Validation
Metrics used:
- ROC-AUC (primary)
- Precision, Recall, F1-score

Results:
- Logistic Regression ROC-AUC: 0.6399
- Random Forest ROC-AUC: 0.7547

Analysis:
- Dataset is highly imbalanced (~2% positive)
- Accuracy is misleading, ROC-AUC preferred
- Random Forest performs significantly better
- Data was split into training and testing sets using an 80/20 stratified split with random_state=42 to ensure reproducibility.

## Model Selection
Random Forest selected due to:
- Higher ROC-AUC
- Better handling of non-linear patterns

## Model Versioning
- Model: rf_model_latest.joblib
- Metadata: phase2/models/model_metadata.txt

Ensures traceability between:
data → features → model → deployment

## Deployment
Architecture:
Request → API → Feature mapping → Model → Prediction

Deployment:
- FastAPI application
- Hosted on Azure VM
- Real-time inference using uvicorn

Endpoint:
http://20.173.41.159:8000/predict

## API Example
Request:
{
  "first_event_type": "view",
  "first_hour": 5,
  "first_dayofweek": 1,
  "first_price": 31.66,
  "brand_missing": 0,
  "category_missing": 1
}

Response:
{
  "prediction": 0,
  "purchase_probability": 0.105887
}

## Deployment Validation
- Successful API calls via curl
- Swagger UI (/docs) accessible
- Consistent predictions between local and deployed model

## Limitations
- Strong class imbalance
- Model trained on sample data
- Limited feature set
- No threshold tuning applied

## Conclusion
This phase delivers a complete ML pipeline:
- Model development and validation
- Leakage-safe feature engineering
- Model comparison and selection
- Real-time deployment on Azure
- Successful deployment validation
