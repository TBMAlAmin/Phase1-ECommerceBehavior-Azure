# Assignment 2 – Model Training & Automation with Azure ML

## Overview

This assignment extends the Amazon Electronics review project by introducing **automated model training, experiment tracking, and deployment using Azure Machine Learning**.

The final workflow implemented was:

**code push → Azure DevOps pipeline → Azure ML training job → MLflow tracking → model registration → endpoint deployment → batch inference**

---

## Repository Structure

The project follows the required structure:

```
src/
env/
jobs/
```

### Main Files

#### src/

- `train.py` – Handles model training, preprocessing, evaluation, MLflow logging, and model saving  
- `score.py` – Defines `init()` and `run()` for Azure ML endpoint inference  
- `invoke_endpoint.py` – Sends batched requests to the deployed endpoint and computes accuracy  

#### env/

- `conda.yml` – Training environment definition  
- `inference_conda.yml` – Deployment/inference environment  

#### jobs/

- `train_job.yml` – Azure ML training job configuration  
- `deployment.yml` – Online endpoint deployment configuration  

---

## Pipeline Automation (Azure DevOps)

A pipeline was configured using Azure DevOps to automate training:

- Trigger: Git push to repository  
- Task: Azure CLI execution  
- Action: Submit Azure ML training job  

This ensures reproducibility and CI/CD integration.

---

## Model Training & Experiment Tracking

Training was executed in Azure ML using:

- **Azure ML Compute**
- **MLflow tracking**

Logged metrics included:

- Accuracy  
- Model artifacts (`model.pkl`)  
- Training parameters  

The trained model was registered as:

```
amazon-review-sentiment-model (version 1)
```

---

## Deployment

The model was deployed using an **Azure ML Managed Online Endpoint**:

- Endpoint name: `amazon-review-endpoint`  
- Deployment name: `blue`  
- Instance type: `Standard_DS3_v2`  

Key configuration:

- Scoring script: `score.py`  
- Model path handled via `AZUREML_MODEL_DIR`  
- Environment defined via `inference_conda.yml`  

---

## Endpoint Invocation

Inference was performed using:

```
python src/invoke_endpoint.py \
  --data ~/Downloads/deploy_data/data.parquet \
  --scoring_uri "<endpoint_url>" \
  --key "<primary_key>" \
  --batch_size 100
```

### Key Observations

- Large requests caused **HTTP 413 (Request Entity Too Large)**  
- Solution: batch processing  
- Optimal batch size significantly reduces runtime  

---

## Results

- Total predictions: **29,998**
- Deployment accuracy: **0.8555**

This confirms that:

- The deployed model behaves consistently with training results  
- Endpoint inference pipeline is functioning correctly  

---

## Issues Encountered & Fixes

### 1. Missing Environment File in Pipeline
- Cause: `.gitignore` excluded `env/`
- Fix: Force-add or update `.gitignore`

### 2. Deployment Failure (Container Crash)
- Cause: incorrect model path in `score.py`
- Fix:
```python
model_path = os.path.join(
    os.environ["AZUREML_MODEL_DIR"],
    "model_output",
    "model.pkl"
)
```

### 3. Python Package Errors (Local)
- Cause: macOS externally-managed Python (PEP 668)
- Fix: use virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
pip install pandas pyarrow requests scikit-learn
```

### 4. Large Payload Error (413)
- Cause: sending entire dataset at once  
- Fix: batch inference  

---

## Cleanup

To avoid unnecessary Azure costs:

```
az ml online-endpoint delete \
  --name amazon-review-endpoint \
  --yes
```

---

## Final Workflow Summary

1. Data preparation (Lab 4 pipeline output)  
2. Automated training via Azure DevOps  
3. MLflow experiment tracking  
4. Model registration  
5. Endpoint deployment  
6. Batch inference and evaluation  
7. Resource cleanup  

---

## Conclusion

This assignment demonstrates a complete **MLOps pipeline** using Azure:

- Automated training  
- Reproducible experiments  
- Scalable deployment  
- Production-style inference  

The system successfully integrates **DevOps + ML lifecycle**, aligning with real-world machine learning engineering practices.
