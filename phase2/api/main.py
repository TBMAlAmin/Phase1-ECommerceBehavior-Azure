from fastapi import FastAPI
from pydantic import BaseModel
from phase2.api.predict import predict_session

app = FastAPI(
    title="Phase 2 Session Purchase Prediction API",
    description="Predicts whether a session will follow the sequential funnel view -> cart -> purchase",
    version="1.0.0",
)

class SessionInput(BaseModel):
    first_event_type: str
    first_hour: int
    first_dayofweek: int
    first_price: float
    brand_missing: int
    category_missing: int

@app.get("/")
def root():
    return {
        "message": "Phase 2 prediction API is running",
        "model": "rf_model_latest.joblib",
        "endpoint": "/predict"
    }

@app.post("/predict")
def predict(input_data: SessionInput):
    result = predict_session(input_data.model_dump())
    return result
