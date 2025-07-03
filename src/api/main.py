from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from typing import List
import mlflow.pyfunc
import pandas as pd
from src.api.pydantic_models import CreditRiskRequest, CreditRiskResponse
import mlflow
import os

app = FastAPI(title="Credit Risk Probability API")

MODEL_NAME = "BestCreditRiskModel"
MODEL_ALIAS = "production"
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        # Set MLflow tracking URI to local file system
        mlflow.set_tracking_uri("file:///app/mlruns")
        
        # Try to load from model registry first
        try:
            model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"Model loaded successfully from registry: {model_uri}")
        except Exception as registry_error:
            print(f"Registry loading failed: {registry_error}")
            # Fallback: load directly from artifacts
            artifacts_path = "/app/mlruns/502138449992107411/models/m-aa99119858ad443c869a01d1b4e3445a/artifacts"
            model = mlflow.pyfunc.load_model(artifacts_path)
            print(f"Model loaded successfully from artifacts: {artifacts_path}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.post("/predict", response_model=CreditRiskResponse)
def predict_risk(request: CreditRiskRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        # Convert request to DataFrame
        input_df = pd.DataFrame([request.dict()])
        # Predict risk probability
        proba = model.predict(input_df)
        # If model returns probability for both classes, take the positive class
        if hasattr(proba, '__len__') and len(proba) == 1:
            risk_prob = float(proba[0])
        else:
            risk_prob = float(proba)
        return CreditRiskResponse(risk_probability=risk_prob)
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}") 