from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "Models")

lr_model = joblib.load(os.path.join(MODEL_DIR, "linear_regression_model.pkl"))
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))

# Model registry
models = {
    "linear_regression": lr_model,
    "random_forest": rf_model,
    "xgboost": xgb_model
}

# Input schema for the API
class CarFeatures(BaseModel):
    Car_Name: str
    Year: int
    Present_Price: float
    Driven_kms: int
    Fuel_Type: str
    Selling_type: str
    Transmission: str
    Owner: int

def calculate_confidence(model, features_df: pd.DataFrame, model_name: str) -> float:
    """Calculate confidence score for a model's prediction"""
    try:
        if model_name == "random_forest":
            # Access the RandomForestRegressor inside the pipeline
            rf = model.named_steps['model']
            predictions = np.array([tree.predict(model.named_steps['preprocessor'].transform(features_df))[0] for tree in rf.estimators_])
            confidence = 1.0 / (1.0 + np.std(predictions))
            
        elif model_name == "xgboost":
            # Access the XGBRegressor inside the pipeline
            pred = model.predict(features_df)[0]
            confidence = 1.0 / (1.0 + abs(pred * 0.1))
            
        elif model_name == "linear_regression":
            pred = model.predict(features_df)[0]
            confidence = 1.0 / (1.0 + abs(pred * 0.05))
            
        else:
            confidence = 0.5  # Default confidence
            
        return min(max(confidence, 0.0), 1.0)
        
    except Exception as e:
        print(f"Error calculating confidence for {model_name}: {e}")
        return 0.0

def make_prediction_with_confidence(model, features: CarFeatures, model_name: str) -> Dict[str, Any]:
    """Make prediction and calculate confidence"""
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([features.dict()])
        
        # Make prediction
        pred = model.predict(input_df)
        prediction = float(pred[0])  # Ensure Python float
        
        # Calculate confidence
        confidence = float(calculate_confidence(model, input_df, model_name))  # Ensure Python float
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "success": True
        }
    except Exception as e:
        return {
            "prediction": 0.0,
            "confidence": 0.0,
            "success": False,
            "error": str(e)
        }

@app.post("/predict/best")
def predict_best_model(features: CarFeatures):
    """Predict using the model with highest confidence"""
    results = {}
    best_model = None
    best_confidence = -1
    best_prediction = None
    
    # Test all models
    for model_name, model in models.items():
        result = make_prediction_with_confidence(model, features, model_name)
        results[model_name] = result
        
        if result["success"] and result["confidence"] > best_confidence:
            best_confidence = result["confidence"]
            best_model = model_name
            best_prediction = result["prediction"]
    
    if best_model is None:
        raise HTTPException(status_code=500, detail="All models failed to make predictions")
    
    return {
        "best_model": best_model,
        "predicted_price": best_prediction,
        "confidence": best_confidence,
        "all_results": results
    }

@app.post("/predict/{model_name}")
def predict_specific_model(model_name: str, features: CarFeatures):
    """Predict using a specific model (original functionality)"""
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    result = make_prediction_with_confidence(models[model_name], features, model_name)
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {result.get('error', 'Unknown error')}")
    
    return {
        "model": model_name,
        "predicted_price": result["prediction"],
        "confidence": result["confidence"]
    }

@app.post("/predict/all")
def predict_all_models(features: CarFeatures):
    """Get predictions from all models with their confidence scores"""
    results = {}
    
    for model_name, model in models.items():
        result = make_prediction_with_confidence(model, features, model_name)
        results[model_name] = {
            "predicted_price": result["prediction"],
            "confidence": result["confidence"],
            "success": result["success"]
        }
        if not result["success"]:
            results[model_name]["error"] = result.get("error", "Unknown error")
    
    return {"results": results}

@app.get("/models")
def list_models():
    """List all available models"""
    return {"available_models": list(models.keys())}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": len(models)}