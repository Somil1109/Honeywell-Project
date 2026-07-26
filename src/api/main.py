from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="SOC Behavioral Anomaly API")

# Load the trained baseline model
try:
    baseline_model = joblib.load("src/models/saved_models/isolation_forest_v1.pkl")
    print("Baseline model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define the expected incoming data schema
class AccessLog(BaseModel):
    session_duration_scaled: float
    hour_of_day_scaled: float
    day_of_week_scaled: float
    entity_id_encoded: int
    entity_type_encoded: int
    source_ip_encoded: int
    resource_accessed_encoded: int
    auth_method_encoded: int
    command_sequence_encoded: int
    device_fingerprint_encoded: int

def calculate_explainability(features_df, threshold=1.5):
    # Lightweight explainability: flags features that deviate significantly from a 0-mean standard
    # For a real production system, SHAP would be implemented here
    reasons = []
    if abs(features_df['session_duration_scaled'].iloc[0]) > threshold:
        reasons.append("Abnormal session duration")
    if abs(features_df['hour_of_day_scaled'].iloc[0]) > threshold:
        reasons.append("Unusual access time")
    
    return reasons if reasons else ["Complex sequential anomaly detected"]

@app.get("/")
def health_check():
    return {"status": "active", "models_loaded": "isolation_forest"}

@app.post("/analyze_event")
def analyze_event(log: AccessLog):
    # Convert incoming JSON to DataFrame
    input_data = pd.DataFrame([log.model_dump()])
    
    # Get anomaly score from Isolation Forest (-1 for anomaly, 1 for normal)
    # The decision function returns a raw score, lower is more anomalous
    is_anomaly = baseline_model.predict(input_data)[0]
    raw_score = baseline_model.decision_function(input_data)[0]
    
    # Normalize risk score to 0-100 scale for the dashboard
    risk_score = round(float(np.clip(50 - (raw_score * 100), 0, 100)), 2)
    
    response = {
        "is_anomalous": bool(is_anomaly == -1),
        "risk_score": risk_score,
        "classification": "normal",
        "contributing_factors": []
    }
    
    if response["is_anomalous"]:
        response["classification"] = "potential_intrusion"
        response["contributing_factors"] = calculate_explainability(input_data)
        
    return response