from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import shap
import os

app = FastAPI(title="SOC Multi-Class Anomaly & Explainability API")

class MultiClassLSTMDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiClassLSTMDetector, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class SecurityContextTracker:
    def __init__(self):
        self.seen_entities = set()
        self.drift_allowance = {} 

tracker = SecurityContextTracker()

baseline_model = None
label_encoder = None
sequence_model = None
explainer = None
rolling_window = []  
EXPECTED_FEATURES = 10

@app.on_event("startup")
def load_models():
    global baseline_model, label_encoder, sequence_model, explainer
    
    baseline_model = joblib.load("src/models/saved_models/isolation_forest_v1.pkl")
    explainer = shap.TreeExplainer(baseline_model)
    label_encoder = joblib.load("src/models/saved_models/label_encoder.pkl")
    num_classes = len(label_encoder.classes_)
    
    sequence_model = MultiClassLSTMDetector(input_size=EXPECTED_FEATURES, hidden_size=64, num_classes=num_classes)
    sequence_model.load_state_dict(torch.load("src/models/saved_models/lstm_detector_v1.pth", weights_only=True))
    sequence_model.eval()
    
    print("All models and SHAP explainer loaded successfully.")

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

def get_shap_explainability(input_df):
    shap_values = explainer.shap_values(input_df)
    feature_names = input_df.columns
    contributions = np.abs(shap_values[0])
    
    total_impact = np.sum(contributions)
    if total_impact == 0:
        return ["Unknown complex anomaly"]
        
    percentages = (contributions / total_impact) * 100
    top_indices = np.argsort(percentages)[::-1][:3]
    reasons = [f"{feature_names[i]} ({percentages[i]:.1f}% impact)" for i in top_indices if percentages[i] > 5.0]
    
    return reasons

@app.get("/")
def health_check():
    return {"status": "active", "models_loaded": "isolation_forest, lstm, shap"}

@app.post("/analyze_event")
def analyze_event(log: AccessLog):
    global rolling_window
    
    input_data = pd.DataFrame([log.model_dump()])
    input_data = input_data[baseline_model.feature_names_in_]
    
    event_array = input_data.values[0]
    rolling_window.append(event_array)
    if len(rolling_window) > 5:
        rolling_window.pop(0)
    
    is_anomaly = baseline_model.predict(input_data)[0]
    raw_score = baseline_model.decision_function(input_data)[0]
    
    # Cold-Start Handling
    is_cold_start = log.entity_id_encoded not in tracker.seen_entities
    if is_cold_start:
        tracker.seen_entities.add(log.entity_id_encoded)
        raw_score += 0.05 
        
    risk_score = round(float(np.clip(50 - (raw_score * 100), 0, 100)), 2)
    
    response = {
        "is_anomalous": bool(risk_score > 65.0), 
        "risk_score": risk_score,
        "classification": "normal",
        "contributing_factors": [],
        "is_cold_start": is_cold_start
    }
    
    if response["is_anomalous"]:
        response["contributing_factors"] = get_shap_explainability(input_data)
        
        if len(rolling_window) == 5:
            with torch.no_grad():
                seq_tensor = torch.tensor(np.array([rolling_window]), dtype=torch.float32)
                predictions = sequence_model(seq_tensor)
                predicted_class_idx = torch.argmax(predictions, dim=1).item()
                predicted_class_name = label_encoder.inverse_transform([predicted_class_idx])[0]
                response["classification"] = predicted_class_name
        else:
            response["classification"] = "insufficient_sequence_data"
            
    # Concept Drift Handling
    if response["classification"] == "insider_drift":
        tracker.drift_allowance[log.entity_id_encoded] = tracker.drift_allowance.get(log.entity_id_encoded, 0) + 1
        if tracker.drift_allowance[log.entity_id_encoded] > 3:
            response["is_anomalous"] = False
            response["risk_score"] -= 20.0
            response["classification"] = "normal (drift adjusted)"

    return response