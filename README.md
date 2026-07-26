# AI-Powered Behavioral Anomaly Detection

## Project Overview
This project is an AI/ML system designed to model "normal" access behavior for users and devices, detecting intrusions in near real-time. It handles extreme class imbalance, concept drift, and cold-start entities, while providing deep SHAP-based explainability for SOC analysts.

## Architecture
- **Data Pipeline:** Synthetic telemetry generator with injected attack taxonomies (brute force, impossible travel, lateral movement, credential stuffing, low and slow, device spoofing, insider drift).
- **Backend (FastAPI):** Hosts the Isolation Forest (baseline profiler), PyTorch LSTM (sequence detector), and SHAP explainer.
- **Frontend (Streamlit):** Enterprise SOC dashboard with live telemetry and entity deep-dive capabilities.

## Setup Instructions
1. **Create Virtual Environment:** `python -m venv venv`
2. **Activate:** `source venv/bin/activate` (Linux/Mac)
3. **Install Dependencies:** `pip install -r requirements.txt`
4. **Run the API:** `uvicorn src.api.main:app --reload`
5. **Run the Dashboard (in a new terminal):** `streamlit run dashboard/app.py`

## Deliverables
- Synthetic Data Generator
- Baseline Profiling Model
- Sequence Detection Model
- Anomaly Classification
- Explainability Layer
- SOC Dashboard