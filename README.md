# AI-Powered Behavioral Anomaly Detection

**GitHub Repository:** https://github.com/Somil1109/Honeywell-Project

## Project Overview
This project is an AI/ML system designed to model "normal" access behavior for users and devices, detecting intrusions in near real-time. It handles extreme class imbalance, concept drift, and cold-start entities, while providing deep SHAP-based explainability for SOC analysts.

## Architecture
- **Data Pipeline:** Synthetic telemetry generator with injected attack taxonomies (brute force, impossible travel, lateral movement, credential stuffing, low and slow, device spoofing, insider drift).
- **Backend (FastAPI):** Hosts the Isolation Forest (baseline profiler), PyTorch LSTM (sequence detector), and SHAP explainer.
- **Frontend (Streamlit):** Enterprise SOC dashboard with live telemetry and entity deep-dive capabilities.

## Setup Instructions
1. **Clone the Repository:** `git clone https://github.com/Somil1109/Honeywell-Project.git`
2. **Navigate to Directory:** `cd Honeywell-Project`
3. **Create Virtual Environment:** `python -m venv venv`
4. **Activate:** `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. **Install Dependencies:** `pip install -r requirements.txt`
6. **Run the API:** `uvicorn src.api.main:app --reload`
7. **Run the Dashboard (in a new terminal):** `streamlit run dashboard/app.py`

## Deliverables
- Synthetic Data Generator
- Baseline Profiling Model
- Sequence Detection Model
- Anomaly Classification
- Explainability Layer
- SOC Dashboard
- Final Report