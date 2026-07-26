import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Enterprise SOC Dashboard", layout="wide")

if 'alert_queue' not in st.session_state:
    st.session_state.alert_queue = []
if 'risk_history' not in st.session_state:
    st.session_state.risk_history = []
if 'total_events' not in st.session_state:
    st.session_state.total_events = 0

st.title("Enterprise SOC Behavioral Anomaly Dashboard")
st.markdown("---")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/features_v1.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Data file not found. Ensure features_v1.csv exists.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Events Processed", st.session_state.total_events)
col2.metric("Active High-Risk Alerts", len([a for a in st.session_state.alert_queue if a['Risk Score'] > 75]))
col3.metric("Total Anomalies Detected", len(st.session_state.alert_queue))
col4.metric("System Status", "Monitoring" if st.session_state.total_events > 0 else "Idle")

st.markdown("---")

tab1, tab2 = st.tabs(["Live Threat Stream", "Entity Deep-Dive"])

with tab1:
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        st.subheader("Ingestion Engine")
        if st.button("Simulate Incoming Access Event", use_container_width=True):
            sample = df.sample(1).iloc[0]
            st.session_state.total_events += 1
            
            payload = {
                "session_duration_scaled": float(sample["session_duration_scaled"]),
                "hour_of_day_scaled": float(sample["hour_of_day_scaled"]),
                "day_of_week_scaled": float(sample["day_of_week_scaled"]),
                "entity_id_encoded": int(sample["entity_id_encoded"]),
                "entity_type_encoded": int(sample["entity_type_encoded"]),
                "source_ip_encoded": int(sample["source_ip_encoded"]),
                "resource_accessed_encoded": int(sample["resource_accessed_encoded"]),
                "auth_method_encoded": int(sample["auth_method_encoded"]),
                "command_sequence_encoded": int(sample["command_sequence_encoded"]),
                "device_fingerprint_encoded": int(sample["device_fingerprint_encoded"])
            }
            
            try:
                response = requests.post("http://127.0.0.1:8000/analyze_event", json=payload)
                result = response.json()
                
                st.session_state.risk_history.append({
                    "time": datetime.now(),
                    "entity": payload["entity_id_encoded"],
                    "risk_score": result["risk_score"]
                })
                
                if result["is_anomalous"]:
                    formatted_class = result["classification"].replace("_", " ").title()
                    alert = {
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "Entity ID": payload["entity_id_encoded"],
                        "Risk Score": result["risk_score"],
                        "Classification": formatted_class,
                        "Cold Start": result.get("is_cold_start", False),
                        "SHAP Explainability Factors": ", ".join(result["contributing_factors"])
                    }
                    st.session_state.alert_queue.insert(0, alert) 
                    st.error(f"Threat Detected: {formatted_class} (Risk: {result['risk_score']})")
                else:
                    msg = "Cold-start entity initialized." if result.get("is_cold_start") else "Normal behavior established."
                    st.success(f"Event analyzed: {msg}")
                    
            except requests.exceptions.ConnectionError:
                st.error("API Connection Error.")

    with right_col:
        st.subheader("Live Risk Telemetry")
        if len(st.session_state.risk_history) > 0:
            risk_df = pd.DataFrame(st.session_state.risk_history)
            fig = px.line(risk_df, x="time", y="risk_score", title="Risk Score Trend Over Time", range_y=[0, 100])
            fig.add_hline(y=65, line_dash="dash", line_color="red", annotation_text="Anomaly Threshold")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Awaiting telemetry data to generate risk charts.")

    st.markdown("---")
    st.subheader("Active Threat Queue")
    if st.session_state.alert_queue:
        st.dataframe(pd.DataFrame(st.session_state.alert_queue), use_container_width=True)
    else:
        st.info("No active alerts.")

with tab2:
    st.subheader("Entity History View")
    st.write("Search historical access logs and risk trajectory for a specific entity.")
    
    unique_entities = sorted(df['entity_id_encoded'].unique())
    selected_entity = st.selectbox("Select Encoded Entity ID to investigate:", unique_entities)
    
    if selected_entity is not None:
        entity_data = df[df['entity_id_encoded'] == selected_entity].copy()
        
        col_metrics1, col_metrics2 = st.columns(2)
        col_metrics1.metric("Total Historical Events", len(entity_data))
        
        st.write("Historical Access Log")
        display_cols = ['session_duration_scaled', 'hour_of_day_scaled', 'resource_accessed_encoded', 'label']
        st.dataframe(entity_data[display_cols], use_container_width=True)