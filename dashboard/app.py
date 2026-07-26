import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="SOC Analyst Dashboard", layout="wide")

# Initialize session state for the alert queue so it persists during interaction
if 'alert_queue' not in st.session_state:
    st.session_state.alert_queue = []

st.title("AI-Powered Behavioral Anomaly Detection")
st.markdown("---")

# Load data to simulate an incoming log stream
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/features_v1.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Data file not found. Ensure features_v1.csv exists.")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("System Controls")
    st.write("Simulate real-time log ingestion and ML evaluation.")
    
    if st.button("Simulate Incoming Access Event", use_container_width=True):
        # Pick a random event from our dataset to simulate a live stream
        sample = df.sample(1).iloc[0]
        
        # Prepare the exact payload the FastAPI backend expects
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
        
        # Send data to the local FastAPI server
        try:
            response = requests.post("http://127.0.0.1:8000/analyze_event", json=payload)
            result = response.json()
            
            # If the model flags it as an anomaly, push it to the SOC queue
            if result["is_anomalous"]:
                alert = {
                    "Entity ID (Encoded)": payload["entity_id_encoded"],
                    "Risk Score": result["risk_score"],
                    "Classification": result["classification"],
                    "Contributing Factors": ", ".join(result["contributing_factors"])
                }
                # Insert at the top of the queue
                st.session_state.alert_queue.insert(0, alert) 
                st.error(f"ALERT: Anomalous behavior detected! Risk Score: {result['risk_score']}")
            else:
                st.success("Event analyzed: Normal behavior established.")
                
        except requests.exceptions.ConnectionError:
            st.error("API Connection Error: Is the FastAPI server running on port 8000?")

with col2:
    st.subheader("Active Alert Queue")
    
    # Display the queue as an interactive table
    if st.session_state.alert_queue:
        alerts_df = pd.DataFrame(st.session_state.alert_queue)
        # Use Streamlit's native dataframe for sorting and scrolling
        st.dataframe(alerts_df, use_container_width=True)
    else:
        st.info("No active alerts. System monitoring nominal.")