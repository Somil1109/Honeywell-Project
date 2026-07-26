import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def preprocess_data(input_path, output_path):
    print(f"Loading raw data from {input_path}...")
    
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}. Make sure to run generator.py first.")
        return

    #Temporal Feature Extraction
    print("Extracting temporal features...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    #Categorical Encoding
    print("Encoding categorical variables...")
    categorical_cols = [
        'entity_id', 'entity_type', 'source_ip', 'resource_accessed', 
        'auth_method', 'command_sequence', 'device_fingerprint'
    ]
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    #Scaling Numerical Features
    print("Scaling numerical features...")
    num_cols = ['session_duration', 'hour_of_day', 'day_of_week']
    scaler = StandardScaler()
    
    # Similarly, this scaler would be saved for inference
    df[[c + '_scaled' for c in num_cols]] = scaler.fit_transform(df[num_cols])

    #Final Feature Selection
    feature_cols = [c + '_encoded' for c in categorical_cols] + \
                   [c + '_scaled' for c in num_cols] + \
                   ['label', 'timestamp']
                   
    processed_df = df[feature_cols].copy()

    # Sort by timestamp to ensure the sequence remains intact for the LSTM later
    processed_df = processed_df.sort_values(by='timestamp').reset_index(drop=True)

    #Save the Processed Data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    
    print(f"Preprocessing complete. Processed data saved to {output_path}")
    print(f"Final feature shape: {processed_df.shape}")
    
    return processed_df

if __name__ == "__main__":
    INPUT_FILE = "data/raw/access_logs_v1.csv"
    OUTPUT_FILE = "data/processed/features_v1.csv"
    
    preprocess_data(INPUT_FILE, OUTPUT_FILE)