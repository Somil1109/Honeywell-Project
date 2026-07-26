import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

def train_baseline_model(input_path, model_output_path):
    print(f"Loading processed data from {input_path}...")
    df = pd.read_csv(input_path)

    # dropping non-features before isolating
    feature_cols = [col for col in df.columns if col.endswith('_encoded') or col.endswith('_scaled')]
    X = df[feature_cols]

    print(f"Training Isolation Forest on {len(X)} samples...")
    # approx 5% anomaly rate based on injection
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X)

    # score and predict
    df['anomaly_score'] = model.decision_function(X)
    df['is_anomaly_pred'] = model.predict(X) 

    print("Baseline model training complete.")
    
    # persist model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

    # quick accuracy check against synthetic labels
    df['pred_label'] = df['is_anomaly_pred'].apply(lambda x: 'normal' if x == 1 else 'anomaly')
    df['actual_label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'anomaly')
    
    matches = (df['pred_label'] == df['actual_label']).sum()
    accuracy = matches / len(df)
    print(f"Sanity Check - Baseline Model Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    INPUT_FILE = "data/processed/features_v1.csv"
    MODEL_OUTPUT_FILE = "src/models/saved_models/isolation_forest_v1.pkl"
    
    train_baseline_model(INPUT_FILE, MODEL_OUTPUT_FILE)