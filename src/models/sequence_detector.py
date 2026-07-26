import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

def create_sequences(data, labels, time_steps=5):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(labels[i + time_steps])
    return np.array(X), np.array(y)

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

def train_sequence_model(input_path, model_output_path, encoder_output_path):
    print(f"Loading multi-class processed data from {input_path}...")
    df = pd.read_csv(input_path)

    features = [col for col in df.columns if col.endswith('_encoded') or col.endswith('_scaled')]
    
    # Encode multi-class text labels
    le = LabelEncoder()
    df['label_int'] = le.fit_transform(df['label'])
    
    # Save the label encoder mapping for API decoding
    os.makedirs(os.path.dirname(encoder_output_path), exist_ok=True)
    joblib.dump(le, encoder_output_path)
    print(f"Saved label mapping classes: {list(le.classes_)}")
    
    X_data = df[features].values
    y_data = df['label_int'].values

    print("Generating time-series sequences (window=5)...")
    time_steps = 5
    X_seq, y_seq = create_sequences(X_data, y_data, time_steps)

    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_classes = len(le.classes_)
    input_size = X_seq.shape[2]

    print(f"Building PyTorch Multi-Class LSTM model for {num_classes} classes...")
    model = MultiClassLSTMDetector(input_size=input_size, hidden_size=64, num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training Multi-Class LSTM...")
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    torch.save(model.state_dict(), model_output_path)
    print(f"Multi-class sequence model saved to {model_output_path}")

if __name__ == "__main__":
    INPUT_FILE = "data/processed/features_v1.csv"
    MODEL_OUTPUT_FILE = "src/models/saved_models/lstm_detector_v1.pth"
    ENCODER_OUTPUT_FILE = "src/models/saved_models/label_encoder.pkl"
    
    train_sequence_model(INPUT_FILE, MODEL_OUTPUT_FILE, ENCODER_OUTPUT_FILE)