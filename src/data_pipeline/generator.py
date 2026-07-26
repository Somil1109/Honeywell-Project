import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# Initialize Faker
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# Define constants
NUM_NORMAL_EVENTS = 1000
NUM_ANOMALY_EVENTS = 50
ENTITY_TYPES = ['user', 'service_account', 'edge_device']
AUTH_METHODS = ['password', 'token', 'certificate', 'biometric']
RESOURCES = ['/api/v1/data', '/login', '/admin/settings', '/db/query', '/metrics']
DEVICES = [
    'OS:Ubuntu|MAC:00-14-22-01-23-45', 
    'OS:Windows|MAC:F0-18-98-33-22-11', 
    'OS:Firmware1.2|MAC:00-1A-2B-3C-4D-5E'
]

def generate_normal_data(num_records):
    data = []
    start_time = datetime.now() - timedelta(days=30)
    
    for _ in range(num_records):
        record = {
            'entity_id': f"ENT_{random.randint(100, 200)}",
            'entity_type': random.choice(ENTITY_TYPES),
            'timestamp': start_time + timedelta(minutes=random.randint(1, 43200)),
            'source_ip': fake.ipv4_private(),
            'resource_accessed': random.choice(RESOURCES),
            'auth_method': random.choice(AUTH_METHODS),
            'session_duration': round(random.uniform(5.0, 3600.0), 2),
            'command_sequence': 'READ, GET' if random.random() > 0.5 else 'POST, UPDATE',
            'device_fingerprint': random.choice(DEVICES),
            'label': 'normal'
        }
        data.append(record)
    return data

def inject_brute_force(data):
    # Simulating a brute force attack: rapid failed attempts from one IP
    target_entity = "ENT_150"
    attacker_ip = fake.ipv4_public()
    attack_time = datetime.now() - timedelta(days=5)
    
    for i in range(NUM_ANOMALY_EVENTS):
        record = {
            'entity_id': target_entity,
            'entity_type': 'user',
            'timestamp': attack_time + timedelta(seconds=i*2), # 2 seconds apart
            'source_ip': attacker_ip,
            'resource_accessed': '/login',
            'auth_method': 'password',
            'session_duration': 0.0,
            'command_sequence': 'FAILED_LOGIN',
            'device_fingerprint': 'OS:Kali|MAC:AA-BB-CC-DD-EE-FF',
            'label': 'brute_force'
        }
        data.append(record)
    return data

# Generate and combine data
baseline_data = generate_normal_data(NUM_NORMAL_EVENTS)
anomalous_data = inject_brute_force(baseline_data)

# Create DataFrame
df = pd.DataFrame(anomalous_data)

# Sort by timestamp to simulate a realistic log stream
df = df.sort_values(by='timestamp').reset_index(drop=True)

print(df.head())
print(f"\nTotal records generated: {len(df)}")
print(f"Anomaly counts:\n{df['label'].value_counts()}")

# Save to the raw data directory
df.to_csv('data/raw/access_logs_v1.csv', index=False)
print("Data saved successfully to data/raw/access_logs_v1.csv")