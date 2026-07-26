import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# initialize deterministic generation
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

NUM_NORMAL_EVENTS = 5000
ENTITY_TYPES = ['user', 'service_account', 'edge_device']
AUTH_METHODS = ['password', 'token', 'certificate', 'biometric']
RESOURCES = ['/api/v1/data', '/login', '/admin/settings', '/db/query', '/metrics', '/hr/payroll', '/vault/secrets']
DEVICES = [
    'OS:Ubuntu|MAC:00-14-22-01-23-45', 
    'OS:Windows|MAC:F0-18-98-33-22-11', 
    'OS:Firmware1.2|MAC:00-1A-2B-3C-4D-5E',
    'OS:macOS|MAC:A1-B2-C3-D4-E5-F6'
]

def generate_normal_data(num_records):
    data = []
    start_time = datetime.now() - timedelta(days=30)
    
    for i in range(num_records):
        entity_id = f"ENT_{random.randint(100, 150)}"
        record = {
            'entity_id': entity_id,
            'entity_type': random.choice(ENTITY_TYPES),
            'timestamp': start_time + timedelta(minutes=random.randint(1, 43200)),
            'source_ip': fake.ipv4_private(),
            'resource_accessed': random.choice(RESOURCES[:4]), 
            'auth_method': random.choice(AUTH_METHODS),
            'session_duration': round(random.uniform(5.0, 3600.0), 2),
            'command_sequence': 'READ, GET' if random.random() > 0.5 else 'POST, UPDATE',
            'device_fingerprint': random.choice(DEVICES[:3]),
            'label': 'normal'
        }
        data.append(record)
    return data

def inject_impossible_travel(start_time):
    return [
        {
            'entity_id': 'ENT_110', 'entity_type': 'user', 'timestamp': start_time, 
            'source_ip': '198.51.100.14', 'resource_accessed': '/login', 
            'auth_method': 'password', 'session_duration': 120.0, 
            'command_sequence': 'LOGIN, GET', 'device_fingerprint': DEVICES[1], 'label': 'impossible_travel'
        },
        {
            'entity_id': 'ENT_110', 'entity_type': 'user', 'timestamp': start_time + timedelta(minutes=15), 
            'source_ip': '203.0.113.45', 'resource_accessed': '/login', 
            'auth_method': 'password', 'session_duration': 45.0, 
            'command_sequence': 'LOGIN, GET', 'device_fingerprint': DEVICES[1], 'label': 'impossible_travel'
        }
    ]

def inject_lateral_movement(start_time):
    events = []
    for i in range(5):
        events.append({
            'entity_id': 'ENT_125', 'entity_type': 'service_account', 
            'timestamp': start_time + timedelta(minutes=i*2), 
            'source_ip': fake.ipv4_private(), 'resource_accessed': RESOURCES[i % len(RESOURCES)], 
            'auth_method': 'token', 'session_duration': 15.0, 
            'command_sequence': 'SCAN, LIST', 'device_fingerprint': DEVICES[0], 'label': 'lateral_movement'
        })
    return events

def inject_device_spoofing(start_time):
    return [{
        'entity_id': 'ENT_130', 'entity_type': 'edge_device', 
        'timestamp': start_time, 'source_ip': fake.ipv4_private(), 
        'resource_accessed': '/api/v1/data', 'auth_method': 'certificate', 
        'session_duration': 300.0, 'command_sequence': 'SYNC, PUSH', 
        'device_fingerprint': 'OS:Unknown|MAC:FF-FF-FF-FF-FF-FF', 'label': 'device_spoofing'
    }]

def inject_low_and_slow(start_time):
    events = []
    for i in range(10):
        events.append({
            'entity_id': 'ENT_140', 'entity_type': 'user', 
            'timestamp': start_time + timedelta(days=i, hours=2),
            'source_ip': fake.ipv4_private(), 'resource_accessed': '/vault/secrets', 
            'auth_method': 'password', 'session_duration': 5.0, 
            'command_sequence': 'EXPORT, DOWNLOAD', 'device_fingerprint': DEVICES[1], 'label': 'low_and_slow'
        })
    return events

def inject_credential_stuffing(start_time):
    events = []
    malicious_ip = fake.ipv4_public()
    for i in range(25):
        events.append({
            'entity_id': f"ENT_{random.randint(900, 999)}",
            'entity_type': 'user', 
            'timestamp': start_time + timedelta(seconds=i*2), 
            'source_ip': malicious_ip, 
            'resource_accessed': '/login', 
            'auth_method': 'password', 
            'session_duration': 0.5, 
            'command_sequence': 'LOGIN_FAILED', 
            'device_fingerprint': 'OS:Unknown|MAC:00-00-00-00-00-00', 
            'label': 'credential_stuffing'
        })
    return events

def inject_insider_drift(start_time):
    events = []
    for i in range(15):
        events.append({
            'entity_id': 'ENT_105', 
            'entity_type': 'user', 
            'timestamp': start_time + timedelta(days=i), 
            'source_ip': fake.ipv4_private(), 
            'resource_accessed': RESOURCES[min(i // 3, len(RESOURCES)-1)], 
            'auth_method': 'token', 
            'session_duration': 120.0 + (i * 10), 
            'command_sequence': 'READ, GET, EXPORT', 
            'device_fingerprint': DEVICES[0], 
            'label': 'insider_drift'
        })
    return events

if __name__ == "__main__":
    print("Generating normal baseline data...")
    data = generate_normal_data(NUM_NORMAL_EVENTS)

    print("Injecting attack taxonomies...")
    base_time = datetime.now() - timedelta(days=15)
    data.extend(inject_impossible_travel(base_time))
    data.extend(inject_lateral_movement(base_time + timedelta(days=1)))
    data.extend(inject_device_spoofing(base_time + timedelta(days=2)))
    data.extend(inject_low_and_slow(base_time + timedelta(days=3)))
    data.extend(inject_credential_stuffing(base_time + timedelta(days=4)))
    data.extend(inject_insider_drift(base_time + timedelta(days=5)))

    df = pd.DataFrame(data)
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    output_path = 'data/raw/access_logs_v1.csv'
    df.to_csv(output_path, index=False)

    print(f"Total records generated: {len(df)}")
    print(f"Anomaly counts:\n{df['label'].value_counts()}")
    print(f"Data saved successfully to {output_path}")