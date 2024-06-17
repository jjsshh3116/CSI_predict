import pandas as pd
import joblib
from csi import *
from scapy.all import *
import os
import numpy as np

# Function to extract features from CSI data
def extract_features(data):
    features = []
    for column in data.columns:
        if data[column].dtype in [np.float64, np.int64]:  # Only process numeric columns
            features.append(data[column].mean())
            features.append(data[column].std())
            features.append(data[column].max())
            features.append(data[column].min())
    return features

# pcap to csv file
save_dir = '.'
pcap_file = '/Users/jeongsehyeon/Downloads/people/0person1.pcap' # 예측할 pcap파일 경로와 파일명 & 이 경로에 csv파일이 저장.

df = pcap_to_df(os.path.join(save_dir, pcap_file))
csv_fname = pcap_file.split('.')[0] + '.csv'
df.to_csv(os.path.join(save_dir, csv_fname), index=False)
print(f'Save {csv_fname}')

# Load the trained model and scaler
clf = joblib.load('csi_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load new data (assume the new data is in a CSV file)
new_data = pd.read_csv(csv_fname)  # Replace with the actual path to your new CSV file

# Check and convert data types
for column in new_data.columns:
    try:
        new_data[column] = pd.to_numeric(new_data[column])
    except ValueError:
        new_data = new_data.drop(columns=[column])

# Extract features from the new data
new_features = extract_features(new_data)
new_features_df = pd.DataFrame([new_features], columns=[f'Sub {i} {stat}' for i in range(len(new_features)//4) for stat in ['mean', 'std', 'max', 'min']])

# Scale the features
new_features_scaled = scaler.transform(new_features_df)

# Make predictions using the loaded model
prediction = clf.predict(new_features_scaled)

# Print the prediction (assuming binary classification: 0 for no person, 1 for person present)
if prediction[0] == 1:
    print("Person present")
else:
    print("No person present")