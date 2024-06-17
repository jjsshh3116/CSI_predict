import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Load and process all CSV files
file_paths = glob.glob('data/csv/*.csv')  # Replace with the actual path to your CSV files
data_list = []

for file_path in file_paths:
    data = pd.read_csv(file_path)
    
    # Check and convert data types
    for column in data.columns:
        try:
            data[column] = pd.to_numeric(data[column])
        except ValueError:
            data = data.drop(columns=[column])
    
    features = extract_features(data)
    label = 1 if '1person' in file_path else 0
    features.append(label)
    data_list.append(features)

# Convert to DataFrame
feature_columns = []
for i in range((len(data_list[0]) - 1) // 4):
    feature_columns.extend([f'Sub {i} mean', f'Sub {i} std', f'Sub {i} max', f'Sub {i} min'])
feature_columns.append('label')

data_df = pd.DataFrame(data_list, columns=feature_columns)

# Split features and labels
X = data_df.drop(columns=['label'])
y = data_df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = clf.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Save the model if needed
joblib.dump(clf, 'csi_model.pkl')
joblib.dump(scaler, 'scaler.pkl')