"""
Stage 1 of the DVC pipeline: Preprocess the raw data.
Reads params, cleans data, performs one-hot encoding, train-test split, and scaling.
Outputs processed data, scaler, and feature columns list.
"""

import os
import yaml
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess():
    print("Starting Preprocessing Stage...")
    
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)['preprocess']
        
    print("Parameters loaded successfully.")

    # Load data
    data_path = 'data/churn_data.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Drop customer_id
    if 'customer_id' in df.columns:
        df = df.drop('customer_id', axis=1)
        
    # One-hot encoding
    print("Performing one-hot encoding...")
    df = pd.get_dummies(df, drop_first=True)

    # Split X and y
    print("Splitting features and target...")
    X = df.drop('churn', axis=1)
    y = df['churn']

    # Train-test split
    print("Performing train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['test_size'], 
        random_state=params['random_state'], 
        stratify=y
    )
    
    # Save feature columns
    feature_columns = list(X_train.columns)
    os.makedirs('models', exist_ok=True)
    with open('models/feature_columns.json', 'w') as f:
        json.dump(feature_columns, f)
    print("Feature columns saved.")

    # Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to save as CSV
    X_train = pd.DataFrame(X_train_scaled, columns=feature_columns)
    X_test = pd.DataFrame(X_test_scaled, columns=feature_columns)

    # Save processed data
    print("Saving processed data...")
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    # Save scaler
    joblib.dump(scaler, 'models/scalar.pkl')
    print("Scaler saved to models/scaler.pkl.")
    print("Preprocessing Stage completed successfully.")

if __name__ == '__main__':
    preprocess()
