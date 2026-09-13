"""
Stage 2 of DVC pipeline: Model Training with MLflow tracking.
Trains a GradientBoostingClassifier using parameters from params.yaml.
Logs metrics and the model to MLflow, and saves the model locally.
"""

import os
import yaml
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train():
    print("Starting Training Stage...")
    
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)['train']
        
    print("Parameters loaded successfully.")
    
    # Load processed data
    print("Loading processed training data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    
    # MLflow tracking
    print("Setting up MLflow tracking...")
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('churn-prediction-session2')
    
    run_name = f'GBM-{params["n_estimators"]}trees'
    with mlflow.start_run(run_name=run_name):
        print(f"Started MLflow run: {run_name}")
        
        # Log parameters and tags
        mlflow.log_params(params)
        mlflow.set_tag('developer', 'Gowtham')
        mlflow.set_tag('model_type', params['model_type'])
        
        # Train model
        print("Training GradientBoostingClassifier...")
        model = GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            random_state=params['random_state']
        )
        model.fit(X_train, y_train)
        
        # Predict on training data
        print("Evaluating on training data...")
        y_pred = model.predict(X_train)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_train, y_pred),
            'precision': precision_score(y_train, y_pred),
            'recall': recall_score(y_train, y_pred),
            'f1_score': f1_score(y_train, y_pred)
        }
        
        # Log to MLflow
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, 'model')
        
        # Save locally
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/model.pkl')
        print("Model saved to models/model.pkl.")
        
        os.makedirs('metrics', exist_ok=True)
        with open('metrics/train_metrics.json', 'w') as f:
            json.dump(metrics, f)
        print("Training metrics saved to metrics/train_metrics.json.")
        
    print("Training Stage completed successfully.")

if __name__ == '__main__':
    train()
