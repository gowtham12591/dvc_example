"""
Stage 3 of DVC pipeline: Model Evaluation & Drift Detection.
Evaluates the trained model on test data and creates a confusion matrix.
Demonstrates drift detection by comparing train and test feature distributions.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def evaluate():
    print("Starting Evaluation Stage...")
    
    # Load model and data
    print("Loading model and test data...")
    model = joblib.load('models/model.pkl')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    # Predict and evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/eval_metrics.json', 'w') as f:
        json.dump(metrics, f)
    print("Evaluation metrics saved.")
    
    # Confusion matrix
    print("Generating confusion matrix plot...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'], 
                yticklabels=['No Churn', 'Churn'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved to plots/confusion_matrix.png.")
    
    # Drift Detection Demo
    print("Running Drift Detection Demo...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    
    num_cols = ['tenure', 'monthly_charges', 'total_charges', 'num_tickets']
    drift_report = []
    
    for col in num_cols:
        if col in X_train.columns and col in X_test.columns:
            train_mean = X_train[col].mean()
            test_mean = X_test[col].mean()
            # For standardized features (mean ~0, std ~1), measure shift relative to standard deviation
            train_std = X_train[col].std()
            eps = 1e-8
            drift_pct = abs((test_mean - train_mean) / (train_std + eps)) * 100
            
            drift_status = 'DRIFT DETECTED' if drift_pct > 25 else 'OK'
            
            drift_report.append({
                'column': col,
                'train_mean': float(train_mean),
                'test_mean': float(test_mean),
                'drift_pct': float(drift_pct),
                'status': drift_status
            })
            
    # Print drift report table
    print("\n--- Drift Report ---")
    print(f"{'Column':<16} | {'Train Mean':<10} | {'Test Mean':<10} | {'Drift %':<8} | {'Status'}")
    print("-" * 65)
    for row in drift_report:
        print(f"{row['column']:<16} | {row['train_mean']:<10.4f} | {row['test_mean']:<10.4f} | {row['drift_pct']:<8.2f} | {row['status']}")
    print("-" * 65)
    
    with open('metrics/drift_report.json', 'w') as f:
        json.dump(drift_report, f, indent=4)
    print("\nDrift report saved to metrics/drift_report.json.")
    
    print("Evaluation Stage completed successfully.")

if __name__ == '__main__':
    evaluate()
