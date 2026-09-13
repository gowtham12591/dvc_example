"""
train.py - Model Training Stage
=================================
This script loads preprocessed data, trains a Gradient Boosting model,
and saves the model + metrics.

Usage:
    python src/train.py

DVC Pipeline Stage:
    This is Stage 2 in our ML pipeline (preprocess -> train -> evaluate)
"""

import pandas as pd
import json
import os
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train():
    """Train a Gradient Boosting model and save model + metrics."""
    
    print("=" * 50)
    print("STAGE 2: TRAINING")
    print("=" * 50)
    
    # Step 1: Load preprocessed data
    print("\n[1/4] Loading preprocessed data ...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    print(f"       Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    
    # Step 2: Define hyperparameters
    print("[2/4] Setting hyperparameters ...")
    params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "random_state": 42
    }
    for k, v in params.items():
        print(f"       {k}: {v}")
    
    # Step 3: Train the model
    print("[3/4] Training Gradient Boosting Classifier ...")
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate on training data
    y_train_pred = model.predict(X_train)
    train_metrics = {
        "train_accuracy": round(accuracy_score(y_train, y_train_pred), 4),
        "train_precision": round(precision_score(y_train, y_train_pred), 4),
        "train_recall": round(recall_score(y_train, y_train_pred), 4),
        "train_f1": round(f1_score(y_train, y_train_pred), 4),
    }
    
    print("       Training Metrics:")
    for k, v in train_metrics.items():
        print(f"         {k}: {v}")
    
    # Step 4: Save model and metrics
    print("[4/4] Saving model and metrics ...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    
    # Save metrics as JSON (DVC tracks this file)
    with open("metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)
    
    # Save hyperparameters for reference
    with open("params.json", "w") as f:
        json.dump(params, f, indent=2)
    
    print("\nTraining complete!")
    print(f"  Saved: models/model.pkl")
    print(f"  Saved: metrics.json")
    print(f"  Saved: params.json")


if __name__ == "__main__":
    train()
