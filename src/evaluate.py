"""
evaluate.py - Model Evaluation Stage
======================================
This script loads the trained model and test data, evaluates performance,
saves evaluation metrics and a confusion matrix plot.

Usage:
    python src/evaluate.py

DVC Pipeline Stage:
    This is Stage 3 in our ML pipeline (preprocess -> train -> evaluate)
"""

import pandas as pd
import json
import os
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)


def evaluate():
    """Load the trained model, evaluate on test data, and save results."""
    
    print("=" * 50)
    print("STAGE 3: EVALUATION")
    print("=" * 50)
    
    # Step 1: Load test data
    print("\n[1/4] Loading test data ...")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    print(f"       Test samples: {X_test.shape[0]}")
    
    # Step 2: Load trained model
    print("[2/4] Loading trained model from models/model.pkl ...")
    model = joblib.load("models/model.pkl")
    print(f"       Model type: {type(model).__name__}")
    
    # Step 3: Make predictions and calculate metrics
    print("[3/4] Evaluating model on test data ...")
    y_pred = model.predict(X_test)
    
    eval_metrics = {
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_precision": round(precision_score(y_test, y_pred), 4),
        "test_recall": round(recall_score(y_test, y_pred), 4),
        "test_f1": round(f1_score(y_test, y_pred), 4),
    }
    
    print("\n       Test Metrics:")
    for k, v in eval_metrics.items():
        print(f"         {k}: {v}")
    
    print("\n       Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    
    # Step 4: Save evaluation results
    print("[4/4] Saving evaluation results ...")
    
    # Save eval metrics to a separate file (DVC tracks this)
    with open("eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)
    
    # Save confusion matrix plot
    os.makedirs("plots", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=["No Churn", "Churn"]
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title("Customer Churn - Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png", dpi=150)
    plt.close()
    
    print("\nEvaluation complete!")
    print(f"  Saved: metrics.json (updated with test metrics)")
    print(f"  Saved: plots/confusion_matrix.png")


if __name__ == "__main__":
    evaluate()
