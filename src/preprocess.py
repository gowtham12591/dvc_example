"""
preprocess.py - Data Preprocessing Stage
=========================================
This script reads the raw churn dataset, performs preprocessing
(drop ID, encode categoricals, train-test split, scale features),
and saves the processed splits to data/processed/.

Usage:
    python src/preprocess.py

DVC Pipeline Stage:
    This is Stage 1 in our ML pipeline (preprocess -> train -> evaluate)
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def preprocess():
    """Load raw data, preprocess it, and save train/test splits."""
    
    print("=" * 50)
    print("STAGE 1: PREPROCESSING")
    print("=" * 50)
    
    # Step 1: Load raw data
    print("\n[1/5] Loading raw data from data/churn_data.csv ...")
    df = pd.read_csv("data/churn_data.csv")
    print(f"       Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 2: Drop customer_id (not useful for prediction)
    print("[2/5] Dropping 'customer_id' column ...")
    df = df.drop("customer_id", axis=1)
    
    # Step 3: One-hot encode categorical variables
    print("[3/5] One-hot encoding categorical variables ...")
    df = pd.get_dummies(df, drop_first=True)
    print(f"       After encoding: {df.shape[1]} columns")
    
    # Step 4: Split into features (X) and target (y), then train/test
    print("[4/5] Splitting into train (80%) and test (20%) sets ...")
    X = df.drop("churn", axis=1)
    y = df["churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"       Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    
    # Step 5: Scale numerical features
    print("[5/5] Scaling features with StandardScaler ...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )
    
    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
    X_test_scaled.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    
    # Save the scaler for later use in production
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    
    print("\nPreprocessing complete!")
    print(f"  Saved: data/processed/X_train.csv ({X_train_scaled.shape})")
    print(f"  Saved: data/processed/X_test.csv  ({X_test_scaled.shape})")
    print(f"  Saved: data/processed/y_train.csv ({y_train.shape[0]},)")
    print(f"  Saved: data/processed/y_test.csv  ({y_test.shape[0]},)")
    print(f"  Saved: models/scaler.pkl")


if __name__ == "__main__":
    preprocess()
