"""
Feature engineering for hospital readmission prediction.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_processed_data(filepath):
    """
    Load processed data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    return pd.read_csv(filepath)

def engineer_features(data, is_training=True, scaler=None):
    """
    Engineer features for modeling.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Processed patient data
    is_training : bool
        Whether this is training data (to fit the scaler)
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for feature scaling
        
    Returns:
    --------
    tuple
        (X, y) for training data or (X, None) for prediction data
    """
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # Separate features and target
    if 'readmission_30d' in df.columns:
        y = df['readmission_30d']
        X = df.drop(columns=['readmission_30d', 'days_to_readmission'])
    else:
        y = None
        X = df
    
    # Get numerical columns for scaling
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Scale numerical features
    if is_training:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        # Save the scaler for later use
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
    else:
        if scaler is None:
            scaler = joblib.load('models/scaler.pkl')
        X[numerical_cols] = scaler.transform(X[numerical_cols])
    
    return X, y, scaler

def process_and_save_features():
    """
    Process and save features for all datasets.
    """
    # Create directory if it doesn't exist
    os.makedirs('data/features', exist_ok=True)
    
    # Process training data
    train_data = load_processed_data('data/processed/train_processed.csv')
    X_train, y_train, scaler = engineer_features(train_data, is_training=True)
    
    # Save training features
    X_train.to_csv('data/features/X_train.csv', index=False)
    y_train.to_csv('data/features/y_train.csv', index=False)
    print(f"Training features saved ({len(X_train)} samples)")
    
    # Process validation data
    val_data = load_processed_data('data/processed/val_processed.csv')
    X_val, y_val, _ = engineer_features(val_data, is_training=False, scaler=scaler)
    
    # Save validation features
    X_val.to_csv('data/features/X_val.csv', index=False)
    y_val.to_csv('data/features/y_val.csv', index=False)
    print(f"Validation features saved ({len(X_val)} samples)")
    
    # Process test data
    test_data = load_processed_data('data/processed/test_processed.csv')
    X_test, y_test, _ = engineer_features(test_data, is_training=False, scaler=scaler)
    
    # Save test features
    X_test.to_csv('data/features/X_test.csv', index=False)
    y_test.to_csv('data/features/y_test.csv', index=False)
    print(f"Test features saved ({len(X_test)} samples)")

if __name__ == "__main__":
    process_and_save_features()
