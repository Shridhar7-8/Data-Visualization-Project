"""
Split preprocessed data for hospital readmission prediction.
"""
import pandas as pd
import os

def load_processed_data(filepath):
    """
    Load processed data from CSV file.
    """
    return pd.read_csv(filepath)

def split_features_and_target(data):
    """
    Split data into features and target.
    """
    if 'readmitted' in data.columns:
        y = data['readmitted']
        X = data.drop(columns=['readmitted'], errors='ignore')
    else:
        y = None
        X = data
    return X, y

def process_and_save_splits():
    """
    Process and save splits for all datasets.
    """
    os.makedirs('data/features', exist_ok=True)

    # Train
    train_data = load_processed_data(r'E:\Data-Visualization-Project\src\data\data\raw\train.csv')
    X_train, y_train = split_features_and_target(train_data)
    X_train.to_csv('data/features/X_train.csv', index=False)
    if y_train is not None:
        y_train.to_csv('data/features/y_train.csv', index=False)
    print(f"Saved training split ({len(X_train)} samples)")

    # Validation
    val_data = load_processed_data(r'E:\Data-Visualization-Project\src\data\data\raw\val.csv')
    X_val, y_val = split_features_and_target(val_data)
    X_val.to_csv('data/features/X_val.csv', index=False)
    if y_val is not None:
        y_val.to_csv('data/features/y_val.csv', index=False)
    print(f"Saved validation split ({len(X_val)} samples)")

    # Test
    test_data = load_processed_data(r'E:\Data-Visualization-Project\src\data\data\raw\test.csv')
    X_test, y_test = split_features_and_target(test_data)
    X_test.to_csv('data/features/X_test.csv', index=False)
    if y_test is not None:
        y_test.to_csv('data/features/y_test.csv', index=False)
    print(f"Saved test split ({len(X_test)} samples)")

if __name__ == "__main__":
    process_and_save_splits()
