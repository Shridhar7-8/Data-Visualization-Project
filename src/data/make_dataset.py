"""
Script to download or generate data for hospital readmission prediction.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/features', exist_ok=True)


def split_and_save_data(data, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into train, validation, and test sets and save to disk.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Patient data
    test_size : float
        Proportion of data to use for testing
    val_size : float
        Proportion of training data to use for validation
    random_state : int
        Random seed for reproducibility
    """
    # First split: training + validation vs test
    train_val, test = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data['readmitted']
    )
    
    # Second split: training vs validation
    # val_size_adjusted is relative to the size of train_val
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_size_adjusted, random_state=random_state, stratify=train_val['readmitted']
    )
    
    # Save to disk
    train.to_csv('data/raw/train.csv', index=False)
    val.to_csv('data/raw/val.csv', index=False)
    test.to_csv('data/raw/test.csv', index=False)
    
    print(f"Data split and saved to disk:")
    print(f"  Training set: {len(train)} samples")
    print(f"  Validation set: {len(val)} samples")
    print(f"  Test set: {len(test)} samples")

if __name__ == "__main__":
    print("Generating synthetic patient data...")
    data = pd.read_csv(r'E:\Data-Visualization-Project\src\data\data\raw\final_selected_data.csv')
    
    # Save raw data
    print(f"final data has ({len(data)} samples)")
    
    # Split and save data
    split_and_save_data(data)
