"""
Data preprocessing for hospital readmission prediction.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_data(filepath):
    """
    Load data from CSV file.
    
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

def preprocess_data(data):
    """
    Preprocess data for modeling.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw patient data
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed data
    """
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # Convert date strings to datetime objects
    df['admission_date'] = pd.to_datetime(df['admission_date'])
    df['discharge_date'] = pd.to_datetime(df['discharge_date'])
    
    # Extract month and day of week from admission date
    df['admission_month'] = df['admission_date'].dt.month
    df['admission_day_of_week'] = df['admission_date'].dt.dayofweek
    
    # Create age groups
    bins = [0, 40, 65, 75, 100]
    labels = ['<40', '40-65', '65-75', '>75']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['gender', 'age_group', 'primary_diagnosis', 'discharge_disposition'], drop_first=True)
    
    # Create interaction features
    df['age_heart_failure'] = df['age'] * df['heart_failure']
    df['previous_emergency'] = df['previous_admissions'] * df['emergency_admission']
    
    # Drop columns not needed for modeling
    columns_to_drop = ['patient_id', 'admission_date', 'discharge_date']
    
    # For prediction data, we don't have the target variables
    if 'readmission_30d' in df.columns:
        # Keep days_to_readmission only for readmitted patients
        df.loc[df['readmission_30d'] == 0, 'days_to_readmission'] = np.nan
    else:
        # For prediction data, we don't drop the ID
        columns_to_drop = columns_to_drop[1:]
    
    df = df.drop(columns=columns_to_drop)
    
    return df

def process_and_save_datasets():
    """
    Process and save all datasets (train, validation, test).
    """
    # Create directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Process training data
    train_data = load_data('data/raw/train.csv')
    processed_train = preprocess_data(train_data)
    processed_train.to_csv('data/processed/train_processed.csv', index=False)
    print(f"Processed training data saved ({len(processed_train)} samples)")
    
    # Process validation data
    val_data = load_data('data/raw/val.csv')
    processed_val = preprocess_data(val_data)
    processed_val.to_csv('data/processed/val_processed.csv', index=False)
    print(f"Processed validation data saved ({len(processed_val)} samples)")
    
    # Process test data
    test_data = load_data('data/raw/test.csv')
    processed_test = preprocess_data(test_data)
    processed_test.to_csv('data/processed/test_processed.csv', index=False)
    print(f"Processed test data saved ({len(processed_test)} samples)")

if __name__ == "__main__":
    process_and_save_datasets()
