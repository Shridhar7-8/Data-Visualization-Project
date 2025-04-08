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

def generate_synthetic_data(n_samples=1000, random_state=42):
    """
    Generate synthetic patient data for hospital readmission prediction.
    
    Parameters:
    -----------
    n_samples : int
        Number of patient records to generate
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        Synthetic patient data
    """
    np.random.seed(random_state)
    
    # Patient demographics
    patient_ids = [f'P-{1000+i}' for i in range(n_samples)]
    ages = np.random.normal(65, 15, n_samples).astype(int)
    ages = np.clip(ages, 18, 100)  # Clip ages to reasonable range
    genders = np.random.choice(['M', 'F'], size=n_samples)
    
    # Generate random admission dates within the last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years ago
    admission_dates = [start_date + timedelta(days=np.random.randint(0, 730)) for _ in range(n_samples)]
    
    # Length of stay between 1 and 30 days, with most stays being shorter
    length_of_stay = np.random.exponential(scale=5, size=n_samples).astype(int) + 1
    length_of_stay = np.clip(length_of_stay, 1, 30)
    
    # Calculate discharge dates
    discharge_dates = [admission_dates[i] + timedelta(days=int(length_of_stay[i])) for i in range(n_samples)]
    
    # Medical conditions (binary features)
    diabetes = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    heart_failure = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    copd = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    hypertension = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    renal_disease = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    
    # Number of previous admissions (0 to 10)
    previous_admissions = np.random.poisson(lam=1.5, size=n_samples)
    previous_admissions = np.clip(previous_admissions, 0, 10)
    
    # Medication count (1 to 15)
    medication_count = np.random.poisson(lam=5, size=n_samples) + 1
    medication_count = np.clip(medication_count, 1, 15)
    
    # Medication adherence score (0 to 1)
    medication_adherence = np.random.beta(5, 2, size=n_samples)
    
    # Emergency admission (binary)
    emergency_admission = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    # Discharge disposition (1: Home, 2: Skilled Nursing, 3: Home Health, 4: Other)
    discharge_disposition = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.7, 0.1, 0.15, 0.05])
    
    # Create factors that influence readmission
    readmission_factors = (
        0.2 * (ages > 75).astype(int) +
        0.15 * diabetes +
        0.25 * heart_failure +
        0.2 * copd +
        0.1 * renal_disease +
        0.05 * (previous_admissions > 2).astype(int) +
        0.15 * (medication_count > 8).astype(int) -
        0.3 * medication_adherence +
        0.1 * emergency_admission +
        0.1 * (discharge_disposition == 2).astype(int)
    )
    
    # Add some noise
    readmission_factors += np.random.normal(0, 0.1, n_samples)
    
    # Scale to 0-1 range
    readmission_factors = (readmission_factors - readmission_factors.min()) / (readmission_factors.max() - readmission_factors.min())
    
    # Generate readmission within 30 days (binary outcome)
    readmission_30d = (readmission_factors > 0.7).astype(int)
    
    # For readmitted patients, generate days to readmission
    days_to_readmission = np.zeros(n_samples)
    readmitted_indices = np.where(readmission_30d == 1)[0]
    days_to_readmission[readmitted_indices] = np.random.randint(1, 30, size=len(readmitted_indices))
    
    # Primary diagnosis
    diagnoses = ['Heart Failure', 'Pneumonia', 'COPD', 'Diabetes', 'Stroke', 
                'Urinary Tract Infection', 'Sepsis', 'Acute Myocardial Infarction',
                'Gastrointestinal Bleed', 'Renal Failure']
    
    primary_diagnosis = np.random.choice(diagnoses, size=n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'patient_id': patient_ids,
        'age': ages,
        'gender': genders,
        'admission_date': admission_dates,
        'discharge_date': discharge_dates,
        'length_of_stay': length_of_stay,
        'diabetes': diabetes,
        'heart_failure': heart_failure,
        'copd': copd,
        'hypertension': hypertension,
        'renal_disease': renal_disease,
        'previous_admissions': previous_admissions,
        'medication_count': medication_count,
        'medication_adherence': medication_adherence,
        'emergency_admission': emergency_admission,
        'discharge_disposition': discharge_disposition,
        'primary_diagnosis': primary_diagnosis,
        'readmission_30d': readmission_30d,
        'days_to_readmission': days_to_readmission
    })
    
    # Convert dates to string format for easier storage
    data['admission_date'] = data['admission_date'].dt.strftime('%Y-%m-%d')
    data['discharge_date'] = data['discharge_date'].dt.strftime('%Y-%m-%d')
    
    return data

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
        data, test_size=test_size, random_state=random_state, stratify=data['readmission_30d']
    )
    
    # Second split: training vs validation
    # val_size_adjusted is relative to the size of train_val
    val_size_adjusted = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_size_adjusted, random_state=random_state, stratify=train_val['readmission_30d']
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
    data = generate_synthetic_data(n_samples=1000)
    
    # Save raw data
    data.to_csv('data/raw/hospital_readmissions.csv', index=False)
    print(f"Raw data saved to data/raw/hospital_readmissions.csv ({len(data)} samples)")
    
    # Split and save data
    split_and_save_data(data)
