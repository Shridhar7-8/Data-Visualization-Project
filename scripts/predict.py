#!/usr/bin/env python
"""
Script to make predictions using the trained model.
This script is called by the API endpoint.
"""
import sys
import json
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import preprocessing functions
from src.data.preprocess import preprocess_data
from src.features.build_features import engineer_features

def load_model(model_path='models/best_model.pkl'):
    """Load the trained model."""
    return joblib.load(model_path)

def load_scaler(scaler_path='models/scaler.pkl'):
    """Load the fitted scaler."""
    return joblib.load(scaler_path)

def load_threshold(threshold_path='models/optimal_threshold.txt'):
    """Load the optimal threshold."""
    with open(threshold_path, 'r') as f:
        return float(f.read().strip())

def preprocess_patient_data(patient_data):
    """Preprocess patient data for prediction."""
    # Convert to DataFrame if it's a list
    if isinstance(patient_data, list):
        df = pd.DataFrame(patient_data)
    else:
        df = patient_data
    
    # Preprocess data
    processed_data = preprocess_data(df)
    
    # Load scaler
    scaler = load_scaler()
    
    # Engineer features
    X, _, _ = engineer_features(processed_data, is_training=False, scaler=scaler)
    
    return X

def predict_readmission_risk(patient_data, threshold=None):
    """Predict readmission risk for patients."""
    # Preprocess data
    X = preprocess_patient_data(patient_data)
    
    # Load model
    model = load_model()
    
    # Load threshold if not provided
    if threshold is None:
        threshold = load_threshold()
    
    # Make predictions
    risk_probabilities = model.predict_proba(X)[:, 1]
    risk_labels = (risk_probabilities >= threshold).astype(int)
    
    # Categorize risk levels
    risk_levels = []
    for prob in risk_probabilities:
        if prob < 0.3:
            risk_levels.append('Low')
        elif prob < 0.7:
            risk_levels.append('Medium')
        else:
            risk_levels.append('High')
    
    # Create results
    results = []
    for i, (patient, prob, label, level) in enumerate(zip(patient_data, risk_probabilities, risk_labels, risk_levels)):
        results.append({
            'patient_id': patient.get('patient_id', f'P-{1000+i}'),
            'readmission_risk_probability': float(prob),
            'readmission_risk_label': int(label),
            'risk_level': level
        })
    
    return results

def main():
    """Main function to process command line arguments and make predictions."""
    if len(sys.argv) != 2:
        print("Usage: python predict.py <patient_data_file>")
        sys.exit(1)
    
    # Load patient data
    patient_data_file = sys.argv[1]
    with open(patient_data_file, 'r') as f:
        patient_data = json.load(f)
    
    # Make predictions
    predictions = predict_readmission_risk(patient_data)
    
    # Print predictions as JSON
    print(json.dumps(predictions))

if __name__ == "__main__":
    main()
