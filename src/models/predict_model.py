"""
Make predictions using trained models.
"""
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

def load_model(model_path='models/best_model.pkl'):
    """
    Load a trained model.
    
    Parameters:
    -----------
    model_path : str
        Path to the model file
        
    Returns:
    --------
    sklearn model
        Trained model
    """
    return joblib.load(model_path)

def load_scaler(scaler_path='models/scaler.pkl'):
    """
    Load the fitted scaler.
    
    Parameters:
    -----------
    scaler_path : str
        Path to the scaler file
        
    Returns:
    --------
    sklearn.preprocessing.StandardScaler
        Fitted scaler
    """
    return joblib.load(scaler_path)

def preprocess_patient_data(patient_data):
    """
    Preprocess patient data for prediction.
    
    Parameters:
    -----------
    patient_data : pandas.DataFrame
        Raw patient data
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed data ready for prediction
    """
    # Import preprocessing functions
    import sys
    sys.path.append('.')
    from src.data.preprocess import preprocess_data
    from src.features.build_features import engineer_features
    
    # Preprocess data
    processed_data = preprocess_data(patient_data)
    
    # Load scaler
    scaler = load_scaler()
    
    # Engineer features
    X, _, _ = engineer_features(processed_data, is_training=False, scaler=scaler)
    
    return X

def predict_readmission_risk(patient_data, threshold=0.5):
    """
    Predict readmission risk for patients.
    
    Parameters:
    -----------
    patient_data : pandas.DataFrame
        Raw patient data
    threshold : float
        Probability threshold for high risk classification
        
    Returns:
    --------
    pandas.DataFrame
        Patient data with readmission risk predictions
    """
    # Preprocess data
    X = preprocess_patient_data(patient_data)
    
    # Load model
    model = load_model()
    
    # Make predictions
    risk_probabilities = model.predict_proba(X)[:, 1]
    risk_labels = (risk_probabilities >= threshold).astype(int)
    
    # Add predictions to patient data
    results = patient_data.copy()
    results['readmission_risk_probability'] = risk_probabilities
    results['readmission_risk_label'] = risk_labels
    
    # Categorize risk levels
    results['risk_level'] = pd.cut(
        results['readmission_risk_probability'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    return results

def generate_sample_patients(n_samples=10):
    """
    Generate sample patient data for demonstration.
    
    Parameters:
    -----------
    n_samples : int
        Number of patient samples to generate
        
    Returns:
    --------
    pandas.DataFrame
        Sample patient data
    """
    # Import data generation function
    import sys
    sys.path.append('.')
    from src.data.make_dataset import generate_synthetic_data
    
    # Generate sample data
    sample_data = generate_synthetic_data(n_samples=n_samples)
    
    return sample_data

def save_predictions(predictions, output_path='data/predictions'):
    """
    Save prediction results.
    
    Parameters:
    -----------
    predictions : pandas.DataFrame
        Prediction results
    output_path : str
        Path to save the results
    """
    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save predictions
    predictions.to_csv(f'{output_path}/predictions_{timestamp}.csv', index=False)
    print(f"Predictions saved to {output_path}/predictions_{timestamp}.csv")

def main():
    """
    Main function to demonstrate prediction functionality.
    """
    # Generate sample patients
    print("Generating sample patient data...")
    sample_patients = generate_sample_patients(n_samples=10)
    
    # Make predictions
    print("Predicting readmission risk...")
    predictions = predict_readmission_risk(sample_patients)
    
    # Display results
    print("\nPrediction Results:")
    print(predictions[['patient_id', 'age', 'primary_diagnosis', 'readmission_risk_probability', 'risk_level']].head())
    
    # Save predictions
    save_predictions(predictions)

if __name__ == "__main__":
    main()
