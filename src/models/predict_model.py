"""
Make predictions using trained models.
"""
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

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



# def predict_readmission_risk(patient_data, threshold=0.5):
#     """
#     Predict readmission risk for patients.
    
#     Parameters:
#     -----------
#     patient_data : pandas.DataFrame
#         Raw patient data
#     threshold : float
#         Probability threshold for high risk classification
        
#     Returns:
#     --------
#     pandas.DataFrame
#         Patient data with readmission risk predictions
#     """
#     # Preprocess data
#     X = preprocess_patient_data(patient_data)
    
#     # Load model
#     model = load_model()
    
#     # Make predictions
#     risk_probabilities = model.predict_proba(X)[:, 1]
#     risk_labels = (risk_probabilities >= threshold).astype(int)
    
#     # Add predictions to patient data
#     results = patient_data.copy()
#     results['readmission_risk_probability'] = risk_probabilities
#     results['readmission_risk_label'] = risk_labels
    
#     # Categorize risk levels
#     results['risk_level'] = pd.cut(
#         results['readmission_risk_probability'],
#         bins=[0, 0.3, 0.7, 1.0],
#         labels=['Low', 'Medium', 'High']
#     )
    
#     return results

def load_test_data():
    """Load preprocessed test data"""
    try:
        X_test = pd.read_csv(r"E:\Data-Visualization-Project\src\features\data\features\X_test.csv")
        y_test = pd.read_csv(r"E:\Data-Visualization-Project\src\features\data\features\y_test.csv")
        print(f"Test data loaded successfully. X shape: {X_test.shape}, y shape: {y_test.shape}")
        return X_test, y_test
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return None, None

def load_models():
    """Load trained models"""
    try:
        models_dir = r'E:\Data-Visualization-Project\src\models\models'
        models = {
            'logistic_regression': joblib.load(os.path.join(models_dir, 'logistic_regression.pkl')),
            'random_forest': joblib.load(os.path.join(models_dir, 'random_forest.pkl')),
            'gradient_boosting': joblib.load(os.path.join(models_dir, 'gradient_boosting.pkl')),
            'best_model': joblib.load(os.path.join(models_dir, 'best_model.pkl')),
            'xgboost': joblib.load(os.path.join(models_dir, 'xgboost.pkl')),
            'catboost': joblib.load(os.path.join(models_dir, 'catboost.pkl')),
            'lightgbm': joblib.load(os.path.join(models_dir, 'lightgbm.pkl')),
            'voting': joblib.load(os.path.join(models_dir, 'voting.pkl')),
            'stacking': joblib.load(os.path.join(models_dir, 'stacking.pkl'))
        }
        print("Models loaded successfully")
        return models
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return {}

def make_predictions(X_test, models):
    """Make predictions using all models"""
    predictions = {}
    probabilities = {}
    
    for model_name, model in models.items():
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class
            
            predictions[model_name] = y_pred
            probabilities[model_name] = y_prob
            print(f"Predictions made successfully for {model_name}")
        except Exception as e:
            print(f"Error making predictions with {model_name}: {str(e)}")
    
    return predictions, probabilities

def calculate_metrics(y_test, predictions, probabilities):
    """Calculate and return performance metrics"""
    metrics = {}
    
    for model_name, y_pred in predictions.items():
        try:
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, probabilities[model_name])
            roc_auc = auc(fpr, tpr)
            
            metrics[model_name] = {
                'classification_report': report,
                'confusion_matrix': cm,
                'roc_curve': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
            }
            print(f"Metrics calculated successfully for {model_name}")
        except Exception as e:
            print(f"Error calculating metrics for {model_name}: {str(e)}")
    
    return metrics

def save_predictions_and_metrics(predictions, probabilities, metrics, models, X_test):
    """Save predictions and metrics to files"""
    try:
        # Create directories if they don't exist
        os.makedirs('src/models/predictions', exist_ok=True)
        os.makedirs('src/models/test_visualizations', exist_ok=True)
        
        for model_name in predictions.keys():
            # Save predictions
            pred_df = pd.DataFrame({
                'prediction': predictions[model_name],
                'probability': probabilities[model_name]
            })
            pred_df.to_csv(f'src/models/predictions/{model_name}_predictions.csv', index=False)
            
            # Save metrics
            metrics_df = pd.DataFrame(metrics[model_name]['classification_report']).transpose()
            metrics_df.to_csv(f'src/models/test_visualizations/{model_name}_metrics.csv')
            
        
        print("Predictions and metrics saved successfully")
    except Exception as e:
        print(f"Error saving predictions and metrics: {str(e)}")

def create_visualizations(metrics):
    """Create and save visualizations for each model"""
    for model_name, model_metrics in metrics.items():
        try:
            # Create directory if it doesn't exist
            viz_dir = f'src/models/test_visualizations/{model_name}'
            os.makedirs(viz_dir, exist_ok=True)
            
            # Confusion Matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(model_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.savefig(f'{viz_dir}/confusion_matrix.png')
            plt.close()
            
            # ROC Curve
            plt.figure(figsize=(8, 6))
            plt.plot(model_metrics['roc_curve']['fpr'], 
                    model_metrics['roc_curve']['tpr'], 
                    label=f'ROC curve (AUC = {model_metrics["roc_curve"]["auc"]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend()
            plt.savefig(f'{viz_dir}/roc_curve.png')
            plt.close()
            
            print(f"Visualizations created successfully for {model_name}")
        except Exception as e:
            print(f"Error creating visualizations for {model_name}: {str(e)}")

def main():
    # Load test data
    X_test, y_test = load_test_data()
    if X_test is None or y_test is None:
        return
    
    # Load models
    models = load_models()
    if not models:
        return
    
    # Make predictions
    predictions, probabilities = make_predictions(X_test, models)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, predictions, probabilities)
    
    # Save predictions and metrics
    save_predictions_and_metrics(predictions, probabilities, metrics, models, X_test)
    
    # Create visualizations
    create_visualizations(metrics)
    
    print("Prediction process completed successfully!")

if __name__ == "__main__":
    main()
