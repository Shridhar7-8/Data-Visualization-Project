"""
Evaluate trained models on test data.
"""
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

def load_test_data():
    """
    Load test data.
    
    Returns:
    --------
    tuple
        (X_test, y_test) test features and target
    """
    X_test = pd.read_csv('data/features/X_test.csv')
    y_test = pd.read_csv('data/features/y_test.csv').values.ravel()
    return X_test, y_test

def load_models():
    """
    Load trained models.
    
    Returns:
    --------
    dict
        Dictionary of trained models
    """
    models = {}
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl') and f != 'scaler.pkl']
    
    for model_file in model_files:
        model_name = model_file.split('.')[0]
        if model_name != 'best_model':  # Skip the best model, we'll load it separately
            models[model_name] = joblib.load(f'models/{model_file}')
    
    return models

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a model on test data.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : pandas.DataFrame
        Test features
    y_test : numpy.ndarray
        Test target
    model_name : str
        Name of the model
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'avg_precision': average_precision_score(y_test, y_prob)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'visualizations/{model_name}_confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'visualizations/{model_name}_roc_curve.png')
    plt.close()
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label=f'PR curve (AP = {metrics["avg_precision"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f'visualizations/{model_name}_pr_curve.png')
    plt.close()
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'visualizations/{model_name}_classification_report.csv')
    
    print(f"Evaluation for {model_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"  Average Precision: {metrics['avg_precision']:.4f}")
    
    return metrics

def evaluate_all_models():
    """
    Evaluate all trained models on test data.
    """
    # Load test data
    X_test, y_test = load_test_data()
    
    # Load models
    models = load_models()
    
    # Evaluate each model
    all_metrics = {}
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name)
        all_metrics[model_name] = metrics
    
    # Compare models
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv('visualizations/test_metrics_comparison.csv')
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    metrics_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].plot(kind='bar')
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison.png')
    plt.close()
    
    print(f"Model comparison saved to visualizations/model_comparison.png")
    print(f"Test metrics saved to visualizations/test_metrics_comparison.csv")
    
    # Find best model based on ROC AUC
    best_model_name = metrics_df['roc_auc'].idxmax()
    print(f"Best model on test data: {best_model_name} (ROC AUC: {metrics_df.loc[best_model_name, 'roc_auc']:.4f})")
    
    return all_metrics

if __name__ == "__main__":
    evaluate_all_models()
