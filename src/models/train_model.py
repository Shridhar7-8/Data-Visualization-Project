"""
Train machine learning models for hospital readmission prediction.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

def load_features(X_path, y_path):
    """
    Load features from CSV files.
    
    Parameters:
    -----------
    X_path : str
        Path to features CSV file
    y_path : str
        Path to target CSV file
        
    Returns:
    --------
    tuple
        (X, y) features and target
    """
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).values.ravel()
    return X, y

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train a logistic regression model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : numpy.ndarray
        Training target
    X_val : pandas.DataFrame
        Validation features
    y_val : numpy.ndarray
        Validation target
        
    Returns:
    --------
    tuple
        (model, validation_metrics)
    """
    print("Training Logistic Regression model...")
    
    # Define parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': [None, 'balanced']
    }
    
    # Initialize model
    lr = LogisticRegression(random_state=42, max_iter=1000)
    
    # Grid search
    grid_search = GridSearchCV(
        lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_lr = grid_search.best_estimator_
    
    # Evaluate on validation set
    y_val_pred = best_lr.predict(X_val)
    y_val_prob = best_lr.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, y_val_prob)
    }
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Validation metrics: {metrics}")
    
    return best_lr, metrics

def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train a random forest model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : numpy.ndarray
        Training target
    X_val : pandas.DataFrame
        Validation features
    y_val : numpy.ndarray
        Validation target
        
    Returns:
    --------
    tuple
        (model, validation_metrics)
    """
    print("Training Random Forest model...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    }
    
    # Initialize model
    rf = RandomForestClassifier(random_state=42)
    
    # Grid search
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_rf = grid_search.best_estimator_
    
    # Evaluate on validation set
    y_val_pred = best_rf.predict(X_val)
    y_val_prob = best_rf.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, y_val_prob)
    }
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Validation metrics: {metrics}")
    
    return best_rf, metrics

def train_gradient_boosting(X_train, y_train, X_val, y_val):
    """
    Train a gradient boosting model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : numpy.ndarray
        Training target
    X_val : pandas.DataFrame
        Validation features
    y_val : numpy.ndarray
        Validation target
        
    Returns:
    --------
    tuple
        (model, validation_metrics)
    """
    print("Training Gradient Boosting model...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 1.0]
    }
    
    # Initialize model
    gb = GradientBoostingClassifier(random_state=42)
    
    # Grid search
    grid_search = GridSearchCV(
        gb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_gb = grid_search.best_estimator_
    
    # Evaluate on validation set
    y_val_pred = best_gb.predict(X_val)
    y_val_prob = best_gb.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, y_val_prob)
    }
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Validation metrics: {metrics}")
    
    return best_gb, metrics

def plot_feature_importance(model, X_train, model_name):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_train : pandas.DataFrame
        Training features
    model_name : str
        Name of the model
    """
    # Create directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Get feature names
        feature_names = X_train.columns
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance - {model_name}')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(f'visualizations/{model_name}_feature_importance.png')
        plt.close()
        
        # Save feature importance to CSV
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_df.to_csv(f'visualizations/{model_name}_feature_importance.csv', index=False)
        
        print(f"Feature importance plot saved to visualizations/{model_name}_feature_importance.png")
        print(f"Feature importance data saved to visualizations/{model_name}_feature_importance.csv")

def train_and_save_models():
    """
    Train and save all models.
    """
    # Create directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Load data
    X_train, y_train = load_features('data/features/X_train.csv', 'data/features/y_train.csv')
    X_val, y_val = load_features('data/features/X_val.csv', 'data/features/y_val.csv')
    
    # Train models
    models = {}
    metrics = {}
    
    # Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
    models['logistic_regression'] = lr_model
    metrics['logistic_regression'] = lr_metrics
    
    # Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    models['random_forest'] = rf_model
    metrics['random_forest'] = rf_metrics
    
    # Gradient Boosting
    gb_model, gb_metrics = train_gradient_boosting(X_train, y_train, X_val, y_val)
    models['gradient_boosting'] = gb_model
    metrics['gradient_boosting'] = gb_metrics
    
    # Save models
    for name, model in models.items():
        joblib.dump(model, f'models/{name}.pkl')
        print(f"Model {name} saved to models/{name}.pkl")
      f'models/{name}.pkl')
        print(f"Model {name} saved to models/{name}.pkl")
    
    # Plot feature importance for tree-based models
    plot_feature_importance(rf_model, X_train, 'random_forest')
    plot_feature_importance(gb_model, X_train, 'gradient_boosting')
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv('visualizations/model_metrics.csv')
    print(f"Model metrics saved to visualizations/model_metrics.csv")
    
    # Find best model based on ROC AUC
    best_model_name = max(metrics, key=lambda k: metrics[k]['roc_auc'])
    best_model = models[best_model_name]
    
    # Save best model separately
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"Best model ({best_model_name}) saved to models/best_model.pkl")
    
    return models, metrics

if __name__ == "__main__":
    train_and_save_models()
