"""
Visualization functions for hospital readmission prediction.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def load_raw_data(filepath='data/raw/hospital_readmissions.csv'):
    """
    Load raw data for visualization.
    
    Parameters:
    -----------
    filepath : str
        Path to raw data file
        
    Returns:
    --------
    pandas.DataFrame
        Raw data
    """
    return pd.read_csv(filepath)

def create_eda_visualizations(data):
    """
    Create exploratory data analysis visualizations.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw data
    """
    # Create directory if it doesn't exist
    os.makedirs('visualizations/eda', exist_ok=True)
    
    # Convert date columns to datetime
    data['admission_date'] = pd.to_datetime(data['admission_date'])
    data['discharge_date'] = pd.to_datetime(data['discharge_date'])
    
    # 1. Readmission rate
    plt.figure(figsize=(10, 6))
    readmission_counts = data['readmission_30d'].value_counts()
    readmission_rate = readmission_counts[1] / len(data) * 100
    
    plt.bar(['Not Readmitted', 'Readmitted'], 
            [readmission_counts[0], readmission_counts[1]],
            color=['#10b981', '#ef4444'])
    plt.title(f'30-Day Readmission Rate: {readmission_rate:.1f}%')
    plt.ylabel('Number of Patients')
    plt.tight_layout()
    plt.savefig('visualizations/eda/readmission_rate.png')
    plt.close()
    
    # 2. Age distribution by readmission status
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='age', hue='readmission_30d', 
                 multiple='dodge', bins=20, palette=['#10b981', '#ef4444'])
    plt.title('Age Distribution by Readmission Status')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend(['Not Readmitted', 'Readmitted'])
    plt.tight_layout()
    plt.savefig('visualizations/eda/age_distribution.png')
    plt.close()
    
    # 3. Length of stay by readmission status
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='readmission_30d', y='length_of_stay',
                palette=['#10b981', '#ef4444'])
    plt.title('Length of Stay by Readmission Status')
    plt.xlabel('Readmitted within 30 days')
    plt.ylabel('Length of Stay (days)')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.tight_layout()
    plt.savefig('visualizations/eda/length_of_stay.png')
    plt.close()
    
    # 4. Readmission rate by primary diagnosis
    plt.figure(figsize=(12, 8))
    diagnosis_readmission = data.groupby('primary_diagnosis')['readmission_30d'].mean() * 100
    diagnosis_readmission = diagnosis_readmission.sort_values(ascending=False)
    
    diagnosis_readmission.plot(kind='bar', color='#10b981')
    plt.title('Readmission Rate by Primary Diagnosis')
    plt.xlabel('Primary Diagnosis')
    plt.ylabel('Readmission Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('visualizations/eda/diagnosis_readmission.png')
    plt.close()
    
    # 5. Correlation matrix of numerical features
    plt.figure(figsize=(12, 10))
    numerical_cols = ['age', 'length_of_stay', 'previous_admissions', 
                      'medication_count', 'medication_adherence', 
                      'readmission_30d']
    corr = data[numerical_cols].corr()
    
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('visualizations/eda/correlation_matrix.png')
    plt.close()
    
    # 6. Readmission rate by number of previous admissions
    plt.figure(figsize=(10, 6))
    prev_admission_readmission = data.groupby('previous_admissions')['readmission_30d'].mean() * 100
    
    prev_admission_readmission.plot(kind='bar', color='#10b981')
    plt.title('Readmission Rate by Number of Previous Admissions')
    plt.xlabel('Number of Previous Admissions')
    plt.ylabel('Readmission Rate (%)')
    plt.tight_layout()
    plt.savefig('visualizations/eda/previous_admissions_readmission.png')
    plt.close()
    
    # 7. Readmission rate by medical conditions
    plt.figure(figsize=(12, 8))
    conditions = ['diabetes', 'heart_failure', 'copd', 'hypertension', 'renal_disease']
    condition_readmission = {}
    
    for condition in conditions:
        condition_readmission[condition] = [
            data[data[condition] == 0]['readmission_30d'].mean() * 100,
            data[data[condition] == 1]['readmission_30d'].mean() * 100
        ]
    
    condition_df = pd.DataFrame(condition_readmission, index=['No', 'Yes'])
    
    condition_df.plot(kind='bar', figsize=(12, 8))
    plt.title('Readmission Rate by Medical Condition')
    plt.xlabel('Condition Present')
    plt.ylabel('Readmission Rate (%)')
    plt.legend(title='Medical Condition')
    plt.tight_layout()
    plt.savefig('visualizations/eda/condition_readmission.png')
    plt.close()
    
    # 8. Medication adherence vs readmission
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='readmission_30d', y='medication_adherence',
                palette=['#10b981', '#ef4444'])
    plt.title('Medication Adherence by Readmission Status')
    plt.xlabel('Readmitted within 30 days')
    plt.ylabel('Medication Adherence Score')
    plt.xticks([0, 1], ['No', 'Yes'])
    plt.tight_layout()
    plt.savefig('visualizations/eda/medication_adherence.png')
    plt.close()
    
    # 9. Readmission rate by discharge disposition
    plt.figure(figsize=(10, 6))
    disposition_mapping = {
        1: 'Home',
        2: 'Skilled Nursing',
        3: 'Home Health',
        4: 'Other'
    }
    data['discharge_disposition_name'] = data['discharge_disposition'].map(disposition_mapping)
    
    disposition_readmission = data.groupby('discharge_disposition_name')['readmission_30d'].mean() * 100
    
    disposition_readmission.plot(kind='bar', color='#10b981')
    plt.title('Readmission Rate by Discharge Disposition')
    plt.xlabel('Discharge Disposition')
    plt.ylabel('Readmission Rate (%)')
    plt.tight_layout()
    plt.savefig('visualizations/eda/disposition_readmission.png')
    plt.close()
    
    # 10. Days to readmission distribution
    plt.figure(figsize=(10, 6))
    readmitted_patients = data[data['readmission_30d'] == 1]
    
    sns.histplot(data=readmitted_patients, x='days_to_readmission', bins=30, color='#ef4444')
    plt.title('Distribution of Days to Readmission')
    plt.xlabel('Days to Readmission')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('visualizations/eda/days_to_readmission.png')
    plt.close()
    
    print("EDA visualizations created in visualizations/eda/")

def create_model_performance_visualizations():
    """
    Create visualizations for model performance comparison.
    """
    # Check if metrics file exists
    if not os.path.exists('visualizations/test_metrics_comparison.csv'):
        print("Model metrics file not found. Run evaluate_model.py first.")
        return
    
    # Load metrics
    metrics_df = pd.read_csv('visualizations/test_metrics_comparison.csv', index_col=0)
    
    # Create directory if it doesn't exist
    os.makedirs('visualizations/model_performance', exist_ok=True)
    
    # 1. Model comparison bar chart
    plt.figure(figsize=(12, 8))
    metrics_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig('visualizations/model_performance/model_comparison.png')
    plt.close()
    
    # 2. ROC AUC comparison
    plt.figure(figsize=(10, 6))
    metrics_df['roc_auc'].sort_values().plot(kind='barh', color='#10b981')
    plt.title('Model Comparison by ROC AUC')
    plt.xlabel('ROC AUC Score')
    plt.ylabel('Model')
    plt.xlim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig('visualizations/model_performance/roc_auc_comparison.png')
    plt.close()
    
    # 3. Precision-Recall tradeoff
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics_df['precision'], metrics_df['recall'], s=100)
    
    # Add model names as labels
    for i, model in enumerate(metrics_df.index):
        plt.annotate(model, 
                     (metrics_df['precision'][i], metrics_df['recall'][i]),
                     xytext=(5, 5), textcoords='offset points')
    
    plt.title('Precision-Recall Tradeoff by Model')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.xlim(0.5, 1.0)
    plt.ylim(0.5, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/model_performance/precision_recall_tradeoff.png')
    plt.close()
    
    print("Model performance visualizations created in visualizations/model_performance/")

def create_feature_importance_visualization():
    """
    Create visualizations for feature importance.
    """
    # Check if feature importance file exists
    if not os.path.exists('visualizations/gradient_boosting_feature_importance.csv'):
        print("Feature importance file not found. Run train_model.py first.")
        return
    
    # Load feature importance
    importance_df = pd.read_csv('visualizations/gradient_boosting_feature_importance.csv')
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=True)
    
    # Create directory if it doesn't exist
    os.makedirs('visualizations/feature_importance', exist_ok=True)
    
    # Plot top 15 features
    plt.figure(figsize=(12, 10))
    top_features = importance_df.tail(15)
    
    plt.barh(top_features['Feature'], top_features['Importance'], color='#10b981')
    plt.title('Top 15 Features by Importance')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance/top_features.png')
    plt.close()
    
    print("Feature importance visualization created in visualizations/feature_importance/")

def main():
    """
    Main function to create all visualizations.
    """
    # Load data
    print("Loading data...")
    data = load_raw_data()
    
    # Create EDA visualizations
    print("Creating EDA visualizations...")
    create_eda_visualizations(data)
    
    # Create model performance visualizations
    print("Creating model performance visualizations...")
    create_model_performance_visualizations()
    
    # Create feature importance visualization
    print("Creating feature importance visualization...")
    create_feature_importance_visualization()

if __name__ == "__main__":
    main()
