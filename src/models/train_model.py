# import pandas as pd
# import numpy as np
# import joblib
# import os
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from sklearn.model_selection import GridSearchCV
# import matplotlib.pyplot as plt
# from xgboost import XGBClassifier


# def load_features(X_path, y_path):
#     """
#     Load features from CSV files.
#     """
#     X = pd.read_csv(X_path)
#     y = pd.read_csv(y_path).values.ravel()
#     return X, y


# def train_logistic_regression(X_train, y_train, X_val, y_val):
#     """
#     Train a logistic regression model with expanded solver options.
#     """
#     print("Training Logistic Regression model...")

#     # Define parameter grid with multiple solvers and compatible penalties
#     param_grid = [
#         {
#             'C': [0.01, 0.1, 1, 10, 100],
#             'penalty': ['l1', 'l2'],
#             'solver': ['liblinear', 'saga'],
#             'class_weight': [None, 'balanced']
#         },
#         {
#             'C': [0.01, 0.1, 1, 10, 100],
#             'penalty': ['l2'],
#             'solver': ['lbfgs', 'newton-cg', 'sag'],
#             'class_weight': [None, 'balanced']
#         }
#     ]

#     lr = LogisticRegression(random_state=42, max_iter=5000)
#     grid_search = GridSearchCV(
#         lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
#     )
#     grid_search.fit(X_train, y_train)

#     best_lr = grid_search.best_estimator_
#     y_val_pred = best_lr.predict(X_val)
#     y_val_prob = best_lr.predict_proba(X_val)[:, 1]

#     metrics = {
#         'accuracy': accuracy_score(y_val, y_val_pred),
#         'precision': precision_score(y_val, y_val_pred),
#         'recall': recall_score(y_val, y_val_pred),
#         'f1': f1_score(y_val, y_val_pred),
#         'roc_auc': roc_auc_score(y_val, y_val_prob)
#     }

#     print(f"Best parameters: {grid_search.best_params_}")
#     print(f"Validation metrics: {metrics}")

#     return best_lr, metrics


# def train_random_forest(X_train, y_train, X_val, y_val):
#     print("Training Random Forest model...")
#     param_grid = {
#         'n_estimators': [100, 200],
#         'max_depth': [None, 10, 20],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4],
#         'class_weight': [None, 'balanced']
#     }
#     rf = RandomForestClassifier(random_state=42)
#     grid_search = GridSearchCV(
#         rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
#     )
#     grid_search.fit(X_train, y_train)
#     best_rf = grid_search.best_estimator_
#     y_val_pred = best_rf.predict(X_val)
#     y_val_prob = best_rf.predict_proba(X_val)[:, 1]
#     metrics = {
#         'accuracy': accuracy_score(y_val, y_val_pred),
#         'precision': precision_score(y_val, y_val_pred),
#         'recall': recall_score(y_val, y_val_pred),
#         'f1': f1_score(y_val, y_val_pred),
#         'roc_auc': roc_auc_score(y_val, y_val_prob)
#     }
#     print(f"Best parameters: {grid_search.best_params_}")
#     print(f"Validation metrics: {metrics}")
#     return best_rf, metrics


# def train_gradient_boosting(X_train, y_train, X_val, y_val):
#     print("Training Gradient Boosting model...")
#     param_grid = {
#         'n_estimators': [100, 200],
#         'learning_rate': [0.01, 0.1],
#         'max_depth': [3, 5, 7],
#         'min_samples_split': [2, 5],
#         'min_samples_leaf': [1, 2],
#         'subsample': [0.8, 1.0]
#     }
#     gb = GradientBoostingClassifier(random_state=42)
#     grid_search = GridSearchCV(
#         gb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
#     )
#     grid_search.fit(X_train, y_train)
#     best_gb = grid_search.best_estimator_
#     y_val_pred = best_gb.predict(X_val)
#     y_val_prob = best_gb.predict_proba(X_val)[:, 1]
#     metrics = {
#         'accuracy': accuracy_score(y_val, y_val_pred),
#         'precision': precision_score(y_val, y_val_pred),
#         'recall': recall_score(y_val, y_val_pred),
#         'f1': f1_score(y_val, y_val_pred),
#         'roc_auc': roc_auc_score(y_val, y_val_prob)
#     }
#     print(f"Best parameters: {grid_search.best_params_}")
#     print(f"Validation metrics: {metrics}")
#     return best_gb, metrics


# def train_xgboost(X_train, y_train, X_val, y_val):
#     print("Training XGBoost model...")
#     param_grid = {
#         'n_estimators': [100, 200],
#         'learning_rate': [0.01, 0.1],
#         'max_depth': [3, 5, 7],
#         'subsample': [0.8, 1.0],
#         'colsample_bytree': [0.8, 1.0]
#     }
#     xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
#     grid_search = GridSearchCV(
#         xgb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
#     )
#     grid_search.fit(X_train, y_train)
#     best_xgb = grid_search.best_estimator_
#     y_val_pred = best_xgb.predict(X_val)
#     y_val_prob = best_xgb.predict_proba(X_val)[:, 1]
#     metrics = {
#         'accuracy': accuracy_score(y_val, y_val_pred),
#         'precision': precision_score(y_val, y_val_pred),
#         'recall': recall_score(y_val, y_val_pred),
#         'f1': f1_score(y_val, y_val_pred),
#         'roc_auc': roc_auc_score(y_val, y_val_prob)
#     }
#     print(f"Best parameters: {grid_search.best_params_}")
#     print(f"Validation metrics: {metrics}")
#     return best_xgb, metrics


# def plot_feature_importance(model, X_train, model_name):
#     os.makedirs('visualizations', exist_ok=True)
#     if hasattr(model, 'feature_importances_'):
#         importances = model.feature_importances_
#         indices = np.argsort(importances)[::-1]
#         feature_names = X_train.columns
#         plt.figure(figsize=(12, 8))
#         plt.title(f'Feature Importance - {model_name}')
#         plt.barh(range(len(indices)), importances[indices], align='center')
#         plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
#         plt.xlabel('Relative Importance')
#         plt.tight_layout()
#         plt.savefig(f'visualizations/{model_name}_feature_importance.png')
#         plt.close()
#         importance_df = pd.DataFrame({
#             'Feature': feature_names,
#             'Importance': importances
#         })
#         importance_df = importance_df.sort_values('Importance', ascending=False)
#         importance_df.to_csv(f'visualizations/{model_name}_feature_importance.csv', index=False)
#         print(f"Feature importance for {model_name} saved.")


# def train_and_save_models():
#     os.makedirs('models', exist_ok=True)
#     os.makedirs('visualizations', exist_ok=True)

#     X_train, y_train = load_features(r'E:\Data-Visualization-Project\src\features\data\features\X_train.csv', 
#                                      r'E:\Data-Visualization-Project\src\features\data\features\y_train.csv')
#     X_val, y_val = load_features(r'E:\Data-Visualization-Project\src\features\data\features\X_val.csv', 
#                                  r'E:\Data-Visualization-Project\src\features\data\features\y_val.csv')

#     models = {}
#     metrics = {}

#     # Logistic Regression
#     lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
#     models['logistic_regression'] = lr_model
#     metrics['logistic_regression'] = lr_metrics

#     # Random Forest
#     rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
#     models['random_forest'] = rf_model
#     metrics['random_forest'] = rf_metrics

#     # Gradient Boosting
#     gb_model, gb_metrics = train_gradient_boosting(X_train, y_train, X_val, y_val)
#     models['gradient_boosting'] = gb_model
#     metrics['gradient_boosting'] = gb_metrics

#     # XGBoost
#     xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)
#     models['xgboost'] = xgb_model
#     metrics['xgboost'] = xgb_metrics

#     # Save all models
#     for name, model in models.items():
#         joblib.dump(model, f'models/{name}.pkl')
#         print(f"Model {name} saved to models/{name}.pkl")

#     # Plot feature importance for tree-based models
#     plot_feature_importance(rf_model, X_train, 'random_forest')
#     plot_feature_importance(gb_model, X_train, 'gradient_boosting')
#     plot_feature_importance(xgb_model, X_train, 'xgboost')

#     # Save metrics
#     metrics_df = pd.DataFrame(metrics).T
#     metrics_df.to_csv('visualizations/model_metrics.csv')
#     print("Model metrics saved to visualizations/model_metrics.csv")

#     best_model_name = max(metrics, key=lambda k: metrics[k]['roc_auc'])
#     best_model = models[best_model_name]
#     joblib.dump(best_model, 'models/best_model.pkl')
#     print(f"Best model ({best_model_name}) saved to models/best_model.pkl")

#     return models, metrics


# if __name__ == "__main__":
#     train_and_save_models()



import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform


def load_features(X_path, y_path):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).values.ravel()
    return X, y


def optimize_threshold(y_true, y_prob):
    """
    Find threshold between 0 and 1 that maximizes accuracy.
    """
    thresholds = np.linspace(0, 1, 101)
    accs = [accuracy_score(y_true, (y_prob > t).astype(int)) for t in thresholds]
    best_idx = np.argmax(accs)
    return thresholds[best_idx], accs[best_idx]


def train_logistic_regression(X_train, y_train, X_val, y_val):
    print("Training Logistic Regression model...")
    param_grid = [
        {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga'], 'class_weight': [None, 'balanced']},
        {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['lbfgs', 'newton-cg', 'sag'], 'class_weight': [None, 'balanced']}
    ]
    lr = LogisticRegression(random_state=42, max_iter=5000, tol=1e-4)
    grid = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    y_prob = model.predict_proba(X_val)[:, 1]
    best_thresh, best_acc = optimize_threshold(y_val, y_prob)

    y_pred = (y_prob > best_thresh).astype(int)
    metrics = {
        'accuracy': best_acc,
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_prob),
        'best_threshold': best_thresh
    }
    print(f"LR best params: {grid.best_params_}")
    print(f"LR optimal threshold: {best_thresh:.2f}, accuracy: {best_acc:.4f}")
    return model, metrics


def train_random_forest(X_train, y_train, X_val, y_val):
    print("Training Random Forest model...")
    param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'class_weight': [None, 'balanced']}
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    y_prob = model.predict_proba(X_val)[:, 1]
    best_thresh, best_acc = optimize_threshold(y_val, y_prob)

    y_pred = (y_prob > best_thresh).astype(int)
    metrics = {
        'accuracy': best_acc,
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_prob),
        'best_threshold': best_thresh
    }
    print(f"RF best params: {grid.best_params_}")
    print(f"RF optimal threshold: {best_thresh:.2f}, accuracy: {best_acc:.4f}")
    return model, metrics


def train_gradient_boosting(X_train, y_train, X_val, y_val):
    print("Training Gradient Boosting model...")
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2], 'subsample': [0.8, 1.0]}
    gb = GradientBoostingClassifier(random_state=42)
    grid = GridSearchCV(gb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    y_prob = model.predict_proba(X_val)[:, 1]
    best_thresh, best_acc = optimize_threshold(y_val, y_prob)

    y_pred = (y_prob > best_thresh).astype(int)
    metrics = {
        'accuracy': best_acc,
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_prob),
        'best_threshold': best_thresh
    }
    print(f"GB best params: {grid.best_params_}")
    print(f"GB optimal threshold: {best_thresh:.2f}, accuracy: {best_acc:.4f}")
    return model, metrics


def train_xgboost(X_train, y_train, X_val, y_val):
    print("Training XGBoost model...")
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0]}
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    grid = GridSearchCV(xgb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    y_prob = model.predict_proba(X_val)[:, 1]
    best_thresh, best_acc = optimize_threshold(y_val, y_prob)

    y_pred = (y_prob > best_thresh).astype(int)
    metrics = {
        'accuracy': best_acc,
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_prob),
        'best_threshold': best_thresh
    }
    print(f"XGB best params: {grid.best_params_}")
    print(f"XGB optimal threshold: {best_thresh:.2f}, accuracy: {best_acc:.4f}")
    return model, metrics


def train_catboost(X_train, y_train, X_val, y_val):
    print("Training CatBoost model...")
    os.makedirs('catboost_info', exist_ok=True)
    param_dist = {'iterations': randint(100, 500), 'learning_rate': uniform(0.01, 0.1), 'depth': randint(4, 10), 'l2_leaf_reg': uniform(1, 5), 'subsample': uniform(0.7, 0.3)}
    cb = CatBoostClassifier(random_state=42, verbose=0, early_stopping_rounds=30, train_dir='catboost_info')
    rand = RandomizedSearchCV(cb, param_dist, n_iter=30, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1, random_state=42)
    rand.fit(X_train, y_train, eval_set=(X_val, y_val))
    model = rand.best_estimator_
    y_prob = model.predict_proba(X_val)[:, 1]
    best_thresh, best_acc = optimize_threshold(y_val, y_prob)

    y_pred = (y_prob > best_thresh).astype(int)
    metrics = {
        'accuracy': best_acc,
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_prob),
        'best_threshold': best_thresh
    }
    print(f"CatBoost optimal threshold: {best_thresh:.2f}, accuracy: {best_acc:.4f}")
    return model, metrics


def train_lightgbm(X_train, y_train, X_val, y_val):
    print("Training LightGBM model...")
    param_dist = {'n_estimators': randint(100, 500), 'learning_rate': uniform(0.01, 0.1), 'num_leaves': randint(20, 100), 'subsample': uniform(0.7, 0.3), 'colsample_bytree': uniform(0.7, 0.3)}
    lgbm = LGBMClassifier(random_state=42)
    rand = RandomizedSearchCV(lgbm, param_dist, n_iter=30, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1, random_state=42)
    import lightgbm as lgb_lib
    fit_params = {'eval_set': [(X_val, y_val)], 'eval_metric': 'auc', 'callbacks': [lgb_lib.callback.early_stopping(stopping_rounds=30)]}
    rand.fit(X_train, y_train, **fit_params)
    model = rand.best_estimator_
    y_prob = model.predict_proba(X_val)[:, 1]
    best_thresh, best_acc = optimize_threshold(y_val, y_prob)

    y_pred = (y_prob > best_thresh).astype(int)
    metrics = {
        'accuracy': best_acc,
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_prob),
        'best_threshold': best_thresh
    }
    print(f"LightGBM optimal threshold: {best_thresh:.2f}, accuracy: {best_acc:.4f}")
    return model, metrics


def train_voting_ensemble(models, X_train, y_train, X_val, y_val):
    print("Training Voting Ensemble...")
    voting = VotingClassifier([('lr', models['logistic_regression']), ('rf', models['random_forest']), ('gb', models['gradient_boosting']), ('xgb', models['xgboost']), ('cb', models['catboost']), ('lgb', models['lightgbm'])], voting='soft', n_jobs=-1)
    voting.fit(X_train, y_train)
    y_prob = voting.predict_proba(X_val)[:, 1]
    best_thresh, best_acc = optimize_threshold(y_val, y_prob)
    y_pred = (y_prob > best_thresh).astype(int)
    metrics = {
        'accuracy': best_acc,
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_prob),
        'best_threshold': best_thresh
    }
    print(f"Voting ensemble optimal threshold: {best_thresh:.2f}, accuracy: {best_acc:.4f}")
    return voting, metrics


def train_stacking(models, X_train, y_train, X_val, y_val):
    print("Training Stacking Ensemble...")
    estimators = [(name, models[name]) for name in ['logistic_regression', 'random_forest', 'gradient_boosting', 'xgboost', 'catboost', 'lightgbm']]
    stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=42, max_iter=2000), cv=5, n_jobs=-1, passthrough=True)
    stack.fit(X_train, y_train)
    y_prob = stack.predict_proba(X_val)[:, 1]
    best_thresh, best_acc = optimize_threshold(y_val, y_prob)
    y_pred = (y_prob > best_thresh).astype(int)
    metrics = {
        'accuracy': best_acc,
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_prob),
        'best_threshold': best_thresh
    }
    print(f"Stacking ensemble optimal threshold: {best_thresh:.2f}, accuracy: {best_acc:.4f}")
    return stack, metrics


def plot_feature_importance(model, X_train, name):
    os.makedirs('visualizations', exist_ok=True)
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1]
        names = X_train.columns
        plt.figure(figsize=(12,8))
        plt.barh(range(len(idx)), imp[idx], align='center')
        plt.yticks(range(len(idx)), [names[i] for i in idx])
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'visualizations/{name}_feature_importance.png')
        plt.close()


def train_and_save_models():
    
    os.makedirs('models', exist_ok=True)
    X_train, y_train = load_features(r'E:\Data-Visualization-Project\src\features\data\features\X_train.csv', 
                                     r'E:\Data-Visualization-Project\src\features\data\features\y_train.csv')
    
    X_val, y_val = load_features(r'E:\Data-Visualization-Project\src\features\data\features\X_val.csv', 
                                 r'E:\Data-Visualization-Project\src\features\data\features\y_val.csv')

    models, metrics = {}, {}
    funcs = [
        ('logistic_regression', train_logistic_regression),
        ('random_forest', train_random_forest),
        ('gradient_boosting', train_gradient_boosting),
        ('xgboost', train_xgboost),
        ('catboost', train_catboost),
        ('lightgbm', train_lightgbm)
    ]
    for name, fn in funcs:
        m, met = fn(X_train, y_train, X_val, y_val)
        models[name], metrics[name] = m, met

    # Ensembles
    models['voting'], metrics['voting'] = train_voting_ensemble(models, X_train, y_train, X_val, y_val)
    models['stacking'], metrics['stacking'] = train_stacking(models, X_train, y_train, X_val, y_val)

    # Save
    for n,m in models.items(): joblib.dump(m, f'models/{n}.pkl')
    pd.DataFrame(metrics).T.to_csv('model_metrics.csv')

    best = max(metrics, key=lambda k: metrics[k]['accuracy'])
    joblib.dump(models[best], 'models/best_model.pkl')
    print(f"Best model: {best} with accuracy {metrics[best]['accuracy']:.4f}")


if __name__ == '__main__':
    train_and_save_models()
