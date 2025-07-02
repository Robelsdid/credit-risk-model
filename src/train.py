# Placeholder for model training script 

import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_processing import process_data

# Paths and config
DATA_PATH = 'data/processed/processed_data_with_risk.csv'
TARGET_COL = 'is_high_risk'
MLFLOW_EXPERIMENT = 'credit-risk-model'

# 1. Load and preprocess data
print('Processing data...')
df_processed, pipeline = process_data(DATA_PATH)

# For quick testing, using a random sample of 5000 rows
df_processed = df_processed.sample(n=5000, random_state=42)

# 2. Split features and target
X = df_processed.drop(columns=[TARGET_COL, 'CustomerId'], errors='ignore')

# Drop datetime columns  if present
X = X.select_dtypes(exclude=['datetime', 'datetime64[ns]'])

# Drop columns where the first non-null value is a Timestamp
for col in X.columns:
    first_valid = X[col].dropna().iloc[0] if not X[col].dropna().empty else None
    if isinstance(first_valid, pd.Timestamp):
        X = X.drop(columns=[col])

# Drop columns that are not numeric (e.g., datetime strings)
non_numeric_cols = X.select_dtypes(include=['object']).columns
X = X.drop(columns=non_numeric_cols)

y = df_processed[TARGET_COL]

print('Unique values in target before mapping:', y.unique())
# Map the higher value to 1 (high risk), lower to 0 (low risk)
high = max(y.unique())
low = min(y.unique())
y = y.apply(lambda v: 1 if v == high else 0)
print('Unique values in target after mapping:', y.unique())

# Ensure target is integer and binary
# y = y.round().astype(int)  # No longer needed since we map above

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Define models and hyperparameters
models = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000, solver='liblinear'),
        'params': {
            'C': [1],  # Only one value
            'penalty': ['l2']  # Only one value
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50],  # Only one value
            'max_depth': [None],   # Only one value
            'min_samples_split': [2]  # Only one value
        }
    }
}

# 4. MLflow experiment setup
mlflow.set_experiment(MLFLOW_EXPERIMENT)
best_score = 0
best_model = None
best_model_name = None
best_run_id = None

for name, cfg in models.items():
    print(f'\nTraining {name}...')
    with mlflow.start_run(run_name=name) as run:
        grid = GridSearchCV(cfg['model'], cfg['params'], cv=2, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"{name} Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")

        # Log params and metrics
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics({
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc_auc
        })
        mlflow.sklearn.log_model(grid.best_estimator_, name + "_model")

        # Track best model
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = grid.best_estimator_
            best_model_name = name
            best_run_id = run.info.run_id

# 5. Register the best model
if best_model is not None:
    print(f"\nRegistering best model: {best_model_name} (ROC-AUC: {best_score:.4f})")
    model_uri = f"runs:/{best_run_id}/{best_model_name}_model"
    mlflow.register_model(model_uri, "BestCreditRiskModel")
    print("Model registered in MLflow Model Registry as 'BestCreditRiskModel'.")
else:
    print("No model was trained.") 