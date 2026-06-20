import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
import joblib

def load_data(workspace_path):
    """Loads Client.csv and Record.csv from the workspace path and merges them."""
    client_path = os.path.join(workspace_path, "Client.csv")
    record_path = os.path.join(workspace_path, "Record.csv")
    
    if not os.path.exists(client_path) or not os.path.exists(record_path):
        raise FileNotFoundError(f"Required CSV files (Client.csv and Record.csv) not found in {workspace_path}")
        
    print("Loading Client.csv...")
    client_df = pd.read_csv(client_path)
    print("Loading Record.csv...")
    record_df = pd.read_csv(record_path)
    
    print("Merging datasets...")
    df = pd.merge(client_df, record_df, on="Customer_ID")
    print(f"Merged Dataset Shape: {df.shape}")
    return df

def assign_plan(row):
    """Assigns optimal plan segment based on usage and overage."""
    mou = row['mou_Mean']
    ov = row['ovrrev_Mean']
    
    # Handle NaNs by replacing with 0
    if pd.isna(mou):
        mou = 0
    if pd.isna(ov):
        ov = 0
        
    if mou > 1200 or ov >= 20:
        return 3 # Unlimited
    elif mou > 600 or ov >= 10:
        return 2 # Premium
    elif mou > 150:
        return 1 # Standard
    else:
        return 0 # Economy

def engineer_features(df):
    """Engineers business-centric features and prepares data by dropping leakage columns."""
    print("Engineering features...")
    df['overage_ratio'] = df['ovrrev_Mean'] / (df['rev_Mean'] + 1e-5)
    df['revenue_per_minute'] = df['rev_Mean'] / (df['mou_Mean'] + 1e-5)
    df['dropped_call_ratio'] = (df['drop_vce_Mean'] + df['drop_dat_Mean']) / (df['attempt_Mean'] + 1e-5)
    df['completed_call_ratio'] = df['complete_Mean'] / (df['attempt_Mean'] + 1e-5)
    df['customer_care_ratio'] = df['custcare_Mean'] / (df['attempt_Mean'] + 1e-5)
    
    # Target Leakage Prevention: drop direct proxies of current usage and overage
    leakage_cols = [
        'Customer_ID', 'optimal_plan', 'churn',
        'mou_Mean', 'ovrrev_Mean', 'ovrmou_Mean', 'vceovr_Mean', 'datovr_Mean', 'rev_Mean'
    ]
    
    X = df.drop(columns=leakage_cols, errors='ignore')
    y = df['optimal_plan']
    return X, y

def preprocess_features(X):
    """Cleans up and preprocesses numeric and categorical columns."""
    X = X.copy()
    # Clean up numerical columns: Impute missing values with median
    num_cols = X.select_dtypes(include=[np.number]).columns
    print(f"Imputing missing values for {len(num_cols)} numerical columns...")
    for col in num_cols:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        
    # Clean up categorical columns: Impute missing values with 'Unknown' and Label Encode
    cat_cols = X.select_dtypes(include=['object']).columns
    print(f"Processing {len(cat_cols)} categorical columns...")
    for col in cat_cols:
        X[col] = X[col].fillna('Unknown')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    return X

def train_and_evaluate(X_train, y_train, X_test, y_test, workspace_path):
    """Trains LightGBM and XGBoost classifiers, selects the best, and saves results."""
    # Model 1: LightGBM
    print("\nTraining LightGBM Classifier...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.08,
        num_leaves=31,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    lgb_model.fit(X_train, y_train)
    
    y_pred_lgb = lgb_model.predict(X_test)
    acc_lgb = accuracy_score(y_test, y_pred_lgb)
    kappa_lgb = cohen_kappa_score(y_test, y_pred_lgb)
    
    print(f"LightGBM Test Accuracy: {acc_lgb:.4f}")
    print(f"LightGBM Cohen's Kappa: {kappa_lgb:.4f}")
    print("\nLightGBM Classification Report:")
    target_names = ['Economy', 'Standard', 'Premium', 'Unlimited']
    print(classification_report(y_test, y_pred_lgb, target_names=target_names))
    
    # Model 2: XGBoost
    print("\nTraining XGBoost Classifier...")
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    
    y_pred_xgb = xgb_model.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    kappa_xgb = cohen_kappa_score(y_test, y_pred_xgb)
    
    print(f"XGBoost Test Accuracy: {acc_xgb:.4f}")
    print(f"XGBoost Cohen's Kappa: {kappa_xgb:.4f}")
    print("\nXGBoost Classification Report:")
    print(classification_report(y_test, y_pred_xgb, target_names=target_names))
    
    # Save the best model
    if acc_lgb >= acc_xgb:
        best_model = lgb_model
        best_model_name = "LightGBM"
        y_pred_best = y_pred_lgb
    else:
        best_model = xgb_model
        best_model_name = "XGBoost"
        y_pred_best = y_pred_xgb
        
    print(f"\nBest Model selected: {best_model_name}")
    
    # Feature Importances
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importances.head(15))
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred_best)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Save artifacts
    model_out = os.path.join(workspace_path, 'best_plan_recommendation_model.pkl')
    feat_out = os.path.join(workspace_path, 'feature_importances.csv')
    conf_out = os.path.join(workspace_path, 'confusion_matrix.npy')
    
    joblib.dump(best_model, model_out)
    feature_importances.to_csv(feat_out, index=False)
    np.save(conf_out, conf_matrix)
    print(f"Model saved to {model_out}")
    print(f"Feature importances saved to {feat_out}")
    print(f"Confusion matrix saved to {conf_out}")

def main():
    # Use script directory as workspace so it is fully portable
    workspace_path = os.path.dirname(os.path.abspath(__file__))
    print(f"Workspace path set to: {workspace_path}")
    
    df = load_data(workspace_path)
    
    print("Creating target variable 'optimal_plan'...")
    df['optimal_plan'] = df.apply(assign_plan, axis=1)
    
    print("Distribution of optimal_plan:")
    dist = df['optimal_plan'].value_counts().sort_index()
    for k, v in dist.items():
        pct = (v / len(df)) * 100
        print(f"  Tier {k}: {v} customers ({pct:.2f}%)")
        
    X, y = engineer_features(df)
    X = preprocess_features(X)
    
    print("Splitting datasets into train and test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    train_and_evaluate(X_train, y_train, X_test, y_test, workspace_path)

if __name__ == "__main__":
    main()
