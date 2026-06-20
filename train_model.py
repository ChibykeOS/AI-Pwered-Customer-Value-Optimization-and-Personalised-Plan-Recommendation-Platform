import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import xgboost as xgb
import lightgbm as lgb
import joblib

workspace = "c:/final-assignment-gci"
client_path = os.path.join(workspace, "Client.csv")
record_path = os.path.join(workspace, "Record.csv")

print("Loading Client.csv...")
client_df = pd.read_csv(client_path)
print("Loading Record.csv...")
record_df = pd.read_csv(record_path)

print("Merging datasets...")
df = pd.merge(client_df, record_df, on="Customer_ID")
print(f"Merged Dataset Shape: {df.shape}")

# Define the Target Variable: Optimal Plan Segment
# Tier 0: Economy (mou_Mean <= 150 & ovrrev_Mean < 10)
# Tier 1: Standard (150 < mou_Mean <= 600 & ovrrev_Mean < 10)
# Tier 2: Premium ((600 < mou_Mean <= 1200) or (mou_Mean <= 600 & 10 <= ovrrev_Mean < 20))
# Tier 3: Unlimited (mou_Mean > 1200 or ovrrev_Mean >= 20)

def assign_plan(row):
    mou = row['mou_Mean']
    ovr = row['ovrrev_Mean']
    
    # Handle NaNs by replacing with median or 0
    if pd.isna(mou):
        mou = 0
    if pd.isna(ovr):
        ovr = 0
        
    if mou > 1200 or ovr >= 20:
        return 3 # Unlimited
    elif mou > 600 or ovr >= 10:
        return 2 # Premium
    elif mou > 150:
        return 1 # Standard
    else:
        return 0 # Economy

print("Creating target variable 'optimal_plan'...")
df['optimal_plan'] = df.apply(assign_plan, axis=1)

print("Distribution of optimal_plan:")
dist = df['optimal_plan'].value_counts().sort_index()
for k, v in dist.items():
    pct = (v / len(df)) * 100
    print(f"  Tier {k}: {v} customers ({pct:.2f}%)")

# Preprocessing & Feature Engineering
# Let's engineer some business-focused features
print("Engineering features...")
df['overage_ratio'] = df['ovrrev_Mean'] / (df['rev_Mean'] + 1e-5)
df['revenue_per_minute'] = df['rev_Mean'] / (df['mou_Mean'] + 1e-5)
df['dropped_call_ratio'] = (df['drop_vce_Mean'] + df['drop_dat_Mean']) / (df['attempt_Mean'] + 1e-5)
df['completed_call_ratio'] = df['complete_Mean'] / (df['attempt_Mean'] + 1e-5)
df['customer_care_ratio'] = df['custcare_Mean'] / (df['attempt_Mean'] + 1e-5)

# Columns to drop to avoid target leakage (direct proxies of current usage and current overage)
leakage_cols = [
    'Customer_ID', 'optimal_plan', 'churn',
    'mou_Mean', 'ovrrev_Mean', 'ovrmou_Mean', 'vceovr_Mean', 'datovr_Mean', 'rev_Mean'
]

# Separate features and target
X = df.drop(columns=leakage_cols, errors='ignore')
y = df['optimal_plan']

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

# Train-Test Split
print("Splitting datasets into train and test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

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
print(classification_report(y_test, y_pred_lgb, target_names=['Economy', 'Standard', 'Premium', 'Unlimited']))

# Model 2: XGBoost
print("\nTraining XGBoost Classifier...")
# We use multi:softmax and tune class weights implicitly or use scale_pos_weight for binary, 
# for multi-class we can balance using sample_weights
from sklearn.utils.class_weight import compute_sample_weight
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
print(classification_report(y_test, y_pred_xgb, target_names=['Economy', 'Standard', 'Premium', 'Unlimited']))

# Let's save the best model and get feature importances
best_model = lgb_model if acc_lgb >= acc_xgb else xgb_model
best_model_name = "LightGBM" if acc_lgb >= acc_xgb else "XGBoost"
print(f"\nBest Model selected: {best_model_name}")

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importances.head(15))

# Save feature importances and confusion matrix details for slides
conf_matrix = confusion_matrix(y_test, lgb_model.predict(X_test) if best_model_name == "LightGBM" else xgb_model.predict(X_test))
print("\nConfusion Matrix:")
print(conf_matrix)

# Save best model to disk
joblib.dump(best_model, os.path.join(workspace, 'best_plan_recommendation_model.pkl'))
feature_importances.to_csv(os.path.join(workspace, 'feature_importances.csv'), index=False)
np.save(os.path.join(workspace, 'confusion_matrix.npy'), conf_matrix)
print("Model, feature importances, and confusion matrix saved successfully.")
