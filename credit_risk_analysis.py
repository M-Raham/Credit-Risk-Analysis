# ===== IMPORT LIBRARIES =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ===== LOAD & PREPROCESS DATA =====
# Load dataset (download from: https://www.kaggle.com/c/GiveMeSomeCredit/data)
df = pd.read_csv('cs-training.csv')

# Drop unnecessary column
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Handle missing values - use direct assignment instead of inplace
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())

# ===== FEATURE ENGINEERING =====
def create_features(df):
    # Debt-to-income ratio (using existing DebtRatio instead of MonthlyDebtPayment)
    df['DebtToIncomeRatio'] = df['DebtRatio'] * df['MonthlyIncome']
    
    # Payment burden approximation (using DebtRatio since MonthlyDebtPayment isn't available)
    df['PaymentBurden'] = df['DebtRatio']
    
    # Past due amounts
    df['TotalPastDue'] = df['NumberOfTimes90DaysLate'] + df['NumberOfTime60-89DaysPastDueNotWorse']
    
    # Credit behavior features
    df['RecentDelinquencies'] = df['NumberOfTime30-59DaysPastDueNotWorse'] / (df['age'] / 12)
    df['RecentDelinquencies'] = df['RecentDelinquencies'].replace([np.inf, -np.inf], 0)
    
    return df

df = create_features(df)

# ===== SPLIT DATA INTO TRAIN & TEST =====
X = df.drop('SeriousDlqin2yrs', axis=1)
y = df['SeriousDlqin2yrs']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== HANDLE CLASS IMBALANCE (SMOTE) =====
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# ===== SCALE FEATURES =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# ===== MODEL TRAINING =====
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]))
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train_smote)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    
    results[name] = {
        'roc_auc': roc_auc_score(y_test, y_prob),
        'pr_auc': average_precision_score(y_test, y_prob),
        'model': model
    }

# ===== SELECT BEST MODEL =====
best_model_name = max(results.items(), key=lambda x: x[1]['pr_auc'])[0]
best_model = results[best_model_name]['model']
print(f"\nBest Model: {best_model_name}")

# ===== OPTIMIZE THRESHOLD FOR HIGH PRECISION =====
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Choose threshold for 80% precision
target_precision = 0.8
optimal_idx = np.argmax(precision >= target_precision)
optimal_threshold = thresholds[optimal_idx]

final_predictions = (y_prob > optimal_threshold).astype(int)

print("\n=== Final Performance ===")
print(classification_report(y_test, final_predictions))

# ===== FEATURE IMPORTANCE =====
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))

# ===== CONFUSION MATRIX VISUALIZATION =====
cm = confusion_matrix(y_test, final_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (Optimized Threshold)')
plt.show()

# ===== PREDICT NEW CUSTOMERS =====
def assess_credit_risk(customer_data, model=best_model, threshold=optimal_threshold, scaler=scaler):
    customer_df = pd.DataFrame([customer_data])
    customer_df = create_features(customer_df)
    customer_scaled = scaler.transform(customer_df)
    risk_prob = model.predict_proba(customer_scaled)[0, 1]
    is_high_risk = risk_prob > threshold
    
    return {
        'risk_probability': float(risk_prob),
        'is_high_risk': bool(is_high_risk),
        'threshold_used': float(threshold)
    }