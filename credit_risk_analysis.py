# ===== IMPORTS =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, roc_auc_score, 
                           precision_recall_curve, f1_score, 
                           confusion_matrix, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ===== DATA PREPARATION =====
def load_and_preprocess():
    """Load and clean the dataset"""
    df = pd.read_csv('cs-training.csv')
    
    # Clean data
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Handle missing values
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())
    
    return df

# ===== FEATURE ENGINEERING =====
def create_features(df):
    """Generate risk-related features"""
    
    # Debt burden features
    df['DebtToIncome'] = df['DebtRatio'] * df['MonthlyIncome']
    df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    
    # Delinquency features
    df['TotalDelinquencies'] = (df['NumberOfTime30-59DaysPastDueNotWorse'] + 
                               df['NumberOfTime60-89DaysPastDueNotWorse'] + 
                               df['NumberOfTimes90DaysLate'])
    
    df['SevereDelinquent'] = (df['NumberOfTimes90DaysLate'] > 0).astype(int)
    df['RecentDelinquent'] = (df['NumberOfTime30-59DaysPastDueNotWorse'] > 0).astype(int)
    
    # Credit utilization
    df['CreditUtilization'] = np.where(
        df['RevolvingUtilizationOfUnsecuredLines'] > 3, 
        3,  # Cap outliers
        df['RevolvingUtilizationOfUnsecuredLines']
    )
    
    # Payment history
    df['LatePaymentsPerYear'] = df['TotalDelinquencies'] / (df['age']/12 + 1)
    
    return df

# ===== MODEL TRAINING =====
def train_models(X_train, y_train):
    """Train multiple models with class weights"""
    
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            class_weight={0:1, 1:12},
            max_depth=8,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            scale_pos_weight=12,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

# ===== THRESHOLD OPTIMIZATION =====
def find_optimal_threshold(model, X_test, y_test):
    """Find threshold that maximizes F1 score"""
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    
    # Calculate F1 scores
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    
    return thresholds[optimal_idx]

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess()
    df = create_features(df)
    
    # Split data
    X = df.drop('SeriousDlqin2yrs', axis=1)
    y = df['SeriousDlqin2yrs']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle class imbalance
    print("Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Feature scaling and selection
    print("Scaling and selecting features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)
    
    selector = SelectKBest(f_classif, k=15)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_smote)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Train models
    print("Training models...")
    models = train_models(X_train_selected, y_train_smote)
    
    # Evaluate models
    print("\n=== Model Evaluation ===")
    best_f1 = 0
    best_model = None
    
    for name, model in models.items():
        y_pred = model.predict(X_test_selected)
        y_prob = model.predict_proba(X_test_selected)[:, 1]
        
        print(f"\n{name} Performance:")
        print(classification_report(y_test, y_pred))
        
        # Check if current model is best
        current_f1 = f1_score(y_test, y_pred)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model = model
    
    # Optimize threshold
    print("\nOptimizing decision threshold...")
    optimal_threshold = find_optimal_threshold(best_model, X_test_selected, y_test)
    final_probs = best_model.predict_proba(X_test_selected)[:, 1]
    final_preds = (final_probs > optimal_threshold).astype(int)
    
    print("\n=== Final Optimized Performance ===")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(classification_report(y_test, final_preds))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, final_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                display_labels=['Low Risk', 'High Risk'])
    disp.plot(cmap='Blues')
    plt.title('Optimized Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        features = X.columns[selector.get_support()]
        importances = best_model.feature_importances_
        
        fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df.head(10))
        plt.title('Top 10 Important Features')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    print("\nAnalysis complete! Check saved visualizations.")