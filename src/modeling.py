'''
This code is for preparing data, training machine learning models, 
simulating a targeted email campaign, and tuning prediction thresholds.'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_score,
    recall_score, f1_score
)

def prepare_model_data(df):
    features = [
        "email_text", "email_version", "hour_bin", "weekday", "user_country",
        "user_past_purchases", "is_weekend", "purchase_bin"
    ]
    
    categorical_cols = ["email_text", "email_version", "hour_bin", "weekday", "user_country", "purchase_bin"]
    numerical_cols = ["user_past_purchases", "is_weekend"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ])
    
    X = df[features]
    y = df["clicked"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    feature_names = preprocessor.get_feature_names_out().tolist()
    
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)
    
    return X_train_transformed, X_test_transformed, y_train, y_test, feature_names, preprocessor

# Model Training#
def train_models(X_train, X_test, y_train, y_test):
    results = {}
    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    cv_scores_lr = cross_val_score(lr, X_train, y_train, cv=5, scoring='roc_auc')
    results['Logistic Regression'] = {
        'report': classification_report(y_test, y_pred_lr, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob_lr),
        'cv_auc': cv_scores_lr.mean(),
        'model': lr
    }
    print("Logistic Regression trained!")
    
    # Random Forest
    print("Training Random Forest...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train_resampled, y_train_resampled)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    cv_scores_rf = cross_val_score(rf, X_train_resampled, y_train_resampled, cv=5, scoring='roc_auc')
    results['Random Forest'] = {
        'report': classification_report(y_test, y_pred_rf, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob_rf),
        'cv_auc': cv_scores_rf.mean(),
        'model': rf
    }
    print("Random Forest trained!")
    
    # XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb.fit(X_train_resampled, y_train_resampled)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    cv_scores_xgb = cross_val_score(xgb, X_train_resampled, y_train_resampled, cv=5, scoring='roc_auc')
    results['XGBoost'] = {
        'report': classification_report(y_test, y_pred_xgb, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob_xgb),
        'cv_auc': cv_scores_xgb.mean(),
        'model': xgb
    }
    print("XGBoost trained!")

    
    # Simple Neural Network
    print("Training Neural Network...")
    nn = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    nn.fit(X_train_resampled, y_train_resampled, epochs=5, batch_size=32, verbose=0)
    y_prob_nn = nn.predict(X_test, verbose=0).ravel()
    y_pred_nn = (y_prob_nn > 0.5).astype(int)
    
    # Manual cross-validation for Neural Network
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_nn = []
    for train_idx, val_idx in kfold.split(X_train_resampled):
        X_train_fold, X_val_fold = X_train_resampled.iloc[train_idx], X_train_resampled.iloc[val_idx]
        y_train_fold, y_val_fold = y_train_resampled.iloc[train_idx], y_train_resampled.iloc[val_idx]
        nn_clone = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        nn_clone.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        nn_clone.fit(X_train_fold, y_train_fold, epochs=5, batch_size=32, verbose=0)
        y_prob_val = nn_clone.predict(X_val_fold, verbose=0).ravel()
        cv_scores_nn.append(roc_auc_score(y_val_fold, y_prob_val))
    
    results['Neural Network'] = {
        'report': classification_report(y_test, y_pred_nn, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob_nn),
        'cv_auc': np.mean(cv_scores_nn),
        'model': nn
    }
    print("Neural Network trained!")
    return results

# Campaign Simulation #
def simulate_targeted_email_strategy(model, X_test, y_test, top_k_percent=0.3):
    probs = model.predict_proba(X_test)[:, 1]
    df_test = pd.DataFrame({'proba': probs, 'actual': y_test}, index=X_test.index)
    df_test_sorted = df_test.sort_values('proba', ascending=False)
    
    cutoff = int(len(df_test_sorted) * top_k_percent)
    top_users = df_test_sorted.iloc[:cutoff]
    
    simulated_ctr = top_users['actual'].mean()
    baseline_ctr = y_test.mean()
    
    cost_per_email = 0.01
    revenue_per_click = 10
    n_emails = cutoff
    baseline_clicks = int(baseline_ctr * n_emails)
    model_clicks = int(simulated_ctr * n_emails)
    baseline_cost = n_emails * cost_per_email
    model_cost = n_emails * cost_per_email
    baseline_revenue = baseline_clicks * revenue_per_click
    model_revenue = model_clicks * revenue_per_click
    roi_baseline = (baseline_revenue - baseline_cost) / baseline_cost if baseline_cost > 0 else 0
    roi_model = (model_revenue - model_cost) / model_cost if model_cost > 0 else 0
    
    return {
        'baseline_ctr': baseline_ctr,
        'simulated_ctr': simulated_ctr,
        'lift_percent': (simulated_ctr - baseline_ctr) / baseline_ctr * 100 if baseline_ctr > 0 else 0,
        'baseline_roi': roi_baseline,
        'model_roi': roi_model
    }



def threshold_tuning(y_test, y_prob):
    print("\nThreshold Tuning:")
    thresholds = np.arange(0.1, 0.9, 0.1)
    best_f1 = 0
    best_threshold = 0.5
    
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        print(f"Threshold: {t:.1f} | Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    print(f"Best Threshold: {best_threshold:.2f} (F1: {best_f1:.2f})")
    return best_threshold

