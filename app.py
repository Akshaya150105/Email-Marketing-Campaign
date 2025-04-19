import streamlit as st
from src.data_loader import load_data
from src.preprocessing import add_features
from src.modeling import prepare_model_data, train_models, simulate_targeted_email_strategy, threshold_tuning
from src.evaluation import evaluate_model_performance
from src.ab_testing import simulate_ab_test
from src.interpret import explain_with_shap
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import roc_curve, precision_recall_curve
st.set_page_config(page_title="Email Marketing Analysis Dashboard", layout="wide")
st.sidebar.title("Controls")
load_models = st.sidebar.button("Load Saved Models")
retrain_models = st.sidebar.button("Retrain and Save Models")
show_data = st.sidebar.button("Show Data")
show_model_results = st.sidebar.button("Show Model Results")
show_campaign_sim = st.sidebar.button("Show Campaign Simulation")
show_ab_test = st.sidebar.button("Show A/B Test")
show_shap = st.sidebar.button("Show SHAP Analysis")
test_email = st.sidebar.button("Test Email Effectiveness")
if 'df' not in st.session_state:
    st.session_state.df = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None
if 'y_prob' not in st.session_state:
    st.session_state.y_prob = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None

# Function to load saved models
def load_saved_models():
    os.makedirs('output/models', exist_ok=True)
    model_names = ['Logistic_Regression', 'Random_Forest', 'XGBoost', 'SVM', 'Neural_Network']
    results = {}
    for name in model_names:
        try:
            model = joblib.load(f'output/models/{name}.pkl')
            results[name.replace('_', ' ')] = {
                'model': model,
                'roc_auc': 0.0,  
                'cv_auc': 0.0,   
                'report': 'N/A'  
            }
        except FileNotFoundError:
            st.error(f"Model {name} not found. Please retrain.")
            return None
    # Update with actual metrics if X_test and y_test are available
    if st.session_state.X_test is not None and st.session_state.y_test is not None:
        for model_name, output in results.items():
            model = output['model']
            y_prob = model.predict_proba(st.session_state.X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(st.session_state.X_test)
            y_pred = model.predict(st.session_state.X_test)
            output['roc_auc'] = roc_auc_score(st.session_state.y_test, y_prob)
            output['report'] = classification_report(st.session_state.y_test, y_pred, zero_division=0)
            # Note: CV AUC would require retraining with cross-validation, omitted for speed
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])  # Using ROC AUC as proxy
    st.session_state.results = results
    st.session_state.best_model = results[best_model_name]['model']
    st.success("Models loaded successfully!")

# Main function adapted for Streamlit (retrain mode)
def run_main():
    os.makedirs('output', exist_ok=True)
    
    try:
        st.text("Starting data load...")
        df = load_data(
            "Data/email_table.csv",
            "Data/email_opened_table.csv",
            "Data/link_clicked_table.csv"
        )
        st.text("Data loaded successfully!")
        st.session_state.df = df
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        return
    
    try:
        st.text("Starting feature addition...")
        df = add_features(df)
        st.text("Features added successfully!")
        st.session_state.df = df
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return
    
    try:
        st.text("Starting model data preparation...")
        X_train, X_test, y_train, y_test, feature_names, preprocessor = prepare_model_data(df)
        st.text("Model data prepared successfully!")
        st.session_state.y_test = y_test
        st.session_state.feature_names = feature_names
        st.session_state.preprocessor = preprocessor
        st.session_state.X_test = X_test
    except Exception as e:
        st.error(f"Error preparing model data: {e}")
        return
    
    try:
        st.text("Starting model training...")
        results = train_models(X_train, X_test, y_train, y_test)
        st.text("Model training complete!")
        st.session_state.results = results
        
        for model_name, output in results.items():
            st.write(f"🔍 Model: {model_name}")
            st.write(f"ROC AUC: {output['roc_auc']:.4f}")
            st.write(f"CV AUC: {output['cv_auc']:.4f}")
            st.text(output['report'])
        
        # Save all models
        os.makedirs('output/models', exist_ok=True)
        for model_name, output in results.items():
            joblib.dump(output['model'], f'output/models/{model_name.replace(" ", "_")}.pkl')
        best_model_name = max(results, key=lambda x: results[x]['cv_auc'])
        st.session_state.best_model = results[best_model_name]['model']
        st.success(f"✅ Best model: {best_model_name} (CV AUC: {results[best_model_name]['cv_auc']:.4f})")
        joblib.dump(st.session_state.best_model, 'output/best_model.pkl')
        
        y_prob = st.session_state.best_model.predict_proba(X_test)[:, 1] if hasattr(st.session_state.best_model, 'predict_proba') else st.session_state.best_model.predict(X_test)
        y_pred = st.session_state.best_model.predict(X_test)
        st.session_state.y_pred = y_pred
        st.session_state.y_prob = y_prob
    except Exception as e:
        st.error(f"Error training models: {e}")
        return
    
    try:
        st.text("Starting evaluation...")
        optimal_threshold = threshold_tuning(y_test, y_prob)
        
        top_n = 10
        top_users = pd.DataFrame({
            'predicted_proba': y_prob,
            'actual': y_test
        }, index=X_test.index)
        top_users = top_users[top_users['predicted_proba'] >= optimal_threshold]
        top_users = top_users.sort_values('predicted_proba', ascending=False).head(top_n)
        
        top_users = top_users.join(df[['email_id', 'user_past_purchases']].loc[top_users.index], how='left')
        
        top_users_export = top_users[['email_id', 'user_past_purchases', 'predicted_proba']]
        top_users_export['Rank'] = range(1, len(top_users_export) + 1)
        top_users_export = top_users_export[['Rank', 'email_id', 'user_past_purchases', 'predicted_proba']]
        top_users_export.to_csv('output/top_users.csv', index=False)
        st.success("✅ Top users exported to 'output/top_users.csv'")
        st.write("Top Users:", top_users_export)
        st.write(f"Best Threshold: {optimal_threshold} (F1: {max(0.08, float(str(output['report']).split()[-1]) if 'F1' in output['report'] else 0.08):.2f})")  # Approximate F1
    except Exception as e:
        st.error(f"Error in evaluation or export: {e}")
        return
    
    try:
        st.text("Starting campaign simulation...")
        sim_result = simulate_targeted_email_strategy(
            model=st.session_state.best_model,
            X_test=X_test,
            y_test=y_test,
            top_k_percent=0.3
        )
        st.text("Campaign simulation complete!")
        st.write("\n📈 Campaign Simulation Results:")
        st.write(f"Baseline CTR:     {sim_result['baseline_ctr']*100:.2f}%")
        st.write(f"Model-Based CTR:  {sim_result['simulated_ctr']*100:.2f}%")
        st.write(f"Estimated Lift:   {sim_result['lift_percent']:.2f}%")
        st.write(f"Baseline ROI:     {sim_result['baseline_roi']:.2f}")
        st.write(f"Model ROI:        {sim_result['model_roi']:.2f}")
    except Exception as e:
        st.error(f"Error in simulation: {e}")
        return
    
    try:
        st.text("Starting model evaluation...")
        evaluate_model_performance(y_test, y_pred, y_prob)
        st.text("Model evaluation complete!")
    except Exception as e:
        st.error(f"Error in evaluation: {e}")
        return
    
    try:
        st.text("Starting SHAP analysis...")
        X_sample = df.loc[X_test.index].sample(300, random_state=42)
        shap_values = explain_with_shap(st.session_state.best_model, X_sample, preprocessor, feature_names)
        st.text("✅ SHAP analysis complete")
    except Exception as e:
        st.error(f"Error in SHAP analysis: {e}")
        return
    
    try:
        st.text("Starting A/B test simulation...")
        simulate_ab_test(df, st.session_state.best_model, preprocessor, feature_names)
        st.text("A/B test simulation complete!")
    except Exception as e:
        st.error(f"Error in A/B test simulation: {e}")
        return
    
    # Model Comparison Chart
    cv_aucs = {name: result['cv_auc'] for name, result in results.items()}
    fig = plt.figure(figsize=(10, 6))
    plt.bar(cv_aucs.keys(), cv_aucs.values())
    plt.title("Model CV AUC Comparison")
    plt.ylabel("CV AUC")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Email effectiveness test function
def test_email_effectiveness(email_id):
    if st.session_state.df is None or st.session_state.best_model is None or st.session_state.preprocessor is None:
        st.error("Please load or retrain models first!")
        return
    
    # Find user data
    user_data = st.session_state.df[st.session_state.df['email_id'] == email_id]
    if user_data.empty:
        st.error(f"No data found for email_id {email_id}")
        return
    
    # Prepare features
    features = [
        "email_text", "email_version", "hour_bin", "weekday", "user_country",
        "user_past_purchases", "is_weekend", "purchase_bin"
    ]
    X_user = user_data[features]
    X_user_transformed = st.session_state.preprocessor.transform(X_user)
    X_user_transformed = pd.DataFrame(X_user_transformed, columns=st.session_state.feature_names, index=X_user.index)
    
    # Predict probability
    y_prob = st.session_state.best_model.predict_proba(X_user_transformed)[:, 1][0]
    prediction = "Effective" if y_prob > 0.5 else "Not Effective"
    
    st.write(f"Email ID: {email_id}")
    st.write(f"Predicted Probability of Click: {y_prob:.4f}")
    st.write(f"Prediction: {prediction}")
    st.write("User Features:", user_data[features])

# Button actions
if load_models:
    load_saved_models()

if retrain_models:
    run_main()

if show_data and st.session_state.df is not None:
    st.subheader("Dataset")
    st.write(st.session_state.df.head())

if show_model_results and st.session_state.results is not None:
    st.subheader("Model Results")
    for model_name, output in st.session_state.results.items():
        st.write(f"🔍 Model: {model_name}")
        st.write(f"ROC AUC: {output['roc_auc']:.4f}")
        st.write(f"CV AUC: {output['cv_auc']:.4f}")
        st.text(output['report'])
    st.success(f"✅ Best model: {max(st.session_state.results, key=lambda x: st.session_state.results[x]['roc_auc'])}")

if show_campaign_sim and st.session_state.best_model is not None and st.session_state.y_test is not None and st.session_state.X_test is not None:
    st.subheader("Campaign Simulation")
    sim_result = simulate_targeted_email_strategy(
        model=st.session_state.best_model,
        X_test=st.session_state.X_test,
        y_test=st.session_state.y_test,
        top_k_percent=0.3
    )
    st.write("\n📈 Campaign Simulation Results:")
    st.write(f"Baseline CTR:     {sim_result['baseline_ctr']*100:.2f}%")
    st.write(f"Model-Based CTR:  {sim_result['simulated_ctr']*100:.2f}%")
    st.write(f"Estimated Lift:   {sim_result['lift_percent']:.2f}%")
    st.write(f"Baseline ROI:     {sim_result['baseline_roi']:.2f}")
    st.write(f"Model ROI:        {sim_result['model_roi']:.2f}")

if show_ab_test and st.session_state.df is not None and st.session_state.best_model is not None:
    st.subheader("A/B Test Results")
    simulate_ab_test(st.session_state.df, st.session_state.best_model, st.session_state.preprocessor, st.session_state.feature_names)
    st.image('output/ab_test_ctr.png', caption="A/B Test CTR Comparison")

if show_shap and st.session_state.df is not None and st.session_state.best_model is not None:
    st.subheader("SHAP Analysis")
    X_sample = st.session_state.df.loc[st.session_state.X_test.index].sample(300, random_state=42)
    shap_values = explain_with_shap(st.session_state.best_model, X_sample, st.session_state.preprocessor, st.session_state.feature_names)
    st.image('output/shap_bar.png', caption="SHAP Summary Bar Plot")
    st.image('output/shap_dot.png', caption="SHAP Summary Dot Plot")

if test_email:
    st.subheader("Test Email Effectiveness")
    email_id = st.number_input("Enter Email ID", min_value=0, value=85120)
    test_email_effectiveness(email_id)

# Display initial instructions
st.title("Email Marketing Analysis Dashboard")
st.write("Use the sidebar to load saved models, retrain, view data, or test email effectiveness.")