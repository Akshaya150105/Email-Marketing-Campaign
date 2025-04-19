'''
Computes SHAP values for a sample of data and generates two types of summary plots (bar and dot) 
to visualize feature importance and their effects on predictions.
'''
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

feature_name_map = {
    'num_user_past_purchases': 'Past Purchases',
    'cat_email_version_personalized': 'Personalized Email',
    'cat_user_country_US': 'User Country: US',
    'cat_email_text_short_email': 'Short Email',
    'cat_hour_bin_morning': 'Hour: Morning',
    'cat_purchase_bin_medium': 'Purchase Bin: Medium',
    'cat_purchase_bin_none': 'Purchase Bin: None',
    'cat_weekday_Wednesday': 'Weekday: Wednesday',
    'cat_hour_bin_night': 'Hour: Night',
    'cat_weekday_Tuesday': 'Weekday: Tuesday',
    'cat_weekday_Thursday': 'Weekday: Thursday',
    'cat_user_country_FR': 'User Country: France',
    'num_is_weekend': 'Is Weekend',
    'cat_hour_bin_evening': 'Hour: Evening',
    'cat_weekday_Sunday': 'Weekday: Sunday',
    'cat_weekday_Monday': 'Weekday: Monday',
    'cat_weekday_Saturday': 'Weekday: Saturday'
}

def explain_with_shap(model, X_sample, preprocessor, feature_names):
    print("🔍 Initializing SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    
    print("Transforming sample data...")
    X_sample_transformed = preprocessor.transform(X_sample)
    X_sample_transformed = pd.DataFrame(X_sample_transformed, columns=feature_names, index=X_sample.index)
    
    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_sample_transformed)
    
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        print("Detected 3D SHAP values, selecting class 1 for binary classification...")
        shap_vals_to_plot = shap_values[:, :, 1]
    else:
        shap_vals_to_plot = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values
    
    print(f" SHAP values shape: {shap_vals_to_plot.shape}")
    print(f" X_sample shape: {X_sample_transformed.shape}")
    
    # Map feature names for plotting
    mapped_feature_names = [feature_name_map.get(name, name) for name in feature_names]
    
    print(" SHAP Summary Bar Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_to_plot, X_sample_transformed, feature_names=mapped_feature_names, plot_type="bar", show=False)
    plt.title("SHAP Summary Bar Plot - Feature Importance")
    plt.tight_layout()
    plt.savefig('output/shap_bar.png', bbox_inches='tight')
    plt.close()
    
    print(" SHAP Summary Dot Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_to_plot, X_sample_transformed, feature_names=mapped_feature_names, show=False)
    plt.title("SHAP Summary Dot Plot - Feature Effects")
    plt.tight_layout()
    plt.savefig('output/shap_dot.png', bbox_inches='tight')
    plt.close()
    
    return shap_values