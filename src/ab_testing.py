import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
def simulate_ab_test(df, rf_model, preprocessor, feature_names, top_k_percent=0.3, sample_size=5000):
    # Sample users with all relevant columns
    sample_users = df.sample(n=sample_size * 2, random_state=42)  # Total 3,940
    
    # Extract features and target
    features = [
        "email_text", "email_version", "hour_bin", "weekday", "user_country",
        "user_past_purchases", "is_weekend", "purchase_bin"
    ]
    X_sample = sample_users[features]
    y_sample = sample_users["clicked"]
    
    # Preprocess sampled data
    X_sample_transformed = preprocessor.transform(X_sample)
    X_sample_transformed = pd.DataFrame(X_sample_transformed, columns=feature_names, index=X_sample.index)
    
    # Predict probabilities
    y_prob = rf_model.predict_proba(X_sample_transformed)[:, 1]
    
    # Split into control (random) and treatment (model-based)
    control_mask = np.random.choice([True, False], size=sample_size * 2, replace=True, p=[0.5, 0.5])
    control_users = sample_users[control_mask].head(sample_size)
    treatment_users = sample_users[~control_mask].head(sample_size)
    
    # Baseline CTR (random)
    control_ctr = control_users["clicked"].mean()
    
    # Model-based CTR (top k%)
    treatment_probs = y_prob[~control_mask][:sample_size]
    treatment_sorted = treatment_users.assign(prob=treatment_probs).sort_values("prob", ascending=False)
    top_treatment = treatment_sorted.head(int(sample_size * top_k_percent))
    treatment_ctr = top_treatment["clicked"].mean()
    
    print("\n📊 A/B Test Simulation Results (Section 4):")
    print(f"Control CTR: {control_ctr*100:.2f}%")
    print(f"Treatment CTR: {treatment_ctr*100:.2f}%")
    print(f"CTR Lift: {(treatment_ctr - control_ctr)*100:.2f}%")
    
    # Statistical test
    count = np.array([top_treatment["clicked"].sum(), control_users["clicked"].sum()])
    nobs = np.array([len(top_treatment), len(control_users)])
    stat, p_value = stats.ttest_ind(
        top_treatment["clicked"], control_users["clicked"], alternative="greater"
    )
    print(f"p-value: {p_value:.4f}")
    plt.figure(figsize=(8, 6))
    plt.bar(["Control", "Treatment"], [control_ctr*100, treatment_ctr*100], color=['blue', 'green'])
    plt.title("A/B Test CTR Comparison")
    plt.ylabel("CTR (%)")
    plt.savefig('output/ab_test_ctr.png', bbox_inches='tight')
    plt.close()
    print("\n📊 A/B Test Simulation Results (Section 4):")
    print(f"Control CTR: {control_ctr*100:.2f}%")
    print(f"Treatment CTR: {treatment_ctr*100:.2f}%")
    print(f"CTR Lift: {(treatment_ctr - control_ctr)*100:.2f}%")
    
    # Statistical test
    stat, p_value = stats.ttest_ind(
        top_treatment["clicked"], control_users["clicked"], alternative="greater"
    )
    print(f"p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("✅ Statistically significant improvement detected!")
    else:
        print("❌ No significant improvement detected.")
