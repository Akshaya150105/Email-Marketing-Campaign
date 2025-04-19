from src.data_loader import load_data
from src.preprocessing import add_features
from src.ab_testing import simulate_ab_test
import matplotlib.pyplot as plt
from src.modeling import (
    prepare_model_data,
    train_models,
    simulate_targeted_email_strategy,
    threshold_tuning
)
from src.evaluation import (
    evaluate_model_performance,
)
from src.interpret import explain_with_shap
import pandas as pd
import joblib
import os

def main():
    os.makedirs('output', exist_ok=True)
    
    try:
        print("Starting data load...")
        df = load_data(
            "Data/email_table.csv",
            "Data/email_opened_table.csv",
            "Data/link_clicked_table.csv"
        )
        print("Data loaded successfully!")
        print("Data head:", df.head())
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    try:
        print("Starting feature addition...")
        df = add_features(df)
        print("Features added successfully!")
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return
    
    try:
        print("Starting model data preparation...")
        X_train, X_test, y_train, y_test, feature_names, preprocessor = prepare_model_data(df)
        print("Model data prepared successfully!")
    except Exception as e:
        print(f"Error preparing model data: {e}")
        return
    
    try:
        print("Starting model training...")
        results = train_models(X_train, X_test, y_train, y_test)
        print("Model training complete!")
        
        for model_name, output in results.items():
            print(f"\n🔍 Model: {model_name}")
            print(f"ROC AUC: {output['roc_auc']:.4f}")
            print(f"CV AUC: {output['cv_auc']:.4f}")
            print(output['report'])
        
        # Select best model based on CV AUC
        best_model_name = max(results, key=lambda x: results[x]['cv_auc'])
        best_model = results[best_model_name]['model']
        print(f"\n✅ Best model: {best_model_name} (CV AUC: {results[best_model_name]['cv_auc']:.4f})")
        
        # Save all models
        os.makedirs('output/models', exist_ok=True)
        for model_name, output in results.items():
            joblib.dump(output['model'], f'output/models/{model_name.replace(" ", "_")}.pkl')
        joblib.dump(best_model, 'output/best_model.pkl')
        
        # Use best model for downstream steps
        y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else best_model.predict(X_test)
        y_pred = best_model.predict(X_test)
    except Exception as e:
        print(f"Error training models: {e}")
        return
    
    try:
        print("Starting evaluation...")
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
        print("✅ Top users exported to 'output/top_users.csv'")
    except Exception as e:
        print(f"Error in evaluation or export: {e}")
        return
    
    try:
        print("Starting campaign simulation...")
        sim_result = simulate_targeted_email_strategy(
            model=best_model,
            X_test=X_test,
            y_test=y_test,
            top_k_percent=0.3
        )
        print("Campaign simulation complete!")
        print("\n📈 Campaign Simulation Results:")
        print(f"Baseline CTR:     {sim_result['baseline_ctr']*100:.2f}%")
        print(f"Model-Based CTR:  {sim_result['simulated_ctr']*100:.2f}%")
        print(f"Estimated Lift:   {sim_result['lift_percent']:.2f}%")
        print(f"Baseline ROI:     {sim_result['baseline_roi']:.2f}")
        print(f"Model ROI:        {sim_result['model_roi']:.2f}")
    except Exception as e:
        print(f"Error in simulation: {e}")
        return
    
    try:
        print("Starting model evaluation...")
        evaluate_model_performance(y_test, y_pred, y_prob)
        print("Model evaluation complete!")
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return
    
    try:
        print("Starting SHAP analysis...")
        X_sample = df.loc[X_test.index].sample(300, random_state=42)
        shap_values = explain_with_shap(best_model, X_sample, preprocessor, feature_names)
        print("✅ SHAP analysis complete")
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        return
    
    try:
        print("Starting A/B test simulation...")
        simulate_ab_test(df, best_model, preprocessor, feature_names)
        print("A/B test simulation complete!")
    except Exception as e:
        print(f"Error in A/B test simulation: {e}")
        return
    
    # Model Comparison Chart
    cv_aucs = {name: result['cv_auc'] for name, result in results.items()}
    plt.figure(figsize=(10, 6))
    plt.bar(cv_aucs.keys(), cv_aucs.values())
    plt.title("Model CV AUC Comparison")
    plt.ylabel("CV AUC")
    plt.xticks(rotation=45)
    plt.savefig('output/model_comparison.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()