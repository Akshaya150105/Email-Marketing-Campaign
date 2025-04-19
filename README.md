# Email Marketing Campaign Optimization Project

## Overview
This project leverages machine learning techniques to optimize an email marketing campaign by predicting click-through rates (CTR), identifying key user segments, and evaluating campaign performance. The analysis is supported by exploratory data analysis (EDA), feature engineering, model training, and simulation, and implemented through Python scripts and Jupyter notebooks.

## Project Structure
- **`Data/`**: Contains input data files:
  - `email_table.csv`: Main email data with columns like `email_id`, `email_text`, `email_version`, `hour`, `weekday`, `user_country`, `user_past_purchases`, `opened`, and `clicked`.
  - `email_opened_table.csv`: Tracks email open events.
  - `link_clicked_table.csv`: Tracks link click events.
- **`src/`**: Source code directory with custom functions:
  - `data_loader.py`: Loads and merges the data tables.
  - Other modules (e.g., `add_features`, `train_models`, `threshold_tuning`, etc.) used for preprocessing, modeling, and evaluation.
- **`output/`**: Stores generated outputs:
  - `best_model.pkl`: Saved best model (XGBoost, CV AUC: 0.9741).
  - `top_users.csv`: Top 10 users selected for targeting based on predicted probabilities.
  - `model_comparison.png`: Bar chart of CV AUCs across models.
  - `ab_test_ctr.png`: Bar chart of A/B test CTR results.
- **`01_exploration.ipynb`**: Initial data exploration notebook providing statistics, categorical distributions, and basic insights (e.g., 100,000 rows, no missing values).
- **`02_eda_and_features.ipynb`**: EDA and feature engineering notebook with click-to-open rate (20.48%), heatmap of click rates (e.g., Saturday 11 PM: 0.12), and derived features (`hour_bin`, `is_weekend`, `purchase_bin`).
- **`README.md`**: This file, providing project overview and instructions.
- **"Email Marketing Campaign Optimization Report"**: Detailed report summarizing the methodology, findings, and results.

## Methodology
1. **Data Preprocessing**:
   - Loaded and merged three tables using `load_data` from `01_exploration.ipynb`.
   - Added features (`hour_bin`, `is_weekend`, `purchase_bin`) using `add_features` in `02_eda_and_features.ipynb`.
2. **Exploratory Data Analysis**:
   - Analyzed data statistics (e.g., mean `user_past_purchases`: 3.88, `hour` range: 1-24) in `01_exploration.ipynb`.
   - Calculated click-to-open rate (20.48%) and identified optimal click times (Saturday 11 PM: 0.12, Tuesday 11 PM: 0.11) in `02_eda_and_features.ipynb`.
3. **Modeling**:
   - Trained Logistic Regression, Random Forest, XGBoost, and Neural Network models using `train_models`.
   - Selected the best model (XGBoost, CV AUC: 0.9741) and saved it to `output/best_model.pkl`.
4. **Inference and Evaluation**:
   - Evaluated model performance with precision, recall, F1 score, and ROC AUC using `evaluate_model_performance`.
   - Determined an optimal threshold (e.g., 0.40) with `threshold_tuning` and exported the top 10 users to `output/top_users.csv`.
5. **Campaign Simulation and A/B Testing**:
   - Simulated targeted email strategy with a CTR improvement from 2.12% to 3.72% (75.58% lift) and ROI from 20.11 to 36.11.
   - Conducted A/B testing with 10,000 users, targeting the top 30% (1,500 users), and visualized results in `ab_test_ctr.png`.

## Requirements
- Python 3.10.11
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `scikit-learn`, `xgboost`, `shap`, `joblib`
- Install dependencies via:
  ```bash
  pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost shap joblib
## To Run
     python main.py
