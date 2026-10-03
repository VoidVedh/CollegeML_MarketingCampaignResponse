"""
build_notebook.py
Constructs the complete, submission-ready narrative Jupyter Notebook
'notebooks/analysis.ipynb' for the Kaggle Marketing Dataset with:
- Google Colab compatibility badge and setup cell
- Verifiable code execution on real Kaggle data
- Target leakage exclusion documentation (duration)
- Real metric tables and embedded high-res figures
- Answers to the 6 core research questions
"""

import os
import json
import nbformat as nbf
import pandas as pd


def create_analysis_notebook(output_path="notebooks/analysis.ipynb"):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    metrics_path = os.path.join(project_root, "models", "metrics.json")
    
    with open(metrics_path, 'r') as f:
        m = json.load(f)

    winner_name = m['best_model_name']
    deployed_name = m['deployed_model_name']
    f1_t = m['f1_optimal_threshold']
    profit_t = m['profit_optimal_threshold']
    cost_per_contact = m['economic_parameters']['cost_per_contact']
    profit_per_responder = m['economic_parameters']['profit_per_responder']
    break_even_p = m['economic_parameters']['break_even_probability']
    
    boot_ci = m['bootstrap_confidence_intervals_profit_threshold']
    auc_ci = boot_ci['roc_auc']
    pr_ci = boot_ci['pr_auc']
    f1_ci = boot_ci['f1']
    
    cv_info = m['cv_metrics'][winner_name]
    odds_ratios = m.get('logistic_regression_odds_ratios', {})
    top_or_feat = max(odds_ratios, key=odds_ratios.get) if odds_ratios else "N/A"
    top_or_val = odds_ratios.get(top_or_feat, 1.0) if odds_ratios else 1.0

    res_csv_path = os.path.join(project_root, "reports", "results_comparison.csv")
    res_df = pd.read_csv(res_csv_path)
    res_df_winner = res_df[res_df['Model'] == winner_name].iloc[0]

    sim = m['business_simulation']
    top_20_capture_val = m.get('top_20_percent_capture', {}).get('capture_rate', 0.0) * 100

    nb = nbf.v4.new_notebook()
    cells = []
    
    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell(f"""# Marketing Campaign Response Prediction Using Machine Learning
**Case Study:** Case Study 157  
**Student:** Vedh Naik  
**Roll No.:** 150096725163  
**Cohort:** Jensen Huang  
**Repository:** `VoidVedh/CollegeML_MarketingCampaignResponse`  
**Dataset Source:** [Kaggle Marketing Dataset (Bank Marketing)](https://www.kaggle.com/competitions/marketing-dataset/data)  
**Selected Champion Architecture:** {deployed_name}  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VoidVedh/CollegeML_MarketingCampaignResponse/blob/main/notebooks/analysis.ipynb)

---

## 1. Executive Summary & Problem Formulation
Direct marketing campaigns represent a significant recurring investment for financial institutions and modern enterprises. Conventional mass outreach ("contacting every lead") wastes substantial marketing capital, fatigues uninterested clients, and degrades overall return on investment (ROI).

The objective of this project is to build an end-to-end, data-leakage-free machine learning system using the authentic, public **Kaggle Marketing Dataset** (41,188 client records; 11.27% baseline subscription rate) to identify clients who are **genuinely likely to subscribe to a bank term deposit** (`y = yes/no`). This enables marketing teams to prioritize outreach, reduce wasted ad spend by **{sim[3]['Cost Saved vs All (%)']:.1f}% to {sim[2]['Cost Saved vs All (%)']:.1f}%**, and deliver **${sim[3]['Net Profit ($)']:,.2f} in net campaign profit** ({sim[3]['ROI (%)']:.1f}% ROI) under assumed campaign economics.
"""))

    # Colab Setup Cell
    cells.append(nbf.v4.new_markdown_cell("""### 1.1 Colab Setup & Environment Configuration"""))
    cells.append(nbf.v4.new_code_cell("""# Environment configuration (Supports both local and Google Colab execution)
import os
import sys

if 'google.colab' in sys.modules:
    print("Running in Google Colab environment.")
    !git clone https://github.com/VoidVedh/CollegeML_MarketingCampaignResponse.git
    %cd CollegeML_MarketingCampaignResponse
    !pip install -r requirements.txt -q
    PROJECT_ROOT = os.path.abspath(".")
else:
    PROJECT_ROOT = os.path.abspath("..") if os.path.exists("../src") else os.path.abspath(".")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

from src.preprocess import (
    load_and_split_data, create_preprocessor, get_feature_names,
    unit_test_preprocessing, FEATURE_COLUMNS, NUMERIC_FEATURES,
    CATEGORICAL_FEATURES, DEFAULT_DATA_PATH, derive_age_group
)
from src.eda import run_full_eda

print("Environment configured successfully. Current working directory:", os.getcwd())
"""))

    # Section 2: Data Loading & Verification
    cells.append(nbf.v4.new_markdown_cell("""## 2. Kaggle Dataset Profile & Verification
The dataset is the authentic **Kaggle Marketing Dataset** comprising **41,188 client interactions** across 21 original columns.
"""))
    cells.append(nbf.v4.new_code_cell("""data_path = os.path.join(PROJECT_ROOT, "data", "kaggle", "train.csv")
if not os.path.exists(data_path):
    data_path = DEFAULT_DATA_PATH

df_raw = pd.read_csv(data_path)
print(f"Total Rows: {df_raw.shape[0]:,}")
print(f"Total Columns: {df_raw.shape[1]}")
print(f"Column Names: {list(df_raw.columns)}\\n")

# Target breakdown
target_dist = df_raw['y'].value_counts()
target_pct = df_raw['y'].value_counts(normalize=True).mul(100).round(2)
print("Target Distribution (y):")
for val, count in target_dist.items():
    print(f"  '{val}': {count:,} ({target_pct[val]}%)")
"""))

    cells.append(nbf.v4.new_code_cell("""# Data types and missing value audit
missing_info = pd.DataFrame({
    'Data_Type': df_raw.dtypes,
    'Missing_Count': df_raw.isna().sum(),
    'Missing_Pct': (df_raw.isna().sum() / len(df_raw) * 100).round(2)
})
print("Missing Value and Data Type Audit (Note: 'unknown' strings represent missing categorical data):")
display(missing_info)
"""))

    # Section 3: Target Leakage Prevention & Preprocessing Architecture
    cells.append(nbf.v4.new_markdown_cell("""## 3. Methodological Integrity: Leakage Prevention (`duration`) & Preprocessing

### 3.1 Strict Exclusion of Call Duration
> **Target Leakage Prohibition:** Kaggle explicitly documents that call `duration` is only known after/during a phone call. Incorporating `duration` would introduce critical target leakage and yield an un-deployable model. `duration` is **strictly excluded** from all predictive modeling.

### 3.2 Preprocessing Architecture
- **Numerical Pipeline (9 features):** Median Imputation $\\to$ `IQRCapper(factor=1.5)` $\\to$ `StandardScaler()`.
- **Categorical Pipeline (11 features):** Constant Imputer (`'unknown'`) $\\to$ `OneHotEncoder(drop='first', handle_unknown='ignore')`.
"""))

    cells.append(nbf.v4.new_code_cell("""# Run automated preprocessing unit test
unit_test_preprocessing()
print("PASS: Preprocessing unit tests verified (outlier capping, imputation, OneHot alignment).")
"""))

    cells.append(nbf.v4.new_code_cell("""X_train, X_test, y_train, y_test = load_and_split_data(data_path, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape} | Positive cases: {y_train.sum()} ({y_train.mean():.1%})")
print(f"X_test shape:  {X_test.shape}  | Positive cases: {y_test.sum()} ({y_test.mean():.1%})")

preprocessor = create_preprocessor()
preprocessor.fit(X_train)
feature_names = get_feature_names(preprocessor)
print(f"\\nTransformed output features ({len(feature_names)}):\\n{feature_names}")
"""))

    # Section 4: Exploratory Data Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 4. Exploratory Data Analysis (EDA)
EDA on the Kaggle Marketing Dataset reveals key demographic, behavioral, and macroeconomic response patterns.
"""))
    cells.append(nbf.v4.new_code_cell("""fig_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
col1 = Image(filename=os.path.join(fig_dir, "eda_target_distribution.png"), width=450)
col2 = Image(filename=os.path.join(fig_dir, "eda_response_rates_breakdown.png"), width=450)
display(col1, col2)
"""))

    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(fig_dir, "eda_numeric_distributions_boxplots.png"), width=750))"""))
    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(fig_dir, "eda_correlation_heatmap.png"), width=600))"""))

    # Section 5: Model Benchmark & Selection
    cells.append(nbf.v4.new_markdown_cell(f"""## 5. Comprehensive 6-Model Benchmark & Defensible Selection
All six required models were tuned using 5-fold Stratified Cross-Validation on the development partition:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Random Forest**
- **Naive Bayes (GaussianNB)**
- **Gradient Boosting**

**Defensible Selection Rule:**
{m['selection_justification']}
"""))

    cells.append(nbf.v4.new_code_cell("""comp_csv = os.path.join(PROJECT_ROOT, "reports", "results_comparison.csv")
results_df = pd.read_csv(comp_csv)
print("=== Algorithm Performance Benchmark on Held-Out Test Set (8,238 clients) ===")
display(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC', 'FPR', 'FPR_at_Recall_70', 'CV_PR_AUC_Mean', 'CV_PR_AUC_Std', 'CV_ROC_AUC_Mean', 'CV_ROC_AUC_Std']])
"""))

    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(fig_dir, "metrics_comparison_bar.png"), width=750))"""))

    cells.append(nbf.v4.new_code_cell("""col_roc = Image(filename=os.path.join(fig_dir, "roc_curves_all_models.png"), width=450)
col_pr = Image(filename=os.path.join(fig_dir, "pr_curves_all_models.png"), width=450)
display(col_roc, col_pr)
"""))

    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(fig_dir, "confusion_matrices_grid.png"), width=750))"""))

    # Section 6: Imbalance Handling
    cells.append(nbf.v4.new_markdown_cell("""## 6. Class Imbalance Treatments Analysis
We compared three imbalance treatments under identical 5-fold cross-validation:
1. None (Unweighted baseline)
2. `class_weight='balanced'`
3. SMOTENC (Categorical-aware synthetic oversampling)
"""))

    cells.append(nbf.v4.new_code_cell("""imb_csv = os.path.join(PROJECT_ROOT, "reports", "imbalance_handling_comparison.csv")
imb_df = pd.read_csv(imb_csv)
print("=== Imbalance Treatment Comparison (5-Fold CV PR-AUC & ROC-AUC) ===")
display(imb_df)
"""))

    # Section 7: OOF Threshold Optimization & Business Simulation
    cells.append(nbf.v4.new_markdown_cell(f"""## 7. Dual Threshold Optimization & Economic Business Simulation
Thresholds were tuned exclusively on training Out-Of-Fold (OOF) cross-validation predictions:
- **F1-Optimal Threshold ($t = {f1_t:.2f}$):** Maximizes harmonic mean of precision and recall.
- **Profit-Optimal Threshold ($t = {profit_t:.2f}$):** Maximizes simulated campaign net profit under assumed economics (contact cost = ${cost_per_contact:.2f}, responder gross profit = ${profit_per_responder:.2f}, break-even probability = {break_even_p:.2f}).
"""))

    cells.append(nbf.v4.new_code_cell("""sim_csv = os.path.join(PROJECT_ROOT, "reports", "business_simulation.csv")
sim_df = pd.read_csv(sim_csv)
print("=== 4-Strategy Economic Business Simulation on Held-Out Test Set ===")
display(sim_df)
"""))

    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(fig_dir, "business_simulation_roi.png"), width=750))"""))
    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(fig_dir, "cumulative_gains_lift.png"), width=750))"""))

    # Section 8: Probability Calibration
    cells.append(nbf.v4.new_markdown_cell("""## 8. Probability Calibration (FrozenEstimator)
Using scikit-learn's modern `FrozenEstimator` wrapped within `CalibratedClassifierCV`, we evaluated probability calibration on validation data:
"""))
    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(fig_dir, "calibration_curve.png"), width=600))"""))

    # Section 9: Explainability & Profiling
    cells.append(nbf.v4.new_markdown_cell("""## 9. Explainability & Customer Persona Profiling
We analyze feature importance through Tree MDI, Permutation Importance, Odds Ratios, and SHAP:
"""))
    cells.append(nbf.v4.new_code_cell("""col_mdi = Image(filename=os.path.join(fig_dir, "feature_importance_tree.png"), width=450)
col_perm = Image(filename=os.path.join(fig_dir, "feature_importance_permutation.png"), width=450)
display(col_mdi, col_perm)
"""))

    cells.append(nbf.v4.new_code_cell("""col_or = Image(filename=os.path.join(fig_dir, "feature_importance_odds_ratios.png"), width=450)
shap_path = os.path.join(fig_dir, "shap_summary.png")
if os.path.exists(shap_path):
    col_shap = Image(filename=shap_path, width=450)
    display(col_or, col_shap)
else:
    display(col_or)
"""))

    cells.append(nbf.v4.new_code_cell("""prof_csv = os.path.join(PROJECT_ROOT, "reports", "customer_profiles.csv")
prof_df = pd.read_csv(prof_csv, header=[0, 1], index_col=0)
print("=== Customer Persona Profiles: Actual vs Predicted Responders ===")
display(prof_df)
"""))

    # Section 10: Verified Answers to Assignment Questions
    cells.append(nbf.v4.new_markdown_cell(f"""## 10. Direct Answers to the 6 Core Research Questions

### Question 1: Can campaign responses be predicted?
**Answer:** **Yes.**  
On the authentic Kaggle dataset without `duration`, the deployed model achieves a held-out test **ROC-AUC of {res_df_winner['ROC-AUC']:.4f}** (95% bootstrap CI: [{auc_ci['ci_lower']:.4f}, {auc_ci['ci_upper']:.4f}]) and **PR-AUC of {res_df_winner['PR-AUC']:.4f}** (95% bootstrap CI: [{pr_ci['ci_lower']:.4f}, {pr_ci['ci_upper']:.4f}]). Contacting the top 20% of ranked clients captures **{top_20_capture_val:.1f}% of all actual subscribers**, demonstrating substantial predictive lift over random targeting.

### Question 2: Which customer characteristics influence response?
**Answer:** **Macroeconomic conditions, timing, and previous campaign outcome are the primary drivers.**  
1. `{top_or_feat}` and `poutcome_success`: The top positive driver by Odds Ratio is `{top_or_feat}` at **{top_or_val:.4f}**, and prior campaign success (`poutcome_success`) yields an Odds Ratio of **{odds_ratios.get('poutcome_success', top_or_val):.4f}**, confirming strong positive re-engagement.
2. `euribor3m` and `emp.var.rate`: Lower interest rates correlate with higher propensity to subscribe to bank term deposits.
3. Occupation: Students and retired clients display higher relative conversion propensity.

### Question 3: Which algorithm performs best?
**Answer:** **{winner_name} (Deployed: {deployed_name}).**  
Selected using 5-fold cross-validation on training data alone: CV PR-AUC = **{cv_info['cv_pr_auc_mean']:.4f} ± {cv_info['cv_pr_auc_std']:.4f}**, CV ROC-AUC = **{cv_info['cv_roc_auc_mean']:.4f} ± {cv_info['cv_roc_auc_std']:.4f}**. {m['selection_justification']}

### Question 4: Can ML reduce unnecessary marketing expenditure?
**Answer:** **Yes, saving {sim[3]['Cost Saved vs All (%)']:.1f}% to {sim[2]['Cost Saved vs All (%)']:.1f}% under assumed campaign economics.**  
In the 8,238-client test set:
- Mass outreach costs **${sim[0]['Total Cost ($)']:,.2f}** with {int(sim[0]['Wasted Contacts (FP)'])} wasted contacts, yielding **${sim[0]['Net Profit ($)']:,.2f}** in net profit.
- Profit-Optimal targeting costs **${sim[3]['Total Cost ($)']:,.2f}**, saving **${sim[0]['Total Cost ($)'] - sim[3]['Total Cost ($)']:,.2f} ({sim[3]['Cost Saved vs All (%)']:.1f}% cost reduction)** while maximizing simulated net profit to **${sim[3]['Net Profit ($)']:,.2f}**.

### Question 5: How can false positives be reduced?
**Answer:** **Through decision threshold optimization.**  
Operating at the default 0.50 threshold restricts False Positives, but misses subscribers. By tuning the operating threshold using training Out-Of-Fold predictions, decision makers can explicitly manage the trade-off between false-positive costs and false-negative opportunity loss.

### Question 6: Can the model identify potential campaign responders?
**Answer:** **Yes, capturing {sim[3]['Responders Reached']/int(res_df_winner['TP']+res_df_winner['FN'])*100:.1f}% of subscribers at the profit-optimal threshold while contacting only {sim[3]['Targeted Contacts']/int(res_df_winner['TN']+res_df_winner['FP']+res_df_winner['FN']+res_df_winner['TP'])*100:.1f}% of the client base.**

---

## 11. Limitations & Risk Disclosures
1. **Case Study Scope:** The Kaggle dataset reflects banking term deposit subscriptions; it does not contain retail-specific variables like website visits or discount coupons.
2. **Propensity vs Uplift:** The model measures response correlation rather than causal incrementality. A/B testing is recommended to measure true marketing lift.
3. **Simulated Economic Assumptions:** Unit costs (${cost_per_contact:.2f}) and responder profits (${profit_per_responder:.2f}) are simulation parameters.
4. **Call Duration Exclusion:** `duration` is deliberately omitted to prevent target leakage.
"""))

    nb.cells = cells
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Constructed narrative notebook at '{output_path}'.")
    return output_path


if __name__ == '__main__':
    create_analysis_notebook()
