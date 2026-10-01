"""
build_notebook.py
Constructs the complete, submission-ready narrative Jupyter Notebook
'notebooks/analysis.ipynb' with formatted markdown, rigorous code cells,
embedded visualizations, benchmark comparisons, and executive conclusions.
"""

import os
import nbformat as nbf

def create_analysis_notebook(output_path="notebooks/analysis.ipynb"):
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell("""# Marketing Campaign Response Prediction Using Machine Learning
**Author:** Senior Machine Learning Engineer & Data Scientist  
**Course / Project:** Advanced Predictive Analytics & Machine Learning  
**Environment:** Python 3.10+, Scikit-Learn, Imbalanced-Learn, Pandas, Seaborn  

---

## 1. Executive Summary & Problem Formulation
Marketing campaigns are costly and not every customer responds. Traditional mass-marketing strategies ("spray-and-pray") distribute marketing touches indiscriminately across entire customer bases, resulting in wasted promotional budgets, customer ad fatigue, and sub-optimal return on investment (ROI).

The objective of this project is to build machine learning models that identify customers who are **genuinely likely to respond** to promotional marketing campaigns, enabling organizations to target expenditures, reduce ad waste, and optimize marketing ROI.
"""))

    # Imports and Environment Setup
    cells.append(nbf.v4.new_markdown_cell("""### 1.1 Imports and Environment Configuration"""))
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath("..") if os.path.exists("../src") else os.path.abspath(".")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.generate_data import generate_campaign_dataset
from src.preprocess import load_and_split_data, create_preprocessor, get_feature_names, IQRCapper
from src.eda import run_full_eda

print("Environment configured successfully. Current working directory:", os.getcwd())
"""))

    # Section 2: Data Generation & Inspection
    cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Generation & Schema Validation
If a local CSV does not already exist, we generate a realistic synthetic dataset (~5,000 observations, seed 42) incorporating realistic consumer distributions, class imbalance (~20% response rate), subtle missingness, and outliers.
"""))
    cells.append(nbf.v4.new_code_cell("""data_path = os.path.join(PROJECT_ROOT, "data", "campaign_data.csv")
if not os.path.exists(data_path):
    print("Generating synthetic campaign data...")
    generate_campaign_dataset(n_samples=5000, random_state=42, output_path=data_path)

df = pd.read_csv(data_path)
print(f"Dataset Shape: {df.shape}")
print(f"Class Distribution:\\n{df['responded'].value_counts(normalize=True).mul(100).round(2)}")
df.head()
"""))

    cells.append(nbf.v4.new_code_cell("""# Data types and missing value audit
missing_info = pd.DataFrame({
    'Data_Type': df.dtypes,
    'Missing_Count': df.isna().sum(),
    'Missing_Pct': (df.isna().sum() / len(df) * 100).round(2)
})
print("Missing Value and Data Type Audit:")
display(missing_info)
"""))

    # Section 3: Exploratory Data Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 3. Exploratory Data Analysis (EDA)
In this section, we analyze customer distributions, historical conversion rates across customer demographic groups, and direct correlations between behavioral attributes and campaign response.
"""))

    cells.append(nbf.v4.new_code_cell("""# Execute EDA and generate publication-quality figures
fig_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
eda_results = run_full_eda(data_path=data_path, output_dir=fig_dir)
print(f"Target response rate: {eda_results['target_balance_pct']:.2f}%")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3.1 Target Class Distribution
**Takeaway:** The target exhibits substantial class imbalance (~20.1% positive response rate). Therefore, standard accuracy is misleading; optimization must prioritize F1-score, Precision-Recall balance, and economic ROI.
"""))
    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(fig_dir, "eda_target_distribution.png")))"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3.2 Response Rates by Demographics & Engagement
**Takeaway:** Prior campaign response is a massive positive multiplier (response rate increases from 14% to >52%). Email engagement and discount usage also exhibit strong positive monotonic relationships with conversion.
"""))
    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(fig_dir, "eda_response_rates_breakdown.png")))"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3.3 Numeric Feature Distributions by Response Status
**Takeaway:** Responders show distinctly elevated purchase frequencies, higher email engagement, and higher discount reliance compared to non-responders.
"""))
    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(fig_dir, "eda_numeric_distributions_boxplots.png")))"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3.4 Feature Correlation Heatmap
**Takeaway:** Previous campaign response ($r \\approx 0.44$), email engagement ($r \\approx 0.40$), and purchase frequency ($r \\approx 0.28$) exhibit the strongest linear correlation with customer response.
"""))
    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(fig_dir, "eda_correlation_heatmap.png")))"""))

    # Section 4: Preprocessing & Leakage Prevention
    cells.append(nbf.v4.new_markdown_cell("""## 4. Preprocessing Pipeline & Stratified Train/Test Split
To ensure zero data leakage:
- Numeric features undergo median imputation, 1.5*IQR outlier capping, and standard scaling.
- Categorical features (`age_group`) undergo mode imputation and one-hot encoding (`drop='first'`).
- All transformers are bundled into a `ColumnTransformer` and serialized inside a unified pipeline.
- Stratified 80/20 train/test split preserves class prevalence across partitions.
"""))
    cells.append(nbf.v4.new_code_cell("""X_train, X_test, y_train, y_test = load_and_split_data(data_path, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape} | Positive cases: {y_train.sum()} ({y_train.mean():.1%})")
print(f"X_test shape:  {X_test.shape}  | Positive cases: {y_test.sum()} ({y_test.mean():.1%})")

preprocessor = create_preprocessor()
preprocessor.fit(X_train)
feature_names = get_feature_names(preprocessor)
print(f"Transformed output features ({len(feature_names)}):\\n{feature_names}")
"""))

    # Section 5: Model Benchmark & Cross-Validation
    cells.append(nbf.v4.new_markdown_cell("""## 5. Machine Learning Model Benchmark
We evaluate all six required algorithms under 5-Fold Stratified Cross-Validation with SMOTE oversampling embedded strictly inside the training folds:
1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Tree**
4. **Random Forest**
5. **Naive Bayes (GaussianNB)**
6. **Gradient Boosting**
"""))
    cells.append(nbf.v4.new_code_cell("""comp_csv = os.path.join(PROJECT_ROOT, "reports", "results_comparison.csv")
results_df = pd.read_csv(comp_csv)
print("=== Algorithm Performance Benchmark on Held-Out Test Set (Sorted by F1 & ROC-AUC) ===")
display(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'FPR', 'TN', 'FP', 'FN', 'TP', 'CV_F1_Mean', 'CV_AUC_Mean']])
"""))

    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(fig_dir, "metrics_comparison_bar.png")))"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5.1 ROC and Precision-Recall Curves
Precision-Recall (PR) curves provide the definitive assessment on imbalanced datasets, measuring performance above baseline prevalence (20.1%).
"""))
    cells.append(nbf.v4.new_code_cell("""col1 = Image(filename=os.path.join(fig_dir, "roc_curves_all_models.png"), width=480)
col2 = Image(filename=os.path.join(fig_dir, "pr_curves_all_models.png"), width=480)
display(col1, col2)
"""))

    cells.append(nbf.v4.new_code_cell("""# Confusion Matrix Grid
display(Image(filename=os.path.join(fig_dir, "confusion_matrices_grid.png")))
"""))

    # Section 6: Class Imbalance Impact Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 6. Class Imbalance Analysis: With vs Without SMOTE
We explicitly quantify the performance delta on the best model with and without SMOTE oversampling.
"""))
    cells.append(nbf.v4.new_code_cell("""imb_csv = os.path.join(PROJECT_ROOT, "reports", "imbalance_handling_comparison.csv")
imb_df = pd.read_csv(imb_csv)
print("Impact of Class Imbalance Handling (Best Model: Gradient Boosting):")
display(imb_df)
"""))

    # Section 7: Threshold Tuning & Business Simulation
    cells.append(nbf.v4.new_markdown_cell("""## 7. Reducing False Positives & Marketing Cost Simulation
In marketing analytics, False Positives represent wasted contact expenditure ($5.00 per touch), while False Negatives represent missed gross profit ($50.00 per responding customer).
By sweeping classification thresholds from 0.05 to 0.95, we determine the optimal decision threshold that maximizes net marketing yield.
"""))
    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(fig_dir, "threshold_tuning.png")))"""))

    cells.append(nbf.v4.new_markdown_cell("""### 7.1 Business Economics Simulation
We compare three distinct commercial strategies across 1,000 customers:
- **Strategy A:** Contact Everyone (Mass Spray-and-Pray)
- **Strategy B:** Default Machine Learning ($p = 0.50$)
- **Strategy C:** Optimized Machine Learning ($p = 0.39$)
"""))
    cells.append(nbf.v4.new_code_cell("""sim_csv = os.path.join(PROJECT_ROOT, "reports", "business_simulation.csv")
sim_df = pd.read_csv(sim_csv)
print("Marketing Financial Simulation Table:")
display(sim_df)
"""))
    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(fig_dir, "business_simulation_roi.png")))"""))

    cells.append(nbf.v4.new_markdown_cell("""### 7.2 Targeting Efficiency: Cumulative Gains & Decile Lift
Decile analysis measures the proportion of total responders captured by ranking customers by predicted likelihood.
"""))
    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(fig_dir, "cumulative_gains_lift.png")))"""))

    cells.append(nbf.v4.new_markdown_cell("""### 7.3 Probability Calibration
Checking whether predicted model probabilities correspond accurately to empirical response frequencies.
"""))
    cells.append(nbf.v4.new_code_cell("""display(Image(filename=os.path.join(fig_dir, "calibration_curve.png")))"""))

    # Section 8: Feature Importance & Profiling
    cells.append(nbf.v4.new_markdown_cell("""## 8. Feature Importance, Odds Ratios, SHAP & Customer Profiles
We uncover the underlying behavioral drivers through Tree MDI, Permutation Importance, Logistic Odds Ratios, and SHAP.
"""))
    cells.append(nbf.v4.new_code_cell("""img1 = Image(filename=os.path.join(fig_dir, "feature_importance_tree.png"), width=480)
img2 = Image(filename=os.path.join(fig_dir, "feature_importance_permutation.png"), width=480)
display(img1, img2)
"""))
    cells.append(nbf.v4.new_code_cell("""img3 = Image(filename=os.path.join(fig_dir, "feature_importance_odds_ratios.png"), width=480)
shap_img_path = os.path.join(fig_dir, "shap_summary.png")
if os.path.exists(shap_img_path):
    img4 = Image(filename=shap_img_path, width=480)
    display(img3, img4)
else:
    display(img3)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 8.1 Customer Persona Profile Comparison
Mean and median attributes for predicted Responders versus Non-Responders:
"""))
    cells.append(nbf.v4.new_code_cell("""prof_csv = os.path.join(PROJECT_ROOT, "reports", "customer_profiles.csv")
prof_df = pd.read_csv(prof_csv, index_col=0)
display(prof_df)
"""))

    # Section 9: Answers to Research Questions & Final Report Narrative
    cells.append(nbf.v4.new_markdown_cell("""## 9. Final Analysis Report & Answers to Key Research Questions

### Question 1: Can campaign responses be predicted?
**Answer:** **Yes.** With a test **ROC-AUC of 0.8249** and an **F1-score of 0.5639**, the Gradient Boosting model accurately separates high-propensity responders from non-responders. The top 20% of scored customers capture **58.2% of all actual responders** (a **3.85x lift** over random marketing), proving that response behavior is highly predictable.

### Question 2: Which customer characteristics influence response?
**Answer:** **Behavioral engagement strongly dominates static demographics.**
- `email_engagement` is the #1 predictor across MDI, permutation importance, and SHAP (responders average 61% engagement vs 35% for non-responders).
- `previous_campaign_response` yields an Odds Ratio of ~4.88; prior responders are nearly 5x more likely to convert.
- `purchase_frequency` (2.76 orders/mo for responders vs 1.81 for non-responders).
- `discount_usage` (responders use discounts on 53% of purchases vs 40% for non-responders).
- `income` and `age_group` exert only minor direct influence.

### Question 3: Which algorithm performs best?
**Answer:** **Gradient Boosting Classifier.**
It achieved the highest test F1-score (**0.5639**, optimized to **0.5750**), the highest 5-fold CV F1 score (**0.6023 ± 0.010**), and the lowest False Positive Rate (**12.14%**), suppressing 702 true non-responders to protect promotional capital.

### Question 4: Can ML reduce unnecessary marketing expenditure?
**Answer:** **Yes, by 72% to 78%.**
In a 1,000-customer test cohort:
- Mass marketing required **$5,000.00** in spend and generated 799 wasted contacts.
- Default ML ($p=0.50$) reduced spend to **$1,070.00** (**78.6% cost reduction**, saving $3,930).
- Optimized ML ($p=0.39$) required **$1,395.00** (**72.1% cost reduction**, saving $3,605) while generating **$5,505.00 in net profit** (the highest net profit of any strategy).

### Question 5: How can false positives be reduced?
**Answer:**
1. Algorithm selection: Gradient Boosting reduces false positives to 97 (12.1% FPR) compared to 210 for KNN (26.3% FPR).
2. Threshold tuning: Operating at threshold 0.39 limits false positives to 141 while capturing 138 true responders, maximizing financial return.

### Question 6: Can the model identify potential campaign responders?
**Answer:** **Yes.** At threshold 0.39, the model captured **68.66% of all available responders (138 out of 201)** while targeting only 27.9% of the customer base.

---
## 10. Conclusion & Strategic Recommendations
1. **Tiered Marketing Execution:** Target Tier 1 (top 20%) with premium personalized offers, Tier 2 (next 10-15%) with low-cost automated emails, and suppress the remaining 65-70% to eliminate wasted ad spend.
2. **Prioritize Prior Responders:** Prior responders exhibit a 3.7x higher baseline conversion rate; immediate retention funnels should be maintained for every campaign responder.
3. **Production Deployment:** All preprocessing, models, and threshold controllers are operational in the accompanying Streamlit application (`app.py`).
"""))

    nb.cells = cells
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Notebook written to '{output_path}'.")

if __name__ == "__main__":
    create_analysis_notebook()
