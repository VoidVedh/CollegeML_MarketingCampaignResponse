# Marketing Campaign Response Prediction Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.64.0-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/project-submission--ready-brightgreen.svg)]()

A complete, production-grade, college-submission-ready machine learning system that predicts customer propensity to respond to promotional marketing campaigns. The project empowers marketing leaders to allocate advertising spend with statistical precision, suppress non-responsive contacts, slash wasted expenditure by **>72%**, and nearly **quadruple campaign ROI to 394.6%**.

---

## 📋 Executive Overview & Key Results

| Metric / Objective | Traditional Mass Outreach | Default ML (Threshold 0.50) | Optimized ML (Threshold 0.39) |
| :--- | :---: | :---: | :---: |
| **Deployed Algorithm** | None (Spray & Pray) | **Gradient Boosting** | **Gradient Boosting** |
| **Contacts Targeted (per 1,000 customers)** | 1,000 | 214 | 279 |
| **Total Campaign Spend ($)** | $5,000.00 | $1,070.00 | $1,395.00 |
| **Responders Reached** | 201 (100%) | 117 (58.2%) | 138 (68.7%) |
| **Wasted Contacts (False Positives)** | 799 (79.9% waste) | 97 | 141 |
| **Gross Revenue ($)** | $10,050.00 | $5,850.00 | $6,900.00 |
| **Net Campaign Profit ($)** | $5,050.00 | $4,780.00 | **$5,505.00 (Highest Profit)** |
| **Marketing ROI (%)** | 101.0% | **446.7%** | **394.6%** |
| **Ad Spend Saved vs. Mass Outreach** | 0% ($0.00) | **78.6% ($3,930 saved)** | **72.1% ($3,605 saved)** |

---

## 🏗️ Project Architecture & Directory Structure

```
marketing-campaign-response/
├── data/
│   └── campaign_data.csv            # 5,000 customer records with realistic distributions
├── notebooks/
│   └── analysis.ipynb               # Fully executed end-to-end narrative Jupyter notebook
├── src/
│   ├── generate_data.py             # Realistic synthetic data generator
│   ├── preprocess.py                # Leakage-free ColumnTransformer & IQRCapper pipeline
│   ├── eda.py                       # Automated EDA and visualization generator
│   ├── train.py                     # Main orchestrator (CV tuning, simulations, serialization)
│   ├── evaluate.py                  # Benchmark suite, threshold tuning, ROI simulations
│   └── build_notebook.py            # Automated notebook compilation script
├── models/
│   ├── best_model.joblib            # Serialized best model pipeline (Gradient Boosting)
│   ├── preprocessor.joblib          # Standalone fitted ColumnTransformer
│   ├── metrics.json                 # Complete performance metrics and simulation numbers
│   └── feature_list.json            # Exact schema column metadata
├── reports/
│   ├── figures/                     # 16 high-resolution publication PNG charts
│   │   ├── eda_target_distribution.png
│   │   ├── eda_response_rates_breakdown.png
│   │   ├── eda_numeric_distributions_boxplots.png
│   │   ├── eda_correlation_heatmap.png
│   │   ├── metrics_comparison_bar.png
│   │   ├── roc_curves_all_models.png
│   │   ├── pr_curves_all_models.png
│   │   ├── confusion_matrices_grid.png
│   │   ├── threshold_tuning.png
│   │   ├── business_simulation_roi.png
│   │   ├── cumulative_gains_lift.png
│   │   ├── calibration_curve.png
│   │   ├── feature_importance_tree.png
│   │   ├── feature_importance_permutation.png
│   │   ├── feature_importance_odds_ratios.png
│   │   └── shap_summary.png
│   ├── results_comparison.csv       # Benchmark table across all 6 models
│   ├── imbalance_handling_comparison.csv # With vs without SMOTE comparison
│   ├── business_simulation.csv      # Financial impact comparison table
│   ├── customer_profiles.csv        # Responder vs Non-Responder average attributes
│   └── final_report.md              # Full academic & management report
├── app.py                           # Interactive Streamlit Web Application
├── requirements.txt                 # Pinned dependencies
└── README.md                        # Documentation and replication guide
```

---

## ⚙️ Quickstart & Execution

### 1. Prerequisites & Environment Setup
Clone or navigate to the project directory:
```bash
cd /Users/ved/.gemini/antigravity-ide/scratch/marketing-campaign-response
```

Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. One-Command Complete Execution
To generate data (if absent), run full EDA, train and tune all 6 algorithms via 5-fold cross-validation, run threshold tuning, execute business simulations, and export all plots and model artifacts:
```bash
python src/train.py
```

### 3. Launching the Interactive Streamlit Web App
Launch the interactive web application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

#### Web Application Features:
- **Tab 1: Single Customer Scoring:** Adjust all 8 behavioral inputs and view instant real-time predictions, probability progress gauges, and actionable marketing recommendations.
- **Tab 2: Algorithm Benchmark & Comparison:** View interactive performance metrics tables, ROC curves, PR curves, confusion matrices, and the business simulation.
- **Tab 3: Feature Importance & Insights:** Explore Tree MDI, Permutation Importance, Logistic Odds Ratios, SHAP summary plots, and persona profiles.
- **Tab 4: Batch CSV Scoring:** Upload any customer CSV, validate schema, generate predictions, and export scored CSV files with calculated response probabilities.

---

## 🔬 Benchmark Comparison Across All 6 Algorithms

All models were evaluated under identical 5-fold Stratified Cross-Validation with SMOTE oversampling embedded strictly in the training folds. Metrics on the 1,000-customer held-out test set:

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | FPR | 5-Fold CV F1 | 5-Fold CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting (Best)** | **0.8190** | **0.5467** | 0.5821 | **0.5639** | 0.8249 | **0.1214** | **0.6023 ± 0.010** | 0.8468 ± 0.006 |
| **Naive Bayes** | 0.7630 | 0.4474 | 0.7612 | 0.5635 | 0.8364 | 0.2365 | 0.5795 ± 0.009 | 0.8564 ± 0.005 |
| **Random Forest** | 0.7860 | 0.4775 | 0.6866 | 0.5633 | 0.8268 | 0.1890 | 0.5987 ± 0.008 | 0.8517 ± 0.004 |
| **Logistic Regression** | 0.7650 | 0.4494 | 0.7512 | 0.5624 | **0.8389** | 0.2315 | 0.5849 ± 0.007 | **0.8620 ± 0.003** |
| **Decision Tree** | 0.7560 | 0.4384 | 0.7612 | 0.5564 | 0.8061 | 0.2453 | 0.5666 ± 0.015 | 0.8282 ± 0.003 |
| **K-Nearest Neighbors** | 0.7470 | 0.4293 | **0.7861** | 0.5554 | 0.8010 | 0.2628 | 0.5503 ± 0.017 | 0.8150 ± 0.012 |

### Why Accuracy is Deceptive
With a 20.14% baseline response rate, a naive dummy model predicting "No Response" achieves ~80% accuracy while failing to capture a single customer. Gradient Boosting is chosen because it achieves the highest F1-score (0.5639) and lowest False Positive Rate (12.14%), preventing wasted marketing spend.

---

## 💡 Top 5 Campaign Response Drivers
1. **Email Engagement Rate (`email_engagement`):** The single strongest predictor across Tree MDI, Permutation Importance, and SHAP. Responders average 61% engagement vs. 35% for non-responders (+74% higher).
2. **Prior Campaign Response (`previous_campaign_response`):** Exponentiated Odds Ratio of **4.88**; customers who accepted prior campaigns are nearly 5x more likely to convert again.
3. **Purchase Frequency (`purchase_frequency`):** Responders order 2.76 times per month vs. 1.81 times for non-responders (+52% higher).
4. **Discount Usage (`discount_usage`):** Responders utilize promotional coupons on 53% of transactions vs. 40% for non-responders.
5. **Lifetime Transaction Count (`previous_purchases`):** Reflects baseline loyalty and ongoing relationship with the brand.

---

## 📌 Documented Assumptions
1. **Synthetic Nature of Data:** The dataset was synthetically generated with realistic probability distributions and noise for demonstration and research purposes.
2. **Economic Simulation Constants:**
   - Marginal cost per contact attempt = **$5.00** (digital ad spend, SMS, mailer).
   - Gross profit per converted responder = **$50.00** (average customer margin).
   - Incurred costs and profits are configurable in `src/evaluate.py`.
3. **Non-Causal Assumption:** The model estimates promotional response propensity under contact, not causal uplift (i.e. it does not isolate customers who would have purchased organically without marketing).

---

## 🎓 College Submission Checklist
- [x] Dataset ready and full EDA executed with 4 saved figures
- [x] All 6 classification algorithms trained and tuned with 5-fold CV
- [x] Accuracy, Precision, Recall, F1, ROC-AUC, FPR, and full Confusion Matrices reported
- [x] ROC curves, PR curves, and comparison charts saved in `reports/figures/`
- [x] Best model chosen with rigorous F1/ROC-AUC justification (not accuracy alone)
- [x] Threshold tuning ($p^* = 0.39$) and financial ROI simulation completed
- [x] Tree importance, permutation importance, odds ratios, SHAP, and customer profiles generated
- [x] Interactive Streamlit app operational with all 8 inputs, tabs, and batch processing
- [x] `reports/final_report.md` answers all 6 research questions backed by exact numbers
- [x] Narrative Jupyter notebook `notebooks/analysis.ipynb` fully executed top-to-bottom
- [x] `README.md` and `requirements.txt` complete and verified
