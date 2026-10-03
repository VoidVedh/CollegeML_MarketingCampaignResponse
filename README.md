# Marketing Campaign Response Prediction Using Machine Learning

**Case Study:** Case Study 157  
**Student:** Vedh Naik  
**Roll No.:** 150096725163  
**Cohort:** Jensen Huang  
**Repository:** `VoidVedh/CollegeML_MarketingCampaignResponse`  
**Dataset:** [Kaggle Customer Personality Analysis (`marketing_campaign.csv`)](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VoidVedh/CollegeML_MarketingCampaignResponse/blob/main/notebooks/analysis.ipynb)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.64.0-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/verification-passed-brightgreen.svg)]()

> **Dataset Notice:** This project uses the authentic, public **Kaggle Customer Personality Analysis** dataset (2,240 customer records). The eight deployment variables of Case Study 157 are engineered directly from real customer demographics, purchasing behavior, and campaign history without data fabrication or target leakage.

A complete, production-grade machine learning system that predicts customer propensity to respond to marketing campaigns (`target = 0/1`). The system optimizes marketing budgets, reduces wasted expenditure by **69.0% to 84.8%**, and delivers **$1,705.00 in net campaign profit** (245.3% ROI) under assumed campaign economics.

---

## Executive Overview & Key Results

| Metric / Objective | Traditional Mass Outreach | Default ML (Threshold 0.50) | F1-Optimal (Threshold 0.27) | Profit-Optimal (Threshold 0.13) [Deployed] |
| :--- | :---: | :---: | :---: | :---: |
| **Deployed Model** | None (Spray & Pray) | **Random Forest** | **Random Forest** | **Random Forest** |
| **Contacts Targeted (Test Set: 448 customers)** | 448 | 25 | 68 | **139** |
| **Total Campaign Spend ($)** | $2,240.00 | $125.00 | $340.00 | **$695.00** |
| **Responders Reached** | 67 (100%) | 18 (26.9%) | 38 (56.7%) | **48 (71.6%)** |
| **Wasted Contacts (False Positives)** | 381 | 7 | 30 | 91 |
| **Gross Revenue ($)** | $3,350.00 | $900.00 | $1,900.00 | **$2,400.00** |
| **Net Campaign Profit ($)** | $1,110.00 | $775.00 | $1,560.00 | **$1,705.00 (Highest Profit)** |
| **Marketing ROI (%)** | 49.6% | 620.0% | 458.8% | **245.3%** |
| **Ad Spend Saved vs. Mass Outreach** | 0% ($0.00) | 94.4% ($2,115.00 saved) | 84.8% ($1,900.00 saved) | **69.0% ($1,545.00 saved)** |

---

## Model Selection & Benchmark Results

### Defensible Selection Rule
The top two models on development cross-validation, Random Forest (CV PR-AUC: 0.5109 ± 0.0450) and Logistic Regression (CV PR-AUC: 0.5069 ± 0.0572), are practically close, with a score difference (0.0040) smaller than the observed fold-to-fold cross-validation variation (0.0450). Based strictly on training data cross-validation and out-of-fold metrics, Random Forest was selected because Random Forest demonstrated lower training out-of-fold FPR at 70% recall (0.1921 vs 0.2492). Additionally, Random Forest captures non-linear feature interactions and provides robust tree-based and permutation feature importances without parametric distribution assumptions.

- **Selected Model:** Random Forest
- **Deployed Estimator:** Random Forest
- **5-Fold CV PR-AUC:** 0.5109 ± 0.0450
- **5-Fold CV ROC-AUC:** 0.8182 ± 0.0288
- **Test Set ROC-AUC:** 0.8171 (95% Bootstrap CI: [0.7535, 0.8738])
- **Test Set PR-AUC:** 0.5706 (95% Bootstrap CI: [0.4489, 0.6860])
- **Test FPR @ Recall=70%:** 0.1890
- **Top-20% Customer Capture Rate:** 61.2% of responders captured in top 20% of customer ranking
- **Top Driver Odds Ratio:** `email_engagement` yields an Odds Ratio of **1.9899**.

### Complete 6-Algorithm Comparative Benchmark
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | FPR | Confusion Matrix (TN / FP / FN / TP) | OOF Tuned Thresh | Tuned F1 | FPR @ Rec=70% | Top-20% Capture | CV PR-AUC (Mean ± Std) | CV ROC-AUC (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.875 | 0.7200 | 0.2687 | 0.3913 | 0.8171 | 0.5706 | 0.0184 | 374 / 7 / 49 / 18 | 0.27 | 0.5630 | 0.1890 | 61.2% | 0.5109 ± 0.0450 | 0.8182 ± 0.0288 |
| **Logistic Regression** | 0.873 | 0.7500 | 0.2239 | 0.3448 | 0.7808 | 0.5053 | 0.0131 | 376 / 5 / 52 / 15 | 0.18 | 0.4667 | 0.2467 | 55.2% | 0.5069 ± 0.0572 | 0.8040 ± 0.0302 |
| **Gradient Boosting** | 0.873 | 0.7273 | 0.2388 | 0.3596 | 0.8175 | 0.5167 | 0.0157 | 375 / 6 / 51 / 16 | 0.22 | 0.5161 | 0.2073 | 59.7% | 0.5002 ± 0.0551 | 0.8052 ± 0.0376 |
| **K-Nearest Neighbors** | 0.884 | 0.7586 | 0.3284 | 0.4583 | 0.7843 | 0.5613 | 0.0184 | 374 / 7 / 45 / 22 | 0.28 | 0.5286 | 0.2152 | 58.2% | 0.4913 ± 0.0521 | 0.7768 ± 0.0506 |
| **Naive Bayes** | 0.810 | 0.4022 | 0.5522 | 0.4654 | 0.7495 | 0.4809 | 0.1444 | 326 / 55 / 30 / 37 | 0.13 | 0.4512 | 0.3570 | 55.2% | 0.4605 ± 0.0476 | 0.7445 ± 0.0468 |
| **Decision Tree** | 0.855 | 0.5625 | 0.1343 | 0.2169 | 0.7283 | 0.3552 | 0.0184 | 374 / 7 / 58 / 9 | 0.35 | 0.5034 | 0.9974 | 55.2% | 0.4013 ± 0.0458 | 0.7509 ± 0.0341 |

---

## Class Imbalance Treatments Comparison
| Imbalance Treatment | 5-Fold CV PR-AUC (Mean ± Std) | 5-Fold CV ROC-AUC (Mean ± Std) | Test PR-AUC | Test ROC-AUC | Test F1 (at 0.50) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **None (Unweighted Baseline)** | 0.5109 ± 0.0450 | 0.8182 ± 0.0288 | 0.5706 | 0.8171 | 0.3913 |
| **class_weight='balanced'** | 0.5028 ± 0.0373 | 0.8076 ± 0.0380 | 0.5334 | 0.8163 | 0.5333 |
| **SMOTENC (Categorical-aware Oversampling)** | 0.4699 ± 0.0296 | 0.7949 ± 0.0342 | 0.5204 | 0.8110 | 0.5175 |

---

## The Eight Case Study 157 Deployment Features

| Case Study Feature | Kaggle Derivation Formula | Range / Domain | Technical Description |
| :--- | :--- | :---: | :--- |
| **Age group** | `pd.cut(2014 - Year_Birth, bins=[-inf, 25, 35, 45, 55, inf])` | `18-25`, `26-35`, `36-45`, `46-55`, `56+` | Binned from birth year using 2014 observation reference. |
| **Income** | `Income` | `$1,730 – $666,666` | Annual household income; median-imputed in pipeline. |
| **Previous purchases** | `NumWebPurchases + NumCatalogPurchases + NumStorePurchases` | `0 – 32` | Sum of actual historical channel purchases. |
| **Purchase frequency** | `previous_purchases / ((2014-12-31 - Dt_Customer).days / 30.4375)` | `0.0 – 3.91` | Monthly purchase frequency across customer tenure. |
| **Previous campaign response** | `(AcceptedCmp1 + ... + AcceptedCmp5 >= 1).astype(int)` | `0` or `1` | Binary flag indicating prior campaign acceptance (no leakage). |
| **Website visits** | `NumWebVisitsMonth` | `0 – 20` | Number of monthly visits to company website. |
| **Email engagement** | `(AcceptedCmp1 + ... + AcceptedCmp5) / 5.0` | `0.00 – 1.00` | Disclosed proxy for historical promotional engagement. |
| **Discount usage** | `NumDealsPurchases / max(previous_purchases, 1)` | `0.00 – 1.00` | Proportion of purchases made using promotional deals. |

---

## Project Architecture

```
CollegeML_MarketingCampaignResponse/
├── data/
│   ├── kaggle/
│   │   └── marketing_campaign.csv   # Authentic Kaggle source dataset (2,240 rows)
│   └── processed/
│       └── campaign_response_features.csv # Processed 8 features + target dataset
├── notebooks/
│   └── analysis.ipynb               # Fully executed Jupyter notebook with Colab badge
├── src/
│   ├── preprocess.py                # Feature engineering & ColumnTransformer pipeline
│   ├── eda.py                       # Automated EDA and visualization suite
│   ├── train.py                     # Main orchestrator (5-fold CV, OOF tuning, evaluation)
│   ├── evaluate.py                  # Evaluation suite, bootstrap CIs, 4-strategy simulation
│   └── build_report.py              # Dynamic markdown generator sourced from metrics.json
├── models/
│   ├── best_model.joblib            # Serialized winning model pipeline
│   ├── preprocessor.joblib          # Standalone fitted ColumnTransformer
│   ├── metrics.json                 # Complete performance metrics and simulation numbers
│   └── feature_list.json            # Feature schema metadata
├── reports/
│   ├── figures/                     # High-resolution evaluation charts (PNG)
│   ├── results_comparison.csv       # Benchmark table across all 6 models
│   ├── imbalance_handling_comparison.csv # None vs balanced vs SMOTENC comparison
│   ├── business_simulation.csv      # 4-strategy economic evaluation table
│   ├── customer_profiles.csv        # Actual vs predicted responder averages & medians
│   └── final_report.md              # Full academic & technical report
├── tests/
│   ├── test_app.py                  # Streamlit AppTest suite (load, predict, batch)
│   ├── verify_consistency.py        # Automated consistency verification script
│   └── final_verification.py        # Comprehensive verification audit
├── app.py                           # Interactive Streamlit Web Application
├── requirements.txt                 # Exact pinned dependencies
└── README.md                        # Documentation and replication guide
```

---

## Quickstart & Execution

### 1. Environment Setup (Python >= 3.11)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (End-to-End)
```bash
python src/train.py
```

### 3. Launch Streamlit Web Application
```bash
streamlit run app.py
```

### 4. Run Automated Verification Tests
```bash
python tests/final_verification.py
python tests/verify_consistency.py
pytest tests/test_app.py -v
```
