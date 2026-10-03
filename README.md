# Marketing Campaign Response Prediction Using Machine Learning

**Case Study:** Case Study 157  
**Student:** Vedh Naik  
**Roll No.:** 150096725163  
**Cohort:** Jensen Huang  
**Repository:** `VoidVedh/CollegeML_MarketingCampaignResponse`  
**Dataset:** [Kaggle Marketing Dataset (Bank Marketing)](https://www.kaggle.com/competitions/marketing-dataset/data)  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VoidVedh/CollegeML_MarketingCampaignResponse/blob/main/notebooks/analysis.ipynb)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.64.0-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/verification-passed-brightgreen.svg)]()

> **Dataset Notice:** This project uses the real, public **Kaggle Marketing Dataset** (41,188 client interactions, 21 columns). To prevent target leakage, call duration (`duration`) is **strictly excluded** from all predictive modeling.

A complete, production-grade machine learning system that predicts client propensity to subscribe to bank term deposits (`y = yes/no`). The system optimizes marketing budgets, reduces wasted expenditure by **81.1% to 85.8%**, and delivers **$22,435.00 in simulated net campaign profit** (288.9% ROI) under assumed campaign economics.

---

## 📋 Executive Overview & Key Results

| Metric / Objective | Traditional Mass Outreach | Default ML (Threshold 0.50) | F1-Optimal (Threshold 0.22) | Profit-Optimal (Threshold 0.11) [Deployed] |
| :--- | :---: | :---: | :---: | :---: |
| **Deployed Model** | None (Spray & Pray) | **Random Forest** | **Random Forest** | **Random Forest** |
| **Contacts Targeted (Test Set: 8,238 clients)** | 8238 | 319 | 1167 | **1553** |
| **Total Campaign Spend ($)** | $41,190.00 | $1,595.00 | $5,835.00 | **$7,765.00** |
| **Subscribers Reached** | 928 (100%) | 220 (23.7%) | 554 (59.7%) | **604 (65.1%)** |
| **Wasted Contacts (False Positives)** | 7310 | 99 | 613 | 949 |
| **Gross Revenue ($)** | $46,400.00 | $11,000.00 | $27,700.00 | **$30,200.00** |
| **Net Campaign Profit ($)** | $5,210.00 | $9,405.00 | $21,865.00 | **$22,435.00 (Highest Profit)** |
| **Marketing ROI (%)** | 12.6% | 589.7% | 374.7% | **288.9%** |
| **Ad Spend Saved vs. Mass Outreach** | 0% ($0.00) | 96.1% ($39,595.00 saved) | 85.8% ($35,355.00 saved) | **81.1% ($33,425.00 saved)** |

---

## 🏆 Model Selection & Benchmark Results

### Defensible Selection Rule
The top two models on development cross-validation, Random Forest (CV PR-AUC: 0.4646 ± 0.0110) and Gradient Boosting (CV PR-AUC: 0.4626 ± 0.0112), are practically close, with a score difference (0.0020) smaller than the observed fold-to-fold cross-validation variation (0.0110). Based strictly on training data cross-validation and out-of-fold metrics, Random Forest was selected because Random Forest demonstrated lower training out-of-fold FPR at 70% recall (0.2316 vs 0.2438). Additionally, Random Forest offers lower architectural complexity, closed-form linear coefficients, and direct interpretability.

- **Selected Model:** Random Forest
- **Deployed Estimator:** Random Forest
- **5-Fold CV PR-AUC:** 0.4646 ± 0.0110
- **5-Fold CV ROC-AUC:** 0.7949 ± 0.0028
- **Test Set ROC-AUC:** 0.8096 (95% Bootstrap CI: [0.7916, 0.8253])
- **Test Set PR-AUC:** 0.4871 (95% Bootstrap CI: [0.4533, 0.5226])
- **Test FPR @ Recall=70%:** 0.1845
- **Top-20% Customer Capture Rate:** 66.4% of subscribers captured in top 20% of customer ranking
- **Top Driver Odds Ratio:** `month_mar` yields an Odds Ratio of **4.0319**.

### Complete 6-Algorithm Comparative Benchmark
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | FPR | Confusion Matrix (TN / FP / FN / TP) | OOF Tuned Thresh | Tuned F1 | FPR @ Rec=70% | Top-20% Capture | CV PR-AUC (Mean ± Std) | CV ROC-AUC (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.902 | 0.6897 | 0.2371 | 0.3528 | 0.8096 | 0.4871 | 0.0135 | 7211 / 99 / 708 / 220 | 0.22 | 0.5289 | 0.1845 | 66.4% | 0.4646 ± 0.0110 | 0.7949 ± 0.0028 |
| **Gradient Boosting** | 0.902 | 0.6855 | 0.2349 | 0.3499 | 0.8093 | 0.4840 | 0.0137 | 7210 / 100 / 710 / 218 | 0.22 | 0.5262 | 0.1925 | 65.7% | 0.4626 ± 0.0112 | 0.7936 ± 0.0050 |
| **Logistic Regression** | 0.902 | 0.6944 | 0.2252 | 0.3401 | 0.8017 | 0.4653 | 0.0126 | 7218 / 92 / 719 / 209 | 0.20 | 0.5160 | 0.1966 | 65.8% | 0.4487 ± 0.0109 | 0.7879 ± 0.0072 |
| **Decision Tree** | 0.904 | 0.6936 | 0.2586 | 0.3768 | 0.7919 | 0.4427 | 0.0145 | 7204 / 106 / 688 / 240 | 0.19 | 0.5176 | 0.6750 | 63.5% | 0.4105 ± 0.0088 | 0.7778 ± 0.0023 |
| **K-Nearest Neighbors** | 0.900 | 0.6537 | 0.2360 | 0.3468 | 0.7764 | 0.4274 | 0.0159 | 7194 / 116 / 709 / 219 | 0.24 | 0.5057 | 0.3978 | 62.5% | 0.4096 ± 0.0052 | 0.7626 ± 0.0058 |
| **Naive Bayes** | 0.860 | 0.3967 | 0.4612 | 0.4265 | 0.7748 | 0.3623 | 0.0891 | 6659 / 651 / 500 / 428 | 0.89 | 0.4313 | 0.2334 | 57.3% | 0.3471 ± 0.0095 | 0.7614 ± 0.0037 |

---

## ⚖️ Class Imbalance Treatments Comparison
| Imbalance Treatment | 5-Fold CV PR-AUC (Mean ± Std) | 5-Fold CV ROC-AUC (Mean ± Std) | Test PR-AUC | Test ROC-AUC | Test F1 (at 0.50) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **None (Unweighted Baseline)** | 0.4646 ± 0.0110 | 0.7949 ± 0.0028 | 0.4871 | 0.8096 | 0.3528 |
| **class_weight='balanced'** | 0.4568 ± 0.0087 | 0.7943 ± 0.0064 | 0.4863 | 0.8130 | 0.5142 |
| **SMOTENC (Categorical-aware Oversampling)** | 0.4227 ± 0.0014 | 0.7811 ± 0.0074 | 0.4570 | 0.8040 | 0.5289 |

---

## 📊 Feature Mapping: Case Study 157 Concepts vs Kaggle Fields

| Case Study Concept | Kaggle Equivalent Field(s) | Status | Technical Justification |
| :--- | :--- | :---: | :--- |
| **Age group** | Derived from `age` (`18-25`, `26-35`, `36-45`, `46-55`, `56+`) | **Can map** | Binned directly from continuous `age` into demographic brackets. |
| **Income** | *No direct field* | **Not available** | Not collected in the UCI/Kaggle Bank Marketing study. Not fabricated. |
| **Previous purchases** | *No direct equivalent* | **Not available** | Bank deposits are service subscriptions, not recurring e-commerce cart items. |
| **Purchase frequency** | *No direct equivalent* | **Not available** | Transaction cadence is not recorded in campaign contact logs. |
| **Previous campaign response** | `poutcome`, `previous`, `pdays` | **Partial / defensible** | `poutcome` records prior campaign outcome, `previous` counts prior touches, `pdays` tracks elapsed days. |
| **Website visits** | *No direct field* | **Not available** | Direct phone campaign data; no web telemetry available. |
| **Email engagement** | *No direct field* | **Not available** | Direct telephone channel; no email interaction logs available. |
| **Discount usage** | *No direct field* | **Not available** | Term deposits provide savings yields rather than retail promotional discounts. |

---

## 🏗️ Project Architecture

```
CollegeML_MarketingCampaignResponse/
├── data/
│   └── kaggle/                      # Primary dataset source of truth
│       ├── train.csv                # 41,188 labelled client records (y = yes/no)
│       ├── test.csv                 # Unlabelled competition evaluation cohort
│       └── sampleSubmission.csv     # Competition submission format
├── notebooks/
│   └── analysis.ipynb               # Fully executed Jupyter notebook with Colab badge
├── src/
│   ├── preprocess.py                # Leakage-free ColumnTransformer & IQRCapper (zero duration)
│   ├── eda.py                       # Automated Kaggle EDA and visualization suite
│   ├── train.py                     # Main orchestrator (CV, OOF thresholds, FrozenEstimator)
│   ├── evaluate.py                  # Evaluation suite, bootstrap CIs, 4-strategy simulation
│   ├── build_report.py              # Dynamic markdown generator sourced from metrics.json
│   ├── build_notebook.py            # Automated notebook compilation script
│   └── generate_data.py             # [DEPRECATED / LEGACY] Retained for historical reference
├── models/
│   ├── best_model.joblib            # Serialized Kaggle-trained model pipeline
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

## ⚙️ Quickstart & Execution

### 1. Environment Setup (Python >= 3.11)
```bash
python3 -m venv venv
source venv/bin/activate
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
