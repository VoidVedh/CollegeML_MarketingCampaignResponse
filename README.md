# Marketing Campaign Response Prediction Using Machine Learning

**Author:** Vedh Naik  
**Roll No.:** 150096725163  
**Cohort:** Jensen Huang  
**Course / Project:** College Machine Learning Project — Case Study 157  

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.64.0-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/verification-passed-brightgreen.svg)]()

> **System Requirement:** Python >= 3.11. All dependencies in `requirements.txt` are exact pinned versions.

A complete, production-grade, verified machine learning system that predicts customer propensity to respond to promotional marketing campaigns. The project empowers marketing leaders to allocate advertising spend with statistical precision, suppress non-responsive contacts, reduce wasted expenditure by **46.6% to 76.6%**, and deliver **$6,480.00 in simulated net campaign profit** (242.7% ROI) under assumed campaign economics.

---

## 📋 Executive Overview & Key Results

| Metric / Objective | Traditional Mass Outreach | Default ML (Threshold 0.50) | F1-Optimal (Threshold 0.30) | Profit-Optimal (Threshold 0.07) [Deployed] |
| :--- | :---: | :---: | :---: | :---: |
| **Deployed Model** | None (Spray & Pray) | **Logistic Regression** | **Logistic Regression** | **Logistic Regression** |
| **Contacts Targeted (per 1,000 customers)** | 1000 | 127 | 234 | **534** |
| **Total Campaign Spend ($)** | $5,000.00 | $635.00 | $1,170.00 | **$2,670.00** |
| **Responders Reached** | 201 (100%) | 100 (49.8%) | 143 (71.1%) | **183 (91.0%)** |
| **Wasted Contacts (False Positives)** | 799 | 27 | 91 | 351 |
| **Gross Revenue ($)** | $10,050.00 | $5,000.00 | $7,150.00 | **$9,150.00** |
| **Net Campaign Profit ($)** | $5,050.00 | $4,365.00 | $5,980.00 | **$6,480.00 (Highest Profit)** |
| **Marketing ROI (%)** | 101.0% | 687.4% | 511.1% | **242.7%** |
| **Ad Spend Saved vs. Mass Outreach** | 0% ($0.00) | 87.3% ($4,365.00 saved) | 76.6% ($3,830.00 saved) | **46.6% ($2,330.00 saved)** |

---

## 🏆 Model Selection & Benchmark Results

### Defensible Selection Rule
The top two models on development cross-validation, Logistic Regression (CV PR-AUC: 0.7223 ± 0.0267) and Gradient Boosting (CV PR-AUC: 0.7169 ± 0.0254), are practically close, with a score difference (0.0054) smaller than the observed fold-to-fold cross-validation variation (0.0267). Based strictly on training data cross-validation and out-of-fold metrics, Logistic Regression was selected because Logistic Regression demonstrated lower training out-of-fold FPR at 70% recall (0.1115 vs 0.1165). Additionally, Logistic Regression offers lower architectural complexity, closed-form linear coefficients, and direct interpretability.

- **Selected Model:** Logistic Regression
- **Deployed Estimator:** Logistic Regression
- **5-Fold CV PR-AUC:** 0.7223 ± 0.0267
- **5-Fold CV ROC-AUC:** 0.8883 ± 0.0104
- **Test Set ROC-AUC:** 0.8759 (95% Bootstrap CI: [0.8473, 0.9029])
- **Test Set PR-AUC:** 0.7286 (95% Bootstrap CI: [0.6707, 0.7832])
- **Test FPR @ Recall=70%:** 0.1026
- **Top-20% Customer Capture Rate:** 65.7% of responders captured in top 20% of customer ranking
- **Top Response Predictor:** Previous campaign response yields an Odds Ratio of **5.4601** (prior responders have elevated response odds).

### Complete 6-Algorithm Comparative Benchmark
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | FPR | Confusion Matrix (TN / FP / FN / TP) | OOF Tuned Thresh | Tuned F1 | FPR @ Rec=70% | Top-20% Capture | CV PR-AUC (Mean ± Std) | CV ROC-AUC (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.872 | 0.7874 | 0.4975 | 0.6098 | 0.8759 | 0.7286 | 0.0338 | 772 / 27 / 101 / 100 | 0.30 | 0.6575 | 0.1026 | 65.7% | 0.7223 ± 0.0267 | 0.8883 ± 0.0104 |
| **Gradient Boosting** | 0.872 | 0.8017 | 0.4826 | 0.6025 | 0.8740 | 0.7135 | 0.0300 | 775 / 24 / 104 / 97 | 0.29 | 0.6540 | 0.1214 | 65.2% | 0.7169 ± 0.0254 | 0.8845 ± 0.0075 |
| **Random Forest** | 0.872 | 0.8230 | 0.4627 | 0.5924 | 0.8744 | 0.7105 | 0.0250 | 779 / 20 / 108 / 93 | 0.28 | 0.6339 | 0.1289 | 65.7% | 0.7143 ± 0.0249 | 0.8819 ± 0.0102 |
| **K-Nearest Neighbors** | 0.846 | 0.7901 | 0.3184 | 0.4539 | 0.8580 | 0.6527 | 0.0213 | 782 / 17 / 137 / 64 | 0.25 | 0.6199 | 0.1502 | 60.2% | 0.6887 ± 0.0291 | 0.8663 ± 0.0131 |
| **Naive Bayes** | 0.853 | 0.6484 | 0.5871 | 0.6162 | 0.8669 | 0.7007 | 0.0801 | 735 / 64 / 83 / 118 | 0.40 | 0.6087 | 0.1489 | 61.2% | 0.6816 ± 0.0346 | 0.8757 ± 0.0141 |
| **Decision Tree** | 0.847 | 0.7000 | 0.4179 | 0.5234 | 0.8353 | 0.6195 | 0.0451 | 763 / 36 / 117 / 84 | 0.27 | 0.5885 | 0.2003 | 58.7% | 0.6051 ± 0.0276 | 0.8382 ± 0.0113 |

---

## ⚖️ Class Imbalance Treatments Comparison
| Imbalance Treatment | 5-Fold CV PR-AUC (Mean ± Std) | 5-Fold CV ROC-AUC (Mean ± Std) | Test PR-AUC | Test ROC-AUC | Test F1 (at 0.50) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **None (Unweighted Baseline)** | 0.7223 ± 0.0267 | 0.8883 ± 0.0104 | 0.7286 | 0.8759 | 0.6098 |
| **class_weight='balanced'** | 0.7204 ± 0.0275 | 0.8878 ± 0.0107 | 0.7297 | 0.8769 | 0.6090 |
| **SMOTENC (Categorical-aware Oversampling)** | 0.7153 ± 0.0188 | 0.8843 ± 0.0108 | 0.7250 | 0.8777 | 0.6142 |

*Imbalance Handling Finding:* Reweighting and SMOTENC do not improve ranking discrimination (PR-AUC remains virtually identical across treatments). Rather than distorting predicted response probabilities, decision threshold optimization on the unweighted model achieves superior, cost-effective targeting.

---

## 🏗️ Project Architecture

```
CollegeML_MarketingCampaignResponse/
├── data/
│   └── campaign_data.csv            # 5,000 customer records with realistic distributions
├── notebooks/
│   └── analysis.ipynb               # Fully executed Jupyter notebook with live tables
├── src/
│   ├── generate_data.py             # Realistic synthetic data generator
│   ├── preprocess.py                # Leakage-free ColumnTransformer & IQRCapper
│   ├── eda.py                       # Automated EDA and visualization suite
│   ├── train.py                     # Main orchestrator (CV, OOF thresholds, FrozenEstimator)
│   ├── evaluate.py                  # Evaluation suite, bootstrap CIs, 4-strategy simulation
│   ├── build_report.py              # Dynamic markdown generator sourced from metrics.json
│   └── build_notebook.py            # Automated notebook compilation script
├── models/
│   ├── best_model.joblib            # Serialized best model pipeline
│   ├── preprocessor.joblib          # Standalone fitted ColumnTransformer
│   ├── metrics.json                 # Complete performance metrics and simulation numbers
│   └── feature_list.json            # Feature schema metadata
├── reports/
│   ├── figures/                     # Publication-quality charts (PNG)
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
pytest tests/ -v
python tests/final_verification.py
python tests/verify_consistency.py
```
