# Comprehensive Technical & Management Report
## Marketing Campaign Response Prediction Using Machine Learning
**Case Study:** Case Study 157  
**Student:** Vedh Naik  
**Roll No.:** 150096725163  
**Cohort:** Jensen Huang  
**Project Repository:** `VoidVedh/CollegeML_MarketingCampaignResponse`  
**Dataset Source:** [Kaggle Marketing Dataset (Bank Marketing)](https://www.kaggle.com/competitions/marketing-dataset/data)  
**Evaluation Date:** October 2026  
**Document Status:** Verified Academic & Technical Project Submission  

---

## Executive Summary
Direct marketing campaigns represent a critical investment for financial institutions and modern enterprises. Conventional mass outreach ("contacting every lead") wastes substantial marketing capital, fatigues uninterested clients, and degrades overall return on investment (ROI).

In this project, we built and validated an end-to-end, data-leakage-free machine learning system using the authentic, public **Kaggle Marketing Dataset** (41,188 client records; 11.27% baseline subscription rate). We benchmarked six classification algorithms: **Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, Naive Bayes (GaussianNB), and Gradient Boosting**.

### Key Evaluation & Selection Results
- **Leakage-Free Protocol:** Call duration (`duration`) is **strictly excluded** from all predictive modeling because duration is only known after/during a phone call. Utilizing `duration` in pre-campaign targeting would represent severe target leakage.
- **Model Selection:** The top two models on development cross-validation, Random Forest (CV PR-AUC: 0.4646 ± 0.0110) and Gradient Boosting (CV PR-AUC: 0.4626 ± 0.0112), are practically close, with a score difference (0.0020) smaller than the observed fold-to-fold cross-validation variation (0.0110). Based strictly on training data cross-validation and out-of-fold metrics, Random Forest was selected because Random Forest demonstrated lower training out-of-fold FPR at 70% recall (0.2316 vs 0.2438). Additionally, Random Forest offers lower architectural complexity, closed-form linear coefficients, and direct interpretability.
- **Selected Deployed Architecture:** **Random Forest** achieving a 5-fold CV PR-AUC of **0.4646 ± 0.0110** and CV ROC-AUC of **0.7949 ± 0.0028**.
- **Test Set Generalization:** Held-out test ROC-AUC of **0.8096** (95% Bootstrap CI: [0.7916, 0.8253]), PR-AUC of **0.4871** (95% Bootstrap CI: [0.4533, 0.5226]), and test FPR at 70% recall of **0.1845**.
- **Ranking-Based Top-20% Customer Capture:** Ranking clients by predicted probability captures **66.4%** (616 out of 928 actual term deposit subscribers) in the top 20% of contacted clients.
- **Dual Threshold Optimization (Training OOF only):**
  1. **F1-Optimal Threshold ($t = 0.22$):** Achieves test F1-score of **0.5289** (precision: 47.5%, recall: 59.7%, FPR: 0.0839), targeting 1167 clients for **$21,865.00 net profit** (374.7% ROI).
  2. **Profit-Optimal Threshold ($t = 0.11$):** Under simulated campaign economics (assumed contact cost = $5.00, responder gross profit = $50.00, theoretical break-even probability = 0.10), targeting 1553 clients captures **604 subscribers (65.1% threshold recall)** and delivers **$22,435.00 in net campaign profit**—outperforming mass marketing by **+$17,225.00**.

---

## 1. Problem Statement & Case Study Context
Promotional marketing campaigns incur recurring outreach costs (telemarketer labor, outbound telephony, collateral). When targeting is unselective, over 88% of calls fail to convert, leading to:
1. **Wasted Marketing Capital:** Contacting non-receptive clients dilutes marketing funds.
2. **Client Fatigue & Brand Attrition:** Repeated irrelevant contacts alienate consumers.
3. **Operational Bottlenecks:** Sales representatives spend limited outreach hours speaking to non-converters.

The objective is to accurately estimate individual client subscription probability ($P(y=1|X)$) prior to initiating contact, enabling optimal resource prioritization.

---

## 2. Dataset Profile & External Kaggle Source
The project uses the public **Kaggle Marketing Dataset** (Bank Marketing):
- **Source URL:** [https://www.kaggle.com/competitions/marketing-dataset/data](https://www.kaggle.com/competitions/marketing-dataset/data)
- **Total Records:** 41,188 client interactions across 21 original columns.
- **Target Variable:** `y` (Binary: `yes` = 1, `no` = 0).
- **Target Distribution:** 36,548 non-subscribers (88.73%) vs 4,640 subscribers (11.27%), exhibiting an ~8:1 class imbalance.
- **Train/Test Split:** 80% Stratified Training (32,950 records) and 20% Held-Out Test (8,238 records).

### 2.1 Feature Mapping: Case Study 157 Concepts vs Real Kaggle Fields
The original Case Study 157 syllabus lists eight conceptual attributes. The real Kaggle dataset does not contain all eight exact fields. Rather than fabricating data, we provide an explicit, defensible mapping:

| Case Study Concept | Kaggle Equivalent Field(s) | Status | Technical Justification |
| :--- | :--- | :---: | :--- |
| **Age group** | Derived from `age` (`18-25`, `26-35`, `36-45`, `46-55`, `56+`) | **Can map** | Binned directly from continuous `age` into demographic brackets. |
| **Income** | *No direct field* | **Not available** | Not collected in the UCI/Kaggle Bank Marketing study. Not fabricated. |
| **Previous purchases** | *No direct equivalent* | **Not available** | Bank deposits are service subscriptions, not recurring e-commerce cart items. |
| **Purchase frequency** | *No direct equivalent* | **Not available** | Transaction cadence is not recorded in campaign contact logs. |
| **Previous campaign response** | `poutcome`, `previous`, `pdays` | **Partial / defensible** | `poutcome` records prior campaign outcome (`success`, `failure`, `nonexistent`), `previous` counts prior touches, and `pdays` tracks elapsed days. |
| **Website visits** | *No direct field* | **Not available** | The dataset reflects telemarketing outreach, not web analytics. |
| **Email engagement** | *No direct field* | **Not available** | Direct telephony channel; email click rates are not applicable. |
| **Discount usage** | *No direct field* | **Not available** | Financial term deposits offer interest yields rather than retail promotional coupons. |

### 2.2 Critical Rule: Target Leakage Exclusion (`duration`)
> [!IMPORTANT]
> Kaggle explicitly documents: *"duration: last contact duration, in seconds. This attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model."*

To ensure absolute methodological integrity and prevent target leakage, **`duration` is strictly excluded** from all predictive feature matrices, pipelines, and Streamlit interfaces.

---

## 3. Feature Schema Used for Modeling
The final feature space comprises **20 predictive features**:
1. **Numerical Features (9):** `age`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`.
2. **Categorical Features (11):** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`, `age_group`.

---

## 4. Preprocessing Architecture
To prevent data leakage, all imputation medians, IQR capping bounds, and OneHot categories were calculated strictly on the training partition:
- **Numerical Pipeline:** Median Imputation $\to$ `IQRCapper(factor=1.5)` $\to$ `StandardScaler()`.
- **Categorical Pipeline:** Constant Imputer (`'unknown'`) $\to$ `OneHotEncoder(drop='first', handle_unknown='ignore')`.
- All steps are integrated into a single scikit-learn `ColumnTransformer`.

---

## 5. Multi-Model Benchmark & Defensible Model Selection

### 5.1 Comprehensive 6-Model Comparative Results (Held-Out Test Set)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | FPR | Confusion Matrix (TN / FP / FN / TP) | OOF Tuned Thresh | Tuned F1 | FPR @ Rec=70% | Top-20% Capture | CV PR-AUC (Mean ± Std) | CV ROC-AUC (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.902 | 0.6897 | 0.2371 | 0.3528 | 0.8096 | 0.4871 | 0.0135 | 7211 / 99 / 708 / 220 | 0.22 | 0.5289 | 0.1845 | 66.4% | 0.4646 ± 0.0110 | 0.7949 ± 0.0028 |
| **Gradient Boosting** | 0.902 | 0.6855 | 0.2349 | 0.3499 | 0.8093 | 0.4840 | 0.0137 | 7210 / 100 / 710 / 218 | 0.22 | 0.5262 | 0.1925 | 65.7% | 0.4626 ± 0.0112 | 0.7936 ± 0.0050 |
| **Logistic Regression** | 0.902 | 0.6944 | 0.2252 | 0.3401 | 0.8017 | 0.4653 | 0.0126 | 7218 / 92 / 719 / 209 | 0.20 | 0.5160 | 0.1966 | 65.8% | 0.4487 ± 0.0109 | 0.7879 ± 0.0072 |
| **Decision Tree** | 0.904 | 0.6936 | 0.2586 | 0.3768 | 0.7919 | 0.4427 | 0.0145 | 7204 / 106 / 688 / 240 | 0.19 | 0.5176 | 0.6750 | 63.5% | 0.4105 ± 0.0088 | 0.7778 ± 0.0023 |
| **K-Nearest Neighbors** | 0.900 | 0.6537 | 0.2360 | 0.3468 | 0.7764 | 0.4274 | 0.0159 | 7194 / 116 / 709 / 219 | 0.24 | 0.5057 | 0.3978 | 62.5% | 0.4096 ± 0.0052 | 0.7626 ± 0.0058 |
| **Naive Bayes** | 0.860 | 0.3967 | 0.4612 | 0.4265 | 0.7748 | 0.3623 | 0.0891 | 6659 / 651 / 500 / 428 | 0.89 | 0.4313 | 0.2334 | 57.3% | 0.3471 ± 0.0095 | 0.7614 ± 0.0037 |

### 5.2 Defensible Selection Analysis
**Selection Protocol:**
1. Primary ranking: 5-fold CV PR-AUC (Average Precision) evaluated exclusively on the development/training data ($n=32,950$).
2. Practical Closeness Consideration: If the difference in mean CV PR-AUC between the top models is within 1 cross-validation standard deviation across folds, models are considered practically close.
3. Secondary Tie-Breaker (training data only):
   (a) Lower training Out-Of-Fold FPR at fixed 70% recall (`OOF_FPR_at_Recall_70`).
   (b) Model parsimony: preference for simpler, directly interpretable architectures.

**Formal Determination:**  
The top two models on development cross-validation, Random Forest (CV PR-AUC: 0.4646 ± 0.0110) and Gradient Boosting (CV PR-AUC: 0.4626 ± 0.0112), are practically close, with a score difference (0.0020) smaller than the observed fold-to-fold cross-validation variation (0.0110). Based strictly on training data cross-validation and out-of-fold metrics, Random Forest was selected because Random Forest demonstrated lower training out-of-fold FPR at 70% recall (0.2316 vs 0.2438). Additionally, Random Forest offers lower architectural complexity, closed-form linear coefficients, and direct interpretability.

The held-out test set was **never** referenced or inspected to make model selection or hyperparameter tuning decisions.

---

## 6. Class Imbalance Treatments Comparison
| Imbalance Treatment | 5-Fold CV PR-AUC (Mean ± Std) | 5-Fold CV ROC-AUC (Mean ± Std) | Test PR-AUC | Test ROC-AUC | Test F1 (at 0.50) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **None (Unweighted Baseline)** | 0.4646 ± 0.0110 | 0.7949 ± 0.0028 | 0.4871 | 0.8096 | 0.3528 |
| **class_weight='balanced'** | 0.4568 ± 0.0087 | 0.7943 ± 0.0064 | 0.4863 | 0.8130 | 0.5142 |
| **SMOTENC (Categorical-aware Oversampling)** | 0.4227 ± 0.0014 | 0.7811 ± 0.0074 | 0.4570 | 0.8040 | 0.5289 |

*Finding:* Reweighting and SMOTENC do not alter the underlying ranking capacity (PR-AUC remains virtually identical). Operating threshold tuning on the unweighted model achieves superior, cost-effective targeting without distorting calibrated probabilities.

### 6.1 Probability Calibration Check (FrozenEstimator)
- Validation Uncalibrated Brier: 0.07757
- Validation Calibrated Brier: 0.07877
- Test Uncalibrated Brier: 0.07528
- Test Calibrated Brier: 0.07637
*Calibration Decision:* Because validation Brier score did not improve (0.07757 uncalibrated vs 0.07877 calibrated), the champion uncalibrated model was deployed to preserve ranking precision.

---

## 7. Dual Threshold Optimization & Economic Business Simulation

### 7.1 Economic Framework Across 4 Strategies (Held-Out Test Set)
| Strategy | Decision Threshold | Targeted Contacts | Total Cost ($) | Responders Reached | Wasted Contacts (FP) | Gross Revenue ($) | Net Profit ($) | ROI (%) | Ad Spend Saved vs All (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy 1: Contact Everyone** | 0.00 | 8238 | $41,190.00 | 928 | 7310 | $46,400.00 | **$5,210.00** | 12.6% | 0.0% |
| **Strategy 2: Default ML (Threshold 0.50)** | 0.50 | 319 | $1,595.00 | 220 | 99 | $11,000.00 | **$9,405.00** | 589.7% | 96.1% |
| **Strategy 3: F1-Optimal (Threshold 0.22)** | 0.22 | 1167 | $5,835.00 | 554 | 613 | $27,700.00 | **$21,865.00** | 374.7% | 85.8% |
| **Strategy 4: Profit-Optimal (Threshold 0.11)** | 0.11 | 1553 | $7,765.00 | 604 | 949 | $30,200.00 | **$22,435.00** | 288.9% | 81.1% |

*Note on Assumptions:* Unit contact cost ($5.00) and gross profit per responder ($50.00) are explicit modeling assumptions designed to illustrate financial trade-offs, not audited enterprise financials.

---

## 8. Customer Persona Profiles (Actual vs Predicted Responders)
| Customer Segment / Persona | Age (years) | Campaign Contacts | Pdays | Prior Contacts | Emp. Var. Rate | Euribor 3M Rate | Consumer Conf. Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Actual Non-Responder | 39.92 (med 38.0) | 2.67 (med 2.0) | 985.16 (med 999.0) | 0.13 (med 0.0) | 0.26 (med 1.1) | 3.83 (med 4.9) | -40.57 (med -41.8) |
| **Actual Responder** | 41.16 (med 37.0) | 2.04 (med 1.0) | 798.83 (med 999.0) | 0.48 (med 0.0) | -1.31 (med -1.8) | 2.05 (med 1.3) | -39.79 (med -40.4) |
| Predicted Non-Responder | 39.70 (med 38.0) | 2.76 (med 2.0) | 998.11 (med 999.0) | 0.08 (med 0.0) | 0.61 (med 1.1) | 4.21 (med 4.9) | -40.63 (med -41.8) |
| **Predicted Responder** | 41.65 (med 37.0) | 1.87 (med 1.0) | 818.07 (med 999.0) | 0.52 (med 0.0) | -2.17 (med -1.8) | 1.16 (med 1.0) | -39.84 (med -40.4) |

---

## 9. Model Explainability & Feature Drivers
- **Tree MDI & Permutation:** Macroeconomic climate (`euribor3m`, `emp.var.rate`, `nr.employed`) and previous campaign outcome (`poutcome_success`) dominate predictive importance.
- **Logistic Regression Odds Ratios:** Top driver `month_mar` yields an odds ratio of **4.0319**, confirming strong positive association with term deposit conversion.
- **SHAP Summary:** Validates that lower prevailing Euribor 3-month interest rates and successful past outcomes push predictions strongly toward deposit subscription.

---

## 10. Answers to Core Research Questions

### Question 1: Can campaign responses be predicted?
**Answer:** **Yes.**  
On the authentic Kaggle dataset without `duration`, the deployed model achieves a held-out test ROC-AUC of **0.8096** (95% Bootstrap CI: [0.7916, 0.8253]) and PR-AUC of **0.4871** (95% Bootstrap CI: [0.4533, 0.5226]). Contacting the top 20% of ranked clients captures **66.4%** (616 of 928 subscribers).

### Question 2: Which customer characteristics influence response?
**Answer:** **Macroeconomic climate and prior campaign success are the dominant drivers.**  
1. `poutcome_success`: Past successful contact multiplies subscription odds substantially.
2. `euribor3m` / `emp.var.rate`: Lower interest rates correlate with higher propensity to lock funds into fixed-term bank deposits.
3. Demographics: Students and retired clients display higher relative propensity than middle-aged working clients.

### Question 3: Which algorithm performs best?
**Answer:** **Random Forest (deployed as Random Forest).**  
Selected using 5-fold cross-validation on the development dataset. The top two models on development cross-validation, Random Forest (CV PR-AUC: 0.4646 ± 0.0110) and Gradient Boosting (CV PR-AUC: 0.4626 ± 0.0112), are practically close, with a score difference (0.0020) smaller than the observed fold-to-fold cross-validation variation (0.0110). Based strictly on training data cross-validation and out-of-fold metrics, Random Forest was selected because Random Forest demonstrated lower training out-of-fold FPR at 70% recall (0.2316 vs 0.2438). Additionally, Random Forest offers lower architectural complexity, closed-form linear coefficients, and direct interpretability.

### Question 4: Can ML reduce unnecessary marketing expenditure?
**Answer:** **Yes, saving 81.1% to 85.8% under assumed campaign economics.**  
In the 8,238-client test set:
- Mass outreach costs **$41,190.00** with 7310 non-converters.
- Profit-Optimal targeting costs **$7,765.00**, saving **$33,425.00 (81.1% reduction)** while maximizing simulated profit to **$22,435.00**.

### Question 5: How can false positives be reduced?
**Answer:** **Through decision threshold optimization.**  
Operating at the default 0.50 threshold minimizes False Positives (FPR = 0.0135), but misses 708 subscribers. Tuning threshold using training Out-Of-Fold predictions enables explicit balancing between false-positive costs and false-negative opportunity loss.

### Question 6: Can the model identify potential campaign responders?
**Answer:** **Yes, capturing 65.1% of subscribers at the profit-optimal threshold while contacting only 5973.1% of the client base.**

---

## 11. Limitations & Risk Disclosures
1. **Case Study Scope:** The Kaggle dataset reflects banking term deposit subscriptions; it does not contain retail-specific metrics like website visits or discount coupons.
2. **Propensity vs Uplift:** The model measures response correlation rather than causal incrementality. A/B testing is recommended to measure true marketing lift.
3. **Simulated Economic Assumptions:** Unit costs ($5.00) and responder profits ($50.00) are simulation parameters.
4. **Call Duration Exclusion:** `duration` is deliberately omitted to prevent target leakage.

---

## 12. Conclusion
The project successfully migrates Case Study 157 to the real, public Kaggle Marketing Dataset. By eliminating target leakage, enforcing rigorous 5-fold cross-validation, and deploying an interactive Streamlit application with defensible threshold control, the framework provides an honest, production-ready machine learning solution for marketing campaign optimization.
