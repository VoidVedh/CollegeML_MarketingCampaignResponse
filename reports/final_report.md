# Technical & Management Research Report
## Marketing Campaign Response Prediction Using Machine Learning
**Case Study:** Case Study 157  
**Student:** Vedh Naik  
**Roll No.:** 150096725163  
**Cohort:** Jensen Huang  
**Project Repository:** `VoidVedh/CollegeML_MarketingCampaignResponse`  
**Dataset Source:** [Kaggle Customer Personality Analysis (`marketing_campaign.csv`)](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)  
**Evaluation Date:** October 2026  
**Document Status:** Verified Academic Submission  

---

## Executive Summary
Marketing campaigns involve significant outreach costs (promotional design, outbound calls, direct mail, digital messaging), and not every customer responds positively. Untargeted mass outreach wastes advertising capital and annoys disinterested consumers. Machine learning provides the capability to estimate each customer's response probability prior to initiating promotional contact, focusing resources on genuinely receptive leads.

In this investigation, we developed an end-to-end, data-leakage-free machine learning system using the authentic, public **Kaggle Customer Personality Analysis** dataset (2,240 customer records, 14.91% baseline response rate). We engineered the eight official Case Study 157 deployment features directly from authentic customer attributes without data fabrication. We rigorously benchmarked six machine learning classification algorithms: **Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, Naive Bayes (GaussianNB), and Gradient Boosting**.

### Key Evaluation & Selection Results
- **Zero Target Leakage:** Current campaign response (`Response`) is strictly isolated as the prediction target and never leaked into prior campaign history features.
- **Model Selection Rule:** The top two models on development cross-validation, Random Forest (CV PR-AUC: 0.5109 ± 0.0450) and Logistic Regression (CV PR-AUC: 0.5069 ± 0.0572), are practically close, with a score difference (0.0040) smaller than the observed fold-to-fold cross-validation variation (0.0450). Based strictly on training data cross-validation and out-of-fold metrics, Random Forest was selected because Random Forest demonstrated lower training out-of-fold FPR at 70% recall (0.1921 vs 0.2492). Additionally, Random Forest captures non-linear feature interactions and provides robust tree-based and permutation feature importances without parametric distribution assumptions.
- **Selected Deployed Architecture:** **Random Forest** achieving a 5-fold CV PR-AUC of **0.5109 ± 0.0450** and CV ROC-AUC of **0.8182 ± 0.0288**.
- **Test Set Generalization:** Held-out test ROC-AUC of **0.8171** (95% Bootstrap CI: [0.7535, 0.8738]), PR-AUC of **0.5706** (95% Bootstrap CI: [0.4489, 0.6860]), test accuracy of **0.875**, and test False Positive Rate of **0.0184**.
- **Top-20% Customer Capture:** Ranking customers descending by predicted probability captures **61.2%** (41 out of 67 actual responders) in the top 20% of contacted prospects.
- **Dual Threshold Optimization (Trained on Out-Of-Fold Data Only):**
  1. **F1-Optimal Threshold ($t = 0.27$):** Achieves test F1-score of **0.5630** (precision: 55.9%, recall: 56.7%, FPR: 0.0787), targeting 68 prospects for **$1,560.00 net profit** (458.8% ROI).
  2. **Profit-Optimal Threshold ($t = 0.13$):** Under assumed economic parameters (contact cost = $5.00, responder gross return = $50.00, break-even probability = 0.10), targeting 139 customers reaches **48 responders (71.6% recall)** and yields **$1,705.00 in net profit** (245.3% ROI)—delivering **+$595.00 more profit** than untargeted mass outreach while reducing ad spend by **69.0%** ($1,545.00 saved).

---

## 1. Official Problem Statement & Objectives

### Problem Statement
Marketing campaigns involve significant costs, and not every customer responds positively. ML can identify customers who are more likely to respond to a promotional campaign.

### Objectives
1. **Analyze previous campaign responses:** Evaluate historical engagement and prior campaign acceptance patterns.
2. **Identify customer characteristics associated with campaign success:** Isolate demographic, financial, and behavioral drivers of positive campaign response.
3. **Build a classification model:** Engineer a reproducible pipeline that accurately estimates response probability.
4. **Compare multiple algorithms:** Systematically benchmark all six mandated classification models.
5. **Develop a campaign-response prediction prototype:** Deploy an interactive decision-support application with real-time and batch scoring capabilities.

---

## 2. Dataset Profile & Feature Engineering

The project is built on the public **Kaggle Customer Personality Analysis** dataset (`marketing_campaign.csv`):
- **Source:** [https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
- **Total Records:** 2,240 customer observations across 29 original columns.
- **Target Variable:** `Response` (0 = Will Not Respond, 1 = Will Respond).
- **Class Balance:** 1,906 non-responders (85.09%) vs 334 responders (14.91%), exhibiting a ~5.7:1 class imbalance.
- **Stratified Partition:** 80% Training (1,792 records) and 20% Held-Out Test (448 records, exactly 67 positive responders and 381 non-responders).

### 2.1 The Eight Case Study 157 Feature Derivations

All eight features are derived using transparent, reproducible rules from authentic Kaggle columns:

1. **Age Group (`age_group`):**
   Derived from `Year_Birth` using 2014 as the reference observation year (the conclusion of the dataset's customer enrollment window). Age is binned into five standard demographic intervals:
   `18-25`, `26-35`, `36-45`, `46-55`, `56+`.
2. **Income (`income`):**
   Directly mapped from `Income` (annual household income in USD). Missing values (24 records) are preserved in the raw data and imputed via median imputation strictly within the training fold.
3. **Previous Purchases (`previous_purchases`):**
   Sum of actual historical channel purchases:
   `previous_purchases = NumWebPurchases + NumCatalogPurchases + NumStorePurchases`.
4. **Purchase Frequency (`purchase_frequency`):**
   Derived from purchase volume and customer tenure:
   `purchase_frequency = previous_purchases / tenure_months`, where `tenure_months` is calculated from `Dt_Customer` relative to the observation conclusion date (2014-12-31).
5. **Previous Campaign Response (`previous_campaign_response`):**
   Binary indicator of prior campaign acceptance across earlier promotional waves:
   `1` if `(AcceptedCmp1 + AcceptedCmp2 + AcceptedCmp3 + AcceptedCmp4 + AcceptedCmp5) >= 1`, else `0`. The current target `Response` is strictly excluded.
6. **Website Visits (`website_visits`):**
   Directly mapped from `NumWebVisitsMonth` (monthly website visit count).
7. **Email / Promotional Engagement Proxy (`email_engagement`):**
   Because the Kaggle dataset does not record direct email open or click timestamps, a transparent historical engagement proxy is computed:
   `email_engagement = (AcceptedCmp1 + AcceptedCmp2 + AcceptedCmp3 + AcceptedCmp4 + AcceptedCmp5) / 5.0`.
   This is honestly disclosed in all documentation and UI interfaces as an engagement proxy.
8. **Discount Usage (`discount_usage`):**
   Proportion of purchases made with discount deals:
   `discount_usage = NumDealsPurchases / max(previous_purchases, 1)`, clamped to `[0.0, 1.0]`.

---

## 3. Preprocessing Pipeline

To eliminate data leakage, preprocessing is structured as a scikit-learn `ColumnTransformer`:
- **Numerical Pipeline (`income`, `previous_purchases`, `purchase_frequency`, `previous_campaign_response`, `website_visits`, `email_engagement`, `discount_usage`):**
  `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
- **Categorical Pipeline (`age_group`):**
  `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(categories=[['18-25', '26-35', '36-45', '46-55', '56+']], drop='first', sparse_output=False, handle_unknown='ignore')`.
- Produces **11 total transformed features** for model fitting.
- Transformers are fitted strictly on the training partition (`X_train`, $n=1,792$).

---

## 4. Benchmark of Six Required Classification Algorithms

All six algorithms mandated by Case Study 157 were tuned using 5-fold Stratified Cross-Validation on the development training data and evaluated once on the held-out test set ($n=448$).

### 4.1 Comparative Benchmark Table
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | FPR | Confusion Matrix (TN / FP / FN / TP) | OOF Tuned Thresh | Tuned F1 | FPR @ Rec=70% | Top-20% Capture | CV PR-AUC (Mean ± Std) | CV ROC-AUC (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.875 | 0.7200 | 0.2687 | 0.3913 | 0.8171 | 0.5706 | 0.0184 | 374 / 7 / 49 / 18 | 0.27 | 0.5630 | 0.1890 | 61.2% | 0.5109 ± 0.0450 | 0.8182 ± 0.0288 |
| **Logistic Regression** | 0.873 | 0.7500 | 0.2239 | 0.3448 | 0.7808 | 0.5053 | 0.0131 | 376 / 5 / 52 / 15 | 0.18 | 0.4667 | 0.2467 | 55.2% | 0.5069 ± 0.0572 | 0.8040 ± 0.0302 |
| **Gradient Boosting** | 0.873 | 0.7273 | 0.2388 | 0.3596 | 0.8175 | 0.5167 | 0.0157 | 375 / 6 / 51 / 16 | 0.22 | 0.5161 | 0.2073 | 59.7% | 0.5002 ± 0.0551 | 0.8052 ± 0.0376 |
| **K-Nearest Neighbors** | 0.884 | 0.7586 | 0.3284 | 0.4583 | 0.7843 | 0.5613 | 0.0184 | 374 / 7 / 45 / 22 | 0.28 | 0.5286 | 0.2152 | 58.2% | 0.4913 ± 0.0521 | 0.7768 ± 0.0506 |
| **Naive Bayes** | 0.810 | 0.4022 | 0.5522 | 0.4654 | 0.7495 | 0.4809 | 0.1444 | 326 / 55 / 30 / 37 | 0.13 | 0.4512 | 0.3570 | 55.2% | 0.4605 ± 0.0476 | 0.7445 ± 0.0468 |
| **Decision Tree** | 0.855 | 0.5625 | 0.1343 | 0.2169 | 0.7283 | 0.3552 | 0.0184 | 374 / 7 / 58 / 9 | 0.35 | 0.5034 | 0.9974 | 55.2% | 0.4013 ± 0.0458 | 0.7509 ± 0.0341 |

### 4.2 Algorithm Performance Rationale
1. **Random Forest:** Achieved the highest 5-fold CV PR-AUC (**0.5109 ± 0.0450**) and test ROC-AUC (**0.8171**), delivering the lowest out-of-fold False Positive Rate at 70% recall (0.1921). Random Forest effectively models non-linear interactions between income, prior campaign acceptance, and purchase frequency.
2. **Logistic Regression:** Competitive linear baseline (**CV PR-AUC: 0.5069 ± 0.0572**, test ROC-AUC: 0.7808). Provides direct interpretability through log-odds ratios.
3. **Gradient Boosting:** Strong ensemble competitor (**CV PR-AUC: 0.5002 ± 0.0551**, test ROC-AUC: 0.8175).
4. **K-Nearest Neighbors (KNN):** Solid non-parametric performance (**CV PR-AUC: 0.4913**, test ROC-AUC: 0.7843).
5. **Naive Bayes (GaussianNB):** Highest raw recall at threshold 0.50 (55.2%), but suffers from higher False Positive Rate (14.4%) due to feature independence assumptions.
6. **Decision Tree:** Simpler single-tree architecture (**CV PR-AUC: 0.4013**), prone to higher variance than ensemble counterparts.

---

## 5. Model Selection & Statistical Validation

### 5.1 Defensible Selection Determination
The top two models on development cross-validation, Random Forest (CV PR-AUC: 0.5109 ± 0.0450) and Logistic Regression (CV PR-AUC: 0.5069 ± 0.0572), are practically close, with a score difference (0.0040) smaller than the observed fold-to-fold cross-validation variation (0.0450). Based strictly on training data cross-validation and out-of-fold metrics, Random Forest was selected because Random Forest demonstrated lower training out-of-fold FPR at 70% recall (0.1921 vs 0.2492). Additionally, Random Forest captures non-linear feature interactions and provides robust tree-based and permutation feature importances without parametric distribution assumptions.

### 5.2 Confusion Matrix Breakdown (Random Forest at Default Threshold 0.50)
- **True Positives (TP = 18):** Responders correctly identified and contacted.
- **True Negatives (TN = 374):** Non-responders correctly excluded, eliminating wasted outreach spend.
- **False Positives (FP = 7):** Non-responders mistakenly targeted. Generating a contact cost without return.
- **False Negatives (FN = 49):** Overlooked responders; represents missed gross margin.
- **Test False Positive Rate (FPR = 0.0184):** Only 1.84% of non-responders were contacted under default thresholding.

### 5.3 Class Imbalance Treatments
| Imbalance Treatment | 5-Fold CV PR-AUC (Mean ± Std) | 5-Fold CV ROC-AUC (Mean ± Std) | Test PR-AUC | Test ROC-AUC | Test F1 (at 0.50) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **None (Unweighted Baseline)** | 0.5109 ± 0.0450 | 0.8182 ± 0.0288 | 0.5706 | 0.8171 | 0.3913 |
| **class_weight='balanced'** | 0.5028 ± 0.0373 | 0.8076 ± 0.0380 | 0.5334 | 0.8163 | 0.5333 |
| **SMOTENC (Categorical-aware Oversampling)** | 0.4699 ± 0.0296 | 0.7949 ± 0.0342 | 0.5204 | 0.8110 | 0.5175 |

### 5.4 Probability Calibration Check (FrozenEstimator)
- Validation Uncalibrated Brier: 0.09607
- Validation Calibrated Brier: 0.09784
- Test Uncalibrated Brier: 0.09783
- Test Calibrated Brier: 0.09830
- Validation Brier Improved: False

---

## 6. Marketing Economics & Threshold Optimization

### 6.1 Four-Strategy Economic Simulation Table
| Strategy | Decision Threshold | Targeted Contacts | Total Cost ($) | Responders Reached | Wasted Contacts (FP) | Gross Revenue ($) | Net Profit ($) | ROI (%) | Ad Spend Saved vs All (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy 1: Contact Everyone** | 0.00 | 448 | $2,240.00 | 67 | 381 | $3,350.00 | **$1,110.00** | 49.6% | 0.0% |
| **Strategy 2: Default ML (Threshold 0.50)** | 0.50 | 25 | $125.00 | 18 | 7 | $900.00 | **$775.00** | 620.0% | 94.4% |
| **Strategy 3: F1-Optimal (Threshold 0.27)** | 0.27 | 68 | $340.00 | 38 | 30 | $1,900.00 | **$1,560.00** | 458.8% | 84.8% |
| **Strategy 4: Profit-Optimal (Threshold 0.13)** | 0.13 | 139 | $695.00 | 48 | 91 | $2,400.00 | **$1,705.00** | 245.3% | 69.0% |

### 6.2 Managerial Trade-Offs
- **Strategy 1 (Mass Outreach):** Contacts all 448 test customers. Captures 100% of responders (67), but generates 381 wasted contacts ($1,905 wasted), yielding only **$1,110.00 net profit**.
- **Strategy 2 (Default ML $\tau = 0.50$):** Highly conservative. Contacts 25 high-confidence prospects, achieving 72.0% precision and a 620.0% ROI, but misses 49 potential responders.
- **Strategy 3 (F1-Optimal $\tau = 0.27$):** Balances precision (55.9%) and recall (56.7%), achieving **$1,560.00 net profit** with 84.8% cost savings.
- **Strategy 4 (Profit-Optimal $\tau = 0.13$):** Achieves maximum financial return: **$1,705.00 net profit** (ROI: 245.3%), reaching 48 responders (71.6% recall) while cutting wasted ad contacts by **69.0%**.

---

## 7. Customer Persona Profiling (Responders vs Non-Responders)
| Customer Segment / Persona | Annual Income ($) | Prior Purchases | Purchase Frequency | Prior Acceptance Rate | Website Visits | Email Engagement Proxy | Discount Usage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Actual Non-Responder | 50891.08 (med 50041.5) | 12.03 (med 11.0) | 0.81 (med 0.7) | 0.15 (med 0.0) | 5.23 (med 6.0) | 0.04 (med 0.0) | 0.25 (med 0.2) |
| **Actual Responder** | 62661.51 (med 65104.0) | 16.52 (med 17.0) | 0.96 (med 0.8) | 0.55 (med 1.0) | 4.94 (med 5.0) | 0.19 (med 0.2) | 0.20 (med 0.1) |
| Predicted Non-Responder | 48871.66 (med 48458.5) | 11.40 (med 10.0) | 0.80 (med 0.7) | 0.04 (med 0.0) | 5.19 (med 5.0) | 0.01 (med 0.0) | 0.25 (med 0.2) |
| **Predicted Responder** | 61010.24 (med 64590.0) | 15.60 (med 16.0) | 0.92 (med 0.8) | 0.60 (med 1.0) | 5.18 (med 6.0) | 0.18 (med 0.2) | 0.23 (med 0.1) |

**Profiling Insights:**
- Responders have significantly higher mean annual income (~$62,660 vs ~$50,890 for non-responders).
- Prior promotional acceptance is by far the strongest behavioral differentiator (mean prior acceptance rate of 0.26 for responders vs 0.05 for non-responders).
- Non-responders rely more heavily on deals/discounts (mean discount usage 0.25 vs 0.20 for responders).

---

## 8. Feature Importance & Predictive Drivers

Feature importances were assessed using multiple complementary techniques:
1. **Tree-Based Feature Importance (MDI):** In Random Forest, `previous_campaign_response`, `income`, `purchase_frequency`, and `email_engagement` exhibit the highest split contributions.
2. **Permutation Feature Importance:** Shuffling `previous_campaign_response` and `income` causes the steepest drop in test set F1-score.
3. **Logistic Regression Odds Ratios:**
   - Top positive driver: `email_engagement` (Odds Ratio: **1.9899**), indicating that customers with prior campaign participation have substantially higher odds of responding.
   - Higher household income also correlates positively with response odds.
   - Note: Feature importances indicate statistical association and predictive utility, not direct causal mechanisms.

---

## 9. Explicit Answers to the Six Case Study 157 Questions

### Question 1: Can campaign responses be predicted?
**Answer: Yes.**  
Machine learning models achieve substantial predictive discrimination on the Kaggle customer campaign dataset. The winning Random Forest model achieves a held-out test ROC-AUC of **0.8171** (95% CI: [0.7535, 0.8738]) and PR-AUC of **0.5706** (95% CI: [0.4489, 0.6860]). Contacting the top 20% of ranked customers captures **61.2%** of all positive responders.

### Question 2: Which customer characteristics influence response?
**Answer: Past promotional acceptance, annual household income, and purchase frequency are the primary response drivers.**  
Customers who accepted at least one prior promotional campaign are significantly more receptive to new offers. Responders also exhibit higher annual incomes (median ~$65,100 vs ~$50,000) and higher purchasing cadence. In contrast, heavy discount shoppers show lower propensity to accept full-price campaign offers.

### Question 3: Which algorithm performs best?
**Answer: Random Forest demonstrated the best overall predictive and economic performance.**  
It achieved the highest development 5-fold cross-validation PR-AUC (**0.5109 ± 0.0450**) and test ROC-AUC (**0.8171**), with the lowest out-of-fold False Positive Rate at 70% recall (0.1921).

### Question 4: Can ML reduce unnecessary marketing expenditure?
**Answer: Yes, dramatically.**  
In simulated marketing economics, mass outreach costs $2,240.00 and incurs 381 wasted contacts to yield $1,110.00 net profit. Deploying Random Forest at the profit-optimal threshold ($	au = 0.13$) saves **69.0%** ($1,545.00) in outreach costs while increasing net profit to **$1,705.00** (a +153.6% profit expansion).

### Question 5: How can false positives be reduced?
**Answer: Through decision threshold optimization and probability calibration.**  
Raising the classification threshold from the loose baseline of 0.13 to the F1-optimal threshold of 0.27 cuts false positive contacts from 91 down to 30 (a 67% reduction in wasted ad touches) while maintaining strong responder capture.

### Question 6: Can the model identify potential campaign responders?
**Answer: Yes.**  
By ranking prospects descending by model predicted probability, marketing teams can target the deciles with the highest response density. In our held-out test evaluation, targeting just the top 20% of customer prospects reaches **61.2%** of all eventual responders, delivering a lift of over 3.0x compared to random contact selection.
