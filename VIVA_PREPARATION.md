# Viva Voce Comprehensive Preparation Guide
## Case Study 157: Marketing Campaign Response Prediction Using Machine Learning

**Student Details:**
- **Name:** Vedh Naik
- **Roll Number:** 150096725163
- **Cohort:** Jensen Huang
- **Semester:** B.Tech CSE Semester V (Machine Learning)
- **Repository:** `https://github.com/VoidVedh/CollegeML_MarketingCampaignResponse`

---

## 1. Thirty-Second Elevator Pitch

> *"Good morning, Professors. My project is Case Study 157: Marketing Campaign Response Prediction Using Machine Learning. Direct telemarketing campaigns are cost-intensive, and mass outreach typically results in wasted ad spend because only a small fraction of clients subscribe. 
> 
> Using the real Kaggle Bank Marketing Dataset of 41,188 clients, I engineered a leakage-free machine learning pipeline where call duration was strictly excluded to ensure realistic pre-call deployment. I implemented and tuned all six required algorithms using 5-fold Stratified Cross-Validation. 
> 
> Random Forest emerged as the champion model with a cross-validation PR-AUC of 0.4646 and test ROC-AUC of 0.8096. By tuning the operating decision threshold to 0.11 based on unit economics ($5 contact cost vs $50 conversion value), the model saves 81.1% in unnecessary marketing expenditure while capturing 66.4% of all subscribers within the top 20% of outreach. I deployed the solution as an interactive Streamlit web application supporting real-time single and batch CSV predictions."*

---

## 2. Problem Statement & Academic Objectives

### What is the core business problem?
Marketing outreach costs money ($5 per call). In our dataset, only **11.27%** of customers actually subscribe to a bank term deposit. Calling everyone (mass outreach) wastes 88.73% of the marketing budget on uninterested clients and causes brand fatigue. Machine learning allows us to rank customers by subscription propensity and contact only those likely to convert.

### What are the five syllabus objectives?
1. Analyze previous campaign interaction patterns.
2. Identify client demographics and macroeconomic context associated with campaign success.
3. Build and preprocess a binary classification pipeline.
4. Systematically benchmark the six mandatory algorithms.
5. Develop and deploy an interactive Streamlit prediction application.

---

## 3. Dataset Architecture & Key Statistics

| Parameter | Exact Value | Explanation |
| :--- | :--- | :--- |
| **Dataset Source** | Kaggle / UCI Bank Marketing | `bank-additional-full` benchmark dataset |
| **Total Records** | **41,188** rows | Genuine client interactions |
| **Raw Features** | **20 predictive features** | 9 numerical + 11 categorical |
| **Target Variable** | `y` | Binary: `'yes'` (1) vs `'no'` (0) |
| **Class Distribution** | `no`: 36,548 (88.73%)<br>`yes`: 4,640 (11.27%) | Severe class imbalance (~8:1 ratio) |
| **Train/Test Split** | **80% Train** (32,950 rows)<br>**20% Held-Out Test** (8,238 rows) | Stratified on target `y`; test set touched strictly once |

---

## 4. The Data Leakage Question (Most Crucial Viva Defense!)

### Question: "Why did you exclude `duration` from the model?"
**Answer:**
> *"Call duration is recorded in seconds during or after a phone conversation takes place. When deciding which customers to target before dialing the phone, call duration is unknown. If duration is included in training, the model achieves an artificially inflated ROC-AUC (>0.93) because longer calls naturally correlate with successful conversions. In a real-world deployment, duration is zero before dialing, causing the model to collapse. Removing `duration` is mandatory to guarantee zero data leakage and ensure a genuinely useful pre-contact decision support system."*

---

## 5. College Prompt Inputs vs. Kaggle Features (Honesty Defense)

### Question: "The college problem statement mentions Income, Website Visits, and Discount Usage. Why are they not in your inputs?"
**Answer:**
> *"The college assignment prompt lists eight conceptual marketing variables as illustrative examples. However, the authentic Kaggle Bank Marketing dataset is an empirical telemarketing study for bank term deposits and does not record website clicks, discount codes, or customer personal income. 
> 
> Rather than fabricating fake synthetic numbers or renaming unrelated fields, I adopted a transparent academic approach:
> - `Age group` was mathematically derived from genuine continuous `age` into standard brackets (18-25, 26-35, 36-45, 46-55, 56+).
> - `Previous campaign response` was mapped to `poutcome` (success/failure/nonexistent), `previous`, and `pdays`.
> - Macroeconomic indicators (`euribor3m`, `emp.var.rate`, `cons.price.idx`) serve as realistic economic proxies.
> - Unavailable fields are documented with full transparency in the project report, README, and Streamlit interface."*

---

## 6. Preprocessing & Feature Engineering Pipeline

### What steps are in your Scikit-Learn `ColumnTransformer`?
1. **Numerical Pipeline (9 features):**
   - `SimpleImputer(strategy='median')`: Replaces missing numeric values with training median.
   - `IQRCapper(factor=1.5)`: Caps extreme outliers using the 1.5 * IQR rule. 
     - *Key Viva Detail:* Features like `pdays` (96.3% are 999) and `previous` (86.3% are 0) have identical Q25 and Q75, giving an IQR of 0. A standard IQR capper would collapse them to constants! We implemented a `zero_iqr` guard that leaves zero-IQR features unclipped, preserving their predictive variance.
   - `StandardScaler()`: Standardizes features to mean 0, variance 1.
2. **Categorical Pipeline (11 features):**
   - `SimpleImputer(strategy='constant', fill_value='unknown')`: Handles missing strings.
   - `OneHotEncoder(drop='first', handle_unknown='ignore')`: Converts categorical strings into 47 binary indicator columns, dropping the reference category to prevent multicollinearity (dummy variable trap).
3. **Total Transformed Dimensionality:** 56 total columns.

---

## 7. The Six Required Algorithms & Held-Out Test Results

Evaluated on the **8,238 held-out test customers** (928 actual subscribers):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | False Positive Rate (FPR) | Top-20% Capture Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Winner)** | **90.20%** | **68.97%** | 23.71% | 0.3528 | **0.8096** | **0.4871** | **0.0135** | **66.38%** (616/928) |
| **Gradient Boosting** | 90.17% | 68.55% | 23.49% | 0.3499 | 0.8093 | 0.4840 | 0.0137 | 65.73% (610/928) |
| **Logistic Regression** | 90.16% | 69.44% | 22.52% | 0.3401 | 0.8017 | 0.4653 | 0.0126 | 65.84% (611/928) |
| **Decision Tree** | 90.36% | 69.36% | 25.86% | 0.3768 | 0.7919 | 0.4427 | 0.0145 | 63.47% (589/928) |
| **K-Nearest Neighbors** | 89.99% | 65.37% | 23.60% | 0.3468 | 0.7764 | 0.4274 | 0.0159 | 62.50% (580/928) |
| **Naive Bayes** | 86.03% | 39.67% | 46.12% | 0.4265 | 0.7748 | 0.3623 | 0.0891 | 57.33% (532/928) |

---

## 8. Model Selection Rationale

### Question: "Why did you choose Random Forest over Gradient Boosting?"
**Answer:**
> *"Model selection was conducted strictly on development training data using 5-fold Stratified Cross-Validation. Random Forest achieved the highest development PR-AUC (0.4646 ± 0.0110 vs 0.4626 ± 0.0112 for Gradient Boosting). 
> 
> When examining out-of-fold predictions at a fixed 70% recall target, Random Forest produced a lower False Positive Rate (0.2316 vs 0.2438). On the final held-out test set, Random Forest confirmed this superiority by achieving the highest test PR-AUC (0.4871), highest ROC-AUC (0.8096), and highest top-quintile capture rate (66.38%). Furthermore, bagging in Random Forest provides lower prediction variance and higher stability against multicollinear macroeconomic indicators compared to sequential boosting."*

---

## 9. Class Imbalance Handling

### Question: "How did you address the 88:12 class imbalance?"
**Answer:**
> *"I systematically compared three methodologies on 5-fold cross-validation:
> 1. Unweighted Baseline: CV PR-AUC = **0.4646** (Selected)
> 2. Cost-sensitive weighting (`class_weight='balanced'`): CV PR-AUC = **0.4568**
> 3. Synthetic oversampling via SMOTENC: CV PR-AUC = **0.4227**
> 
> Surprisingly, SMOTENC degraded precision because synthetic interpolation in a 47-dimensional one-hot space creates unrealistic synthetic points in boundary regions. The most effective approach for imbalanced data was training on the natural empirical distribution and performing post-hoc **decision threshold optimization**."*

---

## 10. Decision Threshold Tuning & Economic Simulation

### Question: "Why didn't you just use the standard 0.50 threshold?"
**Answer:**
> *"The default 0.50 threshold assumes equal misclassification costs and symmetric priors. Because only 11.27% of clients subscribe, a 0.50 cutoff produces low recall (23.7%), missing three out of four potential subscribers.
> 
> Using Out-of-Fold training predictions, I optimized two alternative thresholds:
> 1. **F1-Optimal Threshold ($\tau = 0.22$):** Maximizes harmonic mean of precision and recall (F1 jumps from 0.35 to 0.49).
> 2. **Profit-Optimal Threshold ($\tau = 0.11$):** Maximizes expected financial return under unit economic parameters ($5 per contact, $50 profit per conversion)."*

### Economic Comparison Table (Held-Out Test Set: 8,238 clients)

| Strategy | Threshold | Contacts Made | Wasted Contacts (FP) | Subscribers Reached | Spend ($) | Net Profit ($) | Cost Saved |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Mass Outreach** | 0.00 | 8,238 | 7,310 | 928 (100%) | $41,190 | $5,210 | 0.0% |
| **2. Default ML** | 0.50 | 319 | 99 | 220 (23.7%) | $1,595 | $9,405 | 96.1% |
| **3. F1-Optimal** | 0.22 | 1,167 | 613 | 554 (59.7%) | $5,835 | $21,865 | 85.8% |
| **4. Profit-Optimal** | **0.11** | **1,553** | **949** | **604 (65.1%)** | **$7,765** | **$22,435** | **81.1%** |

*Takeaway:* Strategy 4 saves **$33,425 (81.1%)** in marketing expenditure compared to mass outreach while quadrupling net profit ($22,435 vs $5,210).

---

## 11. Key Drivers & Business Insights

### Top Predictive Features:
1. **Timing (`month_mar`):** Odds Ratio = **4.0319**. Contacting customers in March has 4.03x higher conversion odds compared to other months.
2. **Economic Climate (`euribor3m`, `emp.var.rate`):** Lower benchmark interest rates significantly increase term deposit demand, as competing investment yields drop.
3. **Prior Campaign Success (`poutcome_success`):** Odds Ratio = **1.8534**. A client who responded positively in a previous campaign has an 85% higher propensity to subscribe again.
4. **Demographics:** Retired individuals and students exhibit higher relative conversion rates than prime-age employed workers.

*Causation vs Association Disclaimer:* These metrics reflect statistical associations in historical observational data and do not prove causal intervention mechanisms.

---

## 12. Top 25 Rapid-Fire Viva Questions & Answers

#### Q1: What is the target variable?
> **A:** `y`, representing whether the client subscribed to a bank term deposit (`'yes'` = 1, `'no'` = 0).

#### Q2: What metric did you use for model comparison and why?
> **A:** Precision-Recall AUC (PR-AUC / Average Precision). ROC-AUC can be overly optimistic on highly imbalanced datasets (88:12) because a large number of true negatives keeps the False Positive Rate artificially small. PR-AUC focuses strictly on positive class performance.

#### Q3: What is the formula for False Positive Rate?
> **A:** $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$. It represents the proportion of non-interested customers who were incorrectly targeted.

#### Q4: What is the formula for F1-score?
> **A:** $\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$.

#### Q5: What is data leakage?
> **A:** When information from outside the training dataset (or information that would not be available at the time of prediction) is used to create the model. In our project, call duration is data leakage because it is only known after calling.

#### Q6: How did you prevent data leakage in your code?
> **A:** Dropped `duration` before splitting, fitted all imputers, scalers, and outlier cappers strictly on training data inside a `Pipeline`, and tuned thresholds using Out-of-Fold training predictions without touching the test set.

#### Q7: What is Out-of-Fold (OOF) prediction?
> **A:** During 5-fold cross-validation, predictions for each validation fold are generated by models trained only on the other 4 folds. Concatenating them provides an unbiased prediction for every training record, preventing leakage when tuning decision thresholds.

#### Q8: What is Top-20% Customer Capture Rate?
> **A:** When customers are ranked from highest to lowest predicted probability, the top 20% of the ranked list captures **66.38%** (616 out of 928) of all actual subscribers in the test set. This represents a 3.32x lift over random guessing (which would capture only 20%).

#### Q9: What is the Brier Score?
> **A:** The mean squared difference between predicted probabilities and actual binary outcomes: $\text{Brier} = \frac{1}{N}\sum(p_i - y_i)^2$. Lower is better. Our uncalibrated Random Forest achieved 0.07528.

#### Q10: Why didn't you use probability calibration?
> **A:** We tested Isotonic calibration using `FrozenEstimator(CalibratedClassifierCV)`. The validation Brier score slightly degraded (0.07757 uncalibrated vs 0.07877 calibrated). Per our pre-specified validation rule, we deployed the uncalibrated model.

#### Q11: What is the difference between bagging and boosting?
> **A:** Bagging (Random Forest) trains independent trees in parallel on bootstrap samples and averages predictions to reduce variance. Boosting (Gradient Boosting) trains trees sequentially, where each new tree corrects the residual errors of prior trees to reduce bias.

#### Q12: Why did Naive Bayes have high recall but low precision?
> **A:** Naive Bayes assumes all features are conditionally independent. Real-world economic indicators (`euribor3m`, `emp.var.rate`, `nr.employed`) are strongly correlated ($r > 0.90$). The independence violation causes probability overconfidence, pushing many negative cases into the positive prediction zone.

#### Q13: What does the `var_smoothing` hyperparameter do in Gaussian Naive Bayes?
> **A:** It adds a small portion of the largest variance of all features to the variances to stabilize calculations and prevent zero-division in probability density estimation.

#### Q14: How does K-Nearest Neighbors work and what was its best $K$?
> **A:** KNN computes the Euclidean distance between the test customer and all training customers in normalized feature space, assigning the majority class among the $K$ closest neighbors. The tuned optimal $K$ was **21**.

#### Q15: What is the role of `StandardScaler` in KNN vs Random Forest?
> **A:** Essential for KNN because distance metrics are sensitive to feature scales. Random Forest is scale-invariant (splits depend on feature ordering, not magnitude), but scaling was maintained in the shared pipeline for methodological consistency.

#### Q16: How does the Decision Tree prevent overfitting?
> **A:** Through hyperparameter constraints: `max_depth=5` limits tree depth, and `min_samples_split=10` prevents splitting leaves with too few samples.

#### Q17: What criterion did your Decision Tree use?
> **A:** `entropy` (Information Gain), measuring reduction in uncertainty: $H(S) = -p_1 \log_2(p_1) - p_0 \log_2(p_0)$.

#### Q18: What is an Odds Ratio in Logistic Regression?
> **A:** $\text{OR} = \exp(\beta_j)$. If $\text{OR} > 1$, an increase in that feature increases the odds of response. For example, `month_mar` has $\text{OR} = 4.03$, meaning March contacts have 4 times higher conversion odds than the baseline month.

#### Q19: What is the difference between `pd.cut` and `pd.qcut`?
> **A:** `pd.cut` creates equal-width bins (or custom defined intervals like our age groups: 18-25, 26-35, etc.), while `pd.qcut` creates equal-frequency quantile bins.

#### Q20: What is the dummy variable trap in OneHotEncoder?
> **A:** When all $K$ categories of a categorical variable are encoded as $K$ binary columns, the columns sum to 1, causing perfect collinearity with the intercept. Setting `drop='first'` removes the first category, yielding $K-1$ linearly independent columns.

#### Q21: What is Stratified K-Fold?
> **A:** A variation of K-Fold cross-validation where each fold contains approximately the same percentage of positive (11.27%) and negative (88.73%) samples as the complete dataset.

#### Q22: How does Streamlit handle state persistence?
> **A:** Using `st.session_state`. When widgets (like sliders or buttons) are updated, the script reruns top-to-bottom. Storing predictions in `st.session_state.single_prediction` ensures results persist across slider interactions.

#### Q23: How does your batch CSV prediction handle missing columns or corrupt inputs?
> **A:** The `validate_schema()` function checks for required columns, automatically derives `age_group` if only `age` is given, strips `duration` if accidentally uploaded, and provides informative banner messages without crashing.

#### Q24: What are the economic parameters used in your business simulation?
> **A:** Cost per contact attempt = $5.00; gross profit per subscribed customer = $50.00. The break-even probability for a single customer is $\frac{5}{50} = 10\%$.

#### Q25: What is the main conclusion of your project?
> **A:** Machine learning transforms marketing from inefficient spray-and-pray calling into a targeted, economically disciplined strategy. Random Forest with an operating threshold of 0.11 captures two-thirds of all potential subscribers while eliminating 81% of wasted outreach expenditure.
