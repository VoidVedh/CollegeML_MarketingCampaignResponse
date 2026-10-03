# Comprehensive Viva Voce Defense Guide
## Case Study 157: Marketing Campaign Response Prediction Using Machine Learning

**Student:** Vedh Naik  
**Roll No.:** 150096725163  
**Cohort:** Jensen Huang  
**Project Repository:** `VoidVedh/CollegeML_MarketingCampaignResponse`  
**Dataset Source:** [Kaggle Customer Personality Analysis (`marketing_campaign.csv`)](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)  

---

## 1. 30-Second Elevator Pitch
> *"My project predicts customer response to marketing campaigns using the public Kaggle Customer Personality Analysis dataset of 2,240 customer records. Under severe 85.1% negative class imbalance, standard accuracy is misleading, so I benchmarked six machine learning algorithms evaluated on Precision-Recall AUC (PR-AUC) and False Positive Rate (FPR). To satisfy the Case Study 157 requirements without fabricating data, I engineered the eight required deployment variables directly from real demographic, transaction, and campaign history records, while strictly isolating the current campaign response to prevent target leakage. Random Forest emerged as the winning model (5-fold CV PR-AUC: 0.5109, Test ROC-AUC: 0.8171, Top-20% Customer Capture Rate: 61.19%). Finally, rather than relying on an arbitrary 0.50 cutoff, I optimized decision thresholds on training out-of-fold data to identify the profit-optimal threshold of tau = 0.13, which maximizes net campaign profit at $1,705.00 on the held-out test cohort—delivering a 245.3% ROI and saving 69.0% in wasted marketing expenditure compared to untargeted mass outreach."*

---

## 2. Core Academic & Technical Specifications

| Parameter | Project Specification | Technical Justification |
| :--- | :--- | :--- |
| **Case Study** | Case Study 157 | Official Semester V Machine Learning Curriculum |
| **Project Title** | Marketing Campaign Response Prediction Using Machine Learning | Official Assignment Problem Statement Title |
| **Dataset Source** | Kaggle Customer Personality Analysis (`marketing_campaign.csv`) | Real customer marketing attributes (income, purchases, web visits, discounts, prior campaigns) |
| **Total Cohort Size** | 2,240 records across 29 raw columns | 100% authentic Kaggle empirical customer data |
| **Target Variable** | `Response` (0 = Will Not Respond, 1 = Will Respond) | Binary classification of promotional conversion |
| **Class Distribution** | 1,906 negative (85.09%) vs. 334 positive (14.91%) | Severe class imbalance (~5.7:1 ratio) |
| **Data Partitioning** | 80% Stratified Training (1,792) / 20% Held-Out Test (448) | Test set untouched until final validation |
| **Cross-Validation** | 5-Fold Stratified K-Fold on Training Set only | Preserves class balance across all training folds |
| **Primary Metric** | PR-AUC (Average Precision) | Appropriate metric under severe class imbalance |
| **Winning Algorithm** | Random Forest (`n_estimators=50`, `max_depth=8`, `min_split=2`) | Highest CV PR-AUC (0.5109), lowest OOF FPR@Rec=70% (0.1921) |
| **Deployed Threshold** | $\tau = 0.13$ (Profit-Optimal) | Maximizes campaign net profit under contact economics |

---

## 3. The Five Core Defenses for the Examiner

### Defense 1: Zero Target Leakage Protocol
- **Examiner's Question:** *"How did you ensure that your features do not leak information about the target variable?"*
- **Textbook Defense:**
  > *"Target leakage occurs when an input feature includes information that is only available after or as a direct consequence of the target event. In our dataset, the target is `Response`, representing customer acceptance of the current campaign. The dataset also includes earlier campaign fields (`AcceptedCmp1` through `AcceptedCmp5`). When constructing `previous_campaign_response` and `email_engagement`, I strictly aggregated only campaigns 1 to 5 and excluded the current `Response` field. Furthermore, all customer tenure calculations use a fixed historical observation reference date (2014-12-31). Finally, our preprocessing ColumnTransformer is fitted exclusively on the training partition and never on the full dataset."*

---

### Defense 2: Feature Engineering of the Eight Case Study 157 Variables
- **Examiner's Question:** *"How did you map the assignment's eight deployment variables from the Kaggle dataset?"*
- **Textbook Defense:**
  > *"Every single one of the eight features was derived directly from authentic Kaggle columns using transparent, reproducible formulas without data fabrication:*
  > 1. *`age_group`: Derived by subtracting `Year_Birth` from the reference observation year 2014 and binned into `18-25`, `26-35`, `36-45`, `46-55`, `56+`.*
  > 2. *`income`: Directly mapped from annual household `Income` in USD; 24 missing values are imputed via median strictly inside the training fold.*
  > 3. *`previous_purchases`: Derived as the sum of historical channel purchases: `NumWebPurchases + NumCatalogPurchases + NumStorePurchases`.*
  > 4. *`purchase_frequency`: Computed as `previous_purchases / customer_tenure_in_months`, where tenure is elapsed months between `Dt_Customer` and 2014-12-31.*
  > 5. *`previous_campaign_response`: Binary indicator equal to 1 if the customer accepted any prior campaign (`AcceptedCmp1` to `AcceptedCmp5 >= 1`), else 0 (current target excluded).*
  > 6. *`website_visits`: Directly mapped from monthly website visits (`NumWebVisitsMonth`).*
  > 7. *`email_engagement`: Because the Kaggle dataset does not record raw email click timestamps, I derived an honest engagement proxy: `prior_campaign_acceptances / 5.0`. This is explicitly labeled in documentation and the UI as an engagement proxy.*
  > 8. *`discount_usage`: Proportion of purchases completed with discount deals: `NumDealsPurchases / max(previous_purchases, 1)`, safely clamped to `[0.0, 1.0]`."*

---

### Defense 3: Preprocessing Architecture
- **Examiner's Question:** *"What happens inside your scikit-learn preprocessing pipeline?"*
- **Textbook Defense:**
  > *"We utilize a modular scikit-learn `ColumnTransformer`:*
  > - *Numerical Pipeline (`income`, `previous_purchases`, `purchase_frequency`, `previous_campaign_response`, `website_visits`, `email_engagement`, `discount_usage`): Features pass through `SimpleImputer(strategy='median')` to handle missing income values, followed by `StandardScaler()` to standardize variances for scale-sensitive algorithms like Logistic Regression and KNN.*
  > - *Categorical Pipeline (`age_group`): Passes through `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(drop='first', categories=[['18-25', '26-35', '36-45', '46-55', '56+']])` to eliminate dummy variable trap multicollinearity.*
  > - *This transforms the 8 raw inputs into exactly 11 numeric features for model consumption, with zero data leakage between training and testing folds."*

---

### Defense 4: Defensible Model Selection
- **Examiner's Question:** *"Why did you select Random Forest over the other five algorithms?"*
- **Textbook Defense:**
  > *"Model selection was governed by a strict, pre-registered decision rule based entirely on development cross-validation without inspecting test data:*
  > 1. *Ranking by 5-fold CV PR-AUC: Random Forest achieved the highest score (0.5109 ± 0.0450), closely followed by Logistic Regression (0.5069 ± 0.0572) and Gradient Boosting (0.5002 ± 0.0551).*
  > 2. *Tie-break Evaluation: Because the difference between Random Forest and Logistic Regression (0.0040) is smaller than the cross-validation fold variation (0.0450), they are practically close. Random Forest was selected because it demonstrated a substantially lower training out-of-fold False Positive Rate at 70% recall (0.1921 vs 0.2492).*
  > 3. *Architectural Suitability: Random Forest effectively captures non-linear interactions (e.g. between income and past campaign participation) and handles skewed feature distributions through tree ensembles without requiring parametric distribution assumptions."*

---

### Defense 5: Decision Threshold Optimization & Campaign Economics
- **Examiner's Question:** *"Why shouldn't marketing teams use the default 0.50 probability threshold?"*
- **Textbook Defense:**
  > *"Under an 85.1% negative class imbalance, positive probabilities are inherently compressed. At a 0.50 threshold, the model is overly conservative, achieving only 26.9% recall and capturing just 18 out of 67 test responders ($775 net profit).*
  > *By conducting out-of-fold economic simulation on training data with assumed contact cost = $5.00 and responder gross return = $50.00 (break-even probability = 0.10):*
  > - *At Profit-Optimal Threshold ($\tau = 0.13$): We reach 48 responders (71.6% recall) and achieve maximum net profit of $1,705.00 with an ROI of 245.3%.*
  > - *Compared to Mass Outreach: Mass outreach costs $2,240.00 and generates 381 wasted calls ($1,905 wasted), yielding only $1,110.00 net profit. Threshold optimization saves 69.0% ($1,545.00) in wasted ad spend and increases net profit by +$595.00."*

---

## 4. Multi-Model Benchmark Table (Held-Out Test Set: 448 Customers)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | False Positive Rate (FPR) | Top-20% Capture | 5-Fold CV PR-AUC (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** (Winner) | **0.875** | **0.720** | **0.269** | **0.391** | **0.817** | **0.571** | **0.018** | **61.2%** | **0.5109 ± 0.0450** |
| **Logistic Regression** | 0.873 | 0.750 | 0.224 | 0.345 | 0.781 | 0.505 | 0.013 | 55.2% | 0.5069 ± 0.0572 |
| **Gradient Boosting** | 0.873 | 0.727 | 0.239 | 0.360 | 0.817 | 0.517 | 0.016 | 59.7% | 0.5002 ± 0.0551 |
| **K-Nearest Neighbors** | 0.884 | 0.759 | 0.328 | 0.458 | 0.784 | 0.561 | 0.018 | 58.2% | 0.4913 ± 0.0521 |
| **Naive Bayes** | 0.810 | 0.402 | 0.552 | 0.465 | 0.750 | 0.481 | 0.144 | 55.2% | 0.4605 ± 0.0476 |
| **Decision Tree** | 0.855 | 0.563 | 0.134 | 0.217 | 0.728 | 0.355 | 0.018 | 55.2% | 0.4013 ± 0.0458 |

---

## 5. Four-Strategy Marketing Economics Comparison

| Strategy | Decision Threshold ($\tau$) | Contacts Targeted | Total Campaign Cost | Responders Reached | Wasted Contacts (FP) | Net Profit | Marketing ROI | Cost Saved vs. Mass Outreach |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy 1: Contact Everyone** | 0.00 | 448 | $2,240.00 | 67 (100.0%) | 381 | $1,110.00 | 49.6% | 0.0% ($0.00) |
| **Strategy 2: Default ML** | 0.50 | 25 | $125.00 | 18 (26.9%) | 7 | $775.00 | 620.0% | 94.4% ($2,115.00) |
| **Strategy 3: F1-Optimal** | 0.27 | 68 | $340.00 | 38 (56.7%) | 30 | $1,560.00 | 458.8% | 84.8% ($1,900.00) |
| **Strategy 4: Profit-Optimal** | **0.13** | **139** | **$695.00** | **48 (71.6%)** | **91** | **$1,705.00** | **245.3%** | **69.0% ($1,545.00)** |

---

## 6. Answers to the Six Case Study 157 Questions

### Question 1: Can campaign responses be predicted?
**Answer:** Yes. On held-out test data, our Random Forest classifier achieves a ROC-AUC of 0.8171 and a PR-AUC of 0.5706. Sorting customer prospects by predicted probability captures 61.19% of all responders within the top 20% of customer outreach, confirming strong predictive ranking capability.

### Question 2: Which customer characteristics influence response?
**Answer:** Past promotional acceptance, annual household income, and purchase frequency are the strongest predictors. Customers with previous campaign acceptance exhibit a substantially higher response propensity (odds ratio = 1.9899). Responders also show higher median income ($65,104 vs. $50,042) and higher monthly purchase cadence.

### Question 3: Which algorithm performs best?
**Answer:** Random Forest performed best overall, achieving the highest cross-validation PR-AUC (0.5109 ± 0.0450) and test ROC-AUC (0.8171), while demonstrating the lowest out-of-fold False Positive Rate at 70% recall (0.1921).

### Question 4: Can ML reduce unnecessary marketing expenditure?
**Answer:** Yes. Traditional mass outreach wastes $1,905.00 contacting 381 non-responders. Deploying Random Forest at the profit-optimal threshold ($\tau = 0.13$) reduces total outreach spend from $2,240.00 down to $695.00 (a 69.0% savings) while increasing net campaign profit by +153.6% ($1,705.00 vs. $1,110.00).

### Question 5: How can false positives be reduced?
**Answer:** False positives are directly controlled by decision threshold tuning. Raising the threshold from the loose baseline of 0.13 to the F1-optimal cutoff of 0.27 eliminates 61 false positive contacts (from 91 down to 30), decreasing wasted calls by 67%.

### Question 6: Can the model identify potential campaign responders?
**Answer:** Yes. The cumulative gains analysis proves that contacting only the top two deciles (top 20% ranked by propensity) captures 61.19% of all responders, achieving a lift of over 3.0x compared to random contact selection.

---

## 7. Top 25 Rapid-Fire Viva Questions & Answers

1. **Q: What is the business problem of Case Study 157?**  
   *A:* Marketing outreach costs money per contact attempt. Blanket outreach wastes resources on disinterested consumers. Machine learning identifies high-propensity prospects before contact, maximizing campaign ROI.

2. **Q: What dataset are you using?**  
   *A:* The Kaggle Customer Personality Analysis dataset (`marketing_campaign.csv`), comprising 2,240 customer records across demographic, financial, purchasing, and campaign acceptance attributes.

3. **Q: What are the eight deployment features in your model?**  
   *A:* `age_group`, `income`, `previous_purchases`, `purchase_frequency`, `previous_campaign_response`, `website_visits`, `email_engagement` (proxy), and `discount_usage`.

4. **Q: How did you define the target variable?**  
   *A:* The binary column `Response`, where 1 represents customer acceptance of the promotional offer and 0 represents non-acceptance.

5. **Q: What is the class imbalance ratio in your dataset?**  
   *A:* 85.09% non-responders (1,906) to 14.91% responders (334), an imbalance ratio of approximately 5.7 to 1.

6. **Q: Why is classification accuracy an inadequate metric here?**  
   *A:* A trivial baseline that predicts 'Will Not Respond' for every customer achieves 85.09% accuracy while capturing 0% of responders and generating zero conversions.

7. **Q: What evaluation metrics did you use instead?**  
   *A:* Precision-Recall AUC (PR-AUC), ROC-AUC, Precision, Recall, F1-score, False Positive Rate (FPR), and Confusion Matrices.

8. **Q: What is the formula for False Positive Rate (FPR)?**  
   *A:* $FPR = \frac{FP}{FP + TN}$. It measures the proportion of actual non-responders who were mistakenly targeted.

9. **Q: Why does the False Positive Rate matter in marketing?**  
   *A:* Every false positive represents a wasted contact attempt that incurs marketing expenditure and risks consumer brand fatigue without producing revenue.

10. **Q: How did you calculate customer tenure without data leakage?**  
    *A:* By measuring days from customer enrollment date (`Dt_Customer`) to a fixed historical observation end date (2014-12-31), avoiding any forward-looking leakage.

11. **Q: Why is email engagement described as a proxy?**  
    *A:* The Kaggle dataset does not contain direct email open or click timestamps. To avoid fabricating data, we compute the historical promotional acceptance rate across earlier campaigns 1 to 5 as a defensible engagement proxy.

12. **Q: How did you handle missing values in Income?**  
    *A:* Using `SimpleImputer(strategy='median')` fitted strictly within the training fold during cross-validation, ensuring zero leakage into test data.

13. **Q: Why did you use `drop='first'` in OneHotEncoder?**  
    *A:* To avoid the dummy variable trap (perfect multicollinearity) by dropping the reference category (`18-25`).

14. **Q: What six machine learning algorithms did you implement?**  
    *A:* Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Naive Bayes (GaussianNB), and Gradient Boosting.

15. **Q: How were hyperparameters tuned?**  
    *A:* Using 5-fold Stratified Cross-Validation via `GridSearchCV` on the training dataset, optimizing for Average Precision (PR-AUC).

16. **Q: Why did Random Forest win over Gradient Boosting?**  
    *A:* Random Forest achieved higher CV PR-AUC (0.5109 vs. 0.5002) and a lower out-of-fold False Positive Rate at 70% recall (0.1921 vs. 0.2066).

17. **Q: How do you interpret Random Forest feature importance?**  
    *A:* Using tree-based Mean Decrease in Impurity (MDI) and permutation importance on the test set. Tree ensembles do NOT have linear coefficients.

18. **Q: What is permutation feature importance?**  
    *A:* It measures the decrease in model score (F1-score) after randomly shuffling the values of a feature, breaking its relationship with the target.

19. **Q: What is the top predictor of campaign response?**  
    *A:* Prior promotional acceptance (`previous_campaign_response`), which carries an odds ratio of 1.9899 in Logistic Regression and the highest tree MDI in Random Forest.

20. **Q: What is an odds ratio?**  
    *A:* The exponentiated coefficient ($e^{\beta}$) in Logistic Regression. An odds ratio > 1 indicates that an increase in the feature elevates the relative odds of a positive response.

21. **Q: What is the theoretical break-even probability for marketing outreach?**  
    *A:* $\text{Break-Even Probability} = \frac{\text{Unit Contact Cost}}{\text{Gross Revenue per Conversion}} = \frac{\$5.00}{\$50.00} = 0.10$. Any prospect with predicted probability > 0.10 has positive expected financial value.

22. **Q: What is the difference between F1-optimal and Profit-optimal thresholds?**  
    *A:* The F1-optimal threshold ($\tau = 0.27$) balances harmonic precision and recall. The Profit-optimal threshold ($\tau = 0.13$) aligns with unit contact economics to maximize total net campaign profit.

23. **Q: What is the Brier score?**  
    *A:* The mean squared difference between predicted probabilities and actual binary outcomes ($BS = \frac{1}{N}\sum(p_i - y_i)^2$). Lower scores indicate better probability calibration.

24. **Q: What are the primary outputs of your Streamlit application?**  
    *A:* Given the 8 customer inputs, the app outputs `WILL RESPOND` or `WILL NOT RESPOND`, the predicted probability percentage, expected contact value, and tactical recommendations.

25. **Q: What are the main limitations of your study?**  
    *A:* 
    1. Historical observational data from 2,240 records may not capture macroeconomic shifts or emerging product categories.
    2. Email engagement is an empirical proxy rather than direct real-time telemetry.
    3. Unit economics (contact cost = $5.00, conversion value = $50.00) are assumed parameters for simulation purposes and should be calibrated to specific commercial deployments.
