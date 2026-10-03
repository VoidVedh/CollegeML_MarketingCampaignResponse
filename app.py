"""
app.py
Production-Grade Streamlit Web Application:
Marketing Campaign Response Prediction Using Machine Learning
Case Study 157 — Kaggle Marketing Dataset (Bank Term Deposit Subscription)

Student: Vedh Naik
Cohort: Jensen Huang
Roll No.: 150096725163

Features:
- Real-time customer response prediction with interactive Kaggle parameters.
- State persistence: prediction results remain in st.session_state across widget interactions.
- Dynamic threshold slider with on_click callback to reset to profit-optimal threshold.
- Color-coded decision badge ("WILL RESPOND" / "WILL NOT RESPOND") based on chosen threshold.
- Probability gauges, unit economics loaded from metrics.json, and actionable recommendations.
- Interactive Model Comparison tab with real performance tables and evaluation curves.
- Feature Importance & Interpretability tab with SHAP, Odds Ratios, and Tree Importance.
- Batch Prediction tab with schema validation, NaN preservation, audit warnings, and CSV export.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is accessible
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import IQRCapper and preprocessing functions
from src.preprocess import (
    IQRCapper, FEATURE_COLUMNS, NUMERIC_FEATURES,
    CATEGORICAL_FEATURES, VALID_CATEGORIES, derive_age_group, validate_schema
)

# Page configuration
st.set_page_config(
    page_title="Kaggle Marketing Campaign Response Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .badge-author {
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #334155;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .badge-leakage {
        background-color: #FEF3C7;
        border: 1px solid #F59E0B;
        color: #92400E;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-respond {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 1.25rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #86EFAC;
    }
    .badge-no-respond {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 1.25rem;
        font-weight: 700;
        display: inline-block;
        border: 1px solid #FCA5A5;
    }
    .recommendation-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.85rem 1.2rem;
        border-radius: 4px;
        color: #1E3A8A;
        font-size: 1.0rem;
        margin-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """Loads and caches the deployed ML pipeline and metadata."""
    model_path = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
    metrics_path = os.path.join(PROJECT_ROOT, "models", "metrics.json")
    feature_path = os.path.join(PROJECT_ROOT, "models", "feature_list.json")
    
    required_files = [
        ("Model pipeline ('models/best_model.joblib')", model_path),
        ("Evaluation metrics ('models/metrics.json')", metrics_path),
        ("Feature schema ('models/feature_list.json')", feature_path)
    ]
    missing = [desc for desc, path in required_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing required model artifact(s): {', '.join(missing)}. "
            "Please ensure artifacts are present or run 'python src/train.py' to generate them."
        )
        
    model = joblib.load(model_path)
    
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
    features = {}
    if os.path.exists(feature_path):
        with open(feature_path, 'r') as f:
            features = json.load(f)
            
    return model, metrics, features


def predict_single_customer(model, customer_data: dict, threshold: float):
    """Predicts response probability and class for a single customer dictionary."""
    df_input = pd.DataFrame([customer_data])
    proba = float(model.predict_proba(df_input)[0, 1])
    prediction = int(proba >= threshold)
    return prediction, proba


def compute_model_feature_contributions(model, customer_data: dict) -> List[Dict[str, Any]]:
    """
    Computes feature contributions for transparent model interpretability.
    Supports both linear models (coef * val) and tree-based models (importance * val).
    """
    try:
        pipeline = model
        if hasattr(model, 'estimator'):
            base = model.estimator
            if hasattr(base, 'estimator'):
                pipeline = base.estimator
            else:
                pipeline = base
                
        if not hasattr(pipeline, 'named_steps'):
            return []
            
        preprocessor = pipeline.named_steps.get('preprocessor')
        classifier = pipeline.named_steps.get('classifier')
        
        if preprocessor is None or classifier is None:
            return []
            
        df_in = pd.DataFrame([customer_data])
        X_trans = preprocessor.transform(df_in)
        feat_names = list(preprocessor.get_feature_names_out())
        
        results = []
        if hasattr(classifier, 'coef_'):
            coefs = classifier.coef_[0]
            contributions = X_trans[0] * coefs
            for name, val, coef, contrib in zip(feat_names, X_trans[0], coefs, contributions):
                results.append({
                    'feature': name,
                    'transformed_val': float(val),
                    'weight': float(coef),
                    'contribution': float(contrib)
                })
        elif hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            contributions = np.abs(X_trans[0]) * importances
            for name, val, imp, contrib in zip(feat_names, X_trans[0], importances, contributions):
                results.append({
                    'feature': name,
                    'transformed_val': float(val),
                    'weight': float(imp),
                    'contribution': float(contrib)
                })
        else:
            return []
            
        results.sort(key=lambda x: abs(x['contribution']), reverse=True)
        return results[:10]
    except Exception:
        return []


# Load artifacts
try:
    best_model, metrics_meta, feature_meta = load_model_artifacts()
    opt_threshold = float(metrics_meta.get('profit_optimal_threshold', metrics_meta.get('optimal_threshold', 0.09)))
    f1_threshold = float(metrics_meta.get('f1_optimal_threshold', 0.22))
    best_model_name = metrics_meta.get('deployed_model_name', metrics_meta.get('best_model_name', 'Random Forest'))
    cost_per_contact = float(metrics_meta.get('economic_parameters', {}).get('cost_per_contact', 5.0))
    profit_per_responder = float(metrics_meta.get('economic_parameters', {}).get('profit_per_responder', 50.0))
except Exception as e:
    st.error(f"Application Initialization Error: {e}")
    st.info("Run `python src/train.py` from the project root to generate and serialize all model artifacts.")
    st.stop()


# Session State Initialization
if "threshold_slider" not in st.session_state:
    st.session_state.threshold_slider = opt_threshold

if "single_prediction" not in st.session_state:
    st.session_state.single_prediction = None


def reset_to_optimal_threshold():
    """Callback to reset threshold slider to optimal value."""
    st.session_state.threshold_slider = opt_threshold


# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("# Campaign Analytics")
    st.markdown("**Case Study 157:** Marketing Campaign Response Prediction")
    st.markdown("Automated targeting engine to optimize marketing ROI and minimize wasted ad expenditure.")
    
    st.markdown("---")
    st.subheader("Model Specifications")
    st.markdown(f"**Deployed Model:** `{best_model_name}`")
    st.markdown(f"**Profit-Optimal Threshold:** `{opt_threshold:.2f}` (Default)")
    st.markdown(f"**F1-Optimal Threshold:** `{f1_threshold:.2f}`")
    st.markdown(f"**Standard Default:** `0.50`")
    st.markdown(f"**Unit Contact Cost:** `${cost_per_contact:.2f}`")
    st.markdown(f"**Profit per Responder:** `${profit_per_responder:.2f}`")
    
    st.markdown("---")
    st.subheader("Decision Threshold Controller")
    selected_threshold = st.slider(
        "Operating Threshold",
        min_value=0.01,
        max_value=0.95,
        step=0.01,
        key="threshold_slider",
        help="Adjusting threshold trades off False Positives (wasted ad touches) against False Negatives (missed subscribers)."
    )
    
    st.button("Reset to Profit-Optimal Threshold", on_click=reset_to_optimal_threshold, help="Resets the operating threshold to the simulated profit-maximizing threshold.")
    
    st.markdown("---")
    st.subheader("Student & Project Details")
    st.markdown("**Name:** Vedh Naik")
    st.markdown("**Cohort:** Jensen Huang")
    st.markdown("**Roll No.:** `150096725163`")
    st.markdown("**Dataset:** [Kaggle Marketing Dataset](https://www.kaggle.com/competitions/marketing-dataset/data)")
    
    colab_link = "https://colab.research.google.com/github/VoidVedh/CollegeML_MarketingCampaignResponse/blob/main/notebooks/analysis.ipynb"
    st.markdown(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_link})")


# -------------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------------
st.markdown('<div class="main-header">Kaggle Marketing Campaign Response Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Case Study 157 — Bank Term Deposit Subscription Prediction | '
    'Machine Learning Pipeline with Zero Call-Duration Leakage</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="badge-author"><b>Vedh Naik</b> | Cohort: <b>Jensen Huang</b> | Roll No: <b>150096725163</b></div> '
    '<div class="badge-leakage"><b>Target Leakage Protection:</b> Call duration is strictly excluded</div>',
    unsafe_allow_html=True
)

with st.expander("Case Study 157 Problem Statement & Feature Mapping", expanded=False):
    st.markdown("""
    ### Case Study 157: Marketing Campaign Response Prediction
    **Objective:** Build and compare 6 classification algorithms to identify customers who are genuinely likely to respond to a campaign, minimizing marketing costs and reducing false positives.

    #### Required Algorithms & Evaluation
    - **6 Algorithms:** Logistic Regression, KNN, Decision Tree, Random Forest, Naive Bayes, Gradient Boosting.
    - **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix, and False-Positive Rate ($FPR = \\frac{FP}{FP + TN}$).
    - **Prediction Output:** `Will Respond` / `Will Not Respond`

    #### College Case Study vs. Kaggle Marketing Dataset Feature Mapping
    The college case study description lists 8 conceptual inputs. The authentic Kaggle dataset provides genuine customer, contact, and economic variables without fabricating synthetic fields:

    | Case Study Concept | Kaggle Equivalent Field | Status / Implementation |
    | :--- | :--- | :--- |
    | **Age group** | Derived from `age` (`18-25`, `26-35`, `36-45`, `46-55`, `56+`) | Fully mapped & reproducible |
    | **Previous campaign response** | `poutcome` (`success`, `failure`, `nonexistent`), `previous`, `pdays` | Fully mapped |
    | **Income** | Not in Kaggle dataset (macroeconomic proxy: `emp.var.rate`, `euribor3m`) | Defensible economic proxy |
    | **Previous purchases** | Not applicable (Banking term deposit subscription) | Not available |
    | **Purchase frequency** | Not applicable (Direct banking campaign) | Not available |
    | **Website visits** | Not applicable (Direct telemarketing campaign) | Not available |
    | **Email engagement** | Not applicable (Direct telephone / cellular campaign) | Not available |
    | **Discount usage** | Not applicable (Deposit interest rate yield product) | Not available |
    """)

tab_single, tab_benchmark, tab_importance, tab_batch = st.tabs([
    "Single Client Prediction",
    "Algorithm Benchmark & Comparison",
    "Feature Importance & Insights",
    "Batch CSV Prediction"
])


# -------------------------------------------------------------
# TAB 1: SINGLE CLIENT PREDICTION
# -------------------------------------------------------------
with tab_single:
    st.markdown("### Client Profile & Campaign Context")
    st.markdown("Enter client demographic, contact campaign, and macroeconomic indicators to generate a real-time propensity score and commercial recommendation.")
    
    with st.expander("1. Client Demographics & Financial Status", expanded=True):
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            age = st.slider("Client Age (years)", min_value=18, max_value=95, value=35, step=1, help="Age in years. Used to derive standard age groups.")
            job = st.selectbox("Job Category", options=VALID_CATEGORIES["job"], index=0, help="Client's occupation.")
            marital = st.selectbox("Marital Status", options=VALID_CATEGORIES["marital"], index=0)
        with d_col2:
            education = st.selectbox("Education Level", options=VALID_CATEGORIES["education"], index=0)
            default = st.selectbox("Credit Default History", options=VALID_CATEGORIES["default"], index=0, help="Has credit in default?")
        with d_col3:
            housing = st.selectbox("Housing Loan", options=VALID_CATEGORIES["housing"], index=0, help="Has housing loan?")
            loan = st.selectbox("Personal Loan", options=VALID_CATEGORIES["loan"], index=0, help="Has personal loan?")

    with st.expander("2. Campaign & Contact Interaction History", expanded=True):
        c_col1, c_col2, c_col3 = st.columns(3)
        with c_col1:
            contact = st.selectbox("Contact Communication Channel", options=VALID_CATEGORIES["contact"], index=0)
            month = st.selectbox("Last Contact Month", options=VALID_CATEGORIES["month"], index=0)
            day_of_week = st.selectbox("Last Contact Day of Week", options=VALID_CATEGORIES["day_of_week"], index=1)
        with c_col2:
            campaign = st.number_input("Contacts in Current Campaign", min_value=1, max_value=50, value=2, step=1, help="Number of contacts performed during this campaign.")
            previous = st.number_input("Previous Contacts Count", min_value=0, max_value=10, value=0, step=1, help="Number of contacts performed before this campaign.")
        with c_col3:
            pdays = st.number_input("Days Since Previous Contact (pdays)", min_value=0, max_value=999, value=999, step=1, help="999 means client was not previously contacted.")
            poutcome = st.selectbox("Previous Campaign Outcome", options=VALID_CATEGORIES["poutcome"], index=0, help="Outcome of the previous marketing campaign.")

    with st.expander("3. Macroeconomic Climate Indicators", expanded=True):
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            emp_var_rate = st.number_input("Employment Variation Rate (emp.var.rate)", min_value=-3.4, max_value=1.4, value=1.1, step=0.1, format="%.2f", help="Quarterly economic indicator.")
            cons_price_idx = st.number_input("Consumer Price Index (cons.price.idx)", min_value=92.0, max_value=95.0, value=93.994, step=0.001, format="%.3f")
        with m_col2:
            cons_conf_idx = st.number_input("Consumer Confidence Index (cons.conf.idx)", min_value=-55.0, max_value=-25.0, value=-36.4, step=0.1, format="%.1f")
            euribor3m = st.number_input("Euribor 3-Month Rate (euribor3m)", min_value=0.5, max_value=5.5, value=4.857, step=0.01, format="%.3f", help="Daily 3-month Euribor interbank benchmark rate.")
        with m_col3:
            nr_employed = st.number_input("Number of Employees (nr.employed)", min_value=4900.0, max_value=5300.0, value=5191.0, step=1.0, format="%.1f", help="Quarterly employee benchmark count in thousands.")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Predict Term Deposit Subscription", type="primary")

    if predict_btn:
        # Automatically derive age_group from age
        age_group_val = derive_age_group(pd.Series([age])).iloc[0]
        
        customer_data = {
            'age': float(age),
            'campaign': float(campaign),
            'pdays': float(pdays),
            'previous': float(previous),
            'emp.var.rate': float(emp_var_rate),
            'cons.price.idx': float(cons_price_idx),
            'cons.conf.idx': float(cons_conf_idx),
            'euribor3m': float(euribor3m),
            'nr.employed': float(nr_employed),
            'job': str(job),
            'marital': str(marital),
            'education': str(education),
            'default': str(default),
            'housing': str(housing),
            'loan': str(loan),
            'contact': str(contact),
            'month': str(month),
            'day_of_week': str(day_of_week),
            'poutcome': str(poutcome),
            'age_group': str(age_group_val)
        }
        
        try:
            pred_class, proba = predict_single_customer(best_model, customer_data, selected_threshold)
            model_contributions = compute_model_feature_contributions(best_model, customer_data)
            
            characteristics = []
            if poutcome == "success":
                characteristics.append("Client previously subscribed to a bank term deposit (poutcome='success').")
            elif poutcome == "failure":
                characteristics.append("Client previously contacted but did not subscribe (poutcome='failure').")
            if pdays < 999:
                characteristics.append(f"Recently contacted {pdays} days ago.")
            if euribor3m < 2.0:
                characteristics.append(f"Low interest rate environment (Euribor: {euribor3m:.2f}%), elevating deposit attractiveness.")
            if job in ["retired", "student"]:
                characteristics.append(f"Client in demographic segment with historically high conversion propensity ('{job}').")
                
            st.session_state.single_prediction = {
                'pred_class': pred_class,
                'proba': proba,
                'threshold': selected_threshold,
                'contributions': model_contributions,
                'characteristics': characteristics
            }
        except Exception as err:
            st.error(f"Prediction encountered an error: {err}")

    # Render persisted prediction from session_state
    if st.session_state.single_prediction is not None:
        p_res = st.session_state.single_prediction
        proba = p_res['proba']
        pred_class = int(proba >= selected_threshold)
        
        break_even_p = cost_per_contact / profit_per_responder
        exp_val = proba * profit_per_responder - cost_per_contact
        
        st.markdown("---")
        st.subheader("Prediction Result & Commercial Recommendation")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown("#### Outcome Classification")
            if pred_class == 1:
                st.markdown('<div class="badge-respond">WILL RESPOND</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-no-respond">WILL NOT RESPOND</div>', unsafe_allow_html=True)
                
            st.markdown(f"**Predicted Subscription Probability:** `{proba*100:.2f}%`")
            st.caption(f"Operating Threshold: **{selected_threshold:.2f}** | Theoretical Break-Even: **{break_even_p*100:.1f}%**")
            
            st.progress(min(max(float(proba), 0.0), 1.0))
            
            st.markdown(f"""
            **Unit Economic Simulation:**
            - Contact Cost: `${cost_per_contact:.2f}`
            - Gross Profit per Responder: `${profit_per_responder:.2f}`
            - Individual Expected Net Value: **${exp_val:+.2f}**
            """)
            
        with res_col2:
            st.markdown("#### Actionable Strategy Recommendation")
            if pred_class == 1:
                if proba >= break_even_p:
                    econ_statement = (
                        f"Response probability ({proba*100:.1f}%) exceeds individual break-even ({break_even_p*100:.1f}%). "
                        f"Expected net value per contact attempt is positive (+${exp_val:.2f})."
                    )
                else:
                    econ_statement = (
                        f"Response probability ({proba*100:.1f}%) meets the operating threshold ({selected_threshold:.2f}) "
                        f"even though individual break-even is {break_even_p*100:.1f}%. "
                        "Under portfolio targeting, capturing this volume maximizes overall campaign profit."
                    )
                recommendation = (
                    f"**Target Client:** Propensity ({proba*100:.1f}%) meets or exceeds the operating threshold ({selected_threshold:.2f}). "
                    f"{econ_statement} Prioritize outreach with tailored term deposit product offerings."
                )
            else:
                recommendation = (
                    f"**Suppress Contact:** Predicted probability ({proba*100:.1f}%) falls below operating threshold ({selected_threshold:.2f}). "
                    f"Expected net return is negative (${exp_val:+.2f}). "
                    "Suppress direct telemarketing contact to conserve budget and prevent client fatigue."
                )
                
            st.markdown(f'<div class="recommendation-box">{recommendation}</div>', unsafe_allow_html=True)
            
            if p_res['characteristics']:
                st.markdown("**Key Behavioral Context:**")
                for ch in p_res['characteristics']:
                    st.markdown(f"- {ch}")
                    
        # Feature contributions
        if p_res.get('contributions'):
            st.markdown("#### Top Model Feature Drivers for This Client")
            contrib_df = pd.DataFrame(p_res['contributions'])
            st.dataframe(contrib_df, use_container_width=True)


# -------------------------------------------------------------
# TAB 2: BENCHMARK & COMPARISON
# -------------------------------------------------------------
with tab_benchmark:
    st.markdown("### 6-Algorithm Comparative Benchmark (Held-Out Test Set: 8,238 Clients)")
    st.markdown(
        "All six classification algorithms were tuned strictly via 5-fold Stratified Cross-Validation on the development set. "
        "The held-out test set was evaluated once."
    )
    
    comp_csv_path = os.path.join(PROJECT_ROOT, "reports", "results_comparison.csv")
    if os.path.exists(comp_csv_path):
        benchmark_df = pd.read_csv(comp_csv_path)
        st.dataframe(
            benchmark_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC', 'FPR', 'FPR_at_Recall_70', 'Top20_Capture_Rate', 'CV_PR_AUC_Mean', 'CV_PR_AUC_Std']],
            use_container_width=True
        )
        
    st.markdown("---")
    st.subheader("Performance Visualizations")
    
    b_col1, b_col2 = st.columns(2)
    fig_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    
    with b_col1:
        roc_img = os.path.join(fig_dir, "roc_curves_all_models.png")
        if os.path.exists(roc_img):
            st.image(roc_img, caption="Receiver Operating Characteristic (ROC) Curves", use_container_width=True)
            
    with b_col2:
        pr_img = os.path.join(fig_dir, "pr_curves_all_models.png")
        if os.path.exists(pr_img):
            st.image(pr_img, caption="Precision-Recall (PR) Curves", use_container_width=True)
            
    st.markdown("---")
    bar_img = os.path.join(fig_dir, "metrics_comparison_bar.png")
    if os.path.exists(bar_img):
        st.image(bar_img, caption="Comparative Metric Overview Across All 6 Algorithms", use_container_width=True)
        
    cm_img = os.path.join(fig_dir, "confusion_matrices_grid.png")
    if os.path.exists(cm_img):
        st.image(cm_img, caption="Confusion Matrix Grid Evaluated at Each Model's Optimal Threshold", use_container_width=True)


# -------------------------------------------------------------
# TAB 3: FEATURE IMPORTANCE & INSIGHTS
# -------------------------------------------------------------
with tab_importance:
    st.markdown("### Model Explainability & Key Drivers")
    st.markdown(
        "Identifying which client, campaign, and macroeconomic factors influence bank term deposit subscription propensity. "
        "*Note: Feature importance indicates predictive association and does not prove causal intervention effects.*"
    )
    
    i_col1, i_col2 = st.columns(2)
    
    with i_col1:
        tree_fi = os.path.join(fig_dir, "feature_importance_tree.png")
        if os.path.exists(tree_fi):
            st.image(tree_fi, caption="Tree-Based Feature Importances (Top 20 MDI)", use_container_width=True)
            
        shap_img = os.path.join(fig_dir, "shap_summary.png")
        if os.path.exists(shap_img):
            st.image(shap_img, caption="SHAP Summary Plot (Top 15 Drivers)", use_container_width=True)
            
    with i_col2:
        perm_fi = os.path.join(fig_dir, "feature_importance_permutation.png")
        if os.path.exists(perm_fi):
            st.image(perm_fi, caption="Permutation Feature Importance on Test Set", use_container_width=True)
            
        or_img = os.path.join(fig_dir, "feature_importance_odds_ratios.png")
        if os.path.exists(or_img):
            st.image(or_img, caption="Logistic Regression Odds Ratios (exp(β))", use_container_width=True)
            
    st.markdown("---")
    gains_img = os.path.join(fig_dir, "cumulative_gains_lift.png")
    if os.path.exists(gains_img):
        st.image(gains_img, caption="Cumulative Gains and Decile Lift Charts", use_container_width=True)


# -------------------------------------------------------------
# TAB 4: BATCH CSV PREDICTION
# -------------------------------------------------------------
with tab_batch:
    st.markdown("### Batch Client Propensity Scoring")
    st.markdown("Upload a CSV file containing client records to score their term deposit subscription probabilities in bulk.")
    
    sample_data = pd.DataFrame([
        {
            'age': 35, 'job': 'admin.', 'marital': 'married', 'education': 'university.degree',
            'default': 'no', 'housing': 'yes', 'loan': 'no', 'contact': 'cellular',
            'month': 'may', 'day_of_week': 'mon', 'campaign': 2, 'pdays': 999,
            'previous': 0, 'poutcome': 'nonexistent', 'emp.var.rate': 1.1,
            'cons.price.idx': 93.994, 'cons.conf.idx': -36.4, 'euribor3m': 4.857,
            'nr.employed': 5191.0
        },
        {
            'age': 28, 'job': 'student', 'marital': 'single', 'education': 'high.school',
            'default': 'no', 'housing': 'no', 'loan': 'no', 'contact': 'cellular',
            'month': 'sep', 'day_of_week': 'wed', 'campaign': 1, 'pdays': 6,
            'previous': 2, 'poutcome': 'success', 'emp.var.rate': -1.8,
            'cons.price.idx': 92.893, 'cons.conf.idx': -46.2, 'euribor3m': 1.299,
            'nr.employed': 5099.1
        },
        {
            'age': 55, 'job': 'retired', 'marital': 'married', 'education': 'basic.4y',
            'default': 'no', 'housing': 'yes', 'loan': 'no', 'contact': 'telephone',
            'month': 'aug', 'day_of_week': 'fri', 'campaign': 3, 'pdays': 999,
            'previous': 0, 'poutcome': 'nonexistent', 'emp.var.rate': 1.4,
            'cons.price.idx': 93.444, 'cons.conf.idx': -36.1, 'euribor3m': 4.963,
            'nr.employed': 5228.1
        }
    ])
    st.download_button(
        label="Download Sample CSV Template",
        data=sample_data.to_csv(index=False),
        file_name="sample_kaggle_campaign_input.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload Client CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            if hasattr(uploaded_file, 'size') and uploaded_file.size == 0:
                st.error("Uploaded CSV is empty.")
                st.stop()

            try:
                df_upload = pd.read_csv(uploaded_file)
            except pd.errors.EmptyDataError:
                st.error("Uploaded CSV is empty.")
                st.stop()

            if df_upload.empty:
                st.error("Uploaded CSV is empty.")
                st.stop()

            st.write(f"Uploaded file contains **{len(df_upload)} records**.")
            
            is_valid, warnings_or_errors, df_clean = validate_schema(df_upload)
            
            if not is_valid:
                st.error("Schema Validation Failed:\n- " + "\n- ".join(warnings_or_errors))
                st.stop()
                
            if warnings_or_errors:
                st.info("Processing Notes:\n- " + "\n- ".join(warnings_or_errors))
            
            # Predict
            probas = best_model.predict_proba(df_clean[FEATURE_COLUMNS])[:, 1]
            preds = (probas >= selected_threshold).astype(int)
            
            df_result = df_upload.copy()
            df_result['subscription_probability'] = np.round(probas, 4)
            df_result['predicted_subscribe'] = preds
            df_result['decision'] = np.where(preds == 1, 'Target Client (Will Respond)', 'Suppress / Do Not Contact')
            
            st.success(f"Batch scoring completed successfully using decision threshold {selected_threshold:.2f}!")
            
            n_target = int(preds.sum())
            pct_target = (n_target / len(preds)) * 100 if len(preds) > 0 else 0
            cost_total = n_target * cost_per_contact
            cost_saved = (len(preds) - n_target) * cost_per_contact
            
            bcol1, bcol2, bcol3, bcol4 = st.columns(4)
            bcol1.metric("Evaluated", f"{len(preds):,}")
            bcol2.metric("Targeted", f"{n_target:,} ({pct_target:.1f}%)")
            bcol3.metric("Campaign Spend", f"${cost_total:,.2f}")
            bcol4.metric("Ad Spend Saved", f"${cost_saved:,.2f}")
            
            st.dataframe(df_result.head(25), use_container_width=True)
            
            csv_export = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Scored CSV with Predictions",
                data=csv_export,
                file_name="kaggle_scored_predictions.csv",
                mime="text/csv"
            )
            
        except Exception as ex:
            st.error(f"Error processing uploaded CSV: {ex}")
