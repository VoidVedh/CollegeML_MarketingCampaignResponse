"""
app.py
Production-Grade Streamlit Web Application:
Marketing Campaign Response Prediction Using Machine Learning

Features:
- Real-time customer response prediction with interactive parameter adjustments.
- State persistence: prediction results remain in st.session_state across widget interactions.
- Dynamic threshold slider with on_click callback to reset to profit-optimal threshold.
- Color-coded decision badge ("Will Respond" / "Will Not Respond") based on chosen threshold.
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
    IQRCapper, FEATURE_COLUMNS, CONTINUOUS_FEATURES,
    BINARY_FEATURES, CATEGORICAL_FEATURES, validate_schema
)

# Page configuration
st.set_page_config(
    page_title="Marketing Campaign Response Predictor",
    page_icon="🎯",
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
        margin-bottom: 1.5rem;
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
            "The app runs in zero-training Demo Mode from pre-committed artifacts. "
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


# Load artifacts
try:
    best_model, metrics_meta, feature_meta = load_model_artifacts()
    # Profit-optimal threshold deployed by default
    opt_threshold = float(metrics_meta.get('profit_optimal_threshold', metrics_meta.get('optimal_threshold', 0.07)))
    f1_threshold = float(metrics_meta.get('f1_optimal_threshold', 0.30))
    best_model_name = metrics_meta.get('deployed_model_name', metrics_meta.get('best_model_name', 'Best Model'))
    cost_per_contact = float(metrics_meta.get('economic_parameters', {}).get('cost_per_contact', 5.0))
    profit_per_responder = float(metrics_meta.get('economic_parameters', {}).get('profit_per_responder', 50.0))
except Exception as e:
    st.error(f"⚠️ **Application Initialization Error:** {e}")
    st.info("💡 **Presentation Runbook / Recovery:** Run `python src/train.py` from the project root to generate and serialize all model artifacts.")
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
    st.markdown("# 🎯 Campaign Analytics")
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
        help="Lower thresholds contact more customers (profit-optimal); higher thresholds contact fewer (precision-focused)."
    )
    
    st.button("🔄 Reset to Profit-Optimal Threshold", on_click=reset_to_optimal_threshold)
        
    st.markdown("---")
    st.caption("College ML Submission Project • End-to-End Predictive Analytics")


# -------------------------------------------------------------
# MAIN CONTENT HEADER
# -------------------------------------------------------------
st.markdown('<div class="main-header">Marketing Campaign Response Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Identify high-propensity customers for promotional campaigns, cut ad spend waste, and maximize marketing ROI.</div>', unsafe_allow_html=True)

# Navigation Tabs
tab_single, tab_comparison, tab_features, tab_batch = st.tabs([
    "🎯 Single Customer Scoring",
    "📊 Algorithm Benchmark & Comparison",
    "🔍 Feature Importance & Insights",
    "📁 Batch CSV Prediction"
])


# -------------------------------------------------------------
# TAB 1: SINGLE CUSTOMER PREDICTION
# -------------------------------------------------------------
with tab_single:
    st.markdown("### Customer Profile & Behavioral Attributes")
    st.markdown("Adjust the 8 demographic and engagement variables below to simulate a customer and generate instant targeting predictions.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_group = st.selectbox(
            "Age Group",
            options=['18-25', '26-35', '36-45', '46-55', '56+'],
            index=1,
            help="Demographic age bucket of the customer."
        )
        income = st.number_input(
            "Estimated Annual Income ($)",
            min_value=10000.0,
            max_value=350000.0,
            value=55000.0,
            step=2500.0,
            format="%.0f",
            help="Annual household income (numeric, roughly $15k-$150k)."
        )
        previous_purchases = st.number_input(
            "Lifetime Previous Purchases",
            min_value=0,
            max_value=100,
            value=7,
            step=1,
            help="Total historical order count across lifetime."
        )
        
    with col2:
        purchase_frequency = st.number_input(
            "Purchase Frequency (orders/month)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.25,
            format="%.2f",
            help="Monthly order cadence velocity."
        )
        previous_campaign_response = st.radio(
            "Responded to Previous Campaign?",
            options=[0, 1],
            format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
            index=0,
            horizontal=True,
            help="Did this customer respond to the previous promotional campaign?"
        )
        website_visits = st.slider(
            "Monthly Website Visits",
            min_value=0,
            max_value=40,
            value=8,
            step=1,
            help="Number of web/app browsing sessions in the past 30 days."
        )
        
    with col3:
        email_engagement = st.slider(
            "Email Engagement Rate (0.0 - 1.0)",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.01,
            help="Proportion of promotional emails opened and clicked."
        )
        discount_usage = st.slider(
            "Discount Usage Share (0.0 - 1.0)",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.01,
            help="Proportion of past orders placed with coupon or promo discount."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Predict Campaign Response", type="primary")
    
    if predict_btn:
        customer_data = {
            'age_group': age_group,
            'income': float(income),
            'previous_purchases': int(previous_purchases),
            'purchase_frequency': float(purchase_frequency),
            'previous_campaign_response': int(previous_campaign_response),
            'website_visits': int(website_visits),
            'email_engagement': float(email_engagement),
            'discount_usage': float(discount_usage)
        }
        
        try:
            pred_class, proba = predict_single_customer(best_model, customer_data, selected_threshold)
            
            # Key Drivers analysis
            drivers = []
            if previous_campaign_response == 1:
                drivers.append("Strong positive history (prior campaign conversion - 5.5x odds multiplier)")
            if email_engagement >= 0.50:
                drivers.append("High digital email responsiveness")
            if purchase_frequency >= 2.5:
                drivers.append("Frequent recurring purchaser (high brand engagement)")
            if discount_usage >= 0.45:
                drivers.append("Promotional/deal-sensitive customer profile")
                
            st.session_state.single_prediction = {
                'pred_class': pred_class,
                'proba': proba,
                'threshold': selected_threshold,
                'drivers': drivers
            }
        except Exception as err:
            st.error(f"Prediction encountered an error: {err}")

    # Render persisted prediction from session_state
    if st.session_state.single_prediction is not None:
        p_res = st.session_state.single_prediction
        pred_class = int(p_res['proba'] >= selected_threshold)
        proba = p_res['proba']
        
        st.markdown("---")
        st.subheader("🎯 Prediction Result & Marketing Recommendation")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown("#### Outcome Classification")
            if pred_class == 1:
                st.markdown('<div class="badge-respond">✅ WILL RESPOND</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-no-respond">❌ WILL NOT RESPOND</div>', unsafe_allow_html=True)
                
            st.markdown(f"**Predicted Response Probability:** `{proba*100:.2f}%`")
            st.caption(f"Evaluated against Operating Threshold: **{selected_threshold:.2f}**")
            
            # Visual probability gauge
            st.progress(min(max(float(proba), 0.0), 1.0))
            
        with res_col2:
            st.markdown("#### Actionable Commercial Strategy")
            if pred_class == 1:
                recommendation = f"🌟 **Target Customer:** Allocate budget to contact this customer. Expected return exceeds marginal cost of ${cost_per_contact:.2f}. Provide personalized, high-relevance promotional offer."
            else:
                recommendation = f"🛑 **Suppression Recommended:** Suppress this customer from paid direct outreach (${cost_per_contact:.2f} saved per customer). Funnel into low-cost organic nurture streams instead."
                
            st.markdown(f'<div class="recommendation-box">{recommendation}</div>', unsafe_allow_html=True)
            
            drivers = p_res.get('drivers', [])
            if drivers:
                st.markdown("**Dominant Propensity Drivers:**")
                for d in drivers:
                    st.markdown(f"- {d}")
            else:
                st.markdown("**Dominant Propensity Drivers:** Baseline behavioral signals indicate lower engagement across promotional touchpoints.")


# -------------------------------------------------------------
# TAB 2: MODEL COMPARISON & PERFORMANCE
# -------------------------------------------------------------
with tab_comparison:
    st.markdown("### Comprehensive Benchmark: 6 Machine Learning Algorithms")
    st.markdown("Evaluation across 5-Fold Stratified Cross-Validation (mean ± std) and held-out test set performance.")
    
    comp_csv_path = os.path.join(PROJECT_ROOT, "reports", "results_comparison.csv")
    if os.path.exists(comp_csv_path):
        results_df = pd.read_csv(comp_csv_path)
        
        # Display formatted table
        cols_to_show = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC', 'FPR', 'FPR_at_Recall_70', 'CV_PR_AUC_Mean', 'CV_ROC_AUC_Mean']
        present_cols = [c for c in cols_to_show if c in results_df.columns]
        display_df = results_df[present_cols].copy()
        
        for col in present_cols[1:]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x))
            
        st.dataframe(display_df, hide_index=True)
        
        st.info("💡 **Defensible Model Selection:** With an imbalanced target (~20% response rate), models within 1 standard deviation of CV PR-AUC are statistically tied. The winner is selected based on CV PR-AUC, then tie-broken by lower FPR at 70% recall and model interpretability.")
        
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### Performance Metrics Bar Chart")
        bar_path = os.path.join(PROJECT_ROOT, "reports", "figures", "metrics_comparison_bar.png")
        if os.path.exists(bar_path):
            st.image(bar_path, width="stretch")
            
        st.markdown("#### Precision-Recall (PR) Curves")
        pr_path = os.path.join(PROJECT_ROOT, "reports", "figures", "pr_curves_all_models.png")
        if os.path.exists(pr_path):
            st.image(pr_path, width="stretch")
            
    with col_chart2:
        st.markdown("#### Receiver Operating Characteristic (ROC) Curves")
        roc_path = os.path.join(PROJECT_ROOT, "reports", "figures", "roc_curves_all_models.png")
        if os.path.exists(roc_path):
            st.image(roc_path, width="stretch")
            
        st.markdown("#### Confusion Matrix Grid (All 6 Models)")
        cm_path = os.path.join(PROJECT_ROOT, "reports", "figures", "confusion_matrices_grid.png")
        if os.path.exists(cm_path):
            st.image(cm_path, width="stretch")

    # Business Simulation comparison
    st.markdown("---")
    st.markdown("### Marketing Economics Simulation: Cost vs Net Profit & ROI")
    sim_csv_path = os.path.join(PROJECT_ROOT, "reports", "business_simulation.csv")
    if os.path.exists(sim_csv_path):
        sim_df = pd.read_csv(sim_csv_path)
        st.dataframe(sim_df, hide_index=True)
        
    sim_img_path = os.path.join(PROJECT_ROOT, "reports", "figures", "business_simulation_roi.png")
    if os.path.exists(sim_img_path):
        st.image(sim_img_path, width="stretch")


# -------------------------------------------------------------
# TAB 3: FEATURE IMPORTANCE & INSIGHTS
# -------------------------------------------------------------
with tab_features:
    st.markdown("### What Drives a Customer to Respond?")
    st.markdown("Interpretability insights derived from Tree Impurity, Permutation Importance, Logistic Odds Ratios, and SHAP explainability.")
    
    fcol1, fcol2 = st.columns(2)
    
    with fcol1:
        st.markdown("#### Tree-Based MDI Feature Importance")
        tree_fi_path = os.path.join(PROJECT_ROOT, "reports", "figures", "feature_importance_tree.png")
        if os.path.exists(tree_fi_path):
            st.image(tree_fi_path, width="stretch")
            
        st.markdown("#### Logistic Regression Odds Ratios (exp(β))")
        or_path = os.path.join(PROJECT_ROOT, "reports", "figures", "feature_importance_odds_ratios.png")
        if os.path.exists(or_path):
            st.image(or_path, width="stretch")
            
    with fcol2:
        st.markdown("#### Permutation Importance on Test Set")
        perm_path = os.path.join(PROJECT_ROOT, "reports", "figures", "feature_importance_permutation.png")
        if os.path.exists(perm_path):
            st.image(perm_path, width="stretch")
            
        st.markdown("#### SHAP Summary Plot")
        shap_path = os.path.join(PROJECT_ROOT, "reports", "figures", "shap_summary.png")
        if os.path.exists(shap_path):
            st.image(shap_path, width="stretch")
            
    st.markdown("---")
    st.markdown("### Profile Comparison: Responders vs Non-Responders (Actual & Predicted)")
    prof_csv_path = os.path.join(PROJECT_ROOT, "reports", "customer_profiles.csv")
    if os.path.exists(prof_csv_path):
        prof_df = pd.read_csv(prof_csv_path, header=[0, 1], index_col=0)
        st.dataframe(prof_df)
        st.caption("Averages and medians for actual and predicted classes across core financial and engagement features.")


# -------------------------------------------------------------
# TAB 4: BATCH PREDICTION
# -------------------------------------------------------------
with tab_batch:
    st.markdown("### Batch Customer Scoring from CSV")
    st.markdown("Upload any customer CSV to generate batch predictions, response probabilities, and targeted marketing flags.")
    
    st.markdown("""
    **Required Columns:** `age_group`, `income`, `previous_purchases`, `purchase_frequency`, `previous_campaign_response`, `website_visits`, `email_engagement`, `discount_usage`
    """)
    
    # Download sample template
    sample_data = pd.DataFrame([
        {
            'age_group': '26-35',
            'income': 58000,
            'previous_purchases': 10,
            'purchase_frequency': 3.2,
            'previous_campaign_response': 1,
            'website_visits': 12,
            'email_engagement': 0.75,
            'discount_usage': 0.60
        },
        {
            'age_group': '18-25',
            'income': 24000,
            'previous_purchases': 2,
            'purchase_frequency': 0.5,
            'previous_campaign_response': 0,
            'website_visits': 3,
            'email_engagement': 0.15,
            'discount_usage': 0.20
        },
        {
            'age_group': '46-55',
            'income': 88000,
            'previous_purchases': 15,
            'purchase_frequency': 1.8,
            'previous_campaign_response': 0,
            'website_visits': 6,
            'email_engagement': 0.40,
            'discount_usage': 0.45
        }
    ])
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample_data.to_csv(index=False),
        file_name="sample_campaign_input.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Uploaded file contains **{len(df_upload)} records**.")
            
            # Validate schema without converting NaN in age_group to string
            df_clean, audit_report = validate_schema(df_upload, require_target=False)
            
            # Display audit warnings if anomalies exist
            has_warnings = False
            warning_messages = []
            
            if audit_report['missing_rows']:
                has_warnings = True
                miss_details = ", ".join([f"{col}: {cnt} rows" for col, cnt in audit_report['missing_rows'].items()])
                warning_messages.append(f"**Missing values detected and imputed:** {miss_details} (handled by median/mode imputer).")
                
            if audit_report['out_of_range_rows']:
                has_warnings = True
                range_details = ", ".join([f"{col}: {cnt} rows" for col, cnt in audit_report['out_of_range_rows'].items()])
                warning_messages.append(f"**Out-of-range values clipped to valid bounds:** {range_details}.")
                
            if audit_report['unknown_categories']:
                has_warnings = True
                cat_details = ", ".join([f"{col}: {cnt} rows" for col, cnt in audit_report['unknown_categories'].items()])
                warning_messages.append(f"**Unknown categorical labels imputed with mode:** {cat_details}.")
                
            if has_warnings:
                st.warning("⚠️ **Data Quality Audit Report:**\n\n" + "\n\n".join(warning_messages))
            
            # Predict
            probas = best_model.predict_proba(df_clean[FEATURE_COLUMNS])[:, 1]
            preds = (probas >= selected_threshold).astype(int)
            
            df_result = df_upload.copy()
            df_result['response_probability'] = np.round(probas, 4)
            df_result['predicted_responded'] = preds
            df_result['decision'] = np.where(preds == 1, 'Target Customer', 'Suppress / Do Not Contact')
            
            st.success(f"✅ Batch scoring completed successfully using decision threshold {selected_threshold:.2f}!")
            
            # Summary metrics of the batch
            n_target = int(preds.sum())
            pct_target = (n_target / len(preds)) * 100 if len(preds) > 0 else 0
            cost_total = n_target * cost_per_contact
            cost_saved = (len(preds) - n_target) * cost_per_contact
            
            bcol1, bcol2, bcol3, bcol4 = st.columns(4)
            bcol1.metric("Evaluated", f"{len(preds):,}")
            bcol2.metric("Targeted", f"{n_target:,} ({pct_target:.1f}%)")
            bcol3.metric("Campaign Spend", f"${cost_total:,.2f}")
            bcol4.metric("Ad Spend Saved", f"${cost_saved:,.2f}")
            
            st.dataframe(df_result.head(25))
            
            # Download scored CSV
            csv_export = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored CSV with Predictions",
                data=csv_export,
                file_name="campaign_scored_predictions.csv",
                mime="text/csv"
            )
            
        except Exception as ex:
            st.error(f"Error processing uploaded CSV: {ex}")
