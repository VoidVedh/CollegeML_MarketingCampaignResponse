"""
app.py
Production-Grade Streamlit Web Application:
Marketing Campaign Response Prediction Using Machine Learning

Features:
- Real-time customer response prediction with interactive parameter adjustments.
- Color-coded decision badge ("Will Respond" / "Will Not Respond") based on optimized threshold.
- Probability gauges and actionable marketing recommendations.
- Interactive Model Comparison tab with real performance tables and evaluation curves.
- Feature Importance & Interpretability tab with SHAP, Odds Ratios, and Tree Importance.
- Batch Prediction tab allowing CSV upload, validation, prediction scoring, and CSV export.
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

# Import IQRCapper for unpickling compatibility
from src.preprocess import (
    IQRCapper, FEATURE_COLUMNS, NUMERIC_FEATURES,
    CATEGORICAL_FEATURES, validate_schema
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
    """
    Loads saved model pipeline, preprocessor, and metrics metadata.
    """
    model_path = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
    metrics_path = os.path.join(PROJECT_ROOT, "models", "metrics.json")
    feature_path = os.path.join(PROJECT_ROOT, "models", "feature_list.json")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Please run 'python src/train.py' first.")
        
    model = joblib.load(model_path)
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    with open(feature_path, 'r') as f:
        features = json.load(f)
        
    return model, metrics, features


def predict_single_customer(model, customer_data: dict, threshold: float):
    """
    Predicts response probability and class for a single customer dictionary.
    """
    df_input = pd.DataFrame([customer_data])
    proba = model.predict_proba(df_input)[0, 1]
    prediction = int(proba >= threshold)
    return prediction, proba


# Load artifacts
try:
    best_model, metrics_meta, feature_meta = load_model_artifacts()
    opt_threshold = metrics_meta.get('optimal_threshold', 0.50)
    best_model_name = metrics_meta.get('best_model_name', 'Best Model')
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()


# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullseye.png", width=64)
    st.title("Campaign Analytics")
    st.markdown("Automated targeting engine to optimize marketing ROI and minimize wasted ad expenditure.")
    
    st.markdown("---")
    st.subheader("Model Specifications")
    st.markdown(f"**Deployed Model:** `{best_model_name}`")
    st.markdown(f"**Optimal Threshold:** `{opt_threshold:.2f}`")
    st.markdown(f"**Default Threshold:** `0.50`")
    
    st.markdown("---")
    st.subheader("Decision Threshold Controller")
    selected_threshold = st.slider(
        "Operating Threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(opt_threshold),
        step=0.01,
        help="Higher thresholds reduce False Positives (wasted ad budget); lower thresholds capture more total responders."
    )
    
    if st.button("Reset to Optimal Threshold"):
        selected_threshold = float(opt_threshold)
        
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
            "Historical Total Purchases",
            min_value=0,
            max_value=150,
            value=8,
            step=1,
            help="Total lifetime purchase transactions completed."
        )
        
    with col2:
        purchase_frequency = st.slider(
            "Purchase Frequency (Purchases/Month)",
            min_value=0.0,
            max_value=12.0,
            value=2.5,
            step=0.1,
            help="Average monthly transaction frequency."
        )
        prev_resp_choice = st.selectbox(
            "Responded to Previous Campaign?",
            options=["No", "Yes"],
            index=0,
            help="Historical indicator: did the customer accept the prior promotional campaign?"
        )
        previous_campaign_response = 1 if prev_resp_choice == "Yes" else 0
        
        website_visits = st.number_input(
            "Monthly Website Visits",
            min_value=0,
            max_value=60,
            value=7,
            step=1,
            help="Total visits to company web/mobile portals in the past 30 days."
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
    predict_btn = st.button("🚀 Predict Campaign Response", type="primary", use_container_width=True)
    
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
            
            st.markdown("---")
            st.subheader("🎯 Prediction Result & Marketing Recommendation")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.markdown("#### Outcome Classification")
                if pred_class == 1:
                    st.markdown('<div class="badge-respond">✅ WILL RESPOND</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-no-respond">❌ WILL NOT RESPOND</div>', unsafe_allow_html=True)
                    
                st.markdown(f"**Predicted Response Probability:** `{proba*100:.1f}%`")
                st.markdown(f"**Classification Threshold:** `{selected_threshold:.2f}`")
                
                # Visual probability gauge
                st.progress(float(proba))
                
            with res_col2:
                st.markdown("#### Actionable Strategy")
                if pred_class == 1:
                    recommendation = "🌟 **High-Value Target:** Allocate campaign budget to contact this customer. High probability of conversion offsets acquisition cost. Provide personalized, time-limited promotional incentives."
                else:
                    recommendation = "🛑 **Suppression Recommended:** Suppress this customer from direct outreach to eliminate wasted marketing budget ($5.00 saved per customer). Funnel into low-cost organic email nurture streams instead."
                    
                st.markdown(f'<div class="recommendation-box">{recommendation}</div>', unsafe_allow_html=True)
                
                # Key Drivers analysis for this customer
                drivers = []
                if previous_campaign_response == 1:
                    drivers.append("Strong positive history (prior campaign conversion)")
                if email_engagement >= 0.50:
                    drivers.append("High digital email responsiveness")
                if purchase_frequency >= 3.0:
                    drivers.append("Frequent recurring purchaser")
                if discount_usage >= 0.50:
                    drivers.append("Promotional/deal-sensitive customer profile")
                    
                if drivers:
                    st.markdown("**Dominant Propensity Drivers:**")
                    for d in drivers:
                        st.markdown(f"- {d}")
                else:
                    st.markdown("**Dominant Propensity Drivers:** Baseline behavioral signals indicate low engagement across promotional touchpoints.")
                    
        except Exception as err:
            st.error(f"Prediction encountered an error: {err}")


# -------------------------------------------------------------
# TAB 2: MODEL COMPARISON & PERFORMANCE
# -------------------------------------------------------------
with tab_comparison:
    st.markdown("### Comprehensive Benchmark: 6 Machine Learning Algorithms")
    st.markdown("Comparison across Accuracy, Precision, Recall, F1-Score, ROC-AUC, and False Positive Rate (FPR) on the held-out test set.")
    
    comp_csv_path = os.path.join(PROJECT_ROOT, "reports", "results_comparison.csv")
    if os.path.exists(comp_csv_path):
        results_df = pd.read_csv(comp_csv_path)
        
        # Display formatted table
        display_df = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'FPR', 'CV_F1_Mean']].copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'FPR', 'CV_F1_Mean']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.info("💡 **Why Accuracy is Misleading:** With an imbalanced target (~20% response rate), a naive model predicting 'No Response' for everyone achieves 80% accuracy but catches 0% of responders. Our winner is chosen based on F1-Score and ROC-AUC balance.")
        
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### Performance Metrics Bar Chart")
        bar_path = os.path.join(PROJECT_ROOT, "reports", "figures", "metrics_comparison_bar.png")
        if os.path.exists(bar_path):
            st.image(bar_path, use_column_width=True)
            
        st.markdown("#### Precision-Recall (PR) Curves")
        pr_path = os.path.join(PROJECT_ROOT, "reports", "figures", "pr_curves_all_models.png")
        if os.path.exists(pr_path):
            st.image(pr_path, use_column_width=True)
            
    with col_chart2:
        st.markdown("#### Receiver Operating Characteristic (ROC) Curves")
        roc_path = os.path.join(PROJECT_ROOT, "reports", "figures", "roc_curves_all_models.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_column_width=True)
            
        st.markdown("#### Confusion Matrix Grid (All 6 Models)")
        cm_path = os.path.join(PROJECT_ROOT, "reports", "figures", "confusion_matrices_grid.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_column_width=True)

    # Business Simulation comparison
    st.markdown("---")
    st.markdown("### Marketing Economics Simulation: Cost vs ROI")
    sim_csv_path = os.path.join(PROJECT_ROOT, "reports", "business_simulation.csv")
    if os.path.exists(sim_csv_path):
        sim_df = pd.read_csv(sim_csv_path)
        st.dataframe(sim_df, use_container_width=True, hide_index=True)
        
    sim_img_path = os.path.join(PROJECT_ROOT, "reports", "figures", "business_simulation_roi.png")
    if os.path.exists(sim_img_path):
        st.image(sim_img_path, use_column_width=True)


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
            st.image(tree_fi_path, use_column_width=True)
            
        st.markdown("#### Logistic Regression Odds Ratios (exp(β))")
        or_path = os.path.join(PROJECT_ROOT, "reports", "figures", "feature_importance_odds_ratios.png")
        if os.path.exists(or_path):
            st.image(or_path, use_column_width=True)
            
    with fcol2:
        st.markdown("#### Permutation Importance on Test Set")
        perm_path = os.path.join(PROJECT_ROOT, "reports", "figures", "feature_importance_permutation.png")
        if os.path.exists(perm_path):
            st.image(perm_path, use_column_width=True)
            
        st.markdown("#### SHAP Summary Plot")
        shap_path = os.path.join(PROJECT_ROOT, "reports", "figures", "shap_summary.png")
        if os.path.exists(shap_path):
            st.image(shap_path, use_column_width=True)
            
    st.markdown("---")
    st.markdown("### Profile Comparison: Responders vs Non-Responders")
    prof_csv_path = os.path.join(PROJECT_ROOT, "reports", "customer_profiles.csv")
    if os.path.exists(prof_csv_path):
        prof_df = pd.read_csv(prof_csv_path, index_col=0)
        st.dataframe(prof_df, use_container_width=True)
        st.caption("Averages and medians for predicted classes across core financial and engagement features.")


# -------------------------------------------------------------
# TAB 4: BATCH PREDICTION
# -------------------------------------------------------------
with tab_batch:
    st.markdown("### Batch Customer Scoring from CSV")
    st.markdown("Upload any CSV with the required schema to generate batch response predictions, calculated probabilities, and targeted marketing flags.")
    
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
            
            # Validate schema
            df_clean = validate_schema(df_upload, require_target=False)
            
            # Predict
            probas = best_model.predict_proba(df_clean[FEATURE_COLUMNS])[:, 1]
            preds = (probas >= selected_threshold).astype(int)
            
            df_result = df_upload.copy()
            df_result['response_probability'] = np.round(probas, 4)
            df_result['predicted_responded'] = preds
            df_result['decision'] = np.where(preds == 1, 'Target Customer', 'Suppress / Do Not Contact')
            
            st.success("✅ Batch scoring completed successfully!")
            
            # Summary metrics of the batch
            n_target = int(preds.sum())
            pct_target = (n_target / len(preds)) * 100
            
            bcol1, bcol2, bcol3 = st.columns(3)
            bcol1.metric("Total Customers Evaluated", f"{len(preds):,}")
            bcol2.metric("Recommended for Campaign", f"{n_target:,} ({pct_target:.1f}%)")
            bcol3.metric("Suppressed (Cost Saved)", f"{len(preds) - n_target:,}")
            
            st.dataframe(df_result.head(25), use_container_width=True)
            
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
