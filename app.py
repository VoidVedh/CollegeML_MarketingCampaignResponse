"""
app.py
Production-Grade Streamlit Web Application:
Case Study 157: Marketing Campaign Response Prediction Using Machine Learning
Dataset: Kaggle Customer Personality Analysis (marketing_campaign.csv)

Student: Vedh Naik
Roll No.: 150096725163
Cohort: Jensen Huang

Features:
- Real-time customer response prediction across the 8 Case Study 157 variables.
- State persistence: prediction results remain in st.session_state across widget interactions.
- Sidebar decision threshold slider with one-click reset to profit-optimal threshold (0.13).
- Prominent decision badge: "WILL RESPOND" or "WILL NOT RESPOND".
- Probability gauges, unit economics loaded from metrics.json, and actionable recommendations.
- Interactive Model Comparison tab with real performance tables and evaluation curves.
- Feature Importance & Interpretability tab with Tree MDI, Permutation Importance, and Odds Ratios.
- Batch Prediction tab with schema validation, audit warnings, and CSV export.
- Professional styling with zero informal emojis.
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

from src.preprocess import (
    FEATURE_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    VALID_AGE_GROUPS, validate_schema
)

# Page configuration
st.set_page_config(
    page_title="Marketing Campaign Response Predictor",
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
    .badge-source {
        background-color: #EFF6FF;
        border: 1px solid #93C5FD;
        color: #1E40AF;
        padding: 0.3rem 0.8rem;
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
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-size: 1.35rem;
        font-weight: 700;
        display: inline-block;
        border: 2px solid #86EFAC;
        letter-spacing: 0.05rem;
    }
    .badge-no-respond {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-size: 1.35rem;
        font-weight: 700;
        display: inline-block;
        border: 2px solid #FCA5A5;
        letter-spacing: 0.05rem;
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


# Load artifacts
try:
    best_model, metrics_meta, feature_meta = load_model_artifacts()
    opt_threshold = float(metrics_meta.get('profit_optimal_threshold', 0.13))
    f1_threshold = float(metrics_meta.get('f1_optimal_threshold', 0.27))
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


def reset_to_optimal():
    st.session_state.threshold_slider = opt_threshold


# Sidebar Controls
st.sidebar.markdown("### Decision Controls")
current_threshold = st.sidebar.slider(
    "Decision Cutoff Threshold",
    min_value=0.05,
    max_value=0.90,
    value=st.session_state.threshold_slider,
    step=0.01,
    key="threshold_slider",
    help="Probability cutoff above which a customer is targeted."
)
st.sidebar.button("Reset to Profit-Optimal", on_click=reset_to_optimal, help="Reset cutoff to profit-maximizing threshold (0.13)")

st.sidebar.markdown("---")
st.sidebar.markdown("### Threshold Benchmarks")
st.sidebar.markdown(f"- Default Cutoff: **0.50**")
st.sidebar.markdown(f"- OOF F1-Optimal: **{f1_threshold:.2f}**")
st.sidebar.markdown(f"- OOF Profit-Optimal: **{opt_threshold:.2f}** (Deployed)")

st.sidebar.markdown("---")
st.sidebar.markdown("### Campaign Economics")
st.sidebar.markdown(f"- Outreach Cost: **${cost_per_contact:.2f} / contact**")
st.sidebar.markdown(f"- Gross Return: **${profit_per_responder:.2f} / responder**")
st.sidebar.markdown(f"- Break-Even Probability: **{cost_per_contact / profit_per_responder:.2f}**")


# Header Section
st.markdown('<div class="main-header">Marketing Campaign Response Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Case Study 157: Marketing Campaign Response Prediction Using Machine Learning — '
    'Real-time customer propensity scoring, decision threshold optimization, and marketing resource allocation.'
    '</div>',
    unsafe_allow_html=True
)

col_author, col_src, col_btn = st.columns([2.5, 3.5, 2])
with col_author:
    st.markdown(
        '<span class="badge-author">Student: <strong>Vedh Naik</strong> | '
        'Roll No.: <strong>150096725163</strong> | Cohort: <strong>Jensen Huang</strong></span>',
        unsafe_allow_html=True
    )
with col_src:
    st.markdown(
        '<span class="badge-source">Dataset: <strong>Kaggle Customer Personality Analysis (marketing_campaign.csv)</strong></span>',
        unsafe_allow_html=True
    )
with col_btn:
    st.markdown(
        '<a href="https://colab.research.google.com/github/VoidVedh/CollegeML_MarketingCampaignResponse/blob/main/notebooks/analysis.ipynb" target="_blank">'
        '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" style="vertical-align: middle; height: 28px;"/>'
        '</a>',
        unsafe_allow_html=True
    )

st.markdown("---")

# Main Application Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Single Customer Prediction",
    "Batch CSV Prediction",
    "Model Benchmark & Comparative Study",
    "Feature Importance & Business Economics"
])

# ==============================================================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# ==============================================================================
with tab1:
    st.subheader("Customer Input Parameters (Case Study 157 Features)")
    st.caption("Enter the eight customer attributes to compute campaign response probability:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Demographics**")
        age_group = st.selectbox(
            "1. Age Group",
            options=VALID_AGE_GROUPS,
            index=2,
            help="Customer age bracket derived from Year_Birth using 2014 reference observation."
        )
        income = st.number_input(
            "2. Income ($/year)",
            min_value=0.0,
            max_value=1000000.0,
            value=51380.0,
            step=1000.0,
            help="Annual household income in USD."
        )
        
    with col2:
        st.markdown("**Purchase Behavior**")
        previous_purchases = st.number_input(
            "3. Previous Purchases",
            min_value=0.0,
            max_value=100.0,
            value=12.0,
            step=1.0,
            help="Total prior purchases across web, catalog, and physical store channels."
        )
        purchase_frequency = st.number_input(
            "4. Purchase Frequency (Monthly)",
            min_value=0.0,
            max_value=10.0,
            value=0.70,
            step=0.05,
            help="Historical purchases per month of customer relationship tenure."
        )
        
    with col3:
        st.markdown("**Campaign History**")
        prev_resp_choice = st.radio(
            "5. Previous Campaign Response",
            options=["No (0)", "Yes (1)"],
            index=0,
            help="Whether the customer accepted an offer in any previous campaign wave (1 to 5)."
        )
        previous_campaign_response = 1 if "Yes" in prev_resp_choice else 0
        
        website_visits = st.number_input(
            "6. Website Visits (Monthly)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            help="Number of customer visits to company website within the past month."
        )
        
    with col4:
        st.markdown("**Promotional Engagement**")
        email_engagement = st.slider(
            "7. Email / Promotional Engagement (Proxy)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="The Kaggle dataset does not provide a direct email-open/click rate; this variable is an engagement proxy derived from historical campaign interactions."
        )
        st.caption("Note: Engagement proxy derived from prior campaign acceptance rate.")
        
        discount_usage = st.slider(
            "8. Discount Usage Proportion",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.05,
            help="Proportion of purchases completed with promotional discount deals."
        )

    st.markdown("---")
    if st.button("Predict Campaign Response", type="primary", use_container_width=True):
        customer_dict = {
            "age_group": age_group,
            "income": income,
            "previous_purchases": previous_purchases,
            "purchase_frequency": purchase_frequency,
            "previous_campaign_response": previous_campaign_response,
            "website_visits": website_visits,
            "email_engagement": email_engagement,
            "discount_usage": discount_usage
        }
        pred_label, proba_val = predict_single_customer(best_model, customer_dict, current_threshold)
        st.session_state.single_prediction = {
            "pred_label": pred_label,
            "pred_class": pred_label,
            "proba_val": proba_val,
            "proba": proba_val,
            "customer_dict": customer_dict
        }

    # Render Prediction Output if present
    if st.session_state.single_prediction is not None:
        p_data = st.session_state.single_prediction
        pred_class = int(p_data["proba_val"] >= current_threshold)
        proba_val = p_data["proba_val"]
        expected_net_value = (proba_val * profit_per_responder) - cost_per_contact
        
        st.markdown("### Prediction Result")
        res_col1, res_col2, res_col3 = st.columns([2, 1.5, 1.5])
        
        with res_col1:
            if pred_class == 1:
                st.markdown('<div class="badge-respond">WILL RESPOND</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<p style="margin-top:0.6rem; color:#166534; font-weight:600;">'
                    f'Customer exceeds the decision threshold of {current_threshold:.2f}.</p>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown('<div class="badge-no-respond">WILL NOT RESPOND</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<p style="margin-top:0.6rem; color:#991B1B; font-weight:600;">'
                    f'Customer falls below the decision threshold of {current_threshold:.2f}.</p>',
                    unsafe_allow_html=True
                )
                
        with res_col2:
            st.metric("Predicted Propensity", f"{proba_val * 100:.2f}%")
            st.progress(min(max(proba_val, 0.0), 1.0))
            
        with res_col3:
            st.metric(
                "Expected Value / Contact",
                f"${expected_net_value:+.2f}",
                delta="Profitable Contact" if expected_net_value > 0 else "Negative Expected Return"
            )

        # Actionable Business Recommendation
        st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
        if pred_class == 1:
            st.markdown(
                f"**Recommendation: Target Prospect.** "
                f"Estimated response likelihood is {proba_val * 100:.1f}%. "
                f"At an outreach cost of ${cost_per_contact:.2f} and gross conversion value of ${profit_per_responder:.2f}, "
                f"contacting this lead generates an expected net contribution of **${expected_net_value:+.2f}**."
            )
        else:
            st.markdown(
                f"**Recommendation: Suppress Outreach.** "
                f"Estimated response likelihood is {proba_val * 100:.1f}%. "
                f"Contacting this customer incurs a -${abs(expected_net_value):.2f} expected deficit per contact attempt. "
                f"Suppressing outreach preserves marketing budget and avoids consumer fatigue."
            )
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 2: BATCH CSV PREDICTION
# ==============================================================================
with tab2:
    st.subheader("Batch Customer Scoring via CSV Upload")
    st.write(
        "Upload a CSV containing customer records with the eight Case Study 157 columns. "
        "The model will score every row, compute propensity, and assign the appropriate response classification."
    )
    
    with st.expander("View Required Batch CSV Schema"):
        st.markdown("""
        The uploaded CSV must contain these eight columns:
        - `age_group` (values: `18-25`, `26-35`, `36-45`, `46-55`, `56+`)
        - `income` (numeric, annual household income in USD)
        - `previous_purchases` (numeric, count of prior purchases)
        - `purchase_frequency` (numeric, monthly purchase cadence)
        - `previous_campaign_response` (binary integer: `0` or `1`)
        - `website_visits` (numeric, monthly website visits)
        - `email_engagement` (numeric, 0.00 to 1.00 engagement proxy)
        - `discount_usage` (numeric, 0.00 to 1.00 discount proportion)
        """)

    # Download Template Button
    sample_df = pd.DataFrame([
        {"age_group": "36-45", "income": 58138.0, "previous_purchases": 22.0, "purchase_frequency": 0.85, "previous_campaign_response": 0, "website_visits": 7.0, "email_engagement": 0.0, "discount_usage": 0.14},
        {"age_group": "46-55", "income": 71613.0, "previous_purchases": 20.0, "purchase_frequency": 1.20, "previous_campaign_response": 1, "website_visits": 4.0, "email_engagement": 0.2, "discount_usage": 0.05},
        {"age_group": "26-35", "income": 26646.0, "previous_purchases": 6.0, "purchase_frequency": 0.40, "previous_campaign_response": 0, "website_visits": 6.0, "email_engagement": 0.0, "discount_usage": 0.33}
    ])
    csv_sample = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Sample Batch CSV Template",
        data=csv_sample,
        file_name="case157_batch_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Batch CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            if batch_df.empty:
                st.error("Uploaded CSV file is empty. Please upload a valid CSV file.")
            else:
                is_valid, missing_cols = validate_schema(batch_df)
                if not is_valid:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                else:
                    st.success("Schema verified successfully: All 8 Case Study 157 features present.")
                    
                    # Perform Batch Prediction
                    batch_probas = best_model.predict_proba(batch_df[FEATURE_COLUMNS])[:, 1]
                    batch_preds = (batch_probas >= current_threshold).astype(int)
                    batch_classes = ["WILL RESPOND" if p == 1 else "WILL NOT RESPOND" for p in batch_preds]
                    
                    batch_df["Response_Probability"] = np.round(batch_probas, 4)
                    batch_df["Prediction"] = batch_classes
                    
                    n_targeted = int(np.sum(batch_preds))
                    total_records = len(batch_df)
                    pct_targeted = (n_targeted / total_records) * 100
                    
                    bcol1, bcol2, bcol3 = st.columns(3)
                    bcol1.metric("Total Prospects Scored", f"{total_records:,}")
                    bcol2.metric("Recommended Targets", f"{n_targeted:,} ({pct_targeted:.1f}%)")
                    bcol3.metric("Estimated Ad Spend", f"${n_targeted * cost_per_contact:,.2f}")
                    
                    st.dataframe(batch_df.head(20), use_container_width=True)
                    
                    scored_csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Complete Scored CSV",
                        data=scored_csv,
                        file_name="scored_campaign_predictions.csv",
                        mime="text/csv"
                    )
        except pd.errors.EmptyDataError:
            st.error("Uploaded CSV file is empty. Please upload a valid CSV file.")
        except Exception as e:
            if "no columns to parse" in str(e).lower():
                st.error("Uploaded CSV file is empty. Please upload a valid CSV file.")
            else:
                st.error(f"Error processing CSV: {e}")

# ==============================================================================
# TAB 3: MODEL BENCHMARK & COMPARATIVE STUDY
# ==============================================================================
with tab3:
    st.subheader("Six Algorithm Benchmark & Comparative Evaluation")
    st.caption("All six models evaluated on the held-out test set (448 customers, 67 responders) using Kaggle Customer Personality features:")

    results_csv_path = os.path.join(PROJECT_ROOT, "reports", "results_comparison.csv")
    if os.path.exists(results_csv_path):
        res_df = pd.read_csv(results_csv_path)
        display_cols = [
            'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score',
            'ROC-AUC', 'PR-AUC', 'FPR', 'FPR_at_Recall_70', 'Top20_Capture_Rate', 'CV_PR_AUC_Mean'
        ]
        cols_present = [c for c in display_cols if c in res_df.columns]
        st.dataframe(res_df[cols_present].style.highlight_max(subset=['F1-Score', 'ROC-AUC', 'PR-AUC', 'CV_PR_AUC_Mean'], color='#dcfce7'), use_container_width=True)
    else:
        st.info("Benchmark table not found. Run `python src/train.py` to generate.")

    st.markdown("---")
    st.subheader("Comparative Performance Visualizations")
    
    col_fig1, col_fig2 = st.columns(2)
    roc_fig_path = os.path.join(PROJECT_ROOT, "reports", "figures", "roc_curves_all_models.png")
    pr_fig_path = os.path.join(PROJECT_ROOT, "reports", "figures", "pr_curves_all_models.png")
    
    with col_fig1:
        if os.path.exists(roc_fig_path):
            st.image(roc_fig_path, caption="Receiver Operating Characteristic (ROC) Curves Across All 6 Models")
    with col_fig2:
        if os.path.exists(pr_fig_path):
            st.image(pr_fig_path, caption="Precision-Recall (PR) Curves Across All 6 Models")

    st.markdown("---")
    cm_grid_path = os.path.join(PROJECT_ROOT, "reports", "figures", "confusion_matrices_grid.png")
    if os.path.exists(cm_grid_path):
        st.image(cm_grid_path, caption="Confusion Matrix Heatmaps Evaluated at Each Model's Out-Of-Fold Optimal Threshold")

# ==============================================================================
# TAB 4: FEATURE IMPORTANCE & BUSINESS ECONOMICS
# ==============================================================================
with tab4:
    st.subheader("Feature Importance & Predictive Drivers")
    st.caption("Insights derived via Tree-based MDI, Permutation Importance, and Logistic Regression Odds Ratios:")
    
    fi_col1, fi_col2 = st.columns(2)
    tree_fi_path = os.path.join(PROJECT_ROOT, "reports", "figures", "feature_importance_tree.png")
    perm_fi_path = os.path.join(PROJECT_ROOT, "reports", "figures", "feature_importance_permutation.png")
    
    with fi_col1:
        if os.path.exists(tree_fi_path):
            st.image(tree_fi_path, caption="Tree-Based Feature Importance (Mean Decrease in Impurity)")
    with fi_col2:
        if os.path.exists(perm_fi_path):
            st.image(perm_fi_path, caption="Permutation Feature Importance on Held-Out Test Set")

    st.markdown("---")
    st.subheader("Four-Strategy Marketing Economics Simulation")
    
    sim_csv_path = os.path.join(PROJECT_ROOT, "reports", "business_simulation.csv")
    if os.path.exists(sim_csv_path):
        sim_df = pd.read_csv(sim_csv_path)
        st.dataframe(sim_df, use_container_width=True)
    
    sim_fig_path = os.path.join(PROJECT_ROOT, "reports", "figures", "business_simulation_roi.png")
    gains_fig_path = os.path.join(PROJECT_ROOT, "reports", "figures", "cumulative_gains_lift.png")
    
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        if os.path.exists(sim_fig_path):
            st.image(sim_fig_path, caption="Net Campaign Profit & Return on Investment (ROI) Across Strategies")
    with col_sim2:
        if os.path.exists(gains_fig_path):
            st.image(gains_fig_path, caption="Cumulative Gains and Decile Lift Targeting Analysis")
