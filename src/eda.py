"""
eda.py
Exploratory Data Analysis (EDA) module for the Kaggle Marketing Dataset
(Bank Marketing / Term Deposit Subscription Prediction — Case Study 157).

Analyzes data health, distributions, demographic response rates, correlation structures,
and produces high-resolution figures saved to reports/figures/.
Excludes 'duration' to strictly uphold zero-leakage standards.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import DEFAULT_DATA_PATH, derive_age_group, NUMERIC_FEATURES

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['font.size'] = 10


def run_full_eda(data_path: str = DEFAULT_DATA_PATH, output_dir: str = "reports/figures") -> Dict[str, Any]:
    """
    Executes full EDA pipeline on the Kaggle Marketing Dataset and saves all required figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    try:
        df = pd.read_csv(data_path, sep=",")
        if df.shape[1] == 1:
            df = pd.read_csv(data_path, sep=";")
    except Exception:
        df = pd.read_csv(data_path, sep=";")

    # Drop duration to prevent leakage
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Derive age_group
    if "age_group" not in df.columns and "age" in df.columns:
        df["age_group"] = derive_age_group(df["age"])

    # Standardize target 'y' -> 1/0
    target_col = "y" if "y" in df.columns else "responded"
    if df[target_col].dtype == object or isinstance(df[target_col].iloc[0], str):
        df["y"] = (df[target_col].astype(str).str.lower().str.strip() == "yes").astype(int)
    else:
        df["y"] = df[target_col].astype(int)

    eda_summary = {
        'shape': df.shape,
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isna().sum().to_dict(),
        'duplicates': int(df.duplicated().sum()),
        'summary_statistics': df.describe().to_dict(),
        'target_distribution': df['y'].value_counts().to_dict(),
        'target_balance_pct': float(df['y'].mean() * 100),
        'figures': {}
    }

    # -------------------------------------------------------------
    # 1. Target Distribution Plot & Class Imbalance
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        x='y',
        hue='y',
        data=df,
        palette=['#4a90e2', '#e94e77'],
        edgecolor='black',
        legend=False,
        alpha=0.9
    )
    plt.title("Target Distribution: Term Deposit Subscription (y)", fontweight='bold', pad=12)
    plt.xlabel("Target (0 = No Subscription, 1 = Subscribed)")
    plt.ylabel("Number of Clients")
    plt.xticks([0, 1], ['No Deposit (0)', 'Subscribed (1)'])

    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        pct = (height / total) * 100
        ax.annotate(f"{height:,}\n({pct:.1f}%)",
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', color='white', fontweight='bold', fontsize=11)

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "eda_target_distribution.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    eda_summary['figures']['target_distribution'] = {
        'path': fig1_path,
        'takeaway': f"The target exhibits substantial class imbalance (~{eda_summary['target_balance_pct']:.1f}% positive responders), confirming that standard accuracy will be misleading and cost-sensitive/PR-AUC/F1 evaluation is mandatory."
    }

    # -------------------------------------------------------------
    # 2. Response Rates by Demographic & Engagement Drivers
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (a) Age Group
    age_order = ['18-25', '26-35', '36-45', '46-55', '56+']
    age_rates = df.groupby('age_group', observed=False)['y'].mean().reindex(age_order) * 100
    sns.barplot(x=age_rates.index, y=age_rates.values, hue=age_rates.index, ax=axes[0, 0], palette='Blues_r', edgecolor='black', legend=False)
    axes[0, 0].set_title("Subscription Rate by Age Group", fontweight='bold')
    axes[0, 0].set_ylabel("Subscription Rate (%)")
    axes[0, 0].set_xlabel("Age Group")
    for p in axes[0, 0].patches:
        axes[0, 0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width()/2., p.get_height()),
                            ha='center', va='bottom', xytext=(0, 2), textcoords='offset points', fontweight='bold')

    # (b) Previous Campaign Outcome (poutcome)
    pout_order = ['success', 'failure', 'nonexistent']
    pout_rates = df.groupby('poutcome', observed=False)['y'].mean().reindex(pout_order) * 100
    sns.barplot(x=pout_rates.index, y=pout_rates.values, hue=pout_rates.index, ax=axes[0, 1], palette='Purples_r', edgecolor='black', legend=False)
    axes[0, 1].set_title("Subscription Rate by Previous Campaign Outcome (poutcome)", fontweight='bold')
    axes[0, 1].set_ylabel("Subscription Rate (%)")
    axes[0, 1].set_xlabel("Previous Outcome")
    for p in axes[0, 1].patches:
        axes[0, 1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width()/2., p.get_height()),
                            ha='center', va='bottom', xytext=(0, 2), textcoords='offset points', fontweight='bold')

    # (c) Job Category
    job_rates = (df.groupby('job', observed=False)['y'].mean() * 100).sort_values(ascending=False)
    sns.barplot(x=job_rates.index, y=job_rates.values, hue=job_rates.index, ax=axes[1, 0], palette='Greens_r', edgecolor='black', legend=False)
    axes[1, 0].set_title("Subscription Rate by Job Category", fontweight='bold')
    axes[1, 0].set_ylabel("Subscription Rate (%)")
    axes[1, 0].set_xlabel("Job")
    axes[1, 0].tick_params(axis='x', rotation=45)
    for p in axes[1, 0].patches:
        axes[1, 0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width()/2., p.get_height()),
                            ha='center', va='bottom', xytext=(0, 2), textcoords='offset points', fontsize=8, fontweight='bold')

    # (d) Contact Communication Channel
    contact_rates = (df.groupby('contact', observed=False)['y'].mean() * 100).sort_values(ascending=False)
    sns.barplot(x=contact_rates.index, y=contact_rates.values, hue=contact_rates.index, ax=axes[1, 1], palette='Oranges_r', edgecolor='black', legend=False)
    axes[1, 1].set_title("Subscription Rate by Contact Channel", fontweight='bold')
    axes[1, 1].set_ylabel("Subscription Rate (%)")
    axes[1, 1].set_xlabel("Contact Channel")
    for p in axes[1, 1].patches:
        axes[1, 1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width()/2., p.get_height()),
                            ha='center', va='bottom', xytext=(0, 2), textcoords='offset points', fontweight='bold')

    plt.suptitle("Term Deposit Response Rates Across Demographics & Contact Drivers", fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "eda_response_rates_breakdown.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    eda_summary['figures']['response_rates'] = {
        'path': fig2_path,
        'takeaway': "Prior campaign success (poutcome='success', >65% conversion), student/retired occupations, and cellular contact represent massive positive response multipliers."
    }

    # -------------------------------------------------------------
    # 3. Distributions & Boxplots of Numeric Features by Target
    # -------------------------------------------------------------
    key_numeric = ['age', 'campaign', 'pdays', 'emp.var.rate', 'euribor3m', 'cons.conf.idx']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, col in enumerate(key_numeric):
        if col in df.columns:
            sns.boxplot(
                data=df,
                x='y',
                y=col,
                hue='y',
                ax=axes[i],
                palette=['#4a90e2', '#e94e77'],
                showmeans=True,
                legend=False,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
            )
            axes[i].set_title(f"{col} by Subscription Status", fontweight='bold')
            axes[i].set_xticks([0, 1])
            axes[i].set_xticklabels(['No Deposit (0)', 'Subscribed (1)'])
            axes[i].set_xlabel("")

    plt.suptitle("Distributions of Key Numerical Features by Term Deposit Response", fontweight='bold', fontsize=15, y=0.99)
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "eda_numeric_distributions_boxplots.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    eda_summary['figures']['numeric_distributions'] = {
        'path': fig3_path,
        'takeaway': "Subscribers are concentrated in lower employment variation rate (emp.var.rate) and lower Euribor 3-month interest rate regimes, reflecting macroeconomic sensitivity."
    }

    # -------------------------------------------------------------
    # 4. Correlation Heatmap
    # -------------------------------------------------------------
    corr_cols = [c for c in NUMERIC_FEATURES if c in df.columns] + ['y']
    corr_matrix = df[corr_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=-0.6,
        vmax=1.0,
        center=0,
        linewidths=0.5,
        square=True
    )
    plt.title("Correlation Heatmap: Numerical Features vs Term Deposit Subscription (y)", fontweight='bold', pad=15)
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "eda_correlation_heatmap.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    eda_summary['figures']['correlation_heatmap'] = {
        'path': fig4_path,
        'takeaway': "Macroeconomic features (emp.var.rate, euribor3m, nr.employed) exhibit strong mutual collinearity and negative correlation with term deposit subscriptions."
    }

    return eda_summary


if __name__ == "__main__":
    summary = run_full_eda()
    print("EDA completed successfully on Kaggle dataset!")
    print(f"Dataset Shape: {summary['shape']}")
    print(f"Target Balance: {summary['target_balance_pct']:.2f}% positive responders")
