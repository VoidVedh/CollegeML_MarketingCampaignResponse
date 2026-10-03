"""
eda.py
Exploratory Data Analysis (EDA) module for Case Study 157:
Marketing Campaign Response Prediction Using Machine Learning.

Dataset: Kaggle Customer Personality Analysis (marketing_campaign.csv)
Visualizes data health, target distribution, feature correlations, and response rates.
Outputs high-resolution charts to reports/figures/.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import PROCESSED_DATA_PATH, RAW_DATA_PATH, load_and_split_data

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Helvetica, Arial, DejaVu Sans"
plt.rcParams["font.size"] = 10


def run_full_eda(
    processed_path: str = PROCESSED_DATA_PATH,
    output_dir: str = "reports/figures"
) -> Dict[str, Any]:
    """
    Executes comprehensive EDA on the 8 Case Study 157 features and saves publication figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ensure processed dataset exists
    if not os.path.exists(processed_path):
        load_and_split_data()

    df = pd.read_csv(processed_path)

    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "summary_statistics": df.describe().to_dict(),
        "target_distribution": df["target"].value_counts().to_dict(),
        "target_balance_pct": float(df["target"].mean() * 100),
        "figures": {}
    }

    # -------------------------------------------------------------
    # 1. Target Distribution Plot
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        x="target",
        hue="target",
        data=df,
        palette=["#4a90e2", "#e94e77"],
        edgecolor="black",
        legend=False,
        alpha=0.9
    )
    plt.title("Target Distribution: Campaign Response", fontweight="bold", pad=12)
    plt.xlabel("Target (0 = Will Not Respond, 1 = Will Respond)")
    plt.ylabel("Number of Customers")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Will Not Respond (0)", "Will Respond (1)"])

    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        pct = (height / total) * 100
        ax.annotate(
            f"{int(height):,}\n({pct:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, height / 2),
            ha="center", va="center", color="white", fontweight="bold", fontsize=11
        )

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "eda_target_distribution.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    summary["figures"]["target_distribution"] = fig1_path

    # -------------------------------------------------------------
    # 2. Response Rates by Age Group & Prior Campaign Response
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Age group response rate
    age_resp = df.groupby("age_group", observed=False)["target"].mean().reset_index()
    age_resp["target"] = age_resp["target"] * 100
    sns.barplot(
        data=age_resp,
        x="age_group",
        y="target",
        hue="age_group",
        palette="Blues_r",
        edgecolor="black",
        legend=False,
        ax=axes[0]
    )
    axes[0].set_title("Response Rate by Age Group", fontweight="bold")
    axes[0].set_xlabel("Age Group")
    axes[0].set_ylabel("Response Rate (%)")
    for p in axes[0].patches:
        axes[0].annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontweight="bold"
        )

    # Previous campaign response effect
    prev_resp = df.groupby("previous_campaign_response")["target"].mean().reset_index()
    prev_resp["target"] = prev_resp["target"] * 100
    sns.barplot(
        data=prev_resp,
        x="previous_campaign_response",
        y="target",
        hue="previous_campaign_response",
        palette=["#4a90e2", "#50e3c2"],
        edgecolor="black",
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title("Response Rate by Prior Campaign Acceptance", fontweight="bold")
    axes[1].set_xlabel("Previous Campaign Response")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["No Prior Acceptance (0)", "Accepted Prior Offer (1)"])
    axes[1].set_ylabel("Response Rate (%)")
    for p in axes[1].patches:
        axes[1].annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontweight="bold"
        )

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "eda_response_rates_breakdown.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    summary["figures"]["response_rates"] = fig2_path

    # -------------------------------------------------------------
    # 3. Numeric Distributions & Boxplots
    # -------------------------------------------------------------
    numeric_cols = ["income", "previous_purchases", "purchase_frequency", "website_visits", "discount_usage"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    for i, col in enumerate(numeric_cols):
        sns.boxplot(
            x="target",
            y=col,
            hue="target",
            data=df,
            palette=["#4a90e2", "#e94e77"],
            ax=axes[i],
            legend=False,
            showfliers=False
        )
        axes[i].set_title(col.replace("_", " ").title(), fontweight="bold")
        axes[i].set_xlabel("Response")
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["No", "Yes"])
        axes[i].set_ylabel("Value")

    plt.suptitle("Feature Distributions by Campaign Response Status (Outliers Suppressed)", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "eda_numeric_distributions_boxplots.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    summary["figures"]["boxplots"] = fig3_path

    # -------------------------------------------------------------
    # 4. Correlation Heatmap
    # -------------------------------------------------------------
    corr_cols = numeric_cols + ["email_engagement", "previous_campaign_response", "target"]
    corr_matrix = df[corr_cols].corr()

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.5,
        vmax=0.5,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Matrix of Numeric Features & Target Response", fontweight="bold", pad=12)
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "eda_correlation_heatmap.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    summary["figures"]["correlation_heatmap"] = fig4_path

    print("EDA completed successfully: generated 4 figures in reports/figures/")
    return summary


if __name__ == "__main__":
    run_full_eda()
