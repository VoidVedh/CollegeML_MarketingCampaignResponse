"""
verify_consistency.py
Automated cross-document consistency verification script.
Checks that all key quantitative performance figures, business simulation metrics,
and economic values in README.md, reports/final_report.md, and notebooks/analysis.ipynb
faithfully match models/metrics.json without manual discrepancies.
"""

import os
import json
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def verify_all_consistency():
    metrics_path = os.path.join(PROJECT_ROOT, "models", "metrics.json")
    with open(metrics_path, 'r') as f:
        m = json.load(f)

    report_path = os.path.join(PROJECT_ROOT, "reports", "final_report.md")
    with open(report_path, 'r', encoding='utf-8') as f:
        report_text = f.read()

    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme_text = f.read()

    notebook_path = os.path.join(PROJECT_ROOT, "notebooks", "analysis.ipynb")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_text = f.read()

    winner = m['best_model_name']
    deployed = m['deployed_model_name']
    cv_pr = f"{m['cv_metrics'][winner]['cv_pr_auc_mean']:.4f}"
    cv_roc = f"{m['cv_metrics'][winner]['cv_roc_auc_mean']:.4f}"
    fpr_70 = f"{m['cv_metrics'][winner]['fpr_at_recall_70']:.4f}"
    f1_t = f"{m['f1_optimal_threshold']:.2f}"
    profit_t = f"{m['profit_optimal_threshold']:.2f}"
    
    sim = m['business_simulation']
    # Strategy 4 (profit-optimal) net profit and cost saved
    strat4 = next(s for s in sim if 'Profit-Optimal' in s['Strategy'])
    strat4_profit = f"${strat4['Net Profit ($)']:,.2f}"
    strat4_contacts = str(int(strat4['Targeted Contacts']))
    strat4_responders = str(int(strat4['Responders Reached']))
    strat4_cost = f"${strat4['Total Cost ($)']:,.2f}"
    strat4_saved = f"{strat4['Cost Saved vs All (%)']:.1f}%"
    
    # Strategy 1 (everyone) profit
    strat1 = next(s for s in sim if 'Everyone' in s['Strategy'])
    strat1_profit = f"${strat1['Net Profit ($)']:,.2f}"
    
    # Odds ratio of previous_campaign_response
    odds_ratio_prev = f"{m['logistic_regression_odds_ratios']['previous_campaign_response']:.4f}"
    
    # Brier scores
    val_uncal = f"{m['brier_scores']['validation_uncalibrated']:.5f}"
    val_cal = f"{m['brier_scores']['validation_calibrated']:.5f}"

    targets = [
        ("Winner Model Name", winner),
        ("CV PR-AUC", cv_pr),
        ("CV ROC-AUC", cv_roc),
        ("FPR @ Recall 70%", fpr_70),
        ("F1 Optimal Threshold", f1_t),
        ("Profit Optimal Threshold", profit_t),
        ("Profit-Optimal Net Profit", strat4_profit),
        ("Profit-Optimal Contacts", strat4_contacts),
        ("Profit-Optimal Responders Reached", strat4_responders),
        ("Profit-Optimal Total Cost", strat4_cost),
        ("Profit-Optimal Cost Saved", strat4_saved),
        ("Contact Everyone Net Profit", strat1_profit),
        ("Previous Campaign Response Odds Ratio", odds_ratio_prev),
        ("Validation Uncalibrated Brier", val_uncal),
        ("Validation Calibrated Brier", val_cal),
    ]

    mismatches = []
    
    print("=" * 70)
    print("VERIFYING CONSISTENCY ACROSS README, REPORT, AND NOTEBOOK")
    print("=" * 70)

    for label, val in targets:
        # Check report
        if val not in report_text:
            mismatches.append(f"MISMATCH in reports/final_report.md: '{label}' expected '{val}' but not found.")
        else:
            print(f"PASS [Report]:   {label:<40} -> Found '{val}'")
            
        # Check README (only applicable metrics)
        if label not in ["Validation Uncalibrated Brier", "Validation Calibrated Brier"]:
            if val not in readme_text:
                mismatches.append(f"MISMATCH in README.md: '{label}' expected '{val}' but not found.")
            else:
                print(f"PASS [README]:   {label:<40} -> Found '{val}'")
                
        # Check Notebook
        if label not in ["Validation Uncalibrated Brier", "Validation Calibrated Brier"]:
            if val not in notebook_text:
                mismatches.append(f"MISMATCH in notebooks/analysis.ipynb: '{label}' expected '{val}' but not found.")
            else:
                print(f"PASS [Notebook]: {label:<40} -> Found '{val}'")

    print("-" * 70)
    if mismatches:
        print(f"FAILED: Found {len(mismatches)} consistency mismatches:")
        for m_err in mismatches:
            print(f"  ❌ {m_err}")
        return False
    else:
        print("ALL AUDITED NUMBERS MATCH 100% PERFECTLY WITH METRICS.JSON!")
        return True


if __name__ == '__main__':
    success = verify_all_consistency()
    if not success:
        exit(1)
