"""
train.py
Main execution and training pipeline for Marketing Campaign Response Prediction.
Implements:
- Leakage-free preprocessing with scikit-learn ColumnTransformer
- 5-fold Stratified Cross-Validation across all 6 classification algorithms
- Defensible model selection by CV PR-AUC & ROC-AUC with standard deviation & tie-breaks
- FPR at fixed recall (70%) benchmark
- Imbalance handling comparison: None vs class_weight vs SMOTENC
- Out-Of-Fold (OOF) training threshold optimization (F1-optimal & Profit-optimal)
- 4-Strategy business simulation (Everyone, Default 0.50, F1-optimal, Profit-optimal)
- Probability calibration via FrozenEstimator with validation Brier check
- Bootstrap 95% Confidence Intervals for test metrics
- Explainability (Tree MDI, Permutation Importance, Odds Ratios, SHAP)
- Profiling for both ACTUAL and PREDICTED responders
- Automated report generation via build_report.py
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

# Local imports
from src.generate_data import generate_campaign_dataset
from src.preprocess import (
    load_and_split_data, create_preprocessor, get_feature_names,
    unit_test_preprocessing, FEATURE_COLUMNS, CONTINUOUS_FEATURES,
    BINARY_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN
)
from src.eda import run_full_eda
from src.evaluate import (
    compute_metrics_at_threshold, compute_fpr_at_recall, compute_bootstrap_ci,
    find_optimal_thresholds_oof, run_four_strategy_simulation,
    evaluate_probability_calibration, compute_comprehensive_customer_profiles,
    plot_metrics_comparison, plot_roc_curves, plot_precision_recall_curves,
    plot_confusion_matrices, plot_cumulative_gains_and_lift,
    analyze_feature_importances, COST_PER_CONTACT, PROFIT_PER_RESPONDER
)


def get_model_grid_configs(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Returns estimator templates and focused hyperparameter grids for all 6 models.
    """
    return {
        'Logistic Regression': {
            'estimator': LogisticRegression(random_state=random_state, max_iter=1000),
            'param_grid': {
                'classifier__C': [0.01, 0.1, 1.0, 5.0, 10.0],
                'classifier__solver': ['lbfgs']
            },
            'supports_class_weight': True
        },
        'K-Nearest Neighbors': {
            'estimator': KNeighborsClassifier(),
            'param_grid': {
                'classifier__n_neighbors': [5, 9, 15, 21],
                'classifier__weights': ['uniform', 'distance']
            },
            'supports_class_weight': False
        },
        'Decision Tree': {
            'estimator': DecisionTreeClassifier(random_state=random_state),
            'param_grid': {
                'classifier__max_depth': [3, 5, 8, 12],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__criterion': ['gini', 'entropy']
            },
            'supports_class_weight': True
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=random_state),
            'param_grid': {
                'classifier__n_estimators': [50, 100, 150],
                'classifier__max_depth': [5, 8, 12, None],
                'classifier__min_samples_split': [2, 5]
            },
            'supports_class_weight': True
        },
        'Naive Bayes': {
            'estimator': GaussianNB(),
            'param_grid': {
                'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            },
            'supports_class_weight': False
        },
        'Gradient Boosting': {
            'estimator': GradientBoostingClassifier(random_state=random_state),
            'param_grid': {
                'classifier__n_estimators': [50, 100, 150],
                'classifier__learning_rate': [0.05, 0.1, 0.2],
                'classifier__max_depth': [3, 5]
            },
            'supports_class_weight': False
        }
    }


def train_and_evaluate_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """
    Tunes all 6 models via 5-fold Stratified CV, collects CV metrics (PR-AUC, ROC-AUC),
    determines OOF-tuned thresholds, and evaluates on held-out test data.
    """
    configs = get_model_grid_configs(random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    fitted_models = {}
    oof_thresholds = {}
    model_cv_metrics = {}
    test_results_list = []
    
    print("\n" + "="*70)
    print("STEP 3: 5-FOLD STRATIFIED CV HYPERPARAMETER TUNING & BENCHMARK")
    print("="*70)
    
    for name, conf in configs.items():
        t0 = time.time()
        print(f"\n--> Tuning {name}...")
        
        # Pipeline: Preprocessor -> Classifier
        base_pipe = Pipeline([
            ('preprocessor', create_preprocessor()),
            ('classifier', conf['estimator'])
        ])
        
        # Optimize on average_precision (PR-AUC) as imbalanced target standard
        grid = GridSearchCV(
            base_pipe,
            param_grid=conf['param_grid'],
            cv=cv,
            scoring='average_precision',
            n_jobs=-1,
            refit=True
        )
        grid.fit(X_train, y_train)
        
        best_est = grid.best_estimator_
        best_params = grid.best_params_
        
        # Collect CV PR-AUC and ROC-AUC
        cv_pr_scores = cross_val_score(best_est, X_train, y_train, cv=cv, scoring='average_precision', n_jobs=-1)
        cv_roc_scores = cross_val_score(best_est, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        # Out-Of-Fold probability generation
        oof_proba = cross_val_predict(best_est, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
        
        # Find model's own OOF F1-optimal threshold
        t_sweep = np.linspace(0.05, 0.95, 91)
        oof_f1s = [f1_score(y_train, (oof_proba >= t).astype(int), zero_division=0) for t in t_sweep]
        best_f1_idx = int(np.argmax(oof_f1s))
        model_oof_t = float(t_sweep[best_f1_idx])
        
        # Evaluate on test set using model's own OOF threshold
        test_proba = best_est.predict_proba(X_test)[:, 1]
        m_tuned = compute_metrics_at_threshold(y_test, test_proba, threshold=model_oof_t)
        
        # Also compute metrics at standard 0.50 threshold for assignment requirements
        m_def = compute_metrics_at_threshold(y_test, test_proba, threshold=0.50)
        
        # Compute FPR at fixed recall = 70%
        fpr_70 = compute_fpr_at_recall(y_test, test_proba, target_recall=0.70)
        
        elapsed = time.time() - t0
        print(f"    Completed in {elapsed:.1f}s | Best Params: {best_params}")
        print(f"    CV PR-AUC:  {np.mean(cv_pr_scores):.4f} (+/- {np.std(cv_pr_scores):.4f})")
        print(f"    CV ROC-AUC: {np.mean(cv_roc_scores):.4f} (+/- {np.std(cv_roc_scores):.4f})")
        print(f"    OOF Tuned Threshold: {model_oof_t:.2f} -> Test F1: {m_tuned['f1']:.4f}, FPR@Rec=70%: {fpr_70:.4f}")
        
        fitted_models[name] = best_est
        oof_thresholds[name] = model_oof_t
        model_cv_metrics[name] = {
            'best_params': best_params,
            'cv_pr_auc_mean': float(np.mean(cv_pr_scores)),
            'cv_pr_auc_std': float(np.std(cv_pr_scores)),
            'cv_roc_auc_mean': float(np.mean(cv_roc_scores)),
            'cv_roc_auc_std': float(np.std(cv_roc_scores)),
            'oof_threshold': model_oof_t,
            'oof_f1': float(oof_f1s[best_f1_idx]),
            'fpr_at_recall_70': float(fpr_70)
        }
        
        test_results_list.append({
            'Model': name,
            'Accuracy': m_def['accuracy'],
            'Precision': m_def['precision'],
            'Recall': m_def['recall'],
            'F1-Score': m_def['f1'],
            'ROC-AUC': m_def['roc_auc'],
            'PR-AUC': m_def['pr_auc'],
            'FPR': m_def['fpr'],
            'TN': m_def['tn'],
            'FP': m_def['fp'],
            'FN': m_def['fn'],
            'TP': m_def['tp'],
            'Tuned_Threshold': model_oof_t,
            'Tuned_F1': m_tuned['f1'],
            'Tuned_Precision': m_tuned['precision'],
            'Tuned_Recall': m_tuned['recall'],
            'Tuned_FPR': m_tuned['fpr'],
            'FPR_at_Recall_70': fpr_70,
            'CV_PR_AUC_Mean': float(np.mean(cv_pr_scores)),
            'CV_PR_AUC_Std': float(np.std(cv_pr_scores)),
            'CV_ROC_AUC_Mean': float(np.mean(cv_roc_scores)),
            'CV_ROC_AUC_Std': float(np.std(cv_roc_scores)),
            'Best_Params': json.dumps(best_params)
        })
        
    results_df = pd.DataFrame(test_results_list)
    return fitted_models, results_df, model_cv_metrics


def select_best_model_defensibly(results_df: pd.DataFrame) -> Tuple[str, str]:
    """
    Defensible Model Selection Rule:
    1. Sort models by CV PR-AUC mean.
    2. Check if top-2 models are within 1 standard deviation of CV PR-AUC.
    3. If within 1 std, declare a statistical tie and break tie by:
       - Lower FPR at fixed recall (70%)
       - Model simplicity / inference latency
       - Interpretability
    Returns: (winner_name, selection_justification)
    """
    sorted_df = results_df.sort_values(by='CV_PR_AUC_Mean', ascending=False).reset_index(drop=True)
    top1 = sorted_df.iloc[0]
    top2 = sorted_df.iloc[1]
    
    cv_diff = top1['CV_PR_AUC_Mean'] - top2['CV_PR_AUC_Mean']
    top1_std = top1['CV_PR_AUC_Std']
    
    if cv_diff < top1_std:
        tie_declared = True
        # Break tie by FPR at fixed recall (70%)
        if top1['FPR_at_Recall_70'] <= top2['FPR_at_Recall_70']:
            winner = top1['Model']
            tie_reason = f"{top1['Model']} broke the tie with lower FPR at 70% recall ({top1['FPR_at_Recall_70']:.4f} vs {top2['FPR_at_Recall_70']:.4f})."
        else:
            winner = top2['Model']
            tie_reason = f"{top2['Model']} broke the tie with lower FPR at 70% recall ({top2['FPR_at_Recall_70']:.4f} vs {top1['FPR_at_Recall_70']:.4f})."
            
        justification = (
            f"Statistical Tie Disclosed: The top two models, {top1['Model']} (CV PR-AUC: {top1['CV_PR_AUC_Mean']:.4f} ± {top1['CV_PR_AUC_Std']:.4f}) "
            f"and {top2['Model']} (CV PR-AUC: {top2['CV_PR_AUC_Mean']:.4f} ± {top2['CV_PR_AUC_Std']:.4f}), are within 1 standard deviation ({cv_diff:.4f} < {top1_std:.4f}). "
            f"Therefore, they are statistically tied on ranking performance. {tie_reason}"
        )
    else:
        winner = top1['Model']
        justification = (
            f"{top1['Model']} won decisively with CV PR-AUC of {top1['CV_PR_AUC_Mean']:.4f} ± {top1['CV_PR_AUC_Std']:.4f}, "
            f"leading the runner-up {top2['Model']} ({top2['CV_PR_AUC_Mean']:.4f}) by more than 1 standard deviation."
        )
        
    return winner, justification


def evaluate_imbalance_treatments(
    best_model_name: str,
    base_estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluates 3 class imbalance treatments for the champion model:
      1. None (standard unweighted baseline)
      2. class_weight='balanced' (if supported by estimator)
      3. SMOTENC (with categorical columns declared properly)
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    treatments = []
    
    # Dynamically derive categorical indices from fitted preprocessor:
    # index of 'previous_campaign_response' plus all indices whose feature names start with 'age_group_'
    preproc_fitted = create_preprocessor().fit(X_train)
    all_feat_names = list(preproc_fitted.get_feature_names_out())
    if 'previous_campaign_response' not in all_feat_names:
        raise ValueError("Feature 'previous_campaign_response' not found in preprocessor feature names.")
    prev_idx = all_feat_names.index('previous_campaign_response')
    age_indices = [i for i, name in enumerate(all_feat_names) if name.startswith('age_group_')]
    if not age_indices:
        raise ValueError("No 'age_group_' features found in preprocessor feature names.")
    cat_indices = sorted([prev_idx] + age_indices)
    
    # Treatment 1: None
    pipe_none = Pipeline([
        ('preprocessor', create_preprocessor()),
        ('classifier', base_estimator)
    ])
    cv_pr_none = cross_val_score(pipe_none, X_train, y_train, cv=cv, scoring='average_precision', n_jobs=-1)
    cv_roc_none = cross_val_score(pipe_none, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    pipe_none.fit(X_train, y_train)
    p_none = pipe_none.predict_proba(X_test)[:, 1]
    
    treatments.append({
        'Treatment': 'None (Unweighted Baseline)',
        'CV_PR_AUC_Mean': float(np.mean(cv_pr_none)),
        'CV_PR_AUC_Std': float(np.std(cv_pr_none)),
        'CV_ROC_AUC_Mean': float(np.mean(cv_roc_none)),
        'CV_ROC_AUC_Std': float(np.std(cv_roc_none)),
        'Test_PR_AUC': float(average_precision_score(y_test, p_none)),
        'Test_ROC_AUC': float(roc_auc_score(y_test, p_none)),
        'Test_F1_at_0.50': float(f1_score(y_test, (p_none >= 0.50).astype(int), zero_division=0))
    })
    
    # Treatment 2: class_weight='balanced' (if supported)
    if 'class_weight' in base_estimator.get_params():
        cw_estimator = base_estimator.__class__(**base_estimator.get_params())
        cw_estimator.set_params(class_weight='balanced')
        pipe_cw = Pipeline([
            ('preprocessor', create_preprocessor()),
            ('classifier', cw_estimator)
        ])
        cv_pr_cw = cross_val_score(pipe_cw, X_train, y_train, cv=cv, scoring='average_precision', n_jobs=-1)
        cv_roc_cw = cross_val_score(pipe_cw, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        pipe_cw.fit(X_train, y_train)
        p_cw = pipe_cw.predict_proba(X_test)[:, 1]
        
        treatments.append({
            'Treatment': "class_weight='balanced'",
            'CV_PR_AUC_Mean': float(np.mean(cv_pr_cw)),
            'CV_PR_AUC_Std': float(np.std(cv_pr_cw)),
            'CV_ROC_AUC_Mean': float(np.mean(cv_roc_cw)),
            'CV_ROC_AUC_Std': float(np.std(cv_roc_cw)),
            'Test_PR_AUC': float(average_precision_score(y_test, p_cw)),
            'Test_ROC_AUC': float(roc_auc_score(y_test, p_cw)),
            'Test_F1_at_0.50': float(f1_score(y_test, (p_cw >= 0.50).astype(int), zero_division=0))
        })
    else:
        # For tree/boosting models without class_weight parameter
        treatments.append({
            'Treatment': "class_weight='balanced' (Not supported for estimator)",
            'CV_PR_AUC_Mean': np.nan,
            'CV_PR_AUC_Std': np.nan,
            'CV_ROC_AUC_Mean': np.nan,
            'CV_ROC_AUC_Std': np.nan,
            'Test_PR_AUC': np.nan,
            'Test_ROC_AUC': np.nan,
            'Test_F1_at_0.50': np.nan
        })
        
    # Treatment 3: SMOTENC
    pipe_smotenc = ImbPipeline([
        ('preprocessor', create_preprocessor()),
        ('resampler', SMOTENC(categorical_features=cat_indices, random_state=random_state)),
        ('classifier', base_estimator)
    ])
    cv_pr_smote = cross_val_score(pipe_smotenc, X_train, y_train, cv=cv, scoring='average_precision', n_jobs=-1)
    cv_roc_smote = cross_val_score(pipe_smotenc, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    pipe_smotenc.fit(X_train, y_train)
    p_smote = pipe_smotenc.predict_proba(X_test)[:, 1]
    
    treatments.append({
        'Treatment': 'SMOTENC (Categorical-aware Oversampling)',
        'CV_PR_AUC_Mean': float(np.mean(cv_pr_smote)),
        'CV_PR_AUC_Std': float(np.std(cv_pr_smote)),
        'CV_ROC_AUC_Mean': float(np.mean(cv_roc_smote)),
        'CV_ROC_AUC_Std': float(np.std(cv_roc_smote)),
        'Test_PR_AUC': float(average_precision_score(y_test, p_smote)),
        'Test_ROC_AUC': float(roc_auc_score(y_test, p_smote)),
        'Test_F1_at_0.50': float(f1_score(y_test, (p_smote >= 0.50).astype(int), zero_division=0))
    })
    
    return pd.DataFrame(treatments)


def run_full_pipeline(
    data_path: str = "data/campaign_data.csv",
    models_dir: str = "models",
    reports_dir: str = "reports",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    End-to-end execution of data generation, EDA, training, tuning,
    evaluation, simulation, explainability, profiling, and report building.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Dataset Verification & Unit Testing
    print("\n" + "="*70)
    print("STEP 1: DATA VERIFICATION & UNIT TESTS")
    print("="*70)
    if not os.path.exists(data_path):
        print("Generating campaign dataset...")
        generate_campaign_dataset(output_path=data_path, random_state=random_state)
    unit_test_preprocessing()
    
    # 2. Exploratory Data Analysis
    print("\n" + "="*70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)
    eda_results = run_full_eda(data_path=data_path, output_dir=figures_dir)
    print(f"Target prevalence: {eda_results['target_balance_pct']:.2f}% responders")
    
    # 3. Stratified Train / Test Partitioning
    X_train, X_test, y_train, y_test = load_and_split_data(data_path, test_size=0.2, random_state=random_state)
    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)
    feature_names = get_feature_names(preprocessor)
    print(f"Output transformed features ({len(feature_names)}): {feature_names}")
    
    # 4. Train and Tune All 6 Models
    fitted_models, results_df, model_cv_metrics = train_and_evaluate_all_models(
        X_train, y_train, X_test, y_test, cv_folds=5, random_state=random_state
    )
    
    # Save results comparison table
    comparison_csv_path = os.path.join(reports_dir, "results_comparison.csv")
    results_df = results_df.sort_values(by=['CV_PR_AUC_Mean', 'CV_ROC_AUC_Mean'], ascending=[False, False]).reset_index(drop=True)
    results_df.to_csv(comparison_csv_path, index=False)
    print(f"\nComparative results saved to '{comparison_csv_path}':")
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC', 'FPR', 'FPR_at_Recall_70', 'CV_PR_AUC_Mean']].to_string(index=False))
    
    # 5. Defensible Model Selection
    winner_name, selection_justification = select_best_model_defensibly(results_df)
    print("\n" + "="*70)
    print(f"STEP 4: DEFENSIBLE MODEL SELECTION -> {winner_name}")
    print("="*70)
    print(selection_justification)
    best_model = fitted_models[winner_name]
    
    # 6. Imbalance Treatments Comparison
    print("\n" + "="*70)
    print("STEP 5: CLASS IMBALANCE TREATMENTS COMPARISON")
    print("="*70)
    base_est = get_model_grid_configs(random_state)[winner_name]['estimator']
    best_params = model_cv_metrics[winner_name]['best_params']
    clean_params = {k.replace('classifier__', ''): v for k, v in best_params.items()}
    base_est.set_params(**clean_params)
    
    imbalance_df = evaluate_imbalance_treatments(winner_name, base_est, X_train, y_train, X_test, y_test, cv_folds=5, random_state=random_state)
    imbalance_csv_path = os.path.join(reports_dir, "imbalance_handling_comparison.csv")
    imbalance_df.to_csv(imbalance_csv_path, index=False)
    print(imbalance_df.to_string(index=False))
    
    # 7. Leakage-Free OOF Threshold Optimization
    print("\n" + "="*70)
    print("STEP 6: OOF THRESHOLD OPTIMIZATION & BUSINESS SIMULATION")
    print("="*70)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_proba_winner = cross_val_predict(best_model, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
    
    threshold_results = find_optimal_thresholds_oof(
        y_train, oof_proba_winner,
        cost_per_contact=COST_PER_CONTACT,
        profit_per_responder=PROFIT_PER_RESPONDER,
        output_dir=figures_dir
    )
    f1_t = threshold_results['f1_optimal_threshold']
    profit_t = threshold_results['profit_optimal_threshold']
    print(f"OOF F1-Optimal Threshold:     {f1_t:.2f} (OOF F1: {threshold_results['f1_optimal_oof_score']:.4f})")
    print(f"OOF Profit-Optimal Threshold: {profit_t:.2f} (OOF Net Profit: ${threshold_results['profit_optimal_oof_profit']:,.0f})")
    
    # Evaluate 4-strategy simulation once on held-out test data
    y_test_proba_winner = best_model.predict_proba(X_test)[:, 1]
    sim_results = run_four_strategy_simulation(
        y_test, y_test_proba_winner,
        f1_threshold=f1_t,
        profit_threshold=profit_t,
        cost_per_contact=COST_PER_CONTACT,
        profit_per_responder=PROFIT_PER_RESPONDER,
        output_dir=figures_dir
    )
    sim_csv_path = os.path.join(reports_dir, "business_simulation.csv")
    sim_results['simulation_table'].to_csv(sim_csv_path, index=False)
    print("\nBusiness Simulation Results on Held-Out Test Data:")
    print(sim_results['simulation_table'].to_string(index=False))
    
    # 8. Probability Calibration Check
    print("\n" + "="*70)
    print("STEP 7: PROBABILITY CALIBRATION (FrozenEstimator)")
    print("="*70)
    calib_results = evaluate_probability_calibration(
        best_model, X_train, y_train, X_test, y_test,
        random_state=random_state, output_dir=figures_dir
    )
    print(f"Validation Brier Score: Uncalibrated = {calib_results['brier_val_uncalibrated']:.5f} vs Calibrated = {calib_results['brier_val_calibrated']:.5f}")
    print(f"Test Brier Score:       Uncalibrated = {calib_results['brier_test_uncalibrated']:.5f} vs Calibrated = {calib_results['brier_test_calibrated']:.5f}")
    print(f"Validation Brier Improved? {calib_results['val_improved']}")
    
    # Deterministic deployment policy:
    # If calibration wins on fair out-of-fold comparison, refit base pipeline on 100% of X_train,
    # then wrap in FrozenEstimator + CalibratedClassifierCV fit via CV on X_train;
    # otherwise refit best_model on 100% of X_train.
    if calib_results['val_improved']:
        best_model.fit(X_train, y_train)
        cv_cal = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        deployed_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(best_model),
            method='sigmoid',
            cv=cv_cal
        )
        deployed_model.fit(X_train, y_train)
        deployed_model_name = f"{winner_name} (Calibrated)"
    else:
        best_model.fit(X_train, y_train)
        deployed_model = best_model
        deployed_model_name = winner_name
        
    # 9. Bootstrap 95% Confidence Intervals
    print("\n" + "="*70)
    print("STEP 8: BOOTSTRAP 95% CONFIDENCE INTERVALS (Test Set)")
    print("="*70)
    test_proba_deployed = deployed_model.predict_proba(X_test)[:, 1]
    bootstrap_cis = compute_bootstrap_ci(y_test, test_proba_deployed, threshold=profit_t, n_bootstraps=1000, random_state=random_state)
    print(f"Test ROC-AUC: 95% CI [{bootstrap_cis['roc_auc']['ci_lower']:.4f}, {bootstrap_cis['roc_auc']['ci_upper']:.4f}]")
    print(f"Test PR-AUC:  95% CI [{bootstrap_cis['pr_auc']['ci_lower']:.4f}, {bootstrap_cis['pr_auc']['ci_upper']:.4f}]")
    print(f"Test F1:      95% CI [{bootstrap_cis['f1']['ci_lower']:.4f}, {bootstrap_cis['f1']['ci_upper']:.4f}]")
    
    # 10. Generate Benchmark Visualizations
    plot_metrics_comparison(results_df, output_dir=figures_dir)
    plot_roc_curves(fitted_models, X_test, y_test, output_dir=figures_dir)
    plot_precision_recall_curves(fitted_models, X_test, y_test, output_dir=figures_dir)
    oof_thresh_map = {name: model_cv_metrics[name]['oof_threshold'] for name in fitted_models}
    plot_confusion_matrices(fitted_models, oof_thresh_map, X_test, y_test, output_dir=figures_dir)
    plot_cumulative_gains_and_lift(deployed_model, X_test, y_test, output_dir=figures_dir)
    
    # 11. Feature Importance & SHAP
    print("\n" + "="*70)
    print("STEP 9: EXPLAINABILITY & PROFILING")
    print("="*70)
    tree_model = fitted_models['Gradient Boosting'] if 'Gradient Boosting' in fitted_models else fitted_models['Random Forest']
    lr_model = fitted_models['Logistic Regression']
    analyze_feature_importances(tree_model, lr_model, X_test, y_test, feature_names=feature_names, output_dir=figures_dir)
    
    # Customer Profiles (Actual vs Predicted)
    y_pred_deployed = (test_proba_deployed >= profit_t).astype(int)
    profile_df = compute_comprehensive_customer_profiles(X_test, y_test, y_pred_deployed)
    profile_csv_path = os.path.join(reports_dir, "customer_profiles.csv")
    profile_df.to_csv(profile_csv_path)
    print("\nCustomer Profiles (Actual vs Predicted Responders):")
    print(profile_df)
    
    # 12. Serialize Artifacts
    print("\n" + "="*70)
    print("STEP 10: SERIALIZING ARTIFACTS")
    print("="*70)
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    metrics_json_path = os.path.join(models_dir, "metrics.json")
    feature_list_path = os.path.join(models_dir, "feature_list.json")
    
    joblib.dump(deployed_model, best_model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    with open(feature_list_path, 'w') as f:
        json.dump({
            'raw_features': FEATURE_COLUMNS,
            'continuous_features': CONTINUOUS_FEATURES,
            'binary_features': BINARY_FEATURES,
            'categorical_features': CATEGORICAL_FEATURES,
            'transformed_features': feature_names,
            'target_column': TARGET_COLUMN
        }, f, indent=2)
        
    # Metrics JSON
    test_metrics_default = compute_metrics_at_threshold(y_test, test_proba_deployed, threshold=0.50)
    test_metrics_f1 = compute_metrics_at_threshold(y_test, test_proba_deployed, threshold=f1_t)
    test_metrics_profit = compute_metrics_at_threshold(y_test, test_proba_deployed, threshold=profit_t)
    
    metrics_summary = {
        'best_model_name': winner_name,
        'deployed_model_name': deployed_model_name,
        'selection_justification': selection_justification,
        'best_model_params': model_cv_metrics[winner_name]['best_params'],
        'f1_optimal_threshold': f1_t,
        'profit_optimal_threshold': profit_t,
        'default_threshold': 0.50,
        'test_metrics_default_threshold': test_metrics_default,
        'test_metrics_f1_threshold': test_metrics_f1,
        'test_metrics_profit_threshold': test_metrics_profit,
        'bootstrap_confidence_intervals_profit_threshold': bootstrap_cis,
        'cv_metrics': model_cv_metrics,
        'brier_scores': {
            'validation_uncalibrated': calib_results['brier_val_uncalibrated'],
            'validation_calibrated': calib_results['brier_val_calibrated'],
            'test_uncalibrated': calib_results['brier_test_uncalibrated'],
            'test_calibrated': calib_results['brier_test_calibrated'],
            'validation_improved': calib_results['val_improved']
        },
        'economic_parameters': {
            'cost_per_contact': COST_PER_CONTACT,
            'profit_per_responder': PROFIT_PER_RESPONDER,
            'break_even_probability': COST_PER_CONTACT / PROFIT_PER_RESPONDER
        },
        'business_simulation': sim_results['simulation_table'].to_dict(orient='records'),
        'logistic_regression_odds_ratios': {
            feat: float(np.exp(coef))
            for feat, coef in zip(feature_names, fitted_models['Logistic Regression'].named_steps['classifier'].coef_[0])
        }
    }
    
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
        
    print(f"Saved deployed model: {best_model_path}")
    print(f"Saved preprocessor:   {preprocessor_path}")
    print(f"Saved feature list:   {feature_list_path}")
    print(f"Saved metrics json:   {metrics_json_path}")
    
    # 13. Dynamic Report Generation
    print("\n" + "="*70)
    print("STEP 11: REGENERATING REPORTS DYNAMICALLY FROM METRICS")
    print("="*70)
    from src.build_report import generate_final_report, generate_readme
    generate_final_report()
    generate_readme()
    print("Reports regenerated successfully.")
    
    return metrics_summary


if __name__ == "__main__":
    run_full_pipeline()
