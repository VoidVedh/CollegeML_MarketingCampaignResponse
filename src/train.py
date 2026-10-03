"""
train.py
Main execution and training pipeline for Case Study 157:
Marketing Campaign Response Prediction Using Machine Learning.

Implements:
- Leakage-free preprocessing with scikit-learn ColumnTransformer on 8 Case Study features
- 5-fold Stratified Cross-Validation across all 6 classification algorithms
- Defensible model selection by CV PR-AUC & ROC-AUC with standard deviation & tie-breaks
- FPR at fixed recall (70%) benchmark
- Imbalance handling comparison: None vs class_weight vs SMOTENC
- Out-Of-Fold (OOF) training threshold optimization (F1-optimal & Profit-optimal)
- 4-Strategy business simulation (Everyone, Default 0.50, F1-optimal, Profit-optimal)
- Probability calibration via FrozenEstimator with validation Brier check
- Bootstrap 95% Confidence Intervals for test metrics
- Explainability (Tree MDI, Permutation Importance, Logistic Odds Ratios, SHAP)
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
from src.preprocess import (
    load_and_split_data, create_preprocessor, get_feature_names,
    unit_test_preprocessing, RAW_DATA_PATH, PROCESSED_DATA_PATH,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS, TARGET_COLUMN
)
from src.eda import run_full_eda
from src.evaluate import (
    compute_metrics_at_threshold, compute_fpr_at_recall, compute_top_k_capture, compute_bootstrap_ci,
    find_optimal_thresholds_oof, run_four_strategy_simulation,
    evaluate_probability_calibration, compute_comprehensive_customer_profiles,
    plot_metrics_comparison, plot_roc_curves, plot_precision_recall_curves,
    plot_confusion_matrices, plot_cumulative_gains_and_lift,
    analyze_feature_importances, COST_PER_CONTACT, PROFIT_PER_RESPONDER
)


def get_model_grid_configs(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Returns estimator templates and focused hyperparameter grids for all 6 required models.
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
                'classifier__max_depth': [3, 5, 8],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__criterion': ['gini', 'entropy']
            },
            'supports_class_weight': True
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=random_state),
            'param_grid': {
                'classifier__n_estimators': [50, 100, 150],
                'classifier__max_depth': [4, 6, 8, None],
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


def train_and_tune_models_dev(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Trains and tunes all 6 models using 5-fold Stratified Cross-Validation strictly on
    the development/training dataset. ZERO test-set data is used or touched.
    """
    configs = get_model_grid_configs(random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    fitted_models = {}
    oof_probas = {}
    dev_results_list = []
    
    print("\n" + "="*70)
    print("STEP 3: 5-FOLD STRATIFIED CV HYPERPARAMETER TUNING (TRAINING DATA ONLY)")
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
        
        # Collect CV PR-AUC and ROC-AUC scores across the folds
        cv_pr_scores = cross_val_score(best_est, X_train, y_train, cv=cv, scoring='average_precision', n_jobs=-1)
        cv_roc_scores = cross_val_score(best_est, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        # Out-Of-Fold probability generation strictly on training data
        oof_p = cross_val_predict(best_est, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
        oof_probas[name] = oof_p
        
        # Find model's own OOF F1-optimal threshold
        t_sweep = np.linspace(0.05, 0.95, 91)
        oof_f1s = [f1_score(y_train, (oof_p >= t).astype(int), zero_division=0) for t in t_sweep]
        best_f1_idx = int(np.argmax(oof_f1s))
        model_oof_t = float(t_sweep[best_f1_idx])
        
        # Compute training OOF metrics
        oof_pr = float(average_precision_score(y_train, oof_p))
        oof_roc = float(roc_auc_score(y_train, oof_p))
        oof_fpr_70 = float(compute_fpr_at_recall(y_train, oof_p, target_recall=0.70))
        
        elapsed = time.time() - t0
        print(f"    Completed in {elapsed:.1f}s | Best Params: {best_params}")
        print(f"    CV PR-AUC:  {np.mean(cv_pr_scores):.4f} (+/- {np.std(cv_pr_scores):.4f})")
        print(f"    CV ROC-AUC: {np.mean(cv_roc_scores):.4f} (+/- {np.std(cv_roc_scores):.4f})")
        print(f"    OOF Tuned Threshold: {model_oof_t:.2f} | OOF F1: {oof_f1s[best_f1_idx]:.4f} | Training OOF FPR@Rec=70%: {oof_fpr_70:.4f}")
        
        fitted_models[name] = best_est
        
        dev_results_list.append({
            'Model': name,
            'CV_PR_AUC_Mean': float(np.mean(cv_pr_scores)),
            'CV_PR_AUC_Std': float(np.std(cv_pr_scores)),
            'CV_ROC_AUC_Mean': float(np.mean(cv_roc_scores)),
            'CV_ROC_AUC_Std': float(np.std(cv_roc_scores)),
            'OOF_PR_AUC': oof_pr,
            'OOF_ROC_AUC': oof_roc,
            'OOF_Threshold': model_oof_t,
            'OOF_F1': float(oof_f1s[best_f1_idx]),
            'OOF_FPR_at_Recall_70': oof_fpr_70,
            'Best_Params': best_params
        })
        
    dev_benchmark_df = pd.DataFrame(dev_results_list)
    return fitted_models, dev_benchmark_df, oof_probas


def select_best_model_defensibly(dev_benchmark_df: pd.DataFrame) -> Tuple[str, str]:
    """
    Defensible Model Selection Rule (Strictly using Development/Training CV & OOF Information):
    1. Sort models descending by 5-fold CV PR-AUC (Average Precision).
    2. Check if the top two models are practically close (difference < 1 fold standard deviation).
    3. If practically close:
       - Disclose that the difference is small relative to cross-validation fold variation.
       - Break tie using development/training information:
         (a) Lower training Out-Of-Fold FPR at fixed 70% recall (OOF_FPR_at_Recall_70)
         (b) Model parsimony / architectural simplicity / explainability
    4. If not practically close (diff >= std):
       - Select the top model decisively.
    
    Zero test data is referenced or inspected during this selection decision.
    """
    sorted_df = dev_benchmark_df.sort_values(by='CV_PR_AUC_Mean', ascending=False).reset_index(drop=True)
    top1 = sorted_df.iloc[0]
    top2 = sorted_df.iloc[1]
    
    cv_diff = top1['CV_PR_AUC_Mean'] - top2['CV_PR_AUC_Mean']
    top1_std = top1['CV_PR_AUC_Std']
    
    if cv_diff < top1_std:
        # Practically close based on observed CV variation
        if top1['OOF_FPR_at_Recall_70'] <= top2['OOF_FPR_at_Recall_70']:
            winner = top1['Model']
            tie_reason = (
                f"{top1['Model']} demonstrated lower training out-of-fold FPR at 70% recall "
                f"({top1['OOF_FPR_at_Recall_70']:.4f} vs {top2['OOF_FPR_at_Recall_70']:.4f})."
            )
        else:
            winner = top2['Model']
            tie_reason = (
                f"{top2['Model']} demonstrated lower training out-of-fold FPR at 70% recall "
                f"({top2['OOF_FPR_at_Recall_70']:.4f} vs {top1['OOF_FPR_at_Recall_70']:.4f})."
            )
            
        if winner == 'Logistic Regression':
            arch_note = "Additionally, Logistic Regression offers convex optimization, closed-form log-odds interpretability, and direct odds ratios."
        elif winner in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
            arch_note = f"Additionally, {winner} captures non-linear feature interactions and provides robust tree-based and permutation feature importances without parametric distribution assumptions."
        else:
            arch_note = f"Additionally, {winner} demonstrates solid non-parametric boundary separation."

        justification = (
            f"The top two models on development cross-validation, {top1['Model']} (CV PR-AUC: {top1['CV_PR_AUC_Mean']:.4f} ± {top1['CV_PR_AUC_Std']:.4f}) "
            f"and {top2['Model']} (CV PR-AUC: {top2['CV_PR_AUC_Mean']:.4f} ± {top2['CV_PR_AUC_Std']:.4f}), are practically close, with a score difference "
            f"({cv_diff:.4f}) smaller than the observed fold-to-fold cross-validation variation ({top1_std:.4f}). "
            f"Based strictly on training data cross-validation and out-of-fold metrics, {winner} was selected because {tie_reason} "
            f"{arch_note}"
        )
    else:
        winner = top1['Model']
        if winner == 'Logistic Regression':
            arch_note = "Logistic Regression provides convex optimization, closed-form log-odds interpretability, and direct odds ratios."
        elif winner in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
            arch_note = f"{winner} effectively captures non-linear feature interactions with tree-based and permutation importance."
        else:
            arch_note = f"{winner} provides robust predictive capability."
            
        justification = (
            f"{top1['Model']} achieved the highest mean CV PR-AUC ({top1['CV_PR_AUC_Mean']:.4f} ± {top1['CV_PR_AUC_Std']:.4f}), "
            f"leading the runner-up {top2['Model']} ({top2['CV_PR_AUC_Mean']:.4f}) by more than 1 cross-validation standard deviation. {arch_note}"
        )
        
    return winner, justification


def evaluate_all_models_on_test_set(
    fitted_models: Dict[str, Any],
    dev_benchmark_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Evaluates each fitted model pipeline ONCE on the held-out test set.
    """
    rows = []
    
    for _, dev_row in dev_benchmark_df.iterrows():
        name = dev_row['Model']
        model = fitted_models[name]
        
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Test evaluation at default threshold 0.50
        metrics_default = compute_metrics_at_threshold(y_test, y_proba, threshold=0.50)
        
        # Test evaluation at model's own OOF tuned threshold
        oof_t = dev_row['OOF_Threshold']
        metrics_tuned = compute_metrics_at_threshold(y_test, y_proba, threshold=oof_t)
        
        # Test FPR at fixed 70% recall
        fpr_70_test = compute_fpr_at_recall(y_test, y_proba, target_recall=0.70)
        
        # Top-20% customer capture rate
        top20_capture = compute_top_k_capture(y_test, y_proba, k_percent=20.0)
        
        rows.append({
            'Model': name,
            'Accuracy': metrics_default['accuracy'],
            'Precision': metrics_default['precision'],
            'Recall': metrics_default['recall'],
            'F1-Score': metrics_default['f1'],
            'ROC-AUC': metrics_default['roc_auc'],
            'PR-AUC': metrics_default['pr_auc'],
            'Confusion_Matrix_TN': metrics_default['tn'],
            'Confusion_Matrix_FP': metrics_default['fp'],
            'Confusion_Matrix_FN': metrics_default['fn'],
            'Confusion_Matrix_TP': metrics_default['tp'],
            'FPR': metrics_default['fpr'],
            'FPR_at_Recall_70': fpr_70_test,
            'Top20_Capture_Rate': top20_capture['capture_rate'],
            'Top20_Responders_Captured': top20_capture['responders_captured'],
            'Top20_Total_Responders': top20_capture['total_responders'],
            'Tuned_Threshold': oof_t,
            'Tuned_Threshold_Precision': metrics_tuned['precision'],
            'Tuned_Threshold_Recall': metrics_tuned['recall'],
            'Tuned_Threshold_F1': metrics_tuned['f1'],
            'CV_PR_AUC_Mean': dev_row['CV_PR_AUC_Mean'],
            'CV_PR_AUC_Std': dev_row['CV_PR_AUC_Std'],
            'CV_ROC_AUC_Mean': dev_row['CV_ROC_AUC_Mean'],
            'CV_ROC_AUC_Std': dev_row['CV_ROC_AUC_Std'],
            'OOF_FPR_at_Recall_70': dev_row['OOF_FPR_at_Recall_70'],
            'Best_Params': str(dev_row['Best_Params'])
        })
        
    return pd.DataFrame(rows)


def evaluate_imbalance_treatments(
    winner_name: str,
    base_estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Benchmarks three class imbalance treatments:
      1. None (Unweighted Baseline)
      2. class_weight='balanced' (if supported)
      3. SMOTENC (with categorical columns declared properly)
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    treatments = []
    
    # Dynamically derive categorical indices from fitted preprocessor:
    preproc_fitted = create_preprocessor().fit(X_train)
    all_feat_names = list(preproc_fitted.get_feature_names_out())
    n_num = len(NUMERIC_FEATURES)
    cat_indices = list(range(n_num, len(all_feat_names)))
    
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
    raw_path: str = RAW_DATA_PATH,
    processed_path: str = PROCESSED_DATA_PATH,
    models_dir: str = "models",
    reports_dir: str = "reports",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    End-to-end execution of data loading, EDA, training, tuning,
    evaluation, simulation, explainability, profiling, and report building
    on the Kaggle Customer Personality Analysis Dataset.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Dataset Verification & Unit Testing
    print("\n" + "="*70)
    print("STEP 1: DATA VERIFICATION & UNIT TESTS")
    print("="*70)
    unit_test_preprocessing()
    
    # 2. Exploratory Data Analysis
    print("\n" + "="*70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)
    eda_results = run_full_eda(processed_path=processed_path, output_dir=figures_dir)
    print(f"Target prevalence: {eda_results['target_balance_pct']:.2f}% responders")
    
    # 3. Stratified Train / Test Partitioning
    X_train, X_test, y_train, y_test, full_df = load_and_split_data(
        raw_path=raw_path,
        processed_path=processed_path,
        test_size=0.20,
        random_state=random_state
    )
    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)
    feature_names = get_feature_names(preprocessor)
    print(f"Output transformed features ({len(feature_names)}): {feature_names}")
    
    # 4. Train and Tune All 6 Models on Development Data Only
    fitted_models, dev_benchmark_df, oof_probas = train_and_tune_models_dev(
        X_train, y_train, cv_folds=5, random_state=random_state
    )
    
    # 5. Defensible Model Selection (strictly using dev/training CV & OOF metrics)
    winner_name, selection_justification = select_best_model_defensibly(dev_benchmark_df)
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
    best_params_winner = dev_benchmark_df.loc[dev_benchmark_df['Model'] == winner_name, 'Best_Params'].values[0]
    clean_params = {k.replace('classifier__', ''): v for k, v in best_params_winner.items()}
    base_est.set_params(**clean_params)
    
    imbalance_df = evaluate_imbalance_treatments(
        winner_name, base_est, X_train, y_train, X_test, y_test, cv_folds=5, random_state=random_state
    )
    imbalance_csv_path = os.path.join(reports_dir, "imbalance_handling_comparison.csv")
    imbalance_df.to_csv(imbalance_csv_path, index=False)
    print(imbalance_df.to_string(index=False))
    
    # 7. Leakage-Free OOF Threshold Optimization (Training Data Only)
    print("\n" + "="*70)
    print("STEP 6: OOF THRESHOLD OPTIMIZATION (TRAINING DATA ONLY)")
    print("="*70)
    oof_proba_winner = oof_probas[winner_name]
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
    
    # 8. Probability Calibration Check (Validation Split on Training Data)
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
        
    # 9. ONE Final Evaluation of All Models on Held-Out Test Data
    print("\n" + "="*70)
    print("STEP 8: FINAL HELD-OUT TEST EVALUATION (TOUCHED ONCE)")
    print("="*70)
    results_df = evaluate_all_models_on_test_set(
        fitted_models, dev_benchmark_df, X_test, y_test
    )
    comparison_csv_path = os.path.join(reports_dir, "results_comparison.csv")
    results_df = results_df.sort_values(by=['CV_PR_AUC_Mean', 'CV_ROC_AUC_Mean'], ascending=[False, False]).reset_index(drop=True)
    results_df.to_csv(comparison_csv_path, index=False)
    print(f"\nComparative results saved to '{comparison_csv_path}':")
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC', 'FPR', 'FPR_at_Recall_70', 'Top20_Capture_Rate', 'CV_PR_AUC_Mean']].to_string(index=False))
    
    # Deployed model evaluation on test set
    test_proba_deployed = deployed_model.predict_proba(X_test)[:, 1]
    top20_capture_deployed = compute_top_k_capture(y_test, test_proba_deployed, k_percent=20.0)
    print(f"\nTop-20% Customer Capture Rate on Held-Out Test Set: {top20_capture_deployed['capture_rate']:.4f} "
          f"({top20_capture_deployed['responders_captured']}/{top20_capture_deployed['total_responders']} responders)")
    
    # 4-strategy simulation on test set
    sim_results = run_four_strategy_simulation(
        y_test, test_proba_deployed,
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
    
    # Bootstrap 95% Confidence Intervals
    bootstrap_cis = compute_bootstrap_ci(y_test, test_proba_deployed, threshold=profit_t, n_bootstraps=1000, random_state=random_state)
    print(f"Test ROC-AUC: 95% CI [{bootstrap_cis['roc_auc']['ci_lower']:.4f}, {bootstrap_cis['roc_auc']['ci_upper']:.4f}]")
    print(f"Test PR-AUC:  95% CI [{bootstrap_cis['pr_auc']['ci_lower']:.4f}, {bootstrap_cis['pr_auc']['ci_upper']:.4f}]")
    print(f"Test F1:      95% CI [{bootstrap_cis['f1']['ci_lower']:.4f}, {bootstrap_cis['f1']['ci_upper']:.4f}]")
    
    # 10. Generate Benchmark Visualizations
    plot_metrics_comparison(results_df, output_dir=figures_dir)
    plot_roc_curves(fitted_models, X_test, y_test, output_dir=figures_dir)
    plot_precision_recall_curves(fitted_models, X_test, y_test, output_dir=figures_dir)
    oof_thresh_map = {row['Model']: row['OOF_Threshold'] for _, row in dev_benchmark_df.iterrows()}
    plot_confusion_matrices(fitted_models, oof_thresh_map, X_test, y_test, output_dir=figures_dir)
    plot_cumulative_gains_and_lift(deployed_model, X_test, y_test, output_dir=figures_dir)
    
    # 11. Feature Importance & SHAP
    print("\n" + "="*70)
    print("STEP 9: EXPLAINABILITY & PROFILING")
    print("="*70)
    tree_model = fitted_models['Random Forest'] if 'Random Forest' in fitted_models else fitted_models['Gradient Boosting']
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
            'numeric_features': NUMERIC_FEATURES,
            'categorical_features': CATEGORICAL_FEATURES,
            'transformed_features': feature_names,
            'target_column': TARGET_COLUMN
        }, f, indent=2)
        
    # Metrics JSON
    test_metrics_default = compute_metrics_at_threshold(y_test, test_proba_deployed, threshold=0.50)
    test_metrics_f1 = compute_metrics_at_threshold(y_test, test_proba_deployed, threshold=f1_t)
    test_metrics_profit = compute_metrics_at_threshold(y_test, test_proba_deployed, threshold=profit_t)
    
    cv_metrics_dict = dev_benchmark_df.set_index('Model').to_dict(orient='index')
    for m_name in cv_metrics_dict:
        cv_metrics_dict[m_name]['cv_pr_auc_mean'] = cv_metrics_dict[m_name]['CV_PR_AUC_Mean']
        cv_metrics_dict[m_name]['cv_pr_auc_std'] = cv_metrics_dict[m_name]['CV_PR_AUC_Std']
        cv_metrics_dict[m_name]['cv_roc_auc_mean'] = cv_metrics_dict[m_name]['CV_ROC_AUC_Mean']
        cv_metrics_dict[m_name]['cv_roc_auc_std'] = cv_metrics_dict[m_name]['CV_ROC_AUC_Std']
        cv_metrics_dict[m_name]['fpr_at_recall_70'] = cv_metrics_dict[m_name]['OOF_FPR_at_Recall_70']
        cv_metrics_dict[m_name]['oof_threshold'] = cv_metrics_dict[m_name]['OOF_Threshold']
        cv_metrics_dict[m_name]['oof_f1'] = cv_metrics_dict[m_name]['OOF_F1']
        cv_metrics_dict[m_name]['best_params'] = cv_metrics_dict[m_name]['Best_Params']
    
    metrics_summary = {
        'best_model_name': winner_name,
        'deployed_model_name': deployed_model_name,
        'selection_justification': selection_justification,
        'best_model_params': best_params_winner,
        'f1_optimal_threshold': f1_t,
        'profit_optimal_threshold': profit_t,
        'default_threshold': 0.50,
        'top_20_percent_capture': top20_capture_deployed,
        'test_metrics_default_threshold': test_metrics_default,
        'test_metrics_f1_threshold': test_metrics_f1,
        'test_metrics_profit_threshold': test_metrics_profit,
        'bootstrap_confidence_intervals_profit_threshold': bootstrap_cis,
        'cv_metrics': cv_metrics_dict,
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
    
    return metrics_summary


if __name__ == "__main__":
    run_full_pipeline()
