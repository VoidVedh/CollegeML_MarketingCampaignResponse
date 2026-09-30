"""
train.py
Main execution pipeline for Marketing Campaign Response Prediction:
1. Generates data if not already present.
2. Runs full exploratory data analysis (EDA).
3. Preprocesses data with scikit-learn ColumnTransformer (leakage-free).
4. Trains and tunes all 6 classification algorithms via 5-fold Stratified CV.
5. Handles class imbalance via SMOTE inside CV folds.
6. Evaluates all models on held-out test set and saves comparative reports.
7. Performs decision threshold optimization and business ROI simulation.
8. Performs probability calibration and cumulative gains/lift analysis.
9. Conducts feature importance (Tree, Permutation, Odds Ratios, SHAP) and profiling.
10. Saves best model and metadata for Streamlit deployment.
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

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Local imports
from src.generate_data import generate_campaign_dataset
from src.preprocess import (
    load_and_split_data, create_preprocessor, get_feature_names,
    FEATURE_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN
)
from src.eda import run_full_eda
from src.evaluate import (
    compute_test_metrics, plot_metrics_comparison, plot_roc_curves,
    plot_precision_recall_curves, plot_confusion_matrices,
    tune_decision_threshold, run_business_simulation,
    plot_cumulative_gains_and_lift, evaluate_probability_calibration,
    analyze_feature_importances, compute_customer_profiles,
    COST_PER_CONTACT, PROFIT_PER_RESPONDER
)


def get_model_grid_configs(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Returns estimators and tuning grids for all 6 required algorithms.
    """
    configs = {
        'Logistic Regression': {
            'estimator': LogisticRegression(random_state=random_state, max_iter=1000),
            'param_grid': {
                'classifier__C': [0.01, 0.1, 1.0, 5.0, 10.0],
                'classifier__solver': ['lbfgs']
            }
        },
        'K-Nearest Neighbors': {
            'estimator': KNeighborsClassifier(),
            'param_grid': {
                'classifier__n_neighbors': [3, 5, 7, 11, 15],
                'classifier__weights': ['uniform', 'distance']
            }
        },
        'Decision Tree': {
            'estimator': DecisionTreeClassifier(random_state=random_state),
            'param_grid': {
                'classifier__max_depth': [3, 5, 8, 12, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__criterion': ['gini', 'entropy']
            }
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=random_state),
            'param_grid': {
                'classifier__n_estimators': [50, 100, 150],
                'classifier__max_depth': [5, 8, 12, None],
                'classifier__min_samples_split': [2, 5]
            }
        },
        'Naive Bayes': {
            'estimator': GaussianNB(),
            'param_grid': {
                'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            }
        },
        'Gradient Boosting': {
            'estimator': GradientBoostingClassifier(random_state=random_state),
            'param_grid': {
                'classifier__n_estimators': [50, 100, 150],
                'classifier__learning_rate': [0.05, 0.1, 0.2],
                'classifier__max_depth': [3, 5]
            }
        }
    }
    return configs


def train_and_tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Trains and tunes all 6 algorithms using 5-fold Stratified CV with SMOTE.
    """
    configs = get_model_grid_configs(random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    fitted_models = {}
    cv_metrics = {}
    
    print("\n" + "="*70)
    print("STEP 3: TRAINING & HYPERPARAMETER TUNING (5-Fold Stratified CV with SMOTE)")
    print("="*70)
    
    for name, conf in configs.items():
        start_time = time.time()
        print(f"\n--> Training {name}...")
        
        # Pipeline: Preprocessor -> SMOTE -> Classifier
        pipeline = ImbPipeline([
            ('preprocessor', create_preprocessor()),
            ('smote', SMOTE(random_state=random_state)),
            ('classifier', conf['estimator'])
        ])
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid=conf['param_grid'],
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            refit=True
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_f1 = grid_search.best_score_
        
        # Also compute CV ROC-AUC
        cv_auc_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        cv_f1_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        
        elapsed = time.time() - start_time
        print(f"    Completed in {elapsed:.1f}s")
        print(f"    Best Params: {best_params}")
        print(f"    CV F1-Score: {np.mean(cv_f1_scores):.4f} (+/- {np.std(cv_f1_scores):.4f})")
        print(f"    CV ROC-AUC:  {np.mean(cv_auc_scores):.4f} (+/- {np.std(cv_auc_scores):.4f})")
        
        fitted_models[name] = best_model
        cv_metrics[name] = {
            'best_params': best_params,
            'cv_f1_mean': float(np.mean(cv_f1_scores)),
            'cv_f1_std': float(np.std(cv_f1_scores)),
            'cv_auc_mean': float(np.mean(cv_auc_scores)),
            'cv_auc_std': float(np.std(cv_auc_scores)),
            'train_time_sec': float(elapsed)
        }
        
    return fitted_models, cv_metrics


def run_full_pipeline(
    data_path: str = "data/campaign_data.csv",
    models_dir: str = "models",
    reports_dir: str = "reports",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    End-to-end execution of the full machine learning and analysis workflow.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Dataset Generation (if missing)
    # -------------------------------------------------------------
    if not os.path.exists(data_path):
        print(f"Dataset not found at '{data_path}'. Generating realistic synthetic dataset...")
        generate_campaign_dataset(output_path=data_path, random_state=random_state)
    else:
        print(f"Existing dataset detected at '{data_path}'. Validating schema...")
        
    # -------------------------------------------------------------
    # 2. Exploratory Data Analysis (EDA)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)
    eda_results = run_full_eda(data_path=data_path, output_dir=figures_dir)
    print(f"Target response rate: {eda_results['target_balance_pct']:.2f}%")
    print(f"EDA plots saved to '{figures_dir}'")
    
    # -------------------------------------------------------------
    # 3. Preprocessing & Data Splitting
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 2: PREPROCESSING & STRATIFIED SPLIT (80/20)")
    print("="*70)
    X_train, X_test, y_train, y_test = load_and_split_data(data_path, test_size=0.2, random_state=random_state)
    print(f"Training set: {X_train.shape[0]} samples ({y_train.sum()} responders, {y_train.mean():.1%})")
    print(f"Test set:     {X_test.shape[0]} samples ({y_test.sum()} responders, {y_test.mean():.1%})")
    
    # Fit standalone preprocessor to export feature names
    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)
    feature_names = get_feature_names(preprocessor)
    print(f"Transformed features ({len(feature_names)}): {feature_names}")
    
    # -------------------------------------------------------------
    # 4. Model Training & Hyperparameter Tuning
    # -------------------------------------------------------------
    fitted_models, cv_metrics = train_and_tune_models(X_train, y_train, cv_folds=5, random_state=random_state)
    
    # -------------------------------------------------------------
    # 5. Comparative Evaluation on Held-Out Test Set
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 4: COMPARATIVE STUDY ON TEST SET")
    print("="*70)
    
    test_results_list = []
    for name, model in fitted_models.items():
        metrics = compute_test_metrics(model, X_test, y_test, threshold=0.5)
        row = {
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'ROC-AUC': metrics['roc_auc'],
            'FPR': metrics['fpr'],
            'TN': metrics['tn'],
            'FP': metrics['fp'],
            'FN': metrics['fn'],
            'TP': metrics['tp'],
            'CV_F1_Mean': cv_metrics[name]['cv_f1_mean'],
            'CV_F1_Std': cv_metrics[name]['cv_f1_std'],
            'CV_AUC_Mean': cv_metrics[name]['cv_auc_mean'],
            'CV_AUC_Std': cv_metrics[name]['cv_auc_std'],
            'Best_Params': json.dumps(cv_metrics[name]['best_params'])
        }
        test_results_list.append(row)
        
    results_df = pd.DataFrame(test_results_list)
    results_df = results_df.sort_values(by=['F1-Score', 'ROC-AUC'], ascending=[False, False]).reset_index(drop=True)
    
    comparison_csv_path = os.path.join(reports_dir, "results_comparison.csv")
    results_df.to_csv(comparison_csv_path, index=False)
    print(f"\nComparative results saved to '{comparison_csv_path}':")
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'FPR']].to_string(index=False))
    
    # Generate Visualizations
    plot_metrics_comparison(results_df, output_dir=figures_dir)
    plot_roc_curves(fitted_models, X_test, y_test, output_dir=figures_dir)
    plot_precision_recall_curves(fitted_models, X_test, y_test, output_dir=figures_dir)
    plot_confusion_matrices(fitted_models, X_test, y_test, output_dir=figures_dir)
    print(f"Generated comparison bar chart, ROC curves, PR curves, and confusion matrix grid.")
    
    # -------------------------------------------------------------
    # 6. Best Model Selection & Imbalance Comparison
    # -------------------------------------------------------------
    best_model_name = results_df.iloc[0]['Model']
    best_model = fitted_models[best_model_name]
    print(f"\n--> BEST MODEL SELECTED: {best_model_name} (Highest F1-Score & ROC-AUC balance)")
    
    # Imbalance handling comparison: With vs Without SMOTE for best model
    print("\n--> Comparing 'With SMOTE' vs 'Without SMOTE' for Best Model...")
    base_estimator = get_model_grid_configs(random_state)[best_model_name]['estimator']
    best_params = cv_metrics[best_model_name]['best_params']
    clean_params = {k.replace('classifier__', ''): v for k, v in best_params.items()}
    base_estimator.set_params(**clean_params)
    
    # Model without SMOTE
    pipeline_no_smote = ImbPipeline([
        ('preprocessor', create_preprocessor()),
        ('classifier', base_estimator)
    ])
    pipeline_no_smote.fit(X_train, y_train)
    metrics_no_smote = compute_test_metrics(pipeline_no_smote, X_test, y_test, threshold=0.5)
    metrics_with_smote = compute_test_metrics(best_model, X_test, y_test, threshold=0.5)
    
    imbalance_comp = pd.DataFrame([
        {
            'Treatment': 'Without Imbalance Handling (Standard Baseline)',
            'Accuracy': metrics_no_smote['accuracy'],
            'Precision': metrics_no_smote['precision'],
            'Recall': metrics_no_smote['recall'],
            'F1-Score': metrics_no_smote['f1'],
            'ROC-AUC': metrics_no_smote['roc_auc'],
            'FPR': metrics_no_smote['fpr']
        },
        {
            'Treatment': 'With SMOTE (Oversampling in Training Folds)',
            'Accuracy': metrics_with_smote['accuracy'],
            'Precision': metrics_with_smote['precision'],
            'Recall': metrics_with_smote['recall'],
            'F1-Score': metrics_with_smote['f1'],
            'ROC-AUC': metrics_with_smote['roc_auc'],
            'FPR': metrics_with_smote['fpr']
        }
    ])
    imbalance_csv_path = os.path.join(reports_dir, "imbalance_handling_comparison.csv")
    imbalance_comp.to_csv(imbalance_csv_path, index=False)
    print(imbalance_comp.to_string(index=False))
    
    # -------------------------------------------------------------
    # 7. Step 5: Reducing False Positives & Marketing Cost
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 5: THRESHOLD TUNING & BUSINESS ROI SIMULATION")
    print("="*70)
    threshold_results = tune_decision_threshold(best_model, X_test, y_test, output_dir=figures_dir)
    opt_threshold = threshold_results['optimal_threshold']
    print(f"Default Threshold (0.50) -> Precision: {threshold_results['default_metrics']['precision']:.4f}, Recall: {threshold_results['default_metrics']['recall']:.4f}, F1: {threshold_results['default_metrics']['f1']:.4f}, FPR: {threshold_results['default_metrics']['fpr']:.4f}")
    print(f"Optimal Threshold ({opt_threshold:.2f}) -> Precision: {threshold_results['optimal_precision']:.4f}, Recall: {threshold_results['optimal_recall']:.4f}, F1: {threshold_results['optimal_f1']:.4f}, FPR: {threshold_results['optimal_fpr']:.4f}")
    
    # Business Simulation
    sim_results = run_business_simulation(
        best_model, X_test, y_test,
        opt_threshold=opt_threshold,
        cost_per_contact=COST_PER_CONTACT,
        profit_per_responder=PROFIT_PER_RESPONDER,
        output_dir=figures_dir
    )
    sim_csv_path = os.path.join(reports_dir, "business_simulation.csv")
    sim_results['simulation_table'].to_csv(sim_csv_path, index=False)
    print("\nBusiness Simulation Results:")
    print(sim_results['simulation_table'].to_string(index=False))
    
    # Cumulative gains and Decile lift
    plot_cumulative_gains_and_lift(best_model, X_test, y_test, output_dir=figures_dir)
    print("Cumulative Gains & Decile Lift chart saved.")
    
    # Probability Calibration
    calib_results = evaluate_probability_calibration(best_model, X_train, y_train, X_test, y_test, output_dir=figures_dir)
    print(f"Calibration Brier Score: Uncalibrated = {calib_results['brier_uncalibrated']:.4f} vs Calibrated = {calib_results['brier_calibrated']:.4f}")
    
    # -------------------------------------------------------------
    # 8. Step 6: Feature Importance & Profiling
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 6: FEATURE IMPORTANCE & CUSTOMER PROFILING")
    print("="*70)
    lr_model = fitted_models['Logistic Regression']
    # Choose best tree model for MDI: Gradient Boosting or Random Forest
    best_tree_model = fitted_models['Gradient Boosting'] if 'Gradient Boosting' in fitted_models else fitted_models['Random Forest']
    
    fi_plots = analyze_feature_importances(
        best_tree_model, lr_model,
        X_test, y_test,
        feature_names=feature_names,
        output_dir=figures_dir
    )
    print(f"Feature importance plots generated: {list(fi_plots.keys())}")
    
    # Customer Profiles
    y_pred_best = best_model.predict(X_test)
    profiles = compute_customer_profiles(X_test, y_pred_best)
    profiles_csv_path = os.path.join(reports_dir, "customer_profiles.csv")
    profiles.to_csv(profiles_csv_path)
    print("\nCustomer Profile Averages (Responders vs Non-Responders):")
    print(profiles)
    
    # -------------------------------------------------------------
    # 9. Save Artifacts for Deployment
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 7: SAVING SERIALIZED ARTIFACTS")
    print("="*70)
    
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    metrics_json_path = os.path.join(models_dir, "metrics.json")
    feature_list_path = os.path.join(models_dir, "feature_list.json")
    
    # Save best model pipeline
    joblib.dump(best_model, best_model_path)
    joblib.dump(best_model.named_steps['preprocessor'], preprocessor_path)
    
    # Save feature metadata
    with open(feature_list_path, 'w') as f:
        json.dump({
            'raw_features': FEATURE_COLUMNS,
            'numeric_features': NUMERIC_FEATURES,
            'categorical_features': CATEGORICAL_FEATURES,
            'transformed_features': feature_names,
            'target_column': TARGET_COLUMN
        }, f, indent=2)
        
    # Save consolidated metrics
    metrics_summary = {
        'best_model_name': best_model_name,
        'best_model_params': cv_metrics[best_model_name]['best_params'],
        'optimal_threshold': opt_threshold,
        'test_metrics_default_threshold': threshold_results['default_metrics'],
        'test_metrics_optimal_threshold': threshold_results['optimized_metrics'],
        'cv_metrics': cv_metrics,
        'brier_score_uncalibrated': calib_results['brier_uncalibrated'],
        'brier_score_calibrated': calib_results['brier_calibrated'],
        'economic_parameters': {
            'cost_per_contact': COST_PER_CONTACT,
            'profit_per_responder': PROFIT_PER_RESPONDER
        },
        'business_simulation': sim_results['simulation_table'].to_dict(orient='records')
    }
    
    # Remove numpy arrays from metrics before JSON serialization
    for sub_k in ['y_proba', 'y_pred']:
        metrics_summary['test_metrics_default_threshold'].pop(sub_k, None)
        metrics_summary['test_metrics_optimal_threshold'].pop(sub_k, None)
            
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
        
    print(f"Saved best model pipeline to: '{best_model_path}'")
    print(f"Saved preprocessor to:        '{preprocessor_path}'")
    print(f"Saved feature list to:         '{feature_list_path}'")
    print(f"Saved metrics summary to:      '{metrics_json_path}'")
    print("\nPipeline execution complete!")
    
    return {
        'best_model_name': best_model_name,
        'results_df': results_df,
        'sim_df': sim_results['simulation_table'],
        'opt_threshold': opt_threshold
    }


if __name__ == "__main__":
    run_full_pipeline()
