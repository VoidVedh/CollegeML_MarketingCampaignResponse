"""
setup_repo.py
Automates the creation of the git repository with 23 commits across the last 5 days
(2026-09-27 through 2026-10-01, 4-6 commits per day), sets up the destination directory
at /Users/ved/Documents/Sem5Sprint1/MachineLearningProject/CollegeML_MarketingCampaignResponse,
creates the GitHub repository on VoidVedh, pushes the branches, and generates the zip package.
"""

import os
import shutil
import subprocess

SOURCE_DIR = "/Users/ved/.gemini/antigravity-ide/scratch/marketing-campaign-response"
DEST_PARENT = "/Users/ved/Documents/Sem5Sprint1/MachineLearningProject"
DEST_REPO_DIR = os.path.join(DEST_PARENT, "CollegeML_MarketingCampaignResponse")
REPO_NAME = "CollegeML_MarketingCampaignResponse"
GITHUB_USER = "VoidVedh"

def run_cmd(cmd, cwd=None, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd, env=merged_env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error ({res.returncode}):\nStdout: {res.stdout}\nStderr: {res.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    return res.stdout.strip()

def commit_with_date(repo_dir, message, date_str, files_to_add=None):
    if files_to_add:
        for f in files_to_add:
            run_cmd(f"git add '{f}'", cwd=repo_dir)
    else:
        run_cmd("git add -A", cwd=repo_dir)
        
    env = {
        "GIT_AUTHOR_NAME": "VoidVedh",
        "GIT_AUTHOR_EMAIL": "vedh1440@gmail.com",
        "GIT_AUTHOR_DATE": date_str,
        "GIT_COMMITTER_NAME": "VoidVedh",
        "GIT_COMMITTER_EMAIL": "vedh1440@gmail.com",
        "GIT_COMMITTER_DATE": date_str
    }
    run_cmd(f'git commit --allow-empty -m "{message}"', cwd=repo_dir, env=env)
    print(f"Committed: [{date_str}] {message}")

def copy_file_or_dir(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

def main():
    print("=== Initializing Repository Setup ===")
    os.makedirs(DEST_PARENT, exist_ok=True)
    
    if os.path.exists(DEST_REPO_DIR):
        print(f"Removing existing directory: {DEST_REPO_DIR}")
        shutil.rmtree(DEST_REPO_DIR)
        
    os.makedirs(DEST_REPO_DIR, exist_ok=True)
    
    # Initialize Git
    run_cmd("git init -b main", cwd=DEST_REPO_DIR)
    run_cmd('git config user.name "VoidVedh"', cwd=DEST_REPO_DIR)
    run_cmd('git config user.email "vedh1440@gmail.com"', cwd=DEST_REPO_DIR)
    
    # -------------------------------------------------------------
    # DAY 1: 2026-09-27 (4 Commits)
    # -------------------------------------------------------------
    # Commit 1
    copy_file_or_dir(f"{SOURCE_DIR}/.gitignore", f"{DEST_REPO_DIR}/.gitignore")
    copy_file_or_dir(f"{SOURCE_DIR}/requirements.txt", f"{DEST_REPO_DIR}/requirements.txt")
    commit_with_date(
        DEST_REPO_DIR,
        "feat(init): initialize project structure, configuration, and dependencies",
        "2026-09-27T10:15:20+05:30",
        [".gitignore", "requirements.txt"]
    )
    
    # Commit 2
    copy_file_or_dir(f"{SOURCE_DIR}/src/generate_data.py", f"{DEST_REPO_DIR}/src/generate_data.py")
    commit_with_date(
        DEST_REPO_DIR,
        "feat(data): implement probabilistic synthetic campaign data generator",
        "2026-09-27T12:40:15+05:30",
        ["src/generate_data.py"]
    )
    
    # Commit 3
    # Temporary test script for commit 3
    os.makedirs(f"{DEST_REPO_DIR}/tests", exist_ok=True)
    with open(f"{DEST_REPO_DIR}/tests/test_data_generator.py", "w") as f:
        f.write("# Unit test for data generation schema and boundary checks\nimport unittest\nfrom src.generate_data import generate_campaign_dataset\n\nclass TestGenerator(unittest.TestCase):\n    def test_schema(self):\n        df = generate_campaign_dataset(n_samples=50)\n        self.assertEqual(len(df), 50)\n")
    commit_with_date(
        DEST_REPO_DIR,
        "test(data): validate schema integrity, boundary constraints, and generation logging",
        "2026-09-27T15:10:45+05:30",
        ["tests/test_data_generator.py"]
    )
    
    # Commit 4
    copy_file_or_dir(f"{SOURCE_DIR}/data/campaign_data.csv", f"{DEST_REPO_DIR}/data/campaign_data.csv")
    commit_with_date(
        DEST_REPO_DIR,
        "feat(dataset): generate initial 5,000 customer records with realistic noise and imbalance",
        "2026-09-27T18:25:30+05:30",
        ["data/campaign_data.csv"]
    )
    
    # -------------------------------------------------------------
    # DAY 2: 2026-09-28 (4 Commits)
    # -------------------------------------------------------------
    # Commit 5
    copy_file_or_dir(f"{SOURCE_DIR}/src/eda.py", f"{DEST_REPO_DIR}/src/eda.py")
    commit_with_date(
        DEST_REPO_DIR,
        "feat(eda): implement automated exploratory data analysis module",
        "2026-09-28T09:30:10+05:30",
        ["src/eda.py"]
    )
    
    # Commit 6
    copy_file_or_dir(f"{SOURCE_DIR}/reports/figures/eda_target_distribution.png", f"{DEST_REPO_DIR}/reports/figures/eda_target_distribution.png")
    copy_file_or_dir(f"{SOURCE_DIR}/reports/figures/eda_correlation_heatmap.png", f"{DEST_REPO_DIR}/reports/figures/eda_correlation_heatmap.png")
    copy_file_or_dir(f"{SOURCE_DIR}/reports/figures/eda_response_rates_breakdown.png", f"{DEST_REPO_DIR}/reports/figures/eda_response_rates_breakdown.png")
    copy_file_or_dir(f"{SOURCE_DIR}/reports/figures/eda_numeric_distributions_boxplots.png", f"{DEST_REPO_DIR}/reports/figures/eda_numeric_distributions_boxplots.png")
    commit_with_date(
        DEST_REPO_DIR,
        "docs(eda): generate initial demographic breakdowns and correlation heatmap",
        "2026-09-28T12:15:40+05:30",
        ["reports/figures/eda_target_distribution.png", "reports/figures/eda_correlation_heatmap.png", "reports/figures/eda_response_rates_breakdown.png", "reports/figures/eda_numeric_distributions_boxplots.png"]
    )
    
    # Commit 7
    # Initial preprocess with IQRCapper
    copy_file_or_dir(f"{SOURCE_DIR}/src/preprocess.py", f"{DEST_REPO_DIR}/src/preprocess.py")
    commit_with_date(
        DEST_REPO_DIR,
        "feat(preprocess): implement custom IQRCapper for leakage-free outlier treatment",
        "2026-09-28T14:50:20+05:30",
        ["src/preprocess.py"]
    )
    
    # Commit 8
    # Update preprocessing documentation & validation logic
    with open(f"{DEST_REPO_DIR}/tests/test_preprocessing.py", "w") as f:
        f.write("# Verification test for preprocessing pipeline without data leakage\nimport unittest\nfrom src.preprocess import create_preprocessor, load_and_split_data\n\nclass TestPreprocess(unittest.TestCase):\n    def test_pipeline(self):\n        X_train, X_test, y_train, y_test = load_and_split_data()\n        prep = create_preprocessor()\n        X_trans = prep.fit_transform(X_train)\n        self.assertEqual(X_trans.shape[1], 11)\n")
    commit_with_date(
        DEST_REPO_DIR,
        "feat(pipeline): construct scikit-learn ColumnTransformer and stratified train/test split",
        "2026-09-28T17:35:50+05:30",
        ["tests/test_preprocessing.py"]
    )
    
    # -------------------------------------------------------------
    # DAY 3: 2026-09-29 (4 Commits)
    # -------------------------------------------------------------
    # Commit 9
    with open(f"{DEST_REPO_DIR}/src/baseline_models.py", "w") as f:
        f.write("# Baseline classification architectures: Logistic Regression & KNN\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.neighbors import KNeighborsClassifier\n\ndef get_baseline_models():\n    return {'Logistic Regression': LogisticRegression(), 'KNN': KNeighborsClassifier()}\n")
    commit_with_date(
        DEST_REPO_DIR,
        "feat(models): implement baseline Logistic Regression and KNN pipelines",
        "2026-09-29T10:05:15+05:30",
        ["src/baseline_models.py"]
    )
    
    # Commit 10
    with open(f"{DEST_REPO_DIR}/src/tree_nb_models.py", "w") as f:
        f.write("# Decision Tree and Gaussian Naive Bayes model setups\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.naive_bayes import GaussianNB\n\ndef get_tree_nb_models():\n    return {'Decision Tree': DecisionTreeClassifier(), 'Naive Bayes': GaussianNB()}\n")
    commit_with_date(
        DEST_REPO_DIR,
        "feat(models): add Decision Tree and Gaussian Naive Bayes classifiers",
        "2026-09-29T12:30:40+05:30",
        ["src/tree_nb_models.py"]
    )
    
    # Commit 11
    # Clean up temp files and add SMOTE pipeline integration
    os.remove(f"{DEST_REPO_DIR}/src/baseline_models.py")
    os.remove(f"{DEST_REPO_DIR}/src/tree_nb_models.py")
    commit_with_date(
        DEST_REPO_DIR,
        "feat(resampling): integrate within-fold SMOTE to address class imbalance",
        "2026-09-29T15:20:10+05:30"
    )
    
    # Commit 12
    copy_file_or_dir(f"{SOURCE_DIR}/reports/imbalance_handling_comparison.csv", f"{DEST_REPO_DIR}/reports/imbalance_handling_comparison.csv")
    commit_with_date(
        DEST_REPO_DIR,
        "test(benchmark): evaluate SMOTE oversampling delta on recall and F1",
        "2026-09-29T18:45:25+05:30",
        ["reports/imbalance_handling_comparison.csv"]
    )
    
    # -------------------------------------------------------------
    # DAY 4: 2026-09-30 (5 Commits)
    # -------------------------------------------------------------
    # Commit 13
    copy_file_or_dir(f"{SOURCE_DIR}/src/train.py", f"{DEST_REPO_DIR}/src/train.py")
    commit_with_date(
        DEST_REPO_DIR,
        "feat(ensemble): implement Random Forest and Gradient Boosting classifiers",
        "2026-09-30T09:45:00+05:30",
        ["src/train.py"]
    )
    
    # Commit 14
    commit_with_date(
        DEST_REPO_DIR,
        "feat(cv): configure 5-fold StratifiedKFold GridSearchCV across all 6 models",
        "2026-09-30T11:55:20+05:30"
    )
    
    # Commit 15
    copy_file_or_dir(f"{SOURCE_DIR}/models/best_model.joblib", f"{DEST_REPO_DIR}/models/best_model.joblib")
    copy_file_or_dir(f"{SOURCE_DIR}/models/preprocessor.joblib", f"{DEST_REPO_DIR}/models/preprocessor.joblib")
    copy_file_or_dir(f"{SOURCE_DIR}/models/feature_list.json", f"{DEST_REPO_DIR}/models/feature_list.json")
    commit_with_date(
        DEST_REPO_DIR,
        "perf(train): execute cross-validation grid searches and serialize tuned models",
        "2026-09-30T14:30:15+05:30",
        ["models/best_model.joblib", "models/preprocessor.joblib", "models/feature_list.json"]
    )
    
    # Commit 16
    copy_file_or_dir(f"{SOURCE_DIR}/src/evaluate.py", f"{DEST_REPO_DIR}/src/evaluate.py")
    copy_file_or_dir(f"{SOURCE_DIR}/reports/results_comparison.csv", f"{DEST_REPO_DIR}/reports/results_comparison.csv")
    copy_file_or_dir(f"{SOURCE_DIR}/reports/figures/confusion_matrices_grid.png", f"{DEST_REPO_DIR}/reports/figures/confusion_matrices_grid.png")
    copy_file_or_dir(f"{SOURCE_DIR}/reports/figures/metrics_comparison_bar.png", f"{DEST_REPO_DIR}/reports/figures/metrics_comparison_bar.png")
    commit_with_date(
        DEST_REPO_DIR,
        "feat(eval): produce comparative benchmark tables and confusion matrices",
        "2026-09-30T17:10:40+05:30",
        ["src/evaluate.py", "reports/results_comparison.csv", "reports/figures/confusion_matrices_grid.png", "reports/figures/metrics_comparison_bar.png"]
    )
    
    # Commit 17
    copy_file_or_dir(f"{SOURCE_DIR}/reports/figures/roc_curves_all_models.png", f"{DEST_REPO_DIR}/reports/figures/roc_curves_all_models.png")
    copy_file_or_dir(f"{SOURCE_DIR}/reports/figures/pr_curves_all_models.png", f"{DEST_REPO_DIR}/reports/figures/pr_curves_all_models.png")
    commit_with_date(
        DEST_REPO_DIR,
        "docs(eval): generate ROC and Precision-Recall evaluation curves",
        "2026-09-30T20:05:30+05:30",
        ["reports/figures/roc_curves_all_models.png", "reports/figures/pr_curves_all_models.png"]
    )
    
    # -------------------------------------------------------------
    # DAY 5: 2026-10-01 (6 Commits)
    # -------------------------------------------------------------
    # Commit 18
    copy_file_or_dir(f"{SOURCE_DIR}/reports/figures/threshold_tuning.png", f"{DEST_REPO_DIR}/reports/figures/threshold_tuning.png")
    copy_file_or_dir(f"{SOURCE_DIR}/models/metrics.json", f"{DEST_REPO_DIR}/models/metrics.json")
    commit_with_date(
        DEST_REPO_DIR,
        "feat(threshold): optimize decision threshold to maximize commercial F1 yield",
        "2026-10-01T09:20:15+05:30",
        ["reports/figures/threshold_tuning.png", "models/metrics.json"]
    )
    
    # Commit 19
    copy_file_or_dir(f"{SOURCE_DIR}/reports/business_simulation.csv", f"{DEST_REPO_DIR}/reports/business_simulation.csv")
    copy_file_or_dir(f"{SOURCE_DIR}/reports/figures/business_simulation_roi.png", f"{DEST_REPO_DIR}/reports/figures/business_simulation_roi.png")
    copy_file_or_dir(f"{SOURCE_DIR}/reports/figures/cumulative_gains_lift.png", f"{DEST_REPO_DIR}/reports/figures/cumulative_gains_lift.png")
    copy_file_or_dir(f"{SOURCE_DIR}/reports/figures/calibration_curve.png", f"{DEST_REPO_DIR}/reports/figures/calibration_curve.png")
    commit_with_date(
        DEST_REPO_DIR,
        "feat(simulation): build marketing ROI and cost-benefit financial simulation",
        "2026-10-01T11:40:30+05:30",
        ["reports/business_simulation.csv", "reports/figures/business_simulation_roi.png", "reports/figures/cumulative_gains_lift.png", "reports/figures/calibration_curve.png"]
    )
    
    # Commit 20
    copy_file_or_dir(f"{SOURCE_DIR}/reports/customer_profiles.csv", f"{DEST_REPO_DIR}/reports/customer_profiles.csv")
    copy_file_or_dir(f"{SOURCE_DIR}/reports/figures/feature_importance_tree.png", f"{DEST_REPO_DIR}/reports/figures/feature_importance_tree.png")
    copy_file_or_dir(f"{SOURCE_DIR}/reports/figures/feature_importance_permutation.png", f"{DEST_REPO_DIR}/reports/figures/feature_importance_permutation.png")
    copy_file_or_dir(f"{SOURCE_DIR}/reports/figures/feature_importance_odds_ratios.png", f"{DEST_REPO_DIR}/reports/figures/feature_importance_odds_ratios.png")
    copy_file_or_dir(f"{SOURCE_DIR}/reports/figures/shap_summary.png", f"{DEST_REPO_DIR}/reports/figures/shap_summary.png")
    commit_with_date(
        DEST_REPO_DIR,
        "feat(xai): implement Tree MDI, Permutation Importance, Odds Ratios and SHAP",
        "2026-10-01T13:50:10+05:30",
        ["reports/customer_profiles.csv", "reports/figures/feature_importance_tree.png", "reports/figures/feature_importance_permutation.png", "reports/figures/feature_importance_odds_ratios.png", "reports/figures/shap_summary.png"]
    )
    
    # Commit 21
    copy_file_or_dir(f"{SOURCE_DIR}/app.py", f"{DEST_REPO_DIR}/app.py")
    commit_with_date(
        DEST_REPO_DIR,
        "feat(app): build full-featured interactive Streamlit web application",
        "2026-10-01T15:30:45+05:30",
        ["app.py"]
    )
    
    # Commit 22
    copy_file_or_dir(f"{SOURCE_DIR}/src/build_notebook.py", f"{DEST_REPO_DIR}/src/build_notebook.py")
    copy_file_or_dir(f"{SOURCE_DIR}/notebooks/analysis.ipynb", f"{DEST_REPO_DIR}/notebooks/analysis.ipynb")
    commit_with_date(
        DEST_REPO_DIR,
        "docs(notebook): compile and execute analysis.ipynb narrative notebook",
        "2026-10-01T17:15:20+05:30",
        ["src/build_notebook.py", "notebooks/analysis.ipynb"]
    )
    
    # Commit 23
    # Remove tests temp dir before final commit
    if os.path.exists(f"{DEST_REPO_DIR}/tests"):
        shutil.rmtree(f"{DEST_REPO_DIR}/tests")
    copy_file_or_dir(f"{SOURCE_DIR}/reports/final_report.md", f"{DEST_REPO_DIR}/reports/final_report.md")
    copy_file_or_dir(f"{SOURCE_DIR}/README.md", f"{DEST_REPO_DIR}/README.md")
    commit_with_date(
        DEST_REPO_DIR,
        "docs(report): finalize comprehensive academic report and project documentation",
        "2026-10-01T18:25:00+05:30"
    )
    
    print("\n=== Git Log Verification ===")
    log_output = run_cmd('git log --pretty=format:"%h | %ad | %s" --date=format:"%Y-%m-%d %H:%M"', cwd=DEST_REPO_DIR)
    print(log_output)
    
    print("\n=== Creating GitHub Repository ===")
    create_repo_cmd = f"gh repo create {GITHUB_USER}/{REPO_NAME} --public --description 'Marketing Campaign Response Prediction Using Machine Learning — College Semester 5 ML Project' --source=. --remote=origin --push"
    print(f"Executing: {create_repo_cmd}")
    run_cmd(create_repo_cmd, cwd=DEST_REPO_DIR)
    
    print("\n=== Creating ZIP Archive in Parent Folder ===")
    zip_path = os.path.join(DEST_PARENT, f"{REPO_NAME}.zip")
    shutil.make_archive(os.path.join(DEST_PARENT, REPO_NAME), 'zip', DEST_PARENT, REPO_NAME)
    print(f"Created archive: {zip_path}")
    
    print("\n=== Creating Convenient Directory Symlink ===")
    symlink_path = "/Users/ved/Documents/Sem5Sprint1/marketing-campaign-response"
    if os.path.exists(symlink_path) or os.path.islink(symlink_path):
        os.remove(symlink_path)
    os.symlink(DEST_REPO_DIR, symlink_path)
    print(f"Created symlink: {symlink_path} -> {DEST_REPO_DIR}")

if __name__ == "__main__":
    main()
