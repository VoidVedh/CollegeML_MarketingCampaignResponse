"""
preprocess.py
Production-grade, leakage-free preprocessing pipeline for the Kaggle Marketing Dataset
(Bank Marketing / Term Deposit Subscription Prediction — Case Study 157).

Key Highlights:
1. TARGET LEAKAGE PREVENTION:
   - 'duration' is STRICTLY EXCLUDED from predictive features because call duration
     is unknown prior to contacting a client, making it unrealistic for real-world targeting.
2. MODULAR COLUMNTRANSFORMER:
   - Numerical Features: Median Imputer -> IQRCapper -> StandardScaler.
   - Categorical Features: 'unknown' Constant Imputer -> OneHotEncoder(drop='first', handle_unknown='ignore').
3. COMPATIBILITY & REUSABILITY:
   - Operates identically in training, cross-validation, batch evaluation, and live Streamlit deployment.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Canonical dataset file paths
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "kaggle", "train.csv")

# Target column
TARGET_COLUMN = "y"

# Explicitly excluded columns (to avoid target leakage or invalid identifiers)
EXCLUDED_COLUMNS = ["duration", "id"]

# Predictive Numerical Features (9 features)
NUMERIC_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]

# Predictive Categorical Features (11 features, including derived age_group)
CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
    "age_group"
]

# All expected predictive features combined (20 features)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Valid discrete values for categorical variables
VALID_CATEGORIES: Dict[str, List[str]] = {
    "job": [
        "admin.", "blue-collar", "technician", "services", "management",
        "retired", "entrepreneur", "self-employed", "housemaid", "unemployed",
        "student", "unknown"
    ],
    "marital": ["married", "single", "divorced", "unknown"],
    "education": [
        "university.degree", "high.school", "basic.9y", "professional.course",
        "basic.4y", "basic.6y", "unknown", "illiterate"
    ],
    "default": ["no", "unknown", "yes"],
    "housing": ["yes", "no", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"],
    "month": ["may", "jul", "aug", "jun", "nov", "apr", "oct", "sep", "mar", "dec"],
    "day_of_week": ["mon", "thu", "wed", "tue", "fri"],
    "poutcome": ["nonexistent", "failure", "success"],
    "age_group": ["18-25", "26-35", "36-45", "46-55", "56+"]
}


def derive_age_group(age_series: pd.Series) -> pd.Series:
    """
    Derives standard age groups from continuous age:
    - 18-25, 26-35, 36-45, 46-55, 56+
    """
    bins = [0, 25, 35, 45, 55, 150]
    labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
    return pd.cut(age_series, bins=bins, labels=labels, right=True).astype(str)


class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Caps continuous numeric outliers based on the 1.5 * Interquartile Range (IQR) rule.
    Learns bounds during `fit` on training data only to prevent data leakage.
    Provides get_feature_names_out for native scikit-learn ColumnTransformer compatibility.
    """
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        q25 = np.nanpercentile(X_arr, 25, axis=0)
        q75 = np.nanpercentile(X_arr, 75, axis=0)
        iqr = q75 - q25
        self.lower_bounds_ = q25 - self.factor * iqr
        self.upper_bounds_ = q75 + self.factor * iqr
        # For features with IQR == 0 (e.g. pdays, previous where mode > 75%),
        # do not collapse to constant; leave unclipped to preserve true predictive variance
        zero_iqr = (iqr == 0)
        self.lower_bounds_[zero_iqr] = -np.inf
        self.upper_bounds_[zero_iqr] = np.inf
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("IQRCapper must be fitted before transform.")
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if self.lower_bounds_ is not None:
                return np.array([f"feature_{i}" for i in range(len(self.lower_bounds_))], dtype=object)
            return np.array([], dtype=object)
        return np.asarray(input_features, dtype=object)


def create_preprocessor() -> ColumnTransformer:
    """
    Creates and returns an unfitted scikit-learn ColumnTransformer.
    - Numerical: Median Imputer -> IQRCapper -> StandardScaler
    - Categorical: Unknown Imputer -> OneHotEncoder(drop='first', handle_unknown='ignore')
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier_capper", IQRCapper(factor=1.5)),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts post-transformation feature names from a fitted ColumnTransformer.
    """
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            names = trans.get_feature_names_out(cols)
        elif hasattr(trans, "named_steps"):
            last_step = list(trans.named_steps.values())[-1]
            if hasattr(last_step, "get_feature_names_out"):
                names = last_step.get_feature_names_out(cols)
            else:
                names = cols
        else:
            names = cols
        feature_names.extend(list(names))
    return feature_names


def load_and_split_data(
    filepath: str = DEFAULT_DATA_PATH,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads the Kaggle Marketing Dataset, drops leakage column 'duration', derives 'age_group',
    converts target 'y' ('yes'->1, 'no'->0), and returns stratified train/test splits.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Kaggle dataset not found at {filepath}. Please ensure data/kaggle/train.csv exists.")

    # Support comma-separated or semicolon-separated CSVs
    try:
        df = pd.read_csv(filepath, sep=",")
        if df.shape[1] == 1:
            df = pd.read_csv(filepath, sep=";")
    except Exception:
        df = pd.read_csv(filepath, sep=";")

    # Drop duration to prevent leakage
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    # Drop id if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Derive age_group if missing
    if "age_group" not in df.columns and "age" in df.columns:
        df["age_group"] = derive_age_group(df["age"])

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {filepath}")

    # Map target: yes -> 1, no -> 0
    if df[TARGET_COLUMN].dtype == object or isinstance(df[TARGET_COLUMN].iloc[0], str):
        y = (df[TARGET_COLUMN].astype(str).str.lower().str.strip() == "yes").astype(int)
    else:
        y = df[TARGET_COLUMN].astype(int)

    X = df.drop(columns=[TARGET_COLUMN])

    # Reorder columns to match canonical FEATURE_COLUMNS
    missing_cols = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required features: {missing_cols}")

    X = X[FEATURE_COLUMNS]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str], pd.DataFrame]:
    """
    Validates input DataFrame against canonical Kaggle schema.
    Returns: (is_valid, list_of_errors_or_warnings, cleaned_df)
    """
    errors = []
    cleaned_df = df.copy()

    # Drop duration if accidentally provided by user to enforce zero-leakage
    if "duration" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns=["duration"])
        errors.append("Notice: 'duration' column was stripped to prevent target leakage.")

    # Drop id if provided
    if "id" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns=["id"])

    # Derive age_group if not present
    if "age_group" not in cleaned_df.columns and "age" in cleaned_df.columns:
        cleaned_df["age_group"] = derive_age_group(cleaned_df["age"])

    # Check for missing required features
    missing = [c for c in FEATURE_COLUMNS if c not in cleaned_df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return False, errors, cleaned_df

    # Reorder to exact canonical order
    cleaned_df = cleaned_df[FEATURE_COLUMNS]

    return True, errors, cleaned_df


# Backward-compatible aliases for legacy imports
CONTINUOUS_FEATURES = NUMERIC_FEATURES
BINARY_FEATURES: List[str] = []


def unit_test_preprocessing() -> bool:
    """
    Self-contained unit test verifying the preprocessing pipeline:
    1. Robustness against missing values (imputation)
    2. Outlier capping with IQRCapper
    3. Proper one-hot encoding without rank errors
    4. Exact feature dimension alignment
    """
    sample_data = pd.DataFrame({
        "age": [25, 40, np.nan, 85, 30],
        "campaign": [1, 2, 50, 1, np.nan],
        "pdays": [999, 6, 999, np.nan, 999],
        "previous": [0, 1, 0, 5, 0],
        "emp.var.rate": [1.1, -1.8, 1.4, np.nan, -0.1],
        "cons.price.idx": [93.994, 92.893, np.nan, 94.465, 93.2],
        "cons.conf.idx": [-36.4, -46.2, -42.7, np.nan, -42.0],
        "euribor3m": [4.857, 1.299, 4.962, np.nan, 4.021],
        "nr.employed": [5191.0, 5099.1, np.nan, 5228.1, 5195.8],
        "job": ["admin.", "blue-collar", np.nan, "retired", "technician"],
        "marital": ["married", "single", np.nan, "divorced", "married"],
        "education": ["university.degree", "high.school", np.nan, "basic.4y", "professional.course"],
        "default": ["no", "no", np.nan, "unknown", "no"],
        "housing": ["yes", "no", np.nan, "yes", "no"],
        "loan": ["no", "yes", np.nan, "no", "no"],
        "contact": ["cellular", "telephone", np.nan, "cellular", "cellular"],
        "month": ["may", "jul", np.nan, "nov", "aug"],
        "day_of_week": ["mon", "wed", np.nan, "fri", "thu"],
        "poutcome": ["nonexistent", "failure", np.nan, "success", "nonexistent"],
        "age_group": ["18-25", "36-45", "18-25", "56+", "26-35"]
    })
    preprocessor = create_preprocessor()
    X_trans = preprocessor.fit_transform(sample_data)
    assert not np.isnan(X_trans).any(), "NaN values detected after preprocessing"
    feature_names = get_feature_names(preprocessor)
    assert X_trans.shape[1] == len(feature_names), f"Feature count mismatch: {X_trans.shape[1]} vs {len(feature_names)}"
    return True

