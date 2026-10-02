"""
preprocess.py
Modular, leakage-free preprocessing pipeline using scikit-learn ColumnTransformer,
custom IQRCapper for continuous outlier treatment, imputation, one-hot encoding,
and standard scaling. Reusable in training, evaluation, and Streamlit deployment.
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

# Canonical schema definition
FEATURE_COLUMNS = [
    'age_group',
    'income',
    'previous_purchases',
    'purchase_frequency',
    'previous_campaign_response',
    'website_visits',
    'email_engagement',
    'discount_usage'
]

# Partitioning features strictly by statistical treatment
CATEGORICAL_FEATURES = ['age_group']

# Continuous features requiring outlier capping and scaling
CONTINUOUS_FEATURES = [
    'income',
    'previous_purchases',
    'purchase_frequency',
    'website_visits',
    'email_engagement',
    'discount_usage'
]

# Binary features (must NOT be clipped by IQRCapper or z-score scaled)
BINARY_FEATURES = ['previous_campaign_response']

# All numeric columns combined (continuous + binary)
NUMERIC_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES
TARGET_COLUMN = 'responded'

# Valid discrete categories
VALID_AGE_GROUPS = ['18-25', '26-35', '36-45', '46-55', '56+']


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
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("IQRCapper must be fitted before transform.")
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        """Scikit-learn compliant feature names output."""
        if input_features is None:
            if self.lower_bounds_ is not None:
                return np.array([f"feature_{i}" for i in range(len(self.lower_bounds_))], dtype=object)
            return np.array([], dtype=object)
        return np.asarray(input_features, dtype=object)


def create_preprocessor() -> ColumnTransformer:
    """
    Creates and returns an unfitted scikit-learn ColumnTransformer.
    - Continuous: Median Imputer -> IQRCapper -> StandardScaler
    - Binary: Mode Imputer (no capping, no scaling - preserved as 0/1)
    - Categorical: Mode Imputer -> OneHotEncoder(drop='first', sparse_output=False)
    """
    continuous_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_capper', IQRCapper(factor=1.5)),
        ('scaler', StandardScaler())
    ])

    binary_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cont', continuous_pipeline, CONTINUOUS_FEATURES),
            ('bin', binary_pipeline, BINARY_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    return preprocessor


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates that a DataFrame conforms to the expected feature schema.
    Strictly checks target values when required (no silent conversion of missing/invalid targets).
    Preserves NaN in categorical columns rather than converting to string 'nan'.
    Audits missing values, out-of-range anomalies, and unknown categories.
    
    Returns:
        (sanitized_df, audit_report_dict)
    """
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Data missing required feature columns: {missing_cols}")
    
    if require_target:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Data missing required target column '{TARGET_COLUMN}'")
        
        # Target validation: Must not contain missing values or non-binary labels
        target_series = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
        if target_series.isna().any():
            missing_target_count = int(target_series.isna().sum())
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' contains {missing_target_count} missing or malformed non-numeric values. "
                "Training target labels cannot be silently imputed or converted to zero."
            )
        
        unique_targets = set(target_series.unique())
        if not unique_targets.issubset({0, 1}):
            invalid_vals = unique_targets - {0, 1}
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' contains invalid values {invalid_vals}. "
                "Target values must strictly be binary: 0 (non-responder) or 1 (responder)."
            )
        
    df_clean = df.copy()
    audit_report = {
        'total_rows': len(df_clean),
        'missing_rows': {},
        'out_of_range_rows': {},
        'unknown_categories': {},
        'negative_values_clamped': {}
    }
    
    # 1. Categorical handling: Preserve NaN properly
    raw_age = df_clean['age_group']
    missing_age_mask = raw_age.isna() | raw_age.astype(str).str.strip().str.lower().isin(['nan', 'none', '', 'null', '<na>'])
    
    # Standardize non-missing age values
    def clean_age(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if s.lower() in ['nan', 'none', '', 'null', '<na>']:
            return np.nan
        return s

    df_clean['age_group'] = df_clean['age_group'].apply(clean_age)
    
    if missing_age_mask.any():
        audit_report['missing_rows']['age_group'] = int(missing_age_mask.sum())
        
    unknown_age_mask = df_clean['age_group'].notna() & (~df_clean['age_group'].isin(VALID_AGE_GROUPS))
    if unknown_age_mask.any():
        audit_report['unknown_categories']['age_group'] = int(unknown_age_mask.sum())
        # Set unknown categories to NaN so imputer handles them with mode
        df_clean.loc[unknown_age_mask, 'age_group'] = np.nan
    
    # 2. Continuous and Binary Numeric handling
    for col in CONTINUOUS_FEATURES:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        miss_count = int(df_clean[col].isna().sum())
        if miss_count > 0:
            audit_report['missing_rows'][col] = miss_count
            
        # Range sanity checks
        if col in ['email_engagement', 'discount_usage']:
            # Engagement rates are strictly bounded between 0.0 and 1.0 by mathematical definition
            out_mask = (df_clean[col] < 0.0) | (df_clean[col] > 1.0)
            if out_mask.any():
                audit_report['out_of_range_rows'][col] = int(out_mask.sum())
                df_clean[col] = np.clip(df_clean[col], 0.0, 1.0)
        elif col in ['income', 'previous_purchases', 'purchase_frequency', 'website_visits']:
            # Non-negative counts/amounts: clamp negative values to 0.0 with explicit audit logging
            out_mask = df_clean[col] < 0.0
            if out_mask.any():
                audit_report['negative_values_clamped'][col] = int(out_mask.sum())
                df_clean.loc[out_mask, col] = 0.0
                
    # 3. Binary feature: previous_campaign_response
    df_clean['previous_campaign_response'] = pd.to_numeric(df_clean['previous_campaign_response'], errors='coerce')
    miss_bin = int(df_clean['previous_campaign_response'].isna().sum())
    if miss_bin > 0:
        audit_report['missing_rows']['previous_campaign_response'] = miss_bin
        
    out_bin = ~df_clean['previous_campaign_response'].isna() & ~df_clean['previous_campaign_response'].isin([0, 1])
    if out_bin.any():
        audit_report['out_of_range_rows']['previous_campaign_response'] = int(out_bin.sum())
        df_clean.loc[out_bin, 'previous_campaign_response'] = (df_clean.loc[out_bin, 'previous_campaign_response'] > 0.5).astype(float)
        
    if require_target and TARGET_COLUMN in df_clean.columns:
        df_clean[TARGET_COLUMN] = target_series.astype(int)
        
    return df_clean, audit_report


def load_and_split_data(
    filepath: str = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads dataset, validates schema, and performs stratified 80/20 train/test split.
    Uses robust project-relative paths.
    """
    if filepath is None:
        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "campaign_data.csv"),
            os.path.join(os.getcwd(), "data", "campaign_data.csv"),
            "data/campaign_data.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                filepath = os.path.abspath(c)
                break
        if filepath is None:
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "campaign_data.csv")
            
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    df = pd.read_csv(filepath)
    df_clean, _ = validate_schema(df, require_target=True)
    
    X = df_clean[FEATURE_COLUMNS]
    y = df_clean[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts output feature names from a fitted ColumnTransformer natively.
    """
    return list(preprocessor.get_feature_names_out())


def unit_test_preprocessing(filepath: str = None):
    """
    Automated verification unit test:
    Ensures no transformed numeric column has zero variance and that
    previous_campaign_response flows through with binary variance intact.
    """
    X_train, X_test, y_train, y_test = load_and_split_data(filepath=filepath)
    prep = create_preprocessor()
    X_trans = prep.fit_transform(X_train)
    feature_names = get_feature_names(prep)
    
    variances = np.var(X_trans, axis=0)
    for name, var in zip(feature_names, variances):
        assert var > 0.0, f"Defect detected: Column '{name}' has zero variance ({var})!"
        
    # Verify previous_campaign_response values
    bin_idx = feature_names.index('previous_campaign_response')
    bin_vals = np.unique(X_trans[:, bin_idx])
    assert set(bin_vals).issubset({0.0, 1.0}), f"previous_campaign_response contains invalid values: {bin_vals}"
    assert variances[bin_idx] > 0.05, f"previous_campaign_response has suspiciously low variance: {variances[bin_idx]}"
    print("PASS: Preprocessing unit tests passed (no zero-variance columns, binary feature intact).")
    return True


if __name__ == "__main__":
    unit_test_preprocessing()
