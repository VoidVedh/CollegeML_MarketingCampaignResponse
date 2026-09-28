"""
preprocess.py
Modular, leakage-free preprocessing pipeline using scikit-learn ColumnTransformer,
custom IQRCapper for outlier treatment, median/mode imputation, one-hot encoding,
and standard scaling. Reusable in training, evaluation, and Streamlit deployment.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
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

CATEGORICAL_FEATURES = ['age_group']
NUMERIC_FEATURES = [
    'income',
    'previous_purchases',
    'purchase_frequency',
    'previous_campaign_response',
    'website_visits',
    'email_engagement',
    'discount_usage'
]
TARGET_COLUMN = 'responded'


class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Caps numeric outliers based on the 1.5 * Interquartile Range (IQR) rule.
    Learns bounds during `fit` on training data only to prevent data leakage.
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


def create_preprocessor() -> ColumnTransformer:
    """
    Creates and returns an unfitted scikit-learn ColumnTransformer.
    - Numeric: Median Imputer -> IQRCapper -> StandardScaler
    - Categorical: Mode Imputer -> OneHotEncoder(drop='first' or handle_unknown='ignore')
    """
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_capper', IQRCapper(factor=1.5)),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, NUMERIC_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    return preprocessor


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """
    Validates that a DataFrame conforms to the expected feature schema.
    Converts and sanitizes data types where possible.
    """
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Uploaded data missing required feature columns: {missing_cols}")
    
    if require_target and TARGET_COLUMN not in df.columns:
        raise ValueError(f"Uploaded data missing target column '{TARGET_COLUMN}'")
        
    df_clean = df.copy()
    
    # Standardize string inputs
    df_clean['age_group'] = df_clean['age_group'].astype(str).str.strip()
    
    # Ensure numeric types
    for col in NUMERIC_FEATURES:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
    if require_target and TARGET_COLUMN in df_clean.columns:
        df_clean[TARGET_COLUMN] = pd.to_numeric(df_clean[TARGET_COLUMN], errors='coerce').astype(int)
        
    return df_clean


def load_and_split_data(
    filepath: str = "data/campaign_data.csv",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads dataset, validates schema, and performs stratified 80/20 train/test split.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    df = pd.read_csv(filepath)
    df = validate_schema(df, require_target=True)
    
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts output feature names from a fitted ColumnTransformer.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual reconstruction
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        return NUMERIC_FEATURES + cat_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()
    prep = create_preprocessor()
    X_train_trans = prep.fit_transform(X_train)
    feature_names = get_feature_names(prep)
    print(f"X_train shape: {X_train.shape} -> Transformed shape: {X_train_trans.shape}")
    print(f"Features: {feature_names}")
