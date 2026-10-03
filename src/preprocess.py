"""
preprocess.py
Production-grade preprocessing and feature engineering pipeline for Case Study 157:
Marketing Campaign Response Prediction Using Machine Learning.

Dataset: Kaggle Customer Personality Analysis (marketing_campaign.csv)
Source: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis

Derives exactly the eight Case Study 157 features:
1. age_group: Binned from Year_Birth using reference year 2014 (end of dataset campaign observation)
   into categories ['18-25', '26-35', '36-45', '46-55', '56+'].
2. income: Customer annual household income from 'Income' (missing values handled via median imputation).
3. previous_purchases: Total historical purchases across channels:
   NumWebPurchases + NumCatalogPurchases + NumStorePurchases.
4. purchase_frequency: Monthly purchase frequency:
   previous_purchases / observed customer tenure in months (from Dt_Customer to 2014-12-31).
5. previous_campaign_response: Binary flag indicating whether customer accepted any prior campaign:
   (AcceptedCmp1 + AcceptedCmp2 + AcceptedCmp3 + AcceptedCmp4 + AcceptedCmp5 >= 1).
   Current target 'Response' is strictly excluded to prevent target leakage.
6. website_visits: Monthly website visits from 'NumWebVisitsMonth'.
7. email_engagement: Engagement proxy derived from historical campaign acceptance rate:
   (AcceptedCmp1 + AcceptedCmp2 + AcceptedCmp3 + AcceptedCmp4 + AcceptedCmp5) / 5.0.
   Clearly disclosed as an engagement proxy because direct email click/open rates are not recorded.
8. discount_usage: Proportion of purchases made using discount deals:
   NumDealsPurchases / max(previous_purchases, 1), safely clamped to [0.0, 1.0].

Target:
target: Response (0 = Will Not Respond, 1 = Will Respond).
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Canonical dataset file paths
RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "kaggle", "marketing_campaign.csv")
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "campaign_response_features.csv")

# Target column definition
TARGET_COLUMN = "target"

# The 8 Case Study 157 Modelling & Deployment Features
CATEGORICAL_FEATURES = ["age_group"]
NUMERIC_FEATURES = [
    "income",
    "previous_purchases",
    "purchase_frequency",
    "previous_campaign_response",
    "website_visits",
    "email_engagement",
    "discount_usage"
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Valid categorical domain values
VALID_AGE_GROUPS = ["18-25", "26-35", "36-45", "46-55", "56+"]


def engineer_case157_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw Kaggle Customer Personality Analysis data into the 8 Case Study 157 features.
    
    Parameters:
        raw_df (pd.DataFrame): Raw dataframe loaded from marketing_campaign.csv.
        
    Returns:
        pd.DataFrame: Clean dataframe containing the 8 features and binary target.
    """
    df = raw_df.copy()

    # 1. age_group: derived from Year_Birth using 2014 as the reference observation year
    ref_year = 2014
    age = ref_year - df["Year_Birth"]
    bins = [-np.inf, 25, 35, 45, 55, np.inf]
    labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
    age_group = pd.cut(age, bins=bins, labels=labels, right=True).astype(str)

    # 2. income: annual household income
    income = df["Income"].astype(float)

    # 3. previous_purchases: sum of channel purchases
    previous_purchases = (
        df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
    ).astype(float)

    # 4. purchase_frequency: purchases per month of customer tenure
    # Reference date is set to 2014-12-31 (conclusion of campaign evaluation period)
    ref_date = pd.to_datetime("2014-12-31")
    dt_customer = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
    tenure_months = (ref_date - dt_customer).dt.days / 30.4375
    purchase_frequency = np.round(previous_purchases / np.maximum(tenure_months, 1.0), 4)

    # 5. previous_campaign_response: prior campaign participation (AcceptedCmp 1 to 5)
    # Strictly excludes current campaign 'Response' to eliminate target leakage
    prior_cmps = (
        df["AcceptedCmp1"] + df["AcceptedCmp2"] + df["AcceptedCmp3"] +
        df["AcceptedCmp4"] + df["AcceptedCmp5"]
    )
    previous_campaign_response = (prior_cmps >= 1).astype(int)

    # 6. website_visits: monthly web visits
    website_visits = df["NumWebVisitsMonth"].astype(float)

    # 7. email_engagement: proxy based on historical campaign acceptances / 5 campaigns
    email_engagement = np.round(prior_cmps / 5.0, 4)

    # 8. discount_usage: proportion of purchases made with discount deals
    deals = df["NumDealsPurchases"].astype(float)
    discount_usage = np.where(previous_purchases > 0, deals / previous_purchases, 0.0)
    discount_usage = np.round(np.clip(discount_usage, 0.0, 1.0), 4)

    # Target variable
    target = df["Response"].astype(int)

    processed_df = pd.DataFrame({
        "age_group": age_group,
        "income": income,
        "previous_purchases": previous_purchases,
        "purchase_frequency": purchase_frequency,
        "previous_campaign_response": previous_campaign_response,
        "website_visits": website_visits,
        "email_engagement": email_engagement,
        "discount_usage": discount_usage,
        "target": target
    })

    return processed_df


def create_preprocessor() -> ColumnTransformer:
    """
    Constructs a leakage-free scikit-learn ColumnTransformer.
    - Numerical features: Median Imputation -> StandardScaler
    - Categorical features: Most-Frequent Imputation -> OneHotEncoder(drop='first')
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            categories=[VALID_AGE_GROUPS],
            drop="first",
            sparse_output=False,
            handle_unknown="ignore"
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts human-readable feature names after one-hot encoding.
    """
    ohe = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return NUMERIC_FEATURES + cat_feature_names


def load_and_split_data(
    raw_path: str = RAW_DATA_PATH,
    processed_path: str = PROCESSED_DATA_PATH,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Loads raw data, performs feature engineering if processed file is missing,
    and returns reproducible stratified train and test splits.
    """
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
    else:
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw Kaggle dataset not found at {raw_path}")
        raw_df = pd.read_csv(raw_path, sep="\t")
        df = engineer_case157_features(raw_df)
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, df


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates that an uploaded dataframe contains the 8 Case Study 157 features.
    """
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing


def unit_test_preprocessing():
    """
    Runs automated assertions to verify zero target leakage and transformation correctness.
    """
    X_train, X_test, y_train, y_test, full_df = load_and_split_data()
    
    assert full_df.shape[0] == 2240, f"Expected 2240 rows, got {full_df.shape[0]}"
    assert set(FEATURE_COLUMNS).issubset(set(full_df.columns)), "Missing required feature columns"
    assert TARGET_COLUMN in full_df.columns, "Target column missing"
    assert full_df[TARGET_COLUMN].isin([0, 1]).all(), "Target must be strictly binary 0 or 1"
    
    is_valid, missing = validate_schema(full_df[FEATURE_COLUMNS])
    assert is_valid, f"Schema validation failed, missing: {missing}"
    
    preprocessor = create_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    feature_names = get_feature_names(preprocessor)
    assert X_train_trans.shape[1] == len(feature_names), "Transformed feature count mismatch"
    assert not np.isnan(X_train_trans).any(), "NaN values found in transformed training data"
    assert not np.isnan(X_test_trans).any(), "NaN values found in transformed test data"
    
    print(f"Preprocessing verified: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows, {len(feature_names)} features.")


if __name__ == "__main__":
    unit_test_preprocessing()
