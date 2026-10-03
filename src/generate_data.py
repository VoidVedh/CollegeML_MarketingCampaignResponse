"""
[LEGACY / DEPRECATED] generate_data.py

WARNING: This module was the initial synthetic simulation generator and is now
retained strictly for historical reference. The active, production machine learning
pipeline has been fully migrated to the real Kaggle Marketing Dataset:
https://www.kaggle.com/competitions/marketing-dataset/data (data/kaggle/train.csv).

This synthetic generator is NOT used in the final model, report, or deployed application.
"""

import os
import numpy as np
import pandas as pd

def generate_campaign_dataset(
    n_samples: int = 5000,
    random_state: int = 42,
    output_path: str = "data/campaign_data.csv"
) -> pd.DataFrame:
    """
    Generate realistic synthetic marketing campaign response data.
    
    Columns:
      - age_group: categorical ('18-25', '26-35', '36-45', '46-55', '56+')
      - income: numeric, right-skewed (~$15,000 - $150,000)
      - previous_purchases: int (0 to ~40)
      - purchase_frequency: float purchases/month (~0.1 to ~8.0)
      - previous_campaign_response: binary (0 or 1)
      - website_visits: int visits/month (0 to ~25)
      - email_engagement: float (0.0 to 1.0 open/click engagement score)
      - discount_usage: float (0.0 to 1.0 share of purchases with discount)
      - responded: binary target (0 or 1, ~15-20% positive class rate)
    """
    np.random.seed(random_state)
    
    # 1. Age group
    age_categories = ['18-25', '26-35', '36-45', '46-55', '56+']
    age_probs = [0.15, 0.32, 0.28, 0.15, 0.10]
    age_group = np.random.choice(age_categories, size=n_samples, p=age_probs)
    
    # Age numeric proxy for correlation logic
    age_numeric_map = {'18-25': 22, '26-35': 30, '36-45': 40, '46-55': 50, '56+': 62}
    age_num = np.array([age_numeric_map[a] for a in age_group])
    
    # 2. Income: log-normal right-skewed distribution conditioned somewhat on age
    # Base income roughly 20k to 150k
    log_mean = 10.7 + 0.005 * (age_num - 20)  # median income around $45k - $60k
    log_std = 0.52
    income = np.random.lognormal(mean=log_mean, sigma=log_std, size=n_samples)
    income = np.clip(income, 15000, 165000)
    
    # 3. Previous purchases: poisson distribution tied slightly to age & income
    purchases_lambda = np.clip(3 + 0.00004 * income + 0.05 * age_num, 1, 25)
    previous_purchases = np.random.poisson(lam=purchases_lambda)
    
    # 4. Purchase frequency (purchases per month): Gamma distribution
    purchase_frequency = np.random.gamma(shape=2.5, scale=0.8, size=n_samples) + 0.1 * (previous_purchases > 10)
    purchase_frequency = np.round(np.clip(purchase_frequency, 0.1, 10.0), 2)
    
    # 5. Previous campaign response: Bernoulli with baseline 18%
    # Customers with higher purchases and email engagement more likely responded in past
    prev_response_logits = -1.9 + 0.04 * previous_purchases + 0.15 * purchase_frequency
    prev_response_prob = 1 / (1 + np.exp(-prev_response_logits))
    prev_response_prob = np.clip(prev_response_prob, 0.05, 0.45)
    previous_campaign_response = np.random.binomial(n=1, p=prev_response_prob)
    
    # 6. Website visits per month: Negative binomial (skewed right)
    website_visits = np.random.negative_binomial(n=3, p=0.25, size=n_samples)
    website_visits = np.clip(website_visits, 0, 35)
    
    # 7. Email engagement: Beta distribution (0 to 1)
    email_engagement = np.random.beta(a=2.0, b=3.5, size=n_samples)
    # Higher for previous responders
    email_engagement = np.clip(email_engagement + 0.18 * previous_campaign_response, 0.0, 1.0)
    email_engagement = np.round(email_engagement, 3)
    
    # 8. Discount usage: Beta distribution (0 to 1)
    discount_usage = np.random.beta(a=2.2, b=3.0, size=n_samples)
    discount_usage = np.round(np.clip(discount_usage, 0.0, 1.0), 3)
    
    # 9. Target: responded (0/1)
    # Specified drivers:
    # - previous_campaign_response, email_engagement, purchase_frequency, discount_usage: strongest positive drivers
    # - income and age: weak or moderate effects
    # - target class balance: ~15-20%
    
    # Standardize features for linear combination
    inc_std = (income - np.mean(income)) / np.std(income)
    freq_std = (purchase_frequency - np.mean(purchase_frequency)) / np.std(purchase_frequency)
    purch_std = (previous_purchases - np.mean(previous_purchases)) / np.std(previous_purchases)
    visits_std = (website_visits - np.mean(website_visits)) / np.std(website_visits)
    email_std = (email_engagement - np.mean(email_engagement)) / np.std(email_engagement)
    disc_std = (discount_usage - np.mean(discount_usage)) / np.std(discount_usage)
    
    # Age group effect: 26-35 and 36-45 have slight positive propensity
    age_effect = np.array([
        0.10 if a == '26-35' else (0.15 if a == '36-45' else (-0.10 if a == '18-25' else 0.0))
        for a in age_group
    ])
    
    # Target latent log-odds
    # Intercept tuned to hit ~17-18% response rate
    latent_z = (
        -2.75
        + 1.85 * previous_campaign_response    # Strong driver 1
        + 1.35 * email_std                     # Strong driver 2
        + 0.70 * freq_std                      # Strong driver 3
        + 0.55 * disc_std                      # Strong driver 4
        + 0.25 * purch_std                     # Moderate driver
        + 0.20 * visits_std                    # Moderate driver
        + 0.12 * inc_std                       # Weak driver
        + age_effect                           # Weak/moderate categorical driver
        + 0.40 * (email_std * disc_std)        # Interaction effect: discount-seekers engaged with email
        + np.random.normal(0, 0.45, size=n_samples) # Realistic noise
    )
    
    response_prob = 1 / (1 + np.exp(-latent_z))
    responded = (np.random.uniform(0, 1, size=n_samples) < response_prob).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age_group': age_group,
        'income': np.round(income, 2),
        'previous_purchases': previous_purchases,
        'purchase_frequency': purchase_frequency,
        'previous_campaign_response': previous_campaign_response,
        'website_visits': website_visits,
        'email_engagement': email_engagement,
        'discount_usage': discount_usage,
        'responded': responded
    })
    
    # Introduce ~2.5% missing values in a few columns to make cleaning realistic
    missing_mask_income = np.random.binomial(1, 0.025, size=n_samples).astype(bool)
    df.loc[missing_mask_income, 'income'] = np.nan
    
    missing_mask_email = np.random.binomial(1, 0.02, size=n_samples).astype(bool)
    df.loc[missing_mask_email, 'email_engagement'] = np.nan
    
    missing_mask_visits = np.random.binomial(1, 0.018, size=n_samples).astype(bool)
    df.loc[missing_mask_visits, 'website_visits'] = np.nan
    
    # Introduce a few realistic outliers in income and previous_purchases
    outlier_idx = np.random.choice(n_samples, size=15, replace=False)
    for idx in outlier_idx[:8]:
        df.loc[idx, 'income'] = np.random.uniform(220000, 310000)
    for idx in outlier_idx[8:]:
        df.loc[idx, 'previous_purchases'] = np.random.randint(65, 95)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    response_rate = df['responded'].mean() * 100
    print(f"Generated {len(df)} rows at '{output_path}'.")
    print(f"Target distribution: {df['responded'].value_counts().to_dict()} (Response Rate: {response_rate:.2f}%)")
    print(f"Missing values per column:\n{df.isna().sum()[df.isna().sum() > 0]}")
    
    return df

if __name__ == "__main__":
    generate_campaign_dataset()
