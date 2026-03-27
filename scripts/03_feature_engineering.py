import pandas as pd
import numpy as np

# load raw file
df = pd.read_csv('data/processed/telco_clean.csv')
print('Loaded shape:', df.shape)

# creating tenure group
df['tenure_group'] = pd.cut(
    df['tenure'],
    bins=[0, 12, 24, 48, 72],
    labels=[0, 1, 2, 3],
    include_lowest=True
)
df['tenure_group'] = df['tenure_group'].astype(int)
print('tenure_group value counts:\n', df['tenure_group'].value_counts().sort_index())

# ── Feature 2: avg_monthly_to_total_ratio ────────────────────────────────────
# Flags billing anomalies: high MonthlyCharges vs low TotalCharges
# means the customer is brand new or recently upgraded.
# +1 avoids division by zero for customers with zero TotalCharges.
df['avg_monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
print('avg_monthly_to_total_ratio sample:\n', df['avg_monthly_to_total_ratio'].describe().round(3))

# ── Feature 3: services_count ─────────────────────────────────────────────────
# Total number of add-on services per customer.
# More services = higher switching cost = lower churn probability.
service_cols = [
    'PhoneService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
df['services_count'] = df[service_cols].sum(axis=1)
print('services_count distribution:\n', df['services_count'].value_counts().sort_index())

# ── Feature 4: high_value_at_risk ────────────────────────────────────────────
# Business rule: customers paying >$70/month on a month-to-month contract
# are the highest churn risk segment.
# NOTE: get_dummies(drop_first=True) dropped Contract_Month-to-month because
# it's first alphabetically. A customer is on month-to-month when BOTH
# Contract_One year == 0 AND Contract_Two year == 0.
is_month_to_month = (df['Contract_One year'] == 0) & (df['Contract_Two year'] == 0)
df['high_value_at_risk'] = (
    (df['MonthlyCharges'] > 70) & is_month_to_month
).astype(int)
print('Month-to-month customers:', is_month_to_month.sum())
print('high_value_at_risk counts:\n', df['high_value_at_risk'].value_counts())

print('high_value_at_risk counts:\n', df['high_value_at_risk'].value_counts())

# ── Feature 5: support_bundle ─────────────────────────────────────────────────
# Customers with BOTH TechSupport AND OnlineBackup have a support bundle.
# Bundle customers have deeper product engagement and churn significantly less.
df['support_bundle'] = (
    (df['TechSupport'] == 1) &
    (df['OnlineBackup'] == 1)
).astype(int)
print('support_bundle counts:\n', df['support_bundle'].value_counts())

# ── Final check ───────────────────────────────────────────────────────────────
print('\nFinal shape:', df.shape)
print('New columns added:', ['tenure_group', 'avg_monthly_to_total_ratio',
                              'services_count', 'high_value_at_risk', 'support_bundle'])
print('Any nulls?', df.isnull().sum().sum())

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv('data/processed/telco_engineered.csv', index=False)
print('\nSaved to data/processed/telco_engineered.csv')