import pandas as pd
import numpy as np

# loading the raw file
df = pd.read_csv('data/raw/Telco-Customer-Churn.csv')
print('Shape:', df.shape)          
print('Dtypes:\n', df.dtypes)

#fix the total charge colmn
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print('TotalCharges nulls:', df['TotalCharges'].isna().sum())  
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

#remove customer ID thats of no use 
df.drop(columns=['customerID'], inplace=True)

#churn binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# other columns as binary
binary_cols = [
    'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0,
                           'No internet service': 0,
                           'No phone service': 0})

# one hot encode
df = pd.get_dummies(df, columns=['gender', 'InternetService', 'Contract', 'PaymentMethod'], drop_first=True)

# save the clean file
df.to_csv('data/processed/telco_clean.csv', index=False)
print('Saved. Shape:', df.shape)
print('Columns:', df.columns.tolist())