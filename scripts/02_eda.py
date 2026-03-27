import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load raw file (no encode yet)
df = pd.read_csv('data/raw/Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# find the churn rate
print('Churn rate:')
print(df['Churn'].value_counts(normalize=True).round(3))

plt.figure(figsize=(5, 4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.tight_layout()
plt.savefig('data/processed/plot_01_churn_distribution.png')
plt.close()
print('Saved plot 1')

# chorn by contract type
df_plot = df.copy()
df_plot['Churned'] = (df['Churn'] == 'Yes').astype(int)

plt.figure(figsize=(6, 4))
sns.barplot(x='Contract', y='Churned', data=df_plot)
plt.title('Churn Rate by Contract Type')
plt.ylabel('Churn Rate')
plt.tight_layout()
plt.savefig('data/processed/plot_02_churn_by_contract.png')
plt.close()
print('Saved plot 2')

# histplot
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30)
plt.title('Tenure Distribution by Churn')
plt.tight_layout()
plt.savefig('data/processed/plot_03_tenure_by_churn.png')
plt.close()
print('Saved plot 3')

# boxplot
plt.figure(figsize=(5, 4))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges by Churn')
plt.tight_layout()
plt.savefig('data/processed/plot_04_charges_by_churn.png')
plt.close()
print('Saved plot 4')

# heatmap
df_clean = pd.read_csv('data/processed/telco_clean.csv')
plt.figure(figsize=(14, 10))
sns.heatmap(df_clean.select_dtypes(include='number').corr(),
            annot=True, fmt='.1f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('data/processed/plot_05_correlation_heatmap.png')
plt.close()
print('Saved plot 5')

print('\nAll EDA plots saved to data/processed/')