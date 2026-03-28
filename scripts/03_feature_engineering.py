import pandas as pd
import numpy as np
import shap 
import joblib
import json
import matplotlib.pyplot as plt

# ── Load model and data ───────────────────────────────────────────────────────
model = joblib.load('models/churn_model.joblib')
feature_cols = json.load(open('models/feature_columns.json'))

df = pd.read_csv('data/processed/telco_engineered.csv')
X = df.drop(columns=['Churn'])
y = df['Churn']

print('Model and data loaded.')
print('Features:', len(feature_cols))

# ── Create SHAP explainer ─────────────────────────────────────────────────────
# TreeExplainer is optimised for tree-based models like XGBoost.
# It's much faster than the generic KernelExplainer.
explainer = shap.TreeExplainer(model)

# Compute SHAP values for the full dataset (used for summary plots)
# This may take 10-20 seconds
print('Computing SHAP values...')
shap_values = explainer.shap_values(X)
print('SHAP values computed. Shape:', shap_values.shape)

# ── Plot 1: Summary bar plot (global feature importance) ─────────────────────
# Shows which features matter most ACROSS ALL customers.
# This is the "which features drive churn in general" view.
plt.figure()
shap.summary_plot(shap_values, X, plot_type='bar', show=False)
plt.title('Global Feature Imporance (SHAP)')
plt.tight_layout()
plt.savefig('data/processed/plot_06_shap_summary_bar.png', bbox_inches='tight')
plt.close()
print('Saved plot 6: SHAP summary bar')

# ── Plot 2: Summary dot plot (direction + magnitude) ─────────────────────────
# Red dots = high feature value, Blue = low feature value
# Position on x-axis = impact on churn prediction
# e.g. high tenure (red) on the left = reduces churn risk
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.title('SHAP Feature Impact (Direction + Magnitude)')
plt.tight_layout()
plt.savefig('data/processed/plot_07_shap_summary_dot.png', bbox_inches='tight')
plt.close()
print('Saved plot 7: SHAP summary dot')

# ── Plot 3: Waterfall plot for a single high-risk customer ───────────────────
# Pick a customer the model predicted as high churn risk
y_prob = model.predict_proba(X)[:, 1]
high_risk_idx = y_prob.argmax()  # customer with highest churn probability
print(f'\nHigh-risk customer index: {high_risk_idx}')
print(f'Predicted churn probability: {y_prob[high_risk_idx]:.3f}')
print(f'Actual churn: {y.iloc[high_risk_idx]}')

plt.figure()
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[high_risk_idx],
        base_values=explainer.expected_value,
        data=X.iloc[high_risk_idx].values,
        feature_names=feature_cols
    ),
    show=False
)
plt.title(f'Customer {high_risk_idx} — Churn Risk Breakdown')
plt.tight_layout()
plt.savefig('data/processed/plot_08_shap_waterfall.png', bbox_inches='tight')
plt.close()
print('Saved plot 8: SHAP waterfall for high-risk customer')

# ── explain_customer() function ───────────────────────────────────────────────
# This is the core function used by the LangChain agent and FastAPI.
# Takes one customer row, returns a plain-English explanation string.
def explain_customer(customer_row, top_n=3):
    """
    Takes a pandas Series (one customer row, no Churn column).
    Returns a plain-English explanation of the top churn drivers.
    """
    shap_vals = explainer.shap_values(
        customer_row.values.reshape(1, -1)
    )[0]

    pairs = list(zip(feature_cols, shap_vals))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    top = pairs[:top_n]

    parts = []
    for feat, val in top:
        direction = 'increases' if val > 0 else 'decreases'
        parts.append(f'{feat} {direction} churn risk (SHAP: {val:.2f})')

    return 'Top churn drivers: ' + ', '.join(parts)

# ── Test explain_customer() on the high-risk customer ────────────────────────
print('\n── Testing explain_customer() ─────────────────────────')
test_row = X.iloc[high_risk_idx]
explanation = explain_customer(test_row)
print(explanation)

# ── Test on a low-risk customer ───────────────────────────────────────────────
low_risk_idx = y_prob.argmin()
print(f'\nLow-risk customer index: {low_risk_idx}')
print(f'Predicted churn probability: {y_prob[low_risk_idx]:.3f}')
test_row_low = X.iloc[low_risk_idx]
explanation_low = explain_customer(test_row_low)
print(explanation_low)

print('\nAll SHAP plots saved to data/processed/')
print('explain_customer() function is ready for use in src/explain.py')