import pandas as pd
import numpy as np
import shap
import joblib
import json
import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, required for FastAPI/Streamlit
import matplotlib.pyplot as plt

# ── Load model and feature columns once at import time ───────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, 'models', 'churn_model.joblib'))
feature_cols = json.load(open(os.path.join(BASE_DIR, 'models', 'feature_columns.json')))

explainer = shap.TreeExplainer(model)


def explain_customer(customer_row: pd.Series, top_n: int = 3) -> str:
    """
    Takes a pandas Series (one customer, no Churn column).
    Returns a plain-English explanation of the top churn drivers.

    Args:
        customer_row: pd.Series with 28 feature values
        top_n: number of top drivers to return (default 3)

    Returns:
        String like: "Top churn drivers: tenure increases churn risk (SHAP: 0.46), ..."
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


def get_shap_values(customer_row: pd.Series) -> dict:
    """
    Returns raw SHAP values for a customer as a sorted dict.
    Used by the agent to generate retention recommendations.

    Returns:
        {'feature_name': shap_value, ...} sorted by absolute value descending
    """
    shap_vals = explainer.shap_values(
        customer_row.values.reshape(1, -1)
    )[0]

    pairs = list(zip(feature_cols, shap_vals))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    return {feat: round(float(val), 4) for feat, val in pairs}


def get_waterfall_figure(customer_row: pd.Series) -> plt.Figure:
    """
    Returns a matplotlib Figure of the SHAP waterfall plot for one customer.
    Used by the Streamlit frontend to display in st.pyplot().

    Args:
        customer_row: pd.Series with 28 feature values

    Returns:
        matplotlib Figure object
    """
    shap_vals = explainer.shap_values(
        customer_row.values.reshape(1, -1)
    )[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals,
            base_values=explainer.expected_value,
            data=customer_row.values,
            feature_names=feature_cols
        ),
        show=False
    )
    plt.tight_layout()
    return plt.gcf()


# ── Quick test when run directly ──────────────────────────────────────────────
if __name__ == '__main__':
    test_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'telco_engineered.csv'))
    X = test_df.drop(columns=['Churn'])

    # Test high-risk customer
    high_risk_row = X.iloc[1976]
    print('── High-risk customer explanation ─────────────────')
    print(explain_customer(high_risk_row))

    # Test low-risk customer
    low_risk_row = X.iloc[1815]
    print('\n── Low-risk customer explanation ──────────────────')
    print(explain_customer(low_risk_row))

    # Test raw SHAP values
    print('\n── Raw SHAP values (top 5) ────────────────────────')
    shap_dict = get_shap_values(high_risk_row)
    for feat, val in list(shap_dict.items())[:5]:
        print(f'  {feat}: {val}')

    print('\nexplain.py is working correctly.')
