import pandas as pd
import joblib
import json
import os

# ── Load model and feature columns once at import time ───────────────────────
# These are loaded when the module is first imported, not on every prediction.
# This means FastAPI loads them once at startup — fast response times.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, 'models', 'churn_model.joblib'))
feature_cols = json.load(open(os.path.join(BASE_DIR, 'models', 'feature_columns.json')))


def predict_churn(customer_dict: dict) -> dict:
    """
    Takes a dictionary of customer features.
    Returns churn probability and risk label.

    Args:
        customer_dict: {feature_name: value} for all 28 features

    Returns:
        {
            'churn_probability': float (0-1),
            'risk_level': 'High' | 'Medium' | 'Low',
            'churn_prediction': int (0 or 1)
        }
    """
    # Build DataFrame with correct column order
    df = pd.DataFrame([customer_dict])[feature_cols]

    prob = float(model.predict_proba(df)[0][1])
    pred = int(model.predict(df)[0])

    if prob > 0.75:
        risk = 'High'
    elif prob > 0.45:
        risk = 'Medium'
    else:
        risk = 'Low'

    return {
        'churn_probability': round(prob, 3),
        'risk_level': risk,
        'churn_prediction': pred
    }


def predict_batch(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame of multiple customers.
    Returns the same DataFrame with churn_probability and risk_level columns added.

    Args:
        df_input: DataFrame with same columns as training data

    Returns:
        DataFrame with added prediction columns
    """
    df = df_input.copy()

    # Only keep columns the model knows about, in the right order
    X = df[feature_cols]

    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    df['churn_probability'] = probs.round(3)
    df['churn_prediction'] = preds
    df['risk_level'] = df['churn_probability'].apply(
        lambda p: 'High' if p > 0.75 else ('Medium' if p > 0.45 else 'Low')
    )

    return df


# ── Quick test when run directly ──────────────────────────────────────────────
if __name__ == '__main__':
    # Load a real customer row from the engineered dataset to test
    import numpy as np
    test_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'telco_engineered.csv'))
    test_row = test_df.drop(columns=['Churn']).iloc[1976]  # our known high-risk customer

    result = predict_churn(test_row.to_dict())
    print('Test prediction (should be High risk ~0.986):')
    print(result)

    # Test batch
    batch_result = predict_batch(test_df.drop(columns=['Churn']).head(5))
    print('\nBatch prediction (first 5 customers):')
    print(batch_result[['churn_probability', 'risk_level']])