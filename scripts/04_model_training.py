import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

# ── Load engineered data ──────────────────────────────────────────────────────
df = pd.read_csv('data/processed/telco_engineered.csv')
print('Loaded shape:', df.shape)

X = df.drop(columns=['Churn'])
y = df['Churn']

print('Churn distribution:\n', y.value_counts())
print('Churn rate:', round(y.mean(), 3))

# ── Save feature column names ─────────────────────────────────────────────────
# The API and SHAP explainer need to know exactly which columns and in what
# order the model was trained on. Save this now, before splitting.
import os
os.makedirs('models', exist_ok=True)
json.dump(list(X.columns), open('models/feature_columns.json', 'w'))
print('Feature columns saved:', len(X.columns), 'features')

# ── Train/test split ──────────────────────────────────────────────────────────
# stratify=y preserves the 26/74 churn ratio in both train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}')

# ── Handle class imbalance ────────────────────────────────────────────────────
# 26% churners vs 74% non-churners. Without this, the model just predicts
# "no churn" for everyone and gets 74% accuracy — which is useless.
# scale_pos_weight tells XGBoost to penalise missed churners more heavily.
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = neg / pos
print(f'Class weight scale: {scale:.2f}  (neg={neg}, pos={pos})')

# ── Train XGBoost ─────────────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale,
    eval_metric='logloss',
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)
print('\nModel training complete.')

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print('\n── Classification Report ──────────────────────────────')
print(classification_report(y_test, y_pred, target_names=['Stay', 'Churn']))

auc = roc_auc_score(y_test, y_prob)
print(f'AUC-ROC: {auc:.4f}')

print('\n── Confusion Matrix ───────────────────────────────────')
cm = confusion_matrix(y_test, y_pred)
print(f'                Predicted Stay  Predicted Churn')
print(f'Actual Stay         {cm[0][0]:<10}      {cm[0][1]}')
print(f'Actual Churn        {cm[1][0]:<10}      {cm[1][1]}')

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(model, 'models/churn_model.joblib')
print('\nModel saved to models/churn_model.joblib')
print('Feature columns saved to models/feature_columns.json')