# ChurnIQ — SaaS Revenue Intelligence Co-pilot

> Predict who's leaving. Understand why. Act before they go.

**[🚀 Live Demo](https://huggingface.co/spaces/your-username/churniq-revenue-copilot)** · Built by Khushi Malik

---

## What it does

ChurnIQ is an end-to-end machine learning system that predicts customer churn for SaaS businesses and explains every prediction in plain English through an AI analyst interface. It takes raw customer data, runs it through a trained XGBoost model, surfaces the exact features driving each risk score via SHAP, and lets business users interrogate the results through a conversational LangChain agent — no SQL, no dashboards, just questions and answers.

**The problem it solves:** SaaS companies lose revenue to churn they never saw coming. This tool tells you which customers are about to leave and exactly why, so your retention team can act before they do.

---

## Tech stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost (gradient boosted trees) |
| Explainability | SHAP (SHapley Additive exPlanations) |
| AI Analyst | LangChain + Groq (llama-3.1-8b-instant) |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Deployment | HuggingFace Spaces |
| Data | IBM Telco Customer Churn dataset |

---

## ML pipeline

### 1 · Data Preparation
Raw IBM Telco CSV → clean, properly typed pandas DataFrame. Handles the known `TotalCharges` string bug (11 blank values), drops the non-predictive `customerID` column, encodes binary columns via label encoding, and one-hot encodes multi-category columns (`Contract`, `PaymentMethod`, `InternetService`).

### 2 · Exploratory Data Analysis
Key findings: ~26% overall churn rate (imbalanced dataset), month-to-month contract customers churn at significantly higher rates than annual customers, short-tenure customers (0–12 months) are the highest-risk segment, and high monthly charges correlate with higher churn probability.

### 3 · Feature Engineering
Five engineered features derived from business insight:

| Feature | Logic |
|---|---|
| `tenure_group` | Bins raw tenure (0–72 months) into 4 risk buckets |
| `avg_monthly_to_total_ratio` | Flags billing anomalies for new or escalating accounts |
| `services_count` | Counts add-on services as a proxy for switching cost |
| `high_value_at_risk` | Binary flag: MonthlyCharges > $70 AND month-to-month contract |
| `support_bundle` | Binary flag: TechSupport AND OnlineBackup both active |

### 4 · Model Training
XGBoost classifier with `scale_pos_weight` to handle the 74/26 class imbalance. Trained on an 80/20 stratified split to preserve the churn ratio in both sets.

### 5 · SHAP Explainability
`TreeExplainer` generates per-prediction SHAP values. The `explain_customer()` function converts raw SHAP values into human-readable sentences ranking the top drivers of each customer's risk score. Output feeds both the waterfall chart in the UI and the LangChain agent's responses.

### 6 · LangChain AI Analyst Agent
A `ZERO_SHOT_REACT_DESCRIPTION` agent with 4 tools wired to the live model and dataset:

- **`get_at_risk_customers`** — returns all customers above a given churn probability threshold
- **`explain_customer_risk`** — runs SHAP on a specific customer and returns the explanation
- **`suggest_retention_actions`** — maps SHAP drivers to specific retention recommendations
- **`churn_summary_stats`** — returns dataset-wide churn statistics

### 7 · FastAPI Backend
Three endpoints: `POST /predict` (single customer), `POST /batch_predict` (CSV upload), `POST /chat` (routes to the LangChain agent). All model logic lives here — the frontend never touches the model directly.

### 8 · Streamlit Frontend
Three-tab interface: a dashboard with key metrics and a sortable risk table, a customer deep-dive with SHAP waterfall charts and retention action recommendations, and an AI analyst chat tab for freeform questions.

---

## Model performance

| Metric | Score |
|---|---|
| AUC-ROC | 0.847 |
| F1 Score (churn=1) | — |
| Recall (churn=1) | — |

*Update F1 and Recall with your actual test set results after training.*

---

## Run locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/churniq-revenue-copilot
cd churniq-revenue-copilot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
export GROQ_API_KEY=your_key_here

# 4. Download the dataset
# IBM Telco Customer Churn from Kaggle → data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 5. Run the notebooks in order (01 through 05) to generate processed data and trained model

# 6. Start the FastAPI backend
uvicorn app.main:app --reload

# 7. In a separate terminal, start the Streamlit frontend
streamlit run app/streamlit_app.py
```

---

## Project structure

```
saas-revenue-copilot/
├── data/
│   ├── raw/               # original CSV — never modified
│   └── processed/         # cleaned and engineered datasets
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_shap_explainability.ipynb
├── models/
│   ├── churn_model.joblib      # saved trained model
│   └── feature_columns.json   # feature list the model expects
├── src/
│   ├── predict.py
│   ├── explain.py
│   └── agent.py
├── app/
│   ├── main.py               # FastAPI backend
│   └── streamlit_app.py      # Streamlit frontend
├── requirements.txt
└── README.md
```

---

## Deploy to HuggingFace Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) — select **Streamlit** as the SDK
2. Push your code via git or the web UI
3. Add `GROQ_API_KEY` as a **Secret** in Space settings (never hardcode keys)
4. The Space builds and deploys automatically

---

*Built end-to-end — data prep, EDA, feature engineering, model training, explainability, LLM integration, API, frontend, and cloud deployment.*
