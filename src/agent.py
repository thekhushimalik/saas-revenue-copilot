"""
ChurnIQ - LangChain Agent
Compatible with: langchain 1.2.13, langchain-groq 1.1.2, langgraph 1.1.3

Run from project root: python -m src.agent
"""

import os
import json
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from src.predict import predict_batch
from src.explain import explain_customer

# ─────────────────────────────────────────────
# Load data once at module import time
# ─────────────────────────────────────────────
_DATA_PATH = "data/processed/telco_engineered.csv"
_FEATURE_COLS_PATH = "models/feature_columns.json"

_df_raw = pd.read_csv(_DATA_PATH)
_feature_cols = json.load(open(_FEATURE_COLS_PATH))

# Add churn_probability column if not present
if "churn_probability" not in _df_raw.columns:
    X = _df_raw[_feature_cols]
    _df_raw["churn_probability"] = predict_batch(X)["churn_probability"].values

# Add a stable customer_id column based on DataFrame index
_df_raw["customer_id"] = _df_raw.index.astype(str)

_df = _df_raw.copy()


def _get_customer_row(customer_id: str) -> pd.Series:
    """Return one customer row by integer index (as string)."""
    idx = int(customer_id)
    if idx not in _df.index:
        raise ValueError(f"Customer ID {customer_id} not found. Valid range: 0 to {len(_df) - 1}.")
    return _df.loc[idx]


# ─────────────────────────────────────────────
# Tool 1 — get_at_risk_customers
# ─────────────────────────────────────────────
@tool
def get_at_risk_customers(threshold: str) -> str:
    """
    Returns customers whose churn probability is at or above the given threshold.
    Input: a float between 0 and 1 as a string, e.g. '0.75'.
    Returns a summary string with customer IDs and their churn probabilities.
    """
    try:
        thresh = float(threshold.strip())
    except ValueError:
        return f"Invalid threshold '{threshold}'. Please provide a number between 0 and 1."

    at_risk = _df[_df["churn_probability"] >= thresh][["customer_id", "churn_probability"]]

    if at_risk.empty:
        return f"No customers found above threshold {thresh:.2f}."

    at_risk_sorted = at_risk.sort_values("churn_probability", ascending=False)
    top = at_risk_sorted.head(10)

    lines = [f"Found {len(at_risk)} customers at or above {thresh:.0%} churn probability."]
    lines.append("Top 10 highest risk:")
    for _, row in top.iterrows():
        lines.append(f"  Customer {row['customer_id']}: {row['churn_probability']:.1%}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Tool 2 — explain_customer_risk
# ─────────────────────────────────────────────
@tool
def explain_customer_risk(customer_id: str) -> str:
    """
    Explains why a specific customer is at churn risk using SHAP feature importance.
    Input: customer ID as a string (integer index), e.g. '1976'.
    Returns a plain-English explanation of the top churn drivers for that customer.
    """
    try:
        row = _get_customer_row(customer_id.strip())
    except ValueError as e:
        return str(e)

    feature_row = row[_feature_cols]
    prob = row["churn_probability"]
    explanation = explain_customer(feature_row)

    return (
        f"Customer {customer_id} has a {prob:.1%} churn probability.\n"
        f"{explanation}"
    )


# ─────────────────────────────────────────────
# Tool 3 — suggest_retention_actions
# ─────────────────────────────────────────────
@tool
def suggest_retention_actions(customer_id: str) -> str:
    """
    Suggests specific retention actions for a customer based on their top churn risk drivers.
    Input: customer ID as a string (integer index), e.g. '1976'.
    Returns actionable retention recommendations tailored to that customer.
    """
    # Action map: feature name fragment → recommendation
    ACTION_MAP = {
        "Contract_One year": "Offer a discount to upgrade from month-to-month to a 1-year contract.",
        "Contract_Two year": "Offer a loyalty incentive to upgrade to a 2-year contract.",
        "high_value_at_risk": "Flag for immediate outreach — high monthly spend on a month-to-month plan. Offer a loyalty discount.",
        "tenure_group": "New customer at risk of early churn. Assign a dedicated onboarding specialist.",
        "MonthlyCharges": "Monthly charges are a top risk driver. Offer a temporary bill credit or downgrade option.",
        "TotalCharges": "Billing history suggests value sensitivity. Consider a long-term rate lock.",
        "TechSupport": "No tech support subscription. Offer a free 3-month trial of tech support.",
        "OnlineSecurity": "No online security plan. Offer bundled security at a discount.",
        "OnlineBackup": "No online backup. Offer free trial — bundled customers churn significantly less.",
        "support_bundle": "Customer lacks the full support bundle. Offer a bundled support package at a reduced rate.",
        "services_count": "Low service engagement. Show value by offering a curated service bundle.",
        "avg_monthly_to_total_ratio": "Recent spending spike relative to lifetime value. Check if billing surprise triggered dissatisfaction.",
        "InternetService": "Internet service type is a risk driver. Check if faster or cheaper tier is available.",
        "PaymentMethod": "Payment method may indicate friction. Offer auto-pay discount.",
    }

    try:
        row = _get_customer_row(customer_id.strip())
    except ValueError as e:
        return str(e)

    feature_row = row[_feature_cols]

    # Get SHAP explanation to find top drivers
    explanation = explain_customer(feature_row)

    # Parse top features from explanation string
    actions = []
    for feature_fragment, action in ACTION_MAP.items():
        if feature_fragment.lower() in explanation.lower():
            actions.append(f"• {action}")
        if len(actions) >= 3:
            break

    if not actions:
        actions = [
            "• Schedule a check-in call to understand pain points.",
            "• Offer a loyalty discount on their current plan.",
            "• Assign to proactive support queue for 30 days.",
        ]

    prob = row["churn_probability"]
    lines = [
        f"Retention recommendations for Customer {customer_id} ({prob:.1%} churn risk):",
        "",
    ] + actions

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Tool 4 — churn_summary_stats
# ─────────────────────────────────────────────
@tool
def churn_summary_stats(_: str) -> str:
    """
    Returns overall churn statistics for the full customer dataset.
    Input: pass an empty string. No input needed.
    Returns total customers, churn rate, and high/medium/low risk counts.
    """
    total = len(_df)
    avg_churn_rate = _df["churn_probability"].mean()

    high_risk = (_df["churn_probability"] >= 0.75).sum()
    medium_risk = ((_df["churn_probability"] >= 0.45) & (_df["churn_probability"] < 0.75)).sum()
    low_risk = (_df["churn_probability"] < 0.45).sum()

    avg_charges_high = _df[_df["churn_probability"] >= 0.75]["MonthlyCharges"].mean()
    avg_charges_low = _df[_df["churn_probability"] < 0.45]["MonthlyCharges"].mean()

    return (
        f"ChurnIQ Dataset Summary\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Total customers:          {total:,}\n"
        f"Average churn probability: {avg_churn_rate:.1%}\n"
        f"\n"
        f"Risk breakdown:\n"
        f"  High risk  (≥75%):  {high_risk:,} customers\n"
        f"  Medium risk (45–75%): {medium_risk:,} customers\n"
        f"  Low risk   (<45%):  {low_risk:,} customers\n"
        f"\n"
        f"Avg monthly charges:\n"
        f"  High-risk customers:   ${avg_charges_high:.2f}\n"
        f"  Low-risk customers:    ${avg_charges_low:.2f}\n"
    )


# ─────────────────────────────────────────────
# Build the agent
# ─────────────────────────────────────────────
def build_agent():
    """
    Builds and returns the LangGraph ReAct agent.
    Call this once and reuse the returned agent.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set.\n"
            "Run: export GROQ_API_KEY=your_api_key_here"
        )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        temperature=0,
    )

    tools = [
        get_at_risk_customers,
        explain_customer_risk,
        suggest_retention_actions,
        churn_summary_stats,
    ]

    system_prompt = (
        "You are ChurnIQ, a SaaS revenue intelligence analyst. "
        "You have access to a trained XGBoost churn prediction model and live customer data. "
        "Always use your tools to ground answers in real data before responding. "
        "Be specific, concise, and actionable. "
        "When citing churn probabilities, express them as percentages. "
        "Never guess — if you need data, call the appropriate tool."
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    return agent


def run_agent(agent, user_message: str) -> str:
    """
    Runs the agent with a user message and returns the final text response.
    Use this in FastAPI and Streamlit instead of calling the agent directly.
    """
    result = agent.invoke({"messages": [("human", user_message)]})
    # LangGraph returns a dict with 'messages'; last message is the AI response
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return "I was unable to generate a response. Please try again."


# ─────────────────────────────────────────────
# Quick test when run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Building ChurnIQ agent...")
    agent = build_agent()
    print("Agent ready. Type 'quit' to exit.\n")

    test_questions = [
        "Give me a summary of overall churn stats.",
        "Which customers are at highest risk? Use threshold 0.85.",
        "Explain why customer 1976 is at risk.",
        "What should we do to retain customer 1976?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        print(f"A: {run_agent(agent, q)}")
        print("─" * 60)