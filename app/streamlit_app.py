"""
ChurnIQ - Streamlit Frontend
Three tabs: Dashboard, Customer Deep-Dive, AI Analyst

Run from project root:
    streamlit run app/streamlit_app.py

Note: imports directly from src/ modules (no FastAPI server needed).
This makes it self-contained for HuggingFace Spaces deployment.
"""

import os
import sys
import pandas as pd

# ─────────────────────────────────────────────
# Ensure project root is on sys.path so `src` is importable
# regardless of how/where streamlit is launched from.
# ─────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

# ─────────────────────────────────────────────
# Page config — must be the very first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Imports from src/ — cached so they only run once
# ─────────────────────────────────────────────
from src.predict import predict_batch, feature_cols
from src.explain import explain_customer, get_waterfall_figure
from src.agent import build_agent, run_agent


# ─────────────────────────────────────────────
# Data loading — cached so CSV is read once per session
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading customer data...")
def load_data() -> pd.DataFrame:
    """Load engineered dataset and attach churn predictions."""
    df = pd.read_csv("data/processed/telco_engineered.csv")

    # Preserve original customerID if available in raw CSV
    try:
        raw = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
        df["customerID"] = raw["customerID"].values
    except Exception:
        df["customerID"] = ["CUST-" + str(i).zfill(4) for i in df.index]

    X = df[feature_cols]
    preds = predict_batch(X)
    df["churn_probability"] = preds["churn_probability"].values
    df["churn_prediction"] = preds["churn_prediction"].values
    df["risk_level"] = preds["risk_level"].values

    return df


@st.cache_resource(show_spinner="Loading AI analyst...")
def load_agent():
    """Build the LangGraph agent once and cache it for the session."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return None
    try:
        return build_agent()
    except Exception:
        return None


# ─────────────────────────────────────────────
# Minimal CSS — clean card style + risk colour badges
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.82rem;
        color: #a0a0b8;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #e0e0ff;
    }
    .metric-card .delta {
        font-size: 0.78rem;
        color: #6af06a;
        margin-top: 4px;
    }
    .risk-high   { color: #ff4d4d; font-weight: 700; }
    .risk-medium { color: #ffa940; font-weight: 700; }
    .risk-low    { color: #52c41a; font-weight: 700; }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #c0c0d8;
        margin: 16px 0 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("## 📉 ChurnIQ — SaaS Revenue Intelligence Co-pilot")
st.caption("XGBoost · SHAP Explainability · LangChain AI Analyst")

# ─────────────────────────────────────────────
# Load data (blocks until ready)
# ─────────────────────────────────────────────
df = load_data()

# ─────────────────────────────────────────────
# Precompute summary stats used in multiple tabs
# ─────────────────────────────────────────────
total_customers = len(df)
overall_churn_rate = df["churn_probability"].mean()
high_risk_count = (df["risk_level"] == "High").sum()
medium_risk_count = (df["risk_level"] == "Medium").sum()
low_risk_count = (df["risk_level"] == "Low").sum()

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Customer Deep-Dive", "🤖 AI Analyst"])


# ══════════════════════════════════════════════
# TAB 1 — Dashboard
# ══════════════════════════════════════════════
with tab1:

    # ── Top metrics row ───────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Total Customers</div>
                <div class="value">{total_customers:,}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Avg Churn Probability</div>
                <div class="value">{overall_churn_rate:.1%}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">High Risk</div>
                <div class="value" style="color:#ff4d4d">{high_risk_count:,}</div>
                <div class="delta">≥ 75% probability</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Medium Risk</div>
                <div class="value" style="color:#ffa940">{medium_risk_count:,}</div>
                <div class="delta">45 – 75% probability</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Model AUC-ROC</div>
                <div class="value">0.835</div>
                <div class="delta">test set</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filters row ──────────────────────────
    st.markdown('<div class="section-title">Customer Risk Table</div>', unsafe_allow_html=True)

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 3])

    with filter_col1:
        risk_filter = st.multiselect(
            "Filter by risk level",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )
    with filter_col2:
        prob_min_pct = st.slider(
            "Min churn probability",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            format="%d%%",
        )
        prob_min = prob_min_pct / 100

    # ── Build display table ───────────────────
    display_df = df[
        (df["risk_level"].isin(risk_filter)) &
        (df["churn_probability"] >= prob_min)
    ].copy()

    display_df = display_df.sort_values("churn_probability", ascending=False)

    # Columns to show — keep it readable
    show_cols = [
        "customerID", "churn_probability", "risk_level",
        "tenure", "MonthlyCharges", "TotalCharges",
        "Contract_One year", "Contract_Two year",
        "services_count", "high_value_at_risk",
    ]
    show_cols = [c for c in show_cols if c in display_df.columns]

    # Format probability as percentage string for display
    table_df = display_df[show_cols].copy()
    table_df["churn_probability"] = table_df["churn_probability"].apply(
        lambda x: f"{x:.1%}"
    )

    # Colour-code risk level column
    def colour_risk(val):
        colours = {"High": "color: #ff4d4d", "Medium": "color: #ffa940", "Low": "color: #52c41a"}
        return colours.get(val, "")

    styled = table_df.style.map(colour_risk, subset=["risk_level"])

    st.dataframe(styled, use_container_width=True, height=420)
    st.caption(f"Showing {len(display_df):,} of {total_customers:,} customers")


# ══════════════════════════════════════════════
# TAB 2 — Customer Deep-Dive
# ══════════════════════════════════════════════
with tab2:

    left, right = st.columns([1, 2])

    with left:
        st.markdown('<div class="section-title">Select Customer</div>', unsafe_allow_html=True)

        # Search by ID or pick from high-risk list
        search_mode = st.radio(
            "Find customer by",
            ["Customer ID (index)", "Browse high-risk list"],
            horizontal=True,
        )

        if search_mode == "Customer ID (index)":
            selected_idx = st.number_input(
                "Enter customer index (0 – 7042)",
                min_value=0,
                max_value=len(df) - 1,
                value=1976,
                step=1,
            )
        else:
            high_risk_df = df[df["risk_level"] == "High"].sort_values(
                "churn_probability", ascending=False
            )
            options = [
                f"{int(i)} — {row['customerID']} ({row['churn_probability']:.1%})"
                for i, row in high_risk_df.iterrows()
            ]
            choice = st.selectbox("High-risk customers", options)
            selected_idx = int(choice.split("—")[0].strip())

        # ── Customer snapshot ─────────────────
        customer = df.loc[selected_idx]
        prob = customer["churn_probability"]
        risk = customer["risk_level"]

        risk_colour = {"High": "#ff4d4d", "Medium": "#ffa940", "Low": "#52c41a"}.get(risk, "#ffffff")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Customer {customer['customerID']}</div>
                <div class="value" style="color:{risk_colour}">{prob:.1%}</div>
                <div class="delta" style="color:{risk_colour};">{risk} Risk</div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Key Attributes</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        col_a.metric("Tenure", f"{int(customer['tenure'])} mo")
        col_b.metric("Monthly Charges", f"${customer['MonthlyCharges']:.2f}")
        col_a.metric("Services", int(customer["services_count"]))
        col_b.metric("Total Charges", f"${customer['TotalCharges']:.2f}")

        # Month-to-month flag
        is_mtm = (
            customer.get("Contract_One year", 0) == 0
            and customer.get("Contract_Two year", 0) == 0
        )
        contract_label = (
            "Month-to-month" if is_mtm
            else ("1-year" if customer.get("Contract_One year", 0) == 1 else "2-year")
        )
        st.markdown(f"**Contract:** {contract_label}")
        st.markdown(f"**High-value at risk:** {'✅ Yes' if customer.get('high_value_at_risk', 0) == 1 else '—'}")

    with right:
        # ── SHAP waterfall ────────────────────
        st.markdown('<div class="section-title">Why is this customer at risk?</div>', unsafe_allow_html=True)

        feature_row = customer[feature_cols]
        with st.spinner("Generating SHAP explanation..."):
            fig = get_waterfall_figure(feature_row)
        st.pyplot(fig, use_container_width=True)

        # ── Plain-English explanation ─────────
        explanation = explain_customer(feature_row, top_n=3)
        st.info(f"**Model says:** {explanation}")

        # ── Retention actions ─────────────────
        st.markdown('<div class="section-title">Recommended Retention Actions</div>', unsafe_allow_html=True)

        ACTION_MAP = {
            "Contract_One year": "Offer a discount to upgrade from month-to-month to a 1-year contract.",
            "Contract_Two year": "Offer a loyalty incentive to upgrade to a 2-year contract.",
            "high_value_at_risk": "🚨 Flag for immediate outreach — high spend on month-to-month plan. Offer loyalty discount.",
            "tenure_group": "New customer at early-churn risk. Assign a dedicated onboarding specialist.",
            "MonthlyCharges": "Monthly charges are a top risk driver. Offer a temporary bill credit or downgrade option.",
            "TechSupport": "No tech support plan. Offer a free 3-month trial.",
            "OnlineSecurity": "No online security plan. Offer bundled security at a discount.",
            "OnlineBackup": "No online backup. Bundled customers churn significantly less — offer a free trial.",
            "support_bundle": "Lacks the full support bundle. Offer a bundled package at a reduced rate.",
            "services_count": "Low service engagement. Offer a curated service bundle to increase switching costs.",
            "avg_monthly_to_total_ratio": "Recent billing spike vs lifetime value. Check if a billing surprise triggered dissatisfaction.",
            "PaymentMethod": "Payment method may cause friction. Offer an auto-pay discount.",
        }

        actions_found = []
        for fragment, action in ACTION_MAP.items():
            if fragment.lower() in explanation.lower():
                actions_found.append(action)
            if len(actions_found) >= 3:
                break

        if not actions_found:
            actions_found = [
                "Schedule a check-in call to understand pain points.",
                "Offer a loyalty discount on their current plan.",
                "Assign to proactive support queue for 30 days.",
            ]

        for action in actions_found:
            st.markdown(f"• {action}")


# ══════════════════════════════════════════════
# TAB 3 — AI Analyst Chat
# ══════════════════════════════════════════════
with tab3:

    st.markdown('<div class="section-title">Ask ChurnIQ Anything</div>', unsafe_allow_html=True)
    st.caption(
        "The AI analyst is backed by your trained model and live customer data. "
        "Ask about risk levels, specific customers, retention strategies, or summary stats."
    )

    # ── Check agent availability ──────────────
    agent = load_agent()

    if agent is None:
        st.warning(
            "⚠️ AI Analyst unavailable. "
            "Set the `GROQ_API_KEY` environment variable and restart the app.\n\n"
            "```bash\nexport GROQ_API_KEY=your_key_here\nstreamlit run app/streamlit_app.py\n```"
        )
    else:
        # ── Suggested prompts ─────────────────
        st.markdown("**Try asking:**")
        suggestions = [
            "Give me a summary of overall churn stats.",
            "Which customers are at highest risk? Use threshold 0.85.",
            "Explain why customer 1976 is at risk.",
            "What should we do to retain customer 1976?",
        ]

        suggestion_cols = st.columns(len(suggestions))
        for col, suggestion in zip(suggestion_cols, suggestions):
            if col.button(suggestion, use_container_width=True):
                st.session_state["chat_input"] = suggestion

        st.markdown("---")

        # ── Chat history ──────────────────────
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Render existing messages
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ── Input box ─────────────────────────
        # Pre-fill from suggestion button if clicked
        prefill = st.session_state.pop("chat_input", "")

        user_input = st.chat_input(
            "Ask about your customers...",
        )

        # Accept input from either the chat box or a suggestion button
        final_input = user_input or prefill

        if final_input:
            # Show user message immediately
            st.session_state["chat_history"].append(
                {"role": "user", "content": final_input}
            )
            with st.chat_message("user"):
                st.markdown(final_input)

            # Run agent and stream response
            with st.chat_message("assistant"):
                with st.spinner("Analysing..."):
                    try:
                        response = run_agent(agent, final_input)
                    except Exception as e:
                        response = f"⚠️ Agent error: {e}"
                st.markdown(response)

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": response}
            )

        # ── Clear history button ──────────────
        if st.session_state.get("chat_history"):
            if st.button("🗑️ Clear chat history", key="clear_chat"):
                st.session_state["chat_history"] = []
                st.rerun()