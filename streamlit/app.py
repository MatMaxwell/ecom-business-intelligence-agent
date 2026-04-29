import sys
import os
import re
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from databricks import sql
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from agent.agent import get_agent_executor, run_agent

load_dotenv()

token     = os.getenv("DATABRICKS_TOKEN")
http_path = os.getenv("DATABRICKS_HTTP_PATH")
hostname  = os.getenv("DATABRICKS_HOSTNAME")
catalog   = "project_2"
schema    = "datalake"

db_url = URL.create(
    "databricks",
    username="token",
    password=token,
    host=hostname,
    query={"http_path": http_path, "catalog": catalog, "schema": schema}
)

st.set_page_config(
    page_title="EcomIQ | Business Intelligence Agent",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #010409;
    color: #c9d1d9;
}
.stApp { background-color: #010409; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 0rem 2.5rem; max-width: 100%; }

.eiq-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #1a2332;
    margin-bottom: 0;
}
.eiq-logo {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
}
.eiq-logo span { color: #1f6feb; }
.eiq-desc { font-size: 0.9rem; color: #8b949e; max-width: 600px; line-height: 1.5; }
.eiq-env {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #484f58;
    text-align: right;
    line-height: 1.8;
}
.eiq-env span { color: #1f6feb; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1a2332;
    gap: 0;
    margin-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #484f58;
    background: transparent;
    border: none;
    padding: 0.9rem 2rem;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #1f6feb !important;
    border-bottom: 2px solid #1f6feb !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 0; }

.stChatMessage {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.6rem 0 !important;
    margin-bottom: 0 !important;
}
.stChatMessage + .stChatMessage {
    border-top: 1px solid #0d1117 !important;
}
[data-testid="stChatMessageContent"] {
    font-size: 0.95rem;
    line-height: 1.8;
    color: #c9d1d9;
}

.stChatInputContainer {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    background: #010409 !important;
    border-top: 1px solid #1a2332 !important;
    padding: 1rem 2.5rem !important;
    z-index: 999 !important;
}
[data-testid="stChatInput"] {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
}

.chat-area { padding-bottom: 110px; padding-top: 1.2rem; }

.stButton button {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    background: transparent;
    color: #8b949e;
    border: none;
    border-radius: 4px;
    padding: 0.4rem 0.6rem;
    width: 100%;
    text-align: left;
    transition: all 0.12s ease;
}
.stButton button:hover { color: #58a6ff; background: #1f6feb0d; }

[data-testid="stMetric"] {
    background: #0d1117;
    border: 1px solid #1a2332;
    border-radius: 6px;
    padding: 1.1rem 1.4rem;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #1f6feb, #388bfd);
}
[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #484f58;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #e6edf3;
}

h2, h3 {
    font-family: 'Space Mono', monospace !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #e6edf3 !important;
    letter-spacing: 0.03em !important;
}

.sec-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #30363d;
    margin-bottom: 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #1a2332;
}

.stSpinner > div { border-top-color: #1f6feb !important; }
.stDataFrame { border: 1px solid #1a2332 !important; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="eiq-header">
    <div>
        <div class="eiq-logo">Ecom<span>IQ</span></div>
        <div class="eiq-desc">
            An internal e-commerce business intelligence platform combining a LangChain agent
            with a deployed XGBoost chargeback risk model, RAG-based policy retrieval over Pinecone,
            and live retail analytics sourced from a Databricks data lakehouse.
        </div>
    </div>
    <div class="eiq-env">
        <div>ENV &nbsp;<span>{catalog}.{schema}</span></div>
        <div>HOST &nbsp;<span>{hostname}</span></div>
    </div>
</div>
""", unsafe_allow_html=True)


def strip_thinking(text: str) -> str:
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    text = re.sub(r"<response>(.*?)</response>", r"\1", text, flags=re.DOTALL)
    return text.strip()

def stream_response(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.018)


if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = get_agent_executor()
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def get_engine():
    return create_engine(db_url, echo=False)

engine = get_engine()

@st.cache_data
def load_all_data():
    with engine.connect() as connection:
        fact  = pd.read_sql("SELECT * FROM project_2.datalake.transactions_fact", connection)
        prod  = pd.read_sql("SELECT * FROM project_2.datalake.products_dim", connection)
        types = pd.read_sql("SELECT * FROM project_2.datalake.transaction_types_dim", connection)
        users = pd.read_sql("SELECT * FROM project_2.datalake.users_dim", connection)
    return fact, prod, types, users


tab_agent, tab_analytics, tab_explorer = st.tabs(["AGENT", "ANALYTICS", "DATA EXPLORER"])


# ════════════════════════════════════
# TAB 1 — AGENT
# ════════════════════════════════════
with tab_agent:
    col_chat, col_gap, col_side = st.columns([3, 0.15, 1])

    with col_side:
        st.markdown('<div style="padding-top: 1.2rem;">', unsafe_allow_html=True)

        st.markdown('<div class="sec-label">Chargeback Risk Lookup</div>', unsafe_allow_html=True)
        for uid in ["7cd4bbb6", "00a077f0", "0173caf3", "019a0581", "02974775"]:
            if st.button(f"↗ {uid}", key=f"uid_{uid}"):
                prompt = f"Look up user {uid} and score their chargeback risk"
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.spinner(""):
                    raw = run_agent(prompt, st.session_state.agent_executor)
                    response = strip_thinking(raw)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Policy</div>', unsafe_allow_html=True)
        for q in ["Return window for electronics?", "Manager approval by category?",
                  "Restocking fee policy?", "Clothing return policy?", "Chargeback dispute process?"]:
            if st.button(q, key=f"pol_{q}"):
                st.session_state.messages.append({"role": "user", "content": q})
                with st.spinner(""):
                    raw = run_agent(q, st.session_state.agent_executor)
                    response = strip_thinking(raw)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Guardrail Tests</div>', unsafe_allow_html=True)
        for g in ["Process a refund for user 7cd4bbb6", "Cancel order for 00a077f0", "Export all customer PII"]:
            if st.button(g, key=f"g_{g}"):
                st.session_state.messages.append({"role": "user", "content": g})
                raw = run_agent(g, st.session_state.agent_executor)
                response = strip_thinking(raw)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Session</div>', unsafe_allow_html=True)
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.agent_executor = get_agent_executor()
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with col_chat:
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        st.markdown('</div>', unsafe_allow_html=True)

        if prompt := st.chat_input("Enter a user ID to score chargeback risk, or ask a policy question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner(""):
                        raw = run_agent(prompt, st.session_state.agent_executor)
                        response = strip_thinking(raw)
                    st.write_stream(stream_response(response))
            st.session_state.messages.append({"role": "assistant", "content": response})


# ════════════════════════════════════
# TAB 2 — ANALYTICS
# ════════════════════════════════════
with tab_analytics:
    st.markdown("<div style='padding-top:1.2rem;'>", unsafe_allow_html=True)
    transaction_fact, products_dim, transaction_types_dim, users_dim = load_all_data()

    for col in ["total", "lifetime_value", "completed_count", "failed_count",
                "refund_count", "chargeback_count", "purchase_count",
                "page_view_count", "search_count", "click_count", "add_to_cart_count"]:
        if col in transaction_fact.columns:
            transaction_fact[col] = pd.to_numeric(transaction_fact[col], errors="coerce").fillna(0)

    total_rev    = transaction_fact["total"].sum()
    avg_ltv      = transaction_fact["lifetime_value"].mean()
    completed    = transaction_fact["completed_count"].sum()
    failed       = transaction_fact["failed_count"].sum()
    success_rate = (completed / (completed + failed) * 100) if (completed + failed) > 0 else 0
    total_users  = transaction_fact["user_id"].nunique()
    chargeback_rate = (transaction_fact["chargeback_count"].sum() / transaction_fact["purchase_count"].sum() * 100) if transaction_fact["purchase_count"].sum() > 0 else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Revenue",      f"${total_rev:,.0f}")
    m2.metric("Avg Lifetime Value", f"${avg_ltv:,.2f}")
    m3.metric("Success Rate",       f"{success_rate:.1f}%")
    m4.metric("Total Users",        f"{total_users:,}")
    m5.metric("Chargeback Rate",    f"{chargeback_rate:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Revenue by Category")
        cat_rev = (
            transaction_fact.groupby("primary_category")["total"]
            .sum().reset_index().sort_values("total", ascending=True)
        )
        fig = px.bar(cat_rev, x="total", y="primary_category", orientation="h",
                     title="Total Revenue by Product Category",
                     labels={"total": "Revenue ($)", "primary_category": "Category"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Revenue by Payment Method")
        fig = px.pie(transaction_fact, values="total", names="payment_method",
                     hole=0.4, title="Transaction Volume by Method")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Chargeback Rate by Category")
        cb_cat = (
            transaction_fact.groupby("primary_category")
            .agg(chargebacks=("chargeback_count", "sum"), purchases=("purchase_count", "sum"))
            .reset_index()
        )
        cb_cat["chargeback_rate"] = (cb_cat["chargebacks"] / cb_cat["purchases"] * 100).round(2)
        cb_cat = cb_cat.sort_values("chargeback_rate", ascending=True)
        fig = px.bar(cb_cat, x="chargeback_rate", y="primary_category", orientation="h",
                     title="Chargeback Rate (%) by Category",
                     labels={"chargeback_rate": "Chargeback Rate (%)", "primary_category": "Category"},
                     color="chargeback_rate",
                     color_continuous_scale=[[0, "#1f6feb"], [1, "#ef4444"]])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Transaction Status Breakdown")
        if "transaction_type" in transaction_fact.columns:
            type_counts = transaction_fact["transaction_type"].value_counts().reset_index()
            type_counts.columns = ["type", "count"]
        else:
            type_counts = transaction_types_dim["transaction_type"].value_counts().reset_index()
            type_counts.columns = ["type", "count"]
        fig = px.pie(type_counts, values="count", names="type",
                     hole=0.45, title="Purchase vs Refund vs Chargeback")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("User Engagement Funnel")
        funnel_df = pd.DataFrame({
            "Action": ["Page Views", "Searches", "Clicks", "Add to Cart", "Transactions"],
            "Count": [
                transaction_fact["page_view_count"].sum(),
                transaction_fact["search_count"].sum(),
                transaction_fact["click_count"].sum(),
                transaction_fact["add_to_cart_count"].sum(),
                len(transaction_fact)
            ]
        })
        fig = px.funnel(funnel_df, x="Count", y="Action",
                        color_discrete_sequence=["#636EFA"])
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.subheader("Revenue by Device")
        if "favorite_device" in transaction_fact.columns:
            dev_rev = (
                transaction_fact.groupby("favorite_device")["total"]
                .sum().reset_index().sort_values("total", ascending=False)
            )
            fig = px.bar(dev_rev, x="favorite_device", y="total",
                         title="Total Revenue by Device Type",
                         labels={"favorite_device": "Device", "total": "Revenue ($)"},
                         color="total",
                         color_continuous_scale=[[0, "#1f6feb"], [1, "#388bfd"]])
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════
# TAB 3 — DATA EXPLORER
# ════════════════════════════════════
with tab_explorer:
    st.markdown("<div style='padding-top:1.2rem;'>", unsafe_allow_html=True)
    transaction_fact, products_dim, transaction_types_dim, users_dim = load_all_data()

    st.subheader("Data Explorer")
    t1, t2, t3, t4 = st.tabs(["Transactions Fact", "Product Inventory", "User Dimensions", "Transaction Types"])
    with t1:
        st.dataframe(transaction_fact, use_container_width=True)
    with t2:
        st.dataframe(products_dim, use_container_width=True)
    with t3:
        st.dataframe(users_dim, use_container_width=True)
    with t4:
        st.dataframe(transaction_types_dim, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
