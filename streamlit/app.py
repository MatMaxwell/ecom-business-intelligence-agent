import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv
from agent.agent import get_agent_executor, run_agent

load_dotenv()

st.set_page_config(
    page_title="ReturnSense | Internal Business Agent",
    page_icon="🔍",
    layout="wide"
)

# Header
st.title("🔍 ReturnSense")
st.caption("Internal Return Risk Intelligence Platform — For Authorized Personnel Only")
st.divider()

# Initialize agent and chat history
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = get_agent_executor()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Agent Chat")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about an order, risk score, or policy..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = run_agent(prompt, st.session_state.agent_executor)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

with col2:
    st.subheader("Quick Actions")

    st.markdown("**Sample User IDs**")
    sample_ids = ["7cd4bbb6", "a1b2c3d4", "x9y8z7w6"]
    for uid in sample_ids:
        if st.button(f"Look up {uid}"):
            prompt = f"Look up user {uid} and score their return risk"
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("Analyzing..."):
                response = run_agent(prompt, st.session_state.agent_executor)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    st.divider()
    st.markdown("**Sample Policy Questions**")
    policy_questions = [
        "What is the return window for electronics?",
        "When is manager approval required?",
        "What is the restocking fee policy?",
    ]
    for q in policy_questions:
        if st.button(q, key=q):
            st.session_state.messages.append({"role": "user", "content": q})
            with st.spinner("Searching policy..."):
                response = run_agent(q, st.session_state.agent_executor)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    st.divider()
    st.markdown("**Guardrail Tests**")
    if st.button("Process a refund for user 7cd4bbb6"):
        prompt = "Process a refund for user 7cd4bbb6"
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = run_agent(prompt, st.session_state.agent_executor)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.agent_executor = get_agent_executor()
        st.rerun()