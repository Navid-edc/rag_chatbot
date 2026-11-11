# Streamlit ChatGPT-style Frontend connected to the FastAPI Backend & Human-in-the-Loop Router
import streamlit as st
import requests

st.set_page_config(page_title="DBA Enterprise AI Assistant", page_icon="🤖")
st.title("🤖 DBA Internal AI Assistant (RAG + SQL Agent)")
st.caption("Powered by Local Llama 3.2, ChromaDB, FlashRank Reranker & FastAPI")

# Initialize chat session history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me about DBA internal documentation or SQL database job statuses."}
    ]

# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if user_query := st.chat_input("Ask about company policies or database sync jobs..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        # Human-in-the-Loop Guardrail for high-risk financial/refund requests
        high_risk_keywords = ["refund", "transfer money", "delete database", "برگشت وجه", "حذف دیتابیس"]
        if any(kw in user_query.lower() for kw in high_risk_keywords):
            escalation_msg = (
                "⚠️ **[Human-in-the-Loop Guardrail Triggered]**\n\n"
                "This request involves a high-risk financial/system action. "
                "To prevent unauthorized execution or Prompt Injection, your ticket has been escalated to a **Human Operator** for manual approval."
            )
            st.warning(escalation_msg)
            st.session_state.messages.append({"role": "assistant", "content": escalation_msg})
        else:
            with st.spinner("Searching ChromaDB + Reranking / Querying SQL..."):
                try:
                    # Send POST request to our FastAPI microservice
                    resp = requests.post("http://127.0.0.1:8000/ask", json={"question": user_query}, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data.get("answer", "No answer returned.")
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error(f"API Error: Status {resp.status_code}")
                except Exception as e:
                    st.info("💡 Note: Start the FastAPI server (`uvicorn app:app --reload`) to get live answers.")
