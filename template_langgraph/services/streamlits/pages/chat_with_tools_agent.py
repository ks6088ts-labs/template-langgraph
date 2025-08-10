import streamlit as st
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

from template_langgraph.agents.chat_with_tools_agent.agent import AgentState, graph

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for msg in st.session_state["chat_history"]:
    if isinstance(msg, dict):
        st.chat_message(msg["role"]).write(msg["content"])
    else:
        st.chat_message("assistant").write(msg.content)

if prompt := st.chat_input():
    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        response: AgentState = graph.invoke(
            {"messages": st.session_state["chat_history"]},
            {
                "callbacks": [
                    StreamlitCallbackHandler(st.container()),
                ]
            },
        )
        st.session_state["chat_history"].append(response["messages"][-1])
        st.write(response["messages"][-1].content)
