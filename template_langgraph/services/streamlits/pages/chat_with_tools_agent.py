import streamlit as st
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

from template_langgraph.agents.chat_with_tools_agent.agent import AgentState, graph

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        response: AgentState = graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
            },
            {
                "callbacks": [
                    StreamlitCallbackHandler(st.container()),
                ]
            },
        )
        st.write(response["messages"][-1].content)
