import streamlit as st
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

from template_langgraph.agents.chat_with_tools_agent.agent import (
    AgentState,
    ChatWithToolsAgent,
)
from template_langgraph.tools.common import get_default_tools

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Sidebar: ツール選択とエージェントの構築
with st.sidebar:
    st.subheader("使用するツール")

    # 利用可能なツール一覧を取得
    available_tools = get_default_tools()
    tool_name_to_obj = {t.name: t for t in available_tools}
    tool_names = list(tool_name_to_obj.keys())

    # 初期選択は全選択
    if "selected_tool_names" not in st.session_state:
        st.session_state["selected_tool_names"] = tool_names

    selected_tool_names = st.multiselect(
        "有効化するツールを選択",
        options=tool_names,
        default=st.session_state["selected_tool_names"],
    )
    st.session_state["selected_tool_names"] = selected_tool_names

    # 選択されたツールでグラフを再構築（選択が変わった時のみ）
    selected_tools = [tool_name_to_obj[name] for name in selected_tool_names]
    signature = tuple(selected_tool_names)
    if "graph" not in st.session_state or st.session_state.get("graph_tools_signature") != signature:
        st.session_state["graph"] = ChatWithToolsAgent(tools=selected_tools).create_graph()
        st.session_state["graph_tools_signature"] = signature
    # 選択中のツール表示（簡易）
    st.caption("選択中: " + (", ".join(selected_tool_names) if selected_tool_names else "なし"))

for msg in st.session_state["chat_history"]:
    if isinstance(msg, dict):
        st.chat_message(msg["role"]).write(msg["content"])
    else:
        st.chat_message("assistant").write(msg.content)

if prompt := st.chat_input():
    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        response: AgentState = st.session_state["graph"].invoke(
            {"messages": st.session_state["chat_history"]},
            {
                "callbacks": [
                    StreamlitCallbackHandler(st.container()),
                ]
            },
        )
        last_message = response["messages"][-1]
        st.session_state["chat_history"].append(last_message)
        st.write(last_message.content)
