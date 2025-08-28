import streamlit as st
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

from template_langgraph.agents.demo_agents.parallel_rag_agent.agent import ParallelRagAgent
from template_langgraph.agents.demo_agents.parallel_rag_agent.models import ParallelRagAgentState
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.tools.common import get_default_tools

st.title("Parallel RAG Agent - Interactive Chat")
st.caption("Multi-turn conversation with parallel tool execution for comprehensive information retrieval")

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
        st.session_state["graph"] = ParallelRagAgent(
            llm=AzureOpenAiWrapper().chat_model,
            tools=selected_tools,
        ).create_graph()
        st.session_state["graph_tools_signature"] = signature

    # 選択中のツール表示（簡易）
    st.caption("選択中: " + (", ".join(selected_tool_names) if selected_tool_names else "なし"))

    # チャット履歴をクリア
    if st.button("チャット履歴をクリア"):
        st.session_state["chat_history"] = []
        st.rerun()

# チャット履歴を表示
for msg in st.session_state["chat_history"]:
    if isinstance(msg, dict):
        st.chat_message(msg["role"]).write(msg["content"])
    else:
        # Handle BaseMessage objects
        if hasattr(msg, 'content'):
            role = "user" if "HumanMessage" in str(type(msg)) else "assistant"
            st.chat_message(role).write(msg.content)

if prompt := st.chat_input():
    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("情報を並列で取得・分析中..."):
            # ParallelRagAgentは完全なメッセージ履歴を使用
            response: ParallelRagAgentState = st.session_state["graph"].invoke(
                {"messages": st.session_state["chat_history"]},
                {
                    "callbacks": [
                        StreamlitCallbackHandler(st.container()),
                    ]
                },
            )
            
            # 最後のメッセージ（AIの応答）を取得
            if response.get("messages"):
                last_message = response["messages"][-1]
                st.session_state["chat_history"].append(last_message)
                st.write(last_message.content)
            else:
                # フォールバック: summaryを使用
                summary = response.get("summary", "応答を生成できませんでした。")
                st.write(summary)
                st.session_state["chat_history"].append({"role": "assistant", "content": summary})

# サイドバーに実行結果の詳細情報を表示
with st.sidebar:
    st.subheader("詳細情報")
    if st.session_state.get("chat_history"):
        st.caption(f"メッセージ数: {len(st.session_state['chat_history'])}")
    
    # 最後の実行結果を表示
    if "last_response" in locals() and "response" in locals():
        with st.expander("最後の実行結果"):
            st.json({
                "task_results_count": len(response.get("task_results", [])),
                "summary_length": len(response.get("summary", "")),
                "messages_count": len(response.get("messages", []))
            })