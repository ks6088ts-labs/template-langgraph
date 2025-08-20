import streamlit as st
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langgraph.graph.state import CompiledStateGraph

from template_langgraph.agents.demo_agents.parallel_rag_agent.agent import ParallelRagAgent
from template_langgraph.agents.demo_agents.weather_agent import graph as weather_agent_graph
from template_langgraph.llms.azure_openais import AzureOpenAiWrapper
from template_langgraph.tools.common import get_default_tools


def _make_parallel_rag_graph(tools):
    def build_input(prompt):
        return {
            "query": prompt,
        }

    return {
        "graph": ParallelRagAgent(
            llm=AzureOpenAiWrapper().chat_model,
            tools=tools,
        ).create_graph(),
        "build_input": build_input,
    }


def _make_weather_graph(_tools=None):
    def build_input(prompt):
        return {
            "messages": [
                prompt,
            ],
        }

    return {
        "graph": weather_agent_graph,
        "build_input": build_input,
    }


agent_options = {
    "Parallel RAG Agent": {
        "supports_tools": True,
        "factory": _make_parallel_rag_graph,
    },
    "Weather Agent": {
        "supports_tools": False,
        "factory": _make_weather_graph,
    },
    # "Another Agent": {"supports_tools": True/False, "factory": your_factory},
}


def create_graph() -> CompiledStateGraph:
    cfg = agent_options.get(selected_agent_key) or next(iter(agent_options.values()))
    supports_tools = cfg.get("supports_tools", True)
    factory = cfg["factory"]
    result = factory(selected_tools if supports_tools else None)
    st.session_state["input_builder"] = result.get("build_input") or (lambda p: {"query": p})
    return result["graph"]


# Sidebar: ツール選択とエージェントの構築
with st.sidebar:
    # 追加: エージェント選択 UI
    st.subheader("使用するエージェント")
    available_agent_keys = list(agent_options.keys())
    if "selected_agent_key" not in st.session_state:
        st.session_state["selected_agent_key"] = available_agent_keys[0]
    selected_agent_key = st.selectbox(
        "実行するエージェントを選択",
        options=available_agent_keys,
        index=available_agent_keys.index(st.session_state["selected_agent_key"]),
    )
    st.session_state["selected_agent_key"] = selected_agent_key

    # エージェントの tool call 対応フラグを取得
    supports_tools = agent_options[selected_agent_key].get("supports_tools", True)

    # ツール選択 UI（supports_tools が True の時のみ表示）
    if supports_tools:
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
        selected_tools = [tool_name_to_obj[name] for name in selected_tool_names]
        signature = (selected_agent_key, tuple(selected_tool_names))
    else:
        # 非対応時はツール選択をスキップ
        selected_tool_names = []
        selected_tools = []
        signature = (selected_agent_key,)

    # 選択に応じてグラフを再構築
    if "graph" not in st.session_state or st.session_state.get("graph_signature") != signature:
        st.session_state["graph"] = create_graph()
        st.session_state["graph_signature"] = signature

    # 選択中の表示
    st.caption(f"選択中のエージェント: {selected_agent_key}")
    if supports_tools:
        st.caption("選択中のツール: " + (", ".join(selected_tool_names) if selected_tool_names else "なし"))
    else:
        st.caption("このエージェントはツール呼び出しをサポートしていません")

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("処理中..."):
            # 変更: callbacks は config に渡す。input は入力のみ。
            callbacks = [StreamlitCallbackHandler(st.container())]
            input_builder = st.session_state.get("input_builder") or (lambda p: {"query": p})
            response = st.session_state["graph"].invoke(
                input=input_builder(prompt),
                config={"callbacks": callbacks},
            )
        st.write(response)
