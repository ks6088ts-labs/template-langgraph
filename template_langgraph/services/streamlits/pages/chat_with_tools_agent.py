from base64 import b64encode

import streamlit as st
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

from template_langgraph.agents.chat_with_tools_agent.agent import (
    AgentState,
    ChatWithToolsAgent,
)
from template_langgraph.tools.common import get_default_tools


def image_to_base64(image_bytes: bytes) -> str:
    return b64encode(image_bytes).decode("utf-8")


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
        attachments = msg.get("attachments", [])
        with st.chat_message(msg["role"]):
            if attachments:
                for item in attachments:
                    if item["type"] == "text":
                        st.markdown(item["text"])
                    elif item["type"] == "image_url":
                        st.image(item["image_url"]["url"])
            else:
                st.write(msg["content"])
    else:
        st.chat_message("assistant").write(msg.content)

if prompt := st.chat_input(
    accept_file="multiple",
    file_type=[
        "png",
        "jpg",
        "jpeg",
        "gif",
        "webp",
    ],
):
    user_display_items = []
    message_parts = []

    prompt_text = prompt if isinstance(prompt, str) else getattr(prompt, "text", "") or ""
    prompt_files = [] if isinstance(prompt, str) else (getattr(prompt, "files", []) or [])

    user_text = prompt_text
    if user_text.strip():
        user_display_items.append({"type": "text", "text": user_text})
        message_parts.append(user_text)

    has_unsupported_files = False
    for file in prompt_files:
        if file.type and file.type.startswith("image/"):
            image_bytes = file.getvalue()
            base64_image = image_to_base64(image_bytes)
            image_url = f"data:{file.type};base64,{base64_image}"
            user_display_items.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            )
            message_parts.append(f"![image]({image_url})")
        else:
            has_unsupported_files = True

    if has_unsupported_files:
        st.warning("画像ファイル以外の添付は現在サポートされていません。")

    message_content = "\n\n".join(message_parts).strip()
    if not message_content:
        message_content = "ユーザーが画像をアップロードしました。"

    new_user_message = {"role": "user", "content": message_content}
    if user_display_items:
        new_user_message["attachments"] = user_display_items

    st.session_state["chat_history"].append(new_user_message)

    with st.chat_message("user"):
        if user_display_items:
            for item in user_display_items:
                if item["type"] == "text":
                    st.markdown(item["text"])
                elif item["type"] == "image_url":
                    st.image(item["image_url"]["url"])
        else:
            st.write(message_content)

    graph_messages = []
    for msg in st.session_state["chat_history"]:
        if isinstance(msg, dict):
            graph_messages.append({"role": msg["role"], "content": msg["content"]})
        else:
            graph_messages.append(msg)

    with st.chat_message("assistant"):
        response: AgentState = st.session_state["graph"].invoke(
            {"messages": graph_messages},
            {
                "callbacks": [
                    StreamlitCallbackHandler(st.container()),
                ]
            },
        )
        last_message = response["messages"][-1]
        st.session_state["chat_history"].append(last_message)
        st.write(last_message.content)
