from base64 import b64encode
import tempfile

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS
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

# Sidebar: 入出力モード選択、ツール選択とエージェントの構築
with st.sidebar:
    st.subheader("入出力モード")
    
    # 入出力モード選択
    if "input_output_mode" not in st.session_state:
        st.session_state["input_output_mode"] = "テキスト"
    
    input_output_mode = st.radio(
        "モードを選択してください",
        options=["テキスト", "音声"],
        index=0 if st.session_state["input_output_mode"] == "テキスト" else 1,
        help="テキスト: 従来のテキスト入力/出力, 音声: マイク入力/音声出力"
    )
    st.session_state["input_output_mode"] = input_output_mode
    
    st.divider()
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

# 入力セクション: モードに応じて分岐
prompt = None
prompt_text = ""
prompt_files = []

if input_output_mode == "音声":
    st.subheader("🎤 音声入力")
    audio_bytes = audio_recorder(
        text="クリックして録音",
        recording_color="red",
        neutral_color="black",
        icon_name="microphone",
        icon_size="2x",
        key="audio_input"
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        # 音声データを一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_audio_file.write(audio_bytes)
            temp_audio_file_path = temp_audio_file.name
        
        # TODO: 音声からテキストへの変換実装
        # 現在は音声入力をプレースホルダーテキストに変換
        prompt_text = "音声入力を受信しました（音声認識は後で実装予定）"
        prompt = prompt_text
        
        # 一時ファイルを削除
        import os
        os.unlink(temp_audio_file_path)
        
else:
    # 既存のテキスト入力モード
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
        pass  # promptは既に設定済み

# 共通の入力処理ロジック
if prompt:
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
        
        # レスポンス表示とオーディオ出力
        response_content = last_message.content
        st.write(response_content)
        
        # 音声モードの場合、音声出力を追加
        if input_output_mode == "音声":
            try:
                # gTTSを使って音声生成
                tts = gTTS(text=response_content, lang='ja')
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
                    tts.save(temp_audio_file.name)
                    
                    # 音声ファイルを読み込んでstreamlit audio widgetで再生
                    with open(temp_audio_file.name, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                    
                    # 一時ファイルを削除
                    import os
                    os.unlink(temp_audio_file.name)
                    
            except Exception as e:
                st.warning(f"音声出力でエラーが発生しました: {e}")
