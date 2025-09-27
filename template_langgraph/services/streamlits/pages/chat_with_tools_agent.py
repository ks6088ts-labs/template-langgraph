from base64 import b64encode
import tempfile
from os import getenv

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain_community.document_loaders.parsers.audio import AzureOpenAIWhisperParser
from langchain_core.documents.base import Blob

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
    
    # 音声モードの場合、Azure OpenAI設定を表示
    if input_output_mode == "音声":
        st.subheader("音声認識設定 (オプション)")
        with st.expander("Azure OpenAI Whisper設定", expanded=False):
            azure_openai_endpoint = st.text_input(
                "AZURE_OPENAI_ENDPOINT",
                value=getenv("AZURE_OPENAI_ENDPOINT", ""),
                help="Azure OpenAI リソースのエンドポイント"
            )
            azure_openai_api_key = st.text_input(
                "AZURE_OPENAI_API_KEY",
                value=getenv("AZURE_OPENAI_API_KEY", ""),
                type="password",
                help="Azure OpenAI リソースのAPIキー"
            )
            azure_openai_api_version = st.text_input(
                "AZURE_OPENAI_API_VERSION", 
                value=getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                help="Azure OpenAI APIバージョン"
            )
            azure_openai_model_stt = st.text_input(
                "AZURE_OPENAI_MODEL_STT",
                value=getenv("AZURE_OPENAI_MODEL_STT", "whisper"),
                help="音声認識用のデプロイ名"
            )
            st.caption("※設定しない場合は、音声入力時にプレースホルダーテキストが使用されます")
    
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
        
        # Azure OpenAI Whisperが設定されている場合は音声認識を実施
        try:
            if (input_output_mode == "音声" and 
                azure_openai_endpoint and azure_openai_api_key and 
                azure_openai_model_stt):
                
                with st.spinner("音声を認識中..."):
                    audio_blob = Blob(path=temp_audio_file_path)
                    parser = AzureOpenAIWhisperParser(
                        api_key=azure_openai_api_key,
                        azure_endpoint=azure_openai_endpoint,
                        api_version=azure_openai_api_version,
                        deployment_name=azure_openai_model_stt,
                    )
                    documents = parser.lazy_parse(blob=audio_blob)
                    results = [doc.page_content for doc in documents]
                    prompt_text = "\n".join(results).strip()
                    
                    if prompt_text:
                        st.success(f"音声認識完了: {prompt_text}")
                        prompt = prompt_text
                    else:
                        st.warning("音声が認識できませんでした")
                        prompt = None
            else:
                # Azure OpenAI設定がない場合はプレースホルダー
                prompt_text = "音声入力を受信しました（音声認識設定が必要です）"
                prompt = prompt_text
                st.info("音声認識を使用するには、サイドバーでAzure OpenAI設定を入力してください")
                
        except Exception as e:
            st.error(f"音声認識でエラーが発生しました: {e}")
            prompt_text = "音声入力でエラーが発生しました"
            prompt = prompt_text
        finally:
            # 一時ファイルを削除
            import os
            if os.path.exists(temp_audio_file_path):
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
