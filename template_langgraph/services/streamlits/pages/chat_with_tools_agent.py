import os
import tempfile
from base64 import b64encode
from datetime import datetime

import streamlit as st
import whisper
from audio_recorder_streamlit import audio_recorder
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

from template_langgraph.agents.chat_with_tools_agent.agent import (
    AgentState,
    ChatWithToolsAgent,
)
from template_langgraph.speeches.tts import synthesize_audio
from template_langgraph.tools.common import get_default_tools


def image_to_base64(image_bytes: bytes) -> str:
    return b64encode(image_bytes).decode("utf-8")


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str = "base"):
    """Load a Whisper model only once per session."""

    return whisper.load_model(model_size)


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
        help="テキスト: 従来のテキスト入力/出力, 音声: マイク入力/音声出力",
    )
    st.session_state["input_output_mode"] = input_output_mode

    # 音声モードの場合、Whisper 設定を表示
    if input_output_mode == "音声":
        st.subheader("音声認識設定 (オプション)")
        audio_bytes = audio_recorder(
            text="クリックして音声入力👉️",
            recording_color="red",
            neutral_color="gray",
            icon_name="microphone",
            icon_size="2x",
            key="audio_input",
        )
        selected_model = st.sidebar.selectbox(
            "Whisperモデル",
            [
                "tiny",
                "base",
                "small",
                "medium",
                "large",
            ],
            index=1,
        )
        transcription_language = st.sidebar.selectbox(
            "文字起こし言語",
            [
                "auto",
                "ja",
                "en",
            ],
            index=0,
            help="autoは言語自動判定です",
        )
        tts_language = st.sidebar.selectbox(
            "TTS言語",
            [
                "ja",
                "en",
                "fr",
                "de",
                "ko",
                "zh-CN",
            ],
            index=0,
        )
        tts_speed = st.sidebar.slider(
            "再生速度",
            min_value=0.5,
            max_value=2.0,
            step=0.1,
            value=1.0,
        )
        tts_pitch = st.sidebar.slider(
            "ピッチ (半音)",
            min_value=-12,
            max_value=12,
            value=0,
        )
        tts_volume = st.sidebar.slider(
            "音量 (dB)",
            min_value=-20,
            max_value=10,
            value=0,
        )

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
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        # 音声データを一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_audio_file.write(audio_bytes)
            temp_audio_file_path = temp_audio_file.name
            st.download_button(
                label="🎧 録音データを保存",
                data=audio_bytes,
                file_name=f"recorded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                mime="audio/wav",
                use_container_width=True,
            )
        try:
            if input_output_mode == "音声":
                with st.spinner("音声を認識中..."):
                    model = load_whisper_model(selected_model)
                    language_param = None if transcription_language == "auto" else transcription_language
                    result = model.transcribe(str(temp_audio_file_path), language=language_param)
                    transcribed_text = result.get("text", "").strip()
                    prompt_text = transcribed_text

                    if prompt_text:
                        st.success(f"音声認識完了: {prompt_text}")
                        prompt = prompt_text
                    else:
                        st.warning("音声が認識できませんでした")
        except Exception as e:
            st.error(f"音声認識でエラーが発生しました: {e}")
            prompt_text = "音声入力でエラーが発生しました"
        finally:
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
                with st.spinner("音声を生成中です..."):
                    audio_bytes = synthesize_audio(
                        text=response_content,
                        language=tts_language,
                        speed=tts_speed,
                        pitch_shift=tts_pitch,
                        volume_db=tts_volume,
                    )
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            except Exception as e:
                st.warning(f"音声出力でエラーが発生しました: {e}")
