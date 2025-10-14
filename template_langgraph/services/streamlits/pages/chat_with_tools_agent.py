import os
import sqlite3
import tempfile
import uuid
from base64 import b64encode
from dataclasses import dataclass
from enum import Enum

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph_checkpoint_cosmosdb import CosmosDBSaver

from template_langgraph.agents.chat_with_tools_agent.agent import (
    AgentState,
    ChatWithToolsAgent,
)
from template_langgraph.loggers import get_logger
from template_langgraph.speeches.stt import SttWrapper
from template_langgraph.speeches.tts import TtsWrapper
from template_langgraph.tools.common import get_default_tools

logger = get_logger(__name__)
logger.setLevel("DEBUG")


class CheckpointType(str, Enum):
    SQLITE = "sqlite"
    COSMOSDB = "cosmosdb"
    MEMORY = "memory"
    NONE = "none"


DEFAULT_CHECKPOINT_TYPE = CheckpointType.NONE
CHECKPOINT_LABELS = {
    CheckpointType.COSMOSDB.value: "Cosmos DB",
    CheckpointType.SQLITE.value: "SQLite",
    CheckpointType.MEMORY.value: "メモリ",
    CheckpointType.NONE.value: "なし",
}


store_conn = sqlite3.connect("store.sqlite", check_same_thread=False)
# thread_id はセッション内で保持し、チェックポイント利用時に既存スレッドを再開できるようにする
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())


def image_to_base64(image_bytes: bytes) -> str:
    return b64encode(image_bytes).decode("utf-8")


@st.cache_resource(show_spinner=False)
def load_stt_wrapper(model_size: str = "base"):
    """Load and cache the STT model."""
    stt_wrapper = SttWrapper()
    stt_wrapper.load_model(model_size)
    return stt_wrapper


@dataclass(slots=True)
class AudioSettings:
    audio_bytes: bytes | None
    whisper_model: str
    transcription_language: str
    tts_language: str
    tts_speed: float
    tts_pitch: int
    tts_volume: int


@dataclass(slots=True)
class UserSubmission:
    content: str
    display_items: list[dict[str, object]]

    def to_history_message(self) -> dict[str, object]:
        message: dict[str, object] = {"role": "user", "content": self.content}
        if self.display_items:
            message["attachments"] = self.display_items
        return message


def ensure_session_state_defaults(tool_names: list[str]) -> None:
    st.session_state.setdefault("input_output_mode", "テキスト")
    st.session_state.setdefault("selected_tool_names", tool_names)
    st.session_state.setdefault("checkpoint_type", DEFAULT_CHECKPOINT_TYPE.value)


def get_selected_checkpoint_type() -> CheckpointType:
    raw_value = st.session_state.get("checkpoint_type", DEFAULT_CHECKPOINT_TYPE.value)
    try:
        checkpoint = CheckpointType(raw_value)
    except ValueError:
        st.session_state["checkpoint_type"] = DEFAULT_CHECKPOINT_TYPE.value
        return DEFAULT_CHECKPOINT_TYPE
    return checkpoint


def get_checkpointer():
    checkpoint_type = get_selected_checkpoint_type()
    if checkpoint_type is CheckpointType.SQLITE:
        conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        return SqliteSaver(conn=conn)
    if checkpoint_type is CheckpointType.COSMOSDB:
        from template_langgraph.tools.cosmosdb_tool import get_cosmosdb_settings

        settings = get_cosmosdb_settings()
        os.environ["COSMOSDB_ENDPOINT"] = settings.cosmosdb_host
        os.environ["COSMOSDB_KEY"] = settings.cosmosdb_key

        return CosmosDBSaver(
            database_name=settings.cosmosdb_database_name,
            container_name="checkpoints",
        )
    if checkpoint_type is CheckpointType.MEMORY:
        return InMemorySaver()
    return None


def ensure_agent_graph(selected_tools: list) -> None:
    signature = (tuple(tool.name for tool in selected_tools), get_selected_checkpoint_type().value)
    graph_signature = st.session_state.get("graph_tools_signature")
    if "graph" not in st.session_state or graph_signature != signature:
        st.session_state["graph"] = ChatWithToolsAgent(
            tools=selected_tools,
            checkpointer=get_checkpointer(),
            store=SqliteStore(
                conn=store_conn,
            ),
        ).create_graph()
        st.session_state["graph_tools_signature"] = signature


def _list_existing_thread_ids() -> list[str]:
    """チェックポインタに保存されている thread_id を列挙 (最大50件)。"""
    checkpointer = get_checkpointer()
    if not checkpointer:
        return []
    thread_ids: set[str] = set()
    try:
        for i, snapshot in enumerate(checkpointer.list(config=None)):
            if i > 1000:  # 念のため無限増加防止
                break
            cfg = getattr(snapshot, "config", {}) or {}
            configurable = cfg.get("configurable", {}) if isinstance(cfg, dict) else {}
            tid = configurable.get("thread_id")
            if isinstance(tid, str):
                thread_ids.add(tid)
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"thread list 取得失敗: {exc}")
    # 直近利用を優先できる情報が無いので単純ソート
    return sorted(thread_ids)[:50]


def build_sidebar() -> tuple[str, AudioSettings | None]:
    audio_settings: AudioSettings | None = None

    with st.sidebar:
        st.subheader("入出力モード")

        available_tools = get_default_tools()
        tool_name_to_obj = {tool.name: tool for tool in available_tools}
        tool_names = list(tool_name_to_obj.keys())

        ensure_session_state_defaults(tool_names)

        input_mode = st.radio(
            "モードを選択してください",
            options=["テキスト", "音声"],
            index=0 if st.session_state["input_output_mode"] == "テキスト" else 1,
            help="テキスト: 従来のテキスト入力/出力, 音声: マイク入力/音声出力",
        )
        st.session_state["input_output_mode"] = input_mode

        if input_mode == "音声":
            audio_settings = render_audio_controls()

        st.divider()
        st.subheader("チェックポイント")

        checkpoint_options = [checkpoint.value for checkpoint in CheckpointType]
        current_checkpoint_value = st.session_state["checkpoint_type"]
        if current_checkpoint_value not in checkpoint_options:
            current_checkpoint_value = DEFAULT_CHECKPOINT_TYPE.value
            st.session_state["checkpoint_type"] = current_checkpoint_value
        checkpoint_index = checkpoint_options.index(current_checkpoint_value)
        selected_checkpoint_value = st.selectbox(
            "保存方法",
            options=checkpoint_options,
            index=checkpoint_index,
            format_func=lambda value: CHECKPOINT_LABELS.get(value, value),
        )
        st.session_state["checkpoint_type"] = selected_checkpoint_value

        # スレッド選択 UI （チェックポイント有効時のみ）
        if get_selected_checkpoint_type() is not CheckpointType.NONE:
            existing_threads = _list_existing_thread_ids()
            st.subheader("スレッド")
            new_label = "<新規作成>"
            options = [new_label, *existing_threads]
            current_thread = st.session_state.get("thread_id")
            # 既存に一致するならその index、なければ 0 (新規)
            if current_thread in existing_threads:
                default_index = options.index(current_thread)
            else:
                default_index = 0
            selected = st.selectbox("既存スレッドを選択", options=options, index=default_index)
            if selected == new_label:
                if st.button("スレッドを生成", use_container_width=True):
                    st.session_state["thread_id"] = str(uuid.uuid4())
                    st.experimental_rerun()
            else:
                st.session_state["thread_id"] = selected
            st.caption(f"現在の thread_id: {st.session_state['thread_id']}")

        st.divider()
        st.subheader("使用するツール")

        selected_tool_names = st.multiselect(
            "有効化するツールを選択",
            options=tool_names,
            default=st.session_state["selected_tool_names"],
        )
        st.session_state["selected_tool_names"] = selected_tool_names

        selected_tools = [tool_name_to_obj[name] for name in selected_tool_names]
        ensure_agent_graph(selected_tools)

        st.caption("選択中: " + (", ".join(selected_tool_names) if selected_tool_names else "なし"))

    return input_mode, audio_settings


def render_audio_controls() -> AudioSettings:
    st.subheader("音声認識設定 (オプション)")
    audio_bytes = audio_recorder(
        text="クリックして音声入力👉️",
        recording_color="red",
        neutral_color="gray",
        icon_name="microphone",
        icon_size="2x",
        key="audio_input",
    )
    whisper_model = st.sidebar.selectbox(
        "Whisperモデル",
        ["tiny", "base", "small", "medium", "large"],
        index=1,
    )
    transcription_language = st.sidebar.selectbox(
        "文字起こし言語",
        ["auto", "ja", "en"],
        index=0,
        help="autoは言語自動判定です",
    )
    tts_language = st.sidebar.selectbox(
        "TTS言語",
        ["ja", "en", "fr", "de", "ko", "zh-CN"],
        index=0,
    )
    tts_speed = st.sidebar.slider("再生速度", min_value=0.5, max_value=2.0, step=0.1, value=1.0)
    tts_pitch = st.sidebar.slider("ピッチ (半音)", min_value=-12, max_value=12, value=0)
    tts_volume = st.sidebar.slider("音量 (dB)", min_value=-20, max_value=10, value=0)

    return AudioSettings(
        audio_bytes=audio_bytes,
        whisper_model=whisper_model,
        transcription_language=transcription_language,
        tts_language=tts_language,
        tts_speed=tts_speed,
        tts_pitch=tts_pitch,
        tts_volume=tts_volume,
    )


def render_chat_history() -> None:
    """LangGraph の state 保存されている messages を列挙して表示する。"""
    agent_messages = get_agent_messages()
    for msg in agent_messages:
        role = "assistant"
        content = ""
        attachments = []
        if isinstance(msg, dict):
            role = msg.get("role", role)
            content = msg.get("content", content)
            attachments = msg.get("attachments", []) or []
        else:  # LangChain Message オブジェクト互換
            role = getattr(msg, "role", role)
            content = getattr(msg, "content", content)
        with st.chat_message(role):
            if attachments:
                for item in attachments:
                    render_attachment(item)
            if content:
                st.write(content)


def render_attachment(item: dict[str, object]) -> None:
    item_type = item.get("type")
    if item_type == "text":
        st.markdown(item.get("text", ""))
    elif item_type == "image_url":
        url = item.get("image_url", {}).get("url")
        if url:
            st.image(url)


def collect_user_submission(mode: str, audio_settings: AudioSettings | None) -> UserSubmission | None:
    if mode == "音声":
        return collect_audio_submission(audio_settings)
    if mode == "テキスト":
        return collect_text_submission()
    st.error("不明な入出力モードです")
    return None


def collect_audio_submission(audio_settings: AudioSettings | None) -> UserSubmission | None:
    if not audio_settings or not audio_settings.audio_bytes:
        return None

    st.audio(audio_settings.audio_bytes, format="audio/wav")
    temp_audio_file_path = _write_temp_audio_file(audio_settings.audio_bytes)

    try:
        with st.spinner("音声を認識中..."):
            stt_wrapper = load_stt_wrapper(audio_settings.whisper_model)
            language_param = (
                None if audio_settings.transcription_language == "auto" else audio_settings.transcription_language
            )
            transcribed_text = stt_wrapper.transcribe(temp_audio_file_path, language=language_param)

        if not transcribed_text:
            st.warning("音声が認識できませんでした")
            return None

        st.success(f"音声認識結果: {transcribed_text}")
        return UserSubmission(
            content=transcribed_text,
            display_items=[{"type": "text", "text": transcribed_text}],
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"音声認識でエラーが発生しました: {exc}")
    finally:
        if temp_audio_file_path and os.path.exists(temp_audio_file_path):
            os.unlink(temp_audio_file_path)

    return None


def _write_temp_audio_file(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_bytes)
        return temp_audio_file.name


def collect_text_submission() -> UserSubmission | None:
    prompt = st.chat_input(
        accept_file="multiple",
        file_type=["png", "jpg", "jpeg", "gif", "webp"],
    )

    if not prompt:
        return None

    raw_text = prompt if isinstance(prompt, str) else getattr(prompt, "text", "") or ""
    prompt_files = [] if isinstance(prompt, str) else (getattr(prompt, "files", []) or [])

    display_items: list[dict[str, object]] = []
    message_parts: list[str] = []

    if raw_text.strip():
        display_items.append({"type": "text", "text": raw_text})
        message_parts.append(raw_text)

    has_unsupported_files = False
    for file in prompt_files:
        if file.type and file.type.startswith("image/"):
            image_item = build_image_attachment(file)
            if image_item:
                display_items.append(image_item)
                message_parts.append(f"![image]({image_item['image_url']['url']})")
        else:
            has_unsupported_files = True

    if has_unsupported_files:
        st.warning("画像ファイル以外の添付は現在サポートされていません。")

    content = "\n\n".join(message_parts).strip() or "ユーザーが画像をアップロードしました。"
    return UserSubmission(content=content, display_items=display_items)


def build_image_attachment(file) -> dict[str, object] | None:
    try:
        image_bytes = file.getvalue()
        base64_image = image_to_base64(image_bytes)
        image_url = f"data:{file.type};base64,{base64_image}"
        return {
            "type": "image_url",
            "image_url": {"url": image_url},
        }
    except Exception as exc:  # noqa: BLE001
        st.warning(f"画像の処理に失敗しました: {exc}")
    return None


def render_user_submission(submission: UserSubmission) -> None:
    if submission.display_items:
        for item in submission.display_items:
            render_attachment(item)
    else:
        st.write(submission.content)


def get_agent_messages() -> list:
    """LangGraph の現在 state から messages を取得。エラー時は空配列。"""
    if "graph" not in st.session_state:
        return []
    try:
        state = st.session_state["graph"].get_state(
            {
                "configurable": {
                    "thread_id": st.session_state.get("thread_id"),
                    "user_id": "user_1",
                },
            },
        )
        values = getattr(state, "values", state)
        if isinstance(values, dict):
            return list(values.get("messages", []) or [])
        return []
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"messages state の取得に失敗: {exc}")
        return []


def build_graph_messages_with_new_user(user_content: str) -> list:
    """既存 messages に新しい user メッセージを追加したリストを返す。"""
    return [*get_agent_messages(), {"role": "user", "content": user_content}]


def invoke_agent(graph_messages: list) -> AgentState:
    return st.session_state["graph"].invoke(
        {
            "messages": graph_messages,
        },
        {
            "callbacks": [
                StreamlitCallbackHandler(st.container()),
                CallbackHandler(),
            ],
            "configurable": {
                "thread_id": st.session_state.get("thread_id"),
            },
        },
    )


def synthesize_audio_if_needed(response_content: str, mode: str, audio_settings: AudioSettings | None) -> None:
    if mode != "音声" or not audio_settings:
        return

    try:
        with st.spinner("音声を生成中です..."):
            audio_bytes = TtsWrapper().synthesize_audio(
                text=response_content,
                language=audio_settings.tts_language,
                speed=audio_settings.tts_speed,
                pitch_shift=audio_settings.tts_pitch,
                volume_db=audio_settings.tts_volume,
            )
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"音声出力でエラーが発生しました: {exc}")


input_output_mode, audio_settings = build_sidebar()

render_chat_history()

submission = collect_user_submission(input_output_mode, audio_settings)

if submission:
    with st.chat_message("user"):
        render_user_submission(submission)

    updated_messages = build_graph_messages_with_new_user(submission.content)
    with st.chat_message("assistant"):
        response = invoke_agent(updated_messages)
        latest_messages = response["messages"]
        last_message = latest_messages[-1] if latest_messages else None
        if last_message is not None:
            response_content = getattr(last_message, "content", None) or (
                last_message.get("content") if isinstance(last_message, dict) else ""
            )
            st.write(response_content)
            synthesize_audio_if_needed(response_content, input_output_mode, audio_settings)
