from base64 import b64encode
from os import getenv

import streamlit as st
from dotenv import load_dotenv
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError

from template_langgraph.loggers import get_logger

load_dotenv(override=True)
logger = get_logger(__name__)
logger.setLevel("DEBUG")


def image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return b64encode(image_bytes).decode("utf-8")


with st.sidebar:
    "# Common Settings"
    stream_mode = st.checkbox(
        label="ストリーム出力を有効にする",
        value=True,
        key="STREAM_MODE",
    )
    "# Model"
    model_choice = st.radio(
        label="Active Model",
        options=["azure", "ollama"],
        index=0,
        key="model_choice",
    )
    "## Model Settings"
    if model_choice == "azure":
        azure_openai_endpoint = st.text_input(
            label="AZURE_OPENAI_ENDPOINT",
            value=getenv("AZURE_OPENAI_ENDPOINT"),
            key="AZURE_OPENAI_ENDPOINT",
            type="default",
        )
        azure_openai_api_key = st.text_input(
            label="AZURE_OPENAI_API_KEY",
            value=getenv("AZURE_OPENAI_API_KEY"),
            key="AZURE_OPENAI_API_KEY",
            type="password",
        )
        azure_openai_api_version = st.text_input(
            label="AZURE_OPENAI_API_VERSION",
            value=getenv("AZURE_OPENAI_API_VERSION"),
            key="AZURE_OPENAI_API_VERSION",
            type="default",
        )
        azure_openai_model_chat = st.text_input(
            label="AZURE_OPENAI_MODEL_CHAT",
            value=getenv("AZURE_OPENAI_MODEL_CHAT"),
            key="AZURE_OPENAI_MODEL_CHAT",
            type="default",
        )
        "[Azure Portal](https://portal.azure.com/)"
        "[Azure OpenAI Studio](https://oai.azure.com/resource/overview)"
        "[View the source code](https://github.com/ks6088ts-labs/template-streamlit)"
    else:
        ollama_model_chat = st.text_input(
            label="OLLAMA_MODEL_CHAT",
            value=getenv("OLLAMA_MODEL_CHAT"),
            key="OLLAMA_MODEL_CHAT",
            type="default",
        )
        "[Ollama Docs](https://github.com/ollama/ollama)"
        "[View the source code](https://github.com/ks6088ts-labs/template-streamlit)"


def is_azure_configured():
    return (
        st.session_state.get("AZURE_OPENAI_API_KEY")
        and st.session_state.get("AZURE_OPENAI_ENDPOINT")
        and st.session_state.get("AZURE_OPENAI_API_VERSION")
        and st.session_state.get("AZURE_OPENAI_MODEL_CHAT")
        and st.session_state.get("model_choice") == "azure"
    )


def is_ollama_configured():
    return st.session_state.get("OLLAMA_MODEL_CHAT") and st.session_state.get("model_choice") == "ollama"


def is_configured():
    return is_azure_configured() or is_ollama_configured()


def get_model():
    if is_azure_configured():
        return AzureChatOpenAI(
            azure_endpoint=st.session_state.get("AZURE_OPENAI_ENDPOINT"),
            api_key=st.session_state.get("AZURE_OPENAI_API_KEY"),
            openai_api_version=st.session_state.get("AZURE_OPENAI_API_VERSION"),
            azure_deployment=st.session_state.get("AZURE_OPENAI_MODEL_CHAT"),
        )
    elif is_ollama_configured():
        return ChatOllama(
            model=st.session_state.get("OLLAMA_MODEL_CHAT", ""),
        )
    raise ValueError("No model is configured. Please set up the Azure or Ollama model in the sidebar.")


st.title("chat app with LangChain SDK")

if not is_configured():
    st.warning("Please fill in the required fields at the sidebar.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        AIMessage(content="Hello! I'm a helpful assistant."),
    ]

# Show chat messages
for message in st.session_state.messages:
    with st.chat_message(message.type):
        if isinstance(message.content, str):
            st.markdown(message.content)
        else:
            for item in message.content:
                if item["type"] == "text":
                    st.markdown(item["text"])
                elif item["type"] == "image_url":
                    st.image(item["image_url"]["url"])


# Receive user input
uploaded_file = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg"], key="file_uploader")
if prompt := st.chat_input(disabled=not is_configured()):
    user_message_content = [{"type": "text", "text": prompt}]
    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        base64_image = image_to_base64(image_bytes)
        image_url = f"data:image/jpeg;base64,{base64_image}"
        user_message_content.append({"type": "image_url", "image_url": {"url": image_url}})

    user_message = HumanMessage(content=user_message_content)
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        for item in user_message_content:
            if item["type"] == "text":
                st.markdown(item["text"])
            elif item["type"] == "image_url":
                st.image(item["image_url"]["url"])

    with st.spinner("Thinking..."):
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            llm = get_model()
            callbacks = [StreamlitCallbackHandler(st.container())]

            try:
                if stream_mode:
                    for chunk in llm.stream(st.session_state.messages):
                        if chunk.content is not None:
                            full_response += chunk.content
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                else:
                    response = llm.invoke(input=st.session_state.messages)
                    full_response = response.content if hasattr(response, "content") else str(response)
                    message_placeholder.markdown(full_response)

                st.session_state.messages.append(AIMessage(content=full_response))

            except APITimeoutError as e:
                logger.exception(f"APIタイムアウトエラーが発生しました: {e}")
                st.error(f"APIタイムアウトエラーが発生しました: {e}")
                st.warning("再度お試しいただくか、接続を確認してください。")
            except APIConnectionError as e:
                logger.exception(f"API接続エラーが発生しました: {e}")
                st.error(f"API接続エラーが発生しました: {e}")
                st.warning("ネットワーク接続を確認してください。")
            except APIStatusError as e:
                logger.exception(f"APIステータスエラーが発生しました: {e.status_code} - {e.response}")
                st.error(f"APIステータスエラーが発生しました: {e.status_code} - {e.response}")
                st.warning("Azure OpenAIの設定（デプロイメント名、APIバージョンなど）を確認してください。")
            except Exception as e:
                logger.exception(f"予期せぬエラーが発生しました: {e}")
                st.error(f"予期せぬエラーが発生しました: {e}")
