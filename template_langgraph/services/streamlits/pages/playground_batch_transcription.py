import tempfile
from os import getenv

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
from langchain_community.document_loaders.parsers.audio import AzureOpenAIWhisperParser
from langchain_core.documents.base import Blob

from template_langgraph.loggers import get_logger

load_dotenv(override=True)
logger = get_logger(__name__)
logger.setLevel("DEBUG")

with st.sidebar:
    "# Common Settings"
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
    azure_openai_model_stt = st.text_input(
        label="AZURE_OPENAI_MODEL_STT",
        value=getenv("AZURE_OPENAI_MODEL_STT"),
        key="AZURE_OPENAI_MODEL_STT",
        type="default",
    )
    "### Documents"
    "[Azure OpenAI Whisper Parser](https://python.langchain.com/docs/integrations/document_loaders/parsers/azure_openai_whisper_parser/)"

st.title("🎤 Batch Transcription Playground")

audio_bytes = audio_recorder(
    text="Click to record",
    recording_color="red",
    neutral_color="gray",
    icon_name="microphone",
    icon_size="6x",
)

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_bytes)
        temp_audio_file_path = temp_audio_file.name
    st.write(f"Audio saved to temporary file: {temp_audio_file_path}")

if st.button("Transcribe", disabled=audio_bytes is None):
    with st.spinner("Transcribing..."):
        try:
            audio_blob = Blob(path=temp_audio_file_path)
            parser = AzureOpenAIWhisperParser(
                api_key=azure_openai_api_key,
                azure_endpoint=azure_openai_endpoint,
                api_version=azure_openai_api_version,
                deployment_name=azure_openai_model_stt,
            )
            documents = parser.lazy_parse(
                blob=audio_blob,
            )
            results = [doc.page_content for doc in documents]
            st.success("Transcription completed!")
            st.text_area("Transcription Result", value="\n".join(results), height=200)
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            st.error(f"Error during transcription: {e}")
