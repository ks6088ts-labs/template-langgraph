from functools import lru_cache

from azure.identity import DefaultAzureCredential
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic_settings import BaseSettings, SettingsConfigDict

from template_langgraph.loggers import get_logger

logger = get_logger(__name__)


class Settings(BaseSettings):
    azure_openai_use_microsoft_entra_id: str = "False"
    azure_openai_endpoint: str = "https://<YOUR_AOAI_NAME>.openai.azure.com/"
    azure_openai_api_key: str = "<YOUR_API_KEY>"
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_model_chat: str = "gpt-4o"
    azure_openai_model_embedding: str = "text-embedding-3-small"
    azure_openai_model_reasoning: str = "o4-mini"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_azure_openai_settings() -> Settings:
    return Settings()


class AzureOpenAiWrapper:
    def __init__(self, settings: Settings = None):
        if settings is None:
            settings = get_azure_openai_settings()

        if settings.azure_openai_use_microsoft_entra_id.lower() == "true":
            logger.info("Using Microsoft Entra ID for authentication")
            credential = DefaultAzureCredential()
            token = credential.get_token("https://cognitiveservices.azure.com/.default").token

            self.chat_model = AzureChatOpenAI(
                azure_ad_token=token,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                azure_deployment=settings.azure_openai_model_chat,
                streaming=True,
            )
            self.reasoning_model = AzureChatOpenAI(
                azure_ad_token=token,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                azure_deployment=settings.azure_openai_model_reasoning,
                streaming=True,
            )
            self.embedding_model = AzureOpenAIEmbeddings(
                azure_ad_token=token,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                azure_deployment=settings.azure_openai_model_embedding,
            )
        else:
            logger.info("Using API key for authentication")
            self.chat_model = AzureChatOpenAI(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                azure_deployment=settings.azure_openai_model_chat,
                streaming=True,
            )
            self.reasoning_model = AzureChatOpenAI(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                azure_deployment=settings.azure_openai_model_reasoning,
                streaming=True,
            )
            self.embedding_model = AzureOpenAIEmbeddings(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                azure_deployment=settings.azure_openai_model_embedding,
            )

    def create_embedding(self, text: str):
        """Create an embedding for the given text."""
        return self.embedding_model.embed_query(text)
