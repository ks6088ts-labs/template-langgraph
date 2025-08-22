from functools import lru_cache
import threading
from typing import Optional

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
    # Class-level variables for singleton-like behavior
    _credentials: dict = {}
    _tokens: dict = {}
    _token_lock = threading.Lock()
    
    def __init__(self, settings: Settings = None):
        if settings is None:
            settings = get_azure_openai_settings()
        
        self.settings = settings
        self._chat_model: Optional[AzureChatOpenAI] = None
        self._reasoning_model: Optional[AzureChatOpenAI] = None
        self._embedding_model: Optional[AzureOpenAIEmbeddings] = None

    def _get_auth_key(self) -> str:
        """Generate a key for authentication caching based on settings."""
        return f"{self.settings.azure_openai_endpoint}_{self.settings.azure_openai_use_microsoft_entra_id}"

    def _get_auth_token(self) -> Optional[str]:
        """Get authentication token with lazy initialization and caching."""
        if self.settings.azure_openai_use_microsoft_entra_id.lower() != "true":
            return None
            
        auth_key = self._get_auth_key()
        
        with self._token_lock:
            if auth_key not in self._credentials:
                logger.info("Initializing Microsoft Entra ID authentication")
                self._credentials[auth_key] = DefaultAzureCredential()
            
            if auth_key not in self._tokens:
                logger.info("Getting authentication token")
                self._tokens[auth_key] = self._credentials[auth_key].get_token("https://cognitiveservices.azure.com/.default").token
            
            return self._tokens[auth_key]
    
    @property
    def chat_model(self) -> AzureChatOpenAI:
        """Lazily initialize and return chat model."""
        if self._chat_model is None:
            if self.settings.azure_openai_use_microsoft_entra_id.lower() == "true":
                token = self._get_auth_token()
                self._chat_model = AzureChatOpenAI(
                    azure_ad_token=token,
                    azure_endpoint=self.settings.azure_openai_endpoint,
                    api_version=self.settings.azure_openai_api_version,
                    azure_deployment=self.settings.azure_openai_model_chat,
                    streaming=True,
                )
            else:
                logger.info("Using API key for authentication")
                self._chat_model = AzureChatOpenAI(
                    api_key=self.settings.azure_openai_api_key,
                    azure_endpoint=self.settings.azure_openai_endpoint,
                    api_version=self.settings.azure_openai_api_version,
                    azure_deployment=self.settings.azure_openai_model_chat,
                    streaming=True,
                )
        return self._chat_model
    
    @property
    def reasoning_model(self) -> AzureChatOpenAI:
        """Lazily initialize and return reasoning model."""
        if self._reasoning_model is None:
            if self.settings.azure_openai_use_microsoft_entra_id.lower() == "true":
                token = self._get_auth_token()
                self._reasoning_model = AzureChatOpenAI(
                    azure_ad_token=token,
                    azure_endpoint=self.settings.azure_openai_endpoint,
                    api_version=self.settings.azure_openai_api_version,
                    azure_deployment=self.settings.azure_openai_model_reasoning,
                    streaming=True,
                )
            else:
                self._reasoning_model = AzureChatOpenAI(
                    api_key=self.settings.azure_openai_api_key,
                    azure_endpoint=self.settings.azure_openai_endpoint,
                    api_version=self.settings.azure_openai_api_version,
                    azure_deployment=self.settings.azure_openai_model_reasoning,
                    streaming=True,
                )
        return self._reasoning_model
    
    @property
    def embedding_model(self) -> AzureOpenAIEmbeddings:
        """Lazily initialize and return embedding model."""
        if self._embedding_model is None:
            if self.settings.azure_openai_use_microsoft_entra_id.lower() == "true":
                token = self._get_auth_token()
                self._embedding_model = AzureOpenAIEmbeddings(
                    azure_ad_token=token,
                    azure_endpoint=self.settings.azure_openai_endpoint,
                    api_version=self.settings.azure_openai_api_version,
                    azure_deployment=self.settings.azure_openai_model_embedding,
                )
            else:
                self._embedding_model = AzureOpenAIEmbeddings(
                    api_key=self.settings.azure_openai_api_key,
                    azure_endpoint=self.settings.azure_openai_endpoint,
                    api_version=self.settings.azure_openai_api_version,
                    azure_deployment=self.settings.azure_openai_model_embedding,
                )
        return self._embedding_model

    def create_embedding(self, text: str):
        """Create an embedding for the given text."""
        return self.embedding_model.embed_query(text)
