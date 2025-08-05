from functools import lru_cache

from langchain_ollama import ChatOllama
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_model_chat: str = "phi3:latest"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_ollama_settings() -> Settings:
    return Settings()


class OllamaWrapper:
    def __init__(self, settings: Settings = None):
        if settings is None:
            settings = get_ollama_settings()

        self.chat_model = ChatOllama(
            model=settings.ollama_model_chat,
            temperature=0.0,
            streaming=True,
        )
