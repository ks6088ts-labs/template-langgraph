from functools import lru_cache

from foundry_local import FoundryLocalManager
from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    foundry_local_model_chat: str = "phi-3-mini-4k"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_foundry_local_settings() -> Settings:
    return Settings()


class FoundryLocalWrapper:
    def __init__(self, settings: Settings = None):
        if settings is None:
            settings = get_foundry_local_settings()

        self.foundry_local_manager = FoundryLocalManager(
            alias_or_model_id=settings.foundry_local_model_chat,
        )

        self.chat_model = ChatOpenAI(
            model=self.foundry_local_manager.get_model_info(settings.foundry_local_model_chat).id,
            base_url=self.foundry_local_manager.endpoint,
            api_key=self.foundry_local_manager.api_key,
            temperature=0.0,
            streaming=True,
        )
