from functools import lru_cache

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    azure_ai_foundry_inference_endpoint: str = "https://xxx.services.ai.azure.com/api/projects/xxx"
    azure_ai_foundry_inference_api_version: str = "2025-04-01-preview"
    azure_ai_foundry_inference_model_chat: str = "gpt-5"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_azure_ai_foundry_settings() -> Settings:
    return Settings()


class AzureAiFoundryWrapper:
    def __init__(self, settings: Settings = None):
        if settings is None:
            settings = get_azure_ai_foundry_settings()
        self.settings = settings
        self.project_client = AIProjectClient(
            endpoint=self.settings.azure_ai_foundry_inference_endpoint,
            credential=DefaultAzureCredential(),
        )
        self.openai_client = self.project_client.get_openai_client(
            api_version=self.settings.azure_ai_foundry_inference_api_version
        )

    def get_ai_project_client(self):
        return self.project_client

    def get_openai_client(self):
        return self.openai_client
