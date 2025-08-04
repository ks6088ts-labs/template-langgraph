from functools import lru_cache

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    dify_base_url: str = "https://api.dify.ai/v1"
    dify_api_key: str = "<YOUR_API_KEY>"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_dify_settings() -> Settings:
    """Get Dify settings."""
    return Settings()


class DifyClientWrapper:
    def __init__(
        self,
        settings: Settings = None,
    ):
        if settings is None:
            settings = get_dify_settings()
        self.base_url = settings.dify_base_url
        self.headers = {
            "Authorization": f"Bearer {settings.dify_api_key}",
            "Content-Type": "application/json",
        }

    def run_workflow(
        self,
        inputs: dict,
    ) -> dict:
        """Run a Dify workflow."""
        with httpx.Client() as client:
            response = client.post(
                url=f"{self.base_url}/workflows/run",
                json=inputs,
                headers=self.headers,
                timeout=60 * 5,  # Set a timeout for the request
            )
            response.raise_for_status()
            return response.json()
