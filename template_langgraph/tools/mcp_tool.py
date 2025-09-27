import asyncio
import json
from functools import lru_cache

from langchain_core.tools.base import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mcp_config_path: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_mcp_settings() -> Settings:
    """Get mcp settings."""
    return Settings()


class McpClientWrapper:
    def __init__(
        self,
        settings: Settings = None,
    ):
        if settings is None:
            settings = get_mcp_settings()
        self.settings = settings
        self.client = None

    def get_tools(self) -> list[BaseTool]:
        if self.settings.mcp_config_path == "":
            return []
        with open(self.settings.mcp_config_path) as f:
            config = json.load(f)
            for _, value in config["servers"].items():
                value["transport"] = "stdio"
            self.client = MultiServerMCPClient(config["servers"])
            self.tools = asyncio.run(self.client.get_tools())
            return self.tools
