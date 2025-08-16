from functools import lru_cache

from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools.base import BaseTool
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    sql_database_uri: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_sql_database_settings() -> Settings:
    """Get SQL Database settings."""
    return Settings()


class SqlDatabaseClientWrapper:
    def __init__(
        self,
        settings: Settings = None,
    ):
        if settings is None:
            settings = get_sql_database_settings()
        self.settings = settings

    def get_tools(
        self,
        llm: BaseLanguageModel,
    ) -> list[BaseTool]:
        """Get SQL Database tools."""
        if self.settings.sql_database_uri == "":
            return []

        self.db = SQLDatabase.from_uri(
            database_uri=self.settings.sql_database_uri,
        )
        return SQLDatabaseToolkit(
            db=self.db,
            llm=llm,
        ).get_tools()
