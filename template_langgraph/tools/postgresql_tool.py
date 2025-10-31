from functools import lru_cache

from langchain_core.documents import Document
from langchain_postgres import PGEngine, PGVectorStore
from pydantic_settings import BaseSettings, SettingsConfigDict

from template_langgraph.llms.azure_openais import AzureOpenAiWrapper


class Settings(BaseSettings):
    postgresql_user: str = "user"
    postgresql_password: str = "password"
    postgresql_host: str = "localhost"
    postgresql_port: str = "5432"
    postgresql_database: str = "db"
    postgresql_table_name: str = "reports_kabuto_vectors"
    postgresql_id_column: str = "id"
    postgresql_content_column: str = "user_message"
    postgresql_embedding_column: str = "embedding"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )


@lru_cache
def get_postgresql_settings() -> Settings:
    """Get PostgreSQL Database settings."""
    return Settings()


class PostgreSQLClientWrapper:
    def __init__(
        self,
        settings: Settings = None,
    ):
        if settings is None:
            settings = get_postgresql_settings()

        self.vector_store = PGVectorStore.create_sync(
            engine=PGEngine.from_connection_string(
                url=f"postgresql+psycopg://{settings.postgresql_user}:{settings.postgresql_password}@{settings.postgresql_host}:{settings.postgresql_port}/{settings.postgresql_database}"
            ),
            table_name=settings.postgresql_table_name,
            embedding_service=AzureOpenAiWrapper().embedding_model,
            id_column=settings.postgresql_id_column,
            content_column=settings.postgresql_content_column,
            embedding_column=settings.postgresql_embedding_column,
        )

    def add_documents(
        self,
        documents: list[Document],
    ) -> list[str]:
        """Add documents to a Cosmos DB container."""
        return self.vector_store.add_documents(
            documents=documents,
        )
